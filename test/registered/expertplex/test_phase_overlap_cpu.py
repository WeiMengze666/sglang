"""Independent native P/D policies and constant-slot ablation contracts."""

import importlib
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
launch = importlib.import_module("launch")
config = launch.config


class PhaseOverlap(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(
            config="/node.json",
            bootstrap_port=19000,
            nccl_port=20000,
            cuda_graph=True,
            chunked_prefill=False,
            kv_tokens=65536,
            attention_backend="triton",
            sampling_backend="pytorch",
            ib_device=None,
            skip_tokenizer_init=True,
            tokenizer_path=None,
            sequence_cap=2048,
            host="127.0.0.1",
            port=31000,
        )
        return SimpleNamespace(**(values | kwargs))

    def test_old_json_and_uniform_policies_keep_slot_contract(self):
        for mode in ("none", "tbo", "sbo"):
            node = config.default_config("/model", overlap=mode)
            data = node.to_dict()
            for c in data["clients"]:
                self.assertNotIn("overlap", c)
            restored = config.NodeConfig.from_dict(data)
            self.assertEqual(restored, node)
            self.assertEqual([node.client_overlap(c) for c in node.clients], [mode] * 4)
            self.assertEqual(node.slot_count, 8 if mode == "tbo" else 4)

    def test_factorial_policies_preserve_exact_buffer_layout_and_slots(self):
        layouts = []
        for p, d in itertools.product(("none", "tbo"), repeat=2):
            node = config.default_config(
                "/model",
                overlap="tbo",
                prefill_overlap=p,
                decode_overlap=d,
                prefill_preprocess=True,
            )
            _, servers, _ = launch.commands(node, self.args())
            self.assertEqual(node.slot_count, 8)
            self.assertEqual(node.decode_slots, [2, 3, 4, 5, 6, 7])
            layouts.append(node.buffer_layout())
            for client, cmd in servers:
                mode = p if client.phase == "prefill" else d
                self.assertEqual("--enable-two-batch-overlap" in cmd, mode == "tbo")
                self.assertNotIn("--disable-overlap-schedule", cmd)
                self.assertNotIn("--disable-flashinfer-autotune", cmd)
            self.assertEqual(config.NodeConfig.from_dict(node.to_dict()), node)
        self.assertTrue(all(v == layouts[0] for v in layouts))

    def test_tbo_override_allocates_two_slots_with_global_none(self):
        node = config.default_config("/model", overlap="none", decode_overlap="tbo")
        self.assertEqual(node.mailboxes_per_client, 2)
        self.assertEqual(node.client_overlap(node.clients[0]), "none")

    def test_compact_prefill_requires_actual_prefill_tbo(self):
        node = config.default_config(
            "/model", overlap="none", prefill_overlap="tbo", prefill_mailbox_tokens=1024
        )
        self.assertEqual(node.prefill_mailbox_capacity, 1024)
        with self.assertRaisesRegex(ValueError, "requires TBO"):
            config.default_config(
                "/model",
                overlap="tbo",
                prefill_overlap="none",
                prefill_mailbox_tokens=1024,
            )

    def test_prefill_graph_gate_does_not_disable_decode_tbo(self):
        node = config.default_config("/model", overlap="tbo", prefill_overlap="none")
        _, servers, _ = launch.commands(
            node,
            self.args(prefill_graph_backend="breakable", prefill_graph_max_tokens=2048),
        )
        self.assertTrue(
            all(
                "--enable-two-batch-overlap" in cmd
                for client, cmd in servers
                if client.phase == "decode"
            )
        )
        with self.assertRaisesRegex(ValueError, "Prefill graph"):
            launch.commands(
                config.default_config("/model", prefill_overlap="tbo"),
                self.args(
                    prefill_graph_backend="breakable", prefill_graph_max_tokens=2048
                ),
            )

    def test_shared_model_guards_resolve_actual_client_policies(self):
        from dataclasses import replace

        model = config.ModelSpec(
            hidden_size=2048,
            intermediate_size=1024,
            num_experts=16,
            top_k=4,
            num_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=1024,
        )
        with self.assertRaisesRegex(ValueError, "shared expert"):
            config.default_config("/model", decode_overlap="sbo").validate(model)
        shared = replace(
            model,
            architecture="Qwen2MoeForCausalLM",
            shared_intermediate_size=1024,
            use_qk_norm=False,
        )
        node = config.default_config(
            "/model", prefill_overlap="none", decode_overlap="sbo"
        )
        node.validate(shared)
        _, servers, _ = launch.commands(node, self.args())
        for client, command in servers:
            self.assertEqual(
                "--enable-single-batch-overlap" in command, client.phase == "decode"
            )
        with self.assertRaisesRegex(ValueError, "stage hooks"):
            config.default_config("/model", decode_overlap="tbo").validate(shared)

    def test_optimizer_cannot_silently_model_mixed_policy_as_uniform(self):
        from _bootstrap import module

        optimizer = module("optimizer")
        node = config.default_config("/model", overlap="tbo", decode_overlap="none")
        with self.assertRaisesRegex(optimizer.ProfileCoverageError, "Client-specific"):
            optimizer.search(
                node,
                None,
                None,
                input_length=128,
                average_output_length=128,
                context_length=256,
                prefill_slo_s=1,
                decode_slo_s=0.05,
                kv_tokens_per_attention_gpu=65536,
            )

    def test_cli_installs_policy_before_compact_capacity_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "node.json"
            checkpoint = Path(temp) / "model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="qwen3_moe",
                        architectures=["Qwen3MoeForCausalLM"],
                        torch_dtype="bfloat16",
                        hidden_size=2048,
                        moe_intermediate_size=1024,
                        num_experts=16,
                        num_experts_per_tok=4,
                        num_hidden_layers=2,
                        num_attention_heads=16,
                        num_key_value_heads=4,
                        vocab_size=1024,
                        head_dim=128,
                    )
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples/expertplex/configure.py"),
                    "--model-path",
                    str(checkpoint),
                    "--output",
                    str(dest),
                    "--rendezvous-dir",
                    temp,
                    "--overlap",
                    "none",
                    "--prefill-overlap",
                    "tbo",
                    "--decode-overlap",
                    "none",
                    "--prefill-mailbox-tokens",
                    "1024",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            node = config.NodeConfig.from_dict(json.loads(dest.read_text()))
            self.assertEqual(
                [node.client_overlap(c) for c in node.clients],
                ["tbo", "none", "none", "none"],
            )
            self.assertEqual(node.slot_count, 8)
        for invalid in ("", "TBO", False, 2):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config.default_config("/model", decode_overlap=invalid)


if __name__ == "__main__":
    unittest.main()
