"""Native compile admission must preserve ordinary serving and tuning choices."""

import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "compile_admission_integration",
    ROOT / "python/sglang/srt/expertplex/integration.py",
)
INTEGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTEGRATION)


class CompileAdmission(CustomTestCase):
    def setUp(self):
        self.client = types.SimpleNamespace(tp_size=1, phase="decode", device=1)
        self.config = types.SimpleNamespace(
            overlap="none",
            model_path="/model",
            weight_mode="checkpoint",
            node_gpu_count=8,
            max_prefill_tokens=24576,
            max_decode_tokens=256,
            prefill_mailbox_tokens=None,
        )
        self.args = types.SimpleNamespace(
            enable_expertplex=True,
            enable_torch_compile=True,
            expertplex_role="attention",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            enable_deterministic_inference=False,
            torch_compile_max_bs=32,
            cuda_graph_config={
                "decode": {"backend": "full", "bs": [1, 2, 4, 8, 12, 16, 32]},
                "prefill": {"backend": "disabled"},
            },
            model_path="/model",
            load_format="safetensors",
            dtype="bfloat16",
            device="cuda",
            base_gpu_id=1,
            max_running_requests=32,
            chunked_prefill_size=256,
            max_prefill_tokens=256,
            disable_overlap_schedule=False,
        )
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(
            patch.object(
                INTEGRATION, "attention_config", lambda _: (self.config, self.client)
            )
        )

    def validate(self):
        INTEGRATION.validate_server_args(self.args)

    def test_ordinary_compile_does_not_require_precision_diagnostics(self):
        """Ordinary native compile was rejected unless strict diagnostics were set."""
        self.validate()
        self.config.overlap = "tbo"
        self.args.enable_two_batch_overlap = True
        self.validate()
        self.assertFalse(self.args.enable_deterministic_inference)
        self.assertFalse(self.args.disable_overlap_schedule)
        self.assertEqual(dict(os.environ), {})

    def test_native_tuning_and_explicit_precision_controls_are_preserved(self):
        choices = (
            {
                "SGLANG_TORCH_COMPILE_MODE": "max-autotune-no-cudagraphs",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
            },
            {
                "SGLANG_TORCH_COMPILE_MODE": "default",
                "TORCHINDUCTOR_BATCH_INVARIANT": "1",
                "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN",
            },
        )
        for deterministic in (False, True):
            for env in choices:
                with self.subTest(deterministic=deterministic, env=env):
                    with patch.dict(os.environ, env, clear=True):
                        self.args.enable_deterministic_inference = deterministic
                        self.validate()
                        self.assertEqual(dict(os.environ), env)
                        self.assertEqual(
                            self.args.enable_deterministic_inference, deterministic
                        )

    def test_native_graph_buckets_and_compile_cutoff_are_independent(self):
        self.args.cuda_graph_config["decode"]["bs"] = [1, 4, 16, 32, 64]
        for cutoff in (1, 8, 32):
            with self.subTest(cutoff=cutoff):
                self.args.torch_compile_max_bs = cutoff
                self.validate()
        for cutoff in (0, -1, True, 1.5):
            with self.subTest(cutoff=cutoff):
                self.args.torch_compile_max_bs = cutoff
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.validate()

    def test_client_policy_controls_overlap_with_native_bucket_alignment(self):
        self.config.overlap = "tbo"
        self.client.overlap = "none"
        self.validate()
        self.args.enable_two_batch_overlap = True
        with self.assertRaisesRegex(ValueError, "client overlap policy"):
            self.validate()
        self.config.overlap = "none"
        self.client.overlap = "tbo"
        self.validate()

    def test_full_graph_backend_is_required_for_native_decode_compilation(self):
        for backend in ("disabled", "breakable"):
            with self.subTest(backend=backend):
                self.args.cuda_graph_config["decode"]["backend"] = backend
                with self.assertRaisesRegex(ValueError, "full CUDA graphs"):
                    self.validate()

    def test_current_phase_and_topology_boundaries_are_preserved(self):
        self.args.disaggregation_mode = "prefill"
        with self.assertRaisesRegex(ValueError, "requires decode attention"):
            self.validate()
        self.args.disaggregation_mode = "decode"
        for width, overlap in ((2, "none"), (1, "sbo")):
            with self.subTest(width=width, overlap=overlap):
                self.client.tp_size = width
                self.config.overlap = overlap
                with self.assertRaisesRegex(ValueError, "local attention TP1"):
                    self.validate()

    def test_dict_and_resolved_graph_config_admit_the_same_native_settings(self):
        self.validate()
        self.args.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(backend="full", bs=[1, 4, 32]),
            prefill=types.SimpleNamespace(backend="disabled"),
        )
        self.validate()

    def test_uncompiled_and_non_expertplex_paths_leave_compile_controls_alone(self):
        self.args.enable_torch_compile = False
        self.args.cuda_graph_config["decode"]["backend"] = "disabled"
        self.args.torch_compile_max_bs = 0
        self.validate()
        self.args.enable_torch_compile = True
        self.args.enable_expertplex = False
        self.validate()

    def test_native_attention_dp_uses_local_tp1_compiler(self):
        self.client.tp_size = 3
        self.client.attention_dp_size = 3
        self.args.tp_size = 3
        self.args.attn_dp_size = 3
        self.args.enable_dp_lm_head = True
        self.args.max_running_requests = 48
        self.args.chunked_prefill_size = 16
        self.args.max_prefill_tokens = 16
        self.config.max_decode_tokens = 48
        self.args.torch_compile_max_bs = 16
        self.args.cuda_graph_config["decode"]["bs"] = [1, 2, 4, 8, 12, 16]
        self.validate()
        self.assertFalse(self.args.enable_deterministic_inference)
        self.assertEqual(self.args.torch_compile_max_bs, 16)
        # Preserve native CPU scheduler choices already supported by serving.
        for disabled in (True, False):
            with self.subTest(disable_overlap_schedule=disabled):
                self.args.disable_overlap_schedule = disabled
                self.validate()
                self.assertEqual(self.args.disable_overlap_schedule, disabled)

        for name, value, message in (
            ("enable_dp_lm_head", False, "enable-dp-lm-head"),
            ("attn_dp_size", 1, "attn-dp-size"),
            ("max_running_requests", 47, "divide by attention DP"),
        ):
            with self.subTest(name=name), patch.object(self.args, name, value):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_grouped_launcher_forwards_compile_only_to_decode(self):
        import dataclasses
        import sys
        from sglang.srt.expertplex.config import default_config

        sys.path.insert(0, str(ROOT / "examples/expertplex"))
        self.addCleanup(sys.path.pop, 0)
        import launch

        node = default_config("/model", max_decode_tokens=48, overlap="none")
        node = dataclasses.replace(
            node,
            clients=(
                node.clients[0],
                dataclasses.replace(node.clients[1], tp_size=3, attention_dp_size=3),
            ),
        )
        args = types.SimpleNamespace(
            config="/node.json",
            port=16000,
            bootstrap_port=18000,
            nccl_port=17000,
            cuda_graph=True,
            chunked_prefill=False,
            kv_tokens=None,
            attention_backend="fa4",
            sampling_backend="flashinfer",
            ib_device=None,
            skip_tokenizer_init=False,
            tokenizer_path=None,
            prefill_graph_backend="disabled",
            prefill_graph_max_tokens=None,
            host="127.0.0.1",
            enable_torch_compile=True,
            torch_compile_max_bs=16,
            cuda_graph_bs_decode=[1, 2, 4, 8, 12, 16],
            decode_max_running_requests=48,
            decode_load_balance_method="round_robin",
            decode_boundary_reduction="rs+rsv",
        )
        worker, servers, router = launch.commands(node, args)
        self.assertNotIn("--enable-torch-compile", worker)
        self.assertNotIn("--enable-torch-compile", router)
        for client, command in servers:
            if client.phase == "prefill":
                self.assertNotIn("--enable-torch-compile", command)
                continue
            self.assertIn("--enable-torch-compile", command)
            self.assertIn("--enable-dp-lm-head", command)
            self.assertEqual(
                command[command.index("--load-balance-method") + 1], "round_robin"
            )
            self.assertEqual(
                command[command.index("--boundary-reduction") + 1], "rs+rsv"
            )
            self.assertNotIn("--enable-deterministic-inference", command)
            self.assertEqual(command[command.index("--torch-compile-max-bs") + 1], "16")
            self.assertEqual(command[command.index("--attn-dp-size") + 1], "3")
            self.assertEqual(command[command.index("--max-prefill-tokens") + 1], "16")
            self.assertEqual(command[command.index("--max-running-requests") + 1], "48")
        # The decode-only switch must still leave the P scheduler unchanged.
        args.decode_disable_overlap_schedule = True
        _, cpu_plain_servers, _ = launch.commands(node, args)
        for client, command in cpu_plain_servers:
            self.assertEqual(
                "--disable-overlap-schedule" in command, client.phase == "decode"
            )
        self.assertIn(
            "--decode-disable-overlap-schedule", launch.attention_capacity_options(args)
        )
        args.decode_disable_overlap_schedule = False
        args.enable_torch_compile = False
        _, plain_servers, _ = launch.commands(node, args)
        for (_, compiled), (_, plain) in zip(servers, plain_servers):
            normalized = list(compiled)
            if "--enable-torch-compile" in normalized:
                start = normalized.index("--enable-torch-compile")
                del normalized[start : start + 3]
            self.assertEqual(normalized, plain)


if __name__ == "__main__":
    unittest.main()
