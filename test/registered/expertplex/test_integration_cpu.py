"""CPU-only adapter tests; run with Python's stdlib unittest, without SGLang deps.

AST extraction executes the actual adapter/loader methods with small test
doubles. It verifies integration control flow, not CUDA numerics or speed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = load_module("sglang.srt.expertplex.config", SRT / "expertplex/config.py")
INTEGRATION = load_module(
    "sglang.srt.expertplex.integration", SRT / "expertplex/integration.py"
)


def extracted_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def extracted_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class SchedulerTboTests(unittest.TestCase):
    def test_equal_token_split_fits_compact_mailboxes_at_boundaries(self):
        path = SRT / "batch_overlap/two_batch_overlap.py"
        names = {
            "_is_two_chunk_split_enabled",
            "_split_extend_seqs",
            "_split_array_by_cum_less_than_half",
            "_split_array_by_balanced_sum",
        }
        tree = ast.parse(path.read_text())
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        code = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        namespace = {"get_tbo_token_distribution_threshold": lambda: 0.5}
        exec(compile(code, str(path), "exec"), namespace)
        rng = random.Random(20261015)
        cases = [[1], [2], [24575], [24576], [1] * 256, [1] * 255 + [24321]]
        for _ in range(1000):
            total = rng.randint(1, 24576)
            count = rng.randint(1, min(total, 256))
            cuts = [0, *sorted(rng.sample(range(1, total), count - 1)), total]
            cases.append([b - a for a, b in zip(cuts, cuts[1:])])
        for lengths in cases:
            total = sum(lengths)
            left = (
                total // 2
                if namespace["_is_two_chunk_split_enabled"](lengths)
                else sum(lengths[: namespace["_split_extend_seqs"](lengths)])
            )
            self.assertLessEqual(max(left, total - left), 12288, lengths)

    def test_tp1_tbo_prepares_local_metadata(self):
        path = SRT / "utils/common.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "require_mlp_sync"
        )
        state = types.SimpleNamespace(attn_dp_enabled=False)
        overlap = types.SimpleNamespace(enable_two_batch_overlap=False)
        namespace = {
            "get_parallel": lambda: state,
            "require_gathered_buffer": lambda: False,
            "get_exec": lambda: types.SimpleNamespace(overlap=overlap),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
        required = namespace["require_mlp_sync"]
        self.assertFalse(required())
        overlap.enable_two_batch_overlap = True
        self.assertTrue(required())
        overlap.enable_two_batch_overlap = False
        state.attn_dp_enabled = True
        self.assertTrue(required())


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_path = Path(self.temp.name) / "model"
        self.model_path.mkdir()
        self.hf_config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "dtype": "bfloat16",
            "hidden_size": 4,
            "moe_intermediate_size": 8,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "vocab_size": 16,
        }
        (self.model_path / "config.json").write_text(json.dumps(self.hf_config))
        self.config = CONFIG.default_config(str(self.model_path))
        self.config_path = Path(self.temp.name) / "node.json"
        self.config.write_json(self.config_path)
        self.env = patch.dict(
            os.environ,
            {
                INTEGRATION.CONFIG_ENV: str(self.config_path),
                INTEGRATION.CLIENT_ENV: "1",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(INTEGRATION._load_attention_config.cache_clear)

    def server_args(self, **overrides):
        values = dict(
            model_path=str(self.model_path),
            load_format="safetensors",
            dtype="bfloat16",
            device="cuda",
            disable_overlap_schedule=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            base_gpu_id=1,
            max_running_requests=256,
            chunked_prefill_size=256,
            max_prefill_tokens=256,
            cuda_graph_config={
                "prefill": {"backend": "disabled"},
                "decode": {"backend": "disabled"},
            },
        )
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_supported_attention_arguments_and_geometry(self):
        INTEGRATION.validate_server_args(self.server_args())
        hf = types.SimpleNamespace(to_dict=lambda: self.hf_config)
        INTEGRATION.validate_attention_model(hf)
        self.assertEqual(INTEGRATION.attention_config()[1].phase, "decode")

    def test_unchunked_prefill_has_a_hard_input_bound(self):
        node, _ = INTEGRATION.attention_config()
        self.assertEqual(
            INTEGRATION.bound_request_input_length(100000), node.max_prefill_tokens + 1
        )
        self.assertEqual(INTEGRATION.bound_request_input_length(128), 128)
        with patch.dict(os.environ, {INTEGRATION.CLIENT_ENV: "0"}):
            for disabled in (-1, None):
                INTEGRATION.validate_server_args(
                    self.server_args(
                        disaggregation_mode="prefill",
                        base_gpu_id=0,
                        chunked_prefill_size=disabled,
                    )
                )

    def test_compact_mailboxes_require_equal_token_tbo_split(self):
        node = CONFIG.NodeConfig.from_dict(
            self.config.to_dict() | {"overlap": "tbo", "prefill_mailbox_tokens": 1024}
        )
        node.write_json(self.config_path)
        INTEGRATION._load_attention_config.cache_clear()
        args = self.server_args(enable_two_batch_overlap=True)
        with self.assertRaisesRegex(ValueError, "compact P mailboxes"):
            INTEGRATION.validate_server_args(args)
        args.tbo_token_distribution_threshold = 0.5
        INTEGRATION.validate_server_args(args)
        args.max_running_requests = None
        with self.assertRaisesRegex(ValueError, "compact P mailboxes"):
            INTEGRATION.validate_server_args(args)

    def test_unsupported_modes_are_reported_together(self):
        args = self.server_args(
            tp_size=2,
            ep_size=2,
            enable_two_batch_overlap=True,
            enable_eplb=True,
            speculative_algorithm="EAGLE",
            quantization="fp8",
            disable_overlap_schedule=False,
            cuda_graph_config={"decode": {"backend": "invalid"}},
        )
        with self.assertRaises(ValueError) as caught:
            INTEGRATION.validate_server_args(args)
        for fragment in (
            "--tp-size",
            "--ep-size",
            "--enable-two-batch-overlap",
            "--enable-eplb",
            "speculative",
            "quantization",
            "--disable-overlap-schedule",
            "--cuda-graph-backend-decode",
        ):
            self.assertIn(fragment, str(caught.exception))

    def test_phase_and_checkpoint_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "client phase"):
            INTEGRATION.validate_server_args(
                self.server_args(disaggregation_mode="prefill")
            )
        changed = dict(self.hf_config, hidden_size=16)
        with self.assertRaisesRegex(ValueError, "checkpoint dimensions"):
            INTEGRATION.validate_attention_model(
                types.SimpleNamespace(to_dict=lambda: changed)
            )

    def test_half_configured_environment_fails(self):
        with patch.dict(os.environ):
            del os.environ[INTEGRATION.CLIENT_ENV]
            self.assertTrue(INTEGRATION.expertplex_enabled())
            with self.assertRaisesRegex(ValueError, "requires both"):
                INTEGRATION.attention_config()

    def test_scheduler_caps_and_full_visibility_are_required(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaisesRegex(ValueError, "all eight GPUs"):
                INTEGRATION.validate_server_args(self.server_args())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}):
            INTEGRATION.validate_server_args(self.server_args())
        for field in (
            "max_running_requests",
            "chunked_prefill_size",
            "max_prefill_tokens",
        ):
            for invalid in (None, -1, 257):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaisesRegex(ValueError, field.replace("_", "-")):
                        INTEGRATION.validate_server_args(
                            self.server_args(**{field: invalid})
                        )

    def test_unconfigured_opt_in_does_not_modify_ordinary_servers(self):
        with patch.dict(os.environ):
            del os.environ[INTEGRATION.CONFIG_ENV]
            del os.environ[INTEGRATION.CLIENT_ENV]
            self.assertFalse(INTEGRATION.expertplex_enabled())
            INTEGRATION.validate_server_args(types.SimpleNamespace())
            INTEGRATION.validate_attention_model(None)

    def test_experts_are_filtered_before_tensor_materialization(self):
        reads = []
        names = [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]

        class SafeFile:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return names

            def get_tensor(self, name):
                if INTEGRATION.is_routed_expert_weight(name):
                    raise AssertionError("a routed expert was materialized")
                reads.append(name)
                return types.SimpleNamespace(name=name, dtype="torch.bfloat16")

        safetensors = types.SimpleNamespace(
            safe_open=lambda *args, **kwargs: SafeFile()
        )
        with patch.dict(sys.modules, {"safetensors": safetensors}):
            result = list(INTEGRATION.attention_weights_iterator(["shard.safetensors"]))
        self.assertEqual(reads, names[2:])
        self.assertEqual([name for name, _ in result], names[2:])

    def test_attention_weight_dtype_is_checked_before_loading(self):
        class SafeFile:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return ["model.layers.0.mlp.gate.weight"]

            def get_tensor(self, name):
                return types.SimpleNamespace(dtype="torch.float8_e4m3fn")

        safetensors = types.SimpleNamespace(
            safe_open=lambda *args, **kwargs: SafeFile()
        )
        with patch.dict(sys.modules, {"safetensors": safetensors}):
            with self.assertRaisesRegex(ValueError, "BF16 attention weights"):
                list(INTEGRATION.attention_weights_iterator(["shard.safetensors"]))

    def test_default_loader_chooses_selective_iterator_before_bulk_loaders(self):
        seen = []

        def selective(files):
            seen.extend(files)
            return iter([("model.layers.0.mlp.gate.weight", "router")])

        namespace = {
            "LoadFormat": types.SimpleNamespace(AUTO="auto", SAFETENSORS="safetensors"),
            "time": time,
        }
        method = extracted_method(
            SRT / "model_loader/loader.py",
            "DefaultModelLoader",
            "_get_weights_iterator",
            namespace,
        )
        loader = types.SimpleNamespace(
            load_config=types.SimpleNamespace(
                model_loader_extra_config={},
                load_format="safetensors",
                draft_model_idx=None,
            ),
            counter_before_loading_weights=1.0,
        )
        source = types.SimpleNamespace(prefix="")
        resolved = types.SimpleNamespace(
            hf_folder=self.temp.name,
            weight_files=["shard.safetensors"],
            use_safetensors=True,
        )
        with patch.object(INTEGRATION, "attention_weights_iterator", selective):
            result = list(method(loader, source, resolved_source=resolved))
        self.assertEqual(seen, ["shard.safetensors"])
        self.assertEqual(result, [("model.layers.0.mlp.gate.weight", "router")])
        resolved.use_safetensors = False
        with self.assertRaisesRegex(ValueError, "requires a safetensors"):
            method(loader, source, resolved_source=resolved)

    def test_sparse_block_keeps_router_without_constructing_local_experts(self):
        calls = []

        class Tensor:
            def __init__(self, rows=2):
                self.shape = (rows, 4)

            def reshape(self, *shape):
                return self

            def view(self, *shape):
                return self

        class Linear:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, hidden):
                calls.append("gate")
                return "router logits", None

        class TopK:
            def __init__(self, *args, **kwargs):
                self.output_format = kwargs["output_format"]

            def __call__(self, hidden, logits):
                return types.SimpleNamespace(topk_ids="ids", topk_weights="weights")

        def no_local_experts(*args):
            raise AssertionError("attention rank constructed local expert weights")

        output = Tensor()

        def remote_run(*args):
            calls.append(args)
            return output

        namespace = {
            "nn": types.SimpleNamespace(Module=object),
            "get_parallel": lambda: types.SimpleNamespace(moe_tp_size=1, moe_ep_size=1),
            "expertplex_enabled": lambda: True,
            "TopK": TopK,
            "TopKOutputFormat": types.SimpleNamespace(STANDARD="standard"),
            "TopKOutputChecker": types.SimpleNamespace(
                format_is_standard=lambda output: True
            ),
            "ReplicatedLinear": Linear,
            "get_moe_impl_class": no_local_experts,
            "add_prefix": lambda name, prefix: f"{prefix}.{name}",
            "is_deepep_class_backend": lambda: False,
            "get_moe_a2a_backend": lambda: types.SimpleNamespace(
                is_megamoe=lambda: False
            ),
        }
        block_cls = extracted_class(
            SRT / "models/qwen3_moe.py", "Qwen3MoeSparseMoeBlock", namespace
        )
        gguf = types.SimpleNamespace(GGUFConfig=type("GGUFConfig", (), {}))
        client = types.SimpleNamespace(
            get_client=lambda: types.SimpleNamespace(run=remote_run)
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.quantization.gguf": gguf,
                "sglang.srt.expertplex.client": client,
            },
        ):
            config = types.SimpleNamespace(
                num_experts=8,
                num_experts_per_tok=2,
                hidden_size=4,
                moe_intermediate_size=8,
            )
            block = block_cls(3, config)
            self.assertIsNone(block.experts)
            self.assertIsInstance(block.gate, Linear)
            self.assertEqual(block.get_moe_weights(), [])
            self.assertEqual(block.topk.output_format, "standard")
            hidden = Tensor()
            result = block.forward(hidden, types.SimpleNamespace(forward_mode="extend"))
            self.assertIs(result, output)
            self.assertEqual(calls, ["gate", (3, hidden, "ids", "weights")])
            empty = Tensor(0)
            self.assertIs(block.forward(empty), empty)
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, "unquantized BF16"):
                block_cls(3, config, quant_config=object())

            # The unconfigured model still constructs its usual local expert
            # implementation and lets SGLang choose the TopK format.
            local_experts = object()
            namespace["expertplex_enabled"] = lambda: False
            namespace["get_moe_impl_class"] = lambda quant: (
                lambda **kwargs: local_experts
            )
            namespace["get_exec"] = lambda: types.SimpleNamespace(
                moe=types.SimpleNamespace(ep_num_redundant_experts=0)
            )
            namespace["RoutingMethodType"] = types.SimpleNamespace(
                Renormalize="renormalize"
            )
            ordinary = block_cls(3, config)
            self.assertIs(ordinary.experts, local_experts)
            self.assertIsNone(ordinary.topk.output_format)


if __name__ == "__main__":
    unittest.main()
