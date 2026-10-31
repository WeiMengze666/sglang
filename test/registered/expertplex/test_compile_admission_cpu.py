"""Native argument admission only; actual compiled numerics require GPU tests."""

import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "compile_admission_integration",
    ROOT / "python/sglang/srt/expertplex/integration.py",
)
INTEGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTEGRATION)


class CompileAdmission(unittest.TestCase):
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
            enable_deterministic_inference=True,
            torch_compile_max_bs=16,
            cuda_graph_config={
                "decode": {"backend": "full", "bs": [2, 4, 8, 12, 16]},
                "prefill": {"backend": "disabled"},
            },
            model_path="/model",
            load_format="safetensors",
            dtype="bfloat16",
            device="cuda",
            base_gpu_id=1,
            max_running_requests=16,
            chunked_prefill_size=256,
            max_prefill_tokens=256,
            disable_overlap_schedule=False,
        )
        self.env = {
            "SGLANG_TORCH_COMPILE_MODE": "default",
            "TORCHINDUCTOR_BATCH_INVARIANT": "1",
            "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN",
        }
        self.enterContext(patch.dict(os.environ, self.env, clear=True))
        self.enterContext(
            patch.object(
                INTEGRATION, "attention_config", lambda _: (self.config, self.client)
            )
        )

    def validate(self):
        INTEGRATION.validate_server_args(self.args)

    def test_none_and_tbo_use_native_flags_without_diagnostic_activation(self):
        self.validate()
        self.config.overlap = "tbo"
        self.args.enable_two_batch_overlap = True
        self.validate()
        self.assertFalse(self.args.disable_overlap_schedule)
        self.assertEqual(dict(os.environ), self.env)

    def test_singleton_bucket_is_admitted_only_without_tbo(self):
        self.args.cuda_graph_config["decode"]["bs"].insert(0, 1)
        self.validate()
        self.config.overlap = "tbo"
        self.args.enable_two_batch_overlap = True
        with self.assertRaisesRegex(ValueError, "singleton"):
            self.validate()

    def test_client_policy_controls_native_flags_and_compile_buckets(self):
        # Global TBO reserves two mailboxes but this D uses the native no-TBO
        # path. It may therefore use the existing singleton graph bucket.
        self.config.overlap = "tbo"
        self.client.overlap = "none"
        self.args.cuda_graph_config["decode"]["bs"].insert(0, 1)
        self.validate()
        self.args.enable_two_batch_overlap = True
        with self.assertRaisesRegex(ValueError, "client overlap policy"):
            self.validate()
        self.config.overlap = "none"
        self.client.overlap = "tbo"
        with self.assertRaisesRegex(ValueError, "singleton"):
            self.validate()
        self.args.cuda_graph_config["decode"]["bs"].remove(1)
        self.validate()

    def test_precision_controls_cannot_be_replaced_by_private_switches(self):
        for key in self.env:
            with (
                self.subTest(key=key),
                patch.dict(
                    os.environ,
                    {
                        key: "invalid",
                        "EXPERTPLEX_DIAG_COMPILE": "1",
                        "EXPERTPLEX_DIAG_COMPILED_TBO": "1",
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, key):
                    self.validate()

    def test_unsupported_compiler_combinations_are_rejected(self):
        for field, value in (
            ("disaggregation_mode", "prefill"),
            ("enable_deterministic_inference", False),
            ("torch_compile_max_bs", 32),
        ):
            with self.subTest(field=field), patch.object(self.args, field, value):
                with self.assertRaisesRegex(
                    ValueError, "requires deterministic decode"
                ):
                    self.validate()
        for width, overlap in ((2, "none"), (1, "sbo")):
            self.client.tp_size, self.config.overlap = width, overlap
            with self.assertRaisesRegex(ValueError, "requires TP1"):
                self.validate()

    def test_dict_and_resolved_graph_config_have_same_validation(self):
        self.args.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(backend="full", bs=[2, 4, 8, 12, 16]),
            prefill=types.SimpleNamespace(backend="disabled"),
        )
        self.validate()
        self.args.cuda_graph_config.decode.bs = [2, 4, 8, 12, 16, 32]
        with self.assertRaisesRegex(ValueError, "full CUDA graph buckets"):
            self.validate()

    def test_uncompiled_and_non_expertplex_paths_do_not_require_compiler_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.args.enable_torch_compile = False
            self.validate()
            self.args.enable_torch_compile = True
            self.args.enable_expertplex = False
            self.validate()


if __name__ == "__main__":
    unittest.main()
