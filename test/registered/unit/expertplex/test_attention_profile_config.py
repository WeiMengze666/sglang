"""CPU regressions for the isolated attention profiler's deployment boundary."""

import builtins
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/expertplex"
with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "_attention_profile_config_test", EXAMPLES / "profile_attention.py"
    )
    profiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler)
config = profiler.module("config")


class BeforeGpuImport(Exception):
    pass


class TestAttentionProfileConfig(CustomTestCase):
    def run_preparation(self, node, output, extra=()):
        source = output.parent / "source.json"
        source.write_text(json.dumps(node.to_dict()))
        original_bytes = source.read_bytes()
        argv = [
            "profile_attention.py",
            "--config",
            str(source),
            "--tokenizer-path",
            "/model",
            "--output-dir",
            str(output),
            *extra,
        ]
        original_import = builtins.__import__

        def stop_before_gpu(name, *args, **kwargs):
            # Only replace the external numeric-runtime boundary. The actual
            # CLI, config loader and emitted profiling config all execute.
            if name == "numpy":
                raise BeforeGpuImport
            return original_import(name, *args, **kwargs)

        try:
            with (
                patch.object(sys, "argv", argv),
                patch("builtins.__import__", side_effect=stop_before_gpu),
                redirect_stderr(io.StringIO()),
            ):
                profiler.main()
        finally:
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_compact_tbo_source_becomes_valid_sequential_profile(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as tmp:
                node = config.default_config(
                    "/model",
                    overlap="none" if explicit else "tbo",
                    prefill_overlap="tbo" if explicit else None,
                    decode_overlap="tbo" if explicit else None,
                    max_prefill_tokens=128,
                    max_decode_tokens=16,
                    prefill_mailbox_tokens=64,
                )
                output = Path(tmp) / "profile"
                with self.assertRaises(BeforeGpuImport):
                    self.run_preparation(node, output)
                isolated = config.load_config(output / "node.json")
                self.assertTrue(
                    all(isolated.client_overlap(c) == "none" for c in isolated.clients)
                )
                self.assertIsNone(isolated.prefill_mailbox_tokens)
                self.assertEqual(isolated.max_prefill_tokens, 128)
                self.assertEqual(isolated.max_decode_tokens, 16)

    def test_fa4_backend_reaches_numeric_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BeforeGpuImport):
                self.run_preparation(
                    config.default_config("/model"),
                    Path(tmp) / "profile",
                    ("--attention-backend", "fa4"),
                )

    def test_single_gpu_profiler_rejects_tp_and_dp_groups(self):
        for dp in (False, True):
            with self.subTest(dp=dp), tempfile.TemporaryDirectory() as tmp:
                data = config.default_config(
                    "/model",
                    decode_count=1,
                    decode_tp_size=3 if dp else 2,
                    max_decode_tokens=48,
                ).to_dict()
                if dp:
                    data["clients"][1]["attention_dp_size"] = 3
                node = config.NodeConfig.from_dict(data)
                output = Path(tmp) / "profile"
                with self.assertRaises(SystemExit) as error:
                    self.run_preparation(node, output)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
