"""Download-readiness guards use only tiny safetensors header fixtures."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "examples/expertplex/check_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_readiness", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CheckpointReadinessTests(unittest.TestCase):
    def fixture(self, root, *, dtype="BF16", payload=b"\0\0\0\0"):
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "part.safetensors"}})
        )
        header = json.dumps(
            {"weight": {"dtype": dtype, "shape": [2], "data_offsets": [0, 4]}}
        ).encode()
        (root / "part.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + payload
        )

    def test_complete_header_and_payload_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            report = MODULE.inspect_checkpoint(root)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["complete_shards"], 1)
            self.assertEqual(report["dtype_tensor_counts"], {"BF16": 1})

    def test_missing_and_truncated_downloads_are_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, payload=b"\0")
            self.assertEqual(MODULE.inspect_checkpoint(root)["status"], "incomplete")
            (root / "part.safetensors").unlink()
            report = MODULE.inspect_checkpoint(root)
            self.assertEqual(report["missing_shards"], ["part.safetensors"])

    def test_wrong_dtype_and_missing_indexed_tensor_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, dtype="F16")
            self.assertEqual(MODULE.inspect_checkpoint(root)["status"], "incomplete")
            self.fixture(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"another": "part.safetensors"}})
            )
            self.assertEqual(MODULE.inspect_checkpoint(root)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
