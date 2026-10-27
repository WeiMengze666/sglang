"""Reject a numerically incomparable reference before starting GPU services."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
validate = importlib.import_module("check_numerical_alignment").validate_reference


class AlignmentConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model"
        self.report = {"variant": "native", "status": "passed", "tbo_enabled": True}
        (self.root / "scores.json").write_text("{}")
        (self.root / "histories.json").write_text(
            json.dumps({"model_path": str(self.model)})
        )

    def check(self, enabled=True):
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, enabled)

    def test_matching_and_legacy_reference(self):
        self.check()
        del self.report["tbo_enabled"]
        self.report["commands"] = [
            ["python", "-m", "sglang.launch_server", "--enable-two-batch-overlap"]
        ]
        self.check()
        self.report["commands"][0].pop()
        self.check(False)

    def test_mismatched_overlap_or_failed_reference(self):
        with self.assertRaisesRegex(ValueError, "TBO mode"):
            self.check(False)
        self.report["tbo_enabled"] = 1
        with self.assertRaises(ValueError):
            self.check()
        self.report["tbo_enabled"] = True
        self.report["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "successful native"):
            self.check()

    def test_mismatched_model_or_unknown_overlap(self):
        self.model = self.root / "different-model"
        with self.assertRaisesRegex(ValueError, "model path"):
            self.check()
        del self.report["tbo_enabled"]
        with self.assertRaisesRegex(ValueError, "overlap configuration"):
            self.check()


if __name__ == "__main__":
    unittest.main()
