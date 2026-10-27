"""Reject a numerically incomparable reference before starting GPU services."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
alignment = importlib.import_module("check_numerical_alignment")
validate = alignment.validate_reference


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

    def test_score_step_count_must_match_before_launch(self):
        self.check()
        (self.root / "scores.json").write_text(json.dumps({"cases": [{"steps": 32}]}))
        validate(self.root, self.model, True, 32)
        with self.assertRaisesRegex(ValueError, "step count"):
            validate(self.root, self.model, True, 8)

    def test_trajectory_capacity_and_continuation(self):
        histories = {
            "outputs": [
                {
                    "prompt": [1] * 129,
                    "meta_info": {"output_token_logprobs": [[0.0, 2]] * 8},
                }
            ]
        }
        alignment.validate_workload(histories, 8, 256, 1024)
        for steps, capacity, context in [(8, 128, 1024), (9, 256, 1024), (8, 256, 137)]:
            with (
                self.subTest(steps=steps, capacity=capacity, context=context),
                self.assertRaises(ValueError),
            ):
                alignment.validate_workload(histories, steps, capacity, context)
        with self.assertRaises(ValueError):
            alignment.validate_workload({"outputs": []}, 8, 256, 1024)

    def test_boundary_prompts_have_exact_lengths(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.assert_no_special = not add_special_tokens
                return [10, 20, 30]

        tokenizer = Tokenizer()
        lengths = [1, 127, 128, 129, 511, 512, 513, 2047]
        prompts = alignment.boundary_prompts(tokenizer, lengths)
        self.assertEqual([len(p) for p in prompts], lengths)
        self.assertEqual(prompts[2], prompts[3][:128])
        self.assertTrue(tokenizer.assert_no_special)


if __name__ == "__main__":
    unittest.main()
