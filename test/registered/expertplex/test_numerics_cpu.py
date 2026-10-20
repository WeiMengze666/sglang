"""Saved-distribution comparisons must preserve missing probability mass."""

import importlib.util
import math
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[3] / "examples/expertplex/check_teacher_forcing.py"
)
SPEC = importlib.util.spec_from_file_location("expertplex_probabilities", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(probabilities, history=(3,)):
    return {
        "model_path": "/model",
        "vocab_size": len(probabilities),
        "outputs": [
            {
                "history": list(history),
                "logprobs": [
                    math.log(value) if value is not None else None
                    for value in probabilities
                ],
            }
        ],
    }


class SavedProbabilityTests(unittest.TestCase):
    def test_subset_mass_does_not_turn_into_full_vocabulary_kl(self):
        row = MODULE.compare_distributions(
            report([0.6, None, 0.1]), report([0.5, 0.3, None])
        )[0]
        self.assertEqual(row["common_scored_tokens"], 1)
        self.assertIsNone(row["kl_reference_to_candidate"])
        self.assertAlmostEqual(row["reference_mass_on_common_tokens"], 0.6)
        self.assertAlmostEqual(row["total_variation_lower_bound"], 0.1)

    def test_full_vocabulary_has_exact_tv_and_kl(self):
        row = MODULE.compare_distributions(report([0.6, 0.4]), report([0.5, 0.5]))[0]
        self.assertAlmostEqual(row["total_variation_lower_bound"], 0.1)
        self.assertAlmostEqual(
            row["kl_reference_to_candidate"], 0.6 * math.log(1.2) + 0.4 * math.log(0.8)
        )

    def test_incompatible_or_nonfinite_records_are_rejected(self):
        reference = report([0.6, 0.4])
        with self.assertRaisesRegex(ValueError, "histories differ"):
            MODULE.compare_distributions(reference, report([0.6, 0.4], (4,)))
        bad = report([0.6, 0.4])
        bad["outputs"][0]["logprobs"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            MODULE.compare_distributions(reference, bad)

    def test_single_scored_reference_has_no_invented_runner_up(self):
        row = MODULE.compare_distributions(report([0.6, None]), report([0.5, None]))[0]
        self.assertIsNone(row["reference_top1_margin"])
        self.assertTrue(row["top1_equal"])


if __name__ == "__main__":
    unittest.main()
