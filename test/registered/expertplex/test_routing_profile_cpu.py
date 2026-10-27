import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
summarize = importlib.import_module("summarize_routing").summarize_counts


class RoutingProfileTests(unittest.TestCase):
    def test_weights_tokens_and_retains_layer_imbalance(self):
        result = summarize(
            [[[2, 1, 1], [0, 2, 2]], [[1, 0, 1], [1, 1, 0]], [[0, 0, 0]] * 2],
            layers=2,
            experts=3,
            top_k=2,
        )
        self.assertEqual(result["forward_passes"], 2)
        self.assertEqual(result["tokens_per_layer"], 3)
        self.assertEqual(result["counts_by_layer"], [[3, 1, 2], [1, 3, 2]])
        self.assertEqual(result["expert_probabilities"], [2 / 3] * 3)
        self.assertEqual(result["probabilities_by_layer"][0], [1, 1 / 3, 2 / 3])

    def test_rejects_missing_layers_invalid_counts_and_empty_capture(self):
        for records in (
            [],
            [[[0, 0, 0]] * 2],
            [[[1, 1, 0]]],
            [[[1, 1, 0], [1, 0, 0]]],
            [[[2, 0, 0]] * 2],
            [[[-1, 2, 1]] * 2],
            [[[1.0, 1, 0]] * 2],
            [[[True, 1, 0]] * 2],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                summarize(records, layers=2, experts=3, top_k=2)


if __name__ == "__main__":
    unittest.main()
