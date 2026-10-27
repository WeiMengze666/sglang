import importlib
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
split = importlib.import_module("split_workload")


def row(i):
    return {
        "id": str(i),
        "conversations": [{"value": f"prompt {i}"}, {"value": "answer"}],
        "prompt_len": 3,
        "output_len": 2,
    }


class HeldoutTests(unittest.TestCase):
    def test_larger_pool_excludes_original_pilot_and_duplicate_conversations(self):
        pilot = [row(i) for i in range(8)]
        shuffled = list(pilot)
        random.Random(42).shuffle(shuffled)
        duplicate = dict(shuffled[0], id="different-source-id")
        pool = [row(i) for i in range(20)] + [duplicate, row(19)]
        selected, excluded = split.select_heldout(
            pool,
            pilot,
            pilot_seed=42,
            pilot_requests=4,
            seed=7,
            requests=16,
        )
        actual = {split.identity(r) for r in selected}
        self.assertEqual(excluded, {split.identity(r) for r in shuffled[:4]})
        self.assertFalse(actual & excluded)
        self.assertEqual(len(actual), 16)
        self.assertEqual(pilot, [row(i) for i in range(8)])

    def test_insufficient_unique_pool_and_incomplete_pilot_fail(self):
        for pool, pilot, pilot_n, requested in (
            ([row(0)] * 5 + [row(1)], [row(1)], 1, 2),
            ([row(i) for i in range(10)], [row(0)], 2, 1),
        ):
            with self.assertRaises(ValueError):
                split.select_heldout(
                    pool,
                    pilot,
                    pilot_seed=42,
                    pilot_requests=pilot_n,
                    seed=7,
                    requests=requested,
                )


if __name__ == "__main__":
    unittest.main()
