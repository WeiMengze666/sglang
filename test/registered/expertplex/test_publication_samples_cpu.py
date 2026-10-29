"""Reject partial/failed GPU bursts instead of reporting zero-time samples."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
parse = importlib.import_module("bench_apk_sharing").publication_samples


class PublicationSamplesTests(unittest.TestCase):
    def test_concurrent_epochs_and_gaps_are_checked_on_one_clock(self):
        rows = [
            [1000, 1100, 1500, 1400, 0, 0, 11, 18],
            [1600, 1700, 1900, 2000, 0, 0, 12, 19],
        ]
        result = parse(rows, (True, True), 2, require_epochs=True)
        self.assertEqual(result[0]["response_upper_bound_us"], [0.5, 0.3])
        self.assertEqual(result[1]["inter_sample_gap_us"], 0.1)
        self.assertEqual(result[1]["epochs"], [12, 19])
        self.assertIsNone(result[0]["inter_sample_gap_us"])
        self.assertEqual(
            parse([row[:6] for row in rows], (True, True), 2)[1]["epochs"], None
        )

    def test_inactive_phase_never_completes_or_advances(self):
        rows = [[1000, 1100, 0, 1400, 0, 0, 0, 2]]
        result = parse(rows, (False, True), 1, require_epochs=True)
        self.assertEqual(result[0]["response_upper_bound_us"], [None, 0.3])
        for column in (2, 6):
            bad = copy.deepcopy(rows)
            bad[0][column] = 1
            with self.assertRaises(ValueError):
                parse(bad, (False, True), 1, require_epochs=True)

    def test_missing_error_timeout_epoch_reuse_and_overlap_fail(self):
        rows = [
            [1000, 1100, 1500, 1400, 0, 0, 11, 18],
            [1600, 1700, 1900, 2000, 0, 0, 12, 19],
        ]
        for index, column, value in (
            (0, 0, 0), (0, 1, 999), (0, 2, 0), (0, 3, 1100),
            (0, 4, 1), (1, 5, 2), (1, 6, 11), (1, 7, 20),
            (1, 0, 1499), (0, 6, 0), (0, 0, float("nan")),
        ):
            bad = copy.deepcopy(rows)
            bad[index][column] = value
            with self.subTest(index=index, column=column), self.assertRaises(ValueError):
                parse(bad, (True, True), 2, require_epochs=True)
        for bad in (rows[:1], [rows[0], [0] * 8], [rows[0], rows[1][:6]]):
            with self.assertRaises(ValueError):
                parse(bad, (True, True), 2, require_epochs=True)


if __name__ == "__main__":
    unittest.main()
