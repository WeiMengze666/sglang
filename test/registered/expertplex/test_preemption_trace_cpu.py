"""A trace must not certify mixed decisions, missing peers, or early switches."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
validate = importlib.import_module("bench_preemption").validate_cluster_trace


class CooperativeTraceTests(unittest.TestCase):
    def fixture(self):
        return [
            dict(
                cta=cta,
                sequence=seq,
                epoch=1,
                slot=phase,
                phase=phase,
                stage=1,
                begin_ns=begin + cta,
                finish_ns=finish + cta,
            )
            for seq, phase, begin, finish in [
                (0, 0, 10, 30),
                (1, 0, 20, 40),
                (3, 1, 50, 60),
            ]
            for cta in [0, 1]
        ]

    def test_overlapped_same_phase_and_drained_switch_pass(self):
        self.assertEqual(
            validate(self.fixture(), 2), {"cluster_decisions": 3, "phase_switches": 1}
        )

    def test_missing_duplicate_or_mixed_cluster_members_fail(self):
        rows = self.fixture()
        for invalid in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                validate(invalid, 2)
        invalid = copy.deepcopy(rows)
        invalid[-1]["stage"] = 2
        with self.assertRaisesRegex(ValueError, "different work"):
            validate(invalid, 2)

    def test_all_ctas_must_retire_before_phase_switch(self):
        rows = self.fixture()
        rows[-1]["begin_ns"] = 40  # Leader finished at 40, peer only at 41.
        with self.assertRaisesRegex(ValueError, "before all"):
            validate(rows, 2)
        rows = self.fixture()
        rows[0]["finish_ns"] = 100
        with self.assertRaisesRegex(ValueError, "before all"):
            validate(rows, 2)

    def test_empty_truncated_and_unretired_traces_fail(self):
        with self.assertRaises(ValueError):
            validate([], 2)
        with self.assertRaises(ValueError):
            validate(self.fixture(), 2, 1)
        rows = self.fixture()
        rows[0]["finish_ns"] = 0
        with self.assertRaises(ValueError):
            validate(rows, 2)


if __name__ == "__main__":
    unittest.main()
