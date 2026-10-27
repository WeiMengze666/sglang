"""A trace must not certify mixed decisions, missing peers, or early switches."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
validate = importlib.import_module("bench_preemption").validate_cluster_trace
validate_decisions = importlib.import_module("bench_preemption").validate_decision_trace
validate_progress = importlib.import_module("bench_preemption").validate_progress_trace
overflow = importlib.import_module("bench_preemption").trace_overflow_ctas


class CooperativeTraceTests(unittest.TestCase):
    def test_sequence_counts_detect_overflow_past_unrecorded_flush(self):
        metadata = {"trace_sequence_counts": [8, 10], "trace_capacity_per_cta": 8}
        self.assertEqual(overflow(metadata, 2, 8), 1)
        metadata["trace_sequence_counts"] = [8, 8]
        self.assertEqual(overflow(metadata, 2, 8), 0)
        for invalid in [
            {},
            {**metadata, "trace_sequence_counts": [8]},
            {**metadata, "trace_sequence_counts": [8, -1]},
            {**metadata, "trace_capacity_per_cta": 16},
        ]:
            with self.assertRaises(ValueError):
                overflow(invalid, 2, 8)

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

    def decisions(self):
        return [
            dict(
                cta=0,
                claim_sequence=claim,
                handoff_sequence=handoff,
                observed_ns=observed,
                retired_self=retired,
                retired_peer=retired,
                previous_phase=previous,
                preferred_phase=selected,
                selected_phase=selected,
                preferred_runnable=1 << selected,
            )
            for claim, handoff, observed, retired, previous, selected in [
                (0, 0, 5, 0, 2, 0),
                (1, 1, 15, 0, 0, 0),
                (2, 3, 35, 1, 0, 1),
            ]
        ]

    def test_decisions_include_pre_drain_inflight_work(self):
        result = validate_decisions(self.fixture(), self.decisions(), 2)
        self.assertEqual(result["decisions"], 3)
        switch = result["phase_switches"][0]
        self.assertEqual(switch["old_tiles_in_flight_per_cta"], 1)
        self.assertEqual(switch["unretired_sequence_upper_bound"], 1)
        self.assertTrue(switch["deferred_control_flush"])
        self.assertEqual(switch["observed_to_handoff_us"], 0.016)

    def test_missing_duplicate_and_nonleader_decisions_fail(self):
        decisions = self.decisions()
        for invalid in [decisions[:-1], decisions + [decisions[-1]]]:
            with self.assertRaises(ValueError):
                validate_decisions(self.fixture(), invalid, 2)
        decisions[-1]["cta"] = 1
        with self.assertRaisesRegex(ValueError, "leader"):
            validate_decisions(self.fixture(), decisions, 2)

    def test_invalid_observation_prefix_or_phase_fails(self):
        for field, value in [
            ("observed_ns", 100),
            ("retired_self", 3),
            ("previous_phase", 1),
            ("selected_phase", 0),
            ("handoff_sequence", 4),
            ("claim_sequence", 1),
        ]:
            decisions = self.decisions()
            decisions[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_decisions(self.fixture(), decisions, 2)
        decisions = self.decisions()
        decisions[-1]["retired_self"] = -1
        result = validate_decisions(self.fixture(), decisions, 2)
        self.assertIsNone(result["phase_switches"][0]["unretired_sequence_upper_bound"])

    def test_empty_truncated_and_unretired_traces_fail(self):
        with self.assertRaises(ValueError):
            validate([], 2)
        with self.assertRaises(ValueError):
            validate(self.fixture(), 2, 1)
        rows = self.fixture()
        rows[0]["finish_ns"] = 0
        with self.assertRaises(ValueError):
            validate(rows, 2)

    def progress(self):
        return [
            dict(
                cta=r["cta"],
                sequence=r["sequence"],
                accumulator_ready_ns=r["begin_ns"] + 2,
                output_submitted_ns=r["finish_ns"] - 2,
            )
            for r in self.fixture()
        ]

    def test_progress_distinguishes_two_outputs_from_one_unfinished_gemm(self):
        rows, decisions, progress = self.fixture(), self.decisions(), self.progress()
        decisions[-1]["observed_ns"] = 25
        decisions[-1]["retired_self"] = decisions[-1]["retired_peer"] = 0
        for p in progress:
            if p["sequence"] == 0:
                p["output_submitted_ns"] = 24
            elif p["sequence"] == 1:
                p["accumulator_ready_ns"] = 32
        result = validate_progress(rows, decisions, progress, 2)
        self.assertEqual(result["max_gemms_without_ready_observation_per_cta"], 1)
        for c in result["phase_switches"][0]["ctas"]:
            self.assertEqual(c["unretired_outputs"], 2)
            self.assertEqual(c["submitted_outputs_awaiting_retirement"], 1)

    def test_progress_does_not_hide_multiple_unfinished_gemms(self):
        decisions, progress = self.decisions(), self.progress()
        decisions[-1]["observed_ns"] = 25
        decisions[-1]["retired_self"] = decisions[-1]["retired_peer"] = 0
        for p in progress:
            if p["sequence"] <= 1:
                p["accumulator_ready_ns"] = 27
        result = validate_progress(self.fixture(), decisions, progress, 2)
        self.assertEqual(result["max_gemms_without_ready_observation_per_cta"], 2)

    def test_progress_rejects_incomplete_or_out_of_order_evidence(self):
        for samples in [self.progress()[:-1], self.progress() + [self.progress()[0]]]:
            with self.assertRaises(ValueError):
                validate_progress(self.fixture(), self.decisions(), samples, 2)
        for field, value in [
            ("accumulator_ready_ns", 0),
            ("accumulator_ready_ns", 100),
            ("output_submitted_ns", 1),
            ("output_submitted_ns", 100),
        ]:
            samples = self.progress()
            samples[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_progress(self.fixture(), self.decisions(), samples, 2)

    def test_data_operations_are_not_counted_as_gemms(self):
        rows, samples = self.fixture(), self.progress()
        for r in rows:
            r["stage"] = 3
        for p in samples:
            p["accumulator_ready_ns"] = 0
        result = validate_progress(rows, self.decisions(), samples, 2)
        self.assertEqual(result["max_gemms_without_ready_observation_per_cta"], 0)
        self.assertEqual(
            result["phase_switches"][0]["ctas"][0]["data_operations_before_submission"],
            1,
        )
        samples[0]["accumulator_ready_ns"] = 12
        with self.assertRaisesRegex(ValueError, "Non-GEMM"):
            validate_progress(rows, self.decisions(), samples, 2)


if __name__ == "__main__":
    unittest.main()
