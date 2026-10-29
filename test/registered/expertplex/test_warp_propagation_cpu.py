"""Reject incomplete warp and clock-domain evidence in preemption diagnostics."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
validate = importlib.import_module("bench_preemption").validate_warp_propagation


class WarpPropagationTests(unittest.TestCase):
    def fixture(self):
        records, warps, decisions = [], [], []
        for seq, phase, begin, end, admission in [
            (0, 0, 10, 30, 5),
            (2, 1, 50, 70, 40),
        ]:
            for cta in (0, 1):
                records.append(
                    dict(
                        cta=cta,
                        sequence=seq,
                        epoch=1,
                        slot=phase,
                        phase=phase,
                        stage=1,
                        begin_ns=begin + cta,
                        finish_ns=end + cta,
                        ready_seen_ns=admission,
                    )
                )
                warps.append(
                    dict(
                        cta=cta,
                        sequence=seq,
                        observed_ns=[
                            begin + cta + 1,
                            begin + cta + 2 if cta == 0 else 0,
                            0,
                            0,
                            begin + cta + 3,
                            begin + cta + 4,
                            begin + cta + 5,
                            begin + cta + 6,
                        ],
                    )
                )
            decisions.append(
                dict(
                    cta=0,
                    claim_sequence=seq,
                    handoff_sequence=seq,
                    observed_ns=begin - 3,
                    retired_self=seq,
                    retired_peer=seq,
                    previous_phase=2 if seq == 0 else 0,
                    preferred_phase=phase,
                    selected_phase=phase,
                    preferred_runnable=1 << phase,
                )
            )
        cases = [
            dict(
                decode=dict(epoch=1, start_ns=40),
                admission_roundtrip_upper_bound_us={"decode": 2.0},
            )
        ]
        return records, decisions, warps, cases, 2

    def test_bound_adds_durations_without_cross_device_clock_subtraction(self):
        result = validate(*self.fixture(), ctas=2)
        bound = result["arrivals"][0]["clusters"][0]
        self.assertAlmostEqual(bound["admission_to_all_warps_us"], 0.017)
        self.assertAlmostEqual(bound["publication_to_all_warps_upper_bound_us"], 2.017)
        self.assertAlmostEqual(
            result["phase_switches"][0]["decision_to_all_warps_us"], 0.010
        )
        self.assertEqual(result["tiles"], 4)
        # Moving only E's entire clock cannot change either bound.
        fixture = self.fixture()
        for row in fixture[0]:
            for field in ("begin_ns", "finish_ns", "ready_seen_ns"):
                row[field] += 10**12
        for row in fixture[1]:
            row["observed_ns"] += 10**12
        for row in fixture[2]:
            row["observed_ns"] = [v + 10**12 if v else 0 for v in row["observed_ns"]]
        fixture[3][0]["decode"]["start_ns"] += 10**12
        self.assertEqual(validate(*fixture, ctas=2), result)

    def test_missing_duplicate_unknown_or_malformed_receipts_fail(self):
        for modification in ("missing", "duplicate", "unknown", "shape"):
            f = self.fixture()
            if modification == "missing":
                f[2].pop()
            elif modification == "duplicate":
                f[2].append(copy.deepcopy(f[2][-1]))
            elif modification == "unknown":
                f[2][-1]["sequence"] = 99
            else:
                f[2][-1]["observed_ns"].pop()
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                validate(*f, ctas=2)

    def test_every_pipeline_warp_and_only_those_warps_must_record(self):
        for cta in (0, 1):
            for warp in range(8):
                f = self.fixture()
                values = f[2][cta]["observed_ns"]
                values[warp] = 0 if values[warp] else 20
                with self.subTest(cta=cta, warp=warp), self.assertRaises(ValueError):
                    validate(*f, ctas=2)

    def test_invalid_clock_ordering_fails(self):
        for timestamp in (9, 100, 11.5):
            f = self.fixture()
            f[2][0]["observed_ns"][4] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                validate(*f, ctas=2)
        f = self.fixture()
        f[2][0]["observed_ns"][0] = 29
        with self.assertRaisesRegex(ValueError, "before producer"):
            validate(*f, ctas=2)

    def test_missing_or_stale_admission_and_invalid_roundtrip_fail(self):
        for value in (0, -1, float("inf"), float("nan")):
            f = self.fixture()
            f[3][0]["admission_roundtrip_upper_bound_us"]["decode"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(*f, ctas=2)
        for field, value in (("epoch", 2), ("start_ns", 41)):
            f = self.fixture()
            f[3][0]["decode"][field] = value
            with self.assertRaises(ValueError):
                validate(*f, ctas=2)
        f = self.fixture()
        f[3].append(copy.deepcopy(f[3][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(*f, ctas=2)
        f = self.fixture()
        f[3].clear()
        with self.assertRaises(ValueError):
            validate(*f, ctas=2)

    def test_idle_clusters_remain_in_the_configured_inventory(self):
        result = validate(*self.fixture(), ctas=4)
        self.assertEqual(result["configured_ctas"], 4)
        self.assertEqual(result["configured_clusters"], 2)
        self.assertEqual(result["clusters_without_any_tile"], [1])
        self.assertEqual(result["arrivals"][0]["clusters_without_decode_work"], [1])
        self.assertEqual(len(result["arrivals"][0]["clusters"]), 1)

    def test_missing_or_invalid_launch_inventory_fails(self):
        with self.assertRaises(TypeError):
            validate(*self.fixture())
        for ctas in (0, -2, 1, 3, True, 2.0):
            with self.subTest(ctas=ctas), self.assertRaises(ValueError):
                validate(*self.fixture(), ctas=ctas)
        for cluster_size in (0, -1, 3, True, 2.0):
            args = list(self.fixture())
            args[-1] = cluster_size
            with self.subTest(cluster_size=cluster_size), self.assertRaises(ValueError):
                validate(*args, ctas=4)

    def test_unknown_negative_and_noninteger_ctas_fail(self):
        for cta in (-1, 4, True, 0.0):
            args = self.fixture()
            args[0][0]["cta"] = cta
            with (
                self.subTest(cta=cta),
                self.assertRaisesRegex(ValueError, "configured"),
            ):
                validate(*args, ctas=4)

    def test_truncated_trace_cannot_certify_receipt(self):
        with self.assertRaises(ValueError):
            validate(*self.fixture(), truncated_ctas=1, ctas=2)


if __name__ == "__main__":
    unittest.main()
