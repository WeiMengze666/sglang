"""Do not average away a failed calibration instrumentation replica."""

import copy
import importlib
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
module = importlib.import_module("profile_validation")
validate = module.validate_instrumentation_replicas


def fixture(phase="prefill"):
    variants = []
    sequence = 128 + (3 if phase == "decode" else 0)
    for index, mode in enumerate(("off", "on", "on", "off")):
        ids = [0, 2] if mode == "on" else [0, 1, 2]
        settings = {key: None for key in module.CONTRACT_KEYS}
        settings.update(phase=phase, measure_split="train", measurement_only=True,
                        iterations=2, timing_events=mode, client_id=1,
                        output_dir=f"/fixture/run-{index}")
        proof = {"phase": phase, "batch": 1, "tokens": 1,
                 "sequence_lengths": [sequence], "can_run_graph": True}
        variants.append({
            "status": "measurement_completed" if mode == "on" else "control_completed",
            "source_node": {"model_path": "/fixture", "overlap": "none"},
            "settings": settings, "model_num_layers": 3,
            "measured_layer_ids": ids, "measurement_grid": [["train", 1, 128]],
            "samples": [{"phase": phase, "split": "train", "x": 1,
                "sequence_length": sequence, "forward_samples_s": [100.0, 100.0],
                "execution_proof": [copy.deepcopy(proof), copy.deepcopy(proof)],
                "last_logits_sha256": "a" * 64, "last_logits_shape": [1, 128],
                "last_logits_dtype": "torch.bfloat16", "measured_layer_ids": ids,
                "layer_samples_s": [[1.0] * len(ids)] * 2 if mode == "on" else []}],
        })
    return variants


class InstrumentationTests(unittest.TestCase):
    def test_asymmetric_replica_fails_even_when_mean_is_below_three_percent(self):
        reports = fixture()
        reports[1]["samples"][0]["forward_samples_s"] = [104.0, 104.0]
        self.assertLess((1.04 + 1.0) / 2, 1.03)
        result = validate(reports)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["points"][0]["event_overhead_ratios"], [1.04, 1.0])

    def test_gate_is_exact_and_cannot_use_a_stale_summary(self):
        reports = fixture()
        reports[1]["samples"][0]["forward_samples_s"] = [103.0, 103.0]
        reports[1]["samples"][0]["forward_s"] = 1.0
        self.assertEqual(validate(reports)["status"], "passed")
        reports[1]["samples"][0]["forward_samples_s"] = [math.nextafter(103.0, math.inf)] * 2
        self.assertEqual(validate(reports)["status"], "failed")

    def test_matched_decode_and_unequal_on_off_cohorts_are_valid(self):
        reports = fixture("decode")
        before = copy.deepcopy(reports)
        self.assertEqual(validate(reports)["status"], "passed")
        self.assertEqual(reports, before)

    def test_sequence_requires_both_bracketing_controls(self):
        reports = fixture()
        for invalid in (reports[:3], [reports[1], reports[0], reports[2], reports[3]],
                        [reports[0], reports[1], reports[1], reports[3]]):
            with self.assertRaises(ValueError):
                validate(invalid)

    def test_mismatched_model_contract_or_grid_is_rejected(self):
        mutations = [
            lambda r: r["source_node"].update(model_path="/other"),
            lambda r: r["settings"].update(client_id=2),
            lambda r: r["settings"].update(component="shared"),
            lambda r: r["settings"].update(iterations=3),
            lambda r: r.update(measurement_grid=[["holdout", 1, 128]]),
            lambda r: r.update(model_num_layers=4),
            lambda r: r.update(status="failed"),
        ]
        for mutation in mutations:
            reports = fixture()
            mutation(reports[1])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(reports)

    def test_mismatched_on_cohort_is_rejected(self):
        reports = fixture()
        reports[2]["measured_layer_ids"] = [0, 1]
        reports[2]["samples"][0]["measured_layer_ids"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "different layer cohorts"):
            validate(reports)

    def test_missing_duplicate_and_unrequested_shapes_are_rejected(self):
        for mutation in (
            lambda r: r.update(samples=[]),
            lambda r: r["samples"].append(copy.deepcopy(r["samples"][0])),
            lambda r: r["samples"][0].update(sequence_length=129),
            lambda r: r["samples"][0].update(x=True),
        ):
            reports = fixture()
            mutation(reports[1])
            with self.assertRaises(ValueError):
                validate(reports)

    def test_native_execution_and_logits_must_match_all_four_variants(self):
        for field, value in (("last_logits_sha256", "b" * 64),
                             ("last_logits_dtype", "torch.float32"),
                             ("last_logits_shape", [1, 256]),
                             ("execution_proof", [{"can_run_graph": False}] * 2)):
            reports = fixture()
            reports[1]["samples"][0][field] = value
            with self.assertRaises(ValueError):
                validate(reports)

    def test_nonfinite_bool_and_incomplete_timing_are_rejected(self):
        for values in ([float("nan"), 1.0], [float("inf"), 1.0],
                       [True, 1.0], [0, 1.0], [1.0]):
            reports = fixture()
            reports[1]["samples"][0]["forward_samples_s"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(reports)


if __name__ == "__main__":
    unittest.main()
