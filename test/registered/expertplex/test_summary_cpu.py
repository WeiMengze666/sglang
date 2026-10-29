"""Validate paper attainment denominators and comparable warmup conditions."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[3] / "examples/expertplex/summarize_experiments.py"
)
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("expertplex_summary", PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_fixture(root, warmup=1):
    (root / "sweep").mkdir(parents=True)
    settings = dict(
        model_path="/model",
        tokenizer_path="/tokenizer",
        weight_mode="synthetic",
        dataset_path="/dataset",
        requests=4,
        seed=42,
        sequence_cap=32,
        decode_kv_tokens=96,
        attention_backend="triton",
        ttft_slo_ms=1000,
        tpot_slo_ms=100,
        variant="expertplex",
        warmup_requests=warmup,
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "environment": {"SGLANG_EXPERTPLEX_SYNTHETIC_SEED": "1234"},
            }
        )
    )
    details = dict(
        request_rate=1,
        input_lens=[3] * 4,
        output_lens=[3, 2, 1, 0],
        successes=[True, True, True, False],
        ttfts=[0.1, 2, 0.2, 0],
        latencies=[0.2, 2.05, 0.2, 0],
        duration=3,
    )
    (root / "sweep/detail.json").write_text(json.dumps(details))
    point = dict(
        details_file="detail.json",
        request_rate=1,
        attempted_requests=4,
        completed_requests=3,
        joint_slo_met_requests=1,
        joint_slo_attainment=0.25,
        meets_p90_joint_slo=False,
    )
    (root / "sweep/joint-slo.json").write_text(
        json.dumps({"p90_goodput_max_tested_arrival_rps": None, "results": [point]})
    )


class ServingSummaryTests(unittest.TestCase):
    def test_stale_summary_metrics_cannot_override_raw_requests(self):
        for key, value in (
            ("joint_slo_met_requests", 4),
            ("joint_slo_attainment", 1.0),
            ("meets_p90_joint_slo", True),
            ("attempted_requests", 3),
            ("completed_requests", 4),
            ("request_rate", 2),
            ("observed_slo_goodput_rps", 10),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "run"
                write_fixture(root)
                p = root / "sweep/joint-slo.json"
                data = json.loads(p.read_text())
                data["results"][0][key] = value
                p.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "Raw-derived"):
                    SUMMARY.summarize([root])

    def test_invalid_rate_and_mismatched_slo_headers_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_fixture(root)
            p = root / "sweep/detail.json"
            original = p.read_text()
            for rate in (None, 0, -1, True, float("inf"), float("nan")):
                details = json.loads(original)
                details["request_rate"] = rate
                p.write_text(json.dumps(details))
                with (
                    self.subTest(rate=rate),
                    self.assertRaisesRegex(ValueError, "rate"),
                ):
                    SUMMARY.summarize([root])
            p.write_text(original)
            p = root / "sweep/joint-slo.json"
            data = json.loads(p.read_text())
            data["ttft_slo_ms"] = 2000
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SLO mismatch"):
                SUMMARY.summarize([root])

    def test_truncated_details_or_false_maximum_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_fixture(root)
            p = root / "sweep/detail.json"
            original = p.read_text()
            details = json.loads(original)
            details["latencies"].pop()
            p.write_text(json.dumps(details))
            with self.assertRaises(ValueError):
                SUMMARY.summarize([root])
            p.write_text(original)
            p = root / "sweep/joint-slo.json"
            data = json.loads(p.read_text())
            data["p90_goodput_max_tested_arrival_rps"] = 1
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "maximum"):
                SUMMARY.summarize([root])

    def test_output_lengths_must_match_without_using_failed_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "first", Path(tmp) / "second"]
            for root in roots:
                write_fixture(root)
            p = roots[1] / "sweep/detail.json"
            details = json.loads(p.read_text())
            details["output_lens"][3] = 99  # Failed; not useful delivered tokens.
            p.write_text(json.dumps(details))
            report = SUMMARY.summarize(roots)
            self.assertEqual(report["rows"][1]["output_tokens_per_second"], 2)
            self.assertFalse(report["output_length_reference_complete"])
            self.assertAlmostEqual(report["rows"][1]["observed_slo_goodput_rps"], 1 / 3)
            details["output_lens"][0] = 4  # Still meets SLO, but a different workload.
            p.write_text(json.dumps(details))
            with self.assertRaisesRegex(ValueError, "output lengths"):
                SUMMARY.summarize(roots)

    def test_explicit_memory_tuning_retains_fractions_and_other_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "first", Path(tmp) / "second"]
            for root, fraction in zip(roots, (0.935, 0.9)):
                write_fixture(root)
                path = root / "experiment.json"
                data = json.loads(path.read_text())
                data["settings"]["mem_fraction_static"] = fraction
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                SUMMARY.summarize(roots)
            report = SUMMARY.summarize(roots, allow_memory_tuning=True)
            self.assertEqual(
                [r["mem_fraction_static"] for r in report["runs"]], [0.935, 0.9]
            )
            self.assertNotIn("mem_fraction_static", report["common_conditions"])
            self.assertIn("Per-method tuning", report["memory_fraction_policy"])
            for key, value in (
                ("sequence_cap", 64),
                ("decode_kv_tokens", None),
                ("ttft_slo_ms", 2000),
            ):
                changed = json.loads(path.read_text())
                changed["settings"][key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    SUMMARY.summarize(roots, allow_memory_tuning=True)
                path.write_text(json.dumps(data))

    def test_memory_budget_or_capacity_policy_cannot_be_mixed(self):
        for field, value in (("mem_fraction_static", 0.9), ("decode_kv_tokens", None)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                roots = [Path(tmp) / "first", Path(tmp) / "second"]
                for root in roots:
                    write_fixture(root)
                path = roots[1] / "experiment.json"
                experiment = json.loads(path.read_text())
                experiment["settings"][field] = value
                path.write_text(json.dumps(experiment))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    SUMMARY.summarize(roots)

    def test_failed_requests_stay_in_each_attainment_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_fixture(root)
            row = SUMMARY.summarize([root])["rows"][0]
        self.assertEqual(row["ttft_attainment"], 0.5)
        self.assertEqual(row["tpot_attainment"], 0.5)
        self.assertEqual(row["joint_attainment"], 0.25)
        self.assertEqual(row["completed"], 3)

    def test_different_warmup_counts_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            write_fixture(first)
            write_fixture(second, warmup=8)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                SUMMARY.summarize([first, second])

    def test_sampling_or_native_optimization_defaults_cannot_be_mixed(self):
        for change in ("sampling", "serving_defaults"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                first, second = Path(tmp) / "first", Path(tmp) / "second"
                write_fixture(first)
                write_fixture(second)
                path = second / "experiment.json"
                experiment = json.loads(path.read_text())
                if change == "sampling":
                    experiment["settings"]["sampling_backend"] = "flashinfer"
                else:
                    experiment["serving_defaults_version"] = (
                        "native-overlap-autotune-v1"
                    )
                path.write_text(json.dumps(experiment))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    SUMMARY.summarize([first, second])

    def test_failed_partial_run_is_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            write_fixture(root)
            path = root / "experiment.json"
            experiment = json.loads(path.read_text())
            experiment["status"] = "failed"
            experiment["settings"].update(rates=[1, 2], numerical_check=True)
            path.write_text(json.dumps(experiment))
            report = SUMMARY.summarize([root])
        run = report["runs"][0]
        self.assertEqual(run["experiment_status"], "failed")
        self.assertTrue(run["numerical_diagnostic_requested"])
        self.assertFalse(run["complete_requested_rate_grid"])
        self.assertEqual(run["measured_rates"], [1])
        self.assertEqual(len(report["rows"]), 1)

    def test_same_paths_and_lengths_cannot_hide_changed_input_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "first", Path(tmp) / "second"]
            for root, digest in zip(roots, ("a" * 64, "b" * 64)):
                write_fixture(root)
                path = root / "experiment.json"
                experiment = json.loads(path.read_text())
                experiment["input_artifact_sha256"] = {"/dataset": digest}
                path.write_text(json.dumps(experiment))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                SUMMARY.summarize(roots)


if __name__ == "__main__":
    unittest.main()
