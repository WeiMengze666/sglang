"""Validate paper attainment denominators and comparable warmup conditions."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[3] / "examples/expertplex/summarize_experiments.py"
)
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
