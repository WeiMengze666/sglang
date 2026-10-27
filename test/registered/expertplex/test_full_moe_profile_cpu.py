"""Do not fit traced, incomplete, or repeated concurrent data as isolated latency."""

import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
profile = importlib.import_module("profile_full_moe")
collect = profile.collect_samples


class FullMoEProfileTests(unittest.TestCase):
    def test_cli_measures_both_grids_and_preserves_training_fit(self):
        calls = []

        def measure(command, **kwargs):
            path = Path(command[command.index("--output") + 1])
            decode = int(command[command.index("--decode-rows") + 1])
            prefill = list(
                map(
                    int,
                    command[
                        command.index("--prefill-rows") + 1 : command.index(
                            "--decode-rows"
                        )
                    ],
                )
            )
            calls.append((prefill, decode))
            data = self.fixture()
            data["native_source_sha256"] = "same-runtime"
            data["settings"]["decode_rows"] = decode
            data["cases"] = []
            for tokens in prefill:
                counts = [[tokens, 0, 0], [decode, 0, 0]]
                # A deliberately different holdout timing must not alter the fit.
                factor = 2 if path.name.startswith("holdout") else 1
                data["cases"].append(
                    {
                        "prefill_rows": tokens,
                        "expert_rows": counts,
                        "isolated_samples": [
                            [{"expert_ms": factor * (1 + (row[0] + 127) // 128)}]
                            for row in counts
                        ],
                        "exact_isolated_output": [True, True],
                    }
                )
            path.write_text(json.dumps(data))

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "profile"
            argv = [
                "profile_full_moe.py",
                "--sms",
                "4",
                "--prefill-rows",
                "128",
                "512",
                "--decode-rows",
                "1",
                "256",
                "--holdout-prefill-rows",
                "256",
                "--holdout-decode-rows",
                "128",
                "--output-dir",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(profile.subprocess, "run", side_effect=measure),
            ):
                profile.main()
            report = json.loads((output / "profiles.json").read_text())
        self.assertEqual(calls, [([128, 512], 1), ([128, 512], 256), ([256], 128)])
        self.assertEqual(report["status"], "completed")
        for fit in report["fits"].values():
            self.assertAlmostEqual(fit["alpha"], 0.001)
            self.assertAlmostEqual(fit["beta"], 0.001)
            self.assertAlmostEqual(fit["holdout"]["max_relative_error"], 0.5)

    def test_holdout_does_not_refit_or_hide_extrapolation(self):
        fit = {"alpha": 0.001, "beta": 0.001}
        training = [{"x": 1}, {"x": 3}]
        original = copy.deepcopy(fit)
        result = profile.evaluate_holdout(
            fit,
            training,
            [{"x": 2, "latency_s": 0.006}, {"x": 4, "latency_s": 0.005}],
        )
        self.assertEqual(fit, original)
        self.assertEqual(result["max_relative_error"], 0.5)
        self.assertEqual(result["median_relative_error"], 0.25)
        self.assertEqual(result["training_footprint_range"], [1, 3])
        self.assertEqual(
            [r["within_training_footprint_range"] for r in result["measurements"]],
            [True, False],
        )
        self.assertTrue(result["all_predictions_positive"])

    def test_nonpositive_prediction_is_reported_and_invalid_measurements_rejected(self):
        fit = {"alpha": -0.002, "beta": 0.001}
        training = [{"x": 3}, {"x": 4}]
        result = profile.evaluate_holdout(fit, training, [{"x": 1, "latency_s": 0.001}])
        self.assertFalse(result["all_predictions_positive"])
        self.assertEqual(result["max_relative_error"], 2)
        for rows in (
            [],
            [{"x": 1, "latency_s": 0}],
            [{"x": 1, "latency_s": float("nan")}],
        ):
            with self.assertRaises(ValueError):
                profile.evaluate_holdout(fit, training, rows)

    def fixture(self):
        case = {
            "prefill_rows": 512,
            "expert_rows": [[129, 0, 127], [1, 0, 1]],
            "isolated_samples": [
                [{"expert_ms": x} for x in (2, 3, 20)],
                [{"expert_ms": x} for x in (1, 2, 9)],
            ],
            "exact_isolated_output": [True, True],
        }
        return {
            "status": "passed",
            "trace_status": "disabled",
            "settings": {"prefill_preprocess": True, "decode_rows": 2},
            "executor": {
                "prefill_preprocess": True,
                "benchmark_stage": 0,
                "prefill_tile_rows": 128,
                "decode_tile_rows": 128,
            },
            "cases": [case, copy.deepcopy(case)],
        }

    def test_deduplicates_repeats_and_preserves_padding_and_units(self):
        result = collect(self.fixture())
        self.assertEqual(len(result["prefill"]), 1)
        self.assertEqual(len(result["decode"]), 1)
        self.assertEqual(result["prefill"][0]["x"], 3)
        self.assertEqual(result["decode"][0]["x"], 2)
        self.assertEqual(result["prefill"][0]["latency_s"], 0.003)
        self.assertEqual(result["decode"][0]["latency_s"], 0.002)

    def test_refuses_partial_or_traced_execution(self):
        for section, name, value in [
            (None, "status", "failed"),
            (None, "trace_status", "enabled"),
            ("settings", "prefill_preprocess", False),
            ("executor", "prefill_preprocess", False),
            ("executor", "benchmark_stage", 1),
        ]:
            data = self.fixture()
            (data if section is None else data[section])[name] = value
            with (
                self.subTest(name=name, section=section),
                self.assertRaises(ValueError),
            ):
                collect(data)
        data = self.fixture()
        data["cases"][0]["exact_isolated_output"] = [False, True]
        with self.assertRaises(ValueError):
            collect(data)
        data["cases"] = []
        with self.assertRaises(ValueError):
            collect(data)


if __name__ == "__main__":
    unittest.main()
