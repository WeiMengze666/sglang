import copy
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
validate = importlib.import_module("plot_components").validate_sharing_reports


class SharingComparisonTests(unittest.TestCase):
    def fixture(self):
        settings = dict(
            hidden=512,
            intermediate=256,
            experts=4,
            world=4,
            top_k=8,
            prefill_rows=128,
            decode_rows=16,
            delay_us=10,
            operation="moe",
            iterations=3,
        )
        settings.update(modes=["exclusive"], green_divisions=[])
        row = dict(response_upper_bound_us=[20, 10], finite_outputs=[True, True])
        native = dict(
            workload_identity="fixture",
            settings=settings,
            expert_rows=[[64] * 4, [8] * 4],
            cases=[copy.deepcopy(row)],
            isolated_response_upper_bound_us=[15, 8],
        )
        apk = copy.deepcopy(native)
        apk["settings"]["budgets"] = [32]
        apk["cases"][0].update(
            decode_budget=32, native_reference=[{"passed": True}] * 2
        )
        mps = copy.deepcopy(native)
        mps["settings"]["percentages"] = ["50:50"]
        mps["cases"][0].update(
            requested_percentages=[50, 50],
            clients=[{"expert_rows": r} for r in native["expert_rows"]],
        )
        return native, apk, mps

    def test_matching_actual_dimensions(self):
        settings = validate(*self.fixture())
        self.assertEqual(settings["prefill_rows"], 128)

    def test_rejects_profiler_instrumented_timings(self):
        reports = self.fixture()
        reports[1]["settings"]["profile_range"] = True
        with self.assertRaisesRegex(ValueError, "Profiler-instrumented"):
            validate(*reports)

    def test_rejects_gpu_burst_even_without_profiler(self):
        reports = self.fixture()
        reports[1]["settings"]["gpu_burst"] = True
        with self.assertRaisesRegex(ValueError, "GPU-burst"):
            validate(*reports)

    def test_rejects_same_expert_counts_but_different_workload(self):
        for field, value in [
            ("hidden", 1024),
            ("delay_us", 20),
            ("decode_expert_rows", 2),
        ]:
            reports = self.fixture()
            reports[1]["settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(*reports)

    def test_rejects_partial_failed_and_nonfinite_measurements(self):
        for variant in (
            "partial_native",
            "partial_apk",
            "partial_mps",
            "failed",
            "nonfinite",
        ):
            reports = self.fixture()
            if variant == "partial_native":
                reports[0]["settings"]["modes"].append("priority")
            elif variant == "partial_apk":
                reports[1]["settings"]["budgets"].append(64)
            elif variant == "partial_mps":
                reports[2]["settings"]["percentages"].append("75:25")
            elif variant == "failed":
                reports[1]["cases"][0]["native_reference"][0] = {"passed": False}
            else:
                reports[0]["isolated_response_upper_bound_us"][0] = float("nan")
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate(*reports)


if __name__ == "__main__":
    unittest.main()
