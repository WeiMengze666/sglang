"""Do not fit traced, incomplete, or repeated concurrent data as isolated latency."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
collect = importlib.import_module("profile_full_moe").collect_samples


class FullMoEProfileTests(unittest.TestCase):
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
