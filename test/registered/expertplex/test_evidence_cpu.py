"""Numerical publication exports must omit raw request and log payloads."""

import importlib.util
import json
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "examples/expertplex/export_evidence.py"
SPEC = importlib.util.spec_from_file_location("numerical_evidence", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvidenceExportTests(unittest.TestCase):
    def test_only_whitelisted_numerical_metrics_survive(self):
        raw = {
            "status": "passed",
            "error": "private log",
            "input_ids": [100, 200],
            "outputs": [{"text": "private response"}],
            "cases": [
                {
                    "passed": True,
                    "relative_l2_error": 0.01,
                    "prompt": "private request",
                    "logprobs": [1, 2],
                }
            ],
            "settings": {"hidden": 6144, "model_path": "/private/model", "seed": 42},
            "environment": {"SECRET": "private credential"},
        }
        exported = MODULE.redact(raw)
        self.assertEqual(
            exported,
            {
                "status": "passed",
                "cases": [{"passed": True, "relative_l2_error": 0.01}],
                "settings": {"hidden": 6144, "seed": 42},
            },
        )
        self.assertNotIn("private", json.dumps(exported))
        self.assertNotIn("logprobs", json.dumps(exported))

    def test_joint_slo_metrics_survive_without_private_detail_paths(self):
        raw = {
            "p90_goodput_max_tested_arrival_rps": None,
            "ttft_slo_ms": 10000,
            "results": [
                {
                    "request_rate": 0.1,
                    "attempted_requests": 32,
                    "joint_slo_met_requests": 22,
                    "meets_p90_joint_slo": False,
                    "details_file": "/private/request-data.jsonl",
                }
            ],
        }
        expected = dict(raw)
        expected["results"] = [
            {k: v for k, v in raw["results"][0].items() if k != "details_file"}
        ]
        self.assertEqual(MODULE.redact(raw), expected)


if __name__ == "__main__":
    unittest.main()
