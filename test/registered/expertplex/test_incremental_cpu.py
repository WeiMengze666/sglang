"""Reject incomplete, forced-probability, or mismatched incremental evidence."""

import importlib
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
incremental = importlib.import_module("check_incremental")


class IncrementalCaptureTests(unittest.TestCase):
    def test_direct_pd_preserves_parameters_and_dispatches_concurrently(self):
        barrier = threading.Barrier(2, timeout=3)
        calls = []
        payload = {"sampling_params": {"custom_params": {"forced_ids": [1, 2]}}}

        def post(url, endpoint, body=None):
            calls.append((url, endpoint, body))
            if endpoint == "/generate":
                barrier.wait()
                return {"worker": url}

        with patch.object(incremental, "post", side_effect=post):
            result = incremental.generate(
                "http://localhost:2",
                payload,
                prefill_url="http://localhost:1",
                bootstrap_port=19000,
            )
        self.assertEqual(result, {"worker": "http://localhost:2"})
        requests = [body for _, endpoint, body in calls if endpoint == "/generate"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0]["sampling_params"], payload["sampling_params"])
        self.assertEqual(requests[0]["bootstrap_host"], "localhost")
        self.assertEqual(requests[0]["bootstrap_port"], 19000)
        self.assertNotIn("bootstrap_room", payload)

    def test_prefill_failure_is_not_hidden_by_successful_decode(self):
        def post(url, endpoint, body=None):
            if endpoint == "/generate" and url == "http://prefill":
                raise ValueError("prefill failed")
            return {"ok": True}

        with patch.object(incremental, "post", side_effect=post):
            with self.assertRaisesRegex(ValueError, "prefill failed"):
                incremental.generate(
                    "http://decode", {}, prefill_url="http://prefill", bootstrap_port=1
                )

    def records(self, directory):
        for step in range(2):
            record = {
                "step": step,
                "history": [0] + [1, 2][:step],
                "forced_token": [1, 2][step],
                "vocab_size": 3,
                "pre_forcing_logprobs": [[0, math.log(0.7)], [2, math.log(0.2)]],
                "pid": 10,
                "retraction_count": 0,
            }
            (directory / f"step-{step}-pid-10.json").write_text(json.dumps(record))

    def test_collect_preserves_unforced_subset_scores_at_each_history(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            outputs = incremental.collect(directory, [0], [1, 2], 3, 2)
        self.assertEqual([o["history"] for o in outputs], [[0], [0, 1]])
        self.assertIsNone(outputs[0]["logprobs"][1])
        self.assertEqual(outputs[1]["logprobs"][0], math.log(0.7))

    def test_missing_step_and_changed_history_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            path = directory / "step-1-pid-10.json"
            original = path.read_text()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                incremental.collect(directory, [0], [1, 2], 3, 2)
            record = json.loads(original)
            record["history"] = [0, 2]
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "history"):
                incremental.collect(directory, [0], [1, 2], 3, 2)

    def test_duplicate_tp_ranks_must_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            record = json.loads((directory / "step-1-pid-10.json").read_text())
            record["pid"] = 11
            other = directory / "step-1-pid-11.json"
            other.write_text(json.dumps(record))
            self.assertEqual(len(incremental.collect(directory, [0], [1, 2], 3, 2)), 2)
            record["pre_forcing_logprobs"][0][1] = math.log(0.6)
            other.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "ranks disagree"):
                incremental.collect(directory, [0], [1, 2], 3, 2)

    def test_invalid_scores_do_not_become_successful_evidence(self):
        for entries in [
            [[0, float("nan")], [2, -1]],
            [[0, -1], [0, -2]],
            [[-1, -1], [2, -2]],
            [[0, 0.1], [2, -2]],
        ]:
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.records(directory)
                path = directory / "step-0-pid-10.json"
                record = json.loads(path.read_text())
                record["pre_forcing_logprobs"] = entries
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "probability"):
                    incremental.collect(directory, [0], [1, 2], 3, 2)

    def test_short_reference_and_invalid_tokens_are_rejected(self):
        result = {
            "prompt": [0],
            "meta_info": {"output_token_logprobs": [[-1, 1], [-2, 2]]},
        }
        source = {"outputs": [result, result]}
        self.assertEqual(incremental.trajectories(source, 3, 2), [([0], [1, 2])])
        with self.assertRaisesRegex(ValueError, "shorter"):
            incremental.trajectories(source, 3, 3)
        result["prompt"] = [3]
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            incremental.trajectories(source, 3, 2)


if __name__ == "__main__":
    unittest.main()
