"""Reject incomplete, forced-probability, or mismatched incremental evidence."""

import contextlib
import importlib
import io
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
    def test_binary_full_vocab_is_lossless_and_rejects_corruption(self):
        import hashlib
        import struct

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = struct.pack("<4f", -0.0, -1e-45, -0.6931471805599453, -100.25)
            path = root / "step-000000-pid-1.f32"
            path.write_bytes(data)
            manifest = {
                "file": path.name,
                "format": "float32-le",
                "count": 4,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            record = {"pre_forcing_scores": manifest}
            entries = incremental.captured_score_entries(record, root, 4, 0)
            self.assertEqual([i for i, _ in entries], list(range(4)))
            self.assertEqual(struct.pack("<4f", *(v for _, v in entries)), data)
            for patch_manifest in (
                {"file": "../escape.f32"},
                {"count": 3},
                {"count": True},
                {"format": "float16-le"},
                {"sha256": "0" * 64},
            ):
                with self.subTest(patch=patch_manifest), self.assertRaises(ValueError):
                    incremental.captured_score_entries(
                        {"pre_forcing_scores": manifest | patch_manifest}, root, 4, 0
                    )
            for invalid in ({}, record | {"pre_forcing_logprobs": entries}):
                with self.assertRaises(ValueError):
                    incremental.captured_score_entries(invalid, root, 4, 0)
            with self.assertRaises(ValueError):
                incremental.captured_score_entries(record, root, 4, 2)
            for corrupt in (data[:-1], bytes([data[0] ^ 1]) + data[1:]):
                path.write_bytes(corrupt)
                with self.assertRaises(ValueError):
                    incremental.captured_score_entries(record, root, 4, 0)

    def test_binary_capture_preserves_full_trajectory_and_probability_checks(self):
        import hashlib
        import struct

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for step in range(2):
                data = struct.pack("<3f", -1.0, -2.0, -3.0)
                name = f"step-{step:06d}-pid-1.f32"
                (root / name).write_bytes(data)
                record = {
                    "step": step,
                    "history": [0] + [1, 2][:step],
                    "forced_token": [1, 2][step],
                    "vocab_size": 3,
                    "retraction_count": 0,
                    "sampling_batch_size": 2,
                    "pre_forcing_scores": {
                        "file": name,
                        "format": "float32-le",
                        "count": 3,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    },
                }
                (root / name).with_suffix(".json").write_text(json.dumps(record))
            outputs = incremental.collect(root, [0], [1, 2], 3, 0)
            self.assertEqual([o["logprobs"] for o in outputs], [[-1.0, -2.0, -3.0]] * 2)
            self.assertEqual(incremental.decode_batch_summary(outputs, 2)["maximum"], 2)
            for bad in (float("nan"), float("inf"), 0.1):
                data = struct.pack("<3f", -1.0, bad, -3.0)
                name = "step-000001-pid-1.f32"
                (root / name).write_bytes(data)
                path = (root / name).with_suffix(".json")
                record = json.loads(path.read_text())
                record["pre_forcing_scores"]["sha256"] = hashlib.sha256(
                    data
                ).hexdigest()
                path.write_text(json.dumps(record))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    incremental.collect(root, [0], [1, 2], 3, 0)

    def test_native_batch_routing_is_scalar_and_does_not_modify_payload(self):
        payload = {"input_ids": [[11, 12], [13, 14]]}
        with patch.object(incremental, "post", return_value=[]) as post:
            incremental.generate(
                "http://native", payload, flush=False, routed_dp_rank=0
            )
        post.assert_called_once_with(
            "http://native", "/generate", dict(payload, routed_dp_rank=0)
        )
        self.assertNotIn("routed_dp_rank", payload)

    def test_native_routing_rejects_pd_and_invalid_rank_before_requests(self):
        with patch.object(incremental, "post") as post:
            for rank in (-1, True, 0.5):
                with self.subTest(rank=rank), self.assertRaises(ValueError):
                    incremental.generate("http://native", {}, routed_dp_rank=rank)
            with self.assertRaises(ValueError):
                incremental.generate(
                    "http://decode", {}, prefill_url="http://prefill", routed_dp_rank=0
                )
        post.assert_not_called()

    def test_concurrent_capture_flushes_once_and_keeps_reference_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "config.json").write_text('{"vocab_size": 3}')
            histories = root / "histories.json"
            histories.write_text(
                json.dumps(
                    {
                        "model_path": str(root),
                        "outputs": [
                            {
                                "prompt": [i],
                                "meta_info": {
                                    "output_token_logprobs": [[0, 1], [0, 2]]
                                },
                            }
                            for i in (0, 1)
                        ],
                    }
                )
            )
            barrier = threading.Barrier(2, timeout=3)

            def generate(url, payload, **kwargs):
                self.assertFalse(kwargs["flush"])
                barrier.wait()
                return {"meta_info": {"output_token_logprobs": [[0, 1], [0, 2]]}}

            def collect(directory, prompt, forced, vocab, top_k):
                return [
                    {
                        "history": prompt + forced[:step],
                        "logprobs": [-1.0, -2.0, -3.0],
                        "sampling_batch_size": 2,
                    }
                    for step in range(2)
                ]

            argv = [
                "check_incremental.py",
                "--base-url",
                "http://d",
                "--prefill-url",
                "http://p",
                "--bootstrap-port",
                "19000",
                "--model-path",
                str(root),
                "--histories",
                str(histories),
                "--capture-dir",
                str(root / "capture"),
                "--output",
                str(root / "out.json"),
                "--steps",
                "2",
                "--top-k",
                "0",
                "--concurrency",
                "2",
                "--require-decode-batch-size",
                "2",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(incremental, "make_processor", return_value="fixture"),
                patch.object(incremental, "post") as post,
                patch.object(incremental, "generate", side_effect=generate),
                patch.object(incremental, "collect", side_effect=collect),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                incremental.main()
            self.assertEqual(
                post.call_args_list,
                [
                    unittest.mock.call("http://p", "/flush_cache"),
                    unittest.mock.call("http://d", "/flush_cache"),
                ],
            )
            saved = json.loads((root / "out.json").read_text())
            self.assertEqual([row["case"] for row in saved["cases"]], [0, 1])
            self.assertEqual(
                [row["history"] for row in saved["outputs"]], [[0], [0, 1], [1], [1, 1]]
            )
            reference = root / "reference.json"
            reference.write_text(json.dumps(saved))
            argv[argv.index("--capture-dir") + 1] = str(root / "unbatched")
            argv.extend(["--reference", str(reference), "--require-exact-logprobs"])

            def unbatched(*args):
                rows = collect(*args)
                if args[1] == [0]:
                    for row in rows:
                        row["sampling_batch_size"] = 1
                return rows

            with (
                patch.object(sys, "argv", argv),
                patch.object(incremental, "make_processor", return_value="fixture"),
                patch.object(incremental, "post"),
                patch.object(incremental, "generate", side_effect=generate),
                patch.object(incremental, "collect", side_effect=unbatched),
                self.assertRaisesRegex(ValueError, "required sampling batch size"),
            ):
                incremental.main()
            saved = json.loads((root / "out.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["decode_batch_failures"], [0])
            self.assertEqual(len(saved["outputs"]), 4)
            self.assertEqual(len(saved["comparisons"]), 4)
            self.assertTrue(saved["all_full_vocab_logprobs_exact"])

            argv.append("--batched-request")
            argv[argv.index("--capture-dir") + 1] = str(root / "native-batch")

            def generate_batch(url, payload, **kwargs):
                self.assertEqual(payload["input_ids"], [[0], [1]])
                self.assertEqual(payload["custom_logit_processor"], ["fixture"] * 2)
                self.assertEqual(
                    [
                        p["custom_params"]["prompt_ids"]
                        for p in payload["sampling_params"]
                    ],
                    [[0], [1]],
                )
                return [{"meta_info": {"output_token_logprobs": [[0, 1], [0, 2]]}}] * 2

            with (
                patch.object(sys, "argv", argv),
                patch.object(incremental, "make_processor", return_value="fixture"),
                patch.object(incremental, "post"),
                patch.object(
                    incremental, "generate", side_effect=generate_batch
                ) as send,
                patch.object(incremental, "collect", side_effect=collect),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                incremental.main()
            send.assert_called_once()
            saved = json.loads((root / "out.json").read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertTrue(saved["all_full_vocab_logprobs_exact"])
            self.assertTrue(saved["batched_request"])

    def test_native_batch_pd_uses_same_distinct_rooms_on_both_services(self):
        payload = incremental.batch_payload(
            [
                {"input_ids": [i], "sampling_params": {"custom_params": {"case": i}}}
                for i in (1, 2)
            ]
        )
        with (
            patch.object(incremental, "post", return_value=[{}, {}]) as post,
            patch.object(incremental.secrets, "randbits", side_effect=[101, 102]),
        ):
            incremental.generate(
                "http://d",
                payload,
                prefill_url="http://p",
                bootstrap_port=19000,
                flush=False,
            )
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.args[2]["bootstrap_room"], [101, 102])
            self.assertEqual(
                call.args[2]["sampling_params"], payload["sampling_params"]
            )

    def test_decode_batch_guard_does_not_accept_prefill_only_or_missing_metadata(self):
        for rows in (
            [{"sampling_batch_size": 16}, {"sampling_batch_size": 1}],
            [{}, {}],
        ):
            with self.assertRaisesRegex(ValueError, "No decode step"):
                incremental.decode_batch_summary(rows, 2)
        self.assertEqual(
            incremental.decode_batch_summary(
                [{"sampling_batch_size": 1}, {"sampling_batch_size": 2}], 2
            ),
            {"minimum": 2, "maximum": 2},
        )

    def test_strict_cli_rejects_tail_difference_and_partial_reference(self):
        # Identical top-1 tokens are insufficient: a changed tail score or an
        # unscored tail must fail the CLI and leave a failed report on disk.
        for tail in (-3.0, math.nextafter(-3.0, 0.0), None):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                (root / "config.json").write_text('{"vocab_size": 3}')
                histories = root / "histories.json"
                histories.write_text(
                    json.dumps(
                        {
                            "model_path": str(root),
                            "outputs": [
                                {
                                    "prompt": [0],
                                    "meta_info": {
                                        "output_token_logprobs": [[-0.1, 1], [-0.1, 2]]
                                    },
                                }
                            ],
                        }
                    )
                )
                outputs = [
                    {"history": [0], "logprobs": [-0.1, -2.0, -3.0]},
                    {"history": [0, 1], "logprobs": [-0.1, -2.0, -3.0]},
                ]
                reference = root / "reference.json"
                reference.write_text(
                    json.dumps(
                        {
                            "model_path": str(root),
                            "vocab_size": 3,
                            "outputs": [
                                outputs[0],
                                dict(outputs[1], logprobs=[-0.1, -2.0, tail]),
                            ],
                        }
                    )
                )
                output = root / "report.json"
                argv = [
                    "check_incremental.py",
                    "--base-url",
                    "http://unused",
                    "--model-path",
                    str(root),
                    "--histories",
                    str(histories),
                    "--reference",
                    str(reference),
                    "--capture-dir",
                    str(root / "capture"),
                    "--output",
                    str(output),
                    "--steps",
                    "2",
                    "--top-k",
                    "0",
                    "--require-exact-logprobs",
                ]
                result = {"meta_info": {"output_token_logprobs": [[0, 1], [0, 2]]}}
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(incremental, "make_processor", return_value="fixture"),
                    patch.object(incremental, "generate", return_value=result),
                    patch.object(incremental, "collect", return_value=outputs),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if tail == -3.0:
                        incremental.main()
                    else:
                        with self.assertRaisesRegex(
                            AssertionError, "not exactly equal"
                        ):
                            incremental.main()
                saved = json.loads(output.read_text())
                self.assertEqual(
                    saved["status"], "completed" if tail == -3.0 else "failed"
                )
                self.assertEqual(saved["all_full_vocab_logprobs_exact"], tail == -3.0)

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

        with (
            patch.object(incremental, "post", side_effect=post),
            self.assertRaisesRegex(ValueError, "prefill failed"),
        ):
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
