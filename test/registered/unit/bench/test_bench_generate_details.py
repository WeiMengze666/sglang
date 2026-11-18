import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _GenerateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        prompt = json.loads(body)["text"]
        if prompt == "failed":
            self.send_error(503)
            return
        self.send_response(200)
        self.end_headers()
        for i in (1, 2):
            meta = {"completion_tokens": i}
            if prompt == "ranked":
                meta["dp_rank"] = 2 if i == 1 else 0
            data = json.dumps({"text": "x" * i, "meta_info": meta})
            self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.write(b'data: {"meta_info":{}}\n\ndata: [DONE]\n\n')


class TestBenchGenerateDetails(CustomTestCase):
    def test_request_details_preserve_rank_zero_and_failed_rows(self):
        """Join the last reported rank and client send time to each latency row."""
        rows = [
            DatasetRow(prompt=prompt, prompt_len=1, output_len=2)
            for prompt in ("ranked", "failed", "unreported")
        ]
        with (
            ThreadingHTTPServer(("127.0.0.1", 0), _GenerateHandler) as server,
            tempfile.TemporaryDirectory() as tmp,
        ):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            path = Path(tmp) / "result.jsonl"
            argv = [
                "bench",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "test",
                "--tokenizer",
                "test",
                "--warmup-requests",
                "0",
                "--ready-check-timeout-sec",
                "0",
                "--max-concurrency",
                "1",
                "--output-details",
                "--disable-tqdm",
                "--output-file",
                str(path),
            ]
            tokenizer = Mock(encode=lambda text, **kwargs: list(text))
            try:
                with (
                    patch.object(sys, "argv", argv),
                    patch.dict(os.environ, {"SGLANG_IS_IN_CI": "0"}),
                    patch.object(serving, "args", None, create=True),
                    patch.object(serving, "get_tokenizer", return_value=tokenizer),
                    patch.object(serving, "get_dataset", return_value=rows),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    serving.cli_main()
            finally:
                server.shutdown()
                thread.join()
            details = json.loads(path.read_text())
        self.assertEqual(details["successes"], [True, False, True])
        self.assertEqual(details["dp_ranks"], [0, None, None])
        starts = details["request_start_offsets_s"]
        self.assertEqual(len(starts), len(rows))
        self.assertTrue(0 <= starts[0] < starts[1] < starts[2] <= details["duration"])


if __name__ == "__main__":
    unittest.main()
