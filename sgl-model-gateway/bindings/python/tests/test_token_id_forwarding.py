"""CPU regression for token diagnostics through the real Rust HTTP router.

Only the model workers are simulated. The installed router parses, validates,
normalizes and forwards actual HTTP requests in regular and PD modes.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.parametrize("pd", [False, True], ids=["regular", "pd"])
def test_chat_token_ids_reach_workers(tmp_path, pd):
    captured = []

    class Worker(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                {
                    "model_path": "token-probe",
                    "is_generation": True,
                    "disaggregation_mode": self.server.role,
                    "dp_size": 1,
                }
            )

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            captured.append((self.server.role, body))
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
            if body.get("return_prompt_token_ids") or body.get("return_token_ids"):
                choice["prompt_token_ids"] = [11, 12]
            if body.get("return_token_ids"):
                choice["response_token_ids"] = [13]
            self.reply(
                {
                    "id": "probe",
                    "object": "chat.completion",
                    "model": "token-probe",
                    "choices": [choice],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            )

    workers = []
    process = None
    try:
        for role in ["prefill", "decode"] if pd else ["null"]:
            worker = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
            worker.role = role
            threading.Thread(target=worker.serve_forever, daemon=True).start()
            workers.append(worker)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        urls = [f"http://127.0.0.1:{worker.server_port}" for worker in workers]
        if pd:
            command += [
                "--pd-disaggregation",
                "--prefill",
                urls[0],
                "--decode",
                urls[1],
            ]
        else:
            command += ["--worker-urls", urls[0]]
        with (tmp_path / "router.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "NVIDIA_VISIBLE_DEVICES": "void",
                },
            )
            deadline = time.monotonic() + 120
            while True:
                assert process.poll() is None, (tmp_path / "router.log").read_text()
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2
                    ):
                        break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, (
                        tmp_path / "router.log"
                    ).read_text()
                    time.sleep(0.1)
            for prompt, output in [
                (True, False),
                (False, True),
                (True, True),
                (False, False),
            ]:
                body = {
                    "model": "token-probe",
                    "messages": [{"role": "user", "content": "hello"}],
                    "return_prompt_token_ids": prompt,
                    "return_token_ids": output,
                    "max_tokens": 1,
                    "stream": False,
                }
                start = len(captured)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    choice = json.load(response)["choices"][0]
                forwarded = captured[start:]
                assert sorted(role for role, _ in forwarded) == sorted(
                    w.role for w in workers
                )
                for _, forwarded_body in forwarded:
                    assert (
                        forwarded_body.get("return_prompt_token_ids", False) == prompt
                    )
                    assert forwarded_body.get("return_token_ids", False) == output
                assert choice.get("prompt_token_ids") == (
                    [11, 12] if prompt or output else None
                )
                assert choice.get("response_token_ids") == ([13] if output else None)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for worker in workers:
            worker.shutdown()
            worker.server_close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
