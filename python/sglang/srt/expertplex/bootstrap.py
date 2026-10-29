"""HTTP startup and lease control for independently launched attention servers.

No activation or per-layer synchronization passes through HTTP. CUDA IPC is
single-host: host/PID namespace and visibility are checked before exporting
handles. Cleanly drained clients can reconnect; crashed clients quarantine
their mailbox until E restarts, since their in-flight writes cannot be trusted.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import NodeConfig
from .rendezvous import (
    PROTOCOL_VERSION,
    checkpoint_fingerprint,
    config_digest,
    pid_alive,
)
from .synthetic_weights import weight_identity


def host_identity():
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"
        ),
    }


def request(addr, endpoint, body=None, timeout=5):
    parsed = urlparse(addr)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("ExpertPlex bootstrap address must be http://host:port")
    req = urllib.request.Request(
        addr.rstrip("/") + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_request(addr, endpoint, body, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ExpertPlex startup timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return request(
                addr,
                endpoint,
                body,
                timeout=min(5, max(0.01, deadline - time.monotonic())),
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise RuntimeError(
                    f"ExpertPlex bootstrap rejected request: {exc.read().decode()}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"ExpertPlex bootstrap did not become ready: {addr}{endpoint}"
            )
        time.sleep(0.1)


def fetch_config(addr, timeout):
    data = wait_request(addr, "/bootstrap", None, timeout)
    if data["host"] != host_identity():
        raise ValueError(
            "ExpertPlex CUDA IPC requires the same host, PID namespace and CUDA visibility"
        )
    node, manifest = NodeConfig.from_dict(data["config"]), data["manifest"]
    if manifest["protocol"] != PROTOCOL_VERSION or manifest[
        "config_digest"
    ] != config_digest(node):
        raise ValueError("ExpertPlex bootstrap protocol/configuration mismatch")
    return node, manifest


def register(addr, node, manifest, client_id):
    return wait_request(
        addr,
        "/register",
        {
            "host": host_identity(),
            "run_id": manifest["run_id"],
            "config_digest": config_digest(node),
            "client_id": client_id,
            "checkpoint_fingerprint": checkpoint_fingerprint(node.model_path),
            "weight_identity": weight_identity(node),
            "pid": os.getpid(),
            "lease": uuid.uuid4().hex,
        },
        node.startup_timeout_s,
    )


class BootstrapServer:
    def __init__(self, node, manifest, host, port):
        self.node, self.manifest = node, manifest
        self.host = host_identity()
        self.leases = {}
        self.lock = threading.RLock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.handle_request(None)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16384:
                        raise ValueError("Invalid request length")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise TypeError("Expected a JSON object")
                    self.handle_request(body)
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply(400, {"error": str(exc)})

            def handle_request(self, body):
                with owner.lock:
                    try:
                        status, result = owner.route(self.path, body)
                    except (ValueError, KeyError, TypeError) as exc:
                        status, result = 400, {"error": str(exc)}
                self.reply(status, result)

            def reply(self, status, result):
                payload = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def peers(self):
        peers = []
        if self.manifest["state"] != "ready":
            return peers
        for rank, device in enumerate(self.node.expert_devices):
            try:
                peer = json.loads(
                    (Path(self.node.rendezvous_dir) / f"expert-{rank}.json").read_text()
                )
            except (FileNotFoundError, json.JSONDecodeError):
                return []
            if (
                peer["run_id"] != self.manifest["run_id"]
                or peer["rank"] != rank
                or peer["device"] != device
                or not pid_alive(peer["pid"])
            ):
                return []
            peers.append(peer)
        return peers

    def route(self, path, body):
        if path == "/bootstrap" and body is None:
            if self.manifest["state"] in ("failed", "stopped"):
                return 503, {"state": self.manifest["state"]}
            return 200, {
                "config": self.node.to_dict(),
                "manifest": dict(self.manifest),
                "host": self.host,
            }
        if path in ("/health", "/health_generate", "/get_server_info") and body is None:
            ready = bool(self.peers())
            return (200 if ready else 503), {
                "role": "expert",
                "ready": ready,
                "run_id": self.manifest["run_id"],
                "expert_count": len(self.node.expert_devices),
                "clients": [
                    {"id": i, "pid": lease["pid"], "alive": pid_alive(lease["pid"])}
                    for i, lease in self.leases.items()
                ],
            }
        if path not in ("/register", "/release") or body is None:
            return 404, {"error": "Unknown endpoint"}
        if body.get("run_id") != self.manifest["run_id"]:
            return 409, {
                "error": "Stale MoE server generation; restart this attention server"
            }
        client_id = body["client_id"]
        if isinstance(client_id, bool) or not isinstance(client_id, int):
            raise TypeError("client_id must be an integer")
        self.node.client(client_id)
        if path == "/release":
            lease = self.leases.get(client_id)
            if lease is None or lease["lease"] != body.get("lease"):
                return 409, {"error": "Lease ownership mismatch"}
            if body.get("drained") is not True:
                return 409, {
                    "error": "Undrained mailbox is quarantined until E restarts"
                }
            del self.leases[client_id]
            return 200, {"released": True}
        peers = self.peers()
        if not peers:
            return 503, {"error": "Experts are not ready"}
        for field, expected in (
            ("host", self.host),
            ("config_digest", config_digest(self.node)),
            ("checkpoint_fingerprint", self.manifest["checkpoint_fingerprint"]),
            ("weight_identity", self.manifest["weight_identity"]),
        ):
            if body.get(field) != expected:
                return 409, {"error": f"Mismatched {field}"}
        if (
            isinstance(body.get("pid"), bool)
            or not isinstance(body.get("pid"), int)
            or body["pid"] <= 0
            or not pid_alive(body["pid"])
        ):
            raise ValueError("Client process is not alive")
        if not isinstance(body.get("lease"), str) or not body["lease"]:
            raise ValueError("Missing lease ID")
        old = self.leases.get(client_id)
        if old is not None and (old["lease"], old["pid"]) != (
            body["lease"],
            body["pid"],
        ):
            return 409, {"error": "Attention slot is occupied or quarantined"}
        self.leases[client_id] = {"lease": body["lease"], "pid": body["pid"]}
        return 200, {
            "manifest": dict(self.manifest),
            "peers": peers,
            "lease": body["lease"],
        }

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
