"""Exercise real HTTP bootstrap, exclusive leases and generation fencing."""

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from _bootstrap import module

bootstrap = module("bootstrap")
rv = module("rendezvous")
config = module("config")


class BootstrapTests(unittest.TestCase):
    def test_counts_can_use_a_subset_of_the_node_without_aliasing_devices(self):
        node = config.default_config(
            "/model", prefill_count=1, decode_count=1, expert_count=2
        )
        self.assertEqual([c.device for c in node.clients], [0, 1])
        self.assertEqual(node.expert_devices, (2, 3))
        with self.assertRaisesRegex(ValueError, "not exceed"):
            config.default_config(
                "/model", prefill_count=3, decode_count=3, expert_count=4
            )

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "config.json").write_text("{}")
        self.node = config.default_config(str(root), rendezvous_dir=str(root / "ipc"))
        self.manifest = rv.begin_run(self.node)
        self.host = {
            "boot_id": "test-node",
            "pid_namespace": "pid:[1]",
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        }
        mock = patch.object(bootstrap, "host_identity", return_value=self.host)
        mock.start()
        self.addCleanup(mock.stop)
        self.server = bootstrap.BootstrapServer(
            self.node, self.manifest, "127.0.0.1", 0
        )
        self.addCleanup(self.server.close)
        self.address = f"http://127.0.0.1:{self.server.server.server_port}"
        for rank, device in enumerate(self.node.expert_devices):
            rv.atomic_json(
                root / "ipc" / f"expert-{rank}.json",
                {
                    "run_id": self.manifest["run_id"],
                    "rank": rank,
                    "device": device,
                    "pid": os.getpid(),
                    "buffers": {},
                },
            )

    def ready(self):
        self.manifest["state"] = "ready"

    def registration(self, client_id=0):
        return {
            "host": self.host,
            "run_id": self.manifest["run_id"],
            "config_digest": rv.config_digest(self.node),
            "client_id": client_id,
            "checkpoint_fingerprint": self.manifest["checkpoint_fingerprint"],
            "weight_identity": self.manifest["weight_identity"],
            "pid": os.getpid(),
            "lease": uuid.uuid4().hex,
        }

    def test_discovery_before_workers_ready_without_claiming_mailboxes(self):
        node, manifest = bootstrap.fetch_config(self.address, 0.5)
        self.assertEqual(node, self.node)
        self.assertEqual(manifest["state"], "starting")
        self.assertEqual(self.server.leases, {})
        self.assertEqual(self.server.route("/health", None)[0], 503)
        self.assertEqual(self.server.route("/register", self.registration())[0], 503)

    def test_exclusive_registration_retry_and_clean_reconnect(self):
        self.ready()
        body = self.registration()
        first = bootstrap.request(self.address, "/register", body)
        retry = bootstrap.request(self.address, "/register", body)
        self.assertEqual(first, retry)
        self.assertEqual(len(first["peers"]), 4)
        self.assertEqual(self.server.route("/register", self.registration())[0], 409)
        self.assertEqual(
            self.server.route("/release", body | {"drained": False})[0], 409
        )
        bootstrap.request(self.address, "/release", body | {"drained": True})
        self.assertEqual(self.server.route("/register", self.registration())[0], 200)

    def test_generation_checkpoint_host_and_weight_identity_must_match(self):
        self.ready()
        for field in (
            "run_id",
            "host",
            "config_digest",
            "checkpoint_fingerprint",
            "weight_identity",
        ):
            with self.subTest(field=field):
                body = self.registration() | {field: "wrong"}
                self.assertEqual(self.server.route("/register", body)[0], 409)
        self.assertFalse(self.server.leases)

    def test_release_requires_current_generation_and_lease(self):
        self.ready()
        body = self.registration()
        bootstrap.request(self.address, "/register", body)
        for bad in ({"run_id": "old"}, {"lease": "wrong"}):
            self.assertEqual(
                self.server.route("/release", body | {"drained": True} | bad)[0], 409
            )
        self.assertEqual(len(self.server.leases), 1)

    def test_bootstrap_timeout_is_bounded(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(ValueError, "finite and positive"),
            ):
                bootstrap.wait_request(self.address, "/bootstrap", None, timeout)

    def test_dead_client_is_not_automatically_reused(self):
        self.ready()
        body = self.registration()
        self.assertEqual(self.server.route("/register", body)[0], 200)
        with patch.object(bootstrap, "pid_alive", return_value=False):
            self.assertEqual(self.server.route("/health", None)[0], 503)
        self.assertEqual(self.server.route("/register", self.registration())[0], 409)

    def test_independent_prefill_and_decode_ownership(self):
        self.ready()
        registrations = [self.registration(i) for i in range(4)]
        for body in registrations:
            bootstrap.request(self.address, "/register", body)
        bootstrap.request(
            self.address, "/release", registrations[1] | {"drained": True}
        )
        info = bootstrap.request(self.address, "/get_server_info")
        self.assertTrue(info["ready"])
        self.assertEqual({c["id"] for c in info["clients"]}, {0, 2, 3})
        self.assertEqual(self.server.route("/register", self.registration(1))[0], 200)


if __name__ == "__main__":
    unittest.main()
