"""The delayed router must retain its port while Mooncake starts P/D services."""

import errno
import importlib
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
launch = importlib.import_module("launch")


class RouterPortReservation(CustomTestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.port_socket = self.enterContext(socket.socket())
        self.port_socket.bind(("127.0.0.1", 0))
        self.port = self.port_socket.getsockname()[1]
        node = launch.config.default_config("/model", overlap="none")
        self.enterContext(patch.object(launch.config, "load_config", return_value=node))
        self.enterContext(patch.object(launch.config.NodeConfig, "validate"))
        self.enterContext(
            patch.object(
                launch.config.ModelSpec, "from_model_path", return_value=object()
            )
        )
        self.enterContext(
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        )
        self.enterContext(patch.object(launch.signal, "signal"))
        self.enterContext(
            patch.object(
                sys,
                "argv",
                [
                    "launch.py",
                    "--config",
                    "/node.json",
                    "--port",
                    str(self.port),
                    "--log-dir",
                    temporary,
                ],
            )
        )
        self.started = []
        self.health_checks = []

        def spawn(command, **kwargs):
            self.started.append(command)
            if "sglang_router.launch_router" in command:
                # The real router must be able to bind after the handoff.
                with socket.socket() as router:
                    router.bind(("127.0.0.1", self.port))
                raise KeyboardInterrupt
            return SimpleNamespace(pid=2**30, poll=lambda: None)

        self.enterContext(patch.object(launch.subprocess, "Popen", side_effect=spawn))

    def check_reserved(self, url, *args, **kwargs):
        self.health_checks.append(url)
        with socket.socket() as competitor:
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with self.assertRaises(OSError) as error:
                competitor.bind(("0.0.0.0", self.port))
            self.assertEqual(error.exception.errno, errno.EADDRINUSE)

    def test_startup_keeps_port_reserved_until_router_handoff(self):
        self.port_socket.close()
        with patch.object(launch, "wait_http", side_effect=self.check_reserved):
            launch.main()
        self.assertEqual(len(self.health_checks), 5)  # E and four attention services.
        self.assertIn("sglang_router.launch_router", self.started[-1])

    def test_failed_service_releases_router_reservation(self):
        self.port_socket.close()

        def failed_startup(*args, **kwargs):
            self.check_reserved(*args, **kwargs)
            raise RuntimeError("service startup failed")

        with patch.object(launch, "wait_http", side_effect=failed_startup):
            with self.assertRaisesRegex(RuntimeError, "service startup failed"):
                launch.main()
        with socket.socket() as replacement:
            replacement.bind(("127.0.0.1", self.port))
        self.assertEqual(len(self.started), 1)

    def test_busy_router_port_fails_before_starting_services(self):
        with (
            patch.object(
                launch, "wait_http", side_effect=AssertionError("services started")
            ),
            self.assertRaisesRegex(OSError, "Could not bind port"),
        ):
            launch.main()
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
