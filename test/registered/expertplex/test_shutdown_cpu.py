"""Signals during shutdown must neither orphan experts nor reenter Event locks."""

import importlib
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from _bootstrap import module  # noqa: E402

launch = importlib.import_module("launch")
baseline = importlib.import_module("launch_baseline")
worker = module("worker")


class ShutdownTests(unittest.TestCase):
    def test_repeated_signal_cannot_interrupt_baseline_cleanup(self):
        handlers = {}
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = lambda **_: handlers[signal.SIGTERM](
            signal.SIGTERM, None
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                sys,
                "argv",
                [
                    "launch",
                    "--variant",
                    "colocated",
                    "--model-path",
                    "/model",
                    "--log-dir",
                    str(Path(temp) / "service"),
                ],
            ),
            patch.dict(baseline.os.environ, {}, clear=True),
            patch.object(baseline.subprocess, "Popen", return_value=process),
            patch.object(baseline.signal, "signal", side_effect=handlers.__setitem__),
            patch.object(baseline.os, "killpg") as kill,
            patch.object(
                baseline, "wait_http", side_effect=RuntimeError("startup failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                baseline.main()
        process.wait.assert_called_once_with(timeout=15)
        kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_repeated_signal_cannot_interrupt_launcher_cleanup(self):
        handlers = {}
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = lambda **_: handlers[signal.SIGTERM](
            signal.SIGTERM, None
        )
        node = launch.config.default_config("/model")
        spec = SimpleNamespace(attention_weight_bytes=0, kv_bytes_per_token=1)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                sys, "argv", ["launch", "--config", "/node", "--log-dir", temp]
            ),
            patch.object(launch.config, "load_config", return_value=node),
            patch.object(launch.config.ModelSpec, "from_model_path", return_value=spec),
            patch.object(type(node), "validate"),
            patch.dict(launch.os.environ, {}, clear=True),
            patch.object(launch.subprocess, "Popen", return_value=process),
            patch.object(launch.signal, "signal", side_effect=handlers.__setitem__),
            patch.object(launch.os, "killpg") as kill,
            patch.object(
                launch,
                "wait_http",
                side_effect=RuntimeError("startup failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                launch.main()
        process.wait.assert_called_once_with(timeout=10)
        kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_supervisor_signal_does_not_acquire_event_lock(self):
        handlers, inside_wait = {}, [False]
        event = MagicMock()

        def wait(_):
            inside_wait[0] = True
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            inside_wait[0] = False
            return False

        def publish():
            self.assertFalse(inside_wait[0], "Event.set reentered the wait lock")

        event.wait.side_effect = wait
        event.set.side_effect = publish
        child = MagicMock(exitcode=None)
        child.is_alive.return_value = False
        context = MagicMock()
        context.Event.return_value = event
        context.Process.return_value = child
        with tempfile.TemporaryDirectory() as temp:
            node = MagicMock(
                model_path="/model",
                rendezvous_dir=temp,
                expert_devices=[0],
                startup_timeout_s=60,
            )
            with (
                patch.object(worker, "load_config", return_value=node),
                patch.object(worker.ModelSpec, "from_model_path"),
                patch.object(worker, "begin_run", return_value={"state": "ready"}),
                patch.object(
                    worker.multiprocessing, "get_context", return_value=context
                ),
                patch.object(worker.signal, "signal", side_effect=handlers.__setitem__),
            ):
                worker.supervise("/node")
        event.set.assert_called_once()
        child.start.assert_called_once()
        self.assertEqual(child.join.call_count, 2)


if __name__ == "__main__":
    unittest.main()
