"""Reject false concurrent acceptance and premature client readiness."""

import copy
import importlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
check = importlib.import_module("check_all_decode_alignment")


def source_report(base=41000):
    common = [
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--model-path",
        "checkpoint",
    ]
    e = common + [
        "--expertplex-role",
        "expert",
        "--expertplex-prefill-count",
        "1",
        "--expertplex-decode-count",
        "3",
        "--expertplex-expert-count",
        "4",
        "--port",
        str(base + 200),
    ]
    attention = []
    for gpu, role in ((0, "prefill"), (1, "decode")):
        attention.append(
            common
            + [
                "--disaggregation-mode",
                role,
                "--base-gpu-id",
                str(gpu),
                "--port",
                str(base + gpu),
                "--nccl-port",
                str(base + 1000 + gpu),
                "--disaggregation-bootstrap-port",
                str(base + 2000 + gpu),
                "--enable-two-batch-overlap",
                "--enable-deterministic-inference",
                "--disable-overlap-schedule",
            ]
        )
    capture = [
        "python",
        "/old/checkout/check_incremental.py",
        "--top-k",
        "0",
        "--concurrency",
        "16",
        "--steps",
        "16",
        "--require-decode-batch-size",
        "2",
        "--require-exact-logprobs",
        "--reference",
        "native/scores.json",
        "--batched-request",
        "--prefill-url",
        f"http://127.0.0.1:{base}",
        "--base-url",
        f"http://127.0.0.1:{base + 1}",
    ]
    for command in attention:
        command += ["--expertplex-bootstrap-addr", f"http://127.0.0.1:{base + 200}"]
    capture += ["--model-path", "checkpoint", "--bootstrap-port", str(base + 2000)]
    return dict(
        status="passed",
        variant="expertplex",
        expert_count=4,
        concurrency=16,
        deterministic_inference=True,
        batched_request=True,
        tbo_enabled=True,
        required_tbo_replays=2,
        steps=16,
        cleanup_errors=[],
        commands=[e, *attention, capture],
    )


def progress():
    intervals = [
        dict(start_ns=10 + i, end_ns=30 + i, pids=[100 + i], clients=[i + 1])
        for i in range(3)
    ]
    proofs = [[dict(capture_ns=t) for t in (13, 20, 29)] for _ in range(3)]
    return intervals, proofs


class AllDecodeTests(unittest.TestCase):
    def test_cleanup_failure_cannot_publish_passed_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "report.json").write_text('{"status":"running"}')
            result = {"status": "passed", "results": [{"returncode": 0}]}
            with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
                check.finish_report(result, out, [12345])
            saved = json.loads((out / "report.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["remaining_owned_pids"], [12345])
            self.assertEqual(saved["results"], [{"returncode": 0}])
            self.assertFalse((out / "report.json.tmp").exists())

    def test_cleanup_failure_preserves_original_validation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            result = {"status": "failed", "error": "Strict capture failed"}
            with self.assertRaises(RuntimeError):
                check.finish_report(result, Path(temp), [12345])
            saved = json.loads((Path(temp) / "report.json").read_text())
            self.assertEqual(saved["error"], "Strict capture failed")
            self.assertEqual(saved["cleanup_error"], "Own process cleanup incomplete")
            self.assertEqual(saved["status"], "failed")

    def test_clean_teardown_preserves_validation_outcome(self):
        for status in ("passed", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                check.finish_report({"status": status}, Path(temp), [])
                saved = json.loads((Path(temp) / "report.json").read_text())
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["remaining_owned_pids"], [])
                self.assertNotIn("cleanup_error", saved)

    def test_plan_preserves_source_and_derives_nondefault_ports(self):
        source = source_report()
        saved = copy.deepcopy(source)
        commands, urls, template = check.prepare_commands(source)
        self.assertEqual(source, saved)
        self.assertEqual(len(commands), 5)
        for device, (_, command) in enumerate(commands[1:]):
            self.assertEqual(check.option(command, "--base-gpu-id"), str(device))
            self.assertEqual(urls[device], f"http://127.0.0.1:{41000 + device}")
            self.assertIn("--enable-two-batch-overlap", command)
            self.assertIn("--disable-overlap-schedule", command)
            self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            Path(template[1]), ROOT / "examples/expertplex/check_incremental.py"
        )
        self.assertIn("--require-exact-logprobs", template)

    def test_rejects_weakened_source_and_capture_gates(self):
        for key, value in (
            ("status", "running"),
            ("variant", "native"),
            ("tbo_enabled", False),
            ("required_tbo_replays", 1),
            ("cleanup_errors", ["alive"]),
            ("concurrency", 1),
        ):
            source = source_report()
            source[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.prepare_commands(source)
        source = source_report()
        check.replace(source["commands"][-1], "--top-k", 128)
        with self.assertRaises(ValueError):
            check.prepare_commands(source)
        source = source_report()
        source["commands"][-1].remove("--require-exact-logprobs")
        with self.assertRaises(ValueError):
            check.prepare_commands(source)

    def test_port_collision_overflow_and_layout_rejected(self):
        source = source_report()
        check.replace(source["commands"][0], "--port", 41003)
        with self.assertRaises(ValueError):
            check.prepare_commands(source)
        with self.assertRaises(ValueError):
            check.prepare_commands(source_report(65000))
        source = source_report()
        check.replace(source["commands"][0], "--expertplex-decode-count", 2)
        with self.assertRaises(ValueError):
            check.prepare_commands(source)
        source = source_report()
        check.replace(source["commands"][2], "--base-gpu-id", 2)
        with self.assertRaises(ValueError):
            check.prepare_commands(source)

    def test_mismatched_bootstrap_and_model_rejected(self):
        for command_index, flag, value in (
            (1, "--expertplex-bootstrap-addr", "http://127.0.0.1:49999"),
            (-1, "--bootstrap-port", "49998"),
            (-1, "--model-path", "another-checkpoint"),
        ):
            source = source_report()
            check.replace(source["commands"][command_index], flag, value)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                check.prepare_commands(source)

    def test_true_common_progress(self):
        intervals, proofs = progress()
        self.assertEqual(
            check.common_progress(intervals, proofs), ([12, 30], [3, 3, 3])
        )

    def test_individual_tbo_pass_does_not_prove_common_progress(self):
        intervals, proofs = progress()
        intervals[2].update(start_ns=50, end_ns=70)
        with self.assertRaises(ValueError):
            check.common_progress(intervals, proofs)
        intervals, proofs = progress()
        proofs[1] = [dict(capture_ns=11), dict(capture_ns=20)]
        with self.assertRaises(ValueError):
            check.common_progress(intervals, proofs)

    def test_duplicate_or_wrong_decode_identity_rejected(self):
        for field, value in (("pids", [100]), ("clients", [1]), ("clients", [4])):
            intervals, proofs = progress()
            intervals[2][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check.common_progress(intervals, proofs)

    def test_barrier_follows_preparation_and_preserves_one_wave(self):
        with tempfile.TemporaryDirectory() as temp:
            ready, barrier = Path(temp) / "ready", Path(temp) / "release"
            fake = types.ModuleType("check_incremental")
            calls = []
            errors = []
            fake.post = lambda *a, **kw: calls.append(("post", a))
            fake.generate = lambda *a, **kw: calls.append(("generate", a)) or "result"

            def main():
                self.assertFalse(ready.exists())
                time.sleep(0.02)  # Simulate preparation after entering main().
                self.assertFalse(ready.exists())
                self.assertEqual(
                    fake.post("url", "/flush_cache"),
                    {"status": "coordinator_flushed_before_barrier"},
                )
                self.assertEqual(fake.generate("prepared"), "result")
                with self.assertRaises(RuntimeError):
                    fake.generate("second-wave")

            fake.main = main

            def release():
                try:
                    deadline = time.monotonic() + 3
                    while not ready.exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError("Worker never prepared")
                        time.sleep(0.001)
                    self.assertGreater(json.loads(ready.read_text())["prepared_ns"], 0)
                    tmp = barrier.with_suffix(".tmp")
                    tmp.write_text(str(time.monotonic_ns()))
                    tmp.replace(barrier)
                except BaseException as exc:
                    errors.append(exc)
                    barrier.write_text(str(time.monotonic_ns()))

            with (
                patch.dict(sys.modules, {"check_incremental": fake}),
                patch.object(
                    sys, "argv", ["driver", "worker", str(ready), str(barrier)]
                ),
            ):
                thread = threading.Thread(target=release)
                thread.start()
                try:
                    check.worker()
                finally:
                    thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(calls, [("generate", ("prepared",))])
            stamps = json.loads((Path(temp) / "dispatch.json").read_text())
            self.assertLessEqual(stamps["prepared_ns"], stamps["release_ns"])
            self.assertLessEqual(stamps["release_ns"], stamps["dispatch_ns"])


@unittest.skipUnless(importlib.util.find_spec("psutil"), "psutil required")
class OwnedServiceShutdownTests(unittest.TestCase):
    def test_supervisors_release_workers_before_expert_teardown(self):
        import psutil

        child_code = """
import sys,time
from pathlib import Path
root,name=Path(sys.argv[1]),sys.argv[2]
(root/(name+'.child-ready')).touch()
while not (root/(name+'.release')).exists(): time.sleep(.005)
with (root/'order').open('a') as f: f.write(name+'-child\\n')
"""
        parent_code = """
import signal,subprocess,sys,time
from pathlib import Path
root,name,code=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
stopping=False
def stop(*args):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop)
child=subprocess.Popen([sys.executable,'-c',code,str(root),name])
while not stopping: time.sleep(.005)
(root/(name+'.release')).touch()
child.wait(timeout=5)
with (root/'order').open('a') as f: f.write(name+'-parent\\n')
"""
        with tempfile.TemporaryDirectory() as temp:
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", parent_code, temp, role, child_code]
                )
                for role in ("expert", "decode")
            ]
            owned = [psutil.Process(p.pid) for p in procs]
            try:
                deadline = time.monotonic() + 5
                while len(list(Path(temp).glob("*.child-ready"))) < 2:
                    if time.monotonic() > deadline:
                        self.fail("Fixture children did not become ready")
                    time.sleep(0.01)
                self.assertEqual(check.stop_owned_services(owned, 5, 2), [])
                self.assertEqual(
                    (Path(temp) / "order").read_text().splitlines(),
                    ["decode-child", "decode-parent", "expert-child", "expert-parent"],
                )
            finally:
                check.stop_owned_services(owned, 1, 1)
                for p in procs:
                    p.wait(timeout=2)

    def test_unresponsive_owned_tree_is_bounded_and_reaped(self):
        import psutil

        code = """
import signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as temp:
            ready = Path(temp) / "ready"
            p = subprocess.Popen([sys.executable, "-c", code, str(ready)])
            owned = [psutil.Process(p.pid)]
            try:
                deadline = time.monotonic() + 5
                while not ready.exists():
                    if time.monotonic() > deadline:
                        self.fail("Fixture tree did not become ready")
                    time.sleep(0.01)
                child = psutil.Process(int(ready.read_text()))
                start = time.monotonic()
                self.assertEqual(check.stop_owned_services(owned, 0.1, 0.1), [])
                self.assertLess(time.monotonic() - start, 5)
                self.assertTrue(
                    not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
                )
            finally:
                check.stop_owned_services(owned, 1, 1)
                p.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
