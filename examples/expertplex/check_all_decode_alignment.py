"""Validate three independent decode services against the same strict histories.

Shared P cache is flushed once before a barrier release. Captures disable scheduling
CPU overlap and force histories; all timings are diagnostic, never performance.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/expertplex"))


def option(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Expected exactly one {flag}")
    index = command.index(flag) + 1
    if index == len(command):
        raise ValueError(f"Missing value for {flag}")
    return command[index]


def replace(command, flag, value):
    option(command, flag)
    command[command.index(flag) + 1] = str(value)


def prepare_commands(saved):
    """Replay a successful default-layout run using this checkout/interpreter."""
    required = {
        "status": "passed",
        "variant": "expertplex",
        "expert_count": 4,
        "concurrency": 16,
        "deterministic_inference": True,
        "batched_request": True,
        "tbo_enabled": True,
    }
    if any(saved.get(k) != v for k, v in required.items()):
        raise ValueError("Source must be a passed deterministic 16-request 4E TBO run")
    if saved.get("cleanup_errors") or saved.get("required_tbo_replays", 0) < 2:
        raise ValueError("Source requires clean teardown and at least two TBO replays")
    if type(saved.get("steps")) is not int or saved["steps"] < 2:
        raise ValueError("Source needs at least two generation steps")
    launches = [list(c) for c in saved["commands"] if "sglang.launch_server" in c]
    if len(launches) != 3:
        raise ValueError("Source must contain one E, one P and one D launch")
    e, p, d = launches
    if option(e, "--expertplex-role") != "expert":
        raise ValueError("Missing native expert service")
    for flag, expected in (
        ("--expertplex-prefill-count", "1"),
        ("--expertplex-decode-count", "3"),
        ("--expertplex-expert-count", "4"),
    ):
        if option(e, flag) != expected:
            raise ValueError("This check requires the default 1P3D4E layout")
    for c, role, gpu in ((p, "prefill", "0"), (d, "decode", "1")):
        if (
            option(c, "--disaggregation-mode") != role
            or option(c, "--base-gpu-id") != gpu
        ):
            raise ValueError("Expected P on GPU0 and first D on GPU1")
        for flag in (
            "--enable-two-batch-overlap",
            "--enable-deterministic-inference",
            "--disable-overlap-schedule",
        ):
            if flag not in c:
                raise ValueError(f"Strict source missing {flag}")
    if any(option(c, "--host") != "127.0.0.1" for c in launches):
        raise ValueError("This single-node diagnostic requires loopback listeners")
    for c in launches:
        c[0] = sys.executable
    commands = [("moe", e), ("prefill", p), ("decode-1", d)]
    for device in (2, 3):
        c = list(d)
        replace(c, "--base-gpu-id", device)
        for flag in ("--port", "--nccl-port", "--disaggregation-bootstrap-port"):
            replace(c, flag, int(option(d, flag)) + device - 1)
        commands.append((f"decode-{device}", c))
    ports = [
        int(option(c, flag))
        for _, c in commands
        for flag in ("--port", "--nccl-port", "--disaggregation-bootstrap-port")
        if flag in c
    ]
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("Service ports collide or exceed the valid range")
    urls = {
        i: "http://127.0.0.1:" + option(c, "--port")
        for i, (_, c) in enumerate(commands[1:])
    }
    template = list(saved["commands"][-1])
    if len(template) < 2 or Path(template[1]).name != "check_incremental.py":
        raise ValueError("Missing final incremental capture command")
    if (
        option(template, "--top-k") != "0"
        or option(template, "--concurrency") != "16"
        or int(option(template, "--steps")) != saved["steps"]
        or int(option(template, "--require-decode-batch-size")) < 2
        or "--require-exact-logprobs" not in template
        or "--batched-request" not in template
        or "--reference" not in template
    ):
        raise ValueError("Capture must retain full-vocabulary exact and batching gates")
    if (
        option(template, "--prefill-url") != urls[0]
        or option(template, "--base-url") != urls[1]
    ):
        raise ValueError("Source capture URLs differ from the service endpoints")
    model = option(e, "--model-path")
    if any(option(c, "--model-path") != model for c in (p, d, template)):
        raise ValueError("Source services and capture must use one checkpoint")
    bootstrap = "http://127.0.0.1:" + option(e, "--port")
    if any(option(c, "--expertplex-bootstrap-addr") != bootstrap for c in (p, d)):
        raise ValueError("Attention services must connect to the launched expert pool")
    if option(template, "--bootstrap-port") != option(
        p, "--disaggregation-bootstrap-port"
    ):
        raise ValueError("Capture KV bootstrap port differs from the prefill server")
    template[:2] = [
        sys.executable,
        str(ROOT / "examples/expertplex/check_incremental.py"),
    ]
    return commands, urls, template


def common_progress(intervals, proofs):
    """Require independent D identities and actual completions in their overlap."""
    if len(intervals) != 3 or len(proofs) != 3:
        raise ValueError("Need all three decode workers")
    if any(len(i["pids"]) != 1 or len(i["clients"]) != 1 for i in intervals):
        raise ValueError("Each decode worker must have one sampling process and client")
    if len({i["pids"][0] for i in intervals}) != 3 or {
        i["clients"][0] for i in intervals
    } != {1, 2, 3}:
        raise ValueError(
            "Decode workers must have distinct processes and expected clients"
        )
    start = max(i["start_ns"] for i in intervals)
    end = min(i["end_ns"] for i in intervals)
    counts = [sum(start <= p["capture_ns"] <= end for p in proof) for proof in proofs]
    if end <= start or min(counts) < 2:
        raise ValueError(
            f"All three D must make concurrent nonempty two-child progress: interval={start, end}, counts={counts}"
        )
    return [start, end], counts


def worker():
    import check_incremental as capture

    ready, barrier = map(Path, sys.argv[2:4])
    sys.argv = ["check_incremental.py"] + sys.argv[4:]
    original = capture.post

    # The coordinator flushes all four services before starting these workers.
    def post(url, endpoint, *args, **kwargs):
        if endpoint == "/flush_cache":
            return {"status": "coordinator_flushed_before_barrier"}
        return original(url, endpoint, *args, **kwargs)

    capture.post = post
    original_generate = capture.generate

    def generate(*args, **kwargs):
        # main() imports/serializes the processor and prepares all payloads first.
        # Readiness before main() only synchronized Python process entry; a slow
        # import could let another D finish its entire short trajectory first.
        if ready.exists():
            raise RuntimeError("This concurrent diagnostic expects one prepared wave")
        ready_tmp = ready.with_suffix(".tmp")
        ready_tmp.write_text(json.dumps({"prepared_ns": time.monotonic_ns()}) + "\n")
        ready_tmp.replace(ready)
        deadline = time.monotonic() + 180
        while not barrier.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Coordinator barrier not released")
            time.sleep(0.001)
        dispatched = time.monotonic_ns()
        ready.with_name("dispatch.json").write_text(
            json.dumps(
                {
                    "prepared_ns": json.loads(ready.read_text())["prepared_ns"],
                    "release_ns": int(barrier.read_text()),
                    "dispatch_ns": dispatched,
                }
            )
            + "\n"
        )
        return original_generate(*args, **kwargs)

    capture.generate = generate
    capture.main()


def finish_report(result, out, remaining):
    """Persist cleanup failure before raising, preserving earlier diagnostics."""
    result["remaining_owned_pids"] = list(remaining)
    if remaining:
        result["status"] = "failed"
        result["cleanup_error"] = "Own process cleanup incomplete"
    temp = out / "report.json.tmp"
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(out / "report.json")
    if remaining:
        raise RuntimeError(result["cleanup_error"])


def stop_owned_services(processes, grace_timeout=90, child_timeout=10):
    """Stop clients before E and allow each supervisor to release its workers.

    Snapshot process identities before a parent can exit and orphan children.
    Descendant termination is a bounded fallback, after the supervisor's own
    stop protocol; killing E workers first bypasses executor.stop().
    """
    import psutil

    trees = []
    for proc in processes:
        try:
            trees.append((proc, proc.children(recursive=True)))
        except psutil.NoSuchProcess:
            trees.append((proc, []))
    remaining = []
    for proc, descendants in reversed(trees):
        try:
            proc.terminate()
            try:
                proc.wait(timeout=grace_timeout)
            except psutil.TimeoutExpired:
                proc.kill()
        except psutil.NoSuchProcess:
            pass
        owned = [proc, *descendants]
        for child in descendants:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=child_timeout)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=child_timeout)
        for child in alive:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(child.pid)
            except psutil.NoSuchProcess:
                pass
    return remaining


def run_case(source, out):
    import psutil
    from check_incremental import post
    from check_tbo_completion import check
    from launch import wait_http

    saved = json.loads((source / "report.json").read_text())
    cmds, urls, template = prepare_commands(saved)
    out.mkdir(parents=True)
    runtime_env = dict(
        os.environ,
        SGLANG_TBO_DEBUG="1",
        PYTHONPATH=os.pathsep.join(
            [
                str(ROOT / "python"),
                str(ROOT / "examples/expertplex"),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    )
    processes = []
    logs = []
    result = {
        "scope": "Three active D workers, 16 identical fixed trajectories per D (48 concurrent requests), same 16-trajectory native reference. Exact full vocabulary, unchanged per-request batch and TBO gates. Common decode sampling interval verifies concurrent progress only, not kernel overlap. No performance acceptance.",
        "status": "running",
        "performance_eligible": False,
        "commands": [],
        "results": [],
        "source_report_sha256": hashlib.sha256(
            (source / "report.json").read_bytes()
        ).hexdigest(),
        "diagnostic_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }

    def save():
        temp = out / "report.json.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(out / "report.json")

    save()
    try:
        for label, c in cmds:
            log = (out / (label + ".log")).open("w")
            logs.append(log)
            # Imported diagnostic processors are serialized with their module name.
            # Expose that source module to the receiving P/D subprocesses.
            proc = subprocess.Popen(
                c,
                env=runtime_env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(psutil.Process(proc.pid))
            result["commands"].append(c)
            save()
            port = c[c.index("--port") + 1]
            wait_http(
                "http://127.0.0.1:" + port,
                [(label, proc, log)],
                time.monotonic() + 1800,
            )
        if saved.get("warmup_capture"):
            from check_numerical_alignment import warmup_capture_command

            # Finish all warmups before the shared flush and strict release.
            for device in (1, 2, 3):
                warmup_dir = out / f"warmup-decode-{device}"
                warmup_dir.mkdir()
                warmup = warmup_capture_command(template, warmup_dir)
                replace(warmup, "--base-url", urls[device])
                result["commands"].append(warmup)
                save()
                with (warmup_dir / "capture.log").open("w") as log:
                    subprocess.run(
                        warmup,
                        env=runtime_env,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1800,
                    )
            result["warmup_completed_on_all_decode_services"] = True
        for device in range(4):
            post(urls[device], "/flush_cache")
        result["shared_cache_flush_before_workers"] = True
        barrier = out / "release"
        workers = []
        for device in (1, 2, 3):
            dest = out / f"decode-{device}"
            dest.mkdir()
            c = list(template[2:])
            for flag, value in [
                ("--base-url", urls[device]),
                ("--capture-dir", dest / "capture"),
                ("--output", dest / "scores.json"),
            ]:
                replace(c, flag, value)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                str(dest / "ready"),
                str(barrier),
                *c,
            ]
            log = (dest / "capture.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                cmd,
                env=runtime_env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(psutil.Process(proc.pid))
            workers.append((device, proc, dest))
            result["commands"].append(cmd)
        deadline = time.monotonic() + 180
        while not all((dest / "ready").exists() for _, _, dest in workers):
            if any(proc.poll() is not None for _, proc, _ in workers):
                raise RuntimeError("Capture worker exited before barrier")
            if time.monotonic() > deadline:
                raise TimeoutError("Capture worker readiness")
            time.sleep(0.1)
        release_tmp = barrier.with_suffix(".tmp")
        release_tmp.write_text(str(time.monotonic_ns()))
        release_tmp.replace(barrier)
        save()
        for device, proc, dest in workers:
            rc = proc.wait(timeout=1800)
            result["results"].append({"device": device, "returncode": rc})
            save()
        if any(row["returncode"] for row in result["results"]):
            raise RuntimeError("Strict capture failed on one or more D workers")
        result["prepared_wave_dispatch"] = [
            json.loads((dest / "dispatch.json").read_text()) for _, _, dest in workers
        ]
        if not all(
            r["prepared_ns"] <= r["release_ns"] <= r["dispatch_ns"]
            for r in result["prepared_wave_dispatch"]
        ):
            raise ValueError("Invalid prepared/release/dispatch ordering")
        intervals = []
        proofs = []
        for device, _, dest in workers:
            score = json.loads((dest / "scores.json").read_text())
            if (
                score["status"] != "completed"
                or not score["all_full_vocab_logprobs_exact"]
                or score.get("decode_batch_failures")
                or len(score["comparisons"]) != 16 * saved["steps"]
            ):
                raise ValueError(
                    "Decode full-vocabulary or per-request batch gate failed"
                )
            del score
            check(dest, expert_count=4, min_replays=2)
            proof = json.loads((dest / "tbo-proof.json").read_text())[
                "completed_two_child_replays"
            ]
            proofs.append(proof)
            stamps = []
            pids = set()
            clients = set()
            for path in (dest / "capture").rglob("step-*-pid-*.json"):
                row = json.loads(path.read_text())
                meta = row.get("tbo_debug") or {}
                if row["step"] > 0 and meta.get("phase") == "decode":
                    stamps.append(meta["capture_ns"])
                    pids.add(row["pid"])
                    clients.add(meta["client_id"])
            if len(pids) != 1 or len(clients) != 1 or not stamps:
                raise ValueError("Missing or ambiguous decode sampling evidence")
            intervals.append(
                {
                    "device": device,
                    "start_ns": min(stamps),
                    "end_ns": max(stamps),
                    "pids": sorted(pids),
                    "clients": sorted(clients),
                    "positions": 16 * saved["steps"],
                }
            )
        result["decode_intervals"] = intervals
        common, counts = common_progress(intervals, proofs)
        result.update(
            decode_intervals=intervals,
            common_interval_ns=common,
            completed_two_child_replays_in_common_interval=counts,
        )
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        remaining = stop_owned_services(processes)
        for log in logs:
            log.close()
        finish_report(result, out, remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Passed ExpertPlex check_numerical_alignment run; retained native reference files must exist",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.source_run.resolve(), args.output.resolve())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        main()
