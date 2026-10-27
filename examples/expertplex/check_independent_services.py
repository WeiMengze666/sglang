"""GPU integration gate for native, independently launched P/D/E services.

Run the entire script in one gpuq job. It owns and reaps only its child trees.
This lifecycle gate does not establish baseline numerical alignment or serving
speedup. CUDA graphs and TBO remain enabled on decode workers. Use a tiny BF16
checkpoint for quick checks, or a complete checkpoint for model-scale checks.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import module
from check_incremental import generate
from launch import wait_http

bootstrap = module("bootstrap")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--port", type=int, default=32000)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument(
        "--kv-tokens", default="2048", help="Token count, or auto for native sizing"
    )
    p.add_argument("--mem-fraction-static", type=float, default=0.6)
    p.add_argument("--decode-sm-budget", type=int)
    p.add_argument("--expected-decode-tiles", type=int)
    p.add_argument("--max-decode-sm-fraction", type=float)
    args = p.parse_args()
    if args.startup_timeout <= 0 or not 0 < args.mem_fraction_static < 1:
        p.error("Use positive startup timeout and memory fraction in (0, 1)")
    if args.kv_tokens != "auto" and (
        not args.kv_tokens.isdecimal() or int(args.kv_tokens) < 256
    ):
        p.error("--kv-tokens must be auto or an integer >= 256")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"scope": __doc__, "status": "running", "launches": [], "checks": {}}
    children = {}
    env = dict(os.environ)
    env.update(
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK", MC_INTRANODE_NVLINK="true"
    )
    env.pop("SGLANG_EXPERTPLEX_SYNTHETIC_SEED", None)
    base = args.port
    moe_url = f"http://127.0.0.1:{base}"
    prefix = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path.resolve()),
        "--enable-expertplex",
        "--expertplex-startup-timeout",
        str(args.startup_timeout),
        "--dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--enable-two-batch-overlap",
        "--max-prefill-tokens",
        "256",
        "--max-running-requests",
        "16",
        "--skip-tokenizer-init",
        "--attention-backend",
        "flashinfer",
        "--sampling-backend",
        "flashinfer",
    ]

    def save():
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    def stop_signal(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_signal)

    def start(name, command, port):
        log = (args.output_dir / f"{name}.log").open("w")
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[name] = (proc, log)
        report["launches"].append({"name": name, "command": command, "pid": proc.pid})
        save()
        wait_http(
            f"http://127.0.0.1:{port}",
            [(name, proc, log)],
            time.monotonic() + args.startup_timeout,
        )
        print("READY", name, flush=True)

    def stop(name):
        import psutil

        proc, log = children.pop(name)
        try:
            owned = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            owned = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        for child in owned:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=10)
        for child in alive:
            child.kill()
        _, alive = psutil.wait_procs(alive, timeout=10)
        log.close()
        if alive:
            raise RuntimeError(f"Owned child processes failed to exit: {alive}")

    def attention(name, device, phase):
        command = prefix + [
            "--expertplex-bootstrap-addr",
            moe_url,
            "--base-gpu-id",
            str(device),
            "--port",
            str(base + 10 + device),
            "--nccl-port",
            str(base + 100 + device),
            "--disaggregation-mode",
            phase,
            "--disaggregation-bootstrap-port",
            str(base + 200 + device),
            "--chunked-prefill-size",
            "-1" if phase == "prefill" else "16",
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "disabled" if phase == "prefill" else "full",
            "--cuda-graph-max-bs-decode",
            "16",
        ]
        command[command.index("--max-prefill-tokens") + 1] = (
            "256" if phase == "prefill" else "16"
        )
        if args.kv_tokens != "auto":
            command += ["--max-total-tokens", args.kv_tokens]
        return command

    def one(device, length=32):
        payload = {
            "input_ids": [11 + i % 32 for i in range(length)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
        }
        result = generate(
            f"http://127.0.0.1:{base + 10 + device}",
            payload,
            prefill_url=f"http://127.0.0.1:{base + 10}",
            bootstrap_port=base + 200,
        )
        if result.get("meta_info", {}).get("completion_tokens") != 8:
            raise RuntimeError(f"Incomplete generation: {result}")
        return result

    try:
        policy = {
            name: getattr(args, name)
            for name in (
                "decode_sm_budget",
                "expected_decode_tiles",
                "max_decode_sm_fraction",
            )
            if getattr(args, name) is not None
        }
        policy_flags = [
            value
            for name, setting in policy.items()
            for value in ("--expertplex-" + name.replace("_", "-"), str(setting))
        ]
        start(
            "moe",
            prefix
            + [
                "--expertplex-role",
                "expert",
                "--port",
                str(base),
                "--expertplex-prefill-count",
                "1",
                "--expertplex-decode-count",
                "3",
                "--expertplex-expert-count",
                "4",
            ]
            + policy_flags,
            base,
        )
        node, _ = bootstrap.fetch_config(moe_url, args.startup_timeout)
        if any(getattr(node, name) != value for name, value in policy.items()):
            raise RuntimeError("MoE bootstrap did not publish the selected SM policy")
        report["checks"]["moe_sm_policy"] = {
            name: getattr(node, name)
            for name in (
                "decode_sm_budget",
                "expected_decode_tiles",
                "max_decode_sm_fraction",
            )
        }
        generation = bootstrap.request(moe_url, "/get_server_info")["run_id"]
        # D can join first: no central launcher or all-client startup barrier.
        for device in (1, 2, 3):
            start(
                f"decode-{device}",
                attention(f"decode-{device}", device, "decode"),
                base + 10 + device,
            )
        start("prefill", attention("prefill", 0, "prefill"), base + 10)
        before = one(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outputs = list(
                pool.map(lambda item: one(*item), [(1, 31), (2, 127), (3, 255)])
            )
        report["checks"]["concurrent_pd_requests"] = len(outputs)
        stop("decode-1")
        info = bootstrap.request(moe_url, "/get_server_info")
        if {c["id"] for c in info["clients"]} != {0, 2, 3}:
            raise RuntimeError(f"Decode did not release its lease: {info}")
        one(2)
        start("decode-1-restarted", attention("decode-1", 1, "decode"), base + 11)
        after = one(1)
        old_scores = before["meta_info"]["output_token_logprobs"]
        new_scores = after["meta_info"]["output_token_logprobs"]
        if old_scores != new_scores:
            raise RuntimeError("Tokens/logprobs changed after attention restart")
        if bootstrap.request(moe_url, "/get_server_info")["run_id"] != generation:
            raise RuntimeError("MoE server restarted with the attention client")
        report["checks"].update(
            attention_restart_exact_scores=True, moe_generation_preserved=True
        )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = []
        for name in reversed(list(children)):
            try:
                stop(name)
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc!r}")
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "failed"
        save()
        if cleanup_errors:
            raise RuntimeError(f"Cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
