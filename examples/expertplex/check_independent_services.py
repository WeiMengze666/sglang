"""GPU integration gate for native, independently launched P/D/E services.

Run the entire script in one gpuq job. It owns and reaps only its child trees.
This lifecycle gate does not establish baseline numerical alignment or serving
speedup. CUDA graphs and TBO remain enabled on decode workers. Use a tiny BF16
checkpoint for quick checks, or a complete checkpoint for model-scale checks.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from _bootstrap import module
from check_incremental import generate
from launch import wait_http

bootstrap = module("bootstrap")


def recorder_request(url, action):
    """Native recorder endpoints return text, unlike the JSON bootstrap API."""
    if action not in ("start", "stop", "dump"):
        raise ValueError("Unknown recorder action")
    request = urllib.request.Request(
        url.rstrip("/") + f"/{action}_expert_distribution_record",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def routing_requests(rows, tokenizer, sequence_cap):
    """Preserve native custom-dataset ordering, tokenizer and output lengths."""
    requests = []
    for row in rows:
        tokens = tokenizer.encode(row.prompt)
        if len(tokens) != row.prompt_len or min(len(tokens), row.output_len) < 2:
            raise ValueError("Routing workload tokenizer/length mismatch")
        if len(tokens) + row.output_len > sequence_cap:
            raise ValueError("Routing workload exceeds the frozen sequence cap")
        requests.append((tokens, row.output_len))
    if not requests:
        raise ValueError("Empty routing workload")
    return requests


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
    p.add_argument(
        "--record-routing",
        action="store_true",
        help="Exercise native stat recorder endpoints; diagnostic workload only",
    )
    p.add_argument(
        "--routing-workload",
        type=Path,
        help="Optional frozen custom dataset for routing calibration after lifecycle checks",
    )
    p.add_argument("--routing-requests", type=int, default=64)
    p.add_argument("--routing-seed", type=int, default=42)
    p.add_argument("--routing-sequence-cap", type=int, default=24576)
    p.add_argument(
        "--moe-config",
        type=Path,
        help="Optional E-side JSON to test CLI policy overrides",
    )
    args = p.parse_args()
    if args.startup_timeout <= 0 or not 0 < args.mem_fraction_static < 1:
        p.error("Use positive startup timeout and memory fraction in (0, 1)")
    if args.kv_tokens != "auto" and (
        not args.kv_tokens.isdecimal() or int(args.kv_tokens) < 256
    ):
        p.error("--kv-tokens must be auto or an integer >= 256")
    workload = None
    prefill_capacity = 256
    if args.routing_workload:
        if not args.record_routing or args.moe_config:
            p.error(
                "Routing calibration requires --record-routing and automatic topology"
            )
        if args.routing_requests < 1 or args.routing_sequence_cap < 256:
            p.error("Use positive routing request count and sequence cap >=256")
        if args.kv_tokens != "auto" and int(args.kv_tokens) < args.routing_sequence_cap:
            p.error("Routing calibration KV capacity must cover the sequence cap")
        from sglang.benchmark.datasets.custom import sample_custom_requests
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
        state = random.getstate()
        try:
            random.seed(args.routing_seed)
            rows = sample_custom_requests(
                str(args.routing_workload),
                args.routing_requests,
                tokenizer,
                context_len=args.routing_sequence_cap,
            )
        finally:
            random.setstate(state)
        if len(rows) != args.routing_requests:
            p.error("Native dataset loader returned fewer calibration requests")
        workload = routing_requests(rows, tokenizer, args.routing_sequence_cap)
        prefill_capacity = args.routing_sequence_cap
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"scope": __doc__, "status": "running", "launches": [], "checks": {}}
    children = {}
    recording = {}
    routing_sources = {"prefill": [], "decode": []}
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
        str(prefill_capacity),
        "--max-running-requests",
        "16",
        "--skip-tokenizer-init",
        "--attention-backend",
        "flashinfer",
        "--sampling-backend",
        "flashinfer",
    ]
    if workload is not None:
        # Native PD overlap launches the next decode before retiring the last
        # result. Its extra terminal forward is not a generated token. Disable
        # CPU scheduling overlap for exact N-1 routing accounting only; retain
        # CUDA Graph/TBO, and leave lifecycle/performance defaults unchanged.
        prefix.append("--disable-overlap-schedule")
        report["routing_schedule"] = "serial CPU scheduling; CUDA Graph/TBO enabled"

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
        child_env = dict(env)
        if args.record_routing and name != "moe":
            directory = (args.output_dir / "routing" / name).resolve()
            directory.mkdir(parents=True, exist_ok=False)
            child_env["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = str(directory)
        proc = subprocess.Popen(
            command,
            env=child_env,
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
        if args.record_routing and name != "moe":
            url = f"http://127.0.0.1:{port}"
            recorder_request(url, "start")
            recording[name] = (url, directory)
        print("READY", name, flush=True)

    def dump_routing(name):
        if name not in recording:
            return
        url, directory = recording[name]
        recorder_request(url, "stop")
        recorder_request(url, "dump")
        paths = sorted(directory.glob("expert_distribution_recorder_*.pt"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one native stat dump for {name}: {paths}")
        routing_sources["prefill" if name == "prefill" else "decode"].extend(paths)
        del recording[name]

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
            str(prefill_capacity) if phase == "prefill" else "16"
        )
        if args.kv_tokens != "auto":
            command += ["--max-total-tokens", args.kv_tokens]
        if args.record_routing:
            command += [
                "--expert-distribution-recorder-mode",
                "stat",
                "--expert-distribution-recorder-buffer-size",
                "-1",
            ]
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
        if args.moe_config:
            policy_flags += ["--expertplex-config", str(args.moe_config.resolve())]
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
        dump_routing("decode-1")
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
        if args.record_routing:
            from summarize_routing import summarize_files

            if workload is not None:
                # Discard lifecycle probes and the old D1 dump. Starting the
                # native recorder resets its buffer. Only current live workers
                # contribute to this workload's calibration.
                routing_sources = {"prefill": [], "decode": []}
                for url, _ in recording.values():
                    recorder_request(url, "stop")
                    recorder_request(url, "start")
                report["routing_workload"] = {
                    "dataset_sha256": hashlib.sha256(
                        args.routing_workload.read_bytes()
                    ).hexdigest(),
                    "seed": args.routing_seed,
                    "requests": len(workload),
                    "sequence_cap": args.routing_sequence_cap,
                    "ordered_token_requests_sha256": hashlib.sha256(
                        json.dumps(workload).encode()
                    ).hexdigest(),
                    "prefill_tokens": sum(len(tokens) for tokens, _ in workload),
                    "decode_tokens": sum(length - 1 for _, length in workload),
                    "completed": 0,
                }
                for index, (tokens, length) in enumerate(workload):
                    # Direct PD generation uses native KV handoff and flushes
                    # both caches first, so repeated contexts are fully counted.
                    result = generate(
                        f"http://127.0.0.1:{base + 11 + index % 3}",
                        {
                            "input_ids": tokens,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": length,
                                "ignore_eos": True,
                            },
                        },
                        prefill_url=f"http://127.0.0.1:{base + 10}",
                        bootstrap_port=base + 200,
                    )
                    if result.get("meta_info", {}).get("completion_tokens") != length:
                        raise RuntimeError("Incomplete routing calibration request")
                    report["routing_workload"]["completed"] += 1
                    save()
                    print(f"ROUTING {index + 1}/{len(workload)}", flush=True)
            for name in list(recording):
                dump_routing(name)
            model = module("config").normalize_model_config(
                json.loads((args.model_path / "config.json").read_text())
            )
            report["checks"]["native_routing_recorder"] = {
                phase: summarize_files(
                    paths,
                    layers=model["num_hidden_layers"],
                    experts=model["num_experts"],
                    top_k=model["num_experts_per_tok"],
                )
                for phase, paths in routing_sources.items()
            }
            report["routing_scope"] = (
                "Frozen workload routing with full prompt cache flushes; not performance"
                if workload is not None
                else "Lifecycle diagnostic prompts only, not workload calibration or performance"
            )
            if workload is not None:
                for phase in ("prefill", "decode"):
                    actual = report["checks"]["native_routing_recorder"][phase][
                        "tokens_per_layer"
                    ]
                    if actual != report["routing_workload"][phase + "_tokens"]:
                        raise RuntimeError(
                            f"{phase} recorder did not account for exactly the workload tokens: {actual}"
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
