"""Launch and supervise the shared experts, SGLang P/D servers, and router.

Run with --dry-run on a CPU workstation to inspect exact commands. Each child
has its own process group, so shutdown only terminates this launcher's children.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from _bootstrap import module

config = module("config")
rendezvous = module("rendezvous")


def commands(node, args):
    worker = [
        sys.executable,
        "-m",
        "sglang.srt.expertplex.worker",
        "--config",
        str(Path(args.config).resolve()),
    ]
    servers = []
    for client in node.clients:
        capacity = (
            node.max_prefill_tokens
            if client.phase == "prefill"
            else node.max_decode_tokens
        )
        bootstrap = args.bootstrap_port + client.id
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            node.model_path,
            "--dtype",
            "bfloat16",
            "--device",
            "cuda",
            "--load-format",
            "dummy" if node.weight_mode == "synthetic" else "safetensors",
            "--tp-size",
            "1",
            "--ep-size",
            "1",
            "--pp-size",
            "1",
            "--base-gpu-id",
            str(client.device),
            "--host",
            "127.0.0.1",
            "--port",
            str(client.port),
            "--nccl-port",
            str(args.nccl_port + client.id),
            "--disaggregation-mode",
            client.phase,
            "--disaggregation-bootstrap-port",
            str(bootstrap),
            "--disaggregation-transfer-backend",
            "mooncake",
            "--cuda-graph-backend-decode",
            "full" if args.cuda_graph else "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--disable-overlap-schedule",
            "--disable-flashinfer-autotune",
            "--chunked-prefill-size",
            str(capacity if client.phase == "decode" or args.chunked_prefill else -1),
            "--max-prefill-tokens",
            str(capacity),
            "--max-running-requests",
            str(min(node.max_decode_tokens, capacity)),
            "--max-total-tokens",
            str(args.kv_tokens),
            "--mem-fraction-static",
            "0.75",
            "--attention-backend",
            args.attention_backend,
            "--sampling-backend",
            args.sampling_backend,
        ]
        if args.ib_device:
            command += ["--disaggregation-ib-device", args.ib_device]
        if node.overlap == "tbo":
            command += ["--enable-two-batch-overlap"]
        if node.prefill_mailbox_tokens is not None:
            command += ["--tbo-token-distribution-threshold", "0.5"]
        if args.skip_tokenizer_init:
            command += ["--skip-tokenizer-init"]
        if args.tokenizer_path:
            command += ["--tokenizer-path", args.tokenizer_path]
        if getattr(args, "diagnostic_logit_processor", False):
            command += ["--enable-custom-logit-processor"]
        if getattr(args, "sequence_cap", None):
            command += ["--context-length", str(args.sequence_cap)]
        if args.cuda_graph:
            command += ["--cuda-graph-max-bs-decode", str(min(32, capacity))]
        servers.append((client, command))
    router = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--prefill-policy",
        "round_robin",
        "--decode-policy",
        "power_of_two"
        if sum(client.phase == "decode" for client in node.clients) > 1
        else "round_robin",
    ]
    for client in node.clients:
        url = f"http://127.0.0.1:{client.port}"
        if client.phase == "prefill":
            router += ["--prefill", url, str(args.bootstrap_port + client.id)]
        else:
            router += ["--decode", url]
    return worker, servers, router


def wait_http(url, children, deadline, expected_workers=None):
    while time.monotonic() < deadline:
        for name, process, _ in children:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{name} exited with code {process.returncode}; inspect its log"
                )
        try:
            endpoint = "/readiness" if expected_workers is not None else "/health"
            with urllib.request.urlopen(url + endpoint, timeout=2) as response:
                if response.status == 200 and (
                    expected_workers is None
                    or json.load(response).get("healthy_workers", 0) >= expected_workers
                ):
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Service did not become healthy: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--bootstrap-port", type=int, default=19000)
    parser.add_argument("--nccl-port", type=int, default=20000)
    parser.add_argument("--kv-tokens", type=int, default=65536)
    parser.add_argument("--ib-device")
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        help="Optional chunking ablation; default P processes whole requests",
    )
    parser.add_argument(
        "--kv-transport",
        choices=("nvlink", "rdma"),
        default="nvlink",
        help="P-to-D KV transport; this launcher targets one NVLink node",
    )
    parser.add_argument(
        "--attention-backend",
        default="flashinfer",
        choices=("flashinfer", "triton", "fa3", "fa4"),
    )
    parser.add_argument(
        "--sampling-backend", default="flashinfer", choices=("flashinfer", "pytorch")
    )
    parser.add_argument("--log-dir", default="expertplex-logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--diagnostic-logit-processor",
        action="store_true",
        help="Enable fixed-trajectory score capture on isolated diagnostic servers only",
    )
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Capture decode forward"
    )
    parser.add_argument(
        "--tokenizer-path", help="Optional separate tokenizer directory"
    )
    parser.add_argument(
        "--sequence-cap", type=int, help="Common benchmark sequence cap"
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        help="Accept token-ID requests without a tokenizer",
    )
    args = parser.parse_args()
    node = config.load_config(args.config)
    worker_command, server_commands, router_command = commands(node, args)
    if args.kv_tokens <= 0:
        parser.error("--kv-tokens must be positive")
    all_ports = (
        [args.port]
        + [c.port for c in node.clients]
        + [args.bootstrap_port + c.id for c in node.clients]
        + [args.nccl_port + c.id for c in node.clients]
    )
    if len(all_ports) != len(set(all_ports)) or any(
        not 0 < p < 65536 for p in all_ports
    ):
        parser.error(
            "Router, HTTP, bootstrap and NCCL ports must be valid and distinct"
        )
    if args.dry_run:
        print(shlex.join(worker_command))
        for client, command in server_commands:
            print(
                f"SGLANG_EXPERTPLEX_CONFIG={shlex.quote(str(Path(args.config).resolve()))} "
                f"SGLANG_EXPERTPLEX_CLIENT_ID={client.id} " + shlex.join(command)
            )
        print(shlex.join(router_command))
        return
    spec = config.ModelSpec.from_model_path(node.model_path)
    node.validate(spec)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0,1,2,3,4,5,6,7"):
        parser.error(
            "Use identical full CUDA visibility (0,1,2,3,4,5,6,7) for this launcher"
        )
    if (
        spec.attention_weight_bytes
        + args.kv_tokens * spec.kv_bytes_per_token
        + 8 * 2**30
        > node.gpu_memory_gib * 2**30
    ):
        parser.error(
            "Attention weights plus requested KV pool exceed the memory budget"
        )
    log_root = Path(args.log_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    children = []
    base_env = dict(os.environ)
    base_env.pop("SGLANG_EXPERTPLEX_CONFIG", None)
    base_env.pop("SGLANG_EXPERTPLEX_CLIENT_ID", None)
    if args.kv_transport == "nvlink":
        base_env["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "INTRA_NODE_NVLINK"
        base_env["MC_INTRANODE_NVLINK"] = "true"

    def spawn(name, command, env):
        log = (log_root / f"{name}.log").open("w")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((name, process, log))
        print(
            f"Started {name} PID {process.pid}; log: {log_root / (name + '.log')}",
            flush=True,
        )
        return process

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        spawn("experts", worker_command, base_env)
        rendezvous.wait_for_workers(node)
        for client, command in server_commands:
            env = dict(base_env)
            env["SGLANG_EXPERTPLEX_CONFIG"] = str(Path(args.config).resolve())
            env["SGLANG_EXPERTPLEX_CLIENT_ID"] = str(client.id)
            spawn(f"{client.phase}-{client.id}", command, env)
        deadline = time.monotonic() + node.startup_timeout_s
        for client, _ in server_commands:
            wait_http(f"http://127.0.0.1:{client.port}", children, deadline)
        spawn("router", router_command, base_env)
        wait_http(
            f"http://{args.host}:{args.port}",
            children,
            time.monotonic() + 120,
            expected_workers=len(node.clients),
        )
        rendezvous.atomic_json(
            log_root / "processes.json",
            {
                "config": node.to_dict(),
                "children": [
                    {"name": name, "pid": process.pid} for name, process, _ in children
                ],
            },
        )
        print(f"Ready: http://{args.host}:{args.port}", flush=True)
        while True:
            for name, process, _ in children:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with code {process.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # Keep the expert allocations alive until all attention clients exit.
        for group in (
            [child for child in reversed(children) if child[0] != "experts"],
            [child for child in children if child[0] == "experts"],
        ):
            for _, process, _ in group:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for _, process, log in group:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                log.close()


if __name__ == "__main__":
    main()
