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


def kv_token_limit(value):
    """Leave capacity profiling to SGLang unless a control run requests a cap."""
    if value == "auto":
        return None
    try:
        tokens = int(value)
        if tokens > 0:
            return tokens
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("KV capacity must be 'auto' or a positive integer")


def memory_fraction(value):
    fraction = float(value)
    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError(
            "Memory fraction must lie strictly between 0 and 1"
        )
    return fraction


def prefill_graph_options(args, capacity):
    """Use native graph buckets within the allocated P token capacity."""
    backend = getattr(args, "prefill_graph_backend", "disabled")
    limit = getattr(args, "prefill_graph_max_tokens", None)
    if backend not in ("disabled", "breakable"):
        raise ValueError("Prefill graph backend must be disabled or breakable")
    if backend == "disabled":
        if limit is not None:
            raise ValueError("Prefill graph token limit requires an enabled graph")
        return ["--cuda-graph-backend-prefill", "disabled"]
    limit = capacity if limit is None else limit
    if type(limit) is not int or not 0 < limit <= capacity:
        raise ValueError("Prefill graph token limit must fit the P token capacity")
    return [
        "--cuda-graph-backend-prefill",
        backend,
        "--cuda-graph-max-bs-prefill",
        str(limit),
    ]


def add_prefill_graph_args(parser):
    parser.add_argument(
        "--prefill-graph-backend",
        choices=("disabled", "breakable"),
        default="disabled",
        help="Native prefill graph backend; eager attention boundaries",
    )
    parser.add_argument(
        "--prefill-graph-max-tokens",
        type=int,
        help="Largest captured P token bucket; larger batches use native eager fallback",
    )


def commands(node, args):
    prefill_graph = prefill_graph_options(args, node.max_prefill_tokens)
    if prefill_graph[1] != "disabled" and node.overlap == "tbo":
        raise ValueError("Prefill graph requires overlap=none; native TBO disables it")
    worker = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        node.model_path,
        "--enable-expertplex",
        "--expertplex-role",
        "expert",
        "--expertplex-config",
        str(Path(args.config).resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port + 200),
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
            "--enable-expertplex",
            "--expertplex-bootstrap-addr",
            f"http://127.0.0.1:{args.port + 200}",
            "--model-path",
            node.model_path,
            "--dtype",
            "bfloat16",
            "--device",
            "cuda",
            "--load-format",
            "dummy" if node.weight_mode == "synthetic" else "safetensors",
            "--tp-size",
            str(client.tp_size),
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
            *(
                prefill_graph
                if client.phase == "prefill"
                else ["--cuda-graph-backend-prefill", "disabled"]
            ),
            "--chunked-prefill-size",
            str(capacity if client.phase == "decode" or args.chunked_prefill else -1),
            "--max-prefill-tokens",
            str(capacity),
            "--max-running-requests",
            str(min(node.max_decode_tokens, capacity)),
            "--attention-backend",
            args.attention_backend,
            "--sampling-backend",
            args.sampling_backend,
        ]
        if args.kv_tokens is not None:
            command += ["--max-total-tokens", str(args.kv_tokens)]
        if getattr(args, "mem_fraction_static", None) is not None:
            command += ["--mem-fraction-static", str(args.mem_fraction_static)]
        if args.ib_device:
            command += ["--disaggregation-ib-device", args.ib_device]
        # Ordinary serving retains SGLang's CPU/GPU overlap. The diagnostic
        # capture processor reads CPU request history synchronously and is not
        # a performance configuration.
        if getattr(args, "disable_overlap_schedule", False) or getattr(
            args, "diagnostic_logit_processor", False
        ):
            command += ["--disable-overlap-schedule"]
        if getattr(args, "disable_flashinfer_autotune", False):
            command += ["--disable-flashinfer-autotune"]
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
    add_prefill_graph_args(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--bootstrap-port", type=int, default=19000)
    parser.add_argument("--nccl-port", type=int, default=20000)
    parser.add_argument(
        "--kv-tokens",
        type=kv_token_limit,
        default=None,
        help="Per-P/D KV token cap; default auto uses SGLang memory profiling",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=memory_fraction,
        help="Forward the native weight+KV memory budget; default is SGLang's default",
    )
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
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture decode forward (enabled by default)",
    )
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
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
    all_ports = (
        [args.port, args.port + 200]
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
            print(shlex.join(command))
        print(shlex.join(router_command))
        return
    spec = config.ModelSpec.from_model_path(node.model_path)
    node.validate(spec)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0,1,2,3,4,5,6,7"):
        parser.error(
            "Use identical full CUDA visibility (0,1,2,3,4,5,6,7) for this launcher"
        )
    if args.kv_tokens is not None and (
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

    stopping = False

    def stop(signum, frame):
        # A supervising harness may signal us after a child failure already
        # entered cleanup. Never interrupt that cleanup and orphan expert GPUs.
        if not stopping:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        spawn("experts", worker_command, base_env)
        wait_http(
            f"http://127.0.0.1:{args.port + 200}",
            children,
            time.monotonic() + node.startup_timeout_s,
        )
        for client, command in server_commands:
            spawn(f"{client.phase}-{client.id}", command, base_env)
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
        stopping = True
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
