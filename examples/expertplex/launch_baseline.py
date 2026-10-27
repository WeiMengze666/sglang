"""Launch the paper's single-node baseline topologies on eight GPUs.

PDMux uses this pinned SGLang implementation of Green Context scheduling;
it is not claimed to be the authors' unpublished MoE modifications to MuxWise.
Record the exact configuration, tune on a separate workload, then freeze it.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from launch import kv_token_limit, memory_fraction, wait_http


def commands(args):
    pdd = args.variant == "pdd"
    tp = 4 if pdd else 8
    attention_dp = 1 if args.variant == "pdmux" else tp
    kv = (
        args.decode_kv_tokens // attention_dp
        if args.decode_kv_tokens is not None
        else None
    )
    common = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--dtype",
        "bfloat16",
        "--load-format",
        args.load_format,
        "--tp-size",
        str(tp),
        "--ep-size",
        str(tp),
        "--attn-dp-size",
        str(attention_dp),
        "--dp-size",
        "1",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--moe-runner-backend",
        "deep_gemm",
        "--deepep-dispatcher-output-dtype",
        "bf16",
        "--attention-backend",
        args.attention_backend,
        "--sampling-backend",
        getattr(args, "sampling_backend", "flashinfer"),
        "--max-running-requests",
        str(args.max_running),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--context-length",
        str(args.sequence_cap),
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        str(args.graph_max_bs),
        "--chunked-prefill-size",
        str(args.chunk_size if args.variant == "chunked" else -1),
        "--host",
        "127.0.0.1",
    ]
    if kv is not None:
        common += ["--max-total-tokens", str(kv)]
    if getattr(args, "mem_fraction_static", None) is not None:
        common += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if getattr(args, "deepep_config", None):
        common += ["--deepep-config", args.deepep_config]
    if getattr(args, "enable_dp_lm_head", False):
        if attention_dp <= 1:
            raise ValueError("DP LM head requires a baseline with DP attention")
        common += ["--enable-dp-lm-head"]
    if args.skip_tokenizer_init:
        common += ["--skip-tokenizer-init"]
    if args.tokenizer_path:
        common += ["--tokenizer-path", args.tokenizer_path]
    if getattr(args, "diagnostic_logit_processor", False):
        common += ["--enable-custom-logit-processor"]
    if getattr(args, "disable_flashinfer_autotune", False):
        common += ["--disable-flashinfer-autotune"]
    if args.variant != "pdmux" and (
        getattr(args, "disable_overlap_schedule", False)
        or getattr(args, "diagnostic_logit_processor", False)
    ):
        common += ["--disable-overlap-schedule"]
    if args.variant == "pdmux":
        common += [
            "--enable-pdmux",
            "--disable-overlap-schedule",
            "--pdmux-config-path",
            args.pdmux_config,
        ]
    elif not args.disable_tbo:
        common += ["--enable-two-batch-overlap"]
    if not pdd:
        return [(args.variant, common + ["--port", str(args.port)], args.port)]
    servers = []
    for role, gpu, port in [
        ("prefill", 0, args.port + 1),
        # DP attention derives a block of control ports from the HTTP port.
        # Adjacent HTTP ports therefore collide even with distinct NCCL ports.
        ("decode", 4, args.port + 101),
    ]:
        command = common + [
            "--base-gpu-id",
            str(gpu),
            "--port",
            str(port),
            "--nccl-port",
            str(args.nccl_port + gpu),
            "--disaggregation-mode",
            role,
            "--disaggregation-transfer-backend",
            "mooncake",
            "--disaggregation-bootstrap-port",
            str(args.bootstrap_port + gpu),
        ]
        if role == "prefill":
            # Only the fixed-capacity control run bounds P's transient pool.
            # Auto mode lets each phase use SGLang's memory profiler.
            if kv is not None:
                command[command.index("--max-total-tokens") + 1] = str(
                    max(args.sequence_cap, args.max_prefill_tokens)
                )
            command[command.index("--cuda-graph-backend-decode") + 1] = "disabled"
        servers.append((role, command, port))
    router = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--prefill",
        f"http://127.0.0.1:{args.port + 1}",
        str(args.bootstrap_port),
        "--decode",
        f"http://127.0.0.1:{args.port + 101}",
    ]
    return servers + [("router", router, args.port)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", choices=("colocated", "chunked", "pdd", "pdmux"), required=True
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path")
    p.add_argument(
        "--load-format", choices=("safetensors", "dummy"), default="safetensors"
    )
    p.add_argument("--skip-tokenizer-init", action="store_true")
    p.add_argument(
        "--diagnostic-logit-processor",
        action="store_true",
        help="Enable fixed-trajectory score capture on isolated diagnostic servers only",
    )
    p.add_argument(
        "--attention-backend", choices=("triton", "flashinfer"), default="flashinfer"
    )
    p.add_argument(
        "--sampling-backend", choices=("flashinfer", "pytorch"), default="flashinfer"
    )
    p.add_argument(
        "--decode-kv-tokens",
        type=kv_token_limit,
        default=None,
        help="Aggregate decode KV cap across attention DP ranks; default auto profiles memory",
    )
    p.add_argument("--mem-fraction-static", type=memory_fraction)
    p.add_argument("--deepep-config", help="Native DeepEP tuned JSON string or file")
    p.add_argument(
        "--sequence-cap",
        type=int,
        default=8192,
        help="Common PDD-feasible prompt+output cap, fixed before comparing systems",
    )
    p.add_argument("--max-prefill-tokens", type=int, default=8192)
    p.add_argument("--max-running", type=int, default=128)
    p.add_argument("--graph-max-bs", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--disable-tbo", action="store_true")
    p.add_argument(
        "--enable-dp-lm-head",
        action="store_true",
        help="Use native DP LM head; tune and freeze with the baseline",
    )
    p.add_argument("--disable-overlap-schedule", action="store_true")
    p.add_argument("--disable-flashinfer-autotune", action="store_true")
    p.add_argument(
        "--pdmux-config", help="Explicit B300-tested Green Context divisions YAML"
    )
    p.add_argument("--port", type=int, default=31000)
    p.add_argument("--bootstrap-port", type=int, default=19000)
    p.add_argument("--nccl-port", type=int, default=22000)
    p.add_argument("--log-dir", required=True, type=Path)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.decode_kv_tokens is not None and args.decode_kv_tokens % 24:
        p.error("Aggregate KV token count must be a positive multiple of 24 (DP 3/4/8)")
    if (
        min(
            args.sequence_cap,
            args.max_prefill_tokens,
            args.max_running,
            args.graph_max_bs,
        )
        <= 0
    ):
        p.error("Capacities must be positive")
    if args.graph_max_bs > args.max_running:
        p.error("Graph capacity exceeds maximum running requests")
    if args.variant == "pdmux" and not args.pdmux_config:
        p.error(
            "PDMux requires explicit B300-tested divisions; do not assume H800 defaults"
        )
    planned = commands(args)
    for name, command, _ in planned:
        print(name + ": " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0,1,2,3,4,5,6,7"):
        p.error("Expose all eight GPUs in their original order")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("SGLANG_EXPERTPLEX_CONFIG", None)
    env.pop("SGLANG_EXPERTPLEX_CLIENT_ID", None)
    env["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "INTRA_NODE_NVLINK"
    env["MC_INTRANODE_NVLINK"] = "true"
    children = []
    stopping = False

    def stop(*_):
        if not stopping:
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    manifest = vars(args) | {
        "log_dir": str(args.log_dir),
        "commands": planned,
        "scope": __doc__,
        "processes": [],
    }
    try:
        for name, command, port in planned:
            log = (args.log_dir / f"{name}.log").open("w")
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append((name, process, log))
            manifest["processes"].append({"name": name, "pid": process.pid})
            (args.log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            wait_http(
                f"http://127.0.0.1:{port}",
                children,
                time.monotonic() + args.startup_timeout,
                expected_workers=2 if name == "router" else None,
            )
        print(f"Ready: http://127.0.0.1:{args.port}", flush=True)
        while all(process.poll() is None for _, process, _ in children):
            time.sleep(0.5)
        raise RuntimeError("Baseline child exited; inspect the saved logs")
    except KeyboardInterrupt:
        pass
    finally:
        stopping = True
        for _, process, _ in reversed(children):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process, log in children:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()


if __name__ == "__main__":
    main()
