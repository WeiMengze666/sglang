"""Run one isolated serving deployment and a frozen Poisson/SLO sweep.

Synthetic checkpoints establish execution/performance diagnostics only. They
cannot establish pretrained-model accuracy, realistic routing or acceptance.
Every launch/benchmark command and failure is saved; only owned processes are
stopped. Use separate output directories for tuning and held-out evaluation.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=("expertplex", "colocated", "chunked", "pdd", "pdmux"),
        required=True,
    )
    p.add_argument(
        "--label", help="Explicit experiment label for layout/overlap ablations"
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--config", help="ExpertPlex node configuration")
    p.add_argument("--pdmux-config")
    p.add_argument("--weight-mode", choices=("checkpoint", "synthetic"), required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--ttft-slo-ms", type=float, required=True)
    p.add_argument("--tpot-slo-ms", type=float, required=True)
    p.add_argument("--sequence-cap", type=int, default=24576)
    p.add_argument("--decode-kv-tokens", type=int, default=98304)
    p.add_argument(
        "--attention-backend", choices=("triton", "flashinfer"), default="flashinfer"
    )
    p.add_argument(
        "--sampling-backend", choices=("flashinfer", "pytorch"), default="flashinfer"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable-dp-lm-head",
        action="store_true",
        help="Native DP-attention baseline tuning option",
    )
    p.add_argument(
        "--warmup-requests",
        type=int,
        default=8,
        help="Untimed warmup requests per rate; use the same count for all layouts",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="ChunkedPrefill tuning parameter; freeze before held-out evaluation",
    )
    p.add_argument(
        "--disable-tbo",
        action="store_true",
        help="Baseline ablation; ExpertPlex uses config.overlap",
    )
    p.add_argument(
        "--canonical-synthetic-seed",
        type=int,
        help="Use identical globally named random BF16 tensors in every layout",
    )
    p.add_argument("--port", type=int, default=31000)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--benchmark-timeout", type=int, default=3600)
    p.add_argument(
        "--numerical-check",
        action="store_true",
        help="After timing, record deterministic generation and fixed-history top-128 scores",
    )
    p.add_argument(
        "--numerical-reference-dir",
        type=Path,
        help="Earlier same-weight run's numerical directory; requires --numerical-check",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    if args.enable_dp_lm_head and args.variant in ("expertplex", "pdmux"):
        p.error("DP LM head tuning applies to colocated/chunked/PDD baselines")
    if args.warmup_requests < 1:
        p.error("Warmup requests must be positive")
    if args.numerical_reference_dir and not args.numerical_check:
        p.error("Numerical reference requires --numerical-check")
    if args.numerical_reference_dir:
        for name in ("generation.json", "teacher.json"):
            if not (args.numerical_reference_dir / name).is_file():
                p.error(f"Missing numerical reference {name}")
        provenance = args.numerical_reference_dir.parent / "experiment.json"
        if not provenance.is_file():
            p.error("Numerical reference requires its original experiment.json")
        reference_settings = json.loads(provenance.read_text())["settings"]
        for field in ("model_path", "weight_mode", "canonical_synthetic_seed"):
            if reference_settings.get(field) != getattr(args, field):
                p.error(f"Numerical reference differs in {field}")
    if args.variant == "expertplex" and not args.config:
        p.error("ExpertPlex requires --config")
    if args.variant == "pdmux" and not args.pdmux_config:
        p.error("PDMux requires tested --pdmux-config")
    if args.chunk_size < 1:
        p.error("Chunk size must be positive")
    if args.variant == "expertplex" and args.disable_tbo:
        p.error("Use an explicit ExpertPlex node config with overlap=none")
    if args.canonical_synthetic_seed is not None:
        if args.weight_mode != "synthetic" or args.canonical_synthetic_seed < 0:
            p.error(
                "Canonical synthetic seed requires synthetic mode and a nonnegative seed"
            )
        os.environ["SGLANG_EXPERTPLEX_SYNTHETIC_SEED"] = str(
            args.canonical_synthetic_seed
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Freeze tuning inputs before launch. A later experiment may select another
    # node/Green Context policy at the same source path.
    for field, filename in (("config", "node.json"), ("pdmux_config", "pdmux.yaml")):
        if original := getattr(args, field):
            snapshot = args.output_dir.resolve() / filename
            snapshot.write_bytes(Path(original).read_bytes())
            setattr(args, field, str(snapshot))
    scripts = Path(__file__).resolve().parent
    common = [
        "--port",
        str(args.port),
        "--tokenizer-path",
        args.tokenizer_path,
        "--attention-backend",
        args.attention_backend,
        "--sampling-backend",
        args.sampling_backend,
        "--sequence-cap",
        str(args.sequence_cap),
        "--log-dir",
        str(args.output_dir / "service"),
    ]
    if args.variant == "expertplex":
        node = json.loads(Path(args.config).read_text())
        if (
            node["model_path"] != args.model_path
            or node["weight_mode"] != args.weight_mode
        ):
            p.error("Node config must match model path and weight mode")
        decode_count = sum(c["phase"] == "decode" for c in node["clients"])
        if args.decode_kv_tokens % decode_count:
            p.error("Aggregate KV capacity must divide across D workers")
        if node["max_prefill_tokens"] < args.sequence_cap:
            p.error("P mailbox must cover the common input+output length cap")
        if (
            args.canonical_synthetic_seed is not None
            and node.get("synthetic_seed", 1234) != args.canonical_synthetic_seed
        ):
            p.error("Node synthetic seed differs from the canonical seed")
        node["rendezvous_dir"] = str((args.output_dir / "rendezvous").resolve())
        node_path = args.output_dir / "node.json"
        node_path.write_text(json.dumps(node, indent=2) + "\n")
        launch = [
            sys.executable,
            str(scripts / "launch.py"),
            "--config",
            str(node_path.resolve()),
            "--kv-tokens",
            str(args.decode_kv_tokens // decode_count),
            "--cuda-graph",
            *common,
        ]
    else:
        launch = [
            sys.executable,
            str(scripts / "launch_baseline.py"),
            "--variant",
            args.variant,
            "--model-path",
            args.model_path,
            "--load-format",
            "dummy" if args.weight_mode == "synthetic" else "safetensors",
            "--max-prefill-tokens",
            str(args.sequence_cap),
            "--decode-kv-tokens",
            str(args.decode_kv_tokens),
            "--startup-timeout",
            str(args.startup_timeout),
            *common,
        ]
        if args.pdmux_config:
            launch += ["--pdmux-config", args.pdmux_config]
        if args.variant == "chunked":
            launch += ["--chunk-size", str(args.chunk_size)]
        if args.disable_tbo:
            launch += ["--disable-tbo"]
        if args.enable_dp_lm_head:
            launch += ["--enable-dp-lm-head"]
    bench = [
        sys.executable,
        str(scripts / "bench_sweep.py"),
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.tokenizer_path,
        "--weight-mode",
        args.weight_mode,
        "--label",
        args.label or args.variant,
        "--dataset-name",
        "custom",
        "--dataset-path",
        args.dataset_path,
        "--sequence-cap",
        str(args.sequence_cap),
        "--num-prompts",
        str(args.requests),
        "--warmup-requests",
        str(args.warmup_requests),
        "--rates",
        *map(str, args.rates),
        "--ttft-slo-ms",
        str(args.ttft_slo_ms),
        "--tpot-slo-ms",
        str(args.tpot_slo_ms),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir / "sweep"),
    ]
    repo = scripts.parents[1]
    source_paths = (
        list((repo / "python/sglang/srt/expertplex").rglob("*.py"))
        + list((repo / "python/sglang/srt/expertplex/csrc").glob("*"))
        + [
            repo / "python/sglang/srt/models/qwen3_moe.py",
            repo / "python/sglang/srt/model_loader/loader.py",
            scripts / "launch.py",
            scripts / "launch_baseline.py",
            scripts / "bench_sweep.py",
            scripts / "check_e2e.py",
            scripts / "check_teacher_forcing.py",
            Path(__file__),
        ]
    )
    source_hashes = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
        if p.is_file() and not p.name.startswith(".")
    }
    hardware = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,pstate",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
    )
    report = {
        "scope": __doc__,
        "serving_defaults_version": "native-overlap-autotune-apk-pipeline-v2",
        "input_artifact_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(args.dataset_path),
                Path(args.model_path) / "config.json",
                Path(args.tokenizer_path) / "tokenizer.json",
                *([Path(args.config)] if args.config else []),
                *([Path(args.pdmux_config)] if args.pdmux_config else []),
            )
            if path.is_file()
        },
        "source_files_sha256": source_hashes,
        "hardware_at_start": hardware.stdout
        if hardware.returncode == 0
        else hardware.stderr,
        "settings": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "numerical_reference_dir": str(args.numerical_reference_dir)
            if args.numerical_reference_dir
            else None,
        },
        "launch_command": launch,
        "benchmark_command": bench,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NVSHMEM_", "NCCL_", "SGLANG_", "DG_"))
        },
        "status": "starting",
    }

    def save():
        (args.output_dir / "experiment.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    service = benchmark = None
    save()
    try:
        with (args.output_dir / "launcher.log").open("w") as log:
            service = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
            report["launcher_pid"] = service.pid
            save()
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if service.poll() is not None:
                    raise RuntimeError(f"Service launcher exited {service.returncode}")
                if (
                    f"Ready: http://127.0.0.1:{args.port}"
                    in (args.output_dir / "launcher.log").read_text()
                ):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Startup deadline exceeded")
            report["status"] = "benchmarking"
            save()
            with (args.output_dir / "benchmark.log").open("w") as bench_log:
                benchmark = subprocess.Popen(
                    bench,
                    stdout=bench_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = benchmark.wait(timeout=args.benchmark_timeout)
                if code:
                    raise RuntimeError(f"Benchmark exited {code}")
            if args.numerical_check:
                # Run only after all timed sweeps, so these histories cannot
                # change the benchmark's prefix-cache state or arrival stream.
                numerical = args.output_dir / "numerical"
                numerical.mkdir()
                generation = [
                    sys.executable,
                    str(scripts / "check_e2e.py"),
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--model-path",
                    args.model_path,
                    "--token-id-prompts",
                    "--output-tokens",
                    "32",
                    "--rounds",
                    "1",
                    "--prompt-repeat",
                    "1",
                    "--concurrency",
                    "4",
                    "--output",
                    str(numerical / "generation.json"),
                ]
                teacher = [
                    sys.executable,
                    str(scripts / "check_teacher_forcing.py"),
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--model-path",
                    args.model_path,
                    "--histories",
                    str(
                        (args.numerical_reference_dir or numerical) / "generation.json"
                    ),
                    "--steps",
                    "0",
                    "1",
                    "7",
                    "31",
                    "--top-k",
                    "128",
                    "--flush-between",
                    "--output",
                    str(numerical / "teacher.json"),
                ]
                if args.numerical_reference_dir:
                    generation += [
                        "--reference",
                        str(args.numerical_reference_dir / "generation.json"),
                    ]
                    teacher += [
                        "--reference",
                        str(args.numerical_reference_dir / "teacher.json"),
                    ]
                report["numerical_commands"] = [generation, teacher]
                save()
                for name, command in (("generation", generation), ("teacher", teacher)):
                    with (numerical / (name + ".log")).open("w") as numerical_log:
                        benchmark = subprocess.Popen(
                            command,
                            stdout=numerical_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        if benchmark.wait(timeout=600):
                            raise RuntimeError(f"Numerical diagnostic {name} failed")
            report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        if benchmark is not None and benchmark.poll() is None:
            os.killpg(benchmark.pid, signal.SIGTERM)
            try:
                benchmark.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(benchmark.pid, signal.SIGKILL)
                benchmark.wait(timeout=10)
        if service is not None and service.poll() is None:
            service.terminate()  # Launcher cleans its own worker groups.
            service.wait(timeout=90)
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
