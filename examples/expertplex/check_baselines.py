"""Serially smoke-test eight-GPU baseline deployments with one checkpoint.

This runs finite-generation checks, not paper accuracy or goodput acceptance.
Each launcher owns and cleans its process groups; no global process killer is
used. The output root must be new so failed runs are preserved.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--weight-mode", choices=("checkpoint", "synthetic"), default="checkpoint"
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=("colocated", "chunked", "pdd", "pdmux", "expertplex"),
        default=["colocated", "chunked", "pdd", "pdmux"],
    )
    p.add_argument("--port", type=int, default=32000)
    p.add_argument("--pdmux-config")
    p.add_argument("--expertplex-config")
    p.add_argument("--teacher-reference")
    p.add_argument("--teacher-flush-between", action="store_true")
    p.add_argument(
        "--teacher-histories",
        help="Also record fixed-history full-vocabulary diagnostics",
    )
    p.add_argument("--startup-timeout", type=int, default=600)
    args = p.parse_args()
    if "pdmux" in args.variants and not args.pdmux_config:
        p.error("PDMux requires an explicit tested partition file")
    if "expertplex" in args.variants and not args.expertplex_config:
        p.error("ExpertPlex requires its node configuration")
    for path in (
        args.teacher_histories,
        args.teacher_reference,
        args.expertplex_config,
    ):
        if path and not Path(path).is_file():
            p.error(f"Missing input file: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    report = {
        "scope": __doc__,
        "settings": vars(args) | {"output_dir": str(args.output_dir)},
        "communication_environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NVSHMEM_", "NCCL_", "SGLANG_EXPERTPLEX_"))
        },
        "cases": [],
    }
    if args.teacher_histories:
        report["settings"]["teacher_histories"] = str(args.teacher_histories)
    child = None

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for variant in args.variants:
            log_path = args.output_dir / f"{variant}-launch.log"
            command = [
                sys.executable,
                str(scripts / "launch_baseline.py"),
                "--variant",
                variant,
                "--model-path",
                args.model_path,
                "--load-format",
                "dummy" if args.weight_mode == "synthetic" else "safetensors",
                "--attention-backend",
                "triton",
                "--skip-tokenizer-init",
                "--sequence-cap",
                "4096",
                "--max-prefill-tokens",
                "2048",
                "--port",
                str(args.port),
                "--log-dir",
                str(args.output_dir / variant),
                "--startup-timeout",
                str(args.startup_timeout),
            ]
            if variant == "pdmux":
                command += ["--pdmux-config", args.pdmux_config]
            if variant == "expertplex":
                node = json.loads(Path(args.expertplex_config).read_text())
                if (
                    node["model_path"] != args.model_path
                    or node["weight_mode"] != args.weight_mode
                ):
                    raise ValueError(
                        "ExpertPlex must use the same BF16 fixture and weight mode"
                    )
                node["rendezvous_dir"] = str((args.output_dir / "rendezvous").resolve())
                node_path = args.output_dir / "node.json"
                node_path.write_text(json.dumps(node, indent=2) + "\n")
                command = [
                    sys.executable,
                    str(scripts / "launch.py"),
                    "--config",
                    str(node_path.resolve()),
                    "--attention-backend",
                    "triton",
                    "--sampling-backend",
                    "pytorch",
                    "--skip-tokenizer-init",
                    "--cuda-graph",
                    "--sequence-cap",
                    "4096",
                    "--kv-tokens",
                    "8192",
                    "--port",
                    str(args.port),
                    "--log-dir",
                    str(args.output_dir / variant),
                ]
            row = {"variant": variant, "command": command, "status": "starting"}
            report["cases"].append(row)
            print(f"Starting {variant}", flush=True)
            with log_path.open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + args.startup_timeout + 60
                    while time.monotonic() < deadline:
                        if child.poll() is not None:
                            raise RuntimeError(f"Launcher exited {child.returncode}")
                        # Launcher's readiness includes all workers and router;
                        # do not probe an unrelated pre-existing listener.
                        if (
                            f"Ready: http://127.0.0.1:{args.port}"
                            in log_path.read_text()
                        ):
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{args.port}/health", timeout=10
                            ) as response:
                                if response.status == 200:
                                    break
                        time.sleep(1)
                    else:
                        raise TimeoutError("Baseline startup deadline exceeded")
                    check = [
                        sys.executable,
                        str(scripts / "check_e2e.py"),
                        "--base-url",
                        f"http://127.0.0.1:{args.port}",
                        "--model-path",
                        args.model_path,
                        "--token-id-prompts",
                        "--output-tokens",
                        "128",
                        "--rounds",
                        "2",
                        "--prompt-repeat",
                        "3",
                        "--concurrency",
                        "12",
                        "--output",
                        str(args.output_dir / f"{variant}-e2e.json"),
                    ]
                    with (args.output_dir / f"{variant}-check.log").open("w") as output:
                        subprocess.run(
                            check,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            timeout=240,
                            check=True,
                        )
                    row["status"] = "functional_pass"
                    if args.teacher_histories:
                        teacher = [
                            sys.executable,
                            str(scripts / "check_teacher_forcing.py"),
                            "--base-url",
                            f"http://127.0.0.1:{args.port}",
                            "--model-path",
                            args.model_path,
                            "--histories",
                            args.teacher_histories,
                            "--output",
                            str(args.output_dir / f"{variant}-teacher.json"),
                            "--top-k",
                            "128",
                        ]
                        reference = (
                            Path(args.teacher_reference)
                            if args.teacher_reference
                            else args.output_dir / "colocated-teacher.json"
                        )
                        if args.teacher_flush_between:
                            teacher += ["--flush-between"]
                        if variant != "colocated" and reference.exists():
                            teacher += ["--reference", str(reference)]
                        with (args.output_dir / f"{variant}-teacher.log").open(
                            "w"
                        ) as output:
                            subprocess.run(
                                teacher,
                                stdout=output,
                                stderr=subprocess.STDOUT,
                                timeout=240,
                                check=True,
                            )
                        row["teacher_forcing_recorded"] = True
                except (
                    RuntimeError,
                    TimeoutError,
                    OSError,
                    subprocess.SubprocessError,
                ) as exc:
                    row.update(status="failed", error=str(exc))
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=90)
                    child = None
            print(row["variant"], row["status"], row.get("error", ""), flush=True)
            (args.output_dir / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=90)


if __name__ == "__main__":
    main()
