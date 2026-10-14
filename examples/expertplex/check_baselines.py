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
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=("colocated", "chunked", "pdd", "pdmux"),
        default=["colocated", "chunked", "pdd", "pdmux"],
    )
    p.add_argument("--port", type=int, default=32000)
    p.add_argument("--pdmux-config")
    p.add_argument("--startup-timeout", type=int, default=600)
    args = p.parse_args()
    if "pdmux" in args.variants and not args.pdmux_config:
        p.error("PDMux requires an explicit tested partition file")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    report = {
        "scope": __doc__,
        "settings": vars(args) | {"output_dir": str(args.output_dir)},
        "communication_environment": {
            k: v for k, v in os.environ.items() if k.startswith(("NVSHMEM_", "NCCL_"))
        },
        "cases": [],
    }
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
