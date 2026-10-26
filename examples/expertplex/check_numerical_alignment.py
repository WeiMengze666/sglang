"""Same-weight, same-placement incremental full-vocabulary diagnostic.

Run within one gpuq job. CPU score capture disables scheduling overlap; these
results must not be used as performance measurements. Native reference uses
four expert ranks and DP attention/LM head to isolate arithmetic differences.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from check_e2e import PROMPTS, generate
from launch import wait_http


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["native", "expertplex"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--port-base", type=int, default=33000)
    p.add_argument("--startup-timeout", type=float, default=1800)
    a = p.parse_args()
    if a.variant == "expertplex" and not a.reference:
        p.error("ExpertPlex requires the native reference directory")
    if not 1024 <= a.port_base <= 63000:
        p.error("--port-base must leave space for bootstrap/NCCL ports up to +2001")
    if not 0 < a.startup_timeout < float("inf"):
        p.error("--startup-timeout must be finite and positive")
    a.output = a.output.resolve()
    if a.reference:
        a.reference = a.reference.resolve()
        if not (a.reference / "scores.json").is_file():
            p.error("--reference must contain native scores.json")
    a.output.mkdir(parents=True, exist_ok=False)
    model = str(a.model_path.resolve())
    config = json.loads((a.model_path / "config.json").read_text())
    if config["num_experts"] % 4:
        p.error("The matched reference requires four equal expert shards")
    env = dict(
        os.environ,
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK",
        MC_INTRANODE_NVLINK="true",
    )
    children = []
    report = {
        "scope": __doc__,
        "variant": a.variant,
        "status": "running",
        "commands": [],
    }

    def save():
        (a.output / "report.json").write_text(json.dumps(report, indent=2))

    def stop_signal(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_signal)

    def start(name, command, port):
        log = (a.output / (name + ".log")).open("w")
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((proc, log))
        report["commands"].append(command)
        save()
        wait_http(
            f"http://127.0.0.1:{port}",
            [(name, proc, log)],
            time.monotonic() + a.startup_timeout,
        )
        print("READY", name, flush=True)

    prefix = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--dtype",
        "bfloat16",
        "--skip-tokenizer-init",
        "--host",
        "127.0.0.1",
        "--attention-backend",
        "flashinfer",
        "--sampling-backend",
        "flashinfer",
        "--context-length",
        "4096",
        "--max-prefill-tokens",
        "512",
        "--max-running-requests",
        "16",
        "--max-total-tokens",
        "8192",
        "--mem-fraction-static",
        ".95",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "16",
        "--enable-two-batch-overlap",
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
    ]
    try:
        direct = []
        if a.variant == "native":
            placement = {
                "physical_to_logical_map": [
                    [
                        r + i * 4
                        for r in range(4)
                        for i in range(config["num_experts"] // 4)
                    ]
                ]
                * config["num_hidden_layers"]
            }
            path = a.output / "placement.json"
            path.write_text(json.dumps(placement))
            start(
                "native",
                prefix
                + [
                    "--port",
                    str(a.port_base),
                    "--nccl-port",
                    str(a.port_base + 1000),
                    "--tp-size",
                    "4",
                    "--ep-size",
                    "4",
                    "--attn-dp-size",
                    "4",
                    "--enable-dp-lm-head",
                    "--init-expert-location",
                    str(path),
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "auto",
                    "--moe-runner-backend",
                    "deep_gemm",
                    "--deepep-dispatcher-output-dtype",
                    "bf16",
                    "--chunked-prefill-size",
                    "-1",
                ],
                a.port_base,
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            outputs = [
                generate(f"http://127.0.0.1:{a.port_base}", tokenizer.encode(t), 32)
                for t in PROMPTS
            ]
            histories = a.output / "histories.json"
            histories.write_text(json.dumps({"model_path": model, "outputs": outputs}))
            url = f"http://127.0.0.1:{a.port_base}"
        else:
            start(
                "moe",
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    model,
                    "--dtype",
                    "bfloat16",
                    "--enable-expertplex",
                    "--expertplex-role",
                    "expert",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(a.port_base + 200),
                    "--max-prefill-tokens",
                    "512",
                    "--max-running-requests",
                    "16",
                    "--expertplex-startup-timeout",
                    str(a.startup_timeout),
                    "--enable-two-batch-overlap",
                ],
                a.port_base + 200,
            )
            for device, phase in [(0, "prefill"), (1, "decode")]:
                command = prefix + [
                    "--enable-expertplex",
                    "--expertplex-bootstrap-addr",
                    f"http://127.0.0.1:{a.port_base + 200}",
                    "--disaggregation-mode",
                    phase,
                    "--base-gpu-id",
                    str(device),
                    "--port",
                    str(a.port_base + device),
                    "--nccl-port",
                    str(a.port_base + 1000 + device),
                    "--disaggregation-bootstrap-port",
                    str(a.port_base + 2000 + device),
                    "--chunked-prefill-size",
                    "-1" if phase == "prefill" else "16",
                ]
                if phase == "prefill":
                    command[command.index("--cuda-graph-backend-decode") + 1] = (
                        "disabled"
                    )
                else:
                    command[command.index("--max-prefill-tokens") + 1] = "16"
                start(phase, command, a.port_base + device)
            histories = a.reference / "histories.json"
            url = f"http://127.0.0.1:{a.port_base + 1}"
            direct = [
                "--prefill-url",
                f"http://127.0.0.1:{a.port_base}",
                "--bootstrap-port",
                str(a.port_base + 2000),
                "--reference",
                str(a.reference / "scores.json"),
                "--require-exact-logprobs",
            ]
        command = [
            sys.executable,
            str(ROOT / "examples/expertplex/check_incremental.py"),
            "--base-url",
            url,
            "--model-path",
            model,
            "--histories",
            str(histories),
            "--capture-dir",
            str(a.output / "capture"),
            "--output",
            str(a.output / "scores.json"),
            "--steps",
            "32",
            "--top-k",
            "0",
            *direct,
        ]
        report["commands"].append(command)
        save()
        with (a.output / "capture.log").open("w") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        import psutil

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = []
        for proc, log in reversed(children):
            try:
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
                        proc.wait(timeout=10)
                for child in owned:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(owned, timeout=10)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=10)
                if alive:
                    cleanup_errors.append(str(alive))
            except (psutil.Error, OSError, subprocess.SubprocessError) as exc:
                cleanup_errors.append(repr(exc))
            finally:
                log.close()
        report["cleanup_errors"] = cleanup_errors
        save()
        if cleanup_errors:
            raise RuntimeError(cleanup_errors)


if __name__ == "__main__":
    main()
