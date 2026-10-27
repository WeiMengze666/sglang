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


def validate_reference(reference, model_path, tbo_enabled, steps=None):
    """Reject incomparable references before allocating GPU memory."""
    if not (reference / "scores.json").is_file():
        raise ValueError("Reference must contain native scores.json")
    report = json.loads((reference / "report.json").read_text())
    if report.get("variant") != "native" or report.get("status") != "passed":
        raise ValueError("Reference must be a successful native diagnostic")
    enabled = report.get("tbo_enabled")
    if enabled is None:
        # Backward compatibility with the original diagnostic report format.
        launches = [
            cmd for cmd in report.get("commands", []) if "sglang.launch_server" in cmd
        ]
        if len(launches) != 1:
            raise ValueError("Cannot determine reference overlap configuration")
        enabled = "--enable-two-batch-overlap" in launches[0]
    if type(enabled) is not bool or enabled != tbo_enabled:
        raise ValueError(
            "Reference TBO mode differs; generate a matched native reference"
        )
    histories = json.loads((reference / "histories.json").read_text())
    if Path(histories["model_path"]).resolve() != model_path.resolve():
        raise ValueError("Reference model path differs")
    if steps is not None:
        cases = json.loads((reference / "scores.json").read_text()).get("cases", [])
        if not cases or any(case.get("steps") != steps for case in cases):
            raise ValueError("Reference score capture uses a different step count")


def validate_workload(histories, steps, max_prefill_tokens, context_length):
    """Reject truncated or oversized trajectories before starting GPU services."""
    if not histories.get("outputs"):
        raise ValueError("Reference requires nonempty trajectories")
    for output in histories["outputs"]:
        prompt = output.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("Each trajectory requires explicit prompt token IDs")
        if len(prompt) > max_prefill_tokens or len(prompt) + steps >= context_length:
            raise ValueError("Reference prompt exceeds prefill or context capacity")
        continuation = output.get("meta_info", {}).get("output_token_logprobs", [])
        if len(continuation) < steps:
            raise ValueError("Reference continuation is shorter than requested steps")


def boundary_prompts(tokenizer, lengths):
    """Deterministic token prefixes crossing GEMM and attention tile boundaries."""
    seed = tokenizer.encode(
        "Explain how a hash table resolves collisions and give a Python example.\n",
        add_special_tokens=False,
    )
    if not seed:
        raise ValueError("Tokenizer returned an empty diagnostic seed")
    return [
        (seed * ((length + len(seed) - 1) // len(seed)))[:length] for length in lengths
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["native", "expertplex"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--port-base", type=int, default=33000)
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--disable-tbo", action="store_true")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--max-prefill-tokens", type=int, default=512)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        help="Native reference only: explicit token lengths for boundary tests",
    )
    a = p.parse_args()
    if a.variant == "expertplex" and not a.reference:
        p.error("ExpertPlex requires the native reference directory")
    if a.variant == "native" and a.reference:
        p.error("--reference applies to the ExpertPlex diagnostic")
    if a.steps < 2 or not 0 < a.max_prefill_tokens < a.context_length:
        p.error("Use steps >=2 and 0 < prefill capacity < context length")
    if a.prompt_lengths is not None:
        if a.variant != "native":
            p.error("ExpertPlex replays the reference prompts; omit --prompt-lengths")
        if len(set(a.prompt_lengths)) != len(a.prompt_lengths) or any(
            n <= 0 or n > a.max_prefill_tokens or n + a.steps >= a.context_length
            for n in a.prompt_lengths
        ):
            p.error("Prompt lengths must be distinct, positive and fit both capacities")
    if not 1024 <= a.port_base <= 63000:
        p.error("--port-base must leave space for bootstrap/NCCL ports up to +2001")
    if not 0 < a.startup_timeout < float("inf"):
        p.error("--startup-timeout must be finite and positive")
    a.output = a.output.resolve()
    if a.reference:
        a.reference = a.reference.resolve()
        try:
            validate_reference(a.reference, a.model_path, not a.disable_tbo, a.steps)
            validate_workload(
                json.loads((a.reference / "histories.json").read_text()),
                a.steps,
                a.max_prefill_tokens,
                a.context_length,
            )
        except (OSError, ValueError, KeyError) as exc:
            p.error(str(exc))
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
        "tbo_enabled": not a.disable_tbo,
        "steps": a.steps,
        "max_prefill_tokens": a.max_prefill_tokens,
        "context_length": a.context_length,
        "prompt_lengths": a.prompt_lengths,
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
        str(a.context_length),
        "--max-prefill-tokens",
        str(a.max_prefill_tokens),
        "--max-running-requests",
        "16",
        "--max-total-tokens",
        str(max(8192, a.context_length)),
        "--mem-fraction-static",
        ".95",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "16",
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
    ]
    if not a.disable_tbo:
        prefix.append("--enable-two-batch-overlap")
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
            prompts = (
                boundary_prompts(tokenizer, a.prompt_lengths)
                if a.prompt_lengths
                else [tokenizer.encode(t) for t in PROMPTS]
            )
            if any(
                len(tokens) > a.max_prefill_tokens
                or len(tokens) + a.steps >= a.context_length
                for tokens in prompts
            ):
                raise ValueError(
                    "Diagnostic prompts exceed prefill or context capacity"
                )
            outputs = [
                generate(f"http://127.0.0.1:{a.port_base}", tokens, a.steps)
                for tokens in prompts
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
                    str(a.max_prefill_tokens),
                    "--max-running-requests",
                    "16",
                    "--expertplex-startup-timeout",
                    str(a.startup_timeout),
                    *([] if a.disable_tbo else ["--enable-two-batch-overlap"]),
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
            str(a.steps),
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
