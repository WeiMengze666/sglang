"""Same-weight, same-placement incremental full-vocabulary diagnostic.

Run within one gpuq job. CPU score capture disables scheduling overlap; these
results must not be used as performance measurements. Native reference uses
the selected expert-rank count and DP attention/LM head to isolate arithmetic
differences. Reference and candidate must use the same expert placement.
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


def validate_native_reference_topology(expert_count, buffer_type):
    """Check the installed DeepEP configuration before loading model weights."""
    try:
        buffer_type.get_dispatch_config(expert_count)
        buffer_type.get_combine_config(expert_count)
    except (AssertionError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Installed DeepEP has no default dispatch/combine configuration for "
            f"{expert_count} EP ranks; this matched native reference is unavailable"
        ) from exc


def validate_reference(
    reference,
    model_path,
    tbo_enabled,
    steps=None,
    compile_enabled=False,
    expert_count=4,
    concurrency=1,
    deterministic=False,
):
    """Reject incomparable references before allocating GPU memory."""
    if not (reference / "scores.json").is_file():
        raise ValueError("Reference must contain native scores.json")
    report = json.loads((reference / "report.json").read_text())
    if report.get("variant") != "native" or report.get("status") != "passed":
        raise ValueError("Reference must be a successful native diagnostic")
    if report.get("deterministic_inference", False) != deterministic:
        raise ValueError("Reference deterministic inference mode differs")
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
    compiled = report.get("torch_compile_enabled")
    if compiled is None:
        compiled = any(
            "--enable-torch-compile" in cmd for cmd in report.get("commands", [])
        )
    if type(compiled) is not bool or compiled != compile_enabled:
        raise ValueError(
            "Reference torch.compile mode differs; generate a matched native reference"
        )
    # Reports written before this option always used four expert ranks.
    ranks = report.get("expert_count", 4)
    if type(ranks) is not int or ranks != expert_count:
        raise ValueError(
            "Reference expert count differs; generate a matched native reference"
        )
    if report.get("concurrency", 1) != concurrency:
        raise ValueError("Reference request concurrency differs")
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


def repeat_token_pattern(seed, lengths):
    if not seed:
        raise ValueError("Diagnostic token pattern must not be empty")
    return [
        (seed * ((length + len(seed) - 1) // len(seed)))[:length] for length in lengths
    ]


def validate_prompt_ids(prompts, vocab_size, max_prefill_tokens, context_length, steps):
    """Catch tokenizer/checkpoint mismatches before allocating GPU services."""
    if not prompts or any(
        not isinstance(tokens, list) or not tokens for tokens in prompts
    ):
        raise ValueError("Diagnostic prompts must be nonempty token lists")
    if any(
        type(t) is not int or not 0 <= t < vocab_size
        for tokens in prompts
        for t in tokens
    ):
        raise ValueError("Diagnostic prompt token is outside the model vocabulary")
    if any(
        len(tokens) > max_prefill_tokens or len(tokens) + steps >= context_length
        for tokens in prompts
    ):
        raise ValueError("Diagnostic prompts exceed prefill or context capacity")


def boundary_prompts(tokenizer, lengths):
    """Deterministic token prefixes crossing GEMM and attention tile boundaries."""
    seed = tokenizer.encode(
        "Explain how a hash table resolves collisions and give a Python example.\n",
        add_special_tokens=False,
    )
    return repeat_token_pattern(seed, lengths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["native", "expertplex"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        help="Matched native capture; native may replay it to check baseline repeatability",
    )
    p.add_argument("--model-path", type=Path, required=True)
    # Keep the HTTP, DP control (+233..239), bootstrap and NCCL ports
    # below Linux's usual ephemeral range (32768..60999). Otherwise this
    # process's own outbound connections can occupy a future control port.
    p.add_argument("--port-base", type=int, default=23000)
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--disable-tbo", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true")
    p.add_argument("--enable-deterministic-inference", action="store_true")
    p.add_argument(
        "--require-tbo-replays",
        type=int,
        default=0,
        help="Require this many completed nonempty two-child replays on all E ranks; diagnostic only",
    )
    p.add_argument("--expert-count", type=int, choices=range(1, 7), default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--batched-request", action="store_true")
    p.add_argument("--require-decode-batch-size", type=int, default=1)
    p.add_argument("--max-prefill-tokens", type=int, default=512)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        help="Native reference only: explicit token lengths for boundary tests",
    )
    p.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        help="Explicit repeated boundary-test pattern for tokenizer-free debug checkpoints; requires --prompt-lengths",
    )
    a = p.parse_args()
    if not 1 <= a.require_decode_batch_size <= a.concurrency <= 16:
        p.error("Require 1 <= decode batch size <= concurrency <= 16")
    if a.variant == "expertplex" and not a.reference:
        p.error("ExpertPlex requires the native reference directory")
    if a.require_tbo_replays < 0 or (
        a.require_tbo_replays
        and (a.variant != "expertplex" or a.disable_tbo or a.concurrency < 2)
    ):
        p.error(
            "TBO completion proof requires ExpertPlex, enabled TBO and concurrency >= 2"
        )
    if a.enable_torch_compile and not a.disable_tbo:
        p.error("The torch.compile diagnostic requires --disable-tbo")
    if a.steps < 2 or not 0 < a.max_prefill_tokens < a.context_length:
        p.error("Use steps >=2 and 0 < prefill capacity < context length")
    if a.prompt_lengths is not None:
        if a.variant != "native" or a.reference:
            p.error("Reference replay uses saved prompts; omit --prompt-lengths")
        if len(set(a.prompt_lengths)) != len(a.prompt_lengths) or any(
            n <= 0 or n > a.max_prefill_tokens or n + a.steps >= a.context_length
            for n in a.prompt_lengths
        ):
            p.error("Prompt lengths must be distinct, positive and fit both capacities")
    if a.prompt_token_ids is not None and not a.prompt_lengths:
        p.error("--prompt-token-ids requires native --prompt-lengths")
    if not 1024 <= a.port_base <= 63000:
        p.error("--port-base must leave space for bootstrap/NCCL ports up to +2001")
    if not 0 < a.startup_timeout < float("inf"):
        p.error("--startup-timeout must be finite and positive")
    a.output = a.output.resolve()
    if a.reference:
        a.reference = a.reference.resolve()
        try:
            validate_reference(
                a.reference,
                a.model_path,
                not a.disable_tbo,
                a.steps,
                a.enable_torch_compile,
                a.expert_count,
                a.concurrency,
                a.enable_deterministic_inference,
            )
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
    if a.prompt_token_ids is not None and any(
        not 0 <= token < config["vocab_size"] for token in a.prompt_token_ids
    ):
        p.error("Diagnostic token pattern is outside the model vocabulary")
    if config["num_experts"] % a.expert_count:
        p.error("The matched reference requires equal expert shards")
    prompts = None
    if a.variant == "native" and not a.reference:
        if a.prompt_token_ids is not None:
            prompts = repeat_token_pattern(a.prompt_token_ids, a.prompt_lengths)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            prompts = (
                boundary_prompts(tokenizer, a.prompt_lengths)
                if a.prompt_lengths
                else [tokenizer.encode(t) for t in PROMPTS]
            )
        try:
            validate_prompt_ids(
                prompts,
                config["vocab_size"],
                a.max_prefill_tokens,
                a.context_length,
                a.steps,
            )
        except ValueError as exc:
            p.error(str(exc))
    env = dict(
        os.environ,
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK",
        MC_INTRANODE_NVLINK="true",
    )
    if a.require_tbo_replays:
        env["SGLANG_TBO_DEBUG"] = "1"
    children = []
    report = {
        "scope": __doc__,
        "variant": a.variant,
        "concurrency": a.concurrency,
        "batched_request": a.batched_request,
        "required_decode_batch_size": a.require_decode_batch_size,
        "required_tbo_replays": a.require_tbo_replays,
        "tbo_enabled": not a.disable_tbo,
        "torch_compile_enabled": a.enable_torch_compile,
        "deterministic_inference": a.enable_deterministic_inference,
        "expert_count": a.expert_count,
        "steps": a.steps,
        "max_prefill_tokens": a.max_prefill_tokens,
        "context_length": a.context_length,
        "prompt_lengths": a.prompt_lengths,
        "prompt_token_ids": a.prompt_token_ids,
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
    if a.enable_torch_compile:
        prefix += ["--enable-torch-compile", "--torch-compile-max-bs", "16"]
    if a.enable_deterministic_inference:
        prefix.append("--enable-deterministic-inference")
    try:
        direct = []
        if a.variant == "native":
            from deep_ep import Buffer

            validate_native_reference_topology(a.expert_count, Buffer)
            placement = {
                "physical_to_logical_map": [
                    [
                        r + i * a.expert_count
                        for r in range(a.expert_count)
                        for i in range(config["num_experts"] // a.expert_count)
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
                    str(a.expert_count),
                    "--ep-size",
                    str(a.expert_count),
                    "--attn-dp-size",
                    str(a.expert_count),
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
            if a.reference:
                histories = a.output / "histories.json"
                histories.write_bytes((a.reference / "histories.json").read_bytes())
            else:
                outputs = [
                    generate(f"http://127.0.0.1:{a.port_base}", tokens, a.steps)
                    for tokens in prompts
                ]
                histories = a.output / "histories.json"
                histories.write_text(
                    json.dumps({"model_path": model, "outputs": outputs})
                )
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
                    "--expertplex-prefill-count",
                    "1",
                    "--expertplex-decode-count",
                    str(7 - a.expert_count),
                    "--expertplex-expert-count",
                    str(a.expert_count),
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
                    if a.enable_torch_compile:
                        # Only decode is compiled, matching SGLang's decode graph runner.
                        command.remove("--enable-torch-compile")
                        i = command.index("--torch-compile-max-bs")
                        del command[i : i + 2]
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
            ]
        if a.reference:
            direct += [
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
            "--concurrency",
            str(a.concurrency),
            "--require-decode-batch-size",
            str(a.require_decode_batch_size),
            "--top-k",
            "0",
            *direct,
        ]
        if a.batched_request:
            command.append("--batched-request")
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
        if a.require_tbo_replays:
            from check_tbo_completion import check

            check(
                a.output, expert_count=a.expert_count, min_replays=a.require_tbo_replays
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
