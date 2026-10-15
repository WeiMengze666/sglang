"""Poisson arrival sweep with per-request joint TTFT/TPOT SLO accounting."""

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


def joint_slo(result, ttft_slo_s, tpot_slo_s):
    names = ("ttfts", "latencies", "successes", "output_lens")
    arrays = [result[name] for name in names]
    count = len(arrays[0])
    if count == 0 or any(len(values) != count for values in arrays):
        raise ValueError("Benchmark must contain aligned, nonempty per-request details")
    duration = result["duration"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid benchmark duration")
    if any(not math.isfinite(slo) or slo <= 0 for slo in (ttft_slo_s, tpot_slo_s)):
        raise ValueError("SLOs must be finite and positive")
    met = 0
    unscorable = 0
    for ttft, latency, success, length in zip(*arrays):
        if not success:
            continue
        # A one-token response has no TPOT measurement and cannot establish
        # that the joint SLO was met. The benchmark ignores EOS by default;
        # use output length >=2.
        if length < 2:
            unscorable += 1
            continue
        if (
            not all(math.isfinite(x) for x in (ttft, latency))
            or ttft <= 0
            or latency < ttft
        ):
            raise ValueError("Invalid successful request timing")
        tpot = (latency - ttft) / (length - 1)
        met += ttft <= ttft_slo_s and tpot <= tpot_slo_s
    return {
        "request_rate": result["request_rate"],
        "attempted_requests": count,
        "completed_requests": sum(bool(x) for x in result["successes"]),
        "joint_slo_met_requests": met,
        "joint_slo_attainment": met / count,
        "observed_slo_goodput_rps": met / duration,
        "unscorable_one_token_responses": unscorable,
        "meets_p90_joint_slo": met / count >= 0.9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--weight-mode",
        choices=("checkpoint", "synthetic", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--sequence-cap", type=int, default=8192)
    parser.add_argument(
        "--token-id-prompts",
        action="store_true",
        help="Random dataset only; permits skip-tokenizer-init servers",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--dataset-name", choices=("random", "sharegpt", "custom"), default="random"
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup_requests < 1:
        parser.error("Warmup requests must be positive")
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.rates):
        parser.error("Arrival rates must be finite and positive")
    if args.num_prompts < 1 or args.output_len < 2 or args.input_len < 1:
        parser.error("Use positive prompt/input counts and output length >=2")
    if args.ttft_slo_ms <= 0 or args.tpot_slo_ms <= 0:
        parser.error("SLOs must be positive")
    if args.dataset_name != "random" and not args.dataset_path:
        parser.error("A local dataset path is required for ShareGPT/custom runs")
    if args.sequence_cap < 3:
        parser.error("Sequence cap must permit both prompt and output")
    if (
        args.dataset_name == "random"
        and args.input_len + args.output_len > args.sequence_cap
    ):
        parser.error("Random input+output exceeds the common sequence cap")
    if args.token_id_prompts and args.dataset_name != "random":
        parser.error(
            "The pinned SGLang ShareGPT/custom loader does not support token-ID prompts"
        )
    output = Path(args.output_dir)
    results = []
    for index, rate in enumerate(sorted(set(args.rates))):
        path = output / f"rate-{index}-{rate:g}.jsonl"
        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang",
            "--base-url",
            args.base_url,
            "--model",
            args.model_path,
            "--dataset-name",
            args.dataset_name,
            "--num-prompts",
            str(args.num_prompts),
            "--warmup-requests",
            str(args.warmup_requests),
            "--random-input-len",
            str(args.input_len),
            "--random-output-len",
            str(args.output_len),
            "--random-range-ratio",
            "0",
            "--request-rate",
            str(rate),
            "--seed",
            str(args.seed),
            "--output-details",
            "--flush-cache",
            "--output-file",
            str(path),
        ]
        if args.dataset_path:
            command += ["--dataset-path", args.dataset_path]
        if args.tokenizer_path:
            command += ["--tokenizer", args.tokenizer_path]
        if args.dataset_name != "random":
            command += ["--sharegpt-context-len", str(args.sequence_cap)]
        if args.token_id_prompts:
            command += ["--tokenize-prompt"]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"Use a fresh output directory; refusing to mix benchmark runs: {path}"
            )
        subprocess.run(command, check=True)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(
                "Expected exactly one detailed benchmark record per arrival rate"
            )
        detailed = json.loads(lines[0])
        if len(detailed["successes"]) != args.num_prompts:
            raise ValueError("Dataset did not supply the requested number of requests")
        measured = joint_slo(detailed, args.ttft_slo_ms / 1000, args.tpot_slo_ms / 1000)
        measured["details_file"] = str(path.resolve())
        results.append(measured)
    if args.dry_run:
        return
    feasible = [row["request_rate"] for row in results if row["meets_p90_joint_slo"]]
    report = {
        "label": args.label,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "weight_mode": args.weight_mode,
        "sequence_cap": args.sequence_cap,
        "dataset": args.dataset_name,
        "dataset_path": args.dataset_path,
        "seed": args.seed,
        "warmup_requests": args.warmup_requests,
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "p90_goodput_max_tested_arrival_rps": max(feasible, default=None),
        "note": "Maximum among tested arrival rates only; extend/refine the sweep and repeat with multiple seeds.",
        "results": results,
    }
    (output / "joint-slo.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
