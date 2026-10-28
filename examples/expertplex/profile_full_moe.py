"""Profile full APK layers, including raw-P Scatter1/Scatter2/Gather.

Run inside one GPU allocation. This uses one E rank with synthetic BF16 weights
and random top-k routes, not a pretrained routing distribution. Tracing is off.
The E-clock admission-to-retirement interval includes the scheduler and all
configured MoE operations, but excludes transport, attention and serving.
These isolated measurements are inputs to calibration, not goodput predictions.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from profile_operations import fit_line


def collect_samples(evidence):
    """Keep each isolated measurement once, regardless of concurrent repeats."""
    if evidence.get("status") != "passed" or evidence.get("trace_status") != "disabled":
        raise ValueError("A full-MoE profile requires a passing untraced measurement")
    if not evidence["settings"]["prefill_preprocess"]:
        raise ValueError("Full raw-P profiles require E-side preprocessing")
    executor = evidence["executor"]
    if not executor["prefill_preprocess"] or executor["benchmark_stage"] != 0:
        raise ValueError("The executor must include every configured MoE operation")
    samples = {"prefill": [], "decode": []}
    seen = set()
    for case in evidence["cases"]:
        if case["exact_isolated_output"] != [True, True]:
            raise ValueError("Concurrent/isolated outputs must agree")
        if case["prefill_rows"] in seen:
            continue
        seen.add(case["prefill_rows"])
        for index, phase in enumerate(samples):
            tile_rows = executor[f"{phase}_tile_rows"]
            counts = case["expert_rows"][index]
            timings = [s["expert_ms"] for s in case["isolated_samples"][index]]
            if (
                type(tile_rows) is not int
                or tile_rows <= 0
                or not counts
                or any(type(n) is not int or n < 0 for n in counts)
            ):
                raise ValueError(
                    "Need valid native tile geometry and integer expert counts"
                )
            if not timings or any(not math.isfinite(t) or t <= 0 for t in timings):
                raise ValueError("Need finite positive layer timings")
            footprint = sum(
                (count + tile_rows - 1) // tile_rows
                for count in case["expert_rows"][index]
            )
            samples[phase].append(
                {
                    "x": footprint,
                    "latency_s": statistics.median(timings) / 1000,
                    "expert_rows": case["expert_rows"][index],
                    "tokens": case["prefill_rows"]
                    if index == 0
                    else evidence["settings"]["decode_rows"],
                }
            )
    if not seen:
        raise ValueError("No complete layer measurements")
    return samples


def evaluate_holdout(fit, training, holdout):
    """Evaluate frozen coefficients; never refit using validation samples."""
    lower, upper = min(s["x"] for s in training), max(s["x"] for s in training)
    rows = []
    for sample in holdout:
        predicted = fit["alpha"] + fit["beta"] * sample["x"]
        actual = sample["latency_s"]
        if not math.isfinite(predicted) or not math.isfinite(actual) or actual <= 0:
            raise ValueError("Holdout latency and prediction must be finite")
        rows.append(
            sample
            | {
                "predicted_latency_s": predicted,
                "relative_error": abs(predicted - actual) / actual,
                "within_training_footprint_range": lower <= sample["x"] <= upper,
            }
        )
    if not rows:
        raise ValueError("No holdout measurements")
    return {
        "training_footprint_range": [lower, upper],
        "all_predictions_positive": all(r["predicted_latency_s"] > 0 for r in rows),
        "max_relative_error": max(r["relative_error"] for r in rows),
        "median_relative_error": statistics.median(r["relative_error"] for r in rows),
        "measurements": rows,
    }


def footprint_ambiguity(samples):
    """Lower bound on error for any predictor using only this exact footprint.

    All observations at a given x receive one prediction. For positive extrema
    lo, hi, the best possible worst relative error is (hi-lo)/(hi+lo), attained
    at 2*lo*hi/(lo+hi). This is a training-data diagnostic, not a new fit or an
    error tolerance. Noise and omitted operation footprints can both cause it.
    In particular raw-P ACT/scatter/gather need not share GEMM's tile height.
    """
    grouped = {}
    for sample in samples:
        x, latency = sample["x"], sample["latency_s"]
        if (
            not math.isfinite(x)
            or x < 0
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(
                "Ambiguity requires finite nonnegative footprints and positive times"
            )
        grouped.setdefault(x, []).append(latency)
    if not grouped:
        raise ValueError("Need observations for footprint ambiguity")
    rows = []
    for x, times in sorted(grouped.items()):
        lo, hi = min(times), max(times)
        rows.append(
            {
                "x": x,
                "observations": len(times),
                "min_latency_s": lo,
                "max_latency_s": hi,
                "minimum_worst_relative_error": (hi - lo) / (hi + lo),
            }
        )
    return {
        "scope": "Training observations only; a lower bound for any single-valued footprint model, not a validated latency model",
        "minimum_worst_relative_error": max(
            row["minimum_worst_relative_error"] for row in rows
        ),
        "groups": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sms", nargs="+", type=int, default=[32, 64, 84, 116])
    p.add_argument(
        "--prefill-rows", nargs="+", type=int, default=[128, 512, 2048, 8192]
    )
    p.add_argument("--decode-rows", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--holdout-prefill-rows", nargs="+", type=int)
    p.add_argument("--holdout-decode-rows", nargs="+", type=int)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if bool(args.holdout_prefill_rows) != bool(args.holdout_decode_rows):
        p.error("Provide both P and D holdout sizes")
    if any(
        len(values) != len(set(values))
        for values in (
            args.sms,
            args.prefill_rows,
            args.decode_rows,
            args.holdout_prefill_rows or [],
            args.holdout_decode_rows or [],
        )
    ):
        p.error("Profile grids must not contain duplicate sizes")
    if set(args.prefill_rows) & set(args.holdout_prefill_rows or []) or set(
        args.decode_rows
    ) & set(args.holdout_decode_rows or []):
        p.error("Training and holdout token counts must be disjoint in each phase")
    if min(
        *args.sms,
        *args.prefill_rows,
        *args.decode_rows,
        *(args.holdout_prefill_rows or []),
        *(args.holdout_decode_rows or []),
        args.iterations,
    ) <= 0 or any(s < 4 or s % 2 for s in args.sms):
        p.error("Use positive sizes and even SM counts >= 4")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    report = {
        "scope": __doc__,
        "units": "seconds_per_layer",
        "prefill_preprocess": True,
        "settings": vars(args) | {"output_dir": str(args.output_dir)},
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("profile_full_moe.py", "bench_preemption.py", "check_gpu.py")
        },
        "measurements": [],
        "fits": {},
    }
    try:
        source_identity = None
        for sms in args.sms:
            training = None
            grids = [("training", args.prefill_rows, args.decode_rows)]
            if args.holdout_prefill_rows:
                grids.append(
                    ("holdout", args.holdout_prefill_rows, args.holdout_decode_rows)
                )
            for split, prefill_rows, decode_rows in grids:
                samples = {"prefill": [], "decode": []}
                for decode in decode_rows:
                    path = args.output_dir / f"{split}-sm{sms}-d{decode}.json"
                    command = [
                        sys.executable,
                        str(root / "bench_preemption.py"),
                        "--prefill-preprocess",
                        "--trace-capacity",
                        "0",
                        "--ctas",
                        str(sms),
                        "--decode-budget",
                        str(max(2, sms // 4 * 2)),
                        "--prefill-rows",
                        *map(str, prefill_rows),
                        "--decode-rows",
                        str(decode),
                        "--isolated-repeats",
                        str(args.iterations),
                        "--repeats",
                        "1",
                        "--output",
                        str(path),
                    ]
                    row = {
                        "split": split,
                        "sms": sms,
                        "decode_rows": decode,
                        "command": command,
                        "evidence": str(path),
                    }
                    report["measurements"].append(row)
                    with path.with_suffix(".log").open("w") as log:
                        subprocess.run(
                            command,
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=600,
                        )
                    evidence = json.loads(path.read_text())
                    identity = evidence["native_source_sha256"]
                    if source_identity is not None and source_identity != identity:
                        raise ValueError(
                            "Runtime source changed within the profile experiment"
                        )
                    source_identity = identity
                    measured = collect_samples(evidence)
                    row.update(
                        status="passed",
                        native_source_sha256=evidence["native_source_sha256"],
                        executor=evidence["executor"],
                    )
                    for phase, values in samples.items():
                        values.extend(measured[phase])
                    print(f"Completed {split} SM={sms}, D={decode}", flush=True)
                if split == "training":
                    training = samples
                    for phase, values in samples.items():
                        fit = fit_line([(s["x"], s["latency_s"]) for s in values])
                        fit["measurements"] = values
                        fit["footprint_ambiguity"] = footprint_ambiguity(values)
                        report["fits"][f"{phase}:{sms}"] = fit
                else:
                    for phase, values in samples.items():
                        fit = report["fits"][f"{phase}:{sms}"]
                        fit["holdout"] = evaluate_holdout(fit, training[phase], values)
                # Preserve the frozen training fit before collecting holdout data.
                (args.output_dir / "profiles.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "profiles.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
