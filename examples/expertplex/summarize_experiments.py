"""Compare explicitly selected serving experiments with provenance checks.

Outputs raw-derived JSON/CSV and publication-friendly PDF/SVG figures. Synthetic
weights remain labeled on every figure. Maximum feasible tested rate is reported
without interpolating the boundary or treating it as trained-model acceptance.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from bench_sweep import joint_slo


def read_json(path):
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    text = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz"
        else path.read_text()
    )
    return json.loads(text)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def wilson(k, n):
    z = 1.95996398454
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = (
        z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return [centre - radius, centre + radius]


def summarize(paths, *, allow_memory_tuning=False):
    # The same deterministic initializer seed is required for synthetic runs.
    # Source digests are retained below for review of implementation revisions.
    common = None
    expected_input_digest = None
    expected_output_lengths = None
    rows, runs = [], []
    for root in paths:
        experiment = read_json(root / "experiment.json")
        settings = experiment["settings"]
        identity = {
            k: settings[k]
            for k in (
                "model_path",
                "tokenizer_path",
                "weight_mode",
                "dataset_path",
                "requests",
                "seed",
                "sequence_cap",
                "decode_kv_tokens",
                "attention_backend",
                "ttft_slo_ms",
                "tpot_slo_ms",
            )
        }
        identity["canonical_seed"] = experiment["environment"].get(
            "SGLANG_EXPERTPLEX_SYNTHETIC_SEED"
        )
        identity["warmup_requests"] = settings.get("warmup_requests", 1)
        if not allow_memory_tuning:
            identity["mem_fraction_static"] = settings.get("mem_fraction_static")
        # Legacy experiments forced PyTorch sampling and disabled scheduling
        # overlap/autotune. Do not silently mix them with restored defaults.
        identity["sampling_backend"] = settings.get("sampling_backend", "pytorch")
        identity["serving_defaults_version"] = experiment.get(
            "serving_defaults_version", "legacy-disabled-overlap-autotune"
        )
        artifacts = experiment.get("input_artifact_sha256", {})
        identity["input_fingerprints"] = {
            label: artifacts.get(str(path))
            for label, path in (
                ("dataset", Path(settings["dataset_path"])),
                ("model_config", Path(settings["model_path"]) / "config.json"),
                ("tokenizer", Path(settings["tokenizer_path"]) / "tokenizer.json"),
            )
        }
        if (
            identity["weight_mode"] == "synthetic"
            and identity["canonical_seed"] is None
        ):
            raise ValueError(
                "Legacy shard-dependent synthetic weights cannot support this comparison"
            )
        if common is None:
            common = identity
        elif common != identity:
            raise ValueError(
                f"Workload/resource/weight/SLO mismatch in {root}: {identity}"
            )
        sweep = read_json(root / "sweep/joint-slo.json")
        runs.append(
            {
                "path": root.name,
                "variant": settings.get("label") or settings["variant"],
                "experiment_status": experiment.get("status", "unrecorded"),
                "mem_fraction_static": settings.get("mem_fraction_static"),
                "numerical_diagnostic_requested": settings.get(
                    "numerical_check", False
                ),
                "requested_rates": settings.get("rates"),
                "measured_rates": [r["request_rate"] for r in sweep["results"]],
                "complete_requested_rate_grid": (
                    set(settings["rates"])
                    == {r["request_rate"] for r in sweep["results"]}
                    if "rates" in settings
                    else None
                ),
                "maximum_feasible_tested_rps": sweep[
                    "p90_goodput_max_tested_arrival_rps"
                ],
                "source_files_sha256": experiment.get("source_files_sha256"),
                "input_artifact_sha256": {
                    Path(name).name: digest
                    for name, digest in experiment.get(
                        "input_artifact_sha256", {}
                    ).items()
                },
                # Only reproducibility controls belong in the public aggregate.
                # Raw process environments and host paths stay in local evidence.
                "runtime_controls": {
                    key: int(value)
                    for key in (
                        "SGLANG_EXPERTPLEX_SYNTHETIC_SEED",
                        "SGLANG_EXPERTPLEX_PIPELINED_STORES",
                    )
                    if (value := experiment["environment"].get(key)) is not None
                },
            }
        )
        feasible_rates = []
        for key in ("ttft_slo_ms", "tpot_slo_ms"):
            if key in sweep and sweep[key] != settings[key]:
                raise ValueError(f"Sweep/settings SLO mismatch in {root}: {key}")
        for result in sweep["results"]:
            detailed = read_json(root / "sweep" / Path(result["details_file"]).name)
            rate = detailed.get("request_rate")
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"Invalid measured request rate in {root}")
            measured = joint_slo(
                detailed, settings["ttft_slo_ms"] / 1000, settings["tpot_slo_ms"] / 1000
            )
            count = measured["attempted_requests"]
            if count != settings["requests"] or len(detailed["input_lens"]) != count:
                raise ValueError(f"Requested/detail count mismatch in {root}")
            if any(type(v) is not int or v < 1 for v in detailed["input_lens"]) or any(
                type(v) is not int or v < 0 for v in detailed["output_lens"]
            ):
                raise ValueError(f"Invalid token lengths in {root}")
            if any(type(v) is not bool for v in detailed["successes"]):
                raise ValueError(f"Invalid request success flags in {root}")
            for key, value in measured.items():
                # Older summaries omit goodput and one-token counts. Recompute
                # both; verify every stored field rather than trusting it.
                if key not in result and key in (
                    "observed_slo_goodput_rps",
                    "unscorable_one_token_responses",
                ):
                    continue
                stored = result.get(key)
                matches = (
                    type(stored) in (int, float)
                    and math.isfinite(stored)
                    and math.isclose(stored, value, rel_tol=1e-12, abs_tol=1e-15)
                    if type(value) is float
                    else stored == value
                )
                if not matches:
                    raise ValueError(f"Raw-derived SLO mismatch in {root}: {key}")
            if measured["meets_p90_joint_slo"]:
                feasible_rates.append(measured["request_rate"])
            if expected_output_lengths is None:
                expected_output_lengths = [None] * count
            for i, (length, ok) in enumerate(
                zip(detailed["output_lens"], detailed["successes"], strict=True)
            ):
                if not ok:
                    continue  # A failed response cannot establish requested length.
                previous = expected_output_lengths[i]
                if previous is not None and previous != length:
                    raise ValueError(
                        f"Actual ordered request output lengths differ in {root}"
                    )
                expected_output_lengths[i] = length
            input_digest = hashlib.sha256(
                json.dumps(detailed["input_lens"]).encode()
            ).hexdigest()
            if expected_input_digest is None:
                expected_input_digest = input_digest
            elif input_digest != expected_input_digest:
                raise ValueError(
                    f"Actual ordered request input lengths differ in {root}"
                )
            completed = [
                (ttft, latency, length)
                for ttft, latency, length, ok in zip(
                    detailed["ttfts"],
                    detailed["latencies"],
                    detailed["output_lens"],
                    detailed["successes"],
                )
                if ok
            ]
            row = {
                "variant": settings.get("label") or settings["variant"],
                "experiment": root.name,
                "request_rate": result["request_rate"],
                "attempted": result["attempted_requests"],
                "completed": result["completed_requests"],
                "joint_met": result["joint_slo_met_requests"],
                "joint_attainment": result["joint_slo_attainment"],
                "ttft_attainment": sum(
                    ttft * 1000 <= settings["ttft_slo_ms"] for ttft, _, _ in completed
                )
                / result["attempted_requests"],
                "tpot_attainment": sum(
                    length > 1
                    and (latency - ttft) * 1000 / (length - 1)
                    <= settings["tpot_slo_ms"]
                    for ttft, latency, length in completed
                )
                / result["attempted_requests"],
                "joint_wilson95": wilson(
                    result["joint_slo_met_requests"], result["attempted_requests"]
                ),
                "p90_ttft_ms_completed": percentile(
                    [ttft * 1000 for ttft, _, _ in completed], 0.9
                ),
                "p90_tpot_ms_completed": percentile(
                    [
                        (latency - ttft) * 1000 / (length - 1)
                        for ttft, latency, length in completed
                        if length > 1
                    ],
                    0.9,
                ),
                "observed_slo_goodput_rps": measured["observed_slo_goodput_rps"],
                "unscorable_one_token_responses": measured[
                    "unscorable_one_token_responses"
                ],
                "output_tokens_per_second": sum(length for _, _, length in completed)
                / detailed["duration"],
                "meets_p90_joint_slo": result["meets_p90_joint_slo"],
            }
            rows.append(row)
        if (
            max(feasible_rates, default=None)
            != sweep["p90_goodput_max_tested_arrival_rps"]
        ):
            raise ValueError(
                f"Raw-derived maximum feasible tested rate mismatch in {root}"
            )
    for key in ("model_path", "tokenizer_path", "dataset_path"):
        common[key] = Path(common[key]).name
    return {
        "scope": __doc__,
        "common_conditions": common,
        "ordered_input_lengths_sha256": expected_input_digest,
        "ordered_successful_output_lengths_sha256": hashlib.sha256(
            json.dumps(expected_output_lengths).encode()
        ).hexdigest(),
        "output_length_reference_complete": expected_output_lengths is not None
        and all(v is not None for v in expected_output_lengths),
        "runs": runs,
        "memory_fraction_policy": (
            "Per-method tuning; individual static memory fractions are recorded in runs. "
            "This is not a fixed-KV-pool comparison."
            if allow_memory_tuning
            else "Same static memory fraction required"
        ),
        "rows": rows,
        "uncertainty_note": "Wilson intervals describe finite-request attainment, not uncertainty in the arrival-rate boundary; repeated Poisson seeds remain necessary.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--allow-memory-tuning",
        action="store_true",
        help="Explicitly allow per-method static memory fractions; keep sequence cap, "
        "explicit KV cap, workload, warmup and SLO checks unchanged",
    )
    args = p.parse_args()
    report = summarize(args.experiments, allow_memory_tuning=args.allow_memory_tuning)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=report["rows"][0].keys())
        w.writeheader()
        w.writerows(report["rows"])
    if args.no_plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = report["common_conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for variant in dict.fromkeys(r["variant"] for r in report["rows"]):
        series = sorted(
            [r for r in report["rows"] if r["variant"] == variant],
            key=lambda r: r["request_rate"],
        )
        x = [r["request_rate"] for r in series]
        for ax, key in zip(
            axes, ["p90_ttft_ms_completed", "p90_tpot_ms_completed", "joint_attainment"]
        ):
            ax.plot(
                x,
                [r[key] for r in series],
                "o-",
                label=variant.removeprefix("full480-canonical-"),
                linewidth=1.5,
                markersize=4,
            )
    for ax, label, threshold in zip(
        axes,
        [
            "P90 TTFT (ms; completed)",
            "P90 TPOT (ms; completed)",
            "Joint SLO attainment (all offered)",
        ],
        [conditions["ttft_slo_ms"], conditions["tpot_slo_ms"], 0.9],
    ):
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Offered requests/s")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_ylim(0, 1.04)
    fig.legend(
        *axes[-1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    mode = (
        "CANONICAL SYNTHETIC BF16 — not pretrained-model acceptance"
        if conditions["weight_mode"] == "synthetic"
        else "BF16 checkpoint"
    )
    title = (
        f"Qwen3-Coder-480B | 8×B300 | {Path(conditions['dataset_path']).stem}\n"
        f"{mode}; n={conditions['requests']}, seed={conditions['seed']}, "
        f"warmup={conditions['warmup_requests']}"
    )
    if args.allow_memory_tuning:
        title += "; per-method static memory fractions"
    fig.suptitle(title, fontsize=10)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(args.output_dir / f"serving-curves.{suffix}", dpi=180)
    plt.close(fig)

    # Figures 7--10 use separate TTFT, TPOT and joint attainment curves.
    # Failed requests remain in every denominator; one-token outputs have no
    # TPOT observation and cannot count as a TPOT or joint success.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for variant in dict.fromkeys(r["variant"] for r in report["rows"]):
        series = sorted(
            [r for r in report["rows"] if r["variant"] == variant],
            key=lambda r: r["request_rate"],
        )
        for ax, key in zip(
            axes, ["ttft_attainment", "tpot_attainment", "joint_attainment"]
        ):
            ax.plot(
                [r["request_rate"] for r in series],
                [100 * r[key] for r in series],
                "o-",
                label=variant.removeprefix("full480-canonical-"),
                linewidth=1.5,
                markersize=4,
            )
    for ax, label in zip(
        axes,
        [
            "TTFT SLO attainment (%)",
            "TPOT SLO attainment (%)",
            "Joint SLO attainment (%)",
        ],
    ):
        ax.axhline(90, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Offered requests/s", ylabel=label, ylim=(0, 104))
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[-1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=10)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(args.output_dir / f"slo-attainment.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
