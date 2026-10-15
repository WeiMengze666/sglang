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


def summarize(paths):
    # The same deterministic initializer seed is required for synthetic runs.
    # Source digests are retained below for review of implementation revisions.
    common = None
    expected_input_digest = None
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
        for result in sweep["results"]:
            detailed = read_json(root / "sweep" / Path(result["details_file"]).name)
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
                "output_tokens_per_second": sum(detailed["output_lens"])
                / detailed["duration"],
                "meets_p90_joint_slo": result["meets_p90_joint_slo"],
            }
            rows.append(row)
    for key in ("model_path", "tokenizer_path", "dataset_path"):
        common[key] = Path(common[key]).name
    return {
        "scope": __doc__,
        "common_conditions": common,
        "ordered_input_lengths_sha256": expected_input_digest,
        "runs": runs,
        "rows": rows,
        "uncertainty_note": "Wilson intervals describe finite-request attainment, not uncertainty in the arrival-rate boundary; repeated Poisson seeds remain necessary.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    report = summarize(args.experiments)
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
