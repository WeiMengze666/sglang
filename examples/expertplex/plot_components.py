"""Render measured component experiments; never synthesize missing samples."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="kind", required=True)
    sharing = sub.add_parser("sharing")
    sharing.add_argument("--native", required=True, type=Path)
    sharing.add_argument("--apk", required=True, type=Path)
    sharing.add_argument("--mps", required=True, type=Path)
    intervals = sub.add_parser("intervals")
    intervals.add_argument("--input", required=True, type=Path)
    ablation = sub.add_parser("interval-ablation")
    ablation.add_argument("--input", required=True, type=Path)
    ablation.add_argument("--reference", required=True, type=Path)
    overhead = sub.add_parser("overhead")
    overhead.add_argument("--input", required=True, type=Path)
    communication = sub.add_parser("communication")
    communication.add_argument("--inputs", required=True, type=Path, nargs="+")
    communication.add_argument("--labels", nargs="+", help="One label per input report")
    for parser in (sharing, intervals, ablation, overhead, communication):
        parser.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        v for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"
    ]
    if args.kind == "communication":
        sources = args.inputs
    provenance = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    if args.kind == "sharing":
        native, apk, mps = [
            json.loads(path.read_text()) for path in (args.native, args.apk, args.mps)
        ]
        if (
            len({r.get("workload_identity", "legacy-cuda") for r in (native, apk, mps)})
            != 1
        ):
            raise ValueError("Workload initialization algorithms differ")
        if native["expert_rows"] != apk["expert_rows"]:
            raise ValueError("Native/APK expert vectors differ")
        for case in mps["cases"]:
            if [client["expert_rows"] for client in case["clients"]] != native[
                "expert_rows"
            ]:
                raise ValueError("MPS expert vectors differ")
        if len({r["settings"]["operation"] for r in (native, apk, mps)}) != 1:
            raise ValueError("Operation mismatch")
        isolated = native["isolated_response_upper_bound_us"]
        points = []
        for case in native["cases"]:
            points.append(
                (case["mode"], case["response_upper_bound_us"], str(case["actual_sms"]))
            )
        points += [
            ("APK", case["response_upper_bound_us"], f"q={case['decode_budget']}")
            for case in apk["cases"]
        ]
        points += [
            (
                "MPS",
                case["response_upper_bound_us"],
                str(case["requested_percentages"]) + "%",
            )
            for case in mps["cases"]
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method in dict.fromkeys(x[0] for x in points):
            selected = [x for x in points if x[0] == method]
            ax.scatter(
                [v[1][1] for v in selected],
                [v[1][0] for v in selected],
                label=method,
                s=28,
            )
            frontier = [
                v
                for v in selected
                if not any(
                    w[1][0] <= v[1][0] and w[1][1] <= v[1][1] and w[1] != v[1]
                    for w in selected
                )
            ]
            frontier.sort(key=lambda x: x[1][1])
            ax.plot(
                [v[1][1] for v in frontier],
                [v[1][0] for v in frontier],
                linewidth=0.8,
                alpha=0.6,
            )
        ax.set_xlabel("Decode response upper bound (µs)")
        ax.set_ylabel("Prefill response upper bound (µs)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.suptitle(
            f"B300 BF16 {native['settings']['operation']} | P8192 / D128 / top-8\nAttention-clock release→ack; D released 10 µs later",
            fontsize=10,
        )
        data = {
            "sources_sha256": provenance,
            "points": [
                {
                    "method": m,
                    "prefill_us": v[0],
                    "decode_us": v[1],
                    "policy": policy,
                    "slowdown_vs_native_isolated": [
                        v[i] / isolated[i] for i in range(2)
                    ],
                }
                for m, v, policy in points
            ],
        }
    elif args.kind == "interval-ablation":
        before, after = [
            json.loads(path.read_text()) for path in (args.reference, args.input)
        ]
        for report in (before, after):
            if report["status"] != "passed":
                raise ValueError("Both interval experiments must pass")
        for field in ("expert_rows", "iterations", "delay_us"):
            if before["settings"][field] != after["settings"][field]:
                raise ValueError(f"Interval workload mismatch: {field}")
        for field in (
            "hidden",
            "intermediate",
            "ctas",
            "cluster_size",
            "pipelined_stores",
            "prefill_tile_rows",
            "decode_tile_rows",
            "prefill_preprocess",
        ):
            if before["executor"][field] != after["executor"][field]:
                raise ValueError(f"Interval executor mismatch: {field}")

        def keys(report):
            return {(v["operation"], v["expert_rows"]) for v in report["intervals"]}

        if keys(before) != keys(after):
            raise ValueError("Interval operation/size grids differ")
        operations = list(dict.fromkeys(r["operation"] for r in after["intervals"]))
        if len(operations) != 6:
            raise ValueError("Expected all six persistent operations")
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
        for ax, operation in zip(axes.flat, operations):
            for report, label, color in (
                (before, "Before prefetch", "#777777"),
                (after, "After prefetch", "#1864ab"),
            ):
                series = sorted(
                    (r for r in report["intervals"] if r["operation"] == operation),
                    key=lambda r: r["expert_rows"],
                )
                x = [r["expert_rows"] for r in series]
                y = [r["median_us"] for r in series]
                ax.plot(x, y, "o-", color=color, label=label, markersize=3)
                ax.fill_between(
                    x, y, [r["max_us"] for r in series], color=color, alpha=0.15
                )
            ax.set_title(operation)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Prefill rows per expert")
            ax.set_ylabel("Interval (µs)")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "B300 BF16 data-load prefetch ablation | same tiles, pipelined stores enabled\n"
            "Lines: median; shading: median to observed maximum, not confidence intervals\n"
            "Producer handoff→store retirement includes pipeline waits; excludes A→E propagation",
            fontsize=10,
        )
        data = {
            "sources_sha256": provenance,
            "before": before["intervals"],
            "after": after["intervals"],
            "scope": after["scope"],
        }
    elif args.kind == "intervals":
        report = json.loads(args.input.read_text())
        if report["status"] != "passed":
            raise ValueError("Interval experiment did not pass")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for operation in dict.fromkeys(r["operation"] for r in report["intervals"]):
            series = sorted(
                [r for r in report["intervals"] if r["operation"] == operation],
                key=lambda r: r["expert_rows"],
            )
            for ax, key in zip(axes, ["median_us", "max_us"]):
                ax.plot(
                    [r["expert_rows"] for r in series],
                    [r[key] for r in series],
                    "o-",
                    label=operation,
                    markersize=3,
                )
        for ax, label in zip(
            axes, ["Median interval (µs)", "Maximum observed interval (µs)"]
        ):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Prefill rows per expert")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "B300 BF16 all-operation intervals\nProducer handoff→store retirement; includes pipeline waits, excludes A→E propagation",
            fontsize=10,
        )
        data = {
            "sources_sha256": provenance,
            "intervals": report["intervals"],
            "scope": report["scope"],
        }
    elif args.kind == "overhead":
        report = json.loads(args.input.read_text())
        if report["status"] != "completed":
            raise ValueError("Operator profile incomplete")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        full = max(r["sms"] for r in report["cases"])
        for operation in dict.fromkeys(r["operation"] for r in report["cases"]):
            series = [
                r
                for r in report["cases"]
                if r["sms"] == full and r["operation"] == operation
            ]
            for ax, index, xkey in zip(
                axes, [0, 1], ["prefill_expert_rows", "decode_active_experts"]
            ):
                for backend, style in [("native", "--"), ("apk", "-")]:
                    ax.plot(
                        [r[xkey] for r in series],
                        [r[backend]["isolated_us"][index] for r in series],
                        style,
                        marker="o",
                        markersize=3,
                        label=f"{operation} {backend}",
                    )
        for ax, label in zip(
            axes, ["Prefill rows per expert", "Active decode experts (one row each)"]
        ):
            ax.set_xlabel(label)
            ax.set_ylabel("Response upper bound (µs)")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=7)
        fig.suptitle(
            f"B300 BF16 isolated native/APK profiles | {full} SMs\nMatched weights, inputs and expert-load vectors; communication excluded",
            fontsize=10,
        )
        data = {
            "sources_sha256": provenance,
            "scope": report["scope"],
            "cases": [
                {
                    key: row[key]
                    for key in (
                        "sms",
                        "operation",
                        "prefill_expert_rows",
                        "decode_active_experts",
                        "apk_overhead_us",
                    )
                }
                | {
                    backend: {"isolated_us": row[backend]["isolated_us"]}
                    for backend in ("native", "apk")
                }
                for row in report["cases"]
            ],
        }
    else:
        from summarize_experiments import read_json

        reports = [read_json(path) for path in args.inputs]
        if args.labels and len(args.labels) != len(reports):
            p.error("Supply exactly one communication label per report")
        identity = None
        for report in reports:
            settings = report["settings"]
            if settings["vary_routes"] or not settings["cuda_graph"]:
                raise ValueError("Use the same static-route, graph-replayed workload")
            current = {k: settings[k] for k in ("hidden", "experts", "top_k", "seed")}
            if identity is not None and current != identity:
                raise ValueError("Communication input geometry/seed mismatch")
            identity = current
            if not all(
                row["identity_passed"] for rank in report["ranks"] for row in rank
            ):
                raise ValueError("Communication identity failed")
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        for report_index, report in enumerate(reports):
            settings = report["settings"]
            row = 0 if settings["mode"] == "normal" else 1
            method = settings["backend"]
            if method == "expertplex":
                method += " grouped" if settings["group_transfers"] else " peer"
                if settings.get("prefill_preprocess"):
                    method += " raw"
            else:
                revisions = {
                    r["deepep_source_revision"]
                    for rank in report["ranks"]
                    for r in rank
                }
                if len(revisions) != 1 or not next(iter(revisions)):
                    raise ValueError("DeepEP source revision must be recorded")
                method += " v1"
            if args.labels:
                method = args.labels[report_index]
            for column, operation in enumerate(["dispatch_us", "combine_us"]):
                axes[row, column].plot(
                    [r["rows"] for r in report["summary"]],
                    [r["median_max_rank_us"][operation] for r in report["summary"]],
                    "o-",
                    markersize=3,
                    label=method,
                )
        for row, mode in enumerate(["Normal / prefill", "Low latency / decode"]):
            for column, operation in enumerate(["Dispatch", "Combine"]):
                ax = axes[row, column]
                ax.set_title(f"{mode}: {operation}", fontsize=10)
                ax.set_xlabel("Tokens per active attention client")
                ax.set_ylabel("Median max-rank duration (µs)")
                ax.set_xscale("log", base=2)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
        fig.suptitle(
            "8×B300 NVLink adaptation of §7.5 | BF16, top-8\nStatic routes with uncached dynamic-route APIs; GPU barriers/echo excluded",
            fontsize=10,
        )
        data = {
            "sources_sha256": provenance,
            "reports": [
                {
                    "settings": {
                        key: value
                        for key, value in r["settings"].items()
                        if key
                        in {
                            "backend",
                            "mode",
                            "rows",
                            "hidden",
                            "experts",
                            "top_k",
                            "iterations",
                            "group_transfers",
                            "prefill_preprocess",
                            "vary_routes",
                            "cuda_graph",
                            "seed",
                            "deepep_revision",
                        }
                    },
                    "summary": r["summary"],
                    "scope": r["scope"],
                }
                for r in reports
            ],
            "labels": args.labels,
        }
    (args.output_dir / "plot-data.json").write_text(json.dumps(data, indent=2) + "\n")
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(args.output_dir / f"{args.kind}.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
