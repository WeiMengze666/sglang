"""Paper sections 7.3/7.6 diagnostics: GPU-timed P/D release and tile traces.

Single E rank with BF16 synthetic weights. This is not end-to-end goodput or
proof of the complete paper bound. Tracing perturbs timing; run performance
comparisons separately with tracing disabled. A and E clocks are not subtracted.
"""

import argparse
import json
from pathlib import Path

from _bootstrap import module
from check_gpu import Exporter


def trace_overflow_ctas(metadata, ctas, capacity):
    """Use producer sequence counts; an empty final trace slot proves nothing."""
    counts = metadata.get("trace_sequence_counts")
    if (
        metadata.get("trace_capacity_per_cta") != capacity
        or not isinstance(counts, list)
        or len(counts) != ctas
        or any(type(value) is not int or value < 0 for value in counts)
    ):
        raise ValueError("Missing or inconsistent trace sequence metadata")
    return sum(value > capacity for value in counts)


def validate_cluster_trace(records, cluster_size, truncated_ctas=0):
    """Check cooperative decisions and retirement ordering, not a latency bound."""
    if truncated_ctas or not records or cluster_size < 1:
        raise ValueError("Need a complete nonempty cluster trace")
    groups = {}
    for record in records:
        if record["finish_ns"] < record["begin_ns"]:
            raise ValueError("Unretired tile in cluster trace")
        key = (record["cta"] // cluster_size, record["sequence"])
        members = groups.setdefault(key, {})
        if record["cta"] in members:
            raise ValueError("Duplicate CTA decision")
        members[record["cta"]] = record
    previous, switches = {}, 0
    for (cluster, _), members in sorted(groups.items()):
        expected = set(range(cluster * cluster_size, (cluster + 1) * cluster_size))
        if set(members) != expected:
            raise ValueError("Missing CTA in cluster decision")
        decisions = {
            tuple(row[field] for field in ("epoch", "slot", "phase", "stage"))
            for row in members.values()
        }
        if len(decisions) != 1:
            raise ValueError("Cluster members selected different work")
        phase = next(iter(decisions))[2]
        begin = min(row["begin_ns"] for row in members.values())
        finish = max(row["finish_ns"] for row in members.values())
        before = previous.get(cluster)
        if before is not None and phase != before[0]:
            if begin < before[1]:
                raise ValueError("Phase switched before all old-phase CTAs retired")
            switches += 1
        elif before is not None:
            finish = max(finish, before[1])
        previous[cluster] = (phase, finish)
    return {"cluster_decisions": len(groups), "phase_switches": switches}


DECISION_FIELDS = (
    "observed_ns",
    "retired_self",
    "retired_peer",
    "handoff_sequence",
    "previous_phase",
    "preferred_phase",
    "selected_phase",
    "preferred_runnable",
)
PROGRESS_FIELDS = ("accumulator_ready_ns", "output_submitted_ns")


def validate_single_pending_compute(checks):
    """Check E-side outstanding computation, not the full propagation bound."""
    counts = [
        c["gemms_without_ready_observation"] + c["data_operations_before_submission"]
        for switch in checks["phase_switches"]
        for c in switch["ctas"]
    ]
    if not counts:
        raise ValueError("Need observed phase switches for a pending-computation check")
    if any(count < 0 or count > 1 for count in counts):
        raise ValueError(
            "Phase decision has more than one unfinished compute tile per CTA"
        )
    return {
        "checked_cta_switches": len(counts),
        "max_pending_compute_tiles": max(counts),
        "scope": "E-side computation only; excludes A-to-E propagation and output retirement",
    }


def validate_progress_trace(
    records, decisions, progress, cluster_size, truncated_ctas=0
):
    """Separate unretired outputs from computation, without certifying a bound.

    GEMM readiness is observed by the epilogue after its accumulator barrier;
    this is later than or equal to actual Tensor Core completion. A zero value
    is required for non-GEMM operations. Output submission is distinct from
    global-store completion. All timestamps belong to one expert GPU.
    """
    checks = validate_decision_trace(records, decisions, cluster_size, truncated_ctas)
    indexed = {(r["cta"], r["sequence"]): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate tile progress key")
    seen = set()
    by_cluster = {}
    for sample in progress:
        key = (sample["cta"], sample["sequence"])
        if key in seen or key not in indexed:
            raise ValueError("Duplicate or unknown tile progress")
        seen.add(key)
        row = indexed[key] | sample
        ready, submitted = row["accumulator_ready_ns"], row["output_submitted_ns"]
        if not row["begin_ns"] <= submitted <= row["finish_ns"]:
            raise ValueError("Output submission must lie within tile lifetime")
        if row["stage"] in (1, 2):
            if not row["begin_ns"] <= ready <= submitted:
                raise ValueError("GEMM requires an ordered accumulator observation")
        elif ready != 0:
            raise ValueError("Non-GEMM work has no accumulator observation")
        by_cluster.setdefault(row["cta"] // cluster_size, []).append(row)
    if seen != set(indexed):
        raise ValueError("Missing tile progress")
    decision_index = {
        (d["cta"] // cluster_size, d["claim_sequence"]): d for d in decisions
    }
    switches = []
    for switch in checks["phase_switches"]:
        cluster, claim = switch["cluster"], switch["claim_sequence"]
        decision = decision_index[cluster, claim]
        observed = decision["observed_ns"]
        inflight = [
            row
            for row in by_cluster[cluster]
            if row["sequence"] < claim and row["finish_ns"] > observed
        ]
        counts = []
        for cta in range(cluster * cluster_size, (cluster + 1) * cluster_size):
            rows = [r for r in inflight if r["cta"] == cta]
            counts.append(
                {
                    "cta": cta,
                    "unretired_outputs": len(rows),
                    "gemms_without_ready_observation": sum(
                        r["stage"] in (1, 2) and r["accumulator_ready_ns"] > observed
                        for r in rows
                    ),
                    "data_operations_before_submission": sum(
                        r["stage"] not in (1, 2) and r["output_submitted_ns"] > observed
                        for r in rows
                    ),
                    "submitted_outputs_awaiting_retirement": sum(
                        r["output_submitted_ns"] <= observed for r in rows
                    ),
                }
            )
        switches.append({"cluster": cluster, "claim_sequence": claim, "ctas": counts})
    return {
        "records": len(seen),
        "phase_switches": switches,
        "max_gemms_without_ready_observation_per_cta": max(
            (c["gemms_without_ready_observation"] for s in switches for c in s["ctas"]),
            default=0,
        ),
        "scope": "E-side progress only; not a bound on publication, propagation, or full tile drain",
    }


def validate_decision_trace(records, decisions, cluster_size, truncated_ctas=0):
    """Join pre-claim observations to post-drain handoffs on the E clock only.

    Runnable masks are snapshots, not evidence that an unclaimed tile existed:
    an operation keeps its bit while its final tiles are still in flight.
    """
    validate_cluster_trace(records, cluster_size, truncated_ctas)
    groups = {}
    for row in records:
        groups.setdefault((row["cta"] // cluster_size, row["sequence"]), []).append(row)
    by_cluster = {}
    for (cluster, sequence), members in groups.items():
        by_cluster.setdefault(cluster, []).append((sequence, members))
    seen, previous, switches = set(), {}, []
    for decision in sorted(decisions, key=lambda d: (d["cta"], d["claim_sequence"])):
        cta = decision["cta"]
        claim, handoff = decision["claim_sequence"], decision["handoff_sequence"]
        if cta % cluster_size or claim < 0 or not claim <= handoff <= claim + 1:
            raise ValueError("Invalid leader or deferred handoff sequence")
        cluster = cta // cluster_size
        key = (cluster, handoff)
        if key in seen or key not in groups:
            raise ValueError("Duplicate or missing handoff for decision")
        seen.add(key)
        selected = decision["selected_phase"]
        if selected not in (0, 1) or decision["preferred_phase"] not in (0, 1):
            raise ValueError("Invalid decision phase")
        if decision["previous_phase"] != previous.get(cluster, 2):
            raise ValueError("Decision does not follow the previous phase")
        rows = groups[key]
        observed = decision["observed_ns"]
        if observed <= 0 or observed > min(r["begin_ns"] for r in rows):
            raise ValueError("Observation must precede every member handoff")
        if any(r["phase"] != selected for r in rows):
            raise ValueError("Observed decision and handed-off work differ")
        if any((cluster, seq) in groups for seq in range(claim, handoff)):
            raise ValueError("Real work claimed between decision and deferred handoff")
        prefixes = [decision["retired_self"], decision["retired_peer"]]
        if any(p < -1 or p > claim for p in prefixes):
            raise ValueError("Invalid retirement prefix at observation")
        if cluster in previous and previous[cluster] != selected:
            inflight = [
                r
                for seq, members in by_cluster[cluster]
                if seq < claim
                for r in members
                if r["finish_ns"] > observed
            ]
            switches.append(
                {
                    "cluster": cluster,
                    "claim_sequence": claim,
                    "selected_phase": selected,
                    "observed_to_handoff_us": (
                        max(r["begin_ns"] for r in rows) - observed
                    )
                    / 1000,
                    "old_tiles_in_flight_per_cta": max(
                        sum(r["cta"] == member for r in inflight)
                        for member in range(cta, cta + cluster_size)
                    ),
                    "unretired_sequence_upper_bound": (
                        None if -1 in prefixes else claim - min(prefixes)
                    ),
                    "deferred_control_flush": handoff != claim,
                }
            )
        previous[cluster] = selected
    if seen != set(groups):
        raise ValueError("Missing pre-claim decision snapshots")
    return {"decisions": len(seen), "phase_switches": switches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-rows", type=int, nargs="+", default=[512, 2048, 8192]
    )
    parser.add_argument("--decode-rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2560)
    parser.add_argument("--experts", type=int, default=40)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--decode-budget", type=int, default=32)
    parser.add_argument("--expected-tiles", type=float, default=40)
    parser.add_argument("--delay-us", type=int, default=10)
    parser.add_argument("--trace-capacity", type=int, default=4096)
    parser.add_argument(
        "--check-single-pending-compute",
        action="store_true",
        help="Require at most one unfinished compute tile at E phase decisions; not a full propagation bound",
    )
    parser.add_argument(
        "--ctas", type=int, default=0, help="Zero uses all available SMs"
    )
    parser.add_argument("--isolated-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prefill-preprocess",
        action="store_true",
        help="Include raw prefill Scatter1/Scatter2/Gather in the persistent executor",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.check_single_pending_compute and args.trace_capacity <= 0:
        parser.error("Pending-computation checks require a complete trace")
    if (
        min(*args.prefill_rows, args.decode_rows, args.repeats, args.isolated_repeats)
        <= 0
    ):
        parser.error("Row/repeat counts must be positive")
    if args.trace_capacity < 0 or args.ctas < 0 or args.ctas % 2:
        parser.error(
            "Trace capacity must be nonnegative; CTA count must be zero or even"
        )
    if not 1 <= args.top_k <= args.experts * args.world:
        parser.error("top-k must fit the global expert count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    native = module("native")
    torch.set_num_threads(8)
    native.load_extension()
    capacities = [max(args.prefill_rows), args.decode_rows]
    trace_path = args.output.with_suffix(".trace.pt")
    settings = dict(
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        layers=1,
        capacity=max(capacities),
        capacities=capacities,
        packed_layout=[1, 0],
        top_k=args.top_k,
        seed=20261015,
        weight_std=0.02,
        decode_budget=args.decode_budget,
        expected_tiles=args.expected_tiles,
        trace_capacity=args.trace_capacity,
        trace_path=str(trace_path) if args.trace_capacity else None,
        ctas=args.ctas,
        prefill_preprocess=args.prefill_preprocess,
        expert_rank=0,
        expert_world=1,
    )
    owner = Exporter(settings, 1, execute=True, timeout_s=600)
    owner.__enter__()
    torch.cuda.set_device(0)
    peers = [native.RemoteBuffers(owner.descriptor, slot, 0) for slot in (0, 1)]
    groups = (
        [native.RemoteGroup([peer]) for peer in peers]
        if args.prefill_preprocess
        else []
    )
    report = {
        "scope": __doc__,
        "settings": vars(args) | {"output": str(args.output)},
        "native_source_sha256": native.source_fingerprint(),
        "executor": owner.executor_info,
        "cases": [],
    }

    def prepare(rows, seed):
        generator = torch.Generator(device=0).manual_seed(seed)
        x = torch.randn(
            rows, args.hidden, generator=generator, device=0, dtype=torch.bfloat16
        )
        ids = (
            torch.rand(rows, args.experts * args.world, generator=generator, device=0)
            .topk(args.top_k, dim=-1)
            .indices.int()
            .contiguous()
        )
        if args.prefill_preprocess:
            # Represent rank zero of the requested logical world on one E GPU.
            # Group transport contains that one peer; preserve only its routes.
            ids = torch.where(ids.remainder(args.world) == 0, ids // args.world, -1)
        weights = torch.full(ids.shape, 1 / args.top_k, device=0)
        return (
            x,
            ids,
            weights,
            torch.empty_like(ids),
            torch.zeros_like(x, dtype=torch.float32),
        )

    def push(slot, case, publish=True):
        x, ids, _, row_map, out = case
        out.zero_()
        if args.prefill_preprocess:
            groups[slot].dispatch(x, ids, row_map, 0, slot, case[2], publish=publish)
        else:
            native.dispatch(
                peers[slot], x, ids, row_map, 0, 0, slot, 0, args.world, publish=publish
            )

    def pull(slot, case):
        _, ids, weights, row_map, out = case
        if args.prefill_preprocess:
            groups[slot].combine(out, ids, weights, row_map)
        else:
            native.combine(peers[slot], out, ids, weights, row_map, 0, 0, args.world)
        stamps = native.read_timing(peers[slot]).cpu().tolist()
        if stamps[0] != stamps[1] or stamps[3] < stamps[2]:
            raise AssertionError("Invalid operation timing/epoch")
        if not bool(torch.isfinite(out).all()):
            raise AssertionError("Nonfinite output")
        return {
            "epoch": stamps[0],
            "start_ns": stamps[2],
            "finish_ns": stamps[3],
            "expert_ms": (stamps[3] - stamps[2]) / 1e6,
        }

    try:
        for rows in args.prefill_rows:
            cases = [prepare(rows, rows), prepare(args.decode_rows, 128)]
            expert_rows = []
            for case in cases:
                ids = case[1].flatten()
                local_ids = (
                    ids[ids >= 0]
                    if args.prefill_preprocess
                    else ids[ids.remainder(args.world) == 0] // args.world
                )
                expert_rows.append(
                    torch.bincount(local_ids.long(), minlength=args.experts)
                    .cpu()
                    .tolist()
                )
            isolated, references, isolated_samples = [], [], []
            for slot, case in enumerate(cases):
                # Warm outside the optional repeated profile interval.
                push(slot, case)
                warm = pull(slot, case)
                references.append(case[-1].clone())
                samples = [warm] if args.isolated_repeats == 1 else []
                if args.isolated_repeats > 1:
                    for _ in range(args.isolated_repeats):
                        push(slot, case)
                        samples.append(pull(slot, case))
                        if not bool(torch.equal(case[-1], references[-1])):
                            raise AssertionError("Repeated isolated output changed")
                isolated_samples.append(samples)
                isolated.append(
                    sorted(samples, key=lambda s: s["expert_ms"])[len(samples) // 2]
                )
            for repeat in range(args.repeats):
                for slot, case in enumerate(cases):
                    push(slot, case, publish=False)
                publication = native.publish_pair(peers[0], peers[1], args.delay_us)
                decode = pull(1, cases[1])
                prefill = pull(0, cases[0])
                exact = [
                    bool(torch.equal(case[-1], ref))
                    for case, ref in zip(cases, references)
                ]
                if not all(exact):
                    raise AssertionError(
                        "Concurrent output differs from isolated operation"
                    )
                timestamps = publication.cpu().tolist()
                if len(timestamps) != 4 or any(t <= 0 for t in timestamps):
                    raise AssertionError("Missing expert admission acknowledgement")
                report["cases"].append(
                    {
                        "prefill_rows": rows,
                        "repeat": repeat,
                        "prefill": prefill,
                        "decode": decode,
                        "isolated": isolated,
                        "isolated_samples": isolated_samples,
                        "expert_rows": expert_rows,
                        "actual_attention_release_gap_us": (
                            timestamps[1] - timestamps[0]
                        )
                        / 1000,
                        "expert_admission_gap_us": (
                            decode["start_ns"] - prefill["start_ns"]
                        )
                        / 1000,
                        "admission_roundtrip_upper_bound_us": {
                            "prefill": (timestamps[2] - timestamps[0]) / 1000,
                            "decode": (timestamps[3] - timestamps[1]) / 1000,
                        },
                        "exact_isolated_output": exact,
                        "decode_slowdown": decode["expert_ms"]
                        / isolated[1]["expert_ms"],
                    }
                )
    finally:
        torch.cuda.synchronize(0)
        groups.clear()
        for peer in peers:
            peer.close()
        owner.__exit__(None)
    if not args.trace_capacity:
        report.update(
            status="passed",
            trace_status="disabled",
            timing_scope="E admission-to-retirement includes full configured MoE operations; excludes transport and serving",
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return
    trace = torch.load(trace_path, weights_only=True)
    records = []
    for cta, row in enumerate(trace.tolist()):
        for begin, finish, ready, epoch, sequence, slot, phase, stage in row:
            if begin:
                if finish < begin:
                    raise AssertionError("Incomplete trace record")
                records.append(
                    {
                        "cta": cta,
                        "epoch": epoch,
                        "slot": slot,
                        "phase": phase,
                        "stage": stage,
                        "sequence": sequence,
                        "begin_ns": begin,
                        "finish_ns": finish,
                        "ready_seen_ns": ready,
                    }
                )
    report["trace_file"] = str(trace_path)
    report["trace_fields"] = [
        "begin_ns",
        "finish_ns",
        "ready_seen_ns",
        "epoch",
        "sequence",
        "slot",
        "phase",
        "stage",
    ]
    report["trace_record_count"] = len(records)
    try:
        metadata = json.loads(trace_path.with_suffix(".metadata.json").read_text())
        report["trace_sequence_counts"] = metadata["trace_sequence_counts"]
        report["trace_truncated_ctas"] = trace_overflow_ctas(
            metadata, trace.shape[0], args.trace_capacity
        )
        decision_path = trace_path.with_suffix(".decisions.pt")
        decision_tensor = torch.load(decision_path, weights_only=True)
        if decision_tensor.shape != trace.shape:
            raise ValueError("Decision trace shape differs from tile trace")
        decisions = [
            dict(zip(DECISION_FIELDS, values)) | {"cta": cta, "claim_sequence": seq}
            for cta, row in enumerate(decision_tensor.tolist())
            for seq, values in enumerate(row)
            if values[0]
        ]
        report["decision_trace_file"] = str(decision_path)
        report["decision_trace_fields"] = list(DECISION_FIELDS)
        report["decision_trace_checks"] = validate_decision_trace(
            records,
            decisions,
            report["executor"]["cluster_size"],
            report["trace_truncated_ctas"],
        )
        report["cooperative_trace_checks"] = validate_cluster_trace(
            records, report["executor"]["cluster_size"], report["trace_truncated_ctas"]
        )
        progress_path = trace_path.with_suffix(".progress.pt")
        progress_tensor = torch.load(progress_path, weights_only=True)
        if tuple(progress_tensor.shape) != (*trace.shape[:2], len(PROGRESS_FIELDS)):
            raise ValueError("Progress trace shape differs from tile trace")
        progress = [
            dict(zip(PROGRESS_FIELDS, values)) | {"cta": cta, "sequence": seq}
            for cta, row in enumerate(progress_tensor.tolist())
            for seq, values in enumerate(row)
            if any(values)
        ]
        report["progress_trace_file"] = str(progress_path)
        report["progress_trace_fields"] = list(PROGRESS_FIELDS)
        report["progress_trace_checks"] = validate_progress_trace(
            records,
            decisions,
            progress,
            report["executor"]["cluster_size"],
            report["trace_truncated_ctas"],
        )
        if args.check_single_pending_compute:
            report["pending_compute_checks"] = validate_single_pending_compute(
                report["progress_trace_checks"]
            )
    except (OSError, ValueError, KeyError) as exc:
        report.update(status="failed", error=str(exc))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise

    def distribution(values):
        if not values:
            return None
        values = sorted(values)
        return {
            "count": len(values),
            "min_us": values[0],
            "p50_us": values[len(values) // 2],
            "p99_us": values[min(len(values) - 1, int(len(values) * 0.99))],
            "max_us": values[-1],
        }

    report["tile_lifetime_us"] = {
        f"phase{phase}_stage{stage}": distribution(
            [
                (r["finish_ns"] - r["begin_ns"]) / 1000
                for r in records
                if r["phase"] == phase and r["stage"] == stage
            ]
        )
        for phase in (0, 1)
        for stage in sorted({r["stage"] for r in records})
    }
    for case in report["cases"]:
        first = {}
        for r in records:
            if r["phase"] == 1 and r["epoch"] == case["decode"]["epoch"]:
                first.setdefault(r["cta"], r)
        case["decode_admission_to_first_tile_us"] = distribution(
            [(r["begin_ns"] - r["ready_seen_ns"]) / 1000 for r in first.values()]
        )
    report["trace_interpretation"] = (
        "begin is the producer handoff after phase-switch drain; finish is epilogue retirement. "
        "Lifetime includes pipelining waits. Admission is CTA0's observation, not A publication. "
        "Admission roundtrip bounds use only A's clock and include the return visibility/poll delay. "
        "First-D-tile statistics cover only CTAs that received D work; they do not prove "
        "the full system-scope propagation bound. Decision snapshots precede policy acquisition "
        "and tile claim; their E-clock observation-to-handoff intervals include arbitration and "
        "phase drain, but exclude earlier A-to-E propagation. Retirement prefixes are sampled "
        "after the timestamp; -1 means unavailable. Tracing perturbs all measured latencies. "
        "Progress snapshots distinguish epilogue-observed accumulator readiness from output "
        "submission and final retirement, without equating a pending store to an unfinished GEMM. "
        "Raw traces preserve all records."
    )
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "trace_records"}, indent=2)
    )


if __name__ == "__main__":
    main()
