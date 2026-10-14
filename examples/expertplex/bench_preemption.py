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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.prefill_rows, args.decode_rows, args.repeats) <= 0:
        parser.error("Row/repeat counts must be positive")
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
        trace_path=str(trace_path),
    )
    owner = Exporter(settings, 1, execute=True, timeout_s=600)
    owner.__enter__()
    torch.cuda.set_device(0)
    peers = [native.RemoteBuffers(owner.descriptor, slot, 0) for slot in (0, 1)]
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
        native.dispatch(
            peers[slot], x, ids, row_map, 0, 0, slot, 0, args.world, publish=publish
        )

    def pull(slot, case):
        _, ids, weights, row_map, out = case
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
            isolated, references = [], []
            for slot, case in enumerate(cases):
                push(slot, case)
                isolated.append(pull(slot, case))
                references.append(case[-1].clone())
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
        for peer in peers:
            peer.close()
        owner.__exit__(None)
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
    report["trace_truncated_ctas"] = (
        int((trace[:, -1, 0] != 0).sum()) if args.trace_capacity else 0
    )

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
        "the full system-scope propagation bound. Raw trace preserves all records."
    )
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "trace_records"}, indent=2)
    )


if __name__ == "__main__":
    main()
