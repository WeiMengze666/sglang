"""All-operation tile interval diagnostic for paper section 7.6.

P has a controlled per-expert row count. D128 becomes ready 10 us later on an
attention GPU. Raw payloads are staged before release; E scatter/gather remain
inside the measured persistent invocation. No clocks from distinct GPUs are
subtracted. Handoff-to-retirement intervals include pipeline waits and exclude
A-to-CTA0 propagation; these alone do not prove the full reallocation bound.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from _bootstrap import module
from check_gpu import Exporter, _prepare


STAGES = {
    1: "gate_up",
    2: "down",
    3: "activation",
    5: "scatter1",
    6: "scatter2",
    7: "gather",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expert-rows", type=int, nargs="+", default=[128, 256, 512, 1024, 2048]
    )
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--delay-us", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(*args.expert_rows, args.iterations) <= 0 or args.delay_us < 0:
        p.error("Positive rows/iterations and nonnegative delay are required")
    import torch

    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    native = module("native")
    native.load_extension()
    # Forty local experts match one E rank of the 480B/4E deployment. The
    # diagnostic routes exactly x rows to every local expert, without network
    # peers; this is a controlled operator-size sweep, not a serving load.
    settings = dict(
        hidden=6144,
        intermediate=2560,
        experts=40,
        top_k=8,
        layers=1,
        capacity=max(args.expert_rows) * 5,
        capacities=[max(args.expert_rows) * 5, 128],
        packed_layout=[1, 0],
        prefill_preprocess=True,
        seed=1234,
        weight_std=0.02,
        decode_budget=64,
        expected_tiles=8,
        trace_capacity=32768,
        trace_path=str(args.output.with_suffix(".trace.pt")),
    )
    report = dict(
        scope=__doc__,
        settings=vars(args) | {"output": str(args.output)},
        native_source_sha256=native.source_fingerprint(),
        samples=[],
        status="running",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    epoch_rows = {}
    try:
        with Exporter(settings, 4, execute=True, timeout_s=600) as owner:
            report["executor"] = owner.executor_info
            peers = [
                native.RemoteBuffers(owner.descriptor, phase, 0) for phase in range(2)
            ]
            groups = [native.RemoteGroup([peer]) for peer in peers]
            try:
                for size in args.expert_rows:
                    n = size * 5
                    cases = []
                    for phase, rows in enumerate([n, 128]):
                        ids = (
                            torch.arange(rows)[:, None] * 8 + torch.arange(8)[None, :]
                        ) % 40
                        if phase:
                            ids[:] = torch.arange(8)
                        gen = torch.Generator().manual_seed(1949 + phase + size)
                        case = dict(
                            hidden=(
                                torch.randn(rows, 6144, generator=gen) * 0.5
                            ).bfloat16(),
                            ids=ids.int().contiguous(),
                            weights=torch.full((rows, 8), 0.125),
                            layer=0,
                        )
                        cases.append(_prepare(torch, case, 0))
                    torch.cuda.synchronize()
                    for repeat in range(args.iterations):
                        for phase, case in enumerate(cases):
                            groups[phase].dispatch(
                                case["hidden"],
                                case["ids"],
                                case["row_map"],
                                0,
                                phase,
                                case["weights"],
                                publish=False,
                            )
                        stamps = native.publish_pair(*peers, delay_us=args.delay_us)
                        for phase in (1, 0):
                            case = cases[phase]
                            case["output"].zero_()
                            groups[phase].combine(
                                case["output"],
                                case["ids"],
                                case["weights"],
                                case["row_map"],
                                timeout_ms=30000,
                            )
                        timings = [
                            native.read_timing(peer).cpu().tolist() for peer in peers
                        ]
                        release = stamps.cpu().tolist()
                        if not all(
                            bool(torch.isfinite(case["output"]).all()) for case in cases
                        ):
                            raise AssertionError("Nonfinite diagnostic output")
                        epoch_rows[int(timings[0][0])] = size
                        report["samples"].append(
                            dict(
                                expert_rows=size,
                                repeat=repeat,
                                attention_release_gap_us=(release[1] - release[0])
                                / 1000,
                                attention_admission_ack_upper_bound_us=[
                                    (release[i + 2] - release[i]) / 1000
                                    if release[i + 2]
                                    else None
                                    for i in range(2)
                                ],
                                expert_timings_ns=timings,
                            )
                        )
                    print("completed per-expert rows", size, flush=True)
            finally:
                torch.cuda.synchronize()
                groups.clear()
                for peer in peers:
                    peer.close()
        trace = torch.load(settings["trace_path"], weights_only=True).reshape(-1, 8)
        trace = trace[(trace[:, 0] > 0) & (trace[:, 1] > 0)]
        report["intervals"] = []
        for size in args.expert_rows:
            epochs = [e for e, x in epoch_rows.items() if x == size]
            selected = trace[
                torch.isin(trace[:, 3], torch.tensor(epochs)) & (trace[:, 5] == 0)
            ]
            counts = Counter((int(r[3]), int(r[7])) for r in selected)
            for epoch in epochs:
                for stage, expected in (
                    (5, math.ceil(size * 40 / 512) * 2),
                    (6, math.ceil(size * 40 / 64) * 2),
                    (7, math.ceil(size * 5 / 8) * 2),
                ):
                    if counts[epoch, stage] != expected:
                        raise AssertionError(
                            f"Truncated trace or task count mismatch: {epoch=} {stage=} {counts[epoch, stage]=} {expected=}"
                        )
            for stage, name in STAGES.items():
                rows = selected[selected[:, 7] == stage]
                if not len(rows):
                    raise AssertionError(f"No {name} samples for {size} rows")
                intervals = (rows[:, 1] - rows[:, 0]).double() / 1000
                report["intervals"].append(
                    dict(
                        expert_rows=size,
                        operation=name,
                        tiles=len(rows),
                        median_us=float(intervals.median()),
                        p99_us=float(torch.quantile(intervals, 0.99)),
                        max_us=float(intervals.max()),
                    )
                )
        # Admission-to-first-D handoff uses E's clock at both endpoints. It
        # excludes the A publication-to-admission segment and is not the total
        # preemption/reallocation delay stated in section 4.3.
        decode = trace[trace[:, 6] == 1]
        report["decode_first_handoff_after_admission_us"] = [
            float(
                (
                    decode[decode[:, 3] == epoch, 0].min()
                    - decode[decode[:, 3] == epoch, 2].min()
                )
                / 1000
            )
            for epoch in sorted(set(decode[:, 3].tolist()))
        ]
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
