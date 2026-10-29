"""Validate persistent E-side scatter/gather against native BF16 DeepGEMM.

Two expert GPUs test modulo expert placement and the normal-mode per-rank BF16
EP gather boundary. Dynamic routing, all/partial padding, mixed P/D mailboxes,
fixed-address CUDA Graph replay, and every persistent operation are exercised.
This is operator validation, not full-model accuracy or serving performance.
"""

import argparse
import json
import math
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import module
from check_gpu import Exporter, _numerics, _prepare, _reference_deepgemm, _weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2560)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--graph-replays", type=int, default=20)
    parser.add_argument(
        "--second-prefill",
        action="store_true",
        help="Stress two concurrent raw prefill mailboxes against native arithmetic",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.world <= 4 or args.capacity < 128 or args.experts < 2:
        parser.error("world 1..4, capacity >=128 and experts >=2 are required")
    if not 1 <= args.top_k <= args.experts * args.world:
        parser.error("top_k must fit the global expert count")
    import torch

    torch.cuda.set_device(0)
    native = module("native")
    native.load_extension()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": __doc__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args) | {"output": str(args.output)},
        "native_source_sha256": native.source_fingerprint(),
        "cases": [],
        "status": "running",
        "tolerances": {"atol": 0.005, "rtol": 0.02, "relative_l2": 0.002},
    }
    phases = [0, 0 if args.second_prefill else 1]
    settings = [
        dict(
            hidden=args.hidden,
            intermediate=args.intermediate,
            layers=2,
            experts=args.experts,
            capacity=args.capacity,
            top_k=args.top_k,
            seed=1234 + rank,
            weight_std=0.02,
            capacities=[args.capacity, args.capacity],
            packed_layout=[1, int(args.second_prefill)],
            prefill_preprocess=True,
            expert_rank=rank,
            expert_world=args.world,
            trace_capacity=8192,
            trace_path=str(args.output.with_suffix(f".rank{rank}.trace.pt")),
            decode_budget=64,
            expected_tiles=8,
        )
        for rank in range(args.world)
    ]
    weights = [_weights(torch, item) for item in settings]
    tolerance = SimpleNamespace(**report["tolerances"])

    def make_case(n, seed, padding=False, skew=False):
        gen = torch.Generator().manual_seed(seed)
        ids = (
            torch.rand(n, args.experts * args.world, generator=gen)
            .argsort(-1)[:, : args.top_k]
            .int()
            .contiguous()
        )
        if skew:
            ids[:] = torch.arange(args.top_k)
        if padding:
            ids[:] = -1
        else:
            ids[::7] = -1
        return dict(
            hidden=(torch.randn(n, args.hidden, generator=gen) * 0.5).bfloat16(),
            ids=ids,
            weights=torch.rand(n, args.top_k, generator=gen).softmax(-1),
            layer=seed % 2,
        )

    def reference(case, phase):
        phase = phases[phase]
        partials = []
        for rank, (w13, w2) in enumerate(weights):
            ids = case["ids"].clone()
            local = (ids >= 0) & (ids.remainder(args.world) == rank)
            ids = torch.where(local, ids // args.world, -1)
            partials.append(
                _reference_deepgemm(
                    torch,
                    case | {"ids": ids},
                    w13,
                    w2,
                    settings[rank],
                    0,
                    phase,
                    fp32_combine=phase == 1,
                ).float()
            )
        # P normal gather explicitly rounds each rank. D masked combine has
        # a different reduction boundary; its error is reported separately.
        return torch.stack(partials).sum(0).bfloat16()

    def verify(actual, expected, label, phase):
        result = _numerics(torch, actual.cpu(), expected, tolerance)
        result["exact_fraction"] = float((actual.cpu() == expected).float().mean())
        report["cases"].append(dict(label=label, phase=phase, **result))
        if not result["passed"]:
            raise AssertionError(report["cases"][-1])

    try:
        with ExitStack() as stack:
            owners = [
                stack.enter_context(
                    Exporter(item, 4 + rank, execute=True, timeout_s=300)
                )
                for rank, item in enumerate(settings)
            ]
            peers = [
                [native.RemoteBuffers(owner.descriptor, slot, 0) for owner in owners]
                for slot in range(2)
            ]
            groups = [native.RemoteGroup(p) for p in peers]
            streams = [torch.cuda.Stream() for _ in range(2)]
            report["executors"] = [owner.executor_info for owner in owners]
            try:

                def execute(prepared, phase):
                    groups[phase].dispatch(
                        prepared["hidden"],
                        prepared["ids"],
                        prepared["row_map"],
                        prepared["layer"],
                        phases[phase],
                        prepared["weights"],
                    )
                    prepared["output"].zero_()
                    groups[phase].combine(
                        prepared["output"],
                        prepared["ids"],
                        prepared["weights"],
                        prepared["row_map"],
                        timeout_ms=30000,
                    )
                    return prepared["output"].bfloat16()

                for iteration, n in enumerate(
                    [1, 31, 128, args.capacity - 1, args.capacity]
                ):
                    case = make_case(
                        n, 40 + iteration, padding=iteration == 1, skew=iteration == 4
                    )
                    expected = reference(case, 0)
                    prepared = _prepare(torch, case, 0)
                    if iteration % 2:
                        prepared["ids"] = prepared["ids"].long()
                    actual = execute(prepared, 0)
                    torch.cuda.synchronize()
                    verify(actual, expected, f"eager-{n}", 0)

                cases = [
                    make_case(args.capacity, 60 + i, padding=i == 2, skew=i == 3)
                    for i in range(4)
                ]
                prepared = _prepare(torch, cases[0], 0)
                # Graph reuses a single layer; change routes/data, not Python arguments.
                for c in cases:
                    c["layer"] = prepared["layer"]
                expected = [reference(c, 0) for c in cases]
                with torch.cuda.stream(streams[0]):
                    execute(prepared, 0)
                streams[0].synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=streams[0]):
                    captured_output = execute(prepared, 0)
                for iteration in range(args.graph_replays):
                    index = iteration % len(cases)
                    with torch.cuda.stream(streams[0]):
                        for key in ("hidden", "ids", "weights"):
                            prepared[key].copy_(cases[index][key])
                        graph.replay()
                    streams[0].synchronize()
                    verify(captured_output, expected[index], f"graph-{iteration}", 0)

                # Both mailboxes are pending together. D uses the same weights
                # and native masked templates while P traverses all six stages.
                pair = [
                    make_case(args.capacity, 82),
                    make_case(args.capacity - 17 if args.second_prefill else 17, 84),
                ]
                refs = [reference(c, phase) for phase, c in enumerate(pair)]
                for _ in range(5):
                    buffers = [_prepare(torch, c, 0) for c in pair]
                    torch.cuda.synchronize()
                    outputs = []
                    for phase in range(2):
                        with torch.cuda.stream(streams[phase]):
                            outputs.append(execute(buffers[phase], phase))
                    torch.cuda.synchronize()
                    for phase in range(2):
                        verify(outputs[phase], refs[phase], f"concurrent-{_}", phase)
            finally:
                torch.cuda.synchronize()
                groups.clear()
                for slot_peers in peers:
                    for peer in slot_peers:
                        peer.close()
        stages = set()
        for item in settings:
            trace = torch.load(item["trace_path"], weights_only=False)
            report.setdefault("trace_files", []).append(item["trace_path"])
            # Trace tensor columns: begin/end/admission/epoch/sequence/slot/phase/stage.
            tensor = trace["records"] if isinstance(trace, dict) else trace
            stages.update(int(x) for x in tensor[..., 7].flatten().unique().tolist())
            valid = tensor.reshape(-1, 8)
            valid = valid[(valid[:, 0] > 0) & (valid[:, 5] == 0)]
            observed = Counter((int(r[3]), int(r[7])) for r in valid)
            lengths = [1, 31, 128, args.capacity - 1, args.capacity] + [
                args.capacity
            ] * (1 + args.graph_replays + 5)
            for epoch, length in enumerate(lengths, 1):
                for stage, bound in (
                    (5, math.ceil(length * args.top_k / 512) * 2),
                    (6, math.ceil(length * args.top_k / 64) * 2),
                    (7, math.ceil(length / 8) * 2),
                ):
                    if observed[epoch, stage] != bound:
                        raise AssertionError(
                            f"Operation task count mismatch: epoch={epoch}, stage={stage}, actual={observed[epoch, stage]}, expected={bound}"
                        )
            report.setdefault("trace_task_counts", []).append(
                {
                    "passed": True,
                    "invocations": len(lengths),
                    "diagnostic_tile_interval_us": {
                        str(stage): {
                            "median": float(
                                torch.quantile(
                                    (
                                        valid[valid[:, 7] == stage, 1]
                                        - valid[valid[:, 7] == stage, 0]
                                    ).double()
                                    / 1000,
                                    0.5,
                                )
                            ),
                            "maximum": float(
                                (
                                    (
                                        valid[valid[:, 7] == stage, 1]
                                        - valid[valid[:, 7] == stage, 0]
                                    ).double()
                                    / 1000
                                ).max()
                            ),
                        }
                        for stage in (1, 2, 3, 5, 6, 7)
                    },
                    "interval_scope": "producer handoff to epilogue retirement; includes pipeline wait, excludes attention publication propagation",
                }
            )
        report["observed_stages"] = sorted(stages)
        if not {1, 2, 3, 5, 6, 7} <= stages:
            raise AssertionError(f"Persistent operation traces incomplete: {stages}")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
