"""Measure one expert rank with synthetic routing; not end-to-end goodput."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from _bootstrap import module
from check_gpu import Exporter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--intermediate", type=int, default=2560)
    p.add_argument("--experts", type=int, default=40)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 256, 2048])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch

    torch.set_num_threads(8)
    native = module("native")
    native.load_extension()
    settings = dict(
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        layers=1,
        capacity=max(args.rows),
        top_k=args.top_k,
        seed=20261014,
        weight_std=0.02,
    )
    report = {
        "scope": "synthetic single-layer single-expert-rank microbenchmark, not model throughput",
        "kernel": os.environ.get("SGLANG_EXPERTPLEX_KERNEL", "sm100"),
        "native_source_sha256": native.source_fingerprint(),
        "settings": settings,
        "expert_world": args.world,
        "cases": [],
    }
    with Exporter(settings, 1, execute=True, timeout_s=600) as owner:
        report["executor"] = owner.executor_info
        peer = native.RemoteBuffers(owner.descriptor, 0, 0)
        try:
            for rows in args.rows:
                generator = torch.Generator(device=0).manual_seed(rows)
                x = torch.randn(
                    rows,
                    args.hidden,
                    generator=generator,
                    device=0,
                    dtype=torch.bfloat16,
                )
                scores = torch.rand(
                    rows, args.experts * args.world, generator=generator, device=0
                )
                ids = scores.topk(args.top_k, dim=-1).indices.int().contiguous()
                weights = torch.full((rows, args.top_k), 1 / args.top_k, device=0)
                row_map = torch.empty_like(ids)
                out = torch.zeros(rows, args.hidden, device=0)
                for phase in (0, 1):
                    stream = torch.cuda.Stream(device=0)
                    stream.wait_stream(torch.cuda.current_stream())

                    def invoke():
                        out.zero_()
                        native.dispatch(
                            peer, x, ids, row_map, 0, 0, phase, 0, args.world
                        )
                        native.combine(
                            peer,
                            out,
                            ids,
                            weights,
                            row_map,
                            0,
                            0,
                            args.world,
                            synchronize=False,
                        )

                    with torch.cuda.stream(stream):
                        invoke()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        invoke()
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize(0)
                    times = []
                    for _ in range(args.iterations):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                    result = dict(
                        rows=rows,
                        phase=phase,
                        median_ms=statistics.median(times),
                        min_ms=min(times),
                        max_ms=max(times),
                        samples_ms=times,
                        finite=bool(torch.isfinite(out).all()),
                        local_routes=int((ids % args.world == 0).sum()),
                    )
                    components = []
                    for _ in range(5):
                        out.zero_()
                        start, pushed, pulled = [
                            torch.cuda.Event(enable_timing=True) for _ in range(3)
                        ]
                        start.record()
                        native.dispatch(
                            peer, x, ids, row_map, 0, 0, phase, 0, args.world
                        )
                        pushed.record()
                        native.combine(
                            peer,
                            out,
                            ids,
                            weights,
                            row_map,
                            0,
                            0,
                            args.world,
                            synchronize=False,
                        )
                        pulled.record()
                        stamps = native.read_timing(peer).cpu().tolist()
                        if stamps[0] != stamps[1] or stamps[3] < stamps[2]:
                            raise AssertionError("Inconsistent expert timing epoch")
                        components.append(
                            dict(
                                dispatch_ms=start.elapsed_time(pushed),
                                wait_and_pull_ms=pushed.elapsed_time(pulled),
                                expert_operation_ms=(stamps[3] - stamps[2]) / 1e6,
                            )
                        )
                    result["components"] = components
                    report["cases"].append(result)
                    if not result["finite"]:
                        raise AssertionError(
                            "Nonfinite output invalidates this measurement"
                        )
                    print(
                        json.dumps(
                            {k: v for k, v in result.items() if k != "samples_ms"}
                        ),
                        flush=True,
                    )
        finally:
            peer.close()
    report["finished_unix"] = time.time()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
