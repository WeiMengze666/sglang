"""Native DeepGEMM BF16 expert-compute baseline, excluding communication.

This is a microbenchmark for the B300 adaptation of paper section 7.4.
It does not measure serving goodput. Routing matches bench_native.py's seed
and distribution; contiguous padding follows DeepGEMM's 128-row contract.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=2)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--intermediate", type=int, default=2560)
    p.add_argument("--experts", type=int, default=40)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 256, 2048])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import deep_gemm
    import torch

    from sglang.kernels.ops.activation.activation import silu_and_mul

    torch.cuda.set_device(args.device)
    torch.set_num_threads(8)
    h, inner, experts = args.hidden, args.intermediate, args.experts
    torch.manual_seed(20261014)
    w13 = torch.empty(
        experts, 2 * inner, h, device="cuda", dtype=torch.bfloat16
    ).normal_(std=0.02)
    w2 = torch.empty(experts, h, inner, device="cuda", dtype=torch.bfloat16).normal_(
        std=0.02
    )
    report = dict(
        scope="Native DeepGEMM BF16 GEMM1 + SiLU/mul + GEMM2; no dispatch/combine",
        settings=vars(args) | {"output": str(args.output)},
        pytorch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        cases=[],
    )
    for rows in args.rows:
        gen = torch.Generator(device=args.device).manual_seed(rows)
        x = torch.randn(rows, h, generator=gen, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.rand(rows, experts * args.world, generator=gen, device="cuda")
            .topk(args.top_k, dim=-1)
            .indices
        )
        counts = [int((ids == e * args.world).sum()) for e in range(experts)]
        for phase in (0, 1):
            if phase == 0:
                padded = [(c + 127) // 128 * 128 for c in counts]
                total = max(128, sum(padded))
                a = torch.zeros(total, h, dtype=torch.bfloat16, device="cuda")
                layout = torch.full((total,), -1, dtype=torch.int32, device="cuda")
                cursor = 0
                for expert, (count, size) in enumerate(zip(counts, padded)):
                    indices = (ids == expert * args.world).nonzero()[:, 0]
                    a[cursor : cursor + count] = x[indices]
                    layout[cursor : cursor + size] = expert
                    cursor += size
                gateup = torch.empty(
                    total, 2 * inner, dtype=torch.bfloat16, device="cuda"
                )
                activated = torch.empty(
                    total, inner, dtype=torch.bfloat16, device="cuda"
                )
                out = torch.empty(total, h, dtype=torch.bfloat16, device="cuda")

                def invoke():
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w13, gateup, layout)
                    silu_and_mul(gateup, activated)
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        activated, w2, out, layout
                    )
            else:
                capacity = max(128, (max(counts) + 127) // 128 * 128)
                a = torch.zeros(
                    experts, capacity, h, dtype=torch.bfloat16, device="cuda"
                )
                for expert, count in enumerate(counts):
                    a[expert, :count] = x[(ids == expert * args.world).nonzero()[:, 0]]
                layout = torch.tensor(counts, dtype=torch.int32, device="cuda")
                gateup = torch.zeros(
                    experts, capacity, 2 * inner, dtype=torch.bfloat16, device="cuda"
                )
                activated = torch.empty(
                    experts, capacity, inner, dtype=torch.bfloat16, device="cuda"
                )
                out = torch.zeros(
                    experts, capacity, h, dtype=torch.bfloat16, device="cuda"
                )
                expected = max(1, math.ceil(sum(counts) / experts))

                def invoke():
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(
                        a, w13, gateup, layout, expected
                    )
                    silu_and_mul(gateup.view(-1, 2 * inner), activated.view(-1, inner))
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(
                        activated, w2, out, layout, expected
                    )

            invoke()  # Compile outside graph/timing.
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                invoke()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.iterations):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            finite = bool(torch.isfinite(out).all())
            if not finite:
                raise AssertionError("Nonfinite native baseline output")
            item = dict(
                rows=rows,
                phase=phase,
                expert_rows=counts,
                median_ms=statistics.median(samples),
                samples_ms=samples,
                finite=finite,
            )
            report["cases"].append(item)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                {
                    k: v
                    for k, v in item.items()
                    if k not in ("samples_ms", "expert_rows")
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
