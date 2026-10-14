"""Native BF16 expert sharing ablations for paper section 7.3.

Exclusive, priority streams and Green Context use the same weights and routed
inputs. A separate GPU publishes ready flags 10 us apart; CUDA stream memory
operations wait without reserving an E-side polling SM. Response times are
publication-to-completion-ack upper bounds on that single control GPU's clock.
This is a single-E compute microbenchmark, not serving goodput. MPS requires a
separate multi-process experiment and is not represented by priority streams.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def workload(args, rows, phase, w13, w2, *, return_buffers=False):
    import deep_gemm
    import torch

    from sglang.kernels.ops.activation.activation import silu_and_mul

    h, inner, experts = args.hidden, args.intermediate, args.experts
    gen = torch.Generator(device=args.device).manual_seed(rows)
    x = torch.randn(rows, h, generator=gen, device="cuda", dtype=torch.bfloat16)
    ids = (
        torch.rand(rows, experts * args.world, generator=gen, device="cuda")
        .topk(args.top_k, dim=-1)
        .indices
    )
    counts = [int((ids == e * args.world).sum()) for e in range(experts)]
    forced = False
    if phase == 0 and getattr(args, "prefill_expert_rows", None) is not None:
        counts = [args.prefill_expert_rows] * experts
        forced = True
    if phase == 1 and getattr(args, "decode_active_experts", None) is not None:
        counts = [1] * args.decode_active_experts + [0] * (
            experts - args.decode_active_experts
        )
        forced = True

    def source_rows(expert, count):
        if forced:
            return torch.arange(count, device=args.device) % rows
        return (ids == expert * args.world).nonzero()[:, 0]

    if phase == 0:
        padded = [(c + 127) // 128 * 128 for c in counts]
        total = max(128, sum(padded))
        a = torch.zeros(total, h, dtype=torch.bfloat16, device="cuda")
        layout = torch.full((total,), -1, dtype=torch.int32, device="cuda")
        cursor = 0
        for expert, (count, size) in enumerate(zip(counts, padded)):
            a[cursor : cursor + count] = x[source_rows(expert, count)]
            layout[cursor : cursor + size] = expert
            cursor += size
        gateup = torch.empty(total, 2 * inner, dtype=torch.bfloat16, device="cuda")
        activated = torch.empty(total, inner, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(total, h, dtype=torch.bfloat16, device="cuda")

        def invoke():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w13, gateup, layout)
            silu_and_mul(gateup, activated)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(activated, w2, out, layout)

        if args.operation == "gateup":

            def invoke():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w13, gateup, layout)

            out = gateup
        elif args.operation == "down":
            invoke()  # Prepare the activation outside the timed operation.

            def invoke():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(activated, w2, out, layout)
    else:
        capacity = max(128, (max(counts) + 127) // 128 * 128)
        a = torch.zeros(experts, capacity, h, dtype=torch.bfloat16, device="cuda")
        for expert, count in enumerate(counts):
            a[expert, :count] = x[source_rows(expert, count)]
        layout = torch.tensor(counts, dtype=torch.int32, device="cuda")
        gateup = torch.zeros(
            experts, capacity, 2 * inner, dtype=torch.bfloat16, device="cuda"
        )
        activated = torch.empty(
            experts, capacity, inner, dtype=torch.bfloat16, device="cuda"
        )
        out = torch.zeros(experts, capacity, h, dtype=torch.bfloat16, device="cuda")
        expected = max(1, math.ceil(sum(counts) / experts))

        def invoke():
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, w13, gateup, layout, expected)
            silu_and_mul(gateup.view(-1, 2 * inner), activated.view(-1, inner))
            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                activated, w2, out, layout, expected
            )

        if args.operation == "gateup":

            def invoke():
                deep_gemm.m_grouped_bf16_gemm_nt_masked(
                    a, w13, gateup, layout, expected
                )

            out = gateup
        elif args.operation == "down":
            invoke()

            def invoke():
                deep_gemm.m_grouped_bf16_gemm_nt_masked(
                    activated, w2, out, layout, expected
                )

    if return_buffers:
        return invoke, out, counts, {"input": a, "activation": activated}
    return invoke, out, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--control-device", type=int, default=0)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--intermediate", type=int, default=2560)
    p.add_argument("--experts", type=int, default=40)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--prefill-rows", type=int, default=8192)
    p.add_argument("--decode-rows", type=int, default=128)
    p.add_argument("--delay-us", type=int, default=10)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--prefill-expert-rows",
        type=int,
        help="Controlled per-expert P rows for section 7.4",
    )
    p.add_argument(
        "--decode-active-experts",
        type=int,
        help="Controlled active D experts, one row each",
    )
    p.add_argument(
        "--num-sms", type=int, help="Native kernel SM limit for offline profiles"
    )
    p.add_argument(
        "--operation",
        choices=("gateup", "down", "moe"),
        default="gateup",
        help="Paper 7.3 compares grouped GEMMs; moe is an additional whole-MoE ablation",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=("exclusive", "priority", "green"),
        default=["exclusive", "priority", "green"],
    )
    p.add_argument(
        "--green-divisions",
        nargs="+",
        default=["112:32", "96:48", "80:64", "64:80", "32:112"],
    )
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.prefill_expert_rows is not None and args.prefill_expert_rows <= 0:
        p.error("Prefill expert rows must be positive")
    if (
        args.decode_active_experts is not None
        and not 1 <= args.decode_active_experts <= args.experts
    ):
        p.error("Invalid decode active expert count")
    if args.num_sms is not None and (
        args.num_sms < 2 or args.num_sms % 2 or args.modes != ["exclusive"]
    ):
        p.error("SM-limited profiles require an even SM count and --modes exclusive")
    if (
        args.device == args.control_device
        or min(args.iterations, args.prefill_rows, args.decode_rows) <= 0
    ):
        p.error("Use distinct control/expert GPUs and positive workload sizes")
    import deep_gemm
    import torch
    from torch.utils.cpp_extension import load

    torch.set_num_threads(8)
    control = load(
        name="expertplex_sharing_control",
        sources=[str(Path(__file__).with_name("sharing_control.cu"))],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"],
        extra_ldflags=["-lcuda"],
    )
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261014)
    w13 = torch.empty(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(std=0.02)
    w2 = torch.empty(
        args.experts,
        args.hidden,
        args.intermediate,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(std=0.02)
    works = [
        workload(args, rows, phase, w13, w2)
        for phase, rows in enumerate([args.prefill_rows, args.decode_rows])
    ]
    flags = torch.zeros(4, dtype=torch.int32, device=args.device)
    stamps = torch.zeros(4, dtype=torch.int64, device=args.control_device)
    total_sms = torch.cuda.get_device_properties(args.device).multi_processor_count
    if args.num_sms is not None:
        if args.num_sms > total_sms:
            p.error("SM limit exceeds the device")
        total_sms = args.num_sms
    original_sms = deep_gemm.get_num_sms()
    report = {
        "scope": __doc__,
        "settings": vars(args) | {"output": str(args.output)},
        "gpu": torch.cuda.get_device_name(),
        "native_kernel_sm_limit": total_sms,
        "expert_rows": [w[2] for w in works],
        "cases": [],
    }

    def capture(stream, sms, invoke):
        deep_gemm.set_num_sms(sms)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            invoke()
        stream.synchronize()
        return graph

    def measure(streams, graphs, enabled=(True, True)):
        flags.zero_()
        stamps.zero_()
        torch.cuda.synchronize(args.device)
        torch.cuda.synchronize(args.control_device)
        # Explicitly order all measured work after flag initialization. The
        # peer publisher runs on another device and cannot be starved by E.
        initialized = torch.cuda.Event()
        initialized.record()
        for index, (stream, graph) in enumerate(zip(streams, graphs)):
            stream.wait_event(initialized)
            control.wait_flag(flags, index, stream.cuda_stream)
            with torch.cuda.stream(stream):
                if enabled[index]:
                    graph.replay()
            control.done(flags, index + 2, stream.cuda_stream)
        control.run(flags, stamps, args.control_device, args.delay_us * 1000)
        torch.cuda.synchronize(args.control_device)
        values = stamps.cpu().tolist()
        if not all(values) or values[2] < values[0] or values[3] < values[1]:
            raise RuntimeError(f"GPU release/ack timed out: {values}")
        for stream in streams:
            stream.synchronize()
        return {
            "release_gap_us": (values[1] - values[0]) / 1000,
            "response_upper_bound_us": [
                (values[2] - values[0]) / 1000,
                (values[3] - values[1]) / 1000,
            ],
        }

    try:
        base_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        base_graphs = [
            capture(stream, total_sms, work[0])
            for stream, work in zip(base_streams, works)
        ]
        for _ in range(3):
            measure(base_streams, base_graphs)
        isolated = [
            [
                measure(base_streams, base_graphs, tuple(i == phase for i in range(2)))
                for _ in range(args.iterations)
            ]
            for phase in range(2)
        ]
        reference = [
            statistics.median(
                r["response_upper_bound_us"][phase] for r in isolated[phase]
            )
            for phase in range(2)
        ]
        report["isolated_samples"] = isolated
        report["isolated_response_upper_bound_us"] = reference
        configurations = [(mode, None) for mode in args.modes if mode != "green"]
        if "green" in args.modes:
            configurations += [
                ("green", tuple(map(int, value.split(":"))))
                for value in args.green_divisions
            ]
        for mode, division in configurations:
            if mode == "green":
                import sgl_kernel.spatial  # noqa: F401 - registers native operator

                raw = torch.ops.sgl_kernel.create_greenctx_stream_by_value(
                    *division, args.device
                )
                streams = [
                    torch.cuda.ExternalStream(raw[i], device=args.device)
                    for i in range(2)
                ]
                sms = list(raw[2:])
            elif mode == "priority":
                streams = [
                    torch.cuda.Stream(priority=0),
                    torch.cuda.Stream(priority=-1),
                ]
                sms = [total_sms, total_sms]
            else:
                stream = torch.cuda.Stream()
                streams, sms = [stream, stream], [total_sms, total_sms]
            graphs = [
                capture(stream, count, work[0])
                for stream, count, work in zip(streams, sms, works)
            ]
            for _ in range(3):
                measure(streams, graphs)
            samples = [measure(streams, graphs) for _ in range(args.iterations)]
            response = [
                statistics.median(r["response_upper_bound_us"][i] for r in samples)
                for i in range(2)
            ]
            item = {
                "mode": mode,
                "requested_sms": division,
                "actual_sms": sms,
                "response_upper_bound_us": response,
                "slowdown_vs_isolated_at_same_sm_limit": [
                    a / b for a, b in zip(response, reference)
                ],
                "samples": samples,
                "finite_outputs": [bool(torch.isfinite(w[1]).all()) for w in works],
            }
            if not all(item["finite_outputs"]):
                raise RuntimeError("Nonfinite native outputs")
            report["cases"].append(item)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print({k: v for k, v in item.items() if k != "samples"}, flush=True)
    finally:
        deep_gemm.set_num_sms(original_sms)


if __name__ == "__main__":
    main()
