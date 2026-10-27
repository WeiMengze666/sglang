"""APK grouped-GEMM sharing using the exact bench_sharing.py input/weight data.

Single-operation mode is explicit and diagnostic only; serving always executes
the full GateUp/activation/Down sequence. Inputs are pre-staged before timing,
and a separate attention GPU publishes P/D ready signals 10 us apart. Timings
are publication-to-DONE acknowledgement upper bounds on that GPU's clock.
"""

import argparse
import json
import statistics
from pathlib import Path

from _bootstrap import module
from bench_sharing import WORKLOAD_IDENTITY, weights, workload


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
    p.add_argument("--operation", choices=("gateup", "down", "moe"), default="gateup")
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 48, 64, 96])
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
        help="Controlled active D experts",
    )
    p.add_argument(
        "--decode-expert-rows",
        type=int,
        default=1,
        help="Rows per controlled active D expert (default: one)",
    )
    p.add_argument(
        "--ctas",
        type=int,
        default=0,
        help="Offline-profile SM/CTA limit; zero uses the device",
    )
    p.add_argument(
        "--idle-mailboxes",
        type=int,
        default=0,
        help="Additional inactive mailboxes for scheduler overhead diagnostics",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.decode_expert_rows <= args.decode_rows or (
        args.decode_expert_rows != 1 and args.decode_active_experts is None
    ):
        p.error(
            "Decode expert rows must fit capacity and require controlled active experts"
        )
    if (
        args.prefill_expert_rows is not None
        and not 1 <= args.prefill_expert_rows <= args.prefill_rows
    ):
        p.error("Prefill expert rows exceed the mailbox capacity")
    if (
        args.decode_active_experts is not None
        and not 1 <= args.decode_active_experts <= args.experts
    ):
        p.error("Invalid decode active expert count")
    if not 0 <= args.idle_mailboxes <= 30:
        p.error("Use between zero and thirty inactive mailboxes")
    if args.device == args.control_device or min(args.iterations, *args.budgets) <= 0:
        p.error("Use distinct devices and positive counts")
    import torch

    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    native = module("native")
    extension = native.load_extension()
    w13, w2 = weights(args)
    works = [
        workload(args, rows, phase, w13, w2, return_buffers=True)
        for phase, rows in enumerate([args.prefill_rows, args.decode_rows])
    ]
    for invoke, *_ in works:
        invoke()
    torch.cuda.synchronize(args.device)
    report = {
        "scope": __doc__,
        "workload_identity": WORKLOAD_IDENTITY,
        "settings": vars(args) | {"output": str(args.output)},
        "native_source_sha256": native.source_fingerprint(),
        "expert_rows": [work[2] for work in works],
        "cases": [],
    }
    for budget in args.budgets:
        torch.cuda.set_device(args.device)
        buffers = native.OwnedBuffers(
            2 + args.idle_mailboxes,
            args.experts,
            max(args.prefill_rows, args.decode_rows),
            args.hidden,
            args.intermediate,
            args.device,
            capacities=[args.prefill_rows, args.decode_rows]
            + [1] * args.idle_mailboxes,
            packed=[1, 0] + [0] * args.idle_mailboxes,
            top_k=args.top_k,
        )
        views, valid_rows = [], []
        for phase, (_, reference, counts, inputs) in enumerate(works):
            view = dict(extension.benchmark_views(buffers._impl, phase))
            views.append(view)
            view["counts"].copy_(torch.tensor(counts, dtype=torch.int32, device="cuda"))
            cursor, offsets, selected_rows = 0, [], []
            capacity = ((args.decode_rows + 127) // 128) * 128
            source_stride = inputs["input"].shape[1] if phase else None
            for expert, count in enumerate(counts):
                offsets.append(cursor)
                src = (
                    sum((c + 127) // 128 * 128 for c in counts[:expert])
                    if phase == 0
                    else expert * source_stride
                )
                dest = cursor if phase == 0 else expert * capacity
                for name in ("input", "activation"):
                    if name == "activation" and args.operation != "down":
                        continue
                    source = inputs[name].reshape(-1, inputs[name].shape[-1])
                    view[name][dest : dest + count].copy_(source[src : src + count])
                selected_rows.extend((dest + i, src + i) for i in range(count))
                cursor += (count + 127) // 128 * 128
            view["offsets"].copy_(
                torch.tensor(offsets, dtype=torch.int32, device="cuda")
            )
            valid_rows.append(selected_rows)
        torch.cuda.synchronize(args.device)
        expected_tiles = sum((count + 127) // 128 for count in works[1][2])
        executor = native.PersistentExecutor(
            buffers,
            w13.unsqueeze(0),
            w2.unsqueeze(0),
            decode_sm_budget=budget,
            expected_decode_tiles=max(1, expected_tiles),
            max_decode_sm_fraction=0.875,
            benchmark_operation=args.operation,
            ctas=args.ctas,
        )
        row = {
            "decode_budget": budget,
            "expected_decode_tiles": expected_tiles,
            "executor": executor.describe(),
        }

        def measure(enabled=(True, True), owner=buffers._impl):
            stamps = extension.benchmark_publish(
                owner, args.control_device, *enabled, args.delay_us
            )
            values = stamps.cpu().tolist()
            if any(values[4:]) or any(
                enabled[i] and values[i + 2] <= values[i] for i in range(2)
            ):
                raise RuntimeError(f"APK publication/ack error: {values}")
            return {
                "release_gap_us": (values[1] - values[0]) / 1000,
                "response_upper_bound_us": [
                    (values[i + 2] - values[i]) / 1000 if enabled[i] else None
                    for i in range(2)
                ],
            }

        try:
            for _ in range(3):
                measure()
            isolated = [
                [
                    measure(tuple(i == phase for i in range(2)))
                    for _ in range(args.iterations)
                ]
                for phase in range(2)
            ]
            samples = [measure() for _ in range(args.iterations)]
            row.update(
                samples=samples,
                isolated=isolated,
                response_upper_bound_us=[
                    statistics.median(s["response_upper_bound_us"][i] for s in samples)
                    for i in range(2)
                ],
            )
        finally:
            executor.stop()
        numerics = []
        for phase, view in enumerate(views):
            output = view["gateup" if args.operation == "gateup" else "output"]
            indices = torch.tensor(valid_rows[phase], device=args.device)
            reference_output = works[phase][1].reshape(-1, output.shape[-1])
            passed = True
            delta_squared = reference_squared = exact = maximum = 0.0
            elements = 0
            # Cover every valid element while bounding diagnostic scratch.
            # Fig.13's largest P size must not allocate three full FP32 copies.
            for begin in range(0, len(indices), 4096):
                part = indices[begin : begin + 4096]
                actual = output[part[:, 0]].float()
                reference = reference_output[part[:, 1]].float()
                delta = actual - reference
                passed &= bool(torch.allclose(actual, reference, atol=0.01, rtol=0.03))
                delta_squared += float(delta.square().sum(dtype=torch.float64))
                reference_squared += float(reference.square().sum(dtype=torch.float64))
                exact += float((actual == reference).sum(dtype=torch.float64))
                maximum = max(maximum, float(delta.abs().max()))
                elements += actual.numel()
            numerics.append(
                {
                    "passed": bool(passed),
                    "relative_l2": (delta_squared / max(reference_squared, 1e-60))
                    ** 0.5,
                    "max_abs_difference": maximum,
                    "exact_fraction": exact / elements,
                }
            )
        row["native_reference"] = numerics
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(budget, row["response_upper_bound_us"], numerics, flush=True)
        if not all(result["passed"] for result in numerics):
            raise AssertionError(
                "Standalone APK operation differs from native reference"
            )
        del measure, executor, views, buffers


if __name__ == "__main__":
    main()
