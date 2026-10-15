"""Check APK BF16 activation against native SGLang contiguous/masked kernels.

This uses the production helper with production nvcc flags. It is an isolated
arithmetic contract, not model accuracy or a serving performance benchmark.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.utils.cpp_extension import load

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_fwd
    from sglang.srt.expertplex.native import source_fingerprint

    torch.cuda.set_device(args.device)
    root = Path(__file__).resolve().parents[2]
    extension = load(
        name="expertplex_activation_contract",
        sources=[str(root / "test/registered/expertplex/activation_contract.cu")],
        extra_include_paths=[str(root / "python/sglang/srt/expertplex/csrc")],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        verbose=False,
    )
    report = dict(scope=__doc__, source_sha256=source_fingerprint(), cases=[])
    generator = torch.Generator(device="cuda").manual_seed(1234)

    def check(label, x):
        shape = (*x.shape[:-1], x.shape[-1] // 2)
        expected = torch.empty(shape, dtype=x.dtype, device=x.device)
        candidate = torch.empty_like(expected)
        rows = torch.full((shape[0],), shape[1], dtype=torch.int32, device=x.device)
        for phase in (0, 1):
            if phase == 0:
                silu_and_mul(x.flatten(0, 1), expected.flatten(0, 1))
            else:
                silu_and_mul_masked_fwd(x, expected, rows)
            extension.run(x, candidate, phase)
            different = candidate.view(torch.int16) != expected.view(torch.int16)
            count = int(different.sum())
            report["cases"].append(
                dict(
                    label=label,
                    phase=phase,
                    elements=candidate.numel(),
                    different_bits=count,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if count:
                raise AssertionError(report["cases"][-1])

    for scale in (0.01, 0.25, 1, 4, 16, 64):
        check(
            f"random-{scale}",
            (
                torch.randn((8, 256, 512), generator=generator, device="cuda") * scale
            ).bfloat16(),
        )
    # Every finite BF16 gate value, including signed zeros and subnormals.
    # Deterministic permutations exercise both signs and several up magnitudes.
    values = (
        torch.arange(65536, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    values = values[torch.isfinite(values)]
    gate = values.repeat(2).reshape(2, 255, 256)
    for shift in (1, 127, 4096):
        up = values.roll(shift).repeat(2).reshape_as(gate)
        check(f"finite-bf16-shift-{shift}", torch.cat((gate, up), dim=-1))
    print(json.dumps(dict(cases=len(report["cases"]), all_bitwise_equal=True)))


if __name__ == "__main__":
    main()
