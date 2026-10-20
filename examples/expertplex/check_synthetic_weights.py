"""Check canonical synthetic expert values across devices and fused views."""

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch

    synthetic = module("synthetic_weights")
    rows = []
    for layer, expert in ((0, 0), (0, 159), (61, 159)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            shape = (6144, 2560) if projection == "down_proj" else (2560, 6144)
            source = torch.empty(shape, dtype=torch.bfloat16, device=0)
            if projection == "down_proj":
                target = torch.empty(shape, dtype=torch.bfloat16, device=4)
            else:
                fused = torch.empty((5120, 6144), dtype=torch.bfloat16, device=4)
                target = fused[:2560] if projection == "gate_proj" else fused[2560:]
            synthetic.fill_tensor(source, name, 1234)
            synthetic.fill_tensor(target, name, 1234)
            equal = torch.equal(source.to(4), target)
            sample = (
                source.flatten()[:: max(1, source.numel() // 4096)]
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            rows.append(
                dict(
                    name=name,
                    shape=shape,
                    exact_equal=bool(equal),
                    sample_sha256=hashlib.sha256(sample).hexdigest(),
                )
            )
            if not equal:
                raise AssertionError(rows[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(algorithm=synthetic.ALGORITHM, seed=1234, status="passed", cases=rows),
            indent=2,
        )
        + "\n"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
