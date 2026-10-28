"""Create a deterministic tiny BF16 checkpoint shared by both correctness paths."""

import argparse
import json
from pathlib import Path

from _bootstrap import module

normalize_model_config = module("config").normalize_model_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from safetensors.torch import save_file

    data = normalize_model_config(
        json.loads((args.metadata / "config.json").read_text())
    )
    if data.get("expertplex_synthetic_geometry") not in ("tiny", "tiny-deepep"):
        p.error("Only the tiny test geometry may be materialized by this script")
    args.output.mkdir(parents=True, exist_ok=False)
    h = data["hidden_size"]
    i = data["moe_intermediate_size"]
    d = data["head_dim"]
    generator = torch.Generator().manual_seed(20261015)
    weights = {}

    def random(name, shape):
        weights[name] = (torch.randn(shape, generator=generator) * 0.02).bfloat16()

    random("model.embed_tokens.weight", (data["vocab_size"], h))
    random("lm_head.weight", (data["vocab_size"], h))
    weights["model.norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    for layer in range(data["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = torch.ones(h, dtype=torch.bfloat16)
        for name in ("q_norm", "k_norm"):
            weights[prefix + "self_attn." + name + ".weight"] = torch.ones(
                d, dtype=torch.bfloat16
            )
        for name, heads in (
            ("q_proj", data["num_attention_heads"]),
            ("k_proj", data["num_key_value_heads"]),
            ("v_proj", data["num_key_value_heads"]),
        ):
            random(prefix + "self_attn." + name + ".weight", (heads * d, h))
        random(prefix + "self_attn.o_proj.weight", (h, data["num_attention_heads"] * d))
        random(prefix + "mlp.gate.weight", (data["num_experts"], h))
        for expert in range(data["num_experts"]):
            for projection, shape in (
                ("gate_proj", (i, h)),
                ("up_proj", (i, h)),
                ("down_proj", (h, i)),
            ):
                random(prefix + f"mlp.experts.{expert}.{projection}.weight", shape)
    save_file(weights, str(args.output / "model.safetensors"))
    (args.output / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "TEST_CHECKPOINT.json").write_text(
        json.dumps(
            {
                "scope": "tiny random BF16 correctness fixture; not pretrained accuracy",
                "seed": 20261015,
                "geometry": data["expertplex_synthetic_geometry"],
                "tensors": len(weights),
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
