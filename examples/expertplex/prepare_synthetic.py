"""Copy model metadata into an explicitly synthetic, weight-free test directory."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry", choices=("tiny", "30b", "480b"), default="tiny")
    args = parser.parse_args()
    data = json.loads(args.source_config.read_text())
    if args.geometry == "tiny":
        data.update(
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=1024,
            max_position_embeddings=4096,
            max_window_layers=2,
            eos_token_id=1023,
        )
    elif args.geometry == "30b":
        data.update(
            hidden_size=2048,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=128,
            num_experts_per_tok=8,
            max_window_layers=48,
        )
    elif (data["hidden_size"], data["num_hidden_layers"], data["num_experts"]) != (
        6144,
        62,
        160,
    ):
        parser.error("480b requires the official Coder-480B config")
    data["expertplex_synthetic_geometry"] = args.geometry
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "SYNTHETIC.json").write_text(
        json.dumps(
            {
                "scope": "random weights; no model accuracy or pretrained throughput claim",
                "geometry": args.geometry,
                "source_config": str(args.source_config.resolve()),
                "weight_shards": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
