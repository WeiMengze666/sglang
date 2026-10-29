"""Generate a single-node configuration; runs without PyTorch or GPUs."""

import argparse
import json
from pathlib import Path

from _bootstrap import module

config = module("config")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefill", type=int, default=1)
    parser.add_argument("--decode", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--prefill-tp-size", type=int, default=1)
    parser.add_argument("--decode-tp-size", type=int, default=1)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument(
        "--prefill-mailbox-tokens",
        type=int,
        help="Opt-in TBO mailbox capacity; requires equal-token splitting",
    )
    parser.add_argument("--max-decode-tokens", type=int, default=256)
    parser.add_argument("--decode-sm-budget", type=int, default=32)
    parser.add_argument("--expected-decode-tiles", type=int, default=128)
    parser.add_argument("--max-decode-sm-fraction", type=float, default=0.75)
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--overlap", choices=("none", "tbo"), default="none")
    parser.add_argument(
        "--kernel",
        choices=("sm100", "sm100_cta1", "sm100_fused", "wmma"),
        default="sm100",
    )
    parser.add_argument(
        "--prefill-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run normal-mode scatter and gather inside the persistent expert kernel",
    )
    parser.add_argument(
        "--synthetic-weights",
        action="store_true",
        help="Explicit random weights for geometry/capacity testing only",
    )
    args = parser.parse_args()
    node = config.default_config(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        prefill_count=args.prefill,
        decode_count=args.decode,
        expert_count=args.experts,
        prefill_tp_size=args.prefill_tp_size,
        decode_tp_size=args.decode_tp_size,
        max_prefill_tokens=args.max_prefill_tokens,
        prefill_mailbox_tokens=args.prefill_mailbox_tokens,
        max_decode_tokens=args.max_decode_tokens,
        decode_sm_budget=args.decode_sm_budget,
        expected_decode_tiles=args.expected_decode_tiles,
        max_decode_sm_fraction=args.max_decode_sm_fraction,
        rendezvous_dir=str(Path(args.rendezvous_dir).expanduser().resolve()),
        weight_mode="synthetic" if args.synthetic_weights else "checkpoint",
        overlap=args.overlap,
        kernel=args.kernel,
        prefill_preprocess=args.prefill_preprocess,
    )
    if Path(node.model_path, "config.json").exists():
        spec = config.ModelSpec.from_model_path(node.model_path)
        node.validate(spec)
        capacity = max(node.max_prefill_tokens, node.max_decode_tokens)
        experts = spec.num_experts // len(node.expert_devices)
        capacities, packed = node.buffer_layout()
        buffers = module("native").estimate_buffer_bytes(
            node.slot_count,
            experts,
            capacity,
            spec.hidden_size,
            spec.intermediate_size,
            capacities=capacities,
            packed=packed,
            top_k=spec.top_k,
            prefill_preprocess=node.prefill_preprocess,
        )["total"]
        memory = {
            "expert_weights_gib_per_gpu": spec.expert_weight_bytes
            / args.experts
            / 2**30,
            "expert_mailboxes_gib_per_gpu": buffers / 2**30,
            "attention_weights_gib_per_gpu": spec.attention_weight_bytes / 2**30,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
            "note": "Weight/mailbox estimate; attention weights and KV bytes are conservative TP1 upper bounds per GPU. KV pool, CUDA context and scratch are additional.",
        }
        if (
            memory["expert_weights_gib_per_gpu"]
            + memory["expert_mailboxes_gib_per_gpu"]
            + 8
            > node.gpu_memory_gib
        ):
            parser.error(
                "Expert weights and mailboxes exceed the configured GPU memory budget"
            )
        print(json.dumps(memory, indent=2))
    Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    node.write_json(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
