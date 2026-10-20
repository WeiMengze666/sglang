"""Explicit, partition-independent Qwen3 BF16 weights for synthetic experiments.

A canonical HF tensor name determines its RNG seed. Generate the full tensor
before the regular weight loader slices TP/EP shards; E workers use the same
function for their globally named experts. These are random weights, never a
substitute for the pretrained checkpoint or its routing distribution.
"""

from __future__ import annotations

import hashlib
import math
import os

from .config import ModelSpec

ENVIRONMENT = "SGLANG_EXPERTPLEX_SYNTHETIC_SEED"
ALGORITHM = "qwen3-canonical-bf16-normal-fanin-v1"


def weight_identity(config) -> dict:
    """Startup identity: reject differently initialized A/E processes."""
    if config.weight_mode != "synthetic":
        return {"mode": "checkpoint"}
    raw_seed = os.environ.get(ENVIRONMENT)
    if raw_seed is not None and int(raw_seed) != config.synthetic_seed:
        raise ValueError("Canonical synthetic seed differs from node configuration")
    return {
        "mode": "synthetic",
        "algorithm": ALGORITHM if raw_seed is not None else "legacy-shard-dependent",
        "seed": config.synthetic_seed,
    }


def tensor_seed(name: str, seed: int) -> int:
    if seed < 0:
        raise ValueError("Synthetic seed must be nonnegative")
    digest = hashlib.sha256(f"{ALGORITHM}:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def fill_tensor(tensor, name: str, seed: int):
    import torch

    if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        raise ValueError("Canonical synthetic weights require contiguous BF16")
    if tensor.ndim == 1:
        tensor.fill_(1.0)
    else:
        generator = torch.Generator(device=tensor.device).manual_seed(
            tensor_seed(name, seed)
        )
        standard_deviation = (
            0.02
            if name == "model.embed_tokens.weight"
            else 1 / math.sqrt(tensor.shape[-1])
        )
        tensor.normal_(mean=0, std=standard_deviation, generator=generator)
    return tensor


def canonical_shapes(spec: ModelSpec, *, include_experts: bool = True):
    h, inner = spec.hidden_size, spec.intermediate_size
    yield "model.embed_tokens.weight", (spec.vocab_size, h)
    for layer in range(spec.num_layers):
        prefix = f"model.layers.{layer}"
        yield f"{prefix}.input_layernorm.weight", (h,)
        yield f"{prefix}.post_attention_layernorm.weight", (h,)
        for projection, heads in (
            ("q_proj", spec.num_attention_heads),
            ("k_proj", spec.num_key_value_heads),
            ("v_proj", spec.num_key_value_heads),
        ):
            yield f"{prefix}.self_attn.{projection}.weight", (heads * spec.head_dim, h)
        yield (
            f"{prefix}.self_attn.o_proj.weight",
            (h, spec.num_attention_heads * spec.head_dim),
        )
        for norm in ("q_norm", "k_norm"):
            yield f"{prefix}.self_attn.{norm}.weight", (spec.head_dim,)
        yield f"{prefix}.mlp.gate.weight", (spec.num_experts, h)
        if include_experts:
            for expert in range(spec.num_experts):
                for projection, shape in (
                    ("gate_proj", (inner, h)),
                    ("up_proj", (inner, h)),
                    ("down_proj", (h, inner)),
                ):
                    yield f"{prefix}.mlp.experts.{expert}.{projection}.weight", shape
    yield "model.norm.weight", (h,)
    yield "lm_head.weight", (spec.vocab_size, h)


def initialize_model(model) -> None:
    """Called only by DummyModelLoader under the explicit experiment opt-in."""
    import torch

    from .integration import attention_config, expertplex_enabled

    seed = int(os.environ[ENVIRONMENT])
    if expertplex_enabled():
        weight_identity(attention_config()[0])
    spec = ModelSpec.from_dict(model.config.to_dict())
    parameters = tuple(model.parameters())
    if not parameters or any(p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError(
            "Canonical synthetic initialization supports unquantized BF16 only"
        )
    device = parameters[0].device

    def weights():
        for name, shape in canonical_shapes(
            spec, include_experts=not expertplex_enabled()
        ):
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
            yield name, fill_tensor(tensor, name, seed)

    model.load_weights(weights())
