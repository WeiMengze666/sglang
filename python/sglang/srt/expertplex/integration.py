"""Opt-in SGLang hooks, with no CUDA or Torch imports at module import time."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

CONFIG_ENV = "SGLANG_EXPERTPLEX_CONFIG"
CLIENT_ENV = "SGLANG_EXPERTPLEX_CLIENT_ID"


def expertplex_enabled() -> bool:
    # A half-configured opt-in must fail, instead of silently allocating a
    # complete local model on a GPU assigned only to attention.
    return CONFIG_ENV in os.environ or CLIENT_ENV in os.environ


@lru_cache(maxsize=8)
def _load_attention_config(path: str, raw_client_id: str):
    from sglang.srt.expertplex.config import load_config

    try:
        client_id = int(raw_client_id)
    except ValueError as exc:
        raise ValueError(f"{CLIENT_ENV} must be a nonnegative integer") from exc
    if client_id < 0 or str(client_id) != raw_client_id:
        raise ValueError(f"{CLIENT_ENV} must be a nonnegative integer")
    config = load_config(path)
    return config, config.client(client_id)


def attention_config():
    path, client_id = os.environ.get(CONFIG_ENV), os.environ.get(CLIENT_ENV)
    if not path or client_id is None:
        raise ValueError(f"ExpertPlex requires both {CONFIG_ENV} and {CLIENT_ENV}")
    return _load_attention_config(str(Path(path).expanduser().resolve()), client_id)


def is_routed_expert_weight(name: str) -> bool:
    return ".mlp.experts." in name or name.startswith("mlp.experts.")


def bound_request_input_length(original_limit: int) -> int:
    """Make unchunked P admission safe without shortening decode context.

    SGLang's request validator rejects lengths >= the returned limit, while
    max_prefill_tokens alone permits an oversized first request. Both P and D
    advertise the same P mailbox input bound to the existing tokenizer/router.
    """
    if not expertplex_enabled():
        return original_limit
    config, _ = attention_config()
    return min(original_limit, config.max_prefill_tokens + 1)


def attention_weights_iterator(
    weight_files: Iterable[str],
) -> Iterator[tuple[str, Any]]:
    """Read only attention/router weights, before safetensors materialization.

    The normal multithreaded and eager loaders materialize every tensor before
    a model's ``load_weights`` can filter it. Attention processes instead use
    this mmap iterator, so routed experts are never read into CPU tensors.
    """
    from safetensors import safe_open

    for path in weight_files:
        if not str(path).endswith(".safetensors"):
            raise ValueError("ExpertPlex attention requires a safetensors checkpoint")
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if is_routed_expert_weight(name) or "rotary_emb.inv_freq" in name:
                    continue
                tensor = source.get_tensor(name)
                if str(tensor.dtype) != "torch.bfloat16":
                    raise ValueError(
                        f"ExpertPlex requires BF16 attention weights: {name}"
                    )
                yield name, tensor


def validate_server_args(cfg: Any) -> None:
    """Reject unsupported combinations before the engine spawns GPU processes."""
    if not expertplex_enabled():
        return
    config, client = attention_config()
    errors = []
    for name in (
        "tp_size",
        "pp_size",
        "ep_size",
        "dp_size",
        "attn_dp_size",
        "attn_cp_size",
        "moe_dp_size",
        "dcp_size",
        "dwdp_size",
        "nnodes",
    ):
        if getattr(cfg, name, 1) != 1:
            errors.append(f"--{name.replace('_', '-')} must be 1")
    for name in (
        "enable_single_batch_overlap",
        "enable_torch_compile",
        "enable_eplb",
        "enable_lora",
        "enable_pdmux",
        "enable_pd_role_switch",
        "enable_prefill_cp",
        "enable_dynamic_chunking",
        "allow_auto_truncate",
    ):
        if getattr(cfg, name, False):
            errors.append(f"--{name.replace('_', '-')} is not supported")
    if getattr(cfg, "lora_paths", None):
        errors.append("LoRA adapters are not supported")
    if bool(getattr(cfg, "enable_two_batch_overlap", False)) != (
        config.overlap == "tbo"
    ):
        errors.append("--enable-two-batch-overlap must match ExpertPlex config.overlap")
    if getattr(cfg, "speculative_algorithm", None) is not None:
        errors.append("speculative decoding is not supported")
    if getattr(cfg, "quantization", None) not in (None, "unquant"):
        errors.append("quantization is not supported; use the BF16 checkpoint")
    formats = (
        ("dummy",) if config.weight_mode == "synthetic" else ("auto", "safetensors")
    )
    if getattr(cfg, "load_format", None) not in formats:
        errors.append(
            f"--load-format must be one of {formats} for {config.weight_mode} weights"
        )
    if getattr(cfg, "model_impl", "auto") not in ("auto", "sglang"):
        errors.append("--model-impl must be auto or sglang")
    if getattr(cfg, "device", None) != "cuda":
        errors.append("--device must be cuda")
    if str(getattr(cfg, "dtype", None)) not in ("bfloat16", "bf16", "torch.bfloat16"):
        errors.append("--dtype must be bfloat16")
    if getattr(cfg, "moe_a2a_backend", "none") != "none":
        errors.append("--moe-a2a-backend must be none")
    if getattr(cfg, "ep_num_redundant_experts", 0):
        errors.append("redundant experts are not supported")
    if not getattr(cfg, "disable_overlap_schedule", False):
        errors.append(
            "--disable-overlap-schedule is required by current ExpertPlex mailbox ownership"
        )
    if getattr(cfg, "startup_weight_load_mode", "serial") != "serial":
        errors.append("--startup-weight-load-mode must be serial")
    if getattr(cfg, "cpu_offload_gb", 0):
        errors.append("CPU model offload is not supported")
    if getattr(cfg, "disaggregation_mode", None) != client.phase:
        errors.append(f"--disaggregation-mode must match client phase {client.phase!r}")
    if getattr(cfg, "disaggregation_transfer_backend", None) != "mooncake":
        errors.append("--disaggregation-transfer-backend must be mooncake")
    if getattr(cfg, "base_gpu_id", 0) != client.device:
        errors.append(f"--base-gpu-id must match client device {client.device}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [value.strip() for value in visible.split(",")]
        if (
            len(devices) != config.node_gpu_count
            or len(set(devices)) != len(devices)
            or any(not value or value == "-1" for value in devices)
        ):
            errors.append(
                "CUDA_VISIBLE_DEVICES must expose all eight GPUs in the same order for every process"
            )
    capacity = (
        config.max_prefill_tokens
        if client.phase == "prefill"
        else config.max_decode_tokens
    )
    running = getattr(cfg, "max_running_requests", None)
    if running is None or not 0 < running <= capacity:
        errors.append(f"--max-running-requests must be in [1, {capacity}]")
    # P may run without chunking: tp_worker advertises a hard input bound.
    # D warmup/extend still uses its own smaller mailbox chunk capacity.
    chunk = getattr(cfg, "chunked_prefill_size", None)
    unchunked_prefill = client.phase == "prefill" and chunk in (None, -1)
    if not unchunked_prefill and (chunk is None or not 0 < chunk <= capacity):
        errors.append(f"--chunked-prefill-size must be in [1, {capacity}]")
    prefill = getattr(cfg, "max_prefill_tokens", None)
    if prefill is None or not 0 < prefill <= capacity:
        errors.append(f"--max-prefill-tokens must be in [1, {capacity}]")
    graph_config = getattr(cfg, "cuda_graph_config", None)
    for phase in ("decode", "prefill"):
        phase_config = (
            graph_config.get(phase, {})
            if isinstance(graph_config, dict)
            else getattr(graph_config, phase, None)
        )
        backend = (
            phase_config.get("backend")
            if isinstance(phase_config, dict)
            else getattr(phase_config, "backend", None)
        )
        allowed = (
            ("disabled", "piecewise", "full") if phase == "decode" else ("disabled",)
        )
        if backend not in allowed:
            errors.append(f"--cuda-graph-backend-{phase} must be one of {allowed}")
    if (
        Path(cfg.model_path).expanduser().resolve()
        != Path(config.model_path).expanduser().resolve()
    ):
        errors.append(
            "--model-path must match model_path in the ExpertPlex node config"
        )
    if errors:
        raise ValueError(
            "Invalid ExpertPlex attention configuration:\n- " + "\n- ".join(errors)
        )


def validate_attention_model(hf_config: Any, quant_config: Any = None) -> None:
    """Validate checkpoint identity/geometry even for direct model-loader callers."""
    if not expertplex_enabled():
        return
    from sglang.srt.expertplex.config import ModelSpec

    if quant_config is not None:
        raise ValueError("ExpertPlex does not support quantized attention models")
    config, _ = attention_config()
    model_spec = ModelSpec.from_dict(hf_config.to_dict())
    configured_spec = ModelSpec.from_model_path(config.model_path)
    if model_spec != configured_spec:
        raise ValueError(
            "SGLang checkpoint dimensions must match the ExpertPlex node checkpoint"
        )
    config.validate(model_spec)
