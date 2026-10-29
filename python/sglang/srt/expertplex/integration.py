"""Opt-in SGLang hooks, with no CUDA or Torch imports at module import time."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


def runtime_config():
    from sglang.srt.runtime_context import get_disagg

    return get_disagg()


def expertplex_enabled(cfg=None) -> bool:
    if cfg is None:
        cfg = runtime_config()
    return bool(getattr(cfg, "enable_expertplex", False))


@lru_cache(maxsize=8)
def _load_attention_config(path: str, client_id: int):
    from sglang.srt.expertplex.config import load_config

    if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
        raise ValueError("--expertplex-client-id must be a nonnegative integer")
    config = load_config(path)
    return config, config.client(client_id)


def attention_config(cfg=None):
    cfg = runtime_config() if cfg is None else cfg
    address = getattr(cfg, "expertplex_bootstrap_addr", None)
    if address:
        node, _ = remote_config(
            address, getattr(cfg, "expertplex_startup_timeout", 600.0)
        )
        device = getattr(cfg, "base_gpu_id", None)
        if device is None:
            from sglang.srt.runtime_context import get_device

            device = get_device().base_gpu_id
        matches = [
            c
            for c in node.clients
            if c.device == device and c.phase == cfg.disaggregation_mode
        ]
        if len(matches) != 1:
            raise ValueError(
                "No MoE attention slot matches --base-gpu-id and --disaggregation-mode"
            )
        return node, matches[0]
    path = getattr(cfg, "expertplex_config", None)
    client_id = getattr(cfg, "expertplex_client_id", None)
    if not path or client_id is None:
        raise ValueError(
            "ExpertPlex attention requires both --expertplex-config and --expertplex-client-id"
        )
    return _load_attention_config(str(Path(path).expanduser().resolve()), client_id)


@lru_cache(maxsize=8)
def remote_config(address, timeout):
    from .bootstrap import fetch_config

    return fetch_config(address, timeout)


def expert_node_config(cfg):
    from dataclasses import replace

    from .config import default_config, load_config

    policy = {
        name: getattr(cfg, "expertplex_" + name)
        for name in (
            "decode_sm_budget",
            "expected_decode_tiles",
            "max_decode_sm_fraction",
        )
        if getattr(cfg, "expertplex_" + name, None) is not None
    }
    if cfg.expertplex_config:
        if any(
            getattr(cfg, "expertplex_" + phase + "_tp_size", None) is not None
            for phase in ("prefill", "decode")
        ):
            raise ValueError(
                "Set client tp_size inside the topology JSON when using --expertplex-config"
            )
        node = load_config(cfg.expertplex_config)
        return replace(node, **policy)
    return default_config(
        str(Path(cfg.model_path).expanduser().resolve()),
        prefill_count=getattr(cfg, "expertplex_prefill_count", 1),
        decode_count=getattr(cfg, "expertplex_decode_count", 3),
        expert_count=getattr(cfg, "expertplex_expert_count", 4),
        prefill_tp_size=(
            1
            if getattr(cfg, "expertplex_prefill_tp_size", None) is None
            else cfg.expertplex_prefill_tp_size
        ),
        decode_tp_size=(
            1
            if getattr(cfg, "expertplex_decode_tp_size", None) is None
            else cfg.expertplex_decode_tp_size
        ),
        max_prefill_tokens=cfg.max_prefill_tokens or 24576,
        max_decode_tokens=cfg.max_running_requests or 256,
        startup_timeout_s=getattr(cfg, "expertplex_startup_timeout", 600.0),
        weight_mode="synthetic" if cfg.load_format == "dummy" else "checkpoint",
        synthetic_seed=int(os.environ.get("SGLANG_EXPERTPLEX_SYNTHETIC_SEED", "1234")),
        overlap="tbo" if cfg.enable_two_batch_overlap else "none",
        prefill_preprocess=True,
        **(
            {
                "decode_sm_budget": 96,
                "expected_decode_tiles": 8,
                "max_decode_sm_fraction": 0.875,
            }
            | policy
        ),
    )


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
    policy_options = (
        "expertplex_decode_sm_budget",
        "expertplex_expected_decode_tiles",
        "expertplex_max_decode_sm_fraction",
    )
    if any(
        name in os.environ
        for name in ("SGLANG_EXPERTPLEX_CONFIG", "SGLANG_EXPERTPLEX_CLIENT_ID")
    ):
        raise ValueError(
            "Legacy ExpertPlex environment activation is no longer supported; "
            "unset SGLANG_EXPERTPLEX_CONFIG/SGLANG_EXPERTPLEX_CLIENT_ID and use "
            "--enable-expertplex --expertplex-config --expertplex-client-id"
        )
    topology_options = ("expertplex_prefill_tp_size", "expertplex_decode_tp_size")
    if not expertplex_enabled(cfg):
        if (
            getattr(cfg, "expertplex_config", None) is not None
            or getattr(cfg, "expertplex_bootstrap_addr", None) is not None
            or getattr(cfg, "expertplex_client_id", None) is not None
            or getattr(cfg, "expertplex_role", "attention") != "attention"
            or any(
                getattr(cfg, name, None) is not None
                for name in policy_options + topology_options
            )
        ):
            raise ValueError("ExpertPlex options require --enable-expertplex")
        return
    role = getattr(cfg, "expertplex_role", "attention")
    if role not in ("attention", "expert"):
        raise ValueError("--expertplex-role must be attention or expert")
    if role != "expert" and any(
        getattr(cfg, name, None) is not None for name in topology_options
    ):
        raise ValueError(
            "Attention TP pool sizes belong to the expert role; use --tp-size on attention servers"
        )
    if role != "expert" and any(
        getattr(cfg, name, None) is not None for name in policy_options
    ):
        raise ValueError(
            "ExpertPlex SM policy options belong to --expertplex-role expert; "
            "attention servers inherit the MoE pool configuration"
        )
    if role == "expert":
        from .config import ModelSpec

        if cfg.expertplex_client_id is not None or getattr(
            cfg, "expertplex_bootstrap_addr", None
        ):
            raise ValueError(
                "Expert role cannot connect to another bootstrap or use --expertplex-client-id"
            )
        config = expert_node_config(cfg)
        config.validate(ModelSpec.from_model_path(config.model_path))
        errors = []
        if getattr(cfg, "disaggregation_mode", "null") != "null":
            errors.append("--disaggregation-mode belongs to attention servers")
        for name in (
            "tp_size",
            "pp_size",
            "ep_size",
            "dp_size",
            "attn_dp_size",
            "nnodes",
        ):
            if getattr(cfg, name, 1) != 1:
                errors.append(
                    f"--{name.replace('_', '-')} does not size the MoE pool; "
                    "use --expertplex-expert-count or --expertplex-config"
                )
        if str(getattr(cfg, "dtype", "auto")) not in (
            "auto",
            "bfloat16",
            "bf16",
            "torch.bfloat16",
        ):
            errors.append("the MoE pool supports BF16 weights only")
        if getattr(cfg, "device", "cuda") != "cuda":
            errors.append("--device must be cuda")
        if getattr(cfg, "quantization", None) not in (None, "unquant"):
            errors.append("quantized expert weights are not supported")
        for name in ("enable_lora", "enable_eplb", "enable_pdmux"):
            if getattr(cfg, name, False):
                errors.append(
                    f"--{name.replace('_', '-')} is not supported by the MoE pool"
                )
        if getattr(cfg, "lora_paths", None) or getattr(
            cfg, "speculative_algorithm", None
        ):
            errors.append(
                "LoRA and speculative decoding are not supported by the MoE pool"
            )
        if errors:
            raise ValueError(
                "Invalid ExpertPlex expert arguments: " + "; ".join(errors)
            )
        if Path(cfg.model_path).resolve() != Path(config.model_path).resolve():
            raise ValueError(
                "--model-path must match model_path in the ExpertPlex node config"
            )
        return
    if getattr(cfg, "expertplex_bootstrap_addr", None) and (
        getattr(cfg, "expertplex_config", None)
        or getattr(cfg, "expertplex_client_id", None) is not None
    ):
        raise ValueError(
            "Use either bootstrap discovery or explicit config/client ID, not both"
        )
    config, client = attention_config(cfg)
    errors = []
    if getattr(cfg, "tp_size", 1) != client.tp_size:
        errors.append(
            f"--tp-size must match attention client TP width {client.tp_size}"
        )
    if getattr(cfg, "gpu_id_step", 1) != 1:
        errors.append("--gpu-id-step must be 1 for contiguous attention TP groups")
    if client.tp_size > 1 and getattr(cfg, "enable_layernorm_sp", False):
        errors.append(
            "sequence-parallel FFN rows require an explicit ExpertPlex gather contract"
        )
    for name in (
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
    if config.prefill_mailbox_tokens is not None and (
        getattr(cfg, "tbo_token_distribution_threshold", None) != 0.5
        or not isinstance(getattr(cfg, "max_running_requests", None), int)
        or cfg.max_running_requests > config.max_decode_tokens
    ):
        errors.append(
            "compact P mailboxes require --tbo-token-distribution-threshold 0.5 "
            "and max running requests <= configured max_decode_tokens"
        )
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
            ("disabled", "breakable", "full")
            if phase == "decode"
            else ("disabled", "breakable")
        )
        if backend not in allowed:
            errors.append(f"--cuda-graph-backend-{phase} must be one of {allowed}")
        if phase == "prefill" and backend == "breakable" and config.overlap == "tbo":
            errors.append(
                "ExpertPlex prefill breakable graph with TBO is not yet validated; "
                "use --cuda-graph-backend-prefill disabled with TBO"
            )
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
