"""Explicit grouped-D adapter for native DP row layouts; no scheduler or C++ path."""
from __future__ import annotations


def grouped_attention_dp_size() -> int:
    from sglang.srt.runtime_context import get_disagg, get_parallel
    if get_parallel().attn_dp_size == 1 or not get_disagg().enable_expertplex:
        return 1
    from .integration import attention_config
    _, client = attention_config()
    size = client.attention_dp_size
    if get_parallel().attn_dp_size != size:
        raise ValueError("runtime attention DP differs from explicit client topology")
    return size


def real_dp_counts(batch, num_dp_ranks: int, max_slot_tokens=None):
    counts = getattr(batch, "original_global_num_tokens_cpu", None)
    if not isinstance(counts, (list, tuple)) or len(counts) != num_dp_ranks:
        raise ValueError("raw per-DP token counts are required, not a global sum")
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("raw per-DP token counts must be nonnegative integers")
    if max_slot_tokens is not None and any(n > max_slot_tokens for n in counts):
        raise ValueError("raw DP count exceeds the selected per-rank graph bucket")
    return tuple(counts)


def copy_real_dp_counts_(buffer, batch, max_slot_tokens=None):
    import torch
    if buffer.ndim != 1 or buffer.dtype != torch.int32 or buffer.numel() < 2:
        raise ValueError("grouped-D counts need a stable int32 DP vector")
    counts = real_dp_counts(batch, buffer.numel(), max_slot_tokens)
    host = torch.tensor(counts, dtype=torch.int32, pin_memory=buffer.device.type == "cuda")
    buffer.copy_(host, non_blocking=buffer.device.type == "cuda")
    return buffer


def make_eager_real_dp_counts(batch, num_dp_ranks: int):
    import torch
    real_dp_counts(batch, num_dp_ranks)
    buffer = torch.empty((num_dp_ranks,), dtype=torch.int32, device=batch.input_ids.device)
    return copy_real_dp_counts_(buffer, batch)


def mask_topk_padding(topk, batch, num_dp_ranks: int):
    """Keep valid routes bit-identical; holes use the existing -1 route sentinel."""
    if num_dp_ranks == 1:
        return topk
    import torch
    ids, weights = topk.topk_ids, topk.topk_weights
    if ids.ndim != 2 or ids.shape != weights.shape or ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise ValueError("standard integer IDs and FP32 route weights are required")
    raw = getattr(batch, "expertplex_dp_real_num_tokens_gpu", None)
    if raw is None or raw.shape != (num_dp_ranks,) or raw.dtype != torch.int32 or raw.device != ids.device:
        raise ValueError("missing graph-safe raw DP count vector")
    counts = real_dp_counts(batch, num_dp_ranks)
    mode = getattr(batch, "dp_padding_mode", None)
    from sglang.srt.layers.dp_attention import DpPaddingMode
    if not isinstance(mode, DpPaddingMode):
        raise ValueError("explicit native DP padding mode is required")
    rows = ids.shape[0]
    if mode.is_sum_len():
        if sum(counts) != rows:
            raise ValueError("SUM_LEN must pack exact real token counts with attention TP1")
        return topk
    if not mode.is_max_len():
        raise ValueError("unsupported native DP padding mode")
    if rows == 0:
        if any(counts):
            raise ValueError("nonempty real DP counts have no gathered rows")
        return topk
    if rows % num_dp_ranks:
        raise ValueError("MAX_LEN rows must have equal per-DP slots")
    stride = rows // num_dp_ranks
    if any(n > stride for n in counts):
        raise ValueError("raw DP count exceeds gathered slot length")
    index = torch.arange(rows, device=ids.device)
    valid = (index.remainder(stride) < raw[index.div(stride, rounding_mode="floor")])[:, None]
    return topk._replace(
        topk_ids=torch.where(valid, ids, ids.new_full((), -1)),
        topk_weights=torch.where(valid, weights, weights.new_zeros(())),
    )
