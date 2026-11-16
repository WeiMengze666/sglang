"""Mask padded top-k rows in equal-sized attention-DP rank segments."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_topk_by_rank_counts_kernel(
    ids,
    weights,
    counts,
    out_ids,
    out_weights,
    ROWS: tl.constexpr,
    TOP_K: tl.constexpr,
    ROWS_PER_RANK: tl.constexpr,
    IDS_ROW_STRIDE: tl.constexpr,
    IDS_COL_STRIDE: tl.constexpr,
    WEIGHTS_ROW_STRIDE: tl.constexpr,
    WEIGHTS_COL_STRIDE: tl.constexpr,
    COUNTS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // TOP_K, index % TOP_K
    in_range = row < ROWS
    rank = row // ROWS_PER_RANK
    valid_rows = tl.load(counts + rank * COUNTS_STRIDE, mask=in_range, other=0)
    live = row % ROWS_PER_RANK < valid_rows
    selected = tl.load(
        ids + row * IDS_ROW_STRIDE + column * IDS_COL_STRIDE, mask=in_range, other=-1
    )
    weight = tl.load(
        weights + row * WEIGHTS_ROW_STRIDE + column * WEIGHTS_COL_STRIDE,
        mask=in_range,
        other=0.0,
    )
    tl.store(out_ids + index, tl.where(live, selected, -1), mask=in_range)
    tl.store(out_weights + index, tl.where(live, weight, 0.0), mask=in_range)


def mask_topk_by_rank_counts(ids, weights, counts):
    """Return masked copies of IDs and weights for equal-sized rank segments.

    Valid entries retain their exact values. Padding receives ID -1 and weight
    zero. Counts are read on device at execution time, including graph replay.
    Callers provide integer counts within each segment's row capacity.
    """
    if ids.ndim != 2 or ids.shape != weights.shape:
        raise ValueError("IDs and weights must have the same two-dimensional shape")
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise TypeError("Expected int32/int64 IDs and FP32 weights")
    if counts.ndim != 1 or counts.numel() == 0 or counts.dtype != torch.int32:
        raise ValueError("Rank counts must be a nonempty int32 vector")
    if (
        ids.device.type != "cuda"
        or weights.device != ids.device
        or counts.device != ids.device
    ):
        raise ValueError("IDs, weights and counts must share a CUDA device")
    rows, top_k = ids.shape
    if top_k == 0 or rows % counts.numel():
        raise ValueError("Rows must divide into rank segments with nonempty top-k")
    out_ids = torch.empty(ids.shape, dtype=ids.dtype, device=ids.device)
    out_weights = torch.empty(weights.shape, dtype=weights.dtype, device=weights.device)
    if rows:
        _mask_topk_by_rank_counts_kernel[(triton.cdiv(ids.numel(), 256),)](
            ids,
            weights,
            counts,
            out_ids,
            out_weights,
            ROWS=rows,
            TOP_K=top_k,
            ROWS_PER_RANK=rows // counts.numel(),
            IDS_ROW_STRIDE=ids.stride(0),
            IDS_COL_STRIDE=ids.stride(1),
            WEIGHTS_ROW_STRIDE=weights.stride(0),
            WEIGHTS_COL_STRIDE=weights.stride(1),
            COUNTS_STRIDE=counts.stride(0),
            BLOCK=256,
            num_warps=4,
        )
    return out_ids, out_weights
