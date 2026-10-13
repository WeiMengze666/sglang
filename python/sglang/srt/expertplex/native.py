"""Lazy CUDA bindings for ExpertPlex's experimental single-node executor.

Importing this module does not import torch or require a CUDA installation. The
extension is built on first use, on the deployment host. CUDA 12.9 or newer and
a PyTorch build which supports Blackwell are required for the B300 target.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any


def estimate_buffer_bytes(
    slot_count: int,
    local_experts: int,
    max_rows: int,
    hidden: int,
    intermediate: int,
) -> dict[str, int]:
    """Exact requested cudaMalloc bytes, excluding allocator/driver overhead."""
    dims = (slot_count, local_experts, max_rows, hidden, intermediate)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims):
        raise ValueError("ExpertPlex buffer dimensions must be positive integers")
    rows = slot_count * local_experts * max_rows
    result = {
        "input": rows * hidden * 2,
        "activation": rows * intermediate * 2,
        "output": rows * hidden * 2,
        "counts": slot_count * local_experts * 4,
        "control": slot_count * 8 * 8,
        "scheduler": slot_count * 32,
        "stop": 4,
    }
    result["total"] = sum(result.values())
    return result


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("ExpertPlex native executor requires a CUDA deployment host")
    # Supplying an explicit architecture also prevents cpp_extension from
    # compiling for every visible device on an eight-GPU deployment host.
    return load(
        name="sglang_expertplex_native_v1",
        sources=[str(Path(__file__).parent / "csrc" / "expertplex.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_103,code=sm_103",
            "-gencode=arch=compute_103,code=compute_103",
        ],
        with_cuda=True,
        verbose=False,
    )


class OwnedBuffers:
    """Dedicated cudaMalloc allocations whose IPC handles stay valid until close.

    R is a per-expert row capacity, not a total token capacity. Storage is
    BF16 input[S,E,R,H], activation[S,E,R,I], output[S,E,R,H]. A slot admits
    exactly one outstanding MoE layer invocation.
    """

    def __init__(
        self,
        slot_count: int,
        local_experts: int,
        max_rows: int,
        hidden: int,
        intermediate: int,
        device: int,
    ) -> None:
        self._impl = load_extension().OwnedBuffers(
            slot_count, local_experts, max_rows, hidden, intermediate, device
        )

    def descriptor(self) -> dict[str, Any]:
        return dict(self._impl.descriptor())


class RemoteBuffers:
    """IPC mappings opened by an attention process for its exclusive slot."""

    def __init__(self, descriptor: dict[str, Any], slot_id: int, device: int) -> None:
        self._impl = load_extension().RemoteBuffers(descriptor, slot_id, device)

    def close(self) -> None:
        self._impl.close()


class PersistentExecutor:
    """One independently scheduled CTA per selected SM, with cluster size one.

    The WMMA kernel is a correctness-oriented implementation: it uses committed
    shared-memory tiles and block barriers, with no outstanding TMA/WGMMA work
    at a scheduling boundary. It makes no throughput or goodput claim before
    B300 validation. Weights must remain resident and contiguous in BF16:
    w13[L,E,2I,H] (gate followed by up), w2[L,E,H,I].
    """

    def __init__(
        self,
        buffers: OwnedBuffers,
        w13: Any,
        w2: Any,
        decode_sm_budget: int,
        expected_decode_tiles: float,
        max_decode_sm_fraction: float = 0.875,
        ctas: int = 0,
    ) -> None:
        self._buffers = buffers
        self._weights = (w13, w2)
        self._impl = load_extension().PersistentExecutor(
            buffers._impl,
            w13,
            w2,
            decode_sm_budget,
            expected_decode_tiles,
            max_decode_sm_fraction,
            ctas,
        )

    def stop(self) -> None:
        self._impl.stop()

    def check_error(self) -> list[int]:
        errors = list(self._impl.check_error())
        if any(errors):
            raise RuntimeError(f"ExpertPlex GPU slot errors: {errors}")
        return errors

    def __enter__(self) -> PersistentExecutor:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def dispatch(
    remote: RemoteBuffers,
    hidden: Any,
    topk_ids: Any,
    row_map: Any,
    epoch: int,
    layer: int,
    phase: int,
    expert_rank: int,
    expert_world_size: int,
) -> None:
    """Pack directly into peer final buffers and publish on the current stream.

    hidden is contiguous BF16[N,H]; IDs are int32/int64[N,K]; row_map is
    int32[N,K]. Each peer writes only entries belonging to that expert rank.
    phase is 0 for prefill and 1 for decode. Epochs increase monotonically and
    are restricted to [1, 2**30) for generation-tagged task queue safety.
    """
    load_extension().dispatch(
        remote._impl,
        hidden,
        topk_ids,
        row_map,
        epoch,
        layer,
        phase,
        expert_rank,
        expert_world_size,
    )


def combine(
    remote: RemoteBuffers,
    output: Any,
    topk_ids: Any,
    topk_weights: Any,
    row_map: Any,
    epoch: int,
    expert_rank: int,
    expert_world_size: int,
    timeout_ms: int = 30000,
) -> None:
    """Wait with a bounded GPU poll, then add this peer's routed results.

    output is contiguous FP32[N,H] and must be zeroed once before the first
    peer; routing weights are FP32[N,K]. Synchronizes the attention stream to
    report timeout/protocol errors before returning.
    """
    load_extension().combine(
        remote._impl,
        output,
        topk_ids,
        topk_weights,
        row_map,
        epoch,
        expert_rank,
        expert_world_size,
        timeout_ms,
    )
