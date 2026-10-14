"""Lazy CUDA bindings for ExpertPlex's experimental single-node executor.

Importing this module does not import torch or require a CUDA installation. The
extension is built on first use, on the deployment host. CUDA 12.9 or newer and
a PyTorch build which supports Blackwell are required for the B300 target.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import Any


def source_fingerprint() -> str:
    """Identify the exact native implementation in benchmark artifacts."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in (root / "native.py", *sorted((root / "csrc").glob("*"))):
        if path.is_file():
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def estimate_buffer_bytes(
    slot_count: int,
    local_experts: int,
    max_rows: int,
    hidden: int,
    intermediate: int,
    *,
    capacities: list[int] | None = None,
    packed: list[int] | None = None,
    top_k: int = 0,
) -> dict[str, int]:
    """Exact requested cudaMalloc bytes, excluding allocator/driver overhead."""
    dims = (slot_count, local_experts, max_rows, hidden, intermediate)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims):
        raise ValueError("ExpertPlex buffer dimensions must be positive integers")
    capacities = capacities or [max_rows] * slot_count
    packed = packed or [0] * slot_count
    if len(capacities) != slot_count or len(packed) != slot_count:
        raise ValueError("Mailbox layout length mismatch")
    rows = 0
    for capacity, is_packed in zip(capacities, packed):
        if not 0 < capacity <= max_rows or is_packed not in (0, 1):
            raise ValueError("Invalid mailbox layout")
        if is_packed and top_k <= 0:
            raise ValueError("Packed layout requires top_k")
        storage = (
            capacity * min(top_k, local_experts) + 63 * local_experts
            if is_packed
            else capacity * local_experts
        )
        rows += (storage + 63) // 64 * 64
    result = {
        "input": rows * hidden * 2,
        "activation": rows * intermediate * 2,
        "output": rows * hidden * 2,
        "counts": slot_count * local_experts * 4,
        "offsets": slot_count * local_experts * 4,
        "control": slot_count * 8 * 8,
        "scheduler": slot_count * 32,
        "stop": 4,
    }
    result["total"] = sum(result.values())
    return result


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    import importlib.util

    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("ExpertPlex native executor requires a CUDA deployment host")
    # Supplying an explicit architecture also prevents cpp_extension from
    # compiling for every visible device on an eight-GPU deployment host.
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "ExpertPlex SM100 pipeline requires sgl-deep-gemm 0.2.2 headers"
        )
    include = Path(spec.origin).parent / "include"
    return load(
        name="sglang_expertplex_native_v2",
        sources=[str(Path(__file__).parent / "csrc" / "expertplex.cu")],
        extra_include_paths=[str(include)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        verbose=False,
    )


class OwnedBuffers:
    """Dedicated cudaMalloc allocations whose IPC handles stay valid until close.

    Prefill can use contiguous routed rows with 64-row expert alignment;
    decode uses fixed per-expert masked capacity. A slot admits exactly one
    outstanding invocation. Descriptors and allocations stay fixed for replay.
    """

    def __init__(
        self,
        slot_count: int,
        local_experts: int,
        max_rows: int,
        hidden: int,
        intermediate: int,
        device: int,
        *,
        capacities: list[int] | None = None,
        packed: list[int] | None = None,
        top_k: int = 0,
    ) -> None:
        self._impl = load_extension().OwnedBuffers(
            slot_count,
            local_experts,
            max_rows,
            hidden,
            intermediate,
            device,
            capacities or [],
            packed or [],
            top_k,
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
    """One long-lived expert executor per GPU.

    ``sm100`` uses two-CTA clusters, TMA multicast, DSMEM decisions, TCGEN05
    and separate producer/MMA/epilogue warps. ``sm100_cta1`` is its single-CTA
    ablation for measuring cooperative scheduling overhead.
    ``wmma`` retains the original reference implementation for comparisons.
    See the conformance record for remaining performance/preemption validation.
    Weights must remain resident and contiguous in BF16:
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
        kernel: str | None = None,
    ) -> None:
        self._buffers = buffers
        self._weights = (w13, w2)
        self.kernel = kernel or os.environ.get("SGLANG_EXPERTPLEX_KERNEL", "sm100")
        self._impl = load_extension().PersistentExecutor(
            buffers._impl,
            w13,
            w2,
            decode_sm_budget,
            expected_decode_tiles,
            max_decode_sm_fraction,
            ctas,
            self.kernel,
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


def read_timing(remote: RemoteBuffers):
    """Read [done, ready, expert start ns, expert finish ns] on the current stream.

    Call after combine and before reusing this mailbox. Both timestamps use the
    expert GPU's global timer; their difference excludes attention-side transfer.
    This measures the full expert operation including APK scheduling waits.
    """
    return load_extension().read_timing(remote._impl)


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
    synchronize: bool = True,
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
        synchronize,
    )
