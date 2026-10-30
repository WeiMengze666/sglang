"""Check child-local TBO KV indices and BF16 Triton attention bit-for-bit.

Run through the site's GPU scheduler. This is an attention component regression,
not a full-model numerical or performance result.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.extend_attention import (
    build_unified_kv_indices,
    extend_attention_fwd_unified,
)
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    compute_split_seq_index,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel


def indptr(lengths):
    return torch.tensor(
        [0] + [sum(lengths[: i + 1]) for i in range(len(lengths))],
        dtype=torch.int32,
        device="cuda",
    )


def check_case(lengths, prefixes, seed, *, remote):
    torch.manual_seed(seed)
    total = sum(lengths)
    cache_rows = sum(lengths) + sum(prefixes)
    slots = torch.randperm(cache_rows, device="cuda")
    req_indices = []
    cursor = 0
    for length, prefix in zip(lengths, prefixes, strict=True):
        req_indices.append(slots[cursor : cursor + length + prefix])
        cursor += length + prefix
    new_slots = torch.cat([r[p:] for r, p in zip(req_indices, prefixes, strict=True)])
    seq_lens = torch.tensor(
        [a + b for a, b in zip(lengths, prefixes, strict=True)], dtype=torch.int32
    )
    parent = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(lengths),
        input_ids=torch.arange(total, device="cuda"),
        positions=torch.cat(
            [
                torch.arange(p, p + n, device="cuda")
                for n, p in zip(lengths, prefixes, strict=True)
            ]
        ),
        out_cache_loc=new_slots,
        req_pool_indices=torch.arange(len(lengths), device="cuda"),
        seq_lens=seq_lens.cuda(),
        seq_lens_cpu=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
        extend_prefix_lens_cpu=prefixes,
        extend_start_loc=indptr(lengths)[:-1],
        extend_num_tokens=total,
        spec_info=None,
    )
    parent.tbo_split_seq_index = compute_split_seq_index(
        ForwardMode.EXTEND, total, lengths, None
    )
    # The preparer only checks the dispatcher's type; no backend methods are used.
    dispatcher = object.__new__(TboAttnBackend)
    with (
        patch(
            "sglang.srt.batch_overlap.two_batch_overlap.get_attn_backend",
            return_value=dispatcher,
        ),
        patch(
            "sglang.srt.expertplex.integration.expertplex_enabled", return_value=remote
        ),
    ):
        counts = TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(
            parent
        )
        TboForwardBatchPreparer.prepare_raw(
            parent, tbo_children_num_token_non_padded=counts
        )

    # Actual Coder-480B attention dimensions: 96 Q heads, 8 KV heads, head_dim128.
    query = torch.randn(total, 96, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(cache_rows, 8, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    def run(batch, lo, hi):
        child_prefixes = batch.extend_prefix_lens_cpu
        rows = batch.req_pool_indices.cpu().tolist()
        prefix_indices = torch.cat(
            [req_indices[r][:p] for r, p in zip(rows, child_prefixes, strict=True)]
        )
        kv_ptr, indices, prefix_lens = build_unified_kv_indices(
            indptr(child_prefixes),
            prefix_indices,
            batch.extend_start_loc,
            batch.extend_seq_lens,
            batch.out_cache_loc,
            batch.batch_size,
        )
        expected = torch.cat(
            [
                req_indices[r][: p + n]
                for r, p, n in zip(
                    rows, child_prefixes, batch.extend_seq_lens_cpu, strict=True
                )
            ]
        )
        torch.cuda.synchronize()
        assert torch.equal(indices, expected), "TBO changed request KV identity"
        output = torch.empty_like(query[lo:hi])
        extend_attention_fwd_unified(
            query[lo:hi],
            output,
            key,
            value,
            1.0,
            1.0,
            indptr(batch.extend_seq_lens_cpu),
            kv_ptr,
            indices,
            prefix_lens,
            max(batch.extend_seq_lens_cpu),
        )
        return output

    reference = run(parent, 0, total)
    children = [
        run(child, *child.tbo_parent_token_range) for child in parent.tbo_children
    ]
    result = torch.cat(children)
    torch.cuda.synchronize()
    assert torch.isfinite(reference).all() and torch.isfinite(result).all()
    assert torch.equal(reference.view(torch.int16), result.view(torch.int16)), (
        "TBO attention differs from unsplit batch"
    )
    return dict(
        lengths=lengths,
        prefixes=prefixes,
        remote=remote,
        child_rows=[c.extend_num_tokens for c in parent.tbo_children],
        exact_elements=result.numel(),
        status="bitexact",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ([128, 128], [0, 31]),
        ([3, 5, 7, 9], [7, 11, 13, 17]),
        ([257, 255], [129, 3]),
        ([513], [31]),
        ([4096, 4096], [0, 128]),
    ]
    results = []
    with (
        get_context().override_server_args(
            device="cuda", attention_backend="triton", moe_dense_tp_size=None
        ),
        get_parallel().override(attn_tp_size=1),
    ):
        for remote in (False, True):
            for seed, (lengths, prefixes) in enumerate(cases):
                results.append(check_case(lengths, prefixes, seed, remote=remote))
    args.output.write_text(
        json.dumps(
            dict(
                passed=True,
                results=results,
                scope="Triton BF16 attention component only; no full-model or goodput acceptance",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
