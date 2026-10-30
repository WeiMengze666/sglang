"""TBO child query starts index child-local Q and new-KV arrays."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel


class TboExtendOffsets(unittest.TestCase):
    def split(self, lengths, start, end, *, remote):
        count = sum(lengths)
        starts = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0)[:-1]), dtype=torch.int32
        )
        prefix = [7 + i for i in range(len(lengths))]
        parent = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(lengths),
            input_ids=torch.arange(count),
            positions=torch.arange(count),
            out_cache_loc=torch.arange(count) + 1000,
            req_pool_indices=torch.arange(len(lengths)),
            seq_lens=torch.tensor(
                [a + b for a, b in zip(lengths, prefix)], dtype=torch.int32
            ),
            seq_lens_cpu=torch.tensor(
                [a + b for a, b in zip(lengths, prefix)], dtype=torch.int32
            ),
            seq_lens_sum=sum(lengths) + sum(prefix),
            extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32),
            extend_prefix_lens_cpu=prefix,
            extend_start_loc=starts,
            extend_num_tokens=count,
            spec_info=None,
        )
        lo, hi = sum(lengths[:start]), sum(lengths[:end])
        with (
            get_context().override_server_args(
                attention_backend="triton", moe_dense_tp_size=None
            ),
            get_parallel().override(attn_tp_size=1),
            patch(
                "sglang.srt.expertplex.integration.expertplex_enabled",
                return_value=remote,
            ),
        ):
            child = TboForwardBatchPreparer.filter_batch(
                parent,
                start_token_index=lo,
                end_token_index=hi,
                start_seq_index=start,
                end_seq_index=end,
                out_num_token_non_padded=torch.tensor(hi - lo, dtype=torch.int32),
                out_num_token_non_padded_cpu=hi - lo,
            )
        return parent, child

    def test_child_kv_indices_are_in_bounds_and_preserve_request_identity(self):
        for remote in (False, True):
            for lengths, start, end in [
                ([128, 128], 1, 2),
                ([3, 5, 7, 9], 2, 4),
                ([2, 4, 8], 1, 3),
            ]:
                with self.subTest(remote=remote, lengths=lengths):
                    parent, child = self.split(lengths, start, end, remote=remote)
                    offset = 0
                    for i, n in enumerate(lengths[start:end]):
                        self.assertEqual(child.extend_start_loc[i].item(), offset)
                        indices = child.extend_start_loc[i].item() + torch.arange(n)
                        self.assertLess(
                            indices.max().item(), child.out_cache_loc.numel()
                        )
                        expected_lo = sum(lengths[: start + i])
                        self.assertTrue(
                            torch.equal(
                                child.out_cache_loc[indices],
                                parent.out_cache_loc[expected_lo : expected_lo + n],
                            )
                        )
                        offset += n
                    # Child rebasing must not mutate parent metadata or cached lengths.
                    expected = [sum(lengths[:i]) for i in range(len(lengths))]
                    self.assertEqual(parent.extend_start_loc.tolist(), expected)
                    self.assertEqual(
                        child.extend_prefix_lens_cpu,
                        parent.extend_prefix_lens_cpu[start:end],
                    )

    def test_first_and_empty_children(self):
        for start, end in [(0, 1), (2, 2)]:
            _, child = self.split([3, 5], start, end, remote=True)
            self.assertEqual(child.extend_start_loc.tolist(), [0] if start == 0 else [])


if __name__ == "__main__":
    unittest.main()
