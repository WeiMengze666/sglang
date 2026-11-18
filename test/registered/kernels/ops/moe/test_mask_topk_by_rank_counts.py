"""Bitwise routing preservation and dynamic graph counts for rank padding."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def reference(ids, weights, counts):
    rows = ids.shape[0]
    index = torch.arange(rows, device=ids.device)
    stride = rows // counts.numel()
    valid = (
        index.remainder(stride) < counts[index.div(stride, rounding_mode="floor")]
    )[:, None]
    return torch.where(valid, ids, ids.new_full((), -1)), torch.where(
        valid, weights, weights.new_zeros(())
    )


class RankPaddingTests(CustomTestCase):
    def assert_outputs(self, actual, expected):
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(
            torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32))
        )

    def test_preserves_valid_bits_and_masks_each_rank(self):
        from sglang.kernels.ops.moe.mask_topk_by_rank_counts import (
            mask_topk_by_rank_counts,
        )

        for dtype in (torch.int32, torch.int64):
            for ranks, stride, top_k, strided in (
                (2, 1, 1, False),
                (3, 4, 3, False),
                (3, 16, 8, False),
                (4, 32, 8, True),
                (3, 17, 5, True),
            ):
                with self.subTest(
                    dtype=dtype,
                    ranks=ranks,
                    stride=stride,
                    top_k=top_k,
                    strided=strided,
                ):
                    rows = ranks * stride
                    ids = (
                        torch.arange(rows * top_k, device="cuda", dtype=dtype).reshape(
                            rows, top_k
                        )
                        % 160
                    )
                    weights = (
                        torch.arange(
                            rows * top_k, device="cuda", dtype=torch.float32
                        ).reshape(rows, top_k)
                        / 128
                    )
                    weights[0, 0] = -0.0
                    if top_k > 1:
                        weights[0, 1] = float("nan")
                    counts = torch.tensor(
                        [max(0, stride - r) for r in range(ranks)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    if strided:
                        ids = ids.t().contiguous().t()
                        weights = weights.t().contiguous().t()
                        counts = torch.stack((counts, counts), dim=1)[:, 0]
                    ids_before, weights_before = ids.clone(), weights.clone()
                    self.assert_outputs(
                        mask_topk_by_rank_counts(ids, weights, counts),
                        reference(ids, weights, counts),
                    )
                    self.assertTrue(torch.equal(ids, ids_before))
                    self.assertTrue(
                        torch.equal(
                            weights.contiguous().view(torch.int32),
                            weights_before.contiguous().view(torch.int32),
                        )
                    )

    def test_graph_reads_changed_counts_at_fixed_addresses(self):
        from sglang.kernels.ops.moe.mask_topk_by_rank_counts import (
            mask_topk_by_rank_counts,
        )

        for dtype in (torch.int32, torch.int64):
            ids = torch.arange(48 * 8, device="cuda", dtype=dtype).reshape(48, 8) % 160
            weights = (
                torch.arange(48 * 8, device="cuda", dtype=torch.float32).reshape(48, 8)
                / 512
            )
            counts = torch.tensor([16, 16, 16], device="cuda", dtype=torch.int32)
            mask_topk_by_rank_counts(ids, weights, counts)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = mask_topk_by_rank_counts(ids, weights, counts)
            pointer = counts.data_ptr()
            for sizes in ([1, 0, 9], [16, 2, 0], [0, 0, 0], [16, 16, 16]):
                counts.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
                graph.replay()
                self.assertEqual(counts.data_ptr(), pointer)
                self.assert_outputs(actual, reference(ids, weights, counts))

    def test_empty_rows(self):
        from sglang.kernels.ops.moe.mask_topk_by_rank_counts import (
            mask_topk_by_rank_counts,
        )

        ids = torch.empty((0, 8), device="cuda", dtype=torch.int32)
        weights = torch.empty((0, 8), device="cuda", dtype=torch.float32)
        counts = torch.zeros(3, device="cuda", dtype=torch.int32)
        actual = mask_topk_by_rank_counts(ids, weights, counts)
        self.assertEqual(actual[0].shape, ids.shape)
        self.assertEqual(actual[1].dtype, weights.dtype)

    def test_child_offsets_read_parent_counts_on_graph_replay(self):
        from sglang.kernels.ops.moe.mask_topk_by_rank_counts import (
            mask_topk_by_rank_counts,
        )

        for dtype in (torch.int32, torch.int64):
            ids = torch.arange(12 * 8, device="cuda", dtype=dtype).reshape(12, 8)
            weights = (
                torch.arange(12 * 8, device="cuda", dtype=torch.float32).reshape(12, 8)
                / 128
            )
            weights[0, 0] = -0.0
            counts = torch.full((3,), 8, dtype=torch.int32, device="cuda")
            for offset in (0, 4):
                mask_topk_by_rank_counts(ids, weights, counts, rank_token_offset=offset)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                first = mask_topk_by_rank_counts(
                    ids, weights, counts, rank_token_offset=0
                )
                second = mask_topk_by_rank_counts(
                    ids, weights, counts, rank_token_offset=4
                )
            address = counts.data_ptr()
            for values in ((0, 3, 8), (8, 0, 5), (1, 7, 0), (0, 0, 0), (8, 8, 8)):
                counts.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
                graph.replay()
                torch.cuda.synchronize()
                for offset, output in ((0, first), (4, second)):
                    self.assert_outputs(
                        output, reference(ids, weights, (counts - offset).clamp(0, 4))
                    )
                self.assertEqual(counts.data_ptr(), address)


if __name__ == "__main__":
    unittest.main()
