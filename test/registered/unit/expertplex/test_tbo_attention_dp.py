"""Grouped TBO must preserve rank holes and switch native collective buffers."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import operations, two_batch_overlap as tbo
from sglang.srt.expertplex import attention_dp
from sglang.srt.expertplex.attention_dp import mask_topk_padding
from sglang.srt.layers import dp_attention
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def parent_batch(width=8, counts=(8, 5, 0), idle=False):
    return ForwardBatch(
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        batch_size=width,
        input_ids=torch.zeros(width, dtype=torch.long),
        positions=torch.zeros(width, dtype=torch.long),
        out_cache_loc=torch.zeros(width, dtype=torch.long),
        req_pool_indices=torch.zeros(width, dtype=torch.long),
        seq_lens=torch.ones(width, dtype=torch.int32),
        seq_lens_cpu=torch.ones(width, dtype=torch.int32),
        seq_lens_sum=width,
        global_dp_buffer_len=width * len(counts),
        global_num_tokens_cpu=[width] * len(counts),
        original_global_num_tokens_cpu=list(counts),
        expertplex_dp_real_num_tokens_gpu=torch.tensor(counts, dtype=torch.int32),
        dp_padding_mode=DpPaddingMode.MAX_LEN,
        tbo_split_seq_index=0 if idle else 2,
    )


def child_batch(parent, start, end):
    with (
        patch.object(
            tbo, "get_parallel", return_value=SimpleNamespace(moe_dense_tp_size=3)
        ),
        patch(
            "sglang.srt.expertplex.integration.expertplex_enabled", return_value=True
        ),
    ):
        return tbo.TboForwardBatchPreparer.filter_batch(
            parent,
            start_token_index=start,
            end_token_index=end,
            start_seq_index=start,
            end_seq_index=end,
            out_num_token_non_padded=torch.tensor(end - start, dtype=torch.int32),
        )


def routes(rows):
    ids = torch.arange(rows * 2, dtype=torch.int64).reshape(rows, 2)
    weights = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 128
    if rows:
        weights[0, 0] = -0.0
    return StandardTopKOutput(weights, ids, None)


class GroupedTboChildren(CustomTestCase):
    def setUp(self):
        super().setUp()
        from contextlib import ExitStack
        from sglang.srt.runtime_context import get_context, get_parallel

        context = ExitStack()
        self.addCleanup(context.close)
        context.enter_context(
            get_context().override_server_args(
                model_path="dummy", enable_expertplex=True
            )
        )
        context.enter_context(
            get_parallel().override(
                tp_size=3,
                moe_tp_size=3,
                attn_dp_size=3,
                attn_tp_size=1,
                tp_rank=0,
                attn_dp_rank=0,
                moe_tp_rank=0,
            )
        )

    def test_native_scatter_reads_the_child_rank_segment(self):
        from sglang.srt.runtime_context import get_parallel

        parent = parent_batch()
        for rank in range(3):
            with get_parallel().override(
                tp_rank=rank, attn_dp_rank=rank, moe_tp_rank=rank
            ):
                for start, end in ((0, 3), (3, 8)):
                    child = child_batch(parent, start, end)
                    width = end - start
                    gathered = torch.arange(3 * width * 4).reshape(3 * width, 4)
                    local = torch.empty_like(gathered[:width])
                    with patch.object(
                        dp_attention, "memcpy_func", dp_attention.memcpy_cpu
                    ):
                        dp_attention.dp_scatter(local, gathered, child)
                    self.assertTrue(
                        torch.equal(local, gathered[rank * width : (rank + 1) * width])
                    )
                    self.assertEqual(child.global_num_tokens_gpu.tolist(), [width] * 3)

    def test_filter_preserves_rank_holes_and_child_collective_sizes(self):
        parent = parent_batch()
        first, second = child_batch(parent, 0, 3), child_batch(parent, 3, 8)
        self.assertEqual(first.original_global_num_tokens_cpu, [3, 3, 0])
        self.assertEqual(second.original_global_num_tokens_cpu, [5, 2, 0])
        for child, width in ((first, 3), (second, 5)):
            self.assertEqual(child.global_dp_buffer_len, width * 3)
            self.assertEqual(child.global_num_tokens_cpu, [width] * 3)
            self.assertEqual(child.global_num_tokens_padded_cpu, [width] * 3)
            self.assertEqual(child.tbo_padded_len, width)
            self.assertIs(
                child.expertplex_dp_real_num_tokens_gpu,
                parent.expertplex_dp_real_num_tokens_gpu,
            )
        self.assertEqual(parent.original_global_num_tokens_cpu, [8, 5, 0])
        self.assertEqual(parent.global_dp_buffer_len, 24)

    def test_second_child_uses_parent_offset_and_live_counts(self):
        parent = parent_batch(counts=(8, 8, 8))
        children = [child_batch(parent, 0, 4), child_batch(parent, 4, 8)]
        routed = routes(12)
        # CPU capture metadata stays fixed while the registry-owned device
        # vector changes at replay. Compare the independent parent-row indices.
        for counts in ((0, 3, 8), (8, 0, 5), (1, 7, 0), (0, 0, 0)):
            parent.expertplex_dp_real_num_tokens_gpu.copy_(torch.tensor(counts))
            for child, start in zip(children, (0, 4)):
                out = mask_topk_padding(routed, child, 3)
                live = torch.tensor(
                    [
                        start + row < counts[rank]
                        for rank in range(3)
                        for row in range(4)
                    ]
                )
                self.assertTrue(torch.equal(out.topk_ids[live], routed.topk_ids[live]))
                self.assertTrue(
                    torch.equal(
                        out.topk_weights[live].view(torch.int32),
                        routed.topk_weights[live].view(torch.int32),
                    )
                )
                self.assertTrue(torch.all(out.topk_ids[~live] == -1))
                self.assertTrue(
                    torch.all(out.topk_weights[~live].view(torch.int32) == 0)
                )

    def test_alternating_stages_publish_child_buffer_lengths(self):
        parent = parent_batch()
        children = [child_batch(parent, 0, 3), child_batch(parent, 3, 8)]
        wrapper = dp_attention._DpGatheredBufferWrapper
        names = (
            "_global_dp_buffer_len",
            "_local_dp_buffer_len",
            "_dp_max_padding",
            "_global_num_tokens",
            "_global_num_tokens_gpu",
        )
        saved = {name: getattr(wrapper, name) for name in names}
        seen = []

        def observe(state, **unused):
            seen.append(
                (
                    wrapper.get_global_dp_buffer_len(),
                    wrapper.get_local_dp_buffer_len(),
                    wrapper.get_dp_global_num_tokens(),
                    wrapper.get_dp_global_num_tokens_gpu().tolist(),
                )
            )

        try:
            stage = [operations.ExecutionOperation("observe", observe)]
            executors = [
                operations._StageExecutor(
                    str(i), [stage, stage], {"forward_batch": child}
                )
                for i, child in enumerate(children)
            ]
            with patch.object(
                operations,
                "operations_nvtx_range",
                side_effect=lambda **_: nullcontext(),
            ):
                for executor in (
                    executors[0],
                    executors[1],
                    executors[0],
                    executors[1],
                ):
                    executor.next()
            self.assertEqual(
                seen, [(9, 3, [3] * 3, [3] * 3), (15, 5, [5] * 3, [5] * 3)] * 2
            )
        finally:
            for name, value in saved.items():
                setattr(wrapper, name, value)

    def test_split_is_uniform_for_active_and_idle_ranks(self):
        for width in (2, 3, 8, 16):
            for idle in (False, True):
                parent = parent_batch(width=width, counts=(width, 0, 1), idle=idle)
                split = tbo.TboForwardBatchPreparer._compute_split_token_index(parent)
                self.assertEqual(split, width // 2)
                for start, end in ((0, split), (split, width)):
                    child = child_batch(parent, start, end)
                    self.assertEqual(child.forward_mode, ForwardMode.DECODE)
                    self.assertEqual(child.input_ids.numel(), end - start)

    def test_rejects_inconsistent_parent_collective_geometry(self):
        parent = parent_batch()
        for counts in ([8, 4, 8], [8, 8], None):
            parent.global_num_tokens_cpu = counts
            with self.assertRaisesRegex(ValueError, "equal parent"):
                attention_dp.tbo_child_dp_metadata(parent, 0, 4)
        parent = parent_batch()
        parent.dp_padding_mode = DpPaddingMode.SUM_LEN
        with self.assertRaisesRegex(ValueError, "MAX_LEN"):
            attention_dp.tbo_child_dp_metadata(parent, 0, 4)
        parent = parent_batch()
        for start, end in ((-1, 4), (5, 4), (0, 9), (True, 4)):
            with self.assertRaisesRegex(ValueError, "token range"):
                attention_dp.tbo_child_dp_metadata(parent, start, end)

    def test_qwen_tbo_routes_mask_each_gathered_rank(self):
        from sglang.srt.models import qwen3_moe

        child = parent_batch(width=4, counts=(4, 1, 0))
        child.tbo_parent_token_range = (4, 8)
        child.expertplex_dp_real_num_tokens_gpu.copy_(torch.tensor([8, 5, 0]))
        child.moe_num_token_non_padded = lambda: None
        routed = routes(12)
        block = SimpleNamespace(
            layer_id=0,
            _expertplex=True,
            _expertplex_attention_dp_size=3,
            topk=lambda **unused: routed,
        )
        state = SimpleNamespace(
            hidden_states_mlp_input=torch.zeros((12, 2)),
            forward_batch=child,
            pop=lambda key: torch.ones((12, 160)),
        )
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        with (
            patch.object(
                qwen3_moe,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch.object(
                qwen3_moe.ExpertLocationDispatchInfo, "init_new", return_value=None
            ),
        ):
            qwen3_moe.Qwen3MoeSparseMoeBlock.op_select_experts(block, state)
        self.assertTrue(
            torch.equal(state.topk_output.topk_ids[:5], routed.topk_ids[:5])
        )
        self.assertTrue(torch.all(state.topk_output.topk_ids[5:] == -1))
        self.assertTrue(torch.all(state.topk_output.topk_weights[5:] == 0))

    def test_native_eager_padding_uses_uniform_geometry_only_for_tbo(self):
        from sglang.srt.model_executor import forward_batch_info as fbi

        counts = [0, 3, 8]
        for enabled in (False, True):
            for rank in range(3):
                parent = parent_batch(width=counts[rank], counts=counts, idle=rank == 0)
                parent.global_num_tokens_cpu = counts.copy()
                parent.global_num_tokens_for_logprob_cpu = counts.copy()
                parent.global_num_tokens_gpu = torch.empty(3, dtype=torch.int32)
                parent.tbo_split_seq_index = 0 if enabled else None
                parent.can_run_dp_prefill_cuda_graph = False

                def pad(model_runner, num_tokens, bs):
                    parent.input_ids = torch.zeros(num_tokens, dtype=torch.long)

                parent._pad_inputs_to_size = pad
                runner = SimpleNamespace(
                    model_config=object(),
                    is_draft_worker=False,
                    attn_tp_sequence_sharded=lambda _: False,
                )
                with (
                    patch.object(
                        fbi,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1, attn_dp_size=3),
                    ),
                    patch.object(
                        fbi,
                        "get_exec",
                        return_value=SimpleNamespace(
                            graph=SimpleNamespace(
                                cuda_graph_config=SimpleNamespace(
                                    prefill=SimpleNamespace(bs=[])
                                )
                            )
                        ),
                    ),
                    patch.object(fbi, "mambaish_config", return_value=None),
                    patch.object(
                        fbi,
                        "_elastic_should_preserve_local_token_counts",
                        return_value=False,
                    ),
                    patch.object(fbi, "set_dp_buffer_len_from_batch"),
                    patch.object(fbi, "set_is_extend_in_batch"),
                    patch.object(fbi, "dp_slot_in", return_value=rank),
                    patch.object(fbi, "_is_cpu", True),
                    patch.object(
                        attention_dp, "grouped_attention_dp_size", return_value=3
                    ),
                    patch.object(
                        DpPaddingMode,
                        "get_dp_padding_mode",
                        return_value=DpPaddingMode.SUM_LEN,
                    ),
                    patch.object(tbo.TboForwardBatchPreparer, "prepare"),
                ):
                    parent.prepare_mlp_sync_batch(runner)
                expected = [8, 8, 8] if enabled else counts
                self.assertEqual(parent.global_num_tokens_cpu, expected)
                self.assertEqual(parent.global_dp_buffer_len, sum(expected))
                self.assertEqual(parent.input_ids.numel(), expected[rank])
                self.assertEqual(
                    parent.expertplex_dp_real_num_tokens_gpu.tolist(), counts
                )


if __name__ == "__main__":
    unittest.main()
