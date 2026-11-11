"""Grouped-D autotune warmup must supply the route mask's raw token counts.

Exercise the real dummy-forward builder and route mask with CPU tensors. The
model and distributed barrier are boundaries; no CUDA or NCCL claim is made.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.expertplex.attention_dp import mask_topk_padding
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_runner import (
    BaseRunner,
    _allocate_decode_buffers,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertPlexDummyCounts(CustomTestCase):
    def test_warmup_routes_and_native_metadata(self):
        for expertplex, dp in [(True, 3), (True, 1), (False, 3)]:
            with (
                self.subTest(expertplex=expertplex, dp=dp),
                get_context().override_server_args(
                    enable_expertplex=expertplex,
                    tp_size=dp,
                    attn_dp_size=dp,
                    moe_a2a_backend="none",
                ),
                get_parallel().override(tp_group=SimpleNamespace(barrier=lambda: None)),
            ):
                buffers = _allocate_decode_buffers(
                    device=torch.device("cpu"),
                    max_bs=4,
                    max_num_token=8,
                    hidden_size=4,
                    vocab_size=8,
                    dtype=torch.bfloat16,
                    is_encoder_decoder=False,
                    require_mlp_tp_gather=dp > 1,
                    seq_len_fill_value=1,
                    encoder_len_fill_value=0,
                    num_tokens_per_req=1,
                    cache_loc_dtype=torch.int64,
                    enable_mamba_track=False,
                )
                seen = []

                def forward(
                    input_ids,
                    positions,
                    batch,
                    *,
                    seen=seen,
                    expertplex=expertplex,
                    dp=dp,
                ):
                    seen.append(batch)
                    if expertplex and dp > 1:
                        rows = input_ids.numel() * dp
                        ids = torch.arange(rows * 2).reshape(rows, 2) % 8
                        weights = torch.ones((rows, 2), dtype=torch.float32)
                        routes = StandardTopKOutput(weights, ids, None)
                        result = mask_topk_padding(routes, batch, dp)
                        self.assertTrue(torch.equal(result.topk_ids, ids))
                        self.assertTrue(torch.equal(result.topk_weights, weights))
                        self.assertEqual(
                            batch.original_global_num_tokens_cpu,
                            [input_ids.numel()] * dp,
                        )
                    else:
                        self.assertIsNone(batch.expertplex_dp_real_num_tokens_gpu)

                runner = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        device="cpu",
                        is_generation=True,
                        is_draft_worker=False,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        lora_manager=None,
                        model=SimpleNamespace(forward=forward),
                        kv_index_translator=SimpleNamespace(bind_runner_slots=Mock()),
                        prepare_dummy_forward_batch=lambda batch: batch,
                        attn_backend=SimpleNamespace(init_forward_metadata=Mock()),
                    )
                )
                # Reuse the same buffers at changing shapes, as the autotuner does.
                for mode, bs, width in [
                    (ForwardMode.DECODE, 4, None),
                    (ForwardMode.DECODE, 1, None),
                    (ForwardMode.EXTEND, 2, 3),
                ]:
                    BaseRunner._dummy_run(
                        runner,
                        bs,
                        forward_mode_override=mode,
                        buffers=buffers,
                        extend_num_tokens_per_req=width,
                    )
                self.assertEqual(len(seen), 3)
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
