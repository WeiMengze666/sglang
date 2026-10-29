"""Native residual ownership and real TBO row contracts for remote attention TP."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    _model_forward_filter_inputs,
    _tbo_padded_token_count,
)
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer
from sglang.srt.runtime_context import get_parallel


class TerminalResidualTests(unittest.TestCase):
    @staticmethod
    def layer(remote=True, terminal=True):
        return SimpleNamespace(
            mlp=SimpleNamespace(_expertplex=remote),
            ffn_boundary=SimpleNamespace(plan=SimpleNamespace(terminal=terminal)),
        )

    def test_only_remote_terminal_tp_folds_the_pending_residual(self):
        for remote in (False, True):
            for terminal in (False, True):
                for tp in (1, 2):
                    with self.subTest(remote=remote, terminal=terminal, tp=tp):
                        residual = torch.ones(1, 8, dtype=torch.bfloat16)
                        value = torch.full_like(residual, 0.00390625)
                        stream = ResidualStream(residual)
                        output = stream.record(value, PLAIN_ADD)
                        pending = stream.pending
                        batch = SimpleNamespace(residual_stream=stream)
                        with get_parallel().override(attn_tp_size=tp):
                            result = Qwen3MoeDecoderLayer._finish_remote_tp_residual(
                                self.layer(remote, terminal), output, batch
                            )
                        if remote and terminal and tp > 1:
                            # Half a BF16 ULP at 1: the native store rounds to 1.
                            self.assertTrue(torch.equal(result, residual))
                            self.assertIsNone(stream.pending)
                            self.assertIs(stream.residual, result)
                        else:
                            self.assertIs(result, output)
                            self.assertIs(stream.pending, pending)
                            self.assertIs(stream.residual, residual)

    def test_owed_sum_completed_once_before_bf16_fold_and_final_norm(self):
        for rows in (0, 1, 31, 32):
            with self.subTest(rows=rows):
                residual = torch.ones(rows, 8, dtype=torch.bfloat16)
                value = torch.full_like(residual, 0.00390625)
                group = SimpleNamespace(
                    all_reduce=Mock(side_effect=lambda x: x.clone())
                )
                stream = ResidualStream(residual)
                opaque = stream.record(UnreducedOutput(value, group=group), PLAIN_ADD)
                batch = SimpleNamespace(residual_stream=stream)
                with get_parallel().override(attn_tp_size=2):
                    folded = Qwen3MoeDecoderLayer._finish_remote_tp_residual(
                        self.layer(), opaque, batch
                    )
                expected = (value.float() + residual.float()).to(torch.bfloat16)
                self.assertTrue(torch.equal(folded, expected))
                group.all_reduce.assert_called_once()
                self.assertIsNone(stream.pending)
                norm = Mock(side_effect=lambda x: x.clone())
                result = residual_batch.final_norm(folded, batch, norm, skip_empty=True)
                self.assertTrue(torch.equal(result, expected))
                self.assertIsNone(batch.residual_stream)
                if rows:
                    norm.assert_called_once()
                    self.assertEqual(len(norm.call_args.args), 1)
                    self.assertIs(norm.call_args.args[0], folded)
                else:
                    norm.assert_not_called()


class TboRowContractTests(unittest.TestCase):
    def test_native_padding_and_remote_replicated_rows(self):
        for remote in (False, True):
            for tp in (1, 2, 4, 8):
                with (
                    patch(
                        "sglang.srt.expertplex.integration.expertplex_enabled",
                        return_value=remote,
                    ),
                    get_parallel().override(attn_tp_size=tp),
                ):
                    for rows in (0, 1, 2, 3, 7, 8, 9, 31, 32, 33):
                        expected = rows if remote else ((rows + tp - 1) // tp) * tp
                        self.assertEqual(_tbo_padded_token_count(rows), expected)

    def test_real_child_compute_positions_and_kv_locations(self):
        for remote in (False, True):
            for tp in (1, 2, 4):
                for count in (0, 1, 2, 3, 4):
                    with self.subTest(remote=remote, tp=tp, count=count):
                        size = 4
                        parent = ForwardBatch(
                            forward_mode=ForwardMode.DECODE,
                            batch_size=size,
                            input_ids=torch.arange(size),
                            positions=torch.arange(size),
                            out_cache_loc=torch.arange(size),
                            req_pool_indices=torch.arange(size),
                            seq_lens=torch.ones(size, dtype=torch.int32),
                            seq_lens_cpu=torch.ones(size, dtype=torch.int32),
                            seq_lens_sum=size,
                            spec_info=None,
                        )
                        with (
                            patch(
                                "sglang.srt.expertplex.integration.expertplex_enabled",
                                return_value=remote,
                            ),
                            get_parallel().override(
                                attn_tp_size=tp, moe_dense_tp_size=None
                            ),
                        ):
                            child = TboForwardBatchPreparer.filter_batch(
                                parent,
                                start_token_index=0,
                                end_token_index=count,
                                start_seq_index=0,
                                end_seq_index=count,
                                out_num_token_non_padded=torch.tensor(count),
                                out_num_token_non_padded_cpu=count,
                            )
                            hidden = torch.arange(size * 8).reshape(size, 8).bfloat16()
                            result = _model_forward_filter_inputs(
                                hidden, hidden.clone(), parent.positions, child, 0
                            )
                        expected = count if remote else ((count + tp - 1) // tp) * tp
                        self.assertEqual(child.tbo_padded_len, expected)
                        self.assertEqual(result["hidden_states"].shape[0], expected)
                        self.assertEqual(result["positions"].shape[0], expected)
                        self.assertTrue(
                            torch.equal(result["hidden_states"][:count], hidden[:count])
                        )
                        self.assertEqual(child.out_cache_loc.numel(), count)
                        self.assertEqual(child.batch_size, count)
                        if remote:
                            self.assertEqual(child.out_cache_loc.numel(), expected)


if __name__ == "__main__":
    unittest.main()
