"""Remote EP must reuse native graph token bounds without changing native MoE."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import (
    enable_num_token_non_padded,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.layers.moe.utils import MoeA2ABackend


class PaddingRoutesTests(unittest.TestCase):
    def test_unpublished_native_context_does_not_read_disagg(self):
        with (
            patch("sglang.srt.runtime_context.get_context") as context,
            patch("sglang.srt.expertplex.integration.expertplex_enabled") as enabled,
            get_parallel().override(moe_ep_size=1, attn_dp_size=1),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
            get_flags().dp.override(use_world_group_for_gather=False),
        ):
            context.return_value.is_config_namespace_published.return_value = False
            self.assertFalse(enable_num_token_non_padded())
            enabled.assert_not_called()

    def test_native_forward_keeps_one_argument(self):
        for remote in (False, True):
            layer = SimpleNamespace(
                _expertplex=remote, _use_mega_moe=False, forward_normal=Mock()
            )
            states, batch = object(), object()
            with (
                patch(
                    "sglang.srt.models.qwen3_moe.is_deepep_class_backend",
                    return_value=False,
                ),
                get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
            ):
                Qwen3MoeSparseMoeBlock.forward(layer, states, batch)
            layer.forward_normal.assert_called_once_with(
                *((states, batch) if remote else (states,))
            )

    def test_remote_ep_one_enables_native_graph_count(self):
        for remote in (False, True):
            for backend in (MoeA2ABackend.NONE, MoeA2ABackend.CUSTOMIZED):
                with (
                    self.subTest(remote=remote, backend=backend),
                    patch("sglang.srt.runtime_context.get_context") as context,
                    patch(
                        "sglang.srt.expertplex.integration.expertplex_enabled",
                        return_value=remote,
                    ),
                    get_parallel().override(moe_ep_size=1, attn_dp_size=1),
                    get_flags().moe.override(a2a_backend=backend),
                    get_flags().dp.override(use_world_group_for_gather=False),
                ):
                    context.return_value.is_config_namespace_published.return_value = (
                        True
                    )
                    self.assertEqual(
                        enable_num_token_non_padded(), remote or backend.is_customized()
                    )

    def test_remote_only_forwards_the_native_bound_by_identity(self):
        for remote in (False, True):
            for actual in (0, 1, 3, 4):
                with self.subTest(remote=remote, actual=actual):
                    count = torch.tensor(actual, dtype=torch.int32)
                    batch = SimpleNamespace(
                        moe_num_token_non_padded=Mock(return_value=count)
                    )
                    states = torch.randn(4, 8, dtype=torch.bfloat16)
                    routed = object()
                    layer = SimpleNamespace(
                        _expertplex=remote,
                        gate=Mock(return_value=(torch.zeros(4, 16), None)),
                        topk=Mock(return_value=routed),
                        experts=Mock(return_value=states.clone()),
                    )
                    got = Qwen3MoeSparseMoeBlock.forward_normal(layer, states, batch)
                    self.assertTrue(torch.equal(got, states))
                    layer.experts.assert_called_once()
                    self.assertTrue(
                        torch.equal(layer.experts.call_args.args[0], states)
                    )
                    self.assertIs(layer.experts.call_args.args[1], routed)
                    if remote:
                        batch.moe_num_token_non_padded.assert_called_once_with()
                        self.assertIs(
                            layer.topk.call_args.kwargs["num_token_non_padded"], count
                        )
                    else:
                        batch.moe_num_token_non_padded.assert_not_called()
                        self.assertEqual(layer.topk.call_args.kwargs, {})

    def test_empty_rows_preserve_native_empty_topk_path(self):
        states = torch.empty(0, 8, dtype=torch.bfloat16)
        empty = object()
        topk = Mock()
        topk.empty_topk_output.return_value = empty
        layer = SimpleNamespace(
            _expertplex=True, gate=Mock(), topk=topk, experts=Mock(return_value=states)
        )
        batch = SimpleNamespace(moe_num_token_non_padded=Mock())
        result = Qwen3MoeSparseMoeBlock.forward_normal(layer, states, batch)
        self.assertEqual(result.shape, states.shape)
        layer.gate.assert_not_called()
        topk.assert_not_called()
        batch.moe_num_token_non_padded.assert_not_called()
        layer.experts.assert_called_once()
        self.assertEqual(layer.experts.call_args.args[0].shape, states.shape)
        self.assertIs(layer.experts.call_args.args[1], empty)


if __name__ == "__main__":
    unittest.main()
