"""Granularity-eight layout must preserve checkpoint projection and expert IDs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.expertplex.worker import load_expert_weights
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInterleavedExpertLoading(CustomTestCase):
    def test_checkpoint_projection_and_rank_mapping(self):
        spec = SimpleNamespace(
            num_layers=2, num_experts=4, intermediate_size=16, hidden_size=8
        )
        shards = [{}, {}]
        weights = {}
        index = {}
        for layer in range(2):
            for expert in range(4):
                for projection, offset in enumerate((1, 31, 61)):
                    name = ("gate_proj", "up_proj", "down_proj")[projection]
                    shape = (8, 16) if projection == 2 else (16, 8)
                    tensor = (
                        torch.arange(128).reshape(shape) % 13
                        + layer * 3
                        + expert * 7
                        + offset
                    ).to(torch.bfloat16)
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{name}.weight"
                    shards[expert % 2][key] = tensor
                    weights[layer, expert, projection] = tensor
                    index[key] = f"part{expert % 2}.safetensors"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rank, shard in enumerate(shards):
                save_file(shard, str(root / f"part{rank}.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": index})
            )
            # Loading CPU tensors needs no GPU barrier; retain real file loading
            # and the actual strided copies whose layout is under test.
            with patch.object(torch.cuda, "synchronize"):
                for rank in range(2):
                    gate_up, down = load_expert_weights(
                        root, spec, rank, 2, "cpu", interleaved_swiglu=True
                    )
                    for layer in range(2):
                        for local in range(2):
                            expert = local * 2 + rank
                            gate = weights[layer, expert, 0]
                            up = weights[layer, expert, 1]
                            # Independent block concatenation of the upstream
                            # MegaMoE BF16 contract: G[0:8], U[0:8], G[8:16], U[8:16].
                            expected = torch.cat(
                                [gate[:8], up[:8], gate[8:], up[8:]], dim=0
                            )
                            torch.testing.assert_close(
                                gate_up[layer, local], expected, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                down[layer, local],
                                weights[layer, expert, 2],
                                rtol=0,
                                atol=0,
                            )
                    self.assertTrue(gate_up.is_contiguous())


if __name__ == "__main__":
    unittest.main()
