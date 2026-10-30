"""Exercise Qwen3's real cache-ownership branches with CPU tensor stand-ins.

Extract only the three methods so this test needs neither CUDA extensions nor
model weights. The stand-ins test the RoPE/attention KV contract, not attention
numerics. The compile test executes the actual methods through TorchDynamo.
"""

import ast
import itertools
import unittest
from pathlib import Path

import torch


def load_methods():
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/models/qwen3_moe.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3MoeAttention"
    )
    names = {"_can_fuse_rope_kv", "apply_qk_norm_rope", "forward_core"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    ns = {
        "torch": torch,
        "enable_fused_set_kv_buffer": lambda batch: batch,
        "apply_qk_norm": lambda **kw: (kw["q"], kw["k"]),
        "create_fused_set_kv_buffer_arg": lambda **kw: (kw["layer"], kw["value"]),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), ns)
    return {name: ns[name] for name in names}


class CacheAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("key", torch.full((2, 2), float("nan")))
        self.register_buffer("value", torch.full((2, 2), float("nan")))
        self.register_buffer("writes", torch.zeros((), dtype=torch.int64))

    def save(self, key, value):
        self.key.copy_(key)
        self.value.copy_(value)
        self.writes.add_(1)

    def forward(self, q, k, v, batch, save_kv_cache):
        if save_kv_cache:
            self.save(k, v)
        # Reading the cache exposes a missing write instead of hiding stale KV.
        return q + self.key + self.value


class Rope(torch.nn.Module):
    def __init__(self, compiling):
        super().__init__()
        # BaseFusedOp.enter_torch_compile sets this before tracing and restores
        # it on leave. No GPU implementation is used by this CPU contract test.
        self.is_torch_compile = compiling

    def forward(self, positions, q, k, fused_set_kv_buffer_arg):
        q, k = q + 1, k + 1
        if self.is_torch_compile:
            assert fused_set_kv_buffer_arg is None
        if fused_set_kv_buffer_arg is not None:
            layer, value = fused_set_kv_buffer_arg
            layer.save(k, value)
        return q, k


class Harness(torch.nn.Module):
    def __init__(self, compiling=False, compatible=True):
        super().__init__()
        self.rotary_emb = Rope(compiling)
        self.attn = CacheAttention()
        self.compatible_with_fused_kv_buffer = compatible
        self.use_fused_qk_norm_rope = False
        self.use_fused_qk_norm_rope_cpu = False
        self.q_size = self.kv_size = self.head_dim = 2
        self.q_norm = self.k_norm = self.alt_stream = None

    def o_proj(self, x):
        return x, None

    def forward(self, qkv, enabled=True):
        q, k, v = self.apply_qk_norm_rope(qkv, None, enabled)
        return self.forward_core((None, enabled, (q, k, v, enabled)))


for _name, _method in load_methods().items():
    setattr(Harness, _name, _method)


class CacheOwnership(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def assert_fresh(self, model, invoke):
        for step in range(3):
            x = torch.arange(12, dtype=torch.float32).reshape(2, 6) + step * 20
            actual = invoke(x)
            self.assertTrue(torch.equal(model.attn.key, x[:, 2:4] + 1))
            self.assertTrue(torch.equal(model.attn.value, x[:, 4:6]))
            self.assertTrue(torch.equal(actual, x[:, :2] + x[:, 2:4] + x[:, 4:6] + 2))
            self.assertEqual(model.attn.writes.item(), step + 1)

    def test_each_mode_writes_once_without_stale_kv(self):
        for compiling, compatible, enabled in itertools.product(
            (False, True), repeat=3
        ):
            with self.subTest(
                compiling=compiling, compatible=compatible, enabled=enabled
            ):
                model = Harness(compiling, compatible)
                self.assert_fresh(model, lambda x: model(x, enabled))

    def test_actual_dynamo_trace_keeps_cache_updates(self):
        model = Harness(compiling=True)
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend, fullgraph=True)
        self.assert_fresh(model, compiled)
        self.assertTrue(graphs)

    def test_leaving_compile_restores_fusion(self):
        model = Harness(compiling=True)
        self.assertFalse(model._can_fuse_rope_kv(True))
        model.rotary_emb.is_torch_compile = False
        self.assertTrue(model._can_fuse_rope_kv(True))
        self.assert_fresh(model, model)

    def test_fused_qk_norm_rope_still_requires_attention_write(self):
        for compiling in (False, True):
            model = Harness(compiling)
            model._used_fused_qk_norm_rope_last_call = True
            q = torch.ones(2, 2)
            k, v = q * 2, q * 3
            result = model.forward_core((None, True, (q, k, v, True)))
            self.assertTrue(torch.equal(result, q + k + v))
            self.assertEqual(model.attn.writes.item(), 1)

    def test_empty_state_does_not_touch_cache(self):
        model = Harness()
        empty = torch.empty(0, 2)
        self.assertIs(model.forward_core((empty, True, None)), empty)
        self.assertEqual(model.attn.writes.item(), 0)


if __name__ == "__main__":
    unittest.main()
