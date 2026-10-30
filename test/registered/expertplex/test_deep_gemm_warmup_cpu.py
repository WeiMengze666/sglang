"""Exercise the real warmup entrypoint with CPU TorchDynamo and fake allocation."""

import ast
import contextlib
import logging
import unittest
from pathlib import Path

import torch


class WarmupBoundaryTest(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py"
        )
        tree = ast.parse(source.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_maybe_compile_deep_gemm_one_type_all"
        )
        self.calls = []
        self.graphs = []

        def warmup(**kwargs):
            self.calls.append(
                {
                    "compiling": torch.compiler.is_compiling(),
                    "key": (kwargs["n"], kwargs["k"]),
                    "temporary": torch.ones(3),
                }
            )

        self.ns = {
            "torch": torch,
            "DeepGemmKernelType": object,
            "_INITIALIZATION_DICT": {},
            "_BUILTIN_M_LIST": [1, 2],
            "_ENABLE_JIT_DEEPGEMM_PRECOMPILE": True,
            "_DO_COMPILE_ALL": True,
            "_IN_PRECOMPILE_STAGE": False,
            "_IS_FIRST_RANK_ON_NODE": False,
            "logger": logging.getLogger("warmup-test"),
            "_local_rank_compile_lock": lambda *args: contextlib.nullcontext(),
            "_compile_deep_gemm_one_type_all": warmup,
        }
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                "real-warmup-entrypoint",
                "exec",
            ),
            self.ns,
        )
        self.entry = self.ns[node.name]

        class Kernel:
            name = "BF16_MASKED"

        self.kernel = Kernel()

    def compile(self, n):
        entry, kernel = self.entry, self.kernel

        def model(x):
            entry(kernel, n, 6144, 40)
            return x * 2 + 1

        def backend(graph, _inputs):
            self.graphs.append(graph)
            return graph.forward

        return torch.compile(model, backend=backend, fullgraph=False)

    def test_warmup_eager_once_and_tensor_path_still_compiles(self):
        model = self.compile(2560)
        x = torch.arange(4, dtype=torch.float32)
        for _ in range(3):
            self.assertTrue(torch.equal(model(x), x * 2 + 1))
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.calls[0]["compiling"])
        self.assertTrue(self.graphs)
        self.assertTrue(all("ones" not in g.code for g in self.graphs))

    def test_different_shape_keeps_separate_precompile_keys(self):
        for n in (2560, 5120):
            self.compile(n)(torch.ones(4))
        self.assertEqual([r["key"] for r in self.calls], [(2560, 6144), (5120, 6144)])
        self.assertTrue(all(not r["compiling"] for r in self.calls))

    def test_disabled_precompile_preserves_compute(self):
        self.ns["_ENABLE_JIT_DEEPGEMM_PRECOMPILE"] = False
        self.assertTrue(torch.equal(self.compile(2560)(torch.zeros(4)), torch.ones(4)))
        self.assertEqual(self.calls, [])
        self.assertTrue(self.graphs)


if __name__ == "__main__":
    unittest.main()
