"""CPU contracts for deterministic Inductor configuration and mm.out buffers.

The matrix tests use CPU Torch to check buffer semantics, not GPU arithmetic.
"""

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def extract(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"),
        namespace,
    )
    return namespace[name]


class CompileConfiguration(unittest.TestCase):
    def configure(self, invariant, initial_scale):
        torch = types.ModuleType("torch")
        torch._dynamo = types.ModuleType("torch._dynamo")
        torch._inductor = types.ModuleType("torch._inductor")
        torch._dynamo.config = types.ModuleType("torch._dynamo.config")
        torch._inductor.config = types.ModuleType("torch._inductor.config")
        config = torch._inductor.config
        config.dynamic_scale_rblock = initial_scale
        config.triton = types.SimpleNamespace()
        patched = []
        namespace = dict(
            get_bool_env_var=lambda _: invariant,
            monkey_patch_torch_compile=lambda: patched.append(True),
        )
        fn = extract(
            "compilation/torch_compile_decoration.py",
            "set_torch_compile_config",
            namespace,
        )
        modules = {
            m.__name__: m
            for m in (
                torch,
                torch._dynamo,
                torch._inductor,
                torch._dynamo.config,
                config,
            )
        }
        with patch.dict(sys.modules, modules):
            fn()
        self.assertEqual(patched, [True])
        self.assertTrue(config.fx_graph_cache)
        return config

    def test_ordinary_mode_preserves_native_tuning(self):
        for scale in (True, False):
            c = self.configure(False, scale)
            self.assertTrue(c.coordinate_descent_tuning)
            self.assertIs(c.dynamic_scale_rblock, scale)

    def test_invariant_mode_fixes_reduction_geometry(self):
        c = self.configure(True, True)
        self.assertFalse(c.coordinate_descent_tuning)
        self.assertFalse(c.dynamic_scale_rblock)


class MmOut(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU Torch is required for matrix buffer tests")
        self.torch = torch
        self.calls = []

        def matmul(a, b):
            self.calls.append((a, b))
            return torch.mm(a, b)

        self.mm = extract(
            "batch_invariant_ops/batch_invariant_ops.py",
            "mm_out_batch_invariant",
            dict(matmul_persistent=matmul),
        )
        self.a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.b = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    def test_noncontiguous_destination_identity_and_sentinels(self):
        storage = self.torch.full((3, 10), -999.0)
        out = storage[:, ::2]
        pointer = out.data_ptr()
        self.assertIs(self.mm(self.a, self.b, out=out), out)
        self.assertEqual(out.data_ptr(), pointer)
        self.assertTrue(self.torch.equal(out, self.a @ self.b))
        self.assertTrue(self.torch.all(storage[:, 1::2] == -999).item())
        self.assertEqual(len(self.calls), 1)

    def test_empty_destination_resized_and_input_alias_preserved(self):
        out = self.torch.empty(0)
        self.assertIs(self.mm(self.a, self.b, out=out), out)
        self.assertEqual(tuple(out.shape), (3, 5))
        matrix = self.torch.arange(9, dtype=self.torch.float32).reshape(3, 3)
        right = self.torch.full((3, 3), 2.0)
        expected = matrix @ right
        self.assertIs(self.mm(matrix, right, out=matrix), matrix)
        self.assertTrue(self.torch.equal(matrix, expected))

    def test_invalid_destination_rejected_before_gemm(self):
        for out in (
            self.torch.empty(3, 5, dtype=self.torch.float64),
            self.torch.empty(2, 5),
        ):
            with self.assertRaises(ValueError):
                self.mm(self.a, self.b, out=out)
        with self.assertRaises(ValueError):
            self.mm(self.a, self.b[:2], out=self.torch.empty(3, 5))
        self.assertFalse(self.calls)

    def test_out_overload_is_registered(self):
        tree = ast.parse(
            (ROOT / "batch_invariant_ops/batch_invariant_ops.py").read_text()
        )
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "impl"
            and len(n.args) >= 2
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "aten::mm.out"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[1]), "mm_out_batch_invariant")


if __name__ == "__main__":
    unittest.main()
