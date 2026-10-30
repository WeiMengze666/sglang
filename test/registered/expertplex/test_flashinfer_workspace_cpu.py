"""Exercise the actual workspace configuration blocks without importing CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class FlashInferWorkspace(unittest.TestCase):
    def configure(self, initial, architecture, deterministic):
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/attention/flashinfer_backend.py"
        )
        tree = ast.parse(source.read_text())
        backend = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
        )
        constructor = next(
            n
            for n in backend.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        blocks = [
            n
            for n in constructor.body
            if isinstance(n, ast.If)
            and "SGLANG_FLASHINFER_WORKSPACE_SIZE.set" in ast.unparse(n)
        ]
        self.assertEqual(len(blocks), 2)
        state = [initial]
        variable = SimpleNamespace(
            get=lambda: state[0], set=lambda size: state.__setitem__(0, size)
        )
        scope = {
            "envs": SimpleNamespace(SGLANG_FLASHINFER_WORKSPACE_SIZE=variable),
            "model_runner": SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=[architecture])
                )
            ),
            "self": SimpleNamespace(enable_deterministic=deterministic),
            "get_int_env_var": lambda name, default: default,
        }
        exec(
            compile(ast.Module(body=blocks, type_ignores=[]), str(source), "exec"),
            scope,
        )
        return state[0], scope["self"]

    def test_large_explicit_workspace_survives_both_minimums(self):
        for arch in ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"):
            for deterministic in (False, True):
                with self.subTest(arch=arch, deterministic=deterministic):
                    value, _ = self.configure(4 << 30, arch, deterministic)
                    self.assertEqual(value, 4 << 30)

    def test_defaults_still_meet_original_minimums(self):
        self.assertEqual(
            self.configure(384 << 20, "Qwen3ForCausalLM", False)[0], 512 << 20
        )
        for arch in ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"):
            value, backend = self.configure(384 << 20, arch, True)
            self.assertEqual(value, 2 << 30)
            self.assertEqual(backend.prefill_split_tile_size, 4096)
            self.assertEqual(backend.decode_split_tile_size, 2048)
            self.assertTrue(backend.disable_cuda_graph_kv_split)
            self.assertTrue(backend.decode_use_tensor_cores)


if __name__ == "__main__":
    unittest.main()
