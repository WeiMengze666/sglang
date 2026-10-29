"""Activation benchmark timing boundary; GPU arithmetic is checked separately."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "activation_benchmark_workload", ROOT / "examples/expertplex/bench_sharing.py"
)
sharing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sharing)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class ActivationStageTests(unittest.TestCase):
    def test_only_activation_is_repeated_after_gateup_preparation(self):
        for phase in (0, 1):
            with self.subTest(phase=phase):
                self.check_phase(phase)

    def check_phase(self, phase):
        import torch

        gemm = types.ModuleType("deep_gemm")

        def prepare(a, weights, output, *unused):
            # Distinct gate/up values make an accidental alias observable.
            output[..., :64].fill_(0.5)
            output[..., 64:].fill_(2.0)

        gemm.m_grouped_bf16_gemm_nt_contiguous = Mock(side_effect=prepare)
        gemm.m_grouped_bf16_gemm_nt_masked = Mock(side_effect=prepare)
        activation = types.ModuleType("sglang.kernels.ops.activation.activation")
        masked = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")

        def apply(x, out, *unused):
            gate, up = x.float().chunk(2, dim=-1)
            out.copy_(torch.nn.functional.silu(gate) * up)

        activation.silu_and_mul = Mock(side_effect=apply)
        masked.silu_and_mul_masked_fwd = Mock(side_effect=apply)
        modules = {
            "deep_gemm": gemm,
            activation.__name__: activation,
            masked.__name__: masked,
        }
        original_empty, original_zeros = torch.empty, torch.zeros
        original_full, original_tensor = torch.full, torch.tensor

        def cpu_factory(factory):
            def allocate(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return factory(*args, **kwargs)

            return allocate

        args = SimpleNamespace(
            hidden=64,
            intermediate=64,
            experts=4,
            world=4,
            top_k=2,
            operation="activation",
            device="cpu",
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self),
            patch.object(torch, "empty", cpu_factory(original_empty)),
            patch.object(torch, "zeros", cpu_factory(original_zeros)),
            patch.object(torch, "full", cpu_factory(original_full)),
            patch.object(torch, "tensor", cpu_factory(original_tensor)),
        ):
            invoke, output, counts, buffers = sharing.workload(
                args, 16, phase, object(), object(), return_buffers=True
            )
            expected_gemm = (
                gemm.m_grouped_bf16_gemm_nt_masked
                if phase
                else gemm.m_grouped_bf16_gemm_nt_contiguous
            )
            other_gemm = (
                gemm.m_grouped_bf16_gemm_nt_contiguous
                if phase
                else gemm.m_grouped_bf16_gemm_nt_masked
            )
            expected_activation = (
                masked.silu_and_mul_masked_fwd if phase else activation.silu_and_mul
            )
            expected_gemm.assert_called_once()
            other_gemm.assert_not_called()
            expected_activation.assert_not_called()
            self.assertIs(output, buffers["activation"])
            self.assertTrue(sum(counts) > 0)
            invoke()
            first = output.clone()
            # The repeated closure must read current pre-staged GateUp values,
            # not recompute GEMM or reuse the result of the previous invocation.
            buffers["gateup"].fill_(1.0)
            invoke()
            expected_gemm.assert_called_once()
            self.assertEqual(expected_activation.call_count, 2)
            self.assertFalse(torch.equal(first, output))
            self.assertTrue(
                torch.all(
                    output
                    == torch.nn.functional.silu(torch.tensor(1.0)).to(output.dtype)
                )
            )


if __name__ == "__main__":
    unittest.main()
