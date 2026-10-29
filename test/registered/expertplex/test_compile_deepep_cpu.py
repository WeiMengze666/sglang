"""A compiled bs=1 graph must retain the runner for DeepEP dispatch layouts."""

import unittest
from unittest.mock import patch

from sglang.kernels import fused_op
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import unquant


class CompileDeepEPTests(unittest.TestCase):
    def test_enter_leave_keeps_distributed_runner_and_restores_dispatch(self):
        with patch.object(
            unquant, "get_moe_runner_backend", return_value=MoeRunnerBackend.DEEP_GEMM
        ):
            method = unquant.UnquantizedFusedMoEMethod(use_deep_gemm=True)

        def runner(**kwargs):
            return kwargs["dispatch_output"]

        method._forward_method = runner
        payload = object()  # Has no topk_output, like the packed/masked format.
        with (
            patch.object(
                unquant, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            ),
            patch.object(fused_op, "get_fused_op_backend", return_value=None),
        ):
            for batch in (1, 4, 1):
                method.enter_torch_compile(batch)
                self.assertIs(method._forward_method, runner)
                self.assertIs(method._forward_method(dispatch_output=payload), payload)
                method.enter_torch_compile(batch)
                method.leave_torch_compile()
                self.assertIs(method._forward_method, runner)
        with patch.object(
            unquant, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ):
            self.assertIsNotNone(method._torch_compile_forward(1))
            self.assertIsNone(method._torch_compile_forward(4))


if __name__ == "__main__":
    unittest.main()
