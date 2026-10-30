"""CPU dispatch contracts; GPU arithmetic is covered by separate full-model runs."""

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load_class(relative, name, namespace, methods=None):
    path = ROOT / relative
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    if methods is not None:
        node.bases = []
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class RouterMetadata(unittest.TestCase):
    def invoke(self, cuda, enabled, padding=None, location=None):
        native = Mock(return_value=object())
        reduce = Mock()
        context = Mock(return_value=types.SimpleNamespace(enable_expertplex=enabled))
        klass = load_class(
            "layers/moe/topk.py",
            "TopK",
            dict(
                _is_cuda=cuda,
                get_disagg=context,
                select_experts=native,
                rocm_router_reduce_partials=reduce,
            ),
            {"forward_native"},
        )
        obj = klass()
        obj.topk_config = types.SimpleNamespace(torch_native=False)
        obj.layer_id = 3
        obj.forward_cuda = Mock(return_value=object())
        obj._apply_waterfill = Mock(return_value=object())
        hidden = types.SimpleNamespace(shape=(1, 8))
        logits, bias, partials = object(), object(), object()
        kwargs = dict(
            num_token_non_padded=padding,
            expert_location_dispatch_info=location,
            dynamic_expert_bias=bias,
            router_logits_partials=partials,
        )
        result = obj.forward_native(hidden, logits, **kwargs)
        return obj, native, reduce, context, result, hidden, logits, kwargs

    def test_cuda_metadata_and_expertplex_retain_original_dispatch(self):
        for enabled, padding, location in [
            (True, None, None),
            (False, object(), None),
            (False, None, object()),
            (True, object(), object()),
        ]:
            with self.subTest(enabled=enabled, padding=padding, location=location):
                obj, native, reduce, _, result, hidden, logits, kw = self.invoke(
                    True, enabled, padding, location
                )
                obj.forward_cuda.assert_called_once_with(hidden, logits, **kw)
                self.assertIs(result, obj.forward_cuda.return_value)
                self.assertFalse(obj.topk_config.torch_native)
                native.assert_not_called()
                reduce.assert_not_called()
                obj._apply_waterfill.assert_not_called()

    def test_ordinary_cuda_keeps_native_compile_route_and_waterfill(self):
        obj, native, reduce, _, result, hidden, logits, kw = self.invoke(True, False)
        obj.forward_cuda.assert_not_called()
        reduce.assert_called_once_with(kw["router_logits_partials"], logits)
        native.assert_called_once_with(
            hidden_states=hidden,
            layer_id=3,
            router_logits=logits,
            topk_config=obj.topk_config,
            num_token_non_padded=None,
            expert_location_dispatch_info=None,
            dynamic_expert_bias=kw["dynamic_expert_bias"],
        )
        self.assertTrue(obj.topk_config.torch_native)
        obj._apply_waterfill.assert_called_once_with(native.return_value, 1)
        self.assertIs(result, obj._apply_waterfill.return_value)

    def test_non_cuda_does_not_consult_cuda_runtime_context(self):
        obj, native, _, context, _, _, _, _ = self.invoke(
            False, True, object(), object()
        )
        context.assert_not_called()
        obj.forward_cuda.assert_not_called()
        native.assert_called_once()


class TransportBoundary(unittest.TestCase):
    def test_torch_compile_preserves_dispatch_lifecycle_and_dense_regions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU Torch required for real Dynamo boundary test")
        calls, graphs = [], []

        class Dispatcher:
            def __init__(self, layer_id):
                self.layer_id = layer_id

            def dispatch(self, hidden, routing):
                calls.append(("dispatch", torch._dynamo.is_compiling(), routing))
                return types.SimpleNamespace(pending=(hidden, routing))

            def combine(self, combined):
                hidden, routing = combined.pending
                calls.append(("combine", torch._dynamo.is_compiling(), routing))
                return hidden * 2

        klass = load_class(
            "layers/moe/ep_moe/expertplex.py",
            "ExpertPlexMoE",
            dict(
                nn=torch.nn,
                disable=torch.compiler.disable,
                ExpertPlexDispatcher=Dispatcher,
                ExpertPlexCombineInput=lambda pending: types.SimpleNamespace(
                    pending=pending
                ),
            ),
        )
        layer = klass(layer_id=3, num_experts=4)

        def model(x, routing):
            return layer(x + 1, routing) + 3

        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward

        compiled = torch.compile(model, backend=backend, fullgraph=False)
        try:
            for i in (1, 3):
                x = torch.tensor([float(i)])
                routing = object()
                torch.testing.assert_close(
                    compiled(x, routing), (x + 1) * 2 + 3, rtol=0, atol=0
                )
                self.assertEqual([c[0] for c in calls[-2:]], ["dispatch", "combine"])
                self.assertTrue(
                    all(c[1] is False and c[2] is routing for c in calls[-2:])
                )
            self.assertGreaterEqual(len(graphs), 2)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
