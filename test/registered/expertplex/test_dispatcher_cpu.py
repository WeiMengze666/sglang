"""Exercise the real BaseDispatcher hooks and TBO lifecycle with a fake transport."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.tensor  # Initialize Torch registrations before module mocks.

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        base = load(
            "sglang.srt.layers.moe.token_dispatcher.base",
            ROOT / "layers/moe/token_dispatcher/base.py",
        )
        checker = types.SimpleNamespace(
            format_is_standard=lambda value: value.format == "standard"
        )
        sys.modules["sglang.srt.layers.moe.topk"] = types.SimpleNamespace(
            TopKOutputChecker=checker
        )
        self.parallel = types.SimpleNamespace(tp_rank=0, tp_size=1)
        sys.modules["sglang.srt.runtime_context"] = types.SimpleNamespace(
            get_parallel=lambda: self.parallel
        )
        self.dispatch = load(
            "sglang.srt.layers.moe.token_dispatcher.expertplex",
            ROOT / "layers/moe/token_dispatcher/expertplex.py",
        )
        self.layer = load(
            "test_remote_moe_layer", ROOT / "layers/moe/ep_moe/expertplex.py"
        )
        self.calls = []
        self.owned = {}

        def begin(layer_id, hidden, ids, weights, mailbox=0):
            self.assertNotIn(mailbox, self.owned)
            pending = types.SimpleNamespace(output=hidden + layer_id, mailbox=mailbox)
            self.owned[mailbox] = pending
            self.calls.append(("begin", layer_id, mailbox))
            return pending

        def finish(pending):
            self.assertIs(self.owned.pop(pending.mailbox), pending)
            self.calls.append(("finish", pending.mailbox))
            return pending.output

        sys.modules["sglang.srt.expertplex.client"] = types.SimpleNamespace(
            get_client=lambda: types.SimpleNamespace(begin=begin, finish=finish)
        )
        self.topk = types.SimpleNamespace(
            format="standard", topk_ids=None, topk_weights=None
        )
        self.base = base

    def test_tp_follower_never_acquires_remote_client_and_returns_zero_partial(self):
        self.parallel.tp_rank = 1
        self.parallel.tp_size = 2
        with patch.object(
            sys.modules["sglang.srt.expertplex.client"],
            "get_client",
            side_effect=AssertionError("Follower touched transport"),
        ):
            dispatcher = self.dispatch.ExpertPlexDispatcher(layer_id=3)
            hidden = torch.randn(5, 8)
            pending = dispatcher.dispatch(hidden, self.topk).pending
            output = dispatcher.combine(self.dispatch.ExpertPlexCombineInput(pending))
            self.assertTrue(torch.equal(output, torch.zeros_like(hidden)))
            self.assertEqual(output.dtype, hidden.dtype)
            with self.assertRaisesRegex(RuntimeError, "consumed"):
                dispatcher.combine(self.dispatch.ExpertPlexCombineInput(pending))

    def test_tp_follower_tbo_preserves_distinct_pending_identity(self):
        self.parallel.tp_rank = 1
        self.parallel.tp_size = 2
        with patch.object(
            sys.modules["sglang.srt.expertplex.client"],
            "get_client",
            side_effect=AssertionError("Follower touched transport"),
        ):
            dispatcher = self.dispatch.ExpertPlexDispatcher(layer_id=3)
            for slot, rows in ((0, 5), (1, 0)):
                dispatcher.dispatch_a(torch.randn(rows, 8), self.topk, slot)
            first = dispatcher.dispatch_b(0)
            second = dispatcher.dispatch_b(1)
            with self.assertRaisesRegex(RuntimeError, "own"):
                dispatcher.combine_a(
                    self.dispatch.ExpertPlexCombineInput(second.pending), 0
                )
            for slot, result in ((1, second), (0, first)):
                dispatcher.combine_a(
                    self.dispatch.ExpertPlexCombineInput(result.pending), slot
                )
                out = dispatcher.combine_b(slot)
                self.assertEqual(out.shape, result.hidden_states.shape)
                self.assertTrue(torch.equal(out, torch.zeros_like(out)))
            self.assertEqual(dispatcher._active, {})

    def test_client_belongs_to_runtime_and_closes_once(self):
        sys.modules["sglang.srt.expertplex.config"] = types.SimpleNamespace(
            ModelSpec=None, load_config=None
        )
        sys.modules["sglang.srt.expertplex.rendezvous"] = types.SimpleNamespace(
            claim_client=None, wait_for_workers=None
        )
        client_module = load(
            "sglang.srt.expertplex.client", ROOT / "expertplex/client.py"
        )
        resources = types.SimpleNamespace(buffers={})
        sys.modules["sglang.srt.runtime_context"] = types.SimpleNamespace(
            get_resources=lambda: resources
        )
        sys.modules["sglang.srt.expertplex.integration"] = types.SimpleNamespace(
            attention_config=lambda: ("node.json", types.SimpleNamespace(id=1)),
            runtime_config=lambda: types.SimpleNamespace(expertplex_config="node.json"),
            remote_config=lambda *args: None,
        )
        closed = []

        def create(path, client_id):
            self.assertEqual((path, client_id), ("node.json", 1))
            client = object.__new__(client_module.ExpertPlexClient)
            client.lease = None
            client.streams = [
                types.SimpleNamespace(synchronize=lambda: closed.append("sync"))
            ]
            client.peers = [types.SimpleNamespace(close=lambda: closed.append("peer"))]
            return client

        original = client_module.ExpertPlexClient

        # Keep the real close method while replacing only GPU construction.
        def factory(path, client_id, **kwargs):
            with patch.object(client_module, "ExpertPlexClient", original):
                return create(path, client_id)

        with patch.object(client_module, "ExpertPlexClient", factory):
            first = client_module.get_client()
            self.assertIs(first, client_module.get_client())
            client_module.close_client()
            client_module.close_client()
            first.close()
            self.assertEqual(closed, ["sync", "peer"])
            self.assertEqual(resources.buffers, {})
            # A new runtime gets a fresh client, never the closed global instance.
            resources = types.SimpleNamespace(buffers={})
            self.assertIsNot(first, client_module.get_client())
            client_module.close_client()
            self.assertEqual(closed, ["sync", "peer", "sync", "peer"])

    def test_regular_moe_factory_layer_has_no_weights_and_keeps_hooks(self):
        layer = self.layer.ExpertPlexMoE(layer_id=3, num_experts=128)
        self.assertEqual(list(layer.parameters()), [])
        d = layer.dispatcher
        events = []
        handles = [
            d.register_pre_dispatch_hook(lambda *args: events.append("pre-dispatch")),
            d.register_post_dispatch_hook(lambda *args: events.append("post-dispatch")),
            d.register_pre_combine_hook(lambda *args: events.append("pre-combine")),
            d.register_post_combine_hook(lambda *args: events.append("post-combine")),
        ]
        x = torch.arange(8).reshape(2, 4)
        torch.testing.assert_close(layer(x, self.topk), x + 3)
        self.assertEqual(
            events, ["pre-dispatch", "post-dispatch", "pre-combine", "post-combine"]
        )
        for handle in handles:
            handle.remove()
        layer(x[:0], self.topk)
        self.assertEqual(len(events), 4)
        self.assertEqual(self.owned, {})

    def test_native_qwen_tbo_strategy_keeps_the_full_dispatch_contract(self):
        path = ROOT / "batch_overlap/operations_strategy.py"
        tree = ast.parse(path.read_text())
        names = {
            "_compute_moe_qwen3_layer_operations_strategy_tbo",
            "_compute_moe_qwen3_prefill",
            "_compute_moe_qwen3_decode",
        }
        nodes = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        nodes += [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]

        class Operations:
            _expertplex = True

            def __getattr__(self, name):
                return name

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                get_device_properties=lambda **kw: types.SimpleNamespace(
                    multi_processor_count=148
                )
            )
        )
        modes = types.SimpleNamespace(
            EXTEND="prefill", DECODE="decode", TARGET_VERIFY="verify"
        )
        ns = dict(
            torch=fake_torch,
            _is_hip=False,
            OperationsStrategy=types.SimpleNamespace,
            ForwardMode=modes,
            operations=types.SimpleNamespace(YieldOperation=lambda: "yield"),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        layer = Operations()
        layer.is_layer_sparse = True
        layer.mlp = Operations()
        layer.self_attn = Operations()
        for mode in (modes.EXTEND, modes.DECODE):
            strategy = ns["_compute_moe_qwen3_layer_operations_strategy_tbo"](
                layer, mode
            )
            chain = [
                op
                for op in strategy.operations
                if op
                in {
                    "op_dispatch_a",
                    "op_dispatch_b",
                    "op_experts",
                    "op_combine_a",
                    "op_combine_b",
                }
            ]
            self.assertEqual(
                chain,
                [
                    "op_dispatch_a",
                    "op_dispatch_b",
                    "op_experts",
                    "op_combine_a",
                    "op_combine_b",
                ],
            )
            self.assertIsNone(strategy.deep_gemm_num_sms)

    def test_two_subbatches_use_existing_dispatch_expert_combine_chain(self):
        layer = self.layer.ExpertPlexMoE(layer_id=7, num_experts=128)
        d = layer.dispatcher
        x = torch.ones(2, 4)
        d.dispatch_a(x, self.topk, tbo_subbatch_index=0)
        d.dispatch_a(x * 2, self.topk, tbo_subbatch_index=1)
        with self.assertRaisesRegex(RuntimeError, "already has a dispatch"):
            d.dispatch_a(x, self.topk, tbo_subbatch_index=0)
        a = layer.run_moe_core(d.dispatch_b(0))
        b = layer.run_moe_core(d.dispatch_b(1))
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            d.combine_a(a, 1)
        d.combine_a(b, 1)
        d.combine_a(a, 0)
        torch.testing.assert_close(d.combine_b(1), x * 2 + 7)
        torch.testing.assert_close(d.combine_b(0), x + 7)
        d.dispatch_a(x, self.topk, 0)
        d.combine_a(layer.run_moe_core(d.dispatch_b(0)), 0)
        d.combine_b(0)
        self.assertEqual(self.owned, {})


if __name__ == "__main__":
    unittest.main()
