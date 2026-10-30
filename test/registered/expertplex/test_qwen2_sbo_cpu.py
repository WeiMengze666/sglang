"""Native shared-branch control flow and placement checks, without CUDA imports.

These checks do not establish GPU overlap or numerical equivalence.
"""

import contextlib
import dataclasses
import types
import unittest

import test_integration_cpu as base


def qwen2():
    return base.CONFIG.ModelSpec.from_dict(
        dict(
            model_type="qwen2_moe",
            architectures=["Qwen2MoeForCausalLM"],
            torch_dtype="bfloat16",
            hidden_size=2048,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=1024,
            num_experts=16,
            num_experts_per_tok=4,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=1024,
        )
    )


class SharedModelTests(unittest.TestCase):
    def test_shared_parameters_reside_on_attention_ranks(self):
        spec = qwen2()
        self.assertTrue(spec.qkv_bias)
        self.assertFalse(spec.use_qk_norm)
        wider = dataclasses.replace(spec, shared_intermediate_size=2048)
        self.assertEqual(wider.expert_weight_bytes, spec.expert_weight_bytes)
        for tp in (1, 2):
            self.assertEqual(
                wider.attention_rank_weight_bytes(tp)
                - spec.attention_rank_weight_bytes(tp),
                2 * spec.num_layers * 3 * spec.hidden_size * 1024 // tp,
            )

    def test_overlap_and_dummy_scope_remain_explicit(self):
        for mode in ("none", "sbo"):
            node = base.CONFIG.default_config("/model", overlap=mode)
            node.validate(qwen2())
            self.assertEqual(node.mailboxes_per_client, 1)
        with self.assertRaisesRegex(ValueError, "TBO stage hooks"):
            base.CONFIG.default_config("/model", overlap="tbo").validate(qwen2())
        with self.assertRaisesRegex(ValueError, "Qwen3-only"):
            base.CONFIG.default_config("/model", weight_mode="synthetic").validate(
                qwen2()
            )
        q3 = dataclasses.replace(
            qwen2(),
            architecture="Qwen3MoeForCausalLM",
            shared_intermediate_size=0,
            qkv_bias=False,
            use_qk_norm=True,
        )
        with self.assertRaisesRegex(
            ValueError, "requires a model with a shared expert"
        ):
            base.CONFIG.default_config("/model", overlap="sbo").validate(q3)


class NativeSharedFlowTests(unittest.TestCase):
    def execute(self, *, expertplex, sbo, native_overlap, breakable=False, rows=2):
        events = []

        class Tensor:
            shape = (rows, 2048)
            device = "fake-cuda"

            def __init__(self, value):
                self.value = value

            def record_stream(self, stream):
                events.append("retain_shared")

            def add_(self, other):
                events.append("add_shared")
                self.value += other.value

        class Stream:
            def wait_stream(self, stream):
                events.append("shared_wait_input")

            def record_event(self):
                events.append("shared_record_done")
                return "shared_done"

            def wait_event(self, event):
                assert event == "shared_done"
                events.append("compute_wait_shared")

        alt, current = Stream(), Stream()

        @contextlib.contextmanager
        def stream_scope(stream):
            self.assertIs(stream, alt)
            events.append("enter_shared_stream")
            yield
            events.append("leave_shared_stream")

        def shared(hidden):
            events.append("shared_native_gated")
            return Tensor(2)

        def experts(**kwargs):
            events.append("remote_dispatch_combine")
            return Tensor(3)

        ns = dict(
            torch=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    current_stream=lambda: current, stream=stream_scope
                )
            ),
            envs=types.SimpleNamespace(
                SGLANG_TRACE_QWEN_MOE_DEEPEP_E2E=types.SimpleNamespace(
                    get=lambda: False
                ),
                SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP=types.SimpleNamespace(
                    get=lambda: native_overlap
                ),
            ),
            is_npu=lambda: False,
            _is_cuda=True,
            is_sbo_enabled=lambda: sbo,
            is_in_breakable_cuda_graph=lambda: breakable,
        )
        fn = base.extracted_method(
            base.SRT / "models/qwen2_moe.py",
            "Qwen2MoeSparseMoeBlock",
            "_forward_deepep",
            ns,
        )

        class TopK:
            def __call__(self, *args, **kwargs):
                events.append("native_topk")
                return "routed"

            def empty_topk_output(self, device):
                return "empty"

        block = types.SimpleNamespace(
            _expertplex=expertplex,
            alt_stream=alt,
            shared_expert=object(),
            is_nextn=True,
            gate=lambda h: ("logits", None),
            topk=TopK(),
            _forward_shared_experts=shared,
            experts=experts,
        )
        fb = types.SimpleNamespace(moe_num_token_non_padded=lambda: rows)
        result = fn(block, Tensor(1), fb)
        return events, result.value

    def test_native_gate_and_sum_preserved_in_both_expertplex_modes(self):
        for mode in (False, True):
            events, value = self.execute(
                expertplex=True, sbo=mode, native_overlap=False
            )
            self.assertEqual(value, 5)
            self.assertEqual(events.count("shared_native_gated"), 1)
            self.assertEqual(events.count("remote_dispatch_combine"), 1)
            self.assertLess(
                events.index("remote_dispatch_combine"), events.index("add_shared")
            )
            self.assertEqual("shared_wait_input" in events, mode)
            if mode:
                self.assertLess(
                    events.index("compute_wait_shared"), events.index("add_shared")
                )

    def test_native_flag_is_preserved_and_expertplex_uses_explicit_sbo(self):
        events, _ = self.execute(expertplex=False, sbo=False, native_overlap=True)
        self.assertIn("enter_shared_stream", events)
        events, _ = self.execute(expertplex=True, sbo=False, native_overlap=True)
        self.assertNotIn("enter_shared_stream", events)

    def test_breakable_and_empty_branches_do_not_claim_overlap(self):
        events, value = self.execute(
            expertplex=True, sbo=True, native_overlap=False, breakable=True
        )
        self.assertEqual(value, 5)
        self.assertNotIn("enter_shared_stream", events)
        events, value = self.execute(
            expertplex=True, sbo=True, native_overlap=False, rows=0
        )
        self.assertEqual(value, 3)
        self.assertEqual(events, ["remote_dispatch_combine"])


if __name__ == "__main__":
    unittest.main()
