"""CPU-only contract checks; these do not validate CUDA correctness or speed.

Run with: python3 -m unittest discover -s test/registered/unit/expertplex -v
Direct module loading avoids importing SGLang's optional Torch/CUDA packages.
Artificial latency data below tests algebra only; it is never shipped as a
hardware profile or used to claim B300 performance.
"""

import copy
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "python/sglang/srt/expertplex"
PACKAGE_NAME = "_expertplex_cpu_contract_tests"
if PACKAGE_NAME not in sys.modules:
    module = types.ModuleType(PACKAGE_NAME)
    module.__path__ = [str(PACKAGE)]
    sys.modules[PACKAGE_NAME] = module


def load_module(name):
    qualified = f"{PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


config = load_module("config")
optimizer = load_module("optimizer")
synthetic = load_module("synthetic_weights")


def model_config(*, coder=False, large=False):
    """Dimensions from the official Qwen model config.json files."""
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "torch_dtype": "bfloat16",
        "hidden_size": 6144 if coder else 4096 if large else 2048,
        "moe_intermediate_size": 2560 if coder else 1536 if large else 768,
        "num_experts": 160 if coder else 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 62 if coder else 94 if large else 48,
        "num_attention_heads": 96 if coder else 64 if large else 32,
        "num_key_value_heads": 8 if coder else 4,
        "head_dim": 128,
        "vocab_size": 151936,
    }


def profiles_dict(model):
    phase = {
        "attention": {"alpha": 1e-6, "beta": 2e-6, "gamma": 0.0, "delta": 0.0},
        "moe_by_sm_count": {
            "2": {"alpha": 1e-6, "beta": 2e-7},
            "4": {"alpha": 1e-6, "beta": 1e-7},
            "6": {"alpha": 1e-6, "beta": 5e-8},
        },
        "expert_probabilities": [model.top_k / model.num_experts] * model.num_experts,
    }
    return {
        "units": "seconds_per_layer",
        "provenance": "ARTIFICIAL UNIT TEST DATA, not a GPU measurement",
        "total_sms": 8,
        "tile_rows": 16,
        "nvlink_bytes_per_second": 1e11,
        "prefill": copy.deepcopy(phase),
        "decode": copy.deepcopy(phase),
    }


class ConfigTests(unittest.TestCase):
    def test_compact_prefill_capacity_preserves_batch_and_decode_bounds(self):
        node = config.default_config(
            "/models/coder",
            prefill_count=3,
            decode_count=1,
            expert_count=4,
            max_prefill_tokens=24576,
            max_decode_tokens=256,
            overlap="tbo",
            prefill_mailbox_tokens=12288,
        )
        self.assertEqual(node.buffer_layout()[0], [12288] * 6 + [256] * 2)
        self.assertEqual(node.max_prefill_tokens, 24576)
        model = config.ModelSpec.from_dict(model_config(coder=True))
        memory = optimizer.estimate_memory(
            node, model, kv_tokens_per_attention_gpu=196608
        )
        self.assertLess(memory.expert_total_bytes, 268 * 2**30)
        for invalid in [12287, 24577, 0, True]:
            with self.assertRaises(ValueError):
                config.NodeConfig.from_dict(
                    node.to_dict() | {"prefill_mailbox_tokens": invalid}
                )
        with self.assertRaises(ValueError):
            config.NodeConfig.from_dict(node.to_dict() | {"overlap": "none"})

    def test_transformers_canonical_expert_alias(self):
        raw = model_config(coder=True)
        canonical = {**raw, "num_local_experts": raw["num_experts"]}
        del canonical["num_experts"]
        self.assertEqual(
            config.ModelSpec.from_dict(raw), config.ModelSpec.from_dict(canonical)
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            config.ModelSpec.from_dict({**raw, "num_local_experts": 8})

    def test_tbo_mailbox_capacity(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        sequential = config.default_config("/models/coder")
        overlapped = config.default_config("/models/coder", overlap="tbo")
        self.assertEqual(overlapped.slot_count, 8)
        a = optimizer.estimate_memory(
            sequential, model, kv_tokens_per_attention_gpu=1024
        )
        b = optimizer.estimate_memory(
            overlapped, model, kv_tokens_per_attention_gpu=1024
        )
        # A single shared stop word is independent of mailbox count.
        self.assertEqual(b.expert_buffers_bytes - 4, 2 * (a.expert_buffers_bytes - 4))

    def test_preprocessing_buffers_and_backend_contract(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        base = config.default_config("/models/coder", overlap="tbo")
        raw = config.NodeConfig.from_dict(
            {**base.to_dict(), "prefill_preprocess": True}
        )
        a = optimizer.estimate_memory(base, model, kv_tokens_per_attention_gpu=1024)
        b = optimizer.estimate_memory(raw, model, kv_tokens_per_attention_gpu=1024)
        rows = 2 * base.max_prefill_tokens
        expected = (
            rows * (model.hidden_size * 4 + model.top_k * 12) + base.slot_count * 4
        )
        self.assertEqual(b.expert_buffers_bytes - a.expert_buffers_bytes, expected)
        for fields in (
            {"prefill_preprocess": 1},
            {"prefill_preprocess": True, "kernel": "wmma"},
        ):
            with self.assertRaises(ValueError):
                config.NodeConfig.from_dict({**base.to_dict(), **fields})

    def test_canonical_tensor_manifest_covers_exact_model_size(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        entries = list(synthetic.canonical_shapes(model))
        self.assertEqual(len({name for name, _ in entries}), len(entries))
        total = sum(math.prod(shape) * 2 for _, shape in entries)
        self.assertEqual(
            total, model.expert_weight_bytes + model.attention_weight_bytes
        )
        a = "model.layers.0.mlp.experts.0.gate_proj.weight"
        b = "model.layers.0.mlp.experts.1.gate_proj.weight"
        self.assertNotEqual(
            synthetic.tensor_seed(a, 1234), synthetic.tensor_seed(b, 1234)
        )
        self.assertNotEqual(
            synthetic.tensor_seed(a, 1234), synthetic.tensor_seed(a, 1235)
        )
        with self.assertRaises(ValueError):
            synthetic.tensor_seed(a, -1)

    def test_round_trip_and_mailbox_ids(self):
        original = config.default_config("/models/Qwen3-Coder-480B")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.json"
            original.write_json(path)
            self.assertEqual(config.load_config(path), original)
        self.assertEqual(original.client(0).phase, "prefill")
        self.assertEqual(original.client(3).device, 3)
        with self.assertRaises(ValueError):
            original.client(4)

    def test_invalid_partitions_and_protocol_fields_rejected(self):
        valid = config.default_config("/models/model").to_dict()
        changes = (
            {"expert_devices": [3, 5, 6, 7]},
            {"node_gpu_count": 16},
            {"max_prefill_tokens": 0},
            {"max_decode_tokens": True},
            {"startup_timeout_s": float("inf")},
            {"max_decode_sm_fraction": 1.0},
            {"rendezvous_dir": "https://host/mailboxes"},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.NodeConfig.from_dict({**valid, **change})
        for field, value in (
            ("id", 7),
            ("port", valid["clients"][0]["port"]),
            ("phase", "both"),
        ):
            invalid = copy.deepcopy(valid)
            invalid["clients"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                config.NodeConfig.from_dict(invalid)

    def test_bf16_model_adapter_validation(self):
        data = model_config(coder=True)
        spec = config.ModelSpec.from_dict(data)
        self.assertEqual(
            (spec.num_experts, spec.num_layers, spec.hidden_size), (160, 62, 6144)
        )
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(data))
            self.assertEqual(config.ModelSpec.from_model_path(directory), spec)
        for change in (
            {"torch_dtype": "float16"},
            {"quantization_config": {"quant_method": "fp8"}},
            {"mlp_only_layers": [0]},
            {"qkv_bias": True},
            {"shared_expert_intermediate_size": 512},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.ModelSpec.from_dict({**data, **change})

    def test_checkpoint_expert_count_controls_divisibility(self):
        five_experts = config.default_config(
            "/models/coder", prefill_count=1, decode_count=2, expert_count=5
        )
        five_experts.validate(config.ModelSpec.from_dict(model_config(coder=True)))
        with self.assertRaises(ValueError):
            five_experts.validate(config.ModelSpec.from_dict(model_config()))


class OptimizerTests(unittest.TestCase):
    def test_communication_payload_matches_raw_and_routed_transports(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        layout = optimizer.Layout(1, 3, 4)
        # Seven tokens: raw normal mode broadcasts four copies with IDs and
        # routing weights, then reads four partial vectors; masked mode moves
        # eight dispatched and eight returned vectors per token.
        self.assertEqual(
            optimizer.nvlink_payload_bytes(
                model, layout, "prefill", 7, prefill_preprocess=True
            ),
            689920,
        )
        for phase, preprocess in (("decode", True), ("prefill", False)):
            self.assertEqual(
                optimizer.nvlink_payload_bytes(
                    model, layout, phase, 7, prefill_preprocess=preprocess
                ),
                1376256,
            )

    def test_search_rejects_profiles_from_another_preprocessing_path(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        profiles = optimizer.MeasuredProfiles.from_dict(profiles_dict(model), model)
        node = config.default_config("/models/coder", prefill_preprocess=True)
        with self.assertRaisesRegex(ValueError, "profile prefill_preprocess"):
            optimizer.search(
                node,
                model,
                profiles,
                input_length=128,
                average_output_length=32,
                context_length=512,
                prefill_slo_s=1,
                decode_slo_s=0.05,
                kv_tokens_per_attention_gpu=8192,
            )

    def test_tbo_search_counts_double_buffers_and_exports_selected_policy(self):
        model = config.ModelSpec.from_dict(model_config())
        node = config.default_config(
            "/models/30b", max_prefill_tokens=256, max_decode_tokens=32
        )
        profiles = optimizer.MeasuredProfiles.from_dict(profiles_dict(model), model)
        original = optimizer.estimate_memory
        seen = []

        def memory(candidate, *args, **kwargs):
            seen.append(candidate.overlap)
            estimate = original(candidate, *args, **kwargs)
            # A constrained device can admit sequential but reject doubled TBO buffers.
            if candidate.overlap == "tbo":
                return type(estimate)(
                    **{**estimate.__dict__, "available_bytes_per_gpu": 0}
                )
            return estimate

        arguments = dict(
            input_length=64,
            average_output_length=16,
            context_length=128,
            prefill_slo_s=1,
            decode_slo_s=0.1,
            kv_tokens_per_attention_gpu=1024,
        )
        with patch.object(optimizer, "estimate_memory", side_effect=memory):
            self.assertIsNone(
                optimizer.search(node, model, profiles, overlap="tbo", **arguments)
            )
            result = optimizer.search(
                node, model, profiles, overlap="auto", **arguments
            )
        self.assertEqual(set(seen), {"none", "tbo"})
        self.assertEqual(result.overlap, "sequential")
        selected = result.config(node)
        self.assertEqual(len(selected.expert_devices), result.layout.expert_count)
        self.assertEqual(selected.decode_sm_budget, result.decode_sm_budget)
        self.assertEqual(selected.expected_decode_tiles, result.expected_decode_tiles)
        selected.validate(model)

    def test_tbo_partial_tiles_cannot_merge_across_mailboxes(self):
        model = config.ModelSpec.from_dict(model_config())
        data = profiles_dict(model)
        for phase in ("prefill", "decode"):
            data[phase]["expert_probabilities"] = [1.0] * 8 + [0.0] * 120
        profiles = optimizer.MeasuredProfiles.from_dict(data, model)
        layout = optimizer.Layout(1, 3, 4)
        sequential = optimizer.expected_rank_footprint(
            profiles, layout, "prefill", 1, 16, "sequential"
        )
        tbo = optimizer.expected_rank_footprint(
            profiles, layout, "prefill", 1, 16, "tbo"
        )
        self.assertEqual(sequential, 2)
        self.assertEqual(tbo, 4)

    def test_expected_sparse_tiles_do_not_activate_every_expert(self):
        self.assertAlmostEqual(optimizer.expected_expert_tiles(1, 0.05, 16), 0.05)
        self.assertAlmostEqual(optimizer.expected_expert_tiles(3, 0.5, 16), 0.875)
        self.assertAlmostEqual(optimizer.expected_expert_tiles(3, 0.5, 2), 1.0)
        self.assertEqual(optimizer.expected_expert_tiles(33, 1.0, 16), 3.0)
        self.assertEqual(optimizer.expected_expert_tiles(33, 0.0, 16), 0.0)

    def test_layer_skew_is_not_averaged_before_rank_and_critical_path_maxima(self):
        model = config.ModelSpec.from_dict(model_config() | {"num_hidden_layers": 2})
        data = profiles_dict(model)
        phase = data["decode"]
        phase["expert_probabilities"] = [0.5] * 16 + [0.0] * 112
        phase["probabilities_by_layer"] = [
            [float(e < 16 and e % 2 == rank) for e in range(128)] for rank in range(2)
        ]
        phase["attention"] = {"alpha": 6, "beta": 0}
        phase["moe_by_sm_count"]["2"] = {"alpha": 0, "beta": 1}
        profiles = optimizer.MeasuredProfiles.from_dict(data, model)
        layout = optimizer.Layout(1, 1, 2)
        self.assertEqual(
            optimizer.expected_rank_footprints(
                profiles, layout, "decode", 1, 16, "tbo"
            ),
            (8, 8),
        )
        latency = optimizer._iteration_latency(
            model, profiles, layout, "decode", 1, 16, 16, 2, "tbo"
        )
        communication = (
            optimizer.nvlink_payload_bytes(
                model, layout, "decode", 1, prefill_preprocess=False
            )
            / 1e11
        )
        self.assertAlmostEqual(latency, 16 + 2 * communication)
        del phase["probabilities_by_layer"]
        averaged = optimizer.MeasuredProfiles.from_dict(data, model)
        self.assertEqual(
            optimizer.expected_rank_footprint(averaged, layout, "decode", 1, 16, "tbo"),
            4,
        )
        self.assertEqual(
            optimizer._iteration_latency(
                model, averaged, layout, "decode", 1, 16, 16, 2, "tbo"
            ),
            12,
        )
        # Even after preserving each layer's rank maximum, TBO must take its
        # attention/MoE maximum before summing layers with different bottlenecks.
        layers = [
            [float(e < 16 and e % 2 == 0) for e in range(128)],
            [float(e < 8) for e in range(128)],
        ]
        phase["probabilities_by_layer"] = layers
        phase["expert_probabilities"] = [sum(v) / 2 for v in zip(*layers)]
        mixed = optimizer.MeasuredProfiles.from_dict(data, model)
        self.assertAlmostEqual(
            optimizer._iteration_latency(
                model, mixed, layout, "decode", 1, 16, 16, 2, "tbo"
            ),
            14 + communication,
        )

    def test_per_layer_routing_requires_complete_consistent_model_counts(self):
        model = config.ModelSpec.from_dict(model_config() | {"num_hidden_layers": 2})
        for invalid in (
            [],
            [[1.0] * 8 + [0.0] * 120],
            [[1.0] * 8 + [0.0] * 120] * 2,
            [[True] * 8 + [False] * 120] * 2,
            [[float("nan")] * 128] * 2,
        ):
            data = profiles_dict(model)
            data["decode"]["probabilities_by_layer"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                optimizer.MeasuredProfiles.from_dict(data, model)

    def test_sparse_footprint_counts_partial_tiles_per_expert(self):
        self.assertEqual(optimizer.tile_footprint([0, 1, 128, 129], 128), 4)
        self.assertEqual(optimizer.tile_footprint([1] * 8, 128), 8)
        self.assertEqual(optimizer.tile_footprint([8], 128), 1)
        with self.assertRaises(ValueError):
            optimizer.tile_footprint([-1], 128)

    def test_adaptive_allocation_progress_and_idle_lending(self):
        common = dict(
            total_sms=128,
            decode_sm_budget=32,
            expected_decode_tiles=128,
            cluster_size=2,
            max_decode_sm_fraction=0.75,
        )
        allocation = optimizer.adaptive_sm_allocation(
            **common, current_decode_tiles=129, prefill_ready=True, decode_ready=True
        )
        self.assertEqual(allocation.decode_sms, 34)
        allocation = optimizer.adaptive_sm_allocation(
            **common, current_decode_tiles=10000, prefill_ready=True, decode_ready=True
        )
        self.assertEqual(allocation, optimizer.SmAllocation(32, 96))
        self.assertEqual(
            optimizer.adaptive_sm_allocation(
                **common, current_decode_tiles=1, prefill_ready=False, decode_ready=True
            ),
            optimizer.SmAllocation(0, 128),
        )
        self.assertEqual(
            optimizer.adaptive_sm_allocation(
                **common, current_decode_tiles=0, prefill_ready=True, decode_ready=False
            ),
            optimizer.SmAllocation(128, 0),
        )

    def test_coder_480b_memory_uses_native_mailbox_shapes(self):
        model = config.ModelSpec.from_dict(model_config(coder=True))
        node = config.default_config("/models/coder")
        memory = optimizer.estimate_memory(
            node, model, kv_tokens_per_attention_gpu=65536
        )
        self.assertEqual(memory.expert_weights_bytes, 62 * 40 * 3 * 6144 * 2560 * 2)
        packed_rows = (2048 * 8 + 40 * 127 + 127) // 128 * 128
        decode_rows = 3 * 40 * node.max_decode_tokens
        expected = (
            (packed_rows + decode_rows) * (2 * 6144 + 3 * 2560) * 2
            + 4 * 40 * 8
            + 4 * 64
            + 4 * 48
            + 4
        )
        self.assertEqual(memory.expert_buffers_bytes, expected)
        self.assertTrue(memory.fits)
        self.assertAlmostEqual(memory.expert_weights_bytes / optimizer.GIB, 217.96875)
        smaller = config.default_config(
            "/models/coder", prefill_count=1, decode_count=5, expert_count=2
        )
        self.assertFalse(
            optimizer.estimate_memory(
                smaller, model, kv_tokens_per_attention_gpu=65536
            ).fits
        )

    def test_qwen_235b_two_expert_alternative_fits(self):
        model = config.ModelSpec.from_dict(model_config(large=True))
        node = config.default_config(
            "/models/235b", prefill_count=1, decode_count=5, expert_count=2
        )
        estimate = optimizer.estimate_memory(
            node, model, kv_tokens_per_attention_gpu=65536
        )
        self.assertTrue(estimate.fits)
        self.assertEqual(model.kv_bytes_per_token, 2 * 4 * 128 * 94 * 2)

    def test_layout_enumeration_respects_gpu_and_expert_counts(self):
        layouts = optimizer.enumerate_layouts(160)
        self.assertIn(optimizer.Layout(1, 3, 4), layouts)
        self.assertIn(optimizer.Layout(1, 2, 5), layouts)
        self.assertTrue(
            all(
                layout.prefill_count + layout.decode_count + layout.expert_count == 8
                and 160 % layout.expert_count == 0
                for layout in layouts
            )
        )

    def test_fit_recovers_algebra_and_rejects_rank_deficiency(self):
        coefficients = optimizer.fit_coefficients(
            [{"x": x, "latency_s": 2 + 3 * x} for x in (1, 2, 3, 5)], component="moe"
        )
        self.assertAlmostEqual(coefficients.alpha, 2)
        self.assertAlmostEqual(coefficients.beta, 3)
        samples = [
            {
                "x": x,
                "sequence_length": s,
                "latency_s": 2 + 3 * x + 4 * x * s + 5 * x * s * s,
            }
            for x, s in ((1, 1), (2, 1), (1, 2), (2, 3), (3, 4))
        ]
        coefficients = optimizer.fit_coefficients(samples, component="attention")
        for name, value in (("alpha", 2), ("beta", 3), ("gamma", 4), ("delta", 5)):
            self.assertAlmostEqual(getattr(coefficients, name), value)
        with self.assertRaises(ValueError):
            optimizer.fit_coefficients(
                [{"x": 1, "latency_s": 1}] * 4, component="attention"
            )

    def test_profile_provenance_bandwidth_routing_and_units_required(self):
        model = config.ModelSpec.from_dict(model_config())
        data = profiles_dict(model)
        self.assertEqual(
            optimizer.MeasuredProfiles.from_dict(data, model).tile_rows, 16
        )
        phase_tiles = {"prefill": 64, "decode": 32}
        self.assertEqual(
            optimizer.MeasuredProfiles.from_dict(
                {**data, "tile_rows": phase_tiles}, model
            ).tile_rows,
            phase_tiles,
        )
        for change in (
            {"provenance": ""},
            {"nvlink_bytes_per_second": 0},
            {"units": "milliseconds"},
            {"tile_rows": {"prefill": 64}},
            {"tile_rows": {"prefill": 64, "decode": 0}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                optimizer.MeasuredProfiles.from_dict({**data, **change}, model)
        data["decode"]["expert_probabilities"] = [1.0] * model.num_experts
        with self.assertRaises(ValueError):
            optimizer.MeasuredProfiles.from_dict(data, model)

    def test_search_equation_one_feasibility_and_no_feasible_batch(self):
        model = config.ModelSpec.from_dict(model_config())
        node = config.default_config(
            "/models/30b", max_prefill_tokens=256, max_decode_tokens=32
        )
        profiles = optimizer.MeasuredProfiles.from_dict(profiles_dict(model), model)
        arguments = dict(
            input_length=64,
            average_output_length=16,
            context_length=128,
            prefill_slo_s=1.0,
            decode_slo_s=0.1,
            kv_tokens_per_attention_gpu=1024,
        )
        result = optimizer.search(node, model, profiles, **arguments)
        self.assertIsNotNone(result)
        self.assertLessEqual(result.predicted_prefill_latency_s, 1.0)
        self.assertLessEqual(result.predicted_decode_latency_s, 0.1)
        self.assertLessEqual(result.decode_batch, result.layout.decode_count * 8)
        self.assertAlmostEqual(
            result.predicted_goodput_requests_per_second,
            min(
                result.prefill_batch / result.predicted_prefill_latency_s,
                result.decode_batch / (result.predicted_decode_latency_s * 16),
            ),
        )
        self.assertIsNone(
            optimizer.search(
                node, model, profiles, **{**arguments, "decode_slo_s": 1e-12}
            )
        )


if __name__ == "__main__":
    unittest.main()
