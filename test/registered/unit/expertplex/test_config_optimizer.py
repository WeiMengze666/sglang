"""CPU-only contract checks; these do not validate CUDA correctness or speed.

Run with: python3 -m unittest discover -s test/registered/unit/expertplex -v
Direct module loading avoids importing SGLang's optional Torch/CUDA packages.
Artificial latency data below tests algebra only; it is never shipped as a
hardware profile or used to claim B300 performance.
"""

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
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
    def test_expected_sparse_tiles_do_not_activate_every_expert(self):
        self.assertAlmostEqual(optimizer.expected_expert_tiles(1, 0.05, 16), 0.05)
        self.assertAlmostEqual(optimizer.expected_expert_tiles(3, 0.5, 16), 0.875)
        self.assertAlmostEqual(optimizer.expected_expert_tiles(3, 0.5, 2), 1.0)
        self.assertEqual(optimizer.expected_expert_tiles(33, 1.0, 16), 3.0)
        self.assertEqual(optimizer.expected_expert_tiles(33, 0.0, 16), 0.0)

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
        expected = (
            4 * 40 * 2048 * (2 * 6144 + 2560) * 2 + 4 * 40 * 4 + 4 * 64 + 4 * 32 + 4
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
        for change in (
            {"provenance": ""},
            {"nvlink_bytes_per_second": 0},
            {"units": "milliseconds"},
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
