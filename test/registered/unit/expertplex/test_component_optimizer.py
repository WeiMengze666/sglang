"""Component planner algebra/coverage, using artificial coefficients only."""

import copy
import math
import unittest
from unittest.mock import patch

import test_config_optimizer as base

optimizer = base.optimizer
config = base.config


def fixture():
    model = config.ModelSpec.from_dict(base.model_config())
    data = base.profiles_dict(model)
    data.update(prefill_preprocess=True, tile_rows=128)
    geometry = dict(
        hidden=model.hidden_size,
        intermediate=model.intermediate_size,
        experts=model.num_experts // 4,
        world=4,
        top_k=model.top_k,
        cluster_size=2,
        prefill_gemm_tile_rows=128,
        decode_gemm_tile_rows=128,
        activation_tile_rows=32,
    )
    models = {}
    stages = {"prefill": ("5", "6", "1", "3", "2", "7"), "decode": ("1", "3", "2")}
    for phase in stages:
        data[phase].pop("moe_by_sm_count")
        data[phase]["expert_probabilities"] = [
            float(i < 8) for i in range(model.num_experts)
        ]
        models[phase] = {
            str(sms): {
                **{
                    stage: {
                        "alpha": 1e-6 / sms,
                        "beta": 2e-6 / sms,
                        "x_range": [0, 10000],
                    }
                    for stage in stages[phase]
                },
                "gap": {"alpha": 4e-6 / sms, "beta": 0, "x_range": [0, 0]},
            }
            for sms in (2, 6)
        }
    data["moe_components_by_expert_count"] = {
        "4": {
            "units": "seconds_per_layer",
            "provenance": "ARTIFICIAL CPU DATA",
            "runtime_source_sha256": "0" * 64,
            "geometry": geometry,
            "models": models,
        }
    }
    return model, data


class ComponentOptimizerTests(unittest.TestCase):
    def test_component_parser_rejects_other_model_and_mixed_profiles(self):
        model, data = fixture()
        parsed = optimizer.MeasuredProfiles.from_dict(data, model)
        self.assertEqual(parsed.moe, {})
        self.assertEqual(set(parsed.components_by_world), {4})
        for change in (
            lambda d: d.update(prefill_preprocess=False),
            lambda d: d.update(tile_rows=16),
            lambda d: d["decode"].update(
                moe_by_sm_count={"2": {"alpha": 1, "beta": 1}}
            ),
            lambda d: d["moe_components_by_expert_count"]["4"]["geometry"].update(
                hidden=123
            ),
            lambda d: d["moe_components_by_expert_count"]["4"].update(
                runtime_source_sha256=""
            ),
        ):
            bad = copy.deepcopy(data)
            change(bad)
            with self.assertRaises(ValueError):
                optimizer.MeasuredProfiles.from_dict(bad, model)

    def test_components_keep_per_client_intercepts_and_activation_tiles(self):
        model, data = fixture()
        profiles = optimizer.MeasuredProfiles.from_dict(data, model)
        layout = optimizer.Layout(1, 3, 4)
        actual = optimizer.component_layer_latencies(
            profiles, layout, "decode", 3, 1, 4, "sequential"
        )
        cp = profiles.components_by_world[4]
        counts = [1, 1] + [0] * (cp.geometry.experts - 2)
        expected = 3 * sum(cp.predict_components("decode", 4, 1, counts).values())
        self.assertAlmostEqual(actual[0], expected)
        # At 33 rows each active expert still has one GEMM tile, but two ACT tiles.
        larger = optimizer.component_layer_latencies(
            profiles, layout, "decode", 99, 1, 4, "sequential"
        )
        self.assertGreater(larger[0], actual[0])

    def test_binary_search_never_accepts_unmeasured_batches(self):
        for lower in range(1, 9):
            for upper in range(lower, 12):
                for slo in (0.5, 2, 6, 20):

                    def latency(batch):
                        if batch < lower or batch > upper:
                            raise optimizer.ProfileCoverageError(
                                "domain", below=batch < lower, above=batch > upper
                            )
                        return float(batch)

                    expected = max(
                        [0] + [b for b in range(lower, upper + 1) if b <= slo]
                    )
                    self.assertEqual(
                        optimizer.max_feasible_batch(15, slo, latency), expected
                    )
        with self.assertRaises(optimizer.ProfileCoverageError):
            optimizer.max_feasible_batch(
                10,
                10,
                lambda _: (_ for _ in ()).throw(
                    optimizer.ProfileCoverageError("missing SM")
                ),
            )
        with self.assertRaises(ValueError):
            optimizer.max_feasible_batch(10, 10, lambda _: math.nan)

    def test_search_uses_component_cost_and_fails_on_unprofiled_feasible_layout(self):
        model, data = fixture()
        profiles = optimizer.MeasuredProfiles.from_dict(data, model)
        node = config.default_config(
            "/model",
            prefill_preprocess=True,
            max_prefill_tokens=128,
            max_decode_tokens=32,
        )
        kwargs = dict(
            input_length=16,
            average_output_length=16,
            context_length=64,
            prefill_slo_s=1,
            decode_slo_s=0.05,
            kv_tokens_per_attention_gpu=4096,
        )
        with patch.object(
            optimizer, "enumerate_layouts", return_value=[optimizer.Layout(1, 3, 4)]
        ):
            result = optimizer.search(node, model, profiles, **kwargs)
        self.assertIsNotNone(result)
        self.assertEqual(result.layout.expert_count, 4)
        for phase, batch, sms, value in (
            (
                "prefill",
                result.prefill_batch,
                profiles.total_sms - result.decode_sm_budget,
                result.predicted_prefill_latency_s,
            ),
            (
                "decode",
                result.decode_batch,
                result.decode_sm_budget,
                result.predicted_decode_latency_s,
            ),
        ):
            self.assertEqual(
                value,
                optimizer._iteration_latency(
                    model,
                    profiles,
                    result.layout,
                    phase,
                    batch,
                    16,
                    64,
                    sms,
                    "sequential",
                    True,
                ),
            )
        with (
            patch.object(
                optimizer, "enumerate_layouts", return_value=[optimizer.Layout(1, 5, 2)]
            ),
            patch.object(
                optimizer,
                "estimate_memory",
                return_value=type("Memory", (), {"fits": True})(),
            ),
        ):
            with self.assertRaisesRegex(
                optimizer.ProfileCoverageError, "global optimum"
            ):
                optimizer.search(node, model, profiles, **kwargs)


if __name__ == "__main__":
    unittest.main()
