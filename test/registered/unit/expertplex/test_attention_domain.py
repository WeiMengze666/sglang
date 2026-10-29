"""Attention interpolation boundaries; artificial shapes are not GPU evidence."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import test_config_optimizer as base
import test_component_optimizer as component

optimizer = base.optimizer


def domain(phase="decode", points=None, phase_tokens=100000, kv_tokens=1000000):
    return optimizer.AttentionDomain.from_dict(
        {
            "training_points": points or [[1, 128], [8, 128], [1, 4096], [2, 4096]],
            "max_phase_tokens": phase_tokens,
            "max_kv_tokens": kv_tokens,
            "kv_reserve_per_sequence": 4,
        },
        phase,
    )


class AttentionDomainTests(unittest.TestCase):
    def test_hull_is_stricter_than_bounding_box(self):
        d = domain()
        self.assertEqual(d.batch_bounds(128), (1, 8))
        self.assertEqual(d.batch_bounds(1024), (1, 6))
        self.assertEqual(d.batch_bounds(4096), (1, 2))
        d.validate(6, 1024)
        for batch, below, above in [(0, None, None), (7, False, True)]:
            if batch == 0:
                with self.assertRaises(ValueError):
                    d.validate(batch, 1024)
            else:
                with self.assertRaises(optimizer.ProfileCoverageError) as c:
                    d.validate(batch, 1024)
                self.assertEqual((c.exception.below, c.exception.above), (below, above))
        # Adding interior samples or reordering rows cannot expand support.
        augmented = domain(points=[[2, 1024], [2, 4096], [1, 4096], [1, 128], [8, 128]])
        self.assertEqual(d.hull(), augmented.hull())

    def test_capacity_intersection_and_empty_intervals(self):
        # Convex hull alone admits (3,3), but phase tokens =8 and KV =12 do not.
        points = [[1, 1], [8, 1], [1, 4], [2, 4]]
        p = domain("prefill", points, phase_tokens=8)
        self.assertEqual(p.batch_bounds(3), (1, 2))
        d = optimizer.AttentionDomain("decode", tuple(map(tuple, points)), 8, 8, 0)
        self.assertEqual(d.batch_bounds(3), (1, 2))
        # All three training points fit. Between them there is no feasible integer batch.
        empty = optimizer.AttentionDomain(
            "prefill", ((2, 4), (4, 2), (6, 1)), 8, 100, 0
        )
        lo, hi = empty.batch_bounds(3)
        self.assertGreater(lo, hi)
        self.assertEqual(
            optimizer.max_feasible_batch(20, 1, lambda b: empty.validate(b, 3) or 0.1),
            0,
        )

    def test_sequence_cannot_be_repaired_by_batch_search(self):
        d = domain()
        for sequence in (127, 4097):
            with self.assertRaises(optimizer.ProfileCoverageError) as c:
                optimizer.max_feasible_batch(
                    20, 1, lambda b: d.validate(b, sequence) or 0.1
                )
            self.assertFalse(c.exception.below or c.exception.above)
        for sequence in (0, True, 128.0):
            with self.assertRaises(ValueError):
                d.batch_bounds(sequence)

    def test_batch_search_uses_only_supported_integer_shapes(self):
        d = domain(points=[[3, 128], [8, 128], [3, 2048], [4, 2048]])
        for sequence in (128, 511, 1024, 2048):
            low, high = d.batch_bounds(sequence)
            for slo in (1, 3, 4, 7, 20):
                expected = max([0] + [b for b in range(low, high + 1) if b <= slo])
                actual = optimizer.max_feasible_batch(
                    32, slo, lambda b: d.validate(b, sequence) or float(b)
                )
                self.assertEqual(actual, expected)

    def test_parser_rejects_missing_and_invalid_domain(self):
        model, data = component.fixture()
        for phase in ("prefill", "decode"):
            bad = copy.deepcopy(data)
            del bad[phase]["attention_domain"]
            with self.assertRaisesRegex(ValueError, "attention_domain"):
                optimizer.MeasuredProfiles.from_dict(bad, model)
        valid = data["decode"]["attention_domain"]
        for patcher in [
            lambda d: d.update(training_points=[[1, 1], [2, 2], [3, 3]]),
            lambda d: d.update(training_points=[[1, 1], [2, 1], [1, 1]]),
            lambda d: d.update(training_points=[[True, 1], [2, 1], [1, 2]]),
            lambda d: d.update(training_points=["x", "y", "z"]),
            lambda d: d.update(max_phase_tokens=1),
            lambda d: d.update(max_kv_tokens=False),
            lambda d: d.update(kv_reserve_per_sequence=-1),
            lambda d: d.update(unknown_field=1),
        ]:
            bad = copy.deepcopy(valid)
            patcher(bad)
            with self.assertRaises(ValueError):
                optimizer.AttentionDomain.from_dict(bad, "decode")

    def test_real_planner_applies_local_attention_domain(self):
        model, data = component.fixture()
        for phase in ("prefill", "decode"):
            data[phase]["attention_domain"] = {
                "training_points": [[1, 16], [4, 16], [1, 64], [4, 64]],
                "max_phase_tokens": 512,
                "max_kv_tokens": 4096,
                "kv_reserve_per_sequence": 4,
            }
        profiles = optimizer.MeasuredProfiles.from_dict(data, model)
        node = base.config.default_config(
            "/model",
            prefill_preprocess=True,
            max_prefill_tokens=128,
            max_decode_tokens=32,
        )
        layout = optimizer.Layout(1, 3, 4)
        options = dict(
            input_length=16,
            average_output_length=16,
            context_length=64,
            prefill_slo_s=1,
            decode_slo_s=0.05,
            kv_tokens_per_attention_gpu=4096,
        )
        with patch.object(optimizer, "enumerate_layouts", return_value=[layout]):
            result = optimizer.search(node, model, profiles, **options)
        self.assertEqual(result.prefill_batch, 4)
        self.assertEqual(result.decode_batch, 12)  # Three D, each covered up to4.
        with self.assertRaises(optimizer.ProfileCoverageError):
            optimizer._iteration_latency(
                model, profiles, layout, "decode", 13, 16, 64, 4, "sequential", True
            )
        with self.assertRaises(optimizer.ProfileCoverageError):
            optimizer.search(
                node, model, replace(profiles, attention_domains=None), **options
            )


if __name__ == "__main__":
    unittest.main()
