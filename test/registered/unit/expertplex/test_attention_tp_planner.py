"""CPU algebra and placement coverage; artificial profiles are not GPU evidence."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import test_component_optimizer as component
import test_config_optimizer as base

o, c = base.optimizer, base.config


def fixture():
    model = c.ModelSpec.from_dict(base.model_config(coder=True))
    data = base.profiles_dict(model)
    for phase in ("prefill", "decode"):
        data[phase]["attention"]["alpha"] = 1.0  # Deliberately infeasible TP1.
        data[phase]["attention_by_tp_size"] = {
            "2": {
                "attention": dict(alpha=1e-6, beta=2e-6),
                "attention_domain": {
                    "training_points": [[1, 16], [3, 16], [1, 64], [3, 64]],
                    "max_phase_tokens": 256,
                    "max_kv_tokens": 4096,
                    "kv_reserve_per_sequence": 4,
                },
                "provenance": f"ARTIFICIAL {phase} TP2 algebra fixture",
                "includes_tp_collectives": True,
            }
        }
    node = c.default_config("/model", max_prefill_tokens=128, max_decode_tokens=8)
    opts = dict(
        input_length=16,
        context_length=64,
        average_output_length=16,
        prefill_slo_s=1.0,
        decode_slo_s=0.05,
        kv_tokens_per_attention_gpu=4096,
        attention_tp_sizes=(1, 2),
    )
    return model, data, node, opts


class AttentionTPPlannerTests(unittest.TestCase):
    def test_component_work_counts_clients_not_tp_ranks(self):
        model, data = component.fixture()
        for phase in ("prefill", "decode"):
            data[phase]["attention_by_tp_size"] = {
                "2": {
                    "attention": copy.deepcopy(data[phase]["attention"]),
                    "attention_domain": copy.deepcopy(data[phase]["attention_domain"]),
                    "provenance": "ARTIFICIAL same-cost TP2 fixture",
                    "includes_tp_collectives": True,
                }
            }
        profiles = o.MeasuredProfiles.from_dict(data, model)
        physical = o.Layout(1, 1, 4, 2, 2)
        control = o.Layout(1, 1, 4)
        for phase in ("prefill", "decode"):
            self.assertEqual(
                o.component_layer_latencies(
                    profiles, physical, phase, 3, 16, 4, "sequential"
                ),
                o.component_layer_latencies(
                    profiles, control, phase, 3, 16, 4, "sequential"
                ),
            )
            self.assertEqual(
                o._iteration_latency(
                    model, profiles, physical, phase, 3, 16, 64, 4, "sequential", True
                ),
                o._iteration_latency(
                    model, profiles, control, phase, 3, 16, 64, 4, "sequential", True
                ),
            )
        node = physical.config(
            "/model",
            prefill_preprocess=True,
            max_prefill_tokens=128,
            max_decode_tokens=32,
        )
        with patch.object(o, "enumerate_layouts", return_value=[physical]):
            result = o.search(
                node,
                model,
                profiles,
                input_length=16,
                context_length=64,
                average_output_length=16,
                prefill_slo_s=1,
                decode_slo_s=0.05,
                kv_tokens_per_attention_gpu=4096,
                attention_tp_sizes=(2,),
            )
        self.assertEqual(result.layout, physical)
        self.assertEqual(result.component_runtime_source_sha256, "0" * 64)

    def test_enumeration_matches_exhaustive_physical_gpu_assignments(self):
        widths = (1, 2, 4)
        got = o.enumerate_layouts(160, attention_tp_sizes=widths)
        expected = {
            o.Layout(p, d, e, ptp, dtp)
            for p in range(1, 8)
            for d in range(1, 8)
            for e in range(1, 8)
            for ptp in widths
            for dtp in widths
            if p * ptp + d * dtp + e == 8 and 160 % e == 0
        }
        self.assertEqual(set(got), expected)
        self.assertEqual(len(got), len(expected))
        self.assertIn(o.Layout(1, 1, 4, 2, 2), got)
        self.assertIn(o.Layout(1, 2, 4, 2, 1), got)
        for layout in got:
            node = layout.config("/model")
            devices = [
                x
                for client in node.clients
                for x in range(client.device, client.device + client.tp_size)
            ]
            self.assertEqual(
                sorted(devices + list(node.expert_devices)), list(range(8))
            )
            self.assertEqual(
                node.slot_count, layout.prefill_count + layout.decode_count
            )
        for bad in ((), (1, 1), (True,), (2.0,), (0,), (9,)):
            with self.assertRaises(ValueError):
                o.enumerate_layouts(160, attention_tp_sizes=bad)

    def test_memory_replication_and_native_vocab_padding(self):
        # Q8 / KV2; TP4 replicates each KV head across two ranks. Router/norm
        # parameters also replicate, and native vocabulary127 pads to128.
        model = c.ModelSpec(16, 32, 8, 2, 3, 8, 2, 4, 127)
        for tp, q, kv, vocab in [(1, 8, 2, 128), (2, 4, 1, 64), (4, 2, 1, 32)]:
            parameters = 3 * (
                16 * 4 * (q + 2 * kv) + q * 4 * 16 + 16 * 8 + 2 * 16 + 2 * 4
            )
            parameters += 2 * vocab * 16 + 16
            self.assertEqual(model.attention_rank_weight_bytes(tp), parameters * 2)
            self.assertEqual(
                model.attention_rank_kv_bytes_per_token(tp), 2 * kv * 4 * 3 * 2
            )
        self.assertGreater(
            model.attention_rank_weight_bytes(4), model.attention_weight_bytes / 4
        )
        self.assertEqual(
            model.attention_rank_kv_bytes_per_token(4),
            model.attention_rank_kv_bytes_per_token(2),
        )
        # Native increases the padding multiple if initially padded vocabulary
        # cannot be evenly divided by TP.
        odd = replace(
            model, num_attention_heads=6, num_key_value_heads=3, vocab_size=65
        )
        expected = 3 * (16 * 4 * (2 + 2) + 2 * 4 * 16 + 16 * 8 + 2 * 16 + 2 * 4)
        expected += 2 * 64 * 16 + 16
        self.assertEqual(odd.attention_rank_weight_bytes(3), expected * 2)
        for bad in (0, True, 1.0, 3):
            with self.assertRaises(ValueError):
                model.attention_rank_weight_bytes(bad)

    def test_memory_counts_one_mailbox_per_tp_group(self):
        model, _, _, _ = fixture()
        node = c.default_config(
            "/model",
            prefill_count=1,
            decode_count=1,
            expert_count=4,
            prefill_tp_size=2,
            decode_tp_size=2,
        )
        mem = o.estimate_memory(node, model, kv_tokens_per_attention_gpu=4096)
        self.assertEqual(
            mem.attention_weights_bytes, model.attention_rank_weight_bytes(2)
        )
        self.assertEqual(
            mem.attention_kv_bytes, 4096 * model.attention_rank_kv_bytes_per_token(2)
        )
        self.assertEqual(node.slot_count, 2)
        mixed = c.default_config(
            "/model", prefill_count=1, decode_count=2, expert_count=4, prefill_tp_size=2
        )
        mixed_mem = o.estimate_memory(mixed, model, kv_tokens_per_attention_gpu=4096)
        self.assertEqual(
            mixed_mem.attention_weights_bytes, model.attention_weight_bytes
        )
        self.assertEqual(mixed_mem.attention_kv_bytes, 4096 * model.kv_bytes_per_token)

    def test_missing_or_incomplete_tp_profile_cannot_claim_optimum(self):
        model, data, node, opts = fixture()
        for phase in ("prefill", "decode"):
            for mutate in (
                lambda x: x.pop("provenance"),
                lambda x: x.update(provenance=""),
                lambda x: x.update(includes_tp_collectives=False),
                lambda x: x.update(includes_tp_collectives=1),
                lambda x: x.pop("attention_domain"),
            ):
                bad = copy.deepcopy(data)
                mutate(bad[phase]["attention_by_tp_size"]["2"])
                with self.assertRaises(ValueError):
                    o.MeasuredProfiles.from_dict(bad, model)
            bad = copy.deepcopy(data)
            del bad[phase]["attention_by_tp_size"]
            profiles = o.MeasuredProfiles.from_dict(bad, model)
            with self.assertRaisesRegex(o.ProfileCoverageError, "TP2"):
                o.search(node, model, profiles, **opts)
        for key in ("02", "1", "9", "x", True):
            bad = copy.deepcopy(data)
            bad["prefill"]["attention_by_tp_size"][key] = bad["prefill"][
                "attention_by_tp_size"
            ].pop("2")
            with self.assertRaises(ValueError):
                o.MeasuredProfiles.from_dict(bad, model)

    def test_selection_and_export_use_actual_tp_profile(self):
        model, data, node, opts = fixture()
        profiles = o.MeasuredProfiles.from_dict(data, model)
        selected = o.search(node, model, profiles, **opts)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.layout, o.Layout(1, 1, 4, 2, 2))
        self.assertEqual((selected.prefill_batch, selected.decode_batch), (3, 3))
        self.assertEqual(selected.searched_attention_tp_sizes, (1, 2))
        self.assertEqual(
            selected.attention_profile_provenance,
            (
                "ARTIFICIAL prefill TP2 algebra fixture",
                "ARTIFICIAL decode TP2 algebra fixture",
            ),
        )
        exported = selected.config(node)
        self.assertEqual(
            [(x.device, x.tp_size) for x in exported.clients], [(0, 2), (2, 2)]
        )
        self.assertEqual(exported.expert_devices, (4, 5, 6, 7))
        self.assertEqual(exported.decode_sm_budget, selected.decode_sm_budget)
        self.assertIsNone(
            o.search(node, model, profiles, **dict(opts, attention_tp_sizes=(1,)))
        )
        with self.assertRaises(o.ProfileCoverageError):
            o._iteration_latency(
                model, profiles, selected.layout, "decode", 4, 16, 64, 4, "sequential"
            )
        # Slow measured TP2 must stay slow; the search never divides TP1 time.
        for phase in ("prefill", "decode"):
            data[phase]["attention_by_tp_size"]["2"]["attention"]["alpha"] = 1.0
        self.assertIsNone(
            o.search(node, model, o.MeasuredProfiles.from_dict(data, model), **opts)
        )

    def test_auto_overlap_preserves_tp_domain_and_collective_requirement(self):
        model, data, node, opts = fixture()
        profiles = o.MeasuredProfiles.from_dict(data, model)
        result = o.search(node, model, profiles, **opts, overlap="auto")
        self.assertEqual(result.layout, o.Layout(1, 1, 4, 2, 2))
        self.assertEqual(result.searched_attention_tp_sizes, (1, 2))
        bad = copy.deepcopy(data)
        bad["decode"]["attention_by_tp_size"]["2"]["attention"]["beta"] = -1e-6
        with self.assertRaisesRegex(ValueError, "monotone"):
            o.search(node, model, o.MeasuredProfiles.from_dict(bad, model), **opts)
        # Domain and times are selected per phase, not borrowed from the other role.
        bad = copy.deepcopy(data)
        bad["decode"]["attention_by_tp_size"]["2"]["attention_domain"][
            "training_points"
        ] = [[1, 16], [3, 16], [1, 32], [3, 32]]
        with self.assertRaisesRegex(o.ProfileCoverageError, "Sequence"):
            o.search(node, model, o.MeasuredProfiles.from_dict(bad, model), **opts)


if __name__ == "__main__":
    unittest.main()
