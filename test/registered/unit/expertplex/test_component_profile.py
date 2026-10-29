"""CPU algebra/coverage tests. Artificial coefficients are not hardware profiles."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/expertplex/component_profile.py"
)
SPEC = importlib.util.spec_from_file_location("_component_profile_cpu", PATH)
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


def geometry():
    return profile.ComponentGeometry(6144, 2560, 2, 2, 1, 2, 128, 128, 32)


def fixture():
    models = {}
    for phase, stages in profile.STAGES.items():
        models[phase] = {
            sms: {
                **{
                    stage: profile.ComponentLine(1 / sms, 2 / sms, 1, 100)
                    for stage in stages
                },
                "gap": profile.ComponentLine(4 / sms, 0, 0, 0),
            }
            for sms in (32, 64)
        }
    return profile.ComponentProfiles(geometry(), models), models


class ComponentProfileTests(unittest.TestCase):
    def test_separate_activation_geometry(self):
        a = geometry().features("decode", 64, [32, 32])
        b = geometry().features("decode", 64, [33, 31])
        self.assertEqual(a["1"], b["1"])
        self.assertEqual((a["3"], b["3"]), (2, 3))

    def test_raw_data_work_survives_empty_rank(self):
        p, _ = fixture()
        d = p.predict_components("decode", 48, 8, [0, 0])
        self.assertEqual([d[k] for k in ("1", "2", "3")], [0, 0, 0])
        self.assertAlmostEqual(d["gap"], 4 / 48)
        q = p.predict_components("prefill", 48, 8, [0, 0])
        self.assertTrue(all(q[k] > 0 for k in ("5", "6", "7", "gap")))

    def test_inverse_sm_interpolation(self):
        p, _ = fixture()
        result = p.predict_components("decode", 48, 64, [33, 31])
        self.assertAlmostEqual(result["1"], 5 / 48)
        self.assertAlmostEqual(result["3"], 7 / 48)
        self.assertEqual(p.bracket("decode", 32), (32, 32))

    def test_independent_mailboxes_keep_tiles_and_intercepts(self):
        p, _ = fixture()
        separate = p.rank_latencies("decode", 32, [(1, [[1, 0], [0, 0]])] * 2)
        merged = p.rank_latencies("decode", 32, [(2, [[2, 0], [0, 0]])])
        self.assertAlmostEqual(separate[0], 2 * merged[0])
        self.assertAlmostEqual(separate[1], 2 * merged[1])

    def test_rank_aggregation_does_not_move_each_clients_slowest_rank(self):
        p, _ = fixture()
        left = (1, [[1, 0], [0, 0]])
        right = (1, [[0, 0], [1, 0]])
        both = p.rank_latencies("decode", 32, [left, right])
        wrong = max(p.rank_latencies("decode", 32, [left])) + max(
            p.rank_latencies("decode", 32, [right])
        )
        self.assertLess(max(both), wrong)
        self.assertEqual(both[0], both[1])

    def test_extrapolation_and_malformed_input_rejected(self):
        p, _ = fixture()
        for sms in (16, 96):
            with self.assertRaises(profile.ProfileCoverageError):
                p.predict_components("decode", sms, 1, [1, 0])
        for sms in (0, 33, True):
            with self.assertRaises(ValueError):
                p.predict_components("decode", sms, 1, [1, 0])
        for tokens, counts in [
            (0, [0, 0]),
            (1, [2, 0]),
            (1, [True, 0]),
            (1, [1]),
            (1, [1, 1]),
        ]:
            with self.assertRaises(ValueError):
                p.predict_components("decode", 32, tokens, counts)
        with self.assertRaises(profile.ProfileCoverageError):
            p.predict_components("prefill", 32, 100000, [0, 0])
        with self.assertRaises(ValueError):
            p.rank_latencies("decode", 32, [(1, [[0, 0], [0, 0]])])

    def test_invalid_fit_and_schema_rejected(self):
        for values in [
            (math.nan, 1, 0, 1),
            (1, -1, 0, 1),
            (1, 1, 2, 1),
            (True, 1, 0, 1),
        ]:
            with self.assertRaises(ValueError):
                profile.ComponentLine(*values)
        _, models = fixture()
        del models["prefill"][32]["7"]
        with self.assertRaises(ValueError):
            profile.ComponentProfiles(geometry(), models)

    def test_original_mappings_cannot_change_predictions(self):
        p, models = fixture()
        before = p.predict_components("decode", 48, 1, [1, 0])
        models["decode"][32].clear()
        self.assertEqual(before, p.predict_components("decode", 48, 1, [1, 0]))
        with self.assertRaises(TypeError):
            p.models["decode"][32]["1"] = profile.ComponentLine(1, 1, 1, 2)


if __name__ == "__main__":
    unittest.main()
