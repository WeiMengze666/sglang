"""CPU algebra tests. Synthetic samples are not hardware performance evidence."""

import math
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from test_config_optimizer import optimizer


class MinimaxFitTests(unittest.TestCase):
    def samples(self):
        return [
            dict(
                x=x,
                sequence_length=s,
                latency_s=1e-5 + 2e-6 * x + 3e-8 * x * s + 4e-11 * x * s * s,
            )
            for x in (1, 2, 4)
            for s in (128, 512, 2048)
        ]

    def fit(self, samples, component="attention"):
        return optimizer.fit_coefficients(
            samples, component=component, loss="minimax_relative"
        )

    def maximum_error(self, model, samples):
        return max(
            abs(model.predict(s["x"], s.get("sequence_length", 1)) / s["latency_s"] - 1)
            for s in samples
        )

    def test_exact_attention_and_moe(self):
        samples = self.samples()
        self.assertLess(self.maximum_error(self.fit(samples), samples), 1e-8)
        samples = [dict(x=x, latency_s=2 + 3 * x) for x in (0, 1, 2, 4)]
        model = self.fit(samples, "moe")
        self.assertLess(self.maximum_error(model, samples), 1e-8)
        self.assertEqual((model.gamma, model.delta), (0, 0))

    def test_inconsistent_replicates_have_known_optimum(self):
        samples = self.samples()
        samples += [dict(samples[0], latency_s=samples[0]["latency_s"] * 2)]
        # A single prediction for y and 2y is optimal at 4y/3: error 1/3.
        self.assertAlmostEqual(
            self.maximum_error(self.fit(samples), samples), 1 / 3, places=7
        )

    def test_scale_and_order_invariance(self):
        samples = self.samples()
        samples[0]["latency_s"] *= 1.4
        first = self.fit(samples)
        scaled = [dict(s, latency_s=s["latency_s"] * 1000) for s in reversed(samples)]
        second = self.fit(scaled)
        self.assertAlmostEqual(
            self.maximum_error(first, samples),
            self.maximum_error(second, scaled),
            places=8,
        )

    def test_rank_and_invalid_measurements_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            self.fit(self.samples()[:1] * 4)
        for value in (math.nan, math.inf, 0, -1):
            samples = self.samples()
            samples[0]["latency_s"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.fit(samples)

    def test_failed_solver_rejected(self):
        with (
            patch(
                "scipy.optimize.linprog",
                return_value=SimpleNamespace(success=False, message="failed"),
            ),
            self.assertRaisesRegex(ValueError, "fit failed"),
        ):
            self.fit(self.samples())

    def test_default_unchanged_and_unknown_loss_rejected(self):
        samples = self.samples()
        default = optimizer.fit_coefficients(samples, component="attention")
        explicit = optimizer.fit_coefficients(
            samples, component="attention", loss="least_squares"
        )
        self.assertEqual(asdict(default), asdict(explicit))
        with self.assertRaises(ValueError):
            optimizer.fit_coefficients(samples, component="attention", loss="unknown")


if __name__ == "__main__":
    unittest.main()
