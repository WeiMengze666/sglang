"""LL sweeps must allocate queue depth for the largest case before NVSHMEM init."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
depth = importlib.import_module("bench_communication").low_latency_qp_depth


class CommunicationConfigurationTests(unittest.TestCase):
    def test_maximum_sweep_case_and_completion_slots(self):
        self.assertEqual(depth([16, 64, 256]), 1024)
        self.assertEqual(depth([16, 512, 64]), 2048)
        self.assertEqual(depth([16, 64, 256, 1024]), 4096)

    def test_preserve_larger_user_depth_and_round_up(self):
        self.assertEqual(depth([1024], 8192), 8192)
        self.assertEqual(depth([16], 1500), 2048)

    def test_invalid_inputs_fail_before_starting_workers(self):
        for rows, configured in [
            ([], 1024),
            ([0], 1024),
            ([True], 1024),
            ([-1], 1024),
            ([1], 0),
            ([1], True),
        ]:
            with (
                self.subTest(rows=rows, configured=configured),
                self.assertRaises(ValueError),
            ):
                depth(rows, configured)


if __name__ == "__main__":
    unittest.main()
