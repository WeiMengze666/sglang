"""Regression for Eq. 4 on an expert shard with zero Decode row tiles."""

import unittest

from sglang.srt.expertplex.optimizer import SmAllocation, adaptive_sm_allocation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAdaptiveAllocation(CustomTestCase):
    def test_zero_decode_footprint_gives_prefill_all_clusters(self):
        # A Decode request can route every token to other expert shards.
        # Eq. 4 gives this shard a zero Decode budget even when both phase
        # readiness flags are set. Positive-footprint cluster rounding must
        # preserve that zero boundary, as the GPU budget helper does.
        for total_sms, cluster_size in [(128, 1), (128, 4), (148, 2)]:
            with self.subTest(total_sms=total_sms, cluster_size=cluster_size):
                allocation = adaptive_sm_allocation(
                    total_sms=total_sms,
                    decode_sm_budget=96,
                    expected_decode_tiles=8,
                    current_decode_tiles=0,
                    prefill_ready=True,
                    decode_ready=True,
                    cluster_size=cluster_size,
                    max_decode_sm_fraction=0.99,
                )
                self.assertEqual(allocation, SmAllocation(total_sms, 0))


if __name__ == "__main__":
    unittest.main()
