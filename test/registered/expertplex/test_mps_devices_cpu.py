"""MPS clients must use the selected CUDA-visible GPU's UUID."""

import importlib
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
canonical = importlib.import_module("bench_mps").canonical_gpu_uuid


class MPSDeviceTests(unittest.TestCase):
    def test_cuda_uuid_representations(self):
        identity = uuid.UUID("3d9a718a-6bf0-4d9b-b65d-553f7d938dac")
        expected = "GPU-" + str(identity)
        for value in (identity, str(identity), expected, identity.bytes):
            self.assertEqual(canonical(value), expected)

    def test_rejects_ordinals_and_masks_as_identities(self):
        for value in (
            0,
            "1",
            "0,1,2,3",
            "GPU-",
            "MIG-3d9a718a-6bf0-4d9b-b65d-553f7d938dac",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical(value)


if __name__ == "__main__":
    unittest.main()
