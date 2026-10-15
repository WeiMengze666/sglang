"""Exercise the serialized diagnostic with real CPU Torch tensors."""

import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
incremental = importlib.import_module("check_incremental")
AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "dill", "orjson"))


@contextmanager
def processor_class():
    # Import the real processor API without SGLang's unrelated public API/CUDA
    # dependencies. All replacements are scoped to this CPU test.
    name = "sglang.srt.sampling.custom_logit_processor"
    source = ROOT / "python/sglang/srt/sampling/custom_logit_processor.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    packages = {}
    for package in ("sglang", "sglang.srt", "sglang.srt.sampling"):
        if package not in sys.modules:
            packages[package] = types.ModuleType(package)
            packages[package].__path__ = []
    with patch.dict(sys.modules, packages | {name: module}):
        spec.loader.exec_module(module)
        yield module.CustomLogitProcessor.from_str(incremental.make_processor())


@unittest.skipUnless(AVAILABLE, "Requires CPU torch, dill and orjson")
class IncrementalTensorTests(unittest.TestCase):
    def params(self, directory):
        return {
            "__req__": types.SimpleNamespace(
                output_ids=[], origin_input_ids=[0], retraction_count=0
            ),
            "forced_ids": [1, 2],
            "prompt_ids": [0],
            "vocab_size": 3,
            "top_k": 0,
            "capture_dir": str(directory),
        }

    def test_serialized_processor_captures_before_forcing_across_steps(self):
        import torch

        with tempfile.TemporaryDirectory() as temp, processor_class() as processor:
            params = self.params(Path(temp))
            before = torch.tensor([[3.0, 1.0, 2.0]])
            expected = before.log_softmax(-1)[0].tolist()
            first = processor(before.clone(), [params])
            self.assertEqual(first.argmax(-1).item(), 1)
            self.assertEqual(first[0, 1].item(), 0)
            self.assertTrue(torch.isneginf(first[0, [0, 2]]).all())
            # The original unforced winner is token zero, not forced token one.
            params["__req__"].output_ids.append(1)
            second = processor(before.clone(), [params])
            self.assertEqual(second.argmax(-1).item(), 2)
            rows = incremental.collect(Path(temp), [0], [1, 2], 3, 0)
            self.assertEqual(rows[0]["logprobs"], expected)
            self.assertEqual(rows[1]["logprobs"], expected)

    def test_changed_history_retraction_and_nonfinite_logits_fail_before_capture(self):
        import torch

        for error in ("history", "retraction", "nonfinite"):
            with tempfile.TemporaryDirectory() as temp, processor_class() as processor:
                params = self.params(Path(temp))
                logits = torch.tensor([[3.0, 1.0, 2.0]])
                if error == "history":
                    params["__req__"].output_ids = [0]
                elif error == "retraction":
                    params["__req__"].retraction_count = 1
                else:
                    logits[0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    processor(logits, [params])
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_top_k_capture_and_duplicate_callback_preserve_original_evidence(self):
        import torch

        with tempfile.TemporaryDirectory() as temp, processor_class() as processor:
            params = self.params(Path(temp))
            params["top_k"] = 2
            logits = torch.tensor([[3.0, 1.0, 2.0]])
            processor(logits.clone(), [params])
            path = next(Path(temp).glob("*.json"))
            original = path.read_bytes()
            entries = json.loads(original)["pre_forcing_logprobs"]
            self.assertEqual([entry[0] for entry in entries], [0, 2])
            with self.assertRaises(FileExistsError):
                processor(logits.clone(), [params])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
