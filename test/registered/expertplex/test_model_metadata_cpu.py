"""Model metadata compatibility; checkpoint materialization uses CPU only."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/expertplex"
sys.path.insert(0, str(EXAMPLES))
import summarize_routing
from _bootstrap import module

CONFIG = module("config")


def coder_config():
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "torch_dtype": "bfloat16",
        "hidden_size": 6144,
        "moe_intermediate_size": 2560,
        "num_experts": 160,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 62,
        "num_attention_heads": 96,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
    }


def alias_config(data):
    result = dict(data)
    result["num_local_experts"] = result.pop("num_experts")
    return result


class ModelMetadataTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run(
            [sys.executable, str(EXAMPLES / name), *map(str, args)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )

    def test_normalization_preserves_source_and_rejects_ambiguous_counts(self):
        raw = coder_config()
        alias = alias_config(raw)
        self.assertEqual(CONFIG.normalize_model_config(alias), raw)
        self.assertNotIn("num_experts", alias)
        self.assertEqual(
            CONFIG.normalize_model_config(raw | {"num_local_experts": 160}), raw
        )
        for data in (
            {},
            raw | {"num_local_experts": 8},
            {"num_experts": True},
            {"num_local_experts": 0},
            {"num_experts": 160.0, "num_local_experts": 160},
            {"num_experts": 160, "num_local_experts": "160"},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                CONFIG.normalize_model_config(data)

    def test_synthetic_cli_removes_original_alias_before_changing_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = [coder_config(), alias_config(coder_config())]
            for geometry, experts in (
                ("tiny", 8),
                ("tiny-deepep", 8),
                ("30b", 128),
                ("480b", 160),
            ):
                outputs = []
                for index, config in enumerate(configs):
                    source = root / f"source-{index}.json"
                    source.write_text(json.dumps(config))
                    output = root / f"{geometry}-{index}"
                    self.run_script(
                        "prepare_synthetic.py",
                        "--source-config",
                        source,
                        "--output",
                        output,
                        "--geometry",
                        geometry,
                    )
                    data = json.loads((output / "config.json").read_text())
                    self.assertNotIn("num_local_experts", data)
                    self.assertEqual(
                        CONFIG.ModelSpec.from_dict(data).num_experts, experts
                    )
                    self.assertEqual(json.loads(source.read_text()), config)
                    outputs.append(data)
                self.assertEqual(*outputs)

    def test_routing_cli_accepts_alias_and_hashes_original_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = json.dumps(alias_config(coder_config())).encode()
            (root / "config.json").write_bytes(raw)
            output = root / "routing.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "summarize_routing",
                        "--model-path",
                        str(root),
                        "--prefill-records",
                        str(root / "p.pt"),
                        "--decode-records",
                        str(root / "d.pt"),
                        "--provenance",
                        "CPU test fixture",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(
                    summarize_routing, "summarize_files", return_value={}
                ) as read,
            ):
                summarize_routing.main()
            self.assertEqual(read.call_count, 2)
            for call in read.call_args_list:
                self.assertEqual(
                    call.kwargs, {"layers": 62, "experts": 160, "top_k": 8}
                )
            report = json.loads(output.read_text())
            self.assertEqual(
                report["model_config_sha256"], hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual((root / "config.json").read_bytes(), raw)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
        "CPU checkpoint comparison requires Torch and safetensors",
    )
    def test_tiny_cpu_checkpoint_is_identical_for_both_metadata_spellings(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(coder_config()))
            self.run_script(
                "prepare_synthetic.py",
                "--source-config",
                source,
                "--output",
                root / "metadata",
                "--geometry",
                "tiny",
            )
            (root / "alias").mkdir()
            data = json.loads((root / "metadata/config.json").read_text())
            (root / "alias/config.json").write_text(json.dumps(alias_config(data)))
            for metadata in ("metadata", "alias"):
                self.run_script(
                    "prepare_tiny_checkpoint.py",
                    "--metadata",
                    root / metadata,
                    "--output",
                    root / f"weights-{metadata}",
                )
            # Canonical metadata and every deterministic BF16 tensor are identical.
            for filename in ("config.json", "TEST_CHECKPOINT.json"):
                self.assertEqual(
                    json.loads((root / "weights-metadata" / filename).read_text()),
                    json.loads((root / "weights-alias" / filename).read_text()),
                )
            reference = load_file(
                root / "weights-metadata/model.safetensors", device="cpu"
            )
            candidate = load_file(
                root / "weights-alias/model.safetensors", device="cpu"
            )
            self.assertEqual(reference.keys(), candidate.keys())
            for name, tensor in reference.items():
                self.assertEqual(candidate[name].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(tensor, candidate[name]), name)


if __name__ == "__main__":
    unittest.main()
