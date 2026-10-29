"""Reject a numerically incomparable reference before starting GPU services."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
alignment = importlib.import_module("check_numerical_alignment")
validate = alignment.validate_reference


class AlignmentConfigurationTests(unittest.TestCase):
    def test_attention_tp_reference_width_and_physical_layout_match(self):
        self.assertEqual(alignment.attention_tp_layout(4, 1), (4, 3))
        self.assertEqual(alignment.attention_tp_layout(4, 2), (2, 1))
        for experts, tp in ((4, 0), (4, True), (4, 3), (4, 4), (6, 2)):
            with self.assertRaises(ValueError):
                alignment.attention_tp_layout(experts, tp)
        (self.root / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "attention TP width differs"):
            validate(self.root, self.model, False, attention_tp_size=2)

    def test_warmup_is_separate_and_cannot_weaken_real_capture(self):
        command = [
            "python",
            "capture.py",
            "--capture-dir",
            "/strict/capture",
            "--output",
            "/strict/scores.json",
            "--histories",
            "/reference/histories.json",
            "--reference",
            "/reference/scores.json",
            "--require-exact-logprobs",
            "--require-decode-batch-size",
            "2",
            "--top-k",
            "0",
            "--concurrency",
            "16",
            "--steps",
            "32",
            "--batched-request",
        ]
        original = list(command)
        warmup = alignment.warmup_capture_command(command, Path("/strict/warmup"))
        self.assertEqual(command, original)
        for flag, value in (
            ("--capture-dir", "/strict/warmup/capture"),
            ("--output", "/strict/warmup/scores.json"),
            ("--require-decode-batch-size", "1"),
            ("--top-k", "1"),
            ("--histories", "/reference/histories.json"),
            ("--concurrency", "16"),
            ("--steps", "32"),
        ):
            self.assertEqual(warmup[warmup.index(flag) + 1], value)
        self.assertNotIn("--reference", warmup)
        self.assertNotIn("--require-exact-logprobs", warmup)
        self.assertIn("--batched-request", warmup)

    def test_kv_capacity_includes_concurrent_pd_reservations(self):
        capacity = alignment.diagnostic_kv_capacity(list(range(31, 47)), 32, 16, 4096)
        self.assertEqual(capacity, 12288)
        # Worst-case admission includes every live trajectory plus its reserve,
        # even with no prefix hits. The old 8192 cap could not admit the last one.
        required = sum(n + 32 + 512 for n in range(31, 47))
        self.assertLess(8192, required)
        self.assertGreaterEqual(capacity, required)
        self.assertEqual(alignment.diagnostic_kv_capacity([31, 46], 32, 1, 4096), 8192)
        self.assertEqual(
            alignment.diagnostic_kv_capacity([8191, 16385, 24575], 8, 1, 32768),
            32768,
        )
        self.assertEqual(
            alignment.diagnostic_kv_capacity([100, 15000, 20000], 8, 2, 32768),
            36352,
        )
        for lengths, steps, concurrency, context in (
            ([], 32, 16, 4096),
            ([True], 32, 16, 4096),
            ([31], 1, 16, 4096),
            ([31], 32, 17, 4096),
            ([4064], 32, 16, 4096),
            ([31], 32, True, 4096),
        ):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                alignment.diagnostic_kv_capacity(lengths, steps, concurrency, context)

    def test_native_repeat_must_preserve_capture_routing(self):
        path = self.root / "report.json"
        path.write_text(json.dumps(self.report))
        alignment.validate_native_capture_routing(self.root, None, 4)
        with self.assertRaisesRegex(ValueError, "routing differs"):
            alignment.validate_native_capture_routing(self.root, 0, 4)
        self.report["native_dp_rank"] = 0
        path.write_text(json.dumps(self.report))
        alignment.validate_native_capture_routing(self.root, 0, 4)
        with self.assertRaisesRegex(ValueError, "routing differs"):
            alignment.validate_native_capture_routing(self.root, None, 4)
        for rank in (-1, 4, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                alignment.validate_native_capture_routing(None, rank, 4)

    def test_tokenizer_vocabulary_mismatch_rejected_before_gpu_launch(self):
        alignment.validate_prompt_ids([[0, 1023]], 1024, 4, 16, 2)
        for prompts in ([[1024]], [[-1]], [[True]], [[0.0]], [[]], []):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                alignment.validate_prompt_ids(prompts, 1024, 4, 16, 2)
        with self.assertRaisesRegex(ValueError, "capacity"):
            alignment.validate_prompt_ids([[1] * 5], 1024, 4, 16, 2)

    def test_tokenizer_free_boundary_patterns_keep_exact_lengths(self):
        self.assertEqual(
            alignment.repeat_token_pattern([11, 12], [1, 3, 4]),
            [[11], [11, 12, 11], [11, 12, 11, 12]],
        )
        with self.assertRaises(ValueError):
            alignment.repeat_token_pattern([], [1])

    def test_installed_deepep_topology_is_checked_before_model_launch(self):
        buffer = Mock()
        alignment.validate_native_reference_topology(4, buffer)
        buffer.get_dispatch_config.assert_called_once_with(4)
        buffer.get_combine_config.assert_called_once_with(4)
        for method in ("get_dispatch_config", "get_combine_config"):
            buffer = Mock()
            getattr(buffer, method).side_effect = AssertionError("unsupported")
            with self.assertRaisesRegex(ValueError, "5 EP ranks"):
                alignment.validate_native_reference_topology(5, buffer)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model"
        self.report = {"variant": "native", "status": "passed", "tbo_enabled": True}
        (self.root / "scores.json").write_text("{}")
        (self.root / "histories.json").write_text(
            json.dumps({"model_path": str(self.model)})
        )

    def check(self, enabled=True):
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, enabled)

    def test_matching_and_legacy_reference(self):
        self.check()
        del self.report["tbo_enabled"]
        self.report["commands"] = [
            ["python", "-m", "sglang.launch_server", "--enable-two-batch-overlap"]
        ]
        self.check()
        self.report["commands"][0].pop()
        self.check(False)

    def test_mismatched_overlap_or_failed_reference(self):
        with self.assertRaisesRegex(ValueError, "TBO mode"):
            self.check(False)
        self.report["tbo_enabled"] = 1
        with self.assertRaises(ValueError):
            self.check()
        self.report["tbo_enabled"] = True
        self.report["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "successful native"):
            self.check()

    def test_mismatched_model_or_unknown_overlap(self):
        self.model = self.root / "different-model"
        with self.assertRaisesRegex(ValueError, "model path"):
            self.check()
        del self.report["tbo_enabled"]
        with self.assertRaisesRegex(ValueError, "overlap configuration"):
            self.check()

    def test_score_step_count_must_match_before_launch(self):
        self.check()
        (self.root / "scores.json").write_text(json.dumps({"cases": [{"steps": 32}]}))
        validate(self.root, self.model, True, 32)
        with self.assertRaisesRegex(ValueError, "step count"):
            validate(self.root, self.model, True, 8)

    def test_compilation_must_match_reference(self):
        self.check()
        with self.assertRaisesRegex(ValueError, "torch.compile mode"):
            validate(self.root, self.model, True, compile_enabled=True)
        self.report["torch_compile_enabled"] = True
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, compile_enabled=True)

        with self.assertRaisesRegex(ValueError, "torch.compile mode"):
            validate(self.root, self.model, True)
        del self.report["torch_compile_enabled"]
        self.report["commands"] = [
            ["python", "-m", "sglang.launch_server", "--enable-torch-compile"]
        ]
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, compile_enabled=True)

    def test_determinism_mode_must_match_reference(self):
        self.check()
        with self.assertRaisesRegex(ValueError, "deterministic inference"):
            validate(self.root, self.model, True, deterministic=True)
        self.report["deterministic_inference"] = True
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, deterministic=True)
        with self.assertRaisesRegex(ValueError, "deterministic inference"):
            validate(self.root, self.model, True)

    def test_prefill_graph_mode_and_padding_configuration_must_match(self):
        self.report.update(
            tbo_enabled=False,
            prefill_graph_backend="breakable",
            max_prefill_tokens=1024,
        )
        (self.root / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "prefill graph mode"):
            validate(self.root, self.model, False)
        validate(
            self.root,
            self.model,
            False,
            prefill_graph_backend="breakable",
            prefill_graph_max_tokens=1024,
        )
        with self.assertRaisesRegex(ValueError, "capture configuration"):
            validate(
                self.root,
                self.model,
                False,
                prefill_graph_backend="breakable",
                prefill_graph_max_tokens=2048,
            )
        self.report["prefill_exact_buckets"] = True
        (self.root / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "capture configuration"):
            validate(
                self.root,
                self.model,
                False,
                prefill_graph_backend="breakable",
                prefill_graph_max_tokens=1024,
            )

    def test_prefill_graph_cli_rejects_incomparable_reference_before_loading(self):
        self.report["tbo_enabled"] = False
        (self.root / "report.json").write_text(json.dumps(self.report))
        common = [
            sys.executable,
            alignment.__file__,
            "--variant",
            "expertplex",
            "--reference",
            str(self.root),
            "--model-path",
            str(self.model),
            "--output",
            str(self.root / "candidate"),
            "--prefill-graph-backend",
            "breakable",
        ]
        for extra, message in [
            ([], "requires --disable-tbo"),
            (["--disable-tbo"], "Reference prefill graph mode differs"),
        ]:
            result = subprocess.run(
                common + extra, check=False, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)
            self.assertFalse((self.root / "candidate").exists())

    def test_prefill_graph_capture_does_not_prove_actual_replay(self):
        log = self.root / "prefill.log"
        for text in (
            "Capture CUDA Graph completed",
            "Prefill batch: cuda graph: False",
            "Decode batch: cuda graph: True",
        ):
            log.write_text(text + "\n")
            with self.assertRaisesRegex(ValueError, "No prefill CUDA Graph replay"):
                alignment.require_prefill_graph_replay(log)
        log.write_text("Prefill batch: #new-token: 127, cuda graph: True\n")
        alignment.require_prefill_graph_replay(log)

    def test_expert_placement_must_match_reference(self):
        self.check()
        with self.assertRaisesRegex(ValueError, "expert count"):
            validate(self.root, self.model, True, expert_count=5)
        self.report["expert_count"] = 5
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, expert_count=5)
        with self.assertRaisesRegex(ValueError, "expert count"):
            validate(self.root, self.model, True)

    def test_concurrency_must_match_reference(self):
        self.check()
        with self.assertRaisesRegex(ValueError, "concurrency"):
            validate(self.root, self.model, True, concurrency=4)
        self.report["concurrency"] = 4
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, concurrency=4)

    def test_native_batch_submission_must_match_reference(self):
        # Same concurrency, different submission boundaries: reject before
        # either the score payload or GPU services are loaded.
        self.report["concurrency"] = 4
        for saved, requested in [(False, True), (True, False), (1, True)]:
            with self.subTest(saved=saved, requested=requested):
                self.report["batched_request"] = saved
                (self.root / "report.json").write_text(json.dumps(self.report))
                with self.assertRaisesRegex(ValueError, "batch submission"):
                    validate(
                        self.root,
                        self.model,
                        True,
                        concurrency=4,
                        batched_request=requested,
                    )
        for saved in (False, True):
            self.report["batched_request"] = saved
            (self.root / "report.json").write_text(json.dumps(self.report))
            validate(self.root, self.model, True, concurrency=4, batched_request=saved)
        del self.report["batched_request"]
        (self.root / "report.json").write_text(json.dumps(self.report))
        validate(self.root, self.model, True, concurrency=4)
        with self.assertRaisesRegex(ValueError, "batch submission"):
            validate(self.root, self.model, True, concurrency=4, batched_request=True)

        # The CLI must forward the requested mode and fail before trying to
        # load this intentionally nonexistent checkpoint or the empty scores.
        result = subprocess.run(
            [
                sys.executable,
                alignment.__file__,
                "--variant",
                "expertplex",
                "--reference",
                str(self.root),
                "--model-path",
                str(self.model),
                "--output",
                str(self.root / "candidate"),
                "--concurrency",
                "4",
                "--batched-request",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Reference batch submission mode differs", result.stderr)
        self.assertFalse((self.root / "candidate").exists())

    def test_trajectory_capacity_and_continuation(self):
        histories = {
            "outputs": [
                {
                    "prompt": [1] * 129,
                    "meta_info": {"output_token_logprobs": [[0.0, 2]] * 8},
                }
            ]
        }
        alignment.validate_workload(histories, 8, 256, 1024)
        for steps, capacity, context in [(8, 128, 1024), (9, 256, 1024), (8, 256, 137)]:
            with (
                self.subTest(steps=steps, capacity=capacity, context=context),
                self.assertRaises(ValueError),
            ):
                alignment.validate_workload(histories, steps, capacity, context)
        with self.assertRaises(ValueError):
            alignment.validate_workload({"outputs": []}, 8, 256, 1024)

    def test_boundary_prompts_have_exact_lengths(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.assert_no_special = not add_special_tokens
                return [10, 20, 30]

        tokenizer = Tokenizer()
        lengths = [1, 127, 128, 129, 511, 512, 513, 2047]
        prompts = alignment.boundary_prompts(tokenizer, lengths)
        self.assertEqual([len(p) for p in prompts], lengths)
        self.assertEqual(prompts[2], prompts[3][:128])
        self.assertTrue(tokenizer.assert_no_special)


if __name__ == "__main__":
    unittest.main()
