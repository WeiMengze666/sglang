"""Exercise actual native DP vote methods without importing Torch or CUDA."""

import ast
from enum import IntEnum
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HELPER_NAME = "sglang.srt.layers.attention.flashinfer_tbo"


class Mode(IntEnum):
    IDLE = 0
    EXTEND = 1
    DECODE = 2
    PREBUILT = 3
    MIXED = 4

    def is_target_verify(self):
        return False

    def is_decode(self):
        return self == self.DECODE

    def is_idle(self):
        return self == self.IDLE

    def is_prebuilt(self):
        return self == self.PREBUILT

    def is_extend(self):
        return self == self.EXTEND


class Column(list):
    def tolist(self):
        return list(self)


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return Column(row[key[1]] for row in self.rows)


class Admission(unittest.TestCase):
    def setUp(self):
        self.bf16 = object()
        self.exec_cfg = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True),
            features=SimpleNamespace(enable_mis=False),
        )
        self.model_cfg = SimpleNamespace(kv_cache_dtype="auto")
        self.backend = "flashinfer"
        self.model = SimpleNamespace(
            dtype=self.bf16,
            head_dim=128,
            v_head_dim=128,
            is_multimodal=False,
            sliding_window_size=None,
            hf_text_config=SimpleNamespace(model_type="qwen3_moe"),
            get_max_num_attention_heads=lambda: 96,
            get_num_kv_heads=lambda *_: 8,
        )
        path = ROOT / "python/sglang/srt/batch_overlap/two_batch_overlap.py"
        tree = ast.parse(path.read_text())
        wanted = {
            "compute_split_seq_index",
            "_is_two_chunk_split_enabled",
            "_split_extend_seqs",
            "_split_array_by_cum_less_than_half",
            "_split_array_by_balanced_sum",
            "TboDPAttentionPreparer",
        }
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted
        ]
        self.ns = dict(
            ScheduleBatch=object,
            ForwardMode=Mode,
            Sequence=Sequence,
            Optional=Optional,
            torch=SimpleNamespace(bfloat16=self.bf16),
            get_exec=lambda: self.exec_cfg,
            get_model=lambda: self.model_cfg,
            attention_backends=lambda: (self.backend, self.backend),
            get_parallel=lambda: SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
            get_tbo_token_distribution_threshold=lambda: 0.4,
            get_token_num_per_seq=lambda forward_mode, spec_info: (
                1 if forward_mode.is_decode() else None
            ),
            get_deepep_mode=lambda: SimpleNamespace(
                resolve=lambda _: SimpleNamespace(is_low_latency=lambda: False)
            ),
            get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: True),
            is_tbo_enabled=lambda: True,
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), self.ns
        )
        self.preparer = self.ns["TboDPAttentionPreparer"]
        spec = importlib.util.spec_from_file_location(
            HELPER_NAME, ROOT / "python/sglang/srt/layers/attention/flashinfer_tbo.py"
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.modules = patch.dict("sys.modules", {HELPER_NAME: helper})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def batch(self, lengths, mode=Mode.EXTEND):
        return SimpleNamespace(
            model_config=self.model,
            forward_mode=mode,
            extend_lens=lengths,
            extend_num_tokens=sum(lengths),
            spec_info=None,
            is_extend_in_batch=mode == Mode.EXTEND,
            batch_size=lambda: len(lengths),
        )

    def test_tp1_scheduler_still_prepares_local_tbo_vote(self):
        path = ROOT / "python/sglang/srt/utils/common.py"
        tree = ast.parse(path.read_text())
        method = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "require_mlp_sync"
        )
        enabled = [True]
        ns = dict(
            get_parallel=lambda: SimpleNamespace(attn_dp_enabled=False),
            require_gathered_buffer=lambda: False,
            get_exec=lambda: SimpleNamespace(
                overlap=SimpleNamespace(enable_two_batch_overlap=enabled[0])
            ),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        self.assertTrue(ns["require_mlp_sync"]())
        # DP1 finalizes the same one-row vote without a collective.
        for lengths, expected in [([2], None), ([8191], 0)]:
            worker = self.preparer()
            row = worker.prepare_all_gather(self.batch(lengths))
            self.assertEqual(worker.compute_output(Matrix([row]))[0], expected)
        enabled[0] = False
        self.assertFalse(ns["require_mlp_sync"]())

    def test_actual_backend_hook_matches_scheduler_geometry(self):
        path = ROOT / "python/sglang/srt/layers/attention/flashinfer_backend.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "adjust_tbo_prefill_split"
        )
        ns = dict(ForwardBatch=object, torch=SimpleNamespace(bfloat16=self.bf16))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        updater = SimpleNamespace(
            head_dim=128,
            q_data_type=self.bf16,
            data_type=self.bf16,
            num_qo_heads=96,
            num_kv_heads=8,
        )
        backend = SimpleNamespace(
            enable_deterministic=True,
            tbo_fa2_split_supported=True,
            prefill_backend="fa2",
            num_wrappers=1,
            dispatch_reason=None,
            is_dllm_model=False,
            is_multimodal=False,
            enable_mis=False,
            indices_updater_prefill=updater,
        )
        hook = ns["adjust_tbo_prefill_split"]
        for lengths, expected in [
            ([8191], 4096),
            ([513], 256),
            ([12288, 12287, 12287], 18432),
        ]:
            worker = self.preparer()
            self.assertTrue(worker.prepare_all_gather(self.batch(lengths))[0])
            forward = SimpleNamespace(
                cross_attention_custom_mask=None,
                extend_seq_lens_cpu=lengths,
                tbo_split_seq_index=worker.local_tbo_split_seq_index,
            )
            self.assertEqual(hook(backend, forward, sum(lengths) // 2), expected)
        forward = SimpleNamespace(
            cross_attention_custom_mask=None,
            extend_seq_lens_cpu=[8191],
            tbo_split_seq_index=0,
        )
        for obj, field, value in [
            (backend, "enable_deterministic", False),
            (backend, "tbo_fa2_split_supported", False),
            (backend, "prefill_backend", "fa3"),
            (backend, "num_wrappers", 2),
            (backend, "dispatch_reason", "cross"),
            (backend, "is_dllm_model", True),
            (backend, "is_multimodal", True),
            (backend, "enable_mis", True),
            (forward, "cross_attention_custom_mask", object()),
            (updater, "data_type", "fp8"),
            (updater, "q_data_type", "fp16"),
            (updater, "head_dim", 64),
            (updater, "num_qo_heads", 64),
        ]:
            with self.subTest(field=field):
                old = getattr(obj, field)
                setattr(obj, field, value)
                try:
                    self.assertEqual(hook(backend, forward, 4095), 4095)
                finally:
                    setattr(obj, field, old)

    def test_one_unsafe_rank_disables_overlap_for_every_rank(self):
        workers = [self.preparer() for _ in range(3)]
        rows = [
            w.prepare_all_gather(b)
            for w, b in zip(workers, [self.batch([2]), self.batch([8191]), None])
        ]
        self.assertEqual([x[0] for x in rows], [False, True, True])
        for worker in workers:
            self.assertEqual(worker.compute_output(Matrix(rows)), (None, None))

    def test_compatible_prefill_and_decode_keep_native_overlap(self):
        for batches, mode in [
            ([self.batch([8191]), self.batch([128, 128]), None], Mode.EXTEND),
            (
                [
                    self.batch([1, 1], Mode.DECODE),
                    self.batch([1] * 16, Mode.DECODE),
                    None,
                ],
                Mode.DECODE,
            ),
        ]:
            workers = [self.preparer() for _ in batches]
            rows = [w.prepare_all_gather(b) for w, b in zip(workers, batches)]
            self.assertTrue(all(r[0] for r in rows))
            for worker in workers:
                index, global_mode = worker.compute_output(Matrix(rows))
                self.assertIsNotNone(index)
                self.assertEqual(global_mode, mode)

    def test_unsupported_modes_keep_existing_eligibility(self):
        batch = self.batch([2])
        self.assertFalse(self.preparer().prepare_all_gather(batch)[0])
        for obj, field, value in [
            (self, "backend", "triton"),
            (self.exec_cfg.deterministic, "enable_deterministic_inference", False),
            (self.model_cfg, "kv_cache_dtype", "fp8_e4m3"),
            (self.model, "dtype", "fp16"),
            (self.model, "is_multimodal", True),
            (self.model, "head_dim", 64),
            (self.model, "v_head_dim", 64),
            (self.model, "sliding_window_size", 512),
            (self.model.hf_text_config, "model_type", "qwen2_moe"),
            (self.model, "get_max_num_attention_heads", lambda: 64),
        ]:
            with self.subTest(field=field):
                old = getattr(obj, field)
                setattr(obj, field, value)
                try:
                    self.assertTrue(self.preparer().prepare_all_gather(batch)[0])
                finally:
                    setattr(obj, field, old)

    def test_sequence_boundary_with_changed_cta_votes_no(self):
        # Force a boundary for this check: the real preparer must still apply
        # the CTA check when no sequence is cut internally.
        self.ns["_is_two_chunk_split_enabled"] = lambda _: False
        batch = self.batch([2, 16])
        self.assertFalse(self.preparer().prepare_all_gather(batch)[0])


if __name__ == "__main__":
    unittest.main()
