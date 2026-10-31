"""CPU-only geometry and native dispatch checks; GPU bitexact is separate."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "python/sglang/srt/layers/attention/flashinfer_tbo.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("fa2_tbo", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aligned_fa2_split


def tile_for(tokens, sequences, group):
    average = tokens * group // sequences
    return 128 if average > 64 else 64 if average > 16 else 16


class FlashInferTboSplit(unittest.TestCase):
    def test_child_lengths_use_actual_cut_instead_of_parent_half(self):
        path = ROOT / "python/sglang/srt/batch_overlap/two_batch_overlap.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "TboForwardBatchPreparer"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "derive_fields_related_to_seq_len_for_two_chunk"
        )
        fn.decorator_list = []

        def update(*, batch, cpu_field, device_field, sum_field):
            values = getattr(batch, cpu_field)
            setattr(batch, device_field, list(values))
            if sum_field is not None:
                setattr(batch, sum_field, sum(values))

        namespace = dict(
            ForwardBatch=object,
            Optional=__import__("typing").Optional,
            copy=copy,
            _update_device_and_sum_field_from_cpu_field=update,
            attention_backends=lambda: ("flashinfer", "flashinfer"),
            compute_position=lambda backend, prefix, lengths, count: (
                None,
                [sum(lengths[:i]) for i in range(len(lengths))],
            ),
        )
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        parent = SimpleNamespace(extend_seq_lens_cpu=[8191])
        a = SimpleNamespace(
            extend_seq_lens_cpu=[8191], extend_prefix_lens_cpu=[31], seq_lens_cpu=[8222]
        )
        b = copy.deepcopy(a)
        namespace[fn.name](
            None,
            parent,
            child_a=a,
            child_b=b,
            tbo_split_seq_index=0,
            split_token_index=6144,
        )
        self.assertEqual(a.extend_seq_lens_cpu, [6144])
        self.assertEqual(b.extend_seq_lens_cpu, [2047])
        self.assertEqual(a.seq_lens_cpu, [6175])
        self.assertEqual(b.seq_lens_cpu, [8222])
        self.assertEqual(b.extend_prefix_lens_cpu, [6175])
        self.assertEqual(b.extend_start_loc, [0])
        self.assertEqual(a.extend_num_tokens + b.extend_num_tokens, 8191)
        self.assertEqual(parent.extend_seq_lens_cpu, [8191])

    def test_observed_geometries_and_short_tail(self):
        choose = load_helper()
        for length, cut, expected in [
            (8191, 6143, 6144),
            (4095, 2047, 2048),
            (8193, 8192, 8160),
            (512, 510, 480),
            (512, 511, 480),
            (513, 512, 480),
            (33, 32, 32),  # no compatible cut: preserve existing behavior
        ]:
            self.assertEqual(choose([length], 0, cut, 12), expected)

    def test_matches_enumerated_geometry_without_changing_sequence(self):
        choose = load_helper()
        for lengths in ([n] for n in range(2, 100)):
            for group in (4, 6, 8, 12, 16):
                for desired in (1, sum(lengths) // 2, sum(lengths) - 1):
                    self.check_choices(choose, lengths, 0, desired, group)
        for lengths in ([7, 513, 2], [512, 512, 31], [1, 2, 3], [12288, 12287, 12287]):
            for index, length in enumerate(lengths):
                if length > 1:
                    self.check_choices(
                        choose, lengths, index, sum(lengths[:index]) + length // 2, 12
                    )

    def check_choices(self, choose, lengths, index, desired, group):
        before = list(lengths)
        tile = tile_for(sum(lengths), len(lengths), group)
        origin = sum(lengths[:index])
        candidates = []
        for offset in range(1, lengths[index]):
            cut = origin + offset
            if offset * group % tile:
                continue
            if tile_for(cut, index + 1, group) != tile:
                continue
            if tile_for(sum(lengths) - cut, len(lengths) - index, group) != tile:
                continue
            candidates.append(cut)
        expected = (
            min(candidates, key=lambda c: (abs(c - desired), c))
            if candidates
            else desired
        )
        self.assertEqual(choose(lengths, index, desired, group), expected)
        self.assertEqual(lengths, before)

    def test_unsupported_inputs_remain_unchanged(self):
        choose = load_helper()
        for lengths, sequence, cut, group in [
            ([], 0, 7, 12),
            ([0, 512], 1, 257, 12),
            ([512], 1, 256, 12),
            ([512], 0, 256, 0),
            ([512], 0, 512, 12),
        ]:
            self.assertEqual(choose(lengths, sequence, cut, group), cut)

    def test_native_dispatch_uses_hook_only_for_internal_extend(self):
        path = ROOT / "python/sglang/srt/batch_overlap/two_batch_overlap.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "TboForwardBatchPreparer"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_compute_split_token_index"
        )
        fn.decorator_list = []
        calls = []
        namespace = dict(
            ForwardBatch=object,
            ForwardMode=SimpleNamespace(EXTEND="extend"),
            get_token_num_per_seq=lambda **_: None,
            compute_split_token_index=lambda **_: 511,
            _is_two_chunk_split_enabled=lambda lengths: lengths == [512],
            get_attn_backend=lambda: SimpleNamespace(
                primary=SimpleNamespace(
                    adjust_tbo_prefill_split=lambda batch, cut: calls.append(cut) or 480
                )
            ),
        )
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        compute = namespace[fn.name]
        batch = SimpleNamespace(
            forward_mode="extend",
            spec_info=None,
            tbo_split_seq_index=0,
            extend_seq_lens_cpu=[512],
        )
        self.assertEqual(compute(None, batch), 480)
        self.assertEqual(calls, [511])
        for mode, lengths in [("decode", [512]), ("extend", [256, 256])]:
            batch.forward_mode, batch.extend_seq_lens_cpu = mode, lengths
            self.assertEqual(compute(None, batch), 511)
        namespace["get_attn_backend"] = lambda: SimpleNamespace(primary=object())
        batch.forward_mode, batch.extend_seq_lens_cpu = "extend", [512]
        self.assertEqual(compute(None, batch), 511)
        self.assertEqual(calls, [511])


if __name__ == "__main__":
    unittest.main()
