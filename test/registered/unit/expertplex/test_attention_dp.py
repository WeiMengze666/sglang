"""CPU contracts for optional native attention DP; no CUDA or throughput proof.

Run with CUDA_VISIBLE_DEVICES=999 NVIDIA_VISIBLE_DEVICES=999 and the checkout's
python/ on PYTHONPATH. Fixtures contain model dimensions and API metadata only.
The dispatcher test mocks the collective leaf; it does not validate NCCL.
"""
import argparse
import dataclasses
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from sglang.srt.expertplex.config import ClientConfig, NodeConfig, ModelSpec, load_config
from sglang.srt.expertplex.attention_dp import copy_real_dp_counts_, make_eager_real_dp_counts, mask_topk_padding
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.expertplex import integration
launch = importlib.import_module("launch")
API = json.loads((Path(__file__).parent / "fixtures/attention_dp1_api.json").read_text())
MODEL_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "torch_dtype": "bfloat16", "hidden_size": 6144,
    "moe_intermediate_size": 2560, "num_experts": 160,
    "num_experts_per_tok": 8, "num_hidden_layers": 62,
    "num_attention_heads": 96, "num_key_value_heads": 8,
    "head_dim": 128, "vocab_size": 151936,
}

def setUpModule():
    global _temporary, MODEL
    _temporary = tempfile.TemporaryDirectory(prefix="expertplex-attention-dp-")
    MODEL = str(Path(_temporary.name) / "model")
    Path(MODEL).mkdir()
    (Path(MODEL) / "config.json").write_text(json.dumps(MODEL_CONFIG))

def legacy_node():
    value = dict(API["node"], model_path=MODEL)
    return NodeConfig.from_dict(value)

def group_node():
    old = legacy_node()
    return dataclasses.replace(old, clients=(old.clients[0], dataclasses.replace(old.clients[1], tp_size=3, attention_dp_size=3)), max_decode_tokens=48)

def args():
    return SimpleNamespace(config="/node.json", port=31000, bootstrap_port=19000, nccl_port=20000, cuda_graph=True, chunked_prefill=False, kv_tokens=None, attention_backend="fa4", sampling_backend="flashinfer", ib_device=None, skip_tokenizer_init=False, tokenizer_path=None, prefill_graph_backend="disabled", prefill_graph_max_tokens=None, host="127.0.0.1")

def command_contract(node):
    worker, servers, router = launch.commands(node, args())
    def normalized(command):
        return ["<python>", *("/model" if value == MODEL else value for value in command[1:])]
    return [normalized(worker), [[client.id, normalized(command)] for client, command in servers], normalized(router)]

def parsed_decode_args(node):
    from sglang.srt.server_args import ServerArgs
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(launch.commands(node, args())[1][1][1][3:])

def routes(n=48):
    ids = torch.arange(n * 2, dtype=torch.int64).reshape(n, 2) % 160
    weights = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) / 128
    if n:
        weights[0, 0] = -0.0
    return StandardTopKOutput(weights, ids, object())

def batch(counts, mode=DpPaddingMode.MAX_LEN, raw=None):
    if raw is None:
        raw = torch.tensor(counts, dtype=torch.int32)
    return SimpleNamespace(original_global_num_tokens_cpu=counts, expertplex_dp_real_num_tokens_gpu=raw, dp_padding_mode=mode, input_ids=torch.empty(1, dtype=torch.long))

def registry(enabled=True):
    return build_decode_registry(device=torch.device("cpu"), max_bs=16, max_num_token=16, seq_len_fill_value=1, cache_loc_dtype=torch.int64, require_gathered_buffer=True, require_mlp_tp_gather=True, num_dp_ranks=3, enable_expertplex_attention_dp=enabled, share_pool=False)


class ExplicitDPConfig(unittest.TestCase):

    def test_legacy_json_canonical_identity_and_mailboxes(self):
        node = legacy_node()
        expected = dict(API['node'], model_path=MODEL)
        self.assertEqual(json.loads(json.dumps(node.to_dict())), expected)
        self.assertEqual(NodeConfig.from_dict(node.to_dict()), node)
        self.assertEqual(node.buffer_layout(), ([49152, 49152, 16, 16, 16, 16, 16, 16], [1, 1, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(node.slot_count, 8)
        self.assertTrue(all((c.attention_dp_size == 1 for c in node.clients)))
        self.assertTrue(all(('attention_dp_size' not in c for c in node.to_dict()['clients'])))

    def test_effective_heads_device_ownership_and_memory(self):
        node = group_node()
        spec = ModelSpec.from_model_path(MODEL)
        node.validate(spec)
        self.assertEqual(node.clients[1].attention_tp_size, 1)
        self.assertEqual(node.clients[1].tp_size, 3)
        self.assertEqual(node.buffer_layout(), ([49152, 49152, 48, 48], [1, 1, 0, 0]))
        self.assertEqual(spec.attention_rank_weight_bytes(node.clients[1].attention_tp_size), spec.attention_rank_weight_bytes(1))
        self.assertEqual(spec.attention_rank_kv_bytes_per_token(1), 253952)
        with self.assertRaises(ValueError):
            spec.attention_rank_kv_heads(3)
        original = legacy_node()
        with self.assertRaises(ValueError):
            dataclasses.replace(original, clients=tuple((dataclasses.replace(c, tp_size=3, attention_dp_size=3) if c.id == 1 else c for c in original.clients)))

    def test_invalid_divisors_phases_and_unqualified_optimizations(self):
        for span, dp, phase in [(3, 2, 'decode'), (3, 0, 'decode'), (3, True, 'decode'), (3, 3, 'prefill'), (4, 2, 'decode')]:
            with self.subTest(span=span, dp=dp, phase=phase), self.assertRaises(ValueError):
                ClientConfig(1, phase, 1, 30001, span, 'none', dp)
        node = group_node()
        with self.assertRaises(ValueError):
            dataclasses.replace(node, max_decode_tokens=47)
        with self.assertRaises(ValueError):
            dataclasses.replace(node, clients=(node.clients[0], dataclasses.replace(node.clients[1], overlap='tbo')))

    def test_launcher_actual_native_parser_limits(self):
        node = group_node()
        _, servers, _ = launch.commands(node, args())
        command = servers[1][1]
        from sglang.srt.server_args import ServerArgs
        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        a = p.parse_args(command[3:])
        self.assertEqual((a.tp_size, a.attn_dp_size, a.dp_size), (3, 3, 1))
        self.assertTrue(a.enable_dp_lm_head)
        self.assertEqual(a.max_running_requests, 48)
        self.assertEqual(a.max_prefill_tokens, 16)
        self.assertEqual(a.chunked_prefill_size, 48)
        self.assertEqual(a.cuda_graph_max_bs_decode, 16)
        self.assertFalse(a.enable_torch_compile)
        self.assertFalse(a.disable_overlap_schedule)
        self.assertFalse(a.enable_two_batch_overlap)

    def test_launcher_rejects_group_compile_cpuoff_or_nondivisor_limits(self):
        for key, value in [('enable_torch_compile', True), ('cuda_graph', False), ('disable_overlap_schedule', True), ('diagnostic_logit_processor', True), ('decode_max_running_requests', 47), ('decode_chunked_prefill_size', 16)]:
            a = args()
            setattr(a, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.commands(group_node(), a)

    def test_optimizer_unmeasured_mode_rejected(self):
        from sglang.srt.expertplex.optimizer import search, ProfileCoverageError
        with self.assertRaisesRegex(ProfileCoverageError, 'independently measured DP'):
            search(group_node(), ModelSpec.from_model_path(MODEL), None, input_length=1, average_output_length=1, context_length=1, prefill_slo_s=1, decode_slo_s=0.05, kv_tokens_per_attention_gpu=128)

class RealEntryAndCompatibility(unittest.TestCase):

    def test_actual_import_provenance_and_lazy_native_api(self):
        names = ['sglang', 'sglang.srt.expertplex.config', 'sglang.srt.expertplex.integration', 'sglang.srt.layers.moe.token_dispatcher.expertplex', 'sglang.srt.model_executor.forward_batch_info', 'sglang.srt.model_executor.cuda_graph_buffer_registry']
        for name in names:
            module = importlib.import_module(name)
            self.assertTrue(Path(module.__file__).resolve().is_relative_to((ROOT / 'python').resolve()), (name, module.__file__))
        from sglang.srt.expertplex import native
        with patch.object(native, 'load_extension', side_effect=AssertionError('CPU metadata built CUDA extension')):
            sizes = native.estimate_buffer_bytes(2, 40, 16, 6144, 2560)
        self.assertTrue(sizes)
        self.assertTrue(all((type(n) is int and n >= 0 for n in sizes.values())))

    def test_default_native_commands_match_dp1_api(self):
        self.assertEqual(command_contract(legacy_node()), API['commands'])

    def test_native_non_expertplex_dp8_no_new_slot(self):
        from sglang.srt.expertplex.attention_dp import grouped_attention_dp_size
        import sglang.srt.runtime_context as rt
        with patch.object(rt, 'get_parallel', return_value=SimpleNamespace(attn_dp_size=8)), patch.object(rt, 'get_disagg', return_value=SimpleNamespace(enable_expertplex=False)), patch.object(integration, 'attention_config', side_effect=AssertionError('native fetched external config')):
            self.assertEqual(grouped_attention_dp_size(), 1)
        reg = build_decode_registry(device=torch.device('cpu'), max_bs=16, max_num_token=16, seq_len_fill_value=1, cache_loc_dtype=torch.int64, require_gathered_buffer=True, require_mlp_tp_gather=True, num_dp_ranks=8, share_pool=False)
        self.assertEqual(list(reg._slots), API['native_dp8_decode_slot_names'])
        self.assertNotIn('expertplex_dp_real_num_tokens_gpu', reg._slots)

    def test_actual_memory_estimator_uses_effective_tp(self):
        from sglang.srt.expertplex.optimizer import estimate_memory
        spec = ModelSpec.from_model_path(MODEL)
        value = estimate_memory(group_node(), spec, kv_tokens_per_attention_gpu=128)
        self.assertEqual(value.attention_weights_bytes, spec.attention_rank_weight_bytes(1))
        self.assertEqual(value.attention_kv_bytes, 128 * spec.attention_rank_kv_bytes_per_token(1))

    def test_actual_integration_group_guards_without_engine(self):
        with tempfile.TemporaryDirectory(prefix='expertplex-group-config-') as tmp:
            path = Path(tmp) / 'group.json'
            group_node().write_json(path)
            original = vars(parsed_decode_args(group_node()))
            original.update(expertplex_bootstrap_addr=None, expertplex_config=str(path), expertplex_client_id=1, chunked_prefill_size=16, cuda_graph_config={'decode': {'backend': 'full', 'bs': [1, 2, 4, 8, 16]}, 'prefill': {'backend': 'disabled', 'bs': []}})
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
                integration.validate_server_args(SimpleNamespace(**original))
                for key, value in [('attn_dp_size', 1), ('dp_size', 3), ('enable_dp_lm_head', False), ('max_running_requests', 47), ('max_prefill_tokens', 32), ('chunked_prefill_size', 32), ('disable_overlap_schedule', True), ('enable_torch_compile', True)]:
                    bad = dict(original)
                    bad[key] = value
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        integration.validate_server_args(SimpleNamespace(**bad))

    def test_actual_legacy_alias_normalizes_outer_dp(self):
        import sglang.srt.arg_groups.parallel_hook as parallel
        cfg = SimpleNamespace(enable_dp_attention=True, attn_dp_size=1, dp_size=3)

        def declare(_args, _handler, **changes):
            for key, value in changes.items():
                setattr(cfg, key, value)
        with patch.object(parallel, 'resolving_view', return_value=cfg), patch.object(parallel, 'declare_resolution', side_effect=declare):
            parallel.handle_deprecated_dp_attention(object())
        self.assertEqual((cfg.attn_dp_size, cfg.dp_size, cfg.enable_dp_attention), (3, 1, False))

    def test_actual_native_parallel_hook_divides_raw_chunk_once(self):
        import sglang.srt.arg_groups.parallel_hook as parallel
        from sglang.srt.arg_groups.cuda_graph_hook import Backend
        for dp, outer, want in [(3, 1, 16), (1, 1, 48)]:
            cfg = SimpleNamespace(attn_dp_size=dp, dp_size=outer, tp_size=3, ep_join_mode=None, mm_enable_dp_encoder=False, schedule_conservativeness=1.0, chunked_prefill_size=48, cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=Backend.DISABLED, max_bs=None)), enable_tp_lm_head_all_to_all=False, enable_memory_saver=False)

            def declare(_args, _handler, **changes):
                for key, value in changes.items():
                    setattr(cfg, key, value)
            with patch.object(parallel, 'resolving_view', return_value=cfg), patch.object(parallel, 'resolved_view', return_value=cfg), patch.object(parallel, 'declare_resolution', side_effect=declare), patch.object(parallel, 'run_post_process_pass'):
                parallel.handle_data_parallelism(SimpleNamespace(_cuda_graph_config_locked=set()))
            self.assertEqual(cfg.chunked_prefill_size, want)
        cfg.attn_dp_size = 3
        cfg.dp_size = 3
        with patch.object(parallel, 'resolving_view', return_value=cfg), self.assertRaises(ValueError):
            parallel.handle_data_parallelism(object())

class RealCountsAndMask(unittest.TestCase):

    def test_default_path_identity_has_no_tensor_ops(self):
        t = routes(4)
        with patch('torch.arange', side_effect=AssertionError('default path allocated')):
            self.assertIs(mask_topk_padding(t, None, 1), t)

    def test_max_len_holes_owner_idle_and_valid_bits(self):
        for counts in [[0, 1, 16], [16, 0, 7], [1, 0, 2]]:
            t = routes()
            out = mask_topk_padding(t, batch(counts), 3)
            keep = torch.tensor([j < n for n in counts for j in range(16)])
            self.assertTrue(torch.equal(out.topk_ids[keep], t.topk_ids[keep]))
            self.assertTrue(torch.equal(out.topk_weights[keep].view(torch.int32), t.topk_weights[keep].view(torch.int32)))
            self.assertTrue((out.topk_ids[~keep] == -1).all())
            self.assertTrue((out.topk_weights[~keep] == 0).all())
            self.assertIs(out.router_logits, t.router_logits)

    def test_sum_len_exact_pack_and_empty(self):
        t = routes(3)
        b = batch([1, 0, 2], DpPaddingMode.SUM_LEN)
        self.assertIs(mask_topk_padding(t, b, 3), t)
        with self.assertRaises(ValueError):
            mask_topk_padding(routes(4), b, 3)
        self.assertIs(mask_topk_padding((t := routes(0)), batch([0, 0, 0]), 3), t)

    def test_scalar_prefix_is_not_the_dp_mask(self):
        out = mask_topk_padding(routes(12), batch([1, 0, 2]), 3)
        valid = (out.topk_ids[:, 0] >= 0).nonzero().flatten().tolist()
        self.assertEqual(valid, [0, 8, 9])
        self.assertNotEqual(valid, list(range(3)))

    def test_unknown_counts_modes_and_capacity_fail_closed(self):
        for counts in [None, 3, [1, 2], [1, True, 2], [-1, 0, 2], [17, 0, 0]]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                copy_real_dp_counts_(torch.empty(3, dtype=torch.int32), SimpleNamespace(original_global_num_tokens_cpu=counts), 16)
        b = batch([1, 0, 2])
        b.dp_padding_mode = None
        with self.assertRaises(ValueError):
            mask_topk_padding(routes(), b, 3)
        b = batch([1, 0, 2])
        b.expertplex_dp_real_num_tokens_gpu = None
        with self.assertRaises(ValueError):
            mask_topk_padding(routes(), b, 3)
        with self.assertRaises(ValueError):
            mask_topk_padding(routes(47), batch([1, 0, 2]), 3)

    def test_actual_registry_refreshes_raw_counts_and_preserves_native_padded(self):
        reg = registry()
        buf = reg.get_slot('expertplex_dp_real_num_tokens_gpu').buffer
        ptr = buf.data_ptr()
        for counts in [[1, 0, 2], [0, 7, 16]]:
            reg.fill_from(batch(counts), raw_bs=1, padded_bs=16, raw_num_tokens=1, padded_num_tokens=16)
            self.assertEqual(buf.tolist(), counts)
            self.assertEqual(buf.data_ptr(), ptr)
            self.assertEqual(reg.get_slot('global_num_tokens_gpu').buffer.tolist(), [16] * 3)
            self.assertEqual(reg.get_slot('global_num_tokens_for_logprob_gpu').buffer.tolist(), [16] * 3)
        with self.assertRaises(ValueError):
            reg.fill_from(SimpleNamespace(), raw_bs=1, padded_bs=16, raw_num_tokens=1, padded_num_tokens=16)

    def test_default_registry_has_no_remote_slot(self):
        with self.assertRaises(KeyError):
            registry(False).get_slot('expertplex_dp_real_num_tokens_gpu')

    def test_eager_adapter_uses_actual_raw_cpu_counts(self):
        b = batch([0, 1, 2])
        buffer = make_eager_real_dp_counts(b, 3)
        self.assertEqual(buffer.tolist(), [0, 1, 2])
        self.assertEqual(buffer.dtype, torch.int32)

    def test_fx_where_keeps_dynamic_count_input_and_valid_bits(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        t = routes(12)

        def f(ids, weights, counts):
            b = batch([1, 0, 2], raw=counts)
            out = mask_topk_padding(StandardTopKOutput(weights, ids, None), b, 3)
            return (out.topk_ids, out.topk_weights)
        fx = make_fx(f)(t.topk_ids, t.topk_weights, torch.tensor([1, 0, 2], dtype=torch.int32))
        self.assertNotIn('_local_scalar_dense', str(fx.graph))
        for counts in [[1, 0, 2], [0, 2, 1]]:
            ids, w = fx(t.topk_ids, t.topk_weights, torch.tensor(counts, dtype=torch.int32))
            keep = torch.tensor([j < n for n in counts for j in range(4)])
            self.assertTrue((ids[~keep] == -1).all())
            self.assertTrue(torch.equal(w[keep].view(torch.int32), t.topk_weights[keep].view(torch.int32)))

    def test_actual_qwen_forward_masks_only_explicit_group(self):
        from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
        hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
        t = routes(12)
        b = batch([1, 0, 2])
        b.moe_num_token_non_padded = lambda: None
        seen = []
        block = SimpleNamespace(_expertplex=True, _expertplex_attention_dp_size=3, gate=lambda h: (torch.zeros((12, 160)), None), topk=lambda *a, **kw: t, experts=lambda h, r: seen.append(r) or h)
        result = Qwen3MoeSparseMoeBlock.forward_normal(block, hidden, b)
        self.assertIs(result._base, hidden._base)
        self.assertEqual((seen[0].topk_ids[:, 0] >= 0).nonzero().flatten().tolist(), [0, 8, 9])
        block._expertplex_attention_dp_size = 1
        Qwen3MoeSparseMoeBlock.forward_normal(block, hidden, b)
        self.assertIs(seen[-1], t)

    def test_gather_single_owner_sum_scatter_has_no_factor_three(self):
        from sglang.srt.layers.moe.token_dispatcher.expertplex import ExpertPlexDispatcher, ExpertPlexCombineInput
        from sglang.srt.layers.layer_boundary import ops
        from sglang.srt.layers.layer_boundary.layout import SumGroup
        import sglang.srt.runtime_context as rt
        local = [torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) + rank * 10 for rank, n in enumerate([0, 2, 3])]
        gathered = torch.cat(local)
        transport = SimpleNamespace(begin=Mock(side_effect=lambda layer, hidden, ids, weights, **kw: SimpleNamespace(output=hidden * 2)), finish=Mock(side_effect=lambda pending: pending.output))
        outputs = []
        with patch('sglang.srt.expertplex.client.get_client', return_value=transport):
            for rank in range(3):
                with patch.object(rt, 'get_parallel', return_value=SimpleNamespace(tp_rank=rank)):
                    dispatcher = ExpertPlexDispatcher(layer_id=7)
                dispatched = dispatcher.dispatch(gathered, routes(gathered.shape[0]))
                combined = ExpertPlexCombineInput(dispatched.pending)
                outputs.append(dispatcher.combine(combined))
                if rank:
                    self.assertEqual(torch.count_nonzero(outputs[-1]).item(), 0)
                    with self.assertRaises(RuntimeError):
                        dispatcher.combine(combined)
        self.assertEqual(transport.begin.call_count, 1)
        self.assertEqual(transport.finish.call_count, 1)
        self.assertIs(transport.begin.call_args.args[1], gathered)
        self.assertEqual(transport.begin.call_args.kwargs, {'mailbox': 0})
        summed = sum(outputs)
        with patch.object(ops, 'get_parallel', return_value=SimpleNamespace()), patch.object(ops, 'sum_post_experts_output', return_value=summed) as reduce_output:
            result = ops.sum_output(outputs[0], SumGroup.MOE_OUTPUT, SimpleNamespace(), may_quantize=False)
        reduce_output.assert_called_once_with(outputs[0])
        self.assertTrue(all((torch.equal(a, b * 2) for a, b in zip(result.split([0, 2, 3]), local))))
        self.assertFalse(torch.equal(result * 3, result))

def tearDownModule():
    try:
        assert not torch.cuda.is_initialized(), "CPU test initialized CUDA"
    finally:
        _temporary.cleanup()

if __name__ == "__main__":
    unittest.main()
