"""Regression for colliding DP-attention control ports in PDD launch plans."""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
baseline = importlib.import_module("launch_baseline")


class BaselineLaunchTests(unittest.TestCase):
    def test_pdd_control_port_blocks_do_not_overlap_and_router_matches(self):
        args = SimpleNamespace(
            variant="pdd",
            model_path="/checkpoint",
            load_format="safetensors",
            decode_kv_tokens=98304,
            attention_backend="triton",
            max_running=128,
            max_prefill_tokens=8192,
            sequence_cap=8192,
            graph_max_bs=32,
            chunk_size=2048,
            skip_tokenizer_init=True,
            tokenizer_path=None,
            disable_tbo=False,
            port=31000,
            nccl_port=22000,
            bootstrap_port=19000,
        )
        prefill, decode, router = baseline.commands(args)
        for server in (prefill, decode):
            self.assertNotIn("--enable-custom-logit-processor", server[1])
            self.assertNotIn("--disable-overlap-schedule", server[1])
            self.assertNotIn("--disable-flashinfer-autotune", server[1])
            self.assertIn("--enable-two-batch-overlap", server[1])
            self.assertEqual(
                server[1][server[1].index("--sampling-backend") + 1], "flashinfer"
            )
        args.enable_dp_lm_head = True
        dp_head = baseline.commands(args)
        self.assertTrue(all("--enable-dp-lm-head" in row[1] for row in dp_head[:2]))
        self.assertNotIn("--enable-dp-lm-head", dp_head[-1][1])
        args.enable_dp_lm_head = False
        args.sampling_backend = "pytorch"
        args.diagnostic_logit_processor = True
        diagnostic = baseline.commands(args)
        for server in diagnostic[:2]:
            self.assertIn("--enable-custom-logit-processor", server[1])
            self.assertIn("--disable-overlap-schedule", server[1])
            self.assertEqual(
                server[1][server[1].index("--sampling-backend") + 1], "pytorch"
            )
        self.assertNotIn("--enable-custom-logit-processor", diagnostic[-1][1])
        args.diagnostic_logit_processor = False
        # The pinned PortArgs reserves dist_init_port + six derived TCP ports.
        p_ports = set(range(prefill[2] + 233, prefill[2] + 241))
        d_ports = set(range(decode[2] + 233, decode[2] + 241))
        self.assertFalse(p_ports & d_ports)
        self.assertIn(f"http://127.0.0.1:{prefill[2]}", router[1])
        self.assertIn(f"http://127.0.0.1:{decode[2]}", router[1])
        for server in (prefill, decode):
            command = server[1]
            self.assertEqual(command[command.index("--tp-size") + 1], "4")
            self.assertEqual(command[command.index("--ep-size") + 1], "4")
        args.sequence_cap = 24576
        args.decode_kv_tokens = 196608
        prefill, decode, _ = baseline.commands(args)
        for server, capacity, graph in [
            (prefill, "24576", "disabled"),
            (decode, "49152", "full"),
        ]:
            command = server[1]
            self.assertEqual(command[command.index("--max-total-tokens") + 1], capacity)
            self.assertEqual(
                command[command.index("--cuda-graph-backend-decode") + 1], graph
            )
        args.max_prefill_tokens = 32768
        prefill, _, _ = baseline.commands(args)
        self.assertEqual(
            prefill[1][prefill[1].index("--max-total-tokens") + 1], "32768"
        )
        args.decode_kv_tokens = None
        args.pdmux_config = "/pdmux.yaml"
        for variant in ("pdd", "pdmux", "colocated", "chunked"):
            args.variant = variant
            for fraction in (None, 0.9):
                args.mem_fraction_static = fraction
                for name, command, _ in baseline.commands(args):
                    if name == "router":
                        continue
                    self.assertNotIn("--max-total-tokens", command)
                    if fraction is None:
                        self.assertNotIn("--mem-fraction-static", command)
                    else:
                        self.assertEqual(
                            command[command.index("--mem-fraction-static") + 1], "0.9"
                        )


if __name__ == "__main__":
    unittest.main()
