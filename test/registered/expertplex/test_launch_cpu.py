"""Launch-plan regressions for single-D layouts and bounded TBO mailboxes."""

import importlib
import io
import json
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
launch = importlib.import_module("launch")


class LaunchPlanTests(unittest.TestCase):
    def test_prefill_graph_capacity_and_native_flag_names(self):
        for backend, limit in (
            ("disabled", 128),
            ("breakable", 0),
            ("breakable", 513),
            ("breakable", True),
            ("piecewise", None),
        ):
            with (
                self.subTest(backend=backend, limit=limit),
                self.assertRaises(ValueError),
            ):
                launch.prefill_graph_options(
                    SimpleNamespace(
                        prefill_graph_backend=backend, prefill_graph_max_tokens=limit
                    ),
                    512,
                )
        self.assertEqual(
            launch.prefill_graph_options(
                SimpleNamespace(
                    prefill_graph_backend="breakable", prefill_graph_max_tokens=128
                ),
                512,
            ),
            [
                "--cuda-graph-backend-prefill",
                "breakable",
                "--cuda-graph-max-bs-prefill",
                "128",
            ],
        )

    def test_router_readiness_waits_for_all_workers(self):
        responses = []
        for count in [1, 4]:
            response = io.BytesIO(json.dumps({"healthy_workers": count}).encode())
            response.status = 200
            responses.append(response)
        with (
            patch.object(
                launch.urllib.request, "urlopen", side_effect=responses
            ) as request,
            patch.object(launch.time, "sleep") as sleep,
        ):
            launch.wait_http(
                "http://router", [], time.monotonic() + 10, expected_workers=4
            )
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args[0], "http://router/readiness")
        sleep.assert_called_once_with(1)

    def test_single_decode_router_and_compact_split_contract(self):
        args = SimpleNamespace(
            config="/node.json",
            bootstrap_port=19000,
            nccl_port=20000,
            cuda_graph=True,
            chunked_prefill=False,
            kv_tokens=65536,
            attention_backend="triton",
            sampling_backend="pytorch",
            ib_device=None,
            skip_tokenizer_init=True,
            tokenizer_path=None,
            sequence_cap=2048,
            host="127.0.0.1",
            port=31000,
        )
        for prefill, decode, policy in [(3, 1, "round_robin"), (1, 3, "power_of_two")]:
            node = launch.config.default_config(
                "/checkpoint",
                prefill_count=prefill,
                decode_count=decode,
                expert_count=4,
                overlap="tbo",
                prefill_mailbox_tokens=1024,
            )
            _, servers, router = launch.commands(node, args)
            for _, command in servers:
                self.assertIn("--enable-expertplex", command)
                self.assertIn("--expertplex-bootstrap-addr", command)
                self.assertNotIn("--expertplex-config", command)
                self.assertNotIn("--expertplex-client-id", command)
                self.assertNotIn("--enable-custom-logit-processor", command)
                self.assertNotIn("--disable-overlap-schedule", command)
                self.assertNotIn("--disable-flashinfer-autotune", command)
                self.assertIn("--enable-two-batch-overlap", command)
            self.assertEqual(router[router.index("--decode-policy") + 1], policy)
            self.assertEqual(router.count("--prefill"), prefill)
            self.assertEqual(router.count("--decode"), decode)
            for client, command in servers:
                self.assertEqual(
                    command[command.index("--tbo-token-distribution-threshold") + 1],
                    "0.5",
                )
                if client.phase == "prefill":
                    self.assertEqual(
                        command[command.index("--max-prefill-tokens") + 1], "2048"
                    )
            args.kv_tokens = None
            args.mem_fraction_static = 0.9
            _, automatic_servers, _ = launch.commands(node, args)
            for _, command in automatic_servers:
                self.assertNotIn("--max-total-tokens", command)
                self.assertEqual(
                    command[command.index("--mem-fraction-static") + 1], "0.9"
                )
            args.mem_fraction_static = None
            _, native_servers, _ = launch.commands(node, args)
            for _, command in native_servers:
                self.assertNotIn("--mem-fraction-static", command)
            args.kv_tokens = 65536
            args.diagnostic_logit_processor = True
            _, diagnostic_servers, diagnostic_router = launch.commands(node, args)
            for _, command in diagnostic_servers:
                self.assertIn("--enable-custom-logit-processor", command)
                self.assertIn("--disable-overlap-schedule", command)
            self.assertNotIn("--enable-custom-logit-processor", diagnostic_router)
            args.diagnostic_logit_processor = False
            args.prefill_graph_backend = "breakable"
            args.prefill_graph_max_tokens = 1024
            with self.assertRaisesRegex(ValueError, "overlap=none"):
                launch.commands(node, args)
            _, graph_servers, _ = launch.commands(
                replace(node, overlap="none", prefill_mailbox_tokens=None), args
            )
            for client, command in graph_servers:
                self.assertEqual(
                    command[command.index("--cuda-graph-backend-prefill") + 1],
                    "breakable" if client.phase == "prefill" else "disabled",
                )
                self.assertEqual(
                    "--cuda-graph-max-bs-prefill" in command, client.phase == "prefill"
                )
                self.assertNotIn("--disable-overlap-schedule", command)
            args.prefill_graph_backend = "disabled"
            args.prefill_graph_max_tokens = None


if __name__ == "__main__":
    unittest.main()
