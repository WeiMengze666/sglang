"""Exercise deployment limits through the ordinary CPU launch/benchmark CLIs."""

import dataclasses
import importlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "examples/expertplex"
sys.path.insert(0, str(SCRIPTS))
launch = importlib.import_module("launch")


def node(grouped=False):
    clients = [
        launch.config.ClientConfig(
            0, "prefill", 0, 30000, overlap="tbo", max_running_requests=8
        )
    ]
    if grouped:
        clients.append(
            launch.config.ClientConfig(
                1,
                "decode",
                1,
                30001,
                tp_size=3,
                attention_dp_size=3,
                overlap="none",
                max_running_requests=24,
            )
        )
    else:
        clients.extend(
            launch.config.ClientConfig(
                i, "decode", i, 30000 + i, overlap="none", max_running_requests=8
            )
            for i in range(1, 4)
        )
    return launch.config.NodeConfig(
        model_path="/checkpoint",
        clients=tuple(clients),
        expert_devices=(4, 5, 6, 7),
        max_prefill_tokens=49152,
        max_decode_tokens=48 if grouped else 16,
        prefill_preprocess=True,
    )


def options(command):
    result = {}
    key = None
    for word in command[3:]:
        if word.startswith("--"):
            key = word
            result[key] = []
        else:
            result[key].append(word)
    return result


class LaunchCapacityTests(unittest.TestCase):
    def test_explicit_phase_limits_override_exported_limits(self):
        n = node(True)
        a = SimpleNamespace(
            prefill_max_running_requests=16, decode_max_running_requests=48
        )
        self.assertEqual(
            launch.attention_request_limit(n, n.clients[0], a, n.max_prefill_tokens), 16
        )
        self.assertEqual(
            launch.attention_request_limit(n, n.clients[1], a, n.max_decode_tokens), 48
        )
        self.assertEqual(
            launch.attention_request_limit(n, n.clients[1], SimpleNamespace(), 48), 24
        )
        legacy = dataclasses.replace(n.clients[1], max_running_requests=None)
        self.assertEqual(
            launch.attention_request_limit(n, legacy, SimpleNamespace(), 48), 48
        )

    def test_compact_prefill_matches_native_admission_bound(self):
        n = node(True)
        args = SimpleNamespace(prefill_max_running_requests=64)
        self.assertEqual(
            launch.attention_request_limit(n, n.clients[0], args, n.max_prefill_tokens),
            64,
        )
        compact = dataclasses.replace(n, prefill_mailbox_tokens=24576)
        with self.assertRaisesRegex(ValueError, "compact P mailboxes"):
            launch.attention_request_limit(
                compact, compact.clients[0], args, compact.max_prefill_tokens
            )

    def test_invalid_global_limits_fail(self):
        n = node(True)
        for value in (0, -1, True, 49, 47):
            with self.subTest(value=value), self.assertRaises(ValueError):
                launch.attention_request_limit(
                    n,
                    n.clients[1],
                    SimpleNamespace(decode_max_running_requests=value),
                    48,
                )

    def test_KV_cap_counts_attention_DP_ranks(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = root / "node.json"
                config.write_text(json.dumps(node(grouped).to_dict()))
                command = [
                    sys.executable,
                    str(SCRIPTS / "run_serving_experiment.py"),
                    "--variant",
                    "expertplex",
                    "--model-path",
                    "/checkpoint",
                    "--tokenizer-path",
                    "/checkpoint",
                    "--config",
                    str(config),
                    "--weight-mode",
                    "checkpoint",
                    "--dataset-path",
                    "/corpus.jsonl",
                    "--rates",
                    "2",
                    "--ttft-slo-ms",
                    "1000",
                    "--tpot-slo-ms",
                    "50",
                    "--output-dir",
                    str(root / "plan"),
                    "--decode-kv-tokens",
                    "98304",
                    "--dry-run",
                ]
                outer = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                plan = json.loads(outer.stdout)
                inner = subprocess.run(
                    [*plan["launch"], "--dry-run"],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                commands = [shlex.split(line) for line in inner.stdout.splitlines()]
                for command in commands[2:-1]:
                    self.assertEqual(options(command)["--max-total-tokens"], ["32768"])

    def test_benchmark_to_native_P_D_commands_preserve_tuning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "node.json"
            config.write_text(json.dumps(node(True).to_dict()))
            command = [
                sys.executable,
                str(SCRIPTS / "run_serving_experiment.py"),
                "--variant",
                "expertplex",
                "--model-path",
                "/checkpoint",
                "--tokenizer-path",
                "/checkpoint",
                "--config",
                str(config),
                "--weight-mode",
                "checkpoint",
                "--dataset-path",
                "/corpus.jsonl",
                "--rates",
                "2",
                "--ttft-slo-ms",
                "1000",
                "--tpot-slo-ms",
                "50",
                "--output-dir",
                str(root / "plan"),
                "--prefill-max-running-requests",
                "16",
                "--decode-max-running-requests",
                "48",
                "--prefill-mem-fraction-static",
                "0.90",
                "--decode-mem-fraction-static",
                "0.95",
                "--cuda-graph-bs-decode",
                "16",
                "--decode-chunked-prefill-size",
                "48",
                "--enable-deterministic-inference",
                "--dry-run",
            ]
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            plan = json.loads(result.stdout)
            inner = subprocess.run(
                [*plan["launch"], "--dry-run"],
                check=True,
                text=True,
                capture_output=True,
            )
            commands = [shlex.split(line) for line in inner.stdout.splitlines()]
            p, d = options(commands[1]), options(commands[2])
            for row in (p, d):
                self.assertEqual(row["--cuda-graph-bs-decode"], ["16"])
                self.assertIn("--enable-deterministic-inference", row)
                self.assertNotIn("--max-total-tokens", row)
                self.assertNotIn("--disable-overlap-schedule", row)
            self.assertEqual(p["--max-running-requests"], ["16"])
            self.assertEqual(d["--max-running-requests"], ["48"])
            self.assertEqual(p["--mem-fraction-static"], ["0.9"])
            self.assertEqual(d["--mem-fraction-static"], ["0.95"])
            self.assertIn("--enable-two-batch-overlap", p)
            self.assertNotIn("--enable-two-batch-overlap", d)
            self.assertEqual(d["--attn-dp-size"], ["3"])
            self.assertEqual(d["--chunked-prefill-size"], ["48"])
            self.assertEqual(d["--max-prefill-tokens"], ["16"])
            bench = plan["benchmark"]
            self.assertEqual(bench[bench.index("--ttft-slo-ms") + 1], "1000.0")
            self.assertEqual(bench[bench.index("--tpot-slo-ms") + 1], "50.0")


if __name__ == "__main__":
    unittest.main()
