"""Generation/lifecycle and joint-SLO checks which need no GPU dependencies."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from _bootstrap import module

config = module("config")
rv = module("rendezvous")
bench = importlib.import_module("bench_sweep")


class RendezvousTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.config = config.default_config(
            str(self.model), rendezvous_dir=str(root / "run"), startup_timeout_s=0.01
        )
        self.root = Path(self.config.rendezvous_dir)

    def ready_run(self):
        manifest = rv.begin_run(self.config)
        for rank, device in enumerate(self.config.expert_devices):
            rv.atomic_json(
                self.root / f"expert-{rank}.json",
                {
                    "run_id": manifest["run_id"],
                    "rank": rank,
                    "device": device,
                    "pid": os.getpid(),
                    "buffers": {},
                },
            )
        manifest["state"] = "ready"
        rv.atomic_json(self.root / "run.json", manifest)
        return manifest

    def test_atomic_publish_and_exclusive_supervisor(self):
        self.ready_run()
        manifest, peers = rv.wait_for_workers(self.config)
        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(len(peers), 4)
        with self.assertRaisesRegex(RuntimeError, "already owned"):
            rv.begin_run(self.config)

    def test_configuration_mismatch_fails_before_importing_ipc(self):
        self.ready_run()
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            rv.wait_for_workers(replace(self.config, max_decode_tokens=128))

    def test_changed_checkpoint_cannot_import_live_workers(self):
        self.ready_run()
        (self.model / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
            rv.wait_for_workers(self.config)

    def test_old_generation_descriptor_cannot_be_reused(self):
        self.ready_run()
        peer = self.root / "expert-0.json"
        data = json.loads(peer.read_text())
        data["run_id"] = "old-generation"
        rv.atomic_json(peer, data)
        with self.assertRaises(TimeoutError):
            rv.wait_for_workers(self.config)

    def test_failed_run_fails_immediately(self):
        manifest = self.ready_run()
        manifest["state"] = "failed"
        rv.atomic_json(self.root / "run.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            rv.wait_for_workers(self.config)

    def test_two_attention_processes_cannot_own_the_same_slot(self):
        manifest = self.ready_run()
        rv.claim_client(self.config, manifest, 0)
        with self.assertRaisesRegex(RuntimeError, "already has an owner"):
            rv.claim_client(self.config, manifest, 0)
        rv.claim_client(self.config, manifest, 1)


class JointSloTests(unittest.TestCase):
    def result(self):
        return {
            "request_rate": 4,
            "duration": 10,
            "ttfts": [0.1] * 10,
            "latencies": [0.2] * 10,
            "successes": [True] * 10,
            "output_lens": [3] * 10,
        }

    def test_joint_attainment_is_not_independent_p90(self):
        result = self.result()
        result["ttfts"][0] = 1.1
        result["latencies"][0] = 1.2
        result["latencies"][1] = 1.1
        measured = bench.joint_slo(result, 1.0, 0.1)
        self.assertEqual(measured["joint_slo_attainment"], 0.8)
        self.assertEqual(measured["observed_slo_goodput_rps"], 0.8)
        self.assertFalse(measured["meets_p90_joint_slo"])

    def test_failures_and_one_token_outputs_stay_in_denominator(self):
        result = self.result()
        result["successes"][0] = False
        result["output_lens"][1] = 1
        measured = bench.joint_slo(result, 1.0, 0.1)
        self.assertEqual(measured["completed_requests"], 9)
        self.assertEqual(measured["joint_slo_met_requests"], 8)
        self.assertEqual(measured["unscorable_one_token_responses"], 1)

    def test_missing_or_misaligned_details_are_rejected(self):
        result = self.result()
        result["successes"].pop()
        with self.assertRaises(ValueError):
            bench.joint_slo(result, 1.0, 0.1)


if __name__ == "__main__":
    unittest.main()
