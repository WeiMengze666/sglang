"""Exercise all shared expert weights and all attention slots without a tokenizer.

This validates the MoE deployment; it does not execute attention or evaluate a
pretrained model. Synthetic weights must be explicitly enabled in the config.
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from _bootstrap import module


def attention(config_path, slot, iterations, rows, barrier, output, compare_group):
    try:
        import torch

        config = module("config").load_config(config_path)
        torch.cuda.set_device(config.client(slot).device)
        client = module("client").ExpertPlexClient(config_path, slot)
        group_templates = (
            [client.native.RemoteGroup(peers) for peers in client.mailboxes]
            if compare_group
            else client.groups
        )
        spec = client.spec
        gen = torch.Generator(device=client.device).manual_seed(1234 + slot)
        x = torch.randn(
            rows,
            spec.hidden_size,
            generator=gen,
            device=client.device,
            dtype=torch.bfloat16,
        )
        scores = torch.rand(rows, spec.num_experts, generator=gen, device=client.device)
        ids = scores.topk(spec.top_k, dim=-1).indices.int().contiguous()
        weights = torch.full((rows, spec.top_k), 1 / spec.top_k, device=client.device)
        x_other = (x * 0.75).contiguous()
        ids_other = ((ids + 1) % spec.num_experts).contiguous()
        # Warm up allocator and CUDA module loading before starting the barrier.
        client.run(0, x, ids, weights)
        torch.cuda.synchronize(client.device)
        barrier.wait(timeout=600)
        latencies = []
        overlap_checks = 0
        for iteration in range(iterations):
            start = time.perf_counter()
            for layer in range(spec.num_layers):
                if config.overlap == "tbo":
                    if compare_group:
                        client.groups = []
                    expected_a = client.run(layer, x, ids, weights)
                    expected_b = client.run(layer, x_other, ids_other, weights)
                    if compare_group:
                        client.groups = group_templates
                    a = client.begin(layer, x, ids, weights, mailbox=0)
                    b = client.begin(layer, x_other, ids_other, weights, mailbox=1)
                    actual_a, y = client.finish(a), client.finish(b)
                    torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
                    torch.testing.assert_close(y, expected_b, rtol=0, atol=0)
                    overlap_checks += 2
                else:
                    y = client.run(layer, x, ids, weights)
            torch.cuda.synchronize(client.device)
            if not torch.isfinite(y).all():
                raise AssertionError("Nonfinite MoE result")
            latencies.append((time.perf_counter() - start) * 1000)
        graph_checks = 0
        if config.overlap == "tbo":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                a = client.begin(0, x, ids, weights, mailbox=0)
                b = client.begin(0, x_other, ids_other, weights, mailbox=1)
                graph_a, graph_b = client.finish(a), client.finish(b)
            for replay in range(10):
                x.mul_(0.99)
                x_other.mul_(1.01)
                if compare_group:
                    client.groups = []
                expected_a = client.run(0, x, ids, weights)
                expected_b = client.run(0, x_other, ids_other, weights)
                if compare_group:
                    client.groups = group_templates
                graph.replay()
                torch.testing.assert_close(graph_a, expected_a, rtol=0, atol=0)
                torch.testing.assert_close(graph_b, expected_b, rtol=0, atol=0)
                graph_checks += 2
        client.close()
        output.put(
            dict(
                slot=slot,
                phase=config.client(slot).phase,
                rows=rows,
                layers=spec.num_layers,
                full_moe_pass_ms=latencies,
                exact_overlap_comparisons=overlap_checks,
                exact_graph_comparisons=graph_checks,
                group_vs_peer_reference=compare_group,
                status="passed",
            )
        )
    except BaseException:
        output.put(dict(slot=slot, status="failed", error=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--prefill-rows", type=int, default=128)
    p.add_argument("--decode-rows", type=int, default=8)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--compare-group-comm", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    config = module("config").load_config(args.config)
    if config.weight_mode != "synthetic":
        p.error("This standalone deployment test requires an explicit synthetic config")
    if args.compare_group_comm and config.overlap != "tbo":
        p.error("Group/reference comparison requires both TBO mailboxes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix(".workers.log").open("w")
    worker = subprocess.Popen(
        [sys.executable, "-m", "sglang.srt.expertplex.worker", "--config", args.config],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    children = []
    report = {
        "scope": "all-layer MoE deployment and capacity only; no attention, no pretrained model accuracy",
        "config": config.to_dict(),
        "native_source_sha256": module("native").source_fingerprint(),
        "environment": {
            k: v for k, v in os.environ.items() if k.startswith("SGLANG_EXPERTPLEX_")
        },
        "results": [],
    }
    try:
        module("rendezvous").wait_for_workers(config)
        report["resident_memory"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv",
            ],
            text=True,
        )
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(len(config.clients))
        output = ctx.Queue()
        for c in config.clients:
            rows = args.prefill_rows if c.phase == "prefill" else args.decode_rows
            proc = ctx.Process(
                target=attention,
                args=(
                    args.config,
                    c.id,
                    args.iterations,
                    rows,
                    barrier,
                    output,
                    args.compare_group_comm,
                ),
            )
            proc.start()
            children.append(proc)
        for _ in children:
            report["results"].append(
                output.get(
                    timeout=config.operation_timeout_s * config.node_gpu_count + 600
                )
            )
        for proc in children:
            proc.join(timeout=30)
            if proc.exitcode != 0:
                raise RuntimeError(f"Attention test exit code {proc.exitcode}")
        if any(r["status"] != "passed" for r in report["results"]):
            raise AssertionError("One or more attention clients failed")
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
        log.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
