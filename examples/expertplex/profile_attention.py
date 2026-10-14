"""Profile the actual SGLang Qwen3 attention-side layer on one B300.

This isolated component experiment omits routed experts and substitutes a zero
MoE result AFTER the timed interval. Norms, QKV, QK normalization, RoPE,
attention, output projection and router/top-k use the regular implementation.
It measures shapes with synthetic activations, not pretrained accuracy or
real-model routing probabilities. Embedding, LM head and sampler are excluded.
Prefill is eager (events include submission gaps); decode uses SGLang graphs.
No service process can enable this diagnostic: patches live only in this script.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

from _bootstrap import module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--client-id", type=int, default=1)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--kv-tokens", type=int, default=98304)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if min(*args.batches, *args.lengths, args.iterations, args.kv_tokens) <= 0:
        p.error("Profile sizes must be positive")
    config = module("config")
    source_node = config.load_config(args.config)
    client = source_node.client(args.client_id)
    if client.phase != "decode":
        p.error("Select a D client so SGLang initializes decode graphs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    node_data = source_node.to_dict() | {"overlap": "none"}
    node_path = args.output_dir / "node.json"
    node_path.write_text(json.dumps(node_data, indent=2) + "\n")
    os.environ["SGLANG_EXPERTPLEX_CONFIG"] = str(node_path.resolve())
    os.environ["SGLANG_EXPERTPLEX_CLIENT_ID"] = str(client.id)
    import numpy as np
    import torch

    from sglang.benchmark import one_batch as bench
    from sglang.srt.models.qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeSparseMoeBlock,
    )
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.server_args import ServerArgs

    torch.set_num_threads(8)
    torch.cuda.set_device(client.device)
    np.random.seed(20261015)
    spec = config.ModelSpec.from_model_path(source_node.model_path)
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(spec.num_layers)
    ]
    original = Qwen3MoeDecoderLayer.forward

    def timed_layer(self, *forward_args, **forward_kwargs):
        events[self.layer_id][0].record()
        return original(self, *forward_args, **forward_kwargs)

    def attention_only_mlp(self, hidden_states, forward_batch=None):
        if hidden_states.numel():
            router_logits, _ = self.gate(hidden_states)
            self.topk(hidden_states, router_logits)
        events[self.layer_id][1].record()
        return torch.zeros_like(hidden_states)

    Qwen3MoeDecoderLayer.forward = timed_layer
    Qwen3MoeSparseMoeBlock.forward = attention_only_mlp
    report = {
        "scope": __doc__,
        "source_node": source_node.to_dict(),
        "settings": vars(args)
        | {"config": str(args.config), "output_dir": str(args.output_dir)},
        "samples": [],
        "skipped": [],
    }

    def layer_seconds():
        return [start.elapsed_time(end) / 1000 for start, end in events]

    def profile(server_args, port_args, bench_args, _gpu_id, tp_rank):
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
                gpu_id=client.device,
            ),
        )
        bench._init_process_global_configs()
        runner, _ = bench.load_model(server_args, port_args, client.device, tp_rank)
        for batch_size in args.batches:
            for length in args.lengths:
                if batch_size * (length + args.iterations + 4) > args.kv_tokens:
                    report["skipped"].append(
                        {"batch": batch_size, "length": length, "reason": "KV capacity"}
                    )
                    continue
                phase_samples = {"prefill": [], "decode": []}
                for repeat in range(args.iterations + 2):
                    runner.clear()
                    reqs = bench.prepare_synthetic_inputs_for_latency_test(
                        batch_size, length
                    )
                    next_ids, _, batch = runner.extend(reqs)
                    runner.synchronize()
                    if repeat >= 2:
                        phase_samples["prefill"].append(layer_seconds())
                    # At least two steps exercise the warmed decode graph.
                    for step in range(3):
                        next_ids, _ = runner.decode(next_ids, batch)
                        runner.synchronize()
                        if repeat >= 2 and step == 2:
                            phase_samples["decode"].append(layer_seconds())
                for phase, values in phase_samples.items():
                    row = {
                        "phase": phase,
                        "x": batch_size,
                        "sequence_length": length if phase == "prefill" else length + 2,
                        "latency_s": statistics.median(
                            statistics.mean(v) for v in values
                        ),
                        "layer_samples_s": values,
                    }
                    report["samples"].append(row)
                    print(phase, batch_size, length, row["latency_s"], flush=True)
                (args.output_dir / "attention.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        bench.destroy_model_parallel()
        bench.destroy_distributed_environment()

    # Reuse the pinned benchmark's argument resolution and model initialization.
    bench.latency_test = profile
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    cli = [
        "--model-path",
        source_node.model_path,
        "--tokenizer-path",
        args.tokenizer_path,
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--base-gpu-id",
        str(client.device),
        "--load-format",
        "dummy" if source_node.weight_mode == "synthetic" else "safetensors",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disable-overlap-schedule",
        "--disable-flashinfer-autotune",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        str(max(args.batches)),
        "--max-running-requests",
        str(source_node.max_decode_tokens),
        "--max-prefill-tokens",
        str(source_node.max_decode_tokens),
        "--chunked-prefill-size",
        str(source_node.max_decode_tokens),
        "--max-total-tokens",
        str(args.kv_tokens),
        "--mem-fraction-static",
        "0.75",
        "--context-length",
        str(max(args.lengths) + args.iterations + 4),
    ]
    report["server_arguments"] = cli
    server = ServerArgs.from_cli_args(parser.parse_args(cli))
    try:
        bench.main(server, bench.BenchArgs(batch_size=tuple(args.batches)))
        fit = module("optimizer").fit_coefficients
        from dataclasses import asdict

        report["fits"] = {
            phase: asdict(
                fit(
                    [s for s in report["samples"] if s["phase"] == phase],
                    component="attention",
                )
            )
            for phase in ("prefill", "decode")
        }
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "attention.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
