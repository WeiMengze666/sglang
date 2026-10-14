"""Launch one shared, persistent expert executor on each expert GPU."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import signal
import time
from pathlib import Path

from .config import ModelSpec, load_config
from .rendezvous import atomic_json, begin_run, checkpoint_fingerprint

logger = logging.getLogger(__name__)
_EXPERT_WEIGHT = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def load_expert_weights(model_path, spec, rank, world, device, synthetic_seed=None):
    import torch
    from safetensors import safe_open

    local_experts = spec.num_experts // world
    w13 = torch.empty(
        (spec.num_layers, local_experts, 2 * spec.intermediate_size, spec.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.empty(
        (spec.num_layers, local_experts, spec.hidden_size, spec.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    if synthetic_seed is not None:
        # Explicit synthetic mode: no checkpoint shards are read or fabricated.
        # Independent expert seeds avoid identical experts and routing artifacts.
        for layer in range(spec.num_layers):
            for local in range(local_experts):
                seed = synthetic_seed + layer * spec.num_experts + local * world + rank
                generator = torch.Generator(device=device).manual_seed(seed)
                w13[layer, local].uniform_(-1e-3, 1e-3, generator=generator)
                w2[layer, local].uniform_(-1e-3, 1e-3, generator=generator)
        torch.cuda.synchronize(device)
        logger.warning(
            "SYNTHETIC weights: geometry/capacity testing only, not model accuracy"
        )
        return w13, w2
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        names = {
            shard
            for name, shard in weight_map.items()
            if (match := _EXPERT_WEIGHT.fullmatch(name))
            and int(match[2]) % world == rank
        }
        shards = sorted(root / name for name in names)
    else:
        shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("No routed expert safetensors shards found")
    seen = set()
    for shard in shards:
        if not shard.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Checkpoint shard escapes model directory: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                match = _EXPERT_WEIGHT.fullmatch(name)
                if match is None:
                    continue
                layer, expert, projection = int(match[1]), int(match[2]), match[3]
                if expert % world != rank:
                    continue
                if not 0 <= layer < spec.num_layers or expert >= spec.num_experts:
                    raise ValueError(f"Out-of-range expert weight: {name}")
                key = (layer, expert, projection)
                if key in seen:
                    raise ValueError(f"Duplicate expert weight: {name}")
                tensor = stream.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Expected BF16 checkpoint tensor: {name}")
                local = expert // world
                if projection == "down_proj":
                    destination = w2[layer, local]
                else:
                    start = spec.intermediate_size if projection == "up_proj" else 0
                    destination = w13[
                        layer, local, start : start + spec.intermediate_size
                    ]
                if tensor.shape != destination.shape:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {name}: "
                        f"{tuple(tensor.shape)} != {tuple(destination.shape)}"
                    )
                destination.copy_(tensor)
                seen.add(key)
    expected = spec.num_layers * local_experts * 3
    if len(seen) != expected:
        raise ValueError(f"Incomplete expert weights: loaded {len(seen)} / {expected}")
    torch.cuda.synchronize(device)
    logger.info("Expert rank %s loaded %s tensors", rank, len(seen))
    return w13, w2


def expert_main(config_path, rank, manifest, stop_event):
    import torch

    from .native import (
        OwnedBuffers,
        PersistentExecutor,
        estimate_buffer_bytes,
        load_extension,
    )

    logging.basicConfig(level=logging.INFO)
    config = load_config(config_path)
    spec = ModelSpec.from_model_path(config.model_path)
    config.validate(spec)
    device = config.expert_devices[rank]
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 3):
        raise RuntimeError("This ExpertPlex native executor targets B300 / SM103 only")
    if spec.hidden_size % 16 or spec.intermediate_size % 16:
        raise ValueError("WMMA expert dimensions must be multiples of 16")
    logger.info("Expert rank %s uses CUDA %s (SM%s%s)", rank, device, major, minor)
    if checkpoint_fingerprint(config.model_path) != manifest["checkpoint_fingerprint"]:
        raise RuntimeError("Checkpoint changed while starting ExpertPlex workers")
    load_extension()  # Compile before allocating / publishing any IPC handles.
    local_experts = spec.num_experts // len(config.expert_devices)
    capacity = max(config.max_prefill_tokens, config.max_decode_tokens)
    capacities, packed = config.buffer_layout()
    buffer_bytes = estimate_buffer_bytes(
        config.slot_count,
        local_experts,
        capacity,
        spec.hidden_size,
        spec.intermediate_size,
        capacities=capacities,
        packed=packed,
        top_k=spec.top_k,
    )["total"]
    required_bytes = (
        spec.expert_weight_bytes // len(config.expert_devices) + buffer_bytes
    )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Reserve room for the CUDA context, extension scratch and other libraries.
    reserve_bytes = 8 * 2**30
    if required_bytes + reserve_bytes > free_bytes:
        raise RuntimeError(
            f"Expert GPU {device} needs at least "
            f"{(required_bytes + reserve_bytes) / 2**30:.1f} GiB free "
            f"including an 8 GiB reserve; found {free_bytes / 2**30:.1f} GiB. "
            "Reduce the mailbox capacity or use more expert GPUs."
        )
    w13, w2 = load_expert_weights(
        config.model_path,
        spec,
        rank,
        len(config.expert_devices),
        device,
        synthetic_seed=config.synthetic_seed
        if config.weight_mode == "synthetic"
        else None,
    )
    buffers = OwnedBuffers(
        slot_count=config.slot_count,
        local_experts=spec.num_experts // len(config.expert_devices),
        max_rows=max(config.max_prefill_tokens, config.max_decode_tokens),
        hidden=spec.hidden_size,
        intermediate=spec.intermediate_size,
        device=device,
        capacities=capacities,
        packed=packed,
        top_k=spec.top_k,
    )
    executor = PersistentExecutor(
        buffers,
        w13,
        w2,
        decode_sm_budget=config.decode_sm_budget,
        expected_decode_tiles=config.expected_decode_tiles,
        max_decode_sm_fraction=config.max_decode_sm_fraction,
        ctas=config.cta_count,
        kernel=config.kernel,
    )
    descriptor_path = Path(config.rendezvous_dir) / f"expert-{rank}.json"
    atomic_json(
        descriptor_path,
        {
            "protocol": manifest["protocol"],
            "run_id": manifest["run_id"],
            "rank": rank,
            "device": device,
            "pid": os.getpid(),
            "buffers": buffers.descriptor(),
            "executor": executor.describe(),
        },
    )
    logger.info(
        "Expert rank %s persistent executor is ready: %s", rank, executor.describe()
    )
    try:
        while not stop_event.wait(0.25):
            errors = executor.check_error()
            if any(errors):
                raise RuntimeError(f"Expert rank {rank} protocol error: {errors}")
    finally:
        executor.stop()
        descriptor_path.unlink(missing_ok=True)


def supervise(config_path: str):
    config = load_config(config_path)
    config.validate(ModelSpec.from_model_path(config.model_path))
    manifest = begin_run(config)
    root = Path(config.rendezvous_dir)
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()

    def request_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    children = []
    failed = False
    try:
        for rank in range(len(config.expert_devices)):
            process = context.Process(
                target=expert_main,
                args=(config_path, rank, manifest, stop_event),
                name=f"expertplex-expert-{rank}",
            )
            process.start()
            children.append(process)
        deadline = time.monotonic() + config.startup_timeout_s
        while not stop_event.wait(0.25):
            exited = [child for child in children if child.exitcode is not None]
            if exited:
                failed = True
                raise RuntimeError(
                    "Expert worker exited: "
                    + ", ".join(f"{p.name}: {p.exitcode}" for p in exited)
                )
            if manifest["state"] == "starting":
                descriptors = [root / f"expert-{i}.json" for i in range(len(children))]
                ready = all(
                    path.exists()
                    and json.loads(path.read_text()).get("run_id") == manifest["run_id"]
                    for path in descriptors
                )
                if ready:
                    manifest["state"] = "ready"
                    atomic_json(root / "run.json", manifest)
                    logger.info("All shared expert workers ready")
                elif time.monotonic() > deadline:
                    failed = True
                    raise TimeoutError("Expert worker startup timed out")
    finally:
        stop_event.set()
        manifest["state"] = "failed" if failed else "stopped"
        atomic_json(root / "run.json", manifest)
        for child in children:
            child.join(timeout=5)
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
        (root / "owner.lock").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    supervise(str(Path(args.config).resolve()))


if __name__ == "__main__":
    main()
