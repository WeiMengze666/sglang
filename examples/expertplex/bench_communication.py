"""Single-node adaptation of paper section 7.5: dispatch/combine separately.

Ranks 0..3 are attention clients and 4..7 hold 40 active experts each. DeepEP
requires equal expert ranges on all ranks, so its first 160 expert IDs are
unused and the 160 actual expert IDs are mapped onto ranks 4..7. Prefill uses
client 0; decode uses clients 1..3. This is an 8-GPU NVLink adaptation, not the
paper's 16-GPU hierarchical RDMA experiment. Echo work is outside timing.
"""

import argparse
import datetime
import json
import os
import statistics
import time
from pathlib import Path


def worker(rank, settings):
    import torch
    import torch.distributed as dist

    args = argparse.Namespace(**settings)
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=180),
    )
    from _bootstrap import module

    native = module("native")
    extension = native.load_extension() if args.backend == "expertplex" else None
    reports = []
    for rows in args.rows:
        if rank == 0:
            print(
                f"BEGIN rows={rows} backend={args.backend} mode={args.mode}", flush=True
            )
        active = rank == 0 if args.mode == "normal" else 1 <= rank <= 3
        tokens = rows if active else 0
        generator = torch.Generator(device=rank).manual_seed(args.seed + rank + rows)
        x = torch.randn(
            tokens, args.hidden, generator=generator, device=rank, dtype=torch.bfloat16
        )
        ids = (
            torch.rand(tokens, args.experts * 4, generator=generator, device=rank)
            .topk(args.top_k, dim=-1)
            .indices
        )
        weights = torch.full(
            ids.shape, 1 / args.top_k, device=rank, dtype=torch.float32
        )
        owner, remotes, buffer = None, [], None
        row_maps = []
        remote_group = None
        metadata = {}
        if args.backend == "expertplex":
            if rank >= 4:
                owner = native.OwnedBuffers(
                    4,
                    args.experts,
                    rows,
                    args.hidden,
                    64,
                    rank,
                    packed=[int(args.mode == "normal")] * 4,
                    top_k=args.top_k,
                    prefill_preprocess=args.prefill_preprocess,
                    expert_rank=rank - 4,
                    expert_world=4,
                )
            descriptors = [None] * 8
            dist.all_gather_object(descriptors, owner.descriptor() if owner else None)
            if rank < 4:
                remotes = [
                    native.RemoteBuffers(descriptors[e], rank, rank)
                    for e in range(4, 8)
                ]
                row_maps = [
                    torch.empty(ids.shape, dtype=torch.int32, device=rank)
                    for _ in remotes
                ]
            if args.group_transfers and remotes:
                remote_group = native.RemoteGroup(remotes)
                row_maps = [row_maps[0]] * len(remotes)
            output = torch.zeros(tokens, args.hidden, dtype=torch.float32, device=rank)
            metadata["native_source_sha256"] = native.source_fingerprint()
        else:
            import deep_ep

            Buffer = deep_ep.Buffer
            total_experts = args.experts * 8
            mapped = ((ids % 4 + 4) * args.experts + ids // 4).contiguous()
            low_latency = args.mode == "low_latency"
            nvlink_bytes = (
                0
                if low_latency
                else max(
                    Buffer.get_dispatch_config(8).get_nvl_buffer_size_hint(
                        args.hidden * 2, 8
                    ),
                    Buffer.get_combine_config(8).get_nvl_buffer_size_hint(
                        args.hidden * 2, 8
                    ),
                )
            )
            rdma_bytes = (
                Buffer.get_low_latency_rdma_size_hint(
                    rows, args.hidden, 8, total_experts
                )
                if low_latency
                else 0
            )
            buffer = Buffer(
                dist.group.WORLD,
                nvlink_bytes,
                rdma_bytes,
                low_latency_mode=low_latency,
                num_qps_per_rank=args.experts,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
            )
            metadata.update(
                deepep_module=deep_ep.__file__,
                deepep_version=getattr(deep_ep, "__version__", None),
                nvlink_buffer_bytes=nvlink_bytes,
                rdma_buffer_bytes=rdma_bytes,
                deepep_source_revision=args.deepep_revision,
            )
        samples = []
        recv_x = recv_ids = recv_weights = handle = None

        def dispatch():
            nonlocal recv_x, recv_ids, recv_weights, handle
            if remote_group is not None:
                remote_group.dispatch(
                    x, ids, row_maps[0], 0, int(args.mode == "low_latency"), weights
                )
            elif args.backend == "expertplex":
                for expert, (remote, row_map) in enumerate(zip(remotes, row_maps)):
                    native.dispatch(
                        remote,
                        x,
                        ids,
                        row_map,
                        0,
                        0,
                        int(args.mode == "low_latency"),
                        expert,
                        4,
                    )
            elif args.mode == "normal":
                per_rank, per_rdma, per_expert, in_rank, _ = buffer.get_dispatch_layout(
                    mapped, total_experts
                )
                recv_x, recv_ids, recv_weights, _, handle, _ = buffer.dispatch(
                    x,
                    num_tokens_per_rank=per_rank,
                    num_tokens_per_rdma_rank=per_rdma,
                    num_tokens_per_expert=per_expert,
                    is_token_in_rank=in_rank,
                    topk_idx=mapped,
                    topk_weights=weights,
                    # v1's supported uncached, graph-compatible dynamic route
                    # path; no frozen handle or omitted routing metadata.
                    num_worst_tokens=rows if args.cuda_graph else 0,
                )
            else:
                recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
                    x, mapped, rows, total_experts, use_fp8=False
                )

        def echo():
            nonlocal recv_x
            if args.backend == "expertplex":
                if owner is not None:
                    extension.benchmark_echo(owner._impl)
                output.zero_()
            elif args.mode == "normal":
                recv_x = (
                    recv_x.float()
                    * torch.where(recv_ids >= 0, recv_weights, 0).sum(-1, keepdim=True)
                ).bfloat16()

        def combine():
            nonlocal output
            if remote_group is not None:
                remote_group.combine(output, ids, weights, row_maps[0])
            elif args.backend == "expertplex":
                for expert, (remote, row_map) in enumerate(zip(remotes, row_maps)):
                    native.combine(
                        remote,
                        output,
                        ids,
                        weights,
                        row_map,
                        0,
                        expert,
                        4,
                        synchronize=False,
                    )
            elif args.mode == "normal":
                output, _, _ = buffer.combine(recv_x, handle)
            else:
                output, _, _ = buffer.low_latency_combine(
                    recv_x, mapped, weights, handle
                )

        def identity(stage):
            expected = x.float() * (weights * (ids >= 0)).sum(-1, keepdim=True)
            failure = None
            if active and not torch.allclose(
                output.float(), expected, atol=0.01, rtol=0.03
            ):
                failure = {
                    "rank": rank,
                    "rows": rows,
                    "stage": stage,
                    "max_abs_difference": (output.float() - expected)
                    .abs()
                    .max()
                    .item(),
                    "finite_outputs": bool(torch.isfinite(output).all()),
                }
            # All ranks enter cleanup together on a numerical failure. Without
            # consensus, one rank enters its cleanup barrier while its peers
            # launch the next GPU collective, obscuring the original failure
            # with a process-group timeout. This is outside timed GPU intervals.
            failures = [None] * 8
            dist.all_gather_object(failures, failure)
            failures = [item for item in failures if item is not None]
            if failures:
                raise AssertionError(f"Communication identity mismatch: {failures}")

        def vary(iteration):
            if not args.vary_routes:
                return
            x.mul_(0.99)
            ids.copy_(
                torch.rand(tokens, args.experts * 4, generator=generator, device=rank)
                .topk(args.top_k, dim=-1)
                .indices
            )
            if iteration % 3 == 0:
                ids.fill_(-1)
            elif iteration % 3 == 1:
                ids[:, 0] = -1
            if args.backend == "deepep":
                mapped.copy_(
                    torch.where(ids >= 0, (ids % 4 + 4) * args.experts + ids // 4, -1)
                )

        gpu_group = None
        try:
            if args.cuda_graph:
                # These benchmark-only NCCL barriers align all eight ranks and
                # separate the identity fixture. They are outside both timed
                # intervals and are never part of the serving protocol.
                gpu_group = dist.new_group(backend="nccl")
                sync_word = torch.zeros(1, dtype=torch.int32, device=rank)
                stream = torch.cuda.Stream(device=rank)
                stream.wait_stream(torch.cuda.current_stream(rank))
                events = [
                    torch.cuda.Event(enable_timing=True, external=True)
                    for _ in range(4)
                ]

                def step():
                    dist.all_reduce(sync_word, group=gpu_group)
                    events[0].record()
                    dispatch()
                    events[1].record()
                    dist.all_reduce(sync_word, group=gpu_group)
                    echo()
                    dist.all_reduce(sync_word, group=gpu_group)
                    events[2].record()
                    combine()
                    events[3].record()
                    dist.all_reduce(sync_word, group=gpu_group)

                with torch.cuda.stream(stream):
                    for _ in range(3):
                        step()
                stream.synchronize()
                identity("graph-warmup")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step()
                for iteration in range(args.iterations + 3):
                    vary(iteration)
                    wall = time.perf_counter()
                    graph.replay()
                    torch.cuda.synchronize(rank)
                    wall_us = (time.perf_counter() - wall) * 1e6
                    identity(f"graph-replay-{iteration}")
                    if iteration >= 3:
                        samples.append(
                            {
                                "dispatch_us": events[0].elapsed_time(events[1]) * 1000,
                                "combine_us": events[2].elapsed_time(events[3]) * 1000,
                                "step_wall_us": wall_us,
                            }
                        )
                del graph
                metadata["timing"] = (
                    "CUDA graph, uncached dynamic routing; GPU barriers and echo excluded"
                )
            else:
                for iteration in range(args.iterations + 3):
                    vary(iteration)
                    torch.cuda.synchronize(rank)
                    dist.barrier()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    wall = time.perf_counter()
                    start.record()
                    dispatch()
                    end.record()
                    end.synchronize()
                    dispatch_us = start.elapsed_time(end) * 1000
                    dispatch_wall_us = (time.perf_counter() - wall) * 1e6
                    dist.barrier()
                    echo()
                    torch.cuda.synchronize(rank)
                    dist.barrier()
                    wall = time.perf_counter()
                    start.record()
                    combine()
                    end.record()
                    end.synchronize()
                    combine_us = start.elapsed_time(end) * 1000
                    combine_wall_us = (time.perf_counter() - wall) * 1e6
                    identity(f"eager-{iteration}")
                    if iteration >= 3:
                        samples.append(
                            {
                                "dispatch_us": dispatch_us,
                                "combine_us": combine_us,
                                "dispatch_wall_us": dispatch_wall_us,
                                "combine_wall_us": combine_wall_us,
                            }
                        )
                metadata["timing"] = (
                    "Eager calls; GPU events include host submission gaps and rank skew"
                )
            reports.append(
                {
                    "rows_per_active_client": rows,
                    "active": active,
                    "rank": rank,
                    "samples": samples,
                    "identity_passed": True,
                    **metadata,
                }
            )
        finally:
            torch.cuda.synchronize(rank)
            dist.barrier()
            remote_group = None
            for remote in remotes:
                remote.close()
            dist.barrier()  # Exporters retain allocations until importers close.
            if buffer is not None:
                buffer.destroy()
            if gpu_group is not None:
                dist.destroy_process_group(gpu_group)
            owner, remotes, buffer = None, [], None
    gathered = [None] * 8
    dist.all_gather_object(gathered, reports)
    if rank == 0:
        summary = []
        for index, rows in enumerate(args.rows):
            phase = [r[index] for r in gathered]
            # Critical-path GPU duration: max of local elapsed intervals,
            # never subtract timestamps belonging to different GPU clocks.
            maxima = {
                key: [
                    max(r["samples"][j][key] for r in phase)
                    for j in range(args.iterations)
                ]
                for key in phase[0]["samples"][0]
            }
            summary.append(
                {
                    "rows": rows,
                    "max_rank_samples": maxima,
                    "median_max_rank_us": {
                        k: statistics.median(v) for k, v in maxima.items()
                    },
                }
            )
        Path(args.output).write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "settings": settings,
                    "summary": summary,
                    "ranks": gathered,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("expertplex", "deepep"), required=True)
    p.add_argument("--mode", choices=("normal", "low_latency"), required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--experts", type=int, default=40)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--group-transfers", action="store_true")
    p.add_argument(
        "--prefill-preprocess",
        action="store_true",
        help="Measure the serving raw-token broadcast/partial-gather transport",
    )
    p.add_argument(
        "--vary-routes",
        action="store_true",
        help="Diagnostic changing routes and padding on every replay",
    )
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--port", type=int, default=29650)
    p.add_argument(
        "--deepep-revision",
        help="Record the exact independently installed DeepEP source commit",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.prefill_preprocess and (
        args.backend != "expertplex"
        or args.mode != "normal"
        or not args.group_transfers
    ):
        p.error("E preprocessing requires ExpertPlex normal mode with group transfers")
    if min(*args.rows, args.iterations) <= 0:
        p.error("Counts must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0,1,2,3,4,5,6,7"):
        p.error("Expose eight GPUs in physical order")
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(vars(args),), nprocs=8, join=True)


if __name__ == "__main__":
    main()
