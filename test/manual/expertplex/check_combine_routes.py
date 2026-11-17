"""Check masked group combine route tails and ordered FP32/BF16 outputs.

Run inside a gpuq allocation with at least six visible GPUs. The fixture uses
GPU 0 for attention and GPUs 4 and 5 for expert IPC buffers. --check-cpu only
validates the exact dyadic reference fixtures and never initializes CUDA.
"""

import argparse
from contextlib import ExitStack
import json
import multiprocessing
from pathlib import Path
import queue
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
from _bootstrap import module  # noqa: E402
import check_gpu  # noqa: E402

TOP_K = (1, 2, 7, 8, 9, 16)
VARIANTS = ("full", "duplicates", "partial", "empty", "zero", "cancellation")
HIDDEN = 64
ROWS = 7
EXPERTS = 8
WORLD = 2
CAPACITY = 128


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def expert_value(expert, column):
    return bf16(expert + 1 + (column % 8) / 16)


def host_case(k, variant):
    ids, weights, expected = [], [], []
    for token in range(ROWS):
        routes = [(token * 3 + choice) % (EXPERTS * WORLD) for choice in range(k)]
        if variant == "duplicates":
            routes = [token % (EXPERTS * WORLD)] * k
        elif variant == "partial":
            routes = [
                -1 if (choice + token) % 3 == 0 else expert
                for choice, expert in enumerate(routes)
            ]
        elif variant == "empty":
            routes = [-1] * k
        values = [2.0 ** (-(choice + token) % 4) / 8 for choice in range(k)]
        if variant == "zero":
            values = [0.0] * k
        elif variant == "cancellation":
            values = [
                value * (-1 if (token + choice) % 2 else 1)
                for choice, value in enumerate(values)
            ]
        ids.append(routes)
        weights.append(values)
        expected.append(
            [
                sum(
                    expert_value(expert, column) * weight
                    for expert, weight in zip(routes, values)
                    if expert >= 0
                )
                for column in range(HIDDEN)
            ]
        )
    return ids, weights, expected


def check_cpu():
    cases = []
    for k in TOP_K:
        for variant in VARIANTS:
            ids, weights, expected = host_case(k, variant)
            assert len(ids) == len(weights) == len(expected) == ROWS
            assert all(len(row) == k for row in ids + weights)
            assert all(len(row) == HIDDEN for row in expected)
            # Exact dyadic products/sums have ample FP32 significand range.
            assert all(
                struct.unpack("<f", struct.pack("<f", v))[0] == v
                for row in expected
                for v in row
            )
            assert all(-1 <= v < EXPERTS * WORLD for row in ids for v in row)
            if variant in ("empty", "zero"):
                assert all(v == 0 for row in expected for v in row)
            cases.append((k, variant))
    return dict(
        status="cpu_fixture_passed",
        cases=len(cases),
        scope="Fixture construction and exact dyadic reference only; GPU path unrun",
    )


def exporter(settings, device, start_executor, outbox, stop_event):
    try:
        import torch

        torch.cuda.set_device(device)
        native = module("native")
        buffers = native.OwnedBuffers(
            1,
            EXPERTS,
            CAPACITY,
            HIDDEN,
            64,
            device,
            packed=[0],
            top_k=16,
            expert_rank=settings["rank"],
            expert_world=WORLD,
        )
        extension = native.load_extension()
        views = extension.benchmark_views(buffers._impl, 0)
        pattern = torch.tensor(
            [
                [
                    expert_value(expert * WORLD + settings["rank"], h)
                    for h in range(HIDDEN)
                ]
                for expert in range(EXPERTS)
            ],
            dtype=torch.bfloat16,
            device=device,
        )
        pattern = (
            pattern[:, None, :].expand(EXPERTS, CAPACITY, HIDDEN).reshape(-1, HIDDEN)
        )
        assert tuple(views["output"].shape) == tuple(pattern.shape)
        outbox.put(
            dict(
                state="ready",
                descriptor=buffers.descriptor(),
                hardware=check_gpu._hardware(torch, device),
                executor=None,
            )
        )
        while not stop_event.is_set():
            try:
                command = settings["commands"].get(timeout=0.05)
            except queue.Empty:
                continue
            if command != "complete":
                raise ValueError(command)
            extension.benchmark_echo(buffers._impl)
            views["output"].copy_(pattern)
            torch.cuda.synchronize()
            # Explicit host rendezvous keeps the diagnostic overwrite outside
            # combine. This fixture measures no latency.
            outbox.put(dict(state="complete"))
    except BaseException:
        outbox.put(dict(state="error", traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-cpu", action="store_true")
    args = parser.parse_args()
    if args.check_cpu:
        print(json.dumps(check_cpu()))
        return
    if args.output is None:
        parser.error("--output is required for GPU validation")
    import torch

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    native = module("native")
    native.load_extension()
    check_gpu._exporter = exporter
    report = dict(
        status="running",
        source=native.source_fingerprint(),
        cases=[],
        settings=dict(
            top_k=list(TOP_K),
            variants=list(VARIANTS),
            hidden=HIDDEN,
            rows=ROWS,
            local_experts=EXPERTS,
            world=WORLD,
            capacity=CAPACITY,
        ),
        scope="Exact synthetic IPC combine outputs; no model quality/performance test",
    )
    peers, command_queues, owners = [], [], []
    group = None
    try:
        with ExitStack() as stack:
            try:
                context = multiprocessing.get_context("spawn")
                for rank in range(WORLD):
                    commands = context.Queue()
                    command_queues.append(commands)
                    owner = stack.enter_context(
                        check_gpu.Exporter(
                            dict(rank=rank, commands=commands),
                            rank + 4,
                            execute=False,
                            timeout_s=600,
                        )
                    )
                    owners.append(owner)
                    peers.append(native.RemoteBuffers(owner.descriptor, 0, 0))
                group = native.RemoteGroup(peers, operation_trace_capacity=128)
                hidden = torch.ones((ROWS, HIDDEN), dtype=torch.bfloat16, device=0)
                float_output = torch.empty_like(hidden, dtype=torch.float32)
                bf16_output = torch.empty_like(hidden)
                stream = torch.cuda.Stream()
                for k in TOP_K:
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.empty((ROWS, k), dtype=dtype, device=0)
                        weights = torch.empty((ROWS, k), dtype=torch.float32, device=0)
                        row_map = torch.empty((ROWS, k), dtype=torch.int32, device=0)
                        graph = None
                        try:
                            for variant in VARIANTS:
                                host_ids, host_weights, host_expected = host_case(
                                    k, variant
                                )
                                ids.copy_(torch.tensor(host_ids, dtype=dtype))
                                weights.copy_(
                                    torch.tensor(host_weights, dtype=torch.float32)
                                )
                                expected = torch.tensor(
                                    host_expected, dtype=torch.float32
                                )
                                group.dispatch(hidden, ids, row_map, 0, 1, weights)
                                torch.cuda.synchronize()
                                for commands in command_queues:
                                    commands.put("complete")
                                for owner in owners:
                                    reply = owner.outbox.get(timeout=300)
                                    assert reply["state"] == "complete", reply

                                def combine():
                                    group.combine(
                                        float_output,
                                        ids,
                                        weights,
                                        row_map,
                                        timeout_ms=1000,
                                    )
                                    group.combine(
                                        bf16_output,
                                        ids,
                                        weights,
                                        row_map,
                                        timeout_ms=1000,
                                    )

                                def compare():
                                    actual = float_output.cpu()
                                    assert torch.equal(
                                        actual.view(torch.int32),
                                        expected.view(torch.int32),
                                    ), (k, variant, dtype, "fp32")
                                    assert torch.equal(
                                        bf16_output.cpu().view(torch.int16),
                                        expected.bfloat16().view(torch.int16),
                                    ), (k, variant, dtype, "bf16")

                                float_output.fill_(float("nan"))
                                bf16_output.fill_(float("nan"))
                                combine()
                                compare()
                                if graph is None:
                                    stream.wait_stream(torch.cuda.current_stream())
                                    graph = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(graph, stream=stream):
                                        combine()
                                float_output.fill_(float("nan"))
                                bf16_output.fill_(float("nan"))
                                graph.replay()
                                compare()
                                report["cases"].append(
                                    dict(
                                        top_k=k,
                                        variant=variant,
                                        ids_dtype=str(dtype),
                                        fp32_exact=True,
                                        bf16_exact=True,
                                        eager=True,
                                        graph_replay=True,
                                    )
                                )
                        finally:
                            if graph is not None:
                                torch.cuda.synchronize()
                                graph.reset()
                # Poison every peer through the real invalid-route dispatch.
                # No echo runs here; ERROR must prevent all peer reads.
                ids.fill_(EXPERTS * WORLD)
                group.dispatch(hidden, ids, row_map, 0, 1, weights)
                float_output.fill_(float("nan"))
                bf16_output.fill_(float("nan"))
                group.combine(float_output, ids, weights, row_map, timeout_ms=1000)
                group.combine(bf16_output, ids, weights, row_map, timeout_ms=1000)
                assert torch.count_nonzero(float_output).item() == 0
                assert torch.count_nonzero(bf16_output).item() == 0
                trace = group.read_operation_trace()
                status_index = trace["fields"].index("status")
                counts = trace["counts"].tolist()
                status = [
                    int(trace["records"][rank, counts[rank] - 1, status_index])
                    for rank in range(WORLD)
                ]
                assert status == [3, 3], status
                report["failed_peer_check"] = dict(
                    status=status, top_k=16, fp32_zero=True, bf16_zero=True
                )
            finally:
                torch.cuda.synchronize()
                group = None
                for peer in peers:
                    peer.close()
        assert all(owner.process.exitcode == 0 for owner in owners)
        report.update(status="passed", owned_cleanup_complete=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        for commands in command_queues:
            commands.close()
            commands.join_thread()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
