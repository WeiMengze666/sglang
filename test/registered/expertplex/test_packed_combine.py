"""Normal EP gather rounds each rank before the attention-side rank sum.

The CPU cases are reference counterexamples, not CUDA qualification. The GPU
cases exercise the production grouped and per-peer packed combine entrypoints
against those counterexamples, without introducing GEMM numerical differences.
Run GPU cases only in a gpuq allocation of at least three GPUs, with
SGLANG_RUN_EXPERTPLEX_GPU_TESTS=1. They remain skipped on CPU development hosts.
"""

import importlib.util
import multiprocessing
import os
import struct
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_rn(value):
    """Finite-value reference conversion; the fixtures do not use NaN/Inf."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def combine_reference(ids, weights, rank_values, *, round_ranks=True):
    """Identity expert fixtures with exact power-of-two BF16 rank values.

    Power-of-two products are exact FP32, so the test does not replace CUDA
    FMA behavior with Python's double-precision arithmetic. Inputs and each
    accumulation are explicitly rounded to FP32.
    """
    total = 0.0
    world = len(rank_values)
    for rank, value in enumerate(rank_values):
        partial = 0.0
        for expert, weight in zip(ids, weights):
            if expert >= 0 and expert % world == rank:
                partial = fp32(partial + fp32(fp32(weight) * value))
        total = fp32(total + (bf16_rn(partial) if round_ranks else partial))
    return total


class PackedGatherReferenceTests(unittest.TestCase):
    def test_normalized_routes_expose_missing_rank_boundary(self):
        ids, weights = [0, 1], [0.002, fp32(1 - fp32(0.002))]
        normal = combine_reference(ids, weights, [1.0, 2.0])
        old_packed = combine_reference(ids, weights, [1.0, 2.0], round_ranks=False)
        self.assertEqual(bf16_rn(normal), 1.9921875)
        self.assertEqual(bf16_rn(old_packed), 2.0)

    def test_round_after_rank_routes_not_after_every_route(self):
        # Experts 0 and 2 both belong to rank 0 in a two-rank group.
        actual = combine_reference([0, 2], [0.01, 0.06], [1.0, 2.0])
        route_rounded = bf16_rn(fp32(bf16_rn(0.01) + bf16_rn(0.06)))
        self.assertEqual(bf16_rn(actual), 0.06982421875)
        self.assertEqual(route_rounded, 0.0703125)

    def test_empty_rank_and_padding_contribute_zero(self):
        self.assertEqual(combine_reference([-1, -1], [0.5, 0.5], [1.0, 2.0]), 0)
        self.assertEqual(combine_reference([1, -1], [0.5, 0.5], [1.0, 2.0]), 1.0)

    def test_reference_uses_ties_to_even(self):
        self.assertEqual(bf16_rn(1.00390625), 1.0)
        self.assertEqual(bf16_rn(1.01171875), 1.015625)
        self.assertEqual(bf16_rn(-1.00390625), -1.0)


def load_native():
    spec = importlib.util.spec_from_file_location(
        "packed_combine_native", ROOT / "python/sglang/srt/expertplex/native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expert_fixture(connection, rank):
    """Expose real IPC buffers; overwrite only owned synthetic expert outputs."""
    try:
        import torch

        torch.cuda.set_device(rank + 1)
        native = load_native()
        owner = native.OwnedBuffers(
            slot_count=1,
            local_experts=2,
            max_rows=128,
            hidden=64,
            intermediate=64,
            device=rank + 1,
            packed=[1],
            top_k=2,
            expert_rank=rank,
            expert_world=2,
        )
        extension = native.load_extension()
        views = extension.benchmark_views(owner._impl, 0)
        connection.send(("ready", owner.descriptor()))
        while True:
            command = connection.recv()
            if command == "stop":
                break
            if command != "complete":
                raise ValueError(command)
            extension.benchmark_echo(owner._impl)
            # All ranks have identical dispatch input. Distinct exact BF16
            # results make the rank-gather conversion observable after combine.
            views["output"].fill_(rank + 1)
            torch.cuda.synchronize()
            connection.send(("complete", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_EXPERTPLEX_GPU_TESTS") == "1",
    "B300 GPU regression pending; enable only within a gpuq allocation",
)
class PackedGatherGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            raise unittest.SkipTest("One attention and two expert GPUs are required")
        cls.torch = torch
        torch.cuda.set_device(0)
        cls.native = load_native()
        cls.native.load_extension()

    def read_fixture(self, connection, expected):
        self.assertTrue(connection.poll(600), "Expert fixture timed out")
        state, payload = connection.recv()
        self.assertEqual(state, expected, payload)
        return payload

    def test_grouped_and_per_peer_combine_preserve_normal_gather(self):
        torch, native = self.torch, self.native
        ctx = multiprocessing.get_context("spawn")
        processes, connections, peers = [], [], []
        group = None
        try:
            for rank in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=expert_fixture, args=(child, rank))
                process.start()
                child.close()
                processes.append(process)
                connections.append(parent)
                descriptor = self.read_fixture(parent, "ready")
                peers.append(native.RemoteBuffers(descriptor, 0, 0))
            group = native.RemoteGroup(peers)
            # Includes split ranks, multiple routes on one rank, empty rank,
            # fully padded and partly padded tokens, and both route ID dtypes.
            host_ids = [[0, 1], [0, 2], [1, 3], [-1, -1], [1, -1]]
            host_weights = [
                [0.002, fp32(1 - fp32(0.002))],
                [0.01, 0.06],
                [0.01, 0.06],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
            expected = (
                torch.tensor(
                    [
                        combine_reference(i, w, [1.0, 2.0])
                        for i, w in zip(host_ids, host_weights)
                    ],
                    dtype=torch.float32,
                )[:, None]
                .expand(-1, 64)
                .contiguous()
            )
            hidden = torch.ones((len(host_ids), 64), dtype=torch.bfloat16, device=0)
            weights = torch.tensor(host_weights, dtype=torch.float32, device=0)
            for grouped in (False, True):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(grouped=grouped, dtype=dtype):
                        ids = torch.tensor(host_ids, dtype=dtype, device=0)
                        row_map = torch.empty_like(ids, dtype=torch.int32)
                        if grouped:
                            group.dispatch(hidden, ids, row_map, 0, 0, weights)
                        else:
                            for rank, peer in enumerate(peers):
                                native.dispatch(
                                    peer, hidden, ids, row_map, 0, 0, 0, rank, 2
                                )
                        torch.cuda.synchronize()
                        for connection in connections:
                            connection.send("complete")
                        for connection in connections:
                            self.read_fixture(connection, "complete")
                        output = torch.zeros_like(hidden, dtype=torch.float32)
                        if grouped:
                            group.combine(output, ids, weights, row_map)
                        else:
                            for rank, peer in enumerate(peers):
                                native.combine(
                                    peer, output, ids, weights, row_map, 0, rank, 2
                                )
                        actual = output.cpu()
                        # Check FP32 rank sum as well as the final serving BF16
                        # tensor: an extra/missing conversion cannot hide here.
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int32), expected.view(torch.int32)
                            )
                        )
                        self.assertTrue(
                            torch.equal(actual.bfloat16(), expected.bfloat16())
                        )
        finally:
            torch.cuda.synchronize()
            group = None
            for peer in peers:
                peer.close()
            for connection in connections:
                try:
                    connection.send("stop")
                except (BrokenPipeError, EOFError):
                    pass
                connection.close()
            for process in processes:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(10)


if __name__ == "__main__":
    unittest.main()
