"""Check native ExpertPlex CUDA IPC primitives on two B300 GPUs.

This is a synthetic correctness/progress smoke test, not model accuracy,
throughput, or goodput acceptance. Torch/CUDA are imported only when running
validation, so ``--help`` also works on the CPU development machine.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import module


def _weights(torch, settings):
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    hidden, intermediate = settings["hidden"], settings["intermediate"]
    shape13 = (settings["layers"], settings["experts"], 2 * intermediate, hidden)
    shape2 = (settings["layers"], settings["experts"], hidden, intermediate)
    w13 = (torch.randn(shape13, generator=generator) * 0.05).bfloat16()
    w2 = (torch.randn(shape2, generator=generator) * 0.05).bfloat16()
    return w13, w2


def _hardware(torch, device):
    prop = torch.cuda.get_device_properties(device)
    return {
        "device": device,
        "name": prop.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": prop.multi_processor_count,
        "total_memory_bytes": prop.total_memory,
    }


def _exporter(settings, device, start_executor, outbox, stop_event):
    executor = None
    try:
        import torch

        native = module("native")
        torch.cuda.set_device(device)
        native.load_extension()
        buffers = native.OwnedBuffers(
            slot_count=2,
            local_experts=settings["experts"],
            max_rows=settings["capacity"],
            hidden=settings["hidden"],
            intermediate=settings["intermediate"],
            device=device,
        )
        if start_executor:
            w13, w2 = _weights(torch, settings)
            executor = native.PersistentExecutor(
                buffers,
                w13.to(device),
                w2.to(device),
                decode_sm_budget=2,
                expected_decode_tiles=64,
                max_decode_sm_fraction=0.75,
                ctas=0,
            )
        outbox.put(
            {
                "state": "ready",
                "descriptor": buffers.descriptor(),
                "hardware": _hardware(torch, device),
            }
        )
        while not stop_event.wait(0.05):
            if executor is not None:
                executor.check_error()
    except BaseException:
        outbox.put({"state": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        if executor is not None:
            executor.stop()


class Exporter:
    """A separately spawned CUDA owner, retained until IPC mappings close."""

    def __init__(self, settings, device, *, execute, timeout_s):
        self.settings = settings
        self.device = device
        self.execute = execute
        self.timeout_s = timeout_s

    def __enter__(self):
        context = multiprocessing.get_context("spawn")
        self.outbox = context.Queue()
        self.stop_event = context.Event()
        self.process = context.Process(
            target=_exporter,
            args=(
                self.settings,
                self.device,
                self.execute,
                self.outbox,
                self.stop_event,
            ),
            name=f"expertplex-validation-owner-{self.device}",
        )
        self.process.start()
        try:
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                try:
                    message = self.outbox.get(timeout=min(0.5, self.timeout_s))
                except queue.Empty:
                    if self.process.exitcode is not None:
                        raise RuntimeError(
                            f"CUDA owner exited with code {self.process.exitcode}"
                        )
                    continue
                if message["state"] != "ready":
                    raise RuntimeError(message["traceback"])
                self.descriptor = message["descriptor"]
                self.hardware = message["hardware"]
                return self
            raise TimeoutError(f"CUDA owner did not become ready in {self.timeout_s}s")
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, *_):
        self.stop_event.set()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.outbox.close()
        if exc_type is None and self.process.exitcode != 0:
            raise RuntimeError(
                f"CUDA owner failed with exit code {self.process.exitcode}"
            )


def _case(torch, settings, rows, layer, seed, *, skewed=False, int64_ids=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (
        torch.randn(rows, settings["hidden"], generator=generator) * 0.5
    ).bfloat16()
    token = torch.arange(rows, dtype=torch.int64)
    if skewed:
        ids = torch.stack((torch.zeros_like(token), torch.ones_like(token)), dim=1)
    else:
        first = token % settings["experts"]
        second = (first + 1 + (token // 3) % 2) % settings["experts"]
        ids = torch.stack((first, second), dim=1)
    ids = ids.to(torch.int64 if int64_ids else torch.int32)
    weights = torch.randn(rows, 2, generator=generator).softmax(dim=-1)
    return {"hidden": hidden, "ids": ids, "weights": weights, "layer": layer}


def _reference(torch, case, w13, w2, settings):
    """FP32 products with the native BF16 activation/expert-output boundaries.

    Native gate/up accumulate in FP32, then SiLU(gate)*up is cast to BF16;
    down accumulates in FP32 and is cast to BF16 before weighted combine.
    This reference deliberately does not round gate/up matmuls to BF16.
    """
    result = torch.zeros(case["hidden"].shape, dtype=torch.float32)
    layer = case["layer"]
    intermediate = settings["intermediate"]
    for expert in range(settings["experts"]):
        row, choice = (case["ids"] == expert).nonzero(as_tuple=True)
        if not row.numel():
            continue
        x = case["hidden"][row].float()
        gate = x @ w13[layer, expert, :intermediate].float().T
        up = x @ w13[layer, expert, intermediate:].float().T
        activated = (torch.nn.functional.silu(gate) * up).bfloat16()
        expert_output = (activated.float() @ w2[layer, expert].float().T).bfloat16()
        weighted = expert_output.float() * case["weights"][row, choice, None]
        result.index_add_(0, row, weighted)
    return result.bfloat16()


def _prepare(torch, case, device):
    return {
        "hidden": case["hidden"].to(device),
        "ids": case["ids"].to(device),
        "weights": case["weights"].to(device),
        "row_map": torch.empty(case["ids"].shape, dtype=torch.int32, device=device),
        "output": torch.zeros(case["hidden"].shape, dtype=torch.float32, device=device),
        "layer": case["layer"],
    }


def _run(torch, native, remote, prepared, phase, epoch, device, timeout_ms):
    # RemoteBuffers has per-object completion status storage. Each concurrent
    # slot therefore has its own RemoteBuffers, stream and tensor storage.
    torch.cuda.set_device(device)
    started = time.perf_counter()
    native.dispatch(
        remote,
        prepared["hidden"],
        prepared["ids"],
        prepared["row_map"],
        epoch,
        prepared["layer"],
        phase,
        0,
        1,
    )
    native.combine(
        remote,
        prepared["output"],
        prepared["ids"],
        prepared["weights"],
        prepared["row_map"],
        epoch,
        0,
        1,
        timeout_ms=timeout_ms,
    )
    output = prepared["output"].bfloat16()
    torch.cuda.current_stream(device).synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return output.cpu(), elapsed_ms


def _numerics(torch, actual, expected, args):
    if not torch.isfinite(actual).all():
        raise AssertionError("Native output contains nonfinite values")
    delta = actual.float() - expected.float()
    absolute = delta.abs()
    relative_l2 = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
    passed = bool((absolute <= args.atol + args.rtol * expected.float().abs()).all())
    passed = passed and relative_l2 <= args.relative_l2
    return {
        "max_absolute_error": float(absolute.max()) if absolute.numel() else 0.0,
        "relative_l2_error": relative_l2,
        "passed": passed,
    }


def check_pair(torch, native, args, settings, report):
    attention = args.attention_device
    expert = args.expert_device
    torch.cuda.set_device(attention)
    w13, w2 = _weights(torch, settings)
    epochs = [0, 0]
    with Exporter(
        settings, expert, execute=True, timeout_s=args.startup_timeout_s
    ) as owner:
        remotes = []
        try:
            remotes = [
                native.RemoteBuffers(owner.descriptor, slot, attention)
                for slot in range(2)
            ]
            for iteration in range(args.iterations):
                for slot, rows in (
                    (0, settings["capacity"] - 1),
                    (1, 1 + iteration % 7),
                ):
                    case = _case(
                        torch,
                        settings,
                        rows,
                        iteration % settings["layers"],
                        settings["seed"] + 100 + iteration * 2 + slot,
                        skewed=slot == 0,
                        int64_ids=iteration == 0,
                    )
                    expected = _reference(torch, case, w13, w2, settings)
                    prepared = _prepare(torch, case, attention)
                    torch.cuda.synchronize(attention)
                    epochs[slot] += 1
                    actual, latency = _run(
                        torch,
                        native,
                        remotes[slot],
                        prepared,
                        slot,
                        epochs[slot],
                        attention,
                        args.operation_timeout_ms,
                    )
                    evidence = {
                        "name": "uneven_prefill_inactive_experts"
                        if slot == 0
                        else "small_decode",
                        "phase": "prefill" if slot == 0 else "decode",
                        "rows": rows,
                        "layer": case["layer"],
                        "epoch": epochs[slot],
                        "wall_latency_ms": latency,
                        **_numerics(torch, actual, expected, args),
                    }
                    report["cases"].append(evidence)
                    if not evidence["passed"]:
                        raise AssertionError(f"Numerical mismatch: {evidence}")

            # Exercise the actual client's empty-input fast path without
            # constructing a live client or publishing another slot epoch.
            client_type = module("client").ExpertPlexClient
            uninitialized = client_type.__new__(client_type)
            empty = torch.empty(
                (0, settings["hidden"]), dtype=torch.bfloat16, device=attention
            )
            zero_output = uninitialized.run(0, empty, None, None)
            if zero_output is not empty:
                raise AssertionError("Zero-token forward changed its empty input")
            report["zero_token_fast_path"] = {"passed": True, "ipc_dispatches": 0}

            barrier = threading.Barrier(2)
            prepared_cases = []
            expected_cases = []
            streams = []
            for slot, rows in ((0, settings["capacity"] - 1), (1, 7)):
                case = _case(
                    torch,
                    settings,
                    rows,
                    slot,
                    settings["seed"] + 200 + slot,
                    skewed=slot == 0,
                )
                prepared_cases.append(_prepare(torch, case, attention))
                expected_cases.append(_reference(torch, case, w13, w2, settings))
                streams.append(torch.cuda.Stream(device=attention))
                epochs[slot] += 1
            torch.cuda.synchronize(attention)
            concurrent_results = [None, None]
            report["concurrent_phases"] = concurrent_results
            failures = []

            def run_slot(slot):
                try:
                    torch.cuda.set_device(attention)
                    with torch.cuda.stream(streams[slot]):
                        barrier.wait(timeout=30)
                        actual, latency = _run(
                            torch,
                            native,
                            remotes[slot],
                            prepared_cases[slot],
                            slot,
                            epochs[slot],
                            attention,
                            args.operation_timeout_ms,
                        )
                        concurrent_results[slot] = {
                            "phase": "prefill" if slot == 0 else "decode",
                            "wall_latency_ms": latency,
                            **_numerics(torch, actual, expected_cases[slot], args),
                        }
                except BaseException:
                    failures.append(traceback.format_exc())

            threads = [
                threading.Thread(target=run_slot, args=(slot,)) for slot in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if failures:
                raise RuntimeError(
                    "Concurrent P/D validation failed:\n" + "\n".join(failures)
                )
            if not all(result and result["passed"] for result in concurrent_results):
                raise AssertionError(
                    "Concurrent P/D numerical mismatch or missing progress"
                )
        finally:
            for remote in remotes:
                remote.close()


def check_timeout(torch, native, args, settings, report):
    # A new process/allocation has no persistent consumer. Dispatch once,
    # require a bounded timeout, then discard it without publishing a new epoch.
    with Exporter(
        settings, args.expert_device, execute=False, timeout_s=args.startup_timeout_s
    ) as owner:
        torch.cuda.set_device(args.attention_device)
        remote = native.RemoteBuffers(owner.descriptor, 0, args.attention_device)
        try:
            case = _case(torch, settings, 1, 0, settings["seed"] + 300)
            prepared = _prepare(torch, case, args.attention_device)
            torch.cuda.synchronize(args.attention_device)
            started = time.perf_counter()
            try:
                _run(
                    torch,
                    native,
                    remote,
                    prepared,
                    0,
                    1,
                    args.attention_device,
                    args.timeout_test_ms,
                )
            except RuntimeError as exc:
                if "native code 7" not in str(exc):
                    raise
                report["timeout"] = {
                    "passed": True,
                    "configured_timeout_ms": args.timeout_test_ms,
                    "wall_latency_ms": (time.perf_counter() - started) * 1000,
                    "slot_reused": False,
                    "fresh_owner_allocation": True,
                }
            else:
                raise AssertionError("An unconsumed request did not time out")
        finally:
            remote.close()


def check_full_mesh(torch, native, args, settings, report):
    if torch.cuda.device_count() != 8:
        raise ValueError("--full-mesh requires exactly eight visible B300 GPUs")
    topology = []
    report["full_mesh"] = topology
    for owner_device in range(8):
        with Exporter(
            settings, owner_device, execute=False, timeout_s=args.startup_timeout_s
        ) as owner:
            for importer_device in range(8):
                if importer_device == owner_device:
                    continue
                torch.cuda.set_device(importer_device)
                # RemoteBuffers validates both CUDA peer access and native
                # peer atomics before opening the owner's CUDA IPC handles.
                remote = native.RemoteBuffers(owner.descriptor, 0, importer_device)
                remote.close()
                topology.append(
                    {
                        "owner_device": owner_device,
                        "importer_device": importer_device,
                        "ipc_open_close_and_peer_atomics": "passed",
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-device", type=int, default=0)
    parser.add_argument("--expert-device", type=int, default=1)
    parser.add_argument("--hidden", type=int, choices=(64, 128), default=64)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--startup-timeout-s", type=float, default=600)
    parser.add_argument("--operation-timeout-ms", type=int, default=10000)
    parser.add_argument("--timeout-test-ms", type=int, default=100)
    parser.add_argument("--skip-timeout-test", action="store_true")
    parser.add_argument(
        "--full-mesh",
        action="store_true",
        help="Check IPC/atomics for all 56 ordered GPU pairs.",
    )
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--relative-l2", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the JSON evidence in this file as well as stdout.",
    )
    args = parser.parse_args()
    if args.attention_device == args.expert_device:
        parser.error("attention and expert devices must be distinct")
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to exercise both layers")
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.atol, args.rtol, args.relative_l2)
    ):
        parser.error("numerical tolerances must be finite and nonnegative")
    if (
        not 0 < args.operation_timeout_ms <= 3600000
        or not 0 < args.timeout_test_ms <= 3600000
    ):
        parser.error("timeouts must be in [1, 3600000] milliseconds")
    if not math.isfinite(args.startup_timeout_s) or args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive")
    return args


def main():
    args = parse_args()
    settings = {
        "hidden": args.hidden,
        "intermediate": 64,
        "layers": 2,
        "experts": 4,
        "top_k": 2,
        "capacity": 64,
        "seed": args.seed,
    }
    report = {
        "schema_version": 1,
        "scope": "native CUDA IPC primitive correctness/progress smoke; not full-model accuracy or goodput",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": settings,
        "reference": "CPU Torch FP32 gate/up and down; BF16 activation, expert output and final output",
        "latency_scope": "host dispatch/combine, final BF16 cast and stream completion; validation D2H copy excluded; smoke only",
        "tolerances": {
            "absolute": args.atol,
            "relative": args.rtol,
            "relative_l2": args.relative_l2,
        },
        "cases": [],
    }
    try:
        import torch

        native = module("native")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; run this validation on the B300 deployment host"
            )
        for device in (args.attention_device, args.expert_device):
            if not 0 <= device < torch.cuda.device_count():
                raise ValueError(f"Invalid visible CUDA device: {device}")
            if torch.cuda.get_device_capability(device) != (10, 3):
                raise RuntimeError("This native extension targets B300 SM103 only")
        report["software"] = {"pytorch": torch.__version__, "cuda": torch.version.cuda}
        report["hardware"] = [
            _hardware(torch, device) for device in range(torch.cuda.device_count())
        ]
        torch.cuda.set_device(args.attention_device)
        native.load_extension()
        check_pair(torch, native, args, settings, report)
        if not args.skip_timeout_test:
            check_timeout(torch, native, args, settings, report)
        if args.full_mesh:
            check_full_mesh(torch, native, args, settings, report)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
