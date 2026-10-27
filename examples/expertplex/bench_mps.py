"""Separate-process MPS grouped-GEMM sharing ablation for paper section 7.3.

Starts a task-private MPS daemon for one expert GPU using a fresh pipe/log
directory, and stops that daemon in finally. It does not change compute mode
or signal an existing daemon. A separate control GPU releases P/D 10 us apart
and measures same-clock completion-ack upper bounds, as in bench_sharing.py.
No serving goodput or paper performance claim follows from these measurements.
"""

import argparse
import json
import multiprocessing
import os
import signal
import statistics
import subprocess
import traceback
import uuid as uuid_module
from pathlib import Path


def canonical_gpu_uuid(value):
    """Use the CUDA-visible device identity, never reinterpret a logical ordinal."""
    if isinstance(value, bytes) and len(value) == 16:
        identity = uuid_module.UUID(bytes=value)
    else:
        identity = uuid_module.UUID(str(value).removeprefix("GPU-"))
    return "GPU-" + str(identity)


def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="expertplex_sharing_control",
        sources=[str(Path(__file__).with_name("sharing_control.cu"))],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"],
        extra_ldflags=["-lcuda"],
    )


def worker(settings, phase, handle, pipe):
    try:
        import deep_gemm
        import torch

        from bench_sharing import weights, workload

        args = argparse.Namespace(**settings)
        args.device = 0  # Each MPS client exposes only the expert GPU UUID.
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        control = extension()
        flags = control.import_flags(handle, 0)
        w13, w2 = weights(args)
        invoke, out, counts = workload(
            args, args.prefill_rows if phase == 0 else args.decode_rows, phase, w13, w2
        )
        reported_sms = torch.cuda.get_device_properties(0).multi_processor_count
        kernel_sms = reported_sms // 2 * 2
        deep_gemm.set_num_sms(kernel_sms)
        invoke()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            invoke()
        stream.synchronize()
        pipe.send(
            {
                "state": "ready",
                "pid": os.getpid(),
                "reported_sms": reported_sms,
                "kernel_sms": kernel_sms,
                "expert_rows": counts,
            }
        )
        while True:
            command = pipe.recv()
            if command == "stop":
                break
            control.wait_flag(flags, phase, stream.cuda_stream)
            with torch.cuda.stream(stream):
                if command == "run":
                    graph.replay()
            control.done(flags, phase + 2, stream.cuda_stream)
            pipe.send({"state": "queued"})
            stream.synchronize()
            pipe.send({"state": "completed", "finite": bool(torch.isfinite(out).all())})
    except BaseException:
        pipe.send({"state": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        pipe.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--control-device", type=int, default=0)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--intermediate", type=int, default=2560)
    p.add_argument("--experts", type=int, default=40)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--prefill-rows", type=int, default=8192)
    p.add_argument("--decode-rows", type=int, default=128)
    p.add_argument("--operation", choices=("gateup", "down", "moe"), default="gateup")
    p.add_argument("--delay-us", type=int, default=10)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--percentages", nargs="+", default=["100:100", "75:25", "50:50", "25:75"]
    )
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    if args.device == args.control_device or args.iterations <= 0:
        p.error("Use distinct GPUs and positive iteration count")
    partitions = [tuple(map(int, value.split(":"))) for value in args.percentages]
    if any(
        len(pair) != 2 or any(not 1 <= v <= 100 for v in pair) for pair in partitions
    ):
        p.error("Each percentage must be between 1 and 100")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    (args.work_dir / "pipes").mkdir()
    (args.work_dir / "logs").mkdir()
    import torch

    torch.set_num_threads(4)
    if (
        min(args.device, args.control_device) < 0
        or max(args.device, args.control_device) >= torch.cuda.device_count()
    ):
        p.error("Device indices must belong to the CUDA-visible GPU allocation")
    # GPUQ may set/reorder CUDA_VISIBLE_DEVICES. Resolve through CUDA, which
    # honors that mask; nvidia-smi -i would instead use a physical ordinal.
    # Children narrow the allocation to this UUID and remap it to local GPU 0.
    uuid = canonical_gpu_uuid(torch.cuda.get_device_properties(args.device).uuid)
    control_uuid = canonical_gpu_uuid(
        torch.cuda.get_device_properties(args.control_device).uuid
    )
    mps_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        CUDA_MPS_PIPE_DIRECTORY=str(args.work_dir / "pipes"),
        CUDA_MPS_LOG_DIRECTORY=str(args.work_dir / "logs"),
    )
    torch.cuda.set_device(args.device)
    control = extension()
    flags = control.allocate_flags(args.device)
    stamps = torch.zeros(4, dtype=torch.int64, device=args.control_device)
    handle = control.export_flags(flags)
    ctx = multiprocessing.get_context("spawn")
    from bench_sharing import WORKLOAD_IDENTITY

    report = {
        "scope": __doc__,
        "workload_identity": WORKLOAD_IDENTITY,
        "settings": vars(args)
        | {"work_dir": str(args.work_dir), "output": str(args.output)},
        "gpu_uuid": uuid,
        "control_gpu_uuid": control_uuid,
        "coordinator_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": [],
    }
    processes = []
    channels = []

    def mps_command(command):
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            env=mps_env,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()

    def receive(channel, expected, timeout=180):
        if not channel.poll(timeout):
            raise TimeoutError(f"MPS client did not reach {expected}")
        result = channel.recv()
        if result.get("state") != expected:
            raise RuntimeError(result)
        return result

    def measure(enabled=(True, True)):
        flags.zero_()
        stamps.zero_()
        torch.cuda.synchronize(args.device)
        torch.cuda.synchronize(args.control_device)
        for channel, active in zip(channels, enabled):
            channel.send("run" if active else "skip")
        for channel in channels:
            receive(channel, "queued", timeout=30)
        control.run(flags, stamps, args.control_device, args.delay_us * 1000)
        torch.cuda.synchronize(args.control_device)
        values = stamps.cpu().tolist()
        if not all(values):
            raise RuntimeError(f"MPS GPU release/ack timeout: {values}")
        complete = [receive(channel, "completed", timeout=30) for channel in channels]
        if not all(result["finite"] for result in complete):
            raise RuntimeError("Nonfinite MPS output")
        return {
            "release_gap_us": (values[1] - values[0]) / 1000,
            "response_upper_bound_us": [
                (values[2] - values[0]) / 1000,
                (values[3] - values[1]) / 1000,
            ],
        }

    daemon_started = False
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"], env=mps_env, check=True, timeout=10
        )
        daemon_started = True
        for partition in partitions:
            processes, channels, metadata = [], [], []
            original_env = dict(os.environ)
            try:
                for phase, percentage in enumerate(partition):
                    os.environ.update(mps_env)
                    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
                    parent, child = ctx.Pipe()
                    process = ctx.Process(
                        target=worker, args=(vars(args), phase, handle, child)
                    )
                    process.start()
                    child.close()
                    processes.append(process)
                    channels.append(parent)
                os.environ.clear()
                os.environ.update(original_env)
                metadata = [receive(channel, "ready") for channel in channels]
                servers = mps_command("get_server_list")
                clients = {
                    server: mps_command(f"get_client_list {server}")
                    for server in servers.splitlines()
                    if server.strip().isdigit()
                }
                if not clients or any(
                    str(process.pid) not in "\n".join(clients.values()).split()
                    for process in processes
                ):
                    raise RuntimeError(
                        f"Workers not confirmed as MPS clients: {clients}"
                    )
                for _ in range(3):
                    measure()
                isolated = [
                    [
                        measure(tuple(i == phase for i in range(2)))
                        for _ in range(args.iterations)
                    ]
                    for phase in range(2)
                ]
                samples = [measure() for _ in range(args.iterations)]
                row = {
                    "requested_percentages": partition,
                    "clients": metadata,
                    "mps_server_clients": clients,
                    "samples": samples,
                    "isolated_at_same_partition": isolated,
                    "response_upper_bound_us": [
                        statistics.median(
                            s["response_upper_bound_us"][i] for s in samples
                        )
                        for i in range(2)
                    ],
                }
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(partition, row["response_upper_bound_us"], flush=True)
            finally:
                os.environ.clear()
                os.environ.update(original_env)
                for process, channel in zip(processes, channels):
                    if process.is_alive():
                        try:
                            channel.send("stop")
                        except (BrokenPipeError, EOFError, OSError):
                            pass
                for process, channel in zip(processes, channels):
                    process.join(timeout=15)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
                    channel.close()
                processes, channels = [], []
    finally:
        if daemon_started:
            report["daemon_shutdown"] = mps_command("quit")
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
