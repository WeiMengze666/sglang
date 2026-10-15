"""Collect measured native/APK profiles and section 7.4 scheduling overhead.

P sweeps controlled rows per expert; D sweeps active experts with one row each.
These are isolated BF16 operator profiles. A fixed SM limit is used for each
native kernel and APK grid, with no competing work. The APK measurements also
include its attention-clock publication/ack path, as do native sharing tests.
This does not supply the attention/network profiles required by the optimizer.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def fit_line(samples):
    xs, ys = zip(*samples)
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    variance = sum((x - xbar) ** 2 for x in xs)
    if not variance:
        raise ValueError("A profile needs multiple distinct tile footprints")
    beta = sum((x - xbar) * (y - ybar) for x, y in samples) / variance
    alpha = ybar - beta * xbar
    residuals = [y - (alpha + beta * x) for x, y in samples]
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": 0.0,
        "delta": 0.0,
        "rmse_s": (sum(r * r for r in residuals) / len(residuals)) ** 0.5,
        "nonnegative_monotone": alpha >= 0 and beta >= 0,
        "samples": [{"tile_footprint": x, "latency_s": y} for x, y in samples],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sms", type=int, nargs="+", default=[32, 64, 84, 116, 148])
    p.add_argument(
        "--prefill-expert-rows",
        type=int,
        nargs="+",
        default=[1, 128, 1024, 4096, 8192, 32768],
    )
    p.add_argument(
        "--decode-active-experts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 40]
    )
    p.add_argument(
        "--operations",
        nargs="+",
        choices=("gateup", "down", "moe"),
        default=["gateup", "down", "moe"],
    )
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if len(args.prefill_expert_rows) != len(args.decode_active_experts):
        p.error("Pair each P profile size with one D active-expert count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    report = {
        "scope": __doc__,
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "profile_operations.py",
                "bench_sharing.py",
                "bench_apk_sharing.py",
            )
        },
        "settings": vars(args) | {"output_dir": str(args.output_dir)},
        "cases": [],
    }
    try:
        for sms in args.sms:
            for operation in args.operations:
                for p_rows, active in zip(
                    args.prefill_expert_rows, args.decode_active_experts
                ):
                    row = {
                        "sms": sms,
                        "operation": operation,
                        "prefill_expert_rows": p_rows,
                        "decode_active_experts": active,
                    }
                    report["cases"].append(row)
                    for backend in ("native", "apk"):
                        stem = f"{backend}-{operation}-sm{sms}-p{p_rows}-d{active}"
                        path = args.output_dir / (stem + ".json")
                        command = [
                            sys.executable,
                            str(
                                root
                                / (
                                    "bench_sharing.py"
                                    if backend == "native"
                                    else "bench_apk_sharing.py"
                                )
                            ),
                            "--operation",
                            operation,
                            "--prefill-expert-rows",
                            str(p_rows),
                            # Forty local experts/top-8 require this aggregate
                            # token capacity for the controlled packed rows.
                            "--prefill-rows",
                            str(max(8192, p_rows * 5)),
                            "--decode-active-experts",
                            str(active),
                            "--iterations",
                            str(args.iterations),
                            "--output",
                            str(path),
                        ]
                        command += (
                            ["--modes", "exclusive", "--num-sms", str(sms)]
                            if backend == "native"
                            else [
                                "--ctas",
                                str(sms),
                                "--budgets",
                                str(max(2, sms // 4 * 2)),
                            ]
                        )
                        with (args.output_dir / (stem + ".log")).open("w") as log:
                            subprocess.run(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=300,
                            )
                        evidence = json.loads(path.read_text())
                        if backend == "native":
                            latency = evidence["isolated_response_upper_bound_us"]
                        else:
                            isolated = evidence["cases"][0]["isolated"]
                            latency = [
                                statistics.median(
                                    s["response_upper_bound_us"][i] for s in isolated[i]
                                )
                                for i in range(2)
                            ]
                        row[backend] = {
                            "command": command,
                            "evidence": str(path),
                            "isolated_us": latency,
                            "expert_rows": evidence["expert_rows"],
                        }
                    if row["native"]["expert_rows"] != row["apk"]["expert_rows"]:
                        raise AssertionError("Native/APK expert load vectors differ")
                    row["apk_overhead_us"] = [
                        a - n
                        for a, n in zip(
                            row["apk"]["isolated_us"], row["native"]["isolated_us"]
                        )
                    ]
                    print(
                        sms,
                        operation,
                        p_rows,
                        active,
                        row["apk_overhead_us"],
                        flush=True,
                    )
        fits = {}
        for sms in args.sms:
            for phase, index in (("prefill", 0), ("decode", 1)):
                data = [
                    row
                    for row in report["cases"]
                    if row["sms"] == sms and row["operation"] == "moe"
                ]
                if data:
                    samples = [
                        (
                            sum(
                                (count + 127) // 128
                                for count in row["apk"]["expert_rows"][index]
                            ),
                            row["apk"]["isolated_us"][index] / 1e6,
                        )
                        for row in data
                    ]
                    fits[f"{phase}:{sms}"] = fit_line(samples)
        report["moe_linear_fits"] = fits
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "profiles.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
