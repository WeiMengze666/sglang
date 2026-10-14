"""Run the ExpertPlex CPU contracts and syntax checks without installing SGLang."""

import compileall
import subprocess
import sys

from _bootstrap import ROOT


def main():
    for directory in ("test/registered/expertplex", "test/registered/unit/expertplex"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                directory,
                "-p",
                "test_*cpu.py"
                if directory.endswith("/expertplex") and "/unit/" not in directory
                else "test_*.py",
                "-v",
            ],
            cwd=ROOT,
            check=True,
        )
    for directory in (
        "python/sglang/srt/expertplex",
        "examples/expertplex",
        "test/registered/expertplex",
        "test/registered/unit/expertplex",
    ):
        if not compileall.compile_dir(ROOT / directory, quiet=1):
            raise SystemExit(f"Python syntax check failed: {directory}")
    for script in (
        "configure.py",
        "launch.py",
        "check_gpu.py",
        "check_e2e.py",
        "bench_sweep.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "examples/expertplex" / script), "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    print("CPU contracts, Python syntax, CLI help and whitespace checks passed.")
    print(
        "This CPU-only command does not cover CUDA/GPU, full-model accuracy or goodput."
    )


if __name__ == "__main__":
    main()
