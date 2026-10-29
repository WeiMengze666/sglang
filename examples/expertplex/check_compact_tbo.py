"""Compare full and compact TBO mailboxes on the same small BF16 checkpoint.

Uses the actual SGLang P/D path with three P clients, one D client, and four E
workers. Boundary prompts cover odd/even half/full capacity and concurrent
requests. This is a functional regression, not trained-480B accuracy or goodput.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import module
from check_e2e import compare, generate
from launch import wait_http


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=31000)
    args = p.parse_args()
    config = module("config")
    original = config.load_config(args.config)
    if original.weight_mode != "checkpoint":
        p.error("Use a saved BF16 checkpoint for the two-deployment comparison")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"scope": __doc__, "cases": []}
    reference = None

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for capacity in (1024, 512):
            node = original.to_dict()
            node.update(
                max_prefill_tokens=1024,
                max_decode_tokens=32,
                prefill_mailbox_tokens=capacity,
                overlap="tbo",
            )
            for client in node["clients"]:
                client["phase"] = "prefill" if client["id"] < 3 else "decode"
            case = args.output_dir / str(capacity)
            case.mkdir()
            node["rendezvous_dir"] = str((case / "rendezvous").resolve())
            path = case / "node.json"
            config.NodeConfig.from_dict(node).write_json(path)
            command = [
                sys.executable,
                str(Path(__file__).with_name("launch.py")),
                "--config",
                str(path),
                "--port",
                str(args.port),
                "--kv-tokens",
                "8192",
                "--sequence-cap",
                "2048",
                "--skip-tokenizer-init",
                "--cuda-graph",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--log-dir",
                str(case / "service"),
            ]
            env = dict(os.environ)
            env.pop("SGLANG_EXPERTPLEX_SYNTHETIC_SEED", None)
            url = f"http://127.0.0.1:{args.port}"
            with (case / "launcher.log").open("w") as log:
                proc = subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    wait_http(
                        url,
                        [("launcher", proc, log)],
                        time.monotonic() + 600,
                        expected_workers=len(node["clients"]),
                    )
                    lengths = (1, 511, 512, 513, 1023, 1024)
                    prompts = [[(i * 17 + 3) % 1000 for i in range(n)] for n in lengths]
                    outputs = [generate(url, prompt, 16) for prompt in prompts]
                    raw = {"model_path": original.model_path, "outputs": outputs}
                    row = {"capacity": capacity, "rows": list(lengths), "passed": True}
                    if reference is not None:
                        comparison = compare(reference, raw)
                        row["all_exact_token_match"] = comparison[
                            "all_exact_token_match"
                        ]
                        row["comparisons"] = comparison["comparisons"]
                        if not comparison["all_exact_token_match"]:
                            raise AssertionError(
                                "Compact mailboxes changed sequential token outputs"
                            )
                    else:
                        reference = raw
                    (case / "outputs.json").write_text(json.dumps(raw, indent=2))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        concurrent_outputs = list(
                            pool.map(lambda x: generate(url, x, 16), prompts[2:] * 2)
                        )
                    (case / "concurrent-outputs.json").write_text(
                        json.dumps(concurrent_outputs, indent=2)
                    )
                    row["completed_requests"] = len(outputs) + len(concurrent_outputs)
                    report["cases"].append(row)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=90)
            for _ in range(12):
                if not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    text=True,
                ).strip():
                    break
                time.sleep(5)
            else:
                raise RuntimeError("Owned service did not release GPU resources")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "regression.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
