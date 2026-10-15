"""Freeze a held-out workload disjoint from a seeded custom-dataset pilot.

Input must come from prepare_workloads.py: all rows already satisfy the same
tokenizer/length filter. The pinned custom loader shuffles once with Python's
seeded RNG before selecting requests. Exclusion is by complete conversation,
so duplicate source records cannot leak into the held-out set.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def identity(row):
    return hashlib.sha256(
        json.dumps(row["conversations"], sort_keys=True).encode()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pilot-seed", type=int, default=42)
    p.add_argument("--pilot-requests", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--requests", type=int, default=256)
    args = p.parse_args()
    if args.output.exists() or min(args.pilot_requests, args.requests) <= 0:
        p.error("Use a fresh output and positive request counts")
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    if any(
        len(r["conversations"]) != 2 or min(r["prompt_len"], r["output_len"]) < 2
        for r in rows
    ):
        p.error("Input must be the validated two-turn prepare_workloads output")
    pilot = list(rows)
    random.Random(args.pilot_seed).shuffle(pilot)
    excluded = {identity(r) for r in pilot[: args.pilot_requests]}
    remaining = [r for r in rows if identity(r) not in excluded]
    random.Random(args.seed).shuffle(remaining)
    selected = remaining[: args.requests]
    if len(selected) != args.requests:
        p.error("Insufficient held-out requests")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    )
    manifest = {
        "scope": __doc__,
        "settings": vars(args) | {"input": str(args.input), "output": str(args.output)},
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "excluded_conversation_sha256": sorted(excluded),
        "selected_conversation_sha256": [identity(r) for r in selected],
        "selected_request_ids": [r["id"] for r in selected],
        "mean_prompt_tokens": sum(r["prompt_len"] for r in selected) / len(selected),
        "mean_output_tokens": sum(r["output_len"] for r in selected) / len(selected),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Wrote {len(selected)} held-out requests; excluded {len(excluded)} pilot conversations"
    )


if __name__ == "__main__":
    main()
