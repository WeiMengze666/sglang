"""Freeze a held-out workload disjoint from a seeded custom-dataset pilot.

Input must come from prepare_workloads.py: all rows already satisfy the same
tokenizer/length filter. The pinned custom loader shuffles once with Python's
seeded RNG before selecting requests. Exclusion is by complete conversation,
so duplicate source records cannot leak into the held-out set. When expanding
the candidate pool, pass the original dataset with --pilot-input: the pilot
must be reconstructed from that file, not reshuffled from the enlarged pool.
The held-out set also contains each complete conversation at most once.
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


def select_heldout(rows, pilot_rows, *, pilot_seed, pilot_requests, seed, requests):
    if min(pilot_requests, requests) <= 0 or len(pilot_rows) < pilot_requests:
        raise ValueError("Need positive counts and the complete pilot input")
    if any(
        len(r["conversations"]) != 2 or min(r["prompt_len"], r["output_len"]) < 2
        for r in rows + pilot_rows
    ):
        raise ValueError(
            "Input must be the validated two-turn prepare_workloads output"
        )
    pilot = list(pilot_rows)
    random.Random(pilot_seed).shuffle(pilot)
    excluded = {identity(r) for r in pilot[:pilot_requests]}
    unique = {}
    for row in rows:
        key = identity(row)
        if key not in excluded:
            unique.setdefault(key, row)
    remaining = list(unique.values())
    random.Random(seed).shuffle(remaining)
    if len(remaining) < requests:
        raise ValueError("Insufficient unique held-out requests")
    return remaining[:requests], excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument(
        "--pilot-input",
        type=Path,
        help="Original pilot dataset when --input is a larger candidate pool",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pilot-seed", type=int, default=42)
    p.add_argument("--pilot-requests", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--requests", type=int, default=256)
    args = p.parse_args()
    if (
        args.output.exists()
        or args.output.with_suffix(".manifest.json").exists()
        or min(args.pilot_requests, args.requests) <= 0
    ):
        p.error("Use a fresh output and positive request counts")
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    pilot_input = args.pilot_input or args.input
    pilot_rows = [
        json.loads(line)
        for line in pilot_input.read_text().splitlines()
        if line.strip()
    ]
    try:
        selected, excluded = select_heldout(
            rows,
            pilot_rows,
            pilot_seed=args.pilot_seed,
            pilot_requests=args.pilot_requests,
            seed=args.seed,
            requests=args.requests,
        )
    except ValueError as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    )
    manifest = {
        "scope": __doc__,
        "settings": vars(args)
        | {
            "input": str(args.input),
            "output": str(args.output),
            "pilot_input": str(pilot_input),
        },
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "pilot_input_sha256": hashlib.sha256(pilot_input.read_bytes()).hexdigest(),
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
