"""Freeze dataset requests and lengths before comparing serving systems.

The paper does not publish its exact LooGLE subset/prompt serialization. This
adapter uses official longdep_qa records as context + question and preserves
the reference-answer length. It filters, never truncates or rewrites, requests.
The resulting distribution is an explicitly recorded reproduction choice.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("sharegpt", "loogle"), required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--sequence-cap", type=int, default=24576)
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.sequence_cap < 4 or args.requests < 1:
        p.error("Use a positive request count and a sequence cap >=4")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        p.error("Output exists; use a fresh path")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.kind == "sharegpt":
        raw = json.loads(args.input.read_text())
        pairs = []
        for index, row in enumerate(raw):
            turns = row.get("conversations", row.get("conversation", []))
            if len(turns) >= 2:
                pairs.append(
                    (str(row.get("id", index)), turns[0]["value"], turns[1]["value"])
                )
    else:
        with args.input.open() as f:
            raw = [json.loads(line) for line in f if line.strip()]
        pairs = [
            (row["id"], row["context"] + "\n\n" + row["question"], row["answer"])
            for row in raw
        ]
    random.Random(args.seed).shuffle(pairs)
    selected, excluded, examined = [], {"too_short": 0, "over_cap": 0}, 0
    for identity, prompt, completion in pairs:
        examined += 1
        prompt_len = len(tokenizer.encode(prompt))
        output_len = len(tokenizer.encode(completion))
        if min(prompt_len, output_len) < 2:
            excluded["too_short"] += 1
            continue
        if prompt_len + output_len > args.sequence_cap:
            excluded["over_cap"] += 1
            continue
        selected.append(
            {
                "id": identity,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": completion},
                ],
                "prompt_len": prompt_len,
                "output_len": output_len,
            }
        )
        if len(selected) == args.requests:
            break
    if not selected:
        raise ValueError("No requests meet the frozen length constraints")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(values):
        values = sorted(values)
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    manifest = {
        "scope": __doc__,
        "kind": args.kind,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "tokenizer": args.tokenizer,
        "sequence_cap": args.sequence_cap,
        "seed": args.seed,
        "requested": args.requests,
        "selected": len(selected),
        "candidate_records": len(pairs),
        "examined": examined,
        "excluded": excluded,
        "prompt_tokens": stats([row["prompt_len"] for row in selected]),
        "output_tokens": stats([row["output_len"] for row in selected]),
        "request_ids": [row["id"] for row in selected],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "request_ids"}, indent=2)
    )
    if len(selected) != args.requests:
        raise ValueError(
            f"Only {len(selected)} eligible requests; do not silently benchmark fewer than requested"
        )


if __name__ == "__main__":
    main()
