"""Record or compare deterministic full-model outputs on serial deployments.

Functional checks and token/logprob comparisons do not establish task accuracy
or serving goodput. A reference run must use the same BF16 checkpoint.
"""

import argparse
import concurrent.futures
import json
import math
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a Python function that computes the greatest common divisor.\n",
    "请用 Python 实现二分查找，并说明边界条件。\n",
    "Explain why a hash table can have worst-case linear lookup time.\n",
    "Return a SQL query that finds duplicate email addresses in a users table.\n",
]


def generate(base_url, prompt, output_tokens):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "top_logprobs_num": 5,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    if not result.get("text") or result.get("error"):
        raise RuntimeError(f"Generation failed: {result}")
    meta = result.get("meta_info", {})
    if meta.get("completion_tokens") != output_tokens:
        raise RuntimeError(f"Unexpected output length: {meta}")
    logprobs = meta.get("output_token_logprobs", [])
    if len(logprobs) != output_tokens or any(
        not math.isfinite(item[0]) for item in logprobs
    ):
        raise RuntimeError("Missing or nonfinite per-token logprobs")
    return {"prompt": prompt, "text": result["text"], "meta_info": meta}


def compare(reference, candidate):
    if reference["model_path"] != candidate["model_path"]:
        raise ValueError(
            "Reference and candidate must use the same BF16 checkpoint path"
        )
    if len(reference["outputs"]) != len(candidate["outputs"]):
        raise ValueError("Reference and candidate prompt sets differ")
    rows = []
    for expected, actual in zip(reference["outputs"], candidate["outputs"]):
        if expected["prompt"] != actual["prompt"]:
            raise ValueError("Reference and candidate prompt ordering differs")
        left = expected["meta_info"]["output_token_logprobs"]
        right = actual["meta_info"]["output_token_logprobs"]
        prefix = 0
        for a, b in zip(left, right):
            if a[1] != b[1]:
                break
            prefix += 1
        rows.append(
            {
                "prompt": actual["prompt"],
                "common_prefix_tokens": prefix,
                "reference_tokens": len(left),
                "candidate_tokens": len(right),
                "exact_token_match": len(left) == len(right) == prefix,
                "max_logprob_abs_difference_on_common_prefix": max(
                    (abs(left[i][0] - right[i][0]) for i in range(prefix)), default=None
                ),
            }
        )
    return {
        "comparisons": rows,
        "all_exact_token_match": all(row["exact_token_match"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--require-exact-tokens", action="store_true")
    args = parser.parse_args()
    if args.output_tokens < 1 or args.rounds < 1:
        parser.error("Output token count and rounds must be positive")
    outputs = []
    for _ in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
            outputs.extend(
                pool.map(
                    lambda prompt: generate(args.base_url, prompt, args.output_tokens),
                    PROMPTS,
                )
            )
    report = {
        "base_url": args.base_url,
        "model_path": str(Path(args.model_path).resolve()),
        "outputs": outputs,
    }
    if args.reference:
        report.update(compare(json.loads(Path(args.reference).read_text()), report))
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    if args.require_exact_tokens and not report.get("all_exact_token_match", False):
        raise SystemExit("Token alignment check failed; inspect the saved comparison")
    print(f"Saved {len(outputs)} functional outputs to {args.output}")


if __name__ == "__main__":
    main()
