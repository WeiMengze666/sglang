"""Compare next-token probabilities under identical token histories.

Use a small real BF16 fixture for full-vocabulary checks, or explicitly select
top-k diagnostics for the full model. Free-running
generations diverge after the first different token; their later logits are
not directly comparable. This tool fixes histories from a reference run and
reports numerical differences without claiming task accuracy or exact tokens.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import urllib.request
from pathlib import Path


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def compare_distributions(ref, report):
    """Compare saved same-history scores without issuing another model request."""
    if (
        ref["model_path"] != report["model_path"]
        or ref["vocab_size"] != report["vocab_size"]
        or len(ref["outputs"]) != len(report["outputs"])
        or not report["outputs"]
    ):
        raise ValueError("Reference checkpoint/vocabulary/history count mismatch")
    vocab = report["vocab_size"]
    comparisons = []
    for before, after in zip(ref["outputs"], report["outputs"]):
        if before["history"] != after["history"]:
            raise ValueError("Reference histories differ")
        left, right = before["logprobs"], after["logprobs"]
        if (
            len(left) != vocab
            or len(right) != vocab
            or any(
                value is not None and not math.isfinite(value) for value in left + right
            )
        ):
            raise ValueError("Recorded scores have wrong shape or nonfinite values")
        common = [
            i for i in range(vocab) if left[i] is not None and right[i] is not None
        ]
        if not common:
            raise ValueError("No common scored token IDs")
        lp, rp = (
            [math.exp(left[i]) for i in common],
            [math.exp(right[i]) for i in common],
        )
        ordered = sorted(
            (i for i in range(vocab) if left[i] is not None),
            key=left.__getitem__,
            reverse=True,
        )[:2]
        first = ordered[0]
        candidate = max(
            (i for i in range(vocab) if right[i] is not None), key=right.__getitem__
        )
        differences = [left[i] - right[i] for i in common]
        comparisons.append(
            {
                "history_tokens": len(before["history"]),
                "common_scored_tokens": len(common),
                "reference_mass_on_common_tokens": sum(lp),
                "candidate_mass_on_common_tokens": sum(rp),
                "max_abs_logprob_difference": max(abs(x) for x in differences),
                "rms_logprob_difference": math.sqrt(
                    sum(x * x for x in differences) / len(common)
                ),
                "total_variation_lower_bound": (
                    sum(abs(a - b) for a, b in zip(lp, rp)) + abs(sum(lp) - sum(rp))
                )
                / 2,
                "kl_reference_to_candidate": (
                    sum(a * b for a, b in zip(lp, differences))
                    if len(common) == vocab
                    else None
                ),
                "reference_top1_margin": (
                    left[first] - left[ordered[1]] if len(ordered) > 1 else None
                ),
                "reference_top1": first,
                "candidate_top1": candidate,
                "top1_equal": first == candidate,
            }
        )
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument(
        "--histories", type=Path, required=True, help="check_e2e.py reference report"
    )
    p.add_argument(
        "--reference", type=Path, help="Previously recorded distribution report"
    )
    p.add_argument("--steps", type=int, nargs="+", default=[0, 1, 7, 31, 63, 127])
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Use supported P/D top-k metadata (at most 128); zero requests full vocabulary",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--flush-between",
        action="store_true",
        help="Serial uncached histories to isolate prefix-cache effects",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.steps) < 0 or args.concurrency < 1:
        p.error("Use nonnegative history steps and positive concurrency")
    if not 0 <= args.top_k <= 128:
        p.error("P/D metadata supports up to 128 top logprobs")
    config = json.loads((args.model_path / "config.json").read_text())
    vocab = config["vocab_size"]
    if vocab > 8192 and not args.top_k:
        p.error("Full-vocabulary HTTP diagnostics are restricted to small fixtures")
    source = read(args.histories)
    if source["model_path"] != str(args.model_path.resolve()):
        p.error("History checkpoint differs from the requested model")
    histories, seen = [], set()
    for result in source["outputs"]:
        prompt = result["prompt"]
        if not isinstance(prompt, list):
            p.error("Reference histories must use explicit token IDs")
        if tuple(prompt) in seen:
            continue
        seen.add(tuple(prompt))
        continuation = [
            item[1] for item in result["meta_info"]["output_token_logprobs"]
        ]
        for step in args.steps:
            if step <= len(continuation):
                histories.append(prompt + continuation[:step])

    def measure(history):
        if args.flush_between:
            with urllib.request.urlopen(
                urllib.request.Request(
                    args.base_url.rstrip("/") + "/flush_cache", method="POST"
                ),
                timeout=30,
            ) as response:
                response.read()
        payload = {
            "input_ids": history,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
            "return_logprob": True,
            **(
                {"top_logprobs_num": args.top_k}
                if args.top_k
                else {"token_ids_logprob": list(range(vocab))}
            ),
        }
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        field = "output_top_logprobs" if args.top_k else "output_token_ids_logprobs"
        entries = result["meta_info"][field][0]
        scores = {item[1]: item[0] for item in entries}
        if len(scores) != (min(args.top_k, vocab) if args.top_k else vocab) or not all(
            math.isfinite(v) for v in scores.values()
        ):
            raise ValueError("Missing or nonfinite requested probabilities")
        return {"history": history, "logprobs": [scores.get(i) for i in range(vocab)]}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1 if args.flush_between else args.concurrency
    ) as pool:
        outputs = list(pool.map(measure, histories))
    report = {
        "scope": __doc__,
        "model_path": str(args.model_path.resolve()),
        "vocab_size": vocab,
        "requested_top_k": args.top_k,
        "probability_scope": "top-k subset" if args.top_k else "full vocabulary",
        "flush_between_histories": args.flush_between,
        "history_file_sha256": hashlib.sha256(args.histories.read_bytes()).hexdigest(),
        "outputs": outputs,
    }
    if args.reference:
        report["comparisons"] = compare_distributions(read(args.reference), report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
