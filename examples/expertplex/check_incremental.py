"""Capture unforced scores along a fixed incremental generation trajectory.

Run on the server host with a fresh shared capture directory. The diagnostic
custom processor records log_softmax before forcing the reference next token.
Step zero uses prefill; later steps retain the generated KV history. Unlike
check_teacher_forcing.py, this does not re-prefill every history. CPU copies and
file writes deliberately perturb execution: never use this run for performance.
Requires --enable-custom-logit-processor on the isolated diagnostic servers.
"""

import argparse
import hashlib
import json
import math
import secrets
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_teacher_forcing import compare_distributions, read


def make_processor():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    # Keep imports inside the local class so dill can serialize it for workers
    # without requiring this example module on their Python import path.
    class CaptureAndForce(CustomLogitProcessor):
        def __call__(self, logits, custom_param_list):
            import json
            import os
            from pathlib import Path

            import torch

            if len(custom_param_list) != logits.shape[0]:
                raise ValueError("Capture batch metadata mismatch")
            for row, params in enumerate(custom_param_list):
                req = params["__req__"]
                if req.retraction_count:
                    raise ValueError(
                        "Retracted requests cannot validate retained incremental KV"
                    )
                generated = list(req.output_ids)
                step = len(generated)
                forced = params["forced_ids"]
                if step >= len(forced) or generated != forced[:step]:
                    raise ValueError(
                        "Incremental history differs from forced trajectory"
                    )
                prompt = params["prompt_ids"]
                if list(req.origin_input_ids) != prompt:
                    raise ValueError("Server changed the diagnostic prompt")
                if logits.shape[1] != params["vocab_size"]:
                    raise ValueError("Capture vocabulary mismatch")
                raw = logits[row].float()
                if not bool(torch.isfinite(raw).all()):
                    raise ValueError("Nonfinite pre-forcing model logits")
                scores = torch.log_softmax(raw, dim=-1)
                count = params["top_k"]
                if count:
                    values, ids = scores.topk(count)
                    entries = list(zip(ids.cpu().tolist(), values.cpu().tolist()))
                else:
                    entries = list(enumerate(scores.cpu().tolist()))
                record = {
                    "step": step,
                    "history": prompt + generated,
                    "forced_token": forced[step],
                    "vocab_size": params["vocab_size"],
                    "pre_forcing_logprobs": entries,
                    "pid": os.getpid(),
                    "retraction_count": req.retraction_count,
                }
                # TP ranks may each sample the gathered logits. Preserve every
                # copy and require agreement when collecting; never race writes
                # or silently select the most convenient rank.
                directory = Path(params["capture_dir"])
                target = directory / f"step-{step:06d}-pid-{os.getpid()}.json"
                with target.open("x") as stream:
                    json.dump(record, stream, allow_nan=False)
                logits[row].fill_(-float("inf"))
                logits[row, forced[step]] = 0.0
            return logits

    return CaptureAndForce.to_str()


def trajectories(source, vocab, max_steps):
    cases, seen = [], set()
    for result in source["outputs"]:
        prompt = result["prompt"]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("Reference requires nonempty explicit token prompts")
        tokens = [entry[1] for entry in result["meta_info"]["output_token_logprobs"]]
        if any(type(t) is not int or not 0 <= t < vocab for t in prompt + tokens):
            raise ValueError("Reference token ID outside vocabulary")
        if len(tokens) < max_steps:
            raise ValueError("Reference continuation is shorter than requested steps")
        if tuple(prompt) in seen:
            continue
        seen.add(tuple(prompt))
        cases.append((prompt, tokens[:max_steps]))
    if not cases:
        raise ValueError("No reference trajectories")
    return cases


def collect(directory, prompt, forced, vocab, top_k):
    by_step = {}
    for path in sorted(directory.glob("step-*-pid-*.json")):
        record = json.loads(path.read_text())
        step = record["step"]
        if type(step) is not int or not 0 <= step < len(forced):
            raise ValueError("Unexpected captured step")
        if (
            record["history"] != prompt + forced[:step]
            or record["forced_token"] != forced[step]
            or record["vocab_size"] != vocab
            or record["retraction_count"] != 0
        ):
            raise ValueError("Captured history/token/vocabulary mismatch")
        entries = record["pre_forcing_logprobs"]
        scores = dict(entries)
        if (
            len(entries) != len(scores)
            or len(scores) != (top_k or vocab)
            or any(type(i) is not int or not 0 <= i < vocab for i in scores)
            or any(not math.isfinite(v) or v > 0 for v in scores.values())
        ):
            raise ValueError("Invalid captured probability entries")
        previous = by_step.get(step)
        if previous is not None and previous != scores:
            raise ValueError("Sampling ranks disagree on pre-forcing probabilities")
        by_step[step] = scores
    if set(by_step) != set(range(len(forced))):
        raise ValueError("Missing incremental capture steps")
    return [
        {
            "history": prompt + forced[:step],
            "logprobs": [by_step[step].get(i) for i in range(vocab)],
        }
        for step in range(len(forced))
    ]


def post(base_url, endpoint, payload=None):
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()
        return json.loads(data) if payload is not None else None


def generate(base_url, payload, *, prefill_url=None, bootstrap_port=None):
    if prefill_url is None:
        post(base_url, "/flush_cache")
        return post(base_url, "/generate", payload)
    # Diagnostic-only direct P/D dispatch preserves arbitrary custom_params
    # through gateway versions whose typed sampling schema discards them. Both
    # workers still use the ordinary bootstrap rendezvous and real KV transfer.
    from urllib.parse import urlparse

    post(prefill_url, "/flush_cache")
    post(base_url, "/flush_cache")
    request = dict(
        payload,
        bootstrap_host=urlparse(prefill_url).hostname,
        bootstrap_port=bootstrap_port,
        bootstrap_room=secrets.randbits(63),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        prefill = pool.submit(post, prefill_url, "/generate", request)
        decode = pool.submit(post, base_url, "/generate", request)
        prefill.result()
        return decode.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--prefill-url", help="Direct TP1 P/D diagnostic: base-url is the D worker"
    )
    parser.add_argument("--bootstrap-port", type=int)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--require-exact-logprobs",
        action="store_true",
        help="Fail unless every full-vocabulary score equals the reference exactly",
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--top-k", type=int, default=128, help="Zero captures full vocabulary"
    )
    args = parser.parse_args()
    if args.require_exact_logprobs and (not args.reference or args.top_k != 0):
        parser.error("Exact logprob acceptance requires --reference and --top-k 0")
    if (args.prefill_url is None) != (args.bootstrap_port is None):
        parser.error("Direct P/D requires both prefill-url and bootstrap-port")
    if args.bootstrap_port is not None and not 1 <= args.bootstrap_port <= 65535:
        parser.error("Invalid bootstrap port")
    vocab = json.loads((args.model_path / "config.json").read_text())["vocab_size"]
    if args.steps < 2 or not 0 <= args.top_k <= vocab:
        parser.error("Use at least two steps and top-k within the vocabulary")
    source = read(args.histories)
    if source["model_path"] != str(args.model_path.resolve()):
        parser.error("Reference checkpoint differs from requested model")
    cases = trajectories(source, vocab, args.steps)
    args.capture_dir = args.capture_dir.resolve()
    args.capture_dir.mkdir(parents=True, exist_ok=False)
    processor = make_processor()
    report = {
        "scope": __doc__,
        "model_path": str(args.model_path.resolve()),
        "vocab_size": vocab,
        "requested_top_k": args.top_k,
        "probability_scope": "top-k subset" if args.top_k else "full vocabulary",
        "history_file_sha256": hashlib.sha256(args.histories.read_bytes()).hexdigest(),
        "capture_dir": str(args.capture_dir),
        "diagnostic_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "outputs": [],
        "cases": [],
        "status": "running",
        "transport": "direct-pd" if args.prefill_url else "single-endpoint",
        "require_exact_logprobs": args.require_exact_logprobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    save()
    try:
        for index, (prompt, forced) in enumerate(cases):
            directory = args.capture_dir / f"case-{index:04d}"
            directory.mkdir()
            result = generate(
                args.base_url,
                {
                    "input_ids": prompt,
                    "custom_logit_processor": processor,
                    "return_logprob": True,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": len(forced),
                        "ignore_eos": True,
                        "custom_params": {
                            "prompt_ids": prompt,
                            "forced_ids": forced,
                            "vocab_size": vocab,
                            "top_k": args.top_k,
                            "capture_dir": str(directory),
                        },
                    },
                },
                prefill_url=args.prefill_url,
                bootstrap_port=args.bootstrap_port,
            )
            returned = [x[1] for x in result["meta_info"]["output_token_logprobs"]]
            if returned != forced:
                raise ValueError(
                    "Server did not generate the complete forced trajectory"
                )
            outputs = collect(directory, prompt, forced, vocab, args.top_k)
            report["cases"].append({"case": index, "steps": len(outputs)})
            report["outputs"].extend(outputs)
            save()
        if args.reference:
            report["comparisons"] = compare_distributions(read(args.reference), report)
            exact = all(
                row["common_scored_tokens"] == vocab
                and row["max_abs_logprob_difference"] == 0
                for row in report["comparisons"]
            )
            report["all_full_vocab_logprobs_exact"] = exact
            if args.require_exact_logprobs and not exact:
                raise AssertionError("Full-vocabulary logprobs are not exactly equal")
        # Without the strict option, completion only means the diagnostic ran.
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
