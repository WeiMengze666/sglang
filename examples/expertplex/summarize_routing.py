"""Convert native SGLang stat recorder dumps into measured routing probabilities.

Record P and D separately through the existing start/stop/dump HTTP endpoints.
Use an unbounded recorder buffer (-1), no EPLB remapping, and one dump per TP
group: native stat dumps already reduce across that group. Recording perturbs
execution, so these runs must not be used as serving performance measurements.
The phase-wide probabilities average layers; per-layer counts are retained to
make that approximation visible. They do not capture token correlations.
"""

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import module

normalize_model_config = module("config").normalize_model_config


def summarize_counts(records, *, layers, experts, top_k):
    """Consume CPU integer counts [pass, layer, logical expert]."""
    if not 0 < top_k <= experts or layers <= 0:
        raise ValueError("Invalid model routing geometry")
    totals = [[0] * experts for _ in range(layers)]
    tokens = passes = 0
    for counts in records:
        if len(counts) != layers or any(len(row) != experts for row in counts):
            raise ValueError("Recorder shape does not match the model")
        if any(type(v) is not int or v < 0 for row in counts for v in row):
            raise ValueError("Routing counts must be nonnegative integers")
        sums = [sum(row) for row in counts]
        if len(set(sums)) != 1 or sums[0] % top_k:
            raise ValueError(
                "All layers must account for the same complete top-k routes"
            )
        n = sums[0] // top_k
        if any(v > n for row in counts for v in row):
            raise ValueError("An expert cannot receive a token more than once")
        if not n:
            continue
        passes += 1
        tokens += n
        for total, row in zip(totals, counts):
            for expert, count in enumerate(row):
                total[expert] += count
    if not tokens:
        raise ValueError("No recorded token routes")
    return {
        "forward_passes": passes,
        "tokens_per_layer": tokens,
        "counts_by_layer": totals,
        "probabilities_by_layer": [[v / tokens for v in row] for row in totals],
        "expert_probabilities": [
            sum(row[e] for row in totals) / (tokens * layers) for e in range(experts)
        ],
    }


def summarize_files(paths, **geometry):
    import torch

    paths = [Path(p).resolve() for p in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate recorder file")
    provenance = []

    def records():
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=True)
            counts = data.get("logical_count")
            if not isinstance(counts, torch.Tensor) or counts.ndim != 3:
                raise ValueError(f"Not a native stat recorder dump: {path}")
            provenance.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            yield from counts.tolist()

    return summarize_counts(records(), **geometry) | {"sources": provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prefill-records", type=Path, nargs="+", required=True)
    p.add_argument("--decode-records", type=Path, nargs="+", required=True)
    p.add_argument(
        "--provenance", required=True, help="Workload, weights and recorder settings"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.provenance.strip():
        p.error("Describe the measured workload and recorder settings")
    if set(map(Path.resolve, args.prefill_records)) & set(
        map(Path.resolve, args.decode_records)
    ):
        p.error("P and D must use separate recorder dumps")
    config_path = args.model_path / "config.json"
    config = normalize_model_config(json.loads(config_path.read_text()))
    geometry = {
        "layers": config["num_hidden_layers"],
        "experts": config["num_experts"],
        "top_k": config["num_experts_per_tok"],
    }
    report = {
        "scope": __doc__,
        "provenance": args.provenance,
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "geometry": geometry,
    }
    for phase in ("prefill", "decode"):
        report[phase] = summarize_files(getattr(args, phase + "_records"), **geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
