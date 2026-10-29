"""Inspect downloaded safetensors headers without loading model weights.

This checks shard presence, sizes, indexed tensor names and BF16 dtype. It does
not verify tensor values or cryptographic content hashes; downloading files may
still change, so the checkpoint must remain immutable during a deployment.
"""

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path


def inspect_checkpoint(root):
    root = Path(root).resolve()
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    report = {
        "scope": __doc__,
        "model_path": str(root),
        "present_safetensors_files": sum(
            1 for p in root.glob("*.safetensors") if p.is_file()
        ),
        "shards": [],
        "errors": [],
    }
    if not index_path.is_file() or not config_path.is_file():
        return report | {
            "status": "incomplete",
            "reason": "Missing config or shard index",
        }
    index = json.loads(index_path.read_text())
    report["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    by_shard = collections.defaultdict(set)
    for name, shard in index["weight_map"].items():
        if Path(shard).name != shard:
            raise ValueError("Shard index must contain local basenames")
        by_shard[shard].add(name)
    dtypes = collections.Counter()
    missing = []
    for shard, names in sorted(by_shard.items()):
        path = root / shard
        if not path.is_file():
            missing.append(shard)
            continue
        try:
            before = path.stat()
            with path.open("rb") as stream:
                length = stream.read(8)
                if len(length) != 8:
                    raise ValueError("Truncated header length")
                header_size = struct.unpack("<Q", length)[0]
                if not 2 <= header_size <= 100 * 1024**2:
                    raise ValueError("Invalid safetensors header length")
                header_bytes = stream.read(header_size)
                if len(header_bytes) != header_size:
                    raise ValueError("Truncated safetensors header")
                header = json.loads(header_bytes)
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            if not names <= tensors.keys():
                raise ValueError("Indexed tensor missing from shard header")
            for name, tensor in tensors.items():
                start, end = tensor["data_offsets"]
                if not 0 <= start <= end <= before.st_size - 8 - header_size:
                    raise ValueError("Tensor payload incomplete: " + name)
                if name in names:
                    dtypes[tensor["dtype"]] += 1
                    if tensor["dtype"] != "BF16":
                        raise ValueError("Indexed tensor is not BF16: " + name)
                    elements = 1
                    for dimension in tensor["shape"]:
                        if not isinstance(dimension, int) or dimension < 0:
                            raise ValueError("Invalid tensor shape")
                        elements *= dimension
                    if end - start != elements * 2:
                        raise ValueError("BF16 shape/payload size mismatch: " + name)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Shard changed during inspection")
            report["shards"].append(
                {
                    "name": shard,
                    "bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "indexed_tensors": len(names),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append({"shard": shard, "error": str(exc)})
    report.update(
        expected_shards=len(by_shard),
        complete_shards=len(report["shards"]),
        missing_shards=missing,
        dtype_tensor_counts=dict(dtypes),
    )
    report["status"] = (
        "ready" if by_shard and not missing and not report["errors"] else "incomplete"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_checkpoint(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("shards", "missing_shards")},
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
