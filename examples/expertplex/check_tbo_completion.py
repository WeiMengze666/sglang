"""Validate completed two-child execution, not overlap duration or performance."""

import argparse
import json
from pathlib import Path


def validate(samples, *, expert_count=4, min_replays=2):
    if expert_count < 1 or min_replays < 1:
        raise ValueError("Require positive expert count and replay threshold")
    previous = {}
    accepted = []
    for row in sorted(samples, key=lambda x: x["tbo_debug"]["capture_ns"]):
        meta = row["tbo_debug"]
        pid = row["pid"]
        old = previous.get(pid)
        previous[pid] = meta
        replay = meta.get("replay")
        if meta.get("phase") != "decode" or old is None or replay is None:
            continue
        if not old["capture_ns"] < replay["prepare_ns"] <= meta["capture_ns"]:
            continue
        children = replay["child_tokens"]
        if len(children) != 2 or min(children) <= 0:
            continue
        if (
            sum(children) != replay["actual_tokens"]
            or not 0 < replay["actual_tokens"] <= replay["graph_bs"]
        ):
            raise ValueError("Invalid non-padding child counts")
        before, after = old.get("mailbox_timings"), meta.get("mailbox_timings")
        if before is None or after is None:
            continue
        if (
            len(before) != 2
            or len(after) != 2
            or any(len(v) != expert_count for v in before + after)
        ):
            raise ValueError("Need two mailboxes on every E rank")
        complete = True
        for prev_slot, slot in zip(before, after):
            for earlier, current in zip(prev_slot, slot):
                if len(earlier) != 4 or len(current) != 4:
                    raise ValueError("Invalid native timing record")
                done, ready, start, finish = current
                complete &= done == ready and done > earlier[0] and 0 < start <= finish
        if complete:
            accepted.append(
                {
                    "pid": pid,
                    "capture_ns": meta["capture_ns"],
                    "child_tokens": children,
                    "graph_bs": replay["graph_bs"],
                    "done_epochs": [[v[0] for v in slot] for slot in after],
                }
            )
    if len(accepted) < min_replays:
        raise ValueError(
            f"Need at least {min_replays} fresh completed nonempty two-child replays"
        )
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--expert-count", type=int, default=4)
    p.add_argument("--min-replays", type=int, default=2)
    a = p.parse_args()
    check(a.directory, expert_count=a.expert_count, min_replays=a.min_replays)


def check(directory, *, expert_count=4, min_replays=2):
    samples = {}
    for path in sorted((directory / "capture").rglob("step-*-pid-*.json")):
        row = json.loads(path.read_text())
        meta = row.get("tbo_debug")
        if not meta:
            continue
        key = (row["pid"], meta["capture_ns"])
        if key in samples and samples[key]["tbo_debug"] != meta:
            raise ValueError(
                "One sampling call contains conflicting diagnostic snapshots"
            )
        samples[key] = {"pid": row["pid"], "tbo_debug": meta}
    report = {"scope": __doc__, "sampling_calls": len(samples)}
    try:
        report["completed_two_child_replays"] = validate(
            samples.values(), expert_count=expert_count, min_replays=min_replays
        )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (directory / "tbo-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Verified nonempty two-child replays:",
        len(report["completed_two_child_replays"]),
    )


if __name__ == "__main__":
    main()
