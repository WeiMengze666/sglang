"""TBO proof must reject padding-only and stale replay metadata."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/expertplex"))
validate = importlib.import_module("check_tbo_completion").validate


def snapshot(index, ranks=4, pid=100):
    return {
        "pid": pid,
        "tbo_debug": {
            "phase": "decode",
            "capture_ns": index * 100 + 50,
            "replay": {
                "prepare_ns": index * 100,
                "child_tokens": [8, 5],
                "actual_tokens": 13,
                "graph_bs": 16,
            },
            "mailbox_timings": [
                [[index, index, 10, 20] for _ in range(ranks)] for _ in range(2)
            ],
        },
    }


class TboCompletionTests(unittest.TestCase):
    def test_fresh_completed_children_on_every_expert(self):
        rows = [snapshot(i) for i in (1, 2, 3)]
        accepted = validate(reversed(rows))
        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0]["child_tokens"], [8, 5])
        self.assertEqual(
            len(validate([snapshot(i, 2) for i in (1, 2, 3)], expert_count=2)), 2
        )

    def test_empty_stale_and_incomplete_replays_do_not_count(self):
        for failure in (
            "padding",
            "stale",
            "unfinished",
            "no_progress",
            "future",
            "prefill",
        ):
            with self.subTest(failure=failure):
                rows = [snapshot(i) for i in (1, 2, 3)]
                for row in rows[1:]:
                    meta = row["tbo_debug"]
                    if failure == "padding":
                        meta["replay"]["child_tokens"] = [8, 0]
                    elif failure == "stale":
                        meta["replay"]["prepare_ns"] = 0
                    elif failure == "future":
                        meta["replay"]["prepare_ns"] = meta["capture_ns"] + 1
                    elif failure == "prefill":
                        meta["phase"] = "prefill"
                    elif failure == "unfinished":
                        meta["mailbox_timings"][1][3][0] = 0
                    else:
                        meta["mailbox_timings"] = copy.deepcopy(
                            rows[0]["tbo_debug"]["mailbox_timings"]
                        )
                with self.assertRaisesRegex(ValueError, "fresh completed"):
                    validate(rows)

    def test_no_cross_process_baseline_and_no_vacuous_pass(self):
        with self.assertRaisesRegex(ValueError, "fresh completed"):
            validate([snapshot(1, pid=1), snapshot(2, pid=2), snapshot(3, pid=3)])
        with self.assertRaises(ValueError):
            validate([], min_replays=0)

    def test_malformed_counts_or_missing_expert_are_rejected(self):
        rows = [snapshot(i) for i in (1, 2, 3)]
        rows[1]["tbo_debug"]["replay"]["actual_tokens"] = 14
        with self.assertRaisesRegex(ValueError, "child counts"):
            validate(rows)
        rows = [snapshot(i) for i in (1, 2, 3)]
        rows[1]["tbo_debug"]["mailbox_timings"][1].pop()
        with self.assertRaisesRegex(ValueError, "every E rank"):
            validate(rows)


if __name__ == "__main__":
    unittest.main()
