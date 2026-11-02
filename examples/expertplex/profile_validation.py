"""CPU-only instrumentation checks for isolated attention calibration reports.

Every event-on replica must meet the fixed 3% overhead gate.  Averaging event-on
replicas before this check can hide a failing measurement.  This module neither
fits models nor accepts heldout prediction accuracy or serving performance.
"""

import math
import statistics

MAXIMUM_INSTRUMENTATION_RATIO = 1.03
CONTRACT_KEYS = (
    "phase",
    "attention_backend",
    "prefill_graph_backend",
    "prefill_graph_max_tokens",
    "batches",
    "lengths",
    "holdout_batches",
    "holdout_lengths",
    "iterations",
    "cpu_threads",
    "kv_tokens",
    "disable_flashinfer_autotune",
    "measure_split",
    "measurement_only",
    "client_id",
)


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Instrumentation timings must be finite and positive")
    return value


def _layer_ids(report):
    ids, count = report["measured_layer_ids"], report["model_num_layers"]
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or not 0 <= i < count for i in ids)
        or len(ids) != len(set(ids))
        or ids != sorted(ids)
    ):
        raise ValueError("Invalid instrumentation layer cohort")
    return ids


def _table(report, phase, mode, iterations, expected):
    rows = {}
    for row in report["samples"]:
        key = (row["split"], row["x"], row["sequence_length"])
        if (
            type(row["x"]) is not int
            or type(row["sequence_length"]) is not int
            or key in rows
            or key not in expected
            or row["phase"] != phase
        ):
            raise ValueError("Duplicate or mismatched instrumentation shape")
        if row["measured_layer_ids"] != report["measured_layer_ids"]:
            raise ValueError("Sample layer cohort differs from its report")
        durations, proofs = row["forward_samples_s"], row["execution_proof"]
        if len(durations) != iterations or len(proofs) != iterations:
            raise ValueError("Missing instrumentation repetitions or execution proof")
        for duration in durations:
            _positive(duration)
        for proof in proofs:
            if (
                proof.get("phase") != phase
                or type(proof.get("batch")) is not int
                or proof.get("batch") != row["x"]
                or type(proof.get("can_run_graph")) is not bool
                or proof.get("sequence_lengths") != [row["sequence_length"]] * row["x"]
            ):
                raise ValueError("Invalid native instrumentation execution identity")
        if any(proof != proofs[0] for proof in proofs):
            raise ValueError("Native execution changed between repetitions")
        digest = row["last_logits_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Missing full-logits identity")
        if mode == "off" and row["layer_samples_s"]:
            raise ValueError("Event-off control contains layer measurements")
        if mode == "on":
            layer_samples = row["layer_samples_s"]
            if len(layer_samples) != iterations or any(
                len(values) != len(report["measured_layer_ids"])
                for values in layer_samples
            ):
                raise ValueError("Missing cohort layer measurements")
            for values in layer_samples:
                for value in values:
                    _positive(value)
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("Missing requested instrumentation shapes")
    return rows


def validate_instrumentation_replicas(variants):
    """Validate one off/on/on/off cohort; never average away a failed replica.

    Reports use the measurement-only collector contract. All model layers still
    execute; only the two on reports must time the same layer cohort. The outer
    forward timer is present in controls and measurements. Durations, rather
    than timestamps from different devices, are compared.
    """
    try:
        if not isinstance(variants, (list, tuple)) or len(variants) != 4:
            raise ValueError("Require an off/on/on/off instrumentation sequence")
        first = variants[0]
        settings = first["settings"]
        contract = {key: settings[key] for key in CONTRACT_KEYS}
        contract["component"] = settings.get("component")
        phase, split, iterations = (
            settings["phase"], settings["measure_split"], settings["iterations"]
        )
        if (
            phase not in ("prefill", "decode")
            or split not in ("train", "holdout")
            or settings["measurement_only"] is not True
            or type(iterations) is not int
            or iterations <= 0
            or not isinstance(first["source_node"], dict)
            or not first["source_node"]
        ):
            raise ValueError("Invalid instrumentation measurement contract")
        grid = [tuple(shape) for shape in first["measurement_grid"]]
        if (
            not grid
            or len(grid) != len(set(grid))
            or any(
                len(shape) != 3
                or shape[0] != split
                or any(type(v) is not int or v <= 0 for v in shape[1:])
                for shape in grid
            )
        ):
            raise ValueError("Invalid requested instrumentation grid")
        # These collectors perform three decode steps after the prompt.
        expected = {
            (kind, batch, length + (3 if phase == "decode" else 0))
            for kind, batch, length in grid
        }
        run_ids = [report["settings"]["output_dir"] for report in variants]
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or len(set(run_ids)) != 4:
            raise ValueError("ABBA controls and replicas must be distinct runs")
        tables = []
        for report, mode in zip(variants, ("off", "on", "on", "off")):
            actual_contract = {key: report["settings"][key] for key in CONTRACT_KEYS}
            actual_contract["component"] = report["settings"].get("component")
            if (
                actual_contract != contract
                or report["source_node"] != first["source_node"]
                or report["model_num_layers"] != first["model_num_layers"]
                or [tuple(shape) for shape in report["measurement_grid"]] != grid
                or report["settings"]["timing_events"] != mode
                or report["status"]
                != ("control_completed" if mode == "off" else "measurement_completed")
            ):
                raise ValueError("Mismatched instrumentation ABBA identity")
            _layer_ids(report)
            tables.append(_table(report, phase, mode, iterations, expected))
        if variants[1]["measured_layer_ids"] != variants[2]["measured_layer_ids"]:
            raise ValueError("Event-on replicas measure different layer cohorts")
        points = []
        for key in sorted(expected):
            rows = [table[key] for table in tables]
            if any(
                any(
                    row[name] != rows[0][name]
                    for name in ("last_logits_sha256", "last_logits_shape", "last_logits_dtype")
                )
                or row["execution_proof"] != rows[0]["execution_proof"]
                for row in rows
            ):
                raise ValueError("Instrumentation changed logits or native execution")
            medians = [statistics.median(row["forward_samples_s"]) for row in rows]
            baseline = statistics.mean((medians[0], medians[3]))
            ratios = [medians[index] / baseline for index in (1, 2)]
            points.append(
                {
                    "shape": list(key),
                    "control_forward_s": [medians[0], medians[3]],
                    "event_on_forward_s": [medians[1], medians[2]],
                    "event_overhead_ratios": ratios,
                    "passed": all(ratio <= MAXIMUM_INSTRUMENTATION_RATIO for ratio in ratios),
                }
            )
        return {
            "status": "passed" if all(point["passed"] for point in points) else "failed",
            "maximum_instrumentation_ratio": MAXIMUM_INSTRUMENTATION_RATIO,
            "measured_layer_ids": variants[1]["measured_layer_ids"],
            "points": points,
            "scope": "Per-replica instrumentation only; no heldout model or serving acceptance",
        }
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed instrumentation report") from exc
