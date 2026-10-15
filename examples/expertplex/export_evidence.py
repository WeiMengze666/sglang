"""Export numerical experiment evidence without request/output or log payloads.

Raw reports remain in the local archive. This allowlist deliberately excludes
text, token histories, logprobs, IPC handles, host paths, environment and logs.
It includes source-file hashes so an authorized local reviewer can match them.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

NUMERIC_KEYS = {
    "request_rate",
    "attempted_requests",
    "completed_requests",
    "joint_slo_met_requests",
    "joint_slo_attainment",
    "observed_slo_goodput_rps",
    "unscorable_one_token_responses",
    "meets_p90_joint_slo",
    "p90_goodput_max_tested_arrival_rps",
    "sequence_cap",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "hidden",
    "intermediate",
    "experts",
    "world",
    "top_k",
    "capacity",
    "iterations",
    "graph_replays",
    "second_prefill",
    "seed",
    "weight_std",
    "prefill_rows",
    "decode_rows",
    "delay_us",
    "budgets",
    "percentages",
    "ctas",
    "num_sms",
    "expert_rows",
    "sms",
    "prefill_expert_rows",
    "decode_active_experts",
    "expected_decode_tiles",
    "decode_budget",
    "response_upper_bound_us",
    "isolated_response_upper_bound_us",
    "actual_sms",
    "requested_sms",
    "requested_percentages",
    "reported_sms",
    "kernel_sms",
    "release_gap_us",
    "finite_outputs",
    "passed",
    "max_absolute_error",
    "relative_l2_error",
    "relative_l2",
    "max_abs_difference",
    "exact_fraction",
    "slot",
    "rows",
    "layers",
    "exact_overlap_comparisons",
    "exact_graph_comparisons",
    "full_moe_pass_ms",
    "group_vs_peer_reference",
    "expert_rows",
    "tiles",
    "median_us",
    "p99_us",
    "max_us",
    "decode_first_handoff_after_admission_us",
    "admission_ack_upper_bound_us",
    "attention_admission_ack_upper_bound_us",
    "trace_capacity_per_cta",
    "benchmark_stage",
    "pipelined_stores",
    "cluster_size",
    "prefill_preprocess",
    "loading_stages",
    "threads_per_cta",
    "registers_per_thread",
    "local_bytes_per_thread",
    "dynamic_shared_bytes_per_cta",
    "static_shared_bytes_per_cta",
    "prefill_tile_rows",
    "decode_tile_rows",
    "invocations",
    "median",
    "maximum",
    "row_count",
    "max_prefill_tokens",
    "max_decode_tokens",
    "decode_sm_budget",
    "max_decode_sm_fraction",
    "expected_shards",
    "complete_shards",
    "present_safetensors_files",
    "configured_timeout_ms",
    "wall_latency_ms",
    "slot_reused",
    "fresh_owner_allocation",
    "native_reference",
    "slowdown_vs_full_gpu_isolated",
    "history_tokens",
    "common_scored_tokens",
    "reference_mass_on_common_tokens",
    "candidate_mass_on_common_tokens",
    "max_abs_logprob_difference",
    "rms_logprob_difference",
    "total_variation_lower_bound",
    "kl_reference_to_candidate",
    "reference_top1_margin",
    "top1_equal",
    "flush_between_histories",
    "requested_top_k",
    "dispatch_us",
    "combine_us",
    "step_wall_us",
    "group_transfers",
    "cuda_graph",
    "vary_routes",
    "isolated_us",
    "apk_overhead_us",
    "common_prefix_tokens",
    "reference_tokens",
    "candidate_tokens",
    "exact_token_match",
    "all_exact_token_match",
    "max_logprob_abs_difference_on_common_prefix",
}
CONTAINERS = {
    "settings",
    "cases",
    "results",
    "samples",
    "isolated",
    "isolated_at_same_partition",
    "clients",
    "executor",
    "executors",
    "trace_task_counts",
    "diagnostic_tile_interval_us",
    "intervals",
    "timeout",
    "tolerances",
    "native_reference",
    "comparisons",
    "summary",
    "median_max_rank_us",
    "max_rank_samples",
    "apk",
    "native",
}
ENUMS = {
    "status": {"passed", "failed", "running", "completed", "incomplete", "ready"},
    "phase": {"prefill", "decode"},
    "mode": {"exclusive", "priority", "green", "normal", "low_latency"},
    "operation": {
        "gateup",
        "gate_up",
        "down",
        "moe",
        "activation",
        "scatter1",
        "scatter2",
        "gather",
    },
    "backend": {"sm100", "sm100_fused", "reference", "expertplex", "deepep"},
    "workload_identity": {"cpu-bf16-expert-chunks-v1"},
}


def numeric(value):
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        result = [numeric(item) for item in value]
        return (
            result
            if all(v is not None or old is None for v, old in zip(result, value))
            else None
        )
    return None


def redact(value):
    if isinstance(value, list):
        return [
            redact(item) if isinstance(item, (dict, list)) else numeric(item)
            for item in value
        ]
    if not isinstance(value, dict):
        return numeric(value)
    result = {}
    for key, item in value.items():
        if key in CONTAINERS or key.isdecimal():
            result[key] = redact(item)
        elif key in NUMERIC_KEYS:
            parsed = numeric(item)
            if parsed is not None or item is None:
                result[key] = parsed
        elif key in ENUMS and isinstance(item, str) and item in ENUMS[key]:
            result[key] = item
        elif (
            key == "native_source_sha256"
            and isinstance(item, str)
            and len(item) == 64
            and all(c in "0123456789abcdef" for c in item)
        ):
            result[key] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        raw = path.read_bytes()
        source = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        records.append(
            {
                "name": path.name,
                "source_file_sha256": hashlib.sha256(raw).hexdigest(),
                "failure_recorded": source.get("status") == "failed",
                "metrics": redact(source),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"scope": __doc__, "records": records}, indent=2) + "\n"
    )
    print(
        f"Exported {len(records)} numerical reports; request/output/log payloads excluded"
    )


if __name__ == "__main__":
    main()
