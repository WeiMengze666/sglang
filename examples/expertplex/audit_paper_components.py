"""Audit paper component evidence without loading raw traces or launching CUDA."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/home/axis/mengze/expertplex")


def read(rel):
    path = ROOT / rel
    if not path.exists():
        return {"path": str(path), "present": False}
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        return {"path": str(path), "present": True, "read_error": repr(exc)}
    return {"path": str(path), "present": True, "value": value}


def main():
    evidence = {
        "apk_universal": read("artifacts/20261103-preemption-universal-bound-v4/report.json"),
        "stage_timing": read("artifacts/20261029-recovery-stage-timing-coder/timing-analysis.json"),
        "stage_components_4e": read("artifacts/20261029-recovery-stage-components/gate.json"),
        "stage_components_5e": read("artifacts/20261029-recovery-stage-components-world5-heldout/gate.json"),
        "communication_component": read("artifacts/20261028-recovery-components/gate.json"),
        "warp_propagation": read("artifacts/20261028-recovery-warp-propagation/gate.json"),
        "strict_long": read("artifacts/20261102-recovery-compiled-long-default-v1/gate.json"),
        "fair_fa4": read("artifacts/20261103-fa4-phase-policy-fair-v3/p-tbo-d-none/run/sweep/joint-slo.json"),
        "fair_flashinfer": read("artifacts/20261104-flashinfer-phase-policy-fair-v1/p-tbo-d-none/run/sweep/joint-slo.json"),
    }
    audit = {
        "status": "partial",
        "scope": "CPU/read-only evidence inventory; no CUDA, serving, or raw trace loading",
        "evidence": evidence,
        "paper_gates": {
            "section_7_3_apk_vs_exclusive_priority_mps_green_context_pareto": "missing",
            "section_7_4_deepgemm_active_expert_overhead_sweep": "partial_stage_timing_only",
            "section_7_5_deepep_normal_low_latency_communication_comparison": "missing_or_not_comparable",
            "section_7_6_preemption_interval_all_operation_sweep": "component_gate_present_but_pareto_missing",
            "end_to_end_goodput_1_5x": "failed",
        },
        "next": "Do not label paper Pareto complete or claim 1.5x; any new GPU job needs a unique causal hypothesis and gpuq cleanup gate.",
    }
    out = ROOT / "evidence/20261027-new-b300/paper-component-audit.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": audit["status"], "output": str(out), "gates": audit["paper_gates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
