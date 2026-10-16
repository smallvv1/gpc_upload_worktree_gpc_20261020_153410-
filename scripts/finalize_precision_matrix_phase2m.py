#!/usr/local/bin/python3
"""Add the Phase 2M gate/decision summary to the diagnostic matrix."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def row(candidate, scope, cosine=None, rel=None, recovery=None, repeatability="NOT_RUN", decision="NOT_RUN"):
    return {
        "candidate": candidate,
        "scope": scope,
        "gradient_cosine_reverse1_block0": cosine,
        "relative_l2_reverse1_block0": rel,
        "recovery_fraction": recovery,
        "repeatability": repeatability,
        "first200_score": "NOT_RUN",
        "epoch15": "NOT_RUN",
        "epoch25": "NOT_RUN",
        "epoch60_if_any": "NOT_RUN",
        "decision": decision,
    }


def main() -> None:
    matrix_path = ROOT / "diagnostics/phase2m_precision_policy_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    fake = json.loads((ROOT / "diagnostics/phase2m_fake_quantization.json").read_text())
    am = json.loads((ROOT / "diagnostics/phase2m_precision_policy_AM.json").read_text())
    reps = json.loads((ROOT / "diagnostics/phase2m_precision_policy_repeatability.json").read_text())

    rows = []
    scopes = matrix["config"]["policy_scopes"]
    for candidate in ("current", "N", "R", "A", "AV", "M", "L", "RS", "B", "FULL32"):
        cmp = matrix["comparisons_to_full_fp32"][candidate]["reverse1"]["block_0"]
        rep = reps.get("policies", {}).get(candidate, {}).get("gradient")
        rep_value = rep if rep is not None else "NOT_RUN"
        if candidate == "current":
            decision = "REFERENCE_FORMAL_PATH"
        elif candidate == "FULL32":
            decision = "ORACLE_ONLY"
        else:
            decision = "REJECTED_PRIMARY_GRADIENT_GATE"
        rows.append(row(candidate, scopes.get(candidate, "formal current path"), cmp.get("cosine"), cmp.get("relative_l2"), matrix["recovery_reverse1_block0_input"].get(candidate), rep_value, decision))

    acmp = am["comparisons_to_full_fp32"]["reverse1"]["block_0"]
    rows.append(row("AM", am["scope"], acmp.get("cosine"), acmp.get("relative_l2"), am.get("recovery_reverse1_block0_input"), "NOT_RUN", "REJECTED_COMBINATION_NO_RECOVERY"))
    fcmp = fake["comparisons_to_full_fp32"]["FAKEQ"]["reverse1"]["block_0"]
    frep = reps.get("policies", {}).get("FAKEQ", {}).get("gradient", "NOT_RUN")
    rows.append(row("FAKEQ", fake["scope"], fcmp.get("cosine"), fcmp.get("relative_l2"), fake["recovery_reverse1_block0_input"].get("FAKEQ"), frep, "DIAGNOSTIC_CEILING_NOT_FORMAL_CANDIDATE"))

    matrix["candidate_matrix"] = rows
    matrix["gate_summary"] = {
        "primary_gradient_gate": "cosine >= 0.50 OR recovery >= 0.50",
        "minimal_policy_passed": False,
        "fakeq_boundary_diagnostic": "boundary-only quantization does not reproduce the observed degradation; FAKEQ uses FP32 block arithmetic and is ceiling-like",
        "first200_authorized": False,
        "epoch15_authorized": False,
        "epoch25_authorized": False,
        "epoch60_authorized": False,
        "final_phase_decision": "CUMULATIVE PPU BACKWARD ERROR CONFIRMED BUT MINIMAL SOURCE UNRESOLVED",
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(matrix_path), "candidates": len(rows), "long_run_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
