#!/usr/local/bin/python3
"""PHASE 2N-A: read-only snapshot of existing 2M evidence."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def main() -> None:
    matrix = json.loads((ROOT / "diagnostics/phase2m_precision_policy_matrix.json").read_text())
    rows = matrix.get("candidate_matrix", [])
    lines = [
        "# PHASE 2N EXISTING CANDIDATE MAP",
        "",
        "This is a read-only extraction of Phase 2M artifacts; no prior diagnostic was rerun.",
        "",
        "| Candidate | Scope from artifact | Reverse-1 block0 cosine | Recovery | Decision |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['candidate']} | {row['scope']} | {row['gradient_cosine_reverse1_block0']} | {row['recovery_fraction']} | {row['decision']} |")
    lines += [
        "",
        "Interpretation:",
        "",
        "- `current` is the formal PPU autocast reference.",
        "- `N`, `R`, `A`, `AV`, `M`, `L`, `RS`, and `B` are the exact diagnostic scopes recorded in the Phase 2M matrix; none met the minimal gradient gate.",
        "- `AM` is the one permitted A+M combination and was rejected by its measured recovery.",
        "- `FULL32` is a mathematical oracle only.",
        "- `FAKEQ` uses FP32 arithmetic throughout all blocks with explicit boundary casts; it is ceiling-like and is not a sparse candidate.",
        "",
        "The Phase 2N evaluator will preserve these scopes and use the exact Phase 2M full-block wrapper for selected masks.",
    ]
    report = ROOT / "reports/PHASE_2N_EXISTING_CANDIDATE_MAP.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")

    env = {}
    for key in sorted(os.environ):
        if key.startswith(("CUDA", "PPU", "TORCH", "NCCL", "CUBLAS")):
            env[key] = os.environ[key]
    provenance = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "git_head": run("git rev-parse HEAD"),
        "git_status": run("git status --short"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": env,
        "evidence_files": {},
        "formal_path_unchanged": True,
        "phase2n_scope": "diagnostic-only sparse precision mask search; no optimizer step in mask evaluator",
    }
    for rel in [
        "reports/PHASE_2M_FINAL.md",
        "reports/PHASE_2M_BLOCK_ERROR_DECOMPOSITION.csv",
        "reports/PHASE_2M_ROUNDING_BOUNDARY.csv",
        "reports/PHASE_2M_PROVENANCE.md",
        "diagnostics/phase2m_precision_policy_matrix.json",
        "diagnostics/phase2m_block_error_decomposition.json",
        "diagnostics/phase2m_fake_quantization.json",
        "diagnostics/phase2m_precision_policy_AM.json",
        "diagnostics/phase2m_precision_policy_repeatability.json",
    ]:
        path = ROOT / rel
        if path.exists():
            provenance["evidence_files"][rel] = {"size": path.stat().st_size, "sha256": sha(path)}
    out = ROOT / "reports/PHASE_2N_PROVENANCE.md"
    body = ["# PHASE 2N PROVENANCE", "", "```json", json.dumps(provenance, indent=2, sort_keys=True), "```", ""]
    out.write_text("\n".join(body))
    print(json.dumps({"candidate_report": str(report), "provenance": str(out), "candidates": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
