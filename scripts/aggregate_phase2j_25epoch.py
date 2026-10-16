#!/usr/local/bin/python3
"""Aggregate the bounded PHASE 2J-G three-repeat 25-epoch probe.

This is analysis-only.  It reads the already produced probe JSON files and
the frozen official LTCC curves; it never launches training or changes model
code.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "diagnostics/phase2j_25epoch_aggregate.json"
OUT_REPORT = ROOT / "reports/PHASE_2J_G_25EPOCH.md"

RUN_LABELS = ("P-A", "P-B", "P-C")
RUN_FILES = {
    label: ROOT / f"diagnostics/phase2j_{label.lower().replace('-', '_')}_25epoch.json"
    for label in RUN_LABELS
}

FIELDS = (
    "train_loss", "id_loss", "triplet_loss", "bio_loss", "reverse0_loss", "reverse1_loss",
    "id_acc", "bio_acc", "reverse0_acc", "reverse1_acc", "cc_rank1", "cc_mAP",
)

OFFICIAL_FIELD_MAP = {
    "train_loss": "total_loss",
    "id_loss": "ce_loss",
    "triplet_loss": "triplet_loss",
    "bio_loss": "bio_loss",
    "reverse0_loss": "reverse_0_loss_cumulative",
    "reverse1_loss": "reverse_1_loss_cumulative",
    "id_acc": "train_accuracy",
    "bio_acc": "bio_accuracy",
    "reverse0_acc": "reverse_0_accuracy",
    "reverse1_acc": "reverse_1_accuracy",
}


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return 1.0 if np.allclose(xa, ya) else 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def metric_stats(x: list[float], y: list[float]) -> dict[str, float | None]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    d = xa - ya
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    scale = float(np.std(ya))
    if scale < 1e-12:
        scale = max(float(np.ptp(ya)), 1.0)
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson(x, y),
        "normalized_rmse": float(rmse / scale),
    }


def load_official() -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    train: dict[int, dict[str, float]] = {}
    with (ROOT / "reports/OFFICIAL_LTCC_TRAIN_CURVE.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"])
            train[epoch] = {k: float(v) for k, v in row.items() if k not in ("epoch", "event_file") and v != ""}
    metrics: dict[int, dict[str, float]] = {}
    with (ROOT / "reports/OFFICIAL_LTCC_METRICS_PER_EPOCH.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"])
            metrics[epoch] = {k: float(v) for k, v in row.items() if k != "epoch" and v != ""}
    return train, metrics


def candidate_record(record: dict[str, Any]) -> dict[str, float | None]:
    return {
        "train_loss": finite_float(record.get("train_loss")),
        "id_loss": finite_float(record.get("id_loss")),
        "triplet_loss": finite_float(record.get("triplet_loss")),
        "bio_loss": finite_float(record.get("bio_loss")),
        "reverse0_loss": finite_float(record.get("reverse0_loss")),
        "reverse1_loss": finite_float(record.get("reverse1_loss")),
        "id_acc": finite_float(record.get("id_acc")),
        "bio_acc": finite_float(record.get("bio_acc")),
        "reverse0_acc": finite_float(record.get("reverse0_acc")),
        "reverse1_acc": finite_float(record.get("reverse1_acc")),
        "cc_rank1": finite_float((record.get("cc") or {}).get("rank1")),
        "cc_mAP": finite_float((record.get("cc") or {}).get("mAP")),
        "general_rank1": finite_float((record.get("general") or {}).get("rank1")),
        "general_mAP": finite_float((record.get("general") or {}).get("mAP")),
    }


def main() -> None:
    official_train, official_metrics = load_official()
    runs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for label, path in RUN_FILES.items():
        if not path.exists():
            missing.append(label)
            continue
        payload = json.loads(path.read_text())
        records = [candidate_record(r) for r in payload["records"]]
        runs[label] = {
            "path": str(path),
            "payload": payload,
            "records": records,
        }
    if missing:
        raise SystemExit("missing completed probe JSON: " + ", ".join(missing))

    official_series: dict[str, list[float]] = {}
    candidate_series: dict[str, dict[str, list[float]]] = {label: {} for label in RUN_LABELS}
    for field in FIELDS:
        values: list[float] = []
        for epoch in range(1, 26):
            if field in OFFICIAL_FIELD_MAP:
                value = official_train.get(epoch, {}).get(OFFICIAL_FIELD_MAP[field])
            elif field == "cc_rank1":
                value = official_metrics.get(epoch, {}).get("cc_rank1")
            elif field == "cc_mAP":
                value = official_metrics.get(epoch, {}).get("cc_mAP")
            else:
                value = None
            values.append(float(value) if value is not None else float("nan"))
        official_series[field] = values
        for label in RUN_LABELS:
            candidate_series[label][field] = [float(r[field]) if r[field] is not None else float("nan") for r in runs[label]["records"]]

    vs_official: dict[str, dict[str, dict[str, float | None]]] = {label: {} for label in RUN_LABELS}
    composite: dict[str, float] = {}
    for label in RUN_LABELS:
        normalized: list[float] = []
        for field in FIELDS:
            x = candidate_series[label][field]
            y = official_series[field]
            keep = np.isfinite(x) & np.isfinite(y)
            stats = metric_stats(list(np.asarray(x)[keep]), list(np.asarray(y)[keep])) if keep.any() else {"rmse": None, "mae": None, "pearson_r": None, "normalized_rmse": None}
            vs_official[label][field] = stats
            if stats["normalized_rmse"] is not None:
                normalized.append(float(stats["normalized_rmse"]))
        composite[label] = float(np.mean(normalized)) if normalized else float("nan")

    pairwise: dict[str, dict[str, dict[str, float | None]]] = {}
    for i, left in enumerate(RUN_LABELS):
        for right in RUN_LABELS[i + 1:]:
            key = f"{left}_vs_{right}"
            pairwise[key] = {}
            for field in FIELDS:
                x = candidate_series[left][field]; y = candidate_series[right][field]
                keep = np.isfinite(x) & np.isfinite(y)
                pairwise[key][field] = metric_stats(list(np.asarray(x)[keep]), list(np.asarray(y)[keep])) if keep.any() else {}

    envelope: dict[str, Any] = {}
    for field in ("cc_rank1", "cc_mAP", "reverse1_acc", "bio_acc", "id_acc", "train_loss"):
        rows = []
        inside = 0
        for idx, epoch in enumerate(range(1, 26)):
            vals = np.asarray([candidate_series[label][field][idx] for label in RUN_LABELS], dtype=np.float64)
            official = official_series[field][idx]
            lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals)); mean = float(np.nanmean(vals)); std = float(np.nanstd(vals))
            in_env = bool(np.isfinite(official) and lo <= official <= hi)
            inside += int(in_env)
            rows.append({"epoch": epoch, "mean": mean, "std": std, "min": lo, "max": hi, "official": official, "official_inside": in_env})
        envelope[field] = {"official_inside_count": inside, "rows": rows}

    initial_hashes = {label: runs[label]["payload"].get("initial_head_hashes") for label in RUN_LABELS}
    fingerprints = {label: runs[label]["payload"].get("fingerprints_first_two_per_epoch") for label in RUN_LABELS}
    same_initial = all(initial_hashes[label] == initial_hashes["P-A"] for label in RUN_LABELS)
    same_fingerprint = all(fingerprints[label] == fingerprints["P-A"] for label in RUN_LABELS)
    summary = {
        "phase": "2J-G",
        "epochs": 25,
        "official_curve_epochs": sorted(official_train),
        "runs": {label: {
            "path": runs[label]["path"],
            "total_skips": runs[label]["payload"].get("total_skips"),
            "successful_optimizer_updates": runs[label]["payload"].get("successful_optimizer_updates"),
            "final_scale": runs[label]["payload"].get("final_scale"),
            "initial_head_hashes": initial_hashes[label],
            "final_head_hashes": runs[label]["payload"].get("final_head_hashes"),
        } for label in RUN_LABELS},
        "same_initial_head_hashes": same_initial,
        "same_first_fingerprints": same_fingerprint,
        "official_vs_candidate": vs_official,
        "composite_early_fingerprint_score_lower_is_better": composite,
        "candidate_ranking": sorted(RUN_LABELS, key=lambda label: composite[label]),
        "pairwise": pairwise,
        "variance_envelope": envelope,
        "notes": [
            "Official reverse_1 is compared using the released cumulative factor-0+factor-1 logging semantics.",
            "Composite score is the mean normalized RMSE across the 12 common branch/retrieval curves; lower is closer.",
            "This report does not infer causal equivalence from a single metric or one epoch.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines: list[str] = [
        "# PHASE 2J-G 25-EPOCH SAME-SEED REPEATABILITY",
        "",
        "Three fresh natural-construction biasless Full DIFFER runs were compared against the official LTCC curve for epochs 1–25.",
        "",
        f"Initial head hashes identical: **{same_initial}**",
        f"First-two-batch fingerprints identical: **{same_fingerprint}**",
        "",
        "## Run stability",
        "",
        "| Run | Skipped steps | Successful updates | Final scale |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in RUN_LABELS:
        p = runs[label]["payload"]
        lines.append(f"| {label} | {p.get('total_skips')} | {p.get('successful_optimizer_updates')} | {p.get('final_scale')} |")
    lines += ["", "## Candidate similarity to Official (lower normalized RMSE is closer)", "", "| Candidate | Composite score | Rank |", "| --- | ---: | ---: |"]
    ranking = summary["candidate_ranking"]
    for label in RUN_LABELS:
        lines.append(f"| {label} | {composite[label]:.9f} | {ranking.index(label) + 1} |")
    lines += ["", "| Curve | A RMSE | B RMSE | C RMSE | A r | B r | C r |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for field in FIELDS:
        vals = [vs_official[label][field] for label in RUN_LABELS]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            field,
            *[f"{v['rmse']:.8f}" if v.get("rmse") is not None else "NA" for v in vals],
            *[f"{v['pearson_r']:.6f}" if v.get("pearson_r") is not None else "NA" for v in vals],
        ))
    lines += ["", "## Epoch-wise variance envelope", "", "Official-in-envelope count over epochs 19–25:"]
    for field in ("cc_rank1", "cc_mAP", "reverse1_acc", "bio_acc", "id_acc", "train_loss"):
        rows = envelope[field]["rows"]
        tail = [r for r in rows if r["epoch"] >= 19]
        count = sum(int(r["official_inside"]) for r in tail)
        lines.append(f"- `{field}`: {count}/7")
    lines += ["", "| Epoch | Official CC R1 | P-A mean±std | P-B mean±std | P-C mean±std | Official in envelope |", "| ---: | ---: | ---: | ---: | ---: | :---: |"]
    for row_idx in range(18, 25):
        rows = [envelope["cc_rank1"]["rows"][row_idx],]
        epoch = rows[0]["epoch"]
        parts = [str(epoch), f"{rows[0]['official']:.6f}"]
        for label in RUN_LABELS:
            v = candidate_series[label]["cc_rank1"][row_idx]
            # std is run-level; use the envelope row once for the shared aggregate below.
            parts.append("see aggregate")
        er = envelope["cc_rank1"]["rows"][row_idx]
        lines.append(f"| {epoch} | {er['official']:.6f} | {er['mean']:.6f} ± {er['std']:.6f} | {er['min']:.6f}..{er['max']:.6f} | {er['official_inside']} |")
    lines += ["", "## Interpretation gate", "", "The three runs share the same natural initial head hashes and first batch fingerprints, while their 25-epoch optimizer trajectories, skipped-step histories, and curve distances are reported above. This bounded probe must be interpreted together with the earlier same-state backward nondeterminism and FP16/fused-attention sensitivity diagnostics; it does not authorize a 60-epoch run.", ""]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines))
    print(json.dumps({"output_json": str(OUT_JSON), "output_report": str(OUT_REPORT), "ranking": ranking, "composite": composite}, sort_keys=True))


if __name__ == "__main__":
    main()
