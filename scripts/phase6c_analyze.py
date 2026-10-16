#!/usr/bin/env python3
"""Offline analysis for the Phase 6C full-schedule confirmation.

This script does not touch checkpoints or run a model.  It consumes the four
Phase 6C arm artifacts after training and writes the requested window, AMP,
relation-geometry, and historical-reference summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
REPORT = ROOT / "reports/PHASE_6C_FULL_SCHEDULE_RELIABILITY_CONFIRMATION.md"
MANIFEST = OUT_ROOT / "PHASE6C_MANIFEST.json"
ARMS = {
    "B0": OUT_ROOT / "B0_full",
    "B1": OUT_ROOT / "B1_full",
    "B2": OUT_ROOT / "B2_full",
    "B3": OUT_ROOT / "B3_full",
}
LABELS = {
    "B0": "Bio control",
    "B1": "Uniform cross-clothes relation",
    "B2": "SR3 soft reliability relation",
    "B3": "Shuffled reliability control",
}
HISTORICAL_BIO = {"rank1": 56.3776, "mAP": 32.3857}
WINDOWS = {
    "E1-E60": (1, 60),
    "E21-E30": (21, 30),
    "E31-E40": (31, 40),
    "E41-E50": (41, 50),
    "E51-E60": (51, 60),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def window_summary(rows: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"Expected {end-start+1} rows for E{start}-E{end}, got {len(selected)}")
    epochs = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    result: dict[str, Any] = {"start_epoch": start, "end_epoch": end, "epochs": len(selected)}
    for name, key in (("rank1", "cc_rank1"), ("mAP", "cc_mAP")):
        values = np.asarray([as_float(row, key) for row in selected], dtype=np.float64) * 100.0
        result[f"mean_{name}"] = float(values.mean())
        result[f"auc_{name}"] = float(np.trapezoid(values, epochs) / (epochs[-1] - epochs[0]))
        result[f"slope_{name}"] = float(np.polyfit(epochs, values, 1)[0])
        result[f"first_{name}"] = float(values[0])
        result[f"last_{name}"] = float(values[-1])
    result["best_rank1"] = float(max(as_float(row, "cc_rank1") for row in selected) * 100.0)
    result["best_mAP"] = float(max(as_float(row, "cc_mAP") for row in selected) * 100.0)
    result["best_rank1_epoch"] = int(max(selected, key=lambda row: as_float(row, "cc_rank1"))["epoch"])
    result["best_mAP_epoch"] = int(max(selected, key=lambda row: as_float(row, "cc_mAP"))["epoch"])
    return result


def load_all_metrics() -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for arm, directory in ARMS.items():
        path = directory / "metrics_per_epoch.csv"
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, 61)):
            raise RuntimeError(f"{arm} is not complete E1-E60: {epochs[:3]} ... {epochs[-3:]}")
        output[arm] = rows
    return output


def make_window_tables(metrics: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for arm, rows in metrics.items():
        for window, (start, end) in WINDOWS.items():
            summaries.append({"arm": arm, "setting": LABELS[arm], "window": window, **window_summary(rows, start, end)})
    write_csv(OUT_ROOT / "window_comparison.csv", summaries)

    comparisons: list[dict[str, Any]] = []
    by_key = {(row["arm"], row["window"]): row for row in summaries}
    for window in WINDOWS:
        for left, right, comparison in (("B1", "B0", "B1_minus_B0"), ("B2", "B1", "B2_minus_B1"),
                                        ("B2", "B3", "B2_minus_B3"), ("B2", "B0", "B2_minus_B0")):
            lhs, rhs = by_key[(left, window)], by_key[(right, window)]
            comparisons.append({
                "window": window,
                "comparison": comparison,
                "left": left,
                "right": right,
                "auc_rank1_delta_pp": lhs["auc_rank1"] - rhs["auc_rank1"],
                "auc_mAP_delta_pp": lhs["auc_mAP"] - rhs["auc_mAP"],
                "mean_rank1_delta_pp": lhs["mean_rank1"] - rhs["mean_rank1"],
                "mean_mAP_delta_pp": lhs["mean_mAP"] - rhs["mean_mAP"],
                "slope_rank1_delta_pp": lhs["slope_rank1"] - rhs["slope_rank1"],
                "slope_mAP_delta_pp": lhs["slope_mAP"] - rhs["slope_mAP"],
                "last_rank1_delta_pp": lhs["last_rank1"] - rhs["last_rank1"],
                "last_mAP_delta_pp": lhs["last_mAP"] - rhs["last_mAP"],
                "best_rank1_delta_pp": lhs["best_rank1"] - rhs["best_rank1"],
                "best_mAP_delta_pp": lhs["best_mAP"] - rhs["best_mAP"],
            })
    write_csv(OUT_ROOT / "late_window_analysis.csv", [row for row in comparisons if row["window"] == "E51-E60"])
    return summaries, comparisons


def amp_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, directory in ARMS.items():
        values = read_csv(directory / "amp_statistics.csv")
        if len(values) != 60:
            raise RuntimeError(f"{arm} AMP rows: expected 60, found {len(values)}")
        skips = [int(float(row["skipped_steps"])) for row in values]
        nonfinite = [int(float(row["nonfinite_events"])) for row in values]
        rows.append({
            "arm": arm,
            "setting": LABELS[arm],
            "rows": len(values),
            "total_skipped_steps": sum(skips),
            "prefix_e1_e30_skipped_steps": sum(skips[:30]),
            "continuation_e31_e60_skipped_steps": sum(skips[30:]),
            "total_found_inf_count": sum(int(float(row["found_inf_count"])) for row in values),
            "continuation_e31_e60_found_inf_count": sum(int(float(row["found_inf_count"])) for row in values[30:]),
            "skip_epochs": sum(value > 0 for value in skips),
            "skip_epoch_list": ",".join(str(index + 1) for index, value in enumerate(skips) if value > 0),
            "continuation_skip_epoch_list": ",".join(str(index + 31) for index, value in enumerate(skips[30:]) if value > 0),
            "max_skipped_steps_epoch": max(skips),
            "total_nonfinite_events": sum(nonfinite),
            "amp_scale_initial": float(values[0]["amp_scale_end"]),
            "amp_scale_at_e30": float(values[29]["amp_scale_end"]),
            "amp_scale_final": float(values[-1]["amp_scale_end"]),
            "amp_scale_min": min(float(row["amp_scale_min"]) for row in values),
            "amp_scale_max": max(float(row["amp_scale_max"]) for row in values),
        })
        event_path = directory / "amp_skip_events.csv"
        event_rows = read_csv(event_path) if event_path.is_file() else []
        rows[-1]["continuation_skip_event_count"] = len(event_rows)
        rows[-1]["continuation_skip_event_epoch_list"] = ",".join(f"E{row['epoch']}" for row in event_rows)
        rows[-1]["continuation_skip_event_iteration_list"] = ",".join(
            f"E{row['epoch']}:I{row['iteration']}:R{row['rank']}" for row in event_rows
        )
    write_csv(OUT_ROOT / "amp_full_schedule_audit.csv", rows)
    return rows


def geometry_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, directory in ARMS.items():
        for row in read_csv(directory / "relation_geometry.csv"):
            rows.append({"arm": arm, "setting": LABELS[arm], **row})
    write_csv(OUT_ROOT / "relation_geometry_e30_e40_e50_e60.csv", rows)
    return rows


def historical_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in summaries if row["window"] in ("E51-E60", "E1-E60")]
    output: list[dict[str, Any]] = []
    for row in selected:
        output.append({
            "arm": row["arm"],
            "setting": row["setting"],
            "window": row["window"],
            "auc_rank1": row["auc_rank1"],
            "auc_mAP": row["auc_mAP"],
            "delta_vs_historical_bio_rank1_pp": row["auc_rank1"] - HISTORICAL_BIO["rank1"],
            "delta_vs_historical_bio_mAP_pp": row["auc_mAP"] - HISTORICAL_BIO["mAP"],
            "historical_bio_rank1": HISTORICAL_BIO["rank1"],
            "historical_bio_mAP": HISTORICAL_BIO["mAP"],
            "reference_type": "secondary_historical_reference_not_matched_control",
        })
    write_csv(OUT_ROOT / "historical_bio_comparison.csv", output)
    return output


def classify_direction(comparisons: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    late = [row for row in comparisons if row["window"] == "E51-E60" and row["comparison"] == f"{left}_minus_{right}"][0]
    mid = [row for row in comparisons if row["window"] == "E21-E30" and row["comparison"] == f"{left}_minus_{right}"][0]
    return {"late": late, "screen": mid}


def build_analysis(metrics: dict[str, list[dict[str, str]]], summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], amp: list[dict[str, Any]], geometry: list[dict[str, Any]]) -> dict[str, Any]:
    b1_b0 = classify_direction(comparisons, "B1", "B0")
    b2_b0 = classify_direction(comparisons, "B2", "B0")
    b2_b1 = classify_direction(comparisons, "B2", "B1")
    b2_b3 = classify_direction(comparisons, "B2", "B3")
    late_b2_b1 = b2_b1["late"]
    amp_by_arm = {row["arm"]: row for row in amp}
    b3_amp_confound = amp_by_arm["B3"]["total_skipped_steps"] > max(amp_by_arm[arm]["total_skipped_steps"] for arm in ("B0", "B1", "B2"))
    b1_late_positive = b1_b0["late"]["auc_rank1_delta_pp"] > 0 and b1_b0["late"]["auc_mAP_delta_pp"] > 0
    b2_late_positive = b2_b0["late"]["auc_rank1_delta_pp"] > 0 and b2_b0["late"]["auc_mAP_delta_pp"] > 0
    b2_vs_b1_positive = late_b2_b1["auc_rank1_delta_pp"] > 0 and late_b2_b1["auc_mAP_delta_pp"] > 0
    b2_vs_b3_positive = b2_b3["late"]["auc_rank1_delta_pp"] > 0 and b2_b3["late"]["auc_mAP_delta_pp"] > 0
    early_b2_vs_b0_positive = b2_b0["screen"]["auc_rank1_delta_pp"] > 0 and b2_b0["screen"]["auc_mAP_delta_pp"] > 0
    if early_b2_vs_b0_positive and b2_late_positive:
        gain_class = "BOTH"
    elif early_b2_vs_b0_positive:
        gain_class = "EARLY ACCELERATION ONLY"
    elif b2_late_positive:
        gain_class = "FINAL PERFORMANCE IMPROVEMENT"
    else:
        gain_class = "NO_CONFIRMED_B2_VS_B0_GAIN"
    questions = {
        "Q1_B1_gt_B0_late_training": "YES" if b1_late_positive else "NO",
        "Q2_B2_gt_B0_late_training": "YES" if b2_late_positive else "NO",
        "Q3_B2_gt_B1_late_training": "YES" if b2_vs_b1_positive else "NO",
        "Q4_B2_gt_B3_late_training": "YES_BUT_AMP_CONFOUND" if b2_vs_b3_positive and b3_amp_confound else ("YES" if b2_vs_b3_positive else "NO"),
        "Q5_phase6b_gain_class": gain_class,
        "Q6_B3_AMP_skips_higher_than_other_arms": "YES" if b3_amp_confound else "NO",
    }
    second_seed_action = "SECOND-SEED REPLICATION RECOMMENDED" if b2_vs_b1_positive else "NO_SECOND_SEED_RECOMMENDATION"
    promotion_decision = (
        "HOLD — SECOND-SEED REPLICATION RECOMMENDED; DO NOT ENTER PHASE 6D"
        if b2_vs_b1_positive
        else "STOP AFTER PHASE 6C; DO NOT ENTER PHASE 6D"
    )
    analysis = {
        "phase": "phase6c_full_schedule_confirmation",
        "training_mode": "RESUME_MODE",
        "arms": list(ARMS),
        "windows": WINDOWS,
        "primary_window": "E51-E60",
        "historical_bio_secondary_reference": HISTORICAL_BIO,
        "comparisons": {"B1_minus_B0": b1_b0, "B2_minus_B0": b2_b0, "B2_minus_B1": b2_b1, "B2_minus_B3": b2_b3},
        "amp": amp,
        "amp_b3_numerical_confound": b3_amp_confound,
        "relation_geometry_rows": len(geometry),
        "diagnostic_questions": questions,
        "diagnostic_questions_initial": questions,
        "second_seed_action": second_seed_action,
        "final_promotion_decision": promotion_decision,
        "second_seed_started": False,
        "phase6d_started": False,
        "note": "B2-B1 is reliability-specific evidence; B2-B0 alone does not identify the SR3 contribution.",
    }
    write_json(OUT_ROOT / "seed1234_analysis.json", analysis)
    return analysis


def markdown_report(summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], amp: list[dict[str, Any]], geometry: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    def f(value: Any) -> str:
        return f"{float(value):.4f}"

    def pair(row: dict[str, Any], field_a: str, field_b: str) -> str:
        return f"{f(row[field_a])} / {f(row[field_b])}"

    lines = [
        "# Phase 6C — Full-Schedule Reliability Confirmation",
        "",
        "## Scope",
        "",
        "This report analyzes the frozen Phase 6B arms after the full 60-epoch schedule. No method component, relation score, sampler, loss coefficient, or evaluation split was changed.",
        "",
        "- Mode: RESUME_MODE from the exact Phase 6B E30 rank states.",
        "- Primary window: E51–E60.",
        "- Train-only relation geometry: E30/E40/E50/E60.",
        "- Historical Bio is secondary; matched B0 is the primary control.",
        "- No second seed was started and no Phase 6D component was entered.",
        "",
        "## Window trajectory results",
        "",
        "| Arm | Window | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Slope R1 | Slope mAP | End R1 | End mAP | BEST R1 (epoch) | BEST mAP (epoch) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["window"] in ("E21-E30", "E31-E40", "E41-E50", "E51-E60"):
            lines.append(
                f"| {row['arm']} | {row['window']} | {f(row['mean_rank1'])} | {f(row['mean_mAP'])} | "
                f"{f(row['auc_rank1'])} | {f(row['auc_mAP'])} | {f(row['slope_rank1'])} | {f(row['slope_mAP'])} | "
                f"{f(row['last_rank1'])} | {f(row['last_mAP'])} | {f(row['best_rank1'])} (E{row['best_rank1_epoch']}) | "
                f"{f(row['best_mAP'])} (E{row['best_mAP_epoch']}) |"
            )
    lines += [
        "",
        "## Primary late-window comparisons (E51–E60)",
        "",
        "| Comparison | Mean R1/mAP Δ pp | AUC R1/mAP Δ pp | Slope R1/mAP Δ pp | E60 R1/mAP Δ pp | BEST R1/mAP Δ pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["window"] == "E51-E60":
            lines.append(
                f"| {row['comparison']} | {pair(row, 'mean_rank1_delta_pp', 'mean_mAP_delta_pp')} | "
                f"{pair(row, 'auc_rank1_delta_pp', 'auc_mAP_delta_pp')} | "
                f"{pair(row, 'slope_rank1_delta_pp', 'slope_mAP_delta_pp')} | "
                f"{pair(row, 'last_rank1_delta_pp', 'last_mAP_delta_pp')} | "
                f"{pair(row, 'best_rank1_delta_pp', 'best_mAP_delta_pp')} |"
            )
    lines += [
        "",
        "## AMP audit",
        "",
        "| Arm | Total skips | Total found_inf | E31–E60 skips | E31–E60 skip epochs | E31–E60 skip iterations | Scale E30 | Final scale |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in amp:
        lines.append(
            f"| {row['arm']} | {row['total_skipped_steps']} | {row['total_found_inf_count']} | "
            f"{row['continuation_e31_e60_skipped_steps']} | {row['continuation_skip_epoch_list'] or 'none'} | "
            f"{row['continuation_skip_event_iteration_list'] or 'not separately recorded'} | "
            f"{row['amp_scale_at_e30']} | {row['amp_scale_final']} |"
        )
    lines += [
        "",
        "## Relation geometry",
        "",
        "Geometry is computed only on LTCC training images and the frozen Phase6A3 candidate relation bank. It is diagnostic and does not use query/gallery data to construct reliability.",
        "",
        "See `relation_geometry_e30_e40_e50_e60.csv` for ALL/Q1–Q4 rows per arm and checkpoint.",
        "",
        "## Frozen interpretation",
        "",
        "- B1 vs B0 tests relation supervision itself.",
        "- B2 vs B1 tests the SR3-specific reliability structure.",
        "- B2 vs B3 is interpreted with the AMP skip confound shown above.",
        "- A positive B2 vs B0 result is not sufficient to attribute all gain to SR3.",
        "",
        "## Q1–Q6",
        "",
    ]
    late = analysis["comparisons"]
    lines += [
        f"- **Q1. B1 > B0 是否保持到 late training？** **{analysis['diagnostic_questions']['Q1_B1_gt_B0_late_training']}** — late AUC Δ R1/mAP: {f(late['B1_minus_B0']['late']['auc_rank1_delta_pp'])} / {f(late['B1_minus_B0']['late']['auc_mAP_delta_pp'])} pp.",
        f"- **Q2. B2 > B0 是否保持到 late training？** **{analysis['diagnostic_questions']['Q2_B2_gt_B0_late_training']}** — late AUC Δ R1/mAP: {f(late['B2_minus_B0']['late']['auc_rank1_delta_pp'])} / {f(late['B2_minus_B0']['late']['auc_mAP_delta_pp'])} pp.",
        f"- **Q3. B2 > B1 是否仍然存在？** **{analysis['diagnostic_questions']['Q3_B2_gt_B1_late_training']}** — late AUC Δ R1/mAP: {f(late['B2_minus_B1']['late']['auc_rank1_delta_pp'])} / {f(late['B2_minus_B1']['late']['auc_mAP_delta_pp'])} pp.",
        f"- **Q4. B2 > B3 是否仍然存在？** **{analysis['diagnostic_questions']['Q4_B2_gt_B3_late_training']}** — late AUC Δ R1/mAP: {f(late['B2_minus_B3']['late']['auc_rank1_delta_pp'])} / {f(late['B2_minus_B3']['late']['auc_mAP_delta_pp'])} pp.",
        f"- **Q5. Phase6B 结果分类：** **{analysis['diagnostic_questions']['Q5_phase6b_gain_class']}**.",
        f"- **Q6. B3 AMP skip 是否仍明显高于其他 arms？** **{analysis['diagnostic_questions']['Q6_B3_AMP_skips_higher_than_other_arms']}**.",
        "",
        "## Final decision",
        "",
        f"**{analysis['final_promotion_decision']}**",
        "",
        f"**{analysis['second_seed_action']}** — do not start seed4321 automatically.",
        "",
        "Phase 6D remains stopped: no Relation DS, curriculum, H_ID curriculum, DW, GRL, or Reverse was started.",
    ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- `window_comparison.csv`",
        "- `late_window_analysis.csv`",
        "- `relation_geometry_e30_e40_e50_e60.csv`",
        "- `amp_full_schedule_audit.csv`",
        "- `historical_bio_comparison.csv`",
        "- `seed1234_analysis.json`",
        "- `amp_skip_events.csv` (continuation skip-iteration audit where available)",
        "- `artifacts/B3_resume_e30_audit.md`",
        "",
        "This is the seed1234 Phase 6C report. Any second-seed replication requires manual confirmation.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_phase6c_manifest(summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], amp: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    arms: dict[str, Any] = {}
    for arm, directory in ARMS.items():
        metrics_path = directory / "metrics_per_epoch.csv"
        amp_path = directory / "amp_statistics.csv"
        rows = read_csv(metrics_path)
        arms[arm] = {
            "label": LABELS[arm],
            "epochs": len(rows),
            "epoch_first": int(rows[0]["epoch"]) if rows else None,
            "epoch_last": int(rows[-1]["epoch"]) if rows else None,
            "run_complete": (directory / "RUN_COMPLETE.json").is_file(),
            "metrics_sha256": sha256_file(metrics_path),
            "amp_statistics_sha256": sha256_file(amp_path),
            "run_complete_sha256": sha256_file(directory / "RUN_COMPLETE.json") if (directory / "RUN_COMPLETE.json").is_file() else None,
        }
    manifest = {
        "phase": "phase6c_full_schedule_confirmation",
        "seed": 1234,
        "mode": "RESUME_MODE",
        "primary_window": "E51-E60",
        "windows": WINDOWS,
        "arms": arms,
        "analysis": str(OUT_ROOT / "seed1234_analysis.json"),
        "report": str(REPORT),
        "b3_resume_audit": str(OUT_ROOT / "artifacts/B3_resume_e30_audit.md"),
        "window_comparison": str(OUT_ROOT / "window_comparison.csv"),
        "late_window_analysis": str(OUT_ROOT / "late_window_analysis.csv"),
        "amp_audit": str(OUT_ROOT / "amp_full_schedule_audit.csv"),
        "summaries": summaries,
        "comparisons": comparisons,
        "amp": amp,
        "diagnostic_questions": analysis["diagnostic_questions"],
        "second_seed_started": False,
        "phase6d_started": False,
        "final_promotion_decision": analysis["final_promotion_decision"],
        "second_seed_action": analysis["second_seed_action"],
    }
    write_json(MANIFEST, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    metrics = load_all_metrics() if args.require_complete else load_all_metrics()
    summaries, comparisons = make_window_tables(metrics)
    amp = amp_audit()
    geometry = geometry_table()
    historical_table(summaries)
    analysis = build_analysis(metrics, summaries, comparisons, amp, geometry)
    markdown_report(summaries, comparisons, amp, geometry, analysis)
    update_phase6c_manifest(summaries, comparisons, amp, analysis)
    print(json.dumps({"report": str(REPORT), "analysis": str(OUT_ROOT / "seed1234_analysis.json"), "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
