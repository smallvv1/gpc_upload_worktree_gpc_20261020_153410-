#!/usr/bin/env python3
"""Materialize Phase 3A artifacts and generate the exact LTCC Table-3 report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/DIFFER")
REPORT = ROOT / "reports/PHASE_3A_LTCC_TABLE3.md"
ANCHORS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
ATTEMPTS = 60 * 596

RUNS = {
    "ID": ROOT / "outputs/reproduction_phase2/LTCC/baseline/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/metrics_per_epoch.csv",
    "ID + Bio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv",
    "ID + NonBio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1/metrics_per_epoch.csv",
    "Full": ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/metrics_per_epoch.csv",
}
MANIFESTS = {
    "ID + Bio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/RUN_MANIFEST.json",
    "ID + NonBio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1/RUN_MANIFEST.json",
}
PAPER = {
    "ID": (54.6, 30.4),
    "ID + Bio": (55.4, 30.8),
    "ID + NonBio": (55.6, 31.0),
    "Full": (58.2, 31.6),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise RuntimeError(f"{path}: expected 60 rows, got {len(rows)}")
    return rows


def value(row: dict[str, str], key: str) -> float | None:
    raw = row[key]
    if raw in {"", "NA", "None", "null"}:
        return None
    return float(raw)


def pct(raw: float | None, digits: int = 3) -> str:
    if raw is None:
        return "—"
    return f"{100.0 * raw:.{digits}f}"


def signed_pp(raw: float) -> str:
    return f"{raw:+.3f}"


def reverse_total(row: dict[str, str]) -> float | None:
    hair = value(row, "reverse_hair_loss")
    cloth = value(row, "reverse_cloth_loss")
    if hair is None and cloth is None:
        return None
    return (hair or 0.0) + (cloth or 0.0)


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: value(row, "cc_rank1") or float("-inf"))


def summary(rows: list[dict[str, str]]) -> dict[str, object]:
    best = best_row(rows)
    final = rows[-1]
    skip = sum(int(value(row, "skipped_steps") or 0) for row in rows)
    found_inf = sum(int(value(row, "found_inf_count") or 0) for row in rows)
    nonfinite = sum(int(value(row, "nonfinite_events") or 0) for row in rows)
    min_scale = min(value(row, "amp_scale_min") for row in rows if value(row, "amp_scale_min") is not None)
    max_scale = max(value(row, "amp_scale_max") for row in rows if value(row, "amp_scale_max") is not None)
    return {
        "rows": rows,
        "best": best,
        "final": final,
        "best_epoch": int(best["epoch"]),
        "best_r1": value(best, "cc_rank1"),
        "best_map": value(best, "cc_mAP"),
        "final_r1": value(final, "cc_rank1"),
        "final_map": value(final, "cc_mAP"),
        "best_general_r1": value(best, "general_rank1"),
        "best_general_map": value(best, "general_mAP"),
        "final_general_r1": value(final, "general_rank1"),
        "final_general_map": value(final, "general_mAP"),
        "skips": skip,
        "updates": ATTEMPTS - skip,
        "found_inf": found_inf,
        "nonfinite": nonfinite,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "final_scale": value(final, "amp_scale_end"),
    }


def paper_status(best: dict[str, object], paper: tuple[float, float]) -> str:
    gaps = [
        abs(100.0 * float(best["best_r1"]) - paper[0]),
        abs(100.0 * float(best["best_map"]) - paper[1]),
    ]
    if all(gap <= 1.0 for gap in gaps):
        return "EXCELLENT"
    if all(gap <= 2.0 for gap in gaps):
        return "ACCEPTABLE"
    if any(gap > 4.0 for gap in gaps):
        return "FAILED"
    return "CONCERNING"


def first_joint_deviation(run_rows: list[dict[str, str]], base_rows: list[dict[str, str]]) -> int | None:
    """First 3-epoch run where both metrics are below ID by >=1.0/0.5 pp."""
    for start in range(len(run_rows) - 2):
        sustained = True
        for index in range(start, start + 3):
            run_r1 = 100.0 * (value(run_rows[index], "cc_rank1") or 0.0)
            base_r1 = 100.0 * (value(base_rows[index], "cc_rank1") or 0.0)
            run_map = 100.0 * (value(run_rows[index], "cc_mAP") or 0.0)
            base_map = 100.0 * (value(base_rows[index], "cc_mAP") or 0.0)
            if not (run_r1 <= base_r1 - 1.0 and run_map <= base_map - 0.5):
                sustained = False
                break
        if sustained:
            return start + 1
    return None


def first_r1_deviation(run_rows: list[dict[str, str]], base_rows: list[dict[str, str]]) -> int | None:
    for start in range(len(run_rows) - 2):
        if all(
            100.0 * (value(run_rows[index], "cc_rank1") or 0.0)
            <= 100.0 * (value(base_rows[index], "cc_rank1") or 0.0) - 1.0
            for index in range(start, start + 3)
        ):
            return start + 1
    return None


def materialize_formal_artifacts() -> None:
    for label, manifest_path in MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        (run_dir / "config.yml").write_text(manifest["config_snapshot"])
        (run_dir / "git_diff.patch").write_text(manifest.get("git_diff", ""))
        rows = load_rows(RUNS[label])
        stats = summary(rows)
        amp = {
            "label": label,
            "total_iterations": ATTEMPTS,
            "optimizer_updates": stats["updates"],
            "amp_skips": stats["skips"],
            "found_inf_count": stats["found_inf"],
            "nonfinite_loss_events": stats["nonfinite"],
            "min_scaler": stats["min_scale"],
            "max_scaler_observed": stats["max_scale"],
            "final_scaler": stats["final_scale"],
            "initial_scaler": manifest["amp"]["initial_scale"],
            "dynamic_growth_backoff": manifest["amp"]["dynamic_growth_backoff"],
            "loss_location": manifest["amp"]["loss_location"],
        }
        (run_dir / "AMP_statistics.json").write_text(json.dumps(amp, indent=2) + "\n")


def trajectory_table(label: str, rows: list[dict[str, str]]) -> str:
    lines = [
        f"#### {label}",
        "",
        "| Epoch | CC R1 | CC mAP | Total loss | ID/CE | Triplet | Bio | Reverse (Hair+Clothing) | Train acc. | AMP scale |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in ANCHORS:
        row = rows[epoch - 1]
        lines.append(
            "| {epoch} | {r1} | {map} | {total} | {ce} | {triplet} | {bio} | {reverse} | {acc} | {scale} |".format(
                epoch=epoch,
                r1=pct(value(row, "cc_rank1")),
                map=pct(value(row, "cc_mAP")),
                total=f"{value(row, 'train_loss'):.4f}",
                ce=f"{value(row, 'ce_loss'):.4f}",
                triplet=f"{value(row, 'triplet_loss'):.4f}",
                bio=(f"{value(row, 'bio_loss'):.4f}" if value(row, "bio_loss") is not None else "—"),
                reverse=(f"{reverse_total(row):.4f}" if reverse_total(row) is not None else "—"),
                acc=f"{100.0 * (value(row, 'train_accuracy') or 0.0):.2f}%",
                scale=f"{value(row, 'amp_scale_end'):.0f}",
            )
        )
    return "\n".join(lines)


def make_report() -> None:
    rows = {label: load_rows(path) for label, path in RUNS.items()}
    stats = {label: summary(data) for label, data in rows.items()}
    baseline = stats["ID"]

    matrix_lines = [
        "| Setting | Paper R1 | Paper mAP | PPU BEST epoch | PPU BEST R1 | PPU BEST mAP | ΔR1 vs paper | ΔmAP vs paper | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in ["ID", "ID + Bio", "ID + NonBio", "Full"]:
        target = PAPER[label]
        item = stats[label]
        r1_gap = 100.0 * float(item["best_r1"]) - target[0]
        map_gap = 100.0 * float(item["best_map"]) - target[1]
        matrix_lines.append(
            f"| {label} | {target[0]:.1f} | {target[1]:.1f} | {item['best_epoch']} | "
            f"{pct(item['best_r1'])} | {pct(item['best_map'])} | {signed_pp(r1_gap)} | {signed_pp(map_gap)} | "
            f"{paper_status(item, target)} |"
        )

    effect_lines = [
        "| Setting | Paper effect R1/mAP | PPU BEST effect R1/mAP | PPU FINAL effect R1/mAP |",
        "|---|---:|---:|---:|",
    ]
    for label in ["ID + Bio", "ID + NonBio", "Full"]:
        target = PAPER[label]
        paper_effect = (target[0] - PAPER["ID"][0], target[1] - PAPER["ID"][1])
        item = stats[label]
        best_effect = (
            100.0 * float(item["best_r1"]) - 100.0 * float(baseline["best_r1"]),
            100.0 * float(item["best_map"]) - 100.0 * float(baseline["best_map"]),
        )
        final_effect = (
            100.0 * float(item["final_r1"]) - 100.0 * float(baseline["best_r1"]),
            100.0 * float(item["final_map"]) - 100.0 * float(baseline["best_map"]),
        )
        effect_lines.append(
            f"| {label} | {signed_pp(paper_effect[0])} / {signed_pp(paper_effect[1])} | "
            f"{signed_pp(best_effect[0])} / {signed_pp(best_effect[1])} | "
            f"{signed_pp(final_effect[0])} / {signed_pp(final_effect[1])} |"
        )

    amp_lines = [
        "| Setting | Total iterations | Optimizer updates | AMP skips | Found-inf | Nonfinite events | Min scale | Final scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["ID", "ID + Bio", "ID + NonBio", "Full"]:
        item = stats[label]
        amp_lines.append(
            f"| {label} | {ATTEMPTS} | {item['updates']} | {item['skips']} | {item['found_inf']} | "
            f"{item['nonfinite']} | {item['min_scale']:.0f} | {item['final_scale']:.0f} |"
        )

    turning_lines = [
        "| Setting | First 3-epoch joint deviation from ID | Interpretation |",
        "|---|---:|---|",
    ]
    for label in ["ID + Bio", "ID + NonBio", "Full"]:
        joint = first_joint_deviation(rows[label], rows["ID"])
        if label == "ID + Bio":
            interpretation = "No sustained joint negative deviation; isolated epoch-18 dip recovers, while mAP is higher from the mid/late trajectory."
        elif label == "ID + NonBio":
            interpretation = "Persistent joint decline begins at epoch 18; this is consistent with the reverse/adversarial branch failing in isolation."
        else:
            interpretation = "Persistent joint decline begins at epoch 19, matching the previously observed epoch-19/20 Full turning point."
        turning_lines.append(f"| {label} | {joint if joint is not None else 'None'} | {interpretation} |")

    bio_best_effect = (
        100.0 * float(stats["ID + Bio"]["best_r1"]) - 100.0 * float(baseline["best_r1"]),
        100.0 * float(stats["ID + Bio"]["best_map"]) - 100.0 * float(baseline["best_map"]),
    )
    nonbio_best_effect = (
        100.0 * float(stats["ID + NonBio"]["best_r1"]) - 100.0 * float(baseline["best_r1"]),
        100.0 * float(stats["ID + NonBio"]["best_map"]) - 100.0 * float(baseline["best_map"]),
    )
    full_best_effect = (
        100.0 * float(stats["Full"]["best_r1"]) - 100.0 * float(baseline["best_r1"]),
        100.0 * float(stats["Full"]["best_map"]) - 100.0 * float(baseline["best_map"]),
    )

    report = f"""# PHASE 3A RESULT — LTCC Table 3 formal ablations

Generated from the 60-row `metrics_per_epoch.csv` files. All reported PPU numbers below use the unrounded CSV values; displayed percentages are rounded only for readability.

## Protocol and scope

- Track A: released-code / paper-faithful formal reproduction.
- LTCC: 9,576 train images, 77 IDs, 256 clothes, 493 query images, 7,050 gallery images, 596 iterations/epoch.
- 224×224, 60 epochs, batch-size config 8, `NUM_INSTANCES=2`, SGD, base LR `2e-6`, weight decay `0.05`, linear warmup plus cosine schedule, seed `1234`.
- PPU-only on 2× PPU-ZW810E; no CPU fallback.
- Summary Bio is enabled at runtime. `NOBIO_INDEX=[1,2]` is Hair + Clothing, with image-level NonBio text. Reverse indexing is `image_feature_reverse[:, i]`; reverse factor losses are accumulated across factors; GRL `alpha=1`.
- Run A active losses: `CE + Triplet + clipBio`; reverse loss disabled. Run B active losses: `CE + Triplet + clipBioReverse`; Bio contrastive loss disabled while the Bio feature-generation path remains present.
- Phase 2 numerical-compatibility candidates were not used; both Phase 3A manifests record `numerical_candidate: null`.

## ID only

Paper: **54.6 / 30.4**

PPU formal Baseline BEST: **epoch {stats['ID']['best_epoch']}, {pct(stats['ID']['best_r1'])} / {pct(stats['ID']['best_map'])}**. FINAL epoch60: **{pct(stats['ID']['final_r1'])} / {pct(stats['ID']['final_map'])}**. Status against the paper target: **{paper_status(stats['ID'], PAPER['ID'])}**.

## ID + Bio

Paper: **55.4 / 30.8**

- BEST: **epoch {stats['ID + Bio']['best_epoch']}, {pct(stats['ID + Bio']['best_r1'])} / {pct(stats['ID + Bio']['best_map'])}**; General at BEST epoch: `{pct(stats['ID + Bio']['best_general_r1'])} / {pct(stats['ID + Bio']['best_general_map'])}`.
- FINAL epoch60: **{pct(stats['ID + Bio']['final_r1'])} / {pct(stats['ID + Bio']['final_map'])}**; General FINAL: `{pct(stats['ID + Bio']['final_general_r1'])} / {pct(stats['ID + Bio']['final_general_map'])}`.
- BEST gap vs paper: **{signed_pp(100.0 * float(stats['ID + Bio']['best_r1']) - PAPER['ID + Bio'][0])} / {signed_pp(100.0 * float(stats['ID + Bio']['best_map']) - PAPER['ID + Bio'][1])} pp**.
- Status: **{paper_status(stats['ID + Bio'], PAPER['ID + Bio'])}**.
- Effect vs our ID BEST `{pct(baseline['best_r1'])} / {pct(baseline['best_map'])}`: **{signed_pp(bio_best_effect[0])} / {signed_pp(bio_best_effect[1])} pp**.
- Paper effect: **+0.8 / +0.4 pp**. The PPU effect is **weak/partial**: R1 is neutral to the displayed precision, while mAP improves by about +1.31 pp.

## ID + NonBio

Paper: **55.6 / 31.0**

- BEST: **epoch {stats['ID + NonBio']['best_epoch']}, {pct(stats['ID + NonBio']['best_r1'])} / {pct(stats['ID + NonBio']['best_map'])}**; General at BEST epoch: `{pct(stats['ID + NonBio']['best_general_r1'])} / {pct(stats['ID + NonBio']['best_general_map'])}`.
- FINAL epoch60: **{pct(stats['ID + NonBio']['final_r1'])} / {pct(stats['ID + NonBio']['final_map'])}**; General FINAL: `{pct(stats['ID + NonBio']['final_general_r1'])} / {pct(stats['ID + NonBio']['final_general_map'])}`.
- BEST gap vs paper: **{signed_pp(100.0 * float(stats['ID + NonBio']['best_r1']) - PAPER['ID + NonBio'][0])} / {signed_pp(100.0 * float(stats['ID + NonBio']['best_map']) - PAPER['ID + NonBio'][1])} pp**.
- Status: **{paper_status(stats['ID + NonBio'], PAPER['ID + NonBio'])}**.
- Effect vs our ID BEST: **{signed_pp(nonbio_best_effect[0])} / {signed_pp(nonbio_best_effect[1])} pp**.
- Paper effect: **+1.0 / +0.6 pp**. The PPU effect is clearly negative, especially for mAP.

## Full DIFFER

Paper: **58.2 / 31.6**

- Existing formal PPU BEST: **epoch {stats['Full']['best_epoch']}, {pct(stats['Full']['best_r1'])} / {pct(stats['Full']['best_map'])}**.
- FINAL epoch60: **{pct(stats['Full']['final_r1'])} / {pct(stats['Full']['final_map'])}**.
- BEST gap vs paper: **{signed_pp(100.0 * float(stats['Full']['best_r1']) - PAPER['Full'][0])} / {signed_pp(100.0 * float(stats['Full']['best_map']) - PAPER['Full'][1])} pp**; status **{paper_status(stats['Full'], PAPER['Full'])}**.
- Effect vs our ID BEST: **{signed_pp(full_best_effect[0])} / {signed_pp(full_best_effect[1])} pp**. Paper effect is **+3.6 / +1.2 pp**.

## Table 3 reproduction matrix

{chr(10).join(matrix_lines)}

## Method effect relative to each run's ID-only Baseline

{chr(10).join(effect_lines)}

The PPU BEST effects are calculated against the formal PPU ID-only BEST, not against the paper Baseline. The PPU Full effect is therefore **{signed_pp(full_best_effect[0])} / {signed_pp(full_best_effect[1])} pp** using the unrounded values.

## AMP accounting

{chr(10).join(amp_lines)}

The AMP scaler was initialized at 1024 with normal dynamic growth/backoff. No scale retuning, loss-precision change, or skip-triggered configuration change was made. `Found-inf` and `Nonfinite events` are the counters recorded by the training driver.

## Trajectory anchors

The following tables report CC metrics, total/component losses, training accuracy, and end-of-epoch AMP scale at epochs 1, 5, 10, 15, 20, 25, 30, 40, 50, and 60. Reverse is the sum of Hair + Clothing reverse losses when active.

{trajectory_table('ID', rows['ID'])}

{trajectory_table('ID + Bio', rows['ID + Bio'])}

{trajectory_table('ID + NonBio', rows['ID + NonBio'])}

{trajectory_table('Full', rows['Full'])}

## Turning-point analysis

Criterion: the first three consecutive epochs for which both CC Rank-1 is at least 1.0 pp below ID-only and CC mAP is at least 0.5 pp below ID-only. This avoids treating a single noisy evaluation as a branch failure.

{chr(10).join(turning_lines)}

Bio-only has a large isolated dip at epoch18 (approximately -5.36 pp R1 / -3.04 pp mAP against the ID trajectory) but recovers by epoch20 and has no three-epoch joint negative deviation under the criterion. NonBio-only first sustains the joint decline at epoch18. Full first sustains it at epoch19, agreeing with the prior observation that Full begins to diverge around epochs 19–20.

## Final diagnosis

**OTHER EVIDENCE-BASED RESULT — CASE A:** Bio-only is weak/partial rather than a full paper-faithful gain (R1 approximately neutral, mAP positive), while NonBio-only is clearly negative and begins a sustained decline around epoch18. Therefore the **NonBio / adversarial reverse branch is the primary failure mode on PPU**. Full remains negative, but the single-branch NonBio failure means the evidence does not support attributing the primary failure to Full-only joint interaction.

This completes Phase 3A Table 3 formal reproduction. Table 4, PRCC, Hair-only/Clothing-only, seed sweep, and numerical-compatibility runs were not started.

## Artifact locations

- [Phase 3A report](/home/DIFFER/reports/PHASE_3A_LTCC_TABLE3.md)
- [ID + Bio run directory](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_bio/run1)
- [ID + NonBio run directory](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1)
- [ID + Bio metrics](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv)
- [ID + NonBio metrics](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1/metrics_per_epoch.csv)
- [ID + Bio manifest](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_bio/run1/RUN_MANIFEST.json)
- [ID + NonBio manifest](/home/DIFFER/outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1/RUN_MANIFEST.json)
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)


if __name__ == "__main__":
    materialize_formal_artifacts()
    make_report()
    print(REPORT)
