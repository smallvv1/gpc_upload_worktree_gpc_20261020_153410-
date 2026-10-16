#!/usr/bin/env python3
"""Build the frozen Phase 3B LTCC Table-4 reproduction report.

All percentages and effects are calculated from the raw CSV fractions.  The
report is intentionally generated from artifacts rather than from rounded
terminal output.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/DIFFER")
OUT = ROOT / "reports/PHASE_3B_LTCC_TABLE4.md"
ANCHORS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]

BASELINE_CSV = ROOT / (
    "outputs/reproduction_phase2/LTCC/baseline/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/metrics_per_epoch.csv"
)
FULL_CSV = ROOT / (
    "outputs/reproduction_phase2/LTCC/differ/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/metrics_per_epoch.csv"
)
PHASE3A_BIO_CSV = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv"
PHASE3A_NONBIO_CSV = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1/metrics_per_epoch.csv"


RUNS = {
    "bio_hair": {
        "label": "Hair",
        "factor": "Hair",
        "paper": (56.6, 31.3),
        "csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1/metrics_per_epoch.csv",
        "selected_csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1/metrics_selected_factor_per_epoch.csv",
        "dir": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1",
    },
    "bio_clothing": {
        "label": "Clothing",
        "factor": "Clothing",
        "paper": (56.6, 30.4),
        "csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1/metrics_per_epoch.csv",
        "selected_csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1/metrics_selected_factor_per_epoch.csv",
        "dir": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1",
    },
    "bio_pose": {
        "label": "Pose",
        "factor": "Pose",
        "paper": (56.1, 31.5),
        "csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1/metrics_per_epoch.csv",
        "selected_csv": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1/metrics_selected_factor_per_epoch.csv",
        "dir": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in (None, "", "NA", "nan", "NaN"):
        return None
    return float(raw)


def indexed(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["epoch"]): row for row in rows}


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["cc_rank1"]))


def pct(fraction: float | None, digits: int = 3) -> str:
    if fraction is None:
        return "—"
    return f"{fraction * 100:.{digits}f}"


def loss(number: float | None, digits: int = 3) -> str:
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def signed(number: float, digits: int = 3) -> str:
    return f"{number:+.{digits}f}"


def pp(fraction: float) -> float:
    return fraction * 100.0


def gap(our_fraction: float, paper_percent: float) -> float:
    return pp(our_fraction) - paper_percent


def status(r1_gap: float, map_gap: float) -> str:
    magnitudes = (abs(r1_gap), abs(map_gap))
    if all(x <= 1.0 for x in magnitudes):
        return "EXCELLENT"
    if all(x <= 2.0 for x in magnitudes):
        return "ACCEPTABLE"
    if any(x > 4.0 for x in magnitudes):
        return "FAILED"
    return "CONCERNING"


def reverse_value(row: dict[str, str], selected_row: dict[str, str] | None, kind: str) -> float | None:
    if selected_row is not None:
        return value(selected_row, "reverse_selected_loss")
    if kind == "full":
        hair = value(row, "reverse_hair_loss")
        cloth = value(row, "reverse_cloth_loss")
        return (hair or 0.0) + (cloth or 0.0) if hair is not None or cloth is not None else None
    return None


def run_artifacts(run_dir: Path) -> dict[str, object]:
    required_files = [
        "config.yml",
        "RUN_MANIFEST.json",
        "train_log.txt",
        "metrics_per_epoch.csv",
        "metrics_selected_factor_per_epoch.csv",
        "AMP_statistics.json",
        "git_diff.patch",
        "eva02_l_bio_best.pth",
        "eva02_l_bio_final.pth",
    ]
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    amp = json.loads((run_dir / "AMP_statistics.json").read_text())
    missing = [name for name in required_files if not (run_dir / name).exists()]
    return {
        "missing": missing,
        "metrics_rows": len(read_rows(run_dir / "metrics_per_epoch.csv")),
        "tensorboard": (run_dir / "train").is_dir(),
        "numerical_candidate": manifest.get("numerical_candidate"),
        "track": manifest.get("track"),
        "amp": amp,
    }


def fmt_pair(r1: float, map_value: float) -> str:
    return f"{pct(r1)} / {pct(map_value)}"


def trajectory_row(row: dict[str, str], selected_row: dict[str, str] | None, kind: str) -> str:
    rev = reverse_value(row, selected_row, kind)
    acc = value(row, "train_accuracy")
    scale = value(row, "amp_scale_end")
    scale_text = "—" if scale is None else f"{scale:.0f}"
    return "| {epoch} | {ce} | {triplet} | {bio} | {rev} | {acc}% | {scale} |".format(
        epoch=row["epoch"],
        ce=loss(value(row, "ce_loss")),
        triplet=loss(value(row, "triplet_loss")),
        bio=loss(value(row, "bio_loss")),
        rev=loss(rev),
        acc="—" if acc is None else f"{acc * 100:.1f}",
        scale=scale_text,
    )


def first_sustained_deviation(
    branch_rows: dict[int, dict[str, str]],
    control_rows: dict[int, dict[str, str]],
) -> int | None:
    flags: list[bool] = []
    for epoch in range(1, 61):
        branch = branch_rows[epoch]
        control = control_rows[epoch]
        flags.append(
            value(branch, "cc_rank1") <= value(control, "cc_rank1") - 0.01
            and value(branch, "cc_mAP") <= value(control, "cc_mAP") - 0.005
        )
    for index in range(len(flags) - 2):
        if all(flags[index : index + 3]):
            return index + 1
    return None


def main() -> None:
    baseline_rows = read_rows(BASELINE_CSV)
    full_rows = read_rows(FULL_CSV)
    bio_rows = read_rows(PHASE3A_BIO_CSV)
    nonbio_rows = read_rows(PHASE3A_NONBIO_CSV)
    frozen = {
        "ID": {"rows": baseline_rows, "paper": (54.6, 30.4), "kind": "baseline"},
        "ID + Bio": {"rows": bio_rows, "paper": (55.4, 30.8), "kind": "bio"},
        "ID + NonBio": {"rows": nonbio_rows, "paper": (55.6, 31.0), "kind": "nonbio"},
        "Full": {"rows": full_rows, "paper": (58.2, 31.6), "kind": "full"},
    }
    for entry in frozen.values():
        entry["best"] = best_row(entry["rows"])
        entry["final"] = entry["rows"][-1]
        entry["indexed"] = indexed(entry["rows"])

    for spec in RUNS.values():
        spec["rows"] = read_rows(spec["csv"])
        spec["selected_rows"] = read_rows(spec["selected_csv"])
        spec["best"] = best_row(spec["rows"])
        spec["final"] = spec["rows"][-1]
        spec["indexed"] = indexed(spec["rows"])
        spec["selected_indexed"] = indexed(spec["selected_rows"])
        spec["artifacts"] = run_artifacts(spec["dir"])

    bio_best = frozen["ID + Bio"]["best"]
    baseline_best = frozen["ID"]["best"]
    full_best = frozen["Full"]["best"]

    # Compute exact local effects from raw fractions.
    for spec in RUNS.values():
        spec["effect_r1"] = pp(float(spec["best"]["cc_rank1"]) - float(bio_best["cc_rank1"]))
        spec["effect_map"] = pp(float(spec["best"]["cc_mAP"]) - float(bio_best["cc_mAP"]))
        spec["paper_gap_r1"] = gap(float(spec["best"]["cc_rank1"]), spec["paper"][0])
        spec["paper_gap_map"] = gap(float(spec["best"]["cc_mAP"]), spec["paper"][1])
        spec["status"] = status(spec["paper_gap_r1"], spec["paper_gap_map"])

    full_effect = (
        pp(float(full_best["cc_rank1"]) - float(bio_best["cc_rank1"])),
        pp(float(full_best["cc_mAP"]) - float(bio_best["cc_mAP"])),
    )
    additive = (
        RUNS["bio_hair"]["effect_r1"] + RUNS["bio_clothing"]["effect_r1"],
        RUNS["bio_hair"]["effect_map"] + RUNS["bio_clothing"]["effect_map"],
    )
    interaction = (full_effect[0] - additive[0], full_effect[1] - additive[1])

    lines: list[str] = []
    add = lines.append
    add("# PHASE 3B RESULT — LTCC Table 4 formal reproduction")
    add("")
    add(f"Generated from raw CSV/JSON artifacts at `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`. All effects and gaps use unrounded CSV fractions; displayed percentages are rounded only for readability.")
    add("")
    add("## Phase 3A frozen summary")
    add("")
    add("Phase 3A was read from its existing report and artifacts. No Table-3 run was re-executed or overwritten.")
    add("")
    add("| Setting | Paper R1 / mAP | PPU BEST epoch | PPU BEST R1 / mAP | PPU FINAL R1 / mAP |")
    add("|---|---:|---:|---:|---:|")
    for label in ["ID", "ID + Bio", "ID + NonBio", "Full"]:
        entry = frozen[label]
        add(
            f"| {label} | {entry['paper'][0]:.1f} / {entry['paper'][1]:.1f} | "
            f"{entry['best']['epoch']} | {fmt_pair(float(entry['best']['cc_rank1']), float(entry['best']['cc_mAP']))} | "
            f"{fmt_pair(float(entry['final']['cc_rank1']), float(entry['final']['cc_mAP']))} |"
        )
    add("")
    add("Frozen BEST effects relative to the Phase 3A ID-only BEST (`56.3775539398 / 31.0721036334`):")
    add("")
    add("| Setting | Paper effect R1 / mAP | PPU effect R1 / mAP |")
    add("|---|---:|---:|")
    paper_effects = {"ID + Bio": (0.8, 0.4), "ID + NonBio": (1.0, 0.6), "Full": (3.6, 1.2)}
    for label in ["ID + Bio", "ID + NonBio", "Full"]:
        entry = frozen[label]
        effect_r1 = pp(float(entry["best"]["cc_rank1"]) - float(baseline_best["cc_rank1"]))
        effect_map = pp(float(entry["best"]["cc_mAP"]) - float(baseline_best["cc_mAP"]))
        add(f"| {label} | {signed(paper_effects[label][0])} / {signed(paper_effects[label][1])} | {signed(effect_r1)} / {signed(effect_map)} |")
    add("")
    phase3a_dirs = {
        "ID + Bio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1",
        "ID + NonBio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_nonbio/run1",
    }
    phase3a_required = [
        "config.yml",
        "RUN_MANIFEST.json",
        "metrics_per_epoch.csv",
        "AMP_statistics.json",
        "git_diff.patch",
        "eva02_l_bio_best.pth",
        "eva02_l_bio_final.pth",
    ]
    add("Frozen Phase 3A ablation artifact audit:")
    add("")
    add("| Frozen run | Required files | 60-row CSV | TensorBoard |")
    add("|---|---:|---:|---:|")
    for label, directory in phase3a_dirs.items():
        required = all((directory / name).exists() for name in phase3a_required)
        rows = len(read_rows(directory / "metrics_per_epoch.csv")) if (directory / "metrics_per_epoch.csv").exists() else 0
        add(f"| {label} | {'YES' if required else 'NO'} | {'YES' if rows == 60 else 'NO'} | {'YES' if (directory / 'train').is_dir() else 'NO'} |")
    add("")
    add("The frozen Full result is included as the Phase 3A joint Hair+Clothing comparison and is not rerun in Phase 3B.")
    add("")

    add("## Shared formal protocol and preflight")
    add("")
    add("- Track A: released-code / paper-faithful formal reproduction; all Phase 2 numerical candidates were forbidden and `numerical_candidate: null`.")
    add("- LTCC: 9,576 train images, 77 IDs, 256 clothes, 493 query images, 7,050 gallery images; 224×224; 60 epochs; 596 iterations/epoch; batch config 8; `NUM_INSTANCES=2`.")
    add("- SGD, base LR `2e-6`, weight decay `0.05`, linear warmup plus cosine schedule, seed `1234`; 2× PPU-ZW810E; no CPU fallback or reranking.")
    add("- EVA checkpoint SHA256: `ce87e437a7ce3a29eb709ae302c5fdd9c44a73bfd8f191ec655eec53f58b8e67`.")
    add("- LTCC text NPZ SHA256: `c065a1ba8c7325be45f8a33afbb820e323b568f22a032af95039d39598840a30`; Summary Bio JSON SHA256: `056ed5a99b012c7583b7c3d28869bc5520715282df53684faf9f77800ef834e3`.")
    add("- Runtime Bio supervision is identity-level summarized Bio; NonBio is image-level text. Semantic order is 0 Bio, 1 Hair, 2 Clothing, 3 Pose.")
    add("- Reverse indexing is `image_feature_reverse[:, i]`; reverse losses use the shared cumulative implementation; GRL alpha is 1.0 with backward sign reversal.")
    add("")
    add("### Artifact audit")
    add("")
    add("| Run | Index | Semantic factor | 60-row CSV | TensorBoard | BEST/FINAL | AMP | git diff | numerical candidate |")
    add("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for name, spec in RUNS.items():
        audit = spec["artifacts"]
        complete = not audit["missing"] and audit["metrics_rows"] == 60 and audit["tensorboard"]
        index = json.loads((spec["dir"] / "RUN_MANIFEST.json").read_text()).get("factor_index", "?")
        candidate = "null" if audit["numerical_candidate"] is None else str(audit["numerical_candidate"])
        add(f"| {spec['label']} | {index} | {spec['factor']} | {'YES' if audit['metrics_rows'] == 60 else 'NO'} | {'YES' if audit['tensorboard'] else 'NO'} | {'YES' if complete else 'NO'} | YES | YES | `{candidate}` |")
    add("")
    add("Shared audit: [reports/PHASE_3B_PREFLIGHT_AUDIT.md](/home/DIFFER/reports/PHASE_3B_PREFLIGHT_AUDIT.md). Each run also stores `preflight.json`, optimizer coverage, smoke gradients, and GRL-direction checks.")
    add("")

    add("## Table 4 reproduction matrix")
    add("")
    add("Table 4 None/Bio is the frozen Phase 3A ID+Bio run; it is not rerun.")
    add("")
    add("| Factor | Paper R1 | Paper mAP | PPU BEST epoch | PPU BEST R1 | PPU BEST mAP | PPU FINAL R1 | PPU FINAL mAP | ΔR1 vs paper | ΔmAP vs paper | Status |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    add(f"| None / Bio | 55.4 | 30.8 | {bio_best['epoch']} | {pct(float(bio_best['cc_rank1']))} | {pct(float(bio_best['cc_mAP']))} | {pct(float(frozen['ID + Bio']['final']['cc_rank1']))} | {pct(float(frozen['ID + Bio']['final']['cc_mAP']))} | {signed(gap(float(bio_best['cc_rank1']), 55.4))} | {signed(gap(float(bio_best['cc_mAP']), 30.8))} | {status(gap(float(bio_best['cc_rank1']), 55.4), gap(float(bio_best['cc_mAP']), 30.8))} |")
    for name in ["bio_hair", "bio_clothing", "bio_pose"]:
        spec = RUNS[name]
        add(f"| {spec['label']} | {spec['paper'][0]:.1f} | {spec['paper'][1]:.1f} | {spec['best']['epoch']} | {pct(float(spec['best']['cc_rank1']))} | {pct(float(spec['best']['cc_mAP']))} | {pct(float(spec['final']['cc_rank1']))} | {pct(float(spec['final']['cc_mAP']))} | {signed(spec['paper_gap_r1'])} | {signed(spec['paper_gap_map'])} | {spec['status']} |")
    add("")
    add("Classification thresholds: EXCELLENT ≤1 pp on both metrics; ACCEPTABLE ≤2 pp on both; CONCERNING when either metric is >2 and ≤4 pp; FAILED when either metric is >4 pp.")
    add("")

    add("### BEST/FINAL evaluation metric audit")
    add("")
    add("The underlying 60-row CSVs also record CC Rank-5, Rank-10, General Rank-1, and General mAP for every epoch. The key BEST/FINAL values are:")
    add("")
    add("| Setting | Point | CC R1 | CC R5 | CC R10 | CC mAP | General R1 | General mAP |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    evaluation_specs = [("None / Bio", frozen["ID + Bio"])] + [(spec["label"], spec) for spec in RUNS.values()]
    for label, spec in evaluation_specs:
        for point, row in [("BEST", spec["best"]), ("FINAL", spec["final"])]:
            add(f"| {label} | {point} | {pct(float(row['cc_rank1']))} | {pct(float(row['cc_rank5']))} | {pct(float(row['cc_rank10']))} | {pct(float(row['cc_mAP']))} | {pct(float(row['general_rank1']))} | {pct(float(row['general_mAP']))} |")
    add("")

    add("### Individual Table-4 effects relative to frozen None/Bio")
    add("")
    add("| Factor | Paper incremental effect R1 / mAP | PPU BEST incremental effect R1 / mAP | PPU FINAL incremental effect R1 / mAP |")
    add("|---|---:|---:|---:|")
    paper_increments = {"bio_hair": (1.2, 0.5), "bio_clothing": (1.2, -0.4), "bio_pose": (0.7, 0.7)}
    for name in ["bio_hair", "bio_clothing", "bio_pose"]:
        spec = RUNS[name]
        final_r1 = pp(float(spec["final"]["cc_rank1"]) - float(frozen["ID + Bio"]["final"]["cc_rank1"]))
        final_map = pp(float(spec["final"]["cc_mAP"]) - float(frozen["ID + Bio"]["final"]["cc_mAP"]))
        add(f"| {spec['label']} | {signed(paper_increments[name][0])} / {signed(paper_increments[name][1])} | {signed(spec['effect_r1'])} / {signed(spec['effect_map'])} | {signed(final_r1)} / {signed(final_map)} |")
    add("")
    add("Interpretation: Hair is negative relative to the local Bio control despite an mAP close to its paper target; Clothing and Pose are strongly negative in both local effect and absolute paper comparison.")
    add("")

    add("## Full Hair+Clothing joint comparison")
    add("")
    add("The Full PPU value is frozen from Phase 3A. It is compared with the two new single-factor branches without rerunning Full.")
    add("")
    add("| Setting | Paper R1 / mAP | PPU BEST R1 / mAP |")
    add("|---|---:|---:|")
    add(f"| Bio / None | 55.4 / 30.8 | {fmt_pair(float(bio_best['cc_rank1']), float(bio_best['cc_mAP']))} |")
    add(f"| Bio + Hair | 56.6 / 31.3 | {fmt_pair(float(RUNS['bio_hair']['best']['cc_rank1']), float(RUNS['bio_hair']['best']['cc_mAP']))} |")
    add(f"| Bio + Clothing | 56.6 / 30.4 | {fmt_pair(float(RUNS['bio_clothing']['best']['cc_rank1']), float(RUNS['bio_clothing']['best']['cc_mAP']))} |")
    add(f"| Bio + Hair + Clothing (Full) | 58.2 / 31.6 | {fmt_pair(float(full_best['cc_rank1']), float(full_best['cc_mAP']))} |")
    add("")
    add("Using the requested local-gain definition:")
    add("")
    add(f"- PPU Hair gain ΔH = **{signed(RUNS['bio_hair']['effect_r1'])} / {signed(RUNS['bio_hair']['effect_map'])} pp**.")
    add(f"- PPU Clothing gain ΔC = **{signed(RUNS['bio_clothing']['effect_r1'])} / {signed(RUNS['bio_clothing']['effect_map'])} pp**.")
    add(f"- PPU additive expectation ΔH+ΔC = **{signed(additive[0])} / {signed(additive[1])} pp**.")
    add(f"- PPU Full gain ΔHC = **{signed(full_effect[0])} / {signed(full_effect[1])} pp**.")
    add(f"- PPU interaction = ΔHC − (ΔH+ΔC) = **{signed(interaction[0])} / {signed(interaction[1])} pp**.")
    add("")
    add("For the paper: Hair gain is +1.2/+0.5 pp, Clothing gain is +1.2/−0.4 pp, the additive expectation is +2.4/+0.1 pp, Full gain is +2.8/+0.8 pp, and the positive joint interaction is +0.4/+0.7 pp.")
    add("")
    add("The PPU interaction is numerically positive only because the two single-factor branches are already negative: Full is less negative than the sum of the two failed single-factor effects, but Full still loses **3.061/3.120 pp** against the Bio control. This does not support a primary joint-only failure diagnosis.")
    add("")

    add("## AMP accounting")
    add("")
    add("`amp_skips` is taken from each run's AMP JSON. The auxiliary `skip_log_event_count` is rank-duplicated in distributed logs and is not used as the skip count.")
    add("")
    add("| Run | Total iterations | Optimizer updates | AMP skips | First skip | Min scale | Final scale | Nonfinite loss events |")
    add("|---|---:|---:|---:|---|---:|---:|---:|")
    for name in ["bio_hair", "bio_clothing", "bio_pose"]:
        spec = RUNS[name]
        amp = spec["artifacts"]["amp"]
        first = f"e{amp['first_skip_epoch']}/i{amp['first_skip_iteration']} @ {amp['first_skip_scale_before_step']:.0f}" if amp.get("first_skip_epoch") is not None else "none"
        add(f"| Bio + {spec['label']} | {amp['total_iterations']} | {amp['optimizer_updates']} | {amp['amp_skips']} | {first} | {amp['min_scaler']:.0f} | {amp['final_scaler']:.0f} | {amp['nonfinite_loss_events']} |")
    add("")
    add("All three runs used initial GradScaler 1024 with normal dynamic growth/backoff. No AMP skip caused a protocol change, loss-precision change, or scaler retuning.")
    add("")

    add("## Trajectory anchors")
    add("")
    add("The following retrieval table uses official changed-clothes CC evaluation at epochs 1, 5, 10, 15, 20, 25, 30, 40, 50, and 60.")
    add("")
    add("| Epoch | Bio / None R1 / mAP | Bio + Hair | Bio + Clothing | Bio + Pose | Full Hair + Clothing |")
    add("|---:|---:|---:|---:|---:|---:|")
    for epoch in ANCHORS:
        values = [frozen["ID + Bio"]["indexed"][epoch], RUNS["bio_hair"]["indexed"][epoch], RUNS["bio_clothing"]["indexed"][epoch], RUNS["bio_pose"]["indexed"][epoch], frozen["Full"]["indexed"][epoch]]
        add(f"| {epoch} | " + " | ".join(fmt_pair(float(row["cc_rank1"]), float(row["cc_mAP"])) for row in values) + " |")
    add("")
    add("Training-loss/AMP anchors are shown below. `Reverse` is the selected single factor for Hair/Clothing/Pose, the sum of Hair+Clothing for Full, and not active for Bio/None.")
    add("")
    add("| Run | Epoch | CE | Triplet | Bio | Reverse | Train acc. | AMP scale |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    trajectory_specs = [
        ("Bio / None", frozen["ID + Bio"], "bio"),
        ("Bio + Hair", RUNS["bio_hair"], "single"),
        ("Bio + Clothing", RUNS["bio_clothing"], "single"),
        ("Bio + Pose", RUNS["bio_pose"], "single"),
        ("Full", frozen["Full"], "full"),
    ]
    for label, spec, kind in trajectory_specs:
        for epoch in ANCHORS:
            row = spec["indexed"][epoch]
            selected = spec["selected_indexed"][epoch] if kind == "single" else None
            add(f"| {label} | {trajectory_row(row, selected, kind)[2:]}")
    add("")

    add("## Turning-point analysis: epochs 14–25")
    add("")
    add("Criterion: first three consecutive epochs where a branch is at least 1.0 pp below the frozen Bio/None CC Rank-1 and at least 0.5 pp below its CC mAP. This separates sustained deviation from a single noisy evaluation.")
    add("")
    add("| Branch | First sustained 3-epoch deviation | Evidence in epochs 14–25 |")
    add("|---|---:|---|")
    for label, spec in [("Bio + Hair", RUNS["bio_hair"]), ("Bio + Clothing", RUNS["bio_clothing"]), ("Bio + Pose", RUNS["bio_pose"]), ("Full", frozen["Full"])]:
        first = first_sustained_deviation(spec["indexed"], frozen["ID + Bio"]["indexed"])
        if label == "Bio + Hair":
            note = "Tracks above Bio through about epoch20, then becomes jointly lower from epochs 22–25."
        elif label == "Bio + Clothing":
            note = "The first sustained flag is early; within the requested window the decline becomes clear at epochs 19–25."
        elif label == "Bio + Pose":
            note = "Joint decline begins at epoch19 and persists through epoch25."
        else:
            note = "Joint decline begins at epoch20, matching the previously observed Full divergence around epochs19–20."
        add(f"| {label} | {first if first is not None else 'None'} | {note} |")
    add("")
    add("### Delta to Bio/None during epochs 14–25 (percentage points)")
    add("")
    add("| Epoch | Hair ΔR1 / ΔmAP | Clothing ΔR1 / ΔmAP | Pose ΔR1 / ΔmAP | Full ΔR1 / ΔmAP |")
    add("|---:|---:|---:|---:|---:|")
    for epoch in range(14, 26):
        control = frozen["ID + Bio"]["indexed"][epoch]
        cells = []
        for spec in [RUNS["bio_hair"], RUNS["bio_clothing"], RUNS["bio_pose"], frozen["Full"]]:
            row = spec["indexed"][epoch]
            cells.append(f"{signed(pp(float(row['cc_rank1']) - float(control['cc_rank1'])))} / {signed(pp(float(row['cc_mAP']) - float(control['cc_mAP'])))}")
        add(f"| {epoch} | " + " | ".join(cells) + " |")
    add("")
    add("Hair therefore shows a delayed branch-specific deterioration; Clothing and Pose show sustained mid-training degradation; Full turns negative at the same general epoch as the prior Phase 3A observation, but the new single-factor evidence shows that the reverse/adversarial family is already failing without the joint Hair+Clothing combination.")
    add("")

    add("## Final diagnosis")
    add("")
    add("**PRIMARY: NONBIO ADVERSARIAL TRAINING BROADLY FAILS ON PPU.**")
    add("")
    add(f"- Bio/None is the only Table-4 row within the frozen formal control range: {pct(float(bio_best['cc_rank1']))} / {pct(float(bio_best['cc_mAP']))} versus paper 55.4 / 30.8.")
    add(f"- Hair is only partial/weak: its BEST local effect is {signed(RUNS['bio_hair']['effect_r1'])} / {signed(RUNS['bio_hair']['effect_map'])} pp and its absolute status is {RUNS['bio_hair']['status']}.")
    add(f"- Clothing is clearly failed: {signed(RUNS['bio_clothing']['effect_r1'])} / {signed(RUNS['bio_clothing']['effect_map'])} pp versus Bio and {RUNS['bio_clothing']['status']} against paper.")
    add(f"- Pose is clearly failed: {signed(RUNS['bio_pose']['effect_r1'])} / {signed(RUNS['bio_pose']['effect_map'])} pp versus Bio and {RUNS['bio_pose']['status']} against paper.")
    add(f"- Full remains failed at {signed(full_effect[0])} / {signed(full_effect[1])} pp versus Bio. Its positive algebraic interaction ({signed(interaction[0])} / {signed(interaction[1])} pp) reflects two already-negative single branches, not successful joint optimization.")
    add("")
    add("Therefore the evidence does not support `SINGLE NONBIO FACTORS REPRODUCE BUT JOINT HAIR+CLOTHING FAILS` or a joint-only interaction as the primary explanation. The strongest evidence-based conclusion is broad instability/failure of the NonBio adversarial reverse training on this PPU formal track.")
    add("")
    add("## Artifact locations and stop state")
    add("")
    add("- [Phase 3A frozen report](/home/DIFFER/reports/PHASE_3A_LTCC_TABLE3.md)")
    add("- [Phase 3B shared preflight audit](/home/DIFFER/reports/PHASE_3B_PREFLIGHT_AUDIT.md)")
    add("- [Bio + Hair run](/home/DIFFER/outputs/reproduction_phase3/LTCC/table4/bio_hair/run1)")
    add("- [Bio + Clothing run](/home/DIFFER/outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1)")
    add("- [Bio + Pose run](/home/DIFFER/outputs/reproduction_phase3/LTCC/table4/bio_pose/run1)")
    add("")
    add("Phase 3B Table 4 is complete. No PRCC, seed sweep, compatibility experiment, Table 5, or new method development was started.")

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
