#!/usr/local/bin/python3
"""Read-only Phase 4R-C trajectory/change-point analysis.

Consumes existing CSV/TensorBoard artifacts.  It never launches a model and
never performs training or parameter updates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics"
METRICS = {
    "bio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv",
    "hair": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1/metrics_per_epoch.csv",
    "clothing": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1/metrics_per_epoch.csv",
    "pose": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1/metrics_per_epoch.csv",
    "full": ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/metrics_per_epoch.csv",
}
EVENT_DIRS = {
    "bio": ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/train",
    "hair": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1/train",
    "clothing": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1/train",
    "pose": ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1/train",
    "full": ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/train",
}
OFFICIAL = ROOT / "reports/OFFICIAL_LTCC_METRICS_PER_EPOCH.csv"
ANCHORS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
STEPS_PER_EPOCH = 596


def read_csv(path):
    with path.open() as stream:
        return {int(float(row["epoch"])): row for row in csv.DictReader(stream)}


def scalar_epoch_means(directory, tags):
    files = sorted(directory.glob("events.out.tfevents*"))
    result = {tag: {epoch: [] for epoch in range(1, 61)} for tag in tags}
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in tags:
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                # The released driver starts logging at the first epoch-end
                # global step: 596, then logs every training iteration.
                epoch = int((event.step - STEPS_PER_EPOCH) // STEPS_PER_EPOCH) + 1
                if 1 <= epoch <= 60:
                    result[tag][epoch].append(float(event.value))
    return {
        tag: {str(epoch): float(np.mean(values)) for epoch, values in epochs.items() if values}
        for tag, epochs in result.items()
    }


def trajectory_change_points(metrics):
    result = {}
    for name in ["hair", "clothing", "pose", "full"]:
        candidates = {}
        for metric, threshold in [("cc_rank1", 0.01), ("cc_mAP", 0.005)]:
            first = None
            for epoch in range(1, 59):
                if all(
                    epoch + offset in metrics[name] and epoch + offset in metrics["bio"]
                    and float(metrics["bio"][epoch + offset][metric]) - float(metrics[name][epoch + offset][metric]) >= threshold
                    for offset in range(3)
                ):
                    first = epoch
                    break
            candidates[metric] = first
        result[name] = {
            "first_three_epoch_r1_gap_ge_1pp": candidates["cc_rank1"],
            "first_three_epoch_map_gap_ge_0_5pp": candidates["cc_mAP"],
            "first_joint_gap": max(
                [value for value in candidates.values() if value is not None],
                default=None,
            ),
        }
    return result


def main():
    metrics = {name: read_csv(path) for name, path in METRICS.items()}
    anchor_table = {}
    for epoch in ANCHORS:
        anchor_table[str(epoch)] = {}
        for name, rows in metrics.items():
            row = rows[epoch]
            anchor_table[str(epoch)][name] = {
                "cc_rank1": float(row["cc_rank1"]),
                "cc_mAP": float(row["cc_mAP"]),
                "ce_loss": float(row["ce_loss"]),
                "triplet_loss": float(row["triplet_loss"]),
                "bio_loss": float(row["bio_loss"]),
                "reverse_hair_loss_log_column": None if row["reverse_hair_loss"] == "NA" else float(row["reverse_hair_loss"]),
                "reverse_cloth_loss_log_column": None if row["reverse_cloth_loss"] == "NA" else float(row["reverse_cloth_loss"]),
                "amp_scale_end": float(row["amp_scale_end"]),
            }
    tags_by_run = {
        "bio": ["acc/clip_bio_0", "loss/clip_bio_0"],
        "hair": ["acc/clip_bio_reverse_0", "loss/clip_bio_reverse_0"],
        "clothing": ["acc/clip_bio_reverse_0", "loss/clip_bio_reverse_0"],
        "pose": ["acc/clip_bio_reverse_0", "loss/clip_bio_reverse_0"],
        "full": ["acc/clip_bio_reverse_0", "acc/clip_bio_reverse_1", "loss/clip_bio_reverse_0", "loss/clip_bio_reverse_1"],
    }
    reverse_training = {name: scalar_epoch_means(EVENT_DIRS[name], tags) for name, tags in tags_by_run.items()}
    official = read_csv(OFFICIAL)
    output = {
        "phase": "4R-C10",
        "read_only": True,
        "source_metrics": {name: str(path) for name, path in METRICS.items()},
        "source_tensorboard": {name: str(path) for name, path in EVENT_DIRS.items()},
        "csv_anchor_table": anchor_table,
        "change_points_vs_bio": trajectory_change_points(metrics),
        "reverse_training_tensorboard_epoch_means": reverse_training,
        "official_curve_anchor_table": {
            str(epoch): official[epoch] for epoch in ANCHORS if epoch in official
        },
        "logging_caveat": {
            "single_factor_slot_name": "The released loss logger labels factor slot 0 as reverse_hair even when NOBIO_INDEX is [2] Clothing or [3] Pose; semantic factor identity is taken from the run manifest/config, not this column name.",
            "full_mAP_discrepancy": "The Full metrics CSV FINAL row is 29.4617 mAP while the frozen BEST/evaluator value used by reproduction reports is 29.2657; this report preserves both and uses the frozen value for final model claims.",
        },
    }
    out = OUT / "trajectory_analysis.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
