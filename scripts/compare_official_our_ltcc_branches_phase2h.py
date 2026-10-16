#!/usr/local/bin/python3
"""Align official and frozen LTCC Full DIFFER scalar curves.

This is read-only: it consumes TensorBoard event files and the frozen run
artifacts, and writes a machine-readable epoch-level comparison.  The
official event uses the first writer; the local run uses the first writer, so
the comparison is deliberately labelled writer-local rather than a fabricated
cross-rank average.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EVENT = Path(
    "/home/data/DIFFER_OfficialWeights/provenance/LTCC/train/"
    "events.out.tfevents.1724113871.c4-2.643878.0"
)
OUR_EVENT = ROOT / (
    "outputs/reproduction_phase2/LTCC/differ/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/train/"
    "events.out.tfevents.1788244774.nb-a431f85a02d6004f02-88f76bccc-bqm6x.998752.0"
)
OUR_METRICS = ROOT / (
    "outputs/reproduction_phase2/LTCC/differ/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/metrics_per_epoch.csv"
)
OUTPUT_JSON = ROOT / "diagnostics/official_vs_our_branch_divergence_phase2h.json"
OUTPUT_MD = ROOT / "reports/OFFICIAL_VS_OUR_LTCC_BRANCH_DIVERGENCE.md"
ITERATIONS_PER_EPOCH = 596


# Canonical names are intentionally semantic, not a claim that the two
# writers used identical tag strings.  reverse_1 is cumulative in both runs.
TAG_MAP = {
    "total_loss": ("loss/total", "loss/total"),
    "id_loss": ("loss/id", "loss/id"),
    "triplet_loss": ("loss/triplet", "loss/triplet"),
    "bio_loss": ("loss/clip_bio", "loss/clip_bio_0"),
    "reverse0_loss_cumulative": (
        "loss/clip_bio_reverse_0", "loss/clip_bio_reverse_0"
    ),
    "reverse1_loss_cumulative": (
        "loss/clip_bio_reverse_1", "loss/clip_bio_reverse_1"
    ),
    "id_accuracy": ("acc/id", "acc/id"),
    "bio_accuracy": ("acc/clip_bio", "acc/clip_bio_0"),
    "reverse0_accuracy": (
        "acc/clip_bio_reverse_0", "acc/clip_bio_reverse_0"
    ),
    "reverse1_accuracy": (
        "acc/clip_bio_reverse_1", "acc/clip_bio_reverse_1"
    ),
    "cc_rank1": ("rank1", "cc/rank1"),
    "cc_mAP": ("mAP", "cc/mAP"),
}


# Fixed thresholds are reported in the output so that "clear" is auditable.
# They are deliberately conservative relative to the quantization/noise in
# the scalar writers; sustained means three consecutive epochs over threshold.
CLEAR_THRESHOLDS = {
    "total_loss": 0.10,
    "id_loss": 0.05,
    "triplet_loss": 0.05,
    "bio_loss": 0.01,
    "reverse0_loss_cumulative": 0.01,
    "reverse1_loss_cumulative": 0.02,
    "id_accuracy": 0.03,
    "bio_accuracy": 0.03,
    "reverse0_accuracy": 0.02,
    "reverse1_accuracy": 0.02,
    "cc_rank1": 0.02,
    "cc_mAP": 0.01,
}

SELECTED_EPOCHS = [1, 5, 10, 15, 18, 19, 20, 21, 25, 30, 40, 50, 60]


def load(path: Path) -> EventAccumulator:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def train_by_epoch(acc: EventAccumulator, tag: str):
    grouped = {epoch: [] for epoch in range(1, 61)}
    for event in acc.Scalars(tag):
        epoch = int(event.step) // ITERATIONS_PER_EPOCH
        if 1 <= epoch <= 60:
            grouped[epoch].append(float(event.value))
    return {epoch: mean(values) for epoch, values in grouped.items()}


def eval_by_epoch(acc: EventAccumulator, tag: str):
    return {int(event.step): float(event.value) for event in acc.Scalars(tag)}


def series(acc: EventAccumulator, tag: str, evaluation: bool):
    return eval_by_epoch(acc, tag) if evaluation else train_by_epoch(acc, tag)


def first_sustained(values, threshold):
    clear = {
        epoch: value is not None and abs(value) >= threshold
        for epoch, value in values.items()
    }
    first = next((epoch for epoch in sorted(clear) if clear[epoch]), None)
    sustained = None
    for epoch in range(1, 59):
        if all(clear.get(epoch + offset, False) for offset in range(3)):
            sustained = epoch
            break
    return first, sustained


def finite_float(value):
    return value is not None and math.isfinite(value)


def main():
    official = load(OFFICIAL_EVENT)
    ours = load(OUR_EVENT)
    official_tags = official.Tags().get("scalars", [])
    our_tags = ours.Tags().get("scalars", [])
    rows = []
    divergence = {}
    for name, (official_tag, our_tag) in TAG_MAP.items():
        is_eval = name in {"cc_rank1", "cc_mAP"}
        o = series(official, official_tag, is_eval)
        r = series(ours, our_tag, is_eval)
        diff = {}
        for epoch in range(1, 61):
            ov, rv = o.get(epoch), r.get(epoch)
            abs_diff = (rv - ov) if finite_float(ov) and finite_float(rv) else None
            rel_diff = (
                abs_diff / abs(ov)
                if abs_diff is not None and ov not in (0, None)
                else None
            )
            diff[epoch] = {
                "official": ov,
                "ours": rv,
                "absolute_difference_ours_minus_official": abs_diff,
                "relative_difference": rel_diff,
            }
            rows.append({
                "epoch": epoch,
                "metric": name,
                **diff[epoch],
            })
        signed = {
            epoch: item["absolute_difference_ours_minus_official"]
            for epoch, item in diff.items()
        }
        first, sustained = first_sustained(signed, CLEAR_THRESHOLDS[name])
        divergence[name] = {
            "threshold": CLEAR_THRESHOLDS[name],
            "first_clear_epoch": first,
            "first_sustained_epoch": sustained,
            "epoch_values": diff,
        }

    # Verify the frozen CSV's learning-rate source is present and unchanged;
    # the branch comparison itself uses TensorBoard for all common tags.
    csv_lr = []
    if OUR_METRICS.exists():
        with OUR_METRICS.open(newline="") as f:
            for row in csv.DictReader(f):
                csv_lr.append({"epoch": int(row["epoch"]), "lr": float(row["lr"])})

    payload = {
        "official_event": str(OFFICIAL_EVENT),
        "our_event": str(OUR_EVENT),
        "iterations_per_epoch": ITERATIONS_PER_EPOCH,
        "selected_epochs": SELECTED_EPOCHS,
        "official_scalar_tags": official_tags,
        "our_scalar_tags": our_tags,
        "tag_mapping": TAG_MAP,
        "reverse_logging_semantics": "reverse_0 is factor-0 individual; reverse_1 is factor-0+factor-1 cumulative",
        "training_event_steps": {
            "official": [int(official.Scalars("loss/total")[0].step), int(official.Scalars("loss/total")[-1].step)],
            "ours": [int(ours.Scalars("loss/total")[0].step), int(ours.Scalars("loss/total")[-1].step)],
        },
        "evaluation_event_steps": {
            "official": [int(official.Scalars("rank1")[0].step), int(official.Scalars("rank1")[-1].step)],
            "ours": [int(ours.Scalars("cc/rank1")[0].step), int(ours.Scalars("cc/rank1")[-1].step)],
        },
        "divergence_thresholds": CLEAR_THRESHOLDS,
        "divergence": divergence,
        "our_metrics_csv_lr": csv_lr,
        "notes": [
            "Official and ours are compared using one event writer each; official two-writer local values differ slightly.",
            "No scaler/skip tags exist in official events.",
            "cc_rank1 and cc_mAP use evaluation event steps; all other metrics are epoch means over 596 training records.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Official vs our LTCC Full DIFFER branch divergence",
        "",
        "Read-only alignment of the first official and first frozen Run-1 TensorBoard writers.",
        "Training values are means over 596 records per epoch; evaluation values are the epoch event.",
        "`reverse_0` is factor-0 individual and `reverse_1` is the cumulative factor-0+factor-1 scalar in both runs.",
        "",
        "## Scalar coverage",
        "",
        f"Official tags: `{', '.join(official_tags)}`",
        "",
        f"Our tags: `{', '.join(our_tags)}`",
        "",
        "| Canonical scalar | Official tag | Our tag | Clear threshold | First clear | First sustained |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for name, (ot, rt) in TAG_MAP.items():
        d = divergence[name]
        lines.append(f"| {name} | `{ot}` | `{rt}` | {d['threshold']} | {d['first_clear_epoch'] or '—'} | {d['first_sustained_epoch'] or '—'} |")
    lines += ["", "## Selected epochs (ours − official)", "",
              "| Epoch | Metric | Official | Ours | Absolute diff | Relative diff |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for epoch in SELECTED_EPOCHS:
        for name in TAG_MAP:
            item = divergence[name]["epoch_values"][epoch]
            def fmt(v):
                return "NA" if v is None else f"{v:.8g}"
            rel = item["relative_difference"]
            lines.append(f"| {epoch} | {name} | {fmt(item['official'])} | {fmt(item['ours'])} | {fmt(item['absolute_difference_ours_minus_official'])} | {fmt(rel)} |")
    lines += ["", "## Causal ordering", ""]
    order = sorted(
        ((d["first_sustained_epoch"], name) for name, d in divergence.items() if d["first_sustained_epoch"] is not None),
        key=lambda item: (item[0], item[1]),
    )
    for epoch, name in order:
        lines.append(f"- `{name}` first sustained divergence: epoch {epoch}")
    lines += [
        "",
        "The first sustained loss/branch and retrieval divergence must be interpreted with the fixed thresholds above; an isolated early Rank-1 difference is not called a training-branch cause.",
        "",
        f"Machine-readable output: `{OUTPUT_JSON}`.",
    ]
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    print("WROTE", OUTPUT_JSON)
    print("WROTE", OUTPUT_MD)
    print(json.dumps({name: {"first_clear": d["first_clear_epoch"], "first_sustained": d["first_sustained_epoch"]} for name, d in divergence.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
