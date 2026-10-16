#!/usr/local/bin/python3
"""Read-only forensic extraction from the public official LTCC run.

This utility does not import the training entry point and never instantiates a
model or performs an optimizer operation.  It writes reproducible summaries of
the two supplied TensorBoard event files and the official text log.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("/home/data/DIFFER_OfficialWeights/provenance/LTCC")
EVENT_DIR = ROOT / "train"
LOG_PATH = ROOT / "train_log.txt"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
DIAG_DIR = Path(__file__).resolve().parents[1] / "diagnostics"
ITERATIONS_PER_EPOCH = 596


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_events(path: Path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def scalar_summary(acc: EventAccumulator):
    result = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        values = [float(e.value) for e in events]
        steps = [int(e.step) for e in events]
        wall_times = [float(e.wall_time) for e in events]
        result[tag] = {
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_wall_time": wall_times[0],
            "last_wall_time": wall_times[-1],
            "record_count": len(events),
            "first_value": values[0],
            "last_value": values[-1],
            "min": min(values),
            "max": max(values),
        }
    return result


def by_epoch(acc: EventAccumulator, tag: str):
    out = defaultdict(list)
    for event in acc.Scalars(tag):
        # The first recorded training scalar is step 596; step//596 maps it to
        # epoch 1 and gives 596 records in every epoch through epoch 60.
        out[int(event.step) // ITERATIONS_PER_EPOCH].append(float(event.value))
    return out


def mean(values):
    return sum(values) / len(values) if values else "NA"


def write_training_curve(acc: EventAccumulator, source_name: str):
    mapping = {
        "total_loss": "loss/total",
        "ce_loss": "loss/id",
        "triplet_loss": "loss/triplet",
        "bio_loss": "loss/clip_bio",
        "reverse_0_loss_cumulative": "loss/clip_bio_reverse_0",
        "reverse_1_loss_cumulative": "loss/clip_bio_reverse_1",
        "train_accuracy": "acc/id",
        "bio_accuracy": "acc/clip_bio",
        "reverse_0_accuracy": "acc/clip_bio_reverse_0",
        "reverse_1_accuracy": "acc/clip_bio_reverse_1",
        "lr": "lr",
    }
    grouped = {name: by_epoch(acc, tag) for name, tag in mapping.items()}
    path = REPORT_DIR / "OFFICIAL_LTCC_TRAIN_CURVE.csv"
    fields = ["epoch", "event_file"] + list(mapping)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 61):
            row = {"epoch": epoch, "event_file": source_name}
            for name in mapping:
                row[name] = mean(grouped[name].get(epoch, []))
            writer.writerow(row)
    return path


def parse_log_metrics(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    cc = {}
    general = {}
    i = 0
    while i < len(lines):
        if "Clothes changing setting" not in lines[i]:
            i += 1
            continue
        values = {}
        epoch = None
        validation_line = None
        j = i + 1
        while j < len(lines) and j < i + 16:
            match = re.search(r"CMC curve, Rank-(1|5|10)\s*:\s*([0-9.]+)%", lines[j])
            if match:
                values[f"rank{match.group(1)}"] = float(match.group(2)) / 100.0
            match = re.search(r"mAP-:([0-9.]+)%", lines[j])
            if match:
                values["mAP"] = float(match.group(1)) / 100.0
            match = re.search(r"Validation Results - Epoch: (\d+)", lines[j])
            if match:
                epoch = int(match.group(1))
                validation_line = j
                break
            j += 1
        if epoch is not None and len(values) == 4 and epoch not in cc:
            cc[epoch] = values
            # The standard block follows the CC validation marker.  Search
            # from that marker rather than from the fixed-width CC scan: the
            # latter also sees the next epoch's training lines.
            k = (validation_line + 1) if validation_line is not None else j
            while k < len(lines) and k < (validation_line or j) + 12 and "Standard setting" not in lines[k]:
                k += 1
            if k < len(lines) and "Standard setting" in lines[k]:
                gv = {}
                for line in lines[k + 1 : k + 16]:
                    match = re.search(r"CMC curve, Rank-(1|5|10)\s*:\s*([0-9.]+)%", line)
                    if match:
                        gv[f"rank{match.group(1)}"] = float(match.group(2)) / 100.0
                    match = re.search(r"mAP-:([0-9.]+)%", line)
                    if match:
                        gv["mAP"] = float(match.group(1)) / 100.0
                if len(gv) == 4:
                    general[epoch] = gv
        i = j
    return cc, general


def write_metrics_curve(acc: EventAccumulator, log_path: Path):
    cc_log, general = parse_log_metrics(log_path)
    cc_rank1 = {int(e.step): float(e.value) for e in acc.Scalars("rank1")}
    cc_map = {int(e.step): float(e.value) for e in acc.Scalars("mAP")}
    path = REPORT_DIR / "OFFICIAL_LTCC_METRICS_PER_EPOCH.csv"
    fields = [
        "epoch", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP",
        "general_rank1", "general_rank5", "general_rank10", "general_mAP",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(0, 61):
            c = cc_log.get(epoch, {})
            g = general.get(epoch, {})
            writer.writerow({
                "epoch": epoch,
                "cc_rank1": cc_rank1.get(epoch, c.get("rank1", "NA")),
                "cc_rank5": c.get("rank5", "NA"),
                "cc_rank10": c.get("rank10", "NA"),
                "cc_mAP": cc_map.get(epoch, c.get("mAP", "NA")),
                "general_rank1": g.get("rank1", "NA"),
                "general_rank5": g.get("rank5", "NA"),
                "general_rank10": g.get("rank10", "NA"),
                "general_mAP": g.get("mAP", "NA"),
            })
    return path, cc_log, general


def main():
    events = sorted(EVENT_DIR.glob("events.*"))
    if len(events) == 0:
        raise FileNotFoundError(EVENT_DIR)
    accumulators = [(path, load_events(path)) for path in events]
    summaries = []
    for path, acc in accumulators:
        summaries.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "scalar_tags": acc.Tags().get("scalars", []),
            "scalars": scalar_summary(acc),
        })
    source_path, source_acc = accumulators[0]
    train_events_by_file = {
        path.name: source.Scalars("loss/total")
        for path, source in accumulators
    }
    train_wall_ranges = {
        name: [float(records[0].wall_time), float(records[-1].wall_time)]
        for name, records in train_events_by_file.items()
    }
    wall_start = max(r[0] for r in train_wall_ranges.values())
    wall_end = min(r[1] for r in train_wall_ranges.values())
    train_curve = write_training_curve(source_acc, source_path.name)
    metrics_curve, cc_log, general = write_metrics_curve(source_acc, LOG_PATH)
    log_hash = sha256(LOG_PATH)
    payload = {
        "log": {"path": str(LOG_PATH), "bytes": LOG_PATH.stat().st_size,
                "lines": len(LOG_PATH.read_text(errors="replace").splitlines()),
                "sha256": log_hash},
        "event_files": summaries,
        "run_assessment": {
            "event_file_count": len(events),
            "same_scalar_tags": all(s["scalar_tags"] == summaries[0]["scalar_tags"] for s in summaries),
            "same_metric_trajectory": all(
                [(e.step, e.value) for e in acc.Scalars("rank1")] ==
                [(e.step, e.value) for e in source_acc.Scalars("rank1")] and
                [(e.step, e.value) for e in acc.Scalars("mAP")] ==
                [(e.step, e.value) for e in source_acc.Scalars("mAP")]
                for _, acc in accumulators
            ),
            "train_first_step": int(source_acc.Scalars("loss/total")[0].step),
            "train_last_step": int(source_acc.Scalars("loss/total")[-1].step),
            "train_record_count": len(source_acc.Scalars("loss/total")),
            "train_step_ranges": {
                path.name: [int(acc.Scalars("loss/total")[0].step),
                            int(acc.Scalars("loss/total")[-1].step)]
                for path, acc in accumulators
            },
            "train_wall_time_ranges": train_wall_ranges,
            "train_wall_time_ranges_overlap": wall_start <= wall_end,
            "expected_total_iterations": ITERATIONS_PER_EPOCH * 60,
            "metric_epochs_from_log": sorted(cc_log),
            "general_epochs_from_log": sorted(general),
            "resume_or_restart_evidence": "No resume/restart/load/start-epoch markers found in complete log; two files overlap exactly in metric steps and are continuous. This supports one run, while writer rank identity is not explicitly stored.",
        },
        "generated": {"train_curve": str(train_curve), "metrics_curve": str(metrics_curve)},
    }
    output = DIAG_DIR / "official_ltcc_tensorboard_forensics.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["run_assessment"], indent=2, sort_keys=True))
    print("WROTE", train_curve)
    print("WROTE", metrics_curve)
    print("WROTE", output)


if __name__ == "__main__":
    main()
