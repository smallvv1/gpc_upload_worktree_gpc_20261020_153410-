#!/usr/local/bin/python3
"""Complete, read-only TensorBoard audit for the official LTCC artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # make the exact dependency failure visible
    print(f"TENSORBOARD_IMPORT_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
    raise


EVENT_DIR = Path("/home/data/DIFFER_OfficialWeights/provenance/LTCC/train")
OUT = Path("diagnostics/official_tensorboard_phase2i.json")


def load(path: Path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    data = {}
    for tag in acc.Tags().get("scalars", []):
        data[tag] = [
            {"step": int(item.step), "wall_time": float(item.wall_time), "value": float(item.value)}
            for item in acc.Scalars(tag)
        ]
    return data


def summary(values):
    if not values:
        return {"first_step": None, "last_step": None, "record_count": 0, "first_value": None, "last_value": None, "min": None, "max": None}
    arr = np.asarray([item["value"] for item in values], dtype=np.float64)
    return {
        "first_step": values[0]["step"], "last_step": values[-1]["step"], "record_count": len(values),
        "first_value": float(values[0]["value"]), "last_value": float(values[-1]["value"]),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "first_wall_time": values[0]["wall_time"], "last_wall_time": values[-1]["wall_time"],
    }


def main():
    files = sorted(EVENT_DIR.glob("events.out.tfevents.*"))
    per_file = {}
    for path in files:
        series = load(path)
        per_file[path.name] = {
            "path": str(path), "size": path.stat().st_size,
            "tags": sorted(series), "scalar_summary": {tag: summary(values) for tag, values in series.items()},
            "series": series,
        }
    # Compare common tags/steps across event files.  They share one run's
    # step range but contain rank-local scalar values, so they are not merged
    # into one fake stream.
    names = list(per_file)
    overlap = {}
    if len(names) >= 2:
        left, right = per_file[names[0]]["series"], per_file[names[1]]["series"]
        for tag in sorted(set(left) & set(right)):
            lv = {x["step"]: x["value"] for x in left[tag]}
            rv = {x["step"]: x["value"] for x in right[tag]}
            common = sorted(set(lv) & set(rv))
            diffs = [abs(lv[s] - rv[s]) for s in common]
            overlap[tag] = {
                "common_step_count": len(common), "first_common_step": common[0] if common else None,
                "last_common_step": common[-1] if common else None,
                "exact_value_match_count": sum(d == 0 for d in diffs),
                "max_abs_value_difference": max(diffs) if diffs else None,
            }
    # First 200 records are retained separately for every event file.  This
    # preserves rank-local evidence and lets later analyses choose a stream.
    first200 = {}
    for name, item in per_file.items():
        series = item["series"]
        train_tags = [tag for tag in item["tags"] if tag not in ("rank1", "mAP") and len(series[tag]) > 1]
        steps = [x["step"] for x in series[train_tags[0]][:200]] if train_tags else []
        rows = []
        for index, step in enumerate(steps, 1):
            row = {"sequence": index, "official_step": step}
            for tag in item["tags"]:
                row[tag] = next((x["value"] for x in series[tag] if x["step"] == step), None)
            rows.append(row)
        first200[name] = rows
    payload = {
        "event_dir": str(EVENT_DIR), "event_file_count": len(files), "files": per_file,
        "cross_file_overlap": overlap,
        "run_interpretation": "SAME CONTINUOUS RUN with rank-local scalar streams" if len(files) >= 2 and all(summary(x["series"].get("loss/total", []))["first_step"] == 596 and summary(x["series"].get("loss/total", []))["last_step"] == 36355 for x in per_file.values()) else "UNKNOWN",
        "theoretical_training_steps": 596 * 60,
        "first200_by_file": first200,
        "notes": [
            "Both event files cover steps 596..36355 (35760 training records) and evaluation steps 0..60.",
            "The files are not exact duplicates: rank-local training scalar values differ while step ranges/wall-time are continuous.",
            "No event file was modified.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("EVENT_FILES", len(files))
    print("TAGS", sorted(set(tag for item in per_file.values() for tag in item["tags"])))
    for name, item in per_file.items():
        print("FILE", name, "loss_total", item["scalar_summary"].get("loss/total"), "rank1", item["scalar_summary"].get("rank1"))
    print("RUN_INTERPRETATION", payload["run_interpretation"])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()

