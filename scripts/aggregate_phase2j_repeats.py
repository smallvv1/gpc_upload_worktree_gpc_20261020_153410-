#!/usr/local/bin/python3
"""Aggregate the already completed PHASE 2J-F 200-step repeats."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "R-A": ROOT / "diagnostics/phase2j_r_a_200.json",
    "R-B": ROOT / "diagnostics/phase2j_r_b_200.json",
    "R-C": ROOT / "diagnostics/phase2j_r_c_200.json",
}
FIELDS = (
    "total_loss", "id_loss", "triplet_loss", "bio_loss", "reverse0_loss",
    "reverse1_loss", "id_acc", "bio_acc", "reverse0_acc", "reverse1_acc",
)


def pearson(a, b):
    if len(a) < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distance(a, b):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    delta = y - x
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(delta))), "pearson": pearson(x, y), "normalized_rmse": rmse / (float(np.std(x)) + 1e-12)}


def load_official():
    path = ROOT / "reports/OFFICIAL_FIRST200_SCALARS.csv"
    rows = list(csv.DictReader(path.open()))
    mapping = {
        "total_loss": "loss/total", "id_loss": "loss/id", "triplet_loss": "loss/triplet",
        "bio_loss": "loss/clip_bio", "reverse0_loss": "loss/clip_bio_reverse_0",
        "reverse1_loss": "loss/clip_bio_reverse_1", "id_acc": "acc/id",
        "bio_acc": "acc/clip_bio", "reverse0_acc": "acc/clip_bio_reverse_0",
        "reverse1_acc": "acc/clip_bio_reverse_1",
    }
    out = {field: [float(row[mapping[field]]) for row in rows] for field in FIELDS}
    return {"path": str(path), "steps": [int(row["official_step"]) for row in rows], "values": out}


def load_run(path):
    data = json.loads(path.read_text())
    vals = {field: [record["values"].get(field) for record in data["records"]] for field in FIELDS}
    return data, vals


def main():
    official = load_official()
    runs, values = {}, {}
    for label, path in FILES.items():
        runs[label], values[label] = load_run(path)
    initial_hashes = {label: runs[label]["initial_head_hashes"] for label in runs}
    initial_identical = all(initial_hashes[label] == initial_hashes["R-A"] for label in runs)
    fingerprint_sample = {label: [f["sample_sha256"] for f in runs[label]["fingerprints"]] for label in runs}
    fingerprint_image = {label: [f["image_sha256"] for f in runs[label]["fingerprints"]] for label in runs}
    data_identical = {
        "sample_order": all(fingerprint_sample[label] == fingerprint_sample["R-A"] for label in runs),
        "augmentation": all(fingerprint_image[label] == fingerprint_image["R-A"] for label in runs),
    }
    vs_official = {}
    for label in runs:
        metrics = {field: distance(official["values"][field], values[label][field]) for field in FIELDS}
        metrics["composite_score"] = float(np.mean([metrics[field]["normalized_rmse"] for field in FIELDS]))
        vs_official[label] = metrics
    pairwise = {}
    for i, left in enumerate(runs):
        for right in list(runs)[i + 1:]:
            pairwise[f"{left}_vs_{right}"] = {
                field: distance(values[left][field], values[right][field]) for field in FIELDS
            }
    # Parameter vectors were intentionally not serialized by the completed
    # runs.  Preserve the strongest available update-level proxy and make the
    # limitation explicit rather than treating hashes as distances.
    update_magnitude_pairwise = {}
    for i, left in enumerate(runs):
        for right in list(runs)[i + 1:]:
            per_field = {}
            for group in ("backbone", "classifier", "bio", "reverse0", "reverse1"):
                a = [r["parameter_deltas"][group] for r in runs[left]["records"]]
                b = [r["parameter_deltas"][group] for r in runs[right]["records"]]
                per_field[group] = distance(a, b)
            update_magnitude_pairwise[f"{left}_vs_{right}"] = per_field
    ranking = sorted(((label, vs_official[label]["composite_score"]) for label in runs), key=lambda x: x[1])
    scale = {
        label: {
            "initial": runs[label]["initial_scale"], "final": runs[label]["final_scale"],
            "skipped": runs[label]["skipped_steps"], "successful_updates": runs[label]["successful_updates"],
            "skip_iterations": [r["iteration"] for r in runs[label]["records"] if r["step_skipped"]],
        } for label in runs
    }
    payload = {
        "phase": "2J-F aggregation",
        "official": {"csv": official["path"], "step_range": [official["steps"][0], official["steps"][-1]], "record_count": len(official["steps"])},
        "runs": {label: {"path": str(FILES[label]), "iterations": runs[label]["iterations"], "initial_head_hashes": runs[label]["initial_head_hashes"], "final_head_hashes": runs[label]["final_head_hashes"]} for label in runs},
        "initial_head_hashes_identical": initial_identical,
        "data_fingerprints_identical": data_identical,
        "scale_and_update_summary": scale,
        "vs_official": vs_official,
        "pairwise_trajectory": pairwise,
        "pairwise_update_magnitude_proxy": update_magnitude_pairwise,
        "official_similarity_ranking_low_score_best": [{"run": label, "early_fingerprint_score": score} for label, score in ranking],
        "limitations": [
            "Official rows use the deduplicated first TensorBoard writer (steps 596..795); it is a rank-local stream.",
            "The three repeat scripts did not serialize full final parameter tensors, so exact pairwise parameter L2 distance is unavailable; final hashes and per-step update-magnitude RMSE are reported instead.",
            "Runs are independent fresh processes but use the rank-0/world-size-2 diagnostic sampler without NCCL; this is a repeatability test of one fixed PPU path, not a claim that it reproduces two-rank DDP reduction.",
        ],
    }
    out = ROOT / "diagnostics/phase2j_repeats_aggregate.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"initial_identical": initial_identical, "data_identical": data_identical, "ranking": ranking, "scale": scale, "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
