#!/usr/local/bin/python3
"""Print LTCC cloth-changing filter counts for five valid query examples."""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data import build_dataset
from scripts.qualify_amp_initial_scale_phase1 import make_dataset_config

cfg = make_dataset_config("ltcc")
dataset = build_dataset(cfg)
gallery_pids = np.asarray([x["pid"] for x in dataset.gallery])
gallery_cams = np.asarray([x["camid"] for x in dataset.gallery])
gallery_clothes = np.asarray([x["clothes_id"] for x in dataset.gallery])
samples = []
valid_samples = []
for query in dataset.query:
    pid, cam, clothes = query["pid"], query["camid"], query["clothes_id"]
    cross = (gallery_pids == pid) & (gallery_cams != cam)
    same = cross & (gallery_clothes == clothes)
    changed = cross & (gallery_clothes != clothes)
    if changed.sum() > 0:
        valid_samples.append({
            "path": query["image_path"],
            "pid": int(pid),
            "query_clothes_id": int(clothes),
            "legal_changed_clothes_gallery_positives": int(changed.sum()),
            "filtered_same_clothes_positives": int(same.sum()),
            "general_cross_camera_positives": int(cross.sum()),
        })
sample_rng = np.random.default_rng(20250831)
sample_count = min(5, len(valid_samples))
sample_indices = sample_rng.choice(len(valid_samples), size=sample_count, replace=False)
samples = [valid_samples[int(index)] for index in sample_indices]
result = {
    "query_count": len(dataset.query),
    "gallery_count": len(dataset.gallery),
    "sampling": {"seed": 20250831, "eligible_changed_positive_queries": len(valid_samples)},
    "samples": samples,
}
print("LTCC_FILTER_SAMPLE_RESULT", json.dumps(result, sort_keys=True))
(REPO_ROOT / "diagnostics/ltcc_filter_samples.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
