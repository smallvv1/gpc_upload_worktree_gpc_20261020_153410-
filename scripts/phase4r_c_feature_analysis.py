#!/usr/local/bin/python3
"""Analyze frozen LTCC checkpoint feature banks without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase4r_c_semantic_forensics import (
    OUT_ROOT,
    cross_clothes_centroid_probe,
    cross_clothes_pid_retrieval,
    feature_integrity,
    leakage_pair_metrics,
    load_archive,
    normalize,
    retrieval_metrics,
    sample_pairs,
)


def main():
    root = OUT_ROOT / "checkpoint_features"
    bundle = load_archive()
    records = bundle["records"]
    pids = np.asarray([item["pid_label"] for item in records], dtype=np.int64)
    cloths = np.asarray([item["cloth_key"] for item in records], dtype=object)
    cameras = np.asarray([item["camera"] for item in records], dtype=np.int64)
    pairs = sample_pairs(records, seed=1234, count=100000)
    results = {"phase": "4R-C5/C6/C8", "checkpoints": {}}
    for path in sorted(root.glob("*_feature_bank.npz")):
        name = path.name.replace("_feature_bank.npz", "")
        archive = np.load(path, allow_pickle=True)
        spaces = {
            "global_id_reverse_source": np.asarray(archive["global_feature"], dtype=np.float32),
            "bio_projected": np.asarray(archive["bio_projected"], dtype=np.float32),
        }
        for key in ["reverse_hair_projected", "reverse_clothing_projected"]:
            if archive[key].ndim == 2 and archive[key].shape[1] > 0:
                spaces[key] = np.asarray(archive[key], dtype=np.float32)
        checkpoint_result = {"feature_bank": str(path), "spaces": {}}
        for index, (space_name, features) in enumerate(spaces.items()):
            query = np.arange(len(features))
            if len(query) > 3000:
                query = np.random.default_rng(8000 + index).choice(query, 3000, replace=False)
            cloth_query = np.asarray([i for i in range(len(features)) if np.sum(cloths == cloths[i]) > 1])
            if len(cloth_query) > 3000:
                cloth_query = np.random.default_rng(9000 + index).choice(cloth_query, 3000, replace=False)
            checkpoint_result["spaces"][space_name] = {
                "integrity": feature_integrity(features, 10000 + index),
                "pair_leakage": leakage_pair_metrics(features, pairs),
                "pid_nearest_neighbor": retrieval_metrics(features, pids, query),
                "cross_clothes_pid_nearest_neighbor": cross_clothes_pid_retrieval(features, pids, cloths, 11000 + index),
                "cross_clothes_pid_centroid_probe": cross_clothes_centroid_probe(features, records),
                "clothes_nearest_neighbor": retrieval_metrics(features, cloths, cloth_query),
                "camera_nearest_neighbor": retrieval_metrics(features, cameras, query),
            }
        results["checkpoints"][name] = checkpoint_result

    # Same-image representation agreement is descriptive only; it does not
    # imply that the official and local models share training trajectories.
    names = sorted(results["checkpoints"])
    if "bio_control" in names and "official_full" in names:
        a = np.asarray(np.load(root / "bio_control_feature_bank.npz")["global_feature"], dtype=np.float32)
        b = np.asarray(np.load(root / "official_full_feature_bank.npz")["global_feature"], dtype=np.float32)
        results["same_image_global_feature_agreement"] = {
            "bio_control_vs_official_mean_cosine": float(np.mean(np.sum(normalize(a) * normalize(b), axis=1))),
            "bio_control_vs_official_median_cosine": float(np.median(np.sum(normalize(a) * normalize(b), axis=1))),
        }
    (OUT_ROOT / "checkpoint_feature_leakage.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
