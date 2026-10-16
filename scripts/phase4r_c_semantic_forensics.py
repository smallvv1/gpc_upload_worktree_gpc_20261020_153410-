#!/usr/local/bin/python3
"""PHASE 4R-C: LTCC semantic-signal and identity-leakage forensics.

This module performs frozen-data analysis only.  It never constructs a train
optimizer and never updates an EVA/backbone parameter.  The text archive is
aligned by its metadata keys, matching the released LTCC loader rather than
assuming that NPZ rows are in image-sort order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/data")
TRAIN_ROOT = DATA_ROOT / "LTCC_ReID/train"
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
TEXT_NPZ = TEXT_ROOT / "EVA02-CLIP-B-16/train.npz"
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
OUT_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics"
FACTORS = ["Biometrics", "Hairstyle", "Clothing", "Pose", "Interaction", "Environment"]
NONBIO = {1: "Hair", 2: "Clothing", 3: "Pose"}
STEM_RE = re.compile(r"^(?P<pid>\d+)_(?P<track>\d+)_c(?P<camera>\d+)_(?P<frame>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_train():
    paths = sorted(TRAIN_ROOT.glob("*.png"))
    records = []
    for path in paths:
        match = STEM_RE.fullmatch(path.stem)
        if match is None:
            raise RuntimeError(f"unparseable LTCC train filename: {path.name}")
        item = match.groupdict()
        raw_pid = int(item["pid"])
        track = int(item["track"])
        camera = int(item["camera"])
        records.append({
            "stem": path.stem,
            "filename": path.name,
            "path": str(path),
            "raw_pid": raw_pid,
            "track": track,
            "camera": camera,
            "cloth_key": f"{raw_pid:03d}_{track}",
        })
    raw_pids = sorted({item["raw_pid"] for item in records})
    pid_to_label = {pid: index for index, pid in enumerate(raw_pids)}
    for item in records:
        item["pid_label"] = pid_to_label[item["raw_pid"]]
    return records, pid_to_label


def load_archive():
    records, pid_to_label = parse_train()
    archive = np.load(TEXT_NPZ, allow_pickle=True)
    data = np.asarray(archive["data"])
    metadata = [str(value) for value in archive["metadata"]]
    if data.ndim != 2 or data.shape[1] != 512:
        raise RuntimeError(f"unexpected text data shape: {data.shape}")
    if data.shape[0] != len(metadata) * 6:
        raise RuntimeError(f"text block shape mismatch: {data.shape}, metadata={len(metadata)}")
    if len(metadata) != len(set(metadata)):
        raise RuntimeError("text metadata contains duplicate keys")
    metadata_index = {stem: index for index, stem in enumerate(metadata)}
    missing = [item["stem"] for item in records if item["stem"] not in metadata_index]
    extra = sorted(set(metadata) - {item["stem"] for item in records})
    if missing or extra:
        raise RuntimeError(f"metadata/image set mismatch: missing={len(missing)} extra={len(extra)}")
    order = np.asarray([metadata_index[item["stem"]] for item in records], dtype=np.int64)
    raw_features = data.reshape(len(metadata), 6, 512)[order].astype(np.float32)

    with SUMMARY_JSON.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    summary_key = "ft_EVA02-CLIP-B-16"
    summary_vectors = np.stack([
        np.asarray(summary[str(label)][summary_key], dtype=np.float32).reshape(-1)
        for label in range(len(pid_to_label))
    ])
    if summary_vectors.shape != (len(pid_to_label), 512):
        raise RuntimeError(f"unexpected Summary Bio shape: {summary_vectors.shape}")
    summary_by_image = summary_vectors[np.asarray([item["pid_label"] for item in records])]
    runtime_features = raw_features.copy()
    runtime_features[:, 0] = summary_by_image
    return {
        "records": records,
        "pid_to_label": pid_to_label,
        "metadata": metadata,
        "metadata_index": metadata_index,
        "raw_features": raw_features,
        "runtime_features": runtime_features,
        "summary_vectors": summary_vectors,
        "summary": summary,
    }


def normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def quantiles(values: np.ndarray):
    return {"p01": float(np.quantile(values, 0.01)), "p50": float(np.quantile(values, 0.50)), "p99": float(np.quantile(values, 0.99))}


def nearest_other_cosine(features: np.ndarray, seed: int, sample_count: int = 2000):
    values = normalize(features)
    rng = np.random.default_rng(seed)
    query = rng.choice(len(values), size=min(sample_count, len(values)), replace=False)
    best = []
    for start in range(0, len(query), 256):
        indices = query[start:start + 256]
        scores = values[indices] @ values.T
        for row, index in enumerate(indices):
            scores[row, index] = -np.inf
        best.extend(np.max(scores, axis=1).tolist())
    best = np.asarray(best, dtype=np.float32)
    return {
        "sample_count": int(len(best)),
        "mean": float(best.mean()),
        "std": float(best.std()),
        "quantiles": quantiles(best),
        "fraction_ge_0.999": float(np.mean(best >= 0.999)),
        "fraction_ge_0.9999": float(np.mean(best >= 0.9999)),
    }


def feature_integrity(features: np.ndarray, seed: int):
    values = np.asarray(features, dtype=np.float32)
    finite = np.isfinite(values)
    norms = np.linalg.norm(values, axis=1)
    unique_count = np.unique(values, axis=0).shape[0]
    return {
        "shape": list(values.shape),
        "finite_fraction": float(finite.mean()),
        "nonfinite_value_count": int((~finite).sum()),
        "zero_vector_count": int((norms <= 1e-12).sum()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_quantiles": quantiles(norms),
        "exact_unique_count": int(unique_count),
        "exact_unique_ratio": float(unique_count / len(values)),
        "nearest_other_cosine": nearest_other_cosine(values, seed),
    }


def sample_pairs(records, seed=1234, count=100000):
    rng = np.random.default_rng(seed)
    by_pid = defaultdict(list)
    by_pid_cloth = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(records):
        by_pid[item["pid_label"]].append(index)
        by_pid_cloth[item["pid_label"]][item["cloth_key"]].append(index)
    same_any = []
    same_same_cloth = []
    same_diff_cloth = []
    diff_pid = []
    pids = sorted(by_pid)
    multi_cloth = [pid for pid in pids if len(by_pid_cloth[pid]) >= 2]
    same_cloth_groups = [
        (pid, cloth, indices)
        for pid in pids
        for cloth, indices in by_pid_cloth[pid].items()
        if len(indices) >= 2
    ]
    for _ in range(count):
        pid = int(rng.choice(pids))
        indices = by_pid[pid]
        a, b = rng.choice(indices, size=2, replace=False) if len(indices) >= 2 else (indices[0], indices[0])
        same_any.append((a, b))
        if same_cloth_groups:
            _, _, group = same_cloth_groups[int(rng.integers(len(same_cloth_groups)))]
            if len(group) >= 2:
                same_same_cloth.append(tuple(rng.choice(group, size=2, replace=False)))
        if pid in multi_cloth:
            cloths = list(by_pid_cloth[pid])
            c1, c2 = rng.choice(cloths, size=2, replace=False)
            same_diff_cloth.append((int(rng.choice(by_pid_cloth[pid][c1])), int(rng.choice(by_pid_cloth[pid][c2]))))
        pid_a, pid_b = rng.choice(pids, size=2, replace=False)
        diff_pid.append((int(rng.choice(by_pid[pid_a])), int(rng.choice(by_pid[pid_b]))))
    return {
        "same_pid": np.asarray(same_any, dtype=np.int64),
        "same_pid_same_cloth": np.asarray(same_same_cloth, dtype=np.int64),
        "same_pid_diff_cloth": np.asarray(same_diff_cloth, dtype=np.int64),
        "different_pid": np.asarray(diff_pid, dtype=np.int64),
    }


def pair_cosines(features: np.ndarray, pairs: np.ndarray):
    values = normalize(features)
    if len(pairs) == 0:
        return np.asarray([], dtype=np.float32)
    return np.sum(values[pairs[:, 0]] * values[pairs[:, 1]], axis=1)


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    return float((np.mean(a) - np.mean(b)) / math.sqrt(max(variance, 1e-18)))


def distribution(values):
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "std": float(np.std(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "quantiles": quantiles(values) if len(values) else None,
    }


def leakage_pair_metrics(features: np.ndarray, pairs):
    same = pair_cosines(features, pairs["same_pid"],)
    same_diff = pair_cosines(features, pairs["same_pid_diff_cloth"])
    different = pair_cosines(features, pairs["different_pid"])
    labels = np.r_[np.ones(len(same_diff)), np.zeros(len(different))]
    scores = np.r_[same_diff, different]
    return {
        "same_pid_any_cloth": distribution(same),
        "same_pid_different_cloth": distribution(same_diff),
        "different_pid": distribution(different),
        "same_pid_diff_cloth_minus_different_pid_mean": float(np.mean(same_diff) - np.mean(different)),
        "cohen_d_same_pid_diff_cloth_vs_different_pid": cohen_d(same_diff, different),
        "auc_same_pid_diff_cloth_vs_different_pid": float(roc_auc_score(labels, scores)),
    }


def retrieval_metrics(features: np.ndarray, labels: np.ndarray, query_indices=None, candidate_mask_fn=None, seed=1234):
    values = normalize(features)
    labels = np.asarray(labels)
    if query_indices is None:
        query_indices = np.arange(len(values))
    query_indices = np.asarray(query_indices, dtype=np.int64)
    valid = []
    rank1 = 0
    rank5 = 0
    aps = []
    for start in range(0, len(query_indices), 128):
        qids = query_indices[start:start + 128]
        scores = values[qids] @ values.T
        for local, query in enumerate(qids):
            mask = np.ones(len(values), dtype=bool)
            mask[query] = False
            if candidate_mask_fn is not None:
                mask &= candidate_mask_fn(int(query), labels)
            candidate = np.flatnonzero(mask)
            positive = labels[candidate] == labels[query]
            if not np.any(positive):
                continue
            order = candidate[np.argsort(-scores[local, candidate], kind="stable")]
            rel = (labels[order] == labels[query]).astype(np.int32)
            valid.append(int(query))
            rank1 += int(rel[:1].any())
            rank5 += int(rel[:5].any())
            positive_count = int(rel.sum())
            positions = np.flatnonzero(rel) + 1
            aps.append(float(np.mean(np.cumsum(rel)[positions - 1] / positions)))
    n = len(valid)
    return {
        "query_count": int(n),
        "rank1": float(rank1 / n) if n else None,
        "rank5": float(rank5 / n) if n else None,
        "mAP": float(np.mean(aps)) if aps else None,
        "candidate_rule": "same-label nearest-neighbor retrieval; self excluded",
    }


def cross_clothes_candidate(query: int, labels, pids, cloths):
    return (pids != pids[query]) | (cloths != cloths[query])


def cross_clothes_pid_retrieval(features, pids, cloths, seed=1234):
    valid = np.asarray([index for index in range(len(pids)) if np.sum((pids == pids[index]) & (cloths != cloths[index])) > 0])
    if len(valid) > 3000:
        valid = np.random.default_rng(seed).choice(valid, 3000, replace=False)
    return retrieval_metrics(
        features,
        pids,
        query_indices=valid,
        candidate_mask_fn=lambda q, labels: cross_clothes_candidate(q, labels, pids, cloths),
        seed=seed,
    )


def cross_clothes_centroid_probe(features, records):
    values = normalize(features)
    pids = np.asarray([item["pid_label"] for item in records])
    cloths = np.asarray([item["cloth_key"] for item in records], dtype=object)
    by_pid_cloth = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(records):
        by_pid_cloth[item["pid_label"]][item["cloth_key"]].append(index)
    train = []
    test = []
    eligible = []
    for pid in sorted(by_pid_cloth):
        cloth_keys = sorted(by_pid_cloth[pid])
        if len(cloth_keys) < 2:
            continue
        heldout = cloth_keys[-1]
        eligible.append(pid)
        test.extend(by_pid_cloth[pid][heldout])
        for cloth in cloth_keys[:-1]:
            train.extend(by_pid_cloth[pid][cloth])
    class_ids = np.asarray(sorted(eligible), dtype=np.int64)
    centroids = []
    for pid in class_ids:
        indices = [index for index in train if pids[index] == pid]
        centroids.append(normalize(values[indices].mean(axis=0, keepdims=True))[0])
    centroids = np.asarray(centroids, dtype=np.float32)
    test = np.asarray(test, dtype=np.int64)
    scores = values[test] @ centroids.T
    order = np.argsort(-scores, axis=1, kind="stable")
    truth = np.asarray([np.flatnonzero(class_ids == pids[index])[0] for index in test])
    ranks = np.asarray([int(np.flatnonzero(row == truth[i])[0]) + 1 for i, row in enumerate(order)])
    return {
        "eligible_pid_count": int(len(class_ids)),
        "train_image_count": int(len(train)),
        "heldout_cross_clothes_image_count": int(len(test)),
        "rank1": float(np.mean(ranks <= 1)),
        "rank5": float(np.mean(ranks <= 5)),
        "mAP_single_positive": float(np.mean(1.0 / ranks)),
        "chance_rank1": float(1.0 / len(class_ids)),
        "heldout_rule": "last sorted clothes track per PID; remaining tracks form centroid probe train set",
    }


def structural_coupling(records):
    cloth_to_pids = defaultdict(set)
    pid_to_cloths = defaultdict(set)
    for item in records:
        cloth_to_pids[item["cloth_key"]].add(item["pid_label"])
        pid_to_cloths[item["pid_label"]].add(item["cloth_key"])
    pids_per_cloth = np.asarray([len(value) for value in cloth_to_pids.values()])
    cloths_per_pid = np.asarray([len(value) for value in pid_to_cloths.values()])
    return {
        "pid_count": int(len(pid_to_cloths)),
        "global_cloth_key_count": int(len(cloth_to_pids)),
        "pids_per_cloth_distribution": {str(k): int(v) for k, v in Counter(pids_per_cloth.tolist()).items()},
        "cloths_per_pid_distribution": {str(k): int(v) for k, v in Counter(cloths_per_pid.tolist()).items()},
        "pids_per_cloth_mean": float(pids_per_cloth.mean()),
        "cloths_per_pid_mean": float(cloths_per_pid.mean()),
        "clothes_conditional_pid_purity": float(np.mean(pids_per_cloth == 1)),
        "max_pids_per_cloth": int(pids_per_cloth.max()),
        "min_cloths_per_pid": int(cloths_per_pid.min()),
        "max_cloths_per_pid": int(cloths_per_pid.max()),
        "loader_clothes_parser": r"re.search(r'(\\w+)_c', stem).group(1), yielding PID_track",
    }


def audit_text():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = load_archive()
    records = bundle["records"]
    raw = bundle["raw_features"]
    runtime = bundle["runtime_features"]
    summary = bundle["summary"]
    metadata = bundle["metadata"]
    metadata_index = bundle["metadata_index"]
    rng = np.random.default_rng(1234)
    sample_indices = sorted(rng.choice(len(records), size=1000, replace=False).tolist())
    alignment_sample = []
    for index in sample_indices[:20]:
        item = records[index]
        meta_index = metadata_index[item["stem"]]
        alignment_sample.append({
            "image_index_sorted": index,
            "stem": item["stem"],
            "metadata_index": int(meta_index),
            "block_rows": [int(meta_index * 6 + factor) for factor in range(6)],
            "pid_label": item["pid_label"],
            "cloth_key": item["cloth_key"],
        })
    summary_errors = np.abs(runtime[:, 0] - bundle["summary_vectors"][np.asarray([item["pid_label"] for item in records])])
    source_files = []
    for path in TEXT_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".json", ".csv", ".yaml", ".yml"}:
            source_files.append(str(path))
    audit = {
        "phase": "4R-C",
        "archive": {
            "npz_path": str(TEXT_NPZ),
            "npz_sha256": sha256_file(TEXT_NPZ),
            "summary_json_path": str(SUMMARY_JSON),
            "summary_json_sha256": sha256_file(SUMMARY_JSON),
            "keys": ["data", "metadata"],
            "data_shape": list(np.load(TEXT_NPZ, allow_pickle=True)["data"].shape),
            "metadata_shape": list(np.load(TEXT_NPZ, allow_pickle=True)["metadata"].shape),
            "metadata_unique": len(metadata) == len(set(metadata)),
            "metadata_image_set_equal": set(metadata) == {item["stem"] for item in records},
            "metadata_order_equals_sorted_image_order": metadata == [item["stem"] for item in records],
            "metadata_sorted_position_matches": int(sum(metadata[i] == records[i]["stem"] for i in range(len(records)))),
            "alignment_rule": "metadata key -> 6-row contiguous block; released loader uses metadata.index(stem)",
            "factor_order": FACTORS,
            "sample_count": 1000,
            "sample_alignment": alignment_sample,
        },
        "summary_bio": {
            "pid_count": len(summary),
            "feature_key": "ft_EVA02-CLIP-B-16",
            "feature_shape": list(bundle["summary_vectors"].shape),
            "runtime_replacement_max_abs_error": float(summary_errors.max()),
            "runtime_replacement_mean_abs_error": float(summary_errors.mean()),
            "raw_image_level_bio_shape": list(raw[:, 0].shape),
            "available_text_fields": ["text"],
            "nonbio_raw_text_available": False,
            "source_text_files_under_caption_root": source_files,
        },
        "integrity": {
            factor: feature_integrity(raw[:, index], 1000 + index)
            for index, factor in enumerate(FACTORS)
        },
        "runtime_summary_bio_integrity": feature_integrity(runtime[:, 0], 2000),
        "structural_coupling": structural_coupling(records),
        "representative_summary_bio_text": [
            {"pid_label": int(pid), "text": summary[str(pid)]["text"]}
            for pid in range(min(20, len(summary)))
        ],
        "limitations": [
            "The local archive contains raw NonBio embeddings and metadata but no Hair/Clothing/Pose source strings.",
            "Therefore manual lexical contamination review is possible for Summary Bio only, not for the NonBio captions.",
            "The NPZ metadata is shuffled relative to sorted image filenames; set equality plus key lookup, not row order, is the valid alignment test.",
        ],
    }
    (OUT_ROOT / "text_alignment_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n")
    np.savez_compressed(
        OUT_ROOT / "aligned_text_feature_bank.npz",
        raw_features=raw.astype(np.float16),
        runtime_features=runtime.astype(np.float16),
        summary_vectors=bundle["summary_vectors"].astype(np.float16),
        pid=np.asarray([item["pid_label"] for item in records], dtype=np.int16),
        raw_pid=np.asarray([item["raw_pid"] for item in records], dtype=np.int16),
        camera=np.asarray([item["camera"] for item in records], dtype=np.int16),
        cloth_key=np.asarray([item["cloth_key"] for item in records]),
        stems=np.asarray([item["stem"] for item in records]),
    )
    print(json.dumps({
        "written": str(OUT_ROOT / "text_alignment_audit.json"),
        "metadata_order_equals_sorted": audit["archive"]["metadata_order_equals_sorted_image_order"],
        "alignment_sample_count": audit["archive"]["sample_count"],
        "structural_coupling": audit["structural_coupling"],
    }, indent=2, sort_keys=True), flush=True)


def audit_leakage():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = load_archive()
    records = bundle["records"]
    pids = np.asarray([item["pid_label"] for item in records], dtype=np.int64)
    cloths = np.asarray([item["cloth_key"] for item in records], dtype=object)
    cameras = np.asarray([item["camera"] for item in records], dtype=np.int64)
    pairs = sample_pairs(records, seed=1234, count=100000)
    feature_sources = {
        "Bio_image_level": bundle["raw_features"][:, 0],
        "Bio_summary": bundle["runtime_features"][:, 0],
        "Hair": bundle["raw_features"][:, 1],
        "Clothing": bundle["raw_features"][:, 2],
        "Pose": bundle["raw_features"][:, 3],
    }
    results = {
        "phase": "4R-C2-C7",
        "pair_sample_counts": {key: int(len(value)) for key, value in pairs.items()},
        "factors": {},
    }
    for offset, (name, features) in enumerate(feature_sources.items()):
        pid_query = np.arange(len(features))
        if len(pid_query) > 3000:
            pid_query = np.random.default_rng(3000 + offset).choice(pid_query, 3000, replace=False)
        cloth_query = np.asarray([i for i in range(len(features)) if np.sum(cloths == cloths[i]) > 1])
        if len(cloth_query) > 3000:
            cloth_query = np.random.default_rng(4000 + offset).choice(cloth_query, 3000, replace=False)
        results["factors"][name] = {
            "pair_leakage": leakage_pair_metrics(features, pairs),
            "pid_nearest_neighbor": retrieval_metrics(features, pids, pid_query),
            "clothes_nearest_neighbor": retrieval_metrics(features, cloths, cloth_query),
            "cross_clothes_pid_nearest_neighbor": cross_clothes_pid_retrieval(features, pids, cloths, 5000 + offset),
            "cross_clothes_pid_centroid_probe": cross_clothes_centroid_probe(features, records),
            "camera_nearest_neighbor": retrieval_metrics(features, cameras, pid_query),
        }
        print("LEAKAGE_FACTOR", name, json.dumps(results["factors"][name], sort_keys=True), flush=True)
    results["interpretation_rules"] = {
        "identity_leakage": "same PID different clothes similarity/AUC and cross-clothes PID retrieval",
        "clothes_signal": "within-clothes nearest-neighbor retrieval; label is loader clothes key",
        "cross_clothes_probe": "held-out last track per eligible PID; nearest centroid, no backbone training",
        "chance_cross_clothes_rank1": "approximately 1 / eligible PID count",
    }
    (OUT_ROOT / "identity_leakage_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=json_default) + "\n")
    print("WRITTEN", OUT_ROOT / "identity_leakage_metrics.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text-audit", "leakage"], required=True)
    args = parser.parse_args()
    if args.mode == "text-audit":
        audit_text()
    else:
        audit_leakage()


if __name__ == "__main__":
    main()
