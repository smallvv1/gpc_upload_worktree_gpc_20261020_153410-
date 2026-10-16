#!/usr/bin/env python3
"""Phase 5A: offline LTCC clothing-hardness feasibility analysis.

This script deliberately does not train, evaluate retrieval, touch the sampler,
or load a model onto a GPU.  It consumes the already audited LTCC Bio E27
feature bank and the already audited DIFFER text/branch feature banks.  The
historical V4 DS trace is replayed counterfactually with a fixed event schedule
so that the analysis answers whether a proposed score selects harder, reliable
clothing states.

The primary identity score and primary Clothing score are frozen by the Phase
5A protocol:

  H_ID(i) = d(f_i, C_yi) * d(f_i, C_qi)
  H_cloth(y,c) = mean distance from the (y,c) center to the other clothing
                 centers of the same PID (leave-one-out, all-other-clothes)

The DS-style nearest/mean distances to the currently active clothing states
are also recorded.  The requested product ``H_ID(new)*H_ID(original)`` is
retained in replay outputs, but directional selection improvement is assessed
with ``delta_H_ID = H_ID(new)-H_ID(original)`` because the product is
non-negative and cannot indicate which selection is larger.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery"
FORENSICS_ROOT = PHASE4_ROOT / "C_semantic_forensics"
TEXT_BANK_PATH = FORENSICS_ROOT / "aligned_text_feature_bank.npz"
BIO_BANK_PATH = FORENSICS_ROOT / "checkpoint_features/bio_control_feature_bank.npz"
ORIGINAL_FULL_BANK_PATH = FORENSICS_ROOT / "checkpoint_features/original_full_feature_bank.npz"
OFFICIAL_FULL_BANK_PATH = FORENSICS_ROOT / "checkpoint_features/official_full_feature_bank.npz"
TEXT_AUDIT_PATH = FORENSICS_ROOT / "text_alignment_audit.json"
CHECKPOINT_MANIFEST_PATH = FORENSICS_ROOT / "checkpoint_features/checkpoint_feature_manifest.json"
TEXT_RESOURCE_PATH = Path("/home/data/DIFFER_TextCaption/LTCC_ReID/EVA02-CLIP-B-16/train.npz")
TRAIN_ROOT = Path("/home/data/LTCC_ReID/train")
HISTORICAL_TRACE_PATH = Path("/home/CL-CC-ReID/results/ltcc_phase4_v4_ds/curriculum_trace.csv")
HISTORICAL_REPORT_PATH = Path("/home/CL-CC-ReID/diagnostics/PHASE_6_REPORT.md")
HISTORICAL_PIPELINE_PATH = Path("/home/CL-CC-ReID/diagnostics/ds_pipeline_trace.md")

OUT_ROOT = ROOT / "outputs/new_method_track/phase5a_clothing_hardness"
REPORT_PATH = ROOT / "reports/PHASE_5A_CLOTHING_HARDNESS_FEASIBILITY.md"

SEED = 20260910
TEXT_FACTOR_INDEX = 2
TEXT_FACTOR_NAME = "Clothing"
EXPECTED_IMAGES = 9576
EXPECTED_PIDS = 77
EXPECTED_CLOTHES = 256
PAIR_CAP_SAME_PER_GROUP = 250
PAIR_CAP_DIFFERENT_PER_PID = 250
REPRESENTATIVE_CASES_PER_QUADRANT = 20
BOOTSTRAP_REPLICATES = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def json_safe(value):
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_or_none(value)
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows, fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise RuntimeError("feature bank contains invalid or zero vectors")
    return values / norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    value = 1.0 - float(np.dot(a, b))
    return float(np.clip(value, 0.0, 2.0))


def distance_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    values = 1.0 - np.sum(a * b, axis=1)
    return np.clip(values, 0.0, 2.0).astype(np.float64)


def normalize_for_json_list(values) -> str:
    return json.dumps(json_safe(list(values)), separators=(",", ":"), sort_keys=False)


def parse_json_or_python(value, default):
    if value is None or str(value).strip() in ("", "None", "null"):
        return default
    text = str(value).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return default


def parse_int_list(value) -> list[int]:
    parsed = parse_json_or_python(value, [])
    if isinstance(parsed, (int, float)):
        return [int(parsed)]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [int(item) for item in parsed]


def parse_int_scalar(value) -> int | None:
    parsed = parse_json_or_python(value, None)
    if parsed is None:
        return None
    if isinstance(parsed, (list, tuple)):
        if len(parsed) != 1:
            return None
        parsed = parsed[0]
    return int(parsed)


def safe_corr(kind: str, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    if kind == "pearson":
        result = pearsonr(x, y)
    elif kind == "spearman":
        result = spearmanr(x, y)
    elif kind == "kendall":
        result = kendalltau(x, y)
    else:
        raise ValueError(kind)
    return finite_or_none(result.statistic if hasattr(result, "statistic") else result[0])


def summary_stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def align_bank(path: Path, canonical_stems: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    bank = load_npz(path)
    stems = np.asarray(bank["stems"]).astype(str)
    if len(stems) != len(set(stems.tolist())):
        raise RuntimeError(f"duplicate stems in {path}")
    canonical_stems = np.asarray(canonical_stems).astype(str)
    if set(stems.tolist()) != set(canonical_stems.tolist()):
        raise RuntimeError(f"stem set mismatch for {path}")
    exact_order = bool(np.array_equal(stems, canonical_stems))
    index = {stem: i for i, stem in enumerate(stems.tolist())}
    order = np.asarray([index[stem] for stem in canonical_stems.tolist()], dtype=np.int64)
    aligned = {}
    for key, values in bank.items():
        if key == "stems":
            aligned[key] = canonical_stems.copy()
        elif isinstance(values, np.ndarray) and values.ndim >= 1 and values.shape[0] == len(stems):
            aligned[key] = values[order]
        else:
            aligned[key] = values
    return aligned, {
        "path": str(path),
        "file_sha256": sha256(path),
        "image_count": int(len(stems)),
        "order_exact_before_alignment": exact_order,
        "aligned_to": "canonical text-bank stem order",
        "keys": sorted(bank),
    }


def verify_canonical_alignment(text: dict, bio: dict, original: dict, official: dict) -> dict:
    expected = {"images": EXPECTED_IMAGES, "pids": EXPECTED_PIDS, "clothes": EXPECTED_CLOTHES}
    if len(text["stems"]) != EXPECTED_IMAGES:
        raise RuntimeError(f"unexpected image count: {len(text['stems'])}")
    if len(np.unique(text["pid"])) != EXPECTED_PIDS:
        raise RuntimeError("unexpected PID count")
    if len(np.unique(bio["clothes_id"])) != EXPECTED_CLOTHES:
        raise RuntimeError("unexpected clothing-state count")
    checks = {
        "expected_counts": expected,
        "text_bio_stems_exact": bool(np.array_equal(text["stems"], bio["stems"])),
        "text_original_stems_exact": bool(np.array_equal(text["stems"], original["stems"])),
        "text_official_stems_exact": bool(np.array_equal(text["stems"], official["stems"])),
        "text_bio_pid_exact": bool(np.array_equal(text["pid"], bio["pid"])),
        "text_original_pid_exact": bool(np.array_equal(text["pid"], original["pid"])),
        "text_official_pid_exact": bool(np.array_equal(text["pid"], official["pid"])),
        "original_official_clothes_id_exact": bool(np.array_equal(original["clothes_id"], official["clothes_id"])),
        "bio_full_clothes_id_exact": bool(np.array_equal(bio["clothes_id"], official["clothes_id"])),
        "text_cloth_key_unique_groups": int(len(np.unique(text["cloth_key"]))),
        "camera_difference_not_used": bool(not np.array_equal(text["camera"], bio["camera"])),
        "camera_note": "Text-bank camera labels are one-based while checkpoint-bank camera labels are zero-based; camera is not used in Phase 5A scores.",
    }
    required = [key for key, value in checks.items() if key.endswith("_exact") and not value]
    if required:
        raise RuntimeError(f"canonical alignment failed: {required}")
    return checks


def build_groups(text: dict, bio: dict) -> dict[tuple[int, int], dict]:
    groups = defaultdict(list)
    for index, (pid, cloth_id) in enumerate(zip(bio["pid"], bio["clothes_id"])):
        groups[(int(pid), int(cloth_id))].append(int(index))
    result = {}
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        stem = str(text["stems"][indices[0]])
        cloth_keys = sorted(set(str(item) for item in text["cloth_key"][indices].tolist()))
        if len(cloth_keys) != 1:
            raise RuntimeError(f"clothing key is not pure for {key}: {cloth_keys}")
        result[key] = {
            "pid": int(key[0]),
            "cloth_id": int(key[1]),
            "cloth_key": cloth_keys[0],
            "indices": indices,
            "stems": [str(item) for item in text["stems"][indices].tolist()],
            "image_filenames": [f"{item}.png" for item in text["stems"][indices].tolist()],
            "image_paths": [str(TRAIN_ROOT / f"{item}.png") for item in text["stems"][indices].tolist()],
        }
    if len(result) != EXPECTED_CLOTHES:
        raise RuntimeError(f"unexpected group count: {len(result)}")
    return result


def make_sources(text: dict, bio: dict, original: dict, official: dict, manifest: dict) -> dict:
    runs = manifest.get("runs", {})
    bio_run = runs.get("bio_control", {})
    original_run = runs.get("original_full", {})
    official_run = runs.get("official_full", {})
    return {
        "clothing_text_fixed": {
            "role": "primary clothing semantic representation",
            "raw": np.asarray(text["raw_features"][:, TEXT_FACTOR_INDEX, :], dtype=np.float32),
            "source_file": str(TEXT_RESOURCE_PATH),
            "source_file_sha256": sha256(TEXT_RESOURCE_PATH),
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "layer": "aligned_text_feature_bank.raw_features[:, factor_index=2, :]",
            "factor_index": TEXT_FACTOR_INDEX,
            "factor_name": TEXT_FACTOR_NAME,
            "dimension": int(text["raw_features"].shape[-1]),
            "normalization": "L2 per image; group center is mean of normalized image vectors then L2",
            "bank_file": str(TEXT_BANK_PATH),
            "bank_file_sha256": sha256(TEXT_BANK_PATH),
        },
        "clothing_original_full_failed": {
            "role": "supplementary trained Clothing branch from failed local Original Full",
            "raw": np.asarray(original["reverse_clothing_projected"], dtype=np.float32),
            "source_file": str(original_run.get("feature_bank", ORIGINAL_FULL_BANK_PATH)),
            "source_file_sha256": sha256(ORIGINAL_FULL_BANK_PATH),
            "source_checkpoint": original_run.get("checkpoint"),
            "source_checkpoint_sha256": original_run.get("checkpoint_sha256"),
            "layer": "head_clip_bioReverse[clothing].head_fc(feat_bio)",
            "factor_index": None,
            "factor_name": "Clothing reverse branch",
            "dimension": int(original["reverse_clothing_projected"].shape[-1]),
            "normalization": "L2 per image; group center is mean of normalized image vectors then L2",
            "bank_file": str(ORIGINAL_FULL_BANK_PATH),
            "bank_file_sha256": sha256(ORIGINAL_FULL_BANK_PATH),
        },
        "clothing_official_full": {
            "role": "supplementary trained Clothing branch from official Full checkpoint",
            "raw": np.asarray(official["reverse_clothing_projected"], dtype=np.float32),
            "source_file": str(official_run.get("feature_bank", OFFICIAL_FULL_BANK_PATH)),
            "source_file_sha256": sha256(OFFICIAL_FULL_BANK_PATH),
            "source_checkpoint": official_run.get("checkpoint"),
            "source_checkpoint_sha256": official_run.get("checkpoint_sha256"),
            "layer": "head_clip_bioReverse[clothing].head_fc(feat_bio)",
            "factor_index": None,
            "factor_name": "Clothing reverse branch",
            "dimension": int(official["reverse_clothing_projected"].shape[-1]),
            "normalization": "L2 per image; group center is mean of normalized image vectors then L2",
            "bank_file": str(OFFICIAL_FULL_BANK_PATH),
            "bank_file_sha256": sha256(OFFICIAL_FULL_BANK_PATH),
            "official_reverse_bias_adapters": official_run.get("official_reverse_bias_adapters", 0),
        },
        "eva_global": {
            "role": "EVA/global identity baseline",
            "raw": np.asarray(bio["global_feature"], dtype=np.float32),
            "source_file": str(bio_run.get("feature_bank", BIO_BANK_PATH)),
            "source_file_sha256": sha256(BIO_BANK_PATH),
            "source_checkpoint": bio_run.get("checkpoint"),
            "source_checkpoint_sha256": bio_run.get("checkpoint_sha256"),
            "layer": "Bio E27 model eval output feat_bio after EVA head_bio and pooling",
            "factor_index": None,
            "factor_name": None,
            "dimension": int(bio["global_feature"].shape[-1]),
            "normalization": "L2 per image for analysis; group center is mean of normalized image vectors then L2",
            "bank_file": str(BIO_BANK_PATH),
            "bank_file_sha256": sha256(BIO_BANK_PATH),
        },
        "bio_projected": {
            "role": "Bio identity baseline",
            "raw": np.asarray(bio["bio_projected"], dtype=np.float32),
            "source_file": str(bio_run.get("feature_bank", BIO_BANK_PATH)),
            "source_file_sha256": sha256(BIO_BANK_PATH),
            "source_checkpoint": bio_run.get("checkpoint"),
            "source_checkpoint_sha256": bio_run.get("checkpoint_sha256"),
            "layer": "head_clip_bio(feat_bio)",
            "factor_index": None,
            "factor_name": "Bio",
            "dimension": int(bio["bio_projected"].shape[-1]),
            "normalization": "L2 per image for analysis; group center is mean of normalized image vectors then L2",
            "bank_file": str(BIO_BANK_PATH),
            "bank_file_sha256": sha256(BIO_BANK_PATH),
        },
    }


def prepare_sources(sources: dict) -> None:
    for source in sources.values():
        source["raw_norm"] = np.linalg.norm(source["raw"], axis=1).astype(np.float64)
        source["features"] = l2_normalize(source["raw"])
        if source["features"].shape[1] != source["dimension"]:
            raise RuntimeError(f"dimension mismatch in {source['role']}")
        if not np.isfinite(source["features"]).all():
            raise RuntimeError(f"non-finite normalized source: {source['role']}")


def compute_identity_damage(f_global: np.ndarray, pids: np.ndarray) -> dict:
    f_global = l2_normalize(f_global)
    pids = np.asarray(pids, dtype=np.int64)
    unique_pids = sorted(int(item) for item in np.unique(pids))
    pid_to_center_index = {pid: index for index, pid in enumerate(unique_pids)}
    centers = []
    for pid in unique_pids:
        center = np.mean(f_global[pids == pid], axis=0)
        center = center / max(float(np.linalg.norm(center)), 1e-12)
        centers.append(center)
    centers = np.asarray(centers, dtype=np.float32)
    similarity = np.asarray(f_global @ centers.T, dtype=np.float64)
    own_indices = np.asarray([pid_to_center_index[int(pid)] for pid in pids], dtype=np.int64)
    own_similarity = similarity[np.arange(len(pids)), own_indices]
    competing = similarity.copy()
    competing[np.arange(len(pids)), own_indices] = -np.inf
    nearest_negative_index = np.argmax(competing, axis=1)
    nearest_negative_pid = np.asarray([unique_pids[index] for index in nearest_negative_index], dtype=np.int64)
    negative_similarity = competing[np.arange(len(pids)), nearest_negative_index]
    own_distance = np.clip(1.0 - own_similarity, 0.0, 2.0)
    negative_distance = np.clip(1.0 - negative_similarity, 0.0, 2.0)
    h_id = own_distance * negative_distance
    own_euclidean = np.sqrt(np.maximum(2.0 - 2.0 * own_similarity, 0.0))
    negative_euclidean = np.sqrt(np.maximum(2.0 - 2.0 * negative_similarity, 0.0))
    h_id_euclidean = own_euclidean * negative_euclidean
    return {
        "features": f_global,
        "pids": pids,
        "unique_pids": unique_pids,
        "centers": centers,
        "own_distance": own_distance,
        "nearest_negative_distance": negative_distance,
        "nearest_negative_pid": nearest_negative_pid,
        "h_id": h_id,
        "own_euclidean": own_euclidean,
        "nearest_negative_euclidean": negative_euclidean,
        "h_id_euclidean": h_id_euclidean,
        "definition": "H_ID = cosine_distance(f_i,C_y) * cosine_distance(f_i,C_qi), q_i is nearest competing identity center",
        "supplementary_definition": "H_ID_euclidean = Euclidean_distance(f_i,C_y) * Euclidean_distance(f_i,C_qi) on normalized features",
        "operational_direction": "larger H_ID is treated as harder for replay, per the Phase 5A protocol",
    }


def build_group_centers(groups: dict, features: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    centers = {}
    for key, group in groups.items():
        center = np.mean(features[group["indices"]], axis=0)
        center = center / max(float(np.linalg.norm(center)), 1e-12)
        centers[key] = center.astype(np.float32)
    return centers


def add_reliability_metrics(groups: dict, group_centers: dict, text_source: dict, identity: dict, global_source: dict) -> dict:
    max_support = max(len(group["indices"]) for group in groups.values())
    global_norm_median = float(np.median(global_source["raw_norm"]))
    group_rows = {}
    for key, group in groups.items():
        indices = group["indices"]
        pid = group["pid"]
        identity_center = identity["centers"][identity["unique_pids"].index(pid)]
        text_center = group_centers[key]
        r_consistency = float(np.mean(text_source["features"][indices] @ text_center))
        r_bio = float(np.mean(identity["features"][indices] @ identity_center))
        support = len(indices)
        r_support = float(np.log1p(support) / np.log1p(max_support))
        group_global_norm = float(np.mean(global_source["raw_norm"][indices]))
        norm_anomaly = float(abs(np.log(max(group_global_norm, 1e-12) / max(global_norm_median, 1e-12))))
        norm_reliability = float(np.exp(-norm_anomaly))
        r_consistency_norm = float(np.clip((r_consistency + 1.0) / 2.0, 0.0, 1.0))
        r_bio_norm = float(np.clip((r_bio + 1.0) / 2.0, 0.0, 1.0))
        combined = float(np.mean([r_consistency_norm, r_bio_norm, r_support]))
        group_rows[key] = {
            "r_consistency": r_consistency,
            "r_bio": r_bio,
            "r_support": r_support,
            "feature_norm_mean": group_global_norm,
            "feature_norm_anomaly": norm_anomaly,
            "feature_norm_reliability": norm_reliability,
            "reliability": combined,
            "reliability_definition": "mean(normalized within-clothes consistency, normalized Bio consistency, log-saturated support)",
        }
    return {
        "groups": group_rows,
        "max_support": max_support,
        "global_feature_norm_median": global_norm_median,
        "combined_formula": "R = mean((R_consistency+1)/2, (R_bio+1)/2, log1p(support)/log1p(max_support)); weights are equal and not tuned",
    }


def compute_clothing_scores(groups: dict, centers: dict) -> tuple[dict, dict]:
    by_pid = defaultdict(list)
    for key in sorted(groups):
        by_pid[key[0]].append(key)
    scores = {}
    for key in sorted(groups):
        others = [other for other in by_pid[key[0]] if other != key]
        if not others:
            scores[key] = {"h_cloth_loo_mean": None, "h_cloth_loo_nearest": None, "other_count": 0}
            continue
        distances = np.asarray([cosine_distance(centers[key], centers[other]) for other in others], dtype=np.float64)
        scores[key] = {
            "h_cloth_loo_mean": float(np.mean(distances)),
            "h_cloth_loo_nearest": float(np.min(distances)),
            "other_count": int(len(others)),
        }
    return scores, by_pid


def make_group_rows(groups: dict, text: dict, identity: dict, all_source_scores: dict, reliability: dict) -> list[dict]:
    pid_center_index = {pid: index for index, pid in enumerate(identity["unique_pids"])}
    h_id = identity["h_id"]
    h_id_euclidean = identity["h_id_euclidean"]
    rows = []
    for key in sorted(groups):
        group = groups[key]
        indices = group["indices"]
        row = {
            "pid": group["pid"],
            "cloth_id": group["cloth_id"],
            "cloth_key": group["cloth_key"],
            "support": int(len(indices)),
            "stems_json": normalize_for_json_list(group["stems"]),
            "image_filenames_json": normalize_for_json_list(group["image_filenames"]),
            "image_paths_json": normalize_for_json_list(group["image_paths"]),
            "h_id_mean": float(np.mean(h_id[indices])),
            "h_id_median": float(np.median(h_id[indices])),
            "h_id_p75": float(np.quantile(h_id[indices], 0.75)),
            "h_id_max": float(np.max(h_id[indices])),
            "h_id_euclidean_mean": float(np.mean(h_id_euclidean[indices])),
            "own_center_distance_mean": float(np.mean(identity["own_distance"][indices])),
            "nearest_negative_distance_mean": float(np.mean(identity["nearest_negative_distance"][indices])),
            "nearest_negative_pid_mode": int(np.bincount(identity["nearest_negative_pid"][indices]).argmax()),
            "r_consistency": reliability["groups"][key]["r_consistency"],
            "r_bio": reliability["groups"][key]["r_bio"],
            "r_support": reliability["groups"][key]["r_support"],
            "feature_norm_mean": reliability["groups"][key]["feature_norm_mean"],
            "feature_norm_anomaly": reliability["groups"][key]["feature_norm_anomaly"],
            "feature_norm_reliability": reliability["groups"][key]["feature_norm_reliability"],
            "reliability": reliability["groups"][key]["reliability"],
        }
        for source_name, source_scores in all_source_scores.items():
            score = source_scores[key]
            row[f"{source_name}_h_cloth_loo_mean"] = score["h_cloth_loo_mean"]
            row[f"{source_name}_h_cloth_loo_nearest"] = score["h_cloth_loo_nearest"]
            row[f"{source_name}_other_count"] = score["other_count"]
        rows.append(row)
    return rows


def sample_pairs(groups: dict, seed: int) -> list[tuple[int, int, int, int, int]]:
    rng = np.random.default_rng(seed)
    by_pid = defaultdict(list)
    for key in sorted(groups):
        by_pid[key[0]].append(key)
    same = []
    different = []
    for key in sorted(groups):
        indices = groups[key]["indices"].tolist()
        candidates = list(itertools.combinations(indices, 2))
        if len(candidates) > PAIR_CAP_SAME_PER_GROUP:
            selected = rng.choice(len(candidates), PAIR_CAP_SAME_PER_GROUP, replace=False)
            candidates = [candidates[int(index)] for index in sorted(selected.tolist())]
        same.extend((int(a), int(b), 0, key[0], key[1]) for a, b in candidates)
    for pid in sorted(by_pid):
        candidates = []
        for key_a, key_b in itertools.combinations(by_pid[pid], 2):
            for ia in groups[key_a]["indices"]:
                for ib in groups[key_b]["indices"]:
                    candidates.append((int(ia), int(ib), int(key_a[1]), int(key_b[1])))
        if len(candidates) > PAIR_CAP_DIFFERENT_PER_PID:
            selected = rng.choice(len(candidates), PAIR_CAP_DIFFERENT_PER_PID, replace=False)
            candidates = [candidates[int(index)] for index in sorted(selected.tolist())]
        different.extend((a, b, 1, pid, -1) for a, b, _cid_a, _cid_b in candidates)
    count = min(len(same), len(different))
    if count == 0:
        raise RuntimeError("unable to form balanced same/different clothing pairs")
    return same[:count] + different[:count]


def pair_diagnostics(pairs: list[tuple[int, int, int, int, int]], groups: dict, text: dict, sources: dict, identity: dict) -> tuple[list[dict], dict]:
    rows = []
    for ia, ib, different_flag, pid, _ in pairs:
        row = {
            "index_a": ia,
            "index_b": ib,
            "pid": pid,
            "cloth_id_a": groups[(int(text["pid"][ia]), int(text["clothes_id"][ia]))]["cloth_id"] if "clothes_id" in text else None,
            "cloth_id_b": groups[(int(text["pid"][ib]), int(text["clothes_id"][ib]))]["cloth_id"] if "clothes_id" in text else None,
            "stem_a": str(text["stems"][ia]),
            "stem_b": str(text["stems"][ib]),
            "different_clothes": different_flag,
            "h_id_pair_mean": float(np.mean([identity["h_id"][ia], identity["h_id"][ib]])),
        }
        for source_name, source in sources.items():
            row[f"{source_name}_distance"] = cosine_distance(source["features"][ia], source["features"][ib])
        rows.append(row)

    labels = np.asarray([row["different_clothes"] for row in rows], dtype=np.int64)
    source_summary = {}
    for source_name in sources:
        values = np.asarray([row[f"{source_name}_distance"] for row in rows], dtype=np.float64)
        same = values[labels == 0]
        different = values[labels == 1]
        try:
            auc = float(roc_auc_score(labels, values))
        except ValueError:
            auc = None
        # ROC-AUC is exactly the probability that a random different-clothes
        # pair receives a larger distance than a random same-clothes pair
        # (with half credit for ties), so it is the unbiased pair-ranking
        # accuracy over all cross-class pairings.
        ranking = auc
        source_summary[source_name] = {
            "same_clothes": summary_stats(same),
            "different_clothes": summary_stats(different),
            "roc_auc_different_vs_same": auc,
            "pairwise_ranking_accuracy": ranking,
            "cohen_d": cohen_d(same, different),
            "binary_same_different_label_baseline_auc": 0.5,
            "binary_baseline_note": "A constant same/different label cannot rank two distinct different outfits; its AUC is fixed at 0.5 for this hardness task.",
        }
    return rows, {
        "seed": SEED,
        "pair_count_total": len(rows),
        "pair_count_per_class": int(np.sum(labels == 0)),
        "pair_sampling": {
            "same_per_group_cap": PAIR_CAP_SAME_PER_GROUP,
            "different_per_pid_cap": PAIR_CAP_DIFFERENT_PER_PID,
            "same_vs_different": "balanced prefix after deterministic seeded sampling",
        },
        "sources": source_summary,
    }


def cohen_d(first, second) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) == 0 or len(second) == 0:
        return None
    variance = max((np.var(first, ddof=1) + np.var(second, ddof=1)) / 2.0, 1e-24)
    return float((np.mean(second) - np.mean(first)) / math.sqrt(variance))


def group_pair_diagnostics(groups: dict, all_source_centers: dict, group_rows_by_key: dict) -> list[dict]:
    by_pid = defaultdict(list)
    for key in sorted(groups):
        by_pid[key[0]].append(key)
    rows = []
    for pid in sorted(by_pid):
        for key_a, key_b in itertools.combinations(by_pid[pid], 2):
            row = {
                "pid": pid,
                "cloth_id_a": key_a[1],
                "cloth_id_b": key_b[1],
                "cloth_key_a": groups[key_a]["cloth_key"],
                "cloth_key_b": groups[key_b]["cloth_key"],
                "support_a": len(groups[key_a]["indices"]),
                "support_b": len(groups[key_b]["indices"]),
                "h_id_pair_mean": float(np.mean([group_rows_by_key[key_a]["h_id_mean"], group_rows_by_key[key_b]["h_id_mean"]])),
            }
            for source_name, centers in all_source_centers.items():
                row[f"{source_name}_h_cloth_pair"] = cosine_distance(centers[key_a], centers[key_b])
            rows.append(row)
    return rows


def clothing_identity_leakage(groups: dict, centers: dict) -> dict:
    by_pid = defaultdict(list)
    for key in sorted(groups):
        by_pid[key[0]].append(key)
    same_pid = []
    nearest_cross = []
    for pid in sorted(by_pid):
        for key_a, key_b in itertools.combinations(by_pid[pid], 2):
            same_pid.append(cosine_distance(centers[key_a], centers[key_b]))
        cross_keys = [key for key in sorted(groups) if key[0] != pid]
        for key in by_pid[pid]:
            nearest_cross.append(min(cosine_distance(centers[key], centers[other]) for other in cross_keys))
    ratio = float(np.median(same_pid) / max(np.median(nearest_cross), 1e-12)) if same_pid else None
    if ratio is None:
        level = "UNAVAILABLE"
    elif ratio >= 0.9:
        level = "LOW"
    elif ratio >= 0.7:
        level = "MODERATE"
    else:
        level = "HIGH"
    return {
        "same_pid_different_outfit_count": len(same_pid),
        "nearest_cross_pid_count": len(nearest_cross),
        "same_pid_different_outfit": summary_stats(same_pid),
        "nearest_cross_pid": summary_stats(nearest_cross),
        "median_ratio_same_pid_over_nearest_cross": ratio,
        "identity_leakage_level": level,
        "interpretation": "This is a prototype-level leakage diagnostic, not an identity classifier.",
    }


def source_group_analysis(group_rows: list[dict], groups: dict, source_scores: dict, source_name: str) -> dict:
    h_id_values = {key: float(row["h_id_mean"]) for key, row in zip(sorted(groups), group_rows)}
    cloth_values = {key: source_scores[key]["h_cloth_loo_mean"] for key in sorted(groups)}
    eligible_keys = [key for key in sorted(groups) if cloth_values[key] is not None and math.isfinite(float(cloth_values[key]))]
    x = np.asarray([cloth_values[key] for key in eligible_keys], dtype=np.float64)
    y = np.asarray([h_id_values[key] for key in eligible_keys], dtype=np.float64)
    per_pid = []
    top1 = []
    top2 = []
    for pid in sorted({key[0] for key in eligible_keys}):
        keys = [key for key in eligible_keys if key[0] == pid]
        if len(keys) < 3:
            continue
        cloth_order = sorted(keys, key=lambda key: (-float(cloth_values[key]), key[1]))
        id_order = sorted(keys, key=lambda key: (-float(h_id_values[key]), key[1]))
        rho = safe_corr("spearman", [cloth_values[key] for key in keys], [h_id_values[key] for key in keys])
        tau = safe_corr("kendall", [cloth_values[key] for key in keys], [h_id_values[key] for key in keys])
        person = safe_corr("pearson", [cloth_values[key] for key in keys], [h_id_values[key] for key in keys])
        per_pid.append({"pid": pid, "outfit_count": len(keys), "spearman": rho, "kendall": tau, "pearson": person})
        top1.append(float(cloth_order[0] == id_order[0]))
        top2.append(len(set(cloth_order[:2]).intersection(id_order[:2])) / 2.0)
    per_pid_rho = np.asarray([item["spearman"] for item in per_pid if item["spearman"] is not None], dtype=np.float64)
    per_pid_tau = np.asarray([item["kendall"] for item in per_pid if item["kendall"] is not None], dtype=np.float64)
    # Top/bottom quartiles use exact rank counts, avoiding threshold ties.
    order = np.argsort(y, kind="stable")
    n_quartile = max(1, int(math.floor(len(order) * 0.25)))
    bottom = order[:n_quartile]
    top = order[-n_quartile:]
    task_indices = np.concatenate([bottom, top])
    task_labels = np.concatenate([np.zeros(len(bottom), dtype=np.int64), np.ones(len(top), dtype=np.int64)])
    cloth_task_scores = x[task_indices]
    try:
        auc = float(roc_auc_score(task_labels, cloth_task_scores))
        pr_auc = float(average_precision_score(task_labels, cloth_task_scores))
    except ValueError:
        auc, pr_auc = None, None
    return {
        "source": source_name,
        "eligible_group_count": len(eligible_keys),
        "eligible_pid_count": len({key[0] for key in eligible_keys}),
        "h_cloth_distribution": summary_stats(x),
        "h_id_distribution_on_eligible": summary_stats(y),
        "global_pearson": safe_corr("pearson", x, y),
        "global_spearman": safe_corr("spearman", x, y),
        "per_pid_count_minimum": 3,
        "per_pid_valid_count": len(per_pid),
        "per_pid_spearman_mean": finite_or_none(np.mean(per_pid_rho)) if len(per_pid_rho) else None,
        "per_pid_spearman_median": finite_or_none(np.median(per_pid_rho)) if len(per_pid_rho) else None,
        "per_pid_spearman_positive_rate": finite_or_none(np.mean(per_pid_rho > 0.0)) if len(per_pid_rho) else None,
        "per_pid_kendall_mean": finite_or_none(np.mean(per_pid_tau)) if len(per_pid_tau) else None,
        "per_pid_details": per_pid,
        "top1_agreement_rate": finite_or_none(np.mean(top1)) if top1 else None,
        "top2_overlap_mean": finite_or_none(np.mean(top2)) if top2 else None,
        "topk_pid_count": len(top1),
        "top_bottom_definition": "exact bottom/top floor(25% of eligible groups) by H_ID; middle groups excluded",
        "top_bottom_group_count_each": n_quartile,
        "predictive_roc_auc": auc,
        "predictive_pr_auc": pr_auc,
        "predictive_spearman_all_eligible": safe_corr("spearman", x, y),
        "binary_clothes_label_baseline_roc_auc": 0.5,
        "binary_clothes_label_baseline_pr_auc": 0.5,
    }


def ds_candidate_analysis(candidate_rows: list[dict]) -> dict:
    """Correlate C_seen-conditioned DS scores with candidate H_ID."""
    result = {
        "candidate_event_count": len(candidate_rows),
        "event_count": len({int(row["event_id"]) for row in candidate_rows}),
        "pid_count": len({int(row["pid"]) for row in candidate_rows}),
        "scores": {},
        "definition": "For each historical expansion event, active_original is C_seen; every remaining candidate receives nearest/mean cosine distance to C_seen clothing centers.",
        "original_ds_score_note": "original_ds_score_v4 is copied from the historical V4 trace; it is not recomputed from the current Bio E27 bank.",
    }
    for score_name in ["original_ds_score_v4", "h_cloth_ds_nearest", "h_cloth_ds_mean"]:
        valid = [row for row in candidate_rows if row.get(score_name) is not None]
        x = np.asarray([float(row[score_name]) for row in valid], dtype=np.float64)
        y = np.asarray([float(row["candidate_h_id"]) for row in valid], dtype=np.float64)
        per_pid = []
        for pid in sorted({int(row["pid"]) for row in valid}):
            pid_rows = [row for row in valid if int(row["pid"]) == pid]
            if len(pid_rows) < 3:
                continue
            rho = safe_corr("spearman", [row[score_name] for row in pid_rows], [row["candidate_h_id"] for row in pid_rows])
            tau = safe_corr("kendall", [row[score_name] for row in pid_rows], [row["candidate_h_id"] for row in pid_rows])
            per_pid.append({"pid": pid, "candidate_count": len(pid_rows), "spearman": rho, "kendall": tau})
        rho_values = np.asarray([row["spearman"] for row in per_pid if row["spearman"] is not None], dtype=np.float64)
        result["scores"][score_name] = {
            "valid_candidate_count": len(valid),
            "distribution": summary_stats(x),
            "global_pearson": safe_corr("pearson", x, y),
            "global_spearman": safe_corr("spearman", x, y),
            "per_pid_valid_count": len(per_pid),
            "per_pid_spearman_mean": finite_or_none(np.mean(rho_values)) if len(rho_values) else None,
            "per_pid_spearman_median": finite_or_none(np.median(rho_values)) if len(rho_values) else None,
            "per_pid_spearman_positive_rate": finite_or_none(np.mean(rho_values > 0.0)) if len(rho_values) else None,
            "per_pid_details": per_pid,
        }
    return result


def add_quadrants(group_rows: list[dict], source_scores: dict) -> tuple[dict, dict]:
    eligible = [row for row in group_rows if source_scores[(int(row["pid"]), int(row["cloth_id"]))]["h_cloth_loo_mean"] is not None]
    h_values = np.asarray([row["h_id_mean"] for row in eligible], dtype=np.float64)
    r_values = np.asarray([row["reliability"] for row in eligible], dtype=np.float64)
    h_threshold = float(np.median(h_values))
    r_threshold = float(np.median(r_values))
    counts = defaultdict(int)
    for row in group_rows:
        key = (int(row["pid"]), int(row["cloth_id"]))
        h_high = float(row["h_id_mean"]) >= h_threshold
        r_high = float(row["reliability"]) >= r_threshold
        if h_high and r_high:
            quadrant = "HIGH_HARD_HIGH_RELIABILITY"
        elif h_high and not r_high:
            quadrant = "HIGH_HARD_LOW_RELIABILITY"
        elif not h_high and r_high:
            quadrant = "LOW_HARD_HIGH_RELIABILITY"
        else:
            quadrant = "LOW_HARD_LOW_RELIABILITY"
        row["hardness_reliability_quadrant"] = quadrant
        counts[quadrant] += 1
    return {
        "hardness_median": h_threshold,
        "reliability_median": r_threshold,
        "counts_all_groups": dict(counts),
        "eligible_group_count_for_thresholds": len(eligible),
        "threshold_note": "medians are descriptive partition points, not tuned training parameters",
    }, {key: value for key, value in counts.items()}


def representative_cases(group_rows: list[dict], source_scores: dict) -> dict:
    eligible = [row for row in group_rows if source_scores[(int(row["pid"]), int(row["cloth_id"]))]["h_cloth_loo_mean"] is not None]
    h_threshold = float(np.median([row["h_id_mean"] for row in eligible]))
    c_threshold = float(np.median([source_scores[(int(row["pid"]), int(row["cloth_id"]))]["h_cloth_loo_mean"] for row in eligible]))
    buckets = defaultdict(list)
    for row in eligible:
        key = (int(row["pid"]), int(row["cloth_id"]))
        h_high = row["h_id_mean"] >= h_threshold
        c_high = source_scores[key]["h_cloth_loo_mean"] >= c_threshold
        if c_high and not h_high:
            category = "CASE_A_HIGH_CLOTHING_LOW_ID_DAMAGE"
        elif not c_high and h_high:
            category = "CASE_B_LOW_CLOTHING_HIGH_ID_DAMAGE"
        elif c_high and h_high:
            category = "CASE_C_BOTH_HIGH"
        else:
            category = "CASE_D_BOTH_LOW"
        case = dict(row)
        case["category"] = category
        case["h_cloth_primary"] = source_scores[key]["h_cloth_loo_mean"]
        case["h_cloth_nearest_primary"] = source_scores[key]["h_cloth_loo_nearest"]
        buckets[category].append(case)
    result = {}
    for category, values in buckets.items():
        if category == "CASE_A_HIGH_CLOTHING_LOW_ID_DAMAGE":
            values.sort(key=lambda row: (-row["h_cloth_primary"], row["h_id_mean"], row["pid"], row["cloth_id"]))
        elif category == "CASE_B_LOW_CLOTHING_HIGH_ID_DAMAGE":
            values.sort(key=lambda row: (-row["h_id_mean"], row["h_cloth_primary"], row["pid"], row["cloth_id"]))
        elif category == "CASE_C_BOTH_HIGH":
            values.sort(key=lambda row: (-row["h_id_mean"], -row["h_cloth_primary"], row["pid"], row["cloth_id"]))
        else:
            values.sort(key=lambda row: (row["h_id_mean"], row["h_cloth_primary"], row["pid"], row["cloth_id"]))
        result[category] = values[:REPRESENTATIVE_CASES_PER_QUADRANT]
    return {
        "thresholds": {"h_id_median": h_threshold, "h_cloth_primary_median": c_threshold},
        "available_counts": {category: len(values) for category, values in buckets.items()},
        "selected_counts": {category: len(values) for category, values in result.items()},
        "cases": result,
    }


def bootstrap_ci(values: np.ndarray, seed_offset: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(SEED + seed_offset)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[sampled].mean(axis=1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "PID-level mean over that PID's expansion events",
    }


def load_historical_events(groups: dict) -> tuple[list[dict], dict]:
    if not HISTORICAL_TRACE_PATH.is_file():
        raise FileNotFoundError(HISTORICAL_TRACE_PATH)
    with HISTORICAL_TRACE_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    initial = {}
    for index, row in enumerate(rows):
        pid = int(row["pid"])
        if pid not in initial:
            selected = parse_int_list(row.get("selected_clothes"))
            initial[pid] = {int(cid) for cid in selected}
    all_clothes = defaultdict(set)
    for pid, cid in groups:
        all_clothes[pid].add(cid)
    expansions = []
    validation = {"trace_rows": len(rows), "expansion_rows": 0, "bad_original_mappings": [], "initial_pids": len(initial)}
    for trace_index, row in enumerate(rows):
        selected_new = parse_int_scalar(row.get("new_clothes"))
        if selected_new is None:
            continue
        pid = int(row["pid"])
        active_original = set(parse_int_list(row.get("selected_clothes")))
        candidates_original = set(parse_int_list(row.get("candidate_clothing")))
        expected_candidates = set(all_clothes[pid]) - active_original
        if selected_new not in candidates_original or not candidates_original.issubset(all_clothes[pid]):
            validation["bad_original_mappings"].append({"trace_index": trace_index, "pid": pid, "selected": selected_new, "candidates": sorted(candidates_original)})
        if candidates_original != expected_candidates:
            validation.setdefault("candidate_set_differences", []).append({"trace_index": trace_index, "pid": pid, "trace": sorted(candidates_original), "expected": sorted(expected_candidates)})
        expansions.append({
            "trace_index": trace_index,
            "epoch": int(row["epoch"]),
            "pid": pid,
            "active_original": sorted(active_original),
            "candidates_original": sorted(candidates_original),
            "selected_original": int(selected_new),
            "all_clothes": sorted(all_clothes[pid]),
            "candidate_distances_original": {
                int(key): float(value)
                for key, value in (parse_json_or_python(row.get("candidate_distances"), {}) or {}).items()
            },
        })
    validation["expansion_rows"] = len(expansions)
    validation["candidate_set_difference_count"] = len(validation.get("candidate_set_differences", []))
    validation["bad_original_mapping_count"] = len(validation["bad_original_mappings"])
    if validation["bad_original_mapping_count"]:
        raise RuntimeError(f"historical trace has invalid mappings: {validation['bad_original_mappings'][:2]}")
    if validation["expansion_rows"] != 179:
        raise RuntimeError(f"unexpected historical expansion count: {validation['expansion_rows']}")
    return expansions, {"initial_selected_by_pid": {str(pid): sorted(values) for pid, values in initial.items()}, "validation": validation}


def text_ds_scores(candidate: int, active: set[int], pid: int, centers: dict, mode: str) -> float:
    if not active:
        return 0.0
    candidate_center = centers[(pid, candidate)]
    distances = [cosine_distance(candidate_center, centers[(pid, cid)]) for cid in sorted(active)]
    if mode == "nearest":
        return float(min(distances))
    if mode == "mean":
        return float(np.mean(distances))
    raise ValueError(mode)


def choose_max(candidates: list[int], score_fn) -> tuple[int, dict[int, float]]:
    scores = {int(cid): float(score_fn(int(cid))) for cid in candidates}
    selected = max(candidates, key=lambda cid: (scores[int(cid)], -int(cid)))
    return int(selected), scores


def replay_scheduler(groups: dict, group_rows: list[dict], text_centers: dict, expansions: list[dict], initial: dict) -> tuple[list[dict], dict, list[dict]]:
    group_by_key = {(int(row["pid"]), int(row["cloth_id"])): row for row in group_rows}
    methods = ["id_damage", "reliable_damage", "text_clothing_nearest", "text_clothing_mean"]
    states = {method: {int(pid): set(values) for pid, values in initial.items()} for method in methods}
    sequences = {method: defaultdict(list) for method in methods}
    sequences["original"] = defaultdict(list)
    positions = defaultdict(int)
    event_rows = []
    candidate_rows = []
    for event_id, event in enumerate(expansions, start=1):
        pid = event["pid"]
        positions[pid] += 1
        original_key = (pid, event["selected_original"])
        original_row = group_by_key[original_key]
        base = {
            "event_id": event_id,
            "trace_index": event["trace_index"],
            "epoch": event["epoch"],
            "pid": pid,
            "expansion_position_for_pid": positions[pid],
            "original_active_clothes": normalize_for_json_list(event["active_original"]),
            "original_candidates": normalize_for_json_list(event["candidates_original"]),
            "original_selected": event["selected_original"],
            "original_selected_h_id": original_row["h_id_mean"],
            "original_selected_reliability": original_row["reliability"],
            "original_selected_cloth_key": groups[original_key]["cloth_key"],
        }
        sequences["original"][pid].append(event["selected_original"])
        for candidate in event["candidates_original"]:
            candidate_rows.append({
                "event_id": event_id,
                "trace_index": event["trace_index"],
                "epoch": event["epoch"],
                "pid": pid,
                "expansion_position_for_pid": positions[pid],
                "active_original": normalize_for_json_list(event["active_original"]),
                "candidate_cloth_id": candidate,
                "original_selected": int(candidate == event["selected_original"]),
                "original_ds_score_v4": event["candidate_distances_original"].get(candidate),
                "candidate_h_id": group_by_key[(pid, candidate)]["h_id_mean"],
                "candidate_reliability": group_by_key[(pid, candidate)]["reliability"],
                "h_cloth_ds_nearest": text_ds_scores(candidate, set(event["active_original"]), pid, text_centers, "nearest"),
                "h_cloth_ds_mean": text_ds_scores(candidate, set(event["active_original"]), pid, text_centers, "mean"),
            })
        for method_index, method in enumerate(methods):
            active = states[method][pid]
            candidates = sorted(set(event["all_clothes"]) - active)
            if not candidates:
                raise RuntimeError(f"no candidate left for {method}, pid {pid}, event {event_id}")
            if method == "id_damage":
                selected, score_map = choose_max(candidates, lambda cid: group_by_key[(pid, cid)]["h_id_mean"])
            elif method == "reliable_damage":
                selected, score_map = choose_max(candidates, lambda cid: group_by_key[(pid, cid)]["h_id_mean"] * group_by_key[(pid, cid)]["reliability"])
            elif method == "text_clothing_nearest":
                selected, score_map = choose_max(candidates, lambda cid: text_ds_scores(cid, active, pid, text_centers, "nearest"))
            else:
                selected, score_map = choose_max(candidates, lambda cid: text_ds_scores(cid, active, pid, text_centers, "mean"))
            states[method][pid].add(selected)
            sequences[method][pid].append(selected)
            selected_row = group_by_key[(pid, selected)]
            prefix = method
            base[f"{prefix}_active_clothes"] = normalize_for_json_list(sorted(active))
            base[f"{prefix}_candidates"] = normalize_for_json_list(candidates)
            base[f"{prefix}_candidate_scores"] = json.dumps(json_safe(score_map), sort_keys=True, separators=(",", ":"))
            base[f"{prefix}_selected"] = selected
            base[f"{prefix}_selection_changed"] = int(selected != event["selected_original"])
            base[f"{prefix}_selected_h_id"] = selected_row["h_id_mean"]
            base[f"{prefix}_selected_reliability"] = selected_row["reliability"]
            base[f"{prefix}_h_id_delta"] = selected_row["h_id_mean"] - original_row["h_id_mean"]
            base[f"{prefix}_h_id_gain_product"] = selected_row["h_id_mean"] * original_row["h_id_mean"]
            base[f"{prefix}_reliability_delta"] = selected_row["reliability"] - original_row["reliability"]
            if method_index == 0:
                base["original_h_id_definition_note"] = "Requested product is retained; delta is the directional metric because H_ID is non-negative."
        event_rows.append(base)

    summaries = {}
    for method_index, method in enumerate(methods):
        summaries[method] = summarize_replay_method(method, event_rows, sequences[method], sequences["original"], seed_offset=100 + method_index)
    original_h = np.asarray([row["original_selected_h_id"] for row in event_rows], dtype=np.float64)
    original_r = np.asarray([row["original_selected_reliability"] for row in event_rows], dtype=np.float64)
    original_summary = {
        "method": "original_ds",
        "selection_source": "historical V4 curriculum trace; existing V4 ReID Maximum selection is not recomputed",
        "total_events": len(event_rows),
        "changed_event_count": 0,
        "pid_count": len({int(row["pid"]) for row in event_rows}),
        "changed_pid_count": 0,
        "mean_selected_h_id": float(np.mean(original_h)),
        "mean_selected_reliability": float(np.mean(original_r)),
        "sequence_change_counts": {"first_changed": 0, "second_changed": 0, "full_sequence_changed": 0},
    }
    return event_rows, {
        "trace": str(HISTORICAL_TRACE_PATH),
        "historical_report": str(HISTORICAL_REPORT_PATH),
        "historical_pipeline_rule": str(HISTORICAL_PIPELINE_PATH),
        "event_count": len(event_rows),
        "initial_pool_is_shared": True,
        "expansion_event_timing_is_shared": True,
        "method_states_are_dynamic": True,
        "selection_tie_rule": "larger score; ties resolved by smaller global clothes_id",
        "original_ds": original_summary,
        "methods": summaries,
    }, candidate_rows


def summarize_replay_method(method: str, event_rows: list[dict], sequence: dict, original_sequence: dict, seed_offset: int) -> dict:
    delta_h = np.asarray([row[f"{method}_h_id_delta"] for row in event_rows], dtype=np.float64)
    product = np.asarray([row[f"{method}_h_id_gain_product"] for row in event_rows], dtype=np.float64)
    delta_r = np.asarray([row[f"{method}_reliability_delta"] for row in event_rows], dtype=np.float64)
    changed = np.asarray([row[f"{method}_selection_changed"] for row in event_rows], dtype=bool)
    pids = sorted(set(int(row["pid"]) for row in event_rows))
    per_pid_rows = []
    for pid in pids:
        pid_rows = [row for row in event_rows if int(row["pid"]) == pid]
        pid_delta = float(np.mean([row[f"{method}_h_id_delta"] for row in pid_rows]))
        pid_product = float(np.mean([row[f"{method}_h_id_gain_product"] for row in pid_rows]))
        pid_reliability = float(np.mean([row[f"{method}_reliability_delta"] for row in pid_rows]))
        per_pid_rows.append({
            "pid": pid,
            "event_count": len(pid_rows),
            "selection_changed": bool(any(row[f"{method}_selection_changed"] for row in pid_rows)),
            "mean_h_id_delta": pid_delta,
            "mean_h_id_gain_product": pid_product,
            "mean_reliability_delta": pid_reliability,
        })
    pid_h = np.asarray([row["mean_h_id_delta"] for row in per_pid_rows], dtype=np.float64)
    pid_product = np.asarray([row["mean_h_id_gain_product"] for row in per_pid_rows], dtype=np.float64)
    pid_r = np.asarray([row["mean_reliability_delta"] for row in per_pid_rows], dtype=np.float64)
    changed_pid_rows = [row for row in per_pid_rows if row["selection_changed"]]
    changed_pid_values = np.asarray([row["mean_h_id_delta"] for row in changed_pid_rows], dtype=np.float64)
    top_pid_index = int(np.argmax(np.abs(pid_h))) if len(pid_h) else None
    without_top = np.delete(pid_h, top_pid_index) if top_pid_index is not None and len(pid_h) > 1 else np.asarray([], dtype=np.float64)
    sequence_stats = {}
    for pid in pids:
        original = list(original_sequence[pid])
        candidate = list(sequence[pid])
        sequence_stats[str(pid)] = {
            "original": original,
            "candidate": candidate,
            "first_changed": bool(len(original) >= 1 and len(candidate) >= 1 and original[0] != candidate[0]),
            "second_changed": bool(len(original) >= 2 and len(candidate) >= 2 and original[1] != candidate[1]),
            "full_sequence_changed": bool(original != candidate),
        }
    first_denominator = sum(len(original_sequence[pid]) >= 1 for pid in pids)
    second_denominator = sum(len(original_sequence[pid]) >= 2 for pid in pids)
    first_count = sum(item["first_changed"] for item in sequence_stats.values())
    second_count = sum(item["second_changed"] for item in sequence_stats.values())
    full_count = sum(item["full_sequence_changed"] for item in sequence_stats.values())
    h_ci = bootstrap_ci(pid_h, seed_offset)
    product_ci = bootstrap_ci(pid_product, seed_offset + 100)
    r_ci = bootstrap_ci(pid_r, seed_offset + 200)
    changed_h_ci = bootstrap_ci(changed_pid_values, seed_offset + 300)
    no_clear_reliability_drop = r_ci["upper_95"] is None or r_ci["upper_95"] >= 0.0
    not_single_pid = bool(len(changed_pid_rows) > 1 and len(without_top) > 0 and float(np.mean(without_top)) > 0.0)
    h_positive = bool(h_ci["lower_95"] is not None and h_ci["lower_95"] > 0.0)
    # All outfits may eventually be selected, so an unweighted sum over the
    # complete sequence is invariant to ordering.  The curriculum question is
    # therefore assessed on the first two additions, which are explicitly
    # required in the Phase 5A protocol.  Full-sequence and position-wise
    # values remain in the artifact for auditability.
    prefix_metrics = {}
    for prefix_k in (1, 2, 3):
        prefix_pid_rows = []
        for pid in pids:
            pid_rows = sorted(
                [row for row in event_rows if int(row["pid"]) == pid],
                key=lambda row: int(row["expansion_position_for_pid"]),
            )
            if len(pid_rows) < prefix_k:
                continue
            prefix_rows = pid_rows[:prefix_k]
            prefix_pid_rows.append({
                "pid": pid,
                "mean_h_id_delta": float(np.mean([row[f"{method}_h_id_delta"] for row in prefix_rows])),
                "mean_reliability_delta": float(np.mean([row[f"{method}_reliability_delta"] for row in prefix_rows])),
                "mean_requested_product": float(np.mean([row[f"{method}_h_id_gain_product"] for row in prefix_rows])),
            })
        prefix_h = np.asarray([row["mean_h_id_delta"] for row in prefix_pid_rows], dtype=np.float64)
        prefix_r = np.asarray([row["mean_reliability_delta"] for row in prefix_pid_rows], dtype=np.float64)
        prefix_product = np.asarray([row["mean_requested_product"] for row in prefix_pid_rows], dtype=np.float64)
        prefix_top = int(np.argmax(np.abs(prefix_h))) if len(prefix_h) else None
        prefix_without_top = np.delete(prefix_h, prefix_top) if prefix_top is not None and len(prefix_h) > 1 else np.asarray([], dtype=np.float64)
        prefix_h_ci = bootstrap_ci(prefix_h, seed_offset + 400 + prefix_k)
        prefix_r_ci = bootstrap_ci(prefix_r, seed_offset + 500 + prefix_k)
        prefix_not_single = bool(len(prefix_without_top) > 0 and float(np.mean(prefix_without_top)) > 0.0)
        prefix_no_clear_r_drop = prefix_r_ci["upper_95"] is None or prefix_r_ci["upper_95"] >= 0.0
        prefix_h_positive = bool(prefix_h_ci["lower_95"] is not None and prefix_h_ci["lower_95"] > 0.0)
        prefix_metrics[str(prefix_k)] = {
            "prefix_k": prefix_k,
            "pid_count": len(prefix_pid_rows),
            "mean_h_id_delta": float(np.mean(prefix_h)) if len(prefix_h) else None,
            "mean_reliability_delta": float(np.mean(prefix_r)) if len(prefix_r) else None,
            "mean_requested_product": float(np.mean(prefix_product)) if len(prefix_product) else None,
            "pid_level_h_id_delta_ci": prefix_h_ci,
            "pid_level_reliability_delta_ci": prefix_r_ci,
            "top_absolute_pid": prefix_pid_rows[prefix_top] if prefix_top is not None else None,
            "mean_h_id_delta_without_top_absolute_pid": float(np.mean(prefix_without_top)) if len(prefix_without_top) else None,
            "not_single_pid_positive_support": prefix_not_single,
            "no_clear_reliability_drop": prefix_no_clear_r_drop,
            "h_id_directional_positive_ci": prefix_h_positive,
            "per_pid": prefix_pid_rows,
        }
    prefix_gate_metrics = prefix_metrics["2"]
    replay_gate = bool(
        prefix_gate_metrics["h_id_directional_positive_ci"]
        and prefix_gate_metrics["no_clear_reliability_drop"]
        and prefix_gate_metrics["not_single_pid_positive_support"]
    )
    return {
        "method": method,
        "total_events": len(event_rows),
        "changed_event_count": int(np.sum(changed)),
        "changed_event_rate": float(np.mean(changed)),
        "positive_delta_event_count": int(np.sum(delta_h > 0.0)),
        "positive_delta_event_rate": float(np.mean(delta_h > 0.0)),
        "mean_h_id_delta_all_events": float(np.mean(delta_h)),
        "median_h_id_delta_all_events": float(np.median(delta_h)),
        "mean_h_id_delta_changed_events": float(np.mean(delta_h[changed])) if np.any(changed) else None,
        "median_h_id_delta_changed_events": float(np.median(delta_h[changed])) if np.any(changed) else None,
        "mean_requested_h_id_gain_product": float(np.mean(product)),
        "mean_reliability_delta_all_events": float(np.mean(delta_r)),
        "median_reliability_delta_all_events": float(np.median(delta_r)),
        "mean_reliability_delta_changed_events": float(np.mean(delta_r[changed])) if np.any(changed) else None,
        "median_reliability_delta_changed_events": float(np.median(delta_r[changed])) if np.any(changed) else None,
        "pid_count": len(pids),
        "changed_pid_count": len(changed_pid_rows),
        "pid_level_h_id_delta_ci": h_ci,
        "pid_level_requested_product_ci": product_ci,
        "pid_level_reliability_delta_ci": r_ci,
        "changed_pid_h_id_delta_ci": changed_h_ci,
        "top_absolute_pid": per_pid_rows[top_pid_index] if top_pid_index is not None else None,
        "mean_h_id_delta_without_top_absolute_pid": float(np.mean(without_top)) if len(without_top) else None,
        "not_single_pid_positive_support": not_single_pid,
        "no_clear_reliability_drop": no_clear_reliability_drop,
        "h_id_directional_positive_ci": h_positive,
        "prefix_metrics": prefix_metrics,
        "gate_definition": "prefix_k=2; positive PID-bootstrap lower CI for delta_H_ID, no clear reliability decrease, and positive support after removing the largest absolute PID",
        "replay_gate": replay_gate,
        "sequence_change_counts": {
            "first_changed": first_count,
            "first_total": first_denominator,
            "second_changed": second_count,
            "second_total": second_denominator,
            "full_sequence_changed": full_count,
            "full_sequence_total": len(pids),
        },
        "per_pid": per_pid_rows,
        "sequence_details": sequence_stats,
        "directional_metric_note": "delta_H_ID = selected_new - selected_original; requested product is descriptive only because both terms are non-negative.",
    }


def feature_record_rows(text: dict, bio: dict, identity: dict, sources: dict) -> list[dict]:
    rows = []
    for index in range(len(text["stems"])):
        rows.append({
            "index": index,
            "stem": str(text["stems"][index]),
            "image_filename": f"{text['stems'][index]}.png",
            "image_path": str(TRAIN_ROOT / f"{text['stems'][index]}.png"),
            "pid": int(bio["pid"][index]),
            "cloth_id": int(bio["clothes_id"][index]),
            "cloth_key": str(text["cloth_key"][index]),
            "h_id": identity["h_id"][index],
            "h_id_euclidean": identity["h_id_euclidean"][index],
            "own_center_distance": identity["own_distance"][index],
            "nearest_negative_distance": identity["nearest_negative_distance"][index],
            "nearest_negative_pid": int(identity["nearest_negative_pid"][index]),
            "eva_global_raw_norm": sources["eva_global"]["raw_norm"][index],
            "bio_projected_raw_norm": sources["bio_projected"]["raw_norm"][index],
            "text_clothing_raw_norm": sources["clothing_text_fixed"]["raw_norm"][index],
        })
    return rows


def write_representative_csv(path: Path, cases: dict) -> None:
    rows = []
    for category, values in cases["cases"].items():
        for row in values:
            output = dict(row)
            output["category"] = category
            rows.append(output)
    write_csv(path, rows)


def short(value, digits=4):
    value = finite_or_none(value)
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value):
    value = finite_or_none(value)
    return "—" if value is None else f"{100.0 * value:.1f}%"


def ci_text(ci: dict) -> str:
    if ci.get("mean") is None:
        return "—"
    return f"{ci['mean']:.6f} [{ci['lower_95']:.6f}, {ci['upper_95']:.6f}]"


def write_report(analysis: dict) -> None:
    source_meta = analysis["sources"]
    correlation = analysis["correlation"]
    predictive = analysis["predictive"]
    pair_summary = analysis["pair_summary"]
    replay = analysis["replay"]
    rel = analysis["reliability"]
    source_order = ["clothing_text_fixed", "clothing_original_full_failed", "clothing_official_full", "eva_global", "bio_projected"]
    lines = [
        "# PHASE 5A — Clothing Hardness Feasibility",
        "",
        "## Frozen scope",
        "",
        "This is an offline diagnostic only. No new 60/25-epoch training, PRCC run, GRL/Reverse repair, DS formal training, DW formal training, test retrieval evaluation, or test-set tuning was performed. The Phase 4R reproduction and repair artifacts were not overwritten.",
        "",
        f"Analysis seed: `{SEED}`. Dataset: LTCC train, `{analysis['alignment']['expected_counts']['images']}` images, `{analysis['alignment']['expected_counts']['pids']}` PIDs, `{analysis['alignment']['expected_counts']['clothes']}` clothing states.",
        "",
        "The already extracted Bio E27 feature bank is treated as the frozen Bio Control extraction. It is not regenerated through a model or GPU in this phase.",
        "",
        "## Representation sources",
        "",
        "| Source | Role/layer | Dim | Normalization | Checkpoint/bank SHA |",
        "|---|---|---:|---|---|",
    ]
    for name in source_order:
        item = source_meta[name]
        checkpoint = item.get("source_checkpoint_sha256") or item.get("source_file_sha256") or "—"
        lines.append(f"| `{name}` | {item['layer']} | {item['dimension']} | {item['normalization']} | `{checkpoint}` |")
    lines.extend([
        "",
        f"Fixed semantic Clothing source: `{TEXT_RESOURCE_PATH}`; factor `{TEXT_FACTOR_INDEX}` (`{TEXT_FACTOR_NAME}`); text-resource SHA256 `{source_meta['clothing_text_fixed']['source_file_sha256']}`. The supplied NPZ has no raw clothing strings or independent attribute labels.",
        "",
        f"The failed local Full Clothing branch is included only as a supplementary source, not as the primary signal. The official trained branch is also included as a source-sensitivity check. The primary Clothing source is the fixed semantic factor.",
        "",
        "## Alignment and definitions",
        "",
        f"- Stem set and PID alignment across all banks: **PASS** (`{analysis['alignment']['text_bio_stems_exact']}`).",
        f"- Clothing-state mapping `(PID, global clothes_id)`: **PASS**; `{analysis['group_count']}` groups.",
        f"- Camera labels differ by indexing convention and were not used: **{analysis['alignment']['camera_difference_not_used']}**.",
        "- Main identity feature `f_i`: normalized `Bio E27 global_feature` (1024-D), because normalized identity feature is used at inference.",
        "- Identity centers `C_y`: L2-normalized mean of normalized image features per PID.",
        "- Nearest negative center `C_qi`: most similar competing PID center; distances are cosine distances.",
        "- Main `H_ID`: `d(f_i,C_y) * d(f_i,C_qi)` exactly as frozen in the request. Euclidean-product values are supplementary.",
        "- Main `H_cloth` for a clothes state: leave-one-out mean cosine distance from its normalized Clothing center to all other same-PID clothing centers. DS-style nearest and mean-to-active-center scores are separately recorded during replay.",
        "- Operational replay direction: larger `H_ID` is treated as harder, following the explicit Phase 5A replay instruction. Under the product-of-distances definition, `H_ID<0` cannot occur; the requested `H_ID(new)*H_ID(original)` is retained but cannot encode direction, so the directional metric is `delta_H_ID = new - original`.",
        "",
        "## Clothing hardness",
        "",
        "Pair-level same/different clothing separation is descriptive and is not treated as identity-difficulty validation by itself.",
        "",
        "| Source | Same mean | Different mean | Same/diff AUC | Pair rank | Cohen d |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in source_order:
        item = pair_summary["sources"][name]
        lines.append(f"| `{name}` | {short(item['same_clothes']['mean'])} | {short(item['different_clothes']['mean'])} | {short(item['roc_auc_different_vs_same'])} | {short(item['pairwise_ranking_accuracy'])} | {short(item['cohen_d'])} |")
    lines.extend([
        "",
        "The binary same/different clothes label is a constant class indicator for this within-PID pair task and has AUC 0.5; it cannot tell which of two different outfits is harder.",
        "",
        "## Identity damage",
        "",
        f"Image-level `H_ID` cosine-product distribution: `{json.dumps(json_safe(analysis['identity']['h_id_stats']))}`.",
        f"Image-level own-center distance: `{json.dumps(json_safe(analysis['identity']['own_distance_stats']))}`; nearest-negative distance: `{json.dumps(json_safe(analysis['identity']['negative_distance_stats']))}`.",
        f"Clothes-group mean `H_ID` distribution: `{json.dumps(json_safe(analysis['identity']['group_h_id_stats']))}`.",
        "",
        "## Correlation and within-PID ranking",
        "",
        "The primary decision uses `clothing_text_fixed`; the other Clothing spaces test whether the result depends on a trained branch.",
        "",
        "| Source | Global Pearson | Global Spearman | PID rho mean | PID rho median | PID rho >0 | Top1 | Top2 overlap | Kendall mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in source_order:
        item = correlation[name]
        lines.append(f"| `{name}` | {short(item['global_pearson'])} | {short(item['global_spearman'])} | {short(item['per_pid_spearman_mean'])} | {short(item['per_pid_spearman_median'])} | {pct(item['per_pid_spearman_positive_rate'])} | {pct(item['top1_agreement_rate'])} | {short(item['top2_overlap_mean'])} | {short(item['per_pid_kendall_mean'])} |")
    lines.extend([
        "",
        "## Predictive power for Bio identity damage",
        "",
        "Labels are the exact top and bottom 25% of eligible clothes groups by `H_ID`; the middle 50% is excluded. This uses only training features and does not use query/gallery retrieval.",
        "",
        "| Source | Eligible groups | ROC-AUC | PR-AUC | Spearman |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in source_order:
        item = predictive[name]
        lines.append(f"| `{name}` | {item['eligible_group_count']} | {short(item['predictive_roc_auc'])} | {short(item['predictive_pr_auc'])} | {short(item['predictive_spearman_all_eligible'])} |")
    lines.extend([
        "",
        f"The EVA/global and Bio rows are reference baselines; their H_ID target is derived from the Bio global feature, so they are not independent semantic validation. The trained Clothing branches have stronger apparent prediction than fixed text but show MODERATE prototype-level identity leakage (`{analysis['identity_leakage']['clothing_original_full_failed']['median_ratio_same_pid_over_nearest_cross']:.4f}` and `{analysis['identity_leakage']['clothing_official_full']['median_ratio_same_pid_over_nearest_cross']:.4f}` median ratios), so they are not used as the primary pure-Clothing conclusion. Fixed text Clothing is LOW leakage (`{analysis['identity_leakage']['clothing_text_fixed']['median_ratio_same_pid_over_nearest_cross']:.4f}`).",
        "",
        "### C_seen-conditioned DS-style correlation",
        "",
        f"Candidate-level analysis uses `{analysis['ds_candidate_correlation']['candidate_event_count']}` candidates from `{analysis['ds_candidate_correlation']['event_count']}` historical expansion events. `C_seen` is the original active clothing set at each event; this is separate from the leave-one-out group ranking above.",
        "",
        "| Candidate score | Global Pearson | Global Spearman | PID rho mean | PID rho median | PID rho >0 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    candidate_labels = {
        "original_ds_score_v4": "Original V4 DS score",
        "h_cloth_ds_nearest": "H_cloth_DS nearest",
        "h_cloth_ds_mean": "H_cloth_DS mean",
    }
    for name in ["original_ds_score_v4", "h_cloth_ds_nearest", "h_cloth_ds_mean"]:
        item = analysis["ds_candidate_correlation"]["scores"][name]
        lines.append(f"| {candidate_labels[name]} | {short(item['global_pearson'])} | {short(item['global_spearman'])} | {short(item['per_pid_spearman_mean'])} | {short(item['per_pid_spearman_median'])} | {pct(item['per_pid_spearman_positive_rate'])} |")
    lines.extend([
        "",
        "## Representative mismatch cases",
        "",
        f"Primary thresholds are median `H_ID={analysis['cases']['thresholds']['h_id_median']:.6f}` and median `H_cloth={analysis['cases']['thresholds']['h_cloth_primary_median']:.6f}`. Available counts: `{json.dumps(json_safe(analysis['cases']['available_counts']), sort_keys=True)}`; up to 20 per category are exported to `representative_cases.csv` and `.json`.",
        "",
        "- Case A: high Clothing difference, low Bio identity damage.",
        "- Case B: low Clothing difference, high Bio identity damage.",
        "- Case C: both high.",
        "- Case D: both low.",
        "",
        "Each exported row includes PID, clothes IDs/keys, image stems and paths, `H_cloth`, `H_ID`, own-center distance, nearest-negative distance, support, and reliability proxies.",
        "",
        "## Reliability",
        "",
        f"Reliability is an untuned descriptive combination: `{rel['combined_formula']}`. Feature-norm anomaly is reported separately and is not silently folded into the combined score.",
        "",
        "| Quantity | Mean | Median |",
        "|---|---:|---:|",
        f"| Within-clothes consistency | {short(analysis['reliability_component_stats']['r_consistency']['mean'])} | {short(analysis['reliability_component_stats']['r_consistency']['median'])} |",
        f"| Bio consistency | {short(analysis['reliability_component_stats']['r_bio']['mean'])} | {short(analysis['reliability_component_stats']['r_bio']['median'])} |",
        f"| Log-saturated support | {short(analysis['reliability_component_stats']['r_support']['mean'])} | {short(analysis['reliability_component_stats']['r_support']['median'])} |",
        f"| Combined reliability | {short(analysis['reliability_component_stats']['reliability']['mean'])} | {short(analysis['reliability_component_stats']['reliability']['median'])} |",
        "",
        f"Hardness/reliability quadrant counts: `{json.dumps(json_safe(analysis['quadrants']['counts_all_groups']), sort_keys=True)}`. The full table is `group_metrics.csv`; high-hard/low-reliability examples are included in `quadrant_examples.csv`.",
        "",
        f"Within the top-H_ID quartile, high- and low-reliability group summaries are in `reliability_hardness_summary.json`. Because no external image-quality or clothing ground truth is available, this is evidence from internal consistency/support proxies, not proof of semantic reliability.",
        "",
        "## Offline DS replay",
        "",
        f"Historical trace: `{HISTORICAL_TRACE_PATH}`; `{replay['event_count']}` expansion events. Original DS is the frozen historical V4 ReID Maximum selection (`new_clothes`), not a newly recomputed result. Event timing, initial pool, PID set, and number of expansions are shared; each candidate method dynamically updates its own active clothing set after selecting a state. Tie rule: larger score, then smaller global clothes ID.",
        "",
        "| Method | Changed events | Changed PIDs | Prefix-2 ΔH_ID | PID 95% CI@2 | Prefix-2 ΔR | R 95% CI@2 | Requested product | Gate |",
        "|---|---:|---:|---:|---|---:|---|---:|---|",
    ])
    replay_display = ["id_damage", "reliable_damage", "text_clothing_nearest", "text_clothing_mean"]
    replay_names = {
        "id_damage": "ID-Damage DS",
        "reliable_damage": "Reliable-Damage DS",
        "text_clothing_nearest": "Clothing DS nearest (supp.)",
        "text_clothing_mean": "Clothing DS mean (supp.)",
    }
    original = replay["original_ds"]
    lines.append(f"| Original DS baseline | 0/{original['total_events']} | 0/{original['pid_count']} | 0.000000 | 0.000000 [0.000000, 0.000000] | 0.000000 | 0.000000 [0.000000, 0.000000] | — | reference |")
    for name in replay_display:
        item = replay["methods"][name]
        prefix = item["prefix_metrics"]["2"]
        lines.append(f"| {replay_names[name]} | {item['changed_event_count']}/{item['total_events']} | {item['changed_pid_count']}/{item['pid_count']} | {short(prefix['mean_h_id_delta'], 6)} | {ci_text(prefix['pid_level_h_id_delta_ci'])} | {short(prefix['mean_reliability_delta'], 6)} | {ci_text(prefix['pid_level_reliability_delta_ci'])} | {short(item['mean_requested_h_id_gain_product'], 6)} | **{'PASS' if item['replay_gate'] else 'FAIL'}** |")
    lines.extend([
        "",
        "Because a complete replay eventually selects the same clothing states, its unweighted all-event ΔH_ID is expected to cancel. The curriculum-order diagnostic therefore uses prefix-2 (the first two additions per PID), while all-event and position-wise values remain in `offline_replay_summary.json`. Gate direction is based on prefix-2 ΔH_ID, PID-level bootstrap, absence of a clear reliability decrease, and positive support after removing the single largest absolute PID contribution.",
        "",
        "| Method | First added state changed | Second added state changed | Full sequence changed |",
        "|---|---:|---:|---:|",
    ])
    for name in replay_display:
        item = replay["methods"][name]
        seq = item["sequence_change_counts"]
        lines.append(f"| {replay_names[name]} | {seq['first_changed']}/{seq['first_total']} | {seq['second_changed']}/{seq['second_total']} | {seq['full_sequence_changed']}/{seq['full_sequence_total']} |")
    lines.extend([
        "",
        "## Final decision",
        "",
        f"Primary Clothing-space predictive AUC: `{short(predictive['clothing_text_fixed']['predictive_roc_auc'])}`; global Spearman: `{short(correlation['clothing_text_fixed']['global_spearman'])}`; PID positive-rate: `{pct(correlation['clothing_text_fixed']['per_pid_spearman_positive_rate'])}`.",
        f"ID-Damage replay gate: **{'PASS' if replay['methods']['id_damage']['replay_gate'] else 'FAIL'}**. Reliable-Damage replay gate: **{'PASS' if replay['methods']['reliable_damage']['replay_gate'] else 'FAIL'}**.",
        "",
        f"Q1 — Does Clothing difference equal identity difficulty? **{analysis['decisions']['q1']}**. This is determined from within-PID correlation, top-k agreement, and top/bottom-H_ID prediction rather than same/different clothing separation alone.",
        f"Q2 — Is ID-Damage more suitable than original Clothing distance for DS? **{analysis['decisions']['q2']}**. The answer uses the frozen historical Original DS trace as baseline and the PID-bootstrap replay comparison.",
        f"Q3 — Can reliability filter hard-but-noisy clothing states? **{analysis['decisions']['q3']}**. The answer is limited to consistency/support/norm proxies because no external quality labels are present.",
        "",
        f"Selected conclusion: **{analysis['decisions']['final_decision']}**.",
        f"Phase 5B gate: **{'PASS — enter Bio + new curriculum formal training only in the next phase' if analysis['decisions']['phase5b_gate'] else 'STOP — do not start formal DS/DW training; redesign hardness first'}**.",
        "",
        "This report does not claim a ReID Rank-1/mAP improvement. Any future formal training must compare Bio baseline, DS-only, DW-only, and DS+DW in that order, and must keep this offline evidence separate from test evaluation.",
        "",
        "## Artifacts",
        "",
        "- `outputs/new_method_track/phase5a_clothing_hardness/REPRESENTATION_SOURCES.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/feature_records.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/group_metrics.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/group_pair_metrics.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/pair_metrics.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/historical_candidate_scores.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/ds_candidate_correlation.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/correlation_summary.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/predictive_power.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/representative_cases.csv` and `representative_cases.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/reliability_metrics.csv` and `quadrant_examples.csv`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/offline_replay_events.csv` and `offline_replay_summary.json`",
        "- `outputs/new_method_track/phase5a_clothing_hardness/PHASE_5A_MANIFEST.json`",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_decisions(analysis: dict) -> dict:
    primary_corr = analysis["correlation"]["clothing_text_fixed"]
    primary_pred = analysis["predictive"]["clothing_text_fixed"]
    id_gate = analysis["replay"]["methods"]["id_damage"]["replay_gate"]
    reliable_gate = analysis["replay"]["methods"]["reliable_damage"]["replay_gate"]
    pid_rho = primary_corr["per_pid_spearman_mean"] or 0.0
    positive_rate = primary_corr["per_pid_spearman_positive_rate"] or 0.0
    auc = primary_pred["predictive_roc_auc"] or 0.5
    if auc > 0.70 and pid_rho > 0.30 and positive_rate >= 0.60:
        q1 = "PARTIAL — Clothing space carries a strong aggregate hardness signal, but it is not identical to identity difficulty."
    elif auc > 0.60 or pid_rho > 0.15 or positive_rate > 0.50:
        q1 = "PARTIAL — Clothing difference is informative but not equivalent to identity difficulty."
    else:
        q1 = "NO — Clothing difference does not reliably track identity damage."
    if id_gate or reliable_gate:
        q2 = "YES — offline selection improves directional H_ID without a clear reliability drop under the replay gate."
    else:
        q2 = "NO/INCONCLUSIVE — the offline replay did not meet the predeclared directional-and-reliability gate."
    hard_rows = analysis["reliability_hardness_summary"]
    q3 = hard_rows["decision"]
    if reliable_gate and hard_rows["decision"].startswith("YES"):
        final = "RELIABILITY FILTER IS NECESSARY"
    elif id_gate:
        final = "ID-DAMAGE SCORE OUTPERFORMS CLOTHING DISTANCE"
    elif auc > 0.60 or pid_rho > 0.15:
        final = "CLOTHING SPACE WEAK BUT USEFUL"
    else:
        final = "NO CURRICULUM SIGNAL SUPPORTED"
    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "final_decision": final,
        "phase5b_gate": bool(id_gate or reliable_gate),
        "decision_rules": {
            "strong_clothing_signal": "AUC > 0.70, mean PID Spearman > 0.30, positive PID rate >= 0.60",
            "weak_useful_signal": "AUC > 0.60 or mean PID Spearman > 0.15 or positive PID rate > 0.50",
            "phase5b": "ID-Damage or Reliable-Damage replay gate; PID bootstrap lower CI > 0, no clear reliability decrease, and positive support after removing top absolute PID",
        },
    }


def reliability_hardness_analysis(group_rows: list[dict]) -> dict:
    values = [row for row in group_rows if math.isfinite(float(row["h_id_mean"]))]
    h_values = np.asarray([row["h_id_mean"] for row in values], dtype=np.float64)
    n = max(1, int(math.floor(len(values) * 0.25)))
    order = np.argsort(h_values, kind="stable")
    hard = [values[int(index)] for index in order[-n:]]
    reliability_order = np.argsort([row["reliability"] for row in hard], kind="stable")
    low = [hard[int(index)] for index in reliability_order[:max(1, len(hard) // 2)]]
    high = [hard[int(index)] for index in reliability_order[-max(1, len(hard) // 2):]]
    metrics = ["r_consistency", "r_bio", "r_support", "feature_norm_anomaly", "support"]
    comparisons = {}
    for metric in metrics:
        low_values = np.asarray([row[metric] for row in low], dtype=np.float64)
        high_values = np.asarray([row[metric] for row in high], dtype=np.float64)
        comparisons[metric] = {"low_reliability_mean": float(np.mean(low_values)), "high_reliability_mean": float(np.mean(high_values)), "high_minus_low": float(np.mean(high_values) - np.mean(low_values))}
    filters_consistency_and_support = comparisons["r_consistency"]["high_minus_low"] > 0.0 and comparisons["r_support"]["high_minus_low"] > 0.0
    norm_not_worse = comparisons["feature_norm_anomaly"]["high_minus_low"] <= 0.0
    if filters_consistency_and_support and norm_not_worse:
        decision = "YES — within the available internal proxies, reliability separates harder states with stronger consistency/support and no worse norm anomaly."
    elif filters_consistency_and_support:
        decision = "PARTIAL — reliability separates consistency/support, but norm-anomaly evidence is mixed."
    else:
        decision = "INCONCLUSIVE — current internal proxies do not show a clean hard-but-noisy separation."
    return {
        "hard_group_count": len(hard),
        "low_reliability_hard_count": len(low),
        "high_reliability_hard_count": len(high),
        "comparisons": comparisons,
        "decision": decision,
        "limitations": "No external image-quality, pose, occlusion, detection, or clothing-attribute labels were used.",
    }


def git_snapshot() -> dict:
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
        diff = subprocess.run(["git", "diff", "--binary", "--", "scripts", "configs", "data", "reports"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    (OUT_ROOT / "git_diff.patch").write_text(diff, encoding="utf-8")
    return {"status_short": status.splitlines(), "tracked_diff_path": str(OUT_ROOT / "git_diff.patch"), "tracked_diff_sha256": sha256(OUT_ROOT / "git_diff.patch")}


def run_analysis() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    text = load_npz(TEXT_BANK_PATH)
    bio, bio_audit = align_bank(BIO_BANK_PATH, text["stems"])
    original, original_audit = align_bank(ORIGINAL_FULL_BANK_PATH, text["stems"])
    official, official_audit = align_bank(OFFICIAL_FULL_BANK_PATH, text["stems"])
    alignment = verify_canonical_alignment(text, bio, original, official)
    groups = build_groups(text, bio)
    manifest = json.loads(CHECKPOINT_MANIFEST_PATH.read_text(encoding="utf-8"))
    sources = make_sources(text, bio, original, official, manifest)
    prepare_sources(sources)
    identity = compute_identity_damage(sources["eva_global"]["raw"], bio["pid"])
    all_source_centers = {name: build_group_centers(groups, source["features"]) for name, source in sources.items()}
    all_source_scores = {name: compute_clothing_scores(groups, centers)[0] for name, centers in all_source_centers.items()}
    reliability = add_reliability_metrics(groups, all_source_centers["clothing_text_fixed"], sources["clothing_text_fixed"], identity, sources["eva_global"])
    group_rows = make_group_rows(groups, text, identity, all_source_scores, reliability)
    group_rows_by_key = {(int(row["pid"]), int(row["cloth_id"])): row for row in group_rows}
    for row in group_rows:
        key = (int(row["pid"]), int(row["cloth_id"]))
        row["h_r"] = float(row["h_id_mean"] * row["reliability"])
    quadrant_info, _ = add_quadrants(group_rows, all_source_scores["clothing_text_fixed"])
    cases = representative_cases(group_rows, all_source_scores["clothing_text_fixed"])
    pairs = sample_pairs(groups, SEED)
    pair_rows, pair_summary = pair_diagnostics(pairs, groups, {**text, "clothes_id": bio["clothes_id"]}, sources, identity)
    group_pair_rows = group_pair_diagnostics(groups, all_source_centers, group_rows_by_key)
    correlation = {name: source_group_analysis(group_rows, groups, all_source_scores[name], name) for name in ["clothing_text_fixed", "clothing_original_full_failed", "clothing_official_full", "eva_global", "bio_projected"]}
    predictive = correlation
    identity_leakage = {name: clothing_identity_leakage(groups, all_source_centers[name]) for name in ["clothing_text_fixed", "clothing_original_full_failed", "clothing_official_full"]}
    expansions, trace_meta = load_historical_events(groups)
    initial = {int(pid): set(values) for pid, values in trace_meta["initial_selected_by_pid"].items()}
    event_rows, replay, candidate_rows = replay_scheduler(groups, group_rows, all_source_centers["clothing_text_fixed"], expansions, initial)
    ds_candidate_correlation = ds_candidate_analysis(candidate_rows)
    reliability_rows = []
    for row in group_rows:
        reliability_rows.append({
            "pid": row["pid"], "cloth_id": row["cloth_id"], "cloth_key": row["cloth_key"], "support": row["support"],
            "h_id_mean": row["h_id_mean"], "h_r": row["h_r"], "r_consistency": row["r_consistency"], "r_bio": row["r_bio"],
            "r_support": row["r_support"], "feature_norm_mean": row["feature_norm_mean"], "feature_norm_anomaly": row["feature_norm_anomaly"],
            "feature_norm_reliability": row["feature_norm_reliability"], "reliability": row["reliability"], "quadrant": row.get("hardness_reliability_quadrant", ""),
        })
    quadrant_examples = [row for row in reliability_rows if row["quadrant"] in ("HIGH_HARD_LOW_RELIABILITY", "HIGH_HARD_HIGH_RELIABILITY")]
    reliability_components = {metric: summary_stats([row[metric] for row in reliability_rows]) for metric in ["r_consistency", "r_bio", "r_support", "reliability", "feature_norm_anomaly"]}
    reliability_hard = reliability_hardness_analysis(group_rows)
    analysis = {
        "alignment": alignment,
        "group_count": len(groups),
        "sources": {name: {key: value for key, value in source.items() if key not in ("raw", "features", "raw_norm")} for name, source in sources.items()},
        "identity": {
            "definition": identity["definition"],
            "supplementary_definition": identity["supplementary_definition"],
            "h_id_stats": summary_stats(identity["h_id"]),
            "own_distance_stats": summary_stats(identity["own_distance"]),
            "negative_distance_stats": summary_stats(identity["nearest_negative_distance"]),
            "h_id_euclidean_stats": summary_stats(identity["h_id_euclidean"]),
            "group_h_id_stats": summary_stats([row["h_id_mean"] for row in group_rows]),
        },
        "pair_summary": pair_summary,
        "correlation": correlation,
        "predictive": predictive,
        "identity_leakage": identity_leakage,
        "reliability": reliability,
        "reliability_component_stats": reliability_components,
        "reliability_hardness_summary": reliability_hard,
        "quadrants": quadrant_info,
        "cases": cases,
        "replay": replay,
        "ds_candidate_correlation": ds_candidate_correlation,
        "trace_meta": trace_meta,
        "bank_audits": {"bio": bio_audit, "original_full": original_audit, "official_full": official_audit},
    }
    analysis["decisions"] = make_decisions(analysis)

    feature_rows = feature_record_rows(text, bio, identity, sources)
    write_json(OUT_ROOT / "REPRESENTATION_SOURCES.json", analysis["sources"])
    write_csv(OUT_ROOT / "feature_records.csv", feature_rows)
    write_csv(OUT_ROOT / "group_metrics.csv", group_rows)
    write_csv(OUT_ROOT / "group_pair_metrics.csv", group_pair_rows)
    write_csv(OUT_ROOT / "pair_metrics.csv", pair_rows)
    write_json(OUT_ROOT / "correlation_summary.json", correlation)
    write_json(OUT_ROOT / "predictive_power.json", predictive)
    write_csv(OUT_ROOT / "historical_candidate_scores.csv", candidate_rows)
    write_json(OUT_ROOT / "ds_candidate_correlation.json", ds_candidate_correlation)
    write_json(OUT_ROOT / "identity_damage_summary.json", analysis["identity"])
    write_json(OUT_ROOT / "identity_leakage_summary.json", identity_leakage)
    write_representative_csv(OUT_ROOT / "representative_cases.csv", cases)
    write_json(OUT_ROOT / "representative_cases.json", cases)
    write_csv(OUT_ROOT / "reliability_metrics.csv", reliability_rows)
    write_csv(OUT_ROOT / "quadrant_examples.csv", quadrant_examples)
    write_json(OUT_ROOT / "reliability_hardness_summary.json", reliability_hard)
    write_csv(OUT_ROOT / "offline_replay_events.csv", event_rows)
    write_json(OUT_ROOT / "offline_replay_summary.json", replay)
    write_json(OUT_ROOT / "phase5a_config.json", {
        "seed": SEED,
        "text_factor_index": TEXT_FACTOR_INDEX,
        "text_factor_name": TEXT_FACTOR_NAME,
        "pair_caps": {"same_per_group": PAIR_CAP_SAME_PER_GROUP, "different_per_pid": PAIR_CAP_DIFFERENT_PER_PID},
        "representative_cases_per_quadrant": REPRESENTATIVE_CASES_PER_QUADRANT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "no_training": True,
        "no_test_retrieval": True,
        "h_id": identity["definition"],
        "h_cloth": "leave-one-out mean cosine distance among same-PID normalized clothing centers",
        "replay": "historical event timing and initial pool fixed; candidate method active set updated dynamically",
    })
    analysis["git_snapshot"] = git_snapshot()
    manifest_output = {
        "phase": "5A",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "offline-only clothing hardness/curriculum feasibility; no training or test retrieval",
        "analysis": analysis,
        "artifacts_root": str(OUT_ROOT),
        "report": str(REPORT_PATH),
        "git_snapshot": analysis["git_snapshot"],
    }
    write_json(OUT_ROOT / "PHASE_5A_MANIFEST.json", manifest_output)
    write_report(analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-json", action="store_true", help="print the compact final decision JSON")
    args = parser.parse_args()
    analysis = run_analysis()
    compact = {
        "report": str(REPORT_PATH),
        "output_root": str(OUT_ROOT),
        "primary": {
            "global_spearman": analysis["correlation"]["clothing_text_fixed"]["global_spearman"],
            "pid_spearman_mean": analysis["correlation"]["clothing_text_fixed"]["per_pid_spearman_mean"],
            "pid_positive_rate": analysis["correlation"]["clothing_text_fixed"]["per_pid_spearman_positive_rate"],
            "predictive_auc": analysis["predictive"]["clothing_text_fixed"]["predictive_roc_auc"],
            "predictive_pr_auc": analysis["predictive"]["clothing_text_fixed"]["predictive_pr_auc"],
        },
        "replay_gates": {name: item["replay_gate"] for name, item in analysis["replay"]["methods"].items()},
        "decisions": analysis["decisions"],
    }
    print(json.dumps(json_safe(compact), indent=2, sort_keys=True))
    if args.print_json:
        print(json.dumps(json_safe(analysis), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
