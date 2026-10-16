#!/usr/bin/env python3
"""Phase 5F: cheap proxy audit for reliable hard-pair identification.

This is an offline, train-only analysis.  It consumes the frozen Phase-5E
pair relations and their already-computed gradient-utility labels, but it does
not compute a new gradient, construct an optimizer, update a ReID parameter,
change sampling, or run any ReID training.

The primary target is the frozen Phase-5E distinction among high-hard
relations: ``HIGH_HARD_HIGH_UTILITY`` (61) versus
``HIGH_HARD_LOW_UTILITY`` (133).  Candidate proxy inputs are forward/feature
and inexpensive image statistics only.  Five-fold validation is split by PID,
not by pair, so an identity cannot occur in both sides of a fold.  A closed
form ridge fit is used only as an offline diagnostic combination of fixed
features; no optimizer or ReID model is involved.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234
PHASE = "phase5f_hard_pair_utility_proxy"
N_FOLDS = 5
RIDGE_LAMBDA = 1.0
EXPECTED_IMAGES = 9576
EXPECTED_PIDS = 77
EXPECTED_STATES = 256
EXPECTED_HARD_ROWS = 194
EXPECTED_HIGH_UTILITY = 61
EXPECTED_LOW_UTILITY = 133

P5D_ROOT = ROOT / "outputs/new_method_track/phase5d_hardness_utility"
P5E_ROOT = ROOT / "outputs/new_method_track/phase5e_id_damage_pair_mining"
P5D_BANK = P5D_ROOT / "feature_bank/feature_bank.npz"
P5D_BANK_CSV = P5D_ROOT / "feature_bank/feature_bank.csv"
P5E_QUADRANTS = P5E_ROOT / "gradient_probes/pair_utility_quadrants.csv"
P5E_MANIFEST = P5E_ROOT / "PHASE5E_MANIFEST.json"
P5E_GRADIENT_MANIFEST = P5E_ROOT / "gradient_probes/pair_gradient_manifest.json"

OUT_ROOT = ROOT / "outputs/new_method_track/phase5f_hard_pair_utility_proxy"
REPORT_PATH = ROOT / "reports/PHASE_5F_HARD_PAIR_UTILITY_PROXY.md"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed
    except Exception:
        return float("nan")


def safe_mean(values: Any) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def safe_corr(x: Any, y: Any, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    try:
        value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
        return float(value) if math.isfinite(float(value)) else None
    except Exception:
        return None


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    valid = np.isfinite(score) & np.isfinite(y)
    if valid.sum() < 3 or len(np.unique(y[valid])) < 2:
        return None
    try:
        return float(roc_auc_score(y[valid], score[valid]))
    except Exception:
        return None


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    valid = np.isfinite(score) & np.isfinite(y)
    if valid.sum() < 3 or len(np.unique(y[valid])) < 2:
        return None
    try:
        return float(average_precision_score(y[valid], score[valid]))
    except Exception:
        return None


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def load_data() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    required = [P5D_BANK, P5D_BANK_CSV, P5E_QUADRANTS, P5E_MANIFEST, P5E_GRADIENT_MANIFEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 5F frozen inputs missing: " + ", ".join(missing))
    bank = dict(np.load(P5D_BANK, allow_pickle=False))
    bank_rows = read_csv(P5D_BANK_CSV)
    pair_rows = read_csv(P5E_QUADRANTS)
    if len(bank_rows) != EXPECTED_IMAGES or len(pair_rows) != 384:
        raise RuntimeError(f"unexpected input sizes: {len(bank_rows)} bank rows, {len(pair_rows)} pair rows")
    if len(np.unique(np.asarray(bank["pid"]))) != EXPECTED_PIDS:
        raise RuntimeError("Phase 5D bank PID count changed")
    if len(set(zip(np.asarray(bank["pid"]).tolist(), np.asarray(bank["clothes_id"]).tolist()))) != EXPECTED_STATES:
        raise RuntimeError("Phase 5D bank clothing-state count changed")
    stems = np.asarray(bank["stems"]).astype(str)
    if not np.array_equal(stems, np.asarray([row["stem"] for row in bank_rows]).astype(str)):
        raise RuntimeError("Phase 5D NPZ/CSV order mismatch")
    if any("/query/" in row["anchor_path"] or "/gallery/" in row["anchor_path"] or "/test/" in row["anchor_path"] for row in pair_rows):
        raise RuntimeError("query/gallery/test path found in Phase 5E pair bank")
    for row in pair_rows:
        for key in ["anchor_path", "positive_path", "negative_path"]:
            if "/train/" not in row[key]:
                raise RuntimeError(f"non-train pair path found: {row[key]}")
    data = {
        "bank": bank,
        "bank_rows": bank_rows,
        "bank_by_index": {int(row["index"]): row for row in bank_rows},
        "global_feature": np.asarray(bank["global_feature"], dtype=np.float32),
        "bio_projected": np.asarray(bank["bio_projected"], dtype=np.float32),
        "id_logits": np.asarray(bank["id_logits"], dtype=np.float32),
        "pids": np.asarray(bank["pid"], dtype=np.int64),
        "clothes": np.asarray(bank["clothes_id"], dtype=np.int64),
        "pair_rows": pair_rows,
    }
    data["global_normed"] = normalize(data["global_feature"])
    data["bio_normed"] = normalize(data["bio_projected"])
    data["entropy"] = image_entropy(data["id_logits"])
    data["pid_centers"] = pid_centers(data["global_normed"], data["pids"])
    data["own_similarity"] = np.asarray([1.0 - safe_float(row["own_center_distance"]) for row in bank_rows], dtype=np.float64)
    data["negative_proximity"] = np.asarray([safe_float(row["negative_proximity"]) for row in bank_rows], dtype=np.float64)
    return data, pair_rows, bank_rows


def image_entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-30)
    entropy = -(probs * np.log(np.maximum(probs, 1e-30))).sum(axis=1)
    return entropy / math.log(logits.shape[1])


def pid_centers(features: np.ndarray, pids: np.ndarray) -> dict[int, np.ndarray]:
    output = {}
    for pid in sorted(set(int(value) for value in pids)):
        output[pid] = normalize(np.mean(features[pids == pid], axis=0, keepdims=True))[0]
    return output


def image_value(data: dict[str, Any], index: int, field: str) -> float:
    row = data["bank_by_index"][int(index)]
    if field in row:
        return safe_float(row[field])
    return float("nan")


def pairwise_feature_values(data: dict[str, Any], anchor: int, positive: int, negative: int) -> dict[str, float]:
    g = data["global_normed"]
    b = data["bio_normed"]
    a, p, n = int(anchor), int(positive), int(negative)
    values = {
        "global_pair_cosine_distance": float(1.0 - np.dot(g[a], g[p])),
        "global_pair_euclidean_distance": float(np.linalg.norm(data["global_feature"][a] - data["global_feature"][p])),
        "bio_pair_cosine_distance": float(1.0 - np.dot(b[a], b[p])),
        "global_anchor_negative_cosine_distance": float(1.0 - np.dot(g[a], g[n])),
        "bio_anchor_negative_cosine_distance": float(1.0 - np.dot(b[a], b[n])),
        "pair_camera_same": float(image_value(data, a, "camera_raw") == image_value(data, p, "camera_raw")),
        "pair_camera_delta": abs(image_value(data, a, "camera_raw") - image_value(data, p, "camera_raw")),
    }
    return values


def add_image_features(data: dict[str, Any], features: dict[str, float], prefix: str, index: int) -> None:
    index = int(index)
    for name in [
        "confidence", "true_confidence", "logit_margin", "ce_loss", "triplet_loss", "bio_loss", "correct",
        "support", "resolution", "aspect_ratio", "brightness", "brightness_std", "dark_fraction", "bright_fraction",
        "feature_norm_anomaly", "global_feature_norm", "bio_positive_cosine", "camera_raw",
        "h_id", "h_id_within_pid_percentile", "h_cloth_within_pid_percentile",
    ]:
        value = image_value(data, index, name)
        features[f"{prefix}_{name}"] = value
    features[f"{prefix}_entropy"] = float(data["entropy"][index])
    features[f"{prefix}_own_similarity"] = float(data["own_similarity"][index])
    features[f"{prefix}_negative_proximity"] = float(data["negative_proximity"][index])


def pair_feature_row(data: dict[str, Any], row: dict[str, str]) -> dict[str, float]:
    a, p, n = int(row["anchor_index"]), int(row["positive_index"]), int(row["negative_index"])
    features: dict[str, float] = {}
    add_image_features(data, features, "anchor", a)
    add_image_features(data, features, "positive", p)
    add_image_features(data, features, "negative", n)
    # Pair-level absolute/minimum/mean features are cheap and avoid making a
    # proxy memorize which side was named anchor.
    for base in [
        "confidence", "true_confidence", "logit_margin", "ce_loss", "triplet_loss", "bio_loss", "correct", "entropy",
        "own_similarity", "negative_proximity", "resolution", "aspect_ratio", "brightness", "brightness_std",
        "dark_fraction", "bright_fraction", "feature_norm_anomaly", "global_feature_norm", "bio_positive_cosine",
        "h_id", "h_id_within_pid_percentile", "h_cloth_within_pid_percentile",
    ]:
        av, pv = features[f"anchor_{base}"], features[f"positive_{base}"]
        features[f"pair_{base}_min"] = float(np.nanmin([av, pv])) if np.isfinite([av, pv]).any() else float("nan")
        features[f"pair_{base}_mean"] = float(np.nanmean([av, pv])) if np.isfinite([av, pv]).any() else float("nan")
        features[f"pair_{base}_absdiff"] = abs(av - pv) if np.isfinite(av) and np.isfinite(pv) else float("nan")
    features.update(pairwise_feature_values(data, a, p, n))
    # Frozen Phase-5E relation-level signals.  They are forward/feature
    # values, not the gradient target and not gradient-derived features.
    for name in [
        "positive_candidate_h_id", "positive_candidate_h_cloth", "positive_bio_state_distance",
        "positive_clothing_state_distance", "negative_center_distance", "anchor_negative_proximity",
        "anchor_identity_margin", "positive_euclidean_distance", "negative_euclidean_distance",
        "positive_cosine_similarity", "positive_cosine_distance", "negative_cosine_similarity",
        "negative_cosine_distance", "triplet_margin", "triplet_loss", "margin_violation",
    ]:
        features[f"relation_{name}"] = safe_float(row[name])
    features["relation_negative_pid_center_proximity"] = 1.0 - safe_float(row["negative_center_distance"])
    features["relation_positive_h_id_minus_h_cloth"] = safe_float(row["positive_candidate_h_id"]) - safe_float(row["positive_candidate_h_cloth"])
    features["relation_negative_pid_matches_anchor_nearest"] = float(int(row["negative_pid"]) == int(row["anchor_nearest_negative_pid"]))
    features["relation_anchor_positive_camera_same"] = float(features["anchor_camera_raw"] == features["positive_camera_raw"])
    features["relation_anchor_positive_camera_delta"] = abs(features["anchor_camera_raw"] - features["positive_camera_raw"])
    # Explicit pair symmetry signals.  Geometric distance is symmetric; the
    # image-level losses expose whether the two endpoints are asymmetric under
    # the frozen classifier without pretending the distance has direction.
    features["symmetry_triplet_loss_absdiff"] = features["pair_triplet_loss_absdiff"]
    features["symmetry_ce_absdiff"] = features["pair_ce_loss_absdiff"]
    features["symmetry_bio_loss_absdiff"] = features["pair_bio_loss_absdiff"]
    return features


def feature_group(name: str) -> str:
    if name.startswith("relation_positive_candidate_h_id") or name.startswith("relation_triplet") or name.startswith("relation_positive_") or name in {"global_pair_cosine_distance", "global_pair_euclidean_distance", "bio_pair_cosine_distance"}:
        return "hardness_geometry"
    if "confidence" in name or "entropy" in name or "logit_margin" in name or name.endswith("_correct") or "correct" in name:
        return "confidence"
    if "own_similarity" in name or "negative_proximity" in name or "identity_margin" in name or "negative_center" in name or "pid_center" in name:
        return "center_confusion"
    if "clothing" in name or "h_cloth" in name:
        return "clothing"
    if any(token in name for token in ["ce_loss", "triplet_loss", "bio_loss", "symmetry"]):
        return "loss_quality"
    if any(token in name for token in ["resolution", "aspect_ratio", "brightness", "dark_fraction", "bright_fraction", "feature_norm_anomaly", "support", "camera_"]):
        return "image_quality"
    return "other"


def build_feature_matrix(data: dict[str, Any], pair_rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    rows = []
    all_features = []
    for row in pair_rows:
        values = pair_feature_row(data, row)
        all_features.append(values)
        rows.append({
            "pid": int(row["pid"]), "anchor_index": int(row["anchor_index"]), "positive_index": int(row["positive_index"]), "negative_index": int(row["negative_index"]),
            "positive_strategy": row["positive_strategy"], "negative_strategy": row["negative_strategy"],
            "positive_path": row["positive_path"], "negative_path": row["negative_path"], "anchor_path": row["anchor_path"],
            "utility_quadrant": row["utility_quadrant"], "pair_utility": safe_float(row["pair_utility"]),
            "pair_hardness": safe_float(row["pair_hardness"]), "feature_values": values,
        })
    names = sorted(set(key for values in all_features for key in values))
    matrix = np.asarray([[values.get(name, np.nan) for name in names] for values in all_features], dtype=np.float64)
    return matrix, names, rows


def build_pid_folds(rows: list[dict[str, Any]]) -> tuple[dict[int, int], list[dict[str, Any]]]:
    pids = sorted(set(int(row["pid"]) for row in rows))
    rng = np.random.default_rng(SEED)
    shuffled = np.asarray(pids, dtype=np.int64)[rng.permutation(len(pids))]
    pid_to_fold = {int(pid): int(position % N_FOLDS) for position, pid in enumerate(shuffled)}
    fold_rows = []
    for fold in range(N_FOLDS):
        test = sorted(pid for pid in pids if pid_to_fold[pid] == fold)
        train = sorted(pid for pid in pids if pid_to_fold[pid] != fold)
        if set(test) & set(train):
            raise RuntimeError("PID leakage in fold construction")
        fold_rows.append({"fold": fold, "train_pid_json": json.dumps(train, separators=(",", ":")), "test_pid_json": json.dumps(test, separators=(",", ":")), "train_pid_count": len(train), "test_pid_count": len(test)})
    return pid_to_fold, fold_rows


def preprocess_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(x_train), x_train, medians[None, :])
    test = np.where(np.isfinite(x_test), x_test, medians[None, :])
    means = np.mean(train, axis=0)
    scales = np.std(train, axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (train - means[None, :]) / scales[None, :], (test - means[None, :]) / scales[None, :], {"median": medians, "mean": means, "scale": scales}


def closed_form_ridge_fit(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train, test, preprocess = preprocess_train_test(x_train, x_test)
    x_aug = np.concatenate([np.ones((len(train), 1), dtype=np.float64), train], axis=1)
    xtest_aug = np.concatenate([np.ones((len(test), 1), dtype=np.float64), test], axis=1)
    regularizer = np.eye(x_aug.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + regularizer
    rhs = x_aug.T @ np.asarray(y_train, dtype=np.float64)
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    scores = xtest_aug @ coef
    return scores, {**preprocess, "coefficient": coef}


def univariate_audit(matrix: np.ndarray, names: list[str], rows: list[dict[str, Any]], pid_to_fold: dict[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_indices = np.asarray([index for index, row in enumerate(rows) if row["utility_quadrant"] in {"HIGH_HARD_HIGH_UTILITY", "HIGH_HARD_LOW_UTILITY"}], dtype=np.int64)
    y = np.asarray([int(rows[index]["utility_quadrant"] == "HIGH_HARD_HIGH_UTILITY") for index in hard_indices], dtype=np.int8)
    output = []
    cv_output = []
    for col, name in enumerate(names):
        x = matrix[hard_indices, col]
        auc = safe_auc(y, x)
        reverse_auc = safe_auc(y, -x)
        best_auc = max(value for value in [auc, reverse_auc] if value is not None) if auc is not None or reverse_auc is not None else None
        output.append({
            "feature": name, "group": feature_group(name), "count": int(np.isfinite(x).sum()),
            "high_utility_mean": safe_mean(x[y == 1]), "low_utility_mean": safe_mean(x[y == 0]),
            "delta_high_minus_low": (safe_mean(x[y == 1]) - safe_mean(x[y == 0])) if safe_mean(x[y == 1]) is not None and safe_mean(x[y == 0]) is not None else None,
            "spearman": safe_corr(x, y, "spearman"), "pearson": safe_corr(x, y, "pearson"),
            "roc_auc_raw": auc, "roc_auc_best_direction": best_auc, "pr_auc_raw": safe_pr_auc(y, x),
        })
        for fold in range(N_FOLDS):
            train_mask = np.asarray([pid_to_fold[int(rows[index]["pid"])] != fold for index in hard_indices])
            test_mask = ~train_mask
            if not train_mask.any() or not test_mask.any():
                continue
            train_x = x[train_mask]
            test_x = x[test_mask]
            median = np.nanmedian(train_x) if np.isfinite(train_x).any() else 0.0
            train_x = np.where(np.isfinite(train_x), train_x, median)
            test_x = np.where(np.isfinite(test_x), test_x, median)
            train_auc = safe_auc(y[train_mask], train_x)
            direction = 1.0 if train_auc is None or train_auc >= 0.5 else -1.0
            val_score = direction * test_x
            cv_output.append({"fold": fold, "feature": name, "direction_fit_on_train": direction, "test_count": int(test_mask.sum()), "test_positive_count": int(y[test_mask].sum()), "test_auc": safe_auc(y[test_mask], val_score), "test_pr_auc": safe_pr_auc(y[test_mask], val_score)})
    return output, cv_output


def summarize_univariate_cv(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        grouped[row["feature"]].append(row)
    output = []
    for feature, values in sorted(grouped.items()):
        aucs = [safe_float(row["test_auc"]) for row in values]
        prs = [safe_float(row["test_pr_auc"]) for row in values]
        aucs = [value for value in aucs if math.isfinite(value)]
        prs = [value for value in prs if math.isfinite(value)]
        output.append({
            "feature": feature, "group": feature_group(feature), "fold_count": len(values),
            "mean_test_auc": float(np.mean(aucs)) if aucs else None,
            "std_test_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0 if aucs else None,
            "min_test_auc": float(np.min(aucs)) if aucs else None,
            "max_test_auc": float(np.max(aucs)) if aucs else None,
            "mean_test_pr_auc": float(np.mean(prs)) if prs else None,
        })
    return output


def target_by_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["utility_quadrant"] in {"HIGH_HARD_HIGH_UTILITY", "HIGH_HARD_LOW_UTILITY"}:
            grouped[(row["positive_strategy"], row["negative_strategy"])].append(row)
    output = []
    for (positive, negative), values in sorted(grouped.items()):
        high = sum(row["utility_quadrant"] == "HIGH_HARD_HIGH_UTILITY" for row in values)
        output.append({
            "positive_strategy": positive, "negative_strategy": negative, "count": len(values),
            "high_utility_count": high, "low_utility_count": len(values) - high,
            "high_utility_rate": float(high / len(values)),
        })
    return output


def feature_sets(names: list[str]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[feature_group(name)].append(index)
    all_indices = list(range(len(names)))
    selected = {
        "H_ID_ONLY": [index for index, name in enumerate(names) if name in {"relation_positive_candidate_h_id", "pair_h_id_mean", "relation_positive_h_id"}],
        "CLOTHING_ONLY": [index for index, name in enumerate(names) if "h_cloth" in name or "clothing_state_distance" in name],
        "CONFIDENCE": groups["confidence"],
        "CENTER_CONFUSION": groups["center_confusion"],
        "CLOTHING_GEOMETRY": sorted(set(groups["clothing"] + groups["hardness_geometry"])),
        "LOSS_QUALITY": groups["loss_quality"],
        "IMAGE_QUALITY": groups["image_quality"],
        "CHEAP_UTILITY_PROXY": sorted(set(groups["confidence"] + groups["center_confusion"] + groups["clothing"] + groups["image_quality"] + groups["loss_quality"] + groups["hardness_geometry"])),
        "CHEAP_FULL": all_indices,
    }
    selected = {name: values for name, values in selected.items() if values}
    return selected


def cv_proxy(matrix: np.ndarray, names: list[str], rows: list[dict[str, Any]], pid_to_fold: dict[int, int], indices: list[int], target: str, feature_set_name: str, feature_indices: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray([1.0 if rows[index]["utility_quadrant"] == "HIGH_HARD_HIGH_UTILITY" else 0.0 for index in indices], dtype=np.float64) if target == "utility_binary" else np.asarray([float(rows[index][target]) for index in indices], dtype=np.float64)
    pids = np.asarray([int(rows[index]["pid"]) for index in indices], dtype=np.int64)
    x = matrix[np.asarray(indices)[:, None], np.asarray(feature_indices)[None, :]][:, :]
    predictions = np.full(len(indices), np.nan, dtype=np.float64)
    fold_rows = []
    for fold in range(N_FOLDS):
        train_mask = np.asarray([pid_to_fold[int(pid)] != fold for pid in pids])
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        score, info = closed_form_ridge_fit(x[train_mask], y[train_mask], x[test_mask])
        predictions[test_mask] = score
        fold_y = y[test_mask]
        if target == "utility_binary":
            train_score, _ = closed_form_ridge_fit(x[train_mask], y[train_mask], x[train_mask])
            fold_rows.append(fold_metrics(fold, fold_y, score, train_score, y[train_mask], feature_set_name, len(feature_indices), int(train_mask.sum()), int(test_mask.sum())))
        else:
            fold_rows.append({"fold": fold, "feature_set": feature_set_name, "feature_count": len(feature_indices), "train_count": int(train_mask.sum()), "test_count": int(test_mask.sum()), "test_rmse": float(np.sqrt(np.mean((score - fold_y) ** 2))), "test_spearman": safe_corr(score, fold_y, "spearman"), "test_pearson": safe_corr(score, fold_y, "pearson")})
    pooled = {"feature_set": feature_set_name, "feature_count": len(feature_indices), "target": target, "count": len(indices)}
    if target == "utility_binary":
        pooled.update({"pooled_roc_auc": safe_auc(y, predictions), "pooled_pr_auc": safe_pr_auc(y, predictions), "pooled_spearman": safe_corr(predictions, y, "spearman"), "pooled_positive_rate": float(np.mean(y)), "mean_fold_roc_auc": safe_mean([row["test_roc_auc"] for row in fold_rows]), "mean_fold_pr_auc": safe_mean([row["test_pr_auc"] for row in fold_rows]), "mean_fold_top25_precision": safe_mean([row["top25_precision"] for row in fold_rows]), "mean_fold_top25_recall": safe_mean([row["top25_recall"] for row in fold_rows]), "mean_fold_low_rejection_at_median": safe_mean([row["low_rejection_at_median"] for row in fold_rows])})
    else:
        pooled.update({"pooled_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))), "pooled_spearman": safe_corr(predictions, y, "spearman"), "pooled_pearson": safe_corr(predictions, y, "pearson"), "mean_fold_rmse": safe_mean([row["test_rmse"] for row in fold_rows]), "mean_fold_spearman": safe_mean([row["test_spearman"] for row in fold_rows])})
    for position, index in enumerate(indices):
        rows[index][f"oof_{feature_set_name}"] = float(predictions[position])
    return fold_rows, pooled


def fold_metrics(fold: int, y_test: np.ndarray, score: np.ndarray, train_score: np.ndarray, y_train: np.ndarray, feature_set_name: str, feature_count: int, train_count: int, test_count: int) -> dict[str, Any]:
    median_threshold = float(np.median(train_score))
    top25_threshold = float(np.quantile(train_score, 0.75))
    selected = score >= top25_threshold
    median_selected = score >= median_threshold
    low = y_test == 0
    return {
        "fold": fold, "feature_set": feature_set_name, "feature_count": feature_count, "train_count": train_count, "test_count": test_count,
        "test_positive_count": int(y_test.sum()), "test_roc_auc": safe_auc(y_test, score), "test_pr_auc": safe_pr_auc(y_test, score),
        "top25_selected_count": int(selected.sum()), "top25_precision": safe_mean(y_test[selected]) if selected.any() else None,
        "top25_recall": float(y_test[selected].sum() / max(float(y_test.sum()), 1.0)) if selected.any() else 0.0,
        "median_selected_count": int(median_selected.sum()), "low_rejection_at_median": float(np.sum(low & ~median_selected) / max(float(low.sum()), 1.0)),
        "train_median_threshold": median_threshold, "train_top25_threshold": top25_threshold,
    }


def continuous_cv(matrix: np.ndarray, names: list[str], rows: list[dict[str, Any]], pid_to_fold: dict[int, int], feature_indices: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = list(range(len(rows)))
    fold_rows, pooled = cv_proxy(matrix, names, rows, pid_to_fold, indices, "pair_utility", "CHEAP_FULL_CONTINUOUS", feature_indices)
    return fold_rows, pooled


def representative_cases(rows: list[dict[str, Any]], feature_set_name: str) -> list[dict[str, Any]]:
    key = f"oof_{feature_set_name}"
    valid = [row for row in rows if key in row and math.isfinite(float(row[key]))]
    valid.sort(key=lambda row: float(row[key]), reverse=True)
    output = []
    for rank, row in enumerate(valid[:20], start=1):
        output.append({"rank_by_proxy": rank, "proxy_score": float(row[key]), "pid": row["pid"], "anchor_index": row["anchor_index"], "positive_index": row["positive_index"], "negative_index": row["negative_index"], "positive_strategy": row["positive_strategy"], "negative_strategy": row["negative_strategy"], "utility_quadrant": row["utility_quadrant"], "pair_hardness": row["pair_hardness"], "pair_utility": row["pair_utility"], "anchor_path": row["anchor_path"], "positive_path": row["positive_path"], "negative_path": row["negative_path"]})
    return output


def decision(univariate: list[dict[str, Any]], proxy_summary: list[dict[str, Any]], quadrants: dict[str, int]) -> dict[str, Any]:
    full = next((row for row in proxy_summary if row["feature_set"] == "CHEAP_UTILITY_PROXY"), None)
    if full is None:
        full = next((row for row in proxy_summary if row["feature_set"] == "CHEAP_FULL"), None)
    best = max(proxy_summary, key=lambda row: row.get("pooled_roc_auc") if row.get("pooled_roc_auc") is not None else -1.0) if proxy_summary else None
    auc = full.get("pooled_roc_auc") if full else None
    top25 = full.get("mean_fold_top25_precision") if full else None
    baseline = EXPECTED_HIGH_UTILITY / EXPECTED_HARD_ROWS
    if auc is not None and auc >= 0.70 and top25 is not None and top25 > baseline:
        feasibility = "SUPPORTED"
        q1 = "YES"
    elif auc is not None and auc >= 0.60 and top25 is not None and top25 > baseline:
        feasibility = "PARTIAL"
        q1 = "MIXED"
    else:
        feasibility = "INCONCLUSIVE"
        q1 = "INCONCLUSIVE"
    return {
        "q1_can_cheap_proxy_distinguish_high_hard_utility": q1,
        "feasibility": feasibility,
        "high_hard_count": quadrants.get("HIGH_HARD", EXPECTED_HARD_ROWS),
        "high_hard_high_utility": quadrants.get("HIGH_HARD_HIGH_UTILITY", EXPECTED_HIGH_UTILITY),
        "high_hard_low_utility": quadrants.get("HIGH_HARD_LOW_UTILITY", EXPECTED_LOW_UTILITY),
        "positive_rate_baseline": baseline,
        "cheap_utility_proxy_summary": full,
        "best_proxy_by_pooled_auc": best,
        "interpretation": "A cheap proxy is considered supported only when fixed PID-level OOF discrimination is strong and top-quartile precision exceeds the high-utility base rate; this rule is predeclared and no test metric is used.",
        "recommendation": "Use the proxy only as an offline pair-quality gate/diagnostic until an independent pair-mining training validation is run. Do not use U_grad, gradient norms, or gradient alignment during training.",
    }


def write_report(data: dict[str, Any], rows: list[dict[str, Any]], names: list[str], univariate: list[dict[str, Any]], univ_cv: list[dict[str, Any]], univ_cv_summary: list[dict[str, Any]], strategy_rows: list[dict[str, Any]], proxy_summary: list[dict[str, Any]], proxy_folds: list[dict[str, Any]], continuous_summary: dict[str, Any], continuous_folds: list[dict[str, Any]], decision_payload: dict[str, Any], folds: list[dict[str, Any]], quadrants: dict[str, int], cases: list[dict[str, Any]]) -> None:
    def fmt(value, digits=4):
        if value is None:
            return "—"
        try:
            value = float(value)
            return "—" if not math.isfinite(value) else f"{value:.{digits}f}"
        except Exception:
            return str(value)
    top_uni = sorted(univariate, key=lambda row: row["roc_auc_best_direction"] if row["roc_auc_best_direction"] is not None else -1.0, reverse=True)[:20]
    lines = [
        "# PHASE 5F — Hard-Pair Utility Proxy and Reliable Hard-Pair Identification", "",
        "## Scope and frozen target", "",
        "Offline, LTCC-train-only analysis. No ReID training, no optimizer construction or step, no parameter update, no DS, no DW, no sampling modification, no new gradient computation, no Full DIFFER, and no query/gallery/test retrieval. The target is frozen from Phase 5E: among high-hard relations, `U_grad > 0` is `HIGH_HARD_HIGH_UTILITY` and `U_grad <= 0` is `HIGH_HARD_LOW_UTILITY`.", "",
        f"Frozen target counts: `{quadrants.get('HIGH_HARD_HIGH_UTILITY', 0)}` high-hard/high-utility and `{quadrants.get('HIGH_HARD_LOW_UTILITY', 0)}` high-hard/low-utility, total high-hard `{quadrants.get('HIGH_HARD', 0)}`. Pair-level random splitting was not used; the proxy uses `{N_FOLDS}` PID-level folds over `{len(set(int(row['pid']) for row in rows))}` eligible PIDs.", "",
        "## CHEAP PROXY FEATURES", "",
        "Features are forward/feature-level or inexpensive image statistics: ID classifier confidence/correct-class confidence/margin/entropy/correctness; own-PID center similarity; nearest-negative proximity and identity margin; fixed Clothing state distance and H_cloth; global/Bio pair geometry; frozen CE/Triplet/Bio per-image diagnostics; endpoint symmetry; camera/resolution/aspect/brightness/feature-norm/support proxies. Gradient norm, gradient alignment, U_grad, and utility quadrant are excluded from proxy inputs.", "",
        f"The combination proxy uses a fixed closed-form ridge coefficient `lambda={RIDGE_LAMBDA}` inside each PID-level training fold. This is an offline diagnostic fit only; it is not a ReID optimizer or model update. Feature count: `{len(names)}`.", "",
        "## UNIVARIATE AUDIT", "", "Top univariate signals by best-direction ROC-AUC on the frozen high-hard target (descriptive ranking; PID-level OOF values are in `analysis/univariate_cv.csv`):", "", "| Feature | Group | n | High-U mean | Low-U mean | Spearman | Best ROC-AUC | PR-AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in top_uni:
        lines.append(f"| {row['feature']} | {row['group']} | {row['count']} | {fmt(row['high_utility_mean'])} | {fmt(row['low_utility_mean'])} | {fmt(row['spearman'])} | {fmt(row['roc_auc_best_direction'])} | {fmt(row['pr_auc_raw'])} |")
    lines += ["", "PID-level OOF univariate stability (direction is fitted on training PIDs only):", "", "| Feature | Group | Mean OOF ROC-AUC | Std | Min | Max | Mean OOF PR-AUC |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in sorted(univ_cv_summary, key=lambda item: item["mean_test_auc"] if item["mean_test_auc"] is not None else -1.0, reverse=True)[:20]:
        lines.append(f"| {row['feature']} | {row['group']} | {fmt(row['mean_test_auc'])} | {fmt(row['std_test_auc'])} | {fmt(row['min_test_auc'])} | {fmt(row['max_test_auc'])} | {fmt(row['mean_test_pr_auc'])} |")
    lines += ["", "## PID-LEVEL CROSS-VALIDATION", "", "| Proxy | Features | Pooled ROC-AUC | Pooled PR-AUC | Mean fold ROC-AUC | Mean fold PR-AUC | Mean top-25% precision | Mean top-25% recall | Mean low-utility rejection@median |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in proxy_summary:
        lines.append(f"| {row['feature_set']} | {row['feature_count']} | {fmt(row.get('pooled_roc_auc'))} | {fmt(row.get('pooled_pr_auc'))} | {fmt(row.get('mean_fold_roc_auc'))} | {fmt(row.get('mean_fold_pr_auc'))} | {fmt(row.get('mean_fold_top25_precision'))} | {fmt(row.get('mean_fold_top25_recall'))} | {fmt(row.get('mean_fold_low_rejection_at_median'))} |")
    baseline = EXPECTED_HIGH_UTILITY / EXPECTED_HARD_ROWS
    lines += ["", f"High-utility base rate among high-hard pairs: `{fmt(baseline)}`. Fold-specific top-25% thresholds are learned from the training PIDs only; no global label threshold is used for validation.", "", "## TARGET COMPOSITION AUDIT", "", "High-hard target composition by fixed Phase-5E strategy is shown to expose relation confounding; strategy labels are not proxy inputs.", "", "| Positive | Negative | n high-hard | High-U | Low-U | High-U rate |", "|---|---|---:|---:|---:|---:|"]
    for row in strategy_rows:
        lines.append(f"| {row['positive_strategy']} | {row['negative_strategy']} | {row['count']} | {row['high_utility_count']} | {row['low_utility_count']} | {fmt(row['high_utility_rate'])} |")
    lines += ["", "## CONTINUOUS UTILITY CHECK", "", f"A supplementary closed-form ridge regression predicts the continuous frozen Phase-5E `pair_utility=U_grad` over all 384 pair probes. Pooled RMSE `{fmt(continuous_summary.get('pooled_rmse'))}`, Spearman `{fmt(continuous_summary.get('pooled_spearman'))}`, Pearson `{fmt(continuous_summary.get('pooled_pearson'))}`; fold details are in `analysis/continuous_cv_folds.csv`. This is diagnostic association, not a training objective.", "", "## REPRESENTATIVE PREDICTIONS", "", f"Top OOF cases from `{cases[0]['proxy_score'] if cases else '—'}` proxy are saved in `analysis/representative_cases.csv`; paths are train-only. These cases retain PID, image filenames, clothes IDs through the source pair bank, utility label, hardness, and frozen U_grad.", "", "## FOLD AUDIT", ""]
    for fold in folds:
        lines.append(f"- Fold {fold['fold']}: train PIDs `{fold['train_pid_count']}`, test PIDs `{fold['test_pid_count']}`; PID lists are in `analysis/pid_folds.csv`.")
    lines += ["", "## FINAL DIAGNOSIS", "", f"Q1 — Can a cheap forward/feature proxy distinguish high-hard/high-utility from high-hard/low-utility pairs? **{decision_payload['q1_can_cheap_proxy_distinguish_high_hard_utility']}**.", f"Feasibility: **{decision_payload['feasibility']}**. Frozen counts are `{decision_payload['high_hard_high_utility']}/{decision_payload['high_hard_low_utility']}`; high-utility base rate `{fmt(decision_payload['positive_rate_baseline'])}`.", f"Interpretation: {decision_payload['interpretation']}", f"Recommendation: {decision_payload['recommendation']}", "", "## ARTIFACTS", "", f"- Frozen feature-bank source manifest: `{OUT_ROOT / 'feature_bank/source_manifest.json'}`", f"- Frozen Phase-5D feature bank: `{P5D_BANK}`", f"- Univariate audit: `{OUT_ROOT / 'analysis/univariate_metrics.csv'}`", f"- PID-level CV: `{OUT_ROOT / 'analysis/proxy_cv_summary.csv'}`", f"- Proxy predictions: `{OUT_ROOT / 'analysis/proxy_oof_predictions.csv'}`", f"- Pair feature matrix: `{OUT_ROOT / 'pair_features/pair_features.csv'}`", f"- Manifest: `{OUT_ROOT / 'PHASE5F_MANIFEST.json'}`", "", "STOP: no DW, DS, sampling modification, ReID training, optimizer step, new gradient utility computation, Full DIFFER, or test retrieval was started."]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
    (OUT_ROOT / "git_diff.patch").write_bytes(diff)
    (OUT_ROOT / "git_status.txt").write_text(subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout, encoding="utf-8")
    print("LOAD_FROZEN_PHASE5E_TARGET", flush=True)
    data, pair_rows, bank_rows = load_data()
    write_json(OUT_ROOT / "feature_bank/source_manifest.json", {
        "role": "frozen input source; no feature-bank copy or recomputation",
        "phase5d_feature_bank_npz": {"path": P5D_BANK, "sha256": sha256_file(P5D_BANK), "size": P5D_BANK.stat().st_size},
        "phase5d_feature_bank_csv": {"path": P5D_BANK_CSV, "sha256": sha256_file(P5D_BANK_CSV), "size": P5D_BANK_CSV.stat().st_size},
        "phase5e_pair_target_csv": {"path": P5E_QUADRANTS, "sha256": sha256_file(P5E_QUADRANTS), "size": P5E_QUADRANTS.stat().st_size},
        "scope": "LTCC train only; frozen Phase-5D forward feature bank and frozen Phase-5E pair utility labels",
    })
    matrix, names, rows = build_feature_matrix(data, pair_rows)
    pair_feature_rows = []
    for row, values in zip(rows, matrix):
        pair_feature_rows.append({"pid": row["pid"], "anchor_index": row["anchor_index"], "positive_index": row["positive_index"], "negative_index": row["negative_index"], "positive_strategy": row["positive_strategy"], "negative_strategy": row["negative_strategy"], "utility_quadrant": row["utility_quadrant"], "pair_hardness": row["pair_hardness"], "pair_utility": row["pair_utility"], **{name: value for name, value in zip(names, values)}})
    write_csv(OUT_ROOT / "pair_features/pair_features.csv", pair_feature_rows)
    write_json(OUT_ROOT / "pair_features/feature_definitions.json", {"feature_count": len(names), "features": [{"name": name, "group": feature_group(name)} for name in names], "excluded": ["gradient_norm_*", "gradient_alignment_*", "triplet_gradient_alignment_*", "pair_utility", "utility_quadrant", "hardness_threshold", "utility_threshold"], "input_scope": "frozen Phase5D feature bank plus frozen Phase5E pair forward metrics; LTCC train only"})

    pid_to_fold, fold_rows = build_pid_folds(rows)
    write_csv(OUT_ROOT / "analysis/pid_folds.csv", fold_rows)
    hard_indices = [index for index, row in enumerate(rows) if row["utility_quadrant"] in {"HIGH_HARD_HIGH_UTILITY", "HIGH_HARD_LOW_UTILITY"}]
    quadrants = defaultdict(int)
    for row in rows:
        quadrants[row["utility_quadrant"]] += 1
    quadrants["HIGH_HARD"] = quadrants.get("HIGH_HARD_HIGH_UTILITY", 0) + quadrants.get("HIGH_HARD_LOW_UTILITY", 0)
    if quadrants["HIGH_HARD_HIGH_UTILITY"] != EXPECTED_HIGH_UTILITY or quadrants["HIGH_HARD_LOW_UTILITY"] != EXPECTED_LOW_UTILITY:
        raise RuntimeError(f"frozen Phase5E target counts changed: {dict(quadrants)}")
    univariate, univ_cv = univariate_audit(matrix, names, rows, pid_to_fold)
    write_csv(OUT_ROOT / "analysis/univariate_metrics.csv", univariate)
    write_csv(OUT_ROOT / "analysis/univariate_cv.csv", univ_cv)
    univ_cv_summary = summarize_univariate_cv(univ_cv)
    write_csv(OUT_ROOT / "analysis/univariate_cv_summary.csv", univ_cv_summary)
    strategy_rows = target_by_strategy(rows)
    write_csv(OUT_ROOT / "analysis/target_by_strategy.csv", strategy_rows)
    sets = feature_sets(names)
    proxy_summary = []
    proxy_folds = []
    for set_name, feature_indices in sets.items():
        fold_result, summary_result = cv_proxy(matrix, names, rows, pid_to_fold, hard_indices, "utility_binary", set_name, feature_indices)
        proxy_folds.extend(fold_result)
        proxy_summary.append(summary_result)
    write_csv(OUT_ROOT / "analysis/proxy_cv_folds.csv", proxy_folds)
    write_csv(OUT_ROOT / "analysis/proxy_cv_summary.csv", proxy_summary)
    prediction_fields = ["pid", "anchor_index", "positive_index", "negative_index", "positive_strategy", "negative_strategy", "utility_quadrant", "pair_hardness", "pair_utility"] + [f"oof_{name}" for name in sets]
    prediction_rows = [{key: row.get(key) for key in prediction_fields} for row in rows]
    write_csv(OUT_ROOT / "analysis/proxy_oof_predictions.csv", prediction_rows, prediction_fields)
    continuous_folds, continuous_summary = continuous_cv(matrix, names, rows, pid_to_fold, sets.get("CHEAP_FULL", list(range(len(names)))))
    write_csv(OUT_ROOT / "analysis/continuous_cv_folds.csv", continuous_folds)
    write_json(OUT_ROOT / "analysis/continuous_cv_summary.json", continuous_summary)
    selected_proxy = "CHEAP_UTILITY_PROXY" if any(row["feature_set"] == "CHEAP_UTILITY_PROXY" for row in proxy_summary) else "CHEAP_FULL"
    cases = representative_cases(rows, selected_proxy)
    write_csv(OUT_ROOT / "analysis/representative_cases.csv", cases)
    decision_payload = decision(univariate, proxy_summary, dict(quadrants))
    write_json(OUT_ROOT / "diagnosis.json", decision_payload)
    input_paths = [P5D_BANK, P5D_BANK_CSV, P5E_QUADRANTS, P5E_MANIFEST, P5E_GRADIENT_MANIFEST]
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "scope": "offline train-only cheap utility proxy audit",
        "prohibitions": {"reid_training": False, "optimizer_constructed": False, "optimizer_step_count": 0, "parameter_update": False, "new_gradient_computation": False, "ds": False, "dw": False, "sampling_modification": False, "full_differ": False, "grl": False, "test_retrieval": False, "query_gallery_used": False},
        "dataset": {"split": "LTCC train", "images": EXPECTED_IMAGES, "pids": EXPECTED_PIDS, "clothing_states": EXPECTED_STATES, "pair_probe_rows": len(rows), "high_hard_rows": int(quadrants["HIGH_HARD"]), "high_hard_high_utility": int(quadrants["HIGH_HARD_HIGH_UTILITY"]), "high_hard_low_utility": int(quadrants["HIGH_HARD_LOW_UTILITY"])},
        "target": "frozen Phase5E U_grad sign among Phase5E high-hard pair probes; no gradient values used as proxy features",
        "pid_cv": {"fold_count": N_FOLDS, "eligible_pid_count": len(pid_to_fold), "folds": fold_rows, "pid_overlap_check": False},
        "ridge": {"type": "closed_form_ridge_diagnostic_only", "lambda": RIDGE_LAMBDA, "optimizer": False},
        "features": {"count": len(names), "groups": {group: sum(feature_group(name) == group for name in names) for group in sorted(set(feature_group(name) for name in names))}, "names": names},
        "inputs": {str(path): {"sha256": sha256_file(path), "size": path.stat().st_size} for path in input_paths},
        "artifacts": {"feature_bank_source_manifest": OUT_ROOT / "feature_bank/source_manifest.json"},
        "decision": decision_payload,
        "git": {"status": subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout},
    }
    write_json(OUT_ROOT / "PHASE5F_MANIFEST.json", manifest)
    write_report(data, rows, names, univariate, univ_cv, univ_cv_summary, strategy_rows, proxy_summary, proxy_folds, continuous_summary, continuous_folds, decision_payload, fold_rows, dict(quadrants), cases)
    print(json.dumps({"phase": PHASE, "report": str(REPORT_PATH), "artifact_root": str(OUT_ROOT), "q1": decision_payload["q1_can_cheap_proxy_distinguish_high_hard_utility"], "feasibility": decision_payload["feasibility"], "hard_high": quadrants["HIGH_HARD_HIGH_UTILITY"], "hard_low": quadrants["HIGH_HARD_LOW_UTILITY"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
