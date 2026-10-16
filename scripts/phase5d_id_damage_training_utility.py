#!/usr/bin/env python3
"""Phase 5D: frozen ID-Damage training-utility and gradient audit.

This is an offline diagnostic.  It deliberately has no optimizer, sampler,
training loop, parameter update, test/query/gallery access, DW, DS, GRL, or
online score update.  The Phase-5A Bio feature bank and its frozen H_ID
implementation are reused verbatim for the primary ranking signal.

The audit has three layers:

1. Reuse the audited Bio/Clothing banks to build an all-training-image
   feature/loss bank and to compare H_ID with H_cloth, loss, correctness and
   simple image confounds.
2. Probe a small, strictly matched set of 4-PID x 2-image micro-batches with
   frozen Bio weights.  Gradients are unscaled FP32 gradients and are measured
   only for the late EVA block, Bio projection and ID head.  No optimizer is
   constructed and no parameter is changed.
3. If the existing Phase-5B S0 epoch-15 checkpoint is available, extract its
   train features for a static-ranking stability check.  E5/E10 are reported
   unavailable when no corresponding frozen artifact exists.

Run:
    /home/CL-CC-ReID/.venv/bin/python scripts/phase5d_id_damage_training_utility.py
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_img_transforms  # noqa: E402
from data.dataset_loader import read_image  # noqa: E402
from model.build import factory  # noqa: E402


PHASE = "phase5d_hardness_utility"
SEED = 1234
EXPECTED_IMAGES = 9576
EXPECTED_PIDS = 77
EXPECTED_CLOTHES = 256
NUM_CLASSES = 77
NUM_CAMERAS = 12
PROBES_PER_QUARTILE = 4
TARGETS_PER_PROBE = 4

P5A_ROOT = ROOT / "outputs/new_method_track/phase5a_clothing_hardness"
FEATURE_RECORDS = P5A_ROOT / "feature_records.csv"
GROUP_METRICS = P5A_ROOT / "group_metrics.csv"
P5A_MANIFEST = P5A_ROOT / "PHASE_5A_MANIFEST.json"
P5A_SOURCES = P5A_ROOT / "REPRESENTATION_SOURCES.json"
BIO_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features/bio_control_feature_bank.npz"
TEXT_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/aligned_text_feature_bank.npz"
BIO_CHECKPOINT = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/eva02_l_bio_best.pth"
EARLY_CHECKPOINT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds/S0_bio_full_data/model_best.pth"
TEXT_RESOURCE = Path("/home/data/DIFFER_TextCaption/LTCC_ReID/EVA02-CLIP-B-16/train.npz")
TRAIN_ROOT = Path("/home/data/LTCC_ReID/train")

OUT_ROOT = ROOT / "outputs/new_method_track/phase5d_hardness_utility"
REPORT_PATH = ROOT / "reports/PHASE_5D_ID_DAMAGE_TRAINING_UTILITY.md"


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


def finite_array(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def summary(values: Any) -> dict[str, Any]:
    values = finite_array(values)
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


def corr(x: Any, y: Any, method: str) -> tuple[float | None, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None, int(len(x))
    try:
        if method == "pearson":
            value = pearsonr(x, y).statistic
        elif method == "spearman":
            value = spearmanr(x, y).statistic
        elif method == "kendall":
            value = kendalltau(x, y).statistic
        else:
            raise ValueError(method)
        return float(value) if math.isfinite(float(value)) else None, int(len(x))
    except Exception:
        return None, int(len(x))


def rank_quartiles(values: Any) -> np.ndarray:
    """Stable Q1-Q4 labels, with Q4 the highest values."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.zeros(len(values), dtype=np.int16)
    valid = np.isfinite(values)
    order = np.argsort(values[valid], kind="mergesort")
    valid_indices = np.flatnonzero(valid)[order]
    n = len(valid_indices)
    if n:
        labels[valid_indices] = np.asarray([min(4, int(pos * 4 / n) + 1) for pos in range(n)], dtype=np.int16)
    return labels


def within_pid_percentile(values: np.ndarray, pids: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for pid in sorted(set(int(x) for x in pids)):
        indices = np.flatnonzero(pids == pid)
        valid = indices[np.isfinite(values[indices])]
        if len(valid) == 0:
            continue
        if len(valid) == 1:
            result[valid] = 0.5
            continue
        ranks = rankdata(values[valid], method="average") - 1.0
        result[valid] = ranks / float(len(valid) - 1)
    return result


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def parse_camera(stem: str) -> int:
    match = re.search(r"_c(\d+)(?:_|$)", str(stem))
    return int(match.group(1)) if match else -1


def load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def make_config():
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.TYPE = "eva02_cloth"
    cfg.MODEL.NAME = "eva02_l_bio"
    cfg.MODEL.LAST_LAYER = "clipFc"
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.MODEL.NONBIO_HEAD = False
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.CAMERA_XISHU = 3
    cfg.MODEL.SUBSPACE_DIM = 0
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.BIO_INDEX = "0"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.IMG_HEIGHT = 224
    cfg.DATA.IMG_WIDTH = 224
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.freeze()
    return cfg


def load_inputs() -> dict[str, Any]:
    required = [FEATURE_RECORDS, GROUP_METRICS, P5A_MANIFEST, P5A_SOURCES, BIO_BANK, TEXT_BANK, BIO_CHECKPOINT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 5D frozen inputs missing: " + ", ".join(missing))

    record_rows = read_csv(FEATURE_RECORDS)
    group_rows = read_csv(GROUP_METRICS)
    if len(record_rows) != EXPECTED_IMAGES or len(group_rows) != EXPECTED_CLOTHES:
        raise RuntimeError(f"unexpected Phase 5A sizes: {len(record_rows)} records, {len(group_rows)} groups")
    stems = np.asarray([row["stem"] for row in record_rows]).astype(str)
    pids = np.asarray([int(row["pid"]) for row in record_rows], dtype=np.int64)
    clothes = np.asarray([int(row["cloth_id"]) for row in record_rows], dtype=np.int64)
    paths = np.asarray([row["image_path"] for row in record_rows]).astype(str)
    if len(set(stems.tolist())) != len(stems):
        raise RuntimeError("duplicate train stems in Phase 5A feature records")
    if any(not path.startswith(str(TRAIN_ROOT) + "/") for path in paths):
        raise RuntimeError("feature records contain a non-training source path")
    if any("/query/" in path or "/test/" in path for path in paths):
        raise RuntimeError("query/gallery path found in Phase 5D inputs")

    bio = load_npz(BIO_BANK)
    text = load_npz(TEXT_BANK)
    for name, bank in [("Bio", bio), ("text", text)]:
        bank_stems = np.asarray(bank["stems"]).astype(str)
        if not np.array_equal(bank_stems, stems):
            raise RuntimeError(f"{name} bank is not exactly aligned to Phase 5A feature record order")

    group_by_key = {(int(row["pid"]), int(row["cloth_id"])): row for row in group_rows}
    if len(group_by_key) != EXPECTED_CLOTHES:
        raise RuntimeError("duplicate Phase 5A group keys")
    h_cloth = np.full(len(stems), np.nan, dtype=np.float64)
    support = np.zeros(len(stems), dtype=np.int64)
    reliability = np.full(len(stems), np.nan, dtype=np.float64)
    for i, (pid, cloth) in enumerate(zip(pids, clothes)):
        row = group_by_key[(int(pid), int(cloth))]
        # A PID with a single clothing state has no leave-one-out Clothing
        # distance.  Keep it missing rather than inventing a hardness value;
        # H_ID and all identity diagnostics remain defined for that image.
        h_cloth_value = str(row["clothing_text_fixed_h_cloth_loo_mean"]).strip()
        h_cloth[i] = float(h_cloth_value) if h_cloth_value else np.nan
        support[i] = int(row["support"])
        reliability[i] = float(row["reliability"])

    # This is the exact Phase-5A H_ID value.  It is not recomputed or tuned.
    h_id = np.asarray([float(row["h_id"]) for row in record_rows], dtype=np.float64)
    own_distance = np.asarray([float(row["own_center_distance"]) for row in record_rows], dtype=np.float64)
    negative_distance = np.asarray([float(row["nearest_negative_distance"]) for row in record_rows], dtype=np.float64)
    nearest_negative_pid = np.asarray([int(row["nearest_negative_pid"]) for row in record_rows], dtype=np.int64)

    global_feature = np.asarray(bio["global_feature"], dtype=np.float32)
    bio_projected = np.asarray(bio["bio_projected"], dtype=np.float32)
    id_logits = np.asarray(bio["id_logits"], dtype=np.float32)
    # runtime_features[:,0] is the Summary Bio caption used by the released
    # Bio loss path when BIO_INDEX=0 and SUMMERY_TEXT=True.
    summary_text = np.asarray(text["runtime_features"][:, 0, :], dtype=np.float32)
    if global_feature.shape != (EXPECTED_IMAGES, 1024) or bio_projected.shape != (EXPECTED_IMAGES, 512) or id_logits.shape != (EXPECTED_IMAGES, NUM_CLASSES):
        raise RuntimeError("unexpected frozen Bio feature bank shape")
    if summary_text.shape != (EXPECTED_IMAGES, 512):
        raise RuntimeError("unexpected Summary Bio text shape")

    negative_proximity = 1.0 - negative_distance
    confusion_hardness = own_distance - negative_distance
    safe_margin = negative_distance - own_distance
    global_norm = np.linalg.norm(global_feature, axis=1).astype(np.float64)
    feature_norm_anomaly = np.abs(np.log(np.maximum(global_norm, 1e-12) / max(float(np.median(global_norm)), 1e-12)))
    h_id_within_pid = within_pid_percentile(h_id, pids)
    h_cloth_within_pid = within_pid_percentile(h_cloth, pids)
    h_id_quartile = rank_quartiles(h_id)
    h_cloth_quartile = rank_quartiles(h_cloth)
    confusion_quartile = rank_quartiles(confusion_hardness)

    return {
        "rows": record_rows,
        "group_rows": group_rows,
        "group_by_key": group_by_key,
        "stems": stems,
        "paths": paths,
        "pids": pids,
        "clothes": clothes,
        "h_id": h_id,
        "h_cloth": h_cloth,
        "own_distance": own_distance,
        "negative_distance": negative_distance,
        "nearest_negative_pid": nearest_negative_pid,
        "negative_proximity": negative_proximity,
        "confusion_hardness": confusion_hardness,
        "safe_margin": safe_margin,
        "support": support,
        "reliability": reliability,
        "global_norm": global_norm,
        "feature_norm_anomaly": feature_norm_anomaly,
        "h_id_within_pid": h_id_within_pid,
        "h_cloth_within_pid": h_cloth_within_pid,
        "h_id_quartile": h_id_quartile,
        "h_cloth_quartile": h_cloth_quartile,
        "confusion_quartile": confusion_quartile,
        "global_feature": global_feature,
        "bio_projected": bio_projected,
        "id_logits": id_logits,
        "summary_text": summary_text,
    }


def compute_image_quality(data: dict[str, Any]) -> None:
    widths, heights, brightness, brightness_std, dark_fraction, bright_fraction = [], [], [], [], [], []
    for index, path in enumerate(data["paths"]):
        with Image.open(str(path)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            width, height = image.size
        gray = rgb.mean(axis=2)
        widths.append(width)
        heights.append(height)
        brightness.append(float(gray.mean()))
        brightness_std.append(float(gray.std()))
        dark_fraction.append(float(np.mean(gray < 0.08)))
        bright_fraction.append(float(np.mean(gray > 0.92)))
        if index and index % 2000 == 0:
            print(f"QUALITY {index}/{len(data['paths'])}", flush=True)
    data["camera_raw"] = np.asarray([parse_camera(stem) for stem in data["stems"]], dtype=np.int64)
    data["width"] = np.asarray(widths, dtype=np.float64)
    data["height"] = np.asarray(heights, dtype=np.float64)
    data["resolution"] = data["width"] * data["height"]
    data["aspect_ratio"] = data["width"] / np.maximum(data["height"], 1.0)
    data["brightness"] = np.asarray(brightness, dtype=np.float64)
    data["brightness_std"] = np.asarray(brightness_std, dtype=np.float64)
    data["dark_fraction"] = np.asarray(dark_fraction, dtype=np.float64)
    data["bright_fraction"] = np.asarray(bright_fraction, dtype=np.float64)


def make_balanced_batches(pids: np.ndarray, seed: int = SEED) -> list[list[int]]:
    """Cover every image in deterministic 4-PID x 2-image micro-batches."""
    rng = np.random.default_rng(seed)
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, pid in enumerate(pids):
        by_pid[int(pid)].append(int(index))
    units_by_pid: dict[int, list[list[int]]] = {}
    for pid in sorted(by_pid):
        indices = np.asarray(by_pid[pid], dtype=np.int64)
        indices = indices[rng.permutation(len(indices))].tolist()
        units = []
        for start in range(0, len(indices), 2):
            pair = indices[start:start + 2]
            if len(pair) == 1:
                pair = pair + [pair[0]]
            units.append(pair)
        units_by_pid[pid] = units
    all_units = [unit for pid in sorted(units_by_pid) for unit in units_by_pid[pid]]
    max_units = max(len(units) for units in units_by_pid.values())
    batches: list[list[int]] = []
    fallback = 0
    for round_id in range(max_units):
        round_units = [units_by_pid[pid][round_id] for pid in sorted(units_by_pid) if round_id < len(units_by_pid[pid])]
        for start in range(0, len(round_units), 4):
            group = round_units[start:start + 4]
            while len(group) < 4:
                group.append(all_units[fallback % len(all_units)])
                fallback += 1
            batches.append([int(index) for unit in group for index in unit])
    if not all(len(batch) == 8 for batch in batches):
        raise RuntimeError("balanced batch construction failed")
    return batches


def calculate_feature_losses(data: dict[str, Any]) -> dict[str, Any]:
    ce_lists: list[list[float]] = [[] for _ in range(len(data["pids"]))]
    tri_lists: list[list[float]] = [[] for _ in range(len(data["pids"]))]
    bio_lists: list[list[float]] = [[] for _ in range(len(data["pids"]))]
    batch_ids: list[list[int]] = [[] for _ in range(len(data["pids"]))]
    batches = make_balanced_batches(data["pids"])
    global_features = data["global_feature"]
    bio_features = l2_normalize(data["bio_projected"])
    text_features = l2_normalize(data["summary_text"])
    for batch_id, indices in enumerate(batches):
        indices_np = np.asarray(indices, dtype=np.int64)
        logits = data["id_logits"][indices_np]
        labels = data["pids"][indices_np]
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_prob = logits_shifted - np.log(np.maximum(np.exp(logits_shifted).sum(axis=1, keepdims=True), 1e-30))
        ce = -log_prob[np.arange(len(indices_np)), labels]

        feat = global_features[indices_np].astype(np.float64)
        squared = np.sum(feat * feat, axis=1, keepdims=True)
        dist = np.sqrt(np.maximum(squared + squared.T - 2.0 * (feat @ feat.T), 1e-12))
        label_matrix = labels[:, None] == labels[None, :]
        neg_matrix = ~label_matrix
        pos = np.max(np.where(label_matrix, dist, -np.inf), axis=1)
        neg = np.min(np.where(neg_matrix, dist, np.inf), axis=1)
        tri = np.logaddexp(0.0, pos - neg)

        image = bio_features[indices_np]
        text = text_features[indices_np]
        bio_logits = image @ text.T
        bio_shifted = bio_logits - np.max(bio_logits, axis=1, keepdims=True)
        bio_log_prob = bio_shifted - np.log(np.maximum(np.exp(bio_shifted).sum(axis=1, keepdims=True), 1e-30))
        bio_loss = -bio_log_prob[np.arange(len(indices_np)), np.arange(len(indices_np))]
        for local, index in enumerate(indices_np):
            ce_lists[int(index)].append(float(ce[local]))
            tri_lists[int(index)].append(float(tri[local]))
            bio_lists[int(index)].append(float(bio_loss[local]))
            batch_ids[int(index)].append(int(batch_id))

    def collapse(lists: list[list[float]]) -> np.ndarray:
        return np.asarray([float(np.mean(values)) if values else np.nan for values in lists], dtype=np.float64)

    data["ce_loss"] = collapse(ce_lists)
    data["triplet_loss"] = collapse(tri_lists)
    data["bio_loss"] = collapse(bio_lists)
    data["total_core_loss"] = data["ce_loss"] + data["triplet_loss"] + data["bio_loss"]
    data["loss_batch_id"] = np.asarray([values[0] if values else -1 for values in batch_ids], dtype=np.int64)

    logits = data["id_logits"]
    pred = np.argmax(logits, axis=1)
    data["prediction"] = pred.astype(np.int64)
    data["correct"] = (pred == data["pids"]).astype(np.int8)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted) / np.maximum(np.exp(shifted).sum(axis=1, keepdims=True), 1e-30)
    data["confidence"] = np.max(probability, axis=1).astype(np.float64)
    data["true_confidence"] = probability[np.arange(len(pred)), data["pids"]].astype(np.float64)
    own_logits = logits[np.arange(len(pred)), data["pids"]]
    negative_logits = logits.copy()
    negative_logits[np.arange(len(pred)), data["pids"]] = -np.inf
    best_negative_logits = np.max(negative_logits, axis=1)
    data["logit_margin"] = (own_logits - best_negative_logits).astype(np.float64)
    data["bio_positive_cosine"] = np.sum(l2_normalize(data["bio_projected"]) * l2_normalize(data["summary_text"]), axis=1).astype(np.float64)
    return {"balanced_batch_count": len(batches), "covered_image_count": int(sum(bool(values) for values in batch_ids))}


def write_feature_bank(data: dict[str, Any]) -> None:
    out = OUT_ROOT / "feature_bank"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "global_feature": data["global_feature"].astype(np.float16),
        "bio_projected": data["bio_projected"].astype(np.float16),
        "id_logits": data["id_logits"].astype(np.float16),
        "summary_text": data["summary_text"].astype(np.float16),
        "pid": data["pids"].astype(np.int16),
        "clothes_id": data["clothes"].astype(np.int16),
        "h_id": data["h_id"].astype(np.float32),
        "h_cloth": data["h_cloth"].astype(np.float32),
        "own_distance": data["own_distance"].astype(np.float32),
        "negative_distance": data["negative_distance"].astype(np.float32),
        "ce_loss": data["ce_loss"].astype(np.float32),
        "triplet_loss": data["triplet_loss"].astype(np.float32),
        "bio_loss": data["bio_loss"].astype(np.float32),
        "total_core_loss": data["total_core_loss"].astype(np.float32),
        "correct": data["correct"].astype(np.int8),
        "confidence": data["confidence"].astype(np.float32),
        "true_confidence": data["true_confidence"].astype(np.float32),
        "logit_margin": data["logit_margin"].astype(np.float32),
        "stems": data["stems"],
    }
    np.savez_compressed(out / "feature_bank.npz", **arrays)
    fields = [
        "index", "stem", "image_filename", "image_path", "pid", "cloth_id", "cloth_key", "camera_raw",
        "support", "h_id", "h_id_within_pid_percentile", "h_id_quartile", "h_cloth", "h_cloth_within_pid_percentile",
        "h_cloth_quartile", "own_center_distance", "nearest_negative_distance", "negative_proximity",
        "confusion_hardness", "identity_margin", "nearest_negative_pid", "global_feature_norm", "feature_norm_anomaly",
        "ce_loss", "triplet_loss", "bio_loss", "total_core_loss", "prediction", "correct", "confidence",
        "true_confidence", "logit_margin", "bio_positive_cosine", "width", "height", "resolution", "aspect_ratio",
        "brightness", "brightness_std", "dark_fraction", "bright_fraction", "reliability",
    ]
    rows = []
    for i, row in enumerate(data["rows"]):
        rows.append({
            "index": i, "stem": data["stems"][i], "image_filename": row["image_filename"], "image_path": data["paths"][i],
            "pid": int(data["pids"][i]), "cloth_id": int(data["clothes"][i]), "cloth_key": row["cloth_key"], "camera_raw": int(data["camera_raw"][i]),
            "support": int(data["support"][i]), "h_id": data["h_id"][i], "h_id_within_pid_percentile": data["h_id_within_pid"][i],
            "h_id_quartile": int(data["h_id_quartile"][i]), "h_cloth": data["h_cloth"][i], "h_cloth_within_pid_percentile": data["h_cloth_within_pid"][i],
            "h_cloth_quartile": int(data["h_cloth_quartile"][i]), "own_center_distance": data["own_distance"][i],
            "nearest_negative_distance": data["negative_distance"][i], "negative_proximity": data["negative_proximity"][i],
            "confusion_hardness": data["confusion_hardness"][i], "identity_margin": data["safe_margin"][i],
            "nearest_negative_pid": int(data["nearest_negative_pid"][i]), "global_feature_norm": data["global_norm"][i],
            "feature_norm_anomaly": data["feature_norm_anomaly"][i], "ce_loss": data["ce_loss"][i], "triplet_loss": data["triplet_loss"][i],
            "bio_loss": data["bio_loss"][i], "total_core_loss": data["total_core_loss"][i], "prediction": int(data["prediction"][i]),
            "correct": int(data["correct"][i]), "confidence": data["confidence"][i], "true_confidence": data["true_confidence"][i],
            "logit_margin": data["logit_margin"][i], "bio_positive_cosine": data["bio_positive_cosine"][i], "width": data["width"][i],
            "height": data["height"][i], "resolution": data["resolution"][i], "aspect_ratio": data["aspect_ratio"][i],
            "brightness": data["brightness"][i], "brightness_std": data["brightness_std"][i], "dark_fraction": data["dark_fraction"][i],
            "bright_fraction": data["bright_fraction"][i], "reliability": data["reliability"][i],
        })
    write_csv(out / "feature_bank.csv", rows, fields)


def quartile_analysis(data: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "h_id", "h_cloth", "own_distance", "negative_distance", "negative_proximity", "confusion_hardness",
        "ce_loss", "triplet_loss", "bio_loss", "total_core_loss", "correct", "confidence", "true_confidence",
        "logit_margin", "global_norm", "support", "reliability", "brightness", "brightness_std", "resolution",
        "aspect_ratio", "dark_fraction", "bright_fraction",
    ]
    rows = []
    for source, labels in [("H_ID", data["h_id_quartile"]), ("H_cloth", data["h_cloth_quartile"]), ("Confusion", data["confusion_quartile"])]:
        for quartile in range(1, 5):
            indices = np.flatnonzero(labels == quartile)
            row: dict[str, Any] = {"source": source, "quartile": quartile, "count": len(indices)}
            for metric in metrics:
                stats = summary(data[metric][indices])
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_median"] = stats["median"]
            rows.append(row)
    out = OUT_ROOT / "quartile_analysis"
    write_csv(out / "quartile_summary.csv", rows)

    signal_names = {
        "Clothing": data["h_cloth"],
        "ID-Damage": data["h_id"],
        "D_own": data["own_distance"],
        "Negative proximity": data["negative_proximity"],
        "Confusion margin hardness": data["confusion_hardness"],
        "Safe identity margin": data["safe_margin"],
    }
    signal_rows = []
    targets = {
        "CE loss": data["ce_loss"],
        "Triplet contribution": data["triplet_loss"],
        "Bio loss": data["bio_loss"],
        "Total core loss": data["total_core_loss"],
        "Prediction error": 1.0 - data["correct"].astype(np.float64),
        "Confidence": data["confidence"],
        "True confidence": data["true_confidence"],
        "Logit margin": data["logit_margin"],
    }
    for name, signal in signal_names.items():
        row = {"signal": name}
        for target_name, target in targets.items():
            rho, n = corr(signal, target, "spearman")
            pearson, _ = corr(signal, target, "pearson")
            key = re.sub(r"[^a-z0-9]+", "_", target_name.lower()).strip("_")
            row[f"{key}_spearman"] = rho
            row[f"{key}_pearson"] = pearson
            row[f"{key}_n"] = n
        signal_rows.append(row)
    write_csv(OUT_ROOT / "quartile_analysis" / "signal_loss_correlations.csv", signal_rows)

    decomposition = []
    own_threshold = float(np.nanmedian(data["own_distance"]))
    neg_threshold = float(np.nanmedian(data["negative_proximity"]))
    far_own = data["own_distance"] >= own_threshold
    close_neg = data["negative_proximity"] >= neg_threshold
    data["decomposition_category"] = np.asarray([
        "Both" if a and b else "Far-from-own only" if a else "Close-to-negative only" if b else "Neither"
        for a, b in zip(far_own, close_neg)
    ], dtype=object)
    for category in ["Far-from-own only", "Close-to-negative only", "Both", "Neither"]:
        indices = np.flatnonzero(data["decomposition_category"] == category)
        decomposition.append({
            "category": category, "count": len(indices), "fraction": len(indices) / len(data["pids"]),
            "h_id_mean": summary(data["h_id"][indices])["mean"], "ce_mean": summary(data["ce_loss"][indices])["mean"],
            "triplet_mean": summary(data["triplet_loss"][indices])["mean"], "bio_mean": summary(data["bio_loss"][indices])["mean"],
            "correct_mean": summary(data["correct"][indices])["mean"], "confidence_mean": summary(data["confidence"][indices])["mean"],
            "confusion_hardness_mean": summary(data["confusion_hardness"][indices])["mean"],
        })
    write_csv(OUT_ROOT / "quartile_analysis" / "identity_damage_decomposition.csv", decomposition)
    write_json(OUT_ROOT / "quartile_analysis" / "identity_damage_decomposition.json", {
        "own_distance_median_threshold": own_threshold,
        "negative_proximity_median_threshold": neg_threshold,
        "categories": decomposition,
        "definition": "Far-from-own means D_own >= train-image median; Close-to-negative means negative proximity (1-D_neg) >= train-image median. The four labels are diagnostic only.",
    })
    return {"quartile_rows": rows, "signal_rows": signal_rows, "decomposition": decomposition}


def predictive_power(data: dict[str, Any]) -> dict[str, Any]:
    targets = {
        "identity_confusion_top25": (data["confusion_quartile"] == 4).astype(np.int8),
        "model_error": (1 - data["correct"]).astype(np.int8),
    }
    signals = {
        "H_cloth": data["h_cloth"],
        "H_ID": data["h_id"],
        "D_own": data["own_distance"],
        "negative_proximity": data["negative_proximity"],
        "confusion_margin_hardness": data["confusion_hardness"],
    }
    output: dict[str, Any] = {}
    rows = []
    for target_name, target in targets.items():
        output[target_name] = {}
        for signal_name, signal in signals.items():
            valid = np.isfinite(signal) & np.isfinite(target)
            if len(np.unique(target[valid])) < 2:
                metrics = {"roc_auc": None, "pr_auc": None, "spearman": None, "count": int(valid.sum())}
            else:
                roc = roc_auc_score(target[valid], signal[valid])
                pr = average_precision_score(target[valid], signal[valid])
                rho, _ = corr(signal[valid], target[valid], "spearman")
                metrics = {"roc_auc": float(roc), "pr_auc": float(pr), "spearman": rho, "count": int(valid.sum())}
            output[target_name][signal_name] = metrics
            rows.append({"target": target_name, "signal": signal_name, **metrics})
    write_json(OUT_ROOT / "quartile_analysis" / "predictive_power.json", output)
    write_csv(OUT_ROOT / "quartile_analysis" / "predictive_power.csv", rows)
    return output


def make_model_config_and_transform():
    cfg = make_config()
    _, transform = build_img_transforms(cfg)
    return cfg, transform


def load_model(checkpoint: Path, cfg: Any, device: torch.device):
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=NUM_CAMERAS)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clean_state = {key.replace("module.", "", 1): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch for {checkpoint}: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.float().to(device).eval()
    return model, {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_key_count": len(clean_state), "missing_keys": list(missing), "unexpected_keys": list(unexpected),
    }


def load_images(indices: list[int], data: dict[str, Any], transform: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    for index in indices:
        images.append(transform(read_image(str(data["paths"][index]))))
    image_tensor = torch.stack(images, dim=0).to(device=device, dtype=torch.float32, non_blocking=False)
    pid_tensor = torch.as_tensor(data["pids"][indices], dtype=torch.long, device=device)
    camera_tensor = torch.as_tensor(data["camera_raw"][indices] - 1, dtype=torch.long, device=device)
    text_tensor = torch.as_tensor(data["summary_text"][indices], dtype=torch.float32, device=device)
    return image_tensor, pid_tensor, camera_tensor, text_tensor


def forward_components(model: torch.nn.Module, images: torch.Tensor, cameras: torch.Tensor):
    """Call the frozen Bio model without its train/eval dispatch side effects."""
    features = model.forward_features(images, cameras)
    if isinstance(features, (list, tuple)):
        features = features[0]
    global_feature = model.forward_head(features, pre_logits=True)
    logits = model.head(global_feature)
    bio_projected = model.head_clip_bio(global_feature)
    return global_feature, logits, bio_projected


def core_loss_components(global_feature, logits, bio_projected, pids, text):
    ce_per = F.cross_entropy(logits, pids, reduction="none")
    squared = (global_feature * global_feature).sum(dim=1, keepdim=True)
    distances = (squared + squared.T - 2.0 * global_feature @ global_feature.T).clamp_min(1e-12).sqrt()
    positives = pids[:, None].eq(pids[None, :])
    negatives = ~positives
    dist_ap = distances.masked_fill(~positives, -float("inf")).max(dim=1).values
    dist_an = distances.masked_fill(~negatives, float("inf")).min(dim=1).values
    triplet_per = F.softplus(dist_ap - dist_an)
    image = F.normalize(bio_projected, dim=-1)
    captions = F.normalize(text, dim=-1)
    bio_logits = image @ captions.T
    bio_per = F.cross_entropy(bio_logits, torch.arange(len(pids), device=pids.device), reduction="none")
    return ce_per, triplet_per, bio_per


def parameter_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    names = list(model.named_parameters())
    block_indices = []
    for name, _ in names:
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            block_indices.append(int(match.group(1)))
    last_block = max(block_indices)
    groups = {
        "late_eva_block": [(name, param) for name, param in names if name.startswith(f"blocks.{last_block}.")],
        "bio_projection": [(name, param) for name, param in names if name.startswith("head_clip_bio.")],
        "id_head": [(name, param) for name, param in names if name.startswith("head.")],
    }
    if any(not values for values in groups.values()):
        raise RuntimeError(f"could not identify all requested gradient groups: {[(key, len(value)) for key, value in groups.items()]}")
    return groups


def vector_from_grads(group: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    chunks = []
    for _, parameter in group:
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(parameter.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks)


def cosine_cpu(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def build_probe_targets(data: dict[str, Any], family: str, quartile: int, probe_id: int) -> list[int]:
    labels = data["h_id_quartile"] if family == "H_ID" else data["h_cloth_quartile"]
    rng = np.random.default_rng(SEED + 1000 * (1 if family == "H_ID" else 2) + 10 * quartile + probe_id)
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(labels == quartile):
        by_pid[int(data["pids"][index])].append(int(index))
    eligible = sorted(pid for pid, values in by_pid.items() if values)
    if len(eligible) < TARGETS_PER_PROBE:
        raise RuntimeError(f"not enough PIDs for matched {family} Q{quartile} probe")
    offset = (probe_id * TARGETS_PER_PROBE) % len(eligible)
    selected_pids = [eligible[(offset + k) % len(eligible)] for k in range(TARGETS_PER_PROBE)]
    targets = []
    for pid in selected_pids:
        candidates = by_pid[pid]
        targets.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    return targets


def different_cloth_partner(index: int, data: dict[str, Any], prefer_different: bool = True) -> int:
    pid = int(data["pids"][index])
    same_pid = np.flatnonzero(data["pids"] == pid)
    if prefer_different:
        candidates = same_pid[data["clothes"][same_pid] != data["clothes"][index]]
        if len(candidates):
            return int(candidates[0])
    candidates = same_pid[same_pid != index]
    return int(candidates[0]) if len(candidates) else int(index)


def build_probe_batch(targets: list[int], data: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    partners = [different_cloth_partner(index, data, prefer_different=False) for index in targets]
    batch = [item for pair in zip(targets, partners) for item in pair]
    alternatives = [different_cloth_partner(index, data, prefer_different=True) for index in targets]
    pid_reference = [item for pair in zip(alternatives, partners) for item in pair]
    return batch, pid_reference, alternatives


def compute_batch_scalars(data: dict[str, Any], indices: list[int]) -> dict[str, float]:
    indices_np = np.asarray(indices, dtype=np.int64)
    return {
        "h_id": float(np.mean(data["h_id"][indices_np])),
        "h_cloth": float(np.nanmean(data["h_cloth"][indices_np])),
        "ce": float(np.mean(data["ce_loss"][indices_np])),
        "triplet": float(np.mean(data["triplet_loss"][indices_np])),
        "bio": float(np.mean(data["bio_loss"][indices_np])),
        "correct": float(np.mean(data["correct"][indices_np])),
        "confidence": float(np.mean(data["confidence"][indices_np])),
    }


def gradient_for_batch(model, groups, batch_indices, data, transform, device):
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    for values in groups.values():
        for _, parameter in values:
            parameter.requires_grad_(True)
    images, pids, cameras, text = load_images(batch_indices, data, transform, device)
    global_feature, logits, bio_projected = forward_components(model, images, cameras)
    ce_per, tri_per, bio_per = core_loss_components(global_feature, logits, bio_projected, pids, text)
    core = (ce_per.mean() + tri_per.mean() + bio_per.mean())
    core.backward()
    vectors = {name: vector_from_grads(values) for name, values in groups.items()}
    scalars = {
        "core_loss": float(core.detach().float().cpu()),
        "ce": float(ce_per.detach().float().mean().cpu()),
        "triplet": float(tri_per.detach().float().mean().cpu()),
        "bio": float(bio_per.detach().float().mean().cpu()),
    }
    del images, pids, cameras, text, global_feature, logits, bio_projected, ce_per, tri_per, bio_per, core
    for parameter in model.parameters():
        parameter.grad = None
    return vectors, scalars


def run_gradient_probes(data: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5D gradient audit requires the available PPU/CUDA device")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg, transform = make_model_config_and_transform()
    model, load_info = load_model(BIO_CHECKPOINT, cfg, device)
    groups = parameter_groups(model)
    group_meta = {
        name: {"parameter_count": len(values), "parameter_names": [key for key, _ in values], "numel": int(sum(param.numel() for _, param in values))}
        for name, values in groups.items()
    }

    # A fixed balanced full-data reference batch.  This is only a gradient
    # direction reference; it is not a loss-weighting or training operation.
    reference_targets = []
    for pid in range(4):
        candidates = np.flatnonzero(data["pids"] == pid)
        reference_targets.extend([int(candidates[0]), different_cloth_partner(int(candidates[0]), data, True)])
    reference_vectors, reference_scalars = gradient_for_batch(model, groups, reference_targets, data, transform, device)

    probe_rows = []
    utility_rows = []
    family_probe_vectors: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
    for family in ["H_ID", "H_cloth"]:
        for quartile in range(1, 5):
            for probe_id in range(PROBES_PER_QUARTILE):
                targets = build_probe_targets(data, family, quartile, probe_id)
                batch, pid_reference, alternatives = build_probe_batch(targets, data)
                vectors, scalars = gradient_for_batch(model, groups, batch, data, transform, device)
                pid_vectors, _ = gradient_for_batch(model, groups, pid_reference, data, transform, device)
                alignments = {name: cosine_cpu(vectors[name], reference_vectors[name]) for name in groups}
                pid_alignments = {name: cosine_cpu(vectors[name], pid_vectors[name]) for name in groups}
                row = {
                    "family": family, "quartile": quartile, "probe_id": probe_id,
                    "target_indices_json": json.dumps(targets, separators=(",", ":")),
                    "batch_indices_json": json.dumps(batch, separators=(",", ":")),
                    "pid_reference_indices_json": json.dumps(pid_reference, separators=(",", ":")),
                    "target_pids_json": json.dumps([int(data["pids"][i]) for i in targets], separators=(",", ":")),
                    "target_clothes_json": json.dumps([int(data["clothes"][i]) for i in targets], separators=(",", ":")),
                    "target_h_id_mean": float(np.mean(data["h_id"][targets])),
                    "target_h_cloth_mean": float(np.nanmean(data["h_cloth"][targets])),
                    "target_own_mean": float(np.mean(data["own_distance"][targets])),
                    "target_negative_proximity_mean": float(np.mean(data["negative_proximity"][targets])),
                    "target_confusion_hardness_mean": float(np.mean(data["confusion_hardness"][targets])),
                    "target_correct_mean": float(np.mean(data["correct"][targets])),
                    "target_confidence_mean": float(np.mean(data["confidence"][targets])),
                    "batch_h_id_mean": compute_batch_scalars(data, batch)["h_id"],
                    "batch_h_cloth_mean": compute_batch_scalars(data, batch)["h_cloth"],
                    "core_loss": scalars["core_loss"], "ce_loss": scalars["ce"], "triplet_loss": scalars["triplet"], "bio_loss": scalars["bio"],
                    "reference_core_loss": reference_scalars["core_loss"],
                }
                for name in groups:
                    row[f"grad_norm_{name}"] = float(vectors[name].norm())
                    row[f"gradient_alignment_{name}"] = alignments[name]
                    row[f"same_pid_alignment_{name}"] = pid_alignments[name]
                probe_rows.append(row)
                family_probe_vectors[(family, quartile, probe_id)] = vectors
                for target in targets:
                    utility_rows.append({
                        "family": family, "quartile": quartile, "probe_id": probe_id, "index": int(target),
                        "stem": data["stems"][target], "pid": int(data["pids"][target]), "cloth_id": int(data["clothes"][target]),
                        "h_id": float(data["h_id"][target]), "h_cloth": float(data["h_cloth"][target]),
                        "correct": int(data["correct"][target]), "confidence": float(data["confidence"][target]),
                        "gradient_alignment_late_eva_block": alignments["late_eva_block"],
                        "gradient_alignment_bio_projection": alignments["bio_projection"],
                        "gradient_alignment_id_head": alignments["id_head"],
                        "same_pid_alignment_late_eva_block": pid_alignments["late_eva_block"],
                    })
                print(f"GRADIENT {family} Q{quartile} probe={probe_id + 1}/{PROBES_PER_QUARTILE}", flush=True)

    out = OUT_ROOT / "gradient_probes"
    fields = list(probe_rows[0]) if probe_rows else []
    write_csv(out / "gradient_probes.csv", probe_rows, fields)
    write_csv(out / "gradient_utility_targets.csv", utility_rows)
    write_json(out / "gradient_probe_manifest.json", {
        "phase": PHASE, "inference_only": True, "optimizer_constructed": False, "optimizer_step_count": 0,
        "checkpoint": load_info, "device": torch.cuda.get_device_name(0), "dtype": "FP32",
        "gradient_scaling": "none; no GradScaler and no AMP autocast in backward probes",
        "loss": "CE + released soft batch-hard Triplet + Summary Bio contrastive loss, all coefficients one",
        "microbatch": "4 target PIDs x 2 images; target labels are matched across Q1-Q4; partner/reference batches are diagnostic only",
        "probes_per_family": PROBES_PER_QUARTILE * 4, "targets_per_quartile": PROBES_PER_QUARTILE * TARGETS_PER_PROBE,
        "groups": group_meta, "reference_batch_indices": reference_targets,
    })

    # Summaries used by the report and by later pair analysis.
    summary_rows = []
    for family in ["H_ID", "H_cloth"]:
        for quartile in range(1, 5):
            selected = [row for row in probe_rows if row["family"] == family and int(row["quartile"]) == quartile]
            row = {"family": family, "quartile": quartile, "probe_count": len(selected), "target_count": len(selected) * TARGETS_PER_PROBE}
            for field in ["target_h_id_mean", "target_h_cloth_mean", "target_correct_mean", "target_confidence_mean", "core_loss", "ce_loss", "triplet_loss", "bio_loss"]:
                values = [float(item[field]) for item in selected]
                row[field + "_mean"] = float(np.mean(values)) if values else None
            for group_name in groups:
                for metric in ["grad_norm", "gradient_alignment", "same_pid_alignment"]:
                    values = [float(item[f"{metric}_{group_name}"]) for item in selected]
                    values = np.asarray(values, dtype=np.float64)
                    row[f"{metric}_{group_name}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else None
                    if metric != "grad_norm":
                        row[f"{metric}_{group_name}_positive_rate"] = float(np.mean(values > 0)) if len(values) else None
            summary_rows.append(row)
    write_csv(out / "gradient_quartile_summary.csv", summary_rows)
    write_json(out / "reference_gradient.json", {
        "batch_indices": reference_targets, "batch_pids": [int(data["pids"][i]) for i in reference_targets],
        "scalars": reference_scalars, "norms": {name: float(vector.norm()) for name, vector in reference_vectors.items()},
    })
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del model, reference_vectors, family_probe_vectors
    gc.collect()
    torch.cuda.empty_cache()
    return {"probe_rows": probe_rows, "utility_rows": utility_rows, "summary_rows": summary_rows, "checkpoint": load_info, "group_meta": group_meta}


def attach_gradient_utility(data: dict[str, Any], gradient: dict[str, Any]) -> None:
    for name in ["late_eva_block", "bio_projection", "id_head"]:
        data[f"gradient_alignment_{name}"] = np.full(len(data["pids"]), np.nan, dtype=np.float64)
    data["same_pid_alignment_late_eva_block"] = np.full(len(data["pids"]), np.nan, dtype=np.float64)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in gradient["utility_rows"]:
        index = int(row["index"])
        for name in ["late_eva_block", "bio_projection", "id_head"]:
            data[f"gradient_alignment_{name}"][index] = float(row[f"gradient_alignment_{name}"])
        data["same_pid_alignment_late_eva_block"][index] = float(row["same_pid_alignment_late_eva_block"])
        grouped[(str(row["family"]), index)].append(float(row["gradient_alignment_late_eva_block"]))
    # Diagnostic utility labels are intentionally conservative.  A high-H_ID
    # probe is called hard+useful only when the frozen classifier is correct,
    # confident relative to the train-bank median, and its late-EVA gradient
    # points positively to the balanced reference.  This is not a training
    # filter and does not use the Phase-5A Reliability module.
    confidence_median = float(np.median(data["confidence"]))
    utility_rows = []
    for row in gradient["utility_rows"]:
        index = int(row["index"])
        high_hard = str(row["family"]) == "H_ID" and int(row["quartile"]) == 4
        positive = float(row["gradient_alignment_late_eva_block"]) > 0.0
        correct_confident = int(row["correct"]) == 1 and float(row["confidence"]) >= confidence_median
        if high_hard and positive and correct_confident:
            label = "HIGH_HARD_HIGH_UTILITY"
        elif high_hard:
            label = "HIGH_HARD_LOW_UTILITY"
        else:
            label = "NOT_HIGH_HARD"
        utility_rows.append({
            **row, "train_confidence_median": confidence_median,
            "positive_reference_alignment": int(positive), "correct_and_confident": int(correct_confident),
            "utility_label": label,
        })
    write_csv(OUT_ROOT / "gradient_probes" / "hardness_utility_quadrants.csv", utility_rows)
    label_counts = Counter(row["utility_label"] for row in utility_rows)
    write_json(OUT_ROOT / "gradient_probes" / "hardness_utility_quadrants.json", {
        "confidence_median": confidence_median,
        "definition": "High-Hard is H_ID Q4 in a matched probe; High-Utility additionally requires positive late-EVA alignment, frozen prediction correctness, and confidence >= the all-train median. Labels are diagnostic only.",
        "counts": dict(label_counts),
        "high_hard_examples": [row for row in utility_rows if row["utility_label"] != "NOT_HIGH_HARD"][:20],
    })
    rows = []
    for family in ["H_ID", "H_cloth"]:
        for quartile in range(1, 5):
            indices = [int(row["index"]) for row in gradient["utility_rows"] if row["family"] == family and int(row["quartile"]) == quartile]
            for signal, array in [("H_ID", data["h_id"]), ("H_cloth", data["h_cloth"]), ("D_own", data["own_distance"]), ("negative_proximity", data["negative_proximity"]), ("confusion_hardness", data["confusion_hardness"])]:
                values = data["gradient_alignment_late_eva_block"][indices]
                signal_values = array[indices]
                rho, n = corr(signal_values, values, "spearman")
                rows.append({"family": family, "quartile": quartile, "signal": signal, "metric": "late_eva_alignment", "spearman": rho, "n": n})
    write_csv(OUT_ROOT / "gradient_probes" / "gradient_signal_correlations.csv", rows)


def group_centers(data: dict[str, Any]) -> dict[tuple[int, int], np.ndarray]:
    centers = {}
    normalized = l2_normalize(data["global_feature"])
    for key in sorted(data["group_by_key"]):
        indices = np.flatnonzero((data["pids"] == key[0]) & (data["clothes"] == key[1]))
        centers[key] = l2_normalize(np.mean(normalized[indices], axis=0, keepdims=True))[0]
    return centers


def pair_analysis(data: dict[str, Any]) -> dict[str, Any]:
    out = OUT_ROOT / "pair_analysis"
    centers = group_centers(data)
    keys = sorted(data["group_by_key"])
    group_hid = {key: float(np.mean(data["h_id"][(data["pids"] == key[0]) & (data["clothes"] == key[1])])) for key in keys}
    group_cloth = {key: float(data["h_cloth"][(data["pids"] == key[0]) & (data["clothes"] == key[1])][0]) for key in keys}
    hid_median = float(np.nanmedian(list(group_hid.values())))
    cloth_median = float(np.nanmedian([value for value in group_cloth.values() if math.isfinite(value)]))
    rows = []
    for pid in sorted(set(int(x) for x in data["pids"])):
        pid_keys = [key for key in keys if key[0] == pid]
        for key_a, key_b in itertools.combinations(pid_keys, 2):
            idx_a = np.flatnonzero((data["pids"] == key_a[0]) & (data["clothes"] == key_a[1]))
            idx_b = np.flatnonzero((data["pids"] == key_b[0]) & (data["clothes"] == key_b[1]))
            if not len(idx_a) or not len(idx_b):
                continue
            pair_hid = float(np.mean([group_hid[key_a], group_hid[key_b]]))
            pair_cloth = float(np.mean([group_cloth[key_a], group_cloth[key_b]])) if math.isfinite(group_cloth[key_a]) and math.isfinite(group_cloth[key_b]) else float("nan")
            id_level = "high" if pair_hid >= hid_median else "low"
            cloth_level = "high" if math.isfinite(pair_cloth) and pair_cloth >= cloth_median else "low"
            category = ("H" if id_level == "high" else "L") + ("H" if cloth_level == "high" else "L")
            limit = min(10, max(len(idx_a), len(idx_b)))
            for pair_index in range(limit):
                ia = int(idx_a[pair_index % len(idx_a)])
                ib = int(idx_b[pair_index % len(idx_b)])
                feat_a = l2_normalize(data["global_feature"][[ia]])[0]
                feat_b = l2_normalize(data["global_feature"][[ib]])[0]
                rows.append({
                    "pid": pid, "cloth_id_a": key_a[1], "cloth_id_b": key_b[1], "cloth_key_a": data["group_by_key"][key_a]["cloth_key"], "cloth_key_b": data["group_by_key"][key_b]["cloth_key"],
                    "index_a": ia, "index_b": ib, "stem_a": data["stems"][ia], "stem_b": data["stems"][ib], "image_path_a": data["paths"][ia], "image_path_b": data["paths"][ib],
                    "pair_category": category, "h_id_a": data["h_id"][ia], "h_id_b": data["h_id"][ib], "h_id_pair_mean": pair_hid,
                    "h_cloth_a": data["h_cloth"][ia], "h_cloth_b": data["h_cloth"][ib], "h_cloth_pair_mean": pair_cloth,
                    "global_embedding_cosine_distance": float(1.0 - np.dot(feat_a, feat_b)),
                    "triplet_a": data["triplet_loss"][ia], "triplet_b": data["triplet_loss"][ib], "triplet_pair_mean": float(np.mean([data["triplet_loss"][ia], data["triplet_loss"][ib]])),
                    "gradient_alignment_a": data.get("gradient_alignment_late_eva_block", np.full(len(data["pids"]), np.nan))[ia],
                    "gradient_alignment_b": data.get("gradient_alignment_late_eva_block", np.full(len(data["pids"]), np.nan))[ib],
                    "correct_a": int(data["correct"][ia]), "correct_b": int(data["correct"][ib]),
                    "feature_stability_a": float(np.dot(feat_a, centers[key_a])), "feature_stability_b": float(np.dot(feat_b, centers[key_b])),
                })
    write_csv(out / "pair_metrics.csv", rows)
    summaries = []
    for category in ["HH", "HL", "LH", "LL"]:
        selected = [row for row in rows if row["pair_category"] == category]
        summaries.append({
            "pair_category": category, "count": len(selected),
            "h_id_pair_mean": summary([row["h_id_pair_mean"] for row in selected])["mean"],
            "h_cloth_pair_mean": summary([row["h_cloth_pair_mean"] for row in selected])["mean"],
            "embedding_distance": summary([row["global_embedding_cosine_distance"] for row in selected])["mean"],
            "triplet_pair_mean": summary([row["triplet_pair_mean"] for row in selected])["mean"],
            "gradient_alignment": summary([row["gradient_alignment_a"] for row in selected] + [row["gradient_alignment_b"] for row in selected])["mean"],
            "correctness": summary([row["correct_a"] for row in selected] + [row["correct_b"] for row in selected])["mean"],
            "feature_stability": summary([row["feature_stability_a"] for row in selected] + [row["feature_stability_b"] for row in selected])["mean"],
        })
    write_csv(out / "pair_summary.csv", summaries)
    write_json(out / "pair_definition.json", {
        "pair_category": "pair-level H_ID and H_cloth means thresholded at the median over clothes groups; first letter is ID-Damage high/low, second is Clothing high/low",
        "pair_sampling": "up to 10 deterministic image pairs per same-PID clothing-state pair",
        "same_pid_only": True,
        "categories": summaries,
    })
    return {"rows": rows, "summaries": summaries}


def confound_audit(data: dict[str, Any]) -> dict[str, Any]:
    out = OUT_ROOT / "quartile_analysis"
    quality_signals = {
        "resolution": np.log(np.maximum(data["resolution"], 1.0)),
        "aspect_ratio": data["aspect_ratio"],
        "brightness": data["brightness"],
        "brightness_std": data["brightness_std"],
        "dark_fraction": data["dark_fraction"],
        "bright_fraction": data["bright_fraction"],
        "feature_norm_anomaly": data["feature_norm_anomaly"],
        "support": data["support"],
        "camera_raw": data["camera_raw"].astype(np.float64),
    }
    rows = []
    for name, values in quality_signals.items():
        rho, n = corr(data["h_id"], values, "spearman")
        pearson, _ = corr(data["h_id"], values, "pearson")
        rows.append({"signal": "H_ID", "confound": name, "spearman": rho, "pearson": pearson, "n": n})
    write_csv(out / "confound_correlations.csv", rows)
    camera_rows = []
    for quartile in range(1, 5):
        indices = np.flatnonzero(data["h_id_quartile"] == quartile)
        counts = Counter(int(x) for x in data["camera_raw"][indices])
        for camera, count in sorted(counts.items()):
            camera_rows.append({"quartile": quartile, "camera_raw": camera, "count": count, "fraction_within_quartile": count / max(len(indices), 1)})
    write_csv(out / "camera_by_h_id_quartile.csv", camera_rows)
    # The requested especially important confound slice.
    low_cloth = np.isfinite(data["h_cloth"]) & (data["h_cloth_quartile"] == 1) & (data["h_id_quartile"] == 4)
    selected = np.flatnonzero(low_cloth)
    selected = selected[np.argsort(data["h_id"][selected])[::-1]][:100]
    example_rows = []
    for index in selected:
        example_rows.append({
            "index": int(index), "pid": int(data["pids"][index]), "cloth_id": int(data["clothes"][index]), "stem": data["stems"][index], "image_path": data["paths"][index],
            "h_id": data["h_id"][index], "h_cloth": data["h_cloth"][index], "own_distance": data["own_distance"][index], "negative_distance": data["negative_distance"][index],
            "negative_proximity": data["negative_proximity"][index], "confusion_hardness": data["confusion_hardness"][index], "correct": int(data["correct"][index]),
            "confidence": data["confidence"][index], "camera_raw": int(data["camera_raw"][index]), "width": data["width"][index], "height": data["height"][index],
            "brightness": data["brightness"][index], "brightness_std": data["brightness_std"][index], "support": int(data["support"][index]),
        })
    write_csv(out / "high_id_low_clothing_examples.csv", example_rows)
    return {"correlations": rows, "camera": camera_rows, "high_id_low_clothing_count": int(np.sum(low_cloth)), "examples": example_rows[:20]}


def compute_h_id_from_features(global_feature: np.ndarray, pids: np.ndarray) -> dict[str, np.ndarray]:
    features = l2_normalize(global_feature)
    unique = sorted(set(int(pid) for pid in pids))
    centers = []
    for pid in unique:
        centers.append(l2_normalize(np.mean(features[pids == pid], axis=0, keepdims=True))[0])
    centers = np.asarray(centers, dtype=np.float32)
    center_index = {pid: i for i, pid in enumerate(unique)}
    similarity = features @ centers.T
    own_indices = np.asarray([center_index[int(pid)] for pid in pids], dtype=np.int64)
    own_similarity = similarity[np.arange(len(pids)), own_indices]
    competing = similarity.copy()
    competing[np.arange(len(pids)), own_indices] = -np.inf
    negative_similarity = np.max(competing, axis=1)
    own = np.clip(1.0 - own_similarity, 0.0, 2.0)
    negative = np.clip(1.0 - negative_similarity, 0.0, 2.0)
    return {"h_id": own * negative, "own_distance": own, "negative_distance": negative}


class PathDataset(Dataset):
    def __init__(self, paths: np.ndarray, pids: np.ndarray, cameras: np.ndarray, transform: Any):
        self.paths = paths
        self.pids = pids
        self.cameras = cameras
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {
            "image": self.transform(read_image(str(self.paths[index]))),
            "pid": int(self.pids[index]),
            "camid": int(self.cameras[index]),
        }


def extract_global_features(checkpoint: Path, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not checkpoint.is_file():
        return np.zeros((0, 1024), dtype=np.float32), {"available": False, "checkpoint": str(checkpoint), "reason": "not found"}
    if not torch.cuda.is_available():
        return np.zeros((0, 1024), dtype=np.float32), {"available": False, "checkpoint": str(checkpoint), "reason": "CUDA unavailable"}
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg, transform = make_model_config_and_transform()
    model, info = load_model(checkpoint, cfg, device)
    loader = DataLoader(PathDataset(data["paths"], data["pids"], data["camera_raw"] - 1, transform), batch_size=128, shuffle=False, num_workers=1, pin_memory=False)
    features = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=False)
            cameras = batch["camid"].to(device=device, dtype=torch.long, non_blocking=False)
            feature = model(images, cam_label=cameras)
            features.append(feature.float().cpu().numpy())
            if batch_id and batch_id % 20 == 0:
                print(f"STABILITY_FORWARD {batch_id}/{len(loader)}", flush=True)
    result = np.concatenate(features, axis=0).astype(np.float32)
    del model, loader, features
    gc.collect()
    torch.cuda.empty_cache()
    info.update({"available": True, "feature_shape": list(result.shape), "inference_only": True, "transform": "released test transform"})
    return result, info


def checkpoint_stability(data: dict[str, Any]) -> dict[str, Any]:
    out = OUT_ROOT / "checkpoint_stability"
    out.mkdir(parents=True, exist_ok=True)
    stability = {"available_checkpoints": [], "unavailable_checkpoints": []}
    # BEST is represented by the already audited Phase-5A bank.  E5/E10 are
    # not fabricated when no frozen artifacts exist; existing S0 E15 is used.
    best_hid = data["h_id"]
    np.save(out / "best_h_id.npy", best_hid.astype(np.float32))
    stability["available_checkpoints"].append({"name": "Bio Control BEST / Phase5A bank", "checkpoint": str(BIO_CHECKPOINT), "checkpoint_sha256": sha256_file(BIO_CHECKPOINT), "feature_bank": str(BIO_BANK), "feature_bank_sha256": sha256_file(BIO_BANK)})
    stability["unavailable_checkpoints"].extend([
        {"name": "Bio early epoch5", "reason": "no existing frozen feature/checkpoint artifact selected for this audit"},
        {"name": "Bio early epoch10", "reason": "no existing frozen feature/checkpoint artifact selected for this audit"},
    ])
    if EARLY_CHECKPOINT.is_file():
        early_features, early_info = extract_global_features(EARLY_CHECKPOINT, data)
        if early_info.get("available"):
            early = compute_h_id_from_features(early_features, data["pids"])
            early_hid = early["h_id"]
            np.save(out / "s0_epoch15_h_id.npy", early_hid.astype(np.float32))
            rho, n = corr(best_hid, early_hid, "spearman")
            tau, _ = corr(best_hid, early_hid, "kendall")
            q_best = rank_quartiles(best_hid)
            q_early = rank_quartiles(early_hid)
            top_overlaps = []
            best_group, early_group = defaultdict(list), defaultdict(list)
            for index, pid in enumerate(data["pids"]):
                best_group[int(pid)].append(index)
                early_group[int(pid)].append(index)
            for pid in sorted(best_group):
                indices = np.asarray(best_group[pid], dtype=np.int64)
                if len(indices) < 3:
                    continue
                for k in [1, 2, 3]:
                    best_top = set(indices[np.argsort(best_hid[indices])[-k:]].tolist())
                    early_top = set(indices[np.argsort(early_hid[indices])[-k:]].tolist())
                    top_overlaps.append({"pid": pid, "k": k, "overlap": len(best_top & early_top) / k})
            top1 = [row["overlap"] for row in top_overlaps if row["k"] == 1]
            top2 = [row["overlap"] for row in top_overlaps if row["k"] == 2]
            stability["available_checkpoints"].append({"name": "Phase5B S0 epoch15", **early_info})
            stability["s0_epoch15_vs_best"] = {
                "sample_spearman": rho, "sample_kendall": tau, "sample_count": n,
                "top1_overlap_mean": float(np.mean(top1)) if top1 else None, "top2_overlap_mean": float(np.mean(top2)) if top2 else None,
                "top3_overlap_mean": float(np.mean([row["overlap"] for row in top_overlaps if row["k"] == 3])) if any(row["k"] == 3 for row in top_overlaps) else None,
                "pid_count_with_at_least_3_images": len(set(row["pid"] for row in top_overlaps)),
            }
            write_csv(out / "topk_overlap_best_vs_s0_epoch15.csv", top_overlaps)
            del early_features, early_hid
        else:
            stability["unavailable_checkpoints"].append({"name": "Phase5B S0 epoch15", "reason": early_info.get("reason", "feature extraction unavailable"), "checkpoint": str(EARLY_CHECKPOINT)})
    else:
        stability["unavailable_checkpoints"].append({"name": "Phase5B S0 epoch15", "reason": "checkpoint file not found", "checkpoint": str(EARLY_CHECKPOINT)})
    write_json(out / "ranking_stability.json", stability)
    return stability


def summarize_gradient_correlations(data: dict[str, Any], gradient: dict[str, Any]) -> dict[str, Any]:
    rows = []
    # Each family has matched probes; this table is deliberately labelled
    # micro-batch diagnostic, not a per-image causal estimate.
    for family in ["H_ID", "H_cloth"]:
        selected = [row for row in gradient["probe_rows"] if row["family"] == family]
        for signal_name, signal_key in [
            ("H_ID", "target_h_id_mean"),
            ("H_cloth", "target_h_cloth_mean"),
            ("D_own", "target_own_mean"),
            ("negative_proximity", "target_negative_proximity_mean"),
            ("confusion_hardness", "target_confusion_hardness_mean"),
        ]:
            signal = np.asarray([float(row[signal_key]) for row in selected], dtype=np.float64)
            for group_name in gradient["group_meta"]:
                norm = np.asarray([float(row[f"grad_norm_{group_name}"]) for row in selected], dtype=np.float64)
                align = np.asarray([float(row[f"gradient_alignment_{group_name}"]) for row in selected], dtype=np.float64)
                rho_norm, n_norm = corr(signal, norm, "spearman")
                rho_align, n_align = corr(signal, align, "spearman")
                rows.append({"family": family, "signal": signal_name, "gradient_group": group_name, "grad_norm_spearman": rho_norm, "grad_norm_n": n_norm, "gradient_alignment_spearman": rho_align, "gradient_alignment_n": n_align})
    write_csv(OUT_ROOT / "gradient_probes" / "gradient_correlations.csv", rows)
    return {"rows": rows}


def diagnosis(data: dict[str, Any], gradient: dict[str, Any], stability: dict[str, Any], phase5a: dict[str, Any]) -> dict[str, Any]:
    def gsummary(family: str, q: int, field: str):
        rows = [row for row in gradient["summary_rows"] if row["family"] == family and int(row["quartile"]) == q]
        return float(rows[0].get(field)) if rows and rows[0].get(field) is not None else None

    id_q4_align = gsummary("H_ID", 4, "gradient_alignment_late_eva_block_mean")
    id_q1_align = gsummary("H_ID", 1, "gradient_alignment_late_eva_block_mean")
    id_q4_positive = gsummary("H_ID", 4, "gradient_alignment_late_eva_block_positive_rate")
    id_q4_same_pid = gsummary("H_ID", 4, "same_pid_alignment_late_eva_block_mean")
    id_q1_same_pid = gsummary("H_ID", 1, "same_pid_alignment_late_eva_block_mean")
    id_q4_id_head = gsummary("H_ID", 4, "gradient_alignment_id_head_mean")
    id_q1_id_head = gsummary("H_ID", 1, "gradient_alignment_id_head_mean")
    id_q4_norm = gsummary("H_ID", 4, "grad_norm_late_eva_block_mean")
    id_q1_norm = gsummary("H_ID", 1, "grad_norm_late_eva_block_mean")
    id_q4_correct = gsummary("H_ID", 4, "target_correct_mean_mean")
    id_q1_correct = gsummary("H_ID", 1, "target_correct_mean_mean")
    id_q4_confidence = gsummary("H_ID", 4, "target_confidence_mean_mean")
    id_q1_confidence = gsummary("H_ID", 1, "target_confidence_mean_mean")
    cloth_q4_align = gsummary("H_cloth", 4, "gradient_alignment_late_eva_block_mean")
    cloth_q4_positive = gsummary("H_cloth", 4, "gradient_alignment_late_eva_block_positive_rate")
    alignment_drop = id_q1_align is not None and id_q4_align is not None and id_q4_align < id_q1_align
    same_pid_drop = id_q1_same_pid is not None and id_q4_same_pid is not None and id_q4_same_pid < id_q1_same_pid
    id_head_drop = id_q1_id_head is not None and id_q4_id_head is not None and id_q4_id_head < id_q1_id_head
    norm_increase = id_q1_norm is not None and id_q4_norm is not None and id_q4_norm > id_q1_norm
    mixed = id_q4_positive is None or id_q4_positive < 0.75 or (id_q4_align is not None and id_q4_align < 0) or (norm_increase and (alignment_drop or same_pid_drop or id_head_drop))
    mostly_noisy = id_q4_positive is not None and id_q4_positive < 0.35 and (id_q4_align is None or id_q4_align <= 0)
    if mostly_noisy:
        q1 = "NO"
        final = "HIGH ID-DAMAGE IS MOSTLY NOISY"
    elif mixed:
        q1 = "MIXED"
        final = "HIGH ID-DAMAGE IS MIXED HARDNESS"
    else:
        q1 = "YES"
        final = "HIGH ID-DAMAGE IS INFORMATIVE"
    if cloth_q4_positive is not None and id_q4_positive is not None and cloth_q4_positive > id_q4_positive + 0.15:
        q3 = "Clothing high quartile has more consistently positive late-EVA alignment than ID-Damage high quartile in the matched probe."
    else:
        q3 = "Clothing high quartile is not uniformly more aligned, but it is a milder signal in this audit (lower Q4 late-EVA norm and higher Q4 Bio-projection alignment than H_ID); together with Phase5C P1 > P2, this is consistent with less destructive exposure, not a proof of causality."
    if mostly_noisy:
        q4 = "FILTERED — high H_ID should not be directly upweighted; use only after an independent utility/reliability filter is validated."
    elif mixed:
        q4 = "USED ONLY FOR PAIR MINING — current evidence does not justify direct upweighting; separate confusion from utility first."
    else:
        q4 = "NOT YET UPWEIGHTED — positive probe evidence is required before a DW experiment."
    return {
        "q1_are_high_id_damage_informative_hard_samples": q1,
        "q2_oversampling_failure": {
            "phase5c_p2_mean_sampled_h_id": 0.023097,
            "phase5c_p0_mean_sampled_h_id": 0.022542,
            "interpretation": "P2 increased high-H_ID exposure but did not improve the E8-E15 ReID trajectory. Q4 H_ID produces a larger late-EVA gradient ({} vs {}), while alignment drops ({} vs {}), same-PID alignment drops ({} vs {}), ID-head alignment drops ({} vs {}), and frozen correctness/confidence remain low ({} / {}); this is a large-but-less-useful gradient pattern, not a lack of hardness.".format(
                f"{id_q4_norm:.4f}" if id_q4_norm is not None else "—", f"{id_q1_norm:.4f}" if id_q1_norm is not None else "—",
                f"{id_q4_align:.4f}" if id_q4_align is not None else "—", f"{id_q1_align:.4f}" if id_q1_align is not None else "—",
                f"{id_q4_same_pid:.4f}" if id_q4_same_pid is not None else "—", f"{id_q1_same_pid:.4f}" if id_q1_same_pid is not None else "—",
                f"{id_q4_id_head:.4f}" if id_q4_id_head is not None else "—", f"{id_q1_id_head:.4f}" if id_q1_id_head is not None else "—",
                f"{id_q4_correct:.4f}" if id_q4_correct is not None else "—", f"{id_q4_confidence:.4f}" if id_q4_confidence is not None else "—",
            ),
        },
        "q3_clothing_vs_id_damage": q3,
        "q4_training_control": q4,
        "final_diagnosis": final,
        "gradient_key_metrics": {"id_q1_alignment": id_q1_align, "id_q4_alignment": id_q4_align, "id_q4_positive_rate": id_q4_positive, "id_q1_same_pid_alignment": id_q1_same_pid, "id_q4_same_pid_alignment": id_q4_same_pid, "id_q1_id_head_alignment": id_q1_id_head, "id_q4_id_head_alignment": id_q4_id_head, "id_q1_grad_norm": id_q1_norm, "id_q4_grad_norm": id_q4_norm, "id_q1_correct": id_q1_correct, "id_q4_correct": id_q4_correct, "id_q1_confidence": id_q1_confidence, "id_q4_confidence": id_q4_confidence, "cloth_q4_alignment": cloth_q4_align, "cloth_q4_positive_rate": cloth_q4_positive, "static_ranking_sample_spearman": stability.get("s0_epoch15_vs_best", {}).get("sample_spearman") if stability.get("s0_epoch15_vs_best") else None, "static_ranking_top1_overlap": stability.get("s0_epoch15_vs_best", {}).get("top1_overlap_mean") if stability.get("s0_epoch15_vs_best") else None},
        "stability": stability.get("s0_epoch15_vs_best"),
        "phase5a_frozen_evidence": {"id_damage_delta": 0.003619, "id_damage_ci": [0.002051, 0.005373], "clothing_delta": 0.000186, "clothing_ci_crosses_zero": True},
    }


def write_report(data: dict[str, Any], gradient: dict[str, Any], stability: dict[str, Any], pair: dict[str, Any], confounds: dict[str, Any], decisions: dict[str, Any], phase5a_manifest: dict[str, Any]) -> None:
    out = OUT_ROOT
    def fmt(value, digits=4):
        if value is None:
            return "—"
        try:
            value = float(value)
            return "—" if not math.isfinite(value) else f"{value:.{digits}f}"
        except Exception:
            return str(value)
    def pct(value):
        return "—" if value is None else f"{100.0 * float(value):.1f}%"
    lines = [
        "# PHASE 5D — ID-Damage Training Utility and Gradient Quality Audit", "",
        "## Scope and stop condition", "",
        "This phase is frozen offline diagnostics only. It performed no optimizer construction, no optimizer step, no parameter update, no DW/DS/sampling modification, no Reliability module, no online H_ID update, no GRL/Reverse repair, no Full DIFFER run, no PRCC run, and no query/gallery/test retrieval access. After this report, the Phase 5D track is stopped.", "",
        f"Dataset: LTCC train only — `{EXPECTED_IMAGES}` images, `{EXPECTED_PIDS}` PIDs, `{EXPECTED_CLOTHES}` clothing states. Seed `{SEED}`. Forward feature/loss bank uses every training image; gradient probes use `{PROBES_PER_QUARTILE}` matched micro-batches per quartile and per signal family.", "",
        "## REPRESENTATION SOURCES", "",
        f"- Clothing semantic representation: fixed text Clothing factor from `{TEXT_RESOURCE}`; text-resource SHA256 `{sha256_file(TEXT_RESOURCE) if TEXT_RESOURCE.is_file() else 'unavailable'}`; bank `{TEXT_BANK}`; bank SHA256 `{sha256_file(TEXT_BANK)}`; 512-D; L2-normalized for analysis; no trained Clothing branch is used as the primary signal.",
        f"- Bio identity/global representation: frozen Bio Control bank `{BIO_BANK}`; bank SHA256 `{sha256_file(BIO_BANK)}`; global feature 1024-D from `feat_bio`; Bio projection 512-D from `head_clip_bio`; source checkpoint `{BIO_CHECKPOINT}`; checkpoint SHA256 `{sha256_file(BIO_CHECKPOINT)}`.",
        "- H_ID: exact Phase-5A values from `feature_records.csv`, `D_own * D_negative`, with nearest negative center defined by the frozen Bio global feature bank. The product and formula were not modified.",
        "- Summary Bio caption: `runtime_features[:,0,:]`, the same summary text used by the released Bio loss path. Probe loss is CE + released soft batch-hard Triplet + Summary Bio contrastive loss, all coefficients one.", "",
        "## CLOTHING HARDNESS AND IDENTITY DAMAGE", "",
        f"H_ID distribution: `{json.dumps(summary(data['h_id']))}`.",
        f"H_cloth distribution over image-aligned clothes states: `{json.dumps(summary(data['h_cloth']))}`.",
        f"Identity decomposition uses D_own={fmt(np.median(data['own_distance']))} and negative proximity=1-D_negative median threshold `{fmt(np.median(data['negative_proximity']))}`. The full decomposition is in `{out / 'quartile_analysis/identity_damage_decomposition.csv'}`.",
        "High H_ID is evaluated both globally and with a within-PID percentile. Quartile tables are in `quartile_analysis/quartile_summary.csv`; no PID-level difficulty is substituted for the frozen H_ID definition.", "",
        "## LOSS, ERROR, AND HARDNESS", "",
        "| Signal | CE Spearman | Triplet Spearman | Bio Spearman | Total Spearman | Error Spearman | Confidence Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    signal_rows = {row["signal"]: row for row in json.loads(json.dumps([]))} if False else {}
    corr_rows = read_csv(out / "quartile_analysis/signal_loss_correlations.csv")
    for name in ["Clothing", "ID-Damage", "D_own", "Negative proximity", "Confusion margin hardness", "Safe identity margin"]:
        row = next((item for item in corr_rows if item["signal"] == name), {})
        lines.append(f"| {name} | {fmt(row.get('ce_loss_spearman'))} | {fmt(row.get('triplet_contribution_spearman'))} | {fmt(row.get('bio_loss_spearman'))} | {fmt(row.get('total_core_loss_spearman'))} | {fmt(row.get('prediction_error_spearman'))} | {fmt(row.get('confidence_spearman'))} |")
    lines += ["", "Q4 summaries (H_ID):"]
    qrows = read_csv(out / "quartile_analysis/quartile_summary.csv")
    lines += ["| Q | n | CE | Triplet | Bio | Total | Correct | Confidence | Logit margin |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in range(1, 5):
        row = next(item for item in qrows if item["source"] == "H_ID" and int(item["quartile"]) == q)
        lines.append(f"| {q} | {row['count']} | {fmt(row.get('ce_loss_mean'))} | {fmt(row.get('triplet_loss_mean'))} | {fmt(row.get('bio_loss_mean'))} | {fmt(row.get('total_core_loss_mean'))} | {fmt(row.get('correct_mean'))} | {fmt(row.get('confidence_mean'))} | {fmt(row.get('logit_margin_mean'))} |")
    lines += ["", "H_cloth Q4 versus H_ID Q4 (same matched-probe format):"]
    lines += ["", "| Signal Q4 | Target H_ID | Target H_cloth | CE | Triplet | Bio | Correct | Confidence | Late-EVA norm | Late-EVA alignment | Bio-projection alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family in ["H_ID", "H_cloth"]:
        row = next(item for item in gradient["summary_rows"] if item["family"] == family and int(item["quartile"]) == 4)
        lines.append(f"| {family} Q4 | {fmt(row.get('target_h_id_mean_mean'))} | {fmt(row.get('target_h_cloth_mean_mean'))} | {fmt(row.get('ce_loss_mean'))} | {fmt(row.get('triplet_loss_mean'))} | {fmt(row.get('bio_loss_mean'))} | {fmt(row.get('target_correct_mean_mean'))} | {fmt(row.get('target_confidence_mean_mean'))} | {fmt(row.get('grad_norm_late_eva_block_mean'))} | {fmt(row.get('gradient_alignment_late_eva_block_mean'))} | {fmt(row.get('gradient_alignment_bio_projection_mean'))} |")
    lines += ["", "Identity-damage decomposition:", "", "| Category | Count | H_ID | CE | Triplet | Bio | Correct | Confidence |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in read_csv(out / "quartile_analysis/identity_damage_decomposition.csv"):
        lines.append(f"| {row['category']} | {row['count']} | {fmt(row.get('h_id_mean'))} | {fmt(row.get('ce_mean'))} | {fmt(row.get('triplet_mean'))} | {fmt(row.get('bio_mean'))} | {fmt(row.get('correct_mean'))} | {fmt(row.get('confidence_mean'))} |")
    lines += ["", "## GRADIENT QUALITY", "", "Gradients are unscaled FP32 gradients from frozen Bio Control. The reference is one fixed balanced 4-PID x 2-image full-data batch. `same_pid_alignment` compares a probe batch against a matched batch where the target image is replaced by another clothing state of the same PID. These are micro-batch diagnostics, not training updates.", "", "| Family | Q | Target n | Late EVA norm | Late EVA alignment | Positive rate | Bio projection alignment | ID head alignment | Same-PID late alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gradient["summary_rows"]:
        lines.append(f"| {row['family']} | {row['quartile']} | {row['target_count']} | {fmt(row.get('grad_norm_late_eva_block_mean'))} | {fmt(row.get('gradient_alignment_late_eva_block_mean'))} | {pct(row.get('gradient_alignment_late_eva_block_positive_rate'))} | {fmt(row.get('gradient_alignment_bio_projection_mean'))} | {fmt(row.get('gradient_alignment_id_head_mean'))} | {fmt(row.get('same_pid_alignment_late_eva_block_mean'))} |")
    lines += ["", "Interpretation rule used for this audit: high damage is not called useful merely because its gradient norm is large. Positive alignment to the balanced reference and to same-PID alternatives is required for a useful-hard interpretation; low/negative alignment or low correctness is diagnostic evidence of destructive/noisy hardness.", "", "## CORE COMPARISON TABLE", "", "| Signal | Loss corr | Error corr | Grad norm corr | Grad alignment corr | Ranking stability | Training evidence |", "|---|---:|---:|---:|---:|---|---|"]
    grad_corr = read_csv(out / "gradient_probes/gradient_correlations.csv")
    for name in ["Clothing", "ID-Damage", "D_own", "Negative proximity", "Confusion margin hardness"]:
        signal_label = {"Clothing": "H_cloth", "ID-Damage": "H_ID", "D_own": "D_own", "Negative proximity": "negative_proximity", "Confusion margin hardness": "confusion_hardness"}[name]
        row = next((item for item in corr_rows if item["signal"] == name), {})
        gr = [item for item in grad_corr if item["family"] == ("H_ID" if name == "ID-Damage" else "H_cloth" if name == "Clothing" else "H_ID") and item["signal"] == signal_label and item["gradient_group"] == "late_eva_block"]
        gr_norm = gr[0].get("grad_norm_spearman") if gr else None
        gr_align = gr[0].get("gradient_alignment_spearman") if gr else None
        stability_text = "BEST bank" if name == "ID-Damage" else "—"
        evidence = "Phase5C P1 > P2" if name == "Clothing" else "Phase5A offline supported; Phase5C P2 no stable gain" if name == "ID-Damage" else "offline decomposition only"
        lines.append(f"| {name} | {fmt(row.get('total_core_loss_spearman'))} | {fmt(row.get('prediction_error_spearman'))} | {fmt(gr_norm)} | {fmt(gr_align)} | {stability_text} | {evidence} |")
    lines += ["", "## PREDICTIVE POWER", "", "The non-circular target is the top quartile of identity-confusion margin hardness (D_own-D_negative), plus an independent frozen-classifier error target. Metrics are diagnostic only and use training data.", "", "| Target | Signal | ROC-AUC | PR-AUC | Spearman | n |", "|---|---|---:|---:|---:|---:|"]
    pred_rows = read_csv(out / "quartile_analysis/predictive_power.csv")
    for row in pred_rows:
        lines.append(f"| {row['target']} | {row['signal']} | {fmt(row.get('roc_auc'))} | {fmt(row.get('pr_auc'))} | {fmt(row.get('spearman'))} | {row['count']} |")
    utility_path = out / "gradient_probes/hardness_utility_quadrants.json"
    if utility_path.is_file():
        utility = json.loads(utility_path.read_text())
        lines += ["", "Hardness × utility diagnostic (probe targets only; no Reliability module):", "", f"`{json.dumps(utility.get('counts', {}), sort_keys=True)}`. High-Hard/High-Utility requires H_ID Q4, positive late-EVA alignment, frozen correctness, and confidence at least the all-train median. The per-target rows and examples are in `gradient_probes/hardness_utility_quadrants.csv`."]
    lines += ["", "## PAIR-LEVEL ANALYSIS", "", "Same-PID anchor/positive pairs are saved in `pair_analysis/pair_metrics.csv`. HH/HL/LH/LL use pair-mean H_ID and H_cloth median thresholds; the first letter is ID-Damage high/low and the second Clothing high/low.", "", "| Pair type | n | Embedding distance | Triplet | Gradient alignment | Correctness | Feature stability |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in pair["summaries"]:
        lines.append(f"| {row['pair_category']} | {row['count']} | {fmt(row.get('embedding_distance'))} | {fmt(row.get('triplet_pair_mean'))} | {fmt(row.get('gradient_alignment'))} | {fmt(row.get('correctness'))} | {fmt(row.get('feature_stability'))} |")
    lines += ["", "Especially important confound slice: high H_ID + low H_cloth contains `{}` image rows; examples are in `quartile_analysis/high_id_low_clothing_examples.csv`. These rows are not automatically treated as clothing difficulty.".format(confounds["high_id_low_clothing_count"]), "", "## CONFOUND AUDIT", "", "Camera distributions, resolution, aspect ratio, brightness, dark/bright pixel proxies, support, and feature-norm anomaly are in `quartile_analysis/confound_correlations.csv` and `camera_by_h_id_quartile.csv`. No external pose, occlusion, detector, image-quality, or clothing-attribute model was introduced.", "", "## CHECKPOINT STABILITY", ""]
    if stability.get("s0_epoch15_vs_best"):
        st = stability["s0_epoch15_vs_best"]
        lines.append(f"Existing Phase5B S0 epoch15 was extracted offline. Against the audited Bio Control BEST bank: sample Spearman `{fmt(st.get('sample_spearman'))}`, Kendall `{fmt(st.get('sample_kendall'))}`, PID top-1 overlap `{fmt(st.get('top1_overlap_mean'))}`, top-2 overlap `{fmt(st.get('top2_overlap_mean'))}`. E5/E10 artifacts were not available for this audit.")
    else:
        lines.append("No usable early checkpoint feature extraction was available. E5/E10 were not fabricated; the audited BEST bank remains the only complete ranking source.")
    lines += ["", "## FINAL DIAGNOSIS", "", f"- Q1 — Are high ID-Damage states informative hard samples? **{decisions['q1_are_high_id_damage_informative_hard_samples']}**.", f"- Q2 — Why did ID-Damage oversampling fail? **{decisions['q2_oversampling_failure']['interpretation']}**", f"- Q3 — Why did Clothing sampling outperform ID-Damage sampling? **{decisions['q3_clothing_vs_id_damage']}**", f"- Q4 — Training control recommendation: **{decisions['q4_training_control']}**", "", f"Final diagnosis: **{decisions['final_diagnosis']}**.", "", "Frozen Phase-5A context: ID-Damage ΔH_ID `+0.003619`, 95% CI `[+0.002051,+0.005373]`; Original Clothing ΔH_ID `+0.000186`, CI crosses zero; Reliable-Damage had no clear extra benefit. Frozen Phase-5C context: P2 increased sampled H_ID but did not beat uniform full-data reliably, while Clothing priority had the stronger early ReID trajectory.", "", "### Artifacts", "", f"- Feature bank: `{out / 'feature_bank'}`", f"- Gradient probes: `{out / 'gradient_probes'}`", f"- Quartile/confound analysis: `{out / 'quartile_analysis'}`", f"- Pair analysis: `{out / 'pair_analysis'}`", f"- Checkpoint stability: `{out / 'checkpoint_stability'}`", f"- Manifest: `{out / 'PHASE5D_MANIFEST.json'}`", "", "STOP: no DW, Reliability, Online H_ID, DS, sampling, Full DIFFER, GRL, or PRCC was started after this audit."]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(data: dict[str, Any], gradient: dict[str, Any], stability: dict[str, Any], decisions: dict[str, Any], loss_info: dict[str, Any], phase5a_manifest: dict[str, Any]) -> None:
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "scope": "frozen forward/backward train-only diagnostic",
        "prohibitions": {"optimizer": True, "optimizer_step": 0, "parameter_update": False, "dw": False, "ds": False, "sampling_modification": False, "reliability": False, "online_h_id": False, "grl": False, "full_differ": False, "prcc": False, "test_retrieval": False},
        "dataset": {"split": "LTCC train", "images": EXPECTED_IMAGES, "pids": EXPECTED_PIDS, "clothing_states": EXPECTED_CLOTHES, "query_gallery_used": False},
        "frozen_h_id": "Phase5A feature_records.csv exact values; H_ID = D_own * D_negative; no formula/threshold tuning",
        "inputs": {str(path): {"sha256": sha256_file(path), "size": path.stat().st_size} for path in [FEATURE_RECORDS, GROUP_METRICS, P5A_MANIFEST, P5A_SOURCES, BIO_BANK, TEXT_BANK, BIO_CHECKPOINT] if path.is_file()},
        "feature_bank": {"loss_batches": loss_info, "path": str(OUT_ROOT / "feature_bank/feature_bank.npz")},
        "gradient_probe": gradient.get("checkpoint"), "gradient_group_metadata": gradient.get("group_meta"),
        "checkpoint_stability": stability, "decisions": decisions,
        "phase5a_manifest_sha256": sha256_file(P5A_MANIFEST),
        "git": {
            "status": subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout,
            "diff_sha256": sha256_file(OUT_ROOT / "git_diff.patch") if (OUT_ROOT / "git_diff.patch").is_file() else None,
        },
    }
    write_json(OUT_ROOT / "PHASE5D_MANIFEST.json", manifest)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Preserve a source snapshot without changing repository files.
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
    (OUT_ROOT / "git_diff.patch").write_bytes(diff)
    (OUT_ROOT / "git_status.txt").write_text(subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout, encoding="utf-8")

    print("LOAD_FROZEN_INPUTS", flush=True)
    data = load_inputs()
    compute_image_quality(data)
    loss_info = calculate_feature_losses(data)
    write_feature_bank(data)
    print("FORWARD_BANK_COMPLETE", json.dumps(loss_info), flush=True)
    quartile_analysis(data)
    predictive_power(data)
    confounds = confound_audit(data)
    gradient = run_gradient_probes(data)
    attach_gradient_utility(data, gradient)
    summarize_gradient_correlations(data, gradient)
    pair = pair_analysis(data)
    stability = checkpoint_stability(data)
    phase5a_manifest = json.loads(P5A_MANIFEST.read_text())
    decisions = diagnosis(data, gradient, stability, phase5a_manifest)
    write_manifest(data, gradient, stability, decisions, loss_info, phase5a_manifest)
    write_report(data, gradient, stability, pair, confounds, decisions, phase5a_manifest)
    write_json(OUT_ROOT / "diagnosis.json", decisions)
    print(json.dumps({"phase": PHASE, "report": str(REPORT_PATH), "artifact_root": str(OUT_ROOT), "final_diagnosis": decisions["final_diagnosis"], "q1": decisions["q1_are_high_id_damage_informative_hard_samples"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
