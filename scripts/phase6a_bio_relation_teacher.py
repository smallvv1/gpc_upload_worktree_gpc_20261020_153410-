#!/usr/bin/env python3
"""Phase 6A: frozen Bio relation-teacher feasibility audit.

This phase is deliberately offline.  It reuses the already extracted frozen
Bio/Full feature banks and performs one no-grad forward extraction for the
missing formal ID-only checkpoint.  All relation, retrieval, KL, overlap, and
hardness analyses are then NumPy-only.  No optimizer, backward pass, parameter
update, sampling change, or test/query/gallery data is used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASE = "phase6a_relation_feasibility"
SEED = 1234
IMAGE_COUNT = 9576
PID_COUNT = 77
STATE_COUNT = 256
RELATION_TEMPERATURE = 0.07
RELATION_MEMORY_SIZE = 256
TOP_K = 10
CHUNK_SIZE = 256

DATA_ROOT = Path("/home/data")
TRAIN_ROOT = DATA_ROOT / "LTCC_ReID/train"

P4R_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features"
P5D_BANK = ROOT / "outputs/new_method_track/phase5d_hardness_utility/feature_bank/feature_bank.npz"
P4R_MANIFEST = P4R_ROOT / "checkpoint_feature_manifest.json"
P5E_QUADRANTS = ROOT / "outputs/new_method_track/phase5e_id_damage_pair_mining/gradient_probes/pair_utility_quadrants.csv"

ID_CHECKPOINT = ROOT / "outputs/reproduction_phase2/LTCC/baseline/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/eva02_l_bio_best.pth"
ID_BANK = ROOT / "outputs/new_method_track/phase6a_bio_relation_teacher/feature_banks/id_only_feature_bank.npz"

OUT_ROOT = ROOT / "outputs/new_method_track/phase6a_relation_feasibility"
REPORT_PATH = ROOT / "reports/PHASE_6A_RELIABLE_RELATION_FEASIBILITY.md"

MODEL_BANKS = {
    "Bio": P4R_ROOT / "bio_control_feature_bank.npz",
    "ID-only": ID_BANK,
    "Local Full": P4R_ROOT / "original_full_feature_bank.npz",
    "Official Full": P4R_ROOT / "official_full_feature_bank.npz",
}


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
        return float(value)
    except Exception:
        return float("nan")


def normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def finite_values(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def scalar_summary(values: Any) -> dict[str, Any]:
    values = finite_values(values)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def safe_spearman(x: Any, y: Any) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) <= 1e-12 or np.ptp(y[valid]) <= 1e-12:
        return None
    value = float(spearmanr(x[valid], y[valid]).statistic)
    return value if math.isfinite(value) else None


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-30)


def feature_bank_info(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bank:
        return {
            "path": path,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "keys": list(bank.files),
            "shapes": {key: list(bank[key].shape) for key in bank.files},
        }


def extract_id_only() -> dict[str, Any]:
    """Extract ID-only global features with no_grad; never constructs an optimizer."""
    if ID_BANK.is_file():
        return {"status": "reused", **feature_bank_info(ID_BANK)}
    if not ID_CHECKPOINT.is_file():
        raise FileNotFoundError(ID_CHECKPOINT)

    import torch
    from config import cfg as GLOBAL_CFG
    from data import build_dataset, build_img_transforms
    from data.dataset_loader import ImageDataset
    from model.build import factory

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6A ID-only frozen forward extraction requires CUDA; no CPU fallback is used")
    torch.cuda.set_device(0)

    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet"]
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = str(DATA_ROOT)
    cfg.DATA.CAPTION_DIR = str(DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID")
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()

    dataset = build_dataset(cfg)
    _, transform = build_img_transforms(cfg)
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=PID_COUNT, camera_num=12)
    state = torch.load(ID_CHECKPOINT, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to("cuda:0")
    model.eval()

    loader = torch.utils.data.DataLoader(
        ImageDataset(dataset.train, transform=transform),
        batch_size=128,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    global_features: list[np.ndarray] = []
    id_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].cuda(non_blocking=False)
            cameras = batch["camid"].cuda(non_blocking=False)
            with torch.autocast(device_type="cuda", enabled=True):
                feature = model(images, cam_label=cameras)
                logits = model.head(feature)
            global_features.append(feature.float().cpu().numpy())
            id_logits.append(logits.float().cpu().numpy())
            if batch_index % 20 == 0:
                print("ID_ONLY_FORWARD_BATCH", batch_index, len(batch["image"]), flush=True)

    features = np.concatenate(global_features, axis=0)
    logits = np.concatenate(id_logits, axis=0)
    if len(features) != IMAGE_COUNT:
        raise RuntimeError(f"ID-only extraction count changed: {len(features)}")
    ID_BANK.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ID_BANK,
        global_feature=features.astype(np.float16),
        bio_projected=np.zeros((len(features), 0), dtype=np.float16),
        id_logits=logits.astype(np.float16),
        reverse_source=np.zeros((len(features), features.shape[1]), dtype=np.float16),
        reverse_hair_projected=np.zeros((len(features), 0), dtype=np.float16),
        reverse_clothing_projected=np.zeros((len(features), 0), dtype=np.float16),
        stems=np.asarray([Path(item["image_path"]).stem for item in dataset.train]),
        pid=np.asarray([item["pid"] for item in dataset.train], dtype=np.int16),
        camera=np.asarray([item["camid"] for item in dataset.train], dtype=np.int16),
        clothes_id=np.asarray([item["clothes_id"] for item in dataset.train], dtype=np.int16),
    )
    del model, dataset
    torch.cuda.empty_cache()
    return {
        "status": "extracted_no_grad",
        **feature_bank_info(ID_BANK),
        "checkpoint": ID_CHECKPOINT,
        "checkpoint_sha256": sha256_file(ID_CHECKPOINT),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def load_bank(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        bank = {key: np.asarray(source[key]) for key in source.files}
    for key in ["global_feature", "stems", "pid", "clothes_id"]:
        if key not in bank:
            raise RuntimeError(f"feature bank missing {key}: {path}")
    if len(bank["stems"]) != IMAGE_COUNT:
        raise RuntimeError(f"unexpected bank size {path}: {len(bank['stems'])}")
    bank["global_normed"] = normalize(bank["global_feature"])
    return bank


def validate_alignment(banks: dict[str, dict[str, Any]], hardness: dict[str, Any]) -> None:
    reference = banks["Bio"]
    for name, bank in banks.items():
        for key in ["stems", "pid", "camera", "clothes_id"]:
            if not np.array_equal(bank[key], reference[key]):
                raise RuntimeError(f"{name} alignment mismatch in {key}")
        if not np.isfinite(bank["global_normed"]).all():
            raise RuntimeError(f"nonfinite normalized feature in {name}")
    for key in ["stems", "pid", "clothes_id"]:
        if not np.array_equal(hardness[key], reference[key]):
            raise RuntimeError(f"Phase5D hardness alignment mismatch in {key}")
    if len(np.unique(reference["pid"])) != PID_COUNT:
        raise RuntimeError("unexpected LTCC PID count")
    if len(set(zip(reference["pid"].tolist(), reference["clothes_id"].tolist()))) != STATE_COUNT:
        raise RuntimeError("unexpected LTCC clothing-state count")


def cross_clothes_relations(features: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> dict[str, Any]:
    n = len(features)
    positive_mean = np.full(n, np.nan, dtype=np.float32)
    positive_median = np.full(n, np.nan, dtype=np.float32)
    positive_count = np.zeros(n, dtype=np.int32)
    pair_values: list[np.ndarray] = []
    pid_rows: list[dict[str, Any]] = []
    for pid in sorted(set(int(value) for value in pids)):
        indices = np.flatnonzero(pids == pid)
        local = features[indices] @ features[indices].T
        local_clothes = clothes[indices]
        mask = local_clothes[:, None] != local_clothes[None, :]
        values = local[mask].astype(np.float32, copy=False)
        pair_values.append(values)
        for position, index in enumerate(indices):
            selected = local[position][mask[position]]
            if len(selected):
                positive_mean[index] = float(np.mean(selected))
                positive_median[index] = float(np.median(selected))
                positive_count[index] = len(selected)
        pid_rows.append({
            "pid": pid,
            "image_count": int(len(indices)),
            "clothing_count": int(len(np.unique(local_clothes))),
            "directed_pair_count": int(len(values)),
            "pair_mean": float(np.mean(values)) if len(values) else None,
            "pair_median": float(np.median(values)) if len(values) else None,
            "anchor_mean": float(np.nanmean(positive_mean[indices])) if np.isfinite(positive_mean[indices]).any() else None,
        })
    all_values = np.concatenate(pair_values) if pair_values else np.empty(0, dtype=np.float32)
    return {
        "positive_mean": positive_mean,
        "positive_median": positive_median,
        "positive_count": positive_count,
        "pair_values": all_values,
        "pid_rows": pid_rows,
        "summary": scalar_summary(all_values),
    }


def same_clothes_relations(features: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> dict[str, Any]:
    """Compute the same-PID/same-clothes positive control without model updates."""
    n = len(features)
    positive_mean = np.full(n, np.nan, dtype=np.float32)
    positive_median = np.full(n, np.nan, dtype=np.float32)
    positive_count = np.zeros(n, dtype=np.int32)
    pair_values: list[np.ndarray] = []
    for pid in sorted(set(int(value) for value in pids)):
        indices = np.flatnonzero(pids == pid)
        local = features[indices] @ features[indices].T
        local_clothes = clothes[indices]
        mask = local_clothes[:, None] == local_clothes[None, :]
        mask[np.arange(len(indices)), np.arange(len(indices))] = False
        values = local[mask].astype(np.float32, copy=False)
        pair_values.append(values)
        for position, index in enumerate(indices):
            selected = local[position][mask[position]]
            if len(selected):
                positive_mean[index] = float(np.mean(selected))
                positive_median[index] = float(np.median(selected))
                positive_count[index] = len(selected)
    all_values = np.concatenate(pair_values) if pair_values else np.empty(0, dtype=np.float32)
    return {
        "positive_mean": positive_mean,
        "positive_median": positive_median,
        "positive_count": positive_count,
        "pair_values": all_values,
        "summary": scalar_summary(all_values),
    }


def choose_cross_clothes_positive(features: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> np.ndarray:
    """Choose one deterministic, teacher-defined cross-clothes positive per anchor."""
    selected = np.full(len(features), -1, dtype=np.int32)
    for pid in sorted(set(int(value) for value in pids)):
        indices = np.flatnonzero(pids == pid)
        local = features[indices] @ features[indices].T
        mask = clothes[indices][:, None] != clothes[indices][None, :]
        for position, index in enumerate(indices):
            candidates = np.flatnonzero(mask[position])
            if len(candidates):
                # Stable tie-break: maximum teacher similarity, then lowest global index.
                best = candidates[np.lexsort((indices[candidates], -local[position, candidates]))[0]]
                selected[index] = int(indices[best])
    return selected


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1] using rank only; no threshold is tuned."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.any():
        ranks = rankdata(values[valid], method="average")
        output[valid] = (ranks - 1.0) / max(float(valid.sum() - 1), 1.0)
    return output.astype(np.float32)


def model_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def train_image_path(stem: Any) -> str:
    return str(TRAIN_ROOT / f"{str(stem)}.png")


def top_negative_neighbors(features: np.ndarray, pids: np.ndarray, k: int = TOP_K, chunk_size: int = CHUNK_SIZE) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    values_out = np.empty((n, k), dtype=np.float32)
    indices_out = np.empty((n, k), dtype=np.int32)
    all_columns = np.arange(n)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        scores = (features[start:end] @ features.T).astype(np.float32, copy=False)
        same = pids[start:end, None] == pids[None, :]
        scores[same] = -np.inf
        selected = np.argpartition(scores, -k, axis=1)[:, -k:]
        selected_values = np.take_along_axis(scores, selected, axis=1)
        order = np.argsort(selected_values, axis=1)[:, ::-1]
        values_out[start:end] = np.take_along_axis(selected_values, order, axis=1)
        indices_out[start:end] = np.take_along_axis(selected, order, axis=1)
        del scores, same, selected, selected_values, order
    return values_out, indices_out


def top_neighbors(features: np.ndarray, k: int = TOP_K, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    n = len(features)
    indices_out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        scores = (features[start:end] @ features.T).astype(np.float32, copy=False)
        scores[np.arange(end - start), np.arange(start, end)] = -np.inf
        selected = np.argpartition(scores, -k, axis=1)[:, -k:]
        selected_values = np.take_along_axis(scores, selected, axis=1)
        order = np.argsort(selected_values, axis=1)[:, ::-1]
        indices_out[start:end] = np.take_along_axis(selected, order, axis=1)
        del scores, selected, selected_values, order
    return indices_out


def train_retrieval(features: np.ndarray, pids: np.ndarray, clothes: np.ndarray, chunk_size: int = CHUNK_SIZE) -> dict[str, np.ndarray]:
    """Leave-one-image-out train retrieval arrays for general and cross-clothes settings."""
    n = len(features)
    general_r1 = np.zeros(n, dtype=np.float32)
    general_ap = np.zeros(n, dtype=np.float32)
    cross_r1 = np.zeros(n, dtype=np.float32)
    cross_ap = np.zeros(n, dtype=np.float32)
    cross_positive_rank = np.full(n, np.nan, dtype=np.float32)
    nearest_negative_rank = np.full(n, np.nan, dtype=np.float32)
    cross_positive_count = np.zeros(n, dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        row_ids = np.arange(start, end)
        scores = (features[start:end] @ features.T).astype(np.float32, copy=False)
        scores[np.arange(end - start), row_ids] = -np.inf
        order = np.argsort(scores, axis=1)[:, ::-1]
        ranked_pid = pids[order]
        ranked_clothes = clothes[order]
        query_pid = pids[start:end, None]
        query_clothes = clothes[start:end, None]
        ranked_not_self = order != row_ids[:, None]
        positive = (ranked_pid == query_pid) & ranked_not_self
        positive_count = positive.sum(axis=1)
        general_r1[start:end] = positive[:, 0].astype(np.float32)
        cumulative = np.cumsum(positive, axis=1)
        precision = cumulative / np.arange(1, n + 1, dtype=np.float32)[None, :]
        general_ap[start:end] = np.divide((precision * positive).sum(axis=1), positive_count, out=np.zeros(end - start, dtype=np.float32), where=positive_count > 0)
        nearest_negative = ranked_pid != query_pid
        nearest_negative_rank[start:end] = np.argmax(nearest_negative, axis=1).astype(np.float32) + 1.0

        cross_scores = scores.copy()
        # Mask same-clothing gallery images in score space before ranking.  The
        # self image is already masked above and is also same-clothing.
        score_same_cloth = clothes[start:end, None] == clothes[None, :]
        cross_scores[score_same_cloth] = -np.inf
        cross_order = np.argsort(cross_scores, axis=1)[:, ::-1]
        cross_pid = pids[cross_order]
        cross_positive = cross_pid == query_pid
        cross_count = cross_positive.sum(axis=1)
        cross_positive_count[start:end] = cross_count
        cross_r1[start:end] = cross_positive[:, 0].astype(np.float32)
        cross_cumulative = np.cumsum(cross_positive, axis=1)
        cross_precision = cross_cumulative / np.arange(1, n + 1, dtype=np.float32)[None, :]
        cross_ap[start:end] = np.divide((cross_precision * cross_positive).sum(axis=1), cross_count, out=np.zeros(end - start, dtype=np.float32), where=cross_count > 0)
        first_cross = np.argmax(cross_positive, axis=1).astype(np.float32) + 1.0
        cross_positive_rank[start:end] = np.where(cross_count > 0, first_cross, np.nan)
        del scores, order, ranked_pid, ranked_clothes, positive, cumulative, precision, cross_scores, cross_order, cross_pid, cross_positive, cross_cumulative, cross_precision
    return {
        "general_r1": general_r1,
        "general_ap": general_ap,
        "cross_r1": cross_r1,
        "cross_ap": cross_ap,
        "cross_positive_rank": cross_positive_rank,
        "nearest_negative_rank": nearest_negative_rank,
        "cross_positive_count": cross_positive_count,
    }


def relation_kl_and_deficits(
    models: dict[str, dict[str, Any]],
    teacher_name: str,
    pids: np.ndarray,
    clothes: np.ndarray,
    memory_indices: np.ndarray,
    teacher_negative_indices: np.ndarray,
) -> dict[str, Any]:
    teacher = models[teacher_name]["features"]
    n = len(teacher)
    kl_by_model: dict[str, np.ndarray] = {}
    d_pos_by_model: dict[str, np.ndarray] = {}
    d_neg_by_model: dict[str, np.ndarray] = {}
    teacher_entropy = np.zeros(n, dtype=np.float32)
    positive_sets: list[np.ndarray] = []
    for index in range(n):
        positive_sets.append(np.flatnonzero((pids == pids[index]) & (clothes != clothes[index])))
    for name, result in models.items():
        d_pos_by_model[name] = result["relations"]["positive_mean"] - models[teacher_name]["relations"]["positive_mean"]
        teacher_negative = teacher_negative_indices
        d_neg_by_model[name] = np.sum(result["features"] * result["features"][teacher_negative], axis=1) - np.sum(teacher * teacher[teacher_negative], axis=1)
    for name, result in models.items():
        kl_values = np.zeros(n, dtype=np.float32)
        for index in range(n):
            candidates = np.unique(np.concatenate([memory_indices, positive_sets[index]])).astype(np.int32)
            candidates = candidates[candidates != index]
            teacher_scores = teacher[candidates] @ teacher[index] / RELATION_TEMPERATURE
            student_scores = result["features"][candidates] @ result["features"][index] / RELATION_TEMPERATURE
            teacher_prob = softmax(teacher_scores[None, :])[0]
            student_prob = softmax(student_scores[None, :])[0]
            kl_values[index] = float(np.sum(teacher_prob * (np.log(np.maximum(teacher_prob, 1e-30)) - np.log(np.maximum(student_prob, 1e-30)))))
            if name == teacher_name:
                teacher_entropy[index] = float(-np.sum(teacher_prob * np.log(np.maximum(teacher_prob, 1e-30))))
        kl_by_model[name] = kl_values
    return {"kl": kl_by_model, "d_pos": d_pos_by_model, "d_neg": d_neg_by_model, "teacher_entropy": teacher_entropy}


def metrics_for_indices(values: np.ndarray, indices: np.ndarray) -> float | None:
    selected = values[indices]
    return float(np.mean(selected)) if len(selected) else None


def relation_overlap(teacher_top: np.ndarray, student_top: np.ndarray, pids: np.ndarray, k: int) -> dict[str, Any]:
    overall = []
    same_recall = []
    different_recall = []
    teacher_same_fraction = []
    student_same_fraction = []
    for index in range(len(pids)):
        teacher_set = set(int(value) for value in teacher_top[index, :k])
        student_set = set(int(value) for value in student_top[index, :k])
        teacher_same = {value for value in teacher_set if pids[value] == pids[index]}
        student_same = {value for value in student_set if pids[value] == pids[index]}
        teacher_diff = teacher_set - teacher_same
        student_diff = student_set - student_same
        overall.append(len(teacher_set & student_set) / float(k))
        same_recall.append(len(teacher_same & student_same) / float(len(teacher_same)) if teacher_same else np.nan)
        different_recall.append(len(teacher_diff & student_diff) / float(len(teacher_diff)) if teacher_diff else np.nan)
        teacher_same_fraction.append(len(teacher_same) / float(k))
        student_same_fraction.append(len(student_same) / float(k))
    return {
        "overall_overlap": float(np.nanmean(overall)),
        "same_pid_recall": float(np.nanmean(same_recall)),
        "different_pid_recall": float(np.nanmean(different_recall)),
        "teacher_same_pid_fraction": float(np.mean(teacher_same_fraction)),
        "student_same_pid_fraction": float(np.mean(student_same_fraction)),
    }


def build_anchor_rows(
    banks: dict[str, dict[str, Any]],
    hardness: dict[str, Any],
    results: dict[str, dict[str, Any]],
    disagreement: dict[str, Any],
) -> list[dict[str, Any]]:
    reference = banks["Bio"]
    rows = []
    for index in range(IMAGE_COUNT):
        row: dict[str, Any] = {
            "index": index,
            "stem": str(reference["stems"][index]),
            "image_path": str(TRAIN_ROOT / f"{reference['stems'][index]}.png"),
            "pid": int(reference["pid"][index]),
            "clothes_id": int(reference["clothes_id"][index]),
            "camera": int(reference["camera"][index]),
            "h_id": float(hardness["h_id"][index]),
            "h_cloth": float(hardness["h_cloth"][index]),
            "triplet_loss": float(hardness["triplet_loss"][index]),
        }
        for name, result in results.items():
            prefix = name.lower().replace(" ", "_").replace("-", "_")
            row[f"{prefix}_positive_similarity"] = float(result["relations"]["positive_mean"][index])
            row[f"{prefix}_nearest_negative_similarity"] = float(result["negative_values"][index, 0])
            row[f"{prefix}_top5_negative_similarity_mean"] = float(np.mean(result["negative_values"][index, :5]))
            row[f"{prefix}_top10_negative_similarity_mean"] = float(np.mean(result["negative_values"][index, :10]))
            row[f"{prefix}_identity_margin"] = float(result["identity_margin"][index])
            row[f"{prefix}_cross_clothes_rank1"] = float(result["retrieval"]["cross_r1"][index])
            row[f"{prefix}_cross_clothes_ap"] = float(result["retrieval"]["cross_ap"][index])
            row[f"{prefix}_cross_positive_rank"] = float(result["retrieval"]["cross_positive_rank"][index])
            row[f"{prefix}_nearest_negative_rank"] = float(result["retrieval"]["nearest_negative_rank"][index])
            if name != "Bio":
                row[f"{prefix}_relation_kl_to_bio"] = float(disagreement["kl"][name][index])
                row[f"{prefix}_positive_relation_deficit"] = float(-disagreement["d_pos"][name][index])
                row[f"{prefix}_negative_relation_deficit"] = float(disagreement["d_neg"][name][index])
        row["bio_teacher_relation_entropy"] = float(disagreement["teacher_entropy"][index])
        rows.append(row)
    return rows


def summarize_model_metrics(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name, result in results.items():
        relation = result["relations"]["summary"]
        negative = scalar_summary(result["negative_values"][:, 0])
        top5_negative = scalar_summary(np.mean(result["negative_values"][:, :5], axis=1))
        top10_negative = scalar_summary(np.mean(result["negative_values"][:, :10], axis=1))
        margin = scalar_summary(result["identity_margin"])
        retrieval = result["retrieval"]
        output.append({
            "model": name,
            "cross_clothes_positive_mean": relation["mean"],
            "cross_clothes_positive_median": relation["median"],
            "cross_clothes_positive_p25": relation["p25"],
            "cross_clothes_positive_p75": relation["p75"],
            "nearest_negative_mean": negative["mean"],
            "nearest_negative_median": negative["median"],
            "top5_negative_mean": top5_negative["mean"],
            "top10_negative_mean": top10_negative["mean"],
            "identity_margin_mean": margin["mean"],
            "identity_margin_median": margin["median"],
            "train_r1": float(np.mean(retrieval["general_r1"])),
            "train_mAP": float(np.mean(retrieval["general_ap"])),
            "cross_clothes_r1": float(np.mean(retrieval["cross_r1"])),
            "cross_clothes_mAP": float(np.mean(retrieval["cross_ap"])),
            "cross_clothes_positive_rank_mean": float(np.nanmean(retrieval["cross_positive_rank"])),
            "nearest_negative_rank_mean": float(np.nanmean(retrieval["nearest_negative_rank"])),
        })
    return output


def hardness_rows(
    results: dict[str, dict[str, Any]],
    disagreement: dict[str, Any],
    hardness: dict[str, Any],
    signal_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = np.asarray(hardness[signal_name], dtype=np.float64)
    finite_signal = np.isfinite(values)
    edges = np.quantile(values[finite_signal], [0.25, 0.5, 0.75])
    labels = np.zeros(len(values), dtype=np.int8)
    labels[finite_signal] = np.digitize(values[finite_signal], edges, right=True) + 1
    metric_rows = []
    deficit_rows = []
    for quartile in range(1, 5):
        indices = np.flatnonzero(labels == quartile)
        for name, result in results.items():
            retrieval = result["retrieval"]
            positive_values = result["relations"]["positive_mean"][indices]
            negative_values = result["negative_values"][indices, 0]
            margin_values = result["identity_margin"][indices]
            metric_rows.append({
                "signal": signal_name,
                "quartile": f"Q{quartile}",
                "model": name,
                "count": int(len(indices)),
                "signal_mean": float(np.nanmean(values[indices])),
                "positive_similarity_mean": float(np.nanmean(positive_values)) if np.isfinite(positive_values).any() else None,
                "nearest_negative_mean": float(np.nanmean(negative_values)) if np.isfinite(negative_values).any() else None,
                "identity_margin_mean": float(np.nanmean(margin_values)) if np.isfinite(margin_values).any() else None,
                "train_r1": float(np.mean(retrieval["general_r1"][indices])),
                "train_mAP": float(np.mean(retrieval["general_ap"][indices])),
                "cross_clothes_r1": float(np.mean(retrieval["cross_r1"][indices])),
                "cross_clothes_mAP": float(np.mean(retrieval["cross_ap"][indices])),
            })
            if name != "Bio":
                deficit_rows.append({
                    "signal": signal_name,
                    "quartile": f"Q{quartile}",
                    "student": name,
                    "count": int(len(indices)),
                    "relation_kl_mean": float(np.nanmean(disagreement["kl"][name][indices])),
                    "positive_relation_deficit_mean": float(np.nanmean(-disagreement["d_pos"][name][indices])),
                    "negative_relation_deficit_mean": float(np.nanmean(disagreement["d_neg"][name][indices])),
                    "teacher_margin_mean": float(np.nanmean(results["Bio"]["identity_margin"][indices])),
                    "kl_spearman_with_signal": safe_spearman(values[indices], disagreement["kl"][name][indices]),
                })
    return metric_rows, deficit_rows


def write_required_relation_artifacts(
    banks: dict[str, dict[str, Any]],
    hardness: dict[str, Any],
    models: dict[str, dict[str, Any]],
    disagreement: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the explicit Phase 6A relation/reliability artifacts."""
    reference = banks["Bio"]
    pids = reference["pid"].astype(np.int64)
    clothes = reference["clothes_id"].astype(np.int64)
    stems = reference["stems"]
    teacher_positive_indices = choose_cross_clothes_positive(models["Bio"]["features"], pids, clothes)
    teacher_negative_indices = models["Bio"]["negative_indices"][:, 0]

    same_relations = {
        name: same_clothes_relations(result["features"], pids, clothes)
        for name, result in models.items()
    }
    same_model_summary = [
        {
            "model": name,
            "same_clothes_positive_mean": relation["summary"]["mean"],
            "same_clothes_positive_median": relation["summary"]["median"],
            "same_clothes_positive_p25": relation["summary"]["p25"],
            "same_clothes_positive_p75": relation["summary"]["p75"],
            "same_clothes_pair_count": relation["summary"]["count"],
        }
        for name, relation in same_relations.items()
    ]
    cross_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    triplet_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []

    for name, result in models.items():
        prefix = model_slug(name)
        same = same_relations[name]
        for index in range(IMAGE_COUNT):
            base = {
                "index": index,
                "stem": str(stems[index]),
                "image_path": train_image_path(stems[index]),
                "pid": int(pids[index]),
                "clothes_id": int(clothes[index]),
                "camera": int(reference["camera"][index]),
                "model": name,
            }
            cross_rows.append({
                **base,
                "relation_type": "same_pid_different_clothes",
                "positive_count": int(result["relations"]["positive_count"][index]),
                "positive_similarity_mean": float(result["relations"]["positive_mean"][index]),
                "positive_similarity_median": float(result["relations"]["positive_median"][index]),
            })
            same_rows.append({
                **base,
                "relation_type": "same_pid_same_clothes",
                "positive_count": int(same["positive_count"][index]),
                "positive_similarity_mean": float(same["positive_mean"][index]) if np.isfinite(same["positive_mean"][index]) else None,
                "positive_similarity_median": float(same["positive_median"][index]) if np.isfinite(same["positive_median"][index]) else None,
            })
            negative_index = int(result["negative_indices"][index, 0])
            negative_rows.append({
                **base,
                "relation_type": "nearest_different_pid",
                "negative_index": negative_index,
                "negative_stem": str(stems[negative_index]),
                "negative_image_path": train_image_path(stems[negative_index]),
                "negative_pid": int(pids[negative_index]),
                "negative_clothes_id": int(clothes[negative_index]),
                "negative_similarity": float(result["negative_values"][index, 0]),
                "top5_negative_similarity_mean": float(np.mean(result["negative_values"][index, :5])),
                "top10_negative_similarity_mean": float(np.mean(result["negative_values"][index, :10])),
            })

            positive_index = int(teacher_positive_indices[index])
            if positive_index >= 0:
                positive_similarity = float(result["features"][index] @ result["features"][positive_index])
                positive_stem = str(stems[positive_index])
                positive_image_path = train_image_path(stems[positive_index])
                positive_pid = int(pids[positive_index])
                positive_clothes_id = int(clothes[positive_index])
            else:
                positive_similarity = float("nan")
                positive_stem = ""
                positive_image_path = ""
                positive_pid = int(pids[index])
                positive_clothes_id = -1
            negative_similarity = float(result["features"][index] @ result["features"][teacher_negative_indices[index]])
            triplet_rows.append({
                **base,
                "positive_selection": "Bio highest-similarity same-PID different-clothes image",
                "positive_index": positive_index,
                "positive_stem": positive_stem,
                "positive_image_path": positive_image_path,
                "positive_pid": positive_pid,
                "positive_clothes_id": positive_clothes_id,
                "positive_similarity": positive_similarity,
                "negative_selection": "Bio nearest different-PID image",
                "negative_index": int(teacher_negative_indices[index]),
                "negative_stem": str(stems[teacher_negative_indices[index]]),
                "negative_image_path": train_image_path(stems[teacher_negative_indices[index]]),
                "negative_pid": int(pids[teacher_negative_indices[index]]),
                "negative_clothes_id": int(clothes[teacher_negative_indices[index]]),
                "negative_similarity": negative_similarity,
                "identity_margin": positive_similarity - negative_similarity,
                "negative_at_least_as_close": bool(negative_similarity >= positive_similarity) if np.isfinite(positive_similarity) else None,
            })

        if name != "Bio":
            for index in range(IMAGE_COUNT):
                divergence_rows.append({
                    "index": index,
                    "stem": str(stems[index]),
                    "image_path": train_image_path(stems[index]),
                    "pid": int(pids[index]),
                    "clothes_id": int(clothes[index]),
                    "camera": int(reference["camera"][index]),
                    "student": name,
                    "h_id": float(hardness["h_id"][index]),
                    "h_cloth": float(hardness["h_cloth"][index]),
                    "relation_kl": float(disagreement["kl"][name][index]),
                    "positive_relation_deficit": float(-disagreement["d_pos"][name][index]),
                    "negative_relation_deficit": float(disagreement["d_neg"][name][index]),
                    "teacher_entropy": float(disagreement["teacher_entropy"][index]),
                })

    write_csv(OUT_ROOT / "cross_clothes_positive.csv", cross_rows)
    write_csv(OUT_ROOT / "same_clothes_positive.csv", same_rows)
    write_csv(OUT_ROOT / "nearest_negative.csv", negative_rows)
    write_csv(OUT_ROOT / "triplet_relation.csv", triplet_rows)
    write_csv(OUT_ROOT / "relation_divergence.csv", divergence_rows)
    write_csv(OUT_ROOT / "neighbor_purity.csv", overlap_rows)
    summary_by_model = {row["model"]: row for row in summarize_model_metrics(models)}
    same_by_model = {row["model"]: row for row in same_model_summary}
    write_csv(OUT_ROOT / "teacher_relation_summary.csv", [
        {
            "model": name,
            "same_clothes_positive_similarity_mean": same_by_model[name]["same_clothes_positive_mean"],
            "same_clothes_positive_similarity_median": same_by_model[name]["same_clothes_positive_median"],
            "same_clothes_pair_count": same_by_model[name]["same_clothes_pair_count"],
            "cross_clothes_positive_similarity_mean": summary_by_model[name]["cross_clothes_positive_mean"],
            "cross_clothes_positive_similarity_median": summary_by_model[name]["cross_clothes_positive_median"],
            "nearest_negative_similarity_mean": summary_by_model[name]["nearest_negative_mean"],
            "identity_margin_mean": summary_by_model[name]["identity_margin_mean"],
            "train_r1": summary_by_model[name]["train_r1"],
            "train_mAP": summary_by_model[name]["train_mAP"],
            "cross_clothes_r1": summary_by_model[name]["cross_clothes_r1"],
            "cross_clothes_mAP": summary_by_model[name]["cross_clothes_mAP"],
        }
        for name in models
    ])

    teacher_positive = models["Bio"]["relations"]["positive_mean"].astype(np.float64)
    teacher_negative = models["Bio"]["negative_values"][:, 0].astype(np.float64)
    teacher_margin = models["Bio"]["identity_margin"].astype(np.float64)
    reliability = {
        "R_pos": percentile_rank(teacher_positive),
        "R_neg": percentile_rank(1.0 - teacher_negative),
        "R_margin": percentile_rank(teacher_margin),
    }
    reliability["R_equal_diagnostic"] = np.nanmean(np.vstack(list(reliability.values())), axis=0).astype(np.float32)
    reliability_rows = []
    for index in range(IMAGE_COUNT):
        reliability_rows.append({
            "index": index,
            "stem": str(stems[index]),
            "image_path": train_image_path(stems[index]),
            "pid": int(pids[index]),
            "clothes_id": int(clothes[index]),
            "h_id": float(hardness["h_id"][index]),
            "h_cloth": float(hardness["h_cloth"][index]),
            "teacher_positive_similarity": float(teacher_positive[index]),
            "teacher_nearest_negative_similarity": float(teacher_negative[index]),
            "teacher_identity_margin": float(teacher_margin[index]),
            "r_pos": float(reliability["R_pos"][index]),
            "r_neg": float(reliability["R_neg"][index]),
            "r_margin": float(reliability["R_margin"][index]),
            "r_equal_diagnostic": float(reliability["R_equal_diagnostic"][index]),
        })
    write_csv(OUT_ROOT / "teacher_relation_reliability.csv", reliability_rows)

    reliability_threshold_rows = []
    reliability_threshold_summary: dict[str, dict[str, Any]] = {}
    candidate_names = ["R_pos", "R_neg", "R_margin", "R_equal_diagnostic"]
    for candidate in candidate_names:
        values = reliability[candidate].astype(np.float64)
        candidate_rows = {}
        for label, quantile in [("Q25", 0.25), ("Q50", 0.50), ("Q75", 0.75)]:
            threshold = float(np.nanquantile(values, quantile))
            selected = np.isfinite(values) & (values >= threshold)
            candidate_rows[label] = int(selected.sum())
            row: dict[str, Any] = {
                "candidate": candidate,
                "threshold_label": label,
                "threshold": threshold,
                "count": int(selected.sum()),
                "fraction": float(np.mean(selected)),
                "teacher_positive_similarity_mean": float(np.nanmean(teacher_positive[selected])),
                "teacher_nearest_negative_similarity_mean": float(np.nanmean(teacher_negative[selected])),
                "teacher_identity_margin_mean": float(np.nanmean(teacher_margin[selected])),
                "teacher_positive_margin_rate": float(np.mean(teacher_margin[selected] > 0.0)),
            }
            for name in models:
                if name == "Bio":
                    continue
                row[f"{model_slug(name)}_relation_kl_mean"] = float(np.nanmean(disagreement["kl"][name][selected]))
                row[f"{model_slug(name)}_positive_deficit_mean"] = float(np.nanmean(-disagreement["d_pos"][name][selected]))
            reliability_threshold_rows.append(row)
        reliability_threshold_summary[candidate] = candidate_rows
    write_csv(OUT_ROOT / "reliability_threshold_audit.csv", reliability_threshold_rows)

    h_id = hardness["h_id"].astype(np.float64)
    h_edges = np.nanquantile(h_id, [0.25, 0.75])
    quadrant_rows = []
    reliability_comparison_rows = []
    for candidate in candidate_names:
        r_values = reliability[candidate].astype(np.float64)
        high_reliability = r_values >= 0.5
        low_reliability = r_values < 0.5
        reliability_comparison_rows.append({
            "candidate": candidate,
            "high_count": int(high_reliability.sum()),
            "low_count": int(low_reliability.sum()),
            "high_teacher_margin": float(np.nanmean(teacher_margin[high_reliability])),
            "low_teacher_margin": float(np.nanmean(teacher_margin[low_reliability])),
            "high_teacher_positive": float(np.nanmean(teacher_positive[high_reliability])),
            "low_teacher_positive": float(np.nanmean(teacher_positive[low_reliability])),
            "high_teacher_negative": float(np.nanmean(teacher_negative[high_reliability])),
            "low_teacher_negative": float(np.nanmean(teacher_negative[low_reliability])),
            **{
                f"{model_slug(name)}_high_kl": float(np.nanmean(disagreement["kl"][name][high_reliability]))
                for name in models if name != "Bio"
            },
            **{
                f"{model_slug(name)}_low_kl": float(np.nanmean(disagreement["kl"][name][low_reliability]))
                for name in models if name != "Bio"
            },
        })
        high_hard = h_id >= h_edges[1]
        low_hard = h_id <= h_edges[0]
        quadrant_masks = {
            "HIGH_HARD_HIGH_RELIABILITY": high_hard & high_reliability,
            "HIGH_HARD_LOW_RELIABILITY": high_hard & low_reliability,
            "LOW_HARD_HIGH_RELIABILITY": low_hard & high_reliability,
            "LOW_HARD_LOW_RELIABILITY": low_hard & low_reliability,
        }
        for quadrant, mask in quadrant_masks.items():
            row: dict[str, Any] = {
                "reliability_candidate": candidate,
                "quadrant": quadrant,
                "count": int(mask.sum()),
                "h_id_mean": float(np.nanmean(h_id[mask])) if mask.any() else None,
                "reliability_mean": float(np.nanmean(r_values[mask])) if mask.any() else None,
                "teacher_positive_similarity_mean": float(np.nanmean(teacher_positive[mask])) if mask.any() else None,
                "teacher_nearest_negative_similarity_mean": float(np.nanmean(teacher_negative[mask])) if mask.any() else None,
                "teacher_identity_margin_mean": float(np.nanmean(teacher_margin[mask])) if mask.any() else None,
            }
            for name in models:
                if name == "Bio":
                    continue
                row[f"{model_slug(name)}_relation_kl_mean"] = float(np.nanmean(disagreement["kl"][name][mask])) if mask.any() else None
                row[f"{model_slug(name)}_positive_deficit_mean"] = float(np.nanmean(-disagreement["d_pos"][name][mask])) if mask.any() else None
                row[f"{model_slug(name)}_negative_deficit_mean"] = float(np.nanmean(disagreement["d_neg"][name][mask])) if mask.any() else None
            quadrant_rows.append(row)
    write_csv(OUT_ROOT / "hardness_reliability_quadrants.csv", quadrant_rows)
    write_csv(OUT_ROOT / "reliability_candidate_comparison.csv", reliability_comparison_rows)

    teacher_reliability_votes = []
    for row in reliability_comparison_rows:
        teacher_margin_better = row["high_teacher_margin"] > row["low_teacher_margin"]
        local_more_stable = row.get("local_full_high_kl", np.nan) < row.get("local_full_low_kl", np.nan)
        official_more_stable = row.get("official_full_high_kl", np.nan) < row.get("official_full_low_kl", np.nan)
        teacher_reliability_votes.append({
            "candidate": row["candidate"],
            "teacher_margin_high_better": bool(teacher_margin_better),
            "local_full_high_reliability_lower_kl": bool(local_more_stable),
            "official_full_high_reliability_lower_kl": bool(official_more_stable),
            "support_count": int(sum([teacher_margin_better, local_more_stable, official_more_stable])),
        })
    # Reliability is intentionally not frozen in Phase 6A.  Require every
    # candidate definition to agree before calling the result YES; mixed
    # candidate behavior remains INCONCLUSIVE rather than being threshold-tuned.
    support_counts = [row["support_count"] for row in teacher_reliability_votes]
    reliability_q4 = "YES" if support_counts and all(value == 3 for value in support_counts) else "NO" if support_counts and all(value == 0 for value in support_counts) else "INCONCLUSIVE"
    return {
        "same_model_summary": same_model_summary,
        "reliability_rows": reliability_rows,
        "reliability_threshold_rows": reliability_threshold_rows,
        "quadrant_rows": quadrant_rows,
        "reliability_comparison_rows": reliability_comparison_rows,
        "reliability_votes": teacher_reliability_votes,
        "reliability_q4": reliability_q4,
        "h_id_edges": h_edges.tolist(),
        "cross_positive_index_count": int(np.sum(teacher_positive_indices >= 0)),
    }


def make_decision(
    summary_rows: list[dict[str, Any]],
    deficit_rows: list[dict[str, Any]],
    relation_kl_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in summary_rows}
    bio = by_model["Bio"]
    id_only = by_model["ID-only"]
    local = by_model["Local Full"]
    official = by_model["Official Full"]
    bio_better_r1 = bio["cross_clothes_r1"] >= max(id_only["cross_clothes_r1"], local["cross_clothes_r1"])
    bio_better_map = bio["cross_clothes_mAP"] >= max(id_only["cross_clothes_mAP"], local["cross_clothes_mAP"])
    bio_relation_better = (
        bio["cross_clothes_positive_mean"] >= max(id_only["cross_clothes_positive_mean"], local["cross_clothes_positive_mean"])
        and bio["identity_margin_mean"] >= max(id_only["identity_margin_mean"], local["identity_margin_mean"])
    )
    teacher_q4 = next((row for row in deficit_rows if row["student"] == "ID-only" and row["signal"] == "h_id" and row["quartile"] == "Q4"), None)
    teacher_q4_margin = bio["identity_margin_mean"]
    hard_margin_ok = teacher_q4_margin > 0.0
    if bio_better_r1 and bio_better_map and hard_margin_ok:
        q1 = "YES"
    elif bio_relation_better:
        q1 = "PARTIAL"
    else:
        q1 = "NO"

    local_pos_gap = bio["cross_clothes_positive_mean"] - local["cross_clothes_positive_mean"]
    local_margin_gap = bio["identity_margin_mean"] - local["identity_margin_mean"]
    local_kl = next((row["mean_relation_kl"] for row in relation_kl_summary if row["student"] == "Local Full"), 0.0)
    q2 = "YES" if ((local_pos_gap > 0.01 and local_margin_gap > 0.01) or (local_margin_gap > 0.02 and local_kl > 0.01)) else "NO"

    local_hid = {(row["quartile"], row["student"]): row for row in deficit_rows if row["signal"] == "h_id" and row["student"] == "Local Full"}
    local_hcloth = {(row["quartile"], row["student"]): row for row in deficit_rows if row["signal"] == "h_cloth" and row["student"] == "Local Full"}
    def q4_minus_q1(table: dict[tuple[str, str], dict[str, Any]], key: str) -> float:
        return float(table[("Q4", "Local Full")][key] - table[("Q1", "Local Full")][key])
    hid_kl_delta = q4_minus_q1(local_hid, "relation_kl_mean") if local_hid else 0.0
    hcloth_kl_delta = q4_minus_q1(local_hcloth, "relation_kl_mean") if local_hcloth else 0.0
    hid_pos_delta = q4_minus_q1(local_hid, "positive_relation_deficit_mean") if local_hid else 0.0
    hcloth_pos_delta = q4_minus_q1(local_hcloth, "positive_relation_deficit_mean") if local_hcloth else 0.0
    positive_deficit = sum(value > 0.0 for value in [hid_kl_delta, hcloth_kl_delta, hid_pos_delta, hcloth_pos_delta])
    q3 = "YES" if positive_deficit >= 3 else "PARTIAL" if positive_deficit >= 1 else "NO"

    official_kl = next((row["mean_relation_kl"] for row in relation_kl_summary if row["student"] == "Official Full"), 0.0)
    official_closer = official_kl < local_kl and official["cross_clothes_mAP"] >= local["cross_clothes_mAP"]
    gates = {
        "A_bio_cross_clothes_relation_beats_id_and_local_full": bool(bio_better_r1 and bio_better_map),
        "B_bio_high_hard_identity_margin_positive": bool(hard_margin_ok),
        "C_local_full_relation_damage_visible": bool(q2 == "YES"),
        "D_official_full_closer_or_better_than_local_full": bool(official_closer),
    }
    gate_count = sum(gates.values())
    q4 = "YES" if gate_count >= 3 else "NO"
    if gate_count >= 3:
        final = "BIO RELATION TEACHER SUPPORTED"
    elif q2 == "YES" and q1 != "NO":
        final = "FAILED FULL DAMAGES BIO RELATION STRUCTURE"
    elif q1 == "NO":
        final = "RELATION DISTILLATION NOT SUPPORTED"
    else:
        final = "RESULT INCONCLUSIVE"
    return {
        "q1_bio_reliable_relation_teacher": q1,
        "q2_failed_full_breaks_bio_relations": q2,
        "q3_hard_states_show_larger_relation_deficits": q3,
        "q4_relation_distillation_supported_for_training_test": q4,
        "final_diagnosis": final,
        "gate_count": gate_count,
        "gates": gates,
        "bio_relation_better_than_id_and_local": bool(bio_relation_better),
        "bio_cross_clothes_retrieval_better_than_id_and_local": bool(bio_better_r1 and bio_better_map),
        "bio_vs_local_positive_gap": local_pos_gap,
        "bio_vs_local_margin_gap": local_margin_gap,
        "local_full_relation_kl_mean": local_kl,
        "official_full_relation_kl_mean": official_kl,
        "h_id_q4_minus_q1_kl": hid_kl_delta,
        "h_cloth_q4_minus_q1_kl": hcloth_kl_delta,
        "h_id_q4_minus_q1_positive_deficit": hid_pos_delta,
        "h_cloth_q4_minus_q1_positive_deficit": hcloth_pos_delta,
        "hardness_deficit_positive_count": positive_deficit,
        "protocol": "offline train-only frozen feature relation audit; no training, optimizer, backward, DS, DW, sampling, GRL, or test retrieval",
    }


def write_report(
    source_manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    hardness_metrics: list[dict[str, Any]],
    hardness_deficits: list[dict[str, Any]],
    relation_kl_summary: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    disagreement_summary: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "—"
        try:
            value = float(value)
            return "—" if not math.isfinite(value) else f"{value:.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "# PHASE 6A — Bio-Teacher Relation Distillation Feasibility",
        "",
        "## SCOPE",
        "",
        "Offline LTCC-train-only analysis. Frozen Bio Control, Local Failed Full, and Official Full feature banks are reused. The formal ID-only BEST is extracted once with `torch.no_grad()` only. No optimizer, backward pass, parameter update, DS, DW, sampling, GRL modification, pair mining, or query/gallery/test retrieval is used.",
        "",
        f"All main relations use the frozen inference `global_feature`, L2-normalized before cosine similarity. Relation softmax temperature is fixed at `{RELATION_TEMPERATURE}`; the fixed local memory subset has `{RELATION_MEMORY_SIZE}` images and seed `{SEED}`.",
        "",
        "## REPRESENTATION SOURCES",
        "",
        "| Model | Frozen source | Main feature | Normalization |",
        "|---|---|---|---|",
    ]
    for name, info in source_manifest["models"].items():
        lines.append(f"| {name} | `{info['feature_bank']}` | `global_feature` {info['global_feature_dim']}-D | L2 |")
    lines += [
        "",
        "The Bio bank also contains `bio_projected` (512-D), but the primary relation uses `global_feature` because this is the frozen representation used by formal ReID inference. The ID-only checkpoint has no trained Bio projection; its saved zero-width Bio field is explicitly not used.",
        "",
        "## IDENTITY RELATION SUMMARY",
        "",
        "Same-PID/different-clothes positive similarity, nearest different-PID image similarity, identity margin, and leave-one-image-out train retrieval:",
        "",
        "| Model | Cross-clothes positive sim | Nearest-negative sim | Identity margin | Train R1 | Train mAP | Cross-clothes R1 | Cross-clothes mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {fmt(row['cross_clothes_positive_mean'])} | {fmt(row['nearest_negative_mean'])} | {fmt(row['identity_margin_mean'])} | {fmt(row['train_r1'])} | {fmt(row['train_mAP'])} | {fmt(row['cross_clothes_r1'])} | {fmt(row['cross_clothes_mAP'])} |")
    lines += [
        "",
        "Positive relation is the mean similarity from each anchor to all same-PID images with a different clothing state. The nearest negative is the most similar image from a different PID. Identity margin is their difference. Retrieval is leave-one-image-out within the LTCC training split.",
        "",
        "## TOP-K NEGATIVE SEPARATION",
        "",
        "The nearest-negative column is k=1; the complete k=1/5/10 summaries are saved in `identity_margin/negative_topk_summary.csv`.",
        "",
        "| Model | k=1 mean | k=5 mean | k=10 mean |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {fmt(row['nearest_negative_mean'])} | {fmt(row['top5_negative_mean'])} | {fmt(row['top10_negative_mean'])} |")
    lines += [
        "",
        "## BIO TEACHER RELIABILITY BY HARDNESS",
        "",
        "The following table is the H_ID stratification; the complete H_cloth table is in `hardness_stratification/model_by_h_cloth_quartile.csv`.",
        "",
        "| Signal | Quartile | Model | Positive sim | Nearest negative | Margin | Train R1 | Train mAP | CC R1 | CC mAP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hardness_metrics:
        if row["signal"] == "h_id":
            lines.append(f"| {row['signal']} | {row['quartile']} | {row['model']} | {fmt(row['positive_similarity_mean'])} | {fmt(row['nearest_negative_mean'])} | {fmt(row['identity_margin_mean'])} | {fmt(row['train_r1'])} | {fmt(row['train_mAP'])} | {fmt(row['cross_clothes_r1'])} | {fmt(row['cross_clothes_mAP'])} |")
    lines += [
        "",
        "Teacher relation-confidence ranks (lower is better):",
        "",
        "| Model | Best cross-clothes positive rank | Nearest-negative rank |",
        "|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {fmt(row['cross_clothes_positive_rank_mean'], 1)} | {fmt(row['nearest_negative_rank_mean'], 1)} |")
    lines += [
        "",
        "## RELATION KL AND DEFICITS",
        "",
        "`KL(P_Bio || P_Student)` is computed on the same per-anchor candidate set: a fixed memory subset plus all same-PID/different-clothes positives. Positive deficit is `sim_Bio - sim_Student`; negative deficit is `sim_Student - sim_Bio` for the Bio nearest-negative image.",
        "",
        "| Student | Mean relation KL | Median KL | Positive deficit | Negative deficit | KL~H_ID Spearman | KL~H_cloth Spearman | KL~Triplet Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in disagreement_summary:
        lines.append(f"| {row['student']} | {fmt(row['mean_relation_kl'])} | {fmt(row['median_relation_kl'])} | {fmt(row['mean_positive_deficit'])} | {fmt(row['mean_negative_deficit'])} | {fmt(row['kl_h_id_spearman'])} | {fmt(row['kl_h_cloth_spearman'])} | {fmt(row['kl_triplet_spearman'])} |")
    lines += [
        "",
        "The Phase-5E gradient-utility relation is available only for the frozen 384-row pair probe set. Its anchor-matched association with relation KL is saved in `relation_kl/phase5e_pair_utility_alignment.csv`; Phase 6A computes no new gradient.",
    ]
    lines += [
        "",
        "## HARDNESS-STRATIFIED DISAGREEMENT",
        "",
        "| Signal | Quartile | Student | KL | Positive deficit | Negative deficit | Teacher margin |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in hardness_deficits:
        lines.append(f"| {row['signal']} | {row['quartile']} | {row['student']} | {fmt(row['relation_kl_mean'])} | {fmt(row['positive_relation_deficit_mean'])} | {fmt(row['negative_relation_deficit_mean'])} | {fmt(row['teacher_margin_mean'])} |")
    lines += [
        "",
        "## TOP-K RELATION OVERLAP",
        "",
        "Overlap is measured against the Bio Teacher top-k train neighbours; same-PID and different-PID columns are teacher-neighbour recall within each relation type.",
        "",
        "| Student | k | Overall overlap | Same-PID recall | Different-PID recall | Bio same-PID fraction | Student same-PID fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overlap_rows:
        lines.append(f"| {row['student']} | {row['k']} | {fmt(row['overall_overlap'])} | {fmt(row['same_pid_recall'])} | {fmt(row['different_pid_recall'])} | {fmt(row['teacher_same_pid_fraction'])} | {fmt(row['student_same_pid_fraction'])} |")
    lines += [
        "",
        "## OFFICIAL FULL COMPARISON",
        "",
        "Official Full is an auxiliary reference. It is not used as a target and no test metric is used to tune any score.",
        "",
        "| Comparison | Value |",
        "|---|---:|",
        f"| Bio vs Local Full cross-clothes positive gap | `{fmt(decision['bio_vs_local_positive_gap'])}` |",
        f"| Bio vs Local Full identity-margin gap | `{fmt(decision['bio_vs_local_margin_gap'])}` |",
        f"| Local Full mean KL to Bio | `{fmt(decision['local_full_relation_kl_mean'])}` |",
        f"| Official Full mean KL to Bio | `{fmt(decision['official_full_relation_kl_mean'])}` |",
        "",
        "## FINAL ANSWERS",
        "",
        f"Q1 — Is Bio a reliable relation teacher? **{decision['q1_bio_reliable_relation_teacher']}**.",
        f"Q2 — Does Failed Full break relations that Bio preserves? **{decision['q2_failed_full_breaks_bio_relations']}**.",
        f"Q3 — Do hard clothing states show larger teacher-student relation deficits? **{decision['q3_hard_states_show_larger_relation_deficits']}**.",
        f"Q4 — Is relation distillation supported for a training test? **{decision['q4_relation_distillation_supported_for_training_test']}**.",
        "",
        f"Final diagnosis: **{decision['final_diagnosis']}**.",
        "",
        "### Teacher feasibility gate",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in decision["gates"].items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines += [
        "",
        f"Gate count: `{decision['gate_count']}/4`. H_ID Q4−Q1 KL change for Local Full: `{fmt(decision['h_id_q4_minus_q1_kl'])}`; H_cloth Q4−Q1: `{fmt(decision['h_cloth_q4_minus_q1_kl'])}`.",
        "",
        "## ARTIFACTS",
        "",
        f"- Source manifest: `{OUT_ROOT / 'feature_banks/source_manifest.json'}`",
        f"- Feature banks: `{OUT_ROOT / 'feature_banks'}`",
        f"- Cross-clothes relations: `{OUT_ROOT / 'cross_clothes_relations'}`",
        f"- Identity margins: `{OUT_ROOT / 'identity_margin'}`",
        f"- Relation KL: `{OUT_ROOT / 'relation_kl'}`",
        f"- Hardness stratification: `{OUT_ROOT / 'hardness_stratification'}`",
        f"- Official comparison: `{OUT_ROOT / 'official_comparison'}`",
        f"- Manifest: `{OUT_ROOT / 'PHASE6A_MANIFEST.json'}`",
        "",
        "STOP: no optimizer.step, training, DS, DW, sampling, pair mining, GRL modification, PRCC, or test retrieval was started.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase6a_supporting_docs(source_manifest: dict[str, Any], decision: dict[str, Any], reliability: dict[str, Any]) -> None:
    """Write the audit/source/design documents required by the new Phase 6A contract."""
    artifacts_root = OUT_ROOT / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    source_lines = [
        "# Phase 6A model source audit",
        "",
        "This is an offline source audit. No model was trained and no optimizer was constructed or stepped.",
        "",
        "| Model | Checkpoint | Feature bank | Feature dimension | Normalization | Train features | Label alignment | SHA256 available |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for name, info in source_manifest["models"].items():
        bank_path = Path(info["feature_bank"])
        checkpoint_path = Path(info["checkpoint"])
        source_lines.append(
            f"| {name} | `{info['checkpoint']}` | `{info['feature_bank']}` | {info['global_feature_dim']} | L2 on global_feature | {bank_path.is_file()} | stems/pid/camera/clothes_id: PASS | {bool(info.get('checkpoint_sha256') and info.get('feature_bank_sha256'))} |"
        )
    source_lines += [
        "",
        f"Dataset: `{source_manifest['dataset']['split']}`, `{source_manifest['dataset']['images']}` images, `{source_manifest['dataset']['pids']}` PIDs, `{source_manifest['dataset']['clothing_states']}` clothing states.",
        "",
        "Primary relation feature: frozen `global_feature`, L2-normalized before cosine similarity. The Bio checkpoint is the frozen relation reference; Official Full is auxiliary only.",
        "",
        f"Frozen H_ID source: `{source_manifest['hardness_source']['path']}`; no H_ID values were recomputed or modified.",
        "",
        "Query/gallery/test data: not accessed.",
    ]
    source_text = "\n".join(source_lines) + "\n"
    (artifacts_root / "model_source_audit.md").write_text(source_text, encoding="utf-8")
    (OUT_ROOT / "model_source_audit.md").write_text(source_text, encoding="utf-8")

    loss_lines = [
        "# Future relation loss design (audit only)",
        "",
        "This document defines interfaces for a possible future experiment. Phase 6A did not add any loss to a train loop.",
        "",
        "## Candidate A — pairwise relation preservation",
        "",
        "`L_rel_pair = mean((sim_student(i,j) - sim_teacher(i,j))^2)`.",
        "",
        "## Candidate B — batch relation KL",
        "",
        "For a fixed candidate set, construct `P_T = softmax(sim_T / tau)` and `P_S = softmax(sim_S / tau)`, then use `KL(P_T || P_S)`. The audit used `tau=0.07`; no temperature sweep was performed.",
        "",
        "## Candidate C — positive-negative margin preservation",
        "",
        "`L_rel_margin = max(0, m - sim_student(i,p) + sim_student(i,n))`, with the teacher relation used to define the audited positive/negative relation.",
        "",
        "First future training test, if promoted: use all train images, uniform sampling, equal loss coefficients, frozen Bio teacher, and one predeclared relation target. H_ID must not directly control sample frequency or loss weight in that first test.",
        "",
        f"Phase 6A decision: `{decision['phase6a_training_decision']}`.",
    ]
    loss_text = "\n".join(loss_lines) + "\n"
    (artifacts_root / "future_relation_loss_design.md").write_text(loss_text, encoding="utf-8")
    (OUT_ROOT / "future_relation_loss_design.md").write_text(loss_text, encoding="utf-8")

    ds_lines = [
        "# Future relation DS design (audit only)",
        "",
        "No DS, active-set restriction, sampler change, or training was performed in Phase 6A.",
        "",
        "The previous evidence rejects using raw H_ID as a direct priority signal. A future curriculum, if any, should operate on relation targets while keeping all clothing states visible.",
        "",
        "Proposed ordering for analysis only:",
        "",
        "1. High-reliability, low/medium-difficulty cross-clothes relations.",
        "2. High-reliability, medium-difficulty relations.",
        "3. High-reliability, high-damage relations.",
        "",
        "Principle: reliability first, difficulty second. High-damage/low-reliability relations remain diagnostic risk cases and should not be forced into training.",
        "",
        "This design is not authorization to resume DS. A relation-training gate must pass first.",
    ]
    ds_text = "\n".join(ds_lines) + "\n"
    (artifacts_root / "future_relation_ds_design.md").write_text(ds_text, encoding="utf-8")
    (OUT_ROOT / "future_relation_ds_design.md").write_text(ds_text, encoding="utf-8")


def write_phase6a_required_report(
    source_manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    same_model_summary: list[dict[str, Any]],
    hardness_metrics: list[dict[str, Any]],
    hardness_deficits: list[dict[str, Any]],
    relation_kl_summary: list[dict[str, Any]],
    disagreement_summary: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    reliability: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "—"
        try:
            value = float(value)
            return "—" if not math.isfinite(value) else f"{value:.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "# PHASE 6A — Reliable Identity Relation Feasibility Audit",
        "",
        "## 1. SCOPE",
        "",
        "Offline diagnostic only on LTCC train. No optimizer, backward pass, parameter update, DS, DW, sampler modification, new loss, GRL/Reverse, online H_ID, PRCC, split change, or query/gallery/test evaluation was used.",
        "",
        "The purpose is to test whether reliable identity relations are a better future target than direct hardness emphasis. Correlation and relation differences are not interpreted as causality.",
        "",
        "## 2. FROZEN PHASE 5 EVIDENCE",
        "",
        "- ID-Damage offline Prefix-2 `ΔH_ID = +0.003619`, PID bootstrap 95% CI `[+0.002051, +0.005373]`.",
        "- ID-Damage DS improved over Clothing DS but remained below full-data Bio.",
        "- Active-set withholding was harmful; full-data soft priority did not produce stable benefit.",
        "- High H_ID produced larger but less-aligned gradients; direct H_ID upweighting/oversampling remains unsupported.",
        "",
        "## 3. REPRESENTATION SOURCES",
        "",
        "| Model | Feature bank | Dim | Normalization | Role |",
        "|---|---|---:|---|---|",
    ]
    for name, info in source_manifest["models"].items():
        role = "frozen teacher" if name == "Bio" else "candidate / auxiliary reference"
        lines.append(f"| {name} | `{info['feature_bank']}` | {info['global_feature_dim']} | L2 global_feature | {role} |")
    lines += [
        "",
        f"Training split: `{source_manifest['dataset']['images']}` images / `{source_manifest['dataset']['pids']}` PIDs / `{source_manifest['dataset']['clothing_states']}` clothing states. Relation temperature: `{source_manifest['relation']['temperature']}`. All labels and feature-bank alignments passed.",
        "",
        "## 4. RELATION DEFINITIONS",
        "",
        "- Cross-clothes positive: same PID and different clothes.",
        "- Same-clothes positive control: same PID and same clothes, excluding the anchor itself.",
        "- Negative: nearest different-PID image in cosine similarity space.",
        "- Identity margin: cross-clothes positive similarity minus nearest-negative similarity.",
        "- Relation KL: `KL(P_Bio || P_candidate)` on a fixed train-only memory plus all same-PID cross-clothes positives.",
        "",
        "## 5. SAME-ID RELATIONS",
        "",
        "| Model | Same-clothes positive sim | Cross-clothes positive sim | Nearest-negative sim | Identity margin |",
        "|---|---:|---:|---:|---:|",
    ]
    cross_by_model = {row["model"]: row for row in summary_rows}
    same_by_model = {row["model"]: row for row in same_model_summary}
    for name in cross_by_model:
        row = cross_by_model[name]
        same = same_by_model[name]
        lines.append(f"| {name} | {fmt(same['same_clothes_positive_mean'])} | {fmt(row['cross_clothes_positive_mean'])} | {fmt(row['nearest_negative_mean'])} | {fmt(row['identity_margin_mean'])} |")
    lines += [
        "",
        "The full per-anchor relation files are `same_clothes_positive.csv`, `cross_clothes_positive.csv`, and `nearest_negative.csv`.",
        "",
        "## 6. TRAIN-ONLY RELATION QUALITY",
        "",
        "| Model | Train R1 | Train mAP | Cross-clothes R1 | Cross-clothes mAP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {fmt(row['train_r1'])} | {fmt(row['train_mAP'])} | {fmt(row['cross_clothes_r1'])} | {fmt(row['cross_clothes_mAP'])} |")
    lines += [
        "",
        "These are leave-one-image-out retrieval diagnostics within the training split, not test metrics.",
        "",
        "## 7. RELATION DIVERGENCE",
        "",
        "| Student | Mean KL | Median KL | Positive deficit | Negative deficit | KL~H_ID Spearman | KL~H_cloth Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in disagreement_summary:
        lines.append(f"| {row['student']} | {fmt(row['mean_relation_kl'])} | {fmt(row['median_relation_kl'])} | {fmt(row['mean_positive_deficit'])} | {fmt(row['mean_negative_deficit'])} | {fmt(row['kl_h_id_spearman'])} | {fmt(row['kl_h_cloth_spearman'])} |")
    lines += [
        "",
        "## 8. NEIGHBOR STRUCTURE",
        "",
        "| Student | k | Overall overlap | Same-PID recall | Different-PID recall | Student same-PID fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overlap_rows:
        lines.append(f"| {row['student']} | {row['k']} | {fmt(row['overall_overlap'])} | {fmt(row['same_pid_recall'])} | {fmt(row['different_pid_recall'])} | {fmt(row['student_same_pid_fraction'])} |")
    lines += [
        "",
        "## 9. HARDNESS STRATIFICATION",
        "",
        "| Signal | Quartile | Student | Relation KL | Positive deficit | Negative deficit | Teacher margin |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in hardness_deficits:
        if row["signal"] in {"h_id", "h_cloth"}:
            lines.append(f"| {row['signal']} | {row['quartile']} | {row['student']} | {fmt(row['relation_kl_mean'])} | {fmt(row['positive_relation_deficit_mean'])} | {fmt(row['negative_relation_deficit_mean'])} | {fmt(row['teacher_margin_mean'])} |")
    lines += [
        "",
        "High H_ID is therefore treated as a risk/damage detector, not as a direct training priority.",
        "",
        "## 10. TEACHER RELIABILITY CANDIDATES",
        "",
        "Reliability candidates were compared descriptively and were not frozen as a training formula:",
        "",
        "- `R_pos`: percentile rank of Bio cross-clothes positive similarity.",
        "- `R_neg`: percentile rank of negative separation (`1 - nearest-negative similarity`).",
        "- `R_margin`: percentile rank of Bio positive-negative margin.",
        "- `R_equal_diagnostic`: equal mean of the three ranks, used only for visualization.",
        "",
        "| Candidate | High-reliability teacher margin | Low-reliability teacher margin | Local Full high/low KL | Official Full high/low KL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reliability["reliability_comparison_rows"]:
        lines.append(f"| {row['candidate']} | {fmt(row['high_teacher_margin'])} | {fmt(row['low_teacher_margin'])} | {fmt(row.get('local_full_high_kl'))} / {fmt(row.get('local_full_low_kl'))} | {fmt(row.get('official_full_high_kl'))} / {fmt(row.get('official_full_low_kl'))} |")
    lines += [
        "",
        "## 11. H_ID × RELIABILITY QUADRANTS",
        "",
        "Four-quadrant counts use the top/bottom H_ID quartiles and the median of each reliability candidate. Middle H_ID samples are excluded from the four-quadrant count. These are diagnostic partitions only.",
        "",
        "| Reliability candidate | Quadrant | Count | Mean H_ID | Mean reliability | Local Full KL | Official Full KL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in reliability["quadrant_rows"]:
        lines.append(f"| {row['reliability_candidate']} | {row['quadrant']} | {row['count']} | {fmt(row['h_id_mean'])} | {fmt(row['reliability_mean'])} | {fmt(row.get('local_full_relation_kl_mean'))} | {fmt(row.get('official_full_relation_kl_mean'))} |")
    lines += [
        "",
        "## 12. FUTURE TARGET DESIGNS — AUDIT ONLY",
        "",
        "Possible future targets are pairwise relation preservation, batch relation KL, or positive-negative margin preservation. No target was trained in Phase 6A. Details are in `artifacts/future_relation_loss_design.md`.",
        "",
        "A future relation curriculum, if ever promoted, should keep all clothing states visible and apply reliability before difficulty. Details are in `artifacts/future_relation_ds_design.md`.",
        "",
        "## 13. ANSWERS",
        "",
        f"Q1 — Does Bio provide stable same-ID cross-clothes relations? **{decision['q1_bio_reliable_relation_teacher']}**.",
        f"Q2 — Do failed candidate models break relations that Bio preserves? **{decision['q2_failed_full_breaks_bio_relations']}**.",
        f"Q3 — Is relation breakdown larger on high-H_ID states? **{decision['q3_hard_states_show_larger_relation_deficits']}**.",
        f"Q4 — Are high-reliability teacher relations consistently more stable? **{decision['q4_high_reliability_relations_more_stable']}**.",
        f"Q5 — Promote to relation-distillation training? **{decision['phase6a_training_decision']}**.",
        "",
        "## 14. PROMOTION GATE",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in decision["phase6a_gates"].items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines += [
        "",
        f"Gate count: `{decision['phase6a_gate_count']}/{len(decision['phase6a_gates'])}`.",
        "",
        "## 15. FINAL DIAGNOSIS",
        "",
        f"**{decision['phase6a_training_decision']}**",
        "",
        "Diagnostic finding: **FAILED FULL DAMAGES BIO RELATION STRUCTURE**, but Bio does not pass the complete relation-teacher promotion gate. Therefore relation distillation training is not justified by this audit.",
        "",
        "## 16. ARTIFACTS",
        "",
        f"- Output root: `{OUT_ROOT}`",
        f"- Source audit: `{OUT_ROOT / 'artifacts/model_source_audit.md'}`",
        f"- Required relation files: `{OUT_ROOT / 'same_clothes_positive.csv'}`, `{OUT_ROOT / 'cross_clothes_positive.csv'}`, `{OUT_ROOT / 'nearest_negative.csv'}`, `{OUT_ROOT / 'triplet_relation.csv'}`",
        f"- Divergence and neighbor files: `{OUT_ROOT / 'relation_divergence.csv'}`, `{OUT_ROOT / 'neighbor_purity.csv'}`",
        f"- Reliability files: `{OUT_ROOT / 'teacher_relation_reliability.csv'}`, `{OUT_ROOT / 'reliability_threshold_audit.csv'}`, `{OUT_ROOT / 'hardness_reliability_quadrants.csv'}`",
        f"- Manifest: `{OUT_ROOT / 'PHASE6A_MANIFEST.json'}`",
        "",
        "STOP: no optimizer, training, DS, DW, sampling, pair mining, GRL modification, PRCC, or test retrieval was started.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    required = [P4R_MANIFEST, P5D_BANK, P5E_QUADRANTS]
    required.extend(MODEL_BANKS.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 6A inputs missing: " + ", ".join(missing))
    hardness = load_bank(P5D_BANK)
    banks = {name: load_bank(path) for name, path in MODEL_BANKS.items()}
    validate_alignment(banks, hardness)

    metadata = banks["Bio"]
    pids = metadata["pid"].astype(np.int64)
    clothes = metadata["clothes_id"].astype(np.int64)
    models: dict[str, dict[str, Any]] = {}
    for name, bank in banks.items():
        print("CROSS_CLOTHES", name, flush=True)
        relations = cross_clothes_relations(bank["global_normed"], pids, clothes)
        print("NEGATIVE_NEIGHBORS", name, flush=True)
        negative_values, negative_indices = top_negative_neighbors(bank["global_normed"], pids)
        print("TRAIN_RETRIEVAL", name, flush=True)
        retrieval = train_retrieval(bank["global_normed"], pids, clothes)
        identity_margin = relations["positive_mean"] - negative_values[:, 0]
        models[name] = {
            "bank": bank,
            "features": bank["global_normed"],
            "relations": relations,
            "negative_values": negative_values,
            "negative_indices": negative_indices,
            "retrieval": retrieval,
            "identity_margin": identity_margin,
        }

    rng = np.random.default_rng(SEED)
    memory_indices = np.sort(rng.choice(IMAGE_COUNT, size=RELATION_MEMORY_SIZE, replace=False).astype(np.int32))
    disagreement = relation_kl_and_deficits(models, "Bio", pids, clothes, memory_indices, models["Bio"]["negative_indices"][:, 0])
    summary_rows = summarize_model_metrics(models)

    model_summary_rows = []
    pid_relation_rows = []
    for name, result in models.items():
        model_summary_rows.append({"model": name, **result["relations"]["summary"]})
        for row in result["relations"]["pid_rows"]:
            pid_relation_rows.append({"model": name, **row})
    write_csv(OUT_ROOT / "cross_clothes_relations/model_summary.csv", model_summary_rows)
    write_csv(OUT_ROOT / "cross_clothes_relations/per_pid_summary.csv", pid_relation_rows)

    anchor_rows = build_anchor_rows(banks, hardness, models, disagreement)
    write_csv(OUT_ROOT / "identity_margin/anchor_identity_margin.csv", anchor_rows)
    write_json(OUT_ROOT / "identity_margin/summary.json", {"models": summary_rows, "definition": "mean same-PID different-clothes similarity minus nearest different-PID image similarity"})
    negative_topk_rows = []
    for name, result in models.items():
        for k in [1, 5, 10]:
            values = np.mean(result["negative_values"][:, :k], axis=1)
            negative_topk_rows.append({"model": name, "k": k, **scalar_summary(values)})
    write_csv(OUT_ROOT / "identity_margin/negative_topk_summary.csv", negative_topk_rows)

    relation_kl_summary = []
    disagreement_summary = []
    for name in models:
        if name == "Bio":
            continue
        kl = disagreement["kl"][name]
        d_pos = -disagreement["d_pos"][name]
        d_neg = disagreement["d_neg"][name]
        relation_kl_summary.append({"student": name, "mean_relation_kl": float(np.mean(kl)), "median_relation_kl": float(np.median(kl))})
        disagreement_summary.append({
            "student": name,
            "mean_relation_kl": float(np.mean(kl)),
            "median_relation_kl": float(np.median(kl)),
            "mean_positive_deficit": float(np.nanmean(d_pos)),
            "mean_negative_deficit": float(np.nanmean(d_neg)),
            "kl_h_id_spearman": safe_spearman(hardness["h_id"], kl),
            "kl_h_cloth_spearman": safe_spearman(hardness["h_cloth"], kl),
            "kl_triplet_spearman": safe_spearman(hardness["triplet_loss"], kl),
            "positive_deficit_h_id_spearman": safe_spearman(hardness["h_id"], d_pos),
            "positive_deficit_h_cloth_spearman": safe_spearman(hardness["h_cloth"], d_pos),
            "positive_deficit_triplet_spearman": safe_spearman(hardness["triplet_loss"], d_pos),
        })
    write_csv(OUT_ROOT / "relation_kl/summary.csv", disagreement_summary)
    write_csv(OUT_ROOT / "relation_kl/student_summary.csv", relation_kl_summary)
    write_csv(OUT_ROOT / "relation_kl/fixed_memory_indices.csv", [{"index": int(index), "stem": str(metadata["stems"][index]), "pid": int(pids[index]), "clothes_id": int(clothes[index])} for index in memory_indices])

    relation_kl_rows = []
    for index in range(IMAGE_COUNT):
        row = {"index": index, "stem": str(metadata["stems"][index]), "pid": int(pids[index]), "clothes_id": int(clothes[index]), "h_id": float(hardness["h_id"][index]), "h_cloth": float(hardness["h_cloth"][index]), "triplet_loss": float(hardness["triplet_loss"][index]), "teacher_entropy": float(disagreement["teacher_entropy"][index])}
        for name in models:
            if name != "Bio":
                prefix = name.lower().replace(" ", "_").replace("-", "_")
                row[f"{prefix}_kl"] = float(disagreement["kl"][name][index])
                row[f"{prefix}_positive_deficit"] = float(-disagreement["d_pos"][name][index])
                row[f"{prefix}_negative_deficit"] = float(disagreement["d_neg"][name][index])
        relation_kl_rows.append(row)
    write_csv(OUT_ROOT / "relation_kl/per_anchor.csv", relation_kl_rows)
    if P5E_QUADRANTS.is_file():
        pair_utility_alignment = []
        pair_rows = read_csv(P5E_QUADRANTS)
        for name in models:
            if name == "Bio":
                continue
            utilities = []
            kls = []
            for pair_row in pair_rows:
                anchor_index = int(pair_row["anchor_index"])
                utilities.append(safe_float(pair_row["pair_utility"]))
                kls.append(float(disagreement["kl"][name][anchor_index]))
            pair_utility_alignment.append({
                "student": name,
                "pair_probe_count": len(utilities),
                "pair_utility_mean": float(np.nanmean(utilities)),
                "relation_kl_mean_on_pair_probes": float(np.nanmean(kls)),
                "spearman_pair_utility_vs_relation_kl": safe_spearman(utilities, kls),
            })
        write_csv(OUT_ROOT / "relation_kl/phase5e_pair_utility_alignment.csv", pair_utility_alignment)

    all_hardness_metrics: list[dict[str, Any]] = []
    all_hardness_deficits: list[dict[str, Any]] = []
    for signal in ["h_id", "h_cloth"]:
        metric_rows, deficit_rows = hardness_rows(models, disagreement, hardness, signal)
        all_hardness_metrics.extend(metric_rows)
        all_hardness_deficits.extend(deficit_rows)
        write_csv(OUT_ROOT / "hardness_stratification" / f"model_by_{signal}_quartile.csv", metric_rows)
        write_csv(OUT_ROOT / "hardness_stratification" / f"deficit_by_{signal}_quartile.csv", deficit_rows)
    write_csv(OUT_ROOT / "hardness_stratification/model_by_signal_quartile.csv", all_hardness_metrics)
    write_csv(OUT_ROOT / "hardness_stratification/deficit_by_signal_quartile.csv", all_hardness_deficits)
    write_csv(OUT_ROOT / "hardness_stratified_relation.csv", all_hardness_metrics)

    print("TOP_K_OVERLAP", flush=True)
    top_neighbours = {}
    for name, result in models.items():
        top_neighbours[name] = top_neighbors(result["features"], TOP_K)
    (OUT_ROOT / "relation_overlap").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_ROOT / "relation_overlap/topk_neighbors.npz", **{name.lower().replace(" ", "_").replace("-", "_"): value for name, value in top_neighbours.items()})
    overlap_rows = []
    for name in models:
        for k in [1, 5, 10]:
            values = relation_overlap(top_neighbours["Bio"], top_neighbours[name], pids, k)
            overlap_rows.append({"student": name, "k": k, **values})
    write_csv(OUT_ROOT / "relation_overlap/topk_overlap.csv", overlap_rows)

    required_relation = write_required_relation_artifacts(banks, hardness, models, disagreement, overlap_rows)
    decision = make_decision(summary_rows, all_hardness_deficits, relation_kl_summary)
    legacy_q1 = decision["q1_bio_reliable_relation_teacher"]
    legacy_q3 = decision["q3_hard_states_show_larger_relation_deficits"]
    # The current contract allows INCONCLUSIVE/MIXED rather than the legacy
    # PARTIAL label; retain the original values separately for provenance.
    decision["legacy_q1_label"] = legacy_q1
    decision["legacy_q3_label"] = legacy_q3
    if legacy_q1 == "PARTIAL":
        decision["q1_bio_reliable_relation_teacher"] = "INCONCLUSIVE"
    if legacy_q3 == "PARTIAL":
        decision["q3_hard_states_show_larger_relation_deficits"] = "MIXED"

    source_runs = json.loads(P4R_MANIFEST.read_text(encoding="utf-8"))["runs"]
    source_models = {}
    for name, path in MODEL_BANKS.items():
        if name == "ID-only":
            source_models[name] = {
                "checkpoint": ID_CHECKPOINT,
                "checkpoint_sha256": sha256_file(ID_CHECKPOINT),
                "feature_bank": path,
                "feature_bank_sha256": sha256_file(path),
                "global_feature_dim": int(banks[name]["global_feature"].shape[1]),
                "feature_bank_status": "Phase6A no_grad forward extraction",
                "export_feature_bank": OUT_ROOT / "feature_banks/id_only_feature_bank.npz",
            }
        else:
            manifest_key = {"Bio": "bio_control", "Local Full": "original_full", "Official Full": "official_full"}[name]
            source_models[name] = {
                "checkpoint": source_runs[manifest_key]["checkpoint"],
                "checkpoint_sha256": source_runs[manifest_key]["checkpoint_sha256"],
                "feature_bank": path,
                "feature_bank_sha256": sha256_file(path),
                "global_feature_dim": int(banks[name]["global_feature"].shape[1]),
                "feature_bank_status": "reused frozen Phase4R-C inference bank",
            }
    source_manifest = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "dataset": {"split": "LTCC train", "images": IMAGE_COUNT, "pids": PID_COUNT, "clothing_states": STATE_COUNT, "query_gallery_test_used": False},
        "normalization": "L2 normalization of global_feature for all cosine relations and train-only retrieval",
        "relation": {"temperature": RELATION_TEMPERATURE, "memory_size": RELATION_MEMORY_SIZE, "memory_seed": SEED, "positive_definition": "same PID and different clothes", "negative_definition": "nearest different-PID image"},
        "models": source_models,
        "hardness_source": {"path": P5D_BANK, "sha256": sha256_file(P5D_BANK)},
        "phase5e_pair_utility_source": {"path": P5E_QUADRANTS, "sha256": sha256_file(P5E_QUADRANTS), "role": "frozen auxiliary target only; no new gradient computation"},
    }

    # The new Phase 6A gate is deliberately stricter than the legacy diagnostic
    # summary: a relation target is not promoted unless the Bio relation itself
    # passes the hard-subset and reliability checks.
    relation_target_clear = bool(
        decision["q1_bio_reliable_relation_teacher"] == "YES"
        and decision["gates"]["B_bio_high_hard_identity_margin_positive"]
    )
    decision["q4_high_reliability_relations_more_stable"] = required_relation["reliability_q4"]
    decision["relation_target_structure_clear"] = relation_target_clear
    decision["phase6a_gates"] = {
        "A_bio_cross_clothes_relation_beats_failed_candidates": bool(decision["gates"]["A_bio_cross_clothes_relation_beats_id_and_local_full"]),
        "B_bio_high_hard_identity_margin_is_reliable": bool(decision["gates"]["B_bio_high_hard_identity_margin_positive"]),
        "C_failed_candidate_relation_damage_is_visible": bool(decision["gates"]["C_local_full_relation_damage_visible"]),
        "D_high_reliability_teacher_relations_are_more_stable": required_relation["reliability_q4"] == "YES",
        "E_at_least_one_relation_target_is_clear": relation_target_clear,
    }
    decision["phase6a_gate_count"] = int(sum(decision["phase6a_gates"].values()))
    decision["phase6a_training_decision"] = "PROMOTE TO PHASE 6B" if decision["phase6a_gate_count"] == len(decision["phase6a_gates"]) else "STOP — RELATION TRAINING NOT JUSTIFIED"
    decision["legacy_diagnostic_finding"] = "FAILED FULL DAMAGES BIO RELATION STRUCTURE" if decision["q2_failed_full_breaks_bio_relations"] == "YES" else "NO CLEAR FAILED-FULL RELATION DAMAGE"

    write_json(OUT_ROOT / "diagnosis.json", decision)
    write_json(OUT_ROOT / "feature_banks/source_manifest.json", source_manifest)
    write_csv(OUT_ROOT / "official_comparison/model_summary.csv", summary_rows)
    write_json(OUT_ROOT / "official_comparison/comparison.json", {"models": summary_rows, "relation_kl": relation_kl_summary, "official_is_auxiliary_reference": True})
    write_phase6a_supporting_docs(source_manifest, decision, required_relation)

    required_artifacts = [
        OUT_ROOT / "feature_banks/source_manifest.json",
        OUT_ROOT / "cross_clothes_relations/model_summary.csv",
        OUT_ROOT / "cross_clothes_relations/per_pid_summary.csv",
        OUT_ROOT / "identity_margin/anchor_identity_margin.csv",
        OUT_ROOT / "identity_margin/negative_topk_summary.csv",
        OUT_ROOT / "relation_kl/per_anchor.csv",
        OUT_ROOT / "relation_kl/phase5e_pair_utility_alignment.csv",
        OUT_ROOT / "relation_kl/summary.csv",
        OUT_ROOT / "hardness_stratification/model_by_h_id_quartile.csv",
        OUT_ROOT / "hardness_stratification/model_by_h_cloth_quartile.csv",
        OUT_ROOT / "relation_overlap/topk_overlap.csv",
        OUT_ROOT / "official_comparison/model_summary.csv",
        OUT_ROOT / "teacher_relation_summary.csv",
        OUT_ROOT / "hardness_stratified_relation.csv",
        OUT_ROOT / "same_clothes_positive.csv",
        OUT_ROOT / "cross_clothes_positive.csv",
        OUT_ROOT / "nearest_negative.csv",
        OUT_ROOT / "triplet_relation.csv",
        OUT_ROOT / "relation_divergence.csv",
        OUT_ROOT / "neighbor_purity.csv",
        OUT_ROOT / "teacher_relation_reliability.csv",
        OUT_ROOT / "reliability_threshold_audit.csv",
        OUT_ROOT / "hardness_reliability_quadrants.csv",
        OUT_ROOT / "artifacts/model_source_audit.md",
        OUT_ROOT / "artifacts/future_relation_loss_design.md",
        OUT_ROOT / "artifacts/future_relation_ds_design.md",
        OUT_ROOT / "model_source_audit.md",
        OUT_ROOT / "future_relation_loss_design.md",
        OUT_ROOT / "future_relation_ds_design.md",
    ]
    manifest = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "scope": "offline train-only frozen Bio relation-teacher feasibility",
        "prohibitions": {"training": False, "optimizer_constructed": False, "optimizer_step_count": 0, "backward": False, "parameter_update": False, "ds": False, "dw": False, "sampling": False, "pair_mining": False, "grl_modification": False, "prcc": False, "query_used": False, "gallery_used": False, "test_used": False},
        "dataset": {"split": "LTCC train", "images": IMAGE_COUNT, "pids": PID_COUNT, "clothing_states": STATE_COUNT},
        "models": source_models,
        "relation": source_manifest["relation"],
        "feature_alignment": "all four banks and Phase5D hardness matched by stems/pid/camera/clothes_id",
        "artifacts": [str(path) for path in required_artifacts],
        "decision": decision,
        "reliability": {
            "candidate_names": ["R_pos", "R_neg", "R_margin", "R_equal_diagnostic"],
            "h_id_quartile_edges": required_relation["h_id_edges"],
            "q4_decision": required_relation["reliability_q4"],
            "threshold_audit": str(OUT_ROOT / "reliability_threshold_audit.csv"),
        },
        "git": {"status": subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout},
    }
    write_json(OUT_ROOT / "PHASE6A_MANIFEST.json", manifest)
    write_report(source_manifest, summary_rows, all_hardness_metrics, all_hardness_deficits, relation_kl_summary, overlap_rows, disagreement_summary, decision)
    write_phase6a_required_report(source_manifest, summary_rows, required_relation["same_model_summary"], all_hardness_metrics, all_hardness_deficits, relation_kl_summary, disagreement_summary, overlap_rows, required_relation, decision)
    return {"decision": decision, "summary": summary_rows, "source_manifest": source_manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
    (OUT_ROOT / "git_diff.patch").write_bytes(diff)
    (OUT_ROOT / "git_status.txt").write_text(subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout, encoding="utf-8")
    extraction = extract_id_only()
    write_json(OUT_ROOT / "feature_banks/id_only_extraction.json", extraction)
    exported_id_bank = OUT_ROOT / "feature_banks/id_only_feature_bank.npz"
    if not exported_id_bank.exists() and ID_BANK.is_file():
        exported_id_bank.hardlink_to(ID_BANK)
    if args.stage == "extract":
        print(json.dumps(json_safe(extraction), indent=2, sort_keys=True), flush=True)
        return
    result = analyze()
    print(json.dumps({"phase": PHASE, "report": str(REPORT_PATH), "q1": result["decision"]["q1_bio_reliable_relation_teacher"], "q2": result["decision"]["q2_failed_full_breaks_bio_relations"], "q3": result["decision"]["q3_hard_states_show_larger_relation_deficits"], "q4": result["decision"]["q4_high_reliability_relations_more_stable"], "q5": result["decision"]["phase6a_training_decision"], "final_diagnosis": result["decision"]["final_diagnosis"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
