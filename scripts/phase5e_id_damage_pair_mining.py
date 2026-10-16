#!/usr/bin/env python3
"""Phase 5E: frozen ID-Damage guided pair-mining feasibility audit.

This phase is deliberately offline.  It reuses the Phase-5A/5D frozen LTCC
train feature bank and the Bio Control checkpoint, constructs one comparable
positive and negative relation per anchor, and probes the gradient induced by
the selected triplet relation.  It does not construct an optimizer, change a
parameter, alter sample frequency, alter a loss coefficient, run training, or
access query/gallery data.

Primary question:
    Can ID-Damage select better positive/negative metric relations while
    avoiding the large, poorly aligned gradients seen under oversampling?

The pair-selection statistics cover every training PID with at least two
clothing states.  Gradient probes use the same deterministic anchor set for
all strategies (all PIDs with at least three clothing states), so strategy
comparisons are paired.  Triplet-only gradients have zero gradient through
the Bio projection by construction; that zero is recorded rather than
silently replaced by a core-loss gradient.
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import importlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Phase 5D is an audited, side-effect-free module when imported.  Reusing its
# input/model/gradient helpers keeps the H_ID and Bio Control protocol fixed.
P5D = importlib.import_module("phase5d_id_damage_training_utility")

PHASE = "phase5e_id_damage_pair_mining"
SEED = 1234
EXPECTED_IMAGES = 9576
EXPECTED_PIDS = 77
EXPECTED_STATES = 256
UTILITY_MIN_STATES = 3
BOOTSTRAPS = 2000

P5D_BANK = P5D.OUT_ROOT / "feature_bank" / "feature_bank.npz"
P5D_BANK_CSV = P5D.OUT_ROOT / "feature_bank" / "feature_bank.csv"
TEXT_BANK = P5D.TEXT_BANK
BIO_CHECKPOINT = P5D.BIO_CHECKPOINT

OUT_ROOT = ROOT / "outputs/new_method_track/phase5e_id_damage_pair_mining"
REPORT_PATH = ROOT / "reports/PHASE_5E_ID_DAMAGE_GUIDED_PAIR_MINING.md"


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


def finite(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def safe_mean(values: Any) -> float | None:
    values = finite(values)
    return float(np.mean(values)) if len(values) else None


def safe_median(values: Any) -> float | None:
    values = finite(values)
    return float(np.median(values)) if len(values) else None


def safe_corr(x: Any, y: Any, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    try:
        if method == "pearson":
            value = pearsonr(x, y).statistic
        elif method == "spearman":
            value = spearmanr(x, y).statistic
        elif method == "kendall":
            value = kendalltau(x, y).statistic
        else:
            raise ValueError(method)
        return float(value) if math.isfinite(float(value)) else None
    except Exception:
        return None


def normalize_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def load_frozen_data() -> dict[str, Any]:
    required = [P5D_BANK, P5D_BANK_CSV, TEXT_BANK, BIO_CHECKPOINT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 5E frozen inputs missing: " + ", ".join(missing))

    data = P5D.load_inputs()
    bank = P5D.load_npz(P5D_BANK)
    bank_stems = np.asarray(bank["stems"]).astype(str)
    if not np.array_equal(bank_stems, data["stems"]):
        raise RuntimeError("Phase 5D feature bank is not aligned to Phase 5A record order")
    if bank["global_feature"].shape != (EXPECTED_IMAGES, 1024):
        raise RuntimeError(f"unexpected frozen global feature shape: {bank['global_feature'].shape}")
    for key in ["global_feature", "bio_projected", "id_logits", "summary_text", "ce_loss", "triplet_loss", "bio_loss", "total_core_loss", "correct", "confidence", "true_confidence", "logit_margin"]:
        if key not in bank:
            raise RuntimeError(f"Phase 5D bank missing required key: {key}")
        data[key] = np.asarray(bank[key], dtype=np.float32 if key not in {"correct"} else np.int8)
    # The primary H_ID remains the exact Phase-5A CSV value.  The compressed
    # bank value is only checked as provenance, never substituted in ranking.
    bank_h_id = np.asarray(bank["h_id"], dtype=np.float64)
    if not np.allclose(bank_h_id, data["h_id"], rtol=2e-5, atol=2e-6, equal_nan=True):
        raise RuntimeError("Phase 5D bank H_ID is not consistent with exact Phase-5A H_ID")

    data["camera_raw"] = np.asarray([P5D.parse_camera(stem) for stem in data["stems"]], dtype=np.int64)
    if any(not str(path).startswith(str(P5D.TRAIN_ROOT) + "/") for path in data["paths"]):
        raise RuntimeError("non-training path found in Phase 5E input")
    if any("/query/" in str(path) or "/gallery/" in str(path) or "/test/" in str(path) for path in data["paths"]):
        raise RuntimeError("query/gallery/test path found in Phase 5E input")

    text = P5D.load_npz(TEXT_BANK)
    # Phase 5A fixed text factor index 2 is the Clothing semantic signal.
    data["clothing_feature"] = normalize_np(np.asarray(text["runtime_features"][:, 2, :], dtype=np.float32))
    data["global_feature_normed"] = normalize_np(data["global_feature"])
    return data


def build_state_tables(data: dict[str, Any]) -> dict[str, Any]:
    state_indices: dict[tuple[int, int], np.ndarray] = {}
    global_centers: dict[tuple[int, int], np.ndarray] = {}
    clothing_centers: dict[tuple[int, int], np.ndarray] = {}
    state_h_id: dict[tuple[int, int], float] = {}
    state_h_cloth: dict[tuple[int, int], float] = {}
    representative: dict[tuple[int, int], int] = {}
    pid_indices: dict[int, np.ndarray] = {}
    pid_states: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for key in sorted(data["group_by_key"]):
        pid, cloth = key
        indices = np.flatnonzero((data["pids"] == pid) & (data["clothes"] == cloth)).astype(np.int64)
        if not len(indices):
            continue
        state_indices[key] = indices
        pid_states[pid].append(key)
        gcenter = normalize_np(np.mean(data["global_feature_normed"][indices], axis=0, keepdims=True))[0]
        ccenter = normalize_np(np.mean(data["clothing_feature"][indices], axis=0, keepdims=True))[0]
        global_centers[key] = gcenter
        clothing_centers[key] = ccenter
        state_h_id[key] = float(np.mean(data["h_id"][indices]))
        state_h_cloth[key] = safe_mean(data["h_cloth"][indices]) if np.isfinite(data["h_cloth"][indices]).any() else float("nan")
        # One representative image per state keeps pair strategy comparisons
        # about state selection rather than arbitrary image-count exposure.
        distances = 1.0 - data["global_feature_normed"][indices] @ gcenter
        representative[key] = int(indices[int(np.argmin(distances))])

    for pid in sorted(set(int(x) for x in data["pids"])):
        pid_indices[pid] = np.flatnonzero(data["pids"] == pid).astype(np.int64)
        pid_states[pid] = sorted(pid_states[pid])

    if len(state_indices) != EXPECTED_STATES or len(pid_indices) != EXPECTED_PIDS:
        raise RuntimeError(f"unexpected state/PID table: {len(state_indices)} states, {len(pid_indices)} PIDs")

    pid_centers = {}
    for pid, indices in pid_indices.items():
        pid_centers[pid] = normalize_np(np.mean(data["global_feature_normed"][indices], axis=0, keepdims=True))[0]

    return {
        "state_indices": state_indices,
        "global_centers": global_centers,
        "clothing_centers": clothing_centers,
        "state_h_id": state_h_id,
        "state_h_cloth": state_h_cloth,
        "representative": representative,
        "pid_indices": pid_indices,
        "pid_states": dict(pid_states),
        "pid_centers": pid_centers,
    }


def build_anchors(data: dict[str, Any], tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pid in sorted(tables["pid_states"]):
        states = tables["pid_states"][pid]
        if len(states) < 2:
            continue
        rng = np.random.default_rng(SEED + 100003 * int(pid))
        state = states[int(rng.integers(0, len(states)))]
        index = tables["representative"][state]
        rows.append({
            "pid": int(pid), "anchor_index": int(index), "anchor_stem": data["stems"][index],
            "anchor_path": data["paths"][index], "anchor_cloth_id": int(state[1]),
            "anchor_cloth_key": data["group_by_key"][state]["cloth_key"],
            "state_count": len(states), "utility_eligible": int(len(states) >= UTILITY_MIN_STATES),
        })
    return rows


def candidate_positive_rows(anchor: dict[str, Any], data: dict[str, Any], tables: dict[str, Any]) -> list[dict[str, Any]]:
    pid = int(anchor["pid"])
    anchor_key = (pid, int(anchor["anchor_cloth_id"]))
    anchor_index = int(anchor["anchor_index"])
    anchor_cloth_center = tables["clothing_centers"][anchor_key]
    anchor_global = data["global_feature_normed"][anchor_index]
    rows = []
    for key in tables["pid_states"][pid]:
        if key == anchor_key:
            continue
        rep = tables["representative"][key]
        bio_distance = float(1.0 - np.dot(anchor_global, tables["global_centers"][key]))
        cloth_distance = float(1.0 - np.dot(anchor_cloth_center, tables["clothing_centers"][key]))
        rows.append({
            "pid": pid, "anchor_index": anchor_index, "anchor_cloth_id": anchor_key[1],
            "candidate_cloth_id": key[1], "candidate_cloth_key": data["group_by_key"][key]["cloth_key"],
            "positive_index": rep, "positive_stem": data["stems"][rep], "positive_path": data["paths"][rep],
            "bio_state_distance": bio_distance, "clothing_state_distance": cloth_distance,
            "candidate_h_id": tables["state_h_id"][key], "candidate_h_cloth": tables["state_h_cloth"][key],
            "candidate_support": len(tables["state_indices"][key]),
        })
    return rows


def choose_positive_rows(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_anchor: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_anchor[(int(row["pid"]), int(row["anchor_index"]))].append(row)
    strategies = ["P_easy", "P_random", "P_cloth_hard", "P_IDdamage_hard"]
    selected = []
    for anchor in anchors:
        key = (int(anchor["pid"]), int(anchor["anchor_index"]))
        values = sorted(by_anchor[key], key=lambda row: int(row["candidate_cloth_id"]))
        if not values:
            continue
        rng = np.random.default_rng(SEED + 200003 * int(anchor["pid"]) + 17)
        choices = {
            "P_easy": min(values, key=lambda row: (float(row["bio_state_distance"]), int(row["candidate_cloth_id"]))),
            "P_random": values[int(rng.integers(0, len(values)))],
            "P_cloth_hard": max(values, key=lambda row: (float(row["clothing_state_distance"]), -int(row["candidate_cloth_id"]))),
            "P_IDdamage_hard": max(values, key=lambda row: (float(row["candidate_h_id"]), -int(row["candidate_cloth_id"]))),
        }
        for strategy in strategies:
            row = dict(choices[strategy])
            row["positive_strategy"] = strategy
            selected.append(row)
    return selected, candidates


def choose_negative_rows(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_features = data["global_feature_normed"]
    all_pids = data["pids"]
    unique_pids = sorted(set(int(x) for x in all_pids))
    selected = []
    for anchor in anchors:
        pid = int(anchor["pid"])
        ai = int(anchor["anchor_index"])
        anchor_feature = all_features[ai]
        similarities = all_features @ anchor_feature
        different = np.flatnonzero(all_pids != pid)
        feature_hard_index = int(different[int(np.argmax(similarities[different]))])
        qpid = int(data["nearest_negative_pid"][ai])
        qindices = tables["pid_indices"][qpid]
        confusion_index = int(qindices[int(np.argmax(similarities[qindices]))])
        rng = np.random.default_rng(SEED + 300007 * pid + 31)
        other_pids = [value for value in unique_pids if value != pid]
        random_pid = int(other_pids[int(rng.integers(0, len(other_pids)))])
        random_indices = tables["pid_indices"][random_pid]
        random_index = int(random_indices[int(rng.integers(0, len(random_indices)))])
        choices = {
            "N_random": (random_index, random_pid),
            "N_feature_hard": (feature_hard_index, int(all_pids[feature_hard_index])),
            "N_ID_confusion": (confusion_index, qpid),
        }
        for strategy, (index, selected_pid) in choices.items():
            center_distance = float(1.0 - np.dot(anchor_feature, tables["pid_centers"][selected_pid]))
            selected.append({
                "pid": pid, "anchor_index": ai, "negative_strategy": strategy,
                "negative_index": index, "negative_pid": selected_pid, "negative_stem": data["stems"][index],
                "negative_path": data["paths"][index], "negative_center_distance": center_distance,
                "anchor_nearest_negative_pid": int(data["nearest_negative_pid"][ai]),
                "anchor_negative_distance": float(data["negative_distance"][ai]),
                "anchor_negative_proximity": float(data["negative_proximity"][ai]),
                "anchor_identity_margin": float(data["safe_margin"][ai]),
            })
    return selected


def row_for_positive(selected: list[dict[str, Any]], pid: int, anchor_index: int, strategy: str) -> dict[str, Any]:
    return next(row for row in selected if int(row["pid"]) == pid and int(row["anchor_index"]) == anchor_index and row["positive_strategy"] == strategy)


def row_for_negative(selected: list[dict[str, Any]], pid: int, anchor_index: int, strategy: str) -> dict[str, Any]:
    return next(row for row in selected if int(row["pid"]) == pid and int(row["anchor_index"]) == anchor_index and row["negative_strategy"] == strategy)


def pair_scalars(data: dict[str, Any], anchor: int, positive: int, negative: int) -> dict[str, float | int]:
    gf = data["global_feature"]
    gn = data["global_feature_normed"]
    d_ap = float(np.linalg.norm(gf[anchor] - gf[positive]))
    d_an = float(np.linalg.norm(gf[anchor] - gf[negative]))
    cos_ap = float(np.dot(gn[anchor], gn[positive]))
    cos_an = float(np.dot(gn[anchor], gn[negative]))
    margin = d_ap - d_an
    return {
        "positive_euclidean_distance": d_ap,
        "negative_euclidean_distance": d_an,
        "positive_cosine_similarity": cos_ap,
        "positive_cosine_distance": 1.0 - cos_ap,
        "negative_cosine_similarity": cos_an,
        "negative_cosine_distance": 1.0 - cos_an,
        "triplet_margin": margin,
        "triplet_loss": float(np.logaddexp(0.0, margin)),
        "margin_violation": int(margin > 0.0),
    }


def build_pair_metrics(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]], positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    positive_strategies = ["P_easy", "P_random", "P_cloth_hard", "P_IDdamage_hard"]
    negative_strategies = ["N_random", "N_feature_hard", "N_ID_confusion"]
    for anchor in anchors:
        pid = int(anchor["pid"])
        ai = int(anchor["anchor_index"])
        for ps, ns in itertools.product(positive_strategies, negative_strategies):
            p = row_for_positive(positives, pid, ai, ps)
            n = row_for_negative(negatives, pid, ai, ns)
            scalars = pair_scalars(data, ai, int(p["positive_index"]), int(n["negative_index"]))
            positive_key = (pid, int(p["candidate_cloth_id"]))
            negative_pid = int(n["negative_pid"])
            rows.append({
                "pid": pid, "anchor_index": ai, "anchor_stem": data["stems"][ai], "anchor_path": data["paths"][ai],
                "anchor_cloth_id": int(anchor["anchor_cloth_id"]), "positive_strategy": ps, "negative_strategy": ns,
                "positive_index": int(p["positive_index"]), "positive_stem": p["positive_stem"], "positive_path": p["positive_path"],
                "positive_cloth_id": int(p["candidate_cloth_id"]), "positive_cloth_key": p["candidate_cloth_key"],
                "negative_index": int(n["negative_index"]), "negative_stem": n["negative_stem"], "negative_path": n["negative_path"],
                "negative_pid": negative_pid, "positive_candidate_h_id": float(p["candidate_h_id"]),
                "positive_candidate_h_cloth": p["candidate_h_cloth"], "positive_bio_state_distance": float(p["bio_state_distance"]),
                "positive_clothing_state_distance": float(p["clothing_state_distance"]), "negative_center_distance": float(n["negative_center_distance"]),
                "anchor_nearest_negative_pid": int(n["anchor_nearest_negative_pid"]), "anchor_negative_proximity": float(n["anchor_negative_proximity"]),
                "anchor_identity_margin": float(n["anchor_identity_margin"]), **scalars,
            })
    return rows


def save_selection_artifacts(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]], positives: list[dict[str, Any]], candidates: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> None:
    out = OUT_ROOT / "pair_selection"
    write_csv(out / "anchors.csv", anchors)
    write_csv(out / "positive_candidates.csv", candidates)
    write_csv(out / "positive_selections.csv", positives)
    write_csv(out / "negative_selections.csv", negatives)
    write_json(out / "selection_definition.json", {
        "positive": {
            "P_easy": "same PID, different clothing state with minimum anchor-global-feature to candidate-state-center cosine distance",
            "P_random": "uniform random different clothing state within the same PID; deterministic seed 1234",
            "P_cloth_hard": "same PID, different clothing state with maximum fixed-text Clothing state-center cosine distance",
            "P_IDdamage_hard": "same PID, different clothing state with maximum frozen Phase-5A state-mean H_ID",
        },
        "negative": {
            "N_random": "uniform random different PID followed by a random image within that PID; deterministic seed 1234",
            "N_feature_hard": "different PID image with maximum current Bio global-feature cosine similarity",
            "N_ID_confusion": "Phase-5A nearest competing PID, then its image with maximum anchor Bio-feature similarity; does not use D_own for selection",
        },
        "positive_frequency": "one selected positive relation per strategy and anchor; no sample frequency or loss coefficient is changed",
        "anchor_set": "all train PIDs with at least two clothing states for difficulty; PIDs with at least three states for matched gradient probes",
        "clothing_source": "fixed text runtime_features[:,2,:] from Phase-5A aligned text feature bank; 512-D, L2-normalized",
    })


def make_model_and_cache(data: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5E gradient probes require the available PPU/CUDA device")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg, transform = P5D.make_model_config_and_transform()
    model, load_info = P5D.load_model(BIO_CHECKPOINT, cfg, device)
    groups = P5D.parameter_groups(model)
    cache: dict[int, torch.Tensor] = {}
    return model, groups, transform, device, cache, load_info


def cached_image(index: int, data: dict[str, Any], transform: Any, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    if index not in cache:
        cache[index] = transform(P5D.read_image(str(data["paths"][index]))).float()
    return cache[index]


def triplet_gradient(model, groups, transform, device, cache, data: dict[str, Any], anchor: int, positive: int, negative: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    for values in groups.values():
        for _, parameter in values:
            parameter.requires_grad_(True)
    indices = [int(anchor), int(positive), int(negative)]
    images = torch.stack([cached_image(index, data, transform, cache) for index in indices], dim=0).to(device=device, dtype=torch.float32, non_blocking=False)
    cameras = torch.as_tensor(data["camera_raw"][indices] - 1, dtype=torch.long, device=device)
    features = model.forward_features(images, cameras)
    if isinstance(features, (list, tuple)):
        features = features[0]
    global_feature = model.forward_head(features, pre_logits=True)
    d_ap = (global_feature[0] - global_feature[1]).pow(2).sum().clamp_min(1e-12).sqrt()
    d_an = (global_feature[0] - global_feature[2]).pow(2).sum().clamp_min(1e-12).sqrt()
    loss = F.softplus(d_ap - d_an)
    loss.backward()
    vectors = {name: P5D.vector_from_grads(values) for name, values in groups.items()}
    scalars = {"triplet_loss": float(loss.detach().float().cpu()), "positive_euclidean_distance": float(d_ap.detach().float().cpu()), "negative_euclidean_distance": float(d_an.detach().float().cpu())}
    del images, cameras, features, global_feature, d_ap, d_an, loss
    for parameter in model.parameters():
        parameter.grad = None
    return vectors, scalars


def triplet_reference_gradient(model, groups, transform, device, cache, data: dict[str, Any], indices: list[int]) -> dict[str, torch.Tensor]:
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    for values in groups.values():
        for _, parameter in values:
            parameter.requires_grad_(True)
    images = torch.stack([cached_image(int(index), data, transform, cache) for index in indices], dim=0).to(device=device, dtype=torch.float32, non_blocking=False)
    cameras = torch.as_tensor(data["camera_raw"][indices] - 1, dtype=torch.long, device=device)
    features = model.forward_features(images, cameras)
    if isinstance(features, (list, tuple)):
        features = features[0]
    global_feature = model.forward_head(features, pre_logits=True)
    squared = (global_feature * global_feature).sum(dim=1, keepdim=True)
    distances = (squared + squared.T - 2.0 * global_feature @ global_feature.T).clamp_min(1e-12).sqrt()
    pids = torch.as_tensor(data["pids"][indices], dtype=torch.long, device=device)
    positive_mask = pids[:, None].eq(pids[None, :])
    negative_mask = ~positive_mask
    d_ap = distances.masked_fill(~positive_mask, -float("inf")).max(dim=1).values
    d_an = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    loss = F.softplus(d_ap - d_an).mean()
    loss.backward()
    vectors = {name: P5D.vector_from_grads(values) for name, values in groups.items()}
    del images, cameras, features, global_feature, squared, distances, pids, positive_mask, negative_mask, d_ap, d_an, loss
    for parameter in model.parameters():
        parameter.grad = None
    return vectors


def run_gradient_pair_probes(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    utility_anchors = [row for row in anchors if int(row["utility_eligible"]) == 1]
    utility_pids = {int(row["pid"]) for row in utility_anchors}
    selected = [row for row in pair_rows if int(row["pid"]) in utility_pids]
    model, groups, transform, device, cache, load_info = make_model_and_cache(data)
    reference_targets = []
    for pid in range(4):
        candidates = np.flatnonzero(data["pids"] == pid)
        first = int(candidates[0])
        reference_targets.extend([first, P5D.different_cloth_partner(first, data, True)])
    reference_vectors, reference_scalars = P5D.gradient_for_batch(model, groups, reference_targets, data, transform, device)
    triplet_ref = triplet_reference_gradient(model, groups, transform, device, cache, data, reference_targets)
    output = []
    total = len(selected)
    for count, row in enumerate(selected, start=1):
        vectors, scalars = triplet_gradient(model, groups, transform, device, cache, data, int(row["anchor_index"]), int(row["positive_index"]), int(row["negative_index"]))
        result = dict(row)
        result.update(scalars)
        for name in ["late_eva_block", "bio_projection", "id_head"]:
            vector = vectors[name]
            result[f"gradient_norm_{name}"] = float(vector.norm())
            result[f"gradient_alignment_{name}"] = P5D.cosine_cpu(vector, reference_vectors[name])
            result[f"triplet_gradient_alignment_{name}"] = P5D.cosine_cpu(vector, triplet_ref[name])
        result["gradient_alignment_late_eva_block_positive"] = int(float(result["gradient_alignment_late_eva_block"]) > 0.0)
        output.append(result)
        if count == 1 or count % 24 == 0 or count == total:
            print(f"PAIR_GRADIENT {count}/{total}", flush=True)
    out = OUT_ROOT / "gradient_probes"
    write_csv(out / "pair_gradient_probes.csv", output)
    write_json(out / "pair_gradient_manifest.json", {
        "phase": PHASE, "inference_only": True, "optimizer_constructed": False, "optimizer_step_count": 0,
        "parameter_update": False, "checkpoint": load_info, "device": torch.cuda.get_device_name(0),
        "dtype": "FP32", "gradient_scaling": "none; no GradScaler and no AMP autocast",
        "gradient_objective": "one pair-induced softplus(d(anchor,positive)-d(anchor,negative)) Triplet relation; no CE/Bio term",
        "reference_objective": "Phase5D fixed balanced full-data CE + Triplet + Summary Bio gradient",
        "triplet_reference_objective": "same fixed balanced batch, released soft batch-hard Triplet only",
        "utility_anchor_count": len(utility_anchors), "pair_probe_count": len(output),
        "bio_projection_gradient_note": "Triplet relation has no computational path through head_clip_bio; its gradient norm is expected to be exactly zero and is reported as such.",
        "groups": {name: {"parameter_count": len(values), "parameter_names": [key for key, _ in values], "numel": int(sum(param.numel() for _, param in values))} for name, values in groups.items()},
        "reference_batch_indices": reference_targets, "reference_scalars": reference_scalars,
    })
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del model, reference_vectors, triplet_ref, cache
    gc.collect()
    torch.cuda.empty_cache()
    return {"rows": output, "utility_anchors": utility_anchors, "checkpoint": load_info, "reference_targets": reference_targets}


def load_existing_gradient_pair_probes() -> dict[str, Any] | None:
    """Resume report generation from a completed probe file after a safe failure.

    The probe file is immutable diagnostic output.  Reusing it avoids another
    expensive backward-only pass when a later CSV/statistics step needs a
    repair.  Numeric columns are restored because CSV readers return strings.
    """
    probe_path = OUT_ROOT / "gradient_probes" / "pair_gradient_probes.csv"
    manifest_path = OUT_ROOT / "gradient_probes" / "pair_gradient_manifest.json"
    if not probe_path.is_file() or not manifest_path.is_file():
        return None
    rows = read_csv(probe_path)
    # The completed first pass uses all 32 LTCC PIDs with >=3 clothing
    # states, crossed with 4 positive and 3 negative strategies.
    if len(rows) != 384:
        return None
    numeric = {
        "pid", "anchor_index", "positive_index", "positive_cloth_id", "negative_index", "negative_pid",
        "positive_candidate_h_id", "positive_candidate_h_cloth", "positive_bio_state_distance", "positive_clothing_state_distance",
        "negative_center_distance", "anchor_nearest_negative_pid", "anchor_negative_proximity", "anchor_identity_margin",
        "positive_euclidean_distance", "negative_euclidean_distance", "positive_cosine_similarity", "positive_cosine_distance",
        "negative_cosine_similarity", "negative_cosine_distance", "triplet_margin", "triplet_loss", "margin_violation",
        "gradient_norm_late_eva_block", "gradient_alignment_late_eva_block", "triplet_gradient_alignment_late_eva_block",
        "gradient_norm_bio_projection", "gradient_alignment_bio_projection", "triplet_gradient_alignment_bio_projection",
        "gradient_norm_id_head", "gradient_alignment_id_head", "triplet_gradient_alignment_id_head",
        "gradient_alignment_late_eva_block_positive",
    }
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = float(row[key]) if key not in {"pid", "anchor_index", "positive_index", "positive_cloth_id", "negative_index", "negative_pid", "anchor_nearest_negative_pid", "margin_violation", "gradient_alignment_late_eva_block_positive"} else int(float(row[key]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"REUSE_PAIR_GRADIENT_ARTIFACT rows={len(rows)}", flush=True)
    return {"rows": rows, "utility_anchors": [], "checkpoint": manifest.get("checkpoint", {}), "reference_targets": manifest.get("reference_batch_indices", [])}


def summarize_pair_difficulty(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for ps, ns in itertools.product(["P_easy", "P_random", "P_cloth_hard", "P_IDdamage_hard"], ["N_random", "N_feature_hard", "N_ID_confusion"]):
        selected = [row for row in pair_rows if row["positive_strategy"] == ps and row["negative_strategy"] == ns]
        output.append({
            "positive_strategy": ps, "negative_strategy": ns, "count": len(selected),
            "positive_euclidean_distance_mean": safe_mean([row["positive_euclidean_distance"] for row in selected]),
            "positive_cosine_distance_mean": safe_mean([row["positive_cosine_distance"] for row in selected]),
            "negative_euclidean_distance_mean": safe_mean([row["negative_euclidean_distance"] for row in selected]),
            "negative_cosine_distance_mean": safe_mean([row["negative_cosine_distance"] for row in selected]),
            "triplet_margin_mean": safe_mean([row["triplet_margin"] for row in selected]),
            "triplet_loss_mean": safe_mean([row["triplet_loss"] for row in selected]),
            "margin_violation_rate": safe_mean([row["margin_violation"] for row in selected]),
            "positive_candidate_h_id_mean": safe_mean([row["positive_candidate_h_id"] for row in selected]),
            "positive_candidate_h_cloth_mean": safe_mean([row["positive_candidate_h_cloth"] for row in selected]),
            "positive_bio_state_distance_mean": safe_mean([row["positive_bio_state_distance"] for row in selected]),
            "positive_clothing_state_distance_mean": safe_mean([row["positive_clothing_state_distance"] for row in selected]),
            "negative_center_distance_mean": safe_mean([row["negative_center_distance"] for row in selected]),
        })
    return output


def summarize_pair_utility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for ps, ns in itertools.product(["P_easy", "P_random", "P_cloth_hard", "P_IDdamage_hard"], ["N_random", "N_feature_hard", "N_ID_confusion"]):
        selected = [row for row in rows if row["positive_strategy"] == ps and row["negative_strategy"] == ns]
        output.append({
            "positive_strategy": ps, "negative_strategy": ns, "count": len(selected),
            "triplet_loss_mean": safe_mean([row["triplet_loss"] for row in selected]),
            "triplet_margin_mean": safe_mean([row["triplet_margin"] for row in selected]),
            "margin_violation_rate": safe_mean([row["margin_violation"] for row in selected]),
            "gradient_norm_late_eva_block_mean": safe_mean([row["gradient_norm_late_eva_block"] for row in selected]),
            "gradient_alignment_late_eva_block_mean": safe_mean([row["gradient_alignment_late_eva_block"] for row in selected]),
            "gradient_alignment_late_eva_block_positive_rate": safe_mean([row["gradient_alignment_late_eva_block_positive"] for row in selected]),
            "triplet_gradient_alignment_late_eva_block_mean": safe_mean([row["triplet_gradient_alignment_late_eva_block"] for row in selected]),
            "gradient_norm_bio_projection_mean": safe_mean([row["gradient_norm_bio_projection"] for row in selected]),
            "gradient_norm_id_head_mean": safe_mean([row["gradient_norm_id_head"] for row in selected]),
        })
    return output


def bootstrap_paired(rows: list[dict[str, Any]], group_a: str, group_b: str, varying: str, fixed: str, metric: str, seed_offset: int) -> dict[str, Any]:
    # ``fixed`` is the fixed strategy value (for example N_feature_hard),
    # while the field carrying it is the complementary strategy column.
    fixed_field = "negative_strategy" if varying == "positive_strategy" else "positive_strategy"
    a = {int(row["pid"]): float(row[metric]) for row in rows if row[varying] == group_a and row[fixed_field] == fixed}
    b = {int(row["pid"]): float(row[metric]) for row in rows if row[varying] == group_b and row[fixed_field] == fixed}
    pids = sorted(set(a) & set(b))
    if not pids:
        return {"group_a": group_a, "group_b": group_b, "varying": varying, "fixed": fixed, "metric": metric, "n_pid": 0, "observed_delta_a_minus_b": None, "ci95": [None, None]}
    deltas = np.asarray([a[pid] - b[pid] for pid in pids], dtype=np.float64)
    rng = np.random.default_rng(SEED + seed_offset)
    boot = np.asarray([float(np.mean(rng.choice(deltas, size=len(deltas), replace=True))) for _ in range(BOOTSTRAPS)], dtype=np.float64)
    return {
        "group_a": group_a, "group_b": group_b, "varying": varying, "fixed": fixed, "metric": metric,
        "n_pid": len(pids), "observed_delta_a_minus_b": float(np.mean(deltas)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def add_quadrants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hardness = np.asarray([float(row["triplet_loss"]) for row in rows], dtype=np.float64)
    utility = np.asarray([float(row["gradient_alignment_late_eva_block"]) for row in rows], dtype=np.float64)
    hardness_threshold = float(np.median(hardness))
    utility_threshold = 0.0
    output = []
    for row in rows:
        high_hard = float(row["triplet_loss"]) >= hardness_threshold
        high_utility = float(row["gradient_alignment_late_eva_block"]) > utility_threshold
        category = ("HIGH_HARD" if high_hard else "LOW_HARD") + ("_HIGH_UTILITY" if high_utility else "_LOW_UTILITY")
        output.append({**row, "pair_hardness": float(row["triplet_loss"]), "pair_utility": float(row["gradient_alignment_late_eva_block"]), "hardness_threshold": hardness_threshold, "utility_threshold": utility_threshold, "utility_quadrant": category})
    counts = defaultdict(int)
    for row in output:
        counts[row["utility_quadrant"]] += 1
    write_csv(OUT_ROOT / "gradient_probes" / "pair_utility_quadrants.csv", output)
    write_json(OUT_ROOT / "gradient_probes" / "pair_utility_quadrants.json", {
        "hardness_definition": "pair-induced softplus triplet loss; high means >= median over matched gradient probes",
        "utility_definition": "late-EVA gradient cosine alignment to the frozen Phase5D balanced CE+Triplet+Bio reference; high means > 0",
        "thresholds": {"hardness": hardness_threshold, "utility": utility_threshold}, "counts": dict(counts),
    })
    return {"rows": output, "counts": dict(counts), "hardness_threshold": hardness_threshold, "utility_threshold": utility_threshold}


def correlation_audit(pair_rows: list[dict[str, Any]], utility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signals = ["positive_candidate_h_id", "positive_candidate_h_cloth", "positive_bio_state_distance", "positive_clothing_state_distance", "triplet_loss", "negative_center_distance"]
    targets = ["positive_euclidean_distance", "triplet_loss", "margin_violation"]
    rows = []
    for signal, target in itertools.product(signals, targets):
        rows.append({"signal": signal, "target": target, "spearman": safe_corr([row[signal] for row in pair_rows], [row[target] for row in pair_rows], "spearman"), "pearson": safe_corr([row[signal] for row in pair_rows], [row[target] for row in pair_rows], "pearson"), "n": len(pair_rows)})
    utility_signals = ["positive_candidate_h_id", "positive_candidate_h_cloth", "positive_bio_state_distance", "positive_clothing_state_distance", "triplet_loss"]
    for signal, target in itertools.product(utility_signals, ["gradient_alignment_late_eva_block", "gradient_norm_late_eva_block"]):
        rows.append({"signal": signal, "target": target, "spearman": safe_corr([row[signal] for row in utility_rows], [row[target] for row in utility_rows], "spearman"), "pearson": safe_corr([row[signal] for row in utility_rows], [row[target] for row in utility_rows], "pearson"), "n": len(utility_rows)})
    write_csv(OUT_ROOT / "analysis" / "pair_signal_correlations.csv", rows)
    return {"rows": rows}


def make_decision(pair_rows: list[dict[str, Any]], utility_rows: list[dict[str, Any]], utility_quadrants: dict[str, Any], bootstrap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_delta(group_a: str, group_b: str, varying: str, fixed: str, metric: str):
        values = []
        pids = sorted(set(int(row["pid"]) for row in utility_rows))
        fixed_field = "negative_strategy" if varying == "positive_strategy" else "positive_strategy"
        for pid in pids:
            a = next((row for row in utility_rows if int(row["pid"]) == pid and row[varying] == group_a and row[fixed_field] == fixed), None)
            b = next((row for row in utility_rows if int(row["pid"]) == pid and row[varying] == group_b and row[fixed_field] == fixed), None)
            if a and b:
                values.append(float(a[metric]) - float(b[metric]))
        return safe_mean(values)

    id_hardness_delta = metric_delta("P_IDdamage_hard", "P_random", "positive_strategy", "N_feature_hard", "triplet_loss")
    id_utility_delta = metric_delta("P_IDdamage_hard", "P_random", "positive_strategy", "N_feature_hard", "gradient_alignment_late_eva_block")
    id_positive_rate = safe_mean([row["gradient_alignment_late_eva_block_positive"] for row in utility_rows if row["positive_strategy"] == "P_IDdamage_hard" and row["negative_strategy"] == "N_feature_hard"])
    random_positive_rate = safe_mean([row["gradient_alignment_late_eva_block_positive"] for row in utility_rows if row["positive_strategy"] == "P_random" and row["negative_strategy"] == "N_feature_hard"])
    qcounts = utility_quadrants["counts"]
    high_hard_low = int(qcounts.get("HIGH_HARD_LOW_UTILITY", 0))
    high_hard_high = int(qcounts.get("HIGH_HARD_HIGH_UTILITY", 0))
    id_boot_hard = next((row for row in bootstrap_rows if row["group_a"] == "P_IDdamage_hard" and row["group_b"] == "P_random" and row["metric"] == "triplet_loss"), None)
    id_boot_utility = next((row for row in bootstrap_rows if row["group_a"] == "P_IDdamage_hard" and row["group_b"] == "P_random" and row["metric"] == "gradient_alignment_late_eva_block"), None)
    if id_boot_hard and id_boot_hard["ci95"][0] is not None and id_boot_hard["ci95"][0] > 0 and id_boot_utility and id_boot_utility["ci95"][0] is not None and id_boot_utility["ci95"][0] >= 0:
        feasibility = "SUPPORTED"
        q1 = "YES"
    elif id_hardness_delta is not None and id_hardness_delta > 0 and high_hard_high > 0:
        feasibility = "PARTIAL"
        q1 = "MIXED"
    else:
        feasibility = "INCONCLUSIVE"
        q1 = "INCONCLUSIVE"
    if high_hard_low > high_hard_high:
        utility_conclusion = "High-hardness pair relations are mixed; hardness alone does not justify direct pair upweighting."
    else:
        utility_conclusion = "The matched pair probes do not show a majority high-hard/low-utility relation."
    return {
        "q1_can_id_damage_find_better_pairs_without_upweighting": q1,
        "feasibility": feasibility,
        "id_damage_positive_triplet_delta_vs_random": id_hardness_delta,
        "id_damage_positive_gradient_alignment_delta_vs_random": id_utility_delta,
        "id_damage_positive_gradient_positive_rate": id_positive_rate,
        "random_positive_gradient_positive_rate": random_positive_rate,
        "high_hard_high_utility_count": high_hard_high,
        "high_hard_low_utility_count": high_hard_low,
        "utility_conclusion": utility_conclusion,
        "recommendation": "Use ID-Damage, if at all, as a relation/positive candidate-mining signal only; do not upweight the selected images or change sampling frequency. Negative selection should remain separate and should be evaluated with N_ID_confusion/negative proximity.",
    }


def write_report(data: dict[str, Any], tables: dict[str, Any], anchors: list[dict[str, Any]], positives: list[dict[str, Any]], negatives: list[dict[str, Any]], pair_rows: list[dict[str, Any]], difficulty_summary: list[dict[str, Any]], utility: dict[str, Any], utility_summary: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], correlations: dict[str, Any], decision: dict[str, Any], model_info: dict[str, Any]) -> None:
    def fmt(value, digits=4):
        if value is None:
            return "—"
        try:
            value = float(value)
            return "—" if not math.isfinite(value) else f"{value:.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "# PHASE 5E — ID-Damage Guided Pair Mining Feasibility", "",
        "## Scope and stop condition", "",
        "This phase is frozen train-only pair analysis. It performs no long training, no optimizer construction or step, no DW, no DS, no sampling reweighting, no loss weighting, no Full DIFFER run, no GRL, and no query/gallery/test retrieval. Pair mining changes only the selected metric relation in this diagnostic; all source images remain one-time offline records and every loss coefficient is one.", "",
        f"LTCC train: `{EXPECTED_IMAGES}` images, `{EXPECTED_PIDS}` PIDs, `{EXPECTED_STATES}` clothing states. Difficulty pairs use `{len(anchors)}` anchors with at least two clothing states; matched gradient probes use `{len([row for row in anchors if int(row['utility_eligible']) == 1])}` PIDs with at least three clothing states.", "",
        "## FROZEN REPRESENTATION SOURCES", "",
        f"- Bio Control global feature: `{P5D_BANK}`; source checkpoint `{BIO_CHECKPOINT}`; checkpoint SHA256 `{sha256_file(BIO_CHECKPOINT)}`; global feature 1024-D, L2-normalized only for cosine selection.",
        f"- Fixed Clothing semantic feature: `{TEXT_BANK}` factor `runtime_features[:,2,:]`; bank SHA256 `{sha256_file(TEXT_BANK)}`; 512-D, L2-normalized. This is the Phase-5A fixed text Clothing signal, not a trained reverse branch.",
        "- H_ID: exact Phase-5A `feature_records.csv` values, `D_own * D_negative`; no formula, threshold, or sign convention was changed. Positive ID-Damage selection uses candidate clothing-state mean H_ID only.",
        "- Bio Control gradient reference: Phase-5D fixed balanced 4-PID × 2-image CE + Triplet + Summary Bio reference. Pair utility is late-EVA gradient cosine alignment to this reference. A supplementary Triplet-only reference is also saved.", "",
        "## PAIR SELECTION DEFINITIONS", "",
        "Positive candidates are same PID and different clothing state. `P_easy` minimizes anchor-to-candidate-state Bio distance; `P_random` samples a different state uniformly; `P_cloth_hard` maximizes fixed-text Clothing state distance; `P_IDdamage_hard` maximizes candidate state-mean H_ID. One representative image per state is used so state selection is compared without image-count exposure.", "",
        "Negative candidates are different PID. `N_random` samples a PID uniformly then an image; `N_feature_hard` selects the closest Bio global image; `N_ID_confusion` uses the frozen nearest competing PID and selects its closest image to the anchor. `D_own` is not used to select negatives.", "",
        "## PAIR DIFFICULTY", "",
        "Triplet relation uses the released metric geometry: raw global-feature Euclidean distances and `softplus(d(anchor,positive) - d(anchor,negative))`; margin violation means the positive distance exceeds the negative distance. Complete rows are in `pair_analysis/pair_metrics.csv`.", "",
        "| Positive | Negative | n | Pos cosine dist | Neg cosine dist | Triplet loss | Violation | Pos H_ID | Pos H_cloth |", "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in difficulty_summary:
        lines.append(f"| {row['positive_strategy']} | {row['negative_strategy']} | {row['count']} | {fmt(row['positive_cosine_distance_mean'])} | {fmt(row['negative_cosine_distance_mean'])} | {fmt(row['triplet_loss_mean'])} | {fmt(row['margin_violation_rate'])} | {fmt(row['positive_candidate_h_id_mean'])} | {fmt(row['positive_candidate_h_cloth_mean'])} |")
    lines += ["", "## GRADIENT UTILITY", "", "Each matched probe is one frozen triplet relation. Gradients are unscaled FP32 backward probes; there is no AMP scaler and no optimizer. Because the objective is Triplet-only, Bio-projection gradient is expected to be exactly zero; the report does not treat that structural zero as evidence for or against a Bio update.", "", "| Positive | Negative | n | Triplet loss | Late-EVA norm | Core-ref alignment | Positive alignment rate | Triplet-ref alignment |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in utility_summary:
        lines.append(f"| {row['positive_strategy']} | {row['negative_strategy']} | {row['count']} | {fmt(row['triplet_loss_mean'])} | {fmt(row['gradient_norm_late_eva_block_mean'])} | {fmt(row['gradient_alignment_late_eva_block_mean'])} | {fmt(row['gradient_alignment_late_eva_block_positive_rate'])} | {fmt(row['triplet_gradient_alignment_late_eva_block_mean'])} |")
    lines += ["", f"Pair hardness × utility thresholds: hardness = median matched pair Triplet loss `{fmt(utility['hardness_threshold'])}`; utility = late-EVA alignment `> 0`. Quadrant counts: `{json.dumps(utility['counts'], sort_keys=True)}`. Per-pair rows are in `gradient_probes/pair_utility_quadrants.csv`.", "", "## PAIRED BOOTSTRAP COMPARISONS", "", "Bootstrap is PID-level, paired on the same anchor set, 2000 resamples, and uses training-only relation probes. Positive comparisons fix `N_feature_hard`; negative comparisons fix `P_random`.", "", "| A | B | Varying | Metric | n PID | Δ A−B | 95% CI |", "|---|---|---|---|---:|---:|---:|"]
    for row in bootstrap_rows:
        ci = row["ci95"]
        ci_text = "—" if ci[0] is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(f"| {row['group_a']} | {row['group_b']} | {row['varying']} | {row['metric']} | {row['n_pid']} | {fmt(row['observed_delta_a_minus_b'])} | {ci_text} |")
    lines += ["", "## SIGNAL SEPARATION", "", "Positive hardness and negative confusion are kept separate. The full correlation audit is in `analysis/pair_signal_correlations.csv`; no test retrieval metric is used as a target.", "", "| Signal | Target | Spearman | Pearson | n |", "|---|---|---:|---:|---:|"]
    for row in correlations["rows"]:
        lines.append(f"| {row['signal']} | {row['target']} | {fmt(row['spearman'])} | {fmt(row['pearson'])} | {row['n']} |")
    lines += ["", "## REPRESENTATIVE PAIRS", "", "Selected pair rows retain PID, clothes IDs, image stems/paths, H_ID/H_cloth, distances, Triplet relation values, selected negative PID, and gradient utility where probed. See `pair_analysis/pair_metrics.csv` and `gradient_probes/pair_gradient_probes.csv`. The four hardness/utility categories are directly available in `gradient_probes/pair_utility_quadrants.csv`.", "", "## FINAL ANSWER", "", f"Q1 — Can ID-Damage find better positive/negative pairs without upweighting noisy samples? **{decision['q1_can_id_damage_find_better_pairs_without_upweighting']}**.", f"Feasibility: **{decision['feasibility']}**. P_IDdamage_hard vs P_random Triplet-loss delta (fixed N_feature_hard): `{fmt(decision['id_damage_positive_triplet_delta_vs_random'])}`; corresponding late-EVA alignment delta: `{fmt(decision['id_damage_positive_gradient_alignment_delta_vs_random'])}`. High-hard/high-utility pairs: `{decision['high_hard_high_utility_count']}`; high-hard/low-utility pairs: `{decision['high_hard_low_utility_count']}`.", f"Interpretation: {decision['utility_conclusion']}", f"Recommendation: {decision['recommendation']}", "", "## ARTIFACTS", "", f"- Selection audit: `{OUT_ROOT / 'pair_selection'}`", f"- Pair metrics: `{OUT_ROOT / 'pair_analysis'}`", f"- Gradient probes: `{OUT_ROOT / 'gradient_probes'}`", f"- Correlations: `{OUT_ROOT / 'analysis'}`", f"- Manifest: `{OUT_ROOT / 'PHASE5E_MANIFEST.json'}`", "", "STOP: no subsequent DW, DS, sampling, training, Reliability, Online H_ID, Full DIFFER, GRL, or PRCC was started."]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(data: dict[str, Any], model_info: dict[str, Any], decision: dict[str, Any], loss_bank_info: dict[str, Any], bootstrap_rows: list[dict[str, Any]]) -> None:
    inputs = [P5D_BANK, P5D_BANK_CSV, P5D.P5A_MANIFEST, P5D.P5A_SOURCES, TEXT_BANK, BIO_CHECKPOINT]
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED,
        "scope": "frozen forward/backward train-only pair relation diagnostic",
        "prohibitions": {"optimizer_constructed": False, "optimizer_step_count": 0, "parameter_update": False, "dw": False, "ds": False, "sampling_reweighting": False, "loss_weighting": False, "reliability": False, "online_h_id": False, "full_differ": False, "grl": False, "prcc": False, "test_retrieval": False},
        "dataset": {"split": "LTCC train", "images": EXPECTED_IMAGES, "pids": EXPECTED_PIDS, "clothing_states": EXPECTED_STATES, "query_gallery_used": False},
        "frozen_h_id": "Phase5A feature_records.csv exact values; H_ID = D_own * D_negative; unchanged",
        "inputs": {str(path): {"sha256": sha256_file(path), "size": path.stat().st_size} for path in inputs if path.is_file()},
        "feature_bank": {"path": str(P5D_BANK), "sha256": sha256_file(P5D_BANK), "losses_are_frozen_phase5d_bank": True, "loss_bank_info": loss_bank_info},
        "gradient_probe": model_info, "bootstrap": {"resamples": BOOTSTRAPS, "unit": "PID", "rows": bootstrap_rows},
        "decision": decision,
        "clothing_signal": {"source": str(TEXT_BANK), "factor_index": 2, "dimension": 512, "normalization": "L2"},
        "pair_relation": {"sample_frequency_changed": False, "loss_coefficients_changed": False, "triplet_only_gradient": True},
        "git": {"status": subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout},
    }
    write_json(OUT_ROOT / "PHASE5E_MANIFEST.json", manifest)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
    (OUT_ROOT / "git_diff.patch").write_bytes(diff)
    (OUT_ROOT / "git_status.txt").write_text(subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout, encoding="utf-8")

    print("LOAD_FROZEN_PHASE5D_BANK", flush=True)
    data = load_frozen_data()
    tables = build_state_tables(data)
    anchors = build_anchors(data, tables)
    candidates = []
    for anchor in anchors:
        candidates.extend(candidate_positive_rows(anchor, data, tables))
    positives, candidates = choose_positive_rows(data, tables, anchors, candidates)
    negatives = choose_negative_rows(data, tables, anchors)
    save_selection_artifacts(data, tables, anchors, positives, candidates, negatives)
    pair_rows = build_pair_metrics(data, tables, anchors, positives, negatives)
    write_csv(OUT_ROOT / "pair_analysis" / "pair_metrics.csv", pair_rows)
    difficulty_summary = summarize_pair_difficulty(pair_rows)
    write_csv(OUT_ROOT / "pair_analysis" / "pair_summary.csv", difficulty_summary)
    print(f"PAIR_SELECTION_COMPLETE anchors={len(anchors)} candidates={len(candidates)} pairs={len(pair_rows)}", flush=True)

    gradient = load_existing_gradient_pair_probes()
    if gradient is None:
        gradient = run_gradient_pair_probes(data, tables, anchors, pair_rows)
    utility_rows = gradient["rows"]
    utility_summary = summarize_pair_utility(utility_rows)
    write_csv(OUT_ROOT / "gradient_probes" / "pair_gradient_summary.csv", utility_summary)
    utility = add_quadrants(utility_rows)
    bootstrap_rows = []
    for offset, (a, b, varying, fixed, metric) in enumerate([
        ("P_IDdamage_hard", "P_random", "positive_strategy", "N_feature_hard", "triplet_loss"),
        ("P_IDdamage_hard", "P_random", "positive_strategy", "N_feature_hard", "gradient_alignment_late_eva_block"),
        ("P_cloth_hard", "P_random", "positive_strategy", "N_feature_hard", "triplet_loss"),
        ("P_cloth_hard", "P_random", "positive_strategy", "N_feature_hard", "gradient_alignment_late_eva_block"),
        ("N_ID_confusion", "N_random", "negative_strategy", "P_random", "triplet_loss"),
        ("N_ID_confusion", "N_random", "negative_strategy", "P_random", "gradient_alignment_late_eva_block"),
        ("N_feature_hard", "N_random", "negative_strategy", "P_random", "triplet_loss"),
        ("N_feature_hard", "N_random", "negative_strategy", "P_random", "gradient_alignment_late_eva_block"),
    ]):
        bootstrap_rows.append(bootstrap_paired(utility_rows, a, b, varying, fixed, metric, offset + 1))
    write_csv(OUT_ROOT / "analysis" / "paired_bootstrap.csv", bootstrap_rows)
    correlations = correlation_audit(pair_rows, utility_rows)
    decision = make_decision(pair_rows, utility_rows, utility, bootstrap_rows)
    loss_bank_info = {"source": str(P5D_BANK), "feature_bank_csv": str(P5D_BANK_CSV), "images": EXPECTED_IMAGES, "all_train_only": True}
    write_manifest(data, gradient["checkpoint"], decision, loss_bank_info, bootstrap_rows)
    write_report(data, tables, anchors, positives, negatives, pair_rows, difficulty_summary, utility, utility_summary, bootstrap_rows, correlations, decision, gradient["checkpoint"])
    write_json(OUT_ROOT / "diagnosis.json", decision)
    print(json.dumps({"phase": PHASE, "report": str(REPORT_PATH), "artifact_root": str(OUT_ROOT), "q1": decision["q1_can_id_damage_find_better_pairs_without_upweighting"], "feasibility": decision["feasibility"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
