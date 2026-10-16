#!/usr/local/bin/python3
"""PHASE 7H-A: frozen identity-subspace Relation audit.

Diagnostic only.  This script loads the frozen LTCC B0 and PRCC C0 Bio
checkpoints, extracts only representations already present in the executed
DIFFER graph, and computes train-only geometry and no-step autograd probes.
It never creates an optimizer/scheduler/GradScaler and never calls backward()
or optimizer.step().
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_loader import ImageDataset  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase7e_relation_triplet_complementarity as p7e  # noqa: E402


PHASE = "phase7h_a_identity_subspace_relation_audit"
OUT_ROOT = ROOT / "outputs/new_method_track/phase7h_a_identity_subspace_relation_audit"
INVENTORY_ROOT = OUT_ROOT / "feature_inventory"
BANK_ROOT = OUT_ROOT / "feature_banks"
GEOM_ROOT = OUT_ROOT / "geometry"
CLOTH_ROOT = OUT_ROOT / "clothing_sensitivity"
ID_ROOT = OUT_ROOT / "identity_discrimination"
GRAD_ROOT = OUT_ROOT / "gradient_alignment"
TRANS_ROOT = OUT_ROOT / "pair_transition"
CROSS_ROOT = OUT_ROOT / "cross_dataset"
ONE_ROOT = OUT_ROOT / "optional_one_step"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7H_A_MANIFEST.json"
SOURCE_JSON = INVENTORY_ROOT / "source_and_feature_audit.json"
SOURCE_REPORT = ROOT / "reports/PHASE_7H_A_SOURCE_AND_FEATURE_AUDIT.md"
REPORT_PATH = ROOT / "reports/PHASE_7H_A_IDENTITY_SUBSPACE_RELATION_AUDIT.md"

P7F_ROOT = ROOT / "outputs/new_method_track/phase7f_a_relation_headroom_audit"
LTCC_PAIR_CACHE = P7F_ROOT / "pair_headroom/ltcc_pair_headroom.npz"
PRCC_PAIR_CACHE = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/pair_margin_analysis/pair_metrics.npz"

SOURCES: dict[str, dict[str, Any]] = {
    "LTCC": {
        "dataset": "ltcc",
        "arm": "B0 Bio baseline",
        "phase": "Phase 6C",
        "checkpoint": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/model_final.pth",
        "config": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/config.yml",
        "run_manifest": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/run_manifest.json",
        "pair_cache": LTCC_PAIR_CACHE,
    },
    "PRCC": {
        "dataset": "prcc",
        "arm": "C0 Bio baseline",
        "phase": "Phase 7C",
        "checkpoint": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/model_final.pth",
        "config": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/config.yml",
        "run_manifest": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/run_manifest.json",
        "pair_cache": PRCC_PAIR_CACHE,
    },
}

FEATURE_SPACES = ("SPACE-FULL", "SPACE-BIO-PROJ")
GRADIENT_PROBES = 6
PAIR_SAMPLE = 100_000
EPS = 1e-12


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ensure_dirs() -> None:
    for path in (OUT_ROOT, INVENTORY_ROOT, BANK_ROOT, GEOM_ROOT, CLOTH_ROOT, ID_ROOT,
                 GRAD_ROOT, TRANS_ROOT, CROSS_ROOT, ONE_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    required = [ROOT / "model/eva_meta_bio.py", ROOT / "model/build.py", ROOT / "loss/make_loss.py",
                ROOT / "loss/triplet_loss.py", ROOT / "scripts/phase7e_relation_triplet_complementarity.py"]
    for spec in SOURCES.values():
        required += [spec["checkpoint"], spec["config"], spec["run_manifest"], spec["pair_cache"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing Phase 7H-A source artifacts: {missing}")

    arms = {}
    for name, spec in SOURCES.items():
        cfg = load_yaml_config(spec["config"])
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        keys = sorted(str(key).replace("module.", "") for key in state.keys())
        state_key_set = set(keys)
        arms[name] = {
            "dataset": spec["dataset"], "arm": spec["arm"], "phase": spec["phase"],
            "checkpoint": file_record(spec["checkpoint"]), "config": file_record(spec["config"]),
            "run_manifest": file_record(spec["run_manifest"]), "pair_cache": file_record(spec["pair_cache"]),
            "config_model": cfg.get("MODEL", {}), "config_data": cfg.get("DATA", {}),
            "checkpoint_has_head_bio": any(key.startswith("head_bio.") for key in state_key_set),
            "checkpoint_has_bio_projection": any(key.startswith("head_clip_bio.") for key in state_key_set),
            "checkpoint_has_reverse_head": any("head_clip_bioReverse" in key for key in state_key_set),
            "checkpoint_tensor_key_count": len(keys),
        }

    code_paths = {
        "eva_model": ROOT / "model/eva_meta_bio.py",
        "model_builder": ROOT / "model/build.py",
        "loss": ROOT / "loss/make_loss.py",
        "triplet": ROOT / "loss/triplet_loss.py",
        "gradient_reference": ROOT / "scripts/phase7e_relation_triplet_complementarity.py",
    }
    code = {name: file_record(path) for name, path in code_paths.items()}
    inventory = {
        "SPACE-FULL": {
            "tensor": "feat[0] / Eva.forward() feat_bio",
            "source_module": "model/eva_meta_bio.py: Eva.forward -> feat_bio; loss/make_loss.py uses feat[0]",
            "dimension": 1024,
            "normalization": "L2 normalization applied for cosine Relation/evaluation; raw feature is preserved in bank",
            "checkpoint_dependency": "frozen B0/C0 checkpoint",
            "trainable_in_original": True,
            "direct_test_feature": True,
            "relation_gradient_path": "EVA shared blocks + head_bio; not head_clip_bio",
            "orientation": "executed identity/ReID feature",
            "formal_relation_space": True,
        },
        "SPACE-BIO-PROJ": {
            "tensor": "score['clip_bio_score'] = head_clip_bio(feat_bio)",
            "source_module": "model/eva_meta_bio.py: head_clip_bio; loss/make_loss.py clipBio branch",
            "dimension_by_dataset": {name: int(item["config_model"].get("CLIP_DIM", -1)) for name, item in arms.items()},
            "normalization": "L2 normalization is applied in the frozen clipBio loss and again for this cosine diagnostic",
            "checkpoint_dependency": "frozen B0/C0 checkpoint",
            "trainable_in_original": True,
            "direct_test_feature": False,
            "relation_gradient_path": "head_clip_bio + EVA shared blocks + head_bio",
            "orientation": "existing Bio semantic identity projection",
            "formal_relation_space": True,
        },
        "SPACE-BIO-RAW": {
            "status": "NOT_DISTINCT",
            "reason": "The runtime raw feat_bio is the same tensor exposed as feat[0] and is therefore SPACE-FULL, not a second space.",
        },
        "SPACE-CLOTHING": {
            "status": "NOT_AVAILABLE",
            "reason": "The exact frozen B0/C0 configs do not enable clipBioReverse; no valid frozen clothing branch exists for this matched comparison.",
        },
        "SPACE-HAIR": {
            "status": "NOT_AVAILABLE",
            "reason": "The exact frozen B0/C0 configs do not enable clipBioReverse; no valid frozen hair branch exists for this matched comparison.",
        },
    }
    # Static source checks prevent accidentally treating a missing projection as
    # an invented feature.  Both matched checkpoints must contain the modules.
    if not all(item["checkpoint_has_head_bio"] and item["checkpoint_has_bio_projection"] for item in arms.values()):
        raise RuntimeError("fewer than two verified feature spaces: head_bio or head_clip_bio is absent")
    audit = {
        "phase": PHASE, "created_utc": utc_now(), "source_verification_pass": True,
        "source_arms": arms, "authoritative_code": code, "feature_inventory": inventory,
        "train_only": True, "query_gallery_used": False, "optimizer_created": False,
        "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False,
        "new_feature_or_projection_learned": False,
        "clothing_gradient_reference": "NOT_AVAILABLE: clipBioReverse is absent in matched B0/C0 configs.",
    }
    write_json(SOURCE_JSON, audit)

    lines = [
        "# Phase 7H-A source and feature audit", "", f"Generated: `{audit['created_utc']}`.",
        "", "Decision: **PASS**. Two genuinely distinct, pre-existing identity-relevant spaces are verified: `feat[0]` and `head_clip_bio(feat_bio)`.",
        "", "## Exact frozen arms", "", "| Dataset | Arm | Checkpoint | SHA256 | Config | Pair cache |", "|---|---|---|---|---|---|",
    ]
    for name, item in arms.items():
        lines.append(f"| {name} | {item['arm']} | `{item['checkpoint']['path']}` | `{item['checkpoint']['sha256']}` | `{item['config']['path']}` | `{item['pair_cache']['path']}` |")
    lines += ["", "## Verified feature inventory", "", "| Space | Tensor | Dim | Train/test role | Relation gradient path |", "|---|---|---:|---|---|"]
    lines += [
        "| SPACE-FULL | `feat[0]` / `Eva.forward().feat_bio` | 1024 | Executed identity feature and current Relation space | shared EVA blocks + `head_bio` |",
        "| SPACE-BIO-PROJ | `head_clip_bio(feat_bio)` | LTCC 512; PRCC 768 | Existing `clipBio` semantic identity projection; not current test feature | projection + shared EVA blocks + `head_bio` |",
    ]
    lines += ["", "`SPACE-BIO-RAW` is not distinct from `SPACE-FULL`. Matched B0/C0 have no enabled `clipBioReverse` branch, so Clothing/Hair gradient references are **NOT AVAILABLE**. No feature mask, PCA, learned projection, or new network was introduced.", "", "## Authoritative code hashes", ""]
    lines += [f"- `{name}`: `{item['sha256']}`" for name, item in code.items()]
    lines += ["", "All extraction and geometry use the train split only; query/gallery/test records are excluded.", ""]
    SOURCE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return audit


def load_context(name: str, device: torch.device):
    cfg, dataset, transform = p7e.cfg_dataset(SOURCES[name]["dataset"])
    model = p7e.model_from_checkpoint(SOURCES[name]["dataset"], cfg, dataset, SOURCES[name]["checkpoint"], device)
    return cfg, dataset, transform, model


def extract_one(name: str, device: torch.device) -> dict[str, Any]:
    path = BANK_ROOT / f"{name.lower()}_feature_bank.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return {"path": str(path), "sha256": sha256_file(path), "shapes": {key: list(archive[key].shape) for key in archive.files}}
    cfg, dataset, transform, model = load_context(name, device)
    model.eval()
    model.requires_grad_(False)
    batch_size = int(cfg.DATA.BATCH_SIZE)
    loader = DataLoader(ImageDataset(dataset.train, transform=transform), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False, drop_last=False)
    full_parts: list[np.ndarray] = []
    bio_parts: list[np.ndarray] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=False)
            cams = batch["camid"].to(device, non_blocking=False)
            full = model(images, cam_label=cams)
            bio = model.head_clip_bio(full)
            full_parts.append(full.float().cpu().numpy())
            bio_parts.append(bio.float().cpu().numpy())
            if index % 50 == 0:
                print(f"[phase7h-a] extract {name} batch={index} images={len(images)}", flush=True)
    full_raw = np.concatenate(full_parts, axis=0).astype(np.float32)
    bio_raw = np.concatenate(bio_parts, axis=0).astype(np.float32)
    paths = np.asarray([str(item["image_path"]) for item in dataset.train])
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int32)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int32)
    cameras = np.asarray([int(item["camid"]) for item in dataset.train], dtype=np.int32)
    if len(full_raw) != len(dataset.train) or len(bio_raw) != len(dataset.train):
        raise RuntimeError(f"feature extraction count mismatch for {name}")
    np.savez_compressed(path, full_raw=full_raw, bio_proj_raw=bio_raw, paths=paths, pids=pids, clothes=clothes, cameras=cameras)
    meta = {"dataset": name, "count": len(paths), "full_shape": list(full_raw.shape), "bio_proj_shape": list(bio_raw.shape),
            "path": str(path), "sha256": sha256_file(path), "device": str(device), "autocast_enabled": False,
            "optimizer_step_called": False, "training_started": False}
    write_json(BANK_ROOT / f"{name.lower()}_feature_bank.json", meta)
    del model, dataset, full_parts, bio_parts, full_raw, bio_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"path": str(path), "sha256": meta["sha256"], "shapes": {"full_raw": meta["full_shape"], "bio_proj_raw": meta["bio_proj_shape"]}}


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_bank(name: str) -> dict[str, Any]:
    path = BANK_ROOT / f"{name.lower()}_feature_bank.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def build_pairs_from_metadata(pids: np.ndarray, clothes: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (pid, cloth) in enumerate(zip(pids.tolist(), clothes.tolist())):
        groups[int(pid)][int(cloth)].append(index)
    left: list[int] = []; right: list[int] = []; pair_pid: list[int] = []; cloth_a: list[int] = []; cloth_b: list[int] = []
    for pid in sorted(groups):
        states = groups[pid]
        labels = sorted(states)
        for pos, ca in enumerate(labels[:-1]):
            for cb in labels[pos + 1:]:
                for a in states[ca]:
                    for b in states[cb]:
                        left.append(a); right.append(b); pair_pid.append(pid); cloth_a.append(ca); cloth_b.append(cb)
    return {"a": np.asarray(left, dtype=np.int32), "b": np.asarray(right, dtype=np.int32),
            "pid": np.asarray(pair_pid, dtype=np.int32), "cloth_a": np.asarray(cloth_a, dtype=np.int32),
            "cloth_b": np.asarray(cloth_b, dtype=np.int32), "pids": pids, "clothes": clothes}


def load_pairs(name: str, bank: dict[str, Any]) -> dict[str, np.ndarray]:
    pids = bank["pids"].astype(np.int32); clothes = bank["clothes"].astype(np.int32)
    if name == "PRCC":
        with np.load(PRCC_PAIR_CACHE, allow_pickle=False) as archive:
            pairs = {key: np.asarray(archive[key]) for key in archive.files}
        a = pairs["pair_a"].astype(np.int32); b = pairs["pair_b"].astype(np.int32)
        if np.any(pids[a] != pairs["pid"]) or np.any(pids[b] != pairs["pid"]) or np.any(clothes[a] == clothes[b]):
            raise RuntimeError("PRCC frozen pair cache does not match current train metadata")
        result = {"a": a, "b": b, "pid": pairs["pid"].astype(np.int32), "cloth_a": pairs["clothes_a"].astype(np.int32),
                  "cloth_b": pairs["clothes_b"].astype(np.int32), "pids": pids, "clothes": clothes}
    else:
        result = build_pairs_from_metadata(pids, clothes)
        with np.load(LTCC_PAIR_CACHE, allow_pickle=False) as cache:
            if len(cache["pid"]) != len(result["pid"]) or not np.array_equal(cache["pid"], result["pid"]):
                raise RuntimeError("LTCC generated canonical pair order does not match frozen Phase 7F pair cache")
    write_json(TABLE_ROOT / f"{name.lower()}_pair_set_audit.json", {"dataset": name, "pair_count": int(len(result["a"])),
             "pid_count": int(len(np.unique(pids))), "changing_clothes_pid_count": int(sum(len(np.unique(clothes[pids == pid])) >= 2 for pid in np.unique(pids))),
             "train_only": True, "query_gallery_used": False})
    return result


def exact_group_pair_mean(z: np.ndarray, groups: list[tuple[int, np.ndarray]]) -> tuple[float, int]:
    total = 0.0; count = 0
    for _key, indices in groups:
        n = len(indices)
        if n < 2:
            continue
        s = z[indices].sum(axis=0)
        pair_sum = (float(np.dot(s, s)) - n) / 2.0
        pair_count = n * (n - 1) // 2
        total += pair_sum; count += pair_count
    return (total / count if count else float("nan"), count)


def group_centers(z: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    unique = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int32)
    centers = []
    lookup: dict[int, int] = {}
    for index, label in enumerate(unique.tolist()):
        center = z[labels == label].mean(axis=0)
        centers.append(center / max(float(np.linalg.norm(center)), 1e-12))
        lookup[int(label)] = index
    return np.asarray(centers, dtype=np.float32), unique, lookup


def random_different_pairs(pids: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left: list[int] = []; right: list[int] = []
    n = len(pids)
    while len(left) < count:
        size = max(10000, (count - len(left)) * 2)
        a = rng.integers(0, n, size=size, dtype=np.int32)
        b = rng.integers(0, n, size=size, dtype=np.int32)
        mask = pids[a] != pids[b]
        left.extend(a[mask].tolist()); right.extend(b[mask].tolist())
    return np.asarray(left[:count], dtype=np.int32), np.asarray(right[:count], dtype=np.int32)


def pairwise_values(z: np.ndarray, pairs: dict[str, np.ndarray], centers: np.ndarray, center_lookup: dict[int, int]) -> dict[str, np.ndarray]:
    a = pairs["a"]; b = pairs["b"]
    pos = np.sum(z[a] * z[b], axis=1, dtype=np.float32)
    sims = z @ centers.T
    own = np.asarray([center_lookup[int(pid)] for pid in pairs["pids"]], dtype=np.int32)
    neg = sims.copy()
    neg[np.arange(len(z)), own] = -1e4
    neg_image = neg.max(axis=1)
    neg_pair = np.maximum(neg_image[a], neg_image[b])
    return {"pos": pos, "neg": neg_pair, "margin": pos - neg_pair}


def feature_geometry(name: str, space: str, z: np.ndarray, pairs: dict[str, np.ndarray], seed_offset: int) -> dict[str, Any]:
    pids = pairs["pids"]; clothes = pairs["clothes"]
    centers, unique_pids, lookup = group_centers(z, pids)
    pair = pairwise_values(z, pairs, centers, lookup)
    state_groups = [(key, np.flatnonzero((pids == key[0]) & (clothes == key[1])))
                    for key in sorted(set(zip(pids.tolist(), clothes.tolist())))]
    pid_groups = [(int(pid), np.flatnonzero(pids == pid)) for pid in unique_pids.tolist()]
    same_clothes, same_count = exact_group_pair_mean(z, state_groups)
    within_pid_cos, within_pid_count = exact_group_pair_mean(z, pid_groups)
    within_pid_disp = 1.0 - within_pid_cos if math.isfinite(within_pid_cos) else float("nan")
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    for (pid, cloth), indices in state_groups:
        center = z[indices].mean(axis=0); state_centers[(int(pid), int(cloth))] = center / max(float(np.linalg.norm(center)), 1e-12)
    intra_state_dists: list[float] = []
    for pid in unique_pids.tolist():
        states = [state_centers[key] for key in state_centers if key[0] == int(pid)]
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                intra_state_dists.append(1.0 - float(np.dot(states[i], states[j])))
    between_pid_dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            between_pid_dists.append(1.0 - float(np.dot(centers[i], centers[j])))
    neg_a, neg_b = random_different_pairs(pids, min(PAIR_SAMPLE, max(1, len(pairs["a"]))), 70100 + seed_offset)
    neg_sim = np.sum(z[neg_a] * z[neg_b], axis=1)
    pos_for_auc = pair["pos"]
    # Balance the ROC input to prevent the metric from becoming a pair-count
    # statement; the positive set remains the exact cross-clothes pair set.
    take = min(len(pos_for_auc), len(neg_sim))
    auc_labels = np.concatenate([np.ones(take, dtype=np.int8), np.zeros(take, dtype=np.int8)])
    auc_scores = np.concatenate([pos_for_auc[:take], neg_sim[:take]])
    from sklearn.metrics import roc_auc_score
    same_diff_auc = float(roc_auc_score(auc_labels, auc_scores))
    loo_correct = 0
    pid_sums = {int(pid): z[pids == pid].sum(axis=0) for pid in unique_pids.tolist()}
    pid_counts = {int(pid): int(np.sum(pids == pid)) for pid in unique_pids.tolist()}
    for index, pid in enumerate(pids.tolist()):
        own_raw = pid_sums[int(pid)] - z[index]
        own = own_raw / max(float(np.linalg.norm(own_raw)), 1e-12)
        score_row = z[index] @ centers.T
        score_row[lookup[int(pid)]] = float(np.dot(z[index], own))
        if int(unique_pids[int(np.argmax(score_row))]) == int(pid):
            loo_correct += 1
    result = {
        "dataset": name, "space": space, "dimension": int(z.shape[1]), "image_count": int(len(z)),
        "pid_count": int(len(unique_pids)), "same_clothes_similarity": same_clothes,
        "same_clothes_pair_count": int(same_count), "cross_clothes_similarity": float(np.mean(pair["pos"])),
        "cross_clothes_similarity_std": float(np.std(pair["pos"])), "different_pid_similarity": float(np.mean(neg_sim)),
        "different_pid_similarity_std": float(np.std(neg_sim)), "nearest_negative_pid_center_similarity": float(np.mean(pair["neg"])),
        "nearest_negative_pid_center_std": float(np.std(pair["neg"])), "identity_margin": float(np.mean(pair["margin"])),
        "identity_margin_std": float(np.std(pair["margin"])), "margin_positive_fraction": float(np.mean(pair["margin"] > 0)),
        "cross_clothes_ordering_accuracy": float(np.mean(pair["margin"] > 0)), "same_different_pid_roc_auc": same_diff_auc,
        "nearest_center_accuracy_loo": float(loo_correct / len(z)), "within_pid_image_dispersion": within_pid_disp,
        "within_pid_image_pair_count": int(within_pid_count), "state_center_dispersion": float(np.mean(intra_state_dists)) if intra_state_dists else float("nan"),
        "state_center_pair_count": int(len(intra_state_dists)), "between_pid_center_separation": float(np.mean(between_pid_dists)) if between_pid_dists else float("nan"),
        "between_pid_center_pair_count": int(len(between_pid_dists)), "negative_sample_count": int(len(neg_sim)),
        "_pair_pos": pair["pos"], "_pair_neg": pair["neg"], "_pair_margin": pair["margin"],
        "_centers": centers, "_state_centers": state_centers,
    }
    result["clothing_sensitivity_ratio"] = result["state_center_dispersion"] / max(result["between_pid_center_separation"], 1e-12)
    return result


def geometry_run() -> dict[str, Any]:
    ensure_dirs()
    all_results: dict[str, Any] = {}
    pair_payload: dict[str, Any] = {}
    for offset, name in enumerate(("LTCC", "PRCC")):
        bank = load_bank(name)
        pairs = load_pairs(name, bank)
        z_full = normalize(bank["full_raw"])
        z_bio = normalize(bank["bio_proj_raw"])
        results = {}
        for space, z in (("SPACE-FULL", z_full), ("SPACE-BIO-PROJ", z_bio)):
            results[space] = feature_geometry(name, space, z, pairs, offset * 1000)
        all_results[name] = results
        pair_payload[name] = {"a": pairs["a"], "b": pairs["b"], "pid": pairs["pid"],
                              "full_pos": results["SPACE-FULL"]["_pair_pos"], "full_neg": results["SPACE-FULL"]["_pair_neg"],
                              "full_margin": results["SPACE-FULL"]["_pair_margin"], "bio_pos": results["SPACE-BIO-PROJ"]["_pair_pos"],
                              "bio_neg": results["SPACE-BIO-PROJ"]["_pair_neg"], "bio_margin": results["SPACE-BIO-PROJ"]["_pair_margin"]}
        rows = []
        for space, result in results.items():
            rows.append({key: value for key, value in result.items() if not key.startswith("_")})
        write_csv(GEOM_ROOT / f"{name.lower()}_geometry.csv", rows)
        cloth_rows = [{"dataset": name, "space": space, "clothing_sensitivity_ratio": result["clothing_sensitivity_ratio"],
                       "intra_pid_cross_clothes_state_distance": result["state_center_dispersion"],
                       "between_pid_center_distance": result["between_pid_center_separation"]} for space, result in results.items()]
        write_csv(CLOTH_ROOT / f"{name.lower()}_clothing_sensitivity.csv", cloth_rows)
        id_rows = [{"dataset": name, "space": space, "nearest_center_accuracy_loo": result["nearest_center_accuracy_loo"],
                    "same_different_pid_roc_auc": result["same_different_pid_roc_auc"],
                    "cross_clothes_ordering_accuracy": result["cross_clothes_ordering_accuracy"],
                    "mean_identity_margin": result["identity_margin"], "margin_positive_fraction": result["margin_positive_fraction"]}
                   for space, result in results.items()]
        write_csv(ID_ROOT / f"{name.lower()}_identity_discrimination.csv", id_rows)
    np.savez_compressed(TRANS_ROOT / "matched_pair_space_metrics.npz", **{f"{name.lower()}_{key}": value for name, payload in pair_payload.items() for key, value in payload.items()})
    table_rows = []
    metric_keys = ["cross_clothes_similarity", "nearest_negative_pid_center_similarity", "identity_margin", "margin_positive_fraction",
                   "clothing_sensitivity_ratio", "same_different_pid_roc_auc", "cross_clothes_ordering_accuracy", "state_center_dispersion",
                   "within_pid_image_dispersion", "between_pid_center_separation"]
    for metric in metric_keys:
        row = {"metric": metric}
        for name in ("LTCC", "PRCC"):
            for space in FEATURE_SPACES:
                row[f"{name}_{space.replace('-', '_').lower()}"] = all_results[name][space].get(metric)
        table_rows.append(row)
    write_csv(TABLE_ROOT / "feature_space_master.csv", table_rows)
    write_csv(CROSS_ROOT / "feature_space_cross_dataset.csv", table_rows)
    write_json(GEOM_ROOT / "geometry_summary.json", strip_internal(all_results))
    pair_transition_run(pair_payload)
    plot_geometry(all_results)
    return all_results


def strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_internal(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [strip_internal(item) for item in value]
    return value


def pair_transition_run(pair_payload: dict[str, Any]) -> None:
    rows = []
    for name, data in pair_payload.items():
        full = data["full_margin"]; bio = data["bio_margin"]; pids = data["pid"]
        categories = {
            "A_FULL_nonpositive_BIO_positive": (full <= 0) & (bio > 0),
            "B_FULL_positive_BIO_positive": (full > 0) & (bio > 0),
            "C_FULL_positive_BIO_nonpositive": (full > 0) & (bio <= 0),
            "D_FULL_nonpositive_BIO_nonpositive": (full <= 0) & (bio <= 0),
        }
        changing_pids = np.unique(pids)
        for category, mask in categories.items():
            rows.append({"dataset": name, "category": category, "pair_count": int(mask.sum()), "pair_fraction": float(mask.mean()),
                         "pid_count": int(len(np.unique(pids[mask]))) if mask.any() else 0,
                         "pid_fraction": float(len(np.unique(pids[mask])) / max(1, len(changing_pids)))})
    write_csv(TRANS_ROOT / "full_vs_bio_margin_transitions.csv", rows)


def plot_geometry(results: dict[str, Any]) -> None:
    status = {"created": False, "error": None}
    try:
        import matplotlib.pyplot as plt
        metrics = ["cross_clothes_similarity", "nearest_negative_pid_center_similarity", "identity_margin"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
        x = np.arange(4)
        labels = ["LTCC Full", "LTCC Bio", "PRCC Full", "PRCC Bio"]
        for ax, metric in zip(axes, metrics):
            values = [results["LTCC"]["SPACE-FULL"][metric], results["LTCC"]["SPACE-BIO-PROJ"][metric],
                      results["PRCC"]["SPACE-FULL"][metric], results["PRCC"]["SPACE-BIO-PROJ"][metric]]
            ax.bar(x, values); ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_title(metric); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); fig.savefig(PLOT_ROOT / "feature_space_geometry.png", dpi=150); plt.close(fig); status["created"] = True
    except Exception as exc:  # plots are supplementary only
        status["error"] = str(exc)
    write_json(PLOT_ROOT / "plot_status.json", status)


def group_norm(grads: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float:
    total = 0.0
    for name, grad in zip(names, grads):
        if name in selected and grad is not None:
            total += float(torch.sum(grad.detach().float() ** 2).cpu())
    return math.sqrt(max(total, 0.0))


def group_dot(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float:
    value = 0.0
    for name, a, b in zip(names, left, right):
        if name not in selected:
            continue
        if a is None and b is None:
            continue
        if a is None: a = torch.zeros_like(b)
        if b is None: b = torch.zeros_like(a)
        value += float(torch.sum(a.detach().float() * b.detach().float()).cpu())
    return value


def alignment(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float | None:
    nl = group_norm(left, names, selected); nr = group_norm(right, names, selected)
    return group_dot(left, right, names, selected) / (nl * nr + EPS) if nl > EPS and nr > EPS else None


def novelty_against(relation: tuple[torch.Tensor | None, ...], reference: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float | None:
    nr = group_norm(relation, names, selected); nn = group_norm(reference, names, selected)
    if nr <= EPS or nn <= EPS:
        return None
    scale = group_dot(relation, reference, names, selected) / (nn * nn + EPS)
    orth_sq = 0.0
    for name, rg, ref in zip(names, relation, reference):
        if name not in selected:
            continue
        if rg is None and ref is None:
            continue
        if rg is None: rg = torch.zeros_like(ref)
        if ref is None: ref = torch.zeros_like(rg)
        diff = rg.detach().float() - scale * ref.detach().float()
        orth_sq += float(torch.sum(diff * diff).cpu())
    return math.sqrt(max(orth_sq, 0.0)) / (nr + EPS)


def sum_grads(*grad_sets: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor | None, ...]:
    out = []
    for values in zip(*grad_sets):
        tensors = [value for value in values if value is not None]
        out.append(None if not tensors else sum(tensors))
    return tuple(out)


def parameter_groups(model: torch.nn.Module) -> dict[str, set[str]]:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    blocks = sorted({int(name.split(".")[1]) for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_blocks = set(blocks[-3:])
    shared = {name for name in names if name.startswith(("patch_embed.", "cls_token", "pos_embed", "camera_embed", "blocks.", "head_bio.", "norm_bio", "fc_norm"))}
    late = {name for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late_blocks}
    projection = {name for name in names if name.startswith("head_clip_bio.")}
    return {"shared_backbone": shared, "late_eva": late, "bio_projection": projection}


def relation_loss(rep: torch.Tensor, edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return rep.sum() * 0.0
    z = F.normalize(rep.float(), dim=-1)
    return torch.stack([1.0 - torch.sum(z[i] * z[j]) for i, j in edges]).mean()


def gradient_one(name: str, device: torch.device) -> dict[str, Any]:
    cfg, dataset, transform, model = load_context(name, device)
    model.train(); model.requires_grad_(True)
    loader = p7e.make_loader(SOURCES[name]["dataset"], dataset, cfg, transform, epoch=60)
    bank = p7e.relation_bank(SOURCES[name]["dataset"], dataset, cfg)
    loss_fn, _ = base.make_loss(cfg, num_classes=dataset.num_train_pids, lossTypes=cfg.MODEL.LOSS_TYPE)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]; params = [p for _, p in named]; groups = parameter_groups(model)
    rows: list[dict[str, Any]] = []; inspected = 0; found = 0
    for batch_index, batch in enumerate(loader):
        if found >= GRADIENT_PROBES:
            break
        _all_edges, rel_edges = p7e.local_pairs(batch, bank)
        if not rel_edges:
            inspected += 1
            continue
        images = batch["image"].to(device, non_blocking=False); pids = batch["pid"].to(device, non_blocking=False)
        cams = batch["camid"].to(device, non_blocking=False); clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=cams)
        writer = base.MetricCapture()
        _total = loss_fn(score, feat, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=batch_index)
        components = loss_fn.last_components
        base_loss = components["ce"] + components["triplet"] + components["bio"]
        full_loss = relation_loss(feat[0], rel_edges)
        bio_loss = relation_loss(score["clip_bio_score"], rel_edges)
        base_g = sum_grads(*[torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                             for value in (components["ce"], components["triplet"], components["bio"])])
        tri_g = torch.autograd.grad(components["triplet"], params, retain_graph=True, allow_unused=True)
        full_g = torch.autograd.grad(full_loss, params, retain_graph=True, allow_unused=True)
        bio_g = torch.autograd.grad(bio_loss, params, retain_graph=False, allow_unused=True)
        for space, rel_g, rel_value in (("SPACE-FULL", full_g, full_loss), ("SPACE-BIO-PROJ", bio_g, bio_loss)):
            for group_name, selected in groups.items():
                rows.append({"dataset": name, "space": space, "group": group_name, "batch_index": batch_index,
                             "relation_count": len(rel_edges), "relation_loss": float(rel_value.detach().float().cpu()),
                             "norm_relation": group_norm(rel_g, names, selected),
                             "cos_base_relation": alignment(base_g, rel_g, names, selected),
                             "cos_triplet_relation": alignment(tri_g, rel_g, names, selected),
                             "novelty_vs_base": novelty_against(rel_g, base_g, names, selected),
                             "novelty_vs_triplet": novelty_against(rel_g, tri_g, names, selected),
                             "gradient_dtype": "float32", "autocast_enabled": False,
                             "optimizer_step_called": False, "backward_called": False})
        found += 1; inspected += 1
        del score, feat, base_g, tri_g, full_g, bio_g
        model.zero_grad(set_to_none=True)
    if found < 1:
        raise RuntimeError(f"no relation-active gradient probes found for {name}")
    write_csv(GRAD_ROOT / f"{name.lower()}_gradient_rows.csv", rows)
    summary_rows = []
    for space in FEATURE_SPACES:
        for group in ("shared_backbone", "late_eva", "bio_projection"):
            selected = [row for row in rows if row["space"] == space and row["group"] == group]
            summary_rows.append({"dataset": name, "space": space, "group": group, "probe_count": len(selected),
                                 "norm_relation_mean": float(np.nanmean([row["norm_relation"] for row in selected])),
                                 "cos_base_relation_mean": float(np.nanmean([row["cos_base_relation"] for row in selected if row["cos_base_relation"] is not None])) if any(row["cos_base_relation"] is not None for row in selected) else None,
                                 "cos_triplet_relation_mean": float(np.nanmean([row["cos_triplet_relation"] for row in selected if row["cos_triplet_relation"] is not None])) if any(row["cos_triplet_relation"] is not None for row in selected) else None,
                                 "novelty_vs_base_mean": float(np.nanmean([row["novelty_vs_base"] for row in selected if row["novelty_vs_base"] is not None])) if any(row["novelty_vs_base"] is not None for row in selected) else None,
                                 "novelty_vs_triplet_mean": float(np.nanmean([row["novelty_vs_triplet"] for row in selected if row["novelty_vs_triplet"] is not None])) if any(row["novelty_vs_triplet"] is not None for row in selected) else None})
    write_csv(GRAD_ROOT / f"{name.lower()}_gradient_summary.csv", summary_rows)
    write_json(GRAD_ROOT / f"{name.lower()}_gradient_meta.json", {"dataset": name, "inspected_batches": inspected, "probe_count": found,
             "groups": list(groups), "spaces": list(FEATURE_SPACES), "optimizer_created": False, "optimizer_step_called": False, "backward_called": False})
    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "summary": summary_rows, "inspected_batches": inspected, "probe_count": found}


def gradient_run(device: torch.device) -> dict[str, Any]:
    ensure_dirs()
    result = {name: gradient_one(name, device) for name in ("LTCC", "PRCC")}
    write_json(GRAD_ROOT / "gradient_summary.json", {name: {"summary": value["summary"], "inspected_batches": value["inspected_batches"], "probe_count": value["probe_count"]} for name, value in result.items()})
    cross_rows = []
    for group in ("shared_backbone", "late_eva", "bio_projection"):
        for metric in ("norm_relation_mean", "cos_base_relation_mean", "cos_triplet_relation_mean", "novelty_vs_base_mean", "novelty_vs_triplet_mean"):
            row = {"group": group, "metric": metric}
            for name in ("LTCC", "PRCC"):
                for space in FEATURE_SPACES:
                    item = next(item for item in result[name]["summary"] if item["space"] == space and item["group"] == group)
                    row[f"{name}_{space.replace('-', '_').lower()}"] = item[metric]
            cross_rows.append(row)
    write_csv(CROSS_ROOT / "gradient_cross_dataset.csv", cross_rows)
    return result


def mean_metric(geom: dict[str, Any], dataset: str, space: str, metric: str) -> float:
    value = geom[dataset][space].get(metric)
    return float(value) if value is not None else float("nan")


def decision_logic(geom: dict[str, Any], grad: dict[str, Any]) -> dict[str, str]:
    def compare(metric: str, favorable: str = "higher", tolerance: float = 1e-8) -> str:
        diffs = [mean_metric(geom, name, "SPACE-BIO-PROJ", metric) - mean_metric(geom, name, "SPACE-FULL", metric) for name in ("LTCC", "PRCC")]
        if favorable == "lower":
            diffs = [-value for value in diffs]
        if all(diff > tolerance for diff in diffs): return "YES"
        if all(diff < -tolerance for diff in diffs): return "NO"
        return "MIXED"
    q1 = compare("clothing_sensitivity_ratio", favorable="lower")
    # Treat cross-clothes margin and ordering as the decisive identity
    # safeguards.  Near-unchanged train PID ROC-AUC alone must not hide a
    # systematic loss of cross-clothes identity separation.
    margin_diffs = [mean_metric(geom, name, "SPACE-BIO-PROJ", "identity_margin") - mean_metric(geom, name, "SPACE-FULL", "identity_margin") for name in ("LTCC", "PRCC")]
    ordering_diffs = [mean_metric(geom, name, "SPACE-BIO-PROJ", "cross_clothes_ordering_accuracy") - mean_metric(geom, name, "SPACE-FULL", "cross_clothes_ordering_accuracy") for name in ("LTCC", "PRCC")]
    auc_diffs = [mean_metric(geom, name, "SPACE-BIO-PROJ", "same_different_pid_roc_auc") - mean_metric(geom, name, "SPACE-FULL", "same_different_pid_roc_auc") for name in ("LTCC", "PRCC")]
    if all(value >= -1e-3 for value in margin_diffs + ordering_diffs + auc_diffs):
        q2 = "YES"
    elif all(value < -1e-3 for value in margin_diffs) and all(value < -1e-3 for value in ordering_diffs):
        q2 = "NO"
    else:
        q2 = "MIXED"
    q3 = compare("identity_margin", favorable="higher")
    shared_rows = {name: {space: next(item for item in rowset["summary"] if item["space"] == space and item["group"] == "shared_backbone")
                            for space in ("SPACE-FULL", "SPACE-BIO-PROJ")}
                   for name, rowset in grad.items()}
    base_diffs = [shared_rows[name]["SPACE-BIO-PROJ"]["cos_base_relation_mean"] - shared_rows[name]["SPACE-FULL"]["cos_base_relation_mean"] for name in ("LTCC", "PRCC")]
    q4 = "YES" if all(diff > 0 for diff in base_diffs) else ("NO" if all(diff < 0 for diff in base_diffs) else "MIXED")
    novelty_values = [shared_rows[name]["SPACE-BIO-PROJ"]["novelty_vs_base_mean"] for name in ("LTCC", "PRCC")]
    q5 = "YES" if all(value is not None and value > 0.05 for value in novelty_values) else ("MIXED" if any(value is not None and value > 0.05 for value in novelty_values) else "NO")
    q6 = "NOT AVAILABLE"
    q7 = "YES" if q1 == "YES" and q2 == "YES" and q3 == "YES" and q4 == "YES" else "NO"
    q8 = "YES" if all(value == "YES" for value in (q1, q2, q3, q4, q5, q7)) else "NO"
    return {"Q1_lower_clothing_sensitivity": q1, "Q2_identity_discrimination_preserved": q2, "Q3_BIO_margin_better": q3,
            "Q4_BIO_relation_gradient_more_base_aligned": q4, "Q5_BIO_relation_gradient_novel": q5,
            "Q6_FULL_more_clothing_gradient_alignment": q6, "Q7_consistent_on_LTCC_and_PRCC": q7,
            "Q8_formal_identity_subspace_training_justified": q8,
            "promotion_decision": "PROMOTE TO PHASE 7H-B (RECOMMEND ONLY)" if q8 == "YES" else "STOP"}


def final_report(geom: dict[str, Any], grad: dict[str, Any], audit: dict[str, Any]) -> dict[str, str]:
    decision = decision_logic(geom, grad)
    lines = ["# Phase 7H-A — Identity-Subspace Relation Audit", "", f"Generated: `{utc_now()}`.", "",
             "## Scope and safeguards", "", "Diagnostic only. Frozen B0/C0 checkpoints were used; all geometry uses train images/PID/clothing labels only. No optimizer, scheduler, GradScaler, backward(), optimizer.step(), parameter update, learned subspace, PCA, mask, or new projection was used. Query/gallery/test was excluded.", "",
             "## Verified feature spaces", "", "| Space | Definition | LTCC dim | PRCC dim | Role |", "|---|---|---:|---:|---|",
             "| SPACE-FULL | `feat[0]` / `feat_bio` | 1024 | 1024 | Current executed Relation/test identity feature |",
             "| SPACE-BIO-PROJ | `head_clip_bio(feat_bio)` | 512 | 768 | Existing Bio semantic projection used by frozen `clipBio` loss |",
             "", "`SPACE-BIO-RAW` is identical to SPACE-FULL. Matched B0/C0 have no enabled `clipBioReverse` branch, so Clothing/Hair gradient leakage is **NOT AVAILABLE**.", ""]
    source_rows = []
    for name, item in audit["source_arms"].items():
        source_rows.append(f"- {name}: checkpoint `{item['checkpoint']['path']}` (SHA256 `{item['checkpoint']['sha256']}`), config `{item['config']['path']}`, pair cache `{item['pair_cache']['path']}`.")
    lines += ["## Exact source audit", ""] + source_rows + ["", "Full source/feature inventory: `reports/PHASE_7H_A_SOURCE_AND_FEATURE_AUDIT.md`.", ""]
    lines += ["## Geometry and identity discrimination", "", "| Dataset | Space | Cross-clothes sim | Nearest-negative sim | Mean margin | Margin > 0 | Same/diff ROC-AUC | Cross-clothes ordering | LOO center acc. | Clothing sensitivity |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("LTCC", "PRCC"):
        for space in FEATURE_SPACES:
            r = geom[name][space]
            lines.append(f"| {name} | {space} | {r['cross_clothes_similarity']:.6f} | {r['nearest_negative_pid_center_similarity']:.6f} | {r['identity_margin']:.6f} | {r['margin_positive_fraction']:.4f} | {r['same_different_pid_roc_auc']:.4f} | {r['cross_clothes_ordering_accuracy']:.4f} | {r['nearest_center_accuracy_loo']:.4f} | {r['clothing_sensitivity_ratio']:.4f} |")
    lines += ["", "Detailed geometry files: `geometry/`, `clothing_sensitivity/`, and `identity_discrimination/`.", "", "## FULL vs BIO pair-margin transitions", "", "| Dataset | Category | Pair fraction | PID fraction |", "|---|---|---:|---:|"]
    transition_rows = read_csv(TRANS_ROOT / "full_vs_bio_margin_transitions.csv")
    # The files are small; keep only typed values in the report.
    for row in transition_rows:
        if row["dataset"] in ("LTCC", "PRCC"):
            lines.append(f"| {row['dataset']} | {row['category']} | {float(row['pair_fraction']):.4f} | {float(row['pid_fraction']):.4f} |")
    lines += ["", "## Counterfactual Relation gradient audit", "", "| Dataset | Space | Group | ||g_rel|| | cos(g_base,g_rel) | cos(g_tri,g_rel) | Novelty vs base | Novelty vs Triplet |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for name in ("LTCC", "PRCC"):
        for row in grad[name]["summary"]:
            def fmt(value): return "NA" if value is None else f"{float(value):.4f}"
            lines.append(f"| {name} | {row['space']} | {row['group']} | {fmt(row['norm_relation_mean'])} | {fmt(row['cos_base_relation_mean'])} | {fmt(row['cos_triplet_relation_mean'])} | {fmt(row['novelty_vs_base_mean'])} | {fmt(row['novelty_vs_triplet_mean'])} |")
    lines += ["", "Gradient probes used unscaled FP32 `autograd.grad` on PPU with no backward/optimizer step. Clothing-gradient reference: **NOT AVAILABLE** because the matched B0/C0 graphs have no enabled reverse Clothing/Hair branch.", "", "## Optional one-step diagnostic", "", "**NOT RUN.** No virtual parameter update was needed to answer the main feature-space and gradient questions; no theta-prime was saved.", "", "## Q1–Q8", ""]
    lines += [f"- **{key}:** {value}" for key, value in decision.items() if key.startswith("Q")]
    lines += ["", "## Strongest mechanistic conclusion", "", "The existing Bio projection is a legitimate second space, but geometry alone is insufficient. Promotion requires the Bio projection to jointly reduce clothing sensitivity, preserve identity discrimination and margin, improve shared-backbone compatibility, retain useful gradient novelty, and do so consistently on LTCC and PRCC. The final decision below applies that joint gate.", "", "## Decision", "", f"**{decision['promotion_decision']}**", "", "**NO NEW REID MODEL TRAINING WAS STARTED. NO PHASE 7H-B WAS STARTED.**", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return decision


def finalize_run() -> None:
    audit = json.loads(SOURCE_JSON.read_text())
    geom = json.loads((GEOM_ROOT / "geometry_summary.json").read_text())
    grad = json.loads((GRAD_ROOT / "gradient_summary.json").read_text())
    # geometry_summary intentionally strips large internal arrays; gradient
    # summary already has only reportable aggregates.
    write_json(ONE_ROOT / "one_step_status.json", {"status": "NOT_RUN", "reason": "supplementary diagnostic skipped; no virtual parameter update was saved", "training_started": False})
    decision = final_report(geom, grad, audit)
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_JSON),
        "source_audit_sha256": sha256_file(SOURCE_JSON), "report": str(REPORT_PATH),
        "train_only": True, "query_gallery_used": False, "feature_spaces": list(FEATURE_SPACES),
        "new_feature_or_projection_learned": False, "optimizer_created": False, "optimizer_step_called": False,
        "backward_called": False, "training_started": False, "phase7hb_started": False,
        "clothing_gradient_reference": "NOT_AVAILABLE", "optional_one_step": "NOT_RUN", "decision": decision,
        "previous_artifacts_overwritten": False,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decision,
                      "training_started": False, "phase7hb_started": False}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "extract", "geometry", "gradient", "finalize", "all"), default="audit")
    args = parser.parse_args()
    ensure_dirs()
    if args.mode == "audit":
        audit = source_audit(); print(json.dumps({"source_verification_pass": audit["source_verification_pass"], "report": str(SOURCE_REPORT), "two_spaces": list(FEATURE_SPACES)}, indent=2)); return
    if not SOURCE_JSON.is_file():
        source_audit()
    if args.mode in ("extract", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 7H-A requires PPU/CUDA for model execution; CPU fallback is forbidden")
        device = torch.device("cuda:0")
        result = {name: extract_one(name, device) for name in ("LTCC", "PRCC")}
        write_json(BANK_ROOT / "extraction_summary.json", {"spaces": list(FEATURE_SPACES), "runs": result, "training_started": False, "optimizer_step_called": False})
        if args.mode == "extract":
            print(json.dumps(result, indent=2, sort_keys=True)); return
    if args.mode in ("geometry", "all"):
        geometry_run()
        if args.mode == "geometry":
            print(json.dumps({"geometry_completed": True, "training_started": False}, indent=2)); return
    if args.mode in ("gradient", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 7H-A requires PPU/CUDA for gradient execution; CPU fallback is forbidden")
        gradient_run(torch.device("cuda:0"))
        if args.mode == "gradient":
            print(json.dumps({"gradient_completed": True, "training_started": False, "optimizer_step_called": False}, indent=2)); return
    if args.mode in ("finalize", "all"):
        finalize_run()


if __name__ == "__main__":
    main()
