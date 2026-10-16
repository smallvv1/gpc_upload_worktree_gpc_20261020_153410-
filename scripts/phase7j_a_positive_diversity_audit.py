#!/usr/bin/env python3
"""Phase 7J-A: positive-diversity mechanism audit.

This is an offline diagnostic only.  It consumes the frozen Phase 7I LTCC
I0/I1/I2/I3 feature banks and run artifacts, constructs a train-only
same-PID canonical pair bank, replays the deterministic Phase 7I selector for
pair-budget auditing, and computes no-step FP32 gradient probes on the frozen
I0 model.  It never creates an optimizer and never calls optimizer.step().
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase6b_soft_relation_screen as b6  # noqa: E402
import phase7i_relation_specificity as p7i  # noqa: E402


PHASE = "PHASE 7J-A"
SEED = 1234
DATASET = "LTCC"
I7_ROOT = ROOT / "outputs/new_method_track/phase7i_relation_specificity"
OUT_ROOT = ROOT / "outputs/new_method_track/phase7j_a_positive_diversity_audit"
REPORT_PATH = ROOT / "reports/PHASE_7J_A_POSITIVE_DIVERSITY_MECHANISM_AUDIT.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7J_A_SOURCE_AUDIT.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7J_A_MANIFEST.json"
FEATURE_ROOT = I7_ROOT / "geometry/feature_banks"
PAIR_BANK_PATH = OUT_ROOT / "pair_bank/same_id_pair_bank.npz"
PAIR_META_PATH = OUT_ROOT / "pair_bank/train_metadata.csv"

ARM_FEATURES = {
    "I0": FEATURE_ROOT / "I0_train_features.npz",
    "I1": FEATURE_ROOT / "I1_train_features.npz",
    "I2": FEATURE_ROOT / "I2_train_features.npz",
    "I3": FEATURE_ROOT / "I3_train_features.npz",
}
ARM_CHECKPOINTS = {
    "I0": I7_ROOT.parent / "phase6c_full_schedule_confirmation/B0_full/model_final.pth",
    "I1": I7_ROOT / "I1_matched_cross_clothes/model_final.pth",
    "I2": I7_ROOT / "I2_same_clothes/model_final.pth",
    "I3": I7_ROOT / "I3_id_only_matched/model_final.pth",
}
ARM_RUN_DIRS = {
    "I0": I7_ROOT.parent / "phase6c_full_schedule_confirmation/B0_full",
    "I1": I7_ROOT / "I1_matched_cross_clothes",
    "I2": I7_ROOT / "I2_same_clothes",
    "I3": I7_ROOT / "I3_id_only_matched",
}

ARM_ORDER = ("I0", "I1", "I2", "I3")
RELATION_ARMS = ("I1", "I2", "I3")
MODEL_NAMES = ARM_ORDER
BOOTSTRAP_REPS = 300
PAIR_CHUNK = 150_000
EPS = 1e-6


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
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
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def coerce_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    def coerce(value: str) -> Any:
        if value == "":
            return value
        try:
            if any(token in value for token in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value
    return [{key: coerce(value) for key, value in row.items()} for row in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def ensure_dirs() -> None:
    names = (
        "source_audit", "pair_bank", "category_analysis", "diversity_quartiles",
        "diversity_deciles", "clothing_conditioned", "camera_conditioned",
        "correlations", "predictive", "gradient", "pair_budget", "i1_vs_i3",
        "tables", "plots",
    )
    for name in names:
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def load_feature_banks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    features: dict[str, np.ndarray] = {}
    metadata: dict[str, np.ndarray] = {}
    reference: dict[str, Any] | None = None
    for arm, path in ARM_FEATURES.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen Phase 7I feature bank: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"features", "pid", "clothes", "camid", "image_path"}
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"{path} is missing keys {sorted(missing)}")
            z = np.asarray(archive["features"], dtype=np.float32)
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            meta = {
                "pid": np.asarray(archive["pid"], dtype=np.int64),
                "clothes": np.asarray(archive["clothes"], dtype=np.int64),
                "camid": np.asarray(archive["camid"], dtype=np.int64),
                "image_path": np.asarray(archive["image_path"]).astype(str),
            }
        if reference is None:
            reference = {key: value.copy() for key, value in meta.items()}
            reference["n"] = len(z)
            reference["dim"] = z.shape[1]
        if len(z) != int(reference["n"]) or z.shape[1] != int(reference["dim"]):
            raise RuntimeError(f"feature shape mismatch for {arm}: {z.shape}")
        for key in ("pid", "clothes", "camid", "image_path"):
            if not np.array_equal(meta[key], reference[key]):
                raise RuntimeError(f"Phase 7I feature-bank metadata order mismatch for {arm}: {key}")
        features[arm] = z
        metadata[arm] = meta
    assert reference is not None
    return features, metadata


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    manifest_path = I7_ROOT / "PHASE7I_MANIFEST.json"
    source_json = I7_ROOT / "source_audit/source_audit.json"
    required = [manifest_path, source_json, I7_ROOT / "PHASE7I_ANALYSIS.json"]
    required += list(ARM_FEATURES.values()) + list(ARM_CHECKPOINTS.values())
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Phase 7I frozen source missing: {[str(path) for path in required if not path.is_file()]}")
    manifest = read_json(manifest_path)
    source = read_json(source_json)
    analysis = read_json(I7_ROOT / "PHASE7I_ANALYSIS.json")
    i3_run_manifest = read_json(ARM_RUN_DIRS["I3"] / "run_manifest.json")
    if manifest.get("seed") != SEED or manifest.get("dataset") != DATASET:
        raise RuntimeError("Phase 7I manifest is not the required LTCC seed1234 source")
    if manifest.get("feature") != "normalized feat[0]" or manifest.get("pair_loss") != "1-cosine":
        raise RuntimeError("Phase 7I feature/loss source is not the frozen feat[0] implementation")
    if float(manifest.get("lambda_rel")) != 0.1:
        raise RuntimeError("Phase 7I lambda_rel is not 0.1")
    source_checks = {
        "phase7i_complete": source.get("decision") == "TRAIN_MATCHED_I1_I2_I3",
        "phase7i_no_second_seed": bool(manifest.get("no_second_seed")),
        "phase7i_no_prcc": bool(manifest.get("no_prcc")),
        "phase7i_no_later_phase": bool(manifest.get("no_later_phase")),
        "i3_pair_logic_verified": "matched-budget ID-only" in str(i3_run_manifest.get("label", "")),
        "all_metrics_complete": all((ARM_RUN_DIRS[arm] / "metrics_per_epoch.csv").is_file() for arm in ARM_ORDER),
        "all_run_complete": all((ARM_RUN_DIRS[arm] / "RUN_COMPLETE.json").is_file() for arm in ARM_ORDER),
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"Phase 7I source verification failed: {source_checks}")
    records = {}
    for arm in ARM_ORDER:
        run = ARM_RUN_DIRS[arm]
        records[arm] = {
            "checkpoint": file_record(ARM_CHECKPOINTS[arm]),
            "feature_bank": file_record(ARM_FEATURES[arm]),
            "run_manifest": file_record(run / "run_manifest.json"),
            "config": file_record(run / "config.yml"),
            "metrics": file_record(run / "metrics_per_epoch.csv"),
            "pair_budget": file_record(run / "pair_budget_per_epoch.csv"),
            "relation_exposure": file_record(run / "relation_exposure.csv"),
            "amp_statistics": file_record(run / "amp_statistics.csv"),
            "label": read_json(run / "run_manifest.json").get("label") if (run / "run_manifest.json").is_file() else None,
        }
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "dataset": DATASET,
        "seed": SEED,
        "frozen_phase7i_manifest": file_record(manifest_path),
        "frozen_phase7i_source_audit": file_record(source_json),
        "phase7i_analysis": file_record(I7_ROOT / "PHASE7I_ANALYSIS.json"),
        "feature_source": "normalized feature bank from executed feat[0] relation feature; train-only",
        "pair_target": "I3 endpoint identity-margin change: M_I3 - M_I0",
        "source_checks": source_checks,
        "arms": records,
        "phase7i_questions": analysis.get("questions", {}),
        "training_started_in_phase7j_a": False,
        "optimizer_created_in_phase7j_a": False,
        "optimizer_step_called_in_phase7j_a": False,
        "query_gallery_used": False,
    }
    write_json(OUT_ROOT / "source_audit/source_audit.json", audit)
    lines = [
        f"# Phase 7J-A source audit", "", f"Generated: `{audit['created_utc']}`.", "",
        "Decision: **PASS**. Phase 7I LTCC seed1234 frozen sources are present and internally verified.", "",
        "## Frozen implementation", "",
        "- Dataset: LTCC train split only; Phase 7I metadata is order-checked across all four banks.",
        "- Feature: normalized executed `feat[0]` (1024 dimensions in the frozen banks).",
        "- I1: same PID, different clothes; I2: same PID, same clothes; I3: same PID, matched-budget ID-only.",
        "- Pair loss: `1 - cosine`; relation coefficient: `lambda_rel = 0.1`.",
        "- I3 selector reconstruction uses the Phase 7I deterministic per-batch minimum-feasible budget and stable hash selector.",
        "- No query/gallery/test labels or retrieval outputs are used in pair construction.", "",
        "## Exact frozen sources", "",
        "| Arm | Checkpoint | Checkpoint SHA256 | Feature bank | Feature SHA256 | Config |", 
        "|---|---|---|---|---|---|",
    ]
    for arm in ARM_ORDER:
        r = records[arm]
        lines.append(
            f"| {arm} | `{r['checkpoint']['path']}` | `{r['checkpoint']['sha256']}` | "
            f"`{r['feature_bank']['path']}` | `{r['feature_bank']['sha256']}` | `{r['config']['path']}` |"
        )
    lines += ["", "## Safety", "", "This phase is diagnostic only. No model was trained, no optimizer was created, and no optimizer step was called.", ""]
    SOURCE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return audit


def canonical_pair_arrays(pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    pid_parts: list[np.ndarray] = []
    for pid in np.sort(np.unique(pids)):
        indices = np.flatnonzero(pids == pid)
        if len(indices) < 2:
            continue
        ii, jj = np.triu_indices(len(indices), k=1)
        a_parts.append(indices[ii].astype(np.int32))
        b_parts.append(indices[jj].astype(np.int32))
        pid_parts.append(np.full(len(ii), int(pid), dtype=np.int32))
    return np.concatenate(a_parts), np.concatenate(b_parts), np.concatenate(pid_parts)


def pid_centers(z: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.sort(np.unique(pids)).astype(np.int64)
    lookup = {int(pid): index for index, pid in enumerate(labels.tolist())}
    centers = np.zeros((len(labels), z.shape[1]), dtype=np.float32)
    for index, pid in enumerate(labels.tolist()):
        center = z[pids == pid].mean(axis=0)
        centers[index] = center / max(float(np.linalg.norm(center)), 1e-12)
    sim = z @ centers.T
    own = np.asarray([lookup[int(pid)] for pid in pids], dtype=np.int64)
    sim[np.arange(len(z)), own] = -1e4
    nearest = sim.max(axis=1).astype(np.float32)
    return labels, centers, nearest


def state_geometry(z: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(zip(pids.tolist(), clothes.tolist())))
    lookup = {key: index for index, key in enumerate(keys)}
    state_id = np.asarray([lookup[(int(pid), int(cloth))] for pid, cloth in zip(pids, clothes)], dtype=np.int32)
    centers = np.zeros((len(keys), z.shape[1]), dtype=np.float32)
    dispersion = np.zeros(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        subset = z[state_id == index]
        center = subset.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        centers[index] = center
        dispersion[index] = float(np.mean(1.0 - subset @ center))
    state_keys = np.asarray(keys, dtype=np.int32)
    return state_id, centers, dispersion, state_keys, np.asarray([lookup[(int(pid), int(cloth))] for pid, cloth in zip(pids, clothes)], dtype=np.int32)


def pair_dot(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for start in range(0, len(a), PAIR_CHUNK):
        end = min(len(a), start + PAIR_CHUNK)
        out[start:end] = np.sum(z[a[start:end]] * z[b[start:end]], axis=1, dtype=np.float32)
    return out


def build_pair_bank(features: dict[str, np.ndarray], metadata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if PAIR_BANK_PATH.is_file() and PAIR_META_PATH.is_file():
        with np.load(PAIR_BANK_PATH, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    pids = metadata["I0"]["pid"]
    clothes = metadata["I0"]["clothes"]
    cams = metadata["I0"]["camid"]
    a, b, pair_pid = canonical_pair_arrays(pids)
    d_cloth = (clothes[a] != clothes[b]).astype(np.int8)
    d_cam = (cams[a] != cams[b]).astype(np.int8)
    state_id, state_centers, state_disp, state_keys, _ = state_geometry(features["I0"], pids, clothes)
    pair_state_a = state_id[a]
    pair_state_b = state_id[b]
    d_state = (1.0 - np.sum(state_centers[pair_state_a] * state_centers[pair_state_b], axis=1)).astype(np.float32)
    disp_a = state_disp[pair_state_a]
    disp_b = state_disp[pair_state_b]
    result: dict[str, np.ndarray] = {
        "a": a, "b": b, "pid": pair_pid,
        "clothes_a": clothes[a].astype(np.int16), "clothes_b": clothes[b].astype(np.int16),
        "cam_a": cams[a].astype(np.int16), "cam_b": cams[b].astype(np.int16),
        "d_cloth": d_cloth, "d_cam": d_cam, "d_state": d_state,
        "state_disp_a": disp_a, "state_disp_b": disp_b,
    }
    negatives: dict[str, np.ndarray] = {}
    for arm in ARM_ORDER:
        z = features[arm]
        _labels, _centers, nearest = pid_centers(z, pids)
        neg_a = nearest[a]
        neg_b = nearest[b]
        pos = pair_dot(z, a, b)
        neg_pair = np.maximum(neg_a, neg_b).astype(np.float32)
        margin = (pos - neg_pair).astype(np.float32)
        result[f"s_pos_{arm}"] = pos
        result[f"s_neg_i_{arm}"] = neg_a
        result[f"s_neg_j_{arm}"] = neg_b
        result[f"s_neg_{arm}"] = neg_pair
        result[f"m_{arm}"] = margin
        negatives[arm] = nearest
    result["d_feat"] = (1.0 - result["s_pos_I0"]).astype(np.float32)
    result["m_base"] = result["m_I0"].copy()
    for arm in RELATION_ARMS:
        result[f"delta_pos_{arm}"] = (result[f"s_pos_{arm}"] - result["s_pos_I0"]).astype(np.float32)
        result[f"delta_neg_{arm}"] = (result[f"s_neg_{arm}"] - result["s_neg_I0"]).astype(np.float32)
        result[f"delta_m_{arm}"] = (result[f"m_{arm}"] - result["m_I0"]).astype(np.float32)
        result[f"useful_{arm}"] = (result[f"delta_m_{arm}"] > EPS).astype(np.int8)
        result[f"harmful_{arm}"] = (result[f"delta_m_{arm}"] < -EPS).astype(np.int8)
        result[f"neutral_{arm}"] = (1 - result[f"useful_{arm}"] - result[f"harmful_{arm}"]).astype(np.int8)
    np.savez_compressed(PAIR_BANK_PATH, **result)
    metadata_rows = []
    for index in range(len(pids)):
        metadata_rows.append({
            "index": index, "image_path": metadata["I0"]["image_path"][index],
            "pid": int(pids[index]), "clothes": int(clothes[index]), "camid": int(cams[index]),
        })
    write_csv(PAIR_META_PATH, metadata_rows, ["index", "image_path", "pid", "clothes", "camid"])
    write_json(OUT_ROOT / "pair_bank/state_centers_i0.json", {
        "state_count": len(state_keys),
        "state_keys": state_keys.tolist(),
        "state_dispersion_mean": float(state_disp.mean()),
        "state_dispersion_median": float(np.median(state_disp)),
    })
    write_json(OUT_ROOT / "pair_bank/schema.json", {
        "pair_count": int(len(a)),
        "pair_definition": "unordered i<j, same PID, train set only",
        "feature": "normalized feat[0], 1024 dimensions",
        "d_feat": "1 - baseline I0 cosine",
        "d_state": "1 - cosine of baseline I0 normalized PID-clothing-state centers",
        "state_dispersion": "mean 1-cosine from image feature to its baseline state center",
        "m_arm": "s_pos - max(endpoint nearest wrong-PID center similarity for the two endpoints)",
        "target": "delta_m_I3 = m_I3 - m_I0",
        "all_arrays": sorted(result),
    })
    return result


def load_pair_bank() -> dict[str, np.ndarray]:
    with np.load(PAIR_BANK_PATH, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def pid_count(mask: np.ndarray, pids: np.ndarray) -> int:
    return int(np.unique(pids[mask]).size) if np.any(mask) else 0


def category_name(d_cloth: int, d_cam: int) -> str:
    return f"{'different' if d_cloth else 'same'} clothes / {'different' if d_cam else 'same'} camera"


def metric_summary(data: dict[str, np.ndarray], mask: np.ndarray, arm: str = "I3") -> dict[str, Any]:
    n = int(mask.sum())
    row: dict[str, Any] = {
        "pair_count": n, "pid_count": pid_count(mask, data["pid"]),
        "mean_d_feat": float(np.mean(data["d_feat"][mask])) if n else float("nan"),
        "mean_d_state": float(np.mean(data["d_state"][mask])) if n else float("nan"),
        "mean_m_base": float(np.mean(data["m_base"][mask])) if n else float("nan"),
        "mean_s_neg_base": float(np.mean(data["s_neg_I0"][mask])) if n else float("nan"),
    }
    for current in RELATION_ARMS:
        row[f"mean_delta_pos_{current}"] = float(np.mean(data[f"delta_pos_{current}"][mask])) if n else float("nan")
        row[f"mean_delta_neg_{current}"] = float(np.mean(data[f"delta_neg_{current}"][mask])) if n else float("nan")
        row[f"mean_delta_m_{current}"] = float(np.mean(data[f"delta_m_{current}"][mask])) if n else float("nan")
        row[f"useful_pct_{current}"] = float(np.mean(data[f"useful_{current}"][mask])) if n else float("nan")
        row[f"harmful_pct_{current}"] = float(np.mean(data[f"harmful_{current}"][mask])) if n else float("nan")
    row["mean_state_dispersion"] = float(np.mean((data["state_disp_a"][mask] + data["state_disp_b"][mask]) / 2.0)) if n else float("nan")
    return row


def category_analysis(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for dcloth in (0, 1):
        for dcam in (0, 1):
            mask = (data["d_cloth"] == dcloth) & (data["d_cam"] == dcam)
            rows.append({"category": category_name(dcloth, dcam), "d_cloth": dcloth, "d_cam": dcam, **metric_summary(data, mask)})
    write_csv(OUT_ROOT / "category_analysis/clothing_camera_categories.csv", rows)
    return rows


def rank_bins(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(len(values), dtype=np.int8)
    for index, chunk in enumerate(np.array_split(order, count), start=1):
        labels[chunk] = index
    cutoffs = np.asarray([values[chunk].max() if len(chunk) else np.nan for chunk in np.array_split(order, count)], dtype=np.float32)
    return labels, cutoffs


def diversity_bins(data: dict[str, np.ndarray], count: int, output: Path, label_name: str) -> list[dict[str, Any]]:
    labels, cutoffs = rank_bins(data["d_feat"], count)
    rows = []
    for group in range(1, count + 1):
        mask = labels == group
        row = {"bin": f"Q{group}" if count == 4 else f"D{group}", "bin_index": group, "upper_rank_cutoff": float(cutoffs[group - 1]), **metric_summary(data, mask)}
        rows.append(row)
    write_csv(output, rows)
    return rows


def threshold_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        mask = data["s_pos_I0"] >= threshold
        row = {"s_pos_threshold": threshold, "pair_fraction": float(mask.mean()), "pid_count": pid_count(mask, data["pid"]), "pair_count": int(mask.sum())}
        for arm in RELATION_ARMS:
            row[f"mean_delta_pos_{arm}"] = float(data[f"delta_pos_{arm}"][mask].mean()) if mask.any() else float("nan")
            row[f"mean_delta_neg_{arm}"] = float(data[f"delta_neg_{arm}"][mask].mean()) if mask.any() else float("nan")
            row[f"mean_delta_m_{arm}"] = float(data[f"delta_m_{arm}"][mask].mean()) if mask.any() else float("nan")
        rows.append(row)
    write_csv(OUT_ROOT / "diversity_quartiles/saturation_cutpoints.csv", rows)
    return rows


def conditioned_rows(data: dict[str, np.ndarray], field: str, output: Path) -> list[dict[str, Any]]:
    rows = []
    for value, label in ((0, f"same_{field}"), (1, f"different_{field}")):
        mask = data[f"d_{field}"] == value
        row = {"condition": label, "value": value, **metric_summary(data, mask)}
        for feature_name in ("d_feat", "d_state", "m_base", "s_neg_I0"):
            rho, p = safe_spearman(data[feature_name][mask], data["delta_m_I3"][mask])
            row[f"spearman_{feature_name}_delta_m_I3"] = rho
            row[f"spearman_p_{feature_name}_delta_m_I3"] = p
        rows.append(row)
    write_csv(output, rows)
    return rows


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    value = spearmanr(x, y)
    return float(value.statistic), float(value.pvalue)


def bootstrap_rank_ci(x: np.ndarray, y: np.ndarray, groups: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = 20260920) -> tuple[float, float]:
    """PID-block bootstrap CI using global pair ranks.

    Pair ranks are computed once for the declared subset, then the Pearson
    correlation of those ranks is resampled by PID blocks.  This avoids
    materializing a million-row concatenation for every replicate while
    preserving cluster-level resampling.  The report labels this as a
    global-rank PID bootstrap approximation.
    """
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]) or len(np.unique(groups)) < 2:
        return float("nan"), float("nan")
    rx = rankdata(x, method="average").astype(np.float64)
    ry = rankdata(y, method="average").astype(np.float64)
    unique = np.sort(np.unique(groups))
    group_index = {int(pid): index for index, pid in enumerate(unique.tolist())}
    group_ids = np.asarray([group_index[int(pid)] for pid in groups], dtype=np.int32)
    sx = np.bincount(group_ids, weights=rx, minlength=len(unique))
    sy = np.bincount(group_ids, weights=ry, minlength=len(unique))
    sxx = np.bincount(group_ids, weights=rx * rx, minlength=len(unique))
    syy = np.bincount(group_ids, weights=ry * ry, minlength=len(unique))
    sxy = np.bincount(group_ids, weights=rx * ry, minlength=len(unique))
    sn = np.bincount(group_ids, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        draw = rng.integers(0, len(unique), size=len(unique))
        n = sn[draw].sum()
        sum_x, sum_y = sx[draw].sum(), sy[draw].sum()
        sum_xx, sum_yy, sum_xy = sxx[draw].sum(), syy[draw].sum(), sxy[draw].sum()
        cov = sum_xy - sum_x * sum_y / n
        vx = sum_xx - sum_x * sum_x / n
        vy = sum_yy - sum_y * sum_y / n
        values.append(cov / math.sqrt(max(vx * vy, 1e-30)))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def correlation_analysis(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    feature_map = {
        "D_feat": "d_feat", "D_state": "d_state", "M_base": "m_base", "s_neg_pair": "s_neg_I0",
    }
    subsets = {
        "all": np.ones(len(data["pid"]), dtype=bool),
        "same_clothes": data["d_cloth"] == 0,
        "cross_clothes": data["d_cloth"] == 1,
        "same_camera": data["d_cam"] == 0,
        "different_camera": data["d_cam"] == 1,
    }
    rows = []
    for subset_name, subset_mask in subsets.items():
        for feature_name, key in feature_map.items():
            x = data[key][subset_mask]
            y = data["delta_m_I3"][subset_mask]
            rho, p = safe_spearman(x, y)
            low, high = bootstrap_rank_ci(x, y, data["pid"][subset_mask], seed=20260920 + len(rows))
            rows.append({
                "subset": subset_name, "feature": feature_name, "pair_count": int(subset_mask.sum()),
                "pid_count": pid_count(subset_mask, data["pid"]), "spearman": rho, "p_value": p,
                "pid_bootstrap_ci_low": low, "pid_bootstrap_ci_high": high,
                "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_note": "global pair ranks with PID-block resampling",
            })
    write_csv(OUT_ROOT / "correlations/correlation_table.csv", rows)
    return rows


def effect_size(useful: np.ndarray, harmful: np.ndarray) -> float:
    if len(useful) == 0 or len(harmful) == 0:
        return float("nan")
    var_u = float(np.var(useful, ddof=1)) if len(useful) > 1 else 0.0
    var_h = float(np.var(harmful, ddof=1)) if len(harmful) > 1 else 0.0
    pooled = math.sqrt(max(((len(useful) - 1) * var_u + (len(harmful) - 1) * var_h) / max(len(useful) + len(harmful) - 2, 1), 1e-30))
    return float((np.mean(useful) - np.mean(harmful)) / pooled)


def univariate_predictors(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    useful = data["delta_m_I3"] > EPS
    harmful = data["delta_m_I3"] < -EPS
    valid = useful | harmful
    y = useful[valid].astype(np.int8)
    groups = data["pid"][valid]
    feature_map = {
        "D_cloth_binary": "d_cloth", "D_cam_binary": "d_cam", "D_feat": "d_feat",
        "D_state": "d_state", "M_base": "m_base", "s_neg_pair": "s_neg_I0",
    }
    rows = []
    for name, key in feature_map.items():
        x = data[key][valid].astype(np.float64)
        rho, p = safe_spearman(x, data["delta_m_I3"][valid])
        raw_auc = roc_auc_score(y, x) if len(np.unique(x)) > 1 else float("nan")
        score = x if raw_auc >= 0.5 else -x
        auc = roc_auc_score(y, score) if len(np.unique(score)) > 1 else float("nan")
        ap = average_precision_score(y, score) if len(np.unique(score)) > 1 else float("nan")
        low, high = bootstrap_rank_ci(x, data["delta_m_I3"][valid], groups, seed=20261000 + len(rows))
        rows.append({
            "feature": name, "key": key, "pair_count": int(valid.sum()), "pid_count": int(np.unique(groups).size),
            "useful_count": int(useful.sum()), "harmful_count": int(harmful.sum()), "neutral_count": int((~valid).sum()),
            "spearman_delta_m": rho, "spearman_p": p, "pid_bootstrap_ci_low": low, "pid_bootstrap_ci_high": high,
            "roc_auc_raw": raw_auc, "roc_auc_best_direction": auc, "pr_auc_best_direction": ap,
            "useful_mean": float(x[y == 1].mean()) if np.any(y == 1) else float("nan"),
            "harmful_mean": float(x[y == 0].mean()) if np.any(y == 0) else float("nan"),
            "cohens_d_useful_minus_harmful": effect_size(x[y == 1], x[y == 0]),
        })
    write_csv(OUT_ROOT / "predictive/univariate_predictors.csv", rows)
    return rows


def predictive_cv(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    useful = data["delta_m_I3"] > EPS
    harmful = data["delta_m_I3"] < -EPS
    valid = useful | harmful
    y = useful[valid].astype(np.int8)
    groups = data["pid"][valid]
    feature_sets = {
        "D_cloth_binary": ["d_cloth"],
        "D_cam_binary": ["d_cam"],
        "D_feat": ["d_feat"],
        "D_state": ["d_state"],
        "M_base": ["m_base"],
        "s_neg_pair": ["s_neg_I0"],
        "D_feat + M_base": ["d_feat", "m_base"],
        "D_feat + D_cam_binary": ["d_feat", "d_cam"],
        "D_feat + D_cloth_binary": ["d_feat", "d_cloth"],
        "D_feat + M_base + D_cam_binary + D_cloth_binary": ["d_feat", "m_base", "d_cam", "d_cloth"],
    }
    splitter = GroupKFold(n_splits=5)
    rows = []
    for name, keys in feature_sets.items():
        x = np.column_stack([data[key][valid] for key in keys]).astype(np.float64)
        fold_rows = []
        for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=150, solver="lbfgs", class_weight="balanced", random_state=SEED))
            model.fit(x[train_index], y[train_index])
            score = model.predict_proba(x[test_index])[:, 1]
            fold_rows.append({
                "fold": fold, "roc_auc": roc_auc_score(y[test_index], score),
                "pr_auc": average_precision_score(y[test_index], score),
                "accuracy": float(np.mean((score >= 0.5) == y[test_index])),
                "balanced_accuracy": float(np.mean((score >= 0.5)[y[test_index] == 1])) * 0.5 + float(np.mean((score < 0.5)[y[test_index] == 0])) * 0.5,
                "train_pairs": len(train_index), "test_pairs": len(test_index),
                "train_pids": int(np.unique(groups[train_index]).size), "test_pids": int(np.unique(groups[test_index]).size),
            })
        rows.append({
            "predictor": name, "features": ",".join(keys), "folds": 5, "pair_count": int(valid.sum()), "pid_count": int(np.unique(groups).size),
            "roc_auc_mean": float(np.mean([row["roc_auc"] for row in fold_rows])), "roc_auc_std": float(np.std([row["roc_auc"] for row in fold_rows], ddof=1)),
            "pr_auc_mean": float(np.mean([row["pr_auc"] for row in fold_rows])), "pr_auc_std": float(np.std([row["pr_auc"] for row in fold_rows], ddof=1)),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        })
        write_csv(OUT_ROOT / "predictive" / f"cv_{name.replace(' ', '_').replace('+', 'plus').replace(',', '_')}.csv", fold_rows)
    write_csv(OUT_ROOT / "predictive/grouped_cv.csv", rows)
    return rows


def quadrant_analysis(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    h_med = float(np.median(data["d_feat"]))
    m_med = float(np.median(data["m_base"]))
    high_h = data["d_feat"] >= h_med
    high_m = data["m_base"] >= m_med
    masks = {
        "Q1_high_headroom_low_margin": high_h & ~high_m,
        "Q2_high_headroom_high_margin": high_h & high_m,
        "Q3_low_headroom_low_margin": ~high_h & ~high_m,
        "Q4_low_headroom_high_margin": ~high_h & high_m,
    }
    rows = []
    for name, mask in masks.items():
        rows.append({"quadrant": name, "headroom_median": h_med, "margin_median": m_med, **metric_summary(data, mask)})
    write_csv(OUT_ROOT / "tables/quadrants.csv", rows)
    return rows


def negative_risk_analysis(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    labels, cutoffs = rank_bins(data["s_neg_I0"], 4)
    rows = []
    for group in range(1, 5):
        mask = labels == group
        row = {"risk_quartile": f"NQ{group}", "upper_rank_cutoff_s_neg": float(cutoffs[group - 1]), **metric_summary(data, mask)}
        rows.append(row)
    write_csv(OUT_ROOT / "tables/negative_risk_quartiles.csv", rows)
    return rows


def write_pair_bank_metadata(data: dict[str, np.ndarray], metadata: dict[str, np.ndarray]) -> None:
    if PAIR_META_PATH.is_file():
        return
    rows = []
    for index, path in enumerate(metadata["I0"]["image_path"]):
        rows.append({"index": index, "image_path": path, "pid": int(metadata["I0"]["pid"][index]), "clothes": int(metadata["I0"]["clothes"][index]), "camid": int(metadata["I0"]["camid"][index])})
    write_csv(PAIR_META_PATH, rows, ["index", "image_path", "pid", "clothes", "camid"])


def selected_pair_properties(a: int, b: int, data: dict[str, np.ndarray], features: dict[str, np.ndarray], metadata: dict[str, np.ndarray], sorted_pair_keys: np.ndarray, sorted_pair_rows: np.ndarray) -> dict[str, Any]:
    pids = metadata["I0"]["pid"]
    clothes = metadata["I0"]["clothes"]
    cams = metadata["I0"]["camid"]
    pos = float(np.dot(features["I0"][a], features["I0"][b]))
    key_a, key_b = (a, b) if a < b else (b, a)
    key = key_a * 10000 + key_b
    position = int(np.searchsorted(sorted_pair_keys, key))
    if position >= len(sorted_pair_keys) or int(sorted_pair_keys[position]) != key:
        raise RuntimeError(f"selected pair missing from canonical pair bank: {(key_a, key_b)}")
    row = int(sorted_pair_rows[position])
    return {
        "a": key_a, "b": key_b, "pid": int(pids[a]),
        "d_cloth": int(clothes[a] != clothes[b]), "d_cam": int(cams[a] != cams[b]),
        "d_feat": float(1.0 - pos), "d_state": float(data["d_state"][row]), "m_base": float(data["m_base"][row]),
        "clothes_a": int(clothes[a]), "clothes_b": int(clothes[b]), "cam_a": int(cams[a]), "cam_b": int(cams[b]),
    }


def pair_budget_distribution(data: dict[str, np.ndarray], features: dict[str, np.ndarray], metadata: dict[str, np.ndarray], cfg: Any, dataset: Any) -> dict[str, Any]:
    """Replay Phase 7I deterministic pair selection without model execution."""
    event_path = OUT_ROOT / "pair_budget/selected_events.csv"
    event_rows: list[dict[str, Any]] = []
    # The first run may have completed the expensive replay before an
    # interruption during summary generation.  Reuse only a complete event
    # log with the exact expected 60-epoch matched budget per arm.
    if event_path.is_file():
        with event_path.open(newline="", encoding="utf-8") as stream:
            cached_rows = list(csv.DictReader(stream))
        source_budget = read_json(I7_ROOT / "source_audit/source_audit.json").get("pair_budget_summary", {})
        expected_per_arm = int(source_budget.get("matched_relation_pairs_total", 0))
        counts = Counter(row.get("arm") for row in cached_rows)
        if expected_per_arm > 0 and all(counts[arm] == expected_per_arm for arm in RELATION_ARMS):
            for row in cached_rows:
                event_rows.append({
                    **row,
                    "epoch": int(row["epoch"]), "rank": int(row["rank"]), "iteration": int(row["iteration"]),
                    "a": int(row["a"]), "b": int(row["b"]), "pid": int(row["pid"]),
                    "d_cloth": int(row["d_cloth"]), "d_cam": int(row["d_cam"]),
                    "d_feat": float(row["d_feat"]), "d_state": float(row["d_state"]), "m_base": float(row["m_base"]),
                    "clothes_a": int(row["clothes_a"]), "clothes_b": int(row["clothes_b"]),
                    "cam_a": int(row["cam_a"]), "cam_b": int(row["cam_b"]),
                })
    if not event_rows:
        transform_train, _ = b6.base.build_img_transforms(cfg)
        active_state = b6.full_active_state(dataset)
        index_map = b6.base.build_index_map(dataset)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        relation_bank = b6.parse_relation_bank(dataset, cfg)
        pair_keys = data["a"].astype(np.int64) * 10000 + data["b"].astype(np.int64)
        pair_order = np.argsort(pair_keys, kind="mergesort")
        sorted_pair_keys = pair_keys[pair_order]
        sorted_pair_rows = pair_order.astype(np.int64)
        for epoch in range(1, int(p7i.TOTAL_EPOCHS) + 1):
            for rank in range(int(p7i.WORLD_SIZE)):
                loader, _meta = b6.base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch, rank)
                for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                    entries = [loader.dataset.samples[index][0] for index in batch_indices]
                    candidates = p7i.candidate_lists(entries, path_to_index, relation_bank)
                    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
                    if count <= 0:
                        continue
                    for arm, kind in (("I1", "cross"), ("I2", "same"), ("I3", "id")):
                        selected = p7i.stable_selection(candidates[kind], count, epoch, rank, iteration, kind)
                        for _local_i, _local_j, pair_key, _state_key in selected:
                            a, b = pair_key
                            props = selected_pair_properties(a, b, data, features, metadata, sorted_pair_keys, sorted_pair_rows)
                            event_rows.append({"arm": arm, "epoch": epoch, "rank": rank, "iteration": iteration, **props})
        write_csv(event_path, event_rows)
    counters: dict[str, Counter] = {arm: Counter() for arm in RELATION_ARMS}
    unique_pairs: dict[str, set[tuple[int, int]]] = {arm: set() for arm in RELATION_ARMS}
    pid_sets: dict[str, set[int]] = {arm: set() for arm in RELATION_ARMS}
    for row in event_rows:
        arm = row["arm"]
        counters[arm][("category", category_name(int(row["d_cloth"]), int(row["d_cam"])))] += 1
        counters[arm][("dcloth", int(row["d_cloth"]))] += 1
        counters[arm][("dcam", int(row["d_cam"]))] += 1
        pid_sets[arm].add(int(row["pid"]))
        unique_pairs[arm].add((int(row["a"]), int(row["b"])))
    # The exact same summary is computed for the full canonical bank.
    labels, _cutoffs = rank_bins(data["d_feat"], 4)
    full_rows = []
    for name, mask in (
        ("all", np.ones(len(data["pid"]), dtype=bool)),
        ("same_clothes", data["d_cloth"] == 0), ("cross_clothes", data["d_cloth"] == 1),
        ("same_camera", data["d_cam"] == 0), ("different_camera", data["d_cam"] == 1),
        ("Q1_low_diversity", labels == 1), ("Q2", labels == 2), ("Q3", labels == 3), ("Q4_high_diversity", labels == 4),
    ):
        full_rows.append({"source": "full_canonical_bank", "category": name, "count": int(mask.sum()), "fraction": float(mask.mean()), "pid_count": pid_count(mask, data["pid"])})
    selected_array = {arm: [row for row in event_rows if row["arm"] == arm] for arm in RELATION_ARMS}
    q25, q50, q75 = [float(value) for value in np.quantile(data["d_feat"], [0.25, 0.50, 0.75])]
    for arm in RELATION_ARMS:
        rows = selected_array[arm]
        total = len(rows)
        pids = {row["pid"] for row in rows}
        for name, predicate in (
            ("all", lambda row: True),
            ("same_clothes", lambda row: row["d_cloth"] == 0), ("cross_clothes", lambda row: row["d_cloth"] == 1),
            ("same_camera", lambda row: row["d_cam"] == 0), ("different_camera", lambda row: row["d_cam"] == 1),
            ("Q1_low_diversity", lambda row: row["d_feat"] <= q25),
            ("Q2", lambda row: q25 < row["d_feat"] <= q50),
            ("Q3", lambda row: q50 < row["d_feat"] <= q75),
            ("Q4_high_diversity", lambda row: row["d_feat"] > q75),
        ):
            subset = [row for row in rows if predicate(row)]
            full_rows.append({"source": arm, "category": name, "count": len(subset), "fraction": len(subset) / max(total, 1), "pid_count": len({row["pid"] for row in subset})})
    write_csv(OUT_ROOT / "pair_budget/selected_vs_full_distribution.csv", full_rows)
    write_json(OUT_ROOT / "pair_budget/selection_summary.json", {
        "selected_counts": {arm: len(selected_array[arm]) for arm in RELATION_ARMS},
        "selected_unique_pairs": {arm: len(unique_pairs[arm]) for arm in RELATION_ARMS},
        "selected_pid_count": {arm: len(pid_sets[arm]) for arm in RELATION_ARMS},
        "replay_epochs": int(p7i.TOTAL_EPOCHS), "replay_ranks": int(p7i.WORLD_SIZE),
        "selector": "Phase 7I stable_selection with count=min(cross,same,id) per rank-batch",
    })
    return {"events": event_rows, "unique_pairs": unique_pairs, "pid_sets": pid_sets}


def i1_i3_comparison(selection: dict[str, Any]) -> list[dict[str, Any]]:
    counters = {}
    for arm in ("I1", "I3"):
        counters[arm] = Counter((row["a"], row["b"]) for row in selection["events"] if row["arm"] == arm)
    keys1, keys3 = set(counters["I1"]), set(counters["I3"])
    intersection = keys1 & keys3
    union = keys1 | keys3
    overlap_occ = sum(min(counters["I1"][key], counters["I3"][key]) for key in intersection)
    rows = [{
        "i1_occurrences": sum(counters["I1"].values()), "i3_occurrences": sum(counters["I3"].values()),
        "i1_unique_pairs": len(keys1), "i3_unique_pairs": len(keys3), "unique_pair_intersection": len(intersection),
        "unique_pair_union": len(union), "unique_pair_jaccard": len(intersection) / max(len(union), 1),
        "occurrence_overlap_min_fraction": overlap_occ / max(min(sum(counters["I1"].values()), sum(counters["I3"].values())), 1),
        "i1_pid_count": len({row["pid"] for row in selection["events"] if row["arm"] == "I1"}),
        "i3_pid_count": len({row["pid"] for row in selection["events"] if row["arm"] == "I3"}),
    }]
    write_csv(OUT_ROOT / "i1_vs_i3/overlap_summary.csv", rows)
    diversity_rows = []
    for arm in ("I1", "I3"):
        values = [row for row in selection["events"] if row["arm"] == arm]
        row = {"arm": arm, "occurrence_count": len(values), "unique_pair_count": len(set((r["a"], r["b"]) for r in values)), "pid_count": len({r["pid"] for r in values})}
        for key in ("d_feat", "d_state", "m_base"):
            arr = np.asarray([r[key] for r in values], dtype=np.float64)
            row[f"mean_{key}"] = float(arr.mean())
            row[f"median_{key}"] = float(np.median(arr))
        for name, pred in (("same_clothes", lambda r: r["d_cloth"] == 0), ("cross_clothes", lambda r: r["d_cloth"] == 1), ("same_camera", lambda r: r["d_cam"] == 0), ("different_camera", lambda r: r["d_cam"] == 1)):
            row[f"fraction_{name}"] = float(np.mean([pred(r) for r in values]))
        diversity_rows.append(row)
    write_csv(OUT_ROOT / "i1_vs_i3/selected_diversity_comparison.csv", diversity_rows)
    return rows


def _group_indices(named: list[tuple[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"all": list(range(len(named))), "late_eva": [], "backbone": [], "bio_projection": []}
    block_ids = sorted({int(name.split(".")[1]) for name, _ in named if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    for index, (name, _param) in enumerate(named):
        if name.startswith("blocks.") and name.split(".")[1].isdigit():
            groups["backbone"].append(index)
            if int(name.split(".")[1]) in late_ids:
                groups["late_eva"].append(index)
        if name.startswith("head_bio") or name.startswith("head_clip_bio"):
            groups["bio_projection"].append(index)
    return groups


def grad_cos(a: list[Any], b: list[Any]) -> float:
    if not a or not b:
        return float("nan")
    dot = sum(np.sum(x * y) for x, y in zip(a, b))
    na = math.sqrt(max(sum(np.sum(x * x) for x in a), 0.0))
    nb = math.sqrt(max(sum(np.sum(y * y) for y in b), 0.0))
    return float(dot / (na * nb + 1e-12))


def grad_norm(values: list[Any]) -> float:
    return float(math.sqrt(max(sum(np.sum(value * value) for value in values), 0.0))) if values else 0.0


def grad_novelty(reference: list[Any], target: list[Any]) -> float:
    if not reference or not target:
        return float("nan")
    dot = sum(np.sum(x * y) for x, y in zip(target, reference))
    nr2 = sum(np.sum(x * x) for x in reference)
    nt2 = sum(np.sum(y * y) for y in target)
    orth2 = max(float(nt2 - dot * dot / (nr2 + 1e-12)), 0.0)
    return float(math.sqrt(orth2) / (math.sqrt(max(float(nt2), 0.0)) + 1e-12))


def gradient_diversity(features: dict[str, np.ndarray], metadata: dict[str, np.ndarray], data: dict[str, np.ndarray], cfg: Any, dataset: Any) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch.cuda import amp

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7J-A gradient audit requires PPU; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    state = torch.load(ARM_CHECKPOINTS["I0"], map_location="cpu", weights_only=False)
    model = b6.base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=b6.base.NUM_CLASSES, camera_num=b6.base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    loss_fn, _center = b6.base.make_loss(cfg, num_classes=b6.base.NUM_CLASSES)
    transform_train, _ = b6.base.build_img_transforms(cfg)
    loader, _meta = b6.base.make_epoch_loader(dataset, transform_train, b6.full_active_state(dataset), b6.base.build_index_map(dataset), epoch=60, rank=0)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    all_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    block_ids = sorted({int(name.split(".")[1]) for name, _ in all_named if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    named = [(name, parameter) for name, parameter in all_named if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late_ids]
    groups = {"late_eva": list(range(len(named)))}
    thresholds = np.quantile(data["d_feat"], [0.25, 0.75])
    collected: list[dict[str, Any]] = []
    probes = 0
    iterator = iter(loader)
    max_batches = 80
    while probes < 3 and max_batches > 0:
        max_batches -= 1
        try:
            batch = next(iterator)
        except StopIteration:
            break
        local_pairs: list[tuple[int, int, float, str]] = []
        for i in range(len(batch["pid"])):
            for j in range(i + 1, len(batch["pid"])):
                if int(batch["pid"][i]) != int(batch["pid"][j]):
                    continue
                ia = path_to_index[str(batch["image_path"][i])]
                ib = path_to_index[str(batch["image_path"][j])]
                dfeat = float(1.0 - np.dot(features["I0"][ia], features["I0"][ib]))
                label = "low" if dfeat <= thresholds[0] else ("high" if dfeat > thresholds[1] else "mid")
                local_pairs.append((i, j, dfeat, label))
        if not local_pairs:
            continue
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        camids = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        with amp.autocast(enabled=False):
            score, feat = model(images, cam_label=camids)
            base_loss = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=b6.base.MetricCapture(), step=probes + 1)
        components = getattr(loss_fn, "last_components", {})
        tri_loss = components.get("triplet")
        if tri_loss is None:
            raise RuntimeError("Triplet component unavailable for gradient diversity audit")
        z = F.normalize(feat[0].float(), dim=-1)
        losses = {"base": base_loss, "triplet": tri_loss}
        for label in ("low", "mid", "high"):
            pairs = [(i, j) for i, j, _d, current in local_pairs if current == label]
            if pairs:
                losses[label] = torch.stack([1.0 - torch.sum(z[i] * z[j]) for i, j in pairs]).mean()
        gradients = {key: torch.autograd.grad(value, [parameter for _name, parameter in named], retain_graph=True, allow_unused=True) for key, value in losses.items()}
        for label in ("low", "mid", "high"):
            if label not in gradients:
                continue
            relation_pairs = sum(1 for *_rest, current in local_pairs if current == label)
            for group_name, indices in groups.items():
                def gather(key: str) -> list[np.ndarray]:
                    return [
                        np.zeros(tuple(named[index][1].shape), dtype=np.float32)
                        if gradients[key][index] is None
                        else gradients[key][index].detach().float().cpu().numpy()
                        for index in indices
                    ]
                base_values = gather("base")
                tri_values = gather("triplet")
                rel_values = gather(label)
                collected.append({
                    "probe": probes + 1, "group": group_name, "diversity_group": label,
                    "relation_pairs": relation_pairs, "norm_relation": grad_norm(rel_values),
                    "cos_base_relation": grad_cos(base_values, rel_values),
                    "cos_triplet_relation": grad_cos(tri_values, rel_values),
                    "novelty_base": grad_novelty(base_values, rel_values),
                    "novelty_triplet": grad_novelty(tri_values, rel_values),
                })
        del gradients, losses, feat, score, base_loss
        probes += 1
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    write_csv(OUT_ROOT / "gradient/gradient_diversity.csv", collected)
    summary = []
    for group_name in sorted(set(row["group"] for row in collected)):
        for diversity_group in ("low", "mid", "high"):
            values = [row for row in collected if row["group"] == group_name and row["diversity_group"] == diversity_group]
            if not values:
                continue
            summary.append({
                "group": group_name, "diversity_group": diversity_group, "probe_count": len(values),
                "relation_pairs_mean": float(np.mean([row["relation_pairs"] for row in values])),
                **{key: float(np.nanmean([row[key] for row in values])) for key in ("norm_relation", "cos_base_relation", "cos_triplet_relation", "novelty_base", "novelty_triplet")},
            })
    write_csv(OUT_ROOT / "gradient/gradient_diversity_summary.csv", summary)
    return summary


def make_plots(data: dict[str, np.ndarray], deciles: list[dict[str, Any]], category_rows: list[dict[str, Any]], budget_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    x = [row["bin_index"] for row in deciles]
    y = [row["mean_delta_m_I3"] for row in deciles]
    plt.plot(x, y, marker="o")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Baseline feature-diversity decile (D_feat)")
    plt.ylabel("Mean ΔM: I3 − I0")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "plots/dfeat_decile_delta_margin.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    names = [row["category"] for row in category_rows]
    useful = [row["useful_pct_I3"] for row in category_rows]
    harmful = [row["harmful_pct_I3"] for row in category_rows]
    positions = np.arange(len(names))
    plt.bar(positions, useful, label="Useful")
    plt.bar(positions, harmful, bottom=useful, label="Harmful")
    plt.xticks(positions, names, rotation=25, ha="right")
    plt.ylabel("Fraction of pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "plots/category_useful_harmful_i3.png", dpi=150)
    plt.close()

    selected = [row for row in budget_rows if row["source"] in ("I1", "I3") and row["category"] in ("same_clothes", "cross_clothes")]
    if selected:
        plt.figure(figsize=(6, 4))
        labels = [f"{row['source']}\n{row['category']}" for row in selected]
        values = [row["fraction"] for row in selected]
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels)
        plt.ylabel("Selected occurrence fraction")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "plots/i1_i3_selected_clothing_mix.png", dpi=150)
        plt.close()


def decide(data: dict[str, np.ndarray], correlation_rows: list[dict[str, Any]], univariate_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    def answer(value: bool | None) -> str:
        if value is None:
            return "MIXED"
        return "YES" if value else "NO"
    useful = data["useful_I3"] == 1
    harmful = data["harmful_I3"] == 1
    rho_all = next((row["spearman"] for row in correlation_rows if row["subset"] == "all" and row["feature"] == "D_feat"), float("nan"))
    rho_same = next((row["spearman"] for row in correlation_rows if row["subset"] == "same_clothes" and row["feature"] == "D_feat"), float("nan"))
    rho_cross = next((row["spearman"] for row in correlation_rows if row["subset"] == "cross_clothes" and row["feature"] == "D_feat"), float("nan"))
    clothing_auc = next((row["roc_auc_best_direction"] for row in univariate_rows if row["feature"] == "D_cloth_binary"), float("nan"))
    dfeat_auc = next((row["roc_auc_best_direction"] for row in univariate_rows if row["feature"] == "D_feat"), float("nan"))
    full_cv = next((row["roc_auc_mean"] for row in cv_rows if row["predictor"] == "D_feat + M_base + D_cam_binary + D_cloth_binary"), float("nan"))
    dfeat_cv = next((row["roc_auc_mean"] for row in cv_rows if row["predictor"] == "D_feat"), float("nan"))
    cloth_cv = next((row["roc_auc_mean"] for row in cv_rows if row["predictor"] == "D_cloth_binary"), float("nan"))
    gradient_by_group = defaultdict(dict)
    for row in gradient_rows:
        gradient_by_group[row["group"]][row["diversity_group"]] = row
    gradient_diff = any(
        abs(float(gradient_by_group[group].get("high", {}).get("cos_base_relation", 0.0)) - float(gradient_by_group[group].get("low", {}).get("cos_base_relation", 0.0))) > 0.05
        for group in gradient_by_group if "low" in gradient_by_group[group] and "high" in gradient_by_group[group]
    )
    diversity_labels, _cutoffs = rank_bins(data["d_feat"], 10)
    decile_means = np.asarray([float(data["delta_m_I3"][diversity_labels == index].mean()) for index in range(1, 11)])
    decile_diffs = np.diff(decile_means)
    monotonic = bool((np.all(decile_diffs >= -1e-6) or np.all(decile_diffs <= 1e-6)) and (decile_means.max() - decile_means.min() > 1e-4))
    middle = decile_means[2:8]
    endpoint = np.asarray([decile_means[:2].mean(), decile_means[-2:].mean()])
    medium_sweet = bool(middle.mean() > endpoint.max() + 1e-4 and middle.max() > endpoint.max() + 1e-4)
    same_clothes = data["d_cloth"] == 0
    cross_clothes = data["d_cloth"] == 1
    same_less_diverse = float(data["d_feat"][same_clothes].mean()) < float(data["d_feat"][cross_clothes].mean())
    same_i2_weak = float(data["delta_m_I2"][same_clothes].mean()) < float(data["delta_m_I3"][cross_clothes].mean())
    i1_rows = [row for row in selection["events"] if row["arm"] == "I1"]
    i3_rows = [row for row in selection["events"] if row["arm"] == "I3"]
    i1_mean_div = float(np.mean([row["d_feat"] for row in i1_rows]))
    i3_mean_div = float(np.mean([row["d_feat"] for row in i3_rows]))
    i1_cross = float(np.mean([row["d_cloth"] for row in i1_rows]))
    i3_cross = float(np.mean([row["d_cloth"] for row in i3_rows]))
    similar_selection_structure = abs(i1_mean_div - i3_mean_div) < 0.02 and abs(i1_cross - i3_cross) < 0.10
    # The gate is intentionally conservative: predictive signal must survive
    # clothing conditioning and exceed a descriptive, not inferential, margin.
    q = {
        "Q1_useful_pairs_more_appearance_diverse": answer(float(np.mean(data["d_feat"][useful])) > float(np.mean(data["d_feat"][harmful]))),
        "Q2_D_feat_explains_beyond_clothing": answer((rho_same > 0.10 or rho_cross > 0.10) and dfeat_auc > clothing_auc),
        "Q3_camera_additional_signal": answer(abs(next((row["spearman"] for row in correlation_rows if row["subset"] == "same_camera" and row["feature"] == "D_cam_binary"), 0.0)) > 0.05 or abs(next((row["spearman"] for row in correlation_rows if row["subset"] == "different_camera" and row["feature"] == "D_cam_binary"), 0.0)) > 0.05),
        "Q4_utility_monotonic_with_diversity": answer(monotonic),
        "Q5_medium_diversity_sweet_spot": answer(medium_sweet),
        "Q6_same_clothes_failure_due_to_low_diversity": answer(same_less_diverse and same_i2_weak),
        "Q7_I1_I3_similar_diversity_structure": answer(similar_selection_structure),
        "Q8_diversity_groups_different_gradient_behavior": answer(gradient_diff),
        "Q9_sufficient_for_diversity_aware_training": answer((rho_same > 0.10 or rho_cross > 0.10) and dfeat_cv > 0.60 and full_cv > cloth_cv + 0.03 and gradient_diff and (same_less_diverse or medium_sweet)),
    }
    decision = "PROMOTE TO PHASE 7J-B RECOMMENDATION" if q["Q9_sufficient_for_diversity_aware_training"] == "YES" else "STOP"
    return {"questions": q, "decision": decision, "diagnostics": {"rho_all_D_feat": rho_all, "rho_same_D_feat": rho_same, "rho_cross_D_feat": rho_cross, "dfeat_auc": dfeat_auc, "clothing_auc": clothing_auc, "dfeat_cv_auc": dfeat_cv, "full_cv_auc": full_cv, "clothing_cv_auc": cloth_cv, "decile_means_delta_m": decile_means.tolist(), "i1_mean_d_feat": i1_mean_div, "i3_mean_d_feat": i3_mean_div}}


def build_report(audit: dict[str, Any], data: dict[str, np.ndarray], category_rows: list[dict[str, Any]], quartiles: list[dict[str, Any]], deciles: list[dict[str, Any]], saturation: list[dict[str, Any]], conditioned_clothes: list[dict[str, Any]], conditioned_camera: list[dict[str, Any]], correlations: list[dict[str, Any]], univariate: list[dict[str, Any]], cv_rows: list[dict[str, Any]], quadrants: list[dict[str, Any]], risk_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]], selection: dict[str, Any], i1_i3_rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    n = len(data["pid"])
    useful = int(data["useful_I3"].sum())
    harmful = int(data["harmful_I3"].sum())
    neutral = int(data["neutral_I3"].sum())
    count_i3 = {arm: sum(1 for row in selection["events"] if row["arm"] == arm) for arm in RELATION_ARMS}
    lines = [
        "# Phase 7J-A — Positive-Diversity Mechanism Audit", "",
        "## Scope and safety", "",
        "This is a train-only diagnostic. It uses the frozen Phase 7I LTCC seed1234 I0/I1/I2/I3 feature banks and final checkpoints. No new ReID training, optimizer creation, optimizer step, Phase 7J-B, second seed, PRCC run, or query/gallery/test construction was performed.", "",
        f"- Source audit: `{SOURCE_REPORT_PATH}`.", f"- Output root: `{OUT_ROOT}`.",
        "- Feature source: normalized executed `feat[0]`, 1024 dimensions.",
        "- Endpoint target: `U_I3 = ΔM = M_I3 − M_I0`, with `M = s_pos − max(endpoint nearest wrong-PID center similarity)`.",
        "- Canonical pair definition: unordered image pairs `i<j`, same PID, train set only.", "",
        "## Exact sources", "",
        "| Arm | Checkpoint | SHA256 | Feature bank | SHA256 |",
        "|---|---|---|---|---|",
    ]
    for arm in ARM_ORDER:
        lines.append(f"| {arm} | `{audit['arms'][arm]['checkpoint']['path']}` | `{audit['arms'][arm]['checkpoint']['sha256']}` | `{audit['arms'][arm]['feature_bank']['path']}` | `{audit['arms'][arm]['feature_bank']['sha256']}` |")
    lines += ["", "## Canonical pair bank", "", f"- Exact train-image metadata: `{PAIR_META_PATH}`.", f"- Same-PID canonical pairs: **{n:,}**.", f"- Useful I3 (`ΔM > 1e-6`): **{useful:,} ({useful / n:.4%})**.", f"- Harmful I3 (`ΔM < -1e-6`): **{harmful:,} ({harmful / n:.4%})**.", f"- Neutral I3: **{neutral:,} ({neutral / n:.4%})**.", "", "The pair bank includes image indices, PID/clothing/camera labels, I0/I1/I2/I3 positive and nearest-negative geometry, margins, deltas, baseline feature diversity, state-center diversity, and within-state dispersion.", ""]
    lines += ["## Clothing/camera category analysis", "", "| Category | Pairs | PIDs | D_feat | M_base | I3 Δpos | I3 Δneg | I3 ΔM | Useful | Harmful |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in category_rows:
        lines.append(f"| {row['category']} | {row['pair_count']:,} | {row['pid_count']} | {row['mean_d_feat']:.6f} | {row['mean_m_base']:.6f} | {row['mean_delta_pos_I3']:+.6f} | {row['mean_delta_neg_I3']:+.6f} | {row['mean_delta_m_I3']:+.6f} | {row['useful_pct_I3']:.2%} | {row['harmful_pct_I3']:.2%} |")
    lines += ["", "## Feature-diversity quartiles", "", "Q1 is the lowest baseline feature diversity; Q4 is the highest. Bins are equal-count rank bins, not a training threshold.", "", "| Bin | Pairs | PIDs | D_feat | M_base | I3 Δpos | I3 Δneg | I3 ΔM | Useful | Harmful |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in quartiles:
        lines.append(f"| {row['bin']} | {row['pair_count']:,} | {row['pid_count']} | {row['mean_d_feat']:.6f} | {row['mean_m_base']:.6f} | {row['mean_delta_pos_I3']:+.6f} | {row['mean_delta_neg_I3']:+.6f} | {row['mean_delta_m_I3']:+.6f} | {row['useful_pct_I3']:.2%} | {row['harmful_pct_I3']:.2%} |")
    lines += ["", "## Feature-diversity deciles", "", "| Decile | Pairs | D_feat | I3 ΔM | Useful | Harmful |", "|---|---:|---:|---:|---:|---:|"]
    for row in deciles:
        lines.append(f"| {row['bin']} | {row['pair_count']:,} | {row['mean_d_feat']:.6f} | {row['mean_delta_m_I3']:+.6f} | {row['useful_pct_I3']:.2%} | {row['harmful_pct_I3']:.2%} |")
    lines += ["", "Interpretation is left descriptive: the decile curve is not converted into a selector or threshold.", "", "## Conditioned analyses", "", "### Clothing condition", "", "| Condition | Pairs | D_feat→ΔM Spearman | D_state→ΔM | M_base→ΔM |", "|---|---:|---:|---:|---:|"]
    for row in conditioned_clothes:
        lines.append(f"| {row['condition']} | {row['pair_count']:,} | {row['spearman_d_feat_delta_m_I3']:+.4f} | {row['spearman_d_state_delta_m_I3']:+.4f} | {row['spearman_m_base_delta_m_I3']:+.4f} |")
    lines += ["", "### Camera condition", "", "| Condition | Pairs | D_feat→ΔM Spearman | D_state→ΔM | M_base→ΔM |", "|---|---:|---:|---:|---:|"]
    for row in conditioned_camera:
        lines.append(f"| {row['condition']} | {row['pair_count']:,} | {row['spearman_d_feat_delta_m_I3']:+.4f} | {row['spearman_d_state_delta_m_I3']:+.4f} | {row['spearman_m_base_delta_m_I3']:+.4f} |")
    lines += ["", "## Correlations with I3 geometric utility", "", "PID bootstrap uses 300 resamples of PID blocks on global pair ranks; it is a cluster-bootstrap approximation, not a statistical significance claim.", "", "| Subset | Feature | Spearman | 95% CI | Pairs |", "|---|---|---:|---:|---:|"]
    for row in correlations:
        lines.append(f"| {row['subset']} | {row['feature']} | {row['spearman']:+.4f} | [{row['pid_bootstrap_ci_low']:+.4f}, {row['pid_bootstrap_ci_high']:+.4f}] | {row['pair_count']:,} |")
    lines += ["", "## Useful/harmful target and grouped prediction", "", "| Predictor | CV ROC-AUC | CV PR-AUC | ROC std | Pairs |", "|---|---:|---:|---:|---:|"]
    for row in cv_rows:
        lines.append(f"| {row['predictor']} | {row['roc_auc_mean']:.4f} | {row['pr_auc_mean']:.4f} | {row['roc_auc_std']:.4f} | {row['pair_count']:,} |")
    lines += ["", "Univariate ranking table: `predictive/univariate_predictors.csv`. Neutral pairs are excluded from the binary target; no threshold was tuned.", "", "## H_pos/M_base quadrant and negative-risk analysis", "", "| Quadrant | Pairs | D_feat | M_base | I3 ΔM | Useful | Harmful |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in quadrants:
        lines.append(f"| {row['quadrant']} | {row['pair_count']:,} | {row['mean_d_feat']:.6f} | {row['mean_m_base']:.6f} | {row['mean_delta_m_I3']:+.6f} | {row['useful_pct_I3']:.2%} | {row['harmful_pct_I3']:.2%} |")
    lines += ["", "| Negative-risk quartile | Pairs | s_neg | M_base | I3 Δpos | I3 Δneg | I3 ΔM | Useful | Harmful |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in risk_rows:
        lines.append(f"| {row['risk_quartile']} | {row['pair_count']:,} | {row['mean_s_neg_base']:.6f} | {row['mean_m_base']:.6f} | {row['mean_delta_pos_I3']:+.6f} | {row['mean_delta_neg_I3']:+.6f} | {row['mean_delta_m_I3']:+.6f} | {row['useful_pct_I3']:.2%} | {row['harmful_pct_I3']:.2%} |")
    lines += ["", "## Gradient diversity audit", "", "No-step FP32 gradients were measured on frozen I0 batches. Relation subsets are bottom/middle/top baseline D_feat groups within the matched batch; no optimizer object or update was used.", "", "| Parameter group | Diversity | Probes | Pairs/probe | ||g_rel|| | cos(base,rel) | cos(triplet,rel) | Novelty/base | Novelty/triplet |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gradient_rows:
        lines.append(f"| {row['group']} | {row['diversity_group']} | {row['probe_count']} | {row['relation_pairs_mean']:.2f} | {row['norm_relation']:.5g} | {row['cos_base_relation']:+.4f} | {row['cos_triplet_relation']:+.4f} | {row['novelty_base']:.4f} | {row['novelty_triplet']:.4f} |")
    lines += ["", "## I3 selected-pair budget", "", f"Replayed Phase 7I deterministic selector over 60 epochs × 2 ranks; selected occurrences: I1={count_i3['I1']:,}, I2={count_i3['I2']:,}, I3={count_i3['I3']:,}. Full event log: `{OUT_ROOT / 'pair_budget/selected_events.csv'}`.", "", "The full-vs-selected distribution is in `pair_budget/selected_vs_full_distribution.csv`; this reports prevalence rather than treating selection as a new sampler.", "", "## I1 vs I3", "", f"- Unique/occurrence overlap: `{i1_i3_rows[0]}`.", f"- Diversity comparison: `{OUT_ROOT / 'i1_vs_i3/selected_diversity_comparison.csv'}`.", "", "## Q1–Q9", ""]
    for key, value in decision["questions"].items():
        lines.append(f"- **{key}**: **{value}**")
    lines += ["", f"## Decision: **{decision['decision']}**", "", "The audit does not define a training weight, threshold, selector, or new loss. If the conservative gate is not met, the safe conclusion remains generic same-identity positive consistency rather than a proven diversity-specific mechanism.", "", "## Output index", "", f"- Manifest: `{MANIFEST_PATH}`", f"- Pair bank: `{PAIR_BANK_PATH}`", f"- Report: `{REPORT_PATH}`", "", "## Stop condition", "", "**NO NEW REID TRAINING WAS STARTED. NO PHASE 7J-B WAS STARTED.** No second seed, PRCC run, sampler change, lambda change, or later phase was started."]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_existing_outputs() -> None:
    """Build the report/manifest from completed diagnostic artifacts.

    This recovery path is intentionally separate from the expensive gradient
    run.  It is used when all CSV/NPZ diagnostics exist but report assembly
    was interrupted or omitted; it performs no model execution.
    """
    ensure_dirs()
    audit = read_json(OUT_ROOT / "source_audit/source_audit.json")
    data = load_pair_bank()
    category_rows = coerce_csv_rows(read_csv(OUT_ROOT / "category_analysis/clothing_camera_categories.csv"))
    quartiles = coerce_csv_rows(read_csv(OUT_ROOT / "diversity_quartiles/feature_diversity_quartiles.csv"))
    deciles = coerce_csv_rows(read_csv(OUT_ROOT / "diversity_deciles/feature_diversity_deciles.csv"))
    saturation = coerce_csv_rows(read_csv(OUT_ROOT / "diversity_quartiles/saturation_cutpoints.csv"))
    conditioned_clothes = coerce_csv_rows(read_csv(OUT_ROOT / "clothing_conditioned/by_clothing_condition.csv"))
    conditioned_camera = coerce_csv_rows(read_csv(OUT_ROOT / "camera_conditioned/by_camera_condition.csv"))
    correlations = coerce_csv_rows(read_csv(OUT_ROOT / "correlations/correlation_table.csv"))
    univariate = coerce_csv_rows(read_csv(OUT_ROOT / "predictive/univariate_predictors.csv"))
    cv_rows = coerce_csv_rows(read_csv(OUT_ROOT / "predictive/grouped_cv.csv"))
    quadrants = coerce_csv_rows(read_csv(OUT_ROOT / "tables/quadrants.csv"))
    risk_rows = coerce_csv_rows(read_csv(OUT_ROOT / "tables/negative_risk_quartiles.csv"))
    gradient_rows = coerce_csv_rows(read_csv(OUT_ROOT / "gradient/gradient_diversity_summary.csv"))
    events_raw = read_csv(OUT_ROOT / "pair_budget/selected_events.csv")
    events = []
    for row in events_raw:
        events.append({
            **row,
            "epoch": int(row["epoch"]), "rank": int(row["rank"]), "iteration": int(row["iteration"]),
            "a": int(row["a"]), "b": int(row["b"]), "pid": int(row["pid"]),
            "d_cloth": int(row["d_cloth"]), "d_cam": int(row["d_cam"]),
            "d_feat": float(row["d_feat"]), "d_state": float(row["d_state"]), "m_base": float(row["m_base"]),
            "clothes_a": int(row["clothes_a"]), "clothes_b": int(row["clothes_b"]),
            "cam_a": int(row["cam_a"]), "cam_b": int(row["cam_b"]),
        })
    selection = {"events": events}
    i1_i3_rows = [{key: float(value) for key, value in row.items()} for row in read_csv(OUT_ROOT / "i1_vs_i3/overlap_summary.csv")]
    decision = decide(data, correlations, univariate, cv_rows, gradient_rows, selection)
    build_report(audit, data, category_rows, quartiles, deciles, saturation, conditioned_clothes, conditioned_camera, correlations, univariate, cv_rows, quadrants, risk_rows, gradient_rows, selection, i1_i3_rows, decision)
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "dataset": DATASET, "seed": SEED,
        "diagnostic_only": True, "training_started": False, "optimizer_created": False, "optimizer_step_called": False,
        "query_gallery_used": False, "source_audit": str(OUT_ROOT / "source_audit/source_audit.json"),
        "source_report": str(SOURCE_REPORT_PATH), "report": str(REPORT_PATH), "output_root": str(OUT_ROOT),
        "pair_bank": str(PAIR_BANK_PATH), "pair_count": int(len(data["a"])), "train_images": 9576,
        "useful_count_I3": int(data["useful_I3"].sum()), "harmful_count_I3": int(data["harmful_I3"].sum()), "neutral_count_I3": int(data["neutral_I3"].sum()),
        "feature": "normalized Phase 7I feat[0] feature banks", "target": "delta_m_I3 = M_I3 - M_I0", "epsilon": EPS,
        "bootstrap_reps": BOOTSTRAP_REPS, "decision": decision, "no_new_training": True, "no_phase7j_b": True,
        "artifacts": {"category_analysis": str(OUT_ROOT / "category_analysis/clothing_camera_categories.csv"), "quartiles": str(OUT_ROOT / "diversity_quartiles/feature_diversity_quartiles.csv"), "deciles": str(OUT_ROOT / "diversity_deciles/feature_diversity_deciles.csv"), "correlations": str(OUT_ROOT / "correlations/correlation_table.csv"), "univariate": str(OUT_ROOT / "predictive/univariate_predictors.csv"), "grouped_cv": str(OUT_ROOT / "predictive/grouped_cv.csv"), "gradient": str(OUT_ROOT / "gradient/gradient_diversity_summary.csv"), "pair_budget": str(OUT_ROOT / "pair_budget/selected_vs_full_distribution.csv"), "i1_i3": str(OUT_ROOT / "i1_vs_i3/overlap_summary.csv")},
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decision}, indent=2, sort_keys=True))


def main() -> None:
    ensure_dirs()
    audit = source_audit()
    features, metadata = load_feature_banks()
    write_pair_bank_metadata({}, metadata)
    data = build_pair_bank(features, metadata)
    data = load_pair_bank()
    category_rows = category_analysis(data)
    quartiles = diversity_bins(data, 4, OUT_ROOT / "diversity_quartiles/feature_diversity_quartiles.csv", "quartile")
    deciles = diversity_bins(data, 10, OUT_ROOT / "diversity_deciles/feature_diversity_deciles.csv", "decile")
    saturation = threshold_rows(data)
    conditioned_clothes = conditioned_rows(data, "cloth", OUT_ROOT / "clothing_conditioned/by_clothing_condition.csv")
    conditioned_camera = conditioned_rows(data, "cam", OUT_ROOT / "camera_conditioned/by_camera_condition.csv")
    correlations = correlation_analysis(data)
    univariate = univariate_predictors(data)
    cv_rows = predictive_cv(data)
    quadrants = quadrant_analysis(data)
    risk_rows = negative_risk_analysis(data)
    cfg = p7i.configure()
    dataset = b6.base.build_dataset(cfg)
    selection = pair_budget_distribution(data, features, metadata, cfg, dataset)
    i1_i3_rows = i1_i3_comparison(selection)
    gradient_rows = gradient_diversity(features, metadata, data, cfg, dataset)
    # Recompute only from already-written diagnostics; no model training is
    # invoked by this script.
    decision = decide(data, correlations, univariate, cv_rows, gradient_rows, selection)
    budget_dist = []
    with (OUT_ROOT / "pair_budget/selected_vs_full_distribution.csv").open(newline="", encoding="utf-8") as stream:
        budget_dist = list(csv.DictReader(stream))
    make_plots(data, deciles, category_rows, budget_dist)
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "dataset": DATASET, "seed": SEED,
        "diagnostic_only": True, "training_started": False, "optimizer_created": False, "optimizer_step_called": False,
        "query_gallery_used": False, "source_audit": str(OUT_ROOT / "source_audit/source_audit.json"),
        "source_report": str(SOURCE_REPORT_PATH), "report": str(REPORT_PATH), "output_root": str(OUT_ROOT),
        "pair_bank": str(PAIR_BANK_PATH), "pair_count": int(len(data["a"])), "train_images": int(len(metadata["I0"]["pid"])),
        "useful_count_I3": int(data["useful_I3"].sum()), "harmful_count_I3": int(data["harmful_I3"].sum()), "neutral_count_I3": int(data["neutral_I3"].sum()),
        "feature": "normalized Phase 7I feat[0] feature banks", "target": "delta_m_I3 = M_I3 - M_I0", "epsilon": EPS,
        "bootstrap_reps": BOOTSTRAP_REPS, "decision": decision,
        "no_new_training": True, "no_phase7j_b": True,
        "artifacts": {
            "category_analysis": str(OUT_ROOT / "category_analysis/clothing_camera_categories.csv"),
            "quartiles": str(OUT_ROOT / "diversity_quartiles/feature_diversity_quartiles.csv"),
            "deciles": str(OUT_ROOT / "diversity_deciles/feature_diversity_deciles.csv"),
            "correlations": str(OUT_ROOT / "correlations/correlation_table.csv"),
            "univariate": str(OUT_ROOT / "predictive/univariate_predictors.csv"),
            "grouped_cv": str(OUT_ROOT / "predictive/grouped_cv.csv"),
            "gradient": str(OUT_ROOT / "gradient/gradient_diversity_summary.csv"),
            "pair_budget": str(OUT_ROOT / "pair_budget/selected_vs_full_distribution.csv"),
            "i1_i3": str(OUT_ROOT / "i1_vs_i3/overlap_summary.csv"),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "pair_count": int(len(data["a"])), "decision": decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize_existing_outputs()
    else:
        main()
