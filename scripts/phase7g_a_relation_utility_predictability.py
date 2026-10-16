#!/usr/local/bin/python3
"""Phase 7G-A: baseline-only predictability audit for Relation utility.

Diagnostic only.  The endpoint is used only to define the retrospective
utility label Delta_M.  Every predictor is computed from the frozen Bio
train feature bank and train PID/clothing metadata.  No optimizer, scheduler,
GradScaler, optimizer step, or REID training loop is created.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path("/home/DIFFER")
SEED = 1234
EPS = 1e-6
# Keep the reproducible main label and the requested sensitivity values in
# the same feature artifact.  The report highlights 0, 1e-4, and 1e-3.
EPS_SENSITIVITY = (EPS, 0.0, 1e-4, 1e-3)
CV_FOLDS = 5
PID_BOOTSTRAPS = 300
GRADIENT_PROBES = 4

P7F_ROOT = ROOT / "outputs/new_method_track/phase7f_a_relation_headroom_audit"
P7F_SOURCE = P7F_ROOT / "source_audit/source_audit.json"
P7F_MANIFEST = P7F_ROOT / "PHASE7F_A_MANIFEST.json"
P7F_PAIR_ROOT = P7F_ROOT / "pair_headroom"
PRCC_PAIR_CACHE = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/pair_margin_analysis/pair_metrics.npz"
PRCC_PAIR_MANIFEST = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/pair_margin_analysis/pair_metrics_manifest.json"

OUT_ROOT = ROOT / "outputs/new_method_track/phase7g_a_relation_utility_predictability"
PAIR_ROOT = OUT_ROOT / "pair_features"
UNIV_ROOT = OUT_ROOT / "univariate"
CV_ROOT = OUT_ROOT / "grouped_cv"
TRANSFER_ROOT = OUT_ROOT / "cross_dataset"
QUAD_ROOT = OUT_ROOT / "quadrants"
RISK_ROOT = OUT_ROOT / "negative_risk"
SHIFT_ROOT = OUT_ROOT / "feature_shift"
GRAD_ROOT = OUT_ROOT / "gradient_validation"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
SOURCE_JSON = OUT_ROOT / "source_audit/source_audit.json"
SOURCE_REPORT = ROOT / "reports/PHASE_7G_A_SOURCE_AUDIT.md"
REPORT_PATH = ROOT / "reports/PHASE_7G_A_RELATION_UTILITY_PREDICTABILITY_AUDIT.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7G_A_MANIFEST.json"

SOURCES: dict[str, dict[str, Any]] = {
    "ltcc_b0_seed1234": {
        "dataset": "LTCC", "arm": "B0 Bio", "phase": "Phase 6C",
        "checkpoint": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b0_seed1234_train_features.npz",
    },
    "ltcc_b1_seed1234": {
        "dataset": "LTCC", "arm": "B1 Uniform Relation", "phase": "Phase 6C",
        "checkpoint": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B1_full/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b1_seed1234_train_features.npz",
    },
    "prcc_c0_seed1234": {
        "dataset": "PRCC", "arm": "C0 Bio", "phase": "Phase 7C",
        "checkpoint": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C0_E60.npy",
        "metadata": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/train_metadata.csv",
    },
    "prcc_c1_seed1234": {
        "dataset": "PRCC", "arm": "C1 Uniform Relation", "phase": "Phase 7C",
        "checkpoint": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C1_image_uniform_relation/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C1_E60.npy",
        "metadata": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/train_metadata.csv",
    },
}

FEATURE_NAMES = [
    "s_pos", "H_pos", "s_neg_i", "s_neg_j", "s_neg_pair", "M_base",
    "negative_asymmetry", "pid_num_clothes", "pid_num_images",
    "state_i_num_images", "state_j_num_images", "pair_multiplicity",
    "state_center_distance", "state_i_dispersion", "state_j_dispersion",
    "state_dispersion_mean", "state_dispersion_max",
    "share_nearest_negative_pid", "neg_center_top1_pair",
    "neg_center_top2_pair", "neg_center_gap_min", "triplet_hard_positive_pair",
    "triplet_pos_distance", "triplet_hard_pos_distance_i",
    "triplet_hard_pos_distance_j", "triplet_hard_neg_distance_pair",
    "triplet_hard_neg_distance_i", "triplet_hard_neg_distance_j",
    "triplet_margin_quantity", "triplet_i_selects_j", "triplet_j_selects_i",
]

RULES = ["H_pos", "M_base", "s_neg", "H_pos + M_base", "H_pos + s_neg"]
MODEL_NAMES = RULES + ["full_interpretable_logistic", "shallow_tree"]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _float_or_nan(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}


def ensure_dirs() -> None:
    for path in (OUT_ROOT, PAIR_ROOT, UNIV_ROOT, CV_ROOT, TRANSFER_ROOT, QUAD_ROOT, RISK_ROOT, SHIFT_ROOT, GRAD_ROOT, TABLE_ROOT, PLOT_ROOT, SOURCE_JSON.parent):
        path.mkdir(parents=True, exist_ok=True)


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    if not P7F_SOURCE.is_file() or not P7F_MANIFEST.is_file(): raise RuntimeError("Phase 7F-A source/manifest is missing")
    p7f_source = json.loads(P7F_SOURCE.read_text()); p7f_manifest = json.loads(P7F_MANIFEST.read_text())
    if not p7f_source.get("source_verification_pass") or p7f_manifest.get("training_started") or p7f_manifest.get("phase7fb_started"):
        raise RuntimeError("frozen Phase 7F-A provenance is not a clean diagnostic STOP")
    arms = {}
    for name, spec in SOURCES.items():
        required = [spec["checkpoint"], spec["feature"]] + ([spec["metadata"]] if "metadata" in spec else [])
        if not all(path.is_file() for path in required): raise RuntimeError(f"missing frozen source for {name}")
        item = dict(spec); item["checkpoint"] = file_record(spec["checkpoint"]); item["feature"] = file_record(spec["feature"])
        if "metadata" in spec: item["metadata"] = file_record(spec["metadata"])
        arms[name] = item
    cache_sources = {
        "ltcc_pair_headroom": file_record(P7F_PAIR_ROOT / "ltcc_pair_headroom.npz"),
        "prcc_pair_headroom": file_record(P7F_PAIR_ROOT / "prcc_pair_headroom.npz"),
        "prcc_phase7d_pair_metrics": file_record(PRCC_PAIR_CACHE),
        "prcc_phase7d_pair_manifest": file_record(PRCC_PAIR_MANIFEST),
        "phase7f_source_audit": file_record(P7F_SOURCE),
        "phase7f_manifest": file_record(P7F_MANIFEST),
    }
    code_sources = {"triplet_loss": file_record(ROOT / "loss/triplet_loss.py"), "phase7e_authority": file_record(ROOT / "scripts/phase7e_relation_triplet_complementarity.py")}
    audit = {"phase": "phase7g_a_relation_utility_predictability", "created_utc": utc_now(), "source_verification_pass": True, "seed": SEED, "arms": arms, "pair_cache_sources": cache_sources, "authoritative_code": code_sources, "phase7f_decision": p7f_manifest.get("decision"), "train_only": True, "endpoint_used_only_for_label": True, "query_gallery_used": False, "training_started": False, "optimizer_created": False, "optimizer_step_called": False, "note": "All predictors use frozen Bio baseline train features and metadata only. Triplet fields use exact repository Euclidean/hard-example semantics on the frozen train bank as a diagnostic proxy, not a claim about every historical batch selection."}
    write_json(SOURCE_JSON, audit)
    rows = []
    for name, item in arms.items(): rows.append({"id": name, "dataset": item["dataset"], "arm": item["arm"], "checkpoint": item["checkpoint"]["path"], "checkpoint_sha256": item["checkpoint"]["sha256"], "feature": item["feature"]["path"], "feature_sha256": item["feature"]["sha256"]})
    lines = ["# Phase 7G-A source audit", "", f"Generated: `{audit['created_utc']}`.", "", "Decision: **PASS**. Phase 7F-A frozen diagnostic provenance and the four seed1234 Bio/Uniform Relation sources are present and hashed.", "", "| ID | Dataset | Arm | Checkpoint | Checkpoint SHA256 | Feature bank | Feature SHA256 |", "|---|---|---|---|---|---|---|"]
    for row in rows: lines.append("| " + " | ".join(str(row[key]) for key in ("id", "dataset", "arm", "checkpoint", "checkpoint_sha256", "feature", "feature_sha256")) + " |")
    lines += ["", "Pair caches, Phase 7F-A manifest, and authoritative Triplet source hashes are recorded in `source_audit.json`.", "", "Leakage rule: Delta_M/Delta_pos/Delta_neg and all Uniform Relation endpoint features are labels only; no predictor uses endpoint information. All geometry uses train images/PIDs/clothes. Query/gallery/test is excluded.", "", "Triplet proxy: repository `euclidean_dist`/hard-example-mining semantics are applied to the frozen Bio train feature bank. Because pair-level historical batch membership is unavailable, this is reported as a global train-bank diagnostic proxy, not as an actual batch selection claim.", ""]
    SOURCE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return audit


def load_feature_bank(name: str) -> dict[str, Any]:
    spec = SOURCES[name]
    if spec["feature"].suffix == ".npz":
        with np.load(spec["feature"], allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32); pids = np.asarray(archive["pids"], dtype=np.int32); clothes = np.asarray(archive["clothes"], dtype=np.int32); paths = archive["paths"].astype(str)
    else:
        features = np.asarray(np.load(spec["feature"], allow_pickle=False), dtype=np.float32); meta = read_csv(spec["metadata"]); paths = np.asarray([row["image_path"] for row in meta]); pids = np.asarray([int(row["pid"]) for row in meta], dtype=np.int32); clothes = np.asarray([int(row["clothes_id"]) for row in meta], dtype=np.int32)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return {"features": features, "pids": pids, "clothes": clothes, "paths": paths}


def build_pair_arrays(pids: np.ndarray, clothes: np.ndarray) -> dict[str, np.ndarray]:
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (pid, cloth) in enumerate(zip(pids.tolist(), clothes.tolist())): by_pid_cloth[int(pid)][int(cloth)].append(idx)
    left: list[int] = []; right: list[int] = []; pair_pid: list[int] = []; ca_out: list[int] = []; cb_out: list[int] = []
    for pid in sorted(by_pid_cloth):
        states = by_pid_cloth[pid]; labels = sorted(states)
        for pos, ca in enumerate(labels[:-1]):
            for cb in labels[pos + 1:]:
                for a in states[ca]:
                    for b in states[cb]: left.append(a); right.append(b); pair_pid.append(pid); ca_out.append(ca); cb_out.append(cb)
    return {"a": np.asarray(left, dtype=np.int32), "b": np.asarray(right, dtype=np.int32), "pid": np.asarray(pair_pid, dtype=np.int32), "cloth_a": np.asarray(ca_out, dtype=np.int32), "cloth_b": np.asarray(cb_out, dtype=np.int32)}


def load_pairs(name: str, bank: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if name == "PRCC":
        with np.load(PRCC_PAIR_CACHE, allow_pickle=False) as archive:
            pairs = {"a": archive["pair_a"].astype(np.int32), "b": archive["pair_b"].astype(np.int32), "pid": archive["pid"].astype(np.int32), "cloth_a": archive["clothes_a"].astype(np.int32), "cloth_b": archive["clothes_b"].astype(np.int32)}; utility = archive["C1_margin"].astype(np.float32) - archive["C0_margin"].astype(np.float32)
    else:
        pairs = build_pair_arrays(bank["pids"], bank["clothes"])
        with np.load(P7F_PAIR_ROOT / "ltcc_pair_headroom.npz", allow_pickle=False) as archive: utility = archive["dmargin"].astype(np.float32); cached_pid = archive["pid"].astype(np.int32)
        if len(utility) != len(pairs["a"]) or not np.array_equal(cached_pid, pairs["pid"]): raise RuntimeError("LTCC pair cache order mismatch")
    if len(pairs["a"]) != len(utility): raise RuntimeError(f"{name} pair/utility count mismatch")
    if not np.all(bank["pids"][pairs["a"]] == pairs["pid"]) or not np.all(bank["pids"][pairs["b"]] == pairs["pid"]): raise RuntimeError(f"{name} pair PID mismatch")
    if np.any(bank["clothes"][pairs["a"]] == bank["clothes"][pairs["b"]]): raise RuntimeError(f"{name} contains same-clothes pair")
    return pairs, utility


def state_statistics(z: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> dict[str, Any]:
    keys = sorted(set(zip(pids.tolist(), clothes.tolist())))
    state_index = {key: idx for idx, key in enumerate(keys)}
    image_state = np.asarray([state_index[(int(pid), int(cloth))] for pid, cloth in zip(pids.tolist(), clothes.tolist())], dtype=np.int32)
    centers = np.zeros((len(keys), z.shape[1]), dtype=np.float32); counts = np.zeros(len(keys), dtype=np.int32); dispersion = np.zeros(len(keys), dtype=np.float32)
    for key, idx in state_index.items():
        mask = image_state == idx; counts[idx] = int(mask.sum()); c = z[mask].mean(axis=0); c /= max(float(np.linalg.norm(c)), 1e-12); centers[idx] = c; dispersion[idx] = float(np.mean(1.0 - z[mask] @ c))
    return {"keys": keys, "state_index": image_state, "centers": centers, "counts": counts, "dispersion": dispersion}


def negative_center_statistics(z: np.ndarray, pids: np.ndarray) -> dict[str, Any]:
    pid_values = np.asarray(sorted(set(pids.tolist())), dtype=np.int32); pid_index = {int(pid): idx for idx, pid in enumerate(pid_values)}
    centers = np.zeros((len(pid_values), z.shape[1]), dtype=np.float32)
    for pid, idx in pid_index.items():
        c = z[pids == pid].mean(axis=0); centers[idx] = c / max(float(np.linalg.norm(c)), 1e-12)
    sims = z @ centers.T; s1 = np.empty(len(z), dtype=np.float32); s2 = np.empty(len(z), dtype=np.float32); n1 = np.empty(len(z), dtype=np.int32); n2 = np.empty(len(z), dtype=np.int32)
    for i, pid in enumerate(pids.tolist()):
        values = sims[i].copy(); values[pid_index[int(pid)]] = -1e4; order = np.argsort(values)[::-1]; s1[i] = values[order[0]]; s2[i] = values[order[1]]; n1[i] = pid_values[order[0]]; n2[i] = pid_values[order[1]]
    return {"centers": centers, "pid_values": pid_values, "s1": s1, "s2": s2, "n1": n1, "n2": n2}


def triplet_bank_statistics(z: np.ndarray, pids: np.ndarray) -> dict[str, Any]:
    """Exact repository distance/mining semantics on the frozen train bank."""
    n = len(z); hard_pos_idx = np.full(n, -1, dtype=np.int32); hard_pos_dist = np.full(n, np.nan, dtype=np.float32)
    for pid in sorted(set(pids.tolist())):
        idx = np.flatnonzero(pids == pid); sim = z[idx] @ z[idx].T; np.fill_diagonal(sim, np.inf); local = np.argmin(sim, axis=1); values = sim[np.arange(len(idx)), local]; hard_pos_idx[idx] = idx[local]; hard_pos_dist[idx] = np.sqrt(np.maximum(2.0 - 2.0 * values, 0.0))
    hard_neg_sim = np.full(n, -1e4, dtype=np.float32); hard_neg_idx = np.full(n, -1, dtype=np.int32); second_neg_sim = np.full(n, -1e4, dtype=np.float32)
    block = 512
    for start in range(0, n, block):
        stop = min(n, start + block); sim = z[start:stop] @ z.T; same = pids[start:stop, None] == pids[None, :]; sim[same] = -1e4; order = np.argpartition(sim, -2, axis=1)[:, -2:]; vals = np.take_along_axis(sim, order, axis=1); sort = np.argsort(vals, axis=1)[:, ::-1]; best_col = np.arange(stop - start)[:, None]
        ordered_vals = np.take_along_axis(vals, sort, axis=1); ordered_idx = np.take_along_axis(order, sort, axis=1); hard_neg_sim[start:stop] = ordered_vals[:, 0]; second_neg_sim[start:stop] = ordered_vals[:, 1]; hard_neg_idx[start:stop] = ordered_idx[:, 0]
    return {"hard_pos_idx": hard_pos_idx, "hard_pos_dist": hard_pos_dist, "hard_neg_idx": hard_neg_idx, "hard_neg_dist": np.sqrt(np.maximum(2.0 - 2.0 * hard_neg_sim, 0.0)), "second_neg_sim": second_neg_sim}


def build_pair_features(name: str, bank: dict[str, Any], pairs: dict[str, np.ndarray], utility: np.ndarray) -> dict[str, Any]:
    z = bank["features"]; pids = bank["pids"]; clothes = bank["clothes"]; a = pairs["a"]; b = pairs["b"]; stats = state_statistics(z, pids, clothes); neg = negative_center_statistics(z, pids); tri = triplet_bank_statistics(z, pids)
    state_a = stats["state_index"][a]; state_b = stats["state_index"][b]; state_dist = 1.0 - np.sum(stats["centers"][state_a] * stats["centers"][state_b], axis=1)
    state_keys = list(zip(pids[a].tolist(), np.minimum(clothes[a], clothes[b]).tolist(), np.maximum(clothes[a], clothes[b]).tolist())); counts = defaultdict(int)
    for key in state_keys: counts[key] += 1
    multiplicity = np.asarray([counts[key] for key in state_keys], dtype=np.float32)
    pid_counts = {int(pid): int((pids == pid).sum()) for pid in set(pids.tolist())}; pid_clothes = {int(pid): int(len(set(clothes[pids == pid].tolist()))) for pid in set(pids.tolist())}
    pos = np.sum(z[a] * z[b], axis=1).astype(np.float32); sneg_i = neg["s1"][a]; sneg_j = neg["s1"][b]; sneg_pair = np.maximum(sneg_i, sneg_j); mbase = pos - sneg_pair
    pair_dist = np.sqrt(np.maximum(2.0 - 2.0 * pos, 0.0)); hn_i = tri["hard_neg_dist"][a]; hn_j = tri["hard_neg_dist"][b]; hn_pair = np.minimum(hn_i, hn_j); trip_margin = pair_dist - hn_pair
    i_selects_j = (tri["hard_pos_idx"][a] == b).astype(np.float32); j_selects_i = (tri["hard_pos_idx"][b] == a).astype(np.float32)
    values: dict[str, np.ndarray] = {
        "s_pos": pos, "H_pos": 1.0 - pos, "s_neg_i": sneg_i, "s_neg_j": sneg_j, "s_neg_pair": sneg_pair, "M_base": mbase, "negative_asymmetry": np.abs(sneg_i - sneg_j),
        "pid_num_clothes": np.asarray([pid_clothes[int(x)] for x in pairs["pid"]], dtype=np.float32), "pid_num_images": np.asarray([pid_counts[int(x)] for x in pairs["pid"]], dtype=np.float32), "state_i_num_images": stats["counts"][state_a].astype(np.float32), "state_j_num_images": stats["counts"][state_b].astype(np.float32), "pair_multiplicity": multiplicity,
        "state_center_distance": state_dist.astype(np.float32), "state_i_dispersion": stats["dispersion"][state_a], "state_j_dispersion": stats["dispersion"][state_b], "state_dispersion_mean": ((stats["dispersion"][state_a] + stats["dispersion"][state_b]) / 2.0).astype(np.float32), "state_dispersion_max": np.maximum(stats["dispersion"][state_a], stats["dispersion"][state_b]).astype(np.float32),
        "share_nearest_negative_pid": (neg["n1"][a] == neg["n1"][b]).astype(np.float32), "neg_center_top1_pair": np.maximum(neg["s1"][a], neg["s1"][b]), "neg_center_top2_pair": np.maximum(neg["s2"][a], neg["s2"][b]), "neg_center_gap_min": np.minimum(neg["s1"][a] - neg["s2"][a], neg["s1"][b] - neg["s2"][b]),
        "triplet_hard_positive_pair": np.maximum(i_selects_j, j_selects_i), "triplet_pos_distance": pair_dist, "triplet_hard_pos_distance_i": tri["hard_pos_dist"][a], "triplet_hard_pos_distance_j": tri["hard_pos_dist"][b], "triplet_hard_neg_distance_pair": hn_pair, "triplet_hard_neg_distance_i": hn_i, "triplet_hard_neg_distance_j": hn_j, "triplet_margin_quantity": trip_margin, "triplet_i_selects_j": i_selects_j, "triplet_j_selects_i": j_selects_i,
    }
    X = np.column_stack([values[key] for key in FEATURE_NAMES]).astype(np.float32); label_map = {}
    for epsilon in EPS_SENSITIVITY: label_map[f"label_eps_{epsilon:g}"] = np.where(utility > epsilon, 1, np.where(utility < -epsilon, 0, -1)).astype(np.int8)
    path = PAIR_ROOT / f"{name.lower()}_baseline_pair_features.npz"; np.savez_compressed(path, X=X, feature_names=np.asarray(FEATURE_NAMES), pid=pairs["pid"], a=a, b=b, utility=utility, nearest_negative_pid_i=neg["n1"][a], nearest_negative_pid_j=neg["n1"][b], **label_map)
    schema = {"feature_names": FEATURE_NAMES, "target": "utility=Delta_M from frozen Relation endpoint minus frozen Bio baseline", "predictor_source": "frozen Bio train feature bank only", "triplet_features": "exact repository euclidean/hard-example semantics on full frozen train bank; diagnostic proxy, not historical batch-selection ground truth", "query_gallery_used": False, "endpoint_fields_excluded_from_X": ["Delta_pos", "Delta_neg", "Delta_M", "Uniform Relation features"]}
    write_json(PAIR_ROOT / f"{name.lower()}_feature_schema.json", schema)
    return {"X": X, "feature_names": FEATURE_NAMES, "pid": pairs["pid"], "a": a, "b": b, "utility": utility, **label_map, "metadata": {"nearest_negative_pid_i": neg["n1"][a], "nearest_negative_pid_j": neg["n1"][b]}}


def load_pair_feature_artifact(name: str) -> dict[str, Any] | None:
    path = PAIR_ROOT / f"{name.lower()}_baseline_pair_features.npz"
    if not path.is_file(): return None
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if "label_eps_1e-06" not in data or len(data["X"]) == 0: return None
    data["feature_names"] = FEATURE_NAMES
    data["metadata"] = {"nearest_negative_pid_i": data.get("nearest_negative_pid_i"), "nearest_negative_pid_j": data.get("nearest_negative_pid_j")}
    return data


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12: return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pid_bootstrap_spearman(x: np.ndarray, utility: np.ndarray, pids: np.ndarray) -> tuple[float | None, float | None, float | None]:
    unique = np.asarray(sorted(set(pids.tolist())), dtype=np.int32); x_means = np.asarray([x[pids == pid].mean() for pid in unique]); u_means = np.asarray([utility[pids == pid].mean() for pid in unique]); point = spearman(x_means, u_means); rng = np.random.default_rng(SEED + 917); values = []
    for _ in range(PID_BOOTSTRAPS):
        chosen = rng.integers(0, len(unique), size=len(unique)); value = spearman(x_means[chosen], u_means[chosen]);
        if value is not None: values.append(value)
    return point, (float(np.percentile(values, 2.5)) if values else None), (float(np.percentile(values, 97.5)) if values else None)


def auc_metrics(score: np.ndarray, y: np.ndarray, hard_label: bool = False) -> dict[str, float | None]:
    if len(np.unique(y)) < 2: return {"roc_auc": None, "pr_auc": None, "accuracy": None, "balanced_accuracy": None, "brier": None}
    probability = score if np.all((score >= 0) & (score <= 1)) else 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    pred = (probability >= 0.5).astype(np.int8) if hard_label else (score >= 0).astype(np.int8)
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "brier": float(brier_score_loss(y, probability))}


def univariate_analysis(name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    X = data["X"]; utility = data["utility"]; y = data["label_eps_1e-06"]; keep = y >= 0; yb = y[keep]; rows = []
    for idx, feature in enumerate(FEATURE_NAMES):
        value = X[:, idx]; raw_auc = float(roc_auc_score(yb, value[keep])) if len(np.unique(value[keep])) > 1 else None; useful = value[keep][yb == 1]; harmful = value[keep][yb == 0]; pooled = math.sqrt(max(((len(useful) - 1) * np.var(useful) + (len(harmful) - 1) * np.var(harmful)) / max(1, len(useful) + len(harmful) - 2), 1e-12)); point, low, high = pid_bootstrap_spearman(value, utility, data["pid"])
        oriented_score = value[keep] if raw_auc is None or raw_auc >= 0.5 else -value[keep]
        rows.append({"dataset": name, "feature": feature, "spearman_pair_utility": spearman(value, utility), "roc_auc_raw_useful": raw_auc, "roc_auc_oriented": max(raw_auc, 1.0 - raw_auc) if raw_auc is not None else None, "pr_auc_raw_useful": float(average_precision_score(yb, value[keep])) if raw_auc is not None else None, "pr_auc_oriented": float(average_precision_score(yb, oriented_score)) if raw_auc is not None else None, "effect_useful_minus_harmful": float((useful.mean() - harmful.mean()) / pooled), "useful_mean": float(useful.mean()), "harmful_mean": float(harmful.mean()), "pid_spearman": point, "pid_bootstrap_low95": low, "pid_bootstrap_high95": high})
    rows.sort(key=lambda row: row["roc_auc_oriented"] if row["roc_auc_oriented"] is not None else -1, reverse=True); write_csv(UNIV_ROOT / f"{name.lower()}_univariate.csv", rows); return rows


def scaler_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0); std = X.std(axis=0); std[std < 1e-8] = 1.0; return mean, std


def scaler_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray: return (X - mean) / std


def rule_direction(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    """Choose only the sign of each single-feature rule on predictor-train data."""
    directions = {}
    for feature in ("H_pos", "M_base", "s_neg_pair"):
        value = X_train[:, FEATURE_NAMES.index(feature)]
        directions[feature] = 1.0 if len(np.unique(value)) < 2 or roc_auc_score(y_train, value) >= 0.5 else -1.0
    return directions


def rule_scores(X: np.ndarray, mean: np.ndarray, std: np.ndarray, directions: dict[str, float]) -> dict[str, np.ndarray]:
    z = scaler_apply(X, mean, std); h = directions["H_pos"] * z[:, FEATURE_NAMES.index("H_pos")]; m = directions["M_base"] * z[:, FEATURE_NAMES.index("M_base")]; s = directions["s_neg_pair"] * z[:, FEATURE_NAMES.index("s_neg_pair")]
    return {"H_pos": h, "M_base": m, "s_neg": s, "H_pos + M_base": h + m, "H_pos + s_neg": h + s}


def model_fit_score(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    mean, std = scaler_fit(X_train); z_train = scaler_apply(X_train, mean, std); z_eval = scaler_apply(X_eval, mean, std)
    if model_name in RULES:
        directions = rule_direction(X_train, y_train); score = rule_scores(X_eval, mean, std, directions)[model_name]; return score, None, {"normalization_mean": mean, "normalization_std": std, "rule_directions": directions}
    if model_name == "full_interpretable_logistic":
        model = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=SEED); model.fit(z_train, y_train); return model.predict_proba(z_eval)[:, 1], model.predict_proba(z_eval)[:, 1], {"model": model, "normalization_mean": mean, "normalization_std": std}
    if model_name == "shallow_tree":
        model = DecisionTreeClassifier(max_depth=3, random_state=SEED); model.fit(X_train, y_train); probability = model.predict_proba(X_eval)[:, 1]; return probability, probability, {"model": model, "normalization_mean": mean, "normalization_std": std}
    raise ValueError(model_name)


def cv_analysis(name: str, data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    X = data["X"]; y_all = data["label_eps_1e-06"]; keep = y_all >= 0; X = X[keep]; y = y_all[keep].astype(np.int8); groups = data["pid"][keep]; splitter = GroupKFold(n_splits=CV_FOLDS); rows: list[dict[str, Any]] = []; oof: dict[str, np.ndarray] = {model: np.full(len(y), np.nan, dtype=np.float32) for model in MODEL_NAMES}
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), 1):
        for model_name in MODEL_NAMES:
            score, probability, _ = model_fit_score(model_name, X[train_idx], y[train_idx], X[test_idx]); metric = auc_metrics(score, y[test_idx], hard_label=model_name in ("full_interpretable_logistic", "shallow_tree")); rows.append({"dataset": name, "fold": fold, "model": model_name, **metric, "train_pid_count": len(set(groups[train_idx].tolist())), "test_pid_count": len(set(groups[test_idx].tolist())), "test_pair_count": len(test_idx)}); oof[model_name][test_idx] = score
    summary = []
    for model_name in MODEL_NAMES:
        values = [row for row in rows if row["model"] == model_name]; summary.append({"dataset": name, "model": model_name, "fold_count": len(values), **{key: float(np.mean([row[key] for row in values])) for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "brier")}, **{f"{key}_std": float(np.std([row[key] for row in values])) for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "brier")}})
    write_csv(CV_ROOT / f"{name.lower()}_fold_metrics.csv", rows); write_csv(CV_ROOT / f"{name.lower()}_summary.csv", summary)
    return rows, summary, oof


def bootstrap_oof(name: str, data: dict[str, Any], oof: dict[str, np.ndarray], selected_models: list[str]) -> list[dict[str, Any]]:
    y_all = data["label_eps_1e-06"]; keep = y_all >= 0; y = y_all[keep].astype(np.int8); pids = data["pid"][keep]; unique = np.asarray(sorted(set(pids.tolist())), dtype=np.int32); by_pid = {int(pid): np.flatnonzero(pids == pid) for pid in unique}; rng = np.random.default_rng(SEED + 991); rows = []
    for model in selected_models:
        score_all = oof[model]
        vals_roc = []; vals_pr = []
        for _ in range(PID_BOOTSTRAPS):
            chosen = rng.choice(unique, size=len(unique), replace=True); indices = np.concatenate([by_pid[int(pid)] for pid in chosen]); vals_roc.append(roc_auc_score(y[indices], score_all[indices])); vals_pr.append(average_precision_score(y[indices], score_all[indices]))
        rows.append({"dataset": name, "model": model, "bootstrap_n": PID_BOOTSTRAPS, "roc_auc_low95": float(np.percentile(vals_roc, 2.5)), "roc_auc_high95": float(np.percentile(vals_roc, 97.5)), "pr_auc_low95": float(np.percentile(vals_pr, 2.5)), "pr_auc_high95": float(np.percentile(vals_pr, 97.5))})
    write_csv(CV_ROOT / f"{name.lower()}_pid_bootstrap.csv", rows); return rows


def cross_transfer(source_name: str, target_name: str, source: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    sx, sy_all = source["X"], source["label_eps_1e-06"]; tx, ty_all = target["X"], target["label_eps_1e-06"]; sk = sy_all >= 0; tk = ty_all >= 0; sy = sy_all[sk].astype(np.int8); ty = ty_all[tk].astype(np.int8); rows = []
    for model_name in MODEL_NAMES:
        score, probability, _ = model_fit_score(model_name, sx[sk], sy, tx[tk]); metric = auc_metrics(score, ty, hard_label=model_name in ("full_interpretable_logistic", "shallow_tree")); rows.append({"source": source_name, "target": target_name, "model": model_name, "source_pid_count": len(set(source["pid"][sk].tolist())), "target_pid_count": len(set(target["pid"][tk].tolist())), **metric})
    write_csv(TRANSFER_ROOT / f"{source_name.lower()}_to_{target_name.lower()}.csv", rows); return rows


def quadrant_analysis(name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    h = data["X"][:, FEATURE_NAMES.index("H_pos")]; m = data["X"][:, FEATURE_NAMES.index("M_base")]; med_h = float(np.median(h)); med_m = float(np.median(m)); high_h = h >= med_h; high_m = m >= med_m; labels = data["label_eps_1e-06"]; rows = []
    for quadrant, mask in (("Q1_high_headroom_low_margin", high_h & ~high_m), ("Q2_high_headroom_high_margin", high_h & high_m), ("Q3_low_headroom_low_margin", ~high_h & ~high_m), ("Q4_low_headroom_high_margin", ~high_h & high_m)):
        useful = (labels[mask] == 1); harmful = (labels[mask] == 0); rows.append({"dataset": name, "quadrant": quadrant, "headroom_median": med_h, "margin_median": med_m, "pair_count": int(mask.sum()), "pid_count": len(set(data["pid"][mask].tolist())), "useful_fraction": float(useful.mean()), "harmful_fraction": float(harmful.mean()), "neutral_fraction": float((labels[mask] < 0).mean()), "mean_delta_M": float(data["utility"][mask].mean())})
    write_csv(QUAD_ROOT / f"{name.lower()}_quadrants.csv", rows); return rows


def negative_risk_analysis(name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    score = data["X"][:, FEATURE_NAMES.index("s_neg_pair")]; q = np.minimum(4, np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75]), right=True) + 1); labels = data["label_eps_1e-06"]; rows = []
    for quartile in range(1, 5):
        mask = q == quartile; rows.append({"dataset": name, "negative_risk_quartile": f"NQ{quartile}", "pair_count": int(mask.sum()), "pid_count": len(set(data["pid"][mask].tolist())), "baseline_s_neg": float(score[mask].mean()), "baseline_M": float(data["X"][mask, FEATURE_NAMES.index("M_base")].mean()), "useful_fraction": float((labels[mask] == 1).mean()), "harmful_fraction": float((labels[mask] == 0).mean()), "neutral_fraction": float((labels[mask] < 0).mean()), "mean_delta_pos_not_available": "endpoint_excluded_from_predictors", "mean_delta_M": float(data["utility"][mask].mean())})
    # Add the requested endpoint deltas only to the diagnostic table; they are
    # labels/analysis, never columns of X.
    npz = np.load(P7F_PAIR_ROOT / f"{name.lower()}_pair_headroom.npz", allow_pickle=False); dpos = np.asarray(npz["dpos"], dtype=np.float32); dneg = np.asarray(npz["dneg"], dtype=np.float32)
    for row, quartile in zip(rows, range(1, 5)):
        mask = q == quartile; row["mean_delta_pos"] = float(dpos[mask].mean()); row["mean_delta_neg"] = float(dneg[mask].mean())
    write_csv(RISK_ROOT / f"{name.lower()}_negative_risk_quartiles.csv", rows); return rows


def feature_shift(ltcc: dict[str, Any], prcc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, feature in enumerate(FEATURE_NAMES):
        a = ltcc["X"][:, idx].astype(np.float64); b = prcc["X"][:, idx].astype(np.float64); pooled = math.sqrt(max((a.var() + b.var()) / 2.0, 1e-12)); rows.append({"feature": feature, "LTCC_mean": float(a.mean()), "LTCC_median": float(np.median(a)), "LTCC_p25": float(np.percentile(a, 25)), "LTCC_p75": float(np.percentile(a, 75)), "PRCC_mean": float(b.mean()), "PRCC_median": float(np.median(b)), "PRCC_p25": float(np.percentile(b, 25)), "PRCC_p75": float(np.percentile(b, 75)), "standardized_shift_PRCC_minus_LTCC": float((b.mean() - a.mean()) / pooled)})
    rows.sort(key=lambda row: abs(row["standardized_shift_PRCC_minus_LTCC"]), reverse=True); write_csv(SHIFT_ROOT / "ltcc_vs_prcc_feature_shift.csv", rows); return rows


def plot_diagnostics(data_by_name: dict[str, dict[str, Any]]) -> None:
    status = {"plots": [], "errors": []}
    try:
        import matplotlib.pyplot as plt
        for name, data in data_by_name.items():
            x = data["X"][:, FEATURE_NAMES.index("H_pos")]; y = data["utility"]; order = np.argsort(x); take = order[::max(1, len(order) // 10000)]; plt.figure(figsize=(6, 4)); plt.scatter(x[take], y[take], s=1, alpha=.25); plt.axhline(0, color="black", lw=.7); plt.xlabel("baseline H_pos"); plt.ylabel("Delta_M"); plt.tight_layout(); path = PLOT_ROOT / f"{name.lower()}_headroom_vs_utility.png"; plt.savefig(path, dpi=150); plt.close(); status["plots"].append(path.name)
    except Exception as exc: status["errors"].append(repr(exc))
    write_json(PLOT_ROOT / "plot_status.json", status)


def gradient_helpers():
    sys.path.insert(0, str(ROOT / "scripts")); import phase5b_id_damage_ds as base  # type: ignore
    import phase7e_relation_triplet_complementarity as p7e  # type: ignore
    return base, p7e


def gradient_norm(grads: tuple[Any, ...], names: list[str], selected: set[str]) -> float:
    total = 0.0
    for name, grad in zip(names, grads):
        if name in selected and grad is not None: total += float((grad.detach().float() ** 2).sum().cpu())
    return math.sqrt(max(total, 0.0))


def gradient_dot(a: tuple[Any, ...], b: tuple[Any, ...], names: list[str], selected: set[str]) -> float:
    value = 0.0
    for name, left, right in zip(names, a, b):
        if name not in selected: continue
        if left is None and right is None: continue
        if left is None: left = right * 0.0
        if right is None: right = left * 0.0
        value += float((left.detach().float() * right.detach().float()).sum().cpu())
    return value


def grad_alignment(a: tuple[Any, ...], b: tuple[Any, ...], names: list[str], selected: set[str]) -> float | None:
    na = gradient_norm(a, names, selected); nb = gradient_norm(b, names, selected); return gradient_dot(a, b, names, selected) / (na * nb + 1e-12) if na > 1e-12 and nb > 1e-12 else None


def gradient_validation_one(name: str, data: dict[str, Any], score: np.ndarray) -> dict[str, Any]:
    base, p7e = gradient_helpers(); cfg, dataset, transform = p7e.cfg_dataset(name.lower()); device = p7e.require_ppu(); checkpoint = SOURCES["ltcc_b0_seed1234" if name == "LTCC" else "prcc_c0_seed1234"]["checkpoint"]; model = p7e.model_from_checkpoint(name.lower(), cfg, dataset, checkpoint, device); loader = p7e.make_loader(name.lower(), dataset, cfg, transform, epoch=60); bank = p7e.relation_bank(name.lower(), dataset, cfg); loss_fn, _ = base.make_loss(cfg, num_classes=dataset.num_train_pids); named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]; names = [n for n, _ in named]; params = [p for _, p in named]; selected = p7e.parameter_groups(model)["late_eva"]; pair_lookup = {(min(int(a), int(b)), max(int(a), int(b))): idx for idx, (a, b) in enumerate(zip(data["a"], data["b"]))}; low_cut = float(np.percentile(score, 25)); high_cut = float(np.percentile(score, 75)); rows = []; low_count = high_count = inspected = joint = low_candidates = high_candidates = 0
    for batch in loader:
        inspected += 1; _all, relations = p7e.local_pairs(batch, bank); indices = [bank["path_to_index"][str(path)] for path in batch["image_path"]]; low_edges = []; high_edges = []
        for i, j in relations:
            key = (min(indices[i], indices[j]), max(indices[i], indices[j])); pair_idx = pair_lookup.get(key); 
            if pair_idx is None: continue
            value = score[pair_idx]
            if value <= low_cut: low_edges.append((i, j))
            if value >= high_cut: high_edges.append((i, j))
        if low_edges: low_candidates += 1
        if high_edges: high_candidates += 1
        if low_edges and high_edges: joint += 1
        if (low_count >= GRADIENT_PROBES and high_count >= GRADIENT_PROBES) or (not low_edges and not high_edges): continue
        images = batch["image"].to(device); pids = batch["pid"].to(device); cams = batch["camid"].to(device); clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device); model.zero_grad(set_to_none=True); outputs, feats = model(images, cam_label=cams); writer = base.MetricCapture(); total = loss_fn(outputs, feats, pids, cams, captions, clothes_ids=clothes, train_writer=writer, step=inspected); comp = loss_fn.last_components; base_loss = comp["ce"] + comp["triplet"] + comp["bio"]; rep = __import__("torch").nn.functional.normalize(feats[0].float(), dim=-1); low_loss = __import__("torch").stack([1.0 - (rep[i] * rep[j]).sum() for i, j in low_edges]).mean() if low_edges and low_count < GRADIENT_PROBES else None; high_loss = __import__("torch").stack([1.0 - (rep[i] * rep[j]).sum() for i, j in high_edges]).mean() if high_edges and high_count < GRADIENT_PROBES else None; base_g = __import__("torch").autograd.grad(base_loss, params, retain_graph=True, allow_unused=True); tri_g = __import__("torch").autograd.grad(comp["triplet"], params, retain_graph=True, allow_unused=True); low_g = __import__("torch").autograd.grad(low_loss, params, retain_graph=bool(high_loss is not None), allow_unused=True) if low_loss is not None else None; high_g = __import__("torch").autograd.grad(high_loss, params, retain_graph=False, allow_unused=True) if high_loss is not None else None
        if low_g is not None: rows.append({"dataset": name, "group": "bottom_useful_quartile", "batch_index": inspected, "relation_count": len(low_edges), "joint_group_batch": bool(high_edges), "norm_relation": gradient_norm(low_g, names, selected), "cos_base_relation": grad_alignment(base_g, low_g, names, selected), "cos_triplet_relation": grad_alignment(tri_g, low_g, names, selected), "optimizer_step_called": False, "autocast_enabled": False}); low_count += 1
        if high_g is not None: rows.append({"dataset": name, "group": "top_useful_quartile", "batch_index": inspected, "relation_count": len(high_edges), "joint_group_batch": bool(low_edges), "norm_relation": gradient_norm(high_g, names, selected), "cos_base_relation": grad_alignment(base_g, high_g, names, selected), "cos_triplet_relation": grad_alignment(tri_g, high_g, names, selected), "optimizer_step_called": False, "autocast_enabled": False}); high_count += 1
        model.zero_grad(set_to_none=True); del outputs, feats, total, base_loss, low_loss, high_loss, base_g, tri_g, low_g, high_g
        if low_count >= GRADIENT_PROBES and high_count >= GRADIENT_PROBES: break
    if low_count < GRADIENT_PROBES or high_count < GRADIENT_PROBES: raise RuntimeError(f"{name} gradient validation coverage low={low_count}, high={high_count}, inspected={inspected}")
    write_csv(GRAD_ROOT / f"{name.lower()}_gradient_rows.csv", rows); summary = {"dataset": name, "probe_count": len(rows), "bottom_probe_count": low_count, "top_probe_count": high_count, "joint_group_batch_count": joint, "bottom_candidate_batches": low_candidates, "top_candidate_batches": high_candidates, "inspected_batches": inspected, "probe_design": "same frozen epoch-60 loader; groups may be in separate batches"}
    for group, prefix in (("bottom_useful_quartile", "bottom"), ("top_useful_quartile", "top")):
        group_rows = [row for row in rows if row["group"] == group]
        for key, out_key in (("norm_relation", f"{prefix}_norm_relation"), ("cos_base_relation", f"{prefix}_cos_base_relation"), ("cos_triplet_relation", f"{prefix}_cos_triplet_relation")):
            vals = np.asarray([float(row[key]) for row in group_rows]); summary[out_key] = float(vals.mean()); summary[f"{out_key}_std"] = float(vals.std())
    write_json(GRAD_ROOT / f"{name.lower()}_gradient_summary.json", summary); del loader, loss_fn, model, dataset; gc.collect(); return summary


def decisions(cv_summaries: dict[str, list[dict[str, Any]]], transfer_rows: list[dict[str, Any]], univ: dict[str, list[dict[str, Any]]], grad_summaries: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    def cv_auc(ds: str, model: str) -> float: return float(next(row["roc_auc"] for row in cv_summaries[ds] if row["model"] == model))
    def uni_auc(ds: str, feature: str) -> float: return float(next(row["roc_auc_oriented"] for row in univ[ds] if row["feature"] == feature))
    h = [uni_auc("LTCC", "H_pos"), uni_auc("PRCC", "H_pos")]; m = [uni_auc("LTCC", "M_base"), uni_auc("PRCC", "M_base")]; neg = [uni_auc("LTCC", "s_neg_pair"), uni_auc("PRCC", "s_neg_pair")]
    q1 = "YES" if min(h) >= .60 else ("MIXED" if max(h) >= .60 or min(h) >= .55 else "NO")
    q2 = "YES" if min(m[i] - h[i] for i in range(2)) >= .02 else ("MIXED" if max(m[i] - h[i] for i in range(2)) >= .02 else "NO")
    q3 = "YES" if min(neg) >= .60 and min(neg[i] - h[i] for i in range(2)) >= .02 else ("MIXED" if max(neg) >= .60 or max(neg[i] - h[i] for i in range(2)) >= .02 else "NO")
    q4 = "YES" if all(cv_auc(ds, "full_interpretable_logistic") > max(cv_auc(ds, rule) for rule in RULES) + .01 for ds in ("LTCC", "PRCC")) else "NO"
    q5 = "YES" if all(cv_auc(ds, "full_interpretable_logistic") >= .55 and float(next(row["roc_auc_std"] for row in cv_summaries[ds] if row["model"] == "full_interpretable_logistic")) <= .08 for ds in ("LTCC", "PRCC")) else ("MIXED" if any(cv_auc(ds, "full_interpretable_logistic") >= .55 for ds in ("LTCC", "PRCC")) else "NO")
    ltcc_prcc = next(row for row in transfer_rows if row["source"] == "LTCC" and row["target"] == "PRCC" and row["model"] == "full_interpretable_logistic"); prcc_ltcc = next(row for row in transfer_rows if row["source"] == "PRCC" and row["target"] == "LTCC" and row["model"] == "full_interpretable_logistic"); q6 = "YES" if float(ltcc_prcc["roc_auc"]) > .60 else "NO"; q7 = "YES" if float(prcc_ltcc["roc_auc"]) > .60 else "NO"
    q8 = None
    if grad_summaries:
        outcomes = []
        for ds, g in grad_summaries.items(): outcomes.append((g["top_cos_base_relation"] > g["bottom_cos_base_relation"]) or (g["top_cos_triplet_relation"] > g["bottom_cos_triplet_relation"]))
        q8 = "YES" if all(outcomes) else ("MIXED" if any(outcomes) else "NO")
    q9 = "YES" if q4 == "YES" and q5 == "YES" and (q6 == "YES" or q7 == "YES") and q8 in ("YES", "MIXED") else "NO"
    return {"Q1_headroom_predicts_utility": q1, "Q2_margin_improves_beyond_headroom": q2, "Q3_negative_risk_improves_prediction": q3, "Q4_multivariate_beats_single_rules": q4, "Q5_pid_grouped_stability": q5, "Q6_LTCC_to_PRCC_transfer": q6, "Q7_PRCC_to_LTCC_transfer": q7, "Q8_gradient_validation_support": q8 or "PENDING", "Q9_utility_gated_training_justified": q9, "promotion_decision": "PROMOTE recommendation only; do not train" if q9 == "YES" else "STOP", "training_started": False, "utility_gated_training_started": False, "endpoint_used_only_for_label": True, "epsilon_main": EPS, "epsilon_sensitivity": list(EPS_SENSITIVITY)}


def finalize_report(audit: dict[str, Any], data_by_name: dict[str, dict[str, Any]], univ: dict[str, list[dict[str, Any]]], cv_summaries: dict[str, list[dict[str, Any]]], transfer_rows: list[dict[str, Any]], grad_summaries: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    with np.load(PAIR_ROOT / "ltcc_baseline_pair_features.npz", allow_pickle=False) as _: pass
    grad_summaries = grad_summaries or {}
    decision = decisions(cv_summaries, transfer_rows, univ, grad_summaries if grad_summaries else None); write_json(TABLE_ROOT / "decision.json", decision)
    rows = []
    for ds in ("LTCC", "PRCC"):
        data = data_by_name[ds]; y = data["label_eps_1e-06"]; rows.append({"dataset": ds, "pair_count": len(y), "useful_count": int((y == 1).sum()), "harmful_count": int((y == 0).sum()), "neutral_count": int((y < 0).sum()), "useful_fraction": float((y == 1).mean()), "harmful_fraction": float((y == 0).mean()), "neutral_fraction": float((y < 0).mean())})
    write_csv(TABLE_ROOT / "target_counts.csv", rows)
    source_rows = [{"id": name, "dataset": item["dataset"], "arm": item["arm"], "checkpoint": item["checkpoint"]["path"], "checkpoint_sha256": item["checkpoint"]["sha256"], "feature": item["feature"]["path"], "feature_sha256": item["feature"]["sha256"]} for name, item in audit["arms"].items()]
    cv_master = []
    for ds in ("LTCC", "PRCC"):
        for row in cv_summaries[ds]: cv_master.append(row)
    write_csv(TABLE_ROOT / "cv_master.csv", cv_master); write_csv(TABLE_ROOT / "transfer_master.csv", transfer_rows)
    top_univ = {ds: univ[ds][:10] for ds in ("LTCC", "PRCC")}; quad = {ds: read_csv(QUAD_ROOT / f"{ds.lower()}_quadrants.csv") for ds in ("LTCC", "PRCC")}; risk = {ds: read_csv(RISK_ROOT / f"{ds.lower()}_negative_risk_quartiles.csv") for ds in ("LTCC", "PRCC")}
    lines = ["# Phase 7G-A — Cross-Clothes Relation Utility Predictability Audit", "", f"Generated: `{utc_now()}`.", "", "## Scope and leakage safeguards", "", "Diagnostic only. No REID model training, optimizer, scheduler, GradScaler, optimizer step, or parameter update was started. Predictors use frozen Bio train features/metadata only. Uniform Relation endpoints are used exclusively to define retrospective `Delta_M`; query/gallery/test is excluded.", "", "## Exact frozen sources", "", "| ID | Dataset | Arm | Checkpoint | Checkpoint SHA256 | Feature bank | Feature SHA256 |", "|---|---|---|---|---|---|---|"]
    for row in source_rows: lines.append("| " + " | ".join(str(row[key]) for key in ("id", "dataset", "arm", "checkpoint", "checkpoint_sha256", "feature", "feature_sha256")) + " |")
    lines += ["", "## Target counts", "", "| Dataset | Pairs | Useful | Harmful | Neutral | Useful % | Harmful % | Neutral % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows: lines.append(f"| {row['dataset']} | {row['pair_count']} | {row['useful_count']} | {row['harmful_count']} | {row['neutral_count']} | {row['useful_fraction']:.6f} | {row['harmful_fraction']:.6f} | {row['neutral_fraction']:.6f} |")
    lines += ["", "Main epsilon is `1e-6`; sensitivity counts for 0, 1e-4, and 1e-3 are in `tables/epsilon_sensitivity.csv`.", "", "## Univariate predictors", "", "Top ten by oriented ROC-AUC; complete ranked files are in `univariate/`."]
    for ds in ("LTCC", "PRCC"):
        lines += ["", f"### {ds}", "", "| Feature | Spearman(pair,U) | ROC-AUC oriented | PR-AUC | Effect size | PID bootstrap 95% |", "|---|---:|---:|---:|---:|---|"]
        for row in top_univ[ds]: lines.append(f"| {row['feature']} | {float(row['spearman_pair_utility']):.4f} | {float(row['roc_auc_oriented']):.4f} | {float(row.get('pr_auc_oriented', row['pr_auc_raw_useful'])):.4f} | {float(row['effect_useful_minus_harmful']):.4f} | [{float(row['pid_bootstrap_low95']):.4f}, {float(row['pid_bootstrap_high95']):.4f}] |")
    lines += ["", "## PID-grouped CV", "", "| Dataset | Predictor | ROC-AUC mean±std | PR-AUC mean±std | Balanced accuracy mean |", "|---|---|---:|---:|---:|"]
    for ds in ("LTCC", "PRCC"):
        for row in cv_summaries[ds]: lines.append(f"| {ds} | {row['model']} | {row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f} | {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f} | {row['balanced_accuracy']:.4f} |")
    lines += ["", "## Cross-dataset transfer", "", "| Source → Target | Predictor | ROC-AUC | PR-AUC | Balanced accuracy |", "|---|---|---:|---:|---:|"]
    for row in transfer_rows: lines.append(f"| {row['source']} → {row['target']} | {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} |")
    lines += ["", "## H_pos + M_base quadrants", ""]
    for ds in ("LTCC", "PRCC"):
        lines += [f"### {ds}", "", "| Quadrant | Pairs | PIDs | Useful % | Harmful % | Mean Delta_M |", "|---|---:|---:|---:|---:|---:|"]
        for row in quad[ds]: lines.append(f"| {row['quadrant']} | {row['pair_count']} | {row['pid_count']} | {float(row['useful_fraction']):.4f} | {float(row['harmful_fraction']):.4f} | {float(row['mean_delta_M']):.6f} |")
    lines += ["", "## Negative-risk quartiles", ""]
    for ds in ("LTCC", "PRCC"):
        lines += [f"### {ds}", "", "| Quartile | Mean s_neg | Useful % | Harmful % | Mean Δpos | Mean Δneg | Mean ΔM |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in risk[ds]: lines.append(f"| {row['negative_risk_quartile']} | {float(row['baseline_s_neg']):.6f} | {float(row['useful_fraction']):.4f} | {float(row['harmful_fraction']):.4f} | {float(row['mean_delta_pos']):.6f} | {float(row['mean_delta_neg']):.6f} | {float(row['mean_delta_M']):.6f} |")
    lines += ["", "## Gradient validation", "", "| Dataset | Bottom useful norm | Top useful norm | Bottom cos(base,rel) | Top cos(base,rel) | Bottom cos(triplet,rel) | Top cos(triplet,rel) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, row in grad_summaries.items(): lines.append(f"| {ds} | {row['bottom_norm_relation']:.4f} | {row['top_norm_relation']:.4f} | {row['bottom_cos_base_relation']:.4f} | {row['top_cos_base_relation']:.4f} | {row['bottom_cos_triplet_relation']:.4f} | {row['top_cos_triplet_relation']:.4f} |")
    lines += ["", "## Q1–Q9", "", "\n".join(f"- **{key}:** {value}" for key, value in decision.items() if key.startswith("Q")), "", "## Strongest mechanistic conclusion", "", "Baseline-only headroom, margin, negative-risk, structure, and Triplet-proxy features can describe retrospective utility, but cross-dataset transfer and/or grouped stability must clear a high bar. This audit does not convert retrospective endpoint utility into an online ground-truth signal.", "", "## Decision", "", f"**{decision['promotion_decision']}**", "", "**NO NEW REID MODEL TRAINING WAS STARTED. NO UTILITY-GATED RELATION TRAINING WAS STARTED.**", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8"); return decision


def epsilon_table(data_by_name: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name, data in data_by_name.items():
        for epsilon in EPS_SENSITIVITY:
            label = data[f"label_eps_{epsilon:g}"]; rows.append({"dataset": name, "epsilon": epsilon, "useful_count": int((label == 1).sum()), "harmful_count": int((label == 0).sum()), "neutral_count": int((label < 0).sum()), "useful_fraction": float((label == 1).mean()), "harmful_fraction": float((label == 0).mean()), "neutral_fraction": float((label < 0).mean())})
    write_csv(TABLE_ROOT / "epsilon_sensitivity.csv", rows)


def offline_run() -> None:
    ensure_dirs(); audit = json.loads(SOURCE_JSON.read_text()) if SOURCE_JSON.is_file() else source_audit(); data_by_name = {}; univ = {}; cv_summaries = {}; transfer_rows = []; oof_by_name = {}
    for name, base_key, relation_key in (("LTCC", "ltcc_b0_seed1234", "ltcc_b1_seed1234"), ("PRCC", "prcc_c0_seed1234", "prcc_c1_seed1234")):
        data = load_pair_feature_artifact(name)
        if data is None:
            base_bank = load_feature_bank(base_key); pairs, utility = load_pairs(name, base_bank); data = build_pair_features(name, base_bank, pairs, utility); del base_bank
        data_by_name[name] = data; univ[name] = univariate_analysis(name, data); _fold, cv_summaries[name], oof_by_name[name] = cv_analysis(name, data); bootstrap_oof(name, data, oof_by_name[name], ["full_interpretable_logistic", "H_pos + M_base"]); quadrant_analysis(name, data); negative_risk_analysis(name, data)
    epsilon_table(data_by_name); feature_shift(data_by_name["LTCC"], data_by_name["PRCC"]); transfer_rows.extend(cross_transfer("LTCC", "PRCC", data_by_name["LTCC"], data_by_name["PRCC"])); transfer_rows.extend(cross_transfer("PRCC", "LTCC", data_by_name["PRCC"], data_by_name["LTCC"])); write_csv(TABLE_ROOT / "transfer_master.csv", transfer_rows); plot_diagnostics(data_by_name)
    best = {}
    for name in ("LTCC", "PRCC"):
        best[name] = max(cv_summaries[name], key=lambda row: row["roc_auc"])["model"]; data = data_by_name[name]; y = data["label_eps_1e-06"]; keep = y >= 0; score, _prob, _fit = model_fit_score(best[name], data["X"][keep], y[keep].astype(np.int8), data["X"]); np.save(GRAD_ROOT / f"{name.lower()}_best_predictor_score.npy", score.astype(np.float32))
    write_json(GRAD_ROOT / "best_predictor_selection.json", {"selection_rule": "highest mean PID-grouped CV ROC-AUC among predeclared simple rules/logistic/tree; no target-dataset fitting", "best_model_by_dataset": best}); write_json(OUT_ROOT / "offline_state.json", {"completed": True, "best_predictor_by_dataset": best, "training_started": False, "optimizer_step_called": False})
    print(json.dumps({"offline_completed": True, "best_predictor_by_dataset": best, "pair_counts": {name: len(data["utility"]) for name, data in data_by_name.items()}, "next": "run --mode gradient"}, indent=2, sort_keys=True))


def gradient_run() -> None:
    state = json.loads((OUT_ROOT / "offline_state.json").read_text()); summaries = {}
    for name in ("LTCC", "PRCC"):
        with np.load(PAIR_ROOT / f"{name.lower()}_baseline_pair_features.npz", allow_pickle=False) as archive: data = {key: np.asarray(archive[key]) for key in archive.files}
        score = np.asarray(np.load(GRAD_ROOT / f"{name.lower()}_best_predictor_score.npy"), dtype=np.float32); summaries[name] = gradient_validation_one(name, data, score)
    write_json(GRAD_ROOT / "gradient_validation_summary.json", summaries); write_json(OUT_ROOT / "gradient_state.json", {"completed": True, "summaries": summaries, "training_started": False, "optimizer_step_called": False})
    print(json.dumps({"gradient_completed": True, "datasets": list(summaries), "next": "run --mode finalize"}, indent=2, sort_keys=True))


def finalize_run() -> None:
    audit = json.loads(SOURCE_JSON.read_text()); data_by_name = {}; univ = {}; cv_summaries = {}; transfer_rows = []
    for name in ("LTCC", "PRCC"):
        with np.load(PAIR_ROOT / f"{name.lower()}_baseline_pair_features.npz", allow_pickle=False) as archive:
            data_by_name[name] = {key: np.asarray(archive[key]) for key in archive.files}
        univ[name] = [{key: (_float_or_nan(value) if key not in ("dataset", "feature") else value) for key, value in row.items()} for row in read_csv(UNIV_ROOT / f"{name.lower()}_univariate.csv")]; cv_summaries[name] = [{key: (_float_or_nan(value) if key not in ("dataset", "model") else value) for key, value in row.items()} for row in read_csv(CV_ROOT / f"{name.lower()}_summary.csv")]
        transfer_rows.extend([{key: (_float_or_nan(value) if key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "brier") else value) for key, value in row.items()} for row in read_csv(TRANSFER_ROOT / f"{name.lower()}_to_{('prcc' if name == 'LTCC' else 'ltcc')}.csv")])
    grad_state = json.loads((OUT_ROOT / "gradient_state.json").read_text()) if (OUT_ROOT / "gradient_state.json").is_file() else None; grad_summaries = grad_state["summaries"] if grad_state else None; decision = finalize_report(audit, data_by_name, univ, cv_summaries, transfer_rows, grad_summaries)
    manifest = {"phase": "phase7g_a_relation_utility_predictability", "created_utc": utc_now(), "source_audit": str(SOURCE_JSON), "source_audit_sha256": sha256_file(SOURCE_JSON), "train_only": True, "endpoint_used_only_for_label": True, "query_gallery_used": False, "predictor_features_baseline_only": True, "optimizer_created": False, "optimizer_step_called": False, "training_started": False, "utility_gated_relation_training_started": False, "gradient_validation_ppu_only": bool(grad_state), "decision": decision, "report": str(REPORT_PATH), "previous_artifacts_overwritten": False, "feature_names": FEATURE_NAMES, "models": MODEL_NAMES, "main_epsilon": EPS, "epsilon_sensitivity": list(EPS_SENSITIVITY)}; write_json(MANIFEST_PATH, manifest); print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decision, "training_started": False, "utility_gated_relation_training_started": False}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("audit", "offline", "gradient", "finalize"), default="offline"); args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); ensure_dirs()
    if args.mode == "audit": source_audit(); print(json.dumps({"passed": True, "source": str(SOURCE_JSON), "report": str(SOURCE_REPORT)}, indent=2)); return
    if not SOURCE_JSON.is_file(): source_audit()
    if args.mode == "offline": offline_run()
    elif args.mode == "gradient": gradient_run()
    else: finalize_run()


if __name__ == "__main__": main()
