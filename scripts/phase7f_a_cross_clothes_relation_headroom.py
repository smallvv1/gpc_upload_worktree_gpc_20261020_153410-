#!/usr/local/bin/python3
"""Phase 7F-A: frozen cross-clothes relation headroom audit.

This is a train-only diagnostic.  It reuses frozen feature banks from Phase
7B/7D and probes the frozen Bio models on CUDA without constructing an
optimizer, scheduler, GradScaler, or training loop.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from loss.triplet_loss import euclidean_dist  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase7e_relation_triplet_complementarity as p7e  # noqa: E402


PHASE = "phase7f_a_cross_clothes_relation_headroom"
SEED = 1234
TOLERANCE = 1e-6
BOOTSTRAPS = 1000
# Four probes per headroom group are sufficient for this diagnostic while
# keeping the PPU-only frozen-model audit bounded.  The groups are sampled
# from the same frozen loader/batch construction; they need not co-occur in
# every batch because the frozen LTCC sampler rarely puts both quartiles in
# one mini-batch.
GRADIENT_PROBES = 4

OUT_ROOT = ROOT / "outputs/new_method_track/phase7f_a_relation_headroom_audit"
SOURCE_ROOT = OUT_ROOT / "source_audit"
PAIR_ROOT = OUT_ROOT / "pair_headroom"
QUARTILE_ROOT = OUT_ROOT / "quartile_analysis"
CURVE_ROOT = OUT_ROOT / "saturation_curve"
CORR_ROOT = OUT_ROOT / "correlations"
UTILITY_ROOT = OUT_ROOT / "useful_harmful_analysis"
GRAD_ROOT = OUT_ROOT / "gradient_analysis"
CROSS_ROOT = OUT_ROOT / "cross_dataset"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7F_A_MANIFEST.json"
SOURCE_JSON = SOURCE_ROOT / "source_audit.json"
SOURCE_REPORT = ROOT / "reports/PHASE_7F_A_SOURCE_AUDIT.md"
REPORT_PATH = ROOT / "reports/PHASE_7F_A_CROSS_CLOTHES_RELATION_HEADROOM_AUDIT.md"
PRCC_PAIR_CACHE = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/pair_margin_analysis/pair_metrics.npz"
PRCC_PAIR_CACHE_MANIFEST = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/pair_margin_analysis/pair_metrics_manifest.json"
LTCC_PAIR_CACHE = OUT_ROOT / "pair_headroom/ltcc_pair_headroom.npz"

SOURCES = {
    "ltcc_b0_seed1234": {
        "dataset": "LTCC", "arm": "B0 Bio", "checkpoint": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b0_seed1234_train_features.npz",
        "phase": "Phase 6C",
    },
    "ltcc_b1_seed1234": {
        "dataset": "LTCC", "arm": "B1 Uniform Relation", "checkpoint": ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B1_full/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b1_seed1234_train_features.npz",
        "phase": "Phase 6C",
    },
    "prcc_c0_seed1234": {
        "dataset": "PRCC", "arm": "C0 Bio", "checkpoint": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C0_E60.npy",
        "metadata": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/train_metadata.csv",
        "phase": "Phase 7C",
    },
    "prcc_c1_seed1234": {
        "dataset": "PRCC", "arm": "C1 Uniform Relation", "checkpoint": ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C1_image_uniform_relation/model_final.pth",
        "feature": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C1_E60.npy",
        "metadata": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/train_metadata.csv",
        "phase": "Phase 7C",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if torch.is_tensor(value): return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def sha256_state(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict): state = state["model"]
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode()); digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SOURCE_ROOT, PAIR_ROOT, QUARTILE_ROOT, CURVE_ROOT, CORR_ROOT, UTILITY_ROOT, GRAD_ROOT, CROSS_ROOT, TABLE_ROOT, PLOT_ROOT): path.mkdir(parents=True, exist_ok=True)


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    arms = {}
    for name, spec in SOURCES.items():
        required = [spec["checkpoint"], spec["feature"]]
        if "metadata" in spec: required.append(spec["metadata"])
        if not all(path.is_file() for path in required): raise RuntimeError(f"missing frozen source for {name}: {[str(p) for p in required if not p.is_file()]}")
        arm = dict(spec)
        arm.update({"checkpoint": file_record(spec["checkpoint"]), "feature": file_record(spec["feature"]), "model_tensor_state_hash": sha256_state(spec["checkpoint"])})
        if "metadata" in spec: arm["metadata"] = file_record(spec["metadata"])
        arms[name] = arm
    cache_sources = {
        "ltcc_pair_headroom_cache": file_record(LTCC_PAIR_CACHE),
        "prcc_phase7d_pair_metrics_cache": file_record(PRCC_PAIR_CACHE),
        "prcc_phase7d_pair_metrics_manifest": file_record(PRCC_PAIR_CACHE_MANIFEST),
    }
    audit = {"phase": PHASE, "created_utc": utc_now(), "source_verification_pass": True, "seed": SEED, "arms": arms, "pair_cache_sources": cache_sources,
             "feature_source": "normalized train feature bank from frozen feat[0] extraction; pair construction uses train PID/clothing labels only",
             "training_started": False, "optimizer_created": False, "optimizer_step_called": False, "query_gallery_used_for_thresholds": False,
             "relation_thresholds_trained": False, "note": "No threshold or weighting function is selected by this audit."}
    write_json(SOURCE_JSON, audit)
    lines = ["# Phase 7F-A source audit", "", f"Generated: `{audit['created_utc']}`.", "", "Decision: **PASS**. Frozen seed1234 Bio and Uniform Relation sources are present and hashed.", "", "| ID | Dataset | Arm | Checkpoint | Feature bank |", "|---|---|---|---|---|"]
    for name, item in arms.items(): lines.append(f"| {name} | {item['dataset']} | {item['arm']} | `{item['checkpoint']['path']}` | `{item['feature']['path']}` |")
    lines += ["", "Pair-cache sources used for the frozen train-only endpoint deltas:", "", f"- LTCC cache: `{LTCC_PAIR_CACHE}` (SHA256 `{cache_sources['ltcc_pair_headroom_cache']['sha256']}`).", f"- PRCC Phase 7D cache: `{PRCC_PAIR_CACHE}` (SHA256 `{cache_sources['prcc_phase7d_pair_metrics_cache']['sha256']}`); manifest: `{PRCC_PAIR_CACHE_MANIFEST}`.", "", "All feature geometry and thresholds use `rgb/train` only. Query/gallery/test is excluded. This phase does not choose a training threshold, headroom function, sampler, loss, or lambda.", ""]
    SOURCE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return audit


def load_feature_bank(spec: dict[str, Any], dataset: Any) -> np.ndarray:
    path = spec["feature"]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            paths = archive["paths"].astype(str).tolist(); pids = archive["pids"].astype(int); clothes = archive["clothes"].astype(int)
    else:
        features = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        metadata = read_csv(spec["metadata"]); paths = [row["image_path"] for row in metadata]; pids = np.asarray([int(row["pid"]) for row in metadata]); clothes = np.asarray([int(row["clothes_id"]) for row in metadata])
    if len(features) != len(dataset.train): raise RuntimeError(f"feature count mismatch for {path}")
    actual_paths = [str(item["image_path"]) for item in dataset.train]
    if paths != actual_paths or not np.array_equal(pids, np.asarray([int(item["pid"]) for item in dataset.train])) or not np.array_equal(clothes, np.asarray([int(item["clothes_id"]) for item in dataset.train])):
        raise RuntimeError(f"feature bank order/labels do not match dataset train order: {path}")
    return features


def pair_metrics(dataset: Any, baseline: np.ndarray, endpoint: np.ndarray) -> dict[str, Any]:
    pairs = p7e.build_pair_arrays(dataset)
    pids = pairs["pids"]; clothes = pairs["clothes"]
    def normalized(value: np.ndarray) -> np.ndarray:
        value = value.astype(np.float32); return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    def neg_centers(z: np.ndarray) -> np.ndarray:
        labels = sorted(set(pids.tolist())); centers = []; lookup = {}
        for index, pid in enumerate(labels):
            center = z[pids == pid].mean(axis=0); center /= max(float(np.linalg.norm(center)), 1e-12); centers.append(center); lookup[int(pid)] = index
        sims = z @ np.asarray(centers, dtype=np.float32).T; result = np.empty(len(z), dtype=np.float32)
        for index, pid in enumerate(pids.tolist()):
            values = sims[index].copy(); values[lookup[int(pid)]] = -1e4; result[index] = values.max()
        return result
    z0 = normalized(baseline); z1 = normalized(endpoint)
    neg0 = neg_centers(z0); neg1 = neg_centers(z1)
    pos0 = np.sum(z0[pairs["a"]] * z0[pairs["b"]], axis=1); pos1 = np.sum(z1[pairs["a"]] * z1[pairs["b"]], axis=1)
    sneg0 = np.maximum(neg0[pairs["a"]], neg0[pairs["b"]]); sneg1 = np.maximum(neg1[pairs["a"]], neg1[pairs["b"]])
    margin0 = pos0 - sneg0; margin1 = pos1 - sneg1; dpos = pos1 - pos0; dneg = sneg1 - sneg0; dmargin = margin1 - margin0
    state_keys = list(zip(pairs["pid"].tolist(), np.minimum(pairs["cloth_a"], pairs["cloth_b"]).tolist(), np.maximum(pairs["cloth_a"], pairs["cloth_b"]).tolist()))
    state_counts = Counter(state_keys)
    multiplicity = np.fromiter((state_counts[key] for key in state_keys), dtype=np.int32, count=len(state_keys))
    num_clothes = {int(pid): len(set(clothes[pids == pid].tolist())) for pid in set(pids.tolist())}
    return {"pairs": pairs, "pos0": pos0, "pos1": pos1, "neg0": sneg0, "neg1": sneg1, "margin0": margin0, "margin1": margin1, "dpos": dpos, "dneg": dneg, "dmargin": dmargin, "headroom": 1.0 - pos0, "multiplicity": multiplicity, "pids_per_pair": pairs["pid"], "num_clothes_by_pid": num_clothes}


def cached_prcc_pair_metrics(dataset: Any) -> dict[str, Any]:
    """Reuse the frozen Phase 7D train-only C0/C1 pair audit."""
    if not PRCC_PAIR_CACHE.is_file() or not PRCC_PAIR_CACHE_MANIFEST.is_file():
        raise RuntimeError("frozen PRCC pair cache is missing")
    import json
    manifest = json.loads(PRCC_PAIR_CACHE_MANIFEST.read_text())
    if manifest.get("query_gallery_used") or not manifest.get("train_only") or int(manifest.get("pairs", 0)) != 501040:
        raise RuntimeError("PRCC pair cache provenance is not train-only/complete")
    with np.load(PRCC_PAIR_CACHE, allow_pickle=False) as archive:
        cached = {key: np.asarray(archive[key]) for key in archive.files}
    full_pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int32); full_clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int32)
    if np.any(full_pids[cached["pair_a"]] != cached["pid"]) or np.any(full_pids[cached["pair_b"]] != cached["pid"]) or np.any(full_clothes[cached["pair_a"]] == full_clothes[cached["pair_b"]]):
        raise RuntimeError("PRCC cached pair labels do not match current train dataset")
    pairs = {"a": cached["pair_a"].astype(np.int32), "b": cached["pair_b"].astype(np.int32), "pid": cached["pid"].astype(np.int32), "cloth_a": cached["clothes_a"].astype(np.int32), "cloth_b": cached["clothes_b"].astype(np.int32), "pids": full_pids, "clothes": full_clothes}
    relation_counts = np.bincount(cached["state_relation_id"].astype(np.int64), minlength=int(cached["state_relation_id"].max()) + 1)
    multiplicity = relation_counts[cached["state_relation_id"].astype(np.int64)].astype(np.int32)
    pos0 = cached["C0_pos"].astype(np.float32); pos1 = cached["C1_pos"].astype(np.float32); neg0 = cached["C0_neg_pair"].astype(np.float32); neg1 = cached["C1_neg_pair"].astype(np.float32); margin0 = cached["C0_margin"].astype(np.float32); margin1 = cached["C1_margin"].astype(np.float32)
    num_clothes = {int(pid): len(set(full_clothes[full_pids == pid].tolist())) for pid in set(full_pids.tolist())}
    return {"pairs": pairs, "pos0": pos0, "pos1": pos1, "neg0": neg0, "neg1": neg1, "margin0": margin0, "margin1": margin1, "dpos": pos1 - pos0, "dneg": neg1 - neg0, "dmargin": margin1 - margin0, "headroom": 1.0 - pos0, "multiplicity": multiplicity, "pids_per_pair": pairs["pid"], "num_clothes_by_pid": num_clothes, "pair_cache": str(PRCC_PAIR_CACHE)}


def cached_ltcc_pair_metrics(dataset: Any) -> dict[str, Any]:
    if not LTCC_PAIR_CACHE.is_file():
        raise RuntimeError("LTCC pair cache is missing")
    with np.load(LTCC_PAIR_CACHE, allow_pickle=False) as archive:
        cached = {key: np.asarray(archive[key]) for key in archive.files}
    full_pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int32); full_clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int32)
    if len(cached["pos0"]) != len(cached["pid"]): raise RuntimeError("LTCC cached pair count mismatch")
    pairs = {"pid": cached["pid"].astype(np.int32), "pids": full_pids, "clothes": full_clothes}
    multiplicity = cached["multiplicity"].astype(np.int32)
    pos0 = cached["pos0"].astype(np.float32); pos1 = cached["pos1"].astype(np.float32); neg0 = cached["neg0"].astype(np.float32); neg1 = cached["neg1"].astype(np.float32); margin0 = cached["margin0"].astype(np.float32); margin1 = cached["margin1"].astype(np.float32)
    num_clothes = {int(pid): len(set(full_clothes[full_pids == pid].tolist())) for pid in set(full_pids.tolist())}
    return {"pairs": pairs, "pos0": pos0, "pos1": pos1, "neg0": neg0, "neg1": neg1, "margin0": margin0, "margin1": margin1, "dpos": pos1 - pos0, "dneg": neg1 - neg0, "dmargin": margin1 - margin0, "headroom": 1.0 - pos0, "multiplicity": multiplicity, "pids_per_pair": pairs["pid"], "num_clothes_by_pid": num_clothes, "pair_cache": str(LTCC_PAIR_CACHE)}


def write_pair_csv(path: Path, dataset_name: str, metrics: dict[str, Any]) -> None:
    """Stream the large pair table so PRCC does not require a 500k-dict list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "pair_index", "pid", "s_pos0", "s_pos1", "headroom", "s_neg0", "s_neg1", "margin0", "margin1", "delta_pos", "delta_neg", "delta_margin", "pair_multiplicity"]
    pairs = metrics["pairs"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(metrics["pos0"])):
            writer.writerow({"dataset": dataset_name, "pair_index": i, "pid": int(metrics["pids_per_pair"][i]), "s_pos0": float(metrics["pos0"][i]), "s_pos1": float(metrics["pos1"][i]), "headroom": float(metrics["headroom"][i]), "s_neg0": float(metrics["neg0"][i]), "s_neg1": float(metrics["neg1"][i]), "margin0": float(metrics["margin0"][i]), "margin1": float(metrics["margin1"][i]), "delta_pos": float(metrics["dpos"][i]), "delta_neg": float(metrics["dneg"][i]), "delta_margin": float(metrics["dmargin"][i]), "pair_multiplicity": int(metrics["multiplicity"][i])})


def stat(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0: return {"count": 0}
    return {"count": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)), "median": float(np.median(values)), "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)), "max": float(np.max(values))}


def pid_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = metrics["pairs"]; rows = []
    for pid in sorted(set(pairs["pid"].tolist())):
        mask = pairs["pid"] == pid
        rows.append({"pid": int(pid), "pair_count": int(mask.sum()), "num_clothes": int(metrics["num_clothes_by_pid"][int(pid)]), "mean_pos0": float(metrics["pos0"][mask].mean()), "mean_neg0": float(metrics["neg0"][mask].mean()), "mean_margin0": float(metrics["margin0"][mask].mean()), "mean_headroom": float(metrics["headroom"][mask].mean()), "mean_dpos": float(metrics["dpos"][mask].mean()), "mean_dneg": float(metrics["dneg"][mask].mean()), "mean_dmargin": float(metrics["dmargin"][mask].mean()), "mean_multiplicity": float(metrics["multiplicity"][mask].mean())})
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64); order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), dtype=np.float64); sorted_values = values[order]; start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0; start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12: return None
    rx = rankdata(x); ry = rankdata(y); return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_pid(pid_data: list[dict[str, Any]], xkey: str, ykey: str) -> dict[str, Any]:
    x = np.asarray([row[xkey] for row in pid_data], dtype=np.float64); y = np.asarray([row[ykey] for row in pid_data], dtype=np.float64); point = spearman(x, y); rng = np.random.default_rng(SEED + 77); samples = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(x), size=len(x)); value = spearman(x[indices], y[indices])
        if value is not None: samples.append(value)
    return {"pair_spearman": point, "pid_mean_spearman": point, "bootstrap_n": len(samples), "bootstrap_low95": float(np.percentile(samples, 2.5)) if samples else None, "bootstrap_high95": float(np.percentile(samples, 97.5)) if samples else None}


def quartile_analysis(name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    pos = metrics["pos0"]; labels = np.minimum(4, np.digitize(pos, np.quantile(pos, [0.25, 0.5, 0.75]), right=True) + 1); rows = []
    for q in range(1, 5):
        mask = labels == q; pids = set(metrics["pids_per_pair"][mask].tolist())
        rows.append({"dataset": name, "quartile": f"Q{q}", "count": int(mask.sum()), "pid_count": len(pids), "baseline_s_pos": float(metrics["pos0"][mask].mean()), "baseline_s_neg": float(metrics["neg0"][mask].mean()), "baseline_margin": float(metrics["margin0"][mask].mean()), "delta_pos": float(metrics["dpos"][mask].mean()), "delta_neg": float(metrics["dneg"][mask].mean()), "delta_margin": float(metrics["dmargin"][mask].mean()), "endpoint_s_pos": float(metrics["pos1"][mask].mean()), "endpoint_s_neg": float(metrics["neg1"][mask].mean()), "endpoint_margin": float(metrics["margin1"][mask].mean())})
    return rows


def saturation_bins(name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    edges = [-np.inf, 0.50, 0.60, 0.70, 0.80, 0.90, np.inf]; labels = ["<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", ">=0.90"]; bins = np.digitize(metrics["pos0"], edges[1:-1], right=False); rows = []
    for index, label in enumerate(labels):
        mask = bins == index; rows.append({"dataset": name, "bin": label, "count": int(mask.sum()), "pid_count": len(set(metrics["pids_per_pair"][mask].tolist())), "fraction": float(mask.mean()), "baseline_s_pos": float(metrics["pos0"][mask].mean()) if mask.any() else None, "delta_pos": float(metrics["dpos"][mask].mean()) if mask.any() else None, "delta_neg": float(metrics["dneg"][mask].mean()) if mask.any() else None, "delta_margin": float(metrics["dmargin"][mask].mean()) if mask.any() else None})
    return rows


def threshold_rows(name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        mask = metrics["pos0"] >= threshold; rows.append({"dataset": name, "threshold": threshold, "count": int(mask.sum()), "fraction": float(mask.mean()), "pid_count": len(set(metrics["pids_per_pair"][mask].tolist())), "delta_pos": float(metrics["dpos"][mask].mean()) if mask.any() else None, "delta_neg": float(metrics["dneg"][mask].mean()) if mask.any() else None, "delta_margin": float(metrics["dmargin"][mask].mean()) if mask.any() else None})
    return rows


def useful_harmful(name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, tolerance in (("zero_tolerance", 0.0), ("tolerance_aware", TOLERANCE)):
        labels = np.where(metrics["dmargin"] > tolerance, "USEFUL", np.where(metrics["dmargin"] < -tolerance, "HARMFUL", "NEUTRAL"))
        for label in ("USEFUL", "NEUTRAL", "HARMFUL"):
            mask = labels == label
            rows.append({"dataset": name, "mode": mode, "label": label, "count": int(mask.sum()), "fraction": float(mask.mean()), "pid_count": len(set(metrics["pids_per_pair"][mask].tolist())), "baseline_s_pos": float(metrics["pos0"][mask].mean()) if mask.any() else None, "baseline_s_neg": float(metrics["neg0"][mask].mean()) if mask.any() else None, "baseline_margin": float(metrics["margin0"][mask].mean()) if mask.any() else None, "mean_headroom": float(metrics["headroom"][mask].mean()) if mask.any() else None, "num_clothes_mean": float(np.mean([metrics["num_clothes_by_pid"][int(pid)] for pid in set(metrics["pids_per_pair"][mask].tolist())])) if mask.any() else None, "pair_multiplicity_mean": float(metrics["multiplicity"][mask].mean()) if mask.any() else None, "delta_pos": float(metrics["dpos"][mask].mean()) if mask.any() else None, "delta_neg": float(metrics["dneg"][mask].mean()) if mask.any() else None, "delta_margin": float(metrics["dmargin"][mask].mean()) if mask.any() else None})
    return rows


def roc_auc(score: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool); score = np.asarray(score, dtype=np.float64); pos = score[labels]; neg = score[~labels]
    if len(pos) == 0 or len(neg) == 0: return None
    comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comparisons / (len(pos) * len(neg)))


def predictive_rows(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    pos = metrics["dmargin"] > TOLERANCE; neg = metrics["dmargin"] < -TOLERANCE; keep = pos | neg
    labels = pos[keep]; score = metrics["headroom"][keep]; q = np.minimum(4, np.digitize(metrics["pos0"], np.quantile(metrics["pos0"], [0.25, 0.5, 0.75]), right=True) + 1)
    q1 = metrics["dmargin"][q == 1]; q4 = metrics["dmargin"][q == 4]
    return {"dataset": name, "positive_count": int(pos.sum()), "negative_count": int(neg.sum()), "neutral_count": int((~pos & ~neg).sum()), "roc_auc_headroom_for_positive_delta_margin": roc_auc(score, labels), "q1_delta_margin_mean": float(q1.mean()), "q4_delta_margin_mean": float(q4.mean()), "q1_minus_q4": float(q1.mean() - q4.mean())}


def gradient_probe(name: str, metrics: dict[str, Any], cfg: Any, dataset: Any, transform: Any, bank: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    checkpoint = SOURCES["ltcc_b0_seed1234" if name == "LTCC" else "prcc_c0_seed1234"]["checkpoint"]
    model = p7e.model_from_checkpoint(name.lower(), cfg, dataset, checkpoint, device); loader = p7e.make_loader(name.lower(), dataset, cfg, transform, epoch=60)
    loss_fn, _ = base.make_loss(cfg, num_classes=dataset.num_train_pids); named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]; names = [n for n, _ in named]; params = [p for _, p in named]; groups = p7e.parameter_groups(model); selected = groups["late_eva"]
    pair_lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(metrics["pairs"]["a"], metrics["pairs"]["b"]))}; low_cut = float(np.percentile(metrics["pos0"], 25)); high_cut = float(np.percentile(metrics["pos0"], 75)); rows = []; low_rows = 0; high_rows = 0; joint_batches = 0; low_candidate_batches = 0; high_candidate_batches = 0; inspected = 0
    for batch in loader:
        inspected += 1; all_edges, rel_edges = p7e.local_pairs(batch, bank); indices = [bank["path_to_index"][str(path)] for path in batch["image_path"]]
        low_edges = []; high_edges = []
        for i, j in rel_edges:
            key = (min(indices[i], indices[j]), max(indices[i], indices[j])); value = metrics["pos0"][pair_lookup[key]]
            if value <= low_cut: low_edges.append((i, j))
            if value >= high_cut: high_edges.append((i, j))
        if low_edges: low_candidate_batches += 1
        if high_edges: high_candidate_batches += 1
        if low_edges and high_edges: joint_batches += 1
        if (low_rows >= GRADIENT_PROBES and high_rows >= GRADIENT_PROBES) or (not low_edges and not high_edges): continue
        images = batch["image"].to(device); pids = batch["pid"].to(device); camids = batch["camid"].to(device); clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device)
        model.zero_grad(set_to_none=True); score, feat = model(images, cam_label=camids); writer = base.MetricCapture(); _total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=inspected); comp = loss_fn.last_components; base_loss = comp["ce"] + comp["triplet"] + comp["bio"]; rep = F.normalize(feat[0].float(), dim=-1)
        low_loss = torch.stack([1.0 - torch.sum(rep[i] * rep[j]) for i, j in low_edges]).mean() if low_edges else None; high_loss = torch.stack([1.0 - torch.sum(rep[i] * rep[j]) for i, j in high_edges]).mean() if high_edges else None
        # Keep the base and Triplet graphs until the available relation group
        # has been differentiated.  Low/high groups may come from different
        # frozen sampler batches; this is recorded explicitly below.
        base_g = p7e.component_grads(base_loss, params, True); tri_g = p7e.component_grads(comp["triplet"], params, True)
        low_g = p7e.component_grads(low_loss, params, bool(high_edges)) if low_loss is not None and low_rows < GRADIENT_PROBES else None
        high_g = p7e.component_grads(high_loss, params, False) if high_loss is not None and high_rows < GRADIENT_PROBES else None
        if low_g is not None:
            rows.append({"dataset": name, "batch_index": inspected, "group": "low_headroom", "relation_count": len(low_edges), "joint_low_high_batch": bool(high_edges), "norm": p7e.group_norm(low_g, names, selected), "cos_base": p7e.alignment(base_g, low_g, names, selected), "cos_triplet": p7e.alignment(tri_g, low_g, names, selected), "optimizer_step_called": False, "autocast_enabled": False}); low_rows += 1
        if high_g is not None:
            rows.append({"dataset": name, "batch_index": inspected, "group": "high_similarity", "relation_count": len(high_edges), "joint_low_high_batch": bool(low_edges), "norm": p7e.group_norm(high_g, names, selected), "cos_base": p7e.alignment(base_g, high_g, names, selected), "cos_triplet": p7e.alignment(tri_g, high_g, names, selected), "optimizer_step_called": False, "autocast_enabled": False}); high_rows += 1
        model.zero_grad(set_to_none=True); del score, feat, _total, base_loss, low_loss, high_loss, base_g, tri_g, low_g, high_g
        if low_rows >= GRADIENT_PROBES and high_rows >= GRADIENT_PROBES: break
    if low_rows < GRADIENT_PROBES or high_rows < GRADIENT_PROBES: raise RuntimeError(f"{name} yielded low={low_rows}, high={high_rows} gradient probes after {inspected} batches")
    write_csv(GRAD_ROOT / f"{name.lower()}_low_high_gradient_rows.csv", rows)
    summary = {"dataset": name, "probe_count": len(rows), "low_probe_count": low_rows, "high_probe_count": high_rows, "joint_low_high_batch_count": joint_batches, "low_candidate_batch_count": low_candidate_batches, "high_candidate_batch_count": high_candidate_batches, "inspected_batches": inspected, "probe_design": "same frozen loader/batch construction; low and high groups are matched by dataset/epoch/config but may be in separate batches"}
    for group, prefix in (("low_headroom", "low"), ("high_similarity", "high")):
        group_rows = [row for row in rows if row["group"] == group]
        for source, key in (("norm", f"{prefix}_norm"), ("cos_base", f"{prefix}_cos_base"), ("cos_triplet", f"{prefix}_cos_triplet")):
            values = np.asarray([float(row[source]) for row in group_rows]); summary[key] = float(values.mean()) if len(values) else None; summary[f"{key}_std"] = float(values.std()) if len(values) else None
    write_json(GRAD_ROOT / f"{name.lower()}_low_high_gradient_summary.json", summary); del loader, loss_fn, model; gc.collect(); torch.cuda.empty_cache(); return rows


def finalize_existing() -> None:
    """Finalize a completed pair/gradient audit without rerunning PPU probes."""
    import json
    ensure_dirs()
    if not SOURCE_JSON.is_file():
        source_audit()
    audit = json.loads(SOURCE_JSON.read_text())
    shared = {}; metrics_by_dataset = {}
    for name in ("LTCC", "PRCC"):
        cfg, dataset, transform = p7e.cfg_dataset(name.lower()); shared[name] = (cfg, dataset, transform)
        metrics = cached_ltcc_pair_metrics(dataset) if name == "LTCC" else cached_prcc_pair_metrics(dataset)
        metrics_by_dataset[name] = metrics
        # These files are cheap to regenerate from the frozen pair caches and
        # make finalize mode self-contained after an interrupted PPU pass.
        write_csv(QUARTILE_ROOT / f"{name.lower()}_quartiles.csv", quartile_analysis(name, metrics))
        write_csv(CURVE_ROOT / f"{name.lower()}_saturation_bins.csv", saturation_bins(name, metrics))
        write_csv(UTILITY_ROOT / f"{name.lower()}_useful_harmful.csv", useful_harmful(name, metrics))
        write_json(UTILITY_ROOT / f"{name.lower()}_predictive_utility.json", predictive_rows(name, metrics))
        write_csv(UTILITY_ROOT / f"{name.lower()}_pid_geometry.csv", pid_rows(metrics))
        write_csv(CURVE_ROOT / f"{name.lower()}_thresholds.csv", threshold_rows(name, metrics))
        pids = pid_rows(metrics); corr_rows = []
        for label, xkey, ykey, xpair, ypair in (("s_pos0_vs_delta_pos", "mean_pos0", "mean_dpos", metrics["pos0"], metrics["dpos"]), ("s_pos0_vs_delta_margin", "mean_pos0", "mean_dmargin", metrics["pos0"], metrics["dmargin"]), ("headroom_vs_delta_pos", "mean_headroom", "mean_dpos", metrics["headroom"], metrics["dpos"]), ("headroom_vs_delta_margin", "mean_headroom", "mean_dmargin", metrics["headroom"], metrics["dmargin"])):
            boot = bootstrap_pid(pids, xkey, ykey); boot["dataset"] = name; boot["relationship"] = label; boot["pair_spearman"] = spearman(xpair, ypair); corr_rows.append(boot)
        write_csv(CORR_ROOT / f"{name.lower()}_correlations.csv", corr_rows)
    grad_summary = [json.loads((GRAD_ROOT / f"{name.lower()}_low_high_gradient_summary.json").read_text()) for name in ("LTCC", "PRCC")]
    write_csv(GRAD_ROOT / "low_high_gradient_summary.csv", grad_summary)
    cross_rows = []
    for name in ("LTCC", "PRCC"):
        m = metrics_by_dataset[name]; q = quartile_analysis(name, m); corr = read_csv(CORR_ROOT / f"{name.lower()}_correlations.csv"); g = next(row for row in grad_summary if row["dataset"] == name); thresholds = {float(row["threshold"]): row for row in read_csv(CURVE_ROOT / f"{name.lower()}_thresholds.csv")}; cr = next(row for row in corr if row["relationship"] == "headroom_vs_delta_margin")
        cross_rows.append({"dataset": name, "mean_s_pos_baseline": float(m["pos0"].mean()), "median_s_pos_baseline": float(np.median(m["pos0"])), "mean_headroom": float(m["headroom"].mean()), "frac_s_pos_ge_070": float(thresholds[0.70]["fraction"]), "frac_s_pos_ge_075": float(thresholds[0.75]["fraction"]), "frac_s_pos_ge_080": float(thresholds[0.80]["fraction"]), "frac_s_pos_ge_085": float(thresholds[0.85]["fraction"]), "frac_s_pos_ge_090": float(thresholds[0.90]["fraction"]), "Q1_delta_M": float(q[0]["delta_margin"]), "Q2_delta_M": float(q[1]["delta_margin"]), "Q3_delta_M": float(q[2]["delta_margin"]), "Q4_delta_M": float(q[3]["delta_margin"]), "Spearman_headroom_delta_M": float(cr["pair_spearman"]) if cr["pair_spearman"] not in (None, "", "None") else None, "bootstrap_low95": float(cr["bootstrap_low95"]) if cr["bootstrap_low95"] not in (None, "", "None") else None, "bootstrap_high95": float(cr["bootstrap_high95"]) if cr["bootstrap_high95"] not in (None, "", "None") else None, "low_sim_relation_base_alignment": float(g["low_cos_base"]), "high_sim_relation_base_alignment": float(g["high_cos_base"]), "low_sim_relation_triplet_alignment": float(g["low_cos_triplet"]), "high_sim_relation_triplet_alignment": float(g["high_cos_triplet"])})
    write_csv(CROSS_ROOT / "ltcc_vs_prcc_headroom.csv", cross_rows); write_csv(TABLE_ROOT / "headroom_master.csv", cross_rows)
    decisions = decisions_from(cross_rows, metrics_by_dataset); write_json(TABLE_ROOT / "decision.json", decisions)
    manifest = {"phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_JSON), "source_audit_sha256": sha256_file(SOURCE_JSON), "train_only": True, "query_gallery_used_for_thresholds": False, "model_forward_on_ppu": True, "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_created": False, "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False, "training_started": False, "phase7fb_started": False, "headroom_threshold_selected_for_training": False, "decision": decisions, "report": str(REPORT_PATH), "finalized_from_existing_pair_and_gradient_caches": True}
    write_json(MANIFEST_PATH, manifest); final_report(audit, cross_rows, decisions, grad_summary)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decisions, "training_started": False, "phase7fb_started": False}, indent=2, sort_keys=True))


def finalize_from_files() -> None:
    """Fast report-only finalizer; deliberately avoids importing PPU runtime."""
    import json
    ensure_dirs()
    if not SOURCE_JSON.is_file():
        raise RuntimeError("source audit is missing")
    audit = json.loads(SOURCE_JSON.read_text())
    metrics_by_dataset = {}
    grad_summary = []
    cross_rows = []
    for name in ("LTCC", "PRCC"):
        pair_path = PAIR_ROOT / f"{name.lower()}_pair_headroom.npz"
        if not pair_path.is_file():
            raise RuntimeError(f"pair cache is missing: {pair_path}")
        with np.load(pair_path, allow_pickle=False) as archive:
            pos0 = np.asarray(archive["pos0"], dtype=np.float32); dmargin = np.asarray(archive["dmargin"], dtype=np.float32)
        metrics_by_dataset[name] = {"pos0": pos0, "dmargin": dmargin}
        quartiles = read_csv(QUARTILE_ROOT / f"{name.lower()}_quartiles.csv")
        thresholds = {float(row["threshold"]): row for row in read_csv(CURVE_ROOT / f"{name.lower()}_thresholds.csv")}
        corr = read_csv(CORR_ROOT / f"{name.lower()}_correlations.csv")
        cr = next(row for row in corr if row["relationship"] == "headroom_vs_delta_margin")
        gradient_path = GRAD_ROOT / f"{name.lower()}_low_high_gradient_summary.json"
        g = json.loads(gradient_path.read_text()); grad_summary.append(g)
        def maybe_float(value: Any) -> float | None:
            return None if value in (None, "", "None") else float(value)
        cross_rows.append({"dataset": name, "mean_s_pos_baseline": float(pos0.mean()), "median_s_pos_baseline": float(np.median(pos0)), "mean_headroom": float((1.0 - pos0).mean()), "frac_s_pos_ge_070": float(thresholds[0.70]["fraction"]), "frac_s_pos_ge_075": float(thresholds[0.75]["fraction"]), "frac_s_pos_ge_080": float(thresholds[0.80]["fraction"]), "frac_s_pos_ge_085": float(thresholds[0.85]["fraction"]), "frac_s_pos_ge_090": float(thresholds[0.90]["fraction"]), "Q1_delta_M": float(quartiles[0]["delta_margin"]), "Q2_delta_M": float(quartiles[1]["delta_margin"]), "Q3_delta_M": float(quartiles[2]["delta_margin"]), "Q4_delta_M": float(quartiles[3]["delta_margin"]), "Spearman_headroom_delta_M": maybe_float(cr["pair_spearman"]), "bootstrap_low95": maybe_float(cr["bootstrap_low95"]), "bootstrap_high95": maybe_float(cr["bootstrap_high95"]), "low_sim_relation_base_alignment": float(g["low_cos_base"]), "high_sim_relation_base_alignment": float(g["high_cos_base"]), "low_sim_relation_triplet_alignment": float(g["low_cos_triplet"]), "high_sim_relation_triplet_alignment": float(g["high_cos_triplet"])})
    write_csv(CROSS_ROOT / "ltcc_vs_prcc_headroom.csv", cross_rows); write_csv(TABLE_ROOT / "headroom_master.csv", cross_rows); write_csv(GRAD_ROOT / "low_high_gradient_summary.csv", grad_summary)
    decisions = decisions_from(cross_rows, metrics_by_dataset); write_json(TABLE_ROOT / "decision.json", decisions)
    manifest = {"phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_JSON), "source_audit_sha256": sha256_file(SOURCE_JSON), "train_only": True, "query_gallery_used_for_thresholds": False, "model_forward_on_ppu": True, "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_created": False, "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False, "training_started": False, "phase7fb_started": False, "headroom_threshold_selected_for_training": False, "decision": decisions, "report": str(REPORT_PATH), "finalized_from_existing_pair_and_gradient_caches": True, "finalizer": "file-only; no PPU import/model execution"}
    write_json(MANIFEST_PATH, manifest); final_report(audit, cross_rows, decisions, grad_summary)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decisions, "training_started": False, "phase7fb_started": False}, indent=2, sort_keys=True))


def fmt(value: Any) -> str:
    return "" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows: lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def run() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("audit", "run", "finalize", "finalize_files"), default="run"); args = parser.parse_args()
    if args.mode == "audit":
        audit = source_audit(); print(__import__("json").dumps({"passed": audit["source_verification_pass"], "audit": str(SOURCE_JSON), "report": str(SOURCE_REPORT)}, indent=2)); return
    if args.mode == "finalize":
        finalize_existing(); return
    if args.mode == "finalize_files":
        finalize_from_files(); return
    if not SOURCE_JSON.is_file(): source_audit()
    audit = __import__("json").loads(SOURCE_JSON.read_text()); ensure_dirs(); set_seed = p7e.set_seed; set_seed(); shared = {}; metrics_by_dataset = {}; feature_names = {"LTCC": ("ltcc_b0_seed1234", "ltcc_b1_seed1234"), "PRCC": ("prcc_c0_seed1234", "prcc_c1_seed1234")}
    for name in ("LTCC", "PRCC"):
        cfg, dataset, transform = p7e.cfg_dataset(name.lower()); shared[name] = (cfg, dataset, transform); baseline = load_feature_bank(SOURCES[feature_names[name][0]], dataset); endpoint = load_feature_bank(SOURCES[feature_names[name][1]], dataset); metrics = cached_ltcc_pair_metrics(dataset) if name == "LTCC" and LTCC_PAIR_CACHE.is_file() else (pair_metrics(dataset, baseline, endpoint) if name == "LTCC" else cached_prcc_pair_metrics(dataset)); metrics_by_dataset[name] = metrics; del baseline, endpoint
        pair_npz_path = PAIR_ROOT / f"{name.lower()}_pair_headroom.npz"
        if not pair_npz_path.is_file():
            np.savez_compressed(pair_npz_path, pos0=metrics["pos0"], pos1=metrics["pos1"], neg0=metrics["neg0"], neg1=metrics["neg1"], margin0=metrics["margin0"], margin1=metrics["margin1"], dpos=metrics["dpos"], dneg=metrics["dneg"], dmargin=metrics["dmargin"], headroom=metrics["headroom"], multiplicity=metrics["multiplicity"], pid=metrics["pids_per_pair"])
        pair_csv_path = PAIR_ROOT / f"{name.lower()}_pair_headroom.csv"
        if not pair_csv_path.is_file(): write_pair_csv(pair_csv_path, name, metrics)
        write_csv(QUARTILE_ROOT / f"{name.lower()}_quartiles.csv", quartile_analysis(name, metrics)); write_csv(CURVE_ROOT / f"{name.lower()}_saturation_bins.csv", saturation_bins(name, metrics)); write_csv(UTILITY_ROOT / f"{name.lower()}_useful_harmful.csv", useful_harmful(name, metrics)); write_json(UTILITY_ROOT / f"{name.lower()}_predictive_utility.json", predictive_rows(name, metrics)); write_csv(UTILITY_ROOT / f"{name.lower()}_pid_geometry.csv", pid_rows(metrics)); write_csv(CURVE_ROOT / f"{name.lower()}_thresholds.csv", threshold_rows(name, metrics))
        pids = pid_rows(metrics); corr_rows = []
        for label, xkey, ykey, xpair, ypair in (("s_pos0_vs_delta_pos", "mean_pos0", "mean_dpos", metrics["pos0"], metrics["dpos"]), ("s_pos0_vs_delta_margin", "mean_pos0", "mean_dmargin", metrics["pos0"], metrics["dmargin"]), ("headroom_vs_delta_pos", "mean_headroom", "mean_dpos", metrics["headroom"], metrics["dpos"]), ("headroom_vs_delta_margin", "mean_headroom", "mean_dmargin", metrics["headroom"], metrics["dmargin"])):
            boot = bootstrap_pid(pids, xkey, ykey); boot["dataset"] = name; boot["relationship"] = label; boot["pair_spearman"] = spearman(xpair, ypair); corr_rows.append(boot)
        write_csv(CORR_ROOT / f"{name.lower()}_correlations.csv", corr_rows)
    device = p7e.require_ppu(); gradient_rows = []
    for name in ("LTCC", "PRCC"):
        cfg, dataset, transform = shared[name];
        if "a" not in metrics_by_dataset[name]["pairs"]:
            full_pairs = p7e.build_pair_arrays(dataset)
            if len(full_pairs["a"]) != len(metrics_by_dataset[name]["pids_per_pair"]) or not np.array_equal(full_pairs["pid"], metrics_by_dataset[name]["pids_per_pair"]):
                raise RuntimeError(f"{name} endpoint reconstruction does not match cached pair order")
            metrics_by_dataset[name]["pairs"] = full_pairs
        bank = p7e.relation_bank(name.lower(), dataset, cfg); gradient_rows.extend(gradient_probe(name, metrics_by_dataset[name], cfg, dataset, transform, bank, device))
    grad_summary = []
    for name in ("LTCC", "PRCC"):
        # gradient_probe writes the per-dataset summary together with its
        # rows because low/high groups can be sampled from different frozen
        # loader batches.  Reuse that verified summary here rather than
        # assuming both groups occupy one wide row.
        import json
        summary = json.loads((GRAD_ROOT / f"{name.lower()}_low_high_gradient_summary.json").read_text())
        grad_summary.append(summary)
    write_csv(GRAD_ROOT / "low_high_gradient_summary.csv", grad_summary)
    cross_rows = []
    for name in ("LTCC", "PRCC"):
        m = metrics_by_dataset[name]; q = quartile_analysis(name, m); corr = read_csv(CORR_ROOT / f"{name.lower()}_correlations.csv"); g = next(row for row in grad_summary if row["dataset"] == name); thresholds = {float(row["threshold"]): row for row in read_csv(CURVE_ROOT / f"{name.lower()}_thresholds.csv")}; cr = next(row for row in corr if row["relationship"] == "headroom_vs_delta_margin")
        cross_rows.append({"dataset": name, "mean_s_pos_baseline": float(m["pos0"].mean()), "median_s_pos_baseline": float(np.median(m["pos0"])), "mean_headroom": float(m["headroom"].mean()), "frac_s_pos_ge_070": float(thresholds[0.70]["fraction"]), "frac_s_pos_ge_075": float(thresholds[0.75]["fraction"]), "frac_s_pos_ge_080": float(thresholds[0.80]["fraction"]), "frac_s_pos_ge_085": float(thresholds[0.85]["fraction"]), "frac_s_pos_ge_090": float(thresholds[0.90]["fraction"]), "Q1_delta_M": float(q[0]["delta_margin"]), "Q2_delta_M": float(q[1]["delta_margin"]), "Q3_delta_M": float(q[2]["delta_margin"]), "Q4_delta_M": float(q[3]["delta_margin"]), "Spearman_headroom_delta_M": float(cr["pair_spearman"]) if cr["pair_spearman"] not in (None, "", "None") else None, "bootstrap_low95": float(cr["bootstrap_low95"]) if cr["bootstrap_low95"] not in (None, "", "None") else None, "bootstrap_high95": float(cr["bootstrap_high95"]) if cr["bootstrap_high95"] not in (None, "", "None") else None, "low_sim_relation_base_alignment": float(g["low_cos_base"]), "high_sim_relation_base_alignment": float(g["high_cos_base"]), "low_sim_relation_triplet_alignment": float(g["low_cos_triplet"]), "high_sim_relation_triplet_alignment": float(g["high_cos_triplet"])})
    write_csv(CROSS_ROOT / "ltcc_vs_prcc_headroom.csv", cross_rows)
    # Optional visual diagnostics; failure to render does not invalidate numeric audit.
    plot_status = {"plots": [], "errors": []}
    try:
        import matplotlib.pyplot as plt
        for name in ("LTCC", "PRCC"):
            m = metrics_by_dataset[name]; order = np.argsort(m["pos0"]); x = m["pos0"][order]; y = m["dpos"][order]; z = m["dmargin"][order]; plt.figure(figsize=(6, 4)); plt.scatter(x[::max(1, len(x)//10000)], y[::max(1, len(x)//10000)], s=1, alpha=.25); plt.axhline(0, color="black", lw=.7); plt.xlabel("baseline s_pos"); plt.ylabel("Delta_pos"); plt.tight_layout(); plt.savefig(PLOT_ROOT / f"{name.lower()}_baseline_spos_vs_delta_pos.png", dpi=150); plt.close(); plot_status["plots"].append(f"{name.lower()}_baseline_spos_vs_delta_pos.png"); plt.figure(figsize=(6, 4)); plt.scatter(x[::max(1, len(x)//10000)], z[::max(1, len(x)//10000)], s=1, alpha=.25); plt.axhline(0, color="black", lw=.7); plt.xlabel("baseline s_pos"); plt.ylabel("Delta_M"); plt.tight_layout(); plt.savefig(PLOT_ROOT / f"{name.lower()}_baseline_spos_vs_delta_margin.png", dpi=150); plt.close(); plot_status["plots"].append(f"{name.lower()}_baseline_spos_vs_delta_margin.png")
    except Exception as exc: plot_status["errors"].append(repr(exc))
    write_json(PLOT_ROOT / "plot_status.json", plot_status)
    decisions = decisions_from(cross_rows, metrics_by_dataset)
    write_json(TABLE_ROOT / "decision.json", decisions); write_csv(TABLE_ROOT / "headroom_master.csv", cross_rows)
    manifest = {"phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_JSON), "source_audit_sha256": sha256_file(SOURCE_JSON), "train_only": True, "query_gallery_used_for_thresholds": False, "model_forward_on_ppu": True, "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_created": False, "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False, "training_started": False, "phase7fb_started": False, "headroom_threshold_selected_for_training": False, "decision": decisions, "report": str(REPORT_PATH)}
    write_json(MANIFEST_PATH, manifest); final_report(audit, cross_rows, decisions, grad_summary); print(__import__("json").dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decisions, "training_started": False, "phase7fb_started": False}, indent=2, sort_keys=True))


def decisions_from(cross_rows: list[dict[str, Any]], metrics_by_dataset: dict[str, Any]) -> dict[str, Any]:
    ltcc = next(row for row in cross_rows if row["dataset"] == "LTCC"); prcc = next(row for row in cross_rows if row["dataset"] == "PRCC")
    ltcc_m = metrics_by_dataset["LTCC"]; prcc_m = metrics_by_dataset["PRCC"]
    q1 = "YES" if ltcc["mean_s_pos_baseline"] < prcc["mean_s_pos_baseline"] and ltcc["mean_headroom"] > prcc["mean_headroom"] else ("MIXED" if ltcc["mean_headroom"] != prcc["mean_headroom"] else "NO")
    q2 = "YES" if ltcc["Q1_delta_M"] > ltcc["Q4_delta_M"] and prcc["Q1_delta_M"] > prcc["Q4_delta_M"] else ("MIXED" if ltcc["Q1_delta_M"] > ltcc["Q4_delta_M"] or prcc["Q1_delta_M"] > prcc["Q4_delta_M"] else "NO")
    sat_prcc = prcc["frac_s_pos_ge_080"] > 0.25
    q3 = "YES" if (np.mean(prcc_m["dmargin"][(prcc_m["pos0"] >= .8)]) < np.mean(prcc_m["dmargin"][(prcc_m["pos0"] < .8)])) and (np.mean(ltcc_m["dmargin"][(ltcc_m["pos0"] >= .8)]) < np.mean(ltcc_m["dmargin"][(ltcc_m["pos0"] < .8)])) else ("MIXED" if sat_prcc else "NO")
    q4 = "YES" if (prcc["Q1_delta_M"] - prcc["Q4_delta_M"]) > (ltcc["Q1_delta_M"] - ltcc["Q4_delta_M"]) else ("MIXED" if prcc["Q1_delta_M"] > prcc["Q4_delta_M"] else "NO")
    q5 = "YES" if ltcc["bootstrap_low95"] > 0 or prcc["bootstrap_low95"] > 0 or abs(ltcc["Spearman_headroom_delta_M"]) >= .2 or abs(prcc["Spearman_headroom_delta_M"]) >= .2 else ("MIXED" if ltcc["Spearman_headroom_delta_M"] is not None or prcc["Spearman_headroom_delta_M"] is not None else "NO")
    q6 = "YES" if ltcc["high_sim_relation_base_alignment"] > ltcc["low_sim_relation_base_alignment"] and prcc["high_sim_relation_base_alignment"] > prcc["low_sim_relation_base_alignment"] else ("MIXED" if ltcc["high_sim_relation_base_alignment"] > ltcc["low_sim_relation_base_alignment"] or prcc["high_sim_relation_base_alignment"] > prcc["low_sim_relation_base_alignment"] else "NO")
    q7 = "YES" if q2 == "YES" and sat_prcc and q5 == "YES" and q3 in ("YES", "MIXED") else "NO"
    return {"Q1_LTCC_more_headroom": q1, "Q2_low_headroom_pairs_better_delta_margin": q2, "Q3_saturated_pairs_more_zero_or_negative": q3, "Q4_pattern_stronger_PRCC": q4, "Q5_headroom_predictive_power": q5, "Q6_high_similarity_gradients_less_useful_or_more_redundant": q6, "Q7_sufficient_evidence_for_7F_B": q7, "promotion_decision": "PROMOTE TO PHASE 7F-B (recommendation only; wait for confirmation)" if q7 == "YES" else "STOP", "training_started": False, "optimizer_step_called": False, "phase7fb_started": False, "declared_tolerance": TOLERANCE}


def final_report(audit: dict[str, Any], cross_rows: list[dict[str, Any]], decisions: dict[str, Any], grad_summary: list[dict[str, Any]]) -> None:
    quartiles = []
    for name in ("LTCC", "PRCC"): quartiles.extend(read_csv(QUARTILE_ROOT / f"{name.lower()}_quartiles.csv"))
    bins = []
    for name in ("LTCC", "PRCC"): bins.extend(read_csv(CURVE_ROOT / f"{name.lower()}_saturation_bins.csv"))
    utility = []
    for name in ("LTCC", "PRCC"): utility.extend(read_csv(UTILITY_ROOT / f"{name.lower()}_useful_harmful.csv"))
    source_rows = [{"id": name, "dataset": item["dataset"], "arm": item["arm"], "checkpoint": item["checkpoint"]["path"], "checkpoint_sha256": item["checkpoint"]["sha256"], "feature": item["feature"]["path"], "feature_sha256": item["feature"]["sha256"]} for name, item in audit["arms"].items()]
    report = ["# Phase 7F-A — Cross-Clothes Relation Headroom Audit", "", f"Generated: `{utc_now()}`.", "", "## Scope and safeguards", "", "Diagnostic only. All pair construction, thresholds, geometry, and labels use train images/PIDs/clothing states. Query/gallery/test is excluded. Frozen feature banks were reused; the only model execution was unscaled FP32 gradient probing on PPU. No optimizer, scheduler, GradScaler, optimizer step, or new training run was created.", "", "## Exact sources", "", markdown_table(source_rows, ["id", "dataset", "arm", "checkpoint", "checkpoint_sha256", "feature", "feature_sha256"]), "", "Pair-cache paths and hashes are recorded in the source-audit JSON, including the frozen PRCC Phase 7D train-only cache.", "", "## Definitions", "", "`H_pos = 1 - s_pos^0`; `Delta_pos = s_pos^1-s_pos^0`; `Delta_neg = s_neg^1-s_neg^0`; `Delta_M = (s_pos^1-s_neg^1)-(s_pos^0-s_neg^0)`. These are diagnostic quantities only; no H_ID or training threshold is introduced.", "", "## LTCC vs PRCC headroom table", "", markdown_table(cross_rows, ["dataset", "mean_s_pos_baseline", "median_s_pos_baseline", "mean_headroom", "frac_s_pos_ge_070", "frac_s_pos_ge_075", "frac_s_pos_ge_080", "frac_s_pos_ge_085", "frac_s_pos_ge_090", "Q1_delta_M", "Q2_delta_M", "Q3_delta_M", "Q4_delta_M", "Spearman_headroom_delta_M", "bootstrap_low95", "bootstrap_high95", "low_sim_relation_base_alignment", "high_sim_relation_base_alignment"]), "", "## Headroom quartiles", "", markdown_table(quartiles, ["dataset", "quartile", "count", "pid_count", "baseline_s_pos", "baseline_s_neg", "baseline_margin", "delta_pos", "delta_neg", "delta_margin", "endpoint_s_pos", "endpoint_s_neg", "endpoint_margin"]), "", "## Saturation bins", "", markdown_table(bins, ["dataset", "bin", "count", "pid_count", "fraction", "baseline_s_pos", "delta_pos", "delta_neg", "delta_margin"]), "", "## Useful / neutral / harmful", "", f"Tolerance-aware classification uses `abs(Delta_M) <= {TOLERANCE}` as NEUTRAL; zero-tolerance results are also stored.", "", markdown_table([row for row in utility if row["mode"] == "tolerance_aware"], ["dataset", "label", "count", "fraction", "pid_count", "baseline_s_pos", "mean_headroom", "pair_multiplicity_mean", "delta_pos", "delta_neg", "delta_margin"]), "", "## Low-sim vs high-sim Relation gradients", "", markdown_table(grad_summary, ["dataset", "probe_count", "low_norm", "high_norm", "low_cos_base", "high_cos_base", "low_cos_triplet", "high_cos_triplet"]), "", "## Correlations and predictive utility", "", "Pair-level correlations, PID-bootstrap 95% intervals, ROC-AUC, and top-v-bottom quartile discrimination are in `correlations/` and `useful_harmful_analysis/`. These are predictive diagnostics, not causal evidence.", "", "## Q1–Q7", "", "\n".join(f"- **{key}:** {value}" for key, value in decisions.items() if key.startswith("Q")), "", "## Strongest mechanistic conclusion", "", ("Baseline headroom is not a sufficiently validated cross-dataset predictor of later margin benefit under the frozen endpoints; the observed pattern does not jointly satisfy the promotion gate." if decisions["Q7_sufficient_evidence_for_7F_B"] == "NO" else "Low-headroom versus saturated-pair behavior shows a joint signal, but it remains a diagnostic recommendation only."), "", "## Decision", "", f"**{decisions['promotion_decision']}**", "", "**NO NEW TRAINING WAS STARTED. NO PHASE 7F-B WAS STARTED.**", ""]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    run()
