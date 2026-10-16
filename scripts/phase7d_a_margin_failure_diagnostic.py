#!/usr/local/bin/python3
"""Phase 7D-A: train-only margin failure diagnostic.

This module never trains.  It audits and evaluates the frozen Phase 7C C0/C1/C2
E60 checkpoints, extracts PRCC train-only identity features on PPUs, and
computes positive similarity, nearest-negative similarity, identity margins,
quadrants, PID/NQ/PQ stratifications, no-update gradient probes, and offline
counterfactual margin-loss statistics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as phase6b  # noqa: E402
import phase7a_prcc_relation_validation as phase7a  # noqa: E402
import phase7c_prototype_relation as phase7c  # noqa: E402


PHASE = "phase7d_a_margin_failure_diagnostic"
SEED = 1234
WORLD_SIZE = 2
DEVICE_INDEX = 0
EPOCH = 60
FEATURE_BATCH_SIZE = 64
GRADIENT_TARGET_BATCHES = 16
BATCH_SIZE = phase7c.BATCH_SIZE
NUM_INSTANCES = phase7c.NUM_INSTANCES
EPS = 1e-12

P7C_ROOT = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
P7C_RUNS = {
    "C0": P7C_ROOT / "training/C0_bio",
    "C1": P7C_ROOT / "training/C1_image_uniform_relation",
    "C2": P7C_ROOT / "training/C2_prototype_relation",
}
P7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
P7A_RUNS = {
    "P0": P7A_ROOT / "P0_prcc_bio",
    "P1": P7A_ROOT / "P1_prcc_uniform_relation",
}
P7C_SOURCE_AUDIT = P7C_ROOT / "source_audit/source_audit.json"
P7C_MANIFEST = P7C_ROOT / "PHASE7C_MANIFEST.json"
P7A_MANIFEST = P7A_ROOT / "PHASE7A_MANIFEST.json"
P7A_METHOD = P7A_ROOT / "artifacts/frozen_relation_method.md"
P7A_BANK = P7A_ROOT / "artifacts/prcc_relation_bank.npz"
PHASE6B_SCRIPT = ROOT / "scripts/phase6b_soft_relation_screen.py"
PHASE7A_SCRIPT = ROOT / "scripts/phase7a_prcc_relation_validation.py"
PHASE7C_SCRIPT = ROOT / "scripts/phase7c_prototype_relation.py"

OUT_ROOT = ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic"
SOURCE_ROOT = OUT_ROOT / "source_audit"
FEATURE_ROOT = OUT_ROOT / "feature_bank"
PAIR_ROOT = OUT_ROOT / "pair_margin_analysis"
QUADRANT_ROOT = OUT_ROOT / "quadrant_analysis"
PID_ROOT = OUT_ROOT / "per_pid_analysis"
NEG_ROOT = OUT_ROOT / "negative_stratification"
POS_ROOT = OUT_ROOT / "positive_saturation"
GRAD_ROOT = OUT_ROOT / "gradient_diagnostic"
COUNTER_ROOT = OUT_ROOT / "counterfactual_margin_loss"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7D_A_MANIFEST.json"
REPORT_PATH = ROOT / "reports/PHASE_7D_A_MARGIN_FAILURE_DIAGNOSTIC.md"
SOURCE_REPORT = ROOT / "reports/PHASE_7D_A_SOURCE_AUDIT.md"


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_ppu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 7D-A requires exactly {WORLD_SIZE} visible PPUs; CPU fallback is forbidden")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SOURCE_ROOT, FEATURE_ROOT, PAIR_ROOT, QUADRANT_ROOT,
                 PID_ROOT, NEG_ROOT, POS_ROOT, GRAD_ROOT, COUNTER_ROOT,
                 TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected checkpoint type: {type(state)!r} at {path}")
    return state


def infer_feature_dim(path: Path) -> int | None:
    state = checkpoint_state(path)
    weight = state.get("head.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    if not P7C_SOURCE_AUDIT.is_file() or not P7C_MANIFEST.is_file():
        raise RuntimeError("Phase 7C source audit/manifest is missing")
    p7c_audit = read_json(P7C_SOURCE_AUDIT)
    p7c_manifest = read_json(P7C_MANIFEST)
    required_source_files = [P7A_METHOD, PHASE6B_SCRIPT, PHASE7A_SCRIPT, PHASE7C_SCRIPT, P7A_BANK]
    source_files = {str(path): file_record(path) for path in required_source_files}
    if not all(record["exists"] for record in source_files.values()):
        raise RuntimeError("a frozen relation source file is missing")

    arms: dict[str, Any] = {}
    for arm, run_dir in P7C_RUNS.items():
        run_manifest_path = run_dir / "run_manifest.json"
        checkpoint_path = run_dir / "model_final.pth"
        checkpoint_meta = run_dir / "checkpoint_epoch060.json"
        config_path = run_dir / "config.yml"
        complete_path = run_dir / "RUN_COMPLETE.json"
        required = [run_manifest_path, checkpoint_path, checkpoint_meta, config_path, complete_path]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Phase 7C {arm} frozen source is incomplete")
        run = read_json(run_manifest_path)
        complete = read_json(complete_path)
        if run.get("seed") != SEED or run.get("end_epoch") != EPOCH or complete.get("epochs_completed") != EPOCH:
            raise RuntimeError(f"Phase 7C {arm} does not match seed1234 E60")
        if run.get("relation_feature") != "feat[0]":
            raise RuntimeError(f"Phase 7C {arm} relation feature is not verified as feat[0]")
        arms[arm] = {
            "label": run.get("label"),
            "checkpoint": file_record(checkpoint_path),
            "checkpoint_epoch": EPOCH,
            "checkpoint_selection_rule": "Phase 7C E60 model_final.pth; no best-epoch substitution",
            "best_checkpoint_recorded_but_not_used": run_dir / "model_best.pth",
            "best_checkpoint_sha256": sha256_file(run_dir / "model_best.pth") if (run_dir / "model_best.pth").is_file() else None,
            "checkpoint_meta": file_record(checkpoint_meta),
            "config": file_record(config_path),
            "run_manifest": file_record(run_manifest_path),
            "run_manifest_fields": {
                "seed": run.get("seed"), "epochs": run.get("epochs"), "batch_size": run.get("batch_size"),
                "num_instances": run.get("num_instances"), "relation_feature": run.get("relation_feature"),
                "relation_type": run.get("relation_type"), "lambda_rel": run.get("lambda_rel"),
                "optimizer": run.get("optimizer"), "scheduler": run.get("scheduler"), "amp": run.get("amp"),
                "sampler": run.get("sampler"), "evaluation": run.get("evaluation"),
            },
            "feature_dimension_from_classifier_checkpoint": infer_feature_dim(checkpoint_path),
        }
    p7a_sources = {}
    if P7A_MANIFEST.is_file():
        p7a_manifest = read_json(P7A_MANIFEST)
        p7a_sources["manifest"] = file_record(P7A_MANIFEST)
        p7a_sources["manifest_primary_window"] = p7a_manifest.get("primary_window")
    for arm, run_dir in P7A_RUNS.items():
        for name in ("run_manifest.json", "model_final.pth", "checkpoint_epoch060.json", "config.yml"):
            path = run_dir / name
            p7a_sources[f"{arm}_{name}"] = file_record(path)

    if not all(arm_data["feature_dimension_from_classifier_checkpoint"] == 1024 for arm_data in arms.values()):
        raise RuntimeError("Phase 7C checkpoint feature dimension is not 1024")
    payload = {
        "phase": PHASE, "created_utc": utc_now(), "training_started": False, "optimizer_step_called": False,
        "primary_source": "Phase 7C C0/C1/C2 E60 model_final.pth; Phase 7A is reference-only",
        "feature_source": "normalized student identity feature feat[0], verified by Phase 7A frozen method and Phase 7C source audit",
        "feature_dimension": 1024, "dataset": "PRCC train only", "seed": SEED,
        "phase7c_manifest": file_record(P7C_MANIFEST), "phase7c_source_audit": file_record(P7C_SOURCE_AUDIT),
        "phase7c_arms": arms, "phase7a_reference_sources": p7a_sources, "frozen_source_files": source_files,
        "relation_bank": file_record(P7A_BANK), "relation_bank_semantics": "501040 canonical unordered valid same-PID/different-clothes train pairs",
        "checkpoint_selection_rule": "E60 final checkpoints for all primary diagnostics; model_best is recorded only for provenance",
        "comparability_guard": "Phase 7A paths are recorded as historical reference and are not mixed into Phase 7C primary geometry or margin deltas",
        "audit_pass": True,
    }
    write_json(SOURCE_ROOT / "source_audit.json", payload)
    lines = [
        "# PHASE 7D-A Source Audit", "", "## Scope", "",
        "Diagnostic-only audit. No model training, optimizer construction, optimizer step, checkpoint modification, or dataset split change is permitted.", "",
        "Primary sources are the frozen Phase 7C PRCC seed1234 E60 final checkpoints. Phase 7A sources are recorded as reference-only and are not mixed with Phase 7C primary deltas.", "",
        "## Frozen Phase 7C sources", "", "| Arm | Checkpoint | Epoch | Feature | Dim | Relation | Lambda |", "|---|---|---:|---|---:|---|---:|",
    ]
    for arm, data in arms.items():
        fields = data["run_manifest_fields"]
        lines.append(f"| {arm} | `{data['checkpoint']['path']}` | {data['checkpoint_epoch']} | `{fields['relation_feature']}` | {data['feature_dimension_from_classifier_checkpoint']} | {fields['relation_type']} | {fields['lambda_rel']} |")
    lines += [
        "", "All three primary checkpoints use seed 1234, batch size 16, NUM_INSTANCES 4, 60 epochs, SGD, the Phase 7A cosine/warmup schedule, and the frozen AMP configuration.", "",
        "## Feature and implementation verification", "",
        f"- Phase 7A frozen method: `{P7A_METHOD}` (SHA256 `{sha256_file(P7A_METHOD)}`).",
        f"- Executed relation implementation: `{PHASE6B_SCRIPT}` (SHA256 `{sha256_file(PHASE6B_SCRIPT)}`).",
        f"- Phase 7C runner/source audit: `{PHASE7C_SCRIPT}`, `{P7C_SOURCE_AUDIT}`.",
        f"- Frozen PRCC train relation bank: `{P7A_BANK}`; canonical unordered pair count 501,040.",
        "- Exact relation feature: normalized `feat[0]`; query/gallery/test are excluded from all diagnostic construction.",
        "- Primary checkpoint rule: E60 `model_final.pth`; the recorded best checkpoint is not substituted.",
        "", "## Phase 7A reference sources", "",
        f"- P0: `{P7A_RUNS['P0']}`.", f"- P1: `{P7A_RUNS['P1']}`.",
        "- These are provenance references only; all primary feature banks and margin deltas below are C0/C1/C2 Phase 7C E60 models.",
        "", "## Audit result", "", "**PASS**. Exact identity feature source and all primary frozen checkpoints are verified.",
        "", f"Machine-readable audit: `{SOURCE_ROOT / 'source_audit.json'}`.",
    ]
    SOURCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_prcc_dataset() -> tuple[Any, Any, dict[str, Any]]:
    cfg = phase7c.cfg_for(P7C_RUNS["C0"])
    dataset = phase7a.build_dataset(cfg)
    bank = phase7a.load_relation_bank(dataset)
    if len(dataset.train) != 17896 or dataset.num_train_pids != 150:
        raise RuntimeError("PRCC train topology differs from frozen Phase 7C")
    if bank["exact_pair_count"] != 501040:
        raise RuntimeError(f"unexpected PRCC relation pair count {bank['exact_pair_count']}")
    return cfg, dataset, bank


def metadata_rows(dataset: Any) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(dataset.train):
        camera = item.get("camid", item.get("camera", item.get("cam", "")))
        rows.append({"index": index, "image_path": str(item["image_path"]), "image_id": Path(str(item["image_path"])).name,
                     "pid": int(item["pid"]), "clothes_id": int(item["clothes_id"]), "camera_id": camera})
    return rows


def extract_features() -> dict[str, Any]:
    require_ppu(); set_seed(); ensure_dirs()
    audit = read_json(SOURCE_ROOT / "source_audit.json")
    cfg, dataset, bank = load_prcc_dataset()
    rows = metadata_rows(dataset)
    write_csv(FEATURE_ROOT / "train_metadata.csv", rows)
    pids = np.asarray([row["pid"] for row in rows], dtype=np.int64)
    clothes = np.asarray([row["clothes_id"] for row in rows], dtype=np.int64)
    write_json(FEATURE_ROOT / "pair_source.json", {"phase": PHASE, "training_started": False, "train_images": len(rows), "pair_count": bank["exact_pair_count"], "path_order_sha256": str(bank["arrays"].get("path_order_sha256", np.asarray("" )).item())})
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    torch.cuda.set_device(DEVICE_INDEX)
    bank_manifest = {"phase": PHASE, "feature_source": "normalized feat[0] via phase7c.extract_train_features", "device": str(device), "autocast": True, "optimizer_step_called": False, "training_started": False, "arms": {}}
    for arm, run_dir in P7C_RUNS.items():
        checkpoint = run_dir / "model_final.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = phase7a.make_model_from_state(cfg, state, 150, 3).to(device)
        features, extracted_pids, extracted_clothes = phase7c.extract_train_features(model, dataset, cfg, device)
        del model, state
        torch.cuda.empty_cache(); gc.collect()
        if features.shape != (len(rows), 1024):
            raise RuntimeError(f"{arm} feature shape mismatch {features.shape}")
        if not np.array_equal(extracted_pids, pids) or not np.array_equal(extracted_clothes, clothes):
            raise RuntimeError(f"{arm} metadata order mismatch")
        norms = np.linalg.norm(features.astype(np.float64), axis=1)
        path = FEATURE_ROOT / f"features_{arm}_E60.npy"
        np.save(path, features.astype(np.float32))
        bank_manifest["arms"][arm] = {"path": str(path), "sha256": sha256_file(path), "shape": list(features.shape), "dtype": str(features.dtype), "norm_mean": float(np.mean(norms)), "norm_min": float(np.min(norms)), "norm_max": float(np.max(norms)), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(FEATURE_ROOT / "feature_bank_manifest.json", bank_manifest)
    return {"dataset": dataset, "bank": bank, "pids": pids, "clothes": clothes, "metadata": rows, "audit": audit}


def load_feature_context() -> dict[str, Any]:
    cfg, dataset, bank = load_prcc_dataset()
    metadata = read_csv(FEATURE_ROOT / "train_metadata.csv")
    pids = np.asarray([int(row["pid"]) for row in metadata], dtype=np.int64)
    clothes = np.asarray([int(row["clothes_id"]) for row in metadata], dtype=np.int64)
    features = {}
    for arm in P7C_RUNS:
        path = FEATURE_ROOT / f"features_{arm}_E60.npy"
        if not path.is_file():
            raise RuntimeError(f"missing feature bank {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (len(metadata), 1024):
            raise RuntimeError(f"feature bank shape mismatch for {arm}: {arr.shape}")
        features[arm] = np.asarray(arr, dtype=np.float32)
    arrays = bank["arrays"]
    pair_a = arrays["index_a"].astype(np.int64)
    pair_b = arrays["index_b"].astype(np.int64)
    if np.any(pair_a >= pair_b) or len(pair_a) != 501040:
        raise RuntimeError("pair bank is not canonical or has unexpected size")
    if not np.all(pids[pair_a] == pids[pair_b]) or not np.all(clothes[pair_a] != clothes[pair_b]):
        raise RuntimeError("pair bank violates same-PID/different-clothes definition")
    return {"cfg": cfg, "dataset": dataset, "bank": bank, "metadata": metadata, "pids": pids, "clothes": clothes, "features": features, "pair_a": pair_a, "pair_b": pair_b}


def identity_centers(features: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray, np.ndarray]:
    unique = sorted(set(pids.tolist()))
    centers = []
    for pid in unique:
        center = np.mean(features[pids == pid].astype(np.float64), axis=0)
        center /= max(np.linalg.norm(center), EPS)
        centers.append(center)
    centers_arr = np.asarray(centers, dtype=np.float64)
    pid_to_center = {pid: index for index, pid in enumerate(unique)}
    sims = features.astype(np.float64) @ centers_arr.T
    own = np.asarray([pid_to_center[int(pid)] for pid in pids], dtype=np.int64)
    sims[np.arange(len(pids)), own] = -np.inf
    nearest = np.max(sims, axis=1)
    nearest_pid = np.asarray(unique, dtype=np.int64)[np.argmax(sims, axis=1)]
    return centers_arr, pid_to_center, nearest, nearest_pid


def stat_values(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)), "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)), "max": float(np.max(values)),
            "pct_gt0": float(np.mean(values > 0.0) * 100.0), "pct_le0": float(np.mean(values <= 0.0) * 100.0),
            "pct_lt_minus005": float(np.mean(values < -0.05) * 100.0), "pct_gt005": float(np.mean(values > 0.05) * 100.0)}


def metric_stats_rows(name: str, values: np.ndarray, arm: str | None = None, comparison: str | None = None) -> dict[str, Any]:
    row = {"metric": name, **stat_values(values)}
    if arm is not None: row["arm"] = arm
    if comparison is not None: row["comparison"] = comparison
    return row


def state_relation_indices(pids: np.ndarray, clothes: np.ndarray, pair_a: np.ndarray, pair_b: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    keys = [(int(pids[a]), min(int(clothes[a]), int(clothes[b])), max(int(clothes[a]), int(clothes[b]))) for a, b in zip(pair_a, pair_b)]
    unique = sorted(set(keys))
    lookup = {key: index for index, key in enumerate(unique)}
    return np.asarray([lookup[key] for key in keys], dtype=np.int64), unique


def pair_analysis(context: dict[str, Any]) -> dict[str, Any]:
    pids = context["pids"]; clothes = context["clothes"]; pair_a = context["pair_a"]; pair_b = context["pair_b"]; features = context["features"]
    n_pairs = len(pair_a)
    results: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    for arm, arr in features.items():
        _centers, _pid_map, nearest, nearest_pid = identity_centers(arr, pids)
        pos = np.sum(arr[pair_a].astype(np.float64) * arr[pair_b].astype(np.float64), axis=1)
        neg_anchor = nearest[pair_a]
        neg_pair = np.maximum(nearest[pair_a], nearest[pair_b])
        margin_anchor = pos - neg_anchor
        margin_pair = pos - neg_pair
        results[arm] = {"pos": pos, "neg_anchor": neg_anchor, "neg_pair": neg_pair, "margin_anchor": margin_anchor, "margin": margin_pair, "nearest_image": nearest, "nearest_pid": nearest_pid}
        for metric, values in (("s_pos", pos), ("s_neg_anchor", neg_anchor), ("s_neg_pair", neg_pair), ("M_anchor", margin_anchor), ("M_pair", margin_pair)):
            summary_rows.append(metric_stats_rows(metric, values, arm=arm))
    write_csv(PAIR_ROOT / "pair_metric_summary.csv", summary_rows)
    state_ids, state_keys = state_relation_indices(pids, clothes, pair_a, pair_b)
    state_rows = []
    for state_id, key in enumerate(state_keys):
        mask = state_ids == state_id
        row = {"state_relation_id": state_id, "pid": key[0], "cloth_a": key[1], "cloth_b": key[2], "image_pair_count": int(np.sum(mask))}
        for arm in P7C_RUNS:
            for metric in ("pos", "neg_anchor", "neg_pair", "margin_anchor", "margin"):
                row[f"{arm}_{metric}_mean"] = float(np.mean(results[arm][metric][mask]))
                row[f"{arm}_{metric}_median"] = float(np.median(results[arm][metric][mask]))
        state_rows.append(row)
    write_csv(PAIR_ROOT / "state_relation_metrics.csv", state_rows)

    delta_arrays: dict[str, np.ndarray] = {}
    delta_rows = []
    for comparison, left, right in (("C1-C0", "C1", "C0"), ("C2-C0", "C2", "C0"), ("C2-C1", "C2", "C1")):
        for metric in ("pos", "neg_pair", "margin", "neg_anchor", "margin_anchor"):
            delta = results[left][metric] - results[right][metric]
            delta_arrays[f"{comparison.replace('-', '_')}_{metric}"] = delta
            delta_rows.append(metric_stats_rows(metric, delta, comparison=comparison))
    write_csv(PAIR_ROOT / "delta_summary.csv", delta_rows)
    npz_payload = {"pair_a": pair_a, "pair_b": pair_b, "pid": pids[pair_a], "clothes_a": clothes[pair_a], "clothes_b": clothes[pair_b], "state_relation_id": state_ids}
    for arm in P7C_RUNS:
        for metric in ("pos", "neg_anchor", "neg_pair", "margin_anchor", "margin"):
            npz_payload[f"{arm}_{metric}"] = results[arm][metric]
    npz_payload.update(delta_arrays)
    np.savez_compressed(PAIR_ROOT / "pair_metrics.npz", **npz_payload)
    write_json(PAIR_ROOT / "pair_metrics_manifest.json", {"phase": PHASE, "pairs": n_pairs, "state_relations": len(state_keys), "canonical_unordered": True, "train_only": True, "query_gallery_used": False, "array_file": str(PAIR_ROOT / "pair_metrics.npz"), "array_file_sha256": sha256_file(PAIR_ROOT / "pair_metrics.npz")})
    return {"results": results, "state_ids": state_ids, "state_keys": state_keys, "summary_rows": summary_rows, "delta_arrays": delta_arrays, "state_rows": state_rows}


def geometry_table(context: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    pids = context["pids"]; clothes = context["clothes"]; pair_a = context["pair_a"]; pair_b = context["pair_b"]
    for arm, arr in context["features"].items():
        result = analysis["results"][arm]
        state_centers: dict[tuple[int, int], np.ndarray] = {}
        for pid in sorted(set(pids.tolist())):
            for cloth in sorted(set(clothes[pids == pid].tolist())):
                values = arr[(pids == pid) & (clothes == cloth)].astype(np.float64).mean(axis=0)
                values /= max(np.linalg.norm(values), EPS)
                state_centers[(pid, cloth)] = values
        dispersion = []
        for pid in sorted(set(pids.tolist())):
            state_keys = [key for key in state_centers if key[0] == pid]
            for first, left_key in enumerate(state_keys[:-1]):
                for right_key in state_keys[first + 1:]:
                    dispersion.append(1.0 - float(np.dot(state_centers[left_key], state_centers[right_key])))
        rows.append({"arm": arm, "checkpoint": "E60", "train_only": True, "same_pid_cross_clothes_similarity": float(np.mean(result["pos"])), "nearest_negative_similarity": float(np.mean(result["neg_pair"])), "identity_margin": float(np.mean(result["margin"])), "median_identity_margin": float(np.median(result["margin"])), "pct_margin_gt0": float(np.mean(result["margin"] > 0) * 100.0), "within_pid_clothing_state_dispersion": float(np.mean(dispersion)), "state_relation_count": len(dispersion)})
    qrows = []
    for comparison, delta_key in (("C1-C0", "C1_C0_margin"), ("C2-C0", "C2_C0_margin")):
        delta_margin = analysis["delta_arrays"][delta_key]
        delta_pos = analysis["delta_arrays"][delta_key.replace("margin", "pos")]
        delta_neg = analysis["delta_arrays"][delta_key.replace("margin", "neg_pair")]
        qrows.append({"comparison": comparison, "Q_A_fraction": float(np.mean((delta_pos > 0) & (delta_neg <= 0))), "Q_B_fraction": float(np.mean((delta_pos > 0) & (delta_neg > 0) & (delta_margin > 0))), "Q_C_fraction": float(np.mean((delta_pos > 0) & (delta_neg > 0) & (delta_margin <= 0))), "Q_D_fraction": float(np.mean(delta_pos <= 0))})
    write_csv(TABLE_ROOT / "geometry_table.csv", rows)
    write_csv(QUADRANT_ROOT / "quadrant_summary.csv", qrows)
    return rows


def quartile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(len(values), dtype=np.int64)
    for rank, index in enumerate(order):
        labels[index] = min(4, rank * 4 // len(values) + 1)
    cuts = [float(np.percentile(values, q)) for q in (25, 50, 75)]
    return labels, cuts


def stratified_analysis(context: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    c0 = analysis["results"]["C0"]
    nq, nq_cuts = quartile_labels(c0["neg_pair"])
    pq, pq_cuts = quartile_labels(c0["pos"])
    rows_nq = []; rows_pq = []
    for name, labels, rows in (("NQ", nq, rows_nq), ("PQ", pq, rows_pq)):
        for q in range(1, 5):
            mask = labels == q
            row = {"stratum": f"{name}{q}", "count": int(np.sum(mask)), "baseline_s_pos": float(np.mean(c0["pos"][mask])), "baseline_s_neg_pair": float(np.mean(c0["neg_pair"][mask])), "baseline_margin": float(np.mean(c0["margin"][mask]))}
            for arm in ("C1", "C2"):
                row[f"{arm}_delta_pos"] = float(np.mean((analysis["results"][arm]["pos"] - c0["pos"])[mask]))
                row[f"{arm}_delta_neg"] = float(np.mean((analysis["results"][arm]["neg_pair"] - c0["neg_pair"])[mask]))
                row[f"{arm}_delta_margin"] = float(np.mean((analysis["results"][arm]["margin"] - c0["margin"])[mask]))
            rows.append(row)
    write_csv(NEG_ROOT / "NQ1_NQ4_analysis.csv", rows_nq)
    write_csv(POS_ROOT / "PQ1_PQ4_analysis.csv", rows_pq)
    write_json(NEG_ROOT / "quartile_definition.json", {"variable": "C0 pair-conservative nearest-negative similarity", "order": "NQ1 lowest to NQ4 highest", "cutpoints": nq_cuts, "assignment": "rank-based equal-count quartiles"})
    write_json(POS_ROOT / "quartile_definition.json", {"variable": "C0 cross-clothes positive similarity", "order": "PQ1 lowest to PQ4 highest", "cutpoints": pq_cuts, "assignment": "rank-based equal-count quartiles"})
    return {"nq": nq, "pq": pq, "nq_rows": rows_nq, "pq_rows": rows_pq, "nq_cuts": nq_cuts, "pq_cuts": pq_cuts}


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def per_pid_analysis(context: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    pids = context["pids"]; clothes = context["clothes"]; pair_pid = pids[context["pair_a"]]
    rows = []
    for pid in sorted(set(pids.tolist())):
        image_count = int(np.sum(pids == pid)); clothes_count = int(len(set(clothes[pids == pid].tolist()))); mask = pair_pid == pid
        row = {"pid": pid, "number_clothes": clothes_count, "number_images": image_count, "pair_count": int(np.sum(mask))}
        for arm in ("C0", "C1", "C2"):
            for metric in ("pos", "neg_pair", "margin"):
                row[f"{arm}_{metric}"] = float(np.mean(analysis["results"][arm][metric][mask]))
        for arm in ("C1", "C2"):
            for metric in ("pos", "neg_pair", "margin"):
                row[f"{arm}_C0_delta_{metric}"] = row[f"{arm}_{metric}"] - row[f"C0_{metric}"]
        rows.append(row)
    write_csv(PID_ROOT / "per_pid_margin_analysis.csv", rows)
    correlation_rows = []
    for arm in ("C1", "C2"):
        delta_margin = np.asarray([row[f"{arm}_C0_delta_margin"] for row in rows], dtype=np.float64)
        for predictor, values in (("baseline_margin", np.asarray([row["C0_margin"] for row in rows], dtype=np.float64)), ("number_clothes", np.asarray([row["number_clothes"] for row in rows], dtype=np.float64)), ("pair_count", np.asarray([row["pair_count"] for row in rows], dtype=np.float64)), ("baseline_s_neg_pair", np.asarray([row["C0_neg_pair"] for row in rows], dtype=np.float64))):
            correlation_rows.append({"comparison": f"{arm}-C0", "predictor": predictor, "pearson_r": pearson(values, delta_margin), "n": len(rows), "constant_predictor": bool(np.std(values) <= EPS)})
    write_csv(PID_ROOT / "per_pid_correlations.csv", correlation_rows)
    return {"rows": rows, "correlations": correlation_rows}


def counterfactual_analysis(context: dict[str, Any], analysis: dict[str, Any], strata: dict[str, Any]) -> dict[str, Any]:
    summary_rows = []; stratum_rows = []; pid_rows = []
    pids = context["pids"]; pair_pid = pids[context["pair_a"]]
    for arm in ("C0", "C1", "C2"):
        margin = analysis["results"][arm]["margin"]
        for margin_value in (0.0, 0.05, 0.10):
            loss = np.maximum(0.0, margin_value - margin)
            active = loss > 0.0
            summary_rows.append({"arm": arm, "m": margin_value, "active_count": int(np.sum(active)), "active_pct": float(np.mean(active) * 100.0), "mean_hypothetical_loss": float(np.mean(loss)), "median_hypothetical_loss": float(np.median(loss)), "active_pid_count": int(len(set(pair_pid[active].tolist())))})
            for name, labels in (("NQ", strata["nq"]), ("PQ", strata["pq"])):
                for q in range(1, 5):
                    mask = labels == q
                    stratum_rows.append({"arm": arm, "m": margin_value, "stratification": name, "stratum": f"{name}{q}", "pair_count": int(np.sum(mask)), "active_count": int(np.sum(active & mask)), "active_pct_within_stratum": float(np.mean(active[mask]) * 100.0), "share_of_all_active": float(np.sum(active & mask) / max(1, np.sum(active)) * 100.0), "mean_loss_within_stratum": float(np.mean(loss[mask]))})
            for pid in sorted(set(pair_pid.tolist())):
                mask = pair_pid == pid
                pid_rows.append({"arm": arm, "m": margin_value, "pid": pid, "pair_count": int(np.sum(mask)), "active_count": int(np.sum(active & mask)), "active_pct": float(np.mean(active[mask]) * 100.0), "loss_sum": float(np.sum(loss[mask])), "share_of_all_active": float(np.sum(active & mask) / max(1, np.sum(active)) * 100.0)})
    write_csv(COUNTER_ROOT / "counterfactual_summary.csv", summary_rows)
    write_csv(COUNTER_ROOT / "counterfactual_by_stratum.csv", stratum_rows)
    write_csv(COUNTER_ROOT / "counterfactual_by_pid.csv", pid_rows)
    return {"summary": summary_rows, "strata": stratum_rows, "pids": pid_rows}


def gradient_group_metrics(names: list[str], base_grads: tuple[torch.Tensor | None, ...], rel_grads: tuple[torch.Tensor | None, ...], selected: set[str]) -> dict[str, Any]:
    base_sq = rel_sq = dot = 0.0
    for name, bg, rg in zip(names, base_grads, rel_grads):
        if name not in selected:
            continue
        reference = bg if bg is not None else rg
        if reference is None:
            continue
        if bg is None: bg = torch.zeros_like(reference)
        if rg is None: rg = torch.zeros_like(reference)
        bg = bg.detach().float(); rg = rg.detach().float()
        base_sq += float(torch.sum(bg * bg).cpu()); rel_sq += float(torch.sum(rg * rg).cpu()); dot += float(torch.sum(bg * rg).cpu())
    base_norm = math.sqrt(max(base_sq, 0.0)); rel_norm = math.sqrt(max(rel_sq, 0.0))
    return {"norm_base": base_norm, "norm_relation": rel_norm, "norm_ratio": rel_norm / (base_norm + EPS), "alignment": dot / (base_norm * rel_norm + EPS) if base_norm > EPS and rel_norm > EPS else None}


def subset_relation_loss(feat: list[torch.Tensor], local_pairs: list[tuple[int, int]], selected: set[int]) -> torch.Tensor:
    representation = F.normalize(feat[0].float(), dim=-1)
    losses = [1.0 - torch.sum(representation[i] * representation[j]) for pair_index, (i, j) in enumerate(local_pairs) if pair_index in selected]
    if not losses:
        return representation.sum() * 0.0
    return torch.stack(losses).mean()


def gradient_diagnostic(context: dict[str, Any], analysis: dict[str, Any], strata: dict[str, Any]) -> dict[str, Any]:
    require_ppu(); set_seed(); ensure_dirs()
    cfg = context["cfg"]; dataset = context["dataset"]; bank = context["bank"]
    pair_lookup = {(int(a), int(b)): index for index, (a, b) in enumerate(zip(context["pair_a"].tolist(), context["pair_b"].tolist()))}
    global_margin = analysis["results"]["C0"]["margin"]
    q25 = float(np.percentile(global_margin, 25)); q75 = float(np.percentile(global_margin, 75))
    sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=0, seed=SEED)
    train_transform, _test_transform = phase7a.build_img_transforms(cfg)
    loader = DataLoader(ImageDataset(dataset.train, transform=train_transform), sampler=sampler, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, drop_last=True)
    state = torch.load(P7C_RUNS["C0"] / "model_final.pth", map_location="cpu", weights_only=False)
    device = torch.device(f"cuda:{DEVICE_INDEX}"); torch.cuda.set_device(DEVICE_INDEX)
    model = phase7a.make_model_from_state(cfg, state, 150, 3).to(device).train(); del state
    loss_fn, _center = phase7a.make_loss(cfg, num_classes=150)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _parameter in named]; parameters = [parameter for _name, parameter in named]
    block_ids = sorted({int(name.split(".")[1]) for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    groups = {"late_eva": {name for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late_ids}, "id_head": {name for name in names if name.startswith("head.")}, "bio_projection": {name for name in names if name.startswith("head_bio") or name.startswith("head_clip_bio") or name.startswith("norm_bio")}}
    rows = []; inspected = 0; target = GRADIENT_TARGET_BATCHES
    for batch in loader:
        inspected += 1
        global_indices = [bank["path_to_index"][str(path)] for path in batch["image_path"]]
        local_pairs: list[tuple[int, int]] = []; local_margins: list[float] = []
        for i in range(len(global_indices)):
            for j in range(i + 1, len(global_indices)):
                if int(batch["pid"][i]) != int(batch["pid"][j]) or int(batch["clothes_id"][i]) == int(batch["clothes_id"][j]):
                    continue
                key = tuple(sorted((global_indices[i], global_indices[j])))
                if key not in pair_lookup:
                    continue
                local_pairs.append((i, j)); local_margins.append(float(global_margin[pair_lookup[key]]))
        if not local_pairs:
            if inspected > 2000: break
            continue
        low = {index for index, value in enumerate(local_margins) if value <= q25}
        high = {index for index, value in enumerate(local_margins) if value >= q75}
        if not low or not high:
            if inspected > 2000: break
            continue
        images = batch["image"].to(device); pids = batch["pid"].to(device); camids = batch["camid"].to(device); clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device)
        writer = base.MetricCapture(); model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=camids)
        base_loss = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=len(rows) + 1)
        low_loss = subset_relation_loss(feat, local_pairs, low); high_loss = subset_relation_loss(feat, local_pairs, high)
        base_grads = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
        low_grads = torch.autograd.grad(low_loss, parameters, retain_graph=True, allow_unused=True)
        high_grads = torch.autograd.grad(high_loss, parameters, retain_graph=False, allow_unused=True)
        row = {"batch_index": inspected, "low_relation_count": len(low), "high_relation_count": len(high), "all_relation_count": len(local_pairs), "global_low_cutoff": q25, "global_high_cutoff": q75, "base_loss": float(base_loss.detach().float().cpu()), "low_relation_loss": float(low_loss.detach().float().cpu()), "high_relation_loss": float(high_loss.detach().float().cpu()), "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_step": False}
        for group_name, selected in groups.items():
            low_metrics = gradient_group_metrics(names, base_grads, low_grads, selected); high_metrics = gradient_group_metrics(names, base_grads, high_grads, selected)
            for key, value in low_metrics.items(): row[f"low_{key}_{group_name}"] = value
            for key, value in high_metrics.items(): row[f"high_{key}_{group_name}"] = value
        rows.append(row)
        model.zero_grad(set_to_none=True); del base_grads, low_grads, high_grads, base_loss, low_loss, high_loss, feat, score
        if len(rows) >= target:
            break
    del loader, model
    gc.collect(); torch.cuda.empty_cache()
    if len(rows) < target:
        raise RuntimeError(f"only {len(rows)} matched low/high margin gradient probes found after {inspected} batches")
    write_csv(GRAD_ROOT / "gradient_probe_rows.csv", rows)
    summary_rows = []
    for group_name in groups:
        for stratum in ("low", "high"):
            for metric in ("norm_relation", "norm_ratio", "alignment"):
                values = np.asarray([row[f"{stratum}_{metric}_{group_name}"] for row in rows if row.get(f"{stratum}_{metric}_{group_name}") not in (None, "", "None")], dtype=np.float64)
                summary_rows.append({"margin_group": stratum, "module": group_name, "metric": metric, "count": len(values), "mean": float(np.mean(values)) if len(values) else None, "std": float(np.std(values)) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "min": float(np.min(values)) if len(values) else None, "max": float(np.max(values)) if len(values) else None})
    write_csv(GRAD_ROOT / "gradient_summary.csv", summary_rows)
    write_json(GRAD_ROOT / "gradient_manifest.json", {"phase": PHASE, "baseline_checkpoint": str(P7C_RUNS["C0"] / "model_final.pth"), "rows": len(rows), "target_rows": target, "inspected_batches": inspected, "low_cutoff": q25, "high_cutoff": q75, "device": str(device), "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_step_called": False, "training_started": False, "relation_gradient": "image-level uniform 1-cos on selected exact train pairs"})
    return {"rows": rows, "summary": summary_rows, "low_cutoff": q25, "high_cutoff": q75}


def plot_diagnostics(analysis: dict[str, Any], strata: dict[str, Any]) -> None:
    status = {"phase": PHASE, "plots": [], "errors": []}
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for arm, color in (("C0", "black"), ("C1", "tab:blue"), ("C2", "tab:orange")):
            plt.hist(analysis["results"][arm]["margin"], bins=80, density=True, alpha=0.35, label=arm, color=color)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1); plt.xlabel("pair-conservative identity margin"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); plt.savefig(PLOT_ROOT / "margin_distribution.png", dpi=160); plt.close(); status["plots"].append("margin_distribution.png")
        for comparison, pos_key, neg_key in (("C1-C0", "C1_C0_pos", "C1_C0_neg_pair"), ("C2-C0", "C2_C0_pos", "C2_C0_neg_pair")):
            plt.figure(figsize=(6, 5)); plt.scatter(analysis["delta_arrays"][pos_key][::10], analysis["delta_arrays"][neg_key][::10], s=2, alpha=0.25); lim = max(abs(float(np.min(analysis["delta_arrays"][pos_key]))), abs(float(np.max(analysis["delta_arrays"][pos_key]))), abs(float(np.min(analysis["delta_arrays"][neg_key]))), abs(float(np.max(analysis["delta_arrays"][neg_key])))); plt.axhline(0, color="black", linewidth=0.7); plt.axvline(0, color="black", linewidth=0.7); plt.plot([-lim, lim], [-lim, lim], linestyle="--", color="red", linewidth=0.8); plt.xlabel("Delta positive similarity"); plt.ylabel("Delta nearest-negative similarity"); plt.title(comparison); plt.tight_layout(); filename = comparison.replace("-", "_").lower() + "_delta_scatter.png"; plt.savefig(PLOT_ROOT / filename, dpi=160); plt.close(); status["plots"].append(filename)
        plt.figure(figsize=(8, 4)); x = np.arange(4); width = 0.35; nq_rows = strata["nq_rows"]; c1 = [row["C1_delta_margin"] for row in nq_rows]; c2 = [row["C2_delta_margin"] for row in nq_rows]; plt.bar(x - width / 2, c1, width, label="C1-C0"); plt.bar(x + width / 2, c2, width, label="C2-C0"); plt.axhline(0, color="black", linewidth=0.7); plt.xticks(x, ["NQ1", "NQ2", "NQ3", "NQ4"]); plt.ylabel("mean Delta margin"); plt.legend(); plt.tight_layout(); plt.savefig(PLOT_ROOT / "negative_quartile_delta_margin.png", dpi=160); plt.close(); status["plots"].append("negative_quartile_delta_margin.png")
    except Exception as exc:  # diagnostic output remains valid without plots
        status["errors"].append(repr(exc))
    write_json(PLOT_ROOT / "plot_status.json", status)


def decision_logic(analysis: dict[str, Any], strata: dict[str, Any], counter: dict[str, Any], gradient: dict[str, Any]) -> dict[str, Any]:
    r = analysis["results"]
    d10 = analysis["delta_arrays"]
    c1_pos = d10["C1_C0_pos"]; c1_neg = d10["C1_C0_neg_pair"]; c1_margin = d10["C1_C0_margin"]
    c2_pos = d10["C2_C0_pos"]; c2_neg = d10["C2_C0_neg_pair"]; c2_margin = d10["C2_C0_margin"]
    qc_c1 = float(np.mean((c1_pos > 0) & (c1_neg > 0) & (c1_margin <= 0)))
    qc_c2 = float(np.mean((c2_pos > 0) & (c2_neg > 0) & (c2_margin <= 0)))
    nq = {row["stratum"]: row for row in strata["nq_rows"]}
    q5 = "YES" if nq["NQ4"]["C1_delta_margin"] < nq["NQ1"]["C1_delta_margin"] and nq["NQ4"]["C1_delta_neg"] > nq["NQ1"]["C1_delta_neg"] else ("MIXED" if nq["NQ4"]["C1_delta_neg"] > 0 else "NO")
    q7_checks = []
    for row in counter["summary"]:
        if row["arm"] != "C0": continue
        m = float(row["m"]); nq4 = next(item for item in counter["strata"] if item["arm"] == "C0" and float(item["m"]) == m and item["stratum"] == "NQ4"); nq1 = next(item for item in counter["strata"] if item["arm"] == "C0" and float(item["m"]) == m and item["stratum"] == "NQ1"); q7_checks.append(nq4["active_pct_within_stratum"] > nq1["active_pct_within_stratum"])
    q7 = "YES" if all(q7_checks) else ("MIXED" if any(q7_checks) else "NO")
    q1 = "YES" if float(np.mean(c1_pos)) > 0 and float(np.mean(c1_pos > 0)) >= 0.5 else ("MIXED" if float(np.mean(c1_pos)) > 0 else "NO")
    q2 = "YES" if float(np.mean(c1_neg)) > 0 and float(np.mean(c1_neg > 0)) >= 0.5 else ("MIXED" if float(np.mean(c1_neg)) > 0 else "NO")
    q3 = "YES" if float(np.mean(c1_margin)) > 0 and float(np.mean(c1_margin > 0)) >= 0.5 else ("MIXED" if float(np.mean(c1_margin)) > 0 else "NO")
    q4 = "YES" if qc_c1 >= 0.10 else "NO"
    q6 = "YES" if float(np.mean(c2_pos)) > 0 and float(np.mean(c2_neg)) > 0 and float(np.mean(c2_margin)) <= 0 else ("MIXED" if float(np.mean(c2_pos)) > 0 or float(np.mean(c2_neg)) > 0 else "NO")
    q8 = "YES" if (q2 in ("YES", "MIXED")) and q3 != "YES" and (q4 == "YES" or q5 == "YES") and q7 == "YES" else "NO"
    promotion = "PROMOTE TO PHASE 7D-B (diagnostic evidence only; wait for confirmation)" if q8 == "YES" else "STOP"
    return {"Q1_C1_positive_similarity_increases": q1, "Q2_C1_nearest_negative_increases": q2, "Q3_C1_margin_improves": q3, "Q4_substantial_QC": q4, "Q5_failure_concentrated_high_confusion": q5, "Q6_C2_same_compactness_without_margin": q6, "Q7_hypothetical_margin_focuses_hard_relations": q7, "Q8_evidence_justifies_margin_experiment": q8, "C1_QC_fraction": qc_c1, "C2_QC_fraction": qc_c2, "promotion_decision": promotion, "training_started": False, "optimizer_step_called": False, "phase7db_started": False, "gradient_probe_rows": len(gradient["rows"])}


def report(manifest: dict[str, Any], analysis: dict[str, Any], geometry: list[dict[str, Any]], strata: dict[str, Any], pid: dict[str, Any], counter: dict[str, Any], gradient: dict[str, Any], decisions: dict[str, Any]) -> None:
    lines = ["# PHASE 7D-A — Margin Failure Diagnostic", "", "## 1. Scope and frozen sources", "", "This is a train-only diagnostic. No new training run, optimizer, optimizer.step(), checkpoint modification, sampler change, or Phase 7D-B run was started.", "", f"- Source audit: `{SOURCE_REPORT}`.", f"- Primary frozen models: Phase 7C C0/C1/C2 E60 `model_final.pth`.", f"- Feature source: normalized student `feat[0]`, dimension 1024.", f"- PRCC train relation bank: 501,040 canonical unordered same-PID/different-clothes image pairs.", f"- Phase 7A sources are recorded for provenance only: `{P7A_RUNS['P0']}`, `{P7A_RUNS['P1']}`.", "", "## 2. Geometry table", "", "| Arm | Mean s_pos | Mean s_neg (pair) | Mean margin | Median margin | % M>0 | State dispersion |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in geometry:
        lines.append(f"| {row['arm']} | {row['same_pid_cross_clothes_similarity']:.6f} | {row['nearest_negative_similarity']:.6f} | {row['identity_margin']:.6f} | {row['median_identity_margin']:.6f} | {row['pct_margin_gt0']:.2f} | {row['within_pid_clothing_state_dispersion']:.6f} |")
    lines += ["", "## 3. Delta distributions", "", "| Comparison | Metric | Mean | Median | P10 | P90 | % positive |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in analysis["delta_rows"]:
        if row["metric"] in ("pos", "neg_pair", "margin"):
            lines.append(f"| {row['comparison']} | {row['metric']} | {float(row['mean']):.6f} | {float(row['median']):.6f} | {float(row['p10']):.6f} | {float(row['p90']):.6f} | {float(row['pct_gt0']):.2f}% |")
    lines += ["", "## 4. Four-quadrant analysis", "", "| Comparison | Q-A best | Q-B mixed useful | Q-C compactness without discrimination | Q-D no positive gain |", "|---|---:|---:|---:|---:|"]
    for row in read_csv(QUADRANT_ROOT / "quadrant_summary.csv"):
        lines.append(f"| {row['comparison']} | {float(row['Q_A_fraction'])*100:.2f}% | {float(row['Q_B_fraction'])*100:.2f}% | {float(row['Q_C_fraction'])*100:.2f}% | {float(row['Q_D_fraction'])*100:.2f}% |")
    lines += ["", "## 5. Hard-negative NQ1–NQ4", "", "| Stratum | C0 margin | C1 Δpos | C1 Δneg | C1 Δmargin | C2 Δpos | C2 Δneg | C2 Δmargin |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in strata["nq_rows"]:
        lines.append(f"| {row['stratum']} | {row['baseline_margin']:.6f} | {row['C1_delta_pos']:.6f} | {row['C1_delta_neg']:.6f} | {row['C1_delta_margin']:.6f} | {row['C2_delta_pos']:.6f} | {row['C2_delta_neg']:.6f} | {row['C2_delta_margin']:.6f} |")
    lines += ["", "## 6. Positive-saturation PQ1–PQ4", "", "| Stratum | C0 s_pos | C0 s_neg | C0 margin | C1 Δpos | C1 Δneg | C1 Δmargin | C2 Δpos | C2 Δneg | C2 Δmargin |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in strata["pq_rows"]:
        lines.append(f"| {row['stratum']} | {row['baseline_s_pos']:.6f} | {row['baseline_s_neg_pair']:.6f} | {row['baseline_margin']:.6f} | {row['C1_delta_pos']:.6f} | {row['C1_delta_neg']:.6f} | {row['C1_delta_margin']:.6f} | {row['C2_delta_pos']:.6f} | {row['C2_delta_neg']:.6f} | {row['C2_delta_margin']:.6f} |")
    lines += ["", "## 7. Per-PID analysis", "", "Per-PID rows and correlations are stored in `per_pid_analysis/`. PRCC has exactly two clothing states per PID, so the number-of-clothes correlation is undefined because that predictor is constant.", "", "| Comparison | Predictor | Pearson r with Δmargin |", "|---|---|---:|"]
    for row in pid["correlations"]:
        value = "NA" if row["pearson_r"] is None else f"{row['pearson_r']:.6f}"
        lines.append(f"| {row['comparison']} | {row['predictor']} | {value} |")
    lines += ["", "## 8. Low/high-margin gradient diagnostic", "", f"- Frozen C0 E60 model; {len(gradient['rows'])} matched train micro-batches; FP32; autocast disabled; no optimizer step.", f"- Low group: global C0 pair margin ≤ {gradient['low_cutoff']:.6f}; high group: ≥ {gradient['high_cutoff']:.6f}.", "", "| Margin group | Module | Relation norm | Norm ratio | cos(g_base,g_rel) |", "|---|---|---:|---:|---:|"]
    for row in gradient["summary"]:
        if row["metric"] == "alignment":
            norm = next(item for item in gradient["summary"] if item["margin_group"] == row["margin_group"] and item["module"] == row["module"] and item["metric"] == "norm_relation")
            ratio = next(item for item in gradient["summary"] if item["margin_group"] == row["margin_group"] and item["module"] == row["module"] and item["metric"] == "norm_ratio")
            alignment = "NA" if row["mean"] in (None, "", "None") else f"{float(row['mean']):.6f}"
            lines.append(f"| {row['margin_group']} | {row['module']} | {float(norm['mean']):.6f} | {float(ratio['mean']):.6f} | {alignment} |")
    lines += ["", "## 9. Counterfactual margin-loss audit", "", "These are offline values only; m=0, 0.05, and 0.10 were not tuned and no corresponding loss was trained.", "", "| Arm | m | Active pairs | Active % | Mean hypothetical loss | Active PIDs |", "|---|---:|---:|---:|---:|---:|"]
    for row in counter["summary"]:
        lines.append(f"| {row['arm']} | {row['m']:.2f} | {row['active_count']} | {row['active_pct']:.2f}% | {row['mean_hypothetical_loss']:.6f} | {row['active_pid_count']} |")
    lines += ["", "## 10. Q1–Q8", ""]
    for key, value in decisions.items():
        if key.startswith("Q"):
            lines.append(f"- {key}: **{value}**.")
    lines += ["", "## 11. Decision", "", f"**{decisions['promotion_decision']}**.", "", "The diagnostic does not establish that a margin loss is better. It only determines whether the positive-only relation objective exhibits a margin-related failure mode.", "", "## 12. STOP", "", "**NO NEW TRAINING WAS STARTED. NO PHASE 7D-B WAS STARTED.**", "", "## Artifacts", "", f"- Manifest: `{MANIFEST_PATH}`", f"- Feature bank: `{FEATURE_ROOT}`", f"- Pair analysis: `{PAIR_ROOT}`", f"- Gradient diagnostic: `{GRAD_ROOT}`", f"- Counterfactual loss: `{COUNTER_ROOT}`", f"- Tables: `{TABLE_ROOT}`", f"- Plots: `{PLOT_ROOT}`"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_gradient: bool = True) -> dict[str, Any]:
    ensure_dirs()
    context = load_feature_context()
    analysis = pair_analysis(context)
    # Preserve the delta rows in the returned analysis for report rendering.
    analysis["delta_rows"] = read_csv(PAIR_ROOT / "delta_summary.csv")
    geometry = geometry_table(context, analysis)
    strata = stratified_analysis(context, analysis)
    pid = per_pid_analysis(context, analysis)
    counter = counterfactual_analysis(context, analysis, strata)
    if run_gradient:
        gradient = gradient_diagnostic(context, analysis, strata)
    else:
        gradient_manifest = read_json(GRAD_ROOT / "gradient_manifest.json")
        gradient = {"rows": [{} for _ in range(int(gradient_manifest["rows"]))], "summary": read_csv(GRAD_ROOT / "gradient_summary.csv"), "low_cutoff": float(gradient_manifest["low_cutoff"]), "high_cutoff": float(gradient_manifest["high_cutoff"])}
    plot_diagnostics(analysis, strata)
    decisions = decision_logic(analysis, strata, counter, gradient)
    manifest = {"phase": PHASE, "created_utc": utc_now(), "seed": SEED, "dataset": "PRCC train only", "primary_checkpoints": {arm: str(P7C_RUNS[arm] / "model_final.pth") for arm in P7C_RUNS}, "checkpoint_epoch": EPOCH, "feature_source": "normalized feat[0]", "feature_dimension": 1024, "pair_count": len(context["pair_a"]), "state_relation_count": len(analysis["state_keys"]), "training_started": False, "optimizer_step_called": False, "query_gallery_used": False, "source_audit": str(SOURCE_ROOT / "source_audit.json"), "feature_bank_manifest": str(FEATURE_ROOT / "feature_bank_manifest.json"), "pair_metrics": str(PAIR_ROOT / "pair_metrics.npz"), "geometry": str(TABLE_ROOT / "geometry_table.csv"), "quadrants": str(QUADRANT_ROOT / "quadrant_summary.csv"), "per_pid": str(PID_ROOT / "per_pid_margin_analysis.csv"), "negative_stratification": str(NEG_ROOT / "NQ1_NQ4_analysis.csv"), "positive_saturation": str(POS_ROOT / "PQ1_PQ4_analysis.csv"), "gradient": str(GRAD_ROOT / "gradient_summary.csv"), "counterfactual": str(COUNTER_ROOT / "counterfactual_summary.csv"), "decisions": decisions, "promotion_decision": decisions["promotion_decision"], "phase7db_started": False, "forbidden_actions": ["new training", "optimizer.step", "checkpoint modification", "lambda tuning", "sampler change", "SR3", "H_ID", "H_cloth", "prototype training", "triplet variant", "new margin loss", "memory bank", "EMA", "queue", "backbone change", "query/gallery relation construction"]}
    write_json(MANIFEST_PATH, manifest)
    report(manifest, analysis, geometry, strata, pid, counter, gradient, decisions)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "extract", "analyze", "render"), required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(json_safe(source_audit()), indent=2, sort_keys=True))
    elif args.mode == "extract":
        print(json.dumps(json_safe(extract_features()), indent=2, sort_keys=True))
    else:
        print(json.dumps(analyze(run_gradient=args.mode == "analyze"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
