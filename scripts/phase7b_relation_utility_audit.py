#!/usr/local/bin/python3
"""Phase 7B: train-only cross-dataset relation utility audit.

This file deliberately has no optimizer, scheduler, scaler, or training loop.
It only loads frozen Phase 6C/6C-R/7A checkpoints, extracts train-split
features on CUDA, measures relation geometry, and computes autograd gradients
without an optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
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
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as phase6b  # noqa: E402
import phase7a_prcc_relation_validation as phase7a  # noqa: E402


PHASE = "phase7b_relation_utility_audit"
SEED = 1234
OUT_ROOT = ROOT / "outputs/new_method_track/phase7b_relation_utility_audit"
SOURCE_ROOT = OUT_ROOT / "source_audit"
HEADROOM_ROOT = OUT_ROOT / "relation_headroom"
REDUNDANCY_ROOT = OUT_ROOT / "relation_redundancy"
GRADIENT_ROOT = OUT_ROOT / "gradient_utility"
GEOMETRY_ROOT = OUT_ROOT / "geometry_comparison"
PID_ROOT = OUT_ROOT / "per_pid_analysis"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7B_MANIFEST.json"
SOURCE_REPORT = ROOT / "reports/PHASE_7B_SOURCE_AUDIT.md"
FINAL_REPORT = ROOT / "reports/PHASE_7B_CROSS_DATASET_RELATION_UTILITY_AUDIT.md"

PHASE6C_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
PHASE6CR_ROOT = ROOT / "outputs/new_method_track/phase6cr_relation_replication"
PHASE7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
FROZEN_METHOD = PHASE6CR_ROOT / "artifacts/frozen_uniform_relation_method.md"
PRCC_METHOD = PHASE7A_ROOT / "artifacts/frozen_relation_method.md"
PHASE6B_SCRIPT = ROOT / "scripts/phase6b_soft_relation_screen.py"
PHASE6C_SCRIPT = ROOT / "scripts/phase6c_full_schedule_confirmation.py"
PHASE6CR_SCRIPT = ROOT / "scripts/phase6cr_relation_replication.py"
PHASE7A_SCRIPT = ROOT / "scripts/phase7a_prcc_relation_validation.py"

SPECS: dict[str, dict[str, Any]] = {
    "ltcc_b0_seed1234": {
        "dataset": "ltcc",
        "seed": 1234,
        "role": "Bio baseline",
        "arm": "B0",
        "phase": "phase6c_full_schedule_confirmation",
        "run_dir": PHASE6C_ROOT / "B0_full",
        "model": PHASE6C_ROOT / "B0_full/model_final.pth",
        "state": PHASE6C_ROOT / "B0_full/rank0_state_epoch060.pt",
        "config": PHASE6C_ROOT / "B0_full/config.yml",
        "run_manifest": PHASE6C_ROOT / "B0_full/run_manifest.json",
        "checkpoint": PHASE6C_ROOT / "B0_full/checkpoint_epoch060.json",
        "amp": PHASE6C_ROOT / "B0_full/AMP.json",
        "metrics": PHASE6C_ROOT / "B0_full/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase6B compact frozen candidate bank; uniform B1 semantics",
        "relation_bank_source": PHASE6B_SCRIPT,
        "lambda_applied": 0.0,
        "sampler": "Phase6C custom full-data pair-unit stream, 596 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "ltcc_b1_seed1234",
    },
    "ltcc_b1_seed1234": {
        "dataset": "ltcc",
        "seed": 1234,
        "role": "Uniform relation",
        "arm": "B1",
        "phase": "phase6c_full_schedule_confirmation",
        "run_dir": PHASE6C_ROOT / "B1_full",
        "model": PHASE6C_ROOT / "B1_full/model_final.pth",
        "state": PHASE6C_ROOT / "B1_full/rank0_state_epoch060.pt",
        "config": PHASE6C_ROOT / "B1_full/config.yml",
        "run_manifest": PHASE6C_ROOT / "B1_full/run_manifest.json",
        "checkpoint": PHASE6C_ROOT / "B1_full/checkpoint_epoch060.json",
        "amp": PHASE6C_ROOT / "B1_full/AMP.json",
        "metrics": PHASE6C_ROOT / "B1_full/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase6B compact frozen candidate bank; uniform w_ij=1",
        "relation_bank_source": PHASE6B_SCRIPT,
        "lambda_applied": 0.1,
        "sampler": "Phase6C custom full-data pair-unit stream, 596 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "ltcc_b0_seed1234",
    },
    "ltcc_r0_seed4321": {
        "dataset": "ltcc",
        "seed": 4321,
        "role": "Bio baseline",
        "arm": "R0",
        "phase": "phase6c_r_relation_replication",
        "run_dir": PHASE6CR_ROOT / "R0_seed4321",
        "model": PHASE6CR_ROOT / "R0_seed4321/model_final.pth",
        "state": PHASE6CR_ROOT / "R0_seed4321/rank0_state_epoch060.pt",
        "config": PHASE6CR_ROOT / "R0_seed4321/config.yml",
        "run_manifest": PHASE6CR_ROOT / "R0_seed4321/run_manifest.json",
        "checkpoint": PHASE6CR_ROOT / "R0_seed4321/checkpoint_epoch060.json",
        "amp": PHASE6CR_ROOT / "R0_seed4321/AMP.json",
        "metrics": PHASE6CR_ROOT / "R0_seed4321/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase6B compact frozen candidate bank; uniform B1 semantics",
        "relation_bank_source": PHASE6B_SCRIPT,
        "lambda_applied": 0.0,
        "sampler": "Phase6C-R custom full-data pair-unit stream, 596 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "ltcc_r1_seed4321",
    },
    "ltcc_r1_seed4321": {
        "dataset": "ltcc",
        "seed": 4321,
        "role": "Uniform relation",
        "arm": "R1",
        "phase": "phase6c_r_relation_replication",
        "run_dir": PHASE6CR_ROOT / "R1_seed4321",
        "model": PHASE6CR_ROOT / "R1_seed4321/model_final.pth",
        "state": PHASE6CR_ROOT / "R1_seed4321/rank0_state_epoch060.pt",
        "config": PHASE6CR_ROOT / "R1_seed4321/config.yml",
        "run_manifest": PHASE6CR_ROOT / "R1_seed4321/run_manifest.json",
        "checkpoint": PHASE6CR_ROOT / "R1_seed4321/checkpoint_epoch060.json",
        "amp": PHASE6CR_ROOT / "R1_seed4321/AMP.json",
        "metrics": PHASE6CR_ROOT / "R1_seed4321/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase6B compact frozen candidate bank; uniform w_ij=1",
        "relation_bank_source": PHASE6B_SCRIPT,
        "lambda_applied": 0.1,
        "sampler": "Phase6C-R custom full-data pair-unit stream, 596 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "ltcc_r0_seed4321",
    },
    "prcc_p0_seed1234": {
        "dataset": "prcc",
        "seed": 1234,
        "role": "Bio baseline",
        "arm": "P0",
        "phase": "phase7a_prcc_relation_validation",
        "run_dir": PHASE7A_ROOT / "P0_prcc_bio",
        "model": PHASE7A_ROOT / "P0_prcc_bio/model_final.pth",
        "state": PHASE7A_ROOT / "P0_prcc_bio/rank0_state_epoch060.pt",
        "config": PHASE7A_ROOT / "P0_prcc_bio/config.yml",
        "run_manifest": PHASE7A_ROOT / "P0_prcc_bio/run_manifest.json",
        "checkpoint": PHASE7A_ROOT / "P0_prcc_bio/checkpoint_epoch060.json",
        "amp": PHASE7A_ROOT / "P0_prcc_bio/AMP.json",
        "metrics": PHASE7A_ROOT / "P0_prcc_bio/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase7A full train-only ground-truth bank; diagnostic only",
        "relation_bank_source": PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz",
        "lambda_applied": 0.0,
        "sampler": "repository DistributedRandomIdentitySampler, 551 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "prcc_p1_seed1234",
    },
    "prcc_p1_seed1234": {
        "dataset": "prcc",
        "seed": 1234,
        "role": "Uniform relation",
        "arm": "P1",
        "phase": "phase7a_prcc_relation_validation",
        "run_dir": PHASE7A_ROOT / "P1_prcc_uniform_relation",
        "model": PHASE7A_ROOT / "P1_prcc_uniform_relation/model_final.pth",
        "state": PHASE7A_ROOT / "P1_prcc_uniform_relation/rank0_state_epoch060.pt",
        "config": PHASE7A_ROOT / "P1_prcc_uniform_relation/config.yml",
        "run_manifest": PHASE7A_ROOT / "P1_prcc_uniform_relation/run_manifest.json",
        "checkpoint": PHASE7A_ROOT / "P1_prcc_uniform_relation/checkpoint_epoch060.json",
        "amp": PHASE7A_ROOT / "P1_prcc_uniform_relation/AMP.json",
        "metrics": PHASE7A_ROOT / "P1_prcc_uniform_relation/metrics_per_epoch.csv",
        "relation_bank_policy": "Phase7A full train-only ground-truth bank; uniform w_ij=1",
        "relation_bank_source": PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz",
        "lambda_applied": 0.1,
        "sampler": "repository DistributedRandomIdentitySampler, 551 updates/rank/epoch",
        "feature_source": "model eval feat_bio; training relation feature is feat[0]",
        "pair": "prcc_p0_seed1234",
    },
}

BASELINE_SPECS = [
    "ltcc_b0_seed1234",
    "ltcc_r0_seed4321",
    "prcc_p0_seed1234",
]
GEOMETRY_PAIRS = [
    ("ltcc_b0_seed1234", "ltcc_b1_seed1234"),
    ("ltcc_r0_seed4321", "ltcc_r1_seed4321"),
    ("prcc_p0_seed1234", "prcc_p1_seed1234"),
]


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7B requires PPU/CUDA for every model forward and gradient probe; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SOURCE_ROOT, HEADROOM_ROOT, REDUNDANCY_ROOT,
                 GRADIENT_ROOT, GEOMETRY_ROOT, PID_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def cfg_for(dataset_name: str) -> Any:
    if dataset_name == "ltcc":
        return phase6b.configure()
    if dataset_name == "prcc":
        return phase7a.configure(OUT_ROOT)
    raise ValueError(dataset_name)


def dataset_for(dataset_name: str, cfg: Any | None = None) -> Any:
    cfg = cfg or cfg_for(dataset_name)
    if dataset_name == "ltcc":
        return base.build_dataset(cfg)
    return phase7a.build_dataset(cfg)


def model_factory_for(dataset_name: str):
    return base.factory if dataset_name == "ltcc" else phase7a.factory


def relation_bank_for(dataset_name: str, dataset: Any, cfg: Any) -> dict[str, Any]:
    if dataset_name == "ltcc":
        bank = phase6b.parse_relation_bank(dataset, cfg)
    else:
        bank = phase7a.load_relation_bank(dataset)
    bank["path_to_index"] = {str(item["image_path"]): i for i, item in enumerate(dataset.train)}
    return bank


def load_model(spec_id: str, device: torch.device, train: bool = False) -> tuple[Any, Any, Any]:
    spec = SPECS[spec_id]
    cfg = cfg_for(spec["dataset"])
    dataset = dataset_for(spec["dataset"], cfg)
    factory = model_factory_for(spec["dataset"])
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=dataset.num_train_pids,
        camera_num=dataset.num_camera,
    )
    state = torch.load(spec["model"], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{spec_id} checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.train(mode=train)
    return model, dataset, cfg


def path_order_hash(dataset: Any) -> str:
    return sha256_bytes("\n".join(str(item["image_path"]) for item in dataset.train).encode("utf-8"))


def pair_arrays(dataset: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    for index, item in enumerate(dataset.train):
        by_pid_cloth[int(item["pid"])][int(item["clothes_id"])].append(index)

    cross_a: list[int] = []
    cross_b: list[int] = []
    same_a: list[int] = []
    same_b: list[int] = []
    state_rows: list[dict[str, Any]] = []
    for pid in sorted(by_pid_cloth):
        values = by_pid_cloth[pid]
        cloth_ids = sorted(values)
        for cloth in cloth_ids:
            indices = values[cloth]
            if len(indices) > 1:
                for pos, first in enumerate(indices[:-1]):
                    same_a.extend([first] * (len(indices) - pos - 1))
                    same_b.extend(indices[pos + 1:])
        for first_pos, first_cloth in enumerate(cloth_ids):
            for second_cloth in cloth_ids[first_pos + 1:]:
                left = values[first_cloth]
                right = values[second_cloth]
                cross_a.extend([a for a in left for _ in right])
                cross_b.extend([b for _ in left for b in right])
                state_rows.append({
                    "pid": pid,
                    "cloth_a": first_cloth,
                    "cloth_b": second_cloth,
                    "image_pair_count": len(left) * len(right),
                    "images_cloth_a": len(left),
                    "images_cloth_b": len(right),
                })

    arrays = {
        "cross_a": np.asarray(cross_a, dtype=np.int32),
        "cross_b": np.asarray(cross_b, dtype=np.int32),
        "same_a": np.asarray(same_a, dtype=np.int32),
        "same_b": np.asarray(same_b, dtype=np.int32),
        "pids": pids,
        "clothes": clothes,
    }
    relation_counts = np.asarray([row["image_pair_count"] for row in state_rows], dtype=np.float64)
    structure = {
        "train_images": len(dataset.train),
        "pids": len(by_pid_cloth),
        "clothing_states": len({(pid, cloth) for pid, values in by_pid_cloth.items() for cloth in values}),
        "clothes_per_pid_min": min(len(values) for values in by_pid_cloth.values()),
        "clothes_per_pid_mean": float(np.mean([len(values) for values in by_pid_cloth.values()])),
        "clothes_per_pid_median": float(np.median([len(values) for values in by_pid_cloth.values()])),
        "clothes_per_pid_max": max(len(values) for values in by_pid_cloth.values()),
        "valid_changing_clothes_pids": sum(len(values) >= 2 for values in by_pid_cloth.values()),
        "unique_state_relations": len(state_rows),
        "image_pair_count": len(cross_a),
        "same_clothes_image_pair_count": len(same_a),
        "all_same_pid_image_pairs": len(same_a) + len(cross_a),
        "pair_multiplicity_ratio": float(len(cross_a) / len(state_rows)) if state_rows else None,
        "multiplicity_mean": float(np.mean(relation_counts)) if len(relation_counts) else None,
        "multiplicity_median": float(np.median(relation_counts)) if len(relation_counts) else None,
        "multiplicity_p75": float(np.percentile(relation_counts, 75)) if len(relation_counts) else None,
        "multiplicity_p90": float(np.percentile(relation_counts, 90)) if len(relation_counts) else None,
        "multiplicity_p95": float(np.percentile(relation_counts, 95)) if len(relation_counts) else None,
        "multiplicity_max": float(np.max(relation_counts)) if len(relation_counts) else None,
        "state_relation_rows": state_rows,
        "path_order_sha256": path_order_hash(dataset),
    }
    for pct in (1, 5, 10):
        if len(relation_counts):
            count = max(1, int(math.ceil(len(relation_counts) * pct / 100)))
            structure[f"top_{pct}pct_pair_share"] = float(
                np.sort(relation_counts)[-count:].sum() / relation_counts.sum()
            )
        else:
            structure[f"top_{pct}pct_pair_share"] = None
    return arrays, structure


def actual_bank_summary(dataset_name: str, dataset: Any, cfg: Any, arrays: dict[str, np.ndarray], structure: dict[str, Any]) -> dict[str, Any]:
    bank = relation_bank_for(dataset_name, dataset, cfg)
    exact = bank["exact_lookup"]
    bank_keys = set(exact)
    all_keys = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(arrays["cross_a"].tolist(), arrays["cross_b"].tolist())
    }
    state_keys = {
        (int(arrays["pids"][a]), min(int(arrays["clothes"][a]), int(arrays["clothes"][b])),
         max(int(arrays["clothes"][a]), int(arrays["clothes"][b])))
        for a, b in zip(arrays["cross_a"].tolist(), arrays["cross_b"].tolist())
    }
    bank_state_keys = {
        (int(arrays["pids"][a]), min(int(arrays["clothes"][a]), int(arrays["clothes"][b])),
         max(int(arrays["clothes"][a]), int(arrays["clothes"][b])))
        for a, b in bank_keys
    }
    return {
        "bank_pair_count": len(bank_keys),
        "valid_pair_count": len(all_keys),
        "bank_pair_coverage": float(len(bank_keys) / len(all_keys)) if all_keys else None,
        "bank_state_relation_count": len(bank_state_keys),
        "valid_state_relation_count": len(state_keys),
        "bank_state_relation_coverage": float(len(bank_state_keys) / len(state_keys)) if state_keys else None,
        "bank_source": str(SPECS[
            "ltcc_b0_seed1234" if dataset_name == "ltcc" else "prcc_p0_seed1234"
        ]["relation_bank_source"]),
        "bank_weight_policy": "uniform w_ij=1 for diagnostic B1/P1; LTCC actual bank is compact, PRCC actual bank is full",
        "bank_sha256": sha256_file(
            PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz"
            if dataset_name == "prcc"
            else phase6b.A2_CROSS
        ),
    }


def source_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def audit_sources() -> dict[str, Any]:
    ensure_dirs()
    required_global = [PHASE6B_SCRIPT, PHASE6C_SCRIPT, PHASE6CR_SCRIPT, PHASE7A_SCRIPT,
                       FROZEN_METHOD, PRCC_METHOD,
                       PHASE6C_ROOT / "PHASE6C_MANIFEST.json",
                       PHASE6CR_ROOT / "PHASE6CR_MANIFEST.json",
                       PHASE7A_ROOT / "PHASE7A_MANIFEST.json"]
    missing = [str(path) for path in required_global if not path.is_file()]
    if missing:
        raise RuntimeError("Phase 7B source audit missing required frozen sources: " + ", ".join(missing))
    dataset_summaries: dict[str, Any] = {}
    structure_rows: list[dict[str, Any]] = []
    for dataset_name in ("ltcc", "prcc"):
        cfg = cfg_for(dataset_name)
        dataset = dataset_for(dataset_name, cfg)
        arrays, structure = pair_arrays(dataset)
        bank_summary = actual_bank_summary(dataset_name, dataset, cfg, arrays, structure)
        structure["actual_relation_bank"] = bank_summary
        dataset_summaries[dataset_name] = structure
        np.savez(
            REDUNDANCY_ROOT / f"{dataset_name}_pair_indices.npz",
            **arrays,
            path_order_sha256=np.asarray(structure["path_order_sha256"]),
        )
        write_csv(
            REDUNDANCY_ROOT / f"{dataset_name}_state_relations.csv",
            structure["state_relation_rows"],
        )
        structure_rows.append({
            "dataset": dataset_name,
            **{key: value for key, value in structure.items() if key != "state_relation_rows"},
        })
        write_json(SOURCE_ROOT / f"{dataset_name}_dataset_structure.json", structure)

    source_rows: list[dict[str, Any]] = []
    source_payload: dict[str, Any] = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "source_sha256": {
            str(path): sha256_file(path) for path in required_global
        },
        "global_source_files": [source_file_record(path) for path in required_global],
        "dataset_summaries": dataset_summaries,
        "sources": {},
        "source_verification_pass": True,
        "training_started": False,
        "optimizer_step_called": False,
        "query_gallery_used_for_diagnostic_construction": False,
    }
    for spec_id, spec in SPECS.items():
        required = [
            spec["model"], spec["state"], spec["config"], spec["run_manifest"],
            spec["checkpoint"], spec["amp"], spec["metrics"],
        ]
        missing_spec = [str(path) for path in required if not path.is_file()]
        if missing_spec:
            raise RuntimeError(f"{spec_id} source audit missing: {', '.join(missing_spec)}")
        checkpoint_meta = json.loads(spec["checkpoint"].read_text())
        if int(checkpoint_meta.get("epoch", -1)) != 60:
            raise RuntimeError(f"{spec_id} checkpoint epoch is not 60")
        model_state = torch.load(spec["model"], map_location="cpu", weights_only=False)
        if not isinstance(model_state, dict) or len(model_state) != 515:
            raise RuntimeError(f"{spec_id} model state does not contain the expected 515 keys")
        del model_state
        dataset_summary = dataset_summaries[spec["dataset"]]
        source = {
            "spec_id": spec_id,
            "dataset": spec["dataset"],
            "seed": spec["seed"],
            "role": spec["role"],
            "arm": spec["arm"],
            "phase": spec["phase"],
            "epoch": 60,
            "checkpoint_path": str(spec["model"]),
            "rank_state_epoch060_path": str(spec["state"]),
            "config_path": str(spec["config"]),
            "run_manifest_path": str(spec["run_manifest"]),
            "checkpoint_metadata_path": str(spec["checkpoint"]),
            "amp_path": str(spec["amp"]),
            "metrics_path": str(spec["metrics"]),
            "checkpoint_metadata": checkpoint_meta,
            "file_audit": [source_file_record(path) for path in required],
            "dataset_root": "/home/data/LTCC_ReID" if spec["dataset"] == "ltcc" else "/home/data/prcc",
            "feature_source": spec["feature_source"],
            "relation_implementation": str(FROZEN_METHOD if spec["dataset"] == "ltcc" else PRCC_METHOD),
            "relation_implementation_sha256": sha256_file(FROZEN_METHOD if spec["dataset"] == "ltcc" else PRCC_METHOD),
            "lambda_rel_applied": spec["lambda_applied"],
            "relation_bank_policy": spec["relation_bank_policy"],
            "sampler": spec["sampler"],
            "batch_construction": "same frozen phase runner metadata; diagnostic uses train-only metadata and matched rank-0 sampler stream",
            "amp_settings": json.loads(spec["amp"].read_text()),
            "train_images": dataset_summary["train_images"],
            "train_pids": dataset_summary["pids"],
            "train_clothing_states": dataset_summary["clothing_states"],
        }
        source_payload["sources"][spec_id] = source
        source_rows.append(source)
    source_payload["source_verification_pass"] = True
    write_json(SOURCE_ROOT / "source_audit.json", source_payload)
    write_csv(SOURCE_ROOT / "source_audit.csv", source_rows)
    write_csv(REDUNDANCY_ROOT / "dataset_structure.csv", structure_rows)
    write_source_report(source_payload)
    return source_payload


def write_source_report(payload: dict[str, Any]) -> None:
    lines = [
        "# PHASE 7B Source Audit",
        "",
        "Source verification status: PASS.",
        "",
        "This audit records the exact frozen Phase 6C seed1234 LTCC pair, Phase 6C-R seed4321 LTCC pair, and Phase 7A seed1234 PRCC pair. It does not train, update, or alter any checkpoint.",
        "",
        "| Spec | Dataset | Seed | Role | Epoch | Final model | Config | Relation source | Lambda applied | Sampler |",
        "|---|---|---:|---|---:|---|---|---|---:|---|",
    ]
    for spec_id, source in payload["sources"].items():
        lines.append(
            f"| {spec_id} | {source['dataset']} | {source['seed']} | {source['role']} | {source['epoch']} | "
            f"{source['checkpoint_path']} | {source['config_path']} | {source['relation_implementation']} | "
            f"{source['lambda_rel_applied']} | {source['sampler']} |"
        )
    lines += [
        "",
        "## Frozen method",
        "",
        f"- LTCC method document: {FROZEN_METHOD}; SHA256 {sha256_file(FROZEN_METHOD)}.",
        f"- PRCC method document: {PRCC_METHOD}; SHA256 {sha256_file(PRCC_METHOD)}.",
        f"- Executed Phase6B relation code: {PHASE6B_SCRIPT}; SHA256 {sha256_file(PHASE6B_SCRIPT)}.",
        "- The relation representation is the normalized identity feature feat[0]. The pair loss is 1 minus cosine for same-PID/different-clothes pairs, with uniform weight 1 for the uniform-relation arm and lambda 0.1 when applied.",
        "- For the utility audit, headroom and redundancy use every valid train-only cross-clothes pair. Gradient probes use the actual frozen candidate lookup: compact LTCC Phase6B bank and full PRCC Phase7A bank.",
        "",
        "## Dataset-only source checks",
        "",
    ]
    for name, summary in payload["dataset_summaries"].items():
        lines += [
            f"### {name.upper()}",
            "",
            f"- Train images/PIDs/clothing states: {summary['train_images']} / {summary['pids']} / {summary['clothing_states']}.",
            f"- Valid changing-clothes PIDs: {summary['valid_changing_clothes_pids']}.",
            f"- All valid image-level cross-clothes pairs: {summary['image_pair_count']}.",
            f"- Unique PID/clothing-state relations: {summary['unique_state_relations']}.",
            f"- Actual frozen relation-bank pairs: {summary['actual_relation_bank']['bank_pair_count']}; coverage of valid pairs: {summary['actual_relation_bank']['bank_pair_coverage']}.",
        ]
    lines += [
        "",
        "Source verification PASS; no diagnostic construction uses query/gallery/test data.",
    ]
    SOURCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_path(spec_id: str) -> Path:
    return HEADROOM_ROOT / f"{spec_id}_train_features.npz"


def extract_features(spec_id: str, batch_size: int = 64) -> dict[str, Any]:
    device = require_cuda()
    ensure_dirs()
    source_audit_path = SOURCE_ROOT / "source_audit.json"
    if not source_audit_path.is_file():
        raise RuntimeError("run --mode audit before feature extraction")
    path = feature_path(spec_id)
    if path.is_file():
        raise RuntimeError(f"refusing to overwrite existing feature artifact: {path}")
    set_seed(SPECS[spec_id]["seed"])
    model, dataset, cfg = load_model(spec_id, device, train=False)
    _, transform_test = (
        base.build_img_transforms(cfg) if SPECS[spec_id]["dataset"] == "ltcc"
        else phase7a.build_img_transforms(cfg)
    )
    image_dataset = ImageDataset(dataset.train, transform=transform_test)
    loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            camids = batch["camid"].to(device, non_blocking=True)
            output = model(images, cam_label=camids)
            if isinstance(output, (tuple, list)):
                output = output[-1]
            if not torch.is_tensor(output):
                raise RuntimeError(f"unexpected eval output for {spec_id}: {type(output)}")
            values = F.normalize(output.float(), dim=-1).cpu().numpy().astype(np.float32)
            blocks.append(values)
            if batch_index % 25 == 0:
                print(json.dumps({"spec": spec_id, "batches": batch_index, "images": batch_index * batch_size}))
    features = np.concatenate(blocks, axis=0)
    if len(features) != len(dataset.train):
        raise RuntimeError(f"{spec_id} feature count mismatch: {len(features)} != {len(dataset.train)}")
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    paths = np.asarray([str(item["image_path"]) for item in dataset.train])
    np.savez(path, features=features, pids=pids, clothes=clothes, paths=paths)
    result = {
        "spec_id": spec_id,
        "path": str(path),
        "count": len(features),
        "dim": int(features.shape[1]),
        "feature_source": SPECS[spec_id]["feature_source"],
        "train_only": True,
        "device": str(device),
        "model_forward_on_ppu": True,
        "sha256": sha256_file(path),
    }
    write_json(HEADROOM_ROOT / f"{spec_id}_feature_manifest.json", result)
    del loader, image_dataset, model, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_features(spec_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = feature_path(spec_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        pids = np.asarray(archive["pids"], dtype=np.int64)
        clothes = np.asarray(archive["clothes"], dtype=np.int64)
        paths = np.asarray(archive["paths"])
    norms = np.linalg.norm(features, axis=1)
    if not np.all(np.isfinite(features)) or not np.allclose(norms, 1.0, atol=2e-4):
        raise RuntimeError(f"invalid normalized feature bank: {spec_id}")
    return features, pids, clothes, paths


def numeric_stats(values: np.ndarray, prefix: str = "") -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}{key}": None for key in ("count", "mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max")}
    return {
        f"{prefix}count": int(len(values)),
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}std": float(np.std(values)),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}p10": float(np.percentile(values, 10)),
        f"{prefix}p25": float(np.percentile(values, 25)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p75": float(np.percentile(values, 75)),
        f"{prefix}p90": float(np.percentile(values, 90)),
        f"{prefix}max": float(np.max(values)),
    }


def geometry_metrics(spec_id: str) -> dict[str, Any]:
    features, pids, clothes, _paths = load_features(spec_id)
    dataset_name = SPECS[spec_id]["dataset"]
    pair_path = REDUNDANCY_ROOT / f"{dataset_name}_pair_indices.npz"
    with np.load(pair_path, allow_pickle=False) as archive:
        cross_a = np.asarray(archive["cross_a"], dtype=np.int64)
        cross_b = np.asarray(archive["cross_b"], dtype=np.int64)
        same_a = np.asarray(archive["same_a"], dtype=np.int64)
        same_b = np.asarray(archive["same_b"], dtype=np.int64)
    cross_scores = np.sum(features[cross_a] * features[cross_b], axis=1, dtype=np.float64)
    same_scores = np.sum(features[same_a] * features[same_b], axis=1, dtype=np.float64)
    unique_pids = np.unique(pids)
    centers = np.zeros((len(unique_pids), features.shape[1]), dtype=np.float32)
    pid_to_center = {}
    for pos, pid in enumerate(unique_pids.tolist()):
        center = features[pids == pid].mean(axis=0)
        center = center / max(float(np.linalg.norm(center)), 1e-12)
        centers[pos] = center
        pid_to_center[int(pid)] = pos
    image_to_pid_pos = np.asarray([pid_to_center[int(pid)] for pid in pids], dtype=np.int64)
    negative_matrix = features @ centers.T
    negative_matrix[np.arange(len(features)), image_to_pid_pos] = -np.inf
    nearest_negative = np.max(negative_matrix, axis=1)
    cross_negative = (nearest_negative[cross_a] + nearest_negative[cross_b]) / 2.0
    cross_margin = cross_scores - cross_negative

    state_centers: dict[tuple[int, int], np.ndarray] = {}
    for pid in unique_pids.tolist():
        for cloth in np.unique(clothes[pids == pid]).tolist():
            center = features[(pids == pid) & (clothes == cloth)].mean(axis=0)
            state_centers[(int(pid), int(cloth))] = center / max(float(np.linalg.norm(center)), 1e-12)
    state_dispersion: list[float] = []
    state_dispersion_by_pid: dict[int, float] = {}
    for pid in unique_pids.tolist():
        state_ids = sorted(cloth for (state_pid, cloth) in state_centers if state_pid == int(pid))
        values = []
        for pos, first in enumerate(state_ids):
            for second in state_ids[pos + 1:]:
                values.append(1.0 - float(np.dot(state_centers[(int(pid), first)], state_centers[(int(pid), second)])))
        if values:
            state_dispersion.extend(values)
            state_dispersion_by_pid[int(pid)] = float(np.mean(values))

    cross_pid = pids[cross_a]
    per_pid: dict[int, dict[str, Any]] = {}
    for pid in sorted(set(cross_pid.tolist())):
        mask = cross_pid == pid
        per_pid[int(pid)] = {
            "pid": int(pid),
            "cross_pair_count": int(np.sum(mask)),
            "cross_similarity": float(np.mean(cross_scores[mask])),
            "nearest_negative_similarity": float(np.mean(cross_negative[mask])),
            "identity_margin": float(np.mean(cross_margin[mask])),
            "state_center_dispersion": state_dispersion_by_pid.get(int(pid), None),
            "num_clothes": int(len(np.unique(clothes[pids == pid]))),
            "num_images": int(np.sum(pids == pid)),
            "pair_multiplicity": int(np.sum(mask)),
        }
    result = {
        "spec_id": spec_id,
        "dataset": dataset_name,
        "seed": SPECS[spec_id]["seed"],
        "role": SPECS[spec_id]["role"],
        "feature_path": str(feature_path(spec_id)),
        "geometry_source": "train-only; no query/gallery/test",
        "headroom": {
            **numeric_stats(cross_scores, "cross_similarity_"),
            **numeric_stats(1.0 - cross_scores, "raw_relation_loss_"),
            **numeric_stats(same_scores, "same_similarity_"),
            **numeric_stats(1.0 - same_scores, "same_raw_relation_loss_"),
            "cross_clothes_gap_mean": float(np.mean(same_scores) - np.mean(cross_scores)),
        },
        "geometry": {
            **numeric_stats(nearest_negative, "nearest_negative_image_"),
            **numeric_stats(cross_negative, "nearest_negative_cross_pair_"),
            **numeric_stats(cross_margin, "identity_margin_"),
            "state_center_dispersion_count": len(state_dispersion),
            "state_center_dispersion_mean": float(np.mean(state_dispersion)) if state_dispersion else None,
            "state_center_dispersion_std": float(np.std(state_dispersion)) if state_dispersion else None,
            "state_center_dispersion_median": float(np.median(state_dispersion)) if state_dispersion else None,
        },
        "per_pid": per_pid,
    }
    return result


def write_headroom_and_geometry() -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    headroom_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for spec_id in SPECS:
        result = geometry_metrics(spec_id)
        geometry[spec_id] = result
        write_json(GEOMETRY_ROOT / f"{spec_id}.json", result)
        h = result["headroom"]
        headroom_rows.append({"spec_id": spec_id, "dataset": result["dataset"], "seed": result["seed"], "role": result["role"], **h})
        g = result["geometry"]
        geometry_rows.append({"spec_id": spec_id, "dataset": result["dataset"], "seed": result["seed"], "role": result["role"], **g})
    write_csv(HEADROOM_ROOT / "headroom_summary.csv", headroom_rows)
    write_csv(GEOMETRY_ROOT / "geometry_summary.csv", geometry_rows)

    delta_rows: list[dict[str, Any]] = []
    pid_rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for baseline_id, relation_id in GEOMETRY_PAIRS:
        b = geometry[baseline_id]
        r = geometry[relation_id]
        delta = {
            "dataset": b["dataset"],
            "baseline_spec": baseline_id,
            "relation_spec": relation_id,
            "seed": b["seed"],
            "delta_cross_similarity_mean": r["headroom"]["cross_similarity_mean"] - b["headroom"]["cross_similarity_mean"],
            "delta_same_similarity_mean": r["headroom"]["same_similarity_mean"] - b["headroom"]["same_similarity_mean"],
            "delta_nearest_negative_cross_pair_mean": r["geometry"]["nearest_negative_cross_pair_mean"] - b["geometry"]["nearest_negative_cross_pair_mean"],
            "delta_identity_margin_mean": r["geometry"]["identity_margin_mean"] - b["geometry"]["identity_margin_mean"],
            "delta_state_center_dispersion_mean": r["geometry"]["state_center_dispersion_mean"] - b["geometry"]["state_center_dispersion_mean"],
        }
        delta_rows.append(delta)
        for pid, row in b["per_pid"].items():
            rr = r["per_pid"].get(pid)
            if rr is None:
                continue
            pid_rows.append({
                "dataset": b["dataset"],
                "baseline_spec": baseline_id,
                "relation_spec": relation_id,
                "seed": b["seed"],
                "pid": pid,
                "baseline_cross_similarity": row["cross_similarity"],
                "baseline_state_center_dispersion": row["state_center_dispersion"],
                "baseline_identity_margin": row["identity_margin"],
                "num_clothes": row["num_clothes"],
                "num_images": row["num_images"],
                "pair_multiplicity": row["pair_multiplicity"],
                "delta_cross_similarity": rr["cross_similarity"] - row["cross_similarity"],
                "delta_state_center_dispersion": rr["state_center_dispersion"] - row["state_center_dispersion"],
                "delta_nearest_negative_similarity": rr["nearest_negative_similarity"] - row["nearest_negative_similarity"],
                "delta_identity_margin": rr["identity_margin"] - row["identity_margin"],
            })
    write_csv(GEOMETRY_ROOT / "geometry_deltas.csv", delta_rows)
    write_csv(PID_ROOT / "per_pid_geometry.csv", pid_rows)

    def corr(x: list[float], y: list[float]) -> dict[str, Any]:
        x_arr = np.asarray([v for v in x if v is not None], dtype=np.float64)
        y_arr = np.asarray([v for v in y if v is not None], dtype=np.float64)
        if len(x_arr) != len(y_arr) or len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return {"n": int(min(len(x_arr), len(y_arr))), "pearson": None, "spearman": None}
        pearson = float(np.corrcoef(x_arr, y_arr)[0, 1])
        rx = np.argsort(np.argsort(x_arr))
        ry = np.argsort(np.argsort(y_arr))
        spearman = float(np.corrcoef(rx, ry)[0, 1])
        return {"n": len(x_arr), "pearson": pearson, "spearman": spearman}

    for dataset_name in ("ltcc", "prcc"):
        rows = [row for row in pid_rows if row["dataset"] == dataset_name]
        correlations[dataset_name] = {
            "baseline_cross_similarity_vs_delta_cross_similarity": corr(
                [row["baseline_cross_similarity"] for row in rows],
                [row["delta_cross_similarity"] for row in rows],
            ),
            "baseline_cross_headroom_vs_delta_cross_similarity": corr(
                [1.0 - row["baseline_cross_similarity"] for row in rows],
                [row["delta_cross_similarity"] for row in rows],
            ),
            "baseline_state_dispersion_vs_delta_identity_margin": corr(
                [row["baseline_state_center_dispersion"] for row in rows],
                [row["delta_identity_margin"] for row in rows],
            ),
        }
    write_json(PID_ROOT / "per_pid_correlations.json", correlations)
    write_json(GEOMETRY_ROOT / "geometry_results.json", geometry)
    return {"geometry": geometry, "delta_rows": delta_rows, "pid_rows": pid_rows, "correlations": correlations}


def headroom_tables(geometry: dict[str, Any]) -> dict[str, Any]:
    baseline_ids = ["ltcc_b0_seed1234", "ltcc_r0_seed4321", "prcc_p0_seed1234"]
    ltcc = [geometry[s]["headroom"] for s in baseline_ids[:2]]
    prcc = geometry[baseline_ids[2]]["headroom"]
    aggregate = {}
    for key in ltcc[0]:
        values = [item[key] for item in ltcc if isinstance(item[key], (int, float))]
        if not values:
            aggregate[key] = None
        elif key.endswith("_count"):
            aggregate[key] = int(sum(values))
        else:
            aggregate[key] = float(np.mean(values))
    headroom_rows = [
        {"dataset": "LTCC seed1234", "spec": baseline_ids[0], **geometry[baseline_ids[0]]["headroom"]},
        {"dataset": "LTCC seed4321", "spec": baseline_ids[1], **geometry[baseline_ids[1]]["headroom"]},
        {"dataset": "LTCC two-frozen-seed mean", "spec": "aggregate", **aggregate},
        {"dataset": "PRCC seed1234", "spec": baseline_ids[2], **prcc},
    ]
    write_csv(TABLE_ROOT / "relation_headroom_comparison.csv", headroom_rows)
    return {"ltcc_aggregate": aggregate, "prcc": prcc, "rows": headroom_rows}


def build_probe_loader(spec_id: str, dataset: Any, cfg: Any) -> Any:
    spec = SPECS[spec_id]
    if spec["dataset"] == "ltcc":
        transform_train, _ = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        active = phase6b.full_active_state(dataset)
        loader, _meta = base.make_epoch_loader(dataset, transform_train, active, index_map, epoch=60, rank=0)
        return loader
    transform_train, _ = phase7a.build_img_transforms(cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train, num_instances=phase7a.NUM_INSTANCES,
        num_replicas=phase7a.WORLD_SIZE, rank=0, seed=spec["seed"],
    )
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sampler,
        batch_size=phase7a.BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def batch_has_relation(batch: dict[str, Any], bank: dict[str, Any]) -> bool:
    pids = [int(value) for value in batch["pid"].tolist()]
    clothes = [int(value) for value in batch["clothes_id"].tolist()]
    paths = [str(path) for path in batch["image_path"]]
    indices = [bank["path_to_index"][path] for path in paths]
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
            if pids[i] == pids[j] and clothes[i] != clothes[j] and key in bank["exact_lookup"]:
                return True
    return False


def gradient_group_names(model: torch.nn.Module) -> dict[str, set[str]]:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    block_ids = sorted({
        int(name.split(".")[1]) for name in names
        if name.startswith("blocks.") and len(name.split(".")) > 1 and name.split(".")[1].isdigit()
    })
    late_ids = set(block_ids[-3:])
    return {
        "all": set(names),
        "late_eva": {
            name for name in names
            if name.startswith("blocks.") and name.split(".")[1].isdigit()
            and int(name.split(".")[1]) in late_ids
        },
        "id_head": {name for name in names if name.startswith("head.")},
        "bio_projection_head": {
            name for name in names
            if name.startswith("head_bio") or name.startswith("head_clip_bio") or name.startswith("norm_bio")
        },
    }


def gradient_group_values(
    names: list[str],
    base_grads: tuple[torch.Tensor | None, ...],
    rel_grads: tuple[torch.Tensor | None, ...],
    selected: set[str],
) -> dict[str, float | None]:
    base_sq = 0.0
    rel_sq = 0.0
    dot = 0.0
    total_sq = 0.0
    for name, bg, rg in zip(names, base_grads, rel_grads):
        if name not in selected:
            continue
        if bg is None and rg is None:
            continue
        if bg is None:
            bg = torch.zeros_like(rg)
        if rg is None:
            rg = torch.zeros_like(bg)
        bg = bg.detach().float()
        rg = rg.detach().float()
        base_sq += float(torch.sum(bg * bg).detach().cpu())
        rel_sq += float(torch.sum(rg * rg).detach().cpu())
        dot += float(torch.sum(bg * rg).detach().cpu())
        total = bg + 0.1 * rg
        total_sq += float(torch.sum(total * total).detach().cpu())
    norm_base = math.sqrt(max(base_sq, 0.0))
    norm_rel = math.sqrt(max(rel_sq, 0.0))
    norm_total = math.sqrt(max(total_sq, 0.0))
    eps = 1e-12
    return {
        "norm_base": norm_base,
        "norm_rel": norm_rel,
        "norm_ratio": norm_rel / (norm_base + eps),
        "alignment_rel": dot / (norm_base * norm_rel + eps) if norm_base > eps and norm_rel > eps else None,
        "alignment_total": (base_sq + 0.1 * dot) / (norm_base * norm_total + eps) if norm_base > eps and norm_total > eps else None,
    }


def gradient_probe(spec_id: str, target_active_batches: int = 16) -> dict[str, Any]:
    device = require_cuda()
    ensure_dirs()
    if not (SOURCE_ROOT / "source_audit.json").is_file():
        raise RuntimeError("run --mode audit before gradient probes")
    set_seed(SPECS[spec_id]["seed"])
    model, dataset, cfg = load_model(spec_id, device, train=True)
    model.requires_grad_(True)
    bank = relation_bank_for(SPECS[spec_id]["dataset"], dataset, cfg)
    loader = build_probe_loader(spec_id, dataset, cfg)
    loss_fn, _center = phase7a.make_loss(cfg, num_classes=dataset.num_train_pids)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _parameter in named_parameters]
    parameters = [parameter for _name, parameter in named_parameters]
    groups = gradient_group_names(model)
    rows: list[dict[str, Any]] = []
    seen_batches = 0
    inspected_batches = 0
    for batch in loader:
        inspected_batches += 1
        if not batch_has_relation(batch, bank):
            if inspected_batches > 500:
                break
            continue
        seen_batches += 1
        images = batch["image"].to(device, non_blocking=True)
        pids = batch["pid"].to(device, non_blocking=True)
        camids = batch["camid"].to(device, non_blocking=True)
        clothes = batch["clothes_id"].to(device, non_blocking=True)
        captions = batch["caption_feature"].to(device, non_blocking=True)
        writer = base.MetricCapture()
        model.zero_grad(set_to_none=True)
        # Deliberately no autocast: these are unscaled FP32 autograd probes.
        score, feat = model(images, cam_label=camids)
        base_loss = loss_fn(
            score, feat, pids, camids, captions, clothes_ids=clothes,
            train_writer=writer, step=seen_batches,
        )
        relation_loss, relation_info = phase6b.relation_loss(
            feat, pids, clothes, [str(path) for path in batch["image_path"]],
            bank["path_to_index"], bank, "B1",
        )
        if not relation_info["active"]:
            continue
        base_grads = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
        rel_grads = torch.autograd.grad(relation_loss, parameters, retain_graph=False, allow_unused=True)
        row: dict[str, Any] = {
            "spec_id": spec_id,
            "dataset": SPECS[spec_id]["dataset"],
            "seed": SPECS[spec_id]["seed"],
            "batch_index": seen_batches,
            "inspected_batches": inspected_batches,
            "batch_size": len(pids),
            "relation_pairs": int(relation_info["pair_count"]),
            "base_loss": float(base_loss.detach().float().cpu()),
            "relation_loss": float(relation_loss.detach().float().cpu()),
            "lambda_rel": 0.1,
            "gradient_dtype": "float32",
            "autocast_enabled": False,
            "optimizer_step": False,
        }
        for group_name, selected in groups.items():
            values = gradient_group_values(names, base_grads, rel_grads, selected)
            row.update({f"{key}_{group_name}": value for key, value in values.items()})
        rows.append(row)
        del base_grads, rel_grads, relation_loss, base_loss, feat, score
        model.zero_grad(set_to_none=True)
        if seen_batches >= target_active_batches:
            break
    if len(rows) < target_active_batches:
        raise RuntimeError(f"{spec_id} only produced {len(rows)} active gradient probes")
    output = GRADIENT_ROOT / f"{spec_id}.csv"
    write_csv(output, rows)
    summary = {
        "spec_id": spec_id,
        "rows": len(rows),
        "target_active_batches": target_active_batches,
        "inspected_batches": inspected_batches,
        "device": str(device),
        "model_forward_on_ppu": True,
        "autocast_enabled": False,
        "optimizer_step": False,
        "gradient_dtype": "float32",
        "path": str(output),
        "sha256": sha256_file(output),
    }
    write_json(GRADIENT_ROOT / f"{spec_id}_manifest.json", summary)
    del loader, loss_fn, model, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def summarize_gradient_rows() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec_id in BASELINE_SPECS:
        rows.extend(read_csv(GRADIENT_ROOT / f"{spec_id}.csv"))
    write_csv(GRADIENT_ROOT / "gradient_utility.csv", rows)
    metric_names = ("norm_base", "norm_rel", "norm_ratio", "alignment_rel", "alignment_total")
    modules = ("late_eva", "id_head", "bio_projection_head")
    summary_rows: list[dict[str, Any]] = []
    for spec_id in BASELINE_SPECS:
        subset = [row for row in rows if row["spec_id"] == spec_id]
        for module in modules:
            summary = {"spec_id": spec_id, "dataset": SPECS[spec_id]["dataset"], "seed": SPECS[spec_id]["seed"], "module": module, "rows": len(subset)}
            for metric in metric_names:
                values = np.asarray([
                    float(row[f"{metric}_{module}"]) for row in subset
                    if row.get(f"{metric}_{module}") not in ("", "None", "nan", None)
                ], dtype=np.float64)
                summary.update(numeric_stats(values, metric + "_"))
            summary_rows.append(summary)
    for module in modules:
        subset = [row for row in rows if SPECS[row["spec_id"]]["dataset"] == "ltcc"]
        summary = {"spec_id": "ltcc_two_seed_mean", "dataset": "ltcc", "seed": "1234+4321", "module": module, "rows": len(subset)}
        for metric in metric_names:
            values = np.asarray([
                float(row[f"{metric}_{module}"]) for row in subset
                if row.get(f"{metric}_{module}") not in ("", "None", "nan", None)
            ], dtype=np.float64)
            summary.update(numeric_stats(values, metric + "_"))
        summary_rows.append(summary)
    write_csv(GRADIENT_ROOT / "gradient_summary.csv", summary_rows)
    return {"rows": rows, "summary_rows": summary_rows}


def mean_summary(summary_rows: list[dict[str, Any]], spec_id: str, module: str, metric: str) -> float | None:
    for row in summary_rows:
        if row["spec_id"] == spec_id and row["module"] == module:
            value = row.get(metric + "_mean")
            return float(value) if value not in (None, "", "None") else None
    return None


def format_metric(value: Any, digits: int = 6) -> str:
    if value in (None, "", "None"):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def make_plots(geometry: dict[str, Any], gradient: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(PLOT_ROOT / "plot_status.json", {"available": False, "error": str(exc)})
        return
    labels = ["LTCC B0\n1234", "LTCC R0\n4321", "PRCC P0\n1234"]
    baseline_ids = BASELINE_SPECS
    cross = [geometry[s]["headroom"]["cross_similarity_mean"] for s in baseline_ids]
    same = [geometry[s]["headroom"]["same_similarity_mean"] for s in baseline_ids]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, cross, width, label="cross-clothes")
    ax.bar(x + width / 2, same, width, label="same-clothes")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("Frozen Bio train relation similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "relation_similarity_comparison.png", dpi=160)
    plt.close(fig)

    modules = ["late_eva", "id_head", "bio_projection_head"]
    ltcc_values = [mean_summary(gradient["summary_rows"], "ltcc_two_seed_mean", module, "alignment_rel") for module in modules]
    prcc_values = [mean_summary(gradient["summary_rows"], "prcc_p0_seed1234", module, "alignment_rel") for module in modules]
    ltcc_values = [value if value is not None else np.nan for value in ltcc_values]
    prcc_values = [value if value is not None else np.nan for value in prcc_values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x[:3] - width / 2, ltcc_values, width, label="LTCC two-seed mean")
    ax.bar(x[:3] + width / 2, prcc_values, width, label="PRCC seed1234")
    ax.set_xticks(x[:3], modules)
    ax.set_ylabel("A_rel = cos(g_base, g_rel)")
    ax.set_title("Relation gradient alignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "gradient_alignment_comparison.png", dpi=160)
    plt.close(fig)
    write_json(PLOT_ROOT / "plot_status.json", {"available": True})


def classify_questions(
    structure: dict[str, Any],
    headroom: dict[str, Any],
    geometry_results: dict[str, Any],
    gradient: dict[str, Any],
) -> dict[str, Any]:
    ltcc = headroom["ltcc_aggregate"]
    prcc = headroom["prcc"]
    cross_diff = prcc["cross_similarity_mean"] - ltcc["cross_similarity_mean"]
    raw_loss_diff = prcc["raw_relation_loss_mean"] - ltcc["raw_relation_loss_mean"]
    gap_diff = prcc["cross_clothes_gap_mean"] - ltcc["cross_clothes_gap_mean"]
    q1 = "YES" if cross_diff >= 0.05 and raw_loss_diff < 0 else ("NO" if cross_diff <= 0.02 and raw_loss_diff >= 0 else "MIXED")

    ltcc_struct = structure["ltcc"]
    prcc_struct = structure["prcc"]
    ratio_factor = prcc_struct["pair_multiplicity_ratio"] / ltcc_struct["pair_multiplicity_ratio"]
    q2 = "YES" if ratio_factor >= 2.0 and prcc_struct["unique_state_relations"] < ltcc_struct["unique_state_relations"] else ("NO" if ratio_factor <= 1.25 else "MIXED")

    align_rows = gradient["summary_rows"]
    modules = ("late_eva", "id_head", "bio_projection_head")
    ltcc_align_values = [mean_summary(align_rows, "ltcc_two_seed_mean", module, "alignment_rel") for module in modules]
    prcc_align_values = [mean_summary(align_rows, "prcc_p0_seed1234", module, "alignment_rel") for module in modules]
    align_delta_by_module = {
        module: (prcc_value - ltcc_value if prcc_value is not None and ltcc_value is not None else None)
        for module, ltcc_value, prcc_value in zip(modules, ltcc_align_values, prcc_align_values)
    }
    finite_deltas = np.asarray([value for value in align_delta_by_module.values() if value is not None], dtype=np.float64)
    q3 = "YES" if len(finite_deltas) >= 2 and np.sum(finite_deltas <= -0.05) >= 2 else (
        "NO" if len(finite_deltas) >= 2 and np.sum(finite_deltas >= 0.05) >= 2 else "MIXED"
    )

    prcc_delta = next(row for row in geometry_results["delta_rows"] if row["dataset"] == "prcc")
    q4 = "YES" if prcc_delta["delta_cross_similarity_mean"] > 0 and prcc_delta["delta_identity_margin_mean"] <= 0 else (
        "NO" if prcc_delta["delta_cross_similarity_mean"] <= 0 and prcc_delta["delta_identity_margin_mean"] > 0 else "MIXED"
    )
    supported = [q1 == "YES", q2 == "YES", q3 == "YES", q4 == "YES"]
    q5 = "YES" if any(supported) else "MIXED"
    return {
        "Q1_PRCC_less_relation_headroom_than_LTCC": q1,
        "Q1_cross_similarity_difference_PRCC_minus_LTCC": cross_diff,
        "Q1_raw_relation_loss_difference_PRCC_minus_LTCC": raw_loss_diff,
        "Q1_cross_clothes_gap_difference_PRCC_minus_LTCC": gap_diff,
        "Q1_rule": "YES requires PRCC mean baseline cross similarity at least 0.05 higher and lower raw relation loss (1-cos); cross-clothes gap is reported as a secondary structural descriptor, not used to reverse the absolute headroom call.",
        "Q2_PRCC_more_pair_redundant": q2,
        "Q2_pair_multiplicity_factor_PRCC_over_LTCC": ratio_factor,
        "Q2_rule": "YES requires at least 2x pair multiplicity ratio and fewer unique state relations.",
        "Q3_PRCC_relation_gradient_less_aligned": q3,
        "Q3_alignment_delta_PRCC_minus_LTCC_by_module": align_delta_by_module,
        "Q3_rule": "YES requires PRCC A_rel at least 0.05 lower in at least two audited modules with defined cosine alignment; zero relation gradients are reported as NA, not treated as evidence.",
        "Q4_PRCC_positive_compactness_without_margin": q4,
        "Q4_PRCC_delta": prcc_delta,
        "Q5_mechanistic_difference_supported": q5,
    }


def write_final_report(
    source: dict[str, Any],
    structure: dict[str, Any],
    headroom: dict[str, Any],
    geometry_results: dict[str, Any],
    gradient: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any]:
    delta_rows = geometry_results["delta_rows"]
    ltcc_deltas = [row for row in delta_rows if row["dataset"] == "ltcc"]
    prcc_delta = next(row for row in delta_rows if row["dataset"] == "prcc")
    ltcc_cross_delta = float(np.mean([row["delta_cross_similarity_mean"] for row in ltcc_deltas]))
    ltcc_neg_delta = float(np.mean([row["delta_nearest_negative_cross_pair_mean"] for row in ltcc_deltas]))
    ltcc_margin_delta = float(np.mean([row["delta_identity_margin_mean"] for row in ltcc_deltas]))
    late_rows = read_csv(PHASE7A_ROOT / "late_window_analysis.csv")
    prcc_late = [row for row in late_rows if row.get("window") == "E51-E60" and row.get("kind") == "delta_P1_minus_P0"]
    prcc_late_delta = prcc_late[0] if prcc_late else {}
    primary_table = [
        ("cross-clothes similarity (Bio baseline)", headroom["ltcc_aggregate"]["cross_similarity_mean"], headroom["prcc"]["cross_similarity_mean"], "higher similarity means less raw positive-pair headroom"),
        ("same-clothes similarity (Bio baseline)", headroom["ltcc_aggregate"]["same_similarity_mean"], headroom["prcc"]["same_similarity_mean"], "train-only same-clothes positive compactness"),
        ("cross-clothes gap", headroom["ltcc_aggregate"]["cross_clothes_gap_mean"], headroom["prcc"]["cross_clothes_gap_mean"], "same minus cross similarity"),
        ("raw relation loss (1-cos), cross", headroom["ltcc_aggregate"]["raw_relation_loss_mean"], headroom["prcc"]["raw_relation_loss_mean"], "unweighted train-only raw relation deficit"),
        ("unique state relations", structure["ltcc"]["unique_state_relations"], structure["prcc"]["unique_state_relations"], "PID/clothing-state relation units"),
        ("image pair count", structure["ltcc"]["image_pair_count"], structure["prcc"]["image_pair_count"], "all valid train-only image pairs"),
        ("pair multiplicity ratio", structure["ltcc"]["pair_multiplicity_ratio"], structure["prcc"]["pair_multiplicity_ratio"], "image pairs per unique state relation"),
        ("Late-EVA A_rel", mean_summary(gradient["summary_rows"], "ltcc_two_seed_mean", "late_eva", "alignment_rel"), mean_summary(gradient["summary_rows"], "prcc_p0_seed1234", "late_eva", "alignment_rel"), "cos(g_base,g_rel)"),
        ("ID-head A_rel", mean_summary(gradient["summary_rows"], "ltcc_two_seed_mean", "id_head", "alignment_rel"), mean_summary(gradient["summary_rows"], "prcc_p0_seed1234", "id_head", "alignment_rel"), "cos(g_base,g_rel)"),
        ("Bio-projection A_rel", mean_summary(gradient["summary_rows"], "ltcc_two_seed_mean", "bio_projection_head", "alignment_rel"), mean_summary(gradient["summary_rows"], "prcc_p0_seed1234", "bio_projection_head", "alignment_rel"), "cos(g_base,g_rel)"),
        ("Delta cross similarity after relation", ltcc_cross_delta, prcc_delta["delta_cross_similarity_mean"], "relation model minus matched Bio"),
        ("Delta nearest-negative similarity", ltcc_neg_delta, prcc_delta["delta_nearest_negative_cross_pair_mean"], "relation model minus matched Bio"),
        ("Delta identity margin", ltcc_margin_delta, prcc_delta["delta_identity_margin_mean"], "cross positive minus nearest negative"),
    ]
    lines = [
        "# PHASE 7B — Cross-Dataset Relation Utility Audit",
        "",
        "## 1. Scope",
        "",
        "Diagnostic only. No new model was trained, no optimizer.step() was called, no checkpoint was changed, and no lambda, sampler, loss, architecture, SR3, H_ID, or Bio/Triplet component was modified. All relation construction and geometry use train data only; query/gallery/test are excluded.",
        "",
        "## 2. Exact frozen sources",
        "",
        f"Source audit: {SOURCE_REPORT}.",
        "",
        "| Spec | Dataset | Seed | Role | Epoch | Final model | Config |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for spec_id, item in source["sources"].items():
        lines.append(f"| {spec_id} | {item['dataset']} | {item['seed']} | {item['role']} | 60 | {item['checkpoint_path']} | {item['config_path']} |")
    lines += [
        "",
        "Frozen relation documents:",
        "",
        f"- LTCC: {FROZEN_METHOD}.",
        f"- PRCC: {PRCC_METHOD}.",
        "- Headroom/redundancy use every valid train-only same-PID/different-clothes pair. Gradient probes use the actual frozen Phase6B lookup: compact LTCC candidate bank and full PRCC bank.",
        "",
        "## 3. LTCC versus PRCC relation structure",
        "",
        "| Metric | LTCC | PRCC | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for key, ltcc_value, prcc_value, interpretation in [
        ("Train images", structure["ltcc"]["train_images"], structure["prcc"]["train_images"], "full train split"),
        ("PIDs", structure["ltcc"]["pids"], structure["prcc"]["pids"], "identity count"),
        ("Clothing states", structure["ltcc"]["clothing_states"], structure["prcc"]["clothing_states"], "PID/clothing states"),
        ("Valid changing-clothes PIDs", structure["ltcc"]["valid_changing_clothes_pids"], structure["prcc"]["valid_changing_clothes_pids"], "PIDs with at least two states"),
        ("Unique state relations", structure["ltcc"]["unique_state_relations"], structure["prcc"]["unique_state_relations"], "unique (PID, cloth_a, cloth_b)"),
        ("Image pair count", structure["ltcc"]["image_pair_count"], structure["prcc"]["image_pair_count"], "valid unordered image pairs"),
        ("Pair multiplicity ratio", structure["ltcc"]["pair_multiplicity_ratio"], structure["prcc"]["pair_multiplicity_ratio"], "image pairs / state relation"),
        ("Multiplicity median", structure["ltcc"]["multiplicity_median"], structure["prcc"]["multiplicity_median"], "pairs per state relation"),
        ("Top 1% pair share", structure["ltcc"]["top_1pct_pair_share"], structure["prcc"]["top_1pct_pair_share"], "share of image pairs"),
        ("Top 5% pair share", structure["ltcc"]["top_5pct_pair_share"], structure["prcc"]["top_5pct_pair_share"], "share of image pairs"),
        ("Top 10% pair share", structure["ltcc"]["top_10pct_pair_share"], structure["prcc"]["top_10pct_pair_share"], "share of image pairs"),
    ]:
        lines.append(f"| {key} | {ltcc_value} | {prcc_value} | {interpretation} |")
    lines += [
        "",
        f"Actual LTCC frozen bank coverage: {structure['ltcc']['actual_relation_bank']['bank_pair_count']} / {structure['ltcc']['image_pair_count']} = {structure['ltcc']['actual_relation_bank']['bank_pair_coverage']}.",
        f"Actual PRCC frozen bank coverage: {structure['prcc']['actual_relation_bank']['bank_pair_count']} / {structure['prcc']['image_pair_count']} = {structure['prcc']['actual_relation_bank']['bank_pair_coverage']}.",
        "",
        "## 4. Relation headroom (frozen Bio baselines)",
        "",
        "The two LTCC baseline rows are the frozen seed1234 and seed4321 checkpoints. The LTCC column below is their arithmetic two-checkpoint mean; it is not a statistical estimate.",
        "",
        "| Metric | LTCC | PRCC | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for key, ltcc_value, prcc_value, interpretation in primary_table[:4]:
        lines.append(f"| {key} | {ltcc_value:.6f} | {prcc_value:.6f} | {interpretation} |")
    lines += [
        "",
        "Full distribution statistics are in relation_headroom/headroom_summary.csv. Raw relation loss is 1 minus cosine and is reported for every valid train-only cross-clothes image pair.",
        "",
        "## 5. Relation redundancy",
        "",
        f"LTCC has {structure['ltcc']['unique_state_relations']} unique state relations and {structure['ltcc']['image_pair_count']} image pairs; pair multiplicity ratio {structure['ltcc']['pair_multiplicity_ratio']:.3f}.",
        f"PRCC has {structure['prcc']['unique_state_relations']} unique state relations and {structure['prcc']['image_pair_count']} image pairs; pair multiplicity ratio {structure['prcc']['pair_multiplicity_ratio']:.3f}.",
        f"PRCC/LTCC pair-multiplicity ratio factor: {questions['Q2_pair_multiplicity_factor_PRCC_over_LTCC']:.3f}.",
        "",
        "## 6. Relation gradient utility",
        "",
        "Each baseline checkpoint was probed on multiple matched train micro-batches with an active frozen relation pair. Forward and autograd ran on PPU in unscaled FP32 with no optimizer object or optimizer step. A_rel = cos(g_base, g_rel); A_total = cos(g_base, g_base + 0.1 g_rel).",
        "",
        "| Module | LTCC A_rel mean | PRCC A_rel mean | LTCC norm_rel mean | PRCC norm_rel mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for module, label in [("late_eva", "Late-EVA"), ("id_head", "ID head"), ("bio_projection_head", "Bio projection/head")]:
        lines.append(
            f"| {label} | {format_metric(mean_summary(gradient['summary_rows'], 'ltcc_two_seed_mean', module, 'alignment_rel'))} | "
            f"{format_metric(mean_summary(gradient['summary_rows'], 'prcc_p0_seed1234', module, 'alignment_rel'))} | "
            f"{format_metric(mean_summary(gradient['summary_rows'], 'ltcc_two_seed_mean', module, 'norm_rel'))} | "
            f"{format_metric(mean_summary(gradient['summary_rows'], 'prcc_p0_seed1234', module, 'norm_rel'))} |"
        )
    lines += [
        "",
        "Norms are descriptive only. The primary utility signal is alignment, not gradient magnitude.",
        "",
        "## 7. Geometry before/after frozen training",
        "",
        "| Metric | LTCC | PRCC | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for key, ltcc_value, prcc_value, interpretation in primary_table[4:]:
        lines.append(f"| {key} | {format_metric(ltcc_value)} | {format_metric(prcc_value)} | {interpretation} |")
    lines += [
        "",
        "Per-checkpoint geometry and per-PID deltas are in geometry_comparison/ and per_pid_analysis/.",
        f"Frozen LTCC mean deltas (relation minus Bio): cross similarity {ltcc_cross_delta:.6f}, nearest-negative similarity {ltcc_neg_delta:.6f}, identity margin {ltcc_margin_delta:.6f}.",
        f"Frozen PRCC delta (P1 minus P0): cross similarity {prcc_delta['delta_cross_similarity_mean']:.6f}, nearest-negative similarity {prcc_delta['delta_nearest_negative_cross_pair_mean']:.6f}, identity margin {prcc_delta['delta_identity_margin_mean']:.6f}.",
        "",
        "## 8. Per-PID benefit structure",
        "",
        "Per-PID rows use train-only cross-clothes pairs, state-center dispersion, nearest-negative PID-center similarity, image count, and pair multiplicity. Correlations are descriptive and are not significance tests.",
        "",
        f"Correlation artifact: {PID_ROOT / 'per_pid_correlations.json'}.",
        "",
        "## 9. Q1-Q5",
        "",
        f"- Q1. Does PRCC have less Relation headroom than LTCC? **{questions['Q1_PRCC_less_relation_headroom_than_LTCC']}**. PRCC minus LTCC baseline cross similarity = {questions['Q1_cross_similarity_difference_PRCC_minus_LTCC']:.6f}; raw relation-loss difference = {questions['Q1_raw_relation_loss_difference_PRCC_minus_LTCC']:.6f}; cross-gap difference = {questions['Q1_cross_clothes_gap_difference_PRCC_minus_LTCC']:.6f} (secondary descriptor).",
        f"- Q2. Is PRCC substantially more pair-redundant at image-pair level? **{questions['Q2_PRCC_more_pair_redundant']}**.",
        f"- Q3. Is Relation gradient less aligned with base identity optimization on PRCC? **{questions['Q3_PRCC_relation_gradient_less_aligned']}**.",
        f"- Q4. Does Uniform Relation improve positive compactness on PRCC without improving identity margin? **{questions['Q4_PRCC_positive_compactness_without_margin']}**.",
        f"- Q5. Can the LTCC-vs-PRCC performance difference be explained by at least one clearly supported mechanistic difference? **{questions['Q5_mechanistic_difference_supported']}**.",
        "",
        "PRCC Phase 7A late-window reference:",
        f"- E51-E60 delta Rank-1 AUC: {prcc_late_delta.get('auc_rank1', 'unavailable')} pp.",
        f"- E51-E60 delta mAP AUC: {prcc_late_delta.get('auc_mAP', 'unavailable')} pp.",
        "",
        "## 10. Single strongest supported explanation",
        "",
    ]
    if questions["Q2_PRCC_more_pair_redundant"] == "YES" and questions["Q4_PRCC_positive_compactness_without_margin"] == "YES":
        explanation = "PRCC presents a much smaller set of clothing-state relation units repeated across many image pairs, and the frozen relation arm increases positive cross-clothes compactness without increasing the nearest-negative identity margin. Uniform image-pair averaging can therefore spend most of its relation budget on redundant observations while failing to improve identity separation. This is a mechanistically supported diagnostic explanation, not causal proof."
        recommendation = "CLOTHING-STATE PROTOTYPE RELATION"
    elif questions["Q2_PRCC_more_pair_redundant"] == "YES":
        explanation = "The strongest supported difference is PRCC image-pair redundancy: many repeated image pairs represent fewer clothing-state relation units, so uniform image-pair supervision is structurally less diverse."
        recommendation = "CLOTHING-STATE PROTOTYPE RELATION"
    elif questions["Q4_PRCC_positive_compactness_without_margin"] == "YES":
        explanation = "The strongest supported difference is that PRCC relation training compresses same-ID cross-clothes positives but does not improve the nearest-negative identity margin."
        recommendation = "MARGIN-PRESERVING CROSS-CLOTHES RELATION"
    elif questions["Q3_PRCC_relation_gradient_less_aligned"] == "YES":
        explanation = "The strongest supported difference is lower relation-gradient alignment with the base identity objective on PRCC."
        recommendation = "GRADIENT-COMPATIBLE RELATION SUPERVISION"
    elif questions["Q1_PRCC_less_relation_headroom_than_LTCC"] == "YES":
        explanation = "The strongest supported difference is reduced PRCC relation headroom: the Bio baseline already has substantially higher same-ID cross-clothes similarity."
        recommendation = "SATURATION-AWARE RELATION LOSS"
    else:
        explanation = "No single mechanism meets the predeclared strong diagnostic rule; the observed difference remains mixed and should not be converted into a causal claim."
        recommendation = "STOP; retain the current frozen method and collect targeted evidence before implementation."
    lines += [
        explanation,
        "",
        "## 11. Next-method routing (recommendation only; not implemented)",
        "",
        f"Recommended next branch: **{recommendation}**.",
        "No branch is implemented in Phase 7B. If later authorized, the branch must preserve the current dataset splits, Bio/Triplet losses, and evaluation protocol. Do not choose a margin, threshold, gradient projection, sampler, or lambda value from this audit.",
        "",
        "## 12. STOP",
        "",
        "Phase 7B is complete. No Phase 7C, training, lambda tuning, new sampler, new loss, new architecture, SR3, H_ID weighting, or checkpoint modification was started.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: {MANIFEST_PATH}",
        f"- Source audit: {SOURCE_ROOT / 'source_audit.json'}",
        f"- Headroom: {HEADROOM_ROOT / 'headroom_summary.csv'}",
        f"- Redundancy: {REDUNDANCY_ROOT / 'dataset_structure.csv'}",
        f"- Gradient probes: {GRADIENT_ROOT / 'gradient_utility.csv'}",
        f"- Geometry: {GEOMETRY_ROOT / 'geometry_summary.csv'} and {GEOMETRY_ROOT / 'geometry_deltas.csv'}",
        f"- Per-PID: {PID_ROOT / 'per_pid_geometry.csv'} and {PID_ROOT / 'per_pid_correlations.json'}",
    ]
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"explanation": explanation, "recommended_next_branch": recommendation}


def analyze() -> dict[str, Any]:
    ensure_dirs()
    source = json.loads((SOURCE_ROOT / "source_audit.json").read_text())
    if not source.get("source_verification_pass"):
        raise RuntimeError("source audit did not pass")
    structure = source["dataset_summaries"]
    geometry_results = write_headroom_and_geometry()
    headroom = headroom_tables(geometry_results["geometry"])
    gradient = summarize_gradient_rows()
    questions = classify_questions(structure, headroom, geometry_results, gradient)
    make_plots(geometry_results["geometry"], gradient)
    routing = write_final_report(source, structure, headroom, geometry_results, gradient, questions)
    manifest = {
        "phase": PHASE,
        "mode": "DIAGNOSTIC_ONLY",
        "created_utc": utc_now(),
        "source_audit_pass": True,
        "training_started": False,
        "optimizer_step_called": False,
        "checkpoints_modified": False,
        "query_gallery_used_for_diagnostic_construction": False,
        "ppus_required": True,
        "allowed_frozen_specs": list(SPECS),
        "baseline_gradient_specs": BASELINE_SPECS,
        "geometry_pairs": GEOMETRY_PAIRS,
        "source_audit": str(SOURCE_ROOT / "source_audit.json"),
        "dataset_structure": str(REDUNDANCY_ROOT / "dataset_structure.csv"),
        "headroom": str(HEADROOM_ROOT / "headroom_summary.csv"),
        "gradient_utility": str(GRADIENT_ROOT / "gradient_utility.csv"),
        "geometry_summary": str(GEOMETRY_ROOT / "geometry_summary.csv"),
        "geometry_delta_path": str(GEOMETRY_ROOT / "geometry_deltas.csv"),
        "per_pid": str(PID_ROOT / "per_pid_geometry.csv"),
        "per_pid_correlations": str(PID_ROOT / "per_pid_correlations.json"),
        "report": str(FINAL_REPORT),
        "questions": questions,
        "routing": routing,
        "relation_structure": structure,
        "headroom_summary": headroom,
        "gradient_summary": gradient["summary_rows"],
        "geometry_delta_rows": geometry_results["delta_rows"],
        "frozen_phase7a_late_window": {
            "rank1_auc_delta_pp": -0.10348988903892575,
            "mAP_auc_delta_pp": -0.04013125828952724,
        },
        "stop": True,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def parse_spec_list(value: str | None) -> list[str]:
    if not value:
        return list(SPECS)
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in values if item not in SPECS]
    if unknown:
        raise ValueError(f"unknown specs: {unknown}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "extract", "gradient", "analyze"), required=True)
    parser.add_argument("--specs", default=None, help="comma-separated frozen spec IDs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-batches", type=int, default=16)
    args = parser.parse_args()
    if args.mode == "audit":
        result = audit_sources()
        print(json.dumps({
            "source_verification_pass": result["source_verification_pass"],
            "source_report": str(SOURCE_REPORT),
            "output": str(OUT_ROOT),
            "training_started": False,
        }, indent=2, sort_keys=True))
    elif args.mode == "extract":
        results = [extract_features(spec_id, args.batch_size) for spec_id in parse_spec_list(args.specs)]
        print(json.dumps(results, indent=2, sort_keys=True))
    elif args.mode == "gradient":
        results = [gradient_probe(spec_id, args.gradient_batches) for spec_id in parse_spec_list(args.specs)]
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        result = analyze()
        print(json.dumps({
            "report": str(FINAL_REPORT),
            "manifest": str(MANIFEST_PATH),
            "questions": result["questions"],
            "recommended_next_branch": result["routing"]["recommended_next_branch"],
            "stop": result["stop"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
