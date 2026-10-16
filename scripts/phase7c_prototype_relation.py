#!/usr/local/bin/python3
"""Phase 7C: PRCC clothing-state prototype relation.

The script has an explicit audit -> coverage -> prepare -> train -> gradient
-> analyze flow.  It never writes to Phase 5/6/7A/7B paths.  C0/C1/C2 use
the audited Phase 7A PRCC schedule and the existing Phase 7A common
initialization; C2 changes only the relation term from image pairs to
current-mini-batch clothing-state prototypes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as phase6b  # noqa: E402
import phase7a_prcc_relation_validation as phase7a  # noqa: E402


PHASE = "phase7c_prototype_relation"
SEED = 1234
WORLD_SIZE = 2
FORMAL_EPOCHS = 60
WARMUP_EPOCHS = 20
INITIAL_SCALE = 512
BATCH_SIZE = 16
NUM_INSTANCES = 4
EXPECTED_TRAIN_IMAGES = 17896
EXPECTED_TRAIN_PIDS = 150
EXPECTED_CLOTHING_STATES = 300
EXPECTED_STEPS_PER_EPOCH = 551
LAMBDA_REL = 0.1
PRIMARY_WINDOW = "E51-E60"
WINDOWS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60))
GEOMETRY_EPOCHS = (20, 40, 60)
GRADIENT_BATCHES = 16

OUT_ROOT = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
SOURCE_ROOT = OUT_ROOT / "source_audit"
FEASIBILITY_ROOT = OUT_ROOT / "feasibility"
TRAINING_ROOT = OUT_ROOT / "training"
BALANCE_ROOT = OUT_ROOT / "relation_balance"
GEOMETRY_ROOT = OUT_ROOT / "geometry"
GRADIENT_ROOT = OUT_ROOT / "gradient_diagnostic"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7C_MANIFEST.json"
REPORT_PATH = ROOT / "reports/PHASE_7C_CLOTHING_STATE_PROTOTYPE_RELATION.md"
SOURCE_REPORT = ROOT / "reports/PHASE_7C_SOURCE_AUDIT.md"

PHASE7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
PHASE7A_P0 = PHASE7A_ROOT / "P0_prcc_bio"
PHASE7A_P1 = PHASE7A_ROOT / "P1_prcc_uniform_relation"
PHASE7A_COMMON_INIT = PHASE7A_ROOT / "PRCC_COMMON_INIT.pth"
PHASE7A_COMMON_META = PHASE7A_ROOT / "PRCC_COMMON_INIT_HASHES.json"
PHASE7A_METHOD = PHASE7A_ROOT / "artifacts/frozen_relation_method.md"
PHASE6B_SCRIPT = ROOT / "scripts/phase6b_soft_relation_screen.py"
PHASE7A_SCRIPT = ROOT / "scripts/phase7a_prcc_relation_validation.py"
PRCC_ROOT = Path("/home/data/prcc")

RUN_DIRS = {
    "C0": TRAINING_ROOT / "C0_bio",
    "C1": TRAINING_ROOT / "C1_image_uniform_relation",
    "C2": TRAINING_ROOT / "C2_prototype_relation",
}
ARM_LABELS = {
    "C0": "PRCC matched Bio control",
    "C1": "PRCC image-level uniform relation",
    "C2": "PRCC clothing-state prototype relation",
}
ARM_PHASE6B = {"C0": "B0", "C1": "B1"}

METRIC_FIELDS = [
    "arm", "seed", "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss",
    "relation_loss", "relation_applied_loss", "relation_type", "relation_lambda", "total_loss",
    "lr", "lr_bias", "id_accuracy", "bio_accuracy", "planned_updates", "updates_completed",
    "amp_scale_min", "amp_scale_max", "amp_scale_end", "amp_skip", "skipped_steps",
    "found_inf_count", "nonfinite_events", "active_images", "active_pids", "active_clothing_states",
    "sampled_unique_images", "sampled_unique_pids", "sampled_unique_clothing_states",
    "image_relation_active_batches", "image_relation_batch_rate", "image_relations_per_epoch",
    "unique_image_relation_count", "prototype_valid_batches", "prototype_batch_rate",
    "prototype_relations_per_epoch", "unique_prototype_relation_count", "active_prototypes_per_epoch",
    "prototype_pid_coverage", "prototype_relation_pid_coverage", "prototype_mean_images_per_state",
    "prototype_median_images_per_state", "prototype_mean_similarity", "cc_rank1", "cc_rank5",
    "cc_rank10", "cc_mAP", "general_rank1", "general_rank5", "general_rank10", "general_mAP",
]
EXPOSURE_FIELDS = [
    "arm", "seed", "epoch", "all_images_available", "all_pids_available", "all_clothing_states_available",
    "sampled_unique_images", "sampled_unique_pids", "sampled_unique_clothing_states",
    "image_relation_active_batches", "image_relation_batch_rate", "image_relations_per_epoch",
    "unique_image_relation_count", "prototype_valid_batches", "prototype_batch_rate",
    "prototype_relations_per_epoch", "unique_prototype_relation_count", "active_prototypes_per_epoch",
    "prototype_pid_coverage", "prototype_relation_pid_coverage", "prototype_mean_images_per_state",
    "prototype_median_images_per_state", "prototype_mean_similarity", "planned_updates", "updates_completed",
]
AMP_FIELDS = [
    "arm", "seed", "epoch", "amp_skip", "skipped_steps", "found_inf_count", "nonfinite_events",
    "amp_scale_min", "amp_scale_max", "amp_scale_end", "planned_updates", "updates_completed",
]
AMP_EVENT_FIELDS = [
    "arm", "seed", "rank", "epoch", "iteration", "global_step", "skipped_step", "found_inf",
    "nonfinite_event", "amp_scale_before", "amp_scale_after",
]
SAMPLER_FIELDS = [
    "arm", "seed", "epoch", "rank", "sampler_epoch_used", "samples", "batches", "unique_images",
    "unique_pids", "unique_clothing_states", "plan_sha256",
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def state_hash(state: dict[str, torch.Tensor]) -> str:
    parts: dict[str, str] = {}
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        parts[key] = sha256_bytes(raw)
    return sha256_bytes(json.dumps(parts, sort_keys=True).encode("utf-8"))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SOURCE_ROOT, FEASIBILITY_ROOT, TRAINING_ROOT, BALANCE_ROOT,
                 GEOMETRY_ROOT, GRADIENT_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def require_ppu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 7C requires exactly {WORLD_SIZE} visible PPUs; CPU fallback is forbidden")


def cfg_for(run_dir: Path | None = None) -> Any:
    return phase7a.configure(run_dir or (TRAINING_ROOT / "config_probe"))


def dataset_for(cfg: Any) -> Any:
    return phase7a.build_dataset(cfg)


def p7a_file_audit(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def state_structure(dataset: Any) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any], list[dict[str, Any]]]:
    groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(dataset.train):
        groups[int(item["pid"])][int(item["clothes_id"])].append(index)
    sizes = [len(indices) for states in groups.values() for indices in states.values()]
    relation_rows: list[dict[str, Any]] = []
    for pid, states in sorted(groups.items()):
        clothes = sorted(states)
        for pos, cloth_a in enumerate(clothes):
            for cloth_b in clothes[pos + 1:]:
                relation_rows.append({
                    "pid": pid, "cloth_a": cloth_a, "cloth_b": cloth_b,
                    "images_cloth_a": len(states[cloth_a]), "images_cloth_b": len(states[cloth_b]),
                    "image_pair_count": len(states[cloth_a]) * len(states[cloth_b]),
                })
    stats = {
        "train_images": len(dataset.train),
        "pids": len(groups),
        "clothing_states": len(sizes),
        "changing_clothes_pids": sum(len(states) >= 2 for states in groups.values()),
        "state_size_count": len(sizes),
        "state_size_mean": float(np.mean(sizes)),
        "state_size_median": float(np.median(sizes)),
        "state_size_p25": float(np.percentile(sizes, 25)),
        "state_size_p75": float(np.percentile(sizes, 75)),
        "state_size_p90": float(np.percentile(sizes, 90)),
        "state_size_max": int(max(sizes)),
        "valid_prototype_relations": len(relation_rows),
        "valid_image_relations": int(sum(row["image_pair_count"] for row in relation_rows)),
        "clothes_per_pid_mean": float(np.mean([len(states) for states in groups.values()])),
        "clothes_per_pid_median": float(np.median([len(states) for states in groups.values()])),
    }
    return groups, stats, relation_rows


def audit_label_schema(dataset: Any) -> dict[str, Any]:
    path_to_camera = {str(item["image_path"]): Path(str(item["image_path"])).name[0] for item in dataset.train}
    by_pid: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in dataset.train:
        by_pid[int(item["pid"])][int(item["clothes_id"])].add(path_to_camera[str(item["image_path"])])
    per_pid: dict[str, Any] = {}
    safe = True
    for pid, states in sorted(by_pid.items()):
        mapped = {str(cloth): sorted(cameras) for cloth, cameras in states.items()}
        safe = safe and len(states) == 2
        safe = safe and all(set(cameras).issubset({"A", "B", "C"}) for cameras in states.values())
        safe = safe and any(set(cameras) & {"A", "B"} for cameras in states.values())
        safe = safe and any(set(cameras) == {"C"} for cameras in states.values())
        per_pid[str(pid)] = mapped
    return {"safe": safe, "semantic_rule": "PRCC train A/B share state; C is changed state under repository parser", "per_pid": per_pid}


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    cfg = cfg_for()
    dataset = dataset_for(cfg)
    groups, structure, relation_rows = state_structure(dataset)
    schema = audit_label_schema(dataset)
    if len(dataset.train) != EXPECTED_TRAIN_IMAGES or dataset.num_train_pids != EXPECTED_TRAIN_PIDS or dataset.num_train_clothes != EXPECTED_CLOTHING_STATES:
        raise RuntimeError(f"unexpected PRCC topology: {len(dataset.train)}, {dataset.num_train_pids}, {dataset.num_train_clothes}")
    source_runs = {
        "P0": PHASE7A_P0,
        "P1": PHASE7A_P1,
    }
    sources: dict[str, Any] = {}
    for arm, run_dir in source_runs.items():
        complete_path = run_dir / "RUN_COMPLETE.json"
        complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
        checkpoint = json.loads((run_dir / "checkpoint_epoch060.json").read_text()) if (run_dir / "checkpoint_epoch060.json").is_file() else {}
        manifest = json.loads((run_dir / "run_manifest.json").read_text()) if (run_dir / "run_manifest.json").is_file() else {}
        metrics = read_csv(run_dir / "metrics_per_epoch.csv") if (run_dir / "metrics_per_epoch.csv").is_file() else []
        expected = [run_dir / name for name in ("model_final.pth", "model_best.pth", "config.yml", "run_manifest.json", "checkpoint_epoch060.json", "AMP.json", "amp_audit.csv", "amp_skip_events.csv", "metrics_per_epoch.csv", "relation_exposure.csv", "relation_geometry.csv", "sampler_plan.csv", "RUN_COMPLETE.json")]
        files = [p7a_file_audit(path) for path in expected]
        if complete.get("epochs_completed") != FORMAL_EPOCHS or complete.get("rows") != FORMAL_EPOCHS or len(metrics) != FORMAL_EPOCHS:
            raise RuntimeError(f"Phase 7A {arm} is not a complete 60-epoch source")
        if checkpoint.get("epoch") != FORMAL_EPOCHS or checkpoint.get("global_step") != EXPECTED_STEPS_PER_EPOCH * FORMAL_EPOCHS:
            raise RuntimeError(f"Phase 7A {arm} checkpoint convention mismatch")
        sources[arm] = {
            "run_dir": str(run_dir), "arm": arm, "seed": SEED, "epoch": FORMAL_EPOCHS,
            "checkpoint": checkpoint, "run_manifest": manifest, "complete": complete,
            "files": files, "config_sha256": sha256_file(run_dir / "config.yml"),
            "lambda_rel": float(manifest.get("lambda_rel", 0.0 if arm == "P0" else LAMBDA_REL)),
            "batch_size": int(manifest.get("batch_size", BATCH_SIZE)),
            "num_instances": int(manifest.get("num_instances", NUM_INSTANCES)),
            "optimizer": manifest.get("optimizer"), "scheduler": manifest.get("scheduler"),
            "amp": manifest.get("amp"), "sampler": manifest.get("sampler", "DistributedRandomIdentitySampler"),
            "feature_source": "normalized identity feature feat[0]; eval feat_bio is the same feature",
            "relation_method": str(PHASE7A_METHOD),
        }
    common_meta = json.loads(PHASE7A_COMMON_META.read_text()) if PHASE7A_COMMON_META.is_file() else {}
    if not PHASE7A_COMMON_INIT.is_file() or not common_meta:
        raise RuntimeError("Phase 7A common initialization is missing")
    common_state = torch.load(PHASE7A_COMMON_INIT, map_location="cpu", weights_only=False)
    common_hash = state_hash(common_state)
    if common_hash != common_meta.get("state_hash") or sha256_file(PHASE7A_COMMON_INIT) != common_meta.get("state_sha256"):
        raise RuntimeError("Phase 7A common initialization integrity check failed")
    checks = {
        "source_runs_complete": True,
        "seed_1234": True,
        "epochs_60": True,
        "batch_size_16": all(item["batch_size"] == BATCH_SIZE for item in sources.values()),
        "num_instances_4": all(item["num_instances"] == NUM_INSTANCES for item in sources.values()),
        "lambda_p0_p1": sources["P0"]["lambda_rel"] == 0.0 and abs(sources["P1"]["lambda_rel"] - LAMBDA_REL) < 1e-12,
        "train_topology": True,
        "label_schema_safe": bool(schema["safe"]),
        "common_init_verified": True,
        "phase7a_method_exists": PHASE7A_METHOD.is_file(),
    }
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"Phase 7C source audit failed: {checks}")
    payload = {
        "phase": PHASE, "created_utc": utc_now(), "source_verification_pass": passed,
        "training_started": False, "query_gallery_used_for_relation": False,
        "phase7a_root": str(PHASE7A_ROOT), "sources": sources,
        "common_init": {"path": str(PHASE7A_COMMON_INIT), "meta_path": str(PHASE7A_COMMON_META), "sha256": sha256_file(PHASE7A_COMMON_INIT), "state_hash": common_hash, "initial_parameter_equality": common_meta.get("initial_parameter_equality")},
        "effective_config": {"config_sha256": sha256_bytes(cfg.dump().encode("utf-8")), "config": cfg.dump(), "optimizer": phase7a.optimizer_config(cfg), "scheduler": phase7a.scheduler_config(cfg), "amp": phase7a.amp_config(), "epochs": FORMAL_EPOCHS, "warmup_epochs": WARMUP_EPOCHS, "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH, "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES, "feature_source": "feat[0]"},
        "relation_implementation": {"phase6b_script": str(PHASE6B_SCRIPT), "phase6b_sha256": sha256_file(PHASE6B_SCRIPT), "phase7a_script": str(PHASE7A_SCRIPT), "phase7a_sha256": sha256_file(PHASE7A_SCRIPT), "method_doc": str(PHASE7A_METHOD), "method_doc_sha256": sha256_file(PHASE7A_METHOD), "lambda_rel": LAMBDA_REL, "image_relation": "1 - cosine(feat[0]_i, feat[0]_j), same PID/different clothes, uniform w=1"},
        "dataset": {"root": str(PRCC_ROOT), **structure, "label_schema": schema, "path_order_sha256": sha256_bytes("\n".join(str(item["image_path"]) for item in dataset.train).encode("utf-8"))},
        "state_relation_rows": relation_rows,
        "checks": checks,
    }
    write_json(SOURCE_ROOT / "source_audit.json", payload)
    write_json(FEASIBILITY_ROOT / "prototype_feasibility.json", {"phase": PHASE, "passed": passed, "train_only": True, "query_gallery_used": False, "dataset": payload["dataset"], "relations": relation_rows, "training_started": False})
    write_csv(FEASIBILITY_ROOT / "state_relation_structure.csv", relation_rows)
    lines = [
        "# PHASE 7C Source Audit", "", "Source verification: **PASS**.", "", 
        "Phase 7C uses a new output root and does not overwrite Phase 5, Phase 6, Phase 7A, or Phase 7B artifacts.", "",
        "## Exact Phase 7A sources", "",
        "| Arm | Run directory | Epoch | Seed | Model final | Config | Lambda |", "|---|---|---:|---:|---|---|---:|",
    ]
    for arm, item in sources.items():
        lines.append(f"| {arm} | {item['run_dir']} | {item['epoch']} | {item['seed']} | {item['checkpoint'].get('model', '')} | {item['run_dir']}/config.yml | {item['lambda_rel']} |")
    lines += ["", f"- Phase 7A common initialization: `{PHASE7A_COMMON_INIT}`; SHA256 `{payload['common_init']['sha256']}`; tensor hash `{common_hash}`.", f"- Frozen relation method: `{PHASE7A_METHOD}`; SHA256 `{payload['relation_implementation']['method_doc_sha256']}`.", f"- Executed relation code: `{PHASE6B_SCRIPT}`; SHA256 `{payload['relation_implementation']['phase6b_sha256']}`.", "- Relation feature: `feat[0]`, normalized before cosine/prototype construction.", "- Schedule: 60 epochs, 20-epoch linear warmup, 551 updates/rank/epoch, batch 16, NUM_INSTANCES 4, SGD and Phase 7A AMP.", "", "## Train-only feasibility", "", f"- PRCC train: {structure['train_images']} images, {structure['pids']} PIDs, {structure['clothing_states']} clothing states.", f"- Changing-clothes PIDs: {structure['changing_clothes_pids']}.", f"- Valid prototype relations: {structure['valid_prototype_relations']}.", f"- State-size mean/median/p25/p75/p90/max: {structure['state_size_mean']:.3f} / {structure['state_size_median']:.3f} / {structure['state_size_p25']:.3f} / {structure['state_size_p75']:.3f} / {structure['state_size_p90']:.3f} / {structure['state_size_max']}.", "- Label semantics: A/B same state and C changed state under the repository parser; schema check PASS.", "- Query/gallery were not used to construct relation labels or feasibility relations.", "", f"Machine-readable source audit: `{SOURCE_ROOT / 'source_audit.json'}`.", f"Machine-readable feasibility: `{FEASIBILITY_ROOT / 'prototype_feasibility.json'}`."]
    SOURCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_source_audit() -> dict[str, Any]:
    path = SOURCE_ROOT / "source_audit.json"
    if not path.is_file():
        raise RuntimeError("run --mode audit first")
    payload = json.loads(path.read_text())
    if not payload.get("source_verification_pass") or payload.get("training_started"):
        raise RuntimeError("source audit is not a clean PASS")
    return payload


def batch_exposure(records: list[dict[str, Any]], path_to_index: dict[str, int]) -> dict[str, Any]:
    by_pid: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for local, item in enumerate(records):
        by_pid[int(item["pid"])][int(item["clothes_id"])].append(local)
    image_count = 0
    image_keys: list[tuple[int, int]] = []
    image_state_keys: list[tuple[int, int, int]] = []
    proto_count = 0
    proto_keys: list[tuple[int, int, int]] = []
    active_proto = 0
    proto_pids: set[int] = set()
    proto_sizes: list[int] = []
    proto_sims_placeholder: list[float] = []
    for pid, states in by_pid.items():
        clothes = sorted(states)
        if len(clothes) < 2:
            continue
        active_proto += len(clothes)
        proto_pids.add(pid)
        for cloth in clothes:
            proto_sizes.append(len(states[cloth]))
        for pos, cloth_a in enumerate(clothes):
            for cloth_b in clothes[pos + 1:]:
                image_count += len(states[cloth_a]) * len(states[cloth_b])
                image_state_keys.extend([(pid, cloth_a, cloth_b)] * (len(states[cloth_a]) * len(states[cloth_b])))
                proto_count += 1
                proto_keys.append((pid, cloth_a, cloth_b))
                for left in states[cloth_a]:
                    for right in states[cloth_b]:
                        a = min(int(path_to_index[str(records[left]["image_path"])]), int(path_to_index[str(records[right]["image_path"])]))
                        b = max(int(path_to_index[str(records[left]["image_path"])]), int(path_to_index[str(records[right]["image_path"])]))
                        image_keys.append((a, b))
    return {"image_count": image_count, "image_keys": image_keys, "image_state_keys": image_state_keys, "proto_count": proto_count, "proto_keys": proto_keys, "active_proto": active_proto, "proto_pids": proto_pids, "proto_sizes": proto_sizes, "proto_sims": proto_sims_placeholder}


def coverage() -> dict[str, Any]:
    audit = load_source_audit()
    cfg = cfg_for()
    dataset = dataset_for(cfg)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    total_batches = 0
    image_active_batches = 0
    proto_active_batches = 0
    image_counts: list[int] = []
    proto_counts: list[int] = []
    active_proto_counts: list[int] = []
    proto_size_values: list[int] = []
    pids_seen: set[int] = set()
    epoch_rows: list[dict[str, Any]] = []
    c1_counter: Counter[tuple[int, int, int]] = Counter()
    c2_counter: Counter[tuple[int, int, int]] = Counter()
    state_keys_all: set[tuple[int, int, int]] = set()
    plan_hashes: dict[str, list[str]] = {"rank0": [], "rank1": []}
    for epoch in range(1, FORMAL_EPOCHS + 1):
        epoch_total = epoch_image_active = epoch_proto_active = epoch_image_count = epoch_proto_count = 0
        epoch_pids: set[int] = set()
        for rank in range(WORLD_SIZE):
            sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=rank, seed=SEED)
            indices = list(iter(sampler))
            if len(indices) // BATCH_SIZE != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"coverage sampler yielded {len(indices)//BATCH_SIZE}, expected {EXPECTED_STEPS_PER_EPOCH}")
            # The formal Phase 7A train loader uses drop_last=True and the
            # executed schedule consumes exactly 551 batches/rank/epoch.
            # Do not let a sampler tail partial batch alter the pre-training
            # coverage statistics.
            consumed = EXPECTED_STEPS_PER_EPOCH * BATCH_SIZE
            if len(indices) < consumed:
                raise RuntimeError(f"coverage sampler yielded {len(indices)} indices, expected at least {consumed}")
            plan_hashes[f"rank{rank}"].append(hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest())
            for start in range(0, consumed, BATCH_SIZE):
                batch_indices = indices[start:start + BATCH_SIZE]
                records = [dataset.train[index] for index in batch_indices]
                exposure = batch_exposure(records, path_to_index)
                total_batches += 1; epoch_total += 1
                image_counts.append(exposure["image_count"]); proto_counts.append(exposure["proto_count"]); active_proto_counts.append(exposure["active_proto"])
                proto_size_values.extend(exposure["proto_sizes"])
                if exposure["image_count"] > 0:
                    image_active_batches += 1; epoch_image_active += 1
                if exposure["proto_count"] > 0:
                    proto_active_batches += 1; epoch_proto_active += 1
                    pids_seen.update(exposure["proto_pids"]); epoch_pids.update(exposure["proto_pids"])
                epoch_image_count += exposure["image_count"]; epoch_proto_count += exposure["proto_count"]
                for key in exposure["image_state_keys"]:
                    c1_counter[key] += 1
                    state_keys_all.add(key)
                for key in exposure["proto_keys"]:
                    c2_counter[key] += 1
                    state_keys_all.add(key)
        epoch_rows.append({"epoch": epoch, "global_batches": epoch_total, "image_relation_active_batches": epoch_image_active, "image_relation_batch_rate": epoch_image_active / epoch_total, "image_relations": epoch_image_count, "prototype_valid_batches": epoch_proto_active, "prototype_batch_rate": epoch_proto_active / epoch_total, "prototype_relations": epoch_proto_count, "prototype_pid_coverage": len(epoch_pids), "prototype_mean_relations_per_batch": epoch_proto_count / epoch_total})
    if len(state_keys_all) != audit["dataset"]["valid_prototype_relations"]:
        raise RuntimeError(f"coverage state relation count mismatch: {len(state_keys_all)}")
    changing_pids = audit["dataset"]["changing_clothes_pids"]
    # This gate is intentionally explicit and conservative: at least 10% of
    # frozen-sampler batches must expose a valid prototype relation and at
    # least 80% of changing-clothes PIDs must receive one over 60 epochs.
    gate = {
        "prototype_batch_rate_minimum": 0.10,
        "changing_pid_coverage_minimum": 0.80,
        "prototype_batch_rate": proto_active_batches / total_batches,
        "changing_pid_coverage": len(pids_seen) / max(1, changing_pids),
        "prototype_batch_rate_pass": proto_active_batches / total_batches >= 0.10,
        "changing_pid_coverage_pass": len(pids_seen) / max(1, changing_pids) >= 0.80,
    }
    passed = all(value for key, value in gate.items() if key.endswith("_pass"))
    summary = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "epochs": FORMAL_EPOCHS,
        "global_batches": total_batches, "image_relation_active_batch_rate": image_active_batches / total_batches,
        "prototype_valid_batch_rate": proto_active_batches / total_batches,
        "mean_valid_prototype_relations_per_batch": float(np.mean(proto_counts)),
        "median_valid_prototype_relations_per_batch": float(np.median(proto_counts)),
        "mean_image_relations_per_batch": float(np.mean(image_counts)),
        "median_image_relations_per_batch": float(np.median(image_counts)),
        "active_prototypes_per_batch_mean": float(np.mean(active_proto_counts)),
        "prototype_state_size_mean": float(np.mean(proto_size_values)),
        "prototype_state_size_median": float(np.median(proto_size_values)),
        "pids_ever_prototype_supervised": len(pids_seen), "changing_pids": changing_pids,
        "prototype_pid_coverage": len(pids_seen) / max(1, changing_pids),
        "plan_hashes": plan_hashes, "gate": gate, "coverage_pass": passed,
        "training_started": False, "train_only": True,
    }
    write_csv(FEASIBILITY_ROOT / "coverage_by_epoch.csv", epoch_rows)
    write_json(FEASIBILITY_ROOT / "coverage_summary.json", summary)
    balance_rows = [{"pid": key[0], "cloth_a": key[1], "cloth_b": key[2], "c1_image_level_contributions": c1_counter[key], "c2_prototype_contributions": c2_counter[key]} for key in sorted(state_keys_all)]
    write_csv(BALANCE_ROOT / "planned_contribution_counts.csv", balance_rows)
    balance_summary = []
    for arm, counter in (("C1", c1_counter), ("C2", c2_counter)):
        values = np.asarray([counter[key] for key in sorted(state_keys_all)], dtype=np.float64)
        sorted_values = np.sort(values)
        item = {"arm": arm, "relation_count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(np.max(values)), "std": float(np.std(values)), "coefficient_of_variation": float(np.std(values) / np.mean(values)), "total_contributions": int(np.sum(values))}
        for pct in (1, 5, 10):
            n = max(1, int(math.ceil(len(values) * pct / 100)))
            item[f"top_{pct}pct_share"] = float(np.sum(sorted_values[-n:]) / np.sum(values))
        balance_summary.append(item)
    write_csv(BALANCE_ROOT / "planned_balance_summary.csv", balance_summary)
    if not passed:
        raise RuntimeError(f"PROTOTYPE COVERAGE BOTTLENECK: {summary}")
    return summary


def load_common_init() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    audit = load_source_audit()
    state = torch.load(PHASE7A_COMMON_INIT, map_location="cpu", weights_only=False)
    meta = audit["common_init"]
    if sha256_file(PHASE7A_COMMON_INIT) != meta["sha256"] or state_hash(state) != meta["state_hash"]:
        raise RuntimeError("common init changed after source audit")
    return state, meta


def prepare() -> dict[str, Any]:
    audit = load_source_audit()
    coverage_summary_path = FEASIBILITY_ROOT / "coverage_summary.json"
    if not coverage_summary_path.is_file():
        raise RuntimeError("run --mode coverage before prepare")
    coverage_summary = json.loads(coverage_summary_path.read_text())
    if not coverage_summary.get("coverage_pass"):
        raise RuntimeError("prototype coverage gate failed; training is forbidden")
    state, meta = load_common_init()
    cfg = cfg_for(RUN_DIRS["C0"])
    strict_hashes = {}
    for arm in RUN_DIRS:
        model = phase7a.make_model_from_state(cfg, state, EXPECTED_TRAIN_PIDS, 3)
        strict_hashes[arm] = state_hash({key: value.detach().cpu() for key, value in model.state_dict().items()})
        del model
    if len(set(strict_hashes.values())) != 1 or strict_hashes["C0"] != meta["state_hash"]:
        raise RuntimeError(f"C0/C1/C2 common init mismatch: {strict_hashes}")
    common_audit = {"phase": PHASE, "created_utc": utc_now(), "path": str(PHASE7A_COMMON_INIT), "sha256": meta["sha256"], "state_hash": meta["state_hash"], "initial_parameter_hashes": strict_hashes, "initial_parameter_equality": len(set(strict_hashes.values())) == 1, "strict_load": True, "optimizer": phase7a.optimizer_config(cfg), "scheduler": phase7a.scheduler_config(cfg), "amp": phase7a.amp_config(), "lambda_rel": LAMBDA_REL, "coverage_summary": str(coverage_summary_path), "training_started": False}
    write_json(TRAINING_ROOT / "common_init_audit.json", common_audit)
    write_json(OUT_ROOT / "PREPARE.json", {"phase": PHASE, "prepared_utc": utc_now(), "seed": SEED, "source_audit_pass": True, "coverage_pass": True, "common_init_audit": str(TRAINING_ROOT / "common_init_audit.json"), "allowed_arms": list(RUN_DIRS), "forbidden": ["SR3", "H_ID", "DS", "DW", "GRL", "Reverse", "new sampler", "new optimizer", "new scheduler", "margin loss", "memory bank", "EMA", "prototype classifier"], "training_started": False})
    for arm, run_dir in RUN_DIRS.items():
        if (run_dir / "RUN_COMPLETE.json").exists():
            raise RuntimeError(f"refusing to overwrite completed Phase 7C arm: {run_dir}")
    return common_audit


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def prototype_relation_loss(feat: list[torch.Tensor], pids: torch.Tensor, clothes: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Current-mini-batch Norm(mean(Norm(feat[0]))) prototype relation."""
    representation = F.normalize(feat[0].float(), dim=-1)
    pid_values = [int(value) for value in pids.detach().cpu().tolist()]
    cloth_values = [int(value) for value in clothes.detach().cpu().tolist()]
    by_pid: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (pid, cloth) in enumerate(zip(pid_values, cloth_values)):
        by_pid[pid][cloth].append(index)
    losses: list[torch.Tensor] = []
    keys: list[tuple[int, int, int]] = []
    sizes: list[int] = []
    similarities: list[float] = []
    active_pids: set[int] = set()
    for pid, states in sorted(by_pid.items()):
        clothes_sorted = sorted(states)
        if len(clothes_sorted) < 2:
            continue
        active_pids.add(pid)
        prototypes: dict[int, torch.Tensor] = {}
        for cloth in clothes_sorted:
            indices = torch.tensor(states[cloth], dtype=torch.long, device=representation.device)
            raw = representation.index_select(0, indices).mean(dim=0)
            prototypes[cloth] = F.normalize(raw, dim=0)
            sizes.append(len(states[cloth]))
        for pos, cloth_a in enumerate(clothes_sorted):
            for cloth_b in clothes_sorted[pos + 1:]:
                similarity = torch.sum(prototypes[cloth_a] * prototypes[cloth_b])
                losses.append(1.0 - similarity)
                keys.append((pid, cloth_a, cloth_b))
                similarities.append(float(similarity.detach().cpu()))
    if not losses:
        zero = representation.sum() * 0.0
        return zero, {"active": False, "pair_count": 0, "keys": [], "active_pids": set(), "active_prototypes": 0, "prototype_sizes": [], "similarities": []}
    return torch.stack(losses).mean(), {
        "active": True, "pair_count": len(losses), "keys": keys, "active_pids": active_pids,
        "active_prototypes": len(sizes), "prototype_sizes": sizes, "similarities": similarities,
    }


def c2_forward_batch(model: DDP, loss_fn: Any, batch: dict[str, Any], device: torch.device, global_step: int) -> dict[str, Any]:
    writer = base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    proto, relation_info = prototype_relation_loss(feat, pids, clothes)
    total = base_total + LAMBDA_REL * proto
    components = getattr(loss_fn, "last_components", {})
    return {
        "total": total, "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "relation_raw": proto, "relation_applied": proto, "relation_info": relation_info,
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1], "writer": writer,
    }


def records_from_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    pids = batch["pid"].detach().cpu().tolist()
    clothes = batch["clothes_id"].detach().cpu().tolist()
    paths = [str(path) for path in batch["image_path"]]
    return [{"pid": int(pid), "clothes_id": int(cloth), "image_path": path} for pid, cloth, path in zip(pids, clothes, paths)]


def logger_for(path: Path, arm: str) -> logging.Logger:
    logger = logging.getLogger(f"{PHASE}.{arm}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def reduce_epoch(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] / EXPECTED_STEPS_PER_EPOCH for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].detach().cpu()) for index, key in enumerate(keys)}


def gather_exposure(local: dict[str, Any]) -> list[Any]:
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    return gathered


def save_rank_state(path: Path, model: DDP, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE, "epoch": epoch, "global_step": global_step,
        "model": base.cpu_state_dict(model), "optimizer": base.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": base.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_arm(arm: str) -> None:
    if arm not in RUN_DIRS:
        raise ValueError(f"allowed arms are {list(RUN_DIRS)}, got {arm}")
    load_source_audit()
    prepare_path = OUT_ROOT / "PREPARE.json"
    if not prepare_path.is_file() or not json.loads(prepare_path.read_text()).get("coverage_pass"):
        raise RuntimeError("run --mode prepare after a passing coverage audit")
    require_ppu()
    if (RUN_DIRS[arm] / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"refusing to overwrite completed arm: {RUN_DIRS[arm]}")
    rank, device = phase6b.init_dist()
    run_dir = RUN_DIRS[arm]
    streams: list[Any] = []
    logger = None
    try:
        cfg = cfg_for(run_dir)
        set_seed(SEED)
        state, common_meta = load_common_init()
        dataset = phase7a.build_dataset(cfg)
        bank = phase7a.load_relation_bank(dataset)
        train_loader, _same_loader, _diff_loader, _gallery_loader, dataset_again, train_sampler, val_loader, val_loader_same = phase7a.build_dataloader(cfg)
        if [str(item["image_path"]) for item in dataset_again.train] != [str(item["image_path"]) for item in dataset.train]:
            raise RuntimeError("Phase 7C dataset order differs from audited Phase 7A order")
        dataset = dataset_again
        bank = phase7a.load_relation_bank(dataset)
        if len(train_loader) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(f"expected {EXPECTED_STEPS_PER_EPOCH} batches, got {len(train_loader)}")
        model, loss_fn, optimizer, scheduler, scaler = phase7a.build_runtime(cfg, state, device, dataset)
        local_hash = phase7a.model_identity_hash(model)
        gathered_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_hashes, local_hash)
        if len(set(gathered_hashes)) != 1 or local_hash != common_meta["state_hash"]:
            raise RuntimeError(f"{arm} common init mismatch: {gathered_hashes}")
        path_to_index = bank["path_to_index"]
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            manifest = {
                "phase": PHASE, "arm": arm, "label": ARM_LABELS[arm], "seed": SEED,
                "epochs": FORMAL_EPOCHS, "start_epoch": 1, "end_epoch": FORMAL_EPOCHS,
                "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH, "train_images": len(dataset.train),
                "train_pids": int(dataset.num_train_pids), "train_clothing_states": int(dataset.num_train_clothes),
                "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES,
                "optimizer": phase7a.optimizer_config(cfg), "scheduler": phase7a.scheduler_config(cfg), "amp": phase7a.amp_config(),
                "common_init": {"path": str(PHASE7A_COMMON_INIT), "sha256": common_meta["sha256"], "state_hash": common_meta["state_hash"]},
                "initial_parameter_hash": local_hash, "initial_parameter_strict_load": True,
                "reverse": "OFF", "grl": "OFF", "dw": "OFF", "ds": "OFF", "h_id": "OFF", "online_reliability": "OFF",
                "sampler": "Phase 7A DistributedRandomIdentitySampler; set_epoch remains OFF as in matched source",
                "relation_feature": "feat[0]",
                "relation_type": "none" if arm == "C0" else ("image_level_uniform" if arm == "C1" else "current_batch_clothing_state_prototype"),
                "lambda_rel": 0.0 if arm == "C0" else LAMBDA_REL,
                "relation_formula": "C0 none; C1 mean(1-cos(z_i,z_j)); C2 mean(1-cos(Norm(mean(Norm(z_i))), Norm(mean(Norm(z_j)))))",
                "prototype_memory": "none", "prototype_ema": "none", "prototype_classifier": "none",
                "relation_bank": str(PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz"),
                "relation_bank_sha256": sha256_file(PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz"),
                "evaluation": {"primary": "PRCC changed clothes C -> A", "secondary": "same clothes B -> A", "reranking": False},
                "forbidden_arms_started": [], "training_started": True,
            }
            write_json(run_dir / "run_manifest.json", manifest)
            metrics_file = (run_dir / "metrics_per_epoch.csv").open("w", newline="", encoding="utf-8"); streams.append(metrics_file)
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS); metrics_writer.writeheader(); metrics_file.flush()
            exposure_file = (run_dir / "relation_exposure.csv").open("w", newline="", encoding="utf-8"); streams.append(exposure_file)
            exposure_writer = csv.DictWriter(exposure_file, fieldnames=EXPOSURE_FIELDS); exposure_writer.writeheader(); exposure_file.flush()
            amp_file = (run_dir / "amp_audit.csv").open("w", newline="", encoding="utf-8"); streams.append(amp_file)
            amp_writer = csv.DictWriter(amp_file, fieldnames=AMP_FIELDS); amp_writer.writeheader(); amp_file.flush()
            amp_event_file = (run_dir / "amp_skip_events.csv").open("w", newline="", encoding="utf-8"); streams.append(amp_event_file)
            amp_event_writer = csv.DictWriter(amp_event_file, fieldnames=AMP_EVENT_FIELDS); amp_event_writer.writeheader(); amp_event_file.flush()
            sampler_file = (run_dir / "sampler_plan.csv").open("w", newline="", encoding="utf-8"); streams.append(sampler_file)
            sampler_writer = csv.DictWriter(sampler_file, fieldnames=SAMPLER_FIELDS); sampler_writer.writeheader(); sampler_file.flush()
            logger = logger_for(run_dir / "train_log_rank0.txt", arm)
        else:
            metrics_writer = exposure_writer = amp_writer = amp_event_writer = sampler_writer = None
        best_rank1 = -float("inf"); best_epoch = None; global_step = 0
        geometry_rows: list[dict[str, Any]] = []
        balance_counter: Counter[tuple[int, int, int]] = Counter()
        _train_transform, test_transform = phase7a.build_img_transforms(cfg)
        for epoch in range(1, FORMAL_EPOCHS + 1):
            scheduler.step(epoch)
            audit_sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=rank, seed=SEED)
            audit_indices = list(iter(audit_sampler))
            plan_digest = hashlib.sha256(json.dumps(audit_indices, separators=(",", ":")).encode("utf-8")).hexdigest()
            sampler_meta = {"arm": arm, "seed": SEED, "epoch": epoch, "rank": rank, "sampler_epoch_used": int(audit_sampler.epoch), "samples": len(audit_indices), "batches": len(audit_indices) // BATCH_SIZE, "unique_images": len(set(audit_indices)), "unique_pids": len({int(dataset.train[i]["pid"]) for i in audit_indices}), "unique_clothing_states": len({(int(dataset.train[i]["pid"]), int(dataset.train[i]["clothes_id"])) for i in audit_indices}), "plan_sha256": plan_digest}
            gathered_sampler = gather_exposure(sampler_meta)
            if rank == 0:
                for row in gathered_sampler: sampler_writer.writerow(row)
                sampler_file.flush()
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation", "relation_applied", "id_acc", "bio_acc")}
            local_image_active = 0; local_image_relations = 0; local_proto_active = 0; local_proto_relations = 0; local_active_prototypes = 0
            local_image_keys: set[tuple[int, int]] = set(); local_proto_keys: set[tuple[int, int, int]] = set(); local_image_state_keys: set[tuple[int, int, int]] = set(); local_proto_pids: set[int] = set(); local_sampled_images: set[int] = set(); local_sampled_pids: set[int] = set(); local_sampled_states: set[tuple[int, int]] = set()
            local_proto_sizes: list[int] = []; local_proto_sims: list[float] = []; local_balance: Counter[tuple[int, int, int]] = Counter()
            local_skips = 0; local_found_inf = 0; local_nonfinite = 0; scale_min = float(scaler.get_scale()); scale_max = scale_min; local_amp_events: list[dict[str, Any]] = []
            iteration = 0
            for iteration, batch in enumerate(train_loader, start=1):
                global_step += 1; optimizer.zero_grad()
                records = records_from_batch(batch)
                exposure = batch_exposure(records, path_to_index)
                if exposure["image_count"] > 0: local_image_active += 1
                local_image_relations += exposure["image_count"]; local_image_keys.update(exposure["image_keys"]); local_image_state_keys.update(exposure["image_state_keys"])
                if exposure["proto_count"] > 0: local_proto_active += 1
                local_proto_relations += exposure["proto_count"]; local_proto_keys.update(exposure["proto_keys"]); local_proto_pids.update(exposure["proto_pids"]); local_active_prototypes += exposure["active_proto"]; local_proto_sizes.extend(exposure["proto_sizes"])
                local_sampled_images.update(path_to_index[str(item["image_path"])] for item in records); local_sampled_pids.update(int(item["pid"]) for item in records); local_sampled_states.update((int(item["pid"]), int(item["clothes_id"])) for item in records)
                if arm in ("C0", "C1"):
                    result = phase6b.forward_batch(model, loss_fn, batch, device, bank, path_to_index, ARM_PHASE6B[arm], global_step)
                    if arm == "C0":
                        result["relation_applied"] = result["relation_raw"] * 0.0
                else:
                    result = c2_forward_batch(model, loss_fn, batch, device, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite {arm} loss at E{epoch} I{iteration}")
                scaler.scale(total).backward(); scale_before = float(scaler.get_scale()); scaler.step(optimizer); found_inf = int(base.found_inf(scaler, optimizer) > 0); scaler.update(); scale_after = float(scaler.get_scale())
                scale_min = min(scale_min, scale_after); scale_max = max(scale_max, scale_after); local_skips += found_inf; local_found_inf += found_inf; local_nonfinite += found_inf
                if found_inf:
                    local_amp_events.append({"arm": arm, "seed": SEED, "rank": rank, "epoch": epoch, "iteration": iteration, "global_step": global_step, "skipped_step": 1, "found_inf": found_inf, "nonfinite_event": 1, "amp_scale_before": scale_before, "amp_scale_after": scale_after})
                for key in epoch_sum: epoch_sum[key] += scalar(result["relation_raw"] if key == "relation" else result["relation_applied"] if key == "relation_applied" else result[key])
                info = result["relation_info"]
                if arm == "C1":
                    local_balance.update(exposure["image_state_keys"])
                elif arm == "C2":
                    local_balance.update(exposure["proto_keys"])
                    local_proto_sims.extend(float(value) for value in info.get("similarities", []))
                if rank == 0 and logger is not None and iteration % 100 == 0:
                    logger.info(json.dumps({"arm": arm, "epoch": epoch, "iteration": iteration, "loss": scalar(total), "image_relations": exposure["image_count"], "proto_relations": exposure["proto_count"], "amp_skip": found_inf}))
            if iteration != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"{arm} loader yielded {iteration} batches, expected {EXPECTED_STEPS_PER_EPOCH}")
            gathered_events = [None] * WORLD_SIZE; dist.all_gather_object(gathered_events, local_amp_events)
            if rank == 0:
                for events in gathered_events:
                    for event in events: amp_event_writer.writerow(event)
                amp_event_file.flush()
            reduced = reduce_epoch(epoch_sum, device)
            amp_tensor = torch.tensor([local_skips, local_found_inf, local_nonfinite, scale_min, scale_max, scale_after], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); found_sum = amp_tensor[1].clone(); nonfinite_sum = amp_tensor[2].clone(); dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM); dist.all_reduce(found_sum, op=dist.ReduceOp.SUM); dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            amp_min = amp_tensor[3].clone(); amp_max = amp_tensor[4].clone(); amp_end = amp_tensor[5].clone(); dist.all_reduce(amp_min, op=dist.ReduceOp.MIN); dist.all_reduce(amp_max, op=dist.ReduceOp.MAX); dist.all_reduce(amp_end, op=dist.ReduceOp.MAX)
            updates_completed = EXPECTED_STEPS_PER_EPOCH - float(skip_sum.item()) / WORLD_SIZE
            local_exposure = {"image_active": local_image_active, "image_relations": local_image_relations, "image_keys": local_image_keys, "image_state_keys": local_image_state_keys, "proto_active": local_proto_active, "proto_relations": local_proto_relations, "proto_keys": local_proto_keys, "proto_pids": local_proto_pids, "active_prototypes": local_active_prototypes, "proto_sizes": local_proto_sizes, "proto_sims": local_proto_sims, "sampled_images": local_sampled_images, "sampled_pids": local_sampled_pids, "sampled_states": local_sampled_states, "balance": dict(local_balance)}
            gathered = gather_exposure(local_exposure)
            image_active = sum(int(item["image_active"]) for item in gathered); image_relations = sum(int(item["image_relations"]) for item in gathered); proto_active = sum(int(item["proto_active"]) for item in gathered); proto_relations = sum(int(item["proto_relations"]) for item in gathered); active_prototypes = sum(int(item["active_prototypes"]) for item in gathered); proto_sizes = [value for item in gathered for value in item["proto_sizes"]]; proto_sims = [value for item in gathered for value in item["proto_sims"]]; sampled_images = set().union(*(set(item["sampled_images"]) for item in gathered)); sampled_pids = set().union(*(set(item["sampled_pids"]) for item in gathered)); sampled_states = set().union(*(set(item["sampled_states"]) for item in gathered)); image_keys = set().union(*(set(item["image_keys"]) for item in gathered)); proto_keys = set().union(*(set(item["proto_keys"]) for item in gathered)); proto_pids = set().union(*(set(item["proto_pids"]) for item in gathered))
            for item in gathered: balance_counter.update(item["balance"])
            dist.barrier()
            if rank == 0:
                metrics_eval = phase7a.evaluate_prcc(cfg, model.module, dataset, val_loader, val_loader_same, device, epoch, logger)
            else: metrics_eval = None
            holder = [metrics_eval]; dist.broadcast_object_list(holder, src=0); metrics_eval = holder[0]; dist.barrier()
            lr_values = scheduler._get_lr(epoch)
            row = {"arm": arm, "seed": SEED, "epoch": epoch, "train_loss": reduced["total"], "ce_loss": reduced["ce"], "triplet_loss": reduced["triplet"], "bio_loss": reduced["bio"], "relation_loss": reduced["relation"], "relation_applied_loss": reduced["relation_applied"], "relation_type": "none" if arm == "C0" else ("image_level_uniform" if arm == "C1" else "prototype"), "relation_lambda": 0.0 if arm == "C0" else LAMBDA_REL, "total_loss": reduced["total"], "lr": float(lr_values[0]), "lr_bias": float(lr_values[-1]), "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"], "planned_updates": EXPECTED_STEPS_PER_EPOCH, "updates_completed": updates_completed, "amp_scale_min": float(amp_min.item()), "amp_scale_max": float(amp_max.item()), "amp_scale_end": float(amp_end.item()), "amp_skip": int(skip_sum.item() > 0), "skipped_steps": int(skip_sum.item()), "found_inf_count": int(found_sum.item()), "nonfinite_events": int(nonfinite_sum.item()), "active_images": len(dataset.train), "active_pids": int(dataset.num_train_pids), "active_clothing_states": int(dataset.num_train_clothes), "sampled_unique_images": len(sampled_images), "sampled_unique_pids": len(sampled_pids), "sampled_unique_clothing_states": len(sampled_states), "image_relation_active_batches": image_active, "image_relation_batch_rate": image_active / float(EXPECTED_STEPS_PER_EPOCH * WORLD_SIZE), "image_relations_per_epoch": image_relations, "unique_image_relation_count": len(image_keys), "prototype_valid_batches": proto_active, "prototype_batch_rate": proto_active / float(EXPECTED_STEPS_PER_EPOCH * WORLD_SIZE), "prototype_relations_per_epoch": proto_relations, "unique_prototype_relation_count": len(proto_keys), "active_prototypes_per_epoch": active_prototypes, "prototype_pid_coverage": len(proto_pids), "prototype_relation_pid_coverage": len(proto_pids), "prototype_mean_images_per_state": float(np.mean(proto_sizes)) if proto_sizes else 0.0, "prototype_median_images_per_state": float(np.median(proto_sizes)) if proto_sizes else 0.0, "prototype_mean_similarity": float(np.mean(proto_sims)) if proto_sims else 0.0, **metrics_eval}
            exposure_row = {key: row.get(key, 0.0) for key in EXPOSURE_FIELDS}; exposure_row.update({"arm": arm, "seed": SEED, "epoch": epoch, "all_images_available": True, "all_pids_available": True, "all_clothing_states_available": True, "planned_updates": EXPECTED_STEPS_PER_EPOCH, "updates_completed": updates_completed})
            amp_row = {"arm": arm, "seed": SEED, "epoch": epoch, "amp_skip": row["amp_skip"], "skipped_steps": row["skipped_steps"], "found_inf_count": row["found_inf_count"], "nonfinite_events": row["nonfinite_events"], "amp_scale_min": row["amp_scale_min"], "amp_scale_max": row["amp_scale_max"], "amp_scale_end": row["amp_scale_end"], "planned_updates": EXPECTED_STEPS_PER_EPOCH, "updates_completed": updates_completed}
            if rank == 0:
                metrics_writer.writerow(row); metrics_file.flush(); exposure_writer.writerow(exposure_row); exposure_file.flush(); amp_writer.writerow(amp_row); amp_file.flush()
                if float(row["cc_rank1"]) > best_rank1:
                    best_rank1 = float(row["cc_rank1"]); best_epoch = epoch; phase6b.save_model_checkpoint(model.module, run_dir / "model_best.pth"); write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_rank1, "criterion": "PRCC changed-clothes Rank-1"})
                print(json.dumps({"arm": arm, "epoch": epoch, "cc_rank1": row["cc_rank1"], "cc_mAP": row["cc_mAP"], "train_loss": row["train_loss"], "proto_batch_rate": row["prototype_batch_rate"], "amp_skips": row["skipped_steps"]}, sort_keys=True), flush=True)
            if epoch in GEOMETRY_EPOCHS:
                if rank == 0:
                    geometry_rows.extend(geometry_for_model(model.module, dataset, bank, device, test_transform, epoch, arm)); write_csv(GEOMETRY_ROOT / f"{arm}_geometry.csv", geometry_rows)
                dist.barrier()
            if epoch == FORMAL_EPOCHS:
                if rank == 0: phase6b.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                dist.barrier(); save_rank_state(run_dir / f"rank{rank}_state_epoch060.pt", model, optimizer, scheduler, scaler, epoch, global_step, device); dist.barrier()
                if rank == 0:
                    write_json(run_dir / "checkpoint_epoch060.json", {"phase": PHASE, "arm": arm, "epoch": epoch, "global_step": global_step, "rank_states": [str(run_dir / f"rank{r}_state_epoch060.pt") for r in range(WORLD_SIZE)], "model": str(run_dir / "model_final.pth")})
                    rows = read_csv(run_dir / "metrics_per_epoch.csv")
                    write_json(run_dir / "RUN_COMPLETE.json", {"phase": PHASE, "arm": arm, "mode": "FULL_FORMAL_PRCC_SCHEDULE", "seed": SEED, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates_per_rank": global_step, "metrics": str(run_dir / "metrics_per_epoch.csv"), "best_epoch": best_epoch, "best_cc_rank1": best_rank1, "best_model": str(run_dir / "model_best.pth"), "final_model": str(run_dir / "model_final.pth"), "rows": len(rows), "common_init_sha256": common_meta["sha256"], "initial_parameter_hash": local_hash})
        if rank == 0:
            write_json(run_dir / "AMP.json", {"phase": PHASE, "arm": arm, "seed": SEED, "initial_scale": INITIAL_SCALE, "epochs_completed": FORMAL_EPOCHS, "metrics_source": str(run_dir / "amp_audit.csv"), "skip_event_source": str(run_dir / "amp_skip_events.csv")})
            balance_rows = [{"pid": key[0], "cloth_a": key[1], "cloth_b": key[2], "contributions": int(balance_counter[key])} for key in sorted(balance_counter)]
            write_csv(run_dir / "relation_balance_contributions.csv", balance_rows)
    finally:
        for stream in streams: stream.close()
        if logger is not None:
            for handler in list(logger.handlers): handler.close(); logger.removeHandler(handler)
        phase6b.cleanup_dist()


def extract_train_features(model: torch.nn.Module, dataset: Any, cfg: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    _train_transform, test_transform = phase7a.build_img_transforms(cfg)
    loader = DataLoader(ImageDataset(dataset.train, transform=test_transform), batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            camids = batch["camid"].to(device)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            if not torch.is_tensor(output):
                raise RuntimeError(f"unexpected geometry output: {type(output)}")
            blocks.append(F.normalize(output.float(), dim=-1).detach().cpu().numpy().astype(np.float32))
    features = np.concatenate(blocks, axis=0)
    if len(features) != len(dataset.train):
        raise RuntimeError(f"geometry feature count mismatch: {len(features)} != {len(dataset.train)}")
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    model.train()
    return features, pids, clothes


def geometry_for_model(model: torch.nn.Module, dataset: Any, bank: dict[str, Any], device: torch.device, test_transform: Any, epoch: int, arm: str) -> list[dict[str, Any]]:
    del test_transform
    cfg = cfg_for(RUN_DIRS[arm])
    features, pids, clothes = extract_train_features(model, dataset, cfg, device)
    features = F.normalize(torch.from_numpy(features), dim=-1).numpy()
    pair_a = bank["arrays"]["index_a"].astype(np.int64)
    pair_b = bank["arrays"]["index_b"].astype(np.int64)
    if len(pair_a) == 0:
        raise RuntimeError("geometry found no train cross-clothes pairs")
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (pid, cloth) in enumerate(zip(pids.tolist(), clothes.tolist())):
        by_pid_cloth[int(pid)][int(cloth)].append(index)
    same_a: list[int] = []; same_b: list[int] = []
    for states in by_pid_cloth.values():
        for indices in states.values():
            for pos, left in enumerate(indices[:-1]):
                same_a.extend([left] * (len(indices) - pos - 1)); same_b.extend(indices[pos + 1:])
    same_a_np = np.asarray(same_a, dtype=np.int64); same_b_np = np.asarray(same_b, dtype=np.int64)
    cross_image = np.sum(features[pair_a] * features[pair_b], axis=1, dtype=np.float64)
    same_image = np.sum(features[same_a_np] * features[same_b_np], axis=1, dtype=np.float64)
    unique_pids = sorted(set(pids.tolist()))
    pid_centers = np.stack([F.normalize(torch.from_numpy(features[pids == pid]).mean(dim=0, keepdim=True), dim=-1).numpy()[0] for pid in unique_pids]).astype(np.float32)
    pid_to_center = {pid: index for index, pid in enumerate(unique_pids)}
    all_negative = features @ pid_centers.T
    for index, pid in enumerate(pids.tolist()): all_negative[index, pid_to_center[pid]] = -np.inf
    nearest_negative_image = np.max(all_negative, axis=1)
    cross_negative_image = np.maximum(nearest_negative_image[pair_a], nearest_negative_image[pair_b])
    image_margin = cross_image - cross_negative_image
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    for pid, states in by_pid_cloth.items():
        for cloth, indices in states.items():
            state_centers[(pid, cloth)] = F.normalize(torch.from_numpy(features[indices]).mean(dim=0, keepdim=True), dim=-1).numpy()[0]
    proto_a: list[np.ndarray] = []; proto_b: list[np.ndarray]; proto_neg: list[float] = []; proto_positive: list[float] = []; proto_margin: list[float] = []; state_dispersion: list[float] = []
    proto_b = []
    for pid, states in sorted(by_pid_cloth.items()):
        clothes_sorted = sorted(states)
        for pos, cloth_a in enumerate(clothes_sorted):
            for cloth_b in clothes_sorted[pos + 1:]:
                left = state_centers[(pid, cloth_a)]; right = state_centers[(pid, cloth_b)]
                proto_a.append(left); proto_b.append(right)
                similarity = float(np.dot(left, right)); proto_positive.append(similarity)
                negative_left = float(np.max(np.delete(pid_centers @ left, pid_to_center[pid])))
                negative_right = float(np.max(np.delete(pid_centers @ right, pid_to_center[pid])))
                negative = max(negative_left, negative_right); proto_neg.append(negative); proto_margin.append(similarity - negative)
                state_dispersion.append(1.0 - similarity)
    row = {
        "arm": arm, "seed": SEED, "checkpoint": f"E{epoch}", "image_cross_pair_count": int(len(cross_image)), "same_clothes_image_pair_count": int(len(same_image)), "prototype_relation_count": int(len(proto_positive)), "pid_count": len(unique_pids), "clothing_state_count": len(state_centers),
        "same_pid_same_clothes_similarity_mean": float(np.mean(same_image)), "same_pid_cross_clothes_image_similarity_mean": float(np.mean(cross_image)), "same_pid_cross_clothes_prototype_similarity_mean": float(np.mean(proto_positive)), "nearest_negative_image_similarity_mean": float(np.mean(cross_negative_image)), "nearest_negative_prototype_similarity_mean": float(np.mean(proto_neg)), "image_identity_margin_mean": float(np.mean(image_margin)), "prototype_identity_margin_mean": float(np.mean(proto_margin)), "image_ordering_correct_rate": float(np.mean(image_margin > 0)), "prototype_ordering_correct_rate": float(np.mean(np.asarray(proto_margin) > 0)), "clothing_state_center_dispersion_mean": float(np.mean(state_dispersion)), "geometry_source": "PRCC train-only; query/gallery/test excluded from relation construction",
    }
    return [row]


def gradient_group_metrics(names: list[str], base_grads: tuple[torch.Tensor | None, ...], img_grads: tuple[torch.Tensor | None, ...], proto_grads: tuple[torch.Tensor | None, ...], selected: set[str]) -> dict[str, Any]:
    base_sq = img_sq = proto_sq = base_img_dot = base_proto_dot = 0.0
    for name, bg, ig, pg in zip(names, base_grads, img_grads, proto_grads):
        if name not in selected: continue
        reference = bg if bg is not None else ig if ig is not None else pg
        if reference is None: continue
        if bg is None: bg = torch.zeros_like(reference)
        if ig is None: ig = torch.zeros_like(reference)
        if pg is None: pg = torch.zeros_like(reference)
        bg = bg.detach().float(); ig = ig.detach().float(); pg = pg.detach().float()
        base_sq += float(torch.sum(bg * bg).detach().cpu()); img_sq += float(torch.sum(ig * ig).detach().cpu()); proto_sq += float(torch.sum(pg * pg).detach().cpu()); base_img_dot += float(torch.sum(bg * ig).detach().cpu()); base_proto_dot += float(torch.sum(bg * pg).detach().cpu())
    base_norm = math.sqrt(max(base_sq, 0.0)); img_norm = math.sqrt(max(img_sq, 0.0)); proto_norm = math.sqrt(max(proto_sq, 0.0)); eps = 1e-12
    return {"norm_base": base_norm, "norm_image": img_norm, "norm_proto": proto_norm, "norm_image_ratio": img_norm / (base_norm + eps), "norm_proto_ratio": proto_norm / (base_norm + eps), "alignment_image": base_img_dot / (base_norm * img_norm + eps) if base_norm > eps and img_norm > eps else None, "alignment_proto": base_proto_dot / (base_norm * proto_norm + eps) if base_norm > eps and proto_norm > eps else None}


def gradient_probe() -> dict[str, Any]:
    require_ppu(); audit = load_source_audit(); del audit
    cfg = cfg_for(RUN_DIRS["C0"]); dataset = phase7a.build_dataset(cfg); bank = phase7a.load_relation_bank(dataset); state = torch.load(RUN_DIRS["C0"] / "model_final.pth", map_location="cpu", weights_only=False)
    device = torch.device("cuda:0"); torch.cuda.set_device(0); model = phase7a.make_model_from_state(cfg, state, EXPECTED_TRAIN_PIDS, 3).to(device).train(); loss_fn, _center = phase7a.make_loss(cfg, num_classes=EXPECTED_TRAIN_PIDS)
    sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=0, seed=SEED); loader = DataLoader(ImageDataset(dataset.train, transform=phase7a.build_img_transforms(cfg)[0]), sampler=sampler, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, drop_last=True)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]; names = [name for name, _ in named]; parameters = [parameter for _, parameter in named]; block_ids = sorted({int(name.split(".")[1]) for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit()}); late_ids = set(block_ids[-3:]); groups = {"late_eva": {name for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late_ids}, "id_head": {name for name in names if name.startswith("head.")}, "bio_projection": {name for name in names if name.startswith("head_bio") or name.startswith("head_clip_bio") or name.startswith("norm_bio")}}
    rows: list[dict[str, Any]] = []; inspected = 0
    for batch in loader:
        inspected += 1; records = records_from_batch(batch); exposure = batch_exposure(records, bank["path_to_index"])
        if exposure["image_count"] == 0 or exposure["proto_count"] == 0: continue
        images = batch["image"].to(device); pids = batch["pid"].to(device); camids = batch["camid"].to(device); clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device); writer = base.MetricCapture(); model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=camids); base_loss = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=len(rows) + 1); img_loss, _img_info = phase6b.relation_loss(feat, pids, clothes, [str(path) for path in batch["image_path"]], bank["path_to_index"], bank, "B1"); proto_loss, _proto_info = prototype_relation_loss(feat, pids, clothes)
        base_grads = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True); img_grads = torch.autograd.grad(img_loss, parameters, retain_graph=True, allow_unused=True); proto_grads = torch.autograd.grad(proto_loss, parameters, retain_graph=False, allow_unused=True)
        row = {"batch_index": inspected, "batch_size": len(records), "image_relations": exposure["image_count"], "prototype_relations": exposure["proto_count"], "base_loss": scalar(base_loss), "image_relation_loss": scalar(img_loss), "prototype_relation_loss": scalar(proto_loss), "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_step": False}
        for group_name, selected in groups.items(): row.update({f"{key}_{group_name}": value for key, value in gradient_group_metrics(names, base_grads, img_grads, proto_grads, selected).items()})
        rows.append(row); model.zero_grad(set_to_none=True); del base_grads, img_grads, proto_grads, base_loss, img_loss, proto_loss, feat, score
        if len(rows) >= GRADIENT_BATCHES: break
    if len(rows) < GRADIENT_BATCHES: raise RuntimeError(f"only {len(rows)} matched gradient probes found")
    write_csv(GRADIENT_ROOT / "gradient_probes.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for module in ("late_eva", "id_head", "bio_projection"):
        summary = {"module": module, "rows": len(rows)}
        for metric in ("norm_image", "norm_proto", "norm_image_ratio", "norm_proto_ratio", "alignment_image", "alignment_proto"):
            values = [float(row[f"{metric}_{module}"]) for row in rows if row.get(f"{metric}_{module}") not in (None, "", "None")]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None; summary[metric + "_median"] = float(np.median(values)) if values else None; summary[metric + "_min"] = float(np.min(values)) if values else None; summary[metric + "_max"] = float(np.max(values)) if values else None
        summary_rows.append(summary)
    write_csv(GRADIENT_ROOT / "gradient_summary.csv", summary_rows)
    write_json(GRADIENT_ROOT / "gradient_manifest.json", {"phase": PHASE, "baseline_checkpoint": str(RUN_DIRS["C0"] / "model_final.pth"), "rows": len(rows), "target_rows": GRADIENT_BATCHES, "device": "cuda:0", "model_forward_on_ppu": True, "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_step": False, "training_started": True})
    del loader, model, dataset
    gc.collect(); torch.cuda.empty_cache()
    return {"rows": len(rows), "summary": summary_rows}


def trajectory(rows: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"incomplete window E{start}-E{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    rank1 = np.asarray([float(row["cc_rank1"]) * 100.0 for row in selected], dtype=np.float64)
    mAP = np.asarray([float(row["cc_mAP"]) * 100.0 for row in selected], dtype=np.float64)
    auc = lambda values: float((np.trapezoid(values, x) if hasattr(np, "trapezoid") else np.trapz(values, x)) / (x[-1] - x[0]))
    best_rank1 = max(selected, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    best_map = max(selected, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {"start_epoch": start, "end_epoch": end, "epochs": len(selected), "mean_rank1": float(np.mean(rank1)), "mean_mAP": float(np.mean(mAP)), "auc_rank1": auc(rank1), "auc_mAP": auc(mAP), "slope_rank1": float(np.polyfit(x, rank1, 1)[0]), "slope_mAP": float(np.polyfit(x, mAP, 1)[0]), "E60_rank1": float(rank1[-1]) if end == FORMAL_EPOCHS else None, "E60_mAP": float(mAP[-1]) if end == FORMAL_EPOCHS else None, "best_rank1": float(best_rank1["cc_rank1"]) * 100.0, "best_rank1_epoch": int(best_rank1["epoch"]), "best_mAP": float(best_map["cc_mAP"]) * 100.0, "best_mAP_epoch": int(best_map["epoch"])}


def amp_summary(run_dir: Path, arm: str) -> dict[str, Any]:
    rows = read_csv(run_dir / "amp_audit.csv")
    if len(rows) != FORMAL_EPOCHS: raise RuntimeError(f"{arm} AMP rows incomplete")
    return {"arm": arm, "total_skips": sum(int(row["skipped_steps"]) for row in rows), "total_found_inf": sum(int(row["found_inf_count"]) for row in rows), "total_nonfinite_events": sum(int(row["nonfinite_events"]) for row in rows), "skip_epochs": ",".join(row["epoch"] for row in rows if int(row["skipped_steps"]) > 0), "scale_min": min(float(row["amp_scale_min"]) for row in rows), "scale_max": max(float(row["amp_scale_max"]) for row in rows), "scale_final": float(rows[-1]["amp_scale_end"]), "planned_updates": sum(int(row["planned_updates"]) for row in rows), "completed_updates": sum(float(row["updates_completed"]) for row in rows)}


def format_metric(value: Any, digits: int = 4) -> str:
    if value in (None, "", "None"): return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)): return f"{float(value):.{digits}f}"
    return str(value)


def make_plots(late_rows: list[dict[str, Any]], balance_rows: list[dict[str, Any]], geometry_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(PLOT_ROOT / "plot_status.json", {"available": False, "error": str(exc)})
        return
    arms = ["C0", "C1", "C2"]; labels = ["C0 Bio", "C1 Image", "C2 Prototype"]; x = np.arange(3); width = 0.36
    late = {row["arm"]: row for row in late_rows if row.get("kind") == "arm" and row.get("window") == PRIMARY_WINDOW}
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(x - width / 2, [float(late[a]["auc_rank1"]) for a in arms], width, label="Rank-1 AUC"); ax.bar(x + width / 2, [float(late[a]["auc_mAP"]) for a in arms], width, label="mAP AUC"); ax.set_xticks(x, labels); ax.set_ylabel("E51-E60 AUC (%)"); ax.legend(); fig.tight_layout(); fig.savefig(PLOT_ROOT / "late_window_auc.png", dpi=160); plt.close(fig)
    bal = {row["arm"]: row for row in balance_rows}; fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(x[:2] - width / 2, [float(bal["C1"]["coefficient_of_variation"]), float(bal["C2"]["coefficient_of_variation"])], width, label="CV"); ax.bar(x[:2] + width / 2, [float(bal["C1"]["top_10pct_share"]), float(bal["C2"]["top_10pct_share"])], width, label="Top 10% share"); ax.set_xticks(x[:2], ["C1 Image", "C2 Prototype"]); ax.set_ylabel("contribution imbalance"); ax.legend(); fig.tight_layout(); fig.savefig(PLOT_ROOT / "relation_balance.png", dpi=160); plt.close(fig)
    e60 = {row["arm"]: row for row in geometry_rows if row.get("checkpoint") == "E60"}; fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(x - width / 2, [float(e60[a]["same_pid_cross_clothes_prototype_similarity_mean"]) for a in arms], width, label="prototype positive cosine"); ax.bar(x + width / 2, [float(e60[a]["prototype_identity_margin_mean"]) for a in arms], width, label="prototype margin"); ax.set_xticks(x, labels); ax.legend(); fig.tight_layout(); fig.savefig(PLOT_ROOT / "prototype_geometry.png", dpi=160); plt.close(fig)
    write_json(PLOT_ROOT / "plot_status.json", {"available": True})


def analyze() -> dict[str, Any]:
    audit = load_source_audit(); coverage_summary = json.loads((FEASIBILITY_ROOT / "coverage_summary.json").read_text())
    if not coverage_summary.get("coverage_pass"): raise RuntimeError("coverage gate did not pass")
    all_metrics: list[dict[str, Any]] = []; trajectories: list[dict[str, Any]] = []; lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, run_dir in RUN_DIRS.items():
        marker = run_dir / "RUN_COMPLETE.json"
        if not marker.is_file(): raise RuntimeError(f"{arm} incomplete")
        complete = json.loads(marker.read_text())
        if complete.get("epochs_completed") != FORMAL_EPOCHS or complete.get("rows") != FORMAL_EPOCHS: raise RuntimeError(f"{arm} completion marker invalid")
        rows = read_csv(run_dir / "metrics_per_epoch.csv")
        if len(rows) != FORMAL_EPOCHS: raise RuntimeError(f"{arm} metrics incomplete")
        all_metrics.extend(rows)
        for start, end in WINDOWS:
            item = {"kind": "arm", "arm": arm, "seed": SEED, "window": f"E{start}-E{end}", **trajectory(rows, start, end)}; trajectories.append(item); lookup[(arm, item["window"])] = item
    comparison_rows: list[dict[str, Any]] = []
    for window in [f"E{start}-E{end}" for start, end in WINDOWS]:
        for left, right, label in (("C1", "C0", "C1-C0"), ("C2", "C0", "C2-C0"), ("C2", "C1", "C2-C1")):
            l = lookup[(left, window)]; r = lookup[(right, window)]
            comparison_rows.append({"kind": "delta", "comparison": label, "window": window, "rank1_mean_pp": l["mean_rank1"] - r["mean_rank1"], "mAP_mean_pp": l["mean_mAP"] - r["mean_mAP"], "rank1_auc_pp": l["auc_rank1"] - r["auc_rank1"], "mAP_auc_pp": l["auc_mAP"] - r["auc_mAP"], "rank1_slope_pp": l["slope_rank1"] - r["slope_rank1"], "mAP_slope_pp": l["slope_mAP"] - r["slope_mAP"], "E60_rank1_pp": (l["E60_rank1"] - r["E60_rank1"]) if l["E60_rank1"] is not None else None, "E60_mAP_pp": (l["E60_mAP"] - r["E60_mAP"]) if l["E60_mAP"] is not None else None, "best_rank1_pp": l["best_rank1"] - r["best_rank1"], "best_mAP_pp": l["best_mAP"] - r["best_mAP"]})
    write_csv(TABLE_ROOT / "metrics_per_epoch.csv", all_metrics); write_csv(TABLE_ROOT / "window_analysis.csv", trajectories + comparison_rows)
    primary = {"arm": [lookup[(arm, PRIMARY_WINDOW)] for arm in RUN_DIRS], "comparisons": [row for row in comparison_rows if row["window"] == PRIMARY_WINDOW]}; write_csv(TABLE_ROOT / "late_window_primary.csv", primary["comparisons"])
    geometry_rows = []
    for arm in RUN_DIRS:
        path = GEOMETRY_ROOT / f"{arm}_geometry.csv"
        if not path.is_file(): raise RuntimeError(f"missing geometry {path}")
        geometry_rows.extend(read_csv(path))
    write_csv(GEOMETRY_ROOT / "geometry_comparison.csv", geometry_rows)
    amp_rows = [amp_summary(RUN_DIRS[arm], arm) for arm in RUN_DIRS]; write_csv(TABLE_ROOT / "amp_audit.csv", amp_rows)
    balance_rows = read_csv(BALANCE_ROOT / "planned_balance_summary.csv"); write_csv(TABLE_ROOT / "relation_balance.csv", balance_rows)
    gradient_rows = read_csv(GRADIENT_ROOT / "gradient_summary.csv")
    historical = read_csv(PHASE7A_ROOT / "late_window_analysis.csv")
    historical_p1 = next(row for row in historical if row.get("kind") == "arm" and row.get("arm") == "P1" and row.get("window") == PRIMARY_WINDOW)
    c1_primary = lookup[("C1", PRIMARY_WINDOW)]
    write_csv(TABLE_ROOT / "historical_c1_comparison.csv", [{"metric": "Rank-1 AUC", "C1_seed1234": c1_primary["auc_rank1"], "Phase7A_P1": float(historical_p1["auc_rank1"])}, {"metric": "mAP AUC", "C1_seed1234": c1_primary["auc_mAP"], "Phase7A_P1": float(historical_p1["auc_mAP"])}])
    balance = {row["arm"]: row for row in balance_rows}; q1 = "YES" if float(balance["C2"]["coefficient_of_variation"]) < float(balance["C1"]["coefficient_of_variation"]) and float(balance["C2"]["top_10pct_share"]) < float(balance["C1"]["top_10pct_share"]) else ("NO" if float(balance["C2"]["coefficient_of_variation"]) > float(balance["C1"]["coefficient_of_variation"]) else "MIXED")
    c2c1 = next(row for row in primary["comparisons"] if row["comparison"] == "C2-C1"); c2c0 = next(row for row in primary["comparisons"] if row["comparison"] == "C2-C0"); c1c0 = next(row for row in primary["comparisons"] if row["comparison"] == "C1-C0")
    q2 = "YES" if c2c1["rank1_auc_pp"] > 0 else "NO"; q3 = "YES" if c2c1["mAP_auc_pp"] > 0 else "NO"; q4 = "YES" if (c2c0["rank1_auc_pp"] > 0 or c2c0["mAP_auc_pp"] > 0) and c2c0["rank1_auc_pp"] >= -0.5 and c2c0["mAP_auc_pp"] >= -0.5 else ("MIXED" if c2c0["rank1_auc_pp"] > 0 or c2c0["mAP_auc_pp"] > 0 else "NO")
    e60_geometry = {arm: next(row for row in geometry_rows if row["arm"] == arm and row["checkpoint"] == "E60") for arm in RUN_DIRS}; q5 = "YES" if float(e60_geometry["C2"]["same_pid_cross_clothes_prototype_similarity_mean"]) > float(e60_geometry["C1"]["same_pid_cross_clothes_prototype_similarity_mean"]) and float(e60_geometry["C2"]["prototype_identity_margin_mean"]) >= float(e60_geometry["C1"]["prototype_identity_margin_mean"]) else ("MIXED" if float(e60_geometry["C2"]["same_pid_cross_clothes_prototype_similarity_mean"]) > float(e60_geometry["C1"]["same_pid_cross_clothes_prototype_similarity_mean"]) else "NO")
    gradient_by_module = {row["module"]: row for row in gradient_rows}; comparable = []
    for module in ("late_eva", "bio_projection"):
        img = gradient_by_module[module].get("alignment_image_mean"); proto = gradient_by_module[module].get("alignment_proto_mean")
        if img not in (None, "", "None") and proto not in (None, "", "None"): comparable.append(float(proto) >= float(img) - 0.05)
    q6 = "YES" if len(comparable) >= 2 and all(comparable) else ("NO" if len(comparable) >= 2 and not any(comparable) else "MIXED")
    q7 = "YES" if coverage_summary.get("coverage_pass") else "NO"
    amp = {row["arm"]: row for row in amp_rows}; c2_vs_c1_skip = amp["C2"]["total_skips"] - amp["C1"]["total_skips"]; amp_confound = c2_vs_c1_skip >= max(4, math.ceil(0.25 * max(1, amp["C1"]["total_skips"]))) or amp["C2"]["scale_min"] <= amp["C1"]["scale_min"] / 2
    strong_gate = q7 == "YES" and (c2c1["rank1_auc_pp"] > 0.5 or c2c1["mAP_auc_pp"] > 0.5) and c2c1["rank1_auc_pp"] >= -0.5 and c2c1["mAP_auc_pp"] >= -0.5 and not amp_confound and q5 != "NO"
    promotion = "PROMOTE" if strong_gate else "STOP"; next_step = "WAIT for human confirmation before seed4321; do not auto-run." if strong_gate else "STOP; do not run seed4321. Diagnose coverage, balance, geometry, trajectory, and AMP before any new method."
    questions = {"Q1_relation_balance": q1, "Q2_C2_rank1_AUC_over_C1": q2, "Q3_C2_mAP_AUC_over_C1": q3, "Q4_C2_vs_C0_without_material_other_metric_degradation": q4, "Q5_prototype_compactness_without_nearest_negative_collapse": q5, "Q6_prototype_gradient_alignment_vs_image": q6, "Q7_prototype_coverage_sufficient": q7}
    manifest = {"phase": PHASE, "mode": "FULL_FORMAL_PRCC_THREE_ARM", "seed": SEED, "epochs": FORMAL_EPOCHS, "primary_window": PRIMARY_WINDOW, "source_audit": str(SOURCE_ROOT / "source_audit.json"), "source_report": str(SOURCE_REPORT), "feasibility": str(FEASIBILITY_ROOT / "prototype_feasibility.json"), "coverage_summary": str(FEASIBILITY_ROOT / "coverage_summary.json"), "coverage_pass": coverage_summary.get("coverage_pass"), "common_init_audit": str(TRAINING_ROOT / "common_init_audit.json"), "arms": {arm: str(RUN_DIRS[arm]) for arm in RUN_DIRS}, "training_started": True, "optimizer_step_called": True, "new_sampler": False, "lambda_rel": LAMBDA_REL, "prototype_definition": "Norm(mean(Norm(feat[0]))) within current mini-batch only; no learned/memory/EMA prototype", "metrics": str(TABLE_ROOT / "metrics_per_epoch.csv"), "window_analysis": str(TABLE_ROOT / "window_analysis.csv"), "late_window_primary": str(TABLE_ROOT / "late_window_primary.csv"), "relation_balance": str(TABLE_ROOT / "relation_balance.csv"), "geometry": str(GEOMETRY_ROOT / "geometry_comparison.csv"), "gradient": str(GRADIENT_ROOT / "gradient_summary.csv"), "amp": str(TABLE_ROOT / "amp_audit.csv"), "questions": questions, "amp_numerical_confound": amp_confound, "promotion_decision": promotion, "next_step_recommendation": next_step, "second_seed_started": False, "phase7d_started": False, "forbidden_arms_started": ["SR3", "H_ID", "DS", "DW", "GRL", "Reverse", "new sampler", "margin loss", "memory bank", "EMA", "prototype classifier"]}
    write_json(MANIFEST_PATH, manifest); make_plots([*trajectories], balance_rows, geometry_rows); write_report(manifest, audit, coverage_summary, primary, comparison_rows, balance_rows, geometry_rows, gradient_rows, amp_rows, questions, c1_primary, historical_p1, c2c1, c2c0, c1c0, amp_confound, promotion, next_step)
    return manifest


def write_report(manifest: dict[str, Any], audit: dict[str, Any], coverage_summary: dict[str, Any], primary: dict[str, Any], comparison_rows: list[dict[str, Any]], balance_rows: list[dict[str, str]], geometry_rows: list[dict[str, str]], gradient_rows: list[dict[str, str]], amp_rows: list[dict[str, Any]], questions: dict[str, str], c1_primary: dict[str, Any], historical_p1: dict[str, str], c2c1: dict[str, Any], c2c0: dict[str, Any], c1c0: dict[str, Any], amp_confound: bool, promotion: str, next_step: str) -> None:
    structure = audit["dataset"]; bal = {row["arm"]: row for row in balance_rows}; e60 = {arm: next(row for row in geometry_rows if row["arm"] == arm and row["checkpoint"] == "E60") for arm in RUN_DIRS}; amp = {row["arm"]: row for row in amp_rows}
    lines = ["# PHASE 7C — Clothing-State Prototype Relation", "", "## 1. Scope and source audit", "", "This phase is an independent PRCC seed1234 three-arm experiment. Phase 5/6/7A/7B outputs were not overwritten. No seed4321 or Phase 7D was started.", "", f"- Source audit: `{SOURCE_REPORT}`.", f"- Feasibility: `{FEASIBILITY_ROOT / 'prototype_feasibility.json'}`; coverage: `{FEASIBILITY_ROOT / 'coverage_summary.json'}`.", f"- Exact Phase 7A sources: `{PHASE7A_P0}`, `{PHASE7A_P1}`; epoch 60, seed 1234, batch 16, NUM_INSTANCES 4, 551 updates/rank/epoch.", f"- Frozen common initialization: `{PHASE7A_COMMON_INIT}`; reused by C0/C1/C2 with strict equality.", "", "## 2. Prototype construction", "", "For each current mini-batch and each PID/clothing state:", "", "`p_y,c = Norm(mean(Norm(z_i)))`, where `z_i = feat[0]`. For every same-PID pair of distinct clothing states, C2 uses `1 - cosine(p_y,ca, p_y,cb)` once. There is no learned prototype, memory bank, queue, EMA, prototype classifier, or cross-batch state. If no valid state pair is present, the prototype relation loss is differentiable zero.", "", "## 3. PRCC train-only feasibility and coverage", "", f"- Train images/PIDs/clothing states: {structure['train_images']} / {structure['pids']} / {structure['clothing_states']}.", f"- Changing-clothes PIDs: {structure['changing_clothes_pids']}; valid prototype state relations: {structure['valid_prototype_relations']}.", f"- State size mean/median/p25/p75/p90/max: {structure['state_size_mean']:.3f} / {structure['state_size_median']:.3f} / {structure['state_size_p25']:.3f} / {structure['state_size_p75']:.3f} / {structure['state_size_p90']:.3f} / {structure['state_size_max']}.", f"- Frozen sampler global batches across 60 epochs: {coverage_summary['global_batches']}.", f"- Image-level relation-active batch rate: {coverage_summary['image_relation_active_batch_rate']:.4f}.", f"- Prototype-valid batch rate: {coverage_summary['prototype_valid_batch_rate']:.4f}.", f"- Mean/median prototype relations per batch: {coverage_summary['mean_valid_prototype_relations_per_batch']:.4f} / {coverage_summary['median_valid_prototype_relations_per_batch']:.4f}.", f"- PIDs ever prototype-supervised: {coverage_summary['pids_ever_prototype_supervised']}/{coverage_summary['changing_pids']} ({coverage_summary['prototype_pid_coverage']:.4f}).", f"- Prototype state-size mean/median among active relation prototypes: {coverage_summary['prototype_state_size_mean']:.4f} / {coverage_summary['prototype_state_size_median']:.4f}.", "- Coverage gate: at least 10% valid-prototype batches and at least 80% changing-PID coverage; PASS.", "", "## 4. Arms and matched schedule", "", "| Arm | Definition | Relation |", "|---|---|---|", "| C0 | Bio control | none |", "| C1 | Bio + image-level Uniform Relation | Phase 7A P1, lambda=0.1 |", "| C2 | Bio + clothing-state prototype relation | current-batch prototypes, lambda=0.1 |", "", "- Optimizer/scheduler/AMP and evaluation are identical to Phase 7A PRCC.", "- No sampler, augmentation, backbone, optimizer, scheduler, lambda, SR3, H_ID, DS, DW, GRL, Reverse, margin, memory, or EMA modification.", "", "## 5. Late-window retrieval", "", "| Arm | Mean Rank-1 | Mean mAP | AUC Rank-1 | AUC mAP | Rank-1 slope | mAP slope |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in RUN_DIRS:
        item = primary["arm"][list(RUN_DIRS).index(arm)]; lines.append(f"| {arm} | {item['mean_rank1']:.4f} | {item['mean_mAP']:.4f} | {item['auc_rank1']:.4f} | {item['auc_mAP']:.4f} | {item['slope_rank1']:.4f} | {item['slope_mAP']:.4f} |")
    lines += ["", "| Comparison | Rank-1 AUC Δ (pp) | mAP AUC Δ (pp) | Rank-1 mean Δ (pp) | mAP mean Δ (pp) |", "|---|---:|---:|---:|---:|", f"| C1 - C0 | {c1c0['rank1_auc_pp']:.4f} | {c1c0['mAP_auc_pp']:.4f} | {c1c0['rank1_mean_pp']:.4f} | {c1c0['mAP_mean_pp']:.4f} |", f"| C2 - C0 | {c2c0['rank1_auc_pp']:.4f} | {c2c0['mAP_auc_pp']:.4f} | {c2c0['rank1_mean_pp']:.4f} | {c2c0['mAP_mean_pp']:.4f} |", f"| C2 - C1 | {c2c1['rank1_auc_pp']:.4f} | {c2c1['mAP_auc_pp']:.4f} | {c2c1['rank1_mean_pp']:.4f} | {c2c1['mAP_mean_pp']:.4f} |", "", "| Arm | E60 Rank-1 | E60 mAP | Best Rank-1 (epoch) | Best mAP (epoch) |", "|---|---:|---:|---:|---:|"]
    for arm in RUN_DIRS:
        item = primary["arm"][list(RUN_DIRS).index(arm)]; lines.append(f"| {arm} | {item['E60_rank1']:.4f} | {item['E60_mAP']:.4f} | {item['best_rank1']:.4f} (E{item['best_rank1_epoch']}) | {item['best_mAP']:.4f} (E{item['best_mAP_epoch']}) |")
    lines += ["", f"C1 versus historical Phase 7A P1: Rank-1 AUC {c1_primary['auc_rank1']:.4f} vs {float(historical_p1['auc_rank1']):.4f}; mAP AUC {c1_primary['auc_mAP']:.4f} vs {float(historical_p1['auc_mAP']):.4f}. See `tables/historical_c1_comparison.csv`.", "", "## 6. Relation balance", "", "| Arm | Mean | Median | P90 | Max | CV | Top 1% share | Top 5% share | Top 10% share |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("C1", "C2"):
        row = bal[arm]; lines.append(f"| {arm} | {float(row['mean']):.3f} | {float(row['median']):.3f} | {float(row['p90']):.3f} | {float(row['max']):.3f} | {float(row['coefficient_of_variation']):.4f} | {float(row['top_1pct_share']):.4f} | {float(row['top_5pct_share']):.4f} | {float(row['top_10pct_share']):.4f} |")
    lines += ["", "C2 is counted once per valid state relation per batch; C1 counts every image-level relation contribution. Full counts are in `relation_balance/planned_contribution_counts.csv`.", "", "## 7. Geometry", "", "| Arm | Cross image cosine | Cross prototype cosine | Nearest-neg prototype | Prototype margin | Same-clothes cosine |", "|---|---:|---:|---:|---:|---:|"]
    for arm in RUN_DIRS:
        row = e60[arm]; lines.append(f"| {arm} | {float(row['same_pid_cross_clothes_image_similarity_mean']):.6f} | {float(row['same_pid_cross_clothes_prototype_similarity_mean']):.6f} | {float(row['nearest_negative_prototype_similarity_mean']):.6f} | {float(row['prototype_identity_margin_mean']):.6f} | {float(row['same_pid_same_clothes_similarity_mean']):.6f} |")
    lines += ["", f"C2 versus C1 E60 prototype cosine Δ: {float(e60['C2']['same_pid_cross_clothes_prototype_similarity_mean']) - float(e60['C1']['same_pid_cross_clothes_prototype_similarity_mean']):.6f}; prototype-margin Δ: {float(e60['C2']['prototype_identity_margin_mean']) - float(e60['C1']['prototype_identity_margin_mean']):.6f}.", "", "## 8. Gradient diagnostic", "", "Unscaled FP32 gradients were computed on 16 matched train-only micro-batches from the C0 E60 model. No optimizer was created or stepped in this diagnostic.", "", "| Module | Image A_rel | Prototype A_rel | Image norm | Prototype norm |", "|---|---:|---:|---:|---:|"]
    for row in gradient_rows: lines.append(f"| {row['module']} | {format_metric(row.get('alignment_image_mean'))} | {format_metric(row.get('alignment_proto_mean'))} | {format_metric(row.get('norm_image_mean'))} | {format_metric(row.get('norm_proto_mean'))} |")
    lines += ["", "ID-head relation gradients are zero/undefined because the relation term uses the identity feature before the classifier head; they are reported as NA, not imputed.", "", "## 9. AMP audit", "", "| Arm | Total skips | found_inf | Scale min | Scale final |", "|---|---:|---:|---:|---:|"]
    for arm in RUN_DIRS:
        row = amp[arm]; lines.append(f"| {arm} | {row['total_skips']} | {row['total_found_inf']} | {row['scale_min']:.1f} | {row['scale_final']:.1f} |")
    lines += ["", f"Obvious C2-vs-C1 AMP confound: **{'YES' if amp_confound else 'NO'}**.", "", "## 10. Q1-Q7", ""]
    for key, value in questions.items(): lines.append(f"- {key}: **{value}**.")
    lines += ["", "## 11. Promotion decision", "", f"**{promotion}**.", f"{next_step}", "", "The preferred strong gate requires Q7=YES, C2-C1 Rank-1 or mAP late AUC > +0.5 pp, the other metric not materially negative (>-0.5 pp), no obvious AMP confound, and no geometry margin collapse.", "", "## 12. STOP", "", "Phase 7C seed1234 is complete. No seed4321, Phase 7D, lambda tuning, sampler change, SR3, margin loss, memory bank, EMA, or new architecture was started.", "", "## Artifacts", "", f"- Manifest: `{MANIFEST_PATH}`", f"- Training root: `{TRAINING_ROOT}`", f"- Window tables: `{TABLE_ROOT / 'window_analysis.csv'}`, `{TABLE_ROOT / 'late_window_primary.csv'}`", f"- Relation balance: `{BALANCE_ROOT}`", f"- Geometry: `{GEOMETRY_ROOT / 'geometry_comparison.csv'}`", f"- Gradient: `{GRADIENT_ROOT / 'gradient_summary.csv'}`", f"- AMP: `{TABLE_ROOT / 'amp_audit.csv'}`"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("audit", "coverage", "prepare", "train", "gradient", "analyze"), required=True); parser.add_argument("--arm", choices=tuple(RUN_DIRS), default=None); return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        result = source_audit(); print(json.dumps({"source_verification_pass": result["source_verification_pass"], "source_report": str(SOURCE_REPORT), "feasibility": str(FEASIBILITY_ROOT / "prototype_feasibility.json"), "training_started": False}, indent=2, sort_keys=True))
    elif args.mode == "coverage":
        result = coverage(); print(json.dumps(result, indent=2, sort_keys=True))
    elif args.mode == "prepare":
        result = prepare(); print(json.dumps(result, indent=2, sort_keys=True))
    elif args.mode == "train":
        if args.arm is None: raise SystemExit("--arm C0|C1|C2 is required for train")
        train_arm(args.arm)
    elif args.mode == "gradient":
        print(json.dumps(gradient_probe(), indent=2, sort_keys=True))
    else:
        print(json.dumps(analyze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
