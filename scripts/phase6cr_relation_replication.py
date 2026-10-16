#!/usr/bin/env python3
"""Phase 6C-R: independent-seed replication of frozen B0/B1 supervision.

Only R0 (Bio) and R1 (uniform cross-clothes relation) are supported here.
The relation implementation is imported from the executed Phase 6B runner;
this file provides the seed-4321 common initialization, matched full-schedule
training, train-only geometry audit, and two-seed report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import phase6c_full_schedule_confirmation as p6c  # noqa: E402


PHASE = "phase6c_r_relation_replication"
SEED = 4321
WORLD_SIZE = 2
TOTAL_EPOCHS = 60
WARMUP_EPOCHS = 20
STEPS_PER_EPOCH = 596
BATCH_SIZE = 8
NUM_INSTANCES = 2
NUM_CLASSES = 77
NUM_CLOTHING_STATES = 256
GEOMETRY_EPOCHS = (20, 30, 40, 50, 60)
WINDOWS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60))
PRIMARY_WINDOW = "E51-E60"

OUT_ROOT = ROOT / "outputs/new_method_track/phase6cr_relation_replication"
ARTIFACT_ROOT = OUT_ROOT / "artifacts"
COMMON_INIT = OUT_ROOT / "SEED4321_COMMON_INIT.pth"
COMMON_INIT_META = OUT_ROOT / "SEED4321_COMMON_INIT_HASHES.json"
SEED_AUDIT = ARTIFACT_ROOT / "seed4321_audit.md"
COMMON_AUDIT = ARTIFACT_ROOT / "common_init_seed4321_audit.json"
METHOD_DOC = ARTIFACT_ROOT / "frozen_uniform_relation_method.md"
REPORT_PATH = ROOT / "reports/PHASE_6CR_CROSS_CLOTHES_RELATION_REPLICATION.md"
MANIFEST_PATH = OUT_ROOT / "PHASE6CR_MANIFEST.json"
PHASE6C_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
PHASE6B_ROOT = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen"

RUN_DIRS = {
    "R0": OUT_ROOT / "R0_seed4321",
    "R1": OUT_ROOT / "R1_seed4321",
}
ARM_LABELS = {
    "R0": "Bio baseline",
    "R1": "Uniform cross-clothes relation",
}
TRAIN_ARM = {"R0": "B0", "R1": "B1"}

METRIC_FIELDS = list(b6.METRIC_FIELDS) + [
    "effective_updates_completed",
    "relation_unique_cross_clothes_pairs",
    "relation_pid_coverage",
    "relation_clothing_state_coverage",
]
RELATION_FIELDS = [
    "arm", "seed", "epoch", "all_images_available", "all_states_available",
    "all_pids_available", "sampled_unique_images", "sampled_unique_states",
    "sampled_unique_pids", "relation_active_batches", "relation_active_batch_rate",
    "relation_pairs", "relation_unique_cross_clothes_pairs", "relation_unique_state_pairs",
    "relation_pid_coverage", "relation_clothing_state_coverage", "relation_weight_mean",
    "relation_weight_std", "relation_weight_min", "relation_weight_max",
    "optimizer_updates", "effective_updates_completed",
]


def patch_frozen_modules() -> None:
    """Make the reused sampler/config functions operate under seed 4321."""
    base.PHASE = PHASE
    base.SEED = SEED
    base.TOTAL_EPOCHS = TOTAL_EPOCHS
    b6.PHASE = PHASE
    b6.SEED = SEED
    b6.TOTAL_EPOCHS = TOTAL_EPOCHS


patch_frozen_modules()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.detach().cpu().item()
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


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        key: sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for key, value in sorted(state.items())
    }


def model_state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(json.dumps(tensor_hashes(state), sort_keys=True).encode())


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure() -> Any:
    cfg = base.configure()
    cfg.defrost()
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.MAX_EPOCHS = TOTAL_EPOCHS
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.SOLVER.CHECKPOINT_PERIOD = TOTAL_EPOCHS
    cfg.SOLVER.EVAL_PERIOD = 1
    cfg.OUTPUT_DIR = str(OUT_ROOT)
    cfg.TRAIN.START_EPOCH = 1
    cfg.freeze()
    return cfg


def load_common_init() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not COMMON_INIT.is_file() or not COMMON_INIT_META.is_file():
        raise FileNotFoundError("seed4321 common initialization is missing; run --mode prepare")
    meta = json.loads(COMMON_INIT_META.read_text())
    file_hash = sha256_file(COMMON_INIT)
    if file_hash != meta["state_sha256"]:
        raise RuntimeError("seed4321 common initialization SHA256 mismatch")
    state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    hashes = tensor_hashes(state)
    if hashes != meta["tensor_hashes"]:
        raise RuntimeError("seed4321 common initialization tensor hash mismatch")
    audit = {
        "path": COMMON_INIT,
        "sha256": file_hash,
        "meta_path": COMMON_INIT_META,
        "meta_sha256": sha256_file(COMMON_INIT_META),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "strict_hash_match": True,
    }
    return state, audit


def phase6c_frozen_reference() -> dict[str, Any]:
    manifest_path = PHASE6C_ROOT / "PHASE6C_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("seed") != 1234 or manifest.get("phase6d_started"):
        raise RuntimeError("Phase6C seed1234 frozen reference is not the expected completed reference")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seed": manifest.get("seed"),
        "phase6d_started": manifest.get("phase6d_started"),
        "report": manifest.get("report"),
        "b0_b1_primary": {
            "B0": next(row for row in manifest["summaries"] if row["arm"] == "B0" and row["window"] == PRIMARY_WINDOW),
            "B1": next(row for row in manifest["summaries"] if row["arm"] == "B1" and row["window"] == PRIMARY_WINDOW),
        },
    }


def source_checksums() -> dict[str, Any]:
    paths = [
        Path(__file__),
        Path(b6.__file__),
        Path(base.__file__),
        ROOT / "scripts/phase6c_full_schedule_confirmation.py",
        PHASE6B_ROOT / "PREFLIGHT.json",
        PHASE6B_ROOT / "PHASE6B_MANIFEST.json",
        PHASE6C_ROOT / "PHASE6C_MANIFEST.json",
        b6.A3_MANIFEST,
        b6.A3_SCORES,
        b6.A3_COMPONENTS,
        b6.A2_CROSS,
    ]
    return {
        str(path): {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size": path.stat().st_size if path.is_file() else None,
        }
        for path in paths
    }


def optimizer_config(cfg: Any) -> dict[str, Any]:
    return {
        "name": str(cfg.SOLVER.OPTIMIZER_NAME),
        "base_lr": float(cfg.SOLVER.BASE_LR),
        "bias_lr_factor": float(cfg.SOLVER.BIAS_LR_FACTOR),
        "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
        "weight_decay_bias": float(cfg.SOLVER.WEIGHT_DECAY_BIAS),
        "momentum": float(cfg.SOLVER.MOMENTUM),
    }


def scheduler_config(cfg: Any) -> dict[str, Any]:
    return {
        "factory": "solver.scheduler_factory.create_scheduler",
        "factory_sha256": sha256_file(ROOT / "solver/scheduler_factory.py"),
        "class": "CosineLRScheduler",
        "t_initial": int(cfg.SOLVER.MAX_EPOCHS),
        "warmup_t": int(cfg.SOLVER.WARMUP_EPOCHS),
        "warmup_method": str(cfg.SOLVER.WARMUP_METHOD),
        "warmup_lr_init": float(cfg.SOLVER.WARMUP_LR),
        "lr_min": float(0.1 * cfg.SOLVER.BASE_LR),
        "t_mul": 1.0,
        "decay_rate": 0.1,
        "cycle_limit": 1,
        "t_in_epochs": True,
        "noise_seed": 42,
    }


def amp_config() -> dict[str, Any]:
    return {
        "autocast_enabled": True,
        "grad_scaler": "torch.cuda.amp.GradScaler",
        "init_scale": int(base.INITIAL_SCALE),
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "settings_modified_for_replication": False,
    }


def generate_common_init(cfg: Any) -> dict[str, Any]:
    if COMMON_INIT.exists() or COMMON_INIT_META.exists():
        state, audit = load_common_init()
        return {"state": audit, "already_present": True}
    set_seed(SEED)
    dataset = base.build_dataset(cfg)
    if len(dataset.train) != 9576 or dataset.num_train_pids != NUM_CLASSES:
        raise RuntimeError("seed4321 common init dataset topology mismatch")
    raw = base.build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    raw.eval()
    state = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
    if any("reverse" in key.lower() or "grl" in key.lower() for key in state):
        raise RuntimeError("seed4321 common init contains a forbidden Reverse/GRL parameter")
    COMMON_INIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = COMMON_INIT.with_suffix(".pth.tmp")
    torch.save(state, temporary)
    temporary.replace(COMMON_INIT)
    meta = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "topology": "EVA02-Large Bio-only: CE + Triplet + Summary Bio",
        "reverse": "OFF",
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "state_sha256": sha256_file(COMMON_INIT),
        "tensor_hashes": tensor_hashes(state),
    }
    write_json(COMMON_INIT_META, meta)
    (OUT_ROOT / "seed4321_common_init_config.yml").write_text(cfg.dump(), encoding="utf-8")
    audit = {
        "path": COMMON_INIT,
        "sha256": meta["state_sha256"],
        "meta_path": COMMON_INIT_META,
        "meta_sha256": sha256_file(COMMON_INIT_META),
        "state_key_count": len(state),
        "state_numel": meta["state_numel"],
        "strict_hash_match": True,
    }
    del raw, state, dataset
    gc.collect()
    return {"state": audit, "already_present": False}


def sampler_audit(cfg: Any, dataset: Any, active_state: dict[int, list[int]], index_map: dict[tuple[int, int], list[int]]) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_meta = []
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, base.build_img_transforms(cfg)[0], active_state, index_map, epoch, rank)
            del loader
            if meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"]:
                raise RuntimeError(f"seed4321 sampler mismatch at epoch {epoch}, rank {rank}")
            epoch_meta.append(meta)
            rows.append({
                "seed": SEED,
                "epoch": epoch,
                "rank": rank,
                "batches": meta["batches"],
                "pair_units": meta["pair_units"],
                "active_images": meta["active_images"],
                "active_pids": meta["active_pids"],
                "active_clothes": meta["active_clothes"],
                "covered_pid_count": len(meta["covered_pids"]),
                "batch_unique_pid_min": meta["batch_unique_pid_min"],
                "batch_unique_pid_max": meta["batch_unique_pid_max"],
                "plan_sha256": meta["plan_sha256"],
            })
        if set().union(*(set(meta["covered_pids"]) for meta in epoch_meta)) != set(range(NUM_CLASSES)):
            raise RuntimeError(f"seed4321 sampler global PID coverage mismatch at epoch {epoch}")
    return rows


def write_method_doc(cfg: Any, bank: dict[str, Any]) -> None:
    relation_source = inspect.getsource(b6.relation_loss)
    forward_source = inspect.getsource(b6.forward_batch)
    lines = [
        "# Frozen uniform cross-clothes relation method",
        "",
        "Executed code is authoritative. R1 calls the Phase6B `relation_loss` and `forward_batch` implementation; no textual reimplementation is used for training.",
        "",
        f"- Phase6B runner: `{b6.__file__}` (SHA256 `{sha256_file(Path(b6.__file__))}`).",
        f"- Phase6C runner/manifest source: `{ROOT / 'scripts/phase6c_full_schedule_confirmation.py'}` (SHA256 `{sha256_file(ROOT / 'scripts/phase6c_full_schedule_confirmation.py')}`).",
        f"- Phase6B manifest: `{PHASE6B_ROOT / 'PHASE6B_MANIFEST.json'}` (SHA256 `{sha256_file(PHASE6B_ROOT / 'PHASE6B_MANIFEST.json')}`).",
        f"- Candidate bank: `{b6.A2_CROSS}` and Phase6A3 files; unique image-pair count `{bank['exact_pair_count']}`, source rows `{bank['relation_count']}`.",
        "",
        "## Executed R1 formula",
        "",
        "For each in-batch pair with the same PID and different clothing state, the actual implementation uses normalized identity features and `1 - cosine(feature_i, feature_j)`. R1 assigns `weight = 1.0` to every frozen candidate pair. The weighted mean is divided by the sum of weights; an empty candidate set returns a differentiable zero. The applied total is `base_loss + 0.1 * relation_loss`.",
        "",
        f"- `lambda_rel = {b6.LAMBDA_REL}`.",
        "- `w_ij = 1.0` for R1.",
        "- Non-candidate same-PID/different-clothes pairs receive no new relation term, exactly as in the executed candidate-bank implementation.",
        "- No SR3 weighting, reliability score, H_ID, active-set withholding, sampler change, or clothing-state hiding is used.",
        "",
        "## Frozen configuration",
        "",
        f"- LTCC train: `{len([]) if False else 9576}` images, 77 PIDs, 256 clothing states; image size 224x224.",
        f"- Batch `{BATCH_SIZE}`, NUM_INSTANCES `{NUM_INSTANCES}`, two PPUs, `{STEPS_PER_EPOCH}` planned updates/rank/epoch.",
        f"- Optimizer: `{json.dumps(optimizer_config(cfg), sort_keys=True)}`.",
        f"- Scheduler: `{json.dumps(scheduler_config(cfg), sort_keys=True)}`.",
        f"- AMP: `{json.dumps(amp_config(), sort_keys=True)}`.",
        "- Reverse, GRL, DW, DS, H_ID, online reliability, new sampler, curriculum, and PRCC: OFF/not entered.",
        "",
        "## Source excerpt: relation_loss",
        "",
        "```python",
        relation_source.rstrip(),
        "```",
        "",
        "## Source excerpt: forward_batch",
        "",
        "```python",
        forward_source.rstrip(),
        "```",
    ]
    METHOD_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_audit(cfg: Any, common: dict[str, Any], sampler_rows: list[dict[str, Any]], observed: dict[str, Any] | None = None) -> None:
    phase6c = phase6c_frozen_reference()
    lines = [
        "# Phase 6C-R seed4321 audit",
        "",
        f"Generated: {utc_now()}.",
        "",
        "## Independent seed",
        "",
        f"- Python / NumPy / Torch / CUDA seed: `{SEED}`.",
        "- cuDNN deterministic: `True`; cuDNN benchmark: `False`.",
        "- DataLoader augmentation seeds and fixed sampler plan include seed4321, epoch, rank, iteration, position, and dataset index.",
        "- Both R0 and R1 load the same strict seed4321 common initialization.",
        "",
        "## Frozen reference",
        "",
        f"- Phase6C seed1234 manifest: `{phase6c['manifest']}`.",
        f"- Phase6C seed1234 B0/B1 E51-E60 reference is present; Phase6D started: `{phase6c['phase6d_started']}`.",
        "",
        "## Configuration checks",
        "",
        f"- Dataset: LTCC, 9576 train images, 77 PIDs, 256 clothing states, 224x224.",
        f"- Devices: 2; batch `{BATCH_SIZE}`; NUM_INSTANCES `{NUM_INSTANCES}`; planned updates/rank/epoch `{STEPS_PER_EPOCH}`.",
        f"- Optimizer: `{json.dumps(optimizer_config(cfg), sort_keys=True)}`.",
        f"- Scheduler: `{json.dumps(scheduler_config(cfg), sort_keys=True)}`.",
        f"- AMP: `{json.dumps(amp_config(), sort_keys=True)}`.",
        "- Reverse/GRL/DW/DS/H_ID/online reliability/new sampler/curriculum/PRCC: OFF/not entered.",
        "",
        "## Sampler/DataLoader audit",
        "",
        f"- Rows checked: `{len(sampler_rows)}` (60 epochs × 2 ranks).",
        f"- Every row has 596 batches, full PID coverage, valid pair/batch size, and full clothing-state/image availability.",
        "- R0 and R1 use the same deterministic sampler plan; only the R1 uniform relation term differs.",
        "",
        "## Common init",
        "",
        f"- Common init: `{common['state']['path']}`.",
        f"- Model state SHA256: `{common['state']['sha256']}`.",
        f"- Strict tensor hash check: `{common['state']['strict_hash_match']}`.",
    ]
    if observed:
        lines += [
            "",
            "## Runtime strict-load verification",
            "",
            f"- R0 initial parameter hash: `{observed.get('R0_initial_parameter_hash')}`.",
            f"- R1 initial parameter hash: `{observed.get('R1_initial_parameter_hash')}`.",
            f"- Initial parameter exact equality: `{observed.get('initial_parameter_equality')}`.",
            f"- R0/R1 common-init SHA256 equality: `{observed.get('common_init_equality')}`.",
        ]
    SEED_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = configure()
    common = generate_common_init(cfg)
    state, _audit = load_common_init()
    dataset = base.build_dataset(cfg)
    if len(dataset.train) != 9576:
        raise RuntimeError("unexpected LTCC train image count")
    index_map = base.build_index_map(dataset)
    active_state = b6.full_active_state(dataset)
    sampler_rows = sampler_audit(cfg, dataset, active_state, index_map)
    write_csv(OUT_ROOT / "sampler_audit_seed4321.csv", sampler_rows, list(sampler_rows[0]))
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    write_method_doc(cfg, bank)
    common_audit = {
        "phase": PHASE,
        "seed": SEED,
        "created_utc": utc_now(),
        "common_init": common["state"],
        "model_state_sha256": common["state"]["sha256"],
        "initial_parameter_equality": "PENDING_RUNTIME_R0_R1_STRICT_LOAD_CHECK",
        "optimizer_config": optimizer_config(cfg),
        "scheduler_config": scheduler_config(cfg),
        "amp_config": amp_config(),
        "configuration_hash": sha256_bytes(cfg.dump().encode()),
        "source_checksums": source_checksums(),
        "relation_bank_audit": {
            "relation_count": bank["relation_count"],
            "exact_pair_count": bank["exact_pair_count"],
            "state_pair_count": bank["state_pair_count"],
            "candidate_source": str(b6.A2_CROSS),
            "candidate_source_sha256": sha256_file(b6.A2_CROSS),
        },
        "forbidden_arms_started": [],
    }
    write_json(COMMON_AUDIT, common_audit)
    write_seed_audit(cfg, common, sampler_rows)
    write_json(OUT_ROOT / "PREPARE.json", {
        "phase": PHASE,
        "prepared_utc": utc_now(),
        "seed": SEED,
        "common_init": common_audit,
        "sampler_audit": str(OUT_ROOT / "sampler_audit_seed4321.csv"),
        "method_doc": str(METHOD_DOC),
        "allowed_arms": ["R0", "R1"],
        "forbidden_arms": ["B2", "B3", "H_ID", "DS", "DW", "Relation Curriculum", "GRL", "Reverse", "PRCC", "diffusion"],
    })
    del state, dataset, bank
    gc.collect()
    print(json.dumps({"prepared": True, "root": str(OUT_ROOT), "common_init": str(COMMON_INIT), "audit": str(SEED_AUDIT)}, indent=2, sort_keys=True))


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def relation_audit_keys(batch: dict[str, Any], bank: dict[str, Any], path_to_index: dict[str, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    pids = batch["pid"].detach().cpu().tolist()
    clothes = batch["clothes_id"].detach().cpu().tolist()
    paths = [str(path) for path in batch["image_path"]]
    indices = [path_to_index[path] for path in paths]
    exact_keys: set[tuple[int, int]] = set()
    state_keys: set[tuple[int, int, int]] = set()
    pid_keys: set[tuple[int, int]] = set()
    clothing_keys: set[tuple[int, int]] = set()
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if int(pids[i]) != int(pids[j]) or int(clothes[i]) == int(clothes[j]):
                continue
            exact = (min(indices[i], indices[j]), max(indices[i], indices[j]))
            if exact not in bank["exact_lookup"]:
                continue
            state_key = (int(pids[i]), min(int(clothes[i]), int(clothes[j])), max(int(clothes[i]), int(clothes[j])))
            exact_keys.add(exact)
            state_keys.add(state_key)
            pid_keys.add((int(pids[i]), 0))
            clothing_keys.add((int(pids[i]), int(clothes[i])))
            clothing_keys.add((int(pids[j]), int(clothes[j])))
    return exact_keys, state_keys, pid_keys, clothing_keys


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE,
        "arm": path.parent.name,
        "epoch": epoch,
        "global_step": global_step,
        "model": base.cpu_state_dict(model),
        "optimizer": base.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": base.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def open_csv_append(path: Path, fields: list[str]):
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    stream.flush()
    return stream, writer


def train_arm(arm: str) -> None:
    if arm not in RUN_DIRS:
        raise ValueError(f"only R0/R1 are permitted, got {arm}")
    prepare_path = OUT_ROOT / "PREPARE.json"
    if not prepare_path.is_file():
        raise RuntimeError("run --mode prepare before training")
    run_dir = RUN_DIRS[arm]
    if (run_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"refusing to overwrite completed run: {run_dir}")
    rank, device = b6.init_dist()
    metrics_stream = amp_stream = relation_stream = event_stream = sampler_stream = None
    writer = logger = None
    try:
        cfg = configure()
        set_seed(SEED)
        state, init_audit = load_common_init()
        model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, state, device)
        initial_hash = b6.model_parameter_hash(model)
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, initial_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"{arm} initial model differs across ranks")
        dataset = base.build_dataset(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        transform_train, transform_test = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        active_state = b6.full_active_state(dataset)
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        val_loader = base.build_val_loader(cfg, dataset)
        train_arm_name = TRAIN_ARM[arm]
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            manifest = {
                "phase": PHASE,
                "arm": arm,
                "label": ARM_LABELS[arm],
                "executed_training_arm": train_arm_name,
                "seed": SEED,
                "epochs": TOTAL_EPOCHS,
                "start_epoch": 1,
                "end_epoch": TOTAL_EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "train_images": len(dataset.train),
                "train_pids": NUM_CLASSES,
                "train_clothing_states": NUM_CLOTHING_STATES,
                "batch_size": BATCH_SIZE,
                "num_instances": NUM_INSTANCES,
                "optimizer": optimizer_config(cfg),
                "scheduler": scheduler_config(cfg),
                "amp": amp_config(),
                "common_init": init_audit,
                "initial_parameter_hash": initial_hash,
                "common_init_parameter_strict_load": True,
                "reverse": "OFF",
                "grl": "OFF",
                "dw": "OFF",
                "ds": "OFF",
                "h_id": "OFF",
                "online_reliability": "OFF",
                "active_set_withholding": "OFF",
                "relation_loss": "executed Phase6B positive-only same-PID/different-clothes cosine loss",
                "relation_weight": "uniform w_ij=1.0" if arm == "R1" else "not applied in R0",
                "lambda_rel": b6.LAMBDA_REL,
                "relation_source": str(b6.A2_CROSS),
                "relation_source_sha256": sha256_file(b6.A2_CROSS),
                "only_method_difference": "R1 adds the uniform cross-clothes relation term; R0 does not.",
            }
            write_json(run_dir / "run_manifest.json", manifest)
            logger = base.logger_for(run_dir / "train_log_rank0.txt")
            writer = SummaryWriter(str(run_dir / "TensorBoard"))
            metrics_stream, metrics_writer = open_csv_append(run_dir / "metrics_per_epoch.csv", METRIC_FIELDS)
            amp_stream, amp_writer = open_csv_append(run_dir / "amp_statistics.csv", ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end", "effective_updates_completed"])
            relation_stream, relation_writer = open_csv_append(run_dir / "relation_exposure.csv", RELATION_FIELDS)
            event_stream, event_writer = open_csv_append(run_dir / "amp_skip_events.csv", ["rank", "epoch", "iteration", "global_step", "skipped_step", "found_inf", "nonfinite_event", "amp_scale_before", "amp_scale_after"])
            sampler_stream, sampler_writer = open_csv_append(run_dir / "sampler_plan.csv", ["seed", "arm", "epoch", "rank", "batches", "pair_units", "active_images", "active_pids", "active_clothes", "covered_pid_count", "batch_unique_pid_min", "batch_unique_pid_max", "plan_sha256"])
        else:
            metrics_writer = amp_writer = relation_writer = event_writer = sampler_writer = None

        best_r1 = -float("inf")
        best_epoch = None
        global_step = 0
        geometry_rows: list[dict[str, Any]] = []
        for epoch in range(1, TOTAL_EPOCHS + 1):
            scheduler.step(epoch)
            loader, plan_meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if any(meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"] for meta in gathered_meta):
                raise RuntimeError(f"runtime sampler audit failed for {arm} epoch {epoch}")
            if set().union(*(set(meta["covered_pids"]) for meta in gathered_meta)) != set(range(NUM_CLASSES)):
                raise RuntimeError(f"runtime full PID coverage failed for {arm} epoch {epoch}")
            if rank == 0 and sampler_writer is not None:
                for meta in gathered_meta:
                    sampler_writer.writerow({"seed": SEED, "arm": arm, **meta, "covered_pid_count": len(meta["covered_pids"])})
                sampler_stream.flush()
            local_sampled_images = {path_to_index[str(source["image_path"])] for source, _seed in loader.dataset.samples}
            local_sampled_states = {(int(source["pid"]), int(source["clothes_id"])) for source, _seed in loader.dataset.samples}
            local_sampled_pids = {int(source["pid"]) for source, _seed in loader.dataset.samples}
            local_stats = {"active_batches": 0, "pair_count": 0, "weights": [], "state_keys": set(), "exact_keys": set(), "pid_keys": set(), "clothing_keys": set()}
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation_raw", "relation_applied", "id_acc", "bio_acc")}
            local_skips = 0
            local_nonfinite = 0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            local_amp_events = []
            iterator = iter(loader)
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = b6.forward_batch(model, loss_fn, batch, device, bank, path_to_index, train_arm_name, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite loss for {arm} E{epoch} I{iteration}")
                scaler.scale(total).backward()
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                inf = base.found_inf(scaler, optimizer)
                local_skips += int(inf > 0)
                local_nonfinite += int(inf > 0)
                scaler.update()
                scale_after = float(scaler.get_scale())
                scale_min = min(scale_min, scale_after)
                scale_max = max(scale_max, scale_after)
                if inf > 0:
                    local_amp_events.append({
                        "epoch": epoch, "iteration": iteration, "global_step": global_step,
                        "skipped_step": 1, "found_inf": int(inf), "nonfinite_event": 1,
                        "amp_scale_before": scale_before, "amp_scale_after": scale_after,
                    })
                for key in epoch_sum:
                    epoch_sum[key] += scalar(result[key])
                info = result["relation_info"]
                exact_keys, state_keys, pid_keys, clothing_keys = relation_audit_keys(batch, bank, path_to_index)
                local_stats["exact_keys"].update(exact_keys)
                local_stats["state_keys"].update(state_keys)
                local_stats["pid_keys"].update(pid_keys)
                local_stats["clothing_keys"].update(clothing_keys)
                if info["active"]:
                    local_stats["active_batches"] += 1
                    local_stats["pair_count"] += int(info["pair_count"])
                    local_stats["weights"].extend(float(value) for value in info["weights"])
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", scalar(total), global_step)
            del iterator, loader
            gc.collect()
            reduced = b6.reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            count_tensor = torch.tensor([
                local_stats["active_batches"], local_stats["pair_count"],
                sum(local_stats["weights"]), sum(value * value for value in local_stats["weights"]),
            ], dtype=torch.float64, device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            min_weight = torch.tensor(min(local_stats["weights"]) if local_stats["weights"] else float("inf"), dtype=torch.float64, device=device)
            max_weight = torch.tensor(max(local_stats["weights"]) if local_stats["weights"] else -float("inf"), dtype=torch.float64, device=device)
            dist.all_reduce(min_weight, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_weight, op=dist.ReduceOp.MAX)
            gathered_sets = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_sets, (local_sampled_images, local_sampled_states, local_sampled_pids, local_stats["exact_keys"], local_stats["state_keys"], local_stats["pid_keys"], local_stats["clothing_keys"]))
            sampled_images = set().union(*(item[0] for item in gathered_sets))
            sampled_states = set().union(*(item[1] for item in gathered_sets))
            sampled_pids = set().union(*(item[2] for item in gathered_sets))
            exact_keys = set().union(*(set(item[3]) for item in gathered_sets))
            state_keys = set().union(*(set(item[4]) for item in gathered_sets))
            pid_keys = set().union(*(set(item[5]) for item in gathered_sets))
            clothing_keys = set().union(*(set(item[6]) for item in gathered_sets))
            relation_batches = int(count_tensor[0].item())
            relation_pairs = int(count_tensor[1].item())
            weight_sum = float(count_tensor[2].item())
            weight_sq_sum = float(count_tensor[3].item())
            weight_mean = weight_sum / relation_pairs if relation_pairs else 0.0
            weight_var = max(weight_sq_sum / relation_pairs - weight_mean * weight_mean, 0.0) if relation_pairs else 0.0
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); nonfinite_sum = amp_tensor[1].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM); dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_tensor = amp_tensor[2].clone(); max_tensor = amp_tensor[3].clone(); end_tensor = amp_tensor[4].clone()
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN); dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX); dist.all_reduce(end_tensor, op=dist.ReduceOp.MAX)
            effective_updates = int(round(STEPS_PER_EPOCH - float(skip_sum.item()) / WORLD_SIZE))
            amp_stats = {
                "skipped_steps": int(skip_sum.item()), "found_inf_count": int(skip_sum.item()),
                "nonfinite_events": int(nonfinite_sum.item()), "amp_scale_min": float(min_tensor.item()),
                "amp_scale_max": float(max_tensor.item()), "amp_scale_end": float(end_tensor.item()),
                "effective_updates_completed": effective_updates,
            }
            gathered_amp_events = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_amp_events, local_amp_events)
            if rank == 0 and event_writer is not None:
                for event_rank, event_rows in enumerate(gathered_amp_events):
                    for event in event_rows:
                        event_writer.writerow({"rank": event_rank, **event})
                event_stream.flush()
            dist.barrier()
            if rank == 0:
                metrics = base.evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            holder = [metrics]
            dist.broadcast_object_list(holder, src=0)
            metrics = holder[0]
            row = {
                "epoch": epoch,
                "train_loss": reduced["total"], "ce_loss": reduced["ce"], "triplet_loss": reduced["triplet"], "bio_loss": reduced["bio"],
                "relation_raw_loss": reduced["relation_raw"], "relation_applied_loss": reduced["relation_applied"],
                "relation_lambda": 0.0 if arm == "R0" else b6.LAMBDA_REL,
                "lr": float(scheduler._get_lr(epoch)[0]), "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"], **amp_stats,
                "optimizer_updates": STEPS_PER_EPOCH, "active_images": len(dataset.train), "active_pids": NUM_CLASSES,
                "active_clothes": NUM_CLOTHING_STATES, "mean_clothes_per_pid": float(np.mean([len(values) for values in active_state.values()])),
                "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids),
                "relation_active_batches": relation_batches, "relation_active_batch_rate": relation_batches / float(STEPS_PER_EPOCH * WORLD_SIZE),
                "relation_pairs": relation_pairs, "relation_unique_cross_clothes_pairs": len(exact_keys), "relation_unique_state_pairs": len(state_keys),
                "relation_pid_coverage": len({key[0] for key in pid_keys}), "relation_clothing_state_coverage": len(clothing_keys),
                "relation_weight_mean": weight_mean, "relation_weight_std": math.sqrt(weight_var),
                "relation_weight_min": 0.0 if not relation_pairs else float(min_weight.item()),
                "relation_weight_max": 0.0 if not relation_pairs else float(max_weight.item()),
                **metrics,
            }
            if rank == 0:
                metrics_writer.writerow(row); metrics_stream.flush()
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end", "effective_updates_completed"]}); amp_stream.flush()
                relation_writer.writerow({
                    "arm": arm, "seed": SEED, "epoch": epoch,
                    "all_images_available": True, "all_states_available": True, "all_pids_available": True,
                    "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids),
                    "relation_active_batches": relation_batches, "relation_active_batch_rate": row["relation_active_batch_rate"],
                    "relation_pairs": relation_pairs, "relation_unique_cross_clothes_pairs": len(exact_keys), "relation_unique_state_pairs": len(state_keys),
                    "relation_pid_coverage": len({key[0] for key in pid_keys}), "relation_clothing_state_coverage": len(clothing_keys),
                    "relation_weight_mean": weight_mean, "relation_weight_std": math.sqrt(weight_var),
                    "relation_weight_min": row["relation_weight_min"], "relation_weight_max": row["relation_weight_max"],
                    "optimizer_updates": STEPS_PER_EPOCH, "effective_updates_completed": effective_updates,
                }); relation_stream.flush()
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch); writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"]); best_epoch = epoch
                    b6.save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1 E1-E60"})
            if epoch in GEOMETRY_EPOCHS:
                if rank == 0:
                    rows = p6c.relation_geometry(model.module, cfg, dataset, bank, device, transform_test, epoch)
                    for geometry_row in rows:
                        geometry_row.update({"arm": arm, "seed": SEED, "setting": ARM_LABELS[arm], "geometry_source": "LTCC train-only frozen candidate bank"})
                    geometry_rows.extend(rows)
                    write_csv(run_dir / "relation_geometry.csv", geometry_rows)
                dist.barrier()
            if epoch == TOTAL_EPOCHS:
                save_rank_state(run_dir / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    b6.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                    write_json(run_dir / "checkpoint_epoch060.json", {"phase": PHASE, "arm": arm, "epoch": epoch, "global_step": global_step, "rank_states": [run_dir / f"rank{item}_state_epoch{epoch:03d}.pt" for item in range(WORLD_SIZE)], "model": run_dir / "model_final.pth"})
                    metrics_rows = read_csv(run_dir / "metrics_per_epoch.csv")
                    write_json(run_dir / "RUN_COMPLETE.json", {"phase": PHASE, "arm": arm, "mode": "FULL_SCHEDULE", "seed": SEED, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates_per_rank": global_step, "metrics": run_dir / "metrics_per_epoch.csv", "best_epoch": best_epoch, "best_cc_rank1": best_r1, "best_model": run_dir / "model_best.pth", "final_model": run_dir / "model_final.pth", "rows": len(metrics_rows), "common_init_sha256": init_audit["sha256"], "initial_parameter_hash": initial_hash})
        if rank == 0:
            write_json(run_dir / "AMP.json", {"seed": SEED, "initial_scale": base.INITIAL_SCALE, "epochs_completed": TOTAL_EPOCHS, "metrics_source": run_dir / "amp_statistics.csv", "events_source": run_dir / "amp_skip_events.csv"})
    finally:
        for stream in (metrics_stream, amp_stream, relation_stream, event_stream, sampler_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        b6.cleanup_dist()


def trajectory(rows: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"incomplete metric window E{start}-E{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    # Raw per-epoch metrics stay in [0, 1] in metrics_seed4321.csv. Window
    # summaries are reported in percentage units, so deltas are percentage
    # points and directly comparable to the frozen Phase6C reference.
    rank1 = np.asarray([float(row["cc_rank1"]) * 100.0 for row in selected], dtype=np.float64)
    mAP = np.asarray([float(row["cc_mAP"]) * 100.0 for row in selected], dtype=np.float64)
    auc = lambda values: float(np.trapezoid(values, x) / (x[-1] - x[0])) if hasattr(np, "trapezoid") else float(np.trapz(values, x) / (x[-1] - x[0]))
    best_rank1 = max(selected, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    best_mAP = max(selected, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {
        "start_epoch": start, "end_epoch": end, "epochs": len(selected),
        "mean_rank1": float(np.mean(rank1)), "mean_mAP": float(np.mean(mAP)),
        "auc_rank1": auc(rank1), "auc_mAP": auc(mAP),
        "slope_rank1": float(np.polyfit(x, rank1, 1)[0]), "slope_mAP": float(np.polyfit(x, mAP, 1)[0]),
        "first_rank1": float(rank1[0]), "first_mAP": float(mAP[0]),
        "last_rank1": float(rank1[-1]), "last_mAP": float(mAP[-1]),
        "best_rank1": float(best_rank1["cc_rank1"]) * 100.0, "best_rank1_epoch": int(best_rank1["epoch"]),
        "best_mAP": float(best_mAP["cc_mAP"]) * 100.0, "best_mAP_epoch": int(best_mAP["epoch"]),
    }


def window_name(start: int, end: int) -> str:
    return f"E{start}-E{end}"


def summarize_seed4321() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    summaries = []
    lookup = {}
    for arm, run_dir in RUN_DIRS.items():
        rows = read_csv(run_dir / "metrics_per_epoch.csv")
        if len(rows) != TOTAL_EPOCHS or [int(row["epoch"]) for row in rows] != list(range(1, TOTAL_EPOCHS + 1)):
            raise RuntimeError(f"{arm} does not contain exactly E1-E60")
        for start, end in WINDOWS:
            item = {"seed": SEED, "arm": arm, "setting": ARM_LABELS[arm], "window": window_name(start, end), **trajectory(rows, start, end)}
            summaries.append(item)
            lookup[(arm, item["window"])] = item
    write_csv(OUT_ROOT / "window_analysis_seed4321.csv", summaries)
    combined = []
    for item in summaries:
        combined.append(item)
    write_csv(OUT_ROOT / "metrics_seed4321.csv", [{"seed": SEED, "arm": arm, **row} for arm, run_dir in RUN_DIRS.items() for row in read_csv(run_dir / "metrics_per_epoch.csv")], ["seed", "arm"] + METRIC_FIELDS)
    return summaries, lookup


def old_seed_window_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for arm in ("B0", "B1"):
        rows = read_csv(PHASE6C_ROOT / f"{arm}_full" / "metrics_per_epoch.csv")
        for start, end in WINDOWS:
            result[(arm, window_name(start, end))] = trajectory(rows, start, end)
    return result


def build_two_seed_files(seed4321_lookup: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old = old_seed_window_lookup()
    comparison_rows = []
    summary_rows = []
    for seed, baseline, relation, lookup in [(1234, "B0", "B1", old), (4321, "R0", "R1", seed4321_lookup)]:
        for start, end in WINDOWS:
            window = window_name(start, end)
            left = lookup[(relation, window)]
            right = lookup[(baseline, window)]
            row = {
                "seed": seed, "window": window, "baseline_arm": baseline, "relation_arm": relation,
                "mean_rank1_delta_pp": float(left["mean_rank1"] - right["mean_rank1"]),
                "mean_mAP_delta_pp": float(left["mean_mAP"] - right["mean_mAP"]),
                "auc_rank1_delta_pp": float(left["auc_rank1"] - right["auc_rank1"]),
                "auc_mAP_delta_pp": float(left["auc_mAP"] - right["auc_mAP"]),
                "slope_rank1_delta_pp": float(left["slope_rank1"] - right["slope_rank1"]),
                "slope_mAP_delta_pp": float(left["slope_mAP"] - right["slope_mAP"]),
                "last_rank1_delta_pp": float(left["last_rank1"] - right["last_rank1"]),
                "last_mAP_delta_pp": float(left["last_mAP"] - right["last_mAP"]),
                "best_rank1_delta_pp": float(left["best_rank1"] - right["best_rank1"]),
                "best_mAP_delta_pp": float(left["best_mAP"] - right["best_mAP"]),
            }
            comparison_rows.append(row)
            summary_rows.append({"kind": "seed", **row})
    for start, end in WINDOWS:
        window = window_name(start, end)
        selected = [row for row in comparison_rows if row["window"] == window]
        for kind, function in [("mean", np.mean), ("min", np.min), ("max", np.max)]:
            summary_rows.append({
                "kind": f"two_seed_{kind}", "seed": "1234+4321", "window": window,
                "baseline_arm": "B0/R0", "relation_arm": "B1/R1",
                **{key: float(function([row[key] for row in selected])) for key in selected[0] if key.endswith("_pp")},
            })
    write_csv(OUT_ROOT / "seed1234_vs_seed4321.csv", comparison_rows)
    write_csv(OUT_ROOT / "two_seed_summary.csv", summary_rows)
    return comparison_rows, summary_rows


def amp_audit() -> list[dict[str, Any]]:
    rows = []
    for arm, run_dir in RUN_DIRS.items():
        metrics = read_csv(run_dir / "amp_statistics.csv")
        events = read_csv(run_dir / "amp_skip_events.csv")
        rows.append({
            "seed": SEED, "arm": arm, "setting": ARM_LABELS[arm], "rows": len(metrics),
            "total_skipped_steps": sum(int(row["skipped_steps"]) for row in metrics),
            "total_found_inf_count": sum(int(row["found_inf_count"]) for row in metrics),
            "total_nonfinite_events": sum(int(row["nonfinite_events"]) for row in metrics),
            "skip_epoch_list": ",".join(sorted({row["epoch"] for row in metrics if int(row["skipped_steps"]) > 0}, key=int)),
            "skip_iteration_list": ";".join(f"E{row['epoch']}:I{row['iteration']}:R{row['rank']}" for row in events),
            "amp_scale_initial": float(metrics[0]["amp_scale_min"]),
            "amp_scale_min": min(float(row["amp_scale_min"]) for row in metrics),
            "amp_scale_max": max(float(row["amp_scale_max"]) for row in metrics),
            "amp_scale_final": float(metrics[-1]["amp_scale_end"]),
            "event_rows": len(events),
        })
    write_csv(OUT_ROOT / "amp_audit_seed4321.csv", rows)
    return rows


def geometry_questions() -> tuple[str, dict[str, Any]]:
    rows = read_csv(OUT_ROOT / "R0_seed4321" / "relation_geometry.csv") + read_csv(OUT_ROOT / "R1_seed4321" / "relation_geometry.csv")
    write_csv(OUT_ROOT / "relation_geometry_seed4321.csv", rows)
    selected = {}
    for row in rows:
        if row["checkpoint"] == "E60" and row["sr3_quartile"] == "ALL":
            selected[row["arm"]] = {key: float(row[key]) for key in ["positive_cosine_mean", "positive_negative_margin_mean", "ordering_correct_rate"]}
    if set(selected) != {"R0", "R1"}:
        raise RuntimeError("missing E60 ALL geometry rows")
    deltas = {key: selected["R1"][key] - selected["R0"][key] for key in selected["R0"]}
    positive = sum(value > 0 for value in deltas.values())
    answer = "YES" if positive == 3 else ("NO" if positive == 0 else "MIXED")
    return answer, {"e60_all": selected, "delta_r1_minus_r0": deltas, "positive_dimensions": positive}


def finalize_common_audit() -> dict[str, Any]:
    manifests = {arm: json.loads((run_dir / "run_manifest.json").read_text()) for arm, run_dir in RUN_DIRS.items()}
    observed = {
        "R0_initial_parameter_hash": manifests["R0"]["initial_parameter_hash"],
        "R1_initial_parameter_hash": manifests["R1"]["initial_parameter_hash"],
        "initial_parameter_equality": manifests["R0"]["initial_parameter_hash"] == manifests["R1"]["initial_parameter_hash"],
        "common_init_equality": manifests["R0"]["common_init"]["sha256"] == manifests["R1"]["common_init"]["sha256"],
    }
    if not observed["initial_parameter_equality"] or not observed["common_init_equality"]:
        raise RuntimeError("R0/R1 strict common initialization equality failed")
    audit = json.loads(COMMON_AUDIT.read_text())
    audit.update({"observed_runtime": observed, "initial_parameter_equality": observed["initial_parameter_equality"], "finalized_utc": utc_now()})
    write_json(COMMON_AUDIT, audit)
    cfg = configure()
    sampler_rows = read_csv(OUT_ROOT / "sampler_audit_seed4321.csv")
    _state, common_audit = load_common_init()
    write_seed_audit(cfg, {"state": common_audit}, sampler_rows, observed)
    return observed


def matched_training_audit() -> dict[str, Any]:
    manifests = {arm: json.loads((run_dir / "run_manifest.json").read_text()) for arm, run_dir in RUN_DIRS.items()}
    shared_fields = [
        "seed", "batch_size", "num_instances", "epochs", "start_epoch", "end_epoch",
        "steps_per_epoch", "train_images", "train_pids", "train_clothing_states",
        "optimizer", "scheduler", "amp", "common_init", "initial_parameter_hash",
        "lambda_rel", "relation_source_sha256",
    ]
    shared_config_equal = all(manifests["R0"].get(key) == manifests["R1"].get(key) for key in shared_fields)
    r0_sampler = read_csv(RUN_DIRS["R0"] / "sampler_plan.csv")
    r1_sampler = read_csv(RUN_DIRS["R1"] / "sampler_plan.csv")
    if not r0_sampler or len(r0_sampler) != len(r1_sampler):
        raise RuntimeError("runtime sampler plan row count mismatch")
    sampler_fields = [key for key in r0_sampler[0] if key != "arm"]
    r0_normalized = [{key: row[key] for key in sampler_fields} for row in r0_sampler]
    r1_normalized = [{key: row[key] for key in sampler_fields} for row in r1_sampler]
    sampler_plan_equal = r0_normalized == r1_normalized
    if not shared_config_equal or not sampler_plan_equal:
        raise RuntimeError("R0/R1 matched-training audit failed")
    plan_digest = hashlib.sha256(json.dumps(r0_normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "shared_config_fields": shared_fields,
        "shared_config_exact_equal": shared_config_equal,
        "sampler_plan_rows_per_arm": len(r0_sampler),
        "sampler_plan_exact_equal_ignoring_arm_label": sampler_plan_equal,
        "sampler_plan_digest": plan_digest,
        "only_method_difference": "R1 adds the uniform cross-clothes relation term; R0 does not.",
    }


def analyze() -> dict[str, Any]:
    for arm, run_dir in RUN_DIRS.items():
        marker = run_dir / "RUN_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"{arm} is incomplete")
        complete = json.loads(marker.read_text())
        if complete.get("epochs_completed") != TOTAL_EPOCHS or complete.get("rows") != TOTAL_EPOCHS:
            raise RuntimeError(f"{arm} completion marker is not E1-E60")
    observed = finalize_common_audit()
    matched = matched_training_audit()
    summaries, lookup = summarize_seed4321()
    comparison_rows, two_seed_rows = build_two_seed_files(lookup)
    amp_rows = amp_audit()
    geometry_answer, geometry = geometry_questions()
    primary = next(row for row in comparison_rows if row["seed"] == SEED and row["window"] == PRIMARY_WINDOW)
    seed1234_primary = next(row for row in comparison_rows if row["seed"] == 1234 and row["window"] == PRIMARY_WINDOW)
    seed4321_rank1 = primary["auc_rank1_delta_pp"]
    seed4321_mAP = primary["auc_mAP_delta_pp"]
    q1 = "YES" if seed4321_rank1 > 0 and seed4321_mAP > 0 else ("MIXED" if seed4321_rank1 > 0 or seed4321_mAP > 0 else "NO")
    q2 = "YES" if seed4321_rank1 > 0 else "NO"
    q3 = "YES" if seed4321_mAP > 0 else "NO"
    amp_by_arm = {row["arm"]: row for row in amp_rows}
    skip_delta = amp_by_arm["R1"]["total_skipped_steps"] - amp_by_arm["R0"]["total_skipped_steps"]
    serious_amp = bool(skip_delta >= max(4, math.ceil(0.25 * max(1, amp_by_arm["R0"]["total_skipped_steps"]))) or amp_by_arm["R1"]["amp_scale_min"] <= amp_by_arm["R0"]["amp_scale_min"] / 2)
    q5 = "YES" if serious_amp else "NO"
    seed4321_pass = seed4321_rank1 > 0 and seed4321_mAP > 0 and not serious_amp
    if seed4321_pass and seed1234_primary["auc_rank1_delta_pp"] > 0 and seed1234_primary["auc_mAP_delta_pp"] > 0:
        q6 = "STRONG REPLICATION"
    elif seed4321_rank1 > 0 or seed4321_mAP > 0:
        q6 = "PARTIAL REPLICATION"
    else:
        q6 = "NOT REPLICATED"
    q7 = "PROMOTE" if seed4321_pass else "STOP"
    next_step = "PHASE 7A PRCC CROSS-DATASET VALIDATION — RECOMMENDED; DO NOT AUTO-RUN" if q7 == "PROMOTE" else "STOP; analyze seed sensitivity and do not enter Phase 7A"
    questions = {
        "Q1_seed4321_R1_gt_R0": q1,
        "Q2_seed4321_late_rank1_positive": q2,
        "Q3_seed4321_late_mAP_positive": q3,
        "Q4_relation_geometry_improves": geometry_answer,
        "Q5_serious_AMP_numerical_confound": q5,
        "Q6_two_seed_cross_clothes_replication": q6,
        "Q7_phase7a_prcc": q7,
    }
    manifest = {
        "phase": PHASE,
        "seed": SEED,
        "mode": "FULL_SCHEDULE_TWO_ARM_REPLICATION",
        "allowed_arms": ["R0", "R1"],
        "forbidden_arms_started": ["B2", "B3", "H_ID", "DS", "DW", "Relation Curriculum", "GRL", "Reverse", "online reliability", "new sampler", "PRCC", "diffusion"],
        "epochs": TOTAL_EPOCHS,
        "primary_window": PRIMARY_WINDOW,
        "windows": [window_name(*window) for window in WINDOWS],
        "run_complete": {arm: json.loads((run_dir / "RUN_COMPLETE.json").read_text()) for arm, run_dir in RUN_DIRS.items()},
        "common_init_audit": str(COMMON_AUDIT),
        "matched_training_audit": matched,
        "seed_audit": str(SEED_AUDIT),
        "frozen_uniform_relation_method": str(METHOD_DOC),
        "summaries": summaries,
        "primary_comparison_seed4321": primary,
        "seed1234_primary_reference": seed1234_primary,
        "two_seed_summary_rows": two_seed_rows,
        "amp_audit": amp_rows,
        "amp_serious_confound_rule": "R1 skip delta >= max(4,25% of R0) or R1 scaler floor <= half R0 floor",
        "amp_serious_confound": serious_amp,
        "geometry": geometry,
        "diagnostic_questions": questions,
        "next_step_recommendation": next_step,
        "two_seed_replication_not_statistical_validation": True,
        "second_seed_started": True,
        "phase7a_started": False,
        "phase6d_started": False,
        "source_checksums": source_checksums(),
        "artifacts": {
            "metrics_seed4321": str(OUT_ROOT / "metrics_seed4321.csv"),
            "window_analysis_seed4321": str(OUT_ROOT / "window_analysis_seed4321.csv"),
            "relation_geometry_seed4321": str(OUT_ROOT / "relation_geometry_seed4321.csv"),
            "amp_audit_seed4321": str(OUT_ROOT / "amp_audit_seed4321.csv"),
            "seed1234_vs_seed4321": str(OUT_ROOT / "seed1234_vs_seed4321.csv"),
            "two_seed_summary": str(OUT_ROOT / "two_seed_summary.csv"),
            "report": str(REPORT_PATH),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    write_report(manifest)
    return manifest


def write_report(manifest: dict[str, Any]) -> None:
    summaries = manifest["summaries"]
    primary = manifest["primary_comparison_seed4321"]
    reference = manifest["seed1234_primary_reference"]
    q = manifest["diagnostic_questions"]
    lines = [
        "# PHASE 6C-R — Cross-Clothes Relation Replication",
        "",
        "## Scope and freeze",
        "",
        "This phase runs only R0 Bio baseline and R1 Uniform Cross-Clothes Relation under independent seed4321. B2, B3, SR3, H_ID, DS, DW, curricula, GRL, Reverse, online reliability, PRCC, and diffusion were not started.",
        "",
        f"- Full schedule: E1–E60; primary window: `{PRIMARY_WINDOW}`.",
        f"- Seed4321 common initialization exact equality: `{manifest['common_init_audit']}`.",
        "- Two-seed replication is evidence, not a statistical significance test.",
        "",
        "## Matched-training audit",
        "",
        f"- Shared runtime config exact equality: `{manifest['matched_training_audit']['shared_config_exact_equal']}`.",
        f"- Sampler plan exact equality (R0/R1, arm label ignored): `{manifest['matched_training_audit']['sampler_plan_exact_equal_ignoring_arm_label']}` across `{manifest['matched_training_audit']['sampler_plan_rows_per_arm']}` rows per arm.",
        f"- Runtime sampler-plan digest: `{manifest['matched_training_audit']['sampler_plan_digest']}`.",
        "- The only intended method difference is the R1 uniform cross-clothes relation term.",
        "",
        "## Seed4321 trajectory",
        "",
        "All retrieval metrics are percentages; deltas and slopes are percentage points (pp).",
        "",
        "| Arm | Window | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Slope R1 | Slope mAP | E60 R1 | E60 mAP | BEST R1 (epoch) | BEST mAP (epoch) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['window']} | {row['mean_rank1']:.4f} | {row['mean_mAP']:.4f} | {row['auc_rank1']:.4f} | {row['auc_mAP']:.4f} | {row['slope_rank1']:.4f} | {row['slope_mAP']:.4f} | {row['last_rank1']:.4f} | {row['last_mAP']:.4f} | {row['best_rank1']:.4f} (E{row['best_rank1_epoch']}) | {row['best_mAP']:.4f} (E{row['best_mAP_epoch']}) |")
    lines += [
        "",
        "## Primary comparison: R1 − R0, E51–E60",
        "",
        "| Seed | Mean R1/mAP Δ pp | AUC R1/mAP Δ pp | Slope R1/mAP Δ pp | E60 R1/mAP Δ pp | BEST R1/mAP Δ pp |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 1234 frozen reference | {reference['mean_rank1_delta_pp']:.4f} / {reference['mean_mAP_delta_pp']:.4f} | {reference['auc_rank1_delta_pp']:.4f} / {reference['auc_mAP_delta_pp']:.4f} | {reference['slope_rank1_delta_pp']:.4f} / {reference['slope_mAP_delta_pp']:.4f} | {reference['last_rank1_delta_pp']:.4f} / {reference['last_mAP_delta_pp']:.4f} | {reference['best_rank1_delta_pp']:.4f} / {reference['best_mAP_delta_pp']:.4f} |",
        f"| 4321 replication | {primary['mean_rank1_delta_pp']:.4f} / {primary['mean_mAP_delta_pp']:.4f} | {primary['auc_rank1_delta_pp']:.4f} / {primary['auc_mAP_delta_pp']:.4f} | {primary['slope_rank1_delta_pp']:.4f} / {primary['slope_mAP_delta_pp']:.4f} | {primary['last_rank1_delta_pp']:.4f} / {primary['last_mAP_delta_pp']:.4f} | {primary['best_rank1_delta_pp']:.4f} / {primary['best_mAP_delta_pp']:.4f} |",
        "",
        "## Two-seed window summary: R1 − R0",
        "",
        "Mean, minimum, and maximum are across the two seeds; this is a two-seed replication summary, not statistical validation.",
        "",
        "| Window | Mean Δ R1/mAP pp | Min Δ R1/mAP pp | Max Δ R1/mAP pp |",
        "|---|---:|---:|---:|",
    ]
    summary_rows = manifest["two_seed_summary_rows"]
    for window in [window_name(*item) for item in WINDOWS]:
        by_kind = {row["kind"]: row for row in summary_rows if row["window"] == window and row["kind"].startswith("two_seed_")}
        lines.append(
            f"| {window} | {by_kind['two_seed_mean']['mean_rank1_delta_pp']:.4f} / {by_kind['two_seed_mean']['mean_mAP_delta_pp']:.4f} | "
            f"{by_kind['two_seed_min']['mean_rank1_delta_pp']:.4f} / {by_kind['two_seed_min']['mean_mAP_delta_pp']:.4f} | "
            f"{by_kind['two_seed_max']['mean_rank1_delta_pp']:.4f} / {by_kind['two_seed_max']['mean_mAP_delta_pp']:.4f} |"
        )
    lines += [
        "",
        "## Relation geometry",
        "",
        "Geometry uses only the LTCC train split and the frozen candidate bank. Query/gallery are used only for retrieval evaluation.",
        "",
        f"- Q4 geometry result: **{q['Q4_relation_geometry_improves']}**.",
        f"- E60 ALL R0: `{json.dumps(manifest['geometry']['e60_all']['R0'], sort_keys=True)}`.",
        f"- E60 ALL R1: `{json.dumps(manifest['geometry']['e60_all']['R1'], sort_keys=True)}`.",
        f"- R1 − R0 geometry delta: `{json.dumps(manifest['geometry']['delta_r1_minus_r0'], sort_keys=True)}`.",
        "",
        "## AMP audit",
        "",
        "| Arm | Total skips | found_inf | nonfinite | Skip epochs | Scale min | Scale max | Final scale | Event rows |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in manifest["amp_audit"]:
        lines.append(f"| {row['arm']} | {row['total_skipped_steps']} | {row['total_found_inf_count']} | {row['total_nonfinite_events']} | {row['skip_epoch_list']} | {row['amp_scale_min']:.1f} | {row['amp_scale_max']:.1f} | {row['amp_scale_final']:.1f} | {row['event_rows']} |")
    lines += [
        "",
        f"Serious AMP numerical confound: **{manifest['amp_serious_confound']}**. Rule: `{manifest['amp_serious_confound_rule']}`.",
        "",
        "## Q1–Q7",
        "",
        f"- Q1. seed4321 R1 是否优于 R0？ **{q['Q1_seed4321_R1_gt_R0']}**.",
        f"- Q2. E51–E60 Rank-1 是否复制正向收益？ **{q['Q2_seed4321_late_rank1_positive']}**.",
        f"- Q3. E51–E60 mAP 是否复制正向收益？ **{q['Q3_seed4321_late_mAP_positive']}**.",
        f"- Q4. Relation geometry 是否改善？ **{q['Q4_relation_geometry_improves']}**.",
        f"- Q5. 是否存在明显 AMP numerical confound？ **{q['Q5_serious_AMP_numerical_confound']}**.",
        f"- Q6. seed1234 + seed4321 benefit：**{q['Q6_two_seed_cross_clothes_replication']}**.",
        f"- Q7. 是否进入 Phase 7A PRCC？ **{q['Q7_phase7a_prcc']}**.",
        "",
        "## Final decision",
        "",
        f"**{manifest['next_step_recommendation']}**",
        "",
        "No PRCC or Phase 7A run was started automatically. No SR3 research was reopened. Two seeds do not constitute full statistical validation.",
        "",
        "## Artifacts",
        "",
        f"- `{COMMON_AUDIT}`",
        f"- `{SEED_AUDIT}`",
        f"- `{METHOD_DOC}`",
        f"- `{OUT_ROOT / 'metrics_seed4321.csv'}`",
        f"- `{OUT_ROOT / 'window_analysis_seed4321.csv'}`",
        f"- `{OUT_ROOT / 'relation_geometry_seed4321.csv'}`",
        f"- `{OUT_ROOT / 'amp_audit_seed4321.csv'}`",
        f"- `{OUT_ROOT / 'seed1234_vs_seed4321.csv'}`",
        f"- `{OUT_ROOT / 'two_seed_summary.csv'}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train", "analyze"), required=True)
    parser.add_argument("--arm", choices=("R0", "R1"), default=None)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for train")
        train_arm(args.arm)
    else:
        analyze()
        print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
