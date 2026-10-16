#!/usr/local/bin/python3
"""Phase 7A: PRCC cross-dataset validation of uniform cross-clothes relation.

This driver has an explicit audit -> prepare -> train -> analyze flow.  The
audit and feasibility modes never construct a training model or start a
distributed training process.  The two permitted training arms are:

P0: CE + Triplet + Bio
P1: CE + Triplet + Bio + 0.1 * frozen uniform relation

The PRCC data loader, sampler, evaluation protocol, and PRCC formal schedule
are taken from the repository's existing paper-faithful PRCC path.  The
relation loss itself is called from the executed Phase 6B implementation;
this file does not reimplement its numerical formula.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import inspect
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataloader, build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from model.build import factory  # noqa: E402
from processor.processor import test as formal_test  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval  # noqa: E402

import phase3c_prcc_formal as phase3c  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as phase6b  # noqa: E402


PHASE = "phase7a_prcc_relation_validation"
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
PRIMARY_WINDOW = "E51-E60"
WINDOWS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60))
GEOMETRY_EPOCHS = (20, 40, 60)

OUT_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
ARTIFACT_ROOT = OUT_ROOT / "artifacts"
REPORT_PATH = ROOT / "reports/PHASE_7A_PRCC_CROSS_DATASET_RELATION_VALIDATION.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7A_MANIFEST.json"
COMMON_INIT = OUT_ROOT / "PRCC_COMMON_INIT.pth"
COMMON_INIT_META = OUT_ROOT / "PRCC_COMMON_INIT_HASHES.json"
COMMON_INIT_CONFIG = OUT_ROOT / "prcc_common_init_config.yml"
RELATION_BANK = ARTIFACT_ROOT / "prcc_relation_bank.npz"
AUDIT_PAYLOAD = ARTIFACT_ROOT / "prcc_audit_payload.json"
FEASIBILITY_PAYLOAD = ARTIFACT_ROOT / "prcc_feasibility_payload.json"
CONFIG_AUDIT = ARTIFACT_ROOT / "prcc_config_audit.md"
FEASIBILITY_REPORT = ARTIFACT_ROOT / "prcc_cross_clothes_relation_feasibility.md"
METHOD_DOC = ARTIFACT_ROOT / "frozen_relation_method.md"
COMMON_AUDIT = ARTIFACT_ROOT / "prcc_common_init_audit.json"
STRUCTURE_CSV = ARTIFACT_ROOT / "ltcc_vs_prcc_relation_structure.csv"

PRCC_ROOT = Path("/home/data/prcc")
TEXT_ROOT = Path("/home/data/DIFFER_TextCaption/PRCC")
TRAIN_NPZ = TEXT_ROOT / "EVA02-CLIP-L-14/train.npz"
VAL_NPZ = TEXT_ROOT / "EVA02-CLIP-L-14/val.npz"
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
RELEASED_CONFIG = ROOT / "configs/prcc/eva02_l_bio.yml"
HISTORICAL_BASELINE = ROOT / "outputs/reproduction_phase3/PRCC/baseline/run1"
HISTORICAL_BIO = ROOT / "outputs/reproduction_phase3/PRCC/full_paper_faithful/run1"
OFFICIAL_ROOT = Path("/home/data/DIFFER_OfficialWeights/provenance/PRCC")
OFFICIAL_CONFIG = OFFICIAL_ROOT / "config.yml"
OFFICIAL_CHECKPOINT = OFFICIAL_ROOT / "eva02_l_bio_best.pth"
OFFICIAL_LOG = OFFICIAL_ROOT / "train_log.txt"
EVA_WEIGHT = phase3c.EVA_WEIGHT

RUN_DIRS = {
    "P0": OUT_ROOT / "P0_prcc_bio",
    "P1": OUT_ROOT / "P1_prcc_uniform_relation",
}
ARM_LABELS = {
    "P0": "PRCC Bio baseline",
    "P1": "PRCC Bio + uniform cross-clothes relation",
}
TRAIN_ARM = {"P0": "B0", "P1": "B1"}

METRIC_FIELDS = [
    "arm", "seed", "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss",
    "relation_loss", "relation_applied_loss", "relation_lambda", "total_loss",
    "lr", "lr_bias", "id_accuracy", "bio_accuracy", "planned_updates",
    "updates_completed", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "amp_skip", "skipped_steps", "found_inf_count", "nonfinite_events",
    "active_images", "active_pids", "active_clothing_states", "sampled_unique_images",
    "sampled_unique_pids", "sampled_unique_clothing_states", "relation_active_batches",
    "relation_active_batch_rate", "relations_per_epoch", "unique_pair_count",
    "relation_pid_coverage", "relation_clothing_state_coverage", "cc_rank1",
    "cc_rank5", "cc_rank10", "cc_mAP", "general_rank1", "general_rank5",
    "general_rank10", "general_mAP",
]
EXPOSURE_FIELDS = [
    "arm", "seed", "epoch", "all_images_available", "all_pids_available",
    "all_clothing_states_available", "sampled_unique_images", "sampled_unique_pids",
    "sampled_unique_clothing_states", "relation_active_batches",
    "relation_active_batch_rate", "relations_per_epoch", "unique_pair_count",
    "relation_pid_coverage", "relation_clothing_state_coverage", "relation_weight_mean",
    "relation_weight_std", "relation_weight_min", "relation_weight_max",
    "planned_updates", "updates_completed",
]
AMP_FIELDS = [
    "arm", "seed", "epoch", "amp_skip", "skipped_steps", "found_inf_count",
    "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "planned_updates", "updates_completed",
]
AMP_EVENT_FIELDS = [
    "arm", "seed", "rank", "epoch", "iteration", "global_step",
    "skipped_step", "found_inf", "nonfinite_event", "amp_scale_before",
    "amp_scale_after",
]
SAMPLER_FIELDS = [
    "arm", "seed", "epoch", "rank", "sampler_epoch_used", "samples", "batches",
    "unique_images", "unique_pids", "unique_clothing_states", "plan_sha256",
]
GEOMETRY_FIELDS = [
    "arm", "seed", "checkpoint", "relation_count", "pid_count",
    "clothing_state_count", "positive_cosine_mean", "positive_cosine_median",
    "negative_center_cosine_mean", "positive_negative_margin_mean",
    "positive_negative_margin_median", "ordering_correct_rate", "geometry_source",
]


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
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
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
    result = {}
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        result[key] = sha256_bytes(raw)
    return result


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(json.dumps(tensor_hashes(state), sort_keys=True).encode("utf-8"))


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    return {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure(run_dir: Path = OUT_ROOT) -> Any:
    """Return the audited historical PRCC formal config with Bio-only losses."""
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(RELEASED_CONFIG))
    cfg.defrost()
    cfg.MODEL.TYPE = "eva02_cloth"
    cfg.MODEL.NAME = "eva02_l_bio"
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.MODEL.DIST_TRAIN = True
    cfg.MODEL.NONBIO_HEAD = False
    cfg.MODEL.LAST_LAYER = "clipFc"
    cfg.MODEL.CLIP_DIM = 768
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-L-14"
    cfg.DATA.BIO_INDEX = "0"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.DATASET = "prcc"
    cfg.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.DATA.NUM_INSTANCES = NUM_INSTANCES
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.IMG_HEIGHT = 224
    cfg.DATA.IMG_WIDTH = 224
    cfg.DATA.SAMPLER = "softmax_triplet"
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.OPTIMIZER_NAME = "SGD"
    cfg.SOLVER.BASE_LR = 2e-6
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.05
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_EPOCHS = FORMAL_EPOCHS
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.SOLVER.WARMUP_LR = 7.8125e-7
    cfg.SOLVER.CHECKPOINT_PERIOD = FORMAL_EPOCHS
    cfg.SOLVER.EVAL_PERIOD = 1
    cfg.OUTPUT_DIR = str(run_dir)
    cfg.OUTPUT_NAME = "PHASE7A_PRCC_RELATION"
    cfg.TEST.WEIGHT = ""
    cfg.TEST.FEAT_NORM = "yes"
    cfg.TEST.TYPE = "image_only"
    cfg.TRAIN.START_EPOCH = 1
    cfg.freeze()
    return cfg


def optimizer_config(cfg: Any) -> dict[str, Any]:
    return {
        "name": str(cfg.SOLVER.OPTIMIZER_NAME),
        "base_lr": float(cfg.SOLVER.BASE_LR),
        "bias_lr_factor": float(cfg.SOLVER.BIAS_LR_FACTOR),
        "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
        "weight_decay_bias": float(cfg.SOLVER.WEIGHT_DECAY_BIAS),
        "momentum": float(cfg.SOLVER.MOMENTUM),
        "large_fc_lr": bool(cfg.SOLVER.LARGE_FC_LR),
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
        "enabled": True,
        "autocast_dtype": "FP16",
        "loss_location": "outside autocast",
        "grad_scaler": "torch.cuda.amp.GradScaler",
        "initial_scale": INITIAL_SCALE,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "settings_modified_for_replication": False,
    }


def raw_prcc_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    train_root = PRCC_ROOT / "rgb/train"
    for pid_dir in sorted(train_root.iterdir(), key=lambda item: int(item.name)):
        if not pid_dir.is_dir():
            continue
        for path in sorted(pid_dir.glob("*.jpg")):
            camera = path.name[0]
            if camera not in {"A", "B", "C"}:
                raise RuntimeError(f"unexpected PRCC train camera prefix: {path}")
            semantic_state = pid_dir.name if camera in {"A", "B"} else f"{pid_dir.name}{camera}"
            records.append({
                "pid_raw": pid_dir.name,
                "path": str(path),
                "camera": camera,
                "semantic_state": semantic_state,
            })
    return records


def build_relation_bank(dataset: Any) -> dict[str, Any]:
    """Build the full train-only PRCC ground-truth relation bank.

    The Phase 6B relation function is reused unchanged.  Its exact lookup is
    populated with every valid PRCC train pair because PRCC has no SR3 score
    bank and Phase 7A forbids reliability-based candidate selection.  Every
    bank entry has uniform weight one.
    """
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    path_to_index = {}
    for index, item in enumerate(dataset.train):
        path = str(item["image_path"])
        path_to_index[path] = index
        by_pid_cloth[int(item["pid"])][int(item["clothes_id"])].append(index)

    left: list[int] = []
    right: list[int] = []
    pids: list[int] = []
    cloth_a: list[int] = []
    cloth_b: list[int] = []
    per_pid: dict[int, int] = {}
    for pid in sorted(by_pid_cloth):
        clothes = sorted(by_pid_cloth[pid])
        count = 0
        for first_position, first_cloth in enumerate(clothes):
            for second_cloth in clothes[first_position + 1:]:
                for first_index in by_pid_cloth[pid][first_cloth]:
                    for second_index in by_pid_cloth[pid][second_cloth]:
                        a, b = sorted((int(first_index), int(second_index)))
                        left.append(a)
                        right.append(b)
                        pids.append(int(pid))
                        cloth_a.append(int(first_cloth))
                        cloth_b.append(int(second_cloth))
                        count += 1
        per_pid[pid] = count

    arrays = {
        "index_a": np.asarray(left, dtype=np.int32),
        "index_b": np.asarray(right, dtype=np.int32),
        "pid": np.asarray(pids, dtype=np.int16),
        "clothes_a": np.asarray(cloth_a, dtype=np.int16),
        "clothes_b": np.asarray(cloth_b, dtype=np.int16),
    }
    exact_lookup = {(int(a), int(b)): 1.0 for a, b in zip(arrays["index_a"], arrays["index_b"])}
    if len(exact_lookup) != len(left):
        raise RuntimeError("PRCC relation bank contains duplicate exact image pairs")
    arrays["path_order_sha256"] = np.asarray(sha256_bytes("\n".join(path_to_index).encode("utf-8")))
    return {
        "arrays": arrays,
        "exact_lookup": exact_lookup,
        "exact_weight_b2": exact_lookup,
        "exact_weight_b3": exact_lookup,
        "path_to_index": path_to_index,
        "by_pid_cloth": by_pid_cloth,
        "per_pid": per_pid,
        "relation_count": len(left),
        "exact_pair_count": len(exact_lookup),
        "pid_count": len(by_pid_cloth),
        "clothing_state_count": len({(pid, cloth) for pid, values in by_pid_cloth.items() for cloth in values}),
    }


def load_relation_bank(dataset: Any) -> dict[str, Any]:
    if not RELATION_BANK.is_file():
        raise FileNotFoundError(RELATION_BANK)
    with np.load(RELATION_BANK, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"index_a", "index_b", "pid", "clothes_a", "clothes_b"}
    if not required.issubset(arrays):
        raise RuntimeError(f"incomplete PRCC relation bank: {sorted(arrays)}")
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    expected_path_hash = sha256_bytes("\n".join(path_to_index).encode("utf-8"))
    stored_path_hash = str(arrays.get("path_order_sha256", np.asarray("" )).item())
    if stored_path_hash != expected_path_hash:
        raise RuntimeError("PRCC relation bank image order differs from current dataset loader order")
    exact_lookup = {
        (int(a), int(b)): 1.0
        for a, b in zip(arrays["index_a"].tolist(), arrays["index_b"].tolist())
    }
    if any(int(a) >= int(b) for a, b in exact_lookup):
        raise RuntimeError("relation bank is not in canonical unordered form")
    return {
        "arrays": arrays,
        "exact_lookup": exact_lookup,
        "exact_weight_b2": exact_lookup,
        "exact_weight_b3": exact_lookup,
        "path_to_index": path_to_index,
        "relation_count": len(exact_lookup),
        "exact_pair_count": len(exact_lookup),
        "pid_count": len(set(arrays["pid"].tolist())),
        "clothing_state_count": len({(int(pid), int(cloth)) for pid, cloth in zip(arrays["pid"].tolist(), arrays["clothes_a"].tolist())} | {(int(pid), int(cloth)) for pid, cloth in zip(arrays["pid"].tolist(), arrays["clothes_b"].tolist())}),
    }


def relation_structure_from_dataset(dataset: Any, bank: dict[str, Any]) -> dict[str, Any]:
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(dataset.train):
        by_pid_cloth[int(item["pid"])][int(item["clothes_id"])].append(index)
    all_same_pid_pairs = sum(
        len(indices) * (len(indices) - 1) // 2
        for values in by_pid_cloth.values() for indices in values.values()
    )
    all_cross_pairs = sum(bank["per_pid"].values()) if "per_pid" in bank else int(bank["relation_count"])
    cloth_counts = [len(values) for values in by_pid_cloth.values()]
    return {
        "train_images": len(dataset.train),
        "pids": len(by_pid_cloth),
        "clothing_states": len({(pid, cloth) for pid, values in by_pid_cloth.items() for cloth in values}),
        "pids_with_ge2_clothes": sum(count >= 2 for count in cloth_counts),
        "mean_clothes_per_pid": float(np.mean(cloth_counts)),
        "median_clothes_per_pid": float(np.median(cloth_counts)),
        "min_clothes_per_pid": int(min(cloth_counts)),
        "max_clothes_per_pid": int(max(cloth_counts)),
        "all_same_pid_image_pairs": int(all_same_pid_pairs + all_cross_pairs),
        "valid_cross_clothes_unordered_pairs": int(all_cross_pairs),
        "relation_bank_pairs": int(bank["exact_pair_count"]),
        "relation_bank_density": float(bank["exact_pair_count"] / max(1, all_cross_pairs)),
        "per_pid_relation_min": int(min(bank["per_pid"].values())) if "per_pid" in bank else None,
        "per_pid_relation_median": float(np.median(list(bank["per_pid"].values()))) if "per_pid" in bank else None,
        "per_pid_relation_max": int(max(bank["per_pid"].values())) if "per_pid" in bank else None,
    }


def history_result(path: Path, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "path": str(path), "exists": path.is_dir()}
    metrics_path = path / "metrics_per_epoch.csv"
    eval_path = path / "independent_evaluation.json"
    amp_path = path / "AMP_statistics.json"
    config_path = path / "config.yml"
    result.update({
        "config": str(config_path),
        "config_exists": config_path.is_file(),
        "checkpoint": str(path / "eva02_l_bio_best.pth"),
        "checkpoint_exists": (path / "eva02_l_bio_best.pth").is_file(),
        "checkpoint_sha256": sha256_file(path / "eva02_l_bio_best.pth") if (path / "eva02_l_bio_best.pth").is_file() else None,
    })
    if metrics_path.is_file():
        rows = read_csv(metrics_path)
        result["metrics_rows"] = len(rows)
        result["metrics_last"] = rows[-1] if rows else None
        if rows:
            best = max(rows, key=lambda row: (float(row.get("cc_rank1", "nan")), float(row.get("cc_mAP", "nan"))))
            result["metrics_best"] = {"epoch": int(best["epoch"]), "rank1": float(best["cc_rank1"]), "mAP": float(best["cc_mAP"])}
    if eval_path.is_file():
        result["independent_evaluation"] = json.loads(eval_path.read_text())
    if amp_path.is_file():
        result["amp"] = json.loads(amp_path.read_text())
    return result


def audit_dataset_and_config() -> tuple[dict[str, Any], dict[str, Any]]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = configure()
    raw_records = raw_prcc_records()
    if not PRCC_ROOT.is_dir():
        raise RuntimeError(f"PRCC root missing: {PRCC_ROOT}")
    if not raw_records:
        raise RuntimeError("PRCC train directory has no images")

    set_seed(SEED)
    dataset = build_dataset(cfg)
    # The dataset constructor reads query/gallery to expose the formal
    # protocol, but all relation counts below use only dataset.train.
    bank = build_relation_bank(dataset)
    np.savez_compressed(RELATION_BANK, **bank["arrays"])

    dataset_paths = [str(item["image_path"]) for item in dataset.train]
    raw_by_path = {row["path"]: row for row in raw_records}
    runtime_label_checks = []
    pid_state_map: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in dataset.train:
        path = str(item["image_path"])
        raw = raw_by_path[path]
        pid = int(item["pid"])
        cloth = int(item["clothes_id"])
        pid_state_map[pid][cloth].append(raw["camera"])
        expected_group = "AB" if raw["camera"] in {"A", "B"} else "C"
        runtime_label_checks.append({
            "pid": pid,
            "camera": raw["camera"],
            "clothes_id": cloth,
            "expected_group": expected_group,
        })
    schema_ok = True
    per_pid_schema = {}
    for pid, state_values in sorted(pid_state_map.items()):
        camera_groups = {cloth: sorted(cameras) for cloth, cameras in state_values.items()}
        schema_ok = schema_ok and len(camera_groups) == 2
        schema_ok = schema_ok and all(set(cameras).issubset({"A", "B"}) or set(cameras) == {"C"} for cameras in camera_groups.values())
        schema_ok = schema_ok and any(set(cameras) & {"A", "B"} for cameras in camera_groups.values())
        schema_ok = schema_ok and any(set(cameras) == {"C"} for cameras in camera_groups.values())
        per_pid_schema[pid] = camera_groups

    text_audit = phase3c.text_runtime_audit(dataset, cfg)
    relation_structure = relation_structure_from_dataset(dataset, bank)
    feasibility_checks = {
        "dataset_root_exists": PRCC_ROOT.is_dir(),
        "train_images_expected": len(dataset.train) == EXPECTED_TRAIN_IMAGES,
        "train_pids_expected": int(dataset.num_train_pids) == EXPECTED_TRAIN_PIDS,
        "train_clothing_states_expected": int(dataset.num_train_clothes) == EXPECTED_CLOTHING_STATES,
        "clothing_labels_present": all("clothes_id" in item for item in dataset.train),
        "label_semantics_safe": schema_ok,
        "all_pids_have_at_least_two_clothes": relation_structure["pids_with_ge2_clothes"] == EXPECTED_TRAIN_PIDS,
        "valid_relations_positive": relation_structure["valid_cross_clothes_unordered_pairs"] > 0,
        "relation_bank_matches_valid_pairs": relation_structure["relation_bank_pairs"] == relation_structure["valid_cross_clothes_unordered_pairs"],
        "train_text_keys_match": bool(text_audit["dataset_text_key_set_match_train"]),
        "clip_dim_768": bool(text_audit["clip_dim_assert"]),
        "runtime_caption_dim_768": int(text_audit["runtime_caption_dim"]) == 768,
    }
    passed = all(feasibility_checks.values())
    if not passed:
        raise RuntimeError(f"PRCC relation feasibility gate failed: {feasibility_checks}")

    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "scope": "Audit only; no P0/P1 training started",
        "config_snapshot": cfg.dump(),
        "config_sha256": sha256_bytes(cfg.dump().encode("utf-8")),
        "dataset": {
            "root": str(PRCC_ROOT),
            "train_images": len(dataset.train),
            "train_pids": int(dataset.num_train_pids),
            "train_clothing_states": int(dataset.num_train_clothes),
            "train_cameras": int(dataset.num_camera),
            "val_images": 5002,
            "query_same_B": int(dataset.num_query_imgs_same),
            "query_changed_C": int(dataset.num_query_imgs_diff),
            "gallery_A": int(dataset.num_gallery_imgs),
            "primary_evaluation": "PRCC changed clothes C -> A",
            "secondary_evaluation": "PRCC same clothes B -> A",
            "dataset_train_path_order_sha256": sha256_bytes("\n".join(dataset_paths).encode("utf-8")),
        },
        "clothing_label_schema": {
            "camera_A_B": "same semantic clothing state per PID",
            "camera_C": "different semantic clothing state per PID",
            "runtime_source": "data/datasets/prcc.py actual train parser",
            "per_pid_schema": per_pid_schema,
            "schema_check": schema_ok,
        },
        "text": text_audit,
        "training_config": {
            "image_size": [224, 224],
            "batch_size_per_ppu": BATCH_SIZE,
            "num_instances": NUM_INSTANCES,
            "epochs": FORMAL_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "planned_updates_per_rank_epoch": EXPECTED_STEPS_PER_EPOCH,
            "optimizer": optimizer_config(cfg),
            "scheduler": scheduler_config(cfg),
            "amp": amp_config(),
            "loss_p0": ["ce", "triplet", "clipBio"],
            "loss_p1": ["ce", "triplet", "clipBio", "frozen uniform relation"],
            "bio_index": [0],
            "nonbio_index": [1, 2],
            "clip_dim": 768,
            "reverse": "OFF",
            "grl": "OFF",
            "dw": "OFF",
            "ds": "OFF",
            "h_id": "OFF",
            "reliability": "OFF",
            "new_sampler": "OFF",
        },
        "historical": {
            "historical_prcc_baseline": history_result(HISTORICAL_BASELINE, "historical PRCC ID-only baseline"),
            "historical_prcc_bio": history_result(HISTORICAL_BIO, "historical PRCC Bio+Reverse formal run"),
            "official_checkpoint": {
                "root": str(OFFICIAL_ROOT),
                "config": str(OFFICIAL_CONFIG),
                "config_exists": OFFICIAL_CONFIG.is_file(),
                "checkpoint": str(OFFICIAL_CHECKPOINT),
                "checkpoint_exists": OFFICIAL_CHECKPOINT.is_file(),
                "checkpoint_sha256": sha256_file(OFFICIAL_CHECKPOINT) if OFFICIAL_CHECKPOINT.is_file() else None,
                "train_log": str(OFFICIAL_LOG),
                "train_log_exists": OFFICIAL_LOG.is_file(),
            },
        },
        "relation_structure": relation_structure,
        "relation_bank": {
            "path": str(RELATION_BANK),
            "sha256": sha256_file(RELATION_BANK),
            "pairs": bank["exact_pair_count"],
            "weights": "uniform w_ij=1.0",
            "candidate_policy": "full PRCC train-only ground-truth bank; no SR3/reliability candidate selection",
        },
        "feasibility_checks": feasibility_checks,
        "feasibility_pass": passed,
        "forbidden_arms_started": [],
    }
    write_json(AUDIT_PAYLOAD, audit)
    feasibility = {
        "phase": PHASE,
        "created_utc": audit["created_utc"],
        "passed": passed,
        "checks": feasibility_checks,
        "dataset": audit["dataset"],
        "clothing_label_schema": audit["clothing_label_schema"],
        "relation_structure": relation_structure,
        "relation_bank": audit["relation_bank"],
        "per_pid_relation_counts": {str(k): int(v) for k, v in sorted(bank["per_pid"].items())},
        "train_only": True,
        "query_gallery_used_for_relation": False,
        "training_started": False,
    }
    write_json(FEASIBILITY_PAYLOAD, feasibility)

    config_lines = [
        "# Phase 7A PRCC configuration audit",
        "",
        f"Generated: `{audit['created_utc']}`. This is the pre-training audit; P0/P1 training had not started when it was written.",
        "",
        "## Effective formal PRCC setup",
        "",
        f"- Dataset root: `{PRCC_ROOT}`.",
        f"- Train/query/gallery: `{len(dataset.train)}` / changed-clothes C `{dataset.num_query_imgs_diff}` / same-clothes B `{dataset.num_query_imgs_same}` / gallery A `{dataset.num_gallery_imgs}`.",
        f"- Train IDs/clothing states/cameras: `{dataset.num_train_pids}` / `{dataset.num_train_clothes}` / `{dataset.num_camera}`.",
        "- Image size: `224 × 224`; formal evaluation is Euclidean pairwise distance with normalized features, no re-ranking.",
        "- Primary evaluation: changed-clothes `C → A`; secondary evaluation: same-clothes `B → A`.",
        "- The formal PRCC path uses the repository's `DistributedRandomIdentitySampler`, batch 16, NUM_INSTANCES 4, and 551 updates/rank/epoch.",
        "",
        "## Model and text audit",
        "",
        "- Model: `eva02_l_bio`, type `eva02_cloth`, last layer `clipFc`.",
        "- `CLIP_DIM=768`; runtime caption feature dimension is 768.",
        "- Bio factor: index 0; formal non-Bio indices: `[1,2]` (Hair + Clothing) as used by the historical paper-faithful PRCC path.",
        f"- Bio text source: Summary Bio `{SUMMARY_JSON}`; image-level NPZ source: `{TRAIN_NPZ}` and `{VAL_NPZ}`.",
        f"- Stable caption metadata alignment and runtime key match: `{text_audit['dataset_text_key_set_match_train']}`; Summary Bio runtime match: `{text_audit['summary_bio_runtime_match_tolerance_1e-3']}`.",
        "",
        "## Optimizer, scheduler, and AMP",
        "",
        f"- Optimizer: `{json.dumps(optimizer_config(cfg), sort_keys=True)}`.",
        f"- Scheduler: `{json.dumps(scheduler_config(cfg), sort_keys=True)}`.",
        "- Schedule length is 60 epochs with 20-epoch linear warmup; this is not changed to the LTCC 596-update schedule.",
        f"- AMP: `{json.dumps(amp_config(), sort_keys=True)}`; the PRCC historical initial GradScaler scale is 512 and is not retuned.",
        "",
        "## Historical and official provenance",
        "",
        f"- Historical ID-only PRCC run: `{HISTORICAL_BASELINE}`; its result is provenance only, not the matched P0 control.",
        f"- Historical PRCC Bio+Reverse run/checkpoint: `{HISTORICAL_BIO}`; its checkpoint is not used to initialize P0 or P1.",
        f"- Official packaged PRCC artifact: `{OFFICIAL_ROOT}`; it is not used to initialize either arm.",
        "- P0/P1 will instead strict-load the same freshly created PRCC common initialization derived from the audited pretrained EVA source.",
        "",
        "## Explicit differences from LTCC",
        "",
        "- PRCC uses CLIP_DIM 768, 150 train PIDs, 300 train clothing labels, batch 16, NUM_INSTANCES 4, and 551 planned updates/rank/epoch.",
        "- No LTCC value is copied into PRCC without this audit; in particular, LTCC CLIP_DIM 512 and batch 8 are not used.",
        "",
        f"Effective config SHA256: `{audit['config_sha256']}`.",
        "",
    ]
    CONFIG_AUDIT.write_text("\n".join(config_lines), encoding="utf-8")

    relation_lines = [
        "# Phase 7A PRCC cross-clothes relation feasibility",
        "",
        f"Generated: `{audit['created_utc']}`.",
        "",
        f"Decision: **{'PASS — training may proceed' if passed else 'STOP — do not train'}**.",
        "",
        "This gate uses only `rgb/train`. Query/gallery images are not used to construct relation labels, relation pairs, or any training sampler.",
        "",
        "## Train-only clothing label audit",
        "",
        "- The repository parser maps PRCC train camera A and B to one semantic clothing state per PID, and camera C to the second state.",
        f"- Train images: `{len(dataset.train)}`; train PIDs: `{dataset.num_train_pids}`; global clothing-state labels: `{dataset.num_train_clothes}`.",
        f"- PIDs with at least two clothing states: `{relation_structure['pids_with_ge2_clothes']}/{relation_structure['pids']}`.",
        f"- Label semantics safe: `{schema_ok}`.",
        "",
        "## Relation bank",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Valid same-PID/different-clothes unordered image pairs | {relation_structure['valid_cross_clothes_unordered_pairs']} |",
        f"| Relation-bank pairs | {relation_structure['relation_bank_pairs']} |",
        f"| Relation-bank density over valid pairs | {relation_structure['relation_bank_density']:.6f} |",
        f"| PIDs represented | {relation_structure['pids']} |",
        f"| Relations/PID minimum | {relation_structure['per_pid_relation_min']} |",
        f"| Relations/PID median | {relation_structure['per_pid_relation_median']:.1f} |",
        f"| Relations/PID maximum | {relation_structure['per_pid_relation_max']} |",
        "",
        "The Phase 6B relation function is reused unchanged. Since Phase 7A forbids SR3/reliability candidate selection and PRCC has no corresponding frozen SR3 bank, the PRCC exact lookup contains all valid train-only ground-truth pairs and assigns `w_ij=1.0` to every pair.",
        "",
        f"Machine-readable gate: `{FEASIBILITY_PAYLOAD}`.",
        f"Relation bank: `{RELATION_BANK}` (SHA256 `{sha256_file(RELATION_BANK)}`).",
        "",
    ]
    FEASIBILITY_REPORT.write_text("\n".join(relation_lines), encoding="utf-8")
    return audit, feasibility


def write_frozen_method_doc() -> None:
    if phase6b.LAMBDA_REL != 0.1:
        raise RuntimeError(f"executed Phase6B lambda changed: {phase6b.LAMBDA_REL}")
    source_runner = Path(phase6b.__file__)
    phase6cr_runner = ROOT / "scripts/phase6cr_relation_replication.py"
    phase6cr_doc = ROOT / "outputs/new_method_track/phase6cr_relation_replication/artifacts/frozen_uniform_relation_method.md"
    relation_source = inspect.getsource(phase6b.relation_loss)
    forward_source = inspect.getsource(phase6b.forward_batch)
    lines = [
        "# Frozen relation method for Phase 7A",
        "",
        "Executed code is authoritative. Phase 7A calls the Phase 6B `forward_batch` and `relation_loss` functions directly; the numerical relation loss is not reimplemented here.",
        "",
        f"- Phase 6B executed runner: `{source_runner}`; SHA256 `{sha256_file(source_runner)}`.",
        f"- Phase 6C-R runner: `{phase6cr_runner}`; SHA256 `{sha256_file(phase6cr_runner)}`.",
        f"- Phase 6C-R frozen method document: `{phase6cr_doc}`; SHA256 `{sha256_file(phase6cr_doc) if phase6cr_doc.is_file() else 'MISSING'}`.",
        "",
        "## Frozen numerical behavior",
        "",
        "- Feature source: normalized student identity feature `feat[0]`.",
        "- Positive truth: same PID and different clothing state.",
        "- Pair construction and exact image-path lookup are those of the executed Phase 6B function.",
        "- Pair loss: `1 - cosine(normalized_feat_i, normalized_feat_j)`.",
        "- Empty candidate set: differentiable zero returned by the executed function.",
        f"- `lambda_rel = {phase6b.LAMBDA_REL}`.",
        "- P1 weight: `w_ij = 1.0` for every PRCC ground-truth relation.",
        "- P0 does not apply the relation term; P1 applies `base_loss + 0.1 * relation_loss`.",
        "- PRCC uses a full train-only ground-truth exact lookup because no SR3/reliability candidate selection is allowed in Phase 7A. This is a bank-coverage adaptation, not a score, weighting, sampler, or loss-formula change.",
        "",
        "## Explicitly frozen off",
        "",
        "Reverse, GRL, DW, DS, H_ID, online reliability, SR3 weighting, relation curriculum, new sampler, diffusion augmentation, and query/gallery relation construction are all OFF.",
        "",
        "## Executed source: relation_loss",
        "",
        "```python",
        relation_source.rstrip(),
        "```",
        "",
        "## Executed source: forward_batch",
        "",
        "```python",
        forward_source.rstrip(),
        "```",
        "",
    ]
    METHOD_DOC.write_text("\n".join(lines), encoding="utf-8")


def audit() -> None:
    audit_data, feasibility = audit_dataset_and_config()
    write_frozen_method_doc()
    print(json.dumps({
        "audit": str(CONFIG_AUDIT),
        "feasibility": str(FEASIBILITY_REPORT),
        "feasibility_pass": feasibility["passed"],
        "relation_pairs": audit_data["relation_structure"]["valid_cross_clothes_unordered_pairs"],
        "training_started": False,
    }, indent=2, sort_keys=True))


def load_audit() -> dict[str, Any]:
    if not AUDIT_PAYLOAD.is_file() or not FEASIBILITY_PAYLOAD.is_file():
        raise RuntimeError("run --mode audit first")
    audit_data = json.loads(AUDIT_PAYLOAD.read_text())
    feasibility = json.loads(FEASIBILITY_PAYLOAD.read_text())
    if not feasibility.get("passed"):
        raise RuntimeError("PRCC feasibility did not pass; training is forbidden")
    if not feasibility.get("train_only") or feasibility.get("query_gallery_used_for_relation"):
        raise RuntimeError("relation feasibility audit violates train-only rule")
    return audit_data


def make_model_from_state(cfg: Any, state: dict[str, torch.Tensor], num_classes: int, num_cameras: int) -> torch.nn.Module:
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=num_classes, camera_num=num_cameras)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict common-init load failed: {incompatible}")
    return model


def generate_common_init(cfg: Any, audit_data: dict[str, Any]) -> dict[str, Any]:
    if COMMON_INIT.is_file() and COMMON_INIT_META.is_file():
        meta = json.loads(COMMON_INIT_META.read_text())
        if sha256_file(COMMON_INIT) != meta.get("state_sha256"):
            raise RuntimeError("existing PRCC common init hash mismatch")
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        if state_hash(state) != meta.get("state_hash"):
            raise RuntimeError("existing PRCC common init tensor hash mismatch")
        return meta
    if COMMON_INIT.exists() or COMMON_INIT_META.exists():
        raise RuntimeError("incomplete PRCC common init exists; refusing to overwrite it")

    set_seed(SEED)
    # The common initialization is built from the same audited pretrained EVA
    # source used by the historical formal PRCC path, with fresh PRCC-sized
    # Bio-only classifier/projection heads.  No historical checkpoint enters.
    raw = build_model(cfg, EXPECTED_TRAIN_PIDS, 3)
    state = cpu_state_dict(raw)
    if any("reverse" in key.lower() or "grl" in key.lower() for key in state):
        raise RuntimeError("PRCC common init contains forbidden Reverse/GRL parameters")
    state_sha = state_hash(state)
    COMMON_INIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = COMMON_INIT.with_suffix(".pth.tmp")
    torch.save(state, temporary)
    temporary.replace(COMMON_INIT)

    # Load the same file into independently constructed P0/P1 topology twice;
    # this is a pre-training exact equality check, not a model-only shortcut.
    hashes = {}
    strict_results = {}
    for arm in ("P0", "P1"):
        probe = make_model_from_state(cfg, state, EXPECTED_TRAIN_PIDS, 3)
        hashes[arm] = state_hash(cpu_state_dict(probe))
        strict_results[arm] = {"missing": [], "unexpected": [], "strict": True}
        del probe
        gc.collect()
    if hashes["P0"] != hashes["P1"] or hashes["P0"] != state_sha:
        raise RuntimeError(f"PRCC common init exact equality failed: {hashes}")

    COMMON_INIT_CONFIG.write_text(cfg.dump(), encoding="utf-8")
    meta = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "path": str(COMMON_INIT),
        "state_sha256": sha256_file(COMMON_INIT),
        "state_hash": state_sha,
        "tensor_hashes": tensor_hashes(state),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "model_parameter_hashes": hashes,
        "initial_parameter_equality": hashes["P0"] == hashes["P1"],
        "strict_load_results": strict_results,
        "model": {
            "name": str(cfg.MODEL.NAME),
            "type": str(cfg.MODEL.TYPE),
            "num_classes": EXPECTED_TRAIN_PIDS,
            "num_cameras": 3,
            "active_losses": list(cfg.MODEL.LOSS_TYPE),
            "reverse": "OFF",
            "grl": "OFF",
        },
        "optimizer_config": optimizer_config(cfg),
        "scheduler_config": scheduler_config(cfg),
        "amp_config": amp_config(),
        "config_sha256": sha256_bytes(cfg.dump().encode("utf-8")),
        "pretrained_source": {
            "path": str(EVA_WEIGHT),
            "exists": EVA_WEIGHT.is_file(),
            "sha256": sha256_file(EVA_WEIGHT) if EVA_WEIGHT.is_file() else None,
            "historical_checkpoint_used": False,
        },
        "feasibility_payload_sha256": sha256_file(FEASIBILITY_PAYLOAD),
        "forbidden_arms_started": [],
    }
    write_json(COMMON_INIT_META, meta)
    return meta


def prepare() -> None:
    audit_data = load_audit()
    cfg = configure()
    common = generate_common_init(cfg, audit_data)
    dataset = build_dataset(cfg)
    bank = load_relation_bank(dataset)
    if len(dataset.train) != EXPECTED_TRAIN_IMAGES or dataset.num_train_pids != EXPECTED_TRAIN_PIDS or dataset.num_train_clothes != EXPECTED_CLOTHING_STATES:
        raise RuntimeError("PRCC prepare topology changed after feasibility audit")
    # Audit the actual released sampler without initializing distributed state.
    sampler_rows = []
    for arm in RUN_DIRS:
        for epoch in range(1, FORMAL_EPOCHS + 1):
            for rank in range(WORLD_SIZE):
                sampler = DistributedRandomIdentitySampler(
                    dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE,
                    rank=rank, seed=SEED,
                )
                # The existing PRCC formal processor does not call set_epoch;
                # retain its actual default sampler epoch for strict protocol fidelity.
                indices = list(iter(sampler))
                batches = len(indices) // BATCH_SIZE
                unique_pids = {int(dataset.train[index]["pid"]) for index in indices}
                unique_states = {(int(dataset.train[index]["pid"]), int(dataset.train[index]["clothes_id"])) for index in indices}
                digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest()
                sampler_rows.append({
                    "arm": arm, "seed": SEED, "epoch": epoch, "rank": rank,
                    "sampler_epoch_used": sampler.epoch, "samples": len(indices),
                    "batches": batches, "unique_images": len(set(indices)),
                    "unique_pids": len(unique_pids), "unique_clothing_states": len(unique_states),
                    "plan_sha256": digest,
                })
                if batches != EXPECTED_STEPS_PER_EPOCH:
                    raise RuntimeError(f"PRCC sampler expected {EXPECTED_STEPS_PER_EPOCH} batches, got {batches}")
    write_csv(OUT_ROOT / "sampler_audit.csv", sampler_rows, SAMPLER_FIELDS)
    per_arm_plan = {arm: [row["plan_sha256"] for row in sampler_rows if row["arm"] == arm] for arm in RUN_DIRS}
    if per_arm_plan["P0"] != per_arm_plan["P1"]:
        raise RuntimeError("P0/P1 pre-training sampler plan differs")
    common_audit = {
        "phase": PHASE,
        "seed": SEED,
        "created_utc": utc_now(),
        "common_init": common,
        "model_state_sha256": common["state_sha256"],
        "initial_parameter_equality": common["initial_parameter_equality"],
        "optimizer_config": optimizer_config(cfg),
        "scheduler_config": scheduler_config(cfg),
        "amp_config": amp_config(),
        "configuration_hash": common["config_sha256"],
        "sampler": {
            "class": "DistributedRandomIdentitySampler",
            "batch_size": BATCH_SIZE,
            "num_instances": NUM_INSTANCES,
            "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
            "p0_p1_plan_exact_equal": True,
            "plan_digest": sha256_bytes(json.dumps(per_arm_plan["P0"], sort_keys=True).encode("utf-8")),
            "actual_formal_loop_set_epoch": False,
        },
        "relation_bank": {
            "path": str(RELATION_BANK),
            "sha256": sha256_file(RELATION_BANK),
            "pairs": bank["exact_pair_count"],
            "weights": "uniform w_ij=1.0",
        },
        "historical_reference_not_loaded": True,
        "forbidden_arms_started": [],
        "training_started": False,
    }
    write_json(COMMON_AUDIT, common_audit)
    write_json(OUT_ROOT / "PREPARE.json", {
        "phase": PHASE,
        "prepared_utc": utc_now(),
        "seed": SEED,
        "feasibility_pass": True,
        "common_init": str(COMMON_INIT),
        "common_init_audit": str(COMMON_AUDIT),
        "sampler_audit": str(OUT_ROOT / "sampler_audit.csv"),
        "method_doc": str(METHOD_DOC),
        "allowed_arms": ["P0", "P1"],
        "forbidden_arms": ["B2", "B3", "SR3", "H_ID", "DS", "DW", "GRL", "Reverse", "Relation Curriculum", "new sampler", "diffusion"],
    })
    del dataset, bank
    gc.collect()
    print(json.dumps({
        "prepared": True,
        "feasibility_pass": True,
        "common_init": str(COMMON_INIT),
        "common_init_sha256": common["state_sha256"],
        "initial_parameter_equality": common["initial_parameter_equality"],
        "training_started": False,
    }, indent=2, sort_keys=True))


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


class CaptureWriter:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag: str, value: Any, step: Any = None, *args, **kwargs) -> None:
        self.values.setdefault(str(tag), []).append(float(value))

    def reset_records(self) -> None:
        self.values = {}


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


def relation_keys_from_batch(batch: dict[str, Any], bank: dict[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]], set[int], set[tuple[int, int]]]:
    pids = batch["pid"].detach().cpu().tolist()
    clothes = batch["clothes_id"].detach().cpu().tolist()
    paths = [str(path) for path in batch["image_path"]]
    indices = [bank["path_to_index"][path] for path in paths]
    exact: set[tuple[int, int]] = set()
    state: set[tuple[int, int, int]] = set()
    pids_seen: set[int] = set()
    cloth_seen: set[tuple[int, int]] = set()
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if int(pids[i]) != int(pids[j]) or int(clothes[i]) == int(clothes[j]):
                continue
            key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
            if key not in bank["exact_lookup"]:
                continue
            exact.add(key)
            state.add((int(pids[i]), min(int(clothes[i]), int(clothes[j])), max(int(clothes[i]), int(clothes[j]))))
            pids_seen.add(int(pids[i]))
            cloth_seen.add((int(pids[i]), int(clothes[i])))
            cloth_seen.add((int(pids[j]), int(clothes[j])))
    return exact, state, pids_seen, cloth_seen


def evaluate_prcc(cfg: Any, raw_model: torch.nn.Module, dataset: Any, val_loader: Any, val_loader_same: Any, device: torch.device, epoch: int, logger: logging.Logger) -> dict[str, float]:
    raw_model.eval()
    diff_eval = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    same_eval = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    diff_eval.reset()
    same_eval.reset()
    diff_writer = CaptureWriter()
    same_writer = CaptureWriter()
    with torch.no_grad():
        formal_test(cfg, raw_model, diff_eval, val_loader, logger, str(device), epoch, diff_writer, metric_prefix="cc")
        formal_test(cfg, raw_model, same_eval, val_loader_same, logger, str(device), epoch, same_writer, test=True, metric_prefix="general")
    raw_model.train()
    def last(writer: CaptureWriter, tag: str) -> float:
        values = writer.values.get(tag, [])
        return float(values[-1]) if values else float("nan")
    return {
        "cc_rank1": last(diff_writer, "cc/rank1"),
        "cc_rank5": last(diff_writer, "cc/rank5"),
        "cc_rank10": last(diff_writer, "cc/rank10"),
        "cc_mAP": last(diff_writer, "cc/mAP"),
        "general_rank1": last(same_writer, "general/rank1"),
        "general_rank5": last(same_writer, "general/rank5"),
        "general_rank10": last(same_writer, "general/rank10"),
        "general_mAP": last(same_writer, "general/mAP"),
    }


def model_identity_hash(model: torch.nn.Module) -> str:
    return state_hash(cpu_state_dict(model))


def build_runtime(cfg: Any, state: dict[str, torch.Tensor], device: torch.device, dataset: Any):
    raw = make_model_from_state(cfg, state, int(dataset.num_train_pids), int(dataset.num_camera))
    raw.to(device)
    raw.train()
    model = DDP(raw, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _center_optimizer = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    return model, loss_fn, optimizer, scheduler, scaler


def save_rank_state(path: Path, model: DDP, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE,
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


def train_arm(arm: str) -> None:
    if arm not in RUN_DIRS:
        raise ValueError(f"only P0/P1 are permitted, got {arm}")
    audit_data = load_audit()
    prepare_path = OUT_ROOT / "PREPARE.json"
    if not prepare_path.is_file():
        raise RuntimeError("run --mode prepare before training")
    if not COMMON_INIT.is_file() or not COMMON_INIT_META.is_file():
        raise RuntimeError("PRCC common initialization is missing")
    run_dir = RUN_DIRS[arm]
    if (run_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"refusing to overwrite completed arm: {run_dir}")
    rank, device = phase6b.init_dist()
    streams: list[Any] = []
    logger = None
    try:
        cfg = configure(run_dir)
        set_seed(SEED)
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        common_meta = json.loads(COMMON_INIT_META.read_text())
        if sha256_file(COMMON_INIT) != common_meta["state_sha256"] or state_hash(state) != common_meta["state_hash"]:
            raise RuntimeError("common initialization integrity check failed")
        dataset = build_dataset(cfg)
        if len(dataset.train) != EXPECTED_TRAIN_IMAGES or dataset.num_train_pids != EXPECTED_TRAIN_PIDS or dataset.num_train_clothes != EXPECTED_CLOTHING_STATES:
            raise RuntimeError("PRCC runtime topology differs from audited topology")
        bank = load_relation_bank(dataset)
        train_loader, _same_loader, _diff_loader, _gallery_loader, _dataset_again, train_sampler, val_loader, val_loader_same = build_dataloader(cfg)
        if len(train_loader) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(f"expected {EXPECTED_STEPS_PER_EPOCH} PRCC batches/rank/epoch, got {len(train_loader)}")
        if _dataset_again is not dataset:
            # build_dataloader creates its own dataset object.  It must have
            # exactly the same audited order and labels.
            if [str(item["image_path"]) for item in _dataset_again.train] != [str(item["image_path"]) for item in dataset.train]:
                raise RuntimeError("build_dataloader PRCC dataset order differs from relation-bank order")
            dataset = _dataset_again
            bank = load_relation_bank(dataset)

        model, loss_fn, optimizer, scheduler, scaler = build_runtime(cfg, state, device, dataset)
        local_initial_hash = model_identity_hash(model)
        gathered_initial = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_initial, local_initial_hash)
        if len(set(gathered_initial)) != 1:
            raise RuntimeError(f"{arm} initial model differs across ranks: {gathered_initial}")
        if local_initial_hash != common_meta["state_hash"]:
            raise RuntimeError("runtime model hash differs from common init")

        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            logger = logger_for(run_dir / "train_log_rank0.txt", arm)
            write_json(run_dir / "run_manifest.json", {
                "phase": PHASE,
                "arm": arm,
                "label": ARM_LABELS[arm],
                "seed": SEED,
                "epochs": FORMAL_EPOCHS,
                "start_epoch": 1,
                "end_epoch": FORMAL_EPOCHS,
                "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
                "train_images": len(dataset.train),
                "train_pids": int(dataset.num_train_pids),
                "train_clothing_states": int(dataset.num_train_clothes),
                "train_cameras": int(dataset.num_camera),
                "batch_size": BATCH_SIZE,
                "num_instances": NUM_INSTANCES,
                "optimizer": optimizer_config(cfg),
                "scheduler": scheduler_config(cfg),
                "amp": amp_config(),
                "common_init": {"path": str(COMMON_INIT), "sha256": common_meta["state_sha256"], "state_hash": common_meta["state_hash"]},
                "initial_parameter_hash": local_initial_hash,
                "initial_parameter_strict_load": True,
                "reverse": "OFF",
                "grl": "OFF",
                "dw": "OFF",
                "ds": "OFF",
                "h_id": "OFF",
                "online_reliability": "OFF",
                "active_set_withholding": "OFF",
                "relation_loss": "executed Phase6B positive-only same-PID/different-clothes cosine loss",
                "relation_weight": "uniform w_ij=1.0" if arm == "P1" else "not applied in P0",
                "lambda_rel": 0.0 if arm == "P0" else phase6b.LAMBDA_REL,
                "relation_bank": str(RELATION_BANK),
                "relation_bank_sha256": sha256_file(RELATION_BANK),
                "relation_bank_pairs": bank["exact_pair_count"],
                "relation_candidate_policy": "full PRCC train-only ground-truth bank; no SR3/reliability selection",
                "evaluation": {
                    "primary": "changed clothes C -> A",
                    "secondary": "same clothes B -> A",
                    "query_count": int(dataset.num_query_imgs_diff),
                    "gallery_count": int(dataset.num_gallery_imgs),
                    "feat_norm": str(cfg.TEST.FEAT_NORM),
                    "reranking": False,
                    "metric": "Euclidean pairwise distance + CMC/mAP",
                },
                "only_method_difference": "P1 applies the uniform relation term; P0 does not.",
                "forbidden_arms_started": [],
            })
            metrics_file = (run_dir / "metrics_per_epoch.csv").open("w", newline="", encoding="utf-8")
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
            metrics_writer.writeheader(); metrics_file.flush(); streams.append(metrics_file)
            exposure_file = (run_dir / "relation_exposure.csv").open("w", newline="", encoding="utf-8")
            exposure_writer = csv.DictWriter(exposure_file, fieldnames=EXPOSURE_FIELDS)
            exposure_writer.writeheader(); exposure_file.flush(); streams.append(exposure_file)
            amp_file = (run_dir / "amp_audit.csv").open("w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_file, fieldnames=AMP_FIELDS)
            amp_writer.writeheader(); amp_file.flush(); streams.append(amp_file)
            amp_event_file = (run_dir / "amp_skip_events.csv").open("w", newline="", encoding="utf-8")
            amp_event_writer = csv.DictWriter(amp_event_file, fieldnames=AMP_EVENT_FIELDS)
            amp_event_writer.writeheader(); amp_event_file.flush(); streams.append(amp_event_file)
            sampler_file = (run_dir / "sampler_plan.csv").open("w", newline="", encoding="utf-8")
            sampler_writer = csv.DictWriter(sampler_file, fieldnames=SAMPLER_FIELDS)
            sampler_writer.writeheader(); sampler_file.flush(); streams.append(sampler_file)
        else:
            metrics_writer = exposure_writer = amp_writer = amp_event_writer = sampler_writer = None

        # The standard PRCC sampler's formal loop leaves sampler.epoch at its
        # default zero.  Audit the same plan for every epoch without changing
        # the live sampler state.
        audit_sampler = DistributedRandomIdentitySampler(
            dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE,
            rank=rank, seed=SEED,
        )
        indices = list(iter(audit_sampler))
        plan_digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest()
        if len(indices) // BATCH_SIZE != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError("runtime sampler audit batch count mismatch")

        best_rank1 = -float("inf")
        best_epoch = None
        global_step = 0
        geometry_rows: list[dict[str, Any]] = []
        train_transform, test_transform = build_img_transforms(cfg)
        path_to_index = bank["path_to_index"]
        for epoch in range(1, FORMAL_EPOCHS + 1):
            scheduler.step(epoch)
            sampler_meta = {
                "arm": arm, "seed": SEED, "epoch": epoch, "rank": rank,
                "sampler_epoch_used": int(train_sampler.epoch), "samples": len(indices),
                "batches": len(indices) // BATCH_SIZE, "unique_images": len(set(indices)),
                "unique_pids": len({int(dataset.train[index]["pid"]) for index in indices}),
                "unique_clothing_states": len({(int(dataset.train[index]["pid"]), int(dataset.train[index]["clothes_id"])) for index in indices}),
                "plan_sha256": plan_digest,
            }
            gathered_sampler = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_sampler, sampler_meta)
            if rank == 0 and sampler_writer is not None:
                for row in gathered_sampler:
                    sampler_writer.writerow(row)
                sampler_file.flush()

            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation", "relation_applied", "id_acc", "bio_acc")}
            local_active_batches = 0
            local_relation_pairs = 0
            local_weights: list[float] = []
            local_exact_keys: set[tuple[int, int]] = set()
            local_state_keys: set[tuple[int, int, int]] = set()
            local_pid_keys: set[int] = set()
            local_cloth_keys: set[tuple[int, int]] = set()
            local_sampled_images: set[int] = set()
            local_sampled_pids: set[int] = set()
            local_sampled_states: set[tuple[int, int]] = set()
            local_skips = 0
            local_found_inf = 0
            local_nonfinite = 0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            local_amp_events: list[dict[str, Any]] = []
            loader = train_loader
            for iteration, batch in enumerate(loader, start=1):
                global_step += 1
                optimizer.zero_grad()
                result = phase6b.forward_batch(
                    model, loss_fn, batch, device, bank, path_to_index,
                    TRAIN_ARM[arm], global_step,
                )
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite {arm} loss at E{epoch} I{iteration}")
                scaler.scale(total).backward()
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                found_inf = int(base.found_inf(scaler, optimizer) > 0)
                scaler.update()
                scale_after = float(scaler.get_scale())
                scale_min = min(scale_min, scale_after)
                scale_max = max(scale_max, scale_after)
                local_skips += found_inf
                local_found_inf += found_inf
                local_nonfinite += found_inf
                if found_inf:
                    local_amp_events.append({
                        "rank": rank,
                        "epoch": epoch,
                        "iteration": iteration,
                        "global_step": global_step,
                        "skipped_step": 1,
                        "found_inf": found_inf,
                        "nonfinite_event": 1,
                        "amp_scale_before": scale_before,
                        "amp_scale_after": scale_after,
                    })
                for key in epoch_sum:
                    source_key = {"relation": "relation_raw", "relation_applied": "relation_applied"}.get(key, key)
                    epoch_sum[key] += scalar(result[source_key])
                exact_keys, state_keys, pid_keys, cloth_keys = relation_keys_from_batch(batch, bank)
                local_exact_keys.update(exact_keys)
                local_state_keys.update(state_keys)
                local_pid_keys.update(pid_keys)
                local_cloth_keys.update(cloth_keys)
                info = result["relation_info"]
                if info["active"]:
                    local_active_batches += 1
                    local_relation_pairs += int(info["pair_count"])
                    local_weights.extend(float(value) for value in info["weights"])
                for path, pid, cloth in zip(batch["image_path"], batch["pid"].detach().cpu().tolist(), batch["clothes_id"].detach().cpu().tolist()):
                    local_sampled_images.add(path_to_index[str(path)])
                    local_sampled_pids.add(int(pid))
                    local_sampled_states.add((int(pid), int(cloth)))
            if hasattr(loader, "shutdown"):
                loader.shutdown()
            if iteration != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"runtime loader yielded {iteration} instead of {EXPECTED_STEPS_PER_EPOCH}")

            gathered_amp_events = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_amp_events, local_amp_events)
            if rank == 0 and amp_event_writer is not None:
                for rank_events in gathered_amp_events:
                    for event in rank_events:
                        amp_event_writer.writerow({"arm": arm, "seed": SEED, **event})
                amp_event_file.flush()

            # Reduce scalar losses and AMP counters over the two PPU ranks.
            loss_keys = list(epoch_sum)
            loss_tensor = torch.tensor([epoch_sum[key] / EXPECTED_STEPS_PER_EPOCH for key in loss_keys], dtype=torch.float64, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= WORLD_SIZE
            reduced = {key: float(loss_tensor[index].detach().cpu()) for index, key in enumerate(loss_keys)}
            amp_tensor = torch.tensor([local_skips, local_found_inf, local_nonfinite, scale_min, scale_max, scale_after], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); found_sum = amp_tensor[1].clone(); nonfinite_sum = amp_tensor[2].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM); dist.all_reduce(found_sum, op=dist.ReduceOp.SUM); dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            amp_min = amp_tensor[3].clone(); amp_max = amp_tensor[4].clone(); amp_end = amp_tensor[5].clone()
            dist.all_reduce(amp_min, op=dist.ReduceOp.MIN); dist.all_reduce(amp_max, op=dist.ReduceOp.MAX); dist.all_reduce(amp_end, op=dist.ReduceOp.MAX)
            updates_completed = EXPECTED_STEPS_PER_EPOCH - float(skip_sum.item()) / WORLD_SIZE

            relation_count_tensor = torch.tensor([local_active_batches, local_relation_pairs, sum(local_weights), sum(value * value for value in local_weights)], dtype=torch.float64, device=device)
            dist.all_reduce(relation_count_tensor, op=dist.ReduceOp.SUM)
            relation_batches = int(relation_count_tensor[0].item())
            relation_pairs = int(relation_count_tensor[1].item())
            weight_sum = float(relation_count_tensor[2].item())
            weight_sq_sum = float(relation_count_tensor[3].item())
            weight_mean = weight_sum / relation_pairs if relation_pairs else 0.0
            weight_var = max(weight_sq_sum / relation_pairs - weight_mean * weight_mean, 0.0) if relation_pairs else 0.0
            local_min_weight = min(local_weights) if local_weights else float("inf")
            local_max_weight = max(local_weights) if local_weights else -float("inf")
            min_weight = torch.tensor(local_min_weight, dtype=torch.float64, device=device)
            max_weight = torch.tensor(local_max_weight, dtype=torch.float64, device=device)
            dist.all_reduce(min_weight, op=dist.ReduceOp.MIN); dist.all_reduce(max_weight, op=dist.ReduceOp.MAX)
            gathered_sets = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_sets, (
                local_exact_keys, local_state_keys, local_pid_keys, local_cloth_keys,
                local_sampled_images, local_sampled_pids, local_sampled_states,
            ))
            exact_keys = set().union(*(set(item[0]) for item in gathered_sets))
            state_keys = set().union(*(set(item[1]) for item in gathered_sets))
            pid_keys = set().union(*(set(item[2]) for item in gathered_sets))
            cloth_keys = set().union(*(set(item[3]) for item in gathered_sets))
            sampled_images = set().union(*(set(item[4]) for item in gathered_sets))
            sampled_pids = set().union(*(set(item[5]) for item in gathered_sets))
            sampled_states = set().union(*(set(item[6]) for item in gathered_sets))

            dist.barrier()
            if rank == 0:
                if logger is None:
                    raise RuntimeError("rank0 logger missing")
                metrics = evaluate_prcc(cfg, model.module, dataset, val_loader, val_loader_same, device, epoch, logger)
            else:
                metrics = None
            holder = [metrics]
            dist.broadcast_object_list(holder, src=0)
            metrics = holder[0]
            dist.barrier()
            lr_values = scheduler._get_lr(epoch)
            row = {
                "arm": arm,
                "seed": SEED,
                "epoch": epoch,
                "train_loss": reduced["total"],
                "ce_loss": reduced["ce"],
                "triplet_loss": reduced["triplet"],
                "bio_loss": reduced["bio"],
                "relation_loss": reduced["relation"],
                "relation_applied_loss": reduced["relation_applied"],
                "relation_lambda": 0.0 if arm == "P0" else phase6b.LAMBDA_REL,
                "total_loss": reduced["total"],
                "lr": float(lr_values[0]),
                "lr_bias": float(lr_values[-1]),
                "id_accuracy": reduced["id_acc"],
                "bio_accuracy": reduced["bio_acc"],
                "planned_updates": EXPECTED_STEPS_PER_EPOCH,
                "updates_completed": updates_completed,
                "amp_scale_min": float(amp_min.item()),
                "amp_scale_max": float(amp_max.item()),
                "amp_scale_end": float(amp_end.item()),
                "amp_skip": int(skip_sum.item() > 0),
                "skipped_steps": int(skip_sum.item()),
                "found_inf_count": int(found_sum.item()),
                "nonfinite_events": int(nonfinite_sum.item()),
                "active_images": len(dataset.train),
                "active_pids": int(dataset.num_train_pids),
                "active_clothing_states": int(dataset.num_train_clothes),
                "sampled_unique_images": len(sampled_images),
                "sampled_unique_pids": len(sampled_pids),
                "sampled_unique_clothing_states": len(sampled_states),
                "relation_active_batches": relation_batches,
                "relation_active_batch_rate": relation_batches / float(EXPECTED_STEPS_PER_EPOCH * WORLD_SIZE),
                "relations_per_epoch": relation_pairs,
                "unique_pair_count": len(exact_keys),
                "relation_pid_coverage": len(pid_keys),
                "relation_clothing_state_coverage": len(cloth_keys),
                **metrics,
            }
            exposure = {
                "arm": arm,
                "seed": SEED,
                "epoch": epoch,
                "all_images_available": True,
                "all_pids_available": True,
                "all_clothing_states_available": True,
                "sampled_unique_images": len(sampled_images),
                "sampled_unique_pids": len(sampled_pids),
                "sampled_unique_clothing_states": len(sampled_states),
                "relation_active_batches": relation_batches,
                "relation_active_batch_rate": row["relation_active_batch_rate"],
                "relations_per_epoch": relation_pairs,
                "unique_pair_count": len(exact_keys),
                "relation_pid_coverage": len(pid_keys),
                "relation_clothing_state_coverage": len(cloth_keys),
                "relation_weight_mean": weight_mean,
                "relation_weight_std": math.sqrt(weight_var),
                "relation_weight_min": 0.0 if not relation_pairs else float(min_weight.item()),
                "relation_weight_max": 0.0 if not relation_pairs else float(max_weight.item()),
                "planned_updates": EXPECTED_STEPS_PER_EPOCH,
                "updates_completed": updates_completed,
            }
            amp_row = {
                "arm": arm, "seed": SEED, "epoch": epoch,
                "amp_skip": row["amp_skip"], "skipped_steps": row["skipped_steps"],
                "found_inf_count": row["found_inf_count"], "nonfinite_events": row["nonfinite_events"],
                "amp_scale_min": row["amp_scale_min"], "amp_scale_max": row["amp_scale_max"],
                "amp_scale_end": row["amp_scale_end"], "planned_updates": EXPECTED_STEPS_PER_EPOCH,
                "updates_completed": updates_completed,
            }
            if rank == 0:
                metrics_writer.writerow(row); metrics_file.flush()
                exposure_writer.writerow(exposure); exposure_file.flush()
                amp_writer.writerow(amp_row); amp_file.flush()
                if float(row["cc_rank1"]) > best_rank1:
                    best_rank1 = float(row["cc_rank1"])
                    best_epoch = epoch
                    phase6b.save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_rank1, "criterion": "PRCC changed-clothes Rank-1 E1-E60"})
                print(json.dumps({
                    "arm": arm, "epoch": epoch, "cc_rank1": row["cc_rank1"], "cc_mAP": row["cc_mAP"],
                    "train_loss": row["train_loss"], "relation_active_batch_rate": row["relation_active_batch_rate"],
                    "amp_skips": row["skipped_steps"],
                }, sort_keys=True), flush=True)

            if epoch in GEOMETRY_EPOCHS:
                if rank == 0:
                    geometry = relation_geometry(model.module, dataset, bank, device, test_transform, epoch, arm)
                    geometry_rows.extend(geometry)
                    write_csv(run_dir / "relation_geometry.csv", geometry_rows, GEOMETRY_FIELDS)
                dist.barrier()

            if epoch == FORMAL_EPOCHS:
                if rank == 0:
                    phase6b.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                dist.barrier()
                save_rank_state(run_dir / f"rank{rank}_state_epoch060.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    rows = read_csv(run_dir / "metrics_per_epoch.csv")
                    write_json(run_dir / "checkpoint_epoch060.json", {
                        "phase": PHASE, "arm": arm, "epoch": epoch, "global_step": global_step,
                        "rank_states": [str(run_dir / f"rank{item}_state_epoch060.pt") for item in range(WORLD_SIZE)],
                        "model": str(run_dir / "model_final.pth"),
                    })
                    write_json(run_dir / "RUN_COMPLETE.json", {
                        "phase": PHASE, "arm": arm, "mode": "FULL_FORMAL_PRCC_SCHEDULE",
                        "seed": SEED, "completed_utc": utc_now(), "epochs_completed": epoch,
                        "optimizer_updates_per_rank": global_step, "metrics": str(run_dir / "metrics_per_epoch.csv"),
                        "best_epoch": best_epoch, "best_cc_rank1": best_rank1,
                        "best_model": str(run_dir / "model_best.pth"), "final_model": str(run_dir / "model_final.pth"),
                        "rows": len(rows), "common_init_sha256": common_meta["state_sha256"],
                        "initial_parameter_hash": local_initial_hash,
                    })
        if rank == 0:
            write_json(run_dir / "AMP.json", {
                "phase": PHASE, "arm": arm, "seed": SEED, "initial_scale": INITIAL_SCALE,
                "epochs_completed": FORMAL_EPOCHS, "metrics_source": str(run_dir / "amp_audit.csv"),
                "skip_event_source": "per-epoch amp_audit.csv and iteration-level amp_skip_events.csv",
            })
    finally:
        for stream in streams:
            stream.close()
        if logger is not None:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        phase6b.cleanup_dist()


def relation_geometry(model: torch.nn.Module, dataset: Any, bank: dict[str, Any], device: torch.device, transform_test: Any, checkpoint: int, arm: str) -> list[dict[str, Any]]:
    """Train-only geometry using the exact student identity feature feat[0]."""
    model.eval()
    loader = DataLoader(ImageDataset(dataset.train, transform=transform_test), batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    feature_bank: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            camids = batch["camid"].to(device)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            if torch.is_tensor(output):
                identity_feature = output
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                _score, eval_feat = output
                identity_feature = eval_feat[0] if isinstance(eval_feat, (tuple, list)) else eval_feat
            else:
                raise TypeError(f"unexpected PRCC eval model output: {type(output)!r}")
            values = F.normalize(identity_feature.float(), dim=-1).detach().cpu().numpy()
            for path, value in zip(batch["image_path"], values):
                feature_bank[bank["path_to_index"][str(path)]] = value
    if len(feature_bank) != len(dataset.train):
        raise RuntimeError(f"PRCC geometry feature count mismatch {len(feature_bank)} vs {len(dataset.train)}")
    matrix = np.stack([feature_bank[index] for index in range(len(dataset.train))]).astype(np.float32)
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    unique_pids = sorted(set(pids.tolist()))
    center_matrix = np.stack([F.normalize(torch.from_numpy(matrix[pids == pid]).mean(dim=0, keepdim=True), dim=-1).numpy()[0] for pid in unique_pids]).astype(np.float32)
    pid_to_center = {pid: index for index, pid in enumerate(unique_pids)}
    all_negative = matrix @ center_matrix.T
    for index, pid in enumerate(pids.tolist()):
        all_negative[index, pid_to_center[pid]] = -np.inf
    negative_by_index = np.max(all_negative, axis=1)

    pair_a = bank["arrays"]["index_a"].astype(np.int64)
    pair_b = bank["arrays"]["index_b"].astype(np.int64)
    if len(pair_a) == 0:
        raise RuntimeError("PRCC geometry has no relation pairs")
    positive = np.sum(matrix[pair_a] * matrix[pair_b], axis=1)
    negative = np.maximum(negative_by_index[pair_a], negative_by_index[pair_b])
    margin = positive - negative
    records = [{
        "arm": arm,
        "seed": SEED,
        "checkpoint": f"E{checkpoint}",
        "relation_count": int(len(positive) * 2),
        "pid_count": len(set(pids[pair_a].tolist())),
        "clothing_state_count": len({(int(pid), int(cloth)) for pid, cloth in zip(pids[pair_a].tolist(), clothes[pair_a].tolist())} | {(int(pid), int(cloth)) for pid, cloth in zip(pids[pair_b].tolist(), clothes[pair_b].tolist())}),
        "positive_cosine_mean": float(np.mean(positive)),
        "positive_cosine_median": float(np.median(positive)),
        "negative_center_cosine_mean": float(np.mean(negative)),
        "positive_negative_margin_mean": float(np.mean(margin)),
        "positive_negative_margin_median": float(np.median(margin)),
        "ordering_correct_rate": float(np.mean(positive > negative)),
        "geometry_source": "PRCC train-only full ground-truth relation bank; no query/gallery",
    }]
    model.train()
    return records


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def trajectory(rows: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"incomplete PRCC window E{start}-E{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    rank1 = np.asarray([number(row, "cc_rank1") * 100.0 for row in selected], dtype=np.float64)
    mAP = np.asarray([number(row, "cc_mAP") * 100.0 for row in selected], dtype=np.float64)
    auc = lambda values: float((np.trapezoid(values, x) if hasattr(np, "trapezoid") else np.trapz(values, x)) / (x[-1] - x[0]))
    best_rank1 = max(selected, key=lambda row: (number(row, "cc_rank1"), number(row, "cc_mAP")))
    best_map = max(selected, key=lambda row: (number(row, "cc_mAP"), number(row, "cc_rank1")))
    return {
        "start_epoch": start, "end_epoch": end, "epochs": len(selected),
        "mean_rank1": float(np.mean(rank1)), "mean_mAP": float(np.mean(mAP)),
        "auc_rank1": auc(rank1), "auc_mAP": auc(mAP),
        "slope_rank1": float(np.polyfit(x, rank1, 1)[0]), "slope_mAP": float(np.polyfit(x, mAP, 1)[0]),
        "e_first_rank1": float(rank1[0]), "e_first_mAP": float(mAP[0]),
        "e_last_rank1": float(rank1[-1]), "e_last_mAP": float(mAP[-1]),
        "e60_rank1": float(rank1[-1]) if end == FORMAL_EPOCHS else None,
        "e60_mAP": float(mAP[-1]) if end == FORMAL_EPOCHS else None,
        "best_rank1": number(best_rank1, "cc_rank1") * 100.0,
        "best_rank1_epoch": int(best_rank1["epoch"]),
        "best_mAP": number(best_map, "cc_mAP") * 100.0,
        "best_mAP_epoch": int(best_map["epoch"]),
    }


def window_name(start: int, end: int) -> str:
    return f"E{start}-E{end}"


def amp_summary(run_dir: Path, arm: str) -> dict[str, Any]:
    rows = read_csv(run_dir / "amp_audit.csv")
    if len(rows) != FORMAL_EPOCHS:
        raise RuntimeError(f"{arm} AMP rows incomplete: {len(rows)}")
    return {
        "arm": arm,
        "total_skips": sum(int(row["skipped_steps"]) for row in rows),
        "total_found_inf": sum(int(row["found_inf_count"]) for row in rows),
        "total_nonfinite_events": sum(int(row["nonfinite_events"]) for row in rows),
        "skip_epochs": ",".join(row["epoch"] for row in rows if int(row["skipped_steps"]) > 0),
        "scale_min": min(float(row["amp_scale_min"]) for row in rows),
        "scale_max": max(float(row["amp_scale_max"]) for row in rows),
        "scale_final": float(rows[-1]["amp_scale_end"]),
        "planned_updates": sum(int(row["planned_updates"]) for row in rows),
        "completed_updates": sum(float(row["updates_completed"]) for row in rows),
    }


def load_ltcc_structure() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = base.configure()
    dataset = base.build_dataset(cfg)
    groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(dataset.train):
        groups[int(item["pid"])][int(item["clothes_id"])].append(index)
    all_pairs = 0
    per_pid = {}
    for pid, values in groups.items():
        clothes = sorted(values)
        count = sum(len(values[a]) * len(values[b]) for a_i, a in enumerate(clothes) for b in clothes[a_i + 1:])
        per_pid[pid] = count
        all_pairs += count
    candidate_pairs = 43186
    try:
        ltcc_exposure = read_csv(ROOT / "outputs/new_method_track/phase6cr_relation_replication/R1_seed4321/relation_exposure.csv")
        active_rates = [float(row["relation_active_batch_rate"]) for row in ltcc_exposure]
    except Exception:
        active_rates = []
    clothes_per_pid = [len(values) for values in groups.values()]
    structure = {
        "dataset": "LTCC",
        "source": "Phase6C-R frozen train-only structure and executed relation exposure",
        "train_images": len(dataset.train),
        "pids": len(groups),
        "clothing_states": len({(pid, cloth) for pid, values in groups.items() for cloth in values}),
        "pids_with_ge2_clothes": sum(value >= 2 for value in clothes_per_pid),
        "mean_clothes_per_pid": float(np.mean(clothes_per_pid)),
        "median_clothes_per_pid": float(np.median(clothes_per_pid)),
        "min_clothes_per_pid": min(clothes_per_pid),
        "max_clothes_per_pid": max(clothes_per_pid),
        "valid_cross_clothes_unordered_pairs": all_pairs,
        "relation_bank_pairs": candidate_pairs,
        "relation_bank_density": candidate_pairs / max(1, all_pairs),
        "relation_active_batch_rate_mean": float(np.mean(active_rates)) if active_rates else None,
        "relation_active_batch_rate_min": float(np.min(active_rates)) if active_rates else None,
        "relation_active_batch_rate_max": float(np.max(active_rates)) if active_rates else None,
        "per_pid_relation_min": min(per_pid.values()),
        "per_pid_relation_median": float(np.median(list(per_pid.values()))),
        "per_pid_relation_max": max(per_pid.values()),
    }
    return structure, {"dataset": dataset, "per_pid": per_pid}


def analyze() -> dict[str, Any]:
    audit_data = load_audit()
    for arm, run_dir in RUN_DIRS.items():
        marker = run_dir / "RUN_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"{arm} is incomplete")
        complete = json.loads(marker.read_text())
        if complete.get("epochs_completed") != FORMAL_EPOCHS or complete.get("rows") != FORMAL_EPOCHS:
            raise RuntimeError(f"{arm} does not contain exactly E1-E60")

    summaries = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, run_dir in RUN_DIRS.items():
        rows = read_csv(run_dir / "metrics_per_epoch.csv")
        if len(rows) != FORMAL_EPOCHS or [int(row["epoch"]) for row in rows] != list(range(1, FORMAL_EPOCHS + 1)):
            raise RuntimeError(f"{arm} metrics are not exactly E1-E60")
        for start, end in WINDOWS:
            item = {"arm": arm, "seed": SEED, "setting": ARM_LABELS[arm], "window": window_name(start, end), **trajectory(rows, start, end)}
            summaries.append(item)
            lookup[(arm, item["window"])] = item

    late_rows = []
    for window in [window_name(*item) for item in WINDOWS]:
        p0 = lookup[("P0", window)]
        p1 = lookup[("P1", window)]
        late_rows.extend([
            {"kind": "arm", "arm": "P0", "seed": SEED, "window": window, **{key: value for key, value in p0.items() if key not in {"arm", "seed", "setting", "window"}}},
            {"kind": "arm", "arm": "P1", "seed": SEED, "window": window, **{key: value for key, value in p1.items() if key not in {"arm", "seed", "setting", "window"}}},
            {"kind": "delta_P1_minus_P0", "arm": "P1-P0", "seed": SEED, "window": window,
             **{key: float(p1[key] - p0[key]) for key in p0 if key not in {"arm", "seed", "setting", "window", "start_epoch", "end_epoch", "epochs", "best_rank1_epoch", "best_mAP_epoch"} and isinstance(p0[key], (int, float)) and p0[key] is not None and p1[key] is not None}},
        ])
    write_csv(OUT_ROOT / "late_window_analysis.csv", late_rows)

    metrics_rows = []
    for arm, run_dir in RUN_DIRS.items():
        metrics_rows.extend([{**row, "arm": arm, "seed": SEED} for row in read_csv(run_dir / "metrics_per_epoch.csv")])
    write_csv(OUT_ROOT / "metrics_per_epoch.csv", metrics_rows)
    exposure_rows = []
    geometry_rows = []
    amp_rows = []
    for arm, run_dir in RUN_DIRS.items():
        exposure_rows.extend(read_csv(run_dir / "relation_exposure.csv"))
        geometry_rows.extend(read_csv(run_dir / "relation_geometry.csv"))
        amp_rows.append(amp_summary(run_dir, arm))
    write_csv(OUT_ROOT / "relation_exposure.csv", exposure_rows)
    write_csv(OUT_ROOT / "relation_geometry.csv", geometry_rows, GEOMETRY_FIELDS)
    write_csv(OUT_ROOT / "amp_audit.csv", amp_rows)

    structure_prcc = dict(audit_data["relation_structure"])
    structure_prcc["dataset"] = "PRCC"
    structure_prcc["source"] = "Phase7A train-only full ground-truth relation bank and matched sampler exposure"
    active_rates = [float(row["relation_active_batch_rate"]) for row in exposure_rows if row["arm"] == "P1"]
    structure_prcc["relation_active_batch_rate_mean"] = float(np.mean(active_rates))
    structure_prcc["relation_active_batch_rate_min"] = float(np.min(active_rates))
    structure_prcc["relation_active_batch_rate_max"] = float(np.max(active_rates))
    structure_ltcc, _ltcc_details = load_ltcc_structure()
    write_csv(STRUCTURE_CSV, [structure_ltcc, structure_prcc])

    historical = audit_data["historical"]
    historical_rows = []
    for key, item in historical.items():
        if not isinstance(item, dict):
            continue
        result = item.get("independent_evaluation", {}).get("changed_clothes_C_to_A", {})
        historical_rows.append({
            "source": key,
            "label": item.get("label", key),
            "path": item.get("path", item.get("root", "")),
            "rank1": result.get("rank1", ""),
            "mAP": result.get("mAP", ""),
            "checkpoint_sha256": item.get("checkpoint_sha256", ""),
            "used_as_matched_control": False,
        })
    historical_rows.extend([
        {"source": "historical_reference_user", "label": "PRCC Baseline historical reference", "path": "user-frozen", "rank1": 0.666949, "mAP": 0.636759, "checkpoint_sha256": "", "used_as_matched_control": False},
        {"source": "historical_reference_user", "label": "PRCC Full historical reference", "path": "user-frozen", "rank1": 0.666384, "mAP": 0.639903, "checkpoint_sha256": "", "used_as_matched_control": False},
    ])
    write_csv(OUT_ROOT / "historical_prcc_comparison.csv", historical_rows)

    p0_late = lookup[("P0", PRIMARY_WINDOW)]
    p1_late = lookup[("P1", PRIMARY_WINDOW)]
    primary = {
        "window": PRIMARY_WINDOW,
        "P0": p0_late,
        "P1": p1_late,
        "P1_minus_P0": {
            "mean_rank1_pp": p1_late["mean_rank1"] - p0_late["mean_rank1"],
            "mean_mAP_pp": p1_late["mean_mAP"] - p0_late["mean_mAP"],
            "auc_rank1_pp": p1_late["auc_rank1"] - p0_late["auc_rank1"],
            "auc_mAP_pp": p1_late["auc_mAP"] - p0_late["auc_mAP"],
            "slope_rank1_pp": p1_late["slope_rank1"] - p0_late["slope_rank1"],
            "slope_mAP_pp": p1_late["slope_mAP"] - p0_late["slope_mAP"],
            "E60_rank1_pp": p1_late["e60_rank1"] - p0_late["e60_rank1"],
            "E60_mAP_pp": p1_late["e60_mAP"] - p0_late["e60_mAP"],
            "best_rank1_pp": p1_late["best_rank1"] - p0_late["best_rank1"],
            "best_mAP_pp": p1_late["best_mAP"] - p0_late["best_mAP"],
        },
    }
    amp_by_arm = {row["arm"]: row for row in amp_rows}
    p0_skips = amp_by_arm["P0"]["total_skips"]
    p1_skips = amp_by_arm["P1"]["total_skips"]
    severe_amp = bool(p1_skips - p0_skips >= max(4, math.ceil(0.25 * max(1, p0_skips))) or amp_by_arm["P1"]["scale_min"] <= amp_by_arm["P0"]["scale_min"] / 2)
    q1 = "YES" if primary["P1_minus_P0"]["auc_rank1_pp"] > 0 else "NO"
    q2 = "YES" if primary["P1_minus_P0"]["auc_rank1_pp"] > 0 else "NO"
    q3 = "YES" if primary["P1_minus_P0"]["auc_mAP_pp"] > 0 else "NO"
    geometry_at_late = {}
    for arm in ("P0", "P1"):
        rows = [row for row in geometry_rows if row["arm"] == arm and row["checkpoint"] == "E60"]
        if len(rows) != 1:
            raise RuntimeError(f"missing E60 geometry row for {arm}")
        geometry_at_late[arm] = rows[0]
    geometry_dimensions = ["positive_cosine_mean", "positive_negative_margin_mean", "ordering_correct_rate"]
    geometry_deltas = {key: float(geometry_at_late["P1"][key]) - float(geometry_at_late["P0"][key]) for key in geometry_dimensions}
    geometry_positive = sum(value > 0 for value in geometry_deltas.values())
    q4 = "YES" if geometry_positive == len(geometry_dimensions) else ("NO" if geometry_positive == 0 else "MIXED")
    q5 = "YES" if severe_amp else "NO"
    q6 = "YES" if primary["P1_minus_P0"]["auc_rank1_pp"] > 0 and primary["P1_minus_P0"]["auc_mAP_pp"] > 0 and not severe_amp else ("PARTIAL" if primary["P1_minus_P0"]["auc_rank1_pp"] > 0 or primary["P1_minus_P0"]["auc_mAP_pp"] > 0 else "NO")
    if q6 == "YES":
        promotion = "PROMOTE"
        next_step = "PRCC second-seed replication recommended; do not auto-run. Relation mechanism ablation can follow only after human confirmation."
    else:
        promotion = "STOP"
        next_step = "STOP; diagnose relation exposure, PRCC clothing structure, geometry, trajectory, and AMP before any change. Do not tune lambda or open SR3."
    questions = {
        "Q1_P1_improves_PRCC_Rank1": q1,
        "Q2_late_window_Rank1_positive": q2,
        "Q3_late_window_mAP_positive": q3,
        "Q4_relation_geometry_improves": q4,
        "Q5_obvious_AMP_numerical_confound": q5,
        "Q6_LTCC_benefit_cross_dataset_generalizes": q6,
    }
    common_audit = json.loads(COMMON_AUDIT.read_text())
    manifests = {arm: json.loads((run_dir / "run_manifest.json").read_text()) for arm, run_dir in RUN_DIRS.items()}
    matched = {
        "common_init_sha256_equal": manifests["P0"]["common_init"]["sha256"] == manifests["P1"]["common_init"]["sha256"],
        "initial_parameter_hash_equal": manifests["P0"]["initial_parameter_hash"] == manifests["P1"]["initial_parameter_hash"],
        "config_exact_equal_except_output": manifests["P0"]["optimizer"] == manifests["P1"]["optimizer"] and manifests["P0"]["scheduler"] == manifests["P1"]["scheduler"] and manifests["P0"]["amp"] == manifests["P1"]["amp"],
        "relation_bank_sha256_equal": manifests["P0"]["relation_bank_sha256"] == manifests["P1"]["relation_bank_sha256"],
        "sampler_plan_exact_equal": read_csv(RUN_DIRS["P0"] / "sampler_plan.csv") == [{**row, "arm": "P0"} for row in read_csv(RUN_DIRS["P1"] / "sampler_plan.csv")],
        "only_method_difference": "P1 applies uniform relation; P0 does not",
    }
    # The preceding comparison intentionally normalized only the arm label;
    # compare a second way so a malformed row cannot pass accidentally.
    p0_sampler = read_csv(RUN_DIRS["P0"] / "sampler_plan.csv")
    p1_sampler = read_csv(RUN_DIRS["P1"] / "sampler_plan.csv")
    normalized_p0 = [{key: value for key, value in row.items() if key != "arm"} for row in p0_sampler]
    normalized_p1 = [{key: value for key, value in row.items() if key != "arm"} for row in p1_sampler]
    matched["sampler_plan_exact_equal"] = normalized_p0 == normalized_p1
    if not all(matched.values()):
        raise RuntimeError(f"matched P0/P1 audit failed: {matched}")

    manifest = {
        "phase": PHASE,
        "seed": SEED,
        "mode": "FULL_FORMAL_PRCC_TWO_ARM_VALIDATION",
        "allowed_arms": ["P0", "P1"],
        "forbidden_arms_started": ["B2", "B3", "SR3", "H_ID", "DS", "DW", "GRL", "Reverse", "Relation Curriculum", "new sampler", "diffusion"],
        "epochs": FORMAL_EPOCHS,
        "primary_window": PRIMARY_WINDOW,
        "windows": [window_name(*window) for window in WINDOWS],
        "feasibility": str(FEASIBILITY_REPORT),
        "feasibility_pass": True,
        "config_audit": str(CONFIG_AUDIT),
        "frozen_relation_method": str(METHOD_DOC),
        "common_init_audit": str(COMMON_AUDIT),
        "common_init": common_audit,
        "matched_training_audit": matched,
        "arms": manifests,
        "summaries": summaries,
        "primary_comparison": primary,
        "amp_audit": amp_rows,
        "amp_serious_confound_rule": "P1 total skips − P0 total skips >= max(4,25% of P0) or P1 scale floor <= half P0 floor",
        "amp_serious_confound": severe_amp,
        "relation_geometry": {"E60": geometry_at_late, "delta_P1_minus_P0": geometry_deltas, "answer": q4},
        "ltcc_vs_prcc_structure": str(STRUCTURE_CSV),
        "historical_prcc_comparison": str(OUT_ROOT / "historical_prcc_comparison.csv"),
        "questions": questions,
        "promotion_decision": promotion,
        "next_step_recommendation": next_step,
        "historical_reference": {
            "baseline_rank1": 66.6949,
            "baseline_mAP": 63.6759,
            "full_rank1": 66.6384,
            "full_mAP": 63.9903,
            "matched_control_is_historical": False,
        },
        "two_seed_ltcc_is_not_full_statistical_validation": True,
        "phase7b_started": False,
        "sr3_reopened": False,
        "artifacts": {
            "metrics_per_epoch": str(OUT_ROOT / "metrics_per_epoch.csv"),
            "late_window_analysis": str(OUT_ROOT / "late_window_analysis.csv"),
            "relation_exposure": str(OUT_ROOT / "relation_exposure.csv"),
            "relation_geometry": str(OUT_ROOT / "relation_geometry.csv"),
            "amp_audit": str(OUT_ROOT / "amp_audit.csv"),
            "historical_prcc_comparison": str(OUT_ROOT / "historical_prcc_comparison.csv"),
            "structure": str(STRUCTURE_CSV),
            "report": str(REPORT_PATH),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    write_report(manifest)
    return manifest


def write_report(manifest: dict[str, Any]) -> None:
    primary = manifest["primary_comparison"]
    delta = primary["P1_minus_P0"]
    p0 = primary["P0"]
    p1 = primary["P1"]
    questions = manifest["questions"]
    amp = {row["arm"]: row for row in manifest["amp_audit"]}
    lines = [
        "# PHASE 7A — PRCC Cross-Dataset Uniform Cross-Clothes Relation Validation",
        "",
        "## 1. Scope",
        "",
        "This phase runs only matched PRCC P0 (Bio baseline) and P1 (Bio + Uniform Cross-Clothes Relation), seed 1234, for the full formal 60-epoch PRCC schedule. SR3, reliability weighting, H_ID, DS, DW, GRL, Reverse, curriculum, sampler changes, and diffusion are not entered.",
        "",
        f"- Primary window: `{manifest['primary_window']}`; PRCC formal schedule: 60 epochs × 551 updates/rank/epoch.",
        f"- Feasibility: **PASS**; report `{manifest['feasibility']}`.",
        "- The two-seed LTCC result is replication evidence, not full statistical validation.",
        "",
        "## 2. Frozen LTCC evidence",
        "",
        "| LTCC evidence | E51–E60 AUC Δ Uniform Relation − Bio |",
        "|---|---:|",
        "| Seed 1234 Rank-1 / mAP | +5.2012 / +4.5194 pp |",
        "| Seed 4321 Rank-1 / mAP | +4.1383 / +3.4189 pp |",
        "| Two-seed mean Rank-1 / mAP | +4.6698 / +3.9691 pp |",
        "",
        "These are frozen LTCC references only; they do not determine the PRCC result.",
        "",
        "## 3. PRCC configuration and clothing-label audit",
        "",
        f"- Config audit: `{manifest['config_audit']}`.",
        "- PRCC train: 17,896 images, 150 PIDs, 300 clothing-state labels, 3 camera prefixes; validation images 5,002.",
        "- Primary retrieval: 3,543 changed-clothes C queries against 3,384 A gallery images; secondary B→A is retained unchanged.",
        "- CLIP_DIM=768, Bio text dimension=768, Bio index 0, non-Bio indices [1,2] (Hair + Clothing).",
        "- Batch 16, NUM_INSTANCES 4, SGD LR 2e-6, weight decay 0.05, 20-epoch linear warmup, formal 60-epoch cosine schedule, PRCC AMP initial scale 512.",
        "- Historical PRCC checkpoints were audited but were not used as P0/P1 initialization.",
        "",
        "## 4. PRCC clothing-label feasibility",
        "",
        f"- All 150/150 train PIDs have two clothing states; A/B share one state and C is the changed state under the actual repository parser.",
        f"- Valid train-only same-PID/different-clothes pairs: `{manifest['common_init']['relation_bank']['pairs']}`; every pair has uniform weight 1.0.",
        "- No query/gallery item was used to construct relation labels or the training relation bank.",
        "",
        "## 5. LTCC versus PRCC relation structure",
        "",
        f"- Machine-readable comparison: `{manifest['ltcc_vs_prcc_structure']}`.",
        "- PRCC uses its full train-only ground-truth bank because SR3/reliability candidate selection is prohibited; the Phase 6B relation function remains the executed implementation.",
        "- This distinction is reported explicitly as relation-bank density and exposure, rather than hidden as a sampler change.",
        "",
        "## 6. Frozen loss implementation",
        "",
        f"- Frozen implementation: `{manifest['frozen_relation_method']}`.",
        "- Feature source is normalized student `feat[0]`; positive truth is same PID + different clothes; pair loss is the executed `1 − cosine` term; empty pairs return differentiable zero; λ=0.1.",
        "- P0 applies no relation term. P1 is the only arm applying the uniform relation term. No relation weighting or reliability score is present.",
        "",
        "## 7. Common initialization and matched training",
        "",
        f"- Common-init audit: `{manifest['common_init_audit']}`.",
        f"- Common model-state SHA256: `{manifest['common_init']['model_state_sha256']}`.",
        f"- P0/P1 initial parameter exact equality: `{manifest['common_init']['initial_parameter_equality']}`; runtime equality: `{manifest['matched_training_audit']['initial_parameter_hash_equal']}`.",
        f"- P0/P1 sampler plan exact equality: `{manifest['matched_training_audit']['sampler_plan_exact_equal']}`.",
        f"- Only intended method difference: `{manifest['matched_training_audit']['only_method_difference']}`.",
        "",
        "## 8. Training trajectory",
        "",
        "All retrieval values below are percentages; AUC and slopes are percentage-point quantities.",
        "",
        "| Arm | Window | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Slope R1 | Slope mAP | E60 R1 | E60 mAP | Best R1 (epoch) | Best mAP (epoch) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["summaries"]:
        lines.append(f"| {row['arm']} | {row['window']} | {row['mean_rank1']:.4f} | {row['mean_mAP']:.4f} | {row['auc_rank1']:.4f} | {row['auc_mAP']:.4f} | {row['slope_rank1']:.4f} | {row['slope_mAP']:.4f} | {row['e_last_rank1']:.4f} | {row['e_last_mAP']:.4f} | {row['best_rank1']:.4f} (E{row['best_rank1_epoch']}) | {row['best_mAP']:.4f} (E{row['best_mAP_epoch']}) |")
    lines += [
        "",
        "## 9. Primary late-window comparison: P1 − P0, E51–E60",
        "",
        "| Quantity | P0 Bio | P1 Bio + Uniform Relation | Δ P1−P0 (pp) |",
        "|---|---:|---:|---:|",
        f"| Mean Rank-1 | {p0['mean_rank1']:.4f} | {p1['mean_rank1']:.4f} | {delta['mean_rank1_pp']:+.4f} |",
        f"| Mean mAP | {p0['mean_mAP']:.4f} | {p1['mean_mAP']:.4f} | {delta['mean_mAP_pp']:+.4f} |",
        f"| normalized trapezoidal AUC Rank-1 | {p0['auc_rank1']:.4f} | {p1['auc_rank1']:.4f} | {delta['auc_rank1_pp']:+.4f} |",
        f"| normalized trapezoidal AUC mAP | {p0['auc_mAP']:.4f} | {p1['auc_mAP']:.4f} | {delta['auc_mAP_pp']:+.4f} |",
        f"| slope Rank-1 | {p0['slope_rank1']:.4f} | {p1['slope_rank1']:.4f} | {delta['slope_rank1_pp']:+.4f} |",
        f"| slope mAP | {p0['slope_mAP']:.4f} | {p1['slope_mAP']:.4f} | {delta['slope_mAP_pp']:+.4f} |",
        f"| E60 Rank-1 | {p0['e60_rank1']:.4f} | {p1['e60_rank1']:.4f} | {delta['E60_rank1_pp']:+.4f} |",
        f"| E60 mAP | {p0['e60_mAP']:.4f} | {p1['e60_mAP']:.4f} | {delta['E60_mAP_pp']:+.4f} |",
        f"| BEST Rank-1 | {p0['best_rank1']:.4f} (E{p0['best_rank1_epoch']}) | {p1['best_rank1']:.4f} (E{p1['best_rank1_epoch']}) | {delta['best_rank1_pp']:+.4f} |",
        f"| BEST mAP | {p0['best_mAP']:.4f} (E{p0['best_mAP_epoch']}) | {p1['best_mAP']:.4f} (E{p1['best_mAP_epoch']}) | {delta['best_mAP_pp']:+.4f} |",
        "",
        "## 10. Relation exposure",
        "",
        f"- Combined exposure: `{manifest['artifacts']['relation_exposure']}`.",
        "- P0 and P1 retain all PRCC train images, PIDs, and clothing states as available. The sampler is the audited PRCC identity sampler; no relation-priority sampler is used.",
        "- P1 relation weights are checked per epoch as mean=1, std=0, min=1, max=1; P0 reports the same candidate exposure diagnostically but applies λ=0.",
        "",
        "## 11. Relation geometry",
        "",
        f"- Geometry uses PRCC train-only features at E20/E40/E60; query/gallery are excluded. File: `{manifest['artifacts']['relation_geometry']}`.",
        f"- E60 P1−P0 deltas: `{json.dumps(manifest['relation_geometry']['delta_P1_minus_P0'], sort_keys=True)}`.",
        f"- Geometry answer: **{manifest['relation_geometry']['answer']}**.",
        "",
        "## 12. AMP audit",
        "",
        "| Arm | Total skips | found_inf | nonfinite | Skip epochs | Scale min | Scale max | Final scale | Completed updates |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for arm_name in ("P0", "P1"):
        row = amp[arm_name]
        lines.append(f"| {arm_name} | {row['total_skips']} | {row['total_found_inf']} | {row['total_nonfinite_events']} | {row['skip_epochs']} | {row['scale_min']:.1f} | {row['scale_max']:.1f} | {row['scale_final']:.1f} | {row['completed_updates']:.1f} |")
    lines += [
        "",
        f"- Severe numerical-confound rule: `{manifest['amp_serious_confound_rule']}`.",
        f"- Severe AMP numerical confound: **{manifest['amp_serious_confound']}**.",
        "",
        "## 13. Historical PRCC reference",
        "",
        f"- Historical references are in `{manifest['historical_prcc_comparison']}` and are not substituted for the matched P0 control.",
        "- User-frozen historical values: Baseline 66.6949 Rank-1 / 63.6759 mAP; Full 66.6384 Rank-1 / 63.9903 mAP.",
        "",
        "## 14. Q1–Q6",
        "",
        f"- Q1. P1 是否在 PRCC 提高 Rank-1？ **{questions['Q1_P1_improves_PRCC_Rank1']}**.",
        f"- Q2. E51–E60 Rank-1 是否复制正向收益？ **{questions['Q2_late_window_Rank1_positive']}**.",
        f"- Q3. E51–E60 mAP 是否复制正向收益？ **{questions['Q3_late_window_mAP_positive']}**.",
        f"- Q4. Relation geometry 是否改善？ **{questions['Q4_relation_geometry_improves']}**.",
        f"- Q5. 是否存在明显 AMP numerical confound？ **{questions['Q5_obvious_AMP_numerical_confound']}**.",
        f"- Q6. LTCC relation benefit 是否 cross-dataset generalize？ **{questions['Q6_LTCC_benefit_cross_dataset_generalizes']}**.",
        "",
        "## 15. Promotion decision and final diagnosis",
        "",
        f"**{manifest['promotion_decision']}**",
        "",
        f"{manifest['next_step_recommendation']}",
        "",
        "This result addresses only uniform ground-truth same-ID cross-clothes relation supervision. It does not establish that SR3/reliability works, and it does not authorize Phase 7B or any automatic second seed.",
        "",
        "## Artifacts",
        "",
        f"- `{manifest['artifacts']['metrics_per_epoch']}`",
        f"- `{manifest['artifacts']['late_window_analysis']}`",
        f"- `{manifest['artifacts']['relation_exposure']}`",
        f"- `{manifest['artifacts']['relation_geometry']}`",
        f"- `{manifest['artifacts']['amp_audit']}`",
        f"- `{manifest['artifacts']['structure']}`",
        f"- `{manifest['artifacts']['report']}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "prepare", "train", "analyze"), required=True)
    parser.add_argument("--arm", choices=("P0", "P1"), default=None)
    args = parser.parse_args()
    if args.mode == "audit":
        audit()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for --mode train")
        train_arm(args.arm)
    else:
        result = analyze()
        print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "promotion": result["promotion_decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
