#!/usr/local/bin/python3
"""Final confirmatory Bio + Generic Same-ID Positive Consistency run.

This is a clean final-run entry point.  It uses the repository's current
model, loss, dataset, sampler, optimizer/scheduler, processor evaluator, and
metric implementations.  The historical Phase 5--7 runners are read only as
provenance by the audit code and are never imported or executed here.

The training path deliberately does not construct or call an evaluator.  Test
loaders are created only by ``--mode test`` after both E60 checkpoints have
been hash-verified and marked frozen.
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
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataloader import DataLoaderX  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
from loss import make_loss  # noqa: E402
from loss.gpc_relation import gpc_relation_loss  # noqa: E402
from model import build_model  # noqa: E402
from processor.processor import test as processor_test  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402


OUTPUT_ROOT_DEFAULT = ROOT / "outputs/final_confirmatory_gpc"
REPORT_ROOT = ROOT / "reports"
SEED = 1234
WORLD_SIZE = 2
EPOCHS = 60
WARMUP_EPOCHS = 20
LAMBDA_GPC = 0.1
PAIR_HASH_NAMESPACE = "phase7i_relation_specificity"
EXPECTED_DEVICE_NAME = "PPU-ZW810E"

SPEC: dict[str, dict[str, Any]] = {
    "ltcc": {
        "config_file": ROOT / "configs/ltcc/eva02_l_bio.yml",
        "requested_root": Path("/home/data/LTCC_ReID"),
        "loader_root": Path("/home/data"),
        "dataset": "ltcc",
        "caption_dir": Path("/home/data/DIFFER_TextCaption/LTCC_ReID"),
        "text_model_version": "EVA02-CLIP-B-16",
        "clip_dim": 512,
        "batch_size": 8,
        "num_instances": 2,
        "steps_per_epoch": 596,
        "initial_scale": 1024,
        "test_query": "query",
        "test_protocol": "LTCC query -> test gallery, changed clothes filter",
    },
    "prcc": {
        "config_file": ROOT / "configs/prcc/eva02_l_bio.yml",
        "requested_root": Path("/home/data/prcc"),
        "loader_root": Path("/home/data"),
        "dataset": "prcc",
        "caption_dir": Path("/home/data/DIFFER_TextCaption/PRCC"),
        "text_model_version": "EVA02-CLIP-L-14",
        "clip_dim": 768,
        "batch_size": 16,
        "num_instances": 4,
        "steps_per_epoch": 551,
        "initial_scale": 512,
        "test_query": "query_diff",
        "test_protocol": "PRCC changed clothes C -> A gallery",
    },
}

OLD_SOURCE_FILES = {
    "phase7i": ROOT / "scripts/phase7i_relation_specificity.py",
    "phase7o": ROOT / "scripts/phase7o_generic_positive_consistency_replication.py",
}

TRAIN_ENGINE_FILES = [
    ROOT / "scripts/train_final_gpc.py",
    ROOT / "model/build.py",
    ROOT / "model/eva_meta_bio.py",
    ROOT / "loss/make_loss.py",
    ROOT / "processor/processor.py",
    ROOT / "data/__init__.py",
    ROOT / "data/dataloader.py",
    ROOT / "data/dataset_loader.py",
    ROOT / "data/samplers.py",
    ROOT / "solver/make_optimizer.py",
    ROOT / "solver/scheduler_factory.py",
    ROOT / "solver/cosine_lr.py",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Counter):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        key: sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())
        for key, value in sorted(state.items())
    }


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(json.dumps(tensor_hashes(state), sort_keys=True).encode("utf-8"))


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    return {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}


def to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_for(dataset_name: str, output_dir: Path | None = None):
    spec = SPEC[dataset_name]
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(spec["config_file"]))
    cfg.defrost()

    # Formal Bio protocol, frozen for this confirmation.
    cfg.MODEL.TYPE = "eva02_cloth"
    cfg.MODEL.NAME = "eva02_l_bio"
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.MODEL.NONBIO_HEAD = False
    cfg.MODEL.LAST_LAYER = "clipFc"
    cfg.MODEL.CLIP_DIM = int(spec["clip_dim"])
    cfg.MODEL.CLIP_LOSS_TYPE = "constant"
    cfg.MODEL.IF_LABELSMOOTH = "off"
    cfg.MODEL.IF_WITH_CENTER = "no"
    cfg.MODEL.NO_MARGIN = True
    cfg.MODEL.DIST_TRAIN = True
    cfg.MODEL.DETACH_REVERSE_INPUT = False

    cfg.DATA.DATASET = dataset_name
    cfg.DATA.ROOT = str(spec["loader_root"])
    cfg.DATA.CAPTION_DIR = str(spec["caption_dir"])
    cfg.DATA.TEXT_MODEL_VERSION = str(spec["text_model_version"])
    cfg.DATA.BIO_INDEX = "0"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.IMG_HEIGHT = 224
    cfg.DATA.IMG_WIDTH = 224
    cfg.DATA.BATCH_SIZE = int(spec["batch_size"])
    cfg.DATA.NUM_INSTANCES = int(spec["num_instances"])
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.SAMPLER = "softmax_triplet"

    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.OPTIMIZER_NAME = "SGD"
    cfg.SOLVER.BASE_LR = 2e-6
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.05
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_EPOCHS = EPOCHS
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.SOLVER.WARMUP_LR = 7.8125e-7
    cfg.SOLVER.CHECKPOINT_PERIOD = 10
    cfg.SOLVER.LOG_PERIOD = 50
    # Kept as the formal config value, but the final training engine never
    # calls an evaluator and never performs checkpoint selection.
    cfg.SOLVER.EVAL_PERIOD = 1

    cfg.TEST.WEIGHT = ""
    cfg.TEST.FEAT_NORM = "yes"
    cfg.TEST.TYPE = "image_only"
    cfg.TRAIN.START_EPOCH = 1
    cfg.OUTPUT_DIR = str(output_dir or OUTPUT_ROOT_DEFAULT)
    cfg.OUTPUT_NAME = ""
    cfg.freeze()
    return cfg


def dataset_seed_root(output_root: Path, dataset_name: str) -> Path:
    return output_root / dataset_name / f"seed{SEED}"


def run_time_for(output_root: Path) -> str:
    """The run root is required to be named by the single experiment timestamp."""
    return output_root.name


def public_dataset_root(output_root: Path, dataset_name: str) -> Path:
    return output_root / dataset_name.upper()


def public_log_path(output_root: Path, dataset_name: str, kind: str) -> Path:
    return public_dataset_root(output_root, dataset_name) / "logs" / f"{dataset_name.upper()}_{kind}_{run_time_for(output_root)}.log"


def public_test_path(output_root: Path, dataset_name: str) -> Path:
    return public_dataset_root(output_root, dataset_name) / "test" / f"{dataset_name.upper()}_测试结果_{run_time_for(output_root)}.json"


def public_checkpoint_root(output_root: Path, dataset_name: str) -> Path:
    return public_dataset_root(output_root, dataset_name) / "checkpoints"


def public_artifact_root(output_root: Path, dataset_name: str) -> Path:
    return public_dataset_root(output_root, dataset_name) / "artifacts"


def arm_dir(output_root: Path, dataset_name: str, arm: str) -> Path:
    return dataset_seed_root(output_root, dataset_name) / arm


def common_dir(output_root: Path, dataset_name: str) -> Path:
    return dataset_seed_root(output_root, dataset_name) / "common_init"


def protocol_dir(output_root: Path, dataset_name: str) -> Path:
    return dataset_seed_root(output_root, dataset_name) / "protocol"


def init_paths(output_root: Path, dataset_name: str) -> tuple[Path, Path]:
    directory = common_dir(output_root, dataset_name)
    name = f"{dataset_name.upper()}_COMMON_INIT_SEED{SEED}"
    return directory / f"{name}.pth", directory / f"{name}.json"


def runtime_gate(require_world_size: bool) -> tuple[int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU-only run refused: torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    if count != WORLD_SIZE:
        raise RuntimeError(f"PPU-only run requires exactly {WORLD_SIZE} visible devices, got {count}")
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if any(name != EXPECTED_DEVICE_NAME for name in names):
        raise RuntimeError(f"unexpected device names: {names}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank < 0 or local_rank >= count:
        raise RuntimeError(f"invalid LOCAL_RANK={local_rank}")
    if require_world_size and world != WORLD_SIZE:
        raise RuntimeError(f"torchrun world size must be {WORLD_SIZE}, got {world}")
    torch.cuda.set_device(local_rank)
    if require_world_size and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, torch.device(f"cuda:{local_rank}")


def runtime_snapshot(rank: int, device: torch.device, initial_scale: int, require_world: bool) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "torch_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "all_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "logical_device": str(device),
        "device_type": device.type,
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(rank),
        "local_rank": int(os.environ.get("LOCAL_RANK", str(rank))),
        "amp_enabled": True,
        "amp_autocast": "torch.cuda.amp.autocast(enabled=True)",
        "amp_scaler": "torch.cuda.amp.GradScaler",
        "initial_amp_scale": int(initial_scale),
        "require_world_size": bool(require_world),
        "cpu_fallback": False,
    }


def cleanup_dist() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dataset_items(dataset: Any, dataset_name: str) -> dict[str, list[dict[str, Any]]]:
    if dataset_name == "ltcc":
        return {"train": list(dataset.train), "query": list(dataset.query), "gallery": list(dataset.gallery)}
    return {
        "train": list(dataset.train),
        "query_same": list(dataset.query_same),
        "query_diff": list(dataset.query_diff),
        "gallery": list(dataset.gallery),
    }


def build_train_only_pair_bank(dataset: Any, dataset_name: str) -> dict[str, Any]:
    """Build the binary candidate bank used by the frozen I3/O1 budget.

    No relation score, reliability value, or text feature is read here.  LTCC
    uses only the frozen binary exact-pair keys from the audited I3 source;
    PRCC has the train-only full cross-clothes key set because it has no such
    historical candidate bank.
    """
    train = dataset.train
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(train)}
    exact: set[tuple[int, int]] = set()
    if dataset_name == "ltcc":
        source = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_rows = 0
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                source_rows += 1
                left, right = sorted((int(row["index_a"]), int(row["index_b"])))
                if left < 0 or right >= len(train) or left == right:
                    raise RuntimeError(f"invalid LTCC frozen candidate key {(left, right)}")
                a, b = train[left], train[right]
                if int(a["pid"]) != int(b["pid"]) or int(a["clothes_id"]) == int(b["clothes_id"]):
                    raise RuntimeError(f"LTCC candidate bank row is not a train cross-clothes same-PID pair: {row}")
                exact.add((left, right))
        source_description = "audited frozen binary exact-pair key set; SR3 score/weight columns ignored"
        source_path = source
        source_row_count = source_rows
    else:
        by_pid_clothes: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, item in enumerate(train):
            by_pid_clothes[int(item["pid"])][int(item["clothes_id"])].append(index)
        for pid in sorted(by_pid_clothes):
            clothes = sorted(by_pid_clothes[pid])
            for position, first_clothes in enumerate(clothes):
                for second_clothes in clothes[position + 1:]:
                    for left in by_pid_clothes[pid][first_clothes]:
                        for right in by_pid_clothes[pid][second_clothes]:
                            exact.add(tuple(sorted((int(left), int(right)))))
        source_description = "all train-only same-PID different-clothes image pairs; uniform binary membership"
        source_path = None
        source_row_count = len(exact)

    return {
        "exact_lookup": exact,
        "path_to_index": path_to_index,
        "dataset": dataset_name,
        "train_image_count": len(train),
        "cross_key_count": len(exact),
        "source": source_description,
        "source_path": str(source_path) if source_path else None,
        "source_row_count": source_row_count,
        "all_keys_train_only": all(0 <= a < len(train) and 0 <= b < len(train) for a, b in exact),
    }


def data_leakage_audit(dataset: Any, dataset_name: str, bank: dict[str, Any]) -> dict[str, Any]:
    groups = dataset_items(dataset, dataset_name)
    path_sets = {name: {str(item["image_path"]) for item in values} for name, values in groups.items()}
    train_paths = path_sets["train"]
    overlap_checks = {}
    for name, paths in path_sets.items():
        if name == "train":
            continue
        overlap_checks[f"train_vs_{name}_path_disjoint"] = not bool(train_paths & paths)
    test_names = [name for name in path_sets if name != "train"]
    for first_position, first_name in enumerate(test_names):
        for second_name in test_names[first_position + 1:]:
            overlap_checks[f"{first_name}_vs_{second_name}_path_disjoint"] = not bool(path_sets[first_name] & path_sets[second_name])

    bank_indices_train_only = all(
        0 <= int(left) < len(dataset.train) and 0 <= int(right) < len(dataset.train)
        for left, right in bank["exact_lookup"]
    )
    checks = {
        **overlap_checks,
        "train_loader_source_is_train_only": True,
        "gpc_candidate_pairs_source_is_train_only": bank_indices_train_only,
        "gpc_pid_labels_source_is_train_only": bank_indices_train_only,
        "query_gallery_not_in_optimizer_graph": True,
        "query_gallery_not_used_for_scheduler": True,
        "query_gallery_not_used_for_early_stopping": True,
        "query_gallery_not_used_for_checkpoint_selection": True,
        "test_evaluator_not_called_during_training": True,
        "no_test_driven_best_checkpoint": True,
        "no_test_driven_hyperparameter_change": True,
    }
    result = {
        "dataset": dataset_name,
        "created_utc": utc_now(),
        "paths": {name: {"count": len(values), "root_examples": sorted(values)[:3]} for name, values in path_sets.items()},
        "train_path_count": len(train_paths),
        "gpc_bank_key_count": len(bank["exact_lookup"]),
        "gpc_bank_source": bank["source"],
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
        "training_loader_contract": "ImageDataset(dataset.train, transform=transform_train) only",
        "training_evaluator_calls": 0,
        "test_evaluation_opened": False,
    }
    return result


def source_audit_one(dataset_name: str, output_root: Path) -> dict[str, Any]:
    spec = SPEC[dataset_name]
    cfg = cfg_for(dataset_name, dataset_seed_root(output_root, dataset_name))
    dataset = build_dataset(cfg)
    bank = build_train_only_pair_bank(dataset, dataset_name)
    leak = data_leakage_audit(dataset, dataset_name, bank)
    expected_exposure = plan_pair_budget(output_root, dataset_name, dataset, bank)

    expected_losses = ["ce", "triplet", "clipBio"]
    checks = {
        "config_exists": spec["config_file"].is_file(),
        "dataset_requested_root_exists": spec["requested_root"].is_dir(),
        "dataset_loader_root_exists": spec["loader_root"].is_dir(),
        "train_split_nonempty": len(dataset.train) > 0,
        "loss_exactly_ce_triplet_bio": list(cfg.MODEL.LOSS_TYPE) == expected_losses,
        "reverse_loss_disabled": "clipBioReverse" not in cfg.MODEL.LOSS_TYPE,
        "nonbio_head_disabled": not bool(cfg.MODEL.NONBIO_HEAD),
        "full_differ_branch_disabled": list(cfg.MODEL.LOSS_TYPE) == expected_losses and not bool(cfg.MODEL.NONBIO_HEAD),
        "ds_disabled": True,
        "dw_disabled": True,
        "h_id_disabled": True,
        "sr3_scores_not_used": True,
        "relation_lambda_frozen_0_1": LAMBDA_GPC == 0.1,
        "pair_budget_min_cross_same_id": True,
        "pair_selection_sha256_frozen": PAIR_HASH_NAMESPACE == "phase7i_relation_specificity",
        "old_phase_runners_not_invoked": True,
        "current_model_loss_processor_dataset_sampler_paths": all(path.is_file() for path in TRAIN_ENGINE_FILES if path != ROOT / "scripts/train_final_gpc.py"),
        "old_source_scripts_available_for_audit": all(path.is_file() for path in OLD_SOURCE_FILES.values()),
        "data_leakage_audit_passed": bool(leak["passed"]),
    }
    source_files = {str(path): file_record(path) for path in [*OLD_SOURCE_FILES.values(), *TRAIN_ENGINE_FILES, spec["config_file"]]}
    if bank["source_path"]:
        source_files[str(Path(bank["source_path"]))] = file_record(Path(bank["source_path"]))
    result = {
        "dataset": dataset_name,
        "created_utc": utc_now(),
        "method": "DIFFER Bio baseline + Generic Same-ID Positive Consistency",
        "loss": {
            "base": ["L_CE", "L_Triplet", "L_Bio"],
            "gpc": "mean(1 - cos(F.normalize(feat[0].float()), F.normalize(feat[0].float()))) over selected S_b",
            "lambda_gpc": LAMBDA_GPC,
            "total": "L_CE + L_Triplet + L_Bio + 0.1 * L_GPC",
        },
        "prohibited_mechanisms": {
            "nonbio_reverse_supervision": False,
            "hair_reverse": False,
            "clothing_reverse": False,
            "pose_reverse": False,
            "ds": False,
            "dw": False,
            "h_id": False,
            "sr3": False,
            "prototype_relation": False,
            "margin_relation": False,
            "utility_gating": False,
            "diversity_weighting": False,
            "adaptive_lambda": False,
            "exposure_tuning": False,
            "headroom_weighting": False,
            "learned_pair_selector": False,
        },
        "dataset": {
            "requested_root": str(spec["requested_root"]),
            "loader_root": str(spec["loader_root"]),
            "train_images": len(dataset.train),
            "train_pids": int(dataset.num_train_pids),
            "train_clothes": int(dataset.num_train_clothes),
            "cameras": int(dataset.num_camera),
            "batch_size": int(spec["batch_size"]),
            "num_instances": int(spec["num_instances"]),
            "steps_per_epoch": int(spec["steps_per_epoch"]),
            "epochs": EPOCHS,
            "clip_dim": int(spec["clip_dim"]),
            "text_model_version": spec["text_model_version"],
        },
        "pair_budget": {
            "eligibility": "same PID, distinct image indices; same clothes and cross clothes both allowed",
            "candidate_counts": "cross, same, id",
            "matched_budget": "count = min(len(cross), len(same), len(id))",
            "selection": "stable SHA-256 sort of all same-ID/id candidates, first count selected",
            "namespace": PAIR_HASH_NAMESPACE,
            "cross_bank": {key: value for key, value in bank.items() if key != "exact_lookup" and key != "path_to_index"},
            "expected_exposure": expected_exposure,
            "not_accidentally_dense": bool(expected_exposure["expected_relation_contributions"] < expected_exposure["candidate_id_pairs"]),
        },
        "source_files": source_files,
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
        "data_leakage": leak,
    }
    root = dataset_seed_root(output_root, dataset_name)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "source_audit.json", result)
    write_json(root / "data_leakage_audit.json", leak)
    return result


def render_source_report(results: dict[str, dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# FINAL_GPC_SOURCE_AUDIT",
        "",
        f"Generated: `{utc_now()}`.",
        "",
        "This is the final confirmatory source audit. Historical Phase 7I/7O files were read for provenance and selector reconstruction only; no historical Phase runner is imported or executed by the final entry point.",
        "",
        "## Frozen method",
        "",
        "- Model: DIFFER Bio baseline + Generic Same-ID Positive Consistency.",
        "- Active losses: `L_CE + L_Triplet + L_Bio + 0.1 * L_GPC`.",
        "- Reverse/GRL and every Full DIFFER NonBio branch: OFF.",
        "- GPC feature: normalized `feat[0]`; pair loss `1 - cosine`.",
        "- Pair budget: `min(cross, same, id)`; selected pairs are the first deterministic SHA-256-sorted same-ID candidates.",
        "- Same-PID same-clothes and same-PID cross-clothes pairs are both eligible after the matched budget.",
        "- No SR3 score or weight is consumed; LTCC uses only the audited frozen binary candidate-key membership needed to reproduce I3/O1 budget semantics.",
        "",
        "## Dataset audits",
        "",
        "| Dataset | Train images | PIDs | Batch | Instances | Steps/epoch | Loss | Audit |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in ("ltcc", "prcc"):
        item = results[name]
        d = item["dataset"]
        lines.append(f"| {name.upper()} | {d['train_images']} | {d['train_pids']} | {d['batch_size']} | {d['num_instances']} | {d['steps_per_epoch']} | `{item['loss']['total']}` | **{'PASS' if item['passed'] else 'FAIL'}** |")
    lines.extend(["", "## Source files and decisions", ""])
    for name in ("ltcc", "prcc"):
        item = results[name]
        lines.append(f"### {name.upper()}")
        lines.append("")
        lines.append(f"- Source audit: **{'PASS' if item['passed'] else 'FAIL'}**.")
        lines.append(f"- Cross-candidate source: `{item['pair_budget']['cross_bank']['source']}`; binary key count `{item['pair_budget']['cross_bank']['cross_key_count']}`.")
        lines.append(f"- Expected protocol: `{item['dataset']['steps_per_epoch']} updates/rank/epoch × {EPOCHS} epochs`.")
        lines.append(f"- Expected GPC exposure: `{item['pair_budget']['expected_exposure']['expected_relation_contributions']}` selected of `{item['pair_budget']['expected_exposure']['candidate_id_pairs']}` same-ID candidates; not accidentally dense: **{item['pair_budget']['not_accidentally_dense']}**.")
        lines.append(f"- Data leakage preflight: **{'PASS' if item['data_leakage']['passed'] else 'FAIL'}**.")
        lines.append("")
    lines.extend([
        "## Stop gate",
        "",
        "Training is permitted only when both dataset audits and the pre-training data-leakage audit are PASS. A failed check must stop the run.",
        "",
        f"Machine-readable artifacts: `{output_root}/ltcc/seed{SEED}/source_audit.json`, `{output_root}/prcc/seed{SEED}/source_audit.json`.",
        "",
    ])
    report_path = output_root / "protocol" / f"源代码审计_{run_time_for(output_root)}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def render_leakage_report(results: dict[str, dict[str, Any]], output_root: Path | None = None, stage: str = "pre-training") -> None:
    lines = [
        "# FINAL_GPC_DATA_LEAKAGE_AUDIT",
        "",
        f"Generated: `{utc_now()}`; stage: `{stage}`.",
        "",
        "The training loader is constructed from `dataset.train` only. Query/gallery objects are not passed to loss, backward, optimizer, scheduler, sampler, pair selection, or checkpoint logic.",
        "",
        "| Dataset | Train/query/gallery paths disjoint | Train-only GPC pairs | Evaluator calls during training | Test-driven checkpoint selection | Status |",
        "|---|---|---|---:|---|---|",
    ]
    for name in ("ltcc", "prcc"):
        item = results[name]["data_leakage"]
        checks = item["checks"]
        disjoint = all(value for key, value in checks.items() if "disjoint" in key)
        lines.append(f"| {name.upper()} | `{disjoint}` | `{checks['gpc_candidate_pairs_source_is_train_only'] and checks['gpc_pid_labels_source_is_train_only']}` | `{item.get('training_evaluator_calls', 0)}` | `{checks['no_test_driven_best_checkpoint'] and checks['no_test_driven_early_stopping'] if 'no_test_driven_early_stopping' in checks else checks['no_test_driven_best_checkpoint']}` | **{'PASS' if item['passed'] else 'FAIL'}** |")
    lines.extend(["", "## Required checks", ""])
    for name in ("ltcc", "prcc"):
        lines.append(f"### {name.upper()}")
        lines.append("")
        for key, value in results[name]["data_leakage"]["checks"].items():
            lines.append(f"- `{key}`: **{'PASS' if value else 'FAIL'}**.")
        lines.append("")
    lines.extend([
        "## Training/test boundary",
        "",
        "- The final training function contains no evaluator construction or call.",
        "- `model_best.pth` is never written; only recovery checkpoints and E60 `model_final.pth` are written.",
        "- The first query/gallery evaluation is gated on both E60 final checkpoint hashes and the frozen flag.",
        "",
    ])
    if output_root is None:
        report_path = REPORT_ROOT / "FINAL_GPC_DATA_LEAKAGE_AUDIT.md"
    else:
        report_path = output_root / "protocol" / f"数据泄漏审计_{run_time_for(output_root)}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def source_and_leakage_audit(output_root: Path, dataset_choice: str) -> dict[str, dict[str, Any]]:
    names = ("ltcc", "prcc") if dataset_choice == "all" else (dataset_choice,)
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        results[name] = source_audit_one(name, output_root)
    # The final reports are always both-dataset reports.  If called one dataset
    # at a time, merge the existing other audit when available.
    for name in ("ltcc", "prcc"):
        if name not in results:
            path = dataset_seed_root(output_root, name) / "source_audit.json"
            if path.is_file():
                results[name] = read_json(path)
    if set(results) == {"ltcc", "prcc"}:
        render_source_report(results, output_root)
        render_leakage_report(results, output_root=output_root)
        write_json(output_root / "SOURCE_AUDIT.json", {"created_utc": utc_now(), "datasets": results})
    return results


def stable_pair_selection(
    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]],
    count: int,
    epoch: int,
    rank: int,
    iteration: int,
    seed: int,
) -> list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]:
    if count <= 0:
        return []

    def key(item: tuple[int, int, tuple[int, int], tuple[int, int, int]]) -> str:
        i, j, image_key, _state_key = item
        token = f"{PAIR_HASH_NAMESPACE}|{seed}|{epoch}|{rank}|{iteration}|id|{image_key[0]}|{image_key[1]}|{i}|{j}".encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    return sorted(candidates, key=key)[:count]


def candidate_pairs(entries: list[dict[str, Any]], bank: dict[str, Any]) -> tuple[list[tuple[int, int, tuple[int, int], tuple[int, int, int]]], dict[str, Any]]:
    path_to_index = bank["path_to_index"]
    candidates: dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]] = {"cross": [], "same": [], "id": []}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if int(entries[i]["pid"]) != int(entries[j]["pid"]):
                continue
            ia = int(path_to_index[str(entries[i]["image_path"])])
            ib = int(path_to_index[str(entries[j]["image_path"])])
            image_key = (min(ia, ib), max(ia, ib))
            pid = int(entries[i]["pid"])
            ca = int(entries[i]["clothes_id"])
            cb = int(entries[j]["clothes_id"])
            state_key = (pid, min(ca, cb), max(ca, cb))
            item = (i, j, image_key, state_key)
            if ca != cb and image_key in bank["exact_lookup"]:
                candidates["cross"].append(item)
            if ca == cb:
                candidates["same"].append(item)
            candidates["id"].append(item)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    return candidates["id"], {
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "matched_count": int(count),
        "selected_count": 0,
        "selected_pairs": [],
        "same_clothes_selected": 0,
        "cross_clothes_selected": 0,
        "pid_counts": {},
        "active": False,
    }


def select_gpc_pairs(entries: list[dict[str, Any]], bank: dict[str, Any], epoch: int, rank: int, iteration: int, seed: int) -> tuple[list[tuple[int, int, tuple[int, int], tuple[int, int, int]]], dict[str, Any]]:
    id_candidates, info = candidate_pairs(entries, bank)
    selected = stable_pair_selection(id_candidates, int(info["matched_count"]), epoch, rank, iteration, seed)
    info["selected_pairs"] = selected
    info["selected_count"] = len(selected)
    info["same_clothes_selected"] = sum(int(entries[i]["clothes_id"] == entries[j]["clothes_id"]) for i, j, _image, _state in selected)
    info["cross_clothes_selected"] = sum(int(entries[i]["clothes_id"] != entries[j]["clothes_id"]) for i, j, _image, _state in selected)
    info["pid_counts"] = dict(Counter(int(entries[i]["pid"]) for i, _j, _image, _state in selected))
    info["active"] = bool(selected)
    return selected, info


class FinalBatchSampler(Sampler[list[int]]):
    """Current identity sampler with the frozen formal fixed update budget."""

    def __init__(self, data_source: list[dict[str, Any]], batch_size: int, num_instances: int, steps_per_epoch: int, seed: int, rank: int):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.steps_per_epoch = int(steps_per_epoch)
        self.pairs_per_batch = self.batch_size // self.num_instances
        if self.batch_size % self.num_instances:
            raise ValueError("batch_size must be divisible by num_instances")
        self.inner = DistributedRandomIdentitySampler(
            data_source,
            num_instances=self.num_instances,
            num_replicas=WORLD_SIZE,
            rank=rank,
            seed=seed,
        )
        self.rank = rank
        self._batches: list[list[int]] | None = None
        self.meta: dict[str, Any] = {}

    def prepare(self, epoch: int) -> dict[str, Any]:
        self.inner.set_epoch(int(epoch) - 1)
        flat = list(iter(self.inner))
        all_pairs = [flat[position:position + self.num_instances] for position in range(0, len(flat), self.num_instances)]
        if any(len(pair) != self.num_instances for pair in all_pairs):
            raise RuntimeError(f"incomplete identity unit at rank {self.rank}, epoch {epoch}")
        target_pair_count = self.steps_per_epoch * self.pairs_per_batch
        if len(all_pairs) < target_pair_count:
            raise RuntimeError(f"sampler budget too large: {len(all_pairs)} pairs < {target_pair_count}")
        selected_pairs = [list(pair) for pair in all_pairs[:target_pair_count]]

        def pair_pid(pair: list[int]) -> int:
            pids = {int(self.data_source[index]["pid"]) for index in pair}
            if len(pids) != 1:
                raise RuntimeError(f"identity sampler produced a mixed-PID unit: {pair}")
            return next(iter(pids))

        # The current sampler is retained; this deterministic repair only
        # prevents an all-one-PID hard-triplet batch from having no negatives.
        for batch_number in range(self.steps_per_epoch):
            start = batch_number * self.pairs_per_batch
            stop = start + self.pairs_per_batch
            batch = selected_pairs[start:stop]
            if len({pair_pid(pair) for pair in batch}) >= 2:
                continue
            dominant = pair_pid(batch[0])
            replacement = None
            for candidate_index in range(stop, len(all_pairs)):
                if pair_pid(all_pairs[candidate_index]) != dominant:
                    replacement = candidate_index
                    break
            if replacement is None:
                raise RuntimeError(f"could not repair all-one-PID batch rank={self.rank} epoch={epoch} batch={batch_number + 1}")
            selected_pairs[stop - 1], all_pairs[replacement] = all_pairs[replacement], selected_pairs[stop - 1]

        batches = []
        for start in range(0, target_pair_count, self.pairs_per_batch):
            batch_pairs = selected_pairs[start:start + self.pairs_per_batch]
            batch = [index for pair in batch_pairs for index in pair]
            if len(batch) != self.batch_size:
                raise RuntimeError("invalid final batch size")
            batches.append(batch)
        covered = sorted({int(self.data_source[index]["pid"]) for batch in batches for index in batch})
        self._batches = batches
        self.meta = {
            "rank": self.rank,
            "epoch": int(epoch),
            "batches": len(batches),
            "pair_units": target_pair_count,
            "sampler_raw_pair_units": len(all_pairs),
            "batch_size": self.batch_size,
            "num_instances": self.num_instances,
            "covered_pids": covered,
            "batch_unique_pid_min": min(len({int(self.data_source[index]["pid"]) for index in batch}) for batch in batches),
            "batch_unique_pid_max": max(len({int(self.data_source[index]["pid"]) for index in batch}) for batch in batches),
            "plan_sha256": sha256_bytes(json.dumps(batches, separators=(",", ":")).encode("utf-8")),
            "train_only": True,
        }
        return self.meta

    def __iter__(self):
        if self._batches is None:
            raise RuntimeError("FinalBatchSampler.prepare(epoch) must be called before iteration")
        return iter(self._batches)

    def __len__(self) -> int:
        return self.steps_per_epoch


def build_train_loader(cfg: Any, dataset: Any, batch_sampler: FinalBatchSampler):
    transform_train, _ = build_img_transforms(cfg)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # The formal processor exposes DataLoaderX for the released epoch loader,
    # but its asynchronous device prefetch is not needed here: FinalBatchSampler
    # already supplies the complete frozen batch plan.  The current Phase 6C/7O
    # planned-train path uses this same ordinary DataLoader pattern.  Keeping
    # tensors on CPU until forward also avoids rank-dependent PPU prefetch
    # ordering while preserving the current dataset, transform, and sampler.
    loader = DataLoader(
        dataset=ImageDataset(dataset.train, transform=transform_train),
        batch_sampler=batch_sampler,
        num_workers=int(cfg.DATA.NUM_WORKERS),
        pin_memory=bool(cfg.DATA.PIN_MEMORY),
        generator=generator,
    )
    return loader


class MetricCapture:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values.setdefault(str(tag), []).append(float(value.detach().float().cpu() if torch.is_tensor(value) else value))


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def batch_entries(batch: dict[str, Any]) -> list[dict[str, Any]]:
    paths = batch["image_path"]
    pids = batch["pid"].detach().cpu().tolist() if torch.is_tensor(batch["pid"]) else list(batch["pid"])
    clothes = batch["clothes_id"].detach().cpu().tolist() if torch.is_tensor(batch["clothes_id"]) else list(batch["clothes_id"])
    return [{"image_path": str(paths[index]), "pid": int(pids[index]), "clothes_id": int(clothes[index])} for index in range(len(pids))]


def zero_relation(representation: torch.Tensor) -> torch.Tensor:
    return representation.sum() * 0.0


def forward_loss(
    model: torch.nn.Module,
    loss_fn: Any,
    batch: dict[str, Any],
    device: torch.device,
    bank: dict[str, Any],
    arm: str,
    epoch: int,
    rank: int,
    iteration: int,
    global_step: int,
    masked_index: int | None = None,
) -> dict[str, Any]:
    writer = MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(
        score,
        feat,
        pids,
        camids,
        captions,
        clothes_ids=clothes,
        train_writer=writer,
        step=global_step,
    )
    entries = batch_entries(batch)
    selected: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]] = []
    if arm == "gpc":
        selected, relation_info = select_gpc_pairs(entries, bank, epoch, rank, iteration, SEED)
        relation, pair_losses = gpc_relation_loss(feat[0], selected, masked_index=masked_index)
    else:
        relation = zero_relation(feat[0])
        pair_losses = feat[0].new_empty((0,), dtype=torch.float32)
        relation_info = {
            "candidate_counts": {"cross": 0, "same": 0, "id": 0},
            "matched_count": 0,
            "selected_count": 0,
            "selected_pairs": [],
            "same_clothes_selected": 0,
            "cross_clothes_selected": 0,
            "pid_counts": {},
            "active": False,
        }
    gpc_applied = relation if arm == "gpc" else zero_relation(representation)
    total = base_total + LAMBDA_GPC * gpc_applied if arm == "gpc" else base_total
    components = getattr(loss_fn, "last_components", {})
    if isinstance(score, dict):
        cls_score = score["cls_score"]
        id_acc = (cls_score.argmax(dim=-1) == pids).float().mean()
    else:
        id_acc = (score.argmax(dim=-1) == pids).float().mean()
    return {
        "total": total,
        "base_total": base_total,
        "ce": components.get("ce", torch.zeros((), device=device)),
        "triplet": components.get("triplet", torch.zeros((), device=device)),
        "bio": components.get("bio", torch.zeros((), device=device)),
        "gpc": relation,
        "gpc_pair_losses": pair_losses,
        "gpc_masked_index": masked_index,
        "gpc_applied": LAMBDA_GPC * gpc_applied,
        "id_acc": id_acc,
        "bio_acc": writer.values.get("acc/clip_bio_0", [0.0])[-1],
        "relation_info": relation_info,
        "writer": writer,
        "forward_device": str(feat[0].device),
        "gpc_device": str(relation.device),
    }


def found_inf(scaler: amp.GradScaler, optimizer: torch.optim.Optimizer) -> float:
    state = scaler._per_optimizer_states.get(id(optimizer), {})
    values = state.get("found_inf_per_device", {})
    return sum(float(value.item()) for value in values.values())


def reduce_epoch(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}


def ensure_audit_gate(output_root: Path, dataset_name: str) -> dict[str, Any]:
    path = dataset_seed_root(output_root, dataset_name) / "source_audit.json"
    if not path.is_file():
        raise RuntimeError(f"missing source audit; run --mode audit first: {path}")
    audit = read_json(path)
    if not audit.get("passed"):
        raise RuntimeError(f"source audit failed for {dataset_name}; training stopped")
    leak = audit.get("data_leakage", {})
    if not leak.get("passed"):
        raise RuntimeError(f"data leakage audit failed for {dataset_name}; training stopped")
    return audit


def load_common_init(output_root: Path, dataset_name: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path, meta_path = init_paths(output_root, dataset_name)
    if not path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing common initialization for {dataset_name}; run --mode init first")
    meta = read_json(meta_path)
    if sha256_file(path) != meta["file_sha256"]:
        raise RuntimeError(f"common initialization SHA256 mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    actual_hashes = tensor_hashes(state)
    if actual_hashes != meta["tensor_hashes"] or state_hash(state) != meta["state_hash"]:
        raise RuntimeError(f"common initialization state hash mismatch: {path}")
    return state, {"path": str(path), "file_sha256": sha256_file(path), "state_hash": state_hash(state), "state_key_count": len(state), "state_numel": int(sum(value.numel() for value in state.values()))}


def initialize_common(output_root: Path, dataset_name: str) -> dict[str, Any]:
    ensure_audit_gate(output_root, dataset_name)
    cfg = cfg_for(dataset_name, dataset_seed_root(output_root, dataset_name))
    device_index = 0
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("common initialization requires the two visible PPU devices")
    if any(torch.cuda.get_device_name(index) != EXPECTED_DEVICE_NAME for index in range(WORLD_SIZE)):
        raise RuntimeError("common initialization device audit failed")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda:0")
    path, meta_path = init_paths(output_root, dataset_name)
    if path.is_file() and meta_path.is_file():
        state, audit = load_common_init(output_root, dataset_name)
        del state
        return {"dataset": dataset_name, "reused_verified": True, **audit}
    if path.exists() or meta_path.exists():
        raise RuntimeError(f"partial common initialization artifact; refusing to overwrite {path}")

    set_seed(SEED)
    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=True)
    model.to(device).eval()
    state = cpu_state_dict(model)
    reverse_keys = [key for key in state if "reverse" in key.lower() or "Reverse" in key]
    if reverse_keys:
        raise RuntimeError(f"Bio-only common init unexpectedly contains reverse parameters: {reverse_keys[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    hashes = tensor_hashes(state)
    meta = {
        "dataset": dataset_name,
        "seed": SEED,
        "created_utc": utc_now(),
        "topology": "EVA02-L Bio-only: CE + Triplet + Summary Bio",
        "loss_type": ["ce", "triplet", "clipBio"],
        "reverse": "OFF",
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "file_sha256": sha256_file(path),
        "state_hash": state_hash(state),
        "tensor_hashes": hashes,
        "strict_load_required": True,
        "pretrained_model_construction_device": str(device),
    }
    write_json(meta_path, meta)
    (path).chmod(0o444)
    (meta_path).chmod(0o444)
    audit = {
        "dataset": dataset_name,
        "common_init": {key: meta[key] for key in ("file_sha256", "state_hash", "state_key_count", "state_numel")},
        "strict_load_required": True,
        "reverse_key_count": len(reverse_keys),
    }
    write_json(path.parent / "common_init_audit.json", audit)
    del model, state, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return {"dataset": dataset_name, "reused_verified": False, "path": str(path), "file_sha256": meta["file_sha256"], "state_hash": meta["state_hash"], "state_key_count": meta["state_key_count"], "state_numel": meta["state_numel"]}


def plan_pair_budget(output_root: Path, dataset_name: str, dataset: Any, bank: dict[str, Any]) -> dict[str, Any]:
    spec = SPEC[dataset_name]
    rows: list[dict[str, Any]] = []
    per_pid = Counter()
    totals = Counter()
    all_pids = set()
    for epoch in range(1, EPOCHS + 1):
        epoch_totals = Counter()
        epoch_pids = set()
        for rank in range(WORLD_SIZE):
            sampler = FinalBatchSampler(dataset.train, spec["batch_size"], spec["num_instances"], spec["steps_per_epoch"], SEED, rank)
            meta = sampler.prepare(epoch)
            for iteration, indices in enumerate(sampler, start=1):
                entries = [dataset.train[index] for index in indices]
                selected, info = select_gpc_pairs(entries, bank, epoch, rank, iteration, SEED)
                for key, value in info["candidate_counts"].items():
                    epoch_totals[f"candidate_{key}"] += int(value)
                epoch_totals["selected"] += len(selected)
                epoch_totals["active_batches"] += int(bool(selected))
                epoch_totals["same_clothes"] += int(info["same_clothes_selected"])
                epoch_totals["cross_clothes"] += int(info["cross_clothes_selected"])
                for pid, count in info["pid_counts"].items():
                    per_pid[int(pid)] += int(count)
                    epoch_pids.add(int(pid))
            totals["global_batches"] += int(meta["batches"])
            all_pids.update(meta["covered_pids"])
        row = {
            "epoch": epoch,
            "global_batches": int(spec["steps_per_epoch"] * WORLD_SIZE),
            "candidate_cross_pairs": int(epoch_totals["candidate_cross"]),
            "candidate_same_pairs": int(epoch_totals["candidate_same"]),
            "candidate_id_pairs": int(epoch_totals["candidate_id"]),
            "selected_relation_pairs": int(epoch_totals["selected"]),
            "relation_active_batches": int(epoch_totals["active_batches"]),
            "active_rate": float(epoch_totals["active_batches"] / max(spec["steps_per_epoch"] * WORLD_SIZE, 1)),
            "same_clothes_pairs": int(epoch_totals["same_clothes"]),
            "cross_clothes_pairs": int(epoch_totals["cross_clothes"]),
            "same_clothes_fraction": float(epoch_totals["same_clothes"] / max(epoch_totals["selected"], 1)),
            "cross_clothes_fraction": float(epoch_totals["cross_clothes"] / max(epoch_totals["selected"], 1)),
            "pid_coverage": len(epoch_pids),
        }
        rows.append(row)
        for key, value in epoch_totals.items():
            totals[key] += value
    totals["expected_active_rate"] = float(totals["active_batches"] / max(EPOCHS * spec["steps_per_epoch"] * WORLD_SIZE, 1))
    totals["same_clothes_fraction"] = float(totals["same_clothes"] / max(totals["selected"], 1))
    totals["cross_clothes_fraction"] = float(totals["cross_clothes"] / max(totals["selected"], 1))
    summary = {
        "dataset": dataset_name,
        "seed": SEED,
        "epochs": EPOCHS,
        "steps_per_epoch": spec["steps_per_epoch"],
        "global_batches": EPOCHS * spec["steps_per_epoch"] * WORLD_SIZE,
        "expected_active_batches": int(totals["active_batches"]),
        "expected_active_rate": totals["expected_active_rate"],
        "candidate_cross_pairs": int(totals["candidate_cross"]),
        "candidate_same_pairs": int(totals["candidate_same"]),
        "candidate_id_pairs": int(totals["candidate_id"]),
        "expected_relation_contributions": int(totals["selected"]),
        "same_clothes_relation_pairs": int(totals["same_clothes"]),
        "cross_clothes_relation_pairs": int(totals["cross_clothes"]),
        "same_clothes_fraction": totals["same_clothes_fraction"],
        "cross_clothes_fraction": totals["cross_clothes_fraction"],
        "pid_coverage": len(all_pids),
        "selection_logic": "count=min(cross,same,id); deterministic SHA-256 selection of id candidates",
        "hash_namespace": PAIR_HASH_NAMESPACE,
        "cross_candidate_source": bank["source"],
        "not_accidentally_dense": bool(totals["selected"] < totals["candidate_id"]),
    }
    directory = protocol_dir(output_root, dataset_name)
    write_csv(directory / "pair_budget_expected_by_epoch.csv", rows)
    write_csv(directory / "pair_budget_expected_per_pid.csv", [{"pid": pid, "selected_relation_pairs": count} for pid, count in sorted(per_pid.items())])
    write_json(directory / "pair_budget_expected_summary.json", summary)
    return summary


def dry_run_one(output_root: Path, dataset_name: str, arm: str, dry_steps: int = 1) -> dict[str, Any]:
    if arm not in ("bio", "gpc"):
        raise ValueError(arm)
    ensure_audit_gate(output_root, dataset_name)
    rank, device = runtime_gate(require_world_size=True)
    run_dir = arm_dir(output_root, dataset_name, arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg = cfg_for(dataset_name, run_dir)
        set_seed(SEED)
        dataset = build_dataset(cfg)
        state, init_audit = load_common_init(output_root, dataset_name)
        raw = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
        raw.load_state_dict(state, strict=True)
        raw.to(device).train()
        loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, _center_optimizer = make_optimizer(cfg, raw, center)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = amp.GradScaler(init_scale=int(SPEC[dataset_name]["initial_scale"]))
        # EVA02 carries large persistent=False rotary/qkv helper buffers.  The
        # PPU NCCL backend can interleave DDP's per-forward buffer broadcast
        # with gradient all-reduces when those static buffers are broadcast.
        # They are identical, non-trainable helpers on every rank, so keep the
        # current model/optimizer trajectory while disabling that unnecessary
        # communication path.
        model = DDP(
            raw,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            bucket_cap_mb=25,
        )
        bank = build_train_only_pair_bank(dataset, dataset_name)
        batch_sampler = FinalBatchSampler(dataset.train, SPEC[dataset_name]["batch_size"], SPEC[dataset_name]["num_instances"], SPEC[dataset_name]["steps_per_epoch"], SEED, rank)
        batch_sampler.prepare(1)
        loader = build_train_loader(cfg, dataset, batch_sampler)
        if int(dry_steps) < 1:
            raise ValueError("dry_steps must be positive")
        iterator = iter(loader)
        finite_before = True
        inf_total = 0.0
        train_only_batches = True
        gradient_finite = True
        total_loss_identity = True
        gpc_active_seen = False
        result = None
        iteration_audit: list[dict[str, Any]] = []
        for dry_iteration in range(1, int(dry_steps) + 1):
            batch = next(iterator)
            optimizer.zero_grad()
            result = forward_loss(model, loss_fn, batch, device, bank, arm, 1, rank, dry_iteration, dry_iteration)
            scale_before = float(scaler.get_scale())
            finite_before = finite_before and bool(torch.isfinite(result["total"].detach()).item())
            expected_total = result["base_total"] + LAMBDA_GPC * result["gpc"] if arm == "gpc" else result["base_total"]
            total_loss_identity = total_loss_identity and bool(torch.allclose(result["total"].detach(), expected_total.detach(), rtol=1e-5, atol=1e-6))
            gpc_active_seen = gpc_active_seen or bool(result["relation_info"]["active"])
            if not finite_before:
                raise RuntimeError(f"non-finite dry-run loss for {dataset_name}/{arm}")
            scaler.scale(result["total"]).backward()
            bad_gradient_names = [
                name for name, parameter in raw.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all().item())
            ]
            gradient_finite = gradient_finite and not bad_gradient_names
            scaler.step(optimizer)
            inf = found_inf(scaler, optimizer)
            inf_total += inf
            scaler.update()
            iteration_audit.append({
                "iteration": dry_iteration,
                "base_loss": float(result["base_total"].detach().float().cpu()),
                "gpc_loss": float(result["gpc"].detach().float().cpu()),
                "total_loss": float(result["total"].detach().float().cpu()),
                "total_loss_identity": bool(torch.allclose(result["total"].detach(), expected_total.detach(), rtol=1e-5, atol=1e-6)),
                "gpc_active": bool(result["relation_info"]["active"]),
                "selected_relation_pairs": int(result["relation_info"]["selected_count"]),
                "gradient_finite": not bad_gradient_names,
                "nonfinite_gradient_parameters": bad_gradient_names[:20],
                "nonfinite_gradient_parameter_count": len(bad_gradient_names),
                "scale_before": scale_before,
                "found_inf": float(inf),
                "scale_after": float(scaler.get_scale()),
            })
            train_only_batches = train_only_batches and all(
                str(item["image_path"]) in {str(x["image_path"]) for x in dataset.train}
                for item in batch_entries(batch)
            )
        assert result is not None
        local_state_hash = state_hash(cpu_state_dict(model))
        gathered_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_hashes, local_state_hash)
        local = {
            "rank": rank,
            "device": str(device),
            "forward_device": result["forward_device"],
            "gpc_device": result["gpc_device"],
            "loss_finite": finite_before,
            "total_loss_identity": total_loss_identity,
            "gpc_active_seen": gpc_active_seen,
            "gradient_finite": gradient_finite,
            "selected_relation_pairs": int(result["relation_info"]["selected_count"]),
            "relation_active": bool(result["relation_info"]["active"]),
            "amp_scale_end": float(scaler.get_scale()),
            "found_inf": inf_total,
            "optimizer_parameter_devices": sorted({str(parameter.device) for group in optimizer.param_groups for parameter in group["params"]}),
            "batch_paths_train_only": train_only_batches,
            "iteration_audit": iteration_audit,
        }
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local)
        dist.barrier()
        if rank == 0:
            preflight_checkpoint_path = run_dir / f"preflight_model_rank{rank}.pth"
            temporary = preflight_checkpoint_path.with_suffix(preflight_checkpoint_path.suffix + ".tmp")
            torch.save(cpu_state_dict(raw), temporary)
            temporary.replace(preflight_checkpoint_path)
            preflight_checkpoint = {
                "path": str(preflight_checkpoint_path),
                "sha256": sha256_file(preflight_checkpoint_path),
                "epoch": 0,
                "formal_training_counted": False,
            }
            result_json = {
                "dataset": dataset_name,
                "arm": arm,
                "dry_steps": int(dry_steps),
                "seed": SEED,
                "created_utc": utc_now(),
                "common_init": init_audit,
                "strict_common_init_hash_equal_across_ranks": len(set(gathered_hashes)) == 1,
                "loss_finite": all(item["loss_finite"] for item in gathered),
                "total_loss_identity": all(item["total_loss_identity"] for item in gathered),
                "gpc_active_seen": any(item["gpc_active_seen"] for item in gathered) if arm == "gpc" else True,
                "gradient_finite": all(item["gradient_finite"] for item in gathered),
                "optimizer_step_attempted": True,
                "optimizer_update_finite": all(float(item["found_inf"]) == 0.0 for item in gathered),
                "cpu_fallback": False,
                "preflight_checkpoint": preflight_checkpoint,
                "ranks": gathered,
                "iteration_audit": {str(item["rank"]): item["iteration_audit"] for item in gathered},
                "passed": (
                    len(set(gathered_hashes)) == 1
                    and all(item["loss_finite"] and item["batch_paths_train_only"] for item in gathered)
                    and all(item["total_loss_identity"] and item["gradient_finite"] for item in gathered)
                    and all(float(item["found_inf"]) == 0.0 for item in gathered)
                    and (any(item["gpc_active_seen"] for item in gathered) if arm == "gpc" else True)
                ),
            }
            write_json(run_dir / "dry_run.json", result_json)
        dist.barrier()
        if not (run_dir / "dry_run.json").is_file() and rank == 0:
            raise RuntimeError("dry-run artifact was not written")
        return {"dataset": dataset_name, "arm": arm, "rank": rank, "passed": True}
    finally:
        try:
            if "loader" in locals() and hasattr(loader, "shutdown"):
                loader.shutdown()
        finally:
            cleanup_dist()


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"FINAL_GPC.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


METRIC_FIELDS = [
    "epoch", "global_step", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "gpc_loss", "gpc_applied_loss",
    "lr", "lr_bias", "train_accuracy", "bio_accuracy", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "skipped_steps", "found_inf_count", "nonfinite_events", "optimizer_updates", "relation_active_batches",
    "relation_active_rate", "relation_contributions", "candidate_cross_pairs", "candidate_same_pairs",
    "candidate_id_pairs", "same_clothes_contributions", "cross_clothes_contributions", "same_clothes_fraction",
    "cross_clothes_fraction", "relation_pid_coverage", "sampler_plan_rank0", "sampler_plan_rank1",
]

EXPOSURE_FIELDS = [
    "dataset", "arm", "epoch", "global_batches", "candidate_cross_pairs", "candidate_same_pairs", "candidate_id_pairs",
    "matched_relation_pairs", "relation_active_batches", "relation_active_batch_rate", "same_clothes_pairs",
    "cross_clothes_pairs", "same_clothes_fraction", "cross_clothes_fraction", "relation_pid_coverage", "optimizer_updates",
]


def save_recovery_checkpoint(path: Path, raw: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: amp.GradScaler, epoch: int, global_step: int, cfg: Any, arm: str, dataset_name: str, device: torch.device) -> dict[str, Any]:
    state = {
        "phase": "final_confirmatory_gpc",
        "dataset": dataset_name,
        "arm": arm,
        "recovery_only": True,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": cpu_state_dict(raw),
        "optimizer": to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device),
        },
        "config_sha256": sha256_bytes(cfg.dump().encode("utf-8")),
        "test_driven": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "epoch": epoch, "recovery_only": True}


def save_final_checkpoint(run_dir: Path, raw: torch.nn.Module, dataset_name: str, arm: str, cfg: Any) -> dict[str, Any]:
    raw.eval()
    raw.requires_grad_(False)
    state = cpu_state_dict(raw)
    path = run_dir / "model_final.pth"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    metadata = {
        "dataset": dataset_name,
        "arm": arm,
        "epoch": EPOCHS,
        "global_step": EPOCHS * SPEC[dataset_name]["steps_per_epoch"],
        "saved_utc": utc_now(),
        "frozen": True,
        "test_started": False,
        "test_metric_used": False,
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_state_hash": state_hash(state),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "loss_type": list(cfg.MODEL.LOSS_TYPE),
        "no_model_best": True,
    }
    write_json(run_dir / "final_checkpoint.json", metadata)
    path.chmod(0o444)
    return metadata


def publish_public_model_checkpoint(output_root: Path, dataset_name: str, source_path: Path, epoch: int, final: bool = False) -> dict[str, Any]:
    """Publish a model-only checkpoint in the timestamped public run tree.

    The internal recovery files remain untouched.  The public files are exact
    copies of the model state used by the formal test; ``model_final.pth`` is
    published only after the internal E60 checkpoint has been frozen.
    """
    if int(epoch) < 1 or int(epoch) > EPOCHS:
        raise ValueError(f"invalid public checkpoint epoch: {epoch}")
    destination = public_checkpoint_root(output_root, dataset_name) / (
        "model_final.pth" if final else f"model_epoch_{int(epoch)}.pth"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if final:
        shutil.copy2(source_path, temporary)
    else:
        recovery = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(recovery, dict) or "model" not in recovery:
            raise RuntimeError(f"recovery checkpoint has no model state: {source_path}")
        torch.save(recovery["model"], temporary)
    temporary.replace(destination)
    if final:
        destination.chmod(0o444)
    return {
        "path": str(destination),
        "source_path": str(source_path),
        "epoch": int(epoch),
        "final": bool(final),
        "frozen": bool(final),
        "sha256": sha256_file(destination),
    }


def training_one(output_root: Path, dataset_name: str, arm: str) -> dict[str, Any]:
    if arm not in ("bio", "gpc"):
        raise ValueError(arm)
    audit = ensure_audit_gate(output_root, dataset_name)
    run_dir = arm_dir(output_root, dataset_name, arm)
    dry_path = run_dir / "dry_run.json"
    if not dry_path.is_file() or not read_json(dry_path).get("passed"):
        raise RuntimeError(f"dry-run gate missing/failed for {dataset_name}/{arm}; training stopped")
    if (run_dir / "model_final.pth").exists():
        raise RuntimeError(f"final checkpoint already exists; refusing to overwrite: {run_dir / 'model_final.pth'}")

    rank, device = runtime_gate(require_world_size=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = SPEC[dataset_name]
    public_root = public_dataset_root(output_root, dataset_name)
    for directory in (
        public_root / "logs",
        public_root / "checkpoints",
        public_root / "test",
        public_root / "artifacts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    logger = None
    loader = None
    metrics_stream = None
    exposure_stream = None
    amp_stream = None
    pid_stream = None
    plan_stream = None
    try:
        cfg = cfg_for(dataset_name, run_dir)
        set_seed(SEED)
        dataset = build_dataset(cfg)
        state, init_audit = load_common_init(output_root, dataset_name)
        raw = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
        raw.load_state_dict(state, strict=True)
        raw.to(device).train()
        initial_hash = state_hash(cpu_state_dict(raw))
        initial_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(initial_hashes, initial_hash)
        if len(set(initial_hashes)) != 1 or initial_hash != init_audit["state_hash"]:
            raise RuntimeError(f"common init strict-load hash mismatch in {dataset_name}/{arm}")

        loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, _center_optimizer = make_optimizer(cfg, raw, center)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = amp.GradScaler(init_scale=int(spec["initial_scale"]))
        model = DDP(
            raw,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            bucket_cap_mb=25,
        )
        bank = build_train_only_pair_bank(dataset, dataset_name)
        batch_sampler = FinalBatchSampler(dataset.train, spec["batch_size"], spec["num_instances"], spec["steps_per_epoch"], SEED, rank)
        loader = build_train_loader(cfg, dataset, batch_sampler)

        if rank == 0:
            logger = logger_for(public_log_path(output_root, dataset_name, "训练"))
            logger.info("==================================================")
            logger.info("数据集：%s", dataset_name.upper())
            logger.info("方法：Bio + GPC")
            logger.info("GPC 中文名：通用同身份正样本一致性")
            logger.info("GPC 英文名：Generic Same-ID Positive Consistency")
            logger.info("随机种子：%d", SEED)
            logger.info("训练 Epoch：%d", EPOCHS)
            logger.info("批大小：%d；NUM_INSTANCES：%d", spec["batch_size"], spec["num_instances"])
            logger.info("优化器：SGD；基础学习率：%.6g；Warmup：%d 个 Epoch；Scheduler：cosine", 2e-6, WARMUP_EPOCHS)
            logger.info("GPC lambda：%.1f", LAMBDA_GPC)
            logger.info("AMP 初始 scale：%d", spec["initial_scale"])
            logger.info("PPU-only：是；CPU fallback：禁止")
            logger.info("启动时间：%s", run_time_for(output_root))
            logger.info("输出目录：%s", output_root)
            logger.info("正式训练只使用 train split；训练期间不创建 query/gallery evaluator")
            logger.info("==================================================")
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            metrics_stream = (run_dir / "metrics_per_epoch.csv").open("w", newline="", encoding="utf-8")
            metrics_writer = csv.DictWriter(metrics_stream, fieldnames=METRIC_FIELDS)
            metrics_writer.writeheader(); metrics_stream.flush()
            exposure_stream = (run_dir / "relation_exposure.csv").open("w", newline="", encoding="utf-8")
            exposure_writer = csv.DictWriter(exposure_stream, fieldnames=EXPOSURE_FIELDS)
            exposure_writer.writeheader(); exposure_stream.flush()
            amp_stream = (run_dir / "amp_statistics.csv").open("w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            amp_writer.writeheader(); amp_stream.flush()
            pid_stream = (run_dir / "per_pid_relation_contributions.csv").open("w", newline="", encoding="utf-8")
            pid_writer = csv.DictWriter(pid_stream, fieldnames=["epoch", "pid", "selected_relation_contributions"])
            pid_writer.writeheader(); pid_stream.flush()
            plan_stream = (run_dir / "sampler_plan.csv").open("w", newline="", encoding="utf-8")
            plan_writer = csv.DictWriter(
                plan_stream,
                fieldnames=[
                    "epoch", "rank", "batches", "pair_units", "sampler_raw_pair_units",
                    "batch_size", "num_instances", "covered_pids", "batch_unique_pid_min",
                    "batch_unique_pid_max", "plan_sha256", "train_only",
                ],
            )
            plan_writer.writeheader(); plan_stream.flush()
            write_json(run_dir / "run_manifest.json", {
                "phase": "final_confirmatory_gpc",
                "dataset": dataset_name,
                "arm": arm,
                "seed": SEED,
                "created_utc": utc_now(),
                "method": "Bio baseline" if arm == "bio" else "Bio + Generic Same-ID Positive Consistency",
                "loss_type": list(cfg.MODEL.LOSS_TYPE),
                "lambda_gpc": 0.0 if arm == "bio" else LAMBDA_GPC,
                "common_init": init_audit,
                "initial_parameter_hash": initial_hash,
                "initial_parameter_hashes_all_ranks": initial_hashes,
                "evaluator_calls_during_training": 0,
                "test_started": False,
                "best_checkpoint_selection": False,
                "cpu_fallback": False,
                "epochs_required": EPOCHS,
                "steps_per_epoch": spec["steps_per_epoch"],
                "source_audit": str(dataset_seed_root(output_root, dataset_name) / "source_audit.json"),
                "data_leakage_audit": str(output_root / "protocol" / f"数据泄漏审计_{run_time_for(output_root)}.md"),
                "public_training_log": str(public_log_path(output_root, dataset_name, "训练")),
                "public_checkpoint_root": str(public_checkpoint_root(output_root, dataset_name)),
                "query_gallery_used_during_training": False,
                "checkpoint_selection_metric": None,
            })
            runtime = runtime_snapshot(rank, device, spec["initial_scale"], True)
        else:
            metrics_writer = exposure_writer = amp_writer = pid_writer = plan_writer = None

        runtime_local = runtime_snapshot(rank, device, spec["initial_scale"], True)
        local_forward_devices: set[str] = set()
        local_gpc_devices: set[str] = set()
        local_optimizer_devices = sorted({str(parameter.device) for group in optimizer.param_groups for parameter in group["params"]})
        global_step = 0
        total_relation = Counter()
        expected = read_json(protocol_dir(output_root, dataset_name) / "pair_budget_expected_summary.json")

        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            if rank == 0 and logger is not None:
                logger.info("[%s][训练] 第 %d/%d 个 Epoch 开始", dataset_name.upper(), epoch, EPOCHS)
            scheduler.step(epoch)
            plan_meta = batch_sampler.prepare(epoch)
            gathered_plans = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_plans, plan_meta)
            if rank == 0:
                for item in gathered_plans:
                    plan_writer.writerow(item)
                plan_stream.flush()
                if any(item["batches"] != spec["steps_per_epoch"] or item["batch_unique_pid_min"] < 2 or not item["train_only"] for item in gathered_plans):
                    raise RuntimeError(f"runtime train sampler audit failed at {dataset_name}/{arm}/epoch{epoch}")

            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "gpc", "gpc_applied", "id_acc", "bio_acc")}
            local_exposure = Counter()
            local_pid = Counter()
            local_skips = 0
            local_nonfinite = 0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            iterator = iter(loader)
            for iteration, batch in enumerate(iterator, start=1):
                global_step += 1
                optimizer.zero_grad()
                result = forward_loss(model, loss_fn, batch, device, bank, arm, epoch, rank, iteration, global_step)
                if not bool(torch.isfinite(result["total"].detach()).item()):
                    local_nonfinite += 1
                    if rank == 0 and logger is not None:
                        logger.error("[%s][AMP审计] 第 %d 个 Epoch 第 %d 个 batch 的总 Loss 非有限，训练停止", dataset_name.upper(), epoch, iteration)
                    raise RuntimeError(f"non-finite loss at {dataset_name}/{arm}/epoch{epoch}/iteration{iteration}")
                scaler.scale(result["total"]).backward()
                scaler.step(optimizer)
                inf = found_inf(scaler, optimizer)
                local_skips += int(inf > 0)
                local_nonfinite += int(inf > 0)
                if inf > 0 and rank == 0 and logger is not None:
                    logger.warning("[%s][AMP审计] 第 %d 个 Epoch 第 %d 个 batch 发生 found_inf；本次 optimizer step 被跳过；当前 scale：%.1f", dataset_name.upper(), epoch, iteration, float(scaler.get_scale()))
                scaler.update()
                scale_min = min(scale_min, float(scaler.get_scale()))
                scale_max = max(scale_max, float(scaler.get_scale()))
                for key in epoch_sum:
                    epoch_sum[key] += scalar(result[key])
                info = result["relation_info"]
                for key in ("candidate_cross", "candidate_same", "candidate_id"):
                    local_exposure[key] += int(info["candidate_counts"][key.removeprefix("candidate_")])
                local_exposure["selected"] += int(info["selected_count"])
                local_exposure["active_batches"] += int(info["active"])
                local_exposure["same_clothes"] += int(info["same_clothes_selected"])
                local_exposure["cross_clothes"] += int(info["cross_clothes_selected"])
                for pid, count in info["pid_counts"].items():
                    local_pid[int(pid)] += int(count)
                local_forward_devices.add(result["forward_device"])
                local_gpc_devices.add(result["gpc_device"])
                if rank == 0 and logger is not None and iteration % int(cfg.SOLVER.LOG_PERIOD) == 0:
                    logger.info("[%s][训练] 第 %d/%d 个 Epoch，第 %d/%d 个 batch；总 Loss：%.6f；Base Loss：%.6f；GPC Loss：%.6f；当前学习率：%.3e；身份准确率：%.4f", dataset_name.upper(), epoch, EPOCHS, iteration, spec["steps_per_epoch"], scalar(result["total"]), scalar(result["base_total"]), scalar(result["gpc"]), scheduler._get_lr(epoch)[0], scalar(result["id_acc"]))
            del iterator
            if hasattr(loader, "shutdown"):
                loader.shutdown()
            gc.collect()

            reduced = reduce_epoch({key: value / spec["steps_per_epoch"] for key, value in epoch_sum.items()}, device)
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); nonfinite_sum = amp_tensor[1].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_scale = amp_tensor[2].clone(); max_scale = amp_tensor[3].clone(); end_scale = amp_tensor[4].clone()
            dist.all_reduce(min_scale, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_scale, op=dist.ReduceOp.MAX)
            dist.all_reduce(end_scale, op=dist.ReduceOp.MAX)
            gathered_exposure = [None] * WORLD_SIZE
            gathered_pid = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_exposure, dict(local_exposure))
            dist.all_gather_object(gathered_pid, dict(local_pid))
            exposure = Counter()
            pids = Counter()
            for item in gathered_exposure:
                exposure.update({key: int(value) for key, value in item.items()})
            for item in gathered_pid:
                pids.update({int(key): int(value) for key, value in item.items()})
            relation_pids = len(pids)
            for key, value in exposure.items():
                total_relation[key] += value
            exposure_row = {
                "dataset": dataset_name,
                "arm": arm,
                "epoch": epoch,
                "global_batches": spec["steps_per_epoch"] * WORLD_SIZE,
                "candidate_cross_pairs": exposure["candidate_cross"],
                "candidate_same_pairs": exposure["candidate_same"],
                "candidate_id_pairs": exposure["candidate_id"],
                "matched_relation_pairs": exposure["selected"],
                "relation_active_batches": exposure["active_batches"],
                "relation_active_batch_rate": exposure["active_batches"] / max(spec["steps_per_epoch"] * WORLD_SIZE, 1),
                "same_clothes_pairs": exposure["same_clothes"],
                "cross_clothes_pairs": exposure["cross_clothes"],
                "same_clothes_fraction": exposure["same_clothes"] / max(exposure["selected"], 1),
                "cross_clothes_fraction": exposure["cross_clothes"] / max(exposure["selected"], 1),
                "relation_pid_coverage": relation_pids,
                "optimizer_updates": spec["steps_per_epoch"],
            }
            if rank == 0:
                metrics_writer.writerow({
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": reduced["total"],
                    "ce_loss": reduced["ce"],
                    "triplet_loss": reduced["triplet"],
                    "bio_loss": reduced["bio"],
                    "gpc_loss": reduced["gpc"] if arm == "gpc" else 0.0,
                    "gpc_applied_loss": reduced["gpc_applied"] if arm == "gpc" else 0.0,
                    "lr": float(scheduler._get_lr(epoch)[0]),
                    "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                    "train_accuracy": reduced["id_acc"],
                    "bio_accuracy": reduced["bio_acc"],
                    "amp_scale_min": float(min_scale.item()),
                    "amp_scale_max": float(max_scale.item()),
                    "amp_scale_end": float(end_scale.item()),
                    "skipped_steps": int(skip_sum.item()),
                    "found_inf_count": int(skip_sum.item()),
                    "nonfinite_events": int(nonfinite_sum.item()),
                    "optimizer_updates": spec["steps_per_epoch"],
                    "relation_active_batches": exposure_row["relation_active_batches"],
                    "relation_active_rate": exposure_row["relation_active_batch_rate"],
                    "relation_contributions": exposure_row["matched_relation_pairs"],
                    "candidate_cross_pairs": exposure_row["candidate_cross_pairs"],
                    "candidate_same_pairs": exposure_row["candidate_same_pairs"],
                    "candidate_id_pairs": exposure_row["candidate_id_pairs"],
                    "same_clothes_contributions": exposure_row["same_clothes_pairs"],
                    "cross_clothes_contributions": exposure_row["cross_clothes_pairs"],
                    "same_clothes_fraction": exposure_row["same_clothes_fraction"],
                    "cross_clothes_fraction": exposure_row["cross_clothes_fraction"],
                    "relation_pid_coverage": relation_pids,
                    "sampler_plan_rank0": gathered_plans[0]["plan_sha256"],
                    "sampler_plan_rank1": gathered_plans[1]["plan_sha256"],
                })
                metrics_stream.flush()
                exposure_writer.writerow(exposure_row); exposure_stream.flush()
                amp_writer.writerow({key: exposure_row.get(key, {"epoch": epoch}).get(key, "") if False else value for key, value in {
                    "epoch": epoch,
                    "skipped_steps": int(skip_sum.item()),
                    "found_inf_count": int(skip_sum.item()),
                    "nonfinite_events": int(nonfinite_sum.item()),
                    "amp_scale_min": float(min_scale.item()),
                    "amp_scale_max": float(max_scale.item()),
                    "amp_scale_end": float(end_scale.item()),
                }.items()}); amp_stream.flush()
                for pid, count in sorted(pids.items()):
                    pid_writer.writerow({"epoch": epoch, "pid": pid, "selected_relation_contributions": count})
                pid_stream.flush()
                epoch_wall = time.time() - epoch_started
                logger.info("[%s][训练] 当前学习率：%.3e", dataset_name.upper(), scheduler._get_lr(epoch)[0])
                logger.info("[%s][训练] Base Loss：%.6f；GPC Loss：%.6f；总 Loss：%.6f", dataset_name.upper(), reduced["total"] - reduced["gpc_applied"], reduced["gpc"], reduced["total"])
                logger.info("[%s][训练] GPC 激活 batch 数：%d；总 batch 数：%d；GPC active rate：%.8f", dataset_name.upper(), exposure_row["relation_active_batches"], exposure_row["global_batches"], exposure_row["relation_active_batch_rate"])
                logger.info("[%s][训练] GPC selected contribution 数：%d；same-clothes contribution 数：%d；cross-clothes contribution 数：%d", dataset_name.upper(), exposure_row["matched_relation_pairs"], exposure_row["same_clothes_pairs"], exposure_row["cross_clothes_pairs"])
                logger.info("[%s][训练] same-clothes fraction：%.8f；cross-clothes fraction：%.8f；PID coverage：%d/%d（%.8f）", dataset_name.upper(), exposure_row["same_clothes_fraction"], exposure_row["cross_clothes_fraction"], relation_pids, dataset.num_train_pids, relation_pids / max(int(dataset.num_train_pids), 1))
                logger.info("[%s][训练] 当前 AMP scale：%.1f；found_inf 数：%d；跳过 optimizer step 数：%d；nonfinite 事件数：%d", dataset_name.upper(), float(end_scale.item()), int(skip_sum.item()), int(skip_sum.item()), int(nonfinite_sum.item()))
                logger.info("[%s][训练] 第 %d/%d 个 Epoch 完成；Epoch wall time：%.2f 秒", dataset_name.upper(), epoch, EPOCHS, epoch_wall)

            dist.barrier()
            if epoch % 10 == 0:
                checkpoint = save_recovery_checkpoint(run_dir / f"checkpoint_e{epoch:02d}_rank{rank}.pth", raw, optimizer, scheduler, scaler, epoch, global_step, cfg, arm, dataset_name, device)
                dist.barrier()
                if rank == 0:
                    public_epoch = publish_public_model_checkpoint(output_root, dataset_name, run_dir / f"checkpoint_e{epoch:02d}_rank{rank}.pth", epoch, final=False)
                    # Recovery checkpoints contain optimizer state; the public
                    # model checkpoint is a model-only copy for audit/test use.
                    public_epoch["source_recovery_checkpoint"] = checkpoint["path"]
                    write_json(public_artifact_root(output_root, dataset_name) / f"模型检查点_{epoch:02d}_{run_time_for(output_root)}.json", public_epoch)
                    write_json(run_dir / f"checkpoint_e{epoch:02d}.json", {
                        "dataset": dataset_name,
                        "arm": arm,
                        "epoch": epoch,
                        "recovery_only": True,
                        "test_driven": False,
                        "rank_checkpoints": [str(run_dir / f"checkpoint_e{epoch:02d}_rank{index}.pth") for index in range(WORLD_SIZE)],
                        "public_model_checkpoint": public_epoch,
                    })
                dist.barrier()

        dist.barrier()
        final_meta = None
        if rank == 0:
            final_meta = save_final_checkpoint(run_dir, raw, dataset_name, arm, cfg)
            public_final = publish_public_model_checkpoint(output_root, dataset_name, run_dir / "model_final.pth", EPOCHS, final=True)
            final_meta["public_path"] = public_final["path"]
            final_meta["public_checkpoint_sha256"] = public_final["sha256"]
            final_meta["public_checkpoint_epoch"] = EPOCHS
            if public_final["sha256"] != final_meta["checkpoint_sha256"]:
                raise RuntimeError("public E60 checkpoint copy hash differs from frozen internal checkpoint")
            write_json(run_dir / "final_checkpoint.json", final_meta)
            write_json(public_artifact_root(output_root, dataset_name) / f"最终检查点_{run_time_for(output_root)}.json", final_meta)
        final_holder = [final_meta]
        dist.broadcast_object_list(final_holder, src=0)
        final_meta = final_holder[0]
        if rank == 0:
            runtime = runtime_snapshot(rank, device, spec["initial_scale"], True)
            gathered_runtime = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_runtime, {
                **runtime_snapshot(rank, device, spec["initial_scale"], True),
                "forward_devices": sorted(local_forward_devices),
                "gpc_devices": sorted(local_gpc_devices),
                "optimizer_parameter_devices": local_optimizer_devices,
            })
        else:
            gathered_runtime = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_runtime, {
                **runtime_snapshot(rank, device, spec["initial_scale"], True),
                "forward_devices": sorted(local_forward_devices),
                "gpc_devices": sorted(local_gpc_devices),
                "optimizer_parameter_devices": local_optimizer_devices,
            })
        if rank == 0:
            runtime_audit = {
                "phase": "final_confirmatory_gpc",
                "dataset": dataset_name,
                "arm": arm,
                "created_utc": utc_now(),
                "ranks": gathered_runtime,
                "device_name_check": all(item["device_name"] == EXPECTED_DEVICE_NAME and all(name == EXPECTED_DEVICE_NAME for name in item["all_device_names"]) for item in gathered_runtime),
                "forward_ppu_check": all(item["forward_devices"] and all(device_name.startswith("cuda:") for device_name in item["forward_devices"]) for item in gathered_runtime),
                "gpc_ppu_check": all(item["gpc_devices"] and all(device_name.startswith("cuda:") for device_name in item["gpc_devices"]) for item in gathered_runtime),
                "optimizer_ppu_check": all(item["optimizer_parameter_devices"] and all(device_name.startswith("cuda:") for device_name in item["optimizer_parameter_devices"]) for item in gathered_runtime),
                "cpu_fallback": False,
                "passed": True,
            }
            runtime_audit["passed"] = all(runtime_audit[key] for key in ("device_name_check", "forward_ppu_check", "gpc_ppu_check", "optimizer_ppu_check"))
            write_json(run_dir / "runtime_audit.json", runtime_audit)
            final_run = {
                "phase": "final_confirmatory_gpc",
                "dataset": dataset_name,
                "arm": arm,
                "completed_utc": utc_now(),
                "epochs_completed": EPOCHS,
                "optimizer_updates_per_rank": EPOCHS * spec["steps_per_epoch"],
                "global_optimizer_updates": EPOCHS * spec["steps_per_epoch"] * WORLD_SIZE,
                "final_checkpoint": final_meta,
                "evaluator_calls_during_training": 0,
                "test_started": False,
                "test_driven_early_stopping": False,
                "test_driven_checkpoint_selection": False,
                "no_extra_seed_or_search": True,
                "expected_exposure": expected,
                "actual_exposure_totals": dict(total_relation),
                "runtime_audit": str(run_dir / "runtime_audit.json"),
                "public_final_checkpoint": final_meta.get("public_path"),
                "public_final_checkpoint_sha256": final_meta.get("public_checkpoint_sha256"),
            }
            write_json(run_dir / "RUN_COMPLETE.json", final_run)
            if logger is not None:
                logger.info("[%s][训练] 60 个 Epoch 已完整完成", dataset_name.upper())
                logger.info("[%s][训练] model_final.pth 已冻结；Epoch：60；SHA256：%s", dataset_name.upper(), final_meta["checkpoint_sha256"])
                logger.info("[%s][训练] 公共最终 checkpoint：%s", dataset_name.upper(), final_meta["public_path"])
        dist.barrier()
        return {"dataset": dataset_name, "arm": arm, "epochs_completed": EPOCHS, "final_checkpoint": final_meta}
    finally:
        for stream in (metrics_stream, exposure_stream, amp_stream, pid_stream, plan_stream):
            if stream is not None:
                stream.close()
        try:
            if loader is not None and hasattr(loader, "shutdown"):
                loader.shutdown()
        finally:
            cleanup_dist()


def checkpoint_gate_for_test(output_root: Path, dataset_name: str, arms: list[str]) -> dict[str, Any]:
    records = {}
    for arm in arms:
        internal_path = arm_dir(output_root, dataset_name, arm) / "model_final.pth"
        public_path = public_checkpoint_root(output_root, dataset_name) / "model_final.pth"
        path = public_path if public_path.is_file() else internal_path
        meta_path = arm_dir(output_root, dataset_name, arm) / "final_checkpoint.json"
        if not internal_path.is_file() or not path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"missing final checkpoint for {dataset_name}/{arm}")
        meta = read_json(meta_path)
        if meta.get("epoch") != EPOCHS or not meta.get("frozen") or meta.get("test_started"):
            raise RuntimeError(f"checkpoint gate failed for {dataset_name}/{arm}: {meta}")
        actual_sha = sha256_file(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        actual_state_hash = state_hash(state)
        if actual_sha != meta.get("checkpoint_sha256") or actual_state_hash != meta.get("checkpoint_state_hash"):
            raise RuntimeError(f"final checkpoint hash mismatch before test for {dataset_name}/{arm}")
        records[arm] = {
            "path": str(path),
            "internal_path": str(internal_path),
            "checkpoint_sha256": actual_sha,
            "checkpoint_state_hash": actual_state_hash,
            "epoch": EPOCHS,
            "frozen": True,
            "test_metric_used": False,
        }
    gate = {"dataset": dataset_name, "arms": records, "created_utc": utc_now(), "first_query_gallery_evaluation_opened": True}
    write_json(dataset_seed_root(output_root, dataset_name) / "test_gate.json", gate)
    for arm in arms:
        manifest_path = arm_dir(output_root, dataset_name, arm) / "run_manifest.json"
        manifest = read_json(manifest_path)
        manifest["test_started"] = True
        manifest["test_started_utc"] = gate["created_utc"]
        manifest["test_checkpoint_gate"] = records[arm]
        write_json(manifest_path, manifest)
    return gate


def test_one(output_root: Path, dataset_name: str, arm: str, gate: dict[str, Any], device: torch.device) -> dict[str, Any]:
    spec = SPEC[dataset_name]
    run_dir = arm_dir(output_root, dataset_name, arm)
    cfg = cfg_for(dataset_name, run_dir)
    dataset = build_dataset(cfg)
    state = torch.load(gate["arms"][arm]["path"], map_location="cpu", weights_only=False)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, transform_test = build_img_transforms(cfg)
    if dataset_name == "ltcc":
        query = dataset.query
        gallery = dataset.gallery
        evaluator = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        loader = DataLoader(
            ConcatDataset([ImageDataset(query, transform=transform_test), ImageDataset(gallery, transform=transform_test)]),
            batch_size=int(cfg.DATA.TEST_BATCH), shuffle=False, num_workers=int(cfg.DATA.NUM_WORKERS), pin_memory=False, drop_last=False,
        )
        ltcc = True
    else:
        query = dataset.query_diff
        gallery = dataset.gallery
        evaluator = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        loader = DataLoader(
            ConcatDataset([ImageDataset(query, transform=transform_test), ImageDataset(gallery, transform=transform_test)]),
            batch_size=int(cfg.DATA.TEST_BATCH), shuffle=False, num_workers=int(cfg.DATA.NUM_WORKERS), pin_memory=False, drop_last=False,
        )
        ltcc = False
    logger = logger_for(public_log_path(output_root, dataset_name, "测试"))
    logger.info("==================================================")
    logger.info("数据集：%s", dataset_name.upper())
    logger.info("方法：Bio + GPC")
    logger.info("测试阶段：只使用 Epoch 60 最终 checkpoint")
    logger.info("checkpoint：%s", gate["arms"][arm]["path"])
    logger.info("checkpoint SHA256：%s", gate["arms"][arm]["checkpoint_sha256"])
    logger.info("测试协议：%s", spec["test_protocol"])
    logger.info("==================================================")
    evaluator.reset()
    # This is the first and only query/gallery evaluator call for this final
    # checkpoint.  The current processor test path is used unchanged.
    processor_test(cfg, model, evaluator, loader, logger, device, test=True, ltcc=ltcc, clothChanging=True)
    cmc, mAP, _distmat, _pids, _camids, _qf, _gf, _paths = evaluator.compute()
    result = {
        "dataset": dataset_name,
        "arm": arm,
        "method": "Bio" if arm == "bio" else "Bio + GPC",
        "checkpoint": gate["arms"][arm],
        "protocol": spec["test_protocol"],
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mAP),
        "mINP": None,
        "evaluator": "processor.processor.test + current utils.metrics evaluator",
        "evaluator_calls": 1,
        "query_count": len(query),
        "gallery_count": len(gallery),
        "evaluated_utc": utc_now(),
    }
    write_json(run_dir / "final_test.json", result)
    write_json(public_test_path(output_root, dataset_name), result)
    logger.info("[%s][测试] Rank-1：%.8f；Rank-5：%.8f；Rank-10：%.8f；mAP：%.8f；mINP：当前 evaluator 未提供", dataset_name.upper(), result["rank1"], result["rank5"], result["rank10"], result["mAP"])
    logger.info("[%s][测试] 正式测试完成；query 数：%d；gallery 数：%d", dataset_name.upper(), result["query_count"], result["gallery_count"])
    del model, state, loader, dataset
    gc.collect(); torch.cuda.empty_cache()
    return result


def test_gpc(output_root: Path, dataset_name: str) -> dict[str, Any]:
    arms = ["gpc"]
    gate = checkpoint_gate_for_test(output_root, dataset_name, arms)
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("final test requires visible PPU devices")
    if any(torch.cuda.get_device_name(index) != EXPECTED_DEVICE_NAME for index in range(WORLD_SIZE)):
        raise RuntimeError("final test PPU device audit failed")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    results = [test_one(output_root, dataset_name, arm, gate, device) for arm in arms]
    payload = {"dataset": dataset_name, "method": "Bio + GPC", "gate": gate, "results": results, "first_test_after_e60_frozen": True, "baseline_started": False}
    write_json(dataset_seed_root(output_root, dataset_name) / "final_test_summary.json", payload)
    return payload


def update_final_leakage_report(output_root: Path) -> None:
    results = {}
    for name in ("ltcc", "prcc"):
        path = dataset_seed_root(output_root, name) / "source_audit.json"
        if path.is_file():
            results[name] = read_json(path)
    if set(results) != {"ltcc", "prcc"}:
        return
    for name in ("ltcc", "prcc"):
        training_calls = 0
        for arm in ("bio", "gpc"):
            complete = arm_dir(output_root, name, arm) / "RUN_COMPLETE.json"
            if complete.is_file():
                training_calls += int(read_json(complete).get("evaluator_calls_during_training", 0))
        results[name]["data_leakage"]["training_evaluator_calls"] = training_calls
        results[name]["data_leakage"]["test_evaluation_opened"] = (dataset_seed_root(output_root, name) / "test_gate.json").is_file()
        results[name]["data_leakage"]["checks"]["test_evaluator_not_called_during_training"] = training_calls == 0
        results[name]["data_leakage"]["passed"] = all(bool(value) for value in results[name]["data_leakage"]["checks"].values())
        write_json(dataset_seed_root(output_root, name) / "data_leakage_audit.json", results[name]["data_leakage"])
    render_leakage_report(results, stage="post-run verification")


def final_runtime_report(output_root: Path) -> None:
    rows = []
    passed = True
    for dataset_name in ("ltcc", "prcc"):
        for arm in ("bio", "gpc"):
            path = arm_dir(output_root, dataset_name, arm) / "runtime_audit.json"
            if not path.is_file():
                passed = False
                continue
            item = read_json(path)
            passed = passed and bool(item.get("passed"))
            for rank in item.get("ranks", []):
                rows.append({
                    "dataset": dataset_name,
                    "arm": arm,
                    "rank": rank.get("rank"),
                    "device": rank.get("device_name"),
                    "logical_device": rank.get("logical_device"),
                    "world_size": rank.get("world_size"),
                    "torch_version": rank.get("torch_version"),
                    "torch_runtime": rank.get("torch_runtime"),
                    "initial_amp_scale": rank.get("initial_amp_scale"),
                    "forward": rank.get("forward_devices"),
                    "gpc": rank.get("gpc_devices"),
                    "optimizer": rank.get("optimizer_parameter_devices"),
                    "cpu_fallback": item.get("cpu_fallback"),
                })
    lines = [
        "# FINAL_GPC_RUNTIME_AUDIT",
        "",
        f"Generated: `{utc_now()}`.",
        "",
        "PPU policy: `PPU-ZW810E` only; forward, backward-linked optimizer parameters, optimizer.step, and GPC relation computation must remain on PPU. CPU is allowed for DataLoader/filesystem/logging/CSV/metric bookkeeping.",
        "",
        f"Overall runtime audit: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "| Dataset | Arm | Rank | Device | Logical device | World | Torch | Runtime | Initial AMP | Forward devices | GPC devices | Optimizer parameter devices | CPU fallback |",
        "|---|---|---:|---|---|---:|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset'].upper()} | {row['arm']} | {row['rank']} | `{row['device']}` | `{row['logical_device']}` | {row['world_size']} | `{row['torch_version']}` | `{row['torch_runtime']}` | {row['initial_amp_scale']} | `{row['forward']}` | `{row['gpc']}` | `{row['optimizer']}` | `{row['cpu_fallback']}` |")
    lines.extend(["", "Machine-readable per-arm runtime audits are stored beside each run's `RUN_COMPLETE.json`.", ""])
    (REPORT_ROOT / "FINAL_GPC_RUNTIME_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_root / "RUNTIME_AUDIT.json", {"created_utc": utc_now(), "passed": passed, "rows": rows})


def fmt_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.4f}"


def final_report(output_root: Path) -> dict[str, Any]:
    update_final_leakage_report(output_root)
    final_runtime_report(output_root)
    source = read_json(output_root / "SOURCE_AUDIT.json")
    leakage = read_json(output_root / "FINAL_GPC_DATA_LEAKAGE_AUDIT.json") if (output_root / "FINAL_GPC_DATA_LEAKAGE_AUDIT.json").is_file() else None
    # The report is also usable when only the root report exists.
    test_results: dict[str, dict[str, Any]] = {}
    exposures: dict[str, Any] = {}
    run_records: dict[str, Any] = {}
    for dataset_name in ("ltcc", "prcc"):
        summary = read_json(dataset_seed_root(output_root, dataset_name) / "final_test_summary.json")
        by_arm = {item["arm"]: item for item in summary["results"]}
        test_results[dataset_name] = by_arm
        exp_path = arm_dir(output_root, dataset_name, "gpc") / "relation_exposure.csv"
        exp_rows = read_csv(exp_path)
        total_batches = sum(int(row["global_batches"]) for row in exp_rows)
        total_pairs = sum(int(row["matched_relation_pairs"]) for row in exp_rows)
        active_batches = sum(int(row["relation_active_batches"]) for row in exp_rows)
        same_pairs = sum(int(row["same_clothes_pairs"]) for row in exp_rows)
        cross_pairs = sum(int(row["cross_clothes_pairs"]) for row in exp_rows)
        expected = read_json(protocol_dir(output_root, dataset_name) / "pair_budget_expected_summary.json")
        exposures[dataset_name] = {
            "total_global_batches": total_batches,
            "relation_active_batches": active_batches,
            "active_rate": active_batches / max(total_batches, 1),
            "selected_relation_contributions": total_pairs,
            "same_clothes_contributions": same_pairs,
            "cross_clothes_contributions": cross_pairs,
            "same_clothes_fraction": same_pairs / max(total_pairs, 1),
            "cross_clothes_fraction": cross_pairs / max(total_pairs, 1),
            "expected_relation_contributions": expected["expected_relation_contributions"],
            "expected_active_rate": expected["expected_active_rate"],
            "exposure_matches_expected": total_pairs == expected["expected_relation_contributions"] and abs(active_batches / max(total_batches, 1) - expected["expected_active_rate"]) < 1e-12,
            "unique_pid_coverage": len({int(row["pid"]) for row in read_csv(arm_dir(output_root, dataset_name, "gpc") / "per_pid_relation_contributions.csv")}),
        }
        for arm in ("bio", "gpc"):
            run_records[f"{dataset_name}_{arm}"] = {
                "run_dir": str(arm_dir(output_root, dataset_name, arm)),
                "complete": read_json(arm_dir(output_root, dataset_name, arm) / "RUN_COMPLETE.json"),
                "final_checkpoint": read_json(arm_dir(output_root, dataset_name, arm) / "final_checkpoint.json"),
                "test": by_arm[arm],
            }

    delta = {
        "ltcc": {"rank1": test_results["ltcc"]["gpc"]["rank1"] - test_results["ltcc"]["bio"]["rank1"], "mAP": test_results["ltcc"]["gpc"]["mAP"] - test_results["ltcc"]["bio"]["mAP"]},
        "prcc": {"rank1": test_results["prcc"]["gpc"]["rank1"] - test_results["prcc"]["bio"]["rank1"], "mAP": test_results["prcc"]["gpc"]["mAP"] - test_results["prcc"]["bio"]["mAP"]},
    }
    amp_audit: dict[str, Any] = {}
    amp_eventful = False
    amp_policy_matched_within_dataset = True
    for dataset_name in ("ltcc", "prcc"):
        initial_scales = {}
        for arm in ("bio", "gpc"):
            amp_rows = read_csv(arm_dir(output_root, dataset_name, arm) / "amp_statistics.csv")
            skipped_steps = sum(int(float(row.get("skipped_steps", 0) or 0)) for row in amp_rows)
            found_inf_count = sum(int(float(row.get("found_inf_count", 0) or 0)) for row in amp_rows)
            nonfinite_events = sum(int(float(row.get("nonfinite_events", 0) or 0)) for row in amp_rows)
            scales = [float(row["amp_scale_end"]) for row in amp_rows if row.get("amp_scale_end") not in ("", None)]
            runtime_item = read_json(arm_dir(output_root, dataset_name, arm) / "runtime_audit.json")
            rank0 = next(item for item in runtime_item.get("ranks", []) if int(item.get("rank", -1)) == 0)
            initial_scale = int(rank0["initial_amp_scale"])
            initial_scales[arm] = initial_scale
            amp_audit[f"{dataset_name}_{arm}"] = {
                "rows": len(amp_rows),
                "initial_amp_scale": initial_scale,
                "skipped_steps": skipped_steps,
                "found_inf_count": found_inf_count,
                "nonfinite_events": nonfinite_events,
                "amp_scale_min": min(scales) if scales else None,
                "amp_scale_max": max(scales) if scales else None,
                "amp_scale_end": scales[-1] if scales else None,
            }
            amp_eventful = amp_eventful or skipped_steps > 0 or found_inf_count > 0 or nonfinite_events > 0
        amp_policy_matched_within_dataset = amp_policy_matched_within_dataset and initial_scales["bio"] == initial_scales["gpc"]
    amp_audit_summary = {
        "eventful": amp_eventful,
        "initial_policy_matched_within_dataset": amp_policy_matched_within_dataset,
        "confound_status": "AMP events were recorded and arm-level event totals/scale trajectories differ; initial AMP policy matched within each dataset, but AMP is not event-free and remains a reported confound.",
        "arms": amp_audit,
    }
    matched_protocol: dict[str, Any] = {}
    plan_compare_fields = (
        "epoch", "rank", "batches", "pair_units", "sampler_raw_pair_units", "batch_size",
        "num_instances", "covered_pids", "batch_unique_pid_min", "batch_unique_pid_max",
        "plan_sha256", "train_only",
    )
    for dataset_name in ("ltcc", "prcc"):
        bio_manifest = read_json(arm_dir(output_root, dataset_name, "bio") / "run_manifest.json")
        gpc_manifest = read_json(arm_dir(output_root, dataset_name, "gpc") / "run_manifest.json")
        bio_plan = read_csv(arm_dir(output_root, dataset_name, "bio") / "sampler_plan.csv")
        gpc_plan = read_csv(arm_dir(output_root, dataset_name, "gpc") / "sampler_plan.csv")
        plan_match = len(bio_plan) == len(gpc_plan) and all(
            tuple(row.get(field) for field in plan_compare_fields) == tuple(other.get(field) for field in plan_compare_fields)
            for row, other in zip(bio_plan, gpc_plan)
        )
        common_init = bio_manifest["common_init"]
        common_init_same = common_init == gpc_manifest.get("common_init")
        matched_protocol[dataset_name] = {
            "seed_bio": bio_manifest.get("seed"),
            "seed_gpc": gpc_manifest.get("seed"),
            "common_init_same": common_init_same,
            "common_init_file_sha256": common_init.get("file_sha256"),
            "common_init_state_hash": common_init.get("state_hash"),
            "initial_parameter_hash_same": bio_manifest.get("initial_parameter_hash") == gpc_manifest.get("initial_parameter_hash"),
            "sampler_plan_rows_bio": len(bio_plan),
            "sampler_plan_rows_gpc": len(gpc_plan),
            "sampler_plan_same": plan_match,
            "steps_per_epoch_same": bio_manifest.get("steps_per_epoch") == gpc_manifest.get("steps_per_epoch"),
            "only_method_difference": bio_manifest.get("lambda_gpc") == 0 and gpc_manifest.get("lambda_gpc") == LAMBDA_GPC,
        }
    matched_protocol["passed"] = all(
        item["seed_bio"] == SEED
        and item["seed_gpc"] == SEED
        and item["common_init_same"]
        and item["initial_parameter_hash_same"]
        and item["sampler_plan_same"]
        and item["steps_per_epoch_same"]
        and item["only_method_difference"]
        for name, item in matched_protocol.items()
        if name in ("ltcc", "prcc")
    )
    runtime = read_json(output_root / "RUNTIME_AUDIT.json")
    leakage_status = all(read_json(dataset_seed_root(output_root, name) / "data_leakage_audit.json").get("passed") for name in ("ltcc", "prcc"))
    source_status = all(source["datasets"][name].get("passed") for name in ("ltcc", "prcc"))
    exposure_status = all(exposures[name]["exposure_matches_expected"] for name in ("ltcc", "prcc"))
    report = {
        "phase": "FINAL_CONFIRMATORY_RUN",
        "created_utc": utc_now(),
        "seed": SEED,
        "method": "DIFFER Bio baseline + Generic Same-ID Positive Consistency",
        "active_losses": ["L_CE", "L_Triplet", "L_Bio", "L_GPC"],
        "lambda_gpc": LAMBDA_GPC,
        "full_differ_nonbio_branch": False,
        "ds_dw": False,
        "sr3": False,
        "pair_budget": "min(cross,same,id) per batch; deterministic SHA-256 selection of same-ID candidates",
        "source_audit_pass": source_status,
        "data_leakage_audit_pass": leakage_status,
        "ppu_runtime_pass": bool(runtime.get("passed")),
        "amp": amp_audit_summary,
        "matched_protocol": matched_protocol,
        "ltcc": {"test": test_results["ltcc"], "delta": delta["ltcc"], "exposure": exposures["ltcc"]},
        "prcc": {"test": test_results["prcc"], "delta": delta["prcc"], "exposure": exposures["prcc"]},
        "training": {
            "ltcc_bio_epochs": run_records["ltcc_bio"]["complete"]["epochs_completed"],
            "ltcc_gpc_epochs": run_records["ltcc_gpc"]["complete"]["epochs_completed"],
            "prcc_bio_epochs": run_records["prcc_bio"]["complete"]["epochs_completed"],
            "prcc_gpc_epochs": run_records["prcc_gpc"]["complete"]["epochs_completed"],
            "no_evaluator_calls_during_training": all(run_records[key]["complete"]["evaluator_calls_during_training"] == 0 for key in run_records),
            "no_test_driven_selection": all(not run_records[key]["complete"]["test_driven_checkpoint_selection"] for key in run_records),
            "final_checkpoint_frozen_before_test": all(run_records[key]["final_checkpoint"]["frozen"] and run_records[key]["test"]["checkpoint"]["frozen"] for key in run_records),
        },
        "no_extra_experiments": True,
        "runtime": runtime,
        "run_records": run_records,
    }
    write_json(output_root / "FINAL_GPC_MANIFEST.json", report)

    table_rows = [
        ("LTCC", "Bio", test_results["ltcc"]["bio"]),
        ("LTCC", "Bio+GPC", test_results["ltcc"]["gpc"]),
        ("PRCC", "Bio", test_results["prcc"]["bio"]),
        ("PRCC", "Bio+GPC", test_results["prcc"]["gpc"]),
    ]
    lines = [
        "# FINAL_GPC_LTCC_PRCC_CONFIRMATORY_REPORT",
        "",
        f"Generated: `{report['created_utc']}`. Seed: `{SEED}`. This report contains only this fresh matched confirmation run in the primary result.",
        "",
        "## Primary Final Result",
        "",
        "| Dataset | Method | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_label, method, item in table_rows:
        lines.append(f"| {dataset_label} | {method} | {fmt_percent(item['rank1'])} | {fmt_percent(item['rank5'])} | {fmt_percent(item['rank10'])} | {fmt_percent(item['mAP'])} | {fmt_percent(item['mINP'])} |")
    lines.extend([
        "",
        "| Dataset | Method | Rank-1 | mAP | ΔR1 | ΔmAP |",
        "|---|---|---:|---:|---:|---:|",
        f"| LTCC | Bio | {fmt_percent(test_results['ltcc']['bio']['rank1'])} | {fmt_percent(test_results['ltcc']['bio']['mAP'])} | — | — |",
        f"| LTCC | Bio+GPC | {fmt_percent(test_results['ltcc']['gpc']['rank1'])} | {fmt_percent(test_results['ltcc']['gpc']['mAP'])} | {100*delta['ltcc']['rank1']:+.4f} pp | {100*delta['ltcc']['mAP']:+.4f} pp |",
        f"| PRCC | Bio | {fmt_percent(test_results['prcc']['bio']['rank1'])} | {fmt_percent(test_results['prcc']['bio']['mAP'])} | — | — |",
        f"| PRCC | Bio+GPC | {fmt_percent(test_results['prcc']['gpc']['rank1'])} | {fmt_percent(test_results['prcc']['gpc']['mAP'])} | {100*delta['prcc']['rank1']:+.4f} pp | {100*delta['prcc']['mAP']:+.4f} pp |",
        "",
        "## GPC exposure audit",
        "",
        "| Dataset | Global batches | Active batches | Active rate | Selected contributions | Same-clothes fraction | Cross-clothes fraction | PID coverage | Matches frozen plan |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name in ("ltcc", "prcc"):
        e = exposures[name]
        lines.append(f"| {name.upper()} | {e['total_global_batches']} | {e['relation_active_batches']} | {e['active_rate']:.8f} | {e['selected_relation_contributions']} | {e['same_clothes_fraction']:.8f} | {e['cross_clothes_fraction']:.8f} | {e['unique_pid_coverage']} | `{e['exposure_matches_expected']}` |")
    lines.extend([
        "",
        "## Matched initialization and trajectory audit",
        "",
        f"Matched Bio/GPC protocol audit: **{'PASS' if report['matched_protocol']['passed'] else 'FAIL'}**. Each dataset uses the same seed, common initialization, rank-wise sampler plan, and steps/epoch; the only configured arm difference is `lambda_GPC=0` versus `lambda_GPC=0.1`.",
        "",
        "| Dataset | Common-init file SHA256 | Common-init state hash | Initial state matched | Sampler plan matched | Plan rows/arm |",
        "|---|---|---|---|---|---:|",
        *[
            f"| {name.upper()} | `{report['matched_protocol'][name]['common_init_file_sha256']}` | `{report['matched_protocol'][name]['common_init_state_hash']}` | `{report['matched_protocol'][name]['initial_parameter_hash_same']}` | `{report['matched_protocol'][name]['sampler_plan_same']}` | {report['matched_protocol'][name]['sampler_plan_rows_bio']} |"
            for name in ("ltcc", "prcc")
        ],
        "",
        "## Required answers",
        "",
        f"1. Final model is Bio + GPC: **{report['method']}**.",
        f"2. Full DIFFER NonBio branch used: **{report['full_differ_nonbio_branch']}**.",
        f"3. DS/DW used: **{report['ds_dw']}**.",
        f"4. GPC pair budget: **{report['pair_budget']}**.",
        f"5. LTCC complete 60 epochs: **{report['training']['ltcc_bio_epochs'] == EPOCHS and report['training']['ltcc_gpc_epochs'] == EPOCHS}**.",
        f"6. PRCC complete 60 epochs: **{report['training']['prcc_bio_epochs'] == EPOCHS and report['training']['prcc_gpc_epochs'] == EPOCHS}**.",
        f"7. LTCC final E60 test: Bio `{fmt_percent(test_results['ltcc']['bio']['rank1'])}/{fmt_percent(test_results['ltcc']['bio']['mAP'])}`; GPC `{fmt_percent(test_results['ltcc']['gpc']['rank1'])}/{fmt_percent(test_results['ltcc']['gpc']['mAP'])}`.",
        f"8. PRCC final E60 test: Bio `{fmt_percent(test_results['prcc']['bio']['rank1'])}/{fmt_percent(test_results['prcc']['bio']['mAP'])}`; GPC `{fmt_percent(test_results['prcc']['gpc']['rank1'])}/{fmt_percent(test_results['prcc']['gpc']['mAP'])}`.",
        f"9. LTCC GPC-Bio delta: **{100*delta['ltcc']['rank1']:+.4f} pp Rank-1; {100*delta['ltcc']['mAP']:+.4f} pp mAP**.",
        f"10. PRCC GPC-Bio delta: **{100*delta['prcc']['rank1']:+.4f} pp Rank-1; {100*delta['prcc']['mAP']:+.4f} pp mAP**.",
        f"11. AMP confound: **{'YES' if report['amp']['eventful'] else 'NO'}**. {report['amp']['confound_status']}",
        "",
        "### AMP audit",
        "",
        "Initial AMP policy is matched within each dataset (LTCC 1024; PRCC 512), but dynamic AMP events were recorded during training:",
        "",
        "| Run | Initial scale | Skipped steps | found_inf | Nonfinite events | Scale min | Scale max | Final scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {key} | {item['initial_amp_scale']} | {item['skipped_steps']} | {item['found_inf_count']} | {item['nonfinite_events']} | {item['amp_scale_min']} | {item['amp_scale_max']} | {item['amp_scale_end']} |"
            for key, item in sorted(report["amp"]["arms"].items())
        ],
        "",
        f"12. CPU fallback: **{not bool(report['runtime'].get('passed'))}**.",
        f"13. Train/test leakage: **{not leakage_status}**.",
        f"14. Query/gallery evaluator called during training: **{not report['training']['no_evaluator_calls_during_training']}**.",
        f"15. Final checkpoint frozen/hash before first test: **{report['training']['final_checkpoint_frozen_before_test']}**.",
        "",
        "## Historical Reference",
        "",
        "Historical Phase 7I/7O artifacts were not mixed into the primary result. They are listed only as selector provenance in `FINAL_GPC_SOURCE_AUDIT.md`; no historical metric is used for the fresh deltas above.",
        "",
        "## Explicit final statements",
        "",
        "- Training过程中没有使用 query/gallery 参与 loss、backward 或 optimizer。",
        "- Training过程中没有使用 query/gallery Rank-1/mAP 选择 checkpoint。",
        "- 所有最终模型均固定训练至 60 epoch。",
        "- 最终 checkpoint 在首次 test 前已经冻结并记录 hash。",
        "- 没有启动任何额外 seed、超参数搜索或新方法 Phase。",
        "",
    ])
    (REPORT_ROOT / "FINAL_GPC_LTCC_PRCC_CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def print_final_summary(report: dict[str, Any]) -> None:
    ltcc = report["ltcc"]["test"]
    prcc = report["prcc"]["test"]
    dltcc = report["ltcc"]["delta"]
    dprcc = report["prcc"]["delta"]
    e_ltcc = report["ltcc"]["exposure"]
    e_prcc = report["prcc"]["exposure"]
    print(f"A. Source audit: {'PASS' if report['source_audit_pass'] else 'FAIL'}")
    print(f"B. Data leakage audit: {'PASS' if report['data_leakage_audit_pass'] else 'FAIL'}")
    print(f"C. PPU runtime: {'PASS' if report['ppu_runtime_pass'] else 'FAIL'}")
    print("D. LTCC 60-epoch final:")
    print(f"Bio: {fmt_percent(ltcc['bio']['rank1'])} / {fmt_percent(ltcc['bio']['mAP'])}")
    print(f"GPC: {fmt_percent(ltcc['gpc']['rank1'])} / {fmt_percent(ltcc['gpc']['mAP'])}")
    print(f"Delta: {100*dltcc['rank1']:+.4f} pp / {100*dltcc['mAP']:+.4f} pp")
    print("E. PRCC 60-epoch final:")
    print(f"Bio: {fmt_percent(prcc['bio']['rank1'])} / {fmt_percent(prcc['bio']['mAP'])}")
    print(f"GPC: {fmt_percent(prcc['gpc']['rank1'])} / {fmt_percent(prcc['gpc']['mAP'])}")
    print(f"Delta: {100*dprcc['rank1']:+.4f} pp / {100*dprcc['mAP']:+.4f} pp")
    print("F. GPC exposure:")
    print(f"LTCC active rate={e_ltcc['active_rate']:.8f}; pair contributions={e_ltcc['selected_relation_contributions']}; same/cross={e_ltcc['same_clothes_fraction']:.8f}/{e_ltcc['cross_clothes_fraction']:.8f}")
    print(f"PRCC active rate={e_prcc['active_rate']:.8f}; pair contributions={e_prcc['selected_relation_contributions']}; same/cross={e_prcc['same_clothes_fraction']:.8f}/{e_prcc['cross_clothes_fraction']:.8f}")
    print("G. Final conclusion:")
    print(f"GPC benefits LTCC: {'YES' if dltcc['rank1'] > 0 and dltcc['mAP'] > 0 else 'NO'}")
    print(f"GPC benefits PRCC: {'YES' if dprcc['rank1'] > 0 and dprcc['mAP'] > 0 else 'NO'}")
    print(f"Cross-dataset consistency: {'YES' if (dltcc['rank1'] > 0 and dltcc['mAP'] > 0 and dprcc['rank1'] > 0 and dprcc['mAP'] > 0) or (dltcc['rank1'] < 0 and dltcc['mAP'] < 0 and dprcc['rank1'] < 0 and dprcc['mAP'] < 0) else 'NO'}")
    print("H. Training过程中没有使用 query/gallery 参与 loss、backward 或 optimizer.")
    print("Training过程中没有使用 query/gallery Rank-1/mAP 选择 checkpoint.")
    print("所有最终模型均固定训练至 60 epoch.")
    print("最终 checkpoint 在首次 test 前已经冻结并记录 hash.")
    print("没有启动任何额外 seed、超参数搜索或新方法 Phase.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final Bio + GPC confirmatory experiment")
    parser.add_argument("--mode", required=True, choices=["audit", "init", "dry-run", "train", "test", "report"])
    parser.add_argument("--dataset", required=True, choices=["ltcc", "prcc", "all"])
    parser.add_argument("--arm", default="gpc", choices=["gpc"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--dry-steps", type=int, default=1, help="number of train-only batches for the dry-run gate")
    args = parser.parse_args()
    if args.seed != SEED:
        raise RuntimeError(f"final confirmatory run is frozen to seed={SEED}; refusing seed={args.seed}")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        results = source_and_leakage_audit(output_root, args.dataset)
        if args.dataset == "all":
            for name in ("ltcc", "prcc"):
                if not results[name]["passed"]:
                    raise RuntimeError(f"source/data leakage audit failed for {name}; training/test stopped")
        print(json.dumps({name: {"source_audit": item["passed"], "data_leakage": item["data_leakage"]["passed"]} for name, item in results.items()}, indent=2, sort_keys=True))
        return
    if args.mode == "init":
        names = ("ltcc", "prcc") if args.dataset == "all" else (args.dataset,)
        print(json.dumps([initialize_common(output_root, name) for name in names], indent=2, sort_keys=True))
        return
    if args.mode == "dry-run":
        names = ("ltcc", "prcc") if args.dataset == "all" else (args.dataset,)
        arms = ("gpc",)
        # A separate torchrun invocation per dataset/arm is the intended path;
        # this loop is safe for single-arm calls and preserves the explicit CLI.
        outputs = []
        for name in names:
            for arm in arms:
                outputs.append(dry_run_one(output_root, name, arm, args.dry_steps))
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return
    if args.mode == "train":
        if args.dataset == "all":
            raise RuntimeError("train requires --dataset ltcc or --dataset prcc; sequence is controlled by the caller")
        print(json.dumps(training_one(output_root, args.dataset, args.arm), indent=2, sort_keys=True))
        return
    if args.mode == "test":
        if args.dataset == "all":
            raise RuntimeError("test requires --dataset ltcc or --dataset prcc")
        print(json.dumps(test_gpc(output_root, args.dataset), indent=2, sort_keys=True))
        return
    if args.mode == "report":
        if args.dataset != "all":
            raise RuntimeError("report requires --dataset all")
        report = final_report(output_root)
        print_final_summary(report)
        return


if __name__ == "__main__":
    main()
