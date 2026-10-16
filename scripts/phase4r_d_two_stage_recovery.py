#!/usr/local/bin/python3
"""PHASE 4R-D: LTCC Full recovery with a calibrated alternating adversary.

This is a repair-track experiment.  It is intentionally independent of the
formal reproduction outputs and never overwrites them.

The protocol is:

* freeze the formal LTCC Bio E27 best checkpoint as an anchor;
* create one full topology and one deterministic reverse-head initialization;
* prepare one fixed 60-epoch batch/augmentation plan;
* screen D0 (Bio continuation), D1 (late standard Full), and D2 (reverse
  calibration followed by alternating min-max updates) on E28--E32; and
* leave long-run continuation to the post-screening gate.

The formal E27 checkpoint contains model weights only.  The missing optimizer,
scheduler, GradScaler, and RNG states are therefore reconstructed and are
recorded as such in the anchor manifest.  No claim of an exact optimizer
resume is made.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset, read_image  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
from loss import make_loss  # noqa: E402
from model.build import factory  # noqa: E402
from processor.processor import test  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402
from utils.phase4r_telemetry import collect_component_gradient_telemetry  # noqa: E402


SEED = 1234
WORLD_SIZE = 2
STEPS_PER_EPOCH = 596
BATCH_SIZE = 8
FORMAL_EPOCHS = 60
ANCHOR_EPOCH = 27
SCREEN_START_EPOCH = 28
SCREEN_END_EPOCH = 32
CALIBRATION_EPOCHS = 3
INITIAL_SCALE_FORMAL = 1024.0

DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
BIO_CHECKPOINT = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/eva02_l_bio_best.pth"
BIO_MANIFEST = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/RUN_MANIFEST.json"
BIO_METRICS = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv"
BIO_AMP = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/AMP_statistics.json"
EXPECTED_BIO_SHA256 = "2e36311eb3e730b872049347b30989ce16d4a75f3bbe4a50c56fc3ec158c5e92"

OUT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/D_two_stage_recovery"
ANCHOR_DIR = OUT / "COMMON_E27"
PLAN_DIR = OUT / "batch_plan"
PLAN_MANIFEST = OUT / "BATCH_PLAN_MANIFEST.json"

CANDIDATE_DIRS = {
    "D0": OUT / "D0_bio_continuation",
    "D1": OUT / "D1_late_standard_full",
    "D2": OUT / "D2_calibrated_alternating",
}

PROBE_DIRS = {
    "hair": OUT / "D2_factor_attribution" / "Hair_only",
    "clothing": OUT / "D2_factor_attribution" / "Clothing_only",
}

REVERSE_PREFIX = "head_clip_bioReverse."
GROUP_PREFIXES = {
    "early_eva": ("blocks.0.",),
    "middle_eva": ("blocks.11.",),
    "last_eva": ("blocks.22.",),
    "bio_projection": ("head_clip_bio.",),
}


class ScalarCapture:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag, value, global_step=None, *args, **kwargs):
        self.values.setdefault(str(tag), []).append(float(value))

    def last(self, tag: str, default: float = float("nan")) -> float:
        values = self.values.get(tag, [])
        return float(values[-1]) if values else float(default)


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PlannedTrainDataset(Dataset):
    """Replay one serialized image/augmentation plan without worker RNG."""

    def __init__(self, samples: list[tuple[dict[str, Any], int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, augmentation_seed = self.samples[index]
        item = dict(source)
        image = read_image(item["image_path"])
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            seed = int(augmentation_seed)
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            image = self.transform(image)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        item["image"] = image
        item["augmentation_seed"] = int(augmentation_seed)
        return item


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(type(value).__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    result = {}
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        result[key] = sha256_bytes(value.numpy().tobytes())
    return result


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in raw_model(model).state_dict().items()
    }


def strip_module_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"


def configure_full() -> Any:
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    cfg.MODEL.DIST_TRAIN = True
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = str(DATA_ROOT)
    cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.DATA.NUM_INSTANCES = 2
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.MAX_EPOCHS = FORMAL_EPOCHS
    cfg.TRAIN.START_EPOCH = 1
    cfg.OUTPUT_DIR = str(OUT)
    cfg.TEST.WEIGHT = ""
    cfg.freeze()
    return cfg


def configure_core() -> Any:
    cfg = configure_full()
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.freeze()
    return cfg


def source_metrics() -> dict[int, dict[str, float]]:
    with BIO_METRICS.open() as stream:
        result = {}
        for row in csv.DictReader(stream):
            epoch = int(float(row["epoch"]))
            result[epoch] = {}
            for key, value in row.items():
                if key == "epoch":
                    continue
                try:
                    result[epoch][key] = float(value)
                except (TypeError, ValueError):
                    result[epoch][key] = float("nan")
        return result


def build_raw_model(cfg: Any, state: dict[str, torch.Tensor], device: torch.device | None = None):
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    model.load_state_dict(state, strict=True)
    if device is not None:
        model.to(device)
    return model


def reverse_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items() if key.startswith(REVERSE_PREFIX)}


def shared_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items() if not key.startswith(REVERSE_PREFIX)}


def load_anchor_state() -> dict[str, torch.Tensor]:
    path = ANCHOR_DIR / "FULL_TOPOLOGY_E27_INIT.pth"
    if not path.exists():
        raise FileNotFoundError(f"missing anchor; run --mode anchor first: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def create_anchor() -> None:
    """Freeze the formal Bio E27 model and reconstruct missing runtime state."""
    if not BIO_CHECKPOINT.exists() or not BIO_MANIFEST.exists() or not BIO_METRICS.exists():
        raise FileNotFoundError("formal Bio E27 artifacts are incomplete")
    checkpoint_sha = sha256_file(BIO_CHECKPOINT)
    if checkpoint_sha != EXPECTED_BIO_SHA256:
        raise RuntimeError(f"Bio checkpoint SHA256 mismatch: {checkpoint_sha} != {EXPECTED_BIO_SHA256}")
    metrics = source_metrics()
    if ANCHOR_DIR.exists() and any(ANCHOR_DIR.iterdir()):
        if (ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json").exists():
            existing = json.loads((ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json").read_text())
            if existing.get("source_checkpoint_sha256") == checkpoint_sha:
                print(json.dumps({"anchor": str(ANCHOR_DIR), "reused": True}, sort_keys=True))
                return
        raise RuntimeError(f"refusing to overwrite non-empty anchor directory: {ANCHOR_DIR}")

    e27 = metrics.get(ANCHOR_EPOCH)
    if e27 is None:
        raise RuntimeError("formal Bio metrics do not contain epoch 27")
    best_epoch = max(metrics, key=lambda epoch: metrics[epoch].get("cc_rank1", -float("inf")))
    if best_epoch != ANCHOR_EPOCH:
        raise RuntimeError(f"formal Bio BEST epoch changed: expected 27, got {best_epoch}")

    source_manifest = json.loads(BIO_MANIFEST.read_text())
    source_state = torch.load(BIO_CHECKPOINT, map_location="cpu", weights_only=True)
    source_state = strip_module_keys(dict(source_state))
    if len(source_state) != 515:
        raise RuntimeError(f"expected 515 Bio state keys, got {len(source_state)}")

    cfg = configure_full()
    set_seed(SEED)
    fresh_full = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    fresh_state = cpu_state_dict(fresh_full)
    if len(fresh_state) != 517:
        raise RuntimeError(f"expected 517 full-topology state keys, got {len(fresh_state)}")
    for key, value in source_state.items():
        if key not in fresh_state:
            raise RuntimeError(f"Bio key missing from full topology: {key}")
        if tuple(value.shape) != tuple(fresh_state[key].shape):
            raise RuntimeError(f"shape mismatch for {key}: {tuple(value.shape)} != {tuple(fresh_state[key].shape)}")
        fresh_state[key] = value.detach().cpu().clone()
    fresh_full.load_state_dict(fresh_state, strict=True)

    missing_reverse = sorted(set(fresh_state) - set(source_state))
    if missing_reverse != sorted(reverse_keys(fresh_state)):
        raise RuntimeError(f"unexpected missing full-topology keys: {missing_reverse}")

    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    full_path = ANCHOR_DIR / "FULL_TOPOLOGY_E27_INIT.pth"
    reverse_path = ANCHOR_DIR / "REVERSE_INIT_E27.pth"
    torch.save(fresh_state, full_path)
    torch.save(reverse_keys(fresh_state), reverse_path)
    (ANCHOR_DIR / "config.yml").write_text(cfg.dump())

    # The scheduler itself is stateless, so its E27 position can be rebuilt
    # exactly from the formal config even though the original object was not
    # serialized.  The optimizer momentum and GradScaler growth tracker cannot.
    probe_optimizer = make_optimizer_for(fresh_full, "all")
    probe_scheduler = create_scheduler(cfg, probe_optimizer)
    probe_scheduler.step(ANCHOR_EPOCH)
    e27_scale = float(e27.get("amp_scale_end", INITIAL_SCALE_FORMAL))
    amp_summary = json.loads(BIO_AMP.read_text()) if BIO_AMP.exists() else {}
    manifest = {
        "phase": "4R-D",
        "track": "repair/research; LTCC Full recovery",
        "anchor_name": "BIO_E27_ANCHOR",
        "created_utc": utc_now(),
        "seed": SEED,
        "dataset": "LTCC",
        "anchor_epoch": ANCHOR_EPOCH,
        "source_checkpoint": str(BIO_CHECKPOINT),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_expected_sha256": EXPECTED_BIO_SHA256,
        "source_checkpoint_bytes": BIO_CHECKPOINT.stat().st_size,
        "source_manifest": str(BIO_MANIFEST),
        "source_manifest_sha256": sha256_file(BIO_MANIFEST),
        "source_metrics": str(BIO_METRICS),
        "source_amp_statistics": str(BIO_AMP),
        "formal_best_epoch": best_epoch,
        "formal_e27_metrics": e27,
        "full_topology": {
            "state_key_count": len(fresh_state),
            "state_numel": int(sum(value.numel() for value in fresh_state.values())),
            "state_digest": state_digest(fresh_state),
            "state_file": str(full_path),
            "state_file_sha256": sha256_file(full_path),
            "reverse_init_file": str(reverse_path),
            "reverse_init_file_sha256": sha256_file(reverse_path),
            "reverse_keys": sorted(reverse_keys(fresh_state)),
            "shared_keys_loaded_from_bio": len(source_state),
        },
        "formal_runtime_state_availability": {
            "optimizer_saved": bool(source_manifest.get("checkpoint_policy", {}).get("optimizer_saved", False)),
            "scheduler_saved": bool(source_manifest.get("checkpoint_policy", {}).get("scheduler_saved", False)),
            "gradscaler_saved": bool(source_manifest.get("checkpoint_policy", {}).get("gradscaler_saved", False)),
            "rng_saved": False,
        },
        "reconstructed_runtime_state": {
            "optimizer": {
                "status": "RECONSTRUCTED; exact E27 momentum state unavailable",
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": 0.05,
                "bias_lr_factor": 2.0,
                "momentum_buffers_initialized": "empty",
            },
            "scheduler": {
                "status": "RECONSTRUCTED_FROM_FORMAL_CONFIG",
                "total_epochs": FORMAL_EPOCHS,
                "position_set_by_step": ANCHOR_EPOCH,
                "next_training_epoch": SCREEN_START_EPOCH,
                "lr_at_e27": [float(group["lr"]) for group in probe_optimizer.param_groups[:1]],
                "lr_bias_at_e27": [float(group["lr"]) for group in probe_optimizer.param_groups if group.get("initial_lr", group["lr"]) > cfg.SOLVER.BASE_LR][:1],
            },
            "grad_scaler": {
                "status": "RECONSTRUCTED; exact growth tracker unavailable",
                "formal_initial_scale": INITIAL_SCALE_FORMAL,
                "reconstructed_scale_at_e27": e27_scale,
                "formal_e27_row": {
                    key: e27.get(key)
                    for key in ("amp_scale_min", "amp_scale_max", "amp_scale_end", "skipped_steps", "nonfinite_events")
                },
                "full_run_summary": amp_summary,
            },
            "rng": {
                "status": "NOT_SAVED; fixed seed and serialized batch/augmentation plan used",
                "seed": SEED,
                "augmentation": "per-sample seeds are serialized in BATCH_PLAN_MANIFEST.json",
            },
        },
        "anchor_validation": {
            "source_state_key_count": len(source_state),
            "full_state_key_count": len(fresh_state),
            "missing_keys_are_reverse_only": True,
            "model_strict_load": True,
            "formal_cc_rank1": float(e27["cc_rank1"]),
            "formal_cc_mAP": float(e27["cc_mAP"]),
        },
        "forbidden_scope": [
            "formal reproduction overwrite",
            "GRL lambda sweep",
            "delayed lambda ramp",
            "loss-weight sweep",
            "PRCC",
            "seed sweep",
            "new architecture",
        ],
        "git": {
            "commit": git_output("git", "rev-parse", "HEAD"),
            "status": git_output("git", "status", "--short"),
        },
    }
    (ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    (ANCHOR_DIR / "ANCHOR_COMPLETE.json").write_text(json.dumps({
        "anchor": "BIO_E27_ANCHOR",
        "created_utc": utc_now(),
        "manifest": str(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
        "full_topology_state": str(full_path),
        "reverse_init": str(reverse_path),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "anchor": str(ANCHOR_DIR),
        "reused": False,
        "source_checkpoint_sha256": checkpoint_sha,
        "full_topology_state_sha256": sha256_file(full_path),
        "formal_e27": [e27["cc_rank1"], e27["cc_mAP"]],
        "optimizer_state": "reconstructed; no formal state artifact",
    }, indent=2, sort_keys=True))
    del fresh_full, fresh_state, source_state
    gc.collect()


def init_dist(backend: str, require_cuda: bool) -> int:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(f"expected world size {WORLD_SIZE}, got {dist.get_world_size()}")
    rank = dist.get_rank()
    if require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("Repair Track requires exactly two visible PPUs")
        torch.cuda.set_device(rank)
    return rank


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


def plan_seed(global_step: int, position: int, dataset_index: int, rank: int) -> int:
    token = f"DIFFER_PHASE4R_D_TWO_STAGE|{SEED}|{rank}|{global_step}|{position}|{dataset_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


def plan_path(rank: int) -> Path:
    return PLAN_DIR / f"plan_rank{rank}.jsonl"


def generate_rank_plan(dataset: Any, rank: int, path: Path) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite plan: {path}")
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=2,
        seed=SEED,
    )
    digest = hashlib.sha256()
    rows = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for epoch0 in range(FORMAL_EPOCHS):
            sampler.set_epoch(epoch0)
            indices = list(iter(sampler))
            expected = STEPS_PER_EPOCH * BATCH_SIZE
            if len(indices) < expected:
                raise RuntimeError(f"epoch {epoch0 + 1}: expected at least {expected} indices, got {len(indices)}")
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                start = (iteration - 1) * BATCH_SIZE
                batch_indices = indices[start:start + BATCH_SIZE]
                samples = []
                global_step = epoch0 * STEPS_PER_EPOCH + iteration
                for position, dataset_index in enumerate(batch_indices):
                    source = dataset.train[int(dataset_index)]
                    item = {
                        "dataset_index": int(dataset_index),
                        "image_path": str(source["image_path"]),
                        "pid": int(source["pid"]),
                        "camid": int(source["camid"]),
                        "clothes_id": int(source["clothes_id"]),
                        "augmentation_seed": plan_seed(global_step, position, int(dataset_index), rank),
                    }
                    samples.append(item)
                    digest.update(json.dumps(item, sort_keys=True).encode())
                stream.write(json.dumps({
                    "global_step": global_step,
                    "epoch": epoch0 + 1,
                    "iteration": iteration,
                    "samples": samples,
                }, sort_keys=True) + "\n")
                rows += 1
    return {
        "rank": rank,
        "path": str(path),
        "rows": rows,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": FORMAL_EPOCHS,
        "sampler_length": len(sampler),
        "dropped_sampler_tail": len(sampler) - STEPS_PER_EPOCH * BATCH_SIZE,
        "metadata_sha256": digest.hexdigest(),
        "file_sha256": sha256_file(path),
    }


def prepare_plan() -> None:
    if PLAN_MANIFEST.exists() or any(PLAN_DIR.glob("plan_rank*.jsonl")):
        raise RuntimeError(f"batch plan already exists; refusing to overwrite: {OUT}")
    rank = init_dist("gloo", require_cuda=False)
    try:
        cfg = configure_full()
        set_seed(SEED)
        dataset = build_dataset(cfg)
        info = generate_rank_plan(dataset, rank, plan_path(rank))
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, info)
        dist.barrier()
        if rank == 0:
            manifest = {
                "phase": "4R-D",
                "track": "repair/research; two-stage full recovery",
                "created_utc": utc_now(),
                "seed": SEED,
                "world_size": WORLD_SIZE,
                "epochs": FORMAL_EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "batch_size": BATCH_SIZE,
                "augmentation": "serialized per-sample Python/NumPy seeds",
                "sampler": "DistributedRandomIdentitySampler(seed=1234, num_instances=2)",
                "data": {
                    "train_images": len(dataset.train),
                    "train_ids": dataset.num_train_pids,
                    "train_clothes": dataset.num_train_clothes,
                    "query": len(dataset.query),
                    "gallery": len(dataset.gallery),
                },
                "rank_plans": gathered,
            }
            OUT.mkdir(parents=True, exist_ok=True)
            PLAN_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        dist.barrier()
        print(json.dumps(info, sort_keys=True), flush=True)
    finally:
        cleanup_dist()


def read_plan_slice(path: Path, start_index: int, end_index: int) -> list[dict[str, Any]]:
    if end_index <= start_index:
        return []
    rows = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index >= end_index:
                break
            if index >= start_index and line.strip():
                rows.append(json.loads(line))
    if len(rows) != end_index - start_index:
        raise RuntimeError(f"plan slice {path} returned {len(rows)} rows; expected {end_index - start_index}")
    return rows


def planned_loader(cfg: Any, rank: int, start_epoch: int, end_epoch: int):
    if not PLAN_MANIFEST.exists():
        raise FileNotFoundError(f"missing plan manifest: {PLAN_MANIFEST}")
    start = (start_epoch - 1) * STEPS_PER_EPOCH
    end = end_epoch * STEPS_PER_EPOCH
    selected = read_plan_slice(plan_path(rank), start, end)
    dataset = build_dataset(cfg)
    transform_train, _ = build_img_transforms(cfg)
    samples: list[tuple[dict[str, Any], int]] = []
    batches: list[list[int]] = []
    for row in selected:
        if int(row["epoch"]) < start_epoch or int(row["epoch"]) > end_epoch:
            raise RuntimeError("plan epoch outside requested slice")
        batch = []
        for item in row["samples"]:
            source = dataset.train[int(item["dataset_index"])]
            for key in ("image_path", "pid", "camid", "clothes_id"):
                if str(source[key]) != str(item[key]):
                    raise RuntimeError(f"plan mismatch {key}: {source[key]} != {item[key]}")
            batch.append(len(samples))
            samples.append((dict(source), int(item["augmentation_seed"])))
        if len(batch) != BATCH_SIZE:
            raise RuntimeError("incomplete planned batch")
        batches.append(batch)
    loader = DataLoader(
        PlannedTrainDataset(samples, transform_train),
        batch_sampler=FixedBatchSampler(batches),
        num_workers=0,
        pin_memory=False,
    )
    return loader, dataset, selected


def eval_loader(cfg: Any, dataset: Any):
    _, transform_test = build_img_transforms(cfg)
    query = ImageDataset(dataset.query, transform=transform_test)
    gallery = ImageDataset(dataset.gallery, transform=transform_test)
    combined = ConcatDataset([query, gallery])
    return DataLoader(
        combined,
        batch_size=cfg.DATA.TEST_BATCH,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )


def logger_for(path: Path):
    logger = logging.getLogger(f"phase4r_d_two_stage.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def make_optimizer_for(model: torch.nn.Module, selection: str) -> torch.optim.Optimizer:
    params = []
    selected_names = []
    for name, parameter in model.named_parameters():
        if selection == "reverse":
            keep = name.startswith(REVERSE_PREFIX)
        elif selection == "shared":
            keep = not name.startswith(REVERSE_PREFIX)
        elif selection == "all":
            keep = True
        else:
            raise ValueError(selection)
        if not keep:
            continue
        lr = 2e-6
        weight_decay = 0.05
        if "bias" in name:
            lr *= 2.0
            weight_decay = 0.05
        params.append({"params": [parameter], "lr": lr, "weight_decay": weight_decay})
        selected_names.append(name)
    if not params:
        raise RuntimeError(f"empty optimizer selection: {selection}")
    optimizer = torch.optim.SGD(params, momentum=0.9)
    optimizer._phase4r_selected_names = selected_names
    return optimizer


def make_scaler(scale: float) -> amp.GradScaler:
    return amp.GradScaler(
        init_scale=float(scale),
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )


def set_trainability(model: torch.nn.Module, shared: bool, reverse: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(reverse if name.startswith(REVERSE_PREFIX) else shared)


def factor_head_prefix(factor: str) -> str:
    if factor == "hair":
        return f"{REVERSE_PREFIX}0."
    if factor == "clothing":
        return f"{REVERSE_PREFIX}1."
    raise ValueError(factor)


def set_factor_trainability(model: torch.nn.Module, shared: bool, factor: str | None) -> None:
    """Set shared parameters and optionally only one reverse head trainable."""
    prefix = factor_head_prefix(factor) if factor is not None else None
    for name, parameter in model.named_parameters():
        if name.startswith(REVERSE_PREFIX):
            parameter.requires_grad_(prefix is not None and name.startswith(prefix))
        else:
            parameter.requires_grad_(shared)


def reverse_losses(score: dict[str, torch.Tensor], captions: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    reverse = score["clip_bio_reverse_score"]
    text = captions[:, 1:1 + reverse.shape[1]].to(device=reverse.device, dtype=reverse.dtype)
    losses = []
    accuracies = []
    for factor in range(reverse.shape[1]):
        image = F.normalize(reverse[:, factor], dim=-1)
        target = F.normalize(text[:, factor], dim=-1)
        logits = image @ target.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(F.cross_entropy(logits, labels))
        accuracies.append((logits.argmax(dim=-1) == labels).float().mean())
    return losses, accuracies


def forward_components(
    model,
    core_loss_fn,
    batch: dict[str, Any],
    device: torch.device,
    step: int,
):
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    writer = ScalarCapture()
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    core = core_loss_fn(
        score,
        feat,
        pids,
        camids,
        captions,
        clothes_ids=clothes,
        train_writer=writer,
        step=step,
    )
    reverse, reverse_acc = reverse_losses(score, captions)
    reverse_total = sum(reverse)
    components = dict(getattr(core_loss_fn, "last_components", {}))
    components["reverse_hair"] = reverse[0]
    components["reverse_clothing"] = reverse[1]
    return {
        "score": score,
        "core": core,
        "reverse_hair": reverse[0],
        "reverse_clothing": reverse[1],
        "reverse_total": reverse_total,
        "hair_accuracy": reverse_acc[0],
        "clothing_accuracy": reverse_acc[1],
        "id_accuracy": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_accuracy": writer.last("acc/clip_bio_0"),
        "ce_loss": writer.last("loss/id"),
        "triplet_loss": writer.last("loss/triplet"),
        "bio_loss": writer.last("loss/clip_bio_0"),
        "components": components,
    }


def forward_reverse_only(model, batch: dict[str, Any], device: torch.device):
    images = batch["image"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, _feat = model(images, cam_label=camids)
    losses, accuracies = reverse_losses(score, captions)
    return losses, accuracies


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def optimizer_found_inf(scaler: amp.GradScaler, optimizer: torch.optim.Optimizer) -> float:
    state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    return sum(float(value.item()) for value in state.values())


def finite_grad_audit(model: torch.nn.Module, names: Iterable[str]) -> dict[str, Any]:
    names = list(names)
    parameters = dict(model.named_parameters())
    present = 0
    finite = 0
    nonzero = 0
    nonfinite_names = []
    for name in names:
        grad = parameters[name].grad
        if grad is None:
            continue
        present += 1
        if bool(torch.isfinite(grad).all().item()):
            finite += 1
        else:
            nonfinite_names.append(name)
        if bool(torch.count_nonzero(grad).item() > 0):
            nonzero += 1
    return {
        "parameter_count": len(names),
        "grad_present": present,
        "finite": finite,
        "nonzero": nonzero,
        "nonfinite_names": nonfinite_names[:32],
    }


def evaluate_ltcc(cfg: Any, model: torch.nn.Module, dataset: Any, loader: Any, device: torch.device, epoch: int, logger):
    model.eval()
    cc_eval = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    general_eval = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    cc_writer = ScalarCapture()
    general_writer = ScalarCapture()
    cc_eval.reset()
    general_eval.reset()
    test(
        cfg, model, cc_eval, loader, logger, str(device), epoch, cc_writer,
        test=True, ltcc=True, clothChanging=True, metric_prefix="cc",
    )
    test(
        cfg, model, general_eval, loader, logger, str(device), epoch, general_writer,
        test=True, ltcc=False, clothChanging=False, metric_prefix="general",
    )
    model.train()
    return {
        "cc_rank1": cc_writer.last("cc/rank1"),
        "cc_rank5": cc_writer.last("cc/rank5"),
        "cc_rank10": cc_writer.last("cc/rank10"),
        "cc_mAP": cc_writer.last("cc/mAP"),
        "general_rank1": general_writer.last("general/rank1"),
        "general_rank5": general_writer.last("general/rank5"),
        "general_rank10": general_writer.last("general/rank10"),
        "general_mAP": general_writer.last("general/mAP"),
    }


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def to_cpu(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def save_candidate_state(
    path: Path,
    candidate: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: amp.GradScaler | None,
    epoch: int,
    global_step: int,
    device: torch.device,
    shared_optimizer: torch.optim.Optimizer | None = None,
    reverse_optimizer: torch.optim.Optimizer | None = None,
    shared_scheduler: Any | None = None,
    shared_scaler: amp.GradScaler | None = None,
    reverse_scaler: amp.GradScaler | None = None,
) -> None:
    payload: dict[str, Any] = {
        "phase": "4R-D",
        "track": "repair/research; two-stage full recovery",
        "candidate": candidate,
        "epoch": epoch,
        "global_step": global_step,
        "model": cpu_state_dict(model),
        "rng": rng_state(device),
        "anchor_manifest_sha256": sha256_file(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
        "plan_manifest_sha256": sha256_file(PLAN_MANIFEST),
    }
    if candidate in ("D0", "D1"):
        payload.update({
            "optimizer": to_cpu(optimizer.state_dict()) if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": to_cpu(scaler.state_dict()) if scaler is not None else None,
        })
    else:
        payload.update({
            "shared_optimizer": to_cpu(shared_optimizer.state_dict()) if shared_optimizer is not None else None,
            "reverse_optimizer": to_cpu(reverse_optimizer.state_dict()) if reverse_optimizer is not None else None,
            "shared_scheduler": shared_scheduler.state_dict() if shared_scheduler is not None else None,
            "shared_scaler": to_cpu(shared_scaler.state_dict()) if shared_scaler is not None else None,
            "reverse_scaler": to_cpu(reverse_scaler.state_dict()) if reverse_scaler is not None else None,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_candidate_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: amp.GradScaler | None,
    device: torch.device,
    shared_optimizer: torch.optim.Optimizer | None = None,
    reverse_optimizer: torch.optim.Optimizer | None = None,
    shared_scheduler: Any | None = None,
    shared_scaler: amp.GradScaler | None = None,
    reverse_scaler: amp.GradScaler | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if shared_optimizer is not None:
        shared_optimizer.load_state_dict(payload["shared_optimizer"])
    if reverse_optimizer is not None and payload.get("reverse_optimizer") is not None:
        reverse_optimizer.load_state_dict(payload["reverse_optimizer"])
    if shared_scheduler is not None:
        shared_scheduler.load_state_dict(payload["shared_scheduler"])
    if shared_scaler is not None:
        shared_scaler.load_state_dict(payload["shared_scaler"])
    if reverse_scaler is not None and payload.get("reverse_scaler") is not None:
        reverse_scaler.load_state_dict(payload["reverse_scaler"])
    restore_rng(payload["rng"], device)
    return payload


def reduce_epoch(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}


def reduce_amp(local_skips: float, local_nonfinite: float, scale_min: float, scale_max: float, scale_end: float, device: torch.device):
    sums = torch.tensor([local_skips, local_nonfinite, scale_end], dtype=torch.float64, device=device)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    mins = torch.tensor([scale_min], dtype=torch.float64, device=device)
    maxs = torch.tensor([scale_max], dtype=torch.float64, device=device)
    dist.all_reduce(mins, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
    return {
        "skips": int(round(float(sums[0].cpu()))),
        "nonfinite_events": int(round(float(sums[1].cpu()))),
        "scale_min": float(mins[0].cpu()),
        "scale_max": float(maxs[0].cpu()),
        "scale_end": float(sums[2].cpu()) / WORLD_SIZE,
    }


def batch_fingerprint(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("pid", "camid", "clothes_id", "augmentation_seed"):
        value = batch[key]
        if torch.is_tensor(value):
            value = value.cpu().numpy().tobytes()
        else:
            value = np.asarray(value).tobytes()
        digest.update(value)
    digest.update(batch["image"].contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def candidate_dir(candidate: str) -> Path:
    try:
        return CANDIDATE_DIRS[candidate]
    except KeyError:
        raise ValueError(candidate)


def validate_anchor_and_plan() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not (ANCHOR_DIR / "ANCHOR_COMPLETE.json").exists():
        raise RuntimeError("anchor is not complete; run --mode anchor")
    if not PLAN_MANIFEST.exists():
        raise RuntimeError("batch plan is not prepared; run --mode prepare")
    anchor_manifest = json.loads((ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json").read_text())
    if anchor_manifest.get("anchor_name") != "BIO_E27_ANCHOR":
        raise RuntimeError("unexpected anchor manifest")
    if anchor_manifest.get("source_checkpoint_sha256") != EXPECTED_BIO_SHA256:
        raise RuntimeError("anchor source SHA256 does not match frozen Bio E27")
    return load_anchor_state(), anchor_manifest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibration() -> None:
    rank = init_dist("nccl", require_cuda=True)
    device = torch.device(f"cuda:{rank}")
    run_dir = candidate_dir("D2") / "calibration"
    try:
        anchor_state, anchor_manifest = validate_anchor_and_plan()
        if rank == 0:
            if (run_dir / "CALIBRATION_COMPLETE.json").exists():
                raise RuntimeError("D2 calibration is already complete; refusing to overwrite")
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"refusing to overwrite non-empty calibration directory: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        cfg = configure_full()
        set_seed(SEED)
        train_loader, dataset, selected = planned_loader(cfg, rank, SCREEN_START_EPOCH, SCREEN_START_EPOCH + CALIBRATION_EPOCHS - 1)
        raw = build_raw_model(cfg, anchor_state, device)
        raw.train()
        shared_before = state_hashes(shared_keys(cpu_state_dict(raw)))
        reverse_optimizer = make_optimizer_for(raw, "reverse")
        reverse_scaler = make_scaler(INITIAL_SCALE_FORMAL)
        reverse_names = list(getattr(reverse_optimizer, "_phase4r_selected_names"))
        set_trainability(raw, shared=False, reverse=True)
        # The model has an explicit switch for this path.  Keep it enabled
        # during calibration so the calibration objective is literally a
        # detached-feature reverse-head update, not merely a frozen-parameter
        # approximation.
        raw.detach_reverse_input = True
        ddp = DDP(raw, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
        logger = logger_for(run_dir / f"train_log_rank{rank}.txt")
        writer = SummaryWriter(str(run_dir / "train")) if rank == 0 else None
        fieldnames = [
            "epoch", "hair_loss", "clothing_loss", "reverse_total",
            "hair_accuracy", "clothing_accuracy", "amp_scale_min",
            "amp_scale_max", "amp_scale_end", "amp_skips", "nonfinite_events",
        ]
        metrics_file = (run_dir / "metrics_per_epoch.csv").open("w", newline="") if rank == 0 else None
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=fieldnames) if rank == 0 else None
        if metrics_csv is not None:
            metrics_csv.writeheader()
            metrics_file.flush()
        rows = []
        event_rows = []
        iterator = iter(train_loader)
        global_step = (SCREEN_START_EPOCH - 1) * STEPS_PER_EPOCH
        epoch_sums = None
        local_skips = 0.0
        local_nonfinite = 0.0
        scale_min = float(reverse_scaler.get_scale())
        scale_max = scale_min
        for local_step in range(CALIBRATION_EPOCHS * STEPS_PER_EPOCH):
            epoch = SCREEN_START_EPOCH + local_step // STEPS_PER_EPOCH
            iteration = local_step % STEPS_PER_EPOCH + 1
            if iteration == 1:
                epoch_sums = {
                    "hair_loss": 0.0,
                    "clothing_loss": 0.0,
                    "reverse_total": 0.0,
                    "hair_accuracy": 0.0,
                    "clothing_accuracy": 0.0,
                }
                local_skips = 0.0
                local_nonfinite = 0.0
                scale_min = float(reverse_scaler.get_scale())
                scale_max = scale_min
            batch = next(iterator)
            global_step += 1
            reverse_optimizer.zero_grad(set_to_none=True)
            losses, accuracies = forward_reverse_only(ddp, batch, device)
            reverse_total = sum(losses)
            finite_loss = bool(torch.isfinite(reverse_total.detach()).item())
            local_nonfinite += float(not finite_loss)
            reverse_scaler.scale(reverse_total).backward()
            reverse_scaler.unscale_(reverse_optimizer)
            finite_before = finite_grad_audit(raw, reverse_names)
            scale_before = float(reverse_scaler.get_scale())
            reverse_scaler.step(reverse_optimizer)
            found_inf = optimizer_found_inf(reverse_scaler, reverse_optimizer)
            reverse_scaler.update()
            scale_after = float(reverse_scaler.get_scale())
            local_skips += float(found_inf > 0)
            local_nonfinite += float(found_inf > 0)
            scale_min = min(scale_min, scale_after)
            scale_max = max(scale_max, scale_after)
            for key, value in {
                "hair_loss": losses[0],
                "clothing_loss": losses[1],
                "reverse_total": reverse_total,
                "hair_accuracy": accuracies[0],
                "clothing_accuracy": accuracies[1],
            }.items():
                epoch_sums[key] += scalar(value)
            event_rows.append({
                "global_step": global_step,
                "epoch": epoch,
                "iteration": iteration,
                "reverse_loss_finite": finite_loss,
                "reverse_grad_finite": finite_before["finite"],
                "reverse_grad_nonzero": finite_before["nonzero"],
                "scale_before": scale_before,
                "scale_after": scale_after,
                "amp_skip": int(found_inf > 0),
            })
            if writer:
                writer.add_scalar("calibration/reverse_total", scalar(reverse_total), global_step)
                writer.add_scalar("calibration/hair_loss", scalar(losses[0]), global_step)
                writer.add_scalar("calibration/clothing_loss", scalar(losses[1]), global_step)
                writer.add_scalar("calibration/hair_accuracy", scalar(accuracies[0]), global_step)
                writer.add_scalar("calibration/clothing_accuracy", scalar(accuracies[1]), global_step)
            if iteration == STEPS_PER_EPOCH:
                reduced = reduce_epoch(
                    {key: value / STEPS_PER_EPOCH for key, value in epoch_sums.items()},
                    device,
                )
                amp_stats = reduce_amp(
                    local_skips,
                    local_nonfinite,
                    scale_min,
                    scale_max,
                    float(reverse_scaler.get_scale()),
                    device,
                )
                row = {
                    "epoch": epoch,
                    **reduced,
                    "amp_scale_min": amp_stats["scale_min"],
                    "amp_scale_max": amp_stats["scale_max"],
                    "amp_scale_end": amp_stats["scale_end"],
                    "amp_skips": amp_stats["skips"],
                    "nonfinite_events": amp_stats["nonfinite_events"],
                }
                rows.append(row)
                if rank == 0:
                    metrics_csv.writerow(row)
                    metrics_file.flush()
                    logger.info(
                        "Calibration epoch %d: hair %.6f / cloth %.6f; accuracy %.4f / %.4f; skips %d",
                        epoch, row["hair_loss"], row["clothing_loss"],
                        row["hair_accuracy"], row["clothing_accuracy"], row["amp_skips"],
                    )
                    for key, value in row.items():
                        if key != "epoch":
                            writer.add_scalar(f"calibration/epoch_{key}", float(value), epoch)

        final_state = cpu_state_dict(raw)
        shared_after = state_hashes(shared_keys(final_state))
        shared_unchanged = shared_before == shared_after
        if not shared_unchanged:
            raise RuntimeError("D2 calibration changed shared parameters")
        reverse_state = reverse_keys(final_state)
        rank_state_path = run_dir / f"rank{rank}_calibration_state_e27.pt"
        torch.save({
            "phase": "4R-D",
            "track": "repair/research; D2 reverse calibration",
            "candidate": "D2",
            "stage": "A_reverse_head_calibration",
            "epoch": ANCHOR_EPOCH,
            "calibration_epochs": CALIBRATION_EPOCHS,
            "global_step": global_step,
            "model": final_state,
            "reverse_optimizer": to_cpu(reverse_optimizer.state_dict()),
            "reverse_scaler": to_cpu(reverse_scaler.state_dict()),
            "rng": rng_state(device),
            "anchor_manifest_sha256": sha256_file(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
            "plan_manifest_sha256": sha256_file(PLAN_MANIFEST),
        }, rank_state_path)
        rank_info = {
            "rank": rank,
            "state": str(rank_state_path),
            "sha256": sha256_file(rank_state_path),
            "shared_unchanged": shared_unchanged,
            "reverse_state_digest": state_digest(reverse_state),
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, rank_info)
        dist.barrier()
        if rank == 0:
            reverse_path = run_dir / "CALIBRATED_REVERSE_E27.pth"
            torch.save(reverse_state, reverse_path)
            (run_dir / "reverse_hashes.json").write_text(json.dumps(state_hashes(reverse_state), indent=2, sort_keys=True) + "\n")
            (run_dir / "config.yml").write_text(cfg.dump())
            (run_dir / "events.json").write_text(json.dumps(event_rows, indent=2, sort_keys=True) + "\n")
            manifest = {
                "phase": "4R-D",
                "track": "repair/research; D2 calibrated alternating adversary",
                "candidate": "D2",
                "stage": "A_reverse_head_calibration",
                "created_utc": utc_now(),
                "anchor": "BIO_E27_ANCHOR",
                "anchor_manifest_sha256": sha256_file(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
                "reverse_init": str(ANCHOR_DIR / "REVERSE_INIT_E27.pth"),
                "calibration_epochs": CALIBRATION_EPOCHS,
                "data_epochs": [SCREEN_START_EPOCH, SCREEN_START_EPOCH + CALIBRATION_EPOCHS - 1],
                "data_plan_rows": len(selected),
                "reverse_optimizer": "independent SGD, fixed LR 2e-6, weight_decay 0.05",
                "reverse_scaler": "independent dynamic GradScaler, initial scale 1024",
                "shared_frozen": True,
                "shared_parameter_hash_before": state_digest(shared_keys(anchor_state)),
                "shared_parameter_hash_after": state_digest(shared_keys(final_state)),
                "shared_unchanged": shared_unchanged,
                "reverse_file": str(reverse_path),
                "reverse_file_sha256": sha256_file(reverse_path),
                "rank_states": gathered,
                "metrics": str(run_dir / "metrics_per_epoch.csv"),
                "forbidden": ["loss-weight sweep", "reverse LR sweep", "calibration epoch sweep"],
            }
            (run_dir / "CALIBRATION_COMPLETE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n")
        dist.barrier()
    finally:
        try:
            if "metrics_file" in locals() and metrics_file is not None:
                metrics_file.close()
            if "writer" in locals() and writer is not None:
                writer.close()
        finally:
            cleanup_dist()


def _screen_fieldnames() -> list[str]:
    return [
        "candidate", "epoch", "train_loss", "core_loss", "ce_loss",
        "triplet_loss", "bio_loss", "hair_loss", "clothing_loss",
        "id_accuracy", "bio_accuracy", "hair_accuracy", "clothing_accuracy",
        "stage_a_hair_loss", "stage_a_clothing_loss", "stage_a_reverse_total",
        "stage_a_hair_accuracy", "stage_a_clothing_accuracy",
        "lr", "lr_bias", "reverse_lr",
        "shared_scale_min", "shared_scale_max", "shared_scale_end",
        "shared_skips", "shared_nonfinite_events",
        "reverse_scale_min", "reverse_scale_max", "reverse_scale_end",
        "reverse_skips", "reverse_nonfinite_events",
        "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP",
        "general_rank1", "general_rank5", "general_rank10", "general_mAP",
    ]


def _telemetry_fieldnames() -> list[str]:
    return [
        "candidate", "optimization_phase",
        "epoch", "iteration", "rank", "group", "scaler_scale",
        "core_norm_pre_scaled", "hair_norm_pre_scaled",
        "clothing_norm_pre_scaled", "reverse_norm_pre_scaled",
        "core_norm_scaled", "hair_norm_scaled", "clothing_norm_scaled",
        "reverse_norm_scaled", "core_norm_unscaled", "hair_norm_unscaled",
        "clothing_norm_unscaled", "reverse_norm_unscaled",
        "cos_core_hair", "cos_core_clothing", "cos_hair_clothing",
        "cos_core_reverse", "hair_to_core_norm_ratio",
        "clothing_to_core_norm_ratio", "reverse_to_core_norm_ratio",
        "hair_nonzero", "clothing_nonzero", "reverse_nonzero",
        "hair_conflict", "clothing_conflict", "reverse_conflict",
        "parameter_count",
    ]


def _open_csv(path: Path, fieldnames: list[str], append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    stream = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    if not exists or not append:
        writer.writeheader()
        stream.flush()
    return stream, writer


def _lr_pair(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    values = sorted(float(group["lr"]) for group in optimizer.param_groups)
    return values[0], values[-1]


def _screen_preflight(candidate: str, start_epoch: int, end_epoch: int, resume: bool) -> None:
    if candidate not in CANDIDATE_DIRS:
        raise ValueError(candidate)
    if not (SCREEN_START_EPOCH <= start_epoch <= end_epoch <= FORMAL_EPOCHS):
        raise ValueError(f"screen range must be {SCREEN_START_EPOCH}..{FORMAL_EPOCHS}")
    run_dir = candidate_dir(candidate)
    if not resume and start_epoch != SCREEN_START_EPOCH:
        raise ValueError("a non-resume screen must start at epoch 28")
    if resume and start_epoch <= SCREEN_START_EPOCH:
        raise ValueError("resume requires start epoch > 28")
    if not resume:
        marker = run_dir / f"RUN_COMPLETE_E{end_epoch}.json"
        if marker.exists():
            raise RuntimeError(f"candidate run already complete: {marker}")
        if run_dir.exists() and any(run_dir.iterdir()):
            allowed = candidate == "D2" and all(item.name == "calibration" for item in run_dir.iterdir())
            if not allowed:
                raise RuntimeError(f"refusing to overwrite non-empty candidate directory: {run_dir}")
    else:
        for rank in range(WORLD_SIZE):
            state_path = run_dir / f"rank{rank}_state_e{start_epoch - 1}.pt"
            if not state_path.exists():
                raise FileNotFoundError(f"missing resume state: {state_path}")
    if candidate == "D2" and not (run_dir / "calibration" / "CALIBRATION_COMPLETE.json").exists():
        raise RuntimeError("D2 calibration is incomplete; run --mode calibrate first")


def _screen_runner(candidate: str, start_epoch: int, end_epoch: int, resume: bool) -> None:
    _screen_preflight(candidate, start_epoch, end_epoch, resume)
    rank = init_dist("nccl", require_cuda=True)
    device = torch.device(f"cuda:{rank}")
    run_dir = candidate_dir(candidate)
    metrics_path = run_dir / "metrics_per_epoch.csv"
    telemetry_path = run_dir / f"gradient_telemetry_rank{rank}.csv"
    amp_events_path = run_dir / f"amp_events_rank{rank}.jsonl"
    metrics_file = None
    telemetry_file = None
    amp_events_file = None
    writer = None
    try:
        anchor_state, anchor_manifest = validate_anchor_and_plan()
        cfg = configure_full()
        set_seed(SEED)
        train_loader, dataset, selected_plan = planned_loader(cfg, rank, start_epoch, end_epoch)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        calibration_payload = None
        if candidate == "D2":
            calibration_path = run_dir / "calibration" / f"rank{rank}_calibration_state_e27.pt"
            calibration_payload = torch.load(calibration_path, map_location="cpu", weights_only=False)
            initial_state = calibration_payload["model"]
        else:
            initial_state = anchor_state

        raw = build_raw_model(cfg, initial_state, device)
        raw.train()
        if candidate in ("D0", "D1"):
            optimizer = make_optimizer_for(raw, "all")
            scheduler = create_scheduler(cfg, optimizer)
            scaler = make_scaler(float(anchor_manifest["reconstructed_runtime_state"]["grad_scaler"]["reconstructed_scale_at_e27"]))
            shared_optimizer = reverse_optimizer = None
            shared_scheduler = shared_scaler = reverse_scaler = None
            all_names = list(getattr(optimizer, "_phase4r_selected_names"))
            shared_names = [name for name in all_names if not name.startswith(REVERSE_PREFIX)]
            reverse_names = [name for name in all_names if name.startswith(REVERSE_PREFIX)]
            if resume:
                resume_path = run_dir / f"rank{rank}_state_e{start_epoch - 1}.pt"
                resume_payload = load_candidate_state(
                    resume_path, raw, optimizer, scheduler, scaler, device,
                )
                # The previous run intentionally leaves the scheduler at the
                # last completed epoch.  Move it to the first epoch of this
                # continuation before consuming the resumed plan slice.
                scheduler.step(start_epoch)
                global_step = int(resume_payload["global_step"])
            else:
                scheduler.step(ANCHOR_EPOCH)
                global_step = (start_epoch - 1) * STEPS_PER_EPOCH
        else:
            shared_optimizer = make_optimizer_for(raw, "shared")
            reverse_optimizer = make_optimizer_for(raw, "reverse")
            shared_scheduler = create_scheduler(cfg, shared_optimizer)
            shared_scaler = make_scaler(float(anchor_manifest["reconstructed_runtime_state"]["grad_scaler"]["reconstructed_scale_at_e27"]))
            reverse_scaler = make_scaler(INITIAL_SCALE_FORMAL)
            optimizer = scheduler = scaler = None
            shared_names = list(getattr(shared_optimizer, "_phase4r_selected_names"))
            reverse_names = list(getattr(reverse_optimizer, "_phase4r_selected_names"))
            all_names = shared_names + reverse_names
            if resume:
                resume_path = run_dir / f"rank{rank}_state_e{start_epoch - 1}.pt"
                resume_payload = load_candidate_state(
                    resume_path, raw, None, None, None, device,
                    shared_optimizer=shared_optimizer,
                    reverse_optimizer=reverse_optimizer,
                    shared_scheduler=shared_scheduler,
                    shared_scaler=shared_scaler,
                    reverse_scaler=reverse_scaler,
                )
                shared_scheduler.step(start_epoch)
                global_step = int(resume_payload["global_step"])
            else:
                if calibration_payload is None:
                    raise RuntimeError("internal D2 calibration state missing")
                reverse_optimizer.load_state_dict(calibration_payload["reverse_optimizer"])
                reverse_scaler.load_state_dict(calibration_payload["reverse_scaler"])
                shared_scheduler.step(ANCHOR_EPOCH)
                global_step = (start_epoch - 1) * STEPS_PER_EPOCH

        raw.train()
        if rank == 0:
            (run_dir / "config.yml").write_text(cfg.dump())
            git_diff = git_output("git", "diff")
            (run_dir / "git_diff.patch").write_text(git_diff + ("\n" if git_diff and not git_diff.endswith("\n") else ""))
            if not resume:
                manifest = {
                    "phase": "4R-D",
                    "track": "repair/research; LTCC Full recovery",
                    "candidate": candidate,
                    "created_utc": utc_now(),
                    "anchor": "BIO_E27_ANCHOR",
                    "anchor_manifest_sha256": sha256_file(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
                    "initial_state_digest": state_digest(initial_state),
                    "data_plan_manifest_sha256": sha256_file(PLAN_MANIFEST),
                    "data_plan_slice": {
                        "start_epoch": start_epoch,
                        "end_epoch": end_epoch,
                        "rows": len(selected_plan),
                        "rank": rank,
                    },
                    "optimizer_state_at_anchor": "RECONSTRUCTED; exact formal E27 momentum unavailable",
                    "scheduler": "formal 60-epoch cosine; positioned at E27 before E28",
                    "loss_weights": {"hair": 1.0, "clothing": 1.0},
                    "grl_alpha": 1.0,
                    "nobio_index": [1, 2],
                    "candidate_protocol": {
                        "D0": "CE + Triplet + Bio; reverse heads present but inactive",
                        "D1": "standard Full from E28: CE + Triplet + Bio + Hair Reverse + Clothing Reverse",
                        "D2": "3-epoch detached reverse calibration, then two-step alternating min-max",
                    }[candidate],
                }
                (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        logger = logger_for(run_dir / f"train_log_rank{rank}.txt")
        writer = SummaryWriter(str(run_dir / "train")) if rank == 0 else None
        append_metrics = resume and metrics_path.exists()
        if rank == 0:
            metrics_file, metrics_csv = _open_csv(metrics_path, _screen_fieldnames(), append_metrics)
        else:
            metrics_csv = None
        telemetry_file, telemetry_csv = _open_csv(telemetry_path, _telemetry_fieldnames(), telemetry_path.exists())
        amp_events_file = amp_events_path.open("a" if amp_events_path.exists() else "w")

        # D2 calibration must have left the shared E27 feature extractor
        # unchanged.  The evaluation also verifies that the expanded topology
        # produces the frozen Bio E27 retrieval result.
        if not resume:
            anchor_loader = eval_loader(cfg, dataset) if rank == 0 else None
            if rank == 0:
                anchor_eval = evaluate_ltcc(cfg, raw, dataset, anchor_loader, device, ANCHOR_EPOCH, logger)
                expected = anchor_manifest["formal_e27_metrics"]
                anchor_eval_audit = {
                    "evaluation": anchor_eval,
                    "formal_e27": {
                        "cc_rank1": float(expected["cc_rank1"]),
                        "cc_mAP": float(expected["cc_mAP"]),
                    },
                    "delta": {
                        "cc_rank1": anchor_eval["cc_rank1"] - float(expected["cc_rank1"]),
                        "cc_mAP": anchor_eval["cc_mAP"] - float(expected["cc_mAP"]),
                    },
                }
                (run_dir / "anchor_eval.json").write_text(json.dumps(anchor_eval_audit, indent=2, sort_keys=True) + "\n")
                if abs(anchor_eval_audit["delta"]["cc_rank1"]) > 0.01 or abs(anchor_eval_audit["delta"]["cc_mAP"]) > 0.01:
                    raise RuntimeError(f"anchor evaluation is not the frozen Bio E27 result: {anchor_eval_audit}")
            dist.barrier()
        else:
            anchor_loader = None

        core_loss_fn, _center = make_loss(configure_core(), num_classes=dataset.num_train_pids)
        ddp = DDP(raw, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
        iterator = iter(train_loader)
        total_steps = (end_epoch - start_epoch + 1) * STEPS_PER_EPOCH
        epoch_sums = None
        amp_state = None
        step_rows = []
        best_rank1 = -float("inf")
        best_epoch = None
        if append_metrics:
            with metrics_path.open() as stream:
                previous = list(csv.DictReader(stream))
            if previous:
                best = max(previous, key=lambda row: float(row["cc_rank1"]))
                best_rank1 = float(best["cc_rank1"])
                best_epoch = int(best["epoch"])

        for local_step in range(total_steps):
            epoch = start_epoch + local_step // STEPS_PER_EPOCH
            iteration = local_step % STEPS_PER_EPOCH + 1
            if iteration == 1:
                epoch_sums = {
                    key: 0.0 for key in (
                        "train_loss", "core_loss", "ce_loss", "triplet_loss", "bio_loss",
                        "hair_loss", "clothing_loss", "id_accuracy", "bio_accuracy",
                        "hair_accuracy", "clothing_accuracy", "stage_a_hair_loss",
                        "stage_a_clothing_loss", "stage_a_reverse_total",
                        "stage_a_hair_accuracy", "stage_a_clothing_accuracy",
                    )
                }
                amp_state = {
                    "shared_skips": 0.0, "shared_nonfinite": 0.0,
                    "shared_scale_min": float(shared_scaler.get_scale()) if candidate == "D2" else float(scaler.get_scale()),
                    "shared_scale_max": float(shared_scaler.get_scale()) if candidate == "D2" else float(scaler.get_scale()),
                    "reverse_skips": 0.0, "reverse_nonfinite": 0.0,
                    "reverse_scale_min": float(reverse_scaler.get_scale()) if candidate == "D2" else float("nan"),
                    "reverse_scale_max": float(reverse_scaler.get_scale()) if candidate == "D2" else float("nan"),
                }
            batch = next(iterator)
            expected_global_step = (start_epoch - 1) * STEPS_PER_EPOCH + local_step + 1
            if global_step + 1 != expected_global_step:
                raise RuntimeError(f"global step mismatch: state {global_step + 1}, plan {expected_global_step}")
            global_step += 1

            if candidate in ("D0", "D1"):
                set_trainability(raw, shared=True, reverse=True)
                raw.detach_reverse_input = False
                optimizer.zero_grad(set_to_none=True)
                result = forward_components(ddp, core_loss_fn, batch, device, global_step)
                total = result["core"] if candidate == "D0" else result["core"] + result["reverse_total"]
                if iteration % 100 == 0:
                    for telemetry_row in collect_component_gradient_telemetry(
                        raw, result["components"], float(scaler.get_scale()), epoch, iteration, rank,
                    ):
                        telemetry_row["candidate"] = candidate
                        telemetry_row["optimization_phase"] = "global_standard"
                        telemetry_csv.writerow(telemetry_row)
                    telemetry_file.flush()
                scale_before = float(scaler.get_scale())
                loss_finite = bool(torch.isfinite(total.detach()).item())
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                gradient_audit = {
                    "shared": finite_grad_audit(raw, shared_names),
                    "reverse": finite_grad_audit(raw, reverse_names),
                }
                scaler.step(optimizer)
                found_inf = optimizer_found_inf(scaler, optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                amp_state["shared_skips"] += float(found_inf > 0)
                amp_state["shared_nonfinite"] += float((not loss_finite) or found_inf > 0)
                amp_state["shared_scale_min"] = min(amp_state["shared_scale_min"], scale_after)
                amp_state["shared_scale_max"] = max(amp_state["shared_scale_max"], scale_after)
                step_result = result
                step_result["total"] = total
                step_result["stage_a"] = None
                step_result["scale_before"] = scale_before
                step_result["scale_after"] = scale_after
                step_result["shared_skip"] = int(found_inf > 0)
                step_result["reverse_skip"] = None
                step_result["gradient_audit"] = gradient_audit
            else:
                # STEP A: update only the two adversary heads with detached
                # shared features.  The shared model is frozen for this step.
                set_trainability(raw, shared=False, reverse=True)
                raw.detach_reverse_input = True
                shared_optimizer.zero_grad(set_to_none=True)
                reverse_optimizer.zero_grad(set_to_none=True)
                reverse_losses_a, reverse_acc_a = forward_reverse_only(ddp, batch, device)
                reverse_total_a = sum(reverse_losses_a)
                scale_before_a = float(reverse_scaler.get_scale())
                finite_a = bool(torch.isfinite(reverse_total_a.detach()).item())
                reverse_scaler.scale(reverse_total_a).backward()
                reverse_scaler.unscale_(reverse_optimizer)
                audit_a = {"reverse": finite_grad_audit(raw, reverse_names)}
                reverse_scaler.step(reverse_optimizer)
                found_a = optimizer_found_inf(reverse_scaler, reverse_optimizer)
                reverse_scaler.update()
                scale_after_a = float(reverse_scaler.get_scale())
                amp_state["reverse_skips"] += float(found_a > 0)
                amp_state["reverse_nonfinite"] += float((not finite_a) or found_a > 0)
                amp_state["reverse_scale_min"] = min(amp_state["reverse_scale_min"], scale_after_a)
                amp_state["reverse_scale_max"] = max(amp_state["reverse_scale_max"], scale_after_a)

                # STEP B: freeze reverse-head parameters but retain the
                # reverse computational path, so GRL sends its gradient to the
                # shared representation through a fixed classifier mapping.
                set_trainability(raw, shared=True, reverse=False)
                raw.detach_reverse_input = False
                shared_optimizer.zero_grad(set_to_none=True)
                result = forward_components(ddp, core_loss_fn, batch, device, global_step)
                total = result["core"] + result["reverse_total"]
                if iteration % 100 == 0:
                    for telemetry_row in collect_component_gradient_telemetry(
                        raw, result["components"], float(shared_scaler.get_scale()), epoch, iteration, rank,
                    ):
                        telemetry_row["candidate"] = candidate
                        telemetry_row["optimization_phase"] = "step_b_shared"
                        telemetry_csv.writerow(telemetry_row)
                    telemetry_file.flush()
                scale_before_b = float(shared_scaler.get_scale())
                finite_b = bool(torch.isfinite(total.detach()).item())
                shared_scaler.scale(total).backward()
                shared_scaler.unscale_(shared_optimizer)
                audit_b = {"shared": finite_grad_audit(raw, shared_names)}
                shared_scaler.step(shared_optimizer)
                found_b = optimizer_found_inf(shared_scaler, shared_optimizer)
                shared_scaler.update()
                scale_after_b = float(shared_scaler.get_scale())
                amp_state["shared_skips"] += float(found_b > 0)
                amp_state["shared_nonfinite"] += float((not finite_b) or found_b > 0)
                amp_state["shared_scale_min"] = min(amp_state["shared_scale_min"], scale_after_b)
                amp_state["shared_scale_max"] = max(amp_state["shared_scale_max"], scale_after_b)
                step_result = result
                step_result["total"] = total
                step_result["stage_a"] = {
                    "hair_loss": reverse_losses_a[0],
                    "clothing_loss": reverse_losses_a[1],
                    "reverse_total": reverse_total_a,
                    "hair_accuracy": reverse_acc_a[0],
                    "clothing_accuracy": reverse_acc_a[1],
                    "scale_before": scale_before_a,
                    "scale_after": scale_after_a,
                    "amp_skip": int(found_a > 0),
                    "gradient_audit": audit_a,
                }
                step_result["scale_before"] = scale_before_b
                step_result["scale_after"] = scale_after_b
                step_result["shared_skip"] = int(found_b > 0)
                step_result["reverse_skip"] = int(found_a > 0)
                step_result["gradient_audit"] = audit_b

            for key, value in {
                "train_loss": step_result["total"],
                "core_loss": step_result["core"],
                "ce_loss": step_result["ce_loss"],
                "triplet_loss": step_result["triplet_loss"],
                "bio_loss": step_result["bio_loss"],
                "hair_loss": step_result["reverse_hair"],
                "clothing_loss": step_result["reverse_clothing"],
                "id_accuracy": step_result["id_accuracy"],
                "bio_accuracy": step_result["bio_accuracy"],
                "hair_accuracy": step_result["hair_accuracy"],
                "clothing_accuracy": step_result["clothing_accuracy"],
            }.items():
                epoch_sums[key] += scalar(value)
            if step_result["stage_a"] is not None:
                for key, value in {
                    "stage_a_hair_loss": step_result["stage_a"]["hair_loss"],
                    "stage_a_clothing_loss": step_result["stage_a"]["clothing_loss"],
                    "stage_a_reverse_total": step_result["stage_a"]["reverse_total"],
                    "stage_a_hair_accuracy": step_result["stage_a"]["hair_accuracy"],
                    "stage_a_clothing_accuracy": step_result["stage_a"]["clothing_accuracy"],
                }.items():
                    epoch_sums[key] += scalar(value)

            event = {
                "candidate": candidate,
                "global_step": global_step,
                "epoch": epoch,
                "iteration": iteration,
                "train_loss_finite": bool(torch.isfinite(step_result["total"].detach()).item()),
                "scale_before": step_result["scale_before"],
                "scale_after": step_result["scale_after"],
                "shared_skip": step_result["shared_skip"],
                "reverse_skip": step_result["reverse_skip"],
                "gradient_audit": step_result["gradient_audit"],
            }
            if step_result["stage_a"] is not None:
                event["stage_a"] = {
                    "scale_before": step_result["stage_a"]["scale_before"],
                    "scale_after": step_result["stage_a"]["scale_after"],
                    "amp_skip": step_result["stage_a"]["amp_skip"],
                    "gradient_audit": step_result["stage_a"]["gradient_audit"],
                }
            amp_events_file.write(json.dumps(event, sort_keys=True, default=json_default) + "\n")
            if iteration == 1 or iteration % 100 == 0:
                amp_events_file.flush()
            if writer:
                writer.add_scalar("loss/total", scalar(step_result["total"]), global_step)
                writer.add_scalar("loss/core", scalar(step_result["core"]), global_step)
                writer.add_scalar("loss/hair", scalar(step_result["reverse_hair"]), global_step)
                writer.add_scalar("loss/clothing", scalar(step_result["reverse_clothing"]), global_step)
                writer.add_scalar("accuracy/id", scalar(step_result["id_accuracy"]), global_step)
                writer.add_scalar("accuracy/bio", scalar(step_result["bio_accuracy"]), global_step)
                writer.add_scalar("accuracy/hair", scalar(step_result["hair_accuracy"]), global_step)
                writer.add_scalar("accuracy/clothing", scalar(step_result["clothing_accuracy"]), global_step)

            if iteration == STEPS_PER_EPOCH:
                reduced = reduce_epoch(
                    {key: value / STEPS_PER_EPOCH for key, value in epoch_sums.items()},
                    device,
                )
                if candidate == "D2":
                    shared_amp = reduce_amp(
                        amp_state["shared_skips"], amp_state["shared_nonfinite"],
                        amp_state["shared_scale_min"], amp_state["shared_scale_max"],
                        float(shared_scaler.get_scale()), device,
                    )
                    reverse_amp = reduce_amp(
                        amp_state["reverse_skips"], amp_state["reverse_nonfinite"],
                        amp_state["reverse_scale_min"], amp_state["reverse_scale_max"],
                        float(reverse_scaler.get_scale()), device,
                    )
                    shared_sched = shared_scheduler
                    active_optimizer = shared_optimizer
                else:
                    shared_amp = reduce_amp(
                        amp_state["shared_skips"], amp_state["shared_nonfinite"],
                        amp_state["shared_scale_min"], amp_state["shared_scale_max"],
                        float(scaler.get_scale()), device,
                    )
                    reverse_amp = None
                    shared_sched = scheduler
                    active_optimizer = optimizer
                dist.barrier()
                eval_loader_current = eval_loader(cfg, dataset) if rank == 0 else None
                if rank == 0:
                    epoch_eval = evaluate_ltcc(cfg, raw, dataset, eval_loader_current, device, epoch, logger)
                else:
                    epoch_eval = None
                eval_obj = [epoch_eval]
                dist.broadcast_object_list(eval_obj, src=0)
                epoch_eval = eval_obj[0]
                lr, lr_bias = _lr_pair(active_optimizer)
                row = {
                    "candidate": candidate,
                    "epoch": epoch,
                    **reduced,
                    "stage_a_hair_loss": reduced["stage_a_hair_loss"] if candidate == "D2" else float("nan"),
                    "stage_a_clothing_loss": reduced["stage_a_clothing_loss"] if candidate == "D2" else float("nan"),
                    "stage_a_reverse_total": reduced["stage_a_reverse_total"] if candidate == "D2" else float("nan"),
                    "stage_a_hair_accuracy": reduced["stage_a_hair_accuracy"] if candidate == "D2" else float("nan"),
                    "stage_a_clothing_accuracy": reduced["stage_a_clothing_accuracy"] if candidate == "D2" else float("nan"),
                    "lr": lr,
                    "lr_bias": lr_bias,
                    "reverse_lr": float(reverse_optimizer.param_groups[0]["lr"]) if candidate == "D2" else float("nan"),
                    "shared_scale_min": shared_amp["scale_min"],
                    "shared_scale_max": shared_amp["scale_max"],
                    "shared_scale_end": shared_amp["scale_end"],
                    "shared_skips": shared_amp["skips"],
                    "shared_nonfinite_events": shared_amp["nonfinite_events"],
                    "reverse_scale_min": reverse_amp["scale_min"] if reverse_amp else float("nan"),
                    "reverse_scale_max": reverse_amp["scale_max"] if reverse_amp else float("nan"),
                    "reverse_scale_end": reverse_amp["scale_end"] if reverse_amp else float("nan"),
                    "reverse_skips": reverse_amp["skips"] if reverse_amp else float("nan"),
                    "reverse_nonfinite_events": reverse_amp["nonfinite_events"] if reverse_amp else float("nan"),
                    **epoch_eval,
                }
                step_rows.append(row)
                if writer:
                    for key, value in epoch_eval.items():
                        writer.add_scalar(f"eval/{key}", value, epoch)
                if rank == 0:
                    metrics_csv.writerow(row)
                    metrics_file.flush()
                    logger.info(
                        "%s E%d: CC %.4f / %.4f; General %.4f / %.4f; shared skips %s; reverse skips %s",
                        candidate, epoch, row["cc_rank1"], row["cc_mAP"],
                        row["general_rank1"], row["general_mAP"],
                        row["shared_skips"], row["reverse_skips"],
                    )
                    if row["cc_rank1"] > best_rank1:
                        best_rank1 = row["cc_rank1"]
                        best_epoch = epoch
                        torch.save(cpu_state_dict(raw), run_dir / "best_model.pth")
                        (run_dir / "BEST.json").write_text(json.dumps({
                            "candidate": candidate,
                            "epoch": epoch,
                            "cc_rank1": row["cc_rank1"],
                            "cc_mAP": row["cc_mAP"],
                            "checkpoint": str(run_dir / "best_model.pth"),
                            "checkpoint_sha256": sha256_file(run_dir / "best_model.pth"),
                        }, indent=2, sort_keys=True) + "\n")
                    if writer:
                        writer.add_scalar("eval/best_cc_rank1", best_rank1, epoch)

                if candidate == "D2":
                    shared_scheduler.step(epoch + 1) if epoch < end_epoch else None
                else:
                    scheduler.step(epoch + 1) if epoch < end_epoch else None

        final_model_path = run_dir / f"model_final_e{end_epoch}.pth"
        if rank == 0:
            torch.save(cpu_state_dict(raw), final_model_path)
        if candidate in ("D0", "D1"):
            rank_state_path = run_dir / f"rank{rank}_state_e{end_epoch}.pt"
            save_candidate_state(
                rank_state_path, candidate, raw, optimizer, scheduler, scaler,
                end_epoch, global_step, device,
            )
        else:
            rank_state_path = run_dir / f"rank{rank}_state_e{end_epoch}.pt"
            save_candidate_state(
                rank_state_path, candidate, raw, None, None, None,
                end_epoch, global_step, device,
                shared_optimizer=shared_optimizer,
                reverse_optimizer=reverse_optimizer,
                shared_scheduler=shared_scheduler,
                shared_scaler=shared_scaler,
                reverse_scaler=reverse_scaler,
            )
        rank_info = {
            "rank": rank,
            "state": str(rank_state_path),
            "sha256": sha256_file(rank_state_path),
            "model_state_digest": state_digest(cpu_state_dict(raw)),
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, rank_info)
        dist.barrier()
        if rank == 0:
            marker = {
                "phase": "4R-D",
                "track": "repair/research; LTCC Full recovery",
                "candidate": candidate,
                "created_utc": utc_now(),
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "global_step": global_step,
                "rank_states": gathered,
                "model_final": str(final_model_path),
                "model_final_sha256": sha256_file(final_model_path),
                "best_epoch": best_epoch,
                "best_rank1": best_rank1,
                "metrics": str(metrics_path),
            }
            (run_dir / f"RUN_COMPLETE_E{end_epoch}.json").write_text(json.dumps(marker, indent=2, sort_keys=True, default=json_default) + "\n")
        dist.barrier()
    finally:
        try:
            if metrics_file is not None:
                metrics_file.close()
            if telemetry_file is not None:
                telemetry_file.close()
            if amp_events_file is not None:
                amp_events_file.close()
            if writer is not None:
                writer.close()
        finally:
            cleanup_dist()


def factor_probe(factor: str) -> None:
    """Run one fixed E28 D2-style probe with only one reverse factor active."""
    if factor not in PROBE_DIRS:
        raise ValueError(factor)
    run_dir = PROBE_DIRS[factor]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty factor probe directory: {run_dir}")
    calibration_dir = candidate_dir("D2") / "calibration"
    if not (calibration_dir / "CALIBRATION_COMPLETE.json").exists():
        raise RuntimeError("D2 calibration is incomplete; run --mode calibrate first")

    rank = init_dist("nccl", require_cuda=True)
    device = torch.device(f"cuda:{rank}")
    metrics_file = None
    telemetry_file = None
    amp_events_file = None
    writer = None
    try:
        anchor_state, anchor_manifest = validate_anchor_and_plan()
        cfg = configure_full()
        set_seed(SEED)
        train_loader, dataset, selected_plan = planned_loader(
            cfg, rank, SCREEN_START_EPOCH, SCREEN_START_EPOCH
        )
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        calibration_payload = torch.load(
            calibration_dir / f"rank{rank}_calibration_state_e27.pt",
            map_location="cpu",
            weights_only=False,
        )
        initial_state = calibration_payload["model"]
        initial_reverse = reverse_keys(initial_state)
        inactive_prefix = factor_head_prefix("clothing" if factor == "hair" else "hair")
        inactive_initial = {
            key: value for key, value in initial_reverse.items()
            if key.startswith(inactive_prefix)
        }

        raw = build_raw_model(cfg, initial_state, device)
        raw.train()
        shared_optimizer = make_optimizer_for(raw, "shared")
        reverse_optimizer = make_optimizer_for(raw, "reverse")
        shared_scheduler = create_scheduler(cfg, shared_optimizer)
        shared_scheduler.step(ANCHOR_EPOCH)
        shared_scaler = make_scaler(
            float(anchor_manifest["reconstructed_runtime_state"]["grad_scaler"]["reconstructed_scale_at_e27"])
        )
        reverse_scaler = make_scaler(INITIAL_SCALE_FORMAL)
        reverse_optimizer.load_state_dict(calibration_payload["reverse_optimizer"])
        reverse_scaler.load_state_dict(calibration_payload["reverse_scaler"])
        shared_names = list(getattr(shared_optimizer, "_phase4r_selected_names"))
        reverse_names = list(getattr(reverse_optimizer, "_phase4r_selected_names"))
        active_names = [name for name in reverse_names if name.startswith(factor_head_prefix(factor))]
        inactive_names = [name for name in reverse_names if name.startswith(inactive_prefix)]
        if not active_names or not inactive_names:
            raise RuntimeError(f"factor probe parameter split failed for {factor}")

        set_factor_trainability(raw, shared=False, factor=factor)
        raw.detach_reverse_input = True
        ddp = DDP(raw, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
        logger = logger_for(run_dir / f"train_log_rank{rank}.txt")
        writer = SummaryWriter(str(run_dir / "train")) if rank == 0 else None
        metrics_path = run_dir / "metrics_per_epoch.csv"
        telemetry_path = run_dir / f"gradient_telemetry_rank{rank}.csv"
        amp_events_path = run_dir / f"amp_events_rank{rank}.jsonl"
        if rank == 0:
            metrics_file, metrics_csv = _open_csv(metrics_path, _screen_fieldnames(), append=False)
        else:
            metrics_csv = None
        telemetry_file, telemetry_csv = _open_csv(telemetry_path, _telemetry_fieldnames(), append=False)
        amp_events_file = amp_events_path.open("w")

        anchor_loader = eval_loader(cfg, dataset) if rank == 0 else None
        if rank == 0:
            anchor_eval = evaluate_ltcc(cfg, raw, dataset, anchor_loader, device, ANCHOR_EPOCH, logger)
            expected = anchor_manifest["formal_e27_metrics"]
            audit = {
                "evaluation": anchor_eval,
                "formal_e27": {
                    "cc_rank1": float(expected["cc_rank1"]),
                    "cc_mAP": float(expected["cc_mAP"]),
                },
                "delta": {
                    "cc_rank1": anchor_eval["cc_rank1"] - float(expected["cc_rank1"]),
                    "cc_mAP": anchor_eval["cc_mAP"] - float(expected["cc_mAP"]),
                },
            }
            (run_dir / "anchor_eval.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
            if abs(audit["delta"]["cc_rank1"]) > 0.01 or abs(audit["delta"]["cc_mAP"]) > 0.01:
                raise RuntimeError(f"factor probe anchor evaluation mismatch: {audit}")
            manifest = {
                "phase": "4R-D",
                "track": "repair/research; factor attribution probe",
                "candidate": f"D2_{factor.title()}Only",
                "created_utc": utc_now(),
                "active_factor": factor,
                "inactive_factor": "clothing" if factor == "hair" else "hair",
                "anchor": "BIO_E27_ANCHOR",
                "anchor_manifest_sha256": sha256_file(ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json"),
                "calibration_manifest_sha256": sha256_file(calibration_dir / "CALIBRATION_COMPLETE.json"),
                "data_plan_manifest_sha256": sha256_file(PLAN_MANIFEST),
                "data_plan_slice": {"start_epoch": 28, "end_epoch": 28, "rows": len(selected_plan), "rank": rank},
                "protocol": "D2 alternating min-max; Step A selected reverse head only, Step B core + selected reverse with GRL alpha 1",
                "loss_weights": {"selected_factor": 1.0, "inactive_factor": 0.0},
                "shared_optimizer": "independent SGD; reconstructed E27 scheduler position",
                "reverse_optimizer": "calibrated D2 reverse optimizer/scaler; inactive head frozen",
                "active_reverse_parameter_count": len(active_names),
                "inactive_reverse_parameter_count": len(inactive_names),
            }
            (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        dist.barrier()

        core_loss_fn, _center = make_loss(configure_core(), num_classes=dataset.num_train_pids)
        iterator = iter(train_loader)
        global_step = (SCREEN_START_EPOCH - 1) * STEPS_PER_EPOCH
        epoch_sums = {
            key: 0.0 for key in (
                "train_loss", "core_loss", "ce_loss", "triplet_loss", "bio_loss",
                "hair_loss", "clothing_loss", "id_accuracy", "bio_accuracy",
                "hair_accuracy", "clothing_accuracy", "stage_a_hair_loss",
                "stage_a_clothing_loss", "stage_a_reverse_total", "stage_a_hair_accuracy",
                "stage_a_clothing_accuracy",
            )
        }
        amp_state = {
            "shared_skips": 0.0,
            "shared_nonfinite": 0.0,
            "shared_scale_min": float(shared_scaler.get_scale()),
            "shared_scale_max": float(shared_scaler.get_scale()),
            "reverse_skips": 0.0,
            "reverse_nonfinite": 0.0,
            "reverse_scale_min": float(reverse_scaler.get_scale()),
            "reverse_scale_max": float(reverse_scaler.get_scale()),
        }

        factor_index = 0 if factor == "hair" else 1
        for iteration in range(1, STEPS_PER_EPOCH + 1):
            batch = next(iterator)
            global_step += 1

            # STEP A: only the selected adversary learns from a detached
            # feature.  The other reverse head is present but inactive.
            set_factor_trainability(raw, shared=False, factor=factor)
            raw.detach_reverse_input = True
            shared_optimizer.zero_grad(set_to_none=True)
            reverse_optimizer.zero_grad(set_to_none=True)
            reverse_losses_a, reverse_acc_a = forward_reverse_only(ddp, batch, device)
            reverse_total_a = reverse_losses_a[factor_index]
            finite_a = bool(torch.isfinite(reverse_total_a.detach()).item())
            scale_before_a = float(reverse_scaler.get_scale())
            reverse_scaler.scale(reverse_total_a).backward()
            reverse_scaler.unscale_(reverse_optimizer)
            audit_a = {
                "active": finite_grad_audit(raw, active_names),
                "inactive": finite_grad_audit(raw, inactive_names),
            }
            reverse_scaler.step(reverse_optimizer)
            found_a = optimizer_found_inf(reverse_scaler, reverse_optimizer)
            reverse_scaler.update()
            scale_after_a = float(reverse_scaler.get_scale())
            amp_state["reverse_skips"] += float(found_a > 0)
            amp_state["reverse_nonfinite"] += float((not finite_a) or found_a > 0)
            amp_state["reverse_scale_min"] = min(amp_state["reverse_scale_min"], scale_after_a)
            amp_state["reverse_scale_max"] = max(amp_state["reverse_scale_max"], scale_after_a)

            # STEP B: both reverse heads are frozen; only the selected reverse
            # component is included in the shared objective for attribution.
            set_factor_trainability(raw, shared=True, factor=None)
            raw.detach_reverse_input = False
            shared_optimizer.zero_grad(set_to_none=True)
            result = forward_components(ddp, core_loss_fn, batch, device, global_step)
            selected_reverse = result["reverse_hair" if factor == "hair" else "reverse_clothing"]
            total = result["core"] + selected_reverse
            if iteration % 100 == 0:
                for telemetry_row in collect_component_gradient_telemetry(
                    raw, result["components"], float(shared_scaler.get_scale()),
                    SCREEN_START_EPOCH, iteration, rank,
                ):
                    telemetry_row["candidate"] = f"D2_{factor.title()}Only"
                    telemetry_row["optimization_phase"] = "step_b_shared_factor_probe"
                    telemetry_csv.writerow(telemetry_row)
                telemetry_file.flush()
            finite_b = bool(torch.isfinite(total.detach()).item())
            scale_before_b = float(shared_scaler.get_scale())
            shared_scaler.scale(total).backward()
            shared_scaler.unscale_(shared_optimizer)
            audit_b = {
                "shared": finite_grad_audit(raw, shared_names),
                "reverse": finite_grad_audit(raw, reverse_names),
            }
            shared_scaler.step(shared_optimizer)
            found_b = optimizer_found_inf(shared_scaler, shared_optimizer)
            shared_scaler.update()
            scale_after_b = float(shared_scaler.get_scale())
            amp_state["shared_skips"] += float(found_b > 0)
            amp_state["shared_nonfinite"] += float((not finite_b) or found_b > 0)
            amp_state["shared_scale_min"] = min(amp_state["shared_scale_min"], scale_after_b)
            amp_state["shared_scale_max"] = max(amp_state["shared_scale_max"], scale_after_b)

            for key, value in {
                "train_loss": total,
                "core_loss": result["core"],
                "ce_loss": result["ce_loss"],
                "triplet_loss": result["triplet_loss"],
                "bio_loss": result["bio_loss"],
                "hair_loss": result["reverse_hair"],
                "clothing_loss": result["reverse_clothing"],
                "id_accuracy": result["id_accuracy"],
                "bio_accuracy": result["bio_accuracy"],
                "hair_accuracy": result["hair_accuracy"],
                "clothing_accuracy": result["clothing_accuracy"],
                "stage_a_hair_loss": reverse_losses_a[0],
                "stage_a_clothing_loss": reverse_losses_a[1],
                "stage_a_reverse_total": reverse_total_a,
                "stage_a_hair_accuracy": reverse_acc_a[0],
                "stage_a_clothing_accuracy": reverse_acc_a[1],
            }.items():
                epoch_sums[key] += scalar(value)
            event = {
                "candidate": f"D2_{factor.title()}Only",
                "active_factor": factor,
                "global_step": global_step,
                "epoch": SCREEN_START_EPOCH,
                "iteration": iteration,
                "train_loss_finite": bool(torch.isfinite(total.detach()).item()),
                "scale_before": scale_before_b,
                "scale_after": scale_after_b,
                "shared_skip": int(found_b > 0),
                "reverse_skip": int(found_a > 0),
                "gradient_audit": audit_b,
                "stage_a": {
                    "scale_before": scale_before_a,
                    "scale_after": scale_after_a,
                    "amp_skip": int(found_a > 0),
                    "gradient_audit": audit_a,
                },
            }
            amp_events_file.write(json.dumps(event, sort_keys=True, default=json_default) + "\n")
            if iteration == 1 or iteration % 100 == 0:
                amp_events_file.flush()
            if writer:
                writer.add_scalar("loss/total", scalar(total), global_step)
                writer.add_scalar("loss/core", scalar(result["core"]), global_step)
                writer.add_scalar("loss/hair", scalar(result["reverse_hair"]), global_step)
                writer.add_scalar("loss/clothing", scalar(result["reverse_clothing"]), global_step)
                writer.add_scalar("accuracy/hair", scalar(result["hair_accuracy"]), global_step)
                writer.add_scalar("accuracy/clothing", scalar(result["clothing_accuracy"]), global_step)

        reduced = reduce_epoch(
            {key: value / STEPS_PER_EPOCH for key, value in epoch_sums.items()}, device
        )
        shared_amp = reduce_amp(
            amp_state["shared_skips"], amp_state["shared_nonfinite"],
            amp_state["shared_scale_min"], amp_state["shared_scale_max"],
            float(shared_scaler.get_scale()), device,
        )
        reverse_amp = reduce_amp(
            amp_state["reverse_skips"], amp_state["reverse_nonfinite"],
            amp_state["reverse_scale_min"], amp_state["reverse_scale_max"],
            float(reverse_scaler.get_scale()), device,
        )
        dist.barrier()
        eval_loader_current = eval_loader(cfg, dataset) if rank == 0 else None
        if rank == 0:
            epoch_eval = evaluate_ltcc(cfg, raw, dataset, eval_loader_current, device, SCREEN_START_EPOCH, logger)
        else:
            epoch_eval = None
        eval_obj = [epoch_eval]
        dist.broadcast_object_list(eval_obj, src=0)
        epoch_eval = eval_obj[0]
        lr, lr_bias = _lr_pair(shared_optimizer)
        row = {
            "candidate": f"D2_{factor.title()}Only",
            "epoch": SCREEN_START_EPOCH,
            **reduced,
            "lr": lr,
            "lr_bias": lr_bias,
            "reverse_lr": float(reverse_optimizer.param_groups[0]["lr"]),
            "shared_scale_min": shared_amp["scale_min"],
            "shared_scale_max": shared_amp["scale_max"],
            "shared_scale_end": shared_amp["scale_end"],
            "shared_skips": shared_amp["skips"],
            "shared_nonfinite_events": shared_amp["nonfinite_events"],
            "reverse_scale_min": reverse_amp["scale_min"],
            "reverse_scale_max": reverse_amp["scale_max"],
            "reverse_scale_end": reverse_amp["scale_end"],
            "reverse_skips": reverse_amp["skips"],
            "reverse_nonfinite_events": reverse_amp["nonfinite_events"],
            **epoch_eval,
        }
        if rank == 0:
            metrics_csv.writerow(row)
            metrics_file.flush()
            final_state = cpu_state_dict(raw)
            final_reverse = reverse_keys(final_state)
            inactive_final = {
                key: value for key, value in final_reverse.items()
                if key.startswith(inactive_prefix)
            }
            inactive_unchanged = state_hashes(inactive_initial) == state_hashes(inactive_final)
            torch.save(final_state, run_dir / "model_final_e28.pth")
            torch.save(final_state, run_dir / "best_model.pth")
            (run_dir / "BEST.json").write_text(json.dumps({
                "candidate": f"D2_{factor.title()}Only",
                "epoch": SCREEN_START_EPOCH,
                "cc_rank1": row["cc_rank1"],
                "cc_mAP": row["cc_mAP"],
                "checkpoint": str(run_dir / "best_model.pth"),
                "checkpoint_sha256": sha256_file(run_dir / "best_model.pth"),
            }, indent=2, sort_keys=True) + "\n")
            (run_dir / "FACTOR_PROBE_AUDIT.json").write_text(json.dumps({
                "active_factor": factor,
                "inactive_factor": "clothing" if factor == "hair" else "hair",
                "active_reverse_changed": state_hashes({
                    key: value for key, value in initial_reverse.items()
                    if key.startswith(factor_head_prefix(factor))
                }) != state_hashes({
                    key: value for key, value in final_reverse.items()
                    if key.startswith(factor_head_prefix(factor))
                }),
                "inactive_reverse_unchanged": inactive_unchanged,
                "shared_parameter_changed": state_digest(shared_keys(initial_state)) != state_digest(shared_keys(final_state)),
                "row": row,
            }, indent=2, sort_keys=True, default=json_default) + "\n")
            (run_dir / "PROBE_COMPLETE.json").write_text(json.dumps({
                "phase": "4R-D",
                "track": "repair/research; factor attribution probe",
                "candidate": f"D2_{factor.title()}Only",
                "factor": factor,
                "epoch": SCREEN_START_EPOCH,
                "metrics": str(metrics_path),
                "model_final": str(run_dir / "model_final_e28.pth"),
                "model_final_sha256": sha256_file(run_dir / "model_final_e28.pth"),
            }, indent=2, sort_keys=True) + "\n")
            logger.info(
                "%s factor probe E28: CC %.4f / %.4f; shared skips %s; reverse skips %s",
                factor, row["cc_rank1"], row["cc_mAP"], row["shared_skips"], row["reverse_skips"],
            )

        rank_state_path = run_dir / f"rank{rank}_state_e28.pt"
        save_candidate_state(
            rank_state_path, f"D2_{factor.title()}Only", raw, None, None, None,
            SCREEN_START_EPOCH, global_step, device,
            shared_optimizer=shared_optimizer,
            reverse_optimizer=reverse_optimizer,
            shared_scheduler=shared_scheduler,
            shared_scaler=shared_scaler,
            reverse_scaler=reverse_scaler,
        )
        rank_info = {
            "rank": rank,
            "state": str(rank_state_path),
            "sha256": sha256_file(rank_state_path),
            "model_state_digest": state_digest(cpu_state_dict(raw)),
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, rank_info)
        dist.barrier()
        if rank == 0:
            marker_path = run_dir / "PROBE_COMPLETE.json"
            marker = json.loads(marker_path.read_text())
            marker["rank_states"] = gathered
            marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        try:
            if metrics_file is not None:
                metrics_file.close()
            if telemetry_file is not None:
                telemetry_file.close()
            if amp_events_file is not None:
                amp_events_file.close()
            if writer is not None:
                writer.close()
        finally:
            cleanup_dist()


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open() as stream:
        rows = []
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                if key in ("candidate",):
                    parsed[key] = value
                elif key == "epoch":
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = float("nan")
            rows.append(parsed)
        return rows


def _mean_and_slope(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(np.arange(values.size), values, 1)[0]) if values.size > 1 else 0.0
    return float(values.mean()), slope


def _sustained_collapse(
    rows: list[dict[str, float]],
    reference_rows: list[dict[str, float]] | None = None,
    r1_floor: float = -0.01,
    map_floor: float = -0.008,
) -> bool:
    """Detect a material three-epoch collapse, optionally against D0.

    The earlier implementation compared positive retrieval scores to negative
    floors, which could never detect a collapse.  For the matched screen the
    meaningful quantity is the candidate-minus-D0 gap at the same epoch.
    """
    if reference_rows is not None:
        reference = {int(row["epoch"]): row for row in reference_rows}
        aligned = [
            (row, reference[int(row["epoch"])])
            for row in rows
            if int(row["epoch"]) in reference
        ]
        for left, middle, right in zip(aligned, aligned[1:], aligned[2:]):
            gaps = [
                (
                    float(candidate["cc_rank1"]) - float(control["cc_rank1"]),
                    float(candidate["cc_mAP"]) - float(control["cc_mAP"]),
                )
                for candidate, control in (left, middle, right)
            ]
            if all(r1 <= r1_floor and mAP <= map_floor for r1, mAP in gaps):
                return True
        return False

    for left, middle, right in zip(rows, rows[1:], rows[2:]):
        r1_drop = float(right["cc_rank1"]) - float(left["cc_rank1"])
        map_drop = float(right["cc_mAP"]) - float(left["cc_mAP"])
        if (
            float(middle["cc_rank1"]) <= float(left["cc_rank1"])
            and float(right["cc_rank1"]) <= float(middle["cc_rank1"])
            and float(middle["cc_mAP"]) <= float(left["cc_mAP"])
            and float(right["cc_mAP"]) <= float(middle["cc_mAP"])
            and r1_drop <= r1_floor
            and map_drop <= map_floor
        ):
            return True
    return False


def analyze() -> None:
    """Analyze the matched E28--E32 screening and write the repair report."""
    anchor_manifest = json.loads((ANCHOR_DIR / "BIO_E27_ANCHOR_MANIFEST.json").read_text())
    controls = {
        "Bio Original": {"start": "formal E27", "best": [56.3775539398, 32.3857364139]},
        "Original Full": {"start": "from scratch", "best": [53.3163249493, 29.2657488553]},
    }
    candidates = {}
    for candidate in ("D0", "D1", "D2"):
        path = candidate_dir(candidate) / "metrics_per_epoch.csv"
        if not path.exists():
            candidates[candidate] = {"status": "NOT RUN", "rows": []}
            continue
        rows = _read_rows(path)
        screen_rows = [row for row in rows if 28 <= int(row["epoch"]) <= 32]
        if not screen_rows:
            candidates[candidate] = {"status": "NO SCREEN ROWS", "rows": rows}
            continue
        best = max(screen_rows, key=lambda row: float(row["cc_rank1"]))
        mean_r1, slope_r1 = _mean_and_slope(screen_rows, "cc_rank1")
        mean_map, slope_map = _mean_and_slope(screen_rows, "cc_mAP")
        candidates[candidate] = {
            "status": "SCREENED",
            "rows": screen_rows,
            "best": {"epoch": int(best["epoch"]), "cc_rank1": float(best["cc_rank1"]), "cc_mAP": float(best["cc_mAP"])},
            "mean_cc_rank1": mean_r1,
            "mean_cc_mAP": mean_map,
            "slope_cc_rank1": slope_r1,
            "slope_cc_mAP": slope_map,
            "sustained_collapse": False,
        }

    factor_probes = {}
    for factor in ("hair", "clothing"):
        probe_path = PROBE_DIRS[factor] / "metrics_per_epoch.csv"
        audit_path = PROBE_DIRS[factor] / "FACTOR_PROBE_AUDIT.json"
        if not probe_path.exists() or not audit_path.exists():
            factor_probes[factor] = {"status": "NOT RUN", "rows": []}
            continue
        probe_rows = _read_rows(probe_path)
        audit = json.loads(audit_path.read_text())
        if not probe_rows:
            factor_probes[factor] = {"status": "NO PROBE ROWS", "rows": [], "audit": audit}
            continue
        factor_probes[factor] = {
            "status": "PROBED",
            "rows": probe_rows,
            "row": probe_rows[-1],
            "audit": audit,
        }

    d0_rows_for_collapse = candidates.get("D0", {}).get("rows", [])
    for candidate in ("D0", "D1", "D2"):
        item = candidates.get(candidate, {})
        if item.get("status") != "SCREENED":
            continue
        if candidate == "D0":
            item["sustained_collapse"] = _sustained_collapse(item["rows"])
        elif candidates.get("D0", {}).get("status") == "SCREENED":
            item["sustained_collapse"] = _sustained_collapse(item["rows"], d0_rows_for_collapse)

    def delta(left: str, right: str) -> dict[str, float] | None:
        if candidates.get(left, {}).get("status") != "SCREENED" or candidates.get(right, {}).get("status") != "SCREENED":
            return None
        return {
            "mean_r1": (candidates[left]["mean_cc_rank1"] - candidates[right]["mean_cc_rank1"]) * 100.0,
            "mean_mAP": (candidates[left]["mean_cc_mAP"] - candidates[right]["mean_cc_mAP"]) * 100.0,
            "final_r1": (candidates[left]["rows"][-1]["cc_rank1"] - candidates[right]["rows"][-1]["cc_rank1"]) * 100.0,
            "final_mAP": (candidates[left]["rows"][-1]["cc_mAP"] - candidates[right]["rows"][-1]["cc_mAP"]) * 100.0,
        }

    d1_vs_d0 = delta("D1", "D0")
    d2_vs_d0 = delta("D2", "D0")
    d2_vs_d1 = delta("D2", "D1")
    d2_vs_bio_best = None
    d2_vs_original_full_best = None
    if candidates.get("D2", {}).get("status") == "SCREENED":
        d2_best = candidates["D2"]["best"]
        d2_vs_bio_best = {
            "r1": (d2_best["cc_rank1"] - controls["Bio Original"]["best"][0] / 100.0) * 100.0,
            "mAP": (d2_best["cc_mAP"] - controls["Bio Original"]["best"][1] / 100.0) * 100.0,
            "epoch": d2_best["epoch"],
        }
        d2_vs_original_full_best = {
            "r1": (d2_best["cc_rank1"] - controls["Original Full"]["best"][0] / 100.0) * 100.0,
            "mAP": (d2_best["cc_mAP"] - controls["Original Full"]["best"][1] / 100.0) * 100.0,
            "epoch": d2_best["epoch"],
        }

    factor_probe_summary = {}
    d0_e28 = next(
        (row for row in candidates.get("D0", {}).get("rows", []) if int(row["epoch"]) == 28),
        None,
    )
    d2_e28 = next(
        (row for row in candidates.get("D2", {}).get("rows", []) if int(row["epoch"]) == 28),
        None,
    )
    for factor, item in factor_probes.items():
        if item.get("status") != "PROBED":
            continue
        row = item["row"]
        summary = {
            "epoch": int(row["epoch"]),
            "cc_rank1": float(row["cc_rank1"]),
            "cc_mAP": float(row["cc_mAP"]),
            "general_rank1": float(row["general_rank1"]),
            "general_mAP": float(row["general_mAP"]),
            "shared_skips": float(row["shared_skips"]),
            "reverse_skips": float(row["reverse_skips"]),
            "active_reverse_changed": bool(item["audit"].get("active_reverse_changed")),
            "inactive_reverse_unchanged": bool(item["audit"].get("inactive_reverse_unchanged")),
            "shared_parameter_changed": bool(item["audit"].get("shared_parameter_changed")),
        }
        if d0_e28 is not None:
            summary["vs_d0_e28"] = {
                "r1_pp": (summary["cc_rank1"] - float(d0_e28["cc_rank1"])) * 100.0,
                "mAP_pp": (summary["cc_mAP"] - float(d0_e28["cc_mAP"])) * 100.0,
            }
        if d2_e28 is not None:
            summary["vs_d2_e28"] = {
                "r1_pp": (summary["cc_rank1"] - float(d2_e28["cc_rank1"])) * 100.0,
                "mAP_pp": (summary["cc_mAP"] - float(d2_e28["cc_mAP"])) * 100.0,
            }
        factor_probe_summary[factor] = summary

    d2_available = candidates.get("D2", {}).get("status") == "SCREENED"
    d0_available = candidates.get("D0", {}).get("status") == "SCREENED"
    d1_available = candidates.get("D1", {}).get("status") == "SCREENED"
    d2_gate = None
    if d2_available and d0_available and d1_available:
        d2_gate = {
            "no_sustained_collapse": not candidates["D2"]["sustained_collapse"],
            "positive_vs_D1_on_both_mean_metrics": bool(d2_vs_d1["mean_r1"] > 0.0 and d2_vs_d1["mean_mAP"] > 0.0),
            "within_D0_r1_gap_1pp": bool(d2_vs_d0["mean_r1"] >= -1.0),
            "within_D0_map_gap_0p8pp": bool(d2_vs_d0["mean_mAP"] >= -0.8),
            "epoch32_not_obviously_worsening": bool(candidates["D2"]["slope_cc_rank1"] >= -0.005 and candidates["D2"]["slope_cc_mAP"] >= -0.004),
        }
        d2_gate["passed"] = all(d2_gate.values())

    # The report deliberately keeps screening and long-run status separate.
    if d2_gate is None:
        diagnosis = "SCREENING INCOMPLETE"
    elif d2_gate["passed"]:
        diagnosis = "TWO-STAGE ALTERNATING TRAINING PASSES SCREENING; LONG-RUN CONFIRMATION REQUIRED"
    elif (
        factor_probe_summary.get("hair", {}).get("cc_rank1") is not None
        and factor_probe_summary.get("clothing", {}).get("cc_rank1") is not None
        and factor_probe_summary["clothing"]["cc_rank1"]
        < factor_probe_summary["hair"]["cc_rank1"] - 0.005
        and factor_probe_summary["clothing"]["cc_mAP"]
        < factor_probe_summary["hair"]["cc_mAP"] - 0.005
    ):
        diagnosis = "NONBIO SEMANTIC TARGET REMAINS HARMFUL"
    elif d2_available and d2_vs_d0 is not None and d2_vs_d0["mean_r1"] > 0.0:
        diagnosis = "TWO-STAGE TRAINING ONLY PARTIALLY RECOVERS"
    elif d2_available and candidates["D2"]["sustained_collapse"]:
        diagnosis = "REPAIR FAILED"
    else:
        diagnosis = "REPAIR FAILED"

    repair_level = "NOT EVALUABLE"
    if d2_vs_bio_best is not None and d2_vs_original_full_best is not None:
        if (
            d2_vs_original_full_best["r1"] <= 0.0
            and d2_vs_original_full_best["mAP"] <= 0.0
        ):
            repair_level = "LEVEL 0 — FAIL"
        elif (
            d2_vs_original_full_best["r1"] > 0.0
            and d2_vs_original_full_best["mAP"] > 0.0
            and d2_vs_bio_best["r1"] < 0.0
            and d2_vs_bio_best["mAP"] < 0.0
        ):
            repair_level = "LEVEL 1 — PARTIAL REPAIR"
        elif (
            (
                candidates["D2"]["best"]["cc_rank1"] >= 0.5688
                and d2_vs_bio_best["mAP"] > -0.3
            )
            or (
                candidates["D2"]["best"]["cc_mAP"] >= 0.3269
                and d2_vs_bio_best["r1"] > -0.3
            )
        ):
            repair_level = "LEVEL 3 — NONBIO ADDS POSITIVE VALUE"
        elif (
            candidates["D2"]["best"]["cc_rank1"] >= 0.559
            and candidates["D2"]["best"]["cc_mAP"] >= 0.319
        ):
            repair_level = "LEVEL 2 — BIO NEIGHBORHOOD"
        else:
            repair_level = "LEVEL 1 — PARTIAL REPAIR"

    competence_summary = {}
    for candidate in ("D1", "D2"):
        item = candidates.get(candidate, {})
        if item.get("status") != "SCREENED":
            continue
        rows = item["rows"]
        reverse_skip_values = [float(row["reverse_skips"]) for row in rows]
        competence_summary[candidate] = {
            "hair_accuracy_first": float(rows[0]["hair_accuracy"]),
            "hair_accuracy_last": float(rows[-1]["hair_accuracy"]),
            "hair_accuracy_mean": float(np.mean([float(row["hair_accuracy"]) for row in rows])),
            "clothing_accuracy_first": float(rows[0]["clothing_accuracy"]),
            "clothing_accuracy_last": float(rows[-1]["clothing_accuracy"]),
            "clothing_accuracy_mean": float(np.mean([float(row["clothing_accuracy"]) for row in rows])),
            "shared_skips_total": float(np.nansum([float(row["shared_skips"]) for row in rows])),
            "reverse_skips_total": (
                None if not any(np.isfinite(reverse_skip_values))
                else float(np.nansum(reverse_skip_values))
            ),
        }
    calibration_rows = _read_rows(
        candidate_dir("D2") / "calibration" / "metrics_per_epoch.csv"
    ) if (candidate_dir("D2") / "calibration" / "metrics_per_epoch.csv").exists() else []

    report = [
        "# PHASE 4R-D — LTCC FULL TWO-STAGE RECOVERY",
        "",
        "Repair Track only. This report is not a paper-faithful reproduction result and does not modify formal Phase 3 artifacts.",
        "",
        "## Frozen protocol",
        "",
        f"- Anchor: `BIO_E27_ANCHOR`, formal Bio BEST epoch 27; source checkpoint SHA256 `{anchor_manifest['source_checkpoint_sha256']}`.",
        "- Shared objective weights remain nominal: CE + Triplet + Bio + Hair Reverse + Clothing Reverse, with GRL alpha 1 and factor weights 1.",
        "- D0/D1/D2 share the serialized batch and augmentation plan. D2 uses a separate reverse optimizer and scaler.",
        "- The formal E27 optimizer, GradScaler growth tracker, and RNG state were not saved; runtime state is reconstructed and explicitly marked in the anchor manifest.",
        "",
        "## Frozen controls",
        "",
        "| Setting | CC Rank-1 | CC mAP |",
        "|---|---:|---:|",
        "| Bio Control E27 | 56.3776 | 32.3857 |",
        "| Original Full | 53.3163 | 29.2657 |",
        "",
        "## Comparison with the paper",
        "",
        "These are context values only; all repair gates use the local Bio control, not the paper mAP value.",
        "",
        "| Setting | Paper R1 / mAP | Local R1 / mAP | Local − paper |",
        "|---|---:|---:|---:|",
        "| ID-only | 54.6000 / 30.4000 | 56.3776 / 31.0721 | +1.7776 / +0.6721 pp |",
        "| ID+Bio | 55.4000 / 30.8000 | 56.3776 / 32.3857 | +0.9776 / +1.5857 pp |",
        "| Full | 58.2000 / 31.6000 | 53.3163 / 29.2657 | −4.8837 / −2.3343 pp |",
        "",
        "## Screening results",
        "",
        "| Setting | Start | Reverse training | BEST R1 | BEST mAP | vs Bio | vs Original Full | Status |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    # Candidate metrics are stored as fractions; keep frozen controls in the
    # same units so the pp deltas below are not accidentally inflated by 100x.
    base_values = {
        "Bio Original": [0.563775539398, 0.323857364139],
        "Original Full": [0.533163249493, 0.292657488553],
    }
    report.append("| Bio Original | E27 anchor | none | 56.3776 | 32.3857 | — | +3.0613 / +3.1200 pp | frozen control |")
    report.append("| Original Full | from scratch | simultaneous | 53.3163 | 29.2657 | −3.0613 / −3.1200 pp | — | frozen failure |")
    descriptions = {
        "D0": ("E27", "inactive reverse heads"),
        "D1": ("E27", "standard Full, global optimizer/scaler"),
        "D2": ("E27", "3-epoch calibration + alternating min-max"),
    }
    for candidate in ("D0", "D1", "D2"):
        item = candidates[candidate]
        if item.get("status") != "SCREENED":
            report.append(f"| {candidate} | {descriptions[candidate][0]} | {descriptions[candidate][1]} | — | — | — | — | {item.get('status')} |")
            continue
        best = item["best"]
        vs_bio = f"{(best['cc_rank1'] - base_values['Bio Original'][0]) * 100:+.3f} / {(best['cc_mAP'] - base_values['Bio Original'][1]) * 100:+.3f} pp"
        vs_full = f"{(best['cc_rank1'] - base_values['Original Full'][0]) * 100:+.3f} / {(best['cc_mAP'] - base_values['Original Full'][1]) * 100:+.3f} pp"
        report.append(f"| {candidate} | {descriptions[candidate][0]} | {descriptions[candidate][1]} | {best['cc_rank1'] * 100:.4f} | {best['cc_mAP'] * 100:.4f} | {vs_bio} | {vs_full} | SCREENED |")
    report += [
        "",
        "## Trajectory statistics",
        "",
        "| Candidate | Mean R1 | Mean mAP | R1 slope/epoch | mAP slope/epoch | sustained collapse |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate in ("D0", "D1", "D2"):
        item = candidates[candidate]
        if item.get("status") != "SCREENED":
            report.append(f"| {candidate} | — | — | — | — | — |")
        else:
            report.append(f"| {candidate} | {item['mean_cc_rank1'] * 100:.4f} | {item['mean_cc_mAP'] * 100:.4f} | {item['slope_cc_rank1'] * 100:+.4f} pp | {item['slope_cc_mAP'] * 100:+.4f} pp | {item['sustained_collapse']} |")
    report += [
        "",
        "## BEST versus final screening epoch",
        "",
        "| Candidate | BEST epoch | BEST CC R1 / mAP | FINAL E32 CC R1 / mAP |",
        "|---|---:|---:|---:|",
    ]
    for candidate in ("D0", "D1", "D2"):
        item = candidates[candidate]
        if item.get("status") != "SCREENED":
            report.append(f"| {candidate} | — | — | — |")
            continue
        best = item["best"]
        final = item["rows"][-1]
        report.append(
            f"| {candidate} | {best['epoch']} | {best['cc_rank1'] * 100:.4f} / {best['cc_mAP'] * 100:.4f} | "
            f"{float(final['cc_rank1']) * 100:.4f} / {float(final['cc_mAP']) * 100:.4f} |"
        )
    report += [
        "",
        "## Matched effects",
        "",
        f"- D1 − D0: `{d1_vs_d0}`",
        f"- D2 − D0: `{d2_vs_d0}`",
        f"- D2 − D1: `{d2_vs_d1}`",
        "- Original Full − Bio Control: `−3.0613 / −3.1200 pp`.",
        f"- D2 BEST (CC Rank-1 epoch) − Bio Control: `{d2_vs_bio_best}`.",
        f"- D2 BEST (CC Rank-1 epoch) − Original Full: `{d2_vs_original_full_best}`.",
        "",
        "## Factor attribution probes",
        "",
        "These are the predeclared short probes after D2 screening failure. Each uses the calibrated D2 initialization and alternating framework for E28, but activates only the named reverse factor in both adversary and shared steps. They are attribution evidence, not a repaired Full result.",
        "",
        "| Probe | CC R1 | CC mAP | vs D2 E28 | vs D0 E28 | Active head updated | Inactive head unchanged | Status |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for factor in ("hair", "clothing"):
        item = factor_probe_summary.get(factor)
        if item is None:
            report.append(f"| {factor.title()}-only | — | — | — | — | — | — | NOT RUN |")
            continue
        vs_d2 = item.get("vs_d2_e28")
        vs_d0 = item.get("vs_d0_e28")
        vs_d2_text = (
            f"{vs_d2['r1_pp']:+.3f} / {vs_d2['mAP_pp']:+.3f} pp"
            if vs_d2 is not None else "—"
        )
        vs_d0_text = (
            f"{vs_d0['r1_pp']:+.3f} / {vs_d0['mAP_pp']:+.3f} pp"
            if vs_d0 is not None else "—"
        )
        report.append(
            f"| {factor.title()}-only | {item['cc_rank1'] * 100:.4f} | {item['cc_mAP'] * 100:.4f} | "
            f"{vs_d2_text} | {vs_d0_text} | {item['active_reverse_changed']} | "
            f"{item['inactive_reverse_unchanged']} | E28 probe |"
        )
    if "hair" in factor_probe_summary and "clothing" in factor_probe_summary:
        hair_probe = factor_probe_summary["hair"]
        clothing_probe = factor_probe_summary["clothing"]
        report.append(
            f"- Clothing-only is lower than Hair-only by "
            f"{(clothing_probe['cc_rank1'] - hair_probe['cc_rank1']) * 100:+.3f} pp R1 and "
            f"{(clothing_probe['cc_mAP'] - hair_probe['cc_mAP']) * 100:+.3f} pp mAP at E28. "
            "This makes Clothing the primary remaining suspect; it does not authorize deleting that factor and calling the result a repair."
        )
    report += [
        "",
        "## D2 screening gate",
        "",
        f"`{json.dumps(d2_gate, sort_keys=True) if d2_gate is not None else 'NOT EVALUABLE'}`",
        "",
        "The numerical D2 gate uses the predeclared D0 neighborhood limits (R1 gap no worse than 1.0 pp; mAP gap no worse than 0.8 pp), requires positive mean effect over D1 on both metrics, and requires no sustained collapse. Slopes are retained as a trend check; this is still only a five-epoch screen.",
        "",
        "## Adversary competence and AMP",
        "",
        "Hair/Clothing reverse accuracy and the independent AMP counters are retained here to distinguish a stable adversary from a useful adversary.",
        "",
        "| Candidate | Hair acc E28→E32 (mean) | Clothing acc E28→E32 (mean) | Shared skips | Reverse skips |",
        "|---|---|---|---:|---:|",
    ]
    for candidate in ("D1", "D2"):
        item = competence_summary.get(candidate)
        if item is None:
            report.append(f"| {candidate} | — | — | — | — |")
            continue
        reverse_skips = (
            "n/a" if item["reverse_skips_total"] is None
            else f"{item['reverse_skips_total']:.0f}"
        )
        report.append(
            f"| {candidate} | {item['hair_accuracy_first'] * 100:.2f}% → {item['hair_accuracy_last'] * 100:.2f}% ({item['hair_accuracy_mean'] * 100:.2f}%) | "
            f"{item['clothing_accuracy_first'] * 100:.2f}% → {item['clothing_accuracy_last'] * 100:.2f}% ({item['clothing_accuracy_mean'] * 100:.2f}%) | "
            f"{item['shared_skips_total']:.0f} | {reverse_skips} |"
        )
    if calibration_rows:
        calibration_first = calibration_rows[0]
        calibration_last = calibration_rows[-1]
        report.append(
            f"| D2 calibration (E28→E30) | {float(calibration_first['hair_accuracy']) * 100:.2f}% → {float(calibration_last['hair_accuracy']) * 100:.2f}% | "
            f"{float(calibration_first['clothing_accuracy']) * 100:.2f}% → {float(calibration_last['clothing_accuracy']) * 100:.2f}% | "
            f"— | {float(calibration_last['amp_skips']):.0f} |"
        )
    report += [
        "",
        "## Three questions",
        "",
        f"1. Q1 — Does late standard Full still degrade relative to D0? `{d1_vs_d0}`.",
        f"2. Q2 — Does calibrated alternating optimization remove the degradation at screening horizon? `{d2_vs_d0}`; gate `{d2_gate}`.",
        "3. Q3 — Does stabilized NonBio add value beyond Bio? No at this screening horizon: D2 remains below the Bio control on the matched BEST-by-R1 comparison, even though it improves over the original Full failure.",
        "",
        "## Diagnosis",
        "",
        f"**{diagnosis}**",
        "",
        "## Repair level",
        "",
        f"**{repair_level}**",
        "",
        "No paper mAP target is used as a reason to lower the local Bio control. All candidate claims must be reported against both Bio Control and Original Full.",
        "",
        "## Artifact index",
        "",
        f"- Anchor: `{ANCHOR_DIR}`",
        f"- Batch plan: `{PLAN_MANIFEST}`",
        *(f"- {candidate}: `{candidate_dir(candidate)}`" for candidate in ("D0", "D1", "D2")),
        *(f"- {factor.title()}-only factor probe: `{PROBE_DIRS[factor]}`" for factor in ("hair", "clothing")),
        "",
        "## Stop status",
        "",
        "This report covers matched E28–E32 screening only. Long-run E40/E60 continuation is authorized only by the D0/D2 screening gate and remains separate from the formal reproduction results.",
    ]
    report_path = ROOT / "reports/PHASE_4R_D_TWO_STAGE_FULL_RECOVERY.md"
    report_path.write_text("\n".join(report) + "\n")
    analysis = {
        "phase": "4R-D",
        "track": "repair/research; LTCC Full recovery",
        "created_utc": utc_now(),
        "anchor": anchor_manifest,
        "candidates": candidates,
        "factor_probes": factor_probes,
        "factor_probe_summary": factor_probe_summary,
        "competence_summary": competence_summary,
        "matched_effects": {"D1_vs_D0": d1_vs_d0, "D2_vs_D0": d2_vs_d0, "D2_vs_D1": d2_vs_d1},
        "d2_vs_bio_best": d2_vs_bio_best,
        "d2_vs_original_full_best": d2_vs_original_full_best,
        "d2_gate": d2_gate,
        "diagnosis": diagnosis,
        "repair_level": repair_level,
        "report": str(report_path),
    }
    (OUT / "SCREENING_ANALYSIS.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({"report": str(report_path), "diagnosis": diagnosis, "d2_gate": d2_gate}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("anchor", "prepare", "calibrate", "screen", "probe", "analyze"), required=True)
    parser.add_argument("--candidate", choices=("D0", "D1", "D2"), default=None)
    parser.add_argument("--factor", choices=("hair", "clothing"), default=None)
    parser.add_argument("--start-epoch", type=int, default=SCREEN_START_EPOCH)
    parser.add_argument("--end-epoch", type=int, default=SCREEN_END_EPOCH)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "anchor":
        create_anchor()
    elif args.mode == "prepare":
        prepare_plan()
    elif args.mode == "calibrate":
        calibration()
    elif args.mode == "screen":
        if args.candidate is None:
            raise SystemExit("--candidate is required for --mode screen")
        _screen_runner(args.candidate, args.start_epoch, args.end_epoch, args.resume)
    elif args.mode == "probe":
        if args.factor is None:
            raise SystemExit("--factor is required for --mode probe")
        factor_probe(args.factor)
    else:
        analyze()


if __name__ == "__main__":
    main()
