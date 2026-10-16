#!/usr/local/bin/python3
"""PHASE 4R-D: matched-prefix timing and clothing activation study.

This is a research-track experiment.  It is intentionally separate from the
formal reproduction artifacts.  A full DIFFER topology is created once, a
Bio-only prefix is trained through epoch 20, and four branches are resumed
from the exact same serialized state:

    B0: Bio only
    B1: Bio + Hair reverse
    B2: Bio + Clothing reverse
    B3: Bio + Hair + Clothing reverse

The script uses the released LTCC data/sampler semantics, a fixed 60-epoch
cosine scheduler, and deterministic per-sample augmentation seeds.  It does
not implement any repair.  D2 (the preregistered delayed-clothing candidate)
is deliberately a separate mode and is only allowed after the matched-prefix
analysis has passed its gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import faulthandler
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataloader, build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import read_image  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from model.build import factory  # noqa: E402
from processor.processor import test  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402

faulthandler.enable()

from scripts.phase4r_a2_forensics import (  # noqa: E402
    INITIAL_SCALE,
    ScalarCapture,
    set_seed,
    utc_now,
)


SEED = 1234
WORLD_SIZE = 2
STEPS_PER_EPOCH = 596
BATCH_SIZE = 8
PREFIX_EPOCHS = 20
BRANCH_END_EPOCH = 30
TOTAL_SCHEDULE_EPOCHS = 60
PREFIX_STEPS = PREFIX_EPOCHS * STEPS_PER_EPOCH
BRANCH_STEPS = (BRANCH_END_EPOCH - PREFIX_EPOCHS) * STEPS_PER_EPOCH
PROBE_STEPS = 200
OUT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/D_matched_prefix"
BIO_METRICS = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv"
GROUP_PREFIXES = {
    "eva_early": ("blocks.0.",),
    "eva_middle": ("blocks.11.",),
    "eva_late": ("blocks.22.",),
    "bio_projection": ("head_clip_bio.",),
    "id_classifier": ("head.",),
}
BRANCHES = {
    "B0": {"hair": 0.0, "clothing": 0.0, "label": "Bio continuation"},
    "B1": {"hair": 1.0, "clothing": 0.0, "label": "Bio + Hair"},
    "B2": {"hair": 0.0, "clothing": 1.0, "label": "Bio + Clothing"},
    "B3": {"hair": 1.0, "clothing": 1.0, "label": "Bio + Hair + Clothing"},
}


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PlannedTrainDataset(Dataset):
    def __init__(self, samples: list[tuple[dict[str, Any], int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, augmentation_seed = self.samples[index]
        item = dict(source)
        image = read_image(item["image_path"])
        previous_state = random.getstate()
        try:
            random.seed(int(augmentation_seed))
            image = self.transform(image)
        finally:
            random.setstate(previous_state)
        item["image"] = image
        item["augmentation_seed"] = int(augmentation_seed)
        return item


class MetricCapture:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag, value, global_step=None, *args, **kwargs):
        self.values.setdefault(str(tag), []).append(float(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(type(value).__name__)


def configure() -> Any:
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    cfg.MODEL.DIST_TRAIN = True
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.DATA.NUM_INSTANCES = 2
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.MAX_EPOCHS = TOTAL_SCHEDULE_EPOCHS
    cfg.OUTPUT_DIR = str(OUT)
    cfg.TEST.WEIGHT = ""
    cfg.freeze()
    return cfg


def core_config() -> Any:
    cfg = configure()
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.freeze()
    return cfg


def init_dist() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("PHASE 4R-D requires exactly two visible PPUs; CPU fallback is forbidden")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def cleanup_dist():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def plan_seed(global_step: int, position: int, dataset_index: int, rank: int) -> int:
    token = f"DIFFER_PHASE4R_D|{SEED}|{rank}|{global_step}|{position}|{dataset_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


def plan_path(rank: int) -> Path:
    return OUT / "batch_plan" / f"plan_rank{rank}.jsonl"


def generate_plan(dataset: Any, sampler: Any, rank: int, path: Path, steps: int) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing batch plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w") as stream:
        for global_step in range(1, steps + 1):
            epoch0 = (global_step - 1) // STEPS_PER_EPOCH
            iteration = (global_step - 1) % STEPS_PER_EPOCH + 1
            sampler.set_epoch(epoch0)
            # Avoid regenerating the same sampler epoch 596 times while still
            # retaining the released sampler's exact ordering.
            if iteration == 1:
                epoch_indices = list(iter(sampler))
            start = (iteration - 1) * BATCH_SIZE
            indices = epoch_indices[start:start + BATCH_SIZE]
            if len(indices) != BATCH_SIZE:
                raise RuntimeError(f"incomplete batch at planned step {global_step}")
            samples = []
            for position, dataset_index in enumerate(indices):
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
    return {
        "path": str(path),
        "steps": steps,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "metadata_sha256": digest.hexdigest(),
        "file_sha256": sha256_file(path),
    }


def load_plan(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def planned_loader(cfg: Any, rank: int, start_step: int, end_step: int):
    path = plan_path(rank)
    plan = load_plan(path)
    if len(plan) < end_step:
        raise RuntimeError(f"batch plan {path} has {len(plan)} rows; need {end_step}")
    dataset = build_dataset(cfg)
    transform_train, _ = build_img_transforms(cfg)
    samples: list[tuple[dict[str, Any], int]] = []
    batches: list[list[int]] = []
    selected = plan[start_step:end_step]
    for line in selected:
        batch_indices = []
        for item in line["samples"]:
            source = dataset.train[int(item["dataset_index"])]
            for key in ("image_path", "pid", "camid", "clothes_id"):
                if str(source[key]) != str(item[key]):
                    raise RuntimeError(f"batch plan mismatch for {key}: {source[key]} != {item[key]}")
            batch_indices.append(len(samples))
            samples.append((dict(source), int(item["augmentation_seed"])))
        batches.append(batch_indices)
    loader = DataLoader(
        PlannedTrainDataset(samples, transform_train),
        batch_sampler=FixedBatchSampler(batches),
        num_workers=1,
        pin_memory=False,
    )
    return loader, dataset, selected


def prepare_plans() -> None:
    rank = init_dist()
    try:
        if any(plan_path(r).exists() for r in range(WORLD_SIZE)):
            raise RuntimeError("D matched-prefix batch plan already exists; refusing to overwrite it")
        cfg = configure()
        set_seed(SEED)
        train_loader, _q, _g, dataset, sampler, _v, _o = build_dataloader(cfg)
        if len(train_loader) != STEPS_PER_EPOCH:
            raise RuntimeError(f"expected {STEPS_PER_EPOCH} iterations/epoch, got {len(train_loader)}")
        info = generate_plan(dataset, sampler, rank, plan_path(rank), BRANCH_END_EPOCH * STEPS_PER_EPOCH)
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, info)
        if rank == 0:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "batch_plan_manifest.json").write_text(json.dumps({
                "phase": "4R-D",
                "created_utc": utc_now(),
                "seed": SEED,
                "world_size": WORLD_SIZE,
                "steps": BRANCH_END_EPOCH * STEPS_PER_EPOCH,
                "rank_plans": gathered,
                "data": {
                    "train_images": len(dataset.train),
                    "train_ids": dataset.num_train_pids,
                    "train_clothes": dataset.num_train_clothes,
                    "query": len(dataset.query),
                    "gallery": len(dataset.gallery),
                },
            }, indent=2, sort_keys=True) + "\n")
        dist.barrier()
        print(json.dumps({"rank": rank, "prepared": True, "plan": info}, sort_keys=True), flush=True)
    finally:
        cleanup_dist()


def build_model_from_state(cfg: Any, state: dict[str, torch.Tensor], device: torch.device):
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    return {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}


def tensor_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item) for item in value)
    return value


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: dict[str, Any], device: torch.device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device):
    payload = {
        "phase": "4R-D",
        "epoch": epoch,
        "global_step": global_step,
        "model": cpu_state_dict(model),
        "optimizer": tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_rank_state(path: Path, model, optimizer, scheduler, scaler, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"], device)
    return payload


def save_hashes(state: dict[str, torch.Tensor], path: Path):
    payload = {key: sha256_bytes(value.contiguous().numpy().tobytes()) for key, value in state.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def group_state(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group: {name: state[name].clone() for name in state if name.startswith(prefixes)}
        for group, prefixes in GROUP_PREFIXES.items()
    }


def save_snapshot(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(group_state(cpu_state_dict(model)), path)


def reverse_stats(score: dict[str, torch.Tensor], caption: torch.Tensor, detach: bool = False):
    reverse = score["clip_bio_reverse_score"]
    if detach:
        reverse = reverse.detach()
    text = caption[:, 1:1 + reverse.shape[1]].to(device=reverse.device, dtype=reverse.dtype)
    result = []
    for factor in range(reverse.shape[1]):
        image = F.normalize(reverse[:, factor], dim=-1)
        target = F.normalize(text[:, factor], dim=-1)
        logits = image @ target.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        result.append({
            "loss": F.cross_entropy(logits, labels),
            "accuracy": (logits.argmax(dim=-1) == labels).float().mean(),
        })
    return result


def forward_batch(model, core_loss_fn, batch, device, hair_weight, clothing_weight, step: int):
    writer = ScalarCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    core = core_loss_fn(
        score, feat, pids, camids, captions, clothes_ids=clothes,
        train_writer=writer, step=step,
    )
    reverse = reverse_stats(score, captions, detach=False)
    reverse_hair = reverse[0]["loss"]
    reverse_clothing = reverse[1]["loss"]
    # Do not write ``0.0 * reverse_loss`` into the objective.  Even a zero
    # gradient tensor would make SGD's decoupled-looking weight decay update an
    # inactive reverse head.  The matched-prefix contract requires inactive
    # heads to remain bitwise unchanged, and inactive branch heads must likewise
    # receive no optimizer update.
    total = core
    if hair_weight != 0.0:
        total = total + hair_weight * reverse_hair
    if clothing_weight != 0.0:
        total = total + clothing_weight * reverse_clothing
    return {
        "total": total,
        "core": core,
        "hair": reverse_hair,
        "clothing": reverse_clothing,
        "hair_acc": reverse[0]["accuracy"],
        "clothing_acc": reverse[1]["accuracy"],
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        "writer": writer,
    }


def finite_found_inf(scaler, optimizer) -> float:
    state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    return sum(float(value.item()) for value in state.values())


def scalar_value(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def epoch_reduce(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}


def logger_for(path: Path):
    logger = logging.getLogger(f"phase4r_d.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def evaluate_ltcc(cfg: Any, model, dataset, val_loader, device: torch.device, epoch: int, logger):
    model.eval()
    cc_writer = MetricCapture()
    gen_writer = MetricCapture()
    cc_eval = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    gen_eval = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    cc_eval.reset()
    gen_eval.reset()
    with torch.no_grad():
        test(cfg, model, cc_eval, val_loader, logger, str(device), epoch, cc_writer,
             test=True, ltcc=True, clothChanging=True, metric_prefix="cc")
        test(cfg, model, gen_eval, val_loader, logger, str(device), epoch, gen_writer,
             test=True, ltcc=True, clothChanging=False, metric_prefix="general")
    cc = {key.split("cc/", 1)[1]: values[-1] for key, values in cc_writer.values.items() if key.startswith("cc/")}
    general = {key.split("general/", 1)[1]: values[-1] for key, values in gen_writer.values.items() if key.startswith("general/")}
    model.train()
    return {
        "cc_rank1": float(cc.get("rank1", float("nan"))),
        "cc_rank5": float(cc.get("rank5", float("nan"))),
        "cc_rank10": float(cc.get("rank10", float("nan"))),
        "cc_mAP": float(cc.get("mAP", float("nan"))),
        "general_rank1": float(general.get("rank1", float("nan"))),
        "general_rank5": float(general.get("rank5", float("nan"))),
        "general_rank10": float(general.get("rank10", float("nan"))),
        "general_mAP": float(general.get("mAP", float("nan"))),
    }


def setup_model_runtime(cfg, state: dict[str, torch.Tensor], device):
    model = build_model_from_state(cfg, state, device)
    model.train()
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    core_loss, center = make_loss(core_config(), num_classes=77)
    optimizer, _center_optimizer = make_optimizer(cfg, ddp_model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    return ddp_model, core_loss, optimizer, scheduler, scaler


def make_fresh_state(cfg, dataset):
    # This is only used by prefix, where the requested fresh EVA pretrained
    # initialization is required.  Prefix construction follows the formal
    # driver: seed -> standard dataloader/dataset -> build_model(pretrained).
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    return cpu_state_dict(model)


def prefix() -> None:
    rank = init_dist()
    device = torch.device(f"cuda:{rank}")
    try:
        prefix_dir = OUT / "COMMON_PREFIX_E20"
        prefix_dir.mkdir(parents=True, exist_ok=True)
        if (prefix_dir / "PREFIX_COMPLETE.json").exists():
            raise RuntimeError("COMMON_PREFIX_E20 is already complete; refusing to overwrite it")
        cfg = configure()
        set_seed(SEED)
        standard_train, _q, _g, dataset, _sampler, val_loader, _original = build_dataloader(cfg)
        if len(standard_train) != STEPS_PER_EPOCH:
            raise RuntimeError(f"expected formal {STEPS_PER_EPOCH} iterations/epoch, got {len(standard_train)}")
        # Full topology is built before any reverse loss is activated.
        state = make_fresh_state(cfg, dataset)
        if rank == 0:
            torch.save(state, prefix_dir / "FRESH_FULL_TOPOLOGY_INIT.pth")
            save_hashes(state, prefix_dir / "FRESH_FULL_TOPOLOGY_INIT_HASHES.json")
            (prefix_dir / "config.yml").write_text(cfg.dump())
        dist.barrier()
        loader, plan_dataset, _plan = planned_loader(cfg, rank, 0, PREFIX_STEPS)
        model, core_loss, optimizer, scheduler, scaler = setup_model_runtime(cfg, state, device)
        # Start with the same formal scheduler state and call scheduler.step at
        # each epoch boundary exactly as processor.do_train does.
        logger = logger_for(prefix_dir / f"train_log_rank{rank}.txt")
        writer = SummaryWriter(str(prefix_dir / "train")) if rank == 0 else None
        fieldnames = [
            "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss",
            "reverse_hair_loss", "reverse_cloth_loss", "lr", "lr_bias",
            "id_accuracy", "bio_accuracy", "reverse_hair_accuracy", "reverse_cloth_accuracy",
            "amp_scale_min", "amp_scale_max", "amp_scale_end", "skipped_steps",
            "found_inf_count", "nonfinite_events", "cc_rank1", "cc_rank5", "cc_rank10",
            "cc_mAP", "general_rank1", "general_rank5", "general_rank10", "general_mAP",
        ]
        metrics_file = (prefix_dir / "metrics_per_epoch.csv").open("w", newline="") if rank == 0 else None
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=fieldnames) if rank == 0 else None
        if metrics_csv:
            metrics_csv.writeheader(); metrics_file.flush()
        iterator = iter(loader)
        rows = []
        global_step = 0
        for epoch in range(1, PREFIX_EPOCHS + 1):
            scheduler.step(epoch)
            epoch_sum = {key: 0.0 for key in ["total", "core", "hair", "clothing", "id_acc", "bio_acc", "hair_acc", "clothing_acc"]}
            local_skips = 0.0
            local_nonfinite = 0.0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            for _ in range(STEPS_PER_EPOCH):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = forward_batch(model, core_loss, batch, device, 0.0, 0.0, global_step)
                total = result["total"]
                local_nonfinite += float(not bool(torch.isfinite(total.detach()).item()))
                scaler.scale(total).backward()
                scaler.step(optimizer)
                found_inf = finite_found_inf(scaler, optimizer)
                local_skips += float(found_inf > 0)
                scaler.update()
                scale_min = min(scale_min, float(scaler.get_scale()))
                scale_max = max(scale_max, float(scaler.get_scale()))
                for key in epoch_sum:
                    value = result[key]
                    epoch_sum[key] += scalar_value(value) if key not in ("bio_acc",) else float(value)
                if writer:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(tag, values[-1], global_step)
                    writer.add_scalar("loss/total", scalar_value(total), global_step)
                    writer.add_scalar("lr", float(scheduler._get_lr(epoch)[0]), global_step)
            reduced = epoch_reduce({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            count = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            amp_stats = {
                "skips": int(count[0].item()), "nonfinite": int(count[1].item()),
                "scale_min": float(count[2].item() / WORLD_SIZE),
                "scale_max": float(count[3].item() / WORLD_SIZE),
                "scale_end": float(count[4].item() / WORLD_SIZE),
            }
            dist.barrier()
            # The prefix has only one quality gate (E20).  Avoid spending a
            # full LTCC query+gallery pass on every intermediate prefix epoch;
            # the matched branches perform the required E21--E30 evaluation.
            if rank == 0 and epoch == PREFIX_EPOCHS:
                metrics = evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            gathered_metrics = [None] * WORLD_SIZE
            if rank == 0:
                gathered_metrics[0] = metrics
            dist.broadcast_object_list(gathered_metrics, src=0)
            metrics = gathered_metrics[0]
            metrics = metrics or {
                "cc_rank1": float("nan"), "cc_rank5": float("nan"), "cc_rank10": float("nan"), "cc_mAP": float("nan"),
                "general_rank1": float("nan"), "general_rank5": float("nan"), "general_rank10": float("nan"), "general_mAP": float("nan"),
            }
            row = {
                "epoch": epoch, "train_loss": reduced["total"], "ce_loss": float("nan"),
                "triplet_loss": float("nan"), "bio_loss": float("nan"),
                "reverse_hair_loss": reduced["hair"], "reverse_cloth_loss": reduced["clothing"],
                "lr": float(scheduler._get_lr(epoch)[0]),
                "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"],
                "reverse_hair_accuracy": reduced["hair_acc"], "reverse_cloth_accuracy": reduced["clothing_acc"],
                "amp_scale_min": amp_stats["scale_min"], "amp_scale_max": amp_stats["scale_max"],
                "amp_scale_end": amp_stats["scale_end"], "skipped_steps": amp_stats["skips"],
                "found_inf_count": amp_stats["skips"], "nonfinite_events": amp_stats["nonfinite"], **metrics,
            }
            rows.append(row)
            if writer:
                for key, value in metrics.items(): writer.add_scalar(f"eval/{key}", value, epoch)
            if rank == 0:
                metrics_csv.writerow(row); metrics_file.flush()
        # Reverse heads were created at epoch 0 and must not have changed.
        final_state = cpu_state_dict(model)
        initial_hashes = json.loads((prefix_dir / "FRESH_FULL_TOPOLOGY_INIT_HASHES.json").read_text())
        final_reverse_hashes = {
            key: sha256_bytes(value.contiguous().numpy().tobytes())
            for key, value in final_state.items() if key.startswith("head_clip_bioReverse.")
        }
        initial_reverse_hashes = {key: initial_hashes[key] for key in final_reverse_hashes}
        reverse_unchanged = final_reverse_hashes == initial_reverse_hashes
        rank_state_path = prefix_dir / f"rank{rank}_state.pt"
        save_rank_state(rank_state_path, model, optimizer, scheduler, scaler, PREFIX_EPOCHS, PREFIX_STEPS, device)
        rank_state_hash = sha256_file(rank_state_path)
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, {"rank": rank, "state": str(rank_state_path), "sha256": rank_state_hash})
        if rank == 0:
            bio_rows = {}
            if BIO_METRICS.exists():
                with BIO_METRICS.open() as stream:
                    bio_rows = {int(float(row["epoch"])): row for row in csv.DictReader(stream)}
            e20 = rows[-1]
            historical = bio_rows.get(20, {})
            quality = {
                "prefix_cc_rank1": e20["cc_rank1"], "prefix_cc_mAP": e20["cc_mAP"],
                "historical_bio_cc_rank1": float(historical.get("cc_rank1", "nan")),
                "historical_bio_cc_mAP": float(historical.get("cc_mAP", "nan")),
                "absolute_difference": {
                    "rank1": e20["cc_rank1"] - float(historical.get("cc_rank1", "nan")),
                    "mAP": e20["cc_mAP"] - float(historical.get("cc_mAP", "nan")),
                },
                "neighborhood_rule": "absolute E20 difference <= 0.05 for both CC Rank-1 and mAP",
            }
            quality["passed"] = bool(abs(quality["absolute_difference"]["rank1"]) <= 0.05 and abs(quality["absolute_difference"]["mAP"]) <= 0.05)
            (prefix_dir / "PREFIX_QUALITY_GATE.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
            (prefix_dir / "PREFIX_REVERSE_HEAD_HASH_AUDIT.json").write_text(json.dumps({
                "epoch0_reverse_hashes": initial_reverse_hashes,
                "epoch20_reverse_hashes": final_reverse_hashes,
                "unchanged": reverse_unchanged,
            }, indent=2, sort_keys=True) + "\n")
            manifest = {
                "phase": "4R-D", "track": "repair/research; matched-prefix timing",
                "created_utc": utc_now(), "seed": SEED, "world_size": WORLD_SIZE,
                "prefix_epochs": PREFIX_EPOCHS, "global_step": PREFIX_STEPS,
                "full_topology": True, "active_losses_prefix": ["CE", "Triplet", "Bio"],
                "reverse_heads_created_epoch1": True, "reverse_loss_active_prefix": False,
                "summary_bio": True, "nobio_index": [1, 2], "scheduler_total_epochs": TOTAL_SCHEDULE_EPOCHS,
                "amp_initial_scale": INITIAL_SCALE, "rank_states": gathered,
                "reverse_heads_unchanged": reverse_unchanged, "quality_gate": quality,
            }
            (prefix_dir / "PREFIX_COMPLETE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        try:
            if 'metrics_file' in locals() and metrics_file is not None: metrics_file.close()
            if 'writer' in locals() and writer is not None: writer.close()
        finally:
            cleanup_dist()


def batch_fingerprint(batch) -> str:
    digest = hashlib.sha256()
    for key in ("pid", "camid", "clothes_id", "augmentation_seed"):
        value = batch[key]
        if torch.is_tensor(value): value = value.cpu().numpy().tobytes()
        else: value = np.asarray(value).tobytes()
        digest.update(value)
    image = batch["image"].contiguous().cpu().numpy().tobytes()
    digest.update(image)
    return digest.hexdigest()


def branch(branch_name: str) -> None:
    if branch_name not in BRANCHES:
        raise ValueError(branch_name)
    rank = init_dist()
    device = torch.device(f"cuda:{rank}")
    run_dir = OUT / branch_name
    try:
        if not (OUT / "COMMON_PREFIX_E20" / "PREFIX_COMPLETE.json").exists():
            raise RuntimeError("COMMON_PREFIX_E20 is not complete")
        if not json.loads((OUT / "COMMON_PREFIX_E20" / "PREFIX_QUALITY_GATE.json").read_text()).get("passed"):
            raise RuntimeError("prefix quality gate failed; branching is forbidden")
        if (run_dir / "BRANCH_COMPLETE.json").exists():
            raise RuntimeError(f"{branch_name} already complete; refusing to overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = configure()
        set_seed(SEED)
        standard_train, _q, _g, dataset, _sampler, val_loader, _original = build_dataloader(cfg)
        state0 = torch.load(OUT / "COMMON_PREFIX_E20" / "FRESH_FULL_TOPOLOGY_INIT.pth", map_location="cpu", weights_only=True)
        # Build exact topology, then load the rank-specific E20 optimizer/RNG state.
        model, core_loss, optimizer, scheduler, scaler = setup_model_runtime(cfg, state0, device)
        prefix_state = load_rank_state(OUT / "COMMON_PREFIX_E20" / f"rank{rank}_state.pt", model, optimizer, scheduler, scaler, device)
        loader, _planned_dataset, selected_plan = planned_loader(cfg, rank, PREFIX_STEPS, BRANCH_END_EPOCH * STEPS_PER_EPOCH)
        logger = logger_for(run_dir / f"train_log_rank{rank}.txt")
        writer = SummaryWriter(str(run_dir / "train")) if rank == 0 else None
        fieldnames = ["epoch", "train_loss", "core_loss", "hair_loss", "clothing_loss", "id_accuracy", "bio_accuracy", "hair_accuracy", "clothing_accuracy", "lr", "lr_bias", "amp_scale_min", "amp_scale_max", "amp_scale_end", "skipped_steps", "nonfinite_events", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP", "general_rank1", "general_rank5", "general_rank10", "general_mAP"]
        metrics_file = (run_dir / "metrics_per_epoch.csv").open("w", newline="") if rank == 0 else None
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=fieldnames) if rank == 0 else None
        if metrics_csv:
            metrics_csv.writeheader()
        (run_dir / "config.yml").write_text(cfg.dump()) if rank == 0 else None
        (run_dir / "branch_spec.json").write_text(json.dumps({"branch": branch_name, **BRANCHES[branch_name], "start_epoch": 20, "end_epoch": 30}, indent=2, sort_keys=True) + "\n") if rank == 0 else None
        snapshots = run_dir / "probe_snapshots"
        if rank == 0:
            save_snapshot(model, snapshots / "step0000.pt")
        probe_rows = []
        rows = []
        runtime_fingerprints = []
        iterator = iter(loader)
        global_step = PREFIX_STEPS
        step0_eval = None
        dist.barrier()
        if rank == 0:
            step0_eval = evaluate_ltcc(cfg, model.module, dataset, val_loader, device, 20, logger)
        eval_obj = [step0_eval]
        dist.broadcast_object_list(eval_obj, src=0)
        step0_eval = eval_obj[0]
        # Epoch 21 starts with the inherited E20 scheduler state.  The branch
        # loader contains exactly epochs 21--30, so every branch sees the same
        # global-step/sample mapping.
        epoch_sums = {key: 0.0 for key in ["total", "core", "hair", "clothing", "id_acc", "bio_acc", "hair_acc", "clothing_acc"]}
        local_skips = local_nonfinite = 0.0
        scale_min = scale_max = float(scaler.get_scale())
        probe_train_rows = []
        for local_step in range(1, BRANCH_STEPS + 1):
            epoch = PREFIX_EPOCHS + (local_step - 1) // STEPS_PER_EPOCH + 1
            iteration = (local_step - 1) % STEPS_PER_EPOCH + 1
            if iteration == 1:
                scheduler.step(epoch)
                epoch_sums = {key: 0.0 for key in epoch_sums}
                local_skips = local_nonfinite = 0.0
                scale_min = scale_max = float(scaler.get_scale())
            batch = next(iterator)
            if local_step <= PROBE_STEPS:
                runtime_fingerprints.append(batch_fingerprint(batch))
            global_step += 1
            optimizer.zero_grad()
            result = forward_batch(
                model, core_loss, batch, device,
                BRANCHES[branch_name]["hair"],
                BRANCHES[branch_name]["clothing"],
                global_step,
            )
            total = result["total"]
            local_nonfinite += float(not bool(torch.isfinite(total.detach()).item()))
            scaler.scale(total).backward()
            scaler.step(optimizer)
            found_inf = finite_found_inf(scaler, optimizer)
            local_skips += float(found_inf > 0)
            scaler.update()
            scale_min = min(scale_min, float(scaler.get_scale()))
            scale_max = max(scale_max, float(scaler.get_scale()))
            for key in epoch_sums:
                epoch_sums[key] += scalar_value(result[key]) if key != "bio_acc" else float(result[key])
            if writer:
                for tag, values in result["writer"].values.items():
                    writer.add_scalar(tag, values[-1], global_step)
                writer.add_scalar("loss/total", scalar_value(total), global_step)
                writer.add_scalar("lr", float(scheduler._get_lr(epoch)[0]), global_step)
            if rank == 0 and local_step <= PROBE_STEPS:
                probe_train_rows.append({
                    "step": local_step,
                    "epoch": epoch,
                    "iteration": iteration,
                    "total": scalar_value(total),
                    "core": scalar_value(result["core"]),
                    "hair_loss": scalar_value(result["hair"]),
                    "clothing_loss": scalar_value(result["clothing"]),
                    "id_accuracy": scalar_value(result["id_acc"]),
                    "bio_accuracy": float(result["bio_acc"]),
                    "hair_accuracy": scalar_value(result["hair_acc"]),
                    "clothing_accuracy": scalar_value(result["clothing_acc"]),
                    "amp_scale": float(scaler.get_scale()),
                    "amp_skip": int(found_inf > 0),
                })
            if local_step in (50, 100, 200):
                if rank == 0:
                    save_snapshot(model, snapshots / f"step{local_step:04d}.pt")
                    probe_eval = evaluate_ltcc(
                        cfg, model.module, dataset, val_loader, device,
                        20 + local_step / STEPS_PER_EPOCH, logger,
                    )
                else:
                    probe_eval = None
                probe_obj = [probe_eval]
                dist.broadcast_object_list(probe_obj, src=0)
                probe_rows.append({"step": local_step, **probe_obj[0]})
            if iteration == STEPS_PER_EPOCH:
                reduced = epoch_reduce(
                    {key: value / STEPS_PER_EPOCH for key, value in epoch_sums.items()},
                    device,
                )
                amp_tensor = torch.tensor(
                    [local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())],
                    dtype=torch.float64, device=device,
                )
                dist.all_reduce(amp_tensor, op=dist.ReduceOp.SUM)
                amp_stats = {
                    "skips": int(amp_tensor[0].item()),
                    "nonfinite": int(amp_tensor[1].item()),
                    "scale_min": float(amp_tensor[2].item() / WORLD_SIZE),
                    "scale_max": float(amp_tensor[3].item() / WORLD_SIZE),
                    "scale_end": float(amp_tensor[4].item() / WORLD_SIZE),
                }
                dist.barrier()
                if rank == 0:
                    epoch_eval = evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
                else:
                    epoch_eval = None
                eval_obj = [epoch_eval]
                dist.broadcast_object_list(eval_obj, src=0)
                epoch_eval = eval_obj[0]
                row = {
                    "epoch": epoch,
                    "train_loss": reduced["total"],
                    "core_loss": reduced["core"],
                    "hair_loss": reduced["hair"],
                    "clothing_loss": reduced["clothing"],
                    "id_accuracy": reduced["id_acc"],
                    "bio_accuracy": reduced["bio_acc"],
                    "hair_accuracy": reduced["hair_acc"],
                    "clothing_accuracy": reduced["clothing_acc"],
                    "lr": float(scheduler._get_lr(epoch)[0]),
                    "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                    "amp_scale_min": amp_stats["scale_min"],
                    "amp_scale_max": amp_stats["scale_max"],
                    "amp_scale_end": amp_stats["scale_end"],
                    "skipped_steps": amp_stats["skips"],
                    "nonfinite_events": amp_stats["nonfinite"],
                    **epoch_eval,
                }
                if rank == 0:
                    metrics_csv.writerow(row); metrics_file.flush()
                    for key, value in epoch_eval.items(): writer.add_scalar(f"eval/{key}", value, epoch)
                rows.append(row)
        if rank == 0:
            with (run_dir / "probe_train_rank0.csv").open("w", newline="") as stream:
                writer_csv = csv.DictWriter(stream, fieldnames=list(probe_train_rows[0]))
                writer_csv.writeheader(); writer_csv.writerows(probe_train_rows)
            (run_dir / "probe_eval.json").write_text(json.dumps({"start_e20": step0_eval, "checkpoints": probe_rows}, indent=2, sort_keys=True) + "\n")
            torch.save(cpu_state_dict(model), run_dir / "model_final_e30.pth")
        with (run_dir / f"runtime_batch_fingerprints_rank{rank}.json").open("w") as stream:
            json.dump({"rank": rank, "steps": PROBE_STEPS, "fingerprints": runtime_fingerprints}, stream, indent=2)
        rank_state_path = run_dir / f"rank{rank}_state_e30.pt"
        save_rank_state(rank_state_path, model, optimizer, scheduler, scaler, BRANCH_END_EPOCH, global_step, device)
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, {"rank": rank, "state": str(rank_state_path), "sha256": sha256_file(rank_state_path)})
        dist.barrier()
        if rank == 0:
            (run_dir / "BRANCH_COMPLETE.json").write_text(json.dumps({
                "phase": "4R-D", "branch": branch_name, "label": BRANCHES[branch_name]["label"],
                "created_utc": utc_now(), "start_epoch": PREFIX_EPOCHS, "end_epoch": BRANCH_END_EPOCH,
                "global_step": global_step, "hair_coefficient": BRANCHES[branch_name]["hair"],
                "clothing_coefficient": BRANCHES[branch_name]["clothing"], "probe_steps": PROBE_STEPS,
                "rank_states": gathered, "model_final": str(run_dir / "model_final_e30.pth"),
            }, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        try:
            if 'metrics_file' in locals() and metrics_file is not None: metrics_file.close()
            if 'writer' in locals() and writer is not None: writer.close()
        finally:
            cleanup_dist()


def compare_tensors(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    av = torch.cat([a[name].float().reshape(-1) for name in sorted(a)])
    bv = torch.cat([b[name].float().reshape(-1) for name in sorted(b)])
    diff = av - bv
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(bv).clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(av[None], bv[None]).item()),
        "max_abs": float(diff.abs().max()),
    }


def compare_grouped(a: dict[str, dict[str, torch.Tensor]], b: dict[str, dict[str, torch.Tensor]]):
    return {group: compare_tensors(a[group], b[group]) for group in GROUP_PREFIXES}


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    with path.open() as stream:
        result = {}
        for row in csv.DictReader(stream):
            epoch = int(float(row["epoch"]))
            converted = {}
            for key, value in row.items():
                if key == "epoch":
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = float("nan")
            result[epoch] = converted
        return result


def sustained_harm(deltas: list[tuple[int, float, float]]):
    for index in range(len(deltas) - 2):
        window = deltas[index:index + 3]
        if all(r1 < -0.01 and mAP < -0.005 for _, r1, mAP in window):
            return window[0][0]
    return None


def treatment_summary(branch_rows: dict[int, dict[str, float]], base_rows: dict[int, dict[str, float]]):
    deltas = []
    for epoch in range(PREFIX_EPOCHS + 1, BRANCH_END_EPOCH + 1):
        if epoch not in branch_rows or epoch not in base_rows:
            continue
        deltas.append((
            epoch,
            branch_rows[epoch]["cc_rank1"] - base_rows[epoch]["cc_rank1"],
            branch_rows[epoch]["cc_mAP"] - base_rows[epoch]["cc_mAP"],
        ))
    if not deltas:
        return {"epochs": [], "mean_delta_r1": None, "mean_delta_mAP": None}
    r1 = np.asarray([item[1] for item in deltas], dtype=np.float64)
    mAP = np.asarray([item[2] for item in deltas], dtype=np.float64)
    return {
        "epochs": [{"epoch": e, "delta_r1": float(a), "delta_mAP": float(b)} for e, a, b in deltas],
        "mean_delta_r1": float(r1.mean()), "mean_delta_mAP": float(mAP.mean()),
        "min_delta_r1": float(r1.min()), "min_delta_mAP": float(mAP.min()),
        "final_delta_r1": float(r1[-1]), "final_delta_mAP": float(mAP[-1]),
        "slope_delta_r1_per_epoch": float(np.polyfit(np.arange(len(r1)), r1, 1)[0]) if len(r1) > 1 else 0.0,
        "slope_delta_mAP_per_epoch": float(np.polyfit(np.arange(len(mAP)), mAP, 1)[0]) if len(mAP) > 1 else 0.0,
        "auc_delta_r1": float(r1.mean()),
        "auc_delta_mAP": float(mAP.mean()),
        "first_sustained_harm_epoch": sustained_harm(deltas),
    }


def fingerprint_audit():
    base = {}
    for rank in range(WORLD_SIZE):
        path = OUT / "B0" / f"runtime_batch_fingerprints_rank{rank}.json"
        base[rank] = json.loads(path.read_text())["fingerprints"]
    result = {}
    for branch_name in BRANCHES:
        rank_results = {}
        for rank in range(WORLD_SIZE):
            path = OUT / branch_name / f"runtime_batch_fingerprints_rank{rank}.json"
            values = json.loads(path.read_text())["fingerprints"]
            rank_results[str(rank)] = {
                "length": len(values),
                "identical_to_B0": values == base[rank],
                "first_mismatch": next((i + 1 for i, (a, b) in enumerate(zip(base[rank], values)) if a != b), None),
            }
        result[branch_name] = rank_results
    return result


def snapshot_audit():
    output = {}
    base_snapshots = {}
    for step in (0, 50, 100, 200):
        base_snapshots[step] = torch.load(OUT / "B0" / "probe_snapshots" / f"step{step:04d}.pt", map_location="cpu", weights_only=True)
    for branch_name in BRANCHES:
        output[branch_name] = {}
        for step in (0, 50, 100, 200):
            path = OUT / branch_name / "probe_snapshots" / f"step{step:04d}.pt"
            current = torch.load(path, map_location="cpu", weights_only=True)
            grouped = compare_grouped(current, base_snapshots[step])
            combined_a = {name: value for group in current.values() for name, value in group.items()}
            combined_b = {name: value for group in base_snapshots[step].values() for name, value in group.items()}
            grouped["combined"] = compare_tensors(combined_a, combined_b)
            output[branch_name][str(step)] = grouped
    for branch_name in ("B1", "B2", "B3"):
        output[branch_name]["first_divergence_step_rel_l2_gt_1e-3"] = next(
            (step for step in (50, 100, 200) if output[branch_name][str(step)]["combined"]["relative_l2"] > 1e-3),
            None,
        )
    return output


def analyze() -> None:
    prefix_dir = OUT / "COMMON_PREFIX_E20"
    quality = json.loads((prefix_dir / "PREFIX_QUALITY_GATE.json").read_text())
    prefix_rows = read_metrics(prefix_dir / "metrics_per_epoch.csv")
    branch_rows = {name: read_metrics(OUT / name / "metrics_per_epoch.csv") for name in BRANCHES}
    treatment = {
        name: treatment_summary(branch_rows[name], branch_rows["B0"])
        for name in ("B1", "B2", "B3")
    }
    snapshots = snapshot_audit()
    fingerprints = fingerprint_audit()
    probe = {
        name: json.loads((OUT / name / "probe_eval.json").read_text())
        for name in BRANCHES
    }
    e20 = prefix_rows[PREFIX_EPOCHS]
    e30 = {name: branch_rows[name][BRANCH_END_EPOCH] for name in BRANCHES}
    hair = treatment["B1"]
    clothing = treatment["B2"]
    full = treatment["B3"]
    interaction = {
        "r1_by_epoch": [
            {"epoch": item["epoch"], "interaction": full["epochs"][index]["delta_r1"] - hair["epochs"][index]["delta_r1"] - clothing["epochs"][index]["delta_r1"]}
            for index, item in enumerate(full["epochs"])
        ],
        "mAP_by_epoch": [
            {"epoch": item["epoch"], "interaction": full["epochs"][index]["delta_mAP"] - hair["epochs"][index]["delta_mAP"] - clothing["epochs"][index]["delta_mAP"]}
            for index, item in enumerate(full["epochs"])
        ],
    }
    interaction["mean_r1"] = float(np.mean([item["interaction"] for item in interaction["r1_by_epoch"]])) if interaction["r1_by_epoch"] else None
    interaction["mean_mAP"] = float(np.mean([item["interaction"] for item in interaction["mAP_by_epoch"]])) if interaction["mAP_by_epoch"] else None
    b2_no_sustained = clothing.get("first_sustained_harm_epoch") is None
    b2_neutral = (clothing.get("mean_delta_r1", -1) >= -0.01 and clothing.get("mean_delta_mAP", -1) >= -0.005)
    b1_neutral = (hair.get("mean_delta_r1", -1) >= -0.01 and hair.get("mean_delta_mAP", -1) >= -0.005)
    if not b2_no_sustained or not b2_neutral:
        diagnosis = "CLOTHING REMAINS HARMFUL AFTER BIO MATURATION"
        d2_gate = False
    elif b1_neutral and abs(float(interaction.get("mean_r1") or 0.0)) < 0.01 and abs(float(interaction.get("mean_mAP") or 0.0)) < 0.005:
        diagnosis = "DELAYED ACTIVATION IS A VALID REPAIR"
        d2_gate = True
    elif b1_neutral and (interaction.get("mean_r1", 0.0) < -0.01 or interaction.get("mean_mAP", 0.0) < -0.005):
        diagnosis = "MULTI-FACTOR INTERACTION IS PRIMARY"
        d2_gate = False
    else:
        diagnosis = "OTHER EVIDENCE-BASED RESULT"
        d2_gate = False
    analysis = {
        "phase": "4R-D", "created_utc": utc_now(), "controls": {
            "id": [0.563776, 0.310721], "bio": [0.563776, 0.323857],
            "original_full": [0.533163, 0.292657], "hair": [0.54337, 0.31259],
            "clothing": [0.52296, 0.27748], "pose": [0.52806, 0.27380],
        },
        "prefix": {"e20": e20, "quality_gate": quality},
        "branches_e30": e30,
        "treatment_effects": treatment,
        "interaction": interaction,
        "snapshot_parameter_drift": snapshots,
        "post_branch_batch_fingerprints": fingerprints,
        "probe_evaluation": probe,
        "diagnosis": diagnosis,
        "d2_gate": d2_gate,
        "d2_schedule_if_allowed": "lambda_H=1 from epoch 1; lambda_C=0 epochs 1-19; linear 0->1 epochs 20-30; lambda_C=1 epochs 31-60",
    }
    out_json = OUT / "MATCHED_PREFIX_ANALYSIS.json"
    out_json.write_text(json.dumps(analysis, indent=2, sort_keys=True, default=json_default) + "\n")
    report = [
        "# PHASE 4R-D — LTCC MATCHED-PREFIX TIMING",
        "",
        "Research/repair track only; no formal reproduction artifact is modified.",
        "",
        "## Frozen controls",
        "",
        "| Control | CC Rank-1 | mAP |",
        "|---|---:|---:|",
        "| ID | 56.3776 | 31.0721 |",
        "| Bio | 56.3776 | 32.3857 |",
        "| Original Full | 53.3163 | 29.2657 |",
        "| Bio + Hair | 54.3370 | 31.2590 |",
        "| Bio + Clothing | 52.2960 | 27.7480 |",
        "| Bio + Pose | 52.8060 | 27.3800 |",
        "",
        "## Common prefix",
        "",
        f"The full topology was created at epoch 1; only CE + Triplet + Bio were optimized through E20. Reverse heads were unchanged: **{json.loads((prefix_dir / 'PREFIX_REVERSE_HEAD_HASH_AUDIT.json').read_text())['unchanged']}**.",
        f"E20 matched-prefix result: **{e20['cc_rank1']:.4f} / {e20['cc_mAP']:.4f}**. Historical Bio E20: **{quality['historical_bio_cc_rank1']:.4f} / {quality['historical_bio_cc_mAP']:.4f}**. Quality gate: **{'PASS' if quality['passed'] else 'FAIL'}**.",
        "",
        "## Matched branch results",
        "",
        "| Branch | E20 start | E30 R1 | E30 mAP | mean ΔR1 vs B0 | mean ΔmAP vs B0 | first sustained harm |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name in BRANCHES:
        if name == "B0":
            d1 = d2 = "0"
            harm = "—"
        else:
            d1 = f"{treatment[name]['mean_delta_r1'] * 100:+.3f} pp"
            d2 = f"{treatment[name]['mean_delta_mAP'] * 100:+.3f} pp"
            harm = str(treatment[name]["first_sustained_harm_epoch"] or "—")
        report.append(f"| {name} {BRANCHES[name]['label']} | {e20['cc_rank1']:.4f} / {e20['cc_mAP']:.4f} | {e30[name]['cc_rank1']:.4f} | {e30[name]['cc_mAP']:.4f} | {d1} | {d2} | {harm} |")
    report += [
        "",
        "## 200-step matched probe",
        "",
        "All branch batch fingerprints are compared to B0 separately for both ranks. Parameter divergence is reported at steps 50/100/200; the first divergence threshold is relative L2 > 1e-3.",
        "",
    ]
    for name in ("B1", "B2", "B3"):
        report.append(f"- **{name}** first measured parameter divergence: `{snapshots[name]['first_divergence_step_rel_l2_gt_1e-3']}`; rank-0/rank-1 batch stream identical to B0: `{fingerprints[name]['0']['identical_to_B0']}` / `{fingerprints[name]['1']['identical_to_B0']}`.")
    report += [
        "",
        "## Treatment and interaction",
        "",
        f"Hair matched effect: **{hair['mean_delta_r1'] * 100:+.3f} pp R1 / {hair['mean_delta_mAP'] * 100:+.3f} pp mAP**.",
        f"Clothing matched effect: **{clothing['mean_delta_r1'] * 100:+.3f} pp R1 / {clothing['mean_delta_mAP'] * 100:+.3f} pp mAP**.",
        f"Full matched effect: **{full['mean_delta_r1'] * 100:+.3f} pp R1 / {full['mean_delta_mAP'] * 100:+.3f} pp mAP**.",
        f"Mean Hair×Clothing interaction: **{(interaction['mean_r1'] or 0.0) * 100:+.3f} pp R1 / {(interaction['mean_mAP'] or 0.0) * 100:+.3f} pp mAP**.",
        "",
        "## Final diagnosis",
        "",
        f"**{diagnosis}**",
        "",
        "The decisive comparison is B2−B0 after the shared Bio-only E1–E20 prefix. This avoids interpreting independent fresh-run differences as a timing effect.",
        "",
        f"D2 delayed-clothing candidate gate: **{'PASS — eligible for one preregistered fresh run' if d2_gate else 'NOT ELIGIBLE'}**.",
        "",
        "No D2, later repair, PRCC, seed sweep, or compatibility experiment is started by the analysis step.",
        "",
        "## Artifact index",
        "",
        f"- Prefix: `{prefix_dir}`",
        *(f"- {name}: `{OUT / name}`" for name in BRANCHES),
        f"- Machine-readable analysis: `{out_json}`",
    ]
    report_path = ROOT / "reports/PHASE_4R_D_MATCHED_PREFIX_TIMING.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps({"report": str(report_path), "diagnosis": diagnosis, "d2_gate": d2_gate}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "prefix", "branch", "analyze"), required=True)
    parser.add_argument("--branch", choices=tuple(BRANCHES), default=None)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_plans()
    elif args.mode == "prefix":
        prefix()
    elif args.mode == "branch":
        if args.branch is None: raise SystemExit("--branch is required")
        branch(args.branch)
    else:
        analyze()


if __name__ == "__main__":
    main()
