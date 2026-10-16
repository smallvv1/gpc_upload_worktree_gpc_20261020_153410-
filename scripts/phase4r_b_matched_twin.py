#!/usr/local/bin/python3
"""PHASE 4R-B: matched-twin GRL causal isolation on LTCC.

The only intended difference between the two twins is the adversarial gradient
from the reverse semantic loss into the shared EVA/Bio representation.
Reverse heads are trained in both twins with an isolated optimizer/scaler.

This script deliberately supports only short forensic probes.  It never runs
25 or 60 epochs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
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
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataloader, build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import read_image  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from model.build import factory  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402

from scripts.phase4r_a2_forensics import (  # noqa: E402
    INITIAL_SCALE,
    MAX_STEPS,
    ScalarCapture,
    batch_fingerprint,
    configure_full,
    family_finite_audit,
    grad_distance,
    json_default,
    make_param_optimizers,
    optimizer_found_inf,
    scaler_state,
    set_seed,
    state_distance,
    utc_now,
)


WORLD_SIZE = 2
PLAN_STEPS = 6000
NOISE_REPEATS = 20
TELEMETRY_INTERVAL = 20
GROUP_PREFIXES = {
    "early_eva": ("blocks.0.",),
    "middle_eva": ("blocks.11.",),
    "late_eva": ("blocks.22.",),
    "bio_projection": ("head_clip_bio.",),
    "id_classifier": ("head.",),
}


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PlannedTrainDataset(Dataset):
    """Train dataset with per-sample Python-RNG augmentation seeds."""

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


def init_dist() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("PHASE 4R-B requires exactly two visible PPUs; CPU fallback is forbidden")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    result = {}
    for key, value in state.items():
        result[key] = hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
    return result


def plan_seed(global_step: int, sample_position: int, dataset_index: int, rank: int) -> int:
    token = f"DIFFER_PHASE4R_B|1234|{rank}|{global_step}|{sample_position}|{dataset_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


def generate_batch_plan(
    dataset: Any,
    sampler: Any,
    rank: int,
    output_path: Path,
    iterations_per_epoch: int,
    steps: int = PLAN_STEPS,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 8
    # The released DataLoader uses drop_last=True.  In this repository the
    # distributed identity sampler can contain a short tail even though the
    # DataLoader length is the formal 596 iterations/epoch.  Reproduce the
    # actual DataLoader batches rather than assuming sampler length is itself
    # divisible by the batch size.
    steps_per_epoch = int(iterations_per_epoch)
    usable_samples = steps_per_epoch * batch_size
    if len(sampler) < usable_samples:
        raise RuntimeError(
            f"sampler has {len(sampler)} samples but the formal loader needs "
            f"{usable_samples} samples for {steps_per_epoch} batches"
        )
    lines = 0
    metadata_digest = hashlib.sha256()
    with output_path.open("w") as stream:
        cached_epoch = None
        epoch_indices: list[int] = []
        for global_step in range(1, steps + 1):
            epoch = (global_step - 1) // steps_per_epoch
            iteration = (global_step - 1) % steps_per_epoch + 1
            if cached_epoch != epoch:
                sampler.set_epoch(epoch)
                epoch_indices = list(iter(sampler))
                cached_epoch = epoch
            start = (iteration - 1) * batch_size
            indices = epoch_indices[start:start + batch_size]
            if len(indices) != batch_size:
                raise RuntimeError(f"incomplete planned batch at step {global_step}")
            samples = []
            for position, dataset_index in enumerate(indices):
                item = dataset.train[dataset_index]
                sample = {
                    "dataset_index": int(dataset_index),
                    "image_path": str(item["image_path"]),
                    "pid": int(item["pid"]),
                    "camid": int(item["camid"]),
                    "clothes_id": int(item["clothes_id"]),
                    "augmentation_seed": plan_seed(global_step, position, dataset_index, rank),
                }
                samples.append(sample)
                metadata_digest.update(json.dumps(sample, sort_keys=True).encode())
            line = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "iteration": iteration,
                "samples": samples,
            }
            stream.write(json.dumps(line, sort_keys=True) + "\n")
            lines += 1
    return {
        "path": str(output_path),
        "steps": lines,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "sampler_length": len(sampler),
        "dropped_sampler_tail": len(sampler) - usable_samples,
        "metadata_sha256": metadata_digest.hexdigest(),
        "file_sha256": sha256_file(output_path),
    }


def load_plan(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_planned_loader(cfg: Any, plan_path: Path):
    plan = load_plan(plan_path)
    if len(plan) < PLAN_STEPS:
        raise RuntimeError(f"batch plan too short: {len(plan)} < {PLAN_STEPS}")
    dataset = build_dataset(cfg)
    transform_train, _transform_test = build_img_transforms(cfg)
    samples: list[tuple[dict[str, Any], int]] = []
    batches: list[list[int]] = []
    for line in plan:
        indices = []
        for sample in line["samples"]:
            source = dataset.train[int(sample["dataset_index"])]
            # Verify that the frozen plan still refers to the same file and
            # labels; otherwise silently using a changed dataset is forbidden.
            for key in ["image_path", "pid", "camid", "clothes_id"]:
                if str(source[key]) != str(sample[key]):
                    raise RuntimeError(f"plan mismatch for {key}: {source[key]} != {sample[key]}")
            indices.append(len(samples))
            samples.append((dict(source), int(sample["augmentation_seed"])))
        batches.append(indices)
    loader = DataLoader(
        PlannedTrainDataset(samples, transform_train),
        batch_sampler=FixedBatchSampler(batches),
        num_workers=1,
        pin_memory=False,
    )
    return loader, dataset, plan


def plan_path(root: Path, rank: int) -> Path:
    return root / "batch_plan" / f"plan_rank{rank}.jsonl"


def common_twin_prepare(root: Path) -> None:
    local_rank = init_dist()
    try:
        cfg = configure_full()
        cfg.defrost()
        cfg.MODEL.DIST_TRAIN = True
        cfg.freeze()
        set_seed(1234)
        common_dir = root / "common_init"
        plan_dir = root / "batch_plan"
        common_dir.mkdir(parents=True, exist_ok=True)
        plan_dir.mkdir(parents=True, exist_ok=True)

        # This deliberately follows the requested sequence: seed, dataloader,
        # model construction, EVA pretrained load, then save.
        train_loader, _query, _gallery, dataset, sampler, _val, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected formal 596 iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        hashes = state_hashes(state)
        gathered_hashes: list[dict[str, str] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_hashes, hashes)
        if any(item != hashes for item in gathered_hashes if item is not None):
            raise RuntimeError("COMMON_TWIN_INIT differs between ranks")

        plan_info = generate_batch_plan(
            dataset,
            sampler,
            local_rank,
            plan_path(root, local_rank),
            iterations_per_epoch=len(train_loader),
        )
        gathered_plan_info: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_plan_info, plan_info)
        if local_rank == 0:
            init_path = common_dir / "COMMON_TWIN_INIT.pth"
            hash_path = common_dir / "COMMON_TWIN_INIT_HASHES.json"
            config_path = common_dir / "COMMON_TWIN_CONFIG.yml"
            if init_path.exists():
                raise RuntimeError(f"refusing to overwrite existing {init_path}")
            torch.save(state, init_path)
            hash_path.write_text(json.dumps({
                "created_utc": utc_now(),
                "seed": 1234,
                "state_key_count": len(state),
                "state_numel": int(sum(value.numel() for value in state.values())),
                "hashes": hashes,
            }, indent=2, sort_keys=True) + "\n")
            config_path.write_text(cfg.dump())
            manifest = {
                "phase": "4R-B",
                "track": "Repair/research; matched-twin causal isolation",
                "created_utc": utc_now(),
                "seed": 1234,
                "world_size": WORLD_SIZE,
                "device_names": [torch.cuda.get_device_name(i) for i in range(WORLD_SIZE)],
                "common_init": str(init_path),
                "common_init_sha256": sha256_file(init_path),
                "state_key_count": len(state),
                "state_numel": int(sum(value.numel() for value in state.values())),
                "data": {
                    "dataset": "ltcc",
                    "train_images": len(dataset.train),
                    "train_ids": dataset.num_train_pids,
                    "train_clothes": dataset.num_train_clothes,
                    "query": len(dataset.query),
                    "gallery": len(dataset.gallery),
                    "batch_size": 8,
                    "num_instances": 2,
                    "iterations_per_epoch": len(train_loader),
                    "plan_steps": PLAN_STEPS,
                    "augmentation": "released random crop/flip/erase with per-sample Python RNG seeds",
                },
                "model": {
                    "topology": "full EVA + ID + Bio + Hair reverse + Clothing reverse",
                    "summary_bio": True,
                    "nobio_index": [1, 2],
                    "clip_dim": 512,
                    "grl_alpha": 1.0,
                },
                "optimizer": {
                    "shared": "SGD; EVA + ID + Bio; momentum=0.9; wd=0.05",
                    "reverse": "SGD; Hair + Clothing reverse heads only; momentum=0.9; wd=0.05",
                    "scheduler": "60-epoch formal cosine/warmup for both optimizers",
                },
                "amp": {
                    "shared_scaler": INITIAL_SCALE,
                    "reverse_scaler": INITIAL_SCALE,
                    "dynamic_growth_backoff": True,
                },
                "rank_plan_info": gathered_plan_info,
            }
            (root / "COMMON_TWIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        dist.barrier()
        print(json.dumps({"rank": local_rank, "prepared": True, "plan": plan_info}, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def cfg_core() -> Any:
    cfg = configure_full()
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.MODEL.DIST_TRAIN = False
    cfg.freeze()
    return cfg


def group_names(model: torch.nn.Module) -> dict[str, list[str]]:
    names = dict(model.named_parameters())
    result = {}
    for group, prefixes in GROUP_PREFIXES.items():
        result[group] = [name for name in names if name.startswith(prefixes)]
    return result


def group_parameter_lists(model: torch.nn.Module, groups: dict[str, list[str]]):
    params = dict(model.named_parameters())
    return {group: [params[name] for name in names] for group, names in groups.items()}


def zero_like_gradient_list(values: list[torch.Tensor | None], reference: torch.Tensor) -> torch.Tensor:
    if values:
        output = None
        for value in values:
            if value is None:
                continue
            output = value if output is None else output + value
        if output is not None:
            return output
    return torch.zeros_like(reference)


def grad_stats_pair(core_values: list[torch.Tensor | None], reverse_values: list[torch.Tensor | None], params):
    core_norm_sq = None
    reverse_norm_sq = None
    dot = None
    for core, reverse, param in zip(core_values, reverse_values, params):
        reference = torch.zeros_like(param) if core is None and reverse is None else (core if core is not None else reverse)
        gc_value = torch.zeros_like(reference) if core is None else core
        gr_value = torch.zeros_like(reference) if reverse is None else reverse
        core_norm_sq = (gc_value.float() * gc_value.float()).sum() if core_norm_sq is None else core_norm_sq + (gc_value.float() * gc_value.float()).sum()
        reverse_norm_sq = (gr_value.float() * gr_value.float()).sum() if reverse_norm_sq is None else reverse_norm_sq + (gr_value.float() * gr_value.float()).sum()
        dot = (gc_value.float() * gr_value.float()).sum() if dot is None else dot + (gc_value.float() * gr_value.float()).sum()
    core_norm = torch.sqrt(core_norm_sq) if core_norm_sq is not None else torch.tensor(0.0, device=params[0].device)
    reverse_norm = torch.sqrt(reverse_norm_sq) if reverse_norm_sq is not None else torch.tensor(0.0, device=params[0].device)
    denom = core_norm * reverse_norm
    cosine = dot / denom.clamp_min(1e-12) if dot is not None else torch.tensor(0.0, device=params[0].device)
    return {
        "core_norm": float(core_norm.detach().cpu()),
        "reverse_norm": float(reverse_norm.detach().cpu()),
        "cos_core_reverse": float(cosine.detach().cpu()),
        "reverse_to_core": float((reverse_norm / core_norm.clamp_min(1e-12)).detach().cpu()),
    }


def autograd_group_telemetry(core_loss, reverse_loss, model, groups):
    parameter_groups = group_parameter_lists(model, groups)
    output = {}
    for group, params in parameter_groups.items():
        core_values = list(torch.autograd.grad(
            core_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        ))
        reverse_values = list(torch.autograd.grad(
            reverse_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        ))
        output[group] = grad_stats_pair(core_values, reverse_values, params)
    return output


def current_group_norms(model, groups):
    parameters = dict(model.named_parameters())
    output = {}
    for group, names in groups.items():
        norm_sq = 0.0
        for name in names:
            grad = parameters[name].grad
            if grad is not None:
                norm_sq += float((grad.detach().float() * grad.detach().float()).sum().cpu())
        output[group] = norm_sq ** 0.5
    return output


def reverse_losses(feat_bio: torch.Tensor, caption: torch.Tensor, model: torch.nn.Module, shared_path: bool):
    text_nonbio = F.normalize(caption.to(device=feat_bio.device, dtype=feat_bio.dtype)[:, 1:], dim=-1)
    total = torch.zeros((), device=feat_bio.device, dtype=feat_bio.dtype)
    for factor_index, head in enumerate(model.head_clip_bioReverse):
        source = feat_bio if shared_path else feat_bio.detach()
        source = head.grl(source)
        weight = head.head_fc.weight.detach() if shared_path else head.head_fc.weight
        logits = F.linear(source, weight)
        image = F.normalize(logits, dim=-1)
        labels = torch.arange(image.shape[0], device=image.device)
        total = total + F.cross_entropy(image @ text_nonbio[:, factor_index].T, labels)
    return total


def forward_losses(model, core_loss_fn, batch, device):
    writer = ScalarCapture()
    samples = batch["image"].to(device)
    camids = batch["camid"].to(device)
    targets = batch["pid"].to(device)
    clothes = batch["clothes_id"].to(device)
    caption = batch["caption_feature"].to(device)
    with amp.autocast(enabled=True):
        score, feat = model(samples, cam_label=camids)
    core = core_loss_fn(
        score,
        feat,
        targets,
        camids,
        caption,
        clothes_ids=clothes,
        train_writer=writer,
        step=0,
    )
    feat_bio = feat[0]
    reverse_head = reverse_losses(feat_bio, caption, model, shared_path=False)
    reverse_shared = reverse_losses(feat_bio, caption, model, shared_path=True)
    return core, reverse_head, reverse_shared


def optimizer_signature(optimizer):
    return [
        {
            "lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameter_count": len(group["params"]),
        }
        for group in optimizer.param_groups
    ]


def load_twin_models(cfg, common_path: Path, device):
    state = torch.load(common_path, map_location="cpu", weights_only=True)
    # The common init already contains the loaded EVA pretrained weights.
    # Constructing a probe with build_model() would call the repository's
    # hard-coded pretrained=True path and re-probe Hugging Face on every new
    # process.  That is both unnecessary and an external-state confound.  Use
    # the same factory/topology with pretrained=False, then strict-load every
    # tensor from COMMON_TWIN_INIT.
    model_n = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    model_n.load_state_dict(state, strict=True)
    model_g = copy.deepcopy(model_n)
    # Keep the released reverse forward detached in both twins.  The GRL twin's
    # only shared adversarial path is the explicit shared_path below, with
    # reverse-head weights detached.  This avoids an extra hidden branch.
    model_n.detach_reverse_input = True
    model_g.detach_reverse_input = True
    model_n.to(device)
    model_g.to(device)
    model_n.train()
    model_g.train()
    return model_n, model_g, state


def make_twin_runtime(model_n, model_g, cfg):
    shared_n, reverse_n, shared_names_n, reverse_names_n = make_param_optimizers(model_n)
    shared_g, reverse_g, shared_names_g, reverse_names_g = make_param_optimizers(model_g)
    if shared_names_n != shared_names_g or reverse_names_n != reverse_names_g:
        raise RuntimeError("twin optimizer parameter names differ")
    scheduler_shared_n = create_scheduler(cfg, shared_n)
    scheduler_shared_g = create_scheduler(cfg, shared_g)
    scheduler_reverse_n = create_scheduler(cfg, reverse_n)
    scheduler_reverse_g = create_scheduler(cfg, reverse_g)
    scheduler_shared_n.step(1)
    scheduler_shared_g.step(1)
    scheduler_reverse_n.step(1)
    scheduler_reverse_g.step(1)
    runtime = {
        "shared_n": shared_n,
        "reverse_n": reverse_n,
        "shared_g": shared_g,
        "reverse_g": reverse_g,
        "scheduler_shared_n": scheduler_shared_n,
        "scheduler_shared_g": scheduler_shared_g,
        "scheduler_reverse_n": scheduler_reverse_n,
        "scheduler_reverse_g": scheduler_reverse_g,
        "scaler_shared_n": amp.GradScaler(init_scale=INITIAL_SCALE),
        "scaler_reverse_n": amp.GradScaler(init_scale=INITIAL_SCALE),
        "scaler_shared_g": amp.GradScaler(init_scale=INITIAL_SCALE),
        "scaler_reverse_g": amp.GradScaler(init_scale=INITIAL_SCALE),
        "shared_names": shared_names_n,
        "reverse_names": reverse_names_n,
    }
    return runtime


def compare_scalers(a, b):
    return {"a": scaler_state(a), "b": scaler_state(b), "equal": scaler_state(a) == scaler_state(b)}


def noise_floor(root: Path) -> None:
    local_rank = init_dist()
    try:
        cfg = cfg_core()
        set_seed(1234)
        plan_root = root / "batch_plan"
        loader, dataset, plan = build_planned_loader(cfg, plan_path(root, local_rank))
        common_path = root / "common_init" / "COMMON_TWIN_INIT.pth"
        model, _unused, state = load_twin_models(configure_full(), common_path, torch.device(f"cuda:{local_rank}"))
        device = torch.device(f"cuda:{local_rank}")
        groups = group_names(model)
        loss_fn, _center = make_loss(cfg, dataset.num_train_pids)
        iterator = iter(loader)
        batch = next(iterator)
        initial_cpu = torch.get_rng_state()
        initial_cuda = torch.cuda.get_rng_state(device)
        first_grads: dict[str, dict[str, torch.Tensor]] | None = None
        rows = []
        for repeat in range(1, NOISE_REPEATS + 1):
            model.zero_grad(set_to_none=True)
            torch.set_rng_state(initial_cpu)
            torch.cuda.set_rng_state(initial_cuda, device)
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
            core = loss_fn(
                score,
                feat,
                batch["pid"].to(device),
                batch["camid"].to(device),
                batch["caption_feature"].to(device),
                clothes_ids=batch["clothes_id"].to(device),
                train_writer=ScalarCapture(),
                step=repeat,
            )
            core.backward()
            torch.cuda.synchronize(device)
            parameters = dict(model.named_parameters())
            current: dict[str, dict[str, torch.Tensor]] = {}
            for group, names in groups.items():
                current[group] = {
                    name: parameters[name].grad.detach().cpu().clone()
                    for name in names if parameters[name].grad is not None
                }
            if first_grads is None:
                first_grads = current
                comparison = {group: {"relative_l2": 0.0, "cosine": 1.0, "max_abs": 0.0} for group in groups}
            else:
                comparison = {}
                for group in groups:
                    first_values = first_grads[group]
                    current_values = current[group]
                    dot = 0.0
                    first_norm = 0.0
                    current_norm = 0.0
                    diff_norm = 0.0
                    max_abs = 0.0
                    for name in first_values:
                        a = first_values[name].float()
                        b = current_values[name].float()
                        dot += float((a * b).sum())
                        first_norm += float((a * a).sum())
                        current_norm += float((b * b).sum())
                        diff_norm += float(((a - b) * (a - b)).sum())
                        max_abs = max(max_abs, float((a - b).abs().max()))
                    comparison[group] = {
                        "relative_l2": math.sqrt(diff_norm) / max(math.sqrt(first_norm), 1e-12),
                        "cosine": dot / max(math.sqrt(first_norm * current_norm), 1e-12),
                        "max_abs": max_abs,
                    }
            rows.append({"repeat": repeat, "core_loss": float(core.detach().cpu()), "comparison": comparison})
        rank_result = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "batch_fingerprint": batch_fingerprint(batch),
            "repeats": NOISE_REPEATS,
            "rows": rows,
        }
        rank_dir = root / "noise_floor" / f"rank{local_rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "noise_floor.json").write_text(json.dumps(rank_result, indent=2, sort_keys=True) + "\n")
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, rank_result)
        if local_rank == 0:
            summary = summarize_noise(gathered)
            (root / "noise_floor" / "NOISE_FLOOR_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def summarize_noise(gathered):
    result = {"phase": "4R-B", "created_utc": utc_now(), "repeats": NOISE_REPEATS, "ranks": gathered, "envelope": {}}
    for group in GROUP_PREFIXES:
        values = []
        cosines = []
        maxes = []
        for rank_result in gathered:
            for row in rank_result["rows"][1:]:
                item = row["comparison"][group]
                values.append(item["relative_l2"])
                cosines.append(item["cosine"])
                maxes.append(item["max_abs"])
        values_sorted = sorted(values)
        p95_index = min(len(values_sorted) - 1, max(0, int(math.ceil(0.95 * len(values_sorted)) - 1)))
        result["envelope"][group] = {
            "median_relative_l2": float(np.median(values)),
            "p95_relative_l2": float(values_sorted[p95_index]),
            "minimum_cosine": float(min(cosines)),
            "p95_max_abs": float(np.percentile(maxes, 95)),
            "sample_count": len(values),
        }
    return result


def run_paired_probe(root: Path, name: str, shared_lambda: float, steps: int = 200) -> None:
    local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    run_root = root / name
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        model_cfg = configure_full()
        core_cfg = cfg_core()
        set_seed(1234)
        loader, dataset, plan = build_planned_loader(core_cfg, plan_path(root, local_rank))
        common_path = root / "common_init" / "COMMON_TWIN_INIT.pth"
        model_n, model_g, state = load_twin_models(model_cfg, common_path, device)
        runtime = make_twin_runtime(model_n, model_g, model_cfg)
        shared_names = runtime["shared_names"]
        reverse_names = runtime["reverse_names"]
        groups = group_names(model_n)
        loss_n, _center_n = make_loss(core_cfg, dataset.num_train_pids)
        loss_g, _center_g = make_loss(core_cfg, dataset.num_train_pids)
        all_names = [name for name, _ in model_n.named_parameters()]
        initial_distance = state_distance(model_n, model_g, all_names)
        initial_scheduler_equal = runtime["scheduler_shared_n"].state_dict() == runtime["scheduler_shared_g"].state_dict()
        rows = []
        telemetry_rows = []
        iterator = iter(loader)
        for step in range(1, steps + 1):
            batch = next(iterator)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)

            runtime["shared_n"].zero_grad(set_to_none=True)
            runtime["reverse_n"].zero_grad(set_to_none=True)
            core_n, reverse_head_n, reverse_shared_n = forward_losses(model_n, loss_n, batch, device)
            if step == 1 or step % TELEMETRY_INTERVAL == 0:
                t_n = autograd_group_telemetry(core_n, reverse_shared_n, model_n, groups)
            else:
                t_n = None
            runtime["scaler_shared_n"].scale(core_n).backward()
            runtime["scaler_shared_n"].unscale_(runtime["shared_n"])
            finite_n = family_finite_audit(model_n)
            current_norm_n = current_group_norms(model_n, groups)

            # Restore the exact RNG state before the second twin.  The planned
            # loader gives both twins the same already-augmented batch object.
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            runtime["shared_g"].zero_grad(set_to_none=True)
            runtime["reverse_g"].zero_grad(set_to_none=True)
            core_g, reverse_head_g, reverse_shared_g = forward_losses(model_g, loss_g, batch, device)
            if step == 1 or step % TELEMETRY_INTERVAL == 0:
                t_g = autograd_group_telemetry(core_g, reverse_shared_g, model_g, groups)
            else:
                t_g = None
            # At the zero-GRL control point, do not put a mathematically zero
            # reverse graph into the shared backward.  On PPU this can still
            # alter backward execution/accumulation order, so the exact
            # matched control must use the core loss alone.  For nonzero
            # lambda, the reverse graph is intentionally routed to shared.
            shared_total_g = (
                core_g
                if float(shared_lambda) == 0.0
                else core_g + float(shared_lambda) * reverse_shared_g
            )
            runtime["scaler_shared_g"].scale(shared_total_g).backward()
            runtime["scaler_shared_g"].unscale_(runtime["shared_g"])
            finite_g = family_finite_audit(model_g)
            current_norm_g = current_group_norms(model_g, groups)
            grad_compare = None
            if step == 1 or step % TELEMETRY_INTERVAL == 0:
                grad_compare = grad_distance(model_n, model_g, shared_names)

            runtime["scaler_shared_n"].step(runtime["shared_n"])
            found_shared_n = optimizer_found_inf(runtime["scaler_shared_n"], runtime["shared_n"])
            runtime["scaler_shared_n"].update()
            runtime["scaler_shared_g"].step(runtime["shared_g"])
            found_shared_g = optimizer_found_inf(runtime["scaler_shared_g"], runtime["shared_g"])
            runtime["scaler_shared_g"].update()

            runtime["scaler_reverse_n"].scale(reverse_head_n).backward()
            runtime["scaler_reverse_n"].unscale_(runtime["reverse_n"])
            finite_reverse_n = family_finite_audit(model_n)
            runtime["scaler_reverse_n"].step(runtime["reverse_n"])
            found_reverse_n = optimizer_found_inf(runtime["scaler_reverse_n"], runtime["reverse_n"])
            runtime["scaler_reverse_n"].update()

            runtime["scaler_reverse_g"].scale(reverse_head_g).backward()
            runtime["scaler_reverse_g"].unscale_(runtime["reverse_g"])
            finite_reverse_g = family_finite_audit(model_g)
            runtime["scaler_reverse_g"].step(runtime["reverse_g"])
            found_reverse_g = optimizer_found_inf(runtime["scaler_reverse_g"], runtime["reverse_g"])
            runtime["scaler_reverse_g"].update()
            torch.cuda.synchronize(device)

            distance = None
            reverse_distance = None
            if step == 1 or step % 10 == 0:
                distance = state_distance(model_n, model_g, shared_names)
                reverse_distance = state_distance(model_n, model_g, reverse_names)
            row = {
                "step": step,
                "batch_fingerprint": batch_fingerprint(batch),
                "data_identical": True,
                "shared_lambda": float(shared_lambda),
                "core_loss_nograd": float(core_n.detach().float().cpu()),
                "core_loss_grl": float(core_g.detach().float().cpu()),
                "reverse_head_loss_nograd": float(reverse_head_n.detach().float().cpu()),
                "reverse_head_loss_grl": float(reverse_head_g.detach().float().cpu()),
                "reverse_shared_loss_grl": float(reverse_shared_g.detach().float().cpu()),
                "shared_scale_nograd": scaler_state(runtime["scaler_shared_n"]),
                "shared_scale_grl": scaler_state(runtime["scaler_shared_g"]),
                "reverse_scale_nograd": scaler_state(runtime["scaler_reverse_n"]),
                "reverse_scale_grl": scaler_state(runtime["scaler_reverse_g"]),
                "shared_skip_nograd": bool(found_shared_n > 0),
                "shared_skip_grl": bool(found_shared_g > 0),
                "reverse_skip_nograd": bool(found_reverse_n > 0),
                "reverse_skip_grl": bool(found_reverse_g > 0),
                "shared_grad_finite_nograd": finite_n,
                "shared_grad_finite_grl": finite_g,
                "reverse_grad_finite_nograd": finite_reverse_n,
                "reverse_grad_finite_grl": finite_reverse_g,
                "shared_group_norms_nograd": current_norm_n,
                "shared_group_norms_grl": current_norm_g,
                "shared_gradient_distance": grad_compare,
                "shared_parameter_distance": distance,
                "reverse_parameter_distance": reverse_distance,
            }
            rows.append(row)
            if t_n is not None or t_g is not None:
                for twin, telemetry in [("nograd", t_n), ("grl", t_g)]:
                    if telemetry is None:
                        continue
                    for group, values in telemetry.items():
                        telemetry_rows.append({
                            "step": step,
                            "twin": twin,
                            "lambda": float(shared_lambda),
                            "group": group,
                            **values,
                        })
            if step == 1 or step % 10 == 0:
                print(json.dumps({"rank": local_rank, **row}, sort_keys=True, default=json_default), flush=True)

        rank_dir = run_root / f"rank{local_rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        write_rows(rank_dir / "paired_metrics.csv", rows)
        write_rows(rank_dir / "gradient_telemetry.csv", telemetry_rows)
        result = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "shared_lambda": float(shared_lambda),
            "steps": steps,
            "initial_distance": initial_distance,
            "initial_scheduler_equal": initial_scheduler_equal,
            "optimizer_topology_equal": optimizer_signature(runtime["shared_n"]) == optimizer_signature(runtime["shared_g"])
            and optimizer_signature(runtime["reverse_n"]) == optimizer_signature(runtime["reverse_g"]),
            "all_data_identical": all(row["data_identical"] for row in rows),
            "shared_skip_count_nograd": sum(row["shared_skip_nograd"] for row in rows),
            "shared_skip_count_grl": sum(row["shared_skip_grl"] for row in rows),
            "reverse_skip_count_nograd": sum(row["reverse_skip_nograd"] for row in rows),
            "reverse_skip_count_grl": sum(row["reverse_skip_grl"] for row in rows),
            "final_shared_distance": rows[-1]["shared_parameter_distance"] if rows else None,
            "final_reverse_distance": rows[-1]["reverse_parameter_distance"] if rows else None,
            "final_row": rows[-1] if rows else None,
        }
        (rank_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n")
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, result)
        if local_rank == 0:
            (run_root / "summary.json").write_text(json.dumps({
                "phase": "4R-B",
                "probe": name,
                "created_utc": utc_now(),
                "shared_lambda": float(shared_lambda),
                "steps": steps,
                "rank_results": [item for item in gathered if item is not None],
            }, indent=2, sort_keys=True, default=json_default) + "\n")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = json.dumps(value, sort_keys=True, default=json_default)
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "noise-floor", "zero-grl", "dose"], required=True)
    parser.add_argument("--root", default="outputs/new_method_track/phase4r_ltcc_full_recovery/B_matched_twin")
    parser.add_argument("--lambda", dest="shared_lambda", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == "prepare":
        common_twin_prepare(root)
        return
    if not (root / "common_init" / "COMMON_TWIN_INIT.pth").exists():
        raise FileNotFoundError("COMMON_TWIN_INIT.pth is missing; run --mode prepare first")
    if args.mode == "noise-floor":
        noise_floor(root)
    elif args.mode == "zero-grl":
        run_paired_probe(root, "zero_grl", 0.0, min(args.steps, MAX_STEPS))
    else:
        allowed = {0.0, 0.1, 0.25, 0.5, 1.0}
        if args.shared_lambda not in allowed:
            raise ValueError(f"dose response lambda must be one of {sorted(allowed)}")
        run_paired_probe(root, f"dose_lambda_{str(args.shared_lambda).replace('.', 'p')}", args.shared_lambda, min(args.steps, MAX_STEPS))


if __name__ == "__main__":
    main()
