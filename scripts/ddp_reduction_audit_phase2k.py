#!/usr/local/bin/python3
"""PHASE 2K-D: one-batch DDP reduction numerical audit.

This is a diagnostic-only, two-rank run.  Each rank computes a gradient on its
own real LTCC batch with the frozen official numerical path.  We compare the
gradient produced by an actual DistributedDataParallel backward (the normal
PPU collective path) with an explicit FP64 average of the same rank-local
gradients.  No optimizer step, scheduler step, parameter update, or long
training run is performed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    INITIAL_SCALE,
    SEED,
    batch_fingerprint,
    build_natural_model,
    make_config,
    seed_all,
)
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402


SELECTED = {
    "early_backbone": "blocks.0.attn.q_proj.weight",
    "middle_backbone": "blocks.11.attn.q_proj.weight",
    "last_backbone": "blocks.22.attn.q_proj.weight",
    "classifier": "head.weight",
    "bio_projector": "head_clip_bio.weight",
    "reverse_hair": "head_clip_bioReverse.0.head_fc.weight",
    "reverse_clothing": "head_clip_bioReverse.1.head_fc.weight",
}


class CaptureWriter:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values[str(tag)] = float(value)


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite_stats(grad: torch.Tensor | None) -> dict[str, Any]:
    if grad is None:
        return {"exists": False}
    g = grad.detach()
    gd = g.double()
    finite = torch.isfinite(gd)
    return {
        "exists": True,
        "dtype": str(g.dtype),
        "numel": int(g.numel()),
        "nan_count": int(torch.isnan(gd).sum().item()),
        "inf_count": int(torch.isinf(gd).sum().item()),
        "zero_count": int((g == 0).sum().item()),
        "max_abs_finite": float(gd[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(gd).item()),
        "finite": bool(finite.all().item()),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Compare an FP32 collective result to an FP64 explicit average."""
    x = a.detach().cpu().double().reshape(-1)
    y = b.detach().cpu().double().reshape(-1)
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    return {
        "max_abs": float(d.abs().max().item()) if d.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(d).item()) if d.numel() else 0.0,
        "relative_l2_to_explicit": float(torch.linalg.vector_norm(d).item()) / yn if yn else None,
        "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0).item()) if xn and yn else None,
        "normal_nan_count": int(torch.isnan(x).sum().item()),
        "normal_inf_count": int(torch.isinf(x).sum().item()),
        "explicit_nan_count": int(torch.isnan(y).sum().item()),
        "explicit_inf_count": int(torch.isinf(y).sum().item()),
    }


def make_rank_loader(cfg, rank: int, world_size: int):
    dataset = build_dataset(cfg)
    transform_train, _ = build_img_transforms(cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=cfg.DATA.NUM_INSTANCES,
        num_replicas=world_size,
        rank=rank,
        seed=cfg.SOLVER.SEED,
    )
    sampler.set_epoch(0)
    generator = torch.Generator()
    generator.manual_seed(cfg.SOLVER.SEED)
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sampler,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=generator,
    )
    return dataset, loader, sampler


def to_device(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def local_backward(model, batch, cfg, dataset, device):
    """Compute one unscaled local rank gradient, without reduction."""
    model.train()
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    writer = CaptureWriter()
    with amp.autocast(enabled=True):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    loss = loss_fn(
        score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
        clothes_ids=batch["clothes_id"], train_writer=writer, step=1,
    )
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    names = dict(model.named_parameters())
    grads = {group: names[name].grad.detach().clone() for group, name in SELECTED.items()}
    result = {
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "losses": dict(writer.values),
        "selected_gradients": {group: finite_stats(names[name].grad) for group, name in SELECTED.items()},
        "grads": grads,
    }
    del loss, score, feat, optimizer, optimizer_center, center, loss_fn, scaler
    return result


def ddp_backward(model, batch, cfg, dataset, device):
    """Run one actual DDP backward and return reduced gradients."""
    model.train()
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, ddp_model, center)
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    writer = CaptureWriter()
    with amp.autocast(enabled=True):
        score, feat = ddp_model(batch["image"], cam_label=batch["camid"])
    loss = loss_fn(
        score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
        clothes_ids=batch["clothes_id"], train_writer=writer, step=1,
    )
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    names = dict(ddp_model.module.named_parameters())
    grads = {group: names[name].grad.detach().clone() for group, name in SELECTED.items()}
    result = {
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "losses": dict(writer.values),
        "selected_gradients": {group: finite_stats(names[name].grad) for group, name in SELECTED.items()},
        "grads": grads,
        "ddp_model": ddp_model,
        "optimizer": optimizer,
        "optimizer_center": optimizer_center,
        "center": center,
        "loss_fn": loss_fn,
        "scaler": scaler,
    }
    del loss, score, feat
    return result


def main() -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"This audit requires exactly 2 ranks, got {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")

    cfg = make_config()
    cfg.defrost(); cfg.MODEL.DIST_TRAIN = True; cfg.freeze()
    seed_all(SEED)
    dataset, loader, sampler = make_rank_loader(cfg, rank, world_size)
    raw = next(iter(loader))
    local_fp = batch_fingerprint(raw)
    batch = to_device(raw, device)

    # Natural model construction is identical to the formal biasless path.
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(device)
    base_hashes = {group: sha_tensor(dict(model.named_parameters())[name]) for group, name in SELECTED.items()}
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    local = local_backward(model, batch, cfg, dataset, device)
    local_grads = local.pop("grads")

    # Reset the same model to the exact pre-backward state, then use a real DDP
    # reducer.  No optimizer step is executed in either path.
    model.load_state_dict(base_state, strict=True)
    ddp = ddp_backward(model, batch, cfg, dataset, device)
    ddp_grads = ddp.pop("grads")

    comparisons = {}
    explicit_stats = {}
    for group, local_grad in local_grads.items():
        reduced = local_grad.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(float(world_size))

        gathered = [torch.empty_like(local_grad) for _ in range(world_size)]
        dist.all_gather(gathered, local_grad)
        if rank == 0:
            explicit = torch.stack([x.detach().cpu().double() for x in gathered], dim=0).mean(dim=0)
            comparisons[group] = compare(reduced, explicit)
            explicit_stats[group] = {
                "rank_local": [finite_stats(x) for x in gathered],
                "explicit_fp64": finite_stats(explicit),
                "ddp_reduced_fp32": finite_stats(reduced),
            }
        # Keep all ranks synchronized before the next potentially large tensor.
        dist.barrier()

    metadata = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "batch_fingerprint": local_fp,
        "sampler": type(sampler).__name__,
        "sampler_length": len(sampler),
        "base_hashes": base_hashes,
        "local_loss": local,
        "ddp_loss": {key: value for key, value in ddp.items() if key not in {"ddp_model", "optimizer", "optimizer_center", "center", "loss_fn", "scaler"}},
    }
    rank_path = ROOT / f"diagnostics/phase2k_ddp_reduction_rank{rank}.json"
    rank_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    dist.barrier()

    if rank == 0:
        rank_records = []
        for idx in range(world_size):
            path = ROOT / f"diagnostics/phase2k_ddp_reduction_rank{idx}.json"
            rank_records.append(json.loads(path.read_text()))
        payload = {
            "phase": "2K-D",
            "config": {
                "dataset": "ltcc", "seed": SEED, "world_size": world_size,
                "batch_size": int(cfg.DATA.BATCH_SIZE), "amp_initial_scale": INITIAL_SCALE,
                "loss_path": "official model autocast / loss outside autocast",
                "losses": ["ce", "triplet", "clipBio", "clipBioReverse"],
                "nobio_index": [1, 2], "reverse_indexing": "[:, i]",
                "reduction_under_test": "actual DDP reducer vs explicit FP64 average",
                "optimizer_step": False, "scheduler_step": False,
            },
            "rank_records": rank_records,
            "ddp_vs_explicit_fp64": comparisons,
            "explicit_stats": explicit_stats,
            "notes": [
                "Each rank used one real rank-local LTCC batch from the released sampler.",
                "The model was reset to the exact pre-backward state before the actual DDP backward.",
                "The explicit reference averages the gathered rank-local FP32 gradients in FP64; it is diagnostic only.",
                "No optimizer, scheduler, parameter update, or long training run was executed.",
            ],
        }
        out = ROOT / "diagnostics/phase2k_ddp_reduction.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("PHASE2K_DDP_SUMMARY", json.dumps({"output": str(out), "comparisons": comparisons}, sort_keys=True), flush=True)

    # Release references before shutdown; this is useful on the PPU runtime.
    del ddp, model, dataset, loader, batch, raw
    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
