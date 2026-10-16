#!/usr/local/bin/python3
"""Two LTCC updates followed by official-style model checkpoint round-trip."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from loss import make_loss
from model import build_model
from scripts.qualify_amp_initial_scale_phase1 import ScalarCapture, make_dataset_config, set_seed
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


def main():
    local_cfg = make_dataset_config("ltcc")
    set_seed(local_cfg.SOLVER.SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset = build_dataset(local_cfg)
    train_transform, test_transform = build_img_transforms(local_cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=local_cfg.DATA.NUM_INSTANCES,
        num_replicas=1,
        rank=0,
        seed=local_cfg.SOLVER.SEED,
    )
    train_loader = DataLoader(
        ImageDataset(dataset.train, transform=train_transform),
        sampler=sampler,
        batch_size=local_cfg.DATA.BATCH_SIZE,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(local_cfg.SOLVER.SEED),
    )
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    loss_fn, center = make_loss(local_cfg, dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(local_cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler(init_scale=1024)
    iterator = iter(train_loader)
    update_records = []
    for iteration in range(1, 3):
        batch = next(iterator)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        loss = loss_fn(
            score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
            clothes_ids=batch["clothes_id"], train_writer=ScalarCapture(), step=iteration,
        )
        before = model.patch_embed.proj.weight.detach().clone()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        delta = float((model.patch_embed.proj.weight.detach() - before).abs().max())
        update_records.append({
            "iteration": iteration,
            "loss": float(loss.detach()),
            "loss_finite": bool(torch.isfinite(loss)),
            "scale": float(scaler.get_scale()),
            "backbone_delta": delta,
        })

    eval_loader = DataLoader(
        ImageDataset(dataset.query[:8], transform=test_transform),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    eval_batch = next(iter(eval_loader))
    images = eval_batch["image"].to(device)
    cameras = eval_batch["camid"].to(device)
    model.eval()
    with torch.no_grad():
        reference_features = model(images, cam_label=cameras).detach().cpu()

    checkpoint_path = Path("/tmp/differ_ltcc_official_style_roundtrip.pth")
    torch.save(model.state_dict(), checkpoint_path)
    saved_object = torch.load(checkpoint_path, map_location="cpu")
    new_model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    incompatible = new_model.load_state_dict(saved_object, strict=True)
    new_model.eval()
    with torch.no_grad():
        loaded_features = new_model(images, cam_label=cameras).detach().cpu()
    max_abs_difference = float((loaded_features - reference_features).abs().max())
    result = {
        "updates": update_records,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "saved_top_level_type": type(saved_object).__name__,
        "model_state_saved": True,
        "classifier_saved_inside_model_state": "head.weight" in saved_object,
        "optimizer_saved": False,
        "scheduler_saved": False,
        "gradscaler_saved": False,
        "gradscaler_runtime_scale_before_save": float(scaler.get_scale()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "reference_feature_shape": list(reference_features.shape),
        "features_finite": bool(torch.isfinite(loaded_features).all()),
        "feature_max_abs_difference": max_abs_difference,
        "feature_equal": bool(torch.equal(reference_features, loaded_features)),
        "status": "PASS" if max_abs_difference <= 1e-7 else "FAIL",
    }
    print("CHECKPOINT_ROUNDTRIP_RESULT", json.dumps(result, sort_keys=True))
    (REPO_ROOT / "diagnostics/checkpoint_roundtrip.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    del model, new_model, optimizer, optimizer_center, center, saved_object
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
