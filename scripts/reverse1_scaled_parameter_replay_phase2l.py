#!/usr/local/bin/python3
"""PHASE 2L-B companion: repeat the formal scaled-backward path.

The first 2L parameter replay used an unscaled loss.backward and was bitwise
stable, while the earlier 2J replay used scaler.scale(loss).backward followed
by unscale_.  This diagnostic matches that formal numerical path to determine
whether the previously observed q-projection variation is tied to loss
scaling/unscaling.  It still performs no optimizer or scheduler step.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    SEED,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from layerwise_gradient_frontier_phase2l import component_losses, compare_tensors, stat_tensor  # noqa: E402
from loss.triplet_loss import TripletLoss  # noqa: E402
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402


DEVICE = torch.device("cuda:0")
REPEATS = 20
SCALE = 1024
PARAMETERS = tuple(
    [f"blocks.{i}.attn.q_proj.weight" for i in range(23)]
    + [
        "head.weight",
        "head_clip_bio.weight",
        "head_clip_bioReverse.0.head_fc.weight",
        "head_clip_bioReverse.1.head_fc.weight",
    ]
)


def main():
    cfg = make_config(); seed_all(SEED); torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    names = dict(model.named_parameters())
    records = []
    first = None
    for repeat in range(REPEATS):
        model.load_state_dict(base_state, strict=True)
        restore_rng_state(rng_state)
        optimizer.zero_grad(set_to_none=True)
        if optimizer_center is not None:
            optimizer_center.zero_grad(set_to_none=True)
        model.train()
        scaler = amp.GradScaler(init_scale=SCALE)
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, TripletLoss())
        loss = losses["reverse1"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = {key: names[key].grad.detach().cpu().clone() if names[key].grad is not None else None for key in PARAMETERS}
        if first is None:
            first = grads
        rows = {key: compare_tensors(first[key], grads[key]) for key in PARAMETERS} if repeat else {}
        records.append({"repeat": repeat + 1, "loss": float(loss.detach().double().cpu()), "loss_dtype": str(loss.dtype), "loss_finite": bool(torch.isfinite(loss.detach()).item()), "gradients": {key: stat_tensor(grads[key]) for key in PARAMETERS}, "comparison_to_first": rows})
        del score, feat, loss, losses, scaler, grads
    aggregate = {}
    for key in PARAMETERS:
        rows = [r["comparison_to_first"][key] for r in records[1:]]
        valid = [x for x in rows if "cosine" in x]
        cosine_valid = [x for x in valid if x.get("cosine") is not None]
        aggregate[key] = {
            "comparison_count": len(valid),
            "missing_comparison_count": len(rows) - len(valid),
            "zero_or_undefined_cosine_count": len(valid) - len(cosine_valid),
            "cosine_min": min((x["cosine"] for x in cosine_valid), default=None),
            "relative_l2_max": max((x["relative_l2"] for x in valid if x.get("relative_l2") is not None), default=None),
            "max_abs_diff_max": max((x["max_abs_diff"] for x in valid if x.get("max_abs_diff") is not None), default=None),
            "bitwise_identical_percentage": 100.0 * sum(x.get("max_abs_diff") == 0.0 for x in valid if x.get("max_abs_diff") is not None) / len(valid) if valid else None,
        }
    payload = {
        "phase": "2L-B-scaled",
        "config": {"dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE), "scale": SCALE, "repeats": REPEATS, "loss": "Reverse-1 individual", "backward": "scaler.scale(loss).backward(); scaler.unscale_(optimizer)", "optimizer_step": False, "scheduler_step": False},
        "batch": {"image_shape": list(batch["image"].shape), "pid": batch["pid"].detach().cpu().tolist(), "camid": batch["camid"].detach().cpu().tolist(), "clothes": batch["clothes_id"].detach().cpu().tolist()},
        "aggregate": aggregate, "records": records,
        "notes": ["This isolates the formal scaling/unscale path that was present in 2J; no optimizer.step or scheduler.step was called.", "It is complementary to phase2l_reverse1_parameter_nondeterminism.json, whose unscaled backward replay was bitwise stable."],
    }
    out = ROOT / "diagnostics/phase2l_reverse1_scaled_parameter_replay.json"; out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2L_SCALED_REPLAY_SUMMARY", json.dumps({"output": str(out), "repeats": REPEATS, "aggregate": aggregate}, sort_keys=True), flush=True)
    del model, optimizer, optimizer_center, center_criterion, _loss_fn, dataset, loader, batch; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
