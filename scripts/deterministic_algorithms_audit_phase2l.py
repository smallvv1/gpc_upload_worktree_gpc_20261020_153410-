#!/usr/local/bin/python3
"""PHASE 2L-E: diagnostic deterministic-algorithms audit.

This tests the backend flag in a fresh, no-step Reverse-1 replay.  It records
whether the PPU accepts the flag and whether repeated selected parameter
gradients change under it.  The formal training path is not modified.
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


DEVICE = torch.device("cuda:0")
REPEATS = 5


def replay(model, base_state, batch, rng_state):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True); model.train()
    with amp.autocast(enabled=True):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, TripletLoss())
    losses["reverse1"].backward()
    names = dict(model.named_parameters())
    selected = {}
    for key in ("blocks.10.attn.q_proj.weight", "blocks.10.attn.proj.weight", "blocks.10.mlp.fc2.weight", "blocks.2.attn.q_proj.weight", "head.weight"):
        if key in names:
            selected[key] = names[key].grad.detach().cpu().clone() if names[key].grad is not None else None
    return {"loss": float(losses["reverse1"].detach().double().cpu()), "loss_dtype": str(losses["reverse1"].dtype), "gradients": selected}


def main():
    cfg = make_config(); seed_all(SEED); torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial = bool(torch.are_deterministic_algorithms_enabled())
    before = []
    enabled = False; enable_error = None
    try:
        torch.use_deterministic_algorithms(True)
        enabled = bool(torch.are_deterministic_algorithms_enabled())
        for _ in range(REPEATS):
            before.append(replay(model, base_state, batch, rng_state))
    except Exception as exc:  # retain original backend exception
        enable_error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            torch.use_deterministic_algorithms(initial)
        except Exception:
            pass
    comparisons = {}
    if before:
        for key in before[0]["gradients"]:
            rows = [compare_tensors(before[0]["gradients"][key], item["gradients"].get(key)) for item in before[1:]]
            comparisons[key] = {
                "pairwise_cosine_min": min(r["cosine"] for r in rows if r.get("cosine") is not None),
                "pairwise_relative_l2_max": max(r["relative_l2"] for r in rows if r.get("relative_l2") is not None),
                "pairwise_max_abs_diff_max": max(r["max_abs_diff"] for r in rows if r.get("max_abs_diff") is not None),
                "bitwise_identical_percentage": 100.0 * sum(r.get("max_abs_diff") == 0.0 for r in rows) / len(rows),
                "first_gradient": stat_tensor(before[0]["gradients"][key]),
            }
    payload = {
        "phase": "2L-E", "config": {"dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE), "loss": "Reverse-1 individual", "repeats": REPEATS, "optimizer_step": False, "scheduler_step": False},
        "initial_deterministic_algorithms": initial,
        "enable_succeeded": enabled,
        "enable_error": enable_error,
        "restored_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "repeats": [{"loss": x["loss"], "loss_dtype": x["loss_dtype"]} for x in before],
        "comparisons": comparisons,
        "notes": ["This is a backend capability/diagnostic check only; no formal source or runtime setting was changed."],
    }
    out = ROOT / "diagnostics/phase2l_deterministic_algorithms.json"; out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2L_DETERMINISTIC_SUMMARY", json.dumps({"output": str(out), "initial": initial, "enable_succeeded": enabled, "enable_error": enable_error, "repeat_count": len(before), "comparisons": comparisons}, sort_keys=True), flush=True)
    del model, dataset, loader, batch; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
