#!/usr/local/bin/python3
"""Phase 2J-B: repeated same-state forward/backward replay (diagnostic only)."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    RANK, SEED, WORLD_SIZE, ScalarCapture, build_natural_model,
    build_probe_loader, capture_rng_state, make_config, restore_rng_state,
    seed_all,
)
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402

REPEATS = 10
INITIAL_SCALE = 1024
SELECTED_PARAMS = [
    "blocks.0.attn.q_proj.weight",
    "blocks.11.attn.q_proj.weight",
    "blocks.22.attn.q_proj.weight",
    "head.weight",
    "head_clip_bio.weight",
    "head_clip_bioReverse.0.head_fc.weight",
    "head_clip_bioReverse.1.head_fc.weight",
]


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def vec_stats(a: torch.Tensor, b: torch.Tensor | None = None) -> dict:
    x = a.detach().double().reshape(-1)
    finite = torch.isfinite(x)
    result = {
        "numel": int(x.numel()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "zero_count": int((x == 0).sum().item()),
        "max_abs_finite": float(x[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(x).item()),
    }
    if b is not None:
        y = b.detach().double().reshape(-1)
        d = x - y
        denom = float(torch.linalg.vector_norm(y).item())
        result.update({
            "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
            "rmse": float(torch.sqrt(torch.mean(d * d)).item()) if d.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(d).item()) / (denom + 1e-30),
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0).item()) if x.numel() and y.numel() else None,
            "bitwise_equal": bool(torch.equal(a, b)),
            "bitwise_equal_fraction": float((a == b).double().mean().item()) if a.numel() else 1.0,
        })
    return result


def make_batch(loader, device):
    raw = next(iter(loader))
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def run_replay(model, loss_fn, optimizer, optimizer_center, batch, replay_rng, device):
    restore_rng_state(replay_rng)
    model.train()
    optimizer.zero_grad()
    optimizer_center.zero_grad()
    writer = ScalarCapture()
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    with amp.autocast(enabled=True):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    total = loss_fn(
        score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
        clothes_ids=batch["clothes_id"], train_writer=writer, step=1,
    )
    writer.add_scalar("loss/total", total.detach().item(), 1)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    found_inf = sum(float(value.item()) for value in found.values())
    names = dict(model.named_parameters())
    grads = {}
    for name in SELECTED_PARAMS:
        parameter = names[name]
        grads[name] = parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
    feature = feat[0].detach().cpu().clone()
    cls = score["cls_score"].detach().cpu().clone()
    values = {key: float(value) for key, value in writer.values.items()}
    model_state_hash = {name: sha_tensor(names[name]) for name in SELECTED_PARAMS}
    del score, feat, total, scaler
    optimizer.zero_grad()
    optimizer_center.zero_grad()
    return {
        "values": values,
        "found_inf": found_inf,
        "feature": feature,
        "cls": cls,
        "grads": grads,
        "model_state_hash": model_state_hash,
    }


def main():
    output = ROOT / "diagnostics/same_state_replay_phase2j.json"
    local_cfg = make_config()
    local_cfg.defrost(); local_cfg.MODEL.DIST_TRAIN = False; local_cfg.freeze()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_all(SEED)
    dataset, loader, sampler = build_probe_loader(local_cfg)
    model = build_natural_model(local_cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(device)
    loss_fn, center = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(1)
    batch = make_batch(loader, device)
    replay_rng = capture_rng_state()
    initial_hash = {name: sha_tensor(parameter) for name, parameter in model.named_parameters() if name in SELECTED_PARAMS}
    raw = []
    for index in range(REPEATS):
        result = run_replay(model, loss_fn, optimizer, optimizer_center, batch, replay_rng, device)
        raw.append(result)
        print("PHASE2J_REPLAY", json.dumps({"repeat": index + 1, "found_inf": result["found_inf"], "total": result["values"].get("loss/total")}, sort_keys=True), flush=True)

    base = raw[0]
    comparisons = []
    for index, result in enumerate(raw[1:], start=2):
        feature_diff = vec_stats(base["feature"], result["feature"])
        cls_diff = vec_stats(base["cls"], result["cls"])
        losses = {}
        for key in sorted(set(base["values"]) | set(result["values"])):
            x, y = base["values"].get(key), result["values"].get(key)
            losses[key] = None if x is None or y is None else {"abs_diff": abs(x - y), "a": x, "b": y}
        grads = {}
        for name in SELECTED_PARAMS:
            a, b = base["grads"].get(name), result["grads"].get(name)
            if a is None or b is None:
                grads[name] = {"missing": True}
            else:
                grads[name] = vec_stats(a, b)
        comparisons.append({"repeat": index, "feature": feature_diff, "cls": cls_diff, "losses": losses, "grads": grads, "found_inf": result["found_inf"]})

    state_after = {name: sha_tensor(parameter) for name, parameter in model.named_parameters() if name in SELECTED_PARAMS}
    payload = {
        "phase": "2J-B",
        "config": {"seed": SEED, "world_size": WORLD_SIZE, "rank": RANK, "batch_size": int(local_cfg.DATA.BATCH_SIZE), "initial_scale": INITIAL_SCALE, "dataset": "ltcc", "model": "biasless natural construction", "loss_path": "forward autocast / loss outside autocast", "attention_path": "current runtime"},
        "batch": {"pid": batch["pid"].detach().cpu().tolist(), "camid": batch["camid"].detach().cpu().tolist(), "clothes": batch["clothes_id"].detach().cpu().tolist(), "image_shape": list(batch["image"].shape)},
        "repeats": REPEATS,
        "selected_params": SELECTED_PARAMS,
        "initial_model_state_hash": initial_hash,
        "final_model_state_hash": state_after,
        "state_unchanged": initial_hash == state_after,
        "base_values": base["values"],
        "base_found_inf": base["found_inf"],
        "base_gradient_stats": {name: (vec_stats(value) if value is not None else {"missing": True}) for name, value in base["grads"].items()},
        "comparisons_to_repeat1": comparisons,
        "same_state_forward_deterministic": all(item["feature"]["bitwise_equal"] and item["cls"]["bitwise_equal"] for item in comparisons),
        "same_state_backward_deterministic": all(all(item["grads"][name].get("bitwise_equal", False) for name in SELECTED_PARAMS) for item in comparisons),
        "notes": ["No optimizer.step was executed.", "The same fixed transformed batch and restored RNG state were used for all ten replays.", "Raw gradient vectors are not stored; per-parameter comparisons are retained."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2J_REPLAY_SUMMARY", json.dumps({"forward_deterministic": payload["same_state_forward_deterministic"], "backward_deterministic": payload["same_state_backward_deterministic"], "state_unchanged": payload["state_unchanged"], "output": str(output)}, sort_keys=True), flush=True)
    model.to("cpu")
    del optimizer, optimizer_center, center, loss_fn, scheduler, model, loader, dataset
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
