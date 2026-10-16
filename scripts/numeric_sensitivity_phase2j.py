#!/usr/local/bin/python3
"""PHASE 2J-C/D/E numerical diagnostics (no formal training).

This driver keeps the released numerical path intact and compares it only in
isolated, no-step experiments.  It reports component-wise FP32 versus the
current FP16-autocast path, the runtime fused attention path versus the
explicit implementation in ``model/eva_meta_bio.py``, and a finite GradScaler
scale sweep.  No source model/loss/processor code is changed by this script.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    INITIAL_SCALE,
    SEED,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from loss.triplet_loss import TripletLoss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402


DEVICE = torch.device("cuda:0")
COMPONENTS = ("ce", "triplet", "bio", "reverse0", "reverse1", "full")
SELECTED_PARAMS = (
    "blocks.0.attn.q_proj.weight",
    "blocks.11.attn.q_proj.weight",
    "blocks.22.attn.q_proj.weight",
    "head.weight",
    "head_clip_bio.weight",
    "head_clip_bioReverse.0.head_fc.weight",
    "head_clip_bioReverse.1.head_fc.weight",
)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite_stats(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"exists": False}
    x = value.detach().double().reshape(-1)
    finite = torch.isfinite(x)
    return {
        "exists": True,
        "dtype": str(value.dtype),
        "numel": int(x.numel()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "zero_count": int((x == 0).sum().item()),
        "max_abs_finite": float(x[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
        "finite": bool(finite.all().item()),
    }


def compare_tensors(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True, "a_exists": a is not None, "b_exists": b is not None}
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    dn = float(torch.linalg.vector_norm(d).item())
    finite = torch.isfinite(d)
    return {
        "missing": False,
        "a_dtype": str(a.dtype),
        "b_dtype": str(b.dtype),
        "a_nan_count": int(torch.isnan(x).sum().item()),
        "a_inf_count": int(torch.isinf(x).sum().item()),
        "b_nan_count": int(torch.isnan(y).sum().item()),
        "b_inf_count": int(torch.isinf(y).sum().item()),
        "max_abs": float(d.abs().max().item()) if d.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(d * d)).item()) if d.numel() else 0.0,
        "relative_l2": dn / (xn + 1e-30),
        "relative_l2_to_b": dn / (yn + 1e-30),
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn and yn else None,
        "bitwise_equal": bool(torch.equal(a, b)),
        "bitwise_equal_fraction": float((a == b).double().mean().item()) if a.numel() else 1.0,
        "difference_finite": bool(finite.all().item()),
    }


def make_batch(loader: Any) -> dict[str, Any]:
    raw = next(iter(loader))
    return {
        key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def manual_components(score: dict[str, Any], feat: list[torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Reproduce the released component formulas without changing their graph."""
    target = batch["pid"]
    ce = F.cross_entropy(score["cls_score"], target)
    triplet = TripletLoss()(feat[0], target)[0]

    image_bio = F.normalize(score["clip_bio_score"], dim=-1).unsqueeze(1)
    captions = batch["caption_feature"].to(image_bio.dtype)
    text_bio = F.normalize(captions[:, :1], dim=-1)
    text_nonbio = F.normalize(captions[:, 1:], dim=-1)
    bio_logits = image_bio[:, 0] @ text_bio[:, 0].T
    bio = F.cross_entropy(bio_logits, torch.arange(image_bio.shape[0], device=DEVICE))

    reverse = score["clip_bio_reverse_score"]
    reverse_losses = []
    for index in range(reverse.shape[1]):
        image_i = F.normalize(reverse[:, index], dim=-1).to(text_nonbio.dtype)
        logits = image_i @ text_nonbio[:, index].T
        reverse_losses.append(F.cross_entropy(logits, torch.arange(image_i.shape[0], device=DEVICE)))
    reverse0, reverse1 = reverse_losses
    full = ce + triplet + bio + reverse0 + reverse1
    return {"ce": ce, "triplet": triplet, "bio": bio, "reverse0": reverse0, "reverse1": reverse1, "full": full}


def clear_grads(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def selected_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    names = dict(model.named_parameters())
    return {
        name: None if names[name].grad is None else names[name].grad.detach().cpu().clone()
        for name in SELECTED_PARAMS
    }


def component_forward_backward(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    replay_rng: dict[str, Any],
    component: str,
    path: str,
) -> dict[str, Any]:
    restore_rng_state(replay_rng)
    clear_grads(model)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    model.train()
    with amp.autocast(enabled=(path == "fp16")):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = manual_components(score, feat, batch)
    loss = losses[component]
    loss_dtype = str(loss.dtype)
    if path == "fp32":
        loss.backward()
        found_inf = 0.0
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found.values()))
    result = {
        "component": component,
        "path": path,
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": loss_dtype,
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "found_inf": found_inf,
        "score_dtypes": {key: str(value.dtype) for key, value in score.items() if isinstance(value, torch.Tensor)},
        "feat_dtypes": [str(value.dtype) for value in feat if isinstance(value, torch.Tensor)],
        "grads": {name: finite_stats(value) for name, value in selected_grads(model).items()},
        "raw_grads": selected_grads(model),
    }
    clear_grads(model)
    del score, feat, loss, losses, scaler
    return result


def compare_component_results(fp32: dict[str, Any], fp16: dict[str, Any]) -> dict[str, Any]:
    out = {
        "loss": {
            "fp32": fp32["loss"], "fp16": fp16["loss"],
            "abs_diff": abs(fp32["loss"] - fp16["loss"]),
        },
        "loss_dtype": {"fp32": fp32["loss_dtype"], "fp16": fp16["loss_dtype"]},
        "found_inf_fp16": fp16["found_inf"],
        "gradients": {},
    }
    for name in SELECTED_PARAMS:
        out["gradients"][name] = compare_tensors(fp32["raw_grads"].get(name), fp16["raw_grads"].get(name))
    return out


def full_run(model, optimizer, batch, replay_rng, use_autocast: bool, force_unfused: bool | None = None):
    old_flags = []
    if force_unfused is not None:
        for name, module in model.named_modules():
            if hasattr(module, "fused_attn"):
                old_flags.append((module, bool(module.fused_attn)))
                module.fused_attn = bool(not force_unfused)
    restore_rng_state(replay_rng)
    clear_grads(model)
    model.train()
    with amp.autocast(enabled=use_autocast):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = manual_components(score, feat, batch)
    total = losses["full"]
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    if use_autocast:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found.values()))
    else:
        total.backward()
        found_inf = 0.0
    output = {
        "losses": {key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())} for key, value in losses.items()},
        "total": float(total.detach().double().cpu()),
        "feature": feat[0].detach().cpu().clone(),
        "bio_feature": score["clip_bio_score"].detach().cpu().clone(),
        "reverse_feature": score["clip_bio_reverse_score"].detach().cpu().clone(),
        "cls": score["cls_score"].detach().cpu().clone(),
        "score_dtypes": {key: str(value.dtype) for key, value in score.items() if isinstance(value, torch.Tensor)},
        "feat_dtypes": [str(value.dtype) for value in feat if isinstance(value, torch.Tensor)],
        "found_inf": found_inf,
        "grads": {name: finite_stats(value) for name, value in selected_grads(model).items()},
        "raw_grads": selected_grads(model),
        "attention_flags": {name: bool(module.fused_attn) for name, module in model.named_modules() if hasattr(module, "fused_attn")},
    }
    clear_grads(model)
    del score, feat, total, losses, scaler
    for module, flag in old_flags:
        module.fused_attn = flag
    return output


def scalar_compare(a: dict[str, Any], b: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for field in fields:
        out[field] = compare_tensors(a[field], b[field])
    return out


def main() -> None:
    output = ROOT / "diagnostics/phase2j_numeric_sensitivity.json"
    cfg = make_config()
    cfg.defrost(); cfg.MODEL.DIST_TRAIN = False; cfg.freeze()
    torch.cuda.set_device(DEVICE)
    seed_all(SEED)
    dataset, loader, sampler = build_probe_loader(cfg)
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(DEVICE)
    model.train()
    _, center = __import__("loss", fromlist=["make_loss"]).make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(1)
    batch = make_batch(loader)
    replay_rng = capture_rng_state()

    attention_modules = {name: bool(module.fused_attn) for name, module in model.named_modules() if hasattr(module, "fused_attn")}
    parameter_dtypes: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        group = "backbone" if name.startswith(("blocks.", "patch_embed.", "camera_embed", "pos_embed", "cls_token")) else "classifier" if name.startswith("head.") else "bio" if name.startswith(("head_bio", "norm_bio", "head_clip_bio")) else "reverse" if name.startswith("head_clip_bioReverse") else "other"
        entry = parameter_dtypes.setdefault(group, {})
        key = str(parameter.dtype)
        entry[key] = entry.get(key, 0) + 1

    component_results: dict[str, Any] = {}
    for component in COMPONENTS:
        print("PHASE2J_NUMERIC_COMPONENT", component, flush=True)
        fp32 = component_forward_backward(model, optimizer, batch, replay_rng, component, "fp32")
        fp16 = component_forward_backward(model, optimizer, batch, replay_rng, component, "fp16")
        component_results[component] = {
            "fp32": {key: value for key, value in fp32.items() if key != "raw_grads"},
            "fp16": {key: value for key, value in fp16.items() if key != "raw_grads"},
            "comparison": compare_component_results(fp32, fp16),
        }

    print("PHASE2J_NUMERIC_ATTENTION", flush=True)
    fused = full_run(model, optimizer, batch, replay_rng, use_autocast=True, force_unfused=False)
    unfused = full_run(model, optimizer, batch, replay_rng, use_autocast=True, force_unfused=True)
    attention_cmp = {
        "current_flags": attention_modules,
        "fused_flags_after": fused["attention_flags"],
        "unfused_flags": unfused["attention_flags"],
        "forward": {
            "feature": compare_tensors(fused["feature"], unfused["feature"]),
            "bio_feature": compare_tensors(fused["bio_feature"], unfused["bio_feature"]),
            "reverse_feature": compare_tensors(fused["reverse_feature"], unfused["reverse_feature"]),
            "cls": compare_tensors(fused["cls"], unfused["cls"]),
        },
        "losses": {key: {"fused": fused["losses"][key], "unfused": unfused["losses"][key], "abs_diff": abs(fused["losses"][key]["value"] - unfused["losses"][key]["value"])} for key in fused["losses"]},
        "gradients": {name: compare_tensors(fused["raw_grads"].get(name), unfused["raw_grads"].get(name)) for name in SELECTED_PARAMS},
        "fused_found_inf": fused["found_inf"],
        "unfused_found_inf": unfused["found_inf"],
    }

    print("PHASE2J_NUMERIC_SCALER", flush=True)
    # The same model state/batch is used for each no-step scale probe.  The
    # optimizer is only passed to unscale_ and never stepped.
    scale_results = {}
    for scale in (512, 1024, 2048, 4096):
        restore_rng_state(replay_rng)
        clear_grads(model)
        model.train()
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = manual_components(score, feat, batch)
        total = losses["full"]
        scaler = amp.GradScaler(init_scale=scale)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found.values()))
        raw = selected_grads(model)
        scale_results[str(scale)] = {
            "loss": {key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())} for key, value in losses.items()},
            "found_inf": found_inf,
            "grads": {name: finite_stats(value) for name, value in raw.items()},
            "raw_grads": raw,
        }
        clear_grads(model)
        del score, feat, losses, total, scaler

    # Reuse the unscaled FP32 full reference from the component run for scale
    # comparisons; rerun only the no-step reference with restored RNG to keep
    # the exact batch/model state and avoid storing a second huge graph.
    fp32_full = component_forward_backward(model, optimizer, batch, replay_rng, "full", "fp32")
    for key, item in scale_results.items():
        item["vs_fp32_full"] = {name: compare_tensors(fp32_full["raw_grads"].get(name), item["raw_grads"].get(name)) for name in SELECTED_PARAMS}
        item.pop("raw_grads", None)
    fp32_full.pop("raw_grads", None)

    payload = {
        "phase": "2J-C/D/E",
        "config": {
            "dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE),
            "world_size": 2, "rank": 0, "amp_initial_scale": INITIAL_SCALE,
            "loss_path": "official: model autocast, loss outside autocast",
            "image_shape": list(batch["image"].shape),
            "pid": batch["pid"].detach().cpu().tolist(),
            "camid": batch["camid"].detach().cpu().tolist(),
            "clothes": batch["clothes_id"].detach().cpu().tolist(),
        },
        "parameter_dtypes": parameter_dtypes,
        "attention": {"module_count": len(attention_modules), "modules": attention_modules},
        "component_results": component_results,
        "attention_comparison": attention_cmp,
        "scaler_sweep": scale_results,
        "fp32_full_reference": fp32_full,
        "notes": [
            "All experiments use one fixed real LTCC batch and no optimizer.step.",
            "FP16 uses the official model-autocast/loss-outside-autocast path; FP32 disables autocast only for diagnosis.",
            "Attention flags are temporarily toggled in-memory for the explicit reference and restored immediately; no repository source is modified.",
            "GradScaler scales are diagnostic only and do not alter formal initialization.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2J_NUMERIC_SUMMARY", json.dumps({
        "output": str(output),
        "attention_modules": len(attention_modules),
        "scales": list(scale_results),
        "component_loss_dtypes": {k: {"fp32": v["fp32"]["loss_dtype"], "fp16": v["fp16"]["loss_dtype"]} for k, v in component_results.items()},
    }, sort_keys=True), flush=True)
    model.to("cpu")
    del optimizer, optimizer_center, center, scheduler, model, loader, dataset
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
