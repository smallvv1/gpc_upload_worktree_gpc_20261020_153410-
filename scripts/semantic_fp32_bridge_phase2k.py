#!/usr/local/bin/python3
"""PHASE 2K Candidate S: semantic-only FP32 arithmetic diagnostic.

The released model forward remains under autocast.  This script changes only
the arithmetic used to normalize the Bio/Reverse image and text features and
to form their contrastive logits.  It never calls optimizer.step() and never
edits repository model/loss/processor code.
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


DEVICE = torch.device("cuda:0")
SELECTED_PARAMS = (
    "blocks.0.attn.q_proj.weight",
    "blocks.11.attn.q_proj.weight",
    "blocks.22.attn.q_proj.weight",
    "head.weight",
    "head_clip_bio.weight",
    "head_clip_bioReverse.0.head_fc.weight",
    "head_clip_bioReverse.1.head_fc.weight",
)
MODES = ("current", "semantic_fp32", "fp32_reference")
COMPONENTS = ("bio", "reverse0", "reverse1", "full")


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True, "a_exists": a is not None, "b_exists": b is not None}
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    dn = float(torch.linalg.vector_norm(d).item())
    return {
        "missing": False,
        "a_dtype": str(a.dtype),
        "b_dtype": str(b.dtype),
        "max_abs": float(d.abs().max().item()) if d.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(d * d)).item()) if d.numel() else 0.0,
        "relative_l2": dn / (xn + 1e-30),
        "relative_l2_to_b": dn / (yn + 1e-30),
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn and yn else None,
        "bitwise_equal": bool(torch.equal(a, b)),
        "bitwise_equal_fraction": float((a == b).double().mean().item()) if a.numel() else 1.0,
        "a_nan_count": int(torch.isnan(x).sum().item()),
        "a_inf_count": int(torch.isinf(x).sum().item()),
        "b_nan_count": int(torch.isnan(y).sum().item()),
        "b_inf_count": int(torch.isinf(y).sum().item()),
    }


def grad_stats(value: torch.Tensor | None) -> dict[str, Any]:
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
        "l2": float(torch.linalg.vector_norm(x).item()),
        "finite": bool(finite.all().item()),
    }


def component_losses(score: dict[str, Any], feat: list[torch.Tensor], batch: dict[str, Any], cfg, semantic_fp32: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target = batch["pid"]
    ce = F.cross_entropy(score["cls_score"], target)
    triplet = TripletLoss()(feat[0], target)[0]

    raw_bio = score["clip_bio_score"]
    bio = F.normalize(raw_bio.float() if semantic_fp32 else raw_bio, dim=-1)
    if bio.shape[1] > 200:
        bio = bio.unsqueeze(1)
    captions = batch["caption_feature"].float() if semantic_fp32 else batch["caption_feature"].to(bio.dtype)
    text_bio = F.normalize(captions[:, :bio.shape[1]], dim=-1)
    text_nonbio = F.normalize(captions[:, bio.shape[1]:], dim=-1)

    bio_losses = []
    bio_logits: dict[str, torch.Tensor] = {}
    labels = torch.arange(bio.shape[0], device=bio.device)
    for index in range(bio.shape[1]):
        logits = bio[:, index] @ text_bio[:, index].T
        bio_logits[str(index)] = logits
        bio_losses.append(F.cross_entropy(logits, labels))
    bio_total = sum(bio_losses)

    raw_reverse = score["clip_bio_reverse_score"]
    reverse_losses = []
    reverse_logits: dict[str, torch.Tensor] = {}
    for index in range(raw_reverse.shape[1]):
        if semantic_fp32:
            reverse_i = F.normalize(raw_reverse[:, index].float(), dim=-1)
            text_i = F.normalize(captions[:, bio.shape[1] + index].float(), dim=-1)
        else:
            reverse_i = F.normalize(raw_reverse[:, index], dim=-1).to(text_nonbio.dtype)
            text_i = text_nonbio[:, index]
        logits = reverse_i @ text_i.T
        reverse_logits[str(index)] = logits
        reverse_losses.append(F.cross_entropy(logits, labels))
    reverse0 = reverse_losses[0]
    reverse1 = reverse_losses[1] if len(reverse_losses) > 1 else reverse_losses[0]
    total = (
        float(cfg.MODEL.ID_LOSS_WEIGHT) * ce
        + float(cfg.MODEL.TRIPLET_LOSS_WEIGHT) * triplet
        + bio_total
        + sum(reverse_losses)
    )
    return {
        "ce": ce, "triplet": triplet, "bio": bio_total,
        "reverse0": reverse0, "reverse1": reverse1, "full": total,
    }, {"bio0": bio_logits["0"], "reverse0": reverse_logits["0"], "reverse1": reverse_logits["1"]}


def to_device(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def forward_once(model: torch.nn.Module, batch: dict[str, Any], cfg, mode: str) -> dict[str, Any]:
    semantic_fp32 = mode == "semantic_fp32"
    use_autocast = mode != "fp32_reference"
    model.train()
    with amp.autocast(enabled=use_autocast):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses, logits = component_losses(score, feat, batch, cfg, semantic_fp32)
    return {
        "losses": {key: value for key, value in losses.items()},
        "logits": logits,
        "score_dtypes": {key: str(value.dtype) for key, value in score.items() if isinstance(value, torch.Tensor)},
        "feat_dtypes": [str(value.dtype) for value in feat if isinstance(value, torch.Tensor)],
        "features": feat[0].detach().cpu().clone(),
    }


def run_backward(model: torch.nn.Module, base_state: dict[str, torch.Tensor], rng_state: dict[str, Any], batch: dict[str, Any], cfg, mode: str, component: str) -> dict[str, Any]:
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    out = forward_once(model, batch, cfg, mode)
    loss = out["losses"][component]
    scaler = None
    optimizer = None
    if mode == "fp32_reference":
        loss.backward()
        found_inf = 0.0
    else:
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        # This optimizer is diagnostic-only; no step is ever called. It lets
        # the official scaler unscale path be exercised without changing any
        # optimizer/configuration semantics.
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found.values()))
    names = dict(model.named_parameters())
    raw_grads = {name: None if names[name].grad is None else names[name].grad.detach().cpu().clone() for name in SELECTED_PARAMS}
    out_payload = {
        "mode": mode,
        "component": component,
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "found_inf": found_inf,
        "score_dtypes": out["score_dtypes"],
        "feat_dtypes": out["feat_dtypes"],
        "gradient_stats": {name: grad_stats(value) for name, value in raw_grads.items()},
        "raw_grads": raw_grads,
    }
    del optimizer, scaler, out, loss
    model.zero_grad(set_to_none=True)
    return out_payload


def forward_summary(model: torch.nn.Module, base_state: dict[str, torch.Tensor], rng_state: dict[str, Any], batch: dict[str, Any], cfg, mode: str) -> dict[str, Any]:
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    out = forward_once(model, batch, cfg, mode)
    return {
        "mode": mode,
        "losses": {key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())} for key, value in out["losses"].items()},
        "score_dtypes": out["score_dtypes"],
        "feat_dtypes": out["feat_dtypes"],
        "logits": {key: value.detach().cpu().clone() for key, value in out["logits"].items()},
    }


def main() -> None:
    cfg = make_config()
    torch.cuda.set_device(DEVICE)
    seed_all(SEED)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = to_device(raw)
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(DEVICE); model.train()
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    replay_rng = capture_rng_state()
    names = dict(model.named_parameters())
    parameter_dtypes: dict[str, int] = {}
    for name, parameter in names.items():
        parameter_dtypes[str(parameter.dtype)] = parameter_dtypes.get(str(parameter.dtype), 0) + 1

    print("PHASE2K_S_START", json.dumps({"seed": SEED, "batch_size": int(batch["image"].shape[0]), "initial_scale": INITIAL_SCALE, "dataset": "ltcc"}, sort_keys=True), flush=True)
    forwards = {mode: forward_summary(model, base_state, replay_rng, batch, cfg, mode) for mode in MODES}
    backwards: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        backwards[mode] = {}
        for component in COMPONENTS:
            print("PHASE2K_S_BACKWARD", mode, component, flush=True)
            backwards[mode][component] = run_backward(model, base_state, replay_rng, batch, cfg, mode, component)

    def pairwise_for(component: str, left: str, right: str) -> dict[str, Any]:
        a = backwards[left][component]; b = backwards[right][component]
        return {name: compare(a["raw_grads"].get(name), b["raw_grads"].get(name)) for name in SELECTED_PARAMS}

    gradient_comparison: dict[str, Any] = {}
    for component in COMPONENTS:
        gradient_comparison[component] = {
            "current_vs_fp32": pairwise_for(component, "current", "fp32_reference"),
            "semantic_fp32_vs_fp32": pairwise_for(component, "semantic_fp32", "fp32_reference"),
        }

    def forward_pair(left: str, right: str) -> dict[str, Any]:
        out = {}
        for key in ("bio0", "reverse0", "reverse1"):
            out[key] = compare(forwards[left]["logits"][key], forwards[right]["logits"][key])
        return out

    payload = {
        "phase": "2K-S",
        "config": {
            "dataset": "ltcc", "seed": SEED, "batch_size": int(batch["image"].shape[0]),
            "initial_scale": INITIAL_SCALE, "loss_path_current": "model autocast / released semantic FP16 arithmetic",
            "loss_path_candidate": "model autocast / explicit FP32 Bio+Reverse arithmetic",
            "loss_path_reference": "full FP32 diagnostic oracle", "nobio_index": [1, 2],
            "reverse_indexing": "[:, i]", "model": "current biasless natural construction",
            "sampler": type(sampler).__name__, "sampler_length": len(sampler),
        },
        "batch": {"image_shape": list(batch["image"].shape), "pid": [int(x) for x in batch["pid"].tolist()], "camid": [int(x) for x in batch["camid"].tolist()], "clothes": [int(x) for x in batch["clothes_id"].tolist()]},
        "parameter_dtypes": parameter_dtypes,
        "forward_summaries": {mode: {key: value for key, value in result.items() if key != "logits"} for mode, result in forwards.items()},
        "forward_comparisons": {
            "current_vs_fp32": forward_pair("current", "fp32_reference"),
            "semantic_fp32_vs_fp32": forward_pair("semantic_fp32", "fp32_reference"),
            "semantic_fp32_vs_current": forward_pair("semantic_fp32", "current"),
        },
        "backward_summaries": {
            mode: {component: {key: value for key, value in result.items() if key != "raw_grads"} for component, result in components.items()}
            for mode, components in backwards.items()
        },
        "gradient_comparisons": gradient_comparison,
        "notes": [
            "No optimizer.step() was executed.",
            "Current and semantic_fp32 both retain model autocast and the same loss weights; only semantic normalize/matmul arithmetic is changed for Candidate S.",
            "The explicit FP32 bridge casts autocast-produced semantic features and caption features to float32; gradients continue through the cast to the model outputs.",
            "All modes restore the same post-batch RNG state and reload the same model state before each isolated backward.",
        ],
    }
    out = ROOT / "diagnostics/phase2k_semantic_fp32_bridge.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2K_S_SUMMARY", json.dumps({"output": str(out), "current_total": payload["forward_summaries"]["current"]["losses"]["full"], "semantic_total": payload["forward_summaries"]["semantic_fp32"]["losses"]["full"], "fp32_total": payload["forward_summaries"]["fp32_reference"]["losses"]["full"]}, sort_keys=True), flush=True)
    model.to("cpu"); del model, loader, dataset, batch, base_state
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
