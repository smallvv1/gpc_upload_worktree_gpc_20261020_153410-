#!/usr/local/bin/python3
"""PHASE 2M-C: global precision-policy screen (diagnostic only).

Each policy is applied to one frozen model state and one real LTCC batch.  A
single forward graph is differentiated separately for CE, Triplet, Bio,
Reverse-0, Reverse-1, and Full loss.  No optimizer/scheduler step is made.
The wrappers are local to this process and are restored after every policy.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from layerwise_gradient_frontier_phase2l import component_losses  # noqa: E402
from loss.triplet_loss import TripletLoss  # noqa: E402


DEVICE = torch.device("cuda:0")
SOURCES = ("ce", "triplet", "bio", "reverse0", "reverse1", "full")
BOUNDARIES = ("block_0", "block_10", "block_22")
POLICIES = ("current", "N", "R", "A", "AV", "M", "L", "RS", "B", "FULL32")
EPS = 1e-12


def compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    x = a.detach().double().cpu().reshape(-1)
    y = b.detach().double().cpu().reshape(-1)
    if x.numel() != y.numel():
        return {"missing": False, "shape_mismatch": True}
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    return {
        "missing": False,
        "shape_mismatch": False,
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn > EPS and yn > EPS else None,
        "relative_l2": float(torch.linalg.vector_norm(d).item()) / (yn + EPS),
        "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
        "a_l2": xn,
        "b_l2": yn,
        "a_finite": bool(torch.isfinite(x).all().item()),
        "b_finite": bool(torch.isfinite(y).all().item()),
    }


def stat(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"exists": False}
    x = value.detach().double()
    return {
        "exists": True,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "l2": float(torch.linalg.vector_norm(x).item()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "finite": bool(torch.isfinite(x).all().item()),
    }


def install_boundary_refs(model: nn.Module):
    saved: dict[str, torch.Tensor | None] = {name: None for name in BOUNDARIES}
    handles = []
    for index in (0, 10, 22):
        name = f"block_{index}"

        def hook(_module, inputs, _output, name=name):
            saved[name] = inputs[0] if inputs else None

        handles.append(model.blocks[index].register_forward_hook(hook))
    return saved, handles


def cast_tensor_args(args, kwargs):
    return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args), {
        k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
    }


def wrap_fp32_module(module: nn.Module, output_mode: str):
    original = module.forward

    def forward(*args, _original=original, _output_mode=output_mode, **kwargs):
        cast_args, cast_kwargs = cast_tensor_args(args, kwargs)
        with amp.autocast(enabled=False):
            output = _original(*cast_args, **cast_kwargs)
        if not isinstance(output, torch.Tensor):
            return output
        if _output_mode == "linear":
            return output.to(dtype=torch.get_autocast_dtype("cuda"))
        # LayerNorm is FP32 in the observed current path even when its input is
        # half, so it remains FP32 here.
        return output

    module.forward = forward
    return original


def wrap_mlp(module: nn.Module):
    original = module.forward

    def forward(*args, _original=original, **kwargs):
        cast_args, cast_kwargs = cast_tensor_args(args, kwargs)
        with amp.autocast(enabled=False):
            output = _original(*cast_args, **cast_kwargs)
        if isinstance(output, torch.Tensor):
            output = output.to(dtype=torch.get_autocast_dtype("cuda"))
        return output

    module.forward = forward
    return original


def wrap_residual_block(block: nn.Module):
    """Current block topology with only the two residual additions in FP32."""
    original = block.forward

    def forward(*args, _block=block, **kwargs):
        x = args[0]
        rope = args[1] if len(args) > 1 else kwargs.get("rope", None)
        attn_mask = args[2] if len(args) > 2 else kwargs.get("attn_mask", None)
        x_dtype = x.dtype
        if _block.gamma_1 is None:
            attn_out = _block.drop_path1(_block.attn(_block.norm1(x), rope=rope, attn_mask=attn_mask))
            x = (x.float() + attn_out.float()).to(dtype=x_dtype)
            mlp_out = _block.drop_path2(_block.mlp(_block.norm2(x)))
            x = (x.float() + mlp_out.float()).to(dtype=x_dtype)
        else:
            attn_out = _block.drop_path1(_block.gamma_1 * _block.attn(_block.norm1(x), rope=rope, attn_mask=attn_mask))
            x = (x.float() + attn_out.float()).to(dtype=x_dtype)
            mlp_out = _block.drop_path2(_block.gamma_2 * _block.mlp(_block.norm2(x)))
            x = (x.float() + mlp_out.float()).to(dtype=x_dtype)
        return x

    block.forward = forward
    return original


def wrap_fp32_block(block: nn.Module):
    original = block.forward

    def forward(*args, _original=original, **kwargs):
        cast_args, cast_kwargs = cast_tensor_args(args, kwargs)
        with amp.autocast(enabled=False):
            return _original(*cast_args, **cast_kwargs)

    block.forward = forward
    return original


def wrap_fp32_stream_block(block: nn.Module):
    original = block.forward

    def forward(*args, _original=original, **kwargs):
        cast_args, cast_kwargs = cast_tensor_args(args, kwargs)
        output = _original(*cast_args, **cast_kwargs)
        return output.float() if isinstance(output, torch.Tensor) else output

    block.forward = forward
    return original


def wrap_fake_quant_block(block: nn.Module):
    """FP32 mathematical block followed by an explicit half-rounding boundary.

    This is a diagnostic-only simulation of repeated FP32 -> FP16 -> FP32
    state quantization.  It does not use a PPU FP16 backward kernel for the
    block itself; the cast node remains in the autograd graph so its effect on
    the upstream gradient can be measured independently of operator kernels.
    """
    original = block.forward

    def forward(*args, _original=original, **kwargs):
        cast_args, cast_kwargs = cast_tensor_args(args, kwargs)
        with amp.autocast(enabled=False):
            output = _original(*cast_args, **cast_kwargs)
        if isinstance(output, torch.Tensor):
            return output.half().float()
        return output

    block.forward = forward
    return original


def custom_attention_forward(module: nn.Module, mode: str):
    """Explicit score/value split used only for A/AV diagnostics.

    The current LTCC construction uses q_proj/k_proj/v_proj and fused SDPA.
    To expose score versus value arithmetic, these diagnostics necessarily
    use the mathematically equivalent explicit attention decomposition; the
    scope is recorded in the output.
    """
    def forward(*args, _module=module, _mode=mode, **kwargs):
        x = args[0]
        rope = args[1] if len(args) > 1 else kwargs.get("rope", None)
        attn_mask = args[2] if len(args) > 2 else kwargs.get("attn_mask", None)
        B, N, C = x.shape
        if _module.qkv is not None:
            qkv_bias = torch.cat((_module.q_bias, _module.k_bias, _module.v_bias)) if _module.q_bias is not None else None
            qkv = F.linear(x, weight=_module.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, _module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = _module.q_proj(x).reshape(B, N, _module.num_heads, -1).transpose(1, 2)
            k = _module.k_proj(x).reshape(B, N, _module.num_heads, -1).transpose(1, 2)
            v = _module.v_proj(x).reshape(B, N, _module.num_heads, -1).transpose(1, 2)
        if rope is not None:
            from timm.layers import apply_rot_embed_cat
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)

        if _mode == "A":
            with amp.autocast(enabled=False):
                scores = (q.float() * _module.scale) @ k.float().transpose(-2, -1)
                if attn_mask is not None:
                    mask = attn_mask.to(torch.bool)
                    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
                probs = scores.softmax(dim=-1)
            probs = _module.attn_drop(probs.to(dtype=v.dtype))
            value_out = probs @ v
        else:  # AV: score/softmax current explicit path; value matmul FP32.
            with amp.autocast(enabled=True):
                scores = (q * _module.scale) @ k.transpose(-2, -1)
                if attn_mask is not None:
                    mask = attn_mask.to(torch.bool)
                    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
                probs = scores.softmax(dim=-1)
                probs = _module.attn_drop(probs)
            with amp.autocast(enabled=False):
                value_out = probs.float() @ v.float()
        out = value_out.transpose(1, 2).reshape(B, N, C)
        out = _module.norm(out)
        out = _module.proj(out)
        out = _module.proj_drop(out)
        return out

    return forward


def apply_policy(model: nn.Module, policy: str):
    originals = []
    if policy == "N":
        for name, module in model.named_modules():
            if module is model:
                continue
            cls = module.__class__.__name__.lower()
            if "layernorm" in cls or cls in {"rmsnorm", "norm"} or hasattr(module, "normalized_shape"):
                originals.append((module, module.forward))
                module.forward = wrap_fp32_module(module, "norm")
    elif policy == "R":
        for block in list(model.blocks) + [model.head_bio]:
            originals.append((block, wrap_residual_block(block)))
    elif policy in {"A", "AV"}:
        for module in model.modules():
            if module.__class__.__name__ == "EvaAttention":
                originals.append((module, module.forward))
                module.forward = custom_attention_forward(module, policy)
    elif policy == "AM":
        # The only permitted combination in this phase: two individually
        # medium-recovery, non-overlapping categories (attention score and
        # MLP core).  It remains diagnostic-only.
        originals.extend(apply_policy(model, "A"))
        originals.extend(apply_policy(model, "M"))
    elif policy == "M":
        for block in list(model.blocks) + [model.head_bio]:
            if hasattr(block, "mlp"):
                originals.append((block.mlp, wrap_mlp(block.mlp)))
    elif policy == "L":
        for parent in list(model.blocks) + [model.head_bio]:
            for module in parent.modules():
                if isinstance(module, nn.Linear):
                    originals.append((module, wrap_fp32_module(module, "linear")))
    elif policy == "RS":
        for block in list(model.blocks) + [model.head_bio]:
            originals.append((block, wrap_fp32_stream_block(block)))
    elif policy == "B":
        for block in list(model.blocks) + [model.head_bio]:
            originals.append((block, wrap_fp32_block(block)))
    elif policy == "FAKEQ":
        # Full FP32 blocks with explicit half-rounding at every block output.
        # This isolates repeated boundary quantization from FP16 block
        # backward kernels and is never imported by formal training.
        for block in list(model.blocks) + [model.head_bio]:
            originals.append((block, wrap_fake_quant_block(block)))
    elif policy in {"current", "FULL32"}:
        pass
    else:
        raise ValueError(policy)
    return originals


def restore_policy(originals):
    for module, original in reversed(originals):
        module.forward = original


def run_policy(model, base_state, batch, rng_state, policy: str):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    originals = apply_policy(model, policy)
    saved, handles = install_boundary_refs(model)
    try:
        with amp.autocast(enabled=policy not in {"FULL32", "FAKEQ"}):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, TripletLoss())
        values = {
            key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())}
            for key, value in losses.items()
        }
        gradients: dict[str, Any] = {}
        targets = [saved[name] for name in BOUNDARIES]
        for source in SOURCES:
            grads = torch.autograd.grad(losses[source], targets, retain_graph=True, allow_unused=True)
            gradients[source] = {
                name: stat(grad) for name, grad in zip(BOUNDARIES, grads)
            }
            gradients[source]["_tensors"] = {
                name: grad.detach().cpu().clone() if isinstance(grad, torch.Tensor) else None
                for name, grad in zip(BOUNDARIES, grads)
            }
        return {
            "policy": policy,
            "precision": "full_fp32" if policy == "FULL32" else ("fp32_with_boundary_fake_half" if policy == "FAKEQ" else "current_autocast_with_policy"),
            "losses": values,
            "gradients": {s: {n: {k: v for k, v in body[n].items() if k != "_tensors"} for n in BOUNDARIES} for s, body in gradients.items()},
            "_gradient_tensors": {s: body["_tensors"] for s, body in gradients.items()},
            "forward_finite": all(item["finite"] for item in values.values()),
        }
    finally:
        for handle in handles:
            handle.remove()
        restore_policy(originals)
        del score, feat, losses, saved
        gc.collect()
        torch.cuda.empty_cache()


def main():
    seed_all(SEED)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    results = {}
    for policy in POLICIES:
        print("PHASE2M_POLICY", policy, flush=True)
        results[policy] = run_policy(model, base_state, batch, rng_state, policy)
        print("PHASE2M_POLICY_DONE", json.dumps({"policy": policy, "loss": results[policy]["losses"]["full"]["value"]}, sort_keys=True), flush=True)

    reference = results["FULL32"]
    comparisons: dict[str, Any] = {}
    for policy, body in results.items():
        comparisons[policy] = {}
        for source in SOURCES:
            comparisons[policy][source] = {}
            for boundary in BOUNDARIES:
                comparisons[policy][source][boundary] = compare(
                    body["_gradient_tensors"][source][boundary],
                    reference["_gradient_tensors"][source][boundary],
                )

    # Recovery uses Reverse-1 block-0 input cosine, with the full-FP32 value as
    # the ceiling.  A candidate that produces an undefined/nonfinite gradient
    # remains in the matrix but cannot pass the gate.
    recovery = {}
    for policy in POLICIES:
        item = comparisons[policy]["reverse1"]["block_0"]
        c = item.get("cosine")
        current_c = comparisons["current"]["reverse1"]["block_0"].get("cosine")
        recovery[policy] = None if c is None or current_c is None else (c - current_c) / (1.0 - current_c)

    payload = {
        "phase": "2M-C",
        "config": {
            "dataset": "ltcc",
            "seed": SEED,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "nobio_index": [1, 2],
            "loss_sources": list(SOURCES),
            "boundaries": list(BOUNDARIES),
            "current_path": "autocast forward; official loss arithmetic outside autocast",
            "reference": "full FP32 forward/loss/gradient",
            "optimizer_step": False,
            "scheduler_step": False,
            "policy_scopes": {
                "N": "all LayerNorm-like modules FP32",
                "R": "both residual additions FP32",
                "A": "explicit FP32 QK^T/scaling/softmax; value path current",
                "AV": "explicit current score path; FP32 attention probability @ V",
                "M": "all EVA MLP modules FP32",
                "L": "all Linear modules in EVA blocks/head_bio FP32, cast output to autocast dtype",
                "RS": "block-to-block stream explicitly FP32; current internals",
                "B": "all EVA transformer/head_bio blocks FP32",
                "FULL32": "full FP32 oracle",
            },
        },
        "batch": {"image_shape": list(batch["image"].shape), "pid": [int(x) for x in batch["pid"].cpu().tolist()], "camid": [int(x) for x in batch["camid"].cpu().tolist()], "clothes": [int(x) for x in batch["clothes_id"].cpu().tolist()]},
        "losses": {p: body["losses"] for p, body in results.items()},
        "comparisons_to_full_fp32": comparisons,
        "recovery_reverse1_block0_input": recovery,
        "notes": [
            "All policies use the same frozen model state/batch/RNG and no optimizer step.",
            "A/AV require explicit attention decomposition to expose score versus value arithmetic; this scope is intentional and recorded.",
            "No policy is imported by the formal training entry point.",
        ],
    }
    out = ROOT / "diagnostics/phase2m_precision_policy_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2M_PRECISION_POLICY_SCREEN.csv"
    fields = ["candidate", "source", "boundary", "cosine_vs_full32", "relative_l2_vs_full32", "max_abs_diff", "recovery_reverse1_block0", "loss", "loss_dtype", "loss_finite"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy in POLICIES:
            for source in SOURCES:
                for boundary in BOUNDARIES:
                    cmp = comparisons[policy][source][boundary]
                    lv = results[policy]["losses"][source]
                    writer.writerow({
                        "candidate": policy,
                        "source": source,
                        "boundary": boundary,
                        "cosine_vs_full32": cmp.get("cosine"),
                        "relative_l2_vs_full32": cmp.get("relative_l2"),
                        "max_abs_diff": cmp.get("max_abs_diff"),
                        "recovery_reverse1_block0": recovery[policy],
                        "loss": lv["value"],
                        "loss_dtype": lv["dtype"],
                        "loss_finite": lv["finite"],
                    })

    print("PHASE2M_C_SUMMARY", json.dumps({"json": str(out), "csv": str(csv_path), "reverse1_block0_cosines": {p: comparisons[p]["reverse1"]["block_0"].get("cosine") for p in POLICIES}, "recovery": recovery}, sort_keys=True), flush=True)
    del model, dataset, loader, batch, results
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
