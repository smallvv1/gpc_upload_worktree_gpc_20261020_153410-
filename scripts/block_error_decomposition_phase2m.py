#!/usr/local/bin/python3
"""PHASE 2M-A: local-vs-upstream-vs-combined backward error decomposition.

Diagnostic only.  One fixed LTCC batch and one frozen Full DIFFER model state
are used.  The full current autocast path and a full-FP32 reference capture
the activation/gradient boundaries of every EVA block.  Each block is then
replayed in isolation to separate local block arithmetic error from error
already present in the upstream gradient.
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

# Importing this symbol separately avoids duplicating the released loss
# decomposition.  The conditional above is never evaluated as an import; it
# is kept out of the import list below for Python compatibility.
from layerwise_gradient_frontier_phase2l import component_losses  # noqa: E402
from loss.triplet_loss import TripletLoss  # noqa: E402


DEVICE = torch.device("cuda:0")
LOSS = "reverse1"
EPS = 1e-12


def stat_tensor(value: torch.Tensor | None) -> dict[str, Any]:
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
        "l2": float(torch.linalg.vector_norm(x).item()),
        "max_abs_finite": float(x[finite].abs().max().item()) if finite.any() else None,
        "finite": bool(finite.all().item()),
    }


def compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    x = a.detach().double().cpu().reshape(-1)
    y = b.detach().double().cpu().reshape(-1)
    if x.numel() != y.numel():
        return {"missing": False, "shape_mismatch": True, "a_numel": int(x.numel()), "b_numel": int(y.numel())}
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    return {
        "missing": False,
        "shape_mismatch": False,
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn > EPS and yn > EPS else None,
        "relative_l2": float(torch.linalg.vector_norm(d).item()) / (yn + EPS),
        "norm_ratio_a_over_b": xn / (yn + EPS),
        "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
        "a_l2": xn,
        "b_l2": yn,
        "a_finite": bool(torch.isfinite(x).all().item()),
        "b_finite": bool(torch.isfinite(y).all().item()),
    }


def install_hooks(model: torch.nn.Module):
    saved: dict[str, dict[str, torch.Tensor | None]] = {}
    handles = []
    for i, block in enumerate(model.blocks):
        name = f"block_{i}"
        saved[name] = {"input": None, "output": None}

        def hook(_module, inputs, output, name=name):
            x = inputs[0] if inputs else None
            if isinstance(x, torch.Tensor):
                x.retain_grad()
                saved[name]["input"] = x
            if isinstance(output, torch.Tensor):
                output.retain_grad()
                saved[name]["output"] = output

        handles.append(block.register_forward_hook(hook))
    return saved, handles


def capture_full(model, base_state, batch, rng_state, fp32: bool) -> dict[str, Any]:
    """Capture full-network boundaries and retain only CPU tensors needed for replay."""
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    saved, handles = install_hooks(model)
    triplet = TripletLoss()
    with amp.autocast(enabled=not fp32):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, triplet)
    loss = losses[LOSS]
    loss.backward()

    blocks: dict[str, Any] = {}
    tensors: dict[str, dict[str, torch.Tensor | None]] = {}
    for name, vals in saved.items():
        x = vals["input"]
        y = vals["output"]
        gx = x.grad if isinstance(x, torch.Tensor) and x.grad is not None else None
        gy = y.grad if isinstance(y, torch.Tensor) and y.grad is not None else None
        tensors[name] = {
            "input": x.detach().cpu().clone() if isinstance(x, torch.Tensor) else None,
            "output": y.detach().cpu().clone() if isinstance(y, torch.Tensor) else None,
            "g_in": gx.detach().cpu().clone() if isinstance(gx, torch.Tensor) else None,
            "g_out": gy.detach().cpu().clone() if isinstance(gy, torch.Tensor) else None,
        }
        blocks[name] = {
            "input_activation": stat_tensor(x),
            "output_activation": stat_tensor(y),
            "g_in": stat_tensor(gx),
            "g_out": stat_tensor(gy),
        }
    result = {
        "precision": "fp32_reference" if fp32 else "current_autocast",
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "components": {
            key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())}
            for key, value in losses.items()
        },
        "blocks": blocks,
        "_tensors": tensors,
    }
    for handle in handles:
        handle.remove()
    del score, feat, loss, losses, triplet, saved
    gc.collect()
    torch.cuda.empty_cache()
    return result


def replay_block(block: torch.nn.Module, x_cpu: torch.Tensor, gout_cpu: torch.Tensor, fp32: bool) -> torch.Tensor:
    """Return dX for a frozen block under a specified arithmetic path."""
    x = x_cpu.to(DEVICE, non_blocking=True).detach().requires_grad_(True)
    gout = gout_cpu.to(DEVICE, non_blocking=True)
    # The model has zero dropout/drop-path in this configuration.  Keeping it
    # in train mode still matches the formal construction.
    block.train()
    with amp.autocast(enabled=not fp32):
        y = block(x)
    gout = gout.to(dtype=y.dtype)
    dx = torch.autograd.grad(y, x, grad_outputs=gout, retain_graph=False, create_graph=False)[0]
    out = dx.detach().cpu().clone()
    del x, gout, y, dx
    torch.cuda.empty_cache()
    return out


def local_row(model, index: int, current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    block = model.blocks[index]
    name = f"block_{index}"
    c = current["_tensors"][name]
    r = reference["_tensors"][name]
    # A: fixed high-fidelity activation/upstream, compare local block paths.
    dx32_local = replay_block(block, r["input"], r["g_out"], fp32=True)
    dx16_local = replay_block(block, r["input"], r["g_out"], fp32=False)
    # B: fixed FP32 block, compare propagation of the actual current upstream.
    dx32_gout32 = dx32_local
    dx32_gout16 = replay_block(block, r["input"], c["g_out"], fp32=True)
    # C: current block with current captured activation/upstream.
    dx16_combined = replay_block(block, c["input"], c["g_out"], fp32=False)

    local_cmp = compare(dx16_local, dx32_local)
    prop_cmp = compare(dx32_gout16, dx32_gout32)
    total_cmp = compare(dx16_combined, dx32_gout32)
    eout = compare(c["g_out"], r["g_out"])["relative_l2"]
    ein = compare(c["g_in"], r["g_in"])["relative_l2"]
    return {
        "block": index,
        "local": local_cmp,
        "propagation": prop_cmp,
        "combined": total_cmp,
        "local_damage": None if local_cmp.get("cosine") is None else 1.0 - local_cmp["cosine"],
        "e_out_full": eout,
        "e_in_full": ein,
        "amplification": ein / (eout + EPS) if eout is not None else None,
        "input_activation": stat_tensor(r["input"]),
        "output_activation": stat_tensor(r["output"]),
        "fp32_g_out": stat_tensor(r["g_out"]),
        "fp16_g_out": stat_tensor(c["g_out"]),
        "fp32_g_in": stat_tensor(r["g_in"]),
        "fp16_g_in": stat_tensor(c["g_in"]),
        "full_boundary": {
            "g_out": compare(c["g_out"], r["g_out"]),
            "g_in": compare(c["g_in"], r["g_in"]),
        },
    }


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

    current = capture_full(model, base_state, batch, rng_state, fp32=False)
    reference = capture_full(model, base_state, batch, rng_state, fp32=True)
    current_tensors = current.pop("_tensors")
    reference_tensors = reference.pop("_tensors")
    current["_tensors"] = current_tensors
    reference["_tensors"] = reference_tensors

    rows = []
    for index in range(len(model.blocks)):
        print("PHASE2M_LOCAL_BLOCK", index, flush=True)
        rows.append(local_row(model, index, current, reference))

    serial_current = {k: v for k, v in current.items() if k != "_tensors"}
    serial_reference = {k: v for k, v in reference.items() if k != "_tensors"}
    payload = {
        "phase": "2M-A",
        "config": {
            "dataset": "ltcc",
            "seed": SEED,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "loss": "Reverse-1 individual factor",
            "reverse_indexing": "[:, i]",
            "nobio_index": [1, 2],
            "current_path": "model autocast enabled; loss outside autocast; unscaled diagnostic backward",
            "fp32_reference": "autocast disabled; same frozen model/batch/RNG",
            "optimizer_step": False,
            "scheduler_step": False,
            "regular_block_count": len(model.blocks),
        },
        "batch": {
            "image_shape": list(batch["image"].shape),
            "pid": [int(x) for x in batch["pid"].detach().cpu().tolist()],
            "camid": [int(x) for x in batch["camid"].detach().cpu().tolist()],
            "clothes": [int(x) for x in batch["clothes_id"].detach().cpu().tolist()],
        },
        "current": serial_current,
        "reference": serial_reference,
        "blocks": rows,
        "notes": [
            "Local replay A fixes the FP32 reference activation and upstream gradient.",
            "Propagation replay B uses the FP32 block with current versus reference upstream gradient.",
            "Combined replay C uses the current captured activation and upstream gradient.",
            "All diagnostics are unscaled backward replays; no optimizer or scheduler step was executed.",
        ],
    }
    out = ROOT / "diagnostics/phase2m_block_error_decomposition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2M_BLOCK_ERROR_DECOMPOSITION.csv"
    fields = [
        "block", "local_cosine", "local_rel_l2", "local_damage",
        "propagation_cosine", "propagation_rel_l2", "propagation_norm_ratio",
        "amplification", "total_cosine", "total_rel_l2",
        "input_norm", "output_norm", "fp32_g_in_norm", "fp16_g_in_norm",
        "fp32_g_out_norm", "fp16_g_out_norm", "e_out_full", "e_in_full",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "block": row["block"],
                "local_cosine": row["local"].get("cosine"),
                "local_rel_l2": row["local"].get("relative_l2"),
                "local_damage": row.get("local_damage"),
                "propagation_cosine": row["propagation"].get("cosine"),
                "propagation_rel_l2": row["propagation"].get("relative_l2"),
                "propagation_norm_ratio": row["propagation"].get("norm_ratio_a_over_b"),
                "amplification": row.get("amplification"),
                "total_cosine": row["combined"].get("cosine"),
                "total_rel_l2": row["combined"].get("relative_l2"),
                "input_norm": row["input_activation"].get("l2"),
                "output_norm": row["output_activation"].get("l2"),
                "fp32_g_in_norm": row["fp32_g_in"].get("l2"),
                "fp16_g_in_norm": row["fp16_g_in"].get("l2"),
                "fp32_g_out_norm": row["fp32_g_out"].get("l2"),
                "fp16_g_out_norm": row["fp16_g_out"].get("l2"),
                "e_out_full": row.get("e_out_full"),
                "e_in_full": row.get("e_in_full"),
            })
    print("PHASE2M_A_SUMMARY", json.dumps({"json": str(out), "csv": str(csv_path), "blocks": len(rows), "current_loss": serial_current["loss"], "reference_loss": serial_reference["loss"]}, sort_keys=True), flush=True)
    del model, dataset, loader, batch, current, reference
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
