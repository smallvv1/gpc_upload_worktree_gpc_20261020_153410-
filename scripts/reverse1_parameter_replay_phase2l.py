#!/usr/local/bin/python3
"""PHASE 2L-B: parameter-gradient nondeterminism frontier.

The layer-boundary replay can be deterministic even when a backward kernel
produces different parameter gradients.  This companion diagnostic repeats the
same Reverse-1 individual backward 20 times and compares representative
attention/MLP parameter gradients for every EVA block.  It never steps an
optimizer.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

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
from layerwise_gradient_frontier_phase2l import compare_tensors, component_losses, stat_tensor  # noqa: E402


DEVICE = torch.device("cuda:0")
REPEATS = 20


def main():
    cfg = make_config()
    seed_all(SEED); torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else raw[key] for key, value in raw.items()}
    post_data_rng = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    names = dict(model.named_parameters())
    selected = {}
    for i in range(len(model.blocks)):
        for suffix in ("attn.q_proj.weight", "attn.proj.weight", "mlp.fc2.weight"):
            key = f"blocks.{i}.{suffix}"
            if key in names:
                selected[f"block_{i}.{suffix}"] = key
    if hasattr(model, "head_bio"):
        for suffix in ("attn.q_proj.weight", "attn.proj.weight", "mlp.fc2.weight"):
            key = f"head_bio.{suffix}"
            if key in names:
                selected[f"head_bio.{suffix}"] = key

    first: dict[str, torch.Tensor] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    run_summaries = []
    for repeat in range(REPEATS):
        model.load_state_dict(base_state, strict=True)
        restore_rng_state(post_data_rng)
        model.zero_grad(set_to_none=True); model.train()
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, __import__("loss.triplet_loss", fromlist=["TripletLoss"]).TripletLoss())
        loss = losses["reverse1"]
        loss.backward()
        current = {}
        for alias, key in selected.items():
            grad = names[key].grad.detach().cpu().clone() if names[key].grad is not None else None
            current[alias] = grad
            if repeat == 0:
                if grad is not None:
                    first[alias] = grad
            else:
                comparisons[alias].append(compare_tensors(first.get(alias), grad))
        run_summaries.append({
            "repeat": repeat + 1,
            "loss": float(loss.detach().double().cpu()),
            "loss_dtype": str(loss.dtype),
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "gradients": {alias: stat_tensor(value) for alias, value in current.items()},
        })
        del score, feat, loss, losses, current

    by_boundary: dict[str, dict[str, Any]] = {}
    for alias, rows in comparisons.items():
        boundary, suffix = alias.split(".", 1)
        vals = [x for x in rows if not x.get("missing", False) and not x.get("shape_mismatch", False)]
        agg = by_boundary.setdefault(boundary, {"parameters": {}, "cosines": [], "relative_l2": [], "max_abs_diff": [], "bitwise_equal_count": 0, "comparison_count": 0})
        agg["parameters"][suffix] = {
            "pairwise": rows,
            "first_gradient": stat_tensor(first.get(alias)),
        }
        agg["cosines"].extend([x["cosine"] for x in vals if x.get("cosine") is not None])
        agg["relative_l2"].extend([x["relative_l2"] for x in vals if x.get("relative_l2") is not None])
        agg["max_abs_diff"].extend([x["max_abs_diff"] for x in vals if x.get("max_abs_diff") is not None])
        agg["bitwise_equal_count"] += sum(1 for x in vals if x.get("max_abs_diff") == 0.0)
        agg["comparison_count"] += len(vals)
    for boundary, agg in by_boundary.items():
        agg["pairwise_cosine_min"] = min(agg["cosines"]) if agg["cosines"] else None
        agg["pairwise_cosine_max"] = max(agg["cosines"]) if agg["cosines"] else None
        agg["pairwise_relative_l2_max"] = max(agg["relative_l2"]) if agg["relative_l2"] else None
        agg["pairwise_max_abs_diff_max"] = max(agg["max_abs_diff"]) if agg["max_abs_diff"] else None
        agg["bitwise_identical_percentage"] = 100.0 * agg["bitwise_equal_count"] / agg["comparison_count"] if agg["comparison_count"] else None
        del agg["cosines"], agg["relative_l2"], agg["max_abs_diff"], agg["bitwise_equal_count"], agg["comparison_count"]

    order = ["head_bio"] + [f"block_{i}" for i in range(len(model.blocks) - 1, -1, -1)]
    first_nonzero = None
    for boundary in order:
        if boundary not in by_boundary:
            continue
        if (by_boundary[boundary]["pairwise_max_abs_diff_max"] or 0.0) > 0.0:
            first_nonzero = boundary; break
    first_material = None
    for boundary in order:
        if boundary not in by_boundary:
            continue
        if (by_boundary[boundary]["pairwise_relative_l2_max"] or 0.0) > 1e-6:
            first_material = boundary; break

    payload = {
        "phase": "2L-B",
        "config": {"dataset": "ltcc", "seed": SEED, "repeats": REPEATS, "batch_size": int(cfg.DATA.BATCH_SIZE), "sampler": type(sampler).__name__, "sampler_length": len(sampler), "loss": "Reverse-1 individual factor loss", "optimizer_step": False, "scheduler_step": False},
        "batch": {"image_shape": list(batch["image"].shape), "pid": [int(x) for x in batch["pid"].cpu()], "camid": [int(x) for x in batch["camid"].cpu()], "clothes": [int(x) for x in batch["clothes_id"].cpu()]},
        "selected_parameters": selected,
        "first_nondeterministic_parameter_boundary": first_nonzero,
        "first_material_nondeterministic_parameter_boundary_rel_l2_gt_1e-6": first_material,
        "by_boundary": by_boundary,
        "runs": run_summaries,
        "notes": ["All repeats restore the same model state and post-loader RNG state.", "Boundary activation gradients and parameter gradients are distinct diagnostics; parameter gradients are compared here because 2J first exposed nondeterminism at a q-projection parameter.", "No optimizer or scheduler step was executed."],
    }
    out = ROOT / "diagnostics/phase2l_reverse1_parameter_nondeterminism.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2L_NONDETERMINISM_FRONTIER.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["boundary", "parameter", "pairwise_cosine_min", "pairwise_relative_l2_max", "pairwise_max_abs_diff_max", "bitwise_identical_percentage"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for boundary, agg in by_boundary.items():
            for parameter in agg["parameters"]:
                writer.writerow({"boundary": boundary, "parameter": parameter, "pairwise_cosine_min": agg["pairwise_cosine_min"], "pairwise_relative_l2_max": agg["pairwise_relative_l2_max"], "pairwise_max_abs_diff_max": agg["pairwise_max_abs_diff_max"], "bitwise_identical_percentage": agg["bitwise_identical_percentage"]})
    print("PHASE2L_NONDET_SUMMARY", json.dumps({"output": str(out), "csv": str(csv_path), "first_nonzero": first_nonzero, "first_material": first_material, "repeats": REPEATS}, sort_keys=True), flush=True)
    del model, dataset, loader, batch
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
