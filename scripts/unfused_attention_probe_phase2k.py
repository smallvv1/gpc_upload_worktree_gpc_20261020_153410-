#!/usr/local/bin/python3
"""PHASE 2K Candidate U: unfused attention, 200-step diagnostic.

This probe changes only timm's attention implementation selection before the
candidate model is constructed. Both candidates use the same frozen state,
the same materialized batches, loss, optimizer, scheduler, AMP scale, and
seed. No repository source is modified and no formal epoch run is started.
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
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    INITIAL_SCALE,
    SEED,
    batch_fingerprint,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402


DEVICE = torch.device("cuda:0")
STEPS = 200
SELECTED = {
    "backbone": "blocks.0.attn.q_proj.weight",
    "middle": "blocks.11.attn.q_proj.weight",
    "last": "blocks.22.attn.q_proj.weight",
    "classifier": "head.weight",
    "bio": "head_clip_bio.weight",
    "reverse0": "head_clip_bioReverse.0.head_fc.weight",
    "reverse1": "head_clip_bioReverse.1.head_fc.weight",
}
SCALAR_TAGS = {
    # These are the exact tags emitted by the current released make_loss.py
    # into the diagnostic Capture writer.  Bio uses factor suffix ``_0`` in
    # the current code; reverse_1 retains the released cumulative logging
    # semantics.
    "total_loss": "loss/total", "id_loss": "loss/id", "triplet_loss": "loss/triplet",
    "bio_loss": "loss/clip_bio_0", "reverse0_loss": "loss/clip_bio_reverse_0",
    "reverse1_loss": "loss/clip_bio_reverse_1", "id_acc": "acc/id",
    "bio_acc": "acc/clip_bio_0", "reverse0_acc": "acc/clip_bio_reverse_0",
    "reverse1_acc": "acc/clip_bio_reverse_1",
}

OFFICIAL_SCALAR_TAGS = {
    # These are the exact normalized columns in the frozen official first-200
    # extraction.  The official event used unsuffixed Bio names while the
    # reverse factors retained numeric suffixes.
    "total_loss": "loss/total", "id_loss": "loss/id", "triplet_loss": "loss/triplet",
    "bio_loss": "loss/clip_bio", "reverse0_loss": "loss/clip_bio_reverse_0",
    "reverse1_loss": "loss/clip_bio_reverse_1", "id_acc": "acc/id",
    "bio_acc": "acc/clip_bio", "reverse0_acc": "acc/clip_bio_reverse_0",
    "reverse1_acc": "acc/clip_bio_reverse_1",
}


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def grad_stats(parameter: torch.nn.Parameter) -> dict[str, Any]:
    if parameter.grad is None:
        return {"exists": False}
    g = parameter.grad.detach().double()
    finite = torch.isfinite(g)
    return {
        "exists": True, "dtype": str(parameter.grad.dtype), "numel": int(g.numel()),
        "nan_count": int(torch.isnan(g).sum().item()), "inf_count": int(torch.isinf(g).sum().item()),
        "max_abs_finite": float(g[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(g).item()), "finite": bool(finite.all().item()),
    }


def capture_writer():
    class Capture:
        def __init__(self): self.values = {}
        def add_scalar(self, tag, value, step=None, *args, **kwargs): self.values[str(tag)] = float(value)
    return Capture()


def run_candidate(label: str, model: torch.nn.Module, base_state: dict[str, torch.Tensor], batches: list[dict[str, Any]], rng_state: dict[str, Any], cfg, dataset) -> dict[str, Any]:
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.to(DEVICE); model.train()
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(1)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    names = dict(model.named_parameters())
    initial_hashes = {group: tensor_sha(names[name]) for group, name in SELECTED.items()}
    records = []
    skipped = nonfinite_loss = nonfinite_grad = update_failures = updates = 0
    for iteration, raw in enumerate(batches, start=1):
        batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
        optimizer.zero_grad(set_to_none=True); optimizer_center.zero_grad(set_to_none=True)
        writer = capture_writer(); scale_before = float(scaler.get_scale())
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        total = loss_fn(score, feat, batch["pid"], batch["camid"], batch["caption_feature"], clothes_ids=batch["clothes_id"], train_writer=writer, step=iteration)
        finite_loss = bool(torch.isfinite(total.detach()).item()); nonfinite_loss += int(not finite_loss)
        scaler.scale(total).backward(); scaler.unscale_(optimizer)
        grads = {group: grad_stats(names[name]) for group, name in SELECTED.items()}
        finite_grad = all(v.get("finite", False) for v in grads.values() if v.get("exists", False)); nonfinite_grad += int(not finite_grad)
        before = {group: names[name].detach().clone() for group, name in SELECTED.items()}
        scaler.step(optimizer)
        found_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(v.item()) for v in found_state.values()))
        step_skipped = found_inf > 0; skipped += int(step_skipped); updates += int(not step_skipped)
        deltas = {group: float((names[name].detach() - before[group]).float().abs().max().item()) for group, name in SELECTED.items()}
        updated = (not step_skipped) and all(value > 0 for value in deltas.values()); update_failures += int(not updated)
        scaler.update(); scale_after = float(scaler.get_scale())
        values = {key: writer.values.get(tag) for key, tag in SCALAR_TAGS.items()}; values["total_loss"] = float(total.detach().double().cpu())
        records.append({
            "iteration": iteration, "values": values, "loss_dtype": str(total.dtype), "loss_finite": finite_loss,
            "scale_before": scale_before, "scale_after": scale_after, "found_inf": found_inf,
            "step_skipped": step_skipped, "parameter_updated": updated, "optimizer_update_count": updates,
            "parameter_deltas": deltas, "selected_gradients": grads, "gradient_finite_selected": finite_grad,
            "lr_normal": float(min(group["lr"] for group in optimizer.param_groups)),
            "lr_bias": float(max(group["lr"] for group in optimizer.param_groups)),
        })
        if iteration == 1 or iteration % 50 == 0:
            print("PHASE2K_U_PROGRESS", json.dumps({"label": label, "iteration": iteration, "scale": scale_after, "skipped": step_skipped, "total_loss": values["total_loss"]}, sort_keys=True), flush=True)
        del score, feat, total, batch
    final_hashes = {group: tensor_sha(names[name]) for group, name in SELECTED.items()}
    groups = []
    name_by_id = {id(p): n for n, p in names.items()}
    for index, group in enumerate(optimizer.param_groups):
        groups.append({"index": index, "names": [name_by_id.get(id(p), "<unknown>") for p in group["params"]], "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])})
    result = {"label": label, "records": records, "iterations": len(records), "skipped_steps": skipped, "successful_updates": updates, "nonfinite_loss_count": nonfinite_loss, "nonfinite_gradient_iterations_selected": nonfinite_grad, "parameter_update_failures": update_failures, "initial_scale": records[0]["scale_before"] if records else None, "final_scale": records[-1]["scale_after"] if records else None, "initial_hashes": initial_hashes, "final_hashes": final_hashes, "optimizer_groups": groups}
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler
    model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    return result


def read_official() -> dict[str, list[float]]:
    import csv
    path = ROOT / "reports/OFFICIAL_FIRST200_SCALARS.csv"
    rows = list(csv.DictReader(path.open()))
    out = {}
    for name, tag in OFFICIAL_SCALAR_TAGS.items():
        out[name] = [float(row[tag]) for row in rows]
    return out


def distance(official: list[float], candidate: list[float]) -> dict[str, float | None]:
    n = min(len(official), len(candidate)); x = np.asarray(official[:n], dtype=np.float64); y = np.asarray(candidate[:n], dtype=np.float64); d = y - x
    rmse = float(np.sqrt(np.mean(d * d))); mae = float(np.mean(np.abs(d)))
    sx = float(np.std(x)); sy = float(np.std(y)); corr = float(np.corrcoef(x, y)[0, 1]) if sx > 1e-15 and sy > 1e-15 else None
    return {"rmse": rmse, "mae": mae, "pearson": corr, "normalized_rmse": rmse / (sx + 1e-12)}


def main() -> None:
    cfg = make_config(); cfg.defrost(); cfg.MODEL.DIST_TRAIN = False; cfg.freeze()
    torch.cuda.set_device(DEVICE); seed_all(SEED)
    dataset, loader, sampler = build_probe_loader(cfg)
    batches = []
    fingerprints = []
    for index, raw in enumerate(loader):
        if index >= STEPS: break
        batches.append({key: value.clone() if isinstance(value, torch.Tensor) else raw[key] for key, value in raw.items()})
        fingerprints.append(batch_fingerprint(raw))
    post_data_rng = capture_rng_state()

    # Build the current fused model first and retain its complete state. The U
    # model is built after set_fused_attn(False), then loaded from this exact
    # state so the only experiment variable is the attention implementation.
    import timm
    timm.layers.set_fused_attn(True)
    fused_model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    fused_model.to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in fused_model.state_dict().items()}
    # Build U before running either candidate; all stochastic state is restored
    # to post-data materialization before each training loop.
    timm.layers.set_fused_attn(False)
    unfused_model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    unfused_model.to(DEVICE)
    flags = {name: bool(module.fused_attn) for name, module in unfused_model.named_modules() if hasattr(module, "fused_attn")}
    if not flags or any(flags.values()):
        raise RuntimeError(f"Candidate U construction did not produce all unfused attention modules: {flags}")
    timm.layers.set_fused_attn(True)
    fused_flags = {name: bool(module.fused_attn) for name, module in fused_model.named_modules() if hasattr(module, "fused_attn")}
    if not fused_flags or not all(fused_flags.values()):
        raise RuntimeError(f"Current reference construction did not produce all fused attention modules: {fused_flags}")

    # The model state is common; optimizer and scaler are fresh for each mode.
    current = run_candidate("current_fused", fused_model, base_state, batches, post_data_rng, cfg, dataset)
    unfused = run_candidate("U_unfused", unfused_model, base_state, batches, post_data_rng, cfg, dataset)
    official = read_official()
    distances = {}
    for result in (current, unfused):
        arrays = {name: [record["values"].get(name) for record in result["records"]] for name in SCALAR_TAGS}
        distances[result["label"]] = {name: distance(official[name], arrays[name]) for name in SCALAR_TAGS}
        usable = [item["normalized_rmse"] for item in distances[result["label"]].values() if item["normalized_rmse"] is not None and math.isfinite(item["normalized_rmse"])]
        distances[result["label"]]["composite_score"] = float(np.mean(usable)) if usable else None

    same_batch = [x["sample_sha256"] for x in fingerprints]
    payload = {
        "phase": "2K-U", "config": {"dataset": "ltcc", "seed": SEED, "world_size": 2, "rank": 0, "steps": STEPS, "batch_size": int(cfg.DATA.BATCH_SIZE), "amp_initial_scale": INITIAL_SCALE, "loss_path": "official model autocast / loss outside autocast", "losses": ["ce", "triplet", "clipBio", "clipBioReverse"], "nobio_index": [1, 2], "reverse_indexing": "[:, i]", "sampler": type(sampler).__name__, "sampler_length": len(sampler), "attention_reference": "timm fused_attn=True", "attention_candidate": "timm fused_attn=False before construction"},
        "attention_flags": {"fused_model": fused_flags, "unfused_model": flags},
        "batch_fingerprints": fingerprints,
        "current": current, "unfused": unfused,
        "official_first200": {"path": str(ROOT / "reports/OFFICIAL_FIRST200_SCALARS.csv"), "record_count": len(next(iter(official.values()))), "step_range": [596, 795]},
        "vs_official": distances,
        "notes": ["No optimizer state is shared between current and U; both loops start from the same frozen model state and post-data RNG state.", "U is a numerical diagnostic only. No repository source or formal configuration is changed.", "The same materialized transformed batches are used for both candidates.", "The official Reverse-1 scalar retains released cumulative factor-0+factor-1 logging semantics."],
    }
    out = ROOT / "diagnostics/phase2k_unfused_attention_200.json"; out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2K_U_SUMMARY", json.dumps({"output": str(out), "current_score": distances["current_fused"]["composite_score"], "unfused_score": distances["U_unfused"]["composite_score"], "current_skips": current["skipped_steps"], "unfused_skips": unfused["skipped_steps"], "all_unfused": all(not x for x in flags.values())}, sort_keys=True), flush=True)
    fused_model.to("cpu"); unfused_model.to("cpu"); del fused_model, unfused_model, dataset, loader, batches, current, unfused
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
