#!/usr/local/bin/python3
"""One fresh 200-step, same-seed LTCC Full DIFFER repeat (diagnostic only).

The caller launches this file in a fresh process for each label (R-A/R-B/R-C).
It uses the frozen LTCC configuration and AMP initial scale 1024, performs
real optimizer updates, and records the complete data fingerprint so that
repeatability can be measured without a seed sweep or a formal epoch run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

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
    make_config,
    parameter_group_name,
    seed_all,
)
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402


DEVICE = torch.device("cuda:0")
STEPS = 200
SELECTED = {
    "backbone": "blocks.0.attn.q_proj.weight",
    "classifier": "head.weight",
    "bio": "head_clip_bio.weight",
    "reverse0": "head_clip_bioReverse.0.head_fc.weight",
    "reverse1": "head_clip_bioReverse.1.head_fc.weight",
}


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gradient_stats(parameter: torch.nn.Parameter | None) -> dict[str, Any]:
    if parameter is None or parameter.grad is None:
        return {"exists": False}
    g = parameter.grad.detach()
    gd = g.double()
    finite = torch.isfinite(gd)
    return {
        "exists": True,
        "dtype": str(g.dtype),
        "numel": int(g.numel()),
        "nan_count": int(torch.isnan(gd).sum().item()),
        "inf_count": int(torch.isinf(gd).sum().item()),
        "max_abs_finite": float(gd[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(gd).item()) if g.numel() else 0.0,
    }


def scalar_values(writer, total: torch.Tensor) -> dict[str, float | None]:
    # These names deliberately retain the released writer semantics.  In
    # particular reverse1 is the cumulative factor-0+factor-1 value.
    tags = {
        "total_loss": "loss/total",
        "id_loss": "loss/id",
        "triplet_loss": "loss/triplet",
        "bio_loss": "loss/clip_bio_0",
        "reverse0_loss": "loss/clip_bio_reverse_0",
        "reverse1_loss": "loss/clip_bio_reverse_1",
        "id_acc": "acc/id",
        "bio_acc": "acc/clip_bio_0",
        "reverse0_acc": "acc/clip_bio_reverse_0",
        "reverse1_acc": "acc/clip_bio_reverse_1",
    }
    result = {name: (float(writer.values[tag]) if tag in writer.values else None) for name, tag in tags.items()}
    result["total_loss"] = float(total.detach().double().cpu())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("R-A", "R-B", "R-C"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / f"diagnostics/phase2j_{args.label.lower().replace('-', '_')}_200.json"

    cfg = make_config()
    cfg.defrost(); cfg.MODEL.DIST_TRAIN = False; cfg.freeze()
    torch.cuda.set_device(DEVICE)
    seed_all(SEED)

    dataset, loader, sampler = build_probe_loader(cfg)
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(DEVICE)
    model.train()
    names = dict(model.named_parameters())
    initial_hashes = {key: tensor_sha(names[name]) for key, name in SELECTED.items()}
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(1)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)

    records: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    skipped = 0
    nonfinite_loss = 0
    nonfinite_grad = 0
    update_failures = 0
    update_count = 0

    for iteration, raw in enumerate(loader, start=1):
        if iteration > STEPS:
            break
        fingerprints.append(batch_fingerprint(raw))
        batch = {
            key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        writer = type("Capture", (), {"values": {}, "add_scalar": lambda self, tag, value, step=None, *a, **k: self.values.__setitem__(str(tag), float(value))})()
        scale_before = float(scaler.get_scale())
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        total = loss_fn(
            score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
            clothes_ids=batch["clothes_id"], train_writer=writer, step=iteration,
        )
        writer.add_scalar("loss/total", total.detach().item(), iteration)
        finite_loss = bool(torch.isfinite(total.detach()).item())
        nonfinite_loss += int(not finite_loss)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        selected_grads = {
            group: gradient_stats(names[name]) for group, name in SELECTED.items()
        }
        gradients_finite = all(
            stats.get("nan_count", 0) == 0 and stats.get("inf_count", 0) == 0
            for stats in selected_grads.values() if stats.get("exists", False)
        )
        nonfinite_grad += int(not gradients_finite)
        before = {group: names[name].detach().clone() for group, name in SELECTED.items()}
        scaler.step(optimizer)
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found.values()))
        step_skipped = found_inf > 0
        skipped += int(step_skipped)
        deltas = {
            group: float((names[name].detach() - before[group]).float().abs().max().item())
            for group, name in SELECTED.items()
        }
        parameter_updated = (not step_skipped) and all(delta > 0.0 for delta in deltas.values())
        update_failures += int(not parameter_updated)
        update_count += int(not step_skipped)
        scaler.update()
        scale_after = float(scaler.get_scale())
        values = scalar_values(writer, total)
        id_acc = values.get("id_acc")
        # The released cls score is also the train ID accuracy; retain the
        # explicit value as an audit field if a writer variant omits it.
        if id_acc is None:
            id_acc = float((score["cls_score"].argmax(-1) == batch["pid"]).float().mean().item())
            values["id_acc"] = id_acc
        records.append({
            "iteration": iteration,
            "values": values,
            "loss_dtype": str(total.dtype),
            "loss_finite": finite_loss,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "step_skipped": step_skipped,
            "parameter_updated": parameter_updated,
            "optimizer_update_count": update_count,
            "parameter_deltas": deltas,
            "gradients_finite_selected": gradients_finite,
            "selected_gradients": selected_grads,
            "lr_normal": float(min(group["lr"] for group in optimizer.param_groups)),
            "lr_bias": float(max(group["lr"] for group in optimizer.param_groups)),
        })
        if iteration == 1 or iteration % 25 == 0:
            print("PHASE2J_REPEAT_PROGRESS", json.dumps({
                "label": args.label, "iteration": iteration, "scale_before": scale_before,
                "scale_after": scale_after, "found_inf": found_inf,
                "step_skipped": step_skipped, "parameter_updated": parameter_updated,
                "total_loss": values["total_loss"],
            }, sort_keys=True), flush=True)
        del score, feat, total, batch

    state_hashes_after = {key: tensor_sha(names[name]) for key, name in SELECTED.items()}
    optimizer_names = {}
    for index, group in enumerate(optimizer.param_groups):
        optimizer_names[str(index)] = {
            "names": [next((name for name, p in names.items() if p is parameter), "<unknown>") for parameter in group["params"]],
            "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
        }
    payload = {
        "phase": "2J-F",
        "label": args.label,
        "config": {
            "dataset": "ltcc", "seed": SEED, "world_size": 2, "rank": 0,
            "batch_size": int(cfg.DATA.BATCH_SIZE), "steps": STEPS,
            "amp_initial_scale": INITIAL_SCALE,
            "model": "current biasless natural construction",
            "loss_path": "official model autocast / loss outside autocast",
            "losses": ["ce", "triplet", "clipBio", "clipBioReverse"],
            "nobio_index": [1, 2], "reverse_indexing": "[:, i]",
            "sampler": type(sampler).__name__, "sampler_length": len(sampler),
            "effective_loader": "regular DataLoader, rank 0 of world_size 2 diagnostic stream",
        },
        "initial_head_hashes": initial_hashes,
        "final_head_hashes": state_hashes_after,
        "fingerprints": fingerprints,
        "records": records,
        "iterations": len(records),
        "skipped_steps": skipped,
        "successful_updates": update_count,
        "nonfinite_loss_count": nonfinite_loss,
        "nonfinite_gradient_iterations_selected": nonfinite_grad,
        "parameter_update_failures": update_failures,
        "initial_scale": records[0]["scale_before"] if records else None,
        "final_scale": records[-1]["scale_after"] if records else None,
        "optimizer_groups": optimizer_names,
        "notes": [
            "Fresh independent process expected for each label; no state is shared between repeats.",
            "This is a 200-step diagnostic, not a formal epoch and no evaluation is run.",
            "The sampler uses the proven rank-0/world-size-2 stream without NCCL initialization; all three repeats use exactly this same path.",
            "Fingerprint image checksums are post-transform CPU tensors; sample checksums include path/PID/camera/clothes order.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2J_REPEAT_SUMMARY", json.dumps({
        "label": args.label, "iterations": len(records), "skipped": skipped,
        "updates": update_count, "nonfinite_loss": nonfinite_loss,
        "nonfinite_grad_selected": nonfinite_grad, "update_failures": update_failures,
        "initial_hashes": initial_hashes, "output": str(output),
    }, sort_keys=True), flush=True)
    model.to("cpu")
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler, loader, dataset, model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
