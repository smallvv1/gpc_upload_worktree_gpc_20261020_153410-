#!/usr/local/bin/python3
"""PHASE 2J-G: one natural-construction 25-epoch LTCC repeatability probe.

This is intentionally a bounded diagnostic.  It runs the frozen current
biasless Full DIFFER path to epoch 25, not to epoch 60, and is launched in a
fresh process for each repeat.  No source model/loss/processor code is
modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    INITIAL_SCALE,
    SEED,
    WORLD_SIZE,
    batch_fingerprint,
    build_natural_model,
    build_probe_loader,
    make_config,
    parameter_group_name,
    seed_all,
)
from data import build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402


DEVICE = torch.device("cuda:0")
EPOCHS = 25


class Capture:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values[str(tag)] = float(value)


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_eval_loader(cfg, dataset):
    _, transform_test = build_img_transforms(cfg)
    combined = ConcatDataset([
        ImageDataset(dataset.query, transform=transform_test),
        ImageDataset(dataset.gallery, transform=transform_test),
    ])
    return DataLoader(combined, batch_size=cfg.DATA.TEST_BATCH, num_workers=cfg.DATA.NUM_WORKERS,
                      pin_memory=False, drop_last=False, shuffle=False)


@torch.no_grad()
def evaluate(model, dataset, loader, general=False):
    evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=True) if general else R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=True)
    evaluator.reset(); model.eval()
    for raw in loader:
        image = raw["image"].to(DEVICE, non_blocking=True)
        cam = raw["camid"].to(DEVICE, non_blocking=True)
        feature = model(image, cam_label=cam)
        if not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError("non-finite retrieval feature during 25-epoch probe evaluation")
        evaluator.update((feature, raw["pid"], raw["camid"], raw["clothes_id"], raw["image_path"]))
    cmc, mean_ap, distance, _, _, qf, gf, _ = evaluator.compute()
    return {
        "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]), "mAP": float(mean_ap),
        "query_feature_shape": list(qf.shape), "gallery_feature_shape": list(gf.shape),
        "distance_shape": list(distance.shape), "distance_nan": int(np.isnan(distance).sum()),
        "distance_inf": int(np.isinf(distance).sum()),
    }


def selected_gradient_stats(names):
    selected = {
        "backbone": "blocks.0.attn.q_proj.weight",
        "classifier": "head.weight",
        "bio": "head_clip_bio.weight",
        "reverse0": "head_clip_bioReverse.0.head_fc.weight",
        "reverse1": "head_clip_bioReverse.1.head_fc.weight",
    }
    result = {}
    for group, name in selected.items():
        p = names[name]
        if p.grad is None:
            result[group] = {"exists": False}
            continue
        g = p.grad.detach().double(); finite = torch.isfinite(g)
        result[group] = {
            "exists": True, "nan_count": int(torch.isnan(g).sum().item()),
            "inf_count": int(torch.isinf(g).sum().item()),
            "max_abs_finite": float(g[finite].abs().max().item()) if finite.any() else None,
            "l2": float(torch.linalg.vector_norm(g).item()),
        }
    return result


def mean_tag(sums, tag):
    values = sums.get(tag, [])
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("P-A", "P-B", "P-C"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / f"diagnostics/phase2j_{args.label.lower().replace('-', '_')}_25epoch.json"

    cfg = make_config()
    cfg.defrost(); cfg.MODEL.DIST_TRAIN = False; cfg.freeze()
    torch.cuda.set_device(DEVICE)
    seed_all(SEED)
    dataset, loader, sampler = build_probe_loader(cfg)
    eval_loader = build_eval_loader(cfg, dataset)
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.to(DEVICE); model.train()
    names = dict(model.named_parameters())
    initial_hashes = {name: tensor_sha(names[name]) for name in (
        "head.weight", "head_clip_bio.weight", "head_clip_bioReverse.0.head_fc.weight", "head_clip_bioReverse.1.head_fc.weight")}
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    records = []
    all_fingerprints = []
    total_skips = total_nonfinite_loss = total_nonfinite_grad = total_update_fail = 0
    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch)
        model.train(); sums: dict[str, list[float]] = {}; train_acc_sum = 0.0; train_count = 0
        skipped = nonfinite_loss = nonfinite_grad = update_fail = 0
        first_fp = None
        for index, raw in enumerate(loader):
            batch_fp = batch_fingerprint(raw)
            if first_fp is None:
                first_fp = batch_fp
            if len(all_fingerprints) < EPOCHS * 2:
                all_fingerprints.append({"epoch": epoch, "iteration": index + 1, **batch_fp})
            batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            optimizer.zero_grad(set_to_none=True); optimizer_center.zero_grad(set_to_none=True)
            writer = Capture(); step = (epoch - 1) * len(loader) + index + 1
            scale_before = float(scaler.get_scale())
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
            total = loss_fn(score, feat, batch["pid"], batch["camid"], batch["caption_feature"], clothes_ids=batch["clothes_id"], train_writer=writer, step=step)
            writer.add_scalar("loss/total", total.detach().item(), step)
            finite_loss = bool(torch.isfinite(total.detach()).item()); nonfinite_loss += int(not finite_loss)
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            gs = selected_gradient_stats(names)
            finite_grad = all(item.get("nan_count", 0) == 0 and item.get("inf_count", 0) == 0 for item in gs.values() if item.get("exists", False))
            nonfinite_grad += int(not finite_grad)
            before = {group: names[name].detach().clone() for group, name in {
                "backbone": "blocks.0.attn.q_proj.weight", "classifier": "head.weight",
                "bio": "head_clip_bio.weight", "reverse0": "head_clip_bioReverse.0.head_fc.weight",
                "reverse1": "head_clip_bioReverse.1.head_fc.weight"}.items()}
            scaler.step(optimizer)
            found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            found_inf = float(sum(float(v.item()) for v in found.values()))
            step_skipped = found_inf > 0; skipped += int(step_skipped)
            deltas = {group: float((names[name].detach() - before[group]).float().abs().max().item()) for group, name in {
                "backbone": "blocks.0.attn.q_proj.weight", "classifier": "head.weight",
                "bio": "head_clip_bio.weight", "reverse0": "head_clip_bioReverse.0.head_fc.weight",
                "reverse1": "head_clip_bioReverse.1.head_fc.weight"}.items()}
            updated = (not step_skipped) and all(value > 0 for value in deltas.values()); update_fail += int(not updated)
            scaler.update(); scale_after = float(scaler.get_scale())
            for tag, value in writer.values.items():
                sums.setdefault(tag, []).append(float(value))
            cls_acc = float((score["cls_score"].argmax(-1) == batch["pid"]).float().mean().item()); train_acc_sum += cls_acc; train_count += 1
            del score, feat, total, batch
        cc = evaluate(model, dataset, eval_loader, general=False)
        general = evaluate(model, dataset, eval_loader, general=True)
        lr_values = [float(group["lr"]) for group in optimizer.param_groups]
        records.append({
            "epoch": epoch,
            "train_loss": mean_tag(sums, "loss/total"), "id_loss": mean_tag(sums, "loss/id"),
            "triplet_loss": mean_tag(sums, "loss/triplet"), "bio_loss": mean_tag(sums, "loss/clip_bio_0"),
            "reverse0_loss": mean_tag(sums, "loss/clip_bio_reverse_0"), "reverse1_loss": mean_tag(sums, "loss/clip_bio_reverse_1"),
            "id_acc": mean_tag(sums, "acc/id"), "bio_acc": mean_tag(sums, "acc/clip_bio_0"),
            "reverse0_acc": mean_tag(sums, "acc/clip_bio_reverse_0"), "reverse1_acc": mean_tag(sums, "acc/clip_bio_reverse_1"),
            "train_accuracy": train_acc_sum / train_count if train_count else None,
            "lr_normal": min(lr_values), "lr_bias": max(lr_values), "amp_scale_end": float(scaler.get_scale()),
            "skipped_steps": skipped, "nonfinite_loss": nonfinite_loss, "nonfinite_gradient_selected": nonfinite_grad,
            "parameter_update_failures": update_fail, "first_batch_fingerprint": first_fp,
            "cc": cc, "general": general,
        })
        total_skips += skipped; total_nonfinite_loss += nonfinite_loss; total_nonfinite_grad += nonfinite_grad; total_update_fail += update_fail
        print("PHASE2J_25E_PROGRESS", json.dumps({"label": args.label, "epoch": epoch, "cc_rank1": cc["rank1"], "cc_mAP": cc["mAP"], "skipped": skipped, "scale": scaler.get_scale()}, sort_keys=True), flush=True)

    final_hashes = {name: tensor_sha(names[name]) for name in initial_hashes}
    payload = {
        "phase": "2J-G", "label": args.label, "config": {
            "dataset": "ltcc", "seed": SEED, "world_size": WORLD_SIZE, "rank": 0, "epochs": EPOCHS,
            "batch_size": int(cfg.DATA.BATCH_SIZE), "amp_initial_scale": INITIAL_SCALE,
            "loss_path": "official model autocast / loss outside autocast", "losses": ["ce", "triplet", "clipBio", "clipBioReverse"],
            "nobio_index": [1, 2], "reverse_indexing": "[:, i]", "model": "current biasless natural construction",
            "sampler": type(sampler).__name__, "sampler_length": len(sampler), "effective_loader": "regular DataLoader rank0/world_size2 diagnostic stream",
        },
        "initial_head_hashes": initial_hashes, "final_head_hashes": final_hashes,
        "records": records, "fingerprints_first_two_per_epoch": all_fingerprints,
        "total_skips": total_skips, "total_nonfinite_loss": total_nonfinite_loss,
        "total_nonfinite_gradient_selected": total_nonfinite_grad, "total_parameter_update_failures": total_update_fail,
        "successful_optimizer_updates": EPOCHS * len(loader) - total_skips,
        "final_scale": float(scaler.get_scale()),
        "notes": [
            "Fresh process and natural construction are required for each P-A/P-B/P-C invocation.",
            "This diagnostic intentionally stops at epoch 25; it never enters a formal 60-epoch run.",
            "Every epoch evaluates LTCC clothes-changing and general protocols with the frozen Phase-1 evaluator.",
            "Reverse1 logging retains the released cumulative factor-0+factor-1 semantics.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2J_25E_SUMMARY", json.dumps({"label": args.label, "epochs": EPOCHS, "iterations_per_epoch": len(loader), "total_skips": total_skips, "updates": payload["successful_optimizer_updates"], "output": str(output)}, sort_keys=True), flush=True)
    model.to("cpu"); del model, loader, eval_loader, dataset, optimizer, optimizer_center, center, loss_fn, scheduler, scaler
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
