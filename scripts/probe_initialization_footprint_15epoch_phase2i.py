#!/usr/local/bin/python3
"""Targeted Phase 2I-C paired 15-epoch natural-construction probe.

This is run only after the Phase 2I first-200 diagnostic finds a meaningful
reverse-branch improvement for the native-construction candidate.  It keeps
the public training path unchanged; A is the current biasless model and C is
the native bias=True construction with its two biases zeroed and frozen.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import build_img_transforms
from data.dataset_loader import ImageDataset
from loss import make_loss
from model import build_model
from model import eva_meta_bio
from scripts.initialization_footprint_phase2i import (
    INITIAL_SCALE,
    RANK,
    SEED,
    WORLD_SIZE,
    NativeBiasClassifier,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
    rng_hashes,
    build_natural_model,
    parameter_group_name,
    ScalarCapture,
)
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


EPOCHS = 15


def build_eval_loader(local_cfg, dataset):
    _, transform_test = build_img_transforms(local_cfg)
    combined = ConcatDataset([
        ImageDataset(dataset.query, transform=transform_test),
        ImageDataset(dataset.gallery, transform=transform_test),
    ])
    return DataLoader(
        combined,
        batch_size=local_cfg.DATA.TEST_BATCH,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )


@torch.no_grad()
def evaluate(model, dataset, loader, device, general=False):
    evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=True) if general else R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=True)
    evaluator.reset()
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        cams = batch["camid"].to(device, non_blocking=True)
        feature = model(images, cam_label=cams)
        if not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError("non-finite retrieval feature during 15-epoch evaluation")
        evaluator.update((feature, batch["pid"], batch["camid"], batch["clothes_id"], batch["image_path"]))
    cmc, mean_ap, distance, _, _, qf, gf, _ = evaluator.compute()
    return {
        "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]), "mAP": float(mean_ap),
        "query_feature_shape": list(qf.shape), "gallery_feature_shape": list(gf.shape),
        "distance_shape": list(distance.shape), "distance_nan": int(np.isnan(distance).sum()), "distance_inf": int(np.isinf(distance).sum()),
    }


def mean_tag(values: dict[str, list[float]], tag: str):
    vals = values.get(tag, [])
    return float(np.mean(vals)) if vals else None


def train_15(label, model, local_cfg, dataset, loader, eval_loader, rng_state, device):
    restore_rng_state(rng_state)
    model.to(device)
    loss_fn, center = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center)
    scheduler = create_scheduler(local_cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    names = dict(model.named_parameters())
    records = []
    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch)
        model.train()
        sums: dict[str, list[float]] = {}
        acc_sum = 0.0
        acc_count = 0
        skipped = 0
        nonfinite = 0
        grad_nonfinite = 0
        for idx, raw in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            writer = ScalarCapture()
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
            step = idx + len(loader) * epoch
            total = loss_fn(score, feat, batch["pid"], batch["camid"], batch["caption_feature"], clothes_ids=batch["clothes_id"], train_writer=writer, step=step)
            # The released processor writes this scalar immediately after
            # loss_fn; retain the same total in the diagnostic epoch means.
            writer.add_scalar("loss/total", total.detach().item(), step)
            finite_loss = bool(torch.isfinite(total.detach()).item())
            nonfinite += int(not finite_loss)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            all_finite = True
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all().item()):
                    all_finite = False
                    break
            grad_nonfinite += int(not all_finite)
            scaler.step(optimizer)
            found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            found_inf = sum(float(value.item()) for value in found.values())
            skipped += int(found_inf > 0)
            scaler.update()
            for tag, value in writer.values.items():
                sums.setdefault(tag, []).append(float(value))
            if isinstance(score, dict):
                cls = score["cls_score"]
                acc = (cls.argmax(-1) == batch["pid"]).float().mean().item()
                acc_sum += float(acc)
                acc_count += 1
        cc = evaluate(model, dataset, eval_loader, device, general=False)
        general = evaluate(model, dataset, eval_loader, device, general=True)
        lr_values = [float(group["lr"]) for group in optimizer.param_groups]
        record = {
            "epoch": epoch,
            "train_loss": mean_tag(sums, "loss/total"),
            "id_loss": mean_tag(sums, "loss/id"),
            "triplet_loss": mean_tag(sums, "loss/triplet"),
            "bio_loss": mean_tag(sums, "loss/clip_bio_0"),
            "reverse0_loss": mean_tag(sums, "loss/clip_bio_reverse_0"),
            "reverse1_loss": mean_tag(sums, "loss/clip_bio_reverse_1"),
            "id_acc": mean_tag(sums, "acc/id"),
            "bio_acc": mean_tag(sums, "acc/clip_bio_0"),
            "reverse0_acc": mean_tag(sums, "acc/clip_bio_reverse_0"),
            "reverse1_acc": mean_tag(sums, "acc/clip_bio_reverse_1"),
            "train_accuracy": acc_sum / acc_count if acc_count else None,
            "lr_normal": min(lr_values) if lr_values else None,
            "lr_bias": max(lr_values) if lr_values else None,
            "amp_scale_end": float(scaler.get_scale()),
            "skipped_steps": skipped,
            "nonfinite_loss": nonfinite,
            "nonfinite_gradient": grad_nonfinite,
            "cc": cc,
            "general": general,
        }
        records.append(record)
        print("PHASE2I_15E_PROGRESS", json.dumps({"candidate": label, "epoch": epoch, "cc_rank1": cc["rank1"], "cc_mAP": cc["mAP"], "skipped": skipped, "scale": scaler.get_scale()}, sort_keys=True), flush=True)
    result = {"label": label, "records": records, "total_skipped": sum(r["skipped_steps"] for r in records), "total_nonfinite_loss": sum(r["nonfinite_loss"] for r in records), "total_nonfinite_gradient": sum(r["nonfinite_gradient"] for r in records)}
    model.to("cpu")
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def official_epoch_curve():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    paths = sorted(Path("/home/data/DIFFER_OfficialWeights/provenance/LTCC/train").glob("events.out.tfevents.*"))
    series = {}
    for path in paths:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for item in acc.Scalars(tag):
                series.setdefault(tag, {}).setdefault(int(item.step), float(item.value))
    out = {}
    for epoch in range(1, EPOCHS + 1):
        lo, hi = 596 + (epoch - 1) * 596, 596 + epoch * 596
        row = {"epoch": epoch}
        for tag, target in [("loss/total", "train_loss"), ("loss/id", "id_loss"), ("loss/triplet", "triplet_loss"), ("loss/clip_bio", "bio_loss"), ("loss/clip_bio_reverse_0", "reverse0_loss"), ("loss/clip_bio_reverse_1", "reverse1_loss"), ("acc/id", "id_acc"), ("acc/clip_bio", "bio_acc"), ("acc/clip_bio_reverse_0", "reverse0_acc"), ("acc/clip_bio_reverse_1", "reverse1_acc"), ("lr", "lr_normal")]:
            vals = [v for step, v in series.get(tag, {}).items() if lo <= step < hi]
            row[target] = float(np.mean(vals)) if vals else None
        row["cc_rank1"] = series.get("rank1", {}).get(epoch)
        row["cc_mAP"] = series.get("mAP", {}).get(epoch)
        out[epoch] = row
    return out


def distances(official, ours):
    names = ["train_loss", "id_loss", "triplet_loss", "bio_loss", "reverse0_loss", "reverse1_loss", "id_acc", "bio_acc", "reverse0_acc", "reverse1_acc", "cc_rank1", "cc_mAP"]
    result = {}
    for name in names:
        a, b = [], []
        for epoch in range(1, EPOCHS + 1):
            ov = official.get(epoch, {}).get(name)
            if name == "cc_rank1":
                ov = official.get(epoch, {}).get("cc_rank1")
            elif name == "cc_mAP":
                ov = official.get(epoch, {}).get("cc_mAP")
            rv = ours.get(epoch, {}).get(name)
            if rv is None or ov is None:
                continue
            a.append(float(ov)); b.append(float(rv))
        if not a:
            result[name] = {"count": 0, "rmse": None, "mae": None}
            continue
        arr_a, arr_b = np.asarray(a), np.asarray(b)
        result[name] = {"count": len(a), "rmse": float(np.sqrt(np.mean((arr_b-arr_a)**2))), "mae": float(np.mean(np.abs(arr_b-arr_a)))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostics/initialization_footprint_15epoch_phase2i.json"))
    args = parser.parse_args()
    local_cfg = make_config()
    torch.cuda.set_device(torch.device("cuda:0"))
    candidates = {}
    for label in ("A", "C"):
        seed_all(SEED)
        dataset, loader, sampler = build_probe_loader(local_cfg)
        model = build_natural_model(local_cfg, dataset.num_train_pids, dataset.num_camera, biasful=(label == "C"))
        if label == "C":
            for name, parameter in model.named_parameters():
                if name.endswith("head_fc.bias"):
                    parameter.data.zero_(); parameter.requires_grad_(False)
        rng = capture_rng_state()
        eval_loader = build_eval_loader(local_cfg, dataset)
        candidates[label] = {"model": model, "dataset": dataset, "loader": loader, "eval_loader": eval_loader, "rng": rng, "sampler": {"length": len(sampler), "world_size": WORLD_SIZE, "rank": RANK}}
        print("PHASE2I_15E_CONSTRUCTED", label, json.dumps({"state_keys": len(model.state_dict()), "rng": rng_hashes(), "sampler": candidates[label]["sampler"]}, sort_keys=True), flush=True)
    device = torch.device("cuda:0")
    results = {}
    for label in ("A", "C"):
        results[label] = train_15(label, candidates[label]["model"], local_cfg, candidates[label]["dataset"], candidates[label]["loader"], candidates[label]["eval_loader"], candidates[label]["rng"], device)
        candidates[label]["model"] = None
        candidates[label]["loader"] = None
        candidates[label]["eval_loader"] = None
        gc.collect(); torch.cuda.empty_cache()
    official = official_epoch_curve()
    ours = {}
    for label in results:
        epoch_rows = {}
        for row in results[label]["records"]:
            compact = {key: row.get(key) for key in [
                "train_loss", "id_loss", "triplet_loss", "bio_loss",
                "reverse0_loss", "reverse1_loss", "id_acc", "bio_acc",
                "reverse0_acc", "reverse1_acc", "lr_normal",
            ]}
            compact["cc_rank1"] = row["cc"]["rank1"]
            compact["cc_mAP"] = row["cc"]["mAP"]
            epoch_rows[row["epoch"]] = compact
        ours[label] = epoch_rows
    payload = {"phase": "2I-C", "config": {"seed": SEED, "initial_scale": INITIAL_SCALE, "world_size": WORLD_SIZE, "rank": RANK, "epochs": EPOCHS, "model_difference": "A bias=False; C native bias=True then zero/freeze"}, "official": official, "results": results, "distance_to_official": {label: distances(official, ours[label]) for label in ours}, "notes": ["A and C are independent natural constructions, not common-init models.", "No formal 60-epoch run is started; this probe stops at epoch 15."]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("PHASE2I_15E_SUMMARY", json.dumps({"distance": payload["distance_to_official"], "skips": {l: results[l]["total_skipped"] for l in results}}, sort_keys=True), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
