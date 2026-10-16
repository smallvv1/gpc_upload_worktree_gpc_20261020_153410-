#!/usr/local/bin/python3
"""Qualify one PPU GradScaler initial scale on 100 real LTCC updates."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from loss import make_loss
from loss.triplet_loss import TripletLoss
from model import build_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from config import cfg
from scripts.localize_amp_overflow_phase1 import GROUPS, group_stats, nonfinite_parameters, set_seed


FACTOR_NAMES = {0: "biometric", 1: "hair", 2: "clothing", 3: "pose", 4: "interaction", 5: "environment"}


def make_dataset_config(dataset_name: str):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(REPO_ROOT / f"configs/{dataset_name}/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.DATA.ROOT = "/home/data"
    if dataset_name == "ltcc":
        local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
        local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    else:
        local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/PRCC"
        local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-L-14"
    local_cfg.freeze()
    return local_cfg


class ScalarCapture:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


def parameter_group(name: str) -> str:
    if name.startswith("head_clip_bioReverse."):
        return "Reverse"
    if name.startswith(("head_bio.", "norm_bio.", "head_clip_bio.")):
        return "Bio"
    if name.startswith("head."):
        return "Classifier"
    return "EVA"


def dtype_audit(model: torch.nn.Module) -> dict:
    audit = {}
    for group in ("EVA", "Classifier", "Bio", "Reverse"):
        parameters = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter_group(name) == group
        ]
        counts = Counter(str(parameter.dtype) for parameter in parameters)
        numel = Counter()
        for parameter in parameters:
            numel[str(parameter.dtype)] += parameter.numel()
        audit[group] = {
            "parameter_count": len(parameters),
            "numel": sum(parameter.numel() for parameter in parameters),
            "dtype_parameter_count": dict(counts),
            "dtype_numel": dict(numel),
        }
    return audit


def detached_loss_parts(score, feat, targets, captions, nonbio_indices) -> dict[str, torch.Tensor]:
    labels = torch.arange(targets.shape[0], device=targets.device)
    bio = score["clip_bio_score"].detach()
    if bio.ndim == 2:
        bio = bio.unsqueeze(1)
    reverse = score["clip_bio_reverse_score"].detach()
    captions = captions.type_as(bio)
    bio = F.normalize(bio, dim=-1)
    reverse = F.normalize(reverse, dim=-1)
    text_bio = F.normalize(captions[:, : bio.shape[1]], dim=-1)
    text_nonbio = F.normalize(captions[:, bio.shape[1] :], dim=-1)
    parts = {
        "id": F.cross_entropy(score["cls_score"].detach(), targets),
        "triplet": TripletLoss()(feat[0].detach(), targets)[0],
        "bio": F.cross_entropy(bio[:, 0] @ text_bio[:, 0].T, labels),
    }
    for local_index, factor_index in enumerate(nonbio_indices):
        factor_name = FACTOR_NAMES.get(int(factor_index), f"factor_{factor_index}")
        parts[f"reverse_{factor_name}"] = F.cross_entropy(
            reverse[:, local_index].type_as(text_nonbio) @ text_nonbio[:, local_index].T,
            labels,
        )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ltcc", "prcc"), default="ltcc")
    parser.add_argument("--initial-scale", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    local_cfg = make_dataset_config(args.dataset)
    set_seed(local_cfg.SOLVER.SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dataset = build_dataset(local_cfg)
    train_transform, _ = build_img_transforms(local_cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=local_cfg.DATA.NUM_INSTANCES,
        num_replicas=1,
        rank=0,
        seed=local_cfg.SOLVER.SEED,
    )
    loader = DataLoader(
        ImageDataset(dataset.train, transform=train_transform),
        sampler=sampler,
        batch_size=local_cfg.DATA.BATCH_SIZE,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(local_cfg.SOLVER.SEED),
    )
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    audit = dtype_audit(model)
    print("PARAMETER_DTYPE_AUDIT", json.dumps(audit, sort_keys=True))
    unexpected_fp16 = any(
        dtype in ("torch.float16", "torch.bfloat16")
        for group in audit.values() for dtype in group["dtype_parameter_count"]
    )
    print("UNEXPECTED_FP16_OR_BF16_MASTER_PARAMETERS:", unexpected_fp16)
    if unexpected_fp16:
        raise RuntimeError("Master parameter dtype audit failed; scale qualification is blocked")

    loss_fn, center_criterion = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center_criterion)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(local_cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler(init_scale=args.initial_scale)
    diagnostics_dir = REPO_ROOT / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    csv_path = diagnostics_dir / f"{args.dataset}_amp_scale{int(args.initial_scale)}_{args.iterations}iter.csv"
    fields = [
        "iteration", "scale_before", "scale_after", "found_inf", "step_skipped",
        "parameter_updated", "backbone_delta", "classifier_delta", "lr_normal", "lr_bias",
        "id_value", "id_dtype", "triplet_value", "triplet_dtype", "bio_value", "bio_dtype",
    ]
    for factor_index in local_cfg.DATA.NOBIO_INDEX:
        factor_name = FACTOR_NAMES.get(int(factor_index), f"factor_{factor_index}")
        fields.extend([f"reverse_{factor_name}_value", f"reverse_{factor_name}_dtype"])
    fields.extend(["total_value", "total_dtype", "loss_finite"])
    for group in GROUPS:
        key = group.lower().replace(" ", "_").replace("/", "_")
        fields.extend([f"{key}_nan_count", f"{key}_inf_count", f"{key}_max_abs_finite"])

    skipped = 0
    nonfinite_losses = 0
    nonfinite_grad_iterations = 0
    update_failures = 0
    successful_updates = 0
    first_failure = None
    representative_names = (
        "patch_embed.proj.weight",
        "blocks.0.attn.q_proj.weight",
        "blocks.0.mlp.fc1_g.weight",
        "head.weight",
        "head_clip_bio.weight",
        "head_clip_bioReverse.0.head_fc.weight",
    )
    loader_iterator = iter(loader)
    with csv_path.open("w", newline="") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fields)
        writer_csv.writeheader()
        for iteration in range(1, args.iterations + 1):
            batch = next(loader_iterator)
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            writer = ScalarCapture()
            scale_before = float(scaler.get_scale())
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
            total_loss = loss_fn(
                score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
                clothes_ids=batch["clothes_id"], train_writer=writer, step=iteration,
            )
            parts = detached_loss_parts(
                score, feat, batch["pid"], batch["caption_feature"], local_cfg.DATA.NOBIO_INDEX,
            )
            losses_finite = bool(torch.isfinite(total_loss)) and all(bool(torch.isfinite(value)) for value in parts.values())
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            groups = {group: group_stats(model, prefixes) for group, prefixes in GROUPS.items()}
            gradients_finite = all(stats["nan_count"] == 0 and stats["inf_count"] == 0 for stats in groups.values())
            if iteration == 1:
                representative = {}
                named = dict(model.named_parameters())
                for name in representative_names:
                    parameter = named[name]
                    representative[name] = {
                        "parameter_dtype": str(parameter.dtype),
                        "gradient_dtype": str(parameter.grad.dtype) if parameter.grad is not None else None,
                    }
                print("REPRESENTATIVE_PARAMETER_GRADIENT_DTYPES", json.dumps(representative, sort_keys=True))

            found_inf_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            found_inf = sum(float(value.item()) for value in found_inf_state.values())
            monitored = {
                "backbone": model.patch_embed.proj.weight,
                "classifier": model.head.weight,
            }
            before = {name: parameter.detach().clone() for name, parameter in monitored.items()}
            scaler.step(optimizer)
            deltas = {
                name: float((parameter.detach() - before[name]).float().abs().max())
                for name, parameter in monitored.items()
            }
            parameter_updated = all(delta > 0 for delta in deltas.values())
            step_skipped = found_inf > 0
            scaler.update()
            scale_after = float(scaler.get_scale())
            lr_values = sorted({float(group["lr"]) for group in optimizer.param_groups})

            row = {
                "iteration": iteration,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "found_inf": found_inf,
                "step_skipped": step_skipped,
                "parameter_updated": parameter_updated,
                "backbone_delta": deltas["backbone"],
                "classifier_delta": deltas["classifier"],
                "lr_normal": lr_values[0],
                "lr_bias": lr_values[-1],
                "total_value": float(total_loss.detach()),
                "total_dtype": str(total_loss.dtype),
                "loss_finite": losses_finite,
            }
            for name, value in parts.items():
                row[f"{name}_value"] = float(value)
                row[f"{name}_dtype"] = str(value.dtype)
            for group, stats in groups.items():
                key = group.lower().replace(" ", "_").replace("/", "_")
                row[f"{key}_nan_count"] = stats["nan_count"]
                row[f"{key}_inf_count"] = stats["inf_count"]
                row[f"{key}_max_abs_finite"] = stats["max_abs_finite"]
            writer_csv.writerow(row)
            csv_file.flush()

            nonfinite_losses += int(not losses_finite)
            nonfinite_grad_iterations += int(not gradients_finite)
            skipped += int(step_skipped)
            update_failures += int(not parameter_updated)
            successful_updates += int(parameter_updated)
            failed = not losses_finite or not gradients_finite or step_skipped or not parameter_updated or scale_after < scale_before
            if failed and first_failure is None:
                named_bad = nonfinite_parameters(model)
                first_failure = {
                    "iteration": iteration,
                    "paths": batch.get("img_path"),
                    "losses": {name: float(value) for name, value in parts.items()},
                    "total": float(total_loss.detach()),
                    "groups": groups,
                    "first_nonfinite_parameter": named_bad[0] if named_bad else None,
                    "found_inf": found_inf,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "parameter_updated": parameter_updated,
                }
            if iteration == 1 or iteration % 10 == 0 or failed:
                print("QUALIFICATION_PROGRESS", json.dumps({
                    "iteration": iteration,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "found_inf": found_inf,
                    "step_skipped": step_skipped,
                    "parameter_updated": parameter_updated,
                    "loss": float(total_loss.detach()),
                    "gradients_finite": gradients_finite,
                }, sort_keys=True))
            if failed:
                break

    completed = iteration
    status = (
        completed == args.iterations
        and nonfinite_losses == 0
        and nonfinite_grad_iterations == 0
        and skipped == 0
        and update_failures == 0
    )
    summary = {
        "dataset": args.dataset,
        "initial_scale": args.initial_scale,
        "requested_iterations": args.iterations,
        "completed_iterations": completed,
        "nonfinite_loss_count": nonfinite_losses,
        "nonfinite_gradient_iterations": nonfinite_grad_iterations,
        "skipped_step_count": skipped,
        "parameter_update_failure_count": update_failures,
        "successful_optimizer_updates": successful_updates,
        "final_scale": float(scaler.get_scale()),
        "first_failure": first_failure,
        "csv": str(csv_path),
        "status": "PASS" if status else "FAIL",
    }
    print("QUALIFICATION_SUMMARY", json.dumps(summary, sort_keys=True))
    del model, optimizer, optimizer_center, center_criterion
    gc.collect()
    torch.cuda.empty_cache()
    if not status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
