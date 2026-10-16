#!/usr/local/bin/python3
"""LTCC-only DIFFER AMP overflow localization; never performs an optimizer step."""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from loss import make_loss
from loss.triplet_loss import TripletLoss
from model import build_model
from solver import make_optimizer


class NullWriter:
    def add_scalar(self, name, value, step):
        return None


GROUPS = {
    "EVA backbone": ("patch_embed.", "blocks."),
    "classifier": ("head.",),
    "bio transformer": ("head_bio.",),
    "bio projector": ("head_clip_bio.",),
    "nonbio/reverse branch": ("head_clip_bioReverse.",),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_config():
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(REPO_ROOT / "configs/ltcc/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.DATA.ROOT = "/home/data"
    local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.freeze()
    return local_cfg


def grad_tensor_stats(gradient: torch.Tensor) -> dict:
    diagnostic = gradient.detach().double()
    nan_count = int(torch.isnan(diagnostic).sum().item())
    inf_count = int(torch.isinf(diagnostic).sum().item())
    finite = diagnostic[torch.isfinite(diagnostic)]
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_abs_finite": float(finite.abs().max().item()) if finite.numel() else None,
    }


def group_stats(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict:
    result = {
        "grad_exists": False,
        "tensor_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "max_abs_finite": None,
    }
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) and parameter.grad is not None:
            stats = grad_tensor_stats(parameter.grad)
            result["grad_exists"] = True
            result["tensor_count"] += 1
            result["nan_count"] += stats["nan_count"]
            result["inf_count"] += stats["inf_count"]
            if stats["max_abs_finite"] is not None:
                old = result["max_abs_finite"]
                result["max_abs_finite"] = stats["max_abs_finite"] if old is None else max(old, stats["max_abs_finite"])
    return result


def nonfinite_parameters(model: torch.nn.Module) -> list[dict]:
    result = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        stats = grad_tensor_stats(parameter.grad)
        if stats["nan_count"] or stats["inf_count"]:
            result.append({
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.grad.dtype),
                **stats,
            })
    return result


def component_losses(score, feat, target, captions) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    labels = torch.arange(target.shape[0], device=target.device)
    bio = score["clip_bio_score"]
    if bio.ndim == 2:
        bio = bio.unsqueeze(1)
    reverse = score["clip_bio_reverse_score"]
    bio = F.normalize(bio, dim=-1)
    reverse = F.normalize(reverse, dim=-1)
    captions = captions.type_as(bio)
    text_bio = F.normalize(captions[:, : bio.shape[1]], dim=-1)
    text_nonbio = F.normalize(captions[:, bio.shape[1] :], dim=-1)

    bio_logits = bio[:, 0] @ text_bio[:, 0].T
    reverse_hair_logits = reverse[:, 0].type_as(text_nonbio) @ text_nonbio[:, 0].T
    reverse_clothing_logits = reverse[:, 1].type_as(text_nonbio) @ text_nonbio[:, 1].T
    losses = {
        "ID": F.cross_entropy(score["cls_score"], target),
        "Triplet": TripletLoss()(feat[0], target)[0],
        "Bio": F.cross_entropy(bio_logits, labels),
        "Reverse Hair": F.cross_entropy(reverse_hair_logits, labels),
        "Reverse Clothing": F.cross_entropy(reverse_clothing_logits, labels),
    }
    dtypes = {
        "cls_logits": str(score["cls_score"].dtype),
        "global_feature": str(feat[0].dtype),
        "bio_logits": str(bio_logits.dtype),
        "reverse_hair_logits": str(reverse_hair_logits.dtype),
        "reverse_clothing_logits": str(reverse_clothing_logits.dtype),
    }
    return losses, dtypes


def main() -> None:
    local_cfg = make_config()
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
    batch = next(iter(loader))
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    official_loss_fn, center_criterion = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center_criterion)
    print("DIAGNOSTIC_BATCH", json.dumps({
        "images": list(batch["image"].shape),
        "pids": batch["pid"].detach().cpu().tolist(),
        "captions": list(batch["caption_feature"].shape),
    }))
    print("NO_OPTIMIZER_STEP_WILL_BE_EXECUTED: True")

    combinations = {
        "ID": ("ID",),
        "Triplet": ("Triplet",),
        "Bio": ("Bio",),
        "Reverse Hair": ("Reverse Hair",),
        "Reverse Clothing": ("Reverse Clothing",),
        "ID + Triplet": ("ID", "Triplet"),
        "ID + Bio": ("ID", "Bio"),
        "Reverse Hair + Reverse Clothing": ("Reverse Hair", "Reverse Clothing"),
    }

    def run_case(name: str, scale: float, mode: str = "official", selected: tuple[str, ...] | None = None) -> dict:
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        set_seed(local_cfg.SOLVER.SEED + 1000)
        if mode == "inside_autocast":
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
                if selected is None:
                    loss = official_loss_fn(
                        score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
                        clothes_ids=batch["clothes_id"], train_writer=NullWriter(), step=0,
                    )
                    parts, tensor_dtypes = component_losses(score, feat, batch["pid"], batch["caption_feature"])
                else:
                    parts, tensor_dtypes = component_losses(score, feat, batch["pid"], batch["caption_feature"])
                    loss = sum(parts[item] for item in selected)
        else:
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
            parts, tensor_dtypes = component_losses(score, feat, batch["pid"], batch["caption_feature"])
            if mode == "fp32_ce":
                loss = F.cross_entropy(score["cls_score"].float(), batch["pid"])
            elif selected is None:
                loss = official_loss_fn(
                    score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
                    clothes_ids=batch["clothes_id"], train_writer=NullWriter(), step=0,
                )
            else:
                loss = sum(parts[item] for item in selected)

        scaler = amp.GradScaler(init_scale=scale)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        groups = {group: group_stats(model, prefixes) for group, prefixes in GROUPS.items()}
        named_nonfinite = nonfinite_parameters(model)
        finite = all(item["nan_count"] == 0 and item["inf_count"] == 0 for item in groups.values())
        result = {
            "case": name,
            "mode": mode,
            "scale": scale,
            "loss_value": float(loss.detach()),
            "loss_dtype": str(loss.dtype),
            "loss_finite": bool(torch.isfinite(loss)),
            "component_values": {key: float(value.detach()) for key, value in parts.items()},
            "component_dtypes": {key: str(value.dtype) for key, value in parts.items()},
            "tensor_dtypes": tensor_dtypes,
            "all_existing_gradients_finite": finite,
            "groups": groups,
            "nonfinite_parameter_count": len(named_nonfinite),
            "first_20_nonfinite_parameters": named_nonfinite[:20],
        }
        print("CASE_RESULT", json.dumps(result, sort_keys=True))
        del score, feat, parts, loss, scaler
        gc.collect()
        return result

    fixed_results = {}
    for name, selected in combinations.items():
        fixed_results[name] = run_case(name, 32768.0, selected=selected)
    fixed_results["Full"] = run_case("Full", 32768.0, selected=None)

    sweep_cases = {
        "ID": ("ID",),
        "Triplet": ("Triplet",),
        "Bio": ("Bio",),
        "Reverse Hair": ("Reverse Hair",),
        "Reverse Clothing": ("Reverse Clothing",),
        "Full": None,
    }
    sweep = {}
    for name, selected in sweep_cases.items():
        sweep[name] = {}
        for scale in (32768.0, 16384.0, 8192.0, 4096.0):
            result = run_case(f"SWEEP {name}", scale, selected=selected)
            sweep[name][str(int(scale))] = "FINITE" if result["all_existing_gradients_finite"] else "INF/NAN"
            if result["all_existing_gradients_finite"]:
                break
    print("SCALE_SWEEP", json.dumps(sweep, sort_keys=True))

    official_65536 = run_case("DIAG A official full", 65536.0, mode="official", selected=None)
    inside_65536 = run_case("DIAG B full loss inside autocast", 65536.0, mode="inside_autocast", selected=None)
    fp32_ce_32768 = run_case("DIAG C FP32 CE", 32768.0, mode="fp32_ce", selected=("ID",))
    fp32_ce_65536 = run_case("DIAG C FP32 CE", 65536.0, mode="fp32_ce", selected=("ID",))
    print("CAUSAL_SUMMARY", json.dumps({
        "official_65536": official_65536["all_existing_gradients_finite"],
        "inside_autocast_65536": inside_65536["all_existing_gradients_finite"],
        "fp32_ce_32768": fp32_ce_32768["all_existing_gradients_finite"],
        "fp32_ce_65536": fp32_ce_65536["all_existing_gradients_finite"],
    }, sort_keys=True))

    del model, optimizer, optimizer_center, center_criterion
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
