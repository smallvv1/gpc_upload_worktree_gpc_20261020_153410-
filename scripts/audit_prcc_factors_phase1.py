#!/usr/local/bin/python3
"""Audit PRCC factor selection, runtime shapes, reverse losses, and parameter counts."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from model import build_model
from scripts.qualify_amp_initial_scale_phase1 import FACTOR_NAMES, make_dataset_config, set_seed


def run(indices) -> dict:
    local_cfg = make_dataset_config("prcc").clone()
    local_cfg.defrost()
    local_cfg.DATA.NOBIO_INDEX = list(indices)
    local_cfg.freeze()
    set_seed(local_cfg.SOLVER.SEED)
    dataset = build_dataset(local_cfg)
    train_transform, _ = build_img_transforms(local_cfg)
    loader = DataLoader(
        ImageDataset(dataset.train, transform=train_transform),
        batch_size=local_cfg.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
    )
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    with amp.autocast(enabled=True):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    reverse = score["clip_bio_reverse_score"]
    bio = score["clip_bio_score"]
    if bio.ndim == 2:
        bio = bio.unsqueeze(1)
    text_bio = batch["caption_feature"][:, : bio.shape[1]]
    text_nonbio = batch["caption_feature"][:, bio.shape[1] :]
    labels = torch.arange(batch["image"].shape[0], device=device)
    reverse_losses = []
    for local_index, factor_index in enumerate(indices):
        logits = F.normalize(reverse[:, local_index], dim=-1) @ F.normalize(
            text_nonbio[:, local_index].type_as(reverse), dim=-1,
        ).T
        value = F.cross_entropy(logits, labels)
        reverse_losses.append({
            "head_index": local_index,
            "factor_index": int(factor_index),
            "factor_name": FACTOR_NAMES[int(factor_index)],
            "value": float(value),
            "dtype": str(value.dtype),
        })
    reverse_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("head_clip_bioReverse.") and parameter.requires_grad
    )
    result = {
        "NOBIO_INDEX": list(indices),
        "factor_names": [FACTOR_NAMES[int(index)] for index in indices],
        "caption_feature_shape": list(batch["caption_feature"].shape),
        "text_bio_shape": list(text_bio.shape),
        "text_nonbio_shape": list(text_nonbio.shape),
        "image_feature_reverse_shape": list(reverse.shape),
        "reverse_head_count": len(model.head_clip_bioReverse),
        "reverse_losses": reverse_losses,
        "total_trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "reverse_module_trainable_params": reverse_parameters,
    }
    print("PRCC_FACTOR_RUNTIME", json.dumps(result, sort_keys=True))
    del model, dataset, loader, batch, score, feat
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    released = run([1])
    paper = run([1, 2])
    output = REPO_ROOT / "diagnostics/prcc_factor_audit.json"
    output.write_text(json.dumps({"released_code": released, "paper_setting": paper}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
