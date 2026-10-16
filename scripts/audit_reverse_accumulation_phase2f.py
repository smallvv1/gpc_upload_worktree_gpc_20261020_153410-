#!/usr/local/bin/python3
"""Confirm the released reverse-loss accumulation and logging semantics.

This is an audit-only script.  It builds one real LTCC batch and calls the
current loss factory without an optimizer step or any source modification.
"""

from __future__ import annotations

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
from model import build_model


class ScalarCapture:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, name, value, step=None, *args, **kwargs) -> None:
        self.values.setdefault(str(name), []).append(float(value))


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
    local_cfg.DATA.ROOT = "/home/data"
    local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.MODEL.DIST_TRAIN = False
    local_cfg.freeze()
    return local_cfg


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
    batch = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    with amp.autocast(enabled=True):
        score, feat = model(batch["image"], cam_label=batch["camid"])

    captions = batch["caption_feature"]
    image_bio = score["clip_bio_score"]
    if image_bio.ndim == 2:
        image_bio = image_bio.unsqueeze(1)
    image_reverse = score["clip_bio_reverse_score"]
    image_bio = F.normalize(image_bio, dim=-1)
    captions = captions.type(image_bio.type())
    text_bio = F.normalize(captions[:, : image_bio.shape[1]], dim=-1)
    text_nonbio = F.normalize(captions[:, image_bio.shape[1] :], dim=-1)

    factor_losses = []
    for index in range(image_reverse.shape[1]):
        image_i = F.normalize(image_reverse[:, index], dim=-1).type(text_nonbio.type())
        logits = image_i @ text_nonbio[:, index].T
        labels = torch.arange(image_i.shape[0], device=image_i.device)
        factor_losses.append(F.cross_entropy(logits, labels))

    writer = ScalarCapture()
    reverse_loss_fn, _ = make_loss(local_cfg, dataset.num_train_pids, lossTypes=["clipBioReverse"])
    reverse_from_factory = reverse_loss_fn(
        score,
        feat,
        batch["pid"],
        batch["camid"],
        batch["caption_feature"],
        clothes_ids=batch["clothes_id"],
        train_writer=writer,
        step=1,
    )

    hair, clothing = (float(value.detach()) for value in factor_losses[:2])
    expected_paper = hair + clothing
    claimed_released_inside_loop = 2.0 * hair + clothing
    observed = float(reverse_from_factory.detach())
    logging = {
        key: values[-1]
        for key, values in writer.values.items()
        if key.startswith("loss/clip_bio_reverse_")
    }
    result = {
        "batch_shape": list(batch["image"].shape),
        "image_reverse_shape": list(image_reverse.shape),
        "factor_loss_names": ["Hair", "Clothing"],
        "factor_losses": {"Hair": hair, "Clothing": clothing},
        "expected_paper_reverse": expected_paper,
        "claimed_inside_loop_reverse": claimed_released_inside_loop,
        "observed_factory_reverse": observed,
        "difference_observed_vs_paper": observed - expected_paper,
        "difference_observed_vs_claimed_inside_loop": observed - claimed_released_inside_loop,
        "logging": logging,
        "source_semantics": "loss += loss_clip_reverse is outside the factor loop in current source",
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
