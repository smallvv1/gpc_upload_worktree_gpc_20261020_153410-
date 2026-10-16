#!/usr/local/bin/python3
"""Frozen LTCC checkpoint feature extraction for PHASE 4R-C.

This is inference only.  It loads an existing checkpoint with
``pretrained=False`` and never constructs an optimizer or calls backward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from model.build import factory  # noqa: E402


DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
OUT_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features"
BIO_CHECKPOINT = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/eva02_l_bio_best.pth"
FULL_CHECKPOINT = ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/eva02_l_bio_best.pth"
OFFICIAL_CHECKPOINT = DATA_ROOT / "DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_config(reverse: bool):
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"] + (["clipBioReverse"] if reverse else [])
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = str(DATA_ROOT)
    cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def load_state(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.replace("module.", ""): value for key, value in state.items()}


def build_model_for_checkpoint(cfg, path: Path, official: bool):
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    adapter_count = 0
    if official:
        # The packaged official artifact has reverse-head bias tensors, while
        # the current released constructor is biasless.  Use an evaluation
        # adapter so all official reverse tensors can be audited without
        # altering the shared EVA/Bio representation.
        for index, old in enumerate(model.head_clip_bioReverse):
            if old.head_fc.bias is None:
                replacement = nn.Linear(old.head_fc.in_features, old.head_fc.out_features, bias=True)
                replacement.weight.data.copy_(old.head_fc.weight.data)
                model.head_clip_bioReverse[index].head_fc = replacement
                adapter_count += 1
    state = load_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to("cuda:0")
    model.eval()
    return model, {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_key_count": len(state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "official_reverse_bias_adapters": adapter_count,
        "strict_common_key_load": not missing,
    }


def extract(model, dataset, transform, name: str):
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform),
        batch_size=128,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    global_features = []
    bio_projected = []
    id_logits = []
    reverse_features = {"hair": [], "clothing": []}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].cuda(non_blocking=False)
            cameras = batch["camid"].cuda(non_blocking=False)
            with torch.autocast(device_type="cuda", enabled=True):
                feature = model(images, cam_label=cameras)
                projected = model.head_clip_bio(feature)
                logits = model.head(feature)
                if hasattr(model, "head_clip_bioReverse"):
                    for index, factor in enumerate(["hair", "clothing"]):
                        reverse_features[factor].append(model.head_clip_bioReverse[index].head_fc(feature))
            global_features.append(feature.float().cpu().numpy())
            bio_projected.append(projected.float().cpu().numpy())
            id_logits.append(logits.float().cpu().numpy())
            if hasattr(model, "head_clip_bioReverse"):
                for factor in reverse_features:
                    reverse_features[factor][-1] = reverse_features[factor][-1].float().cpu().numpy()
            if batch_index % 20 == 0:
                print("FEATURE_BATCH", name, batch_index, len(batch["image"]), flush=True)
    result = {
        "global_feature": np.concatenate(global_features, axis=0),
        "bio_projected": np.concatenate(bio_projected, axis=0),
        "id_logits": np.concatenate(id_logits, axis=0),
    }
    if hasattr(model, "head_clip_bioReverse"):
        result["reverse_hair_projected"] = np.concatenate(reverse_features["hair"], axis=0)
        result["reverse_clothing_projected"] = np.concatenate(reverse_features["clothing"], axis=0)
    else:
        result["reverse_hair_projected"] = np.zeros((len(dataset.train), 0), dtype=np.float32)
        result["reverse_clothing_projected"] = np.zeros((len(dataset.train), 0), dtype=np.float32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["bio", "full", "official", "all"], default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 4R-C frozen checkpoint extraction requires PPU/CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    specs = {
        "bio_control": (BIO_CHECKPOINT, False, False),
        "original_full": (FULL_CHECKPOINT, True, False),
        "official_full": (OFFICIAL_CHECKPOINT, True, True),
    }
    selected = list(specs) if args.which == "all" else {
        "bio": ["bio_control"],
        "full": ["original_full"],
        "official": ["official_full"],
    }[args.which]
    manifest = {
        "phase": "4R-C5/C8",
        "inference_only": True,
        "device": torch.cuda.get_device_name(0),
        "seed": 1234,
        "dataset": "LTCC train, sorted by released loader",
        "image_count": 9576,
        "transform": "released test transform: resize 224, tensor, ImageNet normalization",
        "representation_semantics": {
            "global_feature": "model eval output feat_bio after EVA head_bio and pooling; source used by ID head",
            "bio_projected": "head_clip_bio(feat_bio), 512 dimensions",
            "reverse_source": "same feat_bio tensor; released reverse classifier consumes it before its 512-d head",
            "reverse_projected": "reverse head linear output when reverse head weights are available",
        },
        "runs": {},
    }
    for name in selected:
        path, reverse, official = specs[name]
        if not path.exists():
            raise FileNotFoundError(path)
        cfg = make_config(reverse)
        dataset = build_dataset(cfg)
        _, transform = build_img_transforms(cfg)
        model, load_info = build_model_for_checkpoint(cfg, path, official)
        features = extract(model, dataset, transform, name)
        out_path = OUT_ROOT / f"{name}_feature_bank.npz"
        np.savez_compressed(
            out_path,
            global_feature=features["global_feature"].astype(np.float16),
            bio_projected=features["bio_projected"].astype(np.float16),
            id_logits=features["id_logits"].astype(np.float16),
            reverse_source=features["global_feature"].astype(np.float16),
            reverse_hair_projected=features["reverse_hair_projected"].astype(np.float16),
            reverse_clothing_projected=features["reverse_clothing_projected"].astype(np.float16),
            stems=np.asarray([Path(item["image_path"]).stem for item in dataset.train]),
            pid=np.asarray([item["pid"] for item in dataset.train], dtype=np.int16),
            camera=np.asarray([item["camid"] for item in dataset.train], dtype=np.int16),
            clothes_id=np.asarray([item["clothes_id"] for item in dataset.train], dtype=np.int16),
        )
        manifest["runs"][name] = {
            **load_info,
            "feature_bank": str(out_path),
            "feature_bank_sha256": sha256_file(out_path),
            "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        }
        del model, dataset, features
        gc.collect()
        torch.cuda.empty_cache()
    (OUT_ROOT / "checkpoint_feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
