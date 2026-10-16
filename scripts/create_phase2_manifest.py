#!/usr/local/bin/python3
"""Create a frozen manifest for one DIFFER Phase 2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg as _cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
EVA_REPO = "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
EVA_REVISION = "bf4190eb65dd5204ffb03e980108beb1200e0873"
EVA_PATH = Path(
    "/root/.cache/huggingface/hub/models--timm--"
    "eva02_large_patch14_clip_224.merged2b_s4b_b131k/snapshots/"
    f"{EVA_REVISION}/open_clip_pytorch_model.bin"
)
EVA_SHA256 = "ce87e437a7ce3a29eb709ae302c5fdd9c44a73bfd8f191ec655eec53f58b8e67"
FACTOR_NAMES = {
    0: "Biometrics",
    1: "Hairstyle",
    2: "Clothing",
    3: "Pose",
    4: "Interaction",
    5: "Environment",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)


def make_config(dataset: str, protocol: str, run_dir: Path, output_name: str):
    config_path = REPO_ROOT / f"configs/{dataset}/eva02_l_bio.yml"
    local_cfg = _cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    if dataset == "ltcc":
        local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet"] if protocol == "baseline" else [
            "ce", "triplet", "clipBio", "clipBioReverse"
        ]
        local_cfg.DATA.NOBIO_INDEX = [1, 2]
        local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
        local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
        local_cfg.MODEL.CLIP_DIM = 512
    else:
        raise ValueError(dataset)
    local_cfg.DATA.ROOT = "/home/data"
    local_cfg.OUTPUT_DIR = str(run_dir)
    local_cfg.OUTPUT_NAME = output_name
    return local_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ltcc"], required=True)
    parser.add_argument("--protocol", choices=["baseline", "differ"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    local_cfg = make_config(args.dataset, args.protocol, run_dir, args.output_name)
    text_root = Path(local_cfg.DATA.CAPTION_DIR)
    text_files = sorted(str(path) for path in text_root.rglob("*") if path.is_file())
    text_shapes = {
        "/home/data/DIFFER_TextCaption/LTCC_ReID/EVA02-CLIP-B-16/train.npz": {
            "metadata_shape": [9576],
            "data_shape": [57456, 512],
        }
    }
    optimizer_lrs = [float(local_cfg.SOLVER.BASE_LR)]
    optimizer_lrs.append(float(local_cfg.SOLVER.BASE_LR * local_cfg.SOLVER.BIAS_LR_FACTOR))
    warmup_lr = float(local_cfg.SOLVER.WARMUP_LR)
    first_runtime_lr = warmup_lr + (optimizer_lrs[0] - warmup_lr) / float(local_cfg.SOLVER.WARMUP_EPOCHS)
    first_runtime_bias_lr = warmup_lr + (optimizer_lrs[1] - warmup_lr) / float(local_cfg.SOLVER.WARMUP_EPOCHS)
    try:
        torch_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        torch_devices = [f"unavailable: {exc!r}"]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "protocol": args.protocol,
        "command": args.command,
        "config_snapshot": local_cfg.dump(),
        "git_commit": command_output("git", "rev-parse", "HEAD").strip(),
        "git_status": command_output("git", "status", "--short"),
        "git_diff": command_output("git", "diff"),
        "seed": int(local_cfg.SOLVER.SEED),
        "ppu": {
            "torch_distribution": __import__("importlib.metadata", fromlist=["version"]).version("torch"),
            "torch_runtime_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
            "devices": torch_devices,
        },
        "eva_pretrained": {
            "repository": EVA_REPO,
            "revision": EVA_REVISION,
            "path": str(EVA_PATH),
            "bytes": EVA_PATH.stat().st_size,
            "sha256": sha256(EVA_PATH),
        },
        "text_resources": {
            "caption_dir": str(text_root),
            "files": [
                {"path": path, "sha256": sha256(Path(path)), "bytes": Path(path).stat().st_size}
                for path in text_files
            ],
            "shapes": text_shapes,
        },
        "data": {
            "root": str(local_cfg.DATA.ROOT),
            "caption_dir": str(local_cfg.DATA.CAPTION_DIR),
            "image_size": [int(local_cfg.DATA.IMG_HEIGHT), int(local_cfg.DATA.IMG_WIDTH)],
            "batch_size": int(local_cfg.DATA.BATCH_SIZE),
            "num_instances": int(local_cfg.DATA.NUM_INSTANCES),
            "num_workers": int(local_cfg.DATA.NUM_WORKERS),
            "sampler": str(local_cfg.DATA.SAMPLER),
            "bio_index": list(local_cfg.DATA.BIO_INDEX) if not isinstance(local_cfg.DATA.BIO_INDEX, str) else [int(local_cfg.DATA.BIO_INDEX)],
            "nobio_index": list(local_cfg.DATA.NOBIO_INDEX),
            "factor_names": [FACTOR_NAMES[index] for index in local_cfg.DATA.NOBIO_INDEX],
        },
        "model": {
            "name": str(local_cfg.MODEL.NAME),
            "type": str(local_cfg.MODEL.TYPE),
            "last_layer": str(local_cfg.MODEL.LAST_LAYER),
            "clip_dim": int(local_cfg.MODEL.CLIP_DIM),
            "loss_type": list(local_cfg.MODEL.LOSS_TYPE),
            "pretrained": True,
            "grl": "enabled" if "clipBioReverse" in local_cfg.MODEL.LOSS_TYPE else "not active",
            "nbdetach": "enabled" if "clipBioReverse" in local_cfg.MODEL.LOSS_TYPE else "not active",
        },
        "optimizer_scheduler": {
            "optimizer": str(local_cfg.SOLVER.OPTIMIZER_NAME),
            "momentum": float(local_cfg.SOLVER.MOMENTUM),
            "weight_decay": float(local_cfg.SOLVER.WEIGHT_DECAY),
            "weight_decay_bias": float(local_cfg.SOLVER.WEIGHT_DECAY_BIAS),
            "base_lr": optimizer_lrs[0],
            "bias_lr_factor": float(local_cfg.SOLVER.BIAS_LR_FACTOR),
            "normal_lr_epoch1": first_runtime_lr,
            "bias_lr_epoch1": first_runtime_bias_lr,
            "warmup_method": str(local_cfg.SOLVER.WARMUP_METHOD),
            "warmup_lr": warmup_lr,
            "warmup_epochs": int(local_cfg.SOLVER.WARMUP_EPOCHS),
            "scheduler": "epoch-based cosine",
            "epochs": int(local_cfg.SOLVER.MAX_EPOCHS),
        },
        "amp": {
            "enabled": True,
            "autocast_dtype": "FP16",
            "loss_location": "outside autocast",
            "grad_scaler_initial_scale": 1024,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        "evaluation": {
            "ltcc_primary": "clothes-changing",
            "ltcc_secondary": "general",
            "feature_normalization": True,
            "reranking": False,
            "metric": "Euclidean pairwise distance + CMC/mAP",
        },
        "checkpoint_policy": {
            "best": "official processor criterion: clothes-changing Rank-1",
            "final": "epoch 60 model state",
            "optimizer_saved": False,
            "scheduler_saved": False,
            "gradscaler_saved": False,
        },
        "patches": [
            "PATH COMPATIBILITY: LTCC dataset_dir=LTCC_ReID",
            "PATH COMPATIBILITY: PRCC dataset_dir=prcc",
            "BUG FIX: remove forced LAST_LAYER transformer override",
            "BUG FIX: non-biometric indexing image_feature_reverse[:, i]",
            "ENVIRONMENT COMPATIBILITY: TensorFlow removed for PPU TensorBoard",
            "ENVIRONMENT COMPATIBILITY: PPU AMP initial scale LTCC=1024",
            "TEST INFRASTRUCTURE: per-epoch metrics and final checkpoint instrumentation",
        ],
    }
    (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
