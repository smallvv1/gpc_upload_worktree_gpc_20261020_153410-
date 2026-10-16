#!/usr/local/bin/python3
"""Run full LTCC and PRCC feature extraction and official metric dry-runs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from model import build_model
from scripts.qualify_amp_initial_scale_phase1 import make_dataset_config, set_seed
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC


def extract(model, loader, evaluator, device):
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            cameras = batch["camid"].to(device)
            features = model(images, cam_label=cameras)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError(f"non-finite feature at evaluation batch {iteration}")
            evaluator.update((
                features,
                batch["pid"],
                batch["camid"],
                batch["clothes_id"],
                batch["image_path"],
            ))
            if iteration == 1 or iteration % 20 == 0:
                print("EVAL_EXTRACTION_PROGRESS", iteration, len(loader), tuple(features.shape))
    cmc, mean_ap, distance, _, _, query_features, gallery_features, _ = evaluator.compute()
    return {
        "query_feature_shape": list(query_features.shape),
        "gallery_feature_shape": list(gallery_features.shape),
        "distance_matrix_shape": list(distance.shape),
        "distance_nan_count": int(np.isnan(distance).sum()),
        "distance_inf_count": int(np.isinf(distance).sum()),
        "rank1": float(cmc[0]),
        "mAP": float(mean_ap),
        "reranking": evaluator.reranking,
        "feature_normalization": bool(evaluator.feat_norm),
    }


def make_eval_loader(items, transform, batch_size, num_workers):
    datasets = [ImageDataset(part, transform=transform) for part in items]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )


def audit_ltcc(device, checkpoint=None):
    local_cfg = make_dataset_config("ltcc")
    set_seed(local_cfg.SOLVER.SEED)
    dataset = build_dataset(local_cfg)
    _, transform = build_img_transforms(local_cfg)
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print("LTCC_EVAL_LOADED_CHECKPOINT:", checkpoint)
    loader = make_eval_loader(
        [dataset.query, dataset.gallery], transform,
        local_cfg.DATA.TEST_BATCH, local_cfg.DATA.NUM_WORKERS,
    )
    changed = extract(
        model, loader,
        R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=local_cfg.TEST.FEAT_NORM),
        device,
    )
    general = extract(
        model, loader,
        R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=local_cfg.TEST.FEAT_NORM),
        device,
    )
    gallery_pids = np.asarray([item["pid"] for item in dataset.gallery])
    gallery_cams = np.asarray([item["camid"] for item in dataset.gallery])
    gallery_clothes = np.asarray([item["clothes_id"] for item in dataset.gallery])
    samples = []
    for query in dataset.query[:5]:
        pid = query["pid"]
        camera = query["camid"]
        clothes = query["clothes_id"]
        cross_camera_positive = (gallery_pids == pid) & (gallery_cams != camera)
        same_clothes_positive = cross_camera_positive & (gallery_clothes == clothes)
        changed_clothes_positive = cross_camera_positive & (gallery_clothes != clothes)
        samples.append({
            "path": query["image_path"],
            "pid": int(pid),
            "query_clothes_id": int(clothes),
            "legal_changed_clothes_gallery_positives": int(changed_clothes_positive.sum()),
            "filtered_same_clothes_positives": int(same_clothes_positive.sum()),
            "general_cross_camera_positives": int(cross_camera_positive.sum()),
        })
    result = {
        "query_count": len(dataset.query),
        "gallery_count": len(dataset.gallery),
        "changed_clothes": changed,
        "general": general,
        "sample_filter_counts": samples,
        "changed_filter": "remove same-PID/same-camera and same-PID/same-clothes; keep different-clothes positives",
    }
    print("LTCC_EVALUATION_RESULT", json.dumps(result, sort_keys=True))
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def audit_prcc(device):
    local_cfg = make_dataset_config("prcc")
    set_seed(local_cfg.SOLVER.SEED)
    dataset = build_dataset(local_cfg)
    _, transform = build_img_transforms(local_cfg)
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    changed_loader = make_eval_loader(
        [dataset.query_diff, dataset.gallery], transform,
        local_cfg.DATA.TEST_BATCH, local_cfg.DATA.NUM_WORKERS,
    )
    same_loader = make_eval_loader(
        [dataset.query_same, dataset.gallery], transform,
        local_cfg.DATA.TEST_BATCH, local_cfg.DATA.NUM_WORKERS,
    )
    changed = extract(
        model, changed_loader,
        R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=local_cfg.TEST.FEAT_NORM),
        device,
    )
    same = extract(
        model, same_loader,
        R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=local_cfg.TEST.FEAT_NORM),
        device,
    )
    result = {
        "gallery": {"count": len(dataset.gallery), "cameras": sorted(set(item["camid"] for item in dataset.gallery))},
        "same_query": {"count": len(dataset.query_same), "cameras": sorted(set(item["camid"] for item in dataset.query_same))},
        "changed_query": {"count": len(dataset.query_diff), "cameras": sorted(set(item["camid"] for item in dataset.query_diff))},
        "same_clothes": same,
        "changed_clothes": changed,
        "gallery_idx_present": hasattr(dataset, "gallery_idx"),
        "gallery_idx_used_by_runtime_evaluator": False,
    }
    print("PRCC_EVALUATION_RESULT", json.dumps(result, sort_keys=True))
    del model, changed_loader, same_loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltcc-checkpoint")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output = {
        "ltcc": audit_ltcc(device, args.ltcc_checkpoint),
        "prcc": audit_prcc(device),
    }
    (REPO_ROOT / "diagnostics/evaluation_preflight.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
