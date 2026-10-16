#!/usr/local/bin/python3
"""Independent LTCC evaluation of the official weight artifact.

This script deliberately keeps the repository model and evaluator unchanged.
The released HEAD's reverse linear layers have ``bias=False`` while the
official LTCC artifact contains two zero/learned reverse bias keys.  A local,
evaluation-only topology adapter exposes the same key names with bias=True so
the artifact can be loaded with strict=True.  Reverse heads are not used by
the model's eval() forward path; retrieval features remain the 1024-d global
feature.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg as GLOBAL_CFG
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from model import build_model
from model import eva_meta_bio
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class EvaluationOnlyReverseAdapter(nn.Module):
    """State-dict-compatible reverse head for official LTCC evaluation only."""

    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)

    def forward(self, x):
        return self.head_fc(self.grl(x))


def install_eval_only_reverse_adapter(model: nn.Module) -> int:
    """Add only the two artifact bias slots; do not edit repository sources."""
    reverse_heads = getattr(model, "head_clip_bioReverse", None)
    if reverse_heads is None:
        return 0
    replaced = 0
    for index, old_head in enumerate(reverse_heads):
        old_fc = old_head.head_fc
        if old_fc.bias is not None:
            continue
        replacement = EvaluationOnlyReverseAdapter(
            old_fc.in_features, old_fc.out_features, old_head.alpha
        )
        reverse_heads[index] = replacement
        replaced += 1
    return replaced


def load_config(config_path: str):
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(config_path)
    local_cfg.defrost()
    # Use the released-YAML value if the file omitted a field derived by
    # train.py.  No training config is mutated on disk.
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.freeze()
    return local_cfg


def make_loader(dataset, transform, batch_size, num_workers):
    return DataLoader(
        ConcatDataset([
            ImageDataset(dataset.query, transform=transform),
            ImageDataset(dataset.gallery, transform=transform),
        ]),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )


def extract(model, loader, evaluator, device):
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            camids = batch["camid"].to(device)
            features = model(images, cam_label=camids)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError(f"non-finite feature at batch {index}")
            evaluator.update((features, batch["pid"], batch["camid"],
                              batch["clothes_id"], batch["image_path"]))
            if index == 1 or index % 20 == 0:
                print("EVAL_PROGRESS", index, len(loader), tuple(features.shape), flush=True)
    cmc, mean_ap, distance, _, _, qf, gf, paths = evaluator.compute()
    result = {
        "query_feature_shape": list(qf.shape),
        "gallery_feature_shape": list(gf.shape),
        "distance_matrix_shape": list(distance.shape),
        "distance_nan_count": int(np.isnan(distance).sum()),
        "distance_inf_count": int(np.isinf(distance).sum()),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mean_ap),
        "feat_norm": bool(evaluator.feat_norm),
        "reranking": bool(evaluator.reranking),
        "path_count": len(paths),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "diagnostics/reconstructed_ltcc_eval_config.yml"))
    parser.add_argument("--weight", default="/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostics/official_ltcc_independent.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    local_cfg = load_config(args.config)
    set_seed(local_cfg.SOLVER.SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dataset = build_dataset(local_cfg)
    _, test_transform = build_img_transforms(local_cfg)
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    adapter_count = install_eval_only_reverse_adapter(model)
    state = torch.load(args.weight, map_location="cpu")
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load unexpectedly returned missing={missing}, unexpected={unexpected}")
    print("STRICT_LOAD_PASS", json.dumps({
        "missing_keys": [], "unexpected_keys": [], "eval_only_reverse_adapters": adapter_count,
    }, sort_keys=True), flush=True)
    model.to(device)
    loader = make_loader(dataset, test_transform, local_cfg.DATA.TEST_BATCH, local_cfg.DATA.NUM_WORKERS)
    changed = extract(model, loader,
                      R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50,
                                       feat_norm=local_cfg.TEST.FEAT_NORM), device)
    general = extract(model, loader,
                      R1_mAP_eval(dataset.num_query_imgs, max_rank=50,
                                  feat_norm=local_cfg.TEST.FEAT_NORM), device)
    result = {
        "config": str(Path(args.config).resolve()),
        "weight": str(Path(args.weight).resolve()),
        "dataset": "ltcc",
        "counts": {"train": len(dataset.train), "query": len(dataset.query), "gallery": len(dataset.gallery),
                    "num_train_pids": dataset.num_train_pids, "num_camera": dataset.num_camera},
        "strict_load": {"missing_keys": [], "unexpected_keys": [], "eval_only_reverse_adapters": adapter_count},
        "changed_clothes": changed,
        "general": general,
        "evaluator": {"feature_normalization": True, "distance": "euclidean", "reranking": False},
        "note": "reverse bias adapter exists only in this evaluation process; model eval forward returns feat_bio and does not invoke reverse heads",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("INDEPENDENT_EVALUATION_RESULT", json.dumps(result, sort_keys=True), flush=True)
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
