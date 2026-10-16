#!/usr/local/bin/python3
"""Compare fixed LTCC query/gallery representations from three checkpoints."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg as GLOBAL_CFG
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from model import build_model
from model import eva_meta_bio


OFFICIAL = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
BASELINE = Path("/home/DIFFER/outputs/reproduction_phase2/LTCC/baseline/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/eva02_l_bio_best.pth")
FULL = Path("/home/DIFFER/outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/eva02_l_bio_best.pth")
CONFIG = REPO_ROOT / "diagnostics/reconstructed_ltcc_eval_config.yml"


class EvaluationOnlyReverseAdapter(nn.Module):
    def __init__(self, in_num, class_num, alpha=1.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)


def cfg_for(losses):
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(CONFIG))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = list(losses)
    cfg.freeze()
    return cfg


def load_state(path):
    state = torch.load(path, map_location="cpu")
    return {key.replace("module.", ""): value for key, value in state.items()}


def make_model(local_cfg, dataset, path, official=False):
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    adapters = 0
    if official:
        for index, old in enumerate(model.head_clip_bioReverse):
            if old.head_fc.bias is None:
                fc = old.head_fc
                model.head_clip_bioReverse[index] = EvaluationOnlyReverseAdapter(
                    fc.in_features, fc.out_features, old.alpha
                )
                adapters += 1
    state = load_state(path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load failed for {path}: {missing}, {unexpected}")
    return model.cuda(), adapters


def extract(model, items, transform, device):
    loader = DataLoader(ImageDataset(items, transform=transform), batch_size=64,
                        num_workers=1, pin_memory=False, shuffle=False)
    feats = []
    with torch.no_grad():
        for batch in loader:
            feats.append(model(batch["image"].to(device), cam_label=batch["camid"].to(device)).float().cpu())
    return torch.cat(feats, dim=0)


def stats(name, q_items, g_items, qf, gf):
    qn = torch.linalg.vector_norm(qf, dim=1).numpy()
    gn = torch.linalg.vector_norm(gf, dim=1).numpy()
    q = F.normalize(qf, dim=1)
    g = F.normalize(gf, dim=1)
    cosine = q @ g.T
    dist = torch.cdist(q, g).numpy()
    qp = np.asarray([x["pid"] for x in q_items])
    gp = np.asarray([x["pid"] for x in g_items])
    qc = np.asarray([x["clothes_id"] for x in q_items])
    gc = np.asarray([x["clothes_id"] for x in g_items])
    positive = []
    positive_cosine = []
    negative = []
    negative_cosine = []
    for i in range(len(q_items)):
        changed = (gp == qp[i]) & (gc != qc[i])
        different = gp != qp[i]
        positive.extend(dist[i, changed].tolist())
        positive_cosine.extend(cosine[i, changed].numpy().tolist())
        negative.extend(dist[i, different].tolist())
        negative_cosine.extend(cosine[i, different].numpy().tolist())
    if not positive or not negative:
        raise RuntimeError("fixed subset did not contain both positive changed-clothes and negative pairs")
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    all_cos = cosine.numpy().ravel()
    return {
        "checkpoint": name,
        "query_count": len(q_items),
        "gallery_count": len(g_items),
        "raw_query_norm_mean": float(qn.mean()),
        "raw_query_norm_std": float(qn.std()),
        "raw_gallery_norm_mean": float(gn.mean()),
        "raw_gallery_norm_std": float(gn.std()),
        "pairwise_cosine_mean": float(all_cos.mean()),
        "pairwise_cosine_std": float(all_cos.std()),
        "pairwise_cosine_quantiles": [float(x) for x in np.quantile(all_cos, [0.01, 0.5, 0.99])],
        "same_id_changed_clothes_distance_mean": float(positive.mean()),
        "same_id_changed_clothes_distance_std": float(positive.std()),
        "same_id_changed_clothes_cosine_mean": float(np.mean(positive_cosine)),
        "different_id_distance_mean": float(negative.mean()),
        "different_id_distance_std": float(negative.std()),
        "different_id_cosine_mean": float(np.mean(negative_cosine)),
        "margin_negative_minus_positive": float(negative.mean() - positive.mean()),
        "pair_counts": {"positive_changed_clothes": int(len(positive)), "negative_id": int(len(negative))},
    }


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    local_cfg = cfg_for(["ce", "triplet", "clipBio", "clipBioReverse"])
    dataset = build_dataset(local_cfg)
    _, transform = build_img_transforms(local_cfg)
    q_items = dataset.query[:100]
    g_items = dataset.gallery[:500]

    runs = [
        ("our_baseline_best", cfg_for(["ce", "triplet"]), BASELINE, False),
        ("our_full_differ_best", local_cfg, FULL, False),
        ("official_differ", local_cfg, OFFICIAL, True),
    ]
    results = {}
    for name, model_cfg, path, official in runs:
        model, adapters = make_model(model_cfg, dataset, path, official=official)
        model.eval()
        qf = extract(model, q_items, transform, device)
        gf = extract(model, g_items, transform, device)
        results[name] = stats(name, q_items, g_items, qf, gf)
        results[name]["strict_load"] = {"missing_keys": [], "unexpected_keys": [], "eval_only_reverse_adapters": adapters}
        print("CHECKPOINT_FEATURE_STATS", json.dumps(results[name], sort_keys=True), flush=True)
        del model, qf, gf
        gc.collect()
        torch.cuda.empty_cache()
    output = {
        "subset": {"query": 100, "gallery": 500, "selection": "first sorted query/gallery items", "seed": 1234},
        "evaluator_context": {"feature_normalization_for_pair_stats": True, "distance": "euclidean", "reranking": False},
        "results": results,
    }
    out = REPO_ROOT / "diagnostics/ltcc_checkpoint_feature_comparison.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("FEATURE_COMPARISON_WRITTEN", out, flush=True)


if __name__ == "__main__":
    main()
