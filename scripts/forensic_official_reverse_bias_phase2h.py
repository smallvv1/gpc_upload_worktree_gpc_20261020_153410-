#!/usr/local/bin/python3
"""Read-only forensic analysis of historical reverse-head bias parameters.

This script never calls backward or an optimizer.  It loads the official LTCC
checkpoint with a process-local bias=True adapter, computes reverse logits on
500 deterministic LTCC training images, and transiently zeros/restores the
two bias tensors for an apples-to-apples functional comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
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


CONFIG = REPO_ROOT / "reports/FORENSIC_RECONSTRUCTED_LTCC_CONFIG_V2.yml"
CHECKPOINT = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics/official_reverse_bias_phase2h.json"


class EvaluationOnlyReverseAdapter(nn.Module):
    """Historical bias=True reverse head; used only in this process."""

    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)

    def forward(self, x):
        return self.head_fc(self.grl(x))


def normalize_state(state):
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def tensor_stats(tensor: torch.Tensor) -> dict:
    x = tensor.detach().double().cpu()
    finite = torch.isfinite(x)
    return {
        "shape": list(x.shape),
        "dtype": str(tensor.dtype),
        "numel": int(x.numel()),
        "nonzero_count": int(torch.count_nonzero(x)),
        "zero_count": int((x == 0).sum()),
        "nan_count": int(torch.isnan(x).sum()),
        "inf_count": int(torch.isinf(x).sum()),
        "mean": float(x.mean()),
        "std_population": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "max_abs": float(x.abs().max()),
        "l1_norm": float(x.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(x)),
        "finite": bool(finite.all()),
    }


def loss_accuracy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()
    return loss, acc


def reverse_outputs(model, feat_bio: torch.Tensor):
    return [head(feat_bio) for head in model.head_clip_bioReverse]


def compute_reverse_metrics(outputs, caption: torch.Tensor):
    text_nonbio = F.normalize(caption[:, 1:], dim=-1)
    losses, accs, logits = [], [], []
    for factor, output in enumerate(outputs):
        image = F.normalize(output, dim=-1).to(text_nonbio.dtype)
        factor_logits = image @ text_nonbio[:, factor].T
        loss, acc = loss_accuracy(factor_logits)
        losses.append(loss)
        accs.append(acc)
        logits.append(factor_logits)
    return losses, accs, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(CONFIG))
    local_cfg.defrost()
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.freeze()
    dataset = build_dataset(local_cfg)
    _, test_transform = build_img_transforms(local_cfg)
    # The functional test is deliberately deterministic: sorted training
    # images, no shuffle, and the evaluation transform.  It tests real
    # training-image/text pairs without running a sampler or optimizer.
    loader = DataLoader(
        ImageDataset(dataset.train, transform=test_transform),
        batch_size=local_cfg.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    adapters = 0
    for index, old_head in enumerate(model.head_clip_bioReverse):
        if old_head.head_fc.bias is None:
            fc = old_head.head_fc
            model.head_clip_bioReverse[index] = EvaluationOnlyReverseAdapter(
                fc.in_features, fc.out_features, old_head.alpha
            )
            adapters += 1
    state = normalize_state(torch.load(CHECKPOINT, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.train()  # training forward is needed to expose reverse logits; no grad/step is used

    bias_stats = {}
    weight_stats = {}
    ratios = {}
    for factor, head in enumerate(model.head_clip_bioReverse):
        bias_key = f"head_clip_bioReverse.{factor}.head_fc.bias"
        weight_key = f"head_clip_bioReverse.{factor}.head_fc.weight"
        bias_stats[bias_key] = tensor_stats(head.head_fc.bias)
        weight_stats[weight_key] = tensor_stats(head.head_fc.weight)
        ratios[str(factor)] = bias_stats[bias_key]["l2_norm"] / weight_stats[weight_key]["l2_norm"]

    full_losses = [[] for _ in model.head_clip_bioReverse]
    zero_losses = [[] for _ in model.head_clip_bioReverse]
    full_accs = [[] for _ in model.head_clip_bioReverse]
    zero_accs = [[] for _ in model.head_clip_bioReverse]
    full_logit_values = [[] for _ in model.head_clip_bioReverse]
    zero_logit_values = [[] for _ in model.head_clip_bioReverse]
    delta_l2 = [[] for _ in model.head_clip_bioReverse]
    delta_cos = [[] for _ in model.head_clip_bioReverse]
    pairs_checked = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            images = batch["image"].to(device)
            camids = batch["camid"].to(device)
            caption = batch["caption_feature"].to(device)
            _, feat = model(images, cam_label=camids)
            feat_bio = feat[0]

            full_outputs = reverse_outputs(model, feat_bio)
            full_l, full_a, full_logits = compute_reverse_metrics(full_outputs, caption)

            saved_bias = [head.head_fc.bias.detach().clone() for head in model.head_clip_bioReverse]
            for head in model.head_clip_bioReverse:
                head.head_fc.bias.zero_()
            zero_outputs = reverse_outputs(model, feat_bio)
            zero_l, zero_a, zero_logits = compute_reverse_metrics(zero_outputs, caption)
            for head, saved in zip(model.head_clip_bioReverse, saved_bias):
                head.head_fc.bias.copy_(saved)

            for factor in range(len(full_outputs)):
                full_losses[factor].append(float(full_l[factor].cpu()))
                zero_losses[factor].append(float(zero_l[factor].cpu()))
                full_accs[factor].append(float(full_a[factor].cpu()))
                zero_accs[factor].append(float(zero_a[factor].cpu()))
                f = full_logits[factor].detach().double().cpu()
                z = zero_logits[factor].detach().double().cpu()
                full_logit_values[factor].extend(f.reshape(-1).tolist())
                zero_logit_values[factor].extend(z.reshape(-1).tolist())
                delta_l2[factor].append(float(torch.linalg.vector_norm(f - z)))
                flat_f = f.reshape(-1)
                flat_z = z.reshape(-1)
                delta_cos[factor].append(float(F.cosine_similarity(flat_f, flat_z, dim=0)))
            pairs_checked += int(images.shape[0])
            if batch_index == 0 or (batch_index + 1) % 100 == 0:
                print("BIAS_FUNCTIONAL_PROGRESS", batch_index + 1, flush=True)

    factors = {}
    for factor in range(len(full_losses)):
        factors[str(factor)] = {
            "full_bias_loss_mean": float(np.mean(full_losses[factor])),
            "zero_bias_loss_mean": float(np.mean(zero_losses[factor])),
            "delta_loss_zero_minus_full": float(np.mean(zero_losses[factor]) - np.mean(full_losses[factor])),
            "full_bias_accuracy_mean": float(np.mean(full_accs[factor])),
            "zero_bias_accuracy_mean": float(np.mean(zero_accs[factor])),
            "delta_accuracy_zero_minus_full": float(np.mean(zero_accs[factor]) - np.mean(full_accs[factor])),
            "full_bias_logits": tensor_stats(torch.tensor(full_logit_values[factor], dtype=torch.float64)),
            "zero_bias_logits": tensor_stats(torch.tensor(zero_logit_values[factor], dtype=torch.float64)),
            "logit_delta_l2_mean": float(np.mean(delta_l2[factor])),
            "logit_delta_l2_max": float(np.max(delta_l2[factor])),
            "logit_cosine_full_vs_zero_mean": float(np.mean(delta_cos[factor])),
            "logit_cosine_full_vs_zero_min": float(np.min(delta_cos[factor])),
        }

    payload = {
        "config": str(CONFIG),
        "checkpoint": str(CHECKPOINT),
        "strict_load": {"missing_keys": list(missing), "unexpected_keys": list(unexpected)},
        "eval_only_reverse_adapters": adapters,
        "dataset": {"name": "ltcc", "train_images": len(dataset.train), "batch_size": local_cfg.DATA.BATCH_SIZE,
                     "pairs_checked": pairs_checked, "batches_checked": min(args.batches, len(loader)),
                     "transform": "deterministic test transform; sorted training images"},
        "bias_stats": bias_stats,
        "weight_stats": weight_stats,
        "bias_l2_over_weight_l2": ratios,
        "functional_comparison": factors,
        "conclusion": "Bias tensors are nonzero and functional comparison quantifies their effect on reverse logits/loss only; no backward or optimizer operation was performed.",
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("WROTE", out)
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
