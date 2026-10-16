#!/usr/local/bin/python3
"""Frozen LTCC reverse-target permutation diagnostics for PHASE 4R-C.

This script deliberately performs inference plus ``autograd.grad`` only.  It
does not construct an optimizer, call ``backward()``, or update parameters.
The reverse heads and the EVA/Bio representation are frozen.  The measured
gradients are the local gradients that the corresponding losses would send to
the representation; they are not training runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from model.build import factory  # noqa: E402


DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
OUT_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/target_permutation"

CHECKPOINTS = {
    "original_full": (
        ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/eva02_l_bio_best.pth",
        [1, 2],
        False,
    ),
    "official_full": (
        DATA_ROOT / "DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth",
        # The packaged official artifact is audited under its own recorded
        # topology [2,3] (Clothing + Pose).  It is not substituted for the
        # formal [1,2] training protocol.
        [2, 3],
        True,
    ),
    "bio_hair": (
        ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_hair/run1/eva02_l_bio_best.pth",
        [1],
        False,
    ),
    "bio_clothing": (
        ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_clothing/run1/eva02_l_bio_best.pth",
        [2],
        False,
    ),
    "bio_pose": (
        ROOT / "outputs/reproduction_phase3/LTCC/table4/bio_pose/run1/eva02_l_bio_best.pth",
        [3],
        False,
    ),
}


def make_config(nonbio):
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = str(DATA_ROOT)
    cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = list(nonbio)
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.TEST_BATCH = 32
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.freeze()
    return cfg


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.replace("module.", ""): value for key, value in state.items()}


def build_model(cfg, path, official):
    model = factory[cfg.MODEL.NAME](
        pretrained=False,
        config=cfg,
        num_classes=77,
        camera_num=12,
    )
    adapters = 0
    if official:
        for index, old in enumerate(model.head_clip_bioReverse):
            if old.head_fc.bias is None:
                replacement = nn.Linear(old.head_fc.in_features, old.head_fc.out_features, bias=True)
                replacement.weight.data.copy_(old.head_fc.weight.data)
                model.head_clip_bioReverse[index].head_fc = replacement
                adapters += 1
    state = load_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing checkpoint keys for {path}: {missing[:8]}")
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "checkpoint": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "official_bias_adapters": adapters,
    }


def groups_to_permutation(values, rng, mode):
    values = np.asarray(values)
    n = len(values)
    perm = np.arange(n, dtype=np.int64)
    groups = defaultdict(list)
    for index, value in enumerate(values.tolist()):
        groups[str(value)].append(index)
    if mode in {"within_pid", "within_clothes"}:
        for indices in groups.values():
            if len(indices) > 1:
                shuffled = np.asarray(indices, dtype=np.int64).copy()
                rng.shuffle(shuffled)
                if np.all(shuffled == np.asarray(indices)):
                    shuffled = np.roll(shuffled, 1)
                perm[np.asarray(indices)] = shuffled
        return perm
    if mode == "across_pid":
        # Find a derangement whose target PID differs from every source PID.
        for _ in range(1000):
            candidate = rng.permutation(n)
            if np.all(values[candidate] != values):
                return candidate.astype(np.int64)
        # This fallback is only expected for a pathological one-PID batch.
        order = np.argsort(values, kind="stable")
        candidate = np.roll(order, max(1, n // 2))
        return candidate.astype(np.int64)
    return perm


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def clip_loss(image_projection, text_projection):
    logits = normalize(image_projection) @ normalize(text_projection).T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def vector_stats(grad, reference):
    grad_flat = grad.detach().float().reshape(grad.shape[0], -1)
    ref_flat = reference.detach().float().reshape(reference.shape[0], -1)
    grad_norm = grad_flat.norm(dim=1)
    ref_norm = ref_flat.norm(dim=1)
    cosine = F.cosine_similarity(grad_flat, ref_flat, dim=1)
    return {
        "norm_mean": float(grad_norm.mean().item()),
        "norm_median": float(grad_norm.median().item()),
        "norm_p10": float(torch.quantile(grad_norm, 0.10).item()),
        "norm_p90": float(torch.quantile(grad_norm, 0.90).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_median": float(cosine.median().item()),
        "cosine_p10": float(torch.quantile(cosine, 0.10).item()),
        "cosine_p90": float(torch.quantile(cosine, 0.90).item()),
        "conflict_rate": float((cosine < 0).float().mean().item()),
        "norm_ratio_mean": float((grad_norm / (ref_norm + 1e-8)).mean().item()),
        "norm_ratio_median": float((grad_norm / (ref_norm + 1e-8)).median().item()),
    }


def batch_diagnostic(model, batch, factor_indices, factor_names, permutation_modes, seed):
    images = batch["image"].cuda(non_blocking=False)
    pids = np.asarray(batch["pid"], dtype=np.int64)
    clothes = np.asarray(batch["clothes_id"], dtype=np.int64)
    captions = batch["caption_feature"].cuda(non_blocking=False).float()
    # The released loader returns [SummaryBio, raw Hair/Clothing/Pose...].
    captions = captions.reshape(captions.shape[0], captions.shape[1], -1)

    with torch.enable_grad():
        # The checkpoint is frozen.  Detach the evaluated representation and
        # make only that tensor a gradient leaf: this measures the force that
        # each objective would apply at the reverse source, without building
        # or updating an EVA/backbone graph.
        feat = model(images, cam_label=batch["camid"].cuda(non_blocking=False)).detach().requires_grad_(True)
        feat.retain_grad()
        id_logits = model.head(feat)
        labels = torch.as_tensor(pids, device=feat.device, dtype=torch.long)
        id_loss = F.cross_entropy(id_logits, labels)
        bio_loss = clip_loss(model.head_clip_bio(feat), captions[:, 0])
        core_loss = id_loss + bio_loss
        g_id = torch.autograd.grad(id_loss, feat, retain_graph=True)[0]
        g_bio = torch.autograd.grad(bio_loss, feat, retain_graph=True)[0]
        g_core = torch.autograd.grad(core_loss, feat, retain_graph=True)[0]

        result = {
            "batch_size": int(len(pids)),
            "pid_count": int(len(np.unique(pids))),
            "clothes_count": int(len(np.unique(clothes))),
            "id_loss": float(id_loss.item()),
            "bio_loss": float(bio_loss.item()),
            "factors": {},
        }
        rng = np.random.default_rng(seed)
        for slot, (factor_index, factor_name) in enumerate(zip(factor_indices, factor_names)):
            if slot >= len(model.head_clip_bioReverse):
                continue
            head = model.head_clip_bioReverse[slot]
            # The GRL is intentionally retained: this is the actual gradient
            # sign seen by the shared representation in the formal graph.
            reverse_logits = head(feat)
            factor_result = {}
            for mode in permutation_modes:
                if mode == "original":
                    target = captions[:, slot + 1]
                    perm = np.arange(len(pids), dtype=np.int64)
                else:
                    values = pids if mode != "within_clothes" else clothes
                    perm = groups_to_permutation(values, rng, mode)
                    target = captions[torch.as_tensor(perm, device=captions.device), slot + 1]
                reverse_loss = clip_loss(reverse_logits, target)
                reverse_grad = torch.autograd.grad(reverse_loss, feat, retain_graph=True)[0]
                factor_result[mode] = {
                    "target_permutation": perm.tolist(),
                    "loss": float(reverse_loss.item()),
                    "gradient": vector_stats(reverse_grad, g_core),
                    "cosine_with_id_mean": float(F.cosine_similarity(
                        reverse_grad.float().reshape(reverse_grad.shape[0], -1),
                        g_id.float().reshape(g_id.shape[0], -1), dim=1).mean().item()),
                    "cosine_with_bio_mean": float(F.cosine_similarity(
                        reverse_grad.float().reshape(reverse_grad.shape[0], -1),
                        g_bio.float().reshape(g_bio.shape[0], -1), dim=1).mean().item()),
                }
            result["factors"][factor_name] = {
                "semantic_factor_index": int(factor_index),
                "slot": int(slot),
                "modes": factor_result,
            }
    return result


def aggregate(batch_results):
    out = {"batch_count": len(batch_results), "factors": {}}
    for factor in sorted({name for item in batch_results for name in item["factors"]}):
        out["factors"][factor] = {"semantic_factor_index": batch_results[0]["factors"][factor]["semantic_factor_index"], "modes": {}}
        modes = sorted({mode for item in batch_results for mode in item["factors"][factor]["modes"]})
        for mode in modes:
            rows = [item["factors"][factor]["modes"][mode] for item in batch_results]
            numeric = {}
            keys = ["loss", "cosine_with_id_mean", "cosine_with_bio_mean"]
            for key in keys:
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                numeric[key + "_mean"] = float(values.mean())
                numeric[key + "_std"] = float(values.std(ddof=0))
            for key in ["gradient"]:
                for subkey in rows[0][key]:
                    values = np.asarray([row[key][subkey] for row in rows], dtype=np.float64)
                    numeric[key + "_" + subkey + "_mean"] = float(values.mean())
                    numeric[key + "_" + subkey + "_std"] = float(values.std(ddof=0))
            out["factors"][factor]["modes"][mode] = numeric
    return out


def make_fixed_batch_plan(dataset, batch_count, batch_size=32, seed=7400):
    """Choose diverse, deterministic batches for the frozen diagnostic.

    The formal loader remains untouched.  Each diagnostic batch contains four
    identities and eight images per identity, which gives the within-PID
    counterfactual a meaningful support while retaining multiple PID targets.
    """
    by_pid = defaultdict(list)
    for index, item in enumerate(dataset.train):
        by_pid[int(item["pid"])].append(index)
    rng = np.random.default_rng(seed)
    pids = rng.permutation(sorted(by_pid))
    plan = []
    for batch_index in range(batch_count):
        selected = [int(pids[(batch_index * 4 + offset) % len(pids)]) for offset in range(4)]
        indices = []
        for pid in selected:
            candidates = np.asarray(by_pid[pid], dtype=np.int64)
            replace = len(candidates) < batch_size // 4
            picked = rng.choice(candidates, size=batch_size // 4, replace=replace)
            indices.extend(int(value) for value in picked)
        plan.append(indices)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["original_full", "official_full", "bio_hair", "bio_clothing", "bio_pose", "all"], default="all")
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 4R-C target diagnostics require PPU/CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = list(CHECKPOINTS) if args.which == "all" else [args.which]
    results = {
        "phase": "4R-C7",
        "inference_only": True,
        "no_optimizer": True,
        "no_parameter_update": True,
        "device": torch.cuda.get_device_name(0),
        "batch_count_per_run": args.batches,
        "permutation_modes": ["original", "within_pid", "within_clothes", "across_pid"],
        "gradient_definition": "autograd.grad of CLIP reverse loss with the released GRL retained, with respect to frozen feat_bio; core is ID CE + Bio CLIP loss",
        "runs": {},
    }
    for name in selected:
        path, factor_indices, official = CHECKPOINTS[name]
        if not path.exists():
            raise FileNotFoundError(path)
        cfg = make_config(factor_indices)
        dataset = build_dataset(cfg)
        transform = build_img_transforms(cfg)[1]
        fixed_plan = make_fixed_batch_plan(dataset, args.batches)
        fixed_indices = [index for batch_indices in fixed_plan for index in batch_indices]
        loader = DataLoader(
            Subset(ImageDataset(dataset.train, transform=transform), fixed_indices),
            batch_size=32,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            drop_last=False,
        )
        model, load_info = build_model(cfg, path, official)
        factor_names = [{1: "hair", 2: "clothing", 3: "pose"}[index] for index in factor_indices]
        batch_results = []
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            # Fix Python/NumPy state around each diagnostic for reproducible
            # target permutations without changing model/data execution.
            random.seed(1234 + batch_index)
            np.random.seed(1234 + batch_index)
            batch_results.append(batch_diagnostic(
                model,
                batch,
                factor_indices,
                factor_names,
                ["original", "within_pid", "within_clothes", "across_pid"],
                4000 + batch_index,
            ))
            print("TARGET_BATCH", name, batch_index, flush=True)
        results["runs"][name] = {
            **load_info,
            "semantic_factor_indices": factor_indices,
            "factor_names": factor_names,
            "fixed_batch_plan_dataset_indices": fixed_plan,
            "aggregate": aggregate(batch_results),
            "batch_details": batch_results,
        }
        del model, dataset, batch_results
        torch.cuda.empty_cache()
    output = OUT_ROOT / "target_permutation_diagnostics.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
