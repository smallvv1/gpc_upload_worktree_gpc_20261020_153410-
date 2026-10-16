#!/usr/local/bin/python3
"""PHASE 2I: natural-construction RNG footprint and first-200 probe.

This file is deliberately a diagnostic driver.  It does not change any
repository model/loss source and it never starts a formal epoch run.  A, B,
and C are constructed independently after the same seed and loader setup:

  A: public/current reverse heads (bias=False)
  B: native historical candidate (bias=True, trainable)
  C: native historical construction, then zero/frozen biases

The data stream is materialized separately for each candidate so a possible
construction-to-loader RNG coupling is measured rather than accidentally
removed by forcing a common stream.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg as GLOBAL_CFG
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from loss import make_loss
from model import build_model
from model import eva_meta_bio
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


OFFICIAL_EVENT_DIR = Path("/home/data/DIFFER_OfficialWeights/provenance/LTCC/train")
OFFICIAL_CHECKPOINT = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
WORLD_SIZE = 2
RANK = 0
SEED = 1234
INITIAL_SCALE = 1024
DEFAULT_ITERS = 200

FACTOR_NAMES = {0: "bio", 1: "hair", 2: "clothing", 3: "pose"}
SCALAR_MAP = OrderedDict([
    ("total_loss", "loss/total"),
    ("id_loss", "loss/id"),
    ("triplet_loss", "loss/triplet"),
    ("bio_loss", "loss/clip_bio_0"),
    ("reverse0_loss", "loss/clip_bio_reverse_0"),
    ("reverse1_loss", "loss/clip_bio_reverse_1"),
    ("id_acc", "acc/id"),
    ("bio_acc", "acc/clip_bio_0"),
    ("reverse0_acc", "acc/clip_bio_reverse_0"),
    ("reverse1_acc", "acc/clip_bio_reverse_1"),
])


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_bytes(state: Any) -> bytes:
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(state, np.ndarray):
        return state.tobytes()
    if isinstance(state, (list, tuple)):
        return b"[" + b"|".join(_state_bytes(x) for x in state) + b"]"
    if isinstance(state, dict):
        return b"{" + b"|".join(
            str(k).encode() + b":" + _state_bytes(v) for k, v in sorted(state.items(), key=lambda kv: str(kv[0]))
        ) + b"}"
    return repr(state).encode()


def rng_hashes() -> dict[str, str]:
    cuda_states = []
    try:
        cuda_states = torch.cuda.get_rng_state_all()
    except Exception as exc:  # diagnostic output must retain the failure
        cuda_states = [f"cuda_rng_error:{type(exc).__name__}:{exc}"]
    return {
        "torch_cpu": _sha256_bytes(_state_bytes(torch.get_rng_state())),
        "torch_cuda_ppu": _sha256_bytes(_state_bytes(cuda_states)),
        "numpy": _sha256_bytes(_state_bytes(np.random.get_state())),
        "python_random": _sha256_bytes(_state_bytes(random.getstate())),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def make_config():
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(REPO_ROOT / "configs/ltcc/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.MODEL.DIST_TRAIN = False
    local_cfg.DATA.ROOT = "/home/data"
    local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.SOLVER.SEED = SEED
    local_cfg.freeze()
    return local_cfg


class NativeBiasClassifier(nn.Module):
    """Exact native constructor candidate, with only Linear bias=True changed."""

    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        # nn.Linear(reset_parameters) is intentionally allowed to run.  Its
        # bias allocation/reset is the RNG footprint under test.  Eva's
        # enclosing _init_weights subsequently zeroes this bias, as in the
        # native module's intended initialization convention.
        self.head_fc = nn.Linear(in_num, class_num, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            from timm.layers import trunc_normal_
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.head_fc(self.grl(x))


def build_natural_model(local_cfg, num_classes: int, num_camera: int, biasful: bool):
    """Build in the public factory while replacing its constructor symbol."""
    original = eva_meta_bio.GradientReversalClassifier
    if biasful:
        eva_meta_bio.GradientReversalClassifier = NativeBiasClassifier
    try:
        model = build_model(local_cfg, num_classes, num_camera)
    finally:
        eva_meta_bio.GradientReversalClassifier = original
    return model


def build_probe_loader(local_cfg):
    """Build the official train stream without requiring NCCL initialization."""
    dataset = build_dataset(local_cfg)
    transform_train, _ = build_img_transforms(local_cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=local_cfg.DATA.NUM_INSTANCES,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        seed=local_cfg.SOLVER.SEED,
    )
    sampler.set_epoch(0)
    generator = torch.Generator()
    generator.manual_seed(local_cfg.SOLVER.SEED)
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sampler,
        batch_size=local_cfg.DATA.BATCH_SIZE,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=generator,
    )
    return dataset, loader, sampler


def _tensor_checksum(value: torch.Tensor) -> str:
    return _sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def batch_fingerprint(batch: dict[str, Any]) -> dict[str, Any]:
    paths = [str(x) for x in batch["image_path"]]
    pids = [int(x) for x in batch["pid"].tolist()]
    cams = [int(x) for x in batch["camid"].tolist()]
    clothes = [int(x) for x in batch["clothes_id"].tolist()]
    path_payload = "|".join(f"{p}:{c}:{cl}:{path}" for p, c, cl, path in zip(pids, cams, clothes, paths)).encode()
    return {
        "sample_sha256": _sha256_bytes(path_payload),
        "image_sha256": _tensor_checksum(batch["image"]),
        "pids": pids,
        "camids": cams,
        "clothes": clothes,
        "paths": paths,
    }


def materialize_batches(loader, count: int):
    batches = []
    fingerprints = []
    for index, raw in enumerate(loader):
        if index >= count:
            break
        batch = {
            key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in raw.items()
        }
        batches.append(batch)
        fingerprints.append(batch_fingerprint(batch))
    return batches, fingerprints


def state_dict_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def parameter_group_name(name: str) -> str:
    if name.startswith("head_clip_bioReverse."):
        return "reverse"
    if name.startswith(("head_bio.", "norm_bio.", "head_clip_bio.")):
        return "bio"
    if name.startswith("head."):
        return "id_classifier"
    if name.startswith(("blocks.", "patch_embed.", "camera_embed", "pos_embed", "cls_token", "head_bio")):
        return "eva_backbone"
    if name.startswith("base."):
        return "eva_backbone"
    return "other"


def diff_tensor(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)
    d = x - y
    max_abs = float(d.abs().max().item()) if d.numel() else 0.0
    l2 = float(torch.linalg.vector_norm(d).item()) if d.numel() else 0.0
    yn = float(torch.linalg.vector_norm(y).item())
    xn = float(torch.linalg.vector_norm(x).item())
    cosine = float(F.cosine_similarity(x, y, dim=0).item()) if xn and yn else None
    return {"max_abs": max_abs, "l2": l2, "relative_l2": l2 / yn if yn else None, "cosine": cosine}


def compare_states(state_a: dict[str, torch.Tensor], state_b: dict[str, torch.Tensor], state_c: dict[str, torch.Tensor]):
    common = sorted(set(state_a) & set(state_b) & set(state_c))
    identical_ab = []
    differing_ab = []
    differing_by_group: dict[str, int] = {}
    detail = {}
    for name in common:
        item = diff_tensor(state_a[name], state_b[name])
        detail[name] = item
        if torch.equal(state_a[name], state_b[name]):
            identical_ab.append(name)
        else:
            differing_ab.append(name)
            group = parameter_group_name(name)
            differing_by_group[group] = differing_by_group.get(group, 0) + 1
    bc_common = sorted(set(state_b) & set(state_c))
    bc_identical = [name for name in bc_common if torch.equal(state_b[name], state_c[name])]
    bc_diffs = [name for name in bc_common if not torch.equal(state_b[name], state_c[name])]
    focus = {}
    focus_names = [
        "head.weight", "head.bias", "camera_embed", "head_bio.0.attn.q_proj.weight",
        "head_clip_bio.weight", "head_clip_bioReverse.0.head_fc.weight",
        "head_clip_bioReverse.1.head_fc.weight", "head_clip_bioReverse.0.head_fc.bias",
        "head_clip_bioReverse.1.head_fc.bias",
    ]
    for name in focus_names:
        if name in state_a and name in state_b:
            focus[name] = detail.get(name, diff_tensor(state_a[name], state_b[name]))
    return {
        "a_parameter_count": len(state_a),
        "b_parameter_count": len(state_b),
        "c_parameter_count": len(state_c),
        "common_count": len(common),
        "a_b_identical_count": len(identical_ab),
        "a_b_different_count": len(differing_ab),
        "a_b_different_by_group": differing_by_group,
        "b_c_common_count": len(bc_common),
        "b_c_identical_count": len(bc_identical),
        "b_c_different_count": len(bc_diffs),
        "a_b_different_names_first100": differing_ab[:100],
        "b_c_different_names_first100": bc_diffs[:100],
        "focus_differences_a_vs_b": focus,
    }


def optimizer_groups(model: nn.Module, local_cfg):
    # make_loss creates a CenterLoss even though it is not used by this config;
    # this mirrors the released construction order and optimizer coverage.
    _, center = make_loss(local_cfg, num_classes=77)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center)
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append({
            "index": index,
            "names": [name_by_id.get(id(p), "<unknown>") for p in group["params"]],
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        })
    del optimizer, optimizer_center, center
    gc.collect()
    return groups


class ScalarCapture:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values[str(tag)] = float(value)


def grad_summary(model: nn.Module) -> dict[str, Any]:
    out = {}
    for group in ("eva_backbone", "id_classifier", "bio", "reverse"):
        stats = {"parameter_count": 0, "grad_exists": 0, "nan_count": 0, "inf_count": 0, "max_abs_finite": None}
        maxima = []
        for name, parameter in model.named_parameters():
            if parameter_group_name(name) != group or not parameter.requires_grad:
                continue
            stats["parameter_count"] += 1
            if parameter.grad is None:
                continue
            stats["grad_exists"] += 1
            g = parameter.grad.detach().double()
            stats["nan_count"] += int(torch.isnan(g).sum().item())
            stats["inf_count"] += int(torch.isinf(g).sum().item())
            finite = g[torch.isfinite(g)]
            if finite.numel():
                maxima.append(float(finite.abs().max().item()))
        stats["max_abs_finite"] = max(maxima) if maxima else None
        out[group] = stats
    return out


def scalar_record(writer: ScalarCapture, total_loss: torch.Tensor, iteration: int) -> dict[str, Any]:
    values = {key: writer.values.get(tag) for key, tag in SCALAR_MAP.items()}
    values["total_loss"] = float(total_loss.detach().double().cpu())
    return {
        "iteration": iteration,
        "values": values,
        "total_loss_dtype": str(total_loss.dtype),
        "loss_finite": bool(torch.isfinite(total_loss.detach()).item()),
    }


def train_candidate(label: str, model: nn.Module, local_cfg, dataset, batches, rng_state, device):
    restore_rng_state(rng_state)
    model.to(device)
    model.train()
    loss_fn, center = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(local_cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    names = dict(model.named_parameters())
    monitor_names = [
        "base.patch_embed.proj.weight", "head.weight", "head_clip_bio.weight",
        "head_clip_bioReverse.0.head_fc.weight", "head_clip_bioReverse.1.head_fc.weight",
    ]
    monitor_names += [name for name in names if name.endswith("head_fc.bias")]
    records = []
    skipped = 0
    nonfinite_loss = 0
    nonfinite_grad = 0
    update_failures = 0
    train_rng_after_loader = capture_rng_state()
    for iteration, raw_batch in enumerate(batches, start=1):
        batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw_batch.items()}
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        writer = ScalarCapture()
        scale_before = float(scaler.get_scale())
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        total_loss = loss_fn(
            score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
            clothes_ids=batch["clothes_id"], train_writer=writer, step=iteration,
        )
        loss_is_finite = bool(torch.isfinite(total_loss.detach()).item())
        nonfinite_loss += int(not loss_is_finite)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradient_stats = grad_summary(model)
        gradients_finite = all(item["nan_count"] == 0 and item["inf_count"] == 0 for item in gradient_stats.values())
        nonfinite_grad += int(not gradients_finite)
        before = {name: names[name].detach().clone() for name in monitor_names if name in names}
        scaler.step(optimizer)
        found_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = sum(float(value.item()) for value in found_state.values())
        step_skipped = found_inf > 0
        skipped += int(step_skipped)
        deltas = {name: float((names[name].detach() - before[name]).float().abs().max().item()) for name in before}
        required = [name for name in ("base.patch_embed.proj.weight", "head.weight") if name in deltas]
        parameter_updated = (not required) or all(deltas[name] > 0 for name in required)
        update_failures += int(not parameter_updated)
        scaler.update()
        scale_after = float(scaler.get_scale())
        record = scalar_record(writer, total_loss, iteration)
        record.update({
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "step_skipped": step_skipped,
            "parameter_updated": parameter_updated,
            "parameter_deltas": deltas,
            "gradient_finite": gradients_finite,
            "gradient_groups": gradient_stats,
            "lr_normal": float(min(group["lr"] for group in optimizer.param_groups)),
            "lr_bias": float(max(group["lr"] for group in optimizer.param_groups)),
        })
        if label in ("B", "C"):
            record["bias_norms"] = {
                name: float(torch.linalg.vector_norm(parameter.detach().double()).item())
                for name, parameter in names.items() if name.endswith("head_fc.bias")
            }
            record["bias_grads"] = {
                name: {
                    "exists": parameter.grad is not None,
                    "finite": parameter.grad is None or bool(torch.isfinite(parameter.grad.detach()).all().item()),
                    "max_abs": None if parameter.grad is None else float(parameter.grad.detach().abs().max().item()),
                }
                for name, parameter in names.items() if name.endswith("head_fc.bias")
            }
        records.append(record)
        if iteration == 1 or iteration % 25 == 0:
            print("PHASE2I_PROGRESS", json.dumps({
                "candidate": label, "iteration": iteration, "scale_before": scale_before,
                "scale_after": scale_after, "found_inf": found_inf,
                "step_skipped": step_skipped, "parameter_updated": parameter_updated,
                "total_loss": record["values"]["total_loss"],
            }, sort_keys=True), flush=True)
    result = {
        "label": label,
        "iterations": len(records),
        "records": records,
        "skipped_steps": skipped,
        "nonfinite_loss_count": nonfinite_loss,
        "nonfinite_gradient_iterations": nonfinite_grad,
        "parameter_update_failures": update_failures,
        "successful_updates": len(records) - skipped,
        "initial_scale": records[0]["scale_before"] if records else None,
        "final_scale": records[-1]["scale_after"] if records else None,
        "train_rng_after_loader_hashes": {
            "torch_cpu": _sha256_bytes(_state_bytes(train_rng_after_loader["torch"])),
            "torch_cuda_ppu": _sha256_bytes(_state_bytes(train_rng_after_loader["cuda"])),
            "numpy": _sha256_bytes(_state_bytes(train_rng_after_loader["numpy"])),
            "python_random": _sha256_bytes(_state_bytes(train_rng_after_loader["python"])),
        },
    }
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler
    model.to("cpu")
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def find_event_files() -> list[Path]:
    return sorted(OFFICIAL_EVENT_DIR.glob("events.out.tfevents.*"))


def read_official_first200() -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = find_event_files()
    if not files:
        raise FileNotFoundError(f"No official TensorBoard event files in {OFFICIAL_EVENT_DIR}")
    all_series: dict[str, list[dict[str, Any]]] = {}
    file_meta = []
    for path in files:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        file_meta.append({"path": str(path), "size": path.stat().st_size, "scalar_tags": tags})
        for tag in tags:
            all_series.setdefault(tag, []).extend({
                "step": int(item.step), "wall_time": float(item.wall_time), "value": float(item.value), "file": str(path)
            } for item in acc.Scalars(tag))
    # The two official files are rank workers with identical scalar streams.
    # De-duplicate exact (tag, step, value) records, retaining one stream.
    dedup: dict[str, dict[int, dict[str, Any]]] = {}
    for tag, values in all_series.items():
        by_step = dedup.setdefault(tag, {})
        for value in sorted(values, key=lambda item: (item["step"], item["wall_time"], item["file"])):
            by_step.setdefault(value["step"], value)
    tags = sorted(dedup)
    train_tags = [tag for tag in tags if tag not in ("rank1", "mAP") and len(dedup[tag]) > 1]
    if not train_tags:
        raise RuntimeError("Official events contain no training scalar series")
    first_steps = sorted(dedup[train_tags[0]])[:DEFAULT_ITERS]
    # Ensure rows are sequence-aligned even if one scalar lacks a record.
    rows = []
    for sequence, step in enumerate(first_steps, start=1):
        row = {"sequence": sequence, "official_step": step}
        for tag in tags:
            row[tag] = dedup[tag].get(step, {}).get("value")
        rows.append(row)
    return {"files": file_meta, "tags": tags, "dedup_series": dedup, "first200_rows": rows}


def write_official_csv(info: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = info["tags"]
    fields = ["sequence", "official_step"] + tags
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in info["first200_rows"]:
            writer.writerow({field: row.get(field, "") for field in fields})


def official_arrays(info: dict[str, Any]) -> dict[str, list[float | None]]:
    rows = info["first200_rows"]
    output = {}
    for name, tag in SCALAR_MAP.items():
        official_tag = "loss/clip_bio" if tag == "loss/clip_bio_0" else tag
        official_tag = "acc/clip_bio" if tag == "acc/clip_bio_0" else official_tag
        output[name] = [row.get(official_tag) for row in rows]
    return output


def candidate_arrays(result: dict[str, Any]) -> dict[str, list[float | None]]:
    return {name: [record["values"].get(name) for record in result["records"]] for name in SCALAR_MAP}


def trajectory_distance(official: dict[str, list[float | None]], candidate: dict[str, list[float | None]]):
    report = {}
    for name in SCALAR_MAP:
        pairs = [(float(a), float(b)) for a, b in zip(official.get(name, []), candidate.get(name, [])) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
        if not pairs:
            report[name] = {"count": 0, "rmse": None, "mae": None, "pearson": None, "normalized_rmse": None}
            continue
        a = np.asarray([p[0] for p in pairs], dtype=np.float64)
        b = np.asarray([p[1] for p in pairs], dtype=np.float64)
        diff = b - a
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mae = float(np.mean(np.abs(diff)))
        if a.size > 1 and float(np.std(a)) > 1e-15 and float(np.std(b)) > 1e-15:
            pearson = float(np.corrcoef(a, b)[0, 1])
        else:
            pearson = None
        normalized = rmse / (float(np.std(a)) + 1e-12)
        report[name] = {"count": int(len(pairs)), "rmse": rmse, "mae": mae, "pearson": pearson, "normalized_rmse": normalized}
    usable = [v["normalized_rmse"] for v in report.values() if v["normalized_rmse"] is not None and math.isfinite(v["normalized_rmse"])]
    report["composite_score"] = float(np.mean(usable)) if usable else None
    return report


def serializable_rng_trace(trace):
    return trace


def construct_candidate(label: str, local_cfg, iters: int):
    seed_all(SEED)
    trace = {"T0_after_seed": rng_hashes()}
    dataset, loader, sampler = build_probe_loader(local_cfg)
    trace["T1_after_build_dataloader"] = rng_hashes()
    model = build_natural_model(local_cfg, dataset.num_train_pids, dataset.num_camera, biasful=label in ("B", "C"))
    trace["T2_after_build_model"] = rng_hashes()
    # build_model's pretrained=True load is complete before it returns.
    trace["T3_after_pretrained_loading_complete"] = rng_hashes()
    state_constructed = state_dict_parameters(model)
    dtype = dtype_audit(model)
    groups = optimizer_groups(model, local_cfg)
    trace["T4_before_first_train_loader_iteration"] = rng_hashes()
    batches, fingerprints = materialize_batches(loader, iters)
    post_loader_rng = capture_rng_state()
    if label == "C":
        for name, parameter in model.named_parameters():
            if name.endswith("head_fc.bias"):
                parameter.data.zero_()
                parameter.requires_grad_(False)
    state_active = state_dict_parameters(model)
    return {
        "label": label,
        "model": model,
        "dataset": dataset,
        "batches": batches,
        "fingerprints": fingerprints,
        "rng_trace": trace,
        "state_constructed": state_constructed,
        "state_active": state_active,
        "dtype_audit": dtype,
        "optimizer_groups": groups,
        "post_loader_rng": post_loader_rng,
        "sampler": {"type": type(sampler).__name__, "world_size": WORLD_SIZE, "rank": RANK, "epoch": sampler.epoch, "length": len(sampler)},
    }


def dtype_audit(model: nn.Module):
    groups = {"eva_backbone": {}, "id_classifier": {}, "bio": {}, "reverse": {}, "other": {}}
    for name, parameter in model.named_parameters():
        group = parameter_group_name(name)
        d = str(parameter.dtype)
        item = groups[group].setdefault(d, {"count": 0, "numel": 0})
        item["count"] += 1
        item["numel"] += int(parameter.numel())
    return groups


def write_candidate_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iteration", "scale_before", "scale_after", "found_inf", "step_skipped", "parameter_updated", "total_loss", "total_loss_dtype", "loss_finite"] + list(SCALAR_MAP.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result["records"]:
            row = {field: record.get(field, "") for field in fields}
            row.update({name: record["values"].get(name, "") for name in SCALAR_MAP})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostics/initialization_footprint_phase2i.json"))
    args = parser.parse_args()
    local_cfg = make_config()
    torch.cuda.set_device(torch.device("cuda:0"))

    print("PHASE2I_START", json.dumps({"iterations": args.iterations, "seed": SEED, "world_size": WORLD_SIZE, "rank": RANK, "initial_scale": INITIAL_SCALE}, sort_keys=True), flush=True)
    official = read_official_first200()
    write_official_csv(official, REPO_ROOT / "reports/OFFICIAL_FIRST200_SCALARS.csv")
    official_arr = official_arrays(official)

    candidates = {}
    for label in ("A", "B", "C"):
        print("PHASE2I_CONSTRUCT", label, flush=True)
        candidate = construct_candidate(label, local_cfg, args.iterations)
        candidates[label] = candidate
        print("PHASE2I_CONSTRUCTED", label, json.dumps({"rng": candidate["rng_trace"], "state_keys": len(candidate["state_constructed"]), "sampler": candidate["sampler"]}, sort_keys=True), flush=True)

    states_constructed = {label: candidates[label]["state_constructed"] for label in candidates}
    states_active = {label: candidates[label]["state_active"] for label in candidates}
    state_cmp_constructed = compare_states(states_constructed["A"], states_constructed["B"], states_constructed["C"])
    state_cmp_active = compare_states(states_active["A"], states_active["B"], states_active["C"])
    batch_equal = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        batch_equal[f"{left}_{right}_sample_order"] = [x["sample_sha256"] for x in candidates[left]["fingerprints"]] == [x["sample_sha256"] for x in candidates[right]["fingerprints"]]
        batch_equal[f"{left}_{right}_augmentation"] = [x["image_sha256"] for x in candidates[left]["fingerprints"]] == [x["image_sha256"] for x in candidates[right]["fingerprints"]]

    # Run each natural candidate from its own post-loader RNG state.  This is
    # deliberately not a common-init experiment.
    device = torch.device("cuda:0")
    results = {}
    for label in ("A", "B", "C"):
        print("PHASE2I_TRAIN", label, flush=True)
        results[label] = train_candidate(label, candidates[label]["model"], local_cfg, candidates[label]["dataset"], candidates[label]["batches"], candidates[label]["post_loader_rng"], device)
        write_candidate_csv(results[label], REPO_ROOT / f"diagnostics/phase2i_candidate_{label}_200.csv")
        # Do not retain large materialized image batches after each run.
        candidates[label]["batches"] = []
        candidates[label]["model"] = None
        gc.collect()
        torch.cuda.empty_cache()

    distances = {label: trajectory_distance(official_arr, candidate_arrays(results[label])) for label in results}
    ranking = sorted(((label, distances[label].get("composite_score")) for label in distances), key=lambda item: float("inf") if item[1] is None else item[1])
    payload = {
        "phase": "2I",
        "config": {"dataset": "ltcc", "seed": SEED, "world_size": WORLD_SIZE, "rank": RANK, "batch_size": int(local_cfg.DATA.BATCH_SIZE), "initial_scale": INITIAL_SCALE, "loss_path": "official autocast forward / loss outside autocast", "text_model_version": str(local_cfg.DATA.TEXT_MODEL_VERSION), "caption_dir": str(local_cfg.DATA.CAPTION_DIR)},
        "official_first200": {"event_dir": str(OFFICIAL_EVENT_DIR), "files": official["files"], "tags": official["tags"], "first200_step_range": [official["first200_rows"][0]["official_step"], official["first200_rows"][-1]["official_step"]] if official["first200_rows"] else None},
        "candidates": {
            label: {"rng_trace": candidates[label]["rng_trace"], "dtype_audit": candidates[label]["dtype_audit"], "optimizer_groups": candidates[label]["optimizer_groups"], "sampler": candidates[label]["sampler"], "fingerprints_first5": candidates[label]["fingerprints"][:5], "fingerprints_last": candidates[label]["fingerprints"][-1] if candidates[label]["fingerprints"] else None}
            for label in candidates
        },
        "state_comparison_constructed": state_cmp_constructed,
        "state_comparison_active": state_cmp_active,
        "batch_fingerprint_comparison": batch_equal,
        "results": results,
        "trajectory_distance": distances,
        "candidate_ranking_low_score_best": [{"candidate": label, "composite_score": score} for label, score in ranking],
        "decision_inputs": {
            "linear_bias_rng_footprint": None,
            "first_rng_divergence": None,
            "construction_common_params_differ": state_cmp_constructed["a_b_different_count"] > 0,
            "b_c_common_bitwise_identical": state_cmp_constructed["b_c_different_count"] == 0,
        },
        "notes": [
            "A/B/C are independent natural constructions after seed=1234 and train-loader construction; no common initialization is forced.",
            "The diagnostic uses an explicit rank-0/world-size-2 sampler stream without initializing NCCL, matching the proven per-rank batch semantics while remaining single-process.",
            "Candidate C zeros and freezes native reverse biases only after native bias=True construction, preserving its construction RNG footprint.",
            "Official first200 rows are sequence-aligned to training scalar steps 596 onward; duplicate rank event records are deduplicated by exact scalar step.",
        ],
    }
    # Isolated Linear RNG test is independent of the large model construction.
    seed_all(SEED)
    nn.Linear(1024, 512, bias=False)
    iso_false = rng_hashes()
    seed_all(SEED)
    nn.Linear(1024, 512, bias=True)
    iso_true = rng_hashes()
    payload["isolated_linear_rng_test"] = {"bias_false_after": iso_false, "bias_true_after": iso_true, "states_differ": iso_false != iso_true}
    payload["decision_inputs"]["linear_bias_rng_footprint"] = iso_false != iso_true
    traces = [candidates[label]["rng_trace"] for label in ("A", "B", "C")]
    trace_keys = ["T0_after_seed", "T1_after_build_dataloader", "T2_after_build_model", "T3_after_pretrained_loading_complete", "T4_before_first_train_loader_iteration"]
    first_div = None
    for key in trace_keys:
        if not (traces[0][key] == traces[1][key] == traces[2][key]):
            first_div = key
            break
    payload["decision_inputs"]["first_rng_divergence"] = first_div

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2I_SUMMARY", json.dumps({"first_rng_divergence": first_div, "linear_bias_rng_diff": payload["isolated_linear_rng_test"]["states_differ"], "constructed_ab_different": state_cmp_constructed["a_b_different_count"], "constructed_bc_different": state_cmp_constructed["b_c_different_count"], "batch_equal": batch_equal, "ranking": ranking}, sort_keys=True), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
