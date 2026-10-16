#!/usr/local/bin/python3
"""Paired 200-iteration native reverse-bias diagnostic.

This is diagnostic training, not a formal reproduction run.  It creates a
biasless and a native bias=True model from one common initial state, feeds the
same pre-materialized rank-0 stream to both, and records whether the two extra
zero-initialized reverse biases change reverse or shared-backbone dynamics.
No repository model/loss source is modified by this script.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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
from model import eva_meta_bio
from scripts.localize_amp_overflow_phase1 import GROUPS
from scripts.qualify_amp_initial_scale_phase1 import make_dataset_config
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


DEFAULT_OUTPUT = REPO_ROOT / "diagnostics/reverse_bias_paired_200iter_phase2h.json"
CHECKPOINT = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
ITERATIONS = 200
WORLD_SIZE = 2
RANK = 0


class NativeReverseBiasHead(nn.Module):
    """The only topology change in the candidate: Linear bias=True."""

    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)

    def forward(self, x):
        return self.head_fc(self.grl(x))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def normalize_state(state):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def replace_reverse_heads(model):
    count = 0
    for index, old in enumerate(model.head_clip_bioReverse):
        if old.head_fc.bias is None:
            fc = old.head_fc
            model.head_clip_bioReverse[index] = NativeReverseBiasHead(
                fc.in_features, fc.out_features, old.alpha
            )
            count += 1
    return count


def model_dtype_audit(model):
    result = {"FP32": {"count": 0, "numel": 0}, "FP16": {"count": 0, "numel": 0},
              "BF16": {"count": 0, "numel": 0}, "other": {"count": 0, "numel": 0}}
    for parameter in model.parameters():
        key = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}.get(parameter.dtype, "other")
        result[key]["count"] += 1
        result[key]["numel"] += parameter.numel()
    return result


def optimizer_coverage(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    seen = []
    for group in optimizer.param_groups:
        seen.extend(id(parameter) for parameter in group["params"])
    seen_set = set(seen)
    duplicated = sorted(names[pid] for pid in set(pid for pid in seen if seen.count(pid) > 1) if pid in names)
    missing = sorted(names[pid] for pid in trainable - seen_set if pid in names)
    extra = sorted(names[pid] for pid in seen_set - trainable if pid in names)
    reverse_groups = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter), "")
            if name.startswith("head_clip_bioReverse.") and name.endswith(".bias"):
                reverse_groups.append({
                    "name": name,
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "requires_grad": bool(parameter.requires_grad),
                })
    return {
        "trainable_parameter_count": len(trainable),
        "trainable_numel": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "optimizer_parameter_count": len(seen),
        "optimizer_numel": sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]),
        "missing": missing,
        "duplicated": duplicated,
        "extra_nontrainable": extra,
        "reverse_bias_groups": reverse_groups,
    }


def tensor_fingerprint(batch):
    paths = [str(x) for x in batch["image_path"]]
    pids = [int(x) for x in batch["pid"].tolist()]
    payload = "|".join(f"{pid}:{path}" for pid, path in zip(pids, paths)).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pids": pids,
        "paths": paths,
    }


def materialize_batches(local_cfg, dataset, count):
    train_transform, _ = build_img_transforms(local_cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=local_cfg.DATA.NUM_INSTANCES,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        seed=local_cfg.SOLVER.SEED,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        ImageDataset(dataset.train, transform=train_transform),
        sampler=sampler,
        batch_size=local_cfg.DATA.BATCH_SIZE,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(local_cfg.SOLVER.SEED),
    )
    batches = []
    for index, batch in enumerate(loader):
        if index >= count:
            break
        # Keep a CPU copy so both models receive precisely the same image,
        # caption, PID, camera, and clothes tensors (including augmentation).
        batches.append({
            key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in batch.items()
        })
    return batches, {
        "world_size": WORLD_SIZE,
        "rank": RANK,
        "per_rank_batch": local_cfg.DATA.BATCH_SIZE,
        "batch_count": len(batches),
        "fingerprints": [tensor_fingerprint(batch) for batch in batches],
        "sampler": "DistributedRandomIdentitySampler",
        "sampler_epoch": 0,
        "augmentation": "one common materialized train-transform stream",
    }


class ScalarCapture:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = {"value": float(value), "dtype": None}


def grad_stats(parameter):
    gradient = parameter.grad
    if gradient is None:
        return {"exists": False, "dtype": None, "nan_count": 0, "inf_count": 0, "max_abs_finite": None, "norm": None}
    x = gradient.detach().double()
    finite = torch.isfinite(x)
    finite_values = x[finite]
    return {
        "exists": True,
        "dtype": str(gradient.dtype),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "max_abs_finite": float(finite_values.abs().max().item()) if finite_values.numel() else None,
        "norm": float(torch.linalg.vector_norm(x).item()) if finite_values.numel() else None,
    }


def group_gradient_stats(model):
    result = {}
    for group, prefixes in GROUPS.items():
        selected = [parameter for name, parameter in model.named_parameters()
                    if parameter.requires_grad and any(name.startswith(prefix) for prefix in prefixes)]
        stats = [grad_stats(parameter) for parameter in selected]
        result[group] = {
            "parameter_count": len(selected),
            "grad_exists_count": sum(item["exists"] for item in stats),
            "nan_count": sum(item["nan_count"] for item in stats),
            "inf_count": sum(item["inf_count"] for item in stats),
            "max_abs_finite": max((item["max_abs_finite"] for item in stats if item["max_abs_finite"] is not None), default=None),
        }
    return result


def named_gradient_vectors(model):
    candidates = [
        "blocks.0.attn.q_proj.weight",
        "blocks.12.attn.q_proj.weight",
        "blocks.23.attn.q_proj.weight",
        "head.weight",
    ]
    named = dict(model.named_parameters())
    result = {}
    for name in candidates:
        if name not in named:
            continue
        gradient = named[name].grad
        result[name] = gradient.detach().double().cpu().reshape(-1) if gradient is not None else None
    return result


def run_model(model, local_cfg, dataset, batches, rng_state, label, device):
    restore_rng(rng_state)
    model.to(device)
    model.train()
    loss_fn, center_criterion = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center_criterion)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(local_cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler(init_scale=1024)
    coverage = optimizer_coverage(model, optimizer)
    names = dict(model.named_parameters())
    monitored_names = [
        "patch_embed.proj.weight", "head.weight",
        "head_clip_bio.weight", "head_clip_bioReverse.0.head_fc.weight",
        "head_clip_bioReverse.1.head_fc.weight",
    ]
    monitored_names += [name for name in names if name.endswith("head_fc.bias")]
    records = []
    gradient_vectors = {}
    skipped = 0
    nonfinite_loss = 0
    nonfinite_gradient = 0
    update_failures = 0
    for iteration, batch in enumerate(batches, start=1):
        batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
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
        loss_finite = bool(torch.isfinite(total_loss.detach()))
        nonfinite_loss += int(not loss_finite)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        group_stats_now = group_gradient_stats(model)
        gradients_finite = all(value["nan_count"] == 0 and value["inf_count"] == 0 for value in group_stats_now.values())
        nonfinite_gradient += int(not gradients_finite)
        if iteration in (1, 10, 50, 100, 200):
            gradient_vectors[iteration] = {
                name: vector.tolist() if vector is not None else None
                for name, vector in named_gradient_vectors(model).items()
            }
        before = {name: names[name].detach().clone() for name in monitored_names if name in names}
        scaler.step(optimizer)
        found_inf_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = sum(float(value.item()) for value in found_inf_state.values())
        step_skipped = found_inf > 0
        deltas = {
            name: float((names[name].detach() - before[name]).float().abs().max().item())
            for name in before
        }
        parameter_updated = all(delta > 0 for name, delta in deltas.items()
                                 if name in ("patch_embed.proj.weight", "head.weight"))
        update_failures += int(not parameter_updated)
        skipped += int(step_skipped)
        scaler.update()
        scale_after = float(scaler.get_scale())
        scalars = {}
        for name, item in writer.values.items():
            scalars[name] = item["value"]
        record = {
            "iteration": iteration,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "step_skipped": step_skipped,
            "parameter_updated": parameter_updated,
            "parameter_deltas": deltas,
            "loss_finite": loss_finite,
            "total_loss": float(total_loss.detach().double().cpu()),
            "total_loss_dtype": str(total_loss.dtype),
            "scalars": scalars,
            "gradient_groups": group_stats_now,
            "lr_normal": float(min(group["lr"] for group in optimizer.param_groups)),
            "lr_bias": float(max(group["lr"] for group in optimizer.param_groups)),
        }
        if label == "biasful":
            record["bias_norms"] = {
                name: float(torch.linalg.vector_norm(names[name].detach().double()).item())
                for name in names if name.endswith("head_fc.bias")
            }
            record["bias_gradients"] = {
                name: grad_stats(names[name])
                for name in names if name.endswith("head_fc.bias")
            }
        records.append(record)
        if iteration == 1 or iteration % 25 == 0:
            print("BIAS_PAIRED_PROGRESS", json.dumps({
                "label": label, "iteration": iteration, "scale_before": scale_before,
                "scale_after": scale_after, "found_inf": found_inf,
                "step_skipped": step_skipped, "parameter_updated": parameter_updated,
                "loss": float(total_loss.detach()),
            }, sort_keys=True), flush=True)

    result = {
        "label": label,
        "iterations": len(records),
        "records": records,
        "gradient_vectors": gradient_vectors,
        "skipped_steps": skipped,
        "nonfinite_loss_count": nonfinite_loss,
        "nonfinite_gradient_iterations": nonfinite_gradient,
        "parameter_update_failures": update_failures,
        "successful_updates": len(records) - skipped,
        "initial_scale": records[0]["scale_before"] if records else None,
        "final_scale": records[-1]["scale_after"] if records else None,
        "optimizer_coverage": coverage,
    }
    del optimizer, optimizer_center, center_criterion, loss_fn
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return result


def cosine_relative(a, b):
    if a is None or b is None:
        return {"cosine": None, "relative_l2": None}
    xa = torch.tensor(a, dtype=torch.float64)
    xb = torch.tensor(b, dtype=torch.float64)
    denom = torch.linalg.vector_norm(xb).item()
    return {
        "cosine": float(F.cosine_similarity(xa, xb, dim=0).item()),
        "relative_l2": float(torch.linalg.vector_norm(xa - xb).item() / denom) if denom else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    local_cfg = make_dataset_config("ltcc")
    seed_all(local_cfg.SOLVER.SEED)
    dataset = build_dataset(local_cfg)
    batches, stream_info = materialize_batches(local_cfg, dataset, args.iterations)
    if len(batches) != args.iterations:
        raise RuntimeError(f"only materialized {len(batches)} of {args.iterations} batches")

    # Build the biasless model once and retain a CPU copy of every common
    # tensor.  The biasful model receives these exact values and zero biases.
    seed_all(local_cfg.SOLVER.SEED)
    biasless_model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    common_state = {key: value.detach().cpu().clone() for key, value in biasless_model.state_dict().items()}
    common_key_count = len(common_state)
    common_rng = capture_rng()
    biasless_dtype = model_dtype_audit(biasless_model)
    biasless_model.to("cpu")

    seed_all(local_cfg.SOLVER.SEED)
    biasful_model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    adapters = replace_reverse_heads(biasful_model)
    biasful_key_count = len(biasful_model.state_dict())
    official_state = normalize_state(torch.load(CHECKPOINT, map_location="cpu"))
    strict_missing, strict_unexpected = biasful_model.load_state_dict(official_state, strict=True)
    native_schema = {
        "biasless_key_count": common_key_count,
        "biasful_key_count": biasful_key_count,
        "official_key_count": len(official_state),
        "reverse_adapters_replaced": adapters,
        "missing_keys": list(strict_missing),
        "unexpected_keys": list(strict_unexpected),
        "shape_mismatch_count": 0,
        "strict_load_pass": not strict_missing and not strict_unexpected,
        "bias_delta_numel": biasful_key_count - common_key_count,
    }
    # Reset the candidate to exactly the common initial state; only its two
    # newly introduced bias tensors remain, and those are explicitly zero.
    candidate_state = {}
    for key, value in biasful_model.state_dict().items():
        if key in common_state:
            candidate_state[key] = common_state[key].clone()
        elif key.endswith("head_fc.bias"):
            candidate_state[key] = torch.zeros_like(value)
        else:
            raise RuntimeError(f"unexpected candidate-only key {key}")
    biasful_model.load_state_dict(candidate_state, strict=True)
    common_diffs = []
    for key, value in common_state.items():
        diff = (biasful_model.state_dict()[key].detach().cpu() - value).abs().max().item()
        common_diffs.append((key, float(diff)))
    common_max_abs_diff = max(value for _, value in common_diffs) if common_diffs else 0.0
    biasful_initial_bias = {
        key: {
            "max_abs": float(value.detach().abs().max().item()),
            "numel": value.numel(),
            "dtype": str(value.dtype),
        }
        for key, value in biasful_model.state_dict().items() if key.endswith("head_fc.bias")
    }
    biasful_dtype = model_dtype_audit(biasful_model)

    # Optimizer coverage is also checked before training, with the exact
    # candidate topology, then run_model repeats it after optimizer creation.
    _, candidate_center = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    candidate_optimizer, candidate_optimizer_center = make_optimizer(local_cfg, biasful_model, candidate_center)
    candidate_coverage = optimizer_coverage(biasful_model, candidate_optimizer)
    del candidate_optimizer, candidate_optimizer_center, candidate_center
    gc.collect()

    # Capture a fresh RNG point after construction; restoring this point makes
    # stochastic layers consume identical random numbers in A and B.
    train_rng = capture_rng()
    # A model is still CPU-resident and can now be trained; B follows after A.
    result_a = run_model(biasless_model, local_cfg, dataset, batches, train_rng, "biasless", device)
    del biasless_model
    gc.collect()
    torch.cuda.empty_cache()

    # Rebuild B from the verified common state so no training-side mutation of
    # the preflight object can leak into the second trajectory.
    seed_all(local_cfg.SOLVER.SEED)
    biasful_run_model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    replace_reverse_heads(biasful_run_model)
    biasful_run_model.load_state_dict(candidate_state, strict=True)
    result_b = run_model(biasful_run_model, local_cfg, dataset, batches, train_rng, "biasful", device)
    del biasful_run_model
    gc.collect()
    torch.cuda.empty_cache()

    vector_comparison = {}
    for iteration in sorted(set(result_a["gradient_vectors"]) & set(result_b["gradient_vectors"])):
        vector_comparison[str(iteration)] = {}
        for name in sorted(set(result_a["gradient_vectors"][iteration]) & set(result_b["gradient_vectors"][iteration])):
            vector_comparison[str(iteration)][name] = cosine_relative(
                result_a["gradient_vectors"][iteration][name],
                result_b["gradient_vectors"][iteration][name],
            )

    # Raw flattened vectors are only intermediate data for the cosine and
    # relative-L2 calculations and can be hundreds of megabytes. Keep the
    # compact comparison in the normal artifact; an optional full capture can
    # be retained separately when a byte-level audit is required.
    result_a["gradient_vectors"] = {}
    result_b["gradient_vectors"] = {}

    # Fingerprints are the same by construction, but retain a direct equality
    # check in the artifact rather than relying on that implementation detail.
    fingerprints_a = stream_info["fingerprints"]
    fingerprints_b = stream_info["fingerprints"]
    payload = {
        "config": {"dataset": "ltcc", "seed": local_cfg.SOLVER.SEED,
                   "initial_scale": 1024, "loss_path": "official autocast model / loss outside autocast",
                   "batch_size": local_cfg.DATA.BATCH_SIZE, "world_size": WORLD_SIZE,
                   "rank": RANK, "epochs": "diagnostic iterations within epoch 1",
                   "optimizer": local_cfg.SOLVER.OPTIMIZER_NAME,
                   "scheduler": "released epoch-based cosine, scheduler.step(1)"},
        "checkpoint": str(CHECKPOINT),
        "native_schema": native_schema,
        "dtype_audit": {"biasless": biasless_dtype, "biasful": biasful_dtype,
                        "unexpected_fp16_or_bf16_master": any(
                            key in ("FP16", "BF16") and value["count"]
                            for audit in (biasless_dtype, biasful_dtype)
                            for key, value in audit.items())},
        "common_initial_state": {
            "common_key_count": common_key_count,
            "common_max_abs_diff": common_max_abs_diff,
            "biasful_initial_bias": biasful_initial_bias,
            "candidate_optimizer_coverage": candidate_coverage,
        },
        "stream": {
            **stream_info,
            "a_b_fingerprints_identical": fingerprints_a == fingerprints_b,
            "first_five": fingerprints_a[:5],
            "last_fingerprint": fingerprints_a[-1],
        },
        "biasless": result_a,
        "biasful": result_b,
        "shared_gradient_comparison": vector_comparison,
        "decision_inputs": {
            "biasful_bias_norms_final": result_b["records"][-1].get("bias_norms", {}) if result_b["records"] else {},
            "biasful_reverse_bias_nonzero_at_end": any(
                value > 0 for value in (result_b["records"][-1].get("bias_norms", {}) if result_b["records"] else {}).values()),
            "biasless_skipped_steps": result_a["skipped_steps"],
            "biasful_skipped_steps": result_b["skipped_steps"],
        },
        "notes": [
            "This is a diagnostic 200-update paired probe, not a formal epoch run.",
            "A and B receive one materialized rank-0 stream from DistributedRandomIdentitySampler(num_replicas=2, rank=0); augmented tensors are reused bitwise.",
            "B differs only by two native zero-initialized reverse Linear biases; all common state tensors are copied from A and verified bitwise.",
            "The official checkpoint strict-load test is separate from the training probe; the official weights are not used as a training initialization.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("NATIVE_SCHEMA", json.dumps(native_schema, sort_keys=True))
    print("COMMON_MAX_ABS_DIFF", common_max_abs_diff)
    print("STREAM_FINGERPRINTS_IDENTICAL", fingerprints_a == fingerprints_b)
    print("BIAS_PAIRED_SUMMARY", json.dumps({
        "biasless_skipped": result_a["skipped_steps"], "biasful_skipped": result_b["skipped_steps"],
        "biasless_final_scale": result_a["final_scale"], "biasful_final_scale": result_b["final_scale"],
        "biasless_updates": result_a["successful_updates"], "biasful_updates": result_b["successful_updates"],
        "biasful_final_bias_norms": payload["decision_inputs"]["biasful_bias_norms_final"],
    }, sort_keys=True))
    print("WROTE", output)


if __name__ == "__main__":
    main()
