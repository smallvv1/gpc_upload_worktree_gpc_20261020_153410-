#!/usr/local/bin/python3
"""Phase 4R-A2 clean-control reconstruction.

This script is deliberately limited to:

* one full-topology COMMON_INIT construction;
* a paired one-step equivalence test; and
* at most a 200-step B/H probe.

It does not run an epoch training job.  Control B keeps the reverse branch in
the forward/logging path but backpropagates only CE+Triplet+Bio.  Control H
uses the same full topology and detached reverse input, with separate shared
and reverse optimizers/scalers so reverse overflow cannot skip a shared step.
"""

from __future__ import annotations

import argparse
import csv
import copy
import gc
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataloader  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402


SEED = 1234
INITIAL_SCALE = 1024
FORMAL_MAX_EPOCHS = 60
MAX_STEPS = 200
SHORT_GLOBAL_STEPS = 40
WORLD_SIZE = 2


class ScalarCapture:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self.values.setdefault(str(tag), []).append(float(scalar_value))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(type(value).__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_full() -> Any:
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.BATCH_SIZE = 8
    cfg.DATA.NUM_INSTANCES = 2
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.MAX_EPOCHS = FORMAL_MAX_EPOCHS
    cfg.freeze()
    return cfg


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {key: sha256_tensor(value) for key, value in state.items()}


def common_init(output_dir: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("clean-control probe requires the two visible PPU devices; CPU fallback is forbidden")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "COMMON_INIT.pth"
    hashes_path = output_dir / "COMMON_INIT_HASHES.json"
    if path.exists() and hashes_path.exists():
        print(json.dumps({"common_init": str(path), "reused": True}, sort_keys=True), flush=True)
        return

    torch.cuda.set_device(0)
    set_seed(SEED)
    cfg = configure_full()
    model = build_model(cfg, num_classes=77, camera_num=12)
    model.eval()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if len(state) != 517:
        raise RuntimeError(f"expected full topology state key count 517, got {len(state)}")
    torch.save(state, path)
    hashes = state_hashes(state)
    hashes_path.write_text(json.dumps({
        "created_utc": utc_now(),
        "seed": SEED,
        "topology": "full: ID + Bio + Hair reverse + Clothing reverse",
        "detach_reverse_input_at_construction": False,
        "state_key_count": len(state),
        "state_numel": int(sum(t.numel() for t in state.values())),
        "hashes": hashes,
    }, indent=2, sort_keys=True) + "\n")
    (output_dir / "COMMON_INIT_CONFIG.yml").write_text(cfg.dump())
    print(json.dumps({
        "common_init": str(path),
        "reused": False,
        "state_key_count": len(state),
        "state_numel": int(sum(t.numel() for t in state.values())),
    }, sort_keys=True), flush=True)
    del model, state
    gc.collect()
    torch.cuda.empty_cache()


def init_process_group() -> None:
    if dist.is_initialized():
        return
    # Run under torchrun so the same sampler semantics as the formal two-rank
    # path are retained.  Gloo is sufficient here because these controls are
    # intentionally not wrapped in DDP; it is not a training replacement.
    dist.init_process_group(backend="gloo", init_method="env://")


def make_param_optimizers(model: torch.nn.Module):
    shared = []
    reverse = []
    shared_names = []
    reverse_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head_clip_bioReverse."):
            reverse.append({
                "params": [parameter],
                "lr": 2e-6,
                "weight_decay": 0.05,
            })
            reverse_names.append(name)
        else:
            lr = 4e-6 if "bias" in name else 2e-6
            shared.append({"params": [parameter], "lr": lr, "weight_decay": 0.05})
            shared_names.append(name)
    shared_opt = torch.optim.SGD(shared, momentum=0.9)
    reverse_opt = torch.optim.SGD(reverse, momentum=0.9)
    return shared_opt, reverse_opt, shared_names, reverse_names


def finite_grad(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())


def grad_summary(model: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    params = dict(model.named_parameters())
    exists = 0
    nonzero = 0
    finite = 0
    max_abs = 0.0
    l2_sq = 0.0
    for name in names:
        grad = params[name].grad
        if grad is None:
            continue
        exists += 1
        if torch.isfinite(grad).all():
            finite += 1
        if torch.count_nonzero(grad).item() > 0:
            nonzero += 1
        max_abs = max(max_abs, float(grad.detach().abs().max().item()))
        l2_sq += float(torch.sum(grad.detach().float() * grad.detach().float()).item())
    return {
        "parameter_count": len(names),
        "grad_exists_count": exists,
        "finite_count": finite,
        "nonzero_count": nonzero,
        "max_abs": max_abs,
        "l2": l2_sq ** 0.5,
    }


def optimizer_found_inf(scaler: amp.GradScaler, optimizer: torch.optim.Optimizer) -> float:
    state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    return sum(float(value.item()) for value in state.values())


def scaler_state(scaler: amp.GradScaler) -> float:
    return float(scaler.get_scale())


def state_distance(model_a: torch.nn.Module, model_b: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    max_abs = 0.0
    max_rel = 0.0
    exact = True
    allclose = True
    mismatch_tensors = 0
    for name in names:
        av = a[name].detach()
        bv = b[name].detach()
        diff = (av - bv).abs()
        local_max = float(diff.max().item())
        denom = torch.maximum(av.abs(), bv.abs()).clamp_min(1e-12)
        local_rel = float((diff / denom).max().item())
        max_abs = max(max_abs, local_max)
        max_rel = max(max_rel, local_rel)
        is_exact = bool(torch.equal(av, bv))
        is_close = bool(torch.allclose(av, bv, rtol=1e-4, atol=1e-6))
        exact = exact and is_exact
        allclose = allclose and is_close
        mismatch_tensors += int(not is_exact)
    return {
        "parameter_count": len(names),
        "exact": exact,
        "allclose_rtol_1e-4_atol_1e-6": allclose,
        "mismatch_tensor_count": mismatch_tensors,
        "max_abs": max_abs,
        "max_relative": max_rel,
    }


def grad_distance(model_a: torch.nn.Module, model_b: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    max_abs = 0.0
    max_rel = 0.0
    exact = True
    allclose = True
    mismatch_tensors = 0
    missing = 0
    for name in names:
        ga = a[name].grad
        gb = b[name].grad
        if ga is None or gb is None:
            missing += 1
            exact = False
            allclose = False
            continue
        diff = (ga - gb).abs()
        local_max = float(diff.max().item())
        denom = torch.maximum(ga.abs(), gb.abs()).clamp_min(1e-12)
        local_rel = float((diff / denom).max().item())
        max_abs = max(max_abs, local_max)
        max_rel = max(max_rel, local_rel)
        is_exact = bool(torch.equal(ga, gb))
        is_close = bool(torch.allclose(ga, gb, rtol=1e-4, atol=1e-6))
        exact = exact and is_exact
        allclose = allclose and is_close
        mismatch_tensors += int(not is_exact)
    return {
        "parameter_count": len(names),
        "missing_gradient_tensors": missing,
        "exact": exact,
        "allclose_rtol_1e-4_atol_1e-6": allclose,
        "mismatch_tensor_count": mismatch_tensors,
        "max_abs": max_abs,
        "max_relative": max_rel,
    }


def batch_fingerprint(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(batch["image"].detach().contiguous().cpu().numpy().tobytes())
    for key in ["image_path", "pid", "camid", "clothes_id"]:
        value = batch[key]
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(list(value), sort_keys=True, default=str).encode())
    return digest.hexdigest()


def ids_and_accuracy(score: dict[str, Any], batch: dict[str, Any], device: torch.device) -> dict[str, float]:
    targets = batch["pid"].to(device)
    out = {"id_acc": float((score["cls_score"].argmax(-1) == targets).float().mean().item())}
    if "clip_bio_score" in score:
        bio = score["clip_bio_score"]
        labels = torch.arange(bio.shape[0], device=device)
        out["bio_acc"] = float((bio.argmax(-1) == labels).float().mean().item())
    if "clip_bio_reverse_score" in score:
        reverse = score["clip_bio_reverse_score"]
        labels = torch.arange(reverse.shape[0], device=device)
        for index in range(reverse.shape[1]):
            out[f"reverse_{index}_acc"] = float((reverse[:, index].argmax(-1) == labels).float().mean().item())
    return out


def make_core_and_reverse(cfg, model, batch, loss_fn, writer, device, step):
    samples = batch["image"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    targets = batch["pid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    caption = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(samples, cam_label=camids)
    _total = loss_fn(
        score,
        feat,
        targets,
        camids,
        caption,
        clothes_ids=clothes,
        train_writer=writer,
        step=step,
    )
    components = dict(loss_fn.last_components)
    core = components["core"]
    reverse = components.get("reverse_total")
    return score, core, reverse, ids_and_accuracy(score, batch, device)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture_gradients(model: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {
        name: parameters[name].grad.detach().clone()
        for name in names
        if parameters[name].grad is not None
    }


def compare_gradient_dicts(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = sorted(set(first) & set(second))
    missing = len(set(first) ^ set(second))
    max_abs = 0.0
    max_rel = 0.0
    exact = missing == 0
    allclose = missing == 0
    mismatch = 0
    for name in names:
        diff = (first[name] - second[name]).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = torch.maximum(first[name].abs(), second[name].abs()).clamp_min(1e-12)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        current_exact = bool(torch.equal(first[name], second[name]))
        current_close = bool(torch.allclose(first[name], second[name], rtol=1e-4, atol=1e-6))
        exact = exact and current_exact
        allclose = allclose and current_close
        mismatch += int(not current_exact)
    return {
        "parameter_count": len(names),
        "missing_gradient_tensors": missing,
        "exact": exact,
        "allclose_rtol_1e-4_atol_1e-6": allclose,
        "mismatch_tensor_count": mismatch,
        "max_abs": max_abs,
        "max_relative": max_rel,
    }


def repeated_core_backward_probe(output_dir: Path, common_init_path: Path, local_rank: int) -> None:
    """One controlled B->B repetition to quantify PPU backward nondeterminism."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("repetition probe requires exactly two visible PPUs; CPU fallback is forbidden")
    init_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank_dir = output_dir / f"rank{local_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure_full()
    set_seed(SEED)
    train_loader, _query, _gallery, dataset, sampler, _val, _original = build_dataloader(cfg)
    if len(train_loader) != 596:
        raise RuntimeError(f"expected formal world-size-2 loader length 596, got {len(train_loader)}")
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    batch = next(iter(train_loader))

    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    common_state = torch.load(common_init_path, map_location="cpu", weights_only=True)
    model.load_state_dict(common_state, strict=True)
    model.to(device)
    model.train()
    optimizer, _reverse_unused, names, _reverse_names = make_param_optimizers(model)
    loss_fn, _center = make_loss(cfg, num_classes=dataset.num_train_pids)
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device)
    gradient_copies: list[dict[str, torch.Tensor]] = []
    losses: list[float] = []
    for repetition in range(2):
        model.load_state_dict(common_state, strict=True)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        writer = ScalarCapture()
        _score, core, _reverse, _acc = make_core_and_reverse(
            cfg, model, batch, loss_fn, writer, device, repetition + 1)
        scaler.scale(core).backward()
        scaler.unscale_(optimizer)
        gradient_copies.append(capture_gradients(model, names))
        losses.append(float(core.detach().float().cpu()))
    comparison = compare_gradient_dicts(gradient_copies[0], gradient_copies[1])
    result = {
        "rank": local_rank,
        "device": torch.cuda.get_device_name(local_rank),
        "batch_fingerprint": batch_fingerprint(batch),
        "core_losses": losses,
        "core_loss_allclose": bool(abs(losses[0] - losses[1]) <= 1e-6),
        "same_model_same_batch_gradient_repetition": comparison,
    }
    (rank_dir / "repeated_core_backward.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, result)
    if local_rank == 0:
        (output_dir / "repeated_core_backward.json").write_text(json.dumps({
            "phase": "4R-A2",
            "probe": "B_to_B_repeated_core_backward",
            "created_utc": utc_now(),
            "rank_results": [item for item in gathered if item is not None],
        }, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if hasattr(train_loader, "shutdown"):
        try:
            train_loader.shutdown()
        except Exception:
            pass
    del model, common_state
    gc.collect()
    torch.cuda.empty_cache()


def family_finite_audit(model: torch.nn.Module) -> dict[str, Any]:
    families = {
        "backbone": [],
        "id_head": [],
        "bio_head": [],
        "reverse_hair": [],
        "reverse_clothing": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("head_clip_bioReverse.0."):
            family = "reverse_hair"
        elif name.startswith("head_clip_bioReverse.1."):
            family = "reverse_clothing"
        elif name.startswith("head_clip_bio.") or name.startswith("head_bio.") or name.startswith("norm_bio."):
            family = "bio_head"
        elif name.startswith("head."):
            family = "id_head"
        else:
            family = "backbone"
        families[family].append((name, parameter))
    report: dict[str, Any] = {}
    for family, items in families.items():
        present = [name for name, parameter in items if parameter.grad is not None]
        nonfinite = [
            name for name, parameter in items
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        ]
        report[family] = {
            "parameter_count": len(items),
            "grad_exists_count": len(present),
            "nonfinite_count": len(nonfinite),
            "nonfinite_names": nonfinite,
        }
    return report


def global_a2_short_probe(output_dir: Path, common_init_path: Path, local_rank: int) -> None:
    """Short A2-style global-optimizer probe to locate skip sources."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("global A2 probe requires exactly two visible PPUs; CPU fallback is forbidden")
    init_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank_dir = output_dir / f"rank{local_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure_full()
    cfg.defrost()
    cfg.MODEL.DETACH_REVERSE_INPUT = True
    cfg.freeze()
    set_seed(SEED)
    train_loader, _query, _gallery, dataset, sampler, _val, _original = build_dataloader(cfg)
    if len(train_loader) != 596:
        raise RuntimeError(f"expected formal world-size-2 loader length 596, got {len(train_loader)}")
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    common_state = torch.load(common_init_path, map_location="cpu", weights_only=True)
    model.load_state_dict(common_state, strict=True)
    model.to(device)
    model.train()
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _center_optimizer = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(1)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    rows: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    try:
        for step in range(1, SHORT_GLOBAL_STEPS + 1):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            writer = ScalarCapture()
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
            total = loss_fn(
                score,
                feat,
                batch["pid"].to(device),
                batch["camid"].to(device),
                batch["caption_feature"].to(device),
                clothes_ids=batch["clothes_id"].to(device),
                train_writer=writer,
                step=step,
            )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            families = family_finite_audit(model)
            scaler.step(optimizer)
            found_inf = optimizer_found_inf(scaler, optimizer)
            before_update = scaler_state(scaler)
            scaler.update()
            after_update = scaler_state(scaler)
            row = {
                "step": step,
                "batch_fingerprint": batch_fingerprint(batch),
                "total_loss": float(total.detach().float().cpu()),
                "total_finite": bool(torch.isfinite(total.detach()).item()),
                "scaler_before": before_update,
                "scaler_after": after_update,
                "global_found_inf": bool(found_inf > 0),
                "family_audit": families,
            }
            rows.append(row)
            if found_inf > 0 or step in (1, SHORT_GLOBAL_STEPS):
                print(json.dumps({"rank": local_rank, **row}, sort_keys=True), flush=True)
    finally:
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass
    (rank_dir / "global_a2_short.json").write_text(json.dumps({
        "rank": local_rank,
        "steps": rows,
        "global_skip_steps": [row["step"] for row in rows if row["global_found_inf"]],
    }, indent=2, sort_keys=True) + "\n")
    result = {
        "rank": local_rank,
        "device": torch.cuda.get_device_name(local_rank),
        "steps_completed": len(rows),
        "global_skip_steps": [row["step"] for row in rows if row["global_found_inf"]],
        "family_nonfinite_by_skip": {
            str(row["step"]): {
                family: values["nonfinite_count"]
                for family, values in row["family_audit"].items()
            }
            for row in rows if row["global_found_inf"]
        },
        "final_scale": rows[-1]["scaler_after"] if rows else None,
    }
    (rank_dir / "global_a2_short_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, result)
    if local_rank == 0:
        (output_dir / "global_a2_short_summary.json").write_text(json.dumps({
            "phase": "4R-A2",
            "probe": "A2_global_optimizer_short_source_audit",
            "created_utc": utc_now(),
            "steps": SHORT_GLOBAL_STEPS,
            "rank_summaries": [item for item in gathered if item is not None],
        }, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    del model, common_state
    gc.collect()
    torch.cuda.empty_cache()


def paired_probe(output_dir: Path, common_init_path: Path, local_rank: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("paired probe requires exactly two visible PPUs; CPU fallback is forbidden")
    init_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank_dir = output_dir / f"rank{local_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure_full()
    set_seed(SEED)

    train_loader, _query, _gallery, dataset, sampler, _val, _original = build_dataloader(cfg)
    if len(train_loader) != 596:
        raise RuntimeError(f"expected formal world-size-2 loader length 596, got {len(train_loader)}")
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    # Build once, clone once, then load exactly the same COMMON_INIT into both
    # controls.  Only the detach flag differs after cloning.
    model_b = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    common_state = torch.load(common_init_path, map_location="cpu", weights_only=True)
    model_b.load_state_dict(common_state, strict=True)
    model_h = copy.deepcopy(model_b)
    model_h.detach_reverse_input = True
    model_b.to(device)
    model_h.to(device)
    model_b.train()
    model_h.train()

    shared_b, _reverse_b_unused, names_b, reverse_names_b = make_param_optimizers(model_b)
    shared_h, reverse_h, names_h, reverse_names_h = make_param_optimizers(model_h)
    if names_b != names_h or reverse_names_b != reverse_names_h:
        raise RuntimeError("B/H optimizer parameter topology differs")
    if reverse_names_b != ["head_clip_bioReverse.0.head_fc.weight", "head_clip_bioReverse.1.head_fc.weight"]:
        raise RuntimeError(f"unexpected reverse parameter names: {reverse_names_b}")

    # Use the same formal 60-epoch scheduler phase for both shared controls;
    # the probe is only truncated in steps, not by changing t_initial.
    scheduler_b = create_scheduler(cfg, shared_b)
    scheduler_h = create_scheduler(cfg, shared_h)
    scheduler_b.step(1)
    scheduler_h.step(1)

    scaler_b = amp.GradScaler(init_scale=INITIAL_SCALE)
    scaler_h_shared = amp.GradScaler(init_scale=INITIAL_SCALE)
    scaler_h_reverse = amp.GradScaler(init_scale=INITIAL_SCALE)
    loss_b, _center_b = make_loss(cfg, num_classes=dataset.num_train_pids)
    loss_h, _center_h = make_loss(cfg, num_classes=dataset.num_train_pids)

    rows: list[dict[str, Any]] = []
    one_step: dict[str, Any] | None = None
    iterator = iter(train_loader)
    try:
        for step in range(1, MAX_STEPS + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            fp = batch_fingerprint(batch)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)

            # B: core loss only; reverse forward/loss is computed but never
            # enters backward, so reverse heads do not update.
            shared_b.zero_grad(set_to_none=True)
            writer_b = ScalarCapture()
            score_b, core_b, reverse_b, acc_b = make_core_and_reverse(
                cfg, model_b, batch, loss_b, writer_b, device, step)
            core_b_finite = bool(torch.isfinite(core_b.detach()).item())
            reverse_b_finite = reverse_b is None or bool(torch.isfinite(reverse_b.detach()).item())
            scaler_b.scale(core_b).backward()
            scaler_b.unscale_(shared_b)
            grad_b = grad_summary(model_b, names_b)
            scaler_b.step(shared_b)
            found_b = optimizer_found_inf(scaler_b, shared_b)
            scaler_b.update()
            torch.cuda.synchronize(device)

            # H sees exactly the same batch and RNG state.  Its reverse path is
            # detached and isolated in a second optimizer/scaler.
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            shared_h.zero_grad(set_to_none=True)
            reverse_h.zero_grad(set_to_none=True)
            writer_h = ScalarCapture()
            score_h, core_h, reverse_h_loss, acc_h = make_core_and_reverse(
                cfg, model_h, batch, loss_h, writer_h, device, step)
            core_h_finite = bool(torch.isfinite(core_h.detach()).item())
            reverse_h_finite = reverse_h_loss is not None and bool(torch.isfinite(reverse_h_loss.detach()).item())
            scaler_h_shared.scale(core_h).backward()
            scaler_h_shared.unscale_(shared_h)
            grad_h = grad_summary(model_h, names_h)
            grad_compare = grad_distance(model_b, model_h, names_b)
            scaler_h_shared.step(shared_h)
            found_h_shared = optimizer_found_inf(scaler_h_shared, shared_h)
            scaler_h_shared.update()

            if reverse_h_loss is not None:
                scaler_h_reverse.scale(reverse_h_loss).backward()
                scaler_h_reverse.unscale_(reverse_h)
                reverse_grad_h = grad_summary(model_h, reverse_names_h)
                scaler_h_reverse.step(reverse_h)
                found_h_reverse = optimizer_found_inf(scaler_h_reverse, reverse_h)
                scaler_h_reverse.update()
            else:
                reverse_grad_h = {"parameter_count": len(reverse_names_h), "grad_exists_count": 0,
                                  "finite_count": 0, "nonzero_count": 0, "max_abs": 0.0, "l2": 0.0}
                found_h_reverse = 0.0
            torch.cuda.synchronize(device)

            distance = state_distance(model_b, model_h, names_b)
            if step == 1:
                one_step = {
                    "batch_fingerprint": fp,
                    "core_loss_b": float(core_b.detach().float().cpu()),
                    "core_loss_h": float(core_h.detach().float().cpu()),
                    "reverse_loss_b": None if reverse_b is None else float(reverse_b.detach().float().cpu()),
                    "reverse_loss_h": None if reverse_h_loss is None else float(reverse_h_loss.detach().float().cpu()),
                    "core_loss_allclose": bool(torch.allclose(core_b.detach(), core_h.detach(), rtol=1e-4, atol=1e-6)),
                    "shared_gradient_compare_before_step": grad_compare,
                    "shared_parameter_compare_after_step": distance,
                    "reverse_head_grad_b": grad_summary(model_b, reverse_names_b),
                    "reverse_head_grad_h": reverse_grad_h,
                    "b_shared_skip": bool(found_b > 0),
                    "h_shared_skip": bool(found_h_shared > 0),
                    "h_reverse_skip": bool(found_h_reverse > 0),
                }
                one_step_pass = bool(
                    one_step["core_loss_allclose"]
                    and grad_compare["allclose_rtol_1e-4_atol_1e-6"]
                    and distance["allclose_rtol_1e-4_atol_1e-6"]
                    and one_step["reverse_head_grad_b"]["grad_exists_count"] == 0
                    and one_step["reverse_head_grad_h"]["grad_exists_count"] > 0
                )
                one_step["passed"] = one_step_pass
                if not one_step_pass:
                    break

            row = {
                "step": step,
                "batch_fingerprint": fp,
                "data_same_for_B_H": True,
                "core_loss_B": float(core_b.detach().float().cpu()),
                "core_loss_H": float(core_h.detach().float().cpu()),
                "reverse_loss_B": None if reverse_b is None else float(reverse_b.detach().float().cpu()),
                "reverse_loss_H": None if reverse_h_loss is None else float(reverse_h_loss.detach().float().cpu()),
                "core_finite_B": core_b_finite,
                "core_finite_H": core_h_finite,
                "reverse_finite_B": reverse_b_finite,
                "reverse_finite_H": reverse_h_finite,
                "B_shared_scale": scaler_state(scaler_b),
                "H_shared_scale": scaler_state(scaler_h_shared),
                "H_reverse_scale": scaler_state(scaler_h_reverse),
                "B_shared_skip": bool(found_b > 0),
                "H_shared_skip": bool(found_h_shared > 0),
                "H_reverse_skip": bool(found_h_reverse > 0),
                "B_core_grad_nonzero": grad_b["nonzero_count"],
                "H_core_grad_nonzero": grad_h["nonzero_count"],
                "H_reverse_grad_nonzero": reverse_grad_h["nonzero_count"],
                "shared_max_abs_distance": distance["max_abs"],
                "shared_max_relative_distance": distance["max_relative"],
                "shared_allclose": distance["allclose_rtol_1e-4_atol_1e-6"],
                "B_id_acc": acc_b["id_acc"],
                "H_id_acc": acc_h["id_acc"],
            }
            rows.append(row)
            if step == 1 or step % 10 == 0:
                print(json.dumps({"rank": local_rank, **row}, sort_keys=True), flush=True)
    finally:
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass

    write_csv(rank_dir / "paired_probe_metrics.csv", rows)
    rank_summary = {
        "rank": local_rank,
        "device": torch.cuda.get_device_name(local_rank),
        "steps_completed": len(rows),
        "one_step": one_step,
        "final_row": rows[-1] if rows else None,
        "shared_optimizer_parameter_count": len(names_b),
        "reverse_optimizer_parameter_count": len(reverse_names_b),
        "shared_optimizer_lr_after_scheduler_step": [float(x["lr"]) for x in shared_b.param_groups[:2]],
        "reverse_optimizer_lr": [float(x["lr"]) for x in reverse_h.param_groups],
        "batch_fingerprints_unique": len({row["batch_fingerprint"] for row in rows}),
        "all_data_same_for_B_H": all(row["data_same_for_B_H"] for row in rows),
        "same_shared_trajectory_at_final": bool(rows and rows[-1]["shared_allclose"]),
    }
    (rank_dir / "paired_probe_summary.json").write_text(json.dumps(rank_summary, indent=2, sort_keys=True, default=json_default) + "\n")
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, rank_summary)
    if local_rank == 0:
        (output_dir / "paired_probe_summary.json").write_text(json.dumps({
            "phase": "4R-A2",
            "probe": "clean_controls_B_and_H",
            "created_utc": utc_now(),
            "seed": SEED,
            "max_steps": MAX_STEPS,
            "scheduler_t_initial": FORMAL_MAX_EPOCHS,
            "common_init": str(common_init_path),
            "control_B": "full topology; core loss only; reverse logits/loss logged but not backpropagated",
            "control_H": "full topology; detached reverse input; shared and reverse optimizers/scalers isolated",
            "rank_summaries": [item for item in gathered if item is not None],
            "one_step_all_pass": all(item and item.get("one_step", {}).get("passed", False) for item in gathered),
            "two_hundred_step_probe_completed": all(item and item.get("steps_completed", 0) >= MAX_STEPS for item in gathered),
        }, indent=2, sort_keys=True, default=json_default) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    del model_b, model_h, common_state
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["common-init", "probe", "repeat-core", "global-a2-short"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--common-init", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.mode == "common-init":
        common_init(output_dir)
        return
    common_path = Path(args.common_init).resolve() if args.common_init else output_dir / "COMMON_INIT.pth"
    if not common_path.exists():
        raise FileNotFoundError(common_path)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.mode == "repeat-core":
        repeated_core_backward_probe(output_dir, common_path, local_rank)
    elif args.mode == "global-a2-short":
        global_a2_short_probe(output_dir, common_path, local_rank)
    else:
        paired_probe(output_dir, common_path, local_rank)


if __name__ == "__main__":
    main()
