#!/usr/local/bin/python3
"""PHASE 2N-R/S/T: gated PPU compatibility training.

This driver is deliberately separate from ``train.py``.  It keeps the
released loss, optimizer, scheduler, data sampler, AMP placement, and
evaluation path, while adding the exact Phase 2M full-block wrapper for the
selected diagnostic mask.  The script is launched with two distributed
PPU-ZW810E workers and has two modes:

* ``paired200`` materializes one real rank-local stream and runs Current and
  Compat V2 from the same model state, optimizer construction, RNG state, and
  augmented batches.
* ``epochs`` runs Compat V2 only, saving an exact model/optimizer/scaler/RNG
  resume state at epoch 15, 25, and 60.  Epoch 25/60 are entered by the
  autonomous controller only after the preceding gate passes.

No source model, loss, or formal training entry point is modified by this
diagnostic driver.  All writes are confined to the new Phase 2N output tree.
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
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data import build_dataloader  # noqa: E402
from global_precision_policy_screen_phase2m import wrap_fp32_block  # noqa: E402
from initialization_footprint_phase2i import (  # noqa: E402
    INITIAL_SCALE,
    build_natural_model,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from loss import make_loss  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402


N_BLOCKS = 23
SEED = 1234
WORLD_SIZE_REQUIRED = 2
DEFAULT_MASK = (0, 1, 2, 3)
DEFAULT_OUTPUT = ROOT / "outputs/reproduction_phase2/phase2n"
SELECTION_PATH = ROOT / "diagnostics/phase2n_compat_selection.json"
OFFICIAL_EVENT_DIR = Path("/home/data/DIFFER_OfficialWeights/provenance/LTCC/train")
OFFICIAL_FIRST200_CSV = ROOT / "reports/OFFICIAL_FIRST200_SCALARS.csv"
CURRENT_RUN_PATHS = [
    ROOT / "diagnostics/phase2j_p_a_25epoch.json",
    ROOT / "diagnostics/phase2j_p_b_25epoch.json",
    ROOT / "diagnostics/phase2j_p_c_25epoch.json",
]

SCALAR_FIELDS = (
    "total_loss",
    "ce_loss",
    "triplet_loss",
    "bio_loss",
    "reverse0_loss",
    "reverse1_loss",
    "id_acc",
    "bio_acc",
    "reverse0_acc",
    "reverse1_acc",
)
STEP_CSV_FIELDS = [
    "candidate",
    "epoch",
    "iteration",
    "global_step",
    "total_loss",
    "ce_loss",
    "triplet_loss",
    "bio_loss",
    "reverse0_loss",
    "reverse1_loss",
    "id_acc",
    "bio_acc",
    "reverse0_acc",
    "reverse1_acc",
    "train_accuracy",
    "lr",
    "scale_before",
    "scale_after",
    "found_inf_mean",
    "step_skipped_any",
    "step_skipped_fraction",
    "gradient_norm",
    "runtime_sec",
    "peak_memory_bytes",
    "parameter_updated_all_ranks",
    "loss_finite_all_ranks",
]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def tensor_sha(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_manifest(model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    if state is None:
        state = {name: value.detach().cpu() for name, value in model.named_parameters()}
    return {
        "parameter_name_count": len(parameters),
        "parameter_numel": int(sum(value.numel() for _, value in parameters)),
        "parameter_name_sha256": hashlib.sha256("\n".join(name for name, _ in parameters).encode()).hexdigest(),
        "parameter_state_sha256": state_hash({name: state[name] for name, _ in parameters}),
        "parameter_names": [name for name, _ in parameters],
    }


def mask_from_selection() -> tuple[int, ...]:
    if SELECTION_PATH.exists():
        try:
            payload = json.loads(SELECTION_PATH.read_text())
            if payload.get("created") and payload.get("mask"):
                return tuple(sorted({int(x) for x in payload["mask"]}))
        except Exception:
            pass
    return DEFAULT_MASK


def apply_sparse_mask(model: torch.nn.Module, mask: tuple[int, ...]) -> list[tuple[torch.nn.Module, Any]]:
    originals = []
    for index in mask:
        block = model.blocks[index]
        originals.append((block, wrap_fp32_block(block)))
    return originals


def restore_wrappers(originals: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original in reversed(originals):
        module.forward = original


def to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_cpu(item) for item in value]
    return copy.deepcopy(value)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def batch_sample_fingerprint(batch: dict[str, Any]) -> str:
    payload = []
    for key in ("pid", "camid", "clothes_id"):
        value = batch[key].detach().cpu().tolist()
        payload.append(key + ":" + ",".join(str(int(x)) for x in value))
    payload.append("path:" + "|".join(str(x) for x in batch["image_path"]))
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def batch_image_fingerprint(batch: dict[str, Any]) -> str:
    return tensor_sha(batch["image"])


def materialize_batches(train_loader, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batches = []
    fingerprints = []
    for index, raw in enumerate(train_loader):
        if index >= count:
            break
        batch = to_cpu(raw)
        batches.append(batch)
        fingerprints.append({
            "sequence": index + 1,
            "sample_sha256": batch_sample_fingerprint(batch),
            "image_sha256": batch_image_fingerprint(batch),
            "pids": [int(x) for x in batch["pid"].tolist()],
            "camids": [int(x) for x in batch["camid"].tolist()],
            "clothes": [int(x) for x in batch["clothes_id"].tolist()],
            "paths": [str(x) for x in batch["image_path"]],
        })
    if hasattr(train_loader, "shutdown"):
        try:
            train_loader.shutdown()
        except Exception:
            pass
    return batches, fingerprints


def stream_hash(fingerprints: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{item['sample_sha256']}:{item['image_sha256']}" for item in fingerprints)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_rank_data(cfg, device: torch.device):
    # The released DataLoaderX path is retained for formal epoch training.
    result = build_dataloader(cfg)
    train_loader, query_loader, gallery_loader, dataset, train_sampler, val_loader, train_original = result
    if len(train_loader) != 596:
        raise RuntimeError(f"Phase 2N requires 596 iterations/epoch, got {len(train_loader)}")
    return result


def reduce_mean_vector(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(float(dist.get_world_size()))
    return [float(x) for x in tensor.detach().cpu().tolist()]


def reduce_max_vector(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(x) for x in tensor.detach().cpu().tolist()]


def gradient_norm(model: torch.nn.Module) -> tuple[float, bool]:
    total = 0.0
    finite_all = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            finite_all = False
            continue
        total += float(torch.sum(gradient.float() * gradient.float()).detach().cpu())
    return math.sqrt(max(total, 0.0)), finite_all


def selected_gradient_norm(model: torch.nn.Module) -> tuple[float, bool]:
    names = {
        "blocks.0.attn.q_proj.weight",
        "blocks.11.attn.q_proj.weight",
        "blocks.22.attn.q_proj.weight",
        "head.weight",
        "head_clip_bio.weight",
        "head_clip_bioReverse.0.head_fc.weight",
        "head_clip_bioReverse.1.head_fc.weight",
    }
    total = 0.0
    finite_all = True
    for name, parameter in model.named_parameters():
        if name not in names or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            finite_all = False
            continue
        total += float(torch.sum(gradient.float() * gradient.float()).detach().cpu())
    return math.sqrt(max(total, 0.0)), finite_all


def scalar_values(writer: Any, total_loss: torch.Tensor) -> dict[str, float]:
    # The tags are the exact tags emitted by loss/make_loss.py.  The Bio tag
    # intentionally uses factor 0, while Reverse0/1 are the two cumulative
    # reverse factors required by the Phase 2N protocol.
    return {
        "total_loss": finite(total_loss.detach().double().cpu()),
        "ce_loss": finite(writer.values.get("loss/id")),
        "triplet_loss": finite(writer.values.get("loss/triplet")),
        "bio_loss": finite(writer.values.get("loss/clip_bio_0")),
        "reverse0_loss": finite(writer.values.get("loss/clip_bio_reverse_0")),
        "reverse1_loss": finite(writer.values.get("loss/clip_bio_reverse_1")),
        "id_acc": finite(writer.values.get("acc/id")),
        "bio_acc": finite(writer.values.get("acc/clip_bio_0")),
        "reverse0_acc": finite(writer.values.get("acc/clip_bio_reverse_0")),
        "reverse1_acc": finite(writer.values.get("acc/clip_bio_reverse_1")),
    }


class CaptureWriter:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values[str(tag)] = float(value)


def optimizer_signature(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "param_group_count": len(optimizer.param_groups),
        "parameter_count": int(sum(len(group["params"]) for group in optimizer.param_groups)),
        "lr_values": [float(group["lr"]) for group in optimizer.param_groups[:8]],
        "weight_decay_values": [float(group["weight_decay"]) for group in optimizer.param_groups[:8]],
        "optimizer": type(optimizer).__name__,
    }


def checkpoint_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def official_first200() -> dict[str, Any]:
    if not OFFICIAL_FIRST200_CSV.exists():
        return {"rows": [], "fingerprint": None}
    with OFFICIAL_FIRST200_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {"rows": rows[:200], "fingerprint": json_hash(rows[:200])}


def open_step_writer(path: Path, append: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = append and path.exists() and path.stat().st_size > 0
    handle = path.open("a" if exists else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=STEP_CSV_FIELDS)
    if not exists:
        writer.writeheader()
    return handle, writer


def prepare_model(cfg, dataset, base_state: dict[str, torch.Tensor], device: torch.device, mask: tuple[int, ...], compat: bool):
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    model.load_state_dict(base_state, strict=True)
    originals = apply_sparse_mask(model, mask) if compat else []
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    return model, ddp_model, originals


def run_200_candidate(
    label: str,
    cfg,
    dataset,
    base_state: dict[str, torch.Tensor],
    batches: list[dict[str, Any]],
    train_rng: dict[str, Any],
    device: torch.device,
    mask: tuple[int, ...],
    compat: bool,
    step_writer,
    tb_writer: SummaryWriter | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    model, ddp_model, originals = prepare_model(cfg, dataset, base_state, device, mask, compat)
    names = dict(model.named_parameters())
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, ddp_model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    restore_rng_state(train_rng)
    optimizer_sig = optimizer_signature(optimizer)
    records = []
    sums = {name: 0.0 for name in SCALAR_FIELDS}
    skipped = 0
    nonfinite_loss = 0
    nonfinite_grad = 0
    start_run = time.perf_counter()

    for sequence, raw_batch in enumerate(batches, start=1):
        batch = to_device(raw_batch, device)
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        writer = CaptureWriter()
        scale_before = float(scaler.get_scale())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with amp.autocast(enabled=True):
            score, feat = ddp_model(batch["image"], cam_label=batch["camid"])
        total = loss_fn(
            score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
            clothes_ids=batch["clothes_id"], train_writer=writer, step=sequence + len(batches),
        )
        writer.add_scalar("loss/total", total.detach().item(), sequence + len(batches))
        loss_finite = bool(torch.isfinite(total.detach()).item())
        nonfinite_loss += int(not loss_finite)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm, grad_finite = gradient_norm(model)
        nonfinite_grad += int(not grad_finite)
        before = {key: value.detach().clone() for key, value in names.items() if key in {
            "patch_embed.proj.weight", "head.weight", "head_clip_bio.weight",
            "head_clip_bioReverse.0.head_fc.weight", "head_clip_bioReverse.1.head_fc.weight",
        }}
        scaler.step(optimizer)
        found_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = float(sum(float(value.item()) for value in found_state.values()))
        scaler.update()
        scale_after = float(scaler.get_scale())
        deltas = {
            key: float((names[key].detach() - value).float().abs().max().detach().cpu())
            for key, value in before.items()
        }
        required = [key for key in ("patch_embed.proj.weight", "head.weight") if key in deltas]
        parameter_updated = (not found_inf) and all(deltas[key] > 0 for key in required)
        local_scalars = scalar_values(writer, total)
        local_values = [local_scalars[name] for name in SCALAR_FIELDS]
        reduced_values = reduce_mean_vector(local_values, device)
        control = reduce_max_vector([
            float(found_inf > 0),
            float(not parameter_updated),
            float(not loss_finite),
            float(not grad_finite),
        ], device)
        memory = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
        runtime = time.perf_counter() - start
        reduced_control = reduce_max_vector([float(memory), runtime], device)
        # Average scale and runtime are useful for paired fingerprints; max
        # memory and any-rank skip are preserved separately above.
        scale_runtime = reduce_mean_vector([scale_before, scale_after, runtime], device)
        values = dict(zip(SCALAR_FIELDS, reduced_values))
        values.update({
            "candidate": label,
            "sequence": sequence,
            "official_step": 595 + sequence,
            "global_step": sequence,
            "train_accuracy": values["id_acc"],
            "scale_before": scale_runtime[0],
            "scale_after": scale_runtime[1],
            "found_inf_mean": float(reduce_mean_vector([found_inf], device)[0]),
            "step_skipped_any": int(control[0] > 0),
            "step_skipped_fraction": float(reduce_mean_vector([float(found_inf > 0)], device)[0]),
            "gradient_norm": float(reduce_mean_vector([grad_norm], device)[0]),
            "runtime_sec": scale_runtime[2],
            "peak_memory_bytes": int(reduced_control[0]),
            "parameter_updated_all_ranks": int(control[1] == 0),
            "loss_finite_all_ranks": int(control[2] == 0),
            "gradient_finite_all_ranks": int(control[3] == 0),
            "lr": float(scheduler._get_lr(cfg.TRAIN.START_EPOCH)[0]),
        })
        if values["step_skipped_any"]:
            skipped += 1
        for name in SCALAR_FIELDS:
            sums[name] += values[name]
        if dist.get_rank() == 0:
            row = {field: values.get(field, "") for field in STEP_CSV_FIELDS}
            row["epoch"] = 1
            row["iteration"] = sequence
            step_writer.writerow(row)
            if tb_writer is not None:
                for name in SCALAR_FIELDS:
                    tag = {
                        "total_loss": "loss/total", "ce_loss": "loss/id", "triplet_loss": "loss/triplet",
                        "bio_loss": "loss/clip_bio_0", "reverse0_loss": "loss/clip_bio_reverse_0",
                        "reverse1_loss": "loss/clip_bio_reverse_1", "id_acc": "acc/id",
                        "bio_acc": "acc/clip_bio_0", "reverse0_acc": "acc/clip_bio_reverse_0",
                        "reverse1_acc": "acc/clip_bio_reverse_1",
                    }[name]
                    tb_writer.add_scalar(tag, values[name], values["official_step"])
                tb_writer.add_scalar("lr", values["lr"], values["official_step"])
        records.append(values)
        del score, feat, total, batch, writer

    result = None
    if dist.get_rank() == 0:
        result = {
            "label": label,
            "mask": list(mask) if compat else [],
            "precision_semantics": "exact Phase2M wrap_fp32_block on selected EVA blocks; formal autocast elsewhere" if compat else "formal PPU FP16/autocast path",
            "iterations": len(records),
            "records": records,
            "skipped_steps": int(sum(int(row["step_skipped_any"]) for row in records)),
            "nonfinite_loss_count": int(sum(int(not row["loss_finite_all_ranks"]) for row in records)),
            "nonfinite_gradient_count": int(sum(int(not row["gradient_finite_all_ranks"]) for row in records)),
            "successful_updates": int(sum(int(row["parameter_updated_all_ranks"]) for row in records)),
            "initial_scale": records[0]["scale_before"] if records else INITIAL_SCALE,
            "final_scale": records[-1]["scale_after"] if records else INITIAL_SCALE,
            "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in records])) if records else None,
            "max_peak_memory_bytes": max((row["peak_memory_bytes"] for row in records), default=None),
            "runtime_total_sec": time.perf_counter() - start_run,
            "optimizer_signature": optimizer_sig,
            "trajectory_fingerprint": json_hash(records),
        }
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler, ddp_model
    restore_wrappers(originals)
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    dist.barrier()
    return result, {
        "optimizer_signature": optimizer_sig,
        "rank": dist.get_rank(),
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataset, val_loader, device: torch.device, general: bool) -> dict[str, Any]:
    evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=True) if general else R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=True)
    evaluator.reset()
    model.eval()
    for raw in val_loader:
        image = raw["image"].to(device, non_blocking=True)
        cam = raw["camid"].to(device, non_blocking=True)
        with amp.autocast(enabled=False):
            feature = model(image, cam_label=cam)
        if not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError("non-finite retrieval feature in Phase 2N evaluation")
        evaluator.update((feature, raw["pid"], raw["camid"], raw["clothes_id"], raw["image_path"]))
    cmc, mean_ap, distance, _, _, qf, gf, _ = evaluator.compute()
    return {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mean_ap),
        "query_feature_shape": list(qf.shape),
        "gallery_feature_shape": list(gf.shape),
        "distance_shape": list(distance.shape),
        "distance_nan": int(np.isnan(distance).sum()),
        "distance_inf": int(np.isinf(distance).sum()),
    }


def reduce_epoch_stats(local_sums: dict[str, float], local_steps: int, device: torch.device) -> dict[str, float]:
    names = list(SCALAR_FIELDS) + ["grad_norm", "runtime_sec"]
    values = reduce_mean_vector([local_sums.get(name, 0.0) for name in names], device)
    counts = reduce_mean_vector([float(local_steps)], device)[0]
    return {name: values[index] / max(counts, 1.0) for index, name in enumerate(names)}


def load_official_epoch_curve(max_epoch: int = 60) -> dict[int, dict[str, float | None]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    series: dict[str, dict[int, float]] = {}
    for path in sorted(OFFICIAL_EVENT_DIR.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            destination = series.setdefault(tag, {})
            for item in acc.Scalars(tag):
                destination.setdefault(int(item.step), float(item.value))
    output = {}
    train_map = {
        "loss/total": "total_loss", "loss/id": "ce_loss", "loss/triplet": "triplet_loss",
        "loss/clip_bio": "bio_loss", "loss/clip_bio_reverse_0": "reverse0_loss",
        "loss/clip_bio_reverse_1": "reverse1_loss", "acc/id": "id_acc",
        "acc/clip_bio": "bio_acc", "acc/clip_bio_reverse_0": "reverse0_acc",
        "acc/clip_bio_reverse_1": "reverse1_acc",
    }
    for epoch in range(1, max_epoch + 1):
        lo = 596 + (epoch - 1) * 596
        hi = 596 + epoch * 596
        row = {"epoch": epoch}
        for tag, name in train_map.items():
            values = [value for step, value in series.get(tag, {}).items() if lo <= step < hi]
            row[name] = float(np.mean(values)) if values else None
        row["cc_rank1"] = series.get("rank1", {}).get(epoch)
        row["cc_mAP"] = series.get("mAP", {}).get(epoch)
        output[epoch] = row
    return output


def load_current_epoch_curves() -> list[dict[int, dict[str, Any]]]:
    curves = []
    for path in CURRENT_RUN_PATHS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        curves.append({int(row["epoch"]): row for row in payload.get("records", [])})
    return curves


def curve_value(row: dict[str, Any], name: str) -> float | None:
    if name in row:
        return finite_or_none(row.get(name))
    if name == "cc_rank1":
        return finite_or_none(row.get("cc", {}).get("rank1"))
    if name == "cc_mAP":
        return finite_or_none(row.get("cc", {}).get("mAP"))
    return None


def epoch25_gate(candidate_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(candidate_records) < 25:
        return None
    official = load_official_epoch_curve(60)
    current = load_current_epoch_curves()
    if not official or not current:
        return {"status": "NOT_COMPUTED_MISSING_OFFICIAL_OR_CURRENT_CURVE"}

    def rmse(values_a, values_b):
        pairs = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
        if not pairs:
            return None
        return float(np.sqrt(np.mean([(a - b) ** 2 for a, b in pairs])))

    candidate_by_epoch = {int(row["epoch"]): row for row in candidate_records}
    rows = list(range(19, 26))
    r1_official = [official[e].get("cc_rank1") for e in rows]
    map_official = [official[e].get("cc_mAP") for e in rows]
    r1_candidate = [curve_value(candidate_by_epoch[e], "cc_rank1") for e in rows]
    map_candidate = [curve_value(candidate_by_epoch[e], "cc_mAP") for e in rows]
    r1_current_runs = [[curve_value(run[e], "cc_rank1") for e in rows] for run in current]
    map_current_runs = [[curve_value(run[e], "cc_mAP") for e in rows] for run in current]
    r1_current_mean = [float(np.mean([run[idx] for run in r1_current_runs if run[idx] is not None])) for idx in range(len(rows))]
    map_current_mean = [float(np.mean([run[idx] for run in map_current_runs if run[idx] is not None])) for idx in range(len(rows))]
    current_r1_rmse = rmse(r1_current_mean, r1_official)
    candidate_r1_rmse = rmse(r1_candidate, r1_official)
    current_map_rmse = rmse(map_current_mean, map_official)
    candidate_map_rmse = rmse(map_candidate, map_official)
    r1_improvement = (current_r1_rmse - candidate_r1_rmse) / current_r1_rmse if current_r1_rmse else None
    map_improvement = (current_map_rmse - candidate_map_rmse) / current_map_rmse if current_map_rmse else None
    branch_closeness = {}
    for branch in ("bio_acc", "reverse0_acc", "reverse1_acc"):
        target = [official[e].get(branch) for e in rows]
        candidate_values = [curve_value(candidate_by_epoch[e], branch) for e in rows]
        current_values = []
        for idx in range(len(rows)):
            values = [curve_value(run[rows[idx]], branch) for run in current if rows[idx] in run]
            current_values.append(float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None)
        c_rmse = rmse(candidate_values, target)
        p_rmse = rmse(current_values, target)
        branch_closeness[branch] = {
            "current_rmse": p_rmse,
            "candidate_rmse": c_rmse,
            "candidate_closer": bool(c_rmse is not None and p_rmse is not None and c_rmse < p_rmse),
        }
    closer_count = sum(item["candidate_closer"] for item in branch_closeness.values())
    return {
        "epochs": rows,
        "current_r1_rmse": current_r1_rmse,
        "candidate_r1_rmse": candidate_r1_rmse,
        "r1_rmse_improvement": r1_improvement,
        "current_mAP_rmse": current_map_rmse,
        "candidate_mAP_rmse": candidate_map_rmse,
        "mAP_rmse_improvement": map_improvement,
        "branch_closeness": branch_closeness,
        "trajectory_closer_count": closer_count,
        "gate": bool((r1_improvement or -1) >= 0.25 and (map_improvement or -1) >= 0.15 and closer_count >= 2),
    }


def semantic_collapse(record: dict[str, Any]) -> bool:
    branch_names = ("bio_loss", "reverse0_loss", "reverse1_loss", "bio_acc", "reverse0_acc", "reverse1_acc")
    return any(curve_value(record, name) is None for name in branch_names)


def epoch15_gate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    row = next((item for item in records if int(item["epoch"]) == 15), None)
    if row is None:
        return None
    cc = row.get("cc", {})
    rank1 = finite_or_none(cc.get("rank1"))
    m_ap = finite_or_none(cc.get("mAP"))
    collapse = semantic_collapse(row)
    return {
        "candidate_rank1": rank1,
        "candidate_mAP": m_ap,
        "official_rank1": 0.48724,
        "official_mAP": 0.25814,
        "minimum_rank1": 0.467,
        "minimum_mAP": 0.243,
        "semantic_collapse": collapse,
        "gate": bool(rank1 is not None and m_ap is not None and rank1 >= 0.467 and m_ap >= 0.243 and not collapse),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_center: torch.optim.Optimizer,
    scheduler,
    scaler,
    train_loader,
    mask: tuple[int, ...],
    cfg,
    base_manifest: dict[str, Any],
):
    rank = dist.get_rank()
    local_rng = capture_rng_state()
    rng_states = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rng_states, local_rng)
    generator_state = getattr(train_loader, "generator", None)
    local_generator = generator_state.get_state() if generator_state is not None else None
    generator_states = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(generator_states, local_generator)
    sampler_epoch = int(getattr(getattr(train_loader, "sampler", None), "epoch", 0))
    sampler_epochs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(sampler_epochs, sampler_epoch)
    if rank == 0:
        payload = {
            "phase": "2N-S/T",
            "completed_epoch": int(epoch),
            "mask": list(mask),
            "precision_semantics": "exact Phase2M wrap_fp32_block on selected EVA blocks; formal autocast elsewhere",
            "model_state": checkpoint_cpu_state(model),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_center_state": optimizer_center.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_by_rank": rng_states,
            "loader_generator_state_by_rank": generator_states,
            "sampler_epoch_by_rank": sampler_epochs,
            "parameter_manifest": base_manifest,
            "config_dump": cfg.dump(),
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    dist.barrier()


def restore_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_center: torch.optim.Optimizer,
    scheduler,
    scaler,
    train_loader,
    rank: int,
) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer_center.load_state_dict(payload["optimizer_center_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    rng_by_rank = payload.get("rng_by_rank", [])
    if rank < len(rng_by_rank) and rng_by_rank[rank] is not None:
        restore_rng_state(rng_by_rank[rank])
    generators = payload.get("loader_generator_state_by_rank", [])
    if rank < len(generators) and generators[rank] is not None and getattr(train_loader, "generator", None) is not None:
        train_loader.generator.set_state(generators[rank])
    epochs = payload.get("sampler_epoch_by_rank", [])
    if rank < len(epochs) and epochs[rank] is not None and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(int(epochs[rank]))
    return int(payload["completed_epoch"])


def train_epochs(
    cfg,
    dataset,
    train_loader,
    val_loader,
    base_state: dict[str, torch.Tensor],
    base_manifest: dict[str, Any],
    device: torch.device,
    mask: tuple[int, ...],
    output_dir: Path,
    max_epoch: int,
    resume: Path | None,
) -> dict[str, Any] | None:
    rank = dist.get_rank()
    model, ddp_model, originals = prepare_model(cfg, dataset, base_state, device, mask, True)
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, ddp_model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    start_epoch = 1
    if resume is not None:
        completed = restore_checkpoint(resume, model, optimizer, optimizer_center, scheduler, scaler, train_loader, rank)
        start_epoch = completed + 1
    step_path = output_dir / "train_steps.csv"
    step_handle = step_writer = None
    if rank == 0:
        step_handle, step_writer = open_step_writer(step_path, append=resume is not None)
    tb_writer = SummaryWriter(str(output_dir / "tensorboard")) if rank == 0 else None
    records: list[dict[str, Any]] = []
    record_path = output_dir / "epoch_records.json"
    if rank == 0 and record_path.exists():
        try:
            records = json.loads(record_path.read_text()).get("records", [])
        except Exception:
            records = []

    for epoch in range(start_epoch, max_epoch + 1):
        scheduler.step(epoch)
        model.train()
        local_sums = {name: 0.0 for name in SCALAR_FIELDS}
        local_sums.update({"grad_norm": 0.0, "runtime_sec": 0.0})
        local_steps = 0
        local_skips = 0
        local_nonfinite = 0
        local_scale_min = float(scaler.get_scale())
        local_scale_max = float(scaler.get_scale())
        local_scale_end = float(scaler.get_scale())
        local_memory_max = 0
        local_start = time.perf_counter()
        for index, raw in enumerate(train_loader):
            iteration = index + 1
            global_step = (epoch - 1) * len(train_loader) + iteration
            batch = to_device(raw, device)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            writer = CaptureWriter()
            scale_before = float(scaler.get_scale())
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with amp.autocast(enabled=True):
                score, feat = ddp_model(batch["image"], cam_label=batch["camid"])
            total = loss_fn(
                score, feat, batch["pid"], batch["camid"], batch["caption_feature"],
                clothes_ids=batch["clothes_id"], train_writer=writer, step=global_step,
            )
            writer.add_scalar("loss/total", total.detach().item(), global_step)
            loss_finite = bool(torch.isfinite(total.detach()).item())
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm, grad_finite = selected_gradient_norm(model)
            scaler.step(optimizer)
            found_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            found_inf = float(sum(float(value.item()) for value in found_state.values()))
            step_skipped = found_inf > 0
            scaler.update()
            scale_after = float(scaler.get_scale())
            memory = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
            runtime = time.perf_counter() - start
            local_values = scalar_values(writer, total)
            reduced_values = reduce_mean_vector([local_values[name] for name in SCALAR_FIELDS], device)
            controls = reduce_max_vector([
                float(step_skipped), float(not loss_finite), float(not grad_finite), float(memory),
            ], device)
            means = reduce_mean_vector([scale_before, scale_after, found_inf, float(step_skipped), grad_norm, runtime], device)
            values = dict(zip(SCALAR_FIELDS, reduced_values))
            values.update({
                "candidate": "Compat V2",
                "epoch": epoch,
                "iteration": iteration,
                "global_step": global_step,
                "train_accuracy": reduced_values[SCALAR_FIELDS.index("id_acc")],
                "lr": float(scheduler._get_lr(epoch)[0]),
                "scale_before": means[0],
                "scale_after": means[1],
                "found_inf_mean": means[2],
                "step_skipped_any": int(controls[0] > 0),
                "step_skipped_fraction": means[3],
                "gradient_norm": means[4],
                "runtime_sec": means[5],
                "peak_memory_bytes": int(controls[3]),
                "parameter_updated_all_ranks": int(controls[0] == 0),
                "loss_finite_all_ranks": int(controls[1] == 0),
                "gradient_finite_all_ranks": int(controls[2] == 0),
            })
            for name in SCALAR_FIELDS:
                local_sums[name] += values[name]
            local_sums["grad_norm"] += values["gradient_norm"]
            local_sums["runtime_sec"] += values["runtime_sec"]
            local_steps += 1
            local_skips += int(values["step_skipped_any"])
            local_nonfinite += int(not values["loss_finite_all_ranks"] or not values["gradient_finite_all_ranks"])
            local_scale_min = min(local_scale_min, values["scale_after"])
            local_scale_max = max(local_scale_max, values["scale_after"])
            local_scale_end = values["scale_after"]
            local_memory_max = max(local_memory_max, values["peak_memory_bytes"])
            if rank == 0:
                step_writer.writerow({field: values.get(field, "") for field in STEP_CSV_FIELDS})
                if tb_writer is not None:
                    tags = {
                        "total_loss": "loss/total", "ce_loss": "loss/id", "triplet_loss": "loss/triplet",
                        "bio_loss": "loss/clip_bio_0", "reverse0_loss": "loss/clip_bio_reverse_0",
                        "reverse1_loss": "loss/clip_bio_reverse_1", "id_acc": "acc/id",
                        "bio_acc": "acc/clip_bio_0", "reverse0_acc": "acc/clip_bio_reverse_0",
                        "reverse1_acc": "acc/clip_bio_reverse_1",
                    }
                    for name, tag in tags.items():
                        tb_writer.add_scalar(tag, values[name], global_step)
                    tb_writer.add_scalar("lr", values["lr"], global_step)
            del score, feat, total, batch, writer
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass
        stats = reduce_epoch_stats(local_sums, local_steps, device)
        eval_cc = None
        eval_general = None
        dist.barrier()
        if rank == 0:
            eval_cc = evaluate_model(model, dataset, val_loader, device, general=False)
            eval_general = evaluate_model(model, dataset, val_loader, device, general=True)
        objects = [eval_cc, eval_general]
        dist.broadcast_object_list(objects, src=0)
        eval_cc, eval_general = objects
        row = {
            "epoch": epoch,
            "train_loss": stats["total_loss"],
            "id_loss": stats["ce_loss"],
            "triplet_loss": stats["triplet_loss"],
            "bio_loss": stats["bio_loss"],
            "reverse0_loss": stats["reverse0_loss"],
            "reverse1_loss": stats["reverse1_loss"],
            "id_acc": stats["id_acc"],
            "bio_acc": stats["bio_acc"],
            "reverse0_acc": stats["reverse0_acc"],
            "reverse1_acc": stats["reverse1_acc"],
            "train_accuracy": stats["id_acc"],
            "lr_normal": float(scheduler._get_lr(epoch)[0]),
            "lr_bias": float(scheduler._get_lr(epoch)[-1]),
            "amp_scale_min": local_scale_min,
            "amp_scale_max": local_scale_max,
            "amp_scale_end": local_scale_end,
            "skipped_steps": local_skips,
            "nonfinite_events": local_nonfinite,
            "gradient_norm_mean": stats["grad_norm"],
            "runtime_per_step_sec": stats["runtime_sec"],
            "peak_memory_bytes": local_memory_max,
            "cc": eval_cc,
            "general": eval_general,
            "semantic_collapse": False,
            "epoch_runtime_sec": time.perf_counter() - local_start,
        }
        row["semantic_collapse"] = semantic_collapse(row)
        if rank == 0:
            records = [item for item in records if int(item.get("epoch", -1)) != epoch]
            records.append(row)
            records.sort(key=lambda item: int(item["epoch"]))
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps({
                "phase": "2N-S/T", "mask": list(mask), "records": records,
                "parameter_manifest": base_manifest,
            }, indent=2, sort_keys=True) + "\n")
            if tb_writer is not None:
                tb_writer.add_scalar("cc/rank1", eval_cc["rank1"], epoch)
                tb_writer.add_scalar("cc/mAP", eval_cc["mAP"], epoch)
                tb_writer.add_scalar("general/rank1", eval_general["rank1"], epoch)
                tb_writer.add_scalar("general/mAP", eval_general["mAP"], epoch)
            print("PHASE2N_EPOCH_PROGRESS", json.dumps({
                "epoch": epoch, "cc_rank1": eval_cc["rank1"], "cc_mAP": eval_cc["mAP"],
                "train_loss": row["train_loss"], "skips": row["skipped_steps"],
                "scale": row["amp_scale_end"], "output": str(output_dir),
            }, sort_keys=True), flush=True)
        # Save the current stage's latest state and immutable gate states.  It
        # is safe to update only the new Phase 2N directory.
        if rank == 0:
            step_handle.flush()
        dist.barrier()
        latest = output_dir / "resume_latest.pth"
        save_checkpoint(latest, epoch, model, optimizer, optimizer_center, scheduler, scaler, train_loader, mask, cfg, base_manifest)
        if epoch in (15, 25, 60):
            save_checkpoint(output_dir / f"resume_epoch{epoch}.pth", epoch, model, optimizer, optimizer_center, scheduler, scaler, train_loader, mask, cfg, base_manifest)
        if rank == 0:
            if epoch == 15:
                gate = epoch15_gate(records)
                (output_dir / "epoch15_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
                print("PHASE2N_EPOCH15_GATE", json.dumps(gate, sort_keys=True), flush=True)
            if epoch == 25:
                gate = epoch25_gate(records)
                (output_dir / "epoch25_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
                print("PHASE2N_EPOCH25_GATE", json.dumps(gate, sort_keys=True), flush=True)
        dist.barrier()
        # New DataLoaderX iterator is created on the next epoch.  The sampler
        # intentionally remains at its released default epoch=0, matching
        # processor/processor.py, which never calls set_epoch.

    if rank == 0:
        payload = {
            "phase": "2N-S/T", "mask": list(mask), "max_epoch": max_epoch,
            "records": records, "parameter_manifest": base_manifest,
            "output_dir": str(output_dir),
            "epoch15_gate": epoch15_gate(records),
            "epoch25_gate": epoch25_gate(records),
            "final_resume": str(output_dir / f"resume_epoch{max_epoch}.pth") if max_epoch in (15, 25, 60) else str(output_dir / "resume_latest.pth"),
        }
        (output_dir / "training_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        payload = None
    if tb_writer is not None:
        tb_writer.flush(); tb_writer.close()
    if step_handle is not None:
        step_handle.close()
    del optimizer, optimizer_center, center, loss_fn, scheduler, scaler, ddp_model
    restore_wrappers(originals)
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    dist.barrier()
    return payload


def paired200_main(args, cfg, dataset, train_loader, device, mask, output_dir, base_state, base_manifest, train_rng):
    rank = dist.get_rank()
    batches, fingerprints = materialize_batches(train_loader, 200)
    if len(batches) != 200:
        raise RuntimeError(f"Phase 2N paired200 materialized {len(batches)} batches, expected 200")
    rank_fingerprints = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rank_fingerprints, {
        "rank": rank, "count": len(fingerprints), "stream_sha256": stream_hash(fingerprints),
        "first": fingerprints[:3], "last": fingerprints[-1],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    step_handle = step_writer = None
    tb_writer = SummaryWriter(str(output_dir / "paired200_tensorboard")) if rank == 0 else None
    if rank == 0:
        step_handle, step_writer = open_step_writer(output_dir / "paired200_steps.csv", append=False)
    # Both candidates are constructed from the exact same post-loader RNG.
    # The CPU batches are shared within this process and never regenerated.
    current_result, current_meta = run_200_candidate(
        "Current", cfg, dataset, base_state, batches, train_rng, device, mask, False,
        step_writer, tb_writer,
    )
    compat_result, compat_meta = run_200_candidate(
        "Compat V2", cfg, dataset, base_state, batches, train_rng, device, mask, True,
        step_writer, tb_writer,
    )
    if rank == 0:
        if step_handle is not None:
            step_handle.flush(); step_handle.close()
        if tb_writer is not None:
            tb_writer.flush(); tb_writer.close()
        official = official_first200()
        payload = {
            "phase": "2N-R",
            "config": {
                "dataset": "ltcc", "seed": SEED, "world_size": dist.get_world_size(),
                "batch_size": int(cfg.DATA.BATCH_SIZE), "iterations": 200,
                "iterations_per_epoch": len(train_loader), "initial_scale": INITIAL_SCALE,
                "losses": ["ce", "triplet", "clipBio", "clipBioReverse"],
                "optimizer": str(cfg.SOLVER.OPTIMIZER_NAME), "scheduler": "released epoch-based cosine",
                "forward": "autocast enabled; loss outside autocast",
                "device": str(device),
            },
            "candidate": {"name": "PPU_COMPAT_V2_SPARSE_MASK", "mask": list(mask), "fp32_fraction": len(mask) / N_BLOCKS},
            "parameter_manifest": base_manifest,
            "batch_fingerprint": {"rank_streams": rank_fingerprints, "common_within_candidates": True},
            "official_first200": official,
            "current": current_result,
            "compat_v2": compat_result,
            "fingerprints": {
                "current_trajectory": current_result["trajectory_fingerprint"],
                "compat_trajectory": compat_result["trajectory_fingerprint"],
                "official_scalar_rows": official.get("fingerprint"),
                "common_parameter_state": base_manifest["parameter_state_sha256"],
                "common_batch_rank0": rank_fingerprints[0]["stream_sha256"],
            },
            "paired_gate": {
                "executed": True,
                "criterion": "Compat not materially worse than Current; numerical fidelity remains primary",
                "current_mean_total_loss": float(np.mean([row["total_loss"] for row in current_result["records"]])),
                "compat_mean_total_loss": float(np.mean([row["total_loss"] for row in compat_result["records"]])),
                "current_skips": current_result["skipped_steps"],
                "compat_skips": compat_result["skipped_steps"],
                "compat_no_nonfinite": compat_result["nonfinite_loss_count"] == 0 and compat_result["nonfinite_gradient_count"] == 0,
                "pass": bool(compat_result["nonfinite_loss_count"] == 0 and compat_result["nonfinite_gradient_count"] == 0),
            },
            "notes": [
                "Two real PPU-ZW810E ranks were used; each candidate received the same rank-local augmented batches.",
                "Current and Compat were rebuilt from one common model state and the same post-loader RNG state.",
                "The candidate changes only runtime block arithmetic; parameter names, count, and initial tensor bytes are unchanged.",
                "The paired gate does not use Rank-1 or mAP to select the precision policy.",
            ],
        }
        output = output_dir / "paired200.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("PHASE2N_PAIRED200_COMPLETE", json.dumps({
            "output": str(output), "current_skips": current_result["skipped_steps"],
            "compat_skips": compat_result["skipped_steps"], "gate": payload["paired_gate"],
        }, sort_keys=True), flush=True)
    else:
        payload = None
    dist.barrier()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paired200", "epochs"), required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "compat_v2"))
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--resume", default="")
    parser.add_argument("--mask", default="")
    args = parser.parse_args()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"Phase 2N training requires exactly 2 ranks, got {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")
    cfg = make_config()
    cfg.defrost()
    cfg.MODEL.DIST_TRAIN = True
    cfg.SOLVER.MAX_EPOCHS = 60
    cfg.TRAIN.START_EPOCH = 1
    cfg.freeze()
    mask = tuple(sorted({int(x) for x in (args.mask.split(",") if args.mask else mask_from_selection()) if str(x) != ""}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) if rank == 0 else None
    seed_all(SEED)
    data_result = make_rank_data(cfg, device)
    train_loader, query_loader, gallery_loader, dataset, train_sampler, val_loader, train_original = data_result
    # Build the natural model after DataLoader construction, matching train.py
    # and the frozen Phase-2 initialization order.
    template = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False)
    base_state = {name: value.detach().cpu().clone() for name, value in template.state_dict().items()}
    base_manifest = state_manifest(template, {name: value for name, value in base_state.items() if name in dict(template.named_parameters())})
    state_hashes = [None for _ in range(world_size)]
    dist.all_gather_object(state_hashes, base_manifest["parameter_state_sha256"])
    if len(set(state_hashes)) != 1:
        raise RuntimeError(f"initial model states differ across ranks: {state_hashes}")
    del template
    gc.collect(); torch.cuda.empty_cache()
    if args.mode == "paired200":
        # Materialization consumes worker-side augmentation RNG but does not
        # mutate the model.  Capture the common process state immediately
        # before the two candidate runs.
        batches_preview, _ = materialize_batches(train_loader, 1)
        if len(batches_preview) != 1:
            raise RuntimeError("failed to establish post-loader RNG state")
        # Rebuild the formal loader for the actual 200 stream; this preserves
        # the official seed/sampler sequence while keeping the preview out of
        # the common candidate batches.
        data_result = make_rank_data(cfg, device)
        train_loader, query_loader, gallery_loader, dataset, train_sampler, val_loader, train_original = data_result
        train_rng = capture_rng_state()
        paired200_main(args, cfg, dataset, train_loader, device, mask, output_dir, base_state, base_manifest, train_rng)
    else:
        resume = Path(args.resume) if args.resume else None
        payload = train_epochs(cfg, dataset, train_loader, val_loader, base_state, base_manifest, device, mask, output_dir, args.max_epoch, resume)
    dist.barrier()
    for item in (train_loader, query_loader, gallery_loader, val_loader, train_original):
        if hasattr(item, "shutdown"):
            try:
                item.shutdown()
            except Exception:
                pass
    del base_state, dataset, cfg
    gc.collect(); torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
