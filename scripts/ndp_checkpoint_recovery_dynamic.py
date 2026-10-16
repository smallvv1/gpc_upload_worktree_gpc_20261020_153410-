#!/usr/local/bin/python3
"""Recover the historical LTCC I3 GPC state and run a 32-relation NDP pilot.

This is a diagnostic-only runner.  It never writes a model checkpoint and it
does not call a training loop.  Each candidate relation is evaluated with two
one-step virtual branches restored from the same E60 rank-0 model,
optimizer, scheduler, GradScaler and RNG state.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_final_gpc as fg  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from loss import make_loss  # noqa: E402


OUT_ROOT = Path(os.environ.get(
    "NDP_RECOVERY_OUT",
    "/home/DIFFER/outputs/ndp_checkpoint_recovery/20261014_184400",
))
CHECKPOINT = Path(
    "/home/output/phase7i_relation_specificity/I3_id_only_matched/model_final.pth"
)
RANK_STATE = Path(
    "/home/output/phase7i_relation_specificity/I3_id_only_matched/rank0_state_epoch060.pt"
)
RELATION_SOURCE = Path(
    "/data/DIFFER/outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"
)
CAPTION_DIR = Path("/data/data/DIFFER_TextCaption/LTCC_ReID")
TARGET_RELATIONS = 32
K = 10
POOL_K = 20
LAMBDA_GPC = 0.1
DEVICE = torch.device("cuda:0")
EPS = 1e-12


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().float().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> Any:
    cfg = fg.cfg_for("ltcc", OUT_ROOT)
    cfg.defrost()
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = str(CAPTION_DIR)
    cfg.DATA.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def load_rank_state() -> dict[str, Any]:
    # The historical file is trusted and local.  Its NumPy pickle metadata
    # uses the pre-2.6 numpy._core module name; this alias is compatibility
    # handling, not a bypass of weights_only safety for ordinary weights.
    import numpy as np_local
    sys.modules.setdefault("numpy._core", np_local.core)
    sys.modules.setdefault("numpy._core.multiarray", np_local.core.multiarray)
    return torch.load(RANK_STATE, map_location="cpu", weights_only=False)


def make_bank(dataset: Any) -> dict[str, Any]:
    exact: set[tuple[int, int]] = set()
    with RELATION_SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            exact.add(tuple(sorted((int(row["index_a"]), int(row["index_b"])))) )
    return {
        "exact_lookup": exact,
        "path_to_index": {str(item["image_path"]): index for index, item in enumerate(dataset.train)},
        "dataset": "ltcc",
        "source": str(RELATION_SOURCE),
    }


def restore_rng(rng: dict[str, Any]) -> None:
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            torch.cuda.set_rng_state(rng["torch_cuda"][0], device=DEVICE)


def build_train_features(model: torch.nn.Module, cfg: Any, dataset: Any) -> torch.Tensor:
    _, transform_test = fg.build_img_transforms(cfg)
    loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=False)
            camids = batch["camid"].to(DEVICE, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            chunks.append(F.normalize(output.float(), dim=-1).cpu())
    features = torch.cat(chunks, dim=0)
    if features.shape != (len(dataset.train), 1024):
        raise RuntimeError(f"unexpected train feature shape: {tuple(features.shape)}")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("non-finite train feature bank")
    return features


def build_neighborhoods(features: torch.Tensor, dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = features.numpy().astype(np.float32, copy=False)
    pid = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    n = len(pid)
    similarity = matrix @ matrix.T
    similarity[np.arange(n), np.arange(n)] = -np.inf
    similarity[pid[:, None] == pid[None, :]] = -np.inf
    pool = np.argpartition(-similarity, POOL_K - 1, axis=1)[:, :POOL_K]
    pool_scores = np.take_along_axis(similarity, pool, axis=1)
    order = np.argsort(-pool_scores, axis=1)
    pool = np.take_along_axis(pool, order, axis=1)
    neighbors = pool[:, :K].astype(np.int64, copy=False)

    # A deterministic same-PID positive partner is needed only for the
    # requested “nearby different-PID sample's own local margin” telemetry.
    positive_partner = np.full(n, -1, dtype=np.int64)
    for value in np.unique(pid):
        indices = np.flatnonzero(pid == value)
        if len(indices) < 2:
            continue
        block = matrix[indices] @ matrix[indices].T
        np.fill_diagonal(block, -np.inf)
        positive_partner[indices] = indices[np.argmax(block, axis=1)]
    return neighbors, pool, positive_partner


def probe_batch(dataset: Any, cfg: Any, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    _, transform_test = fg.build_img_transforms(cfg)
    items = [dataset.train[int(index)] for index in indices]
    loader = DataLoader(
        ImageDataset(items, transform=transform_test),
        batch_size=len(items),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    return batch["image"].to(DEVICE), batch["camid"].to(DEVICE)


def feature_probe(model: torch.nn.Module, images: torch.Tensor, camids: torch.Tensor, with_grad: bool) -> torch.Tensor:
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        with amp.autocast(enabled=True):
            output = model(images, cam_label=camids)
        if isinstance(output, tuple):
            output = output[1][0]
        return F.normalize(output.float(), dim=-1)


def local_metrics(
    z: torch.Tensor,
    indices: list[int],
    endpoints: tuple[int, int],
    neighbors: np.ndarray,
    pool: np.ndarray,
    base_features: torch.Tensor,
    k: int = K,
) -> dict[str, Any]:
    positions = {int(index): position for position, index in enumerate(indices)}
    left, right = endpoints
    positive = torch.sum(z[positions[left]] * z[positions[right]])
    endpoint_values = []
    endpoint_hard = []
    pool_new = 0
    for endpoint in (left, right):
        endpoint_z = z[positions[endpoint]]
        neg_ids = [int(x) for x in neighbors[endpoint, :k]]
        neg_values = torch.stack([torch.sum(endpoint_z * z[positions[x]]) for x in neg_ids])
        hard = neg_values.max()
        endpoint_values.append(positive - hard)
        endpoint_hard.append(hard)
        pool_ids = [int(x) for x in pool[endpoint, :POOL_K]]
        pool_values = torch.stack([torch.sum(endpoint_z * z[positions[x]]) for x in pool_ids])
        top_ids = {pool_ids[int(x)] for x in torch.topk(pool_values, k=k).indices.cpu().tolist()}
        pool_new += len(top_ids - set(neg_ids))
    margin = torch.stack(endpoint_values).mean()
    return {
        "positive_similarity": positive,
        "hard_negative_similarity": torch.stack(endpoint_hard).mean(),
        "local_margin": margin,
        "endpoint_margins": endpoint_values,
        "pool_new_entries": int(pool_new),
    }


def nearby_negative_margin(
    z: torch.Tensor,
    indices: list[int],
    negative_ids: list[int],
    neighbors: np.ndarray,
    positive_partner: np.ndarray,
) -> torch.Tensor:
    positions = {int(index): position for position, index in enumerate(indices)}
    values = []
    for negative in negative_ids:
        positive = int(positive_partner[negative])
        if positive < 0:
            continue
        neg_neighbors = [int(x) for x in neighbors[negative, :K]]
        anchor = z[positions[negative]]
        pos_sim = torch.sum(anchor * z[positions[positive]])
        neg_sims = torch.stack([torch.sum(anchor * z[positions[x]]) for x in neg_neighbors])
        values.append(pos_sim - neg_sims.max())
    if not values:
        return z.sum() * 0.0
    return torch.stack(values).mean()


def gradient_dot(gm: tuple[torch.Tensor | None, ...], gp: tuple[torch.Tensor | None, ...]) -> dict[str, float]:
    dot = torch.zeros((), device=DEVICE, dtype=torch.float32)
    norm_m = torch.zeros((), device=DEVICE, dtype=torch.float32)
    norm_p = torch.zeros((), device=DEVICE, dtype=torch.float32)
    for left, right in zip(gm, gp):
        if left is None and right is None:
            continue
        if left is None or right is None:
            continue
        left_f = left.float()
        right_f = right.float()
        dot = dot + (left_f * right_f).sum()
        norm_m = norm_m + (left_f * left_f).sum()
        norm_p = norm_p + (right_f * right_f).sum()
    dot_value = float(dot.detach().cpu())
    norm_m_value = math.sqrt(max(float(norm_m.detach().cpu()), 0.0))
    norm_p_value = math.sqrt(max(float(norm_p.detach().cpu()), 0.0))
    return {
        "R_grad": max(dot_value, 0.0),
        "gradient_dot_raw": dot_value,
        "gradient_cosine": dot_value / max(norm_m_value * norm_p_value, EPS),
        "grad_M_norm": norm_m_value,
        "grad_GPC_norm": norm_p_value,
    }


def reset_branch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: amp.GradScaler,
    base_model: dict[str, torch.Tensor],
    base_optimizer: dict[str, Any],
    base_scheduler: dict[str, Any],
    base_scaler: dict[str, Any],
    rng: dict[str, Any],
) -> None:
    model.load_state_dict(base_model, strict=True)
    optimizer.load_state_dict(base_optimizer)
    scheduler.load_state_dict(base_scheduler)
    scaler.load_state_dict(base_scaler)
    restore_rng(copy.deepcopy(rng))
    model.train()
    optimizer.zero_grad(set_to_none=True)


def run() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = configure()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "PPU-ZW810E":
        raise RuntimeError("NDP recovery runner requires PPU-ZW810E on cuda:0")

    dataset = fg.build_dataset(cfg)
    bank = make_bank(dataset)
    rank_state = load_rank_state()
    required = {"model", "optimizer", "scheduler", "scaler", "rng"}
    if not required.issubset(rank_state):
        raise RuntimeError(f"rank state missing {required - set(rank_state)}")

    model = fg.build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(rank_state["model"], strict=True)
    model.to(DEVICE)
    features = build_train_features(model, cfg, dataset)
    neighbors, pool, positive_partner = build_neighborhoods(features, dataset)
    np.save(OUT_ROOT / "LTCC_TRAIN_FEATURES_E60_I3.npy", features.numpy())
    np.savez_compressed(OUT_ROOT / "LTCC_FIXED_NEIGHBORHOODS_K10.npz", neighbors=neighbors, pool=pool, positive_partner=positive_partner)

    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _center_optimizer = fg.make_optimizer(cfg, model, center)
    scheduler = fg.create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=1024)
    optimizer.load_state_dict(rank_state["optimizer"])
    scheduler.load_state_dict(rank_state["scheduler"])
    scaler.load_state_dict(rank_state["scaler"])
    base_model = {key: value.detach().cpu().clone() for key, value in rank_state["model"].items()}
    base_optimizer = rank_state["optimizer"]
    base_scheduler = rank_state["scheduler"]
    base_scaler = rank_state["scaler"]
    base_rng = rank_state["rng"]
    # The historical checkpoint's PPU CUDA RNG tensor has a different
    # serialized length from the current PyTorch/PPU runtime.  Keep its
    # Python/NumPy/CPU states, but replace only the incompatible CUDA part
    # with one current-runtime snapshot; this snapshot is restored identically
    # before both branches of every paired update.
    paired_rng = copy.deepcopy(base_rng)
    paired_rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # The fixed neighborhood is built once at E60.  The dynamic sampler uses
    # epoch 60/rank 0 and the exact historical SHA-256 selector.
    sampler = fg.FinalBatchSampler(dataset.train, fg.SPEC["ltcc"]["batch_size"], fg.SPEC["ltcc"]["num_instances"], fg.SPEC["ltcc"]["steps_per_epoch"], fg.SEED, rank=0)
    sampler.prepare(60)
    train_loader = fg.build_train_loader(cfg, dataset, sampler)
    records: list[dict[str, Any]] = []
    seen_relation_keys: set[tuple[int, int]] = set()
    batches_inspected = 0
    train_iter = iter(train_loader)

    while len(records) < TARGET_RELATIONS:
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        batches_inspected += 1
        entries = fg.batch_entries(batch)
        selected, info = fg.select_gpc_pairs(entries, bank, epoch=60, rank=0, iteration=batches_inspected, seed=fg.SEED)
        if not selected:
            continue
        path_to_index = bank["path_to_index"]
        selected_global = []
        for relation_index, relation in enumerate(selected):
            local_left, local_right, _image_key, state_key = relation
            global_left = int(path_to_index[str(entries[local_left]["image_path"])])
            global_right = int(path_to_index[str(entries[local_right]["image_path"])])
            relation_key = tuple(sorted((global_left, global_right)))
            if relation_key in seen_relation_keys:
                continue
            selected_global.append((relation_index, relation, relation_key, global_left, global_right, state_key))
        for relation_index, relation, relation_key, global_left, global_right, state_key in selected_global:
            if len(records) >= TARGET_RELATIONS:
                break
            endpoint_ids = (global_left, global_right)
            endpoint_neighbors = [int(x) for x in neighbors[global_left, :K]] + [int(x) for x in neighbors[global_right, :K]]
            negative_ids = sorted(set(endpoint_neighbors))
            probe_ids = set(endpoint_ids) | set(int(x) for x in pool[global_left, :POOL_K]) | set(int(x) for x in pool[global_right, :POOL_K])
            for negative in negative_ids:
                probe_ids.add(int(positive_partner[negative]))
                probe_ids.update(int(x) for x in neighbors[negative, :K])
            probe_ids.discard(-1)
            probe_indices = sorted(probe_ids)
            probe_pos = {index: pos for pos, index in enumerate(probe_indices)}
            probe_images, probe_camids = probe_batch(dataset, cfg, probe_indices)
            base_metrics = local_metrics(features[probe_indices], probe_indices, endpoint_ids, neighbors, pool, features)
            base_nearby = nearby_negative_margin(features[probe_indices], probe_indices, negative_ids, neighbors, positive_partner)

            reset_branch(model, optimizer, scheduler, scaler, base_model, base_optimizer, base_scheduler, base_scaler, paired_rng)
            full_result = fg.forward_loss(model, loss_fn, batch, DEVICE, bank, "gpc", 60, 0, batches_inspected, 35760 + batches_inspected, masked_index=None)
            selected_relation_gradient = torch.autograd.grad(full_result["gpc_pair_losses"][relation_index], params, retain_graph=True, allow_unused=True)
            probe_z_grad = feature_probe(model, probe_images, probe_camids, with_grad=True)
            probe_metrics_grad = local_metrics(probe_z_grad, probe_indices, endpoint_ids, neighbors, pool, features)
            nearby_grad = nearby_negative_margin(probe_z_grad, probe_indices, negative_ids, neighbors, positive_partner)
            grad_margin = probe_metrics_grad["local_margin"]
            gM = torch.autograd.grad(grad_margin, params, retain_graph=False, allow_unused=True)
            risk = gradient_dot(gM, selected_relation_gradient)
            del gM, selected_relation_gradient, probe_z_grad, nearby_grad

            if not bool(torch.isfinite(full_result["total"].detach()).item()):
                raise RuntimeError(f"non-finite full loss at relation {relation_key}")
            scaler.scale(full_result["total"]).backward()
            full_bad_grads = any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()) for parameter in params)
            scaler.step(optimizer)
            full_inf = fg.found_inf(scaler, optimizer)
            full_scale_before = float(scaler.get_scale())
            scaler.update()
            full_scale_after = float(scaler.get_scale())
            model.eval()
            with torch.no_grad():
                full_z = feature_probe(model, probe_images, probe_camids, with_grad=False).detach()
            full_metrics = local_metrics(full_z, probe_indices, endpoint_ids, neighbors, pool, features)
            full_nearby = nearby_negative_margin(full_z, probe_indices, negative_ids, neighbors, positive_partner)

            reset_branch(model, optimizer, scheduler, scaler, base_model, base_optimizer, base_scheduler, base_scaler, paired_rng)
            masked_result = fg.forward_loss(model, loss_fn, batch, DEVICE, bank, "gpc", 60, 0, batches_inspected, 35760 + batches_inspected, masked_index=relation_index)
            if not bool(torch.isfinite(masked_result["total"].detach()).item()):
                raise RuntimeError(f"non-finite masked loss at relation {relation_key}")
            scaler.scale(masked_result["total"]).backward()
            masked_bad_grads = any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()) for parameter in params)
            scaler.step(optimizer)
            masked_inf = fg.found_inf(scaler, optimizer)
            masked_scale_before = float(scaler.get_scale())
            scaler.update()
            masked_scale_after = float(scaler.get_scale())
            model.eval()
            with torch.no_grad():
                masked_z = feature_probe(model, probe_images, probe_camids, with_grad=False).detach()
            masked_metrics = local_metrics(masked_z, probe_indices, endpoint_ids, neighbors, pool, features)
            masked_nearby = nearby_negative_margin(masked_z, probe_indices, negative_ids, neighbors, positive_partner)

            base_endpoint_margins = [float(value.detach().cpu()) for value in base_metrics["endpoint_margins"]]
            full_endpoint_margins = [float(value.detach().cpu()) for value in full_metrics["endpoint_margins"]]
            masked_endpoint_margins = [float(value.detach().cpu()) for value in masked_metrics["endpoint_margins"]]
            initially_correct = [value > 0.0 for value in base_endpoint_margins]
            reversals_full = sum(correct and after <= 0.0 for correct, after in zip(initially_correct, full_endpoint_margins))
            reversals_masked = sum(correct and after <= 0.0 for correct, after in zip(initially_correct, masked_endpoint_margins))
            positive_before = float(base_metrics["positive_similarity"].detach().cpu())
            positive_full = float(full_metrics["positive_similarity"].detach().cpu())
            positive_masked = float(masked_metrics["positive_similarity"].detach().cpu())
            margin_before = float(base_metrics["local_margin"].detach().cpu())
            margin_full = float(full_metrics["local_margin"].detach().cpu())
            margin_masked = float(masked_metrics["local_margin"].detach().cpu())
            nearby_before = float(base_nearby.detach().cpu())
            nearby_full = float(full_nearby.detach().cpu())
            nearby_masked = float(masked_nearby.detach().cpu())
            row = {
                "candidate_index": len(records),
                "batch_iteration": batches_inspected,
                "relation_index_in_batch": relation_index,
                "global_left": global_left,
                "global_right": global_right,
                "pid": int(dataset.train[global_left]["pid"]),
                "clothes_left": int(dataset.train[global_left]["clothes_id"]),
                "clothes_right": int(dataset.train[global_right]["clothes_id"]),
                "clothes_condition": "same_clothes" if int(dataset.train[global_left]["clothes_id"]) == int(dataset.train[global_right]["clothes_id"]) else "cross_clothes",
                "relation_key": f"{relation_key[0]}:{relation_key[1]}",
                "state_key": f"{state_key[0]}:{state_key[1]}:{state_key[2]}",
                "selected_count_in_batch": len(selected),
                "gpc_pair_loss": float(full_result["gpc_pair_losses"][relation_index].detach().cpu()),
                "gpc_mean_full": float(full_result["gpc"].detach().cpu()),
                "gpc_mean_masked": float(masked_result["gpc"].detach().cpu()),
                "lambda_gpc": LAMBDA_GPC,
                "positive_similarity_before": positive_before,
                "positive_similarity_full": positive_full,
                "positive_similarity_masked": positive_masked,
                "gpc_positive_consistency_gain_full_minus_masked": positive_full - positive_masked,
                "local_margin_before": margin_before,
                "local_margin_full": margin_full,
                "local_margin_masked": margin_masked,
                "delta_M_full_minus_masked": margin_full - margin_masked,
                "harmful_delta_M_negative": margin_full - margin_masked < 0.0,
                "hard_negative_similarity_before": float(base_metrics["hard_negative_similarity"].detach().cpu()),
                "hard_negative_similarity_full": float(full_metrics["hard_negative_similarity"].detach().cpu()),
                "hard_negative_similarity_masked": float(masked_metrics["hard_negative_similarity"].detach().cpu()),
                "hard_negative_change_full_minus_masked": float(full_metrics["hard_negative_similarity"].detach().cpu() - masked_metrics["hard_negative_similarity"].detach().cpu()),
                "nearby_negative_local_margin_before": nearby_before,
                "nearby_negative_local_margin_full": nearby_full,
                "nearby_negative_local_margin_masked": nearby_masked,
                "nearby_negative_local_margin_change_full_minus_masked": nearby_full - nearby_masked,
                "new_pool_entries_full": int(full_metrics["pool_new_entries"]),
                "new_pool_entries_masked": int(masked_metrics["pool_new_entries"]),
                "ordering_reversal_rate_full": reversals_full / max(sum(initially_correct), 1),
                "ordering_reversal_rate_masked": reversals_masked / max(sum(initially_correct), 1),
                "initially_correct_endpoint_count": sum(initially_correct),
                "full_inf": full_inf,
                "masked_inf": masked_inf,
                "full_bad_gradients": full_bad_grads,
                "masked_bad_gradients": masked_bad_grads,
                "full_scale_before": full_scale_before,
                "full_scale_after": full_scale_after,
                "masked_scale_before": masked_scale_before,
                "masked_scale_after": masked_scale_after,
                "full_loss": float(full_result["total"].detach().cpu()),
                "masked_loss": float(masked_result["total"].detach().cpu()),
                "base_loss_full": float(full_result["base_total"].detach().cpu()),
                "base_loss_masked": float(masked_result["base_total"].detach().cpu()),
                **risk,
            }
            records.append(row)
            seen_relation_keys.add(relation_key)
            print(json.dumps({"progress": len(records), "candidate": row}, ensure_ascii=False, sort_keys=True), flush=True)
            del full_result, masked_result, full_z, masked_z, probe_images, probe_camids
            torch.cuda.empty_cache()

    if len(records) != TARGET_RELATIONS:
        raise RuntimeError(f"only {len(records)} unique real GPC relations collected after {batches_inspected} batches")

    write_csv(OUT_ROOT / "NDP_DYNAMIC_RISK_RAW.csv", records)
    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "rank_state": str(RANK_STATE),
        "rank_state_sha256": sha256_file(RANK_STATE),
        "dataset": "LTCC",
        "split": "train_only",
        "relations_checked": len(records),
        "batches_inspected": batches_inspected,
        "K": K,
        "pool_K": POOL_K,
        "mean_delta_M": float(np.mean([row["delta_M_full_minus_masked"] for row in records])),
        "harmful_delta_M_count": int(sum(bool(row["harmful_delta_M_negative"]) for row in records)),
        "harmful_delta_M_rate": float(np.mean([bool(row["harmful_delta_M_negative"]) for row in records])),
        "mean_positive_gain_full_minus_masked": float(np.mean([row["gpc_positive_consistency_gain_full_minus_masked"] for row in records])),
        "mean_hard_negative_change_full_minus_masked": float(np.mean([row["hard_negative_change_full_minus_masked"] for row in records])),
        "mean_nearby_negative_local_margin_change_full_minus_masked": float(np.mean([row["nearby_negative_local_margin_change_full_minus_masked"] for row in records])),
        "same_clothes_count": int(sum(row["clothes_condition"] == "same_clothes" for row in records)),
        "cross_clothes_count": int(sum(row["clothes_condition"] == "cross_clothes" for row in records)),
        "full_amp_inf_steps": int(sum(row["full_inf"] > 0 for row in records)),
        "masked_amp_inf_steps": int(sum(row["masked_inf"] > 0 for row in records)),
        "full_bad_gradient_steps": int(sum(row["full_bad_gradients"] for row in records)),
        "masked_bad_gradient_steps": int(sum(row["masked_bad_gradients"] for row in records)),
        "optimizer_step_protocol": "restored SGD momentum state; GradScaler state restored; scheduler restored but not advanced; one full and one masked virtual update per relation",
        "neighborhood_protocol": "cosine on E60 train-only features; self and same-PID negatives excluded; K=10 fixed neighborhood; pool_K=20 used only for entrant telemetry",
        "no_query_gallery": True,
        "checkpoint_written": False,
        "full_training_started": False,
    }
    write_json(OUT_ROOT / "NDP_DYNAMIC_RISK_SUMMARY.json", summary)
    write_csv(OUT_ROOT / "NDP_DYNAMIC_RISK_GRADIENT.csv", records, fields=[
        "candidate_index", "relation_key", "pid", "clothes_condition", "R_grad", "gradient_dot_raw",
        "gradient_cosine", "grad_M_norm", "grad_GPC_norm", "delta_M_full_minus_masked",
        "harmful_delta_M_negative", "local_margin_before", "positive_similarity_before",
        "hard_negative_similarity_before",
    ])
    write_json(OUT_ROOT / "NDP_DYNAMIC_RISK_RUN_MANIFEST.json", {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "rank_state": str(RANK_STATE),
        "rank_state_sha256": sha256_file(RANK_STATE),
        "relation_source": str(RELATION_SOURCE),
        "relation_source_sha256": sha256_file(RELATION_SOURCE),
        "dataset_train_images": len(dataset.train),
        "dataset_train_pids": dataset.num_train_pids,
        "dataset_train_cameras": dataset.num_camera,
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(DEVICE),
        "features_sha256": sha256_file(OUT_ROOT / "LTCC_TRAIN_FEATURES_E60_I3.npy"),
        "neighborhood_sha256": sha256_file(OUT_ROOT / "LTCC_FIXED_NEIGHBORHOODS_K10.npz"),
        "selector": "same PID, i<j; matched min(cross,same,id); phase7i_relation_specificity SHA-256 selection",
        "candidate_target": TARGET_RELATIONS,
        "candidate_unique_key": "canonical train image-index pair",
        "gradient_formula": "positive_part(sum_theta g_M(theta)^T g_GPC_ij(theta)); SGD small-step first-order proxy, not exact Adam update",
        "rng_compatibility": "historical Python/NumPy/CPU RNG restored; historical PPU CUDA RNG tensor was incompatible in size, so one current-runtime CUDA snapshot was restored identically for both paired branches",
        "summary": summary,
    })


if __name__ == "__main__":
    run()
