#!/usr/local/bin/python3
"""PHASE 3C: formal PRCC baseline/full reproduction.

The two formal runs in this driver use the paper-faithful PRCC semantic
topology ``NOBIO_INDEX=[1, 2]``.  The released ``[1]`` configuration and the
official packaged ``[2, 3]`` configuration are audited as provenance only.
No Phase-2 numerical candidate is selectable from this driver.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_dataloader, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from processor import do_train  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.logger import setup_logger  # noqa: E402
from utils.metrics import R1_mAP_eval  # noqa: E402


DATA_ROOT = Path("/home/data")
PRCC_ROOT = DATA_ROOT / "prcc"
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/PRCC"
TEXT_MODEL_DIR = TEXT_ROOT / "EVA02-CLIP-L-14"
TRAIN_NPZ = TEXT_MODEL_DIR / "train.npz"
VAL_NPZ = TEXT_MODEL_DIR / "val.npz"
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
RELEASED_CONFIG = ROOT / "configs/prcc/eva02_l_bio.yml"
OFFICIAL_ROOT = DATA_ROOT / "DIFFER_OfficialWeights/provenance/PRCC"
OFFICIAL_CONFIG = OFFICIAL_ROOT / "config.yml"
OFFICIAL_CHECKPOINT = OFFICIAL_ROOT / "eva02_l_bio_best.pth"
OFFICIAL_LOG = OFFICIAL_ROOT / "train_log.txt"
EVA_WEIGHT = Path(
    "/root/.cache/huggingface/hub/models--timm--"
    "eva02_large_patch14_clip_224.merged2b_s4b_b131k/snapshots/"
    "bf4190eb65dd5204ffb03e980108beb1200e0873/open_clip_pytorch_model.bin"
)
EVA_REVISION = "bf4190eb65dd5204ffb03e980108beb1200e0873"
OUTPUT_ROOT = ROOT / "outputs/reproduction_phase3/PRCC"
WORLD_SIZE_REQUIRED = 2
SEED = 1234
INITIAL_SCALE = 512
EXPECTED_NUM_INSTANCES = 4
EXPECTED_ITERATIONS = 551
CLIP_DIM = 768
PAPER_NOBIO = [1, 2]
RELEASED_NOBIO = [1]
PACKAGED_NOBIO = [2, 3]
FACTOR_NAMES = {
    0: "Biometrics",
    1: "Hairstyle",
    2: "Clothing",
    3: "Pose",
    4: "Interaction",
    5: "Environment",
}

SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "PRCC Baseline / ID only",
        "losses": ["ce", "triplet"],
        "paper_rank1": 65.1,
        "paper_map": 63.0,
    },
    "full": {
        "label": "PRCC Full DIFFER / paper-faithful",
        "losses": ["ce", "triplet", "clipBio", "clipBioReverse"],
        "paper_rank1": 68.5,
        "paper_map": 64.7,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def command_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def local_rank_from_args(value: int | None) -> int:
    return int(value) if value is not None else int(os.environ.get("LOCAL_RANK", "0"))


def run_dir_for(kind: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if kind == "baseline":
        return (OUTPUT_ROOT / "baseline/run1").resolve()
    if kind == "full":
        return (OUTPUT_ROOT / "full_paper_faithful/run1").resolve()
    return (OUTPUT_ROOT / kind / "run1").resolve()


def parse_indices(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.replace("*", ",").split(",") if item.strip()]
    return [int(item) for item in value]


def make_config(kind: str, run_dir: Path, nobio: list[int] | None = None):
    if kind not in SPECS:
        raise ValueError(kind)
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(RELEASED_CONFIG))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = list(SPECS[kind]["losses"])
    local_cfg.MODEL.DIST_TRAIN = True
    local_cfg.MODEL.CLIP_DIM = CLIP_DIM
    local_cfg.MODEL.LAST_LAYER = "clipFc"
    local_cfg.DATA.ROOT = str(DATA_ROOT)
    local_cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-L-14"
    local_cfg.DATA.BIO_INDEX = "0"
    local_cfg.DATA.NOBIO_INDEX = list(PAPER_NOBIO if nobio is None else nobio)
    local_cfg.DATA.SUMMERY_TEXT = True
    local_cfg.DATA.BATCH_SIZE = 16
    # This value is read from the released PRCC YAML and is deliberately not
    # inferred from the image count.
    local_cfg.DATA.NUM_INSTANCES = EXPECTED_NUM_INSTANCES
    local_cfg.DATA.IMG_HEIGHT = 224
    local_cfg.DATA.IMG_WIDTH = 224
    local_cfg.SOLVER.SEED = SEED
    local_cfg.SOLVER.MAX_EPOCHS = 60
    local_cfg.OUTPUT_DIR = str(run_dir)
    local_cfg.OUTPUT_NAME = f"PHASE3C_PRCC_{kind.upper()}_RUN1"
    local_cfg.TEST.WEIGHT = ""
    local_cfg.freeze()
    return local_cfg


def family_for(name: str) -> str:
    if name.startswith("head."):
        return "id_classifier"
    if name.startswith(("head_clip_bioReverse.",)):
        return "reverse_heads"
    if name.startswith(("head_clip_bio.",)):
        return "bio_projection"
    if name.startswith(("head_bio.", "norm_bio.")):
        return "bio_transformer"
    return "eva_backbone"


def parameter_coverage(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    name_by_id: dict[int, str] = {}
    trainable_names: list[str] = []
    family_counts: Counter[str] = Counter()
    family_numel: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        name_by_id[id(parameter)] = name
        if parameter.requires_grad:
            trainable_names.append(name)
            family_counts[family_for(name)] += 1
            family_numel[family_for(name)] += parameter.numel()

    occurrences: Counter[int] = Counter()
    group_summary: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name_by_id.get(id(parameter), "<unknown>") for parameter in group["params"]]
        occurrences.update(id(parameter) for parameter in group["params"])
        grouped_families = Counter(family_for(name) for name in names if name != "<unknown>")
        group_summary.append({
            "index": index,
            "parameter_count": len(names),
            "families": dict(sorted(grouped_families.items())),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        })

    id_by_name = {name: id(parameter) for name, parameter in model.named_parameters()}
    missing = [name for name in trainable_names if occurrences[id_by_name[name]] == 0]
    duplicates = [name for name in trainable_names if occurrences[id_by_name[name]] != 1]
    unknown = [
        name for group in group_summary for name in []
    ]
    lr_wd_counts: Counter[tuple[float, float]] = Counter(
        (item["lr"], item["weight_decay"]) for item in group_summary
    )
    return {
        "trainable_parameter_count": len(trainable_names),
        "trainable_parameter_numel": int(sum(family_numel.values())),
        "trainable_by_family": dict(sorted(family_counts.items())),
        "trainable_numel_by_family": dict(sorted(family_numel.items())),
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_parameter_count": sum(item["parameter_count"] for item in group_summary),
        "missing_trainable_count": len(missing),
        "duplicate_or_miscovered_count": len(duplicates),
        "unknown_optimizer_parameters": unknown,
        "missing_trainable_names": missing,
        "duplicate_or_miscovered_names": duplicates,
        "exactly_once": not missing and not duplicates and not unknown,
        "lr_weight_decay_counts": [
            {"lr": lr, "weight_decay": wd, "groups": count}
            for (lr, wd), count in sorted(lr_wd_counts.items())
        ],
        "groups": group_summary,
    }


def module_grad_stats(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, Any]:
    parameter_count = requires_grad_count = grad_exists = nonzero_grad = 0
    nan_count = inf_count = 0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        parameter_count += 1
        if not parameter.requires_grad:
            continue
        requires_grad_count += 1
        if parameter.grad is None:
            continue
        grad_exists += 1
        gradient = parameter.grad.detach()
        nan_count += int(torch.isnan(gradient).sum().item())
        inf_count += int(torch.isinf(gradient).sum().item())
        finite = gradient[torch.isfinite(gradient)]
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
            nonzero_grad += int(bool((finite.abs() > 0).any().item()))
    return {
        "parameter_count": parameter_count,
        "requires_grad_count": requires_grad_count,
        "grad_exists_count": grad_exists,
        "nonzero_grad_count": nonzero_grad,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_abs_finite_grad": max_abs,
    }


class ScalarCapture:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.values[str(tag)] = float(value)


def architecture_audit(kind: str, cfg: Any, model: torch.nn.Module) -> dict[str, Any]:
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    reverse_count = len(reverse_modules) if reverse_modules is not None else 0
    return {
        "kind": kind,
        "active_losses": list(cfg.MODEL.LOSS_TYPE),
        "bio_contrastive_loss_enabled": "clipBio" in cfg.MODEL.LOSS_TYPE,
        "reverse_loss_enabled": "clipBioReverse" in cfg.MODEL.LOSS_TYPE,
        "bio_head_exists": hasattr(model, "head_bio") and hasattr(model, "head_clip_bio"),
        "reverse_head_count": reverse_count,
        "expected_reverse_head_count": 0 if kind == "baseline" else 2,
        "clip_dim": int(cfg.MODEL.CLIP_DIM),
        "nobio_index": parse_indices(cfg.DATA.NOBIO_INDEX),
        "nobio_factor_names": [FACTOR_NAMES[i] for i in parse_indices(cfg.DATA.NOBIO_INDEX)],
        "reverse_factor_axis": "image_feature_reverse[:, i]",
        "reverse_loss_accumulation": "loss_clip_reverse += factor_loss; loss += loss_clip_reverse",
        "grl_alpha": 1.0,
        "summary_bio_required": bool(cfg.DATA.SUMMERY_TEXT),
        "model_parameter_numel": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_state_tensor_count": len(model.state_dict()),
        "modules": {
            "head": hasattr(model, "head"),
            "head_bio": hasattr(model, "head_bio"),
            "norm_bio": hasattr(model, "norm_bio"),
            "head_clip_bio": hasattr(model, "head_clip_bio"),
            "head_clip_bioReverse": hasattr(model, "head_clip_bioReverse"),
        },
    }


def text_runtime_audit(dataset: Any, cfg: Any) -> dict[str, Any]:
    with np.load(TRAIN_NPZ, allow_pickle=True) as archive:
        train_data = np.asarray(archive["data"])
        train_metadata = [item.decode() if isinstance(item, bytes) else str(item) for item in archive["metadata"]]
    with np.load(VAL_NPZ, allow_pickle=True) as archive:
        val_data_shape = list(archive["data"].shape)
        val_metadata_shape = list(archive["metadata"].shape)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    first = dataset.train[0]
    image_path = Path(first["image_path"])
    key = f"{image_path.parent.name}/{image_path.stem}"
    index = {item: i for i, item in enumerate(train_metadata)}
    if key not in index:
        raise RuntimeError(f"stable PRCC text key missing: {key}")
    factor_count = train_data.shape[0] // len(train_metadata)
    raw = train_data[index[key] * factor_count : (index[key] + 1) * factor_count]
    runtime = np.asarray(first["caption_feature"])
    raw_selected = raw[np.asarray(PAPER_NOBIO, dtype=int)]
    summary_pid = image_path.parent.name
    if summary_pid not in summary:
        summary_pid = str(int(summary_pid))
    raw_bio = np.asarray(summary[summary_pid]["ft_EVA02-CLIP-L-14"])
    bio_error = float(np.max(np.abs(runtime[0].astype(np.float32) - raw_bio.astype(np.float32))))
    selected_error = float(np.max(np.abs(runtime[1:].astype(np.float32) - raw_selected.astype(np.float32))))
    train_image_keys = {
        f"{Path(item['image_path']).parent.name}/{Path(item['image_path']).stem}"
        for item in dataset.train
    }
    return {
        "train_npz": str(TRAIN_NPZ),
        "train_npz_sha256": sha256_file(TRAIN_NPZ),
        "train_data_shape": list(train_data.shape),
        "train_metadata_shape": [len(train_metadata)],
        "train_metadata_unique": len(set(train_metadata)),
        "val_npz": str(VAL_NPZ),
        "val_npz_sha256": sha256_file(VAL_NPZ),
        "val_data_shape": val_data_shape,
        "val_metadata_shape": val_metadata_shape,
        "summary_json": str(SUMMARY_JSON),
        "summary_json_sha256": sha256_file(SUMMARY_JSON),
        "summary_entry_count": len(summary),
        "stable_key_format": "PID/filename-stem",
        "first_stable_key": key,
        "basename_duplicate_count_train": int(len(dataset.train) - len({Path(x['image_path']).stem for x in dataset.train})),
        "dataset_text_key_set_match_train": train_image_keys == set(train_metadata),
        "runtime_caption_shape": list(runtime.shape),
        "runtime_caption_dim": int(runtime.shape[-1]),
        "raw_factor_count": int(factor_count),
        "summary_bio_runtime_max_abs_error": bio_error,
        "summary_bio_runtime_match_tolerance_1e-3": bio_error <= 1e-3,
        "selected_nonbio_runtime_max_abs_error": selected_error,
        "selected_nonbio_runtime_match": selected_error == 0.0,
        "summary_bio_source": "identity-level summarized biometric JSON, factor 0",
        "nonbio_source": "image-level NPZ text, factors 1 Hair and 2 Clothing",
        "clip_dim_assert": int(runtime.shape[-1]) == CLIP_DIM,
        "summary_text_enabled": bool(cfg.DATA.SUMMERY_TEXT),
    }


def dataset_audit(dataset: Any, train_loader: Any) -> dict[str, Any]:
    return {
        "root": str(PRCC_ROOT),
        "train_images": len(dataset.train),
        "train_ids": int(dataset.num_train_pids),
        "train_clothes": int(dataset.num_train_clothes),
        "cameras": int(dataset.num_camera),
        "val_images": 5002,
        "query_changed_images_C": int(dataset.num_query_imgs_diff),
        "query_same_images_B": int(dataset.num_query_imgs_same),
        "gallery_images_A": int(dataset.num_gallery_imgs),
        "primary_evaluation": "C -> A (changed clothes)",
        "primary_query_count": int(dataset.num_query_imgs_diff),
        "primary_gallery_count": int(dataset.num_gallery_imgs),
        "sampler": str("DistributedRandomIdentitySampler"),
        "batch_size_per_ppu": int(train_loader.batch_size),
        "num_instances": EXPECTED_NUM_INSTANCES,
        "iterations_per_epoch": int(len(train_loader)),
        "epochs": 60,
    }


def run_smoke(
    kind: str,
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    train_loader: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    center_criterion: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    batch = next(iter(train_loader))
    writer = ScalarCapture()
    optimizer.zero_grad()
    center_criterion.zero_grad()
    grl_records: list[dict[str, Any]] = []

    def grl_hook(module, grad_input, grad_output):
        if not grad_input or grad_input[0] is None or not grad_output or grad_output[0] is None:
            grl_records.append({"valid": False, "reason": "missing gradient tuple"})
            return
        incoming = grad_input[0].detach()
        outgoing = grad_output[0].detach()
        error = incoming + float(module.alpha) * outgoing
        grl_records.append({
            "valid": True,
            "alpha": float(module.alpha),
            "max_abs_error": float(error.abs().max().item()),
            "allclose": bool(torch.allclose(incoming, -float(module.alpha) * outgoing)),
        })

    hooks = []
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    if kind == "full":
        if reverse_modules is None or len(reverse_modules) != 2:
            raise RuntimeError(f"full PRCC expected two reverse heads, got {reverse_modules}")
        for reverse_head in reverse_modules:
            hooks.append(reverse_head.grl.register_full_backward_hook(grl_hook))
    try:
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        total_loss = loss_fn(
            score,
            feat,
            batch["pid"],
            batch["camid"],
            batch["caption_feature"],
            clothes_ids=batch["clothes_id"],
            train_writer=writer,
            step=1,
        )
        expected_tags = ["loss/id", "loss/triplet"]
        if kind == "full":
            expected_tags += ["loss/clip_bio_0", "loss/clip_bio_reverse_0", "loss/clip_bio_reverse_1"]
        forbidden_tags = [] if kind == "full" else ["loss/clip_bio_0", "loss/clip_bio_reverse_0", "loss/clip_bio_reverse_1"]
        component_finite = {
            tag: tag in writer.values and np.isfinite(writer.values[tag])
            for tag in expected_tags
        }
        forbidden_absent = all(tag not in writer.values for tag in forbidden_tags)
        total_finite = bool(torch.isfinite(total_loss.detach()).item())
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients = {
            "eva_backbone": module_grad_stats(model, ("patch_embed.", "pos_embed", "cls_token", "camera_embed", "rope.", "blocks.")),
            "id_classifier": module_grad_stats(model, ("head.",)),
            "bio_transformer": module_grad_stats(model, ("head_bio.", "norm_bio.")),
            "bio_projection": module_grad_stats(model, ("head_clip_bio.",)),
            "reverse_hair": module_grad_stats(model, ("head_clip_bioReverse.0.",)),
            "reverse_clothing": module_grad_stats(model, ("head_clip_bioReverse.1.",)),
        }
        critical = ["eva_backbone", "id_classifier", "bio_transformer"]
        if kind == "full":
            critical += ["bio_projection", "reverse_hair", "reverse_clothing"]
        critical_ok = all(
            gradients[name]["grad_exists_count"] > 0
            and gradients[name]["nonzero_grad_count"] > 0
            and gradients[name]["nan_count"] == 0
            and gradients[name]["inf_count"] == 0
            for name in critical
        )
        if kind == "baseline":
            inactive_modules = {
                "bio_projection_grad_absent": gradients["bio_projection"]["grad_exists_count"] == 0,
                "reverse_heads_absent": reverse_modules is None,
            }
            grl_pass = True
        else:
            inactive_modules = {}
            grl_pass = len(grl_records) == 2 and all(
                item.get("valid") and item.get("allclose") for item in grl_records
            )

        before = model.head.weight.detach().clone()
        scaler.step(optimizer)
        found_inf_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = sum(float(value.item()) for value in found_inf_state.values())
        scaler.update()
        after = model.head.weight.detach()
        step_ok = found_inf == 0.0 and bool(torch.isfinite(after).all().item())
        return {
            "batch_size": int(batch["image"].shape[0]),
            "image_shape": list(batch["image"].shape),
            "global_feature_shape": list(feat[0].shape),
            "id_score_shape": list(score["cls_score"].shape),
            "bio_score_shape": list(score["clip_bio_score"].shape) if "clip_bio_score" in score else None,
            "reverse_score_shape": list(score["clip_bio_reverse_score"].shape) if "clip_bio_reverse_score" in score else None,
            "text_feature_shape": list(batch["caption_feature"].shape),
            "active_writer_scalars": dict(sorted(writer.values.items())),
            "expected_tags_present": all(tag in writer.values for tag in expected_tags),
            "forbidden_tags_absent": forbidden_absent,
            "component_finite": component_finite,
            "total_loss": float(total_loss.detach().double().cpu()),
            "total_loss_dtype": str(total_loss.dtype),
            "total_finite": total_finite,
            "gradients": gradients,
            "critical_gradients_finite_nonzero": critical_ok,
            "inactive_module_expectations": inactive_modules,
            "grl_records": grl_records,
            "grl_direction_pass": bool(grl_pass),
            "scaler_initial_scale": INITIAL_SCALE,
            "found_inf": found_inf,
            "optimizer_step_executed": bool(step_ok),
            "parameter_delta_head_weight_max_abs": float((after - before).abs().max().item()),
            "passed": bool(
                total_finite
                and all(component_finite.values())
                and all(tag in writer.values for tag in expected_tags)
                and forbidden_absent
                and critical_ok
                and grl_pass
                and step_ok
            ),
        }
    finally:
        for hook in hooks:
            hook.remove()
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass


def check_loaded_pretrained(model: torch.nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    finite = all(bool(torch.isfinite(value).all().item()) for value in state.values() if torch.is_tensor(value))
    sample_names = [
        "blocks.0.attn.q_proj.weight",
        "blocks.23.attn.q_proj.weight",
        "head_bio.attn.q_proj.weight",
    ]
    samples = {
        name: {
            "present": name in state,
            "shape": list(state[name].shape) if name in state else None,
            "finite": bool(torch.isfinite(state[name]).all().item()) if name in state else False,
        }
        for name in sample_names
    }
    return {
        "source_path": str(EVA_WEIGHT),
        "source_exists": EVA_WEIGHT.exists(),
        "source_sha256": sha256_file(EVA_WEIGHT) if EVA_WEIGHT.exists() else None,
        "source_revision": EVA_REVISION,
        "model_state_tensor_count": len(state),
        "model_state_all_finite": finite,
        "loaded_parameter_samples": samples,
    }


def preflight(kind: str, run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3C requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"expected exactly {WORLD_SIZE_REQUIRED} visible PPU devices, got {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        cfg = make_config(kind, run_dir)
        set_seed(SEED)
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        train_loader, _same, _diff, _gallery, dataset, _sampler, _val, _val_same = build_dataloader(cfg)
        if int(cfg.DATA.NUM_INSTANCES) != EXPECTED_NUM_INSTANCES:
            raise RuntimeError(f"released PRCC NUM_INSTANCES changed: {cfg.DATA.NUM_INSTANCES}")
        if len(train_loader) != EXPECTED_ITERATIONS:
            raise RuntimeError(f"expected {EXPECTED_ITERATIONS} PRCC iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        architecture = architecture_audit(kind, cfg, model)
        if architecture["reverse_head_count"] != architecture["expected_reverse_head_count"]:
            raise RuntimeError(f"reverse topology audit failed: {architecture}")
        loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, _optimizer_center = make_optimizer(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
        scheduler.step(cfg.TRAIN.START_EPOCH)
        coverage = parameter_coverage(model, optimizer)
        if not coverage["exactly_once"]:
            raise RuntimeError(f"optimizer coverage failed: {coverage}")
        model.to(local_rank)
        model.train()
        device = torch.device(f"cuda:{local_rank}")
        text_audit = text_runtime_audit(dataset, cfg)
        if not text_audit["dataset_text_key_set_match_train"] or not text_audit["clip_dim_assert"]:
            raise RuntimeError(f"PRCC text alignment/CLIP dimension audit failed: {text_audit}")
        if not text_audit["summary_bio_runtime_match_tolerance_1e-3"]:
            raise RuntimeError(f"Summary Bio runtime audit failed: {text_audit}")
        smoke = run_smoke(kind, cfg, model, dataset, train_loader, loss_fn, optimizer, center_criterion, device)
        rank_report = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "kind": kind,
            "dataset": dataset_audit(dataset, train_loader),
            "architecture": architecture,
            "pretrained": check_loaded_pretrained(model),
            "text": text_audit,
            "optimizer_coverage": coverage,
            "smoke": smoke,
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE_REQUIRED
        torch.distributed.all_gather_object(gathered, rank_report)
        if local_rank == 0:
            reports = [item for item in gathered if item is not None]
            passed = all(item["smoke"]["passed"] for item in reports)
            payload = {
                "phase": "3C",
                "mode": "formal_paper_faithful_preflight",
                "created_utc": utc_now(),
                "kind": kind,
                "track": "A: released-code / paper-faithful formal path",
                "numerical_candidate": None,
                "formal_nobio_index": PAPER_NOBIO,
                "released_nobio_index_audited_only": RELEASED_NOBIO,
                "official_packaged_nobio_index_audited_only": PACKAGED_NOBIO,
                "seed": SEED,
                "amp_initial_scale": INITIAL_SCALE,
                "world_size": WORLD_SIZE_REQUIRED,
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "git_status": command_output("git", "status", "--short"),
                "config_snapshot": cfg.dump(),
                "rank_reports": reports,
                "passed": passed,
            }
            output = run_dir / "preflight.json"
            output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
            print("PHASE3C_PREFLIGHT", json.dumps({
                "kind": kind,
                "nobio_index": PAPER_NOBIO,
                "num_instances": EXPECTED_NUM_INSTANCES,
                "iterations_per_epoch": EXPECTED_ITERATIONS,
                "clip_dim": CLIP_DIM,
                "amp_initial_scale": INITIAL_SCALE,
                "optimizer_exactly_once": all(item["optimizer_coverage"]["exactly_once"] for item in reports),
                "smoke_passed_all_ranks": passed,
                "output": str(output),
            }, sort_keys=True), flush=True)
    finally:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def yaml_load(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _metadata(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as archive:
        data = np.asarray(archive["data"])
        metadata = [item.decode() if isinstance(item, bytes) else str(item) for item in archive["metadata"]]
    return data, metadata


def official_checkpoint_audit() -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(OFFICIAL_ROOT),
        "checkpoint_available": OFFICIAL_CHECKPOINT.exists(),
        "config_available": OFFICIAL_CONFIG.exists(),
        "train_log_available": OFFICIAL_LOG.exists(),
    }
    if OFFICIAL_CONFIG.exists():
        config = yaml_load(OFFICIAL_CONFIG)
        result["packaged_config"] = {
            "nobio_index": config.get("DATA", {}).get("NOBIO_INDEX"),
            "batch_size": config.get("DATA", {}).get("BATCH_SIZE"),
            "num_instances": config.get("DATA", {}).get("NUM_INSTANCES"),
            "clip_dim": config.get("MODEL", {}).get("CLIP_DIM"),
            "loss_type": config.get("MODEL", {}).get("LOSS_TYPE"),
            "summary_bio": config.get("DATA", {}).get("SUMMERY_TEXT"),
        }
    if OFFICIAL_CHECKPOINT.exists():
        state = torch.load(OFFICIAL_CHECKPOINT, map_location="cpu", weights_only=True)
        keys = list(state)
        normalized = {key.replace("module.", "", 1): value for key, value in state.items()}
        def shapes(prefix: str) -> list[dict[str, Any]]:
            return [
                {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in normalized.items()
                if key.startswith(prefix) and torch.is_tensor(value)
            ]
        reverse = shapes("head_clip_bioReverse.")
        result["checkpoint"] = {
            "path": str(OFFICIAL_CHECKPOINT),
            "bytes": OFFICIAL_CHECKPOINT.stat().st_size,
            "sha256": sha256_file(OFFICIAL_CHECKPOINT),
            "raw_key_count": len(keys),
            "normalized_tensor_count": len(normalized),
            "classifier": shapes("head."),
            "camera_embed": shapes("camera_embed"),
            "bio_head": shapes("head_bio."),
            "bio_projection": shapes("head_clip_bio."),
            "reverse_heads": reverse,
            "reverse_head_count": len({key.split(".")[1] for key in normalized if key.startswith("head_clip_bioReverse.")}),
            "reverse_bias_keys": [key for key in normalized if key.startswith("head_clip_bioReverse.") and key.endswith("bias")],
            "trainable_numel": int(sum(value.numel() for value in normalized.values() if torch.is_tensor(value))),
        }
    if OFFICIAL_LOG.exists():
        lines = OFFICIAL_LOG.read_text(errors="replace").splitlines()
        epoch_metrics: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for line in lines:
            match = re.search(r"Epoch\[(\d+)\]", line)
            if match:
                current["epoch"] = int(match.group(1))
            match = re.search(r"CMC curve, Rank-1\s*:([0-9.]+)%", line)
            if match:
                current["cc_rank1"] = float(match.group(1))
            match = re.search(r"mAP-:([0-9.]+)%", line)
            if match and "cc_rank1" in current:
                current["cc_mAP"] = float(match.group(1))
                epoch_metrics.append(current)
                current = {}
        best_matches = re.findall(r"==> Best Rank-1 ([0-9.]+)%, achieved at epoch (\d+)", OFFICIAL_LOG.read_text(errors="replace"))
        best = best_matches[-1] if best_matches else None
        result["train_log"] = {
            "path": str(OFFICIAL_LOG),
            "bytes": OFFICIAL_LOG.stat().st_size,
            "iterations_per_epoch_markers": sorted(set(int(x) for x in re.findall(r"Iteration\[\d+/(\d+)\]", OFFICIAL_LOG.read_text(errors="replace")))),
            "best_logged": {"rank1_percent": float(best[0]), "epoch": int(best[1])} if best else None,
            "last_changed_clothes_metrics": epoch_metrics[-1] if epoch_metrics else None,
        }
    result["formal_training_use"] = "provenance diagnostic only; not used to initialize either formal run"
    return result


def shared_audit() -> None:
    if not PRCC_ROOT.exists():
        raise RuntimeError(f"missing PRCC dataset: {PRCC_ROOT}")
    train_data, train_metadata = _metadata(TRAIN_NPZ)
    val_data, val_metadata = _metadata(VAL_NPZ)
    dataset_images = {
        subset: [str(path) for path in (PRCC_ROOT / "rgb" / subset).glob("*/*.jpg")]
        for subset in ("train", "val")
    }
    image_keys = {
        subset: {f"{Path(path).parent.name}/{Path(path).stem}" for path in paths}
        for subset, paths in dataset_images.items()
    }
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    packaged = yaml_load(OFFICIAL_CONFIG) if OFFICIAL_CONFIG.exists() else {}
    released = yaml_load(RELEASED_CONFIG)
    audit = {
        "phase": "3C",
        "created_utc": utc_now(),
        "track": "A: released-code / paper-faithful formal reproduction",
        "numerical_candidate": None,
        "protocol": {
            "paper_nobio_index": PAPER_NOBIO,
            "paper_factor_names": [FACTOR_NAMES[i] for i in PAPER_NOBIO],
            "released_head_nobio_index": RELEASED_NOBIO,
            "official_packaged_nobio_index": PACKAGED_NOBIO,
            "formal_choice": PAPER_NOBIO,
            "reason": "paper-faithful protocol; released/artifact configurations are not formal search arms",
            "summary_bio": True,
            "clip_dim": CLIP_DIM,
            "reverse_indexing": "image_feature_reverse[:, i]",
            "reverse_loss_accumulation": "factor losses accumulated then added once",
            "grl_alpha": 1.0,
        },
        "dataset": {
            "root": str(PRCC_ROOT),
            "train_images": len(dataset_images["train"]),
            "val_images": len(dataset_images["val"]),
            "test_gallery_A": len(list((PRCC_ROOT / "rgb/test/A").glob("*/*.jpg"))),
            "test_query_B_same": len(list((PRCC_ROOT / "rgb/test/B").glob("*/*.jpg"))),
            "test_query_C_changed": len(list((PRCC_ROOT / "rgb/test/C").glob("*/*.jpg"))),
            "primary_evaluation": "C -> A",
            "metadata_key": "PID + filename-stem",
            "basename_duplicates_train": len(dataset_images["train"]) - len({Path(path).stem for path in dataset_images["train"]}),
        },
        "text": {
            "train_path": str(TRAIN_NPZ),
            "train_sha256": sha256_file(TRAIN_NPZ),
            "train_data_shape": list(train_data.shape),
            "train_metadata_shape": [len(train_metadata)],
            "train_metadata_unique": len(set(train_metadata)),
            "train_key_set_matches_images": image_keys["train"] == set(train_metadata),
            "val_path": str(VAL_NPZ),
            "val_sha256": sha256_file(VAL_NPZ),
            "val_data_shape": list(val_data.shape),
            "val_metadata_shape": [len(val_metadata)],
            "val_metadata_unique": len(set(val_metadata)),
            "val_key_set_matches_images": image_keys["val"] == set(val_metadata),
            "summary_bio_path": str(SUMMARY_JSON),
            "summary_bio_sha256": sha256_file(SUMMARY_JSON),
            "summary_bio_entries": len(summary),
            "summary_bio_semantics": "identity-level summarized biometric representation",
            "nonbio_semantics": "image-level Hair + Clothing text",
            "clip_dim": CLIP_DIM,
        },
        "released_protocol": {
            "path": str(RELEASED_CONFIG),
            "nobio_index_raw": released.get("DATA", {}).get("NOBIO_INDEX"),
            "nobio_index_interpreted": RELEASED_NOBIO,
            "num_instances": released.get("DATA", {}).get("NUM_INSTANCES"),
            "batch_size": released.get("DATA", {}).get("BATCH_SIZE"),
        },
        "official_artifact": official_checkpoint_audit(),
        "pretrained": {
            "path": str(EVA_WEIGHT),
            "exists": EVA_WEIGHT.exists(),
            "bytes": EVA_WEIGHT.stat().st_size if EVA_WEIGHT.exists() else None,
            "sha256": sha256_file(EVA_WEIGHT) if EVA_WEIGHT.exists() else None,
            "revision": EVA_REVISION,
            "source": "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        },
        "training": {
            "hardware_required": ["PPU-ZW810E", "PPU-ZW810E"],
            "world_size": WORLD_SIZE_REQUIRED,
            "image_size": [224, 224],
            "epochs": 60,
            "batch_size_per_ppu": 16,
            "num_instances": EXPECTED_NUM_INSTANCES,
            "iterations_per_epoch_observed": EXPECTED_ITERATIONS,
            "optimizer": "SGD",
            "base_lr": 2e-6,
            "weight_decay": 0.05,
            "warmup": "linear",
            "scheduler": "epoch-based cosine",
            "seed": SEED,
            "amp_initial_scale": INITIAL_SCALE,
            "amp_dynamic_growth_backoff": True,
            "cpu_fallback": False,
        },
        "git": {
            "commit": command_output("git", "rev-parse", "HEAD"),
            "status": command_output("git", "status", "--short"),
            "diff": command_output("git", "diff"),
        },
    }
    report = ROOT / "reports/PHASE_3C_PRCC_PREFLIGHT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    artifact = ROOT / "diagnostics/phase3c_prcc_preflight.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n")
    official = audit["official_artifact"]
    report.write_text(
        "\n".join([
            "# PHASE 3C PRCC PREFLIGHT AUDIT",
            "",
            "## Decision",
            "",
            "- Formal track: **Track A / paper-faithful**.",
            "- Formal `NOBIO_INDEX=[1,2]`: **Hair + Clothing**.",
            "- Released `[1]` and packaged `[2,3]` are provenance diagnostics only.",
            "- `numerical_candidate`: **null**; no compatibility patch is active.",
            "- Primary evaluation: **Changed Clothes C → A**, 3543 queries against 3384 gallery images.",
            "",
            "## Dataset and text",
            "",
            f"- Dataset root: `{PRCC_ROOT}`; train `{len(dataset_images['train'])}`, val `{len(dataset_images['val'])}`, test A `{len(list((PRCC_ROOT / 'rgb/test/A').glob('*/*.jpg')))}`, B `{len(list((PRCC_ROOT / 'rgb/test/B').glob('*/*.jpg')))}`, C `{len(list((PRCC_ROOT / 'rgb/test/C').glob('*/*.jpg')))}`.",
            f"- Text: train `{train_data.shape}` / val `{val_data.shape}`; metadata keys are `PID/filename-stem`; train and val key sets match image files: `{image_keys['train'] == set(train_metadata)}` / `{image_keys['val'] == set(val_metadata)}`.",
            f"- Train basename duplicates: `{len(dataset_images['train']) - len({Path(path).stem for path in dataset_images['train']})}`; basename-only alignment is forbidden.",
            f"- Summary Bio: `{SUMMARY_JSON}` with `{len(summary)}` identity entries; runtime source is identity-level summarized biometric text.",
            "",
            "## Frozen protocol",
            "",
            f"- Image 224×224, 60 epochs, batch 16 per PPU, `NUM_INSTANCES={EXPECTED_NUM_INSTANCES}`, observed distributed iterations/epoch `{EXPECTED_ITERATIONS}`.",
            "- SGD, base LR 2e-6, weight decay 0.05, linear warmup, cosine scheduler, seed 1234.",
            "- AMP initial scale 512 with normal dynamic growth/backoff; no scaler retuning.",
            f"- EVA source: `{EVA_WEIGHT}`; revision `{EVA_REVISION}`; SHA256 `{audit['pretrained']['sha256']}`.",
            "- Hardware gate: exactly two visible PPU-ZW810E devices; CPU fallback forbidden.",
            "",
            "## Configuration discrepancy",
            "",
            f"- Paper-faithful formal run: `{PAPER_NOBIO}` = Hair + Clothing.",
            f"- Released head config: `{RELEASED_NOBIO}` = Hair only; not run as a formal arm.",
            f"- Official packaged config: `{PACKAGED_NOBIO}` = Clothing + Pose; not run as a formal arm.",
            "- The packaged checkpoint topology is audited separately and cannot redefine the paper-faithful training semantics.",
            "",
            "## Official artifact audit",
            "",
            f"- Checkpoint available: `{official.get('checkpoint_available')}`; config: `{official.get('config_available')}`; train log: `{official.get('train_log_available')}`.",
            f"- Official checkpoint structural audit: `{official.get('checkpoint', {}).get('raw_key_count', 'N/A')}` keys, classifier shape `{official.get('checkpoint', {}).get('classifier', [])[:1]}`, camera embedding `{official.get('checkpoint', {}).get('camera_embed', [])[:1]}`, reverse heads `{official.get('checkpoint', {}).get('reverse_head_count', 'N/A')}`.",
            f"- Official log evidence: `{official.get('train_log', {}).get('iterations_per_epoch_markers', [])}` iterations markers; logged best `{official.get('train_log', {}).get('best_logged')}`.",
            "- Closed-loop evaluation is optional provenance only and, if run, is recorded separately; it is never used to initialize formal training.",
            "",
            "## Required preflight gates",
            "",
            "Each formal run must pass distributed real-batch finite-loss/backward/optimizer-step smoke, Summary Bio and CLIP_DIM=768 checks, GRL sign check for Full, and exactly-once optimizer coverage before 60-epoch training.",
            "",
            f"Machine-readable audit: `{artifact}`.",
            "",
        ])
    )
    print(report)


def build_manifest(kind: str, cfg: Any, dataset: Any, run_dir: Path, preflight_path: Path, iterations: int) -> dict[str, Any]:
    spec = SPECS[kind]
    return {
        "phase": "3C",
        "kind": kind,
        "label": spec["label"],
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "track": "A: released-code / paper-faithful formal reproduction",
        "numerical_candidate": None,
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "git_diff": command_output("git", "diff"),
        "config_snapshot": cfg.dump(),
        "data": dataset_audit(dataset, type("LoaderInfo", (), {"batch_size": cfg.DATA.BATCH_SIZE, "__len__": lambda self: iterations})()),
        "model": {
            "name": str(cfg.MODEL.NAME),
            "type": str(cfg.MODEL.TYPE),
            "last_layer": str(cfg.MODEL.LAST_LAYER),
            "clip_dim": CLIP_DIM,
            "active_losses": list(cfg.MODEL.LOSS_TYPE),
            "bio_index": [0],
            "nobio_index": PAPER_NOBIO,
            "nobio_factor_names": [FACTOR_NAMES[i] for i in PAPER_NOBIO],
            "summary_bio": True,
            "reverse_indexing": "image_feature_reverse[:, i]",
            "reverse_loss_accumulation": "factor losses accumulated then added once",
            "grl": {"enabled": kind == "full", "alpha": 1.0},
        },
        "optimizer_scheduler": {
            "optimizer": str(cfg.SOLVER.OPTIMIZER_NAME),
            "momentum": float(cfg.SOLVER.MOMENTUM),
            "base_lr": float(cfg.SOLVER.BASE_LR),
            "bias_lr_factor": float(cfg.SOLVER.BIAS_LR_FACTOR),
            "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
            "weight_decay_bias": float(cfg.SOLVER.WEIGHT_DECAY_BIAS),
            "warmup_method": str(cfg.SOLVER.WARMUP_METHOD),
            "warmup_lr": float(cfg.SOLVER.WARMUP_LR),
            "warmup_epochs": int(cfg.SOLVER.WARMUP_EPOCHS),
            "first_runtime_lr": float(cfg.SOLVER.WARMUP_LR + (cfg.SOLVER.BASE_LR - cfg.SOLVER.WARMUP_LR) / cfg.SOLVER.WARMUP_EPOCHS),
            "scheduler": "epoch-based cosine",
            "epochs": 60,
        },
        "amp": {
            "enabled": True,
            "autocast_dtype": "FP16",
            "loss_location": "outside autocast",
            "initial_scale": INITIAL_SCALE,
            "dynamic_growth_backoff": True,
            "scale_not_retuned": True,
        },
        "pretrained": {
            "repository": "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
            "revision": EVA_REVISION,
            "path": str(EVA_WEIGHT),
            "sha256": sha256_file(EVA_WEIGHT),
            "bytes": EVA_WEIGHT.stat().st_size,
        },
        "text_resources": {
            "caption_dir": str(TEXT_ROOT),
            "train_npz": str(TRAIN_NPZ),
            "train_npz_sha256": sha256_file(TRAIN_NPZ),
            "val_npz": str(VAL_NPZ),
            "val_npz_sha256": sha256_file(VAL_NPZ),
            "summary_json": str(SUMMARY_JSON),
            "summary_json_sha256": sha256_file(SUMMARY_JSON),
            "metadata_alignment": "PID + filename-stem stable key",
            "summary_runtime": "DATA.SUMMERY_TEXT=True; factor-0 replaced by Summary Bio",
        },
        "evaluation": {
            "primary": "PRCC changed clothes C -> A",
            "primary_query_count": int(dataset.num_query_imgs_diff),
            "primary_gallery_count": int(dataset.num_gallery_imgs),
            "secondary": "PRCC same clothes B -> A",
            "feature_normalization": True,
            "reranking": False,
            "metric": "Euclidean pairwise distance + CMC/mAP",
        },
        "paper_target": {"rank1": spec["paper_rank1"], "mAP": spec["paper_map"]},
        "provenance_discrepancy": {
            "paper": PAPER_NOBIO,
            "released": RELEASED_NOBIO,
            "official_packaged": PACKAGED_NOBIO,
            "formal_choice": PAPER_NOBIO,
        },
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "forbidden_phase2_candidates": [
            "4-block FP32 compatibility", "sparse FP32 mask", "semantic FP32",
            "unfused attention", "historical bias", "precision island", "full FP32",
        ],
    }


def metrics_amp_stats(run_dir: Path, kind: str, iterations_per_epoch: int) -> dict[str, Any]:
    csv_path = run_dir / "metrics_per_epoch.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise RuntimeError(f"{csv_path}: expected 60 rows, got {len(rows)}")
    def number(row: dict[str, str], key: str) -> float | None:
        value = row.get(key, "")
        return None if value in {"", "NA", "None"} else float(value)
    skips = sum(int(number(row, "skipped_steps") or 0) for row in rows)
    nonfinite = sum(int(number(row, "nonfinite_events") or 0) for row in rows)
    scales = [number(row, "amp_scale_min") for row in rows if number(row, "amp_scale_min") is not None]
    text = (run_dir / "train_log.txt").read_text(errors="replace")
    matches = re.findall(r"AMP_SKIP epoch=(\d+) iteration=(\d+) scale_before_step=([0-9.]+)", text)
    first = matches[0] if matches else None
    return {
        "kind": kind,
        "total_iterations": int(iterations_per_epoch * 60),
        "optimizer_updates": int(iterations_per_epoch * 60 - skips),
        "amp_skips": skips,
        "found_inf_count": sum(int(number(row, "found_inf_count") or 0) for row in rows),
        "nonfinite_loss_count": nonfinite,
        "first_skip": {
            "epoch": int(first[0]), "iteration": int(first[1]), "scale_before_step": float(first[2])
        } if first else None,
        "min_scaler": min(scales) if scales else None,
        "max_scaler_observed": max(number(row, "amp_scale_max") for row in rows if number(row, "amp_scale_max") is not None),
        "final_scaler": number(rows[-1], "amp_scale_end"),
        "initial_scaler": INITIAL_SCALE,
        "dynamic_growth_backoff": True,
        "scale_not_retuned": True,
    }


def materialize_artifacts(kind: str, cfg: Any, run_dir: Path, iterations_per_epoch: int) -> dict[str, Any]:
    (run_dir / "config.yml").write_text(cfg.dump())
    (run_dir / "git_diff.patch").write_text(command_output("git", "diff"))
    amp_stats = metrics_amp_stats(run_dir, kind, iterations_per_epoch)
    amp_path = run_dir / "AMP_statistics.json"
    amp_path.write_text(json.dumps(amp_stats, indent=2, sort_keys=True) + "\n")
    csv_path = run_dir / "metrics_per_epoch.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "config": (run_dir / "config.yml").exists(),
        "manifest": (run_dir / "RUN_MANIFEST.json").exists(),
        "train_log": (run_dir / "train_log.txt").exists(),
        "tensorboard_train_dir": (run_dir / "train").is_dir(),
        "metrics_csv": csv_path.exists(),
        "metrics_rows": len(rows),
        "best_checkpoint": (run_dir / f"{cfg.MODEL.NAME}_best.pth").exists(),
        "final_checkpoint": (run_dir / f"{cfg.MODEL.NAME}_final.pth").exists(),
        "amp_json": amp_path.exists(),
        "git_diff": (run_dir / "git_diff.patch").exists(),
        "amp_statistics": amp_stats,
    }


def formal_train(kind: str, run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3C requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"expected exactly {WORLD_SIZE_REQUIRED} visible PPU devices, got {torch.cuda.device_count()}")
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.exists() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError(f"successful preflight required: {preflight_path}")
    complete_path = run_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        raise RuntimeError(f"refusing to overwrite completed PRCC run: {run_dir}")
    torch.cuda.set_device(local_rank)
    cfg = make_config(kind, run_dir)
    set_seed(SEED)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        logger = setup_logger("DIFFER", str(run_dir), if_train=True)
        logger.info("PHASE 3C formal PRCC run: %s", SPECS[kind]["label"])
        logger.info("Track A; numerical_candidate=null; no compatibility candidate is active")
        logger.info("Active losses: %s", list(cfg.MODEL.LOSS_TYPE))
        logger.info("Formal Summary Bio: identity-level summarized biometric JSON")
        logger.info("Formal NOBIO_INDEX=[1,2] -> Hair + Clothing; CLIP_DIM=768")
        logger.info("PRCC primary evaluation: C -> A; query=3543; gallery=3384")
        logger.info("AMP initial scale=%d; dynamic growth/backoff retained", INITIAL_SCALE)
        logger.info("PPU devices: %s", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        train_loader, _same, _diff, _gallery, dataset, _sampler, val_loader, val_loader_same = build_dataloader(cfg)
        iterations = len(train_loader)
        if iterations != EXPECTED_ITERATIONS:
            raise RuntimeError(f"expected {EXPECTED_ITERATIONS} iterations/epoch, got {iterations}")
        if local_rank == 0:
            manifest = build_manifest(kind, cfg, dataset, run_dir, preflight_path, iterations)
            (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n")
            dataset_text = {
                "dataset": dataset_audit(dataset, train_loader),
                "text": text_runtime_audit(dataset, cfg),
            }
            (run_dir / "dataset_text_manifest.json").write_text(json.dumps(dataset_text, indent=2, sort_keys=True, default=json_default) + "\n")
        torch.distributed.barrier()
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        logger.info("Model parameter count: %d", sum(parameter.numel() for parameter in model.parameters()))
        logger.info("Fresh model/optimizer/scheduler/scaler/sampler; 60 epochs begin")
        do_train(cfg, model, train_loader, local_rank, dataset, val_loader=val_loader, val_loader_same=val_loader_same)
        torch.distributed.barrier()
        if local_rank == 0:
            artifact_audit = materialize_artifacts(kind, cfg, run_dir, iterations)
            best_path = run_dir / f"{cfg.MODEL.NAME}_best.pth"
            final_path = run_dir / f"{cfg.MODEL.NAME}_final.pth"
            complete = {
                "phase": "3C",
                "kind": kind,
                "completed_utc": utc_now(),
                "epochs": 60,
                "best_checkpoint": str(best_path),
                "final_checkpoint": str(final_path),
                "metrics_csv": str(run_dir / "metrics_per_epoch.csv"),
                "preflight": str(preflight_path),
                "artifact_audit": artifact_audit,
            }
            complete_path.write_text(json.dumps(complete, indent=2, sort_keys=True, default=json_default) + "\n")
            logger.info("PHASE 3C formal run complete: %s", json.dumps(complete, sort_keys=True))
    finally:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def independent_evaluate(kind: str, checkpoint: Path, output: Path, packaged: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("independent evaluator requires CUDA/PPU")
    nobio = PACKAGED_NOBIO if packaged else PAPER_NOBIO
    cfg = make_config(kind, output.parent, nobio=nobio)
    cfg.defrost()
    cfg.MODEL.DIST_TRAIN = False
    cfg.freeze()
    dataset = build_dataset(cfg)
    _train_transform, test_transform = build_img_transforms(cfg)
    device = torch.device("cuda:0")
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    normalized = {key.replace("module.", "", 1): value for key, value in state.items()}
    incompatible = model.load_state_dict(normalized, strict=True)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(f"independent checkpoint load mismatch missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()

    def eval_pair(query_data: list[dict[str, Any]], gallery_data: list[dict[str, Any]]) -> dict[str, Any]:
        query_loader = DataLoader(ImageDataset(query_data, transform=test_transform), batch_size=128, shuffle=False, num_workers=1, pin_memory=False)
        gallery_loader = DataLoader(ImageDataset(gallery_data, transform=test_transform), batch_size=128, shuffle=False, num_workers=1, pin_memory=False)
        evaluator = R1_mAP_eval(len(query_data), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        with torch.no_grad():
            for loader in (query_loader, gallery_loader):
                for data in loader:
                    feat = model(data["image"].to(device), cam_label=data["camid"].to(device))
                    evaluator.update((feat, data["pid"], data["camid"], data["clothes_id"], data["image_path"]))
        cmc, mAP, *_ = evaluator.compute()
        return {
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "rank10": float(cmc[9]),
            "mAP": float(mAP),
            "query_count": len(query_data),
            "gallery_count": len(gallery_data),
        }

    result = {
        "kind": kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "topology_nobio_index": nobio,
        "topology_semantics": [FACTOR_NAMES[i] for i in nobio],
        "load": {"strict_effective": True, "missing": [], "unexpected": []},
        "changed_clothes_C_to_A": eval_pair(dataset.query_diff, dataset.gallery),
        "same_clothes_B_to_A": eval_pair(dataset.query_same, dataset.gallery),
        "evaluator": "independent DataLoader + R1_mAP_eval; no reranking",
        "device": torch.cuda.get_device_name(0),
        "created_utc": utc_now(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n")
    return result


def read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics_per_epoch.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value in {"", "NA", "None"} else float(value)


def status_for(best: dict[str, Any], spec: dict[str, Any]) -> str:
    gaps = [abs(float(best["rank1"]) - spec["paper_rank1"]), abs(float(best["mAP"]) - spec["paper_map"])]
    if max(gaps) <= 1.0:
        return "EXCELLENT"
    if max(gaps) <= 2.0:
        return "ACCEPTABLE"
    if max(gaps) <= 4.0:
        return "CONCERNING"
    return "FAILED"


def run_summary(run_dir: Path, kind: str) -> dict[str, Any]:
    rows = read_metrics(run_dir)
    best_row = max(rows, key=lambda row: float(row["cc_rank1"]))
    final_row = rows[-1]
    best = {"epoch": int(best_row["epoch"]), "rank1": 100 * f(best_row, "cc_rank1"), "mAP": 100 * f(best_row, "cc_mAP")}
    final = {"epoch": int(final_row["epoch"]), "rank1": 100 * f(final_row, "cc_rank1"), "mAP": 100 * f(final_row, "cc_mAP")}
    return {
        "kind": kind,
        "best": best,
        "final": final,
        "status": status_for(best, SPECS[kind]),
        "amp": json.loads((run_dir / "AMP_statistics.json").read_text()),
        "rows": rows,
    }


def signed(value: float) -> str:
    return f"{value:+.4f}"


def trajectory_turning_point(baseline: dict[str, Any], full: dict[str, Any]) -> int | None:
    b = {int(row["epoch"]): f(row, "cc_rank1") for row in baseline["rows"]}
    x = {int(row["epoch"]): f(row, "cc_rank1") for row in full["rows"]}
    epochs = sorted(set(b) & set(x))
    for index in range(len(epochs) - 2):
        window = epochs[index:index + 3]
        if all(x[e] < b[e] - 0.01 for e in window):
            return window[0]
    return None


def final_report() -> None:
    baseline_dir = run_dir_for("baseline")
    full_dir = run_dir_for("full")
    baseline = run_summary(baseline_dir, "baseline")
    full = run_summary(full_dir, "full")
    effect = {
        "rank1": full["best"]["rank1"] - baseline["best"]["rank1"],
        "mAP": full["best"]["mAP"] - baseline["best"]["mAP"],
    }
    paper_effect = {"rank1": 3.4, "mAP": 1.7}
    turn = trajectory_turning_point(baseline, full)
    sample_epochs = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    b_rows = {int(row["epoch"]): row for row in baseline["rows"]}
    f_rows = {int(row["epoch"]): row for row in full["rows"]}
    official_path = ROOT / "diagnostics/prcc_official_closed_loop.json"
    official = json.loads(official_path.read_text()) if official_path.exists() else None

    if baseline["status"] == "FAILED":
        diagnosis = "PRCC BASELINE ITSELF NOT REPRODUCED"
    elif effect["rank1"] < 0 and effect["mAP"] < 0:
        diagnosis = "CROSS-DATASET PPU ADVERSARIAL FAILURE CONFIRMED"
    elif abs(effect["rank1"] - paper_effect["rank1"]) > 2 or abs(effect["mAP"] - paper_effect["mAP"]) > 2:
        diagnosis = "PRCC PARTIALLY REPRODUCES DIFFER GAIN"
    else:
        diagnosis = "PRCC FULL REPRODUCES PAPER GAIN"

    def run_block(title: str, result: dict[str, Any], spec: dict[str, Any]) -> list[str]:
        best = result["best"]
        final = result["final"]
        amp_stats = result["amp"]
        return [
            f"## {title}", "",
            f"Paper: {spec['paper_rank1']:.1f} / {spec['paper_map']:.1f}",
            f"BEST epoch: {best['epoch']}",
            f"PPU BEST: {best['rank1']:.4f} / {best['mAP']:.4f}",
            f"FINAL epoch: {final['epoch']}",
            f"PPU FINAL: {final['rank1']:.4f} / {final['mAP']:.4f}",
            f"BEST gap (PPU - paper): {signed(best['rank1'] - spec['paper_rank1'])} / {signed(best['mAP'] - spec['paper_map'])} pp",
            f"Status: **{result['status']}**",
            f"AMP: iterations {amp_stats['total_iterations']}; updates {amp_stats['optimizer_updates']}; skips {amp_stats['amp_skips']}; first skip {amp_stats['first_skip']}; min scale {amp_stats['min_scaler']}; final scale {amp_stats['final_scaler']}; nonfinite loss count {amp_stats['nonfinite_loss_count']}.",
            "",
        ]

    lines = [
        "# PHASE 3C RESULT — PRCC MAIN RESULTS", "",
        "## PRCC PROTOCOL", "",
        "- Formal paper-faithful semantic topology: `NOBIO_INDEX=[1,2]` = Hair + Clothing.",
        "- Bio: identity-level summarized biometric representation; NonBio: image-level text.",
        "- CLIP_DIM=768; primary evaluation is changed clothes C → A (3543 → 3384); no reranking.",
        "- Track A only; `numerical_candidate=null`; AMP initial scale 512 is the frozen PRCC environment setting.",
        "",
        "## PROTOCOL DISCREPANCY", "",
        "| Source | NOBIO_INDEX | Role |", "|---|---:|---|",
        "| Paper-faithful | `[1,2]` | Formal training choice |",
        "| Released head config | `[1]` | Audited only; not run |",
        "| Official packaged artifact | `[2,3]` | Provenance/closed-loop only; not run |",
        "",
    ]
    lines.extend(run_block("BASELINE", baseline, SPECS["baseline"]))
    lines.extend([
        "## BASELINE GATE", "",
        "Baseline formal gate is passed for Full only when neither core metric is more than 4 pp below paper. The classification above is computed from absolute BEST gaps.",
        "",
    ])
    lines.extend(run_block("FULL DIFFER", full, SPECS["full"]))
    lines.extend([
        "## METHOD EFFECT", "",
        f"Paper: +3.4 / +1.7 pp.",
        f"PPU Full BEST − Baseline BEST: {signed(effect['rank1'])} / {signed(effect['mAP'])} pp.",
        "",
        "## LTCC CROSS-DATASET COMPARISON", "",
        "| Setting | LTCC BEST R1 | LTCC BEST mAP | PRCC BEST R1 | PRCC BEST mAP |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | 56.4000 | 31.1000 | {baseline['best']['rank1']:.4f} | {baseline['best']['mAP']:.4f} |",
        f"| ID + Bio | 56.3780 | 32.3860 | — | — |",
        f"| Bio + Hair | 54.3370 | 31.2590 | — | — |",
        f"| Bio + Clothing | 52.2960 | 27.7480 | — | — |",
        f"| Bio + Pose | 52.8060 | 27.3800 | — | — |",
        f"| Full | 53.3160 | 29.2660 | {full['best']['rank1']:.4f} | {full['best']['mAP']:.4f} |",
        "",
        "## TRAJECTORY", "",
        f"First three-epoch sustained Full C→A Rank-1 divergence below the independent Baseline by >1 pp: `{turn if turn is not None else 'not detected'}`.",
        "",
        "| Epoch | Baseline C→A R1 | Baseline mAP | Full C→A R1 | Full mAP | Full Bio acc | Full Hair reverse acc | Full Clothing reverse acc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for epoch in sample_epochs:
        br, fr = b_rows.get(epoch), f_rows.get(epoch)
        if br is None or fr is None:
            continue
        def pct(row: dict[str, str], key: str) -> str:
            value = f(row, key)
            return "NA" if value is None else f"{100 * value:.4f}"
        lines.append(f"| {epoch} | {pct(br, 'cc_rank1')} | {pct(br, 'cc_mAP')} | {pct(fr, 'cc_rank1')} | {pct(fr, 'cc_mAP')} | {pct(fr, 'bio_accuracy')} | {pct(fr, 'reverse_hair_accuracy')} | {pct(fr, 'reverse_cloth_accuracy')} |")
    lines.extend([
        "",
        "Reverse accuracy is reported as supporting trajectory evidence; AMP skips are not treated as a root cause without a Full-specific temporal association.",
        "",
        "## OFFICIAL ARTIFACT", "",
        f"Closed-loop result: `{official if official is not None else 'NOT AVAILABLE'}`.",
        "The packaged `[2,3]` topology remains excluded from formal training even if its checkpoint can be loaded strictly.",
        "",
        "## FINAL DIAGNOSIS", "",
        f"**{diagnosis}**",
        "",
        f"LTCC frozen diagnosis: NonBio adversarial training broadly fails on PPU (Hair, Clothing, and Pose single-factor runs are negative/failed). PRCC Baseline is {baseline['status']}; PRCC Full method effect is {signed(effect['rank1'])} / {signed(effect['mAP'])} pp versus the PRCC Baseline.",
        "",
        "After this report, Phase 3C stops. No released `[1]` run, packaged `[2,3]` run, seed sweep, compatibility patch, PRCC ablation, or new method is started.",
        "",
    ])
    report = ROOT / "reports/PHASE_3C_PRCC_MAIN_RESULTS.md"
    report.write_text("\n".join(lines))
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shared-audit", "preflight", "train", "evaluate", "official-evaluate", "report"), required=True)
    parser.add_argument("--kind", choices=tuple(SPECS), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "shared-audit":
        shared_audit()
        return
    if args.mode == "report":
        final_report()
        return
    if args.mode == "official-evaluate":
        result = independent_evaluate("full", OFFICIAL_CHECKPOINT, ROOT / "diagnostics/prcc_official_closed_loop.json", packaged=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.kind is None:
        parser.error("--kind is required for preflight/train/evaluate")
    run_dir = run_dir_for(args.kind, args.output_dir)
    local_rank = local_rank_from_args(args.local_rank)
    if args.mode == "preflight":
        preflight(args.kind, run_dir, local_rank)
    elif args.mode == "train":
        formal_train(args.kind, run_dir, local_rank)
    elif args.mode == "evaluate":
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "eva02_l_bio_best.pth"
        result = independent_evaluate(args.kind, checkpoint, run_dir / "independent_evaluation.json")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
