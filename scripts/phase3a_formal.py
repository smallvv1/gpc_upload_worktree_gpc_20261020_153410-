#!/usr/local/bin/python3
"""PHASE 3A: formal LTCC Table-3 ablation reproduction.

This driver deliberately uses the released formal training path in
``processor.do_train``.  It adds only the Phase-3A preflight, provenance, and
artifact bookkeeping around that path.  In particular, it never imports or
applies any Phase-2 numerical candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataloader  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from processor import do_train  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.logger import setup_logger  # noqa: E402


DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
TEXT_NPZ = TEXT_ROOT / "EVA02-CLIP-B-16/train.npz"
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
EVA_WEIGHT = Path(
    "/root/.cache/huggingface/hub/models--timm--"
    "eva02_large_patch14_clip_224.merged2b_s4b_b131k/snapshots/"
    "bf4190eb65dd5204ffb03e980108beb1200e0873/open_clip_pytorch_model.bin"
)
OUTPUT_ROOT = ROOT / "outputs/reproduction_phase3/LTCC/table3"
WORLD_SIZE_REQUIRED = 2
SEED = 1234
INITIAL_SCALE = 1024
EVA_REVISION = "bf4190eb65dd5204ffb03e980108beb1200e0873"

SPECS: dict[str, dict[str, Any]] = {
    "id_bio": {
        "label": "ID + Bio",
        "losses": ["ce", "triplet", "clipBio"],
        "paper_rank1": 55.4,
        "paper_map": 30.8,
        "relative_effect": "+0.8 / +0.4",
    },
    "id_nonbio": {
        "label": "ID + NonBio",
        "losses": ["ce", "triplet", "clipBioReverse"],
        "paper_rank1": 55.6,
        "paper_map": 31.0,
        "relative_effect": "+1.0 / +0.6",
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
    """Serialize NumPy/PyTorch scalar metadata without changing experiment data."""
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
    if value is not None:
        return int(value)
    return int(os.environ.get("LOCAL_RANK", "0"))


def run_dir_for(ablation: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (OUTPUT_ROOT / ablation / "run1").resolve()


def make_config(ablation: str, run_dir: Path):
    spec = SPECS[ablation]
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = list(spec["losses"])
    local_cfg.MODEL.DIST_TRAIN = True
    local_cfg.MODEL.CLIP_DIM = 512
    local_cfg.DATA.ROOT = str(DATA_ROOT)
    local_cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    # yacs parses the released LTCC value ``'1,2'`` into [1, 2].  The
    # released EVA meta-bio constructor and LTCC loader both expect that list
    # at runtime, so preserve the resulting type exactly.
    local_cfg.DATA.NOBIO_INDEX = [1, 2]
    # The official LTCC run is identified as OnlySummeryID.  This explicit
    # value is required to replace factor-0 with identity-level Summary Bio.
    local_cfg.DATA.SUMMERY_TEXT = True
    local_cfg.SOLVER.SEED = SEED
    local_cfg.SOLVER.MAX_EPOCHS = 60
    local_cfg.DATA.BATCH_SIZE = 8
    local_cfg.DATA.NUM_INSTANCES = 2
    local_cfg.OUTPUT_DIR = str(run_dir)
    local_cfg.OUTPUT_NAME = f"PHASE3A_LTCC_{ablation.upper()}_RUN1"
    local_cfg.TEST.WEIGHT = ""
    local_cfg.freeze()
    return local_cfg


def family_for(name: str) -> str:
    if name.startswith(("blocks.", "patch_embed.", "pos_embed", "cls_token", "camera_embed", "rope.")):
        return "eva_backbone"
    if name.startswith("head."):
        return "id_classifier"
    if name.startswith(("head_bio.", "norm_bio.", "head_clip_bio.")):
        return "bio_head"
    if name.startswith("head_clip_bioReverse."):
        return "reverse_heads"
    return "other"


def nobio_indices(cfg: Any) -> list[int]:
    raw = cfg.DATA.NOBIO_INDEX
    if isinstance(raw, str):
        raw = raw.replace("*", ",").split(",")
    return [int(item) for item in raw]


def parameter_coverage(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    name_by_id: dict[int, str] = {}
    trainable_names: list[str] = []
    family_counts: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        name_by_id[id(parameter)] = name
        if parameter.requires_grad:
            trainable_names.append(name)
            family_counts[family_for(name)] += 1

    occurrences: Counter[int] = Counter()
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = []
        for parameter in group["params"]:
            occurrences[id(parameter)] += 1
            names.append(name_by_id.get(id(parameter), "<unknown>"))
        groups.append({
            "index": index,
            "parameter_count": len(names),
            "names": names,
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        })

    id_by_name = {name: id(parameter) for name, parameter in model.named_parameters()}
    missing = [name for name in trainable_names if occurrences[id_by_name[name]] == 0]
    duplicates = [name for name in trainable_names if occurrences[id_by_name[name]] != 1]
    unknown = [
        name
        for group in groups
        for name in group["names"]
        if name == "<unknown>"
    ]
    lr_wd_counts: Counter[tuple[float, float]] = Counter(
        (item["lr"], item["weight_decay"])
        for item in groups
    )
    return {
        "trainable_parameter_count": len(trainable_names),
        "trainable_parameter_numel": int(sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )),
        "trainable_by_family": dict(sorted(family_counts.items())),
        "optimizer_group_count": len(groups),
        "optimizer_parameter_count": sum(item["parameter_count"] for item in groups),
        "missing_trainable_names": missing,
        "duplicate_or_miscovered_names": duplicates,
        "unknown_optimizer_parameters": unknown,
        "exactly_once": not missing and not duplicates and not unknown,
        "lr_weight_decay_counts": [
            {"lr": lr, "weight_decay": wd, "groups": count}
            for (lr, wd), count in sorted(lr_wd_counts.items())
        ],
        "groups": groups,
    }


def module_grad_stats(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, Any]:
    parameter_count = 0
    requires_grad_count = 0
    grad_exists = 0
    nonzero_grad = 0
    nan_count = 0
    inf_count = 0
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
            if bool((finite.abs() > 0).any().item()):
                nonzero_grad += 1
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


def scalar_finite(values: dict[str, float], tag: str) -> bool:
    return tag in values and np.isfinite(float(values[tag]))


def summary_bio_audit(dataset: Any, cfg: Any) -> dict[str, Any]:
    if not SUMMARY_JSON.exists():
        raise RuntimeError(f"missing Summary Bio file: {SUMMARY_JSON}")
    with SUMMARY_JSON.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not TEXT_NPZ.exists():
        raise RuntimeError(f"missing LTCC text NPZ: {TEXT_NPZ}")
    with np.load(TEXT_NPZ, allow_pickle=True) as archive:
        npz_shape = list(archive["data"].shape)
        metadata_shape = list(archive["metadata"].shape)
    key = "ft_EVA02-CLIP-B-16"
    first_pid = str(int(dataset.train[0]["pid"]))
    expected = np.asarray(summary[first_pid][key], dtype=np.float32).reshape(-1)
    actual = np.asarray(dataset.train[0]["caption_feature"][0], dtype=np.float32).reshape(-1)
    max_abs = float(np.max(np.abs(expected - actual)))
    return {
        "config_summary_text": bool(cfg.DATA.SUMMERY_TEXT),
        "dataset_load_sum_ft": bool(getattr(dataset, "load_sum_ft", False)),
        "summary_json": str(SUMMARY_JSON),
        "summary_json_sha256": sha256_file(SUMMARY_JSON),
        "summary_entry_count": len(summary),
        "summary_feature_key": key,
        "summary_feature_dim": int(expected.size),
        "npz": str(TEXT_NPZ),
        "npz_sha256": sha256_file(TEXT_NPZ),
        "npz_data_shape": npz_shape,
        "npz_metadata_shape": metadata_shape,
        "first_train_pid_label": int(dataset.train[0]["pid"]),
        "first_sample_factor0_max_abs_vs_summary": max_abs,
        "summary_factor0_runtime_match": max_abs == 0.0,
    }


def architecture_audit(ablation: str, cfg: Any, model: torch.nn.Module) -> dict[str, Any]:
    spec = SPECS[ablation]
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    reverse_count = len(reverse_modules) if reverse_modules is not None else 0
    nobio_index = nobio_indices(cfg)
    return {
        "ablation": ablation,
        "label": spec["label"],
        "active_losses": list(cfg.MODEL.LOSS_TYPE),
        "bio_contrastive_loss_enabled": "clipBio" in cfg.MODEL.LOSS_TYPE,
        "reverse_loss_enabled": "clipBioReverse" in cfg.MODEL.LOSS_TYPE,
        "bio_head_enabled": hasattr(model, "head_bio") and hasattr(model, "head_clip_bio"),
        "reverse_head_count": reverse_count,
        "expected_reverse_head_count": 0 if ablation == "id_bio" else 2,
        "nobio_index": nobio_index,
        "nobio_factor_names": ["Hair", "Clothing"],
        "reverse_factor_axis": "image_feature_reverse[:, i]",
        "reverse_loss_accumulation": "loss_clip_reverse += factor_loss; loss += loss_clip_reverse",
        "grl_alpha": 1.0,
        "summary_bio_required": True,
        "model_parameter_count": sum(1 for _ in model.parameters()),
        "model_parameter_numel": int(sum(parameter.numel() for parameter in model.parameters())),
        "modules": {
            "head": hasattr(model, "head"),
            "head_bio": hasattr(model, "head_bio"),
            "norm_bio": hasattr(model, "norm_bio"),
            "head_clip_bio": hasattr(model, "head_clip_bio"),
            "head_clip_bioReverse": hasattr(model, "head_clip_bioReverse"),
        },
    }


def run_smoke(
    ablation: str,
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
    iterator = iter(train_loader)
    batch = next(iterator)
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

    hook_handles = []
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    if reverse_modules is not None:
        for reverse_head in reverse_modules:
            hook_handles.append(reverse_head.grl.register_full_backward_hook(grl_hook))

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
        component_tags = {
            "ce": "loss/id",
            "triplet": "loss/triplet",
            "bio": "loss/clip_bio_0",
            "reverse_hair": "loss/clip_bio_reverse_0",
            "reverse_clothing_cumulative": "loss/clip_bio_reverse_1",
        }
        active_component_names = ["ce", "triplet"]
        if ablation == "id_bio":
            active_component_names.append("bio")
        else:
            active_component_names.extend(["reverse_hair", "reverse_clothing_cumulative"])
        component_finite = {
            name: scalar_finite(writer.values, component_tags[name])
            for name in active_component_names
        }
        total_finite = bool(torch.isfinite(total_loss.detach()).item())
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        grad_groups = {
            "eva_backbone": ("patch_embed.", "pos_embed", "cls_token", "camera_embed", "blocks."),
            "id_classifier": ("head.",),
            "bio_transformer_head": ("head_bio.", "norm_bio."),
            "bio_projection_head": ("head_clip_bio.",),
            "reverse_hair_head": ("head_clip_bioReverse.0.",),
            "reverse_clothing_head": ("head_clip_bioReverse.1.",),
        }
        gradients = {name: module_grad_stats(model, prefixes) for name, prefixes in grad_groups.items()}
        critical_names = ["eva_backbone", "id_classifier", "bio_transformer_head"]
        if ablation == "id_bio":
            critical_names.append("bio_projection_head")
        else:
            critical_names.extend(["reverse_hair_head", "reverse_clothing_head"])
        critical_finite = all(
            gradients[name]["grad_exists_count"] > 0
            and gradients[name]["nonzero_grad_count"] > 0
            and gradients[name]["nan_count"] == 0
            and gradients[name]["inf_count"] == 0
            for name in critical_names
        )
        if ablation == "id_bio":
            expected_tags = ["loss/id", "loss/triplet", "loss/clip_bio_0"]
            forbidden_tags = ["loss/clip_bio_reverse_0", "loss/clip_bio_reverse_1"]
            branch_expectations = {
                "bio_projection_grad_exists": gradients["bio_projection_head"]["grad_exists_count"] > 0,
                "reverse_hair_grad_exists": gradients["reverse_hair_head"]["grad_exists_count"] == 0,
                "reverse_clothing_grad_exists": gradients["reverse_clothing_head"]["grad_exists_count"] == 0,
            }
        else:
            expected_tags = ["loss/id", "loss/triplet", "loss/clip_bio_reverse_0", "loss/clip_bio_reverse_1"]
            forbidden_tags = ["loss/clip_bio_0"]
            branch_expectations = {
                "bio_projection_grad_exists": gradients["bio_projection_head"]["grad_exists_count"] == 0,
                "reverse_hair_grad_exists": gradients["reverse_hair_head"]["grad_exists_count"] > 0,
                "reverse_clothing_grad_exists": gradients["reverse_clothing_head"]["grad_exists_count"] > 0,
            }
        expected_tags_present = all(tag in writer.values for tag in expected_tags)
        forbidden_tags_absent = all(tag not in writer.values for tag in forbidden_tags)
        grl_pass = (
            ablation == "id_bio"
            or len(grl_records) == 2 and all(item.get("valid") and item.get("allclose") for item in grl_records)
        )
        report = {
            "batch_size": int(batch["image"].shape[0]),
            "image_shape": list(batch["image"].shape),
            "global_feature_shape": list(feat[0].shape),
            "id_score_shape": list(score["cls_score"].shape),
            "bio_score_shape": list(score["clip_bio_score"].shape) if "clip_bio_score" in score else None,
            "reverse_score_shape": list(score["clip_bio_reverse_score"].shape)
            if "clip_bio_reverse_score" in score else None,
            "text_feature_shape": list(batch["caption_feature"].shape),
            "active_writer_scalars": dict(sorted(writer.values.items())),
            "component_finite": component_finite,
            "expected_tags_present": expected_tags_present,
            "forbidden_tags_absent": forbidden_tags_absent,
            "total_loss": float(total_loss.detach().double().cpu()),
            "total_loss_dtype": str(total_loss.dtype),
            "total_finite": total_finite,
            "gradients": gradients,
            "critical_gradients_finite_nonzero": critical_finite,
            "branch_expectations": branch_expectations,
            "grl_records": grl_records,
            "grl_direction_pass": grl_pass,
            "scaler_initial_scale": float(scaler.get_scale()),
            "optimizer_step_executed": False,
            "passed": bool(
                total_finite
                and expected_tags_present
                and forbidden_tags_absent
                and all(component_finite.values())
                and critical_finite
                and all(branch_expectations.values())
                and grl_pass
            ),
        }
        return report
    finally:
        for handle in hook_handles:
            handle.remove()
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass


def preflight(ablation: str, run_dir: Path, local_rank: int) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3A requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(
            f"PHASE 3A requires exactly {WORLD_SIZE_REQUIRED} visible PPU devices, "
            f"got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        cfg = make_config(ablation, run_dir)
        set_seed(SEED)
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        train_loader, _query_loader, _gallery_loader, dataset, _sampler, _val_loader, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        architecture = architecture_audit(ablation, cfg, model)
        loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
        scheduler.step(cfg.TRAIN.START_EPOCH)
        coverage = parameter_coverage(model, optimizer)
        if not coverage["exactly_once"]:
            raise RuntimeError(f"optimizer coverage failed: {coverage}")

        device = torch.device(f"cuda:{local_rank}")
        model.to(device)
        model.train()
        summary_audit = summary_bio_audit(dataset, cfg)
        if not summary_audit["config_summary_text"] or not summary_audit["dataset_load_sum_ft"]:
            raise RuntimeError(f"Summary Bio is not enabled at runtime: {summary_audit}")
        if not summary_audit["summary_factor0_runtime_match"]:
            raise RuntimeError(f"runtime factor-0 does not match Summary Bio: {summary_audit}")

        smoke = run_smoke(
            ablation,
            cfg,
            model,
            dataset,
            train_loader,
            loss_fn,
            optimizer,
            center_criterion,
            device,
        )
        rank_report = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "architecture": architecture,
            "data": {
                "train": len(dataset.train),
                "query": len(dataset.query),
                "gallery": len(dataset.gallery),
                "train_ids": dataset.num_train_pids,
                "train_clothes": dataset.num_train_clothes,
                "cameras": dataset.num_camera,
                "iterations_per_epoch": len(train_loader),
            },
            "summary_bio": summary_audit,
            "optimizer_coverage": coverage,
            "smoke": smoke,
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(WORLD_SIZE_REQUIRED)]
        torch.distributed.all_gather_object(gathered, rank_report)
        if local_rank == 0:
            rank_reports = [item for item in gathered if item is not None]
            passed = all(item["smoke"]["passed"] for item in rank_reports)
            payload = {
                "phase": "3A",
                "mode": "formal_paper_faithful_preflight",
                "created_utc": utc_now(),
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "git_status": command_output("git", "status", "--short"),
                "ablation": ablation,
                "track": "A: released-code / paper-faithful formal path",
                "numerical_candidate": None,
                "world_size": WORLD_SIZE_REQUIRED,
                "seed": SEED,
                "config_snapshot": cfg.dump(),
                "rank_reports": rank_reports,
                "passed": passed,
            }
            output = run_dir / "preflight.json"
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
            )
            print(
                "PHASE3A_PREFLIGHT",
                json.dumps({
                    "ablation": ablation,
                    "output": str(output),
                    "active_losses": list(cfg.MODEL.LOSS_TYPE),
                    "bio_head_enabled": architecture["bio_head_enabled"],
                    "reverse_head_count": architecture["reverse_head_count"],
                    "summary_bio_runtime_match": summary_audit["summary_factor0_runtime_match"],
                    "optimizer_exactly_once": coverage["exactly_once"],
                    "smoke_passed_all_ranks": passed,
                }, sort_keys=True),
                flush=True,
            )
            return payload
        return None
    finally:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def build_manifest(ablation: str, cfg: Any, dataset: Any, run_dir: Path, preflight_path: Path) -> dict[str, Any]:
    spec = SPECS[ablation]
    return {
        "phase": "3A",
        "created_utc": utc_now(),
        "ablation": ablation,
        "label": spec["label"],
        "track": "A: released-code / paper-faithful formal reproduction",
        "numerical_candidate": None,
        "command": " ".join(sys.argv),
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "git_diff": command_output("git", "diff"),
        "config_snapshot": cfg.dump(),
        "data": {
            "dataset": "ltcc",
            "root": str(DATA_ROOT),
            "train_images": len(dataset.train),
            "train_ids": dataset.num_train_pids,
            "train_clothes": dataset.num_train_clothes,
            "cameras": dataset.num_camera,
            "query_images": len(dataset.query),
            "gallery_images": len(dataset.gallery),
            "image_size": [int(cfg.DATA.IMG_HEIGHT), int(cfg.DATA.IMG_WIDTH)],
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "num_instances": int(cfg.DATA.NUM_INSTANCES),
            "iterations_per_epoch": 596,
            "epochs": int(cfg.SOLVER.MAX_EPOCHS),
            "sampler": str(cfg.DATA.SAMPLER),
            "num_workers": int(cfg.DATA.NUM_WORKERS),
        },
        "model": {
            "name": str(cfg.MODEL.NAME),
            "type": str(cfg.MODEL.TYPE),
            "last_layer": str(cfg.MODEL.LAST_LAYER),
            "clip_dim": int(cfg.MODEL.CLIP_DIM),
            "active_losses": list(cfg.MODEL.LOSS_TYPE),
            "bio_index": [0],
            "nobio_index": [1, 2],
            "nobio_factor_names": ["Hair", "Clothing"],
            "summary_bio": True,
            "reverse_indexing": "image_feature_reverse[:, i]",
            "reverse_loss_accumulation": "all factors accumulated before total loss",
            "grl": {"enabled": "clipBioReverse" in cfg.MODEL.LOSS_TYPE, "alpha": 1.0},
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
            "first_runtime_lr": float(
                cfg.SOLVER.WARMUP_LR
                + (cfg.SOLVER.BASE_LR - cfg.SOLVER.WARMUP_LR) / cfg.SOLVER.WARMUP_EPOCHS
            ),
            "first_runtime_bias_lr": float(
                cfg.SOLVER.WARMUP_LR
                + (cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR - cfg.SOLVER.WARMUP_LR)
                / cfg.SOLVER.WARMUP_EPOCHS
            ),
            "scheduler": "epoch-based cosine",
            "epochs": int(cfg.SOLVER.MAX_EPOCHS),
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
            "train_npz": str(TEXT_NPZ),
            "train_npz_sha256": sha256_file(TEXT_NPZ),
            "summary_json": str(SUMMARY_JSON),
            "summary_json_sha256": sha256_file(SUMMARY_JSON),
            "summary_runtime": "DATA.SUMMERY_TEXT=True; factor-0 replaced by Summary Bio",
        },
        "evaluation": {
            "primary": "LTCC clothes-changing",
            "secondary": "LTCC general",
            "feature_normalization": True,
            "reranking": False,
            "metric": "Euclidean pairwise distance + CMC/mAP",
        },
        "checkpoint_policy": {
            "best": "clothes-changing Rank-1",
            "final": "epoch 60",
            "optimizer_saved": False,
            "scheduler_saved": False,
            "gradscaler_saved": False,
        },
        "paper_target": {
            "rank1": spec["paper_rank1"],
            "mAP": spec["paper_map"],
            "paper_effect_vs_id_only": spec["relative_effect"],
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "forbidden_phase2_candidates": [
            "4-block FP32 compatibility",
            "semantic FP32",
            "unfused attention",
            "historical bias=True",
            "precision island",
            "full FP32",
        ],
    }


def formal_train(ablation: str, run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3A requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(
            f"PHASE 3A requires exactly {WORLD_SIZE_REQUIRED} visible PPU devices, "
            f"got {torch.cuda.device_count()}"
        )
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError(f"missing successful preflight: {preflight_path}")
    preflight_payload = json.loads(preflight_path.read_text())
    if not preflight_payload.get("passed"):
        raise RuntimeError(f"preflight did not pass: {preflight_path}")

    torch.cuda.set_device(local_rank)
    cfg = make_config(ablation, run_dir)
    set_seed(SEED)
    if local_rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        logger = setup_logger("DIFFER", str(run_dir), if_train=True)
        logger.info("PHASE 3A formal run: %s", SPECS[ablation]["label"])
        logger.info("Track A; no Phase-2 numerical candidate is active")
        logger.info("Active losses: %s", list(cfg.MODEL.LOSS_TYPE))
        logger.info("Summary Bio: DATA.SUMMERY_TEXT=%s", cfg.DATA.SUMMERY_TEXT)
        logger.info(
            "NOBIO_INDEX: %s (Hair + Clothing)",
            nobio_indices(cfg),
        )
        logger.info("PPU devices: %s", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

        train_loader, _query_loader, _gallery_loader, dataset, _sampler, val_loader, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        if local_rank == 0:
            manifest = build_manifest(ablation, cfg, dataset, run_dir, preflight_path)
            (run_dir / "RUN_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n"
            )
            logger.info("Saved RUN_MANIFEST.json")
        torch.distributed.barrier()

        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        logger.info("Model parameter count: %d", sum(parameter.numel() for parameter in model.parameters()))
        logger.info("Formal 60 epoch training begins from fresh pretrained initialization")
        do_train(
            cfg,
            model,
            train_loader,
            local_rank,
            dataset,
            val_loader=val_loader,
        )
        torch.distributed.barrier()
        if local_rank == 0:
            best_path = run_dir / f"{cfg.MODEL.NAME}_best.pth"
            final_path = run_dir / f"{cfg.MODEL.NAME}_final.pth"
            metrics_path = run_dir / "metrics_per_epoch.csv"
            complete = {
                "phase": "3A",
                "ablation": ablation,
                "completed_utc": utc_now(),
                "epochs": int(cfg.SOLVER.MAX_EPOCHS),
                "best_checkpoint": str(best_path),
                "final_checkpoint": str(final_path),
                "metrics_csv": str(metrics_path),
                "best_exists": best_path.exists(),
                "final_exists": final_path.exists(),
                "metrics_exists": metrics_path.exists(),
                "metrics_rows": max(0, sum(1 for _ in metrics_path.open()) - 1) if metrics_path.exists() else 0,
                "preflight": str(preflight_path),
            }
            (run_dir / "RUN_COMPLETE.json").write_text(
                json.dumps(complete, indent=2, sort_keys=True, default=json_default) + "\n"
            )
            logger.info("PHASE 3A formal run complete: %s", json.dumps(complete, sort_keys=True))
    finally:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--ablation", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    run_dir = run_dir_for(args.ablation, args.output_dir)
    local_rank = local_rank_from_args(args.local_rank)
    if args.mode == "preflight":
        preflight(args.ablation, run_dir, local_rank)
    else:
        formal_train(args.ablation, run_dir, local_rank)


if __name__ == "__main__":
    main()
