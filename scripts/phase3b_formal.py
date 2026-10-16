#!/usr/local/bin/python3
"""PHASE 3B: formal LTCC Table-4 single-factor reproduction.

This driver reuses the released formal training path and adds only
factor-specific configuration, preflight, provenance, and artifact auditing.
It never imports or applies a Phase-2 numerical candidate.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_dataloader  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from processor import do_train  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

from phase3a_formal import (  # noqa: E402
    DATA_ROOT,
    EVA_REVISION,
    EVA_WEIGHT,
    INITIAL_SCALE,
    SUMMARY_JSON,
    TEXT_NPZ,
    TEXT_ROOT,
    WORLD_SIZE_REQUIRED,
    ScalarCapture,
    command_output,
    family_for,
    json_default,
    module_grad_stats,
    nobio_indices,
    parameter_coverage,
    set_seed,
    sha256_file,
    summary_bio_audit,
    utc_now,
)


OUTPUT_ROOT = ROOT / "outputs/reproduction_phase3/LTCC/table4"
PHASE3A_REPORT = ROOT / "reports/PHASE_3A_LTCC_TABLE3.md"
PHASE3A_ROOT = ROOT / "outputs/reproduction_phase3/LTCC/table3"
PHASE3A_BASELINE_CSV = ROOT / (
    "outputs/reproduction_phase2/LTCC/baseline/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/metrics_per_epoch.csv"
)
PHASE3A_FULL_CSV = ROOT / (
    "outputs/reproduction_phase2/LTCC/differ/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/metrics_per_epoch.csv"
)
SEED = 1234

SPECS: dict[str, dict[str, Any]] = {
    "bio_hair": {
        "label": "Bio + Hair",
        "factor_index": 1,
        "factor_name": "Hair",
        "paper_rank1": 56.6,
        "paper_map": 31.3,
        "paper_effect_vs_bio": "+1.2 / +0.5",
    },
    "bio_clothing": {
        "label": "Bio + Clothing",
        "factor_index": 2,
        "factor_name": "Clothing",
        "paper_rank1": 56.6,
        "paper_map": 30.4,
        "paper_effect_vs_bio": "+1.2 / -0.4",
    },
    "bio_pose": {
        "label": "Bio + Pose",
        "factor_index": 3,
        "factor_name": "Pose",
        "paper_rank1": 56.1,
        "paper_map": 31.5,
        "paper_effect_vs_bio": "+0.7 / +0.7",
    },
}


def local_rank_from_args(value: int | None) -> int:
    return int(value) if value is not None else int(os.environ.get("LOCAL_RANK", "0"))


def run_dir_for(factor: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (OUTPUT_ROOT / factor / "run1").resolve()


def make_config(factor: str, run_dir: Path):
    spec = SPECS[factor]
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.MODEL.DIST_TRAIN = True
    local_cfg.MODEL.CLIP_DIM = 512
    local_cfg.DATA.ROOT = str(DATA_ROOT)
    local_cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.DATA.NOBIO_INDEX = [int(spec["factor_index"])]
    local_cfg.DATA.SUMMERY_TEXT = True
    local_cfg.SOLVER.SEED = SEED
    local_cfg.SOLVER.MAX_EPOCHS = 60
    local_cfg.DATA.BATCH_SIZE = 8
    local_cfg.DATA.NUM_INSTANCES = 2
    local_cfg.OUTPUT_DIR = str(run_dir)
    local_cfg.OUTPUT_NAME = f"PHASE3B_LTCC_{factor.upper()}_RUN1"
    local_cfg.TEST.WEIGHT = ""
    local_cfg.freeze()
    return local_cfg


def factor_text_audit(dataset: Any, cfg: Any, factor_index: int, factor_name: str) -> dict[str, Any]:
    with np.load(TEXT_NPZ, allow_pickle=True) as archive:
        data = np.asarray(archive["data"])
        metadata = [item.decode() if isinstance(item, bytes) else str(item) for item in archive["metadata"]]
    first = dataset.train[0]
    image_stem = Path(first["image_path"]).stem
    try:
        metadata_index = metadata.index(image_stem)
    except ValueError as exc:
        raise RuntimeError(f"missing text metadata for {image_stem}") from exc
    factor_count = int(data.shape[0] / len(metadata))
    raw_block = data[metadata_index * factor_count : (metadata_index + 1) * factor_count]
    runtime = np.asarray(first["caption_feature"], dtype=np.float32)
    selected_runtime = runtime[1]
    selected_raw = np.asarray(raw_block[factor_index], dtype=np.float32)
    summary_runtime = np.asarray(runtime[0], dtype=np.float32)
    return {
        "factor_index": factor_index,
        "factor_name": factor_name,
        "official_factor_order": {
            "0": "Biometrics",
            "1": "Hairstyle",
            "2": "Clothing",
            "3": "Pose",
            "4": "Interaction",
            "5": "Environment",
        },
        "runtime_caption_shape": list(runtime.shape),
        "expected_runtime_caption_shape": [2, int(data.shape[1])],
        "runtime_factor_count": int(runtime.shape[0]),
        "raw_text_factor_count": factor_count,
        "first_image": image_stem,
        "selected_factor_runtime_max_abs_vs_raw": float(np.max(np.abs(selected_runtime - selected_raw))),
        "summary_factor0_runtime_max_abs_vs_summary": float(np.max(np.abs(summary_runtime - np.asarray(
            json.loads(SUMMARY_JSON.read_text())[str(int(first["pid"]))]["ft_EVA02-CLIP-B-16"], dtype=np.float32
        )))),
        "selected_factor_runtime_match": bool(np.array_equal(selected_runtime, selected_raw)),
        "summary_factor0_runtime_match": bool(np.array_equal(summary_runtime, np.asarray(
            json.loads(SUMMARY_JSON.read_text())[str(int(first["pid"]))]["ft_EVA02-CLIP-B-16"], dtype=np.float32
        ))),
        "summary_text_enabled": bool(cfg.DATA.SUMMERY_TEXT),
    }


def architecture_audit(factor: str, cfg: Any, model: torch.nn.Module) -> dict[str, Any]:
    spec = SPECS[factor]
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    reverse_count = len(reverse_modules) if reverse_modules is not None else 0
    return {
        "factor": factor,
        "factor_index": int(spec["factor_index"]),
        "factor_name": spec["factor_name"],
        "active_losses": list(cfg.MODEL.LOSS_TYPE),
        "bio_contrastive_loss_enabled": "clipBio" in cfg.MODEL.LOSS_TYPE,
        "reverse_loss_enabled": "clipBioReverse" in cfg.MODEL.LOSS_TYPE,
        "bio_head_enabled": hasattr(model, "head_bio") and hasattr(model, "head_clip_bio"),
        "reverse_head_count": reverse_count,
        "expected_reverse_head_count": 1,
        "nobio_index": nobio_indices(cfg),
        "nobio_factor_names": [spec["factor_name"]],
        "reverse_factor_axis": "image_feature_reverse[:, i]",
        "reverse_loss_accumulation": "loss_clip_reverse += factor_loss; loss += loss_clip_reverse",
        "grl_alpha": 1.0,
        "summary_bio_required": True,
        "modules": {
            "head": hasattr(model, "head"),
            "head_bio": hasattr(model, "head_bio"),
            "norm_bio": hasattr(model, "norm_bio"),
            "head_clip_bio": hasattr(model, "head_clip_bio"),
            "head_clip_bioReverse": hasattr(model, "head_clip_bioReverse"),
        },
    }


def run_smoke(
    factor: str,
    cfg: Any,
    model: torch.nn.Module,
    train_loader: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    center_criterion: Any,
    device: torch.device,
) -> dict[str, Any]:
    spec = SPECS[factor]
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

    hook_handles = []
    reverse_modules = getattr(model, "head_clip_bioReverse", None)
    if reverse_modules is None or len(reverse_modules) != 1:
        raise RuntimeError(f"expected exactly one reverse head, got {reverse_modules}")
    hook_handles.append(reverse_modules[0].grl.register_full_backward_hook(grl_hook))

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
            "reverse_selected": "loss/clip_bio_reverse_0",
        }
        component_finite = {
            name: component_tags[name] in writer.values and np.isfinite(writer.values[component_tags[name]])
            for name in component_tags
        }
        total_finite = bool(torch.isfinite(total_loss.detach()).item())
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients = {
            "eva_backbone": module_grad_stats(model, ("patch_embed.", "pos_embed", "cls_token", "camera_embed", "blocks.")),
            "id_classifier": module_grad_stats(model, ("head.",)),
            "bio_transformer_head": module_grad_stats(model, ("head_bio.", "norm_bio.")),
            "bio_projection_head": module_grad_stats(model, ("head_clip_bio.",)),
            "reverse_selected_head": module_grad_stats(model, ("head_clip_bioReverse.0.",)),
        }
        critical_finite = all(
            gradients[name]["grad_exists_count"] > 0
            and gradients[name]["nonzero_grad_count"] > 0
            and gradients[name]["nan_count"] == 0
            and gradients[name]["inf_count"] == 0
            for name in gradients
        )
        expected_tags_present = all(tag in writer.values for tag in component_tags.values())
        forbidden_tags_absent = "loss/clip_bio_reverse_1" not in writer.values
        reverse_score = score.get("clip_bio_reverse_score")
        reverse_shape_ok = reverse_score is not None and int(reverse_score.shape[1]) == 1
        grl_pass = len(grl_records) == 1 and grl_records[0].get("valid") and grl_records[0].get("allclose")
        report = {
            "factor_index": int(spec["factor_index"]),
            "factor_name": spec["factor_name"],
            "batch_size": int(batch["image"].shape[0]),
            "image_shape": list(batch["image"].shape),
            "global_feature_shape": list(feat[0].shape),
            "id_score_shape": list(score["cls_score"].shape),
            "bio_score_shape": list(score["clip_bio_score"].shape),
            "reverse_score_shape": list(reverse_score.shape),
            "text_feature_shape": list(batch["caption_feature"].shape),
            "active_writer_scalars": dict(sorted(writer.values.items())),
            "component_finite": component_finite,
            "expected_tags_present": expected_tags_present,
            "forbidden_extra_reverse_tags_absent": forbidden_tags_absent,
            "reverse_score_one_factor": reverse_shape_ok,
            "total_loss": float(total_loss.detach().double().cpu()),
            "total_loss_dtype": str(total_loss.dtype),
            "total_finite": total_finite,
            "gradients": gradients,
            "critical_gradients_finite_nonzero": critical_finite,
            "grl_records": grl_records,
            "grl_direction_pass": bool(grl_pass),
            "scaler_initial_scale": float(scaler.get_scale()),
            "optimizer_step_executed": False,
            "passed": bool(
                total_finite
                and expected_tags_present
                and forbidden_tags_absent
                and all(component_finite.values())
                and reverse_shape_ok
                and critical_finite
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


def shared_audit() -> None:
    configs = {factor: make_config(factor, OUTPUT_ROOT / factor / "run1") for factor in SPECS}
    datasets = {factor: build_dataset(config) for factor, config in configs.items()}
    phase3a_manifest = json.loads((PHASE3A_ROOT / "id_bio/run1/RUN_MANIFEST.json").read_text())
    lines = [
        "# PHASE 3B shared preflight audit",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "## Frozen Phase 3A inputs",
        "",
        f"- Report: `{PHASE3A_REPORT}`",
        "- Phase 3A outputs are read-only for this phase; no Table-3 run is re-executed or overwritten.",
        f"- Frozen ID+Bio BEST/FINAL: see `{PHASE3A_ROOT / 'id_bio/run1/metrics_per_epoch.csv'}`.",
        f"- Frozen ID+NonBio BEST/FINAL: see `{PHASE3A_ROOT / 'id_nonbio/run1/metrics_per_epoch.csv'}`.",
        f"- Frozen Bio manifest commit: `{phase3a_manifest['git_commit']}`.",
        "",
        "## Shared resources and protocol",
        "",
        f"- Git commit: `{command_output('git', 'rev-parse', 'HEAD')}`",
        f"- Git status: `{command_output('git', 'status', '--short') or '(clean)'}`",
        f"- Git diff stat: `{command_output('git', 'diff', '--stat') or '(no unstaged diff)'}`",
        f"- EVA checkpoint: `{EVA_WEIGHT}`",
        f"- EVA revision: `{EVA_REVISION}`",
        f"- EVA SHA256: `{sha256_file(EVA_WEIGHT)}`",
        f"- LTCC train text NPZ SHA256: `{sha256_file(TEXT_NPZ)}`",
        f"- Summary Bio JSON SHA256: `{sha256_file(SUMMARY_JSON)}`",
        "- Dataset: LTCC; image size 224×224; epochs 60; batch config 8; 596 iterations/epoch; `NUM_INSTANCES=2`.",
        "- Optimizer/schedule: SGD; base LR 2e-6; weight decay 0.05; linear warmup; cosine schedule; first runtime LR ≈8.421875e-7.",
        "- AMP: initial GradScaler 1024; dynamic growth/backoff; no retuning; loss outside autocast.",
        "- Evaluation: LTCC changed-clothes primary, standard/general secondary; feature normalization enabled; no reranking.",
        "- Sampler: DistributedRandomIdentitySampler, seed 1234, two instances per identity; 2× PPU-ZW810E required.",
        "- Reverse implementation: `image_feature_reverse[:, i]`, cumulative `loss_clip_reverse += factor_loss`, GRL alpha 1.0.",
        "",
        "## Dataset and Summary Bio checks",
        "",
    ]
    for factor, dataset in datasets.items():
        cfg = configs[factor]
        summary = summary_bio_audit(dataset, cfg)
        selected = factor_text_audit(dataset, cfg, SPECS[factor]["factor_index"], SPECS[factor]["factor_name"])
        lines.extend([
            f"### {SPECS[factor]['label']}",
            "",
            f"- Stats: train `{len(dataset.train)}`, IDs `{dataset.num_train_pids}`, clothes `{dataset.num_train_clothes}`, query `{len(dataset.query)}`, gallery `{len(dataset.gallery)}`, cameras `{dataset.num_camera}`.",
            f"- `NOBIO_INDEX=[{SPECS[factor]['factor_index']}]` = **{SPECS[factor]['factor_name']}**; runtime caption shape `{selected['runtime_caption_shape']}`; raw selected factor match `{selected['selected_factor_runtime_match']}`.",
            f"- Summary Bio runtime match `{summary['summary_factor0_runtime_match']}`; summary text enabled `{selected['summary_text_enabled']}`.",
            f"- Active losses: `CE + Triplet + Bio + Reverse-{SPECS[factor]['factor_name']}`.",
            "",
        ])
    lines.extend([
        "## Factor matrix",
        "",
        "| Run | NOBIO_INDEX | Semantic factor | Paper target | Numerical candidate |",
        "|---|---:|---|---:|---|",
    ])
    for factor, spec in SPECS.items():
        lines.append(
            f"| {factor} | `{spec['factor_index']}` | {spec['factor_name']} | {spec['paper_rank1']:.1f} / {spec['paper_map']:.1f} | `null` |"
        )
    lines.extend([
        "",
        "This audit is shared by the three independent formal runs. Individual distributed preflight JSON files additionally record optimizer coverage, architecture, smoke gradients, and GRL direction.",
        "",
    ])
    report = ROOT / "reports/PHASE_3B_PREFLIGHT_AUDIT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(report)


def preflight(factor: str, run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3B requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"expected exactly {WORLD_SIZE_REQUIRED} visible PPU devices, got {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        cfg = make_config(factor, run_dir)
        set_seed(SEED)
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        train_loader, _query_loader, _gallery_loader, dataset, _sampler, _val_loader, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        architecture = architecture_audit(factor, cfg, model)
        if architecture["reverse_head_count"] != 1:
            raise RuntimeError(f"single-factor architecture audit failed: {architecture}")
        loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, _optimizer_center = make_optimizer(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
        scheduler.step(cfg.TRAIN.START_EPOCH)
        coverage = parameter_coverage(model, optimizer)
        if not coverage["exactly_once"]:
            raise RuntimeError(f"optimizer coverage failed: {coverage}")
        device = torch.device(f"cuda:{local_rank}")
        model.to(device)
        model.train()
        summary_audit = summary_bio_audit(dataset, cfg)
        selected_audit = factor_text_audit(dataset, cfg, SPECS[factor]["factor_index"], SPECS[factor]["factor_name"])
        if not summary_audit["summary_factor0_runtime_match"] or not selected_audit["selected_factor_runtime_match"]:
            raise RuntimeError(f"text semantics audit failed: summary={summary_audit}, selected={selected_audit}")
        smoke = run_smoke(factor, cfg, model, train_loader, loss_fn, optimizer, center_criterion, device)
        rank_report = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "factor": factor,
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
            "selected_factor_text": selected_audit,
            "optimizer_coverage": coverage,
            "smoke": smoke,
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(WORLD_SIZE_REQUIRED)]
        torch.distributed.all_gather_object(gathered, rank_report)
        if local_rank == 0:
            reports = [item for item in gathered if item is not None]
            passed = all(item["smoke"]["passed"] for item in reports)
            payload = {
                "phase": "3B",
                "mode": "formal_paper_faithful_preflight",
                "created_utc": utc_now(),
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "git_status": command_output("git", "status", "--short"),
                "factor": factor,
                "factor_index": SPECS[factor]["factor_index"],
                "factor_name": SPECS[factor]["factor_name"],
                "track": "A: released-code / paper-faithful formal path",
                "numerical_candidate": None,
                "world_size": WORLD_SIZE_REQUIRED,
                "seed": SEED,
                "config_snapshot": cfg.dump(),
                "rank_reports": reports,
                "passed": passed,
            }
            output = run_dir / "preflight.json"
            output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
            print(
                "PHASE3B_PREFLIGHT",
                json.dumps({
                    "factor": factor,
                    "factor_index": SPECS[factor]["factor_index"],
                    "factor_name": SPECS[factor]["factor_name"],
                    "active_losses": list(cfg.MODEL.LOSS_TYPE),
                    "reverse_head_count": architecture["reverse_head_count"],
                    "summary_bio_runtime_match": summary_audit["summary_factor0_runtime_match"],
                    "selected_factor_runtime_match": selected_audit["selected_factor_runtime_match"],
                    "optimizer_exactly_once": coverage["exactly_once"],
                    "smoke_passed_all_ranks": passed,
                    "output": str(output),
                }, sort_keys=True),
                flush=True,
            )
    finally:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def build_manifest(factor: str, cfg: Any, dataset: Any, run_dir: Path, preflight_path: Path) -> dict[str, Any]:
    spec = SPECS[factor]
    return {
        "phase": "3B",
        "created_utc": utc_now(),
        "factor": factor,
        "label": spec["label"],
        "factor_index": spec["factor_index"],
        "factor_name": spec["factor_name"],
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
            "nobio_index": [spec["factor_index"]],
            "nobio_factor_names": [spec["factor_name"]],
            "summary_bio": True,
            "reverse_indexing": "image_feature_reverse[:, i]",
            "reverse_loss_accumulation": "single selected factor through generic cumulative implementation",
            "grl": {"enabled": True, "alpha": 1.0},
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
            "selected_factor": spec["factor_name"],
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
            "paper_incremental_effect_vs_bio": spec["paper_effect_vs_bio"],
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


def metrics_amp_stats(run_dir: Path, factor: str) -> dict[str, Any]:
    csv_path = run_dir / "metrics_per_epoch.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise RuntimeError(f"{csv_path}: expected 60 rows, got {len(rows)}")
    def num(row, key):
        raw = row[key]
        return None if raw in {"NA", "", "None"} else float(raw)
    skip = sum(int(num(row, "skipped_steps") or 0) for row in rows)
    inf = sum(int(num(row, "found_inf_count") or 0) for row in rows)
    nonfinite = sum(int(num(row, "nonfinite_events") or 0) for row in rows)
    min_scale = min(num(row, "amp_scale_min") for row in rows)
    max_scale = max(num(row, "amp_scale_max") for row in rows)
    log_text = (run_dir / "train_log.txt").read_text(errors="replace")
    matches = re.findall(r"AMP_SKIP epoch=(\d+) iteration=(\d+) scale_before_step=([0-9.]+)", log_text)
    first = matches[0] if matches else None
    stats = {
        "factor": factor,
        "factor_index": SPECS[factor]["factor_index"],
        "factor_name": SPECS[factor]["factor_name"],
        "total_iterations": 60 * 596,
        "optimizer_updates": 60 * 596 - skip,
        "amp_skips": skip,
        "found_inf_count": inf,
        "nonfinite_loss_events": nonfinite,
        "min_scaler": min_scale,
        "max_scaler_observed": max_scale,
        "final_scaler": num(rows[-1], "amp_scale_end"),
        "initial_scaler": INITIAL_SCALE,
        "first_skip_epoch": int(first[0]) if first else None,
        "first_skip_iteration": int(first[1]) if first else None,
        "first_skip_scale_before_step": float(first[2]) if first else None,
        "skip_log_event_count": len(matches),
        "dynamic_growth_backoff": True,
        "scale_not_retuned": True,
    }
    return stats


def materialize_artifacts(factor: str, cfg: Any, run_dir: Path) -> None:
    (run_dir / "config.yml").write_text(cfg.dump())
    (run_dir / "git_diff.patch").write_text(command_output("git", "diff"))
    stats = metrics_amp_stats(run_dir, factor)
    (run_dir / "AMP_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    # The released CSV names reverse slots hair/cloth. For a single-factor run,
    # slot 0 is the selected semantic factor; materialize an explicit semantic
    # view without changing the released metrics CSV.
    with (run_dir / "metrics_per_epoch.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_path = run_dir / "metrics_selected_factor_per_epoch.csv"
    for row in rows:
        row["factor_index"] = str(SPECS[factor]["factor_index"])
        row["factor_name"] = SPECS[factor]["factor_name"]
        row["reverse_selected_loss"] = row["reverse_hair_loss"]
    fields = list(rows[0].keys())
    with selected_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formal_train(factor: str, run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 3B requires CUDA/PPU; CPU fallback is forbidden")
    if torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"expected exactly {WORLD_SIZE_REQUIRED} visible PPU devices, got {torch.cuda.device_count()}")
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError(f"missing successful preflight: {preflight_path}")
    if not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError(f"preflight did not pass: {preflight_path}")
    if (run_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"refusing to overwrite completed Phase 3B run: {run_dir}")

    torch.cuda.set_device(local_rank)
    cfg = make_config(factor, run_dir)
    set_seed(SEED)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        logger = setup_logger("DIFFER", str(run_dir), if_train=True)
        spec = SPECS[factor]
        logger.info("PHASE 3B formal run: %s", spec["label"])
        logger.info("Track A; numerical_candidate=null; no Phase-2 compatibility candidate is active")
        logger.info("Active losses: %s", list(cfg.MODEL.LOSS_TYPE))
        logger.info("NOBIO_INDEX=[%d] semantic=%s", spec["factor_index"], spec["factor_name"])
        logger.info("Summary Bio: DATA.SUMMERY_TEXT=%s", cfg.DATA.SUMMERY_TEXT)
        logger.info("PPU devices: %s", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        train_loader, _query_loader, _gallery_loader, dataset, _sampler, val_loader, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        if local_rank == 0:
            manifest = build_manifest(factor, cfg, dataset, run_dir, preflight_path)
            (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n")
            logger.info("Saved RUN_MANIFEST.json")
        torch.distributed.barrier()
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        logger.info("Model parameter count: %d", sum(parameter.numel() for parameter in model.parameters()))
        logger.info("Fresh model/optimizer/scheduler/scaler; formal 60 epoch training begins")
        do_train(cfg, model, train_loader, local_rank, dataset, val_loader=val_loader)
        torch.distributed.barrier()
        if local_rank == 0:
            materialize_artifacts(factor, cfg, run_dir)
            best_path = run_dir / f"{cfg.MODEL.NAME}_best.pth"
            final_path = run_dir / f"{cfg.MODEL.NAME}_final.pth"
            metrics_path = run_dir / "metrics_per_epoch.csv"
            complete = {
                "phase": "3B",
                "factor": factor,
                "factor_index": spec["factor_index"],
                "factor_name": spec["factor_name"],
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
                "amp_statistics": str(run_dir / "AMP_statistics.json"),
            }
            (run_dir / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True, default=json_default) + "\n")
            logger.info("PHASE 3B formal run complete: %s", json.dumps(complete, sort_keys=True))
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
    parser.add_argument("--mode", choices=("shared-audit", "preflight", "train"), required=True)
    parser.add_argument("--factor", choices=tuple(SPECS), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "shared-audit":
        shared_audit()
        return
    if args.factor is None:
        parser.error("--factor is required for preflight/train")
    run_dir = run_dir_for(args.factor, args.output_dir)
    local_rank = local_rank_from_args(args.local_rank)
    if args.mode == "preflight":
        preflight(args.factor, run_dir, local_rank)
    else:
        formal_train(args.factor, run_dir, local_rank)


if __name__ == "__main__":
    main()
