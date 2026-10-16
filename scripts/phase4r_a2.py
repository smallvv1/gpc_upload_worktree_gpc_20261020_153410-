#!/usr/local/bin/python3
"""PHASE 4R-A: LTCC detached-reverse causal control.

This is a repair/research-track driver.  It keeps the formal Track-A model,
data, optimizer, scheduler, sampler, AMP, Summary Bio, and Hair+Clothing
semantics, while setting only ``MODEL.DETACH_REVERSE_INPUT=True``.  Reverse
heads still train, but reverse loss gradients cannot enter the shared Bio/EVA
representation.
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
import torch.distributed as dist
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
from utils.phase4r_telemetry import (  # noqa: E402
    collect_component_gradient_telemetry,
)

from scripts.phase3a_formal import (  # noqa: E402
    ScalarCapture,
    module_grad_stats,
    parameter_coverage,
    summary_bio_audit,
)


DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption/LTCC_ReID"
EVA_WEIGHT = Path(
    "/root/.cache/huggingface/hub/models--timm--"
    "eva02_large_patch14_clip_224.merged2b_s4b_b131k/snapshots/"
    "bf4190eb65dd5204ffb03e980108beb1200e0873/open_clip_pytorch_model.bin"
)
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
TEXT_NPZ = TEXT_ROOT / "EVA02-CLIP-B-16/train.npz"
WORLD_SIZE_REQUIRED = 2
SEED = 1234
INITIAL_SCALE = 1024
EPOCHS = 25
TELEMETRY_INTERVAL = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def make_config(run_dir: Path):
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.MODEL.DIST_TRAIN = True
    local_cfg.MODEL.CLIP_DIM = 512
    local_cfg.MODEL.DETACH_REVERSE_INPUT = True
    local_cfg.DATA.ROOT = str(DATA_ROOT)
    local_cfg.DATA.CAPTION_DIR = str(TEXT_ROOT)
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.DATA.NOBIO_INDEX = [1, 2]
    local_cfg.DATA.SUMMERY_TEXT = True
    local_cfg.DATA.BATCH_SIZE = 8
    local_cfg.DATA.NUM_INSTANCES = 2
    local_cfg.SOLVER.SEED = SEED
    local_cfg.SOLVER.MAX_EPOCHS = EPOCHS
    local_cfg.OUTPUT_DIR = str(run_dir)
    local_cfg.OUTPUT_NAME = "PHASE4R_A2_DETACHED_REVERSE_CONTROL"
    local_cfg.TEST.WEIGHT = ""
    local_cfg.TRAIN.GRADIENT_TELEMETRY = True
    local_cfg.TRAIN.GRADIENT_TELEMETRY_INTERVAL = TELEMETRY_INTERVAL
    local_cfg.freeze()
    return local_cfg


def reverse_shared_gradient_audit(
    model: torch.nn.Module,
    components: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = collect_component_gradient_telemetry(
        model=model,
        components=components,
        scaler_scale=INITIAL_SCALE,
        epoch=0,
        iteration=0,
        rank=0,
    )
    shared_rows = [row for row in rows if row["group"] != "bio_projection"]
    max_reverse = max(
        [
            float(row["hair_norm_unscaled"])
            for row in shared_rows
        ]
        + [
            float(row["clothing_norm_unscaled"])
            for row in shared_rows
        ]
        + [
            float(row["reverse_norm_unscaled"])
            for row in shared_rows
        ]
    ) if shared_rows else 0.0
    core_norms = {row["group"]: float(row["core_norm_unscaled"]) for row in rows}
    return {
        "rows": rows,
        "shared_reverse_max_abs_norm": max_reverse,
        "shared_reverse_detached": bool(max_reverse == 0.0),
        "core_norms": core_norms,
        "core_nonzero_shared": bool(any(value > 0.0 for value in core_norms.values())),
    }


def run_smoke(cfg: Any, model: torch.nn.Module, train_loader: Any, dataset: Any, local_rank: int) -> dict[str, Any]:
    model.train()
    batch = next(iter(train_loader))
    writer = ScalarCapture()
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _optimizer_center = make_optimizer(cfg, model, center_criterion)
    optimizer.zero_grad(set_to_none=True)
    device = torch.device(f"cuda:{local_rank}")
    # In A2 the real reverse input is intentionally detached, so a backward
    # hook on the live GRL has no input gradient to observe.  Verify the exact
    # GRL module semantics with an isolated differentiable probe instead; the
    # real-batch audit below separately verifies that this detach blocks the
    # reverse gradient from the shared representation.
    grl_records: list[dict[str, Any]] = []
    for factor_index, reverse_head in enumerate(getattr(model, "head_clip_bioReverse", [])):
        probe = torch.randn(
            (2, int(reverse_head.head_fc.in_features)),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        probe_output = reverse_head.grl(probe)
        probe_grad = torch.autograd.grad(
            probe_output.sum(),
            probe,
            retain_graph=False,
            create_graph=False,
        )[0]
        expected = torch.full_like(probe_grad, -float(reverse_head.grl.alpha))
        error = probe_grad - expected
        grl_records.append({
            "factor_index": factor_index,
            "valid": True,
            "alpha": float(reverse_head.grl.alpha),
            "max_abs_error": float(error.abs().max().item()),
            "allclose": bool(torch.allclose(probe_grad, expected)),
        })

    try:
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
        total_loss = loss_fn(
            score,
            feat,
            batch["pid"].to(device),
            batch["camid"].to(device),
            batch["caption_feature"].to(device),
            clothes_ids=batch["clothes_id"].to(device),
            train_writer=writer,
            step=0,
        )
        telemetry = reverse_shared_gradient_audit(model, getattr(loss_fn, "last_components", {}))
        scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        base_model = model.module if hasattr(model, "module") else model
        gradients = {
            "eva_backbone": module_grad_stats(base_model, ("blocks.", "patch_embed.", "pos_embed", "cls_token", "camera_embed")),
            "bio_projection": module_grad_stats(base_model, ("head_clip_bio.",)),
            "reverse_heads": module_grad_stats(base_model, ("head_clip_bioReverse.",)),
        }
        reverse_head_grad_ok = gradients["reverse_heads"]["grad_exists_count"] > 0 and gradients["reverse_heads"]["nonzero_grad_count"] > 0
        backbone_grad_ok = gradients["eva_backbone"]["grad_exists_count"] > 0 and gradients["eva_backbone"]["nonzero_grad_count"] > 0
        bio_grad_ok = gradients["bio_projection"]["grad_exists_count"] > 0 and gradients["bio_projection"]["nonzero_grad_count"] > 0
        return {
            "batch_size": int(batch["image"].shape[0]),
            "total_finite": bool(torch.isfinite(total_loss.detach()).item()),
            "component_scalars": dict(sorted(writer.values.items())),
            "loss_components": sorted(getattr(loss_fn, "last_components", {}).keys()),
            "telemetry": telemetry,
            "gradients_after_unscale": gradients,
            "reverse_head_grad_exists_finite_nonzero": reverse_head_grad_ok,
            "eva_backbone_grad_exists_finite_nonzero": backbone_grad_ok,
            "bio_projection_grad_exists_finite_nonzero": bio_grad_ok,
            "grl_records": grl_records,
            "grl_direction_pass": len(grl_records) == 2 and all(item.get("valid") and item.get("allclose") for item in grl_records),
            "passed": bool(
                torch.isfinite(total_loss.detach()).item()
                and telemetry["shared_reverse_detached"]
                and telemetry["core_nonzero_shared"]
                and reverse_head_grad_ok
                and backbone_grad_ok
                and bio_grad_ok
                and len(grl_records) == 2
                and all(item.get("valid") and item.get("allclose") for item in grl_records)
            ),
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        if hasattr(train_loader, "shutdown"):
            try:
                train_loader.shutdown()
            except Exception:
                pass


def preflight(run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError("PHASE 4R-A requires exactly two PPU devices; CPU fallback is forbidden")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        cfg = make_config(run_dir)
        set_seed(SEED)
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        train_loader, _query, _gallery, dataset, _sampler, _val, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        model.to(local_rank)
        if not bool(getattr(model, "detach_reverse_input", False)):
            raise RuntimeError("A2 model did not enable detach_reverse_input")
        loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
        optimizer, _optimizer_center = make_optimizer(cfg, model, center_criterion)
        coverage = parameter_coverage(model, optimizer)
        if not coverage["exactly_once"]:
            raise RuntimeError(f"optimizer coverage failed: {coverage}")
        summary = summary_bio_audit(dataset, cfg)
        smoke = run_smoke(cfg, model, train_loader, dataset, local_rank)
        rank_report = {
            "rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "data": {
                "train": len(dataset.train),
                "query": len(dataset.query),
                "gallery": len(dataset.gallery),
                "train_ids": dataset.num_train_pids,
                "train_clothes": dataset.num_train_clothes,
                "cameras": dataset.num_camera,
                "iterations_per_epoch": len(train_loader),
            },
            "model": {
                "detach_reverse_input": bool(getattr(model, "detach_reverse_input", False)),
                "active_losses": list(cfg.MODEL.LOSS_TYPE),
                "nobio_index": [1, 2],
                "summary_bio": True,
                "reverse_head_count": len(getattr(model, "head_clip_bioReverse", [])),
                "grl_alpha": 1.0,
            },
            "summary_bio": summary,
            "optimizer_coverage": coverage,
            "smoke": smoke,
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE_REQUIRED
        dist.all_gather_object(gathered, rank_report)
        if local_rank == 0:
            payload = {
                "phase": "4R-A",
                "candidate": "A2_detached_reverse_control",
                "created_utc": utc_now(),
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "track": "Repair / research track; not paper-faithful reproduction",
                "numerical_candidate": None,
                "seed": SEED,
                "epochs": EPOCHS,
                "config_snapshot": cfg.dump(),
                "rank_reports": [item for item in gathered if item is not None],
                "passed": all(item["smoke"]["passed"] for item in gathered if item is not None),
            }
            (run_dir / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
            print("PHASE4R_A2_PREFLIGHT", json.dumps({"output": str(run_dir / "preflight.json"), "passed": payload["passed"]}, sort_keys=True), flush=True)
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def amp_summary(run_dir: Path, metrics_path: Path, train_log: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(metrics_path.open()))
    skips = sum(int(float(row["skipped_steps"])) for row in rows)
    updates = len(rows) * 596 - skips
    first_skip = None
    if train_log.exists():
        pattern = re.compile(r"AMP_SKIP epoch=(\d+) iteration=(\d+) scale_before_step=([0-9.]+)")
        for line in train_log.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                first_skip = {
                    "epoch": int(match.group(1)),
                    "iteration": int(match.group(2)),
                    "scale_before_step": float(match.group(3)),
                }
                break
    return {
        "candidate": "A2_detached_reverse_control",
        "total_iterations": len(rows) * 596,
        "optimizer_updates": updates,
        "amp_skips": skips,
        "first_skip": first_skip,
        "min_scaler": min(float(row["amp_scale_min"]) for row in rows),
        "max_scaler_observed": max(float(row["amp_scale_max"]) for row in rows),
        "final_scaler": float(rows[-1]["amp_scale_end"]),
        "nonfinite_loss_count": sum(int(float(row["nonfinite_events"])) for row in rows),
        "initial_scaler": INITIAL_SCALE,
        "dynamic_growth_backoff": True,
        "scale_not_retuned": True,
    }


def train(run_dir: Path, local_rank: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE_REQUIRED:
        raise RuntimeError("PHASE 4R-A requires exactly two PPU devices; CPU fallback is forbidden")
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.exists() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError("A2 preflight is missing or failed")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        cfg = make_config(run_dir)
        set_seed(SEED)
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump())
            (run_dir / "git_diff.patch").write_text(command_output("git", "diff"))
        dist.barrier()
        logger = setup_logger("DIFFER", str(run_dir), if_train=True)
        logger.info("PHASE 4R-A A2 detached reverse control")
        logger.info("Repair/research track; formal reproduction artifacts are not modified")
        logger.info("Active losses: %s", list(cfg.MODEL.LOSS_TYPE))
        logger.info("Summary Bio: %s; NOBIO_INDEX=[1,2]; detach_reverse_input=True", cfg.DATA.SUMMERY_TEXT)
        logger.info("Gradient telemetry: every %d iterations", TELEMETRY_INTERVAL)

        train_loader, _query, _gallery, dataset, _sampler, val_loader, _original = build_dataloader(cfg)
        if len(train_loader) != 596:
            raise RuntimeError(f"expected 596 iterations/epoch, got {len(train_loader)}")
        model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
        if local_rank == 0:
            manifest = {
                "phase": "4R-A",
                "candidate": "A2_detached_reverse_control",
                "created_utc": utc_now(),
                "track": "Repair / research track",
                "paper_faithful": False,
                "git_commit": command_output("git", "rev-parse", "HEAD"),
                "git_status": command_output("git", "status", "--short"),
                "git_diff": str(run_dir / "git_diff.patch"),
                "config": str(run_dir / "config.yml"),
                "preflight": str(preflight_path),
                "data": {
                    "dataset": "ltcc",
                    "train_images": len(dataset.train),
                    "train_ids": dataset.num_train_pids,
                    "train_clothes": dataset.num_train_clothes,
                    "query": len(dataset.query),
                    "gallery": len(dataset.gallery),
                    "image_size": [224, 224],
                    "batch_size": 8,
                    "num_instances": 2,
                    "iterations_per_epoch": len(train_loader),
                    "epochs": EPOCHS,
                },
                "model": {
                    "active_losses": list(cfg.MODEL.LOSS_TYPE),
                    "summary_bio": True,
                    "nobio_index": [1, 2],
                    "factor_names": ["Hair", "Clothing"],
                    "reverse_indexing": "image_feature_reverse[:, i]",
                    "grl_alpha": 1.0,
                    "detach_reverse_input": True,
                    "reverse_heads_train": True,
                    "reverse_gradient_to_shared_representation": False,
                },
                "optimizer_scheduler": {
                    "optimizer": "SGD",
                    "base_lr": 2e-6,
                    "weight_decay": 0.05,
                    "warmup": "unchanged formal linear warmup",
                    "scheduler": "unchanged formal cosine",
                },
                "amp": {
                    "initial_scale": INITIAL_SCALE,
                    "autocast": "FP16",
                    "loss_location": "outside autocast",
                    "dynamic_growth_backoff": True,
                },
                "pretrained": {
                    "path": str(EVA_WEIGHT),
                    "sha256": sha256_file(EVA_WEIGHT),
                    "size_bytes": EVA_WEIGHT.stat().st_size,
                },
                "text": {
                    "train_npz": str(TEXT_NPZ),
                    "train_npz_sha256": sha256_file(TEXT_NPZ),
                    "summary_json": str(SUMMARY_JSON),
                    "summary_json_sha256": sha256_file(SUMMARY_JSON),
                },
                "telemetry": {
                    "interval_iterations": TELEMETRY_INTERVAL,
                    "groups": ["early_eva", "middle_eva", "last_eva", "bio_projection"],
                    "gradient_domains": ["pre_scaled", "scaled", "unscaled"],
                },
                "controls": {
                    "id_only": "56.3775539398 / 31.0721036334",
                    "bio_control": "56.3775539398 / 32.3857364139",
                    "original_full": "53.3163249493 / 29.2657488553",
                },
            }
            (run_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n")
        dist.barrier()
        do_train(cfg, model, train_loader, local_rank, dataset, val_loader=val_loader)
        dist.barrier()
        if local_rank == 0:
            metrics_path = run_dir / "metrics_per_epoch.csv"
            best_path = run_dir / f"{cfg.MODEL.NAME}_best.pth"
            final_path = run_dir / f"{cfg.MODEL.NAME}_final.pth"
            amp_stats = amp_summary(run_dir, metrics_path, run_dir / "train_log.txt")
            (run_dir / "AMP_statistics.json").write_text(json.dumps(amp_stats, indent=2, sort_keys=True) + "\n")
            complete = {
                "phase": "4R-A",
                "candidate": "A2_detached_reverse_control",
                "completed_utc": utc_now(),
                "epochs_completed": EPOCHS,
                "metrics_rows": max(0, sum(1 for _ in metrics_path.open()) - 1),
                "best_checkpoint": str(best_path),
                "final_checkpoint": str(final_path),
                "artifact_audit": {
                    "config": (run_dir / "config.yml").exists(),
                    "manifest": (run_dir / "RUN_MANIFEST.json").exists(),
                    "preflight": preflight_path.exists(),
                    "train_log": (run_dir / "train_log.txt").exists(),
                    "metrics_csv": metrics_path.exists(),
                    "tensorboard_train_dir": (run_dir / "train").is_dir(),
                    "gradient_telemetry_rank0": (run_dir / "gradient_telemetry_rank0.csv").exists(),
                    "gradient_telemetry_rank1": (run_dir / "gradient_telemetry_rank1.csv").exists(),
                    "amp_json": (run_dir / "AMP_statistics.json").exists(),
                    "best_checkpoint": best_path.exists(),
                    "final_checkpoint": final_path.exists(),
                    "git_diff": (run_dir / "git_diff.patch").exists(),
                },
            }
            (run_dir / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
            logger.info("PHASE 4R-A complete: %s", json.dumps(complete, sort_keys=True))
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    run_dir = Path(args.output_dir).resolve()
    local_rank = local_rank_from_args(args.local_rank)
    if args.mode == "preflight":
        preflight(run_dir, local_rank)
    else:
        train(run_dir, local_rank)


if __name__ == "__main__":
    main()
