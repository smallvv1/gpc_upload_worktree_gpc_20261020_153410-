#!/usr/bin/env python3
"""Phase 6C: full-schedule confirmation of the frozen Phase 6B arms.

This script deliberately reuses the Phase 6B relation implementation and the
formal Bio-only trainer.  Seed 1234 resumes all four arms from their complete
epoch-30 rank states.  No relation definition, sampler, loss coefficient,
reliability score, H_ID control, Reverse/GRL, or data visibility rule is
changed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase6b_soft_relation_screen as b6  # noqa: E402

base = b6.base

PHASE = "phase6c_full_schedule_confirmation"
SEED = 1234
TOTAL_EPOCHS = 60
SCREEN_EPOCHS = 30
WARMUP_EPOCHS = 20
WORLD_SIZE = 2
STEPS_PER_EPOCH = b6.STEPS_PER_EPOCH
BATCH_SIZE = b6.BATCH_SIZE
NUM_INSTANCES = b6.NUM_INSTANCES
GEOMETRY_EPOCHS = (30, 40, 50, 60)

PHASE6B_ROOT = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen"
OUT_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
ARTIFACT_ROOT = OUT_ROOT / "artifacts"
REPORT_PATH = ROOT / "reports/PHASE_6C_FULL_SCHEDULE_RELIABILITY_CONFIRMATION.md"
MANIFEST_PATH = OUT_ROOT / "PHASE6C_MANIFEST.json"

ARM_DIRS = {
    "B0": OUT_ROOT / "B0_full",
    "B1": OUT_ROOT / "B1_full",
    "B2": OUT_ROOT / "B2_full",
    "B3": OUT_ROOT / "B3_full",
}
PHASE6B_DIRS = b6.ARM_DIRS
ARM_LABELS = b6.ARM_LABELS

METRIC_FIELDS = b6.METRIC_FIELDS


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.detach().cpu().item()
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def source_sha(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


def checkpoint_source(arm: str) -> Path:
    return PHASE6B_DIRS[arm] / "rank0_state_epoch030.pt"


def audit_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["phase", "epoch", "global_step", "model", "optimizer", "scheduler", "scaler", "rng"]
    scheduler = payload.get("scheduler", {})
    optimizer = payload.get("optimizer", {})
    scaler = payload.get("scaler", {})
    rng = payload.get("rng", {})
    checks = {
        "required_keys": all(key in payload for key in required),
        "epoch_30": payload.get("epoch") == SCREEN_EPOCHS,
        "global_step_17880": payload.get("global_step") == SCREEN_EPOCHS * STEPS_PER_EPOCH,
        "model_state_present": isinstance(payload.get("model"), dict) and len(payload["model"]) == 515,
        "optimizer_state_present": isinstance(optimizer.get("state"), dict) and len(optimizer.get("state", {})) == 515 and len(optimizer.get("param_groups", [])) == 515,
        "scheduler_state_present": isinstance(scheduler, dict) and scheduler.get("t_initial") == TOTAL_EPOCHS and scheduler.get("warmup_t") == WARMUP_EPOCHS and scheduler.get("t_in_epochs") is True,
        "scaler_state_present": all(key in scaler for key in ("scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker")),
        "rng_state_present": all(key in rng for key in ("python", "numpy", "torch_cpu", "torch_cuda")),
    }
    summary = {
        "path": path,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "payload_phase": payload.get("phase"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "model_key_count": len(payload.get("model", {})),
        "optimizer_state_count": len(optimizer.get("state", {})),
        "optimizer_param_group_count": len(optimizer.get("param_groups", [])),
        "scheduler_t_initial": scheduler.get("t_initial"),
        "scheduler_warmup_t": scheduler.get("warmup_t"),
        "scheduler_t_mul": scheduler.get("t_mul"),
        "scheduler_cycle_limit": scheduler.get("cycle_limit"),
        "scaler_scale": scaler.get("scale"),
        "rng_keys": sorted(rng.keys()) if isinstance(rng, dict) else [],
        "checks": checks,
        "ready": all(checks.values()),
    }
    del payload
    gc.collect()
    return summary


def resume_audit() -> dict[str, Any]:
    ensure_root()
    arm_results: dict[str, Any] = {}
    all_ready = True
    for arm in ARM_DIRS:
        source_dir = PHASE6B_DIRS[arm]
        rank_results = {}
        for rank in range(WORLD_SIZE):
            path = source_dir / f"rank{rank}_state_epoch030.pt"
            rank_results[str(rank)] = audit_payload(path) if path.is_file() else {"path": path, "ready": False, "missing": True}
        metadata_path = source_dir / "checkpoint_epoch030.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        result = {
            "phase6b_dir": source_dir,
            "checkpoint_metadata": metadata,
            "checkpoint_metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
            "rank_states": rank_results,
            "model_screen_final": source_sha(source_dir / "model_screen_final.pth"),
            "ready": all(item.get("ready", False) for item in rank_results.values()) and metadata.get("epoch") == SCREEN_EPOCHS,
        }
        arm_results[arm] = result
        all_ready = all_ready and result["ready"]

    preflight = json.loads((PHASE6B_ROOT / "PREFLIGHT.json").read_text())
    method = {
        "common_init": source_sha(Path(preflight["common_init"]["path"])),
        "common_init_sha256": preflight["common_init"]["sha256"],
        "phase6b_manifest": source_sha(PHASE6B_ROOT / "PHASE6B_MANIFEST.json"),
        "preflight": source_sha(PHASE6B_ROOT / "PREFLIGHT.json"),
    }
    result = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "resume_mode": "RESUME_MODE" if all_ready else "FULL_RESTART_MODE",
        "all_four_exact_resume_ready": all_ready,
        "method_sources": method,
        "arms": arm_results,
        "decision": "Resume all B0-B3 from epoch30 rank state to epoch60." if all_ready else "Do not mix resume and restart; all arms must restart from frozen common init.",
    }
    lines = [
        "# Phase 6C resume audit",
        "",
        f"Generated: {result['created_utc']}.",
        "",
        f"Decision: {result['resume_mode']}.",
        "",
        "Each arm is accepted only when both rank states contain model, optimizer, scheduler, GradScaler, epoch, global_step, and all four RNG families.",
        "",
        "| Arm | Rank0 ready | Rank1 ready | E30 | Global step | Scheduler | Scaler | RNG |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for arm, item in arm_results.items():
        r0 = item["rank_states"].get("0", {})
        r1 = item["rank_states"].get("1", {})
        checks = r0.get("checks", {})
        lines.append(
            f"| {arm} | {r0.get('ready', False)} | {r1.get('ready', False)} | "
            f"{r0.get('epoch', '')} | {r0.get('global_step', '')} | "
            f"{checks.get('scheduler_state_present', False)} | "
            f"{checks.get('scaler_state_present', False)} | "
            f"{checks.get('rng_state_present', False)} |"
        )
    lines += [
        "",
        f"All four arms exact-resume ready: {all_ready}.",
        "",
        "The scheduler is the formal 60-epoch scheduler with t_initial=60 and warmup_t=20. Epoch 31 will call scheduler.step(31) after loading the E30 state; no 30-to-60 scheduler is constructed.",
        "",
        "The E30 model-only files are not used as a substitute for rank states. Optimizer, scheduler, scaler, and RNG state are restored from the rank state.",
    ]
    (ARTIFACT_ROOT / "resume_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ARTIFACT_ROOT / "resume_audit.json", result)
    return result


def frozen_method_audit() -> None:
    ensure_root()
    preflight = json.loads((PHASE6B_ROOT / "PREFLIGHT.json").read_text())
    manifest = json.loads((PHASE6B_ROOT / "PHASE6B_MANIFEST.json").read_text())
    lines = [
        "# Frozen Phase 6B method used by Phase 6C",
        "",
        "The executed Phase 6B code and preflight are authoritative.",
        "",
        "## Frozen parameters",
        "",
        f"- SR3 candidate: {preflight['relation_source'] if 'relation_source' in preflight else 'SR3_CONSENSUS_MUTUAL_INTRA'}",
        "- SR3 candidate name: SR3_CONSENSUS_MUTUAL_INTRA",
        f"- Relation weight floor: {preflight['relation_weight_floor']}",
        "- B1 mapping: w=1.0.",
        "- B2 mapping: w=0.5 + 0.5 * frozen SR3 score.",
        "- B3 mapping: the same unique image-pair weight multiset as B2, shuffled within PID.",
        f"- Relation lambda: {preflight['lambda_relation']}",
        "- Relation representation: normalized feat[0] student identity feature.",
        "- Pair truth: same PID and different clothes.",
        "- Pair implementation: exact unordered image-pair lookup in the compact Phase6A3 candidate bank.",
        "- Pairs outside the frozen candidate bank remain in the base Bio objective and receive no added relation term.",
        "- All images, PIDs, and clothing states remain visible; no H_ID, DS, DW, online update, Reverse, or GRL.",
        "",
        "## Source checksums",
        "",
        f"- Phase6B manifest: {sha256_file(PHASE6B_ROOT / 'PHASE6B_MANIFEST.json')}",
        f"- Phase6B preflight: {sha256_file(PHASE6B_ROOT / 'PREFLIGHT.json')}",
        f"- SR3 score source: {preflight['source_checksums'][str(b6.A3_SCORES)]['sha256']}",
        f"- Common init: {preflight['common_init']['sha256']}",
        "",
        "Any discrepancy between an earlier design note and the executed code is resolved in favor of phase6b_soft_relation_screen.py and the recorded preflight.",
    ]
    (ARTIFACT_ROOT / "frozen_phase6b_method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ARTIFACT_ROOT / "phase6b_method_snapshot.json", {
        "phase6b_manifest": manifest,
        "phase6b_preflight": preflight,
        "actual_runner": str(ROOT / "scripts/phase6b_soft_relation_screen.py"),
        "actual_runner_sha256": sha256_file(ROOT / "scripts/phase6b_soft_relation_screen.py"),
    })


def common_config_audit() -> dict[str, Any]:
    ensure_root()
    configs = {}
    manifests = {}
    for arm in ARM_DIRS:
        source_dir = PHASE6B_DIRS[arm]
        config = source_dir / "config.yml"
        run_manifest = source_dir / "run_manifest.json"
        configs[arm] = source_sha(config)
        manifests[arm] = json.loads(run_manifest.read_text())
    config_hashes = {item["sha256"] for item in configs.values()}
    init_hashes = {item["common_init"]["sha256"] for item in manifests.values()}
    model_hashes = {item["common_model_hash_across_ranks"] for item in manifests.values()}
    result = {
        "phase": PHASE,
        "configs": configs,
        "config_byte_identical": len(config_hashes) == 1,
        "common_init_hashes": sorted(init_hashes),
        "common_init_identical": len(init_hashes) == 1,
        "common_model_hashes": sorted(model_hashes),
        "common_model_hash_identical": len(model_hashes) == 1,
        "formal_scheduler_epochs": sorted({int(item["formal_scheduler_epochs"]) for item in manifests.values()}),
        "warmup_epochs": sorted({int(item["warmup_epochs"]) for item in manifests.values()}),
        "steps_per_epoch": sorted({int(item["steps_per_epoch"]) for item in manifests.values()}),
        "relation_source_candidates": sorted({item["relation_source_sr3_candidate"] for item in manifests.values()}),
        "reverse_values": sorted({item["reverse"] for item in manifests.values()}),
    }
    write_json(ARTIFACT_ROOT / "common_config_audit.json", result)
    return result


def frozen_b3_mapping() -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Load the Phase6B B3 mapping as an immutable audit source.

    The Phase6B artifact contains the exact image-pair-to-weight assignment
    used by the shuffled control.  Phase6C reconstructs the same assignment
    from the frozen source and seed; it must not create a new shuffle.
    """
    source = PHASE6B_ROOT / "artifacts/relation_source_rows.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = b6.read_csv(source)
    mapping: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (min(int(row["index_a"]), int(row["index_b"])), max(int(row["index_a"]), int(row["index_b"])))
        value = float(row["b3_weight"])
        previous = mapping.get(key)
        if previous is not None and not math.isclose(previous, value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"frozen B3 mapping has conflicting duplicate value for {key}")
        mapping[key] = value
    canonical = [
        {"index_a": key[0], "index_b": key[1], "b3_weight": mapping[key]}
        for key in sorted(mapping)
    ]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return mapping, {
        "path": source,
        "sha256": sha256_file(source),
        "rows": len(rows),
        "unique_image_pairs": len(mapping),
        "mapping_sha256": digest,
        "seed": SEED,
    }


def assert_frozen_b3_mapping(bank: dict[str, Any]) -> dict[str, Any]:
    """Fail before training if deterministic reconstruction differs from E1-E30."""
    expected, source_audit = frozen_b3_mapping()
    actual = {tuple(key): float(value) for key, value in bank["exact_weight_b3"].items()}
    if set(actual) != set(expected):
        raise RuntimeError("B3 exact image-pair key set differs from the frozen Phase6B mapping")
    max_abs_delta = max((abs(actual[key] - expected[key]) for key in expected), default=0.0)
    if max_abs_delta > 1e-12:
        raise RuntimeError(f"B3 mapping changed; maximum absolute weight delta={max_abs_delta}")
    source_audit.update({
        "reconstructed_unique_image_pairs": len(actual),
        "reconstructed_mapping_sha256": hashlib.sha256(json.dumps([
            {"index_a": key[0], "index_b": key[1], "b3_weight": actual[key]}
            for key in sorted(actual)
        ], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "max_abs_delta": max_abs_delta,
        "match": True,
    })
    return source_audit


def b3_resume_audit() -> dict[str, Any]:
    """Audit only the B3 E30 resume source and frozen continuation contract."""
    ensure_root()
    arm = "B3"
    source_dir = PHASE6B_DIRS[arm]
    run_dir = ARM_DIRS[arm]
    checkpoint_metadata_path = source_dir / "checkpoint_epoch030.json"
    phase6b_manifest_path = PHASE6B_ROOT / "PHASE6B_MANIFEST.json"
    preflight_path = PHASE6B_ROOT / "PREFLIGHT.json"
    phase6b_run_manifest_path = source_dir / "run_manifest.json"
    phase6c_run_manifest_path = run_dir / "run_manifest.json"
    prefix_manifest_path = run_dir / "phase6b_prefix_manifest.json"
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text())
    phase6b_manifest = json.loads(phase6b_manifest_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    phase6b_run_manifest = json.loads(phase6b_run_manifest_path.read_text())
    phase6c_run_manifest = json.loads(phase6c_run_manifest_path.read_text())
    prefix_manifest = json.loads(prefix_manifest_path.read_text())
    rank_results = {
        str(rank): audit_payload(source_dir / f"rank{rank}_state_epoch030.pt")
        for rank in range(WORLD_SIZE)
    }
    matched = phase6b_manifest["matched_control_audit"]
    matched_b2 = matched["arms"]["B2"]
    matched_b3 = matched["arms"]["B3"]
    mapping_audit = frozen_b3_mapping()[1]

    prefix_metrics = read_csv(run_dir / "metrics_per_epoch.csv")
    prefix_amp = read_csv(run_dir / "amp_statistics.csv")
    prefix_relation = read_csv(run_dir / "relation_exposure.csv")
    frozen_checks = {
        "ltcc_train_images_9576": matched_b3["active_shape_values"] == [["9576", "77", "256"]],
        "pids_77": matched_b3["active_shape_values"][0][1] == "77",
        "clothing_states_256": matched_b3["active_shape_values"][0][2] == "256",
        "world_size_2_ppus": WORLD_SIZE == 2,
        "batch_size_8": phase6b_run_manifest.get("batch_size") == 8,
        "num_instances_2": phase6b_run_manifest.get("num_instances") == NUM_INSTANCES == 2,
        "optimizer_sgd": phase6b_run_manifest.get("optimizer") == "SGD",
        "base_lr_2e_minus_6": phase6b_run_manifest.get("base_lr") == 2e-6,
        "weight_decay_0_05": phase6b_run_manifest.get("weight_decay") == 0.05,
        "formal_scheduler_60": phase6b_run_manifest.get("formal_scheduler_epochs") == TOTAL_EPOCHS,
        "warmup_20": phase6b_run_manifest.get("warmup_epochs") == WARMUP_EPOCHS,
        "planned_updates_596": phase6b_run_manifest.get("steps_per_epoch") == STEPS_PER_EPOCH == 596,
        "amp_frozen_by_runner": True,
        "reverse_off": phase6b_run_manifest.get("reverse") == "OFF",
        "grl_off": phase6b_run_manifest.get("grl") == "OFF",
        "dw_off": phase6b_run_manifest.get("dw") == "OFF",
        "ds_off": phase6b_run_manifest.get("active_set_withholding") == "OFF",
        "h_id_off": phase6b_run_manifest.get("h_id") == "OFF",
        "online_reliability_off": phase6b_run_manifest.get("online_reliability") == "OFF",
        "relation_candidate_sr3": phase6b_run_manifest.get("relation_source_sr3_candidate") == "SR3_CONSENSUS_MUTUAL_INTRA",
        "relation_loss_frozen": phase6b_run_manifest.get("relation_loss") == "positive-only same-PID/different-clothes cosine loss",
        "w_min_0_5": phase6b_run_manifest.get("relation_weight_floor") == b6.RELATION_W_MIN == 0.5,
        "lambda_rel_0_1": phase6b_run_manifest.get("lambda_relation") == b6.LAMBDA_REL == 0.1,
        "b3_definition_frozen": phase6b_run_manifest.get("relation_weight") == "within-PID shuffled SR3",
        "b2_b3_candidate_bank_equal": matched["B2_B3_unique_pair_weight_multiset_equal"],
        "b2_b3_weight_multiset_equal": matched["weight_audit"]["unique_pair_weight_multiset_equal"],
        "phase6b_seed_1234": phase6b_run_manifest.get("seed") == SEED,
        "phase6c_seed_1234": phase6c_run_manifest.get("seed") == SEED,
        "phase6c_resume_start_31": phase6c_run_manifest.get("start_epoch") == 31,
        "phase6c_resume_end_60": phase6c_run_manifest.get("end_epoch") == TOTAL_EPOCHS,
        "config_byte_identical": sha256_file(source_dir / "config.yml") == sha256_file(run_dir / "config.yml"),
        "common_init_sha256_preserved": phase6b_run_manifest["common_init"]["sha256"] == preflight["common_init"]["sha256"],
        "prefix_rows_are_exactly_e1_e30": len(prefix_metrics) == SCREEN_EPOCHS and [int(row["epoch"]) for row in prefix_metrics] == list(range(1, SCREEN_EPOCHS + 1)),
        "prefix_amp_rows_are_exactly_e1_e30": len(prefix_amp) == SCREEN_EPOCHS and [int(row["epoch"]) for row in prefix_amp] == list(range(1, SCREEN_EPOCHS + 1)),
        "prefix_relation_rows_are_exactly_e1_e30": len(prefix_relation) == SCREEN_EPOCHS and [int(row["epoch"]) for row in prefix_relation] == list(range(1, SCREEN_EPOCHS + 1)),
        "no_b3_phase6c_completion_marker": not (run_dir / "RUN_COMPLETE.json").exists(),
    }
    rank_checks = {
        rank: {
            "model": result["checks"]["model_state_present"],
            "optimizer": result["checks"]["optimizer_state_present"],
            "scheduler": result["checks"]["scheduler_state_present"],
            "grad_scaler": result["checks"]["scaler_state_present"],
            "epoch": result["checks"]["epoch_30"],
            "global_step": result["checks"]["global_step_17880"],
            "rng": result["checks"]["rng_state_present"],
        }
        for rank, result in rank_results.items()
    }
    checks = {
        "checkpoint_metadata_epoch_30": checkpoint_metadata.get("epoch") == SCREEN_EPOCHS,
        "checkpoint_metadata_global_step_17880": checkpoint_metadata.get("global_step") == SCREEN_EPOCHS * STEPS_PER_EPOCH,
        "rank0_complete_resume_state": all(rank_checks["0"].values()),
        "rank1_complete_resume_state": all(rank_checks["1"].values()),
        "rank_states_match_prefix_manifest": all(
            prefix_manifest.get(f"phase6b_rank{rank}_state_sha256") == rank_results[str(rank)]["sha256"]
            for rank in range(WORLD_SIZE)
        ),
        "frozen_b3_mapping_source_present": mapping_audit["rows"] == 43839 and mapping_audit["unique_image_pairs"] == 43186,
        "frozen_checks": all(frozen_checks.values()),
    }
    checks["ready"] = all(checks.values())
    result = {
        "phase": PHASE,
        "arm": arm,
        "created_utc": utc_now(),
        "resume_source": source_dir,
        "checkpoint_metadata": checkpoint_metadata,
        "rank_states": rank_results,
        "rank_restore_fields": rank_checks,
        "frozen_checks": frozen_checks,
        "frozen_b3_mapping": mapping_audit,
        "checks": checks,
        "ready": checks["ready"],
        "resume_rule": "Load model, optimizer, scheduler, GradScaler, epoch=30, global_step=17880, and RNG from both rank states; call scheduler.step(31) and train only E31-E60.",
        "amp_rule": "Record skipped_steps, found_inf_count, nonfinite_events, scaler scale, skip epochs, and skip iterations without changing AMP settings.",
        "no_new_shuffle": "The Phase6B relation_source_rows.csv B3 assignment is the frozen source; the Phase6C runner must match it exactly before the first E31 update.",
    }
    write_json(ARTIFACT_ROOT / "B3_resume_e30_audit.json", result)
    lines = [
        "# B3 Phase 6C E30 Resume Audit",
        "",
        f"Generated: `{result['created_utc']}`.",
        "",
        f"Decision: **{'READY — RESUME E31-E60' if result['ready'] else 'BLOCKED — DO NOT TRAIN'}**.",
        "",
        "This audit covers B3 only. B0, B1, and B2 are not loaded or retrained by the B3 continuation.",
        "",
        "## Checkpoint restoration",
        "",
        "| Rank | Model | Optimizer | Scheduler | GradScaler | Epoch 30 | Global step 17880 | RNG |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for rank in ("0", "1"):
        item = rank_checks[rank]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rank, item["model"], item["optimizer"], item["scheduler"], item["grad_scaler"],
            item["epoch"], item["global_step"], item["rng"],
        ))
    lines += [
        "",
        f"- Checkpoint metadata E30: **{checks['checkpoint_metadata_epoch_30']}**.",
        f"- Checkpoint metadata global step 17880: **{checks['checkpoint_metadata_global_step_17880']}**.",
        f"- Both rank states match the Phase6C prefix manifest: **{checks['rank_states_match_prefix_manifest']}**.",
        "- Optimizer, scheduler, and GradScaler are restored from rank state; model-only loading is not used.",
        "- The formal 60-epoch scheduler is loaded from E30; E31 calls `scheduler.step(31)`.",
        "",
        "## Frozen B3 contract",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for key, value in frozen_checks.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        f"- Frozen B3 mapping source: `{mapping_audit['path']}`.",
        f"- Frozen mapping source SHA256: `{mapping_audit['sha256']}`.",
        f"- Frozen mapping digest: `{mapping_audit['mapping_sha256']}`.",
        "- The B3 relation↔reliability assignment is the Phase6B within-PID shuffled mapping; no new shuffle is permitted.",
        "",
        "## Continuation boundary",
        "",
        f"- Existing B3 Phase6C prefix rows: E1–E30 only (**{checks['frozen_checks']}**).",
        "- Required continuation: E31–E60 only.",
        "- AMP settings remain unchanged; all skip/found-inf/scaler events are recorded.",
        "",
        f"Overall audit ready: **{result['ready']}**.",
    ]
    (ARTIFACT_ROOT / "B3_resume_e30_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE,
        "epoch": epoch,
        "global_step": global_step,
        "model": b6.cpu_state_dict(model),
        "optimizer": base.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": base.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def copy_prefix_artifacts(arm: str, run_dir: Path) -> dict[str, Any]:
    source_dir = PHASE6B_DIRS[arm]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Phase6C run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics_per_epoch.csv", "amp_statistics.csv", "relation_exposure.csv", "reliability_source_sha256.json"):
        shutil.copyfile(source_dir / filename, run_dir / filename)
    model_link = run_dir / "model_best.pth"
    model_link.symlink_to((source_dir / "model_best.pth").resolve())
    (run_dir / "model_epoch030_source.pth").symlink_to((source_dir / "model_screen_final.pth").resolve())
    source_manifest = {
        "phase6b_run_dir": source_dir,
        "phase6b_metrics_sha256": sha256_file(source_dir / "metrics_per_epoch.csv"),
        "phase6b_rank0_state_sha256": sha256_file(source_dir / "rank0_state_epoch030.pt"),
        "phase6b_rank1_state_sha256": sha256_file(source_dir / "rank1_state_epoch030.pt"),
        "phase6b_model_screen_final_sha256": sha256_file(source_dir / "model_screen_final.pth"),
        "resume_source_rank0": source_dir / "rank0_state_epoch030.pt",
        "resume_source_rank1": source_dir / "rank1_state_epoch030.pt",
        "prefix_rows": len(read_csv(source_dir / "metrics_per_epoch.csv")),
    }
    write_json(run_dir / "phase6b_prefix_manifest.json", source_manifest)
    return source_manifest


def prepare_phase6c_dirs() -> None:
    ensure_root()
    for arm, run_dir in ARM_DIRS.items():
        copy_prefix_artifacts(arm, run_dir)
        cfg = b6.configure()
        (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
        diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
        (run_dir / "git_diff.patch").write_bytes(diff)
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        (run_dir / "git_status.txt").write_text(status, encoding="utf-8")
        write_json(run_dir / "run_manifest.json", {
            "phase": PHASE,
            "arm": arm,
            "label": ARM_LABELS[arm],
            "mode": "RESUME_MODE",
            "start_epoch": SCREEN_EPOCHS + 1,
            "end_epoch": TOTAL_EPOCHS,
            "formal_scheduler_epochs": TOTAL_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "seed": SEED,
            "reverse": "OFF",
            "grl": "OFF",
            "dw": "OFF",
            "h_id": "OFF",
            "ds": "OFF",
            "online_reliability": "OFF",
            "relation_source": str(b6.A3_ROOT),
            "relation_source_sr3_candidate": "SR3_CONSENSUS_MUTUAL_INTRA",
            "lambda_relation": b6.LAMBDA_REL,
            "relation_weight_floor": b6.RELATION_W_MIN,
            "resume_source": PHASE6B_DIRS[arm] / "rank{rank}_state_epoch030.pt",
        })


def relation_geometry(model: torch.nn.Module, cfg: Any, dataset: Any, bank: dict[str, Any], device: torch.device, transform_test: Any, checkpoint: int) -> list[dict[str, Any]]:
    """Compute train-only geometry for the frozen candidate relation bank."""
    model.eval()
    image_dataset = base.ImageDataset(dataset.train, transform=transform_test)
    loader = DataLoader(image_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    feature_bank: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            # The released model has different return signatures in train and
            # eval mode: train returns (score, [identity_feature, ...]), while
            # eval returns the identity feature tensor directly.  Geometry is
            # an eval-only, train-split diagnostic and must use the same
            # identity feature as feat[0] in the training relation loss.
            if torch.is_tensor(output):
                identity_feature = output
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                _score, eval_feat = output
                if isinstance(eval_feat, (tuple, list)):
                    identity_feature = eval_feat[0]
                else:
                    identity_feature = eval_feat
            else:
                raise TypeError(
                    "Unexpected model output in eval geometry probe: "
                    f"{type(output)!r}"
                )
            values = F.normalize(identity_feature.float(), dim=-1).detach().cpu().numpy()
            paths = batch["image_path"]
            for path, value in zip(paths, values):
                feature_bank[int(bank["path_to_index"][str(path)])] = value
    if len(feature_bank) != len(dataset.train):
        raise RuntimeError(f"relation geometry feature count mismatch: {len(feature_bank)} vs {len(dataset.train)}")
    feature_matrix = np.stack([feature_bank[index] for index in range(len(dataset.train))]).astype(np.float32)
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    unique_pids = sorted(set(pids.tolist()))
    centers = {}
    for pid in unique_pids:
        center = feature_matrix[pids == pid].mean(axis=0)
        centers[pid] = center / max(np.linalg.norm(center), 1e-12)

    exact_items = sorted(bank["exact_lookup"].items())
    scores = np.asarray([float(value) for _key, value in exact_items], dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    quartile = {}
    for position, index in enumerate(order):
        quartile[exact_items[int(index)][0]] = f"Q{min(4, position * 4 // max(1, len(order)) + 1)}"
    records = []
    for (index_a, index_b), score in exact_items:
        pid = int(pids[index_a])
        sim_pos = float(np.dot(feature_matrix[index_a], feature_matrix[index_b]))
        negative_a = max(float(np.dot(feature_matrix[index_a], centers[other])) for other in unique_pids if other != pid)
        negative_b = max(float(np.dot(feature_matrix[index_b], centers[other])) for other in unique_pids if other != pid)
        for anchor, negative in ((index_a, negative_a), (index_b, negative_b)):
            records.append({
                "checkpoint": f"E{checkpoint}",
                "sr3_quartile": quartile[(index_a, index_b)],
                "pair_score": float(score),
                "pid": pid,
                "clothes_a": int(clothes[index_a]),
                "clothes_b": int(clothes[index_b]),
                "anchor_index": int(anchor),
                "positive_cosine": sim_pos,
                "nearest_negative_center_cosine": negative,
                "positive_negative_margin": sim_pos - negative,
                "ordering_correct": int(sim_pos > negative),
            })
    output = []
    for stratum in ["ALL", "Q1", "Q2", "Q3", "Q4"]:
        selected = records if stratum == "ALL" else [row for row in records if row["sr3_quartile"] == stratum]
        if not selected:
            continue
        output.append({
            "checkpoint": f"E{checkpoint}",
            "sr3_quartile": stratum,
            "relation_count": len(selected),
            "pid_count": len({row["pid"] for row in selected}),
            "clothing_state_count": len({(row["pid"], row["clothes_a"]) for row in selected} | {(row["pid"], row["clothes_b"]) for row in selected}),
            "positive_cosine_mean": float(np.mean([row["positive_cosine"] for row in selected])),
            "positive_cosine_median": float(np.median([row["positive_cosine"] for row in selected])),
            "negative_center_cosine_mean": float(np.mean([row["nearest_negative_center_cosine"] for row in selected])),
            "positive_negative_margin_mean": float(np.mean([row["positive_negative_margin"] for row in selected])),
            "positive_negative_margin_median": float(np.median([row["positive_negative_margin"] for row in selected])),
            "ordering_correct_rate": float(np.mean([row["ordering_correct"] for row in selected])),
        })
    model.train()
    return output


def open_append_csv(path: Path, fields: list[str]):
    stream = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    return stream, writer


def train_arm(arm: str) -> None:
    if arm not in ARM_DIRS:
        raise ValueError(arm)
    audit = json.loads((ARTIFACT_ROOT / "resume_audit.json").read_text())
    if audit.get("resume_mode") != "RESUME_MODE":
        raise RuntimeError("Phase6C is not in exact RESUME_MODE; do not mix resume and restart")
    rank, device = b6.init_dist()
    run_dir = ARM_DIRS[arm]
    metrics_file = amp_stream = relation_stream = amp_event_stream = None
    writer = logger = None
    try:
        cfg = b6.configure()
        b6.set_seed(SEED)
        state, init_audit = b6.load_common_init()
        model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, state, device)
        source_checkpoint = PHASE6B_DIRS[arm] / f"rank{rank}_state_epoch030.pt"
        payload = base.load_rank_state(source_checkpoint, model, optimizer, scheduler, scaler, device)
        if int(payload["epoch"]) != SCREEN_EPOCHS or int(payload["global_step"]) != SCREEN_EPOCHS * STEPS_PER_EPOCH:
            raise RuntimeError(f"invalid resume payload for {arm}: epoch={payload.get('epoch')} step={payload.get('global_step')}")
        local_hash = b6.model_parameter_hash(model)
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"resume model differs across ranks for {arm}")
        dataset = base.build_dataset(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        transform_train, transform_test = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        state_all = b6.full_active_state(dataset)
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        b3_mapping_audit = assert_frozen_b3_mapping(bank) if arm == "B3" else None
        val_loader = base.build_val_loader(cfg, dataset)

        if rank == 0:
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            manifest.update({"common_init": init_audit, "common_model_hash_across_ranks": local_hash, "resume_loaded_rank_state_sha256": sha256_file(source_checkpoint)})
            if b3_mapping_audit is not None:
                manifest["frozen_b3_mapping_audit"] = b3_mapping_audit
            write_json(run_dir / "run_manifest.json", manifest)
            logger = base.logger_for(run_dir / "train_log_rank0.txt")
            writer = SummaryWriter(str(run_dir / "TensorBoard"))
            metrics_file, metrics_writer = open_append_csv(run_dir / "metrics_per_epoch.csv", METRIC_FIELDS)
            amp_stream, amp_writer = open_append_csv(run_dir / "amp_statistics.csv", ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            relation_stream, relation_writer = open_append_csv(run_dir / "relation_exposure.csv", ["arm", "epoch", "all_images_available", "all_states_available", "all_pids_available", "sampled_unique_images", "sampled_unique_states", "sampled_unique_pids", "relation_active_batches", "relation_active_batch_rate", "relation_pairs", "relation_unique_state_pairs", "relation_weight_mean", "relation_weight_std", "relation_weight_min", "relation_weight_max", "optimizer_updates"])
            amp_event_path = run_dir / "amp_skip_events.csv"
            amp_event_stream = amp_event_path.open("a", newline="", encoding="utf-8")
            amp_event_writer = csv.DictWriter(
                amp_event_stream,
                fieldnames=["rank", "epoch", "iteration", "global_step", "skipped_step", "found_inf", "nonfinite_event", "amp_scale_before", "amp_scale_after"],
            )
            if amp_event_path.stat().st_size == 0:
                amp_event_writer.writeheader()
                amp_event_stream.flush()
        else:
            metrics_writer = amp_writer = relation_writer = amp_event_writer = None

        best_meta_path = run_dir / "best_state.json"
        if best_meta_path.is_file():
            best_meta = json.loads(best_meta_path.read_text())
            best_r1 = float(best_meta["best_cc_rank1"])
            best_epoch = int(best_meta["best_epoch"])
        else:
            old_best = json.loads((PHASE6B_DIRS[arm] / "best_state.json").read_text())
            best_r1 = float(old_best["best_cc_rank1"])
            best_epoch = int(old_best["best_epoch"])
            if rank == 0:
                write_json(best_meta_path, old_best)
        global_step = int(payload["global_step"])
        geometry_rows: list[dict[str, Any]] = []
        if rank == 0:
            geometry_rows.extend(relation_geometry(model.module, cfg, dataset, bank, device, transform_test, SCREEN_EPOCHS))
            write_csv(run_dir / "relation_geometry.csv", geometry_rows)
        dist.barrier()

        for epoch in range(SCREEN_EPOCHS + 1, TOTAL_EPOCHS + 1):
            scheduler.step(epoch)
            loader, plan_meta = base.make_epoch_loader(dataset, transform_train, state_all, index_map, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if rank == 0:
                if any(meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"] for meta in gathered_meta):
                    raise RuntimeError(f"runtime sampler audit failed at epoch {epoch}")
                if set(gathered_meta[0]["covered_pids"]) | set(gathered_meta[1]["covered_pids"]) != set(range(b6.NUM_CLASSES)):
                    raise RuntimeError(f"runtime full PID coverage failed at epoch {epoch}")
            local_sampled_images = {path_to_index[str(source["image_path"])] for source, _seed in loader.dataset.samples}
            local_sampled_states = {(int(source["pid"]), int(source["clothes_id"])) for source, _seed in loader.dataset.samples}
            local_sampled_pids = {int(source["pid"]) for source, _seed in loader.dataset.samples}
            local_stats = {"active_batches": 0, "pair_count": 0, "weights": [], "keys": []}
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation_raw", "relation_applied", "id_acc", "bio_acc")}
            local_skips = 0
            local_nonfinite = 0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            local_amp_events = []
            iterator = iter(loader)
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = b6.forward_batch(model, loss_fn, batch, device, bank, path_to_index, arm, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    local_nonfinite += 1
                    raise RuntimeError(f"non-finite loss at arm={arm}, epoch={epoch}, iteration={iteration}")
                scaler.scale(total).backward()
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                inf = base.found_inf(scaler, optimizer)
                local_skips += int(inf > 0)
                local_nonfinite += int(inf > 0)
                scaler.update()
                scale_after = float(scaler.get_scale())
                scale_min = min(scale_min, scale_after)
                scale_max = max(scale_max, scale_after)
                if inf > 0:
                    local_amp_events.append({
                        "epoch": epoch,
                        "iteration": iteration,
                        "global_step": global_step,
                        "skipped_step": 1,
                        "found_inf": int(inf),
                        "nonfinite_event": 1,
                        "amp_scale_before": scale_before,
                        "amp_scale_after": scale_after,
                    })
                for key in epoch_sum:
                    epoch_sum[key] += base.scalar(result[key])
                info = result["relation_info"]
                if info["active"]:
                    local_stats["active_batches"] += 1
                    local_stats["pair_count"] += int(info["pair_count"])
                    local_stats["weights"].extend(float(value) for value in info["weights"])
                    local_stats["keys"].extend(info["keys"])
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", base.scalar(total), global_step)
                    writer.add_scalar("train/loss_relation_raw", base.scalar(result["relation_raw"]), global_step)
            gathered_amp_events = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_amp_events, local_amp_events)
            if rank == 0 and amp_event_writer is not None:
                for event_rank, event_rows in enumerate(gathered_amp_events):
                    for event in event_rows:
                        amp_event_writer.writerow({"rank": event_rank, **event})
                amp_event_stream.flush()
            del iterator, loader
            gc.collect()
            reduced = b6.reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            count_tensor = torch.tensor([
                local_stats["active_batches"], local_stats["pair_count"],
                sum(local_stats["weights"]), sum(value * value for value in local_stats["weights"]),
            ], dtype=torch.float64, device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            min_weight = torch.tensor(min(local_stats["weights"]) if local_stats["weights"] else float("inf"), dtype=torch.float64, device=device)
            max_weight = torch.tensor(max(local_stats["weights"]) if local_stats["weights"] else -float("inf"), dtype=torch.float64, device=device)
            dist.all_reduce(min_weight, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_weight, op=dist.ReduceOp.MAX)
            gathered_sets = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_sets, (local_sampled_images, local_sampled_states, local_sampled_pids, local_stats["keys"]))
            sampled_images = set().union(*(item[0] for item in gathered_sets))
            sampled_states = set().union(*(item[1] for item in gathered_sets))
            sampled_pids = set().union(*(item[2] for item in gathered_sets))
            sampled_relation_keys = set().union(*(set(item[3]) for item in gathered_sets))
            relation_batches = int(count_tensor[0].item())
            relation_pairs = int(count_tensor[1].item())
            weight_sum = float(count_tensor[2].item())
            weight_sq_sum = float(count_tensor[3].item())
            weight_mean = weight_sum / relation_pairs if relation_pairs else 0.0
            weight_var = max(weight_sq_sum / relation_pairs - weight_mean * weight_mean, 0.0) if relation_pairs else 0.0
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone()
            nonfinite_sum = amp_tensor[1].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_tensor = amp_tensor[2].clone()
            max_tensor = amp_tensor[3].clone()
            end_tensor = amp_tensor[4].clone()
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(end_tensor, op=dist.ReduceOp.MAX)
            amp_stats = {
                "skipped_steps": int(skip_sum.item()),
                "found_inf_count": int(skip_sum.item()),
                "nonfinite_events": int(nonfinite_sum.item()),
                "amp_scale_min": float(min_tensor.item()),
                "amp_scale_max": float(max_tensor.item()),
                "amp_scale_end": float(end_tensor.item()),
            }
            dist.barrier()
            if rank == 0:
                metrics = base.evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            holder = [metrics]
            dist.broadcast_object_list(holder, src=0)
            metrics = holder[0]
            row = {
                "epoch": epoch,
                "train_loss": reduced["total"],
                "ce_loss": reduced["ce"],
                "triplet_loss": reduced["triplet"],
                "bio_loss": reduced["bio"],
                "relation_raw_loss": reduced["relation_raw"],
                "relation_applied_loss": reduced["relation_applied"],
                "relation_lambda": 0.0 if arm == "B0" else b6.LAMBDA_REL,
                "lr": float(scheduler._get_lr(epoch)[0]),
                "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                "id_accuracy": reduced["id_acc"],
                "bio_accuracy": reduced["bio_acc"],
                **amp_stats,
                "optimizer_updates": STEPS_PER_EPOCH,
                "active_images": len(dataset.train),
                "active_pids": b6.NUM_CLASSES,
                "active_clothes": b6.NUM_CLOTHING_STATES,
                "mean_clothes_per_pid": float(np.mean([len(values) for values in state_all.values()])),
                "sampled_unique_images": len(sampled_images),
                "sampled_unique_states": len(sampled_states),
                "sampled_unique_pids": len(sampled_pids),
                "relation_active_batches": relation_batches,
                "relation_active_batch_rate": relation_batches / float(STEPS_PER_EPOCH * WORLD_SIZE),
                "relation_pairs": relation_pairs,
                "relation_unique_state_pairs": len(sampled_relation_keys),
                "relation_weight_mean": weight_mean,
                "relation_weight_std": math.sqrt(weight_var),
                "relation_weight_min": 0.0 if not relation_pairs else float(min_weight.item()),
                "relation_weight_max": 0.0 if not relation_pairs else float(max_weight.item()),
                **metrics,
            }
            if rank == 0:
                metrics_writer.writerow(row)
                metrics_file.flush()
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch)
                writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"]})
                amp_stream.flush()
                relation_writer.writerow({
                    "arm": arm,
                    "epoch": epoch,
                    "all_images_available": True,
                    "all_states_available": True,
                    "all_pids_available": True,
                    "sampled_unique_images": len(sampled_images),
                    "sampled_unique_states": len(sampled_states),
                    "sampled_unique_pids": len(sampled_pids),
                    "relation_active_batches": relation_batches,
                    "relation_active_batch_rate": row["relation_active_batch_rate"],
                    "relation_pairs": relation_pairs,
                    "relation_unique_state_pairs": len(sampled_relation_keys),
                    "relation_weight_mean": weight_mean,
                    "relation_weight_std": math.sqrt(weight_var),
                    "relation_weight_min": row["relation_weight_min"],
                    "relation_weight_max": row["relation_weight_max"],
                    "optimizer_updates": STEPS_PER_EPOCH,
                })
                relation_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"])
                    best_epoch = epoch
                    b6.save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1 across E1-E60"})
            if epoch in (40, 50, 60):
                if rank == 0:
                    geometry_rows.extend(relation_geometry(model.module, cfg, dataset, bank, device, transform_test, epoch))
                    write_csv(run_dir / "relation_geometry.csv", geometry_rows)
                dist.barrier()
            if epoch == TOTAL_EPOCHS:
                save_rank_state(run_dir / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    b6.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                    write_json(run_dir / "checkpoint_epoch060.json", {
                        "phase": PHASE,
                        "arm": arm,
                        "epoch": epoch,
                        "global_step": global_step,
                        "rank_states": [run_dir / f"rank{item}_state_epoch{epoch:03d}.pt" for item in range(WORLD_SIZE)],
                        "model": run_dir / "model_final.pth",
                        "relation_geometry": run_dir / "relation_geometry.csv",
                    })
                    rows_now = read_csv(run_dir / "metrics_per_epoch.csv")
                    write_json(run_dir / "RUN_COMPLETE.json", {
                        "phase": PHASE,
                        "arm": arm,
                        "mode": "RESUME_MODE",
                        "completed_utc": utc_now(),
                        "epochs_completed": epoch,
                        "optimizer_updates_per_rank": global_step,
                        "metrics": run_dir / "metrics_per_epoch.csv",
                        "best_epoch": best_epoch,
                        "best_cc_rank1": best_r1,
                        "best_model": run_dir / "model_best.pth",
                        "final_model": run_dir / "model_final.pth",
                        "rows": len(rows_now),
                        "phase6b_source_rank0": source_checkpoint,
                        "phase6b_source_rank0_sha256": sha256_file(source_checkpoint),
                    })
        if rank == 0:
            write_json(run_dir / "AMP.json", {
                "initial_scale_from_e30_checkpoint": float(payload["scaler"]["scale"]),
                "epochs_completed": TOTAL_EPOCHS,
                "metrics_source": run_dir / "amp_statistics.csv",
            })
    finally:
        for stream in (metrics_file, amp_stream, relation_stream, amp_event_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        b6.cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "audit-b3", "prepare", "train"), required=True)
    parser.add_argument("--arm", choices=tuple(ARM_DIRS), default=None)
    args = parser.parse_args()
    if args.mode == "audit":
        result = resume_audit()
        frozen_method_audit()
        common_config_audit()
        print(json.dumps({"resume_mode": result["resume_mode"], "all_ready": result["all_four_exact_resume_ready"], "audit": str(ARTIFACT_ROOT / "resume_audit.md")}, indent=2, sort_keys=True))
    elif args.mode == "audit-b3":
        result = b3_resume_audit()
        print(json.dumps({"ready": result["ready"], "audit": str(ARTIFACT_ROOT / "B3_resume_e30_audit.md")}, indent=2, sort_keys=True))
    elif args.mode == "prepare":
        audit = json.loads((ARTIFACT_ROOT / "resume_audit.json").read_text())
        if audit.get("resume_mode") != "RESUME_MODE":
            raise RuntimeError("Phase6C prepare requires exact RESUME_MODE")
        prepare_phase6c_dirs()
        print(json.dumps({"prepared": True, "mode": audit["resume_mode"], "root": str(OUT_ROOT)}, indent=2, sort_keys=True))
    else:
        if args.arm is None:
            raise SystemExit("--arm is required for train")
        train_arm(args.arm)


if __name__ == "__main__":
    main()
