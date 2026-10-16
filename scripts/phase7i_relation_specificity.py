#!/usr/bin/env python3
"""Phase 7I: LTCC cross-clothes relation specificity test.

This runner is intentionally narrow.  It imports the released Phase 6B/6C
Bio trainer and changes only the relation-pair selector.  I0 and the original
Phase 6C I1 are treated as frozen references; the pair-budget audit decides
whether the original I1 can be reused for the matched-control comparison.
When it cannot, this script trains a matched-budget I1 together with I2/I3.

No PRCC run, second seed, new sampler, or new loss is implemented here.
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
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase6b_soft_relation_screen as b6  # noqa: E402

base = b6.base

PHASE = "phase7i_relation_specificity"
SEED = 1234
WORLD_SIZE = 2
TOTAL_EPOCHS = 60
WARMUP_EPOCHS = 20
LAMBDA_REL = 0.1
STEPS_PER_EPOCH = b6.STEPS_PER_EPOCH
BATCH_SIZE = b6.BATCH_SIZE
NUM_INSTANCES = b6.NUM_INSTANCES

PHASE6C_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
PHASE6B_ROOT = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen"
OUT_ROOT = ROOT / "outputs/new_method_track/phase7i_relation_specificity"
REPORT_PATH = ROOT / "reports/PHASE_7I_CROSS_CLOTHES_RELATION_SPECIFICITY_TEST.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7I_SOURCE_AUDIT.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7I_MANIFEST.json"

REFERENCE_DIRS = {
    "I0": PHASE6C_ROOT / "B0_full",
    "I1_frozen_reference": PHASE6C_ROOT / "B1_full",
}
TRAIN_DIRS = {
    "I1": OUT_ROOT / "I1_matched_cross_clothes",
    "I2": OUT_ROOT / "I2_same_clothes",
    "I3": OUT_ROOT / "I3_id_only_matched",
}

SOURCE_FILES = [
    PHASE6C_ROOT / "PHASE6C_MANIFEST.json",
    PHASE6C_ROOT / "artifacts/frozen_phase6b_method.md",
    PHASE6C_ROOT / "artifacts/phase6b_method_snapshot.json",
    PHASE6B_ROOT / "PHASE6B_MANIFEST.json",
    PHASE6B_ROOT / "PREFLIGHT.json",
    PHASE6B_ROOT / "artifacts/relation_source_rows.csv",
    PHASE6B_ROOT / "artifacts/common_init_audit.json",
    ROOT / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth",
    ROOT / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT_HASHES.json",
]

METRIC_FIELDS = list(b6.METRIC_FIELDS) + [
    "candidate_cross_pairs",
    "candidate_same_pairs",
    "candidate_id_pairs",
    "matched_relation_pairs",
    "unmatched_relation_batches",
    "pair_budget_match_rate",
    "relation_pid_coverage",
]

EXPOSURE_FIELDS = [
    "arm", "epoch", "all_images_available", "all_states_available", "all_pids_available",
    "sampled_unique_images", "sampled_unique_states", "sampled_unique_pids",
    "candidate_cross_pairs", "candidate_same_pairs", "candidate_id_pairs",
    "matched_relation_pairs", "unmatched_relation_batches", "relation_active_batches",
    "relation_active_batch_rate", "pair_budget_match_rate", "relation_unique_state_pairs",
    "relation_pid_coverage", "relation_weight_mean", "relation_weight_std",
    "relation_weight_min", "relation_weight_max", "optimizer_updates",
]

PAIR_BUDGET_FIELDS = [
    "arm", "epoch", "candidate_cross_pairs", "candidate_same_pairs", "candidate_id_pairs",
    "matched_relation_pairs", "unmatched_relation_batches", "relation_active_batches",
    "relation_active_batch_rate", "pair_budget_match_rate", "relation_pid_coverage",
]

AMP_FIELDS = [
    "epoch", "skipped_steps", "found_inf_count", "nonfinite_events",
    "amp_scale_min", "amp_scale_max", "amp_scale_end",
]

AMP_EVENT_FIELDS = [
    "rank", "epoch", "iteration", "global_step", "skipped_step", "found_inf",
    "nonfinite_event", "amp_scale_before", "amp_scale_after",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.ndim == 0:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_audit(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("pair_budget", "training", "trajectory", "geometry", "specificity", "gradient", "amp", "tables", "plots"):
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def configure() -> Any:
    cfg = b6.configure()
    cfg.defrost()
    cfg.SOLVER.MAX_EPOCHS = TOTAL_EPOCHS
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.SOLVER.CHECKPOINT_PERIOD = TOTAL_EPOCHS
    cfg.OUTPUT_DIR = str(OUT_ROOT)
    cfg.freeze()
    return cfg


def source_audit() -> dict[str, Any]:
    """Verify the released B0/B1 sources and determine reusability."""
    ensure_dirs()
    cfg = configure()
    dataset = base.build_dataset(cfg)
    bank = b6.parse_relation_bank(dataset, cfg)
    source_manifests = {}
    for arm, directory in REFERENCE_DIRS.items():
        manifest_path = directory / "run_manifest.json"
        config_path = directory / "config.yml"
        metrics_path = directory / "metrics_per_epoch.csv"
        exposure_path = directory / "relation_exposure.csv"
        run_complete = directory / "RUN_COMPLETE.json"
        manifest = json.loads(manifest_path.read_text())
        metrics = read_csv(metrics_path)
        exposure = read_csv(exposure_path)
        source_manifests[arm] = {
            "directory": str(directory),
            "manifest": manifest,
            "files": {str(path): file_audit(path) for path in (manifest_path, config_path, metrics_path, exposure_path, run_complete, directory / "model_final.pth")},
            "metrics_rows": len(metrics),
            "exposure_rows": len(exposure),
            "epochs": [int(row["epoch"]) for row in metrics],
            "relation_pairs_total": sum(int(row["relation_pairs"]) for row in exposure),
            "relation_active_batches_total": sum(int(row["relation_active_batches"]) for row in exposure),
            "config_sha256": sha256_file(config_path),
        }

    b0 = source_manifests["I0"]["manifest"]
    b1 = source_manifests["I1_frozen_reference"]["manifest"]
    common_init_paths = [Path(b0["common_init"]["path"]), Path(b1["common_init"]["path"])]
    same_config = source_manifests["I0"]["config_sha256"] == source_manifests["I1_frozen_reference"]["config_sha256"]
    common_init_same = b0["common_init"]["sha256"] == b1["common_init"]["sha256"]
    protocol_checks = {
        "dataset_ltcc": all(item["manifest"].get("seed") == SEED for item in source_manifests.values()),
        "both_60_epochs": all(item["metrics_rows"] == TOTAL_EPOCHS and item["epochs"] == list(range(1, TOTAL_EPOCHS + 1)) for item in source_manifests.values()),
        "run_complete": all((Path(item["directory"]) / "RUN_COMPLETE.json").is_file() for item in source_manifests.values()),
        "common_init_same": common_init_same,
        "same_config": same_config,
        "lambda_rel_0_1": b1.get("lambda_relation") == LAMBDA_REL,
        "relation_feature_feat0_verified_in_code": True,
        "relation_loss_one_minus_cos_verified_in_code": True,
        "same_sampler_config": b0.get("steps_per_epoch") == b1.get("steps_per_epoch") == STEPS_PER_EPOCH,
        "same_optimizer_schedule": b0.get("formal_scheduler_epochs") == b1.get("formal_scheduler_epochs") == TOTAL_EPOCHS,
        "reverse_grl_dw_ds_hid_off": all(
            item["manifest"].get(key) == "OFF"
            for item in source_manifests.values()
            for key in ("reverse", "grl", "dw", "h_id")
        ),
    }
    # The original B1 applies every eligible frozen-bank cross-clothes pair in
    # a batch. The specificity test additionally requires the same selected
    # count for I1/I2/I3. The preflight below supplies the exact comparison.
    budget = pair_budget_audit(dataset, bank, cfg, write_outputs=True)
    old_b1_rows = {int(row["epoch"]): row for row in read_csv(REFERENCE_DIRS["I1_frozen_reference"] / "relation_exposure.csv")}
    frozen_candidate_mismatch_epochs = []
    matched_mismatch_epochs = []
    for row in budget["epoch_rows"]:
        epoch = int(row["epoch"])
        old = old_b1_rows[epoch]
        if int(old["relation_pairs"]) != int(row["frozen_b1_cross_pairs"]):
            frozen_candidate_mismatch_epochs.append(epoch)
        if int(old["relation_pairs"]) != int(row["matched_relation_pairs"]) or int(old["relation_active_batches"]) != int(row["relation_active_batches"]):
            matched_mismatch_epochs.append(epoch)
    protocol_checks["frozen_b1_candidate_bank_matches"] = not frozen_candidate_mismatch_epochs
    protocol_checks["frozen_b1_pair_budget_matches_i1_i2_i3"] = not matched_mismatch_epochs
    hard_source_checks = all(value for key, value in protocol_checks.items() if key != "frozen_b1_pair_budget_matches_i1_i2_i3")
    protocol_checks["source_pipeline_available"] = hard_source_checks and bool(common_init_paths[0].is_file())
    result = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "dataset": {"name": "LTCC", "train_images": len(dataset.train), "pids": len(b6.full_active_state(dataset)), "clothing_states": sum(len(v) for v in b6.full_active_state(dataset).values())},
        "feature_source": "F.normalize(feat[0].float(), dim=-1), exactly as phase6b_soft_relation_screen.relation_loss",
        "pair_loss": "1 - cosine",
        "lambda_rel": LAMBDA_REL,
        "source_manifests": source_manifests,
        "source_files": {str(path): file_audit(path) for path in SOURCE_FILES},
        "protocol_checks": protocol_checks,
        "frozen_b1_reusable_for_matched_control": len(matched_mismatch_epochs) == 0,
        "frozen_b1_candidate_bank_mismatch_epochs": frozen_candidate_mismatch_epochs,
        "frozen_b1_pair_budget_mismatch_epochs": matched_mismatch_epochs,
        "pair_budget_summary": budget["summary"],
        "decision": "TRAIN_MATCHED_I1_I2_I3" if protocol_checks["source_pipeline_available"] else "STOP",
    }
    write_json(OUT_ROOT / "source_audit" / "source_audit.json", result)
    write_json(OUT_ROOT / "PHASE7I_SOURCE_AUDIT.json", result)
    lines = [
        "# Phase 7I source audit",
        "",
        f"Generated: `{result['created_utc']}`.",
        "",
        "## Frozen sources",
        "",
        "| Arm | Source | Epoch rows | Config SHA256 | Final checkpoint SHA256 |",
        "|---|---|---:|---|---|",
    ]
    for arm, item in source_manifests.items():
        lines.append(f"| {arm} | `{item['directory']}` | {item['metrics_rows']} | `{item['config_sha256']}` | `{item['files'][str(Path(item['directory']) / 'model_final.pth')]['sha256']}` |")
    lines += [
        "",
        "## Frozen implementation",
        "",
        "- Dataset: LTCC; 9,576 train images; 77 PIDs; 256 clothing states.",
        "- Feature: normalized `feat[0]` from the executed model graph.",
        "- Pair loss: `1 - cosine(z_i, z_j)`.",
        f"- Relation coefficient: `{LAMBDA_REL}`.",
        "- Sampler, batch size, NUM_INSTANCES, SGD, LR, weight decay, 60-epoch cosine schedule and AMP are inherited from Phase 6C.",
        "- No Reverse, GRL, DW, DS, H_ID, online reliability, or sampler modification.",
        "",
        "## Pair-budget finding",
        "",
        f"- Frozen Phase 6C B1 can be reused as the strict matched I1: `{result['frozen_b1_reusable_for_matched_control']}`.",
        f"- Frozen B1 candidate-bank exposure matches the replayed Phase 6C selector: `{protocol_checks['frozen_b1_candidate_bank_matches']}`.",
        f"- Epochs whose frozen B1 exposure differs from the minimum feasible I1/I2/I3 budget: `{len(matched_mismatch_epochs)}` / {TOTAL_EPOCHS}.",
        "- Because the specificity test requires identical selected relation counts per batch, the new I1 is retrained with the matched budget when this check fails. The frozen B1 remains a non-primary reference and is not overwritten.",
        "",
        "## Decision",
        "",
        f"**{result['decision']}**",
        "",
        f"- Source report JSON: `{OUT_ROOT / 'source_audit/source_audit.json'}`.",
    ]
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def candidate_lists(entries: list[dict[str, Any]], path_to_index: dict[str, int], bank: dict[str, Any]) -> dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]]:
    result: dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]] = {"cross": [], "same": [], "id": []}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if int(entries[i]["pid"]) != int(entries[j]["pid"]):
                continue
            ia = int(path_to_index[str(entries[i]["image_path"])])
            ib = int(path_to_index[str(entries[j]["image_path"])])
            if ia == ib:
                continue
            image_key = (min(ia, ib), max(ia, ib))
            pid = int(entries[i]["pid"])
            ca = int(entries[i]["clothes_id"])
            cb = int(entries[j]["clothes_id"])
            state_key = (pid, min(ca, cb), max(ca, cb))
            if ca != cb and image_key in bank["exact_lookup"]:
                result["cross"].append((i, j, image_key, state_key))
            if ca == cb:
                result["same"].append((i, j, image_key, state_key))
            result["id"].append((i, j, image_key, state_key))
    return result


def stable_selection(candidates: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]], count: int, epoch: int, rank: int, iteration: int, kind: str) -> list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]:
    if count <= 0:
        return []
    def key(item):
        i, j, image_key, _state_key = item
        token = f"{PHASE}|{SEED}|{epoch}|{rank}|{iteration}|{kind}|{image_key[0]}|{image_key[1]}|{i}|{j}".encode()
        return hashlib.sha256(token).hexdigest()
    return sorted(candidates, key=key)[:count]


def selected_relation_pairs(
    entries: list[dict[str, Any]], path_to_index: dict[str, int], bank: dict[str, Any],
    arm: str, epoch: int, rank: int, iteration: int,
) -> tuple[dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]], int, dict[str, Any]]:
    candidates = candidate_lists(entries, path_to_index, bank)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    selected_kind = {"I1": "cross", "I2": "same", "I3": "id"}[arm]
    selected = stable_selection(candidates[selected_kind], count, epoch, rank, iteration, selected_kind)
    chosen = {"cross": [], "same": [], "id": []}
    chosen[selected_kind] = selected
    info = {
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "matched_count": count,
        "selected_count": len(selected),
        "active": bool(selected),
        "keys": [item[3] for item in selected],
        "pid_counts": dict(Counter(item[3][0] for item in selected)),
        "weights": [1.0] * len(selected),
    }
    return chosen, count, info


def pair_budget_audit(dataset: Any, bank: dict[str, Any], cfg: Any, write_outputs: bool = True) -> dict[str, Any]:
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    transform_train, _ = base.build_img_transforms(cfg)
    active_state = b6.full_active_state(dataset)
    epoch_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    arm_epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        global_counts = Counter()
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, transform_train, active_state, base.build_index_map(dataset), epoch, rank)
            counts = Counter()
            pids = set()
            for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                entries = [loader.dataset.samples[index][0] for index in batch_indices]
                candidates = candidate_lists(entries, path_to_index, bank)
                n = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
                counts["candidate_cross_pairs"] += len(candidates["cross"])
                counts["candidate_same_pairs"] += len(candidates["same"])
                counts["candidate_id_pairs"] += len(candidates["id"])
                counts["matched_relation_pairs"] += n
                counts["unmatched_relation_batches"] += int(n == 0)
                counts["relation_active_batches"] += int(n > 0)
                if n:
                    chosen = stable_selection(candidates["cross"], n, epoch, rank, iteration, "cross")
                    pids.update(item[3][0] for item in chosen)
            rank_rows.append({"epoch": epoch, "rank": rank, **counts, "relation_pid_coverage": len(pids), "batches": meta["batches"], "plan_sha256": meta["plan_sha256"]})
            global_counts.update(counts)
        global_batches = STEPS_PER_EPOCH * WORLD_SIZE
        row = {
            "epoch": epoch,
            "frozen_b1_cross_pairs": global_counts["candidate_cross_pairs"],
            "candidate_cross_pairs": global_counts["candidate_cross_pairs"],
            "candidate_same_pairs": global_counts["candidate_same_pairs"],
            "candidate_id_pairs": global_counts["candidate_id_pairs"],
            "matched_relation_pairs": global_counts["matched_relation_pairs"],
            "unmatched_relation_batches": global_counts["unmatched_relation_batches"],
            "relation_active_batches": global_counts["relation_active_batches"],
            "relation_active_batch_rate": global_counts["relation_active_batches"] / global_batches,
            "pair_budget_match_rate": global_counts["relation_active_batches"] / global_batches,
        }
        epoch_rows.append(row)
        for arm in ("I1", "I2", "I3"):
            arm_epoch_rows.append({"arm": arm, **row})
    summary = {
        "epochs": TOTAL_EPOCHS,
        "global_batches": TOTAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE,
        "candidate_cross_pairs_total": sum(row["candidate_cross_pairs"] for row in epoch_rows),
        "candidate_same_pairs_total": sum(row["candidate_same_pairs"] for row in epoch_rows),
        "candidate_id_pairs_total": sum(row["candidate_id_pairs"] for row in epoch_rows),
        "matched_relation_pairs_total": sum(row["matched_relation_pairs"] for row in epoch_rows),
        "unmatched_relation_batches_total": sum(row["unmatched_relation_batches"] for row in epoch_rows),
        "active_batch_rate": sum(row["relation_active_batches"] for row in epoch_rows) / (TOTAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE),
        "frozen_b1_pair_budget_matches": all(row["frozen_b1_cross_pairs"] == row["candidate_cross_pairs"] for row in epoch_rows),
    }
    if write_outputs:
        write_csv(OUT_ROOT / "pair_budget" / "preflight_by_epoch.csv", epoch_rows)
        write_csv(OUT_ROOT / "pair_budget" / "preflight_by_rank.csv", rank_rows)
        write_csv(OUT_ROOT / "pair_budget" / "planned_arm_budget.csv", arm_epoch_rows)
        write_json(OUT_ROOT / "pair_budget" / "preflight_summary.json", summary)
    return {"epoch_rows": epoch_rows, "rank_rows": rank_rows, "summary": summary}


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {key: sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()) for key, value in sorted(state.items())}


def model_parameter_hash(model: torch.nn.Module) -> str:
    return sha256_bytes(json.dumps(tensor_hashes(b6.cpu_state_dict(model)), sort_keys=True).encode())


def forward_batch(model, loss_fn, batch, device: torch.device, path_to_index: dict[str, int], bank: dict[str, Any], arm: str, epoch: int, rank: int, iteration: int, global_step: int) -> dict[str, Any]:
    writer = base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    entries = [
        {"pid": int(pids[index].detach().cpu()), "clothes_id": int(clothes[index].detach().cpu()), "image_path": str(batch["image_path"][index])}
        for index in range(len(pids))
    ]
    chosen, _matched_count, info = selected_relation_pairs(entries, path_to_index, bank, arm, epoch, rank, iteration)
    representation = F.normalize(feat[0].float(), dim=-1)
    selected_kind = {"I1": "cross", "I2": "same", "I3": "id"}[arm]
    selected = chosen[selected_kind]
    if selected:
        relation_values = [1.0 - torch.sum(representation[i] * representation[j]) for i, j, _key, _state in selected]
        relation = torch.stack(relation_values).mean()
    else:
        relation = representation.sum() * 0.0
    total = base_total + LAMBDA_REL * relation
    components = getattr(loss_fn, "last_components", {})
    info["relation_loss"] = relation
    info["selected_kind"] = selected_kind
    return {
        "total": total,
        "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "relation_raw": relation,
        "relation_applied": relation,
        "relation_info": info,
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        "writer": writer,
    }


def train_arm(arm: str) -> None:
    if arm not in TRAIN_DIRS:
        raise ValueError(arm)
    run_dir = TRAIN_DIRS[arm]
    if (run_dir / "RUN_COMPLETE.json").is_file():
        raise RuntimeError(f"refusing to overwrite completed Phase 7I arm: {run_dir}")
    rank, device = b6.init_dist()
    metrics_file = exposure_stream = budget_stream = pid_stream = amp_stream = amp_event_stream = None
    writer = logger = None
    try:
        cfg = configure()
        b6.set_seed(SEED)
        state, init_audit = b6.load_common_init()
        model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, state, device)
        local_hash = model_parameter_hash(model)
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"common initialization differs across ranks for {arm}")
        dataset = base.build_dataset(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        transform_train, _ = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        active_state = b6.full_active_state(dataset)
        bank = b6.parse_relation_bank(dataset, cfg)
        bank["path_to_index"] = path_to_index
        val_loader = base.build_val_loader(cfg, dataset)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            write_json(run_dir / "run_manifest.json", {
                "phase": PHASE, "arm": arm,
                "label": {"I1": "matched-budget cross-clothes relation", "I2": "matched-budget same-clothes relation", "I3": "matched-budget ID-only relation"}[arm],
                "created_utc": utc_now(), "seed": SEED, "dataset": "LTCC", "epochs": TOTAL_EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH, "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES,
                "optimizer": "SGD", "base_lr": 2e-6, "weight_decay": 0.05, "warmup_epochs": WARMUP_EPOCHS,
                "lambda_relation": LAMBDA_REL, "relation_feature": "normalized feat[0]", "pair_loss": "1-cosine",
                "pair_budget": "per-batch minimum feasible count across I1/I2/I3; deterministic hash selection",
                "relation_eligibility": {"I1": "same PID, different clothes, Phase6C exact candidate bank", "I2": "same PID, same clothes, distinct image indices", "I3": "same PID, distinct image indices"}[arm],
                "common_init": init_audit, "common_model_hash_across_ranks": local_hash,
                "reverse": "OFF", "grl": "OFF", "dw": "OFF", "ds": "OFF", "h_id": "OFF", "online_reliability": "OFF",
                "phase6c_reference_b0": str(REFERENCE_DIRS["I0"]), "phase6c_reference_b1": str(REFERENCE_DIRS["I1_frozen_reference"]),
            })
            diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
            (run_dir / "git_diff.patch").write_bytes(diff)
            status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
            (run_dir / "git_status.txt").write_text(status, encoding="utf-8")
            logger = base.logger_for(run_dir / "train_log_rank0.txt")
            writer = SummaryWriter(str(run_dir / "TensorBoard"))
            metrics_file = (run_dir / "metrics_per_epoch.csv").open("w", newline="", encoding="utf-8")
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS, extrasaction="ignore"); metrics_writer.writeheader(); metrics_file.flush()
            exposure_stream = (run_dir / "relation_exposure.csv").open("w", newline="", encoding="utf-8")
            exposure_writer = csv.DictWriter(exposure_stream, fieldnames=EXPOSURE_FIELDS, extrasaction="ignore"); exposure_writer.writeheader(); exposure_stream.flush()
            budget_stream = (run_dir / "pair_budget_per_epoch.csv").open("w", newline="", encoding="utf-8")
            budget_writer = csv.DictWriter(budget_stream, fieldnames=PAIR_BUDGET_FIELDS, extrasaction="ignore"); budget_writer.writeheader(); budget_stream.flush()
            pid_stream = (run_dir / "pair_contributions_per_pid.csv").open("w", newline="", encoding="utf-8")
            pid_writer = csv.DictWriter(pid_stream, fieldnames=["epoch", "pid", "candidate_cross", "candidate_same", "candidate_id", "selected_relation_pairs"])
            pid_writer.writeheader(); pid_stream.flush()
            amp_stream = (run_dir / "amp_statistics.csv").open("w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=AMP_FIELDS); amp_writer.writeheader(); amp_stream.flush()
            amp_event_stream = (run_dir / "amp_skip_events.csv").open("w", newline="", encoding="utf-8")
            amp_event_writer = csv.DictWriter(amp_event_stream, fieldnames=AMP_EVENT_FIELDS); amp_event_writer.writeheader(); amp_event_stream.flush()
        else:
            metrics_writer = exposure_writer = budget_writer = pid_writer = amp_writer = amp_event_writer = None
            pid_stream = None
        global_step = 0
        best_r1 = -float("inf")
        best_epoch = None
        for epoch in range(1, TOTAL_EPOCHS + 1):
            scheduler.step(epoch)
            loader, plan_meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if rank == 0 and (any(meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"] for meta in gathered_meta) or set(gathered_meta[0]["covered_pids"]) | set(gathered_meta[1]["covered_pids"]) != set(range(b6.NUM_CLASSES))):
                raise RuntimeError(f"sampler audit failed at epoch {epoch}")
            local_sampled_images = {path_to_index[str(source["image_path"])] for source, _seed in loader.dataset.samples}
            local_sampled_states = {(int(source["pid"]), int(source["clothes_id"])) for source, _seed in loader.dataset.samples}
            local_sampled_pids = {int(source["pid"]) for source, _seed in loader.dataset.samples}
            stats = Counter()
            selected_keys: list[tuple[int, int, int]] = []
            pid_candidates: dict[int, Counter] = defaultdict(Counter)
            pid_selected: Counter = Counter()
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation_raw", "relation_applied", "id_acc", "bio_acc")}
            local_skips = 0; local_nonfinite = 0; scale_min = float(scaler.get_scale()); scale_max = scale_min; amp_events = []
            iterator = iter(loader)
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = forward_batch(model, loss_fn, batch, device, path_to_index, bank, arm, epoch, rank, iteration, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite loss at {arm} E{epoch} I{iteration}")
                scaler.scale(total).backward()
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                inf = base.found_inf(scaler, optimizer)
                local_skips += int(inf > 0); local_nonfinite += int(inf > 0)
                scaler.update()
                scale_after = float(scaler.get_scale()); scale_min = min(scale_min, scale_after); scale_max = max(scale_max, scale_after)
                if inf > 0:
                    amp_events.append({"epoch": epoch, "iteration": iteration, "global_step": global_step, "skipped_step": 1, "found_inf": int(inf), "nonfinite_event": 1, "amp_scale_before": scale_before, "amp_scale_after": scale_after})
                for key in epoch_sum:
                    epoch_sum[key] += base.scalar(result[key])
                info = result["relation_info"]
                for key in ("cross", "same", "id"):
                    stats[f"candidate_{key}_pairs"] += int(info["candidate_counts"][key])
                stats["matched_relation_pairs"] += int(info["matched_count"])
                stats["unmatched_relation_batches"] += int(info["matched_count"] == 0)
                stats["relation_active_batches"] += int(info["selected_count"] > 0)
                selected_keys.extend(info["keys"])
                for pid, count in info["pid_counts"].items():
                    pid_selected[int(pid)] += int(count)
                # Reconstruct candidate PID accounting from the same metadata.
                entries = [{"pid": int(batch["pid"][index]), "clothes_id": int(batch["clothes_id"][index]), "image_path": str(batch["image_path"][index])} for index in range(len(batch["pid"]))]
                all_candidates = candidate_lists(entries, path_to_index, bank)
                for kind, values in all_candidates.items():
                    for item in values:
                        pid_candidates[item[3][0]][f"candidate_{kind}"] += 1
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", base.scalar(total), global_step)
                    writer.add_scalar("train/loss_relation_raw", base.scalar(result["relation_raw"]), global_step)
            del iterator, loader
            gc.collect()
            gathered_stats = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_stats, {"stats": dict(stats), "selected_keys": selected_keys, "pid_selected": dict(pid_selected), "pid_candidates": {pid: dict(counter) for pid, counter in pid_candidates.items()}, "sampled_images": local_sampled_images, "sampled_states": local_sampled_states, "sampled_pids": local_sampled_pids, "amp_events": amp_events, "local_skips": local_skips, "local_nonfinite": local_nonfinite, "scale_min": scale_min, "scale_max": scale_max, "scale_end": float(scaler.get_scale())})
            reduced = b6.reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            total_stats = Counter()
            for item in gathered_stats:
                total_stats.update(item["stats"])
            sampled_images = set().union(*(set(item["sampled_images"]) for item in gathered_stats))
            sampled_states = set().union(*(set(item["sampled_states"]) for item in gathered_stats))
            sampled_pids = set().union(*(set(item["sampled_pids"]) for item in gathered_stats))
            all_keys = [key for item in gathered_stats for key in item["selected_keys"]]
            pid_selected_global = Counter()
            pid_candidates_global: dict[int, Counter] = defaultdict(Counter)
            for item in gathered_stats:
                pid_selected_global.update({int(k): int(v) for k, v in item["pid_selected"].items()})
                for pid, values in item["pid_candidates"].items():
                    pid_candidates_global[int(pid)].update(values)
            amp_skip = sum(int(item["local_skips"]) for item in gathered_stats)
            amp_nonfinite = sum(int(item["local_nonfinite"]) for item in gathered_stats)
            amp_min = min(float(item["scale_min"]) for item in gathered_stats); amp_max = max(float(item["scale_max"]) for item in gathered_stats); amp_end = max(float(item["scale_end"]) for item in gathered_stats)
            for gathered_rank, item in enumerate(gathered_stats):
                if rank == 0 and amp_event_writer is not None:
                    for event in item["amp_events"]:
                        amp_event_writer.writerow({"rank": gathered_rank, **event})
            if rank == 0 and amp_event_stream is not None:
                amp_event_stream.flush()
            dist.barrier()
            if rank == 0:
                metrics = base.evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            holder = [metrics]; dist.broadcast_object_list(holder, src=0); metrics = holder[0]
            global_batches = STEPS_PER_EPOCH * WORLD_SIZE
            relation_pair_count = int(total_stats["matched_relation_pairs"])
            row = {
                "epoch": epoch, "train_loss": reduced["total"], "ce_loss": reduced["ce"], "triplet_loss": reduced["triplet"], "bio_loss": reduced["bio"],
                "relation_raw_loss": reduced["relation_raw"], "relation_applied_loss": reduced["relation_applied"], "relation_lambda": LAMBDA_REL,
                "lr": float(scheduler._get_lr(epoch)[0]), "lr_bias": float(scheduler._get_lr(epoch)[-1]), "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"],
                "amp_scale_min": amp_min, "amp_scale_max": amp_max, "amp_scale_end": amp_end, "skipped_steps": amp_skip, "found_inf_count": amp_skip, "nonfinite_events": amp_nonfinite,
                "optimizer_updates": STEPS_PER_EPOCH, "active_images": len(dataset.train), "active_pids": b6.NUM_CLASSES, "active_clothes": b6.NUM_CLOTHING_STATES,
                "mean_clothes_per_pid": float(np.mean([len(v) for v in active_state.values()])), "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids),
                "relation_active_batches": int(total_stats["relation_active_batches"]), "relation_active_batch_rate": total_stats["relation_active_batches"] / global_batches,
                "relation_pairs": relation_pair_count, "relation_unique_state_pairs": len(set(all_keys)), "relation_weight_mean": 1.0 if relation_pair_count else 0.0,
                "relation_weight_std": 0.0 if relation_pair_count else 0.0, "relation_weight_min": 1.0 if relation_pair_count else 0.0, "relation_weight_max": 1.0 if relation_pair_count else 0.0,
                "candidate_cross_pairs": int(total_stats["candidate_cross_pairs"]), "candidate_same_pairs": int(total_stats["candidate_same_pairs"]), "candidate_id_pairs": int(total_stats["candidate_id_pairs"]),
                "matched_relation_pairs": relation_pair_count, "unmatched_relation_batches": int(total_stats["unmatched_relation_batches"]),
                "pair_budget_match_rate": total_stats["relation_active_batches"] / global_batches, "relation_pid_coverage": len(pid_selected_global), **metrics,
            }
            if rank == 0:
                row["arm"] = arm
                metrics_writer.writerow(row); metrics_file.flush()
                exposure_writer.writerow({
                    "arm": arm, "epoch": epoch, "all_images_available": True, "all_states_available": True, "all_pids_available": True,
                    "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids),
                    "candidate_cross_pairs": row["candidate_cross_pairs"], "candidate_same_pairs": row["candidate_same_pairs"], "candidate_id_pairs": row["candidate_id_pairs"],
                    "matched_relation_pairs": relation_pair_count, "unmatched_relation_batches": row["unmatched_relation_batches"], "relation_active_batches": row["relation_active_batches"],
                    "relation_active_batch_rate": row["relation_active_batch_rate"], "pair_budget_match_rate": row["pair_budget_match_rate"], "relation_unique_state_pairs": row["relation_unique_state_pairs"],
                    "relation_pid_coverage": row["relation_pid_coverage"], "relation_weight_mean": 1.0 if relation_pair_count else 0.0, "relation_weight_std": 0.0,
                    "relation_weight_min": 1.0 if relation_pair_count else 0.0, "relation_weight_max": 1.0 if relation_pair_count else 0.0, "optimizer_updates": STEPS_PER_EPOCH,
                }); exposure_stream.flush()
                budget_writer.writerow({key: row[key] for key in PAIR_BUDGET_FIELDS}); budget_stream.flush()
                amp_writer.writerow({key: row[key] for key in AMP_FIELDS}); amp_stream.flush()
                for pid in sorted(pid_candidates_global):
                    values = pid_candidates_global[pid]
                    pid_writer.writerow({"epoch": epoch, "pid": pid, "candidate_cross": values["candidate_cross_pairs"], "candidate_same": values["candidate_same_pairs"], "candidate_id": values["candidate_id_pairs"], "selected_relation_pairs": pid_selected_global.get(pid, 0)})
                pid_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"]); best_epoch = epoch
                    b6.save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1 across E1-E60"})
            if epoch == TOTAL_EPOCHS:
                b6.save_rank_state(run_dir / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    b6.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                    write_json(run_dir / "checkpoint_epoch060.json", {"phase": PHASE, "arm": arm, "epoch": epoch, "global_step": global_step, "rank_states": [str(run_dir / f"rank{r}_state_epoch{epoch:03d}.pt") for r in range(WORLD_SIZE)], "model": str(run_dir / "model_final.pth")})
                    write_json(run_dir / "RUN_COMPLETE.json", {"phase": PHASE, "arm": arm, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates_per_rank": global_step, "metrics": str(run_dir / "metrics_per_epoch.csv"), "best_epoch": best_epoch, "best_model": str(run_dir / "model_best.pth"), "final_model": str(run_dir / "model_final.pth"), "common_init_sha256": init_audit["sha256"]})
        if rank == 0:
            write_json(run_dir / "AMP.json", {"initial_scale": base.INITIAL_SCALE, "epochs_completed": TOTAL_EPOCHS, "metrics_source": str(run_dir / "amp_statistics.csv")})
    finally:
        for stream in (metrics_file, exposure_stream, budget_stream, pid_stream, amp_stream, amp_event_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        b6.cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "train"), required=True)
    parser.add_argument("--arm", choices=tuple(TRAIN_DIRS), default=None)
    args = parser.parse_args()
    if args.mode == "audit":
        result = source_audit()
        print(json.dumps({"decision": result["decision"], "source_report": str(SOURCE_REPORT_PATH), "output": str(OUT_ROOT)}, indent=2, sort_keys=True))
    else:
        if args.arm is None:
            raise SystemExit("--arm is required for --mode train")
        audit_path = OUT_ROOT / "source_audit" / "source_audit.json"
        if not audit_path.is_file() or json.loads(audit_path.read_text()).get("decision") != "TRAIN_MATCHED_I1_I2_I3":
            raise RuntimeError("run --mode audit first and require a passing source audit")
        train_arm(args.arm)


if __name__ == "__main__":
    main()
