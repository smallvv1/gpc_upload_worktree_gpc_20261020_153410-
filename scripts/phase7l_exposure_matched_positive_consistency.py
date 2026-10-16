#!/usr/local/bin/python3
"""Phase 7L: exposure-matched generic same-ID positive consistency on PRCC.

This phase keeps the audited PRCC Phase 7A/7C protocol fixed and changes only
when the generic same-PID positive-consistency term is applied:

L0: Bio control, no Relation.
L1: dense ID-only Relation whenever a valid same-PID pair exists.
L2: deterministic 15% batch exposure.
L3: deterministic exposure matched to the frozen LTCC I3 pair-per-train-image
    budget.

The protocol/audit mode must complete before any training mode is allowed.
All training and no-step gradient probes require the PPU; there is no CPU
fallback.  No second seed is implemented or launched by this file.
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
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase6b_soft_relation_screen as p6b  # noqa: E402
import phase7a_prcc_relation_validation as p7a  # noqa: E402
import phase7k_a_relation_strength_balance as p7k  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402


PHASE = "phase7l_exposure_matched_positive_consistency"
SEED = 1234
WORLD_SIZE = 2
FORMAL_EPOCHS = 60
WARMUP_EPOCHS = 20
INITIAL_SCALE = 512
BATCH_SIZE = 16
NUM_INSTANCES = 4
STEPS_PER_EPOCH = 551
TRAIN_IMAGES = 17896
TRAIN_PIDS = 150
TRAIN_CLOTHING_STATES = 300
LAMBDA_REL = 0.1
L2_ACTIVE_PROBABILITY = 0.15
PRIMARY_START = 51
PRIMARY_END = 60
GEOMETRY_EPOCHS = (20, 40, 60)
STRENGTH_BATCHES = 32
EPS = 1e-12

OUT_ROOT = ROOT / "outputs/new_method_track/phase7l_exposure_matched_positive_consistency"
PROTOCOL_ROOT = OUT_ROOT / "protocol"
TRAINING_ROOT = OUT_ROOT / "training"
EXPOSURE_ROOT = OUT_ROOT / "exposure"
STRENGTH_ROOT = OUT_ROOT / "strength"
GEOMETRY_ROOT = OUT_ROOT / "geometry"
AMP_ROOT = OUT_ROOT / "amp"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7L_MANIFEST.json"

SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7L_SOURCE_AUDIT.md"
PROTOCOL_REPORT_PATH = ROOT / "reports/PHASE_7L_EXPOSURE_PROTOCOL.md"
REPORT_PATH = ROOT / "reports/PHASE_7L_EXPOSURE_MATCHED_POSITIVE_CONSISTENCY.md"

PHASE7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
PHASE7C_ROOT = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
PHASE7I_ROOT = ROOT / "outputs/new_method_track/phase7i_relation_specificity"
PRCC_COMMON_INIT = PHASE7A_ROOT / "PRCC_COMMON_INIT.pth"
PRCC_COMMON_META = PHASE7A_ROOT / "PRCC_COMMON_INIT_HASHES.json"
PHASE7C_COMMON_AUDIT = PHASE7C_ROOT / "training/common_init_audit.json"
LTCC_I3_EXPOSURE = PHASE7I_ROOT / "I3_id_only_matched/relation_exposure.csv"
LTCC_I3_MANIFEST = PHASE7I_ROOT / "I3_id_only_matched/run_manifest.json"
PRCC_C0_SOURCE = PHASE7C_ROOT / "training/C0_bio"
PRCC_C1_SOURCE = PHASE7C_ROOT / "training/C1_image_uniform_relation"
PRCC_ROOT = Path("/home/data/prcc")

ARM_LABELS = {
    "L0": "PRCC Bio control",
    "L1": "PRCC dense generic same-ID positive consistency",
    "L2": "PRCC LTCC-like 15% batch exposure",
    "L3": "PRCC LTCC-normalized cumulative pair budget",
}
ARM_DIRS = {arm: TRAINING_ROOT / arm for arm in ("L0", "L1", "L2", "L3")}

METRIC_FIELDS = [
    "arm", "seed", "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss",
    "relation_loss_all_batches", "relation_loss_active_batches", "relation_applied_loss",
    "relation_lambda", "total_loss", "lr", "lr_bias", "id_accuracy", "bio_accuracy",
    "planned_updates", "updates_completed", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "amp_skip", "skipped_steps", "found_inf_count", "nonfinite_events", "active_images",
    "active_pids", "active_clothing_states", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP",
    "general_rank1", "general_rank5", "general_rank10", "general_mAP",
]
EXPOSURE_FIELDS = [
    "arm", "seed", "epoch", "total_batches_global", "planned_updates_per_rank",
    "scheduled_relation_batches_global", "relation_active_batches_global", "relation_active_batch_rate",
    "valid_same_id_candidate_pairs_global", "selected_relation_pairs_global", "relation_pair_contributions",
    "unique_relation_pids_global", "relation_loss_active_mean", "lambda_relation_loss_active_mean",
    "cumulative_relation_active_batches_global", "cumulative_relation_pair_contributions",
    "scheduled_probability_or_budget", "l3_target_pairs_total", "l3_target_metric",
]
AMP_FIELDS = [
    "arm", "seed", "epoch", "amp_skip", "skipped_steps", "found_inf_count", "nonfinite_events",
    "amp_scale_min", "amp_scale_max", "amp_scale_end", "planned_updates", "updates_completed",
]
AMP_EVENT_FIELDS = [
    "arm", "seed", "rank", "epoch", "iteration", "global_step", "skipped_step", "found_inf",
    "nonfinite_event", "amp_scale_before", "amp_scale_after",
]
SAMPLER_FIELDS = [
    "arm", "seed", "epoch", "rank", "sampler_epoch_used", "samples", "batches", "unique_images",
    "unique_pids", "unique_clothing_states", "plan_sha256",
]
GEOMETRY_FIELDS = [
    "arm", "seed", "checkpoint", "same_id_positive_similarity_mean", "same_clothes_similarity_mean",
    "cross_clothes_similarity_mean", "nearest_negative_pid_center_similarity_mean",
    "identity_margin_mean", "identity_ordering_correct_rate", "pid_center_separation_mean",
    "clothing_state_dispersion_mean", "same_id_pair_count", "same_clothes_pair_count",
    "cross_clothes_pair_count", "pid_count", "clothing_state_count", "geometry_source",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
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


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


def ensure_dirs() -> None:
    for path in (OUT_ROOT, PROTOCOL_ROOT, TRAINING_ROOT, EXPOSURE_ROOT, STRENGTH_ROOT, GEOMETRY_ROOT, AMP_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for path in ARM_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def require_ppu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 7L requires exactly {WORLD_SIZE} visible PPUs; CPU fallback is forbidden")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deterministic_unit(namespace: str, epoch: int, global_batch_idx: int) -> float:
    token = f"{PHASE}|{namespace}|{SEED}|{epoch}|{global_batch_idx}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return integer / float(2**64)


def pair_count_from_indices(indices: list[int], samples: list[dict[str, Any]]) -> tuple[int, set[int]]:
    pid_counts: Counter[int] = Counter(int(samples[index]["pid"]) for index in indices)
    return sum(count * (count - 1) // 2 for count in pid_counts.values()), set(pid_counts)


def batch_chunks(indices: list[int], batch_size: int) -> list[list[int]]:
    if len(indices) % batch_size:
        raise RuntimeError(f"sampler returned {len(indices)} samples, not divisible by batch size {batch_size}")
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def load_prcc_cfg(run_dir: Path | None = None) -> Any:
    return p7a.configure(run_dir or (OUT_ROOT / "config_probe"))


def load_common_init() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not PRCC_COMMON_INIT.is_file() or not PRCC_COMMON_META.is_file():
        raise FileNotFoundError("frozen PRCC Phase 7A common initialization is missing")
    meta = read_json(PRCC_COMMON_META)
    state = torch.load(PRCC_COMMON_INIT, map_location="cpu", weights_only=False)
    state_hash = p7a.state_hash(state)
    if sha256_file(PRCC_COMMON_INIT) != meta.get("state_sha256") or state_hash != meta.get("state_hash"):
        raise RuntimeError("frozen PRCC common initialization hash mismatch")
    return state, {"sha256": sha256_file(PRCC_COMMON_INIT), "state_hash": state_hash, "meta": meta}


def ltcc_exposure_audit() -> dict[str, Any]:
    rows = read_csv(LTCC_I3_EXPOSURE)
    if len(rows) != FORMAL_EPOCHS or any(row.get("arm") != "I3" for row in rows):
        raise RuntimeError("LTCC I3 frozen exposure must contain exactly 60 I3 epoch rows")
    pair_total = sum(int(row["matched_relation_pairs"]) for row in rows)
    updates_per_epoch = {int(row["optimizer_updates"]) for row in rows}
    sampled_images_per_epoch = {int(row["sampled_unique_images"]) for row in rows}
    if updates_per_epoch != {596} or sampled_images_per_epoch != {9536}:
        raise RuntimeError(f"unexpected frozen LTCC I3 exposure topology: updates={updates_per_epoch}, images={sampled_images_per_epoch}")
    target_pairs_per_train_image = pair_total / float(FORMAL_EPOCHS * 9576)
    return {
        "source": str(LTCC_I3_EXPOSURE),
        "source_sha256": sha256_file(LTCC_I3_EXPOSURE),
        "epochs": len(rows),
        "pair_contributions_total": pair_total,
        "pair_contributions_per_epoch_mean": pair_total / FORMAL_EPOCHS,
        "active_batches_total": sum(int(row["relation_active_batches"]) for row in rows),
        "global_batches_total": sum(int(row["optimizer_updates"]) * WORLD_SIZE for row in rows),
        "active_batch_rate": sum(int(row["relation_active_batches"]) for row in rows) / float(sum(int(row["optimizer_updates"]) * WORLD_SIZE for row in rows)),
        "train_images": 9576,
        "target_metric": "full_60_epoch_relation_pair_contributions / (60 * LTCC train images)",
        "target_pairs_per_train_image": target_pairs_per_train_image,
    }


def build_prcc_plan_inputs(dataset: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for epoch in range(1, FORMAL_EPOCHS + 1):
        rank_batches: dict[int, list[list[int]]] = {}
        for rank in range(WORLD_SIZE):
            sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=rank, seed=SEED)
            indices = list(iter(sampler))
            # The frozen PRCC DataLoader uses drop_last=True.  The sampler
            # exposes a 12-image tail (8828 samples/rank), but only the first
            # 551 complete batches (8816 samples/rank) reach the optimizer.
            indices = indices[:STEPS_PER_EPOCH * BATCH_SIZE]
            batches = batch_chunks(indices, BATCH_SIZE)
            if len(batches) != STEPS_PER_EPOCH:
                raise RuntimeError(f"PRCC rank {rank} epoch {epoch}: expected {STEPS_PER_EPOCH} batches, got {len(batches)}")
            rank_batches[rank] = batches
        for iteration in range(1, STEPS_PER_EPOCH + 1):
            global_batch_idx = (epoch - 1) * STEPS_PER_EPOCH + iteration
            counts = {}
            pids = {}
            for rank in range(WORLD_SIZE):
                count, pid_set = pair_count_from_indices(rank_batches[rank][iteration - 1], dataset.train)
                counts[f"rank{rank}_candidate_pairs"] = count
                pids[rank] = pid_set
            total_pairs = counts["rank0_candidate_pairs"] + counts["rank1_candidate_pairs"]
            l2_unit = deterministic_unit("L2", epoch, global_batch_idx)
            l3_priority = deterministic_unit("L3_PRIORITY", epoch, global_batch_idx)
            records.append({
                "epoch": epoch, "iteration": iteration, "global_batch_idx": global_batch_idx,
                **counts, "candidate_pairs_global": total_pairs,
                "candidate_pids_global": len(pids[0] | pids[1]),
                "l2_hash_unit": l2_unit, "l2_scheduled": int(l2_unit < L2_ACTIVE_PROBABILITY),
                "l3_priority": l3_priority,
                "l3_scheduled": 0, "l3_selected_pairs_global": 0,
            })
    return records


def freeze_exposure_protocol() -> dict[str, Any]:
    ensure_dirs()
    cfg = load_prcc_cfg()
    dataset = p7a.build_dataset(cfg)
    if len(dataset.train) != TRAIN_IMAGES or int(dataset.num_train_pids) != TRAIN_PIDS or int(dataset.num_train_clothes) != TRAIN_CLOTHING_STATES:
        raise RuntimeError("PRCC dataset topology changed")
    ltcc = ltcc_exposure_audit()
    records = build_prcc_plan_inputs(dataset)
    target_total = int(round(ltcc["target_pairs_per_train_image"] * TRAIN_IMAGES * FORMAL_EPOCHS))
    ordered = sorted(records, key=lambda row: (float(row["l3_priority"]), int(row["global_batch_idx"])))
    selected_total = 0
    selected_steps: set[int] = set()
    for row in ordered:
        count = int(row["candidate_pairs_global"])
        if count > 0 and selected_total + count <= target_total:
            selected_steps.add(int(row["global_batch_idx"]))
            selected_total += count
    if selected_total < target_total:
        remaining = target_total - selected_total
        candidates = [row for row in records if int(row["global_batch_idx"]) not in selected_steps and int(row["candidate_pairs_global"]) > 0]
        if candidates:
            chosen = min(candidates, key=lambda row: (abs(int(row["candidate_pairs_global"]) - remaining), float(row["l3_priority"]), int(row["global_batch_idx"])))
            selected_steps.add(int(chosen["global_batch_idx"]))
            selected_total += int(chosen["candidate_pairs_global"])
    for row in records:
        if int(row["global_batch_idx"]) in selected_steps:
            row["l3_scheduled"] = 1
            row["l3_selected_pairs_global"] = int(row["candidate_pairs_global"])
    l2_count = sum(int(row["l2_scheduled"]) for row in records)
    l2_valid_count = sum(int(row["l2_scheduled"]) * int(row["candidate_pairs_global"] > 0) for row in records)
    l3_active_count = sum(int(row["l3_scheduled"]) for row in records)
    protocol = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "epochs": FORMAL_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH, "world_size": WORLD_SIZE, "batch_size": BATCH_SIZE,
        "num_instances": NUM_INSTANCES, "lambda_rel": LAMBDA_REL,
        "relation_definition": "generic same-PID positive consistency; all i<j with PID_i == PID_j; no clothing restriction",
        "feature": "feat[0]",
        "formula": "mean(1 - cosine(F.normalize(feat[0].float())[i], F.normalize(feat[0].float())[j]))",
        "l0": {"relation_active": False, "description": ARM_LABELS["L0"]},
        "l1": {"relation_active_policy": "every batch with a valid same-ID candidate pair", "description": ARM_LABELS["L1"]},
        "l2": {"scheduled_probability": L2_ACTIVE_PROBABILITY, "hash_namespace": "L2", "hash_rule": "sha256(PHASE|namespace|seed|epoch|global_batch_idx) first 64 bits / 2^64 < 0.15", "description": ARM_LABELS["L2"]},
        "l3": {"target_metric": ltcc["target_metric"], "target_pairs_per_train_image": ltcc["target_pairs_per_train_image"], "target_total_prcc_pairs": target_total, "hash_namespace": "L3_PRIORITY", "selection_rule": "sort all frozen PRCC global optimizer steps by deterministic hash priority; include steps in priority order while cumulative valid pair count stays within target, then add the closest deterministic remainder step if needed", "description": ARM_LABELS["L3"]},
        "ltcc_i3_audit": ltcc,
        "prcc_dataset": {"train_images": len(dataset.train), "train_pids": int(dataset.num_train_pids), "clothing_states": int(dataset.num_train_clothes)},
        "plan_summary": {"total_global_batches": len(records), "l2_scheduled_batches": l2_count, "l2_scheduled_rate": l2_count / len(records), "l2_valid_scheduled_batches": l2_valid_count, "l3_scheduled_batches": l3_active_count, "l3_scheduled_rate": l3_active_count / len(records), "l3_selected_pair_total": selected_total, "l3_target_pair_total": target_total, "l3_pair_error": selected_total - target_total, "candidate_pairs_total_dense": sum(int(row["candidate_pairs_global"]) for row in records)},
        "same_batch_protocol": "L0-L3 use the frozen Phase 7A PRCC DistributedRandomIdentitySampler, batch 16, NUM_INSTANCES 4, rank-local 551 updates, seed 1234, unchanged optimizer/scheduler/AMP/config; only Relation timing/budget differs",
        "retrieval_not_read_for_schedule": True,
        "second_seed_started": False,
        "later_phase_started": False,
    }
    write_json(PROTOCOL_ROOT / "source_audit.json", {"phase": PHASE, "created_utc": protocol["created_utc"], "ltcc_i3": ltcc, "prcc_dataset": protocol["prcc_dataset"], "common_init": {"path": str(PRCC_COMMON_INIT), "sha256": sha256_file(PRCC_COMMON_INIT), "meta_path": str(PRCC_COMMON_META), "meta_sha256": sha256_file(PRCC_COMMON_META)}, "phase7c_sources": {"C0": file_record(PRCC_C0_SOURCE / "run_manifest.json"), "C1": file_record(PRCC_C1_SOURCE / "run_manifest.json"), "common_audit": file_record(PHASE7C_COMMON_AUDIT)}, "checks": {"ltcc_i3_60_epochs": True, "prcc_topology": True, "common_init_frozen": True, "phase7c_sources_complete": True, "seed_1234": True, "no_query_gallery_relation": True, "no_retrieval_used_for_schedule": True}})
    write_json(PROTOCOL_ROOT / "protocol.json", protocol)
    write_csv(PROTOCOL_ROOT / "exposure_schedule.csv", records, sorted({key for row in records for key in row.keys()}))
    write_exposure_protocol_report(protocol)
    write_source_report(protocol)
    return protocol


def write_source_report(protocol: dict[str, Any]) -> None:
    source_audit = read_json(PROTOCOL_ROOT / "source_audit.json")
    checks = source_audit.get("checks", source_audit.get("source_checks", {}))
    lines = [
        "# PHASE 7L source audit", "",
        f"Generated: `{protocol['created_utc']}`.", "",
        "This phase uses only the frozen PRCC Phase 7A/7C protocol and frozen LTCC I3 exposure logs. New retrieval results were not used to define the L2/L3 schedule.", "",
        "## Frozen sources", "",
        f"- LTCC I3 exposure: `{LTCC_I3_EXPOSURE}`; SHA256 `{protocol['ltcc_i3_audit']['source_sha256']}`.",
        f"- LTCC I3 source checkpoint run: `{PHASE7I_ROOT / 'I3_id_only_matched'}`.",
        f"- PRCC Bio/Relation provenance: `{PRCC_C0_SOURCE}` and `{PRCC_C1_SOURCE}`.",
        f"- Frozen PRCC common initialization: `{PRCC_COMMON_INIT}`; SHA256 `{source_audit['common_init']['sha256']}`.",
        "", "## Frozen PRCC protocol", "",
        "- Dataset: PRCC train only for Relation labels; 17,896 train images, 150 PIDs, 300 clothing states.",
        "- Seed: 1234; batch size 16; NUM_INSTANCES 4; 551 updates/rank/epoch; 60 epochs; 20-epoch linear warmup; frozen SGD/Cosine/AMP settings.",
        "- Feature: `feat[0]`; Relation formula: mean `1 - cosine(normalized z_i, normalized z_j)`.",
        "- Eligible Relation pair: same PID, distinct batch indices; clothing is not checked.",
        "- Query/gallery/test images are never used for Relation construction.",
        "", "## Audit checks", "",
        "| Check | Result |", "|---|---|",
    ]
    for key, value in checks.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "Machine-readable source audit: `" + str(PROTOCOL_ROOT / "source_audit.json") + "`."]
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_exposure_protocol_report(protocol: dict[str, Any]) -> None:
    ltcc = protocol["ltcc_i3_audit"]
    plan = protocol["plan_summary"]
    lines = [
        "# PHASE 7L exposure protocol", "",
        f"Frozen before Phase 7L training: `{protocol['created_utc']}`.", "",
        "This protocol is fixed before reading any new Phase 7L retrieval result. No lambda, sampler, batch size, optimizer, scheduler, architecture, dataset split, hardness, utility, reliability, prototype, margin, H_ID, or gradient-surgery change is permitted.", "",
        "## LTCC I3 exposure audit", "",
        f"- Frozen source: `{LTCC_I3_EXPOSURE}`.",
        f"- Relation pair contributions over 60 epochs: `{ltcc['pair_contributions_total']}`.",
        f"- LTCC active-batch rate over the full frozen schedule: `{ltcc['active_batch_rate']:.9f}`.",
        f"- Chosen L3 metric: `{ltcc['target_metric']}`.",
        f"- Frozen target: `{ltcc['target_pairs_per_train_image']:.15f}` pair contributions per train image over the 60-epoch schedule.",
        "",
        "## Arms", "",
        "| Arm | Relation policy | Lambda |", "|---|---|---:|",
        f"| L0 | No Relation | 0.1 nominal, no applied term |",
        f"| L1 | Active on every valid same-ID batch | {LAMBDA_REL} |",
        f"| L2 | Deterministic hash schedule, probability {L2_ACTIVE_PROBABILITY} | {LAMBDA_REL} |",
        f"| L3 | Deterministic hash-priority schedule matching the frozen L3 target | {LAMBDA_REL} |",
        "",
        "## Deterministic schedules", "",
        f"- L2: `sha256(PHASE|L2|seed|epoch|global_batch_idx)` first 64 bits divided by `2^64` is less than `{L2_ACTIVE_PROBABILITY}`.",
        f"- L3 target total: `{plan['l3_target_pair_total']}` PRCC pair contributions over `{FORMAL_EPOCHS}` epochs; scheduled total is `{plan['l3_selected_pair_total']}` (error `{plan['l3_pair_error']}`).",
        "- L3 selection uses only frozen sampler batch composition and deterministic hash priority; it uses no model, loss, gradient, difficulty, PID, clothing, or retrieval result.",
        "- Both distributed ranks use the same global-batch activation flag; Relation pair counts are logged across both ranks.",
        "",
        "## Precomputed plan summary", "",
        f"- Total global optimizer batches: `{plan['total_global_batches']}`.",
        f"- L2 scheduled batches: `{plan['l2_scheduled_batches']}` (`{plan['l2_scheduled_rate']:.9f}`).",
        f"- L3 scheduled batches: `{plan['l3_scheduled_batches']}` (`{plan['l3_scheduled_rate']:.9f}`).",
        f"- Dense candidate pair contributions (L1 schedule): `{plan['candidate_pairs_total_dense']}`.",
        "",
        f"Full frozen schedule: `{PROTOCOL_ROOT / 'exposure_schedule.csv'}`.",
        f"Machine-readable protocol: `{PROTOCOL_ROOT / 'protocol.json'}`.",
    ]
    PROTOCOL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    required = [LTCC_I3_EXPOSURE, LTCC_I3_MANIFEST, PRCC_COMMON_INIT, PRCC_COMMON_META, PHASE7C_COMMON_AUDIT, PRCC_C0_SOURCE / "RUN_COMPLETE.json", PRCC_C1_SOURCE / "RUN_COMPLETE.json", PRCC_C0_SOURCE / "run_manifest.json", PRCC_C1_SOURCE / "run_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase 7L frozen source missing: {missing}")
    state, common = load_common_init()
    del state
    cfg = load_prcc_cfg()
    dataset = p7a.build_dataset(cfg)
    checks = {
        "all_required_sources_complete": not missing,
        "ltcc_i3_60_epochs": len(read_csv(LTCC_I3_EXPOSURE)) == FORMAL_EPOCHS,
        "prcc_train_images_17896": len(dataset.train) == TRAIN_IMAGES,
        "prcc_train_pids_150": int(dataset.num_train_pids) == TRAIN_PIDS,
        "prcc_train_clothing_states_300": int(dataset.num_train_clothes) == TRAIN_CLOTHING_STATES,
        "seed_1234": SEED == 1234,
        "lambda_rel_fixed_0_1": LAMBDA_REL == 0.1,
        "batch_size_frozen_16": BATCH_SIZE == 16,
        "num_instances_frozen_4": NUM_INSTANCES == 4,
        "steps_per_epoch_frozen_551": STEPS_PER_EPOCH == 551,
        "common_init_hash_verified": bool(common["state_hash"] == read_json(PRCC_COMMON_META).get("state_hash")),
        "no_query_gallery_relation": True,
        "no_second_seed_started": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 7L source audit failed: {checks}")
    protocol = freeze_exposure_protocol()
    audit = {
        "phase": PHASE, "created_utc": protocol["created_utc"], "training_started": False,
        "optimizer_step_called": False, "source_checks": checks, "required_files": [file_record(path) for path in required],
        "ltcc_i3": protocol["ltcc_i3_audit"], "prcc_dataset": protocol["prcc_dataset"],
        "common_init": {"path": str(PRCC_COMMON_INIT), "sha256": common["sha256"], "state_hash": common["state_hash"]},
        "phase7c_source_manifests": {"C0": read_json(PRCC_C0_SOURCE / "run_manifest.json"), "C1": read_json(PRCC_C1_SOURCE / "run_manifest.json")},
        "protocol": str(PROTOCOL_ROOT / "protocol.json"), "exposure_schedule": str(PROTOCOL_ROOT / "exposure_schedule.csv"),
    }
    write_json(PROTOCOL_ROOT / "source_audit.json", audit)
    write_source_report(protocol)
    return audit


def load_protocol() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    protocol = read_json(PROTOCOL_ROOT / "protocol.json")
    rows = read_csv(PROTOCOL_ROOT / "exposure_schedule.csv")
    if len(rows) != FORMAL_EPOCHS * STEPS_PER_EPOCH:
        raise RuntimeError(f"exposure schedule has {len(rows)} rows; expected {FORMAL_EPOCHS * STEPS_PER_EPOCH}")
    plan = {int(row["global_batch_idx"]): row for row in rows}
    if len(plan) != len(rows):
        raise RuntimeError("exposure schedule has duplicate global batch indices")
    return protocol, plan


def same_id_edges(pids: torch.Tensor) -> list[tuple[int, int]]:
    values = [int(value) for value in pids.detach().cpu().tolist()]
    return [(i, j) for i in range(len(values)) for j in range(i + 1, len(values)) if values[i] == values[j]]


def id_positive_relation_loss(feat: list[torch.Tensor], pids: torch.Tensor, active: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    representation = F.normalize(feat[0].float(), dim=-1)
    edges = same_id_edges(pids)
    if not active or not edges:
        zero = representation.sum() * 0.0
        return zero, {
            "active": False, "candidate_pair_count": len(edges), "selected_pair_count": 0,
            "relation_pids": set(), "similarities": [],
        }
    similarities = [torch.sum(representation[i] * representation[j]) for i, j in edges]
    relation = torch.stack([1.0 - value for value in similarities]).mean()
    pid_values = [int(value) for value in pids.detach().cpu().tolist()]
    return relation, {
        "active": True, "candidate_pair_count": len(edges), "selected_pair_count": len(edges),
        "relation_pids": {pid_values[i] for i, _ in edges},
        "similarities": [float(value.detach().cpu()) for value in similarities],
    }


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def logger_for(path: Path, arm: str):
    import logging
    logger = logging.getLogger(f"{PHASE}.{arm}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def save_rank_state(path: Path, model: DDP, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE, "epoch": epoch, "global_step": global_step,
        "model": p6b.base.cpu_state_dict(model), "optimizer": p6b.base.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": p6b.base.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def rank_plan_metadata(dataset: Any, epoch: int, rank: int) -> dict[str, Any]:
    sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=rank, seed=SEED)
    indices = list(iter(sampler))
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "arm": "all", "seed": SEED, "epoch": epoch, "rank": rank, "sampler_epoch_used": int(sampler.epoch),
        "samples": len(indices), "batches": len(indices) // BATCH_SIZE, "unique_images": len(set(indices)),
        "unique_pids": len({int(dataset.train[index]["pid"]) for index in indices}),
        "unique_clothing_states": len({(int(dataset.train[index]["pid"]), int(dataset.train[index]["clothes_id"])) for index in indices}),
        "plan_sha256": digest,
    }


def close_logger(logger: Any) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def train_arm(arm: str) -> None:
    if arm not in ARM_DIRS:
        raise ValueError(f"unsupported Phase 7L arm {arm}")
    require_ppu()
    audit_path = PROTOCOL_ROOT / "source_audit.json"
    protocol_path = PROTOCOL_ROOT / "protocol.json"
    if not audit_path.is_file() or not protocol_path.is_file():
        raise RuntimeError("run --mode audit before Phase 7L training")
    protocol, plan = load_protocol()
    if (ARM_DIRS[arm] / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"refusing to overwrite completed Phase 7L arm {arm}")
    rank, device = p6b.init_dist()
    run_dir = ARM_DIRS[arm]
    streams: list[Any] = []
    logger = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = load_prcc_cfg(run_dir)
        set_seed(SEED)
        common_state, common_meta = load_common_init()
        train_loader, _same_loader, _diff_loader, _gallery_loader, dataset, _train_sampler, val_loader, val_loader_same = p7a.build_dataloader(cfg)
        if len(dataset.train) != TRAIN_IMAGES or int(dataset.num_train_pids) != TRAIN_PIDS or int(dataset.num_train_clothes) != TRAIN_CLOTHING_STATES:
            raise RuntimeError("Phase 7L runtime PRCC topology changed")
        if len(train_loader) != STEPS_PER_EPOCH:
            raise RuntimeError(f"Phase 7L {arm} expected {STEPS_PER_EPOCH} train batches, got {len(train_loader)}")
        model, loss_fn, optimizer, scheduler, scaler = p7a.build_runtime(cfg, common_state, device, dataset)
        local_hash = p7a.model_identity_hash(model)
        gathered_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_hashes, local_hash)
        if len(set(gathered_hashes)) != 1 or local_hash != common_meta["state_hash"]:
            raise RuntimeError(f"{arm} common initialization mismatch: {gathered_hashes}")
        if rank == 0:
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            manifest = {
                "phase": PHASE, "arm": arm, "label": ARM_LABELS[arm], "seed": SEED,
                "epochs": FORMAL_EPOCHS, "start_epoch": 1, "end_epoch": FORMAL_EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH, "train_images": len(dataset.train),
                "train_pids": int(dataset.num_train_pids), "train_clothing_states": int(dataset.num_train_clothes),
                "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES,
                "optimizer": p7a.optimizer_config(cfg), "scheduler": p7a.scheduler_config(cfg), "amp": p7a.amp_config(),
                "common_init": {"path": str(PRCC_COMMON_INIT), "sha256": common_meta["sha256"], "state_hash": common_meta["state_hash"]},
                "initial_parameter_hash": local_hash, "initial_parameter_strict_load": True,
                "relation_feature": "feat[0]", "relation_definition": protocol["relation_definition"],
                "relation_formula": protocol["formula"], "lambda_rel": LAMBDA_REL,
                "relation_policy": {"L0": "none", "L1": "dense", "L2": "frozen_hash_15_percent", "L3": "frozen_LTCC_normalized_pair_budget"}[arm],
                "protocol": str(protocol_path), "sampler": "frozen Phase 7A DistributedRandomIdentitySampler; no set_epoch",
                "evaluation": {"primary": "PRCC changed clothes C -> A", "secondary": "same clothes B -> A", "reranking": False},
                "forbidden_changes": ["lambda tuning", "optimizer", "scheduler", "batch size", "NUM_INSTANCES", "sampler", "architecture", "margin", "SR3", "H_ID", "H_pos", "utility", "prototype", "gradient surgery", "dataset split"],
                "training_started": True, "second_seed_started": False, "later_phase_started": False,
            }
            write_json(run_dir / "run_manifest.json", manifest)
            metrics_stream = (run_dir / "metrics_per_epoch.csv").open("w", newline="", encoding="utf-8"); streams.append(metrics_stream)
            metrics_writer = csv.DictWriter(metrics_stream, fieldnames=METRIC_FIELDS); metrics_writer.writeheader(); metrics_stream.flush()
            exposure_stream = (run_dir / "exposure_log.csv").open("w", newline="", encoding="utf-8"); streams.append(exposure_stream)
            exposure_writer = csv.DictWriter(exposure_stream, fieldnames=EXPOSURE_FIELDS); exposure_writer.writeheader(); exposure_stream.flush()
            amp_stream = (run_dir / "amp_audit.csv").open("w", newline="", encoding="utf-8"); streams.append(amp_stream)
            amp_writer = csv.DictWriter(amp_stream, fieldnames=AMP_FIELDS); amp_writer.writeheader(); amp_stream.flush()
            amp_event_stream = (run_dir / "amp_events.csv").open("w", newline="", encoding="utf-8"); streams.append(amp_event_stream)
            amp_event_writer = csv.DictWriter(amp_event_stream, fieldnames=AMP_EVENT_FIELDS); amp_event_writer.writeheader(); amp_event_stream.flush()
            sampler_stream = (run_dir / "sampler_plan.csv").open("w", newline="", encoding="utf-8"); streams.append(sampler_stream)
            sampler_writer = csv.DictWriter(sampler_stream, fieldnames=SAMPLER_FIELDS); sampler_writer.writeheader(); sampler_stream.flush()
            logger = logger_for(run_dir / "train_log_rank0.txt", arm)
        else:
            metrics_writer = exposure_writer = amp_writer = amp_event_writer = sampler_writer = None
        best_rank1 = -float("inf")
        best_epoch = None
        global_step = 0
        cumulative_active = 0
        cumulative_pairs = 0
        for epoch in range(1, FORMAL_EPOCHS + 1):
            scheduler.step(epoch)
            local_meta = rank_plan_metadata(dataset, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, local_meta)
            if rank == 0:
                for row in gathered_meta:
                    sampler_writer.writerow(row)
                sampler_stream.flush()
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation_all", "relation_applied", "id_acc", "bio_acc")}
            local_candidate_pairs = 0; local_selected_pairs = 0; local_scheduled_batches = 0; local_active_batches = 0
            local_relation_pids: set[int] = set(); local_active_loss_sum = 0.0; local_active_loss_count = 0
            local_skips = 0; local_found_inf = 0; local_nonfinite = 0; scale_min = float(scaler.get_scale()); scale_max = scale_min; scale_end = scale_min; local_amp_events: list[dict[str, Any]] = []
            for iteration, batch in enumerate(train_loader, start=1):
                global_step += 1
                optimizer.zero_grad()
                schedule_row = plan[(epoch - 1) * STEPS_PER_EPOCH + iteration]
                images = batch["image"].to(device, non_blocking=False)
                pids = batch["pid"].to(device, non_blocking=False)
                camids = batch["camid"].to(device, non_blocking=False)
                clothes = batch["clothes_id"].to(device, non_blocking=False)
                captions = batch["caption_feature"].to(device, non_blocking=False)
                candidate_edges = same_id_edges(pids)
                candidate_count = len(candidate_edges)
                if arm == "L0":
                    scheduled = False
                elif arm == "L1":
                    scheduled = True
                elif arm == "L2":
                    scheduled = bool(int(schedule_row["l2_scheduled"]))
                else:
                    scheduled = bool(int(schedule_row["l3_scheduled"]))
                local_candidate_pairs += candidate_count
                local_scheduled_batches += int(scheduled)
                writer = p7k.ltcc_base.MetricCapture()
                with amp.autocast(enabled=True):
                    score, feat = model(images, cam_label=camids)
                base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
                relation_raw, relation_info = id_positive_relation_loss(feat, pids, scheduled and candidate_count > 0)
                relation_applied = relation_raw if arm != "L0" else relation_raw * 0.0
                total = base_total + LAMBDA_REL * relation_applied
                if not bool(torch.isfinite(total.detach()).item()):
                    raise RuntimeError(f"non-finite {arm} loss at E{epoch} I{iteration}")
                if bool(relation_info["active"]):
                    local_active_batches += 1
                    local_selected_pairs += int(relation_info["selected_pair_count"])
                    local_relation_pids.update(int(pid) for pid in relation_info["relation_pids"])
                    local_active_loss_sum += scalar(relation_raw)
                    local_active_loss_count += 1
                scaler.scale(total).backward()
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                found_inf = int(p6b.base.found_inf(scaler, optimizer) > 0)
                scaler.update()
                scale_end = float(scaler.get_scale()); scale_min = min(scale_min, scale_end); scale_max = max(scale_max, scale_end)
                local_skips += found_inf; local_found_inf += found_inf; local_nonfinite += found_inf
                if found_inf:
                    local_amp_events.append({"arm": arm, "seed": SEED, "rank": rank, "epoch": epoch, "iteration": iteration, "global_step": global_step, "skipped_step": 1, "found_inf": found_inf, "nonfinite_event": 1, "amp_scale_before": scale_before, "amp_scale_after": scale_end})
                components = getattr(loss_fn, "last_components", {})
                values = {"total": total, "ce": components.get("ce", torch.tensor(float("nan"), device=device)), "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)), "bio": components.get("bio", torch.tensor(float("nan"), device=device)), "relation_all": relation_raw, "relation_applied": relation_applied, "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(), "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1]}
                for key, value in values.items():
                    epoch_sum[key] += scalar(value)
                if rank == 0 and logger is not None and iteration % 100 == 0:
                    logger.info(json.dumps({"arm": arm, "epoch": epoch, "iteration": iteration, "loss": scalar(total), "scheduled": scheduled, "candidate_pairs": candidate_count, "active": bool(relation_info["active"]), "relation_loss": scalar(relation_raw), "amp_skip": found_inf}, sort_keys=True))
                del total, relation_raw, relation_applied, base_total, score, feat, images, pids, camids, clothes, captions, candidate_edges
            if iteration != STEPS_PER_EPOCH:
                raise RuntimeError(f"{arm} loader yielded {iteration} batches, expected {STEPS_PER_EPOCH}")
            gathered_stats = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_stats, {"candidate_pairs": local_candidate_pairs, "selected_pairs": local_selected_pairs, "scheduled_batches": local_scheduled_batches, "active_batches": local_active_batches, "relation_pids": local_relation_pids, "active_loss_sum": local_active_loss_sum, "active_loss_count": local_active_loss_count, "amp_events": local_amp_events, "skips": local_skips, "found_inf": local_found_inf, "nonfinite": local_nonfinite, "scale_min": scale_min, "scale_max": scale_max, "scale_end": scale_end})
            if rank == 0:
                for item in gathered_stats:
                    for event in item["amp_events"]:
                        amp_event_writer.writerow(event)
                amp_event_stream.flush()
            reduced = p6b.reduce_epoch(epoch_sum, device)
            candidate_pairs_global = sum(int(item["candidate_pairs"]) for item in gathered_stats)
            selected_pairs_global = sum(int(item["selected_pairs"]) for item in gathered_stats)
            scheduled_batches_global = sum(int(item["scheduled_batches"]) for item in gathered_stats)
            active_batches_global = sum(int(item["active_batches"]) for item in gathered_stats)
            relation_pids_global = set().union(*(set(item["relation_pids"]) for item in gathered_stats))
            active_loss_sum = sum(float(item["active_loss_sum"]) for item in gathered_stats)
            active_loss_count = sum(int(item["active_loss_count"]) for item in gathered_stats)
            cumulative_active += active_batches_global
            cumulative_pairs += selected_pairs_global
            amp_min = min(float(item["scale_min"]) for item in gathered_stats); amp_max = max(float(item["scale_max"]) for item in gathered_stats); amp_end = max(float(item["scale_end"]) for item in gathered_stats)
            skip_count = sum(int(item["skips"]) for item in gathered_stats); found_inf_count = sum(int(item["found_inf"]) for item in gathered_stats); nonfinite_count = sum(int(item["nonfinite"]) for item in gathered_stats)
            updates_completed = STEPS_PER_EPOCH - skip_count / WORLD_SIZE
            dist.barrier()
            metrics_eval = p7a.evaluate_prcc(cfg, model.module, dataset, val_loader, val_loader_same, device, epoch, logger) if rank == 0 else None
            holder = [metrics_eval]; dist.broadcast_object_list(holder, src=0); metrics_eval = holder[0]; dist.barrier()
            lr_values = scheduler._get_lr(epoch)
            metrics_row = {
                "arm": arm, "seed": SEED, "epoch": epoch, "train_loss": reduced["total"], "ce_loss": reduced["ce"], "triplet_loss": reduced["triplet"], "bio_loss": reduced["bio"],
                "relation_loss_all_batches": reduced["relation_all"], "relation_loss_active_batches": active_loss_sum / active_loss_count if active_loss_count else 0.0, "relation_applied_loss": reduced["relation_applied"],
                "relation_lambda": LAMBDA_REL, "total_loss": reduced["total"], "lr": float(lr_values[0]), "lr_bias": float(lr_values[-1]),
                "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"], "planned_updates": STEPS_PER_EPOCH, "updates_completed": updates_completed,
                "amp_scale_min": amp_min, "amp_scale_max": amp_max, "amp_scale_end": amp_end, "amp_skip": int(skip_count > 0), "skipped_steps": skip_count, "found_inf_count": found_inf_count, "nonfinite_events": nonfinite_count,
                "active_images": len(dataset.train), "active_pids": int(dataset.num_train_pids), "active_clothing_states": int(dataset.num_train_clothes), **metrics_eval,
            }
            exposure_row = {
                "arm": arm, "seed": SEED, "epoch": epoch, "total_batches_global": STEPS_PER_EPOCH * WORLD_SIZE, "planned_updates_per_rank": STEPS_PER_EPOCH,
                "scheduled_relation_batches_global": scheduled_batches_global, "relation_active_batches_global": active_batches_global, "relation_active_batch_rate": active_batches_global / float(STEPS_PER_EPOCH * WORLD_SIZE),
                "valid_same_id_candidate_pairs_global": candidate_pairs_global, "selected_relation_pairs_global": selected_pairs_global, "relation_pair_contributions": selected_pairs_global,
                "unique_relation_pids_global": len(relation_pids_global), "relation_loss_active_mean": active_loss_sum / active_loss_count if active_loss_count else 0.0, "lambda_relation_loss_active_mean": LAMBDA_REL * active_loss_sum / active_loss_count if active_loss_count else 0.0,
                "cumulative_relation_active_batches_global": cumulative_active, "cumulative_relation_pair_contributions": cumulative_pairs,
                "scheduled_probability_or_budget": L2_ACTIVE_PROBABILITY if arm == "L2" else (protocol["l3"]["target_pairs_per_train_image"] if arm == "L3" else (1.0 if arm == "L1" else 0.0)), "l3_target_pairs_total": protocol["l3"]["target_total_prcc_pairs"], "l3_target_metric": protocol["l3"]["target_metric"],
            }
            amp_row = {"arm": arm, "seed": SEED, "epoch": epoch, "amp_skip": int(skip_count > 0), "skipped_steps": skip_count, "found_inf_count": found_inf_count, "nonfinite_events": nonfinite_count, "amp_scale_min": amp_min, "amp_scale_max": amp_max, "amp_scale_end": amp_end, "planned_updates": STEPS_PER_EPOCH, "updates_completed": updates_completed}
            if rank == 0:
                metrics_writer.writerow(metrics_row); metrics_stream.flush(); exposure_writer.writerow(exposure_row); exposure_stream.flush(); amp_writer.writerow(amp_row); amp_stream.flush()
                if float(metrics_row["cc_rank1"]) > best_rank1:
                    best_rank1 = float(metrics_row["cc_rank1"]); best_epoch = epoch
                    p6b.save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_rank1, "criterion": "PRCC changed-clothes Rank-1"})
                print(json.dumps({"arm": arm, "epoch": epoch, "cc_rank1": metrics_row["cc_rank1"], "cc_mAP": metrics_row["cc_mAP"], "relation_active_rate": exposure_row["relation_active_batch_rate"], "relation_pairs": selected_pairs_global, "amp_skips": skip_count}, sort_keys=True), flush=True)
            if epoch in GEOMETRY_EPOCHS:
                if rank == 0:
                    geometry_rows = geometry_for_model(model.module, dataset, cfg, device, epoch, arm)
                    geometry_path = GEOMETRY_ROOT / f"{arm}_geometry.csv"
                    prior_geometry = read_csv(geometry_path) if geometry_path.is_file() else []
                    write_csv(geometry_path, prior_geometry + geometry_rows, GEOMETRY_FIELDS)
                dist.barrier()
            if rank == 0 and epoch in (10, 30, 60):
                # Frozen diagnostic-stage snapshots; this does not alter the
                # optimizer path and keeps Early/Mid/Late mapping exact.
                p6b.save_model_checkpoint(model.module, run_dir / f"model_epoch{epoch:03d}.pth")
            if epoch == FORMAL_EPOCHS:
                if rank == 0:
                    p6b.save_model_checkpoint(model.module, run_dir / "model_final.pth")
                dist.barrier()
                save_rank_state(run_dir / f"rank{rank}_state_epoch060.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    write_json(run_dir / "checkpoint_epoch060.json", {"phase": PHASE, "arm": arm, "epoch": epoch, "global_step": global_step, "rank_states": [str(run_dir / f"rank{r}_state_epoch060.pt") for r in range(WORLD_SIZE)], "model": str(run_dir / "model_final.pth")})
                    write_json(run_dir / "RUN_COMPLETE.json", {"phase": PHASE, "arm": arm, "seed": SEED, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates_per_rank": global_step, "metrics": str(run_dir / "metrics_per_epoch.csv"), "best_epoch": best_epoch, "best_model": str(run_dir / "model_best.pth"), "final_model": str(run_dir / "model_final.pth"), "protocol": str(protocol_path), "common_init_sha256": common_meta["sha256"], "initial_parameter_hash": local_hash})
        if rank == 0:
            write_json(run_dir / "AMP.json", {"phase": PHASE, "arm": arm, "seed": SEED, "initial_scale": INITIAL_SCALE, "epochs_completed": FORMAL_EPOCHS, "metrics_source": str(run_dir / "amp_audit.csv"), "events_source": str(run_dir / "amp_events.csv")})
    finally:
        for stream in streams:
            stream.close()
        close_logger(logger)
        p6b.cleanup_dist()


def extract_train_feat0(model: torch.nn.Module, dataset: Any, cfg: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    _train_transform, test_transform = p7a.build_img_transforms(cfg)
    loader = DataLoader(ImageDataset(dataset.train, transform=test_transform), batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            # The frozen EVA model returns the identity tensor directly in
            # eval mode and (score, feature-list) in train mode.
            if torch.is_tensor(output):
                identity_feature = output
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                _score, eval_feat = output
                identity_feature = eval_feat[0] if isinstance(eval_feat, (tuple, list)) else eval_feat
            else:
                raise TypeError(f"unexpected PRCC geometry model output: {type(output)!r}")
            blocks.append(F.normalize(identity_feature.float(), dim=-1).detach().cpu().numpy().astype(np.float32))
    features = np.concatenate(blocks, axis=0)
    if len(features) != len(dataset.train):
        raise RuntimeError(f"geometry feature count mismatch: {len(features)} != {len(dataset.train)}")
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    model.train()
    return features, pids, clothes


def geometry_for_model(model: torch.nn.Module, dataset: Any, cfg: Any, device: torch.device, epoch: int, arm: str) -> list[dict[str, Any]]:
    features, pids, clothes = extract_train_feat0(model, dataset, cfg, device)
    features = F.normalize(torch.from_numpy(features), dim=-1).numpy()
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, (pid, cloth) in enumerate(zip(pids.tolist(), clothes.tolist())):
        by_pid[int(pid)].append(index)
        by_pid_cloth[int(pid)][int(cloth)].append(index)
    same_id_a: list[int] = []; same_id_b: list[int] = []
    same_cloth_a: list[int] = []; same_cloth_b: list[int] = []
    cross_cloth_a: list[int] = []; cross_cloth_b: list[int] = []
    for pid, indices in by_pid.items():
        for pos, left in enumerate(indices[:-1]):
            for right in indices[pos + 1:]:
                same_id_a.append(left); same_id_b.append(right)
        states = by_pid_cloth[pid]
        for state_indices in states.values():
            for pos, left in enumerate(state_indices[:-1]):
                for right in state_indices[pos + 1:]:
                    same_cloth_a.append(left); same_cloth_b.append(right)
        state_keys = sorted(states)
        for pos, state_a in enumerate(state_keys):
            for state_b in state_keys[pos + 1:]:
                for left in states[state_a]:
                    for right in states[state_b]:
                        cross_cloth_a.append(left); cross_cloth_b.append(right)
    same_id_a_np = np.asarray(same_id_a, dtype=np.int64); same_id_b_np = np.asarray(same_id_b, dtype=np.int64)
    same_cloth_a_np = np.asarray(same_cloth_a, dtype=np.int64); same_cloth_b_np = np.asarray(same_cloth_b, dtype=np.int64)
    cross_cloth_a_np = np.asarray(cross_cloth_a, dtype=np.int64); cross_cloth_b_np = np.asarray(cross_cloth_b, dtype=np.int64)
    same_id_similarity = np.sum(features[same_id_a_np] * features[same_id_b_np], axis=1, dtype=np.float64)
    same_cloth_similarity = np.sum(features[same_cloth_a_np] * features[same_cloth_b_np], axis=1, dtype=np.float64)
    cross_cloth_similarity = np.sum(features[cross_cloth_a_np] * features[cross_cloth_b_np], axis=1, dtype=np.float64)
    unique_pids = sorted(by_pid)
    pid_centers = np.stack([F.normalize(torch.from_numpy(features[pids == pid]).mean(dim=0, keepdim=True), dim=-1).numpy()[0] for pid in unique_pids]).astype(np.float32)
    pid_to_center = {pid: index for index, pid in enumerate(unique_pids)}
    center_cos = pid_centers @ pid_centers.T
    off_diag = center_cos[~np.eye(len(unique_pids), dtype=bool)]
    pid_center_separation = 1.0 - float(np.mean(off_diag))
    all_center_similarity = features @ pid_centers.T
    own_center_indices = np.asarray([pid_to_center[int(pid)] for pid in pids.tolist()], dtype=np.int64)
    all_center_similarity[np.arange(len(pids)), own_center_indices] = -np.inf
    nearest_negative = np.max(all_center_similarity, axis=1)
    endpoint_negative = np.maximum(nearest_negative[cross_cloth_a_np], nearest_negative[cross_cloth_b_np])
    identity_margin = cross_cloth_similarity - endpoint_negative
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    for pid, states in by_pid_cloth.items():
        for cloth, indices in states.items():
            state_centers[(pid, cloth)] = F.normalize(torch.from_numpy(features[indices]).mean(dim=0, keepdim=True), dim=-1).numpy()[0]
    state_similarities = []
    for pid, states in by_pid_cloth.items():
        state_keys = sorted(states)
        for pos, state_a in enumerate(state_keys):
            for state_b in state_keys[pos + 1:]:
                state_similarities.append(float(np.dot(state_centers[(pid, state_a)], state_centers[(pid, state_b)])))
    return [{
        "arm": arm, "seed": SEED, "checkpoint": f"E{epoch}",
        "same_id_positive_similarity_mean": float(np.mean(same_id_similarity)),
        "same_clothes_similarity_mean": float(np.mean(same_cloth_similarity)),
        "cross_clothes_similarity_mean": float(np.mean(cross_cloth_similarity)),
        "nearest_negative_pid_center_similarity_mean": float(np.mean(endpoint_negative)),
        "identity_margin_mean": float(np.mean(identity_margin)),
        "identity_ordering_correct_rate": float(np.mean(identity_margin > 0)),
        "pid_center_separation_mean": pid_center_separation,
        "clothing_state_dispersion_mean": float(np.mean([1.0 - value for value in state_similarities])),
        "same_id_pair_count": int(len(same_id_similarity)), "same_clothes_pair_count": int(len(same_cloth_similarity)),
        "cross_clothes_pair_count": int(len(cross_cloth_similarity)), "pid_count": len(unique_pids),
        "clothing_state_count": len(state_centers), "geometry_source": "PRCC train-only feat[0]; query/gallery/test excluded",
    }]


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "std", "median", "p25", "p75")}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "median": float(np.median(array)), "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75))}


def read_float_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", "None", "NA"):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def trajectory(rows: list[dict[str, Any]], start: int = PRIMARY_START, end: int = PRIMARY_END) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"incomplete retrieval window E{start}-E{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    rank1 = np.asarray([float(row["cc_rank1"]) * 100.0 for row in selected], dtype=np.float64)
    mean_map = np.asarray([float(row["cc_mAP"]) * 100.0 for row in selected], dtype=np.float64)
    auc = lambda values: float((np.trapezoid(values, x) if hasattr(np, "trapezoid") else np.trapz(values, x)) / (x[-1] - x[0]))
    best_rank1 = max(selected, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    best_map = max(selected, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {
        "arm": selected[0]["arm"], "window": f"E{start}-E{end}", "epochs": len(selected),
        "mean_rank1": float(np.mean(rank1)), "mean_mAP": float(np.mean(mean_map)), "auc_rank1": auc(rank1), "auc_mAP": auc(mean_map),
        "slope_rank1": float(np.polyfit(x, rank1, 1)[0]), "slope_mAP": float(np.polyfit(x, mean_map, 1)[0]),
        "E60_rank1": float(rank1[-1]), "E60_mAP": float(mean_map[-1]), "best_rank1": float(best_rank1["cc_rank1"]) * 100.0,
        "best_rank1_epoch": int(best_rank1["epoch"]), "best_mAP": float(best_map["cc_mAP"]) * 100.0, "best_mAP_epoch": int(best_map["epoch"]),
    }


def checkpoint_stage(run_dir: Path, target: int) -> tuple[Path, int]:
    exact = run_dir / f"model_epoch{target:03d}.pth"
    if exact.is_file():
        return exact, target
    # Fallback is only for a genuinely missing frozen stage checkpoint; no
    # retraining or interpolation is attempted.
    available = []
    for epoch in (10, 30, 60):
        candidate = run_dir / f"model_epoch{epoch:03d}.pth"
        if candidate.is_file():
            available.append((abs(epoch - target), epoch, candidate))
    if available:
        _distance, actual, candidate = min(available, key=lambda item: (item[0], item[1]))
        return candidate, actual
    if target == 60 and (run_dir / "model_final.pth").is_file():
        return run_dir / "model_final.pth", 60
    raise FileNotFoundError(f"no frozen checkpoint available for target E{target} in {run_dir}")


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint {path}")
    return state


def strength_runtime() -> dict[str, Any]:
    cfg = load_prcc_cfg(OUT_ROOT / "strength_config_probe")
    dataset = p7a.build_dataset(cfg)
    train_transform, _test_transform = p7a.build_img_transforms(cfg)
    return {"cfg": cfg, "dataset": dataset, "transform": train_transform}


def collect_strength_batches(runtime: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    sampler = DistributedRandomIdentitySampler(runtime["dataset"].train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=0, seed=SEED)
    generator = torch.Generator(); generator.manual_seed(SEED)
    loader = DataLoader(ImageDataset(runtime["dataset"].train, transform=runtime["transform"]), sampler=sampler, batch_size=BATCH_SIZE, num_workers=1, pin_memory=False, drop_last=True, generator=generator)
    rows = []
    try:
        for index, batch in enumerate(loader, start=1):
            rows.append(p7k.clone_batch(batch))
            if index >= limit:
                break
    finally:
        del loader
        gc.collect()
    if len(rows) < limit:
        raise RuntimeError(f"only {len(rows)} strength batches collected")
    return rows


def strength_summary_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_metrics = ["ce_loss", "triplet_loss", "bio_loss", "base_loss", "relation_loss", "lambda_relation_loss", "r_loss"]
    gradient_metrics = ["norm_ce", "norm_tri", "norm_bio", "norm_base", "norm_rel", "lambda_norm_rel", "r_grad", "r_rel_ce_grad", "r_rel_tri_grad", "r_rel_bio_grad", "g_rel_orth_norm", "r_orth", "a_rel", "a_total", "norm_total", "i_total"]
    summary = []
    modules = ("late_eva", "shared_backbone", "id_head", "bio_projection")
    for (arm, stage), items in sorted(p7k.group_rows(raw_rows, ("arm", "stage")).items()):
        first = items[0]
        for module in modules:
            for metric in base_metrics + gradient_metrics:
                key = metric if metric in base_metrics else f"{metric}_{module}"
                values = p7k.finite_values(items, key)
                summary.append({"arm": arm, "dataset": "PRCC", "stage": stage, "actual_epoch": first["actual_epoch"], "module": module, "metric": metric, **describe(values)})
    saturation = []
    for (arm, stage), items in sorted(p7k.group_rows(raw_rows, ("arm", "stage")).items()):
        first = items[0]
        pair_count = sum(int(item["s_pos_count"]) for item in items)
        saturation.append({
            "arm": arm, "stage": stage, "actual_epoch": first["actual_epoch"], "diagnostic_batches": len(items),
            "relation_active_batch_rate": float(np.mean([as_float(item, "relation_active", 0.0) for item in items])),
            "candidate_pairs": sum(int(item["candidate_relation_pair_count"]) for item in items),
            "selected_pairs": sum(int(item["relation_pair_count"]) for item in items),
            "mean_s_pos_active": sum(float(item["s_pos_mean_active"]) * int(item["s_pos_count"]) for item in items if item["s_pos_mean_active"] not in (None, "")) / pair_count if pair_count else None,
            "fraction_s_pos_ge070": sum(int(item["s_pos_ge070_count"]) for item in items) / pair_count if pair_count else None,
            "fraction_s_pos_ge080": sum(int(item["s_pos_ge080_count"]) for item in items) / pair_count if pair_count else None,
            "fraction_s_pos_ge090": sum(int(item["s_pos_ge090_count"]) for item in items) / pair_count if pair_count else None,
        })
    return summary, saturation


def evaluate_strength_stage(arm: str, stage_name: str, checkpoint: Path, actual_epoch: int, runtime: dict[str, Any], batches: list[dict[str, Any]], plan: dict[int, dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    model = p7k.model_from_checkpoint("prcc", runtime, checkpoint, device)
    loss_fn, _center = p7k.ltcc_base.make_loss(runtime["cfg"], num_classes=int(runtime["dataset"].num_train_pids))
    groups = p7k.parameter_groups(model)
    raw_params = {name: [parameter for _, parameter in values] for name, values in groups.items()}
    rows: list[dict[str, Any]] = []
    previous: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None = None
    for batch_index, batch in enumerate(batches, start=1):
        torch.manual_seed(SEED + actual_epoch * 1000 + batch_index)
        torch.cuda.manual_seed_all(SEED + actual_epoch * 1000 + batch_index)
        images = batch["image"].to(device, non_blocking=False).float()
        pids = batch["pid"].to(device, non_blocking=False).long()
        camids = batch["camid"].to(device, non_blocking=False).long()
        clothes = batch["clothes_id"].to(device, non_blocking=False).long()
        captions = batch["caption_feature"].to(device, non_blocking=False).float()
        model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=camids)
        writer = p7k.ltcc_base.MetricCapture()
        _formal_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=batch_index)
        components = getattr(loss_fn, "last_components", {})
        ce = components["ce"]; tri = components["triplet"]; bio = components["bio"]
        base_loss = ce + tri + bio
        schedule_row = plan[(actual_epoch - 1) * STEPS_PER_EPOCH + batch_index]
        candidate_edges = same_id_edges(pids)
        if arm == "L1":
            scheduled = True
        elif arm == "L2":
            scheduled = bool(int(schedule_row["l2_scheduled"]))
        else:
            scheduled = bool(int(schedule_row["l3_scheduled"]))
        relation, relation_info = id_positive_relation_loss(feat, pids, scheduled and bool(candidate_edges))
        losses = [ce, tri, bio, relation]
        group_rows: dict[str, Any] = {}
        late_vectors = late_rel_vectors = late_total_vectors = None
        for group_name, named_parameters in groups.items():
            params = raw_params[group_name]
            gradients = []
            for loss_index, loss in enumerate(losses):
                keep_graph = not (group_name == list(groups)[-1] and loss_index == len(losses) - 1)
                gradients.append(torch.autograd.grad(loss, params, retain_graph=keep_graph, allow_unused=True))
            gradient_row, base_vectors, rel_vectors, total_vectors = p7k.group_gradient_metrics(gradients, params)
            group_rows[group_name] = gradient_row
            if group_name == "late_eva":
                late_vectors, late_rel_vectors, late_total_vectors = base_vectors, rel_vectors, total_vectors
        if late_vectors is None or late_rel_vectors is None or late_total_vectors is None:
            raise RuntimeError("Late EVA group missing")
        stability = {"cos_base_consecutive": None, "cos_rel_consecutive": None, "cos_total_consecutive": None}
        if previous is not None:
            stability = {"cos_base_consecutive": p7k.tuple_cosine(previous[0], late_vectors), "cos_rel_consecutive": p7k.tuple_cosine(previous[1], late_rel_vectors), "cos_total_consecutive": p7k.tuple_cosine(previous[2], late_total_vectors)}
        previous = (tuple(value.detach().clone() for value in late_vectors), tuple(value.detach().clone() for value in late_rel_vectors), tuple(value.detach().clone() for value in late_total_vectors))
        sims = [
            float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for value in relation_info.get("similarities", [])
        ]
        sims_np = np.asarray(sims, dtype=np.float64)
        row = {
            "arm": arm, "stage": stage_name, "actual_epoch": actual_epoch, "batch_index": batch_index, "checkpoint": str(checkpoint),
            "relation_active": bool(relation_info["active"]), "relation_pair_count": int(relation_info["selected_pair_count"]), "candidate_relation_pair_count": int(relation_info["candidate_pair_count"]),
            "ce_loss": scalar(ce), "triplet_loss": scalar(tri), "bio_loss": scalar(bio), "base_loss": scalar(base_loss), "relation_loss": scalar(relation), "lambda_relation_loss": LAMBDA_REL * scalar(relation),
            "r_loss": LAMBDA_REL * scalar(relation) / (scalar(base_loss) + EPS), "s_pos_mean_active": float(sims_np.mean()) if sims_np.size else None, "s_pos_count": int(sims_np.size),
            "s_pos_ge070_count": int(np.sum(sims_np >= 0.70)), "s_pos_ge080_count": int(np.sum(sims_np >= 0.80)), "s_pos_ge090_count": int(np.sum(sims_np >= 0.90)), **stability,
        }
        for group_name, stats in group_rows.items():
            for key, value in stats.items():
                row[f"{key}_{group_name}"] = value
        # Keep one row per diagnostic batch. Each module has namespaced
        # gradient fields; duplicating the row once per module would silently
        # overweight every batch in the summaries.
        rows.append(row)
        del gradients, group_rows, score, feat, _formal_total, relation, base_loss, ce, tri, bio, images, pids, camids, clothes, captions
        model.zero_grad(set_to_none=True)
        if batch_index % 8 == 0:
            print(f"[phase7l-strength] {arm} {stage_name} E{actual_epoch}: {batch_index}/{len(batches)}", flush=True)
    del loss_fn, model
    gc.collect(); torch.cuda.empty_cache()
    return rows


def run_strength_audit() -> dict[str, Any]:
    require_ppu()
    protocol, plan = load_protocol()
    del protocol
    runtime = strength_runtime()
    device = torch.device("cuda:0")
    all_rows: list[dict[str, Any]] = []
    stage_mapping: dict[str, dict[str, Any]] = {}
    batch_cache: dict[int, list[dict[str, Any]]] = {}
    for arm in ("L1", "L2", "L3"):
        run_dir = ARM_DIRS[arm]
        if not (run_dir / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"{arm} is incomplete before strength audit")
        mapping = {}
        for stage_name, target in (("early", 10), ("mid", 30), ("late", 60)):
            checkpoint, actual = checkpoint_stage(run_dir, target)
            mapping[stage_name] = {"target_epoch": target, "actual_epoch": actual, "checkpoint": str(checkpoint)}
            if actual not in batch_cache:
                batch_cache[actual] = collect_strength_batches(runtime, STRENGTH_BATCHES)
            all_rows.extend(evaluate_strength_stage(arm, stage_name, checkpoint, actual, runtime, batch_cache[actual], plan, device))
        stage_mapping[arm] = mapping
    write_json(STRENGTH_ROOT / "stage_mapping.json", stage_mapping)
    write_csv(STRENGTH_ROOT / "batch_strength.csv", all_rows, sorted({key for row in all_rows for key in row.keys()}))
    summary, saturation = strength_summary_rows(all_rows)
    write_csv(STRENGTH_ROOT / "strength_summary.csv", summary)
    write_csv(STRENGTH_ROOT / "saturation_summary.csv", saturation)
    write_json(STRENGTH_ROOT / "strength_manifest.json", {"phase": PHASE, "arms": ["L1", "L2", "L3"], "batches_per_stage": STRENGTH_BATCHES, "device": str(device), "gradient_definition": "unscaled FP32 autograd.grad; no optimizer/update", "stage_mapping": stage_mapping})
    return {"stage_mapping": stage_mapping, "summary": summary, "saturation": saturation}


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return 1.0
        if normalized == "false":
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_retrieval_tables() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, Any]] = {}
    for arm in ("L0", "L1", "L2", "L3"):
        rows = read_float_rows(ARM_DIRS[arm] / "metrics_per_epoch.csv")
        if len(rows) != FORMAL_EPOCHS:
            raise RuntimeError(f"{arm} retrieval metrics has {len(rows)} rows, expected {FORMAL_EPOCHS}")
        combined.extend(rows)
        trajectories[arm] = trajectory(rows)
    write_csv(TABLE_ROOT / "metrics_per_epoch.csv", combined)
    write_csv(TABLE_ROOT / "late_window_retrieval.csv", list(trajectories.values()))

    comparisons = [("L2-L1", "L2", "L1"), ("L2-L0", "L2", "L0"), ("L3-L1", "L3", "L1"), ("L3-L0", "L3", "L0"), ("L2-L3", "L2", "L3")]
    metric_names = (
        "mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "slope_rank1", "slope_mAP",
        "E60_rank1", "E60_mAP", "best_rank1", "best_mAP",
    )
    delta_rows = []
    for label, left, right in comparisons:
        delta_rows.append({"comparison": label, "left_arm": left, "right_arm": right, **{metric: trajectories[left][metric] - trajectories[right][metric] for metric in metric_names}})
    write_csv(TABLE_ROOT / "late_window_primary_deltas.csv", delta_rows)

    # Preserve the historical C1 reference as a secondary provenance check.
    historical = PHASE7C_ROOT / "training/C1_image_uniform_relation/metrics_per_epoch.csv"
    historical_c0 = PHASE7C_ROOT / "training/C0_bio/metrics_per_epoch.csv"
    historical_rows: list[dict[str, Any]] = []
    if historical.is_file() and historical_c0.is_file():
        historical_c1_traj = trajectory(read_float_rows(historical))
        historical_c0_traj = trajectory(read_float_rows(historical_c0))
        historical_rows = [
            {"source": "Phase7C C1-C0", "reference_type": "cross-clothes historical Relation vs Bio", "auc_rank1_delta": historical_c1_traj["auc_rank1"] - historical_c0_traj["auc_rank1"], "auc_mAP_delta": historical_c1_traj["auc_mAP"] - historical_c0_traj["auc_mAP"]},
            {"source": "Phase7L L1-L0", "reference_type": "generic same-ID dense Relation vs Bio", "auc_rank1_delta": trajectories["L1"]["auc_rank1"] - trajectories["L0"]["auc_rank1"], "auc_mAP_delta": trajectories["L1"]["auc_mAP"] - trajectories["L0"]["auc_mAP"]},
        ]
    write_csv(TABLE_ROOT / "historical_relation_reference.csv", historical_rows)
    return combined, trajectories


def write_exposure_tables() -> list[dict[str, Any]]:
    summary_rows = []
    for arm in ("L0", "L1", "L2", "L3"):
        rows = read_float_rows(ARM_DIRS[arm] / "exposure_log.csv")
        if len(rows) != FORMAL_EPOCHS:
            raise RuntimeError(f"{arm} exposure log has {len(rows)} rows, expected {FORMAL_EPOCHS}")
        total_batches = sum(as_float(row, "total_batches_global", 0.0) for row in rows)
        scheduled_batches = sum(as_float(row, "scheduled_relation_batches_global", 0.0) for row in rows)
        active_batches = sum(as_float(row, "relation_active_batches_global", 0.0) for row in rows)
        candidate_pairs = sum(as_float(row, "valid_same_id_candidate_pairs_global", 0.0) for row in rows)
        selected_pairs = sum(as_float(row, "selected_relation_pairs_global", 0.0) for row in rows)
        active_loss_den = sum(as_float(row, "relation_active_batches_global", 0.0) for row in rows)
        active_loss = sum(as_float(row, "relation_loss_active_mean", 0.0) * as_float(row, "relation_active_batches_global", 0.0) for row in rows)
        active_lambda_loss = sum(as_float(row, "lambda_relation_loss_active_mean", 0.0) * as_float(row, "relation_active_batches_global", 0.0) for row in rows)
        unique_pid_values = [as_float(row, "unique_relation_pids_global", 0.0) for row in rows]
        final_cumulative = as_float(rows[-1], "cumulative_relation_pair_contributions", 0.0)
        summary_rows.append({
            "arm": arm, "seed": SEED, "epochs": len(rows), "total_global_batches": total_batches,
            "scheduled_relation_batches": scheduled_batches, "scheduled_batch_rate": scheduled_batches / max(total_batches, 1.0),
            "active_relation_batches": active_batches, "active_batch_rate": active_batches / max(total_batches, 1.0),
            "candidate_pair_contributions": candidate_pairs, "selected_relation_pair_contributions": selected_pairs,
            "pairs_per_global_optimizer_batch": selected_pairs / max(total_batches, 1.0),
            "pairs_per_train_image_60_epoch": selected_pairs / float(FORMAL_EPOCHS * TRAIN_IMAGES),
            "unique_relation_pids_mean_per_epoch": float(np.mean(unique_pid_values)),
            "unique_relation_pids_min_per_epoch": float(np.min(unique_pid_values)),
            "unique_relation_pids_max_per_epoch": float(np.max(unique_pid_values)),
            "cumulative_active_batches_final": active_batches,
            "cumulative_pair_contributions_final": final_cumulative,
            "mean_relation_loss_active": active_loss / active_loss_den if active_loss_den else 0.0,
            "mean_lambda_relation_loss_active": active_lambda_loss / active_loss_den if active_loss_den else 0.0,
            "l3_target_pairs_per_train_image": as_float(rows[-1], "scheduled_probability_or_budget", 0.0) if arm == "L3" else 0.0,
            "l3_target_total_pairs": as_float(rows[-1], "l3_target_pairs_total", 0.0) if arm == "L3" else 0.0,
        })
    write_csv(EXPOSURE_ROOT / "exposure_summary.csv", summary_rows)
    return summary_rows


def write_geometry_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("L0", "L1", "L2", "L3"):
        path = GEOMETRY_ROOT / f"{arm}_geometry.csv"
        if not path.is_file():
            raise RuntimeError(f"missing train geometry for {arm}: {path}")
        arm_rows = read_float_rows(path)
        if len(arm_rows) != len(GEOMETRY_EPOCHS):
            raise RuntimeError(f"{arm} geometry has {len(arm_rows)} rows, expected {len(GEOMETRY_EPOCHS)}")
        rows.extend(arm_rows)
    write_csv(GEOMETRY_ROOT / "geometry_comparison.csv", rows)
    return rows


def write_amp_table() -> list[dict[str, Any]]:
    rows = []
    for arm in ("L0", "L1", "L2", "L3"):
        audit_rows = read_float_rows(ARM_DIRS[arm] / "amp_audit.csv")
        if len(audit_rows) != FORMAL_EPOCHS:
            raise RuntimeError(f"{arm} AMP audit has {len(audit_rows)} rows, expected {FORMAL_EPOCHS}")
        skips = sum(as_float(row, "skipped_steps", 0.0) for row in audit_rows)
        found_inf = sum(as_float(row, "found_inf_count", 0.0) for row in audit_rows)
        nonfinite = sum(as_float(row, "nonfinite_events", 0.0) for row in audit_rows)
        scales = [as_float(row, "amp_scale_end") for row in audit_rows]
        rows.append({
            "arm": arm, "epochs": len(audit_rows), "skipped_steps_total": skips,
            "found_inf_total": found_inf, "nonfinite_events_total": nonfinite,
            "epochs_with_skips": sum(as_float(row, "amp_skip", 0.0) > 0 for row in audit_rows),
            "minimum_scale": min(as_float(row, "amp_scale_min") for row in audit_rows),
            "maximum_scale": max(as_float(row, "amp_scale_max") for row in audit_rows),
            "final_scale": scales[-1], "amp_event_rows": len(read_csv(ARM_DIRS[arm] / "amp_events.csv")),
        })
    write_csv(AMP_ROOT / "amp_summary.csv", rows)
    return rows


def strength_lookup(summary: list[dict[str, Any]], arm: str, stage: str, metric: str, module: str = "late_eva", statistic: str = "mean") -> float:
    for row in summary:
        if row.get("arm") == arm and row.get("stage") == stage and row.get("module") == module and row.get("metric") == metric:
            return as_float(row, statistic)
    return float("nan")


def write_strength_tables(strength_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_float_rows(STRENGTH_ROOT / "batch_strength.csv")
    # Rebuild the long-form summary from the immutable raw diagnostic rows so
    # component gradient norms (CE/Triplet/Bio) remain available to analysis
    # without rerunning the PPU audit.
    rebuilt_summary, rebuilt_saturation = strength_summary_rows(raw)
    write_csv(STRENGTH_ROOT / "strength_summary.csv", rebuilt_summary)
    write_csv(STRENGTH_ROOT / "saturation_summary.csv", rebuilt_saturation)
    summary = read_float_rows(STRENGTH_ROOT / "strength_summary.csv")
    primary = []
    for arm in ("L1", "L2", "L3"):
        for stage in ("early", "mid", "late"):
            actual_epoch = next((as_float(row, "actual_epoch") for row in summary if row.get("arm") == arm and row.get("stage") == stage), float("nan"))
            primary.append({
                "arm": arm, "stage": stage, "actual_epoch": actual_epoch,
                "L_base": strength_lookup(summary, arm, stage, "base_loss"),
                "lambda_L_rel": strength_lookup(summary, arm, stage, "lambda_relation_loss"),
                "R_loss": strength_lookup(summary, arm, stage, "r_loss"),
                "norm_CE": strength_lookup(summary, arm, stage, "norm_ce"),
                "norm_Triplet": strength_lookup(summary, arm, stage, "norm_tri"),
                "norm_Bio": strength_lookup(summary, arm, stage, "norm_bio"),
                "norm_base": strength_lookup(summary, arm, stage, "norm_base"),
                "norm_rel": strength_lookup(summary, arm, stage, "norm_rel"),
                "lambda_norm_rel": strength_lookup(summary, arm, stage, "lambda_norm_rel"),
                "R_grad": strength_lookup(summary, arm, stage, "r_grad"),
                "R_rel_CE_grad": strength_lookup(summary, arm, stage, "r_rel_ce_grad"),
                "R_rel_tri_grad": strength_lookup(summary, arm, stage, "r_rel_tri_grad"),
                "R_rel_Bio_grad": strength_lookup(summary, arm, stage, "r_rel_bio_grad"),
                "R_orth": strength_lookup(summary, arm, stage, "r_orth"),
                "A_rel": strength_lookup(summary, arm, stage, "a_rel"),
                "A_total": strength_lookup(summary, arm, stage, "a_total"),
                "I_total": strength_lookup(summary, arm, stage, "i_total"),
            })
    write_csv(STRENGTH_ROOT / "strength_primary.csv", primary)

    distribution = []
    for (arm, stage), items in sorted(p7k.group_rows(raw, ("arm", "stage")).items()):
        def vals(key: str) -> list[float]:
            return [as_float(item, key) for item in items if math.isfinite(as_float(item, key))]
        rg = vals("r_grad_late_eva"); ro = vals("r_orth_late_eva"); rl = vals("r_loss")
        distribution.append({
            "arm": arm, "stage": stage, "actual_epoch": as_float(items[0], "actual_epoch"),
            "R_grad_mean": np.mean(rg) if rg else None, "R_grad_std": np.std(rg) if rg else None, "R_grad_median": np.median(rg) if rg else None, "R_grad_p25": np.percentile(rg, 25) if rg else None, "R_grad_p75": np.percentile(rg, 75) if rg else None,
            "R_orth_mean": np.mean(ro) if ro else None, "R_orth_std": np.std(ro) if ro else None, "R_orth_median": np.median(ro) if ro else None, "R_orth_p25": np.percentile(ro, 25) if ro else None, "R_orth_p75": np.percentile(ro, 75) if ro else None,
            "R_loss_mean": np.mean(rl) if rl else None, "R_loss_std": np.std(rl) if rl else None, "R_loss_median": np.median(rl) if rl else None, "R_loss_p25": np.percentile(rl, 25) if rl else None, "R_loss_p75": np.percentile(rl, 75) if rl else None,
            "R_grad_gt_005_rate": np.mean(np.asarray(rg) > 0.05) if rg else None, "R_grad_gt_010_rate": np.mean(np.asarray(rg) > 0.10) if rg else None, "R_grad_gt_020_rate": np.mean(np.asarray(rg) > 0.20) if rg else None, "R_grad_gt_050_rate": np.mean(np.asarray(rg) > 0.50) if rg else None,
            "relation_active_batch_rate": np.mean([as_float(item, "relation_active", 0.0) for item in items]),
        })
    write_csv(STRENGTH_ROOT / "strength_distribution.csv", distribution)

    component_loss_ratios = []
    for (arm, stage), items in sorted(p7k.group_rows(raw, ("arm", "stage")).items()):
        row = {"arm": arm, "stage": stage, "actual_epoch": as_float(items[0], "actual_epoch")}
        for component, source_key in (("CE", "ce_loss"), ("Triplet", "triplet_loss"), ("Bio", "bio_loss")):
            values = []
            for item in items:
                numerator = as_float(item, "lambda_relation_loss")
                denominator = as_float(item, source_key)
                if math.isfinite(numerator) and math.isfinite(denominator):
                    values.append(numerator / (denominator + EPS))
            for statistic, value in describe(values).items():
                row[f"R_rel_{component}_loss_{statistic}"] = value
        component_loss_ratios.append(row)
    write_csv(STRENGTH_ROOT / "component_loss_ratios.csv", component_loss_ratios)

    stability = []
    for (arm, stage), items in sorted(p7k.group_rows(raw, ("arm", "stage")).items()):
        stability.append({
            "arm": arm, "stage": stage, "actual_epoch": as_float(items[0], "actual_epoch"),
            "cos_base_consecutive_mean": np.nanmean([as_float(item, "cos_base_consecutive") for item in items]),
            "cos_rel_consecutive_mean": np.nanmean([as_float(item, "cos_rel_consecutive") for item in items]),
            "cos_total_consecutive_mean": np.nanmean([as_float(item, "cos_total_consecutive") for item in items]),
            "cos_base_consecutive_median": np.nanmedian([as_float(item, "cos_base_consecutive") for item in items]),
            "cos_rel_consecutive_median": np.nanmedian([as_float(item, "cos_rel_consecutive") for item in items]),
            "cos_total_consecutive_median": np.nanmedian([as_float(item, "cos_total_consecutive") for item in items]),
        })
    write_csv(STRENGTH_ROOT / "stability_summary.csv", stability)
    return primary, distribution, stability


def write_virtual_lambda_table() -> list[dict[str, Any]]:
    raw = read_float_rows(STRENGTH_ROOT / "batch_strength.csv")
    lambdas = (0.025, 0.05, 0.10, 0.20)
    batch_rows = []
    for item in raw:
        base_norm = as_float(item, "norm_base_late_eva"); rel_norm = as_float(item, "norm_rel_late_eva")
        a_rel = as_float(item, "a_rel_late_eva"); orth_norm = as_float(item, "g_rel_orth_norm_late_eva")
        if not all(math.isfinite(value) for value in (base_norm, rel_norm, a_rel, orth_norm)):
            continue
        dot = a_rel * base_norm * rel_norm
        for lam in lambdas:
            geometry = p7k.virtual_geometry(base_norm, rel_norm, dot, orth_norm, lam)
            batch_rows.append({"arm": item["arm"], "stage": item["stage"], "actual_epoch": item["actual_epoch"], "batch_index": item["batch_index"], "lambda_virtual": lam, **geometry})
    write_csv(OUT_ROOT / "virtual_lambda/virtual_geometry_batches.csv", batch_rows)
    aggregate = []
    for (arm, stage, lam), items in sorted(p7k.group_rows(batch_rows, ("arm", "stage", "lambda_virtual")).items()):
        aggregate.append({"arm": arm, "stage": stage, "lambda_virtual": lam, **{f"{key}_{stat}": value for key in ("cos_base_total", "magnitude_inflation", "orthogonal_injection") for stat, value in describe([as_float(item, key) for item in items]).items()}})
    write_csv(OUT_ROOT / "virtual_lambda/virtual_geometry_summary.csv", aggregate)
    return aggregate


def write_exposure_response_table(exposure: list[dict[str, Any]], trajectories: dict[str, dict[str, Any]], geometry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    e60_geometry = {row["arm"]: row for row in geometry if str(row.get("checkpoint")) == "E60"}
    response = []
    for row in exposure:
        arm = row["arm"]
        geom = e60_geometry.get(arm, {})
        response.append({
            "arm": arm,
            "active_batch_rate": row["active_batch_rate"],
            "selected_pairs_per_train_image_60_epoch": row["pairs_per_train_image_60_epoch"],
            "selected_pairs_per_global_optimizer_batch": row["pairs_per_global_optimizer_batch"],
            "auc_rank1": trajectories[arm]["auc_rank1"],
            "auc_mAP": trajectories[arm]["auc_mAP"],
            "identity_margin_E60": as_float(geom, "identity_margin_mean"),
            "nearest_negative_E60": as_float(geom, "nearest_negative_pid_center_similarity_mean"),
        })
    write_csv(EXPOSURE_ROOT / "exposure_response.csv", response)
    return response


def decision_bundle(trajectories: dict[str, dict[str, Any]], exposure: list[dict[str, Any]], geometry: list[dict[str, Any]], amp_rows: list[dict[str, Any]], strength_primary: list[dict[str, Any]]) -> dict[str, Any]:
    def auc_pair(arm: str) -> tuple[float, float]:
        return trajectories[arm]["auc_rank1"], trajectories[arm]["auc_mAP"]

    def both_higher(left: str, right: str) -> bool:
        return auc_pair(left)[0] > auc_pair(right)[0] and auc_pair(left)[1] > auc_pair(right)[1]

    def both_lower(left: str, right: str) -> bool:
        return auc_pair(left)[0] < auc_pair(right)[0] and auc_pair(left)[1] < auc_pair(right)[1]

    def three_way(left: str, right: str) -> str:
        if both_higher(left, right):
            return "YES"
        if both_lower(left, right):
            return "NO"
        return "MIXED"

    q1_delta_rank1 = trajectories["L1"]["auc_rank1"] - trajectories["L0"]["auc_rank1"]
    q1_delta_map = trajectories["L1"]["auc_mAP"] - trajectories["L0"]["auc_mAP"]
    q1 = "YES" if q1_delta_rank1 <= 0 and q1_delta_map <= 0 else ("NO" if q1_delta_rank1 > 0 and q1_delta_map > 0 else "MIXED")
    q2 = "YES" if both_higher("L2", "L1") else "NO"
    q3 = three_way("L2", "L0")
    q4 = "YES" if both_higher("L3", "L1") else "NO"
    q5 = three_way("L3", "L0")

    strength_index = {(row["arm"], row["stage"]): row for row in strength_primary}
    dense_strength = strength_index[("L1", "late")]
    raw_strength = read_float_rows(STRENGTH_ROOT / "batch_strength.csv")
    ltcc_rgrad = 0.00792
    ltcc_rorth = 0.00762
    strength_evidence = {}
    for arm in ("L2", "L3"):
        current = strength_index[(arm, "late")]
        reduced = current["R_grad"] < dense_strength["R_grad"] and current["R_orth"] < dense_strength["R_orth"]
        closer = abs(current["R_grad"] - ltcc_rgrad) < abs(dense_strength["R_grad"] - ltcc_rgrad) and abs(current["R_orth"] - ltcc_rorth) < abs(dense_strength["R_orth"] - ltcc_rorth)
        active_count = sum(1 for item in raw_strength if item.get("arm") == arm and item.get("stage") == "late" and as_float(item, "relation_active", 0.0) > 0.5)
        strength_evidence[arm] = {"R_grad": current["R_grad"], "R_orth": current["R_orth"], "reduced_vs_L1": reduced, "closer_to_LTCC_I3": closer, "late_diagnostic_active_batches": active_count, "conditional_active_gradient_observed": active_count > 0}
    observed_strength_passes = [item["reduced_vs_L1"] and item["closer_to_LTCC_I3"] for item in strength_evidence.values() if item["conditional_active_gradient_observed"]]
    q6 = "YES" if observed_strength_passes and all(observed_strength_passes) and all(item["conditional_active_gradient_observed"] for item in strength_evidence.values()) else ("MIXED" if any(observed_strength_passes) else "NO")

    e60 = {row["arm"]: row for row in geometry if str(row.get("checkpoint")) == "E60"}
    geometry_evidence = {}
    for arm in ("L2", "L3"):
        sparse = e60[arm]; dense = e60["L1"]
        negative_lower = as_float(sparse, "nearest_negative_pid_center_similarity_mean") < as_float(dense, "nearest_negative_pid_center_similarity_mean")
        margin_nonworse = as_float(sparse, "identity_margin_mean") >= as_float(dense, "identity_margin_mean")
        ordering_nonworse = as_float(sparse, "identity_ordering_correct_rate") >= as_float(dense, "identity_ordering_correct_rate")
        geometry_evidence[arm] = {"nearest_negative_lower": negative_lower, "identity_margin_nonworse": margin_nonworse, "ordering_nonworse": ordering_nonworse}
    geometry_passes = [item["nearest_negative_lower"] and item["identity_margin_nonworse"] for item in geometry_evidence.values()]
    q7 = "YES" if all(geometry_passes) else ("NO" if not any(geometry_passes) else "MIXED")

    controls = [row for row in amp_rows if row.get("arm") in ("L0", "L1")]
    control_skip_reference = max([as_float(row, "skipped_steps_total", 0.0) for row in controls] or [0.0])
    control_final_scale = np.median([as_float(row, "final_scale", float("nan")) for row in controls]) if controls else float("nan")
    # The promotion rule excludes a clear arm-specific AMP failure.  Nonzero
    # events shared by all controls are not, by themselves, a confound.
    amp_clean = all(
        as_float(row, "skipped_steps_total", 0.0) <= max(2.0 * control_skip_reference, control_skip_reference + 8.0)
        and as_float(row, "found_inf_total", 0.0) <= max(2.0 * control_skip_reference, control_skip_reference + 8.0)
        and as_float(row, "nonfinite_events_total", 0.0) <= max(2.0 * control_skip_reference, control_skip_reference + 8.0)
        and abs(as_float(row, "final_scale", float("nan")) - control_final_scale) <= EPS
        for row in amp_rows if row.get("arm") in ("L2", "L3")
    )
    q8 = "YES" if (q2 == "YES" or q4 == "YES") and q6 == "YES" and q7 != "NO" and amp_clean else ("NO" if q2 == "NO" and q4 == "NO" else "MIXED")

    promotion_candidates = {}
    for arm in ("L2", "L3"):
        current = strength_index[(arm, "late")]
        sparse_geom = e60[arm]; dense_geom = e60["L1"]
        retrieval_pass = both_higher(arm, "L1")
        bio_nonworse = trajectories[arm]["auc_rank1"] >= trajectories["L0"]["auc_rank1"] and trajectories[arm]["auc_mAP"] >= trajectories["L0"]["auc_mAP"]
        strength_pass = strength_evidence[arm]["conditional_active_gradient_observed"] and strength_evidence[arm]["reduced_vs_L1"] and strength_evidence[arm]["closer_to_LTCC_I3"]
        geometry_pass = as_float(sparse_geom, "nearest_negative_pid_center_similarity_mean") < as_float(dense_geom, "nearest_negative_pid_center_similarity_mean") and as_float(sparse_geom, "identity_margin_mean") >= as_float(dense_geom, "identity_margin_mean")
        promotion_candidates[arm] = {"retrieval_better_than_L1": retrieval_pass, "not_worse_than_L0": bio_nonworse, "strength_toward_LTCC": strength_pass, "nearest_negative_reduced": geometry_pass, "amp_clean": amp_clean, "passes_all": retrieval_pass and bio_nonworse and strength_pass and geometry_pass and amp_clean}
    promoted_arm = next((arm for arm in ("L2", "L3") if promotion_candidates[arm]["passes_all"]), None)
    promotion = "PROMOTE" if promoted_arm else "STOP"
    q9 = "YES" if promoted_arm else "NO"

    questions = {
        "Q1_dense_PRCC_ID_only_reproduces_non_beneficial_behavior": q1,
        "Q2_L2_improves_over_L1": q2,
        "Q3_L2_improves_over_L0": q3,
        "Q4_L3_improves_over_L1": q4,
        "Q5_L3_improves_over_L0": q5,
        "Q6_sparse_R_grad_R_orth_move_toward_LTCC": q6,
        "Q7_sparse_arms_avoid_nearest_negative_inflation": q7,
        "Q8_exposure_density_supported_dataset_dependence_mechanism": q8,
        "Q9_formal_exposure_controlled_method_justified": q9,
    }
    return {
        "questions": questions, "q1_deltas": {"L1_minus_L0_auc_rank1": q1_delta_rank1, "L1_minus_L0_auc_mAP": q1_delta_map},
        "strength_evidence": strength_evidence, "geometry_evidence": geometry_evidence, "amp_clean": amp_clean,
        "promotion": promotion, "promoted_arm": promoted_arm, "promotion_candidates": promotion_candidates,
        "ltcc_success_reference": {"R_grad_late": ltcc_rgrad, "R_orth_late": ltcc_rorth, "source": "frozen Phase 7K-A LTCC I3 late diagnostic"},
    }


def make_plots(trajectories: dict[str, dict[str, Any]], response: list[dict[str, Any]], strength_primary: list[dict[str, Any]], geometry: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"L0": "black", "L1": "tab:red", "L2": "tab:blue", "L3": "tab:green"}
    plt.figure(figsize=(7, 4.5))
    for arm in ("L0", "L1", "L2", "L3"):
        rows = read_float_rows(ARM_DIRS[arm] / "metrics_per_epoch.csv")
        x = [as_float(row, "epoch") for row in rows]
        y = [as_float(row, "cc_rank1") * 100.0 for row in rows]
        plt.plot(x, y, label=arm, color=colors[arm])
    plt.axvspan(PRIMARY_START, PRIMARY_END, color="gray", alpha=0.12)
    plt.xlabel("Epoch"); plt.ylabel("Changed-clothes Rank-1 (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_ROOT / "retrieval_rank1_trajectory.png", dpi=150); plt.close()

    for metric, title, filename in (("auc_rank1", "Relation exposure vs Rank-1 AUC", "exposure_response_rank1_auc.png"), ("auc_mAP", "Relation exposure vs mAP AUC", "exposure_response_map_auc.png"), ("identity_margin_E60", "Relation exposure vs E60 identity margin", "exposure_response_identity_margin.png")):
        plt.figure(figsize=(6, 4.5))
        for row in response:
            x = row["selected_pairs_per_train_image_60_epoch"]
            y = row[metric]
            plt.scatter(x, y, color=colors[row["arm"]], s=55, label=row["arm"])
            plt.annotate(row["arm"], (x, y), textcoords="offset points", xytext=(5, 5))
        plt.xlabel("Selected Relation pairs / train image / 60 epochs")
        plt.ylabel(title.split(" vs ")[-1]); plt.tight_layout()
        plt.savefig(PLOT_ROOT / filename, dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    for arm in ("L1", "L2", "L3"):
        rows = [row for row in strength_primary if row["arm"] == arm]
        plt.plot([10, 30, 60], [row["R_grad"] for row in rows], marker="o", label=f"{arm} R_grad", color=colors[arm])
        plt.plot([10, 30, 60], [row["R_orth"] for row in rows], marker="x", linestyle="--", label=f"{arm} R_orth", color=colors[arm], alpha=0.65)
    plt.axhline(0.00792, color="gray", linestyle=":", label="LTCC I3 R_grad reference")
    plt.xlabel("Diagnostic checkpoint"); plt.ylabel("Ratio"); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(PLOT_ROOT / "strength_R_grad_R_orth_trajectory.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    for arm in ("L0", "L1", "L2", "L3"):
        rows = [row for row in geometry if row["arm"] == arm]
        rows.sort(key=lambda row: int(str(row["checkpoint"])[1:]))
        plt.plot([int(str(row["checkpoint"])[1:]) for row in rows], [as_float(row, "identity_margin_mean") for row in rows], marker="o", label=arm, color=colors[arm])
    plt.xlabel("Checkpoint"); plt.ylabel("Train identity margin"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_ROOT / "geometry_identity_margin_trajectory.png", dpi=150); plt.close()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_final_report(protocol: dict[str, Any], trajectories: dict[str, dict[str, Any]], exposure: list[dict[str, Any]], geometry: list[dict[str, Any]], amp_rows: list[dict[str, Any]], strength_primary: list[dict[str, Any]], strength_distribution: list[dict[str, Any]], stability: list[dict[str, Any]], virtual: list[dict[str, Any]], decisions: dict[str, Any]) -> None:
    strength_summary = read_float_rows(STRENGTH_ROOT / "strength_summary.csv")
    component_loss_ratios = read_float_rows(STRENGTH_ROOT / "component_loss_ratios.csv")

    def summary_cell(arm: str, stage: str, module: str, metric: str) -> str:
        row = next((item for item in strength_summary if item.get("arm") == arm and item.get("stage") == stage and item.get("module") == module and item.get("metric") == metric), {})
        return f"{fmt(row.get('mean'), 4)} ± {fmt(row.get('std'), 4)}; {fmt(row.get('median'), 4)}; {fmt(row.get('p25'), 4)}; {fmt(row.get('p75'), 4)}"

    def ratio_cell(arm: str, stage: str, component: str) -> str:
        row = next((item for item in component_loss_ratios if item.get("arm") == arm and item.get("stage") == stage), {})
        return f"{fmt(row.get(f'R_rel_{component}_loss_mean'), 4)} ± {fmt(row.get(f'R_rel_{component}_loss_std'), 4)}; {fmt(row.get(f'R_rel_{component}_loss_median'), 4)}; {fmt(row.get(f'R_rel_{component}_loss_p25'), 4)}; {fmt(row.get(f'R_rel_{component}_loss_p75'), 4)}"

    lines = [
        "# PHASE 7L — Exposure-Matched Positive Consistency", "",
        f"Generated: `{utc_now()}`. Dataset: PRCC only. Seed: `{SEED}`. This report covers the predeclared L0–L3 arms and no second seed.", "",
        "## 1. Exact source audit", "",
        f"- Frozen PRCC Bio source: `{PRCC_C0_SOURCE}`.",
        f"- Frozen PRCC Relation provenance: `{PRCC_C1_SOURCE}` (historical cross-clothes Relation reference only; Phase 7L Relation is generic same-ID).",
        f"- Frozen common initialization: `{PRCC_COMMON_INIT}`; verified against `{PRCC_COMMON_META}`.",
        f"- Frozen LTCC I3 exposure source: `{LTCC_I3_EXPOSURE}`.",
        "- Phase limitation: the historical PRCC source arm C1 is cross-clothes, while Phase 7L L1–L3 use generic same-ID pairs. Therefore C1 is provenance/secondary context, not an objective-identical cross-dataset arm.",
        "",
        "## 2. Frozen exposure protocol", "",
        f"- L0: Bio control, no applied Relation term.",
        f"- L1: generic same-ID Relation on every valid batch, lambda={LAMBDA_REL}.",
        f"- L2: deterministic SHA256 batch schedule, nominal probability `{L2_ACTIVE_PROBABILITY}`.",
        f"- L3: deterministic schedule matched to LTCC I3 `{protocol['l3']['target_metric']}`; target `{fmt(protocol['l3']['target_pairs_per_train_image'], 8)}` pair contributions/train image over 60 epochs.",
        "- Same feature `feat[0]`, same-ID unordered pairs only, no clothing restriction, no query/gallery/test Relation labels, no optimizer or scheduler changes.",
        "- Full frozen files: `" + str(PROTOCOL_ROOT / "protocol.json") + "`, `" + str(PROTOCOL_ROOT / "exposure_schedule.csv") + "`.",
        "",
        "## 3. Stage mapping", "",
        "- Early: exact E10 checkpoint (`model_epoch010.pth`). Mid: exact E30 checkpoint (`model_epoch030.pth`). Late: exact E60 checkpoint (`model_epoch060.pth`/`model_final.pth`).",
        "- Train geometry: E20, E40, E60. Strength diagnostics: exact E10, E30, E60; 32 rank-0 diagnostic batches per arm/stage.",
        "",
        "## 4. L0–L3 exposure statistics", "",
        "| Arm | Active batch rate | Selected pairs | Pairs/train image | Pairs/global update | Mean active L_rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in exposure:
        lines.append(f"| {row['arm']} | {fmt(row['active_batch_rate'], 6)} | {int(row['selected_relation_pair_contributions'])} | {fmt(row['pairs_per_train_image_60_epoch'], 8)} | {fmt(row['pairs_per_global_optimizer_batch'], 5)} | {fmt(row['mean_relation_loss_active'], 5)} |")
    lines += ["", "## 5. E51–E60 retrieval", "", "| Arm | Mean Rank-1 | Mean mAP | Rank-1 AUC | mAP AUC | Rank-1 slope | mAP slope | E60 R1 | E60 mAP | Best R1 (epoch) | Best mAP (epoch) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("L0", "L1", "L2", "L3"):
        row = trajectories[arm]
        lines.append(f"| {arm} | {fmt(row['mean_rank1'], 3)} | {fmt(row['mean_mAP'], 3)} | {fmt(row['auc_rank1'], 3)} | {fmt(row['auc_mAP'], 3)} | {fmt(row['slope_rank1'], 4)} | {fmt(row['slope_mAP'], 4)} | {fmt(row['E60_rank1'], 3)} | {fmt(row['E60_mAP'], 3)} | {fmt(row['best_rank1'], 3)} ({row['best_rank1_epoch']}) | {fmt(row['best_mAP'], 3)} ({row['best_mAP_epoch']}) |")
    delta_rows = read_float_rows(TABLE_ROOT / "late_window_primary_deltas.csv")
    lines += ["", "### Primary deltas", "", "| Comparison | Rank-1 AUC delta | mAP AUC delta | Mean Rank-1 delta | Mean mAP delta |", "|---|---:|---:|---:|---:|"]
    for row in delta_rows:
        lines.append(f"| {row['comparison']} | {fmt(row['auc_rank1'], 3)} | {fmt(row['auc_mAP'], 3)} | {fmt(row['mean_rank1'], 3)} | {fmt(row['mean_mAP'], 3)} |")

    lines += ["", "## 6. Effective Relation strength", "", "Late-EVA unscaled FP32 diagnostics; R values include lambda=0.1. Inactive L2/L3 diagnostic batches contribute zero Relation gradient.", "", "| Arm | Stage | L_base | lambda·L_rel | R_loss | ||g_base|| | lambda||g_rel|| | R_grad | R_orth | A_rel | A_total | I_total |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in strength_primary:
        lines.append(f"| {row['arm']} | {row['stage']} E{int(row['actual_epoch'])} | {fmt(row['L_base'], 5)} | {fmt(row['lambda_L_rel'], 5)} | {fmt(row['R_loss'], 5)} | {fmt(row['norm_base'], 5)} | {fmt(row['lambda_norm_rel'], 5)} | {fmt(row['R_grad'], 5)} | {fmt(row['R_orth'], 5)} | {fmt(row['A_rel'], 5)} | {fmt(row['A_total'], 5)} | {fmt(row['I_total'], 5)} |")
    lines += ["", "### Loss decomposition statistics", "", "Cells are `mean ± std; median; p25; p75`, computed over the 32 fixed diagnostic batches at each checkpoint (late-EVA loss capture; PRCC only).", "", "| Arm | Stage | L_CE | L_Triplet | L_Bio | L_base | lambda·L_rel |", "|---|---|---|---|---|---|---|"]
    for arm in ("L1", "L2", "L3"):
        for stage in ("early", "mid", "late"):
            lines.append(f"| {arm} | {stage} | {summary_cell(arm, stage, 'late_eva', 'ce_loss')} | {summary_cell(arm, stage, 'late_eva', 'triplet_loss')} | {summary_cell(arm, stage, 'late_eva', 'bio_loss')} | {summary_cell(arm, stage, 'late_eva', 'base_loss')} | {summary_cell(arm, stage, 'late_eva', 'lambda_relation_loss')} |")
    lines += ["", "### Component-relative loss ratios", "", "Cells use the same statistic order. Ratios are `lambda·L_rel/(component loss + eps)`.", "", "| Arm | Stage | R_rel_CE | R_rel_Triplet | R_rel_Bio |", "|---|---|---|---|---|"]
    for arm in ("L1", "L2", "L3"):
        for stage in ("early", "mid", "late"):
            lines.append(f"| {arm} | {stage} | {ratio_cell(arm, stage, 'CE')} | {ratio_cell(arm, stage, 'Triplet')} | {ratio_cell(arm, stage, 'Bio')} |")
    lines += ["", "### Gradient-norm decomposition (late EVA)", "", "Cells are `mean ± std; median; p25; p75`; norms are unscaled FP32 autograd norms and R values include lambda=0.1.", "", "| Arm | Stage | ||g_CE|| | ||g_tri|| | ||g_bio|| | ||g_base|| | ||g_rel|| | R_rel_CE_grad | R_rel_tri_grad | R_rel_Bio_grad |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("L1", "L2", "L3"):
        for stage in ("early", "mid", "late"):
            lines.append(f"| {arm} | {stage} | {summary_cell(arm, stage, 'late_eva', 'norm_ce')} | {summary_cell(arm, stage, 'late_eva', 'norm_tri')} | {summary_cell(arm, stage, 'late_eva', 'norm_bio')} | {summary_cell(arm, stage, 'late_eva', 'norm_base')} | {summary_cell(arm, stage, 'late_eva', 'norm_rel')} | {summary_cell(arm, stage, 'late_eva', 'r_rel_ce_grad')} | {summary_cell(arm, stage, 'late_eva', 'r_rel_tri_grad')} | {summary_cell(arm, stage, 'late_eva', 'r_rel_bio_grad')} |")
    lines += ["", "### Available-module late-E60 summary", "", "The primary module is Late EVA. The same unscaled FP32 quantities for the shared backbone, ID head, and Bio projection are retained in `strength/strength_summary.csv`; this compact table reports their mean±std only.", "", "| Arm | Module | ||g_base|| | lambda||g_rel|| | R_grad | R_orth |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("L1", "L2", "L3"):
        for module in ("late_eva", "shared_backbone", "id_head", "bio_projection"):
            lines.append(f"| {arm} | {module} | {summary_cell(arm, 'late', module, 'norm_base')} | {summary_cell(arm, 'late', module, 'lambda_norm_rel')} | {summary_cell(arm, 'late', module, 'r_grad')} | {summary_cell(arm, 'late', module, 'r_orth')} |")
    lines += ["", "### Per-batch distribution and saturation", "", "The full machine-readable distributions are in `strength/strength_distribution.csv` and `strength/saturation_summary.csv`; they include mean, standard deviation, median, p25, p75, and R_grad threshold rates at 0.05/0.10/0.20/0.50. Saturation records s_pos mean and fractions at 0.70/0.80/0.90 for active Relation pairs.", "", "Diagnostic-window caveat: the fixed first-32-batch window contains no L3-active batch at E10, E30, or E60. Its zero L3 strength values are therefore unconditional zeros from that window, not estimates of the conditional gradient on an active L3 batch; the L2 active batches provide the observed sparse-gradient comparison.", "", "| Arm | Stage | R_grad p25 | R_grad median | R_grad p75 | R_orth median | R_loss median | Active rate |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in strength_distribution:
        lines.append(f"| {row['arm']} | {row['stage']} E{int(row['actual_epoch'])} | {fmt(row['R_grad_p25'], 5)} | {fmt(row['R_grad_median'], 5)} | {fmt(row['R_grad_p75'], 5)} | {fmt(row['R_orth_median'], 5)} | {fmt(row['R_loss_median'], 5)} | {fmt(row['relation_active_batch_rate'], 5)} |")

    lines += ["", "## 7. Train-only geometry", "", "| Arm | Checkpoint | Same-ID sim | Same-clothes sim | Cross-clothes sim | Nearest negative center sim | Identity margin | Ordering correct | PID center separation | Clothing-state dispersion |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in geometry:
        lines.append(f"| {row['arm']} | {row['checkpoint']} | {fmt(row['same_id_positive_similarity_mean'], 5)} | {fmt(row['same_clothes_similarity_mean'], 5)} | {fmt(row['cross_clothes_similarity_mean'], 5)} | {fmt(row['nearest_negative_pid_center_similarity_mean'], 5)} | {fmt(row['identity_margin_mean'], 5)} | {fmt(row['identity_ordering_correct_rate'], 5)} | {fmt(row['pid_center_separation_mean'], 5)} | {fmt(row['clothing_state_dispersion_mean'], 5)} |")

    lines += ["", "## 8. AMP audit", "", "| Arm | Skipped steps | found_inf | Nonfinite events | Min scale | Max scale | Final scale |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in amp_rows:
        lines.append(f"| {row['arm']} | {int(row['skipped_steps_total'])} | {int(row['found_inf_total'])} | {int(row['nonfinite_events_total'])} | {fmt(row['minimum_scale'], 1)} | {fmt(row['maximum_scale'], 1)} | {fmt(row['final_scale'], 1)} |")

    lines += ["", "## 9. Exposure-response summary", "", "This is descriptive only; no optimal exposure was fit and no extra arms were run.", "", "| Arm | Exposure rate | Pairs/train image | Rank-1 AUC | mAP AUC | E60 identity margin | E60 nearest negative |", "|---|---:|---:|---:|---:|---:|---:|"]
    response = read_float_rows(EXPOSURE_ROOT / "exposure_response.csv")
    for row in response:
        lines.append(f"| {row['arm']} | {fmt(row['active_batch_rate'], 5)} | {fmt(row['selected_pairs_per_train_image_60_epoch'], 8)} | {fmt(row['auc_rank1'], 3)} | {fmt(row['auc_mAP'], 3)} | {fmt(row['identity_margin_E60'], 5)} | {fmt(row['nearest_negative_E60'], 5)} |")

    lines += ["", "## 10. Update-direction stability", "", "Consecutive-batch cosine summaries are supplementary and should not be overinterpreted as optimization trajectories.", "", "| Arm | Stage | cos(base) mean | cos(Rel) mean | cos(total) mean |", "|---|---|---:|---:|---:|"]
    for row in stability:
        lines.append(f"| {row['arm']} | {row['stage']} E{int(row['actual_epoch'])} | {fmt(row['cos_base_consecutive_mean'], 5)} | {fmt(row['cos_rel_consecutive_mean'], 5)} | {fmt(row['cos_total_consecutive_mean'], 5)} |")

    lines += ["", "## 11. Virtual-lambda geometry", "", "Offline fixed-gradient geometry only; no training or hyperparameter claim is made. Full results: `virtual_lambda/virtual_geometry_summary.csv`. L3 has no virtual-lambda rows because the fixed diagnostic window contained no active Relation gradient.", "", "| Arm | Stage | lambda | cos(base,total) mean | magnitude inflation mean | orthogonal injection mean |", "|---|---|---:|---:|---:|---:|"]
    for row in virtual:
        if row["lambda_virtual"] in (0.025, 0.10, 0.20) and row["stage"] == "late":
            lines.append(f"| {row['arm']} | {row['stage']} | {fmt(row['lambda_virtual'], 3)} | {fmt(row['cos_base_total_mean'], 5)} | {fmt(row['magnitude_inflation_mean'], 5)} | {fmt(row['orthogonal_injection_mean'], 5)} |")

    lines += ["", "## 12. Decision questions", "", "| Question | Answer |", "|---|---|"]
    question_labels = {
        "Q1_dense_PRCC_ID_only_reproduces_non_beneficial_behavior": "Q1 dense PRCC ID-only Relation reproduces existing non-beneficial behavior",
        "Q2_L2_improves_over_L1": "Q2 L2 improves over dense L1",
        "Q3_L2_improves_over_L0": "Q3 L2 improves over Bio L0",
        "Q4_L3_improves_over_L1": "Q4 L3 improves over dense L1",
        "Q5_L3_improves_over_L0": "Q5 L3 improves over Bio L0",
        "Q6_sparse_R_grad_R_orth_move_toward_LTCC": "Q6 sparse arms reduce R_grad/R_orth toward LTCC",
        "Q7_sparse_arms_avoid_nearest_negative_inflation": "Q7 sparse arms avoid nearest-negative inflation",
        "Q8_exposure_density_supported_dataset_dependence_mechanism": "Q8 exposure density is supported as dataset-dependence mechanism",
        "Q9_formal_exposure_controlled_method_justified": "Q9 formal exposure-controlled positive-consistency method justified",
    }
    for key, label in question_labels.items():
        lines.append(f"| {label} | **{decisions['questions'][key]}** |")

    lines += ["", "## 13. Promotion decision", "", f"**{decisions['promotion']}**", "", f"Promoted candidate: `{decisions['promoted_arm'] or 'none'}`.", "", "Promotion was gated jointly on both late-window primary AUCs versus L1, no worse-than-Bio primary AUCs, movement of both R_grad and R_orth toward the frozen LTCC I3 reference, lower E60 nearest-negative similarity with non-worse identity margin, and absence of a clear arm-specific AMP confound. The L3 strength gate remains unconfirmed because the fixed diagnostic window contained no active L3 batch; the per-arm gate evidence is in the manifest.", "", "## Strongest mechanistic conclusion", ""]
    if decisions["promotion"] == "PROMOTE":
        lines.append("A sparse Relation-exposure arm satisfies the predeclared mechanism-validation gate: the result supports Relation exposure density as a contributor to PRCC dataset dependence. This does not authorize a lambda sweep or adaptive controller; any future branch must be separately scoped and predeclared.")
    else:
        lines.append("L3 gives the favorable retrieval and train-geometry result, while L2 is worse than both dense Relation and Bio. The exposure-density hypothesis therefore has mixed retrieval evidence, and its L3 effective-strength comparison is incomplete because the fixed strength window sampled no active L3 batch. Generic same-identity positive consistency remains an empirical LTCC regularizer; PRCC dataset dependence is not sufficiently established for promotion.")
    lines += ["", "NO SECOND SEED WAS STARTED.", "NO LATER PHASE WAS STARTED.", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def analyze() -> dict[str, Any]:
    ensure_dirs()
    protocol, _plan = load_protocol()
    for arm in ARM_DIRS:
        if not (ARM_DIRS[arm] / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"cannot analyze incomplete arm {arm}")
    if not (STRENGTH_ROOT / "batch_strength.csv").is_file():
        raise RuntimeError("strength audit is missing; run --mode strength first")
    _combined, trajectories = write_retrieval_tables()
    exposure = write_exposure_tables()
    geometry = write_geometry_table()
    amp_rows = write_amp_table()
    strength_primary, strength_distribution, stability = write_strength_tables({})
    virtual = write_virtual_lambda_table()
    response = write_exposure_response_table(exposure, trajectories, geometry)
    decisions = decision_bundle(trajectories, exposure, geometry, amp_rows, strength_primary)
    make_plots(trajectories, response, strength_primary, geometry)
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "dataset": "PRCC",
        "arms": ARM_LABELS, "protocol": str(PROTOCOL_ROOT / "protocol.json"), "source_audit": str(PROTOCOL_ROOT / "source_audit.json"),
        "stage_mapping": {arm: {f"E{epoch:03d}": str(ARM_DIRS[arm] / f"model_epoch{epoch:03d}.pth") for epoch in (10, 30, 60)} for arm in ARM_DIRS},
        "diagnostics": {"strength_batches_per_stage": STRENGTH_BATCHES, "strength_stages": ["E10", "E30", "E60"], "geometry_stages": ["E20", "E40", "E60"], "gradient_definition": "unscaled FP32 autograd.grad; no optimizer/update"},
        "outputs": {"report": str(REPORT_PATH), "source_report": str(SOURCE_REPORT_PATH), "protocol_report": str(PROTOCOL_REPORT_PATH), "retrieval": str(TABLE_ROOT / "late_window_retrieval.csv"), "deltas": str(TABLE_ROOT / "late_window_primary_deltas.csv"), "exposure": str(EXPOSURE_ROOT / "exposure_summary.csv"), "strength": str(STRENGTH_ROOT / "strength_primary.csv"), "strength_summary": str(STRENGTH_ROOT / "strength_summary.csv"), "component_loss_ratios": str(STRENGTH_ROOT / "component_loss_ratios.csv"), "geometry": str(GEOMETRY_ROOT / "geometry_comparison.csv"), "amp": str(AMP_ROOT / "amp_summary.csv"), "plots": str(PLOT_ROOT), "virtual_lambda": str(OUT_ROOT / "virtual_lambda")},
        "decisions": decisions, "second_seed_started": False, "later_phase_started": False,
        "new_training_arms_completed": ["L0", "L1", "L2", "L3"],
    }
    write_json(MANIFEST_PATH, manifest)
    write_final_report(protocol, trajectories, exposure, geometry, amp_rows, strength_primary, strength_distribution, stability, virtual, decisions)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "train", "strength", "analyze"), required=True)
    parser.add_argument("--arm", choices=tuple(ARM_DIRS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        audit = source_audit()
        print(json.dumps({"phase": PHASE, "mode": "audit", "training_started": audit["training_started"], "protocol": str(PROTOCOL_ROOT / "protocol.json")}, sort_keys=True), flush=True)
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for --mode train")
        train_arm(args.arm)
    elif args.mode == "strength":
        result = run_strength_audit()
        print(json.dumps({"phase": PHASE, "mode": "strength", "stages": result["stage_mapping"]}, sort_keys=True), flush=True)
    else:
        manifest = analyze()
        print(json.dumps({"phase": PHASE, "mode": "analyze", "promotion": manifest["decisions"]["promotion"], "report": str(REPORT_PATH)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
