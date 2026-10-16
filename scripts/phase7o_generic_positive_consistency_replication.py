#!/usr/local/bin/python3
"""Phase 7O: seed-4321 replication of Phase 7I generic ID consistency.

The training engine is the frozen Phase 6C-R full-schedule engine.  This
wrapper changes only the relation selector from the historical cross-clothes
probe to the executed Phase 7I I3 selector: the per-batch minimum feasible
budget across cross-clothes, same-clothes, and all same-ID candidates, with
deterministic SHA-256 selection of the all-same-ID candidates.

No new relation mechanism, sampler, optimizer, schedule, or architecture is
implemented here.  The only model execution in preparation is none; training
and E60 train-feature extraction require PPU and reject CPU fallback.
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
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as b6  # noqa: E402
import phase6c_full_schedule_confirmation as p6c  # noqa: E402
import phase6cr_relation_replication as p6cr  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402


PHASE = "phase7o_generic_positive_consistency_replication"
SEED = 4321
WORLD_SIZE = 2
TOTAL_EPOCHS = 60
WARMUP_EPOCHS = 20
STEPS_PER_EPOCH = 596
BATCH_SIZE = 8
NUM_INSTANCES = 2
LAMBDA_REL = 0.1
PAIR_HASH_NAMESPACE = "phase7i_relation_specificity"

OUT_ROOT = ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication"
INIT_ROOT = OUT_ROOT / "initialization"
PROTOCOL_ROOT = OUT_ROOT / "protocol"
TRAIN_ROOT = OUT_ROOT / "training"
EXPOSURE_ROOT = OUT_ROOT / "exposure"
GEOMETRY_ROOT = OUT_ROOT / "geometry"
AMP_ROOT = OUT_ROOT / "amp"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"

REPORT_PATH = ROOT / "reports/PHASE_7O_GENERIC_POSITIVE_CONSISTENCY_REPLICATION.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7O_SOURCE_AUDIT.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7O_MANIFEST.json"
RUNTIME_PATH = OUT_ROOT / "runtime_audit.json"

I7_ROOT = ROOT / "outputs/new_method_track/phase7i_relation_specificity"
I7_SCRIPT = ROOT / "scripts/phase7i_relation_specificity.py"
I7_I3_ROOT = I7_ROOT / "I3_id_only_matched"
I7_I0_ROOT = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full"
SEED4321_INIT = ROOT / "outputs/new_method_track/phase6cr_relation_replication/SEED4321_COMMON_INIT.pth"
SEED4321_INIT_META = ROOT / "outputs/new_method_track/phase6cr_relation_replication/SEED4321_COMMON_INIT_HASHES.json"

RUN_DIRS = {"R0": TRAIN_ROOT / "O0", "R1": TRAIN_ROOT / "O1"}
ARM_LABELS = {"R0": "O0 Bio Control", "R1": "O1 Generic Same-ID Positive Consistency"}

_LAST_SELECTED: dict[int, dict[str, Any]] = {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}


def ensure_dirs() -> None:
    for path in (OUT_ROOT, INIT_ROOT, PROTOCOL_ROOT, TRAIN_ROOT, EXPOSURE_ROOT, GEOMETRY_ROOT, AMP_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def patch_frozen_engine() -> None:
    """Reuse the Phase 6C-R engine with Phase 7O paths and seed metadata."""
    base.PHASE = PHASE
    base.SEED = SEED
    b6.PHASE = PHASE
    b6.SEED = SEED
    b6.TOTAL_EPOCHS = TOTAL_EPOCHS
    p6cr.PHASE = PHASE
    p6cr.SEED = SEED
    p6cr.TOTAL_EPOCHS = TOTAL_EPOCHS
    p6cr.WARMUP_EPOCHS = WARMUP_EPOCHS
    p6cr.STEPS_PER_EPOCH = STEPS_PER_EPOCH
    p6cr.BATCH_SIZE = BATCH_SIZE
    p6cr.NUM_INSTANCES = NUM_INSTANCES
    p6cr.OUT_ROOT = OUT_ROOT
    p6cr.RUN_DIRS = RUN_DIRS
    p6cr.ARM_LABELS = ARM_LABELS
    p6cr.TRAIN_ARM = {"R0": "O0", "R1": "O1"}
    p6cr.COMMON_INIT = SEED4321_INIT
    p6cr.COMMON_INIT_META = SEED4321_INIT_META
    p6cr.GEOMETRY_EPOCHS = ()


patch_frozen_engine()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_ppu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 7O requires exactly {WORLD_SIZE} visible PPUs; CPU fallback is forbidden")


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {key: sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()) for key, value in sorted(state.items())}


def common_init_audit() -> dict[str, Any]:
    if not SEED4321_INIT.is_file() or not SEED4321_INIT_META.is_file():
        raise FileNotFoundError("frozen Phase 6C-R seed4321 common initialization is missing")
    meta = read_json(SEED4321_INIT_META)
    state = torch.load(SEED4321_INIT, map_location="cpu", weights_only=False)
    state_hash = sha256_file(SEED4321_INIT)
    hashes = tensor_hashes(state)
    audit = {
        "source_checkpoint": SEED4321_INIT,
        "source_checkpoint_sha256": state_hash,
        "source_metadata": SEED4321_INIT_META,
        "source_metadata_sha256": sha256_file(SEED4321_INIT_META),
        "seed": SEED,
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "state_sha256_matches_metadata": state_hash == meta.get("state_sha256"),
        "tensor_hashes_match_metadata": hashes == meta.get("tensor_hashes"),
        "strict_hash_match": state_hash == meta.get("state_sha256") and hashes == meta.get("tensor_hashes"),
        "reused_from": "Phase 6C-R seed4321 common-init protocol; no new initialization training",
    }
    write_json(INIT_ROOT / "common_init_audit.json", audit)
    write_json(INIT_ROOT / "common_init_tensor_hashes.json", hashes)
    del state
    gc.collect()
    if not audit["strict_hash_match"]:
        raise RuntimeError("seed4321 common initialization hash audit failed")
    return audit


def source_file_records() -> dict[str, Any]:
    paths = [
        I7_SCRIPT,
        ROOT / "scripts/phase6b_soft_relation_screen.py",
        ROOT / "scripts/phase6c_full_schedule_confirmation.py",
        ROOT / "scripts/phase6cr_relation_replication.py",
        I7_ROOT / "PHASE7I_SOURCE_AUDIT.json",
        ROOT / "reports/PHASE_7I_SOURCE_AUDIT.md",
        I7_I3_ROOT / "run_manifest.json",
        I7_I3_ROOT / "config.yml",
        I7_I3_ROOT / "metrics_per_epoch.csv",
        I7_I3_ROOT / "relation_exposure.csv",
        I7_I3_ROOT / "pair_budget_per_epoch.csv",
        I7_I3_ROOT / "model_final.pth",
        I7_I0_ROOT / "model_final.pth",
        I7_I0_ROOT / "metrics_per_epoch.csv",
        ROOT / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth",
    ]
    return {str(path): file_record(path) for path in paths}


def candidate_lists(entries: list[dict[str, Any]], path_to_index: dict[str, int], bank: dict[str, Any]) -> dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]]:
    result: dict[str, list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]] = {"cross": [], "same": [], "id": []}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if int(entries[i]["pid"]) != int(entries[j]["pid"]):
                continue
            ia = int(path_to_index[str(entries[i]["image_path"])])
            ib = int(path_to_index[str(entries[j]["image_path"])])
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
    def key(item: tuple[int, int, tuple[int, int], tuple[int, int, int]]) -> str:
        i, j, image_key, _state_key = item
        token = f"{PAIR_HASH_NAMESPACE}|{SEED}|{epoch}|{rank}|{iteration}|{kind}|{image_key[0]}|{image_key[1]}|{i}|{j}".encode()
        return hashlib.sha256(token).hexdigest()
    return sorted(candidates, key=key)[:count]


def select_i3_pairs(entries: list[dict[str, Any]], path_to_index: dict[str, int], bank: dict[str, Any], epoch: int, rank: int, iteration: int) -> tuple[list[tuple[int, int, tuple[int, int], tuple[int, int, int]]], dict[str, Any]]:
    candidates = candidate_lists(entries, path_to_index, bank)
    count = min(len(candidates["cross"]), len(candidates["same"]), len(candidates["id"]))
    selected = stable_selection(candidates["id"], count, epoch, rank, iteration, "id")
    info = {
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "matched_count": count,
        "pair_count": len(selected),
        "selected_count": len(selected),
        "active": bool(selected),
        "keys": [item[3] for item in selected],
        "weights": [1.0] * len(selected),
        "selected_pairs": selected,
        "selected_kind": "id",
        "pid_counts": dict(Counter(item[3][0] for item in selected)),
        "same_clothes_selected": sum(int(item[3][1] == item[3][2]) for item in selected),
        "cross_clothes_selected": sum(int(item[3][1] != item[3][2]) for item in selected),
    }
    return selected, info


def _pair_audit_sets(selected: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]], entries: list[dict[str, Any]]) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    exact: set[tuple[int, int]] = set()
    states: set[tuple[int, int, int]] = set()
    pids: set[tuple[int, int]] = set()
    clothes: set[tuple[int, int]] = set()
    for i, j, image_key, state_key in selected:
        exact.add(image_key); states.add(state_key); pids.add((state_key[0], 0))
        clothes.add((state_key[0], int(entries[i]["clothes_id"])))
        clothes.add((state_key[0], int(entries[j]["clothes_id"])))
    return exact, states, pids, clothes


def generic_forward_batch(model, loss_fn, batch, device: torch.device, bank: dict[str, Any], path_to_index: dict[str, int], arm: str, global_step: int) -> dict[str, Any]:
    writer = base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    entries = [{"pid": int(pids[index].detach().cpu()), "clothes_id": int(clothes[index].detach().cpu()), "image_path": str(batch["image_path"][index])} for index in range(len(pids))]
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("LOCAL_RANK", "0"))
    epoch = (global_step - 1) // STEPS_PER_EPOCH + 1
    iteration = (global_step - 1) % STEPS_PER_EPOCH + 1
    representation = F.normalize(feat[0].float(), dim=-1)
    selected: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]] = []
    if arm == "O1":
        selected, info = select_i3_pairs(entries, path_to_index, bank, epoch, rank, iteration)
        if selected:
            relation = torch.stack([1.0 - torch.sum(representation[i] * representation[j]) for i, j, _image, _state in selected]).mean()
        else:
            relation = representation.sum() * 0.0
    else:
        relation = representation.sum() * 0.0
        info = {"candidate_counts": {"cross": 0, "same": 0, "id": 0}, "matched_count": 0, "pair_count": 0, "selected_count": 0, "active": False, "keys": [], "weights": [], "selected_pairs": [], "selected_kind": "none", "pid_counts": {}, "same_clothes_selected": 0, "cross_clothes_selected": 0}
    _LAST_SELECTED[rank] = {"entries": entries, "selected": selected}
    total = base_total + LAMBDA_REL * relation
    components = getattr(loss_fn, "last_components", {})
    info["relation_loss"] = relation
    return {
        "total": total,
        "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "relation_raw": relation,
        "relation_applied": relation if arm == "O1" else relation * 0.0,
        "relation_info": info,
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        "writer": writer,
    }


def generic_relation_audit_keys(batch: dict[str, Any], bank: dict[str, Any], path_to_index: dict[str, int]):
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("LOCAL_RANK", "0"))
    item = _LAST_SELECTED.get(rank, {"entries": [], "selected": []})
    return _pair_audit_sets(item["selected"], item["entries"])


def patch_training_engine() -> None:
    b6.forward_batch = generic_forward_batch
    p6cr.relation_audit_keys = generic_relation_audit_keys


def load_runtime() -> tuple[Any, Any, dict[str, Any]]:
    cfg = p6cr.configure()
    dataset = base.build_dataset(cfg)
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    return cfg, dataset, bank


def phase7i_reference_exposure() -> dict[str, Any]:
    rows = read_csv(I7_I3_ROOT / "relation_exposure.csv")
    total_batches = sum(int(row["optimizer_updates"]) * WORLD_SIZE for row in rows)
    total_pairs = sum(int(row["matched_relation_pairs"]) for row in rows)
    active = sum(int(row["relation_active_batches"]) for row in rows)
    return {
        "source": I7_I3_ROOT / "relation_exposure.csv",
        "epochs": len(rows),
        "total_global_batches": total_batches,
        "total_relation_pairs": total_pairs,
        "active_batches": active,
        "active_rate": active / max(total_batches, 1),
        "candidate_logic": "min(cross-clothes candidate count, same-clothes candidate count, all same-ID candidate count); select all same-ID candidates",
    }


def dry_run_schedule(cfg: Any, dataset: Any, bank: dict[str, Any]) -> dict[str, Any]:
    transform_train, _ = base.build_img_transforms(cfg)
    active_state = b6.full_active_state(dataset)
    index_map = base.build_index_map(dataset)
    epoch_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    pid_counts: Counter[int] = Counter()
    for epoch in range(1, TOTAL_EPOCHS + 1):
        global_count = Counter()
        global_pid = Counter()
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, epoch, rank)
            local = Counter()
            local_pid: Counter[int] = Counter()
            for iteration, batch_indices in enumerate(loader.batch_sampler, start=1):
                entries = [loader.dataset.samples[index][0] for index in batch_indices]
                candidates = candidate_lists(entries, bank["path_to_index"], bank)
                selected, info = select_i3_pairs(entries, bank["path_to_index"], bank, epoch, rank, iteration)
                for key, values in candidates.items():
                    local[f"candidate_{key}_pairs"] += len(values)
                local["matched_relation_pairs"] += len(selected)
                local["relation_active_batches"] += int(bool(selected))
                local["same_clothes_pairs"] += info["same_clothes_selected"]
                local["cross_clothes_pairs"] += info["cross_clothes_selected"]
                for state_key in info["keys"]:
                    local_pid[int(state_key[0])] += 1
            rank_rows.append({"epoch": epoch, "rank": rank, "batches": meta["batches"], "plan_sha256": meta["plan_sha256"], **local, "pid_coverage": len(local_pid)})
            global_count.update(local); global_pid.update(local_pid); pid_counts.update(local_pid)
            del loader
        global_batches = STEPS_PER_EPOCH * WORLD_SIZE
        epoch_rows.append({
            "epoch": epoch,
            "global_batches": global_batches,
            "candidate_cross_pairs": global_count["candidate_cross_pairs"],
            "candidate_same_pairs": global_count["candidate_same_pairs"],
            "candidate_id_pairs": global_count["candidate_id_pairs"],
            "selected_relation_pairs": global_count["matched_relation_pairs"],
            "relation_active_batches": global_count["relation_active_batches"],
            "active_batch_rate": global_count["relation_active_batches"] / global_batches,
            "same_clothes_pairs": global_count["same_clothes_pairs"],
            "cross_clothes_pairs": global_count["cross_clothes_pairs"],
            "same_clothes_fraction": global_count["same_clothes_pairs"] / max(global_count["matched_relation_pairs"], 1),
            "cross_clothes_fraction": global_count["cross_clothes_pairs"] / max(global_count["matched_relation_pairs"], 1),
            "pid_coverage": len(global_pid),
        })
    summary = {
        "seed": SEED,
        "epochs": TOTAL_EPOCHS,
        "global_batches": TOTAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE,
        "expected_active_batches": sum(row["relation_active_batches"] for row in epoch_rows),
        "expected_active_rate": sum(row["relation_active_batches"] for row in epoch_rows) / (TOTAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE),
        "expected_relation_contributions": sum(row["selected_relation_pairs"] for row in epoch_rows),
        "same_clothes_relation_pairs": sum(row["same_clothes_pairs"] for row in epoch_rows),
        "cross_clothes_relation_pairs": sum(row["cross_clothes_pairs"] for row in epoch_rows),
        "same_clothes_fraction": sum(row["same_clothes_pairs"] for row in epoch_rows) / max(sum(row["selected_relation_pairs"] for row in epoch_rows), 1),
        "cross_clothes_fraction": sum(row["cross_clothes_pairs"] for row in epoch_rows) / max(sum(row["selected_relation_pairs"] for row in epoch_rows), 1),
        "pid_coverage": len(pid_counts),
        "selection_logic": "Phase 7I I3 exact: count=min(cross,same,id), deterministic id selection",
        "hash_namespace": PAIR_HASH_NAMESPACE,
    }
    write_csv(PROTOCOL_ROOT / "dry_run_by_epoch.csv", epoch_rows)
    write_csv(PROTOCOL_ROOT / "dry_run_by_rank.csv", rank_rows)
    write_csv(PROTOCOL_ROOT / "dry_run_per_pid.csv", [{"pid": pid, "selected_relation_pairs": count} for pid, count in sorted(pid_counts.items())])
    write_json(PROTOCOL_ROOT / "dry_run_summary.json", summary)
    return {"summary": summary, "epoch_rows": epoch_rows, "rank_rows": rank_rows, "pid_counts": pid_counts}


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    cfg, dataset, bank = load_runtime()
    init = common_init_audit()
    dry = dry_run_schedule(cfg, dataset, bank)
    reference = phase7i_reference_exposure()
    i3_manifest = read_json(I7_I3_ROOT / "run_manifest.json")
    source = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "training_started": False,
        "new_mechanism_started": False,
        "phase7i_sources": {
            "source_report": ROOT / "reports/PHASE_7I_SOURCE_AUDIT.md",
            "source_audit_json": I7_ROOT / "PHASE7I_SOURCE_AUDIT.json",
            "I0": {"run": I7_I0_ROOT, "checkpoint": I7_I0_ROOT / "model_final.pth", "metrics": I7_I0_ROOT / "metrics_per_epoch.csv"},
            "I3": {"run": I7_I3_ROOT, "checkpoint": I7_I3_ROOT / "model_final.pth", "metrics": I7_I3_ROOT / "metrics_per_epoch.csv", "exposure": I7_I3_ROOT / "relation_exposure.csv"},
        },
        "source_files": source_file_records(),
        "protocol": {
            "dataset": "LTCC",
            "train_images": len(dataset.train),
            "train_pids": dataset.num_train_pids,
            "train_clothing_states": sum(len(v) for v in b6.full_active_state(dataset).values()),
            "batch_size": BATCH_SIZE,
            "num_instances": NUM_INSTANCES,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "epochs": TOTAL_EPOCHS,
            "lambda_rel": LAMBDA_REL,
            "feature": "normalized feat[0]",
            "relation_formula": "mean(1-cos(z_i,z_j))",
            "eligibility": "same PID, distinct image indices",
            "pair_budget": "count=min(cross-clothes candidate count, same-clothes candidate count, all same-ID candidate count)",
            "pair_selection": "deterministic SHA-256 ordering of all same-ID candidates",
            "pair_hash_namespace": PAIR_HASH_NAMESPACE,
            "optimizer": p6cr.optimizer_config(cfg),
            "scheduler": p6cr.scheduler_config(cfg),
            "amp": p6cr.amp_config(),
            "sampler": "Phase 6C/Phase 7I LTCC sampler",
            "evaluation": "frozen Phase 6C LTCC evaluation protocol, every epoch",
        },
        "common_initialization": init,
        "phase7i_i3_reference_exposure": reference,
        "seed4321_dry_run": dry["summary"],
        "checks": {
            "all_source_files_present": all(item["exists"] for item in source_file_records().values()),
            "common_init_strict_hash_match": init["strict_hash_match"],
            "dry_run_schedule_frozen_before_training": True,
            "no_query_gallery_test_in_headcount": True,
            "no_cpu_fallback": True,
            "only_O0_O1_allowed": True,
        },
    }
    write_json(PROTOCOL_ROOT / "protocol.json", source)
    write_json(OUT_ROOT / "source_audit.json", source)
    lines = [
        "# PHASE 7O source audit", "", f"Generated: `{source['created_utc']}`. Training started: `{source['training_started']}`.", "",
        "## Exact Phase 7I sources", "",
        f"- Phase 7I implementation: `{I7_SCRIPT}` SHA256 `{sha256_file(I7_SCRIPT)}`.",
        f"- Phase 7I I0 frozen checkpoint: `{I7_I0_ROOT / 'model_final.pth'}` SHA256 `{sha256_file(I7_I0_ROOT / 'model_final.pth')}`.",
        f"- Phase 7I I3 frozen checkpoint: `{I7_I3_ROOT / 'model_final.pth'}` SHA256 `{sha256_file(I7_I3_ROOT / 'model_final.pth')}`.",
        f"- Phase 7I I3 manifest: `{I7_I3_ROOT / 'run_manifest.json'}`.",
        "",
        "## Frozen protocol", "",
        f"- Sampler: Phase 7I/Phase 6C LTCC sampler; batch `{BATCH_SIZE}`, NUM_INSTANCES `{NUM_INSTANCES}`, `{STEPS_PER_EPOCH}` updates/rank/epoch, 60 epochs.",
        f"- Optimizer: `{json.dumps(p6cr.optimizer_config(cfg), sort_keys=True)}`.",
        f"- Scheduler: `{json.dumps(p6cr.scheduler_config(cfg), sort_keys=True)}`.",
        f"- AMP: `{json.dumps(p6cr.amp_config(), sort_keys=True)}`.",
        f"- Relation: normalized `feat[0]`, `mean(1-cos)`, lambda `{LAMBDA_REL}`.",
        "- I3 eligibility: same PID and distinct image indices; no clothing restriction.",
        "- I3 pair budget: `min(cross, same, id)` per batch; deterministic SHA-256 selection from the all-same-ID set.",
        "",
        "## Seed4321 dry-run exposure", "",
        f"- Active batches: `{dry['summary']['expected_active_batches']}` / `{dry['summary']['global_batches']}` (`{dry['summary']['expected_active_rate']:.8f}`).",
        f"- Expected relation contributions: `{dry['summary']['expected_relation_contributions']}`.",
        f"- Same-clothes fraction: `{dry['summary']['same_clothes_fraction']:.8f}`; cross-clothes fraction: `{dry['summary']['cross_clothes_fraction']:.8f}`.",
        f"- PID coverage: `{dry['summary']['pid_coverage']}`.",
        f"- Phase 7I seed1234 I3 reference active rate: `{reference['active_rate']:.8f}`; contributions: `{reference['total_relation_pairs']}`.",
        "",
        "## Initialization", "",
        f"- Common initialization: `{init['source_checkpoint']}`.",
        f"- SHA256: `{init['source_checkpoint_sha256']}`; strict tensor-hash match: `{init['strict_hash_match']}`.",
        "",
        "Machine-readable audit: `outputs/new_method_track/phase7o_generic_positive_consistency_replication/source_audit.json`.",
    ]
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "PREPARE.json", {"phase": PHASE, "seed": SEED, "prepared_utc": utc_now(), "protocol": PROTOCOL_ROOT / "protocol.json", "allowed_internal_arms": ["R0", "R1"], "public_arms": ["O0", "O1"], "training_started": False})
    print(json.dumps({"source_audit": str(SOURCE_REPORT_PATH), "dry_run": dry["summary"], "training_started": False}, indent=2, sort_keys=True))
    return source


def patch_run_manifest(internal_arm: str) -> None:
    run_dir = RUN_DIRS[internal_arm]
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return
    manifest = read_json(path)
    public = "O0" if internal_arm == "R0" else "O1"
    manifest.update({
        "phase": PHASE,
        "arm": public,
        "public_arm": public,
        "label": ARM_LABELS[internal_arm],
        "executed_training_arm": public,
        "relation_loss": "not applied" if public == "O0" else "generic same-PID positive consistency; mean(1-cos(normalized feat[0]))",
        "relation_eligibility": "none" if public == "O0" else "same PID, distinct image indices",
        "pair_budget": "none" if public == "O0" else "Phase 7I I3 exact min(cross,same,id) budget with deterministic id selection",
        "relation_feature": "normalized feat[0]",
        "lambda_rel": 0.0 if public == "O0" else LAMBDA_REL,
        "only_method_difference": "O1 adds the frozen Phase 7I I3 generic same-ID relation term; O0 does not.",
    })
    write_json(path, manifest)
    complete_path = run_dir / "RUN_COMPLETE.json"
    if complete_path.is_file():
        complete = read_json(complete_path); complete.update({"phase": PHASE, "arm": public}); write_json(complete_path, complete)


def train(internal_arm: str) -> None:
    if internal_arm not in ("R0", "R1"):
        raise ValueError(internal_arm)
    if not (OUT_ROOT / "PREPARE.json").is_file():
        raise RuntimeError("run --mode audit first; the dry-run protocol must be frozen before training")
    patch_training_engine()
    p6cr.train_arm(internal_arm)
    if os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) == "0":
        patch_run_manifest(internal_arm)


def trajectory(rows: list[dict[str, str]], start: int = 51, end: int = 60) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) * 100.0 for row in selected])
    mp = np.asarray([float(row["cc_mAP"]) * 100.0 for row in selected])
    auc = lambda values: float(np.trapezoid(values, x) / 9.0) if hasattr(np, "trapezoid") else float(np.trapz(values, x) / 9.0)
    br = max(selected, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    bm = max(selected, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {"mean_rank1": float(r1.mean()), "mean_mAP": float(mp.mean()), "auc_rank1": auc(r1), "auc_mAP": auc(mp), "slope_rank1": float(np.polyfit(x, r1, 1)[0]), "slope_mAP": float(np.polyfit(x, mp, 1)[0]), "e60_rank1": float(r1[-1]), "e60_mAP": float(mp[-1]), "best_rank1": float(br["cc_rank1"]) * 100.0, "best_rank1_epoch": int(br["epoch"]), "best_mAP": float(bm["cc_mAP"]) * 100.0, "best_mAP_epoch": int(bm["epoch"])}


def load_e60_model(cfg: Any, dataset: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=dataset.num_train_pids, camera_num=dataset.num_camera)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval(); model.requires_grad_(False)
    return model


def extract_feature_bank(arm: str, cfg: Any, dataset: Any, device: torch.device) -> dict[str, Any]:
    checkpoint = RUN_DIRS["R0" if arm == "O0" else "R1"] / "model_final.pth"
    _, transform_test = base.build_img_transforms(cfg)
    loader = DataLoader(ImageDataset(dataset.train, transform=transform_test), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    model = load_e60_model(cfg, dataset, checkpoint, device)
    chunks: list[np.ndarray] = []
    forward_devices: set[str] = set()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            if torch.is_tensor(output):
                identity = output
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                _score, feat = output
                identity = feat[0] if isinstance(feat, (tuple, list)) else feat
            else:
                raise TypeError(f"unexpected frozen eval model output: {type(output)!r}")
            identity = F.normalize(identity.float(), dim=-1)
            forward_devices.add(str(identity.device))
            chunks.append(identity.cpu().numpy().astype(np.float32))
    features = np.concatenate(chunks, axis=0)
    bank = {"features": features, "pid": np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64), "clothes": np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64), "image_path": np.asarray([str(item["image_path"]) for item in dataset.train])}
    np.savez_compressed(GEOMETRY_ROOT / f"{arm}_train_features.npz", **bank)
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    return {"arm": arm, "checkpoint": checkpoint, "train_images": len(features), "forward_devices": sorted(forward_devices), "device": str(device), "cpu_fallback": False}


def geometry_audit() -> dict[str, Any]:
    require_ppu(); cfg, dataset, _bank = load_runtime(); device = torch.device("cuda:0")
    runtime_rows = [extract_feature_bank("O0", cfg, dataset, device), extract_feature_bank("O1", cfg, dataset, device)]
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase7n_a_headroom_benefit_audit as p7n  # noqa: E402
    summaries: dict[str, Any] = {}
    for arm in ("O0", "O1"):
        data = np.load(GEOMETRY_ROOT / f"{arm}_train_features.npz", allow_pickle=False)
        bank = {"features": data["features"], "pid": data["pid"], "clothes": data["clothes"], "image_path": data["image_path"].astype(str)}
        geom = p7n.compute_geometry(bank)
        summaries[arm] = geom["summary"]
        write_json(GEOMETRY_ROOT / f"{arm}_E60_geometry.json", geom["summary"])
    keys = ["same_pid_same_clothes_similarity", "same_pid_cross_clothes_similarity", "nearest_negative_pid_center_similarity", "identity_margin", "ordering_correctness", "within_pid_dispersion", "clothing_state_center_dispersion"]
    delta = {key: float(summaries["O1"][key] - summaries["O0"][key]) for key in keys}
    write_csv(GEOMETRY_ROOT / "E60_geometry.csv", [{"arm": arm, **{key: summaries[arm][key] for key in keys}} for arm in ("O0", "O1")])
    write_json(GEOMETRY_ROOT / "E60_geometry_delta.json", delta)
    write_json(RUNTIME_PATH, {"phase": PHASE, "model_execution": "PPU-only frozen E60 train feature extraction", "device_count": torch.cuda.device_count(), "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "sources": runtime_rows, "cpu_fallback": False})
    return {"summaries": summaries, "delta_O1_minus_O0": delta, "runtime": runtime_rows}


def exposure_audit() -> dict[str, Any]:
    dry = read_json(PROTOCOL_ROOT / "dry_run_summary.json")
    expected = {int(row["epoch"]): row for row in read_csv(PROTOCOL_ROOT / "dry_run_by_epoch.csv")}
    actual_rows = read_csv(RUN_DIRS["R1"] / "relation_exposure.csv")
    verified: list[dict[str, Any]] = []
    for row in actual_rows:
        epoch = int(row["epoch"]); exp = expected[epoch]
        verified.append({"epoch": epoch, "expected_pairs": int(exp["selected_relation_pairs"]), "actual_pairs": int(row["relation_pairs"]), "expected_active_batches": int(exp["relation_active_batches"]), "actual_active_batches": int(row["relation_active_batches"]), "expected_active_rate": float(exp["active_batch_rate"]), "actual_active_rate": float(row["relation_active_batch_rate"]), "same_clothes_pairs": int(exp["same_clothes_pairs"]), "cross_clothes_pairs": int(exp["cross_clothes_pairs"]), "same_clothes_fraction": float(exp["same_clothes_fraction"]), "cross_clothes_fraction": float(exp["cross_clothes_fraction"]), "matches": int(exp["selected_relation_pairs"]) == int(row["relation_pairs"]) and int(exp["relation_active_batches"]) == int(row["relation_active_batches"])})
    pid_rows = read_csv(PROTOCOL_ROOT / "dry_run_per_pid.csv")
    summary = {**dry, "actual_training_rows": len(actual_rows), "actual_matches_dry_run": all(bool(row["matches"]) for row in verified), "actual_total_pairs": sum(int(row["actual_pairs"]) for row in verified), "actual_active_batches": sum(int(row["actual_active_batches"]) for row in verified)}
    write_csv(EXPOSURE_ROOT / "verified_by_epoch.csv", verified)
    write_csv(EXPOSURE_ROOT / "per_pid_contributions.csv", pid_rows)
    write_json(EXPOSURE_ROOT / "exposure_summary.json", summary)
    return summary


def amp_audit() -> list[dict[str, Any]]:
    rows = []
    for internal, public in (("R0", "O0"), ("R1", "O1")):
        metrics = read_csv(RUN_DIRS[internal] / "amp_statistics.csv")
        events = read_csv(RUN_DIRS[internal] / "amp_skip_events.csv")
        rows.append({"arm": public, "total_skipped_steps": sum(int(row["skipped_steps"]) for row in metrics), "found_inf": sum(int(row["found_inf_count"]) for row in metrics), "nonfinite_events": sum(int(row["nonfinite_events"]) for row in metrics), "scale_min": min(float(row["amp_scale_min"]) for row in metrics), "scale_max": max(float(row["amp_scale_max"]) for row in metrics), "final_scale": float(metrics[-1]["amp_scale_end"]), "event_rows": len(events)})
    write_csv(AMP_ROOT / "amp_audit.csv", rows)
    return rows


def ppu_probe(steps: int, do_optimizer_step: bool) -> None:
    """Probe the first frozen O1 batches without optimizer updates.

    This is a runtime diagnostic only.  It uses the same model, initialization,
    loader, selector, AMP boundary, and loss as O1, but deliberately performs
    no optimizer.step() and writes no training result.
    """
    require_ppu()
    patch_training_engine()
    rank, device = b6.init_dist()
    try:
        cfg = p6cr.configure()
        set_seed(SEED)
        state, _init_audit = p6cr.load_common_init()
        model, loss_fn, optimizer, _scheduler, scaler = base.setup_runtime(cfg, state, device)
        dataset = base.build_dataset(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        transform_train, _ = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        active_state = b6.full_active_state(dataset)
        bank = b6.parse_relation_bank(dataset, cfg)
        loader, _meta = base.make_epoch_loader(dataset, transform_train, active_state, index_map, 1, rank)
        iterator = iter(loader)
        print(json.dumps({"probe": "ready", "rank": rank, "device": str(device)}, sort_keys=True), flush=True)
        for iteration in range(1, steps + 1):
            batch = next(iterator)
            global_step = iteration
            optimizer.zero_grad()
            if iteration <= 4 or iteration % 10 == 0 or iteration == steps:
                print(json.dumps({"probe": "before_forward", "rank": rank, "iteration": iteration}, sort_keys=True), flush=True)
            result = generic_forward_batch(model, loss_fn, batch, device, bank, path_to_index, "O1", global_step)
            if iteration <= 4 or iteration % 10 == 0 or iteration == steps:
                print(json.dumps({"probe": "after_forward", "rank": rank, "iteration": iteration, "selected": result["relation_info"]["selected_count"], "loss_finite": bool(torch.isfinite(result["total"].detach()).item())}, sort_keys=True), flush=True)
            scaler.scale(result["total"]).backward()
            if iteration <= 4 or iteration % 10 == 0 or iteration == steps:
                print(json.dumps({"probe": "after_backward", "rank": rank, "iteration": iteration}, sort_keys=True), flush=True)
            if do_optimizer_step:
                scaler.step(optimizer)
                scaler.update()
                if iteration <= 4 or iteration % 10 == 0 or iteration == steps:
                    print(json.dumps({"probe": "after_optimizer", "rank": rank, "iteration": iteration}, sort_keys=True), flush=True)
        print(json.dumps({"probe": "complete", "rank": rank}, sort_keys=True), flush=True)
    finally:
        b6.cleanup_dist()


def matched_training_audit() -> dict[str, Any]:
    manifests = {internal: read_json(RUN_DIRS[internal] / "run_manifest.json") for internal in ("R0", "R1")}
    initial_equal = manifests["R0"].get("initial_parameter_hash") == manifests["R1"].get("initial_parameter_hash")
    init_equal = manifests["R0"].get("common_init", {}).get("sha256") == manifests["R1"].get("common_init", {}).get("sha256")
    r0 = read_csv(RUN_DIRS["R0"] / "sampler_plan.csv"); r1 = read_csv(RUN_DIRS["R1"] / "sampler_plan.csv")
    fields = [key for key in r0[0] if key != "arm"]
    sampler_equal = [{key: row[key] for key in fields} for row in r0] == [{key: row[key] for key in fields} for row in r1]
    audit = {"initial_parameter_hash_equal": initial_equal, "common_init_hash_equal": init_equal, "sampler_plan_exact_equal": sampler_equal, "sampler_plan_rows": len(r0), "only_difference": "O1 adds generic same-ID positive consistency"}
    write_json(TABLE_ROOT / "matched_training_audit.json", audit)
    if not all((initial_equal, init_equal, sampler_equal)):
        raise RuntimeError(f"matched training audit failed: {audit}")
    return audit


def plots(summary_rows: list[dict[str, Any]], geometry: dict[str, Any], exposure: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, label in (("cc_rank1", "Changed-clothes Rank-1"), ("cc_mAP", "Changed-clothes mAP")):
        plt.figure(figsize=(7, 4))
        for internal, public in (("R0", "O0"), ("R1", "O1")):
            rows = read_csv(RUN_DIRS[internal] / "metrics_per_epoch.csv")
            plt.plot([int(row["epoch"]) for row in rows], [float(row[metric]) * 100 for row in rows], label=public)
        plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.tight_layout(); plt.savefig(PLOT_ROOT / f"{metric}_trajectory.png", dpi=150); plt.close()
    delta = geometry["delta_O1_minus_O0"]
    plt.figure(figsize=(8, 4)); plt.bar(list(delta), list(delta.values())); plt.axhline(0, color="black", linewidth=0.8); plt.xticks(rotation=30, ha="right"); plt.ylabel("O1 − O0"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "E60_geometry_delta.png", dpi=150); plt.close()
    rows = read_csv(EXPOSURE_ROOT / "verified_by_epoch.csv")
    plt.figure(figsize=(7, 4)); plt.plot([int(row["epoch"]) for row in rows], [float(row["actual_active_rate"]) for row in rows]); plt.xlabel("Epoch"); plt.ylabel("Relation-active batch rate"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "relation_active_rate.png", dpi=150); plt.close()


def analyze() -> dict[str, Any]:
    for internal in ("R0", "R1"):
        marker = RUN_DIRS[internal] / "RUN_COMPLETE.json"
        if not marker.is_file() or read_json(marker).get("epochs_completed") != TOTAL_EPOCHS:
            raise RuntimeError(f"incomplete {internal} replication arm")
    matched = matched_training_audit()
    exposure = exposure_audit()
    amp_rows = amp_audit()
    geometry = geometry_audit()
    summaries = []
    for internal, public in (("R0", "O0"), ("R1", "O1")):
        summaries.append({"arm": public, **trajectory(read_csv(RUN_DIRS[internal] / "metrics_per_epoch.csv"))})
    lookup = {row["arm"]: row for row in summaries}
    primary = {key: lookup["O1"][key] - lookup["O0"][key] for key in ("mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "slope_rank1", "slope_mAP", "e60_rank1", "e60_mAP", "best_rank1", "best_mAP")}
    primary.update({"best_rank1_epoch_O1": lookup["O1"]["best_rank1_epoch"], "best_rank1_epoch_O0": lookup["O0"]["best_rank1_epoch"], "best_mAP_epoch_O1": lookup["O1"]["best_mAP_epoch"], "best_mAP_epoch_O0": lookup["O0"]["best_mAP_epoch"]})
    seed1234 = {"I0": trajectory(read_csv(I7_I0_ROOT / "metrics_per_epoch.csv")), "I3": trajectory(read_csv(I7_I3_ROOT / "metrics_per_epoch.csv"))}
    reference = {key: seed1234["I3"][key] - seed1234["I0"][key] for key in ("mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "slope_rank1", "slope_mAP", "e60_rank1", "e60_mAP", "best_rank1", "best_mAP")}
    average = {key: (primary[key] + reference[key]) / 2.0 for key in reference}
    write_csv(TABLE_ROOT / "seed4321_trajectory.csv", summaries)
    write_csv(TABLE_ROOT / "primary_deltas.csv", [{"comparison": "O1-O0", **primary}])
    write_csv(TABLE_ROOT / "seed1234_vs_seed4321.csv", [{"seed": "1234", **reference}, {"seed": "4321", **primary}])
    write_csv(TABLE_ROOT / "two_seed_average.csv", [{"comparison": "mean(seed1234,seed4321)", **average}])
    delta = geometry["delta_O1_minus_O0"]
    q1 = "YES" if primary["auc_rank1"] > 0 else "NO"
    q2 = "YES" if primary["auc_mAP"] > 0 else "NO"
    q3 = "YES" if primary["auc_rank1"] > 0 and primary["auc_mAP"] > 0 else ("MIXED" if primary["auc_rank1"] > 0 or primary["auc_mAP"] > 0 else "NO")
    q4 = "YES" if delta["identity_margin"] >= 0 and delta["ordering_correctness"] >= 0 else ("MIXED" if delta["identity_margin"] >= 0 else "NO")
    q5 = "YES" if exposure["actual_matches_dry_run"] else "NO"
    skip = {row["arm"]: row for row in amp_rows}
    serious_amp = skip["O1"]["total_skipped_steps"] - skip["O0"]["total_skipped_steps"] >= max(4, math.ceil(0.25 * max(1, skip["O0"]["total_skipped_steps"]))) or skip["O1"]["scale_min"] <= skip["O0"]["scale_min"] / 2
    q6 = "YES" if serious_amp else "NO"
    if q3 == "YES" and q5 == "YES" and q6 == "NO":
        classification = "STRONG REPLICATION"
    elif q3 == "MIXED" or (q1 == "YES" or q2 == "YES"):
        classification = "PARTIAL REPLICATION"
    else:
        classification = "FAIL"
    q7 = "YES" if classification == "STRONG REPLICATION" else ("MIXED" if classification == "PARTIAL REPLICATION" else "NO")
    decisions = {"Q1_O1_rank1_auc_improves": q1, "Q2_O1_map_auc_improves": q2, "Q3_both_directionally_consistent": q3, "Q4_identity_margin_improves_or_preserves": q4, "Q5_exposure_faithful": q5, "Q6_meaningful_amp_confound": q6, "Q7_generic_same_id_replicates": q7, "classification": classification, "promotion": "STOP"}
    write_json(TABLE_ROOT / "decisions.json", decisions)
    plots(summaries, geometry, exposure)
    manifest = {"phase": PHASE, "seed": SEED, "arms": ["O0", "O1"], "internal_arms": ["R0", "R1"], "primary_window": "E51-E60", "source_audit": str(SOURCE_REPORT_PATH), "protocol": str(PROTOCOL_ROOT / "protocol.json"), "common_initialization": common_init_audit(), "matched_training_audit": matched, "exposure": exposure, "geometry": geometry, "amp": amp_rows, "seed4321_primary_delta": primary, "seed1234_reference_delta": reference, "two_seed_average_delta": average, "decisions": decisions, "no_third_seed_started": True, "no_new_mechanism_started": True, "phase7a_started": False, "runtime_audit": str(RUNTIME_PATH), "outputs": {"training": str(TRAIN_ROOT), "exposure": str(EXPOSURE_ROOT), "geometry": str(GEOMETRY_ROOT), "amp": str(AMP_ROOT), "tables": str(TABLE_ROOT), "plots": str(PLOT_ROOT), "report": str(REPORT_PATH)}}
    write_json(MANIFEST_PATH, manifest)
    write_report(manifest, summaries, lookup, seed1234, primary, reference, average, decisions)
    print(json.dumps({"classification": classification, "decisions": decisions, "report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH)}, indent=2, sort_keys=True))
    return manifest


def write_report(manifest: dict[str, Any], summaries: list[dict[str, Any]], lookup: dict[str, Any], seed1234: dict[str, Any], primary: dict[str, Any], reference: dict[str, Any], average: dict[str, Any], decisions: dict[str, Any]) -> None:
    lines = ["# PHASE 7O — Generic Same-ID Positive Consistency Second-Seed Replication", "", "Diagnostic scope: two-arm LTCC replication only; no new mechanism was introduced.", "", "## 1. Exact Phase 7I source audit", "", f"- Phase 7I implementation: `{I7_SCRIPT}`.", f"- Frozen I0: `{I7_I0_ROOT / 'model_final.pth'}`.", f"- Frozen I3: `{I7_I3_ROOT / 'model_final.pth'}`.", "- Full hashes and protocol fields: `reports/PHASE_7O_SOURCE_AUDIT.md` and `protocol/protocol.json`.", "", "## 2. Seed4321 initialization audit", "", f"- Common initialization: `{manifest['common_initialization']['source_checkpoint']}`.", f"- SHA256: `{manifest['common_initialization']['source_checkpoint_sha256']}`.", f"- Strict tensor-hash match: `{manifest['common_initialization']['strict_hash_match']}`.", f"- O0/O1 initial parameter equality: `{manifest['matched_training_audit']['initial_parameter_hash_equal']}`.", "", "## 3. Pair-budget/exposure audit", "", f"- Frozen selector: `{manifest['exposure']['selection_logic']}`.", f"- Dry-run expected active rate: `{manifest['exposure']['expected_active_rate']:.8f}`.", f"- Dry-run expected relation contributions: `{manifest['exposure']['expected_relation_contributions']}`.", f"- Same-clothes fraction: `{manifest['exposure']['same_clothes_fraction']:.8f}`; cross-clothes fraction: `{manifest['exposure']['cross_clothes_fraction']:.8f}`.", f"- PID coverage: `{manifest['exposure']['pid_coverage']}`; actual training matches dry-run: `{manifest['exposure']['actual_matches_dry_run']}`.", "", "## 4. O0/O1 E51-E60 retrieval", "", "| Arm | Mean Rank-1 | Mean mAP | AUC Rank-1 | AUC mAP | Rank-1 slope | mAP slope | E60 Rank-1 | E60 mAP | Best Rank-1 | Best mAP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['mean_rank1']:.4f} | {row['mean_mAP']:.4f} | {row['auc_rank1']:.4f} | {row['auc_mAP']:.4f} | {row['slope_rank1']:.4f} | {row['slope_mAP']:.4f} | {row['e60_rank1']:.4f} | {row['e60_mAP']:.4f} | {row['best_rank1']:.4f} (E{row['best_rank1_epoch']}) | {row['best_mAP']:.4f} (E{row['best_mAP_epoch']}) |")
    lines += ["", "## 5. O1 − O0 primary deltas", "", f"`{json.dumps(primary, sort_keys=True)}`", "", "## 6. Seed1234 versus seed4321", "", "| Metric | Seed1234 I3-I0 | Seed4321 O1-O0 | Two-seed mean |", "|---|---:|---:|---:|"]
    for key, label in (("auc_rank1", "Rank-1 AUC"), ("auc_mAP", "mAP AUC"), ("mean_rank1", "Mean Rank-1"), ("mean_mAP", "Mean mAP"), ("e60_rank1", "E60 Rank-1"), ("e60_mAP", "E60 mAP")):
        lines.append(f"| {label} | {reference[key]:.4f} | {primary[key]:.4f} | {average[key]:.4f} |")
    lines += ["", "Two seeds are replication evidence only; no statistical significance claim is made.", "", "## 7. E60 geometry", "", "| Metric | O0 | O1 | O1-O0 |", "|---|---:|---:|---:|"]
    for key, label in (("same_pid_same_clothes_similarity", "Same-ID same-clothes similarity"), ("same_pid_cross_clothes_similarity", "Same-ID cross-clothes similarity"), ("nearest_negative_pid_center_similarity", "Nearest-negative similarity"), ("identity_margin", "Identity margin"), ("ordering_correctness", "Ordering correctness"), ("within_pid_dispersion", "Within-PID dispersion"), ("clothing_state_center_dispersion", "Clothing-state dispersion")):
        lines.append(f"| {label} | {manifest['geometry']['summaries']['O0'][key]:.6f} | {manifest['geometry']['summaries']['O1'][key]:.6f} | {manifest['geometry']['delta_O1_minus_O0'][key]:+.6f} |")
    lines += ["", "## 8. AMP audit", "", "| Arm | Skipped steps | found_inf | Nonfinite | Scale min | Scale max | Final scale |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in manifest["amp"]: lines.append(f"| {row['arm']} | {row['total_skipped_steps']} | {row['found_inf']} | {row['nonfinite_events']} | {row['scale_min']:.1f} | {row['scale_max']:.1f} | {row['final_scale']:.1f} |")
    lines += ["", "## 9. Q1-Q7", "", "| Question | Answer |", "|---|---|", f"| Q1. O1 improves Rank-1 AUC | **{decisions['Q1_O1_rank1_auc_improves']}** |", f"| Q2. O1 improves mAP AUC | **{decisions['Q2_O1_map_auc_improves']}** |", f"| Q3. Both improvements directionally match seed1234 | **{decisions['Q3_both_directionally_consistent']}** |", f"| Q4. O1 improves/preserves identity margin | **{decisions['Q4_identity_margin_improves_or_preserves']}** |", f"| Q5. Exposure is faithful to I3 | **{decisions['Q5_exposure_faithful']}** |", f"| Q6. Meaningful AMP confound | **{decisions['Q6_meaningful_amp_confound']}** |", f"| Q7. Generic Same-ID Positive Consistency replicates | **{decisions['Q7_generic_same_id_replicates']}** |", "", "## 10. Final decision", "", f"**{decisions['classification']} — STOP**", "", "The result is a replication-only conclusion; no third seed, PRCC run, or mechanism redesign was started.", "", "NO THIRD SEED WAS STARTED.", "NO NEW MECHANISM WAS STARTED."]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "train", "analyze", "probe"), required=True)
    parser.add_argument("--arm", choices=("R0", "R1"), default=None)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--probe-optimizer", action="store_true")
    args = parser.parse_args()
    if args.mode == "audit":
        source_audit()
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for train")
        train(args.arm)
    elif args.mode == "probe":
        ppu_probe(args.probe_steps, args.probe_optimizer)
    else:
        analyze()


if __name__ == "__main__":
    main()
