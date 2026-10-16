#!/usr/bin/env python3
"""Phase 5C: full-data ID-Damage-aware soft sampling.

This runner is a narrow continuation of the new-method track.  Every setting
keeps all LTCC training images and clothing states available from epoch 1.
The only non-uniform operation is a bounded, within-PID clothing-state
sampling priority:

    rank in [0, 1], weight = 1 + rank, beta = 1

P0 is uniform, P1 uses the frozen Phase-5A fixed semantic Clothing score, P2
uses the frozen Phase-5A H_ID score, and P3 uses a precomputed random rank.
PID pair composition, optimizer budget, model, losses, augmentation, and
common initialization are shared.  Sampling weight is not loss weight: every
sample has loss coefficient one.

The new hierarchical sampler changes the ordinary uniform path, so P0 is
rerun as a matched control.  The old Phase-5B S0 artifact remains the frozen
external Bio reference.

Modes:
  prepare   copy/freeze common init and probability tables
  preflight audit topology, PID plan, positive probabilities and coverage
  train     run one P0/P1/P2/P3 setting under torchrun
  analyze   write the 15-epoch analysis and report
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.phase5b_id_damage_ds as p5b  # noqa: E402


PHASE = "phase5c_full_data_id_damage_sampling"
SEED = 1234
BETA = 1.0
WORLD_SIZE = p5b.WORLD_SIZE
STEPS_PER_EPOCH = p5b.STEPS_PER_EPOCH
BATCH_SIZE = p5b.BATCH_SIZE
NUM_INSTANCES = p5b.NUM_INSTANCES
PAIRS_PER_BATCH = BATCH_SIZE // NUM_INSTANCES
TOTAL_EPOCHS = p5b.TOTAL_EPOCHS
SCREEN_EPOCHS = p5b.SCREEN_EPOCHS
INITIAL_SCALE = p5b.INITIAL_SCALE
NUM_CLASSES = p5b.NUM_CLASSES
NUM_CAMERAS = p5b.NUM_CAMERAS
GLOBAL_PAIR_UNITS = p5b.GLOBAL_PAIR_UNITS
LOCAL_PAIR_UNITS = p5b.LOCAL_PAIR_UNITS

P5A_ROOT = ROOT / "outputs/new_method_track/phase5a_clothing_hardness"
P5A_GROUPS = P5A_ROOT / "group_metrics.csv"
P5A_MANIFEST = P5A_ROOT / "PHASE_5A_MANIFEST.json"
P5A_REPORT = ROOT / "reports/PHASE_5A_CLOTHING_HARDNESS_FEASIBILITY.md"

P5B_ROOT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds"
P5B_COMMON_INIT = P5B_ROOT / "common_init/COMMON_PHASE5B_INIT.pth"
P5B_COMMON_HASHES = P5B_ROOT / "common_init/COMMON_PHASE5B_INIT_HASHES.json"
P5B_C0_METRICS = P5B_ROOT / "S0_bio_full_data/metrics_per_epoch.csv"
P5B_C0_RUN = P5B_ROOT / "S0_bio_full_data/RUN_COMPLETE.json"
P5B_C1_METRICS = P5B_ROOT / "S2_id_damage_ds/metrics_per_epoch.csv"
P5B_C1_RUN = P5B_ROOT / "S2_id_damage_ds/RUN_COMPLETE.json"

OUT_ROOT = ROOT / "outputs/new_method_track/phase5c_full_data_hardness_sampling"
COMMON_ROOT = OUT_ROOT / "common_init"
COMMON_INIT = COMMON_ROOT / "COMMON_PHASE5C_INIT.pth"
COMMON_HASHES = COMMON_ROOT / "COMMON_PHASE5C_INIT_HASHES.json"
SCHEDULE_ROOT = OUT_ROOT / "schedules"
RUN_DIRS = {
    "P0": "P0_uniform",
    "P1": "P1_clothing_priority",
    "P2": "P2_id_damage_priority",
    "P3": "P3_random_priority",
}
RUNS = tuple(RUN_DIRS)
REPORT_PATH = ROOT / "reports/PHASE_5C_FULL_DATA_ID_DAMAGE_SAMPLING.md"

BIO_CONTROL_R1 = 0.5637755393981934
BIO_CONTROL_MAP = 0.32385736411857605
ORIGINAL_FULL_R1 = 0.5331632494926453
ORIGINAL_FULL_MAP = 0.292657

METRIC_FIELDS = [
    "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "lr", "lr_bias",
    "id_accuracy", "bio_accuracy", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "skipped_steps", "found_inf_count", "nonfinite_events", "optimizer_updates",
    "active_images", "active_pids", "active_clothes", "mean_clothes_per_pid",
    "new_clothes_count", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP",
    "general_rank1", "general_rank5", "general_rank10", "general_mAP",
    "mean_sampled_h_id", "median_sampled_h_id", "p25_sampled_h_id", "p75_sampled_h_id",
    "mean_sampled_clothing_score", "unique_images_seen", "unique_clothes_seen",
    "image_coverage", "clothing_coverage",
]

EXPOSURE_FIELDS = [
    "run", "epoch", "pid", "cloth_id", "support", "ranking_source", "ranking_score",
    "rank", "weight", "state_probability", "image_probability", "pid_sample_slots",
    "expected_sample_count", "actual_sample_count", "exposure_ratio", "seen_in_epoch",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def stable_int(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


def stable_uniform(*parts: Any) -> float:
    # Avoid exact zero/one so categorical selection always has a valid bin.
    value = int.from_bytes(hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()[:8], "little")
    return (value + 0.5) / float(2**64)


def config() -> Any:
    cfg = p5b.configure()
    cfg.defrost()
    cfg.OUTPUT_DIR = str(OUT_ROOT)
    cfg.SOLVER.SEED = SEED
    cfg.freeze()
    return cfg


def run_dir(run: str) -> Path:
    if run not in RUN_DIRS:
        raise ValueError(run)
    return OUT_ROOT / RUN_DIRS[run]


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() in ("", "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def save_git_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True, text=False).stdout
    (path / "git_diff.patch").write_bytes(diff)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    (path / "git_status.txt").write_text(status, encoding="utf-8")


def load_context() -> dict[str, Any]:
    if not P5A_GROUPS.is_file() or not P5A_MANIFEST.is_file():
        raise FileNotFoundError("Phase 5A group metrics/manifest is missing")
    events, group_by_key, initial, all_clothes = p5b.load_frozen_events()
    if len(group_by_key) != 256 or len(initial) != NUM_CLASSES:
        raise RuntimeError("unexpected Phase 5A group topology")
    records = {}
    for key, row in sorted(group_by_key.items()):
        paths = p5b.parse_json_value(row.get("image_paths_json"), [])
        names = p5b.parse_json_value(row.get("image_filenames_json"), [])
        if not paths or any(not str(path).startswith(str(p5b.TRAIN_ROOT) + "/") for path in paths):
            raise RuntimeError(f"non-training source in group {key}")
        if any(str(path).startswith(str(p5b.QUERY_ROOT)) or str(path).startswith(str(p5b.GALLERY_ROOT)) for path in paths):
            raise RuntimeError(f"query/gallery source in group {key}")
        records[key] = {
            "pid": int(row["pid"]),
            "cloth_id": int(row["cloth_id"]),
            "cloth_key": row["cloth_key"],
            "support": int(row["support"]),
            "image_paths": [str(path) for path in paths],
            "image_filenames": [str(name) for name in names],
            "h_id": float(row["h_id_mean"]),
            # Phase 5A's primary pure Clothing score: fixed semantic
            # representation, leave-one-out mean same-PID center distance.
            "clothing_score": parse_float(row.get("clothing_text_fixed_h_cloth_loo_mean")),
        }
    manifest = json.loads(P5A_MANIFEST.read_text())
    source = manifest.get("sources", {}).get("clothing_text_fixed", {})
    return {
        "events": events,
        "group_by_key": group_by_key,
        "records": records,
        "initial": initial,
        "all_clothes": all_clothes,
        "phase5a_manifest": manifest,
        "phase5a_clothing_source": source,
        "phase5a_group_sha256": sha256_file(P5A_GROUPS),
        "phase5a_manifest_sha256": sha256_file(P5A_MANIFEST),
    }


def copy_common_init(cfg: Any) -> dict[str, Any]:
    if not P5B_COMMON_INIT.is_file() or not P5B_COMMON_HASHES.is_file():
        raise FileNotFoundError("Phase 5B common init artifact is missing")
    COMMON_ROOT.mkdir(parents=True, exist_ok=True)
    if COMMON_INIT.exists() or COMMON_HASHES.exists():
        if not COMMON_INIT.is_file() or not COMMON_HASHES.is_file():
            raise RuntimeError("Phase 5C common init is partially present; refusing to repair it implicitly")
        existing = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        existing_meta = json.loads(COMMON_HASHES.read_text())
        source = torch.load(P5B_COMMON_INIT, map_location="cpu", weights_only=False)
        if existing_meta.get("state_sha256") != sha256_file(COMMON_INIT) or p5b.tensor_hashes(existing) != p5b.tensor_hashes(source):
            raise RuntimeError("existing Phase 5C common init does not match Phase 5B source")
        state = existing
    else:
        shutil.copyfile(P5B_COMMON_INIT, COMMON_INIT)
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    reference = json.loads(P5B_COMMON_HASHES.read_text())
    if p5b.tensor_hashes(state) != reference["tensor_hashes"]:
        raise RuntimeError("copied Phase 5B common init tensor hash differs")
    write_json(COMMON_HASHES, {
        "phase": PHASE,
        "created_utc": utc_now(),
        "source": str(P5B_COMMON_INIT),
        "source_sha256": sha256_file(P5B_COMMON_INIT),
        "state_sha256": sha256_file(COMMON_INIT),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "tensor_hashes": p5b.tensor_hashes(state),
        "topology": "EVA02-Large Bio-only: CE + Triplet + Summary Bio",
        "reverse": "OFF",
    })
    (COMMON_ROOT / "config.yml").write_text(cfg.dump(), encoding="utf-8")
    return {
        "path": str(COMMON_INIT),
        "sha256": sha256_file(COMMON_INIT),
        "hashes_path": str(COMMON_HASHES),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
    }


def random_rank_order(pid: int, clothes: list[int]) -> list[int]:
    order = list(sorted(clothes))
    random.Random(stable_int(PHASE, "random-order", SEED, pid)).shuffle(order)
    return order


def state_order(run: str, pid: int, clothes: list[int], records: dict[tuple[int, int], dict[str, Any]]) -> tuple[list[int], str]:
    if run == "P0":
        return sorted(clothes), "uniform"
    if run == "P1":
        # Undefined LOO score occurs only for one-clothing PIDs.  Those PIDs
        # have no within-ID clothing ranking and therefore remain uniform.
        return sorted(clothes, key=lambda cid: (records[(pid, cid)]["clothing_score"] if records[(pid, cid)]["clothing_score"] is not None else 0.0, cid)), "clothing_text_fixed_h_cloth_loo_mean"
    if run == "P2":
        return sorted(clothes, key=lambda cid: (records[(pid, cid)]["h_id"], cid)), "frozen_h_id_mean"
    if run == "P3":
        return random_rank_order(pid, clothes), "fixed_random_rank_seed1234"
    raise ValueError(run)


def build_probability_table(run: str, context: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    records = context["records"]
    by_pid: dict[int, list[int]] = defaultdict(list)
    for pid, cid in sorted(records):
        by_pid[pid].append(cid)
    table = {}
    for pid in sorted(by_pid):
        clothes = sorted(by_pid[pid])
        order, source = state_order(run, pid, clothes, records)
        rank_by_cid = {cid: (position / max(len(order) - 1, 1)) if len(order) > 1 else 0.0 for position, cid in enumerate(order)}
        # P0 is uniform by construction; P1/P2/P3 use exactly the same
        # bounded formula after within-PID rank normalization.
        weights = {cid: 1.0 + BETA * rank_by_cid[cid] for cid in clothes}
        total = sum(weights.values())
        for cid in clothes:
            rec = records[(pid, cid)]
            state_probability = weights[cid] / total
            table[(pid, cid)] = {
                "pid": pid,
                "cloth_id": cid,
                "support": rec["support"],
                "ranking_source": source,
                "ranking_score": None if run in ("P0", "P3") else rec["clothing_score"] if run == "P1" else rec["h_id"],
                "rank": rank_by_cid[cid],
                "weight": weights[cid],
                "state_probability": state_probability,
                "image_probability": state_probability / rec["support"],
            }
    return table


def probability_rows(table: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(table):
        item = dict(table[key])
        rows.append(item)
    return rows


def make_pid_plans() -> dict[int, dict[int, list[int]]]:
    pids = list(range(NUM_CLASSES))
    plans: dict[int, dict[int, list[int]]] = {}
    for epoch in range(1, TOTAL_EPOCHS + 1):
        rng = random.Random(stable_int(PHASE, "pid-plan", SEED, epoch))
        stream: list[int] = []
        while len(stream) < GLOBAL_PAIR_UNITS:
            cycle = list(pids)
            rng.shuffle(cycle)
            stream.extend(cycle)
        stream = stream[:GLOBAL_PAIR_UNITS]
        rank_streams = [stream[rank::WORLD_SIZE] for rank in range(WORLD_SIZE)]
        for rank in range(WORLD_SIZE):
            local = rank_streams[rank]
            for start in range(0, len(local), PAIRS_PER_BATCH):
                stop = start + PAIRS_PER_BATCH
                batch = local[start:stop]
                if len(batch) != PAIRS_PER_BATCH:
                    raise RuntimeError("incomplete PID batch")
                if len(set(batch)) >= 2:
                    continue
                dominant = batch[0]
                replacement = None
                for candidate in range(stop, len(local)):
                    if local[candidate] != dominant:
                        replacement = candidate
                        break
                if replacement is None:
                    raise RuntimeError("could not repair all-same-PID batch")
                local[stop - 1], local[replacement] = local[replacement], local[stop - 1]
            if any(len(set(local[start:start + PAIRS_PER_BATCH])) < 2 for start in range(0, len(local), PAIRS_PER_BATCH)):
                raise RuntimeError("PID plan contains an empty-negative batch")
        plans[epoch] = {rank: rank_streams[rank] for rank in range(WORLD_SIZE)}
    return plans


def pid_plan_hash(values: list[int]) -> str:
    return sha256_bytes(json.dumps(values, separators=(",", ":")).encode())


def pid_slot_bases(pid_plans: dict[int, dict[int, list[int]]]) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, dict[int, int]]]]:
    cumulative = {pid: 0 for pid in range(NUM_CLASSES)}
    epoch_bases: dict[int, dict[int, int]] = {}
    rank_bases: dict[int, dict[int, dict[int, int]]] = {}
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_bases[epoch] = dict(cumulative)
        rank_bases[epoch] = {}
        within_epoch = {pid: 0 for pid in range(NUM_CLASSES)}
        for rank in range(WORLD_SIZE):
            rank_bases[epoch][rank] = {pid: epoch_bases[epoch][pid] + within_epoch[pid] for pid in range(NUM_CLASSES)}
            for pid in pid_plans[epoch][rank]:
                within_epoch[pid] += NUM_INSTANCES
        for pid in range(NUM_CLASSES):
            cumulative[pid] += within_epoch[pid]
    return epoch_bases, rank_bases


def make_index_context(dataset: Any) -> dict[str, Any]:
    index_map = p5b.build_index_map(dataset)
    if len(index_map) != 256 or sum(len(indices) for indices in index_map.values()) != 9576:
        raise RuntimeError("unexpected LTCC train index map")
    index_to_state = {}
    pid_to_indices: dict[int, list[int]] = defaultdict(list)
    for (pid, cid), indices in sorted(index_map.items()):
        for index in indices:
            index_to_state[int(index)] = (int(pid), int(cid))
            pid_to_indices[int(pid)].append(int(index))
    for pid in pid_to_indices:
        pid_to_indices[pid] = sorted(pid_to_indices[pid])
    coverage_order = {}
    for pid, indices in sorted(pid_to_indices.items()):
        order = list(indices)
        random.Random(stable_int(PHASE, "coverage-order", SEED, pid)).shuffle(order)
        coverage_order[pid] = order
    return {
        "index_map": index_map,
        "index_to_state": index_to_state,
        "pid_to_indices": dict(pid_to_indices),
        "coverage_order": coverage_order,
    }


def choose_categorical(states: list[int], probabilities: list[float], unit: float) -> int:
    cumulative = 0.0
    for state, probability in zip(states, probabilities):
        cumulative += probability
        if unit < cumulative:
            return state
    return states[-1]


def build_rank_plan(
    run: str,
    epoch: int,
    rank: int,
    dataset: Any,
    index_context: dict[str, Any],
    probability_table: dict[tuple[int, int], dict[str, Any]],
    pid_plans: dict[int, dict[int, list[int]]],
    rank_bases: dict[int, dict[int, dict[int, int]]],
) -> dict[str, Any]:
    index_map = index_context["index_map"]
    index_to_state = index_context["index_to_state"]
    coverage_order = index_context["coverage_order"]
    pid_stream = pid_plans[epoch][rank]
    samples: list[tuple[dict[str, Any], int]] = []
    batches: list[list[int]] = []
    state_counts: Counter[tuple[int, int]] = Counter()
    image_counts: Counter[int] = Counter()
    pid_slot_cursor = Counter()
    reserved_count = 0
    pid_counts: Counter[int] = Counter(pid_stream)
    for pair_position, pid in enumerate(pid_stream):
        batch_position = pair_position // PAIRS_PER_BATCH
        pair_in_batch = pair_position % PAIRS_PER_BATCH
        if pair_in_batch == 0:
            batches.append([])
        states = sorted(cid for (state_pid, cid) in probability_table if state_pid == pid)
        probabilities = [probability_table[(pid, cid)]["state_probability"] for cid in states]
        if not states or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"invalid state probability table for PID {pid}")
        for slot in range(NUM_INSTANCES):
            position = rank_bases[epoch][rank][pid] + pid_slot_cursor[pid]
            pid_slot_cursor[pid] += 1
            if position < len(coverage_order[pid]):
                dataset_index = coverage_order[pid][position]
                reserved_count += 1
            else:
                state_unit = stable_uniform(PHASE, "state", SEED, epoch, rank, pair_position, slot)
                cid = choose_categorical(states, probabilities, state_unit)
                candidates = index_map[(pid, cid)]
                image_unit = stable_uniform(PHASE, "image", SEED, epoch, rank, pair_position, slot)
                dataset_index = candidates[min(int(image_unit * len(candidates)), len(candidates) - 1)]
            sample_pid, sample_cid = index_to_state[int(dataset_index)]
            if sample_pid != pid:
                raise RuntimeError("selected image does not belong to planned PID")
            state_counts[(pid, sample_cid)] += 1
            image_counts[int(dataset_index)] += 1
            augmentation_seed = p5b.plan_seed(epoch, rank, batch_position + 1, pair_in_batch * NUM_INSTANCES + slot, int(dataset_index))
            samples.append((dict(dataset.train[int(dataset_index)]), augmentation_seed))
            batches[batch_position].append(len(samples) - 1)
    if len(batches) != STEPS_PER_EPOCH or any(len(batch) != BATCH_SIZE for batch in batches):
        raise RuntimeError(f"rank {rank} sampler did not produce {STEPS_PER_EPOCH} full batches")
    if set(pid_counts) != set(range(NUM_CLASSES)):
        raise RuntimeError("PID plan dropped a training identity")
    plan_digest = hashlib.sha256()
    for source, _seed in samples:
        plan_digest.update(str(source["image_path"]).encode())
        plan_digest.update(b"\0")
    pid_digest = pid_plan_hash(pid_stream)
    meta = {
        "run": run,
        "epoch": epoch,
        "rank": rank,
        "batches": len(batches),
        "pair_units": len(pid_stream),
        "samples": len(samples),
        "pid_plan_sha256": pid_digest,
        "plan_sha256": plan_digest.hexdigest(),
        "pid_counts": {str(pid): int(count) for pid, count in sorted(pid_counts.items())},
        "state_counts": {f"{pid}:{cid}": int(count) for (pid, cid), count in sorted(state_counts.items())},
        "image_counts": {str(index): int(count) for index, count in sorted(image_counts.items())},
        "reserved_image_count": reserved_count,
        "covered_pids": sorted(pid_counts),
        "batch_unique_pid_min": min(len(set(pid_stream[start:start + PAIRS_PER_BATCH])) for start in range(0, len(pid_stream), PAIRS_PER_BATCH)),
        "batch_unique_pid_max": max(len(set(pid_stream[start:start + PAIRS_PER_BATCH])) for start in range(0, len(pid_stream), PAIRS_PER_BATCH)),
        "all_images_available": True,
        "all_clothes_available": True,
    }
    return {"samples": samples, "batches": batches, "meta": meta, "state_counts": state_counts, "image_counts": image_counts}


def merge_counter_dicts(metas: list[dict[str, Any]], key: str) -> Counter:
    result = Counter()
    for meta in metas:
        result.update({item: int(value) for item, value in meta[key].items()})
    return result


def coverage_stats(
    image_counts: Counter,
    index_context: dict[str, Any],
    seen_images: set[int],
    seen_clothes: set[tuple[int, int]],
) -> tuple[dict[str, Any], set[int], set[tuple[int, int]]]:
    epoch_images = set(int(index) for index, count in image_counts.items() if count > 0)
    epoch_clothes = {index_context["index_to_state"][index] for index in epoch_images}
    seen_images = set(seen_images) | epoch_images
    seen_clothes = set(seen_clothes) | epoch_clothes
    return {
        "epoch_unique_images": len(epoch_images),
        "epoch_unique_clothes": len(epoch_clothes),
        "epoch_unique_pids": len({pid for pid, _cid in epoch_clothes}),
        "cumulative_unique_images": len(seen_images),
        "cumulative_unique_clothes": len(seen_clothes),
        "image_coverage": len(seen_images) / 9576.0,
        "clothing_coverage": len(seen_clothes) / 256.0,
        "all_images_seen": len(seen_images) == 9576,
        "all_clothes_seen": len(seen_clothes) == 256,
        "missing_image_count": 9576 - len(seen_images),
        "missing_clothes_count": 256 - len(seen_clothes),
    }, seen_images, seen_clothes


def hardness_stats(state_counts: Counter, probability_table: dict[tuple[int, int], dict[str, Any]], records: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    values = []
    clothing_values = []
    for key, count in state_counts.items():
        values.extend([records[key]["h_id"]] * int(count))
        score = probability_table[key]["ranking_score"]
        if score is not None:
            clothing_values.extend([float(score)] * int(count))
    if not values:
        raise RuntimeError("no sampled state for hardness exposure")
    result = {
        "mean_sampled_h_id": float(np.mean(values)),
        "median_sampled_h_id": float(np.median(values)),
        "p25_sampled_h_id": float(np.percentile(values, 25)),
        "p75_sampled_h_id": float(np.percentile(values, 75)),
        "mean_sampled_clothing_score": float(np.mean(clothing_values)) if clothing_values else None,
    }
    return result


def pid_sample_slots(pid_plans: dict[int, dict[int, list[int]]], epoch: int) -> Counter:
    counts = Counter()
    for rank in range(WORLD_SIZE):
        counts.update(pid_plans[epoch][rank])
    return Counter({pid: count * NUM_INSTANCES for pid, count in counts.items()})


def exposure_rows(
    run: str,
    epoch: int,
    state_counts: Counter,
    image_counts: Counter,
    probability_table: dict[tuple[int, int], dict[str, Any]],
    records: dict[tuple[int, int], dict[str, Any]],
    slots: Counter,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(probability_table):
        item = probability_table[key]
        actual = int(state_counts.get(f"{key[0]}:{key[1]}", state_counts.get(key, 0)))
        expected = float(slots[key[0]] * item["state_probability"])
        rows.append({
            "run": run,
            "epoch": epoch,
            "pid": key[0],
            "cloth_id": key[1],
            "support": records[key]["support"],
            "ranking_source": item["ranking_source"],
            "ranking_score": item["ranking_score"],
            "rank": item["rank"],
            "weight": item["weight"],
            "state_probability": item["state_probability"],
            "image_probability": item["image_probability"],
            "pid_sample_slots": slots[key[0]],
            "expected_sample_count": expected,
            "actual_sample_count": actual,
            "exposure_ratio": actual / expected if expected > 0 else None,
            "seen_in_epoch": int(actual > 0),
        })
    return rows


def write_probability_artifacts(run: str, table: dict[tuple[int, int], dict[str, Any]], context: dict[str, Any]) -> None:
    path = run_dir(run)
    write_csv(path / "sampling_probability.csv", probability_rows(table), [
        "pid", "cloth_id", "support", "ranking_source", "ranking_score", "rank", "weight", "state_probability", "image_probability",
    ])
    by_pid = defaultdict(list)
    for key in sorted(table):
        by_pid[str(key[0])].append({"cloth_id": key[1], "ranking_score": table[key]["ranking_score"], "rank": table[key]["rank"], "weight": table[key]["weight"]})
    write_json(path / "sampling_rank.json", {
        "run": run,
        "beta": BETA,
        "formula": "w=1+beta*within-PID-rank; p=w/sum(w); image_p=p/support",
        "ranking": {"P0": "uniform", "P1": "Phase-5A fixed semantic Clothing LOO mean", "P2": "Phase-5A frozen H_ID mean", "P3": "random rank seed 1234"}[run],
        "pids": by_pid,
        "phase5a_group_metrics_sha256": context["phase5a_group_sha256"],
    })


def make_pid_plan_artifact(pid_plans: dict[int, dict[int, list[int]]]) -> dict[str, Any]:
    return {
        str(epoch): {
            str(rank): {
                "sha256": pid_plan_hash(pid_plans[epoch][rank]),
                "pair_units": len(pid_plans[epoch][rank]),
                "pid_counts": dict(sorted(Counter(pid_plans[epoch][rank]).items())),
            }
            for rank in range(WORLD_SIZE)
        }
        for epoch in range(1, TOTAL_EPOCHS + 1)
    }


def prepare() -> None:
    marker = OUT_ROOT / "PHASE5C_PREPARED.json"
    if marker.exists():
        raise RuntimeError("Phase 5C artifacts already prepared; refusing to overwrite frozen probability tables")
    cfg = config()
    context = load_context()
    common = copy_common_init(cfg)
    pid_plans = make_pid_plans()
    epoch_bases, rank_bases = pid_slot_bases(pid_plans)
    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(SCHEDULE_ROOT / "pid_plan_audit.json", make_pid_plan_artifact(pid_plans))
    write_json(SCHEDULE_ROOT / "pid_slot_bases.json", {"epoch_bases": epoch_bases, "rank_bases": rank_bases})
    write_json(SCHEDULE_ROOT / "sampling_contract.json", {
        "all_images_always_available": True,
        "all_clothing_states_always_available": True,
        "pid_sampling": "shared fixed PID pair plan; only within-PID state priority changes",
        "num_instances": NUM_INSTANCES,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "beta": BETA,
        "weight_formula": "w=1+rank",
        "loss_weight": 1.0,
        "random_seed": SEED,
        "coverage_reserve": "deterministic image reserve distributed across all 15+ epochs; remaining draws use common random numbers",
    })
    cfg_dump = cfg.dump()
    for run in RUNS:
        path = run_dir(run)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yml").write_text(cfg_dump, encoding="utf-8")
        save_git_snapshot(path)
        table = build_probability_table(run, context)
        write_probability_artifacts(run, table, context)
        if any(item["image_probability"] <= 0.0 for item in table.values()):
            raise RuntimeError(f"{run} contains a zero image probability")
        write_json(path / "schedule_hash.json", {
            "run": run,
            "sampling_probability_sha256": sha256_file(path / "sampling_probability.csv"),
            "sampling_rank_sha256": sha256_file(path / "sampling_rank.json"),
            "frozen": True,
        })
        write_json(path / "run_manifest.json", {
            "phase": PHASE,
            "run": run,
            "label": {"P0": "Uniform Full-data Bio", "P1": "Clothing-distance Soft Sampling + Bio", "P2": "ID-Damage Soft Sampling + Bio", "P3": "Random-priority Soft Sampling + Bio"}[run],
            "ranking": {"P0": "uniform", "P1": "frozen Phase-5A clothing_text_fixed_h_cloth_loo_mean", "P2": "frozen Phase-5A h_id_mean", "P3": "fixed random rank seed 1234"}[run],
            "coverage": "all 9576 train images and all 256 clothing states available from epoch 1; no probability is zero",
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "online_h_id": "OFF",
            "loss_weighting": "OFF; loss coefficient is 1 for every sampled image",
            "common_init": common,
            "sampling_probability": str(path / "sampling_probability.csv"),
            "sampling_probability_sha256": sha256_file(path / "sampling_probability.csv"),
            "phase5a_group_metrics_sha256": context["phase5a_group_sha256"],
            "screen_epochs": SCREEN_EPOCHS,
            "total_epochs": TOTAL_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "optimizer_updates": TOTAL_EPOCHS * STEPS_PER_EPOCH,
            "data": {"train_images": 9576, "train_pids": NUM_CLASSES, "train_clothes": 256},
            "pid_plan": str(SCHEDULE_ROOT / "pid_plan_audit.json"),
        })
    write_json(OUT_ROOT / "PHASE5C_PREPARED.json", {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "beta": BETA,
        "world_size": WORLD_SIZE,
        "common_init": common,
        "phase5a_group_metrics_sha256": context["phase5a_group_sha256"],
        "phase5a_manifest_sha256": context["phase5a_manifest_sha256"],
        "pid_plan_hashes": make_pid_plan_artifact(pid_plans),
        "contract": {
            "only_variable": "within-PID clothing-state sampling rank",
            "all_data_available": True,
            "zero_probability_for_any_image": False,
            "pid_frequency_shared": True,
            "same_pid_pair_plan": True,
            "same_optimizer_budget": True,
            "sampling_weight_is_loss_weight": False,
        },
        "external_references": {
            "phase5b_c0_metrics": str(P5B_C0_METRICS),
            "phase5b_c1_metrics": str(P5B_C1_METRICS),
            "phase5b_common_init": str(P5B_COMMON_INIT),
        },
    })
    print(json.dumps({"prepared": True, "output": str(OUT_ROOT), "common_init": common}, indent=2, sort_keys=True))


def load_prepared() -> dict[str, Any]:
    path = OUT_ROOT / "PHASE5C_PREPARED.json"
    if not path.is_file():
        raise FileNotFoundError("run prepare first: " + str(path))
    return json.loads(path.read_text())


def load_context_and_dataset(cfg: Any) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    context = load_context()
    dataset = p5b.build_dataset(cfg)
    if dataset.num_train_pids != NUM_CLASSES or dataset.num_camera != NUM_CAMERAS:
        raise RuntimeError("unexpected LTCC dataset topology")
    index_context = make_index_context(dataset)
    for key, record in context["records"].items():
        if len(index_context["index_map"][key]) != record["support"]:
            raise RuntimeError(f"support mismatch for {key}")
    return context, dataset, index_context


def preflight() -> None:
    prepared = load_prepared()
    cfg = config()
    state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    hashes = json.loads(COMMON_HASHES.read_text())
    if sha256_file(COMMON_INIT) != hashes["state_sha256"] or p5b.tensor_hashes(state) != hashes["tensor_hashes"]:
        raise RuntimeError("Phase 5C common init hash mismatch")
    model = p5b.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    topology = {
        "model_name": cfg.MODEL.NAME,
        "state_key_count": len(model.state_dict()),
        "state_numel": int(sum(value.numel() for value in model.state_dict().values())),
        "has_reverse_parameters": any("Reverse" in key or "reverse" in key for key in model.state_dict()),
        "loss_type": list(cfg.MODEL.LOSS_TYPE),
    }
    del model, state
    gc.collect()
    if topology["has_reverse_parameters"] or topology["loss_type"] != ["ce", "triplet", "clipBio"]:
        raise RuntimeError("Phase 5C topology is not Bio-only")

    context, dataset, index_context = load_context_and_dataset(cfg)
    pid_plans = make_pid_plans()
    epoch_bases, rank_bases = pid_slot_bases(pid_plans)
    pid_hashes = {
        str(epoch): {str(rank): pid_plan_hash(pid_plans[epoch][rank]) for rank in range(WORLD_SIZE)}
        for epoch in range(1, TOTAL_EPOCHS + 1)
    }
    run_audits = {}
    all_seen = {}
    for run in RUNS:
        table = build_probability_table(run, context)
        if len(table) != 256 or any(item["image_probability"] <= 0.0 for item in table.values()):
            raise RuntimeError(f"{run} has invalid positive image probabilities")
        if any(not math.isclose(sum(table[(pid, cid)]["state_probability"] for cid in sorted(context["all_clothes"][pid])), 1.0, rel_tol=1e-9, abs_tol=1e-9) for pid in context["all_clothes"]):
            raise RuntimeError(f"{run} state probabilities do not sum to one within PID")
        seen_images: set[int] = set()
        seen_clothes: set[tuple[int, int]] = set()
        rows = []
        first_pid_hashes = None
        for epoch in range(1, SCREEN_EPOCHS + 1):
            metas = []
            merged_image_counts = Counter()
            for rank in range(WORLD_SIZE):
                plan = build_rank_plan(run, epoch, rank, dataset, index_context, table, pid_plans, rank_bases)
                meta = plan["meta"]
                metas.append(meta)
                merged_image_counts.update(plan["image_counts"])
            if first_pid_hashes is None:
                first_pid_hashes = [meta["pid_plan_sha256"] for meta in metas]
            if any(meta["batches"] != STEPS_PER_EPOCH or meta["batch_unique_pid_min"] < 2 or len(meta["covered_pids"]) != NUM_CLASSES for meta in metas):
                raise RuntimeError(f"{run} sampler preflight failed at epoch {epoch}")
            coverage, seen_images, seen_clothes = coverage_stats(merged_image_counts, index_context, seen_images, seen_clothes)
            rows.append({
                "run": run,
                "epoch": epoch,
                "active_images": 9576,
                "active_clothes": 256,
                "active_pids": NUM_CLASSES,
                "global_samples": sum(meta["samples"] for meta in metas),
                "rank0_steps": metas[0]["batches"],
                "rank1_steps": metas[1]["batches"],
                "pid_plan_hash_equal": metas[0]["pid_plan_sha256"] == metas[1]["pid_plan_sha256"],
                "pid_plan_hash_rank0": metas[0]["pid_plan_sha256"],
                "pid_plan_hash_rank1": metas[1]["pid_plan_sha256"],
                "plan_hash_rank0": metas[0]["plan_sha256"],
                "plan_hash_rank1": metas[1]["plan_sha256"],
                "all_77_pids_present": all(len(meta["covered_pids"]) == NUM_CLASSES for meta in metas),
                "all_images_probability_positive": True,
                "epoch_unique_images": coverage["epoch_unique_images"],
                "epoch_unique_clothes": coverage["epoch_unique_clothes"],
                "cumulative_unique_images": coverage["cumulative_unique_images"],
                "cumulative_unique_clothes": coverage["cumulative_unique_clothes"],
                "image_coverage": coverage["image_coverage"],
                "clothing_coverage": coverage["clothing_coverage"],
            })
        all_seen[run] = {
            "all_images_seen_by_epoch15": len(seen_images) == 9576,
            "all_clothes_seen_by_epoch15": len(seen_clothes) == 256,
            "unique_images": len(seen_images),
            "unique_clothes": len(seen_clothes),
        }
        write_csv(run_dir(run) / "sampler_preflight.csv", rows, list(rows[0]))
        run_audits[run] = {
            "probability_rows": len(table),
            "min_image_probability": min(item["image_probability"] for item in table.values()),
            "max_state_weight": max(item["weight"] for item in table.values()),
            "min_state_weight": min(item["weight"] for item in table.values()),
            "epochs_checked": len(rows),
            "all_steps_596": all(row["rank0_steps"] == STEPS_PER_EPOCH and row["rank1_steps"] == STEPS_PER_EPOCH for row in rows),
            "all_pids_present": all(row["all_77_pids_present"] for row in rows),
            "all_images_seen_by_epoch15": all_seen[run]["all_images_seen_by_epoch15"],
            "all_clothes_seen_by_epoch15": all_seen[run]["all_clothes_seen_by_epoch15"],
            "first_epoch_pid_plan_hashes": first_pid_hashes,
        }
    same_pid_plan = all(run_audits[run]["first_epoch_pid_plan_hashes"] == run_audits["P0"]["first_epoch_pid_plan_hashes"] for run in RUNS)
    if not same_pid_plan:
        raise RuntimeError("P0/P1/P2/P3 PID plans are not identical")
    # Persist a compact common-plan audit with no validation/test paths.
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "passed": True,
        "common_init_sha256": sha256_file(COMMON_INIT),
        "common_topology": topology,
        "same_model_topology": True,
        "same_pid_pair_plan": same_pid_plan,
        "pid_plan_hashes": pid_hashes,
        "same_steps_per_epoch": True,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_updates_screen": SCREEN_EPOCHS * STEPS_PER_EPOCH,
        "total_updates_formal": TOTAL_EPOCHS * STEPS_PER_EPOCH,
        "all_training_sources_only": True,
        "all_data_always_available": True,
        "zero_probability_images": False,
        "coverage_by_epoch15": all_seen,
        "runs": run_audits,
        "sampling_contract": {
            "beta": BETA,
            "weight_formula": "w=1+rank",
            "rank_range": [0.0, 1.0],
            "max_weight_ratio": 2.0,
            "pid_frequency_changes": False,
            "loss_weighting": False,
            "all_sample_loss_coefficient": 1.0,
        },
        "phase5a_group_metrics_sha256": context["phase5a_group_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "prepared_sha256": sha256_file(OUT_ROOT / "PHASE5C_PREPARED.json"),
    }
    write_json(OUT_ROOT / "PREFLIGHT.json", audit)
    write_json(OUT_ROOT / "preflight_summary.json", {"passed": True, "runs": run_audits, "coverage_by_epoch15": all_seen})
    print(json.dumps({"preflight": True, "passed": True, "audit": str(OUT_ROOT / "PREFLIGHT.json")}, indent=2, sort_keys=True))


def init_dist() -> tuple[int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 5C training requires exactly {WORLD_SIZE} visible PPUs")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(WORLD_SIZE)))
    if world != WORLD_SIZE:
        raise RuntimeError(f"torchrun world size must be {WORLD_SIZE}, got {world}")
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, torch.device(f"cuda:{rank}")


def cleanup_dist() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def logger_for(path: Path):
    logger = logging.getLogger(f"{PHASE}.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def save_model_checkpoint(raw_model: torch.nn.Module, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(p5b.cpu_state_dict(raw_model), temporary)
    temporary.replace(path)


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device):
    payload = {
        "phase": PHASE,
        "epoch": epoch,
        "global_step": global_step,
        "model": p5b.cpu_state_dict(model),
        "optimizer": p5b.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": p5b.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_rank_state(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    p5b.restore_rng(payload["rng"], device)
    return payload


def open_metrics(path: Path):
    existing = path.is_file() and path.stat().st_size > 0
    stream = path.open("a" if existing else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
    if not existing:
        writer.writeheader()
        stream.flush()
    return stream, writer


def make_loader(dataset: Any, transform_train: Any, plan: dict[str, Any]):
    loader = torch.utils.data.DataLoader(
        p5b.PlannedTrainDataset(plan["samples"], transform_train),
        batch_sampler=p5b.FixedBatchSampler(plan["batches"]),
        num_workers=1,
        pin_memory=False,
    )
    return loader


def aggregate_seen_before(
    run: str,
    end_epoch: int,
    dataset: Any,
    index_context: dict[str, Any],
    probability_table: dict[tuple[int, int], dict[str, Any]],
    pid_plans: dict[int, dict[int, list[int]]],
    rank_bases: dict[int, dict[int, dict[int, int]]],
) -> tuple[set[int], set[tuple[int, int]]]:
    seen_images: set[int] = set()
    seen_clothes: set[tuple[int, int]] = set()
    for epoch in range(1, end_epoch + 1):
        image_counts = Counter()
        for rank in range(WORLD_SIZE):
            plan = build_rank_plan(run, epoch, rank, dataset, index_context, probability_table, pid_plans, rank_bases)
            image_counts.update(plan["image_counts"])
        _, seen_images, seen_clothes = coverage_stats(image_counts, index_context, seen_images, seen_clothes)
    return seen_images, seen_clothes


def already_written_epoch(path: Path, epoch: int) -> bool:
    if not path.is_file():
        return False
    return any(int(row["epoch"]) == epoch for row in read_csv(path))


def train(run: str, requested_epochs: int, resume: bool = False) -> None:
    if run not in RUNS:
        raise ValueError(run)
    if requested_epochs not in (SCREEN_EPOCHS, TOTAL_EPOCHS):
        raise ValueError("Phase 5C permits only 15-epoch screen or 60-epoch P2 validation")
    prepared = load_prepared()
    preflight_path = OUT_ROOT / "PREFLIGHT.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError("Phase 5C preflight has not passed")
    if requested_epochs == TOTAL_EPOCHS:
        analysis_path = OUT_ROOT / "screen_analysis.json"
        if run != "P2" or not analysis_path.is_file() or not json.loads(analysis_path.read_text())["promotion_gate"]["passed"]:
            raise RuntimeError("60-epoch training is authorized only for P2 after a passed screen gate")
    output = run_dir(run)
    if not resume and (output / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"{run} already has a completed run; use --resume only for authorized P2 continuation")

    rank, device = init_dist()
    metrics_file = None
    amp_stream = None
    exposure_stream = None
    coverage_stream = None
    hardness_stream = None
    writer = None
    try:
        cfg = config()
        state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        model, loss_fn, optimizer, scheduler, scaler = p5b.setup_runtime(cfg, state, device)
        local_hash = sha256_bytes(json.dumps(p5b.tensor_hashes(p5b.cpu_state_dict(model)), sort_keys=True).encode())
        init_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(init_hashes, local_hash)
        if len(set(init_hashes)) != 1:
            raise RuntimeError("P0/P1/P2/P3 model initialization is not identical across ranks")

        context, dataset, index_context = load_context_and_dataset(cfg)
        transform_train, _ = p5b.build_img_transforms(cfg)
        probability_table = build_probability_table(run, context)
        pid_plans = make_pid_plans()
        _epoch_bases, rank_bases = pid_slot_bases(pid_plans)
        val_loader = p5b.build_val_loader(cfg, dataset)
        start_epoch = 1
        global_step = 0
        best_r1 = -float("inf")
        best_epoch = None
        previous_rows = read_csv(output / "metrics_per_epoch.csv") if (output / "metrics_per_epoch.csv").is_file() else []
        if resume:
            checkpoint = output / f"rank{rank}_state_epoch{SCREEN_EPOCHS:03d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = load_rank_state(checkpoint, model, optimizer, scheduler, scaler, device)
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
            if start_epoch != SCREEN_EPOCHS + 1 or len(previous_rows) < SCREEN_EPOCHS:
                raise RuntimeError("P2 continuation requires a complete 15-epoch checkpoint and metrics file")
            best_meta = json.loads((output / "best_state.json").read_text())
            best_r1 = float(best_meta["best_cc_rank1"])
            best_epoch = int(best_meta["best_epoch"])

        if rank == 0:
            logger = logger_for(output / "train_log_rank0.txt")
            writer = p5b.SummaryWriter(str(output / "TensorBoard"))
            metrics_file, metrics_writer = open_metrics(output / "metrics_per_epoch.csv")
            amp_path = output / "amp_statistics.csv"
            amp_existing = amp_path.is_file() and amp_path.stat().st_size > 0
            amp_stream = amp_path.open("a" if amp_existing else "w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            if not amp_existing:
                amp_writer.writeheader(); amp_stream.flush()
            exposure_path = output / "actual_exposure.csv"
            exposure_existing = exposure_path.is_file() and exposure_path.stat().st_size > 0
            exposure_stream = exposure_path.open("a" if exposure_existing else "w", newline="", encoding="utf-8")
            exposure_writer = csv.DictWriter(exposure_stream, fieldnames=EXPOSURE_FIELDS)
            if not exposure_existing:
                exposure_writer.writeheader(); exposure_stream.flush()
            coverage_path = output / "coverage_per_epoch.csv"
            coverage_existing = coverage_path.is_file() and coverage_path.stat().st_size > 0
            coverage_stream = coverage_path.open("a" if coverage_existing else "w", newline="", encoding="utf-8")
            coverage_writer = csv.DictWriter(coverage_stream, fieldnames=["run", "epoch", "epoch_unique_images", "epoch_unique_clothes", "epoch_unique_pids", "cumulative_unique_images", "cumulative_unique_clothes", "image_coverage", "clothing_coverage", "all_images_seen", "all_clothes_seen", "missing_image_count", "missing_clothes_count"])
            if not coverage_existing:
                coverage_writer.writeheader(); coverage_stream.flush()
            hardness_path = output / "hardness_exposure.csv"
            hardness_existing = hardness_path.is_file() and hardness_path.stat().st_size > 0
            hardness_stream = hardness_path.open("a" if hardness_existing else "w", newline="", encoding="utf-8")
            hardness_writer = csv.DictWriter(hardness_stream, fieldnames=["run", "epoch", "mean_sampled_h_id", "median_sampled_h_id", "p25_sampled_h_id", "p75_sampled_h_id", "mean_sampled_clothing_score", "sampled_state_count", "sampled_image_count"])
            if not hardness_existing:
                hardness_writer.writeheader(); hardness_stream.flush()
        else:
            metrics_writer = amp_writer = exposure_writer = coverage_writer = hardness_writer = None

        if start_epoch > 1:
            seen_images, seen_clothes = aggregate_seen_before(run, start_epoch - 1, dataset, index_context, probability_table, pid_plans, rank_bases)
        else:
            seen_images, seen_clothes = set(), set()

        for epoch in range(start_epoch, requested_epochs + 1):
            scheduler.step(epoch)
            plan = build_rank_plan(run, epoch, rank, dataset, index_context, probability_table, pid_plans, rank_bases)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan["meta"])
            if rank == 0:
                if any(meta["batches"] != STEPS_PER_EPOCH or meta["batch_unique_pid_min"] < 2 or len(meta["covered_pids"]) != NUM_CLASSES for meta in gathered_meta):
                    raise RuntimeError(f"runtime sampler audit failed for {run} epoch {epoch}")
                if [meta["pid_plan_sha256"] for meta in gathered_meta] != [pid_plan_hash(pid_plans[epoch][rank]) for rank in range(WORLD_SIZE)]:
                    raise RuntimeError("runtime PID plan hash mismatch")
            loader = make_loader(dataset, transform_train, plan)
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "id_acc", "bio_acc")}
            local_skips = 0
            local_nonfinite = 0
            scale_min = float(scaler.get_scale())
            scale_max = scale_min
            iterator = iter(loader)
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = p5b.forward_batch(model, loss_fn, batch, device, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    local_nonfinite += 1
                    raise RuntimeError(f"non-finite loss at {run} epoch {epoch} iteration {iteration}")
                scaler.scale(total).backward()
                scaler.step(optimizer)
                inf = p5b.found_inf(scaler, optimizer)
                local_skips += int(inf > 0)
                local_nonfinite += int(inf > 0)
                scaler.update()
                scale_min = min(scale_min, float(scaler.get_scale()))
                scale_max = max(scale_max, float(scaler.get_scale()))
                for key in epoch_sum:
                    epoch_sum[key] += p5b.scalar(result[key])
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", p5b.scalar(total), global_step)
            del iterator, loader
            gc.collect()
            reduced = p5b.reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); nonfinite_sum = amp_tensor[1].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM); dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_tensor = amp_tensor[2].clone(); max_tensor = amp_tensor[3].clone(); end_tensor = amp_tensor[4].clone()
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN); dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX); dist.all_reduce(end_tensor, op=dist.ReduceOp.MAX)
            amp_stats = {
                "skipped_steps": int(skip_sum.item()),
                "found_inf_count": int(skip_sum.item()),
                "nonfinite_events": int(nonfinite_sum.item()),
                "amp_scale_min": float(min_tensor.item()),
                "amp_scale_max": float(max_tensor.item()),
                "amp_scale_end": float(end_tensor.item()),
            }
            dist.barrier()
            metrics = p5b.evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger) if rank == 0 else None
            metrics_holder = [metrics]
            dist.broadcast_object_list(metrics_holder, src=0)
            metrics = metrics_holder[0]

            if rank == 0:
                global_state_counts = Counter()
                for meta in gathered_meta:
                    for state_key, count in meta["state_counts"].items():
                        pid, cid = (int(value) for value in state_key.split(":"))
                        global_state_counts[(pid, cid)] += int(count)
                global_image_counts = merge_counter_dicts(gathered_meta, "image_counts")
                exposure = exposure_rows(run, epoch, global_state_counts, global_image_counts, probability_table, context["records"], pid_sample_slots(pid_plans, epoch))
                if not already_written_epoch(output / "actual_exposure.csv", epoch):
                    exposure_writer.writerows(exposure); exposure_stream.flush()
                coverage, seen_images, seen_clothes = coverage_stats(global_image_counts, index_context, seen_images, seen_clothes)
                if not already_written_epoch(output / "coverage_per_epoch.csv", epoch):
                    coverage_writer.writerow({"run": run, "epoch": epoch, **coverage}); coverage_stream.flush()
                hstats = hardness_stats(global_state_counts, probability_table, context["records"])
                if not already_written_epoch(output / "hardness_exposure.csv", epoch):
                    hardness_writer.writerow({"run": run, "epoch": epoch, **hstats, "sampled_state_count": len(global_state_counts), "sampled_image_count": len(global_image_counts)}); hardness_stream.flush()
            else:
                coverage = hstats = None
            holders = [coverage, hstats]
            dist.broadcast_object_list(holders, src=0)
            coverage, hstats = holders
            if rank != 0:
                # Keep non-zero ranks in the same deterministic state for a
                # potential continuation; the saved state is authoritative.
                pass
            row = {
                "epoch": epoch,
                "train_loss": reduced["total"],
                "ce_loss": reduced["ce"],
                "triplet_loss": reduced["triplet"],
                "bio_loss": reduced["bio"],
                "lr": float(scheduler._get_lr(epoch)[0]),
                "lr_bias": float(scheduler._get_lr(epoch)[-1]),
                "id_accuracy": reduced["id_acc"],
                "bio_accuracy": reduced["bio_acc"],
                **amp_stats,
                "optimizer_updates": STEPS_PER_EPOCH,
                "active_images": 9576,
                "active_pids": NUM_CLASSES,
                "active_clothes": 256,
                "mean_clothes_per_pid": 256 / NUM_CLASSES,
                "new_clothes_count": 0,
                **metrics,
                **hstats,
                "unique_images_seen": coverage["cumulative_unique_images"],
                "unique_clothes_seen": coverage["cumulative_unique_clothes"],
                "image_coverage": coverage["image_coverage"],
                "clothing_coverage": coverage["clothing_coverage"],
            }
            if rank == 0:
                metrics_writer.writerow(row); metrics_file.flush()
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch)
                writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"]}); amp_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"]); best_epoch = epoch
                    save_model_checkpoint(model.module, output / "model_best.pth")
                    write_json(output / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1"})
            if epoch == SCREEN_EPOCHS or epoch == requested_epochs:
                save_rank_state(output / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    final_name = "model_screen_final.pth" if requested_epochs == SCREEN_EPOCHS else "model_final.pth"
                    save_model_checkpoint(model.module, output / final_name)
                    write_json(output / f"checkpoint_epoch{epoch:03d}.json", {"epoch": epoch, "global_step": global_step, "rank_states": [str(output / f"rank{r}_state_epoch{epoch:03d}.pt") for r in range(WORLD_SIZE)], "model": str(output / final_name)})
                    if epoch == requested_epochs:
                        rows_now = read_csv(output / "metrics_per_epoch.csv")
                        write_json(output / "RUN_COMPLETE.json", {
                            "phase": PHASE,
                            "run": run,
                            "completed_utc": utc_now(),
                            "epochs_completed": epoch,
                            "optimizer_updates": global_step,
                            "metrics": str(output / "metrics_per_epoch.csv"),
                            "best_epoch": best_epoch,
                            "best_cc_rank1": best_r1,
                            "best_model": str(output / "model_best.pth"),
                            "final_model": str(output / final_name),
                            "rows": len(rows_now),
                            "common_init_sha256": prepared["common_init"]["sha256"],
                            "sampling_probability_sha256": sha256_file(output / "sampling_probability.csv"),
                        })
        if rank == 0:
            write_json(output / "AMP.json", {"initial_scale": INITIAL_SCALE, "epochs_completed": requested_epochs, "metrics_source": str(output / "amp_statistics.csv")})
    finally:
        for stream in (metrics_file, amp_stream, exposure_stream, coverage_stream, hardness_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        cleanup_dist()


def finite_rows(rows: list[dict[str, str]]) -> bool:
    required = ["cc_rank1", "cc_mAP", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "mean_sampled_h_id"]
    return all(math.isfinite(float(row[key])) for row in rows for key in required)


def trajectory(rows: list[dict[str, str]], start: int = 8, end: int = 15) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"trajectory requires complete epochs {start}..{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) for row in selected], dtype=np.float64)
    mAP = np.asarray([float(row["cc_mAP"]) for row in selected], dtype=np.float64)
    return {
        "epochs": [int(value) for value in x],
        "mean_rank1": float(np.mean(r1)),
        "mean_mAP": float(np.mean(mAP)),
        "auc_rank1_normalized": float(np.trapz(r1, x) / (x[-1] - x[0])),
        "auc_mAP_normalized": float(np.trapz(mAP, x) / (x[-1] - x[0])),
        "slope_rank1_per_epoch": float(np.polyfit(x, r1, 1)[0]),
        "slope_mAP_per_epoch": float(np.polyfit(x, mAP, 1)[0]),
        "mean_sampled_h_id": float(np.mean([float(row["mean_sampled_h_id"]) for row in selected])),
        "mean_image_coverage": float(np.mean([float(row["image_coverage"]) for row in selected])),
        "mean_clothing_coverage": float(np.mean([float(row["clothing_coverage"]) for row in selected])),
        "rows": selected,
    }


def best_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    r1row = max(rows, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    maprow = max(rows, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {
        "best_rank1": float(r1row["cc_rank1"]),
        "best_mAP": float(maprow["cc_mAP"]),
        "best_rank1_epoch": int(r1row["epoch"]),
        "best_mAP_epoch": int(maprow["epoch"]),
        "best_rank1_row": r1row,
        "best_mAP_row": maprow,
        "final": rows[-1],
    }


def max_consecutive(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        "auc_rank1_delta": a["auc_rank1_normalized"] - b["auc_rank1_normalized"],
        "auc_mAP_delta": a["auc_mAP_normalized"] - b["auc_mAP_normalized"],
        "mean_rank1_delta": a["mean_rank1"] - b["mean_rank1"],
        "mean_mAP_delta": a["mean_mAP"] - b["mean_mAP"],
        "mean_sampled_h_id_delta": a["mean_sampled_h_id"] - b["mean_sampled_h_id"],
    }


def analysis_payload() -> dict[str, Any]:
    prepared = load_prepared()
    sources = {run: run_dir(run) / "metrics_per_epoch.csv" for run in RUNS}
    runs = {}
    for run, path in sources.items():
        rows = read_csv(path)
        if len(rows) < SCREEN_EPOCHS:
            raise RuntimeError(f"{run} requires a complete 15-epoch screen")
        runs[run] = {
            "rows": rows,
            "row_count": len(rows),
            "finite": finite_rows(rows),
            "trajectory_8_15": trajectory(rows),
            "best": best_metrics(rows),
        }
    t0, t1, t2, t3 = [runs[run]["trajectory_8_15"] for run in RUNS]
    c2_rows = {int(row["epoch"]): row for row in runs["P2"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    p0_rows = {int(row["epoch"]): row for row in runs["P0"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    late = []
    for epoch in range(12, 16):
        p2 = c2_rows[epoch]; p0 = p0_rows[epoch]
        r1_gap = float(p2["cc_rank1"]) - float(p0["cc_rank1"])
        map_gap = float(p2["cc_mAP"]) - float(p0["cc_mAP"])
        late.append({"epoch": epoch, "rank1_gap": r1_gap, "mAP_gap": map_gap, "p2_gt_p0_rank1": r1_gap > 0, "p2_gt_p0_mAP": map_gap > 0, "either": r1_gap > 0 or map_gap > 0, "both_within_minus_1pp": r1_gap >= -0.01 and map_gap >= -0.01})
    p2_vs_p0 = compare(t2, t0)
    p2_vs_p1 = compare(t2, t1)
    p2_vs_p3 = compare(t2, t3)
    p1_vs_p0 = compare(t1, t0)
    p3_vs_p0 = compare(t3, t0)
    bad_vs_p0 = [float(c2_rows[epoch]["cc_rank1"]) < float(p0_rows[epoch]["cc_rank1"]) - 0.01 or float(c2_rows[epoch]["cc_mAP"]) < float(p0_rows[epoch]["cc_mAP"]) - 0.01 for epoch in range(8, 16)]
    p2_promotion = {
        "finite": runs["P2"]["finite"],
        "auc_rank1_gt_p0": p2_vs_p0["auc_rank1_delta"] > 0,
        "auc_mAP_gt_p0": p2_vs_p0["auc_mAP_delta"] > 0,
        "auc_gt_p0_at_least_one": p2_vs_p0["auc_rank1_delta"] > 0 or p2_vs_p0["auc_mAP_delta"] > 0,
        "other_metric_safety_gate": max_consecutive(bad_vs_p0) < 3,
        "max_consecutive_over_1pp_drop_vs_p0": max_consecutive(bad_vs_p0),
        "epoch12_15": late,
        "epoch12_15_either_win_count": sum(item["either"] for item in late),
        "epoch12_15_win_gate": sum(item["either"] for item in late) >= 3,
        "p2_gt_p1_both_auc": p2_vs_p1["auc_rank1_delta"] > 0 and p2_vs_p1["auc_mAP_delta"] > 0,
        "p2_gt_p3_both_auc": p2_vs_p3["auc_rank1_delta"] > 0 and p2_vs_p3["auc_mAP_delta"] > 0,
    }
    p2_promotion["passed"] = bool(
        p2_promotion["finite"]
        and p2_promotion["auc_gt_p0_at_least_one"]
        and p2_promotion["other_metric_safety_gate"]
        and p2_promotion["epoch12_15_win_gate"]
        and p2_promotion["p2_gt_p1_both_auc"]
        and p2_promotion["p2_gt_p3_both_auc"]
    )
    p2_exposure_higher_than_all = t2["mean_sampled_h_id"] > t0["mean_sampled_h_id"] and t2["mean_sampled_h_id"] > t1["mean_sampled_h_id"] and t2["mean_sampled_h_id"] > t3["mean_sampled_h_id"]
    p2_better_than_p0_both = p2_vs_p0["auc_rank1_delta"] > 0 and p2_vs_p0["auc_mAP_delta"] > 0
    p2_better_than_controls = p2_promotion["p2_gt_p1_both_auc"] and p2_promotion["p2_gt_p3_both_auc"]
    # Q1 is about restoring full coverage relative to the active-set failure,
    # not about beating Bio on every metric.  PARTIAL means the old failure is
    # improved but uniform full-data remains better.
    if p2_better_than_p0_both:
        q1 = "YES"
    elif t2["mean_rank1"] > t1["mean_rank1"] and t2["mean_mAP"] > t1["mean_mAP"]:
        q1 = "PARTIAL"
    else:
        q1 = "NO"
    q2 = "YES" if p2_better_than_controls else "NO"
    if p2_exposure_higher_than_all and p2_better_than_p0_both:
        q3 = "YES"
    elif p2_exposure_higher_than_all and not (p2_vs_p0["auc_rank1_delta"] > 0 or p2_vs_p0["auc_mAP_delta"] > 0):
        q3 = "NO"
    else:
        q3 = "INCONCLUSIVE"
    if p2_promotion["passed"] and p2_better_than_p0_both:
        diagnosis = "FULL-DATA ID-DAMAGE SAMPLING WORKS"
    elif p2_better_than_controls and not p2_better_than_p0_both:
        diagnosis = "ID-DAMAGE RANKING HELPS BUT UNIFORM REMAINS BEST"
    elif p2_better_than_p0_both and not p2_better_than_controls:
        diagnosis = "ACTIVE-SET WITHHOLDING WAS THE MAIN PROBLEM"
    elif p2_exposure_higher_than_all and not (p2_vs_p0["auc_rank1_delta"] > 0 or p2_vs_p0["auc_mAP_delta"] > 0):
        diagnosis = "HARD-SAMPLE OVEREXPOSURE IS ITSELF HARMFUL"
    elif all(item["auc_rank1_delta"] <= 0 and item["auc_mAP_delta"] <= 0 for item in (p1_vs_p0, p2_vs_p0, p3_vs_p0)):
        diagnosis = "NONUNIFORM HARDNESS SAMPLING IS NOT BENEFICIAL"
    else:
        diagnosis = "RESULT INCONCLUSIVE"
    return {
        "phase": PHASE,
        "created_utc": utc_now(),
        "runs": runs,
        "comparisons": {
            "P2_vs_P0": p2_vs_p0,
            "P2_vs_P1": p2_vs_p1,
            "P2_vs_P3": p2_vs_p3,
            "P1_vs_P0": p1_vs_p0,
            "P3_vs_P0": p3_vs_p0,
        },
        "promotion_gate": p2_promotion,
        "questions": {
            "Q1_removing_clothing_withholding": q1,
            "Q2_id_damage_outperforms_clothing_and_random": q2,
            "Q3_high_id_damage_exposure_improves_reid": q3,
        },
        "final_diagnosis": diagnosis,
        "exposure_order": {
            "P2_mean_sampled_h_id_gt_P0_P1_P3": p2_exposure_higher_than_all,
            "P0": t0["mean_sampled_h_id"],
            "P1": t1["mean_sampled_h_id"],
            "P2": t2["mean_sampled_h_id"],
            "P3": t3["mean_sampled_h_id"],
        },
        "frozen_evidence": {
            "phase5a_id_damage_delta_h_id_prefix2": 0.003619,
            "phase5a_id_damage_ci": [0.002051, 0.005373],
            "phase5a_original_clothing_delta_h_id_prefix2": 0.000186,
            "phase5b2_active_set_diagnosis": "ACTIVE-SET CURRICULUM ITSELF IS HARMFUL",
        },
        "external_references": {
            "phase5b_bio_control": [BIO_CONTROL_R1, BIO_CONTROL_MAP],
            "phase5b_original_full": [ORIGINAL_FULL_R1, ORIGINAL_FULL_MAP],
            "phase5b_c0_metrics": str(P5B_C0_METRICS),
            "phase5b_c1_metrics": str(P5B_C1_METRICS),
            "phase5b_c0_run": str(P5B_C0_RUN),
            "phase5b_c1_run": str(P5B_C1_RUN),
        },
        "phase5c_prepared_sha256": sha256_file(OUT_ROOT / "PHASE5C_PREPARED.json"),
        "phase5c_preflight_sha256": sha256_file(OUT_ROOT / "PREFLIGHT.json"),
        "common_init_sha256": sha256_file(COMMON_INIT),
        "report_contract": {
            "sampling_weight_is_loss_weight": False,
            "all_images_probability_positive": True,
            "screen_window": "epochs 8-15",
            "p2_60_authorized_only_if_promotion_passed": True,
        },
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.4f}"


def signed_pp(value: float) -> str:
    return f"{value * 100:+.4f} pp"


def write_report(analysis: dict[str, Any]) -> None:
    labels = {
        "P0": ("Uniform", "100%", "none"),
        "P1": ("Clothing distance", "100%", "w=1+rank"),
        "P2": ("ID-Damage", "100%", "w=1+rank"),
        "P3": ("Random", "100%", "w=1+rank"),
    }
    decision_rows = []
    trajectory_rows = []
    for run in RUNS:
        t = analysis["runs"][run]["trajectory_8_15"]
        rank, coverage, priority = labels[run]
        status = "matched 15-epoch screen" if run in RUNS else ""
        decision_rows.append(f"| {run} | {rank} | {coverage} | {pct(t['auc_rank1_normalized'])} | {pct(t['auc_mAP_normalized'])} | {t['mean_sampled_h_id']:.6f} | {status} |")
        trajectory_rows.append(f"| {run} | {pct(t['mean_rank1'])} | {pct(t['mean_mAP'])} | {pct(t['auc_rank1_normalized'])} | {pct(t['auc_mAP_normalized'])} | {t['slope_rank1_per_epoch'] * 100:+.4f} | {t['slope_mAP_per_epoch'] * 100:+.4f} | {t['mean_image_coverage'] * 100:.3f}% | {t['mean_clothing_coverage'] * 100:.3f}% |")
    late_rows = []
    for row in analysis["promotion_gate"]["epoch12_15"]:
        late_rows.append(f"| {row['epoch']} | {signed_pp(row['rank1_gap'])} | {signed_pp(row['mAP_gap'])} | {row['either']} | {row['both_within_minus_1pp']} |")
    comp = analysis["comparisons"]
    lines = [
        "# PHASE 5C — FULL-DATA ID-DAMAGE-AWARE SAMPLING",
        "",
        f"Generated: `{analysis['created_utc']}`.",
        "",
        "This phase keeps every LTCC train image and clothing state available from epoch 1. P0 is rerun because the hierarchical sampler changes the ordinary uniform path; the old Phase-5B S0 remains the frozen external Bio reference.",
        "",
        "## CORE DECISION TABLE",
        "",
        "| Setting | Ranking | Coverage | E8–15 R1 AUC | E8–15 mAP AUC | Mean sampled H_ID | Status |",
        "|---|---|---:|---:|---:|---:|---|",
        *decision_rows,
        "",
        f"Frozen external Bio Control: **{BIO_CONTROL_R1 * 100:.4f} / {BIO_CONTROL_MAP * 100:.4f}**. Phase-5B Original Full remains secondary: **{ORIGINAL_FULL_R1 * 100:.4f} / {ORIGINAL_FULL_MAP * 100:.4f}**.",
        "",
        "## REPRESENTATION AND SAMPLING CONTRACT",
        "",
        f"- Common init: `{COMMON_INIT}`; SHA256 `{analysis['common_init_sha256']}`; copied byte-for-byte from Phase 5B common init.",
        f"- P1 frozen Clothing source: Phase-5A `clothing_text_fixed_h_cloth_loo_mean` from `{P5A_GROUPS}`; group metrics SHA256 `{json.loads((OUT_ROOT / 'PHASE5C_PREPARED.json').read_text())['phase5a_group_metrics_sha256']}`.",
        f"- P2 frozen identity source: Phase-5A `h_id_mean`, derived from the frozen Bio Control feature bank; no online recomputation.",
        "- Bio-only EVA02-Large, 224×224, 2 PPUs, batch 8, NUM_INSTANCES 2, seed 1234, SGD LR 2e-6, weight decay 0.05, CE + Triplet + Summary Bio, cosine scheduler, AMP initial scale 1024.",
        "- Reverse/GRL, DW, Reliability, Online H_ID, Full DIFFER, PRCC, loss weighting, new architecture/backbone/optimizer are OFF.",
        "- Every train image/state has positive probability. PID pair composition is shared. Only within-PID clothing-state priority changes.",
        "- `sampling weight != loss weight`: every sampled image has loss coefficient 1.0.",
        "",
        "## TRAJECTORY (EPOCHS 8–15)",
        "",
        "| Setting | Mean R1 | Mean mAP | R1 AUC | mAP AUC | R1 slope pp/epoch | mAP slope pp/epoch | Image coverage | Clothing coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *trajectory_rows,
        "",
        "AUC is the normalized trapezoidal trajectory average, not BEST-epoch selection.",
        "",
        "## EXPOSURE AND COVERAGE AUDIT",
        "",
        "- `PREFLIGHT.json` confirms identical PID plans, 596 updates/rank/epoch, 77 PIDs present, positive image probabilities, and 100% cumulative image/clothing coverage by the end of the 15-epoch plan.",
        "- The theoretical state probability is `p=(1+rank)/sum(1+rank)` with rank in [0,1], so the maximum state-weight ratio is 2:1. P0 is exactly uniform.",
        "- `actual_exposure.csv` records expected versus actual state exposure and exposure ratio. `coverage_per_epoch.csv` records cumulative image/state coverage. `hardness_exposure.csv` records sampled H_ID quantiles.",
        "- The deterministic reserve only guarantees coverage; the remaining draws use common random numbers, so P1/P2/P3 differ in state priority rather than data availability or PID frequency.",
        "",
        "## PRIMARY COMPARISONS",
        "",
        f"- P2 − P0 AUC: `{signed_pp(comp['P2_vs_P0']['auc_rank1_delta'])}` R1 / `{signed_pp(comp['P2_vs_P0']['auc_mAP_delta'])}` mAP; sampled H_ID delta `{comp['P2_vs_P0']['mean_sampled_h_id_delta']:+.6f}`.",
        f"- P2 − P1 AUC: `{signed_pp(comp['P2_vs_P1']['auc_rank1_delta'])}` R1 / `{signed_pp(comp['P2_vs_P1']['auc_mAP_delta'])}` mAP.",
        f"- P2 − P3 AUC: `{signed_pp(comp['P2_vs_P3']['auc_rank1_delta'])}` R1 / `{signed_pp(comp['P2_vs_P3']['auc_mAP_delta'])}` mAP.",
        "",
        "## P2 PROMOTION GATE",
        "",
        f"- Finite/no-collapse: **{analysis['promotion_gate']['finite']}**.",
        f"- At least one E8–15 AUC above P0: **{analysis['promotion_gate']['auc_gt_p0_at_least_one']}** (R1 `{analysis['promotion_gate']['auc_rank1_gt_p0']}`, mAP `{analysis['promotion_gate']['auc_mAP_gt_p0']}`).",
        f"- Other-metric safety: **{analysis['promotion_gate']['other_metric_safety_gate']}**; maximum consecutive >1 pp drop: `{analysis['promotion_gate']['max_consecutive_over_1pp_drop_vs_p0']}`.",
        f"- E12–15 at least one metric above P0: `{analysis['promotion_gate']['epoch12_15_either_win_count']}/4`; gate: **{analysis['promotion_gate']['epoch12_15_win_gate']}**.",
        f"- P2 > P1 on both AUC metrics: **{analysis['promotion_gate']['p2_gt_p1_both_auc']}**; P2 > P3 on both AUC metrics: **{analysis['promotion_gate']['p2_gt_p3_both_auc']}**.",
        f"- Promotion: **{'PASS' if analysis['promotion_gate']['passed'] else 'FAIL'}**.",
        "",
        "| Epoch | P2−P0 R1 | P2−P0 mAP | Either metric wins | Both within −1 pp |",
        "|---:|---:|---:|---:|---:|",
        *late_rows,
        "",
        "## QUESTIONS",
        "",
        f"- Q1 Does removing clothing withholding fix the curriculum failure? **{analysis['questions']['Q1_removing_clothing_withholding']}**.",
        f"- Q2 Does ID-Damage ranking outperform Clothing and Random under full-data coverage? **{analysis['questions']['Q2_id_damage_outperforms_clothing_and_random']}**.",
        f"- Q3 Does increasing exposure to high ID-Damage states improve ReID? **{analysis['questions']['Q3_high_id_damage_exposure_improves_reid']}**.",
        "",
        "## FINAL DIAGNOSIS",
        "",
        f"**{analysis['final_diagnosis']}**",
        "",
        f"P2 60-epoch continuation authorization: **{'PASS — only P2 may continue.' if analysis['promotion_gate']['passed'] else 'STOP — no 60-epoch continuation authorized.'}**",
        "",
        "This phase does not authorize DW, Reliability, Online H_ID, Full DIFFER, GRL, or PRCC.",
        "",
        "## ARTIFACTS",
        "",
        f"- Root: `{OUT_ROOT}`",
        f"- Preflight: `{OUT_ROOT / 'PREFLIGHT.json'}`",
        f"- Analysis: `{OUT_ROOT / 'screen_analysis.json'}`",
        f"- P0: `{run_dir('P0')}`; P1: `{run_dir('P1')}`; P2: `{run_dir('P2')}`; P3: `{run_dir('P3')}`",
        f"- Report: `{REPORT_PATH}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    analysis = analysis_payload()
    analysis["final_validation"] = analysis["runs"]["P2"]["row_count"] >= TOTAL_EPOCHS
    analysis["continue_to_60"] = bool(analysis["promotion_gate"]["passed"] and not analysis["final_validation"])
    write_json(OUT_ROOT / "screen_analysis.json", analysis)
    write_report(analysis)
    print(json.dumps({"analysis": str(OUT_ROOT / "screen_analysis.json"), "report": str(REPORT_PATH), "promotion_passed": analysis["promotion_gate"]["passed"], "continue_to_60": analysis["continue_to_60"], "diagnosis": analysis["final_diagnosis"]}, indent=2, sort_keys=True))
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "preflight", "train", "analyze"), required=True)
    parser.add_argument("--setting", choices=RUNS, default=None)
    parser.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "train":
        if args.setting is None:
            raise SystemExit("--setting is required for --mode train")
        train(args.setting, args.epochs, args.resume)
    else:
        analyze()


if __name__ == "__main__":
    main()
