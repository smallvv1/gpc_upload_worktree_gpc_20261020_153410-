#!/usr/bin/env python3
"""Phase 5B2: curriculum direction versus clothing-state coverage.

This is a deliberately narrow follow-up to Phase 5B.  C0 and C1 are reused
from the completed Phase 5B artifacts (C0 == S0 Bio full-data, C1 == S2
ID-Damage hard-first); only C2 and C3 are trained here.

The runner reuses the Phase 5B Bio-only model, loss, evaluation, sampler, and
AMP implementation.  It creates two new frozen schedules:

  C2  ID-Damage easy -> hard (ascending frozen H_ID)
  C3  random clothing order (seed 1234)

The schedule timing, clothing-state count, and optimizer budget are inherited
from the frozen Phase 5B C1 schedule.  Because clothing states have different
numbers of images, C2/C3 use a deterministic, stratified effective-image pool
whose per-epoch size is exactly C1's frozen active-image count.  Raw active
image counts are retained in the audit so this normalization is explicit.

No DW, Reliability, online H_ID, reverse/GRL, Full DIFFER, PRCC, loss
weighting, new network, backbone, optimizer, or seed sweep is implemented.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import logging
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
from scipy.stats import kendalltau
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.phase5b_id_damage_ds as p5b  # noqa: E402


PHASE = "phase5b2_curriculum_direction_coverage"
SEED = 1234
WORLD_SIZE = p5b.WORLD_SIZE
STEPS_PER_EPOCH = p5b.STEPS_PER_EPOCH
BATCH_SIZE = p5b.BATCH_SIZE
NUM_INSTANCES = p5b.NUM_INSTANCES
TOTAL_EPOCHS = p5b.TOTAL_EPOCHS
SCREEN_EPOCHS = p5b.SCREEN_EPOCHS
GLOBAL_PAIR_UNITS = p5b.GLOBAL_PAIR_UNITS
LOCAL_PAIR_UNITS = p5b.LOCAL_PAIR_UNITS
NUM_CLASSES = p5b.NUM_CLASSES
NUM_CAMERAS = p5b.NUM_CAMERAS

P5B_ROOT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds"
P5B_COMMON_INIT = P5B_ROOT / "common_init/COMMON_PHASE5B_INIT.pth"
P5B_COMMON_HASHES = P5B_ROOT / "common_init/COMMON_PHASE5B_INIT_HASHES.json"
P5B_C0_DIR = P5B_ROOT / "S0_bio_full_data"
P5B_C1_DIR = P5B_ROOT / "S2_id_damage_ds"
P5B_C1_SCHEDULE = P5B_ROOT / "schedules/schedule_S2.csv"
P5B_C1_STATES = P5B_ROOT / "schedules/active_states_S2.json"
P5B_C1_STATS = P5B_C1_DIR / "active_set_statistics_locked.csv"
P5B_C1_METRICS = P5B_C1_DIR / "metrics_per_epoch.csv"
P5B_C0_METRICS = P5B_C0_DIR / "metrics_per_epoch.csv"

OUT_ROOT = ROOT / "outputs/new_method_track/phase5b2_curriculum_direction_coverage"
SCHEDULE_ROOT = OUT_ROOT / "schedules"
RUN_DIRS = {"C2": "C2_easy_to_hard", "C3": "C3_random_order"}
RUNS = ("C2", "C3")
REPORT_PATH = ROOT / "reports/PHASE_5B2_CURRICULUM_DIRECTION_COVERAGE.md"

BIO_CONTROL_R1 = 0.5637755393981934
BIO_CONTROL_MAP = 0.32385736411857605
ORIGINAL_FULL_R1 = 0.5331632494926453
ORIGINAL_FULL_MAP = 0.292657

SCHEDULE_FIELDS = [
    "run", "event_id", "trace_index", "epoch", "pid",
    "expansion_position_for_pid", "active_clothes_before", "candidate_clothes",
    "new_clothes", "active_clothes_after", "ranking_signal",
    "ranking_description", "ranking_rule", "selected_ranking_score",
    "candidate_scores", "selected_h_id", "selected_delta_h_id_vs_c1",
    "selected_delta_h_id_vs_original_clothing", "selected_reliability",
    "cloth_key", "source_image_filenames", "source_image_paths",
    "random_seed", "random_order_position", "phase5a_replay_row_sha256",
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


def parse_json_value(value: Any, default: Any):
    if value is None:
        return default
    text = str(value).strip()
    if text in ("", "None", "null"):
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import ast
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return default


def int_list(value: Any) -> list[int]:
    parsed = parse_json_value(value, [])
    if isinstance(parsed, (int, float)):
        return [int(parsed)]
    return [int(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []


def set_seed(seed: int = SEED) -> None:
    # The underlying Phase 5B plan_seed intentionally remains the Phase 5B
    # value, so C2/C3 use the same augmentation-seed namespace as reused C1.
    p5b.set_seed(seed)


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


def save_git_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True, text=False).stdout
    (path / "git_diff.patch").write_bytes(diff)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    (path / "git_status.txt").write_text(status, encoding="utf-8")


def load_context():
    events, group_by_key, initial, all_clothes = p5b.load_frozen_events()
    if not P5B_C1_SCHEDULE.is_file() or not P5B_C1_STATES.is_file() or not P5B_C1_STATS.is_file():
        raise FileNotFoundError("Phase 5B C1 frozen schedule/state/statistics artifact is missing")
    c1_rows = read_csv(P5B_C1_SCHEDULE)
    if len(c1_rows) != len(events) or len(c1_rows) != 179:
        raise RuntimeError(f"expected 179 reused C1 schedule rows, got {len(c1_rows)}")
    c1_states_raw = json.loads(P5B_C1_STATES.read_text())
    c1_states = {
        int(epoch): {int(pid): [int(cid) for cid in values] for pid, values in state.items()}
        for epoch, state in c1_states_raw.items()
    }
    c1_stats = read_csv(P5B_C1_STATS)
    if len(c1_stats) != TOTAL_EPOCHS:
        raise RuntimeError(f"expected {TOTAL_EPOCHS} reused C1 active-set rows, got {len(c1_stats)}")
    return events, group_by_key, initial, all_clothes, c1_rows, c1_states, c1_stats


def c1_reference_stats(c1_stats: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    result = {}
    for row in c1_stats:
        epoch = int(row["epoch"])
        result[epoch] = {
            "epoch": epoch,
            "active_images": int(row["active_images"]),
            "active_pids": int(row["active_pids"]),
            "active_clothes": int(row["active_clothes"]),
            "mean_clothes_per_pid": float(row["mean_clothes_per_pid"]),
            "new_clothes_count": int(row["new_clothes_count"]),
            "active_state_sha256": row["active_state_sha256"],
        }
    return result


def random_orders(initial: dict[int, set[int]], all_clothes: dict[int, set[int]]) -> dict[int, list[int]]:
    rng = random.Random(SEED)
    orders = {}
    for pid in sorted(all_clothes):
        remaining = sorted(all_clothes[pid] - initial[pid])
        rng.shuffle(remaining)
        orders[pid] = remaining
    return orders


def selected_delta_vs_original(event: dict[str, str], selected_h_id: float) -> float:
    return float(selected_h_id) - float(event["original_selected_h_id"])


def schedule_row(
    run: str,
    event: dict[str, str],
    active: set[int],
    candidates: set[int],
    selected: int,
    scores: dict[int, float],
    group_by_key: dict[tuple[int, int], dict[str, str]],
    direction: str,
    random_order_position: int | None = None,
) -> dict[str, Any]:
    group = group_by_key[(int(event["pid"]), selected)]
    selected_h = float(group["h_id_mean"])
    if direction == "hard_first":
        description = "frozen H_ID(y,c) mean, descending"
        rule = "larger frozen H_ID(y,c); tie -> smaller cloth_id"
        signal = "id_damage_hard_first"
    elif direction == "easy_to_hard":
        description = "frozen H_ID(y,c) mean, ascending"
        rule = "smaller frozen H_ID(y,c); tie -> smaller cloth_id"
        signal = "id_damage_easy_to_hard"
    else:
        description = "pre-frozen random per-PID clothing order"
        rule = "random order fixed before training; seed 1234"
        signal = "random_order"
    source_paths = parse_json_value(group.get("image_paths_json"), [])
    source_names = parse_json_value(group.get("image_filenames_json"), [])
    return {
        "run": run,
        "event_id": int(event["event_id"]),
        "trace_index": int(event["trace_index"]),
        "epoch": int(event["epoch"]),
        "pid": int(event["pid"]),
        "expansion_position_for_pid": int(event["expansion_position_for_pid"]),
        "active_clothes_before": json.dumps(sorted(active), separators=(",", ":")),
        "candidate_clothes": json.dumps(sorted(candidates), separators=(",", ":")),
        "new_clothes": selected,
        "active_clothes_after": json.dumps(sorted(active | {selected}), separators=(",", ":")),
        "ranking_signal": signal,
        "ranking_description": description,
        "ranking_rule": rule,
        "selected_ranking_score": selected_h if direction != "random" else "",
        "candidate_scores": json.dumps({str(k): scores[k] for k in sorted(scores)}, separators=(",", ":")),
        "selected_h_id": selected_h,
        "selected_delta_h_id_vs_c1": "",
        "selected_delta_h_id_vs_original_clothing": selected_delta_vs_original(event, selected_h),
        "selected_reliability": float(group["reliability"]),
        "cloth_key": group["cloth_key"],
        "source_image_filenames": json.dumps(source_names, separators=(",", ":")),
        "source_image_paths": json.dumps(source_paths, separators=(",", ":")),
        "random_seed": SEED if direction == "random" else "",
        "random_order_position": random_order_position if random_order_position is not None else "",
        "phase5a_replay_row_sha256": sha256_bytes(json.dumps(event, sort_keys=True).encode()),
    }


def build_direction_schedule(
    run: str,
    direction: str,
    events: list[dict[str, str]],
    group_by_key: dict[tuple[int, int], dict[str, str]],
    initial: dict[int, set[int]],
    all_clothes: dict[int, set[int]],
    orders: dict[int, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[int]]]]:
    state = {pid: set(values) for pid, values in initial.items()}
    rows = []
    random_positions = Counter()
    for event in events:
        pid = int(event["pid"])
        active = set(state[pid])
        candidates = all_clothes[pid] - active
        if not candidates:
            raise RuntimeError(f"{run} has no candidate at event {event['event_id']}")
        scores = {cid: float(group_by_key[(pid, cid)]["h_id_mean"]) for cid in candidates}
        if direction == "hard_first":
            selected = min(candidates, key=lambda cid: (-scores[cid], cid))
            random_position = None
        elif direction == "easy_to_hard":
            selected = min(candidates, key=lambda cid: (scores[cid], cid))
            random_position = None
        elif direction == "random":
            if orders is None:
                raise RuntimeError("random order was not initialized")
            random_position = random_positions[pid]
            selected = orders[pid][random_position]
            if selected not in candidates:
                raise RuntimeError(f"random order repeated a clothing state at event {event['event_id']}")
            random_positions[pid] += 1
        else:
            raise ValueError(direction)
        rows.append(schedule_row(run, event, active, candidates, selected, scores, group_by_key, direction, random_position))
        state[pid].add(selected)
    states = active_states_from_rows(initial, rows)
    return rows, states


def normalize_c1_rows(c1_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for source in c1_rows:
        row = {key: source.get(key, "") for key in SCHEDULE_FIELDS}
        row["run"] = "C1"
        row["ranking_signal"] = "id_damage_hard_first"
        row["ranking_description"] = "frozen H_ID(y,c) mean, descending; reused Phase 5B S2"
        row["ranking_rule"] = "larger frozen H_ID(y,c); tie -> smaller cloth_id"
        row["selected_delta_h_id_vs_c1"] = 0.0
        row["selected_delta_h_id_vs_original_clothing"] = source.get("selected_delta_h_id_vs_historical", "")
        row["random_seed"] = ""
        row["random_order_position"] = ""
        rows.append(row)
    return rows


def active_states_from_rows(initial: dict[int, set[int]], rows: list[dict[str, Any]]) -> dict[int, dict[int, list[int]]]:
    by_epoch = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(row)
    current = {pid: set(values) for pid, values in initial.items()}
    result = {}
    for epoch in range(1, TOTAL_EPOCHS + 1):
        for row in by_epoch[epoch]:
            current[int(row["pid"])].add(int(row["new_clothes"]))
        result[epoch] = {pid: sorted(values) for pid, values in sorted(current.items())}
    return result


def raw_active_stats(
    run: str,
    states: dict[int, dict[int, list[int]]],
    schedule_rows: list[dict[str, Any]],
    group_by_key: dict[tuple[int, int], dict[str, str]],
) -> list[dict[str, Any]]:
    result = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        state = states[epoch]
        counts = [len(values) for values in state.values()]
        raw_images = sum(int(group_by_key[(pid, cid)]["support"]) for pid, values in state.items() for cid in values)
        result.append({
            "run": run,
            "epoch": epoch,
            "raw_active_images": raw_images,
            "active_pids": len(state),
            "active_clothes": sum(counts),
            "mean_clothes_per_pid": float(np.mean(counts)),
            "new_clothes_count": sum(int(row["epoch"]) == epoch for row in schedule_rows),
            "raw_active_state_sha256": p5b.active_state_hash(state),
        })
    return result


def make_common_stats(
    run: str,
    raw_rows: list[dict[str, Any]],
    c1_targets: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in raw_rows:
        epoch = int(row["epoch"])
        target = c1_targets[epoch]
        if row["active_pids"] != target["active_pids"] or row["active_clothes"] != target["active_clothes"]:
            raise RuntimeError(f"{run} clothing-state size differs from C1 at epoch {epoch}")
        if abs(float(row["mean_clothes_per_pid"]) - target["mean_clothes_per_pid"]) > 1e-12:
            raise RuntimeError(f"{run} mean clothes/PID differs from C1 at epoch {epoch}")
        result.append({
            "run": run,
            "epoch": epoch,
            "active_images": target["active_images"],
            "raw_active_images": row["raw_active_images"],
            "active_pids": row["active_pids"],
            "active_clothes": row["active_clothes"],
            "mean_clothes_per_pid": row["mean_clothes_per_pid"],
            "new_clothes_count": row["new_clothes_count"],
            "active_state_sha256": row["raw_active_state_sha256"],
            "coverage_target_source": "Phase 5B C1 frozen active_images",
            "coverage_sampling": "deterministic stratified effective image pool",
        })
    return result


def exposure_seed(run: str, epoch: int, pid: int, cid: int) -> int:
    token = f"{PHASE}|{SEED}|effective_pool|{run}|{epoch}|{pid}|{cid}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


def build_effective_index_map(
    run: str,
    epoch: int,
    active_state: dict[int, list[int]],
    index_map: dict[tuple[int, int], list[int]],
    target_images: int,
) -> dict[tuple[int, int], list[int]]:
    """Build an exactly target-sized image pool while retaining every state.

    Each active clothing state receives at least two images, then the remaining
    quota is allocated by a deterministic Hamilton allocation proportional to
    each state's available image capacity.  This is a control for support-size
    differences, not a model signal; H_ID never enters this allocation.
    """
    keys = [(int(pid), int(cid)) for pid, clothes in sorted(active_state.items()) for cid in sorted(clothes)]
    raw = {key: list(index_map[key]) for key in keys}
    raw_total = sum(len(values) for values in raw.values())
    minimum_total = 2 * len(keys)
    if target_images < minimum_total:
        raise RuntimeError(f"target active image count {target_images} cannot retain two images per state ({minimum_total})")
    if any(len(values) < 2 for values in raw.values()):
        raise RuntimeError("active clothing state has fewer than two source images")
    quotas = {key: min(2, len(raw[key])) for key in keys}
    if target_images <= raw_total:
        remaining = target_images - sum(quotas.values())
        capacities = {key: len(raw[key]) - quotas[key] for key in keys}
        capacity_total = sum(capacities.values())
        exact = {key: (remaining * capacities[key] / capacity_total if capacity_total else 0.0) for key in keys}
        floors = {key: min(capacities[key], int(math.floor(exact[key]))) for key in keys}
        for key in keys:
            quotas[key] += floors[key]
        leftover = target_images - sum(quotas.values())
        order = sorted(keys, key=lambda key: (-(exact[key] - math.floor(exact[key])), key[0], key[1]))
        for key in order:
            if leftover <= 0:
                break
            if quotas[key] < len(raw[key]):
                quotas[key] += 1
                leftover -= 1
        if leftover != 0:
            raise RuntimeError("Hamilton effective-pool allocation did not reach target")
    else:
        # This branch is not used during the 15-epoch screen for C2/C3, but it
        # keeps the authorized C2 60-epoch continuation deterministic if a
        # schedule has fewer raw images than the C1 target.
        quotas = {key: len(raw[key]) for key in keys}
    effective = {}
    for key in keys:
        values = list(raw[key])
        rng = random.Random(exposure_seed(run, epoch, key[0], key[1]))
        rng.shuffle(values)
        chosen = values[:quotas[key]]
        effective[key] = chosen
    if target_images > raw_total:
        ordered = list(keys)
        cursor = 0
        while sum(len(values) for values in effective.values()) < target_images:
            key = ordered[cursor % len(ordered)]
            effective[key].append(raw[key][cursor % len(raw[key])])
            cursor += 1
    if sum(len(values) for values in effective.values()) != target_images:
        raise RuntimeError("effective image pool size mismatch")
    if any(len(values) < 2 for values in effective.values()):
        raise RuntimeError("effective pool dropped a clothing state's hard-negative support")
    return effective


def prepare() -> None:
    if (OUT_ROOT / "PHASE5B2_PREPARED.json").exists():
        raise RuntimeError("Phase 5B2 artifacts already prepared; refusing to overwrite frozen schedules")
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(f"Phase 5B2 output directory is not empty: {OUT_ROOT}")
    events, group_by_key, initial, all_clothes, c1_source, c1_states, c1_stats = load_context()
    targets = c1_reference_stats(c1_stats)
    orders = random_orders(initial, all_clothes)
    c1_rows = normalize_c1_rows(c1_source)
    c2_rows, c2_states = build_direction_schedule("C2", "easy_to_hard", events, group_by_key, initial, all_clothes)
    c3_rows, c3_states = build_direction_schedule("C3", "random", events, group_by_key, initial, all_clothes, orders)
    # Verify the reused Phase 5B C1 sequence is exactly hard-first before
    # materializing the three-way comparison artifact.
    hard_rows, hard_states = build_direction_schedule("C1_RECOMPUTED", "hard_first", events, group_by_key, initial, all_clothes)
    if [int(row["new_clothes"]) for row in hard_rows] != [int(row["new_clothes"]) for row in c1_source]:
        raise RuntimeError("reused Phase 5B C1 sequence is not the frozen descending H_ID sequence")
    for epoch in range(1, TOTAL_EPOCHS + 1):
        if hard_states[epoch] != c1_states[epoch]:
            raise RuntimeError(f"reused Phase 5B C1 active state differs at epoch {epoch}")
    schedules = {"C1": c1_rows, "C2": c2_rows, "C3": c3_rows}
    states = {"C1": c1_states, "C2": c2_states, "C3": c3_states}
    raw_rows = {
        "C1": raw_active_stats("C1", c1_states, c1_rows, group_by_key),
        "C2": raw_active_stats("C2", c2_states, c2_rows, group_by_key),
        "C3": raw_active_stats("C3", c3_states, c3_rows, group_by_key),
    }
    matched_stats = {
        run: make_common_stats(run, raw_rows[run], targets) for run in ("C1", "C2", "C3")
    }

    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    for run, rows in schedules.items():
        write_csv(SCHEDULE_ROOT / f"schedule_{run}.csv", rows, SCHEDULE_FIELDS)
        write_json(SCHEDULE_ROOT / f"active_states_{run}.json", {str(epoch): state for epoch, state in states[run].items()})
        write_csv(SCHEDULE_ROOT / f"active_set_statistics_{run}.csv", matched_stats[run], list(matched_stats[run][0]))
    write_json(SCHEDULE_ROOT / "initial_pool.json", {str(pid): sorted(values) for pid, values in sorted(initial.items())})
    write_json(SCHEDULE_ROOT / "all_clothes_by_pid.json", {str(pid): sorted(values) for pid, values in sorted(all_clothes.items())})
    write_json(SCHEDULE_ROOT / "random_orders_seed1234.json", {str(pid): values for pid, values in sorted(orders.items())})

    cfg = config()
    common_meta = json.loads(P5B_COMMON_HASHES.read_text()) if P5B_COMMON_HASHES.is_file() else {}
    for run in RUNS:
        path = run_dir(run)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yml").write_text(cfg.dump(), encoding="utf-8")
        save_git_snapshot(path)
        write_json(path / "schedule_hash.json", {
            "run": run,
            "schedule_file": str(SCHEDULE_ROOT / f"schedule_{run}.csv"),
            "schedule_sha256": sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv"),
            "active_set_statistics": str(SCHEDULE_ROOT / f"active_set_statistics_{run}.csv"),
            "frozen": True,
        })
        write_json(path / "run_manifest.json", {
            "phase": PHASE,
            "run": run,
            "label": "ID-Damage Easy-to-Hard" if run == "C2" else "Random Curriculum",
            "ranking": "frozen H_ID(y,c) clothing-state mean" if run == "C2" else "none (pre-frozen random order)",
            "direction": "easy_to_hard" if run == "C2" else "random",
            "restricted_coverage": "same effective active image count as reused C1; raw support retained in audit",
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "online_h_id": "OFF",
            "random_seed": SEED if run == "C3" else None,
            "common_init": {
                "path": str(P5B_COMMON_INIT),
                "sha256": sha256_file(P5B_COMMON_INIT),
                "state_key_count": common_meta.get("state_key_count"),
                "state_numel": common_meta.get("state_numel"),
            },
            "schedule_file": str(SCHEDULE_ROOT / f"schedule_{run}.csv"),
            "schedule_sha256": sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv"),
            "reused_phase5b_c1_schedule": str(P5B_C1_SCHEDULE),
            "reused_phase5b_c1_schedule_sha256": sha256_file(P5B_C1_SCHEDULE),
            "screen_epochs": SCREEN_EPOCHS,
            "total_epochs": TOTAL_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "optimizer_updates": TOTAL_EPOCHS * STEPS_PER_EPOCH,
            "data": {"train_pids": NUM_CLASSES, "train_clothes": len(group_by_key)},
            "initial_active_state_sha256": p5b.active_state_hash(initial),
        })
    write_json(OUT_ROOT / "C0_reference.json", {
        "setting": "C0",
        "reused_phase5b_run": "S0",
        "metrics": str(P5B_C0_METRICS),
        "run_dir": str(P5B_C0_DIR),
        "run_complete": str(P5B_C0_DIR / "RUN_COMPLETE.json"),
    })
    write_json(OUT_ROOT / "C1_reference.json", {
        "setting": "C1",
        "reused_phase5b_run": "S2",
        "metrics": str(P5B_C1_METRICS),
        "run_dir": str(P5B_C1_DIR),
        "schedule": str(P5B_C1_SCHEDULE),
        "schedule_sha256": sha256_file(P5B_C1_SCHEDULE),
        "active_states": str(P5B_C1_STATES),
        "active_stats": str(P5B_C1_STATS),
    })
    write_json(OUT_ROOT / "PHASE5B2_PREPARED.json", {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "reused_phase5b": {
            "root": str(P5B_ROOT),
            "common_init": str(P5B_COMMON_INIT),
            "common_init_sha256": sha256_file(P5B_COMMON_INIT),
            "C0_metrics": str(P5B_C0_METRICS),
            "C1_metrics": str(P5B_C1_METRICS),
            "C1_schedule": str(P5B_C1_SCHEDULE),
            "C1_schedule_sha256": sha256_file(P5B_C1_SCHEDULE),
            "C1_stats": str(P5B_C1_STATS),
        },
        "event_count": len(events),
        "initial_pid_count": len(initial),
        "schedule_hashes": {run: sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv") for run in schedules},
        "contract": {
            "only_variable": "curriculum direction/order after the common initial clothing state",
            "C1": "reused ID-Damage hard-first",
            "C2": "frozen ID-Damage easy-to-hard",
            "C3": "frozen random per-PID order, seed 1234",
            "same_initial_state": True,
            "same_event_epochs_and_counts": True,
            "same_active_pids_clothes_and_mean_clothes": True,
            "same_effective_active_images": True,
            "effective_image_target": "reused C1 frozen active_images per epoch",
            "optimizer_budget": STEPS_PER_EPOCH,
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "online_h_id": "OFF",
            "test_leakage": "forbidden for schedule construction",
        },
    })
    print(json.dumps({"prepared": True, "output": str(OUT_ROOT), "schedule_hashes": {run: sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv") for run in schedules}}, indent=2, sort_keys=True))


def load_prepared() -> dict[str, Any]:
    path = OUT_ROOT / "PHASE5B2_PREPARED.json"
    if not path.is_file():
        raise FileNotFoundError("run prepare first: " + str(path))
    return json.loads(path.read_text())


def load_states(run: str) -> dict[int, dict[int, list[int]]]:
    path = SCHEDULE_ROOT / f"active_states_{run}.json"
    raw = json.loads(path.read_text())
    return {int(epoch): {int(pid): [int(cid) for cid in values] for pid, values in state.items()} for epoch, state in raw.items()}


def load_schedule(run: str) -> list[dict[str, str]]:
    return read_csv(SCHEDULE_ROOT / f"schedule_{run}.csv")


def sequence_by_pid(rows: list[dict[str, str]]) -> dict[int, list[int]]:
    result = defaultdict(list)
    for row in rows:
        result[int(row["pid"])].append(int(row["new_clothes"]))
    return dict(result)


def schedule_audit(initial: dict[int, set[int]], all_clothes: dict[int, set[int]], group_by_key: dict[tuple[int, int], dict[str, str]]) -> dict[str, Any]:
    schedules = {run: load_schedule(run) for run in ("C1", "C2", "C3")}
    keys = ["event_id", "trace_index", "epoch", "pid", "expansion_position_for_pid"]
    event_keys_equal = all(
        [tuple(row[key] for key in keys) for row in schedules["C1"]]
        == [tuple(row[key] for key in keys) for row in schedules[run]]
        for run in ("C2", "C3")
    )
    if not event_keys_equal:
        raise RuntimeError("C1/C2/C3 event epoch/PID/position keys differ")
    sequences = {run: sequence_by_pid(schedules[run]) for run in schedules}
    pid_rows = []
    for pid in sorted(sequences["C1"]):
        common = sorted(set(sequences["C1"][pid]) & set(sequences["C2"][pid]))
        if len(common) >= 3:
            rank_c1 = [sequences["C1"][pid].index(cid) for cid in common]
            rank_c2 = [sequences["C2"][pid].index(cid) for cid in common]
            tau = kendalltau(rank_c1, rank_c2).statistic
            pid_rows.append({"pid": pid, "common_clothes": len(common), "kendall_c1_c2": float(tau)})
    position_changes = {}
    for position in sorted({int(row["expansion_position_for_pid"]) for row in schedules["C1"]}):
        by_run = {
            run: {int(row["pid"]): int(row["new_clothes"]) for row in schedules[run] if int(row["expansion_position_for_pid"]) == position}
            for run in schedules
        }
        position_changes[str(position)] = {
            "pid_count": len(by_run["C1"]),
            "C1_vs_C2_different": sum(by_run["C1"][pid] != by_run["C2"][pid] for pid in by_run["C1"]),
            "C1_vs_C3_different": sum(by_run["C1"][pid] != by_run["C3"][pid] for pid in by_run["C1"]),
        }
    full_rank_audit = []
    c2_is_exact_reverse = True
    c1_is_descending = True
    c2_is_ascending = True
    for pid in sorted(all_clothes):
        remaining = all_clothes[pid] - initial[pid]
        h = {cid: float(group_by_key[(pid, cid)]["h_id_mean"]) for cid in remaining}
        desc = sorted(remaining, key=lambda cid: (-h[cid], cid))
        asc = sorted(remaining, key=lambda cid: (h[cid], cid))
        # A PID with one clothing state has no expansion event and therefore
        # is intentionally absent from sequence_by_pid().
        c1_seq = sequences["C1"].get(pid, [])
        c2_seq = sequences["C2"].get(pid, [])
        c1_is_descending &= c1_seq == desc
        c2_is_ascending &= c2_seq == asc
        c2_is_exact_reverse &= c2_seq == list(reversed(c1_seq)) if len(c1_seq) == len(desc) else False
        full_rank_audit.append({"pid": pid, "h_id_descending": desc, "h_id_ascending": asc, "c1": c1_seq, "c2": c2_seq, "c3": sequences["C3"].get(pid, [])})
    return {
        "event_keys_identical": event_keys_equal,
        "event_count": len(schedules["C1"]),
        "event_epoch_counts": {run: dict(sorted(Counter(int(row["epoch"]) for row in schedules[run]).items())) for run in schedules},
        "position_changes": position_changes,
        "c1_c2_pid_kendall": {
            "count": len(pid_rows),
            "mean": float(np.mean([row["kendall_c1_c2"] for row in pid_rows])) if pid_rows else None,
            "median": float(np.median([row["kendall_c1_c2"] for row in pid_rows])) if pid_rows else None,
            "rows": pid_rows,
        },
        "c1_is_exact_hard_first": c1_is_descending,
        "c2_is_exact_easy_to_hard": c2_is_ascending,
        "c2_is_reverse_of_c1": c2_is_exact_reverse,
        "full_pid_rankings": full_rank_audit,
        "random_seed": SEED,
    }


def sampler_preflight(cfg: Any, run: str, target_stats: dict[int, dict[str, Any]], epochs: int = SCREEN_EPOCHS) -> list[dict[str, Any]]:
    dataset = p5b.build_dataset(cfg)
    index_map = p5b.build_index_map(dataset)
    states = load_states(run)
    rows = []
    for epoch in range(1, epochs + 1):
        effective = build_effective_index_map(run, epoch, states[epoch], index_map, target_stats[epoch]["active_images"])
        global_pairs = p5b.make_pair_units(states[epoch], effective, epoch)
        rank_meta = []
        for rank in range(WORLD_SIZE):
            rank_pairs = global_pairs[rank::WORLD_SIZE]
            batches = [rank_pairs[start:start + BATCH_SIZE // NUM_INSTANCES] for start in range(0, len(rank_pairs), BATCH_SIZE // NUM_INSTANCES)]
            batch_pids = [
                {int(dataset.train[index]["pid"]) for pair in batch for index in pair}
                for batch in batches
            ]
            rank_pids = set().union(*batch_pids) if batch_pids else set()
            rank_meta.append({
                "rank": rank,
                "pair_units": len(rank_pairs),
                "batches": len(batches),
                "pair_size_ok": all(len(pair) == NUM_INSTANCES and len({int(dataset.train[index]["pid"]) for index in pair}) == 1 for pair in rank_pairs),
                "batch_unique_pid_min": min(len(pids) for pids in batch_pids),
                "covered_pids": sorted(rank_pids),
                "effective_active_images": sum(len(values) for values in effective.values()),
                "plan_sha256": sha256_bytes(json.dumps(rank_pairs, separators=(",", ":")).encode()),
            })
        if any(
            item["batches"] != STEPS_PER_EPOCH
            or not item["pair_size_ok"]
            or item["batch_unique_pid_min"] < 2
            for item in rank_meta
        ):
            raise RuntimeError(f"sampler preflight failed for {run} epoch {epoch}")
        rows.append({
            "run": run,
            "epoch": epoch,
            "target_active_images": target_stats[epoch]["active_images"],
            "effective_active_images": sum(len(values) for values in effective.values()),
            "active_pids": len(states[epoch]),
            "active_clothes": sum(len(values) for values in states[epoch].values()),
            "global_pair_units": len(global_pairs),
            "expected_global_pair_units": GLOBAL_PAIR_UNITS,
            "rank0": json.dumps(rank_meta[0], sort_keys=True, separators=(",", ":")),
            "rank1": json.dumps(rank_meta[1], sort_keys=True, separators=(",", ":")),
            "all_77_pids_present": len(states[epoch]) == NUM_CLASSES,
        })
    return rows


def preflight() -> None:
    prepared = load_prepared()
    events, group_by_key, initial, all_clothes, c1_rows, c1_states, c1_stats = load_context()
    targets = c1_reference_stats(c1_stats)
    cfg = config()
    common_state = torch.load(P5B_COMMON_INIT, map_location="cpu", weights_only=False)
    common_hashes = json.loads(P5B_COMMON_HASHES.read_text())
    if sha256_file(P5B_COMMON_INIT) != common_hashes["state_sha256"]:
        raise RuntimeError("reused Phase 5B common init SHA256 mismatch")
    if p5b.tensor_hashes(common_state) != common_hashes["tensor_hashes"]:
        raise RuntimeError("reused Phase 5B common init tensor hash mismatch")
    model = p5b.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=NUM_CAMERAS)
    model.load_state_dict(common_state, strict=True)
    topology = {
        "model_name": cfg.MODEL.NAME,
        "state_key_count": len(model.state_dict()),
        "state_numel": int(sum(value.numel() for value in model.state_dict().values())),
        "loss_type": list(cfg.MODEL.LOSS_TYPE),
        "has_reverse_parameters": any("Reverse" in key or "reverse" in key for key in model.state_dict()),
    }
    del model, common_state
    gc.collect()
    if topology["has_reverse_parameters"] or topology["loss_type"] != ["ce", "triplet", "clipBio"]:
        raise RuntimeError("C2/C3 topology is not Bio-only")
    schedule_audit_result = schedule_audit(initial, all_clothes, group_by_key)
    if not schedule_audit_result["c1_is_exact_hard_first"] or not schedule_audit_result["c2_is_exact_easy_to_hard"]:
        raise RuntimeError("C1/C2 direction audit failed")
    stats_audit = {}
    for run in ("C1", "C2", "C3"):
        stats = read_csv(SCHEDULE_ROOT / f"active_set_statistics_{run}.csv")
        stats_audit[run] = stats
    for epoch in range(1, TOTAL_EPOCHS + 1):
        reference = stats_audit["C1"][epoch - 1]
        for run in ("C2", "C3"):
            row = stats_audit[run][epoch - 1]
            for key in ("active_images", "active_pids", "active_clothes", "mean_clothes_per_pid", "new_clothes_count"):
                if float(row[key]) != float(reference[key]):
                    raise RuntimeError(f"coverage-size mismatch at epoch {epoch}: {run} {key}")
    source_paths = []
    for run in ("C1", "C2", "C3"):
        for row in load_schedule(run):
            source_paths.extend(parse_json_value(row["source_image_paths"], []))
    train_root = Path("/home/data/LTCC_ReID/train")
    no_test_sources = all(str(path).startswith(str(train_root) + "/") for path in source_paths)
    if not no_test_sources:
        raise RuntimeError("schedule contains a non-train source")
    sampler_audit = {}
    for run in ("C2", "C3"):
        rows = sampler_preflight(cfg, run, targets)
        sampler_audit[run] = rows
        write_csv(run_dir(run) / "sampler_preflight.csv", rows, list(rows[0]))
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "passed": True,
        "reused_phase5b": prepared["reused_phase5b"],
        "common_topology": topology,
        "common_init_sha256": sha256_file(P5B_COMMON_INIT),
        "event_count": len(events),
        "initial_active_state_sha256": p5b.active_state_hash(initial),
        "event_keys_equal": schedule_audit_result["event_keys_identical"],
        "schedule_hashes": {run: sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv") for run in ("C1", "C2", "C3")},
        "schedule_audit": schedule_audit_result,
        "coverage_size_equal": True,
        "coverage_size_definition": "C2/C3 effective active image pool is exactly reused C1 active_images per epoch; raw counts retained",
        "no_test_sources": no_test_sources,
        "optimization_budget": {
            "steps_per_epoch": STEPS_PER_EPOCH,
            "epochs": TOTAL_EPOCHS,
            "total_updates": TOTAL_EPOCHS * STEPS_PER_EPOCH,
            "global_pair_units_per_epoch": GLOBAL_PAIR_UNITS,
            "local_pair_units_per_rank": LOCAL_PAIR_UNITS,
            "batch_size": BATCH_SIZE,
            "num_instances": NUM_INSTANCES,
        },
        "sampler_preflight": {
            run: {
                "epochs_checked": len(rows),
                "all_steps_equal": all(
                    json.loads(row["rank0"])["batches"] == STEPS_PER_EPOCH
                    and json.loads(row["rank1"])["batches"] == STEPS_PER_EPOCH
                    and json.loads(row["rank0"])["batch_unique_pid_min"] >= 2
                    and json.loads(row["rank1"])["batch_unique_pid_min"] >= 2
                    for row in rows
                ),
                "all_77_active_pids": all(row["all_77_pids_present"] for row in rows),
                "effective_images_equal_target": all(int(row["effective_active_images"]) == int(row["target_active_images"]) for row in rows),
            }
            for run, rows in sampler_audit.items()
        },
        "contract": {
            "only_variable": "direction/order after common initial active clothing state",
            "C1": "reused Phase 5B S2 ID-Damage hard-first",
            "C2": "ID-Damage easy-to-hard",
            "C3": "random per-PID order, seed 1234",
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "online_h_id": "OFF",
            "same_initial_state": True,
            "same_event_epochs_and_counts": True,
            "same_active_pids_clothes_mean_and_effective_images": True,
            "test_leakage": "forbidden",
        },
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(OUT_ROOT / "PREFLIGHT.json", audit)
    print(json.dumps({"preflight": True, "passed": True, "audit": str(OUT_ROOT / "PREFLIGHT.json")}, indent=2, sort_keys=True))


def actual_active_row(run: str, epoch: int, active_state: dict[int, list[int]], effective: dict[tuple[int, int], list[int]], schedule_rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = [len(values) for values in active_state.values()]
    return {
        "run": run,
        "epoch": epoch,
        "active_images": sum(len(values) for values in effective.values()),
        "active_pids": len(active_state),
        "active_clothes": sum(counts),
        "mean_clothes_per_pid": float(np.mean(counts)),
        "new_clothes_count": sum(int(row["epoch"]) == epoch for row in schedule_rows),
        "active_state_sha256": p5b.active_state_hash(active_state),
    }


def open_metrics(path: Path, rank: int):
    if rank != 0:
        return None, None
    existing = path.is_file() and path.stat().st_size > 0
    stream = path.open("a" if existing else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=p5b.METRIC_FIELDS)
    if not existing:
        writer.writeheader()
        stream.flush()
    return stream, writer


def train(run: str, requested_epochs: int, resume: bool = False) -> None:
    if run not in RUNS:
        raise ValueError(run)
    if requested_epochs not in (SCREEN_EPOCHS, TOTAL_EPOCHS):
        raise ValueError("Phase 5B2 permits 15-epoch screen or authorized 60-epoch C2 continuation")
    load_prepared()
    preflight_path = OUT_ROOT / "PREFLIGHT.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError("Phase 5B2 preflight has not passed")
    if requested_epochs == TOTAL_EPOCHS:
        analysis_path = OUT_ROOT / "screen_analysis.json"
        if run != "C2" or not analysis_path.is_file() or not json.loads(analysis_path.read_text())["promotion_gate"]["passed"]:
            raise RuntimeError("60-epoch training is authorized only for C2 after a passed screen promotion gate")
    if not resume and (run_dir(run) / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"{run} is already complete")
    rank, device = p5b.init_dist()
    output = run_dir(run)
    metrics_file = writer = logger = amp_stream = active_stream = None
    try:
        cfg = config()
        set_seed(SEED)
        state = torch.load(P5B_COMMON_INIT, map_location="cpu", weights_only=False)
        model, loss_fn, optimizer, scheduler, scaler = p5b.setup_runtime(cfg, state, device)
        local_hash = sha256_bytes(json.dumps(p5b.tensor_hashes(p5b.cpu_state_dict(model)), sort_keys=True).encode())
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError("C2/C3 model initialization is not identical across ranks")
        dataset = p5b.build_dataset(cfg)
        transform_train, _ = p5b.build_img_transforms(cfg)
        full_index_map = p5b.build_index_map(dataset)
        states = load_states(run)
        schedule_rows = load_schedule(run)
        c1_targets = c1_reference_stats(read_csv(P5B_C1_STATS))
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
            payload = p5b.load_rank_state(checkpoint, model, optimizer, scheduler, scaler, device)
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
            if start_epoch != SCREEN_EPOCHS + 1 or len(previous_rows) < SCREEN_EPOCHS:
                raise RuntimeError("C2 continuation requires a complete 15-epoch checkpoint and metrics file")
            best_meta = json.loads((output / "best_state.json").read_text())
            best_r1 = float(best_meta["best_cc_rank1"])
            best_epoch = int(best_meta["best_epoch"])
        if rank == 0:
            logger = p5b.logger_for(output / f"train_log_rank{rank}.txt")
            writer = p5b.SummaryWriter(str(output / "TensorBoard"))
            metrics_file, metrics_writer = open_metrics(output / "metrics_per_epoch.csv", rank)
            amp_path = output / "amp_statistics.csv"
            amp_stream = amp_path.open("a" if amp_path.is_file() and amp_path.stat().st_size > 0 else "w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            if amp_stream.tell() == 0:
                amp_writer.writeheader(); amp_stream.flush()
            active_path = output / "active_set_statistics.csv"
            active_stream = active_path.open("a" if active_path.is_file() and active_path.stat().st_size > 0 else "w", newline="", encoding="utf-8")
            active_writer = csv.DictWriter(active_stream, fieldnames=["run", "epoch", "active_images", "active_pids", "active_clothes", "mean_clothes_per_pid", "new_clothes_count", "active_state_sha256", "rank0_plan_sha256", "rank1_plan_sha256", "optimizer_updates"])
            if active_stream.tell() == 0:
                active_writer.writeheader(); active_stream.flush()
        else:
            metrics_writer = amp_writer = active_writer = None

        for epoch in range(start_epoch, requested_epochs + 1):
            scheduler.step(epoch)
            active_state = states[epoch]
            effective = build_effective_index_map(run, epoch, active_state, full_index_map, c1_targets[epoch]["active_images"])
            loader, plan_meta = p5b.make_epoch_loader(dataset, transform_train, active_state, effective, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if rank == 0:
                if any(item["batches"] != STEPS_PER_EPOCH or item["batch_unique_pid_min"] < 2 or not item["pair_size_ok"] or not item["batch_size_ok"] for item in gathered_meta):
                    raise RuntimeError(f"runtime sampler audit failed at epoch {epoch}")
                if set(gathered_meta[0]["covered_pids"]) | set(gathered_meta[1]["covered_pids"]) != set(range(NUM_CLASSES)):
                    raise RuntimeError(f"runtime PID coverage failed at epoch {epoch}")
                if any(item["active_images"] != c1_targets[epoch]["active_images"] for item in gathered_meta):
                    raise RuntimeError(f"runtime effective active-image target failed at epoch {epoch}")
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
                    raise RuntimeError(f"non-finite loss at run={run}, epoch={epoch}, iteration={iteration}")
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
            active_row = actual_active_row(run, epoch, active_state, effective, schedule_rows)
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
                **{key: active_row[key] for key in ["active_images", "active_pids", "active_clothes", "mean_clothes_per_pid", "new_clothes_count"]},
                **metrics,
            }
            if rank == 0:
                metrics_writer.writerow(row); metrics_file.flush()
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch); writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"]}); amp_stream.flush()
                active_writer.writerow({**active_row, "rank0_plan_sha256": gathered_meta[0]["plan_sha256"], "rank1_plan_sha256": gathered_meta[1]["plan_sha256"], "optimizer_updates": STEPS_PER_EPOCH}); active_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"]); best_epoch = epoch
                    p5b.save_model_checkpoint(model.module, output / "model_best.pth")
                    write_json(output / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1"})
            if epoch == SCREEN_EPOCHS or epoch == requested_epochs:
                p5b.save_rank_state(output / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    final_name = "model_screen_final.pth" if requested_epochs == SCREEN_EPOCHS else "model_final.pth"
                    p5b.save_model_checkpoint(model.module, output / final_name)
                    write_json(output / f"checkpoint_epoch{epoch:03d}.json", {"epoch": epoch, "global_step": global_step, "rank_states": [str(output / f"rank{r}_state_epoch{epoch:03d}.pt") for r in range(WORLD_SIZE)], "model": str(output / final_name)})
                    if epoch == requested_epochs:
                        rows_now = read_csv(output / "metrics_per_epoch.csv")
                        write_json(output / "RUN_COMPLETE.json", {"phase": PHASE, "run": run, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates": global_step, "metrics": str(output / "metrics_per_epoch.csv"), "best_epoch": best_epoch, "best_cc_rank1": best_r1, "best_model": str(output / "model_best.pth"), "final_model": str(output / final_name), "rows": len(rows_now), "common_init_sha256": sha256_file(P5B_COMMON_INIT), "schedule_sha256": sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv")})
        if rank == 0:
            write_json(output / "AMP.json", {"initial_scale": p5b.INITIAL_SCALE, "epochs_completed": requested_epochs, "metrics_source": str(output / "amp_statistics.csv")})
    finally:
        for stream in (metrics_file, amp_stream, active_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        p5b.cleanup_dist()


def finite_rows(rows: list[dict[str, str]]) -> bool:
    required = ["cc_rank1", "cc_mAP", "train_loss", "ce_loss", "triplet_loss", "bio_loss"]
    return all(math.isfinite(float(row[key])) for row in rows for key in required)


def trajectory(rows: list[dict[str, str]], start: int = 8, end: int = 15) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"trajectory requires complete epochs {start}..{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) for row in selected], dtype=np.float64)
    mAP = np.asarray([float(row["cc_mAP"]) for row in selected], dtype=np.float64)
    return {
        "epochs": [int(v) for v in x],
        "mean_rank1": float(np.mean(r1)),
        "mean_mAP": float(np.mean(mAP)),
        "auc_rank1_normalized": float(np.trapz(r1, x) / (x[-1] - x[0])),
        "auc_mAP_normalized": float(np.trapz(mAP, x) / (x[-1] - x[0])),
        "slope_rank1_per_epoch": float(np.polyfit(x, r1, 1)[0]),
        "slope_mAP_per_epoch": float(np.polyfit(x, mAP, 1)[0]),
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


def analysis_payload() -> dict[str, Any]:
    load_prepared()
    sources = {
        "C0": P5B_C0_METRICS,
        "C1": P5B_C1_METRICS,
        "C2": run_dir("C2") / "metrics_per_epoch.csv",
        "C3": run_dir("C3") / "metrics_per_epoch.csv",
    }
    runs = {}
    for setting, path in sources.items():
        rows = read_csv(path)
        runs[setting] = {"rows": rows, "row_count": len(rows), "finite": finite_rows(rows), "trajectory_8_15": trajectory(rows) if len(rows) >= SCREEN_EPOCHS else None, "best": best_metrics(rows)}
    if any(runs[setting]["row_count"] < SCREEN_EPOCHS for setting in runs):
        raise RuntimeError("C0/C1/C2/C3 15-epoch metrics are required")
    c0, c1, c2, c3 = (runs[setting]["trajectory_8_15"] for setting in ("C0", "C1", "C2", "C3"))
    c2_rows = {int(row["epoch"]): row for row in runs["C2"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    c0_rows = {int(row["epoch"]): row for row in runs["C0"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    direction_rows = []
    for epoch in range(12, 16):
        a, b = c2_rows[epoch], c0_rows[epoch]
        r1_gap = float(a["cc_rank1"]) - float(b["cc_rank1"])
        map_gap = float(a["cc_mAP"]) - float(b["cc_mAP"])
        direction_rows.append({"epoch": epoch, "c2_gt_c0_rank1": r1_gap > 0, "c2_gt_c0_mAP": map_gap > 0, "either": r1_gap > 0 or map_gap > 0, "rank1_gap": r1_gap, "mAP_gap": map_gap, "both_within_minus_1pp": r1_gap >= -0.01 and map_gap >= -0.01})
    bad_vs_c0 = [float(c2_rows[epoch]["cc_rank1"]) < float(c0_rows[epoch]["cc_rank1"]) - 0.01 or float(c2_rows[epoch]["cc_mAP"]) < float(c0_rows[epoch]["cc_mAP"]) - 0.01 for epoch in range(8, 16)]
    auc_passed = c2["auc_rank1_normalized"] > c0["auc_rank1_normalized"] and c2["auc_mAP_normalized"] > c0["auc_mAP_normalized"]
    c2_promotion = {
        "auc_rank1_gt_c0": c2["auc_rank1_normalized"] > c0["auc_rank1_normalized"],
        "auc_mAP_gt_c0": c2["auc_mAP_normalized"] > c0["auc_mAP_normalized"],
        "trajectory_auc_passed_both_metrics": auc_passed,
        "epoch12_15": direction_rows,
        "epoch12_15_either_win_count": sum(row["either"] for row in direction_rows),
        "epoch12_15_win_gate": sum(row["either"] for row in direction_rows) >= 3,
        "bad_vs_c0_8_15": bad_vs_c0,
        "max_consecutive_over_1pp_drop": max_consecutive(bad_vs_c0),
        "other_metric_safety_gate": max_consecutive(bad_vs_c0) < 3,
    }
    c2_promotion["passed"] = bool(auc_passed and c2_promotion["epoch12_15_win_gate"] and c2_promotion["other_metric_safety_gate"])
    def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return {"auc_rank1_delta": a["auc_rank1_normalized"] - b["auc_rank1_normalized"], "auc_mAP_delta": a["auc_mAP_normalized"] - b["auc_mAP_normalized"], "mean_rank1_delta": a["mean_rank1"] - b["mean_rank1"], "mean_mAP_delta": a["mean_mAP"] - b["mean_mAP"]}
    c2_vs_c1 = compare(c2, c1)
    c1_vs_c3 = compare(c1, c3)
    c2_vs_c3 = compare(c2, c3)
    if c2_vs_c1["auc_rank1_delta"] > 0 and c2_vs_c1["auc_mAP_delta"] > 0:
        q1 = "YES"
    elif c2_vs_c1["auc_rank1_delta"] <= 0 and c2_vs_c1["auc_mAP_delta"] <= 0:
        q1 = "NO"
    else:
        q1 = "PARTIAL"
    restricted_below = {
        setting: runs[setting]["trajectory_8_15"]["mean_rank1"] < c0["mean_rank1"] and runs[setting]["trajectory_8_15"]["mean_mAP"] < c0["mean_mAP"]
        for setting in ("C1", "C2", "C3")
    }
    q2 = "YES" if all(restricted_below.values()) else "NO" if not any(restricted_below.values()) else "PARTIAL"
    if c1_vs_c3["auc_rank1_delta"] > 0 and c1_vs_c3["auc_mAP_delta"] > 0 and c2_vs_c3["auc_rank1_delta"] > 0 and c2_vs_c3["auc_mAP_delta"] > 0:
        q3 = "YES"
    elif c1_vs_c3["auc_rank1_delta"] <= 0 and c1_vs_c3["auc_mAP_delta"] <= 0 and c2_vs_c3["auc_rank1_delta"] <= 0 and c2_vs_c3["auc_mAP_delta"] <= 0:
        q3 = "NO"
    else:
        q3 = "INCONCLUSIVE"
    if c2_promotion["passed"]:
        diagnosis = "EASY-TO-HARD ID-DAMAGE CURRICULUM WORKS"
    elif q1 == "YES" and restricted_below["C2"]:
        diagnosis = "CURRICULUM DIRECTION MATTERS BUT COVERAGE STILL HURTS"
    elif q2 == "YES":
        diagnosis = "ACTIVE-SET CURRICULUM ITSELF IS HARMFUL"
    elif q3 == "YES":
        diagnosis = "ID-DAMAGE RANKING HELPS BUT NOT ENOUGH"
    elif q3 == "NO":
        diagnosis = "ID-DAMAGE RANKING HAS NO TRAINING BENEFIT"
    else:
        diagnosis = "RESULT INCONCLUSIVE"
    return {
        "phase": PHASE,
        "created_utc": utc_now(),
        "runs": runs,
        "reused_phase5b": {"C0": str(P5B_C0_METRICS), "C1": str(P5B_C1_METRICS), "C1_schedule": str(P5B_C1_SCHEDULE), "C1_schedule_sha256": sha256_file(P5B_C1_SCHEDULE), "common_init_sha256": sha256_file(P5B_COMMON_INIT)},
        "comparisons": {"C2_vs_C1": c2_vs_c1, "C1_vs_C3": c1_vs_c3, "C2_vs_C3": c2_vs_c3},
        "restricted_below_c0": restricted_below,
        "questions": {"Q1_hard_first_is_problem": q1, "Q2_restricted_coverage_is_problem": q2, "Q3_id_damage_training_value_after_structure_control": q3},
        "promotion_gate": c2_promotion,
        "final_diagnosis": diagnosis,
        "offline_evidence": {"original_clothing_delta_h_id_prefix2": 0.000186, "id_damage_delta_h_id_prefix2": 0.003619, "reliable_damage_delta_h_id_prefix2": 0.003547},
    }


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.4f}"


def signed_pp(value: float) -> str:
    return f"{value * 100:+.4f} pp"


def write_report(analysis: dict[str, Any]) -> None:
    rows = []
    labels = {
        "C0": ("Bio full-data", "none", "NO"),
        "C1": ("ID-Damage", "hard-first", "YES"),
        "C2": ("ID-Damage", "easy→hard", "YES"),
        "C3": ("Random", "random", "YES"),
    }
    for setting in ("C0", "C1", "C2", "C3"):
        t = analysis["runs"][setting]["trajectory_8_15"]
        status = "reused Phase 5B" if setting in ("C0", "C1") else "15-epoch screen complete" if analysis["runs"][setting]["row_count"] == SCREEN_EPOCHS else "60-epoch complete"
        rank, direction, restricted = labels[setting]
        rows.append(f"| {setting} | {rank} | {direction} | {restricted} | {pp(t['auc_rank1_normalized'])} | {pp(t['auc_mAP_normalized'])} | {status} |")
    trajectory_rows = []
    for setting in ("C0", "C1", "C2", "C3"):
        t = analysis["runs"][setting]["trajectory_8_15"]
        trajectory_rows.append(f"| {setting} | {pp(t['mean_rank1'])} | {pp(t['mean_mAP'])} | {pp(t['auc_rank1_normalized'])} | {pp(t['auc_mAP_normalized'])} | {t['slope_rank1_per_epoch'] * 100:+.4f} | {t['slope_mAP_per_epoch'] * 100:+.4f} |")
    c2_gate = analysis["promotion_gate"]
    detail_rows = []
    c2_by_epoch = {int(row["epoch"]): row for row in analysis["runs"]["C2"]["rows"]}
    c0_by_epoch = {int(row["epoch"]): row for row in analysis["runs"]["C0"]["rows"]}
    c1_by_epoch = {int(row["epoch"]): row for row in analysis["runs"]["C1"]["rows"]}
    c3_by_epoch = {int(row["epoch"]): row for row in analysis["runs"]["C3"]["rows"]}
    for epoch in range(12, 16):
        detail_rows.append(f"| {epoch} | {pp(float(c0_by_epoch[epoch]['cc_rank1']))}/{pp(float(c0_by_epoch[epoch]['cc_mAP']))} | {pp(float(c1_by_epoch[epoch]['cc_rank1']))}/{pp(float(c1_by_epoch[epoch]['cc_mAP']))} | {pp(float(c2_by_epoch[epoch]['cc_rank1']))}/{pp(float(c2_by_epoch[epoch]['cc_mAP']))} | {pp(float(c3_by_epoch[epoch]['cc_rank1']))}/{pp(float(c3_by_epoch[epoch]['cc_mAP']))} | {signed_pp(float(c2_by_epoch[epoch]['cc_rank1']) - float(c0_by_epoch[epoch]['cc_rank1']))}/{signed_pp(float(c2_by_epoch[epoch]['cc_mAP']) - float(c0_by_epoch[epoch]['cc_mAP']))} |")
    preflight = json.loads((OUT_ROOT / "PREFLIGHT.json").read_text())
    q = analysis["questions"]
    continuation = "PASS — run only C2 to 60 epochs." if c2_gate["passed"] and analysis["runs"]["C2"]["row_count"] < TOTAL_EPOCHS else "STOP — no 60-epoch continuation authorized." if not c2_gate["passed"] else "STOP — authorized C2 60-epoch run complete."
    report = [
        "# PHASE 5B2 — CURRICULUM DIRECTION + COVERAGE DIAGNOSTIC",
        "",
        f"Generated: `{utc_now()}`.",
        "",
        "This phase diagnoses why frozen ID-Damage hard-first improves over clothing ranking but does not reach full-data Bio. C0 and C1 are reused Phase 5B artifacts; only C2 and C3 are newly trained.",
        "",
        "## DECISION TABLE",
        "",
        "| Setting | Ranking | Direction | Restricted Coverage | E8–15 R1 AUC | E8–15 mAP AUC | Status |",
        "|---|---|---|---:|---:|---:|---|",
        *rows,
        "",
        f"Frozen external Bio Control: **{pp(BIO_CONTROL_R1)} / {pp(BIO_CONTROL_MAP)}**. Original Full remains secondary: **{pp(ORIGINAL_FULL_R1)} / {pp(ORIGINAL_FULL_MAP)}**.",
        "",
        "## TRAJECTORY (EPOCHS 8–15)",
        "",
        "| Setting | Mean R1 | Mean mAP | R1 AUC | mAP AUC | R1 slope pp/epoch | mAP slope pp/epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *trajectory_rows,
        "",
        "AUC is the normalized trapezoidal trajectory average, not BEST-epoch selection.",
        "",
        "## REPRESENTATION AND TRAINING CONTRACT",
        "",
        f"- Frozen common init reused from Phase 5B: `{P5B_COMMON_INIT}`; SHA256 `{sha256_file(P5B_COMMON_INIT)}`.",
        f"- Reused C0 metrics: `{P5B_C0_METRICS}`; reused C1 metrics: `{P5B_C1_METRICS}`.",
        f"- Reused C1 schedule SHA256: `{sha256_file(P5B_C1_SCHEDULE)}`. C1 is Phase 5B S2 ID-Damage hard-first.",
        "- Bio-only EVA02-Large, 224×224, 2 PPUs, batch 8, NUM_INSTANCES 2, seed 1234, SGD, LR 2e-6, weight decay 0.05, CE + Triplet + Summary Bio, cosine scheduler, AMP initial scale 1024.",
        "- Reverse/GRL, DW, Reliability, online H_ID, Full DIFFER, PRCC, loss weighting, new network/backbone/optimizer are OFF.",
        f"- Fixed optimizer budget: `{STEPS_PER_EPOCH}` updates/rank/epoch; all curriculum runs use `{TOTAL_EPOCHS * STEPS_PER_EPOCH}` planned updates for 60 epochs.",
        "",
        "## SCHEDULE AUDIT",
        "",
        f"- Preflight: **{'PASS' if preflight['passed'] else 'FAIL'}**; event keys identical: `{preflight['event_keys_equal']}`; event count: `{preflight['event_count']}`.",
        f"- C1 reused hard-first audit: `{preflight['schedule_audit']['c1_is_exact_hard_first']}`; C2 exact easy-to-hard audit: `{preflight['schedule_audit']['c2_is_exact_easy_to_hard']}`; C2 reverse of C1: `{preflight['schedule_audit']['c2_is_reverse_of_c1']}`.",
        f"- C1/C2 PID-wise Kendall mean/median: `{preflight['schedule_audit']['c1_c2_pid_kendall']['mean']:.4f}` / `{preflight['schedule_audit']['c1_c2_pid_kendall']['median']:.4f}`; ties are resolved by cloth_id.",
        f"- Initial active state is shared. C1/C2/C3 active PID count, clothing-state count, mean clothes/PID, expansion timing/count, and effective active-image count are equal at every epoch.",
        f"- C2/C3 effective active images are deterministic stratified pools targeted to C1's frozen `active_images`; raw support counts are retained separately because clothing states have unequal support.",
        f"- Schedule source split audit: no query/gallery schedule source: `{preflight['no_test_sources']}`. Random order seed: `{SEED}`.",
        "",
        "## C1/C2/C3 EPOCH 12–15 COMPARISON",
        "",
        "| Epoch | C0 R1/mAP | C1 R1/mAP | C2 R1/mAP | C3 R1/mAP | C2−C0 R1/mAP |",
        "|---:|---:|---:|---:|---:|---:|",
        *detail_rows,
        "",
        "## HYPOTHESIS TESTS",
        "",
        f"- H1 — Is hard-first the problem? **{q['Q1_hard_first_is_problem']}**. C2−C1 AUC: `{signed_pp(analysis['comparisons']['C2_vs_C1']['auc_rank1_delta'])}` / `{signed_pp(analysis['comparisons']['C2_vs_C1']['auc_mAP_delta'])}`.",
        f"- H2 — Is restricted clothing coverage the problem? **{q['Q2_restricted_coverage_is_problem']}**. Restricted-below-C0 flags: `{json.dumps(analysis['restricted_below_c0'], sort_keys=True)}`.",
        f"- H3 — Does ID-Damage retain training value after structure control? **{q['Q3_id_damage_training_value_after_structure_control']}**. C1−C3 AUC: `{signed_pp(analysis['comparisons']['C1_vs_C3']['auc_rank1_delta'])}` / `{signed_pp(analysis['comparisons']['C1_vs_C3']['auc_mAP_delta'])}`; C2−C3 AUC: `{signed_pp(analysis['comparisons']['C2_vs_C3']['auc_rank1_delta'])}` / `{signed_pp(analysis['comparisons']['C2_vs_C3']['auc_mAP_delta'])}`.",
        "",
        "## C2 PROMOTION GATE",
        "",
        f"- C2 R1 AUC > C0: **{c2_gate['auc_rank1_gt_c0']}**; C2 mAP AUC > C0: **{c2_gate['auc_mAP_gt_c0']}**.",
        f"- E12–15 at least one metric > C0: `{c2_gate['epoch12_15_either_win_count']}/4`; gate: **{c2_gate['epoch12_15_win_gate']}**.",
        f"- Maximum consecutive >1 pp drop versus C0: `{c2_gate['max_consecutive_over_1pp_drop']}`; safety gate: **{c2_gate['other_metric_safety_gate']}**.",
        f"- C2 promotion: **{'PASS' if c2_gate['passed'] else 'FAIL'}**.",
        "",
        "## FINAL DIAGNOSIS",
        "",
        f"**{analysis['final_diagnosis']}**",
        "",
        f"Offline → training interpretation: C2 is **{'promising for 60-epoch validation' if c2_gate['passed'] else 'not promoted'}** under the predeclared gate. This phase does not authorize DW, Reliability, Online H_ID, or Full DIFFER.",
        f"Next step: **{continuation}**",
        "",
        "## ARTIFACTS",
        "",
        f"- Root: `{OUT_ROOT}`",
        f"- Schedules: `{SCHEDULE_ROOT / 'schedule_C1.csv'}`, `{SCHEDULE_ROOT / 'schedule_C2.csv'}`, `{SCHEDULE_ROOT / 'schedule_C3.csv'}`",
        f"- Preflight: `{OUT_ROOT / 'PREFLIGHT.json'}`",
        f"- Analysis: `{OUT_ROOT / 'screen_analysis.json'}`",
        f"- Report: `{REPORT_PATH}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    analysis = analysis_payload()
    analysis["final_validation"] = analysis["runs"]["C2"]["row_count"] >= TOTAL_EPOCHS
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
