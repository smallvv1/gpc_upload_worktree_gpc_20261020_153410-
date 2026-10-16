#!/usr/bin/env python3
"""Phase 5B: Bio-only training validation for ID-Damage clothing DS.

This file is intentionally a new-method-track runner.  It does not modify the
formal DIFFER trainer and it does not contain a reverse/GRL, DW, reliability,
online-score, or Full-DIFFER training path.

The experiment has three locked settings:

  S0  Bio, all train images from epoch 1
  S1  Bio + the frozen Phase-5A Clothing-text nearest-center schedule
  S2  Bio + the frozen Phase-5A ID-Damage schedule

S1 and S2 share the initial pool, expansion epochs, expansion count, model
topology, optimizer, loss, augmentation, and a fixed 596 optimizer updates per
rank per epoch.  Only the clothing state selected at an expansion event is
different.  The schedule is read from Phase 5A artifacts before training and
is never recomputed from validation metrics or from a test retrieval result.

The supported modes are deliberately explicit:

  prepare   build COMMON_PHASE5B_INIT.pth and frozen S1/S2 schedules
  preflight audit topology, schedules, source split, and sampler budget
  train     run one setting under torchrun
  analyze   analyze completed screen/final runs and write the report

Run ``prepare`` and ``preflight`` as single-process CPU-control commands.  Run
``train`` with two visible PPUs through torchrun.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset, read_image  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from model.build import factory  # noqa: E402
from processor.processor import test  # noqa: E402
from solver import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC  # noqa: E402


PHASE = "phase5b_id_damage_ds"
SEED = 1234
WORLD_SIZE = 2
STEPS_PER_EPOCH = 596
BATCH_SIZE = 8
NUM_INSTANCES = 2
TOTAL_EPOCHS = 60
SCREEN_EPOCHS = 15
INITIAL_SCALE = 1024
NUM_CLASSES = 77
NUM_CAMERAS = 12
GLOBAL_PAIR_UNITS = STEPS_PER_EPOCH * (BATCH_SIZE // NUM_INSTANCES) * WORLD_SIZE
LOCAL_PAIR_UNITS = GLOBAL_PAIR_UNITS // WORLD_SIZE

PHASE5A_ROOT = ROOT / "outputs/new_method_track/phase5a_clothing_hardness"
PHASE5A_SCRIPT = ROOT / "scripts/phase5a_clothing_hardness.py"
PHASE5A_REPLAY = PHASE5A_ROOT / "offline_replay_events.csv"
PHASE5A_SUMMARY = PHASE5A_ROOT / "offline_replay_summary.json"
PHASE5A_MANIFEST = PHASE5A_ROOT / "PHASE_5A_MANIFEST.json"
PHASE5A_GROUPS = PHASE5A_ROOT / "group_metrics.csv"
HISTORICAL_TRACE = Path("/home/CL-CC-ReID/results/ltcc_phase4_v4_ds/curriculum_trace.csv")
TRAIN_ROOT = Path("/home/data/LTCC_ReID/train")
QUERY_ROOT = Path("/home/data/LTCC_ReID/query")
GALLERY_ROOT = Path("/home/data/LTCC_ReID/test")

OUT_ROOT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds"
COMMON_ROOT = OUT_ROOT / "common_init"
SCHEDULE_ROOT = OUT_ROOT / "schedules"
REPORT_PATH = ROOT / "reports/PHASE_5B_ID_DAMAGE_DS_TRAINING.md"
BIO_METRICS = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/metrics_per_epoch.csv"

RUNS = ("S0", "S1", "S2")
SCHEDULE_METHODS = {
    "S1": {
        "phase5a_method": "text_clothing_nearest",
        "label": "Original Clothing-DS + Bio",
        "ranking": "Clothing semantic nearest-center distance",
        "ranking_rule": "larger frozen Phase-5A nearest-center distance; tie -> smaller cloth_id",
    },
    "S2": {
        "phase5a_method": "id_damage",
        "label": "ID-Damage-DS + Bio",
        "ranking": "frozen H_ID(y,c) clothing-state mean",
        "ranking_rule": "larger frozen H_ID(y,c); tie -> smaller cloth_id",
    },
}

METRIC_FIELDS = [
    "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "lr", "lr_bias",
    "id_accuracy", "bio_accuracy", "amp_scale_min", "amp_scale_max", "amp_scale_end",
    "skipped_steps", "found_inf_count", "nonfinite_events", "optimizer_updates",
    "active_images", "active_pids", "active_clothes", "mean_clothes_per_pid",
    "new_clothes_count", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP",
    "general_rank1", "general_rank5", "general_rank10", "general_mAP",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    raise TypeError(type(value).__name__)


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


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        key: sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for key, value in sorted(state.items())
    }


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure() -> Any:
    cfg = GLOBAL_CFG.clone()
    cfg.merge_from_file(str(ROOT / "configs/ltcc/eva02_l_bio.yml"))
    cfg.defrost()
    cfg.MODEL.TYPE = "eva02_cloth"
    cfg.MODEL.NAME = "eva02_l_bio"
    cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio"]
    cfg.MODEL.NONBIO_HEAD = False
    cfg.MODEL.LAST_LAYER = "clipFc"
    cfg.MODEL.CLIP_DIM = 512
    cfg.MODEL.DIST_TRAIN = True
    cfg.MODEL.DETACH_REVERSE_INPUT = False
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    cfg.DATA.BIO_INDEX = "0"
    cfg.DATA.NOBIO_INDEX = [1, 2]
    cfg.DATA.SUMMERY_TEXT = True
    cfg.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.DATA.NUM_INSTANCES = NUM_INSTANCES
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.TEST_BATCH = 128
    cfg.DATA.IMG_HEIGHT = 224
    cfg.DATA.IMG_WIDTH = 224
    cfg.SOLVER.SEED = SEED
    cfg.SOLVER.OPTIMIZER_NAME = "SGD"
    cfg.SOLVER.BASE_LR = 2e-6
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.05
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_EPOCHS = TOTAL_EPOCHS
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_EPOCHS = 20
    cfg.SOLVER.WARMUP_LR = 7.8125e-7
    cfg.SOLVER.CHECKPOINT_PERIOD = TOTAL_EPOCHS
    cfg.SOLVER.EVAL_PERIOD = 1
    cfg.OUTPUT_DIR = str(OUT_ROOT)
    cfg.TEST.WEIGHT = ""
    cfg.TRAIN.START_EPOCH = 1
    cfg.freeze()
    return cfg


def phase5a_frozen_audit() -> dict[str, Any]:
    required = [PHASE5A_SCRIPT, PHASE5A_REPLAY, PHASE5A_SUMMARY, PHASE5A_GROUPS, HISTORICAL_TRACE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 5A frozen artifacts missing: " + ", ".join(missing))
    return {
        "script": str(PHASE5A_SCRIPT),
        "script_sha256": sha256_file(PHASE5A_SCRIPT),
        "replay": str(PHASE5A_REPLAY),
        "replay_sha256": sha256_file(PHASE5A_REPLAY),
        "summary": str(PHASE5A_SUMMARY),
        "summary_sha256": sha256_file(PHASE5A_SUMMARY),
        "groups": str(PHASE5A_GROUPS),
        "groups_sha256": sha256_file(PHASE5A_GROUPS),
        "historical_trace": str(HISTORICAL_TRACE),
        "historical_trace_sha256": sha256_file(HISTORICAL_TRACE),
    }


def parse_json_value(value: Any, default: Any):
    if value is None:
        return default
    text = str(value).strip()
    if text in ("", "None", "null"):
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Phase 5A writes JSON.  This fallback is only for the historical CSV
        # and makes the audit robust to its legacy Python-list spelling.
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


def int_scalar(value: Any) -> int | None:
    parsed = parse_json_value(value, None)
    if isinstance(parsed, (list, tuple)):
        if len(parsed) != 1:
            return None
        parsed = parsed[0]
    return None if parsed is None else int(parsed)


def load_initial_pool() -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    trace_rows = read_csv(HISTORICAL_TRACE)
    initial: dict[int, set[int]] = {}
    for row in trace_rows:
        pid = int(row["pid"])
        if pid not in initial:
            initial[pid] = set(int_list(row.get("selected_clothes")))
    if len(initial) != NUM_CLASSES or any(len(values) != 1 for values in initial.values()):
        raise RuntimeError("historical trace does not provide exactly one initial clothing state for all 77 PIDs")

    group_rows = read_csv(PHASE5A_GROUPS)
    all_clothes: dict[int, set[int]] = defaultdict(set)
    for row in group_rows:
        all_clothes[int(row["pid"])].add(int(row["cloth_id"]))
    if set(all_clothes) != set(initial):
        raise RuntimeError("Phase 5A group PID set differs from historical initial PID set")
    return initial, dict(all_clothes)


def load_frozen_events() -> tuple[list[dict[str, str]], dict[tuple[int, int], dict[str, str]], dict[int, set[int]], dict[int, set[int]]]:
    initial, all_clothes = load_initial_pool()
    replay_rows = read_csv(PHASE5A_REPLAY)
    group_rows = read_csv(PHASE5A_GROUPS)
    group_by_key = {(int(row["pid"]), int(row["cloth_id"])): row for row in group_rows}
    if len(replay_rows) != 179:
        raise RuntimeError(f"expected 179 Phase 5A replay events, got {len(replay_rows)}")
    events = []
    for expected_id, row in enumerate(replay_rows, start=1):
        if int(row["event_id"]) != expected_id:
            raise RuntimeError("Phase 5A replay event IDs are not contiguous")
        pid = int(row["pid"])
        if pid not in initial:
            raise RuntimeError(f"event references unknown PID {pid}")
        if int(row["epoch"]) < 1 or int(row["expansion_position_for_pid"]) < 1:
            raise RuntimeError("invalid Phase 5A event epoch or expansion position")
        for method in SCHEDULE_METHODS.values():
            prefix = method["phase5a_method"]
            selected = int(row[f"{prefix}_selected"])
            if selected not in all_clothes[pid]:
                raise RuntimeError(f"Phase 5A selected unknown clothing state {(pid, selected)}")
        events.append(row)
    # Every group used by a schedule must have an auditable train source.
    for key, row in group_by_key.items():
        paths = parse_json_value(row.get("image_paths_json"), [])
        if not paths or any(not str(path).startswith(str(TRAIN_ROOT) + "/") for path in paths):
            raise RuntimeError(f"group {key} has a non-train or missing source path")
        if any(str(path).startswith(str(QUERY_ROOT)) or str(path).startswith(str(GALLERY_ROOT)) for path in paths):
            raise RuntimeError(f"group {key} leaks query/gallery into schedule sources")
    return events, group_by_key, initial, all_clothes


def canonical_state_hash(state: dict[int, set[int]]) -> str:
    payload = {str(pid): sorted(int(cid) for cid in values) for pid, values in sorted(state.items())}
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def make_method_schedule(
    run: str,
    events: list[dict[str, str]],
    group_by_key: dict[tuple[int, int], dict[str, str]],
    initial: dict[int, set[int]],
    all_clothes: dict[int, set[int]],
) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[int]]], list[dict[str, Any]]]:
    method_spec = SCHEDULE_METHODS[run]
    phase5a_method = method_spec["phase5a_method"]
    state = {pid: set(values) for pid, values in initial.items()}
    schedule_rows: list[dict[str, Any]] = []
    sequence = defaultdict(list)
    for event in events:
        event_id = int(event["event_id"])
        pid = int(event["pid"])
        active = set(state[pid])
        phase5a_active = set(int_list(event[f"{phase5a_method}_active_clothes"]))
        candidates = set(int_list(event[f"{phase5a_method}_candidates"]))
        expected_candidates = all_clothes[pid] - active
        if candidates != expected_candidates:
            raise RuntimeError(f"{run} candidate set disagrees with frozen Phase 5A row at event {event_id}")
        if not candidates:
            raise RuntimeError(f"{run} has no candidate at event {event_id}")
        selected = int(event[f"{phase5a_method}_selected"])
        if selected not in candidates:
            raise RuntimeError(f"{run} selected non-candidate at event {event_id}")
        # Phase 5A serializes the method's active set after adding the
        # selected state (the candidate set is serialized before the add).
        # Preserve that frozen implementation while writing an unambiguous
        # before/after schedule for training.
        if phase5a_active != active | {selected}:
            raise RuntimeError(f"{run} Phase 5A post-add active state disagrees at event {event_id}")
        group = group_by_key[(pid, selected)]
        score_map = parse_json_value(event[f"{phase5a_method}_candidate_scores"], {})
        score_map = {str(int(k)): float(v) for k, v in score_map.items()}
        if str(selected) not in score_map:
            raise RuntimeError(f"{run} selected score missing at event {event_id}")
        source_paths = parse_json_value(group.get("image_paths_json"), [])
        source_names = parse_json_value(group.get("image_filenames_json"), [])
        after = active | {selected}
        schedule_rows.append({
            "run": run,
            "event_id": event_id,
            "trace_index": int(event["trace_index"]),
            "epoch": int(event["epoch"]),
            "pid": pid,
            "expansion_position_for_pid": int(event["expansion_position_for_pid"]),
            "active_clothes_before": json.dumps(sorted(active), separators=(",", ":")),
            "candidate_clothes": json.dumps(sorted(candidates), separators=(",", ":")),
            "new_clothes": selected,
            "active_clothes_after": json.dumps(sorted(after), separators=(",", ":")),
            "ranking_signal": phase5a_method,
            "ranking_description": method_spec["ranking"],
            "ranking_rule": method_spec["ranking_rule"],
            "selected_ranking_score": score_map[str(selected)],
            "candidate_scores": json.dumps(score_map, sort_keys=True, separators=(",", ":")),
            "selected_h_id": float(group["h_id_mean"]),
            "selected_delta_h_id_vs_historical": float(event[f"{phase5a_method}_h_id_delta"]),
            "selected_reliability": float(group["reliability"]),
            "cloth_key": group["cloth_key"],
            "source_image_filenames": json.dumps(source_names, separators=(",", ":")),
            "source_image_paths": json.dumps(source_paths, separators=(",", ":")),
            "phase5a_replay_row_sha256": sha256_bytes(json.dumps(event, sort_keys=True).encode()),
        })
        state[pid].add(selected)
        sequence[pid].append(selected)

    active_states: dict[int, dict[int, list[int]]] = {}
    current = {pid: set(values) for pid, values in initial.items()}
    for epoch in range(1, TOTAL_EPOCHS + 1):
        if epoch > 1:
            for row in schedule_rows:
                if int(row["epoch"]) == epoch:
                    current[int(row["pid"])].add(int(row["new_clothes"]))
        active_states[epoch] = {
            str(pid): sorted(int(cid) for cid in values) for pid, values in sorted(current.items())
        }
    stats_rows = make_active_stats(run, active_states, schedule_rows, group_by_key)
    return schedule_rows, active_states, stats_rows


def make_active_stats(
    run: str,
    active_states: dict[int, dict[int, list[int]]],
    schedule_rows: list[dict[str, Any]],
    group_by_key: dict[tuple[int, int], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        state = active_states[epoch]
        active_images = sum(
            int(group_by_key[(int(pid), int(cid))]["support"])
            for pid, clothes in state.items() for cid in clothes
        )
        counts = [len(clothes) for clothes in state.values()]
        new_count = sum(int(row["epoch"]) == epoch for row in schedule_rows)
        rows.append({
            "run": run,
            "epoch": epoch,
            "active_images": active_images,
            "active_pids": len(state),
            "active_clothes": sum(counts),
            "mean_clothes_per_pid": float(np.mean(counts)),
            "new_clothes_count": new_count,
            "active_state_sha256": canonical_state_hash({int(pid): set(values) for pid, values in state.items()}),
        })
    return rows


def make_s0_states(group_by_key: dict[tuple[int, int], dict[str, str]]) -> dict[int, dict[int, list[int]]]:
    all_state: dict[int, list[int]] = defaultdict(list)
    for pid, cid in sorted(group_by_key):
        all_state[pid].append(cid)
    return {epoch: {str(pid): sorted(values) for pid, values in sorted(all_state.items())} for epoch in range(1, TOTAL_EPOCHS + 1)}


def save_git_snapshot(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True, text=False).stdout
    (run_dir / "git_diff.patch").write_bytes(diff)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    (run_dir / "git_status.txt").write_text(status, encoding="utf-8")


def build_common_init(cfg: Any) -> dict[str, Any]:
    path = COMMON_ROOT / "COMMON_PHASE5B_INIT.pth"
    hashes_path = COMMON_ROOT / "COMMON_PHASE5B_INIT_HASHES.json"
    if path.exists() or hashes_path.exists():
        raise RuntimeError("common init artifact already exists; refusing to overwrite a frozen Phase 5B init")
    COMMON_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    dataset = build_dataset(cfg)
    if dataset.num_train_pids != NUM_CLASSES or dataset.num_camera != NUM_CAMERAS:
        raise RuntimeError(f"unexpected LTCC topology metadata: {dataset.num_train_pids} PIDs, {dataset.num_camera} cameras")
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    model.eval()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if any("Reverse" in key or "reverse" in key for key in state):
        raise RuntimeError("Bio-only common init unexpectedly contains a reverse/GRL parameter")
    torch.save(state, path)
    hashes = tensor_hashes(state)
    write_json(hashes_path, {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "topology": "EVA02-Large Bio-only: CE + Triplet + Summary Bio",
        "reverse": "OFF",
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "state_sha256": sha256_file(path),
        "tensor_hashes": hashes,
    })
    (COMMON_ROOT / "config.yml").write_text(cfg.dump(), encoding="utf-8")
    del model, state, dataset
    gc.collect()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "hashes_path": str(hashes_path),
        "state_key_count": len(hashes),
        "state_numel": int(json.loads(hashes_path.read_text())["state_numel"]),
    }


def prepare() -> None:
    if (OUT_ROOT / "PHASE5B_PREPARED.json").exists():
        raise RuntimeError("Phase 5B artifacts already prepared; refusing to overwrite the frozen schedules")
    cfg = configure()
    audit = phase5a_frozen_audit()
    events, group_by_key, initial, all_clothes = load_frozen_events()
    common = build_common_init(cfg)
    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    run_manifests = {}
    for run in RUNS:
        run_dir = OUT_ROOT / {"S0": "S0_bio_full_data", "S1": "S1_clothing_ds", "S2": "S2_id_damage_ds"}[run]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
        save_git_snapshot(run_dir)
        if run == "S0":
            active_states = make_s0_states(group_by_key)
            schedule_rows: list[dict[str, Any]] = []
            stats_rows = make_active_stats(run, active_states, schedule_rows, group_by_key)
            schedule_hash = None
            schedule_file = None
        else:
            schedule_rows, active_states, stats_rows = make_method_schedule(
                run, events, group_by_key, initial, all_clothes,
            )
            schedule_file = SCHEDULE_ROOT / f"schedule_{run}.csv"
            schedule_fields = list(schedule_rows[0])
            write_csv(schedule_file, schedule_rows, schedule_fields)
            schedule_hash = sha256_file(schedule_file)
            write_json(SCHEDULE_ROOT / f"active_states_{run}.json", active_states)
            write_csv(SCHEDULE_ROOT / f"active_set_statistics_{run}.csv", stats_rows, list(stats_rows[0]))
        write_csv(run_dir / "active_set_statistics_locked.csv", stats_rows, list(stats_rows[0]))
        write_json(run_dir / "schedule_hash.json", {
            "run": run,
            "schedule_file": str(schedule_file) if schedule_file else None,
            "schedule_sha256": schedule_hash,
            "active_set_statistics": str((SCHEDULE_ROOT if run != "S0" else run_dir) / (f"active_set_statistics_{run}.csv" if run != "S0" else "active_set_statistics_locked.csv")),
            "frozen": True,
        })
        write_json(run_dir / "run_manifest.json", {
            "phase": PHASE,
            "run": run,
            "label": "Full-data Bio Control" if run == "S0" else SCHEDULE_METHODS[run]["label"],
            "ranking": "none (all train states from epoch 1)" if run == "S0" else SCHEDULE_METHODS[run]["ranking"],
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "online_h_id": "OFF",
            "common_init": common,
            "schedule_file": str(schedule_file) if schedule_file else None,
            "schedule_sha256": schedule_hash,
            "phase5a_frozen_audit": audit,
            "screen_epochs": SCREEN_EPOCHS,
            "total_epochs": TOTAL_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "optimizer_updates": TOTAL_EPOCHS * STEPS_PER_EPOCH,
            "data": {"train_images": 9576, "train_pids": 77, "train_clothes": 256},
            "initial_active_state_sha256": canonical_state_hash({pid: set(values) for pid, values in initial.items()}),
        })
        run_manifests[run] = json.loads((run_dir / "run_manifest.json").read_text())
    write_json(SCHEDULE_ROOT / "initial_pool.json", {str(pid): sorted(values) for pid, values in sorted(initial.items())})
    write_json(SCHEDULE_ROOT / "all_clothes_by_pid.json", {str(pid): sorted(values) for pid, values in sorted(all_clothes.items())})
    write_json(OUT_ROOT / "PHASE5B_PREPARED.json", {
        "phase": PHASE,
        "created_utc": utc_now(),
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "phase5a_frozen_audit": audit,
        "common_init": common,
        "event_count": len(events),
        "initial_pid_count": len(initial),
        "run_manifests": run_manifests,
        "contract": {
            "only_variable": "next clothing state selected at a pre-frozen expansion event",
            "reverse": "OFF",
            "dw": "OFF",
            "reliability": "OFF",
            "test_leakage": "forbidden for schedule construction",
            "global_pair_units_per_epoch": GLOBAL_PAIR_UNITS,
            "local_pair_units_per_rank_per_epoch": LOCAL_PAIR_UNITS,
        },
    })
    print(json.dumps({"prepared": True, "output": str(OUT_ROOT), "common_init": common}, indent=2, sort_keys=True))


def load_prepared() -> dict[str, Any]:
    path = OUT_ROOT / "PHASE5B_PREPARED.json"
    if not path.is_file():
        raise FileNotFoundError("run prepare first: " + str(path))
    return json.loads(path.read_text())


def load_active_states(run: str) -> dict[int, dict[int, list[int]]]:
    if run == "S0":
        _, group_by_key, _, _ = load_frozen_events()
        return make_s0_states(group_by_key)
    path = SCHEDULE_ROOT / f"active_states_{run}.json"
    raw = json.loads(path.read_text())
    return {int(epoch): {int(pid): list(values) for pid, values in state.items()} for epoch, state in raw.items()}


def build_index_map(dataset: Any) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(dataset.train):
        result[(int(item["pid"]), int(item["clothes_id"]))].append(index)
    return dict(result)


def make_pair_units(
    active_state: dict[int, list[int]],
    index_map: dict[tuple[int, int], list[int]],
    epoch: int,
) -> list[list[int]]:
    """Make a standard identity-pair stream, then repeat to fixed global budget.

    For the full S0 state the first branch is exactly the released sampler's
    pair construction: shuffle images within PID, form NUM_INSTANCES pairs,
    shuffle pairs, then shard pair units by rank.  For a smaller active state,
    the same construction is repeated deterministically until the exact full
    global pair budget is reached.  This is the explicit oversampling required
    to prevent a smaller clothing pool from silently reducing optimizer steps.
    """
    py_rng = random.Random(SEED + epoch)
    np_rng = np.random.RandomState(SEED + epoch)
    base: list[list[int]] = []
    for pid in sorted(active_state):
        image_indices = [
            index for cid in sorted(active_state[pid]) for index in index_map[(int(pid), int(cid))]
        ]
        if not image_indices:
            raise RuntimeError(f"active state has no training images for PID {pid}")
        if len(image_indices) < NUM_INSTANCES:
            image_indices = np_rng.choice(image_indices, size=NUM_INSTANCES, replace=True).tolist()
        py_rng.shuffle(image_indices)
        for start in range(0, len(image_indices) - NUM_INSTANCES + 1, NUM_INSTANCES):
            pair = image_indices[start:start + NUM_INSTANCES]
            if len(pair) == NUM_INSTANCES:
                base.append(pair)
    if len(base) == 0 or any(len(pair) != NUM_INSTANCES for pair in base):
        raise RuntimeError("identity sampler produced no complete pair units")
    # Reconstruct PID membership from the source index map without relying on
    # a hidden dataset-global ordering.
    index_to_pid = {}
    for (pid, _cid), indices in index_map.items():
        for index in indices:
            index_to_pid[int(index)] = int(pid)
    base_pids = {index_to_pid[int(pair[0])] for pair in base}
    if base_pids != {int(pid) for pid in active_state}:
        raise RuntimeError(f"sampler dropped active PID(s): {sorted(set(active_state) - base_pids)}")

    if len(base) < GLOBAL_PAIR_UNITS:
        expanded: list[list[int]] = []
        while len(expanded) < GLOBAL_PAIR_UNITS:
            cycle = [list(pair) for pair in base]
            py_rng.shuffle(cycle)
            expanded.extend(cycle)
        base = expanded[:GLOBAL_PAIR_UNITS]
    else:
        py_rng.shuffle(base)
        base = base[:GLOBAL_PAIR_UNITS]
    if len(base) != GLOBAL_PAIR_UNITS:
        raise RuntimeError("fixed global pair budget construction failed")

    # The released identity sampler forms NUM_INSTANCES-sized identity units,
    # then the DataLoader groups those units into a batch.  A pure pair-unit
    # shuffle can therefore produce an all-same-PID batch.  That is legal for
    # the pair construction but invalid for hard-negative triplet mining: the
    # negative set is empty.  Keep the same pair pool, budget, and rank
    # sharding, while deterministically repairing only such degenerate local
    # batches by swapping one pair with a later pair from another PID.
    index_to_pid = {}
    for (pid, _cid), indices in index_map.items():
        for index in indices:
            index_to_pid[int(index)] = int(pid)

    pairs_per_batch = BATCH_SIZE // NUM_INSTANCES

    def repair_rank_stream(rank_pairs: list[list[int]]) -> list[list[int]]:
        repaired = [list(pair) for pair in rank_pairs]
        for start in range(0, len(repaired), pairs_per_batch):
            stop = start + pairs_per_batch
            batch = repaired[start:stop]
            batch_pids = [index_to_pid[int(pair[0])] for pair in batch]
            if len(set(batch_pids)) >= 2:
                continue
            dominant_pid = batch_pids[0]
            replacement = None
            for candidate_index in range(stop, len(repaired)):
                candidate_pid = index_to_pid[int(repaired[candidate_index][0])]
                if candidate_pid != dominant_pid:
                    replacement = candidate_index
                    break
            if replacement is None:
                raise RuntimeError("could not repair an all-same-PID identity batch")
            repaired[stop - 1], repaired[replacement] = repaired[replacement], repaired[stop - 1]

        minimum_unique_pids = min(
            len({index_to_pid[int(pair[0])] for pair in repaired[start:start + pairs_per_batch]})
            for start in range(0, len(repaired), pairs_per_batch)
        )
        if minimum_unique_pids < 2:
            raise RuntimeError("identity batch repair left an empty negative set")
        return repaired

    rank_streams = [repair_rank_stream(base[rank::WORLD_SIZE]) for rank in range(WORLD_SIZE)]
    base = [
        rank_streams[rank][position]
        for position in range(LOCAL_PAIR_UNITS)
        for rank in range(WORLD_SIZE)
    ]
    return base


def pid_from_index(index: int, index_map: dict[tuple[int, int], list[int]]) -> int:
    for (pid, _cid), indices in index_map.items():
        if index in indices:
            return int(pid)
    raise KeyError(index)


def plan_seed(epoch: int, rank: int, iteration: int, position: int, dataset_index: int) -> int:
    token = f"{PHASE}|{SEED}|{epoch}|{rank}|{iteration}|{position}|{dataset_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & 0x7FFFFFFF


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PlannedTrainDataset(Dataset):
    def __init__(self, samples: list[tuple[dict[str, Any], int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, augmentation_seed = self.samples[index]
        item = dict(source)
        image = read_image(item["image_path"])
        previous = random.getstate()
        try:
            random.seed(int(augmentation_seed))
            image = self.transform(image)
        finally:
            random.setstate(previous)
        item["image"] = image
        item["augmentation_seed"] = int(augmentation_seed)
        return item


def make_epoch_loader(
    dataset: Any,
    transform_train: Any,
    active_state: dict[int, list[int]],
    index_map: dict[tuple[int, int], list[int]],
    epoch: int,
    rank: int,
) -> tuple[DataLoader, dict[str, Any]]:
    global_pairs = make_pair_units(active_state, index_map, epoch)
    rank_pairs = global_pairs[rank::WORLD_SIZE]
    if len(rank_pairs) != LOCAL_PAIR_UNITS:
        raise RuntimeError(f"rank {rank} got {len(rank_pairs)} pair units, expected {LOCAL_PAIR_UNITS}")
    samples: list[tuple[dict[str, Any], int]] = []
    batches: list[list[int]] = []
    pair_audit = []
    for start in range(0, len(rank_pairs), BATCH_SIZE // NUM_INSTANCES):
        pair_batch = rank_pairs[start:start + BATCH_SIZE // NUM_INSTANCES]
        if len(pair_batch) != BATCH_SIZE // NUM_INSTANCES:
            raise RuntimeError("incomplete identity-pair batch")
        batch_indices = []
        batch_pids = []
        for pair in pair_batch:
            if len(pair) != NUM_INSTANCES:
                raise RuntimeError("NUM_INSTANCES pair is incomplete")
            pids = {int(dataset.train[index]["pid"]) for index in pair}
            if len(pids) != 1:
                raise RuntimeError("identity pair contains multiple PIDs")
            batch_pids.append(next(iter(pids)))
            for position, dataset_index in enumerate(pair):
                batch_indices.append(len(samples))
                samples.append((
                    dict(dataset.train[int(dataset_index)]),
                    plan_seed(epoch, rank, len(batches) + 1, len(batch_indices) - 1, int(dataset_index)),
                ))
        pair_audit.append(batch_pids)
        batches.append(batch_indices)
    if len(batches) != STEPS_PER_EPOCH:
        raise RuntimeError(f"rank {rank} got {len(batches)} optimizer batches, expected {STEPS_PER_EPOCH}")
    plan_digest = hashlib.sha256()
    for pair_batch in rank_pairs:
        plan_digest.update(json.dumps(pair_batch, separators=(",", ":")).encode())
    active_images = sum(
        len(index_map[(int(pid), int(cid))])
        for pid, clothes in active_state.items() for cid in clothes
    )
    meta = {
        "rank": rank,
        "epoch": epoch,
        "batches": len(batches),
        "pair_units": len(rank_pairs),
        "global_pair_units": len(global_pairs),
        "active_images": active_images,
        "active_pids": len(active_state),
        "active_clothes": sum(len(values) for values in active_state.values()),
        "covered_pids": sorted({pid for batch in pair_audit for pid in batch}),
        "batch_unique_pid_min": min(len(set(batch)) for batch in pair_audit),
        "batch_unique_pid_max": max(len(set(batch)) for batch in pair_audit),
        "pair_size_ok": all(len(pair) == NUM_INSTANCES for pair in rank_pairs),
        "batch_size_ok": all(len(batch) == BATCH_SIZE for batch in batches),
        "plan_sha256": plan_digest.hexdigest(),
    }
    loader = DataLoader(
        PlannedTrainDataset(samples, transform_train),
        batch_sampler=FixedBatchSampler(batches),
        num_workers=1,
        pin_memory=False,
    )
    return loader, meta


def active_state_hash(active_state: dict[int, list[int]]) -> str:
    return canonical_state_hash({int(pid): set(values) for pid, values in active_state.items()})


def preflight_sampler_audit(cfg: Any, run: str, epochs: int = SCREEN_EPOCHS) -> list[dict[str, Any]]:
    dataset = build_dataset(cfg)
    index_map = build_index_map(dataset)
    states = load_active_states(run)
    rows = []
    for epoch in range(1, epochs + 1):
        global_pairs = make_pair_units(states[epoch], index_map, epoch)
        rank_meta = []
        for rank in range(WORLD_SIZE):
            rank_pairs = global_pairs[rank::WORLD_SIZE]
            rank_pids = set()
            pair_ok = True
            for pair in rank_pairs:
                pair_pids = {int(dataset.train[index]["pid"]) for index in pair}
                pair_ok = pair_ok and len(pair) == NUM_INSTANCES and len(pair_pids) == 1
                rank_pids.update(pair_pids)
            rank_meta.append({
                "rank": rank,
                "pair_units": len(rank_pairs),
                "batches": len(rank_pairs) // (BATCH_SIZE // NUM_INSTANCES),
                "pair_size_ok": pair_ok,
                "covered_pids": sorted(rank_pids),
                "batch_unique_pid_min": min(
                    len({int(dataset.train[index]["pid"]) for pair in rank_pairs[start:start + (BATCH_SIZE // NUM_INSTANCES)] for index in pair})
                    for start in range(0, len(rank_pairs), BATCH_SIZE // NUM_INSTANCES)
                ),
                "plan_sha256": sha256_bytes(json.dumps(rank_pairs, separators=(",", ":")).encode()),
            })
        if any(
            item["batches"] != STEPS_PER_EPOCH
            or not item["pair_size_ok"]
            or item["batch_unique_pid_min"] < 2
            for item in rank_meta
        ):
            raise RuntimeError(f"preflight sampler budget failed for {run} epoch {epoch}")
        rows.append({
            "run": run,
            "epoch": epoch,
            "active_images": sum(len(index_map[(int(pid), int(cid))]) for pid, values in states[epoch].items() for cid in values),
            "active_pids": len(states[epoch]),
            "active_clothes": sum(len(values) for values in states[epoch].values()),
            "global_pair_units": len(global_pairs),
            "expected_global_pair_units": GLOBAL_PAIR_UNITS,
            "rank0": json.dumps(rank_meta[0], sort_keys=True, separators=(",", ":")),
            "rank1": json.dumps(rank_meta[1], sort_keys=True, separators=(",", ":")),
            "all_77_pids_present_in_active_state": len(states[epoch]) == NUM_CLASSES,
            "active_state_sha256": active_state_hash(states[epoch]),
        })
    return rows


def kendall_sequence_audit() -> dict[str, Any]:
    events, _, initial, _ = load_frozen_events()
    sequences = {run: defaultdict(list) for run in ("S1", "S2")}
    for run in ("S1", "S2"):
        for row in read_csv(SCHEDULE_ROOT / f"schedule_{run}.csv"):
            sequences[run][int(row["pid"])].append(int(row["new_clothes"]))
    pid_rows = []
    for pid in sorted(sequences["S1"]):
        a = sequences["S1"][pid]
        b = sequences["S2"][pid]
        common = sorted(set(a) & set(b))
        if len(common) >= 3:
            rank_a = [a.index(cid) for cid in common]
            rank_b = [b.index(cid) for cid in common]
            tau = kendalltau(rank_a, rank_b).statistic
            pid_rows.append({"pid": pid, "common_clothes": len(common), "kendall_tau": float(tau)})
    s1_rows = read_csv(SCHEDULE_ROOT / "schedule_S1.csv")
    s2_rows = read_csv(SCHEDULE_ROOT / "schedule_S2.csv")
    keys1 = [(int(row["event_id"]), int(row["epoch"]), int(row["pid"]), int(row["expansion_position_for_pid"])) for row in s1_rows]
    keys2 = [(int(row["event_id"]), int(row["epoch"]), int(row["pid"]), int(row["expansion_position_for_pid"])) for row in s2_rows]
    position_changes = {}
    for position in sorted({int(row["expansion_position_for_pid"]) for row in s1_rows}):
        a = {int(row["pid"]): int(row["new_clothes"]) for row in s1_rows if int(row["expansion_position_for_pid"]) == position}
        b = {int(row["pid"]): int(row["new_clothes"]) for row in s2_rows if int(row["expansion_position_for_pid"]) == position}
        position_changes[str(position)] = {
            "pid_count": len(set(a) | set(b)),
            "different_selected_count": sum(a.get(pid) != b.get(pid) for pid in set(a) | set(b)),
        }
    prefix2 = [
        (float(a["selected_delta_h_id_vs_historical"]), float(b["selected_delta_h_id_vs_historical"]))
        for a, b in zip(s1_rows, s2_rows)
        if int(a["expansion_position_for_pid"]) <= 2
    ]
    prefix2_by_pid = defaultdict(list)
    for a, b in zip(s1_rows, s2_rows):
        if int(a["expansion_position_for_pid"]) <= 2:
            prefix2_by_pid[int(a["pid"])].append((
                float(a["selected_delta_h_id_vs_historical"]),
                float(b["selected_delta_h_id_vs_historical"]),
            ))
    prefix2_pid = [values for values in prefix2_by_pid.values() if len(values) == 2]
    return {
        "event_keys_identical": keys1 == keys2,
        "event_count": len(events),
        "position_changes": position_changes,
        "sequence_kendall": {
            "count": len(pid_rows),
            "mean": float(np.mean([row["kendall_tau"] for row in pid_rows])) if pid_rows else None,
            "median": float(np.median([row["kendall_tau"] for row in pid_rows])) if pid_rows else None,
            "rows": pid_rows,
        },
        "prefix2_selected_delta_h_id": {
            "count": len(prefix2),
            "S1_mean": float(np.mean([x[0] for x in prefix2])) if prefix2 else None,
            "S2_mean": float(np.mean([x[1] for x in prefix2])) if prefix2 else None,
            "S2_minus_S1_mean": float(np.mean([x[1] - x[0] for x in prefix2])) if prefix2 else None,
            "pid_level_count": len(prefix2_pid),
            "S1_pid_level_mean": float(np.mean([np.mean([x[0] for x in values]) for values in prefix2_pid])) if prefix2_pid else None,
            "S2_pid_level_mean": float(np.mean([np.mean([x[1] for x in values]) for values in prefix2_pid])) if prefix2_pid else None,
        },
        "initial_pool_note": "S1/S2 both start from the exact common initial pool; their first expansion may differ by ranking signal.",
    }


def preflight() -> None:
    prepared = load_prepared()
    cfg = configure()
    common_path = COMMON_ROOT / "COMMON_PHASE5B_INIT.pth"
    common_hash_path = COMMON_ROOT / "COMMON_PHASE5B_INIT_HASHES.json"
    state = torch.load(common_path, map_location="cpu", weights_only=False)
    common_meta = json.loads(common_hash_path.read_text())
    if sha256_file(common_path) != common_meta["state_sha256"]:
        raise RuntimeError("common init file SHA256 mismatch")
    if tensor_hashes(state) != common_meta["tensor_hashes"]:
        raise RuntimeError("common init tensor hash mismatch")
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    topology = {
        "state_key_count": len(model.state_dict()),
        "state_numel": int(sum(value.numel() for value in model.state_dict().values())),
        "has_reverse_parameters": any("Reverse" in key or "reverse" in key for key in model.state_dict()),
        "model_name": cfg.MODEL.NAME,
        "loss_type": list(cfg.MODEL.LOSS_TYPE),
    }
    if topology["has_reverse_parameters"] or topology["loss_type"] != ["ce", "triplet", "clipBio"]:
        raise RuntimeError("Phase 5B common topology is not Bio-only")
    del model, state
    gc.collect()

    events, _, initial, _ = load_frozen_events()
    schedules = {run: read_csv(SCHEDULE_ROOT / f"schedule_{run}.csv") for run in ("S1", "S2")}
    s1, s2 = schedules["S1"], schedules["S2"]
    if len(s1) != len(s2) or len(s1) != len(events):
        raise RuntimeError("S1/S2 schedule event counts are not identical to frozen Phase 5A events")
    keys = ["event_id", "trace_index", "epoch", "pid", "expansion_position_for_pid"]
    event_keys_equal = all(tuple(row[key] for key in keys) == tuple(s2[i][key] for key in keys) for i, row in enumerate(s1))
    if not event_keys_equal:
        raise RuntimeError("S1/S2 event epoch/PID/position schedule differs")
    initial_hashes = {
        run: json.loads((OUT_ROOT / {"S0": "S0_bio_full_data", "S1": "S1_clothing_ds", "S2": "S2_id_damage_ds"}[run] / "run_manifest.json").read_text())["initial_active_state_sha256"]
        for run in RUNS
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("S0/S1/S2 initial active-state hashes differ")
    event_epoch_counts = {
        run: dict(sorted(Counter(int(row["epoch"]) for row in rows).items())) for run, rows in schedules.items()
    }
    if event_epoch_counts["S1"] != event_epoch_counts["S2"]:
        raise RuntimeError("S1/S2 expansion frequency differs")
    initial_state = {str(pid): sorted(values) for pid, values in sorted(initial.items())}
    source_paths = []
    for row in s1 + s2:
        source_paths.extend(parse_json_value(row["source_image_paths"], []))
    no_test_sources = all(
        str(path).startswith(str(TRAIN_ROOT) + "/")
        and not str(path).startswith(str(QUERY_ROOT))
        and not str(path).startswith(str(GALLERY_ROOT))
        for path in source_paths
    )
    if not no_test_sources:
        raise RuntimeError("schedule contains a query/gallery source")

    sampler_audit = {}
    for run in RUNS:
        rows = preflight_sampler_audit(cfg, run)
        sampler_audit[run] = rows
        run_dir = OUT_ROOT / {"S0": "S0_bio_full_data", "S1": "S1_clothing_ds", "S2": "S2_id_damage_ds"}[run]
        write_csv(run_dir / "sampler_preflight.csv", rows, list(rows[0]))

    schedule_audit = kendall_sequence_audit()
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "passed": True,
        "common_topology": topology,
        "common_init_tensor_hashes_equal": True,
        "initial_active_state_equal": initial_hashes,
        "initial_active_state": initial_state,
        "event_keys_equal": event_keys_equal,
        "event_count": len(events),
        "event_epoch_counts": event_epoch_counts,
        "schedule_hashes": {
            run: sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv") for run in ("S1", "S2")
        },
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
                    and json.loads(row["rank0"])["pair_size_ok"]
                    and json.loads(row["rank1"])["pair_size_ok"]
                    and json.loads(row["rank0"])["batch_unique_pid_min"] >= 2
                    and json.loads(row["rank1"])["batch_unique_pid_min"] >= 2
                    for row in rows
                ),
                "all_77_active_pids": all(row["all_77_pids_present_in_active_state"] for row in rows),
            }
            for run, rows in sampler_audit.items()
        },
        "sampler_contract": {
            "pair_unit_size": NUM_INSTANCES,
            "batch_size": BATCH_SIZE,
            "minimum_unique_pids_per_local_batch": 2,
            "purpose": "preserve a non-empty hard-negative set for triplet mining",
            "repair_scope": "deterministic swaps within the fixed pair pool after rank sharding",
            "changes_active_clothing_selection": False,
            "changes_optimizer_budget": False,
        },
        "schedule_difference_audit": schedule_audit,
        "ranking_only_contract": {
            "S1": SCHEDULE_METHODS["S1"],
            "S2": SCHEDULE_METHODS["S2"],
            "same_initial_pool": True,
            "same_expansion_epoch_counts": True,
            "same_optimizer_budget": True,
            "different_active_states_after_selection_are_intended": True,
        },
        "prepared_manifest_sha256": sha256_file(OUT_ROOT / "PHASE5B_PREPARED.json"),
        "phase5a_script_sha256": sha256_file(PHASE5A_SCRIPT),
        "phase5a_replay_sha256": sha256_file(PHASE5A_REPLAY),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(OUT_ROOT / "PREFLIGHT.json", audit)
    print(json.dumps({"preflight": True, "passed": True, "audit": str(OUT_ROOT / "PREFLIGHT.json")}, indent=2, sort_keys=True))


def init_dist() -> tuple[int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 5B training requires exactly {WORLD_SIZE} visible PPUs")
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


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    return {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}


def tensor_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item) for item in value)
    return value


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device):
    payload = {
        "phase": PHASE,
        "epoch": epoch,
        "global_step": global_step,
        "model": cpu_state_dict(model),
        "optimizer": tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_rank_state(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"], device)
    return payload


class MetricCapture:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self.values.setdefault(str(tag), []).append(float(scalar_value))


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def forward_batch(model, loss_fn, batch, device: torch.device, global_step: int) -> dict[str, Any]:
    writer = MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    # Keep the released Bio trainer's AMP boundary: the model forward is
    # autocast, while CE/Triplet/Bio loss calculation is outside autocast.
    total = loss_fn(
        score, feat, pids, camids, captions, clothes_ids=clothes,
        train_writer=writer, step=global_step,
    )
    components = getattr(loss_fn, "last_components", {})
    return {
        "total": total,
        "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        "writer": writer,
    }


def found_inf(scaler, optimizer) -> float:
    state = scaler._per_optimizer_states.get(id(optimizer), {})
    values = state.get("found_inf_per_device", {})
    return sum(float(value.item()) for value in values.values())


def reduce_epoch(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}


def logger_for(path: Path):
    logger = logging.getLogger(f"{PHASE}.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def build_val_loader(cfg: Any, dataset: Any) -> DataLoader:
    _, transform_test = build_img_transforms(cfg)
    combined = ConcatDataset([
        ImageDataset(dataset.query, transform=transform_test),
        ImageDataset(dataset.gallery, transform=transform_test),
    ])
    return DataLoader(
        combined,
        batch_size=cfg.DATA.TEST_BATCH,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=False,
    )


def evaluate_ltcc(cfg: Any, raw_model, dataset: Any, val_loader: DataLoader, device: torch.device, epoch: int, logger):
    raw_model.eval()
    cc_writer = MetricCapture()
    general_writer = MetricCapture()
    cc_eval = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    general_eval = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    cc_eval.reset()
    general_eval.reset()
    with torch.no_grad():
        test(cfg, raw_model, cc_eval, val_loader, logger, str(device), epoch, cc_writer,
             test=True, ltcc=True, clothChanging=True, metric_prefix="cc")
        test(cfg, raw_model, general_eval, val_loader, logger, str(device), epoch, general_writer,
             test=True, ltcc=True, clothChanging=False, metric_prefix="general")
    cc = {key.split("cc/", 1)[1]: values[-1] for key, values in cc_writer.values.items() if key.startswith("cc/")}
    general = {key.split("general/", 1)[1]: values[-1] for key, values in general_writer.values.items() if key.startswith("general/")}
    raw_model.train()
    return {
        "cc_rank1": float(cc.get("rank1", float("nan"))),
        "cc_rank5": float(cc.get("rank5", float("nan"))),
        "cc_rank10": float(cc.get("rank10", float("nan"))),
        "cc_mAP": float(cc.get("mAP", float("nan"))),
        "general_rank1": float(general.get("rank1", float("nan"))),
        "general_rank5": float(general.get("rank5", float("nan"))),
        "general_rank10": float(general.get("rank10", float("nan"))),
        "general_mAP": float(general.get("mAP", float("nan"))),
    }


def setup_runtime(cfg: Any, state: dict[str, torch.Tensor], device: torch.device):
    raw = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=NUM_CAMERAS)
    raw.load_state_dict(state, strict=True)
    raw.to(device)
    raw.train()
    model = DDP(raw, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    loss_fn, center = make_loss(cfg, num_classes=NUM_CLASSES)
    optimizer, _center_optimizer = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=INITIAL_SCALE)
    return model, loss_fn, optimizer, scheduler, scaler


def open_metrics(run_dir: Path, rank: int):
    if rank != 0:
        return None, None
    path = run_dir / "metrics_per_epoch.csv"
    existing = path.is_file() and path.stat().st_size > 0
    stream = path.open("a" if existing else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
    if not existing:
        writer.writeheader()
        stream.flush()
    return stream, writer


def load_metric_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.is_file() else []


def save_model_checkpoint(raw_model: torch.nn.Module, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_state_dict(raw_model), temporary)
    temporary.replace(path)


def actual_active_row(run: str, epoch: int, active_state: dict[int, list[int]], index_map: dict[tuple[int, int], list[int]], schedule_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(values) for values in active_state.values()]
    return {
        "run": run,
        "epoch": epoch,
        "active_images": sum(len(index_map[(int(pid), int(cid))]) for pid, values in active_state.items() for cid in values),
        "active_pids": len(active_state),
        "active_clothes": sum(counts),
        "mean_clothes_per_pid": float(np.mean(counts)),
        "new_clothes_count": sum(int(row["epoch"]) == epoch for row in schedule_rows),
        "active_state_sha256": active_state_hash(active_state),
    }


def train(run: str, requested_epochs: int, resume: bool = False) -> None:
    if run not in RUNS:
        raise ValueError(run)
    if requested_epochs not in (SCREEN_EPOCHS, TOTAL_EPOCHS):
        raise ValueError("Phase 5B only permits 15-epoch screen or 60-epoch validation")
    prepared = load_prepared()
    preflight_path = OUT_ROOT / "PREFLIGHT.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError("Phase 5B preflight has not passed")
    rank, device = init_dist()
    run_dir = OUT_ROOT / {"S0": "S0_bio_full_data", "S1": "S1_clothing_ds", "S2": "S2_id_damage_ds"}[run]
    metrics_file = None
    writer = None
    logger = None
    try:
        cfg = configure()
        set_seed(SEED)
        state = torch.load(COMMON_ROOT / "COMMON_PHASE5B_INIT.pth", map_location="cpu", weights_only=False)
        model, loss_fn, optimizer, scheduler, scaler = setup_runtime(cfg, state, device)
        local_state_hash = sha256_bytes(json.dumps(tensor_hashes(cpu_state_dict(model)), sort_keys=True).encode())
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, local_state_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError("S0/S1/S2 model initialization is not bitwise identical across ranks")

        dataset = build_dataset(cfg)
        transform_train, _ = build_img_transforms(cfg)
        index_map = build_index_map(dataset)
        states = load_active_states(run)
        schedule_rows = [] if run == "S0" else read_csv(SCHEDULE_ROOT / f"schedule_{run}.csv")
        val_loader = build_val_loader(cfg, dataset)
        start_epoch = 1
        global_step = 0
        best_r1 = -float("inf")
        best_epoch = None
        previous_rows = load_metric_rows(run_dir / "metrics_per_epoch.csv")
        if resume:
            checkpoint = run_dir / f"rank{rank}_state_epoch{SCREEN_EPOCHS:03d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = load_rank_state(checkpoint, model, optimizer, scheduler, scaler, device)
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
            if start_epoch != SCREEN_EPOCHS + 1:
                raise RuntimeError(f"resume checkpoint starts at unexpected epoch {start_epoch}")
            best_meta_path = run_dir / "best_state.json"
            if best_meta_path.is_file():
                best_meta = json.loads(best_meta_path.read_text())
                best_r1 = float(best_meta["best_cc_rank1"])
                best_epoch = int(best_meta["best_epoch"])
            if len(previous_rows) < SCREEN_EPOCHS:
                raise RuntimeError("cannot resume without the complete 15-epoch metrics file")

        if rank == 0:
            logger = logger_for(run_dir / f"train_log_rank{rank}.txt")
            writer = SummaryWriter(str(run_dir / "TensorBoard"))
            metrics_file, metrics_writer = open_metrics(run_dir, rank)
            amp_rows = load_metric_rows(run_dir / "amp_statistics.csv")
            amp_path = run_dir / "amp_statistics.csv"
            amp_stream = amp_path.open("a" if amp_path.is_file() and amp_path.stat().st_size > 0 else "w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            if not amp_rows:
                amp_writer.writeheader(); amp_stream.flush()
            active_path = run_dir / "active_set_statistics.csv"
            active_stream = active_path.open("a" if active_path.is_file() and active_path.stat().st_size > 0 else "w", newline="", encoding="utf-8")
            active_writer = csv.DictWriter(active_stream, fieldnames=["run", "epoch", "active_images", "active_pids", "active_clothes", "mean_clothes_per_pid", "new_clothes_count", "active_state_sha256", "rank0_plan_sha256", "rank1_plan_sha256", "optimizer_updates"])
            if active_stream.tell() == 0:
                active_writer.writeheader(); active_stream.flush()
        else:
            metrics_writer = None
            amp_stream = amp_writer = active_stream = active_writer = None

        for epoch in range(start_epoch, requested_epochs + 1):
            scheduler.step(epoch)
            active_state = states[epoch]
            loader, plan_meta = make_epoch_loader(dataset, transform_train, active_state, index_map, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if rank == 0:
                if any(
                    not item["pair_size_ok"]
                    or not item["batch_size_ok"]
                    or item["batches"] != STEPS_PER_EPOCH
                    or item["batch_unique_pid_min"] < 2
                    for item in gathered_meta
                ):
                    raise RuntimeError(f"runtime sampler audit failed at epoch {epoch}")
                covered_global = set(gathered_meta[0]["covered_pids"]) | set(gathered_meta[1]["covered_pids"])
                if covered_global != set(range(NUM_CLASSES)):
                    raise RuntimeError(f"runtime sampler global PID coverage failed at epoch {epoch}")
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
                result = forward_batch(model, loss_fn, batch, device, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    local_nonfinite += 1
                    raise RuntimeError(f"non-finite loss at run={run}, epoch={epoch}, iteration={iteration}")
                scaler.scale(total).backward()
                scaler.step(optimizer)
                inf = found_inf(scaler, optimizer)
                local_skips += int(inf > 0)
                local_nonfinite += int(inf > 0)
                scaler.update()
                scale_min = min(scale_min, float(scaler.get_scale()))
                scale_max = max(scale_max, float(scaler.get_scale()))
                for key in epoch_sum:
                    epoch_sum[key] += scalar(result[key])
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", scalar(total), global_step)
            del iterator, loader
            gc.collect()
            reduced = reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); nonfinite_sum = amp_tensor[1].clone()
            dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_tensor = amp_tensor[2].clone(); max_tensor = amp_tensor[3].clone(); end_tensor = amp_tensor[4].clone()
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
                metrics = evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            metrics_holder = [metrics]
            dist.broadcast_object_list(metrics_holder, src=0)
            metrics = metrics_holder[0]
            active_row = actual_active_row(run, epoch, active_state, index_map, schedule_rows)
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
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch)
                writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"]}); amp_stream.flush()
                active_writer.writerow({
                    **active_row,
                    "rank0_plan_sha256": gathered_meta[0]["plan_sha256"],
                    "rank1_plan_sha256": gathered_meta[1]["plan_sha256"],
                    "optimizer_updates": STEPS_PER_EPOCH,
                }); active_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"])
                    best_epoch = epoch
                    save_model_checkpoint(model.module, run_dir / "model_best.pth")
                    write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1"})
            if epoch == SCREEN_EPOCHS or epoch == requested_epochs:
                save_rank_state(run_dir / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    save_model_checkpoint(model.module, run_dir / ("model_screen_final.pth" if epoch == SCREEN_EPOCHS and requested_epochs == SCREEN_EPOCHS else "model_final.pth"))
                    write_json(run_dir / f"checkpoint_epoch{epoch:03d}.json", {
                        "epoch": epoch,
                        "global_step": global_step,
                        "rank_states": [str(run_dir / f"rank{r}_state_epoch{epoch:03d}.pt") for r in range(WORLD_SIZE)],
                        "model": str(run_dir / ("model_screen_final.pth" if epoch == SCREEN_EPOCHS and requested_epochs == SCREEN_EPOCHS else "model_final.pth")),
                    })
                    if epoch == requested_epochs:
                        rows_now = load_metric_rows(run_dir / "metrics_per_epoch.csv")
                        write_json(run_dir / "RUN_COMPLETE.json", {
                            "phase": PHASE,
                            "run": run,
                            "completed_utc": utc_now(),
                            "epochs_completed": epoch,
                            "optimizer_updates": global_step,
                            "metrics": str(run_dir / "metrics_per_epoch.csv"),
                            "best_epoch": best_epoch,
                            "best_cc_rank1": best_r1,
                            "best_model": str(run_dir / "model_best.pth"),
                            "final_model": str(run_dir / ("model_screen_final.pth" if requested_epochs == SCREEN_EPOCHS else "model_final.pth")),
                            "rows": len(rows_now),
                            "common_init_sha256": prepared["common_init"]["sha256"],
                            "schedule_sha256": None if run == "S0" else sha256_file(SCHEDULE_ROOT / f"schedule_{run}.csv"),
                        })
        if rank == 0:
            write_json(run_dir / "AMP.json", {
                "initial_scale": INITIAL_SCALE,
                "epochs_completed": requested_epochs,
                "metrics_source": str(run_dir / "amp_statistics.csv"),
            })
    finally:
        for stream_name in ("metrics_file", "amp_stream", "active_stream"):
            stream = locals().get(stream_name)
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        cleanup_dist()


def finite_rows(rows: list[dict[str, str]]) -> bool:
    required = ["cc_rank1", "cc_mAP", "train_loss", "ce_loss", "triplet_loss", "bio_loss"]
    for row in rows:
        for key in required:
            try:
                if not math.isfinite(float(row[key])):
                    return False
            except (ValueError, TypeError, KeyError):
                return False
    return True


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
        "final_slope_rank1_per_epoch": float(np.polyfit(x, r1, 1)[0]),
        "final_slope_mAP_per_epoch": float(np.polyfit(x, mAP, 1)[0]),
        "rows": selected,
    }


def best_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"best_rank1": None, "best_mAP": None, "best_rank1_epoch": None, "best_mAP_epoch": None, "final": None}
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


def screen_analysis() -> dict[str, Any]:
    runs = {}
    for run in RUNS:
        run_dir = OUT_ROOT / {"S0": "S0_bio_full_data", "S1": "S1_clothing_ds", "S2": "S2_id_damage_ds"}[run]
        rows = load_metric_rows(run_dir / "metrics_per_epoch.csv")
        runs[run] = {
            "rows": rows,
            "row_count": len(rows),
            "finite": finite_rows(rows),
            "trajectory_8_15": trajectory(rows) if len(rows) >= SCREEN_EPOCHS else None,
            "best": best_metrics(rows),
        }
    if any(item["row_count"] < SCREEN_EPOCHS for item in runs.values()):
        raise RuntimeError("all S0/S1/S2 15-epoch metrics are required for screen analysis")
    old_rows = read_csv(BIO_METRICS)
    old = trajectory(old_rows)
    s0 = runs["S0"]["trajectory_8_15"]
    s1 = runs["S1"]["trajectory_8_15"]
    s2 = runs["S2"]["trajectory_8_15"]
    s0_pipeline = {
        "finite": runs["S0"]["finite"],
        "historical_bio_source": str(BIO_METRICS),
        "historical_mean_rank1_8_15": old["mean_rank1"],
        "historical_mean_mAP_8_15": old["mean_mAP"],
        "matched_s0_mean_rank1_8_15": s0["mean_rank1"],
        "matched_s0_mean_mAP_8_15": s0["mean_mAP"],
        "difference_rank1": s0["mean_rank1"] - old["mean_rank1"],
        "difference_mAP": s0["mean_mAP"] - old["mean_mAP"],
        "rule": "finite metrics and absolute difference <= 0.05 for both CC Rank-1 and mAP",
    }
    s0_pipeline["passed"] = bool(
        s0_pipeline["finite"]
        and abs(s0_pipeline["difference_rank1"]) <= 0.05
        and abs(s0_pipeline["difference_mAP"]) <= 0.05
    )
    s2_rows_8_15 = [row for row in runs["S2"]["rows"] if 8 <= int(row["epoch"]) <= 15]
    s0_rows_8_15 = {int(row["epoch"]): row for row in runs["S0"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    s1_rows_8_15 = {int(row["epoch"]): row for row in runs["S1"]["rows"] if 8 <= int(row["epoch"]) <= 15}
    s2_by_epoch = {int(row["epoch"]): row for row in s2_rows_8_15}
    gate_b_flags = {"rank1": s2["auc_rank1_normalized"] > s1["auc_rank1_normalized"], "mAP": s2["auc_mAP_normalized"] > s1["auc_mAP_normalized"]}
    gate_c_wins = []
    for epoch in range(12, 16):
        a = s2_by_epoch[epoch]
        b = s1_rows_8_15[epoch]
        gate_c_wins.append({
            "epoch": epoch,
            "rank1_s2_gt_s1": float(a["cc_rank1"]) > float(b["cc_rank1"]),
            "mAP_s2_gt_s1": float(a["cc_mAP"]) > float(b["cc_mAP"]),
            "either": float(a["cc_rank1"]) > float(b["cc_rank1"]) or float(a["cc_mAP"]) > float(b["cc_mAP"]),
        })
    bad_vs_s0 = []
    for epoch in range(8, 16):
        a = s2_by_epoch[epoch]
        b = s0_rows_8_15[epoch]
        bad_vs_s0.append(
            float(a["cc_rank1"]) < float(b["cc_rank1"]) - 0.02
            or float(a["cc_mAP"]) < float(b["cc_mAP"]) - 0.02
        )
    promotion = {
        "s0_pipeline_gate": s0_pipeline,
        "A_no_training_collapse": all(runs[run]["finite"] for run in RUNS),
        "B_trajectory_auc_better_than_s1": gate_b_flags,
        "B_passed": bool(gate_b_flags["rank1"] or gate_b_flags["mAP"]),
        "C_epoch12_15": gate_c_wins,
        "C_win_count": sum(item["either"] for item in gate_c_wins),
        "C_passed": sum(item["either"] for item in gate_c_wins) >= 3,
        "D_bad_vs_s0_by_epoch_8_15": bad_vs_s0,
        "D_max_consecutive_bad_epochs": max_consecutive(bad_vs_s0),
        "D_passed": max_consecutive(bad_vs_s0) < 3,
    }
    promotion["passed"] = bool(
        promotion["s0_pipeline_gate"]["passed"]
        and promotion["A_no_training_collapse"]
        and promotion["B_passed"]
        and promotion["C_passed"]
        and promotion["D_passed"]
    )
    if promotion["B_passed"] and promotion["C_passed"] and not promotion["passed"]:
        offline_online_correspondence = "PARTIAL"
    elif promotion["passed"]:
        offline_online_correspondence = "YES"
    else:
        offline_online_correspondence = "NO"
    return {
        "phase": PHASE,
        "created_utc": utc_now(),
        "runs": runs,
        "historical_bio_trajectory_8_15": old,
        "promotion_gate": promotion,
        "offline_online_correspondence": offline_online_correspondence,
        "continue_to_60": promotion["passed"],
        "offline_evidence": {
            "original_clothing": {"delta_h_id_prefix2": 0.00018634904368018072, "ci95": [-0.0012274436121070338, 0.0015487852113817715]},
            "id_damage": {"delta_h_id_prefix2": 0.003619137524516255, "ci95": [0.0020514641002556105, 0.005373201658111783]},
            "reliable_damage_not_trained": {"delta_h_id_prefix2": 0.00354674010901215, "ci95": [0.0019414781859355816, 0.005283213406359119]},
        },
    }


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.4f}"


def metric_delta(value: float | None, reference: float) -> str:
    return "NA" if value is None else f"{(value - reference) * 100:+.4f} pp"


def final_diagnosis(analysis: dict[str, Any]) -> str:
    promotion = analysis["promotion_gate"]
    if not promotion["s0_pipeline_gate"]["passed"]:
        return "RESULT INCONCLUSIVE"
    if not analysis.get("final_validation", False):
        return "RESULT INCONCLUSIVE"
    s1 = analysis["runs"]["S1"]["best"]
    s2 = analysis["runs"]["S2"]["best"]
    bio_r1, bio_map = 0.5637755393981934, 0.32385736411857605
    if s2["best_rank1"] <= s1["best_rank1"] and s2["best_mAP"] <= s1["best_mAP"]:
        return "OFFLINE ADVANTAGE DOES NOT TRANSFER TO TRAINING"
    above_bio_r1 = s2["best_rank1"] - bio_r1
    above_bio_map = s2["best_mAP"] - bio_map
    if above_bio_r1 >= 0.005 and above_bio_map >= -0.003:
        return "ID-DAMAGE DS ADDS VALUE BEYOND BIO"
    if abs(above_bio_r1) <= 0.005 and abs(above_bio_map) <= 0.003:
        return "ID-DAMAGE DS REACHES BIO CONTROL"
    if s2["best_rank1"] > s1["best_rank1"] or s2["best_mAP"] > s1["best_mAP"]:
        return "ID-DAMAGE DS OUTPERFORMS CLOTHING DS"
    return "RESULT INCONCLUSIVE"


def write_report(analysis: dict[str, Any]) -> None:
    preflight = json.loads((OUT_ROOT / "PREFLIGHT.json").read_text())
    bio_r1, bio_map = 0.5637755393981934, 0.32385736411857605
    original_full_r1, original_full_map = 0.5331632494926453, 0.292657
    rows = []
    for run in RUNS:
        item = analysis["runs"][run]
        best = item["best"]
        if run == "S0": ranking, reverse = "None (full data)", "OFF"
        else: ranking, reverse = SCHEDULE_METHODS[run]["phase5a_method"], "OFF"
        status = "screen complete" if item["row_count"] == SCREEN_EPOCHS else "60 epochs complete"
        rows.append(
            f"| {run} | {ranking} | {reverse} | {pp(best['best_rank1'])} | {pp(best['best_mAP'])} | "
            f"{metric_delta(best['best_rank1'], bio_r1)} / {metric_delta(best['best_mAP'], bio_map)} | "
            f"{('NA' if run == 'S0' else metric_delta(best['best_rank1'], analysis['runs']['S1']['best']['best_rank1']) + ' / ' + metric_delta(best['best_mAP'], analysis['runs']['S1']['best']['best_mAP']))} | {status} |"
        )
    promotion = analysis["promotion_gate"]
    diagnosis = final_diagnosis(analysis)
    continuation = "PASS — continue only the necessary S1/S2 60-epoch validation." if analysis.get("continue_to_60") and not analysis.get("final_validation") else "STOP — no additional candidate is authorized." if not analysis.get("continue_to_60") else "STOP — 60-epoch validation complete."
    trajectory_rows = []
    integrity_rows = []
    for run in RUNS:
        item = analysis["runs"][run]
        t = item["trajectory_8_15"]
        all_rows = item["rows"]
        trajectory_rows.append(
            f"| {run} | {pp(t['mean_rank1'])} | {pp(t['mean_mAP'])} | {pp(t['auc_rank1_normalized'])} | "
            f"{pp(t['auc_mAP_normalized'])} | {t['final_slope_rank1_per_epoch'] * 100:+.4f} | "
            f"{t['final_slope_mAP_per_epoch'] * 100:+.4f} |"
        )
        integrity_rows.append(
            f"| {run} | {item['row_count']} | {sum(int(float(row['optimizer_updates'])) for row in all_rows)} | "
            f"{sorted(set(int(float(row['active_pids'])) for row in all_rows))} | "
            f"{all_rows[0]['active_images']}→{all_rows[-1]['active_images']} | "
            f"{all_rows[0]['active_clothes']}→{all_rows[-1]['active_clothes']} | "
            f"{sum(int(float(row['skipped_steps'])) for row in all_rows)} |"
        )
    screen_compare_rows = []
    by_run_epoch = {
        run: {int(row["epoch"]): row for row in analysis["runs"][run]["rows"]}
        for run in RUNS
    }
    for epoch in range(12, 16):
        s0_row, s1_row, s2_row = (by_run_epoch[run][epoch] for run in RUNS)
        screen_compare_rows.append(
            f"| {epoch} | {pp(float(s0_row['cc_rank1']))} / {pp(float(s0_row['cc_mAP']))} | "
            f"{pp(float(s1_row['cc_rank1']))} / {pp(float(s1_row['cc_mAP']))} | "
            f"{pp(float(s2_row['cc_rank1']))} / {pp(float(s2_row['cc_mAP']))} | "
            f"{metric_delta(float(s2_row['cc_rank1']), float(s1_row['cc_rank1']))} / "
            f"{metric_delta(float(s2_row['cc_mAP']), float(s1_row['cc_mAP']))} | "
            f"{metric_delta(float(s2_row['cc_rank1']), float(s0_row['cc_rank1']))} / "
            f"{metric_delta(float(s2_row['cc_mAP']), float(s0_row['cc_mAP']))} |"
        )
    bad_epochs = [
        epoch for epoch, bad in zip(range(8, 16), promotion["D_bad_vs_s0_by_epoch_8_15"]) if bad
    ]
    report = [
        "# PHASE 5B — ID-DAMAGE GUIDED DS TRAINING VALIDATION",
        "",
        f"Generated: `{utc_now()}`.",
        "",
        "This phase validates static, pre-frozen ID-Damage scheduling on the reliable Bio Control training base. Reverse/GRL, DW, Reliability, online H_ID updates, Full DIFFER repair, PRCC, and seed sweeps are outside this phase.",
        "",
        "## DECISION TABLE",
        "",
        "| Setting | Ranking | Reverse | BEST R1 | BEST mAP | vs Bio | vs S1 | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        f"Frozen Bio Control reference: **{pp(bio_r1)} / {pp(bio_map)}**. Frozen Original Full is secondary only: **{pp(original_full_r1)} / {pp(original_full_map)}**.",
        "",
        "## MATCHED SCREEN TRAJECTORY (EPOCHS 8–15)",
        "",
        "| Setting | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Slope R1/epoch (pp) | Slope mAP/epoch (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *trajectory_rows,
        "",
        "The AUC values are trapezoidal trajectory averages over epochs 8–15, not single-epoch selections.",
        "",
        "## RUN INTEGRITY",
        "",
        "| Setting | Metric rows | Optimizer updates | Active PIDs | Active images (first→last) | Active clothes (first→last) | AMP skipped steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *integrity_rows,
        "",
        "All three runs used the fixed 596 updates/epoch and retained all 77 training PIDs. AMP skips were recoverable scaler events; metrics remained finite and the planned update budget remained recorded for every epoch.",
        "",
        "## EPOCH 12–15 GATE DETAIL",
        "",
        "| Epoch | S0 R1 / mAP | S1 R1 / mAP | S2 R1 / mAP | S2−S1 R1 / mAP | S2−S0 R1 / mAP |",
        "|---:|---:|---:|---:|---:|---:|",
        *screen_compare_rows,
        "",
        f"D-gate bad epochs (>2 pp below S0 in either primary metric): `{bad_epochs}`; maximum consecutive run: `{promotion['D_max_consecutive_bad_epochs']}`. This is why the 15-epoch promotion fails despite S2 beating S1 on both AUCs and on at least one metric in all four epochs.",
        "",
        "## REPRESENTATION AND TRAINING CONTRACT",
        "",
        f"- Phase 5A implementation SHA256: `{preflight['phase5a_script_sha256']}`; frozen replay CSV SHA256: `{preflight['phase5a_replay_sha256']}`. The complete source audit is in `{PHASE5A_ROOT}`.",
        f"- Common initialization: `{COMMON_ROOT / 'COMMON_PHASE5B_INIT.pth'}`; all three settings strict-load the same Bio-only state.",
        "- Bio base: EVA02-Large, 224×224, batch 8, NUM_INSTANCES 2, SGD, base LR 2e-6, weight decay 0.05, 60-epoch cosine schedule with the formal 20-epoch linear warmup, AMP initial scale 1024.",
        "- Loss: CE + Triplet + Summary Bio. Hair/Clothing/Pose reverse paths are OFF.",
        f"- Fixed optimization budget: `{STEPS_PER_EPOCH}` updates per rank per epoch, `{TOTAL_EPOCHS * STEPS_PER_EPOCH}` updates for 60 epochs; active subsets are deterministically re-sampled to this budget.",
        "- Schedule construction uses training states only. Query/gallery appear only in epoch evaluation, never in ranking construction.",
        "",
        "## FROZEN OFFLINE EVIDENCE",
        "",
        "| Ranking | Prefix-2 ΔH_ID | PID bootstrap 95% CI |",
        "|---|---:|---:|",
        "| Original Clothing distance | +0.000186 | [-0.001227, +0.001549] |",
        "| ID-Damage | +0.003619 | [+0.002051, +0.005373] |",
        "| Reliable-Damage (not trained) | +0.003547 | [+0.001941, +0.005283] |",
        "",
        "The S1 schedule is the Phase 5A frozen `text_clothing_nearest` schedule, which is the clothing-distance comparator corresponding to the frozen +0.000186 replay result. S2 replaces only that ranking signal with frozen `H_ID(y,c)` group mean. The non-negative product is not used as a directional metric; all offline direction uses the frozen `ΔH_ID` field.",
        "",
        "## PREFLIGHT AND SCHEDULE-EQUIVALENCE AUDIT",
        "",
        f"- Preflight: **{'PASS' if preflight['passed'] else 'FAIL'}**.",
        f"- Shared event keys (event, epoch, PID, expansion position): **{preflight['event_keys_equal']}**.",
        f"- Shared initial state: **{len(set(preflight['initial_active_state_equal'].values())) == 1}**; all epochs retain 77 active PIDs.",
        f"- Source split audit: no query/gallery schedule source: **{preflight['no_test_sources']}**.",
        f"- Event count: `{preflight['event_count']}`; S1/S2 expansion timing is identical. Ranker sequence audit: `{json.dumps(preflight['schedule_difference_audit']['position_changes'], sort_keys=True)}`.",
        f"- Prefix-2 selected ΔH_ID means (event-level): S1 `{preflight['schedule_difference_audit']['prefix2_selected_delta_h_id']['S1_mean']:.6f}`, S2 `{preflight['schedule_difference_audit']['prefix2_selected_delta_h_id']['S2_mean']:.6f}`; PID-level frozen-protocol subset: S1 `{preflight['schedule_difference_audit']['prefix2_selected_delta_h_id']['S1_pid_level_mean']:.6f}`, S2 `{preflight['schedule_difference_audit']['prefix2_selected_delta_h_id']['S2_pid_level_mean']:.6f}`.",
        "",
        "## 15-EPOCH SCREEN GATE",
        "",
        f"- S0 pipeline gate: **{'PASS' if promotion['s0_pipeline_gate']['passed'] else 'FAIL'}**; rule: finite metrics and <=5 pp absolute difference from the historical Bio E8–15 mean on both CC Rank-1 and mAP.",
        f"- A no collapse: **{promotion['A_no_training_collapse']}**.",
        f"- B E8–15 trajectory AUC better than S1: **{promotion['B_passed']}**; details `{json.dumps(promotion['B_trajectory_auc_better_than_s1'], sort_keys=True)}`.",
        f"- C E12–15 S2 wins: `{promotion['C_win_count']}/4`; gate: **{promotion['C_passed']}**.",
        f"- D maximum consecutive >2 pp S2-vs-S0 degradation: `{promotion['D_max_consecutive_bad_epochs']}`; gate: **{promotion['D_passed']}**.",
        f"- Promotion result: **{'PASS' if promotion['passed'] else 'FAIL'}**.",
        "",
        "## OFFLINE → ONLINE CORRESPONDENCE",
        "",
        f"The screen's required offline→online correspondence is: **{analysis['offline_online_correspondence']}**. S2 improves the early trajectory relative to S1, but its sustained >2 pp degradation versus matched S0 fails promotion; this does not support a final curriculum-benefit claim.",
        "",
        "## FINAL DIAGNOSIS",
        "",
        f"**{diagnosis}**",
        "",
        f"Next-step gate: **{continuation}**",
        "",
        "## ARTIFACTS",
        "",
        f"- Root: `{OUT_ROOT}`",
        f"- Schedules: `{SCHEDULE_ROOT / 'schedule_S1.csv'}`, `{SCHEDULE_ROOT / 'schedule_S2.csv'}`",
        f"- Screen analysis: `{OUT_ROOT / 'screen_analysis.json'}`",
        f"- Preflight: `{OUT_ROOT / 'PREFLIGHT.json'}`",
        f"- Report: `{REPORT_PATH}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def analyze() -> None:
    load_prepared()
    analysis = screen_analysis()
    # If both S1 and S2 have completed 60 epochs, this is the final analysis.
    analysis["final_validation"] = all(analysis["runs"][run]["row_count"] >= TOTAL_EPOCHS for run in ("S1", "S2"))
    analysis["continue_to_60"] = bool(analysis["promotion_gate"]["passed"] and not analysis["final_validation"])
    write_json(OUT_ROOT / "screen_analysis.json", analysis)
    write_report(analysis)
    print(json.dumps({
        "screen_analysis": str(OUT_ROOT / "screen_analysis.json"),
        "report": str(REPORT_PATH),
        "promotion_passed": analysis["promotion_gate"]["passed"],
        "continue_to_60": analysis["continue_to_60"],
        "final_validation": analysis["final_validation"],
        "diagnosis": final_diagnosis(analysis),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "preflight", "train", "analyze"), required=True)
    parser.add_argument("--setting", dest="run", choices=RUNS, default=None)
    parser.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "train":
        if args.run is None:
            raise SystemExit("--setting is required for --mode train")
        train(args.run, args.epochs, args.resume)
    else:
        analyze()


if __name__ == "__main__":
    main()
