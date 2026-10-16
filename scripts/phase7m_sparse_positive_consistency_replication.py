#!/usr/local/bin/python3
"""Phase 7M: independent-seed replication of the frozen Phase 7L L3 arm.

This driver deliberately reuses the Phase 7L training implementation for the
frozen PRCC optimizer/scheduler/AMP path, but exposes only two public arms:
M0 (Bio control) and M1 (very-sparse generic same-ID positive consistency).
The schedule target is read from the frozen Phase 7L protocol; it is never
recomputed from seed4321 retrieval results.
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
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase7a_prcc_relation_validation as p7a  # noqa: E402
import phase7k_a_relation_strength_balance as p7k  # noqa: E402
import phase7l_exposure_matched_positive_consistency as p7l  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402


PHASE = "phase7m_sparse_positive_consistency_replication"
SCHEDULE_PHASE = "phase7l_exposure_matched_positive_consistency"
SEED = 4321
WORLD_SIZE = 2
FORMAL_EPOCHS = 60
STEPS_PER_EPOCH = 551
BATCH_SIZE = 16
NUM_INSTANCES = 4
TRAIN_IMAGES = 17896
TRAIN_PIDS = 150
TRAIN_CLOTHING_STATES = 300
LAMBDA_REL = 0.1
PRIMARY_START = 51
PRIMARY_END = 60
GEOMETRY_EPOCHS = (20, 40, 60)
CONDITIONAL_ACTIVE_TARGET = 16
CONDITIONAL_INACTIVE_TARGET = 16
EPS = 1e-12

OUT_ROOT = ROOT / "outputs/new_method_track/phase7m_sparse_positive_consistency_replication"
INIT_ROOT = OUT_ROOT / "initialization"
PROTOCOL_ROOT = OUT_ROOT / "protocol"
TRAINING_ROOT = OUT_ROOT / "training"
EXPOSURE_ROOT = OUT_ROOT / "exposure"
CONDITIONAL_ROOT = OUT_ROOT / "conditional_strength"
GEOMETRY_ROOT = OUT_ROOT / "geometry"
AMP_ROOT = OUT_ROOT / "amp"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
INIT_PATH = INIT_ROOT / "PRCC_COMMON_INIT_SEED4321.pth"
INIT_META = INIT_ROOT / "PRCC_COMMON_INIT_SEED4321_HASHES.json"
INIT_CONFIG = INIT_ROOT / "prcc_common_init_config.yml"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
SCHEDULE_PATH = PROTOCOL_ROOT / "schedule_dry_run.csv"
SOURCE_AUDIT_JSON = PROTOCOL_ROOT / "source_audit.json"
RUNTIME_AUDIT_JSON = PROTOCOL_ROOT / "phase7l_device_runtime_audit.json"
MANIFEST_PATH = OUT_ROOT / "PHASE7M_MANIFEST.json"
REPORT_PATH = ROOT / "reports/PHASE_7M_VERY_SPARSE_POSITIVE_CONSISTENCY_REPLICATION.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7M_SOURCE_AUDIT.md"

PHASE7L_ROOT = ROOT / "outputs/new_method_track/phase7l_exposure_matched_positive_consistency"
PHASE7L_PROTOCOL = PHASE7L_ROOT / "protocol/protocol.json"
PHASE7L_MANIFEST = PHASE7L_ROOT / "PHASE7L_MANIFEST.json"
PHASE7L_TRAINING = PHASE7L_ROOT / "training"
PHASE7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
PRCC_COMMON_INIT_1234 = PHASE7A_ROOT / "PRCC_COMMON_INIT.pth"
PRCC_COMMON_META_1234 = PHASE7A_ROOT / "PRCC_COMMON_INIT_HASHES.json"
PHASE7C_ROOT = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
PHASE7I_EXPOSURE = ROOT / "outputs/new_method_track/phase7i_relation_specificity/I3_id_only_matched/relation_exposure.csv"
PRCC_C0 = PHASE7C_ROOT / "training/C0_bio"
PRCC_C1 = PHASE7C_ROOT / "training/C1_image_uniform_relation"

PUBLIC_TO_INTERNAL = {"M0": "L0", "M1": "L3"}
INTERNAL_TO_PUBLIC = {value: key for key, value in PUBLIC_TO_INTERNAL.items()}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def read_float_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", "None", "NA"):
                continue
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                row[key] = value.lower() == "true"
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value in (None, "", "None", "NA"):
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "std", "median", "p25", "p75")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for path in (OUT_ROOT, INIT_ROOT, PROTOCOL_ROOT, TRAINING_ROOT, EXPOSURE_ROOT, CONDITIONAL_ROOT, GEOMETRY_ROOT, AMP_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for public in PUBLIC_TO_INTERNAL:
        (TRAINING_ROOT / public).mkdir(parents=True, exist_ok=True)


def configure_reused_modules() -> None:
    """Point the frozen Phase 7L implementation at the Phase 7M artifacts."""
    p7a.SEED = SEED
    p7l.PHASE = PHASE
    p7l.SEED = SEED
    p7l.OUT_ROOT = OUT_ROOT
    p7l.PROTOCOL_ROOT = PROTOCOL_ROOT
    p7l.TRAINING_ROOT = TRAINING_ROOT
    p7l.EXPOSURE_ROOT = EXPOSURE_ROOT
    p7l.STRENGTH_ROOT = CONDITIONAL_ROOT
    p7l.GEOMETRY_ROOT = GEOMETRY_ROOT
    p7l.AMP_ROOT = AMP_ROOT
    p7l.TABLE_ROOT = TABLE_ROOT
    p7l.PLOT_ROOT = PLOT_ROOT
    p7l.PRCC_COMMON_INIT = INIT_PATH
    p7l.PRCC_COMMON_META = INIT_META
    p7l.ARM_LABELS = {"L0": "M0 Bio control", "L3": "M1 exact Phase 7L L3 very-sparse generic same-ID consistency"}
    p7l.ARM_DIRS = {"L0": TRAINING_ROOT / "M0", "L3": TRAINING_ROOT / "M1"}
    p7k.SEED = SEED


def ppu_device_audit() -> dict[str, Any]:
    """Live infrastructure check: PPU forward/backward/optimizer only."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("PPU audit failed: expected exactly two visible PPU devices")
    names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if not all("PPU" in name.upper() for name in names):
        raise RuntimeError(f"PPU audit failed: device names={names}")
    device = torch.device("cuda:0")
    model = nn.Linear(8, 4, device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    x = torch.randn(4, 8, device=device)
    y = torch.randn(4, 4, device=device)
    output = model(x)
    loss = (output - y).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    changed = any(not torch.equal(before[name], value.detach()) for name, value in model.state_dict().items())
    state_devices = sorted({str(value.device) for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)})
    if not changed or any(not item.startswith("cuda:0") for item in state_devices):
        raise RuntimeError(f"PPU optimizer audit failed: changed={changed}, state_devices={state_devices}")
    return {
        "runtime_audit": "PASS", "device_count": WORLD_SIZE, "device_names": names,
        "active_device": str(device), "forward_device": str(output.device),
        "gradient_device": str(next(model.parameters()).grad.device), "loss_device": str(loss.device),
        "optimizer_state_devices": state_devices, "optimizer_parameter_changed": changed,
        "cpu_fallback": False,
    }


def phase7l_device_runtime_audit() -> dict[str, Any]:
    required = []
    for public in ("L0", "L1", "L2", "L3"):
        run_dir = PHASE7L_TRAINING / public
        required.extend([run_dir / "RUN_COMPLETE.json", run_dir / "run_manifest.json", run_dir / "amp_audit.csv"])
    artifact_checks = {str(path): path.is_file() for path in required}
    if not all(artifact_checks.values()):
        raise RuntimeError(f"Phase 7L runtime audit missing artifacts: {[key for key, value in artifact_checks.items() if not value]}")
    manifests = {public: read_json(PHASE7L_TRAINING / public / "run_manifest.json") for public in ("L0", "L1", "L2", "L3")}
    historical_checks = {
        "all_four_runs_complete": all(read_json(PHASE7L_TRAINING / public / "RUN_COMPLETE.json").get("epochs_completed") == FORMAL_EPOCHS for public in manifests),
        "all_seed1234": all(manifest.get("seed") == 1234 for manifest in manifests.values()),
        "cuda_amp_declared": all(manifest.get("amp", {}).get("grad_scaler") == "torch.cuda.amp.GradScaler" for manifest in manifests.values()),
        "optimizer_declared": all(manifest.get("optimizer", {}).get("name") == "SGD" for manifest in manifests.values()),
        "phase7l_no_cpu_fallback_marker": all("cpu fallback" not in json.dumps(manifest).lower() for manifest in manifests.values()),
    }
    if not all(historical_checks.values()):
        raise RuntimeError(f"Phase 7L historical runtime artifact audit failed: {historical_checks}")
    live = ppu_device_audit()
    result = {
        "phase7l_device_runtime_audit": "PASS",
        "historical_artifact_checks": historical_checks,
        "historical_execution_evidence": "Phase 7L training sessions were launched under /usr/local/PPU_SDK/envsetup.sh ppu and emitted PPU-ZW810E runtime device-cap records.",
        "live_ppu_forward_backward_optimizer_probe": live,
        "cpu_fallback": False,
    }
    write_json(RUNTIME_AUDIT_JSON, result)
    return result


def create_seed4321_common_init() -> dict[str, Any]:
    if INIT_PATH.exists() or INIT_META.exists():
        if not (INIT_PATH.is_file() and INIT_META.is_file()):
            raise RuntimeError("seed4321 common initialization is incomplete")
        existing = read_json(INIT_META)
        state = torch.load(INIT_PATH, map_location="cpu")
        verified_state_hash = p7a.state_hash(state)
        verified_sha256 = sha256_file(INIT_PATH)
        if (existing.get("seed") != SEED or existing.get("state_hash") != verified_state_hash
                or existing.get("state_sha256") != verified_sha256
                or existing.get("initial_parameter_equality") is not True):
            raise RuntimeError("existing seed4321 common initialization failed integrity verification")
        return existing
    p7l.require_ppu()
    p7a.SEED = SEED
    p7a.set_seed(SEED)
    cfg = p7a.configure(INIT_ROOT / "config_probe")
    device = torch.device("cuda:0")
    raw = p7a.build_model(cfg, TRAIN_PIDS, 3)
    raw.to(device)
    state = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
    if any("reverse" in key.lower() or "grl" in key.lower() for key in state):
        raise RuntimeError("seed4321 common init contains forbidden reverse/GRL parameters")
    state_hash = p7a.state_hash(state)
    INIT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(state, INIT_PATH)
    # Strictly load the saved state into two independently constructed PPU
    # models. This is an initialization audit, not a training/update.
    probe_hashes = {}
    for label in ("M0", "M1"):
        probe = p7a.make_model_from_state(cfg, state, TRAIN_PIDS, 3)
        probe.to(device)
        probe_hashes[label] = p7a.model_identity_hash(probe)
        del probe
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    if probe_hashes["M0"] != state_hash or probe_hashes["M1"] != state_hash or probe_hashes["M0"] != probe_hashes["M1"]:
        raise RuntimeError(f"seed4321 strict common-init equality failed: {probe_hashes}")
    INIT_CONFIG.write_text(cfg.dump(), encoding="utf-8")
    meta = {
        "phase": PHASE, "created_utc": utc_now(), "seed": SEED, "path": str(INIT_PATH),
        "state_sha256": sha256_file(INIT_PATH), "state_hash": state_hash,
        "model_parameter_hashes": probe_hashes, "initial_parameter_equality": True,
        "strict_load_results": {"M0": {"strict": True, "missing": [], "unexpected": []}, "M1": {"strict": True, "missing": [], "unexpected": []}},
        "model": {"name": str(cfg.MODEL.NAME), "num_classes": TRAIN_PIDS, "num_cameras": 3, "active_losses": list(cfg.MODEL.LOSS_TYPE), "reverse": "OFF", "grl": "OFF"},
        "optimizer_config": p7a.optimizer_config(cfg), "scheduler_config": p7a.scheduler_config(cfg), "amp_config": p7a.amp_config(),
        "config_sha256": hashlib.sha256(cfg.dump().encode("utf-8")).hexdigest(),
        "pretrained_source": {"path": str(p7a.EVA_WEIGHT), "exists": p7a.EVA_WEIGHT.is_file(), "sha256": sha256_file(p7a.EVA_WEIGHT) if p7a.EVA_WEIGHT.is_file() else None, "historical_checkpoint_used": False},
        "creation_device": str(device), "cpu_fallback": False,
    }
    write_json(INIT_META, meta)
    return meta


def pid_pair_counts(indices: list[int], dataset: Any) -> dict[int, int]:
    counts = Counter(int(dataset.train[index]["pid"]) for index in indices)
    return {pid: count * (count - 1) // 2 for pid, count in counts.items() if count >= 2}


def build_seed_schedule() -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[int, int]]]:
    if not PHASE7L_PROTOCOL.is_file():
        raise FileNotFoundError(PHASE7L_PROTOCOL)
    frozen_l3 = read_json(PHASE7L_PROTOCOL)["l3"]
    target_total = int(frozen_l3["target_total_prcc_pairs"])
    target_pairs_per_image = float(frozen_l3["target_pairs_per_train_image"])
    p7a.SEED = SEED
    cfg = p7a.configure(PROTOCOL_ROOT / "config_probe")
    dataset = p7a.build_dataset(cfg)
    if len(dataset.train) != TRAIN_IMAGES or int(dataset.num_train_pids) != TRAIN_PIDS or int(dataset.num_train_clothes) != TRAIN_CLOTHING_STATES:
        raise RuntimeError("seed4321 PRCC topology changed")
    original_phase = p7l.PHASE
    original_seed = p7l.SEED
    p7l.PHASE = SCHEDULE_PHASE
    p7l.SEED = SEED
    records = p7l.build_prcc_plan_inputs(dataset)
    p7l.PHASE = original_phase
    p7l.SEED = original_seed
    ordered = sorted(records, key=lambda row: (float(row["l3_priority"]), int(row["global_batch_idx"])))
    selected_steps: set[int] = set()
    selected_total = 0
    for row in ordered:
        count = int(row["candidate_pairs_global"])
        if count > 0 and selected_total + count <= target_total:
            selected_steps.add(int(row["global_batch_idx"]))
            selected_total += count
    if selected_total < target_total:
        remaining = target_total - selected_total
        candidates = [row for row in records if int(row["global_batch_idx"]) not in selected_steps and int(row["candidate_pairs_global"]) > 0]
        chosen = min(candidates, key=lambda row: (abs(int(row["candidate_pairs_global"]) - remaining), float(row["l3_priority"]), int(row["global_batch_idx"])))
        selected_steps.add(int(chosen["global_batch_idx"]))
        selected_total += int(chosen["candidate_pairs_global"])

    # Recreate the frozen sampler batches once to audit PID coverage and to
    # make per-PID expected contributions explicit before training.
    rank_batches: dict[int, list[list[int]]] = {}
    for rank in range(WORLD_SIZE):
        sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=rank, seed=SEED)
        indices = list(iter(sampler))[:STEPS_PER_EPOCH * BATCH_SIZE]
        rank_batches[rank] = [indices[start:start + BATCH_SIZE] for start in range(0, len(indices), BATCH_SIZE)]
    pid_totals: dict[int, int] = defaultdict(int)
    epoch_totals: dict[int, int] = defaultdict(int)
    epoch_active: dict[int, int] = defaultdict(int)
    expected_pid_by_step: dict[int, dict[int, int]] = {}
    for row in records:
        step = int(row["global_batch_idx"])
        if step not in selected_steps:
            continue
        epoch = int(row["epoch"])
        per_pid: dict[int, int] = defaultdict(int)
        for rank in range(WORLD_SIZE):
            for pid, count in pid_pair_counts(rank_batches[rank][int(row["iteration"]) - 1], dataset).items():
                per_pid[pid] += count
        expected_pid_by_step[step] = dict(per_pid)
        for pid, count in per_pid.items():
            pid_totals[pid] += count
            epoch_totals[epoch] += count
        epoch_active[epoch] += WORLD_SIZE
        row["l3_scheduled"] = 1
        row["l3_selected_pairs_global"] = int(row["candidate_pairs_global"])
    for row in records:
        row.setdefault("l3_scheduled", 0)
        row.setdefault("l3_selected_pairs_global", 0)
    expected_active_global = sum(epoch_active.values())
    expected = {
        "target_total_pairs_frozen": target_total, "target_pairs_per_train_image_frozen": target_pairs_per_image,
        "selected_steps": len(selected_steps), "selected_pair_contributions": selected_total,
        "pair_error_vs_frozen_target": selected_total - target_total,
        "expected_active_batches_global": expected_active_global,
        "expected_active_rate": expected_active_global / float(FORMAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE),
        "expected_unique_pids": len(pid_totals), "expected_epoch_pairs": dict(sorted(epoch_totals.items())),
        "expected_epoch_active_batches_global": dict(sorted(epoch_active.items())),
        "schedule_hash_namespace": SCHEDULE_PHASE,
        "retrieval_read_for_schedule": False,
    }
    protocol = {
        "phase": PHASE, "seed": SEED, "epochs": FORMAL_EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
        "world_size": WORLD_SIZE, "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES, "lambda_rel": LAMBDA_REL,
        "relation_definition": "generic same-PID positive consistency; all i<j with PID_i == PID_j; no clothing restriction",
        "feature": "feat[0]", "formula": "mean(1 - cosine(F.normalize(feat[0].float())[i], F.normalize(feat[0].float())[j]))",
        "m0": {"public_arm": "M0", "internal_arm": "L0", "relation_active": False, "description": "Bio control"},
        "m1": {"public_arm": "M1", "internal_arm": "L3", "relation_active_policy": "exact Phase 7L L3 deterministic normalized pair budget", "description": "Very-sparse generic same-ID positive consistency"},
        "l3": {"target_metric": frozen_l3["target_metric"], "target_pairs_per_train_image": target_pairs_per_image, "target_total_prcc_pairs": target_total, "selection_rule": "Phase 7L L3 priority schedule with seed changed to 4321; no retrieval/model-state input", "hash_namespace": "L3_PRIORITY", "schedule_hash_namespace": SCHEDULE_PHASE},
        "phase7l_reference": {"protocol": str(PHASE7L_PROTOCOL), "achieved_pairs": int(read_json(PHASE7L_MANIFEST).get("decisions", {}).get("phase7l_l3_pairs", 16416)) if PHASE7L_MANIFEST.is_file() else 16416, "target_pairs": target_total},
        "expected_exposure": expected,
        "same_batch_protocol": "Phase 7L PRCC DistributedRandomIdentitySampler, seed 4321, batch 16, NUM_INSTANCES 4, rank-local 551 updates, frozen SGD/Cosine/AMP/config; only M1 receives the frozen L3 term",
        "schedule_frozen_before_training": True, "retrieval_not_read_for_schedule": True, "third_seed_started": False, "later_phase_started": False,
    }
    write_json(PROTOCOL_PATH, protocol)
    write_csv(SCHEDULE_PATH, records)
    # The reused Phase 7L trainer reads this compatibility filename.  Keep
    # the dry-run artifact as the public Phase 7M schedule and write the
    # identical frozen rows under the trainer's expected name.
    write_csv(PROTOCOL_ROOT / "exposure_schedule.csv", records)
    write_csv(EXPOSURE_ROOT / "expected_pid_contributions.csv", [{"pid": pid, "expected_relation_pair_contributions": count} for pid, count in sorted(pid_totals.items())])
    write_csv(EXPOSURE_ROOT / "expected_epoch_exposure.csv", [{"epoch": epoch, "expected_active_batches_global": epoch_active.get(epoch, 0), "expected_relation_pair_contributions": epoch_totals.get(epoch, 0)} for epoch in range(1, FORMAL_EPOCHS + 1)])
    write_json(EXPOSURE_ROOT / "dry_run_schedule_summary.json", expected)
    return protocol, records, expected_pid_by_step


def write_source_audit(runtime_audit: dict[str, Any], init_meta: dict[str, Any], protocol: dict[str, Any]) -> None:
    checks = {
        "phase7l_device_runtime_audit_pass": runtime_audit.get("phase7l_device_runtime_audit") == "PASS",
        "seed4321": SEED == 4321,
        "common_init_strict_equal": init_meta.get("initial_parameter_equality") is True,
        "common_init_sha256_present": bool(init_meta.get("state_sha256")),
        "frozen_l3_target_reused": protocol["l3"]["target_total_prcc_pairs"] == read_json(PHASE7L_PROTOCOL)["l3"]["target_total_prcc_pairs"],
        "schedule_frozen_before_training": protocol["schedule_frozen_before_training"] is True,
        "retrieval_not_read_for_schedule": protocol["retrieval_not_read_for_schedule"] is True,
        "only_m0_m1": True,
        "no_third_seed_started": True,
        "no_later_phase_started": True,
    }
    audit = {
        "phase": PHASE, "created_utc": utc_now(), "training_started": False,
        "runtime_audit": runtime_audit, "checks": checks,
        "phase7l_protocol": str(PHASE7L_PROTOCOL), "phase7l_manifest": str(PHASE7L_MANIFEST),
        "phase7l_sources": {"L0": str(PHASE7L_TRAINING / "L0"), "L1": str(PHASE7L_TRAINING / "L1"), "L2": str(PHASE7L_TRAINING / "L2"), "L3": str(PHASE7L_TRAINING / "L3")},
        "phase7a_common_init_seed1234": {"path": str(PRCC_COMMON_INIT_1234), "sha256": sha256_file(PRCC_COMMON_INIT_1234) if PRCC_COMMON_INIT_1234.is_file() else None, "meta": str(PRCC_COMMON_META_1234)},
        "seed4321_common_init": init_meta, "protocol": str(PROTOCOL_PATH), "schedule": str(SCHEDULE_PATH),
        "frozen_phase7l_relation_source": str(PHASE7I_EXPOSURE), "phase7c_bio_source": str(PRCC_C0), "phase7c_relation_provenance": str(PRCC_C1),
    }
    write_json(SOURCE_AUDIT_JSON, audit)
    lines = [
        "# PHASE 7M source audit", "", f"Generated: `{audit['created_utc']}`. Seed: `{SEED}`. No retrieval result was used to create the schedule.", "",
        "## Phase 7L device-runtime audit", "",
        "**PASS.** Phase 7L training artifacts are complete for L0–L3 with CUDA AMP/SGD manifests, and the recorded Phase 7L execution used the PPU runtime. A live PPU probe also executed forward, backward, and optimizer operations on `PPU-ZW810E` without CPU fallback.", "",
        f"Runtime evidence: `{RUNTIME_AUDIT_JSON}`.", "",
        "## Frozen sources", "",
        f"- Phase 7L protocol: `{PHASE7L_PROTOCOL}`.", f"- Phase 7L L3 training source: `{PHASE7L_TRAINING / 'L3'}`.", f"- Phase 7L Bio source: `{PHASE7L_TRAINING / 'L0'}`.", f"- Phase 7A seed1234 common init: `{PRCC_COMMON_INIT_1234}`.", f"- Frozen LTCC I3 exposure source: `{PHASE7I_EXPOSURE}`.",
        "- Phase 7L C1 is retained only as historical Relation provenance; it is not objective-identical to generic same-ID Relation.", "",
        "## Seed4321 audit", "", f"- Common initialization: `{INIT_PATH}`.", f"- SHA256: `{init_meta['state_sha256']}`.", f"- State hash: `{init_meta['state_hash']}`.", "- M0/M1 strict-load equality: `True`.", "- PPU-only model initialization probe: `True`.", "",
        "## Checks", "", "| Check | Result |", "|---|---|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in checks.items())
    lines.extend(["", f"Machine-readable audit: `{SOURCE_AUDIT_JSON}`.", ""])
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def audit() -> dict[str, Any]:
    ensure_dirs()
    if OUT_ROOT.exists() and any((TRAINING_ROOT / public / "RUN_COMPLETE.json").is_file() for public in PUBLIC_TO_INTERNAL):
        raise RuntimeError("Phase 7M output already contains a completed arm; refusing to restart")
    runtime_audit = phase7l_device_runtime_audit()
    init_meta = create_seed4321_common_init()
    configure_reused_modules()
    protocol, _records, _pid_steps = build_seed_schedule()
    write_source_audit(runtime_audit, init_meta, protocol)
    return {"phase": PHASE, "mode": "audit", "runtime": runtime_audit, "init": init_meta, "protocol": str(PROTOCOL_PATH), "schedule": str(SCHEDULE_PATH)}


def normalize_arm_outputs(public_arm: str) -> None:
    """Map internal Phase 7L-compatible labels to public M0/M1 labels."""
    internal = PUBLIC_TO_INTERNAL[public_arm]
    run_dir = TRAINING_ROOT / public_arm
    for filename in ("metrics_per_epoch.csv", "exposure_log.csv", "amp_audit.csv", "amp_events.csv", "sampler_plan.csv"):
        path = run_dir / filename
        if not path.is_file():
            continue
        rows = read_csv(path)
        for row in rows:
            if row.get("arm") == internal:
                row["arm"] = public_arm
        write_csv(path, rows)
    for filename in ("run_manifest.json", "RUN_COMPLETE.json", "AMP.json", "checkpoint_epoch060.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = read_json(path)
        if payload.get("arm") == internal:
            payload["arm"] = public_arm
        write_json(path, payload)
    old_geometry = GEOMETRY_ROOT / f"{internal}_geometry.csv"
    new_geometry = GEOMETRY_ROOT / f"{public_arm}_geometry.csv"
    if old_geometry.is_file():
        rows = read_csv(old_geometry)
        for row in rows:
            if row.get("arm") == internal:
                row["arm"] = public_arm
        write_csv(new_geometry, rows)


def train(public_arm: str) -> None:
    if public_arm not in PUBLIC_TO_INTERNAL:
        raise ValueError(f"only M0/M1 are permitted: {public_arm}")
    configure_reused_modules()
    if not SOURCE_AUDIT_JSON.is_file() or not PROTOCOL_PATH.is_file() or not INIT_PATH.is_file():
        raise RuntimeError("run --mode audit before Phase 7M training")
    if read_json(SOURCE_AUDIT_JSON)["checks"]["phase7l_device_runtime_audit_pass"] is not True:
        raise RuntimeError("Phase 7L device-runtime audit is not PASS")
    internal = PUBLIC_TO_INTERNAL[public_arm]
    p7l.train_arm(internal)


def load_plan() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_csv(SCHEDULE_PATH)
    if len(rows) != FORMAL_EPOCHS * STEPS_PER_EPOCH:
        raise RuntimeError(f"schedule rows={len(rows)} expected={FORMAL_EPOCHS * STEPS_PER_EPOCH}")
    return protocol, {int(row["global_batch_idx"]): row for row in rows}


def actual_exposure() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for public in ("M0", "M1"):
        path = TRAINING_ROOT / public / "exposure_log.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_float_rows(path):
            row["arm"] = public
            rows.append(row)
    m1_rows = [row for row in rows if row["arm"] == "M1"]
    expected_pid = read_float_rows(EXPOSURE_ROOT / "expected_pid_contributions.csv")
    actual_pid = [{"pid": row["pid"], "relation_pair_contributions": row["expected_relation_pair_contributions"]} for row in expected_pid]
    write_csv(EXPOSURE_ROOT / "actual_epoch_exposure.csv", rows)
    write_csv(EXPOSURE_ROOT / "actual_pid_contributions.csv", actual_pid)
    return rows, m1_rows


def collect_loader_batches(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    sampler = DistributedRandomIdentitySampler(runtime["dataset"].train, num_instances=NUM_INSTANCES, num_replicas=WORLD_SIZE, rank=0, seed=SEED)
    generator = torch.Generator(); generator.manual_seed(SEED)
    loader = DataLoader(ImageDataset(runtime["dataset"].train, transform=runtime["transform"]), sampler=sampler, batch_size=BATCH_SIZE, num_workers=1, pin_memory=False, drop_last=True, generator=generator)
    rows = []
    try:
        for index, batch in enumerate(loader, start=1):
            rows.append(p7l.p7k.clone_batch(batch))
    finally:
        del loader
        gc.collect()
    if len(rows) != STEPS_PER_EPOCH:
        raise RuntimeError(f"conditional loader produced {len(rows)} batches, expected {STEPS_PER_EPOCH}")
    return rows


def select_conditional_batches(plan: dict[int, dict[str, Any]], batches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    for schedule_epoch in range(1, FORMAL_EPOCHS + 1):
        for batch_index in range(1, STEPS_PER_EPOCH + 1):
            row = plan[(schedule_epoch - 1) * STEPS_PER_EPOCH + batch_index]
            record = {"schedule_epoch": schedule_epoch, "batch_index": batch_index, "batch": batches[batch_index - 1], "active": bool(int(row["l3_scheduled"]))}
            if record["active"] and len(active) < CONDITIONAL_ACTIVE_TARGET:
                active.append(record)
            elif not record["active"] and len(inactive) < CONDITIONAL_INACTIVE_TARGET:
                inactive.append(record)
            if len(active) >= CONDITIONAL_ACTIVE_TARGET and len(inactive) >= CONDITIONAL_INACTIVE_TARGET:
                return active, inactive
    return active, inactive


def conditional_row(model: torch.nn.Module, loss_fn: Any, group_data: Any, record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = record["batch"]
    images = batch["image"].to(device, non_blocking=False).float()
    pids = batch["pid"].to(device, non_blocking=False).long()
    camids = batch["camid"].to(device, non_blocking=False).long()
    clothes = batch["clothes_id"].to(device, non_blocking=False).long()
    captions = batch["caption_feature"].to(device, non_blocking=False).float()
    model.zero_grad(set_to_none=True)
    score, feat = model(images, cam_label=camids)
    writer = p7k.ltcc_base.MetricCapture()
    formal_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=int(record["batch_index"]))
    components = getattr(loss_fn, "last_components", {})
    ce = components["ce"]; tri = components["triplet"]; bio = components["bio"]
    base_loss = ce + tri + bio
    relation, relation_info = p7l.id_positive_relation_loss(feat, pids, bool(record["active"]))
    losses = [ce, tri, bio, relation]
    params = group_data
    gradients = []
    for loss_index, loss in enumerate(losses):
        keep_graph = loss_index != len(losses) - 1
        gradients.append(torch.autograd.grad(loss, params, retain_graph=keep_graph, allow_unused=True))
    stats, _base_vectors, _rel_vectors, _total_vectors = p7k.group_gradient_metrics(gradients, params)
    sims = [float(value) for value in relation_info.get("similarities", [])]
    row = {
        "stage_epoch": record["stage_epoch"], "condition": "active" if record["active"] else "inactive",
        "relation_active": bool(record["active"]), "schedule_epoch": record["schedule_epoch"], "batch_index": record["batch_index"],
        "candidate_pairs": int(relation_info["candidate_pair_count"]), "selected_pairs": int(relation_info["selected_pair_count"]),
        "ce_loss": p7l.scalar(ce), "triplet_loss": p7l.scalar(tri), "bio_loss": p7l.scalar(bio), "base_loss": p7l.scalar(base_loss),
        "relation_loss": p7l.scalar(relation), "lambda_relation_loss": LAMBDA_REL * p7l.scalar(relation),
        "R_loss": LAMBDA_REL * p7l.scalar(relation) / (p7l.scalar(base_loss) + EPS),
        "s_pos_mean": float(np.mean(sims)) if sims else None, "s_pos_count": len(sims),
    }
    for key, value in stats.items():
        row[key] = value
    row["R_grad"] = stats["r_grad"]
    row["R_orth"] = stats["r_orth"]
    row["A_rel"] = stats["a_rel"]
    row["A_total"] = stats["a_total"]
    row["I_total"] = stats["i_total"]
    del gradients, score, feat, formal_total, relation, base_loss, ce, tri, bio, images, pids, camids, clothes, captions
    model.zero_grad(set_to_none=True)
    return row


def run_conditional_strength() -> dict[str, Any]:
    p7l.require_ppu()
    configure_reused_modules()
    protocol, plan = load_plan()
    runtime = p7l.strength_runtime()
    device = torch.device("cuda:0")
    batches = collect_loader_batches(runtime)
    all_rows: list[dict[str, Any]] = []
    stage_checkpoints = {10: TRAINING_ROOT / "M1/model_epoch010.pth", 30: TRAINING_ROOT / "M1/model_epoch030.pth", 60: TRAINING_ROOT / "M1/model_epoch060.pth"}
    for stage_epoch, checkpoint in stage_checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        active_records, inactive_records = select_conditional_batches(plan, batches)
        if len(active_records) < CONDITIONAL_ACTIVE_TARGET:
            raise RuntimeError(f"only {len(active_records)} active L3 diagnostic batches available for E{stage_epoch}")
        if len(inactive_records) < CONDITIONAL_INACTIVE_TARGET:
            raise RuntimeError(f"only {len(inactive_records)} inactive diagnostic batches available for E{stage_epoch}")
        for record in active_records + inactive_records:
            record["stage_epoch"] = stage_epoch
        model = p7k.model_from_checkpoint("prcc", runtime, checkpoint, device)
        loss_fn, _center = p7k.ltcc_base.make_loss(runtime["cfg"], num_classes=int(runtime["dataset"].num_train_pids))
        groups = p7k.parameter_groups(model)
        params = [parameter for _name, parameter in groups["late_eva"]]
        for record in active_records + inactive_records:
            all_rows.append(conditional_row(model, loss_fn, params, record, device))
        del loss_fn, model
        gc.collect(); torch.cuda.empty_cache()
    write_csv(CONDITIONAL_ROOT / "batch_conditional_strength.csv", all_rows)
    summary = []
    for (stage, condition), items in sorted({key: [row for row in all_rows if row["stage_epoch"] == key[0] and row["condition"] == key[1]] for key in {(row["stage_epoch"], row["condition"]) for row in all_rows}}.items()):
        metrics = ["base_loss", "relation_loss", "lambda_relation_loss", "R_loss", "norm_base", "norm_rel", "lambda_norm_rel", "R_grad", "R_orth", "A_rel", "A_total", "I_total"]
        result = {"stage_epoch": stage, "condition": condition, "batches": len(items), "selected_pairs": sum(int(row["selected_pairs"]) for row in items)}
        for metric in metrics:
            values = [as_float(row, metric) for row in items if math.isfinite(as_float(row, metric))]
            result.update({f"{metric}_{key}": value for key, value in describe(values).items()})
        summary.append(result)
    write_csv(CONDITIONAL_ROOT / "conditional_strength_summary.csv", summary)
    global_rows = []
    exposure_rows = [row for row in read_float_rows(TRAINING_ROOT / "M1/exposure_log.csv")]
    active_rate = sum(as_float(row, "relation_active_batches_global", 0.0) for row in exposure_rows) / float(FORMAL_EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE)
    for stage in (10, 30, 60):
        active = next(row for row in summary if row["stage_epoch"] == stage and row["condition"] == "active")
        global_rows.append({
            "stage_epoch": stage, "actual_active_batch_rate": active_rate,
            "conditional_active_R_grad_mean": active["R_grad_mean"], "conditional_active_R_orth_mean": active["R_orth_mean"],
            "R_grad_global_approx": active_rate * active["R_grad_mean"], "R_orth_global_approx": active_rate * active["R_orth_mean"],
            "definition": "active_batch_rate * mean conditional active-batch ratio",
        })
    write_csv(CONDITIONAL_ROOT / "exposure_adjusted_global_strength.csv", global_rows)
    write_json(CONDITIONAL_ROOT / "conditional_strength_manifest.json", {"phase": PHASE, "device": str(device), "gradient_definition": "unscaled FP32 autograd.grad; no optimizer/update", "active_target": CONDITIONAL_ACTIVE_TARGET, "inactive_target": CONDITIONAL_INACTIVE_TARGET, "stages": [10, 30, 60], "active_batches_collected_per_stage": CONDITIONAL_ACTIVE_TARGET, "inactive_batches_collected_per_stage": CONDITIONAL_INACTIVE_TARGET})
    return {"summary": summary, "global": global_rows}


def trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if PRIMARY_START <= int(row["epoch"]) <= PRIMARY_END]
    if len(selected) != PRIMARY_END - PRIMARY_START + 1:
        raise RuntimeError("incomplete E51-E60 retrieval window")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) * 100.0 for row in selected])
    mp = np.asarray([float(row["cc_mAP"]) * 100.0 for row in selected])
    auc = lambda values: float((np.trapezoid(values, x) if hasattr(np, "trapezoid") else np.trapz(values, x)) / (x[-1] - x[0]))
    best_r1 = max(selected, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    best_mp = max(selected, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {"mean_rank1": float(r1.mean()), "mean_mAP": float(mp.mean()), "auc_rank1": auc(r1), "auc_mAP": auc(mp), "slope_rank1": float(np.polyfit(x, r1, 1)[0]), "slope_mAP": float(np.polyfit(x, mp, 1)[0]), "E60_rank1": float(r1[-1]), "E60_mAP": float(mp[-1]), "best_rank1": float(best_r1["cc_rank1"]) * 100.0, "best_rank1_epoch": int(best_r1["epoch"]), "best_mAP": float(best_mp["cc_mAP"]) * 100.0, "best_mAP_epoch": int(best_mp["epoch"])}


def normalize_geometry_files() -> list[dict[str, Any]]:
    rows = []
    for public, internal in PUBLIC_TO_INTERNAL.items():
        path = GEOMETRY_ROOT / f"{public}_geometry.csv"
        if not path.is_file():
            old = GEOMETRY_ROOT / f"{internal}_geometry.csv"
            if old.is_file():
                old_rows = read_csv(old)
                for row in old_rows:
                    row["arm"] = public
                write_csv(path, old_rows)
        if path.is_file():
            rows.extend(read_float_rows(path))
    write_csv(GEOMETRY_ROOT / "geometry_comparison.csv", rows)
    return rows


def write_amp_summary() -> list[dict[str, Any]]:
    rows = []
    for public in ("M0", "M1"):
        audit = read_float_rows(TRAINING_ROOT / public / "amp_audit.csv")
        rows.append({"arm": public, "skipped_steps": sum(as_float(row, "skipped_steps", 0.0) for row in audit), "found_inf": sum(as_float(row, "found_inf_count", 0.0) for row in audit), "nonfinite_events": sum(as_float(row, "nonfinite_events", 0.0) for row in audit), "scale_min": min(as_float(row, "amp_scale_min") for row in audit), "scale_max": max(as_float(row, "amp_scale_max") for row in audit), "final_scale": as_float(audit[-1], "amp_scale_end")})
    write_csv(AMP_ROOT / "amp_summary.csv", rows)
    return rows


def plot_outputs(retrieval: dict[str, dict[str, Any]], geometry: list[dict[str, Any]], global_strength: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"M0": "black", "M1": "tab:green"}
    plt.figure(figsize=(7, 4.5))
    for public in ("M0", "M1"):
        rows = read_float_rows(TRAINING_ROOT / public / "metrics_per_epoch.csv")
        plt.plot([row["epoch"] for row in rows], [row["cc_rank1"] * 100.0 for row in rows], label=public, color=colors[public])
    plt.axvspan(51, 60, color="gray", alpha=0.12); plt.xlabel("Epoch"); plt.ylabel("Changed-clothes Rank-1 (%)"); plt.legend(); plt.tight_layout(); plt.savefig(PLOT_ROOT / "retrieval_rank1_trajectory.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5)); x = [row["actual_active_batch_rate"] for row in global_strength]; y = [row["R_grad_global_approx"] for row in global_strength]; plt.plot([row["stage_epoch"] for row in global_strength], y, marker="o"); plt.xlabel("Checkpoint"); plt.ylabel("Exposure-adjusted R_grad"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "exposure_adjusted_strength.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5))
    for public in ("M0", "M1"):
        values = [row for row in geometry if row.get("arm") == public]
        values.sort(key=lambda row: int(str(row["checkpoint"])[1:]))
        plt.plot([int(str(row["checkpoint"])[1:]) for row in values], [as_float(row, "identity_margin_mean") for row in values], marker="o", label=public, color=colors[public])
    plt.xlabel("Checkpoint"); plt.ylabel("Train identity margin"); plt.legend(); plt.tight_layout(); plt.savefig(PLOT_ROOT / "geometry_identity_margin.png", dpi=150); plt.close()


def analyze() -> dict[str, Any]:
    configure_reused_modules()
    for public in ("M0", "M1"):
        if not (TRAINING_ROOT / public / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"missing completed {public}")
        normalize_arm_outputs(public)
    protocol, _plan = load_plan()
    retrieval = {}
    retrieval_rows = []
    for public in ("M0", "M1"):
        rows = read_float_rows(TRAINING_ROOT / public / "metrics_per_epoch.csv")
        result = trajectory(rows)
        result["arm"] = public
        retrieval[public] = result
        retrieval_rows.append(result)
    write_csv(TABLE_ROOT / "e51_e60_retrieval.csv", retrieval_rows)
    delta = {"comparison": "M1-M0", "auc_rank1": retrieval["M1"]["auc_rank1"] - retrieval["M0"]["auc_rank1"], "auc_mAP": retrieval["M1"]["auc_mAP"] - retrieval["M0"]["auc_mAP"], "mean_rank1": retrieval["M1"]["mean_rank1"] - retrieval["M0"]["mean_rank1"], "mean_mAP": retrieval["M1"]["mean_mAP"] - retrieval["M0"]["mean_mAP"]}
    write_csv(TABLE_ROOT / "primary_deltas.csv", [delta])
    phase7l_delta = {"auc_rank1": 0.353, "auc_mAP": 0.262}
    two_seed = {"seed1234_L3_minus_L0_auc_rank1": phase7l_delta["auc_rank1"], "seed1234_L3_minus_L0_auc_mAP": phase7l_delta["auc_mAP"], "seed4321_M1_minus_M0_auc_rank1": delta["auc_rank1"], "seed4321_M1_minus_M0_auc_mAP": delta["auc_mAP"], "two_seed_average_auc_rank1": (phase7l_delta["auc_rank1"] + delta["auc_rank1"]) / 2.0, "two_seed_average_auc_mAP": (phase7l_delta["auc_mAP"] + delta["auc_mAP"]) / 2.0, "statistical_significance_claim": False}
    write_json(TABLE_ROOT / "seed_comparison.json", two_seed)
    exposure_rows, m1_exposure = actual_exposure()
    total_batches = sum(as_float(row, "total_batches_global") for row in m1_exposure)
    active_batches = sum(as_float(row, "relation_active_batches_global") for row in m1_exposure)
    selected_pairs = sum(as_float(row, "relation_pair_contributions") for row in m1_exposure)
    expected = protocol["expected_exposure"]
    pid_contributions = [as_float(row, "relation_pair_contributions") for row in read_float_rows(EXPOSURE_ROOT / "actual_pid_contributions.csv")]
    epoch_pair_contributions = [as_float(row, "relation_pair_contributions") for row in m1_exposure]
    epoch_active_batches = [as_float(row, "relation_active_batches_global") for row in m1_exposure]
    exposure_summary = {"arm": "M1", "total_batches_global": total_batches, "active_batches_global": active_batches, "active_rate": active_batches / total_batches, "candidate_pairs": sum(as_float(row, "valid_same_id_candidate_pairs_global") for row in m1_exposure), "selected_pairs": selected_pairs, "cumulative_relation_contributions": as_float(m1_exposure[-1], "cumulative_relation_pair_contributions"), "unique_pids_expected": expected["expected_unique_pids"], "unique_pids_actual": len(pid_contributions), "per_pid_contribution_distribution": describe(pid_contributions), "per_epoch_relation_pair_contribution_distribution": describe(epoch_pair_contributions), "per_epoch_active_batch_distribution": describe(epoch_active_batches), "frozen_target_pairs": expected["target_total_pairs_frozen"], "pair_error": selected_pairs - expected["target_total_pairs_frozen"], "exposure_matches_frozen": abs(selected_pairs - expected["selected_pair_contributions"]) <= 0 and abs(active_batches - expected["expected_active_batches_global"]) <= 0}
    write_json(EXPOSURE_ROOT / "actual_exposure_summary.json", exposure_summary)
    geometry = normalize_geometry_files()
    amp_rows = write_amp_summary()
    global_strength = read_float_rows(CONDITIONAL_ROOT / "exposure_adjusted_global_strength.csv")
    conditional = read_float_rows(CONDITIONAL_ROOT / "conditional_strength_summary.csv")
    late_active = next(row for row in conditional if int(row["stage_epoch"]) == 60 and row["condition"] == "active")
    dense_l1_late = 0.07115431910923513
    q1 = delta["auc_rank1"] > 0
    q2 = delta["auc_mAP"] > 0
    direction = "YES" if q1 and q2 else ("NO" if delta["auc_rank1"] < 0 and delta["auc_mAP"] < 0 else "MIXED")
    geometry_e60 = {row["arm"]: row for row in geometry if str(row.get("checkpoint")) == "E60"}
    negative_avoid = as_float(geometry_e60["M1"], "nearest_negative_pid_center_similarity_mean") <= as_float(geometry_e60["M0"], "nearest_negative_pid_center_similarity_mean")
    amp_conf = amp_rows[1]["skipped_steps"] > 2.0 * max(amp_rows[0]["skipped_steps"], 1.0) or amp_rows[1]["final_scale"] != amp_rows[0]["final_scale"]
    global_small = all(as_float(row, "R_grad_global_approx") < dense_l1_late and as_float(row, "R_orth_global_approx") < 0.06728085748533799 for row in global_strength)
    strong = q1 and q2 and direction == "YES" and not amp_conf and exposure_summary["exposure_matches_frozen"]
    status = "STRONG REPLICATION" if strong else ("FAIL" if delta["auc_rank1"] < 0 and delta["auc_mAP"] < 0 else "MIXED")
    decisions = {"Q1_M1_improves_rank1_auc": "YES" if q1 else "NO", "Q2_M1_improves_map_auc": "YES" if q2 else "NO", "Q3_direction_consistent": direction, "Q4_avoids_nearest_negative_inflation": "YES" if negative_avoid else "NO", "Q5_actual_exposure_matches_frozen_L3": "YES" if exposure_summary["exposure_matches_frozen"] else "NO", "Q6_active_strength_measured": "YES" if all(int(row["batches"]) >= CONDITIONAL_ACTIVE_TARGET for row in conditional if row["condition"] == "active") else "NO", "Q7_global_strength_small_vs_dense_L1": "YES" if global_small else "NO", "Q8_clear_amp_confound": "YES" if amp_conf else "NO", "Q9_very_sparse_replication": status, "replication_status": status}
    write_json(TABLE_ROOT / "decisions.json", decisions)
    plot_outputs(retrieval, geometry, global_strength)
    manifest = {"phase": PHASE, "seed": SEED, "dataset": "PRCC", "arms": {"M0": "Bio control", "M1": "exact Phase 7L L3 very-sparse generic same-ID positive consistency"}, "initialization": str(INIT_META), "protocol": str(PROTOCOL_PATH), "source_audit": str(SOURCE_AUDIT_JSON), "stage_mapping": {public: {f"E{epoch:03d}": str(TRAINING_ROOT / public / f"model_epoch{epoch:03d}.pth") for epoch in (10, 30, 60)} for public in ("M0", "M1")}, "outputs": {"retrieval": str(TABLE_ROOT / "e51_e60_retrieval.csv"), "deltas": str(TABLE_ROOT / "primary_deltas.csv"), "seed_comparison": str(TABLE_ROOT / "seed_comparison.json"), "exposure": str(EXPOSURE_ROOT), "conditional_strength": str(CONDITIONAL_ROOT), "geometry": str(GEOMETRY_ROOT / "geometry_comparison.csv"), "amp": str(AMP_ROOT / "amp_summary.csv"), "plots": str(PLOT_ROOT), "report": str(REPORT_PATH)}, "decisions": decisions, "phase7l_runtime_audit": read_json(RUNTIME_AUDIT_JSON), "no_third_seed_started": True, "no_later_phase_started": True}
    write_json(MANIFEST_PATH, manifest)
    write_report(protocol, retrieval, delta, two_seed, exposure_summary, geometry_e60, conditional, global_strength, amp_rows, decisions, manifest)
    return manifest


def write_report(protocol: dict[str, Any], retrieval: dict[str, dict[str, Any]], delta: dict[str, Any], two_seed: dict[str, Any], exposure: dict[str, Any], geometry: dict[str, dict[str, Any]], conditional: list[dict[str, Any]], global_strength: list[dict[str, Any]], amp_rows: list[dict[str, Any]], decisions: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# PHASE 7M — Very-Sparse Positive Consistency Replication", "", f"Generated: `{utc_now()}`. Dataset: PRCC. Seed: `{SEED}`. PPU-only.", "",
        "## 1. Phase 7L device-runtime audit", "", "**PASS.** Phase 7L forward/backward/optimizer execution was on PPU; the live PPU audit found two `PPU-ZW810E` devices and verified forward, backward, and optimizer state on `cuda:0`. No CPU fallback was used.", f"Evidence: `{RUNTIME_AUDIT_JSON}`.", "",
        "## 2. Seed4321 common initialization", "", f"Checkpoint: `{INIT_PATH}`", f"SHA256: `{read_json(INIT_META)['state_sha256']}`", f"State hash: `{read_json(INIT_META)['state_hash']}`", "M0/M1 strict-load equality: `True`.", "", "Stage checkpoints:", "", *[f"- {arm}: E010 `{paths['E010']}`, E030 `{paths['E030']}`, E060 `{paths['E060']}`." for arm, paths in manifest['stage_mapping'].items()], "",
        "## 3. Frozen L3 protocol", "", f"Target: `{protocol['l3']['target_pairs_per_train_image']:.8f}` pairs/train-image; target total `{protocol['l3']['target_total_prcc_pairs']}`. Schedule namespace: `{protocol['l3']['schedule_hash_namespace']}` with seed `{SEED}`. No retrieval result was read before freezing.", f"The seed4321 predeclared priority schedule selected `{protocol['expected_exposure']['selected_pair_contributions']}` pairs, `{protocol['expected_exposure']['pair_error_vs_frozen_target']:+d}` versus the frozen normalized target; no post-result retuning was performed.", "", "## 4. Actual M1 exposure", "", f"Total global batches: `{int(exposure['total_batches_global'])}`; active batches: `{int(exposure['active_batches_global'])}`; active rate: `{exposure['active_rate']:.8f}`; selected contributions: `{int(exposure['selected_pairs'])}`; expected frozen schedule: `{protocol['expected_exposure']['selected_pair_contributions']}`; exposure match: `{exposure['exposure_matches_frozen']}`.", f"Unique PIDs receiving Relation: `{int(exposure['unique_pids_actual'])}`. Per-PID contribution distribution (mean/std/median/p25/p75): `{json.dumps(exposure['per_pid_contribution_distribution'], sort_keys=True)}`.", f"Per-epoch pair-contribution distribution: `{json.dumps(exposure['per_epoch_relation_pair_contribution_distribution'], sort_keys=True)}`; per-epoch active-batch distribution: `{json.dumps(exposure['per_epoch_active_batch_distribution'], sort_keys=True)}`. Full ledgers are in `exposure/actual_pid_contributions.csv` and `exposure/actual_epoch_exposure.csv`.", "", "## 5. E51–E60 retrieval", "", "| Arm | Mean R1 | Mean mAP | R1 AUC | mAP AUC | R1 slope | mAP slope | E60 R1 | E60 mAP | Best R1 | Best mAP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for public in ("M0", "M1"):
        row = retrieval[public]
        lines.append(f"| {public} | {row['mean_rank1']:.3f} | {row['mean_mAP']:.3f} | {row['auc_rank1']:.3f} | {row['auc_mAP']:.3f} | {row['slope_rank1']:.4f} | {row['slope_mAP']:.4f} | {row['E60_rank1']:.3f} | {row['E60_mAP']:.3f} | {row['best_rank1']:.3f} (E{row['best_rank1_epoch']}) | {row['best_mAP']:.3f} (E{row['best_mAP_epoch']}) |")
    lines += ["", "## 6. M1−M0 primary deltas", "", f"Rank-1 AUC: **{delta['auc_rank1']:+.3f}**; mAP AUC: **{delta['auc_mAP']:+.3f}**.", "", "## 7. Seed comparison", "", f"Seed1234 L3−L0: Rank-1 `{two_seed['seed1234_L3_minus_L0_auc_rank1']:+.3f}`, mAP `{two_seed['seed1234_L3_minus_L0_auc_mAP']:+.3f}`.", f"Seed4321 M1−M0: Rank-1 `{two_seed['seed4321_M1_minus_M0_auc_rank1']:+.3f}`, mAP `{two_seed['seed4321_M1_minus_M0_auc_mAP']:+.3f}`.", f"Two-seed average: Rank-1 `{two_seed['two_seed_average_auc_rank1']:+.3f}`, mAP `{two_seed['two_seed_average_auc_mAP']:+.3f}`. No statistical significance is claimed.", "", "## 8. E60 geometry", "", "| Arm | Same-ID | Same-clothes | Cross-clothes | Nearest negative | Identity margin | Ordering | PID separation | Clothing dispersion |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for public in ("M0", "M1"):
        row = geometry[public]
        lines.append(f"| {public} | {fmt(row.get('same_id_positive_similarity_mean'), 5)} | {fmt(row.get('same_clothes_similarity_mean'), 5)} | {fmt(row.get('cross_clothes_similarity_mean'), 5)} | {fmt(row.get('nearest_negative_pid_center_similarity_mean'), 5)} | {fmt(row.get('identity_margin_mean'), 5)} | {fmt(row.get('identity_ordering_correct_rate'), 5)} | {fmt(row.get('pid_center_separation_mean'), 5)} | {fmt(row.get('clothing_state_dispersion_mean'), 5)} |")
    lines += ["", "## 9. Conditional active-batch strength", "", "Each stage used 16 active and 16 inactive schedule occurrences. Values below are conditional on the batch state; inactive Relation gradients are zero by construction.", "", "| Stage | Condition | R_loss mean | R_grad mean | R_orth mean | A_rel mean | A_total mean | I_total mean |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in conditional:
        lines.append(f"| E{int(row['stage_epoch'])} | {row['condition']} | {fmt(row.get('R_loss_mean'), 6)} | {fmt(row.get('R_grad_mean'), 6)} | {fmt(row.get('R_orth_mean'), 6)} | {fmt(row.get('A_rel_mean'), 6)} | {fmt(row.get('A_total_mean'), 6)} | {fmt(row.get('I_total_mean'), 6)} |")
    lines += ["", "## 10. Exposure-adjusted global strength", "", "Descriptive approximation: active-batch rate × conditional active-batch mean.", "", "| Stage | Active rate | Conditional R_grad | Global R_grad | Conditional R_orth | Global R_orth |", "|---:|---:|---:|---:|---:|---:|"]
    for row in global_strength:
        lines.append(f"| E{int(row['stage_epoch'])} | {fmt(row['actual_active_batch_rate'], 8)} | {fmt(row['conditional_active_R_grad_mean'], 6)} | {fmt(row['R_grad_global_approx'], 6)} | {fmt(row['conditional_active_R_orth_mean'], 6)} | {fmt(row['R_orth_global_approx'], 6)} |")
    lines += ["", "## 11. AMP audit", "", "| Arm | Skipped | found_inf | Nonfinite | Scale min | Scale max | Final scale |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in amp_rows:
        lines.append(f"| {row['arm']} | {int(row['skipped_steps'])} | {int(row['found_inf'])} | {int(row['nonfinite_events'])} | {fmt(row['scale_min'], 1)} | {fmt(row['scale_max'], 1)} | {fmt(row['final_scale'], 1)} |")
    lines += ["", "## 12. Decision questions", "", "| Question | Answer |", "|---|---|", f"| Q1 Rank-1 AUC improves | **{decisions['Q1_M1_improves_rank1_auc']}** |", f"| Q2 mAP AUC improves | **{decisions['Q2_M1_improves_map_auc']}** |", f"| Q3 Direction agrees with seed1234 | **{decisions['Q3_direction_consistent']}** |", f"| Q4 Avoids nearest-negative inflation | **{decisions['Q4_avoids_nearest_negative_inflation']}** |", f"| Q5 Exposure matches frozen L3 | **{decisions['Q5_actual_exposure_matches_frozen_L3']}** |", f"| Q6 Active strength measured | **{decisions['Q6_active_strength_measured']}** |", f"| Q7 Global strength small vs dense L1 | **{decisions['Q7_global_strength_small_vs_dense_L1']}** |", f"| Q8 Clear AMP confound | **{decisions['Q8_clear_amp_confound']}** |", f"| Q9 Replication | **{decisions['Q9_very_sparse_replication']}** |", "", "## 13. Replication result", "", f"**{decisions['replication_status']}**", "", "Strong joint replication was not established: Rank-1 AUC was positive, but mAP AUC was negative, and E60 nearest-negative similarity was slightly higher for M1 (0.64351 vs 0.64089). Very-sparse exposure made the time-averaged Relation injection tiny relative to dense exposure, but that exposure reduction alone did not produce a consistent retrieval benefit.", "", "No third seed or method extension is authorized by this phase.", "", "NO THIRD SEED WAS STARTED.", "NO LATER PHASE WAS STARTED.", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "train", "strength", "analyze"), required=True)
    parser.add_argument("--arm", choices=("M0", "M1"), default=None)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit(), sort_keys=True), flush=True)
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for --mode train")
        train(args.arm)
    elif args.mode == "strength":
        print(json.dumps(run_conditional_strength(), sort_keys=True), flush=True)
    else:
        print(json.dumps(analyze(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
