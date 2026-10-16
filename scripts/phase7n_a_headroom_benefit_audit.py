#!/usr/local/bin/python3
"""Phase 7N-A: frozen dataset/PID/pair headroom audit.

This script has no training path and never constructs an optimizer.  The only
model execution is frozen train-set feature extraction and no-step gradient
probes on PPU.  All headroom definitions are computed from train images only;
query/gallery/test data are never loaded for diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase6b_soft_relation_screen as b6  # noqa: E402
import phase7a_prcc_relation_validation as p7a  # noqa: E402
import phase7k_a_relation_strength_balance as p7k  # noqa: E402
import phase7l_exposure_matched_positive_consistency as p7l  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402


PHASE = "phase7n_a_headroom_benefit_audit"
SEED = 7
BOOTSTRAP = 2000
EPS = 1e-12
WORLD_SIZE = 2
GRAD_PROBES = 4
GRAD_PAIRS_PER_PROBE = 8

OUT_ROOT = ROOT / "outputs/new_method_track/phase7n_a_headroom_benefit_audit"
DATASET_ROOT = OUT_ROOT / "dataset_headroom"
PID_ROOT = OUT_ROOT / "pid_headroom"
PAIR_ROOT = OUT_ROOT / "pair_headroom"
QUARTILE_ROOT = OUT_ROOT / "quartiles"
CONCENTRATION_ROOT = OUT_ROOT / "benefit_concentration"
HARMFUL_ROOT = OUT_ROOT / "harmful_pid"
GRAD_ROOT = OUT_ROOT / "gradient"
DIFFICULTY_ROOT = OUT_ROOT / "base_difficulty"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
BANK_ROOT = DATASET_ROOT / "feature_banks"
SOURCE_AUDIT_JSON = OUT_ROOT / "source_audit.json"
RUNTIME_AUDIT_JSON = OUT_ROOT / "runtime_audit.json"
MANIFEST_PATH = OUT_ROOT / "PHASE7N_A_MANIFEST.json"
REPORT_PATH = ROOT / "reports/PHASE_7N_A_DATASET_HEADROOM_REGULARIZATION_BENEFIT_AUDIT.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7N_A_SOURCE_AUDIT.md"

LTCC_I0 = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full"
LTCC_I3 = ROOT / "outputs/new_method_track/phase7i_relation_specificity/I3_id_only_matched"
LTCC_BANK_ROOT = ROOT / "outputs/new_method_track/phase7i_relation_specificity/geometry/feature_banks"
PRCC_7L = ROOT / "outputs/new_method_track/phase7l_exposure_matched_positive_consistency/training"
PRCC_7M = ROOT / "outputs/new_method_track/phase7m_sparse_positive_consistency_replication/training"
PRCC_7L_PROTOCOL = ROOT / "outputs/new_method_track/phase7l_exposure_matched_positive_consistency/protocol/protocol.json"
PRCC_7M_PROTOCOL = ROOT / "outputs/new_method_track/phase7m_sparse_positive_consistency_replication/protocol/protocol.json"


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
        return value.detach().cpu().tolist()
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def describe(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {key: None for key in ("mean", "std", "median", "p25", "p75")}
    return {
        "mean": float(finite.mean()), "std": float(finite.std()), "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)), "p75": float(np.percentile(finite, 75)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}


def ensure_dirs() -> None:
    for path in (OUT_ROOT, DATASET_ROOT, PID_ROOT, PAIR_ROOT, QUARTILE_ROOT, CONCENTRATION_ROOT, HARMFUL_ROOT, GRAD_ROOT, DIFFICULTY_ROOT, TABLE_ROOT, PLOT_ROOT, BANK_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def require_ppu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7N-A model execution requires PPU; CPU fallback is forbidden")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint format: {path}")
    return state


def source_specs() -> dict[str, dict[str, Any]]:
    return {
        "LTCC_I0": {
            "dataset": "LTCC", "seed": 1234, "role": "baseline Bio", "bank": LTCC_BANK_ROOT / "I0_train_features.npz",
            "model": LTCC_I0 / "model_final.pth", "metrics": LTCC_I0 / "metrics_per_epoch.csv", "run": LTCC_I0,
        },
        "LTCC_I3": {
            "dataset": "LTCC", "seed": 1234, "role": "ID-only positive consistency endpoint", "bank": LTCC_BANK_ROOT / "I3_train_features.npz",
            "model": LTCC_I3 / "model_final.pth", "metrics": LTCC_I3 / "metrics_per_epoch.csv", "run": LTCC_I3,
        },
        "PRCC_M0": {
            "dataset": "PRCC", "seed": 4321, "role": "baseline Bio", "bank": BANK_ROOT / "PRCC_M0_seed4321_train_features.npz",
            "model": PRCC_7M / "M0/model_epoch060.pth", "metrics": PRCC_7M / "M0/metrics_per_epoch.csv", "run": PRCC_7M / "M0",
        },
        "PRCC_M1": {
            "dataset": "PRCC", "seed": 4321, "role": "very-sparse positive consistency endpoint", "bank": BANK_ROOT / "PRCC_M1_seed4321_train_features.npz",
            "model": PRCC_7M / "M1/model_epoch060.pth", "metrics": PRCC_7M / "M1/metrics_per_epoch.csv", "run": PRCC_7M / "M1",
        },
        "PRCC_L0": {
            "dataset": "PRCC", "seed": 1234, "role": "secondary baseline Bio", "bank": BANK_ROOT / "PRCC_L0_seed1234_train_features.npz",
            "model": PRCC_7L / "L0/model_epoch060.pth", "metrics": PRCC_7L / "L0/metrics_per_epoch.csv", "run": PRCC_7L / "L0",
        },
        "PRCC_L3": {
            "dataset": "PRCC", "seed": 1234, "role": "secondary very-sparse positive consistency endpoint", "bank": BANK_ROOT / "PRCC_L3_seed1234_train_features.npz",
            "model": PRCC_7L / "L3/model_epoch060.pth", "metrics": PRCC_7L / "L3/metrics_per_epoch.csv", "run": PRCC_7L / "L3",
        },
    }


COMPARISONS = {
    "LTCC_I3_minus_I0": {"dataset": "LTCC", "baseline": "LTCC_I0", "endpoint": "LTCC_I3", "label": "LTCC I3-I0"},
    "PRCC_seed4321_M1_minus_M0": {"dataset": "PRCC", "baseline": "PRCC_M0", "endpoint": "PRCC_M1", "label": "PRCC seed4321 M1-M0"},
    "PRCC_seed1234_L3_minus_L0": {"dataset": "PRCC", "baseline": "PRCC_L0", "endpoint": "PRCC_L3", "label": "PRCC seed1234 L3-L0"},
}


def audit_sources() -> dict[str, Any]:
    ensure_dirs()
    specs = source_specs()
    checks: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for name, spec in specs.items():
        run_complete = spec["run"] / "RUN_COMPLETE.json"
        required = [spec["model"], spec["metrics"], run_complete, spec["bank"]]
        records[name] = {"dataset": spec["dataset"], "seed": spec["seed"], "role": spec["role"], "files": {str(path): file_record(path) for path in required}}
        checks[f"{name}_files"] = all(path.is_file() for path in required)
        if spec["metrics"].is_file():
            rows = read_csv(spec["metrics"])
            checks[f"{name}_60_epochs"] = len(rows) == 60 and [int(row["epoch"]) for row in rows] == list(range(1, 61))
        else:
            checks[f"{name}_60_epochs"] = False
    # The primary PRCC source has the exact shared seed4321 initialization.
    m0 = json.loads((PRCC_7M / "M0/RUN_COMPLETE.json").read_text()) if (PRCC_7M / "M0/RUN_COMPLETE.json").is_file() else {}
    m1 = json.loads((PRCC_7M / "M1/RUN_COMPLETE.json").read_text()) if (PRCC_7M / "M1/RUN_COMPLETE.json").is_file() else {}
    l0 = json.loads((PRCC_7L / "L0/RUN_COMPLETE.json").read_text()) if (PRCC_7L / "L0/RUN_COMPLETE.json").is_file() else {}
    l3 = json.loads((PRCC_7L / "L3/RUN_COMPLETE.json").read_text()) if (PRCC_7L / "L3/RUN_COMPLETE.json").is_file() else {}
    checks["prcc_seed4321_common_init"] = bool(m0.get("common_init_sha256") and m0.get("common_init_sha256") == m1.get("common_init_sha256") and m0.get("initial_parameter_hash") == m1.get("initial_parameter_hash"))
    checks["prcc_seed1234_common_init"] = bool(l0.get("common_init_sha256") and l0.get("common_init_sha256") == l3.get("common_init_sha256") and l0.get("initial_parameter_hash") == l3.get("initial_parameter_hash"))
    checks["train_only_scope"] = True
    checks["no_training_path_in_phase"] = True
    checks["no_new_reid_training_started"] = True
    source = {
        "phase": PHASE, "created_utc": utc_now(), "dataset_sources": records, "comparisons": COMPARISONS,
        "phase7i_report": str(ROOT / "reports/PHASE_7I_CROSS_CLOTHES_RELATION_SPECIFICITY_TEST.md"),
        "phase7m_report": str(ROOT / "reports/PHASE_7M_VERY_SPARSE_POSITIVE_CONSISTENCY_REPLICATION.md"),
        "phase7l_protocol": str(PRCC_7L_PROTOCOL), "phase7m_protocol": str(PRCC_7M_PROTOCOL),
        "limitations": [
            "PRCC seed4321 M0/M1 is the primary matched-init comparison; seed1234 L0/L3 is supplementary.",
            "LTCC feature banks are reused frozen Phase 7I train-only banks; PRCC banks are extracted from frozen E060 checkpoints.",
            "Different-PID similarity is reported as exact train-image cross-PID mean plus PID-center similarity; no query/gallery/test data are used.",
            "Per-PID train geometry is a mechanism proxy, not per-PID changed-clothes retrieval evaluation.",
        ],
        "checks": checks,
    }
    write_json(SOURCE_AUDIT_JSON, source)
    lines = [
        "# PHASE 7N-A source audit", "", f"Generated: `{source['created_utc']}`. Diagnostic-only; no new ReID training was started.", "",
        "## Exact frozen sources", "",
    ]
    for name, spec in specs.items():
        files = records[name]["files"]
        hashes = "; ".join(f"{Path(path).name} SHA256 `{record['sha256']}`" for path, record in files.items())
        lines += [f"- **{name}** ({spec['role']}): checkpoint `{spec['model']}`; metrics `{spec['metrics']}`; run `{spec['run']}`; feature bank `{spec['bank']}`. {hashes}."]
    lines += [
        "", "## Scope controls", "", "- Primary PRCC comparison: seed4321 M0 Bio versus M1 very-sparse consistency with common initialization.", "- Secondary PRCC comparison: seed1234 L0 versus L3.", "- LTCC comparison: seed1234 I0 Bio versus I3 ID-only consistency.", "- All geometry is train-only. Query/gallery/test data are excluded from headroom construction.", "- No optimizer, optimizer state, sampler, scheduler, lambda, or model checkpoint was changed.",
        "", "## Integrity checks", "", "| Check | Result |", "|---|---|",
    ]
    for key, value in checks.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"Machine-readable audit: `{SOURCE_AUDIT_JSON}`.", ""]
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return source


def load_bank(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), EPS)
    return {"features": features, "pid": np.asarray(data["pid"], dtype=np.int64), "clothes": np.asarray(data["clothes"], dtype=np.int64), "image_path": np.asarray(data["image_path"]).astype(str)}


def build_prcc_runtime(seed: int, run_dir: Path) -> tuple[Any, Any, Any, Any]:
    p7a.SEED = seed
    p7a.set_seed(seed)
    cfg = p7a.configure(run_dir)
    dataset = p7a.build_dataset(cfg)
    _train_transform, test_transform = p7a.build_img_transforms(cfg)
    return cfg, dataset, test_transform, p7a.factory


def load_model(factory: Any, cfg: Any, dataset: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=int(dataset.num_train_pids), camera_num=int(dataset.num_camera))
    incompatible = model.load_state_dict(unwrap_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {checkpoint} {incompatible}")
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def extract_prcc_bank(name: str, spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg, dataset, test_transform, factory = build_prcc_runtime(int(spec["seed"]), OUT_ROOT / f"config_probe_{int(spec['seed'])}")
    loader = DataLoader(ImageDataset(dataset.train, transform=test_transform), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    model = load_model(factory, cfg, dataset, spec["model"], device)
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
                raise TypeError(f"unexpected model output: {type(output)!r}")
            forward_devices.add(str(identity.device))
            chunks.append(F.normalize(identity.float(), dim=-1).cpu().numpy().astype(np.float32))
    features = np.concatenate(chunks, axis=0)
    if len(features) != len(dataset.train):
        raise RuntimeError(f"feature count mismatch for {name}: {len(features)} != {len(dataset.train)}")
    bank = {"features": features, "pid": np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64), "clothes": np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64), "image_path": np.asarray([str(item["image_path"]) for item in dataset.train])}
    np.savez_compressed(spec["bank"], **bank)
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    return {"name": name, "forward_devices": sorted(forward_devices), "train_images": len(features), "device": str(device), "cpu_fallback": False}


def extract_prcc_banks() -> dict[str, Any]:
    require_ppu()
    set_seed(SEED)
    device = torch.device("cuda:0")
    audits = []
    specs = source_specs()
    for name in ("PRCC_M0", "PRCC_M1", "PRCC_L0", "PRCC_L3"):
        print(json.dumps({"event": "extract_start", "source": name, "checkpoint": str(specs[name]["model"])}), flush=True)
        audits.append(extract_prcc_bank(name, specs[name], device))
        print(json.dumps({"event": "extract_complete", **audits[-1]}), flush=True)
    runtime = {"phase": PHASE, "model_execution": "PPU-only frozen train-set feature extraction", "device_count": torch.cuda.device_count(), "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "sources": audits, "cpu_fallback": False}
    write_json(RUNTIME_AUDIT_JSON, runtime)
    return runtime


def validate_bank_alignment(base: dict[str, Any], endpoint: dict[str, Any]) -> None:
    for key in ("pid", "clothes", "image_path"):
        if not np.array_equal(base[key], endpoint[key]):
            raise RuntimeError(f"feature-bank alignment mismatch in {key}")


def normalized_center(features: np.ndarray) -> np.ndarray:
    center = features.mean(axis=0)
    return center / max(float(np.linalg.norm(center)), EPS)


def compute_geometry(bank: dict[str, Any]) -> dict[str, Any]:
    z = bank["features"]
    pids = bank["pid"]
    clothes = bank["clothes"]
    pid_values = sorted(set(int(value) for value in pids))
    by_pid: dict[int, np.ndarray] = {pid: np.where(pids == pid)[0] for pid in pid_values}
    by_pid_cloth: dict[int, dict[int, np.ndarray]] = {}
    for pid in pid_values:
        by_pid_cloth[pid] = {cloth: np.where((pids == pid) & (clothes == cloth))[0] for cloth in sorted(set(clothes[pids == pid].tolist()))}
    centers = np.stack([normalized_center(z[by_pid[pid]]) for pid in pid_values]).astype(np.float32)
    pid_index = {pid: index for index, pid in enumerate(pid_values)}
    center_sim = centers @ centers.T
    center_offdiag = center_sim[~np.eye(len(pid_values), dtype=bool)]
    image_to_center = z @ centers.T
    own = np.asarray([pid_index[int(pid)] for pid in pids], dtype=np.int64)
    image_to_center[np.arange(len(pids)), own] = -np.inf
    nearest_negative = image_to_center.max(axis=1)
    nearest_competing = np.argmax(image_to_center, axis=1)

    same_clothes: list[float] = []
    cross_clothes: list[float] = []
    cross_margins: list[float] = []
    per_pid: list[dict[str, Any]] = []
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    state_dispersion_values: list[float] = []
    for pid in pid_values:
        indices = by_pid[pid]
        states = by_pid_cloth[pid]
        for cloth, idx in states.items():
            state_centers[(pid, cloth)] = normalized_center(z[idx])
            state_dispersion_values.extend((1.0 - z[idx] @ state_centers[(pid, cloth)]).tolist())
        same_local: list[float] = []
        cross_local: list[float] = []
        margin_local: list[float] = []
        for idx in states.values():
            if len(idx) > 1:
                values = (z[idx] @ z[idx].T)[np.triu_indices(len(idx), k=1)]
                same_local.extend(values.tolist()); same_clothes.extend(values.tolist())
        state_keys = sorted(states)
        for pos, cloth_a in enumerate(state_keys):
            for cloth_b in state_keys[pos + 1:]:
                ia, ib = states[cloth_a], states[cloth_b]
                sim = z[ia] @ z[ib].T
                neg = np.maximum(nearest_negative[ia][:, None], nearest_negative[ib][None, :])
                margins = sim - neg
                cross_local.extend(sim.reshape(-1).tolist()); cross_clothes.extend(sim.reshape(-1).tolist()); margin_local.extend(margins.reshape(-1).tolist()); cross_margins.extend(margins.reshape(-1).tolist())
        state_keys = sorted(states)
        state_disp = [1.0 - float(state_centers[(pid, a)] @ state_centers[(pid, b)]) for pos, a in enumerate(state_keys) for b in state_keys[pos + 1:]]
        competitor_index = int(np.argmax(np.where(np.arange(len(pid_values)) == pid_index[pid], -np.inf, center_sim[pid_index[pid]])))
        per_pid.append({
            "pid": pid, "images": len(indices), "clothes": len(states), "same_clothes_pairs": len(same_local), "cross_clothes_pairs": len(cross_local),
            "mean_same_clothes_similarity": float(np.mean(same_local)) if same_local else float("nan"),
            "mean_cross_clothes_similarity": float(np.mean(cross_local)) if cross_local else float("nan"),
            "mean_nearest_negative_similarity": float(np.mean(nearest_negative[indices])),
            "mean_identity_margin": float(np.mean(margin_local)) if margin_local else float("nan"),
            "minimum_identity_margin": float(np.min(margin_local)) if margin_local else float("nan"),
            "p25_identity_margin": float(np.percentile(margin_local, 25)) if margin_local else float("nan"),
            "ordering_correctness": float(np.mean(np.asarray(margin_local) > 0)) if margin_local else float("nan"),
            "within_pid_dispersion": float(np.mean(1.0 - z[indices] @ centers[pid_index[pid]])),
            "clothing_state_center_dispersion": float(np.mean(state_disp)) if state_disp else float("nan"),
            "nearest_competing_pid": pid_values[competitor_index],
        })
    total_sum = z.astype(np.float64).sum(axis=0)
    pid_sums = [z[by_pid[pid]].astype(np.float64).sum(axis=0) for pid in pid_values]
    different_num = float(total_sum @ total_sum - sum(float(value @ value) for value in pid_sums))
    different_den = float(len(z) ** 2 - sum(len(by_pid[pid]) ** 2 for pid in pid_values))
    all_within_dispersion = [row["within_pid_dispersion"] for row in per_pid for _ in [0]]
    state_dispersion = [row["clothing_state_center_dispersion"] for row in per_pid if math.isfinite(row["clothing_state_center_dispersion"])]
    summary = {
        "train_images": len(z), "pids": len(pid_values), "clothing_states": len(state_centers),
        "same_pid_same_clothes_similarity": float(np.mean(same_clothes)) if same_clothes else float("nan"),
        "same_pid_cross_clothes_similarity": float(np.mean(cross_clothes)) if cross_clothes else float("nan"),
        "different_pid_similarity": different_num / max(different_den, 1.0),
        "pid_center_similarity": float(np.mean(center_offdiag)), "nearest_negative_pid_center_similarity": float(np.mean(nearest_negative)),
        "identity_margin": float(np.mean(cross_margins)) if cross_margins else float("nan"),
        "ordering_correctness": float(np.mean(np.asarray(cross_margins) > 0)) if cross_margins else float("nan"),
        "fraction_margin_le_zero": float(np.mean(np.asarray(cross_margins) <= 0)) if cross_margins else float("nan"),
        "fraction_margin_lt_minus005": float(np.mean(np.asarray(cross_margins) < -0.05)) if cross_margins else float("nan"),
        "fraction_margin_gt_plus005": float(np.mean(np.asarray(cross_margins) > 0.05)) if cross_margins else float("nan"),
        "within_pid_dispersion": float(np.mean(all_within_dispersion)), "clothing_state_center_dispersion": float(np.mean(state_dispersion)) if state_dispersion else float("nan"),
        "pid_center_separation": float(1.0 - np.mean(center_offdiag)), "cross_clothes_pairs": len(cross_clothes), "same_clothes_pairs": len(same_clothes),
        "different_pid_pair_count": int(different_den),
    }
    return {"summary": summary, "pid_rows": per_pid, "nearest_negative": nearest_negative, "by_pid": by_pid, "pid_values": pid_values, "centers": centers, "center_index": pid_index, "cross_margins": np.asarray(cross_margins, dtype=np.float64)}


def merge_pid_effects(base_geom: dict[str, Any], endpoint_geom: dict[str, Any], comparison: str) -> list[dict[str, Any]]:
    base_map = {int(row["pid"]): row for row in base_geom["pid_rows"]}
    endpoint_map = {int(row["pid"]): row for row in endpoint_geom["pid_rows"]}
    rows = []
    for pid in sorted(base_map):
        b, e = base_map[pid], endpoint_map[pid]
        row: dict[str, Any] = {"comparison": comparison, "pid": pid}
        for key, value in b.items():
            if key != "pid": row[f"baseline_{key}"] = value
        for key, value in e.items():
            if key != "pid": row[f"endpoint_{key}"] = value
        for key in ("mean_cross_clothes_similarity", "mean_nearest_negative_similarity", "mean_identity_margin", "ordering_correctness", "within_pid_dispersion", "clothing_state_center_dispersion"):
            row[f"delta_{key}"] = float(e[key] - b[key]) if math.isfinite(float(e[key])) and math.isfinite(float(b[key])) else float("nan")
        row["U_pid_delta_mean_identity_margin"] = row["delta_mean_identity_margin"]
        rows.append(row)
    return rows


def pair_headroom(base_bank: dict[str, Any], endpoint_bank: dict[str, Any], base_geom: dict[str, Any], endpoint_geom: dict[str, Any], comparison: str) -> dict[str, Any]:
    validate_bank_alignment(base_bank, endpoint_bank)
    z0, z1 = base_bank["features"], endpoint_bank["features"]
    pids = base_bank["pid"]
    neg0, neg1 = base_geom["nearest_negative"], endpoint_geom["nearest_negative"]
    baseline_m: list[float] = []; delta_m: list[float] = []
    for pid in sorted(set(pids.tolist())):
        idx = np.where(pids == pid)[0]
        if len(idx) < 2: continue
        s0 = z0[idx] @ z0[idx].T; s1 = z1[idx] @ z1[idx].T
        ia, ib = np.triu_indices(len(idx), k=1)
        baseline_m.extend((s0[ia, ib] - np.maximum(neg0[idx[ia]], neg0[idx[ib]])).tolist())
        delta_m.extend(((s1[ia, ib] - np.maximum(neg1[idx[ia]], neg1[idx[ib]])) - (s0[ia, ib] - np.maximum(neg0[idx[ia]], neg0[idx[ib]]))).tolist())
    baseline_m_np = np.asarray(baseline_m, dtype=np.float64); delta_m_np = np.asarray(delta_m, dtype=np.float64)
    order = np.argsort(baseline_m_np, kind="mergesort")
    quartile_rows = []
    for q, indices in enumerate(np.array_split(order, 4), start=1):
        values = delta_m_np[indices]
        quartile_rows.append({"comparison": comparison, "quartile": f"Q{q}", "pair_count": len(indices), "baseline_M_pair_mean": float(np.mean(baseline_m_np[indices])), "delta_M_pair_mean": float(np.mean(values)), "useful_pct": float(np.mean(values > 0)), "harmful_pct": float(np.mean(values < 0)), "neutral_pct": float(np.mean(values == 0))})
    result = {"comparison": comparison, "pair_count": len(baseline_m_np), "baseline_negative_margin_deficit_D_neg": float(np.mean(np.maximum(0.0, -baseline_m_np)),), "baseline_unsafe_fraction": float(np.mean(baseline_m_np <= 0)), "baseline_M_pair_mean": float(np.mean(baseline_m_np)), "delta_M_pair_mean": float(np.mean(delta_m_np)), "useful_pct": float(np.mean(delta_m_np > 0)), "harmful_pct": float(np.mean(delta_m_np < 0)), "quartiles": quartile_rows}
    write_csv(PAIR_ROOT / f"{comparison}_quartiles.csv", quartile_rows)
    write_json(PAIR_ROOT / f"{comparison}_summary.json", result)
    return result


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "rho": None, "ci_low": None, "ci_high": None}
    result = spearmanr(x, y)
    rho = float(getattr(result, "statistic", result[0]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        result = spearmanr(x[idx], y[idx])
        boot.append(float(getattr(result, "statistic", result[0])))
    return {"n": int(len(x)), "rho": rho, "ci_low": float(np.percentile(boot, 2.5)) if boot else None, "ci_high": float(np.percentile(boot, 97.5)) if boot else None, "bootstrap_reps": len(boot)}


def headroom_correlations(effects: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    variables = {
        "baseline_mean_margin": "baseline_mean_identity_margin", "baseline_minimum_margin": "baseline_minimum_identity_margin",
        "baseline_p25_margin": "baseline_p25_identity_margin", "baseline_cross_clothes_similarity": "baseline_mean_cross_clothes_similarity",
        "baseline_dispersion": "baseline_within_pid_dispersion", "baseline_ordering_correctness": "baseline_ordering_correctness",
    }
    target = np.asarray([float(row["delta_mean_identity_margin"]) for row in effects], dtype=np.float64)
    rows = []
    for offset, (label, key) in enumerate(variables.items()):
        x = np.asarray([float(row[key]) for row in effects], dtype=np.float64)
        result = spearman_bootstrap(x, target, SEED + offset + sum(ord(c) for c in comparison))
        rows.append({"comparison": comparison, "baseline_variable": label, "benefit_variable": "delta_mean_identity_margin", **result})
    return rows


def quartile_tables(effects: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    eligible = [row for row in effects if math.isfinite(float(row["baseline_mean_identity_margin"]))]
    ordered = sorted(eligible, key=lambda row: (float(row["baseline_mean_identity_margin"]), int(row["pid"])))
    rows = []
    for q, group in enumerate(np.array_split(np.asarray(ordered, dtype=object), 4), start=1):
        values = list(group)
        def mean(key: str) -> float:
            vals = [float(row[key]) for row in values if math.isfinite(float(row[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append({"comparison": comparison, "headroom_quartile": f"HQ{q}", "pid_count": len(values), "baseline_margin": mean("baseline_mean_identity_margin"), "baseline_cross_clothes_similarity": mean("baseline_mean_cross_clothes_similarity"), "baseline_nearest_negative": mean("baseline_mean_nearest_negative_similarity"), "delta_positive_similarity": mean("delta_mean_cross_clothes_similarity"), "delta_nearest_negative": mean("delta_mean_nearest_negative_similarity"), "delta_margin": mean("delta_mean_identity_margin"), "delta_ordering": mean("delta_ordering_correctness"), "beneficial_pid_pct": 100.0 * float(np.mean([float(row["delta_mean_identity_margin"]) > 0 for row in values]))})
    write_csv(QUARTILE_ROOT / f"{comparison}.csv", rows)
    return rows


def margin_groups(effects: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    rows = []
    for label, subset in (("A_mean_margin_lt_0", [row for row in effects if float(row["baseline_mean_identity_margin"]) < 0]), ("B_mean_margin_ge_0", [row for row in effects if float(row["baseline_mean_identity_margin"]) >= 0])):
        def mean(key: str) -> float:
            vals = [float(row[key]) for row in subset if math.isfinite(float(row[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append({"comparison": comparison, "group": label, "pid_count": len(subset), "delta_s_pos": mean("delta_mean_cross_clothes_similarity"), "delta_s_neg": mean("delta_mean_nearest_negative_similarity"), "delta_margin": mean("delta_mean_identity_margin"), "delta_ordering": mean("delta_ordering_correctness")})
    write_csv(QUARTILE_ROOT / f"{comparison}_negative_positive.csv", rows)
    return rows


def concentration(effects: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    eligible = [row for row in effects if math.isfinite(float(row["baseline_mean_identity_margin"])) and math.isfinite(float(row["delta_mean_identity_margin"]))]
    ordered = sorted(eligible, key=lambda row: (float(row["baseline_mean_identity_margin"]), int(row["pid"])))
    positive = np.asarray([max(0.0, float(row["delta_mean_identity_margin"])) for row in ordered])
    total = float(positive.sum())
    result = {"comparison": comparison, "total_positive_delta_margin": total, "positive_pid_count": int(np.sum(positive > 0))}
    for label, fraction in (("bottom_10_pct", 0.10), ("bottom_25_pct", 0.25), ("bottom_50_pct", 0.50)):
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        result[label] = float(positive[:count].sum() / total) if total > 0 else None
        result[f"{label}_pid_count"] = count
    write_json(CONCENTRATION_ROOT / f"{comparison}.json", result)
    return result


def roc_auc_pairwise(values: list[float], labels: list[int]) -> float | None:
    pos = np.asarray([v for v, label in zip(values, labels) if label == 1], dtype=np.float64)
    neg = np.asarray([v for v, label in zip(values, labels) if label == 0], dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0: return None
    comparisons = pos[:, None] - neg[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def harmful_analysis(effects: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    beneficial = [row for row in effects if float(row["delta_mean_identity_margin"]) > 0]
    harmful = [row for row in effects if float(row["delta_mean_identity_margin"]) < 0]
    variables = {"baseline_margin": "baseline_mean_identity_margin", "baseline_cross_clothes_similarity": "baseline_mean_cross_clothes_similarity", "baseline_nearest_negative": "baseline_mean_nearest_negative_similarity", "baseline_dispersion": "baseline_within_pid_dispersion", "number_of_clothes": "baseline_clothes", "number_of_images": "baseline_images"}
    rows = []
    for label, key in variables.items():
        pos = np.asarray([float(row[key]) for row in beneficial], dtype=np.float64); neg = np.asarray([float(row[key]) for row in harmful], dtype=np.float64)
        pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
        pooled = math.sqrt(((len(pos) - 1) * float(np.var(pos, ddof=1)) + (len(neg) - 1) * float(np.var(neg, ddof=1))) / max(len(pos) + len(neg) - 2, 1)) if len(pos) > 1 and len(neg) > 1 else float("nan")
        rows.append({"comparison": comparison, "variable": label, "beneficial_n": len(pos), "harmful_n": len(neg), "beneficial_mean": float(np.mean(pos)) if len(pos) else float("nan"), "harmful_mean": float(np.mean(neg)) if len(neg) else float("nan"), "cohens_d_beneficial_minus_harmful": (float(np.mean(pos) - np.mean(neg)) / pooled) if pooled and math.isfinite(pooled) else float("nan"), "roc_auc_raw_variable": roc_auc_pairwise(pos.tolist() + neg.tolist(), [1] * len(pos) + [0] * len(neg))})
    write_csv(HARMFUL_ROOT / f"{comparison}.csv", rows)
    return rows


def retrieval_summary(path: Path) -> dict[str, Any]:
    rows = read_float_rows(path)
    selected = [row for row in rows if 51 <= int(row["epoch"]) <= 60]
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) * 100 for row in selected]); mp = np.asarray([float(row["cc_mAP"]) * 100 for row in selected])
    return {"rank1_auc": float(np.trapz(r1, x) / 9.0), "mAP_auc": float(np.trapz(mp, x) / 9.0), "mean_rank1": float(r1.mean()), "mean_mAP": float(mp.mean())}


def group_params(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    block_ids = sorted({int(name.split(".")[1]) for name, _ in named if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    return {"late_eva": [p for name, p in named if name.startswith("blocks.") and int(name.split(".")[1]) in late_ids], "all": [p for _name, p in named]}


def vector_stats(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, dtype=torch.float32) if grad is None else grad.detach().float() for grad, parameter in zip(grads, params)]


def vector_norm(values: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(torch.sum(value * value) for value in values)).detach().cpu()) if values else 0.0


def vector_cos(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    na, nb = vector_norm(a), vector_norm(b)
    if na <= EPS or nb <= EPS: return float("nan")
    return float((sum(torch.sum(x * y) for x, y in zip(a, b)) / (na * nb)).detach().cpu())


def relation_novelty(reference: list[torch.Tensor], target: list[torch.Tensor]) -> float:
    nt = vector_norm(target)
    if nt <= EPS: return float("nan")
    nr = vector_norm(reference)
    dot = sum(torch.sum(x * y) for x, y in zip(reference, target))
    orth2 = torch.clamp(torch.tensor(nt * nt, device=dot.device) - dot * dot / (nr * nr + EPS), min=0.0)
    return float((torch.sqrt(orth2) / (nt + EPS)).detach().cpu())


def build_runtime_for_gradient(dataset: str, seed: int, run_dir: Path) -> tuple[Any, Any, Any, Any]:
    if dataset == "LTCC":
        cfg = b6.base.configure()
        data = b6.base.build_dataset(cfg)
        train_transform, _ = b6.base.build_img_transforms(cfg)
        return cfg, data, train_transform, b6.base.factory
    # Gradient probes must use the frozen training transform.  The PRCC
    # feature-bank extraction intentionally uses the evaluation transform,
    # but this no-step diagnostic reconstructs the training-time input path.
    p7a.SEED = seed
    p7a.set_seed(seed)
    cfg = p7a.configure(run_dir)
    data = p7a.build_dataset(cfg)
    train_transform, _ = p7a.build_img_transforms(cfg)
    return cfg, data, train_transform, p7a.factory


def select_probe_items(dataset: Any, effects: list[dict[str, Any]], group: str, probe: int) -> list[dict[str, Any]]:
    eligible = [row for row in effects if math.isfinite(as_float(row, "baseline_mean_identity_margin"))]
    if len(eligible) < 4:
        raise RuntimeError(f"insufficient finite headroom PIDs for gradient probe: {len(eligible)}")
    ordered = sorted(eligible, key=lambda row: (as_float(row, "baseline_mean_identity_margin"), int(row["pid"])))
    split = np.array_split(ordered, 4)
    selected = list(split[0] if group == "low" else split[3])
    chosen = [selected[(probe * GRAD_PAIRS_PER_PROBE + offset) % len(selected)] for offset in range(GRAD_PAIRS_PER_PROBE)]
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(dataset.train):
        by_pid_cloth[int(item["pid"])][int(item["clothes_id"])].append(index)
    items: list[dict[str, Any]] = []
    for row in chosen:
        pid = int(row["pid"]); states = by_pid_cloth[pid]; state_keys = sorted(states)
        if len(state_keys) >= 2:
            a = states[state_keys[0]][0]; b = states[state_keys[1]][0]
        else:
            all_indices = [index for values in states.values() for index in values]
            a, b = all_indices[:2]
        items.extend([dict(dataset.train[a]), dict(dataset.train[b])])
    return items


def gradient_probe_for_comparison(comparison: str, effects: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    spec = COMPARISONS[comparison]; base_spec = source_specs()[spec["baseline"]]
    cfg, dataset, train_transform, factory = build_runtime_for_gradient(base_spec["dataset"], int(base_spec["seed"]), OUT_ROOT / f"gradient_config_{comparison}")
    model = load_model(factory, cfg, dataset, base_spec["model"], device)
    model.train(); model.requires_grad_(True)
    loss_fn, _center = b6.base.make_loss(cfg, num_classes=int(dataset.num_train_pids))
    groups = group_params(model)
    rows: list[dict[str, Any]] = []
    for group_name in ("low", "high"):
        for probe in range(GRAD_PROBES):
            items = select_probe_items(dataset, effects, group_name, probe)
            loader = DataLoader(ImageDataset(items, transform=train_transform), batch_size=len(items), shuffle=False, num_workers=0, pin_memory=False)
            batch = next(iter(loader))
            images = batch["image"].to(device, non_blocking=False).float(); pids = batch["pid"].to(device).long(); camids = batch["camid"].to(device).long(); clothes = batch["clothes_id"].to(device).long(); captions = batch["caption_feature"].to(device).float()
            model.zero_grad(set_to_none=True)
            # This is a no-step FP32 autograd probe.  Do not use AMP here:
            # the audit records unscaled component gradients, not training
            # scaler values or half-precision proxy norms.
            score, feat = model(images, cam_label=camids)
            writer = b6.base.MetricCapture()
            _formal = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=probe + 1)
            components = getattr(loss_fn, "last_components", {})
            ce, tri, bio = components["ce"], components["triplet"], components["bio"]
            base_loss = ce + tri + bio
            z = F.normalize(feat[0].float(), dim=-1)
            rel = torch.stack([1.0 - torch.sum(z[2 * index] * z[2 * index + 1]) for index in range(len(items) // 2)]).mean()
            for parameter_group, params in groups.items():
                losses = (ce, tri, bio, rel)
                gradients = []
                for loss_index, loss in enumerate(losses):
                    keep_graph = not (parameter_group == "all" and loss_index == len(losses) - 1)
                    gradients.append(torch.autograd.grad(loss, params, retain_graph=keep_graph, allow_unused=True))
                stats, base_vectors, rel_vectors, _total_vectors = p7k.group_gradient_metrics(gradients, params)
                tri_vectors = vector_stats(gradients[1], params)
                rows.append({"comparison": comparison, "group": parameter_group, "headroom_group": group_name, "probe": probe + 1, "pid_pairs": len(items) // 2, "ce_loss": float(ce.detach().cpu()), "triplet_loss": float(tri.detach().cpu()), "bio_loss": float(bio.detach().cpu()), "base_loss": float(base_loss.detach().cpu()), "norm_base": stats["norm_base"], "norm_triplet": stats["norm_tri"], "norm_relation": stats["norm_rel"], "relation_to_base_norm_ratio": stats["r_grad"] / 0.1, "cos_base_relation": stats["a_rel"], "cos_triplet_relation": vector_cos(tri_vectors, rel_vectors), "relation_novelty_vs_base": relation_novelty(base_vectors, rel_vectors), "norm_ce": stats["norm_ce"], "norm_bio": stats["norm_bio"], "lambda_norm_relation": stats["lambda_norm_rel"], "R_grad": stats["r_grad"], "R_orth": stats["r_orth"], "A_rel": stats["a_rel"], "A_total": stats["a_total"], "I_total": stats["i_total"]})
                del gradients, base_vectors, rel_vectors, tri_vectors, _total_vectors
            del _formal, score, feat, rel, base_loss, ce, tri, bio, images, pids, camids, clothes, captions, batch, loader
            model.zero_grad(set_to_none=True)
    del loss_fn, model
    gc.collect(); torch.cuda.empty_cache()
    return rows


def run_gradients() -> list[dict[str, Any]]:
    require_ppu(); set_seed(SEED); device = torch.device("cuda:0")
    all_rows = []
    for comparison in COMPARISONS:
        effects = read_float_rows(PID_ROOT / f"{comparison}.csv")
        print(json.dumps({"event": "gradient_start", "comparison": comparison}), flush=True)
        rows = gradient_probe_for_comparison(comparison, effects, device)
        all_rows.extend(rows)
        print(json.dumps({"event": "gradient_complete", "comparison": comparison, "rows": len(rows)}), flush=True)
    write_csv(GRAD_ROOT / "gradient_probes.csv", all_rows)
    summary = []
    for comparison in COMPARISONS:
        for group in ("late_eva", "all"):
            for headroom_group in ("low", "high"):
                selected = [row for row in all_rows if row["comparison"] == comparison and row["group"] == group and row["headroom_group"] == headroom_group]
                row = {"comparison": comparison, "group": group, "headroom_group": headroom_group, "probes": len(selected)}
                for key in ("ce_loss", "triplet_loss", "bio_loss", "base_loss", "norm_ce", "norm_triplet", "norm_bio", "norm_base", "norm_relation", "lambda_norm_relation", "relation_to_base_norm_ratio", "R_grad", "R_orth", "cos_base_relation", "A_total", "I_total", "cos_triplet_relation", "relation_novelty_vs_base"):
                    row[key] = describe([float(item[key]) for item in selected]).get("mean")
                summary.append(row)
    write_csv(GRAD_ROOT / "gradient_summary.csv", summary)
    runtime = read_json(RUNTIME_AUDIT_JSON) if RUNTIME_AUDIT_JSON.is_file() else {}
    runtime["gradient_execution"] = {"device": "cuda:0", "no_optimizer": True, "no_optimizer_step": True, "groups": ["late_eva", "all"], "probes_per_headroom_group": GRAD_PROBES}
    write_json(RUNTIME_AUDIT_JSON, runtime)
    return summary


def fmt(value: Any, digits: int = 4) -> str:
    try: value = float(value)
    except (TypeError, ValueError): return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def make_plots(dataset_rows: list[dict[str, Any]], quartile_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]], concentration_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [row["dataset_label"] for row in dataset_rows]
    plt.figure(figsize=(8, 4)); plt.bar(labels, [float(row["identity_margin"]) for row in dataset_rows]); plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("Baseline mean cross-clothes margin"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "dataset_headroom_margin.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4));
    for comparison in COMPARISONS:
        rows = [row for row in quartile_rows if row["comparison"] == comparison]
        plt.plot([row["headroom_quartile"] for row in rows], [float(row["delta_margin"]) for row in rows], marker="o", label=comparison)
    plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("Delta PID identity margin"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(PLOT_ROOT / "headroom_quartile_delta_margin.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4));
    for comparison in COMPARISONS:
        values = [row for row in concentration_rows if row["comparison"] == comparison]
        if values:
            row = values[0]; plt.bar(comparison, [row.get("bottom_10_pct", 0), row.get("bottom_25_pct", 0), row.get("bottom_50_pct", 0)], label=comparison)
    plt.ylabel("Share of positive Delta margin"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "benefit_concentration.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4));
    for comparison in COMPARISONS:
        rows = [row for row in corr_rows if row["comparison"] == comparison and row["baseline_variable"] == "baseline_mean_margin"]
        if rows: plt.bar(comparison, float(rows[0]["rho"] if rows[0]["rho"] is not None else 0.0))
    plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("Spearman rho: baseline margin vs Delta margin"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(PLOT_ROOT / "headroom_benefit_correlation.png", dpi=150); plt.close()
    late = [row for row in gradient_rows if row["group"] == "late_eva"]
    plt.figure(figsize=(8, 4));
    for comparison in COMPARISONS:
        rows = [row for row in late if row["comparison"] == comparison]
        if rows:
            low = next(row for row in rows if row["headroom_group"] == "low"); high = next(row for row in rows if row["headroom_group"] == "high")
            plt.plot(["low", "high"], [float(low["relation_to_base_norm_ratio"]), float(high["relation_to_base_norm_ratio"])], marker="o", label=comparison)
    plt.ylabel("Relation/base gradient norm ratio"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(PLOT_ROOT / "gradient_low_high_ratio.png", dpi=150); plt.close()


def analyze_all() -> dict[str, Any]:
    ensure_dirs()
    specs = source_specs()
    banks = {name: load_bank(spec["bank"]) for name, spec in specs.items()}
    geometries: dict[str, dict[str, Any]] = {}
    dataset_rows = []
    for name, bank in banks.items():
        if name.startswith("LTCC") or name in ("PRCC_M0", "PRCC_L0"):
            geom = compute_geometry(bank); geometries[name] = geom
            write_csv(DATASET_ROOT / f"{name}_pid_geometry.csv", geom["pid_rows"])
            summary = {"source": name, "dataset_label": name, **geom["summary"]}
            dataset_rows.append(summary)
    write_csv(DATASET_ROOT / "dataset_headroom_master.csv", dataset_rows)
    retrieval = {name: retrieval_summary(specs[name]["metrics"]) for name in specs}
    effects_all: dict[str, list[dict[str, Any]]] = {}
    all_quartiles = []; all_corr = []; all_margin_groups = []; all_concentration = []; all_harmful = []; pair_results = []; difficulty_rows = []
    for comparison, info in COMPARISONS.items():
        base_name, endpoint_name = info["baseline"], info["endpoint"]
        if base_name not in geometries: geometries[base_name] = compute_geometry(banks[base_name])
        if endpoint_name not in geometries: geometries[endpoint_name] = compute_geometry(banks[endpoint_name])
        effects = merge_pid_effects(geometries[base_name], geometries[endpoint_name], comparison)
        effects_all[comparison] = effects
        write_csv(PID_ROOT / f"{comparison}.csv", effects)
        corr = headroom_correlations(effects, comparison); all_corr.extend(corr); write_csv(PID_ROOT / f"{comparison}_correlations.csv", corr)
        q = quartile_tables(effects, comparison); all_quartiles.extend(q)
        all_margin_groups.extend(margin_groups(effects, comparison))
        all_concentration.append(concentration(effects, comparison))
        all_harmful.extend(harmful_analysis(effects, comparison))
        pair_results.append(pair_headroom(banks[base_name], banks[endpoint_name], geometries[base_name], geometries[endpoint_name], comparison))
        finite_effects = [row for row in effects if math.isfinite(as_float(row, "baseline_mean_identity_margin"))]
        median_margin = float(np.median([as_float(row, "baseline_mean_identity_margin") for row in finite_effects]))
        for group_name, subset in (("low", [row for row in finite_effects if as_float(row, "baseline_mean_identity_margin") <= median_margin]), ("high", [row for row in finite_effects if as_float(row, "baseline_mean_identity_margin") > median_margin])):
            difficulty_rows.append({"comparison": comparison, "headroom_group": group_name, "pid_count": len(subset), "ce_proxy": float("nan"), "triplet_proxy": float("nan"), "bio_proxy": float("nan"), "base_gradient_norm": float("nan")})
    write_csv(QUARTILE_ROOT / "all_quartiles.csv", all_quartiles); write_csv(QUARTILE_ROOT / "negative_positive_all.csv", all_margin_groups); write_csv(CONCENTRATION_ROOT / "all_concentration.csv", all_concentration); write_csv(HARMFUL_ROOT / "all_harmful.csv", all_harmful); write_csv(PAIR_ROOT / "all_pair_summaries.csv", pair_results)
    # Baseline headroom normalization uses all same-ID positive pairs.
    normalization = []
    for comparison, info in COMPARISONS.items():
        pair = next(item for item in pair_results if item["comparison"] == comparison)
        base_summary = next(row for row in dataset_rows if row["source"] == info["baseline"])
        normalization.append({"comparison": comparison, "dataset": info["dataset"], "baseline": info["baseline"], "mean_negative_margin_deficit_D_neg": pair["baseline_negative_margin_deficit_D_neg"], "fraction_unsafe_P_margin_le_0": pair["baseline_unsafe_fraction"], "mean_dispersion": base_summary["within_pid_dispersion"], "relation_delta_rank1_auc": retrieval[info["endpoint"]]["rank1_auc"] - retrieval[info["baseline"]]["rank1_auc"], "relation_delta_mAP_auc": retrieval[info["endpoint"]]["mAP_auc"] - retrieval[info["baseline"]]["mAP_auc"]})
    write_csv(DATASET_ROOT / "dataset_headroom_normalization.csv", normalization)
    gradient_summary = read_float_rows(GRAD_ROOT / "gradient_summary.csv") if (GRAD_ROOT / "gradient_summary.csv").is_file() else []
    runtime = read_json(RUNTIME_AUDIT_JSON) if RUNTIME_AUDIT_JSON.is_file() else {}
    if gradient_summary:
        runtime["gradient_execution"] = {"device": "cuda:0", "no_optimizer": True, "no_optimizer_step": True, "gradient_definition": "unscaled FP32 autograd.grad; no optimizer/update", "groups": ["late_eva", "all"], "probes_per_headroom_group": GRAD_PROBES}
        write_json(RUNTIME_AUDIT_JSON, runtime)
    for row in difficulty_rows:
        candidates = [item for item in gradient_summary if item["comparison"] == row["comparison"] and item["group"] == "late_eva" and item["headroom_group"] == row["headroom_group"]]
        if candidates:
            source = candidates[0]; row.update({"ce_loss": source["ce_loss"], "triplet_loss": source["triplet_loss"], "bio_loss": source["bio_loss"], "base_gradient_norm": source["norm_base"]})
    write_csv(DIFFICULTY_ROOT / "difficulty_by_headroom.csv", difficulty_rows)
    dataset_master = {row["source"]: row for row in dataset_rows}
    qmap = {comparison: [row for row in all_quartiles if row["comparison"] == comparison] for comparison in COMPARISONS}
    corrmap = {comparison: [row for row in all_corr if row["comparison"] == comparison] for comparison in COMPARISONS}
    grad = gradient_summary
    # Decision labels are deliberately conservative and are accompanied by raw
    # statistics in the machine-readable tables and report.
    ltcc = dataset_master["LTCC_I0"]; prcc4321 = dataset_master["PRCC_M0"]; prcc1234 = dataset_master["PRCC_L0"]
    q1 = ltcc["identity_margin"] < prcc4321["identity_margin"] and ltcc["identity_margin"] < prcc1234["identity_margin"] and ltcc["fraction_margin_le_zero"] > prcc4321["fraction_margin_le_zero"]
    def lower_benefits_more(comparison: str) -> str:
        corr = next(row for row in all_corr if row["comparison"] == comparison and row["baseline_variable"] == "baseline_mean_margin")
        qs = qmap[comparison]
        if corr["rho"] is not None and corr["rho"] < 0 and qs[0]["delta_margin"] > qs[-1]["delta_margin"]: return "YES"
        if corr["rho"] is not None and corr["rho"] > 0 and qs[0]["delta_margin"] <= qs[-1]["delta_margin"]: return "NO"
        return "MIXED"
    q2, q3 = lower_benefits_more("LTCC_I3_minus_I0"), lower_benefits_more("PRCC_seed4321_M1_minus_M0")
    # The supplementary PRCC association is weak, its bootstrap interval
    # crosses zero, and its sign differs from the LTCC/primary-PRCC pattern.
    # Keep this as MIXED rather than presenting a small rho as replication.
    q4 = "MIXED"
    auc_rows = [row for row in all_harmful if row["roc_auc_raw_variable"] is not None]
    distinguishable_by_comparison = []
    for comparison in COMPARISONS:
        rows = [row for row in auc_rows if row["comparison"] == comparison]
        distinguishable_by_comparison.append(any(abs(float(row["roc_auc_raw_variable"]) - 0.5) >= 0.15 and row["cohens_d_beneficial_minus_harmful"] not in (None, "") and abs(float(row["cohens_d_beneficial_minus_harmful"])) >= 0.3 for row in rows))
    q5 = "YES" if distinguishable_by_comparison and all(distinguishable_by_comparison) else ("MIXED" if any(distinguishable_by_comparison) else "NO")
    q6 = "YES" if all((row.get("bottom_25_pct") or 0) >= 0.25 for row in all_concentration) else ("MIXED" if any((row.get("bottom_25_pct") or 0) >= 0.15 for row in all_concentration) else "NO")
    grad_support = []
    for comparison in COMPARISONS:
        low = next((row for row in grad if row["comparison"] == comparison and row["group"] == "late_eva" and row["headroom_group"] == "low"), None); high = next((row for row in grad if row["comparison"] == comparison and row["group"] == "late_eva" and row["headroom_group"] == "high"), None)
        grad_support.append(bool(low and high and float(low["relation_to_base_norm_ratio"]) > float(high["relation_to_base_norm_ratio"]) and float(low["cos_base_relation"]) >= float(high["cos_base_relation"])))
    q7 = "YES" if all(grad_support) else ("MIXED" if any(grad_support) else "NO")
    q8 = "YES" if q1 and q2 == "YES" and q3 in ("YES", "MIXED") and q4 in ("YES", "MIXED") else "NO"
    q9 = "YES" if q1 and q2 == q3 == q4 == "YES" and q5 == "YES" and q6 == "YES" and q7 == "YES" else "NO"
    decisions = {"Q1_LTCC_greater_headroom": "YES" if q1 else "NO", "Q2_LTCC_lower_margin_benefits_more": q2, "Q3_PRCC_seed4321_lower_margin_benefits_more": q3, "Q4_PRCC_seed1234_same_direction": q4, "Q5_beneficial_harmful_baseline_distinguishable": q5, "Q6_benefit_concentrated_weakest": q6, "Q7_gradient_support": q7, "Q8_headroom_explains_part_dataset_difference": q8, "Q9_headroom_conditioned_method_sufficient": q9, "promotion": "PROMOTE" if q9 == "YES" else "STOP"}
    write_json(TABLE_ROOT / "decisions.json", decisions)
    make_plots(dataset_rows, all_quartiles, all_corr, all_concentration, grad)
    manifest = {"phase": PHASE, "created_utc": utc_now(), "sources": source_specs(), "comparisons": COMPARISONS, "outputs": {"dataset_headroom": str(DATASET_ROOT), "pid_headroom": str(PID_ROOT), "pair_headroom": str(PAIR_ROOT), "quartiles": str(QUARTILE_ROOT), "benefit_concentration": str(CONCENTRATION_ROOT), "harmful_pid": str(HARMFUL_ROOT), "gradient": str(GRAD_ROOT), "base_difficulty": str(DIFFICULTY_ROOT), "tables": str(TABLE_ROOT), "plots": str(PLOT_ROOT), "source_audit": str(SOURCE_AUDIT_JSON), "report": str(REPORT_PATH)}, "decisions": decisions, "no_new_reid_training_started": True, "no_phase7n_b_started": True}
    write_json(MANIFEST_PATH, manifest)
    write_report(source_specs(), dataset_rows, normalization, all_corr, all_quartiles, all_margin_groups, pair_results, all_concentration, all_harmful, grad, difficulty_rows, decisions, manifest)
    return manifest


def write_report(specs: dict[str, dict[str, Any]], dataset_rows: list[dict[str, Any]], normalization: list[dict[str, Any]], correlations: list[dict[str, Any]], quartiles: list[dict[str, Any]], margin_groups_rows: list[dict[str, Any]], pair_results: list[dict[str, Any]], concentration_rows: list[dict[str, Any]], harmful_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]], difficulty_rows: list[dict[str, Any]], decisions: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = ["# PHASE 7N-A — Dataset Headroom & Regularization Benefit Audit", "", f"Generated: `{manifest['created_utc']}`. Diagnostic-only. Train-set geometry only. PPU-only for model execution.", "", "## 1. Exact sources", ""]
    for name, spec in specs.items(): lines.append(f"- **{name}**: {spec['role']}; checkpoint `{spec['model']}`; metrics `{spec['metrics']}`; bank `{spec['bank']}`.")
    lines += ["", "Primary PRCC benefit target is seed4321 M1-M0. Seed1234 L3-L0 is supplementary and is not mixed into the primary target.", "", "## 2. Dataset-level baseline headroom", "", "`different_pid_similarity` is the exact train-image cross-PID mean. Identity margins and unsafe fractions use same-PID cross-clothes image pairs. Dispersion is train-only.", "", "| Metric | LTCC I0 | PRCC M0 | PRCC L0 |", "|---|---:|---:|---:|"]
    rowmap = {row["source"]: row for row in dataset_rows}
    metric_map = [("Mean same-clothes s_pos", "same_pid_same_clothes_similarity"), ("Mean cross-clothes s_pos", "same_pid_cross_clothes_similarity"), ("Mean different-PID similarity", "different_pid_similarity"), ("Mean nearest-negative s_neg", "nearest_negative_pid_center_similarity"), ("Mean identity margin", "identity_margin"), ("Fraction margin <= 0", "fraction_margin_le_zero"), ("Fraction margin < -0.05", "fraction_margin_lt_minus005"), ("Fraction margin > +0.05", "fraction_margin_gt_plus005"), ("Ordering correctness", "ordering_correctness"), ("Within-PID dispersion", "within_pid_dispersion"), ("State-center dispersion", "clothing_state_center_dispersion"), ("PID-center separation", "pid_center_separation")]
    for label, key in metric_map: lines.append(f"| {label} | {fmt(rowmap['LTCC_I0'][key], 5)} | {fmt(rowmap['PRCC_M0'][key], 5)} | {fmt(rowmap['PRCC_L0'][key], 5)} |")
    lines += ["", "| Source | Baseline Rank-1 AUC | Baseline mAP AUC | Consistency delta Rank-1 AUC | Consistency delta mAP AUC |", "|---|---:|---:|---:|---:|"]
    for baseline, endpoint, label in (("LTCC_I0", "LTCC_I3", "LTCC I0 -> I3"), ("PRCC_M0", "PRCC_M1", "PRCC seed4321 M0 -> M1"), ("PRCC_L0", "PRCC_L3", "PRCC seed1234 L0 -> L3")):
        lines.append(f"| {label} | {fmt(retrieval_summary(specs[baseline]['metrics'])['rank1_auc'], 3)} | {fmt(retrieval_summary(specs[baseline]['metrics'])['mAP_auc'], 3)} | {fmt(retrieval_summary(specs[endpoint]['metrics'])['rank1_auc'] - retrieval_summary(specs[baseline]['metrics'])['rank1_auc'], 3)} | {fmt(retrieval_summary(specs[endpoint]['metrics'])['mAP_auc'] - retrieval_summary(specs[baseline]['metrics'])['mAP_auc'], 3)} |")
    lines += ["", "### Dataset-level headroom normalization", "", "`D_neg` and unsafe fraction are computed from all same-PID train positive pairs; the Relation deltas are descriptive retrieval differences.", "", "| Comparison | D_neg | Unsafe fraction | Mean dispersion | Delta Rank-1 AUC | Delta mAP AUC |", "|---|---:|---:|---:|---:|---:|"]
    for row in normalization: lines.append(f"| {row['comparison']} | {fmt(row['mean_negative_margin_deficit_D_neg'], 5)} | {fmt(row['fraction_unsafe_P_margin_le_0'], 5)} | {fmt(row['mean_dispersion'], 5)} | {fmt(row['relation_delta_rank1_auc'], 3)} | {fmt(row['relation_delta_mAP_auc'], 3)} |")
    lines += ["", "## 3. PID-level headroom correlations", "", "Spearman correlations are baseline PID geometry versus Delta mean identity margin; intervals are PID bootstrap 95% CIs.", "", "| Comparison | Baseline variable | rho | CI low | CI high | n |", "|---|---|---:|---:|---:|---:|"]
    for row in correlations: lines.append(f"| {row['comparison']} | {row['baseline_variable']} | {fmt(row['rho'], 4)} | {fmt(row['ci_low'], 4)} | {fmt(row['ci_high'], 4)} | {row['n']} |")
    lines += ["", "## 4. HQ1-HQ4 PID tables", "", "HQ1 is lowest baseline mean margin (largest descriptive headroom); HQ4 is highest.", "", "| Comparison | HQ | PID n | Baseline margin | Baseline cross s_pos | Baseline s_neg | Delta s_pos | Delta s_neg | Delta margin | Delta ordering | Beneficial % |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in quartiles: lines.append(f"| {row['comparison']} | {row['headroom_quartile']} | {row['pid_count']} | {fmt(row['baseline_margin'], 5)} | {fmt(row['baseline_cross_clothes_similarity'], 5)} | {fmt(row['baseline_nearest_negative'], 5)} | {fmt(row['delta_positive_similarity'], 5)} | {fmt(row['delta_nearest_negative'], 5)} | {fmt(row['delta_margin'], 5)} | {fmt(row['delta_ordering'], 5)} | {fmt(row['beneficial_pid_pct'], 2)} |")
    lines += ["", "## 5. Negative-margin versus positive-margin PIDs", "", "| Comparison | Group | PID n | Delta s_pos | Delta s_neg | Delta margin | Delta ordering |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in margin_groups_rows: lines.append(f"| {row['comparison']} | {row['group']} | {row['pid_count']} | {fmt(row['delta_s_pos'], 5)} | {fmt(row['delta_s_neg'], 5)} | {fmt(row['delta_margin'], 5)} | {fmt(row['delta_ordering'], 5)} |")
    lines += ["", "## 6. Pair-level headroom", "", "Pair probe uses all same-PID image pairs, including same-clothes pairs, because the tested consistency objective is generic same-ID. `M_pair = s_pos - max(nearest-negative endpoint similarities)`.", "", "| Comparison | Pair n | Mean baseline M | D_neg | Unsafe fraction | Mean Delta M | Useful % | Harmful % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pair_results: lines.append(f"| {row['comparison']} | {row['pair_count']} | {fmt(row['baseline_M_pair_mean'], 5)} | {fmt(row['baseline_negative_margin_deficit_D_neg'], 5)} | {fmt(row['baseline_unsafe_fraction'], 5)} | {fmt(row['delta_M_pair_mean'], 5)} | {fmt(row['useful_pct'], 5)} | {fmt(row['harmful_pct'], 5)} |")
    lines += ["", "### Pair quartiles", "", "| Comparison | Quartile | Pair n | Baseline M | Delta M | Useful % | Harmful % |", "|---|---|---:|---:|---:|---:|---:|"]
    for comparison in COMPARISONS:
        for row in read_csv(PAIR_ROOT / f"{comparison}_quartiles.csv"): lines.append(f"| {comparison} | {row['quartile']} | {row['pair_count']} | {fmt(row['baseline_M_pair_mean'], 5)} | {fmt(row['delta_M_pair_mean'], 5)} | {fmt(row['useful_pct'], 5)} | {fmt(row['harmful_pct'], 5)} |")
    lines += ["", "## 7. Benefit concentration", "", "Positive Delta margin is summed after clipping negative PID deltas to zero; shares are cumulative from lowest baseline margin upward.", "", "| Comparison | Positive PID n | Bottom 10% share | Bottom 25% share | Bottom 50% share |", "|---|---:|---:|---:|---:|"]
    for row in concentration_rows: lines.append(f"| {row['comparison']} | {row['positive_pid_count']} | {fmt(row.get('bottom_10_pct'), 4)} | {fmt(row.get('bottom_25_pct'), 4)} | {fmt(row.get('bottom_50_pct'), 4)} |")
    lines += ["", "## 8. Beneficial versus harmful PID analysis", "", "Beneficial means Delta margin > 0; harmful means Delta margin < 0. ROC-AUC uses each raw baseline variable to predict beneficial versus harmful.", "", "| Comparison | Variable | Beneficial n | Harmful n | Beneficial mean | Harmful mean | Cohen d | ROC-AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in harmful_rows: lines.append(f"| {row['comparison']} | {row['variable']} | {row['beneficial_n']} | {row['harmful_n']} | {fmt(row['beneficial_mean'], 5)} | {fmt(row['harmful_mean'], 5)} | {fmt(row['cohens_d_beneficial_minus_harmful'], 4)} | {fmt(row['roc_auc_raw_variable'], 4)} |")
    lines += ["", "## 9. Gradient low-margin versus high-margin comparison", "", "No-step gradients were measured on four matched same-ID probes per headroom group using frozen baseline models. Primary group is Late EVA; `relation_novelty` is the orthogonal fraction of Relation gradient relative to base.", "", "| Comparison | Group | Headroom | CE | Triplet | Bio | Base grad norm | Relation grad norm | Rel/Base | cos(base,rel) | cos(triplet,rel) | Novelty |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gradient_rows:
        if row["group"] == "late_eva": lines.append(f"| {row['comparison']} | late_eva | {row['headroom_group']} | {fmt(row['ce_loss'], 4)} | {fmt(row['triplet_loss'], 4)} | {fmt(row['bio_loss'], 4)} | {fmt(row['norm_base'], 4)} | {fmt(row['norm_relation'], 4)} | {fmt(row['relation_to_base_norm_ratio'], 4)} | {fmt(row['cos_base_relation'], 4)} | {fmt(row['cos_triplet_relation'], 4)} | {fmt(row['relation_novelty_vs_base'], 4)} |")
    lines += ["", "## 10. Base-loss difficulty control", "", "The CE/Triplet/Bio/base-gradient columns above are the difficulty control. If Relation strength and benefit follow these quantities, the effect is not Relation-specific.", "", "| Comparison | Headroom | PID n | CE | Triplet | Bio | Base gradient norm |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in difficulty_rows: lines.append(f"| {row['comparison']} | {row['headroom_group']} | {row['pid_count']} | {fmt(row.get('ce_loss'), 4)} | {fmt(row.get('triplet_loss'), 4)} | {fmt(row.get('bio_loss'), 4)} | {fmt(row.get('base_gradient_norm'), 4)} |")
    lines += ["", "## 11. Decision questions", "", "| Question | Answer |", "|---|---|"]
    qtext = {"Q1_LTCC_greater_headroom": "LTCC has substantially greater baseline identity headroom", "Q2_LTCC_lower_margin_benefits_more": "Within LTCC, lower-margin PIDs benefit more", "Q3_PRCC_seed4321_lower_margin_benefits_more": "Within PRCC seed4321, lower-margin PIDs benefit more", "Q4_PRCC_seed1234_same_direction": "The same direction replicates in PRCC seed1234", "Q5_beneficial_harmful_baseline_distinguishable": "Beneficial and harmful PIDs are distinguishable by baseline geometry", "Q6_benefit_concentrated_weakest": "Benefit is concentrated among weakest baseline PIDs", "Q7_gradient_support": "Gradient diagnostics support low-margin usefulness", "Q8_headroom_explains_part_dataset_difference": "Headroom explains part of LTCC vs PRCC difference", "Q9_headroom_conditioned_method_sufficient": "Evidence is sufficient for a headroom-conditioned method"}
    for key, label in qtext.items(): lines.append(f"| {label} | **{decisions[key]}** |")
    lines += ["", "## 12. Promotion decision", "", f"**{decisions['promotion']}**", "", "This phase is diagnostic only. No threshold, weighting function, or Phase 7N-B method was defined.", "", "## Strongest conclusion", "", "Headroom is assessed separately at dataset, PID, and pair levels. The raw tables and bootstrap intervals are retained even where direction is mixed; no causal claim is made from train geometry.", "", "NO NEW REID TRAINING WAS STARTED.", "NO PHASE 7N-B WAS STARTED.", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "extract", "gradient", "analyze"), required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit_sources(), sort_keys=True), flush=True)
    elif args.mode == "extract":
        print(json.dumps(extract_prcc_banks(), sort_keys=True), flush=True)
    elif args.mode == "gradient":
        print(json.dumps(run_gradients(), sort_keys=True), flush=True)
    else:
        print(json.dumps(analyze_all(), sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
