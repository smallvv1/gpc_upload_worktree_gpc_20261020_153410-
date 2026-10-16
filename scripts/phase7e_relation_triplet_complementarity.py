#!/usr/local/bin/python3
"""Phase 7E: train-only Relation--Triplet complementarity audit.

This module is intentionally diagnostic only.  It loads frozen checkpoints,
replays the frozen train batch plans, and uses unscaled FP32 autograd probes.
There is no optimizer, scheduler, GradScaler, backward(), or optimizer.step().
No query/gallery/test record is used for relation construction.
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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402
from loss.triplet_loss import euclidean_dist, hard_example_mining  # noqa: E402
import phase5b_id_damage_ds as base  # noqa: E402
import phase6b_soft_relation_screen as phase6b  # noqa: E402
import phase7a_prcc_relation_validation as phase7a  # noqa: E402


PHASE = "phase7e_relation_triplet_complementarity"
SEED = 1234
WORLD_SIZE = 2
DEVICE_INDEX = 0
TARGET_GRADIENT_BATCHES = 8
EPS = 1e-12
MARGIN = 0.05

OUT_ROOT = ROOT / "outputs/new_method_track/phase7e_relation_triplet_complementarity"
SOURCE_ROOT = OUT_ROOT / "source_audit"
ALIGN_ROOT = OUT_ROOT / "gradient_alignment"
NOVELTY_ROOT = OUT_ROOT / "gradient_novelty"
COVERAGE_ROOT = OUT_ROOT / "triplet_positive_coverage"
RELONLY_ROOT = OUT_ROOT / "relation_only_pairs"
NEG_ROOT = OUT_ROOT / "hard_negative_overlap"
ACT_ROOT = OUT_ROOT / "activation_overlap"
STAGE_ROOT = OUT_ROOT / "stage_analysis"
PID_ROOT = OUT_ROOT / "per_pid_analysis"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"
MANIFEST_PATH = OUT_ROOT / "PHASE7E_MANIFEST.json"
SOURCE_AUDIT_JSON = SOURCE_ROOT / "source_audit.json"
SOURCE_REPORT = ROOT / "reports/PHASE_7E_SOURCE_AUDIT.md"
REPORT_PATH = ROOT / "reports/PHASE_7E_RELATION_TRIPLET_COMPLEMENTARITY_AUDIT.md"

P6C = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
P6B = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen"
P6CR = ROOT / "outputs/new_method_track/phase6cr_relation_replication"
P7A = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
P7C = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
P7DB = ROOT / "outputs/new_method_track/phase7d_b_margin_preserving_relation"

CODE_FILES = {
    "loss_make_loss": ROOT / "loss/make_loss.py",
    "loss_triplet_loss": ROOT / "loss/triplet_loss.py",
    "phase6b_relation_runner": ROOT / "scripts/phase6b_soft_relation_screen.py",
    "phase7a_relation_runner": ROOT / "scripts/phase7a_prcc_relation_validation.py",
    "phase7d_b_margin_runner": ROOT / "scripts/phase7d_b_margin_preserving_relation.py",
}

SOURCE_ARMS: dict[str, dict[str, Any]] = {
    "ltcc_b0_seed1234": {
        "dataset": "LTCC", "seed": 1234, "arm": "B0", "role": "Bio control",
        "phase": "Phase 6C", "run": P6C / "B0_full", "model": P6C / "B0_full/model_final.pth",
    },
    "ltcc_b1_seed1234": {
        "dataset": "LTCC", "seed": 1234, "arm": "B1", "role": "Uniform relation",
        "phase": "Phase 6C", "run": P6C / "B1_full", "model": P6C / "B1_full/model_final.pth",
    },
    "ltcc_r0_seed4321": {
        "dataset": "LTCC", "seed": 4321, "arm": "R0", "role": "Bio control secondary",
        "phase": "Phase 6C-R", "run": P6CR / "R0_seed4321", "model": P6CR / "R0_seed4321/model_final.pth",
    },
    "ltcc_r1_seed4321": {
        "dataset": "LTCC", "seed": 4321, "arm": "R1", "role": "Uniform relation secondary",
        "phase": "Phase 6C-R", "run": P6CR / "R1_seed4321", "model": P6CR / "R1_seed4321/model_final.pth",
    },
    "prcc_c0_seed1234": {
        "dataset": "PRCC", "seed": 1234, "arm": "C0", "role": "Bio control",
        "phase": "Phase 7C", "run": P7C / "training/C0_bio", "model": P7C / "training/C0_bio/model_final.pth",
    },
    "prcc_c1_seed1234": {
        "dataset": "PRCC", "seed": 1234, "arm": "C1", "role": "Image-level uniform relation",
        "phase": "Phase 7C", "run": P7C / "training/C1_image_uniform_relation", "model": P7C / "training/C1_image_uniform_relation/model_final.pth",
    },
    "prcc_d1_seed1234": {
        "dataset": "PRCC", "seed": 1234, "arm": "D1", "role": "Margin-preserving relation",
        "phase": "Phase 7D-B", "run": P7DB / "training/D1_margin_preserving_relation", "model": P7DB / "training/D1_margin_preserving_relation/model_final.pth",
    },
}

STAGES: list[dict[str, Any]] = [
    {"stage": "ltcc_e30", "dataset": "ltcc", "epoch": 30, "source": P6B / "B0_uniform_base/model_screen_final.pth", "relation_source": P6B / "B1_uniform_relation/model_screen_final.pth", "d1": False},
    {"stage": "ltcc_e60", "dataset": "ltcc", "epoch": 60, "source": P6C / "B0_full/model_final.pth", "relation_source": P6C / "B1_full/model_final.pth", "d1": False},
    {"stage": "prcc_e28", "dataset": "prcc", "epoch": 28, "source": P7C / "training/C0_bio/model_best.pth", "relation_source": P7C / "training/C1_image_uniform_relation/model_best.pth", "d1": False},
    {"stage": "prcc_e60", "dataset": "prcc", "epoch": 60, "source": P7C / "training/C0_bio/model_final.pth", "relation_source": P7C / "training/C1_image_uniform_relation/model_final.pth", "d1": False},
    {"stage": "prcc_d1_e44", "dataset": "prcc", "epoch": 44, "source": P7DB / "training/D1_margin_preserving_relation/model_best.pth", "relation_source": None, "d1": True},
    {"stage": "prcc_d1_e60", "dataset": "prcc", "epoch": 60, "source": P7DB / "training/D1_margin_preserving_relation/model_final.pth", "relation_source": None, "d1": True},
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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sha256_state(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SOURCE_ROOT, ALIGN_ROOT, NOVELTY_ROOT, COVERAGE_ROOT,
                 RELONLY_ROOT, NEG_ROOT, ACT_ROOT, STAGE_ROOT, PID_ROOT,
                 TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def require_ppu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase 7E requires exactly {WORLD_SIZE} visible PPUs; CPU fallback is forbidden")
    torch.cuda.set_device(DEVICE_INDEX)
    return torch.device(f"cuda:{DEVICE_INDEX}")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def checkpoint_feature_dim(path: Path) -> int | None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    weight = state.get("head.weight") if isinstance(state, dict) else None
    return int(weight.shape[1]) if weight is not None else None


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    required_code = {name: file_record(path) for name, path in CODE_FILES.items()}
    if not all(item["exists"] for item in required_code.values()):
        raise RuntimeError("missing authoritative loss/relation source code")
    triplet_hashes = {required_code["loss_make_loss"]["sha256"], required_code["loss_triplet_loss"]["sha256"]}
    arms: dict[str, Any] = {}
    for name, spec in SOURCE_ARMS.items():
        run = Path(spec["run"])
        required = [spec["model"], run / "config.yml", run / "run_manifest.json", run / "checkpoint_epoch060.json"]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"incomplete frozen source for {name}: {[str(p) for p in required if not p.is_file()]}")
        manifest = json.loads((run / "run_manifest.json").read_text())
        arm = dict(spec)
        arm.update({
            "run": str(run),
            "model": file_record(spec["model"]),
            "config": file_record(run / "config.yml"),
            "run_manifest": file_record(run / "run_manifest.json"),
            "checkpoint_meta": file_record(run / "checkpoint_epoch060.json"),
            "model_tensor_state_hash": sha256_state(spec["model"]),
            "manifest": manifest,
            "feature_source_verified": manifest.get("relation_feature", "feat[0]") in ("feat[0]", "normalized feat[0]"),
        })
        arms[name] = arm
    stage_records = []
    for stage in STAGES:
        for key in ("source", "relation_source"):
            path = stage.get(key)
            if path is not None and not path.is_file():
                raise RuntimeError(f"missing stage checkpoint {path}")
        stage_records.append({
            "stage": stage["stage"], "dataset": stage["dataset"], "actual_epoch": stage["epoch"],
            "source": file_record(stage["source"]),
            "relation_source": file_record(stage["relation_source"]) if stage.get("relation_source") else None,
            "is_nearest_available_checkpoint": stage["stage"] in ("prcc_e28", "prcc_d1_e44"),
        })
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "source_verification_pass": True,
        "training_started": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "scheduler_created": False,
        "gradscaler_created": False,
        "query_gallery_used_for_relation": False,
        "primary_seed": SEED,
        "world_size": WORLD_SIZE,
        "code": required_code,
        "triplet_code_hash_set": sorted(triplet_hashes),
        "triplet_implementation_identical_within_dataset_comparisons": True,
        "source_arms": arms,
        "stage_sources": stage_records,
        "ltcc_secondary_seed4321_audited": True,
        "ltcc_secondary_seed4321_probed": False,
        "loss_decomposition": "loss/make_loss.py components ce + triplet + bio; relation is frozen phase6b 1-cos on feat[0]",
        "gradient_mode": "unscaled FP32 autograd.grad, model forward on cuda:0, no optimizer step",
    }
    write_json(SOURCE_AUDIT_JSON, audit)
    lines = [
        "# Phase 7E source audit", "", f"Generated: `{audit['created_utc']}`.", "",
        "Decision: **PASS**. Exact frozen sources and shared Triplet implementation were found.", "",
        "This audit is for diagnostics only. Phase 7E creates no optimizer, scheduler, GradScaler, or training loop.", "",
        "## Frozen sources", "", "| ID | Dataset | Arm | Checkpoint | Epoch/role | Feature source |", "|---|---|---|---|---|---|",
    ]
    for name, item in arms.items():
        manifest = item["manifest"]
        lines.append(f"| {name} | {item['dataset']} | {item['arm']} | `{item['model']['path']}` | {item['phase']} | `{manifest.get('relation_feature', 'feat[0]')}` |")
    lines += ["", "## Authoritative implementation", "", f"- `loss/make_loss.py` SHA256: `{required_code['loss_make_loss']['sha256']}`.", f"- `loss/triplet_loss.py` SHA256: `{required_code['loss_triplet_loss']['sha256']}`.", "- Matched LTCC and PRCC configs use `LOSS_TYPE=[ce,triplet,clipBio]`, `DATA.SAMPLER=softmax_triplet`, `NO_MARGIN=true`, so the exact repository SoftMargin Triplet implementation is shared.", "- Relation feature is `feat[0]` (normalized inside relation loss); D1 margin uses the frozen normalized-`feat[0]` pair-conservative negative definition.", "", "## Stage availability", "", "| Stage | Dataset | Actual checkpoint epoch | Note |", "|---|---|---:|---|"]
    for stage in stage_records:
        note = "nearest available frozen checkpoint" if stage["is_nearest_available_checkpoint"] else "exact requested endpoint"
        lines.append(f"| {stage['stage']} | {stage['dataset']} | {stage['actual_epoch']} | {note} |")
    lines += ["", "Secondary LTCC seed4321 sources were audited but not used as an additional diagnostic run; primary analysis uses seed1234 as requested.", ""]
    SOURCE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return audit


def cfg_dataset(dataset_name: str) -> tuple[Any, Any, Any]:
    if dataset_name == "ltcc":
        cfg = base.configure()
        dataset = base.build_dataset(cfg)
        _train_transform, test_transform = base.build_img_transforms(cfg)
        return cfg, dataset, test_transform
    cfg = phase7a.configure(OUT_ROOT / "_config_probe")
    dataset = phase7a.build_dataset(cfg)
    _train_transform, test_transform = phase7a.build_img_transforms(cfg)
    return cfg, dataset, test_transform


def relation_bank(dataset_name: str, dataset: Any, cfg: Any) -> dict[str, Any]:
    if dataset_name == "ltcc":
        bank = phase6b.parse_relation_bank(dataset, cfg)
    else:
        bank = phase7a.load_relation_bank(dataset)
    bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    return bank


def make_loader(dataset_name: str, dataset: Any, cfg: Any, test_transform: Any, epoch: int) -> Any:
    if dataset_name == "ltcc":
        active = phase6b.full_active_state(dataset)
        index_map = base.build_index_map(dataset)
        loader, _meta = base.make_epoch_loader(dataset, test_transform, active, index_map, epoch=epoch, rank=0)
        return loader
    sampler = DistributedRandomIdentitySampler(dataset.train, num_instances=4, num_replicas=WORLD_SIZE, rank=0, seed=SEED)
    return DataLoader(ImageDataset(dataset.train, transform=test_transform), sampler=sampler, batch_size=16,
                      num_workers=0, pin_memory=False, drop_last=True)


def build_pair_arrays(dataset: Any) -> dict[str, np.ndarray]:
    by_pid_cloth: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    pids = np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64)
    clothes = np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64)
    for index, item in enumerate(dataset.train):
        by_pid_cloth[int(item["pid"])][int(item["clothes_id"])].append(index)
    left: list[int] = []; right: list[int] = []; pair_pid: list[int] = []; cloth_a: list[int] = []; cloth_b: list[int] = []
    for pid in sorted(by_pid_cloth):
        states = by_pid_cloth[pid]
        labels = sorted(states)
        for pos, ca in enumerate(labels[:-1]):
            for cb in labels[pos + 1:]:
                for a in states[ca]:
                    for b in states[cb]:
                        left.append(a); right.append(b); pair_pid.append(pid); cloth_a.append(ca); cloth_b.append(cb)
    return {"a": np.asarray(left, dtype=np.int32), "b": np.asarray(right, dtype=np.int32),
            "pid": np.asarray(pair_pid, dtype=np.int32), "cloth_a": np.asarray(cloth_a, dtype=np.int32),
            "cloth_b": np.asarray(cloth_b, dtype=np.int32), "pids": pids, "clothes": clothes}


def local_pairs(batch: dict[str, Any], bank: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pids = batch["pid"].tolist(); clothes = batch["clothes_id"].tolist()
    indices = [bank["path_to_index"][str(path)] for path in batch["image_path"]]
    all_edges: list[tuple[int, int]] = []; rel_edges: list[tuple[int, int]] = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if int(pids[i]) != int(pids[j]) or int(clothes[i]) == int(clothes[j]):
                continue
            edge = (i, j); all_edges.append(edge)
            key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
            if key in bank["exact_lookup"]:
                rel_edges.append(edge)
    return all_edges, rel_edges


def model_from_checkpoint(dataset_name: str, cfg: Any, dataset: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    factory = base.factory if dataset_name == "ltcc" else phase7a.factory
    model = factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=dataset.num_train_pids, camera_num=dataset.num_camera)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch at {checkpoint}: {incompatible}")
    model.to(device)
    model.train()  # matches the frozen training forward mode used by prior probes
    model.requires_grad_(True)
    return model


def parameter_groups(model: torch.nn.Module) -> dict[str, set[str]]:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    blocks = sorted({int(name.split(".")[1]) for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late = set(blocks[-3:])
    return {
        "late_eva": {name for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late},
        "id_head": {name for name in names if name.startswith("head.")},
        "bio_projection": {name for name in names if name.startswith("head_bio") or name.startswith("head_clip_bio") or name.startswith("norm_bio")},
    }


def component_grads(loss: torch.Tensor, params: list[torch.nn.Parameter], keep: bool) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(loss, params, retain_graph=keep, allow_unused=True)


def group_norm(grads: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float:
    total = 0.0
    for name, grad in zip(names, grads):
        if name in selected and grad is not None:
            total += float(torch.sum(grad.detach().float() ** 2).cpu())
    return math.sqrt(max(total, 0.0))


def group_dot(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float:
    value = 0.0
    for name, a, b in zip(names, left, right):
        if name not in selected:
            continue
        if a is None and b is None:
            continue
        if a is None: a = torch.zeros_like(b)
        if b is None: b = torch.zeros_like(a)
        value += float(torch.sum(a.detach().float() * b.detach().float()).cpu())
    return value


def sum_grads(*grad_sets: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor | None, ...]:
    result: list[torch.Tensor | None] = []
    for values in zip(*grad_sets):
        tensors = [value for value in values if value is not None]
        result.append(None if not tensors else sum(tensors))
    return tuple(result)


def alignment(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> float | None:
    nl = group_norm(left, names, selected); nr = group_norm(right, names, selected)
    return group_dot(left, right, names, selected) / (nl * nr + EPS) if nl > EPS and nr > EPS else None


def novelty(relation: tuple[torch.Tensor | None, ...], triplet: tuple[torch.Tensor | None, ...], names: list[str], selected: set[str]) -> dict[str, float | None]:
    nr = group_norm(relation, names, selected); nt = group_norm(triplet, names, selected)
    dot = group_dot(relation, triplet, names, selected)
    if nr <= EPS:
        return {"relation_norm": nr, "relation_orth_norm": 0.0, "novelty_ratio": None}
    scale = dot / (nt * nt + EPS)
    orth_sq = 0.0
    for name, rg, tg in zip(names, relation, triplet):
        if name not in selected:
            continue
        if rg is None and tg is None:
            continue
        if rg is None: rg = torch.zeros_like(tg)
        if tg is None: tg = torch.zeros_like(rg)
        diff = rg.detach().float() - scale * tg.detach().float()
        orth_sq += float(torch.sum(diff * diff).cpu())
    orth = math.sqrt(max(orth_sq, 0.0))
    return {"relation_norm": nr, "relation_orth_norm": orth, "novelty_ratio": orth / (nr + EPS)}


def d1_margin_loss(representation: torch.Tensor, pids: torch.Tensor, relation_edges: list[tuple[int, int]]) -> tuple[torch.Tensor, dict[int, int], torch.Tensor]:
    sim = representation @ representation.t()
    neg_mask = pids[:, None].ne(pids[None, :])
    neg_sim = sim.masked_fill(~neg_mask, -1e4)
    values, indices = neg_sim.max(dim=1)
    losses = []
    for i, j in relation_edges:
        negative = torch.maximum(values[i], values[j])
        losses.append(F.relu(MARGIN + negative - sim[i, j]))
    if not losses:
        zero = representation.sum() * 0.0
        return zero, {i: int(indices[i]) for i in range(len(indices))}, indices
    return torch.stack(losses).mean(), {i: int(indices[i]) for i in range(len(indices))}, indices


def pairwise_batch_stats(feat: torch.Tensor, pids: torch.Tensor, clothes: torch.Tensor, all_edges: list[tuple[int, int]], rel_edges: list[tuple[int, int]], d1: bool) -> dict[str, Any]:
    z = F.normalize(feat.float(), dim=-1)
    dist = euclidean_dist(feat.float(), feat.float())
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist, pids, return_inds=True)
    hard_edges = {(min(i, int(p_inds[i])), max(i, int(p_inds[i]))) for i in range(len(p_inds))
                  if int(pids[i]) == int(pids[int(p_inds[i])]) and int(clothes[i]) != int(clothes[int(p_inds[i])])}
    relation_set = set(rel_edges)
    covered = relation_set & hard_edges
    relation_only = relation_set - hard_edges
    sim = z @ z.t()
    neg_mask = pids[:, None].ne(pids[None, :])
    neg_values, d1_inds = sim.masked_fill(~neg_mask, -1e4).max(dim=1)
    margin_active = set()
    relation_metrics: list[dict[str, Any]] = []
    for i, j in rel_edges:
        neg = max(float(neg_values[i].detach().cpu()), float(neg_values[j].detach().cpu()))
        value = float((MARGIN + neg - float(sim[i, j].detach().cpu())))
        if value > 0:
            margin_active.add((i, j))
        if (i, j) in relation_only:
            relation_metrics.append({
                "pid": int(pids[i]), "pos_similarity": float(sim[i, j].detach().cpu()),
                "nearest_negative_similarity": neg, "identity_margin": float(sim[i, j].detach().cpu()) - neg,
                "triplet_loss_contribution": 0.0,
            })
    active_anchor = (dist_an <= dist_ap).detach().cpu().numpy().astype(bool)
    relation_triplet_active = sum(bool(active_anchor[i] or active_anchor[j]) for i, j in rel_edges)
    margin_triplet_active = sum(bool(active_anchor[i] or active_anchor[j]) for i, j in margin_active)
    same_negative = different_negative = exact_negative = 0
    if d1:
        for i in range(len(n_inds)):
            ti = int(n_inds[i]); di = int(d1_inds[i])
            if ti == di: exact_negative += 1
            elif int(pids[ti]) == int(pids[di]): same_negative += 1
            else: different_negative += 1
    hard_cross = sum(int(clothes[i]) != int(clothes[int(p_inds[i])]) for i in range(len(p_inds)))
    cross_sim = [float(sim[i, j].detach().cpu()) for i, j in all_edges]
    same_sim = [float(sim[i, j].detach().cpu()) for i in range(len(pids)) for j in range(i + 1, len(pids))
                if int(pids[i]) == int(pids[j]) and int(clothes[i]) == int(clothes[j])]
    return {
        "all_relation_count": len(all_edges), "relation_count": len(rel_edges), "covered_relation_count": len(covered),
        "relation_only_count": len(relation_only), "hard_positive_cross_count": hard_cross,
        "hard_positive_count": len(p_inds), "triplet_active_anchors": int(active_anchor.sum()),
        "relation_triplet_active_count": relation_triplet_active, "relation_triplet_inactive_count": len(rel_edges) - relation_triplet_active,
        "margin_active_count": len(margin_active), "margin_triplet_active_count": margin_triplet_active,
        "margin_triplet_inactive_count": len(margin_active) - margin_triplet_active,
        "triplet_hard_positive_similarity_mean": float(dist_ap.detach().float().mean().cpu()),
        "triplet_hard_negative_distance_mean": float(dist_an.detach().float().mean().cpu()),
        "hard_positive_cross_clothes_rate": hard_cross / max(1, len(p_inds)),
        "cross_clothes_triplet_coverage": len(covered) / max(1, len(rel_edges)),
        "relation_only_fraction": len(relation_only) / max(1, len(rel_edges)),
        "relation_active_and_triplet_active_fraction": relation_triplet_active / max(1, len(rel_edges)),
        "relation_active_but_triplet_inactive_fraction": (len(rel_edges) - relation_triplet_active) / max(1, len(rel_edges)),
        "margin_active_and_triplet_active_fraction": margin_triplet_active / max(1, len(margin_active)),
        "margin_active_but_triplet_inactive_fraction": (len(margin_active) - margin_triplet_active) / max(1, len(margin_active)),
        "same_clothes_positive_similarity_mean": float(np.mean(same_sim)) if same_sim else None,
        "cross_clothes_positive_similarity_mean": float(np.mean(cross_sim)) if cross_sim else None,
        "relation_only_metrics": relation_metrics,
        "exact_negative_count": exact_negative, "same_negative_pid_count": same_negative,
        "different_negative_pid_count": different_negative, "negative_comparison_count": len(n_inds),
    }


def gradient_row(model: torch.nn.Module, loss_fn: Any, batch: dict[str, Any], device: torch.device,
                 bank: dict[str, Any], all_edges: list[tuple[int, int]], rel_edges: list[tuple[int, int]],
                 stage: dict[str, Any], batch_index: int, names: list[str], params: list[torch.nn.Parameter], groups: dict[str, set[str]],
                 do_margin: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    model.zero_grad(set_to_none=True)
    score, feat = model(images, cam_label=camids)
    writer = base.MetricCapture()
    total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=batch_index)
    components = loss_fn.last_components
    ce = components["ce"]; tri = components["triplet"]; bio = components["bio"]
    base_loss = ce + tri + bio
    rep = F.normalize(feat[0].float(), dim=-1)
    rel_terms = [1.0 - torch.sum(rep[i] * rep[j]) for i, j in rel_edges]
    rel = torch.stack(rel_terms).mean() if rel_terms else rep.sum() * 0.0
    margin, _d1_map, _d1_indices = d1_margin_loss(rep, pids, rel_edges)
    losses = [ce, tri, bio, rel]
    if do_margin:
        losses.append(margin)
    grads: list[tuple[torch.Tensor | None, ...]] = []
    for index, value in enumerate(losses):
        grads.append(component_grads(value, params, keep=index < len(losses) - 1))
    ce_g, tri_g, bio_g, rel_g = grads[:4]
    margin_g = grads[4] if do_margin else None
    base_g = sum_grads(ce_g, tri_g, bio_g)
    row: dict[str, Any] = {
        "stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "batch_index": batch_index,
        "batch_size": int(len(pids)), "all_relation_count": len(all_edges), "relation_count": len(rel_edges),
        "ce_loss": float(ce.detach().float().cpu()), "triplet_loss": float(tri.detach().float().cpu()),
        "bio_loss": float(bio.detach().float().cpu()), "base_loss": float(base_loss.detach().float().cpu()),
        "relation_loss": float(rel.detach().float().cpu()), "margin_loss": float(margin.detach().float().cpu()) if do_margin else None,
        "total_loss_from_frozen_components": float(total.detach().float().cpu()), "gradient_dtype": "float32",
        "autocast_enabled": False, "optimizer_step_called": False,
    }
    novelty_row: dict[str, Any] = {"stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "batch_index": batch_index}
    for group_name, selected in groups.items():
        prefix = group_name
        row[f"norm_ce_{prefix}"] = group_norm(ce_g, names, selected)
        row[f"norm_tri_{prefix}"] = group_norm(tri_g, names, selected)
        row[f"norm_bio_{prefix}"] = group_norm(bio_g, names, selected)
        row[f"norm_rel_{prefix}"] = group_norm(rel_g, names, selected)
        row[f"cos_ce_tri_{prefix}"] = alignment(ce_g, tri_g, names, selected)
        row[f"cos_ce_rel_{prefix}"] = alignment(ce_g, rel_g, names, selected)
        row[f"cos_tri_rel_{prefix}"] = alignment(tri_g, rel_g, names, selected)
        row[f"cos_bio_rel_{prefix}"] = alignment(bio_g, rel_g, names, selected)
        row[f"cos_base_rel_{prefix}"] = alignment(base_g, rel_g, names, selected)
        nov = novelty(rel_g, tri_g, names, selected)
        for key, value in nov.items():
            novelty_row[f"{key}_{prefix}"] = value
        if margin_g is not None:
            row[f"norm_margin_{prefix}"] = group_norm(margin_g, names, selected)
            row[f"cos_tri_margin_{prefix}"] = alignment(tri_g, margin_g, names, selected)
            row[f"cos_rel_margin_{prefix}"] = alignment(rel_g, margin_g, names, selected)
            row[f"cos_base_margin_{prefix}"] = alignment(base_g, margin_g, names, selected)
            mnov = novelty(margin_g, tri_g, names, selected)
            for key, value in mnov.items():
                novelty_row[f"margin_{key}_{prefix}"] = value
    del grads, score, feat, total, base_loss, rel, margin
    model.zero_grad(set_to_none=True)
    return row, novelty_row


def run_stage(stage: dict[str, Any], shared: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg, dataset, test_transform = shared[stage["dataset"]]
    bank = shared[f"{stage['dataset']}_bank"]
    model = model_from_checkpoint(stage["dataset"], cfg, dataset, stage["source"], device)
    loader = make_loader(stage["dataset"], dataset, cfg, test_transform, stage["epoch"])
    loss_fn, _center = base.make_loss(cfg, num_classes=dataset.num_train_pids)
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    names = [name for name, _ in named]; params = [param for _, param in named]; groups = parameter_groups(model)
    coverage_rows: list[dict[str, Any]] = []; gradient_rows: list[dict[str, Any]] = []; novelty_rows: list[dict[str, Any]] = []
    relation_only_rows: list[dict[str, Any]] = []; negative_rows: list[dict[str, Any]] = []
    inspected = 0; relation_batches = 0; gradient_probes = 0
    sums: defaultdict[str, float] = defaultdict(float); counts: defaultdict[str, int] = defaultdict(int)
    for batch in loader:
        inspected += 1
        all_edges, rel_edges = local_pairs(batch, bank)
        has_relation = bool(rel_edges)
        if has_relation: relation_batches += 1
        need_graph = has_relation and gradient_probes < TARGET_GRADIENT_BATCHES
        if need_graph:
            grad_row, nov_row = gradient_row(model, loss_fn, batch, device, bank, all_edges, rel_edges, stage, inspected, names, params, groups, stage["d1"])
            gradient_rows.append(grad_row); novelty_rows.append(nov_row); gradient_probes += 1
            # Recompute the non-gradient batch statistics from the same forward
            # would be wasteful; gradient_row has already released its graph.
            with torch.no_grad():
                _score, _feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
            feature_for_stats = _feat[0]
        else:
            with torch.no_grad():
                _score, _feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
            feature_for_stats = _feat[0]
        pids = batch["pid"].to(device); clothes = batch["clothes_id"].to(device)
        stats = pairwise_batch_stats(feature_for_stats, pids, clothes, all_edges, rel_edges, stage["d1"])
        coverage_rows.append({"stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "batch_index": inspected, **{k: v for k, v in stats.items() if k != "relation_only_metrics"}})
        for key, value in stats.items():
            if isinstance(value, (int, float)) and key not in ("exact_negative_count", "same_negative_pid_count", "different_negative_pid_count", "negative_comparison_count"):
                sums[key] += float(value); counts[key] += 1
        for item in stats["relation_only_metrics"]:
            item.update({"stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "batch_index": inspected})
            relation_only_rows.append(item)
        if stage["d1"]:
            negative_rows.append({"stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "batch_index": inspected,
                                  "exact_negative_count": stats["exact_negative_count"], "same_negative_pid_count": stats["same_negative_pid_count"],
                                  "different_negative_pid_count": stats["different_negative_pid_count"], "negative_comparison_count": stats["negative_comparison_count"]})
        del _score, _feat, feature_for_stats
        if inspected % 100 == 0:
            print(f"[phase7e] {stage['stage']}: inspected {inspected}/{len(loader)}; gradient probes {gradient_probes}", flush=True)
    summary = {"stage": stage["stage"], "dataset": stage["dataset"], "epoch": stage["epoch"], "inspected_batches": inspected,
               "relation_active_batches": relation_batches, "relation_active_batch_rate": relation_batches / max(1, inspected),
               "gradient_probes": gradient_probes, "checkpoint": str(stage["source"])}
    for key, value in sums.items(): summary[key] = value / max(1, counts[key])
    write_csv(STAGE_ROOT / f"{stage['stage']}_coverage.csv", coverage_rows)
    write_csv(ALIGN_ROOT / f"{stage['stage']}_gradient_alignment.csv", gradient_rows)
    write_csv(NOVELTY_ROOT / f"{stage['stage']}_gradient_novelty.csv", novelty_rows)
    write_csv(RELONLY_ROOT / f"{stage['stage']}_relation_only_pairs.csv", relation_only_rows)
    if negative_rows: write_csv(NEG_ROOT / f"{stage['stage']}_hard_negative_overlap.csv", negative_rows)
    write_json(STAGE_ROOT / f"{stage['stage']}_summary.json", summary)
    del loader, loss_fn, model, dataset
    gc.collect(); torch.cuda.empty_cache()
    return {"summary": summary, "gradient_rows": gradient_rows, "novelty_rows": novelty_rows,
            "coverage_rows": coverage_rows, "relation_only_rows": relation_only_rows, "negative_rows": negative_rows}


def summarize_rows(rows: list[dict[str, Any]], keys: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for group, values in groups.items():
        result = {key: value for key, value in zip(group_keys, group)}; result["count"] = len(values)
        for key in keys:
            nums = [float(row[key]) for row in values if row.get(key) not in (None, "", "None") and str(row.get(key)) != "nan"]
            if nums:
                result[f"{key}_mean"] = float(np.mean(nums)); result[f"{key}_std"] = float(np.std(nums)); result[f"{key}_median"] = float(np.median(nums)); result[f"{key}_p25"] = float(np.percentile(nums, 25)); result[f"{key}_p75"] = float(np.percentile(nums, 75))
            else:
                result[f"{key}_mean"] = None; result[f"{key}_std"] = None; result[f"{key}_median"] = None; result[f"{key}_p25"] = None; result[f"{key}_p75"] = None
        out.append(result)
    return out


def load_feature_file(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["features"], dtype=np.float32)
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def geometry_stats(features: np.ndarray, pairs: dict[str, np.ndarray], pids: np.ndarray, clothes: np.ndarray) -> dict[str, Any]:
    z = features.astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    pid_values = sorted(set(pids.tolist()))
    centers = []
    center_pid = []
    for pid in pid_values:
        center = z[pids == pid].mean(axis=0); center /= max(np.linalg.norm(center), 1e-12)
        centers.append(center); center_pid.append(pid)
    centers = np.asarray(centers, dtype=np.float32)
    center_sim = z @ centers.T
    pid_to_center = {pid: index for index, pid in enumerate(center_pid)}
    neg = np.empty(len(z), dtype=np.float32)
    for i, pid in enumerate(pids.tolist()):
        values = center_sim[i].copy(); values[pid_to_center[int(pid)]] = -1e4; neg[i] = values.max()
    pos = np.sum(z[pairs["a"]] * z[pairs["b"]], axis=1)
    neg_pair = np.maximum(neg[pairs["a"]], neg[pairs["b"]])
    margin = pos - neg_pair
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    for pid, cloth in sorted(set(zip(pids.tolist(), clothes.tolist()))):
        value = z[(pids == pid) & (clothes == cloth)].mean(axis=0); value /= max(np.linalg.norm(value), 1e-12)
        state_centers[(int(pid), int(cloth))] = value
    dispersions = []
    for pid in pid_values:
        states = [value for (ypid, _), value in state_centers.items() if ypid == pid]
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dispersions.append(1.0 - float(np.dot(states[i], states[j])))
    return {
        "pair_count": int(len(pos)), "s_pos_mean": float(np.mean(pos)), "s_pos_median": float(np.median(pos)),
        "s_neg_mean": float(np.mean(neg_pair)), "s_neg_median": float(np.median(neg_pair)),
        "margin_mean": float(np.mean(margin)), "margin_median": float(np.median(margin)),
        "margin_pct_gt0": float(np.mean(margin > 0) * 100), "margin_pct_le0": float(np.mean(margin <= 0) * 100),
        "cross_clothes_similarity": float(np.mean(pos)), "nearest_negative_similarity": float(np.mean(neg_pair)),
        "identity_margin": float(np.mean(margin)), "state_center_dispersion": float(np.mean(dispersions)) if dispersions else None,
        "state_center_pair_count": len(dispersions), "normalized_feature_mean": float(np.mean(np.linalg.norm(z, axis=1))),
    }


def geometry_analysis(shared: dict[str, Any]) -> dict[str, Any]:
    pairs_by_ds = {name: build_pair_arrays(shared[name][1]) for name in ("ltcc", "prcc")}
    feature_paths = {
        "ltcc_b0": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b0_seed1234_train_features.npz",
        "ltcc_b1": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b1_seed1234_train_features.npz",
        "prcc_c0": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C0_E60.npy",
        "prcc_c1": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C1_E60.npy",
    }
    results = {}
    for key, path in feature_paths.items():
        ds = "ltcc" if key.startswith("ltcc") else "prcc"
        pair = pairs_by_ds[ds]
        features = load_feature_file(path)
        if len(features) != len(pair["pids"]): raise RuntimeError(f"feature order/count mismatch: {path}")
        results[key] = geometry_stats(features, pair, pair["pids"], pair["clothes"])
    delta_rows = []
    for ds, b, r in (("LTCC", "ltcc_b0", "ltcc_b1"), ("PRCC", "prcc_c0", "prcc_c1")):
        delta_rows.append({"dataset": ds, "baseline": b, "relation": r, "delta_s_pos": results[r]["s_pos_mean"] - results[b]["s_pos_mean"], "delta_s_neg": results[r]["s_neg_mean"] - results[b]["s_neg_mean"], "delta_margin": results[r]["margin_mean"] - results[b]["margin_mean"], "delta_state_center_dispersion": results[r]["state_center_dispersion"] - results[b]["state_center_dispersion"]})
    write_csv(TABLE_ROOT / "geometry_primary.csv", [{"model": key, **value} for key, value in results.items()])
    write_csv(TABLE_ROOT / "geometry_deltas.csv", delta_rows)
    return {"results": results, "deltas": delta_rows}


def per_pid_analysis(shared: dict[str, Any], geom: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds, bkey, rkey in (("LTCC", "ltcc_b0", "ltcc_b1"), ("PRCC", "prcc_c0", "prcc_c1")):
        pair = build_pair_arrays(shared[ds.lower()][1])
        feature_paths = {"ltcc_b0": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b0_seed1234_train_features.npz", "ltcc_b1": ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/ltcc_b1_seed1234_train_features.npz", "prcc_c0": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C0_E60.npy", "prcc_c1": ROOT / "outputs/new_method_track/phase7d_a_margin_failure_diagnostic/feature_bank/features_C1_E60.npy"}
        base_feat = load_feature_file(feature_paths[bkey])
        rel_feat = load_feature_file(feature_paths[rkey])
        for features, prefix in ((base_feat, "baseline"), (rel_feat, "relation")):
            z = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            centers = {}; neg = np.empty(len(z), dtype=np.float32)
            for pid in sorted(set(pair["pids"].tolist())):
                c = z[pair["pids"] == pid].mean(axis=0); c /= max(np.linalg.norm(c), 1e-12); centers[pid] = c
            for i, pid in enumerate(pair["pids"].tolist()):
                neg[i] = max(float(np.dot(z[i], centers[q])) for q in centers if q != pid)
            for pid in sorted(set(pair["pid"].tolist())):
                mask = pair["pid"] == pid; ia = pair["a"][mask]; ib = pair["b"][mask]
                values = np.sum(z[ia] * z[ib], axis=1); margins = values - np.maximum(neg[ia], neg[ib])
                state_count = len(set(pair["cloth_a"][mask].tolist()) | set(pair["cloth_b"][mask].tolist()))
                images = int(np.sum(pair["pids"] == pid))
                rows.append({"dataset": ds, "pid": int(pid), "model": prefix, "num_clothes": state_count, "num_images": images, "pair_count": int(mask.sum()), "s_pos": float(values.mean()), "s_neg": float(np.maximum(neg[ia], neg[ib]).mean()), "margin": float(margins.mean())})
    pivot: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for row in rows: pivot[(row["dataset"], row["pid"])][row["model"]] = row
    delta = []
    for (ds, pid), values in pivot.items():
        if "baseline" not in values or "relation" not in values: continue
        b = values["baseline"]; r = values["relation"]
        delta.append({"dataset": ds, "pid": pid, "num_clothes": b["num_clothes"], "num_images": b["num_images"], "pair_count": b["pair_count"], "baseline_margin": b["margin"], "baseline_s_neg": b["s_neg"], "delta_s_pos": r["s_pos"] - b["s_pos"], "delta_s_neg": r["s_neg"] - b["s_neg"], "delta_margin": r["margin"] - b["margin"]})
    write_csv(PID_ROOT / "per_pid_geometry.csv", rows)
    write_csv(PID_ROOT / "per_pid_delta.csv", delta)
    return delta


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS: return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def summarize_all(stage_results: list[dict[str, Any]], geometry: dict[str, Any], per_pid: list[dict[str, Any]]) -> dict[str, Any]:
    all_grad = [row for result in stage_results for row in result["gradient_rows"]]
    all_nov = [row for result in stage_results for row in result["novelty_rows"]]
    all_cov = [row for result in stage_results for row in result["coverage_rows"]]
    gradient_summary = summarize_rows(all_grad, ["norm_ce_late_eva", "norm_tri_late_eva", "norm_bio_late_eva", "norm_rel_late_eva", "cos_ce_tri_late_eva", "cos_ce_rel_late_eva", "cos_tri_rel_late_eva", "cos_bio_rel_late_eva", "cos_base_rel_late_eva", "norm_margin_late_eva", "cos_tri_margin_late_eva", "cos_rel_margin_late_eva", "cos_base_margin_late_eva"], ["stage", "dataset"])
    novelty_summary = summarize_rows(all_nov, ["novelty_ratio_late_eva", "relation_norm_late_eva", "relation_orth_norm_late_eva", "margin_novelty_ratio_late_eva", "margin_relation_norm_late_eva", "margin_relation_orth_norm_late_eva"], ["stage", "dataset"])
    coverage_summary = summarize_rows(all_cov, ["hard_positive_cross_clothes_rate", "cross_clothes_triplet_coverage", "relation_only_fraction", "relation_active_and_triplet_active_fraction", "relation_active_but_triplet_inactive_fraction", "margin_active_count", "margin_triplet_active_count"], ["stage", "dataset"])
    write_csv(ALIGN_ROOT / "gradient_alignment_summary.csv", gradient_summary)
    write_csv(NOVELTY_ROOT / "gradient_novelty_summary.csv", novelty_summary)
    write_csv(COVERAGE_ROOT / "triplet_positive_coverage_summary.csv", coverage_summary)
    write_csv(ACT_ROOT / "activation_overlap_summary.csv", coverage_summary)
    relation_only_rows = [row for result in stage_results for row in result["relation_only_rows"]]
    relation_only_summary = summarize_rows(relation_only_rows, ["pos_similarity", "nearest_negative_similarity", "identity_margin", "triplet_loss_contribution"], ["stage", "dataset"])
    write_csv(RELONLY_ROOT / "relation_only_summary.csv", relation_only_summary)
    negative_summary = []
    for result in stage_results:
        rows = result["negative_rows"]
        if not rows:
            continue
        fields = ("exact_negative_count", "same_negative_pid_count", "different_negative_pid_count", "negative_comparison_count")
        totals = {field: sum(int(float(row[field])) for row in rows) for field in fields}
        denom = max(1, totals["negative_comparison_count"])
        negative_summary.append({"stage": rows[0]["stage"], "dataset": rows[0]["dataset"], **totals,
                                 "exact_same_sample_rate": totals["exact_negative_count"] / denom,
                                 "same_negative_pid_rate": totals["same_negative_pid_count"] / denom,
                                 "different_negative_pid_rate": totals["different_negative_pid_count"] / denom})
    write_csv(NEG_ROOT / "hard_negative_overlap_summary.csv", negative_summary)
    # Primary comparison is the late seed1234 endpoint.  E30/E28/E44 remain in
    # the stage table and are never silently substituted for E60.
    def find(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
        matches = [row for row in rows if row.get("stage") == stage]
        return matches[0] if matches else {}
    ltcc_g = find(gradient_summary, "ltcc_e60"); prcc_g = find(gradient_summary, "prcc_e60")
    ltcc_n = find(novelty_summary, "ltcc_e60"); prcc_n = find(novelty_summary, "prcc_e60")
    ltcc_c = find(coverage_summary, "ltcc_e60"); prcc_c = find(coverage_summary, "prcc_e60")
    d1_n = find(novelty_summary, "prcc_d1_e60"); d1_g = find(gradient_summary, "prcc_d1_e60")
    master = [
        {"metric": "Triplet hard-positive cross-clothes rate", "LTCC": ltcc_c.get("hard_positive_cross_clothes_rate_mean"), "PRCC": prcc_c.get("hard_positive_cross_clothes_rate_mean"), "interpretation": "higher means Triplet itself selects more changing-clothes positives"},
        {"metric": "Cross-Clothes Triplet Coverage", "LTCC": ltcc_c.get("cross_clothes_triplet_coverage_mean"), "PRCC": prcc_c.get("cross_clothes_triplet_coverage_mean"), "interpretation": "frozen relation-bank pairs selected as Triplet hard positives"},
        {"metric": "Relation-only fraction", "LTCC": ltcc_c.get("relation_only_fraction_mean"), "PRCC": prcc_c.get("relation_only_fraction_mean"), "interpretation": "relation pairs not selected as an active hard positive for either endpoint"},
        {"metric": "cos(g_tri,g_rel)", "LTCC": ltcc_g.get("cos_tri_rel_late_eva_mean"), "PRCC": prcc_g.get("cos_tri_rel_late_eva_mean"), "interpretation": "higher is more directionally redundant"},
        {"metric": "cos(g_base,g_rel)", "LTCC": ltcc_g.get("cos_base_rel_late_eva_mean"), "PRCC": prcc_g.get("cos_base_rel_late_eva_mean"), "interpretation": "relation compatibility with CE+Triplet+Bio"},
        {"metric": "Relation novelty ratio", "LTCC": ltcc_n.get("novelty_ratio_late_eva_mean"), "PRCC": prcc_n.get("novelty_ratio_late_eva_mean"), "interpretation": "orthogonal relation gradient / relation gradient"},
        {"metric": "||g_tri||", "LTCC": ltcc_g.get("norm_tri_late_eva_mean"), "PRCC": prcc_g.get("norm_tri_late_eva_mean"), "interpretation": "unscaled FP32 late-EVA gradient norm"},
        {"metric": "||g_rel||", "LTCC": ltcc_g.get("norm_rel_late_eva_mean"), "PRCC": prcc_g.get("norm_rel_late_eva_mean"), "interpretation": "unscaled FP32 late-EVA relation gradient norm"},
        {"metric": "||g_rel_orth||", "LTCC": ltcc_n.get("relation_orth_norm_late_eva_mean"), "PRCC": prcc_n.get("relation_orth_norm_late_eva_mean"), "interpretation": "relation gradient component orthogonal to Triplet"},
        {"metric": "same-PID cross-clothes similarity", "LTCC": geometry["results"]["ltcc_b0"]["s_pos_mean"], "PRCC": geometry["results"]["prcc_c0"]["s_pos_mean"], "interpretation": "Bio endpoint, train-only geometry"},
        {"metric": "nearest-negative similarity", "LTCC": geometry["results"]["ltcc_b0"]["s_neg_mean"], "PRCC": geometry["results"]["prcc_c0"]["s_neg_mean"], "interpretation": "Bio endpoint, pair-conservative negative"},
        {"metric": "identity margin", "LTCC": geometry["results"]["ltcc_b0"]["margin_mean"], "PRCC": geometry["results"]["prcc_c0"]["margin_mean"], "interpretation": "s_pos - pair-conservative nearest negative"},
        {"metric": "D1 cos(g_tri,g_margin)", "LTCC": None, "PRCC": d1_g.get("cos_tri_margin_late_eva_mean"), "interpretation": "PRCC only; D1 margin versus repository Triplet"},
        {"metric": "D1 margin novelty ratio", "LTCC": None, "PRCC": d1_n.get("margin_novelty_ratio_late_eva_mean"), "interpretation": "PRCC only; D1 margin direction not explained by Triplet"},
    ]
    write_csv(TABLE_ROOT / "ltcc_vs_prcc_master.csv", master)
    pid_corr = []
    for ds in ("LTCC", "PRCC"):
        rows = [row for row in per_pid if row["dataset"] == ds]
        pid_corr.append({"dataset": ds, "n_pids": len(rows), "num_clothes_vs_delta_margin": pearson([r["num_clothes"] for r in rows], [r["delta_margin"] for r in rows]), "pair_count_vs_delta_margin": pearson([r["pair_count"] for r in rows], [r["delta_margin"] for r in rows]), "baseline_margin_vs_delta_margin": pearson([r["baseline_margin"] for r in rows], [r["delta_margin"] for r in rows]), "baseline_s_neg_vs_delta_margin": pearson([r["baseline_s_neg"] for r in rows], [r["delta_margin"] for r in rows])})
    write_csv(PID_ROOT / "per_pid_correlations.csv", pid_corr)
    return {"gradient_summary": gradient_summary, "novelty_summary": novelty_summary, "coverage_summary": coverage_summary, "relation_only_summary": relation_only_summary, "negative_summary": negative_summary, "master": master, "pid_correlations": pid_corr}


def load_stage_results() -> list[dict[str, Any]]:
    """Reload completed PPU probes so a post-processing failure never forces a rerun."""
    results = []
    for stage in STAGES:
        name = stage["stage"]
        summary_path = STAGE_ROOT / f"{name}_summary.json"
        coverage_path = STAGE_ROOT / f"{name}_coverage.csv"
        gradient_path = ALIGN_ROOT / f"{name}_gradient_alignment.csv"
        novelty_path = NOVELTY_ROOT / f"{name}_gradient_novelty.csv"
        relation_only_path = RELONLY_ROOT / f"{name}_relation_only_pairs.csv"
        if not all(path.is_file() for path in (summary_path, coverage_path, gradient_path, novelty_path, relation_only_path)):
            raise RuntimeError(f"completed stage output is missing for {name}")
        negative_path = NEG_ROOT / f"{name}_hard_negative_overlap.csv"
        results.append({
            "summary": json.loads(summary_path.read_text()),
            "coverage_rows": read_csv(coverage_path),
            "gradient_rows": read_csv(gradient_path),
            "novelty_rows": read_csv(novelty_path),
            "relation_only_rows": read_csv(relation_only_path),
            "negative_rows": read_csv(negative_path) if negative_path.is_file() else [],
        })
    return results


def decision_logic(summary: dict[str, Any]) -> dict[str, Any]:
    def row(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
        return next((item for item in rows if item.get("stage") == stage), {})
    lc = row(summary["coverage_summary"], "ltcc_e60"); pc = row(summary["coverage_summary"], "prcc_e60")
    lg = row(summary["gradient_summary"], "ltcc_e60"); pg = row(summary["gradient_summary"], "prcc_e60")
    ln = row(summary["novelty_summary"], "ltcc_e60"); pn = row(summary["novelty_summary"], "prcc_e60")
    def val(item: dict[str, Any], key: str) -> float | None:
        value = item.get(key)
        return None if value in (None, "", "None") else float(value)
    cov_l, cov_p = val(lc, "cross_clothes_triplet_coverage_mean"), val(pc, "cross_clothes_triplet_coverage_mean")
    only_l, only_p = val(lc, "relation_only_fraction_mean"), val(pc, "relation_only_fraction_mean")
    nov_l, nov_p = val(ln, "novelty_ratio_late_eva_mean"), val(pn, "novelty_ratio_late_eva_mean")
    ali_l, ali_p = val(lg, "cos_tri_rel_late_eva_mean"), val(pg, "cos_tri_rel_late_eva_mean")
    q1 = "YES" if cov_p is not None and cov_l is not None and cov_p - cov_l > 0.05 else ("MIXED" if cov_p is not None and cov_l is not None and cov_p > cov_l else "NO")
    q2 = "YES" if only_l is not None and only_p is not None and only_l - only_p > 0.05 else ("MIXED" if only_l is not None and only_p is not None and only_l > only_p else "NO")
    q3 = "YES" if nov_l is not None and nov_p is not None and nov_l - nov_p > 0.05 else ("MIXED" if nov_l is not None and nov_p is not None and nov_l > nov_p else "NO")
    q4 = "YES" if ali_l is not None and ali_p is not None and ali_p - ali_l > 0.05 else ("MIXED" if ali_l is not None and ali_p is not None and ali_p > ali_l else "NO")
    d1g = row(summary["gradient_summary"], "prcc_d1_e60"); d1n = row(summary["novelty_summary"], "prcc_d1_e60")
    d1_align = val(d1g, "cos_tri_margin_late_eva_mean"); d1_novel = val(d1n, "margin_novelty_ratio_late_eva_mean")
    q5 = "YES" if d1_align is not None and d1_novel is not None and d1_align > 0.8 and d1_novel < 0.35 else ("MIXED" if d1_align is not None and d1_align > 0.5 else "NO")
    q6 = "YES" if nov_p is not None and ali_p is not None and nov_p > 0.5 and ali_p < 0.2 else ("MIXED" if nov_p is not None and ali_p is not None and nov_p > 0.5 else "NO")
    q7 = "YES" if q2 == "YES" and q3 == "YES" else ("MIXED" if q2 in ("YES", "MIXED") and q3 in ("YES", "MIXED") else "NO")
    q8 = "YES" if q7 == "YES" else "NO"
    if q7 == "YES": recommendation = "TRIPLET-COMPLEMENTARY RELATION"
    elif q6 == "YES": recommendation = "GRADIENT-COMPATIBLE RELATION"
    elif q5 == "YES": recommendation = "STOP MARGIN-LOSS BRANCH"
    else: recommendation = "STOP THIS HYPOTHESIS"
    result = {"Q1_triplet_cross_clothes_coverage_higher_PRCC": q1, "Q2_relation_only_fraction_higher_LTCC": q2, "Q3_relation_novelty_higher_LTCC": q3, "Q4_tri_rel_more_redundant_PRCC": q4, "Q5_D1_margin_overlaps_triplet": q5, "Q6_PRCC_novel_but_poorly_aligned": q6, "Q7_interaction_explains_part": q7, "Q8_sufficient_evidence_for_new_training": q8, "primary_values": {"ltcc_coverage": cov_l, "prcc_coverage": cov_p, "ltcc_relation_only": only_l, "prcc_relation_only": only_p, "ltcc_novelty": nov_l, "prcc_novelty": nov_p, "ltcc_alignment": ali_l, "prcc_alignment": ali_p, "d1_alignment": d1_align, "d1_novelty": d1_novel}, "recommended_next_branch": recommendation, "training_started": False, "optimizer_step_called": False, "phase7f_started": False}
    write_json(TABLE_ROOT / "decision.json", result)
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if row.get(col) is None else str(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def final_report(audit: dict[str, Any], summary: dict[str, Any], decision: dict[str, Any], geometry: dict[str, Any]) -> None:
    master = summary["master"]
    stage_rows = []
    for row in summary["coverage_summary"]:
        stage_rows.append({"stage": row.get("stage"), "dataset": row.get("dataset"), "hard_positive_cross_rate": row.get("hard_positive_cross_clothes_rate_mean"), "cross_clothes_coverage": row.get("cross_clothes_triplet_coverage_mean"), "relation_only_fraction": row.get("relation_only_fraction_mean"), "relation_active_triplet_active": row.get("relation_active_and_triplet_active_fraction_mean"), "relation_active_triplet_inactive": row.get("relation_active_but_triplet_inactive_fraction_mean")})
    grad_rows = []
    for row in summary["gradient_summary"]:
        grad_rows.append({"stage": row.get("stage"), "dataset": row.get("dataset"), "cos_tri_rel": row.get("cos_tri_rel_late_eva_mean"), "cos_base_rel": row.get("cos_base_rel_late_eva_mean"), "cos_tri_margin": row.get("cos_tri_margin_late_eva_mean"), "cos_base_margin": row.get("cos_base_margin_late_eva_mean")})
    nov_rows = []
    for row in summary["novelty_summary"]:
        nov_rows.append({"stage": row.get("stage"), "dataset": row.get("dataset"), "relation_novelty": row.get("novelty_ratio_late_eva_mean"), "relation_orth_norm": row.get("relation_orth_norm_late_eva_mean"), "margin_novelty": row.get("margin_novelty_ratio_late_eva_mean")})
    geom_rows = [{"model": key, "s_pos": value["s_pos_mean"], "s_neg": value["s_neg_mean"], "margin": value["margin_mean"], "state_dispersion": value["state_center_dispersion"]} for key, value in geometry["results"].items()]
    q_lines = [
        f"- **Q1:** {decision['Q1_triplet_cross_clothes_coverage_higher_PRCC']}",
        f"- **Q2:** {decision['Q2_relation_only_fraction_higher_LTCC']}",
        f"- **Q3:** {decision['Q3_relation_novelty_higher_LTCC']}",
        f"- **Q4:** {decision['Q4_tri_rel_more_redundant_PRCC']}",
        f"- **Q5:** {decision['Q5_D1_margin_overlaps_triplet']}",
        f"- **Q6:** {decision['Q6_PRCC_novel_but_poorly_aligned']}",
        f"- **Q7:** {decision['Q7_interaction_explains_part']}",
        f"- **Q8:** {decision['Q8_sufficient_evidence_for_new_training']}",
    ]
    strongest = ("The strongest supported conclusion is that Relation–Triplet interaction is " + ("dataset-dependent and supports a future Triplet-complementary relation branch." if decision["Q7_interaction_explains_part"] == "YES" else "not established by the joint coverage/alignment/novelty evidence; no new method is justified."))
    report = [
        "# Phase 7E — Relation–Triplet Complementarity Audit", "", f"Generated: `{utc_now()}`.", "",
        "## 1. Scope and safety", "",
        "This was a diagnostic-only audit. All relation construction used train images and train PID/clothing labels. Query/gallery/test records were not used. Frozen checkpoints were loaded on `cuda:0`; forwards were FP32 and unscaled for autograd probes. No optimizer, scheduler, GradScaler, or optimizer step was created or called.", "",
        "## 2. Exact sources", "", "Primary seed1234 sources:", "", markdown_table([{"id": name, "dataset": item["dataset"], "arm": item["arm"], "checkpoint": item["model"]["path"]} for name, item in audit["source_arms"].items()], ["id", "dataset", "arm", "checkpoint"]), "",
        "The LTCC seed4321 R0/R1 sources were audited as a secondary replication source but not probed in this phase. Exact Triplet code hashes are recorded in `source_audit/source_audit.json`; LTCC and PRCC use the same repository `loss/make_loss.py` and `loss/triplet_loss.py` implementation.", "",
        "## 3. Exact loss decomposition", "",
        "`L_base = L_CE + L_Triplet + L_Bio` was taken from the frozen component tensors. Uniform Relation was the executed `mean(1-cos(normalized feat[0]))` over the frozen relation-bank pairs. D1 used the frozen `mean(relu(0.05 + s_neg_pair - s_pos))`, margin 0.05, only for the PRCC D1 diagnostic stages.", "",
        "## 4. LTCC vs PRCC master table", "", markdown_table(master, ["metric", "LTCC", "PRCC", "interpretation"]), "",
        "## 5. Triplet coverage and relation-only supervision", "", markdown_table(stage_rows, ["stage", "dataset", "hard_positive_cross_rate", "cross_clothes_coverage", "relation_only_fraction", "relation_active_triplet_active", "relation_active_triplet_inactive"]), "",
        "Coverage is computed against the actual frozen relation bank: compact Phase6B candidate pairs for LTCC and the full train-only PRCC bank. `Relation-only` means a relation pair was not the Triplet hard positive for either endpoint in that batch. Because the repository Triplet is SoftMargin, the activation-overlap diagnostic uses `dist_an <= dist_ap` as an operational hard-example activity indicator; it is not a claim that SoftMargin has a binary active set.", "",
        "## 6. Gradient alignment", "", markdown_table(grad_rows, ["stage", "dataset", "cos_tri_rel", "cos_base_rel", "cos_tri_margin", "cos_base_margin"]), "", markdown_table(nov_rows, ["stage", "dataset", "relation_novelty", "relation_orth_norm", "margin_novelty"]), "",
        "`Relation novelty = ||g_rel - proj(g_rel on g_tri)|| / ||g_rel||`. ID-head and Bio-projection rows are present in the CSV outputs; Relation has no direct path to those heads, so their relation norms/alignments are correctly recorded as zero/NA where disconnected.", "",
        "## 7. Train-only geometry and per-PID structure", "", markdown_table(geom_rows, ["model", "s_pos", "s_neg", "margin", "state_dispersion"]), "", "The geometry delta files compare the matched Bio and Uniform Relation endpoints. Per-PID deltas and correlations are in `per_pid_analysis/`; no per-PID retrieval metric was computed.", "",
        "## 8. Relation-only, hard-negative, and activation overlap", "",
        "Relation-only pair values:", "", markdown_table(summary["relation_only_summary"], ["stage", "dataset", "count", "pos_similarity_mean", "nearest_negative_similarity_mean", "identity_margin_mean"]), "",
        "D1 hard-negative overlap:", "", markdown_table(summary["negative_summary"], ["stage", "dataset", "exact_same_sample_rate", "same_negative_pid_rate", "different_negative_pid_rate"]), "",
        "Activation overlap is in `activation_overlap/` and distinguishes Relation/Margin pairs whose endpoints satisfy the diagnostic Triplet activity proxy from those that do not.", "",
        "## 9. Q1–Q8", "", "\n".join(q_lines), "",
        "## 10. Strongest mechanistic conclusion", "", strongest, "", f"Decision values used for the late primary comparison: `{json.dumps(decision['primary_values'], sort_keys=True)}`.", "",
        "## 11. Recommended next branch", "", f"**{decision['recommended_next_branch']}**. This is only a routing recommendation; no branch is implemented here. If a future branch is authorized, it must preserve the frozen dataset splits, sampler, Bio/Triplet losses, and evaluation protocol. A high alignment/low novelty D1 result also means the margin branch should not be tuned from this audit.", "",
        "## 12. Stop statement", "", "**NO NEW TRAINING WAS STARTED. NO PHASE 7F WAS STARTED.**", "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "run", "finalize"), default="run")
    args = parser.parse_args()
    if args.mode == "audit":
        audit = source_audit()
        print(json.dumps({"source_audit": str(SOURCE_AUDIT_JSON), "report": str(SOURCE_REPORT), "passed": audit["source_verification_pass"]}, indent=2))
        return
    if not SOURCE_AUDIT_JSON.is_file(): source_audit()
    audit = json.loads(SOURCE_AUDIT_JSON.read_text())
    if not audit.get("source_verification_pass"):
        raise RuntimeError("source audit did not pass")
    if args.mode == "finalize":
        ensure_dirs()
        shared: dict[str, Any] = {}
        for name in ("ltcc", "prcc"):
            cfg, dataset, transform = cfg_dataset(name)
            shared[name] = (cfg, dataset, transform)
        stage_results = load_stage_results()
        geometry = geometry_analysis(shared)
        per_pid = per_pid_analysis(shared, geometry)
        summary = summarize_all(stage_results, geometry, per_pid)
        decision = decision_logic(summary)
        manifest = {
            "phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_AUDIT_JSON),
            "source_audit_sha256": sha256_file(SOURCE_AUDIT_JSON), "primary_seed": SEED,
            "stages": [{"stage": stage["stage"], "dataset": stage["dataset"], "actual_epoch": stage["epoch"], "checkpoint": str(stage["source"]), "gradient_probes": result["summary"].get("gradient_probes")} for stage, result in zip(STAGES, stage_results)],
            "train_only": True, "query_gallery_used_for_relation": False, "model_forward_on_ppu": True,
            "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_created": False,
            "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False,
            "training_started": False, "phase7f_started": False, "diagnostic_stage_outputs_reloaded": True,
            "decision": decision, "output_report": str(REPORT_PATH),
        }
        write_json(MANIFEST_PATH, manifest)
        final_report(audit, summary, decision, geometry)
        print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decision, "training_started": False, "phase7f_started": False}, indent=2, sort_keys=True))
        return
    device = require_ppu(); set_seed(); ensure_dirs()
    shared: dict[str, Any] = {}
    for name in ("ltcc", "prcc"):
        cfg, dataset, transform = cfg_dataset(name)
        shared[name] = (cfg, dataset, transform)
        shared[f"{name}_bank"] = relation_bank(name, dataset, cfg)
        print(f"[phase7e] {name}: train images={len(dataset.train)} pids={dataset.num_train_pids} relation bank={shared[f'{name}_bank']['exact_pair_count']}", flush=True)
    stage_results = []
    for stage in STAGES:
        print(f"[phase7e] starting {stage['stage']} checkpoint={stage['source']}", flush=True)
        stage_results.append(run_stage(stage, shared, device))
    geometry = geometry_analysis(shared)
    per_pid = per_pid_analysis(shared, geometry)
    summary = summarize_all(stage_results, geometry, per_pid)
    decision = decision_logic(summary)
    manifest = {
        "phase": PHASE, "created_utc": utc_now(), "source_audit": str(SOURCE_AUDIT_JSON),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT_JSON), "primary_seed": SEED,
        "stages": [{"stage": stage["stage"], "dataset": stage["dataset"], "actual_epoch": stage["epoch"], "checkpoint": str(stage["source"]), "gradient_probes": result["summary"]["gradient_probes"]} for stage, result in zip(STAGES, stage_results)],
        "train_only": True, "query_gallery_used_for_relation": False, "model_forward_on_ppu": True,
        "gradient_dtype": "float32", "autocast_enabled": False, "optimizer_created": False,
        "optimizer_step_called": False, "scheduler_created": False, "gradscaler_created": False,
        "training_started": False, "phase7f_started": False, "decision": decision,
        "output_report": str(REPORT_PATH),
    }
    write_json(MANIFEST_PATH, manifest)
    final_report(audit, summary, decision, geometry)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "decision": decision, "training_started": False, "phase7f_started": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
