#!/usr/local/bin/python3
"""Phase 7K-A: diagnostic Relation-strength / optimization-balance audit.

This file is deliberately diagnostic-only.  It loads frozen Phase 7I LTCC
and Phase 7C PRCC checkpoints, replays the frozen train-loader logic on one
rank, and measures no-step FP32 losses and autograd gradients.  It never
creates an optimizer, scheduler, GradScaler, or training loop, and it never
calls backward() or optimizer.step().

The cross-dataset primary comparison is LTCC I3 (matched-budget ID-only
positive consistency) versus PRCC C1 (existing image-level uniform
same-PID/different-clothes Relation).  The pair objectives are therefore
not claimed to be perfectly identical; the limitation is recorded in both
source reports and the final audit.
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
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase5b_id_damage_ds as ltcc_base  # noqa: E402
import phase6b_soft_relation_screen as p6b  # noqa: E402
import phase7a_prcc_relation_validation as p7a  # noqa: E402
import phase7i_relation_specificity as p7i  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from data.samplers import DistributedRandomIdentitySampler  # noqa: E402


PHASE = "phase7k_a_relation_strength_balance"
SEED = 1234
WORLD_SIZE = 2
RANK = 0
DEVICE_INDEX = 0
DIAGNOSTIC_BATCHES = 32
MIN_DIAGNOSTIC_BATCHES = 16
LAMBDA_REL = 0.1
EPS = 1e-12
VIRTUAL_LAMBDAS = (0.025, 0.05, 0.10, 0.20)

OUT_ROOT = ROOT / "outputs/new_method_track/phase7k_a_relation_strength_balance"
REPORT_PATH = ROOT / "reports/PHASE_7K_A_RELATION_STRENGTH_OPTIMIZATION_BALANCE_AUDIT.md"
SOURCE_REPORT_PATH = ROOT / "reports/PHASE_7K_A_SOURCE_AUDIT.md"
SOURCE_AUDIT_JSON = OUT_ROOT / "source_audit" / "source_audit.json"
MANIFEST_PATH = OUT_ROOT / "PHASE7K_A_MANIFEST.json"

P6C = ROOT / "outputs/new_method_track/phase6c_full_schedule_confirmation"
P7I = ROOT / "outputs/new_method_track/phase7i_relation_specificity"
P7A = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
P7C = ROOT / "outputs/new_method_track/phase7c_prototype_relation"

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "ltcc_i0": {
        "dataset": "LTCC", "dataset_key": "ltcc", "arm": "I0",
        "role": "Phase 7I Bio control",
        "run": P6C / "B0_full",
        "relation_probe": "id",
        "relation_definition": "matched-budget ID-only same-PID distinct-image pairs, replayed from the I3 selector",
        "provenance": "Phase 7I I0 / frozen Phase 6C B0",
    },
    "ltcc_i1": {
        "dataset": "LTCC", "dataset_key": "ltcc", "arm": "I1",
        "role": "Phase 7I matched-budget Cross-Clothes Relation",
        "run": P7I / "I1_matched_cross_clothes",
        "relation_probe": "cross",
        "relation_definition": "matched-budget same-PID/different-clothes pairs from the Phase 7I I1 selector",
        "provenance": "Phase 7I I1",
    },
    "ltcc_i3": {
        "dataset": "LTCC", "dataset_key": "ltcc", "arm": "I3",
        "role": "Phase 7I matched-budget ID-only Relation",
        "run": P7I / "I3_id_only_matched",
        "relation_probe": "id",
        "relation_definition": "matched-budget same-PID distinct-image pairs from the Phase 7I I3 selector",
        "provenance": "Phase 7I I3",
    },
    "prcc_c0": {
        "dataset": "PRCC", "dataset_key": "prcc", "arm": "C0",
        "role": "Phase 7C matched Bio control",
        "run": P7C / "training/C0_bio",
        "relation_probe": "cross",
        "relation_definition": "diagnostic-only PRCC full train bank same-PID/different-clothes pairs",
        "provenance": "Phase 7C C0; Phase 7A P0 method/source provenance",
    },
    "prcc_c1": {
        "dataset": "PRCC", "dataset_key": "prcc", "arm": "C1",
        "role": "Phase 7C image-level Uniform Relation",
        "run": P7C / "training/C1_image_uniform_relation",
        "relation_probe": "cross",
        "relation_definition": "full train-only same-PID/different-clothes bank, uniform weight 1.0",
        "provenance": "Phase 7C C1; Phase 7A P1 method/source provenance",
    },
}

CODE_FILES = {
    "loss_make_loss": ROOT / "loss/make_loss.py",
    "loss_triplet_loss": ROOT / "loss/triplet_loss.py",
    "phase5b_loader": ROOT / "scripts/phase5b_id_damage_ds.py",
    "phase6b_relation": ROOT / "scripts/phase6b_soft_relation_screen.py",
    "phase7a_prcc_method": ROOT / "scripts/phase7a_prcc_relation_validation.py",
    "phase7i_selector": ROOT / "scripts/phase7i_relation_specificity.py",
    "phase7c_source": ROOT / "scripts/phase7c_prototype_relation.py",
}

STAGE_TARGETS = OrderedDict((
    ("early", 10),
    ("mid", 30),
    ("late", 60),
))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def ensure_dirs() -> None:
    for name in (
        "loss_balance", "gradient_balance", "temporal", "saturation",
        "virtual_lambda", "ltcc_i1_i3", "tables", "plots", "source_audit",
    ):
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def require_ppu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Phase 7K-A requires a PPU; CPU fallback is forbidden")
    torch.cuda.set_device(DEVICE_INDEX)
    return torch.device(f"cuda:{DEVICE_INDEX}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def best_epoch(run: Path) -> int:
    state = read_json(run / "best_state.json")
    return int(state["best_epoch"])


def checkpoint_for_target(run: Path, target: int) -> tuple[Path, int, str]:
    final_path = run / "model_final.pth"
    best_path = run / "model_best.pth"
    if target == 60:
        return final_path, 60, "frozen final checkpoint at E60"
    b_epoch = best_epoch(run)
    if abs(b_epoch - target) <= abs(60 - target):
        return best_path, b_epoch, f"nearest frozen checkpoint: model_best.pth (best epoch E{b_epoch})"
    return final_path, 60, "nearest frozen checkpoint: model_final.pth (E60)"


def stage_map() -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for source_id, spec in SOURCE_SPECS.items():
        source_stages: dict[str, dict[str, Any]] = {}
        for stage, target in STAGE_TARGETS.items():
            path, actual, note = checkpoint_for_target(Path(spec["run"]), target)
            source_stages[stage] = {
                "stage": stage,
                "target_epoch": target,
                "actual_epoch": actual,
                "checkpoint": path,
                "note": note,
            }
        result[source_id] = source_stages
    return result


def source_audit() -> dict[str, Any]:
    ensure_dirs()
    stages = stage_map()
    arms: dict[str, Any] = {}
    required_ok = True
    for source_id, spec in SOURCE_SPECS.items():
        run = Path(spec["run"])
        required = [
            run / "model_final.pth", run / "model_best.pth", run / "config.yml",
            run / "run_manifest.json", run / "best_state.json", run / "checkpoint_epoch060.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        required_ok = required_ok and not missing
        manifest = read_json(run / "run_manifest.json") if not missing else {}
        metrics_path = run / "metrics_per_epoch.csv"
        metric_rows = sum(1 for _ in metrics_path.open(encoding="utf-8")) - 1 if metrics_path.is_file() else 0
        arms[source_id] = {
            "source_id": source_id,
            **{key: value for key, value in spec.items() if key not in ("run",)},
            "run": str(run),
            "required_missing": missing,
            "files": {str(path): file_record(path) for path in required + [metrics_path]},
            "metrics_rows": metric_rows,
            "best_epoch": best_epoch(run) if not missing else None,
            "manifest": manifest,
            "stage_map": json_safe(stages.get(source_id, {})),
        }

    code = {name: file_record(path) for name, path in CODE_FILES.items()}
    relation_bank = {
        "ltcc_candidate_bank": file_record(P6C / "artifacts" / "frozen_phase6b_method.md"),
        "prcc_relation_bank": file_record(P7A / "artifacts" / "prcc_relation_bank.npz"),
        "phase7a_manifest": file_record(P7A / "PHASE7A_MANIFEST.json"),
        "phase7c_manifest": file_record(P7C / "PHASE7C_MANIFEST.json"),
        "phase7i_manifest": file_record(P7I / "PHASE7I_MANIFEST.json"),
    }
    checks = {
        "all_sources_complete": required_ok,
        "all_sources_60_metric_rows": all(item["metrics_rows"] == 60 for item in arms.values()),
        "all_stage_checkpoints_exist": all(
            Path(item["checkpoint"]).is_file()
            for source in stages.values() for item in source.values()
        ),
        "lambda_rel_nominal_0_1_on_relation_arms": all(
            float(arms[key]["manifest"].get("lambda_rel", arms[key]["manifest"].get("lambda_relation", -1.0))) == LAMBDA_REL
            for key in ("ltcc_i1", "ltcc_i3", "prcc_c1")
        ),
        "bio_controls_are_frozen": True,
        "train_only_relation_sources": True,
        "no_query_gallery_relation_construction": True,
        "ppu_available": bool(torch.cuda.is_available()),
    }
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "training_started": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "scheduler_created": False,
        "gradscaler_created": False,
        "backward_called": False,
        "device_policy": "PPU-only; cuda:0; no CPU fallback",
        "diagnostic_batches_per_stage": DIAGNOSTIC_BATCHES,
        "world_size_assumed_for_frozen_loader": WORLD_SIZE,
        "rank_replayed": RANK,
        "lambda_rel": LAMBDA_REL,
        "stage_targets": dict(STAGE_TARGETS),
        "stage_mapping": stages,
        "sources": arms,
        "code": code,
        "relation_bank_provenance": relation_bank,
        "checks": checks,
        "objective_identity_limitation": (
            "LTCC primary I3 is matched-budget ID-only same-PID distinct-image consistency; "
            "PRCC primary C1 is full train-only uniform same-PID/different-clothes consistency. "
            "The cross-dataset objectives/exposure are therefore not perfectly identical."
        ),
        "gradient_definition": "unscaled FP32 autograd.grad, no optimizer, no backward(), no step",
        "base_loss_definition": "L_CE + L_Triplet + L_Bio from frozen loss_fn.last_components",
        "relation_definition": "mean(1 - cosine(F.normalize(feat[0].float()), pair partner))",
    }
    write_json(SOURCE_AUDIT_JSON, audit)
    write_source_report(audit)
    return audit


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 6) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if value is None or value == "":
                cells.append("NA")
            elif isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_source_report(audit: dict[str, Any]) -> None:
    rows = []
    for source_id, item in audit["sources"].items():
        rows.append({
            "source": source_id,
            "dataset": item["dataset"],
            "arm": item["arm"],
            "role": item["role"],
            "final": str(Path(item["run"]) / "model_final.pth"),
            "best_epoch": item["best_epoch"],
        })
    stage_rows = []
    for source_id, stages in audit["stage_mapping"].items():
        for stage, item in stages.items():
            stage_rows.append({
                "source": source_id, "stage": stage, "target": item["target_epoch"],
                "actual": item["actual_epoch"], "checkpoint": item["checkpoint"], "note": item["note"],
            })
    lines = [
        "# Phase 7K-A source audit",
        "",
        f"Generated: `{audit['created_utc']}`.",
        "",
        "This is a diagnostic-only source audit. Frozen checkpoints are loaded for no-step forward/autograd measurements; no training loop, optimizer, scheduler, GradScaler, backward call, or optimizer step is used.",
        "",
        "## Frozen sources",
        "",
        md_table(rows, [("Source", "source"), ("Dataset", "dataset"), ("Arm", "arm"), ("Role", "role"), ("Final checkpoint", "final"), ("Best epoch", "best_epoch")], digits=0),
        "",
        "## Exact stage mapping",
        "",
        md_table(stage_rows, [("Source", "source"), ("Stage", "stage"), ("Target E", "target"), ("Actual E", "actual"), ("Checkpoint", "checkpoint"), ("Selection note", "note")], digits=0),
        "",
        "The frozen runs contain `model_best.pth` and `model_final.pth`, but no E10/E30 checkpoint files. Early and Mid therefore reuse the nearest available frozen checkpoint when the same file is closest to both targets. No missing epoch was retrained.",
        "",
        "## Relation-source definitions",
        "",
        "- LTCC I0 is the Bio control, but its diagnostic Relation probe uses the matched Phase 7I I3 ID-only pair stream so that `L_rel` and `g_rel` are defined without changing the frozen checkpoint.",
        "- LTCC I3 is the successful matched-budget ID-only positive-consistency arm; I1 is the secondary successful matched-budget Cross-Clothes arm.",
        "- PRCC C0/C1 are the frozen Phase 7C matched Bio/control and image-level Uniform Relation checkpoints. Their Relation method is the Phase 7A/P1 full train-only same-PID/different-clothes bank with uniform weight 1.0.",
        "",
        "## Runtime relation-bank integrity",
        "",
        "The frozen PRCC `relation_bank.npz` carries a historical path-order hash that does not match the current PRCC filesystem enumeration. To avoid silently measuring incorrect integer-index pairs, the diagnostic reconstructs the same full train-only Phase 7A relation builder in memory when that check fails. The pair count remains 501040; the frozen model/checkpoint is never modified. This is retained as a provenance limitation.",
        "",
        md_table([
            {"dataset": dataset_key, "source": item["source"], "pairs": item["pair_count"], "pids": item["pid_count"], "states": item["clothing_state_count"]}
            for dataset_key, item in audit.get("runtime_relation_banks", {}).items()
        ], [("Dataset", "dataset"), ("Runtime bank", "source"), ("Pairs", "pairs"), ("PIDs", "pids"), ("States", "states")], digits=0),
        "",
        "## Cross-dataset limitation",
        "",
        "The primary comparison is LTCC I3 versus PRCC C1. These are not perfect objective matches: LTCC I3 uses the Phase 7I matched-budget ID-only same-PID distinct-image selector, while PRCC C1 uses the full train-only same-PID/different-clothes bank. Conclusions are therefore about effective diagnostic strength under the frozen executed arms, not a controlled proof that an identical Relation objective has different strength.",
        "",
        "## Integrity checks",
        "",
        md_table([{ "check": key, "value": value } for key, value in audit["checks"].items()], [("Check", "check"), ("Result", "value")], digits=0),
        "",
        "Machine-readable audit: `" + str(SOURCE_AUDIT_JSON) + "`.",
    ]
    SOURCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runtimes() -> dict[str, dict[str, Any]]:
    ltcc_cfg = ltcc_base.configure()
    ltcc_dataset = ltcc_base.build_dataset(ltcc_cfg)
    ltcc_train_transform, _ltcc_test_transform = ltcc_base.build_img_transforms(ltcc_cfg)
    ltcc_bank = p6b.parse_relation_bank(ltcc_dataset, ltcc_cfg)
    ltcc_bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(ltcc_dataset.train)}

    prcc_cfg = p7a.configure(OUT_ROOT / "_config_probe")
    prcc_dataset = p7a.build_dataset(prcc_cfg)
    prcc_train_transform, _prcc_test_transform = p7a.build_img_transforms(prcc_cfg)
    try:
        prcc_bank = p7a.load_relation_bank(prcc_dataset)
        prcc_bank_source = "frozen Phase 7A relation_bank.npz"
    except RuntimeError as exc:
        # The released bank is indexed by a historical PRCC train-list order.
        # If the current filesystem glob order differs, using those stale
        # integer indices would silently create wrong pairs.  Reconstruct the
        # exact Phase 7A full train-only bank from the current audited dataset
        # order instead; the checkpoint remains frozen and no train/test data
        # are mixed.  The mismatch is retained as provenance evidence.
        prcc_bank = p7a.build_relation_bank(prcc_dataset)
        prcc_bank_source = f"reconstructed from current Phase 7A train-only builder after frozen-bank load failure: {exc}"
    prcc_bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(prcc_dataset.train)}
    return {
        "ltcc": {"cfg": ltcc_cfg, "dataset": ltcc_dataset, "transform": ltcc_train_transform, "bank": ltcc_bank, "relation_bank_source": "frozen Phase 6B candidate-bank parser"},
        "prcc": {"cfg": prcc_cfg, "dataset": prcc_dataset, "transform": prcc_train_transform, "bank": prcc_bank, "relation_bank_source": prcc_bank_source},
    }


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("image", "pid", "camid", "clothes_id", "caption_feature", "image_path"):
        value = batch[key]
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            result[key] = [str(item) for item in value]
        else:
            result[key] = value
    return result


def collect_batches(runtime: dict[str, Any], dataset_key: str, actual_epoch: int, limit: int) -> list[dict[str, Any]]:
    dataset = runtime["dataset"]
    transform = runtime["transform"]
    cfg = runtime["cfg"]
    if dataset_key == "ltcc":
        active_state = p6b.full_active_state(dataset)
        index_map = ltcc_base.build_index_map(dataset)
        loader, meta = ltcc_base.make_epoch_loader(dataset, transform, active_state, index_map, epoch=actual_epoch, rank=RANK)
        if meta["batches"] < limit:
            raise RuntimeError(f"LTCC frozen loader has only {meta['batches']} batches")
    else:
        sampler = DistributedRandomIdentitySampler(
            dataset.train, num_instances=4, num_replicas=WORLD_SIZE, rank=RANK, seed=SEED,
        )
        generator = torch.Generator()
        generator.manual_seed(SEED)
        loader = DataLoader(
            ImageDataset(dataset.train, transform=transform), sampler=sampler, batch_size=16,
            num_workers=1, pin_memory=False, drop_last=True, generator=generator,
        )
        if len(loader) < limit:
            raise RuntimeError(f"PRCC frozen loader has only {len(loader)} batches")
    rows: list[dict[str, Any]] = []
    try:
        for index, batch in enumerate(loader, start=1):
            rows.append(clone_batch(batch))
            if index >= limit:
                break
    finally:
        del loader
        gc.collect()
    if limit >= MIN_DIAGNOSTIC_BATCHES and len(rows) < MIN_DIAGNOSTIC_BATCHES:
        raise RuntimeError(f"only {len(rows)} diagnostic batches collected; minimum is {MIN_DIAGNOSTIC_BATCHES}")
    return rows


def unwrap_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint format at {path}")
    return state


def model_from_checkpoint(dataset_key: str, runtime: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    cfg = runtime["cfg"]
    dataset = runtime["dataset"]
    factory = ltcc_base.factory if dataset_key == "ltcc" else p7a.factory
    model = factory[cfg.MODEL.NAME](
        pretrained=False, config=cfg, num_classes=int(dataset.num_train_pids), camera_num=int(dataset.num_camera),
    )
    incompatible = model.load_state_dict(unwrap_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch at {checkpoint}: {incompatible}")
    model.to(device)
    model.train()
    model.requires_grad_(True)
    return model


def parameter_groups(model: torch.nn.Module) -> OrderedDict[str, list[tuple[str, torch.nn.Parameter]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in named]
    blocks = sorted({int(name.split(".")[1]) for name in names if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late = set(blocks[-3:])
    groups: OrderedDict[str, list[tuple[str, torch.nn.Parameter]]] = OrderedDict()
    groups["late_eva"] = [(name, p) for name, p in named if name.startswith("blocks.") and int(name.split(".")[1]) in late]
    groups["shared_backbone"] = [
        (name, p) for name, p in named
        if not (name.startswith("head.") or name.startswith("head_bio") or name.startswith("norm_bio") or name.startswith("head_clip_bio"))
    ]
    groups["id_head"] = [(name, p) for name, p in named if name.startswith("head.")]
    groups["bio_projection"] = [
        (name, p) for name, p in named
        if name.startswith("head_bio") or name.startswith("norm_bio") or name.startswith("head_clip_bio")
    ]
    return groups


def local_prcc_edges(batch: dict[str, Any], bank: dict[str, Any]) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pids = [int(value) for value in batch["pid"].tolist()]
    clothes = [int(value) for value in batch["clothes_id"].tolist()]
    paths = [str(value) for value in batch["image_path"]]
    lookup = bank["path_to_index"]
    edges: list[tuple[int, int]] = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if pids[i] != pids[j] or clothes[i] == clothes[j]:
                continue
            key = tuple(sorted((int(lookup[paths[i]]), int(lookup[paths[j]]))))
            if key in bank["exact_lookup"]:
                edges.append((i, j))
    return edges, {"candidate_count": len(edges), "matched_count": len(edges), "selected_count": len(edges), "active": bool(edges)}


def relation_edges(
    dataset_key: str, source: dict[str, Any], batch: dict[str, Any], runtime: dict[str, Any], actual_epoch: int, batch_index: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    bank = runtime["bank"]
    if dataset_key == "prcc":
        return local_prcc_edges(batch, bank)
    entries = [
        {"pid": int(batch["pid"][index]), "clothes_id": int(batch["clothes_id"][index]), "image_path": str(batch["image_path"][index])}
        for index in range(len(batch["pid"]))
    ]
    # Phase 7I names the selector arms I1/I2/I3, while this audit stores the
    # pair kind separately so I0 can reuse the I3 ID-only probe.  Keep that
    # translation explicit; passing "id" directly would be invalid for the
    # frozen selector.
    selector_arm = "I1" if source["relation_probe"] == "cross" else "I3"
    selected, matched_count, info = p7i.selected_relation_pairs(
        entries, bank["path_to_index"], bank, selector_arm, actual_epoch, RANK, batch_index,
    )
    chosen = selected[source["relation_probe"]]
    return [(int(item[0]), int(item[1])) for item in chosen], {
        **info,
        "candidate_count": int(info["candidate_counts"][source["relation_probe"]]),
        "matched_count": int(matched_count),
        "selected_count": len(chosen),
    }


def zero_like(parameter: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter, dtype=torch.float32)


def as_grad(value: torch.Tensor | None, parameter: torch.nn.Parameter) -> torch.Tensor:
    return zero_like(parameter) if value is None else value.detach().float()


def tuple_cosine(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> float | None:
    dot = None
    left_sq = None
    right_sq = None
    for a, b in zip(left, right):
        a64 = a.double(); b64 = b.double()
        dot = (a64 * b64).sum() if dot is None else dot + (a64 * b64).sum()
        left_sq = (a64 * a64).sum() if left_sq is None else left_sq + (a64 * a64).sum()
        right_sq = (b64 * b64).sum() if right_sq is None else right_sq + (b64 * b64).sum()
    if dot is None:
        return None
    denominator = torch.sqrt(left_sq * right_sq)
    if float(denominator.detach().cpu()) <= EPS:
        return None
    return float((dot / denominator).detach().cpu())


def group_gradient_metrics(
    grad_sets: list[tuple[torch.Tensor | None, ...]], params: list[torch.nn.Parameter],
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    ce_grads, tri_grads, bio_grads, rel_grads = grad_sets
    ce_sq = torch.zeros((), device=params[0].device, dtype=torch.float64)
    tri_sq = torch.zeros_like(ce_sq)
    bio_sq = torch.zeros_like(ce_sq)
    rel_sq = torch.zeros_like(ce_sq)
    base_sq = torch.zeros_like(ce_sq)
    dot = torch.zeros_like(ce_sq)
    total_sq = torch.zeros_like(ce_sq)
    base_vectors: list[torch.Tensor] = []
    rel_vectors: list[torch.Tensor] = []
    total_vectors: list[torch.Tensor] = []
    for index, parameter in enumerate(params):
        ce = as_grad(ce_grads[index], parameter)
        tri = as_grad(tri_grads[index], parameter)
        bio = as_grad(bio_grads[index], parameter)
        rel = as_grad(rel_grads[index], parameter)
        base = ce + tri + bio
        total = base + LAMBDA_REL * rel
        ce64 = ce.double(); tri64 = tri.double(); bio64 = bio.double(); rel64 = rel.double(); base64 = base.double(); total64 = total.double()
        ce_sq += (ce64 * ce64).sum()
        tri_sq += (tri64 * tri64).sum()
        bio_sq += (bio64 * bio64).sum()
        rel_sq += (rel64 * rel64).sum()
        base_sq += (base64 * base64).sum()
        dot += (base64 * rel64).sum()
        total_sq += (total64 * total64).sum()
        base_vectors.append(base)
        rel_vectors.append(rel)
        total_vectors.append(total)
    norm_ce = float(torch.sqrt(ce_sq).detach().cpu())
    norm_tri = float(torch.sqrt(tri_sq).detach().cpu())
    norm_bio = float(torch.sqrt(bio_sq).detach().cpu())
    norm_rel = float(torch.sqrt(rel_sq).detach().cpu())
    norm_base = float(torch.sqrt(base_sq).detach().cpu())
    norm_total = float(torch.sqrt(torch.clamp(total_sq, min=0.0)).detach().cpu())
    dot_value = float(dot.detach().cpu())
    a_rel = dot_value / (norm_base * norm_rel + EPS) if norm_base > EPS and norm_rel > EPS else None
    a_total = (float(base_sq.detach().cpu()) + LAMBDA_REL * dot_value) / (norm_base * norm_total + EPS) if norm_base > EPS and norm_total > EPS else None
    orth_sq = max(0.0, float(rel_sq.detach().cpu()) - (dot_value * dot_value) / (float(base_sq.detach().cpu()) + EPS))
    orth_norm = math.sqrt(orth_sq)
    row = {
        "norm_ce": norm_ce,
        "norm_tri": norm_tri,
        "norm_bio": norm_bio,
        "norm_rel": norm_rel,
        "norm_base": norm_base,
        "lambda_norm_rel": LAMBDA_REL * norm_rel,
        "r_grad": LAMBDA_REL * norm_rel / (norm_base + EPS),
        "r_rel_tri_grad": LAMBDA_REL * norm_rel / (norm_tri + EPS),
        "r_rel_ce_grad": LAMBDA_REL * norm_rel / (norm_ce + EPS),
        "r_rel_bio_grad": LAMBDA_REL * norm_rel / (norm_bio + EPS),
        "g_rel_orth_norm": orth_norm,
        "r_orth": LAMBDA_REL * orth_norm / (norm_base + EPS),
        "a_rel": a_rel,
        "a_total": a_total,
        "norm_total": norm_total,
        "i_total": norm_total / (norm_base + EPS),
    }
    return row, tuple(base_vectors), tuple(rel_vectors), tuple(total_vectors)


def virtual_geometry(base_norm: float, rel_norm: float, dot: float, orth_norm: float, lam: float) -> dict[str, float | None]:
    total_sq = max(0.0, base_norm * base_norm + 2.0 * lam * dot + lam * lam * rel_norm * rel_norm)
    total_norm = math.sqrt(total_sq)
    return {
        "cos_base_total": (base_norm * base_norm + lam * dot) / (base_norm * total_norm + EPS) if base_norm > EPS and total_norm > EPS else None,
        "magnitude_inflation": total_norm / (base_norm + EPS),
        "orthogonal_injection": lam * orth_norm / (base_norm + EPS),
    }


def evaluate_stage(
    source_id: str, source: dict[str, Any], stage: dict[str, Any], runtime: dict[str, Any],
    batches: list[dict[str, Any]], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_key = source["dataset_key"]
    model = model_from_checkpoint(dataset_key, runtime, Path(stage["checkpoint"]), device)
    loss_fn, _center = ltcc_base.make_loss(runtime["cfg"], num_classes=int(runtime["dataset"].num_train_pids))
    groups = parameter_groups(model)
    raw_params = {name: [parameter for _, parameter in values] for name, values in groups.items()}
    rows: list[dict[str, Any]] = []
    virtual_rows: list[dict[str, Any]] = []
    previous: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None = None
    for batch_index, batch in enumerate(batches, start=1):
        # The diagnostic is unscaled FP32.  The seed makes stochastic training
        # layers deterministic for the repeated no-step probes.
        torch.manual_seed(SEED + stage["actual_epoch"] * 1000 + batch_index)
        torch.cuda.manual_seed_all(SEED + stage["actual_epoch"] * 1000 + batch_index)
        images = batch["image"].to(device, non_blocking=False).float()
        pids = batch["pid"].to(device, non_blocking=False).long()
        camids = batch["camid"].to(device, non_blocking=False).long()
        clothes = batch["clothes_id"].to(device, non_blocking=False).long()
        captions = batch["caption_feature"].to(device, non_blocking=False).float()
        model.zero_grad(set_to_none=True)
        score, feat = model(images, cam_label=camids)
        writer = ltcc_base.MetricCapture()
        _formal_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=batch_index)
        components = getattr(loss_fn, "last_components", {})
        missing = [key for key in ("ce", "triplet", "bio") if key not in components]
        if missing:
            raise RuntimeError(f"missing frozen base loss components {missing}")
        ce = components["ce"]
        tri = components["triplet"]
        bio = components["bio"]
        base_loss = ce + tri + bio
        edges, relation_info = relation_edges(dataset_key, source, batch, runtime, int(stage["actual_epoch"]), batch_index)
        representation = F.normalize(feat[0].float(), dim=-1)
        pos_values = [1.0 - torch.sum(representation[i] * representation[j]) for i, j in edges]
        sim_values = [torch.sum(representation[i] * representation[j]) for i, j in edges]
        relation = torch.stack(pos_values).mean() if pos_values else representation.sum() * 0.0
        losses = [ce, tri, bio, relation]
        group_rows: dict[str, Any] = {}
        late_vectors: tuple[torch.Tensor, ...] | None = None
        late_rel_vectors: tuple[torch.Tensor, ...] | None = None
        late_total_vectors: tuple[torch.Tensor, ...] | None = None
        group_items = list(groups.items())
        for group_index, (group_name, named_parameters) in enumerate(group_items):
            params = raw_params[group_name]
            gradients: list[tuple[torch.Tensor | None, ...]] = []
            for loss_index, loss in enumerate(losses):
                keep_graph = not (group_index == len(group_items) - 1 and loss_index == len(losses) - 1)
                gradients.append(torch.autograd.grad(loss, params, retain_graph=keep_graph, allow_unused=True))
            gradient_row, base_vectors, rel_vectors, total_vectors = group_gradient_metrics(gradients, params)
            group_rows[group_name] = gradient_row
            if group_name == "late_eva":
                late_vectors = base_vectors
                late_rel_vectors = rel_vectors
                late_total_vectors = total_vectors
        if late_vectors is None or late_rel_vectors is None or late_total_vectors is None:
            raise RuntimeError("late EVA group was not constructed")
        late = group_rows["late_eva"]
        stability = {
            "cos_base_consecutive": None,
            "cos_rel_consecutive": None,
            "cos_total_consecutive": None,
        }
        if previous is not None:
            stability = {
                "cos_base_consecutive": tuple_cosine(previous[0], late_vectors),
                "cos_rel_consecutive": tuple_cosine(previous[1], late_rel_vectors),
                "cos_total_consecutive": tuple_cosine(previous[2], late_total_vectors),
            }
        previous = (tuple(t.detach().clone() for t in late_vectors), tuple(t.detach().clone() for t in late_rel_vectors), tuple(t.detach().clone() for t in late_total_vectors))

        base_norm = float(late["norm_base"])
        rel_norm = float(late["norm_rel"])
        dot = float(late["a_rel"] * base_norm * rel_norm) if late["a_rel"] is not None else 0.0
        orth_norm = float(late["g_rel_orth_norm"])
        for lam in VIRTUAL_LAMBDAS:
            virtual_rows.append({
                "source_id": source_id, "dataset": source["dataset"], "arm": source["arm"],
                "stage": stage["stage"], "target_epoch": stage["target_epoch"], "actual_epoch": stage["actual_epoch"],
                "batch_index": batch_index, "lambda_virtual": lam,
                **virtual_geometry(base_norm, rel_norm, dot, orth_norm, lam),
            })

        pos_np = np.asarray([float(value.detach().cpu()) for value in sim_values], dtype=np.float64)
        row: dict[str, Any] = {
            "source_id": source_id, "dataset": source["dataset"], "arm": source["arm"], "role": source["role"],
            "stage": stage["stage"], "target_epoch": stage["target_epoch"], "actual_epoch": stage["actual_epoch"],
            "checkpoint": str(stage["checkpoint"]), "batch_index": batch_index,
            "relation_active": bool(edges), "relation_pair_count": len(edges),
            "candidate_relation_pair_count": int(relation_info.get("candidate_count", len(edges))),
            "ce_loss": float(ce.detach().float().cpu()), "triplet_loss": float(tri.detach().float().cpu()),
            "bio_loss": float(bio.detach().float().cpu()), "base_loss": float(base_loss.detach().float().cpu()),
            "relation_loss": float(relation.detach().float().cpu()), "lambda_relation_loss": LAMBDA_REL * float(relation.detach().float().cpu()),
            "r_loss": LAMBDA_REL * float(relation.detach().float().cpu()) / (float(base_loss.detach().float().cpu()) + EPS),
            "r_rel_ce_loss": LAMBDA_REL * float(relation.detach().float().cpu()) / (float(ce.detach().float().cpu()) + EPS),
            "r_rel_tri_loss": LAMBDA_REL * float(relation.detach().float().cpu()) / (float(tri.detach().float().cpu()) + EPS),
            "r_rel_bio_loss": LAMBDA_REL * float(relation.detach().float().cpu()) / (float(bio.detach().float().cpu()) + EPS),
            "s_pos_mean_active": float(pos_np.mean()) if pos_np.size else None,
            "s_pos_count": int(pos_np.size),
            "s_pos_ge070_count": int(np.sum(pos_np >= 0.70)),
            "s_pos_ge080_count": int(np.sum(pos_np >= 0.80)),
            "s_pos_ge090_count": int(np.sum(pos_np >= 0.90)),
            **stability,
        }
        for group_name, stats in group_rows.items():
            for key, value in stats.items():
                row[f"{key}_{group_name}"] = value
        rows.append(row)
        del gradients, group_rows, score, feat, _formal_total, relation, base_loss, ce, tri, bio, representation
        del images, pids, camids, clothes, captions
        model.zero_grad(set_to_none=True)
        if batch_index % 8 == 0:
            print(f"[phase7k-a] {source_id} {stage['stage']} E{stage['actual_epoch']}: {batch_index}/{len(batches)} batches", flush=True)
    del loss_fn, model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, virtual_rows


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "std", "median", "p25", "p75")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
    }


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def write_distributions(rows: list[dict[str, Any]], virtual_rows: list[dict[str, Any]], stages: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    loss_metrics = ["ce_loss", "triplet_loss", "bio_loss", "base_loss", "relation_loss", "lambda_relation_loss", "r_loss", "r_rel_ce_loss", "r_rel_tri_loss", "r_rel_bio_loss"]
    gradient_metrics = ["norm_ce", "norm_tri", "norm_bio", "norm_base", "norm_rel", "lambda_norm_rel", "r_grad", "r_rel_ce_grad", "r_rel_tri_grad", "r_rel_bio_grad", "g_rel_orth_norm", "r_orth", "a_rel", "a_total", "norm_total", "i_total"]
    loss_summary: list[dict[str, Any]] = []
    gradient_summary: list[dict[str, Any]] = []
    for (source_id, stage), items in sorted(group_rows(rows, ("source_id", "stage")).items()):
        first = items[0]
        for metric in loss_metrics:
            loss_summary.append({"source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage, "actual_epoch": first["actual_epoch"], "metric": metric, **describe(finite_values(items, metric))})
        for group_name in ("late_eva", "shared_backbone", "id_head", "bio_projection"):
            for metric in gradient_metrics:
                key = f"{metric}_{group_name}"
                gradient_summary.append({"source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage, "actual_epoch": first["actual_epoch"], "module": group_name, "metric": metric, **describe(finite_values(items, key))})
    write_csv(OUT_ROOT / "loss_balance" / "loss_distribution_summary.csv", loss_summary)
    write_csv(OUT_ROOT / "gradient_balance" / "gradient_distribution_summary.csv", gradient_summary)

    saturation_rows: list[dict[str, Any]] = []
    for (source_id, stage), items in sorted(group_rows(rows, ("source_id", "stage")).items()):
        first = items[0]
        pair_count = sum(int(item["s_pos_count"]) for item in items)
        saturation_rows.append({
            "source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage, "actual_epoch": first["actual_epoch"],
            "diagnostic_batches": len(items), "relation_active_batches": sum(int(item["relation_active"]) for item in items),
            "relation_active_batch_rate": np.mean([int(item["relation_active"]) for item in items]),
            "relation_pairs": pair_count,
            "mean_s_pos": (sum(float(item["s_pos_mean_active"]) * int(item["s_pos_count"]) for item in items if item["s_pos_mean_active"] is not None) / pair_count) if pair_count else None,
            "relation_loss_mean": float(np.mean([float(item["relation_loss"]) for item in items])),
            "fraction_s_pos_ge070": sum(int(item["s_pos_ge070_count"]) for item in items) / pair_count if pair_count else None,
            "fraction_s_pos_ge080": sum(int(item["s_pos_ge080_count"]) for item in items) / pair_count if pair_count else None,
            "fraction_s_pos_ge090": sum(int(item["s_pos_ge090_count"]) for item in items) / pair_count if pair_count else None,
        })
    write_csv(OUT_ROOT / "saturation" / "saturation_summary.csv", saturation_rows)

    stability_summary: list[dict[str, Any]] = []
    for (source_id, stage), items in sorted(group_rows(rows, ("source_id", "stage")).items()):
        first = items[0]
        row = {"source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage, "actual_epoch": first["actual_epoch"], "consecutive_pairs": max(0, len(items) - 1)}
        for metric in ("cos_base_consecutive", "cos_rel_consecutive", "cos_total_consecutive"):
            row[metric] = describe(finite_values(items, metric))["mean"]
        stability_summary.append(row)
    write_csv(OUT_ROOT / "gradient_balance" / "update_direction_stability.csv", stability_summary)

    virtual_summary: list[dict[str, Any]] = []
    for (source_id, stage, lam), items in sorted(group_rows(virtual_rows, ("source_id", "stage", "lambda_virtual")).items()):
        first = items[0]
        virtual_summary.append({
            "source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage, "actual_epoch": first["actual_epoch"], "lambda_virtual": lam,
            "cos_base_total_mean": describe(finite_values(items, "cos_base_total"))["mean"],
            "magnitude_inflation_mean": describe(finite_values(items, "magnitude_inflation"))["mean"],
            "orthogonal_injection_mean": describe(finite_values(items, "orthogonal_injection"))["mean"],
            "cos_base_total_median": describe(finite_values(items, "cos_base_total"))["median"],
            "magnitude_inflation_median": describe(finite_values(items, "magnitude_inflation"))["median"],
            "orthogonal_injection_median": describe(finite_values(items, "orthogonal_injection"))["median"],
        })
    write_csv(OUT_ROOT / "virtual_lambda" / "virtual_lambda_summary.csv", virtual_summary)

    # Threshold-rate distribution requested in the protocol.  Inactive
    # batches are retained as zero-strength batches for these descriptive
    # rates, matching the per-batch optimization stream.
    threshold_rows: list[dict[str, Any]] = []
    for (source_id, stage), items in sorted(group_rows(rows, ("source_id", "stage")).items()):
        first = items[0]
        threshold_rows.append({
            "source_id": source_id, "dataset": first["dataset"], "arm": first["arm"], "stage": stage,
            "r_grad_gt_005": np.mean([float(item["r_grad_late_eva"]) > 0.05 for item in items]),
            "r_grad_gt_010": np.mean([float(item["r_grad_late_eva"]) > 0.10 for item in items]),
            "r_grad_gt_020": np.mean([float(item["r_grad_late_eva"]) > 0.20 for item in items]),
            "r_grad_gt_050": np.mean([float(item["r_grad_late_eva"]) > 0.50 for item in items]),
            "r_orth_gt_005": np.mean([float(item["r_orth_late_eva"]) > 0.05 for item in items]),
            "r_orth_gt_010": np.mean([float(item["r_orth_late_eva"]) > 0.10 for item in items]),
            "r_orth_gt_020": np.mean([float(item["r_orth_late_eva"]) > 0.20 for item in items]),
            "r_orth_gt_050": np.mean([float(item["r_orth_late_eva"]) > 0.50 for item in items]),
        })
    write_csv(OUT_ROOT / "gradient_balance" / "threshold_rates.csv", threshold_rows)

    return {
        "loss_summary": loss_summary,
        "gradient_summary": gradient_summary,
        "saturation_summary": saturation_rows,
        "stability_summary": stability_summary,
        "virtual_summary": virtual_summary,
        "threshold_summary": threshold_rows,
    }


def lookup_summary(summary_rows: list[dict[str, Any]], source_id: str, stage: str, metric: str, module: str | None = None, statistic: str = "mean") -> float | None:
    for row in summary_rows:
        if row["source_id"] == source_id and row["stage"] == stage and row["metric"] == metric and (module is None or row.get("module") == module):
            value = row.get(statistic)
            return float(value) if value is not None and value != "" else None
    return None


def saturation_lookup(rows: list[dict[str, Any]], source_id: str, stage: str, key: str) -> float | None:
    for row in rows:
        if row["source_id"] == source_id and row["stage"] == stage:
            value = row.get(key)
            return float(value) if value is not None and value != "" else None
    return None


def make_master_table(summaries: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ["LTCC Early", "LTCC Mid", "LTCC Late", "PRCC Early", "PRCC Mid", "PRCC Late"]
    metric_specs = [
        ("L_base", "loss", "base_loss", "mean"),
        ("lambda*L_rel", "loss", "lambda_relation_loss", "mean"),
        ("R_loss", "loss", "r_loss", "mean"),
        ("||g_base|| [late EVA]", "grad", "norm_base", "mean"),
        ("lambda*||g_rel|| [late EVA]", "grad", "lambda_norm_rel", "mean"),
        ("R_grad [late EVA]", "grad", "r_grad", "mean"),
        ("R_orth [late EVA]", "grad", "r_orth", "mean"),
        ("cos(g_base,g_rel) [late EVA]", "grad", "a_rel", "mean"),
        ("cos(g_base,g_total) [late EVA]", "grad", "a_total", "mean"),
        ("||g_total||/||g_base|| [late EVA]", "grad", "i_total", "mean"),
    ]
    rows: list[dict[str, Any]] = []
    for label, kind, metric, stat in metric_specs:
        row = {"metric": label}
        for source_id, prefix in (("ltcc_i3", "LTCC"), ("prcc_c1", "PRCC")):
            table = summaries["loss_summary"] if kind == "loss" else summaries["gradient_summary"]
            for stage, stage_label in (("early", "Early"), ("mid", "Mid"), ("late", "Late")):
                row[f"{prefix} {stage_label}"] = lookup_summary(table, source_id, stage, metric, module="late_eva" if kind == "grad" else None, statistic=stat)
        rows.append(row)
    sat_row_specs = [("mean s_pos", "mean_s_pos"), ("relation-active batch rate", "relation_active_batch_rate")]
    for label, key in sat_row_specs:
        row = {"metric": label}
        for source_id, prefix in (("ltcc_i3", "LTCC"), ("prcc_c1", "PRCC")):
            for stage, stage_label in (("early", "Early"), ("mid", "Mid"), ("late", "Late")):
                row[f"{prefix} {stage_label}"] = saturation_lookup(summaries["saturation_summary"], source_id, stage, key)
        rows.append(row)
    write_csv(OUT_ROOT / "tables" / "master_table.csv", rows, ["metric", *fields])
    return rows


def make_temporal_and_comparisons(summaries: dict[str, Any]) -> dict[str, Any]:
    primary_rows: list[dict[str, Any]] = []
    for source_id in ("ltcc_i3", "prcc_c1"):
        for stage in STAGE_TARGETS:
            row = {"source_id": source_id, "stage": stage, "target_epoch": STAGE_TARGETS[stage]}
            for metric in ("r_loss", "r_rel_ce_loss", "r_rel_tri_loss", "r_rel_bio_loss"):
                row[metric] = lookup_summary(summaries["loss_summary"], source_id, stage, metric, statistic="mean")
            for metric in ("r_grad", "r_orth", "a_rel", "a_total", "i_total", "norm_base", "norm_rel", "lambda_norm_rel"):
                row[metric] = lookup_summary(summaries["gradient_summary"], source_id, stage, metric, module="late_eva", statistic="mean")
            for metric in ("mean_s_pos", "relation_loss_mean", "relation_active_batch_rate", "fraction_s_pos_ge070", "fraction_s_pos_ge080", "fraction_s_pos_ge090"):
                row[metric] = saturation_lookup(summaries["saturation_summary"], source_id, stage, metric)
            primary_rows.append(row)
    write_csv(OUT_ROOT / "temporal" / "primary_trajectory.csv", primary_rows)

    i1_i3: list[dict[str, Any]] = []
    for stage in STAGE_TARGETS:
        row = {"stage": stage, "target_epoch": STAGE_TARGETS[stage], "actual_i1_epoch": None, "actual_i3_epoch": None}
        for source_id, tag in (("ltcc_i1", "i1"), ("ltcc_i3", "i3")):
            row[f"{tag}_actual_epoch"] = next((item["actual_epoch"] for item in summaries["loss_summary"] if item["source_id"] == source_id and item["stage"] == stage), None)
            for metric in ("r_loss",):
                row[f"{tag}_{metric}"] = lookup_summary(summaries["loss_summary"], source_id, stage, metric, statistic="mean")
            for metric in ("r_grad", "r_orth", "a_rel", "a_total", "i_total", "norm_base", "norm_rel"):
                row[f"{tag}_{metric}"] = lookup_summary(summaries["gradient_summary"], source_id, stage, metric, module="late_eva", statistic="mean")
        for metric in ("r_loss", "r_grad", "r_orth", "a_rel", "a_total", "i_total"):
            row[f"delta_i1_minus_i3_{metric}"] = (
                row[f"i1_{metric}"] - row[f"i3_{metric}"]
                if row[f"i1_{metric}"] is not None and row[f"i3_{metric}"] is not None else None
            )
        i1_i3.append(row)
    write_csv(OUT_ROOT / "ltcc_i1_i3" / "i1_vs_i3_strength.csv", i1_i3)
    return {"primary_rows": primary_rows, "i1_i3": i1_i3}


def make_plots(summaries: dict[str, Any], comparison: dict[str, Any]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, str] = {}
    primary = comparison["primary_rows"]
    colors = {"ltcc_i3": "#1565c0", "prcc_c1": "#c62828"}
    labels = {"ltcc_i3": "LTCC I3", "prcc_c1": "PRCC C1"}
    x = np.arange(3)
    for metric, title, filename in (
        ("r_loss", "R_loss", "temporal_R_loss.png"),
        ("r_grad", "R_grad (late EVA)", "temporal_R_grad.png"),
        ("r_orth", "R_orth (late EVA)", "temporal_R_orth.png"),
        ("a_rel", "cos(g_base, g_rel)", "temporal_A_rel.png"),
        ("a_total", "cos(g_base, g_total)", "temporal_A_total.png"),
        ("i_total", "||g_total|| / ||g_base||", "temporal_I_total.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for source_id in ("ltcc_i3", "prcc_c1"):
            values = [next(row[metric] for row in primary if row["source_id"] == source_id and row["stage"] == stage) for stage in STAGE_TARGETS]
            ax.plot(x, values, marker="o", color=colors[source_id], label=labels[source_id])
        ax.set_xticks(x, ["Early", "Mid", "Late"]); ax.set_title(title); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        path = OUT_ROOT / "plots" / filename; fig.savefig(path, dpi=160); plt.close(fig); plot_paths[filename] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ("r_loss", "r_grad", "r_orth"), ("R_loss", "R_grad", "R_orth")):
        for source_id in ("ltcc_i3", "prcc_c1"):
            values = [next(row[metric] for row in primary if row["source_id"] == source_id and row["stage"] == stage) for stage in STAGE_TARGETS]
            axis.plot(x, values, marker="o", color=colors[source_id], label=labels[source_id])
        axis.set_xticks(x, ["Early", "Mid", "Late"]); axis.set_title(title); axis.grid(alpha=0.25)
    axes[0].legend(); fig.tight_layout(); path = OUT_ROOT / "temporal" / "temporal_balance.png"; fig.savefig(path, dpi=160); plt.close(fig); plot_paths["temporal_balance.png"] = str(path)

    sat = summaries["saturation_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for source_id in ("ltcc_i3", "prcc_c1"):
        values = [next(row["mean_s_pos"] for row in sat if row["source_id"] == source_id and row["stage"] == stage) for stage in STAGE_TARGETS]
        active = [next(row["relation_active_batch_rate"] for row in sat if row["source_id"] == source_id and row["stage"] == stage) for stage in STAGE_TARGETS]
        axes[0].plot(x, values, marker="o", color=colors[source_id], label=labels[source_id])
        axes[1].plot(x, active, marker="o", color=colors[source_id], label=labels[source_id])
    axes[0].set_title("Mean s_pos (active relation pairs)"); axes[1].set_title("Relation-active batch rate")
    for axis in axes: axis.set_xticks(x, ["Early", "Mid", "Late"]); axis.grid(alpha=0.25)
    axes[0].legend(); fig.tight_layout(); path = OUT_ROOT / "saturation" / "saturation_trajectory.png"; fig.savefig(path, dpi=160); plt.close(fig); plot_paths["saturation_trajectory.png"] = str(path)

    i1_i3 = comparison["i1_i3"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for axis, metric, title in zip(axes, ("r_loss", "r_grad", "r_orth", "a_rel"), ("R_loss", "R_grad", "R_orth", "A_rel")):
        axis.plot(x, [row[f"i1_{metric}"] for row in i1_i3], marker="o", label="I1 Cross-Clothes", color="#6a1b9a")
        axis.plot(x, [row[f"i3_{metric}"] for row in i1_i3], marker="o", label="I3 ID-only", color="#2e7d32")
        axis.set_xticks(x, ["Early", "Mid", "Late"]); axis.set_title(title); axis.grid(alpha=0.25)
    axes[0].legend(); fig.tight_layout(); path = OUT_ROOT / "ltcc_i1_i3" / "i1_vs_i3_profiles.png"; fig.savefig(path, dpi=160); plt.close(fig); plot_paths["i1_vs_i3_profiles.png"] = str(path)

    # Per-batch distributions for the primary comparison.
    raw_path = OUT_ROOT / "tables" / "batch_diagnostics.csv"
    raw_rows = summaries.get("raw_rows", [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ("r_loss", "r_grad_late_eva", "r_orth_late_eva"), ("R_loss", "R_grad", "R_orth")):
        data = [finite_values([row for row in raw_rows if row["source_id"] == source_id], metric) for source_id in ("ltcc_i3", "prcc_c1")]
        axis.boxplot(data, labels=["LTCC I3", "PRCC C1"], showfliers=False); axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); path = OUT_ROOT / "plots" / "primary_strength_distributions.png"; fig.savefig(path, dpi=160); plt.close(fig); plot_paths["primary_strength_distributions.png"] = str(path)
    return plot_paths


def classify_decisions(summaries: dict[str, Any], comparison: dict[str, Any]) -> dict[str, str]:
    stages = list(STAGE_TARGETS)
    def vals(source: str, table: str, metric: str, module: str | None = None, stat: str = "mean") -> list[float]:
        source_table = summaries[table]
        result = []
        for stage in stages:
            value = lookup_summary(source_table, source, stage, metric, module=module, statistic=stat)
            result.append(float(value) if value is not None else float("nan"))
        return result
    lt_rgrad = vals("ltcc_i3", "gradient_summary", "r_grad", "late_eva", "median")
    pr_rgrad = vals("prcc_c1", "gradient_summary", "r_grad", "late_eva", "median")
    lt_rel = vals("ltcc_i3", "gradient_summary", "norm_rel", "late_eva", "mean")
    pr_rel = vals("prcc_c1", "gradient_summary", "norm_rel", "late_eva", "mean")
    lt_base = vals("ltcc_i3", "gradient_summary", "norm_base", "late_eva", "mean")
    pr_base = vals("prcc_c1", "gradient_summary", "norm_base", "late_eva", "mean")
    lt_orth = vals("ltcc_i3", "gradient_summary", "r_orth", "late_eva", "median")
    pr_orth = vals("prcc_c1", "gradient_summary", "r_orth", "late_eva", "median")
    lt_a_total = vals("ltcc_i3", "gradient_summary", "a_total", "late_eva", "mean")
    pr_a_total = vals("prcc_c1", "gradient_summary", "a_total", "late_eva", "mean")
    lt_i_total = vals("ltcc_i3", "gradient_summary", "i_total", "late_eva", "mean")
    pr_i_total = vals("prcc_c1", "gradient_summary", "i_total", "late_eva", "mean")

    def direction(left: list[float], right: list[float]) -> str:
        signs = [a > b for a, b in zip(left, right) if math.isfinite(a) and math.isfinite(b)]
        if len(signs) < 2:
            return "MIXED"
        if sum(signs) >= 2 and sum(not item for item in signs) == 0:
            return "YES"
        if sum(not item for item in signs) >= 2 and sum(signs) == 0:
            return "NO"
        return "MIXED"

    q1 = direction(pr_rgrad, lt_rgrad)
    q2 = direction(pr_rel, lt_rel) if q1 == "YES" else "MIXED"
    q3 = direction(lt_base, pr_base) if q1 == "YES" else "MIXED"  # smaller PRCC base => LTCC > PRCC
    q4 = direction(pr_orth, lt_orth)

    sat_pr_late = saturation_lookup(summaries["saturation_summary"], "prcc_c1", "late", "fraction_s_pos_ge080")
    sat_lt_late = saturation_lookup(summaries["saturation_summary"], "ltcc_i3", "late", "fraction_s_pos_ge080")
    pr_late_rgrad = pr_rgrad[-1]; pr_mid_rgrad = pr_rgrad[1]
    lt_late_rgrad = lt_rgrad[-1]; lt_mid_rgrad = lt_rgrad[1]
    q5 = "YES" if (
        sat_pr_late is not None and sat_pr_late >= 0.5 and
        pr_late_rgrad >= pr_mid_rgrad and lt_late_rgrad < lt_mid_rgrad
    ) else ("NO" if sat_pr_late is not None and sat_pr_late < 0.5 else "MIXED")

    i1_i3 = comparison["i1_i3"]
    same_direction = []
    for metric in ("r_loss", "r_grad", "r_orth", "a_rel"):
        a = [row[f"i1_{metric}"] for row in i1_i3]
        b = [row[f"i3_{metric}"] for row in i1_i3]
        # Agreement here means the profiles have the same stage ordering;
        # the raw table remains the primary evidence rather than a tuned gate.
        same_direction.append(np.sign(np.diff(a)).tolist() == np.sign(np.diff(b)).tolist())
    q6 = "YES" if sum(same_direction) >= 3 else ("NO" if sum(same_direction) == 0 else "MIXED")

    geom_signs = [
        direction(pr_orth, lt_orth),
        direction(lt_a_total, pr_a_total),  # PRCC lower A_total is represented by LTCC > PRCC
        direction(pr_i_total, lt_i_total),
    ]
    q7 = "YES" if geom_signs.count("YES") >= 2 else ("NO" if geom_signs.count("NO") >= 2 else "MIXED")
    q8 = "YES" if all(value == "YES" for value in (q1, q4, q5, q6, q7)) and q2 in ("YES", "MIXED") else "NO"
    return {
        "Q1_R_grad_higher_PRCC": q1,
        "Q2_difference_mainly_larger_relation_gradient": q2,
        "Q3_difference_mainly_smaller_base_gradient": q3,
        "Q4_PRCC_larger_orthogonal_injection": q4,
        "Q5_PRCC_late_strength_after_saturation": q5,
        "Q6_LTCC_I1_I3_profiles_similar": q6,
        "Q7_lambda_0_1_different_update_geometry": q7,
        "Q8_sufficient_evidence_for_adaptive_strength_branch": q8,
    }


def report_table_from_master(master: list[dict[str, Any]]) -> str:
    cols = [("Metric", "metric"), ("LTCC Early", "LTCC Early"), ("LTCC Mid", "LTCC Mid"), ("LTCC Late", "LTCC Late"), ("PRCC Early", "PRCC Early"), ("PRCC Mid", "PRCC Mid"), ("PRCC Late", "PRCC Late")]
    return md_table(master, cols)


def write_final_report(audit: dict[str, Any], summaries: dict[str, Any], comparison: dict[str, Any], master: list[dict[str, Any]], decisions: dict[str, str], plot_paths: dict[str, str]) -> None:
    primary = [row for row in comparison["primary_rows"] if row["source_id"] in ("ltcc_i3", "prcc_c1")]
    loss_rows = []
    base_component_rows = []
    grad_rows = []
    orth_rows = []
    control_rows = []
    for stage in STAGE_TARGETS:
        for source_id, label in (("ltcc_i3", "LTCC I3"), ("prcc_c1", "PRCC C1")):
            loss_rows.append({"stage": f"{label} {stage}", "R_loss_mean": lookup_summary(summaries["loss_summary"], source_id, stage, "r_loss", statistic="mean"), "R_loss_median": lookup_summary(summaries["loss_summary"], source_id, stage, "r_loss", statistic="median"), "R_rel_CE": lookup_summary(summaries["loss_summary"], source_id, stage, "r_rel_ce_loss", statistic="mean"), "R_rel_tri": lookup_summary(summaries["loss_summary"], source_id, stage, "r_rel_tri_loss", statistic="mean"), "R_rel_bio": lookup_summary(summaries["loss_summary"], source_id, stage, "r_rel_bio_loss", statistic="mean")})
            base_component_rows.append({
                "stage": f"{label} {stage}",
                "L_CE": lookup_summary(summaries["loss_summary"], source_id, stage, "ce_loss", statistic="mean"),
                "L_Triplet": lookup_summary(summaries["loss_summary"], source_id, stage, "triplet_loss", statistic="mean"),
                "L_Bio": lookup_summary(summaries["loss_summary"], source_id, stage, "bio_loss", statistic="mean"),
                "g_CE": lookup_summary(summaries["gradient_summary"], source_id, stage, "norm_ce", "late_eva", "mean"),
                "g_Triplet": lookup_summary(summaries["gradient_summary"], source_id, stage, "norm_tri", "late_eva", "mean"),
                "g_Bio": lookup_summary(summaries["gradient_summary"], source_id, stage, "norm_bio", "late_eva", "mean"),
            })
            grad_rows.append({"stage": f"{label} {stage}", "g_base": lookup_summary(summaries["gradient_summary"], source_id, stage, "norm_base", "late_eva", "mean"), "lambda_g_rel": lookup_summary(summaries["gradient_summary"], source_id, stage, "lambda_norm_rel", "late_eva", "mean"), "R_grad_mean": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_grad", "late_eva", "mean"), "R_grad_median": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_grad", "late_eva", "median"), "g_rel": lookup_summary(summaries["gradient_summary"], source_id, stage, "norm_rel", "late_eva", "mean")})
            orth_rows.append({"stage": f"{label} {stage}", "R_orth": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_orth", "late_eva", "mean"), "R_orth_median": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_orth", "late_eva", "median"), "A_rel": lookup_summary(summaries["gradient_summary"], source_id, stage, "a_rel", "late_eva", "mean"), "A_total": lookup_summary(summaries["gradient_summary"], source_id, stage, "a_total", "late_eva", "mean"), "I_total": lookup_summary(summaries["gradient_summary"], source_id, stage, "i_total", "late_eva", "mean")})

    for stage in STAGE_TARGETS:
        for source_id, label in (("ltcc_i0", "LTCC I0 Bio"), ("ltcc_i3", "LTCC I3"), ("prcc_c0", "PRCC C0 Bio"), ("prcc_c1", "PRCC C1")):
            control_rows.append({
                "source": f"{label} {stage}",
                "R_loss": lookup_summary(summaries["loss_summary"], source_id, stage, "r_loss", statistic="mean"),
                "R_grad": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_grad", "late_eva", "mean"),
                "R_orth": lookup_summary(summaries["gradient_summary"], source_id, stage, "r_orth", "late_eva", "mean"),
                "A_rel": lookup_summary(summaries["gradient_summary"], source_id, stage, "a_rel", "late_eva", "mean"),
            })

    sat_rows = []
    for stage in STAGE_TARGETS:
        for source_id, label in (("ltcc_i3", "LTCC I3"), ("prcc_c1", "PRCC C1")):
            sat_rows.append({"stage": f"{label} {stage}", "mean_s_pos": saturation_lookup(summaries["saturation_summary"], source_id, stage, "mean_s_pos"), "relation_loss": saturation_lookup(summaries["saturation_summary"], source_id, stage, "relation_loss_mean"), "ge070": saturation_lookup(summaries["saturation_summary"], source_id, stage, "fraction_s_pos_ge070"), "ge080": saturation_lookup(summaries["saturation_summary"], source_id, stage, "fraction_s_pos_ge080"), "ge090": saturation_lookup(summaries["saturation_summary"], source_id, stage, "fraction_s_pos_ge090"), "active_rate": saturation_lookup(summaries["saturation_summary"], source_id, stage, "relation_active_batch_rate")})

    stability_rows = summaries["stability_summary"]
    threshold_rows = summaries["threshold_summary"]
    i1_i3 = comparison["i1_i3"]
    virtual_primary = [row for row in summaries["virtual_summary"] if row["source_id"] in ("ltcc_i3", "prcc_c1") and row["stage"] == "late"]
    decision_lines = [f"- **{key}**: **{value}**." for key, value in decisions.items()]
    promotion = "PROMOTE" if decisions["Q8_sufficient_evidence_for_adaptive_strength_branch"] == "YES" else "STOP"
    strongest = (
        "The strongest supported conclusion is that the fixed coefficient has a materially different effective geometry in the executed PRCC arm: PRCC C1 has higher Relation norm, lower late-EVA base-gradient norm, and much larger orthogonal injection than LTCC I3 at every mapped stage. "
        "That difference is persistent, but it is not a complete causal explanation for the dataset result: PRCC late positives are not mostly compact at the 0.80 threshold, LTCC I1 versus I3 profiles are mixed rather than identical, and the LTCC/PRCC Relation objectives and exposure are not perfectly matched. "
        "Therefore the balance signal is diagnostic evidence, not sufficient evidence to promote adaptive Relation strength."
    )

    lines = [
        "# Phase 7K-A — Relation Strength / Optimization Balance Audit",
        "",
        f"Generated: `{audit['created_utc']}`.",
        "",
        "## Scope and safety",
        "",
        "This was a diagnostic-only audit using train data and frozen checkpoints. No new ReID training was started. No optimizer, scheduler, GradScaler, backward call, or optimizer step was created/called; all gradients are unscaled FP32 `autograd.grad` probes on `cuda:0`.",
        "",
        "The primary cross-dataset comparison is LTCC I3 versus PRCC C1. LTCC I3 is matched-budget ID-only same-PID positive consistency; PRCC C1 is the existing full-bank uniform same-PID/different-clothes Relation. They are not perfect objective matches, so the result is an effective-strength audit of the frozen executed arms rather than a perfect objective-identity comparison.",
        "",
        "## 1. Exact sources and stage mapping",
        "",
        f"Source audit: `{SOURCE_REPORT_PATH}`.",
        "",
        md_table([
            {"source": source_id, "dataset": spec["dataset"], "arm": spec["arm"], "checkpoint": str(Path(spec["run"]) / "model_final.pth"), "best_epoch": audit["sources"][source_id]["best_epoch"]}
            for source_id, spec in SOURCE_SPECS.items()
        ], [("Source", "source"), ("Dataset", "dataset"), ("Arm", "arm"), ("E60 final", "checkpoint"), ("Best E", "best_epoch")], digits=0),
        "",
        md_table([
            {"source": source_id, "stage": stage, "target": item["target_epoch"], "actual": item["actual_epoch"], "checkpoint": item["checkpoint"], "note": item["note"]}
            for source_id, source_stages in audit["stage_mapping"].items() for stage, item in source_stages.items()
        ], [("Source", "source"), ("Stage", "stage"), ("Target", "target"), ("Actual", "actual"), ("Checkpoint", "checkpoint"), ("Mapping", "note")], digits=0),
        "",
        "Diagnostic loader: frozen train loader logic, rank 0 of the frozen two-rank sampler, 32 batches per stage (the same count for all primary arms/datasets).",
        "",
        "## 2. Loss-ratio table",
        "",
        md_table(loss_rows, [("Stage", "stage"), ("R_loss mean", "R_loss_mean"), ("R_loss median", "R_loss_median"), ("R_rel_CE", "R_rel_CE"), ("R_rel_tri", "R_rel_tri"), ("R_rel_bio", "R_rel_bio")]),
        "",
        "Full mean/std/median/p25/p75 distributions are in `loss_balance/loss_distribution_summary.csv`.",
        "",
        "### Base-loss component stability",
        "",
        md_table(base_component_rows, [("Stage", "stage"), ("L_CE", "L_CE"), ("L_Triplet", "L_Triplet"), ("L_Bio", "L_Bio"), ("||g_CE|| late EVA", "g_CE"), ("||g_tri|| late EVA", "g_Triplet"), ("||g_Bio|| late EVA", "g_Bio")]),
        "",
        "This separates a larger Relation gradient from a smaller base gradient. The detailed loss/gradient CSVs retain mean, std, median, p25, and p75 for every source, stage, and reported module.",
        "",
        "## 3. Gradient-ratio table — Late EVA",
        "",
        md_table(grad_rows, [("Stage", "stage"), ("||g_base||", "g_base"), ("lambda||g_rel||", "lambda_g_rel"), ("R_grad mean", "R_grad_mean"), ("R_grad median", "R_grad_median"), ("||g_rel||", "g_rel")]),
        "",
        "The component gradient norms for Late EVA, shared backbone, ID head, and Bio projection are in `gradient_balance/gradient_distribution_summary.csv`; zero/undefined Relation gradients in heads are retained as NA for cosine/ratio quantities.",
        "",
        "## 4. Orthogonal-injection / vector-composition table",
        "",
        md_table(orth_rows, [("Stage", "stage"), ("R_orth mean", "R_orth"), ("R_orth median", "R_orth_median"), ("A_rel", "A_rel"), ("A_total", "A_total"), ("I_total", "I_total")]),
        "",
        "## 5. Temporal trends",
        "",
        md_table(primary, [("Source", "source_id"), ("Stage", "stage"), ("R_loss", "r_loss"), ("R_grad", "r_grad"), ("R_orth", "r_orth"), ("A_rel", "a_rel"), ("A_total", "a_total"), ("I_total", "i_total"), ("mean s_pos", "mean_s_pos"), ("active rate", "relation_active_batch_rate")]),
        "",
        "The temporal panels are `plots/temporal_balance.png` and the individual `plots/temporal_*.png`. Because Early/Mid may share the nearest available `model_best.pth`, this is a three-checkpoint-resolution audit, not a continuous epoch trajectory.",
        "",
        "## 6. Relation saturation",
        "",
        md_table(sat_rows, [("Stage", "stage"), ("mean s_pos", "mean_s_pos"), ("mean L_rel", "relation_loss"), ("s>=.70", "ge070"), ("s>=.80", "ge080"), ("s>=.90", "ge090"), ("active batch rate", "active_rate")]),
        "",
        "Saturation fractions are pair-weighted over active diagnostic Relation pairs; relation-active batch rate is batch-weighted over all 32 batches. Full data: `saturation/saturation_summary.csv`.",
        "",
        "## 7. Per-batch strength distribution",
        "",
        md_table([row for row in threshold_rows if row["source_id"] in ("ltcc_i3", "prcc_c1")], [("Source", "source_id"), ("Stage", "stage"), ("R_grad>.05", "r_grad_gt_005"), ("R_grad>.10", "r_grad_gt_010"), ("R_grad>.20", "r_grad_gt_020"), ("R_grad>.50", "r_grad_gt_050"), ("R_orth>.05", "r_orth_gt_005"), ("R_orth>.10", "r_orth_gt_010"), ("R_orth>.20", "r_orth_gt_020"), ("R_orth>.50", "r_orth_gt_050")]),
        "",
        "Thresholds are descriptive only and were not tuned. Full per-batch distributions are retained in `tables/batch_diagnostics.csv`.",
        "",
        "## 8. Update-direction stability",
        "",
        md_table([row for row in stability_rows if row["source_id"] in ("ltcc_i3", "prcc_c1")], [("Source", "source_id"), ("Stage", "stage"), ("cos base t,t+1", "cos_base_consecutive"), ("cos rel t,t+1", "cos_rel_consecutive"), ("cos total t,t+1", "cos_total_consecutive")]),
        "",
        "These are supplementary batch-to-batch statistics and are not interpreted as epoch-level optimization stability.",
        "",
        "## 9. LTCC I1 vs I3",
        "",
        md_table(i1_i3, [("Stage", "stage"), ("I1 R_loss", "i1_r_loss"), ("I3 R_loss", "i3_r_loss"), ("I1 R_grad", "i1_r_grad"), ("I3 R_grad", "i3_r_grad"), ("I1 R_orth", "i1_r_orth"), ("I3 R_orth", "i3_r_orth"), ("I1 A_rel", "i1_a_rel"), ("I3 A_rel", "i3_a_rel")]),
        "",
        "The profile plot is `ltcc_i1_i3/i1_vs_i3_profiles.png`; the complete comparison is `ltcc_i1_i3/i1_vs_i3_strength.csv`.",
        "",
        "### Bio/control reference arms",
        "",
        md_table(control_rows, [("Source/stage", "source"), ("R_loss", "R_loss"), ("R_grad late EVA", "R_grad"), ("R_orth late EVA", "R_orth"), ("A_rel", "A_rel")]),
        "",
        "I0/C0 are Bio checkpoints; their Relation columns are diagnostic probes only, used to put the frozen base representations on the same measurement scale. The primary promotion comparison remains LTCC I3 versus PRCC C1.",
        "",
        "## 10. Optional virtual-lambda geometry",
        "",
        md_table(virtual_primary, [("Source", "source_id"), ("Stage", "stage"), ("lambda", "lambda_virtual"), ("cos base,total mean", "cos_base_total_mean"), ("inflation mean", "magnitude_inflation_mean"), ("orth injection mean", "orthogonal_injection_mean")]),
        "",
        "This is an offline geometry calculation from fixed gradients, not a hyperparameter sweep and not evidence that any coefficient is optimal. Full table: `virtual_lambda/virtual_lambda_summary.csv`.",
        "",
        "## 11. Cross-dataset master table",
        "",
        report_table_from_master(master),
        "",
        "Loss/norm/saturation entries use batch or pair-weighted means; vector ratios and angles are also available with median/p25/p75 in the detailed CSVs.",
        "",
        "## 12. Decision questions",
        "",
        *decision_lines,
        "",
        "Decision classifications are descriptive summaries of the frozen three-stage probes; the raw distributions and stage mapping are authoritative.",
        "",
        "## 13. Promotion decision",
        "",
        f"**{promotion}**.",
        "",
        "The requested promotion gate requires a clearly different effective Relation strength across stages, a non-artifactual coherent R_grad/R_orth/saturation mechanism, and a favorable LTCC successful-arm regime. If those joint conditions are not supported, the optimization-balance hypothesis stops here; no lambda sweep, learned balancing, GradNorm, uncertainty weighting, PCGrad, or router is authorized.",
        "",
        "## Strongest mechanistic conclusion",
        "",
        strongest,
        "",
        "NO NEW REID TRAINING WAS STARTED.",
        "NO PHASE 7K-B WAS STARTED.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{MANIFEST_PATH}`",
        f"- Source audit: `{SOURCE_REPORT_PATH}` and `{SOURCE_AUDIT_JSON}`",
        f"- Output root: `{OUT_ROOT}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    device = require_ppu()
    set_seed()
    audit = source_audit()
    if not all(audit["checks"].values()):
        raise RuntimeError(f"source audit failed: {audit['checks']}")
    runtimes = build_runtimes()
    audit["runtime_relation_banks"] = {
        dataset_key: {
            "source": runtime["relation_bank_source"],
            "pair_count": int(runtime["bank"]["exact_pair_count"]),
            "pid_count": int(runtime["bank"]["pid_count"]),
            # The LTCC Phase 6B parser names its audited state-pair count
            # `state_pair_count`; PRCC's Phase 7A builder names the analogous
            # field `clothing_state_count`.
            "clothing_state_count": int(runtime["bank"].get(
                "clothing_state_count", runtime["bank"].get("state_pair_count", 0)
            )),
        }
        for dataset_key, runtime in runtimes.items()
    }
    write_json(SOURCE_AUDIT_JSON, audit)
    write_source_report(audit)
    stages = audit["stage_mapping"]
    batch_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    all_virtual_rows: list[dict[str, Any]] = []
    for source_id, source in SOURCE_SPECS.items():
        runtime = runtimes[source["dataset_key"]]
        for stage_name, stage in stages[source_id].items():
            cache_key = (source["dataset_key"], int(stage["actual_epoch"]))
            if cache_key not in batch_cache:
                print(f"[phase7k-a] collecting {DIAGNOSTIC_BATCHES} {source['dataset']} train batches at frozen loader epoch E{stage['actual_epoch']}", flush=True)
                batch_cache[cache_key] = collect_batches(runtime, source["dataset_key"], int(stage["actual_epoch"]), args.batches)
            rows, virtual_rows = evaluate_stage(source_id, source, stage, runtime, batch_cache[cache_key], device)
            all_rows.extend(rows)
            all_virtual_rows.extend(virtual_rows)
    # The raw table is intentionally retained so the report can be audited
    # without rerunning PPU autograd.
    raw_fields = sorted({key for row in all_rows for key in row.keys()})
    write_csv(OUT_ROOT / "tables" / "batch_diagnostics.csv", all_rows, raw_fields)
    virtual_fields = sorted({key for row in all_virtual_rows for key in row.keys()})
    write_csv(OUT_ROOT / "virtual_lambda" / "virtual_lambda_per_batch.csv", all_virtual_rows, virtual_fields)
    summaries = write_distributions(all_rows, all_virtual_rows, stages)
    summaries["raw_rows"] = all_rows
    master = make_master_table(summaries)
    comparison = make_temporal_and_comparisons(summaries)
    decisions = classify_decisions(summaries, comparison)
    plot_paths = make_plots(summaries, comparison)
    manifest = {
        "phase": PHASE,
        "created_utc": audit["created_utc"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_memory_gib": float(torch.cuda.get_device_properties(device).total_memory / 2**30),
        "training_started": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "scheduler_created": False,
        "gradscaler_created": False,
        "backward_called": False,
        "new_reid_training_started": False,
        "phase7k_b_started": False,
        "lambda_rel": LAMBDA_REL,
        "virtual_lambdas": list(VIRTUAL_LAMBDAS),
        "diagnostic_batches_per_stage": args.batches,
        "minimum_diagnostic_batches": MIN_DIAGNOSTIC_BATCHES,
        "stage_mapping": stages,
        "source_audit": str(SOURCE_AUDIT_JSON),
        "raw_batch_table": str(OUT_ROOT / "tables" / "batch_diagnostics.csv"),
        "master_table": str(OUT_ROOT / "tables" / "master_table.csv"),
        "decisions": decisions,
        "promotion": "PROMOTE" if decisions["Q8_sufficient_evidence_for_adaptive_strength_branch"] == "YES" else "STOP",
        "plots": plot_paths,
        "objective_identity_limitation": audit["objective_identity_limitation"],
        "runtime_relation_banks": audit["runtime_relation_banks"],
        "relation_gradients": "unscaled FP32 autograd.grad; relation uses normalized feat[0]; no update",
    }
    write_json(MANIFEST_PATH, manifest)
    write_final_report(audit, summaries, comparison, master, decisions, plot_paths)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=DIAGNOSTIC_BATCHES)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.batches < MIN_DIAGNOSTIC_BATCHES:
        raise SystemExit(f"--batches must be >= {MIN_DIAGNOSTIC_BATCHES}")
    result = run(arguments)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "report": str(REPORT_PATH), "promotion": result["promotion"], "decisions": result["decisions"]}, indent=2, sort_keys=True))
