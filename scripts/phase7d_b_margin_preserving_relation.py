"""Phase 7D-B: first PRCC margin-preserving relation training test.

This runner deliberately reuses the audited Phase 7C PRCC training loop and
creates a new output root.  The existing Phase 7C C0 E60 run is the matched
Bio reference; only the new margin-preserving relation arm is trained here.

The first test is predeclared, not tuned:

    m = 0.05
    L_margin = mean(relu(m + s_neg_pair - s_pos))

For a same-PID/different-clothes image pair in the current mini-batch,
``s_pos`` is its cosine similarity.  ``s_neg_pair`` is the larger of the
hardest different-PID in-batch cosine similarities for the two endpoints.
The frozen PRCC relation bank is used only to identify valid train pairs.
There is no memory bank, sampler change, teacher, EMA, or second seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import phase6b_soft_relation_screen as phase6b  # noqa: E402
import phase7a_prcc_relation_validation as phase7a  # noqa: E402
import phase7c_prototype_relation as phase7c  # noqa: E402


PHASE = "phase7d_b_margin_preserving_relation"
SEED = 1234
WORLD_SIZE = 2
FORMAL_EPOCHS = 60
MARGIN = 0.05
LAMBDA_REL = 0.1

OUT_ROOT = ROOT / "outputs/new_method_track/phase7d_b_margin_preserving_relation"
SOURCE_ROOT = OUT_ROOT / "source_audit"
FEASIBILITY_ROOT = OUT_ROOT / "feasibility"
TRAINING_ROOT = OUT_ROOT / "training"
ANALYSIS_ROOT = OUT_ROOT / "analysis"
REPORT_PATH = ROOT / "reports/PHASE_7D_B_MARGIN_PRESERVING_RELATION.md"
SOURCE_REPORT = ROOT / "reports/PHASE_7D_B_SOURCE_AUDIT.md"
MANIFEST_PATH = OUT_ROOT / "PHASE7D_B_MANIFEST.json"
RUN_DIR = TRAINING_ROOT / "D1_margin_preserving_relation"

PHASE7C_ROOT = ROOT / "outputs/new_method_track/phase7c_prototype_relation"
PHASE7C_C0 = PHASE7C_ROOT / "training/C0_bio"
PHASE7C_C1 = PHASE7C_ROOT / "training/C1_image_uniform_relation"
PHASE7C_C2 = PHASE7C_ROOT / "training/C2_prototype_relation"
PHASE7A_ROOT = ROOT / "outputs/new_method_track/phase7a_prcc_relation_validation"
COMMON_INIT = PHASE7A_ROOT / "PRCC_COMMON_INIT.pth"
COMMON_META = PHASE7A_ROOT / "PRCC_COMMON_INIT_HASHES.json"
RELATION_BANK = PHASE7A_ROOT / "artifacts/prcc_relation_bank.npz"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_phase7c_module() -> None:
    """Point the audited Phase 7C loop at the new, isolated D1 root."""
    phase7c.PHASE = PHASE
    phase7c.OUT_ROOT = OUT_ROOT
    phase7c.SOURCE_ROOT = SOURCE_ROOT
    phase7c.FEASIBILITY_ROOT = FEASIBILITY_ROOT
    phase7c.TRAINING_ROOT = TRAINING_ROOT
    phase7c.BALANCE_ROOT = ANALYSIS_ROOT / "relation_balance"
    phase7c.GEOMETRY_ROOT = ANALYSIS_ROOT / "geometry"
    phase7c.GRADIENT_ROOT = ANALYSIS_ROOT / "gradient_diagnostic"
    phase7c.TABLE_ROOT = ANALYSIS_ROOT / "tables"
    phase7c.PLOT_ROOT = ANALYSIS_ROOT / "plots"
    phase7c.MANIFEST_PATH = MANIFEST_PATH
    phase7c.REPORT_PATH = REPORT_PATH
    phase7c.SOURCE_REPORT = SOURCE_REPORT
    phase7c.RUN_DIRS = {"C2": RUN_DIR}
    phase7c.ARM_LABELS = {"C2": "D1 PRCC margin-preserving relation"}
    phase7c.LAMBDA_REL = LAMBDA_REL
    phase7c.FORMAL_EPOCHS = FORMAL_EPOCHS
    phase7c.PHASE7A_ROOT = PHASE7A_ROOT
    phase7c.PHASE7A_P0 = PHASE7A_ROOT / "P0_prcc_bio"
    phase7c.PHASE7A_P1 = PHASE7A_ROOT / "P1_prcc_uniform_relation"
    phase7c.PHASE7A_COMMON_INIT = COMMON_INIT
    phase7c.PHASE7A_COMMON_META = COMMON_META
    phase7c.PHASE7A_METHOD = PHASE7A_ROOT / "artifacts/frozen_relation_method.md"
    phase7c.PHASE6B_SCRIPT = ROOT / "scripts/phase6b_soft_relation_screen.py"
    phase7c.PHASE7A_SCRIPT = ROOT / "scripts/phase7a_prcc_relation_validation.py"


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


def source_audit() -> dict[str, Any]:
    configure_phase7c_module()
    for path in (COMMON_INIT, COMMON_META, RELATION_BANK):
        if not path.is_file():
            raise RuntimeError(f"missing frozen source: {path}")
    cfg = phase7a.configure(OUT_ROOT / "config_probe")
    dataset = phase7a.build_dataset(cfg)
    expected = (17896, 150, 300, 551)
    if (len(dataset.train), dataset.num_train_pids, dataset.num_train_clothes) != expected[:3]:
        raise RuntimeError(f"unexpected PRCC topology: {len(dataset.train)}, {dataset.num_train_pids}, {dataset.num_train_clothes}")
    relation_bank = phase7a.load_relation_bank(dataset)
    if len(relation_bank["exact_lookup"]) != 501040:
        raise RuntimeError(f"unexpected relation bank pair count: {len(relation_bank['exact_lookup'])}")
    common_meta = json.loads(COMMON_META.read_text(encoding="utf-8"))
    common_state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    common_hash = phase7c.state_hash(common_state)
    if common_hash != common_meta.get("state_hash") or sha256_file(COMMON_INIT) != common_meta.get("state_sha256"):
        raise RuntimeError("common initialization hash mismatch")
    c0_manifest = json.loads((PHASE7C_C0 / "run_manifest.json").read_text(encoding="utf-8"))
    c1_manifest = json.loads((PHASE7C_C1 / "run_manifest.json").read_text(encoding="utf-8"))
    c2_manifest = json.loads((PHASE7C_C2 / "run_manifest.json").read_text(encoding="utf-8"))
    for name, manifest in (("C0", c0_manifest), ("C1", c1_manifest), ("C2", c2_manifest)):
        if manifest.get("seed") != SEED or manifest.get("epochs") != FORMAL_EPOCHS or manifest.get("steps_per_epoch") != expected[3]:
            raise RuntimeError(f"Phase 7C {name} schedule is not the audited PRCC schedule")
    if abs(float(c0_manifest["lambda_rel"])) > 1e-12 or abs(float(c1_manifest["lambda_rel"]) - 0.1) > 1e-12 or abs(float(c2_manifest["lambda_rel"]) - 0.1) > 1e-12:
        raise RuntimeError("Phase 7C relation coefficients are not frozen as expected")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        if (RUN_DIR / "RUN_COMPLETE.json").is_file():
            raise RuntimeError(f"refusing to overwrite completed D1 run: {RUN_DIR}")
        raise RuntimeError(f"incomplete D1 output exists; manual audit required before resume: {RUN_DIR}")
    for directory in (OUT_ROOT, SOURCE_ROOT, FEASIBILITY_ROOT, TRAINING_ROOT, ANALYSIS_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    source = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "audit_pass": True,
        "source_verification_pass": True,
        "training_started": False,
        "optimizer_step_called": False,
        "seed": SEED,
        "dataset": {"name": "PRCC", "root": "/home/data/prcc", "train_images": len(dataset.train), "train_pids": dataset.num_train_pids, "train_clothing_states": dataset.num_train_clothes, "valid_image_pairs": len(relation_bank["exact_lookup"])},
        "primary_control": {"arm": "C0", "checkpoint": str(PHASE7C_C0 / "model_final.pth"), "checkpoint_sha256": sha256_file(PHASE7C_C0 / "model_final.pth"), "run_manifest": str(PHASE7C_C0 / "run_manifest.json")},
        "reference_relation": {"arm": "C1", "checkpoint": str(PHASE7C_C1 / "model_final.pth"), "checkpoint_sha256": sha256_file(PHASE7C_C1 / "model_final.pth"), "run_manifest": str(PHASE7C_C1 / "run_manifest.json")},
        "prototype_reference": {"arm": "C2", "checkpoint": str(PHASE7C_C2 / "model_final.pth"), "checkpoint_sha256": sha256_file(PHASE7C_C2 / "model_final.pth"), "run_manifest": str(PHASE7C_C2 / "run_manifest.json")},
        "common_init": {"path": str(COMMON_INIT), "sha256": sha256_file(COMMON_INIT), "state_hash": common_hash, "strict_load": True},
        "feature_source": "normalized student feat[0], 1024 dimensions",
        "optimizer": phase7a.optimizer_config(cfg),
        "scheduler": phase7a.scheduler_config(cfg),
        "amp": phase7a.amp_config(),
        "sampler": "Phase 7C/Phase 7A DistributedRandomIdentitySampler; unchanged",
        "batch_size": 16,
        "num_instances": 4,
        "epochs": FORMAL_EPOCHS,
        "relation_bank": file_record(RELATION_BANK),
        "relation_lambda": LAMBDA_REL,
        "margin": MARGIN,
        "negative_definition": "pair-conservative hardest different-PID feature in current mini-batch for each endpoint; max of endpoint negatives",
        "relation_formula": "relu(m + s_neg_pair - s_pos), mean over canonical same-PID/different-clothes in-batch image pairs",
        "new_components": ["margin-preserving relation term only"],
        "forbidden": ["SR3", "H_ID", "H_cloth", "DS", "DW", "GRL", "Reverse", "new sampler", "new backbone", "new optimizer", "new scheduler", "memory bank", "EMA", "queue", "second seed"],
        "checks": {"phase7c_control_complete": True, "common_init_integrity": True, "relation_bank_count": 501040, "query_gallery_used_for_relation": False},
    }
    write_json(SOURCE_ROOT / "source_audit.json", source)
    report_lines = [
        "# PHASE 7D-B Source Audit", "", "Source verification: **PASS**.", "",
        "The frozen Phase 7C C0 E60 run is reused as the matched Bio control; it is not retrained or overwritten.", "",
        "| Item | Path |", "|---|---|",
        f"| C0 control E60 | `{PHASE7C_C0 / 'model_final.pth'}` |",
        f"| C1 image-relation reference | `{PHASE7C_C1 / 'model_final.pth'}` |",
        f"| C2 prototype reference | `{PHASE7C_C2 / 'model_final.pth'}` |",
        f"| Common initialization | `{COMMON_INIT}` |",
        f"| Train relation bank | `{RELATION_BANK}` |", "",
        "## Frozen D1 definition", "",
        f"- PRCC train-only, seed `{SEED}`, 60 epochs, batch 16, NUM_INSTANCES 4, 551 updates/rank/epoch.",
        f"- `lambda_rel = {LAMBDA_REL}`; predeclared margin `m = {MARGIN}`; no margin sweep.",
        "- Valid pair: same PID and different clothes; one unordered image pair per in-batch pair.",
        "- `s_neg_pair` is the maximum of the hardest different-PID in-batch cosine for the two pair endpoints.",
        "- No memory bank, sampler change, teacher, EMA, queue, or query/gallery relation construction.", "",
        f"Machine-readable audit: `{SOURCE_ROOT / 'source_audit.json'}`.",
    ]
    SOURCE_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return source


def prepare() -> dict[str, Any]:
    configure_phase7c_module()
    source_path = SOURCE_ROOT / "source_audit.json"
    if not source_path.is_file():
        source_audit()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("audit_pass") or source.get("training_started"):
        raise RuntimeError("source audit is not a clean PASS")
    # Reuse the exact sampler implementation for a train-only coverage check.
    # This does not execute a model or optimizer.
    coverage = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "train_only": True,
        "query_gallery_used_for_relation": False,
        "coverage_pass": True,
        "global_batches_per_epoch": 2 * phase7c.EXPECTED_STEPS_PER_EPOCH,
        "steps_per_rank_epoch": phase7c.EXPECTED_STEPS_PER_EPOCH,
        "valid_relation_pairs": source["dataset"]["valid_image_pairs"],
        "margin": MARGIN,
        "sampler": source["sampler"],
        "training_started": False,
        "optimizer_step_called": False,
    }
    write_json(FEASIBILITY_ROOT / "coverage_summary.json", coverage)
    write_json(OUT_ROOT / "PREPARE.json", {"phase": PHASE, "source_audit_pass": True, "coverage_pass": True, "allowed_arm": "D1", "margin": MARGIN, "lambda_rel": LAMBDA_REL, "training_started": False})
    return coverage


def margin_relation_loss(feat: list[torch.Tensor], pids: torch.Tensor, clothes: torch.Tensor, image_paths: list[str], path_to_index: dict[str, int], bank: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    representation = F.normalize(feat[0].float(), dim=-1)
    pid_values = [int(value) for value in pids.detach().cpu().tolist()]
    cloth_values = [int(value) for value in clothes.detach().cpu().tolist()]
    image_indices = [path_to_index[str(path)] for path in image_paths]
    similarity = representation @ representation.t()
    losses: list[torch.Tensor] = []
    keys: list[tuple[int, int, int]] = []
    positive_values: list[float] = []
    negative_values: list[float] = []
    margin_values: list[float] = []
    active_values: list[float] = []
    for i in range(len(pid_values)):
        for j in range(i + 1, len(pid_values)):
            if pid_values[i] != pid_values[j] or cloth_values[i] == cloth_values[j]:
                continue
            exact_key = (min(image_indices[i], image_indices[j]), max(image_indices[i], image_indices[j]))
            if exact_key not in bank["exact_lookup"]:
                continue
            negative_i = similarity[i].masked_fill(torch.tensor([pid == pid_values[i] for pid in pid_values], device=similarity.device), -torch.inf).max()
            negative_j = similarity[j].masked_fill(torch.tensor([pid == pid_values[j] for pid in pid_values], device=similarity.device), -torch.inf).max()
            if not torch.isfinite(negative_i) or not torch.isfinite(negative_j):
                continue
            positive = similarity[i, j]
            negative = torch.maximum(negative_i, negative_j)
            margin = positive - negative
            losses.append(F.relu(MARGIN - margin))
            keys.append((pid_values[i], min(cloth_values[i], cloth_values[j]), max(cloth_values[i], cloth_values[j])))
            positive_values.append(float(positive.detach().cpu()))
            negative_values.append(float(negative.detach().cpu()))
            margin_values.append(float(margin.detach().cpu()))
            active_values.append(float((MARGIN - margin > 0).detach().cpu()))
    if not losses:
        zero = representation.sum() * 0.0
        return zero, {"active": False, "pair_count": 0, "keys": [], "weights": [], "similarities": [], "positive": [], "negative": [], "margin": [], "active_pair_count": 0}
    return torch.stack(losses).mean(), {
        "active": True,
        "pair_count": len(losses),
        "keys": keys,
        "weights": [1.0] * len(losses),
        "similarities": positive_values,
        "positive": positive_values,
        "negative": negative_values,
        "margin": margin_values,
        "active_pair_count": int(sum(active_values)),
    }


def margin_forward_batch(model: Any, loss_fn: Any, batch: dict[str, Any], device: torch.device, global_step: int) -> dict[str, Any]:
    writer = phase7c.base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with phase7c.amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    raw_relation, relation_info = margin_relation_loss(feat, pids, clothes, [str(path) for path in batch["image_path"]], phase7c._D1_PATH_TO_INDEX, phase7c._D1_BANK)
    total = base_total + LAMBDA_REL * raw_relation
    components = getattr(loss_fn, "last_components", {})
    return {
        "total": total,
        "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "relation_raw": raw_relation,
        "relation_applied": raw_relation,
        "relation_info": relation_info,
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1],
        "writer": writer,
    }


def margin_batch_exposure(records: list[dict[str, Any]], path_to_index: dict[str, int]) -> dict[str, Any]:
    by_pid: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for local, item in enumerate(records):
        by_pid[int(item["pid"])][int(item["clothes_id"])].append(local)
    image_keys: list[tuple[int, int]] = []
    state_keys: list[tuple[int, int, int]] = []
    pids_seen: set[int] = set()
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if int(records[i]["pid"]) != int(records[j]["pid"]) or int(records[i]["clothes_id"]) == int(records[j]["clothes_id"]):
                continue
            a = path_to_index[str(records[i]["image_path"])]
            b = path_to_index[str(records[j]["image_path"])]
            image_keys.append((min(a, b), max(a, b)))
            state_keys.append((int(records[i]["pid"]), min(int(records[i]["clothes_id"]), int(records[j]["clothes_id"])), max(int(records[i]["clothes_id"]), int(records[j]["clothes_id"]))))
            pids_seen.add(int(records[i]["pid"]))
    return {
        "image_count": len(image_keys), "image_keys": image_keys, "image_state_keys": state_keys,
        "proto_count": len(image_keys), "proto_keys": state_keys, "active_proto": len(pids_seen),
        "proto_pids": pids_seen, "proto_sizes": [], "proto_sims": [],
    }


def patch_training_globals() -> None:
    configure_phase7c_module()
    phase7c.c2_forward_batch = margin_forward_batch
    phase7c.batch_exposure = margin_batch_exposure
    phase7c._D1_BANK = phase7a.load_relation_bank(phase7a.build_dataset(phase7a.configure(OUT_ROOT / "config_probe")))
    phase7c._D1_PATH_TO_INDEX = phase7c._D1_BANK["path_to_index"]


def train() -> None:
    configure_phase7c_module()
    prepare()
    patch_training_globals()
    phase7c.train_arm("C2")


def finalize_manifest() -> dict[str, Any]:
    manifest_path = RUN_DIR / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("D1 run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "phase": PHASE,
        "arm": "D1",
        "label": "PRCC Bio + margin-preserving cross-clothes relation",
        "relation_type": "margin_preserving_image_relation",
        "relation_formula": "mean(relu(0.05 + s_neg_pair - s_pos))",
        "margin": MARGIN,
        "negative_definition": "pair-conservative max of endpoint hardest different-PID in-batch cosine",
        "relation_lambda": LAMBDA_REL,
        "relation_feature": "normalized feat[0]",
        "only_method_difference": "D1 applies the predeclared margin-preserving relation term; matched C0 is the frozen Phase 7C E60 control",
        "control_checkpoint": str(PHASE7C_C0 / "model_final.pth"),
        "training_started": True,
        "optimizer_step_called": True,
        "second_seed_started": False,
        "phase7db_started": True,
        "query_gallery_used_for_relation": False,
    })
    write_json(manifest_path, manifest)
    write_json(MANIFEST_PATH, {
        "phase": PHASE, "seed": SEED, "epochs": FORMAL_EPOCHS, "margin": MARGIN, "lambda_rel": LAMBDA_REL,
        "source_audit": str(SOURCE_ROOT / "source_audit.json"), "coverage": str(FEASIBILITY_ROOT / "coverage_summary.json"),
        "control_checkpoint": str(PHASE7C_C0 / "model_final.pth"), "D1_run": str(RUN_DIR),
        "training_started": True, "optimizer_step_called": True, "phase7db_started": True,
        "second_seed_started": False, "query_gallery_used_for_relation": False,
        "relation_formula": "mean(relu(m + s_neg_pair - s_pos))",
        "negative_definition": "current-mini-batch pair-conservative hard negative",
        "promotion_status": "training_complete_pending_analysis",
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "prepare", "train", "finalize"), required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(source_audit(), indent=2, sort_keys=True, default=str))
    elif args.mode == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True, default=str))
    elif args.mode == "train":
        train()
    else:
        print(json.dumps(finalize_manifest(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
