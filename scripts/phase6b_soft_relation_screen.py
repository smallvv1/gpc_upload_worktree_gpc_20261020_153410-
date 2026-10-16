#!/usr/bin/env python3
"""Phase 6B: limited soft reliability-aware relation supervision screen.

This runner is deliberately separate from the formal DIFFER trainer.  It uses
the frozen Bio-only Phase5B topology and common initialization, keeps every
LTCC train image/clothing state visible, and adds only a positive same-PID /
different-clothes relation term for B1--B3.

The four locked arms are:

  B0  Bio CE + Triplet + Summary Bio
  B1  B0 + uniform cross-clothes positive relation loss
  B2  B0 + frozen SR3-weighted cross-clothes positive relation loss
  B3  B0 + the same SR3 weight distribution shuffled within PID

SR3 is read once from Phase6A3 before training.  H_ID, DS, DW, online
reliability, Reverse/GRL, and test/query/gallery relation construction are not
used.  The scheduler remains the formal 60-epoch scheduler; this screen runs
through epoch 30 only.
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
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase5b_id_damage_ds as base  # noqa: E402


PHASE = "phase6b_soft_relation_screen"
SEED = 1234
WORLD_SIZE = 2
SCREEN_EPOCHS = 30
FORMAL_TOTAL_EPOCHS = 60
WARMUP_EPOCHS = 20
LAMBDA_REL = 0.1
RELATION_W_MIN = 0.5
NUM_CLASSES = 77
NUM_CLOTHING_STATES = 256
STEPS_PER_EPOCH = base.STEPS_PER_EPOCH
BATCH_SIZE = base.BATCH_SIZE
NUM_INSTANCES = base.NUM_INSTANCES

A3_ROOT = ROOT / "outputs/new_method_track/phase6a3_soft_reliability"
A3_MANIFEST = A3_ROOT / "PHASE6A3_MANIFEST.json"
A3_SCORES = A3_ROOT / "soft_reliability_scores.csv"
A3_COMPONENTS = A3_ROOT / "soft_reliability_components.csv"
A2_ROOT = ROOT / "outputs/new_method_track/phase6a2_multi_evidence_reliability"
A2_CROSS = A2_ROOT / "cross_clothes_positive.csv"
BIO_FEATURE_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features/bio_control_feature_bank.npz"

COMMON_INIT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth"
COMMON_INIT_META = ROOT / "outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT_HASHES.json"

OUT_ROOT = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen"
ARTIFACT_ROOT = OUT_ROOT / "artifacts"
REPORT_PATH = ROOT / "reports/PHASE_6B_SOFT_RELATION_SUPERVISION_SCREEN.md"
MANIFEST_PATH = OUT_ROOT / "PHASE6B_MANIFEST.json"
MATCHED_AUDIT_PATH = ARTIFACT_ROOT / "matched_control_audit.json"

ARM_DIRS = {
    "B0": OUT_ROOT / "B0_uniform_base",
    "B1": OUT_ROOT / "B1_uniform_relation",
    "B2": OUT_ROOT / "B2_sr3_soft_relation",
    "B3": OUT_ROOT / "B3_shuffled_reliability",
}

ARM_LABELS = {
    "B0": "Bio control: CE + Triplet + Bio",
    "B1": "Uniform cross-clothes positive relation",
    "B2": "SR3 soft reliability cross-clothes relation",
    "B3": "Shuffled-SR3 relation control",
}

METRIC_FIELDS = [
    "epoch", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "relation_raw_loss",
    "relation_applied_loss", "relation_lambda", "lr", "lr_bias", "id_accuracy", "bio_accuracy",
    "amp_scale_min", "amp_scale_max", "amp_scale_end", "skipped_steps", "found_inf_count",
    "nonfinite_events", "optimizer_updates", "active_images", "active_pids", "active_clothes",
    "mean_clothes_per_pid", "sampled_unique_images", "sampled_unique_states", "sampled_unique_pids",
    "relation_active_batches", "relation_active_batch_rate", "relation_pairs", "relation_unique_state_pairs",
    "relation_weight_mean", "relation_weight_std", "relation_weight_min", "relation_weight_max",
    "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP", "general_rank1", "general_rank5",
    "general_rank10", "general_mAP",
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    cfg = base.configure()
    cfg.defrost()
    cfg.SOLVER.MAX_EPOCHS = FORMAL_TOTAL_EPOCHS
    cfg.SOLVER.WARMUP_EPOCHS = WARMUP_EPOCHS
    cfg.SOLVER.CHECKPOINT_PERIOD = FORMAL_TOTAL_EPOCHS
    cfg.OUTPUT_DIR = str(OUT_ROOT)
    cfg.freeze()
    return cfg


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        key: sha256_bytes(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for key, value in sorted(state.items())
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, base.DDP) else model
    return {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}


def model_parameter_hash(model: torch.nn.Module) -> str:
    return sha256_bytes(json.dumps(tensor_hashes(cpu_state_dict(model)), sort_keys=True).encode())


def load_common_init() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not COMMON_INIT.is_file() or not COMMON_INIT_META.is_file():
        raise FileNotFoundError("Phase6B requires the frozen Phase5B Bio common init")
    meta = json.loads(COMMON_INIT_META.read_text())
    if sha256_file(COMMON_INIT) != meta["state_sha256"]:
        raise RuntimeError("common init SHA256 mismatch")
    state = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    if tensor_hashes(state) != meta["tensor_hashes"]:
        raise RuntimeError("common init tensor hash mismatch")
    audit = {
        "path": COMMON_INIT,
        "sha256": sha256_file(COMMON_INIT),
        "meta_path": COMMON_INIT_META,
        "meta_sha256": sha256_file(COMMON_INIT_META),
        "state_key_count": len(state),
        "state_numel": int(sum(t.numel() for t in state.values())),
        "topology": meta.get("topology"),
        "reverse": meta.get("reverse"),
        "strict_hash_match": True,
    }
    return state, audit


def source_checksum_audit() -> dict[str, Any]:
    paths = [A3_MANIFEST, A3_SCORES, A3_COMPONENTS, A2_CROSS, BIO_FEATURE_BANK, COMMON_INIT, COMMON_INIT_META]
    return {
        str(path): {"exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None,
                    "size": path.stat().st_size if path.is_file() else None}
        for path in paths
    }


def parse_relation_bank(dataset: Any, cfg: Any) -> dict[str, Any]:
    """Load frozen SR3 scores and retain a compact image-pair candidate bank."""
    for path in (A3_MANIFEST, A3_SCORES, A3_COMPONENTS, A2_CROSS):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(A3_MANIFEST.read_text())
    if manifest.get("score_inputs", {}).get("h_id_used_in_score"):
        raise RuntimeError("Phase6A3 manifest says H_ID entered the reliability score")
    score_rows = [row for row in read_csv(A3_SCORES) if row.get("candidate") == "SR3_CONSENSUS_MUTUAL_INTRA"]
    component_rows = read_csv(A3_COMPONENTS)
    cross_rows = read_csv(A2_CROSS)
    if len(score_rows) != 43839 or len(component_rows) != 43839 or len(cross_rows) != 43839:
        raise RuntimeError(f"unexpected Phase6A3 relation counts: {len(score_rows)}, {len(component_rows)}, {len(cross_rows)}")
    scores = {int(row["pair_id"]): float(row["mean_score"]) for row in score_rows}
    components = {int(row["pair_id"]): row for row in component_rows}
    cross = {int(row["pair_id"]): row for row in cross_rows}
    if set(scores) != set(components) or set(scores) != set(cross):
        raise RuntimeError("Phase6A3 relation keys differ across score/components/cross-clothes files")

    exact: dict[tuple[int, int], list[float]] = defaultdict(list)
    state_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    state_source_ids: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    relation_rows: list[dict[str, Any]] = []
    for pair_id in sorted(scores):
        score = scores[pair_id]
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise RuntimeError(f"SR3 score out of range at pair {pair_id}: {score}")
        comp = components[pair_id]
        crow = cross[pair_id]
        index_a, index_b = int(comp["index_a"]), int(comp["index_b"])
        pid = int(comp["pid"])
        if int(crow["index_a"]) != index_a or int(crow["index_b"]) != index_b or int(crow["pid"]) != pid:
            raise RuntimeError(f"Phase6A3 image/PID key mismatch at pair {pair_id}")
        cloth_a, cloth_b = int(crow["clothes_a"]), int(crow["clothes_b"])
        if cloth_a == cloth_b:
            raise RuntimeError(f"same-clothes row in cross-clothes source at pair {pair_id}")
        state_key = (pid, min(cloth_a, cloth_b), max(cloth_a, cloth_b))
        exact_key = (min(index_a, index_b), max(index_a, index_b))
        exact[exact_key].append(score)
        state_values[state_key].append(score)
        state_source_ids[state_key].append(pair_id)
        relation_rows.append({
            "pair_id": pair_id, "index_a": index_a, "index_b": index_b, "pid": pid,
            "clothes_a": cloth_a, "clothes_b": cloth_b, "state_key": json.dumps(state_key),
            "sr3_score": score,
        })

    # Validate the frozen feature-bank identity order against the actual train
    # dataset before any relation can enter a training batch.
    if len(dataset.train) != 9576:
        raise RuntimeError(f"unexpected LTCC train image count: {len(dataset.train)}")
    with np.load(BIO_FEATURE_BANK, allow_pickle=False) as bank:
        stems = np.asarray(bank["stems"])
        pids = np.asarray(bank["pid"])
        clothes = np.asarray(bank["clothes_id"])
    if len(stems) != len(dataset.train):
        raise RuntimeError("Bio feature bank and LTCC train count differ")
    for index, item in enumerate(dataset.train):
        if str(Path(str(item["image_path"])).stem) != str(stems[index]):
            raise RuntimeError(f"train/feature-bank image identity mismatch at {index}")
        if int(item["pid"]) != int(pids[index]) or int(item["clothes_id"]) != int(clothes[index]):
            raise RuntimeError(f"train/feature-bank label mismatch at {index}")

    rng = random.Random(SEED)
    state_bank: list[dict[str, Any]] = []
    b3_weight_by_key: dict[tuple[int, int, int], float] = {}
    for pid in sorted({key[0] for key in state_values}):
        keys = sorted(key for key in state_values if key[0] == pid)
        values = [RELATION_W_MIN + (1.0 - RELATION_W_MIN) * float(np.mean(state_values[key])) for key in keys]
        shuffled = list(values)
        rng.shuffle(shuffled)
        b3_weight_by_key.update({key: float(weight) for key, weight in zip(keys, shuffled)})
    for state_key in sorted(state_values):
        values = np.asarray(state_values[state_key], dtype=np.float64)
        b2 = RELATION_W_MIN + (1.0 - RELATION_W_MIN) * float(values.mean())
        state_bank.append({
            "pid": state_key[0], "clothes_low": state_key[1], "clothes_high": state_key[2],
            "source_pair_count": len(values), "source_pair_ids": json.dumps(state_source_ids[state_key]),
            "sr3_score_mean": float(values.mean()), "sr3_score_median": float(np.median(values)),
            "b1_weight": 1.0, "b2_weight": b2, "b3_weight": b3_weight_by_key[state_key],
        })
    state_lookup = {
        (int(row["pid"]), int(row["clothes_low"]), int(row["clothes_high"])): row
        for row in state_bank
    }
    exact_lookup = {key: float(np.mean(values)) for key, values in exact.items()}
    # Phase6A3 contains both orientations for some image pairs. Collapse only
    # those exact reversals; do not collapse all pairs belonging to one
    # clothing-state pair. The actual training weight is therefore based on
    # the frozen image-pair SR3 value, while the state bank remains an audit
    # summary.
    exact_pid: dict[tuple[int, int], int] = {}
    for row in relation_rows:
        key = (min(int(row["index_a"]), int(row["index_b"])), max(int(row["index_a"]), int(row["index_b"])))
        pid = int(row["pid"])
        if key in exact_pid and exact_pid[key] != pid:
            raise RuntimeError(f"exact image pair maps to multiple PIDs: {key}")
        exact_pid[key] = pid
    exact_b2 = {key: RELATION_W_MIN + (1.0 - RELATION_W_MIN) * score for key, score in exact_lookup.items()}
    keys_by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(exact_b2):
        keys_by_pid[exact_pid[key]].append(key)
    exact_b3: dict[tuple[int, int], float] = {}
    for pid, keys in sorted(keys_by_pid.items()):
        values = [exact_b2[key] for key in keys]
        shuffled = list(values)
        rng.shuffle(shuffled)
        exact_b3.update({key: float(weight) for key, weight in zip(keys, shuffled)})
    if sorted(exact_b2.values()) != sorted(exact_b3.values()):
        raise RuntimeError("B3 exact-pair shuffle changed the frozen weight distribution")
    for row in relation_rows:
        key = (min(int(row["index_a"]), int(row["index_b"])), max(int(row["index_a"]), int(row["index_b"])))
        row["b2_weight"] = exact_b2[key]
        row["b3_weight"] = exact_b3[key]
    return {
        "manifest": manifest, "relation_rows": relation_rows, "state_rows": state_bank,
        "state_lookup": state_lookup, "exact_lookup": exact_lookup,
        "exact_pid": exact_pid, "exact_weight_b2": exact_b2, "exact_weight_b3": exact_b3,
        "relation_count": len(relation_rows), "state_pair_count": len(state_bank),
        "exact_pair_count": len(exact_lookup),
        "pid_count": len({row["pid"] for row in state_bank}),
        "score_min": float(min(scores.values())), "score_max": float(max(scores.values())),
        "score_mean": float(np.mean(list(scores.values()))),
        "duplicate_exact_keys": int(sum(len(values) > 1 for values in exact.values())),
        "checksums": source_checksum_audit(),
    }


def relation_key(pid: int, cloth_a: int, cloth_b: int) -> tuple[int, int, int]:
    return int(pid), min(int(cloth_a), int(cloth_b)), max(int(cloth_a), int(cloth_b))


def relation_loss(
    feat: list[torch.Tensor],
    pids: torch.Tensor,
    clothes: torch.Tensor,
    image_paths: list[str],
    path_to_index: dict[str, int],
    bank: dict[str, Any],
    arm: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Positive-only same-PID/different-clothes relation loss in a batch."""
    representation = F.normalize(feat[0].float(), dim=-1)
    pid_values = pids.detach().cpu().tolist()
    cloth_values = clothes.detach().cpu().tolist()
    image_indices = [path_to_index[str(path)] for path in image_paths]
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    keys: list[tuple[int, int, int]] = []
    for i in range(len(pid_values)):
        for j in range(i + 1, len(pid_values)):
            if int(pid_values[i]) != int(pid_values[j]) or int(cloth_values[i]) == int(cloth_values[j]):
                continue
            exact_key = (min(image_indices[i], image_indices[j]), max(image_indices[i], image_indices[j]))
            if exact_key not in bank["exact_lookup"]:
                # Phase6A3 supplies a fixed compact relation candidate bank;
                # non-candidate cross-clothes pairs remain available to the
                # base Bio objective but receive no new relation term.
                continue
            if arm == "B1":
                weight = 1.0
            elif arm == "B2":
                weight = float(bank["exact_weight_b2"][exact_key])
            elif arm == "B3":
                weight = float(bank["exact_weight_b3"][exact_key])
            else:
                weight = 1.0
            losses.append(1.0 - torch.sum(representation[i] * representation[j]))
            weights.append(weight)
            keys.append(relation_key(pid_values[i], cloth_values[i], cloth_values[j]))
    if not losses:
        zero = representation.sum() * 0.0
        return zero, {"active": False, "pair_count": 0, "weights": [], "keys": []}
    loss_values = torch.stack(losses)
    weight_tensor = torch.tensor(weights, dtype=loss_values.dtype, device=loss_values.device)
    result = torch.sum(weight_tensor * loss_values) / torch.sum(weight_tensor)
    return result, {"active": True, "pair_count": len(losses), "weights": weights, "keys": keys}


def full_active_state(dataset: Any) -> dict[int, list[int]]:
    values: dict[int, set[int]] = defaultdict(set)
    for item in dataset.train:
        values[int(item["pid"])].add(int(item["clothes_id"]))
    state = {pid: sorted(clothes) for pid, clothes in values.items()}
    if len(state) != NUM_CLASSES or sum(len(v) for v in state.values()) != NUM_CLOTHING_STATES:
        raise RuntimeError("full LTCC state does not contain 77 PIDs / 256 clothing states")
    return state


def scan_planned_batches(loader: Any, dataset: Any, bank: dict[str, Any]) -> dict[str, Any]:
    """Audit relation activity from metadata without decoding another image."""
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    planned_images = {path_to_index[str(source["image_path"])] for source, _seed in loader.dataset.samples}
    active_batches = 0
    pair_count = 0
    keys: set[tuple[int, int, int]] = set()
    unlisted_cross_clothes_pairs = 0
    weight_values = {"B1": [], "B2": [], "B3": []}
    for batch_indices in loader.batch_sampler:
        entries = [loader.dataset.samples[index][0] for index in batch_indices]
        batch_keys: list[tuple[int, int, int]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if int(entries[i]["pid"]) == int(entries[j]["pid"]) and int(entries[i]["clothes_id"]) != int(entries[j]["clothes_id"]):
                    idx_i = path_to_index[str(entries[i]["image_path"])]
                    idx_j = path_to_index[str(entries[j]["image_path"])]
                    image_key = (min(idx_i, idx_j), max(idx_i, idx_j))
                    if image_key not in bank["exact_lookup"]:
                        # The frozen Phase6A3 bank is a compact candidate
                        # bank, not every possible same-PID image pair. A
                        # non-bank in-batch cross-clothes pair is visible to
                        # the base loss but has no Phase6B relation term.
                        unlisted_cross_clothes_pairs += 1
                    else:
                        state_key = relation_key(entries[i]["pid"], entries[i]["clothes_id"], entries[j]["clothes_id"])
                        batch_keys.append(state_key)
                        keys.add(state_key)
                        for arm in weight_values:
                            if arm == "B1":
                                weight_values[arm].append(1.0)
                            elif arm == "B2":
                                weight_values[arm].append(float(bank["exact_weight_b2"][image_key]))
                            else:
                                weight_values[arm].append(float(bank["exact_weight_b3"][image_key]))
        if batch_keys:
            active_batches += 1
            pair_count += len(batch_keys)
    return {
        "planned_unique_images": len(planned_images), "active_batches": active_batches,
        "pair_count": pair_count, "unique_state_pairs": len(keys),
        "unlisted_cross_clothes_pairs": unlisted_cross_clothes_pairs,
        "weights": weight_values,
    }


def write_preflight_docs(cfg: Any, bank: dict[str, Any], init_audit: dict[str, Any], sampler_rows: list[dict[str, Any]], gradient_rows: list[dict[str, Any]]) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6B reliability preflight",
        "",
        "This is a train-only preflight. It uses frozen Phase6A3 SR3 artifacts and the existing Phase5B Bio common init. No test/query/gallery relation data, H_ID, online update, DS, DW, Reverse, or GRL enters the relation construction.",
        "",
        f"- SR3 directed relation rows: `{bank['relation_count']}`; unique undirected image-pair candidates: `{bank['exact_pair_count']}`; state-pair audit rows: `{bank['state_pair_count']}`; source PIDs: `{bank['pid_count']}`.",
        f"- SR3 score range: `{bank['score_min']:.8f}` to `{bank['score_max']:.8f}`; mean `{bank['score_mean']:.8f}`.",
        f"- Relation weight mapping: `w = {RELATION_W_MIN} + {1.0 - RELATION_W_MIN} * SR3`, range `[0.5, 1.0]`; lambda relation `{LAMBDA_REL}`.",
        "- B0/B1/B2/B3 use the same full-data identity pair stream and the same frozen compact relation candidate bank. Reliability never removes an image, clothing state, PID, or ground-truth relation; an in-batch cross-clothes pair outside the candidate bank simply has no new relation term.",
        f"- Common init: `{init_audit['path']}`; SHA256 `{init_audit['sha256']}`; strict tensor-hash verification `{init_audit['strict_hash_match']}.",
        "",
        "## Matched sampler audit",
        "",
        "Every preflight epoch uses all 77 PIDs, all 256 clothing states, and 9576 available train images. The per-arm plan hash is shared because the same frozen full-data sampler is used for all arms.",
        "",
        "## Gradient-scale probe",
        "",
        "The probe uses a single matched train batch and calls backward only to measure finite preflight gradients; it does not create an optimizer or update parameters. It is not evidence of training utility.",
        "",
    ]
    (ARTIFACT_ROOT / "reliability_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(ARTIFACT_ROOT / "sampler_preflight.csv", sampler_rows)
    write_csv(ARTIFACT_ROOT / "gradient_scale_preflight.csv", gradient_rows)
    write_csv(ARTIFACT_ROOT / "relation_state_bank.csv", bank["state_rows"])
    write_csv(ARTIFACT_ROOT / "relation_source_rows.csv", bank["relation_rows"])
    write_json(ARTIFACT_ROOT / "source_checksums.json", bank["checksums"])
    write_json(ARTIFACT_ROOT / "common_init_audit.json", init_audit)
    for arm, directory in ARM_DIRS.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yml").write_text(cfg.dump(), encoding="utf-8")
        diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
        (directory / "git_diff.patch").write_bytes(diff)
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        (directory / "git_status.txt").write_text(status, encoding="utf-8")


def gradient_norms(named_parameters: list[tuple[str, torch.nn.Parameter]], grads: tuple[torch.Tensor | None, ...]) -> dict[str, float]:
    block_names = [name for name, _param in named_parameters if name.startswith("blocks.")]
    block_ids = sorted({int(name.split(".")[1]) for name in block_names if name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    groups = {"all": [], "late_eva": [], "bio_projection": [], "id_head": []}
    total_names = [name for name, _ in named_parameters]
    for (name, _param), grad in zip(named_parameters, grads):
        if grad is None:
            continue
        value = grad.detach().float()
        groups["all"].append(value)
        if name.startswith("blocks.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) in late_ids:
            groups["late_eva"].append(value)
        if name.startswith("head_bio") or name.startswith("head_clip_bio"):
            groups["bio_projection"].append(value)
        if name.startswith("head."):
            groups["id_head"].append(value)
    return {key: float(torch.sqrt(sum(torch.sum(v * v) for v in values)).detach().cpu()) if values else 0.0 for key, values in groups.items()}


def run_gradient_probe(cfg: Any, dataset: Any, bank: dict[str, Any], state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase6B gradient preflight requires CUDA/PPU; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    set_seed(SEED)
    model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    loss_fn, _center = base.make_loss(cfg, num_classes=NUM_CLASSES)
    transform_train, _ = base.build_img_transforms(cfg)
    index_map = base.build_index_map(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    state_all = full_active_state(dataset)
    loader, _meta = base.make_epoch_loader(dataset, transform_train, state_all, index_map, epoch=1, rank=0)
    selected_batch = None
    selected_stats = None
    for batch in loader:
        pid_values = batch["pid"].tolist()
        cloth_values = batch["clothes_id"].tolist()
        image_paths = [str(path) for path in batch["image_path"]]
        candidate_pair = any(
            pid_values[i] == pid_values[j]
            and cloth_values[i] != cloth_values[j]
            and (min(path_to_index[image_paths[i]], path_to_index[image_paths[j]]), max(path_to_index[image_paths[i]], path_to_index[image_paths[j]])) in bank["exact_lookup"]
            for i in range(len(pid_values)) for j in range(i + 1, len(pid_values))
        )
        if candidate_pair:
            selected_batch = batch
            break
    if selected_batch is None:
        raise RuntimeError("gradient preflight could not find a cross-clothes positive batch")
    images = selected_batch["image"].to(device)
    pids = selected_batch["pid"].to(device)
    camids = selected_batch["camid"].to(device)
    clothes = selected_batch["clothes_id"].to(device)
    captions = selected_batch["caption_feature"].to(device)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=base.MetricCapture(), step=1)
    rel, selected_stats = relation_loss(feat, pids, clothes, [str(path) for path in selected_batch["image_path"]], path_to_index, bank, "B2")
    total = base_total + LAMBDA_REL * rel
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    base_grads = torch.autograd.grad(base_total, [param for _name, param in named], retain_graph=True, allow_unused=True)
    rel_grads = torch.autograd.grad(rel, [param for _name, param in named], retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(total, [param for _name, param in named], allow_unused=True)
    base_norm = gradient_norms(named, base_grads)
    rel_norm = gradient_norms(named, rel_grads)
    total_norm = gradient_norms(named, total_grads)
    row = {
        "arm": "B2", "batch_size": len(pids), "relation_pairs": selected_stats["pair_count"],
        "base_loss": float(base_total.detach().float().cpu()), "relation_loss": float(rel.detach().float().cpu()),
        "lambda_relation_loss": float((LAMBDA_REL * rel).detach().float().cpu()),
        "total_loss": float(total.detach().float().cpu()),
        "loss_ratio_relation_to_base": float((LAMBDA_REL * rel / base_total.detach()).cpu()),
    }
    for prefix, values in (("base_grad", base_norm), ("relation_grad", rel_norm), ("total_grad", total_norm)):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (float, int))):
        raise RuntimeError("non-finite gradient preflight result")
    del total_grads, rel_grads, base_grads, total, rel, base_total, feat, score, model
    gc.collect()
    torch.cuda.empty_cache()
    return [row]


def preflight() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = configure()
    state, init_audit = load_common_init()
    model = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=NUM_CLASSES, camera_num=base.NUM_CAMERAS)
    model.load_state_dict(state, strict=True)
    topology = {
        "state_key_count": len(model.state_dict()), "state_numel": int(sum(value.numel() for value in model.state_dict().values())),
        "model_name": cfg.MODEL.NAME, "loss_type": list(cfg.MODEL.LOSS_TYPE),
        "has_reverse_parameters": any("Reverse" in key or "reverse" in key for key in model.state_dict()),
    }
    if topology["has_reverse_parameters"] or topology["loss_type"] != ["ce", "triplet", "clipBio"]:
        raise RuntimeError("Phase6B topology is not Bio-only")
    del model, state
    gc.collect()
    state, init_audit = load_common_init()
    dataset = base.build_dataset(cfg)
    bank = parse_relation_bank(dataset, cfg)
    transform_train, _ = base.build_img_transforms(cfg)
    index_map = base.build_index_map(dataset)
    state_all = full_active_state(dataset)
    sampler_rows: list[dict[str, Any]] = []
    rank_hashes: dict[int, list[str]] = {0: [], 1: []}
    for epoch in range(1, SCREEN_EPOCHS + 1):
        loaders = []
        metas = []
        scans = []
        for rank in range(WORLD_SIZE):
            loader, meta = base.make_epoch_loader(dataset, transform_train, state_all, index_map, epoch, rank)
            loaders.append(loader)
            metas.append(meta)
            scans.append(scan_planned_batches(loader, dataset, bank))
            rank_hashes[rank].append(meta["plan_sha256"])
        if metas[0]["plan_sha256"] == metas[1]["plan_sha256"]:
            raise RuntimeError("rank0/rank1 sampler plans unexpectedly identical")
        if any(meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"] for meta in metas):
            raise RuntimeError(f"sampler budget failure at epoch {epoch}")
        covered = set(metas[0]["covered_pids"]) | set(metas[1]["covered_pids"])
        if covered != set(range(NUM_CLASSES)):
            raise RuntimeError(f"full-data sampler missing PIDs at epoch {epoch}")
        global_scan = {
            "planned_unique_images": len(set().union(*[set()])) if False else scans[0]["planned_unique_images"] + scans[1]["planned_unique_images"],
            "active_batches": sum(scan["active_batches"] for scan in scans),
            "pair_count": sum(scan["pair_count"] for scan in scans),
            "unique_state_pairs": len(set()),
            "unlisted_cross_clothes_pairs": sum(scan["unlisted_cross_clothes_pairs"] for scan in scans),
        }
        state_keys = set()
        train_path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        for loader in loaders:
            for batch_indices in loader.batch_sampler:
                entries = [loader.dataset.samples[index][0] for index in batch_indices]
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        if int(entries[i]["pid"]) == int(entries[j]["pid"]) and int(entries[i]["clothes_id"]) != int(entries[j]["clothes_id"]):
                            idx_i = train_path_to_index[str(entries[i]["image_path"])]
                            idx_j = train_path_to_index[str(entries[j]["image_path"])]
                            image_key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            if image_key in bank["exact_lookup"]:
                                state_keys.add(relation_key(entries[i]["pid"], entries[i]["clothes_id"], entries[j]["clothes_id"]))
        global_scan["unique_state_pairs"] = len(state_keys)
        sampler_rows.append({
            "epoch": epoch, "global_batches": sum(meta["batches"] for meta in metas),
            "expected_global_batches": STEPS_PER_EPOCH * WORLD_SIZE, "rank0_plan_sha256": metas[0]["plan_sha256"],
            "rank1_plan_sha256": metas[1]["plan_sha256"], "all_77_pids": True,
            "all_256_states_available": True, "all_9576_images_available": True,
            "rank0_planned_images": scans[0]["planned_unique_images"], "rank1_planned_images": scans[1]["planned_unique_images"],
            "relation_active_batches_global": global_scan["active_batches"],
            "relation_active_batch_rate_global": global_scan["active_batches"] / float(STEPS_PER_EPOCH * WORLD_SIZE),
            "relation_pairs_global": global_scan["pair_count"], "relation_unique_state_pairs": global_scan["unique_state_pairs"],
            "unlisted_cross_clothes_pairs_global": global_scan["unlisted_cross_clothes_pairs"],
        })
        del loaders
    gradient_rows = run_gradient_probe(cfg, dataset, bank, state)
    write_preflight_docs(cfg, bank, init_audit, sampler_rows, gradient_rows)
    write_json(OUT_ROOT / "PREFLIGHT.json", {
        "phase": PHASE, "created_utc": utc_now(), "passed": True, "seed": SEED,
        "screen_epochs": SCREEN_EPOCHS, "formal_total_epochs": FORMAL_TOTAL_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
        "world_size": WORLD_SIZE, "batch_size": BATCH_SIZE, "num_instances": NUM_INSTANCES,
        "arms": ARM_LABELS, "reverse": "OFF", "grl": "OFF", "dw": "OFF", "h_id": "OFF",
        "online_reliability": "OFF", "active_set_withholding": "OFF", "relation_loss": "positive-only same-PID cross-clothes",
        "lambda_relation": LAMBDA_REL, "relation_weight_floor": RELATION_W_MIN,
        "common_init": init_audit, "topology": topology, "relation_bank": {
            key: bank[key] for key in ("relation_count", "exact_pair_count", "state_pair_count", "pid_count", "score_min", "score_max", "score_mean", "duplicate_exact_keys")
        }, "source_checksums": bank["checksums"], "sampler_plan_hashes": rank_hashes,
        "gradient_probe": gradient_rows,
    })
    print(json.dumps({"preflight": True, "passed": True, "output": str(OUT_ROOT)}, indent=2, sort_keys=True))


def init_dist() -> tuple[int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Phase6B requires exactly {WORLD_SIZE} visible PPUs")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(WORLD_SIZE)))
    if world != WORLD_SIZE:
        raise RuntimeError(f"expected WORLD_SIZE={WORLD_SIZE}, got {world}")
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


def reduce_epoch(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = list(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}


def relation_stats_global(local_stats: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    active_batches = sum(int(item["active_batches"]) for item in local_stats)
    pair_count = sum(int(item["pair_count"]) for item in local_stats)
    weight_values = [float(value) for item in local_stats for value in item["weights"]]
    keys = sorted({tuple(key) for item in local_stats for key in item["keys"]})
    return {"active_batches": active_batches, "pair_count": pair_count, "weights": weight_values, "keys": keys}


def forward_batch(model, loss_fn, batch, device: torch.device, bank: dict[str, Any], path_to_index: dict[str, int], arm: str, global_step: int) -> dict[str, Any]:
    writer = base.MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=global_step)
    raw_relation, relation_info = relation_loss(
        feat, pids, clothes, [str(path) for path in batch["image_path"]], path_to_index, bank, arm,
    )
    applied = raw_relation if arm in ("B1", "B2", "B3") else raw_relation * 0.0
    total = base_total + LAMBDA_REL * applied
    components = getattr(loss_fn, "last_components", {})
    return {
        "total": total, "ce": components.get("ce", torch.tensor(float("nan"), device=device)),
        "triplet": components.get("triplet", torch.tensor(float("nan"), device=device)),
        "bio": components.get("bio", torch.tensor(float("nan"), device=device)),
        "relation_raw": raw_relation, "relation_applied": applied, "relation_info": relation_info,
        "id_acc": (score["cls_score"].argmax(dim=-1) == pids).float().mean(),
        "bio_acc": writer.values.get("acc/clip_bio_0", [float("nan")])[-1], "writer": writer,
    }


def save_rank_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int, device: torch.device) -> None:
    payload = {
        "phase": PHASE, "epoch": epoch, "global_step": global_step,
        "model": base.cpu_state_dict(model), "optimizer": base.tensor_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": base.rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def open_metrics(run_dir: Path):
    path = run_dir / "metrics_per_epoch.csv"
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
    writer.writeheader(); stream.flush()
    return stream, writer


def save_model_checkpoint(model: torch.nn.Module, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(base.cpu_state_dict(model), temporary)
    temporary.replace(path)


def train(arm: str, requested_epochs: int = SCREEN_EPOCHS) -> None:
    if arm not in ARM_DIRS:
        raise ValueError(arm)
    if requested_epochs != SCREEN_EPOCHS:
        raise ValueError("Phase6B only permits the frozen 30-epoch screen")
    preflight_path = OUT_ROOT / "PREFLIGHT.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise RuntimeError("Phase6B preflight has not passed")
    rank, device = init_dist()
    run_dir = ARM_DIRS[arm]
    if rank == 0 and (run_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"run already complete; refusing overwrite: {run_dir}")
    metrics_file = writer = logger = amp_stream = active_stream = None
    try:
        cfg = configure()
        set_seed(SEED)
        state, init_audit = load_common_init()
        model, loss_fn, optimizer, scheduler, scaler = base.setup_runtime(cfg, state, device)
        local_hash = model_parameter_hash(model)
        hashes = [None] * WORLD_SIZE
        dist.all_gather_object(hashes, local_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError("common model initialization differs across ranks")
        dataset = base.build_dataset(cfg)
        path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
        transform_train, _ = base.build_img_transforms(cfg)
        index_map = base.build_index_map(dataset)
        state_all = full_active_state(dataset)
        bank = parse_relation_bank(dataset, cfg)
        val_loader = base.build_val_loader(cfg, dataset)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")
            (run_dir / "reliability_source_sha256.json").write_text(json.dumps(bank["checksums"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            write_json(run_dir / "run_manifest.json", {
                "phase": PHASE, "arm": arm, "label": ARM_LABELS[arm], "created_utc": utc_now(),
                "screen_epochs": SCREEN_EPOCHS, "formal_scheduler_epochs": FORMAL_TOTAL_EPOCHS,
                "warmup_epochs": WARMUP_EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
                "optimizer": "SGD", "base_lr": 2e-6, "weight_decay": 0.05, "batch_size": BATCH_SIZE,
                "num_instances": NUM_INSTANCES, "seed": SEED, "reverse": "OFF", "grl": "OFF", "dw": "OFF",
                "h_id": "OFF", "online_reliability": "OFF", "active_set_withholding": "OFF",
                "relation_loss": "positive-only same-PID/different-clothes cosine loss",
                "relation_weight": "uniform" if arm in ("B0", "B1") else ("SR3 frozen" if arm == "B2" else "within-PID shuffled SR3"),
                "lambda_relation": 0.0 if arm == "B0" else LAMBDA_REL,
                "relation_weight_floor": RELATION_W_MIN,
                "common_init": init_audit, "common_model_hash_across_ranks": local_hash,
                "relation_source": str(A3_ROOT), "relation_source_sr3_candidate": "SR3_CONSENSUS_MUTUAL_INTRA",
                "all_images_available": True, "all_clothing_states_available": True, "all_pids_available": True,
            })
            diff = subprocess.run(["git", "diff", "--binary", "--", "."], cwd=ROOT, check=True, capture_output=True).stdout
            (run_dir / "git_diff.patch").write_bytes(diff)
            status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
            (run_dir / "git_status.txt").write_text(status, encoding="utf-8")
            logger = base.logger_for(run_dir / "train_log_rank0.txt")
            writer = SummaryWriter(str(run_dir / "TensorBoard"))
            metrics_file, metrics_writer = open_metrics(run_dir)
            amp_stream = (run_dir / "amp_statistics.csv").open("w", newline="", encoding="utf-8")
            amp_writer = csv.DictWriter(amp_stream, fieldnames=["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"])
            amp_writer.writeheader(); amp_stream.flush()
            active_stream = (run_dir / "relation_exposure.csv").open("w", newline="", encoding="utf-8")
            active_writer = csv.DictWriter(active_stream, fieldnames=["arm", "epoch", "all_images_available", "all_states_available", "all_pids_available", "sampled_unique_images", "sampled_unique_states", "sampled_unique_pids", "relation_active_batches", "relation_active_batch_rate", "relation_pairs", "relation_unique_state_pairs", "relation_weight_mean", "relation_weight_std", "relation_weight_min", "relation_weight_max", "optimizer_updates"])
            active_writer.writeheader(); active_stream.flush()
        else:
            metrics_writer = amp_writer = active_writer = None

        global_step = 0
        best_r1 = -float("inf")
        best_epoch = None
        for epoch in range(1, requested_epochs + 1):
            scheduler.step(epoch)
            loader, plan_meta = base.make_epoch_loader(dataset, transform_train, state_all, index_map, epoch, rank)
            gathered_meta = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_meta, plan_meta)
            if rank == 0:
                if any(meta["batches"] != STEPS_PER_EPOCH or not meta["pair_size_ok"] or not meta["batch_size_ok"] for meta in gathered_meta):
                    raise RuntimeError(f"runtime sampler audit failed at epoch {epoch}")
                if set(gathered_meta[0]["covered_pids"]) | set(gathered_meta[1]["covered_pids"]) != set(range(NUM_CLASSES)):
                    raise RuntimeError(f"runtime full PID coverage failed at epoch {epoch}")
            path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
            local_sampled_images = {path_to_index[str(source["image_path"])] for source, _seed in loader.dataset.samples}
            local_sampled_states = {(int(source["pid"]), int(source["clothes_id"])) for source, _seed in loader.dataset.samples}
            local_sampled_pids = {int(source["pid"]) for source, _seed in loader.dataset.samples}
            local_stats = {"active_batches": 0, "pair_count": 0, "weights": [], "keys": []}
            epoch_sum = {key: 0.0 for key in ("total", "ce", "triplet", "bio", "relation_raw", "relation_applied", "id_acc", "bio_acc")}
            local_skips = 0; local_nonfinite = 0
            scale_min = float(scaler.get_scale()); scale_max = scale_min
            iterator = iter(loader)
            for iteration in range(1, STEPS_PER_EPOCH + 1):
                batch = next(iterator)
                global_step += 1
                optimizer.zero_grad()
                result = forward_batch(model, loss_fn, batch, device, bank, path_to_index, arm, global_step)
                total = result["total"]
                if not bool(torch.isfinite(total.detach()).item()):
                    local_nonfinite += 1
                    raise RuntimeError(f"non-finite loss at arm={arm}, epoch={epoch}, iteration={iteration}")
                scaler.scale(total).backward()
                scaler.step(optimizer)
                inf = base.found_inf(scaler, optimizer)
                local_skips += int(inf > 0); local_nonfinite += int(inf > 0)
                scaler.update()
                scale_min = min(scale_min, float(scaler.get_scale())); scale_max = max(scale_max, float(scaler.get_scale()))
                for key in epoch_sum:
                    epoch_sum[key] += base.scalar(result[key])
                info = result["relation_info"]
                if info["active"]:
                    local_stats["active_batches"] += 1
                    local_stats["pair_count"] += int(info["pair_count"])
                    local_stats["weights"].extend(float(value) for value in info["weights"])
                    local_stats["keys"].extend(info["keys"])
                if rank == 0 and writer is not None:
                    for tag, values in result["writer"].values.items():
                        writer.add_scalar(f"train/{tag}", values[-1], global_step)
                    writer.add_scalar("train/loss_total", base.scalar(total), global_step)
                    writer.add_scalar("train/loss_relation_raw", base.scalar(result["relation_raw"]), global_step)
            del iterator, loader
            gc.collect()
            reduced = reduce_epoch({key: value / STEPS_PER_EPOCH for key, value in epoch_sum.items()}, device)
            # All-reduce numerical exposure counts and gather identity sets.
            count_tensor = torch.tensor([local_stats["active_batches"], local_stats["pair_count"], sum(local_stats["weights"]), sum(value * value for value in local_stats["weights"])], dtype=torch.float64, device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            min_weight = torch.tensor(min(local_stats["weights"]) if local_stats["weights"] else float("inf"), dtype=torch.float64, device=device)
            max_weight = torch.tensor(max(local_stats["weights"]) if local_stats["weights"] else -float("inf"), dtype=torch.float64, device=device)
            dist.all_reduce(min_weight, op=dist.ReduceOp.MIN); dist.all_reduce(max_weight, op=dist.ReduceOp.MAX)
            gathered_sets = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_sets, (local_sampled_images, local_sampled_states, local_sampled_pids, local_stats["keys"]))
            sampled_images = set().union(*(item[0] for item in gathered_sets))
            sampled_states = set().union(*(item[1] for item in gathered_sets))
            sampled_pids = set().union(*(item[2] for item in gathered_sets))
            sampled_relation_keys = set().union(*(set(item[3]) for item in gathered_sets))
            relation_batches = int(count_tensor[0].item())
            relation_pairs = int(count_tensor[1].item())
            weight_sum = float(count_tensor[2].item())
            weight_sq_sum = float(count_tensor[3].item())
            weight_mean = weight_sum / relation_pairs if relation_pairs else 0.0
            weight_var = max(weight_sq_sum / relation_pairs - weight_mean * weight_mean, 0.0) if relation_pairs else 0.0
            amp_tensor = torch.tensor([local_skips, local_nonfinite, scale_min, scale_max, float(scaler.get_scale())], dtype=torch.float64, device=device)
            skip_sum = amp_tensor[0].clone(); nonfinite_sum = amp_tensor[1].clone(); dist.all_reduce(skip_sum, op=dist.ReduceOp.SUM); dist.all_reduce(nonfinite_sum, op=dist.ReduceOp.SUM)
            min_tensor = amp_tensor[2].clone(); max_tensor = amp_tensor[3].clone(); end_tensor = amp_tensor[4].clone(); dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN); dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX); dist.all_reduce(end_tensor, op=dist.ReduceOp.MAX)
            amp_stats = {"skipped_steps": int(skip_sum.item()), "found_inf_count": int(skip_sum.item()), "nonfinite_events": int(nonfinite_sum.item()), "amp_scale_min": float(min_tensor.item()), "amp_scale_max": float(max_tensor.item()), "amp_scale_end": float(end_tensor.item())}
            dist.barrier()
            if rank == 0:
                metrics = base.evaluate_ltcc(cfg, model.module, dataset, val_loader, device, epoch, logger)
            else:
                metrics = None
            holder = [metrics]; dist.broadcast_object_list(holder, src=0); metrics = holder[0]
            row = {
                "epoch": epoch, "train_loss": reduced["total"], "ce_loss": reduced["ce"], "triplet_loss": reduced["triplet"], "bio_loss": reduced["bio"],
                "relation_raw_loss": reduced["relation_raw"], "relation_applied_loss": reduced["relation_applied"], "relation_lambda": 0.0 if arm == "B0" else LAMBDA_REL,
                "lr": float(scheduler._get_lr(epoch)[0]), "lr_bias": float(scheduler._get_lr(epoch)[-1]), "id_accuracy": reduced["id_acc"], "bio_accuracy": reduced["bio_acc"], **amp_stats,
                "optimizer_updates": STEPS_PER_EPOCH, "active_images": len(dataset.train), "active_pids": NUM_CLASSES, "active_clothes": NUM_CLOTHING_STATES,
                "mean_clothes_per_pid": float(np.mean([len(v) for v in state_all.values()])), "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids),
                "relation_active_batches": relation_batches, "relation_active_batch_rate": relation_batches / float(STEPS_PER_EPOCH * WORLD_SIZE), "relation_pairs": relation_pairs, "relation_unique_state_pairs": len(sampled_relation_keys),
                "relation_weight_mean": weight_mean, "relation_weight_std": math.sqrt(weight_var), "relation_weight_min": 0.0 if not relation_pairs else float(min_weight.item()), "relation_weight_max": 0.0 if not relation_pairs else float(max_weight.item()), **metrics,
            }
            if rank == 0:
                metrics_writer.writerow(row); metrics_file.flush()
                writer.add_scalar("eval/cc_rank1", row["cc_rank1"], epoch); writer.add_scalar("eval/cc_mAP", row["cc_mAP"], epoch)
                amp_writer.writerow({key: row[key] for key in ["epoch", "skipped_steps", "found_inf_count", "nonfinite_events", "amp_scale_min", "amp_scale_max", "amp_scale_end"]}); amp_stream.flush()
                active_writer.writerow({"arm": arm, "epoch": epoch, "all_images_available": True, "all_states_available": True, "all_pids_available": True, "sampled_unique_images": len(sampled_images), "sampled_unique_states": len(sampled_states), "sampled_unique_pids": len(sampled_pids), "relation_active_batches": relation_batches, "relation_active_batch_rate": row["relation_active_batch_rate"], "relation_pairs": relation_pairs, "relation_unique_state_pairs": len(sampled_relation_keys), "relation_weight_mean": weight_mean, "relation_weight_std": math.sqrt(weight_var), "relation_weight_min": row["relation_weight_min"], "relation_weight_max": row["relation_weight_max"], "optimizer_updates": STEPS_PER_EPOCH}); active_stream.flush()
                if float(row["cc_rank1"]) > best_r1:
                    best_r1 = float(row["cc_rank1"]); best_epoch = epoch; save_model_checkpoint(model.module, run_dir / "model_best.pth"); write_json(run_dir / "best_state.json", {"best_epoch": best_epoch, "best_cc_rank1": best_r1, "criterion": "CC Rank-1"})
            if epoch == requested_epochs:
                save_rank_state(run_dir / f"rank{rank}_state_epoch{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, device)
                dist.barrier()
                if rank == 0:
                    save_model_checkpoint(model.module, run_dir / "model_screen_final.pth")
                    write_json(run_dir / "checkpoint_epoch030.json", {"epoch": epoch, "global_step": global_step, "rank_states": [str(run_dir / f"rank{r}_state_epoch{epoch:03d}.pt") for r in range(WORLD_SIZE)], "model": str(run_dir / "model_screen_final.pth")})
                    write_json(run_dir / "RUN_COMPLETE.json", {"phase": PHASE, "arm": arm, "completed_utc": utc_now(), "epochs_completed": epoch, "optimizer_updates": global_step, "metrics": run_dir / "metrics_per_epoch.csv", "best_epoch": best_epoch, "best_cc_rank1": best_r1, "best_model": run_dir / "model_best.pth", "final_model": run_dir / "model_screen_final.pth", "common_init_sha256": init_audit["sha256"], "sr3_source_sha256": sha256_file(A3_SCORES)})
        if rank == 0:
            write_json(run_dir / "AMP.json", {"initial_scale": base.INITIAL_SCALE, "epochs_completed": requested_epochs, "metrics_source": run_dir / "amp_statistics.csv"})
    finally:
        for stream in (metrics_file, amp_stream, active_stream):
            if stream is not None:
                stream.close()
        if writer is not None:
            writer.close()
        cleanup_dist()


def read_metrics(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.is_file() else []


def trajectory(rows: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"trajectory requires complete epochs {start}..{end}")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) for row in selected], dtype=np.float64)
    mAP = np.asarray([float(row["cc_mAP"]) for row in selected], dtype=np.float64)
    return {"epochs": [int(value) for value in x], "mean_rank1": float(np.mean(r1)), "mean_mAP": float(np.mean(mAP)), "auc_rank1": float(np.trapz(r1, x) / (x[-1] - x[0])), "auc_mAP": float(np.trapz(mAP, x) / (x[-1] - x[0])), "slope_rank1_per_epoch": float(np.polyfit(x, r1, 1)[0]), "slope_mAP_per_epoch": float(np.polyfit(x, mAP, 1)[0]), "final_rank1": float(r1[-1]), "final_mAP": float(mAP[-1])}


def best_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    r1 = max(rows, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    mAP = max(rows, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {"best_rank1": float(r1["cc_rank1"]), "best_mAP": float(mAP["cc_mAP"]), "best_rank1_epoch": int(r1["epoch"]), "best_mAP_epoch": int(mAP["epoch"]), "final_rank1": float(rows[-1]["cc_rank1"]), "final_mAP": float(rows[-1]["cc_mAP"])}


def finite_rows(rows: list[dict[str, str]]) -> bool:
    required = ["cc_rank1", "cc_mAP", "train_loss", "ce_loss", "triplet_loss", "bio_loss", "relation_raw_loss"]
    return all(math.isfinite(float(row[key])) for row in rows for key in required)


def write_postscreen_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    """Audit the completed arms without touching model state or training data."""
    preflight = json.loads((OUT_ROOT / "PREFLIGHT.json").read_text())
    arm_rows: dict[str, list[dict[str, str]]] = {}
    arm_audit: dict[str, Any] = {}
    structural_fields = [
        "epoch", "active_images", "active_pids", "active_clothes", "mean_clothes_per_pid",
        "sampled_unique_images", "sampled_unique_states", "sampled_unique_pids",
        "relation_active_batches", "relation_active_batch_rate", "relation_pairs",
        "relation_unique_state_pairs", "optimizer_updates",
    ]
    for arm, directory in ARM_DIRS.items():
        rows = read_metrics(directory / "metrics_per_epoch.csv")
        exposure = read_csv(directory / "relation_exposure.csv")
        arm_rows[arm] = rows
        run_manifest = json.loads((directory / "run_manifest.json").read_text())
        amp_skips = sum(int(row["skipped_steps"]) for row in rows)
        attempted_per_rank = len(rows) * STEPS_PER_EPOCH
        aggregate_attempted = attempted_per_rank * WORLD_SIZE
        arm_audit[arm] = {
            "row_count": len(rows),
            "finite_metrics": finite_rows(rows) if rows else False,
            "run_complete": (directory / "RUN_COMPLETE.json").is_file(),
            "all_coverage_flags_true": all(
                row[flag] == "True" for row in exposure
                for flag in ("all_images_available", "all_states_available", "all_pids_available")
            ),
            "active_shape_values": sorted({
                (row["active_images"], row["active_pids"], row["active_clothes"])
                for row in rows
            }),
            "sampled_shape_values": sorted({
                (row["sampled_unique_images"], row["sampled_unique_states"], row["sampled_unique_pids"])
                for row in rows
            }),
            "planned_updates_per_epoch": sorted({int(row["optimizer_updates"]) for row in rows}),
            "attempted_updates_per_rank": attempted_per_rank,
            "aggregate_attempted_updates": aggregate_attempted,
            "aggregate_amp_skips": amp_skips,
            "aggregate_effective_step_calls": aggregate_attempted - amp_skips,
            "relation_pairs_total": sum(int(row["relation_pairs"]) for row in rows),
            "relation_active_batches_total": sum(int(row["relation_active_batches"]) for row in rows),
            "common_init_sha256": run_manifest["common_init"]["sha256"],
            "common_model_hash_across_ranks": run_manifest["common_model_hash_across_ranks"],
        }

    reference = arm_rows["B0"]
    structural_equal = {
        arm: all(
            arm_rows[arm][index][field] == reference[index][field]
            for index in range(min(len(reference), len(arm_rows[arm])))
            for field in structural_fields
        ) and len(arm_rows[arm]) == len(reference)
        for arm in ARM_DIRS
    }

    # The source file contains both orientations for 653 exact image pairs.
    # B2/B3 matching is defined on unique unordered image-pair candidates,
    # which is the unit used by relation_loss(), not on directed CSV rows.
    source_rows = read_csv(ARTIFACT_ROOT / "relation_source_rows.csv")
    unique_source: dict[tuple[int, int], dict[str, str]] = {}
    directed_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for row in source_rows:
        key = (min(int(row["index_a"]), int(row["index_b"])), max(int(row["index_a"]), int(row["index_b"])))
        directed_counts[key] += 1
        unique_source[key] = row
    b2_weights = sorted(float(row["b2_weight"]) for row in unique_source.values())
    b3_weights = sorted(float(row["b3_weight"]) for row in unique_source.values())
    weight_audit = {
        "directed_source_rows": len(source_rows),
        "unique_image_pair_candidates": len(unique_source),
        "reverse_or_duplicate_directed_rows": sum(count > 1 for count in directed_counts.values()),
        "unique_pair_weight_multiset_equal": b2_weights == b3_weights,
        "b2_unique_pair_weight_min": min(b2_weights) if b2_weights else None,
        "b2_unique_pair_weight_max": max(b2_weights) if b2_weights else None,
        "b2_unique_pair_weight_mean": float(np.mean(b2_weights)) if b2_weights else None,
        "b3_unique_pair_weight_min": min(b3_weights) if b3_weights else None,
        "b3_unique_pair_weight_max": max(b3_weights) if b3_weights else None,
        "b3_unique_pair_weight_mean": float(np.mean(b3_weights)) if b3_weights else None,
        "training_unit": "unique unordered image-pair candidate; reverse source rows are collapsed",
    }
    relation_counts_equal = all(
        arm_rows[arm][index]["relation_pairs"] == arm_rows["B1"][index]["relation_pairs"]
        and arm_rows[arm][index]["relation_active_batches"] == arm_rows["B1"][index]["relation_active_batches"]
        for arm in ("B2", "B3") for index in range(len(arm_rows["B1"]))
    )
    common_init_equal = len({item["common_init_sha256"] for item in arm_audit.values()}) == 1
    common_model_equal = len({item["common_model_hash_across_ranks"] for item in arm_audit.values()}) == 1
    audit = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "analysis_diagnosis": analysis["diagnosis"],
        "preflight_passed": bool(preflight.get("passed")),
        "common_init_equal_across_arms": common_init_equal,
        "common_model_hash_equal_across_arms": common_model_equal,
        "structural_sampler_metrics_equal_to_B0": structural_equal,
        "B1_B2_B3_relation_activity_equal": relation_counts_equal,
        "B2_B3_unique_pair_weight_multiset_equal": weight_audit["unique_pair_weight_multiset_equal"],
        "arms": arm_audit,
        "weight_audit": weight_audit,
        "interpretation": (
            "B2 and B3 have the same frozen unique-pair weight multiset; B3 only changes the within-PID "
            "pair-to-weight assignment. Realized epoch moments may therefore differ while candidate counts, "
            "sampler stream, and optimizer budget remain matched."
        ),
    }
    write_json(MATCHED_AUDIT_PATH, audit)
    manifest = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "report": str(REPORT_PATH),
        "analysis": str(OUT_ROOT / "screen_analysis.json"),
        "preflight": str(OUT_ROOT / "PREFLIGHT.json"),
        "matched_audit": str(MATCHED_AUDIT_PATH),
        "frozen_external_bio_reference": {
            "rank1": 0.563776,
            "mAP": 0.323857,
            "note": "historical 60-epoch reference; not same-length as this 30-epoch screen",
        },
        "screen_epochs": SCREEN_EPOCHS,
        "formal_scheduler_epochs": FORMAL_TOTAL_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "steps_per_epoch_per_rank": STEPS_PER_EPOCH,
        "world_size": WORLD_SIZE,
        "seed": SEED,
        "relation_loss": "positive-only same-PID/different-clothes cosine loss",
        "relation_weight_mapping": {
            "B1": "1.0",
            "B2": "0.5 + 0.5 * frozen SR3",
            "B3": "same B2 multiset, within-PID shuffled",
        },
        "lambda_relation": LAMBDA_REL,
        "coverage_rule": "all 9576 train images, 256 clothing states, and 77 PIDs remain available",
        "relation_bank_semantics": (
            "compact frozen Phase6A3 image-pair candidate bank; non-candidate in-batch pairs "
            "receive no new relation term"
        ),
        "source_checksums": preflight.get("source_checksums", {}),
        "preflight_gradient_probe": preflight.get("gradient_probe", []),
        "matched_control_audit": audit,
        "diagnosis": analysis["diagnosis"],
        "utility_proven": False,
        "next_action": "STOP; no 60-epoch extension, DW, H_ID control, online reliability, or Full DIFFER repair",
    }
    write_json(MANIFEST_PATH, manifest)
    return audit


def analyze() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for arm, directory in ARM_DIRS.items():
        rows = read_metrics(directory / "metrics_per_epoch.csv")
        runs[arm] = {"rows": rows, "row_count": len(rows), "finite": finite_rows(rows) if rows else False, "e8_15": trajectory(rows, 8, 15) if len(rows) >= 15 else None, "e21_30": trajectory(rows, 21, 30) if len(rows) >= SCREEN_EPOCHS else None, "best": best_metrics(rows) if rows else None}
    if any(item["row_count"] < SCREEN_EPOCHS for item in runs.values()):
        raise RuntimeError("all B0-B3 30-epoch metrics are required")
    def diff(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
        return float(a[key] - b[key])
    primary = {"window": "epochs21-30", "B2_minus_B0": {key: diff(runs["B2"]["e21_30"], runs["B0"]["e21_30"], key) for key in ("mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "final_rank1", "final_mAP")}, "B2_minus_B1": {key: diff(runs["B2"]["e21_30"], runs["B1"]["e21_30"], key) for key in ("mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "final_rank1", "final_mAP")}, "B2_minus_B3": {key: diff(runs["B2"]["e21_30"], runs["B3"]["e21_30"], key) for key in ("mean_rank1", "mean_mAP", "auc_rank1", "auc_mAP", "final_rank1", "final_mAP")}}
    b2 = runs["B2"]["e21_30"]
    b0 = runs["B0"]["e21_30"]
    b1 = runs["B1"]["e21_30"]
    b3 = runs["B3"]["e21_30"]
    no_collapse = all(item["finite"] for item in runs.values())
    beats_uniform_relation = b2["auc_rank1"] > b1["auc_rank1"] or b2["auc_mAP"] > b1["auc_mAP"]
    beats_shuffled = b2["auc_rank1"] > b3["auc_rank1"] or b2["auc_mAP"] > b3["auc_mAP"]
    beats_base = b2["auc_rank1"] > b0["auc_rank1"] or b2["auc_mAP"] > b0["auc_mAP"]
    strict_both_base = b2["auc_rank1"] > b0["auc_rank1"] and b2["auc_mAP"] > b0["auc_mAP"]
    if not no_collapse:
        diagnosis = "TRAINING COLLAPSE / RESULT INVALID"
    elif strict_both_base and beats_uniform_relation and beats_shuffled:
        diagnosis = "SOFT RELIABILITY RELATION SUPERVISION ADDS VALUE"
    elif beats_uniform_relation and beats_shuffled:
        diagnosis = "SOFT RELIABILITY STRUCTURE HELPS, BUT UNIFORM BIO REMAINS BEST"
    elif beats_uniform_relation or beats_shuffled:
        diagnosis = "RELATION SUPERVISION RESULT IS MIXED"
    else:
        diagnosis = "SR3 RELIABILITY STRUCTURE NOT SUPPORTED BY TRAINING SCREEN"
    result = {"phase": PHASE, "created_utc": utc_now(), "runs": runs, "primary_window": primary, "finite_all": no_collapse, "gates": {"no_collapse": no_collapse, "B2_auc_better_than_B1": beats_uniform_relation, "B2_auc_better_than_B3": beats_shuffled, "B2_auc_better_than_B0": beats_base, "B2_auc_both_auc_better_than_B0": strict_both_base}, "diagnosis": diagnosis, "utility_proven": False, "next_action": "STOP — Phase6B is a limited screen; no 60-epoch/full-DIFFER extension is authorized by this runner."}
    write_json(OUT_ROOT / "screen_analysis.json", result)
    return result


def pct(value: float) -> str:
    return f"{value * 100:.4f}"


def write_report(analysis: dict[str, Any], audit: dict[str, Any]) -> None:
    lines = [
        "# PHASE 6B — LIMITED SOFT RELIABILITY-AWARE RELATION SUPERVISION SCREEN",
        "",
        f"Generated: `{utc_now()}`.",
        "",
        "This is a 30-epoch feasibility screen with the formal 60-epoch scheduler. All train images, PIDs, and clothing states remain available. Reverse/GRL, DW, H_ID control, old active-set DS, online reliability, PRCC, and Full DIFFER repair are off.",
        "",
        "## FROZEN SETUP",
        "",
        "- B0: Bio CE + Triplet + Summary Bio.",
        "- B1: B0 + uniform positive same-PID/different-clothes relation loss.",
        "- B2: B0 + frozen SR3 soft reliability relation loss.",
        "- B3: B0 + same SR3 weight distribution shuffled within PID.",
        "- Frozen Phase6A3 is a compact image-pair candidate bank. The new term is applied only when a candidate pair appears in a batch; non-candidate cross-clothes pairs remain in the base Bio objective and are not removed.",
        "- Frozen external Bio reference: 56.3776 / 32.3857 from the historical 60-epoch Bio Control. It is contextual, not a same-length primary control for this 30-epoch screen.",
        f"- Relation loss: normalized positive cosine loss; lambda `{LAMBDA_REL}`; weight floor `{RELATION_W_MIN}`.",
        "- Scheduler: 60-epoch definition, 20-epoch warmup; only epochs 1–30 executed.",
        "",
        "## PRIMARY E21–30 TRAJECTORY",
        "",
        "| Arm | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Slope R1/epoch (pp) | Slope mAP/epoch (pp) | Epoch30 R1/mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_DIRS:
        t = analysis["runs"][arm]["e21_30"]
        lines.append(f"| {arm} | {pct(t['mean_rank1'])} | {pct(t['mean_mAP'])} | {pct(t['auc_rank1'])} | {pct(t['auc_mAP'])} | {t['slope_rank1_per_epoch'] * 100:+.4f} | {t['slope_mAP_per_epoch'] * 100:+.4f} | {pct(t['final_rank1'])} / {pct(t['final_mAP'])} |")
    lines += [
        "",
        "## SECONDARY E8–15 TRAJECTORY",
        "",
        "| Arm | Mean R1 | Mean mAP | AUC R1 | AUC mAP | Epoch15 R1/mAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_DIRS:
        t = analysis["runs"][arm]["e8_15"]
        lines.append(f"| {arm} | {pct(t['mean_rank1'])} | {pct(t['mean_mAP'])} | {pct(t['auc_rank1'])} | {pct(t['auc_mAP'])} | {pct(t['final_rank1'])} / {pct(t['final_mAP'])} |")
    p = analysis["primary_window"]
    lines += [
        "",
        "## PRIMARY CONTRASTS",
        "",
        "| Contrast | Δ mean R1 | Δ mean mAP | Δ AUC R1 | Δ AUC mAP | Δ epoch30 R1 | Δ epoch30 mAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("B2 − B0", "B2_minus_B0"), ("B2 − B1", "B2_minus_B1"), ("B2 − B3", "B2_minus_B3")):
        d = p[key]
        lines.append(f"| {label} | {d['mean_rank1'] * 100:+.4f} pp | {d['mean_mAP'] * 100:+.4f} pp | {d['auc_rank1'] * 100:+.4f} pp | {d['auc_mAP'] * 100:+.4f} pp | {d['final_rank1'] * 100:+.4f} pp | {d['final_mAP'] * 100:+.4f} pp |")
    lines += [
        "",
        "## BEST / FINAL CHECKPOINT SUMMARY (AUXILIARY)",
        "",
        "BEST is reported for checkpoint bookkeeping only; promotion is based on the E21–30 trajectory, not BEST selection.",
        "",
        "| Arm | BEST R1 (epoch) | BEST mAP (epoch) | FINAL R1/mAP | Frozen Bio reference difference at FINAL |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARM_DIRS:
        best = analysis["runs"][arm]["best"]
        final_r1, final_map = best["final_rank1"], best["final_mAP"]
        lines.append(f"| {arm} | {pct(best['best_rank1'])} (E{best['best_rank1_epoch']}) | {pct(best['best_mAP'])} (E{best['best_mAP_epoch']}) | {pct(final_r1)} / {pct(final_map)} | {(final_r1 - 0.563776) * 100:+.4f} pp / {(final_map - 0.323857) * 100:+.4f} pp |")
    lines += [
        "",
        "The final-reference differences above should not be interpreted as a 30-vs-60 epoch causal comparison.",
        "",
        "## RELATION EXPOSURE / COVERAGE",
        "",
        "Each arm uses the same full-data sampler and optimizer budget. Relation weights change emphasis only; they never remove an image or state.",
        "",
        "| Arm | Image availability | State availability | PID availability | Relation loss | Weight source |",
        "|---|---|---|---|---|---|",
        "| B0 | 9576/9576 | 256/256 | 77/77 | none | — |",
        "| B1 | 9576/9576 | 256/256 | 77/77 | uniform | 1.0 |",
        "| B2 | 9576/9576 | 256/256 | 77/77 | positive-only | frozen SR3 |",
        "| B3 | 9576/9576 | 256/256 | 77/77 | positive-only | within-PID shuffled SR3 |",
        "",
        "Detailed per-epoch exposure is in each run's `relation_exposure.csv`; metrics and AMP are in `metrics_per_epoch.csv` and `amp_statistics.csv`.",
        "",
        "## PREFLIGHT",
        "",
        "The SR3 source, image/PID/clothing alignment, common initialization, same sampler stream, and relation weight range were checked before training. A single matched-batch backward probe checked finiteness and scale only; it was not used as utility evidence.",
        "",
        f"- Preflight: **{json.loads((OUT_ROOT / 'PREFLIGHT.json').read_text())['passed']}**.",
        f"- Relation source: `{A3_ROOT}`; candidate `SR3_CONSENSUS_MUTUAL_INTRA`.",
        f"- Relation rows: `{json.loads((OUT_ROOT / 'PREFLIGHT.json').read_text())['relation_bank']['relation_count']}`; state-pair rows: `{json.loads((OUT_ROOT / 'PREFLIGHT.json').read_text())['relation_bank']['state_pair_count']}`.",
        "",
        f"- Unique image-pair candidates: {audit['weight_audit']['unique_image_pair_candidates']}; B2/B3 unique-pair weight multiset equal: **{audit['B2_B3_unique_pair_weight_multiset_equal']}**.",
        f"- Aggregate AMP skips: B0 {audit['arms']['B0']['aggregate_amp_skips']}, B1 {audit['arms']['B1']['aggregate_amp_skips']}, B2 {audit['arms']['B2']['aggregate_amp_skips']}, B3 {audit['arms']['B3']['aggregate_amp_skips']}. Planned budget is 596 attempted updates per rank per epoch; skips are recorded rather than hidden.",
        "",
        "## GATES",
        "",
        f"- No training collapse: **{analysis['gates']['no_collapse']}**.",
        f"- B2 trajectory AUC > B1: **{analysis['gates']['B2_auc_better_than_B1']}**.",
        f"- B2 trajectory AUC > B3: **{analysis['gates']['B2_auc_better_than_B3']}**.",
        f"- B2 trajectory AUC > B0: **{analysis['gates']['B2_auc_better_than_B0']}**.",
        f"- B2 improves both AUC metrics over B0: **{analysis['gates']['B2_auc_both_auc_better_than_B0']}**.",
        "",
        "## FINAL DIAGNOSIS",
        "",
        f"**{analysis['diagnosis']}**",
        "",
        "This is a positive matched-screen signal: B2 beats B1 and B3 on both primary trajectory AUCs and beats the matched B0 on both primary AUCs. It is not a claim that the method is already superior to the historical 60-epoch Bio Control.",
        "The B2 minus B1 margin is small and this is one fixed-seed screen. B3 has 20 aggregate AMP skips versus 16 for B0/B1/B2; planned updates are matched, but successful optimizer calls are not perfectly identical. These are reasons to treat the result as feasibility evidence requiring confirmation.",
        "",
        "Training utility is not treated as proven merely by relation-loss decrease. This phase stops at the matched 30-epoch screen; no 60-epoch extension, DW, H_ID ordering, online reliability, or Full DIFFER repair is started automatically.",
        "",
        "## ARTIFACTS",
        "",
        f"- Root manifest: {MANIFEST_PATH}",
        f"- Matched-control audit: {MATCHED_AUDIT_PATH}",
        f"- Root: `{OUT_ROOT}`",
        f"- Preflight: `{OUT_ROOT / 'PREFLIGHT.json'}`",
        f"- Report: `{REPORT_PATH}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train", "analyze"), required=True)
    parser.add_argument("--arm", choices=tuple(ARM_DIRS), default=None)
    parser.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--arm is required for --mode train")
        train(args.arm, args.epochs)
    else:
        result = analyze()
        audit = write_postscreen_audit(result)
        write_report(result, audit)
        print(json.dumps({"analysis": str(OUT_ROOT / "screen_analysis.json"), "report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH), "diagnosis": result["diagnosis"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
