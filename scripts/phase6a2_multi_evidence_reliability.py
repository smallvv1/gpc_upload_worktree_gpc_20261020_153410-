#!/usr/bin/env python3
"""Phase 6A2: train-only multi-evidence relation reliability audit.

This script is diagnostic only. It reads frozen LTCC train feature banks and
the existing Phase5D gradient-probe artifact. It does not construct a model,
optimizer, sampler, loss, or backward pass. PID labels define relation truth;
the evidence columns remain separate from that label.

The compact primary pair bank contains one label-only deterministic image per
anchor and target clothing state. Selection never uses a feature or a result.
The full per-image cross-clothes relation summaries are retained separately.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase6a_bio_relation_teacher as p6a  # noqa: E402

SEED = 1234
BOOTSTRAP_RESAMPLES = 10_000
IMAGE_COUNT = 9_576
PID_COUNT = 77
STATE_COUNT = 256
TOP_K = 10
TRAIN_ROOT = Path("/home/data/LTCC_ReID/train")

P4R_ROOT = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features"
P5D_ROOT = ROOT / "outputs/new_method_track/phase5d_hardness_utility"
P5D_BANK = P5D_ROOT / "feature_bank/feature_bank.npz"
P5D_UTILITY = P5D_ROOT / "gradient_probes/gradient_utility_targets.csv"
P5D_QUADRANTS = P5D_ROOT / "gradient_probes/hardness_utility_quadrants.csv"
P5D_STABILITY = P5D_ROOT / "checkpoint_stability/ranking_stability.json"

BIO_CHECKPOINT = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/eva02_l_bio_best.pth"
ID_CHECKPOINT = ROOT / "outputs/reproduction_phase2/LTCC/baseline/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_BASELINE_RUN1]/eva02_l_bio_best.pth"
FAILED_CHECKPOINT = ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/eva02_l_bio_best.pth"
OFFICIAL_CHECKPOINT = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
ID_BANK = ROOT / "outputs/new_method_track/phase6a_bio_relation_teacher/feature_banks/id_only_feature_bank.npz"

OUT_ROOT = ROOT / "outputs/new_method_track/phase6a2_multi_evidence_reliability"
REPORT_PATH = ROOT / "reports/PHASE_6A2_MULTI_EVIDENCE_RELIABLE_RELATION.md"

BANK_PATHS = {
    "Bio": P4R_ROOT / "bio_control_feature_bank.npz",
    "ID-only": ID_BANK,
    "Local Failed Full": P4R_ROOT / "original_full_feature_bank.npz",
    "Official Full": P4R_ROOT / "official_full_feature_bank.npz",
}

MODEL_ROLES = {
    "Bio": "reference and E1 support; not treated as ground truth",
    "ID-only": "independent non-Bio consensus evidence",
    "Local Failed Full": "evaluation-only candidate; never votes on reliability",
    "Official Full": "supplementary evaluation-only candidate; never votes on reliability",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_value(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def mean_or_nan(values: Iterable[Any]) -> float:
    values = finite(list(values))
    return float(np.mean(values)) if len(values) else float("nan")


def pair_nanmean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Elementwise mean over available values without all-NaN warnings."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    result = np.full(len(left), np.nan, dtype=np.float32)
    left_ok = np.isfinite(left)
    right_ok = np.isfinite(right)
    both = left_ok & right_ok
    result[both] = (left[both] + right[both]) / 2.0
    result[left_ok & ~right_ok] = left[left_ok & ~right_ok]
    result[~left_ok & right_ok] = right[~left_ok & right_ok]
    return result


def median_or_nan(values: Iterable[Any]) -> float:
    values = finite(list(values))
    return float(np.median(values)) if len(values) else float("nan")


def q_or_nan(values: Iterable[Any], q: float) -> float:
    values = finite(list(values))
    return float(np.quantile(values, q)) if len(values) else float("nan")


def safe_spearman(x: Any, y: Any) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.ptp(x[valid]) <= 1e-12 or np.ptp(y[valid]) <= 1e-12:
        return float("nan")
    value = float(spearmanr(x[valid], y[valid]).statistic)
    return value if math.isfinite(value) else float("nan")


def rank_within_groups(values: np.ndarray, groups: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Groupwise percentile ranks in [0, 1], preserving NaNs."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    output = np.full(len(values), np.nan, dtype=np.float32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        valid = np.isfinite(values[indices])
        if not valid.any():
            continue
        selected = values[indices[valid]]
        if not higher_is_better:
            selected = -selected
        ranks = rankdata(selected, method="average")
        output[indices[valid]] = ((ranks - 1.0) / max(float(len(selected) - 1), 1.0)).astype(np.float32)
    return output


def global_percentile(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    if valid.any():
        selected = values[valid] if higher_is_better else -values[valid]
        ranks = rankdata(selected, method="average")
        output[valid] = ((ranks - 1.0) / max(float(valid.sum() - 1), 1.0)).astype(np.float32)
    return output


def load_bank(path: Path) -> dict[str, Any]:
    bank = p6a.load_bank(path)
    bank["features"] = bank["global_normed"].astype(np.float32, copy=False)
    return bank


def check_alignment(banks: dict[str, dict[str, Any]], hardness: dict[str, Any]) -> None:
    reference = banks["Bio"]
    if len(reference["stems"]) != IMAGE_COUNT:
        raise RuntimeError(f"Bio bank count is {len(reference['stems'])}, expected {IMAGE_COUNT}")
    for name, bank in banks.items():
        for key in ("stems", "pid", "clothes_id", "camera"):
            if not np.array_equal(bank[key], reference[key]):
                raise RuntimeError(f"{name} is not aligned with Bio in {key}")
    for key in ("stems", "pid", "clothes_id"):
        if not np.array_equal(hardness[key], reference[key]):
            raise RuntimeError(f"Phase5D H_ID bank is not aligned with Bio in {key}")
    if len(np.unique(reference["pid"])) != PID_COUNT:
        raise RuntimeError("unexpected LTCC PID count")
    if len(set(zip(reference["pid"].tolist(), reference["clothes_id"].tolist()))) != STATE_COUNT:
        raise RuntimeError("unexpected LTCC clothing-state count")


def source_record(name: str, checkpoint: Path | None, bank_path: Path | None, role: str, used_for_vote: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "checkpoint": checkpoint,
        "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "feature_bank": bank_path,
        "feature_bank_exists": bool(bank_path and bank_path.is_file()),
        "feature_bank_sha256": sha256_file(bank_path) if bank_path else None,
        "role": role,
        "used_for_reliability_vote": used_for_vote,
        "feature_field": "global_feature",
        "normalization": "L2 normalization of global_feature, float32",
        "feature_semantics": "Phase4R eval output feat_bio after EVA head/pooling; source used by ID head",
    }
    if bank_path and bank_path.is_file():
        with np.load(bank_path, allow_pickle=False) as source:
            record["feature_dimension"] = int(source["global_feature"].shape[1])
            record["feature_count"] = int(source["global_feature"].shape[0])
            record["keys"] = list(source.files)
            record["has_bio_projected"] = bool("bio_projected" in source and source["bio_projected"].shape[1] > 0)
    else:
        record["feature_dimension"] = None
        record["feature_count"] = None
        record["keys"] = []
        record["has_bio_projected"] = False
    return record


def write_source_audit(records: list[dict[str, Any]], hardness_record: dict[str, Any]) -> None:
    lines = [
        "# Phase 6A2 source audit",
        "",
        "This is a train-only offline audit. No model, optimizer, sampler, or backward pass was created.",
        "Relation truth is read from LTCC PID/clothing labels. Local Failed Full is evaluation-only and does not vote on reliability.",
        "All relation features use the frozen `global_feature` field followed by float32 L2 normalization, matching the Phase6A relation protocol; the available 512-D `bio_projected` field is not substituted.",
        "",
        "| Name | Feature bank | Checkpoint | Dim | Count | Bank SHA256 | Role | Reliability vote |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['name']} | `{record['feature_bank']}` | `{record['checkpoint']}` | "
            f"{record['feature_dimension'] or 'NA'} | {record['feature_count'] or 'NA'} | "
            f"`{record['feature_bank_sha256'] or 'NA'}` | {record['role']} | "
            f"{'yes' if record['used_for_reliability_vote'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Frozen H_ID source",
        "",
        f"`{hardness_record['path']}`; SHA256 `{hardness_record['sha256']}`; exact frozen Phase5D `h_id`/`h_cloth` values. H_ID is damage/risk evidence, not a reliability vote.",
        "",
        "## Optional evidence availability",
        "",
        "- Deterministic augmentation relation bank: NOT AVAILABLE. No pre-existing fair multi-view relation bank was present; no new extraction was used to fill this column.",
        "- Temporal Bio relation bank: NOT AVAILABLE. Only the Bio BEST relation bank was already available. Phase5D BEST-vs-S0 H_ID rank stability is auxiliary, not relation stability.",
        "- Official Full is supplementary evaluation only and is not part of any reliability candidate.",
        "",
    ])
    (OUT_ROOT / "source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_label_pairs(pids: np.ndarray, clothes: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    by_state: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, (pid, cloth) in enumerate(zip(pids.tolist(), clothes.tolist())):
        by_state[(int(pid), int(cloth))].append(index)
        by_pid[int(pid)].append(index)
    cross: list[tuple[int, int]] = []
    same: list[tuple[int, int]] = []
    for anchor in range(len(pids)):
        pid = int(pids[anchor])
        cloth = int(clothes[anchor])
        target_clothes = sorted({int(clothes[index]) for index in by_pid[pid] if int(clothes[index]) != cloth})
        for target_cloth in target_clothes:
            # Stable label-only selection. Feature values never enter this choice.
            cross.append((anchor, by_state[(pid, target_cloth)][0]))
        same_candidates = [index for index in by_state[(pid, cloth)] if index != anchor]
        if same_candidates:
            same.append((anchor, same_candidates[0]))
    return cross, same, by_state


def pair_arrays(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray([pair[0] for pair in pairs], dtype=np.int32), np.asarray([pair[1] for pair in pairs], dtype=np.int32)


def pair_sim(features: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(features[left] * features[right], axis=1).astype(np.float32)


def same_pid_purity(top_indices: np.ndarray, pids: np.ndarray) -> np.ndarray:
    return np.mean(pids[top_indices] == pids[:, None], axis=1).astype(np.float32)


def nearest_negative_arrays(banks: dict[str, dict[str, Any]], pids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    indices: dict[str, np.ndarray] = {}
    for name, bank in banks.items():
        print(f"PHASE6A2_TOP_NEGATIVE {name}", flush=True)
        top_values, top_indices = p6a.top_negative_neighbors(bank["features"], pids, k=1, chunk_size=256)
        values[name] = top_values[:, 0]
        indices[name] = top_indices[:, 0]
    return values, indices


def write_pair_bank(
    cross_left: np.ndarray,
    cross_right: np.ndarray,
    same_left: np.ndarray,
    same_right: np.ndarray,
    negative_indices: np.ndarray,
    stems: np.ndarray,
    pids: np.ndarray,
    clothes: np.ndarray,
    cameras: np.ndarray,
) -> None:
    rows: list[dict[str, Any]] = []
    pair_id = 0
    for relation_type, left, right in (
        ("same_pid_different_clothes_positive", cross_left, cross_right),
        ("same_pid_same_clothes_positive_control", same_left, same_right),
    ):
        for anchor, other in zip(left.tolist(), right.tolist()):
            rows.append({
                "pair_id": pair_id, "relation_type": relation_type, "relation_label": "positive",
                "index_a": anchor, "index_b": other, "stem_a": str(stems[anchor]), "stem_b": str(stems[other]),
                "image_path_a": str(TRAIN_ROOT / f"{stems[anchor]}.png"), "image_path_b": str(TRAIN_ROOT / f"{stems[other]}.png"),
                "pid_a": int(pids[anchor]), "pid_b": int(pids[other]), "clothes_a": int(clothes[anchor]),
                "clothes_b": int(clothes[other]), "camera_a": int(cameras[anchor]), "camera_b": int(cameras[other]),
            })
            pair_id += 1
    for anchor, negative in enumerate(negative_indices.tolist()):
        rows.append({
            "pair_id": pair_id, "relation_type": "different_pid_nearest_bio_negative", "relation_label": "negative",
            "index_a": anchor, "index_b": negative, "stem_a": str(stems[anchor]), "stem_b": str(stems[negative]),
            "image_path_a": str(TRAIN_ROOT / f"{stems[anchor]}.png"), "image_path_b": str(TRAIN_ROOT / f"{stems[negative]}.png"),
            "pid_a": int(pids[anchor]), "pid_b": int(pids[negative]), "clothes_a": int(clothes[anchor]),
            "clothes_b": int(clothes[negative]), "camera_a": int(cameras[anchor]), "camera_b": int(cameras[negative]),
        })
        pair_id += 1
    write_csv(OUT_ROOT / "relation_pair_bank.csv", rows)


def evidence_support_arrays(
    relation_stats: dict[str, dict[str, Any]],
    pids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bio_top: np.ndarray,
    id_top: np.ndarray,
    bio_negative_indices: np.ndarray,
    id_negative_indices: np.ndarray,
    banks: dict[str, dict[str, Any]],
    hardness: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compute evidence arrays without collapsing relation truth and evidence."""
    bio = banks["Bio"]["features"]
    ident = banks["ID-only"]["features"]
    bio_sim = pair_sim(bio, left, right)
    id_sim = pair_sim(ident, left, right)
    bio_rank = rank_within_groups(bio_sim, left, higher_is_better=True)
    id_rank = rank_within_groups(id_sim, left, higher_is_better=True)
    consensus = (1.0 - np.abs(bio_rank - id_rank)).astype(np.float32)

    mutual = np.zeros((len(left), 6), dtype=np.float32)
    for row, (anchor, positive) in enumerate(zip(left.tolist(), right.tolist())):
        for column, k in enumerate((1, 5, 10)):
            mutual[row, column] = float(
                np.any(bio_top[anchor, :k] == positive)
                and np.any(bio_top[positive, :k] == anchor)
            )
            mutual[row, column + 3] = float(
                np.any(id_top[anchor, :k] == positive)
                and np.any(id_top[positive, :k] == anchor)
            )
    mutual_score = np.mean(mutual, axis=1).astype(np.float32)

    bio_cross = relation_stats["Bio"]
    id_cross = relation_stats["ID-only"]
    bio_support = np.asarray(bio_cross["positive_mean"], dtype=np.float32)
    id_support = np.asarray(id_cross["positive_mean"], dtype=np.float32)
    bio_support_rank = rank_within_groups(bio_support, pids, higher_is_better=True)
    id_support_rank = rank_within_groups(id_support, pids, higher_is_better=True)
    endpoint_support = np.stack([bio_support_rank[left], bio_support_rank[right]], axis=1)
    intra_support = np.nanmean(endpoint_support, axis=1).astype(np.float32)
    intra_support_min = np.nanmin(endpoint_support, axis=1).astype(np.float32)

    bio_negative_sim = pair_sim(bio, left, bio_negative_indices[left])
    id_negative_sim = pair_sim(ident, left, id_negative_indices[left])
    bio_margin = (bio_sim - bio_negative_sim).astype(np.float32)
    id_margin = (id_sim - id_negative_sim).astype(np.float32)
    bio_ordering = (bio_margin > 0).astype(np.float32)
    id_ordering = (id_margin > 0).astype(np.float32)
    ordering_consensus = ((bio_ordering + id_ordering) / 2.0).astype(np.float32)

    h_id = np.asarray(hardness["h_id"], dtype=np.float32)
    h_cloth = np.asarray(hardness["h_cloth"], dtype=np.float32)
    pair_h_id = np.nanmean(np.stack([h_id[left], h_id[right]], axis=1), axis=1).astype(np.float32)
    pair_h_cloth = np.nanmean(np.stack([h_cloth[left], h_cloth[right]], axis=1), axis=1).astype(np.float32)
    pair_pid = pids[left].astype(np.int16)
    pair_h_id_within_pid = rank_within_groups(pair_h_id, pair_pid, higher_is_better=True)
    pair_h_cloth_within_pid = rank_within_groups(pair_h_cloth, pair_pid, higher_is_better=True)

    # Equal-evidence aggregates are explicit rank constructions, not tuned
    # weights. R4 is named PARTIAL because E4/E5 are unavailable.
    r0 = bio_rank
    r1 = np.nanmean(np.stack([bio_rank, consensus], axis=1), axis=1).astype(np.float32)
    r2 = np.nanmean(np.stack([consensus, mutual_score], axis=1), axis=1).astype(np.float32)
    r3 = np.nanmean(np.stack([consensus, mutual_score, intra_support], axis=1), axis=1).astype(np.float32)
    r4 = np.nanmean(np.stack([consensus, mutual_score, intra_support, ordering_consensus], axis=1), axis=1).astype(np.float32)
    return {
        "bio_support": bio_rank.astype(np.float32),
        "id_support": id_rank.astype(np.float32),
        "consensus": consensus,
        "mutual_bio_k1": mutual[:, 0], "mutual_bio_k5": mutual[:, 1], "mutual_bio_k10": mutual[:, 2],
        "mutual_id_k1": mutual[:, 3], "mutual_id_k5": mutual[:, 4], "mutual_id_k10": mutual[:, 5],
        "mutual_score": mutual_score,
        "intra_support": intra_support,
        "intra_support_min": intra_support_min,
        "bio_similarity": bio_sim,
        "id_similarity": id_sim,
        "bio_negative_similarity": bio_negative_sim,
        "id_negative_similarity": id_negative_sim,
        "bio_margin": bio_margin,
        "id_margin": id_margin,
        "bio_ordering": bio_ordering,
        "id_ordering": id_ordering,
        "ordering_consensus": ordering_consensus,
        "pair_h_id": pair_h_id,
        "pair_h_cloth": pair_h_cloth,
        "pair_h_id_within_pid": pair_h_id_within_pid,
        "pair_h_cloth_within_pid": pair_h_cloth_within_pid,
        "r0_bio_only": r0,
        "r1_bio_id_consensus": r1,
        "r2_consensus_mutual": r2,
        "r3_consensus_mutual_intra": r3,
        "r4_available_multi_evidence": r4,
    }


def write_positive_and_evidence_artifacts(
    cross_left: np.ndarray,
    cross_right: np.ndarray,
    same_left: np.ndarray,
    same_right: np.ndarray,
    evidence: dict[str, np.ndarray],
    relation_stats: dict[str, dict[str, Any]],
    negative_indices: dict[str, np.ndarray],
    banks: dict[str, dict[str, Any]],
    hardness: dict[str, Any],
    bio_top: np.ndarray,
    id_top: np.ndarray,
) -> None:
    stems = banks["Bio"]["stems"]
    pids = banks["Bio"]["pid"]
    clothes = banks["Bio"]["clothes_id"]
    cameras = banks["Bio"]["camera"]
    model_names = list(banks)
    h_id = np.asarray(hardness["h_id"], dtype=np.float32)
    h_cloth = np.asarray(hardness["h_cloth"], dtype=np.float32)
    support_bio = relation_stats["Bio"]["positive_mean"]
    support_id = relation_stats["ID-only"]["positive_mean"]
    count_bio = relation_stats["Bio"]["positive_count"]
    count_id = relation_stats["ID-only"]["positive_count"]

    cross_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    bio_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    mutual_rows: list[dict[str, Any]] = []
    intra_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    triplet_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    id_rank = evidence["id_support"]

    for row, (anchor, positive) in enumerate(zip(cross_left.tolist(), cross_right.tolist())):
        base = {
            "pair_id": row, "index_a": anchor, "index_b": positive,
            "stem_a": str(stems[anchor]), "stem_b": str(stems[positive]),
            "image_path_a": str(TRAIN_ROOT / f"{stems[anchor]}.png"),
            "image_path_b": str(TRAIN_ROOT / f"{stems[positive]}.png"),
            "pid": int(pids[anchor]), "clothes_a": int(clothes[anchor]), "clothes_b": int(clothes[positive]),
            "camera_a": int(cameras[anchor]), "camera_b": int(cameras[positive]),
            "relation_label": "positive", "relation_type": "same_pid_different_clothes",
            "h_id_a": h_id[anchor], "h_id_b": h_id[positive], "h_id_pair": evidence["pair_h_id"][row],
            "h_cloth_a": h_cloth[anchor], "h_cloth_b": h_cloth[positive], "h_cloth_pair": evidence["pair_h_cloth"][row],
        }
        for name in model_names:
            sim = float(pair_sim(banks[name]["features"], np.asarray([anchor]), np.asarray([positive]))[0])
            base[f"{p6a.model_slug(name)}_similarity"] = sim
            divergence_rows.append({
                "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]),
                "model": name, "relation_label": "positive", "similarity": sim,
                "delta_to_bio": sim - float(evidence["bio_similarity"][row]),
                "is_reliability_voter": int(name in ("Bio", "ID-only")),
            })
        cross_rows.append(base)
        bio_rows.append({
            "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]),
            "clothes_a": int(clothes[anchor]), "clothes_b": int(clothes[positive]), "relation_label": "positive",
            "bio_similarity": evidence["bio_similarity"][row],
            "bio_within_anchor_percentile": evidence["bio_support"][row],
            "h_id_pair": evidence["pair_h_id"][row],
        })
        consensus_rows.append({
            "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]),
            "bio_similarity": evidence["bio_similarity"][row], "id_similarity": evidence["id_similarity"][row],
            "bio_rank_percentile": evidence["bio_support"][row], "id_rank_percentile": id_rank[row],
            "absolute_similarity_difference": abs(float(evidence["bio_similarity"][row] - evidence["id_similarity"][row])),
            "rank_gap": abs(float(evidence["bio_support"][row] - id_rank[row])),
            "model_consensus_score": evidence["consensus"][row],
            "consensus_high_at_0_5": int(evidence["consensus"][row] >= 0.5),
        })
        for k, label in ((1, "k1"), (5, "k5"), (10, "k10")):
            mutual_rows.append({
                "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]), "k": k,
                "bio_one_way_a_to_p": int(np.any(bio_top[anchor, :k] == positive)),
                "bio_one_way_p_to_a": int(np.any(bio_top[positive, :k] == anchor)),
                "bio_mutual": int(evidence[f"mutual_bio_{label}"][row]),
                "id_one_way_a_to_p": int(np.any(id_top[anchor, :k] == positive)),
                "id_one_way_p_to_a": int(np.any(id_top[positive, :k] == anchor)),
                "id_mutual": int(evidence[f"mutual_id_{label}"][row]),
            })
        intra_rows.append({
            "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]),
            "bio_anchor_cross_clothes_mean": support_bio[anchor], "bio_positive_cross_clothes_mean": support_bio[positive],
            "bio_anchor_cross_clothes_count": int(count_bio[anchor]), "bio_positive_cross_clothes_count": int(count_bio[positive]),
            "id_anchor_cross_clothes_mean": support_id[anchor], "id_positive_cross_clothes_mean": support_id[positive],
            "id_anchor_cross_clothes_count": int(count_id[anchor]), "id_positive_cross_clothes_count": int(count_id[positive]),
            "pair_intra_id_support": evidence["intra_support"][row],
            "pair_intra_id_support_min": evidence["intra_support_min"][row],
        })
        negative_rows.append({
            "pair_id": row, "index_a": anchor, "positive_index": positive, "pid": int(pids[anchor]),
            "positive_clothes": int(clothes[positive]),
            "bio_negative_index": int(negative_indices["Bio"][anchor]),
            "bio_negative_pid": int(pids[negative_indices["Bio"][anchor]]),
            "bio_negative_similarity": evidence["bio_negative_similarity"][row],
            "id_negative_index": int(negative_indices["ID-only"][anchor]),
            "id_negative_pid": int(pids[negative_indices["ID-only"][anchor]]),
            "id_negative_similarity": evidence["id_negative_similarity"][row],
            "bio_ordering_correct": int(evidence["bio_ordering"][row]),
            "id_ordering_correct": int(evidence["id_ordering"][row]),
            "consensus_ordering_correct": evidence["ordering_consensus"][row],
            "bio_margin": evidence["bio_margin"][row], "id_margin": evidence["id_margin"][row],
        })
        triplet = {
            "pair_id": row, "index_a": anchor, "positive_index": positive, "pid": int(pids[anchor]),
            "bio_positive_similarity": evidence["bio_similarity"][row], "bio_negative_similarity": evidence["bio_negative_similarity"][row],
            "bio_margin": evidence["bio_margin"][row], "bio_ordering_correct": int(evidence["bio_ordering"][row]),
            "id_positive_similarity": evidence["id_similarity"][row], "id_negative_similarity": evidence["id_negative_similarity"][row],
            "id_margin": evidence["id_margin"][row], "id_ordering_correct": int(evidence["id_ordering"][row]),
        }
        for name in ("Local Failed Full", "Official Full"):
            slug = p6a.model_slug(name)
            pos_sim = float(pair_sim(banks[name]["features"], np.asarray([anchor]), np.asarray([positive]))[0])
            neg_idx = int(negative_indices[name][anchor])
            neg_sim = float(np.dot(banks[name]["features"][anchor], banks[name]["features"][neg_idx]))
            triplet[f"{slug}_positive_similarity"] = pos_sim
            triplet[f"{slug}_negative_similarity"] = neg_sim
            triplet[f"{slug}_margin"] = pos_sim - neg_sim
            triplet[f"{slug}_ordering_correct"] = int(pos_sim > neg_sim)
        triplet_rows.append(triplet)

    for row, (anchor, other) in enumerate(zip(same_left.tolist(), same_right.tolist())):
        item = {
            "pair_id": row, "index_a": anchor, "index_b": other, "pid": int(pids[anchor]),
            "clothes_id": int(clothes[anchor]), "relation_label": "positive", "relation_type": "same_pid_same_clothes_control",
        }
        for name in model_names:
            item[f"{p6a.model_slug(name)}_similarity"] = float(pair_sim(banks[name]["features"], np.asarray([anchor]), np.asarray([other]))[0])
        same_rows.append(item)

    write_csv(OUT_ROOT / "cross_clothes_positive.csv", cross_rows)
    write_csv(OUT_ROOT / "same_clothes_positive.csv", same_rows)
    write_csv(OUT_ROOT / "bio_support.csv", bio_rows)
    write_csv(OUT_ROOT / "model_consensus.csv", consensus_rows)
    write_csv(OUT_ROOT / "mutual_neighbor.csv", mutual_rows)
    write_csv(OUT_ROOT / "intra_id_support.csv", intra_rows)
    write_csv(OUT_ROOT / "negative_relation_support.csv", negative_rows)
    write_csv(OUT_ROOT / "triplet_relation.csv", triplet_rows)
    write_csv(OUT_ROOT / "relation_divergence.csv", divergence_rows)


def candidate_gate(evidence: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    e_bio = evidence["bio_support"] >= threshold
    e_consensus = evidence["consensus"] >= threshold
    e_mutual = evidence["mutual_score"] >= threshold
    e_intra = evidence["intra_support"] >= threshold
    e_ordering = evidence["ordering_consensus"] >= threshold
    return {
        "gate_2of3": (e_bio.astype(np.int8) + e_consensus.astype(np.int8) + e_mutual.astype(np.int8) >= 2),
        "gate_3of4": (e_bio.astype(np.int8) + e_consensus.astype(np.int8) + e_mutual.astype(np.int8) + e_intra.astype(np.int8) >= 3),
        "gate_4of5": (e_bio.astype(np.int8) + e_consensus.astype(np.int8) + e_mutual.astype(np.int8) + e_intra.astype(np.int8) + e_ordering.astype(np.int8) >= 4),
    }


def write_reliability_candidates(evidence: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    names = [
        "R0_BIO_ONLY", "R1_BIO_ID_CONSENSUS", "R2_CONSENSUS_MUTUAL",
        "R3_CONSENSUS_MUTUAL_INTRA", "R4_AVAILABLE_MULTI_EVIDENCE_PARTIAL",
    ]
    keys = [
        "r0_bio_only", "r1_bio_id_consensus", "r2_consensus_mutual",
        "r3_consensus_mutual_intra", "r4_available_multi_evidence",
    ]
    gates_by_threshold: dict[str, dict[str, np.ndarray]] = {}
    for threshold in (1.0 / 3.0, 0.5, 2.0 / 3.0):
        gates_by_threshold[f"{threshold:.6f}"] = candidate_gate(evidence, threshold)

    rows: list[dict[str, Any]] = []
    for row in range(len(evidence["bio_support"])):
        item: dict[str, Any] = {
            "pair_id": row,
            "relation_label": "positive",
            "e_bio_support": evidence["bio_support"][row],
            "e_id_support": evidence["id_support"][row],
            "e_model_consensus": evidence["consensus"][row],
            "e_mutual_score": evidence["mutual_score"][row],
            "e_intra_id_support": evidence["intra_support"][row],
            "e_ordering_consensus": evidence["ordering_consensus"][row],
            "R0_BIO_ONLY": evidence[keys[0]][row],
            "R1_BIO_ID_CONSENSUS": evidence[keys[1]][row],
            "R2_CONSENSUS_MUTUAL": evidence[keys[2]][row],
            "R3_CONSENSUS_MUTUAL_INTRA": evidence[keys[3]][row],
            "R4_AVAILABLE_MULTI_EVIDENCE_PARTIAL": evidence[keys[4]][row],
            "R4_missing_view_evidence": 1,
            "R4_missing_temporal_evidence": 1,
        }
        for threshold_key, gates in gates_by_threshold.items():
            for gate_name, gate_values in gates.items():
                item[f"{gate_name}_at_{threshold_key}"] = int(gate_values[row])
        rows.append(item)
    write_csv(OUT_ROOT / "reliability_candidates.csv", rows)

    candidates: dict[str, dict[str, np.ndarray]] = {}
    for name, key in zip(names, keys):
        candidates[name] = {"score": evidence[key]}
    for threshold_key, gates in gates_by_threshold.items():
        for gate_name, gate_values in gates.items():
            candidates[f"{gate_name}_at_{threshold_key}"] = {
                "score": gate_values.astype(np.float32), "is_gate": np.ones(len(rows), dtype=bool)
            }
    return candidates


def tier_mask(score: np.ndarray, tier: str) -> np.ndarray:
    score = np.asarray(score)
    if score.dtype.kind == "b":
        return score.astype(bool) if tier == "high" else (~score.astype(bool) if tier == "low" else np.zeros(len(score), dtype=bool))
    score = score.astype(np.float64)
    valid = np.isfinite(score)
    if tier == "high":
        return valid & (score >= 2.0 / 3.0)
    if tier == "low":
        return valid & (score <= 1.0 / 3.0)
    return valid & (score > 1.0 / 3.0) & (score < 2.0 / 3.0)


def pid_bootstrap_mean(values: np.ndarray, pids: np.ndarray, mask: np.ndarray, seed_offset: int = 0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    pids = np.asarray(pids)
    selected: dict[int, list[float]] = defaultdict(list)
    for value, pid, keep in zip(values.tolist(), pids.tolist(), mask.tolist()):
        if keep and math.isfinite(value):
            selected[int(pid)].append(float(value))
    group_values = np.asarray([np.mean(group) for group in selected.values()], dtype=np.float64)
    if len(group_values) == 0:
        return {"n": 0, "pid_count": 0, "mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(SEED + seed_offset)
    sample_indices = rng.integers(0, len(group_values), size=(BOOTSTRAP_RESAMPLES, len(group_values)))
    bootstrap = np.mean(group_values[sample_indices], axis=1)
    return {
        "n": int(np.sum(mask)), "pid_count": int(len(group_values)),
        "mean": float(np.mean(group_values)), "median": float(np.median(values[mask])),
        "ci_low": float(np.quantile(bootstrap, 0.025)), "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def pid_bootstrap_difference(values: np.ndarray, pids: np.ndarray, high: np.ndarray, low: np.ndarray, seed_offset: int = 0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    pids = np.asarray(pids)
    high_by_pid: dict[int, list[float]] = defaultdict(list)
    low_by_pid: dict[int, list[float]] = defaultdict(list)
    for value, pid, high_keep, low_keep in zip(values.tolist(), pids.tolist(), high.tolist(), low.tolist()):
        if not math.isfinite(value):
            continue
        if high_keep:
            high_by_pid[int(pid)].append(float(value))
        if low_keep:
            low_by_pid[int(pid)].append(float(value))
    common = sorted(set(high_by_pid) & set(low_by_pid))
    if not common:
        return {"pid_count": 0, "difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    group_values = np.asarray([np.mean(high_by_pid[pid]) - np.mean(low_by_pid[pid]) for pid in common], dtype=np.float64)
    rng = np.random.default_rng(SEED + 10_000 + seed_offset)
    sample_indices = rng.integers(0, len(group_values), size=(BOOTSTRAP_RESAMPLES, len(group_values)))
    bootstrap = np.mean(group_values[sample_indices], axis=1)
    return {
        "pid_count": int(len(common)), "difference": float(np.mean(group_values)),
        "ci_low": float(np.quantile(bootstrap, 0.025)), "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def write_threshold_audit(evidence: dict[str, np.ndarray], candidates: dict[str, dict[str, np.ndarray]]) -> None:
    rows: list[dict[str, Any]] = []
    pids = evidence["pair_pid"]
    for threshold in (1.0 / 3.0, 0.5, 2.0 / 3.0):
        gates = candidate_gate(evidence, threshold)
        for gate_name, gate_values in gates.items():
            mask = gate_values.astype(bool)
            states = set(zip(pids[mask].tolist(), evidence["pair_cloth_b"].astype(int)[mask].tolist())) if np.any(mask) else set()
            rows.append({
                "threshold": threshold, "gate": gate_name,
                "threshold_definition": "evidence component percentile >= threshold",
                "selected_pairs": int(mask.sum()), "selected_pair_fraction": float(np.mean(mask)),
                "selected_pids": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                "selected_states": int(len(states)),
                "coverage_sufficient_50pct_pid": int(len(np.unique(pids[mask])) >= math.ceil(PID_COUNT * 0.5)) if np.any(mask) else 0,
            })
    for name in ("R0_BIO_ONLY", "R1_BIO_ID_CONSENSUS", "R2_CONSENSUS_MUTUAL", "R3_CONSENSUS_MUTUAL_INTRA", "R4_AVAILABLE_MULTI_EVIDENCE_PARTIAL"):
        score = candidates[name]["score"]
        rows.append({
            "threshold": "continuous", "gate": name,
            "threshold_definition": "rank aggregate; tiers <=1/3, middle, >=2/3",
            "selected_pairs": len(score), "selected_pair_fraction": 1.0,
            "selected_pids": int(len(np.unique(pids))),
            "selected_states": int(len(set(zip(pids.tolist(), evidence["pair_cloth_b"].astype(int).tolist())))),
            "coverage_sufficient_50pct_pid": 1,
        })
    write_csv(OUT_ROOT / "reliability_threshold_audit.csv", rows)


def utility_index() -> dict[int, dict[str, Any]]:
    """Read existing Phase5D H_ID-family probes; never run new backward."""
    target_rows = [row for row in read_csv(P5D_UTILITY) if row.get("family") == "H_ID"]
    quadrant_rows = [row for row in read_csv(P5D_QUADRANTS) if row.get("family") == "H_ID"]
    labels: dict[tuple[str, str, str], str] = {}
    for row in quadrant_rows:
        labels[(row.get("quartile", ""), row.get("probe_id", ""), row.get("index", ""))] = row.get("utility_label", "")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        idx = int(row["index"])
        label = labels.get((row.get("quartile", ""), row.get("probe_id", ""), row.get("index", "")), "")
        grouped[idx].append({
            "late_alignment": f(row.get("gradient_alignment_late_eva_block")),
            "same_pid_alignment": f(row.get("same_pid_alignment_late_eva_block")),
            "bio_projection_alignment": f(row.get("gradient_alignment_bio_projection")),
            "id_head_alignment": f(row.get("gradient_alignment_id_head")),
            "correct": f(row.get("correct")),
            "confidence": f(row.get("confidence")),
            "utility_label": label,
        })
    result: dict[int, dict[str, Any]] = {}
    for idx, items in grouped.items():
        result[idx] = {
            "late_alignment": mean_or_nan(item["late_alignment"] for item in items),
            "same_pid_alignment": mean_or_nan(item["same_pid_alignment"] for item in items),
            "bio_projection_alignment": mean_or_nan(item["bio_projection_alignment"] for item in items),
            "id_head_alignment": mean_or_nan(item["id_head_alignment"] for item in items),
            "correct": mean_or_nan(item["correct"] for item in items),
            "confidence": mean_or_nan(item["confidence"] for item in items),
            "utility_label": next((item["utility_label"] for item in items if item["utility_label"]), ""),
            "source_rows": len(items),
        }
    return result


def write_phase5d_join(cross_left: np.ndarray, cross_right: np.ndarray, pids: np.ndarray) -> dict[str, np.ndarray]:
    utilities = utility_index()
    n = len(cross_left)
    anchor_late = np.full(n, np.nan, dtype=np.float32)
    positive_late = np.full(n, np.nan, dtype=np.float32)
    anchor_same = np.full(n, np.nan, dtype=np.float32)
    positive_same = np.full(n, np.nan, dtype=np.float32)
    anchor_utility = np.zeros(n, dtype=np.int8)
    positive_utility = np.zeros(n, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for row, (anchor, positive) in enumerate(zip(cross_left.tolist(), cross_right.tolist())):
        ua = utilities.get(anchor, {})
        up = utilities.get(positive, {})
        a_label = ua.get("utility_label", "")
        p_label = up.get("utility_label", "")
        anchor_late[row] = ua.get("late_alignment", np.nan)
        positive_late[row] = up.get("late_alignment", np.nan)
        anchor_same[row] = ua.get("same_pid_alignment", np.nan)
        positive_same[row] = up.get("same_pid_alignment", np.nan)
        anchor_utility[row] = int(a_label == "HIGH_HARD_HIGH_UTILITY")
        positive_utility[row] = int(p_label == "HIGH_HARD_HIGH_UTILITY")
        rows.append({
            "pair_id": row, "index_a": anchor, "index_b": positive, "pid": int(pids[anchor]),
            "anchor_utility_available": int(bool(ua)), "positive_utility_available": int(bool(up)),
            "anchor_late_eva_alignment": anchor_late[row], "positive_late_eva_alignment": positive_late[row],
            "pair_late_eva_alignment": mean_or_nan([anchor_late[row], positive_late[row]]),
            "anchor_same_pid_alignment": anchor_same[row], "positive_same_pid_alignment": positive_same[row],
            "anchor_utility_label": a_label, "positive_utility_label": p_label,
            "anchor_high_hard_high_utility": anchor_utility[row], "positive_high_hard_high_utility": positive_utility[row],
            "both_high_hard_high_utility": int(anchor_utility[row] and positive_utility[row]),
            "any_high_hard_high_utility": int(anchor_utility[row] or positive_utility[row]),
            "phase5d_source": str(P5D_UTILITY),
            "note": "existing anchor-level gradient probe; no new backward and not pair-specific utility",
        })
    write_csv(OUT_ROOT / "phase5d_utility_join.csv", rows)
    return {
        "anchor_late_alignment": anchor_late, "positive_late_alignment": positive_late,
        "pair_late_alignment": pair_nanmean(anchor_late, positive_late),
        "anchor_same_pid_alignment": anchor_same, "positive_same_pid_alignment": positive_same,
        "anchor_high_utility": anchor_utility.astype(np.float32), "positive_high_utility": positive_utility.astype(np.float32),
        "any_high_utility": np.maximum(anchor_utility, positive_utility).astype(np.float32),
        "both_high_utility": (anchor_utility * positive_utility).astype(np.float32),
        "available": (np.isfinite(anchor_late) | np.isfinite(positive_late)).astype(np.float32),
    }


def write_validation_tables(
    candidates: dict[str, dict[str, np.ndarray]],
    evidence: dict[str, np.ndarray],
    banks: dict[str, dict[str, Any]],
    negative_indices: dict[str, np.ndarray],
    bio_top: np.ndarray,
    failed_top: np.ndarray,
    utility: dict[str, np.ndarray],
) -> dict[str, Any]:
    pids = evidence["pair_pid"]
    left = evidence["pair_left"]
    right = evidence["pair_right"]
    bio = banks["Bio"]["features"]
    failed = banks["Local Failed Full"]["features"]
    failed_pos = pair_sim(failed, left, right)
    failed_neg = np.asarray([
        np.dot(failed[int(anchor)], failed[int(negative_indices["Bio"][int(anchor)])])
        for anchor in left.tolist()
    ], dtype=np.float32)
    failed_margin = failed_pos - failed_neg
    bio_neighbor_purity = same_pid_purity(bio_top, banks["Bio"]["pid"])[left]
    failed_neighbor_purity = same_pid_purity(failed_top, banks["Bio"]["pid"])[left]
    relation_error = failed_pos - evidence["bio_similarity"]
    margin_error = failed_margin - evidence["bio_margin"]
    neighbor_error = failed_neighbor_purity - bio_neighbor_purity
    failed_rank = rank_within_groups(failed_pos, left, higher_is_better=True)
    rank_error = failed_rank - evidence["bio_support"]

    metric_arrays = {
        "bio_positive_negative_margin": evidence["bio_margin"],
        "id_positive_negative_margin": evidence["id_margin"],
        "bio_ordering_correct": evidence["bio_ordering"],
        "id_ordering_correct": evidence["id_ordering"],
        "consensus_ordering_correct": evidence["ordering_consensus"],
        "relation_similarity_failed_minus_bio": relation_error,
        "margin_failed_minus_bio": margin_error,
        "neighbor_purity_failed_minus_bio": neighbor_error,
        "rank_failed_minus_bio": rank_error,
        "pair_h_id": evidence["pair_h_id"],
        "pair_h_cloth": evidence["pair_h_cloth"],
    }
    candidates_continuous = [name for name in candidates if not name.startswith("gate_")]
    validation_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    primary: dict[str, Any] = {"ordering": {}, "degradation": {}, "utility": {}}
    tier_offsets = {"low": 1, "medium": 2, "high": 3}
    for candidate_index, name in enumerate(candidates_continuous):
        score = candidates[name]["score"]
        tier_result: dict[str, Any] = {}
        for tier in ("low", "medium", "high"):
            mask = tier_mask(score, tier)
            tier_result[tier] = {}
            for metric_name, values in metric_arrays.items():
                stats = pid_bootstrap_mean(values, pids, mask, seed_offset=100 * candidate_index + tier_offsets[tier])
                validation_rows.append({
                    "candidate": name, "tier": tier, "metric": metric_name, **stats,
                    "coverage_pid": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                    "coverage_pair": int(mask.sum()),
                })
                tier_result[tier][metric_name] = stats
        primary["ordering"][name] = tier_result

        high = tier_mask(score, "high")
        low = tier_mask(score, "low")
        for metric_name, values in metric_arrays.items():
            diff = pid_bootstrap_difference(values, pids, high, low, seed_offset=1000 + 100 * candidate_index + len(metric_name))
            ordering_rows.append({
                "candidate": name, "comparison": "high_minus_low", "metric": metric_name, **diff,
                "high_pairs": int(high.sum()), "low_pairs": int(low.sum()),
                "high_pids": int(len(np.unique(pids[high]))) if np.any(high) else 0,
                "low_pids": int(len(np.unique(pids[low]))) if np.any(low) else 0,
            })

        for tier in ("low", "medium", "high"):
            mask = tier_mask(score, tier)
            for metric_name, values in (
                ("failed_minus_bio_similarity", relation_error),
                ("failed_minus_bio_margin", margin_error),
                ("failed_minus_bio_neighbor_purity", neighbor_error),
                ("failed_minus_bio_rank_percentile", rank_error),
            ):
                stats = pid_bootstrap_mean(values, pids, mask, seed_offset=3000 + 100 * candidate_index + tier_offsets[tier])
                degradation_rows.append({
                    "candidate": name, "tier": tier, "metric": metric_name, **stats,
                    "interpretation": "negative means Failed Full is below Bio on this metric",
                })
        primary["degradation"][name] = {
            "high_relation_error": mean_or_nan(relation_error[high]), "low_relation_error": mean_or_nan(relation_error[low]),
            "high_margin_error": mean_or_nan(margin_error[high]), "low_margin_error": mean_or_nan(margin_error[low]),
        }

        utility_metric_arrays = {
            "pair_late_eva_alignment": utility["pair_late_alignment"],
            "pair_same_pid_alignment": pair_nanmean(utility["anchor_same_pid_alignment"], utility["positive_same_pid_alignment"]),
            "any_high_hard_high_utility": utility["any_high_utility"],
            "both_high_hard_high_utility": utility["both_high_utility"],
        }
        primary["utility"][name] = {}
        for utility_name, values in utility_metric_arrays.items():
            for tier in ("low", "medium", "high"):
                mask = tier_mask(score, tier) & (utility["available"] > 0)
                stats = pid_bootstrap_mean(values, pids, mask, seed_offset=5000 + 100 * candidate_index + tier_offsets[tier])
                utility_rows.append({
                    "candidate": name, "tier": tier, "metric": utility_name, **stats,
                    "utility_source": "Phase5D existing anchor-level probe; not pair-specific",
                })
            high_u = tier_mask(score, "high") & (utility["available"] > 0)
            low_u = tier_mask(score, "low") & (utility["available"] > 0)
            primary["utility"][name][utility_name] = pid_bootstrap_difference(values, pids, high_u, low_u, seed_offset=6000 + 100 * candidate_index + len(utility_name))

    write_csv(OUT_ROOT / "reliability_validation.csv", validation_rows)
    write_csv(OUT_ROOT / "ordering_validation.csv", ordering_rows)
    write_csv(OUT_ROOT / "failed_full_degradation_by_reliability.csv", degradation_rows)
    # Extra table for compact report inspection; the required pair join is
    # written by write_phase5d_join.
    write_csv(OUT_ROOT / "phase5d_utility_validation.csv", utility_rows)
    return {
        "primary": primary, "relation_error": relation_error, "margin_error": margin_error,
        "neighbor_error": neighbor_error, "rank_error": rank_error,
        "failed_positive": failed_pos, "failed_margin": failed_margin,
        "failed_neighbor_purity": failed_neighbor_purity, "bio_neighbor_purity": bio_neighbor_purity,
    }


def write_quadrants_and_curriculum(
    candidates: dict[str, dict[str, np.ndarray]],
    evidence: dict[str, np.ndarray],
    validation: dict[str, Any],
    utility: dict[str, np.ndarray],
) -> None:
    pids = evidence["pair_pid"]
    rows: list[dict[str, Any]] = []
    curriculum_rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        if name.startswith("gate_"):
            continue
        score = candidate["score"]
        high_r = score >= 2.0 / 3.0
        low_r = score <= 1.0 / 3.0
        high_h = evidence["pair_h_id_within_pid"] >= 0.75
        low_h = evidence["pair_h_id_within_pid"] <= 0.25
        for quadrant, mask in (
            ("HIGH_H_ID_HIGH_RELIABILITY", high_h & high_r),
            ("HIGH_H_ID_LOW_RELIABILITY", high_h & low_r),
            ("LOW_H_ID_HIGH_RELIABILITY", low_h & high_r),
            ("LOW_H_ID_LOW_RELIABILITY", low_h & low_r),
        ):
            states = set(zip(pids[mask].tolist(), evidence["pair_cloth_b"].astype(int)[mask].tolist())) if np.any(mask) else set()
            rows.append({
                "candidate": name, "quadrant": quadrant, "pairs": int(mask.sum()),
                "pids": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                "pid_coverage_fraction": float(len(np.unique(pids[mask])) / PID_COUNT) if np.any(mask) else 0.0,
                "clothing_states": int(len(states)),
                "bio_similarity": mean_or_nan(evidence["bio_similarity"][mask]),
                "bio_margin": mean_or_nan(evidence["bio_margin"][mask]),
                "bio_ordering_correct": mean_or_nan(evidence["bio_ordering"][mask]),
                "model_consensus": mean_or_nan(evidence["consensus"][mask]),
                "mutual_score": mean_or_nan(evidence["mutual_score"][mask]),
                "intra_support": mean_or_nan(evidence["intra_support"][mask]),
                "failed_full_minus_bio_similarity": mean_or_nan(validation["relation_error"][mask]),
                "failed_full_minus_bio_margin": mean_or_nan(validation["margin_error"][mask]),
                "phase5d_pair_late_alignment": mean_or_nan(utility["pair_late_alignment"][mask]),
                "phase5d_any_high_utility_rate": mean_or_nan(utility["any_high_utility"][mask]),
            })

        # This is the requested offline Reliable-Easy/Medium/Hard audit. The
        # reliability cutoff is kept explicit and is not test-tuned.
        reliable = high_r & np.isfinite(evidence["pair_h_id_within_pid"])
        h = evidence["pair_h_id_within_pid"]
        for tier, mask in (
            ("reliable_easy", reliable & (h <= 1.0 / 3.0)),
            ("reliable_medium", reliable & (h > 1.0 / 3.0) & (h < 2.0 / 3.0)),
            ("reliable_hard", reliable & (h >= 2.0 / 3.0)),
        ):
            states = set(zip(pids[mask].tolist(), evidence["pair_cloth_b"].astype(int)[mask].tolist())) if np.any(mask) else set()
            curriculum_rows.append({
                "candidate": name, "reliability_gate": "continuous score >= 2/3", "hardness_tier": tier,
                "pairs": int(mask.sum()), "pids": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                "pid_coverage_fraction": float(len(np.unique(pids[mask])) / PID_COUNT) if np.any(mask) else 0.0,
                "clothing_states": int(len(states)), "h_id_within_pid_mean": mean_or_nan(h[mask]),
                "bio_margin_mean": mean_or_nan(evidence["bio_margin"][mask]),
                "bio_ordering_correct_rate": mean_or_nan(evidence["bio_ordering"][mask]),
                "id_margin_mean": mean_or_nan(evidence["id_margin"][mask]),
                "phase5d_pair_late_alignment": mean_or_nan(utility["pair_late_alignment"][mask]),
                "phase5d_any_high_utility_rate": mean_or_nan(utility["any_high_utility"][mask]),
            })
    write_csv(OUT_ROOT / "hardness_reliability_quadrants.csv", rows)
    write_csv(OUT_ROOT / "reliable_hardness_curriculum_audit.csv", curriculum_rows)


def write_coverage_audit(candidates: dict[str, dict[str, np.ndarray]], evidence: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pids = evidence["pair_pid"]
    for name, candidate in candidates.items():
        score = candidate["score"]
        tiers = [("selected", score.astype(bool))] if name.startswith("gate_") else [(tier, tier_mask(score, tier)) for tier in ("low", "medium", "high")]
        for tier, mask in tiers:
            states = set(zip(pids[mask].tolist(), evidence["pair_cloth_b"].astype(int)[mask].tolist())) if np.any(mask) else set()
            pid_count = int(len(np.unique(pids[mask]))) if np.any(mask) else 0
            rows.append({
                "candidate": name, "tier": tier, "pairs": int(mask.sum()), "pair_fraction": float(np.mean(mask)),
                "pids": pid_count, "pid_coverage_fraction": float(pid_count / PID_COUNT),
                "clothing_states": int(len(states)), "clothing_state_coverage_fraction": float(len(states) / STATE_COUNT),
                "minimum_pid_coverage_rule_50pct": int(pid_count >= math.ceil(PID_COUNT * 0.5)),
            })
    write_csv(OUT_ROOT / "coverage_audit.csv", rows)
    return rows


def write_model_summary(
    banks: dict[str, dict[str, Any]],
    relation_stats: dict[str, dict[str, Any]],
    negative_values: dict[str, np.ndarray],
    pids: np.ndarray,
    clothes: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, bank in banks.items():
        stats = relation_stats[name]
        nearest = negative_values[name]
        margin = stats["positive_mean"] - nearest
        top10 = p6a.top_neighbors(bank["features"], k=TOP_K, chunk_size=256)
        purity = same_pid_purity(top10, pids)
        item = {
            "model": name, "role": MODEL_ROLES[name],
            "cross_clothes_pair_count_directed": len(stats["pair_values"]),
            "cross_clothes_positive_similarity_mean": mean_or_nan(stats["pair_values"]),
            "cross_clothes_positive_similarity_median": median_or_nan(stats["pair_values"]),
            "cross_clothes_positive_similarity_p25": q_or_nan(stats["pair_values"], 0.25),
            "cross_clothes_positive_similarity_p75": q_or_nan(stats["pair_values"], 0.75),
            "nearest_negative_similarity_mean": mean_or_nan(nearest),
            "identity_margin_mean": mean_or_nan(margin), "identity_margin_median": median_or_nan(margin),
            "negative_closer_than_positive_rate": float(np.mean(nearest >= stats["positive_mean"])),
            "top10_same_pid_purity": mean_or_nan(purity), "train_pid_count": int(len(np.unique(pids))),
            "train_clothing_state_count": int(len(set(zip(pids.tolist(), clothes.tolist())))),
        }
        rows.append(item)
        summary[name] = item
    write_csv(OUT_ROOT / "teacher_relation_summary.csv", rows)
    return summary


def write_unavailable_evidence() -> None:
    write_csv(OUT_ROOT / "augmentation_stability.csv", [{
        "status": "NOT_AVAILABLE", "evidence_family": "E4_view_stability",
        "reason": "No pre-existing deterministic augmentation relation bank with the same frozen protocol; no new extraction was used to fill this column.",
        "used_in_reliability_candidate": 0,
    }])
    auxiliary: dict[str, Any] = {}
    if P5D_STABILITY.is_file():
        try:
            auxiliary = json.loads(P5D_STABILITY.read_text(encoding="utf-8")).get("s0_epoch15_vs_best", {})
        except Exception:
            auxiliary = {}
    write_csv(OUT_ROOT / "temporal_stability.csv", [{
        "status": "NOT_AVAILABLE", "evidence_family": "E5_temporal_relation_stability",
        "reason": "Only one frozen Bio relation feature bank was available; Phase5D H_ID rank stability is auxiliary, not relation stability.",
        "available_relation_banks": 1,
        "auxiliary_phase5d_sample_spearman": auxiliary.get("sample_spearman", ""),
        "auxiliary_phase5d_sample_kendall": auxiliary.get("sample_kendall", ""),
        "used_in_reliability_candidate": 0,
    }])


def write_future_design_docs() -> None:
    ds = """# Future Relation DS v2 (audit only)

Phase 6A2 does not implement or train this design.

## Principle

`relation_label` comes only from LTCC PID labels. A train-only reliability gate
may select which relation/pair/triplet is emphasized, but it must not hide any
image or clothing state. All images and clothing states remain available.

## Candidate curriculum

1. High-reliability cross-clothes positives with low/medium H_ID risk.
2. High-reliability medium-difficulty relations.
3. High-reliability high-H_ID relations only if independent utility remains acceptable.

The ordering is reliability-first, difficulty-second. H_ID is a risk detector,
not a direct sample-frequency command. Any future comparison must keep a
uniform-relation control and report PID/clothing coverage.

## Not supported by this audit

No active-set withholding, hard-only sampling, direct H_ID oversampling, or
test/query/gallery thresholding.
"""
    loss = """# Future Relation Loss v2 (audit only)

No loss is added in Phase 6A2 and no optimizer step is permitted.

## Ground-truth relation direction

- Positive: same PID; primary subset same PID and different clothes.
- Negative: different PID.
- Bio and ID-only features may help identify stable evidence, but the student is not required to copy Bio feature vectors.

## Candidate forms

`L_pos_rel` can operate on high-reliability same-PID cross-clothes pairs.

`L_triplet_rel = max(0, m - sim_S(i,p) + sim_S(i,n))` can use a high-reliability positive and negative relation.

`L_rank_rel` can preserve the positive-greater-than-negative ordering.

These are interfaces for a future ablation, not a claim that distillation will
improve ReID. Any future training must compare uniform relation selection first
and preserve full image/clothing availability.
"""
    (OUT_ROOT / "future_relation_ds_v2.md").write_text(ds, encoding="utf-8")
    (OUT_ROOT / "future_relation_loss_v2.md").write_text(loss, encoding="utf-8")


def report_num(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def choose_diagnosis(
    candidates: dict[str, dict[str, np.ndarray]],
    evidence: dict[str, np.ndarray],
    utility: dict[str, np.ndarray],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply a conservative offline gate; no test metric enters this decision."""
    continuous = [name for name in candidates if not name.startswith("gate_")]
    # R4 contains positive-negative ordering as a definition component. It is
    # still exported, but cannot be used for the primary ordering promotion
    # test because that would validate a definition signal against itself.
    validated = [name for name in continuous if name != "R4_AVAILABLE_MULTI_EVIDENCE_PARTIAL"]
    pids = evidence["pair_pid"]
    comparisons: list[dict[str, Any]] = []
    for index, name in enumerate(validated[1:], start=1):
        score = candidates[name]["score"]
        high = tier_mask(score, "high")
        low = tier_mask(score, "low")
        margin = pid_bootstrap_difference(evidence["bio_margin"], pids, high, low, seed_offset=9000 + index)
        ordering = pid_bootstrap_difference(evidence["bio_ordering"], pids, high, low, seed_offset=9100 + index)
        high_coverage = next((row for row in coverage_rows if row["candidate"] == name and row["tier"] == "high"), {})
        comparisons.append({
            "candidate": name, "margin_high_minus_low": margin, "ordering_high_minus_low": ordering,
            "high_pid_coverage": f(high_coverage.get("pid_coverage_fraction")),
            "high_pairs": int(high.sum()), "low_pairs": int(low.sum()),
        })
    supported = [
        item for item in comparisons
        if item["margin_high_minus_low"]["ci_low"] > 0
        and item["ordering_high_minus_low"]["ci_low"] > 0
        and item["high_pid_coverage"] >= 0.5
    ]
    q1 = "YES" if supported else ("INCONCLUSIVE" if comparisons else "NO")
    best = max(comparisons, key=lambda item: item["margin_high_minus_low"].get("difference", -float("inf"))) if comparisons else None
    if best is None:
        q2 = "INCONCLUSIVE"
    elif best["margin_high_minus_low"]["ci_low"] > 0 and best["ordering_high_minus_low"]["ci_low"] > 0:
        q2 = "YES"
    elif best["margin_high_minus_low"]["difference"] > 0 or best["ordering_high_minus_low"]["difference"] > 0:
        q2 = "MIXED"
    else:
        q2 = "INCONCLUSIVE"

    utility_comparisons: list[dict[str, Any]] = []
    for index, name in enumerate(validated[1:], start=1):
        score = candidates[name]["score"]
        high = tier_mask(score, "high") & (utility["available"] > 0)
        low = tier_mask(score, "low") & (utility["available"] > 0)
        late = pid_bootstrap_difference(utility["pair_late_alignment"], pids, high, low, seed_offset=9200 + index)
        any_high = pid_bootstrap_difference(utility["any_high_utility"], pids, high, low, seed_offset=9300 + index)
        utility_comparisons.append({"candidate": name, "late_alignment_high_minus_low": late, "any_high_utility_high_minus_low": any_high, "high_pairs": int(high.sum()), "low_pairs": int(low.sum())})
    utility_supported = [item for item in utility_comparisons if item["late_alignment_high_minus_low"]["ci_low"] > 0]
    if utility_supported:
        q3 = "YES"
    elif any(item["late_alignment_high_minus_low"]["difference"] > 0 for item in utility_comparisons):
        q3 = "MIXED"
    else:
        q3 = "INCONCLUSIVE"

    # Q4 is deliberately a separate test: high H_ID + high/low reliability.
    q4_candidates: list[dict[str, Any]] = []
    if best is not None:
        score = candidates[best["candidate"]]["score"]
        high_h = evidence["pair_h_id_within_pid"] >= 0.75
        high_r = tier_mask(score, "high")
        low_r = tier_mask(score, "low")
        high = high_h & high_r & (utility["available"] > 0)
        low = high_h & low_r & (utility["available"] > 0)
        q4_candidates.append({
            "candidate": best["candidate"],
            "late_alignment": pid_bootstrap_difference(utility["pair_late_alignment"], pids, high, low, seed_offset=9400),
            "any_high_utility": pid_bootstrap_difference(utility["any_high_utility"], pids, high, low, seed_offset=9401),
            "high_pairs": int(high.sum()), "low_pairs": int(low.sum()),
        })
    if q4_candidates and q4_candidates[0]["late_alignment"]["ci_low"] > 0 and q4_candidates[0]["any_high_utility"]["ci_low"] > 0:
        q4 = "YES"
    elif q4_candidates and (q4_candidates[0]["late_alignment"]["difference"] > 0 or q4_candidates[0]["any_high_utility"]["difference"] > 0):
        q4 = "MIXED"
    else:
        q4 = "INCONCLUSIVE"

    r3 = candidates["R3_CONSENSUS_MUTUAL_INTRA"]["score"]
    reliable = r3 >= 2.0 / 3.0
    low_h = reliable & (evidence["pair_h_id_within_pid"] <= 1.0 / 3.0)
    high_h = reliable & (evidence["pair_h_id_within_pid"] >= 2.0 / 3.0)
    # Easier minus harder: positive means the H_ID ordering agrees with lower
    # positive-negative margin / lower ordering correctness for harder pairs.
    h_margin = pid_bootstrap_difference(evidence["bio_margin"], pids, low_h, high_h, seed_offset=9500)
    h_order = pid_bootstrap_difference(evidence["bio_ordering"], pids, low_h, high_h, seed_offset=9501)
    if h_margin["ci_low"] > 0 and h_order["ci_low"] > 0:
        q5 = "YES"
    elif h_margin["difference"] > 0 or h_order["difference"] > 0:
        q5 = "MIXED"
    else:
        q5 = "INCONCLUSIVE"

    promotion = q1 == "YES" and q2 == "YES" and q3 in ("YES", "MIXED") and q4 == "YES"
    return {
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
        "q6": "PROMOTE" if promotion else "STOP",
        "final": "PROMOTE TO PHASE 6B" if promotion else "STOP — RELATION TRAINING NOT JUSTIFIED",
        "best_candidate": best["candidate"] if best else None,
        "candidate_comparisons": comparisons,
        "utility_comparisons": utility_comparisons,
        "high_h_reliability_utility_comparison": q4_candidates,
        "reliable_h_low_minus_high_margin": h_margin,
        "reliable_h_low_minus_high_ordering": h_order,
    }


def write_report(
    source_records: list[dict[str, Any]],
    model_summary: dict[str, Any],
    evidence: dict[str, np.ndarray],
    validation: dict[str, Any],
    utility: dict[str, np.ndarray],
    decisions: dict[str, Any],
) -> None:
    lines = [
        "# PHASE 6A2 — MULTI-EVIDENCE RELIABLE RELATION DISCOVERY",
        "",
        "## 1. Scope",
        "",
        "Offline-only LTCC train audit. No optimizer, backward pass, DS, DW, sampler change, relation loss, model update, or query/gallery/test data was used. PID labels define relation truth; evidence families estimate reliability; frozen H_ID estimates damage/difficulty.",
        "",
        "## 2. Frozen Phase 6A conclusion",
        "",
        "Phase 6A was inconclusive for a single Bio teacher: Bio retained some cross-clothes structure, but it was not a sufficient standalone reliability target. Phase 6A2 therefore does not turn Bio similarity into ground truth and does not enter training.",
        "",
        "## 3. Relation truth definition",
        "",
        f"Primary relation: same PID + different clothing. Controls: same PID + same clothing. Negative: different PID. The compact pair bank contains {len(evidence['pair_left'])} directed cross-clothes positives across {len(np.unique(evidence['pair_pid']))} PIDs; its target image selection is label-only and deterministic.",
        "",
        "## 4. Evidence source audit",
        "",
        "| Model | Bank images | Dim | Role | Reliability vote |",
        "|---|---:|---:|---|---|",
    ]
    for record in source_records:
        lines.append(f"| {record['name']} | {record['feature_count'] or 'NA'} | {record['feature_dimension'] or 'NA'} | {record['role']} | {'yes' if record['used_for_reliability_vote'] else 'no'} |")
    lines.extend([
        "",
        "Feature field: frozen `global_feature`, float32 L2 normalization. E4 deterministic augmentation and E5 temporal relation banks were unavailable and are explicitly excluded rather than imputed.",
        "",
        "## 5. Bio-only baseline",
        "",
        "| Model | Cross-clothes sim | Nearest-negative sim | Margin | Negative closer rate | Top-10 PID purity |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("ID-only", "Bio", "Local Failed Full", "Official Full"):
        item = model_summary.get(name, {})
        lines.append(
            f"| {name} | {report_num(item.get('cross_clothes_positive_similarity_mean'))} | {report_num(item.get('nearest_negative_similarity_mean'))} | "
            f"{report_num(item.get('identity_margin_mean'))} | {report_num(item.get('negative_closer_than_positive_rate'))} | {report_num(item.get('top10_same_pid_purity'))} |"
        )
    lines.extend([
        "",
        "Bio and ID-only are the only model evidence allowed to vote in R0–R4. Local Failed Full and Official Full are evaluation-only.",
        "",
        "## 6. Model consensus",
        "",
        f"Mean Bio/ID-only within-anchor rank-consensus evidence: {report_num(np.mean(evidence['consensus']))}; mean mutual-neighbor evidence over both voters and k=1,5,10: {report_num(np.mean(evidence['mutual_score']))}.",
        "",
        "The raw columns are in `model_consensus.csv`; no model agreement column replaces the label-defined relation truth.",
        "",
        "## 7. Mutual-neighbor consistency",
        "",
        "Mutual support requires both a→p and p→a membership. It is retained at k=1,5,10 for Bio and ID-only in `mutual_neighbor.csv`.",
        "",
        "## 8. Augmentation stability",
        "",
        "NOT AVAILABLE: no pre-existing deterministic multi-view relation bank with the same frozen protocol was present. No new forward extraction was introduced solely to fill this column.",
        "",
        "## 9. Temporal stability",
        "",
        "NOT AVAILABLE for relation similarity/rank. The existing Phase5D BEST-vs-S0 H_ID stability remains auxiliary context only and is not substituted for a second Bio relation bank.",
        "",
        "## 10. Intra-ID support",
        "",
        f"Mean pair-level intra-ID support evidence: {report_num(np.mean(evidence['intra_support']))}. The underlying support is computed against all same-PID/different-clothes train images at each endpoint, not only the compact representative.",
        "",
        "## 11. Reliability candidates",
        "",
        "- R0: Bio-only support percentile.",
        "- R1: equal-rank Bio support + Bio/ID consensus.",
        "- R2: equal-rank consensus + mutual-neighbor support.",
        "- R3: R2 evidence plus intra-ID support.",
        "- R4: available multi-evidence aggregate including positive/negative ordering; partial because augmentation and temporal evidence are unavailable.",
        "",
        "All candidates preserve relation truth. Conservative 2/3, 3/4, and 4/5 gates at thresholds 1/3, 1/2, and 2/3 are sensitivity analyses only.",
        "",
        "## 12. Reliability threshold sensitivity",
        "",
        "See `reliability_threshold_audit.csv` and `coverage_audit.csv`. The high/medium/low continuous tiers use rank percentiles <=1/3, middle, and >=2/3; no test metric is used.",
        "",
        "## 13. Independent validation",
        "",
        "Definition and validation are separated: Local Failed Full is introduced only after reliability candidates are formed; positive-negative ordering uses PID truth and nearest negative; Phase5D utility is an existing anchor-level gradient probe and no new backward is run.",
        "",
        "## 14. Positive-negative ordering",
        "",
        f"Compact-pair Bio ordering-correct rate: {report_num(np.mean(evidence['bio_ordering']))}; ID-only: {report_num(np.mean(evidence['id_ordering']))}; consensus ordering rate: {report_num(np.mean(evidence['ordering_consensus']))}.",
        "",
        "`ordering_validation.csv` contains PID-bootstrap high-minus-low comparisons for every continuous candidate.",
        "",
        "## 15. Failed-Full degradation analysis",
        "",
        f"Overall Failed Full minus Bio pair similarity: {report_num(np.mean(validation['relation_error']))}; margin difference: {report_num(np.mean(validation['margin_error']))}; top-10 same-PID purity difference: {report_num(np.mean(validation['neighbor_error']))}. Failed Full never votes on reliability.",
        "",
        "The reliability-stratified values and 95% PID bootstrap CIs are in `failed_full_degradation_by_reliability.csv`.",
        "",
        "## 16. Phase5D utility linkage",
        "",
        f"Existing Phase5D utility matched at least one endpoint for {int(np.sum(utility['available'] > 0))}/{len(utility['available'])} compact pairs (anchor matches: {int(np.sum(np.isfinite(utility['anchor_late_alignment'])))}; positive-endpoint matches: {int(np.sum(np.isfinite(utility['positive_late_alignment'])))}). These are sparse anchor-level probes, not pair-specific utility; they are not used to manufacture a positive result.",
        "",
        "## 17. H_ID × Reliability quadrants",
        "",
        "`hardness_reliability_quadrants.csv` separates high/low within-PID H_ID risk from high/low reliability tiers. High H_ID remains a damage detector, not a direct training priority.",
        "",
        "## 18. Reliable-Easy / Medium / Hard audit",
        "",
        "`reliable_hardness_curriculum_audit.csv` tests whether H_ID still orders difficulty after a conservative reliability screen. This is offline monotonicity evidence, not a training proposal.",
        "",
        "## 19. Coverage audit",
        "",
        "Coverage is reported in pairs, PIDs, and clothing states. A high-precision subset with narrow PID/state coverage is not sufficient for promotion.",
        "",
        "## 20. Future Relation DS v2",
        "",
        "The only admissible future design is reliability-first relation emphasis with all images and clothing states visible. It must not restore active-set withholding. See `future_relation_ds_v2.md`.",
        "",
        "## 21. Future Relation Loss v2",
        "",
        "Future loss interfaces are label-directional and relation-based; Bio/ID-only evidence may filter relations but does not force feature-vector imitation. See `future_relation_loss_v2.md`.",
        "",
        "## 22. Q1–Q6",
        "",
        f"- Q1 — Multi-evidence reliability clearly better than Bio-only: **{decisions['q1']}**.",
        f"- Q2 — High-reliability cross-clothes positives have better margin/order/stability: **{decisions['q2']}**.",
        f"- Q3 — High reliability is positively linked to existing Phase5D utility: **{decisions['q3']}**.",
        f"- Q4 — Reliability separates high-H_ID useful-hard from low-utility-hard: **{decisions['q4']}**.",
        f"- Q5 — Within high reliability, H_ID remains a suitable difficulty ordering: **{decisions['q5']}**.",
        f"- Q6 — Enter Phase 6B relation training: **{decisions['q6']}**.",
        "",
        "### Candidate high-vs-low comparisons",
        "",
        "| Candidate | Margin Δ high-low | 95% CI | Ordering Δ high-low | 95% CI | High PID coverage |",
        "|---|---:|---|---:|---|---:|",
    ])
    for item in decisions["candidate_comparisons"]:
        margin = item["margin_high_minus_low"]
        ordering = item["ordering_high_minus_low"]
        lines.append(
            f"| {item['candidate']} | {report_num(margin.get('difference'))} | [{report_num(margin.get('ci_low'))}, {report_num(margin.get('ci_high'))}] | "
            f"{report_num(ordering.get('difference'))} | [{report_num(ordering.get('ci_low'))}, {report_num(ordering.get('ci_high'))}] | {report_num(item.get('high_pid_coverage'))} |"
        )
    lines.extend([
        "",
        "## 23. Promotion gate",
        "",
        "The Phase 6B gate requires Q1=YES, Q2=YES (or strongly consistent MIXED), adequate coverage, no independent utility/stability contradiction, and a train-only reliability gate. Missing evidence is not treated as success.",
        "",
        f"Best candidate in the conservative comparison: `{decisions.get('best_candidate') or 'none'}`.",
        "",
        "## 24. Final diagnosis",
        "",
        f"# {decisions['final']}",
        "",
        "Relation truth remains label-defined; multi-evidence reliability is an audit object. No claim is made that a better offline relation subset will improve ReID until a separately approved Phase 6B training ablation is justified.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def analyze() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("PHASE6A2_LOAD_BANKS", flush=True)
    banks = {name: load_bank(path) for name, path in BANK_PATHS.items() if path.is_file()}
    required = ("Bio", "ID-only", "Local Failed Full")
    missing = [name for name in required if name not in banks]
    if missing:
        raise FileNotFoundError(f"missing required banks: {missing}")
    hardness = load_bank(P5D_BANK)
    check_alignment(banks, hardness)

    source_records = [
        source_record("Bio", BIO_CHECKPOINT, BANK_PATHS["Bio"], MODEL_ROLES["Bio"], True),
        source_record("ID-only", ID_CHECKPOINT, BANK_PATHS["ID-only"], MODEL_ROLES["ID-only"], True),
        source_record("Local Failed Full", FAILED_CHECKPOINT, BANK_PATHS["Local Failed Full"], MODEL_ROLES["Local Failed Full"], False),
        source_record("Official Full", OFFICIAL_CHECKPOINT, BANK_PATHS["Official Full"], MODEL_ROLES["Official Full"], False),
    ]
    h_record = {"path": P5D_BANK, "sha256": sha256_file(P5D_BANK), "feature_field": "h_id exact frozen Phase5D values"}
    write_source_audit(source_records, h_record)
    write_unavailable_evidence()

    pids = banks["Bio"]["pid"].astype(np.int16)
    clothes = banks["Bio"]["clothes_id"].astype(np.int16)
    cameras = banks["Bio"]["camera"].astype(np.int16)
    stems = banks["Bio"]["stems"]
    cross_pairs, same_pairs, _ = build_label_pairs(pids, clothes)
    cross_left, cross_right = pair_arrays(cross_pairs)
    same_left, same_right = pair_arrays(same_pairs)
    print(f"PHASE6A2_PAIRS cross={len(cross_left)} same={len(same_left)}", flush=True)

    relation_stats = {
        name: p6a.cross_clothes_relations(bank["features"], pids, clothes)
        for name, bank in banks.items()
    }
    _same_stats = {
        name: p6a.same_clothes_relations(bank["features"], pids, clothes)
        for name, bank in banks.items()
    }
    negative_values, negative_indices = nearest_negative_arrays(banks, pids)
    write_pair_bank(cross_left, cross_right, same_left, same_right, negative_indices["Bio"], stems, pids, clothes, cameras)

    print("PHASE6A2_TOP_NEIGHBORS", flush=True)
    bio_top = p6a.top_neighbors(banks["Bio"]["features"], k=TOP_K, chunk_size=256)
    id_top = p6a.top_neighbors(banks["ID-only"]["features"], k=TOP_K, chunk_size=256)
    failed_top = p6a.top_neighbors(banks["Local Failed Full"]["features"], k=TOP_K, chunk_size=256)
    evidence = evidence_support_arrays(
        relation_stats, pids, cross_left, cross_right, bio_top, id_top,
        negative_indices["Bio"], negative_indices["ID-only"], banks, hardness,
    )
    evidence["pair_left"] = cross_left
    evidence["pair_right"] = cross_right
    evidence["pair_pid"] = pids[cross_left]
    evidence["pair_cloth_a"] = clothes[cross_left]
    evidence["pair_cloth_b"] = clothes[cross_right]

    write_positive_and_evidence_artifacts(
        cross_left, cross_right, same_left, same_right, evidence, relation_stats,
        negative_indices, banks, hardness, bio_top, id_top,
    )
    candidates = write_reliability_candidates(evidence)
    write_threshold_audit(evidence, candidates)
    utility = write_phase5d_join(cross_left, cross_right, pids[cross_left])
    validation = write_validation_tables(candidates, evidence, banks, negative_indices, bio_top, failed_top, utility)
    write_quadrants_and_curriculum(candidates, evidence, validation, utility)
    coverage_rows = write_coverage_audit(candidates, evidence)
    model_summary = write_model_summary(banks, relation_stats, negative_values, pids, clothes)
    write_future_design_docs()
    decisions = choose_diagnosis(candidates, evidence, utility, coverage_rows)
    write_report(source_records, model_summary, evidence, validation, utility, decisions)

    manifest = {
        "phase": "phase6a2_multi_evidence_reliability",
        "created_utc": utc_now(), "seed": SEED,
        "scope": "offline train-only relation discovery; no optimizer/backward/update",
        "dataset": {"name": "LTCC", "split": "train", "images": IMAGE_COUNT, "pids": PID_COUNT, "clothing_states": STATE_COUNT, "test_used": False, "query_gallery_used": False},
        "relation_truth": {"positive_primary": "same PID + different clothes", "positive_control": "same PID + same clothes", "negative": "different PID", "pair_selection": "first image per target clothing state, labels only"},
        "representation": {"field": "global_feature", "normalization": "float32 L2", "relation_feature_dimension": 1024},
        "models": source_records, "hardness_source": h_record,
        "evidence_families": {"E1_bio_support": True, "E2_bio_id_consensus": True, "E3_mutual_neighbor": True, "E4_augmentation": False, "E5_temporal": False, "E6_intra_id_support": True, "negative_ordering": True},
        "reliability_candidates": [name for name in candidates if not name.startswith("gate_")],
        "bootstrap": {"unit": "PID", "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "phase5d_utility": {
            "path": P5D_UTILITY, "exists": P5D_UTILITY.is_file(), "new_backward": False, "pair_specific": False,
            "matched_pair_count": int(np.sum(utility["available"] > 0)),
            "anchor_endpoint_matches": int(np.sum(np.isfinite(utility["anchor_late_alignment"]))),
            "positive_endpoint_matches": int(np.sum(np.isfinite(utility["positive_late_alignment"]))),
        },
        "decisions": decisions,
        "prohibitions": {"training": True, "optimizer": True, "optimizer_step": True, "ds": True, "dw": True, "sampler": True, "relation_loss": True, "grl": True, "online_h_id": True, "prcc": True, "test_leakage": True},
        "outputs": sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.rglob("*") if path.is_file()),
    }
    write_json(OUT_ROOT / "PHASE6A2_MANIFEST.json", manifest)
    return {"decisions": decisions, "manifest": manifest, "validation": validation}


def main() -> None:
    result = analyze()
    print(json.dumps(json_safe(result["decisions"]), indent=2, sort_keys=True), flush=True)
    print(f"PHASE6A2_REPORT {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
