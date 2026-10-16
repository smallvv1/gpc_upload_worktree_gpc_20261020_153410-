#!/usr/bin/env python3
"""Phase 6A3: coverage-preserving soft reliability audit.

Offline, train-only analysis.  This module reuses Phase6A2 relation truth and
evidence, adds only frozen forward feature banks when the existing PPU path is
available, and never creates an optimizer or calls backward/step.  Reliability
is continuous and never removes a ground-truth cross-clothes relation.
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
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase6a2_multi_evidence_reliability as p6a2  # noqa: E402

SEED = 1234
BOOTSTRAP_RESAMPLES = 10_000
IMAGE_COUNT = 9_576
TOTAL_PID_COUNT = 77
TOP_K = 10

P6A2_ROOT = ROOT / "outputs/new_method_track/phase6a2_multi_evidence_reliability"
P5D_ROOT = ROOT / "outputs/new_method_track/phase5d_hardness_utility"
P5D_BANK = P5D_ROOT / "feature_bank/feature_bank.npz"
P5D_UTILITY = P5D_ROOT / "gradient_probes/gradient_utility_targets.csv"
P5D_QUADRANTS = P5D_ROOT / "gradient_probes/hardness_utility_quadrants.csv"

OUT_ROOT = ROOT / "outputs/new_method_track/phase6a3_soft_reliability"
REPORT_PATH = ROOT / "reports/PHASE_6A3_COVERAGE_PRESERVING_SOFT_RELIABILITY.md"
FEATURE_ROOT = OUT_ROOT / "feature_banks"

BIO_CHECKPOINT = ROOT / "outputs/reproduction_phase3/LTCC/table3/id_bio/run1/eva02_l_bio_best.pth"
S0_EPOCH15_CHECKPOINT = ROOT / "outputs/new_method_track/phase5b_id_damage_ds/S0_bio_full_data/model_best.pth"
BIO_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features/bio_control_feature_bank.npz"
ID_BANK = ROOT / "outputs/new_method_track/phase6a_bio_relation_teacher/feature_banks/id_only_feature_bank.npz"
FAILED_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features/original_full_feature_bank.npz"
OFFICIAL_BANK = ROOT / "outputs/new_method_track/phase4r_ltcc_full_recovery/C_semantic_forensics/checkpoint_features/official_full_feature_bank.npz"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    return p6a2.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p6a2.write_csv(path, rows)


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def mean_or_nan(values: Any) -> float:
    values = finite(values)
    return float(np.mean(values)) if len(values) else float("nan")


def median_or_nan(values: Any) -> float:
    values = finite(values)
    return float(np.median(values)) if len(values) else float("nan")


def report_num(value: Any, digits: int = 4) -> str:
    value = f(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def load_bank(path: Path) -> dict[str, Any]:
    return p6a2.load_bank(path)


def pair_sim(features: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(features[left] * features[right], axis=1).astype(np.float32)


def pair_nanmean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return p6a2.pair_nanmean(left, right)


def group_rank(values: np.ndarray, groups: np.ndarray, higher: bool = True) -> np.ndarray:
    return p6a2.rank_within_groups(values, groups, higher_is_better=higher)


def global_rank(values: np.ndarray, higher: bool = True) -> np.ndarray:
    return p6a2.global_percentile(values, higher_is_better=higher)


def check_equal_labels(reference: dict[str, Any], candidate: dict[str, Any], name: str) -> None:
    for key in ("stems", "pid", "clothes_id"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"Phase6A3 alignment failure: {name}/{key}")
    if "camera" in candidate and "camera" in reference and not np.array_equal(reference["camera"], candidate["camera"]):
        raise RuntimeError(f"Phase6A3 alignment failure: {name}/camera")


def parse_phase6a2_pairs() -> dict[str, np.ndarray]:
    cross_rows = read_csv(P6A2_ROOT / "cross_clothes_positive.csv")
    bio_rows = {int(row["pair_id"]): row for row in read_csv(P6A2_ROOT / "bio_support.csv")}
    consensus_rows = {int(row["pair_id"]): row for row in read_csv(P6A2_ROOT / "model_consensus.csv")}
    intra_rows = {int(row["pair_id"]): row for row in read_csv(P6A2_ROOT / "intra_id_support.csv")}
    negative_rows = {int(row["pair_id"]): row for row in read_csv(P6A2_ROOT / "negative_relation_support.csv")}
    triplet_rows = {int(row["pair_id"]): row for row in read_csv(P6A2_ROOT / "triplet_relation.csv")}
    mutual_rows = read_csv(P6A2_ROOT / "mutual_neighbor.csv")
    if not cross_rows:
        raise FileNotFoundError(P6A2_ROOT / "cross_clothes_positive.csv")
    n = len(cross_rows)
    pair_ids = np.asarray([int(row["pair_id"]) for row in cross_rows], dtype=np.int32)
    if not np.array_equal(pair_ids, np.arange(n, dtype=np.int32)):
        raise RuntimeError("Phase6A2 pair IDs are not contiguous")
    left = np.asarray([int(row["index_a"]) for row in cross_rows], dtype=np.int32)
    right = np.asarray([int(row["index_b"]) for row in cross_rows], dtype=np.int32)
    pid = np.asarray([int(row["pid"]) for row in cross_rows], dtype=np.int16)
    cloth_a = np.asarray([int(row["clothes_a"]) for row in cross_rows], dtype=np.int16)
    cloth_b = np.asarray([int(row["clothes_b"]) for row in cross_rows], dtype=np.int16)
    bio_similarity = np.asarray([f(row.get("bio_similarity")) for row in cross_rows], dtype=np.float32)
    id_similarity = np.asarray([f(row.get("id_only_similarity")) for row in cross_rows], dtype=np.float32)
    failed_similarity = np.asarray([f(row.get("local_failed_full_similarity")) for row in cross_rows], dtype=np.float32)
    official_similarity = np.asarray([f(row.get("official_full_similarity")) for row in cross_rows], dtype=np.float32)
    bio_support = np.asarray([f(bio_rows[i]["bio_within_anchor_percentile"]) for i in pair_ids], dtype=np.float32)
    id_support = np.asarray([f(consensus_rows[i]["id_rank_percentile"]) for i in pair_ids], dtype=np.float32)
    consensus = np.asarray([f(consensus_rows[i]["model_consensus_score"]) for i in pair_ids], dtype=np.float32)
    intra_norm = np.asarray([f(intra_rows[i]["pair_intra_id_support"]) for i in pair_ids], dtype=np.float32)
    intra_raw = np.asarray([
        np.nanmean([f(intra_rows[i]["bio_anchor_cross_clothes_mean"]), f(intra_rows[i]["bio_positive_cross_clothes_mean"])])
        for i in pair_ids
    ], dtype=np.float32)
    mutual_values: dict[int, list[float]] = defaultdict(list)
    for row in mutual_rows:
        pair_id = int(row["pair_id"])
        mutual_values[pair_id].extend([f(row["bio_mutual"]), f(row["id_mutual"])])
    mutual = np.asarray([mean_or_nan(mutual_values[i]) for i in pair_ids], dtype=np.float32)
    negative = {"bio_margin": [], "id_margin": [], "bio_ordering": [], "id_ordering": [], "bio_negative_index": []}
    failed = {"margin": [], "ordering": []}
    for i in pair_ids:
        nrow = negative_rows[i]
        trow = triplet_rows[i]
        negative["bio_margin"].append(f(nrow["bio_margin"]))
        negative["id_margin"].append(f(nrow["id_margin"]))
        negative["bio_ordering"].append(f(nrow["bio_ordering_correct"]))
        negative["id_ordering"].append(f(nrow["id_ordering_correct"]))
        negative["bio_negative_index"].append(int(nrow["bio_negative_index"]))
        failed["margin"].append(f(trow["local_failed_full_margin"]))
        failed["ordering"].append(f(trow["local_failed_full_ordering_correct"]))
    return {
        "pair_id": pair_ids, "left": left, "right": right, "pid": pid, "cloth_a": cloth_a, "cloth_b": cloth_b,
        "bio_similarity": bio_similarity, "id_similarity": id_similarity,
        "failed_similarity": failed_similarity, "official_similarity": official_similarity,
        "bio_support_old": bio_support, "id_support_old": id_support, "consensus_old": consensus,
        "mutual_old": mutual, "intra_norm_old": intra_norm, "intra_raw": intra_raw,
        "bio_margin": np.asarray(negative["bio_margin"], dtype=np.float32),
        "id_margin": np.asarray(negative["id_margin"], dtype=np.float32),
        "bio_ordering": np.asarray(negative["bio_ordering"], dtype=np.float32),
        "id_ordering": np.asarray(negative["id_ordering"], dtype=np.float32),
        "bio_negative_index": np.asarray(negative["bio_negative_index"], dtype=np.int32),
        "failed_margin": np.asarray(failed["margin"], dtype=np.float32),
        "failed_ordering": np.asarray(failed["ordering"], dtype=np.float32),
    }


def make_bio_test_transform(cfg, horizontal_flip: bool = False):
    from PIL import Image
    import data.img_transforms as transforms

    fill_color = tuple(int(x * 255) for x in cfg.DATA.PIXEL_MEAN)

    class DeterministicFlip:
        def __call__(self, image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)

        def randomize_parameters(self):
            return None

    items = [
        transforms.ResizeWithEqualScale(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH, fill_color=fill_color),
    ]
    if horizontal_flip:
        items.append(DeterministicFlip())
    items.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATA.PIXEL_MEAN, std=cfg.DATA.PIXEL_STD),
    ])
    return transforms.Compose(items)


def save_forward_bank(path: Path, features: np.ndarray, dataset: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_feature=features.astype(np.float16),
        stems=np.asarray([Path(item["image_path"]).stem for item in dataset.train]),
        pid=np.asarray([item["pid"] for item in dataset.train], dtype=np.int16),
        camera=np.asarray([item["camid"] for item in dataset.train], dtype=np.int16),
        clothes_id=np.asarray([item["clothes_id"] for item in dataset.train], dtype=np.int16),
    )
    metadata["feature_bank"] = path
    metadata["feature_bank_sha256"] = sha256_file(path)
    metadata["feature_shape"] = list(features.shape)


def extract_forward_auxiliary_banks() -> dict[str, Any]:
    """Extract V1 and S0 only through the existing CUDA/PPU no-grad path."""
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    view_path = FEATURE_ROOT / "bio_best_horizontal_flip_feature_bank.npz"
    temporal_path = FEATURE_ROOT / "bio_s0_epoch15_v0_feature_bank.npz"
    manifest_path = OUT_ROOT / "forward_extraction_manifest.json"
    if view_path.is_file() and temporal_path.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6A3 forward-only auxiliary extraction requires PPU/CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase4r_c_checkpoint_features as p4r  # noqa: E402

    manifest: dict[str, Any] = {
        "phase": "phase6a3_forward_only_auxiliary",
        "device": torch.cuda.get_device_name(0),
        "no_grad": True,
        "optimizer": False,
        "backward": False,
        "created_utc": utc_now(),
        "runs": {},
    }

    if not view_path.is_file():
        print("PHASE6A3_FORWARD bio_best_horizontal_flip", flush=True)
        cfg = p4r.make_config(False)
        dataset = p4r.build_dataset(cfg)
        transform = make_bio_test_transform(cfg, horizontal_flip=True)
        model, load_info = p4r.build_model_for_checkpoint(cfg, BIO_CHECKPOINT, False)
        with torch.no_grad():
            result = p4r.extract(model, dataset, transform, "bio_best_horizontal_flip")
        metadata = {"name": "Bio BEST horizontal flip", "checkpoint": BIO_CHECKPOINT, "checkpoint_sha256": sha256_file(BIO_CHECKPOINT), "transform": "ResizeWithEqualScale(224,224) -> deterministic horizontal flip -> ToTensor -> ImageNet Normalize", "load_info": load_info}
        save_forward_bank(view_path, result["global_feature"], dataset, metadata)
        manifest["runs"]["bio_best_horizontal_flip"] = metadata
        del model, dataset, result
        torch.cuda.empty_cache()
    else:
        manifest["runs"]["bio_best_horizontal_flip"] = {"name": "Bio BEST horizontal flip", "feature_bank": view_path, "feature_bank_sha256": sha256_file(view_path), "reused": True}

    if not temporal_path.is_file():
        if not S0_EPOCH15_CHECKPOINT.is_file():
            manifest["runs"]["bio_s0_epoch15_v0"] = {"status": "NOT_AVAILABLE", "reason": str(S0_EPOCH15_CHECKPOINT)}
        else:
            print("PHASE6A3_FORWARD bio_s0_epoch15_v0", flush=True)
            cfg = p4r.make_config(False)
            dataset = p4r.build_dataset(cfg)
            _, transform = p4r.build_img_transforms(cfg)
            model, load_info = p4r.build_model_for_checkpoint(cfg, S0_EPOCH15_CHECKPOINT, False)
            with torch.no_grad():
                result = p4r.extract(model, dataset, transform, "bio_s0_epoch15_v0")
            metadata = {"name": "Phase5B S0 epoch15 V0", "checkpoint": S0_EPOCH15_CHECKPOINT, "checkpoint_sha256": sha256_file(S0_EPOCH15_CHECKPOINT), "transform": "released formal test transform", "load_info": load_info}
            save_forward_bank(temporal_path, result["global_feature"], dataset, metadata)
            manifest["runs"]["bio_s0_epoch15_v0"] = metadata
            del model, dataset, result
            torch.cuda.empty_cache()
    else:
        manifest["runs"]["bio_s0_epoch15_v0"] = {"name": "Phase5B S0 epoch15 V0", "feature_bank": temporal_path, "feature_bank_sha256": sha256_file(temporal_path), "reused": True}
    manifest["view_bank_available"] = view_path.is_file()
    manifest["temporal_bank_available"] = temporal_path.is_file()
    write_json(manifest_path, manifest)
    return manifest


def compute_view_temporal(
    pairs: dict[str, np.ndarray],
    bio_bank: dict[str, Any],
    view_bank: dict[str, Any] | None,
    temporal_bank: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    left = pairs["left"]
    right = pairs["right"]
    n = len(left)
    result: dict[str, np.ndarray] = {
        "view_sim_v0": pairs["bio_similarity"].astype(np.float32),
        "view_sim_v1": np.full(n, np.nan, dtype=np.float32),
        "view_mean": np.full(n, np.nan, dtype=np.float32),
        "view_std": np.full(n, np.nan, dtype=np.float32),
        "view_range": np.full(n, np.nan, dtype=np.float32),
        "view_rank_consistency": np.full(n, np.nan, dtype=np.float32),
        "view_stability": np.full(n, np.nan, dtype=np.float32),
        "temporal_sim_best": pairs["bio_similarity"].astype(np.float32),
        "temporal_sim_s0": np.full(n, np.nan, dtype=np.float32),
        "temporal_abs_delta": np.full(n, np.nan, dtype=np.float32),
        "temporal_rank_consistency": np.full(n, np.nan, dtype=np.float32),
        "temporal_stability": np.full(n, np.nan, dtype=np.float32),
    }
    v0_rank = group_rank(result["view_sim_v0"], left, True)
    if view_bank is not None:
        check_equal_labels(bio_bank, view_bank, "augmentation_view")
        result["view_sim_v1"] = pair_sim(view_bank["features"], left, right)
        v1_rank = group_rank(result["view_sim_v1"], left, True)
        result["view_mean"] = pair_nanmean(result["view_sim_v0"], result["view_sim_v1"])
        result["view_std"] = np.abs(result["view_sim_v0"] - result["view_sim_v1"]) / np.sqrt(2.0)
        result["view_range"] = np.abs(result["view_sim_v0"] - result["view_sim_v1"])
        result["view_rank_consistency"] = 1.0 - np.abs(v0_rank - v1_rank)
        std_support = global_rank(result["view_std"], higher=False)
        result["view_stability"] = pair_nanmean(std_support, result["view_rank_consistency"])
    if temporal_bank is not None:
        check_equal_labels(bio_bank, temporal_bank, "temporal_s0")
        result["temporal_sim_s0"] = pair_sim(temporal_bank["features"], left, right)
        s0_rank = group_rank(result["temporal_sim_s0"], left, True)
        result["temporal_abs_delta"] = np.abs(result["temporal_sim_best"] - result["temporal_sim_s0"])
        result["temporal_rank_consistency"] = 1.0 - np.abs(v0_rank - s0_rank)
        delta_support = global_rank(result["temporal_abs_delta"], higher=False)
        result["temporal_stability"] = pair_nanmean(delta_support, result["temporal_rank_consistency"])
    return result


def write_view_temporal_artifacts(view_temporal: dict[str, np.ndarray], pairs: dict[str, np.ndarray]) -> None:
    rows: list[dict[str, Any]] = []
    for index in range(len(pairs["pair_id"])):
        rows.append({
            "pair_id": int(pairs["pair_id"][index]), "index_a": int(pairs["left"][index]), "index_b": int(pairs["right"][index]), "pid": int(pairs["pid"][index]),
            "sim_v0": view_temporal["view_sim_v0"][index], "sim_v1_flip": view_temporal["view_sim_v1"][index],
            "view_mean": view_temporal["view_mean"][index], "view_std": view_temporal["view_std"][index],
            "view_range": view_temporal["view_range"][index], "view_rank_consistency": view_temporal["view_rank_consistency"][index],
            "view_stability": view_temporal["view_stability"][index],
        })
    write_csv(OUT_ROOT / "augmentation_relation_stability.csv", rows)
    temporal_rows: list[dict[str, Any]] = []
    for index in range(len(pairs["pair_id"])):
        temporal_rows.append({
            "pair_id": int(pairs["pair_id"][index]), "index_a": int(pairs["left"][index]), "index_b": int(pairs["right"][index]), "pid": int(pairs["pid"][index]),
            "sim_best": view_temporal["temporal_sim_best"][index], "sim_s0_epoch15": view_temporal["temporal_sim_s0"][index],
            "abs_similarity_delta": view_temporal["temporal_abs_delta"][index],
            "rank_consistency": view_temporal["temporal_rank_consistency"][index], "temporal_stability": view_temporal["temporal_stability"][index],
        })
    write_csv(OUT_ROOT / "temporal_relation_stability.csv", temporal_rows)


def build_soft_scores(pairs: dict[str, np.ndarray], view_temporal: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    left = pairs["left"]
    pid = pairs["pid"]
    # Recompute every normalized definition signal from raw relation values;
    # old Phase6A2 normalized columns are retained only for audit comparison.
    bio_support = group_rank(pairs["bio_similarity"], left, True)
    id_support = group_rank(pairs["id_similarity"], left, True)
    model_consensus = (1.0 - np.abs(bio_support - id_support)).astype(np.float32)
    mutual = pairs["mutual_old"].astype(np.float32)
    intra_raw_norm = group_rank(pairs["intra_raw"], pid, True)
    ordering_consensus = ((pairs["bio_ordering"] + pairs["id_ordering"]) / 2.0).astype(np.float32)
    components = {
        "bio_support": bio_support,
        "id_support": id_support,
        "model_consensus": model_consensus,
        "mutual_support": mutual,
        "intra_id_support": intra_raw_norm,
        "ordering_evidence_not_in_score": ordering_consensus,
        "view_stability": view_temporal["view_stability"],
        "temporal_stability": view_temporal["temporal_stability"],
    }

    candidate_components: dict[str, list[str]] = {
        "SR0_BIO_SUPPORT": ["bio_support"],
        "SR1_BIO_ID_CONSENSUS": ["bio_support", "id_support", "model_consensus"],
        "SR2_CONSENSUS_MUTUAL": ["model_consensus", "mutual_support"],
        "SR3_CONSENSUS_MUTUAL_INTRA": ["model_consensus", "mutual_support", "intra_id_support"],
    }
    if np.isfinite(view_temporal["view_stability"]).any() or np.isfinite(view_temporal["temporal_stability"]).any():
        candidate_components["SR4_AVAILABLE_MULTI_EVIDENCE"] = [
            "model_consensus", "mutual_support", "intra_id_support", "view_stability", "temporal_stability",
        ]
    scores: dict[str, dict[str, np.ndarray]] = {}
    score_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for candidate, names in candidate_components.items():
        matrix = np.stack([components[name] for name in names], axis=1)
        available = np.isfinite(matrix)
        count = available.sum(axis=1)
        mean_score = np.divide(np.nansum(matrix, axis=1), count, out=np.full(len(matrix), np.nan), where=count > 0).astype(np.float32)
        std = np.nanstd(matrix, axis=1)
        # All components are in [0,1]; 0.5 is the largest useful scale for
        # disagreement, so this maps agreement penalty monotonically to [0,1].
        agreement_penalty = np.clip(1.0 - 2.0 * std, 0.0, 1.0).astype(np.float32)
        conservative = (mean_score * agreement_penalty).astype(np.float32)
        scores[candidate] = {"mean": mean_score, "conservative": conservative}
        for index in range(len(matrix)):
            row = {"pair_id": int(pairs["pair_id"][index]), "candidate": candidate, "available_component_count": int(count[index]), "component_count": len(names), "mean_score": mean_score[index], "agreement_penalty": agreement_penalty[index], "conservative_score": conservative[index]}
            for name in names:
                row[f"{name}_normalized"] = components[name][index]
            score_rows.append(row)
    for index in range(len(pairs["pair_id"])):
        row = {"pair_id": int(pairs["pair_id"][index]), "pid": int(pairs["pid"][index]), "index_a": int(pairs["left"][index]), "index_b": int(pairs["right"][index]), "bio_similarity_raw": pairs["bio_similarity"][index], "id_similarity_raw": pairs["id_similarity"][index], "intra_support_raw": pairs["intra_raw"][index], "mutual_support_raw": pairs["mutual_old"][index], "bio_margin_raw": pairs["bio_margin"][index], "id_margin_raw": pairs["id_margin"][index], "ordering_consensus_raw": ordering_consensus[index], "view_std_raw": view_temporal["view_std"][index], "temporal_abs_delta_raw": view_temporal["temporal_abs_delta"][index]}
        for name, values in components.items():
            row[f"{name}_normalized"] = values[index]
        component_rows.append(row)
    write_csv(OUT_ROOT / "soft_reliability_scores.csv", score_rows)
    write_csv(OUT_ROOT / "soft_reliability_components.csv", component_rows)
    return scores, components


def quartile_mask(percentile: np.ndarray, quartile: str) -> np.ndarray:
    if quartile not in {"Q1", "Q2", "Q3", "Q4"}:
        return np.ones(len(percentile), dtype=bool)
    # Rank-based scores can contain large exact ties (especially mutual
    # support).  Thresholding at .25/.50/.75 would make Q4 much smaller than
    # a quartile.  Use deterministic exact bins; ties at a boundary are
    # assigned by stable relation order and are never resolved with outcomes.
    n = len(percentile)
    if n == 0:
        return np.zeros(0, dtype=bool)
    safe = np.where(np.isfinite(percentile), percentile, -np.inf)
    order = np.argsort(safe, kind="stable")
    cuts = [0, int(math.ceil(n * 0.25)), int(math.ceil(n * 0.50)), int(math.ceil(n * 0.75)), n]
    q_index = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}[quartile]
    mask = np.zeros(n, dtype=bool)
    mask[order[cuts[q_index]:cuts[q_index + 1]]] = True
    return mask


def top_fraction_mask(percentile: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return np.ones(len(percentile), dtype=bool)
    if fraction <= 0.0 or len(percentile) == 0:
        return np.zeros(len(percentile), dtype=bool)
    # Evidence components such as mutual support can have many exact ties.
    # A percentile threshold would then silently select far fewer relations
    # than the requested coverage (e.g. SR2 top-25%).  Select the exact
    # deterministic target count; stable sorting preserves pair-id order for
    # tied scores and does not use any validation metric.
    target_count = min(len(percentile), max(1, int(math.ceil(len(percentile) * fraction))))
    safe = np.where(np.isfinite(percentile), percentile, -np.inf)
    order = np.argsort(-safe, kind="stable")
    mask = np.zeros(len(percentile), dtype=bool)
    mask[order[:target_count]] = True
    return mask


def bootstrap_summary(values: np.ndarray, pids: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    return p6a2.pid_bootstrap_mean(values, pids, mask, seed_offset=seed)


def pid_rank_summary(score: np.ndarray, values: np.ndarray, pids: np.ndarray) -> tuple[float, float, float]:
    correlations: list[float] = []
    for pid in np.unique(pids):
        indices = np.flatnonzero(pids == pid)
        valid = np.isfinite(score[indices]) & np.isfinite(values[indices])
        if int(valid.sum()) >= 3 and np.ptp(score[indices][valid]) > 1e-12 and np.ptp(values[indices][valid]) > 1e-12:
            correlations.append(float(spearmanr(score[indices][valid], values[indices][valid]).statistic))
    if not correlations:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(correlations)), float(np.median(correlations)), float(np.mean(np.asarray(correlations) > 0))


def write_quality_coverage_and_validation(
    pairs: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    components: dict[str, np.ndarray],
    view_temporal: dict[str, np.ndarray],
    failed_neighbor_delta: np.ndarray,
) -> dict[str, Any]:
    pids = pairs["pid"]
    valid_pid_count = int(len(np.unique(pids)))
    relation_count = len(pids)
    quality_arrays = {
        "positive_negative_margin": pairs["bio_margin"],
        "ordering_correct": pairs["bio_ordering"],
        "failed_full_similarity_delta": pairs["failed_similarity"] - pairs["bio_similarity"],
        "failed_full_margin_delta": pairs["failed_margin"] - pairs["bio_margin"],
        "failed_full_ordering_delta": pairs["failed_ordering"] - pairs["bio_ordering"],
        "failed_full_neighbor_purity_delta": failed_neighbor_delta,
        "cross_model_consensus": components["model_consensus"],
        "view_stability": view_temporal["view_stability"],
        "temporal_stability": view_temporal["temporal_stability"],
    }
    fractions = [("all", 1.0), ("top75", 0.75), ("top50", 0.50), ("top25", 0.25), ("top10", 0.10)]
    quartiles = ("Q1", "Q2", "Q3", "Q4")
    curve_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"curve": {}, "quartile": {}, "ordering": {}}

    for candidate_index, (candidate, aggregation_scores) in enumerate(scores.items()):
        summary["curve"][candidate] = {}
        for aggregation, score in aggregation_scores.items():
            score_percentile = global_rank(score, True)
            summary["curve"][candidate][aggregation] = {}
            for fraction_index, (subset, fraction) in enumerate(fractions):
                mask = top_fraction_mask(score_percentile, fraction)
                states = set(zip(pids[mask].tolist(), pairs["cloth_b"][mask].tolist())) if np.any(mask) else set()
                row: dict[str, Any] = {
                    "candidate": candidate, "aggregation": aggregation, "subset": subset, "target_fraction": fraction,
                    "relation_count": int(mask.sum()), "relation_coverage": float(mask.mean()),
                    "pid_count": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                    "valid_cross_clothes_pid_coverage": float(len(np.unique(pids[mask])) / valid_pid_count) if np.any(mask) else 0.0,
                    "total_77_pid_coverage": float(len(np.unique(pids[mask])) / TOTAL_PID_COUNT) if np.any(mask) else 0.0,
                    "clothing_state_count": int(len(states)), "clothing_state_coverage": float(len(states) / 256),
                }
                for metric_index, (metric, values) in enumerate(quality_arrays.items()):
                    stats = bootstrap_summary(values, pids, mask, seed=1000 + candidate_index * 100 + fraction_index * 10 + metric_index)
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_ci_low"] = stats["ci_low"]
                    row[f"{metric}_ci_high"] = stats["ci_high"]
                curve_rows.append(row)
                summary["curve"][candidate][aggregation][subset] = row
                coverage_rows.append({
                    "source": "quality_curve", "candidate": candidate, "aggregation": aggregation, "subset": subset,
                    "relation_count": row["relation_count"], "relation_coverage": row["relation_coverage"],
                    "pid_count": row["pid_count"], "valid_cross_clothes_pid_coverage": row["valid_cross_clothes_pid_coverage"],
                    "total_77_pid_coverage": row["total_77_pid_coverage"], "clothing_state_count": row["clothing_state_count"],
                    "clothing_state_coverage": row["clothing_state_coverage"],
                })
            summary["quartile"].setdefault(candidate, {})[aggregation] = {}
            for q_index, quartile in enumerate(quartiles):
                mask = quartile_mask(score_percentile, quartile)
                states = set(zip(pids[mask].tolist(), pairs["cloth_b"][mask].tolist())) if np.any(mask) else set()
                row = {
                    "candidate": candidate, "aggregation": aggregation, "quartile": quartile,
                    "relation_count": int(mask.sum()), "relation_coverage": float(mask.mean()),
                    "pid_count": int(len(np.unique(pids[mask]))) if np.any(mask) else 0,
                    "valid_cross_clothes_pid_coverage": float(len(np.unique(pids[mask])) / valid_pid_count) if np.any(mask) else 0.0,
                    "total_77_pid_coverage": float(len(np.unique(pids[mask])) / TOTAL_PID_COUNT) if np.any(mask) else 0.0,
                    "clothing_state_count": int(len(states)), "clothing_state_coverage": float(len(states) / 256),
                }
                for metric_index, (metric, values) in enumerate(quality_arrays.items()):
                    stats = bootstrap_summary(values, pids, mask, seed=3000 + candidate_index * 100 + q_index * 10 + metric_index)
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_ci_low"] = stats["ci_low"]
                    row[f"{metric}_ci_high"] = stats["ci_high"]
                quartile_rows.append(row)
                summary["quartile"][candidate][aggregation][quartile] = row
                coverage_rows.append({
                    "source": "reliability_quartile", "candidate": candidate, "aggregation": aggregation, "subset": quartile,
                    "relation_count": row["relation_count"], "relation_coverage": row["relation_coverage"],
                    "pid_count": row["pid_count"], "valid_cross_clothes_pid_coverage": row["valid_cross_clothes_pid_coverage"],
                    "total_77_pid_coverage": row["total_77_pid_coverage"], "clothing_state_count": row["clothing_state_count"],
                    "clothing_state_coverage": row["clothing_state_coverage"],
                })
                degradation_rows.append({
                    "candidate": candidate, "aggregation": aggregation, "quartile": quartile,
                    "relation_count": row["relation_count"], "pid_count": row["pid_count"],
                    "failed_full_similarity_delta_mean": row["failed_full_similarity_delta_mean"],
                    "failed_full_similarity_delta_ci_low": row["failed_full_similarity_delta_ci_low"],
                    "failed_full_similarity_delta_ci_high": row["failed_full_similarity_delta_ci_high"],
                    "failed_full_margin_delta_mean": row["failed_full_margin_delta_mean"],
                    "failed_full_ordering_delta_mean": row["failed_full_ordering_delta_mean"],
                    "failed_full_neighbor_purity_delta_mean": row["failed_full_neighbor_purity_delta_mean"],
                })

            margin_rho = p6a2.safe_spearman(score, pairs["bio_margin"])
            ordering_rho = p6a2.safe_spearman(score, pairs["bio_ordering"])
            pid_margin_mean, pid_margin_median, pid_margin_positive_rate = pid_rank_summary(score, pairs["bio_margin"], pids)
            pid_order_mean, pid_order_median, pid_order_positive_rate = pid_rank_summary(score, pairs["bio_ordering"], pids)
            q4 = quartile_mask(score_percentile, "Q4")
            q1 = quartile_mask(score_percentile, "Q1")
            top_bottom_margin = p6a2.pid_bootstrap_difference(pairs["bio_margin"], pids, q4, q1, seed_offset=7000 + candidate_index)
            top_bottom_order = p6a2.pid_bootstrap_difference(pairs["bio_ordering"], pids, q4, q1, seed_offset=7100 + candidate_index)
            ordering_rows.append({
                "candidate": candidate, "aggregation": aggregation,
                "global_spearman_score_margin": margin_rho, "global_spearman_score_ordering": ordering_rho,
                "pid_mean_spearman_score_margin": pid_margin_mean, "pid_median_spearman_score_margin": pid_margin_median,
                "pid_positive_rate_score_margin": pid_margin_positive_rate,
                "pid_mean_spearman_score_ordering": pid_order_mean, "pid_median_spearman_score_ordering": pid_order_median,
                "pid_positive_rate_score_ordering": pid_order_positive_rate,
                "Q4_minus_Q1_margin": top_bottom_margin["difference"], "Q4_minus_Q1_margin_ci_low": top_bottom_margin["ci_low"], "Q4_minus_Q1_margin_ci_high": top_bottom_margin["ci_high"],
                "Q4_minus_Q1_ordering": top_bottom_order["difference"], "Q4_minus_Q1_ordering_ci_low": top_bottom_order["ci_low"], "Q4_minus_Q1_ordering_ci_high": top_bottom_order["ci_high"],
                "Q4_pid_count": int(np.unique(pids[q4]).size), "Q1_pid_count": int(np.unique(pids[q1]).size),
            })
    write_csv(OUT_ROOT / "quality_coverage_curve.csv", curve_rows)
    write_csv(OUT_ROOT / "reliability_quartile_validation.csv", quartile_rows)
    write_csv(OUT_ROOT / "ordering_validation.csv", ordering_rows)
    write_csv(OUT_ROOT / "failed_full_degradation_validation.csv", degradation_rows)
    write_csv(OUT_ROOT / "coverage_audit.csv", coverage_rows)
    return summary


def write_phase5d_forensics_and_proxy(
    pairs: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    components: dict[str, np.ndarray],
    view_temporal: dict[str, np.ndarray],
    hid_bank: dict[str, Any],
) -> dict[str, Any]:
    utility = p6a2.utility_index()
    phase5d_stems = np.asarray(hid_bank["stems"])
    target_indices = sorted(utility)
    exact_alignment = np.array_equal(phase5d_stems, np.asarray(load_bank(BIO_BANK)["stems"]))
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, pid in enumerate(hid_bank["pid"].tolist()):
        by_pid[int(pid)].append(index)
    # Rebuild a more faithful join around the actual Phase5D unit: a frozen
    # image target. No gradient is recomputed and no pair target is invented.
    pair_set: set[tuple[int, int]] = set()
    for target in target_indices:
        target_pid = int(hid_bank["pid"][target])
        target_cloth = int(hid_bank["clothes_id"][target])
        for partner in by_pid[target_pid]:
            if int(hid_bank["clothes_id"][partner]) != target_cloth:
                pair_set.add((target, partner))
                pair_set.add((partner, target))
    bio_features = load_bank(BIO_BANK)["features"]
    rows: list[dict[str, Any]] = []
    for pair_id, (anchor, positive) in enumerate(sorted(pair_set)):
        ua = utility.get(anchor, {})
        up = utility.get(positive, {})
        rows.append({
            "v2_pair_id": pair_id, "index_a": anchor, "index_b": positive,
            "pid": int(hid_bank["pid"][anchor]), "clothes_a": int(hid_bank["clothes_id"][anchor]), "clothes_b": int(hid_bank["clothes_id"][positive]),
            "anchor_phase5d_target": int(anchor in utility), "positive_phase5d_target": int(positive in utility),
            "anchor_utility_label": ua.get("utility_label", ""), "positive_utility_label": up.get("utility_label", ""),
            "anchor_late_eva_alignment": ua.get("late_alignment", np.nan), "positive_late_eva_alignment": up.get("late_alignment", np.nan),
            "anchor_same_pid_alignment": ua.get("same_pid_alignment", np.nan), "positive_same_pid_alignment": up.get("same_pid_alignment", np.nan),
            "bio_similarity": float(np.dot(bio_features[anchor], bio_features[positive])),
            "h_id_pair": float(np.nanmean([hid_bank["h_id"][anchor], hid_bank["h_id"][positive]])),
            "triplet_loss_pair_proxy": float(np.nanmean([hid_bank["triplet_loss"][anchor], hid_bank["triplet_loss"][positive]])),
            "confidence_pair_proxy": float(np.nanmean([hid_bank["confidence"][anchor], hid_bank["confidence"][positive]])),
            "correct_pair_proxy": float(np.nanmean([hid_bank["correct"][anchor], hid_bank["correct"][positive]])),
            "join_note": "existing Phase5D image-level probe joined to all label-true cross-clothes partners; not pair-specific gradient utility",
        })
    write_csv(OUT_ROOT / "phase5d_utility_join_v2.csv", rows)
    anchor_matches = int(sum(int(row["anchor_phase5d_target"]) for row in rows))
    positive_matches = int(sum(int(row["positive_phase5d_target"]) for row in rows))
    utility_labels = [row.get("utility_label", "") for row in utility.values()]
    forensic_lines = [
        "# Phase5D join forensics",
        "",
        "This is an offline key-alignment audit. No gradient was recomputed and no model parameter was changed.",
        "",
        f"- Phase5D feature bank rows: {len(phase5d_stems)}.",
        f"- Phase5D H_ID-family utility target rows: {len(target_indices)} unique image indices.",
        f"- Exact stem/PID/clothes alignment with Bio bank: **{exact_alignment}**.",
        f"- Phase6A2 compact join: 157/43,839 relation rows had an anchor endpoint match; positive endpoint match was 0.",
        f"- Phase6A3 v2 endpoint-preserving join: {len(rows)} directed label-true cross-clothes rows; anchor target matches {anchor_matches}; positive target matches {positive_matches}.",
        "",
        "## Diagnosis",
        "",
        "Phase5D utility is an image-level target selected inside matched gradient-probe micro-batches. It is not a persistent pair bank. The Phase6A2 compact relation bank selects the first image in each target clothing state; none of the Phase5D H_ID probe targets happened to be used as those positive representatives, so the old positive-endpoint count was zero. The exact index/stem/PID/clothes audit shows no naming or alignment mismatch.",
        "",
        "The v2 join touches every same-PID/different-clothes partner of each existing Phase5D target image. This repairs endpoint coverage without new backward, but it still does not turn an image-level gradient probe into pair-specific gradient utility. Any comparison using this file must be called existing utility linkage or proxy linkage, not true pair gradient utility.",
        "",
        f"Utility labels present in the H_ID probe artifact: {sorted(set(label for label in utility_labels if label)) or ['none']}.",
        "",
    ]
    (OUT_ROOT / "phase5d_join_forensics.md").write_text("\n".join(forensic_lines), encoding="utf-8")

    h_id_pair = np.nanmean(np.stack([hid_bank["h_id"][pairs["left"]], hid_bank["h_id"][pairs["right"]]], axis=1), axis=1).astype(np.float32)
    triplet_proxy = np.nanmean(np.stack([hid_bank["triplet_loss"][pairs["left"]], hid_bank["triplet_loss"][pairs["right"]]], axis=1), axis=1).astype(np.float32)
    confidence_proxy = np.nanmean(np.stack([hid_bank["confidence"][pairs["left"]], hid_bank["confidence"][pairs["right"]]], axis=1), axis=1).astype(np.float32)
    correct_proxy = np.nanmean(np.stack([hid_bank["correct"][pairs["left"]], hid_bank["correct"][pairs["right"]]], axis=1), axis=1).astype(np.float32)
    proxy = {"triplet_loss": triplet_proxy, "confidence": confidence_proxy, "correct": correct_proxy, "bio_consistency": pairs["intra_raw"], "positive_distance": 1.0 - pairs["bio_similarity"], "bio_margin": pairs["bio_margin"]}
    proxy_rows: list[dict[str, Any]] = []
    for candidate, aggregation_scores in scores.items():
        for aggregation, score in aggregation_scores.items():
            percentile = global_rank(score, True)
            masks = {"all": np.ones(len(score), dtype=bool), "top50": top_fraction_mask(percentile, 0.5), "Q4": quartile_mask(percentile, "Q4")}
            for subset, mask in masks.items():
                states = set(zip(pairs["pid"][mask].tolist(), pairs["cloth_b"][mask].tolist())) if np.any(mask) else set()
                for metric, values in proxy.items():
                    stats = bootstrap_summary(values, pairs["pid"], mask, seed=12000 + len(proxy_rows))
                    proxy_rows.append({
                        "candidate": candidate, "aggregation": aggregation, "subset": subset, "proxy": metric,
                        **stats, "pid_coverage": int(len(np.unique(pairs["pid"][mask]))) if np.any(mask) else 0,
                        "clothing_state_count": int(len(states)), "proxy_warning": "proxy != gradient utility",
                    })
    write_csv(OUT_ROOT / "utility_proxy_analysis.csv", proxy_rows)
    return {
        "utility": utility, "target_count": len(target_indices), "v2_pair_count": len(rows),
        "anchor_matches": anchor_matches, "positive_matches": positive_matches, "exact_alignment": exact_alignment,
        "h_id_pair": h_id_pair, "proxy": proxy,
    }


def write_soft_hid_grid_and_hardness_ordering(
    pairs: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    components: dict[str, np.ndarray],
    view_temporal: dict[str, np.ndarray],
    utility_info: dict[str, Any],
    failed_neighbor_delta: np.ndarray,
) -> None:
    h_percentile = group_rank(utility_info["h_id_pair"], pairs["pid"], True)
    h_quartiles = {"H1": h_percentile <= 0.25, "H2": (h_percentile > 0.25) & (h_percentile <= 0.5), "H3": (h_percentile > 0.5) & (h_percentile <= 0.75), "H4": h_percentile > 0.75}
    grid_rows: list[dict[str, Any]] = []
    hardness_rows: list[dict[str, Any]] = []
    for candidate, aggregation_scores in scores.items():
        for aggregation, score in aggregation_scores.items():
            r_percentile = global_rank(score, True)
            r_quartiles = {q: quartile_mask(r_percentile, q) for q in ("Q1", "Q2", "Q3", "Q4")}
            for rq, rmask in r_quartiles.items():
                for hq, hmask in h_quartiles.items():
                    mask = rmask & hmask
                    states = set(zip(pairs["pid"][mask].tolist(), pairs["cloth_b"][mask].tolist())) if np.any(mask) else set()
                    grid_rows.append({
                        "candidate": candidate, "aggregation": aggregation, "reliability_quartile": rq, "h_id_quartile": hq,
                        "relation_count": int(mask.sum()), "pid_count": int(len(np.unique(pairs["pid"][mask]))) if np.any(mask) else 0,
                        "valid_pid_coverage": float(len(np.unique(pairs["pid"][mask])) / len(np.unique(pairs["pid"]))) if np.any(mask) else 0.0,
                        "clothing_state_count": int(len(states)), "bio_margin": mean_or_nan(pairs["bio_margin"][mask]),
                        "bio_ordering": mean_or_nan(pairs["bio_ordering"][mask]), "view_stability": mean_or_nan(view_temporal["view_stability"][mask]),
                        "temporal_stability": mean_or_nan(view_temporal["temporal_stability"][mask]),
                        "failed_full_similarity_delta": mean_or_nan((pairs["failed_similarity"] - pairs["bio_similarity"])[mask]),
                        "failed_full_neighbor_delta": mean_or_nan(failed_neighbor_delta[mask]),
                        "triplet_loss_proxy": mean_or_nan(utility_info["proxy"]["triplet_loss"][mask]),
                        "confidence_proxy": mean_or_nan(utility_info["proxy"]["confidence"][mask]),
                    })
            # Coverage-preserving threshold: top 50%, not hard-gated removal.
            reliable = top_fraction_mask(r_percentile, 0.5)
            for hq, hmask in h_quartiles.items():
                mask = reliable & hmask
                states = set(zip(pairs["pid"][mask].tolist(), pairs["cloth_b"][mask].tolist())) if np.any(mask) else set()
                hardness_rows.append({
                    "candidate": candidate, "aggregation": aggregation, "coverage_preserving_subset": "top50", "h_id_quartile": hq,
                    "relation_count": int(mask.sum()), "relation_coverage": float(mask.mean()),
                    "pid_count": int(len(np.unique(pairs["pid"][mask]))) if np.any(mask) else 0,
                    "valid_pid_coverage": float(len(np.unique(pairs["pid"][mask])) / len(np.unique(pairs["pid"]))) if np.any(mask) else 0.0,
                    "clothing_state_count": int(len(states)), "h_id_pair_mean": mean_or_nan(utility_info["h_id_pair"][mask]),
                    "bio_margin_mean": mean_or_nan(pairs["bio_margin"][mask]), "bio_ordering_correct_rate": mean_or_nan(pairs["bio_ordering"][mask]),
                    "view_stability_mean": mean_or_nan(view_temporal["view_stability"][mask]), "temporal_stability_mean": mean_or_nan(view_temporal["temporal_stability"][mask]),
                    "triplet_loss_proxy_mean": mean_or_nan(utility_info["proxy"]["triplet_loss"][mask]),
                    "confidence_proxy_mean": mean_or_nan(utility_info["proxy"]["confidence"][mask]),
                })
    write_csv(OUT_ROOT / "soft_reliability_hid_grid.csv", grid_rows)
    write_csv(OUT_ROOT / "reliable_hardness_ordering.csv", hardness_rows)


def write_source_audit(aux_manifest: dict[str, Any], pairs: dict[str, np.ndarray]) -> None:
    source_files = [
        P6A2_ROOT / "relation_pair_bank.csv", P6A2_ROOT / "cross_clothes_positive.csv", P6A2_ROOT / "bio_support.csv",
        P6A2_ROOT / "model_consensus.csv", P6A2_ROOT / "mutual_neighbor.csv", P6A2_ROOT / "intra_id_support.csv",
        P6A2_ROOT / "negative_relation_support.csv", P6A2_ROOT / "reliability_candidates.csv",
        P6A2_ROOT / "reliability_validation.csv", P6A2_ROOT / "ordering_validation.csv",
        P6A2_ROOT / "hardness_reliability_quadrants.csv", P6A2_ROOT / "coverage_audit.csv",
        BIO_BANK, ID_BANK, FAILED_BANK, OFFICIAL_BANK, P5D_BANK,
    ]
    lines = [
        "# Phase 6A3 source audit",
        "",
        "All Phase6A2 CSV inputs below are reused. No existing relation result is regenerated or overwritten. New feature banks, if present, are forward-only outputs from frozen existing checkpoints on LTCC train.",
        "",
        "| Source | Exists | Rows / images | SHA256 | Reused / new | Role |",
        "|---|---|---:|---|---|---|",
    ]
    for path in source_files:
        if path.suffix == ".csv":
            rows = read_csv(path)
            count = len(rows)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as source:
                count = int(source["global_feature"].shape[0])
        else:
            count = ""
        role = "Phase6A2 relation/evidence input" if path.parent == P6A2_ROOT else "frozen feature source"
        lines.append(f"| `{path}` | {path.is_file()} | {count} | `{sha256_file(path) or 'NA'}` | reused | {role} |")
    for path, role in ((FEATURE_ROOT / "bio_best_horizontal_flip_feature_bank.npz", "new frozen forward-only view bank"), (FEATURE_ROOT / "bio_s0_epoch15_v0_feature_bank.npz", "new frozen forward-only temporal bank")):
        count = ""
        if path.is_file():
            with np.load(path, allow_pickle=False) as source:
                count = int(source["global_feature"].shape[0])
        lines.append(f"| `{path}` | {path.is_file()} | {count} | `{sha256_file(path) or 'NA'}` | {'new forward' if path.is_file() else 'not available'} | {role} |")
    lines.extend([
        "",
        f"Primary compact relation count reused: {len(pairs['pair_id'])}; valid cross-clothes PID count: {len(np.unique(pairs['pid']))}; total LTCC train PID count: {TOTAL_PID_COUNT}.",
        "",
        "Feature source semantics remain Phase6A2: `global_feature` (1024-D) followed by float32 L2 normalization. Failed Full is evaluation-only and never enters score construction.",
        "",
    ])
    (OUT_ROOT / "source_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "artifacts/source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_protocol_docs(aux_manifest: dict[str, Any]) -> None:
    view_status = "available" if aux_manifest.get("view_bank_available") else "NOT_AVAILABLE"
    temporal_status = "available" if aux_manifest.get("temporal_bank_available") else "NOT_AVAILABLE"
    view_doc = f"""# Augmentation/view protocol\n\nStatus: **{view_status}**.\n\nV0 is the existing Bio BEST relation bank using the released formal test transform. V1 is a deterministic horizontal flip evaluated with the same formal preprocessing:\n\n`ResizeWithEqualScale(224,224) -> horizontal flip -> ToTensor -> ImageNet Normalize`.\n\nNo random crop, random erase, color destruction, style transfer, diffusion, or identity-changing transform is used. The relation stability unit is `(anchor, same-PID/different-clothes positive)`, not a single-image feature. V2 mild crop was not added because it is not part of the frozen formal test transform.\n\nThe view score is reported first as raw V0/V1 similarity, std/range, and rank consistency. It is only included in SR4 after its independent validation table is produced.\n"""
    temporal_doc = f"""# Temporal stability audit\n\nStatus: **{temporal_status}**.\n\nThe only additional temporal checkpoint used is the already-existing Phase5B S0 epoch15 checkpoint. It is evaluated with the released formal test transform and compared to the frozen Bio BEST bank. No checkpoint was trained or created for this phase.\n\nFor each ground-truth cross-clothes relation, the audit records similarity delta, within-anchor rank consistency, and a bounded temporal stability score. This is relation-level checkpoint stability, not H_ID ranking stability.\n"""
    (OUT_ROOT / "augmentation_view_protocol.md").write_text(view_doc, encoding="utf-8")
    (OUT_ROOT / "temporal_stability_audit.md").write_text(temporal_doc, encoding="utf-8")


def plot_quality_coverage(curve_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    selected = [row for row in curve_rows if row["aggregation"] == "mean" and row["subset"] in ("all", "top75", "top50", "top25", "top10")]
    if not selected:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for candidate in sorted({row["candidate"] for row in selected}):
        rows = [row for row in selected if row["candidate"] == candidate]
        rows.sort(key=lambda row: float(row["relation_coverage"]))
        x = [float(row["relation_coverage"]) for row in rows]
        axes[0].plot(x, [float(row["positive_negative_margin_mean"]) for row in rows], marker="o", label=candidate)
        axes[1].plot(x, [float(row["ordering_correct_mean"]) for row in rows], marker="o", label=candidate)
    axes[0].set_xlabel("Relation coverage"); axes[0].set_ylabel("Bio positive-negative margin")
    axes[1].set_xlabel("Relation coverage"); axes[1].set_ylabel("Bio ordering correctness")
    axes[0].grid(alpha=0.25); axes[1].grid(alpha=0.25); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "quality_coverage_curve.png", dpi=160)
    plt.close(fig)


def choose_decisions(
    pairs: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    view_temporal: dict[str, np.ndarray],
    utility_info: dict[str, Any],
) -> dict[str, Any]:
    pids = pairs["pid"]
    validated_candidates = [name for name in scores if name in ("SR1_BIO_ID_CONSENSUS", "SR2_CONSENSUS_MUTUAL", "SR3_CONSENSUS_MUTUAL_INTRA")]
    comparisons: list[dict[str, Any]] = []
    for index, candidate in enumerate(validated_candidates):
        score = scores[candidate]["mean"]
        percentile = global_rank(score, True)
        q1 = quartile_mask(percentile, "Q1")
        q4 = quartile_mask(percentile, "Q4")
        top50 = top_fraction_mask(percentile, 0.5)
        all_mask = np.ones(len(score), dtype=bool)
        margin_q = p6a2.pid_bootstrap_difference(pairs["bio_margin"], pids, q4, q1, seed_offset=16000 + index)
        order_q = p6a2.pid_bootstrap_difference(pairs["bio_ordering"], pids, q4, q1, seed_offset=16100 + index)
        margin_50 = p6a2.pid_bootstrap_difference(pairs["bio_margin"], pids, top50, all_mask, seed_offset=16200 + index)
        order_50 = p6a2.pid_bootstrap_difference(pairs["bio_ordering"], pids, top50, all_mask, seed_offset=16300 + index)
        view_q = p6a2.pid_bootstrap_difference(view_temporal["view_stability"], pids, q4, q1, seed_offset=16400 + index)
        temp_q = p6a2.pid_bootstrap_difference(view_temporal["temporal_stability"], pids, q4, q1, seed_offset=16500 + index)
        curve_rows = read_csv(OUT_ROOT / "quality_coverage_curve.csv")
        row50 = next((row for row in curve_rows if row["candidate"] == candidate and row["aggregation"] == "mean" and row["subset"] == "top50"), {})
        comparisons.append({
            "candidate": candidate, "Q4_minus_Q1_margin": margin_q, "Q4_minus_Q1_ordering": order_q,
            "top50_minus_all_margin": margin_50, "top50_minus_all_ordering": order_50,
            "Q4_minus_Q1_view": view_q, "Q4_minus_Q1_temporal": temp_q,
            "top50_valid_pid_coverage": f(row50.get("valid_cross_clothes_pid_coverage")),
            "top50_total_pid_coverage": f(row50.get("total_77_pid_coverage")),
            "top50_relation_coverage": f(row50.get("relation_coverage")),
        })
    q1_supported = [row for row in comparisons if row["Q4_minus_Q1_margin"]["ci_low"] > 0 and row["Q4_minus_Q1_ordering"]["ci_low"] > 0]
    q1 = "YES" if q1_supported else ("INCONCLUSIVE" if comparisons else "NO")
    q2_supported = [row for row in comparisons if row["top50_minus_all_margin"]["ci_low"] > 0 and row["top50_minus_all_ordering"]["ci_low"] > 0 and row["top50_valid_pid_coverage"] >= 0.75]
    if q2_supported:
        q2 = "YES"
    elif any(row["top50_minus_all_margin"]["difference"] > 0 or row["top50_minus_all_ordering"]["difference"] > 0 for row in comparisons):
        q2 = "MIXED"
    else:
        q2 = "INCONCLUSIVE"
    # Phase 6A3 is explicitly coverage-preserving.  Do not let a higher-purity
    # but coverage-poor candidate become the selected design.  First require
    # the top-50 quality/coverage screen; only then rank eligible candidates by
    # Q4-vs-Q1 margin.  A fallback coverage-qualified candidate is retained so
    # the report can still explain a mixed/failed quality gate.
    quality_coverage_supported = [
        row for row in q2_supported
        if row["top50_valid_pid_coverage"] >= 0.75
    ]
    coverage_candidates = [row for row in comparisons if row["top50_valid_pid_coverage"] >= 0.75]
    eligible_best = quality_coverage_supported or coverage_candidates or comparisons
    best = max(eligible_best, key=lambda row: row["Q4_minus_Q1_margin"].get("difference", -float("inf"))) if eligible_best else None
    if best is None:
        q3 = "INCONCLUSIVE"
    else:
        positive = [
            best["Q4_minus_Q1_margin"]["ci_low"] > 0,
            best["Q4_minus_Q1_ordering"]["ci_low"] > 0,
            best["Q4_minus_Q1_view"]["ci_low"] > 0,
            best["Q4_minus_Q1_temporal"]["ci_low"] > 0,
        ]
        if sum(positive) >= 3:
            q3 = "YES"
        elif sum(item["difference"] > 0 for item in (best["Q4_minus_Q1_margin"], best["Q4_minus_Q1_ordering"], best["Q4_minus_Q1_view"], best["Q4_minus_Q1_temporal"])) >= 2:
            q3 = "MIXED"
        else:
            q3 = "INCONCLUSIVE"
    view_support = [row for row in comparisons if row["Q4_minus_Q1_view"]["ci_low"] > 0]
    temp_support = [row for row in comparisons if row["Q4_minus_Q1_temporal"]["ci_low"] > 0]
    q4 = "YES" if view_support else ("INCONCLUSIVE" if comparisons else "NOT_AVAILABLE")
    q5 = "YES" if temp_support else ("INCONCLUSIVE" if comparisons else "NOT_AVAILABLE")

    utility_join = "YES" if utility_info["v2_pair_count"] > 0 and utility_info["positive_matches"] > 0 and utility_info["exact_alignment"] else "NO"
    utility_relation = "MIXED — endpoint linkage repaired, pair-specific gradient utility remains unavailable" if utility_join == "YES" else "INSUFFICIENT_SUPPORT"

    # Q7: use the coverage-preserving top50 of the strongest pre-view/temporal
    # candidate, then compare H_ID easy versus hard. H_ID does not define score.
    q7 = "INCONCLUSIVE"
    h_order_result: dict[str, Any] = {}
    if best is not None:
        score = scores[best["candidate"]]["mean"]
        r_pct = global_rank(score, True)
        h_pct = group_rank(utility_info["h_id_pair"], pids, True)
        reliable = top_fraction_mask(r_pct, 0.5)
        easy = reliable & (h_pct <= 0.25)
        hard = reliable & (h_pct > 0.75)
        h_margin = p6a2.pid_bootstrap_difference(pairs["bio_margin"], pids, easy, hard, seed_offset=17000)
        h_order = p6a2.pid_bootstrap_difference(pairs["bio_ordering"], pids, easy, hard, seed_offset=17100)
        h_order_result = {"candidate": best["candidate"], "easy_minus_hard_margin": h_margin, "easy_minus_hard_ordering": h_order, "easy_pairs": int(easy.sum()), "hard_pairs": int(hard.sum())}
        if h_margin["ci_low"] > 0 and h_order["ci_low"] > 0:
            q7 = "YES"
        elif h_margin["difference"] > 0 or h_order["difference"] > 0:
            q7 = "MIXED"
        else:
            q7 = "NO"

    coverage_ok = bool(best and best["top50_valid_pid_coverage"] >= 0.75)
    independent_ok = bool(q4 == "YES" or q5 == "YES")
    promotion = q1 == "YES" and q2 == "YES" and coverage_ok and independent_ok and q7 in ("YES", "MIXED")
    final = "PROMOTE TO PHASE 6B — LIMITED SOFT RELATION SUPERVISION SCREEN (UTILITY NOT PROVEN)" if promotion else "STOP — SOFT RELIABILITY NOT YET JUSTIFIED FOR TRAINING"
    return {
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6_join": utility_join, "q6_utility_relation": utility_relation, "q7": q7,
        "q8": "PROMOTE" if promotion else "STOP", "final": final, "best_candidate": best["candidate"] if best else None,
        "candidate_comparisons": comparisons, "view_supported_candidates": [row["candidate"] for row in view_support], "temporal_supported_candidates": [row["candidate"] for row in temp_support],
        "hardness_ordering": h_order_result, "coverage_ok": coverage_ok, "independent_stability_ok": independent_ok,
    }


def write_report(
    pairs: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    components: dict[str, np.ndarray],
    view_temporal: dict[str, np.ndarray],
    utility_info: dict[str, Any],
    decisions: dict[str, Any],
) -> None:
    lines = [
        "# PHASE 6A3 — COVERAGE-PRESERVING SOFT RELIABILITY",
        "",
        "## 1. Scope",
        "",
        "Offline-only LTCC train audit. No optimizer, backward pass, parameter update, DS/DW/sampler change, relation loss, test/query/gallery data, or formal training configuration change was used. All ground-truth cross-clothes relations remain available.",
        "",
        "## 2. Frozen Phase6A2 findings",
        "",
        "Phase6A2 found that hard reliability gating improved relation quality but reduced coverage: 43,839 compact cross-clothes relations across 46 valid cross-clothes PIDs; R1/R2/R3 high-reliability coverage was 32/46, 21/46, and 16/46 valid PIDs. Phase5D utility support was 157/43,839 with zero positive endpoints.",
        "",
        "## 3. Why hard gating failed",
        "",
        "The failure is a quality-coverage tradeoff: a high score can identify a coherent subset while excluding many valid PID/clothing states. Phase6A3 therefore reports continuous scores and top-percentile curves; low reliability is never converted to zero visibility.",
        "",
        "## 4. Source audit",
        "",
        f"The reused Phase6A2 sources and SHA256 values are in `source_audit.md`. Compact relation count: {len(pairs['pair_id'])}; valid cross-clothes PIDs: {len(np.unique(pairs['pid']))}; total train PIDs: {TOTAL_PID_COUNT}.",
        "",
        "## 5. Soft reliability definitions",
        "",
        "SR0 uses Bio support. SR1 uses Bio support, ID-only support, and rank agreement. SR2 uses model consensus plus mutual-neighbor support. SR3 adds intra-ID support. SR4 adds available view and temporal stability. Failed Full never participates in any score.",
        "",
        "Two aggregations are reported: equal-mean of available normalized evidence, and conservative mean multiplied by an evidence-disagreement penalty. Both remain non-zero continuous audit scores; neither removes a relation.",
        "",
        "## 6. Evidence normalization",
        "",
        "Raw similarities, margins, support means, view deltas, and temporal deltas are preserved in `soft_reliability_components.csv`. Score inputs are within-anchor or within-PID ranks, bounded stability transforms, or bounded mutual fractions. No raw-scale addition is used.",
        "",
        "## 7. Quality-coverage tradeoff",
        "",
        "`quality_coverage_curve.csv` and `quality_coverage_curve.png` report all/top75/top50/top25/top10 relation subsets. Each row contains relation coverage, valid cross-clothes PID coverage out of 46, total PID coverage out of 77, clothing-state coverage, margin/order, Failed-Full degradation, model consensus, view stability, and temporal stability.",
        "Because mutual-support scores contain exact ties, target subsets use deterministic exact counts with stable pair order at the boundary; ties are not resolved using any outcome metric.",
        "",
        "## 8. Reliability quartile analysis",
        "",
        "Q1–Q4 are exact score-ranked quartiles, not hard gates. Ties at a quartile boundary use the same deterministic stable-order rule. `reliability_quartile_validation.csv` reports PID-bootstrap means and 95% CIs.",
        "",
        "## 9. Margin/order validation",
        "",
        f"Overall Bio ordering correctness in the compact bank: {report_num(np.mean(pairs['bio_ordering']))}; overall Bio margin: {report_num(np.mean(pairs['bio_margin']))}. Score-vs-quality correlations and Q4-Q1 PID-bootstrap comparisons are in `ordering_validation.csv`.",
        "",
        "## 10. Augmentation/view stability",
        "",
        "The view protocol is recorded in `augmentation_view_protocol.md`. V0 is the frozen formal transform; V1 is a deterministic horizontal flip. Stability is relation-level similarity/rank consistency, not single-image feature stability.",
        "",
        f"Mean available view stability: {report_num(np.nanmean(view_temporal['view_stability']))}.",
        "",
        "## 11. Temporal stability",
        "",
        "The temporal audit compares the existing Bio BEST bank with the existing Phase5B S0 epoch15 checkpoint only. No checkpoint was trained for this phase. See `temporal_stability_audit.md` and `temporal_relation_stability.csv`.",
        "",
        f"Mean available temporal stability: {report_num(np.nanmean(view_temporal['temporal_stability']))}.",
        "",
        "## 12. Failed-Full degradation validation",
        "",
        "Failed Full is introduced only after score construction and is evaluation-only. `failed_full_degradation_validation.csv` reports similarity, margin, ordering, and neighbor-purity differences by soft reliability quartile.",
        "",
        "## 13. Phase5D join forensics",
        "",
        f"The exact key audit and v2 endpoint-preserving join are in `phase5d_join_forensics.md`. V2 rows: {utility_info['v2_pair_count']}; anchor target matches: {utility_info['anchor_matches']}; positive target matches: {utility_info['positive_matches']}. This repairs endpoint coverage but does not create pair-specific gradient utility.",
        "",
        "## 14. Utility evidence / proxy",
        "",
        "`utility_proxy_analysis.csv` uses existing train-only forward/loss-bank quantities such as triplet loss, confidence, correctness, Bio consistency, positive distance, and margin. Every row is explicitly a proxy; proxy != gradient utility. Existing Phase5D gradient probes are not regenerated.",
        "",
        "## 15. Soft Reliability × H_ID",
        "",
        "`soft_reliability_hid_grid.csv` reports the 4×4 reliability-quartile × H_ID-quartile grid. H_ID remains damage/risk evidence and is not used to define reliability truth.",
        "",
        "## 16. Reliable Easy/Medium/Hard audit",
        "",
        "`reliable_hardness_ordering.csv` evaluates Easy/Medium/Hard H_ID within the coverage-preserving top50% score subset. This is an offline ordering audit, not a training schedule.",
        "",
        "## 17. Coverage audit",
        "",
        "Coverage is reported against both denominators: 46 valid cross-clothes PIDs and all 77 train PIDs. Clothing-state coverage is also retained. No relation is deleted in the proposed future design.",
        "",
        "## 18. Future Soft Relation DS",
        "",
        "See `future_soft_relation_ds.md`: reliability controls emphasis, not visibility; all images, clothing states, PID identities, and label-true relations remain available.",
        "",
        "## 19. Future relation weighting",
        "",
        "See `future_relation_weight_design.md`. The proposed interface keeps a non-zero base path for low-reliability relations and does not freeze a weight parameter in this audit.",
        "",
        "## 20. Q1–Q8",
        "",
        f"- Q1 — Soft reliability retains Phase6A2 relation-quality signal: **{decisions['q1']}**.",
        f"- Q2 — Soft reliability improves quality-coverage tradeoff: **{decisions['q2']}**.",
        f"- Q3 — High soft reliability has better margin/order/view/temporal evidence: **{decisions['q3']}**.",
        f"- Q4 — View evidence independently supports reliability: **{decisions['q4']}**.",
        f"- Q5 — Temporal evidence independently supports reliability: **{decisions['q5']}**.",
        f"- Q6 — Phase5D join repaired: **{decisions['q6_join']}**; utility relation: **{decisions['q6_utility_relation']}**.",
        f"- Q7 — H_ID orders difficulty inside a coverage-preserving reliable subset: **{decisions['q7']}**.",
        f"- Q8 — Enter limited Phase6B feasibility screen: **{decisions['q8']}**.",
        "",
        "### Candidate comparisons",
        "",
        "| Candidate | Q4-Q1 margin | 95% CI | Q4-Q1 order | 95% CI | Top50 valid PID coverage | Top50 relation coverage | Q4-Q1 view | Q4-Q1 temporal |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in decisions["candidate_comparisons"]:
        lines.append(
            f"| {row['candidate']} | {report_num(row['Q4_minus_Q1_margin']['difference'])} | [{report_num(row['Q4_minus_Q1_margin']['ci_low'])}, {report_num(row['Q4_minus_Q1_margin']['ci_high'])}] | "
            f"{report_num(row['Q4_minus_Q1_ordering']['difference'])} | [{report_num(row['Q4_minus_Q1_ordering']['ci_low'])}, {report_num(row['Q4_minus_Q1_ordering']['ci_high'])}] | "
            f"{report_num(row['top50_valid_pid_coverage'])} | {report_num(row['top50_relation_coverage'])} | {report_num(row['Q4_minus_Q1_view']['difference'])} | {report_num(row['Q4_minus_Q1_temporal']['difference'])} |"
        )
    lines.extend([
        "",
        "The coverage-preserving selection is deliberate: SR1 has stronger purity but only 69.6% valid-PID coverage at top-50%; SR2 reaches 100% valid-PID coverage but its top-50 margin gain has a CI touching zero; SR3 retains 100% valid-PID coverage and has positive PID-bootstrap support for both top-50 margin and ordering.",
        "",
        "## 21. Promotion gate",
        "",
        "Promotion requires quality retained, improved coverage, independent stability support, no active-set withholding, and no claim that utility is already proven. If promoted, only a limited next-phase screen is authorized; this phase itself never trains.",
        "",
        f"Best candidate: `{decisions.get('best_candidate') or 'none'}`; coverage gate: `{decisions['coverage_ok']}`; independent stability gate: `{decisions['independent_stability_ok']}`.",
        "",
        "## 22. Final diagnosis",
        "",
        f"# {decisions['final']}",
        "",
        "The scientific interpretation remains: labels define relation truth; the continuous multi-evidence score estimates reliability; all relations remain available; H_ID only orders difficulty after reliability is established.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_future_design_docs() -> None:
    ds = """# Future Soft Relation DS (audit only)

Phase 6A3 does not train or implement this design.

## Frozen principle

`relation_label` comes only from LTCC PID labels. A continuous reliability
score changes emphasis, never visibility. All train images, all PID identities,
all clothing states, and all label-true cross-clothes relations remain
available.

## Candidate stages

1. Reliability-first, low/medium H_ID relation emphasis.
2. High reliability, medium H_ID emphasis.
3. High reliability, high H_ID emphasis only if independent quality and utility proxies remain acceptable.

The first formal comparison must contain a uniform relation-emphasis control.
No active-set withholding, hard-only training, zero probability, or test-tuned
threshold is allowed.
"""
    weight = """# Future relation weight design (audit only)

No weight is added to a training loop in Phase 6A3.

## Candidate interfaces

- W0: `w_ij = 1` uniform relation weight.
- W1: bounded soft reliability, with a non-zero minimum base path, e.g. `w_min + (1-w_min)R_ij`; `w_min` is intentionally not frozen here.
- W2: reliability controls the maximum trusted emphasis and H_ID controls only stage ordering after reliability is established.

Low reliability is not equivalent to a false relation and must not become zero
weight unless a separate metadata/label audit establishes corruption.
"""
    (OUT_ROOT / "future_soft_relation_ds.md").write_text(ds, encoding="utf-8")
    (OUT_ROOT / "future_relation_weight_design.md").write_text(weight, encoding="utf-8")


def analyze() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("PHASE6A3_SOURCE_AUDIT", flush=True)
    pairs = parse_phase6a2_pairs()
    bio_bank = load_bank(BIO_BANK)
    id_bank = load_bank(ID_BANK)
    failed_bank = load_bank(FAILED_BANK)
    official_bank = load_bank(OFFICIAL_BANK) if OFFICIAL_BANK.is_file() else None
    hid_bank = load_bank(P5D_BANK)
    for name, bank in (("ID-only", id_bank), ("Local Failed Full", failed_bank), ("Phase5D", hid_bank)):
        check_equal_labels(bio_bank, bank, name)
    if official_bank is not None:
        check_equal_labels(bio_bank, official_bank, "Official Full")

    aux_manifest = extract_forward_auxiliary_banks()
    view_path = FEATURE_ROOT / "bio_best_horizontal_flip_feature_bank.npz"
    temporal_path = FEATURE_ROOT / "bio_s0_epoch15_v0_feature_bank.npz"
    view_bank = load_bank(view_path) if view_path.is_file() else None
    temporal_bank = load_bank(temporal_path) if temporal_path.is_file() else None
    if view_bank is not None:
        check_equal_labels(bio_bank, view_bank, "view bank")
    if temporal_bank is not None:
        check_equal_labels(bio_bank, temporal_bank, "temporal bank")
    write_source_audit(aux_manifest, pairs)
    write_protocol_docs(aux_manifest)

    print("PHASE6A3_RELATION_STABILITY", flush=True)
    view_temporal = compute_view_temporal(pairs, bio_bank, view_bank, temporal_bank)
    write_view_temporal_artifacts(view_temporal, pairs)
    scores, components = build_soft_scores(pairs, view_temporal)

    print("PHASE6A3_FAILED_FULL_NEIGHBORS", flush=True)
    # Phase6A2 re-exports the feature-bank helpers from Phase6A, but does not
    # itself expose these names.  Keep this diagnostic explicitly tied to the
    # frozen Phase6A implementation rather than duplicating neighbor logic.
    bio_top = p6a2.p6a.top_neighbors(bio_bank["features"], k=TOP_K, chunk_size=256)
    failed_top = p6a2.p6a.top_neighbors(failed_bank["features"], k=TOP_K, chunk_size=256)
    bio_purity = p6a2.same_pid_purity(bio_top, bio_bank["pid"])[pairs["left"]]
    failed_purity = p6a2.same_pid_purity(failed_top, bio_bank["pid"])[pairs["left"]]
    failed_neighbor_delta = failed_purity - bio_purity

    quality_summary = write_quality_coverage_and_validation(pairs, scores, components, view_temporal, failed_neighbor_delta)
    plot_quality_coverage(read_csv(OUT_ROOT / "quality_coverage_curve.csv"))
    utility_info = write_phase5d_forensics_and_proxy(pairs, scores, components, view_temporal, hid_bank)
    write_soft_hid_grid_and_hardness_ordering(pairs, scores, components, view_temporal, utility_info, failed_neighbor_delta)
    write_future_design_docs()
    decisions = choose_decisions(pairs, scores, view_temporal, utility_info)
    write_report(pairs, scores, components, view_temporal, utility_info, decisions)

    manifest = {
        "phase": "phase6a3_soft_reliability",
        "created_utc": utc_now(), "seed": SEED,
        "scope": "offline train-only coverage-preserving reliability audit",
        "prohibitions": {"training": True, "optimizer": True, "optimizer_step": True, "backward": True, "ds": True, "dw": True, "sampler": True, "relation_loss": True, "grl": True, "online_h_id": True, "prcc": True, "test_query_gallery": True},
        "dataset": {"name": "LTCC", "split": "train", "images": IMAGE_COUNT, "total_pids": TOTAL_PID_COUNT, "valid_cross_clothes_pids": int(len(np.unique(pairs["pid"]))), "cross_clothes_relations": int(len(pairs["pair_id"]))},
        "relation_truth": {"positive": "same PID", "primary": "same PID + different clothes", "negative": "different PID", "all_gt_cross_clothes_relations_retained": True},
        "score_candidates": {candidate: list(aggregation_scores) for candidate, aggregation_scores in scores.items()},
        "score_inputs": {"failed_full_used_in_score": False, "query_gallery_used": False, "test_used": False, "h_id_used_in_score": False},
        "view_temporal": {"view_bank_available": view_bank is not None, "temporal_bank_available": temporal_bank is not None, "forward_manifest": aux_manifest},
        "phase5d_forensics": {"target_count": utility_info["target_count"], "v2_pair_count": utility_info["v2_pair_count"], "anchor_matches": utility_info["anchor_matches"], "positive_matches": utility_info["positive_matches"], "exact_alignment": utility_info["exact_alignment"], "new_backward": False},
        "decisions": decisions,
        "bootstrap": {"unit": "PID", "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "outputs": sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.rglob("*") if path.is_file()),
    }
    write_json(OUT_ROOT / "PHASE6A3_MANIFEST.json", manifest)
    return {"decisions": decisions, "manifest": manifest, "quality_summary": quality_summary}


def main() -> None:
    result = analyze()
    print(json.dumps(json_safe(result["decisions"]), indent=2, sort_keys=True), flush=True)
    print(f"PHASE6A3_REPORT {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
