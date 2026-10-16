#!/usr/local/bin/python3
"""PHASE SEMANTIC-RELIABILITY-AUDIT 的只读 LTCC 审计脚本。

脚本不构造模型、不建立优化器、不执行反向传播，也不读取 LTCC query/gallery。
它只检查：
1. 正式 Bio+GPC 的关系密度与损失归约；
2. LTCC train-only 语义 archive、PID Summary Bio 和索引完整性；
3. 逐 PID 的文本字段/因子 embedding 质量与既有 Bio 局部指标的观察性关系。

所有输出均写入本阶段独立目录，方便复现和审计。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/DIFFER")
TRAIN_ROOT = Path("/home/data/LTCC_ReID/train")
TEXT_ROOT = Path("/home/data/DIFFER_TextCaption/LTCC_ReID")
TEXT_NPZ = TEXT_ROOT / "EVA02-CLIP-B-16/train.npz"
SUMMARY_JSON = TEXT_ROOT / "train_caption_summary_biometric.json"
HP_ROOT = ROOT / "outputs/cc_gpc_hp_pilot/20261012_150038"
MARGIN_ROOT = ROOT / "outputs/ndp_margin_diagnostic/20261008_154849"
HP_PROTOCOL = HP_ROOT / "protocol.json"
ANCHOR_METRICS = HP_ROOT / "per_anchor_local_metrics.csv"
MARGIN_ANCHOR_METRICS = MARGIN_ROOT / "per_anchor_margin.csv"

FEATURES = {
    "seed1234_baseline": MARGIN_ROOT / "features/seed1234_bio_train_features.npz",
    "seed1234_gpc": MARGIN_ROOT / "features/seed1234_gpc_train_features.npz",
    "seed4321_baseline": MARGIN_ROOT / "features/seed4321_bio_train_features.npz",
    "seed4321_gpc": MARGIN_ROOT / "features/seed4321_gpc_train_features.npz",
}

FACTOR_NAMES = ["Bio原始槽", "Hair", "Clothing", "Pose", "Interaction", "Environment"]
STEM_RE = re.compile(r"^(?P<pid>\d+)_(?P<track>\d+)_c(?P<camera>\d+)_(?P<frame>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def quantile_or_none(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def parse_records() -> tuple[list[dict[str, Any]], dict[int, int]]:
    paths = sorted(TRAIN_ROOT.glob("*.png"))
    records: list[dict[str, Any]] = []
    raw_pids: set[int] = set()
    for path in paths:
        match = STEM_RE.fullmatch(path.stem)
        if match is None:
            raise RuntimeError(f"无法解析 LTCC train 文件名：{path.name}")
        raw_pid = int(match.group("pid"))
        track = int(match.group("track"))
        camera = int(match.group("camera"))
        raw_pids.add(raw_pid)
        records.append({
            "index": len(records),
            "stem": path.stem,
            "image_path": str(path),
            "raw_pid": raw_pid,
            "track": track,
            "camera": camera,
            "cloth_key": f"{raw_pid:03d}_{track}",
        })
    pid_map = {raw: label for label, raw in enumerate(sorted(raw_pids))}
    for item in records:
        item["pid"] = pid_map[item["raw_pid"]]
    return records, pid_map


def load_summary() -> dict[str, Any]:
    with SUMMARY_JSON.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise RuntimeError("Summary Bio JSON 不是对象")
    return summary


def load_text_archive(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    archive = np.load(TEXT_NPZ, allow_pickle=True)
    if set(archive.files) != {"data", "metadata"}:
        raise RuntimeError(f"语义 NPZ 字段异常：{archive.files}")
    data = np.asarray(archive["data"])
    metadata = [str(x) for x in archive["metadata"].tolist()]
    if data.ndim != 2 or data.shape[1] != 512 or data.shape[0] != len(metadata) * 6:
        raise RuntimeError(f"语义 NPZ 形状异常：{data.shape}，metadata={len(metadata)}")
    if len(metadata) != len(set(metadata)):
        raise RuntimeError("语义 metadata 存在重复 key")
    stems = [item["stem"] for item in records]
    index = {stem: i for i, stem in enumerate(metadata)}
    missing = sorted(set(stems) - set(index))
    extra = sorted(set(metadata) - set(stems))
    if missing or extra:
        raise RuntimeError(f"语义 metadata 与训练图像不一致：missing={len(missing)} extra={len(extra)}")
    order = np.asarray([index[stem] for stem in stems], dtype=np.int64)
    blocks = data.reshape(len(metadata), 6, 512)[order].astype(np.float32)
    return {"data": data, "metadata": metadata, "order": order, "blocks": blocks}


def text_flags(text: str) -> dict[str, int]:
    value = text.lower()
    def has(*terms: str) -> int:
        return int(any(term in value for term in terms))
    def has_word(*terms: str) -> int:
        return int(any(re.search(r"\b" + re.escape(term) + r"\b", value) for term in terms))
    return {
        "has_male": has_word("male", "man", "boy"),
        "has_female": has_word("female", "woman", "girl"),
        "has_long_hair": has("long hair", "medium to long", "medium-length to long"),
        "has_short_hair": has("short hair"),
        "has_dark_hair": has("dark hair", "black hair", "brown hair"),
        "has_tied_hair": has("tied back"),
        "has_glasses": has("glasses", "spectacles"),
        "has_slender": has("slender"),
        "has_average_build": has("average build", "average height and build", "average height and generally"),
        "has_medium_build": has("medium build", "medium to slender", "medium to average"),
        "has_robust_build": has("robust build", "robust"),
        "has_clothing_term": has("t-shirt", "shirt", "shorts", "pants", "jacket", "coat", "dress", "skirt", "clothing"),
        "has_observation_uncertainty": has("possibly", "likely", "appears", "estimated", "seems", "based on the surroundings", "difficult to determine"),
        "has_explicit_variation": has("varies", "ranges", "ranging", "some descriptions", "most descriptions", "frequently", "occasionally"),
        "has_blur_or_quality_warning": has("blurry", "image quality", "ambiguity", "ambiguous", "contradictions", "inconsistent"),
    }


def semantic_quality(records: list[dict[str, Any]], summary: dict[str, Any], blocks: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts = []
    per_pid: list[dict[str, Any]] = []
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_pid[int(item["pid"])].append(item)
    missing = []
    field_keys = Counter()
    for pid in range(len(by_pid)):
        record = summary.get(str(pid))
        if not isinstance(record, dict):
            missing.append(pid)
            continue
        field_keys.update(record.keys())
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            missing.append(pid)
            text = ""
        text = text.strip()
        texts.append(text)
        flags = text_flags(text)
        cloths = sorted({item["cloth_key"] for item in by_pid[pid]})
        cams = sorted({item["camera"] for item in by_pid[pid]})
        hair_mixed = int(flags["has_long_hair"] and flags["has_short_hair"])
        build_mixed = int(flags["has_slender"] and (flags["has_robust_build"] or flags["has_medium_build"] or flags["has_average_build"]))
        candidate_uncertain = int(flags["has_observation_uncertainty"] or flags["has_explicit_variation"] or flags["has_blur_or_quality_warning"] or hair_mixed or build_mixed)
        per_pid.append({
            "pid": pid,
            "raw_pid": by_pid[pid][0]["raw_pid"],
            "image_count": len(by_pid[pid]),
            "clothing_count": len(cloths),
            "camera_count": len(cams),
            "clothes": "|".join(cloths),
            "text_chars": len(text),
            "text_words": len(text.split()),
            "text_unique_chars": len(set(text.lower())),
            "text": text,
            "hair_mixed_lexical": hair_mixed,
            "build_mixed_lexical": build_mixed,
            "candidate_lexical_ambiguity": candidate_uncertain,
            **flags,
            "within_pid_text_unique_count": 1,
            "within_pid_cross_cloth_text_unique_count": 1,
        })
    duplicates = len(texts) - len(set(texts))
    all_summary_keys = set(summary)
    expected = {str(pid) for pid in range(len(by_pid))}
    quality = {
        "summary_path": str(SUMMARY_JSON),
        "summary_sha256": sha256_file(SUMMARY_JSON),
        "summary_pid_count": len(summary),
        "train_pid_count": len(by_pid),
        "summary_missing_pid_count": len(missing),
        "summary_extra_pid_count": len(all_summary_keys - expected),
        "nonempty_text_count": sum(bool(x) for x in texts),
        "empty_text_count": sum(not bool(x) for x in texts) + len(missing),
        "duplicate_summary_text_count": duplicates,
        "text_chars": {
            "min": min((len(x) for x in texts), default=0),
            "median": float(np.median([len(x) for x in texts])) if texts else None,
            "max": max((len(x) for x in texts), default=0),
        },
        "text_words": {
            "min": min((len(x.split()) for x in texts), default=0),
            "median": float(np.median([len(x.split()) for x in texts])) if texts else None,
            "max": max((len(x.split()) for x in texts), default=0),
        },
        "field_keys": dict(field_keys),
        "per_image_text_available": False,
        "per_image_embedding_available": True,
        "text_is_pid_summary": True,
        "train_image_count": len(records),
        "semantic_factor_count": int(blocks.shape[1]),
        "semantic_factor_dim": int(blocks.shape[2]),
        "interpretation": "summary JSON 仅提供每个 PID 一条身份汇总文本；NPZ 只含 embedding，不含每图可读原文。",
    }
    return quality, per_pid


def sample_cross_cloth_pairs(indices_by_cloth: dict[str, list[int]], rng: np.random.Generator, limit: int = 4000) -> tuple[np.ndarray, int]:
    cloths = sorted(indices_by_cloth)
    total = sum(len(indices_by_cloth[a]) * len(indices_by_cloth[b]) for pos, a in enumerate(cloths) for b in cloths[pos + 1:])
    if total <= limit:
        pairs = [(i, j) for pos, a in enumerate(cloths) for b in cloths[pos + 1:] for i in indices_by_cloth[a] for j in indices_by_cloth[b]]
        return np.asarray(pairs, dtype=np.int64), total
    selected: set[tuple[int, int]] = set()
    while len(selected) < limit:
        a, b = rng.choice(len(cloths), size=2, replace=False)
        ca, cb = cloths[int(min(a, b))], cloths[int(max(a, b))]
        i = int(rng.choice(indices_by_cloth[ca]))
        j = int(rng.choice(indices_by_cloth[cb]))
        selected.add((i, j))
    return np.asarray(sorted(selected), dtype=np.int64), total


def factor_consistency(records: list[dict[str, Any]], blocks: np.ndarray, summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_pid_cloth: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for item in records:
        by_pid_cloth[int(item["pid"])][item["cloth_key"]].append(int(item["index"]))
    summary_vectors = np.stack([
        np.asarray(summary[str(pid)]["ft_EVA02-CLIP-B-16"], dtype=np.float32).reshape(-1)
        for pid in range(len(by_pid_cloth))
    ])
    raw = blocks / np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
    rng = np.random.default_rng(20261012)
    rows: list[dict[str, Any]] = []
    for pid in sorted(by_pid_cloth):
        pairs, total = sample_cross_cloth_pairs(by_pid_cloth[pid], rng)
        if len(pairs) == 0:
            continue
        for factor_index, factor_name in enumerate(FACTOR_NAMES):
            values = np.sum(raw[pairs[:, 0], factor_index] * raw[pairs[:, 1], factor_index], axis=1)
            rows.append({
                "pid": pid,
                "factor_index": factor_index,
                "factor": factor_name,
                "all_cross_cloth_image_pair_count": total,
                "sampled_pair_count": len(values),
                "mean_cosine": float(np.mean(values)),
                "p10_cosine": float(np.quantile(values, 0.10)),
                "min_sampled_cosine": float(np.min(values)),
                "source": "LTCC train NPZ embedding；不是可读原文",
            })
        summary_norm = summary_vectors[pid] / max(float(np.linalg.norm(summary_vectors[pid])), 1e-12)
        rows.append({
            "pid": pid,
            "factor_index": 0,
            "factor": "Bio Summary运行替换槽",
            "all_cross_cloth_image_pair_count": total,
            "sampled_pair_count": len(pairs),
            "mean_cosine": 1.0,
            "p10_cosine": 1.0,
            "min_sampled_cosine": 1.0,
            "source": "每个 PID 的同一 Summary Bio 向量复制到其所有图像",
        })
    return rows


def load_anchor_metrics() -> list[dict[str, Any]]:
    """读取正常 I0->I3、O0->O1 的逐锚点指标。

    Hard Positive 目录的逐锚点文件在当前历史 artifact 中只含 Seed4321，
    因而本阶段不使用它来做双种子语义关联；改用 NDP-MARGIN 的双种子文件。
    """
    normalized: list[dict[str, Any]] = []
    with MARGIN_ANCHOR_METRICS.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row = {
                "seed": source["seed"],
                "pid": source["pid"],
                "baseline_hard_margin": source.get("bio_margin_hard"),
                "control_gpc_hard_margin": source.get("gpc_margin_hard"),
                "baseline_cross_cloth_margin": source.get("bio_margin_cross_clothes"),
                "control_gpc_cross_cloth_margin": source.get("gpc_margin_cross_clothes"),
                "baseline_cross_cloth_positive_mean": source.get("bio_pos_cross_clothes_mean"),
                "control_gpc_cross_cloth_positive_mean": source.get("gpc_pos_cross_clothes_mean"),
                "baseline_cc_recall10": "",
                "control_gpc_cc_recall10": "",
                "baseline_purity10": source.get("bio_k10_local_same_pid_purity"),
                "control_gpc_purity10": source.get("gpc_k10_local_same_pid_purity"),
                "baseline_negative_max": source.get("bio_n_max"),
                "control_gpc_negative_max": source.get("gpc_n_max"),
            }
            normalized.append(row)
    return normalized


def aggregate_feature_relation(semantic_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sem_by_pid = {int(row["pid"]): row for row in semantic_rows}
    fields = [
        "baseline_hard_margin", "control_gpc_hard_margin", "baseline_cross_cloth_margin", "control_gpc_cross_cloth_margin",
        "baseline_cross_cloth_positive_mean", "control_gpc_cross_cloth_positive_mean", "baseline_cc_recall10", "control_gpc_cc_recall10",
        "baseline_purity10", "control_gpc_purity10", "baseline_negative_max", "control_gpc_negative_max",
    ]
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in anchor_rows:
        seed = int(row["seed"])
        pid = int(row["pid"])
        if pid not in sem_by_pid:
            continue
        for field in fields:
            value = finite_float(row.get(field))
            if value is not None:
                grouped[(seed, pid)][field].append(value)
    output: list[dict[str, Any]] = []
    for (seed, pid), values in sorted(grouped.items()):
        sem = sem_by_pid[pid]
        row: dict[str, Any] = {"seed": seed, "pid": pid}
        for key in ("image_count", "clothing_count", "text_chars", "text_words", "candidate_lexical_ambiguity", "hair_mixed_lexical", "build_mixed_lexical", "has_clothing_term", "has_observation_uncertainty", "has_explicit_variation", "has_blur_or_quality_warning"):
            row[key] = sem.get(key)
        for field in fields:
            row[field] = mean_or_none(values[field])
        pairs = [
            ("delta_hard_margin", "baseline_hard_margin", "control_gpc_hard_margin"),
            ("delta_cross_cloth_margin", "baseline_cross_cloth_margin", "control_gpc_cross_cloth_margin"),
            ("delta_cross_cloth_positive_mean", "baseline_cross_cloth_positive_mean", "control_gpc_cross_cloth_positive_mean"),
            ("delta_cc_recall10", "baseline_cc_recall10", "control_gpc_cc_recall10"),
            ("delta_purity10", "baseline_purity10", "control_gpc_purity10"),
            ("delta_negative_max", "baseline_negative_max", "control_gpc_negative_max"),
        ]
        for out, left, right in pairs:
            row[out] = row[right] - row[left] if row[right] is not None and row[left] is not None else None
        output.append(row)

    def rank_values(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    def corr(rows: list[dict[str, Any]], xkey: str, ykey: str, method: str) -> dict[str, Any]:
        xs, ys = [], []
        for row in rows:
            x, y = finite_float(row.get(xkey)), finite_float(row.get(ykey))
            if x is not None and y is not None:
                xs.append(x); ys.append(y)
        if len(xs) < 4 or np.std(xs) == 0 or np.std(ys) == 0:
            value = None
        else:
            xa, ya = np.asarray(xs), np.asarray(ys)
            if method == "spearman":
                xa, ya = rank_values(xa), rank_values(ya)
            value = float(np.corrcoef(xa, ya)[0, 1])
        return {"n_pid": len(xs), "x": xkey, "y": ykey, "method": method, "correlation": value}

    relation_summary: dict[str, Any] = {"unit": "PID级；同一 PID 内图像不视为独立重复", "seeds": {}}
    for seed in (1234, 4321):
        rows = [row for row in output if int(row["seed"]) == seed and int(row["clothing_count"]) >= 2]
        relation_summary["seeds"][str(seed)] = {
            "pid_count_with_multiple_clothes": len(rows),
            "lexical_ambiguity_vs_delta_cross_cloth_margin_pearson": corr(rows, "candidate_lexical_ambiguity", "delta_cross_cloth_margin", "pearson"),
            "lexical_ambiguity_vs_delta_cross_cloth_margin_spearman": corr(rows, "candidate_lexical_ambiguity", "delta_cross_cloth_margin", "spearman"),
            "lexical_ambiguity_vs_baseline_cross_cloth_margin_pearson": corr(rows, "candidate_lexical_ambiguity", "baseline_cross_cloth_margin", "pearson"),
            "text_words_vs_delta_cross_cloth_margin_pearson": corr(rows, "text_words", "delta_cross_cloth_margin", "pearson"),
            "text_words_vs_delta_cc_recall10_pearson": corr(rows, "text_words", "delta_cc_recall10", "pearson"),
            "clothing_term_vs_delta_cross_cloth_margin": corr(rows, "has_clothing_term", "delta_cross_cloth_margin", "pearson"),
        }
    return output, relation_summary


def gpc_density() -> dict[str, Any]:
    with HP_PROTOCOL.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    result: dict[str, Any] = {
        "source_protocol": str(HP_PROTOCOL),
        "formula_from_code": "L_total = L_CE + L_Triplet + L_Bio + 0.1 * mean_{selected pairs}(1-cos(normalize(feat[0])))",
        "selector_source": "train-only；LTCC 使用已审计的跨衣同 PID 候选 bank；选择 all same-ID candidates 中稳定哈希前 min(cross,same,id) 个",
        "reduction": "selected pair 内 mean；无选中 pair 时 relation=0；不按候选总数求和",
        "density_confound": [],
    }
    for seed, info in protocol["precheck"].items():
        original = info["original"]
        hard = info["hard_positive"]
        def pack(label: str, x: dict[str, Any]) -> dict[str, Any]:
            selected = int(x["selected"])
            active = int(x["active_batches"])
            return {
                "label": label,
                "active_batches": active,
                "active_rate_over_global_batches": active / max(int(info["global_batches"]), 1),
                "selected_relations": selected,
                "selected_relations_per_active_batch": selected / max(active, 1),
                "cross_selected": int(x["cross_selected"]),
                "same_selected": int(x["same_selected"]),
                "cross_fraction_among_selected": int(x["cross_selected"]) / max(selected, 1),
                "candidate_cross": int(x["candidate_cross"]),
                "candidate_same": int(x["candidate_same"]),
                "candidate_id": int(x["candidate_id"]),
            }
        original_pack, hard_pack = pack("Original GPC", original), pack("Hard Positive", hard)
        result["density_confound"].append({
            "seed": int(seed),
            "global_batches_in_60_epoch_replay": int(info["global_batches"]),
            "original": original_pack,
            "hard_positive": hard_pack,
            "selected_relation_ratio_hard_over_original": hard_pack["selected_relations"] / max(original_pack["selected_relations"], 1),
            "active_batch_ratio_hard_over_original": hard_pack["active_batches"] / max(original_pack["active_batches"], 1),
            "cross_selection_rate_original": float(info["original_cross_selected_rate"]),
            "cross_selection_rate_hard": float(info["hard_cross_selected_rate"]),
        })
    return result


def feature_audit() -> dict[str, Any]:
    output = {}
    for name, path in FEATURES.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = sha256_file(path)
            with np.load(path, allow_pickle=True) as archive:
                item["keys"] = list(archive.files)
                item["shapes"] = {key: list(archive[key].shape) for key in archive.files}
                item["dtypes"] = {key: str(archive[key].dtype) for key in archive.files}
                values = np.asarray(archive["features"])
                item["finite"] = bool(np.isfinite(values).all())
                item["feature_dim"] = int(values.shape[1])
                item["sample_count"] = int(values.shape[0])
                item["image_path_sha256"] = hashlib.sha256("\n".join(map(str, archive["image_path"].tolist())).encode()).hexdigest()
                for key in ("pid", "clothes", "camid"):
                    if key in archive.files:
                        item[f"{key}_sha256"] = hashlib.sha256(np.asarray(archive[key]).tobytes()).hexdigest()
        output[name] = item
    reference = output.get("seed1234_baseline", {})
    alignment_fields = ["image_path_sha256", "pid_sha256", "clothes_sha256"]
    output["all_four_alignment"] = {
        field: bool(reference.get(field)) and all(item.get(field) == reference.get(field) for key, item in output.items() if key.startswith("seed"))
        for field in alignment_fields
    }
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    log = Path(args.log)
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log_msg(message: str, filename: str = "execution.log") -> None:
        log_lines.append(message)
        with (log / filename).open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    log_msg("开始执行语义监督可靠性只读审计；不构造模型、不训练、不读取 LTCC query/gallery。")
    records, pid_map = parse_records()
    summary = load_summary()
    archive = load_text_archive(records, summary)
    quality, semantic_rows = semantic_quality(records, summary, archive["blocks"])
    factor_rows = factor_consistency(records, archive["blocks"], summary)
    anchor_rows = load_anchor_metrics()
    relation_rows, relation_summary = aggregate_feature_relation(semantic_rows, anchor_rows)
    density = gpc_density()
    features = feature_audit()

    write_json(out / "semantic_data_quality.json", quality)
    write_json(out / "gpc_density.json", density)
    write_json(out / "feature_audit.json", features)
    write_json(out / "semantic_feature_relation.json", relation_summary)
    write_csv(out / "per_pid_semantic.csv", semantic_rows)
    write_csv(out / "per_pid_feature_relation.csv", relation_rows)
    write_csv(out / "per_pid_factor_consistency.csv", factor_rows)

    protocol = {
        "phase": "PHASE SEMANTIC-RELIABILITY-AUDIT",
        "analysis_timestamp_beijing": out.name,
        "output_dir": str(out),
        "log_dir": str(log),
        "dataset_scope": "LTCC train-only；仅读取 /home/data/LTCC_ReID/train",
        "query_gallery_test_used": False,
        "model_training_or_optimizer_step": False,
        "semantic_source": {
            "npz": str(TEXT_NPZ),
            "summary_json": str(SUMMARY_JSON),
            "npz_factor_order": FACTOR_NAMES,
            "npz_alignment": "metadata key 映射到 sorted train image stem，再 reshape 为 6 factor block",
            "raw_caption_text_per_image": False,
            "pid_summary_text": True,
        },
        "reference_feature_sources": {name: str(path) for name, path in FEATURES.items()},
        "semantic_relationship_statistics": "仅作 PID 级观察性关联；不将词面标记视为真实矛盾标签，不作因果结论",
        "factor_consistency_sampling": "每个 PID 跨衣图像对最多 4000 对，seed=20261012；少于上限时全量",
        "source_files": {
            "dataset_loader": str(ROOT / "data/datasets/ltcc.py"),
            "dataset_builder": str(ROOT / "data/__init__.py"),
            "formal_train_script": str(ROOT / "scripts/train_final_gpc.py"),
            "loss": str(ROOT / "loss/make_loss.py"),
            "model": str(ROOT / "model/eva_meta_bio.py"),
        },
        "source_hashes": {str(path): sha256_file(path) for path in [ROOT / "data/datasets/ltcc.py", ROOT / "data/__init__.py", ROOT / "scripts/train_final_gpc.py", ROOT / "loss/make_loss.py", ROOT / "model/eva_meta_bio.py"]},
        "counts": {
            "train_images": len(records),
            "train_pids": len(pid_map),
            "multi_cloth_pids": len({item["pid"] for item in records if sum(1 for x in records if x["pid"] == item["pid"]) > 0}),
        },
        "conclusion_gate": "若需要逐图语义冲突/属性正确性判断，当前 archive 不足；本阶段主判定不得高于 INSUFFICIENT_SEMANTIC_DATA。",
    }
    # Replace the deliberately simple multi-cloth count with the exact value.
    pid_cloths = defaultdict(set)
    for item in records:
        pid_cloths[item["pid"]].add(item["cloth_key"])
    protocol["counts"]["multi_cloth_pids"] = sum(len(v) >= 2 for v in pid_cloths.values())
    write_json(out / "protocol.json", protocol)

    log_msg(f"LTCC train 图像={len(records)}，PID={len(pid_map)}，多服装 PID={protocol['counts']['multi_cloth_pids']}。", "audit.log")
    log_msg(f"语义 NPZ={archive['data'].shape}，metadata 对齐通过；Summary 文本={quality['summary_pid_count']} 条。", "semantic_source.log")
    log_msg("确认：没有逐图可读 caption；仅有 PID 汇总文本和六因子 embedding。", "semantic_source.log")
    log_msg(f"Summary 文本非空={quality['nonempty_text_count']}，重复文本={quality['duplicate_summary_text_count']}；这些不是逐图语义质量标签。", "data_quality.log")
    for item in density["density_confound"]:
        log_msg(f"Seed{item['seed']}：Original 选中关系={item['original']['selected_relations']}，Hard Positive={item['hard_positive']['selected_relations']}，比例={item['selected_relation_ratio_hard_over_original']:.4f}；关系内均值归约未按数量求和。", "audit.log")
    for seed, info in relation_summary["seeds"].items():
        log_msg(f"Seed{seed}：PID级词面模糊标记与 GPC 跨衣间隔变化的 Pearson={info['lexical_ambiguity_vs_delta_cross_cloth_margin_pearson']['correlation']}；仅作探索性关联。", "feature_relation.log")
    log_msg("审计结束：逐图身份语义冲突无法由当前资源可靠识别。", "execution.log")


if __name__ == "__main__":
    main()
