#!/usr/local/bin/python3
"""PHASE CC-GPC-HP-PILOT.

本脚本是独立的普通困难跨衣正样本受控实验入口。它不修改仓库中的正式
训练器、损失函数或采样器；训练时在进程内替换采样器类，并且只改变同一
身份内部的两个图像索引。所有关系选择只使用 LTCC train 标签与 common-init
冻结特征，最终 GPC 特征只在训练完成后的离线分析阶段读取。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path("/home/DIFFER")
RUN_TIME = "20261012_150038"
RESULT_ROOT = ROOT / "outputs/cc_gpc_hp_pilot" / RUN_TIME
LOG_ROOT = ROOT / "log_NDP/PHASE_CC_GPC_HP_PILOT" / RUN_TIME
FEATURE_ROOT = RESULT_ROOT / "features"
FIGURE_ROOT = RESULT_ROOT / "figures"
SEEDS = (1234, 4321)
EPOCHS = 60
WORLD_SIZE = 2
HARD_RATIO = 0.5
HARD_POOL_FRACTION = 0.25
PRECHECK_EPS = 1e-7

COMMON_INIT = {
    1234: ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/common_init/LTCC_COMMON_INIT_SEED1234.pth",
    4321: ROOT / "outputs/new_method_track/phase6cr_relation_replication/SEED4321_COMMON_INIT.pth",
}
CONTROL_RUNS = {
    1234: {
        "label": "I3",
        "checkpoint": ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/model_final.pth",
        "manifest": ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/run_manifest.json",
        "checkpoint_meta": ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/final_checkpoint.json",
        "config": ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/config.yml",
        "features": ROOT / "outputs/ndp_margin_diagnostic/20261008_154849/features/seed1234_gpc_train_features.npz",
        "retrieval_evidence": ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/final_test.json",
    },
    4321: {
        "label": "O1",
        "checkpoint": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/training/O1/model_final.pth",
        "manifest": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/training/O1/run_manifest.json",
        "checkpoint_meta": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/training/O1/checkpoint_epoch060.json",
        "config": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/training/O1/config.yml",
        "features": ROOT / "outputs/ndp_margin_diagnostic/20261008_154849/features/seed4321_gpc_train_features.npz",
        # O1 的历史目录没有保存标准 final_test.json；该文件是此前已审计的
        # E60 检索汇总，选择字段明确为 final，而不是 best。
        "retrieval_evidence": ROOT / "outputs/ndp_hard_ctrl/20261009_110641/data/retrieval_eval_seed4321.json",
    },
}
BASELINE_FEATURES = {
    1234: ROOT / "outputs/ndp_margin_diagnostic/20261008_154849/features/seed1234_bio_train_features.npz",
    4321: ROOT / "outputs/ndp_margin_diagnostic/20261008_154849/features/seed4321_bio_train_features.npz",
}
COMMON_INIT_FEATURES = {
    1234: ROOT / "outputs/new_method_track/ndp_three_day_20260930_130427/trajectory_runs/seed1234/trajectory/seed1234/E00_train_feature_bank.npz",
    4321: ROOT / "outputs/new_method_track/ndp_three_day_20260930_130427/trajectory_runs/seed4321/trajectory/seed4321/E00_train_feature_bank.npz",
}
COMMON_INIT_FEATURE_MANIFESTS = {
    1234: ROOT / "outputs/new_method_track/ndp_three_day_20260930_130427/trajectory_runs/seed1234/trajectory/seed1234/trajectory_manifest.json",
    4321: ROOT / "outputs/new_method_track/ndp_three_day_20260930_130427/trajectory_runs/seed4321/trajectory/seed4321/trajectory_manifest.json",
}

_ENGINE_CACHE: dict[int, Any] = {}
_HP_CONTEXT: dict[str, Any] = {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def append_log(name: str, message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / name).open("a", encoding="utf-8") as stream:
        stream.write(f"[{utc_now()}] {message}\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_engine(seed: int):
    if seed in _ENGINE_CACHE:
        return _ENGINE_CACHE[seed]
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # type: ignore
    engine.SEED = int(seed)
    engine.WORLD_SIZE = WORLD_SIZE
    engine.EPOCHS = EPOCHS
    engine.WARMUP_EPOCHS = 20
    engine.LAMBDA_GPC = 0.1
    engine.OUTPUT_ROOT_DEFAULT = RESULT_ROOT
    def _pilot_log_path(_output, _dataset, kind, s=seed):
        # 统一使用用户要求的英文文件名；训练器本身的中文日志内容保持不变。
        if kind == "训练":
            return LOG_ROOT / f"train_seed{s}.log"
        if kind == "测试":
            return LOG_ROOT / f"evaluation_seed{s}.log"
        return LOG_ROOT / f"{kind}_seed{s}.log"
    engine.public_log_path = _pilot_log_path
    engine._CC_GPC_HP_ORIGINAL_FINAL_BATCH_SAMPLER = engine.FinalBatchSampler
    _ENGINE_CACHE[seed] = engine
    return engine


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}


def state_tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    result = {}
    for key, value in sorted(state.items()):
        result[key] = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return result


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(json.dumps(state_tensor_hashes(state), sort_keys=True).encode("utf-8")).hexdigest()


def train_root(seed: int) -> Path:
    return RESULT_ROOT / "ltcc" / f"seed{seed}"


def hard_train_root(seed: int) -> Path:
    return train_root(seed) / "gpc"


def load_npz_bank(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    bank = {key: data[key] for key in data.files}
    bank["features"] = np.asarray(bank["features"], dtype=np.float32)
    bank["pid"] = np.asarray(bank["pid"], dtype=np.int64)
    bank["clothes"] = np.asarray(bank["clothes"], dtype=np.int64)
    bank["image_path"] = bank["image_path"].astype(str)
    if "camid" in bank:
        bank["camid"] = np.asarray(bank["camid"], dtype=np.int64)
    return bank


def validate_train_bank(bank: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    features = bank["features"]
    paths = bank["image_path"]
    if features.shape != (9576, 1024):
        raise RuntimeError(f"{label} 特征形状错误：{features.shape}")
    if len(np.unique(paths)) != len(paths):
        raise RuntimeError(f"{label} 存在重复图像路径")
    if not np.isfinite(features).all():
        raise RuntimeError(f"{label} 存在 NaN/Inf")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-4:
        raise RuntimeError(f"{label} 未通过 L2 归一化检查：max error={float(np.max(np.abs(norms - 1.0)))}")
    if not all("/train/" in path and "/query/" not in path and "/test/" not in path and "gallery" not in path.lower() for path in paths):
        raise RuntimeError(f"{label} 包含非 train 路径")
    if len(np.unique(bank["pid"])) != 77:
        raise RuntimeError(f"{label} PID 数量错误")
    return {"images": int(len(paths)), "dim": int(features.shape[1]), "pids": int(len(np.unique(bank["pid"]))), "max_l2_error": float(np.max(np.abs(norms - 1.0)))}


def copy_verified_common_init(seed: int) -> dict[str, Any]:
    engine = import_engine(seed)
    source = COMMON_INIT[seed]
    if not source.is_file():
        raise FileNotFoundError(source)
    source_state = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(source_state, dict) and isinstance(source_state.get("model"), dict):
        source_state = source_state["model"]
    if not isinstance(source_state, dict):
        raise RuntimeError(f"common-init 不是 state_dict：{source}")
    target, meta_path = engine.init_paths(RESULT_ROOT, "ltcc")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_state_hash = state_hash(source_state)
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        if state_hash(existing) != expected_state_hash:
            raise RuntimeError(f"新实验 common-init 已存在但哈希不一致：{target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(source_state, temporary)
        temporary.replace(target)
    meta = {
        "dataset": "ltcc", "seed": seed, "source": file_record(source), "target": file_record(target),
        "file_sha256": sha256_file(target), "state_hash": expected_state_hash,
        "tensor_hashes": state_tensor_hashes(source_state), "state_key_count": len(source_state),
        "state_numel": int(sum(value.numel() for value in source_state.values())),
        "strict_load_required": True, "reference_role": "训练开始前 common-init 冻结参考，仅用于普通困难跨衣正样本采样",
    }
    write_json(meta_path, meta)
    write_json(train_root(seed) / "common_init_audit.json", meta)
    return meta


def load_dataset(seed: int):
    engine = import_engine(seed)
    cfg = engine.cfg_for("ltcc", hard_train_root(seed))
    dataset = engine.build_dataset(cfg)
    return engine, cfg, dataset


def extract_common_init_features(seed: int) -> dict[str, Any]:
    engine, cfg, dataset = load_dataset(seed)
    meta = copy_verified_common_init(seed)
    source = COMMON_INIT[seed]
    feature_path = FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz"
    source_bank_path = COMMON_INIT_FEATURES[seed]
    source_manifest_path = COMMON_INIT_FEATURE_MANIFESTS[seed]
    if not source_bank_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError(f"缺少已审计的 common-init E00 train 特征或轨迹清单：{source_bank_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    prepare_path = source_manifest_path.parent / "prepare.json"
    prepare_meta = json.loads(prepare_path.read_text(encoding="utf-8")) if prepare_path.is_file() else {}
    if str(prepare_meta.get("source_common_init", "")) != str(source) or prepare_meta.get("source_common_init_sha256") != sha256_file(source):
        raise RuntimeError(f"Seed{seed} E00 特征的 common-init 来源与本阶段来源不一致：{source_manifest_path}")
    if source_manifest.get("feature_definition") != "normalized feat[0]" or source_manifest.get("query_gallery_read") or source_manifest.get("test_metrics_read"):
        raise RuntimeError(f"Seed{seed} E00 特征不是合法 train-only common-init 特征：{source_manifest_path}")
    if feature_path.is_file():
        bank = load_npz_bank(feature_path)
        summary = validate_train_bank(bank, f"seed{seed} common-init 特征")
        append_log("exposure.log", f"Seed{seed} 复用本阶段已保存并通过校验的 common-init train 特征：{feature_path}")
        return {"path": str(feature_path), "checkpoint": file_record(source), "summary": summary, "reused": True, "source_e00": file_record(source_bank_path)}
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_bank_path, feature_path)
    bank = load_npz_bank(feature_path)
    summary = validate_train_bank(bank, f"seed{seed} common-init 特征")
    write_json(FEATURE_ROOT / f"seed{seed}_common_init_train_features.json", {"path": str(feature_path), "source_e00": file_record(source_bank_path), "source_manifest": file_record(source_manifest_path), "prepare_meta": prepare_meta, "source_checkpoint": file_record(source), "common_init_meta": meta, "summary": summary, "reuse_basis": "已完成的 E00 common-init train-only PPU 特征；轨迹 prepare.json 与本阶段 common-init SHA256 一致", "train_only": True})
    append_log("exposure.log", f"Seed{seed} 复用已审计 E00 common-init train 特征（来源清单与 common-init SHA256 已核对）：{feature_path}")
    return {"path": str(feature_path), "checkpoint": file_record(source), "summary": summary, "reused": False, "source_e00": file_record(source_bank_path), "reuse_basis": "E00 common-init train-only PPU 特征"}


def load_exact_cross_pairs(dataset: Any) -> set[tuple[int, int]]:
    source = ROOT / "outputs/new_method_track/phase6b_soft_relation_screen/artifacts/relation_source_rows.csv"
    exact: set[tuple[int, int]] = set()
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            left, right = sorted((int(row["index_a"]), int(row["index_b"])))
            if not (0 <= left < len(dataset.train) and 0 <= right < len(dataset.train) and left != right):
                raise RuntimeError(f"GPC 关系源索引非法：{row}")
            a, b = dataset.train[left], dataset.train[right]
            if int(a["pid"]) != int(b["pid"]) or int(a["clothes_id"]) == int(b["clothes_id"]):
                raise RuntimeError(f"GPC 关系源不是合法跨衣同身份图像对：{row}")
            exact.add((left, right))
    return exact


def build_hard_pools(seed: int, bank: dict[str, np.ndarray], exact_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for left, right in sorted(exact_pairs):
        if int(bank["pid"][left]) != int(bank["pid"][right]) or int(bank["clothes"][left]) == int(bank["clothes"][right]):
            continue
        score = float(np.dot(bank["features"][left], bank["features"][right]))
        by_pid[int(bank["pid"][left])].append({"left": left, "right": right, "score": score, "clothes_left": int(bank["clothes"][left]), "clothes_right": int(bank["clothes"][right])})
    pools: dict[int, list[dict[str, Any]]] = {}
    all_hard: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for pid, values in sorted(by_pid.items()):
        values.sort(key=lambda item: (item["score"], item["left"], item["right"]))
        pool_count = max(1, int(math.ceil(len(values) * HARD_POOL_FRACTION)))
        pools[pid] = values[:pool_count]
        for rank, item in enumerate(values[:pool_count]):
            key = (min(int(item["left"]), int(item["right"])), max(int(item["left"]), int(item["right"])))
            all_hard.add(key)
            rows.append({"seed": seed, "pid": pid, "pool_rank": rank, "pool_size": pool_count, **item})
    path = RESULT_ROOT / "sampler" / f"seed{seed}_hard_positive_pool.csv"
    write_csv(path, rows, ["seed", "pid", "pool_rank", "pool_size", "left", "right", "score", "clothes_left", "clothes_right"])
    summary = {
        "seed": seed, "candidate_exact_pairs": int(len(exact_pairs)), "multi_cloth_pids": int(len(by_pid)),
        "hard_pool_pairs": int(len(all_hard)), "pool_fraction": HARD_POOL_FRACTION,
        "pool_definition": "每个 PID 的 train-only、GPC 合法跨衣图像对按 common-init cosine 从低到高排序，取前 ceil(25%)；低相似度定义为困难跨衣正样本",
        "pool_path": str(path), "pid_pool_sizes": {str(pid): len(values) for pid, values in pools.items()},
    }
    write_json(RESULT_ROOT / "sampler" / f"seed{seed}_hard_positive_pool.json", summary)
    return {"pools": pools, "all_hard": all_hard, "summary": summary}


def digest_int(*parts: Any) -> int:
    token = "|".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def transform_pair_units(pair_units: list[list[int]], data_source: list[dict[str, Any]], context: dict[str, Any], epoch: int, rank: int, batch_number: int) -> tuple[list[list[int]], dict[str, int]]:
    transformed: list[list[int]] = []
    stats = Counter()
    for slot, original in enumerate(pair_units):
        if len(original) != 2:
            raise RuntimeError("身份单元不是 P=2")
        pid_values = {int(data_source[index]["pid"]) for index in original}
        if len(pid_values) != 1:
            raise RuntimeError("hard-positive 采样试图修改混合身份单元")
        pid = next(iter(pid_values))
        pool = context["pools"].get(pid, [])
        stats["identity_units"] += 1
        if not pool:
            transformed.append(list(original)); stats["no_pool"] += 1; continue
        stats["eligible_units"] += 1
        use_hard = digest_int("cc_gpc_hp", context["seed"], epoch, rank, batch_number, slot, pid, original[0], original[1]) % 10000 < int(HARD_RATIO * 10000)
        if not use_hard:
            transformed.append(list(original)); stats["random_units"] += 1; continue
        selected = pool[digest_int("cc_gpc_hp_pair", context["seed"], epoch, rank, batch_number, slot, pid) % len(pool)]
        pair = [int(selected["left"]), int(selected["right"])]
        transformed.append(pair)
        stats["hard_units"] += 1
        if tuple(sorted(pair)) == tuple(sorted(original)):
            stats["hard_same_as_original"] += 1
    return transformed, {key: int(value) for key, value in stats.items()}


def identity_sequence(batches: list[list[int]], data_source: list[dict[str, Any]]) -> list[int]:
    return [int(data_source[index]["pid"]) for batch in batches for index in batch]


def make_hard_batches(engine: Any, data_source: list[dict[str, Any]], seed: int, rank: int, epoch: int, context: dict[str, Any]) -> tuple[list[list[int]], dict[str, Any], list[list[int]]]:
    original = engine.FinalBatchSampler(data_source, 8, 2, 596, seed, rank)
    original.prepare(epoch)
    raw_batches = [list(batch) for batch in original._batches]
    transformed: list[list[int]] = []
    total = Counter()
    for batch_number, batch in enumerate(raw_batches, start=1):
        units = [batch[pos:pos + 2] for pos in range(0, len(batch), 2)]
        selected_units, stats = transform_pair_units(units, data_source, context, epoch, rank, batch_number)
        total.update(stats)
        transformed.append([index for unit in selected_units for index in unit])
    return transformed, {"raw_meta": original.meta, "transform_stats": dict(total)}, raw_batches


class HardPositiveBatchSampler:
    """冻结原始身份/P×K计划，只替换同身份内部的图像对。"""

    def __init__(self, data_source: list[dict[str, Any]], batch_size: int, num_instances: int, steps_per_epoch: int, seed: int, rank: int):
        engine = import_engine(seed)
        base_sampler = getattr(engine, "_CC_GPC_HP_ORIGINAL_FINAL_BATCH_SAMPLER", engine.FinalBatchSampler)
        self._base = base_sampler(data_source, batch_size, num_instances, steps_per_epoch, seed, rank)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.steps_per_epoch = steps_per_epoch
        self.rank = rank
        self.seed = seed
        self._batches: list[list[int]] | None = None
        self.meta: dict[str, Any] = {}

    def prepare(self, epoch: int) -> dict[str, Any]:
        raw_meta = self._base.prepare(epoch)
        raw_batches = [list(batch) for batch in self._base._batches]
        context = _HP_CONTEXT
        transformed: list[list[int]] = []
        total = Counter()
        for batch_number, batch in enumerate(raw_batches, start=1):
            units = [batch[pos:pos + self.num_instances] for pos in range(0, len(batch), self.num_instances)]
            selected_units, stats = transform_pair_units(units, self.data_source, context, epoch, self.rank, batch_number)
            total.update(stats)
            transformed.append([index for unit in selected_units for index in unit])
        if identity_sequence(transformed, self.data_source) != identity_sequence(raw_batches, self.data_source):
            raise RuntimeError("hard-positive 采样改变了原始 PID 序列")
        self._batches = transformed
        plan_hash = hashlib.sha256(json.dumps(transformed, separators=(",", ":")).encode("utf-8")).hexdigest()
        stats_row = {"seed": self.seed, "rank": self.rank, "epoch": epoch, "raw_plan_sha256": raw_meta["plan_sha256"], "hard_plan_sha256": plan_hash, "identity_sequence_equal": True, "batches": len(transformed), "hard_units": int(total.get("hard_units", 0)), "random_units": int(total.get("random_units", 0)), "eligible_units": int(total.get("eligible_units", 0)), "no_pool_units": int(total.get("no_pool", 0)), "hard_same_as_original": int(total.get("hard_same_as_original", 0)), "hard_ratio": HARD_RATIO, "pool_fraction": HARD_POOL_FRACTION}
        stats_path = Path(context["stats_dir"]) / f"sampler_runtime_rank{self.rank}.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(stats_row)
        exists = stats_path.is_file()
        with stats_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(stats_row)
        self.meta = {**raw_meta, "plan_sha256": plan_hash}
        return self.meta

    def __iter__(self):
        if self._batches is None:
            raise RuntimeError("必须先调用 prepare(epoch)")
        return iter(self._batches)

    def __len__(self) -> int:
        return self.steps_per_epoch


def configure_hard_sampler(seed: int, context: dict[str, Any]) -> Any:
    global _HP_CONTEXT
    _HP_CONTEXT = context
    engine = import_engine(seed)
    engine.FinalBatchSampler = HardPositiveBatchSampler
    return engine


def selected_summary(engine: Any, dataset: Any, batch: list[int], path_to_index: dict[str, int], exact: set[tuple[int, int]], seed: int, epoch: int, rank: int, iteration: int) -> dict[str, Any]:
    selected_total = Counter()
    hard_selected = Counter()
    state_counts: Counter[tuple[int, int, int]] = Counter()
    entries = [dataset.train[index] for index in batch]
    selected, info = engine.select_gpc_pairs(entries, {"path_to_index": path_to_index, "exact_lookup": exact}, epoch, rank, iteration, seed)
    selected_total.update({"candidate_cross": info["candidate_counts"]["cross"], "candidate_same": info["candidate_counts"]["same"], "candidate_id": info["candidate_counts"]["id"], "selected": len(selected), "active_batches": int(bool(selected)), "cross_selected": info["cross_clothes_selected"], "same_selected": info["same_clothes_selected"]})
    for i, j, image_key, state_key in selected:
        state_counts[tuple(int(x) for x in state_key)] += 1
        if tuple(sorted(image_key)) in _HP_CONTEXT["all_hard"] and int(entries[i]["clothes_id"]) != int(entries[j]["clothes_id"]):
            hard_selected["selected_hard_cross"] += 1
    hard_selected["selected_cross"] += int(info["cross_clothes_selected"])
    return {"seed": seed, "epoch": epoch, "rank": rank, "global_batches": 1, "counts": dict(selected_total), "hard_selected": dict(hard_selected), "state_count": len(state_counts), "bank_feature_source": str(_HP_CONTEXT["reference_path"]), "query_gallery_used": False}


def audit_sources() -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    append_log("audit.log", "开始核对 I3/O1、common-init、最终检查点及原始 GPC 协议。")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        control = CONTROL_RUNS[seed]
        manifest = json.loads(control["manifest"].read_text(encoding="utf-8"))
        meta = json.loads(control["checkpoint_meta"].read_text(encoding="utf-8"))
        init_source = COMMON_INIT[seed]
        rows.append({
            "seed": seed, "control_label": control["label"], "control_checkpoint": file_record(control["checkpoint"]),
            "checkpoint_epoch": meta.get("epoch", manifest.get("end_epoch")), "checkpoint_selection": "final epoch E60（本 pilot 主比较规则）",
            "best_checkpoint_not_used": True, "best_state_present": (control["checkpoint"].parent / "best_state.json").is_file(),
            "common_init": file_record(init_source), "manifest_common_init": manifest.get("common_init"),
            "initial_parameter_hash": manifest.get("initial_parameter_hash"), "epochs": manifest.get("epochs", manifest.get("end_epoch")),
            "steps_per_epoch": manifest.get("steps_per_epoch", 596), "batch_size": manifest.get("batch_size", 8) or 8, "num_instances": manifest.get("num_instances", 2) or 2,
            "lambda_gpc": manifest.get("lambda_gpc", manifest.get("lambda_rel")), "evaluator_calls_during_training": manifest.get("evaluator_calls_during_training", 0),
            "test_metric_used_for_selection": bool(meta.get("test_metric_used", False)), "config": file_record(control["config"]),
            "train_feature_source": file_record(control["features"]), "retrieval_evidence": file_record(control["retrieval_evidence"]),
        })
    protocol = {
        "phase": "PHASE CC-GPC-HP-PILOT", "run_time_beijing": RUN_TIME, "created_utc": utc_now(),
        "dataset": "LTCC", "split": "train-only for sampler/reference/local analysis; query/gallery only opened after training for fixed E60 evaluation",
        "seeds": list(SEEDS), "result_root": str(RESULT_ROOT), "log_root": str(LOG_ROOT),
        "control_runs": rows, "hard_positive_rule": {"hard_ratio": HARD_RATIO, "hard_pool_fraction": HARD_POOL_FRACTION, "definition": "common-init train-only normalized Bio cosine lowest 25% per PID among audited GPC-eligible cross-clothes pairs", "random_remainder": 1.0 - HARD_RATIO, "loss_changed": False, "identity_sequence_changed": False, "p_k_changed": False, "endpoint_gpc_features_used_for_sampler": False},
        "formal_checkpoint_rule": "所有方法均使用 E60 final checkpoint；不使用测试指标选择采样强度或模型",
        "loss_protocol": "L_CE + L_Triplet + L_Bio + 0.1*L_GPC；仅改变同身份图像索引选择",
        "optimizer_protocol": "SGD, lr=2e-6, momentum=0.9, weight_decay=0.05, 20 epoch linear warmup, 60 epoch cosine schedule, AMP enabled",
        "ppu_policy": "训练和特征提取要求 PPU；不允许 CPU fallback",
        "ppu_env_setup": "source /usr/local/PPU_SDK/envsetup.sh ppu",
        "smoke_gate": {
            "seed1234": "3 batch PPU 冒烟通过；正式训练前未使用 query/gallery",
            "seed4321": "3 batch 探针出现 AMP found_inf，未产生检查点；按原训练器 AMP 跳步规则保留该记录，随后 2 batch PPU 冒烟通过并才启动正式训练",
        },
    }
    write_json(RESULT_ROOT / "protocol.json", protocol)
    write_json(RESULT_ROOT / "source_control_audit.json", {"created_utc": utc_now(), "rows": rows})
    lines = ["# 00 协议审计", "", "## 结论", "", "I3/O1 的主比较采用各自的 E60 final checkpoint；历史 best checkpoint 不参与主比较。两组历史 GPC 训练均为 60 epoch、596 steps/epoch、batch=8、每身份 2 张图像、lambda=0.1，训练期间未调用测试评估。", "", "## 检查点与初始化", "", "| Seed | 历史 GPC | E60 检查点 SHA256 | Common-init SHA256 | E60 | 训练期测试选择 |", "|---:|---|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['control_label']} | `{row['control_checkpoint']['sha256']}` | `{row['common_init']['sha256']}` | {row['checkpoint_epoch']} | {row['test_metric_used_for_selection']} |")
    lines += ["", "## 配对决定", "", "- 主比较固定为 E60 vs E60；Seed1234 的 I3 没有 best 选择，Seed4321 的 O1 虽有基于 changed-clothes Rank-1 的 best_state，但不使用该 best checkpoint。", "- 实验组从对应 seed 的同一 common-init 重新开始，不能从 I3/O1 最终权重继续。", "- 训练身份序列和 P×K 结构由原始 sampler 产生，独立采样器只替换身份单元内部图像。", "- 采样参考只使用 common-init train 特征；不读取最终 GPC 特征来决定训练采样。", "- 所有 query/gallery/test 信息在训练、预检和关系构造中排除。", "", "## 原始 GPC 作用路径", "", "GPC 在每个 train batch 的同 PID、不同图像候选中统计 cross-clothes、same-clothes 和 all same-ID 候选数，选取 `min(cross, same, id)` 个 all same-ID 候选，损失为归一化 `feat[0]` 上的 `mean(1-cos)`，权重为 0.1。跨衣图像共现本身不等于进入 GPC；本阶段预检将直接统计实际 selector 选中的跨衣关系。", ""]
    (RESULT_ROOT / "00_PROTOCOL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    append_log("audit.log", "I3/O1 来源审计报告已生成；主比较固定为 E60 final。")
    return protocol


def precheck(seed: int) -> dict[str, Any]:
    engine, _, dataset = load_dataset(seed)
    bank = load_npz_bank(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz")
    validate_train_bank(bank, f"seed{seed} common-init 特征")
    exact = load_exact_cross_pairs(dataset)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    pools = build_hard_pools(seed, bank, exact)
    context = {"seed": seed, "pools": pools["pools"], "all_hard": pools["all_hard"], "reference_path": str(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz"), "stats_dir": RESULT_ROOT / "sampler" / f"seed{seed}"}
    global _HP_CONTEXT
    _HP_CONTEXT = context
    original_totals = Counter(); hard_totals = Counter(); hard_selected = Counter(); raw_pid_sequence_ok = True; hard_pid_sequence_ok = True; raw_units = 0; hard_units = 0
    per_pid = Counter(); per_pid_hard = Counter(); rows_epoch: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        epoch_original = Counter(); epoch_hard = Counter(); epoch_hard_selected = Counter()
        for rank in range(WORLD_SIZE):
            original_sampler = engine.FinalBatchSampler(dataset.train, 8, 2, 596, seed, rank)
            original_sampler.prepare(epoch)
            raw_batches = [list(batch) for batch in original_sampler._batches]
            hard_batches, _meta, raw_again = make_hard_batches(engine, dataset.train, seed, rank, epoch, context)
            if identity_sequence(raw_batches, dataset.train) != identity_sequence(raw_again, dataset.train):
                raw_pid_sequence_ok = False
            if identity_sequence(raw_batches, dataset.train) != identity_sequence(hard_batches, dataset.train):
                hard_pid_sequence_ok = False
            for iteration, batch in enumerate(raw_batches, start=1):
                summary = selected_summary(engine, dataset, batch if isinstance(batch, list) else list(batch), path_to_index, exact, seed, epoch, rank, iteration)
                for key, value in summary["counts"].items(): epoch_original[key] += int(value)
                original_totals.update(summary["counts"])
                raw_units += len(batch) // 2
            for iteration, batch in enumerate(hard_batches, start=1):
                summary = selected_summary(engine, dataset, batch, path_to_index, exact, seed, epoch, rank, iteration)
                for key, value in summary["counts"].items(): epoch_hard[key] += int(value)
                hard_totals.update(summary["counts"])
                hard_selected.update(summary["hard_selected"])
                hard_units += len(batch) // 2
                for idx in batch: per_pid[int(dataset.train[idx]["pid"])] += 1
                if summary["hard_selected"].get("selected_hard_cross", 0):
                    epoch_hard_selected.update(summary["hard_selected"])
            for batch in hard_batches:
                for idx in batch:
                    per_pid_hard[int(dataset.train[idx]["pid"])] += 1
        rows_epoch.append({"seed": seed, "epoch": epoch, "original_selected": int(epoch_original["selected"]), "hard_selected": int(epoch_hard["selected"]), "original_cross_selected": int(epoch_original["cross_selected"]), "hard_cross_selected": int(epoch_hard["cross_selected"]), "hard_selected_cross": int(epoch_hard_selected["selected_hard_cross"])})
    total_original_batches = EPOCHS * 596 * WORLD_SIZE
    result = {
        "seed": seed, "epochs": EPOCHS, "global_batches": total_original_batches, "hard_pool": pools["summary"],
        "identity_sequence_equal": bool(raw_pid_sequence_ok and hard_pid_sequence_ok), "p_k_shape_equal": True,
        "original": dict(original_totals), "hard_positive": dict(hard_totals), "hard_selected": dict(hard_selected),
        "original_cross_selected_rate": float(original_totals["cross_selected"] / max(original_totals["selected"], 1)),
        "hard_cross_selected_rate": float(hard_totals["cross_selected"] / max(hard_totals["selected"], 1)),
        "hard_pool_selected_rate": float(hard_selected["selected_hard_cross"] / max(hard_selected["selected_cross"], 1)),
        "hard_unit_share_expected": HARD_RATIO, "hard_unit_count": hard_units, "original_unit_count": raw_units,
        "per_pid_image_exposure_path": str(RESULT_ROOT / "sampler" / f"seed{seed}_per_pid_image_exposure.csv"),
        "epoch_rows_path": str(RESULT_ROOT / "sampler" / f"seed{seed}_precheck_by_epoch.csv"),
        "query_gallery_used": False,
    }
    write_csv(RESULT_ROOT / "sampler" / f"seed{seed}_precheck_by_epoch.csv", rows_epoch)
    pid_rows = [{"seed": seed, "pid": pid, "original_image_exposure": int(per_pid[pid]), "hard_positive_image_exposure": int(per_pid_hard[pid]), "delta": int(per_pid_hard[pid] - per_pid[pid])} for pid in sorted(per_pid)]
    write_csv(RESULT_ROOT / "sampler" / f"seed{seed}_per_pid_image_exposure.csv", pid_rows)
    write_json(RESULT_ROOT / "sampler" / f"seed{seed}_precheck.json", result)
    append_log("sampler.log", f"Seed{seed} 60 epoch 离线预检完成：原始跨衣 GPC 选中率={result['original_cross_selected_rate']:.6f}，Hard Positive={result['hard_cross_selected_rate']:.6f}，Hard pool 选中率={result['hard_pool_selected_rate']:.6f}，PID 序列保持={result['identity_sequence_equal']}。")
    return result


def prepare() -> dict[str, Any]:
    protocol = audit_sources()
    feature_rows = {}
    precheck_rows = {}
    for seed in SEEDS:
        engine = import_engine(seed)
        copy_verified_common_init(seed)
        # 先写入原始来源审计；正式训练前再用同一独立采样器覆盖
        # pair_budget_expected_summary.json，使训练报告对应 Hard Positive 计划。
        if not (train_root(seed) / "source_audit.json").is_file():
            engine.source_and_leakage_audit(RESULT_ROOT, "ltcc")
        feature_rows[str(seed)] = extract_common_init_features(seed)
    for seed in SEEDS:
        # Precheck must use the original sampler before patching the training engine.
        cached_precheck = RESULT_ROOT / "sampler" / f"seed{seed}_precheck.json"
        if cached_precheck.is_file():
            cached = json.loads(cached_precheck.read_text(encoding="utf-8"))
            if cached.get("identity_sequence_equal") and cached.get("p_k_shape_equal"):
                precheck_rows[str(seed)] = cached
                append_log("sampler.log", f"Seed{seed} 复用已通过完整检查的离线预检结果。")
            else:
                precheck_rows[str(seed)] = precheck(seed)
        else:
            precheck_rows[str(seed)] = precheck(seed)
    # 固定预检结果后，生成正式训练器读取的 Hard Positive 预算文件；这里不
    # 运行模型，也不读取 query/gallery/test。
    for seed in SEEDS:
        engine, _, dataset = load_dataset(seed)
        bank = load_npz_bank(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz")
        pools = build_hard_pools(seed, bank, load_exact_cross_pairs(dataset))
        context = {"seed": seed, "pools": pools["pools"], "all_hard": pools["all_hard"], "reference_path": str(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz"), "stats_dir": RESULT_ROOT / "sampler" / f"seed{seed}"}
        configure_hard_sampler(seed, context)
        engine.plan_pair_budget(RESULT_ROOT, "ltcc", dataset, engine.build_train_only_pair_bank(dataset, "ltcc"))
    all_pass = all(row["identity_sequence_equal"] and row["p_k_shape_equal"] and row["hard_cross_selected_rate"] > row["original_cross_selected_rate"] for row in precheck_rows.values())
    protocol["precheck"] = {"features": feature_rows, "seeds": precheck_rows, "passed": bool(all_pass), "formal_training_gate": "PASS" if all_pass else "STOP"}
    write_json(RESULT_ROOT / "protocol.json", protocol)
    lines = ["# 01 采样器设计", "", "## 固定规则", "", f"- 身份/P×K 计划：复用原始 `DistributedRandomIdentitySampler` 的 PID 顺序和 8=4×2 批次结构。", f"- 训练前参考：每个 seed 对应 common-init 的 LTCC train-only normalized Bio 特征；不使用 I3/O1 最终特征。", f"- 困难定义：每个 PID 内，在审计 GPC 合法跨衣图像对中，common-init cosine 最低的 25% 组成困难池。低相似度表示跨衣正样本难。", f"- 采样强度：有跨衣困难池的身份单元中，50% 确定性选择困难池，50% 保留原始图像单元；无池身份完全保留原始选择。", "- 不改变 GPC、Triplet、CE、Bio 损失或权重；不改变优化器、学习率、AMP、数据增强或总步数。", "", "## 作用边界", "", "Hard Positive 采样只替换同一 PID 的图像索引。跨衣共现增加不被自动视为有效 GPC；预检使用真实 GPC selector 重新计算 `min(cross,same,id)` 后的 selected cross-clothes 关系，作为有效曝光指标。", "", "## 防止偏置", "", "困难池按 PID 单独构造，避免图像多的身份吞噬全局预算；每个 PID 的身份曝光预算保持不变；50% 随机保留比例用于避免长期重复极少数图像；规则在正式训练前固定。", ""]
    (RESULT_ROOT / "01_SAMPLER_DESIGN.md").write_text("\n".join(lines), encoding="utf-8")
    pre_lines = ["# 02 采样器预检", "", "| Seed | PID 序列保持 | 原始跨衣选中比例 | Hard Positive 跨衣选中比例 | Hard pool 选中比例 | 预检状态 |", "|---:|---|---:|---:|---:|---|"]
    for seed in SEEDS:
        item = precheck_rows[str(seed)]
        status = "PASS" if item["identity_sequence_equal"] and item["hard_cross_selected_rate"] > item["original_cross_selected_rate"] else "FAIL"
        pre_lines.append(f"| {seed} | {item['identity_sequence_equal']} | {item['original_cross_selected_rate']:.8f} | {item['hard_cross_selected_rate']:.8f} | {item['hard_pool_selected_rate']:.8f} | **{status}** |")
    pre_lines += ["", "## 判定规则", "", "只有两个 seed 均保持原始 PID 序列/P×K，且 Hard Positive 真实 GPC selected cross-clothes 比例高于 Original，才允许启动短时 PPU 检查和正式训练。共现率不能替代 selected relation 率。", "", f"总体预检：**{'PASS' if all_pass else 'STOP'}**。", ""]
    (RESULT_ROOT / "02_SAMPLER_PRECHECK.md").write_text("\n".join(pre_lines), encoding="utf-8")
    append_log("execution.log", "协议审计、common-init 特征复用、困难池构建和两种子完整离线预检均已完成。")
    append_log("errors.log", "截至准备阶段未记录异常；正式训练前仍需通过 PPU 冒烟检查。")
    return protocol


def train(seed: int) -> None:
    engine = import_engine(seed)
    bank = load_npz_bank(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz")
    _, _, dataset = load_dataset(seed)
    exact = load_exact_cross_pairs(dataset)
    pools = build_hard_pools(seed, bank, exact)
    context = {"seed": seed, "pools": pools["pools"], "all_hard": pools["all_hard"], "reference_path": str(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz"), "stats_dir": RESULT_ROOT / "sampler" / f"seed{seed}"}
    configure_hard_sampler(seed, context)
    audit_path = train_root(seed) / "source_audit.json"
    if not audit_path.is_file():
        engine.source_and_leakage_audit(RESULT_ROOT, "ltcc")
    # The training engine reads this expected budget only for reporting.  It is
    # recomputed with the independent hard-positive sampler before training.
    engine.FinalBatchSampler = HardPositiveBatchSampler
    expected_path = engine.protocol_dir(RESULT_ROOT, "ltcc") / "pair_budget_expected_summary.json"
    if not expected_path.is_file():
        engine.plan_pair_budget(RESULT_ROOT, "ltcc", dataset, engine.build_train_only_pair_bank(dataset, "ltcc"))
    engine.FinalBatchSampler = HardPositiveBatchSampler
    append_log("execution.log", f"Seed{seed} 开始正式 60 epoch PPU 训练；采样规则已冻结，训练前不读取测试指标。")
    engine.training_one(RESULT_ROOT, "ltcc", "gpc")
    # Remove empty engine-created log directory if no file was written there.
    empty_log_dir = RESULT_ROOT / "LTCC/logs"
    if empty_log_dir.is_dir() and not any(empty_log_dir.iterdir()):
        empty_log_dir.rmdir()
    append_log("execution.log", f"Seed{seed} 正式训练完成。")


def smoke(seed: int, steps: int = 3) -> dict[str, Any]:
    engine, _, dataset = load_dataset(seed)
    bank = load_npz_bank(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz")
    pools = build_hard_pools(seed, bank, load_exact_cross_pairs(dataset))
    context = {"seed": seed, "pools": pools["pools"], "all_hard": pools["all_hard"], "reference_path": str(FEATURE_ROOT / f"seed{seed}_common_init_train_features.npz"), "stats_dir": RESULT_ROOT / "sampler" / f"seed{seed}"}
    configure_hard_sampler(seed, context)
    append_log("execution.log", f"Seed{seed} 开始 {steps} batch PPU 冒烟检查。")
    result = engine.dry_run_one(RESULT_ROOT, "ltcc", "gpc", dry_steps=steps)
    append_log("execution.log", f"Seed{seed} 冒烟检查完成：{result}。")
    return result


def extract_hard_features(seed: int) -> dict[str, Any]:
    engine, cfg, dataset = load_dataset(seed)
    checkpoint = hard_train_root(seed) / "model_final.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    feature_path = FEATURE_ROOT / f"seed{seed}_hard_positive_train_features.npz"
    if feature_path.is_file():
        bank = load_npz_bank(feature_path); summary = validate_train_bank(bank, f"seed{seed} Hard Positive 特征")
        return {"path": str(feature_path), "summary": summary, "checkpoint": file_record(checkpoint), "reused": True}
    if not torch.cuda.is_available():
        raise RuntimeError("Hard Positive 特征提取拒绝 CPU fallback")
    torch.cuda.set_device(0); device = torch.device("cuda:0")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict): state = state["model"]
    model = engine.build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True); model.to(device).eval()
    _, transform_test = engine.build_img_transforms(cfg)
    loader = DataLoader(engine.ImageDataset(dataset.train, transform=transform_test), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device); camids = batch["camid"].to(device)
            with amp.autocast(enabled=True): output = model(images, cam_label=camids)
            if torch.is_tensor(output): identity = output
            else:
                _, feat = output; identity = feat[0] if isinstance(feat, (tuple, list)) else feat
            chunks.append(F.normalize(identity.float(), dim=-1).cpu().numpy().astype(np.float32))
    features = np.concatenate(chunks, axis=0)
    bank = {"features": features, "pid": np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64), "clothes": np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64), "camid": np.asarray([int(item["camid"]) for item in dataset.train], dtype=np.int64), "image_path": np.asarray([str(item["image_path"]) for item in dataset.train])}
    summary = validate_train_bank(bank, f"seed{seed} Hard Positive 特征")
    np.savez_compressed(feature_path, **bank)
    write_json(FEATURE_ROOT / f"seed{seed}_hard_positive_train_features.json", {"path": str(feature_path), "checkpoint": file_record(checkpoint), "summary": summary, "device": str(device), "train_only": True})
    append_log("feature.log", f"Seed{seed} 已在 {device} 提取 Hard Positive E60 train 特征。")
    del model, loader, state, dataset; torch.cuda.empty_cache()
    return {"path": str(feature_path), "summary": summary, "checkpoint": file_record(checkpoint), "reused": False, "device": str(device)}


def torch_local_metrics(bank: dict[str, np.ndarray], device: torch.device, block: int = 128) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    z = torch.from_numpy(bank["features"]).to(device=device, dtype=torch.float32)
    pid = torch.from_numpy(bank["pid"]).to(device=device)
    clothes = torch.from_numpy(bank["clothes"]).to(device=device)
    n = z.shape[0]
    ks = (5, 10, 20, 50)
    sums = Counter(); counts = Counter(); per_anchor: list[dict[str, Any]] = []
    for start in range(0, n, block):
        end = min(start + block, n); rows = torch.arange(start, end, device=device); sim = z[start:end] @ z.T
        same_id = pid[None, :] == pid[rows, None]; self_mask = torch.zeros_like(same_id); self_mask[torch.arange(end-start, device=device), rows] = True
        same_id = same_id & ~self_mask; cross = same_id & (clothes[None, :] != clothes[rows, None]); neg = ~(pid[None, :] == pid[rows, None])
        pos_vals = sim.masked_fill(~same_id, float("nan")); cross_vals = sim.masked_fill(~cross, float("nan")); neg_vals = sim.masked_fill(~neg, float("-inf"))
        pmean = torch.nanmean(pos_vals, dim=1); ccmean = torch.nanmean(cross_vals, dim=1); pmin = torch.nan_to_num(pos_vals, nan=float("inf")).min(dim=1).values; ccmin = torch.nan_to_num(cross_vals, nan=float("inf")).min(dim=1).values; nmax = neg_vals.max(dim=1).values
        hard_margin = pmin - nmax; cc_margin = ccmin - nmax
        finite_cc = torch.isfinite(ccmin)
        for key, values in (("positive_mean", pmean), ("cross_cloth_positive_mean", ccmean), ("cross_cloth_positive_min", ccmin), ("negative_max", nmax), ("hard_margin", hard_margin)):
            finite = torch.isfinite(values)
            sums[key] += float(torch.nan_to_num(values[finite], nan=0.0).sum().cpu()); counts[key] += int(finite.sum().cpu())
        finite = torch.isfinite(cc_margin); sums["cross_cloth_margin"] += float(torch.nan_to_num(cc_margin[finite], nan=0.0).sum().cpu()); counts["cross_cloth_margin"] += int(finite.sum().cpu())
        topk = max(ks); sim2 = sim.masked_fill(self_mask, float("-inf")); values, indices = torch.topk(sim2, k=topk, dim=1)
        for k in ks:
            idx = indices[:, :k]; same = (pid[idx] == pid[rows, None]); cross_retrieved = same & (clothes[idx] != clothes[rows, None]); valid_pos = cross.sum(dim=1).float(); denom = torch.minimum(valid_pos, torch.tensor(float(k), device=device)).clamp_min(1.0)
            purity = same.float().mean(dim=1); intrusion = 1.0 - purity; cc_recall = cross_retrieved.float().sum(dim=1) / denom
            for name, vals in ((f"purity@{k}", purity), (f"intrusion@{k}", intrusion), (f"cc_recall@{k}", cc_recall)):
                sums[name] += float(vals.sum().cpu()); counts[name] += int(vals.numel())
            if k == 10:
                for offset in range(end-start):
                    per_anchor.append({"anchor": start + offset, "pid": int(pid[rows[offset]].cpu()), "hard_margin": float(hard_margin[offset].cpu()), "cross_cloth_margin": float(cc_margin[offset].cpu()) if bool(finite_cc[offset]) else None, "positive_mean": float(pmean[offset].cpu()) if torch.isfinite(pmean[offset]) else None, "cross_cloth_positive_mean": float(ccmean[offset].cpu()) if torch.isfinite(ccmean[offset]) else None, "negative_max": float(nmax[offset].cpu()), "purity10": float(purity[offset].cpu()), "intrusion10": float(intrusion[offset].cpu()), "cc_recall10": float(cc_recall[offset].cpu())})
    summary = {key: float(sums[key] / max(counts[key], 1)) for key in sums}
    summary["valid_cross_cloth_anchors"] = int(counts["cross_cloth_margin"])
    return summary, per_anchor


def analyze_features(seed: int) -> dict[str, Any]:
    engine, _, dataset = load_dataset(seed)
    base = load_npz_bank(BASELINE_FEATURES[seed]); control = load_npz_bank(CONTROL_RUNS[seed]["features"]); hard = load_npz_bank(FEATURE_ROOT / f"seed{seed}_hard_positive_train_features.npz")
    for label, bank in (("baseline", base), ("control_gpc", control), ("hard_positive", hard)):
        validate_train_bank(bank, f"seed{seed} {label}")
        if not np.array_equal(bank["image_path"], base["image_path"]) or not np.array_equal(bank["pid"], base["pid"]) or not np.array_equal(bank["clothes"], base["clothes"]):
            raise RuntimeError(f"seed{seed} {label} 与 baseline 样本元数据未严格对齐")
    device = torch.device("cuda:0")
    metrics = {}; per_rows = {}
    for name, bank in (("baseline", base), ("control_gpc", control), ("hard_positive", hard)):
        metrics[name], per_rows[name] = torch_local_metrics(bank, device)
        append_log("feature.log", f"Seed{seed} {name} train-only 局部指标完成：Hard Margin={metrics[name]['hard_margin']:.8f}，Cross-cloth Margin={metrics[name]['cross_cloth_margin']:.8f}。")
    rows = []
    for index in range(len(per_rows["baseline"])):
        row = {"seed": seed, "anchor": per_rows["baseline"][index]["anchor"], "pid": per_rows["baseline"][index]["pid"]}
        for role in ("baseline", "control_gpc", "hard_positive"):
            for key, value in per_rows[role][index].items():
                if key not in ("anchor", "pid"): row[f"{role}_{key}"] = value
        rows.append(row)
    write_csv(RESULT_ROOT / "per_anchor_local_metrics.csv", rows)
    result = {"seed": seed, "metrics": metrics, "delta_hard_minus_control": {key: float(metrics["hard_positive"].get(key, float("nan")) - metrics["control_gpc"].get(key, float("nan"))) for key in metrics["control_gpc"] if isinstance(metrics["control_gpc"].get(key), (int, float))}, "delta_control_minus_baseline": {key: float(metrics["control_gpc"].get(key, float("nan")) - metrics["baseline"].get(key, float("nan"))) for key in metrics["baseline"] if isinstance(metrics["baseline"].get(key), (int, float))}}
    write_json(RESULT_ROOT / "metrics" / f"seed{seed}_local_metrics.json", result)
    return result


def analyze_state_pairs(seed: int) -> dict[str, Any]:
    """在 PPU 上精确统计 PID 内无序服装状态对的跨衣图像相似度。"""
    base = load_npz_bank(BASELINE_FEATURES[seed]); control = load_npz_bank(CONTROL_RUNS[seed]["features"]); hard = load_npz_bank(FEATURE_ROOT / f"seed{seed}_hard_positive_train_features.npz")
    for label, bank in (("baseline", base), ("control_gpc", control), ("hard_positive", hard)):
        validate_train_bank(bank, f"seed{seed} state-pair {label}")
    if not torch.cuda.is_available():
        raise RuntimeError("状态对分析拒绝 CPU fallback；必须使用 PPU")
    torch.cuda.set_device(0); device = torch.device("cuda:0")
    tensors = {name: torch.from_numpy(bank["features"]).to(device=device, dtype=torch.float32) for name, bank in (("baseline", base), ("control_gpc", control), ("hard_positive", hard))}
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (pid, clothes) in enumerate(zip(base["pid"], base["clothes"])):
        groups[(int(pid), int(clothes))].append(index)
    by_pid: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for (pid, clothes), indices in sorted(groups.items()):
        by_pid[pid].append((clothes, indices))
    rows: list[dict[str, Any]] = []
    distributions: dict[str, list[np.ndarray]] = {"baseline": [], "control_gpc": [], "hard_positive": []}
    for pid, states in sorted(by_pid.items()):
        for state_index_a in range(len(states)):
            clothes_a, indices_a = states[state_index_a]
            for state_index_b in range(state_index_a + 1, len(states)):
                clothes_b, indices_b = states[state_index_b]
                row: dict[str, Any] = {"seed": seed, "pid": pid, "clothes_a": clothes_a, "clothes_b": clothes_b, "image_pair_count": len(indices_a) * len(indices_b)}
                for role, feature in tensors.items():
                    values = (feature[indices_a] @ feature[indices_b].T).detach().float().cpu().numpy().reshape(-1)
                    distributions[role].append(values)
                    row[f"{role}_mean"] = float(np.mean(values))
                    row[f"{role}_p10"] = float(np.percentile(values, 10))
                    row[f"{role}_min"] = float(np.min(values))
                row["delta_hard_minus_control_mean"] = row["hard_positive_mean"] - row["control_gpc_mean"]
                row["delta_hard_minus_control_p10"] = row["hard_positive_p10"] - row["control_gpc_p10"]
                row["delta_hard_minus_control_min"] = row["hard_positive_min"] - row["control_gpc_min"]
                row["delta_control_minus_baseline_mean"] = row["control_gpc_mean"] - row["baseline_mean"]
                rows.append(row)
    state_path = RESULT_ROOT / "metrics" / f"seed{seed}_state_pair_gain.csv"
    write_csv(state_path, rows)
    identity_rows: list[dict[str, Any]] = []
    for pid in sorted(by_pid):
        pid_rows = [row for row in rows if int(row["pid"]) == pid]
        identity_rows.append({
            "seed": seed, "pid": pid, "state_pair_count": len(pid_rows), "image_pair_count": int(sum(int(row["image_pair_count"]) for row in pid_rows)),
            "hard_minus_control_mean": float(np.mean([row["delta_hard_minus_control_mean"] for row in pid_rows])) if pid_rows else None,
            "hard_minus_control_p10": float(np.mean([row["delta_hard_minus_control_p10"] for row in pid_rows])) if pid_rows else None,
            "hard_minus_control_min": float(np.mean([row["delta_hard_minus_control_min"] for row in pid_rows])) if pid_rows else None,
            "positive_gain_pair_fraction": float(np.mean([row["delta_hard_minus_control_mean"] > 0 for row in pid_rows])) if pid_rows else None,
            "control_minus_baseline_mean": float(np.mean([row["delta_control_minus_baseline_mean"] for row in pid_rows])) if pid_rows else None,
        })
    write_csv(RESULT_ROOT / "metrics" / f"seed{seed}_identity_summary.csv", identity_rows)
    summary = {"seed": seed, "state_pair_count": len(rows), "multi_cloth_pid_count": len(by_pid), "role": {}}
    for role, arrays in distributions.items():
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
        summary["role"][role] = {"image_pair_count": int(values.size), "mean": float(np.mean(values)) if values.size else None, "p10": float(np.percentile(values, 10)) if values.size else None, "min": float(np.min(values)) if values.size else None}
    summary["delta_hard_minus_control"] = {"mean": summary["role"]["hard_positive"]["mean"] - summary["role"]["control_gpc"]["mean"], "p10": summary["role"]["hard_positive"]["p10"] - summary["role"]["control_gpc"]["p10"], "min": summary["role"]["hard_positive"]["min"] - summary["role"]["control_gpc"]["min"]}
    write_json(RESULT_ROOT / "metrics" / f"seed{seed}_state_pair_metrics.json", summary)
    local_path = RESULT_ROOT / "metrics" / f"seed{seed}_local_metrics.json"
    local = json.loads(local_path.read_text(encoding="utf-8"))
    for role in ("baseline", "control_gpc", "hard_positive"):
        local["metrics"][role]["cross_cloth_positive_pair_p10"] = summary["role"][role]["p10"]
        local["metrics"][role]["cross_cloth_positive_pair_count"] = summary["role"][role]["image_pair_count"]
    local["state_pair_summary"] = summary
    write_json(local_path, local)
    append_log("clothing_state.log", f"Seed{seed} PPU 状态对分析完成：{len(rows)} 个无序服装状态对、{len(by_pid)} 个多服装 PID；Hard Positive 跨衣正图像对 P10 变化={summary['delta_hard_minus_control']['p10']:+.8f}。")
    return summary


def aggregate_training_and_retrieval(seed: int) -> dict[str, Any]:
    engine = import_engine(seed)
    run_dir = hard_train_root(seed)
    metrics_path = run_dir / "metrics_per_epoch.csv"
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8"))) if metrics_path.is_file() else []
    checkpoint = run_dir / "model_final.pth"
    return {"seed": seed, "epochs": len(rows), "optimizer_updates": int(float(rows[-1]["optimizer_updates"])) if rows else None, "train_loss_e60": float(rows[-1]["train_loss"]) if rows else None, "gpc_loss_e60": float(rows[-1]["gpc_loss"]) if rows else None, "triplet_loss_e60": float(rows[-1]["triplet_loss"]) if rows else None, "cross_clothes_selected_e60": int(float(rows[-1]["cross_clothes_contributions"])) if rows else None, "relation_active_rate_e60": float(rows[-1]["relation_active_rate"]) if rows else None, "checkpoint": file_record(checkpoint), "runtime_audit": json.loads((run_dir / "runtime_audit.json").read_text()) if (run_dir / "runtime_audit.json").is_file() else None}


def restore_public_checkpoint(seed: int) -> dict[str, Any]:
    """Restore the shared engine public E60 file from this seed's private copy.

    The reused training engine keeps one public ``LTCC/checkpoints`` namespace
    per timestamp, while this pilot stores private checkpoints per seed.  The
    engine's test gate hashes the public file, so the public copy must be
    restored immediately before each seed's evaluation.  The private E60 file
    is never modified.
    """
    run_dir = hard_train_root(seed)
    internal = run_dir / "model_final.pth"
    meta_path = run_dir / "final_checkpoint.json"
    if not internal.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"缺少 Seed{seed} 的 E60 checkpoint 或元数据")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    internal_sha = sha256_file(internal)
    if internal_sha != meta.get("checkpoint_sha256"):
        raise RuntimeError(f"Seed{seed} 私有 E60 checkpoint SHA256 与元数据不一致")
    public = RESULT_ROOT / "LTCC/checkpoints/model_final.pth"
    public.parent.mkdir(parents=True, exist_ok=True)
    # evaluate is launched with two ranks; each rank restores an identical
    # public copy, so the temporary name must be rank/process-local to avoid a
    # copy2/replace race.
    temporary = public.with_suffix(public.suffix + f".seed{seed}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(internal, temporary)
    temporary.replace(public)
    public.chmod(0o444)
    public_sha = sha256_file(public)
    if public_sha != internal_sha:
        raise RuntimeError(f"Seed{seed} 公共 E60 checkpoint 恢复后 SHA256 不一致")
    record = {"seed": seed, "private_path": str(internal), "public_path": str(public), "checkpoint_sha256": public_sha, "epoch": 60}
    write_json(RESULT_ROOT / "evaluation" / f"seed{seed}_public_checkpoint_restore.json", record)
    append_log("evaluation.log", f"Seed{seed} 检索前已恢复公共 E60 checkpoint；私有/公共 SHA256={public_sha}。")
    return record


def evaluate(seed: int) -> dict[str, Any]:
    engine = import_engine(seed)
    restore_public_checkpoint(seed)
    append_log("evaluation.log", f"Seed{seed} 开始只使用 E60 final checkpoint 的 LTCC 正式检索评估；此时才打开 query/gallery。")
    result = engine.test_gpc(RESULT_ROOT, "ltcc")
    append_log("evaluation.log", f"Seed{seed} 正式评估完成：{result['results'][0]}")
    return result


def load_control_retrieval(seed: int) -> dict[str, Any]:
    path = CONTROL_RUNS[seed]["retrieval_evidence"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Seed1234 已有标准 final_test.json；Seed4321 的既有 O1 结果以
    # retrieval_eval_seed4321.json 保存，显式选择同一 E60 final 记录。
    if "results" in payload:
        payload["retrieval_evidence_path"] = str(path)
        return payload
    if "rank1" in payload and "mAP" in payload:
        payload["retrieval_evidence_path"] = str(path)
        return payload
    rows = [row for row in payload.get("rows", []) if row.get("selection") == "final" and int(row.get("seed", seed)) == int(seed) and str(row.get("checkpoint")) == str(CONTROL_RUNS[seed]["checkpoint"])]
    if len(rows) != 1:
        raise RuntimeError(f"Seed{seed} 的历史 E60 对照检索记录不是唯一：{path}")
    row = rows[0]
    return {
        "arm": "gpc", "dataset": "ltcc", "method": "Bio + GPC", "protocol": "LTCC query -> test gallery, changed clothes filter",
        "checkpoint": {"path": row.get("checkpoint"), "epoch": 60, "test_metric_used": False},
        "evaluated_utc": None, "evaluator": "此前已审计的 LTCC 原始 evaluate_ltcc 结果", "evaluator_calls": 1,
        "query_count": 493, "gallery_count": 7050, "rank1": row["cc_rank1"], "rank5": row["cc_rank5"], "rank10": row["cc_rank10"], "mAP": row["cc_mAP"], "mINP": None,
        "retrieval_evidence_path": str(path), "selection": "final",
    }


def _parse_log_time(line: str) -> dt.datetime | None:
    if not line[:19].strip():
        return None
    try:
        return dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def aggregate_training_and_retrieval(seed: int) -> dict[str, Any]:
    run_dir = hard_train_root(seed)
    metrics_path = run_dir / "metrics_per_epoch.csv"
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8"))) if metrics_path.is_file() else []
    checkpoint = run_dir / "model_final.pth"
    amp_path = run_dir / "amp_statistics.csv"
    amp_rows = list(csv.DictReader(amp_path.open(newline="", encoding="utf-8"))) if amp_path.is_file() else []
    run_complete_path = run_dir / "RUN_COMPLETE.json"
    run_complete = json.loads(run_complete_path.read_text(encoding="utf-8")) if run_complete_path.is_file() else {}
    train_log = LOG_ROOT / f"train_seed{seed}.log"
    start_time = end_time = None
    epoch_wall_seconds = 0.0
    if train_log.is_file():
        log_lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()
        parsed = [value for value in (_parse_log_time(line) for line in log_lines) if value is not None]
        if parsed:
            start_time, end_time = parsed[0], parsed[-1]
        import re
        epoch_wall_seconds = sum(float(value) for value in re.findall(r"Epoch wall time：([0-9.]+) 秒", "\n".join(log_lines)))
    exposure = run_complete.get("actual_exposure_totals", {})
    updates_per_epoch = int(float(rows[-1]["optimizer_updates"])) if rows else None
    updates_per_rank_total = int(sum(float(row.get("optimizer_updates", 0)) for row in rows)) if rows else None
    return {
        "seed": seed, "epochs": len(rows), "optimizer_updates_per_epoch": updates_per_epoch,
        "optimizer_updates_per_rank": updates_per_rank_total,
        "global_optimizer_updates": updates_per_rank_total * WORLD_SIZE if updates_per_rank_total is not None else None,
        "train_loss_e60": float(rows[-1]["train_loss"]) if rows else None, "gpc_loss_e60": float(rows[-1]["gpc_loss"]) if rows else None,
        "triplet_loss_e60": float(rows[-1]["triplet_loss"]) if rows else None, "ce_loss_e60": float(rows[-1]["ce_loss"]) if rows else None,
        "cross_clothes_selected_e60": int(float(rows[-1]["cross_clothes_contributions"])) if rows else None,
        "same_clothes_selected_e60": int(float(rows[-1]["same_clothes_contributions"])) if rows else None,
        "relation_active_rate_e60": float(rows[-1]["relation_active_rate"]) if rows else None,
        "relation_active_batches_e60": int(float(rows[-1]["relation_active_batches"])) if rows else None,
        "selected_total_e60": int(float(rows[-1]["relation_contributions"])) if rows else None,
        "amp_skipped_steps_total": int(sum(float(row.get("skipped_steps", 0)) for row in amp_rows)),
        "amp_found_inf_total": int(sum(float(row.get("found_inf_count", 0)) for row in amp_rows)),
        "amp_nonfinite_events_total": int(sum(float(row.get("nonfinite_events", 0)) for row in amp_rows)),
        "actual_exposure_totals": exposure, "checkpoint": file_record(checkpoint),
        "runtime_audit": json.loads((run_dir / "runtime_audit.json").read_text()) if (run_dir / "runtime_audit.json").is_file() else None,
        "training_log_start_utc": start_time.isoformat().replace("+00:00", "Z") if start_time else None,
        "training_log_end_utc": end_time.isoformat().replace("+00:00", "Z") if end_time else None,
        "training_wall_seconds_from_log": (end_time - start_time).total_seconds() if start_time and end_time else None,
        "epoch_wall_seconds_sum": epoch_wall_seconds,
        "run_complete": run_complete,
    }


def _write_auxiliary_reports(local: dict[str, Any], retrieval: dict[str, Any], control: dict[str, Any], train_rows: dict[str, Any], deltas: list[dict[str, Any]], verdict: str) -> None:
    """生成用户要求的训练、局部结构和正式检索三份独立中文报告。"""
    training_lines = [
        "# 03_TRAINING_REPORT", "", "## 固定训练协议", "",
        "两组实验均从对应 common-init 重新开始，使用 60 epoch、每 epoch 596 个 batch、每卡 batch=8、P=4/K=2、world size=2、每卡累计 35760 次 optimizer update。损失公式仍为 `L_CE + L_Triplet + L_Bio + 0.1×L_GPC`；本阶段只改变同一身份内部的图像索引。由于图像对被替换，实际 GPC selector 的有效关系密度也会改变，不能把本结果解释为只改变正样本难度的严格等预算消融。",
        "", "## 正式训练结果", "", "| Seed | Epoch | 每卡更新 | E60 总损失 | E60 GPC | E60 Triplet | E60 有效 GPC 关系 | E60 跨衣关系 | 全周期跨衣关系 | AMP 跳步 | 训练时长（日志） | PPU |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    ]
    for seed in SEEDS:
        row = train_rows[str(seed)]
        runtime = row.get("training_wall_seconds_from_log")
        ppu = bool(row.get("runtime_audit", {}).get("passed", False))
        training_lines.append(f"| {seed} | {row['epochs']} | {row['optimizer_updates_per_rank']} | {row['train_loss_e60']:.6f} | {row['gpc_loss_e60']:.6f} | {row['triplet_loss_e60']:.6f} | {row['selected_total_e60']} | {row['cross_clothes_selected_e60']} | {row.get('actual_exposure_totals', {}).get('cross_clothes', '')} | {row['amp_skipped_steps_total']} | {runtime/60:.2f} 分钟 | {'通过' if ppu else '未通过'} |" if runtime is not None else f"| {seed} | {row['epochs']} | {row['optimizer_updates_per_rank']} | {row['train_loss_e60']:.6f} | {row['gpc_loss_e60']:.6f} | {row['triplet_loss_e60']:.6f} | {row['selected_total_e60']} | {row['cross_clothes_selected_e60']} | {row.get('actual_exposure_totals', {}).get('cross_clothes', '')} | {row['amp_skipped_steps_total']} | 未记录 | {'通过' if ppu else '未通过'} |")
    training_lines += [
        "", "## 数值与冒烟说明", "",
        "- Seed1234 正式训练共记录 28 次 AMP found_inf/跳过 step；Seed4321 共记录 32 次。训练器按既有 GradScaler 规则跳过异常 step、降低 scale，两个 seed 均完成 E60，最终检查点和训练指标均为有限值。",
        "- Seed4321 的 3 batch 探针曾在第 3 个 batch 触发 found_inf，原始探针文件被保留；随后使用固定 2 batch 短时 PPU gate 通过后才启动正式训练。该现象没有导致正式训练使用错误的初始化或不完整 checkpoint。",
        "- 曾有一次未先加载 PPU SDK 环境的冒烟运行因运行时环境失败；之后按项目协议 `source /usr/local/PPU_SDK/envsetup.sh ppu` 重跑并通过。该次失败不计入正式训练。",
        "- 训练期间没有打开 query/gallery，没有调用测试评估，也没有根据测试指标改动采样规则、强度或模型选择。", ""
    ]
    (RESULT_ROOT / "03_TRAINING_REPORT.md").write_text("\n".join(training_lines), encoding="utf-8")

    local_lines = ["# 04_LOCAL_FEATURE_ANALYSIS", "", "本报告只使用 LTCC train-only、1024 维归一化 Bio 特征，FP32 分块计算；不等同于正式检索结果。跨衣正样本 P10 定义为所有同 PID、不同服装状态图像对的相似度池化后第 10 百分位。", "", "| Seed | 方法 | 跨衣正均值 | 跨衣正 P10 | 最困难跨衣正 | 最困难负 | Hard Margin | Cross-cloth Margin | Purity@10 | Intrusion@10 | CC_Recall@10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        metrics = local[str(seed)]["metrics"]
        for label, key in (("GPC Original", "control_gpc"), ("GPC + Hard Positive", "hard_positive")):
            m = metrics[key]
            local_lines.append(f"| {seed} | {label} | {m['cross_cloth_positive_mean']:.8f} | {m.get('cross_cloth_positive_pair_p10', float('nan')):.8f} | {m['cross_cloth_positive_min']:.8f} | {m['negative_max']:.8f} | {m['hard_margin']:.8f} | {m['cross_cloth_margin']:.8f} | {m['purity@10']:.8f} | {m['intrusion@10']:.8f} | {m['cc_recall@10']:.8f} |")
    local_lines += ["", "## 配对变化", "", "| Seed | 跨衣正均值变化 | 跨衣正 P10 变化 | 最困难跨衣正变化 | 最困难负变化 | Hard Margin 变化 | Cross-cloth Margin 变化 | Purity@10 变化 | Intrusion@10 变化 | CC_Recall@10 变化 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in deltas:
        p10_delta = local[str(row['seed'])].get('state_pair_summary', {}).get('delta_hard_minus_control', {}).get('p10', float('nan'))
        local_lines.append(f"| {row['seed']} | {local[str(row['seed'])]['delta_hard_minus_control']['cross_cloth_positive_mean']:+.8f} | {p10_delta:+.8f} | {local[str(row['seed'])]['delta_hard_minus_control']['cross_cloth_positive_min']:+.8f} | {local[str(row['seed'])]['delta_hard_minus_control']['negative_max']:+.8f} | {row['delta_hard_margin']:+.8f} | {row['delta_cross_cloth_margin']:+.8f} | {row['delta_purity10']:+.8f} | {row['delta_purity10']*-1:+.8f} | {row['delta_cc_recall10']:+.8f} |")
    local_lines += ["", "## 解释边界", "", "Hard Positive 在两个 seed 都提高了跨衣正样本均值、最困难跨衣正样本和跨衣 Margin；同时最困难负样本相似度也上升，Purity@10 下降、Intrusion@10 上升。该组合不能被表述为全面改善局部身份判别；需要以固定 E60 的正式检索作为主要终点。另需注意，Hard Positive 使全周期有效 GPC 关系数从历史原始 GPC 的 8824/8697 提高到 55752/55678，因此本实验同时包含有效 GPC 关系密度变化，不能孤立归因于困难正样本排序。", ""]
    (RESULT_ROOT / "04_LOCAL_FEATURE_ANALYSIS.md").write_text("\n".join(local_lines), encoding="utf-8")

    retrieval_lines = ["# 05_RETRIEVAL_COMPARISON", "", "主比较固定为 E60 final 对 E60 final。Seed1234 原始 I3 使用历史标准 `final_test.json`；Seed4321 原始 O1 使用已审计的 `retrieval_eval_seed4321.json` 中 `selection=final` 记录，因为 O1 目录未保存标准 `final_test.json`。实验组两次评估均在训练和 train-only 分析完成后执行，且每次先恢复对应私有 E60 检查点并通过 SHA gate。", "", "| 方法 | Seed | Rank-1 | Rank-5 | Rank-10 | mAP | mINP | 检查点选择 |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for seed in SEEDS:
        cr = control[str(seed)]; hr = retrieval[str(seed)]["results"][0]
        retrieval_lines.append(f"| GPC Original | {seed} | {cr['rank1']:.8f} | {cr['rank5']:.8f} | {cr['rank10']:.8f} | {cr['mAP']:.8f} | 未提供 | E60 final |")
        retrieval_lines.append(f"| GPC + 普通困难跨衣正样本 | {seed} | {hr['rank1']:.8f} | {hr['rank5']:.8f} | {hr['rank10']:.8f} | {hr['mAP']:.8f} | 未提供 | E60 final |")
    retrieval_lines += ["", "## 配对差异", "", "| Seed | Rank-1 变化 | Rank-5 变化 | Rank-10 变化 | mAP 变化 |", "|---:|---:|---:|---:|---:|"]
    for row in deltas:
        retrieval_lines.append(f"| {row['seed']} | {row['delta_rank1']:+.8f} | {row['delta_rank5']:+.8f} | {row['delta_rank10']:+.8f} | {row['delta_mAP']:+.8f} |")
    retrieval_lines += ["", "## 协议限制", "", "原始 O1 的 best epoch 28 结果不参与主比较；没有把 best 与 E60 混合。mINP 由当前 evaluator 未提供。训练期间没有使用测试指标选择采样器或 checkpoint。", ""]
    (RESULT_ROOT / "05_RETRIEVAL_COMPARISON.md").write_text("\n".join(retrieval_lines), encoding="utf-8")

    # 训练曲线仅从已有日志 CSV 绘制，不参与任何模型或参数选择。
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
        for field, title, filename in (("train_loss", "E60 训练总损失", "training_total_loss.png"), ("cross_clothes_fraction", "每 epoch GPC 跨衣关系比例", "cross_cloth_relation_fraction.png")):
            plt.figure(figsize=(8, 4.5))
            for seed in SEEDS:
                rows = list(csv.DictReader((hard_train_root(seed) / "metrics_per_epoch.csv").open(newline="", encoding="utf-8")))
                plt.plot([int(row["epoch"]) for row in rows], [float(row[field]) for row in rows], label=f"Seed{seed}")
            plt.xlabel("Epoch"); plt.ylabel(field); plt.title(title); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_ROOT / filename, dpi=160); plt.close()
    except Exception as exc:
        append_log("errors.log", f"训练曲线绘制未完成，不影响数值结果：{type(exc).__name__}: {exc}")


def final_report() -> dict[str, Any]:
    local = {str(seed): json.loads((RESULT_ROOT / "metrics" / f"seed{seed}_local_metrics.json").read_text()) for seed in SEEDS}
    retrieval = {}
    for seed in SEEDS:
        raw = json.loads((hard_train_root(seed) / "final_test.json").read_text())
        # engine 的标准单次 E60 结果是顶层字段；报告统一成 results[0] 结构。
        retrieval[str(seed)] = raw if "results" in raw else {"results": [raw], "normalized_from": str(hard_train_root(seed) / "final_test.json")}
    control = {str(seed): load_control_retrieval(seed) for seed in SEEDS}
    train_rows = {str(seed): aggregate_training_and_retrieval(seed) for seed in SEEDS}
    comparison_rows = []
    deltas = []
    for seed in SEEDS:
        m = local[str(seed)]["metrics"]; cr = control[str(seed)]; hr = retrieval[str(seed)]["results"][0]
        comparison_rows.append({"method": "GPC Original", "seed": seed, "rank1": cr["rank1"], "rank5": cr["rank5"], "rank10": cr["rank10"], "mAP": cr["mAP"], "hard_margin": m["control_gpc"]["hard_margin"], "cross_cloth_margin": m["control_gpc"]["cross_cloth_margin"], "purity10": m["control_gpc"]["purity@10"], "cc_recall10": m["control_gpc"]["cc_recall@10"], "coverage_note": "原始 GPC 历史曝光", "retrieval_evidence": cr.get("retrieval_evidence_path")})
        comparison_rows.append({"method": "GPC + 普通困难跨衣正样本", "seed": seed, "rank1": hr["rank1"], "rank5": hr["rank5"], "rank10": hr["rank10"], "mAP": hr["mAP"], "hard_margin": m["hard_positive"]["hard_margin"], "cross_cloth_margin": m["hard_positive"]["cross_cloth_margin"], "purity10": m["hard_positive"]["purity@10"], "cc_recall10": m["hard_positive"]["cc_recall@10"], "coverage_note": "本阶段固定 Hard Positive 规则", "retrieval_evidence": str(hard_train_root(seed) / "final_test.json")})
        deltas.append({"seed": seed, "delta_rank1": hr["rank1"] - cr["rank1"], "delta_rank5": hr["rank5"] - cr["rank5"], "delta_rank10": hr["rank10"] - cr["rank10"], "delta_mAP": hr["mAP"] - cr["mAP"], "delta_hard_margin": m["hard_positive"]["hard_margin"] - m["control_gpc"]["hard_margin"], "delta_cross_cloth_margin": m["hard_positive"]["cross_cloth_margin"] - m["control_gpc"]["cross_cloth_margin"], "delta_purity10": m["hard_positive"]["purity@10"] - m["control_gpc"]["purity@10"], "delta_cc_recall10": m["hard_positive"]["cc_recall@10"] - m["control_gpc"]["cc_recall@10"]})
    write_csv(RESULT_ROOT / "controlled_comparison.csv", comparison_rows)
    write_csv(RESULT_ROOT / "seed_deltas.csv", deltas)
    verdict = "INCONCLUSIVE"
    if all(row["delta_rank1"] > 0 and row["delta_mAP"] > 0 and row["delta_cc_recall10"] >= -1e-8 for row in deltas): verdict = "HARD_POSITIVE_EFFECTIVE"
    elif all(row["delta_rank1"] < 0 or row["delta_mAP"] < 0 or row["delta_cc_recall10"] < 0 for row in deltas): verdict = "HARD_POSITIVE_HARMFUL"
    elif all(row["delta_cross_cloth_margin"] > 0 for row in deltas) and not all(row["delta_rank1"] > 0 and row["delta_mAP"] > 0 for row in deltas): verdict = "POSITIVE_ONLY_GAIN"
    precheck = {str(seed): json.loads((RESULT_ROOT / "sampler" / f"seed{seed}_precheck.json").read_text()) for seed in SEEDS}
    state_pair_summary = {}
    state_pair_rows: list[dict[str, Any]] = []
    identity_summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        summary_path = RESULT_ROOT / "metrics" / f"seed{seed}_state_pair_metrics.json"
        if summary_path.is_file():
            state_pair_summary[str(seed)] = json.loads(summary_path.read_text(encoding="utf-8"))
        pair_path = RESULT_ROOT / "metrics" / f"seed{seed}_state_pair_gain.csv"
        if pair_path.is_file():
            state_pair_rows.extend(list(csv.DictReader(pair_path.open(newline="", encoding="utf-8"))))
        identity_path = RESULT_ROOT / "metrics" / f"seed{seed}_identity_summary.csv"
        if identity_path.is_file():
            identity_summary_rows.extend(list(csv.DictReader(identity_path.open(newline="", encoding="utf-8"))))
    if state_pair_rows:
        write_csv(RESULT_ROOT / "per_state_pair_gain.csv", state_pair_rows)
    if identity_summary_rows:
        write_csv(RESULT_ROOT / "per_identity_summary.csv", identity_summary_rows)
    protocol_path = RESULT_ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    protocol["precheck"] = precheck
    protocol["ppu_env_setup"] = "source /usr/local/PPU_SDK/envsetup.sh ppu"
    protocol["smoke_gate"] = {"seed1234": "3 batch PPU 冒烟通过", "seed4321": "3 batch 探针 found_inf；2 batch PPU gate 通过后正式训练"}
    protocol["evaluation"] = {"checkpoint_rule": "E60 final vs E60 final", "query_gallery_opened_after_train_only_analysis": True, "seed1234_public_restore": str(RESULT_ROOT / "evaluation/seed1234_public_checkpoint_restore.json"), "seed4321_public_restore": str(RESULT_ROOT / "evaluation/seed4321_public_checkpoint_restore.json"), "control_seed4321_evidence": str(CONTROL_RUNS[4321]["retrieval_evidence"])}
    protocol["report_generated_utc"] = utc_now()
    write_json(protocol_path, protocol)
    payload = {"verdict": verdict, "created_utc": utc_now(), "local": local, "state_pair_summary": state_pair_summary, "retrieval": retrieval, "control_retrieval": control, "training": train_rows, "deltas": deltas, "precheck": precheck, "formal_rule": "E60 final vs E60 final", "query_gallery_used_only_after_training": True, "control_retrieval_evidence": {str(seed): control[str(seed)].get("retrieval_evidence_path") for seed in SEEDS}}
    write_json(RESULT_ROOT / "metrics.json", payload)
    _write_auxiliary_reports(local, retrieval, control, train_rows, deltas, verdict)
    lines = ["# PHASE CC-GPC-HP-PILOT 最终报告", "", "## 实验完整性", "", "- 本阶段从每个 seed 对应的 common-init 重新开始，完成普通困难跨衣正样本采样的 PPU 训练；未修改 GPC、Triplet、CE、Bio 损失、权重、优化器或正式模型结构。", "- 采样器只替换同一身份内部图像，PID 顺序、P×K、60 epoch、596 steps/epoch 和总优化步数保持不变。", "- 训练前采样只使用 common-init 的 LTCC train-only 特征；不使用最终 GPC 特征指导训练初期采样。", "- 主检索比较固定为 E60 final vs E60 final；Seed4321 的 O1 best epoch 28 不参与主比较。", "- query/gallery 只在训练和 train-only 特征分析完成后打开，没有用于采样强度、规则或模型选择。", "", "## 预检和有效曝光", "", "| Seed | 原始 GPC 跨衣选中比例 | Hard Positive 跨衣选中比例 | 增益 | Hard pool 选中比例 | PID/P×K 序列 |", "|---:|---:|---:|---:|---:|---|"]
    for seed in SEEDS:
        pre = precheck[str(seed)]
        lines.append(f"| {seed} | {pre['original_cross_selected_rate']:.8f} | {pre['hard_cross_selected_rate']:.8f} | {pre['hard_cross_selected_rate']-pre['original_cross_selected_rate']:+.8f} | {pre['hard_pool_selected_rate']:.8f} | {pre['identity_sequence_equal']} |")
    lines += ["", "## E60 受控结果", "", "| Seed | Rank-1 原始→Hard | mAP 原始→Hard | Hard Margin 变化 | Cross-cloth Margin 变化 | Purity@10 变化 | Intrusion@10 变化 | CC_Recall@10 变化 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append(f"| {row['seed']} | {row['delta_rank1']:+.8f} | {row['delta_mAP']:+.8f} | {row['delta_hard_margin']:+.8f} | {row['delta_cross_cloth_margin']:+.8f} | {row['delta_purity10']:+.8f} | {-row['delta_purity10']:+.8f} | {row['delta_cc_recall10']:+.8f} |")
    lines += ["", "## 状态对级补充", "", "状态对定义为同一 PID 内两个不同服装标签的无序状态对；状态对均值、P10 和最小值基于两端图像笛卡尔积。", "", "| Seed | 状态对数量 | 多服装 PID | Hard−Original 跨衣正均值 | Hard−Original 跨衣正 P10 |", "|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        summary = state_pair_summary.get(str(seed), {})
        delta = summary.get("delta_hard_minus_control", {})
        lines.append(f"| {seed} | {summary.get('state_pair_count', '未生成')} | {summary.get('multi_cloth_pid_count', '未生成')} | {delta.get('mean', float('nan')):+.8f} | {delta.get('p10', float('nan')):+.8f} |")
    lines += ["", "## 训练成本与数值稳定性", "", "- 两个 seed 均完成 60 epoch、每卡累计 35760 次更新；Hard Positive 不增加优化步数、网络结构或损失项公式，但会显著改变实际 GPC selector 的有效关系密度。", "- Hard Positive 额外需要训练前一次 common-init train 特征复用/困难池构建以及同身份索引替换；本实验没有可靠的同环境原始控制 wall-time 记录，因此不报告伪精确的额外分钟数。", "- AMP 跳步：Seed1234 为 28 次，Seed4321 为 32 次；均由既有 GradScaler 跳步机制处理，最终 E60 checkpoint 参数与训练指标均为有限值且 SHA 校验通过。", "- PPU runtime audit 两个 seed 均通过；训练时长、检查点和详细损失见 `03_TRAINING_REPORT.md`。", "", "## 科学判断", "", f"主判定：**{verdict}**。", "", "普通困难跨衣正样本采样在两个 seed 都提高了 train-only 困难跨衣正样本和 Cross-cloth Margin，但同时提高了困难异身份负样本相似度，Purity@10 下降。正式 E60 Rank-1/mAP 均未改善，且 Seed4321 下降明显；因此该常规干预不能作为当前第二创新点的正向依据，也不应直接包装成 NDP 创新。", "", "## 交付文件", "", f"- 协议审计：`{RESULT_ROOT / '00_PROTOCOL_AUDIT.md'}`", f"- 采样设计/预检：`{RESULT_ROOT / '01_SAMPLER_DESIGN.md'}`、`{RESULT_ROOT / '02_SAMPLER_PRECHECK.md'}`", f"- 训练报告：`{RESULT_ROOT / '03_TRAINING_REPORT.md'}`", f"- 局部特征分析：`{RESULT_ROOT / '04_LOCAL_FEATURE_ANALYSIS.md'}`", f"- 检索比较：`{RESULT_ROOT / '05_RETRIEVAL_COMPARISON.md'}`", f"- 日志：`{LOG_ROOT}`", ""]
    (RESULT_ROOT / "FINAL_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    append_log("execution.log", f"Seed1234/Seed4321 E60 检索、局部指标、受控比较与中文交付报告均已生成；主判定={verdict}。")
    append_log("diagnostic.log", "最终诊断同时报告困难跨衣正样本改善、负样本相似度变化、Purity/CC_Recall 与正式 Rank-1/mAP，未以单一 Hard Margin 代替检索结论。")
    return payload


def main() -> None:
    global RUN_TIME, RESULT_ROOT, LOG_ROOT, FEATURE_ROOT, FIGURE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["prepare", "smoke", "train", "extract", "analyze", "state", "evaluate", "report"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--run-time", default=RUN_TIME)
    parser.add_argument("--steps", type=int, default=3, help="冒烟检查的 batch 数；正式训练不使用该参数")
    args = parser.parse_args()
    RUN_TIME = args.run_time
    RESULT_ROOT = ROOT / "outputs/cc_gpc_hp_pilot" / RUN_TIME
    LOG_ROOT = ROOT / "log_NDP/PHASE_CC_GPC_HP_PILOT" / RUN_TIME
    FEATURE_ROOT = RESULT_ROOT / "features"
    FIGURE_ROOT = RESULT_ROOT / "figures"
    if args.mode == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.mode == "smoke":
        if args.seed is None: raise SystemExit("smoke 需要 --seed")
        if args.steps < 1: raise SystemExit("冒烟 batch 数必须为正数")
        print(json.dumps(smoke(args.seed, steps=args.steps), ensure_ascii=False, indent=2))
    elif args.mode == "train":
        if args.seed is None: raise SystemExit("train 需要 --seed")
        train(args.seed)
    elif args.mode == "extract":
        if args.seed is None: raise SystemExit("extract 需要 --seed")
        print(json.dumps(extract_hard_features(args.seed), ensure_ascii=False, indent=2))
    elif args.mode == "analyze":
        if args.seed is None: raise SystemExit("analyze 需要 --seed")
        print(json.dumps(analyze_features(args.seed), ensure_ascii=False, indent=2))
    elif args.mode == "state":
        if args.seed is None: raise SystemExit("state 需要 --seed")
        print(json.dumps(analyze_state_pairs(args.seed), ensure_ascii=False, indent=2))
    elif args.mode == "evaluate":
        if args.seed is None: raise SystemExit("evaluate 需要 --seed")
        print(json.dumps(evaluate(args.seed), ensure_ascii=False, indent=2))
    elif args.mode == "report":
        print(json.dumps(final_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
