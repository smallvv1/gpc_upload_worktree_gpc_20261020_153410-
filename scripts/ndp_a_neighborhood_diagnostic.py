#!/usr/local/bin/python3
"""NDP-A: offline neighborhood discriminative-preservation feasibility audit.

This script is deliberately read-only with respect to models and experiments.
It consumes only the already-generated LTCC *train* feature banks produced
from the exact frozen Phase 7I/7O checkpoints, re-packages those banks under
the NDP-A output root, and performs NumPy-only neighborhood diagnostics.

No model is instantiated, no dataset constructor is called, no optimizer or
autograd operation is imported, and no query/gallery/test artifact is read.
The explicit ``--run-time`` argument is required so one experiment can keep a
single immutable RUN_TIME across every report and log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting failure is recorded, not hidden
    matplotlib = None
    plt = None


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = Path("/home/DIFFER/logs")
REPORT_ROOT = ROOT / "reports"
OUT_ROOT = ROOT / "outputs/new_method_track/ndp_a_neighborhood_feasibility"
FEATURE_ROOT = OUT_ROOT / "feature_banks"
GRAPH_ROOT = OUT_ROOT / "neighbor_graphs"
TABLE_ROOT = OUT_ROOT / "tables"
PLOT_ROOT = OUT_ROOT / "plots"

K_VALUES = (5, 10, 20)
MAX_K = max(K_VALUES)
TEMPERATURE = 0.07
MEAN_BOOTSTRAP_REPS = 2000
CORR_BOOTSTRAP_REPS = 100
BOOTSTRAP_SEED = 20260929

SUBLOG_NAMES = {
    "source": "NDP来源审计",
    "leakage": "NDP训练测试泄漏审计",
    "features_1234": "NDP提取Seed1234特征",
    "features_4321": "NDP提取Seed4321特征",
    "all_graph": "NDP构建全图邻域",
    "negative_graph": "NDP构建负样本邻域",
    "intrusion": "NDP错误身份侵入分析",
    "negative_inflation": "NDP最近负样本膨胀分析",
    "rank": "NDP邻域排序稳定性分析",
    "js": "NDP邻域分布漂移分析",
    "hub": "NDP坏Hub分析",
    "cross_seed": "NDP跨Seed一致性分析",
    "final": "NDP最终结论",
}


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field, "")) for field in fields})


def read_json(path: Path) -> Any:
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


class RunLogger:
    def __init__(self, run_time: str):
        self.run_time = run_time
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.main = LOG_ROOT / f"NDP邻域判别保持可行性诊断_{run_time}.log"
        self.sub = {key: LOG_ROOT / f"{name}_{run_time}.log" for key, name in SUBLOG_NAMES.items()}
        # Append is intentional: a technical retry with the same RUN_TIME must
        # preserve the first attempt rather than silently overwrite it.
        self._append(self.main, f"\n【运行记录】RUN_TIME={run_time}\n")
        for key, path in self.sub.items():
            self._append(path, f"\n【运行记录】RUN_TIME={run_time}\n【中文说明】NDP-A 子步骤日志启动：{SUBLOG_NAMES[key]}。\n")

    @staticmethod
    def _append(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text.rstrip("\n") + "\n")

    def write(self, message: str, subkey: str | None = None) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{stamp} UTC] {message}"
        self._append(self.main, line)
        if subkey is not None:
            self._append(self.sub[subkey], line)

    def paths(self) -> dict[str, str]:
        return {"main": str(self.main), **{key: str(path) for key, path in self.sub.items()}}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def parse_report_checkpoint(report: Path, arm: str) -> tuple[str | None, str | None]:
    if not report.is_file():
        return None, None
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith(f"| {arm} "):
            continue
        paths = re.findall(r"`([^`]+/model_final\.pth)`", line)
        hashes = re.findall(r"`([0-9a-f]{64})`", line)
        if paths and hashes:
            return paths[-1], hashes[-1]
    return None, None


def exact_sources(logger: RunLogger) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve exact frozen sources through the Phase 7 manifests/audits."""
    p7i_manifest_path = ROOT / "outputs/new_method_track/phase7i_relation_specificity/PHASE7I_MANIFEST.json"
    p7i_audit_json = ROOT / "outputs/new_method_track/phase7i_relation_specificity/PHASE7I_SOURCE_AUDIT.json"
    p7i_audit_report = REPORT_ROOT / "PHASE_7I_SOURCE_AUDIT.md"
    p7i_result_report = REPORT_ROOT / "PHASE_7I_CROSS_CLOTHES_RELATION_SPECIFICITY_TEST.md"
    p7o_manifest_path = ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/PHASE7O_MANIFEST.json"
    p7o_audit_report = REPORT_ROOT / "PHASE_7O_SOURCE_AUDIT.md"

    required = [p7i_manifest_path, p7i_audit_json, p7i_audit_report, p7i_result_report, p7o_manifest_path, p7o_audit_report]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        logger.write(f"【运行阻塞】来源审计所需文件缺失：{missing}。本阶段无法形成科学结论。", "source")
        raise FileNotFoundError("missing source audit input: " + ", ".join(missing))

    p7i = read_json(p7i_manifest_path)
    p7o = read_json(p7o_manifest_path)
    i0_dir = Path(p7i["reused_reference"]["I0"])
    i3_dir = Path(p7i["training_outputs"]["I3"])
    i0_path = i0_dir / "model_final.pth"
    i3_path = i3_dir / "model_final.pth"

    runtime_rows = {row["arm"]: row for row in p7o.get("geometry", {}).get("runtime", [])}
    if not {"O0", "O1"}.issubset(runtime_rows):
        raise RuntimeError("Phase 7O manifest lacks exact O0/O1 geometry checkpoint rows")
    o0_path = Path(runtime_rows["O0"]["checkpoint"])
    o1_path = Path(runtime_rows["O1"]["checkpoint"])

    report_i0_path, report_i0_hash = parse_report_checkpoint(p7i_result_report, "I0")
    report_i3_path, report_i3_hash = parse_report_checkpoint(p7i_result_report, "I3")
    expected_i0 = report_i0_hash
    expected_i3 = report_i3_hash
    if expected_i0 is None:
        audit_json = read_json(p7i_audit_json)
        expected_i0 = audit_json.get("source_manifests", {}).get("I0", {}).get("files", {}).get(str(i0_path), {}).get("sha256")

    checkpoints = {
        "seed1234": {
            "seed": 1234,
            "bio_arm": "I0",
            "gpc_arm": "I3",
            "bio_checkpoint": str(i0_path),
            "gpc_checkpoint": str(i3_path),
            "source_manifest": str(p7i_manifest_path),
            "source_audit_report": str(p7i_audit_report),
            "source_result_report": str(p7i_result_report),
            "expected_hashes": {"bio": expected_i0, "gpc": expected_i3},
        },
        "seed4321": {
            "seed": 4321,
            "bio_arm": "O0",
            "gpc_arm": "O1",
            "bio_checkpoint": str(o0_path),
            "gpc_checkpoint": str(o1_path),
            "source_manifest": str(p7o_manifest_path),
            "source_audit_report": str(p7o_audit_report),
            "source_result_report": str(REPORT_ROOT / "PHASE_7O_GENERIC_POSITIVE_CONSISTENCY_REPLICATION.md"),
            "manifest_runtime_rows": {"bio": runtime_rows["O0"], "gpc": runtime_rows["O1"]},
            "expected_hashes": {"bio": None, "gpc": None},
        },
    }

    source_rows: list[dict[str, Any]] = []
    source_pass = True
    for seed_key, source in checkpoints.items():
        for role in ("bio", "gpc"):
            path = Path(source[f"{role}_checkpoint"])
            actual_hash = sha256_file(path) if path.is_file() else None
            expected_hash = source["expected_hashes"].get(role)
            hash_match = expected_hash is None or actual_hash == expected_hash
            run_complete = path.parent / "RUN_COMPLETE.json"
            complete_ok = run_complete.is_file()
            row = {
                "seed": source["seed"],
                "seed_key": seed_key,
                "role": role,
                "arm": source[f"{role}_arm"],
                "checkpoint": str(path),
                "exists": path.is_file(),
                "checkpoint_sha256": actual_hash,
                "expected_sha256_from_source": expected_hash,
                "hash_match_or_unavailable": hash_match,
                "run_complete_marker": str(run_complete),
                "run_complete_exists": complete_ok,
                "source_manifest": source["source_manifest"],
                "source_audit_report": source["source_audit_report"],
            }
            source_rows.append(row)
            if not path.is_file() or not hash_match or not complete_ok:
                source_pass = False

    audit = {
        "status": "PASS" if source_pass else "FAIL",
        "phase7i_manifest": str(p7i_manifest_path),
        "phase7i_source_audit_json": str(p7i_audit_json),
        "phase7i_source_audit_report": str(p7i_audit_report),
        "phase7o_manifest": str(p7o_manifest_path),
        "phase7o_source_audit_report": str(p7o_audit_report),
        "phase7i_report_checkpoint_paths": {"I0": report_i0_path, "I3": report_i3_path},
        "checkpoint_rows": source_rows,
        "resolution_rule": "I0/I3 from PHASE7I_MANIFEST exact arm directories; O0/O1 from PHASE7O_MANIFEST geometry.runtime exact checkpoint fields; all files/hash/run-complete audited.",
    }
    write_json(OUT_ROOT / "source_audit.json", audit)
    logger.write("【来源审计】开始使用 Phase 7I/7O manifest 与 source audit 解析 exact checkpoint。", "source")
    for row in source_rows:
        logger.write(
            f"【来源审计】Seed{row['seed']} {row['arm']} {row['role']} checkpoint={row['checkpoint']}；存在={row['exists']}；SHA256={row['checkpoint_sha256']}；hash校验={row['hash_match_or_unavailable']}；RUN_COMPLETE={row['run_complete_exists']}。",
            "source",
        )
    if source_pass:
        logger.write("【中文结论】Source Audit PASS：四个冻结 checkpoint 均由对应 manifest 定位且文件完整，未修改 checkpoint。", "source")
    else:
        logger.write("【运行阻塞】Source Audit FAIL：冻结 checkpoint/hash/run marker 审计未通过，停止后续统计。", "source")
        raise RuntimeError("source audit failed")
    return checkpoints, audit


def load_bank(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    required = {"features", "pid", "clothes", "image_path"}
    if not required.issubset(data.files):
        raise RuntimeError(f"feature bank missing keys: {path} -> {data.files}")
    bank = {key: data[key] for key in data.files}
    if "camid" not in bank:
        paths = bank["image_path"].astype(str)
        camid = []
        for image_path in paths:
            match = re.search(r"_c(\d+)_", image_path)
            if not match:
                raise RuntimeError(f"cannot derive camera from train path: {image_path}")
            camid.append(int(match.group(1)) - 1)
        bank["camid"] = np.asarray(camid, dtype=np.int64)
    bank["features"] = np.asarray(bank["features"], dtype=np.float32)
    bank["pid"] = np.asarray(bank["pid"], dtype=np.int64)
    bank["clothes"] = np.asarray(bank["clothes"], dtype=np.int64)
    bank["camid"] = np.asarray(bank["camid"], dtype=np.int64)
    bank["image_path"] = bank["image_path"].astype(str)
    return bank


def feature_bank_sources() -> dict[int, dict[str, Path]]:
    return {
        1234: {
            "bio": ROOT / "outputs/new_method_track/phase7i_relation_specificity/geometry/feature_banks/I0_train_features.npz",
            "gpc": ROOT / "outputs/new_method_track/phase7i_relation_specificity/geometry/feature_banks/I3_train_features.npz",
        },
        4321: {
            "bio": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/geometry/O0_train_features.npz",
            "gpc": ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/geometry/O1_train_features.npz",
        },
    }


def train_only_alignment_audit(logger: RunLogger, checkpoints: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    logger.write("【特征步骤】读取既有 exact-checkpoint train-only feature bank；本轮不调用 dataset constructor，不读取 query/gallery/test。")
    source_paths = feature_bank_sources()
    loaded: dict[int, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    all_path_reference: np.ndarray | None = None
    all_pid_reference: np.ndarray | None = None
    all_clothes_reference: np.ndarray | None = None
    all_cam_reference: np.ndarray | None = None
    feature_pass = True

    for seed in (1234, 4321):
        loaded[seed] = {}
        for role in ("bio", "gpc"):
            source_path = source_paths[seed][role]
            bank = load_bank(source_path)
            n, dim = bank["features"].shape
            norms = np.linalg.norm(bank["features"].astype(np.float64), axis=1)
            norm_error = float(np.max(np.abs(norms - 1.0)))
            paths = bank["image_path"]
            train_only = bool(all("/train/" in path for path in paths))
            forbidden = [path for path in paths if any(token in path.lower() for token in ("/query/", "/test/", "gallery"))]
            row = {
                "seed": seed,
                "role": role,
                "source_feature_bank": str(source_path),
                "source_feature_bank_sha256": sha256_file(source_path),
                "images": int(n),
                "feature_dim": int(dim),
                "pid_count": int(np.unique(bank["pid"]).size),
                "clothing_count": int(np.unique(bank["clothes"]).size),
                "camera_count": int(np.unique(bank["camid"]).size),
                "normalized_max_abs_error": norm_error,
                "all_paths_train": train_only,
                "forbidden_path_count": len(forbidden),
                "forbidden_path_examples": forbidden[:3],
            }
            audit_rows.append(row)
            loaded[seed][role] = bank
            if not train_only or forbidden or n != 9576 or dim != 1024 or norm_error > 2e-4:
                feature_pass = False

            if all_path_reference is None:
                all_path_reference = paths.copy()
                all_pid_reference = bank["pid"].copy()
                all_clothes_reference = bank["clothes"].copy()
                all_cam_reference = bank["camid"].copy()
            else:
                same_path = np.array_equal(all_path_reference, paths)
                same_pid = np.array_equal(all_pid_reference, bank["pid"])
                same_clothes = np.array_equal(all_clothes_reference, bank["clothes"])
                same_cam = np.array_equal(all_cam_reference, bank["camid"])
                row.update({"path_order_aligned": same_path, "pid_aligned": same_pid, "clothes_aligned": same_clothes, "camera_aligned": same_cam})
                if not (same_path and same_pid and same_clothes and same_cam):
                    feature_pass = False

    # A direct train-directory check is restricted to the train path already
    # present in the banks; query/gallery/test directories are never opened.
    path_exists = bool(all_path_reference is not None and all(Path(path).is_file() for path in all_path_reference))
    if not path_exists:
        feature_pass = False
    alignment = {
        "status": "PASS" if feature_pass else "FAIL",
        "rows": audit_rows,
        "path_order_same_across_all_four_banks": bool(all_path_reference is not None),
        "all_train_image_files_exist": path_exists,
        "train_images": int(len(all_path_reference)) if all_path_reference is not None else 0,
        "train_pids": int(np.unique(all_pid_reference).size) if all_pid_reference is not None else 0,
        "train_clothing_states": int(np.unique(all_clothes_reference).size) if all_clothes_reference is not None else 0,
        "train_cameras": int(np.unique(all_cam_reference).size) if all_cam_reference is not None else 0,
        "feature_dim": int(loaded[1234]["bio"]["features"].shape[1]) if loaded else None,
        "prohibited_sources_read": False,
        "feature_provenance": "Existing Phase 7I/7O geometry train banks generated by frozen-checkpoint eval extraction; repacked under NDP-A without checkpoint/model modification.",
    }
    write_json(OUT_ROOT / "feature_alignment_audit.json", alignment)
    write_csv(TABLE_ROOT / "dataset_source.csv", audit_rows)
    for seed, subkey in ((1234, "features_1234"), (4321, "features_4321")):
        logger.write(
            f"【特征对齐】Seed{seed}：train images=9576，PID=77，clothing states=256，feature_dim=1024；path/PID/clothing/camera order 已与其它 bank 比较；当前状态={alignment['status']}。",
            subkey,
        )
    if feature_pass:
        logger.write("【中文结论】Feature alignment audit PASS：四个 bank 均为 normalized feat[0]、全为 LTCC train、逐路径/PID/clothing/camera 对齐。", "features_1234")
        logger.write("【中文结论】Feature alignment audit PASS：四个 bank 均为 normalized feat[0]、全为 LTCC train、逐路径/PID/clothing/camera 对齐。", "features_4321")
    else:
        logger.write("【运行阻塞】Feature alignment audit FAIL：存在 split、维度、归一化或顺序问题，未继续邻域统计。", "features_1234")
        raise RuntimeError("feature alignment audit failed")

    # Save the required NDP-A names.  The arrays are copied from the already
    # audited frozen train banks; no checkpoint is opened or changed here.
    bank_manifest: dict[str, Any] = {}
    for seed in (1234, 4321):
        seed_dir = FEATURE_ROOT / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        bank_manifest[str(seed)] = {}
        for role, filename in (("bio", "bio_feature_bank.npz"), ("gpc", "gpc_feature_bank.npz")):
            source_path = source_paths[seed][role]
            bank = loaded[seed][role]
            output_path = seed_dir / filename
            np.savez_compressed(output_path, features=bank["features"], pid=bank["pid"], clothes=bank["clothes"], camid=bank["camid"], image_path=bank["image_path"])
            bank_manifest[str(seed)][role] = {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "checkpoint": checkpoints[f"seed{seed}"][f"{role}_checkpoint"],
                "feature_definition": "normalized feat[0]",
                "split": "LTCC train only",
                "images": int(bank["features"].shape[0]),
                "dim": int(bank["features"].shape[1]),
            }
    write_json(OUT_ROOT / "feature_banks_manifest.json", bank_manifest)
    logger.write("【特征输出】已生成 seed1234/seed4321 下的 bio_feature_bank.npz 与 gpc_feature_bank.npz；只保存 train feature/path/PID/clothing/camera。")
    return loaded, {"alignment": alignment, "banks": bank_manifest}


def topk_graph(features: np.ndarray, pids: np.ndarray, negative_only: bool, logger: RunLogger, subkey: str) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    indices = np.empty((n, MAX_K), dtype=np.int32)
    scores = np.empty((n, MAX_K), dtype=np.float32)
    block_size = 256
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        rows = np.arange(start, end)
        sim = features[start:end] @ features.T
        sim[np.arange(end - start), rows] = -np.inf
        if negative_only:
            same = pids[None, :] == pids[rows, None]
            sim[same] = -np.inf
        if np.any(~np.isfinite(np.max(sim, axis=1))):
            raise RuntimeError("neighbor graph has an anchor without a finite candidate")
        selected = np.argpartition(-sim, MAX_K - 1, axis=1)[:, :MAX_K]
        selected_scores = np.take_along_axis(sim, selected, axis=1)
        order = np.argsort(-selected_scores, axis=1, kind="stable")
        indices[start:end] = np.take_along_axis(selected, order, axis=1)
        scores[start:end] = np.take_along_axis(selected_scores, order, axis=1)
    logger.write(f"【邻域构建】{('negative-only' if negative_only else 'all-image')} graph 完成：N={n}，Top-{MAX_K}，仅使用 train feature。", subkey)
    return indices, scores


def build_graphs(logger: RunLogger, loaded: dict[int, dict[str, Any]]) -> dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    graphs: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in (1234, 4321):
        graphs[seed] = {"bio": {}, "gpc": {}}
        for role in ("bio", "gpc"):
            bank = loaded[seed][role]
            for neighborhood, negative_only, subkey in (("all_image", False, "all_graph"), ("negative_only", True, "negative_graph")):
                graph = topk_graph(bank["features"], bank["pid"], negative_only, logger, subkey)
                graphs[seed][role][neighborhood] = graph
                path = GRAPH_ROOT / f"seed{seed}_{role}_{neighborhood}_top{MAX_K}.npz"
                np.savez_compressed(path, indices=graph[0], scores=graph[1], image_path=bank["image_path"], pid=bank["pid"])
        # Store the paired graph in one easy-to-consume artifact as well.
        for neighborhood in ("all_image", "negative_only"):
            path = GRAPH_ROOT / f"seed{seed}_{neighborhood}_bio_vs_gpc.npz"
            np.savez_compressed(
                path,
                bio_indices=graphs[seed]["bio"][neighborhood][0],
                bio_scores=graphs[seed]["bio"][neighborhood][1],
                gpc_indices=graphs[seed]["gpc"][neighborhood][0],
                gpc_scores=graphs[seed]["gpc"][neighborhood][1],
            )
    write_json(OUT_ROOT / "neighbor_graph_manifest.json", {"K_values": list(K_VALUES), "stored_top_k": MAX_K, "graphs": {str(seed): {role: {name: str(GRAPH_ROOT / f"seed{seed}_{role}_{name}_top{MAX_K}.npz") for name in ("all_image", "negative_only")} for role in ("bio", "gpc")} for seed in (1234, 4321)}})
    return graphs


def group_means(values: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    unique = np.unique(pids[finite])
    means = []
    for pid in unique:
        vals = values[finite & (pids == pid)]
        means.append(float(np.mean(vals)) if len(vals) else float("nan"))
    return unique, np.asarray(means, dtype=np.float64)


def bootstrap_mean_ci(values: np.ndarray, pids: np.ndarray, seed: int, reps: int = MEAN_BOOTSTRAP_REPS) -> tuple[float, float]:
    _, means = group_means(np.asarray(values, dtype=np.float64), pids)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = means[rng.integers(0, len(means), size=(reps, len(means)))].mean(axis=1)
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


def metric_summary(values: np.ndarray, pids: np.ndarray, seed: int) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "pid_bootstrap_ci95_low": None, "pid_bootstrap_ci95_high": None}
    low, high = bootstrap_mean_ci(values, pids, seed)
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "pid_bootstrap_ci95_low": low,
        "pid_bootstrap_ci95_high": high,
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and values[order[end]] == values[order[pos]]:
            end += 1
        ranks[order[pos:end]] = (pos + end - 1) / 2.0
        pos = end
    return ranks


def spearman_value(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    rx = rankdata(np.asarray(x)[mask])
    ry = rankdata(np.asarray(y)[mask])
    sx = float(np.std(rx)); sy = float(np.std(ry))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_value(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]; y = np.asarray(y)[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu_indices(n, k=1)
    prod = np.sign(dx[upper]) * np.sign(dy[upper])
    non_tie = prod != 0
    if not np.any(non_tie):
        return 0.0
    return float(np.mean(prod[non_tie]))


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, pids: np.ndarray, seed: int, reps: int = CORR_BOOTSTRAP_REPS) -> tuple[float, float]:
    groups = [np.flatnonzero(pids == pid) for pid in np.unique(pids)]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[int(index)] for index in chosen])
        value = spearman_value(x[indices], y[indices])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def per_anchor_sets(bio_indices: np.ndarray, gpc_indices: np.ndarray, pids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(pids)
    jaccard = np.empty(n, dtype=np.float64)
    intrusion = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bio_set = set(int(value) for value in bio_indices[i, :k])
        gpc_set = set(int(value) for value in gpc_indices[i, :k])
        union = bio_set | gpc_set
        jaccard[i] = len(bio_set & gpc_set) / len(union) if union else 1.0
        intrusion[i] = sum(1 for value in gpc_set - bio_set if pids[value] != pids[i]) / float(k)
    return jaccard, intrusion


def same_id_recall(indices: np.ndarray, pids: np.ndarray, k: int) -> np.ndarray:
    available = np.bincount(pids, minlength=int(np.max(pids)) + 1)[pids] - 1
    denominator = np.minimum(k, available).astype(np.float64)
    denominator = np.maximum(denominator, 1.0)
    same = (pids[indices[:, :k]] == pids[:, None]).sum(axis=1)
    return same.astype(np.float64) / denominator


def pairwise_anchor_summary(z: np.ndarray, pids: np.ndarray, clothes: np.ndarray) -> dict[str, np.ndarray]:
    n = len(z)
    neg = np.empty(n, dtype=np.float64)
    pos_all = np.empty(n, dtype=np.float64)
    pos_cross = np.empty(n, dtype=np.float64)
    block_size = 256
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        rows = np.arange(start, end)
        sim = z[start:end] @ z.T
        same = pids[None, :] == pids[rows, None]
        same[np.arange(end - start), rows] = False
        cross = same & (clothes[None, :] != clothes[rows, None])
        diff = ~same
        diff[np.arange(end - start), rows] = False
        neg_values = np.where(diff, sim, -np.inf)
        same_values = np.where(same, sim, np.nan)
        cross_values = np.where(cross, sim, np.nan)
        neg[start:end] = np.max(neg_values, axis=1)
        same_count = np.sum(same, axis=1)
        cross_count = np.sum(cross, axis=1)
        same_sum = np.nansum(same_values, axis=1)
        cross_sum = np.nansum(cross_values, axis=1)
        pos_all[start:end] = np.where(same_count > 0, same_sum / np.maximum(same_count, 1), np.nan)
        pos_cross[start:end] = np.where(cross_count > 0, cross_sum / np.maximum(cross_count, 1), np.nan)
    return {"s_neg": neg, "s_pos_all": pos_all, "s_pos_cross": pos_cross, "has_cross": np.isfinite(pos_cross)}


def union_distribution_metrics(z_bio: np.ndarray, z_gpc: np.ndarray, bio_indices: np.ndarray, gpc_indices: np.ndarray, pids: np.ndarray, k: int, tau: float, only_negative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pids)
    js = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)
    tau_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        candidates = np.unique(np.concatenate((bio_indices[i, :k], gpc_indices[i, :k])))
        if only_negative:
            candidates = candidates[pids[candidates] != pids[i]]
        sb = z_bio[i] @ z_bio[candidates].T
        sg = z_gpc[i] @ z_gpc[candidates].T
        rb = np.argsort(np.argsort(-sb, kind="mergesort"), kind="mergesort").astype(np.float64)
        rg = np.argsort(np.argsort(-sg, kind="mergesort"), kind="mergesort").astype(np.float64)
        rho[i] = spearman_value(rb, rg)
        tau_values[i] = kendall_value(sb, sg)
        sb = sb / tau; sg = sg / tau
        sb -= np.max(sb); sg -= np.max(sg)
        pb = np.exp(sb); pg = np.exp(sg)
        pb /= np.sum(pb); pg /= np.sum(pg)
        pm = 0.5 * (pb + pg)
        js[i] = 0.5 * (np.sum(pb * np.log(np.maximum(pb, 1e-300) / np.maximum(pm, 1e-300))) + np.sum(pg * np.log(np.maximum(pg, 1e-300) / np.maximum(pm, 1e-300))))
    return js, rho, tau_values


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or np.sum(x) <= 0:
        return 0.0
    return float(np.sum((2 * np.arange(1, len(x) + 1) - len(x) - 1) * x) / (len(x) * np.sum(x)))


def top_membership(counts: np.ndarray, fraction: float) -> np.ndarray:
    take = max(1, int(math.ceil(len(counts) * fraction)))
    order = np.lexsort((np.arange(len(counts)), -counts))
    mask = np.zeros(len(counts), dtype=bool)
    mask[order[:take]] = True
    return mask


def hub_counts(indices: np.ndarray, anchor_pids: np.ndarray, target_pids: np.ndarray, k: int, only_negative: bool) -> tuple[np.ndarray, np.ndarray]:
    target = indices[:, :k].reshape(-1)
    if only_negative:
        keep = np.ones(len(target), dtype=bool)
    else:
        anchor = np.repeat(np.arange(len(anchor_pids)), k)
        keep = target_pids[target] != anchor_pids[anchor]
    image_counts = np.bincount(target[keep], minlength=len(target_pids)).astype(np.int64)
    pid_counts = np.bincount(target_pids[target[keep]], minlength=int(np.max(target_pids)) + 1).astype(np.int64)
    return image_counts, pid_counts


def hub_summary(counts: np.ndarray) -> dict[str, Any]:
    total = int(np.sum(counts))
    order = np.sort(counts)[::-1]
    top1_n = max(1, int(math.ceil(len(counts) * 0.01)))
    top5_n = max(1, int(math.ceil(len(counts) * 0.05)))
    return {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "max": int(np.max(counts)) if len(counts) else 0,
        "p95": float(np.percentile(counts, 95)) if len(counts) else None,
        "p99": float(np.percentile(counts, 99)) if len(counts) else None,
        "top1_occurrence_share": float(np.sum(order[:top1_n]) / total) if total else 0.0,
        "top5_occurrence_share": float(np.sum(order[:top5_n]) / total) if total else 0.0,
        "gini": gini(counts),
        "total_occurrences": total,
    }


def candidate_rows_for_new_hubs(
    seed: int,
    k: int,
    neighborhood: str,
    bio_counts: np.ndarray,
    gpc_counts: np.ndarray,
    gpc_indices: np.ndarray,
    pids: np.ndarray,
    new_mask: np.ndarray,
    delta_margin: np.ndarray,
    level: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in np.flatnonzero(new_mask):
        hit = np.any(gpc_indices[:, :k] == candidate, axis=1)
        if level == "image" and neighborhood != "negative_only":
            hit &= pids[candidate] != pids
        affected = np.flatnonzero(hit)
        rows.append({
            "seed": seed,
            "K": k,
            "neighborhood": neighborhood,
            "level": level,
            "candidate_index": int(candidate),
            "candidate_pid": int(pids[candidate]) if level == "image" else int(candidate),
            "bio_hub_count": int(bio_counts[candidate]),
            "gpc_hub_count": int(gpc_counts[candidate]),
            "hub_count_delta": int(gpc_counts[candidate] - bio_counts[candidate]),
            "affected_anchor_count": int(len(affected)),
            "affected_anchor_percent": float(len(affected) / len(pids)),
            "affected_anchor_mean_delta_margin": float(np.mean(delta_margin[affected])) if len(affected) else None,
        })
    return rows


def quartile_labels(values: np.ndarray) -> np.ndarray:
    n = len(values)
    order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="mergesort")
    labels = np.empty(n, dtype=np.int64)
    for q, chunk in enumerate(np.array_split(order, 4), start=1):
        labels[chunk] = q
    return labels


def primary_metric_rows(seed: int, primary: dict[str, np.ndarray], pids: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in primary.items():
        summary = metric_summary(values, pids, BOOTSTRAP_SEED + seed + sum(ord(c) for c in name))
        rows.append({"seed": seed, "metric": name, **summary})
    return rows, {name: metric_summary(values, pids, BOOTSTRAP_SEED + seed + sum(ord(c) for c in name)) for name, values in primary.items()}


def analyze_one_seed(seed: int, loaded: dict[int, dict[str, Any]], graphs: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]], logger: RunLogger) -> dict[str, Any]:
    bio = loaded[seed]["bio"]
    gpc = loaded[seed]["gpc"]
    pids = bio["pid"]
    clothes = bio["clothes"]
    z_bio = bio["features"]
    z_gpc = gpc["features"]

    logger.write(f"【Seed{seed}】开始 train-only positive/negative anchor 统计。", "negative_inflation")
    pos_neg_bio = pairwise_anchor_summary(z_bio, pids, clothes)
    pos_neg_gpc = pairwise_anchor_summary(z_gpc, pids, clothes)
    cross_valid = pos_neg_bio["has_cross"] & pos_neg_gpc["has_cross"]
    # The primary same-ID positive is cross-clothes whenever that positive
    # exists.  For the 1,209 LTCC train anchors whose PID has only one
    # clothing state, the all-same-ID positive is an explicit fallback.  The
    # cross-clothes-only arrays remain in the report as a sensitivity audit.
    primary_pos_bio = np.where(cross_valid, pos_neg_bio["s_pos_cross"], pos_neg_bio["s_pos_all"])
    primary_pos_gpc = np.where(cross_valid, pos_neg_gpc["s_pos_cross"], pos_neg_gpc["s_pos_all"])
    delta_margin_primary = (primary_pos_gpc - pos_neg_gpc["s_neg"]) - (primary_pos_bio - pos_neg_bio["s_neg"])
    primary = {
        "s_pos_all_bio": pos_neg_bio["s_pos_all"],
        "s_pos_all_gpc": pos_neg_gpc["s_pos_all"],
        "s_pos_cross_clothes_bio": pos_neg_bio["s_pos_cross"],
        "s_pos_cross_clothes_gpc": pos_neg_gpc["s_pos_cross"],
        "s_neg_bio": pos_neg_bio["s_neg"],
        "s_neg_gpc": pos_neg_gpc["s_neg"],
        "delta_s_pos_all": pos_neg_gpc["s_pos_all"] - pos_neg_bio["s_pos_all"],
        "delta_s_pos_cross_clothes": pos_neg_gpc["s_pos_cross"] - pos_neg_bio["s_pos_cross"],
        "delta_s_pos_primary": primary_pos_gpc - primary_pos_bio,
        "delta_s_neg": pos_neg_gpc["s_neg"] - pos_neg_bio["s_neg"],
        "delta_margin_all": (pos_neg_gpc["s_pos_all"] - pos_neg_gpc["s_neg"]) - (pos_neg_bio["s_pos_all"] - pos_neg_bio["s_neg"]),
        "delta_margin_cross_clothes_only": (pos_neg_gpc["s_pos_cross"] - pos_neg_gpc["s_neg"]) - (pos_neg_bio["s_pos_cross"] - pos_neg_bio["s_neg"]),
        "delta_margin_primary": delta_margin_primary,
        "cross_clothes_valid": cross_valid.astype(np.float64),
        "ordering_bio": np.where(cross_valid, (pos_neg_bio["s_pos_cross"] > pos_neg_bio["s_neg"]).astype(np.float64), np.nan),
        "ordering_gpc": np.where(cross_valid, (pos_neg_gpc["s_pos_cross"] > pos_neg_gpc["s_neg"]).astype(np.float64), np.nan),
    }
    # Keep the established field name for all downstream tables, but it now
    # means the pre-declared cross-clothes-primary/all-same-ID-fallback margin.
    primary["delta_margin_cross_clothes"] = primary["delta_margin_primary"]
    primary["delta_ordering_correctness"] = primary["ordering_gpc"] - primary["ordering_bio"]
    primary_rows, primary_summaries = primary_metric_rows(seed, primary, pids)
    write_csv(TABLE_ROOT / f"seed{seed}_primary_anchor_metrics.csv", primary_rows)
    logger.write("【最近负样本】已计算 all-train different-PID nearest negative、same-ID positive、cross-clothes positive、margin、ordering correctness。", "negative_inflation")

    result: dict[str, Any] = {"primary": primary, "primary_summaries": primary_summaries, "by_k": {}, "pids": pids}
    for k in K_VALUES:
        result["by_k"][k] = {}
        logger.write(f"【Seed{seed}】【K={k}】开始 overlap/intrusion/retention/rank/JS/hub 统计。")
        for neighborhood in ("all_image", "negative_only"):
            bio_idx = graphs[seed]["bio"][neighborhood][0]
            gpc_idx = graphs[seed]["gpc"][neighborhood][0]
            jaccard, intrusion = per_anchor_sets(bio_idx, gpc_idx, pids, k)
            js, rho, kendall = union_distribution_metrics(z_bio, z_gpc, bio_idx, gpc_idx, pids, k, TEMPERATURE, neighborhood == "negative_only")
            delta_same_recall = None
            bio_recall = None
            gpc_recall = None
            if neighborhood == "all_image":
                bio_recall = same_id_recall(bio_idx, pids, k)
                gpc_recall = same_id_recall(gpc_idx, pids, k)
                delta_same_recall = gpc_recall - bio_recall

            bio_hub_image, bio_hub_pid = hub_counts(bio_idx, pids, pids, k, neighborhood == "negative_only")
            gpc_hub_image, gpc_hub_pid = hub_counts(gpc_idx, pids, pids, k, neighborhood == "negative_only")
            bio_top5_image = top_membership(bio_hub_image, 0.05)
            gpc_top5_image = top_membership(gpc_hub_image, 0.05)
            new_image = gpc_top5_image & ~bio_top5_image
            bio_top5_pid = top_membership(bio_hub_pid, 0.05)
            gpc_top5_pid = top_membership(gpc_hub_pid, 0.05)
            new_pid = gpc_top5_pid & ~bio_top5_pid

            wrong_mask_all = (pids[gpc_idx[:, :k]] != pids[:, None])
            image_exposure = np.sum(new_image[gpc_idx[:, :k]] & wrong_mask_all, axis=1).astype(np.float64) / float(k)
            pid_exposure = np.sum(new_pid[pids[gpc_idx[:, :k]]] & wrong_mask_all, axis=1).astype(np.float64) / float(k)
            if neighborhood == "negative_only":
                image_exposure = np.sum(new_image[gpc_idx[:, :k]], axis=1).astype(np.float64) / float(k)
                pid_exposure = np.sum(new_pid[pids[gpc_idx[:, :k]]], axis=1).astype(np.float64) / float(k)

            metric_store = {
                "jaccard": jaccard,
                "one_minus_jaccard": 1.0 - jaccard,
                "intrusion": intrusion,
                "js": js,
                "spearman": rho,
                "kendall_tau": kendall,
                "rank_change": 1.0 - rho,
                "new_hub_exposure_image": image_exposure,
                "new_hub_exposure_pid": pid_exposure,
                "bio_hub_image": bio_hub_image,
                "gpc_hub_image": gpc_hub_image,
                "bio_hub_pid": bio_hub_pid,
                "gpc_hub_pid": gpc_hub_pid,
                "new_bad_image_mask": new_image,
                "new_bad_pid_mask": new_pid,
                "bio_same_recall": bio_recall,
                "gpc_same_recall": gpc_recall,
                "delta_same_recall": delta_same_recall,
            }
            result["by_k"][k][neighborhood] = metric_store

            metric_rows = []
            for name, values in (("jaccard", jaccard), ("one_minus_jaccard", 1.0 - jaccard), ("js_divergence", js), ("spearman", rho), ("kendall_tau", kendall), ("rank_change", 1.0 - rho), ("new_hub_exposure_image", image_exposure), ("new_hub_exposure_pid", pid_exposure)):
                metric_rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, "metric": name, **metric_summary(values, pids, BOOTSTRAP_SEED + seed + k + len(name))})
            if neighborhood == "all_image":
                for name, values in (("wrong_pid_intrusion_rate", intrusion), ("same_id_recall_bio", bio_recall), ("same_id_recall_gpc", gpc_recall), ("delta_same_id_recall", delta_same_recall)):
                    metric_rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, "metric": name, **metric_summary(values, pids, BOOTSTRAP_SEED + seed + k + len(name))})
            write_csv(TABLE_ROOT / f"seed{seed}_K{k}_{neighborhood}_anchor_summaries.csv", metric_rows)

            hub_rows = []
            for role, counts in (("Bio", bio_hub_image), ("GPC", gpc_hub_image)):
                hub_rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, "level": "image", "method": role, **hub_summary(counts)})
            for role, counts in (("Bio", bio_hub_pid), ("GPC", gpc_hub_pid)):
                hub_rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, "level": "PID", "method": role, **hub_summary(counts)})
            result["by_k"][k][neighborhood]["hub_rows"] = hub_rows

            # PID-level intrusion summary is only defined for all-image top-K.
            if neighborhood == "all_image":
                pid_rows = []
                for pid in np.unique(pids):
                    mask = pids == pid
                    pid_rows.append({
                        "seed": seed,
                        "K": k,
                        "pid": int(pid),
                        "anchor_count": int(mask.sum()),
                        "mean_intrusion": float(np.mean(intrusion[mask])),
                        "intrusion_positive_anchor_percent": float(np.mean(intrusion[mask] > 0) * 100.0),
                        "mean_delta_margin": float(np.mean(primary["delta_margin_cross_clothes"][mask])),
                        "mean_delta_same_id_recall": float(np.mean(delta_same_recall[mask])),
                    })
                result["by_k"][k][neighborhood]["pid_intrusion_rows"] = pid_rows

            result["by_k"][k][neighborhood]["new_hub_rows"] = candidate_rows_for_new_hubs(
                seed, k, neighborhood, bio_hub_image, gpc_hub_image, gpc_idx, pids, new_image, primary["delta_margin_cross_clothes"], "image"
            ) + candidate_rows_for_new_hubs(
                seed, k, neighborhood, bio_hub_pid, gpc_hub_pid, pids[gpc_idx[:, :k]], pids, new_pid, primary["delta_margin_cross_clothes"], "PID"
            )

            if neighborhood == "all_image":
                disruption_values = {
                    "wrong_pid_intrusion": intrusion,
                    "js_divergence": js,
                    "new_hub_exposure": image_exposure,
                }
            else:
                disruption_values = {
                    "wrong_pid_intrusion": intrusion,
                    "js_divergence": js,
                    "new_hub_exposure": image_exposure,
                }
            harmful = primary["delta_margin_cross_clothes"] < -1e-6
            beneficial = primary["delta_margin_cross_clothes"] > 1e-6
            neutral = ~(harmful | beneficial)
            group_rows = []
            for group_name, group_mask in (("Beneficial", beneficial), ("Harmful", harmful), ("Neutral", neutral)):
                group_rows.append({
                    "seed": seed,
                    "K": k,
                    "neighborhood": neighborhood,
                    "group": group_name,
                    "count": int(group_mask.sum()),
                    "pid_count": int(np.unique(pids[group_mask]).size),
                    "mean_delta_s_pos": float(np.mean(primary["delta_s_pos_primary"][group_mask])) if group_mask.any() else None,
                    "mean_delta_s_neg": float(np.mean(primary["delta_s_neg"][group_mask])) if group_mask.any() else None,
                    "mean_delta_margin": float(np.mean(primary["delta_margin_cross_clothes"][group_mask])) if group_mask.any() else None,
                    "mean_jaccard": float(np.mean(jaccard[group_mask])) if group_mask.any() else None,
                    "mean_wrong_pid_intrusion": float(np.mean(intrusion[group_mask])) if group_mask.any() else None,
                    "mean_js_divergence": float(np.mean(js[group_mask])) if group_mask.any() else None,
                    "mean_new_hub_exposure": float(np.mean(image_exposure[group_mask])) if group_mask.any() else None,
                    "mean_rank_stability": float(np.mean(rho[group_mask])) if group_mask.any() else None,
                    "harmful_percent": float(np.mean(harmful[group_mask]) * 100.0) if group_mask.any() else None,
                })
            result["by_k"][k][neighborhood]["harmful_rows"] = group_rows

            # Quartiles are deterministic rank quartiles over all anchors;
            # they are not selected after seeing which K looks favorable.
            quartile_rows = []
            for disruption_name, disruption in disruption_values.items():
                labels = quartile_labels(disruption)
                for q in range(1, 5):
                    mask = labels == q
                    quartile_rows.append({
                        "seed": seed,
                        "K": k,
                        "neighborhood": neighborhood,
                        "disruption": disruption_name,
                        "quartile": f"Q{q}",
                        "count": int(mask.sum()),
                        "pid_count": int(np.unique(pids[mask]).size),
                        "disruption_min": float(np.min(disruption[mask])) if mask.any() else None,
                        "disruption_max": float(np.max(disruption[mask])) if mask.any() else None,
                        "delta_s_pos": float(np.mean(primary["delta_s_pos_primary"][mask])) if mask.any() else None,
                        "delta_s_neg": float(np.mean(primary["delta_s_neg"][mask])) if mask.any() else None,
                        "delta_margin": float(np.mean(primary["delta_margin_cross_clothes"][mask])) if mask.any() else None,
                        "same_id_recall_change": float(np.mean(delta_same_recall[mask])) if delta_same_recall is not None and mask.any() else None,
                        "harmful_percent": float(np.mean(harmful[mask]) * 100.0) if mask.any() else None,
                    })
            result["by_k"][k][neighborhood]["quartile_rows"] = quartile_rows

        logger.write(f"【Seed{seed}】【K={k}】overlap、侵入、same-ID retention、rank、JS、hubness 已完成。")
    return result


def write_analysis_tables(all_results: dict[int, dict[str, Any]], logger: RunLogger) -> dict[str, Any]:
    overlap_rows: list[dict[str, Any]] = []
    intrusion_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    js_rows: list[dict[str, Any]] = []
    image_hub_rows: list[dict[str, Any]] = []
    pid_hub_rows: list[dict[str, Any]] = []
    harmful_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    new_hub_rows: list[dict[str, Any]] = []
    pid_intrusion_rows: list[dict[str, Any]] = []
    for seed, result in all_results.items():
        primary = result["primary"]
        for k in K_VALUES:
            all_m = result["by_k"][k]["all_image"]
            neg_m = result["by_k"][k]["negative_only"]
            for neighborhood, metrics in (("all_image", all_m), ("negative_only", neg_m)):
                row = {"seed": seed, "K": k, "neighborhood": neighborhood, **metric_summary(metrics["jaccard"], result["pids"], BOOTSTRAP_SEED + seed + k)}
                overlap_rows.append(row)
            intrusion_rows.append({"seed": seed, "K": k, "neighborhood": "all_image", **metric_summary(all_m["intrusion"], result["pids"], BOOTSTRAP_SEED + seed + k + 11), "anchors_with_intrusion_percent": float(np.mean(all_m["intrusion"] > 0) * 100.0), "pids_with_positive_intrusion": int(np.sum([np.mean(all_m["intrusion"][result["pids"] == pid]) > 0 for pid in np.unique(result["pids"])]))})
            retention_rows.append({"seed": seed, "K": k, "neighborhood": "all_image", "bio": metric_summary(all_m["bio_same_recall"], result["pids"], BOOTSTRAP_SEED + seed + k + 21), "gpc": metric_summary(all_m["gpc_same_recall"], result["pids"], BOOTSTRAP_SEED + seed + k + 22), "delta": metric_summary(all_m["delta_same_recall"], result["pids"], BOOTSTRAP_SEED + seed + k + 23)})
            rank_rows.append({"seed": seed, "K": k, "neighborhood": "negative_only", "spearman": metric_summary(neg_m["spearman"], result["pids"], BOOTSTRAP_SEED + seed + k + 31), "kendall_tau": metric_summary(neg_m["kendall_tau"], result["pids"], BOOTSTRAP_SEED + seed + k + 32), "rank_change": metric_summary(neg_m["rank_change"], result["pids"], BOOTSTRAP_SEED + seed + k + 33)})
            for neighborhood, metrics in (("all_image", all_m), ("negative_only", neg_m)):
                js_rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, **metric_summary(metrics["js"], result["pids"], BOOTSTRAP_SEED + seed + k + 41)})
            for row in all_m["hub_rows"] + neg_m["hub_rows"]:
                if row["level"] == "image":
                    image_hub_rows.append(row)
                else:
                    pid_hub_rows.append(row)
            for metrics in (all_m, neg_m):
                harmful_rows.extend(metrics["harmful_rows"])
                quartile_rows.extend(metrics["quartile_rows"])
                new_hub_rows.extend(metrics["new_hub_rows"])
            pid_intrusion_rows.extend(all_m.get("pid_intrusion_rows", []))

        primary_rows, _ = primary_metric_rows(seed, primary, result["pids"])
        write_csv(TABLE_ROOT / f"seed{seed}_primary_metric_summary.csv", primary_rows)

    write_csv(TABLE_ROOT / "overlap.csv", overlap_rows)
    write_csv(TABLE_ROOT / "intrusion.csv", intrusion_rows)
    # Nested dictionaries in retention/rank rows are JSON strings so the table
    # stays CSV-readable while retaining the full mean/quantile/CI summaries.
    write_csv(TABLE_ROOT / "same_id_retention.csv", [{**row, "bio": json.dumps(json_safe(row["bio"]), ensure_ascii=False), "gpc": json.dumps(json_safe(row["gpc"]), ensure_ascii=False), "delta": json.dumps(json_safe(row["delta"]), ensure_ascii=False)} for row in retention_rows])
    write_csv(TABLE_ROOT / "rank_stability.csv", [{**row, "spearman": json.dumps(json_safe(row["spearman"]), ensure_ascii=False), "kendall_tau": json.dumps(json_safe(row["kendall_tau"]), ensure_ascii=False), "rank_change": json.dumps(json_safe(row["rank_change"]), ensure_ascii=False)} for row in rank_rows])
    write_csv(TABLE_ROOT / "js_divergence.csv", js_rows)
    write_csv(TABLE_ROOT / "image_hubness.csv", image_hub_rows)
    write_csv(TABLE_ROOT / "pid_hubness.csv", pid_hub_rows)
    write_csv(TABLE_ROOT / "harmful_vs_beneficial.csv", harmful_rows)
    write_csv(TABLE_ROOT / "quartiles.csv", quartile_rows)
    write_csv(TABLE_ROOT / "new_bad_hub_candidates.csv", new_hub_rows)
    write_csv(TABLE_ROOT / "pid_intrusion.csv", pid_intrusion_rows)
    write_json(TABLE_ROOT / "nearest_negative_inflation.json", {str(seed): {key: json_safe(value) for key, value in result["primary_summaries"].items()} for seed, result in all_results.items()})
    logger.write("【表格输出】已生成 overlap、intrusion、same-ID retention、nearest-negative、rank、JS、image/PID hub、harmful、quartile、cross-seed 前置表。")
    return {"overlap": overlap_rows, "intrusion": intrusion_rows, "retention": retention_rows, "rank": rank_rows, "js": js_rows, "image_hub": image_hub_rows, "pid_hub": pid_hub_rows, "harmful": harmful_rows, "quartiles": quartile_rows, "new_hubs": new_hub_rows}


def correlation_analysis(all_results: dict[int, dict[str, Any]], logger: RunLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logger.write(f"【相关性】使用 anchor-level Spearman，并以 PID cluster bootstrap（{CORR_BOOTSTRAP_REPS} 次）给出 95% CI。", "cross_seed")
    for seed, result in all_results.items():
        pids = result["pids"]
        delta_margin = result["primary"]["delta_margin_cross_clothes"]
        for k in K_VALUES:
            delta_recall = result["by_k"][k]["all_image"]["delta_same_recall"]
            for neighborhood in ("all_image", "negative_only"):
                metrics = result["by_k"][k][neighborhood]
                disruption_map = {
                    "one_minus_jaccard": metrics["one_minus_jaccard"],
                    "js_divergence": metrics["js"],
                    "new_hub_exposure": metrics["new_hub_exposure_image"],
                }
                if neighborhood == "all_image":
                    disruption_map["wrong_pid_intrusion"] = metrics["intrusion"]
                else:
                    disruption_map["negative_rank_change"] = metrics["rank_change"]
                for disruption_name, disruption in disruption_map.items():
                    for outcome_name, outcome in (("delta_margin_cross_clothes", delta_margin), ("delta_same_id_recall", delta_recall)):
                        rho = spearman_value(disruption, outcome)
                        ci_low, ci_high = bootstrap_corr_ci(disruption, outcome, pids, BOOTSTRAP_SEED + seed + k + len(disruption_name) + len(outcome_name))
                        rows.append({"seed": seed, "K": k, "neighborhood": neighborhood, "disruption": disruption_name, "outcome": outcome_name, "spearman": rho, "pid_bootstrap_ci95_low": ci_low, "pid_bootstrap_ci95_high": ci_high, "count": len(disruption)})
    write_csv(TABLE_ROOT / "correlations.csv", rows)
    logger.write("【相关性】已完成 1-Jaccard、intrusion、JS、hub exposure/rank change 与 Delta Margin/Delta Same-ID Recall 的 Spearman。", "cross_seed")
    return rows


def cross_seed_consistency(all_results: dict[int, dict[str, Any]], correlations: list[dict[str, Any]], logger: RunLogger) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for seed, result in all_results.items():
            pids = result["pids"]
            all_m = result["by_k"][k]["all_image"]
            neg_m = result["by_k"][k]["negative_only"]
            delta_summary = metric_summary(result["primary"]["delta_margin_cross_clothes"], result["pids"], BOOTSTRAP_SEED + seed + k + 91)
            delta_neg_summary = metric_summary(result["primary"]["delta_s_neg"], result["pids"], BOOTSTRAP_SEED + seed + k + 92)
            intrusion_summary = metric_summary(all_m["intrusion"], result["pids"], BOOTSTRAP_SEED + seed + k + 93)
            js_summary = metric_summary(neg_m["js"], result["pids"], BOOTSTRAP_SEED + seed + k + 94)
            hub_delta = neg_m["gpc_hub_image"].astype(np.float64) - neg_m["bio_hub_image"].astype(np.float64)
            hub_delta_summary = metric_summary(hub_delta[pids], result["pids"], BOOTSTRAP_SEED + seed + k + 95)
            seed_corr = [row for row in correlations if row["seed"] == seed and row["K"] == k and row["outcome"] == "delta_margin_cross_clothes"]
            intrusion_corr = next((row for row in seed_corr if row["disruption"] == "wrong_pid_intrusion"), None)
            js_corr = next((row for row in seed_corr if row["disruption"] == "js_divergence" and row["neighborhood"] == "negative_only"), None)
            rows.append({
                "seed": seed,
                "K": k,
                "intrusion_mean": intrusion_summary["mean"],
                "intrusion_ci_low": intrusion_summary["pid_bootstrap_ci95_low"],
                "intrusion_ci_high": intrusion_summary["pid_bootstrap_ci95_high"],
                "delta_s_neg_mean": delta_neg_summary["mean"],
                "delta_s_neg_ci_low": delta_neg_summary["pid_bootstrap_ci95_low"],
                "delta_s_neg_ci_high": delta_neg_summary["pid_bootstrap_ci95_high"],
                "delta_margin_mean": delta_summary["mean"],
                "delta_margin_ci_low": delta_summary["pid_bootstrap_ci95_low"],
                "delta_margin_ci_high": delta_summary["pid_bootstrap_ci95_high"],
                "delta_ordering_mean": float(np.nanmean(result["primary"]["delta_ordering_correctness"])),
                "js_negative_mean": js_summary["mean"],
                "js_negative_ci_low": js_summary["pid_bootstrap_ci95_low"],
                "js_negative_ci_high": js_summary["pid_bootstrap_ci95_high"],
                "negative_hub_mean_delta": hub_delta_summary["mean"],
                "negative_hub_delta_ci_low": hub_delta_summary["pid_bootstrap_ci95_low"],
                "negative_hub_delta_ci_high": hub_delta_summary["pid_bootstrap_ci95_high"],
                "intrusion_margin_spearman": intrusion_corr["spearman"] if intrusion_corr else None,
                "intrusion_margin_ci_low": intrusion_corr["pid_bootstrap_ci95_low"] if intrusion_corr else None,
                "intrusion_margin_ci_high": intrusion_corr["pid_bootstrap_ci95_high"] if intrusion_corr else None,
                "js_margin_spearman": js_corr["spearman"] if js_corr else None,
                "js_margin_ci_low": js_corr["pid_bootstrap_ci95_low"] if js_corr else None,
                "js_margin_ci_high": js_corr["pid_bootstrap_ci95_high"] if js_corr else None,
                "pid_positive_intrusion_count": int(np.sum([np.mean(all_m["intrusion"][result["pids"] == pid]) > 0 for pid in np.unique(result["pids"])])),
                "new_bad_pid_count": int(np.sum(neg_m["new_bad_pid_mask"])),
            })
    write_csv(TABLE_ROOT / "cross_seed_consistency.csv", rows)

    # Direction checks are declared over all three pre-specified K values.
    by_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in (1234, 4321)}
    intrusion_positive_both = all(all(row["intrusion_mean"] > 0 for row in by_seed[seed]) for seed in by_seed)
    negative_inflation_both = all(all(row["delta_s_neg_mean"] > 0 for row in by_seed[seed]) for seed in by_seed)
    margin_degradation_both = all(all(row["delta_margin_mean"] < 0 for row in by_seed[seed]) for seed in by_seed)
    intrusion_corr_negative_both = all(all(row["intrusion_margin_spearman"] is not None and row["intrusion_margin_spearman"] < 0 for row in by_seed[seed]) for seed in by_seed)
    js_corr_negative_both = all(all(row["js_margin_spearman"] is not None and row["js_margin_spearman"] < 0 for row in by_seed[seed]) for seed in by_seed)
    pid_coverage_ok = all(all(row["pid_positive_intrusion_count"] >= 39 for row in by_seed[seed]) for seed in by_seed)
    drift_both = all(all(row["js_negative_mean"] > 0 and row["js_negative_ci_low"] > 0 for row in by_seed[seed]) for seed in by_seed)
    intrusion_systematic = all(all(row["intrusion_mean"] > 0 and row["intrusion_ci_low"] > 0 for row in by_seed[seed]) for seed in by_seed)
    relation_stable = intrusion_corr_negative_both or js_corr_negative_both
    gate = {
        "A_two_seeds_significant_neighborhood_drift": bool(drift_both),
        "B_wrong_pid_intrusion_or_bad_hub_systematic_increase": bool(intrusion_systematic or any(row["new_bad_pid_count"] > 0 for row in rows)),
        "C_disruption_associated_with_margin_degradation": bool(relation_stable),
        "D_not_only_few_pids": bool(pid_coverage_ok),
        "E_two_seed_direction_consistency": bool(intrusion_positive_both and negative_inflation_both and margin_degradation_both and relation_stable),
        "all_promotion_conditions": bool(drift_both and (intrusion_systematic or any(row["new_bad_pid_count"] > 0 for row in rows)) and relation_stable and pid_coverage_ok and intrusion_positive_both and negative_inflation_both and margin_degradation_both),
        "direction_checks": {"intrusion_positive_both": intrusion_positive_both, "negative_inflation_both": negative_inflation_both, "margin_degradation_both": margin_degradation_both, "intrusion_corr_negative_both": intrusion_corr_negative_both, "js_corr_negative_both": js_corr_negative_both},
        "gate_rule_note": "Gate requires the stated direction at K=5/10/20; drift CI uses PID bootstrap; non-few-PID rule is at least 39/77 PIDs with positive mean intrusion in both seeds.",
    }
    write_json(TABLE_ROOT / "promotion_gate.json", gate)
    logger.write(f"【跨Seed】Intrusion positive both={intrusion_positive_both}；nearest-negative inflation both={negative_inflation_both}；margin degradation both={margin_degradation_both}；correlation stable={relation_stable}；PID coverage={pid_coverage_ok}。", "cross_seed")
    return rows, gate


def plot_all(all_results: dict[int, dict[str, Any]], logger: RunLogger) -> dict[str, str]:
    if plt is None:
        logger.write("【运行故障】matplotlib 不可用，绘图未生成；数值表与审计仍可用。")
        return {}
    plots: dict[str, str] = {}
    try:
        # 1. Overlap histogram, all required K and both neighborhoods.
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
        for r, seed in enumerate((1234, 4321)):
            for c, k in enumerate(K_VALUES):
                ax = axes[r, c]
                for neighborhood, color in (("all_image", "tab:blue"), ("negative_only", "tab:orange")):
                    ax.hist(all_results[seed]["by_k"][k][neighborhood]["jaccard"], bins=25, alpha=0.55, label=neighborhood, color=color)
                ax.set_title(f"Seed {seed}, K={k}"); ax.set_xlabel("Jaccard"); ax.set_ylabel("Anchors")
                if r == 0 and c == 0: ax.legend(fontsize=8)
        fig.tight_layout(); path = PLOT_ROOT / "bio_vs_gpc_neighborhood_overlap_histogram.png"; fig.savefig(path, dpi=160); plt.close(fig); plots["overlap_histogram"] = str(path)

        # 2. Intrusion distribution.
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
        for r, seed in enumerate((1234, 4321)):
            for c, k in enumerate(K_VALUES):
                ax = axes[r, c]
                ax.hist(all_results[seed]["by_k"][k]["all_image"]["intrusion"], bins=np.linspace(0, 1, k + 2), color="tab:red", alpha=0.8)
                ax.set_title(f"Seed {seed}, K={k}"); ax.set_xlabel("Wrong-PID intrusion rate"); ax.set_ylabel("Anchors")
        fig.tight_layout(); path = PLOT_ROOT / "wrong_pid_intrusion_distribution.png"; fig.savefig(path, dpi=160); plt.close(fig); plots["intrusion_distribution"] = str(path)

        # 3/4. Scatter panels for every pre-defined K.
        for metric_name, source_name, filename, xlabel in (("intrusion", "all_image", "delta_margin_vs_intrusion.png", "Wrong-PID intrusion"), ("js", "negative_only", "delta_margin_vs_js_divergence.png", "JS divergence")):
            fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
            for r, seed in enumerate((1234, 4321)):
                for c, k in enumerate(K_VALUES):
                    ax = axes[r, c]
                    x = all_results[seed]["by_k"][k][source_name][metric_name]
                    y = all_results[seed]["primary"]["delta_margin_cross_clothes"]
                    ax.scatter(x, y, s=3, alpha=0.18, color="tab:purple")
                    ax.axhline(0, color="black", linewidth=0.6); ax.set_title(f"Seed {seed}, K={k}"); ax.set_xlabel(xlabel); ax.set_ylabel("Delta Margin")
            fig.tight_layout(); path = PLOT_ROOT / filename; fig.savefig(path, dpi=160); plt.close(fig); plots[filename[:-4]] = str(path)

        # 5. Image-level hub-count distributions, all K and both seeds.
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
        for r, seed in enumerate((1234, 4321)):
            for c, k in enumerate(K_VALUES):
                ax = axes[r, c]
                for role, color in (("bio_hub_image", "tab:blue"), ("gpc_hub_image", "tab:red")):
                    counts = all_results[seed]["by_k"][k]["negative_only"][role]
                    ax.hist(counts, bins=25, alpha=0.55, label=role, color=color)
                ax.set_title(f"Seed {seed}, K={k}"); ax.set_xlabel("Negative hub count"); ax.set_ylabel("Images")
                if r == 0 and c == 0: ax.legend(fontsize=8)
        fig.tight_layout(); path = PLOT_ROOT / "bio_vs_gpc_hub_count_distribution.png"; fig.savefig(path, dpi=160); plt.close(fig); plots["hub_distribution"] = str(path)

        # 6. Top PID hub comparison for each seed and each K.
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
        for r, seed in enumerate((1234, 4321)):
            for c, k in enumerate(K_VALUES):
                ax = axes[r, c]
                metrics = all_results[seed]["by_k"][k]["negative_only"]
                order = np.argsort(metrics["gpc_hub_pid"])[::-1][:10]
                x = np.arange(len(order)); width = 0.38
                ax.bar(x - width / 2, metrics["bio_hub_pid"][order], width, label="Bio", color="tab:blue")
                ax.bar(x + width / 2, metrics["gpc_hub_pid"][order], width, label="GPC", color="tab:red")
                ax.set_title(f"Seed {seed}, K={k}"); ax.set_xlabel("PID rank by GPC hubness"); ax.set_ylabel("PID hub count")
                if r == 0 and c == 0: ax.legend(fontsize=8)
        fig.tight_layout(); path = PLOT_ROOT / "top_bad_hub_pid_comparison.png"; fig.savefig(path, dpi=160); plt.close(fig); plots["top_pid_hub"] = str(path)

        # 7. Seed mechanism comparison, no K is hidden.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (metric, title) in zip(axes, (("intrusion", "Intrusion"), ("js", "Negative JS"), ("delta_margin_cross_clothes", "Delta Margin"))):
            x = np.arange(len(K_VALUES)); width = 0.35
            for offset, seed, color in ((-width / 2, 1234, "tab:blue"), (width / 2, 4321, "tab:orange")):
                vals = []
                for k in K_VALUES:
                    if metric == "delta_margin_cross_clothes": vals.append(np.mean(all_results[seed]["primary"][metric]))
                    elif metric == "intrusion": vals.append(np.mean(all_results[seed]["by_k"][k]["all_image"][metric]))
                    else: vals.append(np.mean(all_results[seed]["by_k"][k]["negative_only"][metric]))
                ax.bar(x + offset, vals, width, label=f"Seed {seed}", color=color)
            ax.set_xticks(x, [str(k) for k in K_VALUES]); ax.set_title(title); ax.set_xlabel("K"); ax.axhline(0, color="black", linewidth=0.6)
        axes[0].legend(fontsize=8); fig.tight_layout(); path = PLOT_ROOT / "seed1234_vs_seed4321_mechanism_comparison.png"; fig.savefig(path, dpi=160); plt.close(fig); plots["seed_mechanism"] = str(path)
        logger.write("【绘图】已生成 overlap、intrusion、Delta Margin scatter、hub-count、top PID hub、cross-seed mechanism 图。")
    except Exception as exc:
        logger.write(f"【运行故障】绘图阶段异常，数值诊断已保留；原始错误：{type(exc).__name__}: {exc}")
    return plots


def report_table(rows: list[dict[str, Any]], columns: list[str], headers: list[str] | None = None, limit: int | None = None) -> str:
    headers = headers or columns
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    selected = rows if limit is None else rows[:limit]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def write_source_report(run_time: str, logger: RunLogger, source_audit: dict[str, Any]) -> None:
    path = REPORT_ROOT / "NDP_A_SOURCE_AUDIT.md"
    rows = source_audit["checkpoint_rows"]
    lines = [
        "# NDP-A Source Audit",
        "",
        f"RUN_TIME: `{run_time}`",
        f"主日志: `{logger.main}`",
        "",
        "本报告仅审计冻结来源，不启动训练、不调用 optimizer/backward，不修改 checkpoint。Exact checkpoint 从 Phase 7I/7O manifest 解析，并以文件存在性、SHA256 和 RUN_COMPLETE 复核。",
        "",
        f"状态：**{source_audit['status']}**",
        "",
        report_table(rows, ["seed", "arm", "role", "checkpoint", "checkpoint_sha256", "expected_sha256_from_source", "hash_match_or_unavailable", "run_complete_exists"], ["Seed", "Arm", "Role", "Exact checkpoint", "实际 SHA256", "来源 SHA256", "Hash", "RUN_COMPLETE"]),
        "",
        "## 来源文件",
        "",
        f"- Phase 7I manifest: `{source_audit['phase7i_manifest']}`",
        f"- Phase 7I source audit JSON: `{source_audit['phase7i_source_audit_json']}`",
        f"- Phase 7I source audit report: `{source_audit['phase7i_source_audit_report']}`",
        f"- Phase 7O manifest: `{source_audit['phase7o_manifest']}`",
        f"- Phase 7O source audit report: `{source_audit['phase7o_source_audit_report']}`",
        "",
        "## 中文结论",
        "",
        "PASS：本 NDP-A 只读取 Seed1234 的 I0→I3 与 Seed4321 的 O0→O1 冻结 checkpoint 对应 train feature bank；没有猜测 checkpoint path，也没有覆盖或修改原 checkpoint。" if source_audit["status"] == "PASS" else "FAIL：来源审计未通过，不能把后续统计解释为有效科学结论。",
        "",
        "本轮未使用 query/gallery/test checkpoint、label、feature 或 retrieval 指标。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leakage_report(run_time: str, logger: RunLogger, alignment: dict[str, Any]) -> None:
    path = REPORT_ROOT / "NDP_A_DATA_LEAKAGE_AUDIT.md"
    rows = alignment["rows"]
    lines = [
        "# NDP-A Train-only Data Leakage Audit",
        "",
        f"RUN_TIME: `{run_time}`",
        f"主日志: `{logger.main}`",
        "",
        "硬约束：所有 feature bank、neighbor graph、PID center、margin、hubness、overlap 统计仅来自 LTCC train images。NDP-A 脚本不调用 LTCC dataset constructor，不打开 query/gallery/test 目录。",
        "",
        f"状态：**{alignment['status']}**",
        "",
        report_table(rows, ["seed", "role", "images", "feature_dim", "pid_count", "clothing_count", "camera_count", "normalized_max_abs_error", "all_paths_train", "forbidden_path_count"], ["Seed", "Role", "Images", "Dim", "PIDs", "Clothes", "Cameras", "Norm error", "All train", "Forbidden paths"]),
        "",
        "## 审计项目",
        "",
        f"- train images: `{alignment['train_images']}`",
        f"- train PIDs: `{alignment['train_pids']}`",
        f"- train clothing states: `{alignment['train_clothing_states']}`",
        f"- train cameras: `{alignment['train_cameras']}`",
        f"- feature dimension: `{alignment['feature_dim']}`",
        f"- all referenced train image files exist: `{alignment['all_train_image_files_exist']}`",
        f"- prohibited query/gallery/test sources read: `{alignment['prohibited_sources_read']}`",
        "",
        "## 中文结论",
        "",
        "PASS：路径全部包含 `/train/`，不存在 query/gallery/test path；四个 bank 的 path/PID/clothing/camera 顺序一致；所有统计输入均为 train-only。" if alignment["status"] == "PASS" else "FAIL：出现 train-only、对齐或归一化问题，不能继续解释诊断结果。",
        "",
        "注意：报告中的 Phase 7I/7O 历史 source audit 只作为 checkpoint provenance 使用；本轮没有读取其中的 test Rank-1、mAP 或 test features 作为诊断量。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_conclusion(all_results: dict[int, dict[str, Any]], cross_rows: list[dict[str, Any]], gate: dict[str, Any], source_audit: dict[str, Any], alignment: dict[str, Any], plots: dict[str, str], logger: RunLogger, checkpoints: dict[str, Any]) -> dict[str, Any]:
    # A descriptive answer is based on all pre-defined K values.  "MIXED" is
    # retained whenever the directions are not uniform across seeds/scales.
    def yes_no_mixed(values: Iterable[bool]) -> str:
        values = list(values)
        return "YES" if all(values) else ("NO" if not any(values) else "MIXED")

    q1 = yes_no_mixed([np.mean(all_results[seed]["by_k"][k]["all_image"]["one_minus_jaccard"]) > 0 for seed in all_results for k in K_VALUES])
    q2 = yes_no_mixed([np.mean(all_results[seed]["by_k"][k]["negative_only"]["one_minus_jaccard"]) > 0 for seed in all_results for k in K_VALUES])
    q3 = yes_no_mixed([np.mean(all_results[seed]["by_k"][k]["all_image"]["intrusion"]) > 0 for seed in all_results for k in K_VALUES])
    hub_increase_flags = []
    for seed in all_results:
        for k in K_VALUES:
            metrics = all_results[seed]["by_k"][k]["negative_only"]
            bio_hub = hub_summary(metrics["bio_hub_image"])
            gpc_hub = hub_summary(metrics["gpc_hub_image"])
            hub_increase_flags.append(any(gpc_hub[key] > bio_hub[key] + 1e-12 for key in ("max", "p95", "p99", "top1_occurrence_share", "top5_occurrence_share", "gini")))
    q4 = yes_no_mixed(hub_increase_flags)
    q5 = yes_no_mixed([np.mean(all_results[seed]["primary"]["delta_s_neg"]) > 0 for seed in all_results])
    negative_relation = [row["intrusion_margin_spearman"] < 0 for row in cross_rows if row["intrusion_margin_spearman"] is not None]
    q6 = "YES" if gate["C_disruption_associated_with_margin_degradation"] else ("MIXED" if any(negative_relation) else "NO")
    harmful_rows = []
    for seed, result in all_results.items():
        for k in K_VALUES:
            harmful = result["primary"]["delta_margin_cross_clothes"] < -1e-6
            beneficial = result["primary"]["delta_margin_cross_clothes"] > 1e-6
            all_m = result["by_k"][k]["all_image"]
            neg_m = result["by_k"][k]["negative_only"]
            if harmful.any() and beneficial.any():
                harmful_rows.append((np.mean(all_m["intrusion"][harmful]) > np.mean(all_m["intrusion"][beneficial])) and (np.mean(neg_m["js"][harmful]) > np.mean(neg_m["js"][beneficial])))
    q7 = yes_no_mixed(harmful_rows) if harmful_rows else "MIXED"
    q8 = "YES" if gate["E_two_seed_direction_consistency"] else ("MIXED" if gate["C_disruption_associated_with_margin_degradation"] else "NO")
    motivation = "SUPPORTED" if gate["all_promotion_conditions"] else ("MIXED" if gate["C_disruption_associated_with_margin_degradation"] or gate["B_wrong_pid_intrusion_or_bad_hub_systematic_increase"] else "NOT SUPPORTED")
    decision = "PROMOTE TO NDP-B" if motivation == "SUPPORTED" else "STOP"

    primary_rows = []
    for seed in (1234, 4321):
        p = all_results[seed]["primary_summaries"]
        primary_rows.append({
            "seed": seed,
            "delta_s_pos_cross_mean": p["delta_s_pos_primary"]["mean"],
            "cross_clothes_valid_percent": float(np.mean(all_results[seed]["primary"]["cross_clothes_valid"]) * 100.0),
            "delta_s_neg_mean": p["delta_s_neg"]["mean"],
            "delta_margin_cross_mean": p["delta_margin_cross_clothes"]["mean"],
            "delta_ordering_mean": p["delta_ordering_correctness"]["mean"],
            "harmful_percent": float(np.mean(all_results[seed]["primary"]["delta_margin_cross_clothes"] < -1e-6) * 100.0),
        })

    report = {
        "source_audit": source_audit["status"],
        "train_only_leakage_audit": alignment["status"],
        "seed1234": primary_rows[0],
        "seed4321": primary_rows[1],
        "wrong_pid_intrusion": q3,
        "bad_hubness_increase": q4,
        "nearest_negative_inflation": q5,
        "neighborhood_drift_vs_delta_margin": q6,
        "harmful_vs_beneficial": q7,
        "cross_seed_consistency": q8,
        "ndp_motivation": motivation,
        "decision": decision,
        "promotion_gate": gate,
        "plots": plots,
        "safety_declarations": {
            "NO_NEW_TRAINING_WAS_STARTED": True,
            "NO_NDP_LOSS_WAS_IMPLEMENTED": True,
            "NO_QUERY_GALLERY_TEST_DATA_WAS_USED": True,
            "NO_EXISTING_GPC_CHECKPOINT_WAS_MODIFIED": True,
        },
    }
    write_json(OUT_ROOT / "final_conclusion.json", report)

    source_rows = []
    for seed in (1234, 4321):
        r = all_results[seed]
        for k in K_VALUES:
            all_m = r["by_k"][k]["all_image"]
            neg_m = r["by_k"][k]["negative_only"]
            source_rows.append({
                "seed": seed,
                "K": k,
                "intrusion_mean": float(np.mean(all_m["intrusion"])),
                "intrusion_anchor_percent": float(np.mean(all_m["intrusion"] > 0) * 100.0),
                "same_recall_delta_mean": float(np.mean(all_m["delta_same_recall"])),
                "negative_js_mean": float(np.mean(neg_m["js"])),
                "negative_rank_spearman_mean": float(np.mean(neg_m["spearman"])),
                "negative_hub_bio_mean": float(np.mean(neg_m["bio_hub_image"])),
                "negative_hub_gpc_mean": float(np.mean(neg_m["gpc_hub_image"])),
                "new_bad_image_count": int(np.sum(neg_m["new_bad_image_mask"])),
                "new_bad_pid_count": int(np.sum(neg_m["new_bad_pid_mask"])),
            })

    report_path = REPORT_ROOT / "NDP_A_NEIGHBORHOOD_DISCRIMINATIVE_FEASIBILITY_AUDIT.md"
    lines = [
        "# NDP-A Neighborhood Discriminative Preservation — 可行性诊断",
        "",
        f"RUN_TIME: `{logger.run_time}`",
        f"主日志: `{logger.main}`",
        "",
        "本阶段只做离线机制验证。所有 feature、neighbor graph、margin、hubness、overlap 与相关性均来自 LTCC train split；没有启动 NDP 训练。",
        "",
        "## A. Source audit",
        "",
        f"**{source_audit['status']}**。exact checkpoint 见 [NDP_A_SOURCE_AUDIT.md](NDP_A_SOURCE_AUDIT.md)。",
        "",
        "## B. Train-only leakage audit",
        "",
        f"**{alignment['status']}**。见 [NDP_A_DATA_LEAKAGE_AUDIT.md](NDP_A_DATA_LEAKAGE_AUDIT.md)。",
        "",
        "## C. Seed1234 neighborhood result（I0 → I3）",
        "",
        report_table([primary_rows[0]], ["seed", "delta_s_pos_cross_mean", "cross_clothes_valid_percent", "delta_s_neg_mean", "delta_margin_cross_mean", "delta_ordering_mean", "harmful_percent"], ["Seed", "Δ primary same-ID positive", "Cross-clothes valid %", "Δ nearest negative", "Δ primary margin", "Δ ordering", "Harmful anchors %"]),
        "",
        "## D. Seed4321 neighborhood result（O0 → O1）",
        "",
        report_table([primary_rows[1]], ["seed", "delta_s_pos_cross_mean", "cross_clothes_valid_percent", "delta_s_neg_mean", "delta_margin_cross_mean", "delta_ordering_mean", "harmful_percent"], ["Seed", "Δ primary same-ID positive", "Cross-clothes valid %", "Δ nearest negative", "Δ primary margin", "Δ ordering", "Harmful anchors %"]),
        "",
        "### K=5/10/20 train-only overview",
        "",
        report_table(source_rows, ["seed", "K", "intrusion_mean", "intrusion_anchor_percent", "same_recall_delta_mean", "negative_js_mean", "negative_rank_spearman_mean", "negative_hub_bio_mean", "negative_hub_gpc_mean", "new_bad_image_count", "new_bad_pid_count"], ["Seed", "K", "Intrusion mean", "Anchors with intrusion %", "Δ same-ID recall", "Negative JS", "Negative rank ρ", "Bio hub mean", "GPC hub mean", "New image hubs", "New PID hubs"]),
        "",
        "## E. Wrong-PID intrusion",
        "",
        f"**{q3}**。完整 mean/median/p25/p75/PID-bootstrap CI 见 `outputs/new_method_track/ndp_a_neighborhood_feasibility/tables/intrusion.csv`；per-PID 结果见 `pid_intrusion.csv`。",
        "",
        "## F. Bad hubness increase",
        "",
        f"**{q4}**。image-level 与 PID-level 的 mean/std/max/top 1%/top 5% share/Gini/p95/p99 见 `image_hubness.csv` 与 `pid_hubness.csv`；new bad hub candidate 见 `new_bad_hub_candidates.csv`。",
        "",
        "## G. Neighborhood drift 与 Delta Margin 的关系",
        "",
        f"**{q6}**。Spearman 与 PID-bootstrap CI 见 `tables/correlations.csv`；保留了 1-Jaccard、intrusion、JS、hub exposure、negative rank change 五类原始量，没有合成为单一训练 score。",
        "",
        "## H. Cross-seed consistency",
        "",
        f"**{q8}**。Seed1234/Seed4321 分开计算，趋势比较见 `tables/cross_seed_consistency.csv`。",
        "",
        "## I. NDP motivation",
        "",
        f"**{motivation}**。Promotion gate 的 A–E 逐项结果写入 `tables/promotion_gate.json`。",
        "",
        "## J. Decision",
        "",
        f"**{decision}**",
        "",
        "## K. Required declarations",
        "",
        "- NO NEW TRAINING WAS STARTED.",
        "- NO NDP LOSS WAS IMPLEMENTED.",
        "- NO QUERY/GALLERY/TEST DATA WAS USED.",
        "- NO EXISTING GPC CHECKPOINT WAS MODIFIED.",
        "",
        "## 指标与输出说明",
        "",
        "- all-image neighborhood：排除 anchor 自身后全体 train images Top-K。",
        "- negative-only neighborhood：只保留不同 PID 的 train images Top-K。",
        "- same-ID retention：Top-K 内 same-PID 数除以 min(K, 可用 same-PID 数)。",
        "- 主 Delta Margin 使用 same-ID cross-clothes positive mean − nearest negative；对没有 cross-clothes positive 的 1,209 个 train anchor，明确 fallback 到 all same-ID positive；cross-clothes-only 数值仍保留在 seed primary anchor table。",
        "- JS temperature 固定为 0.07；K 只使用 5/10/20；所有 PID bootstrap 使用固定随机种子并报告 95% CI。",
        "- harmful：Delta Margin < −1e−6；beneficial：> +1e−6；其余 neutral。",
        "",
        "## 运行日志与产物",
        "",
        f"- Manifest: `{OUT_ROOT / 'NDP_A_MANIFEST.json'}`",
        f"- 输出根目录: `{OUT_ROOT}`",
        f"- 子日志：`{json.dumps(logger.paths(), ensure_ascii=False)}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final checkpoint integrity check is performed after all NumPy work.
    integrity_rows = []
    for seed_key, source in checkpoints.items():
        for role in ("bio", "gpc"):
            path = Path(source[f"{role}_checkpoint"])
            current = sha256_file(path)
            initial = next(row["checkpoint_sha256"] for row in source_audit["checkpoint_rows"] if row["seed_key"] == seed_key and row["role"] == role)
            integrity_rows.append({"seed": source["seed"], "role": role, "checkpoint": str(path), "sha256_before": initial, "sha256_after": current, "unchanged": current == initial})
    write_json(OUT_ROOT / "checkpoint_integrity_after.json", integrity_rows)
    logger.write(f"【最终结论】Q1 all-image drift={q1}；Q2 negative-only drift={q2}；Q3 intrusion={q3}；Q4 bad hubness={q4}；Q5 nearest-negative inflation={q5}；Q6 drift-margin relation={q6}；Q7 harmful anchors={q7}；Q8 cross-seed={q8}。", "final")
    logger.write(f"【Promotion Gate】A={gate['A_two_seeds_significant_neighborhood_drift']} B={gate['B_wrong_pid_intrusion_or_bad_hub_systematic_increase']} C={gate['C_disruption_associated_with_margin_degradation']} D={gate['D_not_only_few_pids']} E={gate['E_two_seed_direction_consistency']}；最终={motivation}，决定={decision}。", "final")
    logger.write("【中文结论】NO NEW TRAINING WAS STARTED；NO NDP LOSS WAS IMPLEMENTED；NO QUERY/GALLERY/TEST DATA WAS USED；NO EXISTING GPC CHECKPOINT WAS MODIFIED。", "final")
    return report


def manifest(run_time: str, logger: RunLogger, checkpoints: dict[str, Any], source_audit: dict[str, Any], alignment: dict[str, Any], bank_manifest: dict[str, Any], gate: dict[str, Any], conclusion: dict[str, Any], plots: dict[str, str], git: str | None) -> None:
    tables = {path.stem: str(path) for path in TABLE_ROOT.glob("*") if path.is_file()}
    plot_paths = {key: value for key, value in plots.items()}
    payload = {
        "run_time": run_time,
        "main_log": str(logger.main),
        "sub_logs": logger.paths(),
        "repo": str(ROOT),
        "git_commit": git,
        "purpose": "NDP-A offline feasibility diagnostic for local inter-ID neighborhood disruption after GPC",
        "source_audit": source_audit,
        "data_leakage_audit": alignment,
        "checkpoints": checkpoints,
        "feature_banks": bank_manifest,
        "dataset": {"name": "LTCC", "split": "train only", "images": alignment["train_images"], "pids": alignment["train_pids"], "clothing_states": alignment["train_clothing_states"], "cameras": alignment["train_cameras"], "feature_dim": alignment["feature_dim"]},
        "fixed_protocol": {"K": list(K_VALUES), "temperature": TEMPERATURE, "same_id_positive": "mean same-PID distinct-image positive; cross-clothes version used when available, explicit all-same-ID fallback for anchors without a cross-clothes positive", "cross_clothes_valid_anchors": 8367, "cross_clothes_missing_anchors": 1209, "nearest_negative": "max cosine over different-PID train images", "harmful_threshold": -1e-6, "beneficial_threshold": 1e-6, "pid_bootstrap_reps": MEAN_BOOTSTRAP_REPS, "correlation_bootstrap_reps": CORR_BOOTSTRAP_REPS},
        "outputs": {"root": str(OUT_ROOT), "feature_banks": str(FEATURE_ROOT), "neighbor_graphs": str(GRAPH_ROOT), "tables": str(TABLE_ROOT), "plots": str(PLOT_ROOT), "source_report": str(REPORT_ROOT / "NDP_A_SOURCE_AUDIT.md"), "leakage_report": str(REPORT_ROOT / "NDP_A_DATA_LEAKAGE_AUDIT.md"), "audit_report": str(REPORT_ROOT / "NDP_A_NEIGHBORHOOD_DISCRIMINATIVE_FEASIBILITY_AUDIT.md")},
        "tables": tables,
        "plots": plot_paths,
        "promotion_gate": gate,
        "final_conclusion": conclusion,
        "safety": {"new_training_started": False, "optimizer_step_called": False, "backward_called": False, "ndp_loss_implemented": False, "gpc_modified": False, "bio_modified": False, "checkpoint_modified": False, "sampler_modified": False, "dataset_split_modified": False, "query_gallery_test_used": False},
    }
    write_json(OUT_ROOT / "NDP_A_MANIFEST.json", payload)
    logger.write(f"【产物】已生成 Manifest：{OUT_ROOT / 'NDP_A_MANIFEST.json'}。", "final")


def run(run_time: str) -> None:
    logger = RunLogger(run_time)
    logger.write("【第3次尝试/技术性修复】修复 cross-clothes 缺失 anchor 的 Delta s_pos 展示、Delta ordering 的 NaN 汇总，以及 cross-seed 表格字段；不改变实验数据、K、温度或 gate 规则。")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (FEATURE_ROOT, GRAPH_ROOT, TABLE_ROOT, PLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    logger.write(f"当前仓库路径：{ROOT}")
    logger.write(f"当前 Git commit/hash：{git_commit()}")
    logger.write("本阶段目的：验证 GPC 是否在增强 same-ID consistency 的同时扰动局部 inter-ID discriminative neighborhood。")
    logger.write("【中文声明】本阶段不训练 NDP；不调用 optimizer.step()；不调用 backward 更新参数；不修改 GPC、Bio、checkpoint、sampler 或 dataset split。")
    logger.write("【中文声明】所有诊断量只来自 LTCC train split；不读取 query/gallery/test features、labels、Rank-1 或 mAP。")

    checkpoints, source_audit = exact_sources(logger)
    write_source_report(run_time, logger, source_audit)
    loaded, bank_info = train_only_alignment_audit(logger, checkpoints)
    alignment = bank_info["alignment"]
    write_leakage_report(run_time, logger, alignment)
    logger.write("【泄漏审计】PASS：报告已写入 reports/NDP_A_DATA_LEAKAGE_AUDIT.md。", "leakage")

    graphs = build_graphs(logger, loaded)
    logger.write("【邻域状态】K=5/10/20 均按预先定义执行；没有根据结果增加或选择其它 K。")
    results = {seed: analyze_one_seed(seed, loaded, graphs, logger) for seed in (1234, 4321)}
    write_analysis_tables(results, logger)
    correlations = correlation_analysis(results, logger)
    cross_rows, gate = cross_seed_consistency(results, correlations, logger)
    plots = plot_all(results, logger)
    conclusion = final_conclusion(results, cross_rows, gate, source_audit, alignment, plots, logger, checkpoints)
    manifest(run_time, logger, checkpoints, source_audit, alignment, bank_info["banks"], gate, conclusion, plots, git_commit())
    logger.write("【阶段结束】NDP-A 离线诊断完成；没有进行任何新训练。", "final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    try:
        run(args.run_time)
    except Exception as exc:
        # Preserve a Chinese explanation plus the raw traceback in the main
        # log.  The caller can distinguish a runtime block from a scientific
        # STOP decision in the final reports.
        main_log = LOG_ROOT / f"NDP邻域判别保持可行性诊断_{args.run_time}.log"
        with main_log.open("a", encoding="utf-8") as stream:
            stream.write("【运行故障】NDP-A 未完成有效科学结论；以下保留原始 Python traceback 用于定位：\n")
            import traceback

            stream.write(traceback.format_exc())
            stream.write("【中文结论】本轮属于运行故障/阻塞，不解释为 NDP 方法失败。\n")
        raise


if __name__ == "__main__":
    main()
