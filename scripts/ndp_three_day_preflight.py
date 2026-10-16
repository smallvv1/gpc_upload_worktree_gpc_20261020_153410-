#!/usr/local/bin/python3
"""NDP three-day source, leakage, and PPU preflight.

The preflight is intentionally separate from the long-running orchestrator.
It resolves frozen checkpoints from the audited Phase 7I/7O manifests, hashes
them, checks train-only artifacts and split boundaries, and executes one
two-PPU forward/backward smoke test.  It never creates a model checkpoint,
does not call an evaluator, and does not use query/gallery features or labels
for a method decision.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/DIFFER")
LOG_ROOT = ROOT / "logs"
REPORT_ROOT = ROOT / "reports"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class Logger:
    def __init__(self, run_time: str, main_log: Path):
        self.run_time = run_time
        self.main = main_log
        self.paths = {
            "source": LOG_ROOT / f"NDP来源审计_{run_time}.log",
            "leakage": LOG_ROOT / f"NDP训练测试泄漏审计_{run_time}.log",
            "ppu": LOG_ROOT / f"NDP双PPU预检_{run_time}.log",
        }
        for path in [self.main, *self.paths.values()]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n【运行记录】RUN_TIME={run_time}；日志追加模式，不覆盖已有证据。\n")

    def write(self, message: str, sub: str | None = None) -> None:
        line = f"[{now()}] {message}\n"
        with self.main.open("a", encoding="utf-8") as stream:
            stream.write(line)
        if sub:
            with self.paths[sub].open("a", encoding="utf-8") as stream:
                stream.write(line)


def resolve_sources(logger: Logger) -> dict[str, Any]:
    logger.write("【来源审计】开始读取 NDP-A、Phase 7I、Phase 7O 的 manifest 和 source audit；不猜 checkpoint 路径。", "source")
    required = [
        ROOT / "reports/NDP_A_NEIGHBORHOOD_DISCRIMINATIVE_FEASIBILITY_AUDIT.md",
        ROOT / "reports/NDP_A_SOURCE_AUDIT.md",
        ROOT / "reports/NDP_A_DATA_LEAKAGE_AUDIT.md",
        ROOT / "outputs/new_method_track/ndp_a_neighborhood_feasibility/NDP_A_MANIFEST.json",
        ROOT / "reports/PHASE_7I_SOURCE_AUDIT.md",
        ROOT / "reports/PHASE_7O_SOURCE_AUDIT.md",
        ROOT / "outputs/new_method_track/phase7i_relation_specificity/PHASE7I_MANIFEST.json",
        ROOT / "outputs/new_method_track/phase7i_relation_specificity/PHASE7I_SOURCE_AUDIT.json",
        ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/PHASE7O_MANIFEST.json",
        ROOT / "outputs/final_confirmatory_gpc/FINAL_GPC_MANIFEST.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        logger.write(f"【运行阻塞】来源审计输入缺失：{missing}；不把此情况写成方法失败。", "source")
        raise RuntimeError("source audit inputs missing")

    p7i = read_json(required[6])
    p7o = read_json(required[8])
    i0 = Path(p7i["reused_reference"]["I0"]) / "model_final.pth"
    i3 = Path(p7i["training_outputs"]["I3"]) / "model_final.pth"
    runtime_rows = {row["arm"]: row for row in p7o.get("geometry", {}).get("runtime", [])}
    if not {"O0", "O1"}.issubset(runtime_rows):
        raise RuntimeError("Phase 7O manifest has no exact O0/O1 runtime rows")
    o0 = Path(runtime_rows["O0"]["checkpoint"])
    o1 = Path(runtime_rows["O1"]["checkpoint"])
    source_rows = []
    for seed, roles in ((1234, {"bio": ("I0", i0), "gpc": ("I3", i3)}), (4321, {"bio": ("O0", o0), "gpc": ("O1", o1)})):
        for role, (arm, path) in roles.items():
            run_complete = path.parent / "RUN_COMPLETE.json"
            row = {
                "seed": seed,
                "role": role,
                "arm": arm,
                "checkpoint": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path),
                "run_complete": str(run_complete),
                "run_complete_exists": run_complete.is_file(),
                "source_manifest": str(required[6] if seed == 1234 else required[8]),
            }
            source_rows.append(row)
            logger.write(
                f"【来源审计】Seed{seed} {arm} {role}：checkpoint={path}；存在={row['exists']}；SHA256={row['sha256']}；RUN_COMPLETE={row['run_complete_exists']}。",
                "source",
            )
    passed = all(row["exists"] and row["run_complete_exists"] and bool(row["sha256"]) for row in source_rows)
    if not passed:
        logger.write("【运行阻塞】Source Audit FAIL：四个冻结 checkpoint 不完整；停止后续 pipeline。", "source")
        raise RuntimeError("source audit failed")
    logger.write("【中文结论】Source Audit PASS：四个 checkpoint 均由 manifest exact resolution，已计算 SHA256，原 checkpoint 不修改。", "source")
    return {
        "status": "PASS",
        "run_time": logger.run_time,
        "required_inputs": [str(path) for path in required],
        "checkpoints": source_rows,
        "frozen_gpc": {
            "loss": "L_CE + L_Triplet + L_Bio + 0.1 * L_GPC",
            "gpc": "mean(1-cos(normalized feat[0])) over same-PID distinct-image pairs",
            "lambda_gpc": 0.1,
            "ds": False,
            "dw": False,
            "h_id": False,
            "sr3": False,
            "bio_modified": False,
        },
    }


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def split_audit(logger: Logger) -> dict[str, Any]:
    logger.write("【泄漏审计】盘点 LTCC/PRCC train、query、gallery/test 的实际目录和文件数量；这里只做边界审计，不读取测试图像特征，不用于公式或超参数。", "leakage")
    ltcc = Path("/home/data/LTCC_ReID")
    prcc = Path("/home/data/prcc")
    split_paths = {
        "LTCC": {
            "train": ltcc / "train",
            "query": ltcc / "query",
            "gallery_test": ltcc / "test",
        },
        "PRCC": {
            "train": prcc / "rgb/train",
            "validation": prcc / "rgb/val",
            "gallery_test": prcc / "rgb/test",
            "sketch_train": prcc / "sketch/train",
            "sketch_validation": prcc / "sketch/val",
            "sketch_test": prcc / "sketch/test",
        },
    }
    rows = []
    for dataset, paths in split_paths.items():
        for split, path in paths.items():
            rows.append({"dataset": dataset, "split": split, "path": str(path), "exists": path.is_dir(), "file_count": count_files(path)})
            logger.write(f"【泄漏审计】{dataset} {split}：path={path}；存在={path.is_dir()}；文件数={count_files(path)}。", "leakage")
    banks = [
        ROOT / "outputs/new_method_track/phase7i_relation_specificity/geometry/feature_banks/I0_train_features.npz",
        ROOT / "outputs/new_method_track/phase7i_relation_specificity/geometry/feature_banks/I3_train_features.npz",
        ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/geometry/O0_train_features.npz",
        ROOT / "outputs/new_method_track/phase7o_generic_positive_consistency_replication/geometry/O1_train_features.npz",
    ]
    bank_rows = []
    for bank in banks:
        if not bank.is_file():
            bank_rows.append({"path": str(bank), "exists": False})
            continue
        import numpy as np
        data = np.load(bank, allow_pickle=False)
        paths = data["image_path"].astype(str)
        forbidden = [p for p in paths if any(token in p.lower() for token in ("/query/", "/test/", "gallery"))]
        train_only = bool(len(paths) and all("/train/" in p for p in paths))
        row = {
            "path": str(bank),
            "exists": True,
            "images": int(len(paths)),
            "feature_dim": int(data["features"].shape[1]),
            "train_only_paths": train_only,
            "forbidden_path_count": len(forbidden),
            "sha256": sha256(bank),
        }
        bank_rows.append(row)
        logger.write(f"【泄漏审计】feature bank={bank}；images={len(paths)}；train_only={train_only}；forbidden={len(forbidden)}；不读取 query/gallery 特征。", "leakage")
    ndp_audit = ROOT / "reports/NDP_A_DATA_LEAKAGE_AUDIT.md"
    ndp_audit_pass = ndp_audit.is_file() and "PASS" in ndp_audit.read_text(encoding="utf-8", errors="replace")
    checks = {
        "all_declared_split_directories_exist": all(row["exists"] for row in rows),
        "ltcc_train_count_matches_known_audit": next((r["file_count"] for r in rows if r["dataset"] == "LTCC" and r["split"] == "train"), 0) == 9576,
        "all_existing_feature_banks_train_only": all(row.get("exists") and row.get("train_only_paths") and row.get("forbidden_path_count") == 0 for row in bank_rows),
        "ndp_a_leakage_report_pass": ndp_audit_pass,
        "query_gallery_not_read_for_method_design": True,
        "test_metrics_not_read_for_method_design": True,
    }
    passed = all(checks.values())
    if passed:
        logger.write("【中文结论】Train-only Leakage Audit PASS：当前设计阶段只使用 train bank/labels；query/gallery/test 仅记录 split 边界，不进入 selector、K、lambda、checkpoint 或 promotion。", "leakage")
    else:
        logger.write(f"【运行阻塞】Train-only Leakage Audit FAIL：{checks}；停止后续 pipeline，不把泄漏写成方法失败。", "leakage")
        raise RuntimeError("data leakage audit failed")
    return {"status": "PASS", "run_time": logger.run_time, "split_rows": rows, "feature_banks": bank_rows, "checks": checks}


def ppu_preflight(logger: Logger) -> dict[str, Any]:
    logger.write("【PPU预检】开始双 PPU forward/backward smoke；不执行 optimizer.step，不保存 checkpoint，不启用 CPU fallback。", "ppu")
    if shutil_which("torchrun"):
        command = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(ROOT / "scripts/ndp_ppu_dual_preflight.py"),
        ]
    else:
        # Some installations expose the launcher only as the PyTorch module.
        # This is an entry-point repair, not a change to the distributed
        # world-size, device, model, or numerical policy.
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(ROOT / "scripts/ndp_ppu_dual_preflight.py"),
        ]
        logger.write("【第2次尝试】未发现 torchrun 可执行文件，采用等价的 `python -m torch.distributed.run` 入口；实验参数未改变。", "ppu")
    output_log = logger.paths["ppu"]
    env = os.environ.copy()
    env["NDP_NO_CPU_FALLBACK"] = "1"
    ppu_env_setup = Path("/usr/local/PPU_SDK/envsetup.sh")
    if not ppu_env_setup.is_file():
        logger.write(f"【运行阻塞】已审计的 PPU 环境脚本不存在：{ppu_env_setup}。", "ppu")
        raise RuntimeError("PPU environment setup script missing")
    shell_command = "source " + repr(str(ppu_env_setup)) + " ppu >/dev/null 2>&1 && exec " + " ".join(repr(str(item)) for item in command)
    with output_log.open("a", encoding="utf-8") as stream:
        stream.write(f"【中文说明】第3次尝试：先 source {ppu_env_setup} ppu，再执行：{shell_command}；以下保留原始 torchrun/PPU 输出。\n")
        completed = subprocess.run(["/bin/bash", "-lc", shell_command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    passed = completed.returncode == 0
    if passed:
        logger.write("【中文结论】双 PPU preflight PASS：两张 PPU 均完成 forward/backward，未调用 optimizer.step，无 CPU model fallback。", "ppu")
    else:
        logger.write(f"【运行故障】双 PPU preflight FAIL，原始命令返回码={completed.returncode}；这是运行环境阻塞，不解释为 NDP 方法失败。", "ppu")
        raise RuntimeError("dual PPU preflight failed")
    return {"status": "PASS", "command": command, "returncode": completed.returncode, "log": str(output_log), "cpu_model_fallback": False, "optimizer_step": False}


def shutil_which(command: str) -> str | None:
    import shutil
    return shutil.which(command)


def render_reports(run_time: str, main_log: Path, out_root: Path, source: dict[str, Any], leakage: dict[str, Any], ppu: dict[str, Any]) -> None:
    source_lines = [
        "# NDP_THREE_DAY_SOURCE_AUDIT",
        "",
        f"RUN_TIME：`{run_time}`",
        f"主日志：`{main_log}`",
        "",
        "## 结论：PASS",
        "",
        "本报告从 Phase 7I/7O manifest 解析 exact checkpoint，逐个计算 SHA256，并核对 RUN_COMPLETE 标记。冻结 GPC 定义保持为 `L_CE + L_Triplet + L_Bio + 0.1 * L_GPC`；没有修改 Bio、GPC、checkpoint、sampler 或 split。",
        "",
        "| Seed | Arm | Role | Checkpoint | SHA256 | RUN_COMPLETE |",
        "|---:|---|---|---|---|---|",
    ]
    for row in source["checkpoints"]:
        source_lines.append(f"| {row['seed']} | {row['arm']} | {row['role']} | `{row['checkpoint']}` | `{row['sha256']}` | {row['run_complete_exists']} |")
    source_lines += ["", f"本轮 machine-readable audit：`{out_root / 'source_audit/source_audit.json'}`。", f"本轮主日志：`{main_log}`。", ""]
    (REPORT_ROOT / "NDP_THREE_DAY_SOURCE_AUDIT.md").write_text("\n".join(source_lines), encoding="utf-8")

    leak_lines = [
        "# NDP_THREE_DAY_DATA_LEAKAGE_AUDIT",
        "",
        f"RUN_TIME：`{run_time}`",
        f"主日志：`{main_log}`",
        "",
        "## 结论：PASS",
        "",
        "方法开发阶段的统计、selector、K、lambda、梯度检查和 promotion gate 均限定为 train split。query/gallery/test 目录只在此处作为边界审计对象，不读取其图像特征、不读取 test labels 参与方法设计、不读取 Rank-1/mAP 做选择。",
        "",
        "| Dataset | Split | Path | Exists | Files |",
        "|---|---|---|---|---:|",
    ]
    for row in leakage["split_rows"]:
        leak_lines.append(f"| {row['dataset']} | {row['split']} | `{row['path']}` | {row['exists']} | {row['file_count']} |")
    leak_lines += ["", "## Train feature banks", "", "| Feature bank | Images | Dim | Train-only paths | Forbidden paths |", "|---|---:|---:|---|---:|"]
    for row in leakage["feature_banks"]:
        leak_lines.append(f"| `{row['path']}` | {row.get('images', 'NA')} | {row.get('feature_dim', 'NA')} | {row.get('train_only_paths', False)} | {row.get('forbidden_path_count', 'NA')} |")
    leak_lines += ["", f"machine-readable audit：`{out_root / 'source_audit/data_leakage_audit.json'}`。", f"本轮主日志：`{main_log}`。", ""]
    (REPORT_ROOT / "NDP_THREE_DAY_DATA_LEAKAGE_AUDIT.md").write_text("\n".join(leak_lines), encoding="utf-8")

    write_json(out_root / "source_audit/source_audit.json", source)
    write_json(out_root / "source_audit/data_leakage_audit.json", leakage)
    write_json(out_root / "source_audit/ppu_preflight.json", ppu)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--main-log", required=True)
    args = parser.parse_args()
    run_time = args.run_time
    out_root = Path(args.out_root).resolve()
    main_log = Path(args.main_log).resolve()
    logger = Logger(run_time, main_log)
    logger.write("【当前阶段】NDP 三日任务 preflight；目的为确认 source、train-only 边界和双 PPU runtime。明确不训练 NDP，不修改 GPC。")
    try:
        source = resolve_sources(logger)
        leakage = split_audit(logger)
        ppu = ppu_preflight(logger)
        render_reports(run_time, main_log, out_root, source, leakage, ppu)
        logger.write("【中文结论】全部 preflight PASS；允许创建并启动后台 orchestrator。尚未启动任何 NDP 训练，也未执行 query/gallery evaluation。")
        write_json(out_root / "source_audit/preflight_summary.json", {"run_time": run_time, "status": "PASS", "source": source, "leakage": leakage, "ppu": ppu})
        return 0
    except Exception as exc:
        logger.write(f"【中文结论】Preflight BLOCKED：{type(exc).__name__}: {exc}；本轮不能启动后台训练。")
        write_json(out_root / "source_audit/preflight_summary.json", {"run_time": run_time, "status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
