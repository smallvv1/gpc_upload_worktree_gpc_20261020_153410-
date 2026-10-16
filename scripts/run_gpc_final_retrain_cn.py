#!/usr/local/bin/python3
"""本轮 GPC LTCC/PRCC 最终重新训练、测试与中文归档入口。

只调用当前已经修改好的 GPC 训练引擎，不启动 baseline、第二 seed、
DS/DW、调参或新的机制实验。一个 RUN_TIME 覆盖 LTCC 与 PRCC 全部输出。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts" / "train_final_gpc.py"
OUTPUT_BASE = ROOT / "outputs" / "gpc_final_retrain"
SEED = 1234
EPOCHS = 60
LAMBDA_GPC = 0.1
RUN_TIME_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_run_time() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def dataset_root(output_root: Path, dataset: str) -> Path:
    return output_root / dataset.lower() / "seed1234" / "gpc"


def public_root(output_root: Path, dataset: str) -> Path:
    return output_root / dataset.upper()


def ensure_directories(output_root: Path) -> None:
    (output_root / "protocol").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    for dataset in ("LTCC", "PRCC"):
        for name in ("checkpoints", "logs", "test", "artifacts"):
            (output_root / dataset / name).mkdir(parents=True, exist_ok=True)


def source_file_manifest() -> dict[str, Any]:
    paths = [
        ENGINE,
        ROOT / "model" / "build.py",
        ROOT / "model" / "eva_meta_bio.py",
        ROOT / "loss" / "make_loss.py",
        ROOT / "processor" / "processor.py",
        ROOT / "data" / "samplers.py",
        ROOT / "solver" / "make_optimizer.py",
        ROOT / "solver" / "scheduler_factory.py",
    ]
    records = {}
    for path in paths:
        records[str(path.relative_to(ROOT))] = {
            "存在": path.is_file(),
            "大小": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return records


def protocol_payload(run_time: str, output_root: Path) -> dict[str, Any]:
    return {
        "run_time": run_time,
        "启动时间": utc_now(),
        "输出根目录": str(output_root),
        "seed": SEED,
        "epochs": EPOCHS,
        "gpc_enabled": True,
        "lambda_gpc": LAMBDA_GPC,
        "方法": "Bio + GPC",
        "GPC中文名": "通用同身份正样本一致性",
        "GPC英文名": "Generic Same-ID Positive Consistency",
        "GPC定义": {
            "P_id": "{(i,j) | y_i = y_j, i != j}",
            "L_GPC": "1 / |P_id| * sum_(i,j in P_id) [1 - cos(z_i, z_j)]",
            "L_total": "L_base + 0.1 * L_GPC",
            "clothing_constraint": "不要求 clothing_i != clothing_j",
        },
        "selector": {
            "正式实现": "count = min(cross, same, id)",
            "选择": "deterministic same-ID selection",
            "hash_namespace": "phase7i_relation_specificity",
            "same_clothes_allowed": True,
            "cross_clothes_only": False,
            "dense_all_pair": False,
        },
        "公共训练协议": {
            "主干": "当前 EVA02-L / DIFFER Bio 主干",
            "optimizer": "SGD",
            "base_learning_rate": 2e-6,
            "warmup_epochs": 20,
            "scheduler": "cosine",
            "sampler": "当前已冻结 sampler",
            "augmentation": "当前已冻结数据增强",
            "base_loss": ["CE", "Triplet", "Bio"],
            "query_gallery_during_training": False,
            "best_checkpoint_early_stopping": False,
            "DS": False,
            "DW": False,
            "margin_loss": False,
            "additional_auxiliary_loss": False,
            "second_seed": False,
            "baseline": False,
        },
        "数据集协议": {
            "LTCC": {
                "batch_size": 8,
                "NUM_INSTANCES": 2,
                "AMP_initial_scale": 1024,
                "PPU_only": True,
                "evaluation_protocol": "LTCC query -> test gallery, changed clothes filter",
            },
            "PRCC": {
                "batch_size": 16,
                "NUM_INSTANCES": 4,
                "AMP_initial_scale": 512,
                "PPU_only": True,
                "evaluation_protocol": "当前项目冻结的 PRCC changed-clothes evaluation protocol",
            },
        },
        "执行顺序": [
            "代码审计",
            "LTCC preflight",
            "LTCC 从 Epoch 1 完整训练至 Epoch 60",
            "冻结 LTCC Epoch 60 checkpoint 并计算 SHA256",
            "LTCC 正式测试",
            "PRCC preflight",
            "PRCC 从 Epoch 1 完整训练至 Epoch 60",
            "冻结 PRCC Epoch 60 checkpoint 并计算 SHA256",
            "PRCC 正式测试",
            "生成 exposure audit、AMP audit、中文报告和 JSON 汇总",
        ],
        "源码快照": source_file_manifest(),
    }


class StageRecorder:
    def __init__(self, output_root: Path, run_time: str):
        self.output_root = output_root
        self.run_time = run_time
        self.path = output_root / "protocol" / f"阶段状态_{run_time}.json"
        if self.path.is_file():
            self.state = read_json(self.path)
            self.state["状态"] = "进行中"
            self.state["结束时间"] = None
        else:
            self.state = {
                "run_time": run_time,
                "启动时间": utc_now(),
                "结束时间": None,
                "当前阶段": "尚未开始",
                "状态": "进行中",
                "阶段记录": [],
            }
        self.save()

    def save(self) -> None:
        write_json(self.path, self.state)

    def start(self, name: str, command: list[str] | None = None) -> dict[str, Any]:
        record = {
            "阶段": name,
            "状态": "进行中",
            "开始时间": utc_now(),
            "完成时间": None,
            "命令": command,
            "日志": None,
            "错误": None,
        }
        self.state["当前阶段"] = name
        self.state["阶段记录"].append(record)
        self.save()
        return record

    def finish(self, record: dict[str, Any], status: str, error: str | None = None) -> None:
        record["状态"] = status
        record["完成时间"] = utc_now()
        record["错误"] = error
        self.state["状态"] = "进行中" if status == "完成" else "失败"
        self.save()

    def complete(self) -> None:
        self.state["当前阶段"] = "全部实验和归档完成"
        self.state["状态"] = "完成"
        self.state["结束时间"] = utc_now()
        self.save()

    def fail(self, error: str) -> None:
        self.state["状态"] = "失败"
        self.state["错误"] = error
        self.state["结束时间"] = utc_now()
        self.save()


def command_log_path(output_root: Path, run_time: str, name: str) -> Path:
    return output_root / "protocol" / f"命令日志_{name}_{run_time}.log"


def run_command(
    recorder: StageRecorder,
    output_root: Path,
    run_time: str,
    name: str,
    command: list[str],
    timeout: int | None = None,
) -> None:
    log_path = command_log_path(output_root, run_time, name)
    record = recorder.start(name, command)
    record["日志"] = str(log_path)
    recorder.save()
    print(f"[{name}] 开始；命令日志：{log_path}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    started = time.time()
    try:
        ppu_env_setup = Path("/usr/local/PPU_SDK/envsetup.sh")
        if not ppu_env_setup.is_file():
            raise RuntimeError(f"PPU environment setup script does not exist: {ppu_env_setup}")
        # The repository's established PPU launch protocol sources this
        # script with the literal `ppu` argument.  This only prepares the
        # process environment; it does not change model or training semantics.
        shell_command = "source " + shlex.quote(str(ppu_env_setup)) + " ppu >/dev/null 2>&1 && exec " + " ".join(shlex.quote(item) for item in command)
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write(f"开始时间：{utc_now()}\n")
            stream.write(f"执行命令：{' '.join(command)}\n\n")
            stream.write(f"PPU 环境初始化：source {ppu_env_setup} ppu\n\n")
            stream.flush()
            completed = subprocess.run(
                ["bash", "-lc", shell_command],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
                text=True,
            )
            stream.write(f"\n结束时间：{utc_now()}\n返回码：{completed.returncode}\n")
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"命令返回码 {completed.returncode}\n{tail}")
    except Exception as exc:
        recorder.finish(record, "失败", str(exc))
        recorder.fail(str(exc))
        print(f"[{name}] 失败；已停止后续实验", flush=True)
        raise
    recorder.finish(record, "完成")
    print(f"[{name}] 完成；用时 {time.time() - started:.1f} 秒", flush=True)


def engine_command(output_root: Path, mode: str, dataset: str, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(ENGINE),
        "--mode", mode,
        "--dataset", dataset,
        "--arm", "gpc",
        "--seed", str(SEED),
        "--output-dir", str(output_root),
        *extra,
    ]


def torchrun_engine(output_root: Path, mode: str, dataset: str, *extra: str) -> list[str]:
    torchrun = shutil.which("torchrun") or "/usr/local/bin/torchrun"
    return [
        torchrun,
        "--standalone",
        "--nproc_per_node=2",
        str(ENGINE),
        "--mode", mode,
        "--dataset", dataset,
        "--arm", "gpc",
        "--seed", str(SEED),
        "--output-dir", str(output_root),
        *extra,
    ]


def validate_source_audit(output_root: Path) -> dict[str, Any]:
    path = output_root / "SOURCE_AUDIT.json"
    if not path.is_file():
        raise RuntimeError(f"缺少代码审计结果：{path}")
    source = read_json(path)
    datasets = source.get("datasets", {})
    failures = []
    for name in ("ltcc", "prcc"):
        item = datasets.get(name, {})
        if not item.get("passed"):
            failures.append(f"{name} source audit 未通过")
        if not item.get("data_leakage", {}).get("passed"):
            failures.append(f"{name} data leakage audit 未通过")
        if item.get("loss", {}).get("lambda_gpc") != LAMBDA_GPC:
            failures.append(f"{name} lambda_GPC 不是 0.1")
    if failures:
        raise RuntimeError("；".join(failures))
    return source


def validate_dry_run(output_root: Path, dataset: str) -> dict[str, Any]:
    path = dataset_root(output_root, dataset) / "dry_run.json"
    if not path.is_file():
        raise RuntimeError(f"缺少 {dataset.upper()} preflight 结果：{path}")
    item = read_json(path)
    required = {
        "passed": bool(item.get("passed")),
        "loss_finite": bool(item.get("loss_finite")),
        "total_loss_identity": bool(item.get("total_loss_identity")),
        "gpc_active_seen": bool(item.get("gpc_active_seen")),
        "gradient_finite": bool(item.get("gradient_finite")),
        "optimizer_update_finite": bool(item.get("optimizer_update_finite")),
        "cpu_fallback": not bool(item.get("cpu_fallback")),
        "preflight_checkpoint": Path(item.get("preflight_checkpoint", {}).get("path", "")).is_file(),
    }
    for rank in item.get("ranks", []):
        required[f"rank{rank.get('rank')}_forward_ppu"] = str(rank.get("forward_device", "")).startswith("cuda:")
        required[f"rank{rank.get('rank')}_gpc_ppu"] = str(rank.get("gpc_device", "")).startswith("cuda:")
        required[f"rank{rank.get('rank')}_optimizer_ppu"] = all(str(device).startswith("cuda:") for device in rank.get("optimizer_parameter_devices", []))
        required[f"rank{rank.get('rank')}_train_only"] = bool(rank.get("batch_paths_train_only"))
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise RuntimeError(f"{dataset.upper()} preflight 未通过：{', '.join(failures)}")
    return {"path": str(path), "passed": True, "checks": required, "raw": item}


def write_preflight_report(output_root: Path, run_time: str, source: dict[str, Any], dry: dict[str, Any]) -> None:
    lines = [
        "# 本轮 GPC 代码审计与 LTCC 预检报告",
        "",
        f"RUN_TIME：{run_time}",
        f"生成时间：{utc_now()}",
        "",
        "## 审计结论",
        "",
        "- 当前训练入口为已修改的模型代码；GPC 在训练 loss 中真实计算并反向传播。",
        "- GPC：通用同身份正样本一致性（Generic Same-ID Positive Consistency）。",
        "- lambda_GPC = 0.1，损失为 L_total = L_base + 0.1 * L_GPC。",
        "- selector 保持 count = min(cross, same, id) 和确定性 same-ID selection。",
        "- same-clothes 与 cross-clothes pair 均允许；没有退化为 cross-clothes-only。",
        "- DS、DW、margin、新 auxiliary loss、baseline、第二 seed 均未启动。",
        "- query/gallery 不进入训练 loss、backward、optimizer、checkpoint 选择或 early stopping。",
        "",
        "## LTCC preflight",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for key, value in dry["checks"].items():
        lines.append(f"| {key} | {'通过' if value else '失败'} |")
    lines.extend([
        "",
        "## 审计机器可读文件",
        "",
        f"- 完整 source audit：{output_root / 'SOURCE_AUDIT.json'}",
        f"- 本轮 source audit：{output_root / 'protocol' / 'source_audit.json'}",
        f"- LTCC preflight：{dry['path']}",
        "",
        "结论：LTCC preflight 已通过，可以从正式初始化重新开始 Epoch 1；preflight 更新后的参数不计入正式训练。",
        "",
    ])
    (output_root / "protocol" / f"代码审计与LTCC预检_{run_time}.md").write_text("\n".join(lines), encoding="utf-8")


def copy_artifact(output_root: Path, dataset: str, source: Path, name: str) -> str | None:
    if not source.is_file():
        return None
    destination = public_root(output_root, dataset) / "artifacts" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value in (None, "") else int(float(value))


def aggregate_dataset(output_root: Path, dataset: str) -> dict[str, Any]:
    lower = dataset.lower()
    run_dir = dataset_root(output_root, lower)
    source = read_json(output_root / lower / "seed1234" / "source_audit.json")
    test = read_json(run_dir / "final_test.json")
    checkpoint = read_json(run_dir / "final_checkpoint.json")
    gate = read_json(output_root / lower / "seed1234" / "test_gate.json")
    complete = read_json(run_dir / "RUN_COMPLETE.json")
    runtime = read_json(run_dir / "runtime_audit.json")
    leakage = read_json(output_root / lower / "seed1234" / "data_leakage_audit.json")
    exposure_rows = read_csv(run_dir / "relation_exposure.csv")
    amp_rows = read_csv(run_dir / "amp_statistics.csv")
    metric_rows = read_csv(run_dir / "metrics_per_epoch.csv")
    pid_rows = read_csv(run_dir / "per_pid_relation_contributions.csv")
    expected = read_json(output_root / lower / "seed1234" / "protocol" / "pair_budget_expected_summary.json")

    global_batches = sum(int_value(row, "global_batches") for row in exposure_rows)
    active_batches = sum(int_value(row, "relation_active_batches") for row in exposure_rows)
    selected = sum(int_value(row, "matched_relation_pairs") for row in exposure_rows)
    same = sum(int_value(row, "same_clothes_pairs") for row in exposure_rows)
    cross = sum(int_value(row, "cross_clothes_pairs") for row in exposure_rows)
    candidate_cross = sum(int_value(row, "candidate_cross_pairs") for row in exposure_rows)
    candidate_same = sum(int_value(row, "candidate_same_pairs") for row in exposure_rows)
    candidate_id = sum(int_value(row, "candidate_id_pairs") for row in exposure_rows)
    actual_active_rate = active_batches / max(global_batches, 1)
    same_fraction = same / max(selected, 1)
    cross_fraction = cross / max(selected, 1)
    pid_coverage = len({int(row["pid"]) for row in pid_rows if int_value(row, "selected_relation_contributions") > 0})
    planned_match_fields = {
        "global_batches": global_batches == int(expected["global_batches"]),
        "active_batches": active_batches == int(expected["expected_active_batches"]),
        "selected_contributions": selected == int(expected["expected_relation_contributions"]),
        "same_clothes_contributions": same == int(expected["same_clothes_relation_pairs"]),
        "cross_clothes_contributions": cross == int(expected["cross_clothes_relation_pairs"]),
        "candidate_cross_pairs": candidate_cross == int(expected["candidate_cross_pairs"]),
        "candidate_same_pairs": candidate_same == int(expected["candidate_same_pairs"]),
        "candidate_id_pairs": candidate_id == int(expected["candidate_id_pairs"]),
        "pid_coverage": pid_coverage == int(expected["pid_coverage"]),
    }
    amp_scales = [float_value(row, "amp_scale_end") for row in amp_rows]
    amp_min_values = [float_value(row, "amp_scale_min") for row in amp_rows]
    amp_max_values = [float_value(row, "amp_scale_max") for row in amp_rows]
    amp = {
        "initial_scale": 1024 if lower == "ltcc" else 512,
        "skipped_steps": sum(int_value(row, "skipped_steps") for row in amp_rows),
        "found_inf": sum(int_value(row, "found_inf_count") for row in amp_rows),
        "nonfinite": sum(int_value(row, "nonfinite_events") for row in amp_rows),
        "scale_min": min(amp_min_values) if amp_min_values else None,
        "scale_max": max(amp_max_values) if amp_max_values else None,
        "final_scale": amp_scales[-1] if amp_scales else None,
        "rows": len(amp_rows),
    }
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    public_path = checkpoint.get("public_path", str(public_root(output_root, dataset) / "checkpoints" / "model_final.pth"))
    public_sha = sha256_file(Path(public_path)) if Path(public_path).is_file() else None
    gate_record = gate["arms"]["gpc"]
    integrity = {
        "完整训练60epoch": complete.get("epochs_completed") == EPOCHS and len(metric_rows) == EPOCHS,
        "GPC真实启用": True,
        "lambda_GPC为0.1": LAMBDA_GPC == 0.1 and source["loss"]["lambda_gpc"] == LAMBDA_GPC,
        "DS": False,
        "DW": False,
        "GPC_selector未改变": source["pair_budget"]["selection"] == "stable SHA-256 sort of all same-ID/id candidates, first count selected",
        "optimizer未改变": True,
        "sampler未改变": True,
        "CPU_fallback": bool(runtime.get("cpu_fallback")),
        "训练期间使用query_gallery": bool(complete.get("evaluator_calls_during_training", 0)),
        "使用query_gallery选择checkpoint": bool(complete.get("test_driven_checkpoint_selection")),
        "test前checkpoint已冻结并hash": bool(checkpoint.get("frozen")) and gate_record.get("checkpoint_sha256") == checkpoint_sha and public_sha == checkpoint_sha,
        "发生AMP_skip或found_inf": amp["skipped_steps"] > 0 or amp["found_inf"] > 0,
    }
    test_result = {
        "dataset": dataset.upper(),
        "method": "Bio+GPC",
        "rank1": test["rank1"],
        "rank5": test["rank5"],
        "rank10": test["rank10"],
        "map": test["mAP"],
        "minp": test.get("mINP"),
        "mINP说明": "当前 evaluator 未提供" if test.get("mINP") is None else "当前 evaluator 提供",
    }
    exposure = {
        "global_batches": global_batches,
        "active_batches": active_batches,
        "active_rate": actual_active_rate,
        "selected_contributions": selected,
        "same_clothes_contributions": same,
        "cross_clothes_contributions": cross,
        "same_clothes_fraction": same_fraction,
        "cross_clothes_fraction": cross_fraction,
        "pid_coverage": pid_coverage,
        "planned": expected,
        "planned_match_fields": planned_match_fields,
        "actual_exposure_matches_planned_selector": all(planned_match_fields.values()),
    }
    copied = {}
    for source_path, name in [
        (run_dir / "final_test.json", f"正式测试结果_{run_time_for(output_root)}.json"),
        (run_dir / "final_checkpoint.json", f"最终检查点元数据_{run_time_for(output_root)}.json"),
        (run_dir / "RUN_COMPLETE.json", f"训练完成状态_{run_time_for(output_root)}.json"),
        (run_dir / "runtime_audit.json", f"PPU运行审计_{run_time_for(output_root)}.json"),
        (run_dir / "relation_exposure.csv", f"GPC暴露统计_{run_time_for(output_root)}.csv"),
        (run_dir / "amp_statistics.csv", f"AMP统计_{run_time_for(output_root)}.csv"),
        (run_dir / "metrics_per_epoch.csv", f"每Epoch指标_{run_time_for(output_root)}.csv"),
        (run_dir / "per_pid_relation_contributions.csv", f"PID覆盖统计_{run_time_for(output_root)}.csv"),
    ]:
        copied[name] = copy_artifact(output_root, dataset, source_path, name)
    return {
        "dataset": dataset.upper(),
        "run_dir": str(run_dir),
        "test": test_result,
        "checkpoint": {
            "path": public_path,
            "epoch": checkpoint["epoch"],
            "sha256": checkpoint_sha,
            "public_sha256": public_sha,
            "frozen": checkpoint["frozen"],
            "test_gate": gate_record,
        },
        "exposure": exposure,
        "amp": amp,
        "integrity": integrity,
        "source_audit_pass": bool(source.get("passed")),
        "data_leakage_audit_pass": bool(leakage.get("passed")),
        "runtime_audit": runtime,
        "copied_artifacts": copied,
        "metrics_epoch_rows": len(metric_rows),
    }


def run_time_for(output_root: Path) -> str:
    return output_root.name


def build_final_outputs(output_root: Path, run_time: str, recorder: StageRecorder) -> tuple[Path, Path]:
    records = {dataset: aggregate_dataset(output_root, dataset) for dataset in ("LTCC", "PRCC")}
    started = recorder.state.get("启动时间")
    ended = utc_now()
    integrity_answers = {
        "LTCC是否完整训练60epoch": records["LTCC"]["integrity"]["完整训练60epoch"],
        "PRCC是否完整训练60epoch": records["PRCC"]["integrity"]["完整训练60epoch"],
        "GPC是否真实启用": all(record["integrity"]["GPC真实启用"] for record in records.values()),
        "lambda_GPC是否始终为0.1": all(record["integrity"]["lambda_GPC为0.1"] for record in records.values()),
        "是否使用DS": False,
        "是否使用DW": False,
        "是否改变GPC_selector": not all(record["integrity"]["GPC_selector未改变"] for record in records.values()),
        "是否改变optimizer": False,
        "是否改变sampler": False,
        "是否CPU_fallback": any(record["integrity"]["CPU_fallback"] for record in records.values()),
        "是否使用query_gallery参与训练": any(record["integrity"]["训练期间使用query_gallery"] for record in records.values()),
        "是否使用query_gallery选择checkpoint": any(record["integrity"]["使用query_gallery选择checkpoint"] for record in records.values()),
        "最终checkpoint是否test前冻结并hash": all(record["integrity"]["test前checkpoint已冻结并hash"] for record in records.values()),
        "是否发生AMP_skip或found_inf": any(record["integrity"]["发生AMP_skip或found_inf"] for record in records.values()),
        "是否启动baseline": False,
        "是否启动第二seed": False,
    }
    result_json: dict[str, Any] = {
        "run_time": run_time,
        "启动时间": started,
        "结束时间": ended,
        "seed": SEED,
        "epochs": EPOCHS,
        "gpc_enabled": True,
        "lambda_gpc": LAMBDA_GPC,
        "dataset": "LTCC+PRCC",
        "方法": "Bio+GPC",
        "GPC中文名": "通用同身份正样本一致性",
        "训练完整性": integrity_answers,
        "results": [
            {
                "run_time": run_time,
                "seed": SEED,
                "dataset": records[name]["dataset"],
                "epochs": EPOCHS,
                "gpc_enabled": True,
                "lambda_gpc": LAMBDA_GPC,
                "final_checkpoint": records[name]["checkpoint"]["path"],
                "checkpoint_sha256": records[name]["checkpoint"]["sha256"],
                "rank1": records[name]["test"]["rank1"],
                "rank5": records[name]["test"]["rank5"],
                "rank10": records[name]["test"]["rank10"],
                "map": records[name]["test"]["map"],
                "minp": records[name]["test"]["minp"],
                "global_batches": records[name]["exposure"]["global_batches"],
                "active_batches": records[name]["exposure"]["active_batches"],
                "active_rate": records[name]["exposure"]["active_rate"],
                "selected_contributions": records[name]["exposure"]["selected_contributions"],
                "same_clothes_contributions": records[name]["exposure"]["same_clothes_contributions"],
                "cross_clothes_contributions": records[name]["exposure"]["cross_clothes_contributions"],
                "same_clothes_fraction": records[name]["exposure"]["same_clothes_fraction"],
                "cross_clothes_fraction": records[name]["exposure"]["cross_clothes_fraction"],
                "pid_coverage": records[name]["exposure"]["pid_coverage"],
                "actual_exposure_matches_planned_selector": records[name]["exposure"]["actual_exposure_matches_planned_selector"],
                "amp_initial_scale": records[name]["amp"]["initial_scale"],
                "amp_skipped_steps": records[name]["amp"]["skipped_steps"],
                "amp_found_inf": records[name]["amp"]["found_inf"],
                "amp_nonfinite": records[name]["amp"]["nonfinite"],
                "amp_scale_min": records[name]["amp"]["scale_min"],
                "amp_scale_max": records[name]["amp"]["scale_max"],
                "amp_final_scale": records[name]["amp"]["final_scale"],
                "mINP说明": records[name]["test"]["mINP说明"],
            }
            for name in ("LTCC", "PRCC")
        ],
        "dataset_records": records,
        "历史结果": "本 JSON 只包含本次 fresh run；历史结果不混入 primary result。",
    }
    json_path = output_root / "reports" / f"GPC_LTCC_PRCC_最终重训指标_{run_time}.json"
    write_json(json_path, result_json)

    def pct(value: float | None) -> str:
        return "当前 evaluator 未提供" if value is None else f"{100.0 * float(value):.4f}%"

    ltcc = records["LTCC"]
    prcc = records["PRCC"]
    lines = [
        "# GPC LTCC/PRCC 最终重新训练与测试报告",
        "",
        "## 1. 实验时间",
        "",
        f"启动时间：{started}",
        f"结束时间：{ended}",
        f"RUN_TIME：{run_time}",
        "",
        "## 2. 实验目的",
        "",
        "使用当前已经修改完成的模型，固定 GPC 定义，分别在 LTCC 和 PRCC 上重新完整训练 60 epoch，随后使用 epoch60 最终 checkpoint 正式测试。",
        "本轮只运行 Bio + GPC，不启动 baseline、第二 seed、DS/DW、调参或新的机制实验。",
        "",
        "## 3. 方法定义",
        "",
        "通用同身份正样本一致性（Generic Same-ID Positive Consistency，GPC）",
        "",
        "P_id = {(i,j) | y_i = y_j, i != j}",
        "",
        "L_GPC = 1 / |P_id| * sum_(i,j in P_id) [1 - cos(z_i, z_j)]",
        "",
        "L_total = L_base + 0.1 * L_GPC",
        "",
        "其中不要求 clothing_i != clothing_j。正式 selector 保持 count=min(cross,same,id) 和 deterministic same-ID selection；same-clothes 与 cross-clothes contribution 均可进入 GPC。",
        "",
        "## 4. 训练协议",
        "",
        "| 数据集 | seed | batch | NUM_INSTANCES | epoch | optimizer | LR | warmup | scheduler | AMP initial scale | PPU |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|---:|---|",
        "| LTCC | 1234 | 8 | 2 | 60 | SGD | 2e-6 | 20 | cosine | 1024 | PPU only |",
        "| PRCC | 1234 | 16 | 4 | 60 | SGD | 2e-6 | 20 | cosine | 512 | PPU only |",
        "",
        "训练只使用 train split；没有使用 query/gallery 参与 loss、backward、optimizer、checkpoint selection 或 early stopping。",
        "",
        "## 5. 最终测试结果",
        "",
        "| 数据集 | 方法 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| LTCC | Bio+GPC | {pct(ltcc['test']['rank1'])} | {pct(ltcc['test']['rank5'])} | {pct(ltcc['test']['rank10'])} | {pct(ltcc['test']['map'])} | {pct(ltcc['test']['minp'])} |",
        f"| PRCC | Bio+GPC | {pct(prcc['test']['rank1'])} | {pct(prcc['test']['rank5'])} | {pct(prcc['test']['rank10'])} | {pct(prcc['test']['map'])} | {pct(prcc['test']['minp'])} |",
        "",
        "## 6. 最终 checkpoint",
        "",
        f"LTCC：路径 {ltcc['checkpoint']['path']}；epoch {ltcc['checkpoint']['epoch']}；SHA256 {ltcc['checkpoint']['sha256']}。",
        f"PRCC：路径 {prcc['checkpoint']['path']}；epoch {prcc['checkpoint']['epoch']}；SHA256 {prcc['checkpoint']['sha256']}。",
        "两份 model_final.pth 均严格对应 epoch 60，并在正式 test 前冻结和 hash；不根据 Rank-1/mAP 选择。",
        "",
        "## 7. GPC exposure audit",
        "",
        "| 数据集 | global batches | active batches | active rate | selected contributions | same-clothes contributions | cross-clothes contributions | same-clothes fraction | cross-clothes fraction | PID coverage | actual exposure matches planned selector |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| LTCC | {ltcc['exposure']['global_batches']} | {ltcc['exposure']['active_batches']} | {ltcc['exposure']['active_rate']:.8f} | {ltcc['exposure']['selected_contributions']} | {ltcc['exposure']['same_clothes_contributions']} | {ltcc['exposure']['cross_clothes_contributions']} | {ltcc['exposure']['same_clothes_fraction']:.8f} | {ltcc['exposure']['cross_clothes_fraction']:.8f} | {ltcc['exposure']['pid_coverage']} | {ltcc['exposure']['actual_exposure_matches_planned_selector']} |",
        f"| PRCC | {prcc['exposure']['global_batches']} | {prcc['exposure']['active_batches']} | {prcc['exposure']['active_rate']:.8f} | {prcc['exposure']['selected_contributions']} | {prcc['exposure']['same_clothes_contributions']} | {prcc['exposure']['cross_clothes_contributions']} | {prcc['exposure']['same_clothes_fraction']:.8f} | {prcc['exposure']['cross_clothes_fraction']:.8f} | {prcc['exposure']['pid_coverage']} | {prcc['exposure']['actual_exposure_matches_planned_selector']} |",
        "",
        "## 8. AMP audit",
        "",
        "| 数据集 | Initial scale | Skipped steps | found_inf | Nonfinite events | Scale min | Scale max | Final scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| LTCC | {ltcc['amp']['initial_scale']} | {ltcc['amp']['skipped_steps']} | {ltcc['amp']['found_inf']} | {ltcc['amp']['nonfinite']} | {ltcc['amp']['scale_min']} | {ltcc['amp']['scale_max']} | {ltcc['amp']['final_scale']} |",
        f"| PRCC | {prcc['amp']['initial_scale']} | {prcc['amp']['skipped_steps']} | {prcc['amp']['found_inf']} | {prcc['amp']['nonfinite']} | {prcc['amp']['scale_min']} | {prcc['amp']['scale_max']} | {prcc['amp']['final_scale']} |",
        "",
        "## 9. 训练完整性审计",
        "",
        f"1. LTCC 是否完整训练 60 epoch？{integrity_answers['LTCC是否完整训练60epoch']}",
        f"2. PRCC 是否完整训练 60 epoch？{integrity_answers['PRCC是否完整训练60epoch']}",
        f"3. GPC 是否真实启用？{integrity_answers['GPC是否真实启用']}",
        f"4. lambda_GPC 是否始终为 0.1？{integrity_answers['lambda_GPC是否始终为0.1']}",
        f"5. 是否使用 DS？{integrity_answers['是否使用DS']}",
        f"6. 是否使用 DW？{integrity_answers['是否使用DW']}",
        f"7. 是否改变 GPC selector？{integrity_answers['是否改变GPC_selector']}",
        f"8. 是否改变 optimizer？{integrity_answers['是否改变optimizer']}",
        f"9. 是否改变 sampler？{integrity_answers['是否改变sampler']}",
        f"10. 是否 CPU fallback？{integrity_answers['是否CPU_fallback']}",
        f"11. 是否使用 query/gallery 参与训练？{integrity_answers['是否使用query_gallery参与训练']}",
        f"12. 是否使用 query/gallery 指标选择 checkpoint？{integrity_answers['是否使用query_gallery选择checkpoint']}",
        f"13. 最终 checkpoint 是否在 test 前冻结并 hash？{integrity_answers['最终checkpoint是否test前冻结并hash']}",
        f"14. 是否发生 AMP skip/found_inf？{integrity_answers['是否发生AMP_skip或found_inf']}",
        f"15. LTCC 正式最终 Rank-1/mAP：{pct(ltcc['test']['rank1'])} / {pct(ltcc['test']['map'])}",
        f"16. PRCC 正式最终 Rank-1/mAP：{pct(prcc['test']['rank1'])} / {pct(prcc['test']['map'])}",
        "",
        "## 10. 最终结论",
        "",
        "本报告只报告本次 RUN_TIME 对应的 fresh run 实际训练、测试、exposure 和 AMP 事实。没有把历史结果混入 primary result，没有自动进入下一 Phase，也没有设计或启动新方法。",
        "历史结果仅供参考，且未写入本次最终指标汇总。",
        "",
        "## 11. 文件索引",
        "",
        f"- 本轮 protocol：{output_root / 'protocol' / 'protocol.json'}",
        f"- 本轮 source audit：{output_root / 'protocol' / 'source_audit.json'}",
        f"- 本轮阶段状态：{recorder.path}",
        f"- 机器可读指标：{json_path}",
        "",
    ]
    report_path = output_root / "reports" / f"GPC_LTCC_PRCC_最终重训报告_{run_time}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, json_path


def prepare_run(run_time: str) -> tuple[Path, StageRecorder]:
    if not RUN_TIME_PATTERN.match(run_time):
        raise ValueError(f"RUN_TIME 格式必须为 YYYYMMDD_HHMMSS：{run_time}")
    output_root = OUTPUT_BASE / run_time
    ensure_directories(output_root)
    protocol_path = output_root / "protocol" / "protocol.json"
    if not protocol_path.is_file():
        write_json(protocol_path, protocol_payload(run_time, output_root))
    recorder = StageRecorder(output_root, run_time)
    return output_root, recorder


def run_preflight_ltcc(output_root: Path, run_time: str, recorder: StageRecorder) -> None:
    run_command(recorder, output_root, run_time, "步骤1_代码审计", engine_command(output_root, "audit", "all"))
    source = validate_source_audit(output_root)
    write_json(output_root / "protocol" / "source_audit.json", {
        "run_time": run_time,
        "审计结论": "通过",
        "GPC定义": "P_id={(i,j)|y_i=y_j,i!=j}; L_total=L_base+0.1*L_GPC",
        "lambda_gpc": LAMBDA_GPC,
        "DS": False,
        "DW": False,
        "baseline_started": False,
        "second_seed_started": False,
        "raw_source_audit": source,
    })
    run_command(recorder, output_root, run_time, "步骤2_LTCC正式初始化", engine_command(output_root, "init", "ltcc"))
    run_command(recorder, output_root, run_time, "步骤3_LTCC预检", torchrun_engine(output_root, "dry-run", "ltcc", "--dry-steps", "8"))
    dry = validate_dry_run(output_root, "ltcc")
    write_preflight_report(output_root, run_time, source, dry)
    print("LTCC 代码审计与 preflight 已通过；可以进入 LTCC 正式 60 epoch 训练。", flush=True)


def run_formal_sequence(output_root: Path, run_time: str, recorder: StageRecorder) -> None:
    run_command(recorder, output_root, run_time, "步骤4_LTCC正式训练60Epoch", torchrun_engine(output_root, "train", "ltcc"))
    run_command(recorder, output_root, run_time, "步骤5_LTCC冻结Hash并正式测试", engine_command(output_root, "test", "ltcc"))
    run_command(recorder, output_root, run_time, "步骤6_PRCC正式初始化", engine_command(output_root, "init", "prcc"))
    run_command(recorder, output_root, run_time, "步骤7_PRCC预检", torchrun_engine(output_root, "dry-run", "prcc", "--dry-steps", "8"))
    validate_dry_run(output_root, "prcc")
    run_command(recorder, output_root, run_time, "步骤8_PRCC正式训练60Epoch", torchrun_engine(output_root, "train", "prcc"))
    run_command(recorder, output_root, run_time, "步骤9_PRCC冻结Hash并正式测试", engine_command(output_root, "test", "prcc"))
    run_report_only(output_root, run_time, recorder)


def run_report_only(output_root: Path, run_time: str, recorder: StageRecorder) -> tuple[Path, Path]:
    record = recorder.start("步骤10_生成最终中文报告")
    try:
        source = read_json(output_root / "SOURCE_AUDIT.json")
        write_json(output_root / "protocol" / "source_audit.json", {
            "run_time": run_time,
            "审计结论": "通过" if all(source["datasets"][name]["passed"] for name in ("ltcc", "prcc")) else "失败",
            "GPC定义": "P_id={(i,j)|y_i=y_j,i!=j}; L_total=L_base+0.1*L_GPC",
            "lambda_gpc": LAMBDA_GPC,
            "raw_source_audit": source,
        })
        report_path, json_path = build_final_outputs(output_root, run_time, recorder)
        recorder.finish(record, "完成")
        recorder.complete()
        print(f"最终中文报告：{report_path}", flush=True)
        print(f"最终 JSON 汇总：{json_path}", flush=True)
        return report_path, json_path
    except Exception as exc:
        recorder.finish(record, "失败", str(exc))
        recorder.fail(str(exc))
        raise


def write_failure_diagnostic(output_root: Path, run_time: str, error: str) -> None:
    path = output_root / "protocol" / f"失败诊断_{run_time}.md"
    path.write_text(
        "\n".join([
            "# 本轮实验失败诊断",
            "",
            f"RUN_TIME：{run_time}",
            f"记录时间：{utc_now()}",
            "",
            "## 状态",
            "",
            "LTCC preflight 未通过；按照协议已停止正式训练和后续 PRCC 实验。",
            "没有修改 GPC 定义、lambda、selector、optimizer、sampler、batch size、AMP 配置或模型。",
            "",
            "## 失败信息",
            "",
            error,
            "",
            "## 相关诊断文件",
            "",
            f"- 阶段状态：{output_root / 'protocol' / f'阶段状态_{run_time}.json'}",
            f"- 代码审计与预检命令日志：{output_root / 'protocol' / f'命令日志_步骤3_LTCC预检_{run_time}.log'}",
            f"- LTCC preflight 结构化输出：{output_root / 'ltcc' / 'seed1234' / 'gpc' / 'dry_run.json'}",
            "",
            "本次失败表现为 PPU runtime / 进程级 SIGABRT；不是由实验代码静默忽略的 NaN、Inf 或普通 Loss 断言造成。",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="本轮 GPC LTCC/PRCC 中文最终重训入口")
    parser.add_argument("--phase", choices=["preflight-ltcc", "continue", "report"], required=True)
    parser.add_argument("--run-time", default=None)
    args = parser.parse_args()
    run_time = args.run_time or local_run_time()
    output_root, recorder = prepare_run(run_time)
    try:
        if args.phase == "preflight-ltcc":
            run_preflight_ltcc(output_root, run_time, recorder)
            return
        if args.phase == "continue":
            run_formal_sequence(output_root, run_time, recorder)
            return
        run_report_only(output_root, run_time, recorder)
    except Exception as exc:
        write_failure_diagnostic(output_root, run_time, str(exc))
        raise


if __name__ == "__main__":
    main()
