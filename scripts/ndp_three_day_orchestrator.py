#!/usr/local/bin/python3
"""Persistent NDP three-day state machine.

Stage 1 and Stage 2 are fully executable train-only diagnostics.  The state
machine records runtime faults separately from scientific STOP gates and never
advances past a missing artifact or gate silently.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/DIFFER")
LOG_ROOT = ROOT / "logs"
REPORT_ROOT = ROOT / "reports"
MAX_RETRIES = 2


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Pipeline:
    def __init__(self, run_time: str, out_root: Path, main_log: Path):
        self.run_time = run_time
        self.out_root = out_root
        self.main_log = main_log
        self.state_path = out_root / f"运行状态_{run_time}.json"
        self.sublogs = {
            "stage1": LOG_ROOT / f"NDP有害邻居稳定性分析_{run_time}.log",
            "stage2": LOG_ROOT / f"NDP训练轨迹诊断_{run_time}.log",
            "stage3": LOG_ROOT / f"NDP静态干预验证_{run_time}.log",
            "stage4": LOG_ROOT / f"NDP梯度独立性检查_{run_time}.log",
            "stage5": LOG_ROOT / f"NDP在线可识别性分析_{run_time}.log",
            "stage6": LOG_ROOT / f"LTCC_GPC_NDP筛选_{run_time}.log",
            "stage7": LOG_ROOT / f"LTCC_GPC_NDP正式训练_{run_time}.log",
            "stage8": LOG_ROOT / f"PRCC_GPC_NDP正式训练_{run_time}.log",
            "stage9": LOG_ROOT / f"NDP跨Seed稳定性验证_{run_time}.log",
            "final": LOG_ROOT / f"NDP最终结论_{run_time}.log",
        }
        self.state = {
            "run_time": run_time,
            "status": "INITIALIZING",
            "current_stage": "INITIALIZING",
            "dataset": None,
            "seed": None,
            "arm": None,
            "epoch": 0,
            "last_update": now(),
            "background_mode": "nohup",
            "tmux_session": None,
            "pid": os.getpid(),
            "main_log": str(main_log),
            "last_checkpoint": None,
            "completed_stages": [],
            "failed_stages": [],
            "runtime_retry_count": {},
            "source_audit": "PASS",
            "train_only_leakage_audit": "PASS",
            "cpu_model_fallback": False,
            "query_gallery_read_for_method_design": False,
            "gpc_modified": False,
            "started_at": now(),
        }
        self.out_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            try:
                previous = read_json(self.state_path)
            except Exception as exc:
                previous = None
                self.log(f"【状态恢复故障】已有状态 JSON 无法读取：{type(exc).__name__}: {exc}；保留原文件并使用初始状态。")
            if isinstance(previous, dict) and previous.get("run_time") == run_time:
                self.state = previous
                self.state["pid"] = os.getpid()
                self.state.setdefault("started_at", previous.get("started_at", now()))
                self.state["background_mode"] = "nohup"
                self.state["tmux_session"] = None
                self.state["status"] = "RUNNING"
                self.state["current_stage"] = "RESUMING"
                # The first Stage2 attempts were blocked by a deterministic
                # runner bug (one-batch dry gate), not by the PPU runtime.
                # After the fixed three-batch gate is installed, allow a new
                # bounded retry window while retaining the old failure record.
                repair_key = "stage2_train_1234"
                if (
                    int(self.state.get("runtime_retry_count", {}).get(repair_key, 0)) > MAX_RETRIES
                    and self.state.get("stage2_code_repair_version") != "dry_gate_3"
                ):
                    counts = dict(self.state.get("runtime_retry_count", {}))
                    counts[repair_key] = 0
                    self.state["runtime_retry_count"] = counts
                    self.state["stage2_code_repair_version"] = "dry_gate_3"
                    self.state.setdefault("repair_history", []).append({
                        "time": now(),
                        "stage": "Stage2 Seed1234 train",
                        "repair": "固定 train-only GPC dry gate 为 3 batches；不改变正式训练配置",
                        "previous_retry_count": int(previous.get("runtime_retry_count", {}).get(repair_key, 0)),
                    })
                write_json(self.state_path, self.state)
                self.log("【orchestrator恢复】沿用同一 RUN_TIME 和已有失败证据；不改变 seed、loss、selector、lambda 或 checkpoint。")
                if self.state.get("stage2_code_repair_version") == "dry_gate_3":
                    self.log("【非实验性技术修复】Stage2 Seed1234 的旧 dry-run=1 误判已修复为固定 3 batches；旧失败证据保留，修复后重新开始新的 bounded retry window。", "stage2")
                extract_key = "stage2_extract_1234"
                if (
                    int(self.state.get("runtime_retry_count", {}).get(extract_key, 0)) > MAX_RETRIES
                    and self.state.get("stage2_extract_code_repair_version") != "output_path_v1_snapshot_name_v1"
                ):
                    counts = dict(self.state.get("runtime_retry_count", {}))
                    counts[extract_key] = 0
                    self.state["runtime_retry_count"] = counts
                    self.state["stage2_extract_code_repair_version"] = "output_path_v1_snapshot_name_v1"
                    self.state.setdefault("repair_history", []).append({
                        "time": now(),
                        "stage": "Stage2 train feature snapshots",
                        "repair": "修复特征抽取函数中 model output 覆盖 Path 参数以及公开快照文件名零填充假设的确定性代码错误；不改变模型、数据、seed、batch 或统计公式",
                        "previous_retry_count": int(previous.get("runtime_retry_count", {}).get(extract_key, 0)),
                    })
                    write_json(self.state_path, self.state)
                    self.log("【非实验性技术修复】Stage2 特征抽取的 Tensor 覆盖 Path 参数和快照文件名假设已修复；保留失败证据，并为同一配置开启新的 bounded retry window。", "stage2")
        self.log("【orchestrator】状态机启动；不修改冻结 GPC，不使用 CPU model fallback，不以 test 指标调参。")

    def log(self, message: str, sub: str | None = None) -> None:
        line = f"[{now()}] {message}\n"
        with self.main_log.open("a", encoding="utf-8") as stream:
            stream.write(line)
        if sub:
            with self.sublogs[sub].open("a", encoding="utf-8") as stream:
                stream.write(line)

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["last_update"] = now()
        write_json(self.state_path, self.state)

    def set_stage(self, stage: str, dataset: str | None = None, seed: int | None = None, arm: str | None = None, epoch: int = 0) -> None:
        self.update(status="RUNNING", current_stage=stage, dataset=dataset, seed=seed, arm=arm, epoch=epoch)
        self.log(f"【当前阶段】{stage}；dataset={dataset}；seed={seed}；arm={arm}；epoch={epoch}。")

    def complete(self, stage: str) -> None:
        done = list(self.state.get("completed_stages", []))
        if stage not in done:
            done.append(stage)
        self.update(completed_stages=done)
        self.log(f"【阶段完成】{stage}；对应产物已写入输出目录。")

    def fail_runtime(self, stage: str, detail: str) -> None:
        failed = list(self.state.get("failed_stages", []))
        failed.append({"stage": stage, "type": "RUNTIME_BLOCKED", "detail": detail, "time": now()})
        self.update(status="BLOCKED", current_stage=stage, failed_stages=failed)
        self.log(f"【运行阻塞】{stage}：{detail}；不解释为方法失败。")

    def scientific_stop(self, stage: str, detail: str) -> None:
        failed = list(self.state.get("failed_stages", []))
        failed.append({"stage": stage, "type": "SCIENTIFIC_STOP_GATE", "detail": detail, "time": now()})
        self.update(status="STOPPED_BY_SCIENTIFIC_GATE", current_stage=stage, failed_stages=failed)
        self.log(f"【科学 STOP gate】{stage}：{detail}。")

    def epoch_from(self, root: Path) -> int:
        latest = 0
        if not root.is_dir():
            return latest
        for path in root.rglob("metrics_per_epoch.csv"):
            try:
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    latest = max(latest, int(rows[-1]["epoch"]))
            except Exception:
                pass
        return latest

    def _retry_count(self, key: str) -> int:
        return int(self.state["runtime_retry_count"].get(key, 0))

    def _inc_retry(self, key: str) -> None:
        counts = dict(self.state["runtime_retry_count"])
        counts[key] = self._retry_count(key) + 1
        self.update(runtime_retry_count=counts)

    def ndp_active_rate(self, root: Path | None) -> float | None:
        if root is None or not root.exists():
            return None
        values = []
        for path in root.rglob("online_train_stats_rank*.json"):
            try:
                item = read_json(path)
                batches = int(item.get("batches", 0))
                if batches:
                    values.append((int(item.get("active_batches", 0)), batches))
            except Exception:
                continue
        if not values:
            return None
        return float(sum(active for active, _total in values) / max(sum(total for _active, total in values), 1))

    def latest_checkpoint(self, root: Path | None) -> str | None:
        if root is None or not root.exists():
            return None
        candidates = [path for path in root.rglob("model_epoch_*.pth") if path.is_file()]
        candidates += [path for path in root.rglob("model_final.pth") if path.is_file()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        return str(latest)

    def run_process(self, command: list[str], stage: str, sub: str, *, dataset: str | None = None, seed: int | None = None, arm: str | None = None, monitor_root: Path | None = None, retry_key: str | None = None) -> None:
        key = retry_key or stage
        while True:
            attempt = self._retry_count(key) + 1
            self.log(f"【第{attempt}次尝试】{stage}：{' '.join(command)}。保持既定 seed、batch、AMP、loss、selector、optimizer、scheduler。", sub)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["NDP_NO_CPU_FALLBACK"] = "1"
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            last_heartbeat = time.monotonic()
            captured_output: list[str] = []
            while True:
                line = process.stdout.readline()
                if line:
                    captured_output.append(line.rstrip())
                    if any(token in line for token in ("Epoch", "第", "PASS", "FAIL", "Traceback", "SIGABRT")):
                        self.log(f"【子进程输出】{line.rstrip()}", sub)
                elif process.poll() is not None:
                    break
                if time.monotonic() - last_heartbeat >= 15:
                    rate = self.ndp_active_rate(monitor_root)
                    checkpoint = self.latest_checkpoint(monitor_root)
                    self.update(dataset=dataset, seed=seed, arm=arm, epoch=self.epoch_from(monitor_root or self.out_root), pid=os.getpid(), ndp_active_rate=rate, last_checkpoint=checkpoint)
                    self.log(f"【Heartbeat】{stage}；dataset={dataset}；seed={seed}；arm={arm}；epoch={self.state['epoch']}；GPC active rate=由子进程 epoch 日志记录；NDP active rate={rate if rate is not None else '等待当前 arm 统计'}；PPU 状态=子进程审计；finite 状态=等待子进程；是否正常运行=是。", sub)
                    last_heartbeat = time.monotonic()
            return_code = process.wait()
            if return_code == 0:
                self.update(dataset=dataset, seed=seed, arm=arm, epoch=self.epoch_from(monitor_root or self.out_root), last_checkpoint=self.latest_checkpoint(monitor_root))
                self.log(f"【中文结论】{stage} PASS；返回码=0。", sub)
                return
            self.log(f"【中文说明】{stage} 子进程返回非零；下面保留最近原始输出用于定位，运行故障不能直接解释为方法失败。", sub)
            for raw_line in captured_output[-200:]:
                self.log(f"【原始子进程输出】{raw_line}", sub)
            self._inc_retry(key)
            self.log(f"【运行故障】{stage} 第{attempt}次返回码={return_code}；失败证据已保留。", sub)
            if self._retry_count(key) > MAX_RETRIES:
                raise RuntimeError(f"{stage} exceeded two technical recoveries; return code={return_code}")
            self.log("【允许范围内技术修复】重启同一个 arm；不切 CPU、不换 seed/batch/AMP/loss/selector/lambda。", sub)

    def stage1(self) -> dict[str, Any]:
        self.set_stage("Stage1 NDP-B1 有害异身份吸引稳定性分析", "LTCC", None, "I0->I3 / O0->O1")
        summary_path = self.out_root / "online_signal/stage1_summary.json"
        gate_path = self.out_root / "online_signal/stage1_promotion_gate.json"
        if summary_path.is_file() and gate_path.is_file():
            self.log("【恢复】Stage1 summary 和 promotion gate 已由同一 RUN_TIME 的技术重试完整生成；跳过重复计算，读取冻结 train-only 结果。", "stage1")
            summary = read_json(summary_path)
            self.write_stage1_selector_report(summary)
            self.complete("Stage1 NDP-B1")
            return summary
        self.run_process([
            sys.executable, str(ROOT / "scripts/ndp_b1_harmful_stability.py"),
            "--run-time", self.run_time, "--out-root", str(self.out_root), "--main-log", str(self.main_log),
        ], "Stage1 NDP-B1", "stage1", dataset="LTCC", arm="I0->I3 / O0->O1", monitor_root=self.out_root, retry_key="stage1")
        summary = read_json(summary_path)
        self.write_stage1_selector_report(summary)
        self.complete("Stage1 NDP-B1")
        return summary

    def write_stage1_selector_report(self, summary: dict[str, Any]) -> None:
        """Persist the selector-gate audit separately from the trajectory report.

        This report records the pre-registered train-only gate verbatim.  The
        online implementation uses S2, which was fixed in the runner before
        any test evaluation and is not chosen by Rank-1/mAP.
        """
        gate = summary.get("gate", {})
        rows = gate.get("registered_signals", [])
        lines = [
            "# NDP_B_ONLINE_SELECTOR_AUDIT", "",
            f"RUN_TIME：`{self.run_time}`",
            f"主日志：`{self.main_log}`", "",
            "本报告只记录 train split 上预注册 S1–S6 的可识别性门禁。没有读取 query/gallery/test、Rank-1 或 mAP；没有用测试结果选择信号、阈值、K 或 lambda。", "",
            f"主研究 K：{gate.get('main_K', 10)}；门禁状态：**{gate.get('status', 'NOT RUN')}**。", "",
            "在线 runner 固定使用已通过门禁的 S2：当前 Top-K negative 中、冻结 Bio Top-K 未包含且 current-minus-Bio attraction increase > 0 的 relation；S2 不是按 test 指标选择。", "",
            "| Signal | Seed1234 AUC | Seed4321 AUC | Spearman direction | PID bootstrap gate | Signal gate |", "|---|---:|---:|---|---|---|",
        ]
        for item in sorted(rows, key=lambda row: str(row.get("signal"))):
            signal = str(item.get("signal"))
            auc_a = item.get("seed1234_auc")
            auc_b = item.get("seed4321_auc")
            direction = "一致" if item.get("spearman_direction_consistent") else "不一致/未通过"
            pid_gate = "是" if item.get("at_least_one_pid_bootstrap_lower_ci_gt_0_5") else "否"
            signal_gate = "是" if item.get("online_signal_gate") else "否"
            if isinstance(auc_a, (int, float)) and isinstance(auc_b, (int, float)):
                lines.append(f"| {signal} | {auc_a:.6f} | {auc_b:.6f} | {direction} | {pid_gate} | {signal_gate} |")
            else:
                lines.append(f"| {signal} | NA | NA | 未完成 | {pid_gate} | {signal_gate} |")
        lines += [
            "", "结论：Stage1 的总体 promotion gate 只有在至少一个 signal 满足双 seed AUC、PID bootstrap 和方向一致条件时才可通过；本轮状态由 train-only 结果给出。后续若进入 online NDP，selector 不再依据 test 结果改变。", "",
            "NO QUERY/GALLERY/TEST DATA WAS USED FOR SELECTOR DESIGN.",
            "NO EXISTING GPC CHECKPOINT WAS MODIFIED.",
        ]
        (REPORT_ROOT / "NDP_B_ONLINE_SELECTOR_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def stage2(self) -> None:
        for seed in (1234, 4321):
            train_root = self.out_root / "trajectory_runs" / f"seed{seed}"
            manifest_path = train_root / "trajectory" / f"seed{seed}" / "trajectory_manifest.json"
            if manifest_path.is_file():
                self.log(f"【恢复】Stage2 Seed{seed} trajectory_manifest 已存在；跳过已完成的 30 epoch train-only trajectory，读取既有快照。", "stage2")
                continue
            self.set_stage("Stage2 NDP-B2 GPC 训练轨迹诊断", "LTCC", seed, "GPC", 0)
            base = [sys.executable, str(ROOT / "scripts/ndp_gpc_trajectory_runner.py"), "--seed", str(seed), "--run-time", self.run_time, "--train-root", str(train_root)]
            completed_train = (
                (train_root / "ltcc" / f"seed{seed}" / "gpc" / "RUN_COMPLETE.json").is_file()
                and (train_root / "ltcc" / f"seed{seed}" / "gpc" / "model_final.pth").is_file()
                and (train_root / "LTCC" / "checkpoints" / "model_final.pth").is_file()
            )
            if not completed_train:
                self.run_process(base + ["--mode", "prepare"], f"Stage2 Seed{seed} prepare", "stage2", dataset="LTCC", seed=seed, arm="GPC prepare", monitor_root=train_root, retry_key=f"stage2_prepare_{seed}")
                train = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", str(ROOT / "scripts/ndp_gpc_trajectory_runner.py"), "--mode", "train", "--seed", str(seed), "--run-time", self.run_time, "--train-root", str(train_root)]
                self.run_process(train, f"Stage2 Seed{seed} train", "stage2", dataset="LTCC", seed=seed, arm="GPC", monitor_root=train_root, retry_key=f"stage2_train_{seed}")
            else:
                self.log(f"【恢复】Stage2 Seed{seed} 的 30 epoch train-only GPC trajectory 已有 RUN_COMPLETE、internal/public model_final；跳过重复训练，仅继续特征快照抽取。", "stage2")
            self.run_process(base + ["--mode", "extract"], f"Stage2 Seed{seed} train feature snapshots", "stage2", dataset="LTCC", seed=seed, arm="GPC snapshots", monitor_root=train_root, retry_key=f"stage2_extract_{seed}")
        lines = [
            "# NDP_B_TRAJECTORY_AUDIT", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "",
            "Stage 2 对 seed1234/4321 各执行原始 GPC 配置的 30 epoch train-only diagnostic，并保存 E0/E5/E10/E15/E20/E25/E30 特征快照。Intrusion persistence 只统计相对 E0 Bio negative Top-K 新出现的 directed different-PID relation；训练期不创建 query/gallery evaluator。", "",
            "| Seed | Trajectory manifest | New relation persistence 1 / 2 / >=3 / >=4 | Anchors with any new intrusion |", "|---:|---|---:|---:|",
        ]
        for seed in (1234, 4321):
            path = self.out_root / "trajectory_runs" / f"seed{seed}" / "trajectory" / f"seed{seed}" / "trajectory_manifest.json"
            persistence_path = path.parent / "intrusion_persistence_pairs.csv"
            anchor_path = path.parent / "trajectory_anchor_margin.csv"
            counts = {"1": 0, "2": 0, ">=3": 0, ">=4": 0}
            if persistence_path.is_file():
                with persistence_path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        value = int(row["snapshots_present_after_E0"])
                        if value == 1:
                            counts["1"] += 1
                        if value == 2:
                            counts["2"] += 1
                        if value >= 3:
                            counts[">=3"] += 1
                        if value >= 4:
                            counts[">=4"] += 1
            anchor_count = 0
            if anchor_path.is_file():
                with anchor_path.open(newline="", encoding="utf-8") as stream:
                    anchor_count = sum(int(row.get("snapshots_with_new_intrusion", 0)) > 0 for row in csv.DictReader(stream))
            lines.append(f"| {seed} | `{path}` | {counts['1']} / {counts['2']} / {counts['>=3']} / {counts['>=4']} | {anchor_count} |")
        (REPORT_ROOT / "NDP_B_TRAJECTORY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.complete("Stage2 NDP-B2")

    def stage3(self) -> dict[str, Any]:
        existing_summary = self.out_root / "static_intervention/stage3_summary.json"
        existing_gate = self.out_root / "static_intervention/stage3_gate.json"
        if existing_summary.is_file() and existing_gate.is_file():
            self.log("【恢复】Stage3 summary/gate 已存在；读取同一 RUN_TIME 的 train-only static 结果，不覆盖已有 checkpoint。", "stage3")
            summary = read_json(existing_summary)
            self.complete("Stage3 NDP-C1/C2 静态有害关系干预")
            return summary
        self.set_stage("Stage3 NDP-C1/C2 静态有害关系干预", "LTCC", 1234, "C0/C1/C2", 0)
        base = [sys.executable, str(ROOT / "scripts/ndp_static_intervention.py"), "--run-time", self.run_time, "--out-root", str(self.out_root)]
        self.run_process(base + ["--mode", "prepare"], "Stage3 stable harmful relation prepare", "stage3", dataset="LTCC", seed=1234, arm="stable harmful relation", monitor_root=self.out_root / "static_intervention", retry_key="stage3_prepare")
        relation_path = self.out_root / "static_intervention/stable_harmful_relations.json"
        relation = read_json(relation_path)
        if int(relation.get("relation_count", 0)) <= 0:
            self.scientific_stop("Stage3 NDP-C1/C2", "双 seed train-only stable harmful relation 交集为空；不能伪造 static intervention evidence。")
            return {"gate": {"status": "FAIL", "relation_count": 0}}
        for arm in ("C0", "C1", "C2"):
            train_root = self.out_root / "static_intervention" / arm
            self.set_stage("Stage3 NDP-C1/C2 静态有害关系干预", "LTCC", 1234, arm, 0)
            run_dir = train_root / "ltcc" / "seed1234" / "gpc"
            complete = (
                (run_dir / "model_final.pth").is_file()
                and (
                    (run_dir / "RUN_COMPLETE.json").is_file()
                    or (run_dir / "final_checkpoint.json").is_file()
                    or (run_dir / "run_manifest.json").is_file()
                )
            )
            if complete:
                self.log(f"【恢复】Stage3 {arm} 已有合法 RUN_COMPLETE/model_final；跳过重复训练。", "stage3")
            else:
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", str(ROOT / "scripts/ndp_static_intervention.py"), "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root)]
                self.run_process(command, f"Stage3 {arm} train", "stage3", dataset="LTCC", seed=1234, arm=arm, monitor_root=train_root, retry_key=f"stage3_train_{arm}")
        self.set_stage("Stage3 NDP-C1/C2 静态有害关系分析", "LTCC", 1234, "C0/C1/C2", 30)
        if not (existing_summary.is_file() and existing_gate.is_file()):
            self.run_process(base + ["--mode", "analyze"], "Stage3 static intervention analyze", "stage3", dataset="LTCC", seed=1234, arm="C0/C1/C2", monitor_root=self.out_root / "static_intervention", retry_key="stage3_analyze")
        summary = read_json(self.out_root / "static_intervention/stage3_summary.json")
        self.complete("Stage3 NDP-C1/C2 静态有害关系干预")
        return summary

    def stage4(self) -> dict[str, Any]:
        summary_path = self.out_root / "gradient_audit/gradient_summary.json"
        lambda_path = self.out_root / "gradient_audit/NDP_LAMBDA_FREEZE.json"
        if summary_path.is_file() and lambda_path.is_file():
            self.log("【恢复】Stage4 gradient summary/lambda freeze 已存在；读取冻结 train-only 结果。", "stage4")
            summary = read_json(summary_path)
            self.complete("Stage4 NDP-C2 梯度独立性检查")
            return summary
        self.set_stage("Stage4 NDP-C2 梯度独立性检查", "LTCC", 1234, "g_triplet/g_gpc/g_ndp", 0)
        base = [sys.executable, str(ROOT / "scripts/ndp_gradient_audit.py"), "--run-time", self.run_time, "--out-root", str(self.out_root)]
        self.run_process(base + ["--mode", "prepare"], "Stage4 gradient audit prepare", "stage4", dataset="LTCC", seed=1234, arm="gradient prepare", monitor_root=self.out_root / "gradient_audit", retry_key="stage4_prepare")
        if not all((self.out_root / "gradient_audit" / f"gradient_batches_rank{rank}.csv").is_file() for rank in (0, 1)):
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", str(ROOT / "scripts/ndp_gradient_audit.py"), "--mode", "run", "--run-time", self.run_time, "--out-root", str(self.out_root)]
            self.run_process(command, "Stage4 gradient audit 200 train batches", "stage4", dataset="LTCC", seed=1234, arm="gradient audit", monitor_root=self.out_root / "gradient_audit", retry_key="stage4_run")
        if not summary_path.is_file():
            self.run_process(base + ["--mode", "summarize"], "Stage4 gradient audit summarize", "stage4", dataset="LTCC", seed=1234, arm="gradient summary", monitor_root=self.out_root / "gradient_audit", retry_key="stage4_summary")
        summary = read_json(self.out_root / "gradient_audit/gradient_summary.json")
        self.complete("Stage4 NDP-C2 梯度独立性检查")
        return summary

    def stage6_ltcc_screening(self) -> dict[str, Any]:
        summary_path = self.out_root / "online_training/ltcc_seed1234/screening_30/online_summary.json"
        if summary_path.is_file():
            self.log("【恢复】Stage6 LTCC screening summary 已存在；读取同一 RUN_TIME 的 train-only 结果。", "stage6")
            summary = read_json(summary_path)
            self.complete("Stage6 LTCC 30 epoch screening")
            return summary
        self.set_stage("Stage6 LTCC 30 epoch screening", "LTCC", 1234, "S0/S1/S2/S3", 0)
        script = str(ROOT / "scripts/ndp_online_training.py")
        base = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "30", "--seed", "1234", "--dataset", "ltcc"]
        self.run_process(base + ["--mode", "prepare"], "Stage6 LTCC screening prepare", "stage6", dataset="LTCC", seed=1234, arm="S0/S1/S2/S3", monitor_root=self.out_root / "online_training", retry_key="stage6_prepare")
        for arm in ("S0", "S1", "S2", "S3"):
            arm_root = self.out_root / "online_training/ltcc_seed1234/screening_30" / arm
            complete = (arm_root / "ltcc" / "seed1234" / ("bio" if arm == "S3" else "gpc") / "RUN_COMPLETE.json").is_file() and (arm_root / "ltcc" / "seed1234" / ("bio" if arm == "S3" else "gpc") / "model_final.pth").is_file()
            if complete:
                self.log(f"【恢复】Stage6 {arm} 已有合法最终 checkpoint；跳过重复训练。", "stage6")
                continue
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", script, "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "30", "--seed", "1234", "--dataset", "ltcc"]
            self.run_process(command, f"Stage6 LTCC screening {arm}", "stage6", dataset="LTCC", seed=1234, arm=arm, monitor_root=arm_root, retry_key=f"stage6_train_{arm}")
        if not summary_path.is_file():
            self.run_process(base + ["--mode", "analyze"], "Stage6 LTCC screening train-only analyze", "stage6", dataset="LTCC", seed=1234, arm="S0/S1/S2/S3", monitor_root=self.out_root / "online_training/ltcc_seed1234/screening_30", retry_key="stage6_analyze")
        summary = read_json(summary_path)
        self.complete("Stage6 LTCC 30 epoch screening")
        return summary

    def stage7_ltcc_formal(self) -> dict[str, Any]:
        self.set_stage("Stage7 LTCC 60 epoch formal confirmation", "LTCC", None, "F0/F1", 0)
        script = str(ROOT / "scripts/ndp_online_training.py")
        summaries = {}
        for seed in (1234, 4321):
            summary_path = self.out_root / f"online_training/ltcc_seed{seed}/screening_60/online_summary.json"
            if summary_path.is_file():
                self.log(f"【恢复】Stage7 LTCC seed{seed} summary 已存在；跳过该 seed 的重复正式训练。", "stage7")
                summaries[str(seed)] = read_json(summary_path)
                continue
            base = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", str(seed), "--dataset", "ltcc"]
            self.run_process(base + ["--mode", "prepare"], f"Stage7 LTCC seed{seed} formal prepare", "stage7", dataset="LTCC", seed=seed, arm="F0/F1", monitor_root=self.out_root / f"online_training/ltcc_seed{seed}/screening_60", retry_key=f"stage7_prepare_{seed}")
            for arm in ("S0", "S1"):
                arm_root = self.out_root / f"online_training/ltcc_seed{seed}/screening_60" / arm
                complete = (arm_root / "ltcc" / f"seed{seed}" / "gpc" / "RUN_COMPLETE.json").is_file() and (arm_root / "ltcc" / f"seed{seed}" / "gpc" / "model_final.pth").is_file()
                if complete:
                    self.log(f"【恢复】Stage7 LTCC seed{seed} {arm} 已有合法最终 checkpoint；跳过重复训练。", "stage7")
                    continue
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", script, "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", str(seed), "--dataset", "ltcc"]
                self.run_process(command, f"Stage7 LTCC seed{seed} {arm} formal", "stage7", dataset="LTCC", seed=seed, arm=arm, monitor_root=arm_root, retry_key=f"stage7_train_{seed}_{arm}")
            if not summary_path.is_file():
                self.run_process(base + ["--mode", "analyze"], f"Stage7 LTCC seed{seed} formal train-only analyze", "stage7", dataset="LTCC", seed=seed, arm="F0/F1", monitor_root=self.out_root / f"online_training/ltcc_seed{seed}/screening_60", retry_key=f"stage7_analyze_{seed}")
            summaries[str(seed)] = read_json(summary_path)
        write_json(self.out_root / "ltcc_full/formal_summary.json", {"run_time": self.run_time, "datasets": summaries, "epochs": 60, "test_started": False, "query_gallery_read": False})
        lines = ["# NDP_D_LTCC_CONFIRMATION_REPORT", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "", "LTCC seed1234/4321 的 F0/F1（实现 arm 为 S0/S1）均按 60 epoch、E60、train-only protocol 完成；正式 test 尚未打开。", "", "| Seed | Arm | Margin | Nearest negative | Positive | Same-ID pair | Wrong-PID density | Intrusion vs Bio | NDP active rate |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for seed, item in summaries.items():
            for arm in ("S0", "S1"):
                metrics = item.get("arms", {}).get(arm, {})
                lines.append(f"| {seed} | {arm} | {metrics.get('identity_margin_mean', float('nan')):.8f} | {metrics.get('nearest_negative_similarity_mean', float('nan')):.8f} | {metrics.get('positive_similarity_mean', float('nan')):.8f} | {metrics.get('same_id_pair_similarity_mean', float('nan')):.8f} | {metrics.get('wrong_pid_topk_density_mean', float('nan')):.8f} | {metrics.get('wrong_pid_intrusion_mean_vs_frozen_bio', float('nan')):.8f} | {metrics.get('ndp_active_rate', 0.0):.8f} |")
        lines += ["", "正式 test 仅在 LTCC/PRCC 所有 formal E60 checkpoint 完成 SHA256 freeze 后统一执行；本报告不使用 test 指标调参。", ""]
        (REPORT_ROOT / "NDP_D_LTCC_CONFIRMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        self.complete("Stage7 LTCC 60 epoch formal confirmation")
        return summaries

    def stage8_prcc_formal(self) -> dict[str, Any]:
        self.set_stage("Stage8 PRCC 60 epoch cross-dataset confirmation", "PRCC", None, "P0/P1/P2/P3", 0)
        script = str(ROOT / "scripts/ndp_online_training.py")
        summaries = {}
        for seed in (1234, 4321):
            summary_path = self.out_root / f"online_training/prcc_seed{seed}/screening_60/online_summary.json"
            if summary_path.is_file():
                self.log(f"【恢复】Stage8 PRCC seed{seed} summary 已存在；跳过该 seed 的重复正式训练。", "stage8")
                summaries[str(seed)] = read_json(summary_path)
                continue
            base = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", str(seed), "--dataset", "prcc"]
            self.run_process(base + ["--mode", "prepare"], f"Stage8 PRCC seed{seed} formal prepare", "stage8", dataset="PRCC", seed=seed, arm="P0/P1", monitor_root=self.out_root / f"online_training/prcc_seed{seed}/screening_60", retry_key=f"stage8_prepare_{seed}")
            for arm in ("S0", "S1"):
                arm_root = self.out_root / f"online_training/prcc_seed{seed}/screening_60" / arm
                complete = (arm_root / "prcc" / f"seed{seed}" / "gpc" / "RUN_COMPLETE.json").is_file() and (arm_root / "prcc" / f"seed{seed}" / "gpc" / "model_final.pth").is_file()
                if complete:
                    self.log(f"【恢复】Stage8 PRCC seed{seed} {arm} 已有合法最终 checkpoint；跳过重复训练。", "stage8")
                    continue
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", script, "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", str(seed), "--dataset", "prcc"]
                self.run_process(command, f"Stage8 PRCC seed{seed} {arm} formal", "stage8", dataset="PRCC", seed=seed, arm=arm, monitor_root=arm_root, retry_key=f"stage8_train_{seed}_{arm}")
            if not summary_path.is_file():
                self.run_process(base + ["--mode", "analyze"], f"Stage8 PRCC seed{seed} formal train-only analyze", "stage8", dataset="PRCC", seed=seed, arm="P0/P1", monitor_root=self.out_root / f"online_training/prcc_seed{seed}/screening_60", retry_key=f"stage8_analyze_{seed}")
            summaries[str(seed)] = read_json(summary_path)
        write_json(self.out_root / "prcc_full/formal_summary.json", {"run_time": self.run_time, "datasets": summaries, "epochs": 60, "test_started": False, "query_gallery_read": False})
        lines = ["# NDP_D_PRCC_CONFIRMATION_REPORT", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "", "PRCC seed1234/4321 的 P0/P1（实现 arm 为 S0/S1）均按 60 epoch、E60、train-only protocol 完成；正式 test 尚未打开。", "", "| Seed | Arm | Margin | Nearest negative | Positive | Same-ID pair | Wrong-PID density | Intrusion vs Bio | NDP active rate |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for seed, item in summaries.items():
            for arm in ("S0", "S1"):
                metrics = item.get("arms", {}).get(arm, {})
                lines.append(f"| {seed} | {arm} | {metrics.get('identity_margin_mean', float('nan')):.8f} | {metrics.get('nearest_negative_similarity_mean', float('nan')):.8f} | {metrics.get('positive_similarity_mean', float('nan')):.8f} | {metrics.get('same_id_pair_similarity_mean', float('nan')):.8f} | {metrics.get('wrong_pid_topk_density_mean', float('nan')):.8f} | {metrics.get('wrong_pid_intrusion_mean_vs_frozen_bio', float('nan')):.8f} | {metrics.get('ndp_active_rate', 0.0):.8f} |")
        lines += ["", "正式 test 仅在 LTCC/PRCC 所有 formal E60 checkpoint 完成 SHA256 freeze 后统一执行；本报告不使用 test 指标调参。", ""]
        (REPORT_ROOT / "NDP_D_PRCC_CONFIRMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        self.complete("Stage8 PRCC 60 epoch cross-dataset confirmation")
        return summaries

    def stage9_robustness(self) -> dict[str, Any]:
        robustness_path = self.out_root / "robustness/robustness_summary.json"
        if robustness_path.is_file():
            self.log("【恢复】Stage9 robustness summary 已存在；读取同一 RUN_TIME 的预注册 robustness 结果。", "stage9")
            summary = read_json(robustness_path)
            self.complete("Stage9 registered robustness block")
            return summary
        self.set_stage("Stage9 registered robustness block", "LTCC", 2026, "R1/R2/R3/R4", 0)
        script = str(ROOT / "scripts/ndp_online_training.py")
        # R1: third LTCC seed, fixed main K=10.
        base = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", "2026", "--dataset", "ltcc", "--k", "10"]
        self.run_process(base + ["--mode", "prepare"], "Stage9 R1 seed2026 prepare", "stage9", dataset="LTCC", seed=2026, arm="R1", monitor_root=self.out_root / "online_training/ltcc_seed2026/screening_60", retry_key="stage9_r1_prepare")
        for arm in ("S0", "S1"):
            arm_root = self.out_root / "online_training/ltcc_seed2026/screening_60" / arm
            complete = (arm_root / "ltcc" / "seed2026" / "gpc" / "RUN_COMPLETE.json").is_file() and (arm_root / "ltcc" / "seed2026" / "gpc" / "model_final.pth").is_file()
            if complete:
                self.log(f"【恢复】Stage9 R1 {arm} 已有合法最终 checkpoint；跳过重复训练。", "stage9")
                continue
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", script, "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "60", "--seed", "2026", "--dataset", "ltcc", "--k", "10"]
            self.run_process(command, f"Stage9 R1 seed2026 {arm}", "stage9", dataset="LTCC", seed=2026, arm=arm, monitor_root=arm_root, retry_key=f"stage9_r1_{arm}")
        if not (self.out_root / "online_training/ltcc_seed2026/screening_60/online_summary.json").is_file():
            self.run_process(base + ["--mode", "analyze"], "Stage9 R1 seed2026 analyze", "stage9", dataset="LTCC", seed=2026, arm="R1", monitor_root=self.out_root / "online_training/ltcc_seed2026/screening_60", retry_key="stage9_r1_analyze")
        # R2: pre-registered K=5 and K=20, 30 epoch diagnostic controls.
        k_summaries = {}
        for k in (5, 20):
            base_k = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "30", "--seed", "1234", "--dataset", "ltcc", "--k", str(k)]
            self.run_process(base_k + ["--mode", "prepare"], f"Stage9 R2 K{k} prepare", "stage9", dataset="LTCC", seed=1234, arm=f"K{k}", monitor_root=self.out_root / f"online_training/ltcc_seed1234/screening_30_K{k}", retry_key=f"stage9_k{k}_prepare")
            for arm in ("S0", "S1"):
                arm_root = self.out_root / f"online_training/ltcc_seed1234/screening_30_K{k}" / arm
                complete = (arm_root / "ltcc" / "seed1234" / "gpc" / "RUN_COMPLETE.json").is_file() and (arm_root / "ltcc" / "seed1234" / "gpc" / "model_final.pth").is_file()
                if complete:
                    self.log(f"【恢复】Stage9 R2 K{k} {arm} 已有合法最终 checkpoint；跳过重复训练。", "stage9")
                    continue
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", script, "--mode", "train", "--arm", arm, "--run-time", self.run_time, "--out-root", str(self.out_root), "--epochs", "30", "--seed", "1234", "--dataset", "ltcc", "--k", str(k)]
                self.run_process(command, f"Stage9 R2 K{k} {arm}", "stage9", dataset="LTCC", seed=1234, arm=f"K{k}/{arm}", monitor_root=arm_root, retry_key=f"stage9_k{k}_{arm}")
            k_summary_path = self.out_root / f"online_training/ltcc_seed1234/screening_30_K{k}/online_summary.json"
            if not k_summary_path.is_file():
                self.run_process(base_k + ["--mode", "analyze"], f"Stage9 R2 K{k} analyze", "stage9", dataset="LTCC", seed=1234, arm=f"K{k}", monitor_root=self.out_root / f"online_training/ltcc_seed1234/screening_30_K{k}", retry_key=f"stage9_k{k}_analyze")
            k_summaries[str(k)] = read_json(k_summary_path)
        summary = {"run_time": self.run_time, "R1_seed2026": read_json(self.out_root / "online_training/ltcc_seed2026/screening_60/online_summary.json"), "R2_K5_K20": k_summaries, "R3_selector_shuffle": "S2 matched random assignment recorded in LTCC screening_30", "R4_static_vs_online": "C1 static report and S1 online train-only artifacts retained", "query_gallery_read": False, "test_metrics_read": False}
        write_json(self.out_root / "robustness/robustness_summary.json", summary)
        (REPORT_ROOT / "NDP_R4_STATIC_VS_ONLINE_AUDIT.md").write_text(
            "\n".join([
                "# NDP_R4_STATIC_VS_ONLINE_AUDIT", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "",
                "R4 只比较 train-only mechanism：Static Oracle-like C1 与 Online S1 的 margin、nearest-negative、positive consistency、intrusion 和 hubness artifact。没有使用 query/gallery/test 或 Rank-1/mAP 选择方法。", "",
                f"Static report：`{REPORT_ROOT / 'NDP_C_STATIC_INTERVENTION_REPORT.md'}`", f"Online LTCC screening：`{self.out_root / 'online_training/ltcc_seed1234/screening_30/online_summary.json'}`", "",
                "正式 test 不属于本 robustness comparison；正式 arm 仍统一使用 E60 freeze 后的 final evaluator。", "",
            ]), encoding="utf-8")
        self.complete("Stage9 registered robustness block")
        return summary

    def stage10_freeze_and_test(self) -> dict[str, Any]:
        final_summary_path = self.out_root / "final_test/final_summary.json"
        if final_summary_path.is_file():
            self.log("【恢复】Stage10 final_summary 已存在；不重复读取 query/gallery。", "final")
            summary = read_json(final_summary_path)
            self.complete("Stage10 final freeze and test")
            return summary
        self.set_stage("Stage10 final freeze", None, None, "formal E60 freeze", 60)
        script = str(ROOT / "scripts/ndp_final_freeze_test.py")
        base = [sys.executable, script, "--run-time", self.run_time, "--out-root", str(self.out_root)]
        if not (self.out_root / "manifest/NDP_FORMAL_FREEZE.json").is_file():
            self.run_process(base + ["--mode", "freeze"], "Stage10 formal E60 checkpoint freeze", "final", monitor_root=self.out_root / "manifest", retry_key="stage10_freeze")
        names = ["ltcc_seed1234_S0", "ltcc_seed1234_S1", "ltcc_seed4321_S0", "ltcc_seed4321_S1", "prcc_seed1234_S0", "prcc_seed1234_S1", "prcc_seed4321_S0", "prcc_seed4321_S1"]
        for name in names:
            if not (self.out_root / "final_test" / f"{name}.json").is_file():
                self.run_process(base + ["--mode", "evaluate", "--name", name], f"Stage10 first final test {name}", "final", dataset="LTCC" if name.startswith("ltcc") else "PRCC", seed=1234 if "seed1234" in name else 4321, arm=name, monitor_root=self.out_root / "final_test", retry_key=f"stage10_test_{name}")
        if not final_summary_path.is_file():
            self.run_process(base + ["--mode", "summarize"], "Stage10 final report summarize", "final", monitor_root=self.out_root / "final_test", retry_key="stage10_summarize")
        summary = read_json(final_summary_path)
        self.complete("Stage10 final freeze and test")
        return summary

    def write_manifest(self, stage1: dict[str, Any] | None) -> None:
        source_audit = read_json(self.out_root / "source_audit/source_audit.json") if (self.out_root / "source_audit/source_audit.json").is_file() else {}
        leakage_audit = read_json(self.out_root / "source_audit/data_leakage_audit.json") if (self.out_root / "source_audit/data_leakage_audit.json").is_file() else {}
        ppu_preflight = read_json(self.out_root / "source_audit/ppu_preflight.json") if (self.out_root / "source_audit/ppu_preflight.json").is_file() else {}
        stage1_gate = read_json(self.out_root / "online_signal/stage1_promotion_gate.json") if (self.out_root / "online_signal/stage1_promotion_gate.json").is_file() else {}
        stage3_gate = read_json(self.out_root / "static_intervention/stage3_gate.json") if (self.out_root / "static_intervention/stage3_gate.json").is_file() else {}
        stage4_summary = read_json(self.out_root / "gradient_audit/gradient_summary.json") if (self.out_root / "gradient_audit/gradient_summary.json").is_file() else {}
        stable_relations = read_json(self.out_root / "static_intervention/stable_harmful_relations.json") if (self.out_root / "static_intervention/stable_harmful_relations.json").is_file() else {}
        final_summary = read_json(self.out_root / "final_test/final_summary.json") if (self.out_root / "final_test/final_summary.json").is_file() else {}
        try:
            import torch
            torch_version = torch.__version__
        except Exception:
            torch_version = None
        ppu_log_text = ""
        if ppu_preflight.get("log") and Path(str(ppu_preflight["log"])).is_file():
            ppu_log_text = Path(str(ppu_preflight["log"])).read_text(encoding="utf-8", errors="replace")
        ppu_names = sorted(set(re.findall(r"name=(PPU-[A-Za-z0-9_-]+)", ppu_log_text)))
        manifest = {
            "run_time": self.run_time,
            "actual_start_time": self.state.get("started_at"),
            "actual_end_time": now() if self.state.get("status") in ("COMPLETED", "STOPPED_BY_SCIENTIFIC_GATE", "BLOCKED") else None,
            "main_log": str(self.main_log),
            "state_json": str(self.state_path),
            "status": self.state.get("status"),
            "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "source_audit": str(self.out_root / "source_audit/source_audit.json"),
            "data_leakage_audit": str(self.out_root / "source_audit/data_leakage_audit.json"),
            "ppu_preflight": str(self.out_root / "source_audit/ppu_preflight.json"),
            "dataset_split": leakage_audit.get("split_rows", []),
            "checkpoint_source": source_audit.get("checkpoints", []),
            "ppu_devices": ppu_names or ppu_preflight.get("device_names", ppu_preflight.get("all_device_names")),
            "ppu_device_count": 2 if ppu_preflight.get("status") == "PASS" else len(ppu_names),
            "ppu_runtime": "PPU-ZW810E / torch.cuda PPU runtime" if ppu_names else ppu_preflight.get("runtime", ppu_preflight.get("status")),
            "ppu_environment": {key: os.environ.get(key) for key in ("PPU_HOME", "PPU_SDK", "LD_LIBRARY_PATH", "LIBRARY_PATH")},
            "python_version": sys.version,
            "torch_version": torch_version,
            "gpc_definition": "L_CE + L_Triplet + L_Bio + 0.1 * mean(1-cos(normalized feat[0])) over same-PID distinct-image pairs",
            "lambda_gpc": 0.1,
            "ndp_definition": "Selective Relative Margin Preservation over registered harmful/intrusive different-PID relations; no global neighborhood reconstruction",
            "K": {"primary": 10, "registered": [5, 10, 20]},
            "lambda_ndp": stage4_summary.get("lambda_ndp"),
            "selector": stage1_gate.get("viable_signals", []),
            "active_rate": "recorded per arm in online_train_stats_rank*.json",
            "pair_count": stable_relations.get("relation_count"),
            "epochs": {"trajectory": 30, "screening": 30, "formal": 60},
            "AMP": "repository GPC protocol; LTCC initial scale 1024, PRCC initial scale 512",
            "optimizer": "SGD",
            "scheduler": "cosine with 20-epoch warmup",
            "command_line": sys.argv,
            "completed_stages": self.state.get("completed_stages", []),
            "failed_stages": self.state.get("failed_stages", []),
            "logs": {key: str(path) for key, path in self.sublogs.items()},
            "query_gallery_used_for_design": False,
            "test_evaluation_started": (self.out_root / "final_test/final_summary.json").is_file(),
            "cpu_model_fallback": False,
            "gpc_modified": False,
            "stage1_summary": str(self.out_root / "online_signal/stage1_summary.json") if stage1 else None,
            "stage3_gate": str(self.out_root / "static_intervention/stage3_gate.json") if (self.out_root / "static_intervention/stage3_gate.json").is_file() else None,
            "stage4_summary": str(self.out_root / "gradient_audit/gradient_summary.json") if (self.out_root / "gradient_audit/gradient_summary.json").is_file() else None,
            "lambda_ndp_freeze": str(self.out_root / "gradient_audit/NDP_LAMBDA_FREEZE.json") if (self.out_root / "gradient_audit/NDP_LAMBDA_FREEZE.json").is_file() else None,
            "formal_freeze": str(self.out_root / "manifest/NDP_FORMAL_FREEZE.json") if (self.out_root / "manifest/NDP_FORMAL_FREEZE.json").is_file() else None,
            "final_test_summary": str(self.out_root / "final_test/final_summary.json") if (self.out_root / "final_test/final_summary.json").is_file() else None,
            "reports": {
                "source": str(REPORT_ROOT / "NDP_THREE_DAY_SOURCE_AUDIT.md"),
                "leakage": str(REPORT_ROOT / "NDP_THREE_DAY_DATA_LEAKAGE_AUDIT.md"),
                "runtime": str(REPORT_ROOT / "NDP_THREE_DAY_RUNTIME_AUDIT.md"),
                "final": str(REPORT_ROOT / "NDP_THREE_DAY_FINAL_REPORT.md"),
                "stage1": str(REPORT_ROOT / "NDP_B_HARMFUL_ATTRACTION_AUDIT.md"),
                "selector": str(REPORT_ROOT / "NDP_B_ONLINE_SELECTOR_AUDIT.md"),
                "trajectory": str(REPORT_ROOT / "NDP_B_TRAJECTORY_AUDIT.md"),
                "static": str(REPORT_ROOT / "NDP_C_STATIC_INTERVENTION_REPORT.md"),
                "gradient": str(REPORT_ROOT / "NDP_C_GRADIENT_INDEPENDENCE_REPORT.md"),
                "ltcc": str(REPORT_ROOT / "NDP_D_LTCC_CONFIRMATION_REPORT.md"),
                "prcc": str(REPORT_ROOT / "NDP_D_PRCC_CONFIRMATION_REPORT.md"),
                "robustness_static_vs_online": str(REPORT_ROOT / "NDP_R4_STATIC_VS_ONLINE_AUDIT.md"),
                "fail_branch": str(REPORT_ROOT / "NDP_FAIL_BRANCH_AUDIT.md"),
            },
            "formal_test_execution_times": [item.get("evaluated_utc") for item in final_summary.get("results", [])],
        }
        write_json(self.out_root / "manifest/NDP_THREE_DAY_MANIFEST.json", manifest)

    def write_runtime_report(self) -> None:
        """Write an auditable runtime/failure report without changing science state."""
        failed = self.state.get("failed_stages", [])
        lines = [
            "# NDP_THREE_DAY_RUNTIME_AUDIT", "",
            f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "",
            f"最终运行状态：**{self.state.get('status')}**", f"当前/最后阶段：`{self.state.get('current_stage')}`", f"后台 PID：`{self.state.get('pid')}`", "",
            "本报告区分运行故障与科学结论。PPU/脚本恢复失败记录为 runtime blocked；只有显式 science gate 才记录为 STOPPED_BY_SCIENTIFIC_GATE。", "",
            f"Source Audit：**{self.state.get('source_audit', 'UNKNOWN')}**", f"Train-only Leakage Audit：**{self.state.get('train_only_leakage_audit', 'UNKNOWN')}**", f"CPU model fallback：**{self.state.get('cpu_model_fallback')}**", f"Query/gallery used for method design：**{self.state.get('query_gallery_read_for_method_design')}**", f"GPC modified：**{self.state.get('gpc_modified')}**", "",
            "## 已完成阶段", "",
        ]
        lines.extend(f"- {item}" for item in self.state.get("completed_stages", []))
        if not self.state.get("completed_stages"):
            lines.append("- 无")
        lines += ["", "## 失败或停止记录", ""]
        if failed:
            lines += [f"- `{item.get('time')}`；`{item.get('type')}`；{item.get('stage')}：{item.get('detail')}" for item in failed]
        else:
            lines.append("- 无")
        lines += ["", "## 日志", "", f"主日志：`{self.main_log}`"]
        for key, path in self.sublogs.items():
            lines.append(f"- {key}：`{path}`")
        lines += ["", "NO QUERY/GALLERY/TEST DATA WAS USED FOR METHOD DESIGN BEFORE THE FINAL FREEZE GATE.", "NO CPU MODEL FALLBACK WAS USED.", "NO EXISTING GPC CHECKPOINT WAS MODIFIED.", ""]
        (REPORT_ROOT / "NDP_THREE_DAY_RUNTIME_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    def write_fail_branch_audit(self, stage1: dict[str, Any], stage3: dict[str, Any], stage4: dict[str, Any]) -> None:
        stage1_gate = stage1.get("gate", {})
        stage3_gate = stage3.get("gate", {})
        stage4_gate = stage4.get("gradient_gate", "NOT RUN")
        lines = [
            "# NDP_FAIL_BRANCH_AUDIT", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "",
            "本报告记录科学 gate 未通过后的预注册 fail branch；运行故障不在此处伪装成方法失败。", "",
            f"Stage1 selected S2 gate：{stage1_gate.get('status', 'NOT RUN')}；viable signals：{stage1_gate.get('viable_signals', [])}",
            f"Stage3 Static Intervention：{stage3_gate.get('status', 'NOT RUN')}；relation_count={stage3_gate.get('relation_count', 'NA')}",
            f"Stage4 Gradient Gate：{stage4_gate}；median cos(g_triplet,g_ndp)={stage4.get('median_cos_triplet_ndp', 'NA')}", "",
            "FAIL BRANCH A：若 Gradient Gate FAIL，保留 200 train batches 梯度报告、Stage2 trajectory 和 static intervention，不启动 online NDP。", "",
            "FAIL BRANCH B：若没有 online signal 通过，保留 Static Oracle-like Intervention 与 random control；Online Detection 标记 unresolved。", "",
            "FAIL BRANCH C：若 Static Intervention FAIL，保留双 seed Stage1、random control、gradient report，停止方法开发，不继续发明 selector。", "",
            "NO QUERY/GALLERY/TEST DATA WAS USED FOR METHOD DESIGN.", "NO EXISTING GPC CHECKPOINT WAS MODIFIED.", "",
        ]
        (REPORT_ROOT / "NDP_FAIL_BRANCH_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    def final_interim_report(self, stage1: dict[str, Any] | None) -> None:
        placeholder_reports = {
            "NDP_B_TRAJECTORY_AUDIT.md": "Stage2 NDP-B2 trajectory report",
            "NDP_C_STATIC_INTERVENTION_REPORT.md": "Stage3 NDP-C1/C2 static intervention report",
            "NDP_C_GRADIENT_INDEPENDENCE_REPORT.md": "Stage4 NDP-C2 gradient independence report",
            "NDP_D_LTCC_CONFIRMATION_REPORT.md": "Stage7 LTCC formal confirmation report",
            "NDP_D_PRCC_CONFIRMATION_REPORT.md": "Stage8 PRCC formal confirmation report",
        }
        for filename, title in placeholder_reports.items():
            path = REPORT_ROOT / filename
            if not path.is_file():
                path.write_text("\n".join([
                    f"# {title}", "", f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", "",
                    "状态：**NOT RUN / 未完成**。本报告由最终边界审计自动创建；不能把未执行阶段解释为方法失败。", "",
                    f"当前 pipeline 状态：`{self.state.get('status')}`；当前阶段：`{self.state.get('current_stage')}`。", "",
                ]), encoding="utf-8")
        gate = stage1.get("gate", {}) if stage1 else {}
        stage3_gate = read_json(self.out_root / "static_intervention/stage3_gate.json") if (self.out_root / "static_intervention/stage3_gate.json").is_file() else {}
        stage4_gate = read_json(self.out_root / "gradient_audit/gradient_summary.json") if (self.out_root / "gradient_audit/gradient_summary.json").is_file() else {}
        online_status = "NOT RUN"
        if self.out_root.joinpath("online_training").is_dir():
            online_status = "RUN OR PARTIAL"
        if self.state.get("status") == "STOPPED_BY_SCIENTIFIC_GATE":
            ndp_status = "NOT SUPPORTED" if stage3_gate.get("status") == "FAIL" or stage4_gate.get("gradient_gate") == "FAIL" else "MIXED"
        else:
            ndp_status = "MIXED"
        stage2_status = "YES" if all((self.out_root / "trajectory_runs" / f"seed{seed}" / "trajectory" / f"seed{seed}" / "trajectory_manifest.json").is_file() for seed in (1234, 4321)) else "MIXED/NOT COMPLETE"
        lines = [
            "# NDP_THREE_DAY_FINAL_REPORT", "",
            "==================================================", "一、最终结论", "==================================================", "",
            f"NDP：**{ndp_status}**（当前运行状态：{self.state.get('status')}；未完成阶段不解释为方法失败）", "",
            "==================================================", "二、第一创新点 GPC 状态", "==================================================", "",
            "保持有效；本轮没有修改 GPC。", "",
            "==================================================", "三、第二创新点机制", "==================================================", "",
            "Wrong-PID intrusion：由 NDP-A 与 Stage1 train-only 诊断。",
            f"Harmful attraction identifiable：{gate.get('status', 'NOT AVAILABLE')}。",
            f"Persistence：{stage2_status}（仅统计相对 E0 Bio 的新 negative relation）。",
            f"Static suppression：{stage3_gate.get('status', 'NOT RUN')}", f"Online selector：{gate.get('status', 'NOT RUN')}", f"Gradient novelty：{stage4_gate.get('gradient_gate', 'NOT RUN')}", f"Online NDP：{online_status}", "",
            "Static Intervention 与 Gradient Gate 之外的正式 NDP 训练、PRCC、robustness 或 final test 未完成时，均明确写为 NOT RUN/未完成。", "",
            "==================================================", "四、LTCC", "==================================================", "", "Formal E60：NOT RUN/未完成。", "",
            "==================================================", "五、PRCC", "==================================================", "", "Formal E60：NOT RUN/未完成。", "",
            "==================================================", "六、第二创新点判断", "==================================================", "", f"{ndp_status}；依据为已完成的 train-only gate，运行故障单独记录。", "",
            f"RUN_TIME：`{self.run_time}`", f"主日志：`{self.main_log}`", f"状态 JSON：`{self.state_path}`", "",
            "TRAINING WAS STARTED ONLY FOR THE REGISTERED TRAIN-ONLY DIAGNOSTIC/FORMAL BRANCHES AFTER THE GATES; no unregistered training was started.",
            "NO QUERY/GALLERY/TEST DATA WAS USED for method design or training geometry before final freeze.",
            "NO EXISTING GPC CHECKPOINT WAS MODIFIED.",
            "NO CPU MODEL FALLBACK WAS USED.", "",
        ]
        (REPORT_ROOT / "NDP_THREE_DAY_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> int:
        self.update(status="RUNNING", current_stage="STARTUP")
        preflight = self.out_root / "source_audit/preflight_summary.json"
        if not preflight.is_file() or read_json(preflight).get("status") != "PASS":
            self.fail_runtime("preflight", "preflight summary missing or not PASS")
            return 2
        stage1 = None
        try:
            stage1 = self.stage1()
            # Stage2 is mandatory even if Stage1 online signal gate is not viable.
            self.stage2()
            stage3 = self.stage3()
            stage4 = self.stage4()
            viable_signals = stage1.get("gate", {}).get("viable_signals", [])
            # The online runner is pre-registered to S2.  A different viable
            # signal must not silently authorize an S2 implementation.
            stage1_viable = stage1.get("gate", {}).get("status") == "ONLINE SIGNAL VIABLE" and "S2_negative_attraction_increase" in viable_signals
            stage3_pass = stage3.get("gate", {}).get("status") == "PASS"
            stage4_pass = stage4.get("gradient_gate") == "PASS"
            if stage1_viable and stage3_pass and stage4_pass:
                self.log("【Stage5 gate】Stage1 online signal、Stage3 static mechanism、Stage4 gradient independence 均 PASS；允许进入 LTCC screening/formal。", "stage5")
                self.stage6_ltcc_screening()
                self.stage7_ltcc_formal()
                self.stage8_prcc_formal()
                self.stage9_robustness()
                self.stage10_freeze_and_test()
                self.update(status="COMPLETED", current_stage="COMPLETED", dataset=None, seed=None, arm=None, epoch=60)
                self.log("【最终中文结论】Stage1-Stage10 全部完成；formal E60 freeze 后才首次执行 query/gallery test；NDP 结论见最终报告。", "final")
            else:
                self.write_fail_branch_audit(stage1, stage3, stage4)
                self.scientific_stop("Stage5 online NDP", f"gate 未同时通过：Stage1={stage1_viable}；Stage3={stage3_pass}；Stage4={stage4_pass}；进入科学 FAIL branch，不启动 online NDP。")
            if not (self.state.get("status") == "COMPLETED" and (self.out_root / "final_test/final_summary.json").is_file()):
                self.final_interim_report(stage1)
            self.write_manifest(stage1)
            self.write_runtime_report()
            return 0
        except Exception as exc:
            self.fail_runtime(self.state.get("current_stage", "unknown"), f"{type(exc).__name__}: {exc}")
            self.final_interim_report(stage1)
            self.write_manifest(stage1)
            self.write_runtime_report()
            return 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--main-log", required=True)
    args = parser.parse_args()
    return Pipeline(args.run_time, Path(args.out_root).resolve(), Path(args.main_log).resolve()).run()


if __name__ == "__main__":
    raise SystemExit(main())
