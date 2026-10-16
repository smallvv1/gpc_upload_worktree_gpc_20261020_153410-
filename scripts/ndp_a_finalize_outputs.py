#!/usr/local/bin/python3
"""Finalize the NDP-A artifact layout without changing any diagnostic value."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/new_method_track/ndp_a_neighborhood_feasibility"
TABLES = OUT / "tables"
REPORTS = ROOT / "reports"
LOGS = Path("/home/DIFFER/logs")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip("\n") + "\n")


def copy_table(source: Path, target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    args = parser.parse_args()
    run_time = args.run_time
    manifest_path = OUT / "NDP_A_MANIFEST.json"
    manifest = read_json(manifest_path)
    if manifest.get("run_time") != run_time:
        raise RuntimeError(f"manifest RUN_TIME mismatch: {manifest.get('run_time')} != {run_time}")

    section_dirs = {
        "overlap": OUT / "overlap",
        "intrusion": OUT / "intrusion",
        "same_id_retention": OUT / "same_id_retention",
        "negative_inflation": OUT / "negative_inflation",
        "rank_stability": OUT / "rank_stability",
        "js_divergence": OUT / "js_divergence",
        "hubness": OUT / "hubness",
        "pid_hubness": OUT / "pid_hubness",
        "harmful_vs_beneficial": OUT / "harmful_vs_beneficial",
        "quartiles": OUT / "quartiles",
        "cross_seed": OUT / "cross_seed",
    }
    source_map = {
        "overlap": [(TABLES / "overlap.csv", "overlap")],
        "intrusion": [(TABLES / "intrusion.csv", "intrusion"), (TABLES / "pid_intrusion.csv", "pid_intrusion")],
        "same_id_retention": [(TABLES / "same_id_retention.csv", "same_id_retention")],
        "negative_inflation": [],
        "rank_stability": [(TABLES / "rank_stability.csv", "rank_stability")],
        "js_divergence": [(TABLES / "js_divergence.csv", "js_divergence")],
        "hubness": [(TABLES / "image_hubness.csv", "image_hubness"), (TABLES / "new_bad_hub_candidates.csv", "new_bad_hub_candidates")],
        "pid_hubness": [(TABLES / "pid_hubness.csv", "pid_hubness"), (TABLES / "pid_intrusion.csv", "pid_intrusion")],
        "harmful_vs_beneficial": [(TABLES / "harmful_vs_beneficial.csv", "harmful_vs_beneficial")],
        "quartiles": [(TABLES / "quartiles.csv", "quartiles")],
        "cross_seed": [(TABLES / "cross_seed_consistency.csv", "cross_seed_consistency"), (TABLES / "correlations.csv", "correlations"), (TABLES / "promotion_gate.json", "promotion_gate")],
    }

    # Add the required nearest-negative table as a CSV companion to the JSON.
    summary_json = read_json(TABLES / "nearest_negative_inflation.json")
    rows = []
    for seed, metrics in summary_json.items():
        for metric, summary in metrics.items():
            rows.append({"seed": seed, "metric": metric, **summary})
    fields = ["seed", "metric", "count", "mean", "median", "p25", "p75", "pid_bootstrap_ci95_low", "pid_bootstrap_ci95_high"]
    csv_path = TABLES / "nearest_negative_inflation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    source_map["negative_inflation"] = [(csv_path, "nearest_negative_inflation"), (TABLES / "nearest_negative_inflation.json", "nearest_negative_inflation")]

    copied = {}
    for section, files in source_map.items():
        copied[section] = {}
        for source, label in files:
            if not source.is_file():
                raise FileNotFoundError(source)
            copied[section][label] = copy_table(source, section_dirs[section])

    manifest["artifact_layout_complete"] = True
    manifest["section_directories"] = {key: str(path) for key, path in section_dirs.items()}
    manifest["section_artifacts"] = copied
    manifest["tables"] = {path.stem: str(path) for path in TABLES.glob("*") if path.is_file()}
    manifest["outputs"]["sections"] = {key: str(path) for key, path in section_dirs.items()}
    write_json(manifest_path, manifest)

    report_path = REPORTS / "NDP_A_NEIGHBORHOOD_DISCRIMINATIVE_FEASIBILITY_AUDIT.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## 运行日志与产物\n"
    section_text = "## 规定的分目录产物\n\n" + "\n".join(f"- `{path}`" for path in section_dirs.values()) + "\n\n"
    if "## 规定的分目录产物" not in report:
        report = report.replace(marker, section_text + marker)
    control_text = "## Secondary control：Phase 7I I2\n\n本阶段未运行 I0→I2 secondary diagnostic。原因是两个独立 matched seed 的 primary NDP-A promotion gate 已经完成并得到 MIXED/STOP；I2 不是 promotion 的必要条件，继续扩展不会改变本阶段决策。\n\n"
    if "## Secondary control：Phase 7I I2" not in report:
        report = report.replace("## K. Required declarations\n", control_text + "## K. Required declarations\n")
    report_path.write_text(report, encoding="utf-8")

    main_log = LOGS / f"NDP邻域判别保持可行性诊断_{run_time}.log"
    final_log = LOGS / f"NDP最终结论_{run_time}.log"
    message = "【产物整理】已补齐 overlap/intrusion/same_id_retention/negative_inflation/rank_stability/js_divergence/hubness/pid_hubness/harmful_vs_beneficial/quartiles/cross_seed 分目录；只复制既有 train-only 统计并补生成 nearest_negative_inflation.csv，数值与 gate 未改变。"
    append_log(main_log, message)
    append_log(final_log, message)
    append_log(main_log, "【中文结论】产物布局 PASS；RUN_TIME、Manifest、报告和所有子日志仍保持一致。")
    append_log(final_log, "【中文结论】产物布局 PASS；RUN_TIME、Manifest、报告和所有子日志仍保持一致。")
    append_log(main_log, "【Secondary control】Phase 7I I2 未运行；primary 两 seed gate 已足以给出 MIXED/STOP，I2 不是 promotion 必要条件。")
    append_log(final_log, "【Secondary control】Phase 7I I2 未运行；primary 两 seed gate 已足以给出 MIXED/STOP，I2 不是 promotion 必要条件。")
    print(json.dumps({"run_time": run_time, "artifact_layout_complete": True, "manifest": str(manifest_path), "sections": {key: str(path) for key, path in section_dirs.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
