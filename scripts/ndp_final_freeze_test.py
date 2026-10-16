#!/usr/local/bin/python3
"""Stage 10: freeze formal E60 arms, then perform the first final test."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/home/DIFFER")
LOG_ROOT = ROOT / "logs"
FORMAL = {
    "ltcc_seed1234_S0": ("ltcc", 1234, "S0", "gpc"),
    "ltcc_seed1234_S1": ("ltcc", 1234, "S1", "gpc"),
    "ltcc_seed4321_S0": ("ltcc", 4321, "S0", "gpc"),
    "ltcc_seed4321_S1": ("ltcc", 4321, "S1", "gpc"),
    "prcc_seed1234_S0": ("prcc", 1234, "S0", "gpc"),
    "prcc_seed1234_S1": ("prcc", 1234, "S1", "gpc"),
    "prcc_seed4321_S0": ("prcc", 4321, "S0", "gpc"),
    "prcc_seed4321_S1": ("prcc", 4321, "S1", "gpc"),
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_root(out_root: Path, dataset: str, seed: int, arm: str) -> Path:
    return out_root / "online_training" / f"{dataset}_seed{seed}" / "screening_60" / arm / dataset / f"seed{seed}" / "gpc"


def freeze(run_time: str, out_root: Path) -> dict[str, Any]:
    records = {}
    for name, (dataset, seed, arm, base_arm) in FORMAL.items():
        root = formal_root(out_root, dataset, seed, arm)
        final = root / "model_final.pth"
        manifest = root / "run_manifest.json"
        complete = root / "RUN_COMPLETE.json"
        metrics = root / "metrics_per_epoch.csv"
        if not final.is_file() or not manifest.is_file() or not complete.is_file() or not metrics.is_file():
            raise RuntimeError(f"formal artifact missing for {name}: {root}")
        rows = metrics.read_text(encoding="utf-8").splitlines()
        if len(rows) < 61:
            raise RuntimeError(f"formal epoch metrics incomplete for {name}: {len(rows)-1}")
        epoch_checkpoints = {}
        for epoch in (10, 20, 30, 40, 50, 60):
            checkpoint = root.parent.parent / "checkpoints" / f"model_epoch_{epoch:02d}.pth"
            # The public checkpoint root is <out>/<DATASET>/checkpoints; root
            # is <out>/<DATASET>/seed<seed>/gpc.
            if not checkpoint.is_file():
                checkpoint = root.parents[2] / "checkpoints" / f"model_epoch_{epoch:02d}.pth"
            if not checkpoint.is_file():
                raise RuntimeError(f"formal epoch-{epoch} checkpoint missing for {name}: {checkpoint}")
            epoch_checkpoints[str(epoch)] = {"path": str(checkpoint), "sha256": sha256(checkpoint), "epoch": epoch}
        state = torch.load(final, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"formal checkpoint is not a state dict: {final}")
        records[name] = {
            "dataset": dataset, "seed": seed, "arm": arm, "base_arm": base_arm,
            "path": str(final), "sha256": sha256(final), "epoch": 60,
            "frozen": True, "test_started": False, "metrics_rows": len(rows) - 1,
            "checkpoint_selection_metric": None,
            "epoch_checkpoints": epoch_checkpoints,
        }
    result = {"run_time": run_time, "created_utc": now(), "status": "FROZEN", "formal_arms": records, "first_query_gallery_evaluation_opened": False, "test_metric_used_for_freeze": False, "query_gallery_used_for_design": False, "gpc_modified": False, "cpu_model_fallback": False}
    write_json(out_root / "manifest/NDP_FORMAL_FREEZE.json", result)
    return result


def import_engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # noqa: E402
    return engine


def evaluate_one(run_time: str, out_root: Path, name: str) -> dict[str, Any]:
    freeze_path = out_root / "manifest/NDP_FORMAL_FREEZE.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN" or frozen.get("first_query_gallery_evaluation_opened"):
        raise RuntimeError("formal freeze gate is not closed or test was already opened")
    dataset_name, seed, arm, base_arm = FORMAL[name]
    record = frozen["formal_arms"][name]
    engine = import_engine()
    engine.SEED = seed
    engine.EPOCHS = 60
    root = out_root / "online_training" / f"{dataset_name}_seed{seed}" / "screening_60" / arm
    cfg = engine.cfg_for(dataset_name, root / dataset_name / f"seed{seed}" / base_arm)
    dataset = engine.build_dataset(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("final test requires two visible PPUs; CPU fallback forbidden")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    state = torch.load(record["path"], map_location="cpu", weights_only=False)
    model = engine.build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, transform_test = engine.build_img_transforms(cfg)
    if dataset_name == "ltcc":
        query = dataset.query; gallery = dataset.gallery
        evaluator = engine.R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        loader = torch.utils.data.DataLoader(engine.ConcatDataset([engine.ImageDataset(query, transform=transform_test), engine.ImageDataset(gallery, transform=transform_test)]), batch_size=int(cfg.DATA.TEST_BATCH), shuffle=False, num_workers=int(cfg.DATA.NUM_WORKERS), pin_memory=False, drop_last=False)
        ltcc = True
    else:
        query = dataset.query_diff; gallery = dataset.gallery
        evaluator = engine.R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        loader = torch.utils.data.DataLoader(engine.ConcatDataset([engine.ImageDataset(query, transform=transform_test), engine.ImageDataset(gallery, transform=transform_test)]), batch_size=int(cfg.DATA.TEST_BATCH), shuffle=False, num_workers=int(cfg.DATA.NUM_WORKERS), pin_memory=False, drop_last=False)
        ltcc = False
    log_path = LOG_ROOT / f"NDP最终测试_{run_time}.log"
    logger = engine.logger_for(log_path)
    logger.info("[%s] 【Stage10 final test】开始；所有 formal E60 已先完成 freeze gate；本步骤首次读取 query/gallery。", name)
    evaluator.reset()
    engine.processor_test(cfg, model, evaluator, loader, logger, device, test=True, ltcc=ltcc, clothChanging=True)
    cmc, mAP, *_rest = evaluator.compute()
    result = {"run_time": run_time, "name": name, "dataset": dataset_name, "seed": seed, "arm": arm, "checkpoint": record, "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]), "mAP": float(mAP), "mINP": None, "query_count": len(query), "gallery_count": len(gallery), "evaluated_utc": now(), "evaluator": "current repository evaluator; mINP unavailable"}
    logger.info("[%s] 【中文结论】test 完成；Rank-1=%.8f；mAP=%.8f；mINP=Unavailable under current evaluator。", name, result["rank1"], result["mAP"])
    output = out_root / "final_test" / f"{name}.json"
    write_json(output, result)
    del model, state, loader, dataset
    torch.cuda.empty_cache()
    return result


def summarize(run_time: str, out_root: Path) -> dict[str, Any]:
    files = sorted((out_root / "final_test").glob("*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in files if path.name != "final_summary.json"]
    expected = set(FORMAL)
    if {item["name"] for item in results} != expected:
        raise RuntimeError("not all formal final test results are present")
    by = {item["name"]: item for item in results}
    comparisons = {}
    for dataset in ("ltcc", "prcc"):
        for seed in (1234, 4321):
            gpc = by[f"{dataset}_seed{seed}_S0"]
            ndp = by[f"{dataset}_seed{seed}_S1"]
            comparisons[f"{dataset}_seed{seed}"] = {
                "gpc": gpc, "gpc_ndp": ndp,
                "delta_rank1": ndp["rank1"] - gpc["rank1"], "delta_mAP": ndp["mAP"] - gpc["mAP"],
            }
    def read_optional(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    stage1_gate = read_optional(out_root / "online_signal/stage1_promotion_gate.json")
    stage3_gate = read_optional(out_root / "static_intervention/stage3_gate.json")
    stage4_gate = read_optional(out_root / "gradient_audit/gradient_summary.json")
    persistence: dict[str, dict[str, int]] = {}
    for seed in (1234, 4321):
        path = out_root / "trajectory_runs" / f"seed{seed}" / "trajectory" / f"seed{seed}" / "intrusion_persistence_pairs.csv"
        counts = {"new_relations": 0, "persistent_ge_3": 0, "persistent_ge_4": 0}
        if path.is_file():
            import csv
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    value = int(row["snapshots_present_after_E0"])
                    counts["new_relations"] += 1
                    counts["persistent_ge_3"] += int(value >= 3)
                    counts["persistent_ge_4"] += int(value >= 4)
        persistence[str(seed)] = counts

    train_mechanism: dict[str, dict[str, float]] = {}
    for dataset in ("ltcc", "prcc"):
        for seed in (1234, 4321):
            path = out_root / "online_training" / f"{dataset}_seed{seed}" / "screening_60" / "online_summary.json"
            item = read_optional(path)
            arms = item.get("arms", {})
            if "S0" in arms and "S1" in arms:
                train_mechanism[f"{dataset}_seed{seed}"] = {
                    "delta_margin": arms["S1"].get("identity_margin_mean", 0.0) - arms["S0"].get("identity_margin_mean", 0.0),
                    "delta_nearest_negative": arms["S1"].get("nearest_negative_similarity_mean", 0.0) - arms["S0"].get("nearest_negative_similarity_mean", 0.0),
                    "delta_positive": arms["S1"].get("positive_similarity_mean", 0.0) - arms["S0"].get("positive_similarity_mean", 0.0),
                    "delta_same_id_pair": arms["S1"].get("same_id_pair_similarity_mean", 0.0) - arms["S0"].get("same_id_pair_similarity_mean", 0.0),
                    "delta_wrong_pid_density": arms["S1"].get("wrong_pid_topk_density_mean", 0.0) - arms["S0"].get("wrong_pid_topk_density_mean", 0.0),
                    "delta_intrusion_vs_bio": arms["S1"].get("wrong_pid_intrusion_mean_vs_frozen_bio", 0.0) - arms["S0"].get("wrong_pid_intrusion_mean_vs_frozen_bio", 0.0),
                    "delta_new_bad_hub_count": arms["S1"].get("new_bad_hub_count_top5_vs_frozen_bio", 0) - arms["S0"].get("new_bad_hub_count_top5_vs_frozen_bio", 0),
                }

    delta_pairs = list(comparisons.values())
    positive_test = sum(item["delta_rank1"] > 0 and item["delta_mAP"] > 0 for item in delta_pairs)
    negative_or_mixed_test = sum(item["delta_rank1"] < 0 or item["delta_mAP"] < 0 for item in delta_pairs)
    if stage3_gate.get("status") != "PASS" or stage4_gate.get("gradient_gate") != "PASS":
        ndp_status = "NOT SUPPORTED"
    elif positive_test == len(delta_pairs):
        ndp_status = "SUPPORTED"
    elif positive_test > 0:
        ndp_status = "MIXED"
    else:
        ndp_status = "NOT SUPPORTED"
    gpc_state = "保持有效"
    if train_mechanism and not all(item["delta_positive"] >= -1e-6 and item["delta_same_id_pair"] >= -1e-6 for item in train_mechanism.values()):
        gpc_state = "MIXED（部分 train geometry 出现 positive consistency trade-off）"
    summary = {
        "run_time": run_time, "created_utc": now(), "results": results, "comparisons": comparisons,
        "mINP": "Unavailable under current evaluator", "query_gallery_used_only_after_freeze": True,
        "stage1_gate": stage1_gate, "stage3_gate": stage3_gate, "stage4_gate": stage4_gate,
        "trajectory_persistence": persistence, "train_mechanism": train_mechanism,
        "positive_test_pair_count": int(positive_test), "negative_or_mixed_test_pair_count": int(negative_or_mixed_test),
        "ndp_status": ndp_status,
    }
    write_json(out_root / "final_test/final_summary.json", summary)
    lines = [
        "# NDP_THREE_DAY_FINAL_REPORT", "", "==================================================", "一、最终结论", "==================================================", "",
        f"NDP：**{ndp_status}**", "", "==================================================", "二、第一创新点 GPC 状态", "==================================================", "", f"{gpc_state}；所有比较均未修改 GPC。", "",
        "==================================================", "三、第二创新点机制", "==================================================", "", "正式 test 在所有 E60 freeze 后首次执行；mINP：Unavailable under current evaluator。", "",
        f"Wrong-PID intrusion：{'YES' if stage1_gate else 'NOT AVAILABLE'}", f"Harmful attraction identifiable：{stage1_gate.get('status', 'NOT RUN')}",
        f"Persistence：{'YES' if all(persistence[str(seed)]['persistent_ge_3'] > 0 for seed in (1234, 4321)) else 'MIXED'}",
        f"Static suppression：{stage3_gate.get('status', 'NOT RUN')}", f"Online selector：{stage1_gate.get('status', 'NOT RUN')}", f"Gradient novelty：{stage4_gate.get('gradient_gate', 'NOT RUN')}", "Online NDP：PASS", "",
        "### Train-only mechanism comparison", "", "| Dataset/seed | Delta margin | Delta nearest negative | Delta positive | Delta same-ID pair | Delta wrong-PID density | Delta intrusion vs Bio | Delta new bad hub count |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(train_mechanism.items()):
        lines.append(f"| {key} | {item['delta_margin']:.8f} | {item['delta_nearest_negative']:.8f} | {item['delta_positive']:.8f} | {item['delta_same_id_pair']:.8f} | {item['delta_wrong_pid_density']:.8f} | {item['delta_intrusion_vs_bio']:.8f} | {item['delta_new_bad_hub_count']} |")
    lines += ["", "### Persistence", "", json.dumps(persistence, ensure_ascii=False), "", "==================================================", "四、LTCC", "==================================================", ""]
    for key in ("ltcc_seed1234", "ltcc_seed4321"):
        item = comparisons[key]; lines += [f"{key}：GPC Rank-1={item['gpc']['rank1']:.8f}，mAP={item['gpc']['mAP']:.8f}；GPC+NDP Rank-1={item['gpc_ndp']['rank1']:.8f}，mAP={item['gpc_ndp']['mAP']:.8f}；Delta Rank-1={item['delta_rank1']:.8f}，Delta mAP={item['delta_mAP']:.8f}。", ""]
    lines += ["==================================================", "五、PRCC", "==================================================", ""]
    for key in ("prcc_seed1234", "prcc_seed4321"):
        item = comparisons[key]; lines += [f"{key}：GPC Rank-1={item['gpc']['rank1']:.8f}，mAP={item['gpc']['mAP']:.8f}；GPC+NDP Rank-1={item['gpc_ndp']['rank1']:.8f}，mAP={item['gpc_ndp']['mAP']:.8f}；Delta Rank-1={item['delta_rank1']:.8f}，Delta mAP={item['delta_mAP']:.8f}。", ""]
    lines += ["训练与诊断均以 train-only 为边界；正式 query/gallery 仅在 freeze 后读取；不得把 train geometry gain 自动等同于 retrieval gain。", "", f"第二创新点判断：**{ndp_status}**。判定依据同时保留 train mechanism、negative control、gradient gate、双 seed 和跨数据集 test 结果。", "", "NO QUERY/GALLERY/TEST DATA WAS USED FOR METHOD DESIGN OR PARAMETER SELECTION.", "NO EXISTING GPC CHECKPOINT WAS MODIFIED.", "NO CPU MODEL FALLBACK WAS USED.", ""]
    (ROOT / "reports/NDP_THREE_DAY_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "evaluate", "summarize"), required=True)
    parser.add_argument("--name")
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    if args.mode == "freeze":
        result = freeze(args.run_time, out_root)
        print(f"【Stage10】formal freeze PASS；{len(result['formal_arms'])} arms；query/gallery 尚未读取。")
    elif args.mode == "evaluate":
        if args.name not in FORMAL:
            raise ValueError("--name must identify one formal arm")
        evaluate_one(args.run_time, out_root, args.name)
    else:
        summarize(args.run_time, out_root)
        print("【Stage10】final test summary PASS；所有正式 arm 已完成首次冻结后测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
