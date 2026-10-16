#!/usr/local/bin/python3
"""使用冻结的 Bio+GPC E60 checkpoint 做 PRCC 同衣 B→A 测试。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_loader import ImageDataset  # noqa: E402
from model import build_model  # noqa: E402
from processor.processor import test as processor_test  # noqa: E402
from train_final_gpc import (  # noqa: E402
    EXPECTED_DEVICE_NAME,
    SEED,
    build_dataset,
    build_img_transforms,
    cfg_for,
)
from utils.metrics import R1_mAP_eval  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"PRCC_SAME_CLOTHES_{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-time", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/home/DIFFER/outputs/gpc_final_retrain/20260928_173834/PRCC/checkpoints/model_final.pth",
    )
    parser.add_argument(
        "--checkpoint-meta",
        default="/home/DIFFER/outputs/gpc_final_retrain/20260928_173834/PRCC/artifacts/最终检查点元数据_20260928_173834.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="默认写入 outputs/gpc_same_clothes_test/${run_time}",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("同衣测试要求 PPU/CUDA；禁止 CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"PPU 设备审计失败：期望 {EXPECTED_DEVICE_NAME}，实际 {device_name}")

    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = Path(args.checkpoint_meta).resolve()
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint 或元数据不存在：{checkpoint} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != "prcc" or metadata.get("epoch") != 60 or not metadata.get("frozen"):
        raise RuntimeError(f"checkpoint gate 失败：{metadata}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != metadata.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint SHA256 与冻结元数据不一致")

    root = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs/gpc_same_clothes_test" / args.run_time
    dataset_root = root / "PRCC"
    log_path = dataset_root / "logs" / f"PRCC_同衣测试_{args.run_time}.log"
    result_path = dataset_root / "test" / f"PRCC_同衣测试结果_{args.run_time}.json"
    protocol_path = root / "protocol" / "protocol.json"
    report_path = root / "reports" / f"PRCC_同衣测试报告_{args.run_time}.md"
    logger = make_logger(log_path)

    cfg = cfg_for("prcc", root)
    dataset = build_dataset(cfg)
    query = dataset.query_same
    gallery = dataset.gallery
    if len(query) != 3873 or len(gallery) != 3384:
        raise RuntimeError(f"PRCC 同衣 query/gallery 数量异常：{len(query)}/{len(gallery)}")

    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, transform_test = build_img_transforms(cfg)
    loader = DataLoader(
        ConcatDataset([ImageDataset(query, transform=transform_test), ImageDataset(gallery, transform=transform_test)]),
        batch_size=int(cfg.DATA.TEST_BATCH),
        shuffle=False,
        num_workers=int(cfg.DATA.NUM_WORKERS),
        pin_memory=False,
        drop_last=False,
    )
    evaluator = R1_mAP_eval(len(query), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    started = utc_now()
    logger.info("==================================================")
    logger.info("数据集：PRCC")
    logger.info("方法：Bio + GPC")
    logger.info("测试类型：同衣测试")
    logger.info("测试协议：query_same（B） -> gallery（A）")
    logger.info("query 数：%d；gallery 数：%d", len(query), len(gallery))
    logger.info("checkpoint：%s", checkpoint)
    logger.info("checkpoint epoch：60")
    logger.info("checkpoint SHA256：%s", checkpoint_sha256)
    logger.info("PPU：是；CPU fallback：禁止；设备：%s", device_name)
    logger.info("本次只做测试，不训练、不更新参数、不选择 checkpoint")
    logger.info("==================================================")

    evaluator.reset()
    processor_test(cfg, model, evaluator, loader, logger, device, test=True, clothChanging=True)
    cmc, mAP, *_ = evaluator.compute()
    finished = utc_now()
    result = {
        "run_time": args.run_time,
        "dataset": "PRCC",
        "method": "Bio + GPC",
        "protocol": "PRCC same clothes B -> A gallery",
        "query_definition": "query_same / camera B / same clothes as gallery A",
        "query_count": len(query),
        "gallery_count": len(gallery),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": 60,
        "checkpoint_sha256": checkpoint_sha256,
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "map": float(mAP),
        "minp": None,
        "minp_note": "当前 evaluator 未提供",
        "evaluator": "processor.processor.test + current utils.metrics evaluator",
        "evaluator_calls": 1,
        "device": device_name,
        "cpu_fallback": False,
        "training_started": False,
        "checkpoint_selection_used": False,
        "started_utc": started,
        "finished_utc": finished,
    }
    write_json(result_path, result)
    protocol = {
        "run_time": args.run_time,
        "dataset": "PRCC",
        "test_type": "same_clothes",
        "protocol": "query_same B -> gallery A",
        "seed": SEED,
        "method": "Bio + GPC",
        "gpc_enabled_in_checkpoint": True,
        "lambda_gpc_at_training": 0.1,
        "checkpoint_epoch": 60,
        "checkpoint_frozen": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "query_count": len(query),
        "gallery_count": len(gallery),
        "cpu_fallback": False,
        "training_started": False,
        "checkpoint_selection_used": False,
        "started_utc": started,
        "finished_utc": finished,
    }
    write_json(protocol_path, protocol)
    report = "\n".join([
        "# PRCC 同衣测试报告",
        "",
        f"RUN_TIME：{args.run_time}",
        f"测试时间：{started} 至 {finished}",
        "",
        "## 测试协议",
        "",
        "使用 PRCC 的 query_same（B 摄像机、与 A gallery 同衣）测试，不进行训练，不更新参数，不选择 checkpoint。",
        "",
        f"query：{len(query)}；gallery：{len(gallery)}；checkpoint epoch：60。",
        f"checkpoint SHA256：`{checkpoint_sha256}`",
        "",
        "## 结果",
        "",
        "| Rank-1 | Rank-5 | Rank-10 | mAP | mINP |",
        "|---:|---:|---:|---:|---|",
        f"| {100 * result['rank1']:.4f}% | {100 * result['rank5']:.4f}% | {100 * result['rank10']:.4f}% | {100 * result['map']:.4f}% | 当前 evaluator 未提供 |",
        "",
        "## 完整性审计",
        "",
        "- 使用冻结的 PRCC E60 Bio+GPC checkpoint：是",
        "- CPU fallback：否",
        "- 本次测试是否训练或更新参数：否",
        "- 是否使用测试指标选择 checkpoint：否",
        "",
        "## 文件",
        "",
        f"- JSON：`{result_path}`",
        f"- 中文日志：`{log_path}`",
        f"- protocol：`{protocol_path}`",
    ]) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("[PRCC][同衣测试] Rank-1：%.8f；Rank-5：%.8f；Rank-10：%.8f；mAP：%.8f；mINP：当前 evaluator 未提供", result["rank1"], result["rank5"], result["rank10"], result["map"])
    logger.info("[PRCC][同衣测试] 测试完成；结果：%s", result_path)
    print(json.dumps({"result": result, "report": str(report_path), "protocol": str(protocol_path), "log": str(log_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
