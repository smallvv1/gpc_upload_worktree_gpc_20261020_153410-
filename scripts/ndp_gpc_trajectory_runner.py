#!/usr/local/bin/python3
"""Stage 2 GPC trajectory runner with train-only feature snapshots.

It reuses the frozen repository GPC engine in-process without editing that
engine.  The only runtime overrides are the registered diagnostic seed and
30-epoch horizon.  Recovery/public model snapshots are emitted at E5, E10,
E15, E20, E25, and E30 by an in-memory source transformation of the recovery
interval; the frozen GPC loss and selector remain unchanged.  E0 is the
audited common initialization.  All feature extraction and geometry use the
LTCC train split only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
LOG_ROOT = ROOT / "logs"
SEEDS = (1234, 4321)
SNAPSHOTS = (0, 5, 10, 15, 20, 25, 30)
K = 10


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # noqa: E402
    return engine


def source_common(seed: int) -> Path:
    if seed == 1234:
        return ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/common_init/LTCC_COMMON_INIT_SEED1234.pth"
    return ROOT / "outputs/new_method_track/phase6cr_relation_replication/SEED4321_COMMON_INIT.pth"


def configure_engine(engine: Any, seed: int, train_root: Path, run_time: str) -> None:
    engine.SEED = int(seed)
    engine.EPOCHS = 30
    engine.WARMUP_EPOCHS = 20
    engine.LAMBDA_GPC = 0.1
    engine.OUTPUT_ROOT_DEFAULT = train_root
    # Redirect the engine's rank-0 file logger to the required global log dir;
    # no output-tree .log file is permitted by the NDP task contract.
    engine.public_log_path = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP训练轨迹诊断_{run_time}.log"


def prepare(seed: int, train_root: Path, run_time: str) -> None:
    engine = import_engine()
    configure_engine(engine, seed, train_root, run_time)
    train_root.mkdir(parents=True, exist_ok=True)
    # This audit constructs only the dataset metadata and train candidate
    # bank; the training function later creates a train-only image loader.
    engine.source_and_leakage_audit(train_root, "ltcc")
    target, meta_path = engine.init_paths(train_root, "ltcc")
    source = source_common(seed)
    if not source.is_file():
        raise FileNotFoundError(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"common init is not a state dict: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verified, _ = engine.load_common_init(train_root, "ltcc")
        if engine.state_hash(verified) != engine.state_hash(state):
            raise RuntimeError(f"existing trajectory common init mismatch: {target}")
    else:
        torch.save(state, target)
    metadata = {
        "dataset": "ltcc",
        "seed": seed,
        "created_utc": now(),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": engine.sha256_file(source),
        "file_sha256": engine.sha256_file(target),
        "state_hash": engine.state_hash(state),
        "tensor_hashes": engine.tensor_hashes(state),
        "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "strict_load_required": True,
        "diagnostic_only": True,
    }
    engine.write_json(meta_path, metadata)
    # E0 is an exact model-only copy of the common init, kept under the new
    # run tree; the source checkpoint itself is never modified.
    snapshot_root = train_root / "trajectory" / f"seed{seed}"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    e0 = snapshot_root / "model_epoch_00.pth"
    if not e0.exists():
        torch.save(state, e0)
    write_json(snapshot_root / "prepare.json", {
        "run_time": run_time,
        "seed": seed,
        "source_common_init": str(source),
        "source_common_init_sha256": engine.sha256_file(source),
        "target_common_init": str(target),
        "target_common_init_sha256": engine.sha256_file(target),
        "snapshots_required": list(SNAPSHOTS),
        "train_only": True,
        "query_gallery_read_for_features": False,
    })


def dynamic_training_function(engine: Any, run_time: str):
    source = inspect.getsource(engine.training_one)
    if "if epoch % 10 == 0:" not in source:
        raise RuntimeError("frozen GPC engine recovery interval was not found; refusing an ambiguous trajectory run")
    source = source.replace("if epoch % 10 == 0:", "if epoch % 5 == 0:")
    # The frozen engine has a literal "60" in its completion log.  Rewrite
    # only that in-memory message so this 30-epoch diagnostic is reported
    # truthfully; the GPC implementation and schedule remain untouched.
    source = source.replace(
        'logger.info("[%s][训练] 60 个 Epoch 已完整完成", dataset_name.upper())',
        'logger.info("[%s][训练] %d 个 Epoch 已完整完成", dataset_name.upper(), EPOCHS)',
    ).replace(
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：60；SHA256：%s", dataset_name.upper(), final_meta["checkpoint_sha256"])',
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：%d；SHA256：%s", dataset_name.upper(), EPOCHS, final_meta["checkpoint_sha256"])',
    )
    namespace = dict(engine.__dict__)
    namespace["public_log_path"] = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP训练轨迹诊断_{run_time}.log"
    exec(compile(source, "<ndp_dynamic_gpc_training>", "exec"), namespace)
    return namespace["training_one"]


def train(seed: int, train_root: Path, run_time: str) -> None:
    engine = import_engine()
    configure_engine(engine, seed, train_root, run_time)
    run_dir = engine.arm_dir(train_root, "ltcc", "gpc")
    dry_run_path = run_dir / "dry_run.json"
    if not dry_run_path.is_file() or not bool(engine.read_json(dry_run_path).get("passed")):
        # The registered GPC protocol uses three train-only smoke batches.
        # One batch can legitimately contain no selected relation; that is not
        # evidence that the frozen selector is broken.
        engine.dry_run_one(train_root, "ltcc", "gpc", 3)
    train_function = dynamic_training_function(engine, run_time)
    train_function(train_root, "ltcc", "gpc")


def load_model_bank(engine: Any, cfg: Any, dataset: Any, model_path: Path, output: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    model = engine.build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, transform_test = engine.build_img_transforms(cfg)
    loader = torch.utils.data.DataLoader(
        engine.ImageDataset(dataset.train, transform=transform_test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    features: list[np.ndarray] = []
    forward_devices: set[str] = set()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with engine.amp.autocast(enabled=True):
                model_output = model(images, cam_label=camids)
            if torch.is_tensor(model_output):
                identity = model_output
            elif isinstance(model_output, (tuple, list)) and len(model_output) == 2:
                _score, feat = model_output
                identity = feat[0] if isinstance(feat, (list, tuple)) else feat
            else:
                raise RuntimeError(f"unexpected model eval output: {type(output)!r}")
            identity = F.normalize(identity.float(), dim=-1)
            if identity.device.type != "cuda":
                raise RuntimeError("trajectory feature forward left PPU; CPU model fallback is forbidden")
            forward_devices.add(str(identity.device))
            features.append(identity.cpu().numpy().astype(np.float32))
    bank = {
        "features": np.concatenate(features, axis=0),
        "pid": np.asarray([int(item["pid"]) for item in dataset.train], dtype=np.int64),
        "clothes": np.asarray([int(item["clothes_id"]) for item in dataset.train], dtype=np.int64),
        "camid": np.asarray([int(item["camid"]) for item in dataset.train], dtype=np.int64),
        "image_path": np.asarray([str(item["image_path"]) for item in dataset.train]),
    }
    if bank["features"].shape != (len(dataset.train), 1024):
        raise RuntimeError(f"unexpected trajectory feature shape: {bank['features'].shape}")
    if not all("/train/" in path and "/query/" not in path and "/test/" not in path and "gallery" not in path.lower() for path in bank["image_path"]):
        raise RuntimeError("trajectory bank is not train-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bank)
    del model, loader
    torch.cuda.empty_cache()
    return {"path": str(output), "images": len(bank["pid"]), "dim": int(bank["features"].shape[1]), "forward_devices": sorted(forward_devices), "cpu_fallback": False}


def topk_graph(features: np.ndarray, pid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(pid)
    all_i = np.empty((n, k), dtype=np.int64)
    all_s = np.empty((n, k), dtype=np.float32)
    neg_i = np.empty((n, k), dtype=np.int64)
    neg_s = np.empty((n, k), dtype=np.float32)
    block = 128
    for start in range(0, n, block):
        stop = min(n, start + block)
        sim = features[start:stop] @ features.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        ai = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        av = np.take_along_axis(sim, ai, axis=1)
        order = np.argsort(-av, axis=1)
        all_i[start:stop] = np.take_along_axis(ai, order, axis=1)
        all_s[start:stop] = np.take_along_axis(av, order, axis=1)
        neg = sim.copy()
        neg[pid[start:stop, None] == pid[None, :]] = -np.inf
        ni = np.argpartition(-neg, kth=k - 1, axis=1)[:, :k]
        nv = np.take_along_axis(neg, ni, axis=1)
        order = np.argsort(-nv, axis=1)
        neg_i[start:stop] = np.take_along_axis(ni, order, axis=1)
        neg_s[start:stop] = np.take_along_axis(nv, order, axis=1)
    return all_i, all_s, neg_i, neg_s


def exact_pos_neg(features: np.ndarray, pid: np.ndarray, clothes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(pid)
    pos_cross = np.full(n, -np.inf, dtype=np.float32)
    pos_any = np.full(n, -np.inf, dtype=np.float32)
    neg = np.full(n, -np.inf, dtype=np.float32)
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        sim = features[start:stop] @ features.T
        same = pid[start:stop, None] == pid[None, :]
        same_any = same.copy()
        cross = same & (clothes[start:stop, None] != clothes[None, :])
        different = ~same
        same_any[np.arange(stop - start), np.arange(start, stop)] = False
        cross[np.arange(stop - start), np.arange(start, stop)] = False
        different[np.arange(stop - start), np.arange(start, stop)] = False
        sim_pos = sim.copy(); sim_pos[~cross] = -np.inf
        sim_any = sim.copy(); sim_any[~same_any] = -np.inf
        sim_neg = sim.copy(); sim_neg[~different] = -np.inf
        pos_cross[start:stop] = sim_pos.max(axis=1)
        pos_any[start:stop] = sim_any.max(axis=1)
        neg[start:stop] = sim_neg.max(axis=1)
    pos = np.where(np.isfinite(pos_cross), pos_cross, pos_any).astype(np.float32)
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise RuntimeError("trajectory train anchor lacks any same-PID positive or different-PID negative")
    return pos, neg


def snapshot_metrics(bank: dict[str, np.ndarray], all_i: np.ndarray, neg_i: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict[str, Any]:
    pid = bank["pid"]
    k = neg_i.shape[1]
    hub = np.bincount(neg_i.reshape(-1), minlength=len(pid)).astype(np.float32)
    same_recall = np.mean(pid[all_i] == pid[:, None], axis=1)
    return {
        "positive_similarity_mean": float(pos.mean()),
        "nearest_negative_similarity_mean": float(neg.mean()),
        "identity_margin_mean": float((pos - neg).mean()),
        "ordering_correctness": float(np.mean(pos > neg)),
        "same_id_topk_recall_mean": float(same_recall.mean()),
        "wrong_pid_topk_density_mean": float(1.0 - same_recall.mean()),
        "negative_hub_mean": float(hub.mean()),
        "negative_hub_std": float(hub.std()),
        "negative_hub_max": float(hub.max()),
        "negative_hub_p95": float(np.percentile(hub, 95)),
        "negative_hub_p99": float(np.percentile(hub, 99)),
        "negative_hub_total": int(hub.sum()),
        "K": k,
    }


def extract(seed: int, train_root: Path, run_time: str) -> None:
    engine = import_engine()
    configure_engine(engine, seed, train_root, run_time)
    cfg = engine.cfg_for("ltcc", train_root / "ltcc" / f"seed{seed}" / "gpc")
    dataset = engine.build_dataset(cfg)
    snapshot_root = train_root / "trajectory" / f"seed{seed}"
    model_root = train_root / "LTCC" / "checkpoints"
    model_paths = {0: snapshot_root / "model_epoch_00.pth"}
    for epoch in (5, 10, 15, 20, 25, 30):
        # The repository publisher uses the non-padded public names
        # model_epoch_5.pth, ..., model_epoch_30.pth.  Keep E0's local
        # diagnostic name padded, but resolve published snapshots exactly.
        model_paths[epoch] = model_root / f"model_epoch_{epoch}.pth"
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("trajectory extraction requires two visible PPUs; CPU fallback forbidden")
    torch.cuda.set_device(0)
    banks = {}
    for epoch in SNAPSHOTS:
        if not model_paths[epoch].is_file():
            raise RuntimeError(f"required trajectory snapshot missing: E{epoch}: {model_paths[epoch]}")
        output = snapshot_root / f"E{epoch:02d}_train_feature_bank.npz"
        banks[epoch] = load_model_bank(engine, cfg, dataset, model_paths[epoch], output, device)
    metric_rows = []
    graphs = {}
    raw_banks = {}
    for epoch in SNAPSHOTS:
        data = np.load(snapshot_root / f"E{epoch:02d}_train_feature_bank.npz", allow_pickle=False)
        bank = {key: data[key] for key in data.files}
        bank["image_path"] = bank["image_path"].astype(str)
        raw_banks[epoch] = bank
        all_i, all_s, neg_i, neg_s = topk_graph(bank["features"], bank["pid"], K)
        pos, neg = exact_pos_neg(bank["features"], bank["pid"], bank["clothes"])
        graphs[epoch] = {"all_i": all_i, "all_s": all_s, "neg_i": neg_i, "neg_s": neg_s, "pos": pos, "neg": neg}
        row = {"seed": seed, "epoch": epoch, **snapshot_metrics(bank, all_i, neg_i, pos, neg)}
        metric_rows.append(row)
        np.savez_compressed(snapshot_root / f"E{epoch:02d}_negative_top{K}.npz", indices=neg_i, scores=neg_s, pid=bank["pid"], image_path=bank["image_path"])
    write_csv(snapshot_root / "trajectory_train_metrics.csv", metric_rows)
    e0 = graphs[0]
    # Persistence is defined for *new* negative relations only.  Counting all
    # negative neighbors would conflate ordinary Bio hard negatives with the
    # GPC-induced intrusive relations that Stage 2 is meant to diagnose.
    e0_negative_sets = {anchor: set(map(int, e0["neg_i"][anchor].tolist())) for anchor in range(len(e0["neg_i"]))}
    persistence = defaultdict(int)
    last_seen = {}
    for epoch in SNAPSHOTS[1:]:
        current = graphs[epoch]["neg_i"]
        for anchor in range(len(current)):
            for neighbor in current[anchor].tolist():
                if int(neighbor) in e0_negative_sets[anchor]:
                    continue
                key = (int(anchor), int(neighbor))
                persistence[key] += 1
                last_seen[key] = epoch
    persistence_rows = [{"anchor": a, "neighbor": b, "snapshots_present_after_E0": count, "last_snapshot": last_seen[(a, b)], "anchor_pid": int(raw_banks[30]["pid"][a]), "neighbor_pid": int(raw_banks[30]["pid"][b]), "definition": "GPC snapshot negative Top-K relation absent from E0 Bio negative Top-K"} for (a, b), count in sorted(persistence.items())]
    write_csv(snapshot_root / "intrusion_persistence_pairs.csv", persistence_rows)
    final_delta_margin = graphs[30]["pos"] - graphs[30]["neg"] - (graphs[0]["pos"] - graphs[0]["neg"])
    per_anchor = []
    for anchor in range(len(final_delta_margin)):
        present = sum(1 for epoch in SNAPSHOTS[1:] if any(int(x) not in e0_negative_sets[anchor] for x in graphs[epoch]["neg_i"][anchor]))
        new_at_final = sum(1 for x in graphs[30]["neg_i"][anchor].tolist() if int(x) not in e0_negative_sets[anchor])
        per_anchor.append({"anchor": anchor, "pid": int(raw_banks[30]["pid"][anchor]), "delta_margin_E30_minus_E0": float(final_delta_margin[anchor]), "harmful_final": bool(final_delta_margin[anchor] < -1e-6), "snapshots_with_new_intrusion": present, "new_intrusion_count_E30": new_at_final})
    write_csv(snapshot_root / "trajectory_anchor_margin.csv", per_anchor)
    write_json(snapshot_root / "trajectory_manifest.json", {
        "run_time": run_time,
        "seed": seed,
        "dataset": "LTCC",
        "snapshots": list(SNAPSHOTS),
        "K": K,
        "feature_definition": "normalized feat[0]",
        "train_images": len(raw_banks[0]["pid"]),
        "feature_dim": int(raw_banks[0]["features"].shape[1]),
        "feature_banks": {str(epoch): str(snapshot_root / f"E{epoch:02d}_train_feature_bank.npz") for epoch in SNAPSHOTS},
        "model_snapshots": {str(epoch): str(model_paths[epoch]) for epoch in SNAPSHOTS},
        "metrics": str(snapshot_root / "trajectory_train_metrics.csv"),
        "persistence": str(snapshot_root / "intrusion_persistence_pairs.csv"),
        "query_gallery_read": False,
        "test_metrics_read": False,
        "cpu_model_fallback": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train", "extract"), required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--train-root", required=True)
    args = parser.parse_args()
    train_root = Path(args.train_root).resolve()
    if args.mode == "prepare":
        prepare(args.seed, train_root, args.run_time)
    elif args.mode == "train":
        train(args.seed, train_root, args.run_time)
    else:
        extract(args.seed, train_root, args.run_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
