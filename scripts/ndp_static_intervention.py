#!/usr/local/bin/python3
"""NDP-C1/C2 train-only static harmful-relation intervention.

This runner deliberately keeps the repository GPC engine untouched.  C0, C1,
and C2 all call the frozen GPC training path with the same LTCC seed-1234
initialization.  C1 adds only the oracle-like cross-seed stable harmful
relation penalty; C2 receives the same per-batch relation budget with a
deterministic random different-PID assignment.
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
NDP_A = ROOT / "outputs/new_method_track/ndp_a_neighborhood_feasibility"
LOG_ROOT = ROOT / "logs"
SEED = 1234
K = 10
ARMS = ("C0", "C1", "C2")
BOOTSTRAP_REPS = 400


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def identity_feature_from_output(model_output: Any) -> torch.Tensor:
    """Extract the Bio identity feature in both eval and train modes.

    The frozen model returns a tensor in eval mode, while its training path
    returns ``(score_output, feat_output)``.  Treating the eval tensor as a
    two-item tuple attempts to unpack the batch dimension and caused the
    deterministic C1 memory-refresh failure.  This helper changes only output
    parsing; it does not change the model, loss, optimizer, or data protocol.
    """
    if torch.is_tensor(model_output):
        identity = model_output
    elif isinstance(model_output, (tuple, list)) and len(model_output) == 2:
        _score, feat = model_output
        identity = feat[0] if isinstance(feat, (tuple, list)) else feat
    else:
        raise RuntimeError(f"unexpected model output while extracting identity feature: {type(model_output)!r}")
    if not torch.is_tensor(identity) or identity.ndim != 2:
        raise RuntimeError(f"unexpected identity feature shape/type: {type(identity)!r}, {getattr(identity, 'shape', None)}")
    return identity


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


def load_bank(seed: int, role: str) -> dict[str, np.ndarray]:
    path = NDP_A / "feature_banks" / f"seed{seed}" / f"{role}_feature_bank.npz"
    data = np.load(path, allow_pickle=False)
    bank = {key: data[key] for key in data.files}
    bank["features"] = np.asarray(bank["features"], dtype=np.float32)
    bank["pid"] = np.asarray(bank["pid"], dtype=np.int64)
    bank["clothes"] = np.asarray(bank["clothes"], dtype=np.int64)
    bank["image_path"] = bank["image_path"].astype(str)
    if bank["features"].shape != (9576, 1024):
        raise RuntimeError(f"unexpected train bank shape: {path}: {bank['features'].shape}")
    if not all("/train/" in item and "/query/" not in item and "/test/" not in item and "gallery" not in item.lower() for item in bank["image_path"]):
        raise RuntimeError(f"static intervention train-only path audit failed: {path}")
    return bank


def load_intrusive_rows(seed: int, bank: dict[str, np.ndarray]) -> dict[tuple[str, str], dict[str, Any]]:
    path = ROOT / "outputs/new_method_track/ndp_three_day_20260930_130427" / "harmful_pairs" / f"seed{seed}_intrusive_pairs_K10.csv"
    # The path is replaced by the caller for a different RUN_TIME.  This
    # function is retained as a small compatibility helper for direct use.
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_intrusive_rows_from_path(path, bank)


def load_intrusive_rows_from_path(path: Path, bank: dict[str, np.ndarray]) -> dict[tuple[str, str], dict[str, Any]]:
    index_to_path = {index: str(value) for index, value in enumerate(bank["image_path"])}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            anchor = int(row["anchor"])
            neighbor = int(row["neighbor"])
            if not bool(json.loads(row["harmful_anchor"].lower())):
                continue
            if bank["pid"][anchor] == bank["pid"][neighbor]:
                continue
            key = (index_to_path[anchor], index_to_path[neighbor])
            rows[key] = {
                "anchor_path": key[0],
                "neighbor_path": key[1],
                "anchor": anchor,
                "neighbor": neighbor,
                "anchor_pid": int(bank["pid"][anchor]),
                "neighbor_pid": int(bank["pid"][neighbor]),
                "harmful_anchor": True,
                "delta_margin_anchor": float(row["delta_margin_anchor"]),
                "delta_s_neg_pair": float(row["delta_s_neg_pair"]),
                "reciprocal": bool(json.loads(row["reciprocal"].lower())),
                "source_csv": str(path),
            }
    return rows


def cross_clothes_positive(bank: dict[str, np.ndarray], index: int) -> float:
    members = np.flatnonzero((bank["pid"] == bank["pid"][index]) & (bank["clothes"] != bank["clothes"][index]))
    members = members[members != index]
    if len(members) == 0:
        return float("nan")
    return float(np.max(bank["features"][members] @ bank["features"][index]))


def build_stable_relations(run_time: str, out_root: Path) -> dict[str, Any]:
    bio1234 = load_bank(1234, "bio")
    bio4321 = load_bank(4321, "bio")
    if not np.array_equal(bio1234["image_path"], bio4321["image_path"]):
        raise RuntimeError("cross-seed train image order/path alignment failed")
    pair_root = out_root / "harmful_pairs"
    rows1234 = load_intrusive_rows_from_path(pair_root / "seed1234_intrusive_pairs_K10.csv", bio1234)
    rows4321 = load_intrusive_rows_from_path(pair_root / "seed4321_intrusive_pairs_K10.csv", bio4321)
    path_to_index = {str(path): index for index, path in enumerate(bio1234["image_path"])}
    stable: list[dict[str, Any]] = []
    for key in sorted(set(rows1234) & set(rows4321)):
        first, second = rows1234[key], rows4321[key]
        anchor = path_to_index[key[0]]
        neighbor = path_to_index[key[1]]
        pos = cross_clothes_positive(bio1234, anchor)
        if not np.isfinite(pos):
            continue
        ref_sim = float(bio1234["features"][anchor] @ bio1234["features"][neighbor])
        stable.append({
            "relation_id": len(stable),
            "anchor_path": key[0],
            "neighbor_path": key[1],
            "anchor_index": int(anchor),
            "neighbor_index": int(neighbor),
            "anchor_pid": int(bio1234["pid"][anchor]),
            "neighbor_pid": int(bio1234["pid"][neighbor]),
            "m_ref": float(pos - ref_sim),
            "bio_reference_positive": float(pos),
            "bio_reference_relation_similarity": ref_sim,
            "seed1234_delta_margin": float(first["delta_margin_anchor"]),
            "seed4321_delta_margin": float(second["delta_margin_anchor"]),
            "seed1234_delta_s_neg_pair": float(first["delta_s_neg_pair"]),
            "seed4321_delta_s_neg_pair": float(second["delta_s_neg_pair"]),
            "seed1234_reciprocal": bool(first["reciprocal"]),
            "seed4321_reciprocal": bool(second["reciprocal"]),
            "evidence": ["seed1234_intrusive_harmful", "seed4321_intrusive_harmful", "different_pid", "cross_clothes_valid_reference"],
        })
    relation_root = out_root / "static_intervention"
    relation_root.mkdir(parents=True, exist_ok=True)
    write_csv(relation_root / "stable_harmful_relations.csv", stable)
    write_json(relation_root / "stable_harmful_relations.json", {
        "run_time": run_time,
        "dataset": "LTCC",
        "seed_sources": [1234, 4321],
        "K": K,
        "relation_definition": "intersection of harmful intrusive train-only directed pairs from both seeds; different PID; valid cross-clothes Bio reference",
        "relation_count": len(stable),
        "query_gallery_read": False,
        "test_metrics_read": False,
        "relations": stable,
    })
    return {"relation_count": len(stable), "path": str(relation_root / "stable_harmful_relations.json"), "relations": stable}


def import_engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # noqa: E402
    return engine


def source_common() -> Path:
    return ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/common_init/LTCC_COMMON_INIT_SEED1234.pth"


def configure_engine(engine: Any, train_root: Path, run_time: str) -> None:
    engine.SEED = SEED
    engine.EPOCHS = 30
    engine.WARMUP_EPOCHS = 20
    engine.LAMBDA_GPC = 0.1
    engine.OUTPUT_ROOT_DEFAULT = train_root
    engine.public_log_path = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP静态干预验证_{run_time}.log"


def copy_common_init(engine: Any, train_root: Path) -> None:
    target, meta_path = engine.init_paths(train_root, "ltcc")
    source = source_common()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_state = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(source_state, dict) and isinstance(source_state.get("model"), dict):
        source_state = source_state["model"]
    if target.is_file() and meta_path.is_file():
        current = torch.load(target, map_location="cpu", weights_only=False)
        if engine.state_hash(current) != engine.state_hash(source_state):
            raise RuntimeError(f"static common init mismatch: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(source_state, target)
    meta = {
        "dataset": "ltcc", "seed": SEED, "created_utc": now(),
        "source_checkpoint": str(source), "source_checkpoint_sha256": engine.sha256_file(source),
        "file_sha256": engine.sha256_file(target), "state_hash": engine.state_hash(source_state),
        "tensor_hashes": engine.tensor_hashes(source_state), "state_key_count": len(source_state),
        "state_numel": int(sum(value.numel() for value in source_state.values())),
        "strict_load_required": True, "diagnostic_only": True,
    }
    engine.write_json(meta_path, meta)


def build_memory_hook(engine: Any, cfg: Any, dataset: Any, device: torch.device, memory: dict[str, Any]) -> Any:
    """Refresh a train-only current feature memory at epoch 1 and every 5 epochs.

    Stable relations are defined over the complete train bank, while the frozen
    sampler only places a small subset of images in each mini-batch.  Keeping
    the non-anchor endpoints as detached train-memory vectors makes the static
    intervention test the registered relations instead of testing whether two
    particular images happened to co-occur in a batch.
    """
    train_items = list(dataset.train)
    _, transform_test = engine.build_img_transforms(cfg)
    loader = torch.utils.data.DataLoader(
        engine.ImageDataset(train_items, transform=transform_test),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=False,
    )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    n = len(train_items)
    memory["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(train_items)}
    memory["pid"] = np.asarray([int(item["pid"]) for item in train_items], dtype=np.int64)
    memory["clothes"] = np.asarray([int(item["clothes_id"]) for item in train_items], dtype=np.int64)

    def refresh(model: Any, _cfg: Any, _dataset: Any, epoch: int, _rank: int, device_local: torch.device) -> None:
        if epoch != 1 and epoch % 5 != 0:
            return
        was_training = model.training
        model.eval()
        local = torch.zeros((n, 1024), dtype=torch.float32, device=device_local)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                count = len(batch["image_path"])
                indices = list(range(offset, offset + count))
                offset += count
                selected = [position for position, index in enumerate(indices) if index % world == rank]
                if not selected:
                    continue
                images = batch["image"][selected].to(device_local, non_blocking=False)
                camids = batch["camid"][selected].to(device_local, non_blocking=False)
                with engine.amp.autocast(enabled=True):
                    model_output = model(images, cam_label=camids)
                current = F.normalize(identity_feature_from_output(model_output).float(), dim=-1)
                local[torch.as_tensor([indices[position] for position in selected], device=device_local)] = current
        if world > 1:
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        memory["features"] = local.detach().cpu().numpy().astype(np.float32)
        positive_index = np.full(n, -1, dtype=np.int64)
        for pid_value in np.unique(memory["pid"]):
            members = np.flatnonzero(memory["pid"] == pid_value)
            for anchor in members.tolist():
                candidates = members[memory["clothes"][members] != memory["clothes"][anchor]]
                if len(candidates):
                    scores = memory["features"][candidates] @ memory["features"][anchor]
                    positive_index[anchor] = int(candidates[int(np.argmax(scores))])
        # Some LTCC train images have no cross-clothes image for the same PID.
        # They are valid train images but cannot supply the registered
        # cross-clothes positive reference.  Keep their index at -1 and let
        # the selected-relation path skip them; stable harmful relations were
        # built only from anchors with a valid Bio reference.
        memory["positive_valid_count"] = int(np.count_nonzero(positive_index >= 0))
        memory["positive_invalid_count"] = int(np.count_nonzero(positive_index < 0))
        memory["positive_index"] = positive_index
        memory["epoch"] = int(epoch)
        memory["refresh_count"] = int(memory.get("refresh_count", 0)) + 1
        if was_training:
            model.train()

    memory["refresh"] = refresh
    return refresh


def prepare_arm(arm: str, run_time: str, out_root: Path, relation_summary: dict[str, Any]) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    train_root = out_root / "static_intervention" / arm
    engine = import_engine()
    configure_engine(engine, train_root, run_time)
    train_root.mkdir(parents=True, exist_ok=True)
    engine.source_and_leakage_audit(train_root, "ltcc")
    copy_common_init(engine, train_root)
    write_json(train_root / "static_protocol.json", {
        "run_time": run_time, "dataset": "LTCC", "seed": SEED, "arm": arm,
        "relation_file": relation_summary["path"], "relation_count": relation_summary["relation_count"],
        "epochs": 30, "K": K, "query_gallery_read": False, "test_metrics_read": False,
        "gpc_definition": "frozen repository GPC, lambda=0.1",
        "static_loss": "mean(ReLU(m_ref - (cos(anchor,current_same_pid_positive)-cos(anchor,current_negative))))" if arm == "C1" else "none" if arm == "C0" else "matched random different-PID relation assignment",
    })


def dynamic_training_function(engine: Any, run_time: str, custom_forward: Any | None, hook: Any | None = None) -> Any:
    source = inspect.getsource(engine.training_one)
    if "if epoch % 10 == 0:" not in source:
        raise RuntimeError("frozen GPC recovery interval not found; refusing ambiguous static run")
    source = source.replace("if epoch % 10 == 0:", "if epoch % 5 == 0:")
    # Keep the frozen GPC engine unchanged while correcting its legacy
    # hard-coded 60-epoch completion text for this 30-epoch diagnostic.
    source = source.replace(
        'logger.info("[%s][训练] 60 个 Epoch 已完整完成", dataset_name.upper())',
        'logger.info("[%s][训练] %d 个 Epoch 已完整完成", dataset_name.upper(), EPOCHS)',
    ).replace(
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：60；SHA256：%s", dataset_name.upper(), final_meta["checkpoint_sha256"])',
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：%d；SHA256：%s", dataset_name.upper(), EPOCHS, final_meta["checkpoint_sha256"])',
    )
    namespace = dict(engine.__dict__)
    namespace["public_log_path"] = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP静态干预验证_{run_time}.log"
    if custom_forward is not None:
        namespace["forward_loss"] = custom_forward
    if hook is not None:
        marker = "        for epoch in range(1, EPOCHS + 1):"
        source = source.replace(marker, marker + "\n            static_hook(model, cfg, dataset, epoch, rank, device)", 1)
        namespace["static_hook"] = hook
    exec(compile(source, "<ndp_static_training>", "exec"), namespace)
    return namespace["training_one"]


def relation_maps(relation_summary: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], dict[str, int]]:
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ref_pos: dict[str, float] = {}
    pid: dict[str, int] = {}
    for row in relation_summary["relations"]:
        by_anchor[str(row["anchor_path"])].append(row)
        ref_pos[str(row["anchor_path"])] = float(row["bio_reference_positive"])
        pid[str(row["anchor_path"])] = int(row["anchor_pid"])
        pid[str(row["neighbor_path"])] = int(row["neighbor_pid"])
    return dict(by_anchor), ref_pos, pid


def make_static_forward(engine: Any, mode: str, relation_summary: dict[str, Any], run_time: str, seed: int, memory: dict[str, Any]) -> tuple[Any, dict[str, int]]:
    by_anchor, ref_pos, _pid_from_relation = relation_maps(relation_summary)
    bio_reference = load_bank(1234, "bio")
    bio_path_to_index = {str(path): index for index, path in enumerate(bio_reference["image_path"])}
    all_ref_pos = {
        str(path): cross_clothes_positive(bio_reference, index)
        for index, path in enumerate(bio_reference["image_path"])
    }
    stats = {"batches": 0, "active_batches": 0, "target_relations": 0, "selected_relations": 0, "violating_relations": 0, "loss_sum": 0.0}

    def static_forward(model, loss_fn, batch, device, bank, arm, epoch, rank, iteration, global_step):
        result = engine.forward_loss(model, loss_fn, batch, device, bank, "gpc", epoch, rank, iteration, global_step)
        if mode == "C0":
            return result
        entries = engine.batch_entries(batch)
        path_to_local = {entry["image_path"]: index for index, entry in enumerate(entries)}
        local_pid = [entry["pid"] for entry in entries]
        stable_local: list[tuple[int, int, float]] = []
        for entry in entries:
            for row in by_anchor.get(entry["image_path"], []):
                anchor_global = bio_path_to_index.get(entry["image_path"])
                neighbor_global = bio_path_to_index.get(str(row["neighbor_path"]))
                if anchor_global is None or neighbor_global is None:
                    continue
                anchor_local = path_to_local[entry["image_path"]]
                if memory["pid"][anchor_global] == memory["pid"][neighbor_global]:
                    continue
                stable_local.append((anchor_local, int(neighbor_global), float(row["m_ref"])))
        target_count = len(stable_local)
        selected: list[tuple[int, int, float]] = stable_local
        if mode == "C2":
            rng_seed = int(seed * 1000003 + epoch * 10007 + rank * 1009 + iteration)
            rng = np.random.default_rng(rng_seed)
            if target_count:
                selected = []
                for left, _stable_right, _stable_ref in stable_local:
                    anchor_index = bio_path_to_index[entries[left]["image_path"]]
                    candidates = np.flatnonzero(memory["pid"] != memory["pid"][anchor_index])
                    if not len(candidates):
                        continue
                    neighbor_index = int(candidates[int(rng.integers(len(candidates)))])
                    positive_ref = all_ref_pos.get(entries[left]["image_path"])
                    if positive_ref is None or not np.isfinite(positive_ref):
                        continue
                    ref_sim = float(bio_reference["features"][anchor_index] @ bio_reference["features"][neighbor_index])
                    selected.append((left, neighbor_index, float(positive_ref - ref_sim)))
        images = batch["image"].to(device, non_blocking=False)
        camids = batch["camid"].to(device, non_blocking=False)
        with engine.amp.autocast(enabled=True):
            model_output = model(images, cam_label=camids)
        representation = F.normalize(identity_feature_from_output(model_output).float(), dim=-1)
        losses = []
        for left, global_right, m_ref in selected:
            global_left = bio_path_to_index[entries[left]["image_path"]]
            global_positive = int(memory["positive_index"][global_left])
            if global_positive < 0:
                continue
            positive_memory = torch.as_tensor(memory["features"][global_positive], dtype=representation.dtype, device=device)
            negative_memory = torch.as_tensor(memory["features"][global_right], dtype=representation.dtype, device=device)
            current_margin = torch.sum(representation[left] * positive_memory) - torch.sum(representation[left] * negative_memory)
            losses.append(F.relu(torch.as_tensor(m_ref, dtype=current_margin.dtype, device=device) - current_margin))
        if losses:
            static_loss = torch.stack(losses).mean()
            stats["active_batches"] += 1
            stats["violating_relations"] += int(sum(float(item.detach().cpu()) > 0.0 for item in losses))
        else:
            static_loss = representation.sum() * 0.0
        stats["batches"] += 1
        stats["target_relations"] += int(target_count)
        stats["selected_relations"] += int(len(selected))
        stats["loss_sum"] += float(static_loss.detach().float().cpu())
        result["total"] = result["total"] + static_loss
        result["static_loss"] = static_loss
        result["ndp_active"] = bool(losses)
        result["static_target_relations"] = int(target_count)
        return result

    return static_forward, stats


def train_arm(arm: str, run_time: str, out_root: Path, relation_summary: dict[str, Any]) -> None:
    train_root = out_root / "static_intervention" / arm
    engine = import_engine()
    configure_engine(engine, train_root, run_time)
    run_dir = engine.arm_dir(train_root, "ltcc", "gpc")
    dry_path = run_dir / "dry_run.json"
    if not dry_path.is_file() or not bool(engine.read_json(dry_path).get("passed")):
        engine.dry_run_one(train_root, "ltcc", "gpc", 3)
    custom = None
    stats = {"batches": 0, "active_batches": 0, "target_relations": 0, "selected_relations": 0, "violating_relations": 0, "loss_sum": 0.0}
    memory: dict[str, Any] = {}
    hook = None
    if arm != "C0":
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        cfg = engine.cfg_for("ltcc", run_dir)
        dataset = engine.build_dataset(cfg)
        bio_reference = load_bank(1234, "bio")
        train_paths = np.asarray([str(item["image_path"]) for item in dataset.train])
        if not np.array_equal(train_paths, bio_reference["image_path"]):
            raise RuntimeError("static train/current-memory path alignment failed against frozen Bio train bank")
        hook = build_memory_hook(engine, cfg, dataset, device, memory)
        custom, stats = make_static_forward(engine, arm, relation_summary, run_time, SEED, memory)
    training = dynamic_training_function(engine, run_time, custom, hook)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    training(train_root, "ltcc", "gpc")
    # Parenthesize the formatted filename before joining with Path.  The
    # first C0 run completed all 30 epochs and froze model_final.pth, but the
    # old expression applied ``%`` to the Path object while writing the
    # train-only stats artifact.  This is a non-experimental bookkeeping fix;
    # it does not alter the model, loss, sampler, seed, or checkpoint.
    stats_path = train_root / ("static_train_stats_rank%d.json" % rank)
    write_json(stats_path, {"run_time": run_time, "arm": arm, "rank": rank, "memory_refreshes": memory.get("refresh_count", 0), "positive_valid_count": memory.get("positive_valid_count", 0), "positive_invalid_count": memory.get("positive_invalid_count", 0), **stats, "train_only": True, "query_gallery_read": False, "test_metrics_read": False})


def topk_metrics(features: np.ndarray, pid: np.ndarray, k: int = K) -> dict[str, Any]:
    n = len(pid)
    pos = np.full(n, -np.inf, dtype=np.float32)
    neg = np.full(n, -np.inf, dtype=np.float32)
    all_i = np.empty((n, k), dtype=np.int64)
    neg_i = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        sim = features[start:stop] @ features.T
        self_rows = np.arange(stop - start)
        self_cols = np.arange(start, stop)
        sim[self_rows, self_cols] = -np.inf
        same = pid[start:stop, None] == pid[None, :]
        pos_sim = sim.copy(); pos_sim[~same] = -np.inf
        neg_sim = sim.copy(); neg_sim[same] = -np.inf
        pos[start:stop] = pos_sim.max(axis=1)
        neg[start:stop] = neg_sim.max(axis=1)
        ai = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        av = np.take_along_axis(sim, ai, axis=1)
        order = np.argsort(-av, axis=1)
        all_i[start:stop] = np.take_along_axis(ai, order, axis=1)
        ni = np.argpartition(-neg_sim, kth=k - 1, axis=1)[:, :k]
        nv = np.take_along_axis(neg_sim, ni, axis=1)
        order = np.argsort(-nv, axis=1)
        neg_i[start:stop] = np.take_along_axis(ni, order, axis=1)
    same_recall = np.mean(pid[all_i] == pid[:, None], axis=1)
    hub = np.bincount(neg_i.reshape(-1), minlength=n).astype(np.float32)
    same_pair_mean = []
    for label in np.unique(pid):
        members = np.flatnonzero(pid == label)
        if len(members) > 1:
            sub = features[members] @ features[members].T
            same_pair_mean.append(float((sub.sum() - np.trace(sub)) / (len(members) * (len(members) - 1))))
    return {
        "positive_similarity_mean": float(np.mean(pos)),
        "nearest_negative_similarity_mean": float(np.mean(neg)),
        "identity_margin_mean": float(np.mean(pos - neg)),
        "ordering_correctness": float(np.mean(pos > neg)),
        "same_id_topk_recall_mean": float(np.mean(same_recall)),
        "wrong_pid_topk_density_mean": float(1.0 - np.mean(same_recall)),
        "same_id_pair_similarity_mean": float(np.mean(same_pair_mean)),
        "negative_hub_mean": float(hub.mean()),
        "negative_hub_std": float(hub.std()),
        "negative_hub_p95": float(np.percentile(hub, 95)),
        "negative_hub_p99": float(np.percentile(hub, 99)),
        "negative_hub_max": float(hub.max()),
        "pos_per_anchor": pos,
        "neg_per_anchor": neg,
        "margin_per_anchor": pos - neg,
    }


def analyze(run_time: str, out_root: Path, relation_summary: dict[str, Any]) -> dict[str, Any]:
    engine = import_engine()
    result: dict[str, Any] = {"run_time": run_time, "dataset": "LTCC", "seed": SEED, "arms": {}, "query_gallery_read": False, "test_metrics_read": False}
    for arm in ARMS:
        train_root = out_root / "static_intervention" / arm
        cfg = engine.cfg_for("ltcc", train_root / "ltcc" / f"seed{SEED}" / "gpc")
        dataset = engine.build_dataset(cfg)
        model_path = train_root / "ltcc" / f"seed{SEED}" / "gpc" / "model_final.pth"
        if not model_path.is_file():
            raise RuntimeError(f"static arm final checkpoint missing: {model_path}")
        from ndp_gpc_trajectory_runner import load_model_bank
        device = torch.device("cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("static analysis requires two visible PPU devices; CPU model fallback forbidden")
        torch.cuda.set_device(0)
        bank_path = train_root / "final_train_feature_bank.npz"
        bank_info = load_model_bank(engine, cfg, dataset, model_path, bank_path, device)
        data = np.load(bank_path, allow_pickle=False)
        bank = {key: data[key] for key in data.files}
        bank["image_path"] = bank["image_path"].astype(str)
        metrics = topk_metrics(bank["features"], bank["pid"], K)
        serial = {key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)}
        serial.update({"train_images": int(len(bank["pid"])), "feature_dim": int(bank["features"].shape[1]), "forward_devices": bank_info["forward_devices"], "cpu_fallback": False})
        serial["margin_per_anchor_path"] = str(train_root / "margin_per_anchor.npy")
        np.save(train_root / "margin_per_anchor.npy", metrics["margin_per_anchor"])
        result["arms"][arm] = serial
    c0 = result["arms"]["C0"]
    c1 = result["arms"]["C1"]
    c2 = result["arms"]["C2"]
    d1 = {"delta_margin": c1["identity_margin_mean"] - c0["identity_margin_mean"], "delta_neg": c1["nearest_negative_similarity_mean"] - c0["nearest_negative_similarity_mean"], "delta_pos": c1["positive_similarity_mean"] - c0["positive_similarity_mean"], "delta_same_id_pair": c1["same_id_pair_similarity_mean"] - c0["same_id_pair_similarity_mean"], "delta_wrong_pid_density": c1["wrong_pid_topk_density_mean"] - c0["wrong_pid_topk_density_mean"]}
    d2 = {"delta_margin": c2["identity_margin_mean"] - c0["identity_margin_mean"], "delta_neg": c2["nearest_negative_similarity_mean"] - c0["nearest_negative_similarity_mean"], "delta_pos": c2["positive_similarity_mean"] - c0["positive_similarity_mean"]}
    gate = {
        "relation_count": int(relation_summary["relation_count"]),
        "static_margin_gain_vs_C0": bool(d1["delta_margin"] > 1e-6),
        "static_nearest_negative_not_inflated_vs_C0": bool(d1["delta_neg"] <= 1e-6),
        "static_positive_consistency_preserved_vs_C0": bool(d1["delta_pos"] >= -1e-6 and d1["delta_same_id_pair"] >= -1e-6),
        "static_specificity_beats_random_control": bool(d1["delta_margin"] > d2["delta_margin"] + 1e-6),
        "status": "PASS" if relation_summary["relation_count"] > 0 and d1["delta_margin"] > 1e-6 and d1["delta_neg"] <= 1e-6 and d1["delta_pos"] >= -1e-6 and d1["delta_same_id_pair"] >= -1e-6 and d1["delta_margin"] > d2["delta_margin"] + 1e-6 else "FAIL",
        "decision_rule": "C1 margin gain, no nearest-negative inflation, no positive-consistency loss, and C1 margin gain greater than matched random C2; all train-only.",
    }
    result["deltas_C1_vs_C0"] = d1
    result["deltas_C2_vs_C0"] = d2
    result["gate"] = gate
    write_json(out_root / "static_intervention/stage3_summary.json", result)
    write_json(out_root / "static_intervention/stage3_gate.json", gate)
    report = [
        "# NDP_C_STATIC_INTERVENTION_REPORT", "", f"RUN_TIME：`{run_time}`", f"主日志：`{LOG_ROOT / f'NDP有害异身份吸引三日验证_{run_time}.log'}`", "",
        "Stage3 使用双 seed train-only intrusive+harmful 交集作为 oracle-like stable relation；C0 为 GPC，C1 为 selective relative margin preservation，C2 为 matched random different-PID control。", "",
        f"Stable relation 数：**{relation_summary['relation_count']}**。", "",
        "| Arm | Margin | Nearest negative | Positive similarity | Same-ID pair similarity | Wrong-PID Top-K density |", "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = result["arms"][arm]
        report.append(f"| {arm} | {item['identity_margin_mean']:.8f} | {item['nearest_negative_similarity_mean']:.8f} | {item['positive_similarity_mean']:.8f} | {item['same_id_pair_similarity_mean']:.8f} | {item['wrong_pid_topk_density_mean']:.8f} |")
    report += ["", f"C1 vs C0：`{json.dumps(d1, ensure_ascii=False)}`", f"C2 vs C0：`{json.dumps(d2, ensure_ascii=False)}`", "", f"Static Intervention Gate：**{gate['status']}**。", "", "只使用 LTCC train；没有读取 query/gallery/test feature、label、Rank-1 或 mAP；没有修改冻结 GPC。", ""]
    (ROOT / "reports/NDP_C_STATIC_INTERVENTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train", "analyze"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    relation_path = out_root / "static_intervention/stable_harmful_relations.json"
    if args.mode == "prepare":
        relation_summary = build_stable_relations(args.run_time, out_root)
        for arm in ARMS:
            prepare_arm(arm, args.run_time, out_root, relation_summary)
        print(f"【Stage3】stable harmful relation 构建完成：{relation_summary['relation_count']} 条；只来自 train。")
    else:
        if not relation_path.is_file():
            raise RuntimeError(f"stable relation artifact missing: {relation_path}")
        relation_summary = json.loads(relation_path.read_text(encoding="utf-8"))
        if args.mode == "train":
            if args.arm is None:
                raise ValueError("--arm is required in train mode")
            train_arm(args.arm, args.run_time, out_root, relation_summary)
            print(f"【Stage3】{args.arm} static intervention train PASS；PPU-only；train-only。")
        else:
            result = analyze(args.run_time, out_root, relation_summary)
            print(f"【Stage3】Static Intervention Gate：{result['gate']['status']}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
