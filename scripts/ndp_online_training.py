#!/usr/local/bin/python3
"""Train-only NDP screening/formal runner with a frozen Bio memory.

The only online selector implemented here is registered Stage1 S2 together
with the frozen-Bio intrusion condition: a current Top-10 different-PID
neighbor is selected when it is absent from the frozen Bio negative Top-10 and
its current-vs-Bio attraction increase is positive.  The frozen Bio bank is
never updated.  Current train memory is refreshed at epoch 1 and every fifth
epoch.  No query/gallery/test object is constructed.
"""

from __future__ import annotations

import argparse
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
ARMS = ("S0", "S1", "S2", "S3")


def horizon_dir(epochs: int, k: int) -> str:
    return f"screening_{epochs}" if int(k) == 10 else f"screening_{epochs}_K{k}"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def identity_feature_from_output(model_output: Any) -> torch.Tensor:
    """Extract identity features from frozen-model eval or train outputs."""
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


def load_bio_bank(seed: int, dataset_name: str = "ltcc") -> dict[str, np.ndarray]:
    if dataset_name == "prcc":
        if int(seed) == 4321:
            path = ROOT / "outputs/new_method_track/phase7n_a_headroom_benefit_audit/dataset_headroom/feature_banks/PRCC_M0_seed4321_train_features.npz"
        else:
            path = ROOT / "outputs/new_method_track/phase7b_relation_utility_audit/relation_headroom/prcc_p0_seed1234_train_features.npz"
        data = np.load(path, allow_pickle=False)
        bank = {key: data[key] for key in data.files}
        bank["features"] = np.asarray(bank["features"], dtype=np.float32)
        if "pids" in bank:
            bank["pid"] = np.asarray(bank["pids"], dtype=np.int64)
        else:
            bank["pid"] = np.asarray(bank["pid"], dtype=np.int64)
        bank["clothes"] = np.asarray(bank["clothes"], dtype=np.int64)
        if "paths" in bank:
            bank["image_path"] = bank["paths"].astype(str)
        else:
            bank["image_path"] = bank["image_path"].astype(str)
    else:
        reference_seed = int(seed) if int(seed) in (1234, 4321) else 1234
        data = np.load(NDP_A / f"feature_banks/seed{reference_seed}/bio_feature_bank.npz", allow_pickle=False)
        bank = {key: data[key] for key in data.files}
        bank["features"] = np.asarray(bank["features"], dtype=np.float32)
        bank["pid"] = np.asarray(bank["pid"], dtype=np.int64)
        bank["clothes"] = np.asarray(bank["clothes"], dtype=np.int64)
        bank["image_path"] = bank["image_path"].astype(str)
    expected = 17896 if dataset_name == "prcc" else 9576
    if bank["features"].shape != (expected, 1024):
        raise RuntimeError(f"frozen Bio bank shape mismatch: {bank['features'].shape}")
    if not all("/train/" in path and "/query/" not in path and "/test/" not in path and "gallery" not in path.lower() for path in bank["image_path"]):
        raise RuntimeError("frozen Bio bank is not train-only")
    return bank


def cross_positive(bank: dict[str, np.ndarray], index: int) -> float:
    members = np.flatnonzero((bank["pid"] == bank["pid"][index]) & (bank["clothes"] != bank["clothes"][index]))
    members = members[members != index]
    if len(members) == 0:
        return float("nan")
    return float(np.max(bank["features"][members] @ bank["features"][index]))


def import_engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # noqa: E402
    return engine


def configure_engine(engine: Any, train_root: Path, run_time: str, epochs: int, seed: int) -> None:
    engine.SEED = int(seed)
    engine.EPOCHS = int(epochs)
    engine.WARMUP_EPOCHS = 20
    engine.LAMBDA_GPC = 0.1
    engine.OUTPUT_ROOT_DEFAULT = train_root
    engine.public_log_path = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP在线训练_{run_time}.log"


def copy_common_init(engine: Any, train_root: Path, dataset_name: str) -> None:
    seed = int(engine.SEED)
    if dataset_name == "prcc":
        source = ROOT / ("outputs/final_confirmatory_gpc/prcc/seed1234/common_init/PRCC_COMMON_INIT_SEED1234.pth" if seed == 1234 else "outputs/new_method_track/phase7m_sparse_positive_consistency_replication/initialization/PRCC_COMMON_INIT_SEED4321.pth")
    else:
        source = ROOT / ("outputs/final_confirmatory_gpc/ltcc/seed1234/common_init/LTCC_COMMON_INIT_SEED1234.pth" if seed == 1234 else "outputs/new_method_track/phase6cr_relation_replication/SEED4321_COMMON_INIT.pth")
    state = torch.load(source, map_location="cpu", weights_only=False)
    target, meta_path = engine.init_paths(train_root, dataset_name)
    if target.is_file() and meta_path.is_file():
        current = torch.load(target, map_location="cpu", weights_only=False)
        if engine.state_hash(current) != engine.state_hash(state):
            raise RuntimeError(f"online common init mismatch: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, target)
    engine.write_json(meta_path, {
        "dataset": dataset_name, "seed": seed, "created_utc": now(),
        "source_checkpoint": str(source), "source_checkpoint_sha256": engine.sha256_file(source),
        "file_sha256": engine.sha256_file(target), "state_hash": engine.state_hash(state),
        "tensor_hashes": engine.tensor_hashes(state), "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())), "strict_load_required": True,
    })


def prepare_arm(arm: str, run_time: str, out_root: Path, epochs: int, seed: int, dataset_name: str = "ltcc", k: int = 10) -> None:
    train_root = out_root / "online_training" / f"{dataset_name}_seed{seed}" / horizon_dir(epochs, k) / arm
    engine = import_engine()
    configure_engine(engine, train_root, run_time, epochs, seed)
    train_root.mkdir(parents=True, exist_ok=True)
    engine.source_and_leakage_audit(train_root, dataset_name)
    copy_common_init(engine, train_root, dataset_name)
    lambda_path = out_root / "gradient_audit/NDP_LAMBDA_FREEZE.json"
    if not lambda_path.is_file():
        raise RuntimeError(f"frozen NDP lambda missing: {lambda_path}")
    lambda_value = float(json.loads(lambda_path.read_text(encoding="utf-8"))["lambda_ndp"])
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise RuntimeError(f"invalid frozen NDP lambda: {lambda_value}")
    write_json(train_root / "online_protocol.json", {
        "run_time": run_time, "dataset": dataset_name.upper(), "seed": seed, "arm": arm,
        "epochs": epochs, "K": k, "lambda_gpc": 0.1, "lambda_ndp": lambda_value,
        "selector": "Stage1 S2 negative attraction increase + frozen Bio intrusion condition" if arm in ("S1", "S2", "S3") else "none",
        "current_memory_refresh": "epoch 1 and every 5 epochs; train-only",
        "query_gallery_read": False, "test_metrics_read": False, "gpc_modified": False,
    })


def build_memory_hook(engine: Any, cfg: Any, dataset: Any, device: torch.device, memory: dict[str, Any]) -> Any:
    train_items = list(dataset.train)
    _, transform_test = engine.build_img_transforms(cfg)
    loader = torch.utils.data.DataLoader(engine.ImageDataset(train_items, transform=transform_test), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(train_items)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    n = len(train_items)

    def refresh(model: Any, cfg_local: Any, dataset_local: Any, epoch: int, rank_local: int, device_local: torch.device) -> None:
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
        # Pick a deterministic current same-ID cross-clothes positive for each
        # train anchor.  The relation loss uses this memory vector when the
        # selected negative is not in the current mini-batch; requiring both
        # endpoints to co-occur would make a train-wide relation selector
        # almost always inactive under the frozen sampler.
        positive_index = np.full(n, -1, dtype=np.int64)
        for pid_value in np.unique(memory["pid"]):
            members = np.flatnonzero(memory["pid"] == pid_value)
            for anchor in members.tolist():
                candidates = members[memory["clothes"][members] != memory["clothes"][anchor]]
                if len(candidates):
                    scores = memory["features"][candidates] @ memory["features"][anchor]
                    positive_index[anchor] = int(candidates[int(np.argmax(scores))])
        # LTCC contains train images whose PID has no second clothing state.
        # They remain in the train bank, but cannot provide a cross-clothes
        # positive.  Keep -1 and exclude only those anchors from relation
        # selection; this matches the frozen Bio-reference validity rule.
        memory["positive_valid_count"] = int(np.count_nonzero(positive_index >= 0))
        memory["positive_invalid_count"] = int(np.count_nonzero(positive_index < 0))
        memory["positive_index"] = positive_index
        memory["epoch"] = int(epoch)
        memory["refresh_count"] = int(memory.get("refresh_count", 0)) + 1
        if was_training:
            model.train()

    memory["path_to_index"] = path_to_index
    memory["pid"] = np.asarray([int(item["pid"]) for item in train_items], dtype=np.int64)
    memory["clothes"] = np.asarray([int(item["clothes_id"]) for item in train_items], dtype=np.int64)
    memory["paths"] = np.asarray([str(item["image_path"]) for item in train_items])
    memory["refresh"] = refresh
    return refresh


def choose_online_pairs(entries: list[dict[str, Any]], memory: dict[str, Any], bio: dict[str, np.ndarray], randomize: bool, epoch: int, rank: int, iteration: int) -> tuple[list[tuple[int, int, float]], int]:
    current = memory.get("features")
    if current is None:
        return [], 0
    path_to_index = memory["path_to_index"]
    batch_pid = [entry["pid"] for entry in entries]
    candidate: list[tuple[int, int, float]] = []
    for left, entry in enumerate(entries):
        global_left = path_to_index[entry["image_path"]]
        sim_current = current[global_left] @ current.T
        sim_bio = bio["features"][global_left] @ bio["features"].T
        neg_mask = memory["pid"] != entry["pid"]
        sim_current = sim_current.copy(); sim_current[~neg_mask] = -np.inf
        sim_bio = sim_bio.copy(); sim_bio[~neg_mask] = -np.inf
        top = np.argpartition(-sim_current, kth=K - 1)[:K]
        top = top[np.argsort(-sim_current[top])]
        bio_top = set(np.argpartition(-sim_bio, kth=K - 1)[:K].tolist())
        for global_right in top.tolist():
            if int(global_right) in bio_top:
                continue
            if memory["pid"][global_right] == batch_pid[left]:
                continue
            delta = float(sim_current[global_right] - sim_bio[global_right])
            if delta <= 0.0:
                continue
            if int(memory["positive_index"][global_left]) < 0:
                continue
            pos_ref = cross_positive(bio, global_left)
            if not np.isfinite(pos_ref):
                continue
            # The negative endpoint is a train-memory vector and need not be
            # present in the same mini-batch as the live anchor.
            candidate.append((left, int(global_right), float(pos_ref - sim_bio[global_right])))
    # Remove duplicate directed pairs but preserve deterministic ordering.
    unique = {(left, right): (left, right, ref) for left, right, ref in candidate}
    candidate = [unique[key] for key in sorted(unique)]
    target = len(candidate)
    if not randomize:
        return candidate, target
    all_pairs = [(left, right) for left in range(len(entries)) for right in np.flatnonzero(memory["pid"] != batch_pid[left]).tolist()]
    target = min(target, len(all_pairs))
    if target == 0:
        return [], 0
    rng = np.random.default_rng(20260930 + epoch * 10007 + rank * 1009 + iteration)
    selected_indices = rng.choice(len(all_pairs), size=target, replace=False)
    random_pairs = []
    for position in selected_indices.tolist():
        left, right = all_pairs[int(position)]
        gl = path_to_index[entries[left]["image_path"]]
        gr = int(right)
        pos_ref = cross_positive(bio, gl)
        if np.isfinite(pos_ref):
            ref_sim = float(bio["features"][gl] @ bio["features"][gr])
            random_pairs.append((left, right, float(pos_ref - ref_sim)))
    return random_pairs, target


def make_forward(engine: Any, mode: str, base_arm: str, bio: dict[str, np.ndarray], memory: dict[str, Any], lambda_ndp: float, stats: dict[str, Any]) -> Any:
    def forward(model, loss_fn, batch, device, bank, arm, epoch, rank, iteration, global_step):
        result = engine.forward_loss(model, loss_fn, batch, device, bank, base_arm, epoch, rank, iteration, global_step)
        if mode == "S0":
            return result
        entries = engine.batch_entries(batch)
        pairs, target = choose_online_pairs(entries, memory, bio, randomize=(mode == "S2"), epoch=epoch, rank=rank, iteration=iteration)
        images = batch["image"].to(device, non_blocking=False)
        camids = batch["camid"].to(device, non_blocking=False)
        with engine.amp.autocast(enabled=True):
            model_output = model(images, cam_label=camids)
        representation = F.normalize(identity_feature_from_output(model_output).float(), dim=-1)
        pid = [entry["pid"] for entry in entries]
        losses = []
        for left, global_right, m_ref in pairs:
            global_left = memory["path_to_index"][entries[left]["image_path"]]
            global_positive = int(memory["positive_index"][global_left])
            if global_positive < 0:
                continue
            positive_memory = torch.as_tensor(memory["features"][global_positive], dtype=representation.dtype, device=device)
            negative_memory = torch.as_tensor(memory["features"][global_right], dtype=representation.dtype, device=device)
            current_margin = torch.sum(representation[left] * positive_memory) - torch.sum(representation[left] * negative_memory)
            losses.append(F.relu(torch.as_tensor(m_ref, dtype=current_margin.dtype, device=device) - current_margin))
        if losses:
            ndp = torch.stack(losses).mean()
            stats["active_batches"] += 1
            stats["violating_relations"] += int(sum(float(item.detach().cpu()) > 0 for item in losses))
        else:
            ndp = representation.sum() * 0.0
        stats["batches"] += 1; stats["target_relations"] += int(target); stats["selected_relations"] += int(len(pairs)); stats["loss_sum"] += float(ndp.detach().float().cpu())
        result["total"] = result["total"] + lambda_ndp * ndp
        result["ndp_loss"] = ndp
        result["ndp_active"] = bool(losses)
        return result
    return forward


def dynamic_training_function(engine: Any, run_time: str, custom_forward: Any, hook: Any) -> Any:
    source = inspect.getsource(engine.training_one)
    if "if epoch % 10 == 0:" not in source:
        raise RuntimeError("frozen GPC recovery interval not found")
    source = source.replace("if epoch % 10 == 0:", "if epoch % 5 == 0:")
    # The repository GPC engine keeps a legacy literal 60 in its final
    # training log. Rewrite only that string in this diagnostic wrapper so
    # 30-epoch screening and robustness logs report their true horizon.
    source = source.replace(
        'logger.info("[%s][训练] 60 个 Epoch 已完整完成", dataset_name.upper())',
        'logger.info("[%s][训练] %d 个 Epoch 已完整完成", dataset_name.upper(), EPOCHS)',
    ).replace(
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：60；SHA256：%s", dataset_name.upper(), final_meta["checkpoint_sha256"])',
        'logger.info("[%s][训练] model_final.pth 已冻结；Epoch：%d；SHA256：%s", dataset_name.upper(), EPOCHS, final_meta["checkpoint_sha256"])',
    )
    marker = "        for epoch in range(1, EPOCHS + 1):"
    if marker not in source:
        raise RuntimeError("training epoch marker not found")
    source = source.replace(marker, marker + "\n            online_hook(model, cfg, dataset, epoch, rank, device)", 1)
    namespace = dict(engine.__dict__)
    namespace["forward_loss"] = custom_forward
    namespace["online_hook"] = hook
    namespace["public_log_path"] = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP在线训练_{run_time}.log"
    exec(compile(source, "<ndp_online_training>", "exec"), namespace)
    return namespace["training_one"]


def train_arm(arm: str, run_time: str, out_root: Path, epochs: int, seed: int, dataset_name: str = "ltcc", k: int = 10) -> None:
    train_root = out_root / "online_training" / f"{dataset_name}_seed{seed}" / horizon_dir(epochs, k) / arm
    engine = import_engine()
    configure_engine(engine, train_root, run_time, epochs, seed)
    base_arm = "bio" if arm == "S3" else "gpc"
    run_dir = engine.arm_dir(train_root, dataset_name, base_arm)
    dry_path = run_dir / "dry_run.json"
    if not dry_path.is_file() or not bool(engine.read_json(dry_path).get("passed")):
        engine.dry_run_one(train_root, dataset_name, base_arm, 3)
    lambda_path = out_root / "gradient_audit/NDP_LAMBDA_FREEZE.json"
    lambda_ndp = float(json.loads(lambda_path.read_text(encoding="utf-8"))["lambda_ndp"])
    bio = load_bio_bank(seed, dataset_name)
    memory: dict[str, Any] = {}
    cfg = engine.cfg_for(dataset_name, run_dir)
    dataset = engine.build_dataset(cfg)
    train_paths = np.asarray([str(item["image_path"]) for item in dataset.train])
    if not np.array_equal(train_paths, bio["image_path"]):
        raise RuntimeError(f"online frozen Bio/train path alignment failed for {dataset_name}/seed{seed}")
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    if arm == "S0":
        # The matched GPC control has no selector-dependent loss.  Avoid an
        # unnecessary train-bank refresh while preserving the same model,
        # sampler, optimizer, scheduler, and epoch protocol.
        hook = lambda _model, _cfg, _dataset, _epoch, _rank, _device: None
    else:
        hook = build_memory_hook(engine, cfg, dataset, device, memory)
    stats: dict[str, Any] = {"batches": 0, "active_batches": 0, "target_relations": 0, "selected_relations": 0, "violating_relations": 0, "loss_sum": 0.0}
    custom = make_forward(engine, arm, base_arm, bio, memory, lambda_ndp, stats)
    training = dynamic_training_function(engine, run_time, custom, hook)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    training(train_root, dataset_name, base_arm)
    write_json(train_root / f"online_train_stats_rank{rank}.json", {"run_time": run_time, "dataset": dataset_name, "seed": seed, "arm": arm, "rank": rank, "lambda_ndp": lambda_ndp, "memory_refreshes": memory.get("refresh_count", 0), "positive_valid_count": memory.get("positive_valid_count", 0), "positive_invalid_count": memory.get("positive_invalid_count", 0), **stats, "train_only": True, "query_gallery_read": False, "test_metrics_read": False})


def analyze_arm(arm: str, run_time: str, out_root: Path, epochs: int, seed: int, dataset_name: str = "ltcc", k: int = 10) -> dict[str, Any]:
    train_root = out_root / "online_training" / f"{dataset_name}_seed{seed}" / horizon_dir(epochs, k) / arm
    base_arm = "bio" if arm == "S3" else "gpc"
    engine = import_engine()
    configure_engine(engine, train_root, run_time, epochs, seed)
    cfg = engine.cfg_for(dataset_name, train_root / dataset_name / f"seed{seed}" / base_arm)
    dataset = engine.build_dataset(cfg)
    model_path = train_root / dataset_name / f"seed{seed}" / base_arm / "model_final.pth"
    if not model_path.is_file():
        raise RuntimeError(f"online final checkpoint missing: {model_path}")
    from ndp_gpc_trajectory_runner import load_model_bank
    from ndp_static_intervention import topk_metrics
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("online analysis requires two visible PPUs")
    torch.cuda.set_device(0)
    bank_path = train_root / "final_train_feature_bank.npz"
    info = load_model_bank(engine, cfg, dataset, model_path, bank_path, device)
    data = np.load(bank_path, allow_pickle=False)
    bank = {key: data[key] for key in data.files}
    metrics = topk_metrics(bank["features"], bank["pid"], K)
    serial = {key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)}
    serial.update({"arm": arm, "epochs": epochs, "seed": seed, "dataset": dataset_name, "checkpoint": str(model_path), "forward_devices": info["forward_devices"], "cpu_fallback": False})
    bio = load_bio_bank(seed, dataset_name)
    bio_metrics = topk_metrics(bio["features"], bio["pid"], K)

    def negative_topk(features: np.ndarray, pids: np.ndarray, top_k: int) -> np.ndarray:
        result = np.empty((len(pids), top_k), dtype=np.int64)
        for start in range(0, len(pids), 256):
            stop = min(len(pids), start + 256)
            sim = features[start:stop] @ features.T
            sim[pids[start:stop, None] == pids[None, :]] = -np.inf
            idx = np.argpartition(-sim, kth=top_k - 1, axis=1)[:, :top_k]
            val = np.take_along_axis(sim, idx, axis=1)
            order = np.argsort(-val, axis=1)
            result[start:stop] = np.take_along_axis(idx, order, axis=1)
        return result

    current_neg = negative_topk(bank["features"], bank["pid"], K)
    bio_neg = negative_topk(bio["features"], bio["pid"], K)
    intrusion = np.asarray([
        sum(int(neighbor) not in set(bio_neg[anchor].tolist()) for neighbor in current_neg[anchor]) / K
        for anchor in range(len(bank["pid"]))
    ], dtype=np.float32)
    current_hub = np.bincount(current_neg.reshape(-1), minlength=len(bank["pid"])).astype(np.float32)
    bio_hub = np.bincount(bio_neg.reshape(-1), minlength=len(bio["pid"])).astype(np.float32)
    bio_top5 = bio_hub >= np.percentile(bio_hub, 95)
    current_top5 = current_hub >= np.percentile(current_hub, 95)
    serial.update({
        "frozen_bio_checkpoint_role": "Bio train feature bank",
        "wrong_pid_intrusion_mean_vs_frozen_bio": float(intrusion.mean()),
        "wrong_pid_intrusion_anchor_pct_vs_frozen_bio": float(np.mean(intrusion > 0.0)),
        "new_bad_hub_count_top5_vs_frozen_bio": int(np.sum(current_top5 & ~bio_top5)),
        "new_bad_hub_anchor_occurrence_share": float(current_hub[current_top5 & ~bio_top5].sum() / max(current_hub.sum(), 1.0)),
        "delta_nearest_negative_vs_frozen_bio": float(metrics["nearest_negative_similarity_mean"] - bio_metrics["nearest_negative_similarity_mean"]),
        "delta_margin_vs_frozen_bio": float(metrics["identity_margin_mean"] - bio_metrics["identity_margin_mean"]),
        "delta_positive_vs_frozen_bio": float(metrics["positive_similarity_mean"] - bio_metrics["positive_similarity_mean"]),
    })
    rank_stats = []
    for rank in (0, 1):
        path = train_root / f"online_train_stats_rank{rank}.json"
        if path.is_file():
            rank_stats.append(json.loads(path.read_text(encoding="utf-8")))
    if rank_stats:
        batches = sum(int(item.get("batches", 0)) for item in rank_stats)
        active_batches = sum(int(item.get("active_batches", 0)) for item in rank_stats)
        serial.update({
            "ndp_active_rate": active_batches / max(batches, 1),
            "ndp_batches": batches,
            "ndp_active_batches": active_batches,
            "ndp_target_relations": sum(int(item.get("target_relations", 0)) for item in rank_stats),
            "ndp_selected_relations": sum(int(item.get("selected_relations", 0)) for item in rank_stats),
            "ndp_violating_relations": sum(int(item.get("violating_relations", 0)) for item in rank_stats),
            "memory_refreshes_per_rank": [int(item.get("memory_refreshes", 0)) for item in rank_stats],
        })
    write_json(train_root / "online_train_metrics.json", serial)
    return serial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train", "analyze"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(1234, 4321, 2026), default=1234)
    parser.add_argument("--dataset", choices=("ltcc", "prcc"), default="ltcc")
    parser.add_argument("--k", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    global K
    K = int(args.k)
    out_root = Path(args.out_root).resolve()
    if args.mode == "prepare":
        for arm in ARMS:
            prepare_arm(arm, args.run_time, out_root, args.epochs, args.seed, args.dataset, args.k)
    elif args.mode == "train":
        if args.arm is None:
            raise ValueError("--arm required in train mode")
        train_arm(args.arm, args.run_time, out_root, args.epochs, args.seed, args.dataset, args.k)
        print(f"【在线 NDP】{args.arm} {args.epochs} epoch train PASS；PPU-only、train-only。")
    else:
        metrics = {arm: analyze_arm(arm, args.run_time, out_root, args.epochs, args.seed, args.dataset, args.k) for arm in ARMS}
        write_json(out_root / "online_training" / f"{args.dataset}_seed{args.seed}" / horizon_dir(args.epochs, args.k) / "online_summary.json", {"run_time": args.run_time, "epochs": args.epochs, "seed": args.seed, "K": args.k, "dataset": args.dataset, "arms": metrics, "query_gallery_read": False, "test_metrics_read": False})
        print(f"【在线 NDP】{args.epochs} epoch train-only analysis PASS；4 arms={len(metrics)}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
