#!/usr/local/bin/python3
"""NDP-C2 gradient independence and train-only lambda calibration."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/DIFFER")
LOG_ROOT = ROOT / "logs"
SEED = 1234
LOCAL_BATCHES = 100


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
        writer.writeheader(); writer.writerows(rows)


def import_engine():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_final_gpc as engine  # noqa: E402
    return engine


def source_common() -> Path:
    return ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/common_init/LTCC_COMMON_INIT_SEED1234.pth"


def configure_engine(engine: Any, root: Path, run_time: str) -> None:
    engine.SEED = SEED
    engine.EPOCHS = 1
    engine.OUTPUT_ROOT_DEFAULT = root
    engine.public_log_path = lambda _out, _dataset, _kind: LOG_ROOT / f"NDP梯度独立性检查_{run_time}.log"


def prepare(run_time: str, out_root: Path) -> None:
    root = out_root / "gradient_audit"
    engine = import_engine()
    configure_engine(engine, root, run_time)
    root.mkdir(parents=True, exist_ok=True)
    engine.source_and_leakage_audit(root, "ltcc")
    target, meta_path = engine.init_paths(root, "ltcc")
    source = source_common()
    state = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if target.is_file() and meta_path.is_file():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        if engine.state_hash(existing) != engine.state_hash(state):
            raise RuntimeError("gradient audit common init hash mismatch")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, target)
    engine.write_json(meta_path, {
        "dataset": "ltcc", "seed": SEED, "created_utc": now(),
        "source_checkpoint": str(source), "source_checkpoint_sha256": engine.sha256_file(source),
        "file_sha256": engine.sha256_file(target), "state_hash": engine.state_hash(state),
        "tensor_hashes": engine.tensor_hashes(state), "state_key_count": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())), "strict_load_required": True,
    })
    relation_path = out_root / "static_intervention/stable_harmful_relations.json"
    if not relation_path.is_file():
        raise RuntimeError(f"gradient audit requires stable relation artifact: {relation_path}")
    relation = json.loads(relation_path.read_text(encoding="utf-8"))
    write_json(root / "gradient_protocol.json", {
        "run_time": run_time, "dataset": "LTCC", "seed": SEED,
        "global_valid_batches": 200, "local_batches_per_ppu": LOCAL_BATCHES,
        "relation_file": str(relation_path), "relation_count": len(relation.get("relations", [])),
        "optimizer_step_called": False, "query_gallery_read": False, "test_metrics_read": False,
    })


def relation_loss(engine: Any, model_output: Any, batch: dict[str, Any], device: torch.device, relation: dict[str, Any], memory: dict[str, Any]) -> tuple[torch.Tensor, int]:
    _score, feat = model_output
    representation = F.normalize(feat[0].float(), dim=-1)
    entries = engine.batch_entries(batch)
    path_to_global = memory["path_to_index"]
    pid = [entry["pid"] for entry in entries]
    values = []
    for row in relation.get("relations", []):
        left = next((index for index, entry in enumerate(entries) if entry["image_path"] == str(row["anchor_path"])), None)
        global_left = path_to_global.get(str(row["anchor_path"]))
        global_right = path_to_global.get(str(row["neighbor_path"]))
        if left is None or global_left is None or global_right is None or memory["pid"][global_left] == memory["pid"][global_right]:
            continue
        global_positive = int(memory["positive_index"][global_left])
        if global_positive < 0:
            continue
        positive_memory = torch.as_tensor(memory["features"][global_positive], dtype=representation.dtype, device=device)
        negative_memory = torch.as_tensor(memory["features"][global_right], dtype=representation.dtype, device=device)
        current_margin = torch.sum(representation[left] * positive_memory) - torch.sum(representation[left] * negative_memory)
        values.append(F.relu(torch.as_tensor(float(row["m_ref"]), dtype=current_margin.dtype, device=device) - current_margin))
    if values:
        return torch.stack(values).mean(), len(values)
    return representation.sum() * 0.0, 0


def flat_grad(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    pieces = []
    for parameter, value in zip(parameters, values):
        pieces.append(torch.zeros_like(parameter).reshape(-1) if value is None else value.detach().reshape(-1))
    return torch.cat(pieces)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.detach().cpu()) == 0.0 or float(right_norm.detach().cpu()) == 0.0:
        return float("nan")
    return float((torch.sum(left * right) / (left_norm * right_norm)).detach().cpu())


def run_rank(run_time: str, out_root: Path) -> None:
    root = out_root / "gradient_audit"
    relation_path = out_root / "static_intervention/stable_harmful_relations.json"
    relation = json.loads(relation_path.read_text(encoding="utf-8"))
    engine = import_engine()
    configure_engine(engine, root, run_time)
    rank, device = engine.runtime_gate(require_world_size=True)
    torch.cuda.set_device(rank)
    cfg = engine.cfg_for("ltcc", root / "ltcc" / f"seed{SEED}" / "gradient")
    dataset = engine.build_dataset(cfg)
    state, _audit = engine.load_common_init(root, "ltcc")
    raw = engine.build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    raw.load_state_dict(state, strict=True)
    raw.to(device).train()
    # Stable relations are train-bank directed pairs.  Their two endpoints
    # almost never co-occur in the frozen mini-batch sampler, so use a
    # detached current train memory for non-anchor endpoints.  This keeps the
    # gradient audit about the registered relation loss rather than accidental
    # batch co-occurrence.
    from ndp_static_intervention import build_memory_hook
    memory: dict[str, Any] = {}
    memory_hook = build_memory_hook(engine, cfg, dataset, device, memory)
    memory_hook(raw, cfg, dataset, 1, rank, device)
    loss_fn, _center = engine.make_loss(cfg, num_classes=dataset.num_train_pids)
    bank = engine.build_train_only_pair_bank(dataset, "ltcc")
    sampler = engine.FinalBatchSampler(dataset.train, engine.SPEC["ltcc"]["batch_size"], engine.SPEC["ltcc"]["num_instances"], engine.SPEC["ltcc"]["steps_per_epoch"], SEED, rank)
    sampler.prepare(1)
    loader = engine.build_train_loader(cfg, dataset, sampler)
    parameters = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    rows: list[dict[str, Any]] = []
    iterator = iter(loader)
    for local_index in range(LOCAL_BATCHES):
        batch = next(iterator)
        raw.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=False)
        pids = batch["pid"].to(device, non_blocking=False)
        camids = batch["camid"].to(device, non_blocking=False)
        clothes = batch["clothes_id"].to(device, non_blocking=False)
        captions = batch["caption_feature"].to(device, non_blocking=False)
        with engine.amp.autocast(enabled=True):
            score, feat = raw(images, cam_label=camids)
        base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=engine.MetricCapture(), step=local_index)
        components = getattr(loss_fn, "last_components", {})
        triplet = components.get("triplet", base_total * 0.0)
        entries = engine.batch_entries(batch)
        representation = F.normalize(feat[0].float(), dim=-1)
        selected, _info = engine.select_gpc_pairs(entries, bank, 1, rank, local_index + 1, SEED)
        if selected:
            gpc = torch.stack([1.0 - torch.sum(representation[left] * representation[right]) for left, right, _image, _state in selected]).mean()
        else:
            gpc = representation.sum() * 0.0
        ndp, ndp_pairs = relation_loss(engine, (score, feat), batch, device, relation, memory)
        g_base = flat_grad(base_total, parameters, retain_graph=True)
        g_triplet = flat_grad(triplet, parameters, retain_graph=True)
        g_gpc = flat_grad(gpc, parameters, retain_graph=True)
        g_ndp = flat_grad(ndp, parameters, retain_graph=False)
        rows.append({
            "rank": rank, "local_batch": local_index, "global_batch": rank * LOCAL_BATCHES + local_index,
            "base_norm": float(torch.linalg.vector_norm(g_base).cpu()),
            "triplet_norm": float(torch.linalg.vector_norm(g_triplet).cpu()),
            "gpc_norm": float(torch.linalg.vector_norm(g_gpc).cpu()),
            "ndp_norm": float(torch.linalg.vector_norm(g_ndp).cpu()),
            "cos_triplet_ndp": cosine(g_triplet, g_ndp), "cos_gpc_ndp": cosine(g_gpc, g_ndp),
            "ndp_loss": float(ndp.detach().float().cpu()), "ndp_pairs": int(ndp_pairs),
            "finite": bool(torch.isfinite(g_base).all().item() and torch.isfinite(g_triplet).all().item() and torch.isfinite(g_gpc).all().item() and torch.isfinite(g_ndp).all().item()),
            "optimizer_step_called": False, "train_only": True,
        })
    write_csv(root / f"gradient_batches_rank{rank}.csv", rows)
    del loader, raw, dataset
    torch.cuda.empty_cache()


def summarize(run_time: str, out_root: Path) -> dict[str, Any]:
    root = out_root / "gradient_audit"
    rows: list[dict[str, Any]] = []
    for rank in (0, 1):
        path = root / f"gradient_batches_rank{rank}.csv"
        if not path.is_file():
            raise RuntimeError(f"gradient batch file missing: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(list(csv.DictReader(stream)))
    if len(rows) < 200:
        raise RuntimeError(f"gradient audit has only {len(rows)} batches; required 200")
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows if row[key] not in ("", "nan", "NaN")], dtype=np.float64)
    cos_triplet = values("cos_triplet_ndp")
    cos_gpc = values("cos_gpc_ndp")
    base_norm = values("base_norm")
    ndp_norm = values("ndp_norm")
    ratios = base_norm[(ndp_norm > 0) & (base_norm > 0)] / ndp_norm[(ndp_norm > 0) & (base_norm > 0)]
    if len(ratios) == 0:
        lambda_ndp = float("nan")
    else:
        lambda_ndp = float(0.15 * np.median(ratios))
    median_triplet = float(np.median(cos_triplet)) if len(cos_triplet) else float("nan")
    median_gpc = float(np.median(cos_gpc)) if len(cos_gpc) else float("nan")
    lambda_valid = bool(np.isfinite(lambda_ndp) and lambda_ndp > 0.0)
    gate = {
        "batch_count": len(rows), "finite_batch_count": sum(row["finite"].lower() == "true" for row in rows),
        "median_cos_triplet_ndp": median_triplet, "median_cos_gpc_ndp": median_gpc,
        "triplet_redundancy_gate": "FAIL" if np.isfinite(median_triplet) and median_triplet >= 0.90 else "PASS",
        "lambda_valid": lambda_valid,
        "gradient_gate": "FAIL" if not np.isfinite(median_triplet) or median_triplet >= 0.90 or not lambda_valid else "PASS",
        "lambda_ndp": lambda_ndp, "lambda_target_fraction": 0.15,
        "lambda_calibration_train_only": True, "optimizer_step_called": False,
        "query_gallery_read": False, "test_metrics_read": False,
    }
    write_json(root / "NDP_LAMBDA_FREEZE.json", {"run_time": run_time, "dataset": "LTCC", "seed": SEED, **gate, "frozen": True})
    write_json(root / "gradient_summary.json", {"run_time": run_time, **gate})
    report = [
        "# NDP_C_GRADIENT_INDEPENDENCE_REPORT", "", f"RUN_TIME：`{run_time}`", f"主日志：`{LOG_ROOT / f'NDP有害异身份吸引三日验证_{run_time}.log'}`", "",
        f"有效 train batches：**{len(rows)}**；每个 batch 均不调用 optimizer.step；模型 forward/backward-linked gradient 只在 PPU。", "",
        f"median cos(g_triplet,g_ndp)：**{median_triplet:.8f}**；median cos(g_gpc,g_ndp)：**{median_gpc:.8f}**。", "",
        f"Gradient Gate：**{gate['gradient_gate']}**；lambda_NDP 一次性 train-only 冻结为：**{lambda_ndp:.8g}**（目标 raw NDP gradient 约为 base 的 15%）。", "",
        "没有读取 query/gallery/test；没有根据 retrieval 指标调 lambda；没有修改 GPC。", "",
    ]
    (ROOT / "reports/NDP_C_GRADIENT_INDEPENDENCE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run", "summarize"), required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    if args.mode == "prepare":
        prepare(args.run_time, out_root)
    elif args.mode == "run":
        run_rank(args.run_time, out_root)
    else:
        result = summarize(args.run_time, out_root)
        print(f"【Stage4】Gradient Gate：{result['gradient_gate']}；median cos(triplet,NDP)={result['median_cos_triplet_ndp']:.6f}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
