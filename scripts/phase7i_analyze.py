#!/usr/bin/env python3
"""Offline Phase 7I trajectory, geometry, and gradient analysis.

The only model execution here is frozen-checkpoint feature extraction and
no-step gradient measurement on LTCC train batches.  It never calls an
optimizer or starts a distributed training process.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase6b_soft_relation_screen as b6  # noqa: E402
import phase7i_relation_specificity as p7i  # noqa: E402

base = b6.base
OUT_ROOT = p7i.OUT_ROOT
REPORT_PATH = p7i.REPORT_PATH
CFG = None

ARM_METRICS = {
    "I0": p7i.REFERENCE_DIRS["I0"] / "metrics_per_epoch.csv",
    "I1_frozen_reference": p7i.REFERENCE_DIRS["I1_frozen_reference"] / "metrics_per_epoch.csv",
    "I1": p7i.TRAIN_DIRS["I1"] / "metrics_per_epoch.csv",
    "I2": p7i.TRAIN_DIRS["I2"] / "metrics_per_epoch.csv",
    "I3": p7i.TRAIN_DIRS["I3"] / "metrics_per_epoch.csv",
}
ARM_MODELS = {
    "I0": p7i.REFERENCE_DIRS["I0"] / "model_final.pth",
    "I1": p7i.TRAIN_DIRS["I1"] / "model_final.pth",
    "I2": p7i.TRAIN_DIRS["I2"] / "model_final.pth",
    "I3": p7i.TRAIN_DIRS["I3"] / "model_final.pth",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p7i.json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: p7i.json_safe(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def trajectory(rows: list[dict[str, str]], start: int = 51, end: int = 60) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["epoch"]) <= end]
    if len(selected) != end - start + 1:
        raise RuntimeError(f"incomplete trajectory {start}-{end}: {len(selected)} rows")
    x = np.asarray([int(row["epoch"]) for row in selected], dtype=np.float64)
    r1 = np.asarray([float(row["cc_rank1"]) for row in selected], dtype=np.float64)
    m = np.asarray([float(row["cc_mAP"]) for row in selected], dtype=np.float64)
    return {
        "start": start, "end": end, "mean_rank1": float(r1.mean()), "mean_mAP": float(m.mean()),
        "auc_rank1": float(np.trapz(r1, x) / (x[-1] - x[0])), "auc_mAP": float(np.trapz(m, x) / (x[-1] - x[0])),
        "slope_rank1": float(np.polyfit(x, r1, 1)[0]), "slope_mAP": float(np.polyfit(x, m, 1)[0]),
        "e60_rank1": float(r1[-1]), "e60_mAP": float(m[-1]),
    }


def best_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    r1 = max(rows, key=lambda row: (float(row["cc_rank1"]), float(row["cc_mAP"])))
    m = max(rows, key=lambda row: (float(row["cc_mAP"]), float(row["cc_rank1"])))
    return {
        "best_rank1": float(r1["cc_rank1"]), "best_rank1_epoch": int(r1["epoch"]),
        "best_mAP": float(m["cc_mAP"]), "best_mAP_epoch": int(m["epoch"]),
    }


def analyze_trajectories() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for arm, path in ARM_METRICS.items():
        rows = read_csv(path)
        late = trajectory(rows)
        run = {"metrics": str(path), "rows": len(rows), "late": late, "best": best_metrics(rows), "amp_skips": sum(int(row["skipped_steps"]) for row in rows), "amp_found_inf": sum(int(row["found_inf_count"]) for row in rows), "amp_nonfinite": sum(int(row["nonfinite_events"]) for row in rows)}
        runs[arm] = run
        rows_out.append({"arm": arm, **late, **run["best"], "amp_skips": run["amp_skips"], "amp_found_inf": run["amp_found_inf"], "amp_nonfinite": run["amp_nonfinite"]})
    primary = {arm: runs[arm]["late"] for arm in ("I0", "I1", "I2", "I3")}
    deltas: list[dict[str, Any]] = []
    for left, right in (("I1", "I0"), ("I1", "I2"), ("I1", "I3"), ("I2", "I0"), ("I3", "I0")):
        deltas.append({
            "comparison": f"{left}-{right}",
            "delta_mean_rank1": primary[left]["mean_rank1"] - primary[right]["mean_rank1"],
            "delta_mean_mAP": primary[left]["mean_mAP"] - primary[right]["mean_mAP"],
            "delta_auc_rank1": primary[left]["auc_rank1"] - primary[right]["auc_rank1"],
            "delta_auc_mAP": primary[left]["auc_mAP"] - primary[right]["auc_mAP"],
            "delta_e60_rank1": primary[left]["e60_rank1"] - primary[right]["e60_rank1"],
            "delta_e60_mAP": primary[left]["e60_mAP"] - primary[right]["e60_mAP"],
        })
    write_csv(OUT_ROOT / "trajectory" / "late_window_summary.csv", rows_out)
    write_csv(OUT_ROOT / "tables" / "late_window_comparison.csv", [{"arm": arm, **primary[arm]} for arm in ("I0", "I1", "I2", "I3")])
    write_csv(OUT_ROOT / "tables" / "primary_deltas.csv", deltas)
    write_json(OUT_ROOT / "trajectory" / "trajectory_analysis.json", {"runs": runs, "deltas": deltas})
    return runs, deltas


def load_raw_model(path: Path, cfg: Any, device: torch.device) -> torch.nn.Module:
    raw = base.factory[cfg.MODEL.NAME](pretrained=False, config=cfg, num_classes=base.NUM_CLASSES, camera_num=base.NUM_CAMERAS)
    state = torch.load(path, map_location="cpu", weights_only=False)
    raw.load_state_dict(state, strict=True)
    raw.to(device).eval()
    return raw


def extract_feature_bank(arm: str, model_path: Path, dataset: Any, cfg: Any, device: torch.device) -> Path:
    out = OUT_ROOT / "geometry" / "feature_banks" / f"{arm}_train_features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        return out
    items = [dict(item) for item in dataset.train]
    _, transform_test = base.build_img_transforms(cfg)
    loader = DataLoader(base.ImageDataset(items, transform=transform_test), batch_size=cfg.DATA.TEST_BATCH, shuffle=False, num_workers=1, pin_memory=False)
    model = load_raw_model(model_path, cfg, device)
    all_features: list[np.ndarray] = []
    all_pid: list[np.ndarray] = []
    all_clothes: list[np.ndarray] = []
    all_cam: list[np.ndarray] = []
    all_paths: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                output = model(images, cam_label=camids)
            # The released graph returns the identity feature tensor in eval
            # mode, while training mode returns (score, feat).  Keep the
            # executed relation source: normalized feat[0].
            if torch.is_tensor(output):
                identity_feature = output
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                _score, eval_feat = output
                identity_feature = eval_feat[0] if isinstance(eval_feat, (tuple, list)) else eval_feat
            else:
                raise TypeError(f"Unexpected model output in feature extraction: {type(output)!r}")
            z = F.normalize(identity_feature.float(), dim=-1).cpu().numpy().astype(np.float32)
            all_features.append(z)
            all_pid.append(np.asarray(batch["pid"], dtype=np.int64))
            all_clothes.append(np.asarray(batch["clothes_id"], dtype=np.int64))
            all_cam.append(np.asarray(batch["camid"], dtype=np.int64))
            all_paths.extend(str(value) for value in batch["image_path"])
    np.savez_compressed(out, features=np.concatenate(all_features), pid=np.concatenate(all_pid), clothes=np.concatenate(all_clothes), camid=np.concatenate(all_cam), image_path=np.asarray(all_paths))
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    return out


def pair_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def pairwise_group_similarity(z: np.ndarray) -> tuple[int, float]:
    if len(z) < 2:
        return 0, float("nan")
    sim = z @ z.T
    upper = sim[np.triu_indices(len(z), k=1)]
    return len(upper), float(upper.mean())


def geometry_from_bank(bank_path: Path) -> dict[str, Any]:
    bank = np.load(bank_path, allow_pickle=False)
    z = np.asarray(bank["features"], dtype=np.float32)
    pids = np.asarray(bank["pid"], dtype=np.int64)
    clothes = np.asarray(bank["clothes"], dtype=np.int64)
    pid_values = sorted(set(int(value) for value in pids))
    state_groups: dict[tuple[int, int], np.ndarray] = {}
    pid_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for pid in pid_values:
        for cloth in sorted(set(int(value) for value in clothes[pids == pid])):
            indices = np.where((pids == pid) & (clothes == cloth))[0]
            state_groups[(pid, cloth)] = indices
            pid_groups[pid].append(indices)
    pid_centers = []
    for pid in pid_values:
        center = F.normalize(torch.from_numpy(z[pids == pid]).mean(dim=0, keepdim=True), dim=-1).numpy()[0]
        pid_centers.append(center)
    pid_centers = np.asarray(pid_centers, dtype=np.float32)
    center_sim = pid_centers @ pid_centers.T
    offdiag = center_sim[~np.eye(len(pid_values), dtype=bool)]
    center_index = {pid: i for i, pid in enumerate(pid_values)}
    image_center_sim = z @ pid_centers.T
    for idx, pid in enumerate(pids):
        image_center_sim[idx, center_index[int(pid)]] = -np.inf
    nearest_neg = np.max(image_center_sim, axis=1)
    cross_values: list[float] = []
    same_values: list[float] = []
    cross_margins: list[float] = []
    same_margins: list[float] = []
    state_centers: dict[tuple[int, int], np.ndarray] = {}
    state_dispersion: list[float] = []
    for key, indices in state_groups.items():
        state_centers[key] = F.normalize(torch.from_numpy(z[indices]).mean(dim=0, keepdim=True), dim=-1).numpy()[0]
        if len(indices) > 1:
            state_dispersion.extend((1.0 - z[indices] @ state_centers[key]).tolist())
        count, mean = pairwise_group_similarity(z[indices])
        if count:
            sim = z[indices] @ z[indices].T
            upper = sim[np.triu_indices(len(indices), k=1)]
            same_values.extend(upper.tolist())
            for pos, a in enumerate(indices):
                for b in indices[pos + 1:]:
                    same_margins.append(float(z[a] @ z[b] - max(nearest_neg[a], nearest_neg[b])))
    for pid in pid_values:
        state_keys = sorted(key for key in state_centers if key[0] == pid)
        for pos, key_a in enumerate(state_keys):
            idx_a = state_groups[key_a]
            for key_b in state_keys[pos + 1:]:
                idx_b = state_groups[key_b]
                sim = z[idx_a] @ z[idx_b].T
                cross_values.extend(sim.reshape(-1).tolist())
                for ia, a in enumerate(idx_a):
                    for ib, b in enumerate(idx_b):
                        cross_margins.append(float(sim[ia, ib] - max(nearest_neg[a], nearest_neg[b])))
    state_center_dispersion = []
    for pid in pid_values:
        keys = sorted(key for key in state_centers if key[0] == pid)
        for pos, key_a in enumerate(keys):
            for key_b in keys[pos + 1:]:
                state_center_dispersion.append(float(1.0 - state_centers[key_a] @ state_centers[key_b]))
    return {
        "train_images": len(z), "pids": len(pid_values), "clothing_states": len(state_groups),
        "same_clothes_pairs": len(same_values), "same_clothes_similarity": pair_mean(same_values),
        "cross_clothes_pairs": len(cross_values), "cross_clothes_similarity": pair_mean(cross_values),
        "nearest_negative_similarity": float(np.mean(nearest_neg)),
        "cross_clothes_margin": pair_mean(cross_margins), "same_clothes_margin": pair_mean(same_margins),
        "cross_clothes_ordering": float(np.mean(np.asarray(cross_margins) > 0)) if cross_margins else float("nan"),
        "same_clothes_ordering": float(np.mean(np.asarray(same_margins) > 0)) if same_margins else float("nan"),
        "clothing_state_center_dispersion": pair_mean(state_center_dispersion),
        "within_state_image_dispersion": pair_mean(state_dispersion),
        "pid_center_similarity": pair_mean(offdiag.tolist()),
        "pid_center_separation": float(1.0 - np.mean(offdiag)),
    }


def run_geometry(dataset: Any, cfg: Any, device: torch.device) -> dict[str, Any]:
    banks: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for arm in ("I0", "I1", "I2", "I3"):
        path = extract_feature_bank(arm, ARM_MODELS[arm], dataset, cfg, device)
        banks[arm] = path
        rows.append({"arm": arm, **geometry_from_bank(path)})
    write_csv(OUT_ROOT / "geometry" / "endpoint_geometry.csv", rows)
    write_json(OUT_ROOT / "geometry" / "endpoint_geometry.json", {row["arm"]: row for row in rows})
    return {row["arm"]: row for row in rows}


def group_name(name: str) -> str | None:
    if name.startswith("blocks."):
        return "backbone"
    if name.startswith("head."):
        return "id_head"
    if name.startswith("head_bio") or name.startswith("head_clip_bio"):
        return "bio_projection"
    return None


def late_group(name: str, late_ids: set[int]) -> bool:
    parts = name.split(".")
    return len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit() and int(parts[1]) in late_ids


def gradient_measure(named: list[tuple[str, torch.nn.Parameter]], grads: tuple[torch.Tensor | None, ...], groups: dict[str, list[int]]) -> dict[str, tuple[float, torch.Tensor | None]]:
    result = {}
    for group, indices in groups.items():
        values = [grads[i].detach().float() for i in indices if grads[i] is not None]
        if not values:
            result[group] = (0.0, None)
        else:
            norm2 = sum(torch.sum(value * value) for value in values)
            result[group] = (float(torch.sqrt(norm2).detach().cpu()), values)
    return result


def grad_cos(a: list[torch.Tensor] | None, b: list[torch.Tensor] | None) -> float:
    if not a or not b:
        return float("nan")
    dot = sum(torch.sum(x * y) for x, y in zip(a, b))
    na = torch.sqrt(sum(torch.sum(x * x) for x in a)); nb = torch.sqrt(sum(torch.sum(y * y) for y in b))
    return float((dot / (na * nb + 1e-12)).detach().cpu())


def grad_novelty(reference: list[torch.Tensor] | None, target: list[torch.Tensor] | None) -> float:
    if not reference or not target:
        return float("nan")
    dot = sum(torch.sum(x * y) for x, y in zip(target, reference))
    nr2 = sum(torch.sum(x * x) for x in reference); nt2 = sum(torch.sum(y * y) for y in target)
    orth2 = torch.clamp(nt2 - dot * dot / (nr2 + 1e-12), min=0.0)
    return float((torch.sqrt(orth2) / (torch.sqrt(nt2) + 1e-12)).detach().cpu())


def relation_tensor(feat: list[torch.Tensor], selected: list[tuple[int, int, tuple[int, int], tuple[int, int, int]]]) -> torch.Tensor:
    z = F.normalize(feat[0].float(), dim=-1)
    if not selected:
        return z.sum() * 0.0
    return torch.stack([1.0 - torch.sum(z[i] * z[j]) for i, j, _key, _state in selected]).mean()


def run_gradient_diagnostic(dataset: Any, cfg: Any, device: torch.device, bank: dict[str, Any]) -> list[dict[str, Any]]:
    model = load_raw_model(ARM_MODELS["I0"], cfg, device)
    model.train()
    loss_fn, _center = base.make_loss(cfg, num_classes=base.NUM_CLASSES)
    transform_train, _ = base.build_img_transforms(cfg)
    index_map = base.build_index_map(dataset)
    loader, _meta = base.make_epoch_loader(dataset, transform_train, b6.full_active_state(dataset), index_map, epoch=60, rank=0)
    path_to_index = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    block_ids = sorted({int(name.split(".")[1]) for name, _ in named if name.startswith("blocks.") and name.split(".")[1].isdigit()})
    late_ids = set(block_ids[-3:])
    groups: dict[str, list[int]] = {"backbone": [], "late_eva": [], "id_head": [], "bio_projection": [], "all": list(range(len(named)))}
    for index, (name, _parameter) in enumerate(named):
        group = group_name(name)
        if group:
            groups[group].append(index)
        if late_group(name, late_ids):
            groups["late_eva"].append(index)
    rows: list[dict[str, Any]] = []
    iterator = iter(loader)
    probes = 0
    while probes < 4:
        batch = next(iterator)
        entries = [{"pid": int(batch["pid"][i]), "clothes_id": int(batch["clothes_id"][i]), "image_path": str(batch["image_path"][i])} for i in range(len(batch["pid"]))]
        # Build all three relation gradients on the same batch.  The selector
        # returns only the requested arm's pairs, so calling it once for I1
        # would leave the same-clothes and ID-only diagnostics empty.
        chosen_cross, count, info = p7i.selected_relation_pairs(entries, path_to_index, bank, "I1", 60, 0, probes + 1)
        chosen_same, count_same, _ = p7i.selected_relation_pairs(entries, path_to_index, bank, "I2", 60, 0, probes + 1)
        chosen_id, count_id, _ = p7i.selected_relation_pairs(entries, path_to_index, bank, "I3", 60, 0, probes + 1)
        if count <= 0:
            continue
        if not (count == count_same == count_id):
            raise RuntimeError("matched relation counts diverged in gradient probe")
        images = batch["image"].to(device); pids = batch["pid"].to(device); camids = batch["camid"].to(device); clothes = batch["clothes_id"].to(device); captions = batch["caption_feature"].to(device)
        writer = base.MetricCapture()
        with amp.autocast(enabled=True):
            score, feat = model(images, cam_label=camids)
        base_total = loss_fn(score, feat, pids, camids, captions, clothes_ids=clothes, train_writer=writer, step=probes + 1)
        components = getattr(loss_fn, "last_components", {})
        losses = {"base": base_total, "tri": components["triplet"], "cross": relation_tensor(feat, chosen_cross["cross"]), "same": relation_tensor(feat, chosen_same["same"]), "idonly": relation_tensor(feat, chosen_id["id"])}
        gradients = {key: torch.autograd.grad(value, [parameter for _name, parameter in named], retain_graph=True, allow_unused=True) for key, value in losses.items()}
        for group, indices in groups.items():
            def gather(key):
                # Keep the exact parameter-index alignment across losses.  A
                # separate filtering of None gradients would make zip() pair
                # gradients from different parameters/shapes.
                return [
                    torch.zeros_like(named[index][1], dtype=torch.float32)
                    if gradients[key][index] is None
                    else gradients[key][index].detach().float()
                    for index in indices
                ]
            row = {"probe": probes + 1, "relation_pairs": count, "group": group}
            for key in ("base", "tri", "cross", "same", "idonly"):
                values = gather(key)
                row[f"norm_{key}"] = float(torch.sqrt(sum(torch.sum(value * value) for value in values)).detach().cpu()) if values else 0.0
            for key in ("cross", "same", "idonly"):
                row[f"cos_base_{key}"] = grad_cos(gather("base"), gather(key))
                row[f"cos_tri_{key}"] = grad_cos(gather("tri"), gather(key))
                row[f"novelty_base_{key}"] = grad_novelty(gather("base"), gather(key))
                row[f"novelty_tri_{key}"] = grad_novelty(gather("tri"), gather(key))
            row["cos_cross_same"] = grad_cos(gather("cross"), gather("same"))
            row["cos_cross_idonly"] = grad_cos(gather("cross"), gather("idonly"))
            rows.append(row)
        del gradients, losses, feat, score, base_total
        probes += 1
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    write_csv(OUT_ROOT / "gradient" / "gradient_specificity.csv", rows)
    summary = []
    for group in sorted(set(row["group"] for row in rows)):
        selected = [row for row in rows if row["group"] == group]
        keys = [key for key in selected[0] if key not in ("probe", "relation_pairs", "group")]
        summary.append({"group": group, **{key: float(np.nanmean([float(row[key]) for row in selected])) for key in keys}})
    write_csv(OUT_ROOT / "tables" / "gradient_specificity_summary.csv", summary)
    return rows


def amp_summary(runs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("I0", "I1", "I2", "I3"):
        row = {"arm": arm, "amp_skips": runs[arm]["amp_skips"], "found_inf": runs[arm]["amp_found_inf"], "nonfinite_events": runs[arm]["amp_nonfinite"]}
        path = (p7i.REFERENCE_DIRS["I0"] if arm == "I0" else p7i.TRAIN_DIRS[arm]) / "amp_statistics.csv"
        values = read_csv(path)
        scales = [float(item["amp_scale_end"]) for item in values]
        row.update({"scale_min": min(float(item["amp_scale_min"]) for item in values), "scale_max": max(float(item["amp_scale_max"]) for item in values), "scale_final": scales[-1]})
        rows.append(row)
    write_csv(OUT_ROOT / "amp" / "amp_summary.csv", rows)
    return rows


def build_report(source: dict[str, Any], runs: dict[str, Any], deltas: list[dict[str, Any]], geometry: dict[str, Any], gradient_rows: list[dict[str, Any]], amp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    late = {arm: runs[arm]["late"] for arm in ("I0", "I1", "I2", "I3")}
    geom = {arm: geometry[arm] for arm in geometry}
    dmap = {row["comparison"]: row for row in deltas}
    grad_summary = {}
    for group in sorted(set(row["group"] for row in gradient_rows)):
        rs = [row for row in gradient_rows if row["group"] == group]
        grad_summary[group] = {key: float(np.nanmean([float(row[key]) for row in rs])) for key in rs[0] if key not in ("probe", "relation_pairs", "group")}
    i1_better_i2 = dmap["I1-I2"]["delta_auc_rank1"] > 0 and dmap["I1-I2"]["delta_auc_mAP"] > 0
    i1_better_i3 = dmap["I1-I3"]["delta_auc_rank1"] > 0 and dmap["I1-I3"]["delta_auc_mAP"] > 0
    i1_reproduces = dmap["I1-I0"]["delta_auc_rank1"] > 0 and dmap["I1-I0"]["delta_auc_mAP"] > 0
    i2_over_bio = dmap["I2-I0"]["delta_auc_rank1"] > 0 and dmap["I2-I0"]["delta_auc_mAP"] > 0
    i3_over_bio = dmap["I3-I0"]["delta_auc_rank1"] > 0 and dmap["I3-I0"]["delta_auc_mAP"] > 0
    specificity = {
        arm: (geom[arm]["cross_clothes_similarity"] - geom["I0"]["cross_clothes_similarity"])
        - (geom[arm]["same_clothes_similarity"] - geom["I0"]["same_clothes_similarity"])
        for arm in ("I1", "I2", "I3")
    }
    # A positive index is required: a merely larger cross-clothes gain is not
    # specific if same-ID same-clothes compactness rises by the same amount.
    geometry_specific = specificity["I1"] > 0.0 and specificity["I1"] > max(specificity["I2"], specificity["I3"])
    grad_distinct = not all(abs(grad_summary["late_eva"].get(key, 0.0)) > 0.98 for key in ("cos_cross_same", "cos_cross_idonly"))
    if i1_better_i2 and i1_better_i3 and geometry_specific and grad_distinct:
        q8 = "YES"
        mechanism = "CROSS-CLOTHES SPECIFIC"
    elif i1_reproduces and (i2_over_bio or i3_over_bio):
        q8 = "NO"
        mechanism = "GENERIC POSITIVE CONSISTENCY"
    else:
        q8 = "MIXED"
        mechanism = "UNRESOLVED"
    q = {
        "Q1_I1_reproduces_frozen_benefit": "YES" if i1_reproduces else "NO",
        "Q2_I2_same_clothes_over_bio": "YES" if i2_over_bio else "NO/MIXED",
        "Q3_I3_id_only_over_bio": "YES" if i3_over_bio else "NO/MIXED",
        "Q4_I1_better_than_I2": "YES" if i1_better_i2 else "NO/MIXED",
        "Q5_I1_better_than_I3": "YES" if i1_better_i3 else "NO/MIXED",
        "Q6_I1_cross_clothes_specific_geometry": "YES" if geometry_specific else "NO/MIXED",
        "Q7_gradients_meaningfully_distinct": "YES" if grad_distinct else "NO/MIXED",
        "Q8_original_gain_specific": q8,
    }
    result = {"questions": q, "mechanism": mechanism, "late": late, "deltas": deltas, "geometry": geometry, "gradient_summary": grad_summary, "amp": amp_rows}
    write_json(OUT_ROOT / "PHASE7I_ANALYSIS.json", result)
    specificity_rows = []
    for arm in ("I1", "I2", "I3"):
        cross_gain = geom[arm]["cross_clothes_similarity"] - geom["I0"]["cross_clothes_similarity"]
        same_gain = geom[arm]["same_clothes_similarity"] - geom["I0"]["same_clothes_similarity"]
        specificity_rows.append({
            "arm": arm,
            "cross_clothes_gain": cross_gain,
            "same_clothes_gain": same_gain,
            "specificity_index": cross_gain - same_gain,
        })
    write_csv(OUT_ROOT / "specificity" / "specificity_index.csv", specificity_rows)
    write_csv(OUT_ROOT / "training" / "arm_completeness.csv", [
        {
            "arm": arm,
            "metrics": str(runs[arm]["metrics"]),
            "metrics_rows": runs[arm]["rows"],
            "final_checkpoint": str(ARM_MODELS[arm]),
            "final_checkpoint_sha256": p7i.sha256_file(ARM_MODELS[arm]),
            "amp_skips": runs[arm]["amp_skips"],
            "amp_found_inf": runs[arm]["amp_found_inf"],
        }
        for arm in ("I0", "I1", "I2", "I3")
    ])
    lines = [
        "# Phase 7I — Cross-Clothes Relation Specificity Test",
        "",
        "## Scope and protocol",
        "",
        "LTCC only; seed 1234; four conceptual arms. I0 is the frozen Phase 6C B0 Bio control. The original Phase 6C B1 is preserved as a frozen historical reference. Because its relation exposure was not matched to the same-clothes and ID-only selectors, the primary I1 below is a newly trained matched-budget cross-clothes arm.",
        "",
        f"- Source audit: `{p7i.SOURCE_REPORT_PATH}`.",
        f"- Output root: `{OUT_ROOT}`.",
        "- No PRCC run, second seed, new sampler, new optimizer, margin term, SR3, H_ID, or later phase was started.",
        "",
        "## Frozen source and pair-budget audit",
        "",
        f"- Frozen Phase 6C B0: `{p7i.REFERENCE_DIRS['I0']}`.",
        f"- Frozen Phase 6C B1 reference: `{p7i.REFERENCE_DIRS['I1_frozen_reference']}`.",
        f"- Matched source candidate bank: `{p7i.PHASE6B_ROOT / 'artifacts/relation_source_rows.csv'}`.",
        f"- Frozen B1 candidate exposure exactly replays: `{source['protocol_checks'].get('frozen_b1_candidate_bank_matches')}`.",
        f"- Frozen B1 strict matched-budget reusable: `{source['frozen_b1_reusable_for_matched_control']}`.",
        f"- Matched relation contributions: `{source['pair_budget_summary']['matched_relation_pairs_total']}` across `{source['pair_budget_summary']['global_batches']}` rank-batches; active rate `{source['pair_budget_summary']['active_batch_rate']:.4%}`.",
        "",
        "## Exact run checkpoints",
        "",
        "| Arm | Role | Final checkpoint | SHA256 |",
        "|---|---|---|---|",
        f"| I0 | reused frozen Bio | `{ARM_MODELS['I0']}` | `{p7i.sha256_file(ARM_MODELS['I0'])}` |",
        f"| I1 | new matched-budget cross-clothes | `{ARM_MODELS['I1']}` | `{p7i.sha256_file(ARM_MODELS['I1'])}` |",
        f"| I2 | new matched-budget same-clothes | `{ARM_MODELS['I2']}` | `{p7i.sha256_file(ARM_MODELS['I2'])}` |",
        f"| I3 | new matched-budget ID-only | `{ARM_MODELS['I3']}` | `{p7i.sha256_file(ARM_MODELS['I3'])}` |",
        "",
        "## E51–E60 primary comparison",
        "",
        "| Arm | Mean Rank-1 | Mean mAP | AUC Rank-1 | AUC mAP | Slope Rank-1 | Slope mAP | E60 Rank-1/mAP | Best Rank-1 (E) | Best mAP (E) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("I0", "I1", "I2", "I3"):
        t = late[arm]; b = runs[arm]["best"]
        lines.append(f"| {arm} | {pct(t['mean_rank1'])} | {pct(t['mean_mAP'])} | {pct(t['auc_rank1'])} | {pct(t['auc_mAP'])} | {t['slope_rank1']*100:+.4f} pp | {t['slope_mAP']*100:+.4f} pp | {pct(t['e60_rank1'])} / {pct(t['e60_mAP'])} | {pct(b['best_rank1'])} (E{b['best_rank1_epoch']}) | {pct(b['best_mAP'])} (E{b['best_mAP_epoch']}) |")
    lines += ["", "### Primary deltas", "", "| Comparison | Δ AUC Rank-1 | Δ AUC mAP | Δ E60 Rank-1 | Δ E60 mAP |", "|---|---:|---:|---:|---:|"]
    for row in deltas:
        if row["comparison"] in ("I1-I0", "I1-I2", "I1-I3"):
            lines.append(f"| {row['comparison']} | {row['delta_auc_rank1']*100:+.4f} pp | {row['delta_auc_mAP']*100:+.4f} pp | {row['delta_e60_rank1']*100:+.4f} pp | {row['delta_e60_mAP']*100:+.4f} pp |")
    lines += ["", "## Endpoint train-only geometry", "", "| Arm | Cross-clothes sim | Same-clothes sim | Nearest-negative sim | Cross margin | Same margin | Cross ordering | State dispersion | PID separation |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("I0", "I1", "I2", "I3"):
        g = geom[arm]
        lines.append(f"| {arm} | {g['cross_clothes_similarity']:.6f} | {g['same_clothes_similarity']:.6f} | {g['nearest_negative_similarity']:.6f} | {g['cross_clothes_margin']:.6f} | {g['same_clothes_margin']:.6f} | {g['cross_clothes_ordering']:.4%} | {g['clothing_state_center_dispersion']:.6f} | {g['pid_center_separation']:.6f} |")
    lines += ["", "### Specificity index", "", "`Specificity = Δ cross-clothes positive similarity − Δ same-clothes positive similarity`, relative to I0.", "", "| Arm | CrossClothesGain | SameClothesGain | Specificity |", "|---|---:|---:|---:|"]
    for arm in ("I1", "I2", "I3"):
        cg = geom[arm]["cross_clothes_similarity"] - geom["I0"]["cross_clothes_similarity"]
        sg = geom[arm]["same_clothes_similarity"] - geom["I0"]["same_clothes_similarity"]
        lines.append(f"| {arm} | {cg:+.6f} | {sg:+.6f} | {cg-sg:+.6f} |")
    lines += ["", "## Gradient specificity", "", "No-step FP32 gradients were measured on four matched train batches using the frozen I0 endpoint. The relation terms use the same matched per-batch count; no optimizer state was changed.", "", "| Group | cos(base,cross) | cos(base,same) | cos(base,id-only) | cos(triplet,cross) | cos(triplet,same) | cos(triplet,id-only) | cos(cross,same) | cos(cross,id-only) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("late_eva", "id_head", "bio_projection", "backbone", "all"):
        if group not in grad_summary: continue
        g = grad_summary[group]
        lines.append(f"| {group} | {g.get('cos_base_cross', float('nan')):.4f} | {g.get('cos_base_same', float('nan')):.4f} | {g.get('cos_base_idonly', float('nan')):.4f} | {g.get('cos_tri_cross', float('nan')):.4f} | {g.get('cos_tri_same', float('nan')):.4f} | {g.get('cos_tri_idonly', float('nan')):.4f} | {g.get('cos_cross_same', float('nan')):.4f} | {g.get('cos_cross_idonly', float('nan')):.4f} |")
    lines += ["", "## AMP audit", "", "| Arm | Skips | found_inf | nonfinite | Scale min | Scale max | Final scale |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in amp_rows:
        lines.append(f"| {row['arm']} | {row['amp_skips']} | {row['found_inf']} | {row['nonfinite_events']} | {row['scale_min']:.1f} | {row['scale_max']:.1f} | {row['scale_final']:.1f} |")
    lines += ["", "## Q1–Q8", ""]
    for key, value in q.items():
        lines.append(f"- **{key}**: **{value}**")
    lines += ["", "## Final mechanism conclusion", "", f"**{mechanism}**", "", "The conclusion is based jointly on late-window trajectory, matched relation exposure, endpoint train-only geometry, and no-step gradient comparison. The frozen original B1 remains reported as a reference and is not treated as the matched I1 when the exposure audit fails.", "", "## Stop condition", "", "No second seed was started. No PRCC run was started. No later Phase 7 phase was started. Phase 7I stops here."]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    cfg = p7i.configure()
    p7i.ensure_dirs()
    source = json.loads((OUT_ROOT / "source_audit" / "source_audit.json").read_text())
    if any(not path.is_file() for path in ARM_METRICS.values() if path != ARM_METRICS["I1_frozen_reference"]):
        raise RuntimeError("I1/I2/I3 training metrics are incomplete")
    runs, deltas = analyze_trajectories()
    if any(not path.is_file() for path in ARM_MODELS.values()):
        raise RuntimeError("I0/I1/I2/I3 final checkpoints are incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7I analysis requires PPU; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dataset = base.build_dataset(cfg)
    bank = b6.parse_relation_bank(dataset, cfg)
    bank["path_to_index"] = {str(item["image_path"]): index for index, item in enumerate(dataset.train)}
    geometry = run_geometry(dataset, cfg, device)
    gradient_rows = run_gradient_diagnostic(dataset, cfg, device, bank)
    amp_rows = amp_summary(runs)
    result = build_report(source, runs, deltas, geometry, gradient_rows, amp_rows)
    manifest = {
        "phase": "PHASE 7I", "created_utc": p7i.utc_now(), "seed": p7i.SEED, "dataset": "LTCC",
        "new_training_arms": ["I1", "I2", "I3"], "reused_reference": {"I0": str(p7i.REFERENCE_DIRS["I0"]), "frozen_I1": str(p7i.REFERENCE_DIRS["I1_frozen_reference"])},
        "training_outputs": {arm: str(path) for arm, path in p7i.TRAIN_DIRS.items()},
        "source_audit": str(OUT_ROOT / "source_audit/source_audit.json"),
        "analysis": str(OUT_ROOT / "PHASE7I_ANALYSIS.json"), "report": str(REPORT_PATH),
        "lambda_rel": p7i.LAMBDA_REL, "feature": "normalized feat[0]", "pair_loss": "1-cosine",
        "no_second_seed": True, "no_prcc": True, "no_later_phase": True,
        "questions": result["questions"], "mechanism": result["mechanism"],
    }
    write_json(OUT_ROOT / "PHASE7I_MANIFEST.json", manifest)
    print(json.dumps({"report": str(REPORT_PATH), "manifest": str(OUT_ROOT / "PHASE7I_MANIFEST.json"), "mechanism": result["mechanism"], "questions": result["questions"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
