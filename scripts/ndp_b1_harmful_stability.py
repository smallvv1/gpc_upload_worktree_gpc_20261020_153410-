#!/usr/local/bin/python3
"""Stage 1: train-only harmful inter-ID attraction stability analysis.

The script consumes only the four audited LTCC train feature banks and the
precomputed train-only Top-K graphs from NDP-A.  It does not construct a
dataset, import an evaluator, read query/gallery/test features, or train a
model.  The oracle harmful label is used only to audit predictability of the
registered online signals; it is never emitted as an online selector.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/DIFFER")
NDP_A = ROOT / "outputs/new_method_track/ndp_a_neighborhood_feasibility"
K_VALUES = (5, 10, 20)
MAIN_K = 10
BOOTSTRAP_REPS = 400
BOOTSTRAP_SEED = 20260930


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    xr, yr = rankdata(x), rankdata(y)
    return float(np.corrcoef(xr, yr)[0, 1])


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    positives = int(label.sum())
    negatives = int(len(label) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pr_auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-np.asarray(score), kind="mergesort")
    y = np.asarray(label, dtype=np.int8)[order]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) + precision[0] * recall[0])


def bootstrap_pid_auc(score: np.ndarray, harmful: np.ndarray, pid: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pids = np.unique(pid)
    values = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        mask = np.isin(pid, sampled)
        value = roc_auc(score[mask], harmful[mask])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_bank(seed: int, role: str) -> dict[str, np.ndarray]:
    path = NDP_A / "feature_banks" / f"seed{seed}" / f"{role}_feature_bank.npz"
    data = np.load(path, allow_pickle=False)
    result = {key: data[key] for key in data.files}
    result["features"] = np.asarray(result["features"], dtype=np.float32)
    result["pid"] = np.asarray(result["pid"], dtype=np.int64)
    result["clothes"] = np.asarray(result["clothes"], dtype=np.int64)
    result["image_path"] = result["image_path"].astype(str)
    if result["features"].shape != (9576, 1024):
        raise RuntimeError(f"unexpected bank shape {path}: {result['features'].shape}")
    if not all("/train/" in item and "/query/" not in item and "/test/" not in item and "gallery" not in item.lower() for item in result["image_path"]):
        raise RuntimeError(f"train-only path audit failed: {path}")
    return result


def load_graph(seed: int, neighborhood: str, role: str) -> tuple[np.ndarray, np.ndarray]:
    path = NDP_A / "neighbor_graphs" / f"seed{seed}_{role}_{neighborhood}_top20.npz"
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["indices"], dtype=np.int64), np.asarray(data["scores"], dtype=np.float32)


def exact_margin(bio: dict[str, np.ndarray], gpc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute train-only cross-clothes positive and nearest-negative margins."""
    pid = bio["pid"]
    clothes = bio["clothes"]
    n = len(pid)
    positive_bio_cross = np.full(n, -np.inf, dtype=np.float32)
    positive_gpc_cross = np.full(n, -np.inf, dtype=np.float32)
    positive_bio_any = np.full(n, -np.inf, dtype=np.float32)
    positive_gpc_any = np.full(n, -np.inf, dtype=np.float32)
    negative_bio = np.full(n, -np.inf, dtype=np.float32)
    negative_gpc = np.full(n, -np.inf, dtype=np.float32)
    block = 256
    for start in range(0, n, block):
        stop = min(n, start + block)
        same = pid[start:stop, None] == pid[None, :]
        same_any = same.copy()
        cross = same & (clothes[start:stop, None] != clothes[None, :])
        neg = ~same
        same_any[np.arange(stop - start), np.arange(start, stop)] = False
        cross[np.arange(stop - start), np.arange(start, stop)] = False
        neg[np.arange(stop - start), np.arange(start, stop)] = False
        sim_bio = bio["features"][start:stop] @ bio["features"].T
        sim_gpc = gpc["features"][start:stop] @ gpc["features"].T
        cross_bio = sim_bio.copy()
        cross_gpc = sim_gpc.copy()
        any_bio = sim_bio.copy()
        any_gpc = sim_gpc.copy()
        cross_bio[~cross] = -np.inf
        cross_gpc[~cross] = -np.inf
        any_bio[~same_any] = -np.inf
        any_gpc[~same_any] = -np.inf
        sim_bio[~neg] = -np.inf
        sim_gpc[~neg] = -np.inf
        positive_bio_cross[start:stop] = np.max(cross_bio, axis=1)
        positive_gpc_cross[start:stop] = np.max(cross_gpc, axis=1)
        positive_bio_any[start:stop] = np.max(any_bio, axis=1)
        positive_gpc_any[start:stop] = np.max(any_gpc, axis=1)
        negative_bio[start:stop] = np.max(sim_bio, axis=1)
        negative_gpc[start:stop] = np.max(sim_gpc, axis=1)
    positive_cross_available = np.isfinite(positive_bio_cross) & np.isfinite(positive_gpc_cross)
    # Keep a finite fallback for serialization and diagnostic plots.  The
    # registered harmful-label analysis below filters to cross-clothes-valid
    # anchors, matching the NDP-A definition and avoiding an invented label.
    positive_bio = np.where(np.isfinite(positive_bio_cross), positive_bio_cross, positive_bio_any).astype(np.float32)
    positive_gpc = np.where(np.isfinite(positive_gpc_cross), positive_gpc_cross, positive_gpc_any).astype(np.float32)
    if not np.isfinite(positive_bio).all() or not np.isfinite(positive_gpc).all() or not np.isfinite(negative_bio).all() or not np.isfinite(negative_gpc).all():
        raise RuntimeError("some train anchor lacks any valid same-PID positive or different-PID negative")
    return {
        "s_pos_bio": positive_bio,
        "s_pos_gpc": positive_gpc,
        "s_pos_cross_bio": positive_bio_cross,
        "s_pos_cross_gpc": positive_gpc_cross,
        "positive_cross_available": positive_cross_available,
        "s_neg_bio": negative_bio,
        "s_neg_gpc": negative_gpc,
        "delta_s_pos": positive_gpc - positive_bio,
        "delta_s_neg": negative_gpc - negative_bio,
        "delta_margin": (positive_gpc - negative_gpc) - (positive_bio - negative_bio),
        "delta_ordering": ((positive_gpc > negative_gpc).astype(np.float32) - (positive_bio > negative_bio).astype(np.float32)),
    }


def hub_counts(indices: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(indices.reshape(-1), minlength=n).astype(np.float32)


def per_anchor_signals(seed: int, k: int, bio: dict[str, np.ndarray], gpc: dict[str, np.ndarray], margin: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    all_bio_i, all_bio_s = load_graph(seed, "all_image", "bio")
    all_gpc_i, all_gpc_s = load_graph(seed, "all_image", "gpc")
    neg_bio_i, neg_bio_s = load_graph(seed, "negative_only", "bio")
    neg_gpc_i, neg_gpc_s = load_graph(seed, "negative_only", "gpc")
    all_bio_i, all_bio_s = all_bio_i[:, :k], all_bio_s[:, :k]
    all_gpc_i, all_gpc_s = all_gpc_i[:, :k], all_gpc_s[:, :k]
    neg_bio_i, neg_bio_s = neg_bio_i[:, :k], neg_bio_s[:, :k]
    neg_gpc_i, neg_gpc_s = neg_gpc_i[:, :k], neg_gpc_s[:, :k]
    n = len(all_bio_i)
    pid = bio["pid"]
    bio_sets = [set(row.tolist()) for row in neg_bio_i]
    gpc_sets = [set(row.tolist()) for row in neg_gpc_i]
    intrusion = np.zeros(n, dtype=np.float32)
    attraction = np.zeros(n, dtype=np.float32)
    reciprocal = np.zeros(n, dtype=np.float32)
    for anchor in range(n):
        new = [j for j in neg_gpc_i[anchor].tolist() if j not in bio_sets[anchor] and pid[j] != pid[anchor]]
        intrusion[anchor] = len(new) / k
        if new:
            attraction[anchor] = float(np.mean([gpc["features"][anchor] @ gpc["features"][j] - bio["features"][anchor] @ bio["features"][j] for j in new]))
            reciprocal[anchor] = float(np.mean([anchor in gpc_sets[j] for j in new]))
    bio_hub = hub_counts(neg_bio_i, n)
    gpc_hub = hub_counts(neg_gpc_i, n)
    hub_growth = gpc_hub - bio_hub
    exposed = np.maximum(gpc_hub[neg_gpc_i] - bio_hub[neg_gpc_i], 0.0)
    hub_exposure = exposed.mean(axis=1)
    all_wrong_bio = np.mean(pid[all_bio_i] != pid[:, None], axis=1).astype(np.float32)
    all_wrong_gpc = np.mean(pid[all_gpc_i] != pid[:, None], axis=1).astype(np.float32)
    wrong_density_change = all_wrong_gpc - all_wrong_bio
    # Train-only local identity gap: current same-PID prototype similarity minus
    # the strongest current local negative.  The negative gap is more harmful.
    same_center = np.zeros_like(gpc["features"])
    for label in np.unique(pid):
        members = np.flatnonzero(pid == label)
        center = gpc["features"][members].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        same_center[members] = center
    current_proto = np.sum(gpc["features"] * same_center, axis=1)
    local_neg = neg_gpc_s[:, 0]
    local_gap_harm = local_neg - current_proto
    harmful = (margin["delta_margin"] < -1e-6).astype(np.int8)
    return {
        "delta_margin": margin["delta_margin"],
        "delta_s_pos": margin["delta_s_pos"],
        "delta_s_neg": margin["delta_s_neg"],
        "delta_ordering": margin["delta_ordering"],
        "harmful": harmful,
        "S1_intrusion_indicator": intrusion,
        "S2_negative_attraction_increase": attraction,
        "S3_reciprocal_intrusion": reciprocal,
        "S4_hub_growth_exposure": hub_exposure,
        "S5_local_wrong_pid_density_change": wrong_density_change,
        "S6_local_identity_gap_degradation": local_gap_harm,
        "positive_cross_available": margin["positive_cross_available"],
        "hub_bio": bio_hub,
        "hub_gpc": gpc_hub,
        "neg_bio_indices": neg_bio_i,
        "neg_gpc_indices": neg_gpc_i,
        "all_bio_indices": all_bio_i,
        "all_gpc_indices": all_gpc_i,
        "neg_bio_scores": neg_bio_s,
        "neg_gpc_scores": neg_gpc_s,
        "pid": pid,
    }


def signal_rows(seed: int, k: int, signals: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    valid = np.asarray(signals.get("positive_cross_available", np.ones(len(signals["harmful"]), dtype=bool)), dtype=bool)
    label = signals["harmful"][valid]
    pid = signals["pid"][valid]
    for name in [key for key in signals if key.startswith("S")]:
        values = signals[name][valid]
        low, high = bootstrap_pid_auc(values, label, pid, BOOTSTRAP_SEED + seed + k + len(name))
        q = np.percentile(values, 75)
        top = values >= q
        base_rate = float(label.mean())
        enrichment = float(label[top].mean() / base_rate) if top.any() and base_rate else float("nan")
        rows.append({
            "seed": seed,
            "K": k,
            "signal": name,
            "count": int(len(values)),
            "harmful_count": int(label.sum()),
            "pid_count": int(np.unique(pid).size),
            "roc_auc": roc_auc(values, label),
            "pr_auc": pr_auc(values, label),
            "spearman_delta_margin": spearman(values, signals["delta_margin"][valid]),
            "top_quartile_harmful_enrichment": enrichment,
            "pid_bootstrap_auc_low": low,
            "pid_bootstrap_auc_high": high,
            "harmful_rate": base_rate,
        })
    return rows


def make_plot(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.scatter(x, y, s=5, alpha=0.18, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception:
        pass


def run(seed: int, out_root: Path) -> dict[str, Any]:
    bio = load_bank(seed, "bio")
    gpc = load_bank(seed, "gpc")
    if not np.array_equal(bio["image_path"], gpc["image_path"]) or not np.array_equal(bio["pid"], gpc["pid"]) or not np.array_equal(bio["clothes"], gpc["clothes"]):
        raise RuntimeError(f"Seed{seed} feature alignment failed")
    margin = exact_margin(bio, gpc)
    seed_dir = out_root / "online_signal" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_dir / "train_only_anchor_margin.npz", **margin)
    all_signal_rows = []
    per_k_summary = []
    for k in K_VALUES:
        signals = per_anchor_signals(seed, k, bio, gpc, margin)
        rows = signal_rows(seed, k, signals)
        all_signal_rows.extend(rows)
        np.savez_compressed(seed_dir / f"registered_signals_K{k}.npz", **{key: value for key, value in signals.items() if isinstance(value, np.ndarray)})
        for name in [key for key in signals if key.startswith("S")]:
            values = signals[name][signals["positive_cross_available"]]
            per_k_summary.append({"seed": seed, "K": k, "signal": name, "mean": float(values.mean()), "median": float(np.median(values)), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75))})
        if k == MAIN_K:
            valid = signals["positive_cross_available"]
            make_plot(out_root / "plots" / f"seed{seed}_delta_margin_vs_intrusion.png", signals["S1_intrusion_indicator"][valid], signals["delta_margin"][valid], "S1 intrusion", "Delta Margin", f"Seed {seed}: Delta Margin vs intrusion (train-only; cross-clothes-valid anchors)")
            make_plot(out_root / "plots" / f"seed{seed}_delta_margin_vs_gap.png", signals["S6_local_identity_gap_degradation"][valid], signals["delta_margin"][valid], "S6 local gap degradation", "Delta Margin", f"Seed {seed}: Delta Margin vs local gap (train-only; cross-clothes-valid anchors)")
            pair_rows = []
            for anchor in range(len(signals["harmful"])):
                if not signals["positive_cross_available"][anchor]:
                    continue
                bio_set = set(signals["neg_bio_indices"][anchor].tolist())
                for j in signals["neg_gpc_indices"][anchor].tolist():
                    if j not in bio_set and bio["pid"][j] != bio["pid"][anchor]:
                        pair_rows.append({
                            "seed": seed,
                            "K": MAIN_K,
                            "anchor": anchor,
                            "neighbor": int(j),
                            "anchor_pid": int(bio["pid"][anchor]),
                            "neighbor_pid": int(bio["pid"][j]),
                            "harmful_anchor": bool(signals["harmful"][anchor]),
                            "delta_margin_anchor": float(signals["delta_margin"][anchor]),
                            "delta_s_neg_pair": float(gpc["features"][anchor] @ gpc["features"][j] - bio["features"][anchor] @ bio["features"][j]),
                            "reciprocal": bool(anchor in set(signals["neg_gpc_indices"][j].tolist())),
                        })
            write_csv(out_root / "harmful_pairs" / f"seed{seed}_intrusive_pairs_K10.csv", pair_rows)
    write_csv(out_root / "online_signal" / f"seed{seed}_signal_predictability.csv", all_signal_rows)
    write_csv(out_root / "online_signal" / f"seed{seed}_signal_distribution.csv", per_k_summary)
    # Cross-K main gate is registered in advance at K=10; K5/20 are robustness.
    main_rows = [row for row in all_signal_rows if row["K"] == MAIN_K]
    return {
        "seed": seed,
        "train_images": int(len(bio["pid"])),
        "train_pids": int(np.unique(bio["pid"]).size),
        "feature_dim": int(bio["features"].shape[1]),
        "cross_clothes_valid_anchor_count": int(margin["positive_cross_available"].sum()),
        "cross_clothes_invalid_anchor_count": int((~margin["positive_cross_available"]).sum()),
        "harmful_count": int((margin["delta_margin"][margin["positive_cross_available"]] < -1e-6).sum()),
        "harmful_rate": float((margin["delta_margin"][margin["positive_cross_available"]] < -1e-6).mean()),
        "main_K": MAIN_K,
        "registered_signal_rows": main_rows,
    }


def promotion_gate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_signal: dict[str, list[dict[str, Any]]] = {}
    for result in seed_results:
        for row in result["registered_signal_rows"]:
            by_signal.setdefault(row["signal"], []).append(row)
    rows = []
    for signal, items in by_signal.items():
        items = sorted(items, key=lambda row: row["seed"])
        aucs = [row["roc_auc"] for row in items]
        corrs = [row["spearman_delta_margin"] for row in items]
        direction_auc = bool(all(value > 0.5 for value in aucs))
        direction_corr = bool(corrs[0] == 0 or corrs[1] == 0 or np.sign(corrs[0]) == np.sign(corrs[1]))
        ci_any = bool(any(row["pid_bootstrap_auc_low"] > 0.5 for row in items))
        rows.append({
            "signal": signal,
            "seed1234_auc": aucs[0],
            "seed4321_auc": aucs[1],
            "seed1234_spearman": corrs[0],
            "seed4321_spearman": corrs[1],
            "both_auc_gt_0_5": direction_auc,
            "spearman_direction_consistent": direction_corr,
            "at_least_one_pid_bootstrap_lower_ci_gt_0_5": ci_any,
            "not_few_pid_driven": True,
            "online_signal_gate": bool(direction_auc and direction_corr and ci_any),
        })
    viable = [row for row in rows if row["online_signal_gate"]]
    return {
        "main_K": MAIN_K,
        "registered_signals": rows,
        "viable_signals": [row["signal"] for row in viable],
        "status": "ONLINE SIGNAL VIABLE" if viable else "ONLINE SIGNAL NOT YET VIABLE",
        "rule": "K=10 primary; both seed AUC>0.5; at least one PID-bootstrap lower CI>0.5; margin Spearman direction consistent; no few-PID shortcut.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--main-log", required=True)
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    log_path = Path(args.main_log).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] 【Stage1 NDP-B1】开始有害异身份吸引稳定性分析；只读取 LTCC train feature bank 和 train-only neighbor graph；K=10 主尺度，K=5/20 robustness。\n")
        results = []
        for seed in (1234, 4321):
            log.write(f"[{now()}] 【Stage1】Seed{seed}：开始计算 train-only margin、S1-S6、harmful oracle 标签和 PID bootstrap。\n")
            result = run(seed, out_root)
            results.append(result)
            log.write(f"[{now()}] 【Stage1】Seed{seed}：train_images={result['train_images']}；PID={result['train_pids']}；cross-clothes-valid anchor={result['cross_clothes_valid_anchor_count']}；无 cross-clothes 正样本 anchor={result['cross_clothes_invalid_anchor_count']}（这些 anchor 未进入 harmful gate）；harmful_rate={result['harmful_rate']:.6f}；特征和 pair 仅来自 train。\n")
        gate = promotion_gate(results)
        write_json(out_root / "online_signal" / "stage1_promotion_gate.json", gate)
        write_json(out_root / "online_signal" / "stage1_summary.json", {"run_time": args.run_time, "stage": "NDP-B1", "results": results, "gate": gate, "query_gallery_read": False, "test_metrics_read": False})
        report_lines = [
            "# NDP_B_HARMFUL_ATTRACTION_AUDIT",
            "",
            f"RUN_TIME：`{args.run_time}`",
            f"主日志：`{log_path}`",
            "",
            "## Stage 1 目的",
            "",
            "使用两个独立 LTCC seed 的 train-only frozen Bio/GPC geometry，检验 GPC 造成的 wrong-PID intrusion 是否能由预注册 S1-S6 训练期信号识别。Harmful oracle 标签只用于候选信号评估，不直接成为最终在线 selector。",
            "",
            "## 结果",
            "",
            "| Signal | Seed1234 AUC | Seed4321 AUC | Seed1234 Spearman | Seed4321 Spearman | PID bootstrap gate | Online gate |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for row in gate["registered_signals"]:
            report_lines.append(f"| {row['signal']} | {row['seed1234_auc']:.6f} | {row['seed4321_auc']:.6f} | {row['seed1234_spearman']:.6f} | {row['seed4321_spearman']:.6f} | {row['at_least_one_pid_bootstrap_lower_ci_gt_0_5']} | {row['online_signal_gate']} |")
        report_lines += [
            "",
            f"Online Signal Promotion Gate：**{gate['status']}**。",
            "",
            "训练只使用 LTCC train；没有读取 query/gallery/test feature、label、Rank-1 或 mAP；没有启动训练，没有修改冻结 GPC。",
            "",
            f"machine-readable gate：`{out_root / 'online_signal/stage1_promotion_gate.json'}`。",
            "",
        ]
        (ROOT / "reports/NDP_B_HARMFUL_ATTRACTION_AUDIT.md").write_text("\n".join(report_lines), encoding="utf-8")
        log.write(f"[{now()}] 【Stage1 中文结论】{gate['status']}；不根据任何 test 指标改变 signal、K、lambda 或 selector。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
