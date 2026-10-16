"""Analyze the completed Phase 7D-B PRCC margin-preserving run.

This script is intentionally read-only with respect to model training.  It
uses the frozen Phase 7C C0 control, the completed D1 run, and the train-only
geometry written by the audited Phase 7C loop.  It writes derived CSV/JSON
tables and the final Phase 7D-B report without loading a model or touching an
optimizer.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/home/DIFFER")
OUT = ROOT / "outputs/new_method_track/phase7d_b_margin_preserving_relation"
ANALYSIS = OUT / "analysis"
TABLES = ANALYSIS / "tables"
GEOMETRY = ANALYSIS / "geometry"
EXPOSURE = ANALYSIS / "exposure"
BALANCE = ANALYSIS / "relation_balance"
REPORT = ROOT / "reports/PHASE_7D_B_MARGIN_PRESERVING_RELATION.md"
MANIFEST = OUT / "PHASE7D_B_MANIFEST.json"

C0_DIR = ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C0_bio"
C1_DIR = ROOT / "outputs/new_method_track/phase7c_prototype_relation/training/C1_image_uniform_relation"
D1_DIR = OUT / "training/D1_margin_preserving_relation"
C0_GEOMETRY = ROOT / "outputs/new_method_track/phase7c_prototype_relation/geometry/C0_geometry.csv"
D1_GEOMETRY = OUT / "analysis/geometry/C2_geometry.csv"
C1_BALANCE = ROOT / "outputs/new_method_track/phase7c_prototype_relation/relation_balance/planned_contribution_counts.csv"
D1_BALANCE = D1_DIR / "relation_balance_contributions.csv"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def epoch(row: dict[str, str]) -> int:
    return int(float(row["epoch"]))


def pct(row: dict[str, str], key: str) -> float:
    return num(row, key) * 100.0


def slope(rows: list[dict[str, str]], key: str) -> float:
    xs = [float(epoch(r)) for r in rows]
    ys = [pct(r, key) for r in rows]
    xm = statistics.mean(xs)
    ym = statistics.mean(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom if denom else 0.0


def normalized_auc(rows: list[dict[str, str]], key: str) -> float:
    """Normalized trapezoidal AUC in percentage points over the window."""
    ys = [pct(r, key) for r in rows]
    if len(ys) <= 1:
        return ys[0] if ys else float("nan")
    return sum((ys[i - 1] + ys[i]) / 2.0 for i in range(1, len(ys))) / (len(ys) - 1)


def window_rows(rows: list[dict[str, str]], lo: int, hi: int) -> list[dict[str, str]]:
    return [r for r in rows if lo <= epoch(r) <= hi]


def window_summary(arm: str, rows: list[dict[str, str]], label: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "window": label,
        "start_epoch": epoch(rows[0]),
        "end_epoch": epoch(rows[-1]),
        "mean_rank1_pp": statistics.mean(pct(r, "cc_rank1") for r in rows),
        "mean_mAP_pp": statistics.mean(pct(r, "cc_mAP") for r in rows),
        "auc_rank1_pp": normalized_auc(rows, "cc_rank1"),
        "auc_mAP_pp": normalized_auc(rows, "cc_mAP"),
        "rank1_slope_pp_per_epoch": slope(rows, "cc_rank1"),
        "mAP_slope_pp_per_epoch": slope(rows, "cc_mAP"),
    }


def metric_summary(arm: str, rows: list[dict[str, str]], window: list[dict[str, str]], window_name: str) -> dict[str, Any]:
    best_rank = max(rows, key=lambda r: num(r, "cc_rank1"))
    best_map = max(rows, key=lambda r: num(r, "cc_mAP"))
    best_window_rank = max(window, key=lambda r: num(r, "cc_rank1"))
    best_window_map = max(window, key=lambda r: num(r, "cc_mAP"))
    late = window_summary(arm, window, window_name)
    return {
        **late,
        "e60_rank1_pp": pct(rows[-1], "cc_rank1"),
        "e60_mAP_pp": pct(rows[-1], "cc_mAP"),
        "best_rank1_pp": pct(best_rank, "cc_rank1"),
        "best_rank1_epoch": epoch(best_rank),
        "best_mAP_pp": pct(best_map, "cc_mAP"),
        "best_mAP_epoch": epoch(best_map),
        "best_late_rank1_pp": pct(best_window_rank, "cc_rank1"),
        "best_late_rank1_epoch": epoch(best_window_rank),
        "best_late_mAP_pp": pct(best_window_map, "cc_mAP"),
        "best_late_mAP_epoch": epoch(best_window_map),
    }


def read_metrics(directory: Path) -> list[dict[str, str]]:
    rows = read_csv(directory / "metrics_per_epoch.csv")
    if len(rows) != 60 or [epoch(r) for r in rows] != list(range(1, 61)):
        raise RuntimeError(f"incomplete or non-contiguous metrics: {directory}")
    return rows


def make_trajectory(c0: list[dict[str, str]], d1: list[dict[str, str]], c1: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    for arm, source in (("C0", c0), ("C1_reference", c1), ("D1", d1)):
        for r in source:
            rows.append({
                "arm": arm,
                "epoch": epoch(r),
                "rank1_pp": pct(r, "cc_rank1"),
                "mAP_pp": pct(r, "cc_mAP"),
                "train_loss": num(r, "train_loss"),
                "relation_loss": num(r, "relation_loss"),
                "total_loss": num(r, "total_loss"),
                "lr": num(r, "lr"),
                "amp_scale_end": num(r, "amp_scale_end"),
                "amp_skip_flag": num(r, "amp_skip"),
                "skipped_steps": num(r, "skipped_steps"),
                "found_inf_count": num(r, "found_inf_count"),
                "updates_completed": num(r, "updates_completed"),
            })
    write_csv(TABLES / "trajectory.csv", rows)


def make_window_tables(c0: list[dict[str, str]], d1: list[dict[str, str]], c1: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    arms = {"C0": c0, "C1_reference": c1, "D1": d1}
    windows = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
    all_windows: list[dict[str, Any]] = []
    for label_lo, label_hi in windows:
        label = f"E{label_lo}-E{label_hi}"
        for arm, rows in arms.items():
            all_windows.append(window_summary(arm, window_rows(rows, label_lo, label_hi), label))
    write_csv(TABLES / "window_analysis.csv", all_windows)
    late: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    late_rows = {}
    for arm, rows in arms.items():
        late_rows[arm] = window_rows(rows, 51, 60)
        metrics[arm] = metric_summary(arm, rows, late_rows[arm], "E51-E60")
        late.append(metrics[arm])
    c0m = metrics["C0"]
    for arm in ("C1_reference", "D1"):
        am = metrics[arm]
        late.append({
            "arm": f"{arm}-C0",
            "window": "E51-E60 / delta",
            "start_epoch": 51,
            "end_epoch": 60,
            "mean_rank1_pp": am["mean_rank1_pp"] - c0m["mean_rank1_pp"],
            "mean_mAP_pp": am["mean_mAP_pp"] - c0m["mean_mAP_pp"],
            "auc_rank1_pp": am["auc_rank1_pp"] - c0m["auc_rank1_pp"],
            "auc_mAP_pp": am["auc_mAP_pp"] - c0m["auc_mAP_pp"],
            "rank1_slope_pp_per_epoch": am["rank1_slope_pp_per_epoch"] - c0m["rank1_slope_pp_per_epoch"],
            "mAP_slope_pp_per_epoch": am["mAP_slope_pp_per_epoch"] - c0m["mAP_slope_pp_per_epoch"],
        })
    write_csv(TABLES / "late_window_primary.csv", late)
    final_rows: list[dict[str, Any]] = []
    for arm in ("C0", "C1_reference", "D1"):
        m = metrics[arm]
        final_rows.append({
            "arm": arm,
            "e60_rank1_pp": m["e60_rank1_pp"],
            "e60_mAP_pp": m["e60_mAP_pp"],
            "best_full_rank1_pp": m["best_rank1_pp"],
            "best_full_rank1_epoch": m["best_rank1_epoch"],
            "best_full_mAP_pp": m["best_mAP_pp"],
            "best_full_mAP_epoch": m["best_mAP_epoch"],
            "best_late_rank1_pp": m["best_late_rank1_pp"],
            "best_late_rank1_epoch": m["best_late_rank1_epoch"],
            "best_late_mAP_pp": m["best_late_mAP_pp"],
            "best_late_mAP_epoch": m["best_late_mAP_epoch"],
        })
    for arm in ("C1_reference", "D1"):
        a = metrics[arm]
        final_rows.append({
            "arm": f"{arm}-C0",
            "e60_rank1_pp": a["e60_rank1_pp"] - c0m["e60_rank1_pp"],
            "e60_mAP_pp": a["e60_mAP_pp"] - c0m["e60_mAP_pp"],
            "best_full_rank1_pp": a["best_rank1_pp"] - c0m["best_rank1_pp"],
            "best_full_rank1_epoch": "",
            "best_full_mAP_pp": a["best_mAP_pp"] - c0m["best_mAP_pp"],
            "best_full_mAP_epoch": "",
            "best_late_rank1_pp": a["best_late_rank1_pp"] - c0m["best_late_rank1_pp"],
            "best_late_rank1_epoch": "",
            "best_late_mAP_pp": a["best_late_mAP_pp"] - c0m["best_late_mAP_pp"],
            "best_late_mAP_epoch": "",
        })
    write_csv(TABLES / "final_comparison.csv", final_rows)
    return all_windows, late, metrics


def amp_summary(arm: str, path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    return {
        "arm": arm,
        "total_skip_epochs": sum(1 for r in rows if num(r, "amp_skip") > 0),
        "total_skipped_steps": sum(int(num(r, "skipped_steps")) for r in rows),
        "total_found_inf": sum(int(num(r, "found_inf_count")) for r in rows),
        "total_nonfinite_events": sum(int(num(r, "nonfinite_events")) for r in rows),
        "skip_epochs": ",".join(str(int(num(r, "epoch"))) for r in rows if num(r, "amp_skip") > 0),
        "scale_min": min(num(r, "amp_scale_min") for r in rows),
        "scale_max": max(num(r, "amp_scale_max") for r in rows),
        "scale_final": num(rows[-1], "amp_scale_end"),
        "planned_updates": sum(int(num(r, "planned_updates")) for r in rows),
        "completed_updates": sum(int(num(r, "updates_completed")) for r in rows),
    }


def make_amp_table() -> list[dict[str, Any]]:
    rows = [
        amp_summary("C0", C0_DIR / "amp_audit.csv"),
        amp_summary("C1_reference", C1_DIR / "amp_audit.csv"),
        amp_summary("D1", D1_DIR / "amp_audit.csv"),
    ]
    write_csv(TABLES / "amp_audit.csv", rows)
    return rows


def make_geometry_table() -> list[dict[str, Any]]:
    c0 = read_csv(C0_GEOMETRY)
    d1_raw = read_csv(D1_GEOMETRY)
    d1 = []
    for r in d1_raw:
        item = dict(r)
        item["arm"] = "D1"
        item["geometry_source"] = item.get("geometry_source", "") + "; D1 checkpoint; source row originally labeled C2 by reused Phase 7C loop"
        d1.append(item)
    rows = [r for r in c0 + d1 if r.get("checkpoint") in {"E20", "E40", "E60"}]
    fields = list(rows[0].keys())
    write_csv(GEOMETRY / "geometry_comparison.csv", rows, fields)
    deltas: list[dict[str, Any]] = []
    lookup = {(r["arm"], r["checkpoint"]): r for r in rows}
    numeric = [
        "same_pid_same_clothes_similarity_mean",
        "same_pid_cross_clothes_image_similarity_mean",
        "nearest_negative_image_similarity_mean",
        "image_identity_margin_mean",
        "image_ordering_correct_rate",
        "clothing_state_center_dispersion_mean",
    ]
    for check in ("E20", "E40", "E60"):
        c = lookup[("C0", check)]
        d = lookup[("D1", check)]
        row: dict[str, Any] = {"comparison": "D1-C0", "checkpoint": check}
        for key in numeric:
            row[f"delta_{key}"] = float(d[key]) - float(c[key])
        deltas.append(row)
    write_csv(GEOMETRY / "geometry_delta_D1_minus_C0.csv", deltas)
    return rows


def distribution(values: list[int]) -> dict[str, Any]:
    values = sorted(values)
    mean = statistics.mean(values)
    def quantile(q: float) -> float:
        if len(values) == 1:
            return float(values[0])
        position = (len(values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(values[low])
        return values[low] + (values[high] - values[low]) * (position - low)
    def top_share(q: float) -> float:
        n = max(1, math.ceil(len(values) * q))
        return sum(sorted(values, reverse=True)[:n]) / sum(values)
    return {
        "relation_count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "p90": quantile(0.90),
        "max": max(values),
        "std": statistics.pstdev(values),
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0,
        "total_contributions": sum(values),
        "top_1pct_share": top_share(0.01),
        "top_5pct_share": top_share(0.05),
        "top_10pct_share": top_share(0.10),
    }


def make_exposure_and_balance() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exposure = read_csv(D1_DIR / "relation_exposure.csv")
    derived: list[dict[str, Any]] = []
    for r in exposure:
        derived.append({
            "arm": "D1",
            "epoch": int(float(r["epoch"])),
            "all_images_available": r["all_images_available"],
            "all_pids_available": r["all_pids_available"],
            "all_clothing_states_available": r["all_clothing_states_available"],
            "sampled_unique_images": int(float(r["sampled_unique_images"])),
            "sampled_unique_pids": int(float(r["sampled_unique_pids"])),
            "sampled_unique_clothing_states": int(float(r["sampled_unique_clothing_states"])),
            "margin_active_batches_or_valid_relation_batches": int(float(r["image_relation_active_batches"])),
            "margin_batch_rate": float(r["image_relation_batch_rate"]),
            "margin_pairs_per_epoch": int(float(r["image_relations_per_epoch"])),
            "unique_image_relation_count": int(float(r["unique_image_relation_count"])),
            "planned_updates": int(float(r["planned_updates"])),
            "updates_completed": int(float(r["updates_completed"])),
        })
    write_csv(EXPOSURE / "D1_relation_exposure.csv", derived)
    d1_rows = read_csv(D1_BALANCE)
    d1_values = [int(r["contributions"]) for r in d1_rows]
    c1_rows = read_csv(C1_BALANCE)
    c1_values = [int(r["c1_image_level_contributions"]) for r in c1_rows]
    summary = [
        {"arm": "C1_reference_image_relation", **distribution(c1_values)},
        {"arm": "D1_margin_image_relation", **distribution(d1_values)},
    ]
    write_csv(BALANCE / "balance_summary.csv", summary)
    # Keep the per-state counts visible while clarifying that D1 remains image-level.
    counts: list[dict[str, Any]] = []
    for r in d1_rows:
        counts.append({"pid": r["pid"], "cloth_a": r["cloth_a"], "cloth_b": r["cloth_b"], "D1_image_relation_contributions": r["contributions"]})
    write_csv(BALANCE / "D1_contribution_counts.csv", counts)
    return derived[-1], summary


def fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return "NA"
    return f"{float(x):.{digits}f}"


def make_report(metrics: dict[str, dict[str, Any]], late: list[dict[str, Any]], amp: list[dict[str, Any]], geometry: list[dict[str, Any]], exposure_last: dict[str, Any], balance: list[dict[str, Any]]) -> None:
    c0 = metrics["C0"]
    d1 = metrics["D1"]
    d_rank = d1["auc_rank1_pp"] - c0["auc_rank1_pp"]
    d_map = d1["auc_mAP_pp"] - c0["auc_mAP_pp"]
    d_mean_rank = d1["mean_rank1_pp"] - c0["mean_rank1_pp"]
    d_mean_map = d1["mean_mAP_pp"] - c0["mean_mAP_pp"]
    d_e60_rank = d1["e60_rank1_pp"] - c0["e60_rank1_pp"]
    d_e60_map = d1["e60_mAP_pp"] - c0["e60_mAP_pp"]
    c0g = next(r for r in geometry if r["arm"] == "C0" and r["checkpoint"] == "E60")
    d1g = next(r for r in geometry if r["arm"] == "D1" and r["checkpoint"] == "E60")
    amp_map = {r["arm"]: r for r in amp}
    bal_map = {r["arm"]: r for r in balance}
    q_rank = "YES" if d_rank > 0 else "NO"
    q_map = "YES" if d_map > 0 else "NO"
    report = f"""# Phase 7D-B — Margin-Preserving Relation Training

## Outcome

The first predeclared PRCC margin-preserving relation test completed the full 60-epoch schedule at seed 1234. The result is **STOP / NO PROMOTION**: D1 is below the frozen C0 Bio control in the primary E51-E60 window on both changed-clothes Rank-1 and mAP.

No second seed, lambda sweep, sampler change, or follow-on phase was started.

## 1. Scope and exact sources

This was one new arm only:

- D1: PRCC Bio plus a margin-preserving image-level cross-clothes relation term.
- C0: frozen Phase 7C PRCC Bio E60 run, reused as the matched control and not retrained.
- Seed: `1234`.
- Dataset: PRCC train-only for relation construction; query/gallery were used only for the formal retrieval evaluation.

Source audit: `{ROOT / 'reports/PHASE_7D_B_SOURCE_AUDIT.md'}`.

| Source | Checkpoint | SHA256 |
|---|---|---|
| C0 control | `{C0_DIR / 'model_final.pth'}` | `{sha256(C0_DIR / 'model_final.pth')}` |
| D1 final | `{D1_DIR / 'model_final.pth'}` | `{sha256(D1_DIR / 'model_final.pth')}` |
| D1 best Rank-1 | `{D1_DIR / 'model_best.pth'}` | `{sha256(D1_DIR / 'model_best.pth')}` |
| Common initialization | `/home/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth` | `{sha256(ROOT / 'outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth')}` |

The common initialization state hash was `6567bd0d54e6919be355757c5a2d7fb2e5d1c2128368357e51b7047f5ded97fc`, and D1 strict-loaded it. The frozen relation bank contains 501,040 valid train image pairs.

## 2. D1 definition

For a current-mini-batch same-PID/different-clothes image pair:

```text
s_pos     = cos(z_i, z_j)
s_neg_pair = max(hardest different-PID cosine for i,
                  hardest different-PID cosine for j)
L_margin = mean(relu(0.05 + s_neg_pair - s_pos))
L_total  = L_base + 0.1 * L_margin
```

The feature was normalized `feat[0]` (1024 dimensions). The margin `m=0.05` was predeclared as the middle offline-audit value; it was not tuned. There was no memory bank, queue, EMA, teacher, new sampler, SR3, H_ID, H_cloth, DS, DW, GRL, Reverse, or architecture change.

Schedule: 60 epochs, batch 16, NUM_INSTANCES 4, 551 planned updates/rank/epoch, SGD, base LR `2e-6`, weight decay `0.05`, the frozen 60-epoch cosine scheduler and frozen AMP configuration.

## 3. Completion and exposure audit

- D1 metrics: 60 contiguous rows, E1–E60.
- Planned updates/rank: 33,060.
- Final checkpoint: `{D1_DIR / 'model_final.pth'}`.
- Best checkpoint criterion: changed-clothes Rank-1; D1 best was E44.
- At E60, all 17,896 train images, 150 PIDs, and 300 clothing states remained available.
- The frozen sampler exposed valid cross-clothes image pairs in 99.7278% of batches and 12,232 canonical pairs per epoch. D1 is image-level; its relation multiplicity distribution is therefore expected to match C1, not C2.

Detailed exposure: `{EXPOSURE / 'D1_relation_exposure.csv'}`.

## 4. Training trajectory and primary comparison

All retrieval values below are percentages; AUC is normalized trapezoidal AUC over the stated epoch window.

| Arm | Mean Rank-1 E51-E60 | Mean mAP E51-E60 | AUC Rank-1 | AUC mAP | Rank-1 slope | mAP slope |
|---|---:|---:|---:|---:|---:|---:|
| C0 Bio | {fmt(c0['mean_rank1_pp'])} | {fmt(c0['mean_mAP_pp'])} | {fmt(c0['auc_rank1_pp'])} | {fmt(c0['auc_mAP_pp'])} | {fmt(c0['rank1_slope_pp_per_epoch'])} | {fmt(c0['mAP_slope_pp_per_epoch'])} |
| D1 margin relation | {fmt(d1['mean_rank1_pp'])} | {fmt(d1['mean_mAP_pp'])} | {fmt(d1['auc_rank1_pp'])} | {fmt(d1['auc_mAP_pp'])} | {fmt(d1['rank1_slope_pp_per_epoch'])} | {fmt(d1['mAP_slope_pp_per_epoch'])} |
| D1 - C0 | {fmt(d_mean_rank)} | {fmt(d_mean_map)} | {fmt(d_rank)} | {fmt(d_map)} | {fmt(d1['rank1_slope_pp_per_epoch'] - c0['rank1_slope_pp_per_epoch'])} | {fmt(d1['mAP_slope_pp_per_epoch'] - c0['mAP_slope_pp_per_epoch'])} |

| Arm | E60 Rank-1 | E60 mAP | Best full Rank-1 (epoch) | Best full mAP (epoch) | Best late Rank-1 (epoch) | Best late mAP (epoch) |
|---|---:|---:|---:|---:|---:|---:|
| C0 | {fmt(c0['e60_rank1_pp'])} | {fmt(c0['e60_mAP_pp'])} | {fmt(c0['best_rank1_pp'])} (E{c0['best_rank1_epoch']}) | {fmt(c0['best_mAP_pp'])} (E{c0['best_mAP_epoch']}) | {fmt(c0['best_late_rank1_pp'])} (E{c0['best_late_rank1_epoch']}) | {fmt(c0['best_late_mAP_pp'])} (E{c0['best_late_mAP_epoch']}) |
| D1 | {fmt(d1['e60_rank1_pp'])} | {fmt(d1['e60_mAP_pp'])} | {fmt(d1['best_rank1_pp'])} (E{d1['best_rank1_epoch']}) | {fmt(d1['best_mAP_pp'])} (E{d1['best_mAP_epoch']}) | {fmt(d1['best_late_rank1_pp'])} (E{d1['best_late_rank1_epoch']}) | {fmt(d1['best_late_mAP_pp'])} (E{d1['best_late_mAP_epoch']}) |
| D1 - C0 | {fmt(d_e60_rank)} | {fmt(d_e60_map)} | {fmt(d1['best_rank1_pp'] - c0['best_rank1_pp'])} | {fmt(d1['best_mAP_pp'] - c0['best_mAP_pp'])} | {fmt(d1['best_late_rank1_pp'] - c0['best_late_rank1_pp'])} | {fmt(d1['best_late_mAP_pp'] - c0['best_late_mAP_pp'])} |

For context only, the already completed C1 image-level reference had E51-E60 AUC `65.3605 / 63.2110`; it was not retrained in this phase. D1 was not an improvement over C0, and therefore also did not provide evidence that replacing the positive-only term with this first margin formulation solves the PRCC failure.

Window table: `{TABLES / 'window_analysis.csv'}`. Primary table: `{TABLES / 'late_window_primary.csv'}`.

## 5. Relation balance

D1 retains image-level pair contributions. Across 150 PID/clothing-state relations:

| Arm | Mean | Median | P90 | Max | CV | Top 1% share | Top 5% share | Top 10% share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 image relation reference | {fmt(bal_map['C1_reference_image_relation']['mean'], 2)} | {fmt(bal_map['C1_reference_image_relation']['median'], 2)} | {fmt(bal_map['C1_reference_image_relation']['p90'], 2)} | {fmt(bal_map['C1_reference_image_relation']['max'], 2)} | {fmt(bal_map['C1_reference_image_relation']['coefficient_of_variation'], 4)} | {fmt(bal_map['C1_reference_image_relation']['top_1pct_share'], 4)} | {fmt(bal_map['C1_reference_image_relation']['top_5pct_share'], 4)} | {fmt(bal_map['C1_reference_image_relation']['top_10pct_share'], 4)} |
| D1 margin image relation | {fmt(bal_map['D1_margin_image_relation']['mean'], 2)} | {fmt(bal_map['D1_margin_image_relation']['median'], 2)} | {fmt(bal_map['D1_margin_image_relation']['p90'], 2)} | {fmt(bal_map['D1_margin_image_relation']['max'], 2)} | {fmt(bal_map['D1_margin_image_relation']['coefficient_of_variation'], 4)} | {fmt(bal_map['D1_margin_image_relation']['top_1pct_share'], 4)} | {fmt(bal_map['D1_margin_image_relation']['top_5pct_share'], 4)} | {fmt(bal_map['D1_margin_image_relation']['top_10pct_share'], 4)} |

The balance profile is the same as the image-level C1 reference. D1 did not test state-level balancing.

## 6. Train-only geometry

The E60 geometry is:

| Metric | C0 | D1 | D1 - C0 |
|---|---:|---:|---:|
| Same-PID same-clothes cosine | {fmt(c0g['same_pid_same_clothes_similarity_mean'])} | {fmt(d1g['same_pid_same_clothes_similarity_mean'])} | {fmt(float(d1g['same_pid_same_clothes_similarity_mean']) - float(c0g['same_pid_same_clothes_similarity_mean']))} |
| Same-PID cross-clothes cosine | {fmt(c0g['same_pid_cross_clothes_image_similarity_mean'])} | {fmt(d1g['same_pid_cross_clothes_image_similarity_mean'])} | {fmt(float(d1g['same_pid_cross_clothes_image_similarity_mean']) - float(c0g['same_pid_cross_clothes_image_similarity_mean']))} |
| Nearest-negative PID-center cosine | {fmt(c0g['nearest_negative_image_similarity_mean'])} | {fmt(d1g['nearest_negative_image_similarity_mean'])} | {fmt(float(d1g['nearest_negative_image_similarity_mean']) - float(c0g['nearest_negative_image_similarity_mean']))} |
| Image identity margin | {fmt(c0g['image_identity_margin_mean'])} | {fmt(d1g['image_identity_margin_mean'])} | {fmt(float(d1g['image_identity_margin_mean']) - float(c0g['image_identity_margin_mean']))} |
| Ordering correctness | {fmt(c0g['image_ordering_correct_rate'])} | {fmt(d1g['image_ordering_correct_rate'])} | {fmt(float(d1g['image_ordering_correct_rate']) - float(c0g['image_ordering_correct_rate']))} |
| Clothing-state center dispersion | {fmt(c0g['clothing_state_center_dispersion_mean'])} | {fmt(d1g['clothing_state_center_dispersion_mean'])} | {fmt(float(d1g['clothing_state_center_dispersion_mean']) - float(c0g['clothing_state_center_dispersion_mean']))} |

D1 increases cross-clothes positive cosine only slightly (`+{fmt(float(d1g['same_pid_cross_clothes_image_similarity_mean']) - float(c0g['same_pid_cross_clothes_image_similarity_mean']))}`), but nearest-negative cosine increases more (`+{fmt(float(d1g['nearest_negative_image_similarity_mean']) - float(c0g['nearest_negative_image_similarity_mean']))}`). The resulting identity margin decreases by `{fmt(float(d1g['image_identity_margin_mean']) - float(c0g['image_identity_margin_mean']))}`. This is the same compactness-without-discrimination failure mode, not a successful margin repair.

Train-only geometry tables: `{GEOMETRY / 'geometry_comparison.csv'}` and `{GEOMETRY / 'geometry_delta_D1_minus_C0.csv'}`.

## 7. AMP audit

| Arm | Skip epochs | Skipped steps | found_inf | Nonfinite | Scale min | Scale max | Final scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | {amp_map['C0']['total_skip_epochs']} | {amp_map['C0']['total_skipped_steps']} | {amp_map['C0']['total_found_inf']} | {amp_map['C0']['total_nonfinite_events']} | {fmt(amp_map['C0']['scale_min'], 0)} | {fmt(amp_map['C0']['scale_max'], 0)} | {fmt(amp_map['C0']['scale_final'], 0)} |
| D1 | {amp_map['D1']['total_skip_epochs']} | {amp_map['D1']['total_skipped_steps']} | {amp_map['D1']['total_found_inf']} | {amp_map['D1']['total_nonfinite_events']} | {fmt(amp_map['D1']['scale_min'], 0)} | {fmt(amp_map['D1']['scale_max'], 0)} | {fmt(amp_map['D1']['scale_final'], 0)} |

D1 has 24 skipped/`found_inf` steps versus C0's 22, with a lower minimum scale (128 versus 256). This is a measurable numerical difference, but it is small relative to the 33,060 planned updates and does not explain a positive D1 result—the retrieval result is negative even with the audit recorded. It should be treated as a caveat, not as evidence for promotion.

AMP table: `{TABLES / 'amp_audit.csv'}`.

## 8. Decision

| Question | Answer | Evidence |
|---|---|---|
| Did D1 improve E51-E60 Rank-1 AUC over C0? | {q_rank} | `{fmt(d_rank)} pp` |
| Did D1 improve E51-E60 mAP AUC over C0? | {q_map} | `{fmt(d_map)} pp` |
| Did D1 improve E60 Rank-1 / mAP? | NO | `{fmt(d_e60_rank)} / {fmt(d_e60_map)} pp` |
| Did D1 repair the train-only identity margin? | NO | E60 margin delta `{fmt(float(d1g['image_identity_margin_mean']) - float(c0g['image_identity_margin_mean']))}` |
| Is this a clean promotion? | NO | both primary AUC deltas are negative |

### Strongest mechanistic conclusion

The first margin-preserving relation implementation did not repair the PRCC failure. At E60 it produced a very small positive compactness change, while nearest-negative identity similarity rose more, leaving the identity margin lower than C0. Its primary retrieval trajectory was also below C0. Thus the current evidence supports **STOP**, not a claim that margin-aware relation training works.

## 9. Final stop statement

- D1 seed1234 completed E1–E60.
- No second seed was started.
- No lambda tuning, sampler change, new architecture, SR3, H_ID, prototype/memory/EMA/queue mechanism, or Phase 7D-C was started.
- Phase 7D-B is **STOP / NO PROMOTION** pending any future human decision.

## Artifacts

- Source audit: `{ROOT / 'reports/PHASE_7D_B_SOURCE_AUDIT.md'}`
- Final manifest: `{MANIFEST}`
- D1 run: `{D1_DIR}`
- Report tables: `{TABLES}`
- Geometry: `{GEOMETRY}`
- Exposure: `{EXPOSURE}`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (TABLES, GEOMETRY, EXPOSURE, BALANCE, ANALYSIS / "plots", ANALYSIS / "source_audit"):
        directory.mkdir(parents=True, exist_ok=True)
    c0 = read_metrics(C0_DIR)
    c1 = read_metrics(C1_DIR)
    d1 = read_metrics(D1_DIR)
    make_trajectory(c0, d1, c1)
    _, late, metrics = make_window_tables(c0, d1, c1)
    amp = make_amp_table()
    geometry = make_geometry_table()
    exposure_last, balance = make_exposure_and_balance()
    make_report(metrics, late, amp, geometry, exposure_last, balance)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    manifest.update({
        "analysis_completed": True,
        "analysis_completed_utc": now(),
        "promotion_status": "STOP_NO_PROMOTION",
        "second_seed_started": False,
        "phase7dc_started": False,
        "primary_comparison": {
            "window": "E51-E60",
            "D1_minus_C0_rank1_auc_pp": metrics["D1"]["auc_rank1_pp"] - metrics["C0"]["auc_rank1_pp"],
            "D1_minus_C0_mAP_auc_pp": metrics["D1"]["auc_mAP_pp"] - metrics["C0"]["auc_mAP_pp"],
            "D1_minus_C0_rank1_mean_pp": metrics["D1"]["mean_rank1_pp"] - metrics["C0"]["mean_rank1_pp"],
            "D1_minus_C0_mAP_mean_pp": metrics["D1"]["mean_mAP_pp"] - metrics["C0"]["mean_mAP_pp"],
        },
        "analysis": {
            "window_analysis": str(TABLES / "window_analysis.csv"),
            "late_window_primary": str(TABLES / "late_window_primary.csv"),
            "final_comparison": str(TABLES / "final_comparison.csv"),
            "amp_audit": str(TABLES / "amp_audit.csv"),
            "geometry_comparison": str(GEOMETRY / "geometry_comparison.csv"),
            "geometry_delta": str(GEOMETRY / "geometry_delta_D1_minus_C0.csv"),
            "relation_exposure": str(EXPOSURE / "D1_relation_exposure.csv"),
            "relation_balance": str(BALANCE / "balance_summary.csv"),
            "report": str(REPORT),
        },
    })
    write_json(MANIFEST, manifest)
    print(json.dumps({"report": str(REPORT), "manifest": str(MANIFEST), "D1_minus_C0": manifest["primary_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
