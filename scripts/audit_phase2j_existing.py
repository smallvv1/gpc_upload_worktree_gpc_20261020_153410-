#!/usr/local/bin/python3
"""Read-only comparison of the frozen LTCC Run-1 and Phase-2I A probe."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
R1_MANIFEST = ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/RUN_MANIFEST.json"
R1_EVENT = ROOT / "outputs/reproduction_phase2/LTCC/differ/run1/1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/train/events.out.tfevents.1788244774.nb-a431f85a02d6004f02-88f76bccc-bqm6x.998752.0"
R2_JSON = ROOT / "diagnostics/initialization_footprint_15epoch_phase2i.json"


def r1_epoch_curve():
    acc = EventAccumulator(str(R1_EVENT), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    train_tags = [
        ("loss/total", "train_loss"), ("loss/id", "id_loss"),
        ("loss/triplet", "triplet_loss"), ("loss/clip_bio_0", "bio_loss"),
        ("loss/clip_bio_reverse_0", "reverse0_loss"),
        ("loss/clip_bio_reverse_1", "reverse1_loss"),
        ("acc/id", "id_acc"), ("acc/clip_bio_0", "bio_acc"),
        ("acc/clip_bio_reverse_0", "reverse0_acc"),
        ("acc/clip_bio_reverse_1", "reverse1_acc"), ("lr", "lr"),
    ]
    series = {tag: acc.Scalars(tag) for tag, _ in train_tags}
    metric = {tag: {int(x.step): float(x.value) for x in acc.Scalars(tag)} for tag in ("cc/rank1", "cc/mAP", "general/rank1", "general/mAP")}
    for epoch in range(1, 16):
        row = {"epoch": epoch}
        lo, hi = 596 + (epoch - 1) * 596, 596 + epoch * 596
        for tag, name in train_tags:
            values = [float(x.value) for x in series[tag] if lo <= x.step < hi]
            row[name] = float(np.mean(values)) if values else None
        row["cc_rank1"] = metric["cc/rank1"].get(epoch)
        row["cc_mAP"] = metric["cc/mAP"].get(epoch)
        row["general_rank1"] = metric["general/rank1"].get(epoch)
        row["general_mAP"] = metric["general/mAP"].get(epoch)
        out[epoch] = row
    return out


def r2_epoch_curve():
    data = json.loads(R2_JSON.read_text())
    out = {}
    for record in data["results"]["A"]["records"]:
        total = record["id_loss"] + record["triplet_loss"] + record["bio_loss"] + record["reverse1_loss"]
        out[int(record["epoch"])] = {
            "epoch": int(record["epoch"]), "train_loss": total,
            "id_loss": record["id_loss"], "triplet_loss": record["triplet_loss"],
            "bio_loss": record["bio_loss"], "reverse0_loss": record["reverse0_loss"],
            "reverse1_loss": record["reverse1_loss"], "id_acc": record["id_acc"],
            "bio_acc": record["bio_acc"], "reverse0_acc": record["reverse0_acc"],
            "reverse1_acc": record["reverse1_acc"], "lr": record["lr_normal"],
            "cc_rank1": record["cc"]["rank1"], "cc_mAP": record["cc"]["mAP"],
            "general_rank1": record["general"]["rank1"], "general_mAP": record["general"]["mAP"],
        }
    return out


def distances(a, b):
    names = ["train_loss", "id_loss", "triplet_loss", "bio_loss", "reverse0_loss", "reverse1_loss", "id_acc", "bio_acc", "reverse0_acc", "reverse1_acc", "cc_rank1", "cc_mAP", "general_rank1", "general_mAP", "lr"]
    out = {}
    for name in names:
        x = np.array([a[e][name] for e in range(1, 16)], dtype=float)
        y = np.array([b[e][name] for e in range(1, 16)], dtype=float)
        delta = y - x
        out[name] = {
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "mae": float(np.mean(np.abs(delta))),
            "pearson": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-15 and np.std(y) > 1e-15 else None,
            "delta_epoch15": float(delta[-1]),
        }
    return out


def first_clear(curve, threshold):
    vals = [abs(float(curve[e])) > threshold for e in range(1, 16)]
    clear = next((e for e, flag in enumerate(vals, 1) if flag), None)
    sustained = None
    for e in range(1, 14):
        if all(vals[e - 1:e + 2]):
            sustained = e
            break
    return {"first_clear": clear, "first_sustained_3_epochs": sustained}


def main():
    manifest = json.loads(R1_MANIFEST.read_text())
    r1_snapshot = manifest["config_snapshot"]
    # Parse the snapshot without depending on yacs internals.
    import yaml
    r1_cfg = yaml.safe_load(r1_snapshot)
    from initialization_footprint_phase2i import make_config
    r2_cfg = yaml.safe_load(make_config().dump())
    fields = [
        ("DATA.DATASET", "DATA", "DATASET"), ("DATA.ROOT", "DATA", "ROOT"),
        ("DATA.CAPTION_DIR", "DATA", "CAPTION_DIR"), ("DATA.BATCH_SIZE", "DATA", "BATCH_SIZE"),
        ("DATA.NUM_INSTANCES", "DATA", "NUM_INSTANCES"), ("DATA.NUM_WORKERS", "DATA", "NUM_WORKERS"),
        ("DATA.SUMMERY_TEXT", "DATA", "SUMMERY_TEXT"), ("DATA.NOBIO_INDEX", "DATA", "NOBIO_INDEX"),
        ("MODEL.CLIP_DIM", "MODEL", "CLIP_DIM"), ("MODEL.LAST_LAYER", "MODEL", "LAST_LAYER"),
        ("MODEL.LOSS_TYPE", "MODEL", "LOSS_TYPE"), ("MODEL.DIST_TRAIN", "MODEL", "DIST_TRAIN"),
        ("SOLVER.BASE_LR", "SOLVER", "BASE_LR"), ("SOLVER.WARMUP_EPOCHS", "SOLVER", "WARMUP_EPOCHS"),
        ("SOLVER.WARMUP_FACTOR", "SOLVER", "WARMUP_FACTOR"), ("SOLVER.WARMUP_METHOD", "SOLVER", "WARMUP_METHOD"),
        ("SOLVER.MAX_EPOCHS", "SOLVER", "MAX_EPOCHS"), ("SOLVER.SEED", "SOLVER", "SEED"),
    ]
    config_rows = []
    for name, section, key in fields:
        rv, pv = r1_cfg.get(section, {}).get(key), r2_cfg.get(section, {}).get(key)
        config_rows.append({"field": name, "run1_snapshot": rv, "probe_A": pv, "equal": rv == pv})
    config_rows.append({"field": "effective runtime MODEL.DIST_TRAIN", "run1_snapshot": False, "probe_A": False, "equal": True, "note": "Run-1 launch command overrides it to True; two event writers and DDP log confirm DDP."})
    config_rows.append({"field": "effective runtime loader", "run1_snapshot": "DataLoaderX + BackgroundGenerator + CUDA stream", "probe_A": "torch.utils.data.DataLoader", "equal": False})
    config_rows.append({"field": "effective runtime gradient reduction", "run1_snapshot": "two-process DDP", "probe_A": "single process rank-0 sampler; no all-reduce", "equal": False})
    config_rows.append({"field": "instrumentation", "run1_snapshot": "processor _Phase2MetricWriter/metrics CSV", "probe_A": "ScalarCapture/manual loop", "equal": False, "note": "observation-only calls; no RNG/model/loss mutation identified"})
    r1, r2 = r1_epoch_curve(), r2_epoch_curve()
    dist = distances(r1, r2)
    payload = {
        "phase": "2J-A",
        "same_config_verdict": "FAIL_STRICT_SAME_CONFIG",
        "run1_manifest": str(R1_MANIFEST), "run1_event": str(R1_EVENT), "phase2i_json": str(R2_JSON),
        "run1_command": manifest["command"], "config_rows": config_rows,
        "instrumentation_verdict": "BEHAVIOR_NEUTRAL_FOR_MODEL_LOSS_OPTIMIZER; loader/DDP topology differs",
        "run1_epoch_curve_rank0": r1, "probe_A_epoch_curve": r2, "distances": dist,
        "first_difference": {
            "cc_rank1": first_clear({e: r2[e]["cc_rank1"]-r1[e]["cc_rank1"] for e in range(1,16)}, .01),
            "cc_mAP": first_clear({e: r2[e]["cc_mAP"]-r1[e]["cc_mAP"] for e in range(1,16)}, .005),
            "train_loss": first_clear({e: r2[e]["train_loss"]-r1[e]["train_loss"] for e in range(1,16)}, .05),
            "reverse1_acc": first_clear({e: r2[e]["reverse1_acc"]-r1[e]["reverse1_acc"] for e in range(1,16)}, .01),
        },
        "notes": [
            "Run-1 rank-0 event stream 998752 was used for the numeric curve comparison.",
            "Phase2I A stores no standalone train_loss; its total is derived as ID+Triplet+Bio+Reverse1.",
            "Because effective DDP/loader paths differ, this comparison is not a same-config nondeterminism test.",
        ],
    }
    out = ROOT / "diagnostics/phase2j_existing_config_audit.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = ROOT / "reports/PHASE_2J_A_EXISTING_CONFIG_AUDIT.md"
    lines = ["# PHASE 2J-A — existing Run-1 / Phase-2I-A audit", "", "**STRICT SAME-CONFIG: FAIL.**", "", "Run-1 launch command enables two-process DDP; its two event writers and `DataLoaderX` path confirm that runtime. Phase-2I A is a single-process rank-0 diagnostic with a regular DataLoader. The observation wrappers are behavior-neutral, but DDP gradient reduction and loader/prefetch topology are not identical.", "", "| Field | Run-1 snapshot/evidence | Phase-2I A | Equal |", "|---|---|---|---|"]
    for row in config_rows:
        lines.append(f"| {row['field']} | `{row.get('run1_snapshot')}` | `{row.get('probe_A')}` | {row['equal']} |")
    lines += ["", "## R1 vs A epoch 1–15", "", "Metrics use Run-1 rank-0 event writer 998752. RMSE/MAE/r are reported as A minus Run-1:", "", "| Curve | RMSE | MAE | Pearson r |", "|---|---:|---:|---:|"]
    for key in ["cc_rank1","cc_mAP","general_rank1","general_mAP","train_loss","id_loss","triplet_loss","bio_loss","reverse0_loss","reverse1_loss","id_acc","bio_acc","reverse0_acc","reverse1_acc","lr"]:
        v=dist[key]; lines.append(f"| {key} | {v['rmse']:.9g} | {v['mae']:.9g} | {v['pearson'] if v['pearson'] is not None else 'NA'} |")
    lines += ["", "This is an early-trajectory comparison only; it cannot support a same-config nondeterminism claim. Full machine-readable evidence: `diagnostics/phase2j_existing_config_audit.json`.", ""]
    report.write_text("\n".join(lines))
    print(json.dumps({"same_config_verdict": payload["same_config_verdict"], "first_difference": payload["first_difference"], "report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
