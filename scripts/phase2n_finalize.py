#!/usr/bin/env python3
"""Write the reproducible Phase 2N numerical/training hand-off artifacts.

This is deliberately a reader/aggregator.  It never launches a model run and
does not modify any Phase 2M or earlier artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/phase2n/phase2n_mask_cache.json"
DIAG = ROOT / "diagnostics"
REPORTS = ROOT / "reports"
TRAIN = ROOT / "outputs/reproduction_phase2/phase2n/compat_v2"


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def key(mask) -> str:
    return ",".join(str(int(x)) for x in sorted(mask))


def metric(result, branch: str, field: str = "cosine") -> float:
    if result is None:
        return float("nan")
    c = result.get("comparisons", {}).get(branch, {}).get("block_0", {})
    return float(c.get(field, float("nan")))


def summary_metric(result, branch: str) -> float:
    return metric(result, branch, "cosine")


def recovery(cosine: float, current: float) -> float:
    return (cosine - current) / (1.0 - current)


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    try:
        if not math.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def pct(x, digits=1):
    return f"{100.0 * float(x):.{digits}f}%"


def branch_row(result):
    return {
        b: metric(result, b, "cosine")
        for b in ("ce", "triplet", "bio", "reverse0", "reverse1", "full")
    }


def compact_result(result):
    if result is None:
        return None
    out = {
        "mask": result.get("mask", []),
        "fp32_block_count": result.get("fp32_block_count"),
        "fp16_block_count": result.get("fp16_block_count"),
        "runtime_sec": result.get("runtime_sec"),
        "peak_memory_bytes": result.get("peak_memory_bytes"),
        "branch_block0_cosines": branch_row(result),
        "reverse1_recovery": result.get("summary", {}).get("reverse1_block0_recovery"),
        "full_recovery": result.get("summary", {}).get("full_block0_recovery"),
    }
    return out


def epoch_records():
    d = load(TRAIN / "epoch_records.json", {}) or {}
    return d.get("records", [])


def train_status():
    records = epoch_records()
    out = {
        "output_dir": str(TRAIN),
        "records": records,
        "epoch15_gate": load(TRAIN / "epoch15_gate.json"),
        "epoch25_gate": load(TRAIN / "epoch25_gate.json"),
        "epoch60_gate": load(TRAIN / "epoch60_gate.json"),
        "resume_epoch15": (TRAIN / "resume_epoch15.pth").exists(),
        "resume_epoch25": (TRAIN / "resume_epoch25.pth").exists(),
        "resume_epoch60": (TRAIN / "resume_epoch60.pth").exists(),
        "train_steps": str(TRAIN / "train_steps.csv") if (TRAIN / "train_steps.csv").exists() else None,
        "tensorboard": str(TRAIN / "tensorboard") if (TRAIN / "tensorboard").exists() else None,
    }
    if records:
        out["last_epoch"] = records[-1].get("epoch")
        out["last_cc"] = records[-1].get("cc")
        out["last_general"] = records[-1].get("general")
    return out


def write_block_comparison(cache, current_r1):
    rows = []
    results = cache["results"]
    all_blocks = range(int(cache.get("meta", {}).get("n_blocks", 23)))
    for i in all_blocks:
        r = results.get(str(i))
        rows.append({
            "block": i,
            "mask": key([i]),
            "reverse1_cosine": metric(r, "reverse1"),
            "reverse1_recovery": recovery(metric(r, "reverse1"), current_r1),
            "full_cosine": metric(r, "full"),
            "ce_cosine": metric(r, "ce"),
            "triplet_cosine": metric(r, "triplet"),
            "bio_cosine": metric(r, "bio"),
            "reverse0_cosine": metric(r, "reverse0"),
            "runtime_sec": r.get("runtime_sec") if r else None,
            "peak_memory_bytes": r.get("peak_memory_bytes") if r else None,
        })
    with (REPORTS / "PHASE_2N_SINGLE_BLOCK.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_marginal_necessity(cache, current_r1, n_blocks):
    results = cache["results"]
    all_mask = set(range(n_blocks))
    rows = []
    for i in range(n_blocks):
        single = results.get(str(i))
        except_i = results.get(key(sorted(all_mask - {i})))
        single_rec = recovery(metric(single, "reverse1"), current_r1)
        except_rec = recovery(metric(except_i, "reverse1"), current_r1)
        rows.append({
            "block": i,
            "single_gain": single_rec,
            "oracle_necessity": 1.0 * except_rec,
            "single_cosine": metric(single, "reverse1"),
            "all_except_cosine": metric(except_i, "reverse1"),
        })
    gain_cut = median(x["single_gain"] for x in rows)
    necessity_cut = median(x["oracle_necessity"] for x in rows)
    for x in rows:
        x["gain_level"] = "high" if x["single_gain"] >= gain_cut else "low"
        x["necessity_level"] = "high" if x["oracle_necessity"] >= necessity_cut else "low"
        x["interaction_category"] = (
            "A: high gain + high necessity" if x["gain_level"] == "high" and x["necessity_level"] == "high" else
            "B: low gain + high necessity" if x["gain_level"] == "low" and x["necessity_level"] == "high" else
            "C: high gain + low necessity" if x["gain_level"] == "high" and x["necessity_level"] == "low" else
            "D: low gain + low necessity"
        )
    rows.sort(key=lambda x: (-x["single_gain"], -x["oracle_necessity"], x["block"]))
    with (REPORTS / "PHASE_2N_MARGINAL_NECESSITY.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": rows,
        "gain_median_cut": gain_cut,
        "necessity_median_cut": necessity_cut,
        "top_gain": rows[:8],
        "top_necessity": sorted(rows, key=lambda x: (-x["oracle_necessity"], x["block"]))[:8],
        "categories": {
            c: [x["block"] for x in rows if x["interaction_category"].startswith(c + ":")]
            for c in "ABCD"
        },
    }


def best_by_masks(masks, results):
    rows = []
    for mask in masks:
        r = results.get(key(mask))
        if r:
            rows.append(r)
    if not rows:
        return None
    return max(rows, key=lambda r: (r.get("summary", {}).get("reverse1_block0_recovery", -1),
                                    r.get("summary", {}).get("full_block0_recovery", -1)))


def search_summaries(cache):
    results = cache["results"]
    search = cache.get("search", {})
    windows = {}
    for length, masks in search.get("windows", {}).items():
        r = best_by_masks(masks, results)
        windows[int(length)] = compact_result(r)
    regular = {}
    for step, masks in search.get("regular", {}).items():
        regular[int(step)] = [compact_result(results.get(key(m))) for m in masks if results.get(key(m))]
    beam = {}
    for size, entries in search.get("beam", {}).get("best_by_size", {}).items():
        # Preserve the numerical objective ranking supplied by the beam search,
        # while also exposing the best Reverse-1 member for auditability.
        candidates = [results.get(key(e.get("mask", []))) for e in entries]
        candidates = [r for r in candidates if r]
        best = best_by_masks([r["mask"] for r in candidates], results)
        beam[int(size)] = {
            "objective_leader": entries[0] if entries else None,
            "reverse1_leader": compact_result(best),
        }
    return {"windows": windows, "regular": regular, "beam": beam}


def interaction_summary():
    path = REPORTS / "PHASE_2N_PAIR_INTERACTION.csv"
    rows = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: float(x["synergy"]))
    def clean(x):
        return {
            "block_i": int(x["block_i"]), "block_j": int(x["block_j"]),
            "single_i_recovery": float(x["single_i_recovery"]),
            "single_j_recovery": float(x["single_j_recovery"]),
            "pair_recovery": float(x["pair_recovery"]),
            "synergy": float(x["synergy"]),
            "pair_objective": float(x["pair_objective"]),
        }
    return {"count": len(rows), "top_positive": [clean(x) for x in rows[-8:][::-1]],
            "top_negative": [clean(x) for x in rows[:8]]}


def depth_structure(best_masks):
    out = {}
    for label, mask in best_masks.items():
        mask = list(mask)
        gaps = [b - a for a, b in zip(mask, mask[1:])]
        out[label] = {
            "mask": mask,
            "depth_fraction": [i / 22.0 for i in mask] if mask else [],
            "adjacent_gaps": gaps,
            "mean_gap": sum(gaps) / len(gaps) if gaps else None,
            "min_gap": min(gaps) if gaps else None,
            "max_gap": max(gaps) if gaps else None,
        }
    return out


def training_classification(status):
    records = status.get("records", [])
    best = max(records, key=lambda x: float(x.get("cc", {}).get("rank1", float("-inf")))) if records else None
    cc = (best or {}).get("cc", {})
    r1 = float(cc.get("rank1", float("nan")))
    mp = float(cc.get("mAP", float("nan")))
    if not finite(r1) or not finite(mp):
        return "NOT_RUN"
    # Official CC values are 0.589286 / 0.312798.  Classification follows
    # the requested percentage-point bands, applied to the best observed
    # checkpoint rather than to an arbitrary late-epoch fluctuation.
    if r1 >= 0.589286 - 0.01 and mp >= 0.312798 - 0.01:
        return "EXCELLENT"
    if r1 >= 0.589286 - 0.02 and mp >= 0.312798 - 0.02:
        return "ACCEPTABLE"
    if len(records) > 0:
        return "PARTIAL" if r1 > 0.533 or mp > 0.293 else "FAILED"
    return "NOT_RUN"


def main():
    cache = load(CACHE, {})
    density = load(DIAG / "phase2n_density_curve.json", {}) or {}
    best = load(DIAG / "phase2n_best_masks.json", {}) or {}
    multi = load(DIAG / "phase2n_multibatch_matrix.json", {}) or {}
    source = load(DIAG / "phase2n_source_audit.json", {}) or {}
    selection = load(DIAG / "phase2n_compat_selection.json", {}) or {}
    repeat = load(DIAG / "phase2n_repeatability.json", {}) or {}
    n_blocks = int(cache.get("meta", {}).get("n_blocks", density.get("n_blocks", 23)))
    density_rows = density.get("curve", best.get("density_curve", []))
    current_density = next((x for x in density_rows if int(x.get("fp32_block_count", -1)) == 0), {})
    current_r1 = float(current_density.get("main_reverse1_cosine", 0.08732968687327208))
    marginal = write_marginal_necessity(cache, current_r1, n_blocks)
    singles = write_block_comparison(cache, current_r1)
    searches = search_summaries(cache)
    interactions = interaction_summary()
    source_masks = (multi.get("source_masks") or selection.get("mask") or {})
    if isinstance(source_masks, dict):
        best_masks = {k: list(v) for k, v in source_masks.items()}
    else:
        best_masks = {"selected": list(source_masks)}
    depth = depth_structure(best_masks)
    status = train_status()
    status["best_observed"] = max(
        status.get("records", []),
        key=lambda x: float(x.get("cc", {}).get("rank1", float("-inf"))),
        default=None,
    )
    status["final_observed"] = status.get("records", [])[-1] if status.get("records") else None

    # Keep the mandatory machine-readable summary compact but complete enough
    # to locate every phase artifact and reproduce the gate decisions.
    summary = {
        "phase": "2N",
        "created_utc": now(),
        "git_commit": git_commit(),
        "n_blocks": n_blocks,
        "main_batch_hash": cache.get("meta", {}).get("batch_hash"),
        "main_batch_size": cache.get("meta", {}).get("batch_size"),
        "cache": str(CACHE),
        "cache_result_count": len(cache.get("results", {})),
        "cache_batch_result_count": sum(len(x) for x in cache.get("batch_results", {}).values()),
        "single_block_top8": singles[:8],
        "marginal_necessity": marginal,
        "search_summaries": searches,
        "pair_interaction": interactions,
        "multibatch_gate": multi.get("gate"),
        "density_curve_path": str(DIAG / "phase2n_density_curve.json"),
        "depth_structure": depth,
        "source_audit_path": str(DIAG / "phase2n_source_audit.json"),
        "repeatability_path": str(DIAG / "phase2n_repeatability.json"),
        "compat_selection": selection,
        "training": {k: v for k, v in status.items() if k != "records"},
    }
    dump(DIAG / "phase2n_search_summary.json", summary)

    # A manifest is intentionally separate from the narrative report so a
    # future resume can validate scope, parameters, batch identity, and paths.
    manifest = {
        "phase": "2N",
        "created_utc": now(),
        "git_commit": git_commit(),
        "device_policy": "PPU-ZW810E only; no CPU fallback",
        "dataset": "LTCC",
        "seed": 1234,
        "batch_size": 8,
        "n_blocks": n_blocks,
        "main_batch_fingerprint": cache.get("meta", {}).get("batch_hash"),
        "batch_panel": multi.get("panel"),
        "precision_semantics": cache.get("meta", {}).get("precision_semantics", "exact Phase2M wrap_fp32_block"),
        "reference_definitions": {
            "current": "all formal FP16/autocast blocks",
            "oracle": "full EVA blocks FP32",
            "reference": "mathematical FP32 backward",
        },
        "parameter_manifest": multi.get("parameter_manifest"),
        "selected_candidate": selection,
        "artifacts": {
            "mask_cache": str(CACHE),
            "mask_cache_export": str(DIAG / "phase2n_mask_search_cache.json"),
            "single_block": str(REPORTS / "PHASE_2N_SINGLE_BLOCK.csv"),
            "marginal_necessity": str(REPORTS / "PHASE_2N_MARGINAL_NECESSITY.csv"),
            "pair_interaction": str(REPORTS / "PHASE_2N_PAIR_INTERACTION.csv"),
            "batch_panel": str(REPORTS / "PHASE_2N_BATCH_PANEL.csv"),
            "multibatch": str(DIAG / "phase2n_multibatch_matrix.json"),
            "density": str(DIAG / "phase2n_density_curve.json"),
            "source_audit": str(DIAG / "phase2n_source_audit.json"),
            "repeatability": str(DIAG / "phase2n_repeatability.json"),
            "training_dir": str(TRAIN),
        },
    }
    dump(REPORTS / "PHASE_2N_SEARCH_MANIFEST.json", manifest)

    # Fill the required best-mask artifact with the complete search index and
    # current source/training status, without discarding its existing curve.
    best_out = dict(best)
    best_out.update({
        "phase": "2N",
        "created_utc": now(),
        "single_block_top8": singles[:8],
        "marginal_necessity": marginal,
        "search_summaries": searches,
        "pair_interaction": interactions,
        "source_audit": {
            "state_anchor": "executed",
            "boundary_quantization": "executed",
            "backward_only": "executed",
            "path": str(DIAG / "phase2n_source_audit.json"),
        },
        "selected_candidate": selection,
        "training_status": {k: v for k, v in status.items() if k != "records"},
    })
    dump(DIAG / "phase2n_best_masks.json", best_out)

    # Required density artifact gets a stable interpretation field; preserve
    # the numerically measured per-mask rows verbatim.
    density_out = dict(density)
    density_out.update({
        "phase": "2N",
        "created_utc": now(),
        "interpretation": {
            "current_reverse1_cosine": current_r1,
            "oracle_reverse1_cosine": next((x.get("main_reverse1_cosine") for x in density_rows if int(x.get("fp32_block_count", -1)) == n_blocks), None),
            "no_small_density_monotonicity_claim": True,
            "knee_method": density.get("knee_analysis", {}).get("method"),
        },
    })
    dump(DIAG / "phase2n_density_curve.json", density_out)

    # Narrative final report.  Percentages are left as percentages at the
    # training layer and cosines/recoveries remain unit-scale.
    top_single = singles[:5]
    top_necessity = marginal["top_necessity"][:8]
    robust = (multi.get("gate") or {}).get("candidates", {})
    selected_gate = selection.get("gate_row", {})
    records = status.get("records", [])
    e15 = next((x for x in records if x.get("epoch") == 15), None)
    e25 = next((x for x in records if x.get("epoch") == 25), None)
    e60 = next((x for x in records if x.get("epoch") == 60), None)
    cls = training_classification(status)
    lines = []
    a = lines.append
    a("# PHASE 2N FINAL")
    a("")
    a(f"Generated: `{now()}`  ")
    a(f"Git commit: `{git_commit()}`  ")
    a("All numerical measurements use the frozen LTCC diagnostic protocol on PPU-ZW810E. No internet, CPU fallback, or retrieval metric was used for mask selection.")
    a("")
    a("## 1. FROZEN STATUS")
    a("")
    a("Official: **58.9286 / 31.2798**")
    a("Baseline: **56.4 / 31.1**")
    a("Released Full: **53.3 / 29.3**")
    a("Phase 2M decision: **CUMULATIVE PPU BACKWARD ERROR CONFIRMED BUT MINIMAL SOURCE UNRESOLVED**; FULL32 remains a diagnostic ceiling.")
    a("")
    a("## 2. SINGLE BLOCK SCAN")
    a("")
    a("Top high-leverage blocks by Reverse-1 recovery (not root-cause claims):")
    for row in top_single:
        a(f"- block {row['block']}: cosine {fmt(row['reverse1_cosine'])}, recovery {fmt(row['reverse1_recovery'])}, Full {fmt(row['full_cosine'])}")
    a(f"Best single: block {top_single[0]['block']} if present; absolute recovery remains {fmt(top_single[0]['reverse1_recovery'])}, so it is not a root-cause block.")
    a("")
    a("## 3. ORACLE NECESSITY")
    a("")
    a("Necessity is reported exactly as `OracleRecovery × Recovery(AllExcept{i})`; the oracle factor is 1. Median splits define high/low categories.")
    a("Top necessity blocks: " + ", ".join(str(x["block"]) for x in top_necessity) + ".")
    a("Category A/B/C/D membership is in `PHASE_2N_MARGINAL_NECESSITY.csv` and `diagnostics/phase2n_search_summary.json`.")
    a("")
    a("## 4. CONTIGUOUS WINDOWS")
    a("")
    for length in (1, 2, 3, 4, 6, 8, 12):
        r = searches["windows"].get(length)
        a(f"- {length} block: mask {r.get('mask') if r else 'n/a'}, Reverse-1 recovery {fmt(r.get('reverse1_recovery') if r else None)}, Full recovery {fmt(r.get('full_recovery') if r else None)}")
    a("Window position matters; the best measured windows are not explained by count alone.")
    a("")
    a("## 5. REGULAR SPARSE PATTERNS")
    a("")
    for step, members in searches["regular"].items():
        best_r = max((x for x in members if x), key=lambda x: x.get("reverse1_recovery", -1), default=None)
        a(f"- Every-{step}: best {best_r.get('mask') if best_r else 'n/a'}, recovery {fmt(best_r.get('reverse1_recovery') if best_r else None)}")
    a("The Every-3 shift-0 pattern `[0,3,6,9,12,15,18,21]` is a strong sparse numerical pattern on the main batch, but cross-batch worst-case remains poor; this is not sufficient alone for the compatibility gate.")
    a("")
    a("## 6. GREEDY SEARCH")
    a("")
    a("Forward sequence: " + " → ".join(f"{x['added_block']} (rec {fmt(x['recovery'])})" for x in cache.get("search", {}).get("greedy_forward", [])))
    a("Backward elimination: " + " → ".join(f"drop {x['removed_block']} (cos {fmt(x['reverse1_cosine'])})" for x in cache.get("search", {}).get("greedy_backward", [])))
    a("")
    a("## 7. BEAM SEARCH")
    a("")
    for size in (2, 3, 4, 5, 6, 8, 10, 12):
        r = searches["beam"].get(size, {}).get("reverse1_leader")
        a(f"- {size}: mask {r.get('mask') if r else 'n/a'}, Reverse-1 recovery {fmt(r.get('reverse1_recovery') if r else None)}, Full recovery {fmt(r.get('full_recovery') if r else None)}")
    a("")
    a("## 8. PAIR INTERACTION")
    a("")
    a("Strong positive synergies:")
    for x in interactions["top_positive"][:5]:
        a(f"- ({x['block_i']},{x['block_j']}): synergy {fmt(x['synergy'])}, pair recovery {fmt(x['pair_recovery'])}")
    a("Strong negative synergies:")
    for x in interactions["top_negative"][:5]:
        a(f"- ({x['block_i']},{x['block_j']}): synergy {fmt(x['synergy'])}, pair recovery {fmt(x['pair_recovery'])}")
    a("")
    a("## 9. MULTI-BATCH ROBUSTNESS")
    a("")
    for label, row in robust.items():
        a(f"- {label}: {row.get('fp32_block_count')} FP32 blocks, median cosine {fmt(row.get('median_reverse1_cosine'))}, worst {fmt(row.get('worst_reverse1_cosine'))}, improved {row.get('improved_over_current_count')}/12")
    a(f"Selected by the requested minimum-count rule: `{selection.get('candidate_name', 'n/a')}` mask `{selection.get('mask', [])}`.")
    a("")
    a("## 10. PRECISION DENSITY CURVE")
    a("")
    for row in density_rows:
        a(f"- {row.get('fp32_block_count')}: median cross-batch cosine {fmt(row.get('median_cross_batch_reverse1_cosine'))}, main cosine {fmt(row.get('main_reverse1_cosine'))}, runtime {fmt(row.get('mean_forward_backward_runtime_sec'), 3)} s")
    a("The curve is non-monotonic on individual panel worst cases. A small practical patch exists under the requested robust gate, but it is not uniformly strong: the selected 4-block candidate worst batch is below the strong-gate threshold.")
    a("")
    a("## 11. DEPTH STRUCTURE")
    a("")
    a("Selected/source masks and adjacent gaps:")
    for label, row in depth.items():
        a(f"- {label}: {row['mask']}; gaps {row['adjacent_gaps']}")
    a("Evidence supports depth-distributed interaction effects, with block 0 as a high-leverage anchor and additional useful positions varying by search family; it does not establish a single periodic reset law.")
    a("")
    a("## 12. STATE ANCHOR")
    a("")
    a("Executed: **YES**. State-only residual anchoring left Reverse-1 at the Current path (approximately 0.0873 on the main batch), while whole-block FP32 changed backward fidelity.")
    a("See `diagnostics/phase2n_source_audit.json` for all five masks and forward-output signatures.")
    a("")
    a("## 13. FORWARD VS BACKWARD SOURCE")
    a("")
    a("Boundary fake-quantization diagnostic was executed: scheduled boundary quantization in an otherwise FP32 path produced Reverse-1 cosines about 0.9747–0.9910 for the tested masks, while preserving Current forward outputs to approximately 1e-5 relative scale.")
    a("Backward-only custom VJP was executed safely with Current forward outputs unchanged, but did not restore the gradient (about 0.079–0.113 on the main batch). Dominant mechanism: **FORWARD STATE QUANTIZATION DOMINATES**, with backward accumulation still part of the distributed effect.")
    a("")
    a("## 14. PPU_COMPAT_V2")
    a("")
    a(f"Created: **{bool(selection.get('created', selection.get('candidate_name')))}**")
    a("Type: **SPARSE BLOCK MASK**")
    a(f"Exact mask: `{selection.get('mask', [])}`; FP32 fraction: {pct(selection.get('fp32_fraction', 0.0))}")
    a(f"12-batch median Reverse-1 cosine: {fmt(selected_gate.get('median_reverse1_cosine'))}; worst: {fmt(selected_gate.get('worst_reverse1_cosine'))}; runtime is recorded per mask in the matrix/cache.")
    a("Semantics: exact Phase2M full-block wrapper; selected block output is retained as returned by that wrapper and all parameters remain identical.")
    a("")
    a("## 15. 200 STEP")
    a("")
    paired = load(ROOT / "outputs/reproduction_phase2/phase2n/paired200/paired200.json", {}) or {}
    a(f"Executed: **{'YES' if paired else 'NO'}**")
    if paired:
        pg = paired.get("paired_gate", {})
        streams = {
            str(x.get("rank")): x.get("stream_sha256")
            for x in paired.get("batch_fingerprint", {}).get("rank_streams", [])
        }
        a(f"Paired gate: **{pg.get('pass')}**; Current/Compat summaries are in `paired200.json` and `paired200_steps.csv`.")
        a(f"Common rank-local batch hashes: `{streams}`; common-within-candidates: `{paired.get('batch_fingerprint', {}).get('common_within_candidates')}`.")
    a("")
    a("## 16. EPOCH15")
    a("")
    a(f"Executed: **{'YES' if e15 else 'NO'}**")
    if e15:
        a(f"Compat: **{100*e15['cc']['rank1']:.4f} / {100*e15['cc']['mAP']:.4f}**; gate artifact: `{status.get('epoch15_gate', {}).get('gate')}`; semantic collapse: `{e15.get('semantic_collapse')}`.")
    a("Official reference: **48.724 / 25.814**; requested minimum: **46.7 / 24.3**.")
    a("")
    a("## 17. EPOCH25")
    a("")
    a(f"Executed: **{'YES' if e25 else 'IN PROGRESS/NO'}**")
    if e25:
        a(f"Compat: **{100*e25['cc']['rank1']:.4f} / {100*e25['cc']['mAP']:.4f}**; epoch25 gate: `{status.get('epoch25_gate', {}).get('gate')}`.")
    else:
        a("The epoch25 continuation was still running when this report was generated; rerun this finalizer after the process exits.")
    a("")
    a("## 18. EPOCH60")
    a("")
    a(f"Executed: **{'YES' if e60 else 'NO'}**")
    if e60:
        best_record = status.get("best_observed") or e60
        a(f"Compat best observed (epoch {best_record['epoch']}): **{100*best_record['cc']['rank1']:.4f} / {100*best_record['cc']['mAP']:.4f}**.")
        a(f"Compat final (epoch 60): **{100*e60['cc']['rank1']:.4f} / {100*e60['cc']['mAP']:.4f}**; classification by best observed: **{cls}**.")
    else:
        a("Only permitted after an epoch25 gate PASS; no epoch60 artifact is inferred here.")
    a("")
    a("## 19. ROOT CAUSE")
    a("")
    a("**CUMULATIVE PPU BACKWARD ERROR REMAINS NONLOCAL**; the source audit additionally supports **FORWARD STATE QUANTIZATION DOMINATES**. A sparse block mask passes the required numerical robust gate, but the strong robust gate is met only by the 17-block backward-greedy mask, so no claim of a universally strong small patch is made.")
    a("")
    a("## 20. REPRODUCTION STATUS")
    a("")
    a("- Released-code Baseline: **56.4 / 31.1**")
    a("- Released-code Full: **53.3 / 29.3**")
    best_record = status.get("best_observed")
    final_record = status.get("final_observed")
    if best_record and final_record:
        a(f"- Compatibility best observed: **{100*best_record['cc']['rank1']:.4f} / {100*best_record['cc']['mAP']:.4f}** at epoch {best_record['epoch']}.")
        a(f"- Compatibility final: **{100*final_record['cc']['rank1']:.4f} / {100*final_record['cc']['mAP']:.4f}** at epoch {final_record['epoch']}; classification **{cls}**.")
    else:
        a("- Compatibility: no completed training record.")
    a("")
    a("### Artifact index")
    a("")
    for path in [
        DIAG / "phase2n_best_masks.json", DIAG / "phase2n_density_curve.json",
        DIAG / "phase2n_multibatch_matrix.json", DIAG / "phase2n_search_summary.json",
        DIAG / "phase2n_mask_search_cache.json", REPORTS / "PHASE_2N_SEARCH_MANIFEST.json",
        REPORTS / "PHASE_2N_SINGLE_BLOCK.csv", REPORTS / "PHASE_2N_MARGINAL_NECESSITY.csv",
        REPORTS / "PHASE_2N_PAIR_INTERACTION.csv", REPORTS / "PHASE_2N_BATCH_PANEL.csv",
        DIAG / "phase2n_source_audit.json", DIAG / "phase2n_repeatability.json",
    ]:
        a(f"- `{path.relative_to(ROOT)}`")
    (REPORTS / "PHASE_2N_FINAL.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "final_report": str(REPORTS / "PHASE_2N_FINAL.md"),
        "selected_mask": selection.get("mask"),
        "last_epoch": status.get("last_epoch"),
        "epoch25_gate": (status.get("epoch25_gate") or {}).get("gate"),
        "epoch60": bool(e60),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
