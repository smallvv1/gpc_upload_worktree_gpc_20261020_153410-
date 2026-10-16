#!/usr/local/bin/python3
"""PHASE 2N sparse FP32-block mask search (diagnostic only).

The evaluator reuses the exact Phase 2M full-block FP32 wrapper and compares
each mask against one frozen full-FP32 reference on the Phase 2M main LTCC
batch.  Results are cached after every mask so an interrupted search can be
resumed safely.  No optimizer or scheduler step is made.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from global_precision_policy_screen_phase2m import (  # noqa: E402
    DEVICE,
    SOURCES,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    component_losses,
    make_config,
    restore_rng_state,
    seed_all,
    stat,
    wrap_fp32_block,
)
from loss.triplet_loss import TripletLoss  # noqa: E402


N_BLOCKS = 23
TARGETS = ("block_0", "block_2", "block_5", "block_10", "block_11", "block_22")
EPS = 1e-12
CACHE_DIR = ROOT / "cache/phase2n"
CACHE_PATH = CACHE_DIR / "phase2n_mask_cache.json"
DIAG_DIR = ROOT / "diagnostics"
REPORT_DIR = ROOT / "reports"


def compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    x = a.detach().double().cpu().reshape(-1)
    y = b.detach().double().cpu().reshape(-1)
    if x.numel() != y.numel():
        return {"missing": False, "shape_mismatch": True}
    d = x - y
    xn = float(torch.linalg.vector_norm(x).item())
    yn = float(torch.linalg.vector_norm(y).item())
    return {
        "missing": False,
        "shape_mismatch": False,
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn > EPS and yn > EPS else None,
        "relative_l2": float(torch.linalg.vector_norm(d).item()) / (yn + EPS),
        "norm_ratio": xn / (yn + EPS),
        "max_abs": float(d.abs().max().item()) if d.numel() else 0.0,
        "a_l2": xn,
        "b_l2": yn,
        "a_finite": bool(torch.isfinite(x).all().item()),
        "b_finite": bool(torch.isfinite(y).all().item()),
    }


def mask_key(mask: Iterable[int]) -> str:
    return ",".join(str(int(i)) for i in sorted(set(mask)))


def mask_indices(key: str) -> list[int]:
    return [] if not key else [int(x) for x in key.split(",") if x]


def stable_batch_hash(batch: dict[str, Any]) -> str:
    parts = []
    for key in ("pid", "camid", "clothes_id"):
        parts.append(key + ":" + ",".join(str(int(v)) for v in batch[key].detach().cpu().tolist()))
    parts.append("image:" + hashlib.sha256(batch["image"].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def install_hooks(model: torch.nn.Module):
    saved: dict[str, torch.Tensor | None] = {name: None for name in TARGETS}
    handles = []
    for index in (0, 2, 5, 10, 11, 22):
        name = f"block_{index}"

        def hook(_module, inputs, _output, name=name):
            value = inputs[0] if inputs else None
            if isinstance(value, torch.Tensor):
                value.retain_grad()
                saved[name] = value

        handles.append(model.blocks[index].register_forward_hook(hook))
    return saved, handles


def apply_sparse(model: torch.nn.Module, mask: tuple[int, ...]):
    originals = []
    for index in mask:
        block = model.blocks[index]
        originals.append((block, wrap_fp32_block(block)))
    return originals


def restore(originals):
    for module, original in reversed(originals):
        module.forward = original


def run_once(model, base_state, batch, rng_state, mask: tuple[int, ...], fp32_all: bool = False):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    originals = [] if fp32_all else apply_sparse(model, mask)
    saved, handles = install_hooks(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    try:
        if fp32_all:
            with amp.autocast(enabled=False):
                score, feat = model(batch["image"], cam_label=batch["camid"])
        else:
            with amp.autocast(enabled=True):
                score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, TripletLoss())
        gradients: dict[str, dict[str, torch.Tensor | None]] = {}
        targets = [saved[name] for name in TARGETS]
        for source in SOURCES:
            values = torch.autograd.grad(losses[source], targets, retain_graph=True, allow_unused=True)
            gradients[source] = {name: (value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None) for name, value in zip(TARGETS, values)}
        elapsed = time.perf_counter() - start
        try:
            peak = int(torch.cuda.max_memory_allocated(DEVICE))
        except Exception:
            peak = None
        return {
            "mask": list(mask),
            "fp32_all": bool(fp32_all),
            "precision": "full_fp32_reference" if fp32_all else "selected_blocks_fp32_with_formal_autocast_elsewhere",
            "losses": {
                key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value.detach()).item())}
                for key, value in losses.items()
            },
            "gradients": gradients,
            "runtime_sec": elapsed,
            "peak_memory_bytes": peak,
        }
    finally:
        for handle in handles:
            handle.remove()
        restore(originals)
        for name in ("score", "feat", "losses", "saved"):
            if name in locals():
                del locals()[name]
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def metric_result(run: dict[str, Any], reference: dict[str, Any], current: dict[str, Any], mask: tuple[int, ...]) -> dict[str, Any]:
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for source in SOURCES:
        comparisons[source] = {}
        for target in TARGETS:
            comparisons[source][target] = compare(run["gradients"][source][target], reference["gradients"][source][target])
    ccur = compare(current["gradients"]["reverse1"]["block_0"], reference["gradients"]["reverse1"]["block_0"]).get("cosine")
    rcur = compare(current["gradients"]["full"]["block_0"], reference["gradients"]["full"]["block_0"]).get("cosine")
    cr = comparisons["reverse1"]["block_0"].get("cosine")
    fr = comparisons["full"]["block_0"].get("cosine")
    return {
        "mask": list(mask),
        "mask_key": mask_key(mask),
        "fp32_block_count": len(mask),
        "fp16_block_count": N_BLOCKS - len(mask),
        "runtime_sec": run["runtime_sec"],
        "peak_memory_bytes": run["peak_memory_bytes"],
        "losses": run["losses"],
        "comparisons": comparisons,
        "summary": {
            "reverse1_block0_cosine": cr,
            "reverse1_block0_relative_l2": comparisons["reverse1"]["block_0"].get("relative_l2"),
            "reverse1_block0_recovery": None if cr is None or ccur is None else (cr - ccur) / (1.0 - ccur),
            "full_block0_cosine": fr,
            "full_block0_relative_l2": comparisons["full"]["block_0"].get("relative_l2"),
            "full_block0_recovery": None if fr is None or rcur is None else (fr - rcur) / (1.0 - rcur),
            "branch_block0_cosines": {source: comparisons[source]["block_0"].get("cosine") for source in SOURCES},
        },
    }


def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {"phase": "2N", "results": {}, "search": {}, "notes": []}


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    tmp.replace(CACHE_PATH)


def evaluate(model, state, batch, rng, key: str, cache: dict[str, Any], reference, current, verbose: bool = True):
    if key in cache["results"]:
        return cache["results"][key]
    mask = tuple(mask_indices(key))
    print(f"PHASE2N_MASK {key or 'empty'}", flush=True)
    run = run_once(model, state, batch, rng, mask)
    result = metric_result(run, reference, current, mask)
    cache["results"][key] = result
    save_cache(cache)
    if verbose:
        print("PHASE2N_MASK_DONE", json.dumps({"mask": key, **result["summary"], "runtime_sec": result["runtime_sec"]}, sort_keys=True), flush=True)
    return result


def recovery(result: dict[str, Any]) -> float:
    value = result.get("summary", {}).get("reverse1_block0_recovery")
    return float(value) if value is not None and math.isfinite(value) else -1e9


def objective(result: dict[str, Any]) -> float:
    s = result["summary"]
    vals = [s.get("reverse1_block0_recovery"), s.get("full_block0_recovery"), s.get("branch_block0_cosines", {}).get("bio"), s.get("branch_block0_cosines", {}).get("reverse0")]
    vals = [0.0 if v is None or not math.isfinite(v) else float(v) for v in vals]
    return 0.60 * vals[0] + 0.20 * vals[1] + 0.10 * vals[2] + 0.10 * vals[3]


def all_except(i: int) -> tuple[int, ...]:
    return tuple(j for j in range(N_BLOCKS) if j != i)


def windows(length: int):
    for start in range(0, N_BLOCKS - length + 1):
        yield tuple(range(start, start + length))


def regular(step: int, shift: int):
    return tuple(i for i in range(N_BLOCKS) if (i - shift) % step == 0)


def main() -> None:
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, _sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    cache = load_cache()
    cache.setdefault("meta", {})
    cache["meta"].update({
        "dataset": "ltcc", "seed": 1234, "batch_size": int(cfg.DATA.BATCH_SIZE), "n_blocks": N_BLOCKS,
        "target_boundaries": list(TARGETS), "batch_hash": stable_batch_hash(batch),
        "reverse_indexing": "[:, i]", "nobio_index": [1, 2], "optimizer_step": False, "scheduler_step": False,
        "scope": "selected transformer blocks use exact Phase2M wrap_fp32_block; head_bio remains formal path",
    })
    save_cache(cache)

    # Full mathematical reference and current formal baseline contain CPU
    # gradient tensors and therefore are intentionally kept process-local.
    # If the search is resumed, these two fixed runs are recomputed while all
    # serializable mask results remain cached.
    print("PHASE2N_REFERENCE", flush=True)
    reference = run_once(model, base_state, batch, rng_state, tuple(), fp32_all=True)
    print("PHASE2N_CURRENT", flush=True)
    current = run_once(model, base_state, batch, rng_state, tuple(), fp32_all=False)

    # Ordered scan bookkeeping.  Every requested mask is evaluated through
    # the cache before the next search family starts.
    scans: dict[str, Any] = cache.setdefault("search", {})
    single = []
    for i in range(N_BLOCKS):
        key = mask_key((i,))
        single.append(evaluate(model, base_state, batch, rng_state, key, cache, reference, current))
    scans["single_block"] = [r["mask"] for r in single]
    save_cache(cache)

    loo = []
    for i in range(N_BLOCKS):
        key = mask_key(all_except(i))
        loo.append(evaluate(model, base_state, batch, rng_state, key, cache, reference, current))
    scans["leave_one_out"] = [r["mask"] for r in loo]
    save_cache(cache)

    window_results: dict[str, list[dict[str, Any]]] = {}
    for length in (1, 2, 3, 4, 6, 8, 12):
        vals = []
        for mask in windows(length):
            vals.append(evaluate(model, base_state, batch, rng_state, mask_key(mask), cache, reference, current))
        window_results[str(length)] = vals
    scans["windows"] = {length: [r["mask"] for r in vals] for length, vals in window_results.items()}
    save_cache(cache)

    regular_results: dict[str, list[dict[str, Any]]] = {}
    for step in (2, 3, 4, 6):
        vals = []
        for shift in range(step):
            vals.append(evaluate(model, base_state, batch, rng_state, mask_key(regular(step, shift)), cache, reference, current))
        regular_results[str(step)] = vals
    scans["regular"] = {step: [r["mask"] for r in vals] for step, vals in regular_results.items()}
    save_cache(cache)

    # Greedy forward: one candidate per unselected block at each iteration.
    forward_masks = []
    selected: set[int] = set()
    forward_sequence = []
    for step in range(1, N_BLOCKS + 1):
        trials = []
        for i in range(N_BLOCKS):
            if i in selected:
                continue
            mask = tuple(sorted(selected | {i}))
            trials.append((objective(evaluate(model, base_state, batch, rng_state, mask_key(mask), cache, reference, current)), i, mask))
        if not trials:
            break
        _, chosen, chosen_mask = max(trials, key=lambda x: x[0])
        selected.add(chosen)
        chosen_result = cache["results"][mask_key(chosen_mask)]
        forward_sequence.append({"step": step, "added_block": chosen, "mask": list(chosen_mask), "objective": objective(chosen_result), "recovery": recovery(chosen_result)})
        if recovery(chosen_result) >= 0.95 and chosen_result["summary"].get("full_block0_cosine", -1) >= 0.8:
            break
    scans["greedy_forward"] = forward_sequence
    save_cache(cache)

    # Greedy backward: start at all FP32, remove the least damaging block at
    # each step.  Continue six removals after first crossing reverse cosine
    # below .90, as specified, to expose whether a cliff exists.
    backward_sequence = []
    selected = set(range(N_BLOCKS))
    below90_seen = False
    extra_after_below = 0
    for step in range(N_BLOCKS):
        if not selected:
            break
        trials = []
        for i in sorted(selected):
            mask = tuple(sorted(selected - {i}))
            result = evaluate(model, base_state, batch, rng_state, mask_key(mask), cache, reference, current)
            trials.append((objective(result), i, mask, result))
        # Removing the block with the highest remaining objective causes the
        # smallest fidelity damage.
        _, removed, new_mask, result = max(trials, key=lambda x: x[0])
        selected.remove(removed)
        c = result["summary"].get("reverse1_block0_cosine")
        if c is not None and c < 0.90:
            below90_seen = True
        if below90_seen:
            extra_after_below += 1
            if extra_after_below >= 6:
                backward_sequence.append({"step": step + 1, "removed_block": removed, "mask": list(new_mask), "objective": objective(result), "recovery": recovery(result), "reverse1_cosine": c})
                break
        backward_sequence.append({"step": step + 1, "removed_block": removed, "mask": list(new_mask), "objective": objective(result), "recovery": recovery(result), "reverse1_cosine": c})
    scans["greedy_backward"] = backward_sequence
    save_cache(cache)

    # Candidate pool: top single/LOO/window/regular/greedy masks.  Limit to
    # twelve blocks for the beam as required, with deterministic tie breaks.
    pool_scores: dict[int, float] = {}
    for result in single:
        i = result["mask"][0]
        pool_scores[i] = max(pool_scores.get(i, -1e9), objective(result))
    for result in loo:
        for i in result["mask"]:
            pool_scores[i] = max(pool_scores.get(i, -1e9), objective(result))
    for vals in window_results.values():
        for result in vals:
            for i in result["mask"]:
                pool_scores[i] = max(pool_scores.get(i, -1e9), objective(result))
    for vals in regular_results.values():
        for result in vals:
            for i in result["mask"]:
                pool_scores[i] = max(pool_scores.get(i, -1e9), objective(result))
    for item in forward_sequence:
        for i in item["mask"]:
            pool_scores[i] = max(pool_scores.get(i, -1e9), item["objective"])
    for item in backward_sequence:
        for i in item["mask"]:
            pool_scores[i] = max(pool_scores.get(i, -1e9), item["objective"])
    pool = tuple(sorted(pool_scores, key=lambda i: (-pool_scores[i], i))[:12])
    scans["beam_pool"] = {"blocks": list(pool), "scores": {str(i): pool_scores[i] for i in pool}}
    save_cache(cache)

    # Beam search over the deterministic candidate pool.  Keep the best eight
    # masks at each requested size; all evaluated masks remain in cache.
    beam = [tuple()]
    beam_summary = {}
    for target_size in (2, 3, 4, 5, 6, 8, 10, 12):
        while beam and len(beam[0]) < target_size:
            expanded = set()
            for base in beam:
                for i in pool:
                    if i not in base:
                        expanded.add(tuple(sorted(base + (i,))))
            scored = []
            for mask in sorted(expanded):
                result = evaluate(model, base_state, batch, rng_state, mask_key(mask), cache, reference, current)
                scored.append((objective(result), recovery(result), mask))
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
            beam = [mask for _, _, mask in scored[:8]]
            if not beam:
                break
        entries = []
        for mask in beam:
            result = cache["results"][mask_key(mask)]
            entries.append({"mask": list(mask), "objective": objective(result), "recovery": recovery(result), "reverse1_cosine": result["summary"].get("reverse1_block0_cosine")})
        beam_summary[str(target_size)] = entries
    scans["beam"] = {"pool": list(pool), "width": 8, "best_by_size": beam_summary}
    save_cache(cache)

    # Pairwise interaction matrix uses the union-selected pool, capped at 12.
    interaction_blocks = list(pool)
    interaction_rows = []
    for pos, i in enumerate(interaction_blocks):
        ri = cache["results"].get(mask_key((i,))) or evaluate(model, base_state, batch, rng_state, mask_key((i,)), cache, reference, current)
        for j in interaction_blocks[pos + 1:]:
            rj = cache["results"].get(mask_key((j,))) or evaluate(model, base_state, batch, rng_state, mask_key((j,)), cache, reference, current)
            rij = evaluate(model, base_state, batch, rng_state, mask_key((i, j)), cache, reference, current)
            interaction_rows.append({"block_i": i, "block_j": j, "single_i_recovery": recovery(ri), "single_j_recovery": recovery(rj), "pair_recovery": recovery(rij), "synergy": recovery(rij) - recovery(ri) - recovery(rj), "pair_objective": objective(rij)})
    scans["interaction_blocks"] = interaction_blocks
    scans["interaction_count"] = len(interaction_rows)
    save_cache(cache)
    interaction_path = REPORT_DIR / "PHASE_2N_PAIR_INTERACTION.csv"
    interaction_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["block_i", "block_j", "single_i_recovery", "single_j_recovery", "pair_recovery", "synergy", "pair_objective"]
    with interaction_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(interaction_rows)

    # Flat scan CSV for later report/analysis.
    scan_path = REPORT_DIR / "PHASE_2N_MASK_SEARCH.csv"
    rows = []
    for key, result in sorted(cache["results"].items(), key=lambda kv: (len(kv[0].split(",")) if kv[0] else 0, kv[0])):
        rows.append({"mask": key, "fp32_blocks": len(result["mask"]), "reverse1_cosine": result["summary"].get("reverse1_block0_cosine"), "reverse1_rel_l2": result["summary"].get("reverse1_block0_relative_l2"), "reverse1_recovery": recovery(result), "full_cosine": result["summary"].get("full_block0_cosine"), "full_recovery": result["summary"].get("full_block0_recovery"), "runtime_sec": result.get("runtime_sec"), "peak_memory_bytes": result.get("peak_memory_bytes"), "objective": objective(result)})
    with scan_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["mask"])
        writer.writeheader()
        writer.writerows(rows)
    cache["search"]["flat_csv"] = str(scan_path)
    cache["search"]["interaction_csv"] = str(interaction_path)
    cache["search"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_cache(cache)

    summary = {
        "cache": str(CACHE_PATH), "mask_count": len(cache["results"]), "single_top": sorted(({"mask": r["mask"], "recovery": recovery(r), "cosine": r["summary"].get("reverse1_block0_cosine")} for r in single), key=lambda x: -x["recovery"])[:5],
        "leave_one_out_top": sorted(({"mask": r["mask"], "recovery": recovery(r), "cosine": r["summary"].get("reverse1_block0_cosine")} for r in loo), key=lambda x: -x["recovery"])[:5],
        "beam": beam_summary,
        "pool": list(pool),
        "interaction_count": len(interaction_rows),
    }
    out = DIAG_DIR / "phase2n_search_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("PHASE2N_SEARCH_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)
    del model, dataset, loader, batch, cache
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
