#!/usr/local/bin/python3
"""PHASE 2N-J through 2N-L numerical characterization.

This driver is diagnostic-only.  It resumes the Phase 2N mask cache created by
``sparse_mask_search_phase2n.py`` and never performs an optimizer or scheduler
step.  The batch panel is frozen from deterministic metadata and current/oracle
diagnostics before any sparse candidate is measured on that panel.

The model remains on PPU-ZW810E.  Gradient vectors are copied to CPU only for
double-precision comparison and compact JSON serialization.
"""
from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sparse_mask_search_phase2n import (  # noqa: E402
    DEVICE,
    N_BLOCKS,
    SOURCES,
    TARGETS,
    apply_sparse,
    compare,
    evaluate,
    load_cache,
    mask_key,
    metric_result,
    objective,
    recovery,
    restore,
    run_once,
    save_cache,
    stable_batch_hash,
)
from initialization_footprint_phase2i import (  # noqa: E402
    batch_fingerprint,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from loss.triplet_loss import TripletLoss  # noqa: E402


CACHE_PATH = ROOT / "cache/phase2n/phase2n_mask_cache.json"
DIAG_DIR = ROOT / "diagnostics"
REPORT_DIR = ROOT / "reports"
PANEL_JSON = DIAG_DIR / "phase2n_multibatch_matrix.json"
PANEL_CSV = REPORT_DIR / "PHASE_2N_BATCH_PANEL.csv"
SEARCH_CACHE_EXPORT = DIAG_DIR / "phase2n_mask_search_cache.json"
DENSITY_JSON = DIAG_DIR / "phase2n_density_curve.json"
BEST_JSON = DIAG_DIR / "phase2n_best_masks.json"
SUMMARY_JSON = DIAG_DIR / "phase2n_search_summary.json"
PANEL_POOL_COUNT = 24
PANEL_COUNT = 12


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def tensor_l2(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_manifest(model: torch.nn.Module) -> dict[str, Any]:
    params = list(model.named_parameters())
    state = model.state_dict()
    return {
        "parameter_names_sha256": hashlib.sha256("\n".join(name for name, _ in params).encode()).hexdigest(),
        "parameter_count": len(params),
        "state_key_count": len(state),
        "parameter_numel": int(sum(int(p.numel()) for _, p in params)),
        "state_hashes_sha256": hashlib.sha256(
            "\n".join(f"{name}:{sha_tensor(value)}" for name, value in sorted(state.items())).encode()
        ).hexdigest(),
        "selected_state_hashes": {
            name: sha_tensor(state[name])
            for name in (
                "patch_embed.proj.weight",
                "blocks.0.attn.q_proj.weight",
                "blocks.11.attn.q_proj.weight",
                "blocks.22.attn.q_proj.weight",
                "head.weight",
                "head_clip_bio.weight",
                "head_clip_bioReverse.0.head_fc.weight",
                "head_clip_bioReverse.1.head_fc.weight",
            )
            if name in state
        },
    }


def clone_batch(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in raw.items()}


def device_batch(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def metadata_signature(raw: dict[str, Any], key: str) -> str:
    values = raw[key].detach().cpu().tolist()
    return ",".join(str(int(value)) for value in sorted(set(values)))


def batch_descriptor(raw: dict[str, Any], current: dict[str, Any], oracle: dict[str, Any], index: int) -> dict[str, Any]:
    current_reverse = current["losses"]["reverse1"]["value"]
    current_grad = current["gradients"]["reverse1"]["block_0"]
    oracle_grad = oracle["gradients"]["reverse1"]["block_0"]
    return {
        "pool_index": index,
        "batch_hash": stable_batch_hash(raw),
        "sample_sha256": batch_fingerprint(raw)["sample_sha256"],
        "image_sha256": batch_fingerprint(raw)["image_sha256"],
        "pid_signature": metadata_signature(raw, "pid"),
        "camera_signature": metadata_signature(raw, "camid"),
        "clothes_signature": metadata_signature(raw, "clothes_id"),
        "pids": [int(x) for x in raw["pid"].tolist()],
        "camids": [int(x) for x in raw["camid"].tolist()],
        "clothes": [int(x) for x in raw["clothes_id"].tolist()],
        "current_reverse1_loss": float(current_reverse),
        "current_reverse1_grad_norm": tensor_l2(current_grad),
        "oracle_reverse1_grad_norm": tensor_l2(oracle_grad),
        "current_full_loss": float(current["losses"]["full"]["value"]),
    }


def quantile_indices(items: list[int], values: list[float], fractions: list[float]) -> list[int]:
    order = sorted(items, key=lambda i: (finite_float(values[i], 0.0), i))
    if not order:
        return []
    result = []
    for fraction in fractions:
        pos = min(len(order) - 1, max(0, int(round(fraction * (len(order) - 1)))))
        result.append(order[pos])
    return result


def freeze_panel(descriptors: list[dict[str, Any]]) -> tuple[list[int], dict[int, str]]:
    """Select 12 batches deterministically without candidate observations.

    Quantiles of current loss/norm establish numerical range.  The fill pass
    maximizes unseen PID/camera/clothes signatures, then uses a stable distance
    tie-break.  Candidate results are not consulted anywhere in this function.
    """
    all_indices = list(range(len(descriptors)))
    losses = [item["current_reverse1_loss"] for item in descriptors]
    norms = [finite_float(item["current_reverse1_grad_norm"], 0.0) for item in descriptors]
    selected: list[int] = []
    reasons: dict[int, str] = {}

    for index in quantile_indices(all_indices, losses, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        if index not in selected:
            selected.append(index)
            reasons[index] = "reverse1_loss_quantile"
    for index in quantile_indices(all_indices, norms, [0.0, 0.5, 1.0]):
        if index not in selected:
            selected.append(index)
            reasons[index] = "reverse1_grad_norm_quantile"

    while len(selected) < PANEL_COUNT:
        selected_set = set(selected)
        seen_pid = {descriptors[i]["pid_signature"] for i in selected}
        seen_cam = {descriptors[i]["camera_signature"] for i in selected}
        seen_clothes = {descriptors[i]["clothes_signature"] for i in selected}

        def fill_score(index: int) -> tuple[float, float, int]:
            if index in selected_set:
                return (-1e9, -1e9, index)
            item = descriptors[index]
            novel = (
                int(item["pid_signature"] not in seen_pid)
                + int(item["camera_signature"] not in seen_cam)
                + int(item["clothes_signature"] not in seen_clothes)
            )
            # Stable numerical spread, based only on current diagnostics.
            spread = min(
                abs(item["current_reverse1_loss"] - descriptors[j]["current_reverse1_loss"])
                + abs(norms[index] - norms[j]) / (max(norms) + 1e-12)
                for j in selected
            ) if selected else 0.0
            return (float(novel), float(spread), -index)

        chosen = max(all_indices, key=fill_score)
        selected.append(chosen)
        reasons[chosen] = "metadata_coverage_and_stable_spread"
    return selected[:PANEL_COUNT], reasons


def compact_current_oracle(current: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        source: {
            target: compare(current["gradients"][source][target], oracle["gradients"][source][target])
            for target in TARGETS
        }
        for source in SOURCES
    }
    cmp = comparisons["reverse1"]["block_0"]
    return {
        "current_reverse1_cosine": cmp.get("cosine"),
        "current_reverse1_relative_l2": cmp.get("relative_l2"),
        "current_reverse1_norm_ratio": cmp.get("norm_ratio"),
        "current_reverse1_max_abs": cmp.get("max_abs"),
        "current_reverse1_grad_norm": tensor_l2(current["gradients"]["reverse1"]["block_0"]),
        "oracle_reverse1_grad_norm": tensor_l2(oracle["gradients"]["reverse1"]["block_0"]),
        "oracle_reverse1_cosine": 1.0,
        "oracle_reverse1_relative_l2": 0.0,
        "current_loss": current["losses"]["reverse1"]["value"],
        "oracle_loss": oracle["losses"]["reverse1"]["value"],
        "current_full_cosine": comparisons["full"]["block_0"].get("cosine"),
        "current_branch_block0_cosines": {
            source: comparisons[source]["block_0"].get("cosine") for source in SOURCES
        },
        "current_target_cosines": {
            source: {target: comparisons[source][target].get("cosine") for target in TARGETS}
            for source in SOURCES
        },
    }


def get_batch_result(cache: dict[str, Any], model: torch.nn.Module, state: dict[str, torch.Tensor], raw: dict[str, Any], rng_state: dict[str, Any], key: str, reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    batch_hash = stable_batch_hash(raw)
    batches = cache.setdefault("batch_results", {}).setdefault(batch_hash, {})
    if key in batches:
        return batches[key]
    # Do not call the main-batch ``evaluate`` helper here: its key space is
    # intentionally one-dimensional (mask only).  Batch-panel results have a
    # separate namespace so they can never overwrite the frozen main-batch
    # result for the same mask.
    mask = tuple(int(value) for value in key.split(",") if value)
    run = run_once(model, state, device_batch(raw), rng_state, mask)
    result = metric_result(run, reference, current, mask)
    batches[key] = result
    # The primary evaluator cache is written after every new batch/mask.
    save_cache(cache)
    return result


def result_row(result: dict[str, Any], batch_index: int, batch_hash: str, candidate: str) -> dict[str, Any]:
    summary = result["summary"]
    branch = summary.get("branch_block0_cosines", {})
    return {
        "candidate": candidate,
        "mask": result["mask"],
        "mask_key": result["mask_key"],
        "batch_index": batch_index,
        "batch_hash": batch_hash,
        "fp32_block_count": result["fp32_block_count"],
        "reverse1_cosine": summary.get("reverse1_block0_cosine"),
        "reverse1_recovery": summary.get("reverse1_block0_recovery"),
        "reverse1_relative_l2": summary.get("reverse1_block0_relative_l2"),
        "full_cosine": summary.get("full_block0_cosine"),
        "full_recovery": summary.get("full_block0_recovery"),
        "ce_cosine": branch.get("ce"),
        "triplet_cosine": branch.get("triplet"),
        "bio_cosine": branch.get("bio"),
        "reverse0_cosine": branch.get("reverse0"),
        "target_cosines": {
            source: {
                target: result["comparisons"][source][target].get("cosine")
                for target in TARGETS
            }
            for source in SOURCES
        },
        "runtime_sec": result.get("runtime_sec"),
        "peak_memory_bytes": result.get("peak_memory_bytes"),
        "objective": objective(result),
    }


def choose_source_masks(cache: dict[str, Any]) -> dict[str, list[int]]:
    results = cache["results"]
    search = cache.get("search", {})
    choices: dict[str, list[int]] = {}

    forward = search.get("greedy_forward", [])
    if forward:
        choices["best_forward_greedy"] = list(max(forward, key=lambda x: (finite_float(x.get("objective"), -1e9), -len(x["mask"]), tuple(x["mask"])))['mask'])

    backward = search.get("greedy_backward", [])
    if backward:
        eligible = [x for x in backward if finite_float(x.get("reverse1_cosine"), -1.0) >= 0.90]
        choices["best_backward_greedy"] = list(min(eligible or backward, key=lambda x: (len(x["mask"]), -finite_float(x.get("objective"), -1e9), tuple(x["mask"])))['mask'])

    beam = search.get("beam", {}).get("best_by_size", {})
    beam_entries = [entry for entries in beam.values() for entry in entries]
    if beam_entries:
        choices["best_beam"] = list(min(beam_entries, key=lambda x: (-finite_float(x.get("objective"), -1e9), len(x["mask"]), tuple(x["mask"])))['mask'])

    regular_masks = [mask for masks in search.get("regular", {}).values() for mask in masks]
    regular_results = [results.get(mask_key(mask)) for mask in regular_masks if mask_key(mask) in results]
    if regular_results:
        choices["best_regular_sparse"] = list(min(regular_results, key=lambda x: (-objective(x), len(x["mask"]), tuple(x["mask"])))['mask'])

    window_masks = [mask for masks in search.get("windows", {}).values() for mask in masks]
    window_results = [results.get(mask_key(mask)) for mask in window_masks if mask_key(mask) in results]
    if window_results:
        choices["best_contiguous_window"] = list(min(window_results, key=lambda x: (-objective(x), len(x["mask"]), tuple(x["mask"])))['mask'])

    # Deterministic de-duplication is intentional: a source family can point
    # to the same mask without inflating the five-candidate panel.
    unique: dict[str, list[int]] = {}
    for label, mask in choices.items():
        unique.setdefault(mask_key(mask), mask)
    return {label: unique[mask_key(mask)] for label, mask in choices.items() if mask_key(mask) in unique}


def ensure_density_masks(cache: dict[str, Any], model: torch.nn.Module, state: dict[str, torch.Tensor], batch: dict[str, Any], rng_state: dict[str, Any], reference: dict[str, Any], current: dict[str, Any]) -> None:
    """Extend the cached deterministic backward chain for missing density k."""
    search = cache.setdefault("search", {})
    continuation = search.setdefault("density_backward_continuation", [])
    if continuation:
        selected = set(continuation[-1]["mask"])
    else:
        backward = search.get("greedy_backward", [])
        selected = set(backward[-1]["mask"]) if backward else set(range(N_BLOCKS))

    while len(selected) > 12:
        trials = []
        for index in sorted(selected):
            mask = tuple(sorted(selected - {index}))
            result = evaluate(model, state, batch, rng_state, mask_key(mask), cache, reference, current, verbose=False)
            trials.append((objective(result), recovery(result), index, mask))
        _, _, removed, new_mask = max(trials, key=lambda x: (x[0], x[1], -x[2]))
        selected.remove(removed)
        result = cache["results"][mask_key(new_mask)]
        continuation.append({
            "step": len(continuation) + 1,
            "removed_block": removed,
            "mask": list(new_mask),
            "objective": objective(result),
            "recovery": recovery(result),
            "reverse1_cosine": result["summary"].get("reverse1_block0_cosine"),
        })
        save_cache(cache)


def synthetic_result(mask: list[int], label: str, current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if label == "current":
        source = current
    else:
        source = reference
    cmp = {
        source_name: {
            target: compare(source["gradients"][source_name][target], reference["gradients"][source_name][target])
            for target in TARGETS
        }
        for source_name in SOURCES
    }
    current_cmp = compare(current["gradients"]["reverse1"]["block_0"], reference["gradients"]["reverse1"]["block_0"])
    full_current_cmp = compare(current["gradients"]["full"]["block_0"], reference["gradients"]["full"]["block_0"])
    reverse_cmp = cmp["reverse1"]["block_0"]
    full_cmp = cmp["full"]["block_0"]
    return {
        "mask": mask,
        "mask_key": mask_key(mask),
        "fp32_block_count": len(mask),
        "fp16_block_count": N_BLOCKS - len(mask),
        "runtime_sec": 0.0,
        "peak_memory_bytes": None,
        "losses": source["losses"],
        "comparisons": cmp,
        "summary": {
            "reverse1_block0_cosine": reverse_cmp.get("cosine"),
            "reverse1_block0_relative_l2": reverse_cmp.get("relative_l2"),
            "reverse1_block0_recovery": (reverse_cmp.get("cosine") - current_cmp.get("cosine")) / (1.0 - current_cmp.get("cosine")),
            "full_block0_cosine": full_cmp.get("cosine"),
            "full_block0_relative_l2": full_cmp.get("relative_l2"),
            "full_block0_recovery": (full_cmp.get("cosine") - full_current_cmp.get("cosine")) / (1.0 - full_current_cmp.get("cosine")),
            "branch_block0_cosines": {name: cmp[name]["block_0"].get("cosine") for name in SOURCES},
        },
    }


def main() -> None:
    start = time.time()
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, sampler = build_probe_loader(cfg)

    # Materialize a small deterministic pool before candidate evaluation.  It
    # is deliberately larger than the final panel so metadata coverage is not
    # an artifact of the first twelve batches.
    raw_batches: list[dict[str, Any]] = []
    for index, raw in enumerate(loader):
        if index >= PANEL_POOL_COUNT:
            break
        raw_batches.append(clone_batch(raw))
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    manifest = parameter_manifest(model)

    cache = load_cache()
    cache.setdefault("meta", {})["phase2n_characterization_parameter_manifest"] = manifest
    cache["meta"]["panel_pool_count"] = len(raw_batches)
    save_cache(cache)

    print("PHASE2N_PANEL_BASELINES", len(raw_batches), flush=True)
    descriptors: list[dict[str, Any]] = []
    baseline_runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for index, raw in enumerate(raw_batches):
        print("PHASE2N_PANEL_BASELINE", index, flush=True)
        current = run_once(model, base_state, device_batch(raw), rng_state, tuple(), fp32_all=False)
        oracle = run_once(model, base_state, device_batch(raw), rng_state, tuple(), fp32_all=True)
        desc = batch_descriptor(raw, current, oracle, index)
        desc.update(compact_current_oracle(current, oracle))
        descriptors.append(desc)
        baseline_runs[desc["batch_hash"]] = (current, oracle)

    selected_indices, reasons = freeze_panel(descriptors)
    panel_descriptors = []
    for index in selected_indices:
        item = dict(descriptors[index])
        item["panel_order"] = len(panel_descriptors)
        item["selection_reason"] = reasons.get(index)
        panel_descriptors.append(item)

    # Persist the frozen panel before any candidate mask is evaluated.
    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel_fields = [
        "panel_order", "pool_index", "batch_hash", "sample_sha256", "image_sha256",
        "pid_signature", "camera_signature", "clothes_signature", "pids", "camids", "clothes",
        "selection_reason", "current_reverse1_loss", "current_reverse1_cosine",
        "current_reverse1_relative_l2", "current_reverse1_grad_norm", "oracle_reverse1_grad_norm",
        "oracle_reverse1_cosine", "oracle_reverse1_relative_l2", "current_loss", "oracle_loss",
    ]
    with PANEL_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=panel_fields)
        writer.writeheader()
        for item in panel_descriptors:
            row = {field: item.get(field) for field in panel_fields}
            for field in ("pids", "camids", "clothes"):
                row[field] = json.dumps(row[field], separators=(",", ":"))
            writer.writerow(row)

    cache["search"]["batch_panel_frozen"] = {
        "pool_count": len(raw_batches),
        "panel_count": len(panel_descriptors),
        "panel_batch_hashes": [item["batch_hash"] for item in panel_descriptors],
        "panel_pool_indices": selected_indices,
        "selection_rule": "current/oracle-only deterministic loss/norm quantiles plus PID/camera/clothes signature coverage",
        "csv": str(PANEL_CSV),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_cache(cache)

    source_masks = choose_source_masks(cache)
    print("PHASE2N_PANEL_CANDIDATES", json.dumps(source_masks, sort_keys=True), flush=True)
    candidate_rows: list[dict[str, Any]] = []
    candidate_main: dict[str, Any] = {}
    for label, mask in source_masks.items():
        key = mask_key(mask)
        # Main-batch results are already in the Phase 2N cache for all source
        # families; this call is a cache lookup and provides a single schema.
        main_result = cache["results"].get(key)
        if main_result is None:
            raise RuntimeError(f"missing main-batch result for selected candidate {label}: {key}")
        candidate_main[label] = main_result
        for panel_order, item in enumerate(panel_descriptors):
            raw = raw_batches[item["pool_index"]]
            current, oracle = baseline_runs[item["batch_hash"]]
            result = get_batch_result(cache, model, base_state, raw, rng_state, key, oracle, current)
            candidate_rows.append(result_row(result, item["pool_index"], item["batch_hash"], label))
            if panel_order == 0 or panel_order == len(panel_descriptors) - 1:
                print("PHASE2N_PANEL_CANDIDATE", json.dumps({"candidate": label, "batch": panel_order, "cosine": result["summary"].get("reverse1_block0_cosine")}, sort_keys=True), flush=True)

    # Main-batch density masks: continue the cached backward chain only where
    # the requested density counts were not previously represented.
    main_raw = raw_batches[0]
    main_current, main_oracle = baseline_runs[stable_batch_hash(main_raw)]
    ensure_density_masks(cache, model, base_state, device_batch(main_raw), rng_state, main_oracle, main_current)

    all_main_results = cache["results"]
    density: list[dict[str, Any]] = []
    for k in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, N_BLOCKS):
        if k == 0:
            result = synthetic_result([], "current", main_current, main_oracle)
        elif k == N_BLOCKS:
            result = synthetic_result(list(range(N_BLOCKS)), "oracle", main_current, main_oracle)
        else:
            choices = [result for result in all_main_results.values() if len(result.get("mask", [])) == k]
            if not choices:
                raise RuntimeError(f"density count {k} has no evaluated mask")
            result = min(choices, key=lambda x: (-objective(x), -recovery(x), tuple(x["mask"])))

        panel_cosines = []
        panel_runtimes = []
        panel_memory = []
        density_mask = tuple(result["mask"])
        for item in panel_descriptors:
            raw = raw_batches[item["pool_index"]]
            current, oracle = baseline_runs[item["batch_hash"]]
            if k == 0:
                batch_result = metric_result(current, oracle, current, tuple())
            elif k == N_BLOCKS:
                batch_result = metric_result(oracle, oracle, current, density_mask)
            else:
                batch_result = get_batch_result(cache, model, base_state, raw, rng_state, mask_key(density_mask), oracle, current)
            panel_cosines.append(finite_float(batch_result["summary"].get("reverse1_block0_cosine")))
            panel_runtimes.append(finite_float(batch_result.get("runtime_sec"), 0.0))
            memory = batch_result.get("peak_memory_bytes")
            if memory is not None:
                panel_memory.append(float(memory))
        density.append({
            "fp32_block_count": k,
            "fp32_fraction": k / N_BLOCKS,
            "mask": list(density_mask),
            "main_reverse1_cosine": result["summary"].get("reverse1_block0_cosine"),
            "main_reverse1_recovery": result["summary"].get("reverse1_block0_recovery"),
            "main_full_cosine": result["summary"].get("full_block0_cosine"),
            "median_cross_batch_reverse1_cosine": statistics.median(panel_cosines),
            "worst_cross_batch_reverse1_cosine": min(panel_cosines),
            "mean_forward_backward_runtime_sec": statistics.mean(panel_runtimes),
            "mean_peak_memory_bytes": statistics.mean(panel_memory) if panel_memory else None,
            "panel_cosines": panel_cosines,
        })
        save_cache(cache)

    # Structure analysis is deliberately descriptive; it does not select a
    # mask using retrieval metrics.
    structure_masks: dict[str, list[int]] = {}
    structure_masks.update(source_masks)
    for item in density:
        structure_masks[f"density_k{item['fp32_block_count']}"] = item["mask"]
    structure: dict[str, Any] = {}
    for label, mask in structure_masks.items():
        ordered = sorted(mask)
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        structure[label] = {
            "mask": ordered,
            "count": len(ordered),
            "fraction": len(ordered) / N_BLOCKS,
            "mean_depth": statistics.mean(ordered) if ordered else None,
            "min_depth": min(ordered) if ordered else None,
            "max_depth": max(ordered) if ordered else None,
            "adjacent_gaps": gaps,
            "gap_mean": statistics.mean(gaps) if gaps else None,
            "gap_std": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0 if gaps else None,
            "gap_periodic_score": (
                max(gaps.count(value) for value in set(gaps)) / len(gaps)
                if gaps else None
            ),
        }

    panel_payload = {
        "phase": "2N-J",
        "config": {
            "dataset": "ltcc",
            "seed": 1234,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "world_size_for_sampler": 2,
            "rank_for_sampler": 0,
            "loss_sources": list(SOURCES),
            "targets": list(TARGETS),
            "nobio_index": [1, 2],
            "reverse_indexing": "[:, i]",
            "precision_semantics": "selected blocks use exact Phase2M wrap_fp32_block; all other blocks formal autocast",
            "optimizer_step": False,
            "scheduler_step": False,
            "panel_selection_before_candidate_measurement": True,
        },
        "parameter_manifest": manifest,
        "pool": descriptors,
        "panel": panel_descriptors,
        "source_masks": source_masks,
        "main_results": candidate_main,
        "candidate_rows": candidate_rows,
        "gate": {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Robust candidate gate.  The required gate is median cosine >= .70 and
    # strict improvement over current on at least 10/12 batches.  Collateral
    # fields are recorded separately so they cannot silently replace it.
    baseline_by_hash = {item["batch_hash"]: item for item in panel_descriptors}
    gate_rows = {}
    for label in source_masks:
        rows = [row for row in candidate_rows if row["candidate"] == label]
        cosines = [finite_float(row["reverse1_cosine"]) for row in rows]
        improvements = [
            cos > finite_float(baseline_by_hash[row["batch_hash"]]["current_reverse1_cosine"])
            for row, cos in zip(rows, cosines)
        ]
        full_values = [finite_float(row["full_cosine"]) for row in rows]
        # The baseline comparisons were computed once, before candidate
        # measurement.  Reusing the compact descriptor here avoids repeatedly
        # comparing every large gradient tensor during report generation.
        current_full = [finite_float(baseline_by_hash[row["batch_hash"]].get("current_full_cosine")) for row in rows]
        collateral = {}
        for branch in ("ce_cosine", "triplet_cosine", "bio_cosine", "reverse0_cosine"):
            candidate_values = [finite_float(row[branch]) for row in rows]
            collateral[branch] = {
                "candidate_median": statistics.median(candidate_values),
                "current_median": statistics.median(
                    finite_float(baseline_by_hash[row["batch_hash"]].get("current_branch_block0_cosines", {}).get(branch.replace("_cosine", "")))
                    for row in rows
                ),
            }
        gate_rows[label] = {
            "mask": source_masks[label],
            "fp32_block_count": len(source_masks[label]),
            "median_reverse1_cosine": statistics.median(cosines),
            "worst_reverse1_cosine": min(cosines),
            "improved_over_current_count": int(sum(improvements)),
            "required_robust_gate": bool(statistics.median(cosines) >= 0.70 and sum(improvements) >= 10),
            "strong_robust_gate": bool(statistics.median(cosines) >= 0.85 and min(cosines) >= 0.50),
            "full_median_cosine": statistics.median(full_values),
            "current_full_median_cosine": statistics.median(current_full),
            "full_no_decline": bool(statistics.median(full_values) >= statistics.median(current_full) - 0.02),
            "collateral": collateral,
        }
    robust = [label for label, row in gate_rows.items() if row["required_robust_gate"] and row["full_no_decline"]]
    panel_payload["gate"] = {"candidates": gate_rows, "robust_candidates": robust, "strong_candidates": [label for label in robust if gate_rows[label]["strong_robust_gate"]]}
    PANEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    PANEL_JSON.write_text(json.dumps(panel_payload, indent=2, sort_keys=True) + "\n")

    density_payload = {
        "phase": "2N-K",
        "n_blocks": N_BLOCKS,
        "panel_batch_hashes": [item["batch_hash"] for item in panel_descriptors],
        "curve": density,
        "knee_analysis": {
            "method": "largest positive second difference of median cross-batch cosine after sorting by block count; descriptive only",
            "note": "No retrieval metric is used for mask selection.",
        },
        "depth_structure": structure,
    }
    DENSITY_JSON.write_text(json.dumps(density_payload, indent=2, sort_keys=True) + "\n")

    best_payload = {
        "phase": "2N-P-preselection",
        "source_masks": source_masks,
        "main_batch": {
            label: {
                "mask": value["mask"],
                "reverse1_cosine": value["summary"].get("reverse1_block0_cosine"),
                "reverse1_recovery": value["summary"].get("reverse1_block0_recovery"),
                "full_cosine": value["summary"].get("full_block0_cosine"),
            }
            for label, value in candidate_main.items()
        },
        "multibatch_gate": gate_rows,
        "robust_candidates": robust,
        "strong_candidates": panel_payload["gate"]["strong_candidates"],
        "density_curve": density,
        "depth_structure": structure,
    }
    BEST_JSON.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")

    # Mirror the complete serializable cache under the explicitly requested
    # diagnostics path.  The primary cache remains in cache/phase2n so the
    # large JSON does not get confused with a report.
    SEARCH_CACHE_EXPORT.write_text(json.dumps({
        "primary_cache": str(CACHE_PATH),
        "phase": "2N",
        "meta": cache.get("meta", {}),
        "results": cache.get("results", {}),
        "batch_results": cache.get("batch_results", {}),
        "search": cache.get("search", {}),
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True) + "\n")

    summary = {
        "phase": "2N-J/K/L",
        "cache": str(CACHE_PATH),
        "diagnostics_cache_export": str(SEARCH_CACHE_EXPORT),
        "mask_count_main": len(cache.get("results", {})),
        "batch_result_count": sum(len(values) for values in cache.get("batch_results", {}).values()),
        "panel_count": len(panel_descriptors),
        "panel_pool_count": len(raw_batches),
        "source_masks": source_masks,
        "robust_candidates": robust,
        "strong_candidates": panel_payload["gate"]["strong_candidates"],
        "density_counts": [entry["fp32_block_count"] for entry in density],
        "runtime_sec": time.time() - start,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# PHASE 2N MULTI-BATCH ROBUSTNESS AND DENSITY",
        "",
        f"The deterministic pool contained {len(raw_batches)} batches; the frozen panel contains {len(panel_descriptors)} batches.",
        "Candidate observations were not used to select the panel.",
        "",
        "## Frozen panel",
        "",
        "| Order | Pool batch | PID signature | Camera signature | Clothes signature | Reverse-1 loss | Current cosine |",
        "|---:|---:|---|---|---|---:|---:|",
    ]
    for item in panel_descriptors:
        report_lines.append(f"| {item['panel_order']} | {item['pool_index']} | {item['pid_signature']} | {item['camera_signature']} | {item['clothes_signature']} | {item['current_reverse1_loss']:.6f} | {item['current_reverse1_cosine']:.6f} |")
    report_lines += ["", "## Candidate gate", "", "| Candidate | Mask | FP32 blocks | Median R1 cosine | Worst | Better than Current | Robust | Strong | Full no decline |", "|---|---|---:|---:|---:|---:|---|---|---|"]
    for label, row in gate_rows.items():
        report_lines.append(f"| {label} | `{row['mask']}` | {row['fp32_block_count']} | {row['median_reverse1_cosine']:.6f} | {row['worst_reverse1_cosine']:.6f} | {row['improved_over_current_count']}/12 | {row['required_robust_gate']} | {row['strong_robust_gate']} | {row['full_no_decline']} |")
    report_lines += ["", "## Density curve", "", "| FP32 blocks | Mask | Main cosine | Median panel cosine | Worst panel cosine |", "|---:|---|---:|---:|---:|"]
    for item in density:
        report_lines.append(f"| {item['fp32_block_count']} | `{item['mask']}` | {item['main_reverse1_cosine']:.6f} | {item['median_cross_batch_reverse1_cosine']:.6f} | {item['worst_cross_batch_reverse1_cosine']:.6f} |")
    (REPORT_DIR / "PHASE_2N_MULTIBATCH_ROBUSTNESS.md").write_text("\n".join(report_lines) + "\n")
    print("PHASE2N_CHARACTERIZATION_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)

    model.to("cpu")
    del model, dataset, loader, raw_batches, baseline_runs, cache
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
