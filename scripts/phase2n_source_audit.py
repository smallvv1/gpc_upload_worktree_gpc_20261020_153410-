#!/usr/local/bin/python3
"""PHASE 2N-M/N/O/Q: source audit and repeatability for sparse masks.

All experiments in this file are no-step numerical diagnostics on PPU.  It
does not modify the formal model or training entry point.  The selected block
wrapper is the exact Phase 2M ``wrap_fp32_block`` implementation.
"""
from __future__ import annotations

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

import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from global_precision_policy_screen_phase2m import wrap_fp32_block  # noqa: E402
from initialization_footprint_phase2i import (  # noqa: E402
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from layerwise_gradient_frontier_phase2l import component_losses  # noqa: E402
from loss.triplet_loss import TripletLoss  # noqa: E402
from sparse_mask_search_phase2n import (  # noqa: E402
    DEVICE,
    N_BLOCKS,
    SOURCES,
    TARGETS,
    compare,
    install_hooks,
    mask_key,
    metric_result,
    run_once,
)


DIAG_DIR = ROOT / "diagnostics"
REPORT_DIR = ROOT / "reports"
BEST_PATH = DIAG_DIR / "phase2n_best_masks.json"
SOURCE_PATH = DIAG_DIR / "phase2n_source_audit.json"
REPEAT_PATH = DIAG_DIR / "phase2n_repeatability.json"
SELECTION_PATH = DIAG_DIR / "phase2n_compat_selection.json"


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clone_to_cpu(item) for item in value]
    return value


def forward_snapshot(score: dict[str, Any], feat: list[Any]) -> dict[str, torch.Tensor | None]:
    return {
        "cls_score": clone_to_cpu(score.get("cls_score")),
        "clip_bio_score": clone_to_cpu(score.get("clip_bio_score")),
        "clip_bio_reverse_score": clone_to_cpu(score.get("clip_bio_reverse_score")),
        "feat0": clone_to_cpu(feat[0] if feat else None),
    }


def compare_snapshots(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {key: compare(a.get(key), b.get(key)) for key in a}


def snapshot_signature(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        key: {
            "exists": isinstance(value, torch.Tensor),
            "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
            "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else None,
            "sha256": tensor_sha(value) if isinstance(value, torch.Tensor) else None,
        }
        for key, value in snapshot.items()
    }


def run_formal_forward_snapshot(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    batch: dict[str, Any],
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    """Capture Current's formal forward outputs without building a graph.

    ``run_once`` is intentionally the authoritative gradient evaluator and
    does not retain forward tensors.  This small replay gives the source
    audit a byte-level comparator for the backward-only diagnostic while
    preserving the same state, RNG, train mode, and autocast semantics.
    """
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.train()
    with torch.no_grad():
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        snapshot = forward_snapshot(score, feat)
    del score, feat
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return snapshot


def cast_arg(value: Any, dtype: torch.dtype = torch.float32) -> Any:
    return value.to(dtype=dtype) if isinstance(value, torch.Tensor) else value


def state_anchor_forward(block: torch.nn.Module):
    """Formal FP16 branch arithmetic with a reconstructed FP32 residual state."""
    original = block.forward

    def forward(*args, _block=block, _original=original, **kwargs):
        x = args[0]
        rope = args[1] if len(args) > 1 else kwargs.get("rope", None)
        attn_mask = args[2] if len(args) > 2 else kwargs.get("attn_mask", None)
        # The two branch calls remain inside the formal autocast context.  Only
        # residual reconstruction is explicitly FP32; the selected block is
        # not sent through the Phase 2M full-block wrapper.
        with amp.autocast(enabled=True):
            if getattr(_block, "gamma_1", None) is None:
                attn_out = _block.drop_path1(_block.attn(_block.norm1(x), rope=rope, attn_mask=attn_mask))
                x = x.float() + attn_out.float()
                mlp_out = _block.drop_path2(_block.mlp(_block.norm2(x)))
                x = x.float() + mlp_out.float()
            else:
                attn_out = _block.drop_path1(_block.gamma_1 * _block.attn(_block.norm1(x), rope=rope, attn_mask=attn_mask))
                x = x.float() + attn_out.float()
                mlp_out = _block.drop_path2(_block.gamma_2 * _block.mlp(_block.norm2(x)))
                x = x.float() + mlp_out.float()
        return x

    return original, forward


def apply_state_anchor(model: torch.nn.Module, mask: tuple[int, ...]):
    originals = []
    for index in mask:
        block = model.blocks[index]
        original, wrapped = state_anchor_forward(block)
        originals.append((block, original))
        block.forward = wrapped
    return originals


def apply_boundary_schedule(model: torch.nn.Module, mask: tuple[int, ...]):
    """Full FP32 blocks with half-state rounding at unselected boundaries."""
    originals = []
    for index, block in enumerate(list(model.blocks) + [model.head_bio]):
        original = wrap_fp32_block(block)
        full_forward = block.forward
        quantize = index == N_BLOCKS or index not in mask

        def forward(*args, _full=full_forward, _quantize=quantize, **kwargs):
            output = _full(*args, **kwargs)
            if _quantize and isinstance(output, torch.Tensor):
                return output.half().float()
            return output

        originals.append((block, original))
        block.forward = forward
    return originals


class _BackwardOnlyFunction(torch.autograd.Function):
    """Forward formal path; selected block VJP recomputed in FP32.

    The module and its original bound forward are non-tensor context values;
    block parameters are explicit Function inputs so their manually computed
    FP32 gradients are returned to autograd.  The implementation is used only
    for a bounded diagnostic and is rejected safely if the runtime cannot
    support it.
    """

    @staticmethod
    def forward(ctx, block, original, x, rope, attn_mask, *params):
        ctx.block = block
        ctx.original = original
        ctx.rope = rope
        ctx.attn_mask = attn_mask
        ctx.param_count = len(params)
        ctx.active_indices = [index for index, parameter in enumerate(params) if parameter.requires_grad]
        ctx.save_for_backward(x, *params)
        with amp.autocast(enabled=True):
            with torch.no_grad():
                return original(x, rope=rope, attn_mask=attn_mask)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x = saved[0]
        params = saved[1:]
        x32 = x.detach().float().requires_grad_(True)
        rope32 = cast_arg(ctx.rope)
        active_params = [params[index] for index in ctx.active_indices]
        with torch.enable_grad():
            with amp.autocast(enabled=False):
                output = ctx.original(x32, rope=rope32, attn_mask=ctx.attn_mask)
            grads = torch.autograd.grad(
                output,
                (x32, *active_params),
                grad_outputs=grad_output.float(),
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        dx = grads[0]
        parameter_grads = [None] * ctx.param_count
        for position, index in enumerate(ctx.active_indices, start=1):
            parameter_grads[index] = grads[position]
        # block, original, x, rope, attn_mask, params...
        return (None, None, dx.to(dtype=x.dtype), None, None, *parameter_grads)


def apply_backward_only(model: torch.nn.Module, mask: tuple[int, ...]):
    originals = []
    for index in mask:
        block = model.blocks[index]
        original = block.forward
        params = tuple(block.parameters())

        def forward(*args, _block=block, _original=original, _params=params, **kwargs):
            x = args[0]
            rope = args[1] if len(args) > 1 else kwargs.get("rope", None)
            attn_mask = args[2] if len(args) > 2 else kwargs.get("attn_mask", None)
            return _BackwardOnlyFunction.apply(_block, _original, x, rope, attn_mask, *_params)

        originals.append((block, original))
        block.forward = forward
    return originals


def restore_wrappers(originals):
    for module, original in reversed(originals):
        module.forward = original


def run_wrapped(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    batch: dict[str, Any],
    rng_state: dict[str, Any],
    mask: tuple[int, ...],
    mode: str,
    outer_autocast: bool = True,
    capture_outputs: bool = False,
) -> dict[str, Any]:
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    if mode == "state_anchor":
        originals = apply_state_anchor(model, mask)
    elif mode == "boundary_schedule":
        originals = apply_boundary_schedule(model, mask)
    elif mode == "backward_only":
        originals = apply_backward_only(model, mask)
    else:
        raise ValueError(mode)
    saved, handles = install_hooks(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    try:
        with amp.autocast(enabled=outer_autocast):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        snapshot = forward_snapshot(score, feat) if capture_outputs else None
        losses = component_losses(score, feat, batch, TripletLoss())
        targets = [saved[name] for name in TARGETS]
        gradients: dict[str, dict[str, torch.Tensor | None]] = {}
        for source in SOURCES:
            values = torch.autograd.grad(losses[source], targets, retain_graph=True, allow_unused=True)
            gradients[source] = {
                name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None
                for name, value in zip(TARGETS, values)
            }
        return {
            "mode": mode,
            "mask": list(mask),
            "precision": mode,
            "losses": {
                key: {
                    "value": float(value.detach().double().cpu()),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value.detach()).item()),
                }
                for key, value in losses.items()
            },
            "gradients": gradients,
            "forward_outputs": snapshot,
            "runtime_sec": time.perf_counter() - start,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)) if torch.cuda.is_available() else None,
        }
    finally:
        for handle in handles:
            handle.remove()
        restore_wrappers(originals)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def compact_result(run: dict[str, Any], reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result = metric_result(run, reference, current, tuple(run["mask"]))
    result["mode"] = run["mode"]
    result["forward_output_signature"] = snapshot_signature(run.get("forward_outputs"))
    return result


def selected_masks() -> dict[str, tuple[int, ...]]:
    best = json.loads(BEST_PATH.read_text())
    return {label: tuple(value) for label, value in best["source_masks"].items()}


def choose_compat_mask() -> dict[str, Any]:
    best = json.loads(BEST_PATH.read_text())
    gate = best.get("multibatch_gate", {})
    eligible = []
    for label, row in gate.items():
        collateral = row.get("collateral", {})
        collateral_ok = all(
            finite(item.get("candidate_median")) >= finite(item.get("current_median")) - 0.10
            for item in collateral.values()
        )
        if row.get("required_robust_gate") and row.get("full_no_decline") and collateral_ok:
            eligible.append((
                len(row["mask"]),
                -finite(row.get("worst_reverse1_cosine"), -1e9),
                -finite(row.get("median_reverse1_cosine"), -1e9),
                label,
                collateral_ok,
            ))
    if not eligible:
        return {"created": False, "reason": "no robust candidate with collateral preservation", "eligible": []}
    eligible.sort()
    chosen = eligible[0]
    row = gate[chosen[3]]
    return {
        "created": True,
        "type": "SPARSE BLOCK MASK",
        "label": chosen[3],
        "mask": row["mask"],
        "fp32_block_count": len(row["mask"]),
        "fp32_fraction": len(row["mask"]) / N_BLOCKS,
        "selection_rule": "minimum FP32 block count, then best worst-batch Reverse-1 cosine, then median; no retrieval metric",
        "eligible": [item[3] for item in eligible],
        "gate_row": row,
        "candidate_name": "PPU_COMPAT_V2_SPARSE_MASK",
    }


def pairwise_repeatability(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    batch: dict[str, Any],
    rng_state: dict[str, Any],
    mask: tuple[int, ...],
    mode: str,
    repeats: int = 20,
) -> dict[str, Any]:
    runs = []
    for index in range(repeats):
        print("PHASE2N_REPEAT", mode, index + 1, flush=True)
        if mode == "current":
            run = run_once(model, state, batch, rng_state, tuple())
        else:
            run = run_wrapped(model, state, batch, rng_state, mask, mode, outer_autocast=True)
        runs.append(run)
    base = runs[0]
    per_source: dict[str, Any] = {}
    for source in SOURCES:
        per_target = {}
        for target in TARGETS:
            comparisons = [compare(base["gradients"][source][target], item["gradients"][source][target]) for item in runs[1:]]
            cosines = [finite(item.get("cosine")) for item in comparisons]
            rels = [finite(item.get("relative_l2")) for item in comparisons]
            bitwise = [
                bool(torch.equal(base["gradients"][source][target], item["gradients"][source][target]))
                if base["gradients"][source][target] is not None and item["gradients"][source][target] is not None
                else False
                for item in runs[1:]
            ]
            per_target[target] = {
                "pairwise_cosine_min": min(cosines),
                "pairwise_cosine_max": max(cosines),
                "pairwise_relative_l2_max": max(rels),
                "std_over_mean_norm": statistics.pstdev([tensor_l2(item["gradients"][source][target]) for item in runs]) / (statistics.mean([tensor_l2(item["gradients"][source][target]) for item in runs]) + 1e-12),
                "bitwise_identical_fraction": sum(bitwise) / len(bitwise),
            }
        per_source[source] = per_target
    return {
        "mode": mode,
        "mask": list(mask),
        "repeats": repeats,
        "same_state_forward_bitwise_repeat_rate": None,
        "gradient_repeatability": per_source,
    }


def tensor_l2(value: torch.Tensor | None) -> float:
    return float(torch.linalg.vector_norm(value.detach().double()).item()) if value is not None else float("nan")


def main() -> None:
    started = time.time()
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, _sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    masks = selected_masks()

    print("PHASE2N_SOURCE_REFERENCE", flush=True)
    reference = run_once(model, base_state, batch, rng_state, tuple(), fp32_all=True)
    print("PHASE2N_SOURCE_CURRENT", flush=True)
    current = run_once(model, base_state, batch, rng_state, tuple(), fp32_all=False)
    current_snapshot = run_formal_forward_snapshot(model, base_state, batch, rng_state)
    audit: dict[str, Any] = {
        "phase": "2N-M/N/O",
        "config": {
            "dataset": "ltcc",
            "seed": 1234,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "nobio_index": [1, 2],
            "reverse_indexing": "[:, i]",
            "device": str(DEVICE),
            "current": "formal autocast path",
            "oracle": "full mathematical FP32 forward/loss/backward",
            "precision_block_semantics": "exact Phase2M wrap_fp32_block; selected block output retained as returned by that wrapper",
            "optimizer_step": False,
            "scheduler_step": False,
        },
        "batch": {
            "pid": batch["pid"].detach().cpu().tolist(),
            "camid": batch["camid"].detach().cpu().tolist(),
            "clothes": batch["clothes_id"].detach().cpu().tolist(),
            "image_shape": list(batch["image"].shape),
        },
        "masks": {label: list(mask) for label, mask in masks.items()},
        "state_anchor": {},
        "forward_boundary_quantization": {},
        "backward_only": {},
    }

    for label, mask in masks.items():
        print("PHASE2N_STATE_ANCHOR", label, flush=True)
        run = run_wrapped(model, base_state, batch, rng_state, mask, "state_anchor", True, True)
        item = compact_result(run, reference, current)
        item["forward_vs_current"] = compare_snapshots(run["forward_outputs"], current_snapshot)
        audit["state_anchor"][label] = item

    # Full FP32 boundary schedules use identical selected depths.  A schedule
    # with no quantized boundary is the ceiling; all unselected boundaries are
    # half-rounded for the sparse prediction.
    for label, mask in masks.items():
        print("PHASE2N_BOUNDARY_SCHEDULE", label, flush=True)
        run = run_wrapped(model, base_state, batch, rng_state, mask, "boundary_schedule", False, True)
        item = compact_result(run, reference, current)
        item["forward_vs_current"] = compare_snapshots(run["forward_outputs"], current_snapshot)
        audit["forward_boundary_quantization"][label] = item

    # The custom VJP is a bounded feasibility audit.  If any PPU/autograd
    # invariant rejects it, preserve the error and continue to repeatability.
    backward_errors = {}
    for label, mask in masks.items():
        print("PHASE2N_BACKWARD_ONLY", label, flush=True)
        try:
            run = run_wrapped(model, base_state, batch, rng_state, mask, "backward_only", True, True)
            item = compact_result(run, reference, current)
            item["forward_vs_current"] = compare_snapshots(run["forward_outputs"], current_snapshot)
            item["status"] = "EXECUTED"
            backward_errors[label] = item
        except Exception as exc:  # diagnostic must not block later phases
            backward_errors[label] = {
                "status": "NOT IMPLEMENTED SAFELY",
                "error": f"{type(exc).__name__}: {exc}",
                "mask": list(mask),
            }
            print("PHASE2N_BACKWARD_ONLY_ERROR", json.dumps(backward_errors[label], sort_keys=True), flush=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    audit["backward_only"] = backward_errors
    audit["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit["runtime_sec"] = time.time() - started
    SOURCE_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    selection = choose_compat_mask()
    selection["source_audit_path"] = str(SOURCE_PATH)
    selection["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    repeatability: dict[str, Any] = {
        "phase": "2N-Q",
        "config": {"repeats": 20, "loss": "Reverse-1 individual loss", "same_state": True, "optimizer_step": False},
        "candidate": selection,
        "runs": {},
    }
    if selection.get("created"):
        chosen_mask = tuple(selection["mask"])
        repeatability["runs"]["current"] = pairwise_repeatability(model, base_state, batch, rng_state, chosen_mask, "current", 20)
        # Use a small local loop for exact sparse-block repeatability because
        # run_once owns the exact Phase2M wrapper and is the authoritative path.
        exact_runs = []
        for index in range(20):
            # run_once without a wrapper is Current, so install the exact
            # Phase2M block wrapper explicitly and replay its gradient path.
            model.load_state_dict(base_state, strict=True)
            restore_rng_state(rng_state)
            model.train(); model.zero_grad(set_to_none=True)
            originals = []
            for block_index in chosen_mask:
                block = model.blocks[block_index]
                originals.append((block, wrap_fp32_block(block)))
            saved, handles = install_hooks(model)
            try:
                with amp.autocast(enabled=True):
                    score, feat = model(batch["image"], cam_label=batch["camid"])
                losses = component_losses(score, feat, batch, TripletLoss())
                grads = {}
                for source in SOURCES:
                    values = torch.autograd.grad(losses[source], [saved[name] for name in TARGETS], retain_graph=True, allow_unused=True)
                    grads[source] = {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None for name, value in zip(TARGETS, values)}
                exact_runs.append({"gradients": grads})
            finally:
                for handle in handles: handle.remove()
                restore_wrappers(originals)
                del score, feat, losses, saved
                gc.collect(); torch.cuda.empty_cache()
        base = exact_runs[0]
        exact_summary = {"mode": "sparse_block", "mask": list(chosen_mask), "repeats": 20, "gradient_repeatability": {}}
        for source in SOURCES:
            exact_summary["gradient_repeatability"][source] = {}
            for target in TARGETS:
                vals = [compare(base["gradients"][source][target], run["gradients"][source][target]) for run in exact_runs[1:]]
                cos = [finite(x.get("cosine")) for x in vals]; rel = [finite(x.get("relative_l2")) for x in vals]
                bits = [torch.equal(base["gradients"][source][target], run["gradients"][source][target]) for run in exact_runs[1:]]
                norms = [tensor_l2(run["gradients"][source][target]) for run in exact_runs]
                exact_summary["gradient_repeatability"][source][target] = {
                    "pairwise_cosine_min": min(cos), "pairwise_cosine_max": max(cos),
                    "pairwise_relative_l2_max": max(rel), "std_over_mean_norm": statistics.pstdev(norms) / (statistics.mean(norms) + 1e-12),
                    "bitwise_identical_fraction": sum(bits) / len(bits),
                }
        repeatability["runs"]["compat_v2"] = exact_summary
    else:
        repeatability["status"] = "NOT_RUN_NO_ROBUST_CANDIDATE"
    REPEAT_PATH.write_text(json.dumps(repeatability, indent=2, sort_keys=True) + "\n")

    report = [
        "# PHASE 2N SOURCE AUDIT",
        "",
        f"Selected training candidate: `{selection.get('candidate_name', 'NONE')}` with mask `{selection.get('mask')}`.",
        "",
        "## State-anchor interpretation",
        "",
        "The state-anchor wrapper keeps attention/MLP branch arithmetic under formal autocast and reconstructs only selected residual states in FP32. It is a diagnostic, not the training candidate unless explicitly selected by the numerical gate.",
        "",
        "| Candidate | Mask | Reverse-1 cosine | Recovery | Full cosine |",
        "|---|---|---:|---:|---:|",
    ]
    for label, item in audit["state_anchor"].items():
        report.append(f"| {label} | `{item['mask']}` | {item['summary']['reverse1_block0_cosine']:.6f} | {item['summary']['reverse1_block0_recovery']:.6f} | {item['summary']['full_block0_cosine']:.6f} |")
    report += ["", "## Full-FP32 forward with scheduled boundary quantization", "", "| Candidate | Mask | Reverse-1 cosine | Recovery | Full cosine |", "|---|---|---:|---:|---:|"]
    for label, item in audit["forward_boundary_quantization"].items():
        report.append(f"| {label} | `{item['mask']}` | {item['summary']['reverse1_block0_cosine']:.6f} | {item['summary']['reverse1_block0_recovery']:.6f} | {item['summary']['full_block0_cosine']:.6f} |")
    report += ["", "## Backward-only VJP", "", "| Candidate | Status | Reverse-1 cosine | Recovery |", "|---|---|---:|---:|"]
    for label, item in audit["backward_only"].items():
        if item.get("status") == "EXECUTED":
            report.append(f"| {label} | {item['status']} | {item['summary']['reverse1_block0_cosine']:.6f} | {item['summary']['reverse1_block0_recovery']:.6f} |")
        else:
            report.append(f"| {label} | {item.get('status')} | — | — |")
    (REPORT_DIR / "PHASE_2N_SOURCE_AUDIT.md").write_text("\n".join(report) + "\n")

    print("PHASE2N_SOURCE_AUDIT_COMPLETE", json.dumps({"selection": selection, "source": str(SOURCE_PATH), "repeatability": str(REPEAT_PATH)}, sort_keys=True), flush=True)
    model.to("cpu")
    del model, dataset, loader, batch
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
