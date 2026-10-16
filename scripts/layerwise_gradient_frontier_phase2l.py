#!/usr/local/bin/python3
"""PHASE 2L-A/B: layerwise FP16-vs-FP32 gradient frontier.

The script is diagnostic-only.  It uses one fixed real LTCC batch and one
freshly loaded biasless Full DIFFER model.  For each loss source it performs a
new forward/backward in the current autocast path and in a full-FP32 reference,
capturing the input/output gradient at every EVA block boundary (including the
biometric head block).  It also repeats the Reverse-1 current path 20 times to
locate the first boundary whose gradient is not replay-stable.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from initialization_footprint_phase2i import (  # noqa: E402
    SEED,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)
from loss.triplet_loss import TripletLoss  # noqa: E402


DEVICE = torch.device("cuda:0")
SOURCES = ("ce", "triplet", "bio", "reverse0", "reverse1", "full")
REPEAT_COUNT = 20


class CaptureWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def stat_tensor(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"exists": False}
    x = value.detach()
    xd = x.double()
    finite = torch.isfinite(xd)
    return {
        "exists": True,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "numel": int(x.numel()),
        "nan_count": int(torch.isnan(xd).sum().item()),
        "inf_count": int(torch.isinf(xd).sum().item()),
        "zero_count": int((x == 0).sum().item()),
        "max_abs_finite": float(xd[finite].abs().max().item()) if finite.any() else None,
        "l2": float(torch.linalg.vector_norm(xd).item()),
        "finite": bool(finite.all().item()),
    }


def compare_tensors(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True, "a_exists": a is not None, "b_exists": b is not None}
    x = a.detach().double().cpu().reshape(-1)
    y = b.detach().double().cpu().reshape(-1)
    if x.numel() != y.numel():
        return {"missing": False, "shape_mismatch": True, "a_numel": int(x.numel()), "b_numel": int(y.numel())}
    d = x - y
    yn = float(torch.linalg.vector_norm(y).item())
    xn = float(torch.linalg.vector_norm(x).item())
    return {
        "missing": False,
        "shape_mismatch": False,
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn and yn else None,
        "relative_l2": float(torch.linalg.vector_norm(d).item()) / yn if yn else None,
        "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
        "a_l2": xn,
        "b_l2": yn,
        "a_nan_count": int(torch.isnan(x).sum().item()),
        "a_inf_count": int(torch.isinf(x).sum().item()),
        "b_nan_count": int(torch.isnan(y).sum().item()),
        "b_inf_count": int(torch.isinf(y).sum().item()),
    }


def detach_batch(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else raw[key] for key, value in raw.items()}


def component_losses(score, feat, batch, triplet):
    """Reproduce the current loss components without writer side effects."""
    target = batch["pid"]
    ce = F.cross_entropy(score["cls_score"], target)
    tri = triplet(feat[0], target)[0]

    image_bio = F.normalize(score["clip_bio_score"], dim=-1)
    if image_bio.shape[1] > 200:
        image_bio = image_bio.unsqueeze(1)
    caption = batch["caption_feature"].type(image_bio.type())
    text_bio = F.normalize(caption[:, :image_bio.shape[1]], dim=-1)
    text_nonbio = F.normalize(caption[:, image_bio.shape[1]:], dim=-1)

    bio_terms = []
    for i in range(image_bio.shape[1]):
        logits = image_bio[:, i] @ text_bio[:, i].T
        labels = torch.arange(image_bio[:, i].shape[0], device=image_bio.device)
        bio_terms.append(F.cross_entropy(logits, labels))
    bio = sum(bio_terms)

    reverse = score["clip_bio_reverse_score"]
    reverse_terms = []
    for i in range(reverse.shape[1]):
        image_i = F.normalize(reverse[:, i], dim=-1)
        image_i = image_i.type(text_nonbio.type())
        logits = image_i @ text_nonbio[:, i].T
        labels = torch.arange(image_i.shape[0], device=image_i.device)
        reverse_terms.append(F.cross_entropy(logits, labels))
    return {
        "ce": ce,
        "triplet": tri,
        "bio": bio,
        "reverse0": reverse_terms[0],
        "reverse1": reverse_terms[1],
        "full": ce + tri + bio + sum(reverse_terms),
    }


def install_boundary_hooks(model):
    saved: dict[str, dict[str, torch.Tensor | None]] = {}
    handles = []
    # The LTCC `eva02_cloth` factory returns the Eva backbone directly (the
    # wrapper used by some other configurations is not present here).
    modules = [(f"block_{i}", block) for i, block in enumerate(model.blocks)]
    if hasattr(model, "head_bio"):
        modules.append(("head_bio", model.head_bio))
    for name, module in modules:
        saved[name] = {"input": None, "output": None}

        def hook(_module, inputs, output, name=name):
            x = inputs[0] if inputs else None
            if isinstance(x, torch.Tensor):
                x.retain_grad()
                saved[name]["input"] = x
            if isinstance(output, torch.Tensor):
                output.retain_grad()
                saved[name]["output"] = output

        handles.append(module.register_forward_hook(hook))
    return saved, handles


def capture_parameter_grads(model):
    names = dict(model.named_parameters())
    result = {}
    for name in [
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.proj.weight",
        "blocks.11.attn.qkv.weight",
        "blocks.22.attn.qkv.weight",
        "head_bio.attn.qkv.weight",
    ]:
        result[name] = stat_tensor(names[name].grad if name in names else None)
    return result


def one_backward(model, base_state, batch, rng_state, cfg, dataset, source: str, fp32: bool):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    saved, handles = install_boundary_hooks(model)
    triplet = TripletLoss()  # config has NO_MARGIN=True for the official LTCC path
    with amp.autocast(enabled=not fp32):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, triplet)
    loss = losses[source]
    loss.backward()
    blocks = {}
    for name, tensors in saved.items():
        blocks[name] = {
            "input": stat_tensor(tensors["input"].grad if isinstance(tensors["input"], torch.Tensor) else None),
            "output": stat_tensor(tensors["output"].grad if isinstance(tensors["output"], torch.Tensor) else None),
            "input_tensor_dtype": str(tensors["input"].dtype) if isinstance(tensors["input"], torch.Tensor) else None,
            "output_tensor_dtype": str(tensors["output"].dtype) if isinstance(tensors["output"], torch.Tensor) else None,
        }
    result = {
        "source": source,
        "precision": "fp32_reference" if fp32 else "current_autocast",
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "all_component_values": {key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype)} for key, value in losses.items()},
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "blocks": blocks,
        "parameter_grads": capture_parameter_grads(model),
    }
    for handle in handles:
        handle.remove()
    del score, feat, loss, losses, triplet
    return result


def compare_runs(current, reference):
    out = {}
    for boundary in current["blocks"]:
        out[boundary] = {
            direction: compare_tensors(
                # The stats above are deliberately compact; reload is not
                # possible, so this function is populated by run_with_tensors.
                None, None
            ) for direction in ("input", "output")
        }
    return out


def one_backward_with_tensors(model, base_state, batch, rng_state, cfg, dataset, source: str, fp32: bool):
    """As one_backward, retaining detached boundary gradients for comparison."""
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    saved, handles = install_boundary_hooks(model)
    triplet = TripletLoss()
    with amp.autocast(enabled=not fp32):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, triplet)
    loss = losses[source]
    loss.backward()
    tensors = {
        name: {
            "input": saved[name]["input"].grad.detach().cpu().clone() if isinstance(saved[name]["input"], torch.Tensor) and saved[name]["input"].grad is not None else None,
            "output": saved[name]["output"].grad.detach().cpu().clone() if isinstance(saved[name]["output"], torch.Tensor) and saved[name]["output"].grad is not None else None,
        }
        for name in saved
    }
    summary = {
        "source": source,
        "precision": "fp32_reference" if fp32 else "current_autocast",
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "all_component_values": {key: {"value": float(value.detach().double().cpu()), "dtype": str(value.dtype)} for key, value in losses.items()},
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "blocks": {
            name: {
                "input": stat_tensor(tensors[name]["input"]),
                "output": stat_tensor(tensors[name]["output"]),
                "input_tensor_dtype": str(saved[name]["input"].dtype) if isinstance(saved[name]["input"], torch.Tensor) else None,
                "output_tensor_dtype": str(saved[name]["output"].dtype) if isinstance(saved[name]["output"], torch.Tensor) else None,
            }
            for name in saved
        },
        "parameter_grads": capture_parameter_grads(model),
        "_tensors": tensors,
    }
    for handle in handles:
        handle.remove()
    del score, feat, loss, losses, triplet
    return summary


def summarize_replay(replays):
    names = list(replays[0]["_tensors"])
    output = {}
    for name in names:
        output[name] = {}
        for direction in ("input", "output"):
            values = [item["_tensors"][name][direction] for item in replays]
            first = values[0]
            pairs = [compare_tensors(first, value) for value in values[1:]]
            cos = [x.get("cosine") for x in pairs if x.get("cosine") is not None]
            rel = [x.get("relative_l2") for x in pairs if x.get("relative_l2") is not None]
            # elementwise std/mean-norm is computed in FP64 on CPU for the
            # selected boundary; it is a diagnostic summary, not training.
            stack = torch.stack([value.double() for value in values], dim=0)
            mean_norm = float(torch.linalg.vector_norm(stack.mean(dim=0)).item())
            std_norm = float(torch.linalg.vector_norm(stack.std(dim=0, unbiased=False)).item())
            bitwise = [bool(torch.equal(first, value)) for value in values[1:]]
            output[name][direction] = {
                "pairwise_cosine_min": min(cos) if cos else None,
                "pairwise_cosine_max": max(cos) if cos else None,
                "pairwise_relative_l2_max": max(rel) if rel else None,
                "std_over_mean_norm": std_norm / mean_norm if mean_norm else None,
                "bitwise_identical_percentage": 100.0 * sum(bitwise) / len(bitwise) if bitwise else None,
            }
    return output


def threshold_first(comparisons, direction="output"):
    thresholds = {"0.99": {}, "0.90": {}, "0.50": {}, "0": {}}
    for source in SOURCES:
        rows = comparisons[source]
        # Gradient flows from the loss/head toward the input.  Report the
        # first threshold crossing in that reverse execution order, rather
        # than the dictionary's lexical/forward block order.
        ordered = ["head_bio"] + [f"block_{i}" for i in range(22, -1, -1)]
        for threshold_name, threshold in (("0.99", 0.99), ("0.90", 0.90), ("0.50", 0.50), ("0", 0.0)):
            first = None
            for name in ordered:
                cosine = rows[name][direction].get("cosine")
                if cosine is not None and cosine < threshold:
                    first = name
                    break
            thresholds[threshold_name][source] = first
    return thresholds


def main():
    cfg = make_config()
    seed_all(SEED)
    torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in detach_batch(raw).items()}
    post_data_rng = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    all_results = {"current_autocast": {}, "fp32_reference": {}}
    all_tensors = {"current_autocast": {}, "fp32_reference": {}}
    for source in SOURCES:
        current = one_backward_with_tensors(model, base_state, batch, post_data_rng, cfg, dataset, source, fp32=False)
        reference = one_backward_with_tensors(model, base_state, batch, post_data_rng, cfg, dataset, source, fp32=True)
        all_tensors["current_autocast"][source] = current.pop("_tensors")
        all_tensors["fp32_reference"][source] = reference.pop("_tensors")
        all_results["current_autocast"][source] = current
        all_results["fp32_reference"][source] = reference

    comparisons = {}
    for source in SOURCES:
        comparisons[source] = {}
        for boundary in all_tensors["current_autocast"][source]:
            comparisons[source][boundary] = {
                direction: compare_tensors(
                    all_tensors["current_autocast"][source][boundary][direction],
                    all_tensors["fp32_reference"][source][boundary][direction],
                ) for direction in ("input", "output")
            }

    # Twenty same-state current-path Reverse-1 replays.  The first run above is
    # kept separate because this replay is specifically a nondeterminism audit.
    replays = []
    for _ in range(REPEAT_COUNT):
        item = one_backward_with_tensors(model, base_state, batch, post_data_rng, cfg, dataset, "reverse1", fp32=False)
        replays.append(item)
    replay_summary = summarize_replay(replays)

    payload = {
        "phase": "2L-A/B",
        "config": {
            "dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE),
            "world_size_assumed": 2, "sampler": type(sampler).__name__, "sampler_length": len(sampler),
            "loss_sources": list(SOURCES), "reverse_indexing": "[:, i]", "nobio_index": [1, 2],
            "current_path": "model autocast enabled; loss arithmetic outside autocast",
            "fp32_reference": "autocast disabled; same model state/batch/RNG reset",
            "optimizer_step": False, "scheduler_step": False,
        },
        "batch": {"image_shape": list(batch["image"].shape), "pid": [int(x) for x in batch["pid"].cpu()], "camid": [int(x) for x in batch["camid"].cpu()], "clothes": [int(x) for x in batch["clothes_id"].cpu()]},
        "model": {"state_hashes": {name: tensor_sha(value) for name, value in base_state.items() if name in {"head.weight", "head_clip_bio.weight", "blocks.0.attn.qkv.weight"}}, "regular_block_count": len(model.blocks), "has_head_bio": hasattr(model, "head_bio")},
        "results": all_results,
        "fp16_vs_fp32": comparisons,
        "threshold_first_collapse_output": threshold_first(comparisons, "output"),
        "threshold_first_collapse_input": threshold_first(comparisons, "input"),
        "reverse1_replay": {"count": REPEAT_COUNT, "summary": replay_summary, "runs": [{"loss": x["loss"], "loss_dtype": x["loss_dtype"], "loss_finite": x["loss_finite"], "blocks": x["blocks"]} for x in replays]},
        "notes": [
            "Each loss source was independently recomputed and backwarded from the same restored model state and post-loader RNG state.",
            "Reverse-1 means the individual second-factor loss; it is not the released cumulative TensorBoard reverse_1 scalar.",
            "The full-FP32 mode is a mathematical oracle only; no optimizer update was performed.",
        ],
    }
    out = ROOT / "diagnostics/phase2l_layerwise_gradient_frontier.json"
    # Remove private tensor payloads before serialization; comparisons and
    # replay summaries contain the diagnostic information needed downstream.
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = ROOT / "reports/PHASE_2L_LAYERWISE_GRADIENT_FRONTIER.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["source", "boundary", "direction", "cosine", "relative_l2", "max_abs_diff", "fp16_l2", "fp32_l2", "fp16_nan", "fp16_inf", "fp32_nan", "fp32_inf"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for source, rows in comparisons.items():
            for boundary, directions in rows.items():
                for direction, row in directions.items():
                    writer.writerow({"source": source, "boundary": boundary, "direction": direction, "cosine": row.get("cosine"), "relative_l2": row.get("relative_l2"), "max_abs_diff": row.get("max_abs_diff"), "fp16_l2": row.get("a_l2"), "fp32_l2": row.get("b_l2"), "fp16_nan": row.get("a_nan_count"), "fp16_inf": row.get("a_inf_count"), "fp32_nan": row.get("b_nan_count"), "fp32_inf": row.get("b_inf_count")})

    print("PHASE2L_FRONTIER_SUMMARY", json.dumps({"output": str(out), "csv": str(csv_path), "sources": list(SOURCES), "regular_blocks": len(model.blocks), "reverse1_replays": REPEAT_COUNT}, sort_keys=True), flush=True)
    del model, dataset, loader, batch
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
