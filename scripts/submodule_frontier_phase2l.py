#!/usr/local/bin/python3
"""PHASE 2L-C: diagnostic submodule gradient frontier.

This is deliberately a no-step diagnostic.  It reuses one real LTCC batch and
one fresh biasless Full DIFFER state, then compares the current autocast path
with a full-FP32 oracle for Reverse-1 (the individual second-factor loss) and
the complete loss.  Only the two transition blocks selected from the
layerwise audit are instrumented; no formal model or loss source is changed.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

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
from layerwise_gradient_frontier_phase2l import (  # noqa: E402
    component_losses,
    compare_tensors,
    stat_tensor,
)
from loss.triplet_loss import TripletLoss  # noqa: E402


DEVICE = torch.device("cuda:0")
SUSPECT_BLOCKS = (10, 2)
SOURCES = ("reverse1", "full")


def _module_names(block: torch.nn.Module) -> list[str]:
    # Include only real modules, retaining the root block as a boundary.
    return [name for name, _module in block.named_modules()]


def install_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    saved: dict[str, dict[str, Any]] = {}
    handles = []
    for index in SUSPECT_BLOCKS:
        block = model.blocks[index]
        for suffix, module in block.named_modules():
            name = f"block_{index}" + (f".{suffix}" if suffix else "")
            saved[name] = {"module_class": type(module).__name__, "input": None, "output": None}

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


def _safe_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.grad is None:
        return None
    return value.grad.detach().cpu().clone()


def capture_one(model, base_state, batch, rng_state, source: str, fp32: bool):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    saved, handles = install_hooks(model)
    triplet = TripletLoss()
    with amp.autocast(enabled=not fp32):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, triplet)
    loss = losses[source]
    loss.backward()
    tensors = {
        name: {
            "input": _safe_clone(item["input"]),
            "output": _safe_clone(item["output"]),
        }
        for name, item in saved.items()
    }
    params = {}
    for pname, param in model.named_parameters():
        if pname.startswith(tuple(f"blocks.{i}." for i in SUSPECT_BLOCKS)):
            params[pname] = param.grad.detach().cpu().clone() if param.grad is not None else None
    summary = {
        "source": source,
        "precision": "fp32_reference" if fp32 else "current_autocast",
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "components": {k: {"value": float(v.detach().double().cpu()), "dtype": str(v.dtype)} for k, v in losses.items()},
        "modules": {
            name: {
                "class": item["module_class"],
                "input": stat_tensor(tensors[name]["input"]),
                "output": stat_tensor(tensors[name]["output"]),
                "input_grad_dtype": str(item["input"].grad.dtype) if isinstance(item["input"], torch.Tensor) and item["input"].grad is not None else None,
                "output_grad_dtype": str(item["output"].grad.dtype) if isinstance(item["output"], torch.Tensor) and item["output"].grad is not None else None,
            }
            for name, item in saved.items()
        },
        "parameters": {name: stat_tensor(value) for name, value in params.items()},
        "_tensors": tensors,
        "_params": params,
    }
    for handle in handles:
        handle.remove()
    del score, feat, loss, losses, triplet, saved
    return summary


def compare_capture(current, reference):
    out = {}
    for name in current["_tensors"]:
        out[name] = {
            direction: compare_tensors(current["_tensors"][name][direction], reference["_tensors"][name][direction])
            for direction in ("input", "output")
        }
    params = {}
    for name in current["_params"]:
        params[name] = compare_tensors(current["_params"][name], reference["_params"].get(name))
    return {"modules": out, "parameters": params}


def main():
    cfg = make_config()
    seed_all(SEED)
    torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    captures = {"current_autocast": {}, "fp32_reference": {}}
    comparisons = {}
    for source in SOURCES:
        cur = capture_one(model, base_state, batch, rng_state, source, fp32=False)
        ref = capture_one(model, base_state, batch, rng_state, source, fp32=True)
        comparisons[source] = compare_capture(cur, ref)
        for key, item in (("current_autocast", cur), ("fp32_reference", ref)):
            item.pop("_tensors")
            item.pop("_params")
            captures[key][source] = item

    payload = {
        "phase": "2L-C",
        "config": {
            "dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE),
            "sampler": type(sampler).__name__, "sampler_length": len(sampler),
            "suspect_blocks": list(SUSPECT_BLOCKS), "sources": list(SOURCES),
            "reverse_loss_semantics": "reverse1 is individual second-factor loss; not cumulative TensorBoard scalar",
            "current_path": "autocast model forward, loss outside autocast",
            "reference_path": "autocast disabled, same state/batch/RNG",
            "optimizer_step": False, "scheduler_step": False,
        },
        "module_tree": {f"block_{i}": _module_names(model.blocks[i]) for i in SUSPECT_BLOCKS},
        "captures": captures,
        "comparisons": comparisons,
        "notes": [
            "Hooks are observation-only and no optimizer/scheduler update is performed.",
            "Layerwise audit showed a smooth degradation, with Reverse-1 crossing 0.90 around block_10 and 0.50 around block_2; no single block was assumed to be causal.",
        ],
    }
    out = ROOT / "diagnostics/phase2l_submodule_frontier.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2L_SUBMODULE_FRONTIER.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["source", "kind", "name", "direction", "cosine", "relative_l2", "max_abs_diff", "fp16_l2", "fp32_l2", "fp16_nan", "fp16_inf", "fp32_nan", "fp32_inf"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for source, comp in comparisons.items():
            for kind in ("modules", "parameters"):
                entries = comp[kind]
                for name, dirs in entries.items():
                    if kind == "modules":
                        for direction, row in dirs.items():
                            writer.writerow({"source": source, "kind": kind, "name": name, "direction": direction, **{k: row.get(k) for k in fields[4:]}})
                    else:
                        row = dirs
                        writer.writerow({"source": source, "kind": kind, "name": name, "direction": "parameter", **{k: row.get(k) for k in fields[4:]}})
    print("PHASE2L_SUBMODULE_SUMMARY", json.dumps({"output": str(out), "csv": str(csv_path), "suspect_blocks": SUSPECT_BLOCKS, "sources": SOURCES}, sort_keys=True), flush=True)
    del model, dataset, loader, batch
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
