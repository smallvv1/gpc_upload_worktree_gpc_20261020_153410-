#!/usr/local/bin/python3
"""PHASE 2L-G: blockwise FP32 precision-island bisection (diagnostic only).

The layer/operator replays do not identify one isolated bad kernel.  This
script therefore tests the three prescribed contiguous quarter intervals of
the 23 EVA backbone blocks.  A selected interval is run with autocast disabled
and its output cast back to the surrounding dtype; every other operation is
the current official PPU path.  One real LTCC batch and one restored model
state are used for all cases, with no optimizer or scheduler step.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
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
    compare_tensors,
    component_losses,
    one_backward_with_tensors,
)


DEVICE = torch.device("cuda:0")


def install_island(model: torch.nn.Module, indices: set[int]):
    originals = {}
    for index in indices:
        block = model.blocks[index]
        original = block.forward
        originals[index] = original

        def fp32_forward(*args, _original=original, **kwargs):
            x = args[0]
            input_dtype = x.dtype
            # This is the smallest block-level diagnostic island available
            # when no unique internal operator has been established.  It does
            # not change parameters, objective, or the surrounding autocast
            # graph; only this block's arithmetic is explicitly FP32.
            with amp.autocast(enabled=False):
                out = _original(x.float(), *args[1:], **kwargs)
            return out.to(input_dtype) if isinstance(out, torch.Tensor) else out

        block.forward = fp32_forward
    return originals


def remove_island(model, originals):
    for index, original in originals.items():
        model.blocks[index].forward = original


def summarize_boundaries(tensors, reference):
    out = {}
    for name in tensors:
        out[name] = {
            direction: compare_tensors(tensors[name][direction], reference[name][direction])
            for direction in ("input", "output")
        }
    return out


def main():
    cfg = make_config(); seed_all(SEED); torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    nblocks = len(model.blocks)

    # Quarter intervals are intentionally contiguous and disjoint.  The final
    # interval absorbs the remainder so every block is represented once.
    q = nblocks // 4
    intervals = {
        "first_quarter": set(range(0, q + (1 if nblocks % 4 else 0))),
        "middle_quarter": set(range(nblocks // 2 - q // 2, nblocks // 2 - q // 2 + q)),
        "last_quarter": set(range(nblocks - q, nblocks)),
    }

    # Current and mathematical full-FP32 references provide the two anchors.
    current = one_backward_with_tensors(model, base_state, batch, rng_state, cfg, dataset, "reverse1", fp32=False)
    reference = one_backward_with_tensors(model, base_state, batch, rng_state, cfg, dataset, "reverse1", fp32=True)
    current_tensors = current.pop("_tensors")
    reference_tensors = reference.pop("_tensors")
    results: dict[str, Any] = {
        "current_autocast": {"summary": current, "vs_fp32": summarize_boundaries(current_tensors, reference_tensors)},
        "full_fp32_reference": {"summary": reference},
        "islands": {},
    }
    for name, indices in intervals.items():
        model.load_state_dict(base_state, strict=True)
        restore_rng_state(rng_state)
        originals = install_island(model, indices)
        candidate = one_backward_with_tensors(model, base_state, batch, rng_state, cfg, dataset, "reverse1", fp32=False)
        tensors = candidate.pop("_tensors")
        results["islands"][name] = {
            "blocks": sorted(indices),
            "summary": candidate,
            "vs_fp32": summarize_boundaries(tensors, reference_tensors),
            "vs_current": summarize_boundaries(tensors, current_tensors),
        }
        remove_island(model, originals)

    payload = {
        "phase": "2L-G",
        "config": {"dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE), "regular_block_count": nblocks, "loss": "Reverse-1 individual factor", "optimizer_step": False, "scheduler_step": False, "island_semantics": "selected blocks execute under autocast disabled on float input, output cast to surrounding input dtype"},
        "results": results,
        "notes": ["This bisection is allowed only because layerwise/operator audits did not establish one unique collapse operator.", "It is a diagnostic precision island, not a formal training patch."],
    }
    out = ROOT / "diagnostics/phase2l_blockwise_precision_bisection.json"; out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2L_BLOCKWISE_PRECISION_BISECTION.csv"
    with csv_path.open("w", newline="") as h:
        fields = ["case", "blocks", "boundary", "direction", "cos_vs_fp32", "rel_l2_vs_fp32", "cos_vs_current", "rel_l2_vs_current"]
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for case, body in [("current_autocast", results["current_autocast"])] + list(results["islands"].items()):
            for boundary, dirs in body["vs_fp32"].items():
                for direction, row in dirs.items():
                    currow = body.get("vs_current", {}).get(boundary, {}).get(direction, {})
                    w.writerow({"case": case, "blocks": ",".join(map(str, body.get("blocks", []))), "boundary": boundary, "direction": direction, "cos_vs_fp32": row.get("cosine"), "rel_l2_vs_fp32": row.get("relative_l2"), "cos_vs_current": currow.get("cosine"), "rel_l2_vs_current": currow.get("relative_l2")})
    anchors = {"current": {b: results["current_autocast"]["vs_fp32"][b]["input"] for b in ("block_0", "block_2", "block_10", "block_22")}, "islands": {name: {b: body["vs_fp32"][b]["input"] for b in ("block_0", "block_2", "block_10", "block_22")} for name, body in results["islands"].items()}}
    print("PHASE2L_BLOCKWISE_SUMMARY", json.dumps({"output": str(out), "csv": str(csv_path), "intervals": {k: sorted(v) for k, v in intervals.items()}, "anchors": anchors}, sort_keys=True), flush=True)
    del model, dataset, loader, batch; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
