#!/usr/local/bin/python3
"""Run the single permitted medium-recovery AM diagnostic combination."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from global_precision_policy_screen_phase2m import (  # noqa: E402
    BOUNDARIES,
    DEVICE,
    SOURCES,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    compare,
    make_config,
    run_policy,
    seed_all,
)


def main():
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, _sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print("PHASE2M_POLICY AM", flush=True)
    am = run_policy(model, base_state, batch, rng_state, "AM")
    print("PHASE2M_POLICY FULL32", flush=True)
    ref = run_policy(model, base_state, batch, rng_state, "FULL32")
    comparisons = {
        source: {
            boundary: compare(am["_gradient_tensors"][source][boundary], ref["_gradient_tensors"][source][boundary])
            for boundary in BOUNDARIES
        }
        for source in SOURCES
    }
    # Current and reference values are read from the already completed matrix
    # so this run adds only the permitted AM combination.
    old = json.loads((ROOT / "diagnostics/phase2m_precision_policy_matrix.json").read_text())
    current_cos = old["comparisons_to_full_fp32"]["current"]["reverse1"]["block_0"].get("cosine")
    am_cos = comparisons["reverse1"]["block_0"].get("cosine")
    recovery = None if am_cos is None else (am_cos - current_cos) / (1.0 - current_cos)
    payload = {
        "phase": "2M-C-combination",
        "candidate": "AM",
        "scope": "A (FP32 QK^T/scaling/softmax) + M (all EVA MLP cores FP32)",
        "config": {"dataset": "ltcc", "seed": 1234, "batch_size": int(cfg.DATA.BATCH_SIZE), "optimizer_step": False, "scheduler_step": False},
        "losses": am["losses"],
        "comparisons_to_full_fp32": comparisons,
        "recovery_reverse1_block0_input": recovery,
        "notes": ["This is the one allowed combination of two individually medium-recovery categories; no long training was run."],
    }
    out = ROOT / "diagnostics/phase2m_precision_policy_AM.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2M_AM_SUMMARY", json.dumps({"json": str(out), "reverse1_block0_cosine": am_cos, "recovery": recovery}, sort_keys=True), flush=True)
    del model, dataset, loader, batch, am, ref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
