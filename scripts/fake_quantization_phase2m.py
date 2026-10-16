#!/usr/local/bin/python3
"""PHASE 2M-E: repeated block-boundary fake-half quantization diagnostic.

The run keeps every EVA block mathematically in FP32, but inserts an
explicit FP32 -> FP16 -> FP32 cast after each block output.  It is a
same-batch, no-optimizer-step diagnostic and is intentionally not wired into
the formal training entry point.
"""
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


def main() -> None:
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, _sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {
        key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    results = {}
    for policy in ("current", "FULL32", "FAKEQ"):
        print("PHASE2M_FAKEQ_POLICY", policy, flush=True)
        results[policy] = run_policy(model, base_state, batch, rng_state, policy)
    reference = results["FULL32"]
    comparisons = {
        policy: {
            source: {
                boundary: compare(
                    results[policy]["_gradient_tensors"][source][boundary],
                    reference["_gradient_tensors"][source][boundary],
                )
                for boundary in BOUNDARIES
            }
            for source in SOURCES
        }
        for policy in results
    }
    current_cos = comparisons["current"]["reverse1"]["block_0"].get("cosine")
    recovery = {}
    for policy in results:
        cosine = comparisons[policy]["reverse1"]["block_0"].get("cosine")
        recovery[policy] = None if cosine is None or current_cos is None else (cosine - current_cos) / (1.0 - current_cos)
    payload = {
        "phase": "2M-E",
        "candidate": "FAKEQ",
        "scope": "all EVA block/head_bio outputs: FP32 -> FP16 -> FP32 boundary cast; block arithmetic FP32",
        "config": {
            "dataset": "ltcc",
            "seed": 1234,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "reverse_indexing": "[:, i]",
            "nobio_index": [1, 2],
            "optimizer_step": False,
            "scheduler_step": False,
            "note": "boundary fake-quantization only; not a formal training path",
        },
        "losses": {policy: results[policy]["losses"] for policy in results},
        "comparisons_to_full_fp32": comparisons,
        "recovery_reverse1_block0_input": recovery,
        "notes": [
            "Current and FULL32 are rerun on the same frozen model state/batch for this diagnostic.",
            "FAKEQ does not invoke PPU FP16 block kernels; it isolates repeated boundary rounding in an FP32 mathematical path.",
        ],
    }
    out = ROOT / "diagnostics/phase2m_fake_quantization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "PHASE2M_FAKEQ_SUMMARY",
        json.dumps(
            {"json": str(out), "reverse1_block0_cosines": {p: comparisons[p]["reverse1"]["block_0"].get("cosine") for p in results}, "recovery": recovery},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
