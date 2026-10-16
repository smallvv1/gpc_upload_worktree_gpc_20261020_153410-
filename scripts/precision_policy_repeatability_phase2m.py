#!/usr/local/bin/python3
"""PHASE 2M-F: repeated same-state Reverse-1 backward diagnostics.

No optimizer or scheduler step is made.  Each repetition restores the same
model state and RNG state before one forward/autograd.grad pass.  This tests
whether a promising numerical policy changes PPU backward repeatability; it
does not constitute a training run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from global_precision_policy_screen_phase2m import (  # noqa: E402
    DEVICE,
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    run_policy,
    seed_all,
)


def pair_stats(values: list[torch.Tensor]) -> dict[str, Any]:
    if len(values) < 2:
        return {}
    cosines: list[float] = []
    rels: list[float] = []
    maxes: list[float] = []
    identical = 0
    total = 0
    for i in range(len(values)):
        a = values[i].double().reshape(-1)
        for j in range(i + 1, len(values)):
            b = values[j].double().reshape(-1)
            total += 1
            if torch.equal(values[i], values[j]):
                identical += 1
            an = float(torch.linalg.vector_norm(a).item())
            bn = float(torch.linalg.vector_norm(b).item())
            if an > 1e-12 and bn > 1e-12:
                cosines.append(float(F.cosine_similarity(a, b, dim=0).item()))
            rels.append(float(torch.linalg.vector_norm(a - b).item()) / (bn + 1e-12))
            maxes.append(float((a - b).abs().max().item()))
    return {
        "pairs": total,
        "cosine_min": min(cosines) if cosines else None,
        "cosine_mean": sum(cosines) / len(cosines) if cosines else None,
        "cosine_max": max(cosines) if cosines else None,
        "relative_l2_max": max(rels) if rels else None,
        "relative_l2_mean": sum(rels) / len(rels) if rels else None,
        "max_abs_diff_max": max(maxes) if maxes else None,
        "bitwise_identical_pairs": identical,
        "bitwise_identical_fraction": identical / total if total else None,
    }


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

    # B is the strongest policy in the broad screen; FAKEQ is retained as a
    # boundary-rounding diagnostic/ceiling and is never treated as a formal
    # training candidate.
    policies = ("current", "B", "FAKEQ")
    repeats = 20
    payload: dict[str, Any] = {
        "phase": "2M-F",
        "repeats": repeats,
        "source": "Reverse-1 individual loss",
        "boundary": "block_0 input gradient",
        "optimizer_step": False,
        "scheduler_step": False,
        "policies": {},
    }
    for policy in policies:
        values: list[torch.Tensor] = []
        losses: list[float] = []
        for index in range(repeats):
            print(f"PHASE2M_REPEAT {policy} {index + 1}/{repeats}", flush=True)
            result = run_policy(model, base_state, batch, rng_state, policy)
            values.append(result["_gradient_tensors"]["reverse1"]["block_0"].clone())
            losses.append(float(result["losses"]["reverse1"]["value"]))
        stats = pair_stats(values)
        payload["policies"][policy] = {
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_max_abs_pair_diff": max(abs(a - b) for i, a in enumerate(losses) for b in losses[i + 1:]) if len(losses) > 1 else 0.0,
            "gradient": stats,
        }
    out = ROOT / "diagnostics/phase2m_precision_policy_repeatability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE2M_REPEATABILITY_SUMMARY", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
