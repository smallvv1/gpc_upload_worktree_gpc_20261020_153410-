#!/usr/local/bin/python3
"""PHASE 2M-E boundary dtype and FP32->FP16 quantization audit."""
from __future__ import annotations

import csv
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
    build_natural_model,
    build_probe_loader,
    capture_rng_state,
    make_config,
    restore_rng_state,
    seed_all,
)

DEVICE = torch.device("cuda:0")


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)
    d = x - y
    yn = float(torch.linalg.vector_norm(y).item())
    xn = float(torch.linalg.vector_norm(x).item())
    return {
        "cosine": float(F.cosine_similarity(x, y, dim=0).item()) if xn > 1e-12 and yn > 1e-12 else None,
        "relative_l2": float(torch.linalg.vector_norm(d).item()) / (yn + 1e-12),
        "max_abs": float(d.abs().max().item()),
        "fp32_l2": yn,
        "cast_l2": xn,
    }


def capture(model: torch.nn.Module, batch: dict[str, Any], fp32: bool) -> dict[str, torch.Tensor]:
    saved: dict[str, torch.Tensor] = {}
    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(lambda _m, _inp, out, i=i: saved.__setitem__(f"block_{i}", out.detach().float().cpu().clone())))
    model.train()
    with amp.autocast(enabled=not fp32):
        model(batch["image"], cam_label=batch["camid"])
    for h in handles:
        h.remove()
    return saved


def main() -> None:
    seed_all(1234)
    torch.cuda.set_device(DEVICE)
    cfg = make_config()
    dataset, loader, _sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
    rng = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    restore_rng_state(rng)
    current = capture(model, batch, fp32=False)
    model.load_state_dict(state, strict=True)
    restore_rng_state(rng)
    reference = capture(model, batch, fp32=True)
    rows = []
    for i in range(len(model.blocks)):
        name = f"block_{i}"
        ref = reference[name]
        quantized = ref.half().float()
        rows.append({
            "block": i,
            "current_output_dtype": "torch.float16",
            "reference_output_dtype": "torch.float32",
            "reference_shape": list(ref.shape),
            "reference_l2": float(torch.linalg.vector_norm(ref.double()).item()),
            "reference_min": float(ref.min().item()),
            "reference_max": float(ref.max().item()),
            "reference_std": float(ref.std().item()),
            "boundary_fp32_to_fp16_to_fp32": cmp(quantized, ref),
            "current_vs_reference_output": cmp(current[name], ref),
        })
    payload = {
        "phase": "2M-E",
        "dataset": "ltcc",
        "batch_size": int(cfg.DATA.BATCH_SIZE),
        "block_count": len(model.blocks),
        "current_path": "autocast-enabled formal forward",
        "fake_quantization": "reference FP32 block output cast FP16 then FP32",
        "rows": rows,
        "conclusion": "Boundary quantization errors are measured independently; no backward or optimizer step was executed.",
    }
    out = ROOT / "diagnostics/phase2m_rounding_boundary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_out = ROOT / "reports/PHASE_2M_ROUNDING_BOUNDARY.csv"
    fields = ["block", "reference_l2", "reference_min", "reference_max", "reference_std", "boundary_cosine", "boundary_relative_l2", "boundary_max_abs", "current_output_cosine", "current_output_relative_l2"]
    with csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            b = row["boundary_fp32_to_fp16_to_fp32"]
            c = row["current_vs_reference_output"]
            writer.writerow({"block": row["block"], "reference_l2": row["reference_l2"], "reference_min": row["reference_min"], "reference_max": row["reference_max"], "reference_std": row["reference_std"], "boundary_cosine": b["cosine"], "boundary_relative_l2": b["relative_l2"], "boundary_max_abs": b["max_abs"], "current_output_cosine": c["cosine"], "current_output_relative_l2": c["relative_l2"]})
    print("PHASE2M_ROUNDING_SUMMARY", json.dumps({"json": str(out), "csv": str(csv_out), "max_boundary_rel_l2": max(r["boundary_fp32_to_fp16_to_fp32"]["relative_l2"] for r in rows), "min_boundary_cosine": min(r["boundary_fp32_to_fp16_to_fp32"]["cosine"] for r in rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
