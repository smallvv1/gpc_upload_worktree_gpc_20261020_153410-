#!/usr/local/bin/python3
"""Minimal two-PPU forward/backward preflight for the NDP pipeline.

This file deliberately performs no optimizer step, writes no checkpoint, and
does not import a dataset or an evaluator.  It is a runtime gate only.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("NDP preflight requires visible PPU devices; CPU fallback is forbidden")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"NDP preflight requires exactly two visible PPUs, got {torch.cuda.device_count()}")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if any(name != "PPU-ZW810E" for name in names):
        raise RuntimeError(f"unexpected PPU device names: {names}")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2 or rank not in (0, 1):
        raise RuntimeError(f"torchrun must provide WORLD_SIZE=2 and LOCAL_RANK in {{0,1}}, got {world}/{rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        torch.manual_seed(20260930 + rank)
        torch.cuda.manual_seed_all(20260930 + rank)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(64, 8, bias=True),
        ).to(device)
        x = torch.randn(4, 32, device=device)
        target = torch.randn(4, 8, device=device)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        if output.device != device or loss.device != device:
            raise RuntimeError(f"forward tensors escaped PPU: output={output.device}, loss={loss.device}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(g.device != device for g in grads):
            raise RuntimeError("backward did not produce PPU-resident gradients")
        finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
        if not finite:
            raise RuntimeError("preflight backward produced non-finite gradients")
        checksum = torch.stack([g.detach().float().abs().mean() for g in grads]).sum()
        dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
        dist.barrier()
        print(
            f"NDP_PPU_PREFLIGHT_PASS rank={rank} world={world} device={device} "
            f"name={names[rank]} loss={float(loss.detach().cpu()):.8f} "
            f"grad_checksum={float(checksum.detach().cpu()):.8f} optimizer_step=False",
            flush=True,
        )
        return 0
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NDP_PPU_PREFLIGHT_FAIL type={type(exc).__name__} message={exc}", file=sys.stderr, flush=True)
        raise
