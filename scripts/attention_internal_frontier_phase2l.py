#!/usr/local/bin/python3
"""PHASE 2L-C/D diagnostic for the internal EVA attention path.

The layerwise/submodule audit points to a gradual frontier, with a pronounced
Reverse-1 loss of fidelity around block 10 and the q path.  This script traces
q, k, v and the scaled-dot-product-attention output in blocks 10 and 2.  It
compares the normal PPU autocast execution with the same model/batch under a
full-FP32 mathematical reference.  It also replays the q-projection and SDPA
operators on captured input/upstream tensors, with no optimizer step.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
import types
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
from layerwise_gradient_frontier_phase2l import component_losses, compare_tensors, stat_tensor  # noqa: E402
from loss.triplet_loss import TripletLoss  # noqa: E402


DEVICE = torch.device("cuda:0")
TRACED_BLOCKS = (10, 2)
REPLAY_COUNT = 20


def traced_attention_factory(store: dict[str, Any], block_index: int):
    def traced_forward(self, x, rope=None, attn_mask=None):
        B, N, C = x.shape
        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        if rope is not None:
            # This model's LTCC path does not use RoPE, but retain the exact
            # official operation if a future config enables it.
            from model.eva_meta_bio import apply_rot_embed_cat
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)
        for name, value in (("q", q), ("k", k), ("v", v)):
            value.retain_grad()
            store[f"block_{block_index}.{name}"] = value
        if self.fused_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p)
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            y = attn @ v
        y.retain_grad()
        store[f"block_{block_index}.sdpa_output"] = y
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.norm(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y
    return traced_forward


def install_traces(model):
    original = {}
    stores: dict[str, dict[str, torch.Tensor]] = {}
    for index in TRACED_BLOCKS:
        attn = model.blocks[index].attn
        original[index] = attn.forward
        stores[f"block_{index}"] = {}
        attn.forward = types.MethodType(traced_attention_factory(stores[f"block_{index}"], index), attn)
    return original, stores


def restore_traces(model, original):
    for index, forward in original.items():
        model.blocks[index].attn.forward = forward


def capture(model, base_state, batch, rng_state, fp32: bool):
    model.load_state_dict(base_state, strict=True)
    restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    original, stores = install_traces(model)
    with amp.autocast(enabled=not fp32):
        score, feat = model(batch["image"], cam_label=batch["camid"])
    losses = component_losses(score, feat, batch, TripletLoss())
    loss = losses["reverse1"]
    loss.backward()
    tensors = {}
    for block_name, values in stores.items():
        tensors[block_name] = {}
        for name, value in values.items():
            tensors[block_name][name.rsplit(".", 1)[-1]] = {
                "value": value.detach().cpu().clone(),
                "grad": value.grad.detach().cpu().clone() if value.grad is not None else None,
            }
    q_param_grads = {}
    for index in TRACED_BLOCKS:
        for pname in ("q_proj.weight", "q_proj.bias", "k_proj.weight", "v_proj.weight"):
            key = f"blocks.{index}.attn.{pname}"
            p = dict(model.named_parameters()).get(key)
            q_param_grads[key] = p.grad.detach().cpu().clone() if p is not None and p.grad is not None else None
    restore_traces(model, original)
    return {
        "precision": "fp32_reference" if fp32 else "current_autocast",
        "loss": float(loss.detach().double().cpu()),
        "loss_dtype": str(loss.dtype),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "components": {k: {"value": float(v.detach().double().cpu()), "dtype": str(v.dtype)} for k, v in losses.items()},
        "tensors": tensors,
        "parameter_grads": q_param_grads,
    }


def compare_capture(cur, ref):
    out = {"tensors": {}, "parameter_grads": {}}
    for block, entries in cur["tensors"].items():
        out["tensors"][block] = {}
        for name, values in entries.items():
            out["tensors"][block][name] = {
                "forward": compare_tensors(values["value"], ref["tensors"][block][name]["value"]),
                "gradient": compare_tensors(values["grad"], ref["tensors"][block][name]["grad"]),
            }
    for name, value in cur["parameter_grads"].items():
        out["parameter_grads"][name] = compare_tensors(value, ref["parameter_grads"].get(name))
    return out


def replay_linear(module, x, upstream, count=REPLAY_COUNT):
    # Captured tensors are CPU; move to the PPU only for the isolated op.
    x0 = x.to(DEVICE)
    dy0 = upstream.to(DEVICE)
    weight = module.weight.detach().to(DEVICE)
    bias = module.bias.detach().to(DEVICE) if module.bias is not None else None
    current = []
    for _ in range(count):
        xi = x0.detach().clone().requires_grad_(True)
        wi = weight.detach().clone().requires_grad_(True)
        bi = bias.detach().clone().requires_grad_(True) if bias is not None else None
        with amp.autocast(enabled=True):
            y = F.linear(xi, wi, bi)
        grads = torch.autograd.grad(y, (xi, wi) if bi is None else (xi, wi, bi), dy0, retain_graph=False)
        current.append(grads)
    xi = x0.float().detach().requires_grad_(True)
    wi = weight.float().detach().requires_grad_(True)
    bi = bias.float().detach().requires_grad_(True) if bias is not None else None
    y = F.linear(xi, wi, bi)
    ref = torch.autograd.grad(y, (xi, wi) if bi is None else (xi, wi, bi), dy0.float(), retain_graph=False)
    return current, ref


def replay_sdpa(q, k, v, upstream, count=REPLAY_COUNT):
    q0, k0, v0, dy0 = (t.to(DEVICE) for t in (q, k, v, upstream))
    current = []
    for _ in range(count):
        qi = q0.detach().clone().requires_grad_(True)
        ki = k0.detach().clone().requires_grad_(True)
        vi = v0.detach().clone().requires_grad_(True)
        with amp.autocast(enabled=True):
            y = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        current.append(torch.autograd.grad(y, (qi, ki, vi), dy0, retain_graph=False))
    qi = q0.float().detach().requires_grad_(True)
    ki = k0.float().detach().requires_grad_(True)
    vi = v0.float().detach().requires_grad_(True)
    y = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
    ref = torch.autograd.grad(y, (qi, ki, vi), dy0.float(), retain_graph=False)
    return current, ref


def aggregate_replay(repeats, reference):
    result = []
    for index, label in enumerate(("dx", "dw", "db")):
        vals = [row[index].detach().cpu() for row in repeats]
        ref = reference[index].detach().cpu()
        comparisons = [compare_tensors(value, ref) for value in vals]
        repeat_cmp = [compare_tensors(vals[0], value) for value in vals[1:]]
        result.append({
            "gradient": label,
            "vs_fp32": {
                "cosine_mean": float(sum(x["cosine"] for x in comparisons) / len(comparisons)),
                "cosine_min": float(min(x["cosine"] for x in comparisons)),
                "relative_l2_mean": float(sum(x["relative_l2"] for x in comparisons) / len(comparisons)),
                "relative_l2_max": float(max(x["relative_l2"] for x in comparisons)),
                "max_abs_diff_max": float(max(x["max_abs_diff"] for x in comparisons)),
            },
            "repeat_fp16": {
                "cosine_min": float(min(x["cosine"] for x in repeat_cmp)),
                "relative_l2_max": float(max(x["relative_l2"] for x in repeat_cmp)),
                "max_abs_diff_max": float(max(x["max_abs_diff"] for x in repeat_cmp)),
                "bitwise_identical_percentage": 100.0 * sum(x["max_abs_diff"] == 0 for x in repeat_cmp) / len(repeat_cmp),
            },
        })
    return result


def main():
    cfg = make_config(); seed_all(SEED); torch.cuda.set_device(DEVICE)
    dataset, loader, sampler = build_probe_loader(cfg)
    raw = next(iter(loader))
    batch = {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    rng_state = capture_rng_state()
    model = build_natural_model(cfg, dataset.num_train_pids, dataset.num_camera, biasful=False).to(DEVICE)
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    cur = capture(model, base_state, batch, rng_state, fp32=False)
    ref = capture(model, base_state, batch, rng_state, fp32=True)
    comparisons = compare_capture(cur, ref)
    replay = {}
    # Isolate the q-projection GEMM and the fused attention operation at each
    # traced block.  Captured q/k/v gradients are the actual Reverse-1
    # upstream signals from the corresponding model execution.
    for index in TRACED_BLOCKS:
        key = f"block_{index}"
        q_in = None
        # q projection input is the input to norm1, obtainable from the
        # module-level trace by using the q tensor's predecessor is not
        # retained; recover it with a one-op hook on q_proj below.
        holder = {}
        original = model.blocks[index].attn.q_proj.forward
        def q_hook(module, x, output):
            holder["x"] = x[0].detach().cpu().clone()
            holder["y"] = output.detach().cpu().clone()
        handle = model.blocks[index].attn.q_proj.register_forward_hook(q_hook)
        # Re-run only once to capture q input and its actual dY.
        model.load_state_dict(base_state, strict=True); restore_rng_state(rng_state); model.zero_grad(set_to_none=True); model.train()
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, TripletLoss()); losses["reverse1"].backward()
        # q_proj output hook did not retain grad, so use a temporary tensor hook
        # through a second forward hook with retain_grad.
        handle.remove()
        # The captured q output value cannot provide dY after the graph is
        # gone; do a compact second capture with a retaining hook.
        holder = {}
        def q_hook2(module, x, output):
            x0 = x[0]; x0.retain_grad(); output.retain_grad()
            holder["x"] = x0; holder["y"] = output
        handle = model.blocks[index].attn.q_proj.register_forward_hook(q_hook2)
        model.zero_grad(set_to_none=True); restore_rng_state(rng_state)
        with amp.autocast(enabled=True):
            score, feat = model(batch["image"], cam_label=batch["camid"])
        losses = component_losses(score, feat, batch, TripletLoss()); losses["reverse1"].backward()
        x_cpu = holder["x"].detach().cpu().clone(); dy_cpu = holder["y"].grad.detach().cpu().clone()
        handle.remove()
        qmodule = model.blocks[index].attn.q_proj
        lin_repeats, lin_ref = replay_linear(qmodule, x_cpu, dy_cpu)
        sdpa = cur["tensors"][key]
        # Current traced q/k/v output gradients and values.
        sdpa_repeats, sdpa_ref = replay_sdpa(sdpa["q"]["value"], sdpa["k"]["value"], sdpa["v"]["value"], sdpa["sdpa_output"]["grad"])
        replay[key] = {
            "q_projection_input_shape": list(x_cpu.shape),
            "q_projection_output_shape": list(holder["y"].shape),
            "q_projection_linear": aggregate_replay(lin_repeats, lin_ref),
            "sdpa_shapes": {name: list(sdpa[name]["value"].shape) for name in ("q", "k", "v", "sdpa_output")},
            "sdpa": aggregate_replay(sdpa_repeats, sdpa_ref),
        }
    payload = {
        "phase": "2L-C/D",
        "config": {"dataset": "ltcc", "seed": SEED, "batch_size": int(cfg.DATA.BATCH_SIZE), "traced_blocks": list(TRACED_BLOCKS), "loss": "Reverse-1 individual factor", "optimizer_step": False, "scheduler_step": False},
        "module_tree": {f"block_{i}.attn": [name for name, _ in model.blocks[i].attn.named_modules()] for i in TRACED_BLOCKS},
        "current": {"loss": cur["loss"], "loss_dtype": cur["loss_dtype"], "loss_finite": cur["loss_finite"], "components": cur["components"]},
        "reference": {"loss": ref["loss"], "loss_dtype": ref["loss_dtype"], "loss_finite": ref["loss_finite"], "components": ref["components"]},
        "comparisons": comparisons,
        "operator_replay": replay,
        "notes": ["The traced forward is an exact diagnostic copy of EvaAttention; formal source files are untouched.", "q-projection replay uses the real q input and upstream dY from the current model graph; SDPA replay uses captured q/k/v and SDPA upstream dY.", "The FP32 path is a mathematical oracle, not a proposed formal training configuration."],
    }
    out = ROOT / "diagnostics/phase2l_attention_internal_frontier.json"; out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = ROOT / "reports/PHASE_2L_OPERATOR_REPLAY.csv"
    with csv_path.open("w", newline="") as h:
        fields = ["block", "operator", "gradient", "fp16_vs_fp32_cosine_mean", "fp16_vs_fp32_cosine_min", "fp16_vs_fp32_relative_l2_mean", "fp16_vs_fp32_relative_l2_max", "fp16_repeat_cosine_min", "fp16_repeat_relative_l2_max", "fp16_repeat_max_abs_diff", "fp16_bitwise_identical_pct"]
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for block, ops in replay.items():
            for op, rows in (("q_projection_linear", ops["q_projection_linear"]), ("sdpa", ops["sdpa"])):
                for row in rows:
                    w.writerow({"block": block, "operator": op, "gradient": row["gradient"], "fp16_vs_fp32_cosine_mean": row["vs_fp32"]["cosine_mean"], "fp16_vs_fp32_cosine_min": row["vs_fp32"]["cosine_min"], "fp16_vs_fp32_relative_l2_mean": row["vs_fp32"]["relative_l2_mean"], "fp16_vs_fp32_relative_l2_max": row["vs_fp32"]["relative_l2_max"], "fp16_repeat_cosine_min": row["repeat_fp16"]["cosine_min"], "fp16_repeat_relative_l2_max": row["repeat_fp16"]["relative_l2_max"], "fp16_repeat_max_abs_diff": row["repeat_fp16"]["max_abs_diff_max"], "fp16_bitwise_identical_pct": row["repeat_fp16"]["bitwise_identical_percentage"]})
    print("PHASE2L_ATTENTION_INTERNAL_SUMMARY", json.dumps({"output": str(out), "csv": str(csv_path), "traced_blocks": TRACED_BLOCKS}, sort_keys=True), flush=True)
    del model, dataset, loader, batch; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
