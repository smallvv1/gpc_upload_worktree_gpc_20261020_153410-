#!/usr/local/bin/python3
"""Render the persisted Phase 2I diagnostics into a human-readable report."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("phase2i", ROOT / "scripts/initialization_footprint_phase2i.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_arrays(module, rows):
    output = {}
    for name, tag in module.SCALAR_MAP.items():
        if tag == "loss/clip_bio_0":
            tag = "loss/clip_bio"
        elif tag == "acc/clip_bio_0":
            tag = "acc/clip_bio"
        output[name] = [row.get(tag) for row in rows]
    return output


def candidate_arrays(result):
    return {name: [record["values"].get(name) for record in result["records"]] for name in [
        "total_loss", "id_loss", "triplet_loss", "bio_loss", "reverse0_loss", "reverse1_loss",
        "id_acc", "bio_acc", "reverse0_acc", "reverse1_acc",
    ]}


def distance(module, official, candidate):
    out = {}
    for name in candidate:
        pairs = [(float(a), float(b)) for a, b in zip(official.get(name, []), candidate[name]) if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
        if not pairs:
            continue
        a = np.asarray([p[0] for p in pairs]); b = np.asarray([p[1] for p in pairs]); d = b-a
        std = float(np.std(a))
        out[name] = {
            "rmse": float(np.sqrt(np.mean(d*d))),
            "mae": float(np.mean(np.abs(d))),
            "nrmse": float(np.sqrt(np.mean(d*d))/(std+1e-12)),
            "pearson": float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>1e-15 and np.std(b)>1e-15 else None,
        }
    vals = [item["nrmse"] for item in out.values()]
    out["composite"] = float(np.mean(vals)) if vals else None
    return out


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if abs(float(value)) < 0.001 and value != 0:
        return f"{float(value):.3e}"
    return f"{float(value):.{digits}f}"


def main():
    module = load_module()
    main = json.loads((ROOT / "diagnostics/initialization_footprint_phase2i.json").read_text())
    tb = json.loads((ROOT / "diagnostics/official_tensorboard_phase2i.json").read_text())
    probe = json.loads((ROOT / "diagnostics/initialization_footprint_15epoch_phase2i.json").read_text())
    lines = []
    add = lines.append
    add("# PHASE 2I — HISTORICAL INITIALIZATION FOOTPRINT")
    add("")
    add("本报告只包含诊断性自然构造/轨迹实验；没有启动新的正式 60-epoch、PRCC 或 Run-2/Run-3。")
    add("")
    add("## 1. RNG construction trace")
    add("")
    add("- seed: `1234`; rank stream: rank 0 / world size 2; AMP initial scale: `1024`.")
    add("- isolated `nn.Linear(1024,512)`: bias=False 与 bias=True 的 CPU RNG 状态不同；CUDA/NumPy/Python 状态相同。")
    add("- first A/B/C divergence: **T2_after_build_model** (CPU RNG)。T0/T1 相同；T3 与 T2 相同，说明 pretrained load 本身没有继续消耗 RNG。")
    add("")
    add("| Candidate | reverse construction | parameter keys | T2 CPU hash (prefix) | T4 CPU hash (prefix) |")
    add("| --- | --- | ---: | --- | --- |")
    for label in "ABC":
        c = main["candidates"][label]
        add(f"| {label} | {'bias=True' if label in 'BC' else 'bias=False'} | {len(c.get('optimizer_groups', []))} | {c['rng_trace']['T2_after_build_model']['torch_cpu'][:16]} | {c['rng_trace']['T4_before_first_train_loader_iteration']['torch_cpu'][:16]} |")
    add("")
    sc = main["state_comparison_constructed"]
    add(f"A/B common parameters: `{sc['common_count']}`; bitwise identical `{sc['a_b_identical_count']}`; different `{sc['a_b_different_count']}`. B/C common parameters: `{sc['b_c_common_count']}`; bitwise different `{sc['b_c_different_count']}`.")
    add(f"A/B differing common names: `{', '.join(sc['a_b_different_names_first100'])}`.")
    add("")
    add("### Focused A → B initialization differences")
    add("")
    add("| Parameter | max abs | L2 | relative L2 | cosine |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for name, item in sc["focus_differences_a_vs_b"].items():
        add(f"| `{name}` | {fmt(item['max_abs'])} | {fmt(item['l2'])} | {fmt(item['relative_l2'])} | {fmt(item['cosine'])} |")
    add("")
    add("All trainable master parameters in A/B/C are FP32; no unexpected FP16/BF16 master parameters were found. Data sample-order and augmented-image fingerprints are identical for all 200 batches (A/B, A/C, B/C). Therefore construction RNG did not couple into this loader's sampler/augmentation stream.")
    add("")
    add("### Optimizer ordering")
    add("")
    add("A has 517 one-parameter groups. B has 519; each native reverse bias is inserted after its corresponding reverse weight and receives LR `4e-6`, weight decay `0.05` under the released bias rule. C freezes both biases before its actual optimizer is created, so the C training optimizer excludes them; its common-parameter order otherwise follows B. This is an ordering/topology diagnostic only.")
    add("")
    add("## 2. Official TensorBoard forensic")
    add("")
    add("Scalar tags: `" + "`, `".join(tb["files"][next(iter(tb["files"]))]["tags"]) + "`.")
    add("")
    add("| Event stream | train steps | metric steps | loss/total first→last | loss/total min→max | rank1 first→last | mAP first→last |")
    add("| --- | ---: | ---: | --- | --- | --- | --- |")
    for name, item in tb["files"].items():
        s = item["scalar_summary"]
        lt, r1, ma = s["loss/total"], s["rank1"], s["mAP"]
        add(f"| `{name}` | {lt['record_count']} (`{lt['first_step']}..{lt['last_step']}`) | 61 (`0..60`) | {fmt(lt['first_value'])} → {fmt(lt['last_value'])} | {fmt(lt['min'])} → {fmt(lt['max'])} | {fmt(r1['first_value'])} → {fmt(r1['last_value'])} | {fmt(ma['first_value'])} → {fmt(ma['last_value'])} |")
    add("")
    add("Interpretation: the two files are **one continuous 60-epoch run with rank-local scalar streams**, not duplicate files. Each has 35,760 training records (`596×60`), steps 596–36,355, and evaluation steps 0–60. Reverse tags are present for both factors; `reverse_0` is factor-0 individual and `reverse_1` is cumulative factor-0 + factor-1 under the released logging semantics.")
    add("")
    add("## 3. Natural 200-update A/B/C fingerprint")
    add("")
    add("A = current biasless; B = native biasful/trainable; C = native biasful construction with zero/frozen biases. Each was independently seeded and constructed; no common initialization was forced.")
    add("")
    add("| Candidate | skipped | nonfinite loss | nonfinite gradient iterations | successful updates | final scale | composite score vs stream 643878 | composite score vs stream 643879 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    stream_scores = {name: {} for name in tb["files"]}
    for name, rows in tb["first200_by_file"].items():
        off = official_arrays(module, rows)
        for label in "ABC":
            stream_scores[name][label] = distance(module, off, candidate_arrays(main["results"][label]))
    for label in "ABC":
        res = main["results"][label]
        scores = [stream_scores[name][label]["composite"] for name in tb["files"]]
        add(f"| {label} | {res['skipped_steps']} | {res['nonfinite_loss_count']} | {res['nonfinite_gradient_iterations']} | {res['successful_updates']} | {fmt(res['final_scale'])} | {fmt(scores[0])} | {fmt(scores[1])} |")
    add("")
    add("The 200-step ranking is stream-dependent because the official event files are rank-local: stream `643878` ranks C slightly ahead of A, while stream `643879` ranks A slightly ahead. The branch-specific pattern is more informative: reverse-1 accuracy normalized RMSE is lower for C than A on both streams (approximately 1.462 vs 1.543 and 1.461 vs 1.472); reverse-1 loss is also lower for C on both. B/C are therefore not uniformly closer on every scalar, but the earliest reverse branch consistently moves toward the official trajectory.")
    add("")
    add("## 4. Fifteen-epoch targeted A/C probe")
    add("")
    add("Because B≈C in the 200-step comparison, C was paired with A to isolate construction footprint. Both runs stopped at epoch 15; no checkpoint was continued.")
    add("")
    add("| Candidate | epoch-15 CC R1 | epoch-15 CC mAP | total skipped | nonfinite loss | nonfinite gradient iterations | scale sequence (end of epoch) |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for label in "AC":
        res = probe["results"][label]
        last = res["records"][-1]
        seq = ", ".join(str(int(r["amp_scale_end"])) for r in res["records"])
        add(f"| {label} | {last['cc']['rank1']*100:.3f}% | {last['cc']['mAP']*100:.3f}% | {res['total_skipped']} | {res['total_nonfinite_loss']} | {res['total_nonfinite_gradient']} | `{seq}` |")
    add("")
    add("For both official rank-local streams, C is substantially closer than A on reverse-1 accuracy and reverse-1 loss over epochs 1–15; C also improves reverse-0/bio/triplet branch distances in the stored comparison. A remains closer on CC R1/mAP and slightly on ID trajectory. Thus the construction footprint is supported as an early reverse-branch contributor, but it does not account for the full retrieval gap by itself.")
    add("")
    add("| Metric (15-epoch RMSE) | Official stream 643878 A | 643878 C | Official stream 643879 A | 643879 C |")
    add("| --- | ---: | ---: | ---: | ---: |")
    # Recompute epoch-level distances using the two raw streams and stored A/C
    # results.  The compact result JSON already stores the 643878 comparison;
    # these values are retained explicitly for the rank-local audit.
    epoch_rows_by_file = {}
    for fname, item in tb["files"].items():
        s = item["series"]
        epoch_rows = {}
        for epoch in range(1, 16):
            lo, hi = 596 + (epoch-1)*596, 596 + epoch*596
            row = {}
            for tag, key in [("loss/total","train_loss"),("loss/id","id_loss"),("loss/triplet","triplet_loss"),("loss/clip_bio","bio_loss"),("loss/clip_bio_reverse_0","reverse0_loss"),("loss/clip_bio_reverse_1","reverse1_loss"),("acc/id","id_acc"),("acc/clip_bio","bio_acc"),("acc/clip_bio_reverse_0","reverse0_acc"),("acc/clip_bio_reverse_1","reverse1_acc")]:
                vals=[x["value"] for x in s[tag] if lo<=x["step"]<hi]
                row[key]=float(np.mean(vals)) if vals else None
            row["cc_rank1"] = s["rank1"][epoch]["value"]
            row["cc_mAP"] = s["mAP"][epoch]["value"]
            epoch_rows[epoch]=row
        epoch_rows_by_file[fname]=epoch_rows
    for metric in ["reverse1_acc","reverse1_loss","reverse0_acc","bio_acc","triplet_loss","cc_rank1","cc_mAP"]:
        vals=[]
        for fname in tb["files"]:
            d=[]
            for label in "AC":
                a=np.asarray([epoch_rows_by_file[fname][e][metric] for e in range(1,16)],float)
                rr=probe["results"][label]["records"]
                b=np.asarray([rr[e-1]["cc"]["rank1"] if metric=="cc_rank1" else rr[e-1]["cc"]["mAP"] if metric=="cc_mAP" else (sum((rr[e-1].get(k) or 0) for k in (["id_loss","triplet_loss","bio_loss","reverse1_loss"] if metric=="train_loss" else [metric])) if metric=="train_loss" else rr[e-1].get(metric)) for e in range(1,16)],float)
                d.append(float(np.sqrt(np.mean((a-b)**2))))
            vals.extend(d)
        add(f"| `{metric}` | {fmt(vals[0],6)} | {fmt(vals[1],6)} | {fmt(vals[2],6)} | {fmt(vals[3],6)} |")
    add("")
    add("## 5. Platform context")
    add("")
    add("- pip torch: `2.4.0+ppu1.5.0.oe`; runtime `torch.__version__` reports `2.4.0`; devices: 2× PPU-ZW810E.")
    add("- timm: `0.9.7`; xformers: `0.0.27`; tensorboard: `2.21.0`; protobuf: `6.33.6`.")
    add("- `timm.layers.use_fused_attn()` is `True`; released EVA bio factory sets `qkv_fused=False`, while `EvaAttention.forward` selects `F.scaled_dot_product_attention` when `fused_attn` is true. A fused/unfused numerical comparison was not promoted to the primary experiment because the historical-construction candidate passed the branch-trajectory gate.")
    add("")
    add("## 6. Final decision")
    add("")
    add("**HISTORICAL INITIALIZATION FOOTPRINT SUPPORTED**")
    add("")
    add("Evidence supports the narrow claim that a native `bias=True` reverse-head constructor changes the random initialization footprint and that the resulting C/A trajectory is systematically closer on the earliest reverse branch. It does not prove that this is the sole cause of the 53.3/29.3 retrieval result: rank-local official streams disagree on aggregate 200-step ranking, and A is closer on CC retrieval during the 15-epoch probe. No model/loss/optimizer/evaluator source was patched and no formal 60-epoch run was started. Next work should remain a training-path audit; PRCC remains untouched.")
    out = ROOT / "reports/PHASE_2I_INITIALIZATION_FORENSICS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()

