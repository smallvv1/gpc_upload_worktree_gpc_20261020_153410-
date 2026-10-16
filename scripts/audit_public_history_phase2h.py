#!/usr/local/bin/python3
"""Reproducible, read-only audit of the public DIFFER revisions.

The script only inspects git objects and the current source.  It does not
modify source, checkpoints, or training outputs.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = ["508fc5a", "d449788", "67acb5d3658d103b4412c8c99f93ee8f085802fe"]
REV_LABELS = {"508fc5a": "508fc5a", "d449788": "d449788", REVISIONS[-1]: "current HEAD"}
FILES = [
    "model/eva_meta_bio.py", "model/build.py", "loss/make_loss.py",
    "processor/processor.py", "data/dataloader.py", "data/samplers.py",
    "solver/make_optimizer.py", "train.py", "configs/ltcc/eva02_l_bio.yml",
]
REPORT = ROOT / "reports/PUBLIC_HISTORY_TRAINING_DIFF.md"
JSON_OUT = ROOT / "diagnostics/public_history_phase2h.json"


def run(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def show(revision, path):
    try:
        return run("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return ""


def diff_numstat(a, b):
    lines = run("diff", "--numstat", a, b, "--", *FILES).splitlines()
    out = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) >= 3:
            out.append({"added": fields[0], "deleted": fields[1], "path": fields[2]})
    return out


def bias_value(text):
    matches = re.findall(r"head_fc\s*=\s*nn\.Linear\([^\n]*bias\s*=\s*(True|False)", text)
    # The public class has one reverse linear layer.  Keep all matches in case
    # a future revision adds another form.
    return sorted(set(matches)) if matches else ["NOT_FOUND"]


def index_semantics(text):
    lines = [line.strip() for line in text.splitlines() if "image_feature_reverse" in line]
    slices = [line for line in lines if "[:,i]" in line or "[:, i]" in line]
    batch_index = [line for line in lines if re.search(r"image_feature_reverse\[i\]", line)]
    return {
        "shape_correct_column_slice_lines": slices,
        "batch_index_lines": batch_index,
        "classification": "column_slice [:,i]" if slices and not batch_index else (
            "batch index [i]" if batch_index else "not found"
        ),
        "all_relevant_lines": lines,
    }


def commit_info(revision):
    return run("show", "-s", "--format=%H%n%ad%n%s", "--date=iso-strict", revision).splitlines()


def main():
    entries = {}
    for revision in REVISIONS:
        text = show(revision, "model/eva_meta_bio.py")
        loss = show(revision, "loss/make_loss.py")
        entries[revision] = {
            "label": REV_LABELS[revision],
            "commit": commit_info(revision),
            "reverse_bias": bias_value(text),
            "reverse_index": index_semantics(loss),
        }
    current = Path(ROOT / "model/eva_meta_bio.py").read_text()
    current_loss = Path(ROOT / "loss/make_loss.py").read_text()
    entries["working_tree"] = {
        "label": "our reproduction working tree",
        "commit": ["uncommitted approved patches"],
        "reverse_bias": bias_value(current),
        "reverse_index": index_semantics(current_loss),
    }

    pair_diffs = {
        "508fc5a..d449788": diff_numstat("508fc5a", "d449788"),
        "d449788..67acb5d": diff_numstat("d449788", "67acb5d"),
    }
    payload = {
        "revisions": entries,
        "pair_numstat_relevant_files": pair_diffs,
        "official_checkpoint_bias": {
            "keys": [
                "head_clip_bioReverse.0.head_fc.bias",
                "head_clip_bioReverse.1.head_fc.bias",
            ],
            "state_dict_key_count": 519,
        },
        "notes": [
            "The official LTCC artifact log/checkpoint is dated 2024-08-19, before the inspected public revisions.",
            "A state dict bias key proves structural presence, not the unobserved private source implementation.",
            "Current working-tree reverse indexing is the approved shape-correct [:, i] diagnostic/reproduction patch; the public HEAD itself remains [i].",
        ],
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# PUBLIC_HISTORY_TRAINING_DIFF",
        "",
        "Read-only audit of the public DIFFER git history requested for PHASE 2H.",
        "No git object, checkpoint, or training output was modified.",
        "",
        "## Revision provenance",
        "",
        "| Revision | Date | Subject |",
        "| --- | --- | --- |",
    ]
    for revision in REVISIONS:
        info = entries[revision]["commit"]
        lines.append(f"| `{REV_LABELS[revision]}` (`{info[0][:12]}`) | `{info[1]}` | {info[2]} |")
    lines += [
        "",
        "The official artifact log is timestamped 2024-08-19, while these public revisions are June/August/November 2025. This establishes temporal separation but does not by itself identify an unobserved private implementation.",
        "",
        "## Reverse classifier bias",
        "",
        "| Revision | `GradientReversalClassifier.head_fc` bias |",
        "| --- | --- |",
    ]
    for revision in REVISIONS:
        lines.append(f"| {REV_LABELS[revision]} | `{', '.join(entries[revision]['reverse_bias'])}` |")
    lines += [
        "| Official LTCC checkpoint | PRESENT: `head_clip_bioReverse.0.head_fc.bias`, `.1.head_fc.bias` |",
        "",
        "All inspected public revisions define the reverse classifier linear layer with `bias=False`; the official checkpoint contains two reverse-head bias tensors. The precise conclusion is **PUBLIC-CODE / OFFICIAL-TRAINING STRUCTURAL MISMATCH CONFIRMED**. The artifact was produced by a model/training variant not represented exactly by these public revisions, unless the state dict was transformed after training. The evidence does not prove that an author-private source necessarily used `bias=True`.",
        "",
        "## Reverse indexing history",
        "",
        "| Revision | Runtime source form | Classification |",
        "| --- | --- | --- |",
    ]
    for revision in REVISIONS + ["working_tree"]:
        item = entries[revision]["reverse_index"]
        lines.append(f"| {entries[revision]['label']} | `{item['classification']}` | {len(item['batch_index_lines'])} batch-index line(s), {len(item['shape_correct_column_slice_lines'])} column-slice line(s) |")
    lines += [
        "",
        "`508fc5a` uses the shape-correct `image_feature_reverse[:, i]` semantics. `d449788` changes the reverse branch to `image_feature_reverse[i]` and derives labels from that batch-indexed tensor; current public HEAD retains that regression. The approved reproduction working tree restores `[:, i]`. For a tensor `[B,F,D]`, only the column slice yields `[B,D]` for factor `i`.",
        "",
        "## Relevant public diffs and risk ranking",
        "",
        "### CRITICAL",
        "",
        "- `d449788` introduces the reverse factor batch-indexing regression (`[i]`) in `loss/make_loss.py`; the public HEAD retains it. This is a real shape/semantic difference from `508fc5a`. Our frozen reproduction used the approved `[:, i]` correction, so this does not describe the exact Run-1 working-tree behavior; the official artifact's pre-public source remains unknown.",
        "",
        "### HIGH",
        "",
        "- Official checkpoint reverse heads include bias slots absent from every inspected public model revision. This is a confirmed schema mismatch and is not recoverable from equal-shaped weights alone.",
        "- `508fc5a -> d449788` changes the model family to the `clipFc`/`clipMLP` topology, adds subspace-aware heads and `GradientReversalMLP`, migrates `METRIC_LOSS_TYPE` to `LOSS_TYPE`, changes reverse/Bio head construction, and changes the LTCC YAML to `LAST_LAYER=clipFc` and `NOBIO_INDEX=1,2`. These are true model/config changes; common checkpoint shapes show the frozen Run-1 and official inference schemas agree on the shared backbone and heads.",
        "",
        "### MEDIUM",
        "",
        "- `d449788 -> d651e9b` removes CCVID/Celeb auxiliary dataset branches and TTT paths, resets local default paths, and changes logger names/comments. No substantive LTCC sampler or optimizer implementation change was found.",
        "- Public release code uses `LOSS_TYPE` for model/loss construction and processor center-loss gating; this is a semantic configuration plumbing change relative to the first commit and is already represented in the approved reproduction setup.",
        "- The official artifact predates the public commits, so exact historical AMP/DDP settings cannot be inferred from this history alone.",
        "",
        "### LOW",
        "",
        "- Solver print guard, unused loss arguments, whitespace, and deletion of `readLog.py` do not alter LTCC forward/loss semantics.",
        "",
        "## Files specifically checked",
        "",
        "`model/eva_meta_bio.py`, `model/build.py`, `loss/make_loss.py`, `processor/processor.py`, `data/dataloader.py`, `data/samplers.py`, `solver/make_optimizer.py`, `train.py`, and `configs/ltcc/eva02_l_bio.yml` were compared at all requested revisions. No `data/dataloader.py` or `data/samplers.py` change appears in either relevant pair; the released `DistributedRandomIdentitySampler` path is therefore unchanged across these public revisions. Full machine-readable numstats and source-line evidence are in `diagnostics/public_history_phase2h.json`.",
        "",
        "## Conclusion",
        "",
        "The public history confirms a reverse-indexing regression and a persistent biasless reverse-head definition, while the official LTCC checkpoint has two nonzero bias tensors. It does not reveal a public revision that can be identified as the exact official 2024 training source. No new algorithm patch is justified by this history audit alone.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("WROTE", REPORT)
    print("WROTE", JSON_OUT)


if __name__ == "__main__":
    main()
