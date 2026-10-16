#!/usr/local/bin/python3
"""Read-only strict schema audit for the official LTCC checkpoint.

The repository HEAD reverse heads use bias=False, whereas the historical
official LTCC artifact contains two reverse bias tensors.  The adapter below
is process-local and evaluation-only; it lets this audit perform a strict
load without changing source, checkpoint, or training behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg as GLOBAL_CFG
from data import build_dataset
from model import build_model
from model import eva_meta_bio


CONFIG = REPO_ROOT / "reports/FORENSIC_RECONSTRUCTED_LTCC_CONFIG_V2.yml"
OFFICIAL = Path("/home/data/DIFFER_OfficialWeights/LTCC/eva02_l_bio_best.pth")
OURS = REPO_ROOT / (
    "outputs/reproduction_phase2/LTCC/differ/run1/"
    "1[ltcc][eva02_l_bio][PHASE2_LTCC_DIFFER_RUN1]/eva02_l_bio_best.pth"
)
OUTPUT = REPO_ROOT / "diagnostics/official_ltcc_schema_phase2g_a3.json"


class EvaluationOnlyReverseAdapter(nn.Module):
    def __init__(self, in_num: int, class_num: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)

    def forward(self, x):
        return self.head_fc(self.grl(x))


def normalize(state):
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def schema(state):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in state.items()}


def main() -> None:
    local_cfg = GLOBAL_CFG.clone()
    local_cfg.merge_from_file(str(CONFIG))
    local_cfg.defrost()
    local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    local_cfg.freeze()

    dataset = build_dataset(local_cfg)
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera)
    model_keys_before_adapter = len(model.state_dict())
    adapters = 0
    for index, old_head in enumerate(model.head_clip_bioReverse):
        if old_head.head_fc.bias is None:
            fc = old_head.head_fc
            model.head_clip_bioReverse[index] = EvaluationOnlyReverseAdapter(
                fc.in_features, fc.out_features, old_head.alpha
            )
            adapters += 1
    model_keys_after_adapter = len(model.state_dict())

    official_state = normalize(torch.load(OFFICIAL, map_location="cpu"))
    model_state = model.state_dict()
    missing, unexpected = model.load_state_dict(official_state, strict=True)
    model_shape_mismatch = []
    for key, value in official_state.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            model_shape_mismatch.append({
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(model_state[key].shape),
            })

    ours_state = normalize(torch.load(OURS, map_location="cpu"))
    official_keys = set(official_state)
    ours_keys = set(ours_state)
    common = official_keys & ours_keys
    ours_schema_mismatch = []
    for key in sorted(common):
        if tuple(official_state[key].shape) != tuple(ours_state[key].shape) or official_state[key].dtype != ours_state[key].dtype:
            ours_schema_mismatch.append({
                "key": key,
                "official_shape": list(official_state[key].shape),
                "ours_shape": list(ours_state[key].shape),
                "official_dtype": str(official_state[key].dtype),
                "ours_dtype": str(ours_state[key].dtype),
            })

    payload = {
        "config": str(CONFIG),
        "dataset": {
            "name": local_cfg.DATA.DATASET,
            "train": len(dataset.train),
            "query": len(dataset.query),
            "gallery": len(dataset.gallery),
            "num_train_pids": dataset.num_train_pids,
            "num_camera": dataset.num_camera,
        },
        "official_checkpoint": {
            "path": str(OFFICIAL),
            "key_count": len(official_state),
            "critical_shapes": {
                key: list(official_state[key].shape)
                for key in official_state
                if key in {"head.weight", "camera_embed", "head_clip_bio.weight"}
                or key.startswith("head_clip_bioReverse")
            },
        },
        "reconstructed_model": {
            "key_count_before_eval_adapter": model_keys_before_adapter,
            "key_count_after_eval_adapter": model_keys_after_adapter,
            "evaluation_only_reverse_adapters": adapters,
            "critical_shapes": {
                key: list(model_state[key].shape)
                for key in model_state
                if key in {"head.weight", "camera_embed", "head_clip_bio.weight"}
                or key.startswith("head_clip_bioReverse")
            },
        },
        "strict_load": {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "shape_mismatch_count": len(model_shape_mismatch),
            "shape_mismatches": model_shape_mismatch,
            "pass": not missing and not unexpected and not model_shape_mismatch,
        },
        "official_vs_our_best": {
            "ours_path": str(OURS),
            "ours_key_count": len(ours_state),
            "key_set_identical": official_keys == ours_keys,
            "official_only_keys": sorted(official_keys - ours_keys),
            "ours_only_keys": sorted(ours_keys - official_keys),
            "shape_schema_identical_on_common_keys": not ours_schema_mismatch,
            "common_shape_mismatches": ours_schema_mismatch,
        },
        "note": "The two official reverse bias keys are exposed only by the process-local evaluation adapter; no repository or checkpoint mutation occurred.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
