#!/usr/local/bin/python3
"""Run the repository test.py path with an evaluation-only bias adapter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import eva_meta_bio
import data as data_module


class EvaluationOnlyReverseAdapter(nn.Module):
    def __init__(self, in_num, class_num, alpha=1.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.grl = eva_meta_bio.GradientReversalLayer(alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=True)
        if weight is not None:
            self.head_fc.weight.data.copy_(weight)

    def forward(self, x):
        return self.head_fc(self.grl(x))


# build_model resolves this symbol when it constructs the bio model.  This is
# process-local and does not modify model/eva_meta_bio.py or any checkpoint.
eva_meta_bio.GradientReversalClassifier = EvaluationOnlyReverseAdapter

# The current repository dataloader returns an additional trainOriginalLoader
# for LTCC (seven values), while the released test.py unpacks six.  Adapt only
# this process so the test.py inference/evaluator path can be exercised; no
# repository source or training behavior is changed.
_released_build_dataloader = data_module.build_dataloader


def _testpy_build_dataloader(config):
    result = _released_build_dataloader(config)
    if config.DATA.DATASET == "ltcc" and len(result) == 7:
        return result[:6]
    return result


data_module.build_dataloader = _testpy_build_dataloader

if __name__ == "__main__":
    runpy.run_path(str(REPO_ROOT / "test.py"), run_name="__main__")
