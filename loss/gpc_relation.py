"""Diagnostic-friendly implementation of the frozen GPC relation loss.

The default reduction is deliberately the historical reduction:

    mean(1 - cosine(F.normalize(feat[0].float())[i],
                     F.normalize(feat[0].float())[j]))

``masked_index`` is only for counterfactual diagnostics. It removes one
selected relation's numerator contribution while retaining the original
selected-relation count as the denominator. Training callers leave it as
``None`` and therefore use the unchanged historical path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F


def gpc_pair_loss_values(
    feat0: torch.Tensor,
    selected_pairs: Iterable[tuple[int, int, Any, Any]],
) -> torch.Tensor:
    """Return one FP32 ``1 - cosine`` value for each selected pair."""

    representation = F.normalize(feat0.float(), dim=-1)
    values = [
        1.0 - torch.sum(representation[int(left)] * representation[int(right)])
        for left, right, *_metadata in selected_pairs
    ]
    if not values:
        return representation.new_empty((0,))
    return torch.stack(values)


def reduce_gpc_pair_loss(
    pair_losses: torch.Tensor,
    anchor: torch.Tensor,
    masked_index: int | None = None,
) -> torch.Tensor:
    """Reduce GPC terms using the frozen selected-relation denominator."""

    budget = int(pair_losses.numel())
    if budget == 0:
        return anchor.sum() * 0.0
    if masked_index is None:
        return pair_losses.mean()
    if not 0 <= int(masked_index) < budget:
        raise IndexError(f"masked_index={masked_index} outside relation budget {budget}")
    mask = torch.zeros(budget, dtype=torch.bool, device=pair_losses.device)
    mask[int(masked_index)] = True
    return pair_losses.masked_fill(mask, 0.0).sum() / budget


def gpc_relation_loss(
    feat0: torch.Tensor,
    selected_pairs: Iterable[tuple[int, int, Any, Any]],
    masked_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(reduced_loss, per_relation_losses)`` for GPC diagnostics."""

    pair_losses = gpc_pair_loss_values(feat0, selected_pairs)
    return reduce_gpc_pair_loss(pair_losses, feat0, masked_index), pair_losses
