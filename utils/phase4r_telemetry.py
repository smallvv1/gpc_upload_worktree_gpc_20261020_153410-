"""Gradient telemetry helpers for the Phase-4R repair track.

The helpers deliberately operate on the three unscaled component losses before
the normal GradScaler backward.  They report the corresponding scaled norms by
applying the current scaler value, so pre-scaled, scaled, and unscaled values
remain distinguishable in the artifact.
"""

from __future__ import annotations

import math
from typing import Any

import torch


SHARED_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "early_eva": ("blocks.0.",),
    "middle_eva": ("blocks.11.",),
    "last_eva": ("blocks.22.",),
    "bio_projection": ("head_clip_bio.",),
}


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _group_parameters(
    model: torch.nn.Module,
    prefixes: dict[str, tuple[str, ...]] = SHARED_GROUP_PREFIXES,
) -> dict[str, list[torch.nn.Parameter]]:
    base = _base_model(model)
    result: dict[str, list[torch.nn.Parameter]] = {}
    for group_name, group_prefixes in prefixes.items():
        result[group_name] = [
            parameter
            for name, parameter in base.named_parameters()
            if parameter.requires_grad and name.startswith(group_prefixes)
        ]
    return result


def _component_gradients(
    component: torch.Tensor | None,
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    if not parameters:
        return []
    if component is None or not component.requires_grad:
        return [None for _ in parameters]
    return list(
        torch.autograd.grad(
            component,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    )


def _norm(values: list[torch.Tensor | None]) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        finite_value = value.detach().float()
        total += float((finite_value * finite_value).sum().item())
    return math.sqrt(max(total, 0.0))


def _dot(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        total += float(
            (left_value.detach().float() * right_value.detach().float()).sum().item()
        )
    return total


def _add(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    result: list[torch.Tensor | None] = []
    for left_value, right_value in zip(left, right):
        if left_value is None:
            result.append(right_value)
        elif right_value is None:
            result.append(left_value)
        else:
            result.append(left_value + right_value)
    return result


def _cosine(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
    left_norm: float | None = None,
    right_norm: float | None = None,
) -> float:
    left_norm = _norm(left) if left_norm is None else left_norm
    right_norm = _norm(right) if right_norm is None else right_norm
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def collect_component_gradient_telemetry(
    model: torch.nn.Module,
    components: dict[str, torch.Tensor],
    scaler_scale: float,
    epoch: int,
    iteration: int,
    rank: int,
) -> list[dict[str, Any]]:
    """Collect component gradients for fixed shared parameter groups.

    ``components`` must contain ``core``, ``reverse_hair`` and
    ``reverse_clothing`` for a Full DIFFER/A2 run.  Missing components are
    represented by zero gradients, which is useful for explicit controls.
    """

    groups = _group_parameters(model)
    source_components: dict[str, torch.Tensor | None] = {
        "core": components.get("core"),
        "hair": components.get("reverse_hair"),
        "clothing": components.get("reverse_clothing"),
    }
    gradients: dict[str, dict[str, list[torch.Tensor | None]]] = {}
    for source_name, component in source_components.items():
        gradients[source_name] = {}
        for group_name, parameters in groups.items():
            gradients[source_name][group_name] = _component_gradients(component, parameters)

    rows: list[dict[str, Any]] = []
    scale = float(scaler_scale)
    for group_name in groups:
        core = gradients["core"][group_name]
        hair = gradients["hair"][group_name]
        clothing = gradients["clothing"][group_name]
        reverse = _add(hair, clothing)
        core_norm = _norm(core)
        hair_norm = _norm(hair)
        clothing_norm = _norm(clothing)
        reverse_norm = _norm(reverse)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "iteration": int(iteration),
            "rank": int(rank),
            "group": group_name,
            "scaler_scale": scale,
            "core_norm_pre_scaled": core_norm,
            "hair_norm_pre_scaled": hair_norm,
            "clothing_norm_pre_scaled": clothing_norm,
            "reverse_norm_pre_scaled": reverse_norm,
            "core_norm_scaled": core_norm * scale,
            "hair_norm_scaled": hair_norm * scale,
            "clothing_norm_scaled": clothing_norm * scale,
            "reverse_norm_scaled": reverse_norm * scale,
            "core_norm_unscaled": core_norm,
            "hair_norm_unscaled": hair_norm,
            "clothing_norm_unscaled": clothing_norm,
            "reverse_norm_unscaled": reverse_norm,
            "cos_core_hair": _cosine(core, hair, core_norm, hair_norm),
            "cos_core_clothing": _cosine(core, clothing, core_norm, clothing_norm),
            "cos_hair_clothing": _cosine(hair, clothing, hair_norm, clothing_norm),
            "cos_core_reverse": _cosine(core, reverse, core_norm, reverse_norm),
            "hair_to_core_norm_ratio": _ratio(hair_norm, core_norm),
            "clothing_to_core_norm_ratio": _ratio(clothing_norm, core_norm),
            "reverse_to_core_norm_ratio": _ratio(reverse_norm, core_norm),
            "hair_nonzero": bool(hair_norm > 0.0),
            "clothing_nonzero": bool(clothing_norm > 0.0),
            "reverse_nonzero": bool(reverse_norm > 0.0),
            "hair_conflict": bool(core_norm > 0.0 and hair_norm > 0.0 and _cosine(core, hair, core_norm, hair_norm) < 0.0),
            "clothing_conflict": bool(core_norm > 0.0 and clothing_norm > 0.0 and _cosine(core, clothing, core_norm, clothing_norm) < 0.0),
            "reverse_conflict": bool(core_norm > 0.0 and reverse_norm > 0.0 and _cosine(core, reverse, core_norm, reverse_norm) < 0.0),
            "parameter_count": len(groups[group_name]),
        }
        rows.append(row)
    return rows


def summarize_component_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return simple counts/medians for the report without changing raw rows."""

    import statistics

    summary: dict[str, Any] = {}
    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        group_summary: dict[str, Any] = {}
        for metric in (
            "cos_core_hair",
            "cos_core_clothing",
            "cos_core_reverse",
            "hair_to_core_norm_ratio",
            "clothing_to_core_norm_ratio",
            "reverse_to_core_norm_ratio",
        ):
            values = [float(row[metric]) for row in group_rows]
            group_summary[metric] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else 0.0,
                "median": statistics.median(values) if values else 0.0,
                "p10": min(values) if values else 0.0,
                "p90": max(values) if values else 0.0,
            }
        for key in ("hair_conflict", "clothing_conflict", "reverse_conflict"):
            count = sum(bool(row[key]) for row in group_rows)
            group_summary[key + "_count"] = count
            group_summary[key + "_rate"] = count / len(group_rows) if group_rows else 0.0
        summary[group] = group_summary
    return summary
