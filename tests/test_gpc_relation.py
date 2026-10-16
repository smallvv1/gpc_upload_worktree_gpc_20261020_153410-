from __future__ import annotations

import torch
import torch.nn.functional as F

from loss.gpc_relation import gpc_relation_loss


def _pairs():
    return [
        (0, 1, (10, 11), (2, 3)),
        (2, 3, (12, 13), (4, 5)),
        (0, 2, (10, 12), (2, 4)),
    ]


def _historical(feat0: torch.Tensor, pairs):
    z = F.normalize(feat0.float(), dim=-1)
    values = [1.0 - torch.sum(z[i] * z[j]) for i, j, *_ in pairs]
    return torch.stack(values).mean(), torch.stack(values)


def test_all_relations_match_historical_value_and_gradient():
    base = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, -1.0, 2.0], [-2.0, 4.0, 1.0], [0.5, 1.5, -3.0]],
        dtype=torch.float64,
    )
    old_input = base.clone().requires_grad_(True)
    new_input = base.clone().requires_grad_(True)

    old_loss, old_values = _historical(old_input, _pairs())
    new_loss, new_values = gpc_relation_loss(new_input, _pairs())
    old_grad = torch.autograd.grad(old_loss, old_input)[0]
    new_grad = torch.autograd.grad(new_loss, new_input)[0]

    assert torch.equal(old_values, new_values)
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_grad, new_grad)


def test_mask_removes_only_one_numerator_term_and_keeps_budget_denominator():
    feat0 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    full, values = gpc_relation_loss(feat0, _pairs(), masked_index=None)
    masked, _ = gpc_relation_loss(feat0, _pairs(), masked_index=1)
    expected = (values.sum() - values[1]) / values.numel()

    assert torch.equal(masked, expected)
    assert torch.equal(full - masked, values[1] / values.numel())


def test_empty_relation_set_is_differentiable_zero():
    feat0 = torch.randn(4, 3, requires_grad=True)
    loss, values = gpc_relation_loss(feat0, [])
    assert values.shape == (0,)
    assert torch.equal(loss, feat0.sum() * 0.0)
    loss.backward()
    assert torch.equal(feat0.grad, torch.zeros_like(feat0))
