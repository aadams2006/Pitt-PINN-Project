import torch

from pitt_pinn.physics import (
    bounded_by_total_penalty,
    constraint_report,
    monotonic_increasing_penalty,
    smoothness_penalty,
)


def test_bounded_by_total_penalty_zero_when_valid():
    pred = torch.tensor([[0.1], [0.2]])
    total = torch.tensor([[0.2], [0.3]])
    assert bounded_by_total_penalty(pred, total).item() == 0.0


def test_monotonic_increasing_penalty_positive_when_negative_derivative():
    deriv = torch.tensor([0.2, -0.1, 0.0])
    assert monotonic_increasing_penalty(deriv).item() > 0.0


def test_smoothness_penalty_nonnegative():
    grad = torch.tensor([1.0, -1.0, 0.5])
    assert smoothness_penalty(grad).item() >= 0.0


def test_constraint_report_counts_violations():
    pred = torch.tensor([[0.4], [0.1], [0.8]])
    total = torch.tensor([[0.5], [0.2], [0.7]])
    report = constraint_report(pred, total)
    assert report["bounded_violation_count"] == 1
    assert report["sample_count"] == 3
