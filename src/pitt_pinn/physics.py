from __future__ import annotations

import torch


def bounded_by_total_penalty(pred_bonded: torch.Tensor, total_thickness: torch.Tensor) -> torch.Tensor:
    """Penalty when bonded thickness exceeds total thickness."""
    return torch.mean(torch.relu(pred_bonded - total_thickness) ** 2)


def monotonic_increasing_penalty(derivative: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Penalty when derivative should be positive but drops below margin."""
    return torch.mean(torch.relu(margin - derivative) ** 2)


def smoothness_penalty(gradient: torch.Tensor) -> torch.Tensor:
    """Simple smoothness regularizer on local derivatives."""
    return torch.mean(gradient**2)


def constraint_report(pred_bonded: torch.Tensor, total_thickness: torch.Tensor) -> dict:
    violations = (pred_bonded > total_thickness).float()
    return {
        "bounded_violation_rate": float(torch.mean(violations).item()),
        "bounded_violation_count": int(torch.sum(violations).item()),
        "sample_count": int(pred_bonded.numel()),
    }
