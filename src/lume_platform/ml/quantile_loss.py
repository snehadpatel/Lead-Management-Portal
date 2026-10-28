"""
Pinball (Quantile) Loss Implementation for PyTorch.

Mathematical formulation:
For a target y, prediction y_hat, and quantile q in (0, 1):
    error = y - y_hat
    L_q(y, y_hat) = max(q * error, (q - 1) * error)
                  = 0.5 * |error| + (q - 0.5) * error

For multi-quantile multi-step forecasting:
    Total Loss = mean_{batch, horizon, quantiles} L_q(y_h, y_hat_{h, q})
"""

from __future__ import annotations

from typing import List, Sequence, Union
import numpy as np
import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Pinball (Quantile) Loss for multi-quantile regression."""

    def __init__(self, quantiles: Sequence[float] = (0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = list(quantiles)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted quantiles of shape (batch, horizon, num_quantiles)
            y_true: True future values of shape (batch, horizon) or (batch, horizon, 1)

        Returns:
            Scalar pinball loss averaged across batch, horizon, and quantiles.
        """
        if y_true.dim() == 2:
            # Expand to (batch, horizon, 1)
            y_true = y_true.unsqueeze(-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[..., i : i + 1]
            error = y_true - pred_q
            # L_q = max(q * error, (q - 1) * error)
            loss_q = torch.maximum(q * error, (q - 1.0) * error)
            losses.append(loss_q)

        total_loss = torch.cat(losses, dim=-1).mean()
        return total_loss


def compute_pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
) -> torch.Tensor:
    """Functional wrapper for PinballLoss."""
    criterion = PinballLoss(quantiles=quantiles)
    return criterion(y_pred, y_true)


def empirical_quantile_coverage(
    y_true: Union[torch.Tensor, np.ndarray, Sequence[float]],
    p_lower: Union[torch.Tensor, np.ndarray, Sequence[float]],
    p_upper: Union[torch.Tensor, np.ndarray, Sequence[float]],
) -> float:
    """
    Computes empirical coverage rate: percentage of actual observations
    that fall within [p_lower, p_upper].

    Nominal coverage for [P10, P90] is 80.0%.
    """
    if isinstance(y_true, torch.Tensor):
        y_arr = y_true.detach().cpu().numpy().flatten()
    else:
        y_arr = np.asarray(y_true, dtype=np.float64).flatten()

    if isinstance(p_lower, torch.Tensor):
        low_arr = p_lower.detach().cpu().numpy().flatten()
    else:
        low_arr = np.asarray(p_lower, dtype=np.float64).flatten()

    if isinstance(p_upper, torch.Tensor):
        high_arr = p_upper.detach().cpu().numpy().flatten()
    else:
        high_arr = np.asarray(p_upper, dtype=np.float64).flatten()

    assert len(y_arr) == len(low_arr) == len(high_arr), "Array length mismatch"
    if len(y_arr) == 0:
        return 0.0

    in_bounds = np.sum((y_arr >= low_arr) & (y_arr <= high_arr))
    return float(in_bounds / len(y_arr))
