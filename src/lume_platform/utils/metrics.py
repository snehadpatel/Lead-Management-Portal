"""Time-series and regression evaluation metrics."""

from __future__ import annotations

from typing import Dict, Any, Union
import numpy as np


def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    last_known: Union[np.ndarray, list, None] = None,
) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, MAPE, and Directional Accuracy.

    All metrics are computed on unscaled real values.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yt - yp)))

    # MAPE with epsilon to avoid division by zero
    non_zero = np.abs(yt) > 1e-6
    if np.any(non_zero):
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0)
    else:
        mape = 0.0

    metrics: Dict[str, float] = {
        "mse": round(mse, 6),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 2),
    }

    # Directional accuracy if last known price is provided
    if last_known is not None:
        lk = np.asarray(last_known, dtype=np.float64)
        if yt.ndim == 2:
            # Multi-step: compare step 1 to last_known, or trajectory end to last_known
            actual_dir = np.sign(yt[:, -1] - lk.flatten())
            pred_dir = np.sign(yp[:, -1] - lk.flatten())
            dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)
        else:
            actual_dir = np.sign(yt[-1] - lk)
            pred_dir = np.sign(yp[-1] - lk)
            dir_acc = 100.0 if actual_dir == pred_dir else 0.0
        metrics["directional_accuracy_pct"] = round(dir_acc, 2)

    return metrics
