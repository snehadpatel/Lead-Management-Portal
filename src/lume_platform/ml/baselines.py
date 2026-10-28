"""
Time-Series Forecasting Baselines:
1. Naive Last-Value Persistence with historical quantile bands.
2. ARIMA(p, 1, 0) / Autoregressive Integrated Model with analytical quantile intervals.
3. Old Univariate LSTM model wrapper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class NaiveQuantileBaseline:
    """
    Deterministic Naive Last-Value Baseline.
    P50 = last observed value across all horizon steps.
    P10 and P90 are calibrated using empirical residuals from training data.
    """

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.q10_rel: np.ndarray = np.zeros(horizon)
        self.q90_rel: np.ndarray = np.zeros(horizon)
        self.is_fit = False

    def fit(self, history: Sequence[float]) -> NaiveQuantileBaseline:
        """Estimate empirical h-step return distribution on training history."""
        hist = np.asarray(history, dtype=np.float64)
        if len(hist) < self.horizon + 10:
            # Fallback default historical spread for mutual funds
            self.q10_rel = -0.005 * np.sqrt(np.arange(1, self.horizon + 1))
            self.q90_rel = 0.005 * np.sqrt(np.arange(1, self.horizon + 1))
            self.is_fit = True
            return self

        # Collect h-step relative changes: (hist[i+h] - hist[i]) / hist[i]
        errors_by_h = [[] for _ in range(self.horizon)]
        for i in range(len(hist) - self.horizon):
            curr = hist[i]
            if curr <= 0:
                continue
            for h in range(1, self.horizon + 1):
                future = hist[i + h]
                rel_change = (future - curr) / curr
                errors_by_h[h - 1].append(rel_change)

        self.q10_rel = np.array([
            float(np.percentile(errors_by_h[h], 10)) if len(errors_by_h[h]) > 0 else -0.01
            for h in range(self.horizon)
        ])
        self.q90_rel = np.array([
            float(np.percentile(errors_by_h[h], 90)) if len(errors_by_h[h]) > 0 else 0.01
            for h in range(self.horizon)
        ])
        self.is_fit = True
        return self

    def predict(self, history: Sequence[float]) -> Dict[str, np.ndarray]:
        """
        Returns:
            dict with:
                'p50': shape (horizon,)
                'p10': shape (horizon,)
                'p90': shape (horizon,)
        """
        last_val = float(history[-1])
        if not self.is_fit:
            self.fit(history)

        p50 = np.full(self.horizon, last_val, dtype=np.float64)
        p10 = last_val * (1.0 + self.q10_rel)
        p90 = last_val * (1.0 + self.q90_rel)

        # Enforce monotonicity: P10 <= P50 <= P90
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        return {"p10": p10, "p50": p50, "p90": p90}


class ARIMABaseline:
    """
    ARIMA(p, 1, 0) / Autoregressive Integrated Model.
    Differences the series (d=1) to achieve stationarity, fits an AR(p) model via OLS,
    and reconstructs future integrated values with Gaussian analytical quantiles (z=1.28155 for 80% CI).
    """

    def __init__(self, p: int = 3, horizon: int = 5):
        self.p = p
        self.horizon = horizon
        self.phi: np.ndarray = np.zeros(p)
        self.intercept: float = 0.0
        self.sigma: float = 0.0
        self.is_fit = False

    def fit(self, series: Sequence[float]) -> ARIMABaseline:
        """Fit AR(p) on first differences w_t = y_t - y_{t-1}."""
        y = np.asarray(series, dtype=np.float64)
        if len(y) < self.p + 10:
            self.is_fit = False
            return self

        # 1. Differencing (d=1)
        w = np.diff(y)
        N = len(w)

        # 2. Build lag matrix for AR(p)
        X = np.zeros((N - self.p, self.p))
        Y = w[self.p :]
        for i in range(self.p):
            X[:, i] = w[self.p - 1 - i : N - 1 - i]

        # Add intercept column
        X_const = np.column_stack([np.ones(len(X)), X])

        # 3. OLS estimation via lstsq (numerically stable)
        try:
            beta, _, _, _ = np.linalg.lstsq(X_const, Y, rcond=None)
            self.intercept = float(beta[0])
            self.phi = beta[1:]
            res = Y - X_const @ beta
            self.sigma = float(np.std(res))
            if np.isnan(self.sigma) or self.sigma < 1e-4:
                self.sigma = 0.5
            self.is_fit = True
        except Exception:
            self.is_fit = False

        return self

    def predict(self, series: Sequence[float]) -> Dict[str, np.ndarray]:
        """Generate h-step ahead forecasts and analytical quantiles."""
        y = np.asarray(series, dtype=np.float64)
        last_val = float(y[-1])

        if not self.is_fit or len(y) < self.p + 2:
            # Fallback to naive persistence
            p50 = np.full(self.horizon, last_val, dtype=np.float64)
            p10 = p50 - 0.01 * last_val
            p90 = p50 + 0.01 * last_val
            return {"p10": p10, "p50": p50, "p90": p90}

        # Recent differenced values: most recent at index 0
        w = np.diff(y)
        recent_w = list(w[-self.p :][::-1])

        # Forecast differences forward
        pred_diffs = []
        var_cum = []
        running_var = 0.0

        for h in range(self.horizon):
            # AR step: intercept + sum(phi_i * lag_i)
            w_hat = self.intercept + sum(
                float(self.phi[i]) * recent_w[i] for i in range(len(self.phi))
            )
            pred_diffs.append(w_hat)
            running_var += (self.sigma ** 2) * (1.0 + 0.1 * h)
            var_cum.append(running_var)

            # Update rolling lags
            recent_w = [w_hat] + recent_w[:-1]

        # Cumulative sum + last known price
        p50 = last_val + np.cumsum(pred_diffs)

        # Analytical 80% confidence bounds (z_0.90 ~ 1.28155)
        std_cum = np.sqrt(np.array(var_cum))
        z = 1.28155
        p10 = p50 - z * std_cum
        p90 = p50 + z * std_cum

        # Monotonicity check
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        return {"p10": p10, "p50": p50, "p90": p90}
