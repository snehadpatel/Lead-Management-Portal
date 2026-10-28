"""PyTorch Deep Learning Time-Series Forecaster for Mutual Fund NAV.

Features:
1. Modular LSTMForecaster (nn.Module) with dropout for epistemic uncertainty.
2. Monte Carlo Dropout (MCD) multi-horizon confidence intervals (90% prediction interval).
3. Deterministic Moving Average / Naive Last-Value baseline as an explicit fallback.
4. Scale-safe inference using fitted training scaler parameters (zero leakage).
5. Explicit 'model_source' tagging ("lstm" | "fallback").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

from lume_platform.config import ARTIFACTS_DIR, MODELS_DIR


class LSTMForecaster(nn.Module):
    """Multi-horizon PyTorch LSTM for sequential price forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch, seq_len, input_dim)
        returns: (batch, horizon)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout_layer(last_hidden)
        preds = self.fc(last_hidden)
        return preds


# Backward-compatible alias for older imports
NAVPredictorLSTM = LSTMForecaster


class DeterministicNaiveBaseline:
    """Canonical persistence baseline: projects the last observed value forward."""

    @staticmethod
    def forecast(history: List[float], horizon: int = 5) -> List[float]:
        if not history:
            return [0.0] * horizon
        last_val = float(history[-1])
        return [round(last_val, 4)] * horizon


class DeterministicMovingAverageBaseline:
    """Moving-average baseline: projects the average of the last N periods forward."""

    @staticmethod
    def forecast(history: List[float], horizon: int = 5, window: int = 5) -> List[float]:
        if not history:
            return [0.0] * horizon
        w = min(len(history), window)
        avg = float(np.mean(history[-w:]))
        return [round(avg, 4)] * horizon


def compute_deterministic_baseline(
    history: List[float],
    horizon: int = 5,
    window: int = 5,
) -> Dict[str, Any]:
    """Deterministic, honest baseline without random noise.

    Projects a simple moving average of the most recent `window` days.
    """
    if len(history) < window:
        raise ValueError(f"History must contain at least {window} values for baseline.")

    last_val = float(history[-1])
    recent = history[-window:]
    ma_val = float(np.mean(recent))

    # Deterministic linear projection towards the moving average
    trajectory = []
    for step in range(1, horizon + 1):
        weight = step / horizon
        proj = (1 - weight) * last_val + weight * ma_val
        trajectory.append(round(proj, 2))

    trend = "UPWARD" if trajectory[-1] > last_val else "DOWNWARD" if trajectory[-1] < last_val else "STABLE"

    # Historical sample variance for uncertainty estimation
    volatility = float(np.std(history[-10:])) if len(history) >= 10 else 1.0
    lower = [round(v - 1.645 * volatility, 2) for v in trajectory]
    upper = [round(v + 1.645 * volatility, 2) for v in trajectory]

    return {
        "historical_last": last_val,
        "forecast_trajectory": trajectory,
        "lower_bound": lower,
        "upper_bound": upper,
        "baseline_trajectory": trajectory,
        "confidence_score": 50.0,  # Explicitly labeled baseline confidence
        "trend": trend,
        "horizon_days": horizon,
        "accuracy_metric": "Deterministic Baseline (5-Day Simple Moving Average)",
        "model_source": "fallback",
        "disclaimer": "Educational and analytical tool only. Not investment advice.",
    }


compute_fallback_forecast = compute_deterministic_baseline


class LUMEForecaster:
    """Production inference engine for NAV forecasting with uncertainty bands."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        self.device = torch.device("cpu")  # CPU-safe for inference
        self.model_path = model_path or (MODELS_DIR / "lstm_nav_pattern_predictor.pth")
        self.scaler_path = scaler_path or ((ARTIFACTS_DIR / "ml_scalers") / "mf_nav_global_scaler.pkl")
        self.metrics_path = metrics_path or (MODELS_DIR / "forecaster_metrics.json")

        self.model: Optional[LSTMForecaster] = None
        self.scaler: Any = None
        self.lookback = 30
        self.horizon = 5
        self.real_accuracy_metric = "Trained PyTorch LSTM Forecaster"
        self.is_loaded = False

    def load(self) -> bool:
        """Load PyTorch checkpoint, scaler, and training metadata."""
        if not self.model_path.is_file() or not self.scaler_path.is_file():
            self.is_loaded = False
            return False

        try:
            try:
                import pickle
                with open(str(self.scaler_path), "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = joblib.load(str(self.scaler_path))

            self.model = LSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2, horizon=self.horizon)
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            if self.metrics_path.is_file():
                try:
                    with open(self.metrics_path, "r") as f:
                        meta = json.load(f)
                        rmse = meta.get("test_rmse", meta.get("val_rmse", None))
                        mape = meta.get("test_mape_pct", meta.get("val_mape_pct", None))
                        if rmse is not None and mape is not None:
                            self.real_accuracy_metric = f"Test RMSE: {rmse:.2f} | MAPE: {mape:.2f}%"
                except Exception:
                    pass

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Failed to load LSTM Forecaster: {e}")
            self.is_loaded = False
            self.model = None
            return False

    def forecast(
        self,
        history: List[float],
        mc_samples: int = 20,
    ) -> Dict[str, Any]:
        """Perform multi-horizon forecast with Monte Carlo Dropout uncertainty estimation."""
        if len(history) < self.lookback:
            return {"error": f"Insufficient history (minimum {self.lookback} values required)"}

        if not self.is_loaded or self.model is None or self.scaler is None:
            if not self.load():
                # Fallback to deterministic moving average baseline
                return compute_deterministic_baseline(history, horizon=self.horizon)

        last_val = float(history[-1])

        # Baseline trajectory for honest comparison
        baseline_res = compute_deterministic_baseline(history, horizon=self.horizon)
        baseline_trajectory = baseline_res["forecast_trajectory"]

        # Scale using trained scaler parameters
        seq = np.array(history[-self.lookback:], dtype=np.float32).reshape(-1, 1)
        # Transform using target scaler
        min_val = getattr(self.scaler, "data_min_", [min(history)])[0]
        max_val = getattr(self.scaler, "data_max_", [max(history)])[0]
        denom = max_val - min_val if max_val != min_val else 1.0

        seq_scaled = (seq - min_val) / denom
        input_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Monte Carlo Dropout: keep dropout active during evaluation to capture epistemic uncertainty
        self.model.train()
        preds_list = []
        with torch.no_grad():
            for _ in range(max(1, mc_samples)):
                scaled_pred = self.model(input_tensor).cpu().numpy()[0]
                real_pred = scaled_pred * denom + min_val
                preds_list.append(real_pred)

        self.model.eval()
        preds_arr = np.array(preds_list)  # (mc_samples, horizon)

        # Mean trajectory and 90% confidence intervals (approx. 5th and 95th percentiles)
        mean_trajectory = np.mean(preds_arr, axis=0)
        lower_bounds = np.percentile(preds_arr, 5, axis=0)
        upper_bounds = np.percentile(preds_arr, 95, axis=0)

        # Ensure sensible bounds ordering
        lower_bounds = np.minimum(lower_bounds, mean_trajectory)
        upper_bounds = np.maximum(upper_bounds, mean_trajectory)

        trend = "UPWARD" if mean_trajectory[-1] > last_val else "DOWNWARD" if mean_trajectory[-1] < last_val else "STABLE"

        # Honest confidence calculation based on prediction spread
        spread = np.mean(upper_bounds - lower_bounds) / max(last_val, 1e-4)
        # Lower spread -> higher confidence (bounded between 50% and 95%)
        confidence = float(np.clip(95.0 - (spread * 200.0), 50.0, 95.0))

        return {
            "historical_last": last_val,
            "forecast_trajectory": [round(float(v), 2) for v in mean_trajectory],
            "lower_bound": [round(float(v), 2) for v in lower_bounds],
            "upper_bound": [round(float(v), 2) for v in upper_bounds],
            "baseline_trajectory": baseline_trajectory,
            "confidence_score": round(confidence, 1),
            "trend": trend,
            "horizon_days": self.horizon,
            "accuracy_metric": self.real_accuracy_metric,
            "model_source": "lstm",
            "disclaimer": "Educational and analytical tool only. Not investment advice.",
        }
