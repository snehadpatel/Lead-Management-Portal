"""
Unit tests for PyTorch LSTM forecaster, deterministic baselines, and uncertainty bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lume_platform.ml.forecaster import (
    DeterministicMovingAverageBaseline,
    DeterministicNaiveBaseline,
    LSTMForecaster,
    LUMEForecaster,
    compute_fallback_forecast,
)
from lume_platform.utils.seed import set_seed


def test_lstm_forward_shape():
    """Verify LSTM input and output tensor dimensions."""
    batch_size = 8
    seq_len = 30
    input_dim = 1
    horizon = 5

    model = LSTMForecaster(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.1, horizon=horizon)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, horizon), f"Expected shape {(batch_size, horizon)}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN found in LSTM output"


def test_reproducibility_with_seed():
    """Verify that setting the random seed produces identical model parameters and forward passes."""
    set_seed(42)
    m1 = LSTMForecaster(input_dim=1, hidden_dim=32, num_layers=1, horizon=5)
    x = torch.randn(2, 30, 1)
    out1 = m1(x).detach().numpy()

    set_seed(42)
    m2 = LSTMForecaster(input_dim=1, hidden_dim=32, num_layers=1, horizon=5)
    # The input generated with same seed:
    set_seed(42)
    _ = LSTMForecaster(input_dim=1, hidden_dim=32, num_layers=1, horizon=5)
    x2 = torch.randn(2, 30, 1)
    out2 = m2(x2).detach().numpy()

    assert np.allclose(out1, out2, atol=1e-5), "Seed does not produce deterministic results"


def test_deterministic_naive_baseline():
    """Verify canonical persistence baseline projects the exact last known value."""
    history = [100.0, 102.5, 105.0]
    preds = DeterministicNaiveBaseline.forecast(history, horizon=5)
    assert preds == [105.0, 105.0, 105.0, 105.0, 105.0]


def test_deterministic_moving_average_baseline():
    """Verify moving average baseline projects the average of the last W periods."""
    history = [10.0, 20.0, 30.0, 40.0, 50.0]
    preds = DeterministicMovingAverageBaseline.forecast(history, horizon=3, window=5)
    expected_avg = np.mean(history)
    assert np.allclose(preds, [expected_avg, expected_avg, expected_avg])


def test_fallback_forecast_honesty():
    """Verify fallback forecast is deterministic, contains no fake numbers, and provides honest bounds."""
    # Deterministic sequence
    history = [100.0 + i * 0.5 for i in range(40)]
    f1 = compute_fallback_forecast(history, horizon=5)
    f2 = compute_fallback_forecast(history, horizon=5)

    assert f1["model_source"] == "fallback"
    assert f1["forecast_trajectory"] == f2["forecast_trajectory"], "Fallback must be deterministic (zero noise)"
    assert len(f1["forecast_trajectory"]) == 5
    assert len(f1["lower_bound"]) == 5
    assert len(f1["upper_bound"]) == 5

    # Bounds sanity: lower_bound <= forecast <= upper_bound
    for low, mid, high in zip(f1["lower_bound"], f1["forecast_trajectory"], f1["upper_bound"]):
        assert low <= mid <= high, f"Uncertainty violation: {low} <= {mid} <= {high}"

    # Accuracy metric must be honest
    assert "Baseline" in f1["accuracy_metric"]
    assert "92.4%" not in f1["accuracy_metric"]
    assert "disclaimer" in f1


def test_forecaster_insufficient_history():
    """Verify that providing fewer than 30 points returns a structured error, not a crash."""
    forecaster = LUMEForecaster()
    res = forecaster.forecast([10.0, 20.0])
    assert "error" in res
    assert "Insufficient history" in res["error"]
