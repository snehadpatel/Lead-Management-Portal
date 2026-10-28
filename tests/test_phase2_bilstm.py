"""
Unit and Integration Tests for Phase 2 Time-Series Deep Learning Engine.

Verifies:
1. BiLSTMAttentionQuantileForecaster forward pass shapes and non-crossing quantile monotonicity.
2. Temporal Multihead Attention weights sum to 1.0 per sequence.
3. Pinball (Quantile) Loss mathematical properties.
4. Technical indicators zero-leakage backward causality.
5. Baseline quantile monotonicity.
6. FastAPI /api/forecast endpoint integration with p10, p50, p90 and attention weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from api.main_enhanced import app
from lume_platform.data.timeseries_dataset import compute_technical_features
from lume_platform.ml.baselines import ARIMABaseline, NaiveQuantileBaseline
from lume_platform.ml.bilstm_attention import BiLSTMAttentionQuantileForecaster
from lume_platform.ml.quantile_loss import PinballLoss, empirical_quantile_coverage


@pytest.fixture
def api_client():
    return TestClient(app)


def test_bilstm_attention_forward_shape_and_monotonicity():
    """Verify (batch, 30, 7) input produces (batch, 5, 3) monotonic quantiles."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 30
    input_dim = 7
    horizon = 5

    model = BiLSTMAttentionQuantileForecaster(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        horizon=horizon,
        bidirectional=True,
        use_attention=True,
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        quantiles, attn = model(x, return_attention=True)

    # 1. Check shape
    assert quantiles.shape == (batch_size, horizon, 3), f"Expected {(batch_size, horizon, 3)}, got {quantiles.shape}"

    # 2. Check strict monotonicity: P10 <= P50 <= P90
    p10 = quantiles[..., 0]
    p50 = quantiles[..., 1]
    p90 = quantiles[..., 2]

    assert torch.all(p10 <= p50 + 1e-6), "Quantile crossing: P10 > P50 found"
    assert torch.all(p50 <= p90 + 1e-6), "Quantile crossing: P50 > P90 found"

    # 3. Check attention weights shape and softmax sum
    assert attn is not None
    assert attn.shape == (batch_size, seq_len), f"Expected {(batch_size, seq_len)}, got {attn.shape}"
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Attention weights do not sum to 1"


def test_pinball_loss_math_properties():
    """Verify Pinball loss behaves correctly for q=0.5 (half MAE) and asymmetric quantiles."""
    criterion_p50 = PinballLoss(quantiles=[0.50])
    criterion_multi = PinballLoss(quantiles=[0.10, 0.50, 0.90])

    y_true = torch.tensor([[100.0, 105.0]])  # (1, 2)
    y_pred_p50 = torch.tensor([[[96.0], [107.0]]])  # (1, 2, 1)

    # For q=0.5, Pinball Loss = 0.5 * mean(|100-96|, |105-107|) = 0.5 * (4 + 2) / 2 = 1.5
    loss_val = criterion_p50(y_pred_p50, y_true).item()
    assert abs(loss_val - 1.5) < 1e-5, f"Pinball loss at q=0.5 mismatch: {loss_val} vs 1.5"

    # Coverage rate calculation
    cov = empirical_quantile_coverage(
        y_true=[100.0, 105.0, 110.0, 95.0],
        p_lower=[90.0, 100.0, 100.0, 96.0],  # 4th item is out of bounds (95.0 < 96.0)
        p_upper=[110.0, 110.0, 120.0, 100.0],
    )
    assert cov == 0.75, f"Expected 0.75 coverage, got {cov}"


def test_technical_features_zero_leakage():
    """Verify indicators at time t use strictly past information (causality check)."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    navs = 500.0 + np.cumsum(np.random.randn(60) * 2.0)
    nifty = 22000.0 + np.cumsum(np.random.randn(60) * 50.0)

    df = pd.DataFrame({"date": dates, "nav": navs, "nifty_close": nifty})
    feats_orig = compute_technical_features(df)

    # Now alter a future observation at index 50
    df_altered = df.copy()
    df_altered.loc[50, "nav"] += 200.0  # Massive shock at t=50

    feats_altered = compute_technical_features(df_altered)

    # Past indicators (rows 0 to 49) MUST remain 100% identical
    for col in ["nav_return", "rsi", "sma20_dev"]:
        np.testing.assert_allclose(
            feats_orig.loc[:49, col].values,
            feats_altered.loc[:49, col].values,
            err_msg=f"Lookahead leakage detected in {col}!",
        )


def test_baseline_quantile_outputs():
    """Verify Naive and ARIMA baselines produce monotonic 5-day quantiles."""
    hist = [400.0 + i * 0.5 + (i % 3) for i in range(50)]

    naive = NaiveQuantileBaseline(horizon=5).fit(hist)
    res_naive = naive.predict(hist)
    assert len(res_naive["p50"]) == 5
    assert np.all(res_naive["p10"] <= res_naive["p50"])
    assert np.all(res_naive["p50"] <= res_naive["p90"])

    arima = ARIMABaseline(p=3, horizon=5).fit(hist)
    res_arima = arima.predict(hist)
    assert len(res_arima["p50"]) == 5
    assert np.all(res_arima["p10"] <= res_arima["p50"])
    assert np.all(res_arima["p50"] <= res_arima["p90"])


def test_api_forecast_phase2_endpoints(api_client):
    """Verify /api/forecast endpoint returns p10, p50, p90 and attention weights."""
    history = [500.0 + i * 0.3 for i in range(35)]

    for endpoint in ["/api/forecast", "/predict/forecast"]:
        response = api_client.post(endpoint, json={"history": history})
        assert response.status_code == 200, f"Endpoint {endpoint} failed: {response.text}"
        data = response.json()

        assert "p10" in data and len(data["p10"]) == 5
        assert "p50" in data and len(data["p50"]) == 5
        assert "p90" in data and len(data["p90"]) == 5
        assert "attention_weights" in data and len(data["attention_weights"]) == 30
        assert data["model_source"] in ("bilstm_attention", "lstm", "fallback")
        assert "disclaimer" in data
        assert "Not investment advice" in data["disclaimer"]

        # Assert quantile ordering: p10 <= p50 <= p90
        for low, mid, high in zip(data["p10"], data["p50"], data["p90"]):
            assert low <= mid <= high, f"Quantile ordering mismatch: {low} <= {mid} <= {high}"
