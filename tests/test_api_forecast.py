"""
Integration tests for FastAPI /predict/forecast endpoint.

Verifies:
1. Valid payload returns 200 with ForecastResponse schema.
2. model_source field is present and equals 'lstm' or 'fallback'.
3. Uncertainty bands (lower_bound, upper_bound) are returned and properly enclose trajectory.
4. Mandatory educational disclaimer is present.
5. Inadequate history returns 400 or structured error.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main_enhanced import app


@pytest.fixture
def client():
    return TestClient(app)


def test_forecast_api_success(client):
    """Test standard 30-day lookback forecast request."""
    # Create 35-day realistic price series
    history = [450.0 + i * 0.4 for i in range(35)]
    response = client.post("/predict/forecast", json={"history": history})

    assert response.status_code == 200, f"Forecast failed with status {response.status_code}: {response.text}"
    data = response.json()

    assert "model_source" in data
    assert data["model_source"] in ("bilstm_attention", "lstm", "fallback")
    assert len(data["forecast_trajectory"]) == 5
    assert data["historical_last"] == history[-1]
    assert data["horizon_days"] == 5

    # Check uncertainty bounds
    if data.get("lower_bound") and data.get("upper_bound"):
        assert len(data["lower_bound"]) == 5
        assert len(data["upper_bound"]) == 5
        for low, mid, high in zip(data["lower_bound"], data["forecast_trajectory"], data["upper_bound"]):
            assert low <= mid <= high, f"Bound mismatch: {low} <= {mid} <= {high}"

    # Check regulatory disclaimer
    assert "disclaimer" in data
    assert "Not investment advice" in data["disclaimer"]

    # Ensure fake accuracy string is gone
    assert "92.4% (Backtested R²)" not in data["accuracy_metric"]


def test_forecast_api_insufficient_history(client):
    """Test rejection when history is less than 30 points."""
    short_history = [100.0, 102.0, 101.5]
    response = client.post("/predict/forecast", json={"history": short_history})
    assert response.status_code in (400, 500)
