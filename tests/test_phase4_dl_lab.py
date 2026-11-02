"""
Unit and Integration Tests for Phase 4 Deep Learning Lab & Visual Explainability.
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from api.main_enhanced import app
from lume_platform.ml.dl_lab_service import DLLabService
from lume_platform.ml.mlp_trainer import LeadMLP


@pytest.fixture
def client():
    return TestClient(app)


def test_lead_mlp_architecture():
    """Verify LeadMLP layer shapes, batch normalization, and parameter count."""
    model = LeadMLP(in_features=91)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 8193, f"Expected 8,193 parameters, got {total_params}"

    x = torch.randn(4, 91)
    out = model(x)
    assert out.shape == (4,), f"Expected output shape (4,), got {out.shape}"


def test_dl_lab_service_metrics_and_history():
    """Verify real metrics and genuine training history retrieval."""
    svc = DLLabService.get_instance()
    comp = svc.get_comparison_metrics()

    assert "random_forest" in comp
    assert "lead_mlp" in comp
    assert comp["lead_mlp"]["accuracy"] == "85.26%"
    assert comp["random_forest"]["accuracy"] == "85.00%"
    assert comp["lead_mlp"]["parameters"] == "8,193 Trainable Parameters"
    assert comp["lead_mlp"]["confusion_matrix"] == [[198, 38], [18, 126]]

    hist = svc.get_training_history()
    assert "epochs" in hist
    assert len(hist["epochs"]) == 40
    assert len(hist["train_loss"]) == 40
    assert len(hist["val_loss"]) == 40
    assert hist["train_loss"][0] > hist["train_loss"][-1], "Training loss must decrease"


def test_dl_lab_service_dual_prediction():
    """Verify live dual prediction and permutation feature importance attribution."""
    svc = DLLabService.get_instance()
    sample = {
        "totalvisits": 10,
        "total_time_spent_on_website": 1500,
        "page_views_per_visit": 5.0,
        "asymmetrique_activity_score": 17,
        "asymmetrique_profile_score": 18,
        "lead_quality": "High in Relevance",
        "lead_source": "Google",
        "what_is_your_current_occupation": "Working Professional",
    }
    result = svc.predict_dual(sample)

    assert "random_forest" in result
    assert "lead_mlp" in result
    assert result["random_forest"]["status"] == "online"
    assert result["lead_mlp"]["status"] == "online"
    assert 0.0 <= result["random_forest"]["probability"] <= 1.0
    assert 0.0 <= result["lead_mlp"]["probability"] <= 1.0
    assert "model_agreement" in result
    assert "key_driving_factors" in result
    assert len(result["key_driving_factors"]) > 0


def test_api_dl_lab_endpoints(client):
    """Test REST API contracts for all Phase 4 endpoints."""
    # 1. Comparison Endpoint
    res_comp = client.get("/api/dl-lab/comparison")
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert "lead_mlp" in data_comp
    assert data_comp["lead_mlp"]["recall"] == 0.875

    # 2. Training History Endpoint
    res_hist = client.get("/api/dl-lab/training-history")
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert len(data_hist["epochs"]) == 40

    # 3. Metadata Endpoint
    res_meta = client.get("/api/dl-lab/metadata")
    assert res_meta.status_code == 200
    data_meta = res_meta.json()
    assert "honest_limitations" in data_meta

    # 4. Dual Prediction Endpoint
    res_pred = client.post(
        "/api/dl-lab/predict-lead",
        json={
            "totalvisits": 4,
            "total_time_spent_on_website": 400,
            "page_views_per_visit": 2.0,
            "lead_quality": "Might be",
            "lead_source": "Direct Traffic",
        },
    )
    assert res_pred.status_code == 200
    data_pred = res_pred.json()
    assert data_pred["random_forest"]["status"] == "online"
    assert data_pred["lead_mlp"]["status"] == "online"

