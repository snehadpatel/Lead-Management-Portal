"""
Unit and Integration Tests for Phase 3 Computer Vision & Grad-CAM.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from lume_platform.vision.gradcam import (
    GradCAM,
    base64_to_image,
    image_to_base64,
    overlay_gradcam_on_image,
)
from lume_platform.vision.models import CandlestickCNN, CandlestickResNet18
from lume_platform.vision.service import VisionPatternService


def test_custom_cnn_architecture():
    model = CandlestickCNN(num_classes=3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 422659, f"Expected 422,659 parameters, got {total_params}"

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), f"Expected output shape (2, 3), got {out.shape}"


def test_resnet18_architecture():
    model = CandlestickResNet18(num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), f"Expected output shape (2, 3), got {out.shape}"


def test_gradcam_generation_and_overlay():
    model = CandlestickCNN(num_classes=3)
    target_layer = model.get_target_layer()
    gradcam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 224, 224)
    cam, pred_class, conf = gradcam.generate(x)

    assert cam.shape == (224, 224), f"Expected CAM shape (224, 224), got {cam.shape}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0, "CAM values must be normalized in [0, 1]"
    assert pred_class in [0, 1, 2], f"Invalid predicted class {pred_class}"
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range"

    # Test overlay
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    overlay = overlay_gradcam_on_image(dummy_img, cam, alpha=0.5)
    assert overlay.size == (224, 224)
    assert overlay.mode == "RGB"

    # Test base64 conversion
    b64 = image_to_base64(overlay)
    assert b64.startswith("data:image/png;base64,")
    decoded = base64_to_image(b64)
    assert decoded.size == (224, 224)

    gradcam.remove_hooks()


def test_vision_service_scheme_prediction():
    service = VisionPatternService(model_type="custom_cnn")
    result = service.predict_from_scheme("100119", lookback_days=40)

    assert "pattern" in result
    assert result["pattern"] in ["Bullish Breakout", "Bearish Breakdown", "Consolidation"]
    assert "pattern_class" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert "gradcam_overlay_base64" in result
    assert result["gradcam_overlay_base64"].startswith("data:image/png;base64,")


def test_api_vision_pattern_endpoint():
    from fastapi.testclient import TestClient
    from api.main_enhanced import app

    client = TestClient(app)
    response = client.post("/api/vision/pattern", json={"scheme_code": "100119"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "pattern" in data
    assert data["pattern"] in ["Bullish Breakout", "Bearish Breakdown", "Consolidation"]
    assert "confidence" in data
    assert "gradcam_overlay_base64" in data
    assert data["gradcam_overlay_base64"].startswith("data:image/png;base64,")

