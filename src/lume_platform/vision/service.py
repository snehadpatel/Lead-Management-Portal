"""
Vision Inference Service for Candlestick Pattern Recognition & Grad-CAM.

Provides single-point inference for the FastAPI /api/vision/pattern endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from lume_platform.config import EXPORT_DIR, MODELS_DIR, PROJECT_ROOT
from lume_platform.vision.candlestick_dataset import (
    LABEL_NAMES,
    render_candlestick_image,
    synthesize_ohlc_from_nav,
)
from lume_platform.vision.gradcam import (
    GradCAM,
    base64_to_image,
    image_to_base64,
    overlay_gradcam_on_image,
)
from lume_platform.vision.models import CandlestickCNN, CandlestickResNet18


class VisionPatternService:
    """
    Production inference service exposing Candlestick Pattern Recognition and Grad-CAM.
    """

    def __init__(
        self,
        model_type: str = "resnet18",
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        default_cnn = MODELS_DIR / "candlestick_cnn_custom.pth"
        default_resnet = MODELS_DIR / "candlestick_resnet18_finetuned.pth"

        if model_type == "custom_cnn" and not default_cnn.is_file() and default_resnet.is_file():
            model_type = "resnet18"
        self.model_type = model_type

        if model_type == "resnet18":
            self.model_path = model_path or default_resnet
            self.model = CandlestickResNet18(num_classes=3)
        else:
            self.model_path = model_path or default_cnn
            self.model = CandlestickCNN(num_classes=3)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.gradcam: Optional[GradCAM] = None
        self.is_loaded = False
        self._load_weights()

    def _load_weights(self) -> None:
        if self.model_path.is_file():
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                # Attach GradCAM
                target_layer = self.model.get_target_layer()
                self.gradcam = GradCAM(self.model, target_layer)
                self.is_loaded = True
                print(f"VisionPatternService: Loaded {self.model_type} from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load vision weights from {self.model_path}: {e}")
                self.is_loaded = False
        else:
            print(f"Notice: Vision weights not found at {self.model_path}, initialized in untrained mode.")
            self.model.to(self.device)
            self.model.eval()
            target_layer = self.model.get_target_layer()
            self.gradcam = GradCAM(self.model, target_layer)
            self.is_loaded = False

    def predict_image(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs pattern classification and generates Grad-CAM overlay for a PIL image.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Grad-CAM requires gradients
        if self.gradcam is None:
            self.gradcam = GradCAM(self.model, self.model.get_target_layer())

        cam, pred_class, confidence = self.gradcam.generate(input_tensor, target_class=target_class)
        overlay_pil = overlay_gradcam_on_image(image.resize((224, 224)), cam, alpha=0.55)
        overlay_b64 = image_to_base64(overlay_pil)

        prob_dict = {
            LABEL_NAMES[i]: float(probs[i]) for i in range(3)
        }

        return {
            "pattern": LABEL_NAMES[pred_class],
            "pattern_class": pred_class,
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "gradcam_overlay_base64": overlay_b64,
            "model_architecture": self.model_type,
            "is_trained": self.is_loaded,
        }

    def predict_from_scheme(
        self,
        scheme_code: str,
        lookback_days: int = 40,
    ) -> Dict[str, Any]:
        """
        Renders latest chart for mutual fund scheme and classifies pattern.
        """
        code = str(scheme_code)
        cache_file = EXPORT_DIR / "nav_cache" / f"hist_{code}.json"
        data = None
        if cache_file.is_file():
            try:
                with open(cache_file, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = None

        if not data:
            from lume_platform.data.nav_fetcher import nav_fetcher
            data = nav_fetcher.get_historical_nav(code, days=max(lookback_days * 3, 180))

        if not data:
            raise FileNotFoundError(f"Scheme history not found for {code} at {cache_file}")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)

        if len(df) < lookback_days:
            raise ValueError(f"Insufficient historical days ({len(df)}) for scheme {code}")

        recent = df.tail(lookback_days).copy()
        nav_vals = recent["nav"].values
        seed = int(code) % 10000 if code.isdigit() else (abs(hash(code)) % 10000)
        opens, highs, lows, closes = synthesize_ohlc_from_nav(nav_vals, seed=seed)

        chart_img = render_candlestick_image(opens, highs, lows, closes)
        result = self.predict_image(chart_img)
        result["scheme_code"] = code
        result["date_range"] = {
            "start": str(recent["date"].iloc[0].date()),
            "end": str(recent["date"].iloc[-1].date()),
        }
        result["latest_nav"] = float(nav_vals[-1])
        return result

