"""Load all pickle bundles once (API / Streamlit startup)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from lume_platform.config import DATA_ROOT, MODELS_DIR, ARTIFACTS_DIR
from lume_platform.ml.bundles import InvestorClusterBundle, LeadScoringPipelineBundle, SentimentBundle
from lume_platform.ml.forecaster import LUMEForecaster


class ModelRegistry:
    _instance: ModelRegistry | None = None

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.legacy_dir = DATA_ROOT / "models/saved_models"
        self.lead_bundle: LeadScoringPipelineBundle | None = None
        self.investor_bundle: InvestorClusterBundle | None = None
        self.sentiment_bundle: SentimentBundle | None = None
        self.forecaster: LUMEForecaster | None = None

    def _load_pickle(self, name: str) -> Any:
        primary = self.models_dir / name
        if primary.is_file():
            with open(primary, "rb") as f:
                return pickle.load(f)
        alt = self.legacy_dir / name
        if alt.is_file():
            with open(alt, "rb") as f:
                return pickle.load(f)
        return None

    def load(self) -> None:
        raw = self._load_pickle("lead_classifier_bundle.pkl")
        if isinstance(raw, LeadScoringPipelineBundle):
            self.lead_bundle = raw
        self.investor_bundle = self._load_pickle("investor_cluster_bundle.pkl")
        if not isinstance(self.investor_bundle, InvestorClusterBundle):
            self.investor_bundle = None
        self.sentiment_bundle = self._load_pickle("sentiment_bundle.pkl")
        if not isinstance(self.sentiment_bundle, SentimentBundle):
            self.sentiment_bundle = None
            
        # LSTM Forecaster
        lstm_path = self.models_dir / "lstm_nav_pattern_predictor.pth"
        scaler_path = (ARTIFACTS_DIR / "ml_scalers") / "mf_nav_global_scaler.pkl"
        if lstm_path.is_file() and scaler_path.is_file():
            self.forecaster = LUMEForecaster(lstm_path, scaler_path)
            try:
                self.forecaster.load()
            except Exception:
                self.forecaster = None
