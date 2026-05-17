"""Load all pickle bundles once (API / Streamlit startup)."""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any
from lume_platform.config import DATA_ROOT, MODELS_DIR, ARTIFACTS_DIR
from lume_platform.ml.bundles import InvestorClusterBundle, LeadScoringPipelineBundle, SentimentBundle, SBERTSentimentBundle
from lume_platform.ml.custom_buddy_model import LUMEBuddy

class ModelRegistry:
    _instance: ModelRegistry | None = None

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
            # Removed automatic .load() call to prevent startup deadlocks
        return cls._instance

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.legacy_dir = DATA_ROOT / "models/saved_models"
        self.lead_bundle = None
        self.investor_bundle = None
        self.sentiment_bundle = None
        self.forecaster = None
        self.sbert_search = None
        self.buddy = None
        self.is_loaded = False

    def _load_pickle(self, name: str) -> Any:
        primary = self.models_dir / name
        if primary.is_file():
            with open(primary, "rb") as f: return pickle.load(f)
        return None

    def load(self) -> None:
        if self.is_loaded: return
        print("📥 Loading AI Model Bundles...")
        raw = self._load_pickle("lead_classifier_bundle.pkl")
        if isinstance(raw, LeadScoringPipelineBundle):
            self.lead_bundle = raw
        self.investor_bundle = self._load_pickle("investor_cluster_bundle.pkl")
        self.sentiment_bundle = self._load_pickle("sentiment_bundle.pkl")

        if os.getenv("LUME_ENABLE_FORECASTER", "0") == "1":
            lstm_path = self.models_dir / "lstm_nav_pattern_predictor.pth"
            scaler_path = (ARTIFACTS_DIR / "ml_scalers") / "mf_nav_global_scaler.pkl"
            if lstm_path.is_file() and scaler_path.is_file():
                try:
                    from lume_platform.ml.forecaster import LUMEForecaster

                    self.forecaster = LUMEForecaster(lstm_path, scaler_path)
                    self.forecaster.load()
                except Exception:
                    self.forecaster = None

        if os.getenv("LUME_ENABLE_SEMANTIC_SEARCH", "0") == "1":
            sbert_cache = self.models_dir / "fund_embeddings.pkl"
            if sbert_cache.is_file():
                try:
                    from lume_platform.ml.semantic_search import SBERTMutualFundSearch

                    self.sbert_search = SBERTMutualFundSearch(sbert_cache)
                except Exception:
                    self.sbert_search = None
        
        buddy_path = self.models_dir / "custom_buddy_model.pth"
        if buddy_path.is_file():
            try:
                self.buddy = LUMEBuddy(str(buddy_path))
                self.buddy.load()
            except Exception as e:
                print(f"⚠️ Failed to load Buddy model: {e}")
                self.buddy = None

        self.is_loaded = True
