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
        self.finbert_pipeline = None
        self.latest_signals: list[dict] = []
        self.forecaster = None
        self.sbert_search = None
        self.buddy = None
        self.is_loaded = False

    def _load_pickle(self, name: str) -> Any:
        primary = self.models_dir / name
        if primary.is_file():
            with open(primary, "rb") as f: return pickle.load(f)
        return None

    def load(self, force: bool = False) -> None:
        if self.is_loaded and not force:
            return
        self.is_loaded = False
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

    def load_sbert_cache(self, cache_path: str | Path | None = None) -> bool:
        """Load or reload SBERT mutual fund embeddings from a cache file.

        Returns True if loaded successfully, False otherwise.
        """
        try:
            from lume_platform.ml.semantic_search import SBERTMutualFundSearch
            cp = Path(cache_path) if cache_path else (self.models_dir / "fund_embeddings.pkl")
            if not cp.is_file():
                return False
            self.sbert_search = SBERTMutualFundSearch(cp)
            return True
        except Exception:
            self.sbert_search = None
            return False

    def reload(self) -> None:
        """Force reload all model bundles from disk."""
        self.load(force=True)

        # Attempt to load FinBERT / transformers sentiment pipeline (if available)
        try:
            from transformers import pipeline
            finbert_model = os.getenv("FINBERT_MODEL", "yiyanghkust/finbert-tone")
            try:
                self.finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, device=-1)
                print(f"✅ Loaded FinBERT pipeline: {finbert_model}")
            except Exception as e:
                print(f"⚠️ Failed to init FinBERT pipeline '{finbert_model}': {e}")
                self.finbert_pipeline = None
        except Exception:
            self.finbert_pipeline = None

    def predict_sentiment(self, text: str) -> tuple[str, float]:
        """Unified sentiment prediction: prefer serialized sentiment bundle, then FinBERT pipeline, then simple fallback heuristic."""
        # 1) Use pickled sentiment bundle if available
        try:
            if self.sentiment_bundle is not None and hasattr(self.sentiment_bundle, 'predict_text'):
                try:
                    label, conf = self.sentiment_bundle.predict_text(text)
                    return label, float(conf)
                except Exception:
                    pass

            # 2) Use transformers FinBERT pipeline
            if self.finbert_pipeline is not None:
                try:
                    out = self.finbert_pipeline(text[:1000])[0]
                    label = out.get('label')
                    score = float(out.get('score', 0.0))
                    return label, score
                except Exception:
                    pass

        except Exception:
            pass

        # 3) Fallback heuristic
        text_lower = (text or "").lower()
        positive_words = ["good", "bull", "growth", "buy", "up", "high", "positive", "gain", "profit", "recommend", "great", "best", "benefit", "outperform", "bullish", "strong"]
        negative_words = ["bad", "bear", "loss", "sell", "down", "low", "negative", "drop", "risk", "panic", "crash", "fall", "pause", "drawdown", "bearish", "weak"]
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive", 0.80
        elif neg_count > pos_count:
            return "negative", 0.80
        else:
            return "neutral", 0.50

    def add_market_signal(self, signal: dict, max_len: int = 250) -> None:
        """Store incoming market signals in a capped in-memory buffer."""
        try:
            if not isinstance(signal, dict):
                return
            self.latest_signals.insert(0, signal)
            # cap buffer
            if len(self.latest_signals) > max_len:
                self.latest_signals = self.latest_signals[:max_len]
        except Exception:
            pass
