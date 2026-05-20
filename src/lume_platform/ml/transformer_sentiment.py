"""Transformer-based financial sentiment analysis using HuggingFace DistilBERT.

Uses `distilbert-base-uncased-finetuned-sst-2-english` for general sentiment classification.
Falls back gracefully if model can't be loaded.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton to avoid reloading the model on every request
_analyzer_instance: Optional["TransformerSentimentAnalyzer"] = None


class TransformerSentimentAnalyzer:
    """Wraps a HuggingFace pipeline for sentiment analysis with caching."""

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self) -> None:
        self.pipeline = None
        self._available = False
        try:
            from transformers import pipeline as hf_pipeline

            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=-1,  # CPU — safe for all environments
                truncation=True,
                max_length=512,
            )
            self._available = True
            logger.info("TransformerSentimentAnalyzer loaded: %s", self.MODEL_NAME)
        except Exception as exc:
            logger.warning("Could not load transformer sentiment model: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, text: str) -> tuple[str, float]:
        """Return (label, confidence) for a single text.

        Labels are normalized to: 'Positive', 'Negative', 'Neutral'.
        """
        if not self._available or not text.strip():
            return "Neutral", 0.5

        try:
            result = self.pipeline(text[:512])[0]
            raw_label = result["label"]  # 'POSITIVE' or 'NEGATIVE'
            score = float(result["score"])

            # Normalize labels
            if raw_label == "POSITIVE":
                label = "Positive"
            elif raw_label == "NEGATIVE":
                label = "Negative"
            else:
                label = "Neutral"

            # If confidence is low (< 0.6), treat as neutral
            if score < 0.6:
                label = "Neutral"

            return label, round(score, 4)
        except Exception as exc:
            logger.warning("Transformer sentiment prediction failed: %s", exc)
            return "Neutral", 0.5

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Batch prediction for multiple texts."""
        if not self._available:
            return [("Neutral", 0.5) for _ in texts]

        try:
            results = self.pipeline([t[:512] for t in texts], batch_size=8)
            output = []
            for result in results:
                raw_label = result["label"]
                score = float(result["score"])
                if raw_label == "POSITIVE":
                    label = "Positive"
                elif raw_label == "NEGATIVE":
                    label = "Negative"
                else:
                    label = "Neutral"
                if score < 0.6:
                    label = "Neutral"
                output.append((label, round(score, 4)))
            return output
        except Exception as exc:
            logger.warning("Batch sentiment prediction failed: %s", exc)
            return [("Neutral", 0.5) for _ in texts]


def get_transformer_sentiment() -> TransformerSentimentAnalyzer:
    """Get or create the singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TransformerSentimentAnalyzer()
    return _analyzer_instance
