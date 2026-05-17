"""Artifacts for inference (pickle-friendly)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LeadScoringPipelineBundle:
    """Sklearn Pipeline with ColumnTransformer + classifier; column-safe inference."""

    pipeline: Any
    numeric_features: list[str]
    cat_features: list[str]
    decision_threshold: float = 0.5

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.cat_features

    def _frame(self, row: dict) -> pd.DataFrame:
        frame = pd.DataFrame([row])
        for c in self.numeric_features:
            if c not in frame.columns:
                frame[c] = 0
            frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0)
        for c in self.cat_features:
            if c not in frame.columns:
                frame[c] = "Unknown"
            frame[c] = frame[c].fillna("Unknown").astype(str)
        return frame[self.feature_columns]

    def predict_row(self, row: dict) -> tuple[int, float]:
        X = self._frame(row)
        if hasattr(self.pipeline, "predict_proba"):
            proba = float(self.pipeline.predict_proba(X)[0][1])
            pred = int(proba >= float(self.decision_threshold))
        else:
            pred = int(self.pipeline.predict(X)[0])
            proba = float(pred)
        return pred, proba


@dataclass
class InvestorClusterBundle:
    scaler: Any
    kmeans: Any
    behavior_cols: list[str]

    def predict_row(self, row: dict) -> int:
        frame = pd.DataFrame([row])
        x = frame[self.behavior_cols].astype(float).values
        xs = self.scaler.transform(x)
        return int(self.kmeans.predict(xs)[0])


@dataclass
class SentimentBundle:
    vectorizer: Any
    model: Any
    label_classes: np.ndarray

    def predict_text(self, text: str) -> tuple[str, float]:
        vec = self.vectorizer.transform([text])
        pred_idx = int(self.model.predict(vec)[0])
        label = str(self.label_classes[pred_idx])
        if hasattr(self.model, "predict_proba"):
            proba = float(np.max(self.model.predict_proba(vec)))
        else:
            proba = 1.0
        return label, proba


@dataclass
class SBERTSentimentBundle:
    """Uses SBERT embeddings + Classifier for sentiment."""
    model: Any
    label_classes: np.ndarray
    sbert_model_name: str = 'all-MiniLM-L6-v2'

    def predict_text(self, text: str, sbert_model: Any) -> tuple[str, float]:
        """sbert_model should be an instance of SentenceTransformer passed from registry."""
        embedding = sbert_model.encode([text], convert_to_numpy=True)
        pred_idx = int(self.model.predict(embedding)[0])
        label = str(self.label_classes[pred_idx])
        if hasattr(self.model, "predict_proba"):
            proba = float(np.max(self.model.predict_proba(embedding)))
        else:
            proba = 1.0
        return label, proba
