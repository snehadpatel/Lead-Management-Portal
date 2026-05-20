"""Semantic search helpers using Sentence-BERT for mutual fund lookup.

This module expects a cached embeddings pickle at initialization containing:
  {
    "funds": [ {"scheme_code":..., "scheme_name":..., "category":...}, ... ],
    "embeddings": np.ndarray of shape (N, D)
  }

If the cache is missing, the class will attempt to lazily load a SentenceTransformer
model for encoding queries but will require a precomputed cache for fast search.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

class SBERTMutualFundSearch:
    def __init__(self, cache_path: Path | str, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.model = None
        self.funds: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        if self.cache_path.is_file():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.funds = data.get('funds', [])
                    self.embeddings = np.asarray(data.get('embeddings'))
            except Exception:
                self.funds = []
                self.embeddings = None

    def _ensure_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k matching funds with cosine similarity scores.

        Each result is a dict: {scheme_code, scheme_name, category, match_score}
        """
        if not text or (self.embeddings is None) or (len(self.funds) == 0):
            return []

        self._ensure_model()
        if self.model is None:
            return []

        q_emb = self.model.encode(text, convert_to_numpy=True)
        # Normalize
        try:
            q_norm = q_emb / np.linalg.norm(q_emb)
            emb_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            sims = emb_norms.dot(q_norm)
        except Exception:
            sims = np.dot(self.embeddings, q_emb)

        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            score = float(sims[i]) if i < len(sims) else 0.0
            fund = self.funds[int(i)] if i < len(self.funds) else {}
            results.append({
                "scheme_code": fund.get("scheme_code", f"F{i}"),
                "scheme_name": fund.get("scheme_name", fund.get("name", "Unknown Fund")),
                "category": fund.get("category", "Unknown"),
                "match_score": round(score, 4)
            })

        return results
