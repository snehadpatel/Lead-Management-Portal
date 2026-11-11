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

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from lume_platform.config import ARTIFACTS_DIR


class SBERTMutualFundSearch:
    def __init__(
        self,
        cache_path: Path | str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.cache_path = Path(cache_path)
        local_minilm = ARTIFACTS_DIR / "hf_models" / "all-MiniLM-L6-v2"
        if model_name == "sentence-transformers/all-MiniLM-L6-v2" and local_minilm.is_dir():
            self.model_name = str(local_minilm)
        else:
            self.model_name = model_name
        self.model = None
        self.funds: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        if self.cache_path.is_file():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.funds = data.get('funds', [])
                    if data.get('embeddings') is not None:
                        self.embeddings = np.asarray(data.get('embeddings'))
            except Exception:
                self.funds = []
                self.embeddings = None

        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        """Build fast zero-deadlock TF-IDF cosine index over loaded fund corpus."""
        if not self.funds:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            docs = [
                f"{f.get('scheme_name', f.get('name', ''))} {f.get('category', '')} {f.get('amc', '')} {f.get('description', '')}".strip()
                for f in self.funds
            ]
            self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(docs)
        except Exception:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None

    def _ensure_model(self):
        # Guard against macOS/sandbox transformers mutex deadlocks unless explicitly enabled
        if self.model is None and os.environ.get("LUME_ENABLE_TRANSFORMERS_SBERT", "0") == "1":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k matching funds with cosine similarity scores.

        Each result is a dict: {scheme_code, scheme_name, category, match_score}
        """
        if not text or (len(self.funds) == 0):
            return []

        sims = None
        self._ensure_model()

        # 1. Use dense SBERT embeddings if encoder is active & dims align
        if self.model is not None and self.embeddings is not None:
            try:
                q_emb = self.model.encode(text, convert_to_numpy=True)
                if q_emb.shape[0] == self.embeddings.shape[1]:
                    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
                    emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
                    sims = emb_norms.dot(q_norm)
            except Exception:
                sims = None

        # 2. Fast offline/safe TF-IDF cosine similarity + intent synonym expansion
        if sims is None and self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_lower = text.lower()
            expansions = []
            if "conservative" in q_lower or "safe" in q_lower or "liquid" in q_lower:
                expansions.append("liquid overnight ultra short duration debt income gilt corporate bond")
            if "growth" in q_lower or "aggressive" in q_lower or "alpha" in q_lower:
                expansions.append("mid cap small cap flexi cap equity growth opportunities focused")
            if "passive" in q_lower or "index" in q_lower or "etf" in q_lower:
                expansions.append("nifty 50 index etf sensex equal weight")
            if "balanced" in q_lower or "moderate" in q_lower or "hybrid" in q_lower:
                expansions.append("balanced advantage hybrid multi asset dynamic asset allocation")
            if "tax" in q_lower or "elss" in q_lower:
                expansions.append("elss tax saver long term equity")
            expanded_query = f"{text} {' '.join(expansions)}".strip()
            q_vec = self._tfidf_vectorizer.transform([expanded_query])
            sims = cosine_similarity(self._tfidf_matrix, q_vec).ravel()

        # 3. Last-resort token overlap
        if sims is None:
            tokens = [t for t in text.lower().split() if len(t) > 2]
            sims = np.array([
                sum(1 for t in tokens if t in f"{f.get('scheme_name', '')} {f.get('category', '')}".lower()) / max(len(tokens), 1)
                for f in self.funds
            ], dtype=float)

        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            score = float(sims[i]) if i < len(sims) else 0.0
            fund = self.funds[int(i)] if i < len(self.funds) else {}
            results.append({
                "scheme_code": str(fund.get("scheme_code", f"F{i}")),
                "scheme_name": fund.get("scheme_name", fund.get("name", "Unknown Fund")),
                "category": fund.get("category", "Unknown"),
                "match_score": round(max(0.0, score), 4)
            })

        return results
