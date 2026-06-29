"""
Lume AI — MVP FastAPI Backend
Slim, public demo API. No auth, no MongoDB, no Kafka.
Serves 5 pre-trained model demos + cached evaluation metrics.
"""

from __future__ import annotations

import json
import os
import sys
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np

# ── Rate Limiting ──────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# ── Path setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("LUME_PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
ML_ARTIFACTS = Path(os.environ.get("LUME_ML_ARTIFACTS", PROJECT_ROOT / "ml-artifacts"))
MODELS_DIR = ML_ARTIFACTS / "models"
SCALERS_DIR = ML_ARTIFACTS / "ml_scalers"
EVAL_DIR = ML_ARTIFACTS / "evaluations"

# Add src to path for model imports
SRC_DIR = Path(os.environ.get("LUME_SRC_DIR", PROJECT_ROOT.parent / "src"))
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

# ── Model Loading ──────────────────────────────────────────────────────────
import pickle

# Load model bundles at module level (cold-start friendly)
_lead_bundle = None
_investor_bundle = None
_sentiment_bundle = None
_forecaster = None
_sbert_search = None
_models_status: Dict[str, bool] = {}


def _load_pickle(path: Path):
    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_models():
    """Load all model bundles once at startup."""
    global _lead_bundle, _investor_bundle, _sentiment_bundle, _forecaster, _sbert_search, _models_status

    print("📥 Loading AI Model Bundles for MVP demo...")

    # Lead scoring
    try:
        _lead_bundle = _load_pickle(MODELS_DIR / "lead_classifier_bundle.pkl")
        _models_status["lead_scoring"] = _lead_bundle is not None
        if _lead_bundle:
            print("  ✅ Lead scoring bundle loaded")
    except Exception as e:
        print(f"  ⚠️ Lead scoring failed: {e}")
        _models_status["lead_scoring"] = False

    # Investor clustering
    try:
        _investor_bundle = _load_pickle(MODELS_DIR / "investor_cluster_bundle.pkl")
        _models_status["investor_cluster"] = _investor_bundle is not None
        if _investor_bundle:
            print("  ✅ Investor cluster bundle loaded")
    except Exception as e:
        print(f"  ⚠️ Investor cluster failed: {e}")
        _models_status["investor_cluster"] = False

    # Sentiment
    try:
        _sentiment_bundle = _load_pickle(MODELS_DIR / "sentiment_bundle.pkl")
        _models_status["sentiment"] = _sentiment_bundle is not None
        if _sentiment_bundle:
            print("  ✅ Sentiment bundle loaded")
    except Exception as e:
        print(f"  ⚠️ Sentiment failed: {e}")
        _models_status["sentiment"] = False

    # LSTM Forecaster
    try:
        lstm_path = MODELS_DIR / "lstm_nav_pattern_predictor.pth"
        scaler_path = SCALERS_DIR / "mf_nav_global_scaler.pkl"
        if lstm_path.is_file() and scaler_path.is_file():
            import torch
            import joblib

            class NAVPredictorLSTM(torch.nn.Module):
                def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=5):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                    self.fc = torch.nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    return self.fc(out[:, -1, :])

            model = NAVPredictorLSTM(input_size=1, output_size=5)
            model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
            model.eval()
            scaler = joblib.load(scaler_path)
            _forecaster = {"model": model, "scaler": scaler}
            _models_status["forecaster"] = True
            print("  ✅ LSTM forecaster loaded")
        else:
            _models_status["forecaster"] = False
    except Exception as e:
        print(f"  ⚠️ Forecaster failed: {e}")
        _models_status["forecaster"] = False

    # SBERT Search
    try:
        cache_path = MODELS_DIR / "fund_embeddings.pkl"
        if cache_path.is_file():
            _sbert_search = _load_pickle(cache_path)
            _models_status["semantic_search"] = _sbert_search is not None
            if _sbert_search:
                print("  ✅ SBERT search cache loaded")
        else:
            _models_status["semantic_search"] = False
            print("  ⚠️ SBERT fund_embeddings.pkl not found — search disabled")
    except Exception as e:
        print(f"  ⚠️ SBERT search failed: {e}")
        _models_status["semantic_search"] = False

    print(f"📊 Models loaded: {_models_status}")


# ── Sentiment Fallback ─────────────────────────────────────────────────────
def _fallback_sentiment(text: str) -> tuple[str, float]:
    """Keyword-based fallback when the trained sentiment bundle is unavailable."""
    t = (text or "").lower()
    pos = ["good", "bull", "growth", "buy", "up", "high", "positive", "gain",
           "profit", "recommend", "great", "best", "benefit", "outperform",
           "bullish", "strong", "rally", "surge", "boom", "optimistic"]
    neg = ["bad", "bear", "loss", "sell", "down", "low", "negative", "drop",
           "risk", "panic", "crash", "fall", "pause", "drawdown", "bearish",
           "weak", "decline", "recession", "inflation", "slump"]
    pc = sum(1 for w in pos if w in t)
    nc = sum(1 for w in neg if w in t)
    if pc > nc:
        return "positive", 0.80
    elif nc > pc:
        return "negative", 0.80
    return "neutral", 0.50


def predict_sentiment(text: str) -> tuple[str, float]:
    """Use trained bundle, then fallback."""
    if _sentiment_bundle is not None and hasattr(_sentiment_bundle, "predict_text"):
        try:
            label, conf = _sentiment_bundle.predict_text(text)
            return str(label), float(conf)
        except Exception:
            pass
    return _fallback_sentiment(text)


# ── Cached Metrics Loading ─────────────────────────────────────────────────
@functools.lru_cache(maxsize=16)
def _load_json(path_str: str) -> dict:
    p = Path(path_str)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return {}


def get_all_metrics() -> dict:
    return {
        "random_forest": _load_json(str(EVAL_DIR / "rf_real_metrics.json")),
        "classification_report": _load_json(str(EVAL_DIR / "rf_classification_report.json")),
        "confusion_matrix": _load_json(str(EVAL_DIR / "rf_confusion_matrix.json")),
        "kmeans": _load_json(str(EVAL_DIR / "kmeans_metrics.json")),
        "kmeans_pca": _load_json(str(EVAL_DIR / "kmeans_pca_data.json")),
        "sentiment": _load_json(str(EVAL_DIR / "nlp_metrics.json")),
        "lstm": _load_json(str(EVAL_DIR / "lstm_metrics.json")),
        "lstm_holdout": _load_json(str(EVAL_DIR / "lstm_holdout_data.json")),
        "manifest": _load_json(str(MODELS_DIR / "model_manifest.json")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

VALID_TASKS = ["lead_scoring", "investor_cluster", "sentiment"]


class LeadFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    totalvisits: int = Field(0, alias="TotalVisits", ge=0, le=100000)
    total_time_spent_on_website: float = Field(0, alias="Total Time Spent on Website", ge=0, le=100000)
    page_views_per_visit: float = Field(0, alias="Page Views Per Visit", ge=0, le=1000)
    asymmetrique_activity_score: float = Field(0, alias="Asymmetrique Activity Score", ge=0, le=100)
    asymmetrique_profile_score: float = Field(0, alias="Asymmetrique Profile Score", ge=0, le=100)
    lead_origin: str = Field("Unknown", alias="Lead Origin", max_length=100)
    lead_source: str = Field("Unknown", alias="Lead Source", max_length=100)
    specialization: str = Field("Unknown", alias="Specialization", max_length=100)
    occupation: str = Field("Unknown", alias="What is your current occupation", max_length=100)
    last_activity: str = Field("Unknown", alias="Last Activity", max_length=100)
    country: str = Field("Unknown", alias="Country", max_length=100)
    lead_quality: str = Field("Unknown", alias="Lead Quality", max_length=100)
    do_not_email: str = Field("Unknown", alias="Do Not Email", max_length=10)
    do_not_call: str = Field("Unknown", alias="Do Not Call", max_length=10)


class InvestorBehavior(BaseModel):
    ProfManage: float = Field(5.0, ge=0, le=10)
    Diversification: float = Field(5.0, ge=0, le=10)
    Affordability: float = Field(5.0, ge=0, le=10)
    Liquidity: float = Field(5.0, ge=0, le=10)
    Growth: float = Field(5.0, ge=0, le=10)
    Trustworthiness: float = Field(5.0, ge=0, le=10)
    Technology: float = Field(5.0, ge=0, le=10)


class PredictRequest(BaseModel):
    task: str = Field(..., description="lead_scoring | investor_cluster | sentiment")
    lead: Optional[LeadFeatures] = None
    investor_behavior: Optional[InvestorBehavior] = None
    text: Optional[str] = Field(None, max_length=500)

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=200, min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class PredictionResponse(BaseModel):
    task: str
    timestamp: str
    prediction: Any
    confidence: Optional[float] = None
    explanation: str = ""
    disclaimer: str = "Demo using synthetic/sample data. Not financial advice."
    model_version: str = "2.0.0"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str = "2.0.0-mvp"


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")

app = FastAPI(
    title="Lume AI — MVP Demo API",
    description="Public demo API for Lume AI mutual fund intelligence models.",
    version="2.0.0-mvp",
    docs_url="/docs",
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — locked to allowed origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGIN.split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def startup():
    load_models()


# ── Global error handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal error occurred. The demo may be temporarily unavailable.",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=_models_status.copy(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@limiter.limit("20/minute")
def predict(request: PredictRequest, req: Request):
    """
    Single prediction endpoint for lead scoring, investor clustering, or sentiment.
    Rate limited to 20 requests/minute per IP.
    """
    timestamp = datetime.utcnow().isoformat()

    # ── Lead Scoring ───────────────────────────────────────────────────
    if request.task == "lead_scoring":
        if _lead_bundle is None:
            raise HTTPException(status_code=503, detail="Lead scoring model not loaded. Demo temporarily unavailable.")
        if request.lead is None:
            raise HTTPException(status_code=422, detail="Missing 'lead' field for lead_scoring task.")

        lead_dict = request.lead.model_dump(by_alias=True)
        try:
            pred, proba = _lead_bundle.predict_row(lead_dict)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Prediction failed. Demo temporarily unavailable.")

        # Plain-English explanation
        if proba >= 0.85:
            expl = f"This lead has a {proba:.0%} probability of converting — a Hot lead. High activity scores and website engagement are the strongest indicators."
        elif proba >= 0.65:
            expl = f"This lead shows a {proba:.0%} conversion probability — Warm. Moderate engagement signals suggest follow-up could push conversion."
        else:
            expl = f"This lead has a {proba:.0%} conversion probability — Cold. Low engagement or mismatched profile characteristics reduce the likelihood."

        return PredictionResponse(
            task="lead_scoring",
            timestamp=timestamp,
            prediction={"converted": bool(pred), "conversion_probability": round(proba, 4)},
            confidence=round(proba, 4),
            explanation=expl,
        )

    # ── Investor Clustering ────────────────────────────────────────────
    elif request.task == "investor_cluster":
        if _investor_bundle is None:
            raise HTTPException(status_code=503, detail="Clustering model not loaded. Demo temporarily unavailable.")
        if request.investor_behavior is None:
            raise HTTPException(status_code=422, detail="Missing 'investor_behavior' for investor_cluster task.")

        behavior_dict = request.investor_behavior.model_dump()
        try:
            cluster_id = _investor_bundle.predict_row(behavior_dict)
        except Exception:
            raise HTTPException(status_code=500, detail="Prediction failed. Demo temporarily unavailable.")

        personas = {
            0: {"name": "Growth Seekers", "desc": "High-risk equity investors focused on capital appreciation. Prioritize growth over stability and prefer actively managed funds.", "color": "#ef4444"},
            1: {"name": "Safety-First Savers", "desc": "Conservative investors preferring liquid and debt funds. Value capital preservation, low volatility, and predictable returns.", "color": "#22c55e"},
            2: {"name": "Balanced Allocators", "desc": "Moderate risk-takers using hybrid allocation strategies. Seek a mix of equity growth and debt stability.", "color": "#3b82f6"},
            3: {"name": "Passive Indexers", "desc": "Cost-conscious investors tracking market indices. Prefer low-expense ETFs and index funds with minimal active management.", "color": "#f59e0b"},
        }
        persona = personas.get(int(cluster_id), {"name": "Unknown", "desc": "Unclassified investor profile.", "color": "#6b7280"})

        return PredictionResponse(
            task="investor_cluster",
            timestamp=timestamp,
            prediction={"cluster_id": int(cluster_id), "persona": persona["name"], "description": persona["desc"], "color": persona["color"]},
            explanation=f"Based on the behavioral profile, this investor maps to the '{persona['name']}' segment — {persona['desc'].split('.')[0].lower()}.",
        )

    # ── Sentiment ──────────────────────────────────────────────────────
    elif request.task == "sentiment":
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=422, detail="Missing or empty 'text' field for sentiment task.")

        label, conf = predict_sentiment(request.text.strip())
        expl_map = {
            "positive": "The text expresses a positive or bullish market outlook.",
            "negative": "The text signals negative sentiment or bearish market conditions.",
            "neutral": "The text appears neutral with no strong directional sentiment.",
        }
        return PredictionResponse(
            task="sentiment",
            timestamp=timestamp,
            prediction={"sentiment": label, "confidence": round(conf, 4)},
            confidence=round(conf, 4),
            explanation=expl_map.get(label.lower(), "Sentiment classified."),
        )

    raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")


@app.post("/search", tags=["Search"])
@limiter.limit("20/minute")
def search_funds(request: SearchRequest, req: Request):
    """Semantic fund search using SBERT embeddings."""
    if _sbert_search is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic search is not available. The fund embeddings cache is missing.",
        )

    try:
        # Expect _sbert_search to be a dict with 'funds' and 'embeddings'
        from sentence_transformers import SentenceTransformer

        funds = _sbert_search.get("funds", [])
        embeddings = np.asarray(_sbert_search.get("embeddings"))

        if len(funds) == 0 or embeddings.size == 0:
            raise HTTPException(status_code=503, detail="Search index is empty.")

        # Lazy-load the model
        if not hasattr(search_funds, "_model"):
            search_funds._model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        q_emb = search_funds._model.encode(request.query, convert_to_numpy=True)
        q_norm = q_emb / np.linalg.norm(q_emb)
        emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = emb_norms.dot(q_norm)

        idx = np.argsort(-sims)[: request.top_k]
        results = []
        for i in idx:
            fund = funds[int(i)] if int(i) < len(funds) else {}
            results.append({
                "scheme_code": fund.get("scheme_code", f"F{i}"),
                "scheme_name": fund.get("scheme_name", fund.get("name", "Unknown Fund")),
                "category": fund.get("category", "Unknown"),
                "match_score": round(float(sims[i]), 4),
            })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "query": request.query,
            "results": results,
            "disclaimer": "Demo using synthetic/sample data. Not financial advice.",
        }
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=503, detail="SentenceTransformers not installed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search failed. Demo temporarily unavailable.")


@app.get("/forecast/demo", tags=["Forecast"])
def forecast_demo():
    """Return pre-computed LSTM holdout data for the interactive chart."""
    data = _load_json(str(EVAL_DIR / "lstm_holdout_data.json"))
    if not data:
        raise HTTPException(status_code=503, detail="Forecast data not available.")
    data["disclaimer"] = "Demo using synthetic/sample data. Not financial advice."
    data["timestamp"] = datetime.utcnow().isoformat()
    return data


@app.get("/metrics", tags=["Metrics"])
def all_metrics():
    """Return all model evaluation metrics for the Insights page."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        **get_all_metrics(),
    }


@app.get("/metrics/{model}", tags=["Metrics"])
def model_metrics(model: str):
    """Return metrics for a specific model."""
    all_m = get_all_metrics()
    if model not in all_m:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}. Available: {list(all_m.keys())}")
    return {"timestamp": datetime.utcnow().isoformat(), model: all_m[model]}
