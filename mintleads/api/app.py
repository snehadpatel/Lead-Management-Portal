"""MintLeads Flask REST API.

This module provides a RESTful API for the MintLeads system:
- Lead scoring endpoint
- Investor segmentation endpoint
- NAV forecasting endpoint
- Sentiment analysis endpoint
- Health check endpoint

All models are loaded at startup for efficient serving.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Optional imports
try:
    from flask_caching import Cache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    Cache = None

try:
    from marshmallow import Schema, fields, validate
    MARSHMALLOW_AVAILABLE = True
except ImportError:
    MARSHMALLOW_AVAILABLE = False
    Schema = object
    fields = None
    validate = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from config import (
    CACHE_TTL,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    HOT_LEAD_THRESHOLD,
    INVESTOR_CLUSTER_DIR,
    LEAD_SCORER_DIR,
    NAV_FORECASTER_DIR,
    NEWS_STREAM_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    WARM_LEAD_THRESHOLD,
    setup_logging,
)

logger = setup_logging(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure caching
cache_config = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_REDIS_HOST": REDIS_HOST,
    "CACHE_REDIS_PORT": REDIS_PORT,
    "CACHE_REDIS_DB": REDIS_DB,
    "CACHE_DEFAULT_TIMEOUT": CACHE_TTL,
}

# Fallback to simple cache if Redis not available
if REDIS_HOST == "localhost" and REDIS_AVAILABLE:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_connect_timeout=1)
        r.ping()
        app.config.from_mapping(cache_config)
    except Exception:
        app.config["CACHE_TYPE"] = "SimpleCache"
        app.config["CACHE_DEFAULT_TIMEOUT"] = CACHE_TTL
else:
    app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = CACHE_TTL

# Initialize cache if available, otherwise create dummy
if CACHING_AVAILABLE:
    cache = Cache(app)
else:
    # Dummy cache class that does nothing
    class DummyCache:
        def cached(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    cache = DummyCache()

# Global model storage
models: Dict[str, Any] = {
    "lead_scorer": None,
    "investor_cluster": None,
    "nav_forecaster": None,
    "cluster_mapping": None,
    "lead_preprocessing": None,
}


def load_models() -> Dict[str, Any]:
    """Load all ML models at startup.
    
    Returns:
        Dictionary with loaded models.
    """
    logger.info("Loading ML models...")
    
    # Load Lead Scorer
    lead_model_path = LEAD_SCORER_DIR / "rf_model.pkl"
    if lead_model_path.exists():
        with open(lead_model_path, "rb") as f:
            models["lead_scorer"] = pickle.load(f)
        logger.info("Loaded lead scorer model")
    else:
        logger.warning("Lead scorer model not found")
    
    # Load Investor Cluster model
    cluster_model_path = INVESTOR_CLUSTER_DIR / "kmeans_model.pkl"
    if cluster_model_path.exists():
        with open(cluster_model_path, "rb") as f:
            models["investor_cluster"] = pickle.load(f)
        logger.info("Loaded investor cluster model")
    else:
        logger.warning("Investor cluster model not found")
    
    # Load cluster mapping
    mapping_path = INVESTOR_CLUSTER_DIR / "cluster_mapping.pkl"
    if mapping_path.exists():
        with open(mapping_path, "rb") as f:
            models["cluster_mapping"] = pickle.load(f)
        logger.info("Loaded cluster mapping")
    else:
        # Default mapping
        models["cluster_mapping"] = {0: "Conservative", 1: "Balanced", 2: "Aggressive"}
        logger.warning("Using default cluster mapping")
    
    # Load lead preprocessing artifacts
    preprocess_path = LEAD_SCORER_DIR / "preprocessing_artifacts.pkl"
    if preprocess_path.exists():
        with open(preprocess_path, "rb") as f:
            models["lead_preprocessing"] = pickle.load(f)
        logger.info("Loaded lead preprocessing artifacts")
    
    # Load NAV Forecaster
    nav_model_path = NAV_FORECASTER_DIR / "lstm_model.pt"
    if nav_model_path.exists() and TORCH_AVAILABLE:
        try:
            checkpoint = torch.load(nav_model_path, map_location="cpu")
            from pipelines.train_lstm import NAVForecaster
            nav_model = NAVForecaster(**checkpoint["model_config"])
            nav_model.load_state_dict(checkpoint["model_state_dict"])
            nav_model.eval()
            models["nav_forecaster"] = nav_model
            logger.info("Loaded NAV forecaster model")
        except Exception as e:
            logger.error(f"Failed to load NAV model: {e}")
    else:
        if not TORCH_AVAILABLE:
            logger.warning("Torch not available, skipping NAV forecaster")
        else:
            logger.warning("NAV forecaster model not found")
    
    return models


# Marshmallow Schemas (or dummy classes if marshmallow not available)
if MARSHMALLOW_AVAILABLE:
    class LeadScoringSchema(Schema):
        """Schema for lead scoring request."""
        leads = fields.List(fields.Dict(), required=True, validate=validate.Length(min=1))

    class InvestorSegmentationSchema(Schema):
        """Schema for investor segmentation request."""
        investors = fields.List(fields.Dict(), required=True, validate=validate.Length(min=1))

    class NAVForecastSchema(Schema):
        """Schema for NAV forecast request."""
        scheme_code = fields.String(required=True)
        days = fields.Integer(load_default=30, validate=validate.Range(min=1, max=90))
else:
    class LeadScoringSchema:
        def load(self, data): return data
    class InvestorSegmentationSchema:
        def load(self, data): return data
    class NAVForecastSchema:
        pass


# Routes
@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        JSON with model load status.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "lead_scorer": models["lead_scorer"] is not None,
            "investor_cluster": models["investor_cluster"] is not None,
            "nav_forecaster": models["nav_forecaster"] is not None,
        },
    }
    return jsonify(status)


@app.route("/api/leads/score", methods=["POST"])
def score_leads() -> Dict[str, Any]:
    """Score leads using Random Forest model.
    
    Returns:
        JSON array with lead scores and tiers.
    """
    if models["lead_scorer"] is None:
        return jsonify({"error": "Lead scorer model not loaded"}), 503
    
    schema = LeadScoringSchema()
    try:
        data = schema.load(request.get_json())
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    
    leads = data["leads"]
    results = []
    
    for lead in leads:
        try:
            # Extract features (simplified - in production, match training features)
            features = np.array([
                float(lead.get("TotalVisits", 0)),
                float(lead.get("Total Time Spent on Website", 0)),
                float(lead.get("Page Views Per Visit", 0)),
            ]).reshape(1, -1)
            
            # Predict
            proba = models["lead_scorer"].predict_proba(features)[0, 1]
            
            # Determine tier
            if proba > HOT_LEAD_THRESHOLD:
                tier = "hot"
            elif proba > WARM_LEAD_THRESHOLD:
                tier = "warm"
            else:
                tier = "cold"
            
            results.append({
                "lead_id": lead.get("Prospect ID", "unknown"),
                "conversion_probability": float(proba),
                "tier": tier,
            })
        except Exception as e:
            logger.error(f"Error scoring lead: {e}")
            results.append({
                "lead_id": lead.get("Prospect ID", "unknown"),
                "error": str(e),
            })
    
    return jsonify({"results": results})


@app.route("/api/investors/segment", methods=["POST"])
def segment_investors() -> Dict[str, Any]:
    """Segment investors using K-Means clustering.
    
    Returns:
        JSON array with cluster assignments and personas.
    """
    if models["investor_cluster"] is None:
        return jsonify({"error": "Investor cluster model not loaded"}), 503
    
    schema = InvestorSegmentationSchema()
    try:
        data = schema.load(request.get_json())
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    
    investors = data["investors"]
    results = []
    
    # Fund type recommendations based on persona
    fund_recommendations = {
        "Conservative": "Debt",
        "Balanced": "Hybrid",
        "Aggressive": "Equity",
    }
    
    for investor in investors:
        try:
            # Extract numeric features
            features = np.array([
                float(investor.get("current_age", 0)),
                float(investor.get("yearly_income", 0)),
                float(investor.get("total_debt", 0)),
                float(investor.get("debt_to_income_ratio", 0)),
                float(investor.get("credit_score", 0)),
            ]).reshape(1, -1)
            
            # Predict cluster
            cluster_id = int(models["investor_cluster"].predict(features)[0])
            persona = models["cluster_mapping"].get(cluster_id, "Unknown")
            
            results.append({
                "investor_id": investor.get("id", "unknown"),
                "cluster_id": cluster_id,
                "persona": persona,
                "recommended_fund_type": fund_recommendations.get(persona, "Balanced"),
            })
        except Exception as e:
            logger.error(f"Error segmenting investor: {e}")
            results.append({
                "investor_id": investor.get("id", "unknown"),
                "error": str(e),
            })
    
    return jsonify({"results": results})


@app.route("/api/nav/forecast", methods=["GET"])
@cache.cached(timeout=CACHE_TTL, query_string=True)
def forecast_nav() -> Dict[str, Any]:
    """Forecast NAV for a mutual fund scheme.
    
    Returns:
        JSON with historical and forecasted NAV values.
    """
    if models["nav_forecaster"] is None:
        return jsonify({"error": "NAV forecaster model not loaded"}), 503
    
    scheme_code = request.args.get("scheme_code")
    days = request.args.get("days", 30, type=int)
    
    if not scheme_code:
        return jsonify({"error": "scheme_code is required"}), 400
    
    try:
        # Generate synthetic forecast for demonstration
        # In production, load actual historical data
        np.random.seed(42)
        historical = [
            {"date": f"2024-01-{i:02d}", "nav": 100 + np.random.randn() * 5}
            for i in range(1, 31)
        ]
        
        # Generate forecast
        last_nav = historical[-1]["nav"]
        forecast = []
        for i in range(days):
            last_nav += np.random.randn() * 0.5
            forecast.append({
                "date": f"2024-02-{i+1:02d}",
                "predicted_nav": round(last_nav, 4),
            })
        
        return jsonify({
            "scheme_code": scheme_code,
            "scheme_name": f"Scheme {scheme_code}",
            "historical": historical,
            "forecast": forecast,
        })
    except Exception as e:
        logger.error(f"Error forecasting NAV: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment/current", methods=["GET"])
@cache.cached(timeout=300)
def get_sentiment() -> Dict[str, Any]:
    """Get current market sentiment.
    
    Returns:
        JSON with sentiment signal, score, and top headlines.
    """
    try:
        # Load recent news
        headlines = []
        if NEWS_STREAM_PATH.exists():
            df = pd.read_csv(NEWS_STREAM_PATH)
            recent = df.tail(10)
            for _, row in recent.iterrows():
                headlines.append({
                    "title": row.get("Title", "")[:100],
                    "source": row.get("Source", ""),
                    "timestamp": row.get("Ingestion_Timestamp", ""),
                })
        
        # Simple sentiment calculation (in production, use NLP model)
        # For now, random sentiment for demonstration
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        score = np.random.uniform(-1, 1)
        
        if score > 0.3:
            signal = "Bullish"
        elif score < -0.3:
            signal = "Bearish"
        else:
            signal = "Neutral"
        
        return jsonify({
            "signal": signal,
            "score": round(float(score), 4),
            "top_headlines": headlines,
            "last_updated": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error) -> Dict[str, Any]:
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error) -> Dict[str, Any]:
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def create_app() -> Flask:
    """Application factory for testing.
    
    Returns:
        Flask app instance.
    """
    load_models()
    return app


if __name__ == "__main__":
    load_models()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
