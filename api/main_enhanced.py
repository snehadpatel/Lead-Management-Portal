"""
Enhanced FastAPI Service — Production-Grade API
Endpoints: /predict, /analytics, /insights, /batch_predict, /health
Models load once at startup via ModelRegistry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, ConfigDict, Field, validator
import pandas as pd
import numpy as np

from lume_platform.config import DATA_ROOT, EXPORT_DIR, MODELS_DIR, PROJECT_ROOT, TABLEAU_PUBLIC_EMBED_URL
from lume_platform.inference.registry import ModelRegistry

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application Setup
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Lume AI Platform API",
    description="Production-grade Big Data + AI inference API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry
registry = ModelRegistry()

@app.on_event("startup")
def startup() -> None:
    """Load all models on startup"""
    registry.load()
    print("✅ Model registry loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════════

class LeadFeatures(BaseModel):
    """Lead scoring feature schema"""
    model_config = ConfigDict(populate_by_name=True)
    
    totalvisits: int = Field(0, alias="TotalVisits")
    total_time_spent_on_website: float = Field(0, alias="Total Time Spent on Website")
    page_views_per_visit: float = Field(0, alias="Page Views Per Visit")
    asymmetrique_activity_score: float = Field(0, alias="Asymmetrique Activity Score")
    asymmetrique_profile_score: float = Field(0, alias="Asymmetrique Profile Score")
    lead_origin: str = Field("Unknown", alias="Lead Origin")
    lead_source: str = Field("Unknown", alias="Lead Source")
    specialization: str = Field("Unknown", alias="Specialization")
    occupation: str = Field("Unknown", alias="What is your current occupation")
    last_activity: str = Field("Unknown", alias="Last Activity")
    country: str = Field("Unknown", alias="Country")
    lead_quality: str = Field("Unknown", alias="Lead Quality")
    do_not_email: str = Field("Unknown", alias="Do Not Email")
    do_not_call: str = Field("Unknown", alias="Do Not Call")


class InvestorBehavior(BaseModel):
    """Investor behavior features for clustering"""
    ProfManage: float = Field(5.0, ge=0, le=10)
    Diversification: float = Field(5.0, ge=0, le=10)
    Affordability: float = Field(5.0, ge=0, le=10)
    Liquidity: float = Field(5.0, ge=0, le=10)
    Growth: float = Field(5.0, ge=0, le=10)
    Trustworthiness: float = Field(5.0, ge=0, le=10)
    Technology: float = Field(5.0, ge=0, le=10)


class PredictRequest(BaseModel):
    """Single prediction request"""
    task: str = Field(..., description="lead_scoring | investor_cluster | sentiment")
    lead: Optional[LeadFeatures] = None
    investor_behavior: Optional[InvestorBehavior] = None
    text: Optional[str] = None
    
    @validator('task')
    def validate_task(cls, v):
        allowed = ['lead_scoring', 'investor_cluster', 'sentiment']
        if v not in allowed:
            raise ValueError(f'task must be one of {allowed}')
        return v


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    task: str = Field(..., description="lead_scoring | investor_cluster | sentiment")
    leads: Optional[List[LeadFeatures]] = None
    investor_behaviors: Optional[List[InvestorBehavior]] = None
    texts: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    """Single prediction response"""
    task: str
    timestamp: str
    prediction: Any
    confidence: Optional[float] = None
    model_version: str = "2.0.0"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    task: str
    timestamp: str
    predictions: List[Dict[str, Any]]
    model_version: str = "2.0.0"


class HealthResponse(BaseModel):
    """Health check response with pipeline manifest"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str
    manifest: Optional[Dict[str, Any]] = None


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    timestamp: str
    data_summary: Dict[str, Any]
    model_metrics: Dict[str, Any]


class InsightsResponse(BaseModel):
    """Insights response"""
    timestamp: str
    models: Dict[str, Any]
    bi_tools: Dict[str, str]
    data_sources: List[str]


# Risk Analysis Models
class PortfolioHolding(BaseModel):
    """Portfolio holding for risk analysis"""
    fund_id: str
    fund_name: str
    category: str
    value: float
    returns: float


class RiskAnalysisRequest(BaseModel):
    """Risk analysis request"""
    persona: str = Field(..., description="growth | conservative | balanced | passive")
    holdings: List[PortfolioHolding]


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response with live market data"""
    timestamp: str
    market_status: str
    nifty_50: float
    nifty_change_pct: float
    vix: float
    portfolio_risk_score: float
    risk_level: str
    var_95: float
    max_drawdown_pct: float
    sharpe_ratio: float
    beta: float
    alerts: List[str]
    recommendations: List[str]
    sector_performance: Dict[str, float]
    market_sentiment: str


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
def root():
    """API root with basic info"""
    return {
        "name": "Lume AI Platform API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint with manifest info"""
    manifest_path = MODELS_DIR / "model_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
            
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded={
            "lead_bundle": registry.lead_bundle is not None,
            "investor_bundle": registry.investor_bundle is not None,
            "sentiment_bundle": registry.sentiment_bundle is not None,
        },
        api_version="2.0.0",
        manifest=manifest
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictRequest):
    """
    Single prediction endpoint.
    
    - **lead_scoring**: Predict lead conversion probability
    - **investor_cluster**: Assign investor to persona cluster
    - **sentiment**: Classify text sentiment
    """
    try:
        if request.task == "lead_scoring":
            if not registry.lead_bundle or not request.lead:
                raise HTTPException(status_code=503, detail="Lead model not loaded or missing payload")
            
            # Convert to dict with original column names
            lead_dict = request.lead.model_dump(by_alias=True)
            pred, proba = registry.lead_bundle.predict_row(lead_dict)
            
            return PredictionResponse(
                task=request.task,
                timestamp=datetime.utcnow().isoformat(),
                prediction={"converted": bool(pred), "conversion_probability": round(proba, 4)},
                confidence=round(proba, 4),
                model_version="2.0.0"
            )
        
        elif request.task == "investor_cluster":
            if not registry.investor_bundle or not request.investor_behavior:
                raise HTTPException(status_code=503, detail="Cluster model not loaded or missing payload")
            
            behavior_dict = request.investor_behavior.model_dump()
            cluster_id = registry.investor_bundle.predict_row(behavior_dict)
            
            # Map cluster to persona
            personas = {
                0: "Equity Schemes (High Risk)",
                1: "Liquid/Debt Funds (Safe)",
                2: "Hybrid Allocation Funds",
                3: "Index Trackers (Passive)"
            }
            
            return PredictionResponse(
                task=request.task,
                timestamp=datetime.utcnow().isoformat(),
                prediction={
                    "cluster_id": int(cluster_id),
                    "persona": personas.get(int(cluster_id), "Unknown")
                },
                confidence=None,
                model_version="2.0.0"
            )
        
        elif request.task == "sentiment":
            if not registry.sentiment_bundle or not request.text:
                raise HTTPException(status_code=503, detail="NLP model not loaded or missing text")
            
            label, conf = registry.sentiment_bundle.predict_text(request.text)
            
            return PredictionResponse(
                task=request.task,
                timestamp=datetime.utcnow().isoformat(),
                prediction={"sentiment": label},
                confidence=round(conf, 4),
                model_version="2.0.0"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unknown task")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
def batch_predict(request: BatchPredictRequest):
    """
    Batch prediction endpoint for multiple records.
    
    - **lead_scoring**: Batch lead conversion predictions
    - **investor_cluster**: Batch cluster assignments
    - **sentiment**: Batch sentiment classification
    """
    try:
        predictions = []
        
        if request.task == "lead_scoring":
            if not registry.lead_bundle or not request.leads:
                raise HTTPException(status_code=503, detail="Lead model not loaded or missing payload")
            
            for lead in request.leads:
                lead_dict = lead.model_dump(by_alias=True)
                pred, proba = registry.lead_bundle.predict_row(lead_dict)
                predictions.append({
                    "converted": bool(pred),
                    "conversion_probability": round(proba, 4)
                })
        
        elif request.task == "investor_cluster":
            if not registry.investor_bundle or not request.investor_behaviors:
                raise HTTPException(status_code=503, detail="Cluster model not loaded or missing payload")
            
            personas = {
                0: "Equity Schemes (High Risk)",
                1: "Liquid/Debt Funds (Safe)",
                2: "Hybrid Allocation Funds",
                3: "Index Trackers (Passive)"
            }
            
            for behavior in request.investor_behaviors:
                behavior_dict = behavior.model_dump()
                cluster_id = registry.investor_bundle.predict_row(behavior_dict)
                predictions.append({
                    "cluster_id": int(cluster_id),
                    "persona": personas.get(int(cluster_id), "Unknown")
                })
        
        elif request.task == "sentiment":
            if not registry.sentiment_bundle or not request.texts:
                raise HTTPException(status_code=503, detail="NLP model not loaded or missing texts")
            
            for text in request.texts:
                label, conf = registry.sentiment_bundle.predict_text(text)
                predictions.append({
                    "sentiment": label,
                    "confidence": round(conf, 4)
                })
        
        else:
            raise HTTPException(status_code=400, detail="Unknown task")
        
        return BatchPredictionResponse(
            task=request.task,
            timestamp=datetime.utcnow().isoformat(),
            predictions=predictions,
            model_version="2.0.0"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def analytics():
    """
    Aggregated analytics from production exports.
    Returns data summaries and model performance metrics.
    """
    try:
        data_summary = {"data_root": str(DATA_ROOT)}
        model_metrics = {}
        
        # Lead analytics
        leads_path = EXPORT_DIR / "distributor_leads_master.csv"
        if leads_path.is_file():
            df = pd.read_csv(leads_path)
            data_summary["distributor_leads"] = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "mean_conversion_probability": float(df["Conversion_Probability"].mean())
                if "Conversion_Probability" in df.columns else None,
                "tier_hot_count": int((df["Conversion_Probability"] > 0.85).sum())
                if "Conversion_Probability" in df.columns else None,
                "tier_warm_count": int(((df["Conversion_Probability"] > 0.65) & (df["Conversion_Probability"] <= 0.85)).sum())
                if "Conversion_Probability" in df.columns else None,
                "tier_cold_count": int((df["Conversion_Probability"] <= 0.65).sum())
                if "Conversion_Probability" in df.columns else None,
            }
        
        # Investor analytics
        inv_path = EXPORT_DIR / "investor_routing_matches.csv"
        if inv_path.is_file():
            df = pd.read_csv(inv_path)
            data_summary["investor_routing"] = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "persona_counts": df["Persona_Cluster"].value_counts().to_dict()
                if "Persona_Cluster" in df.columns else {},
            }
        
        # Model metrics
        rf = PROJECT_ROOT / "model_evaluations/random_forest/real_metrics.json"
        if rf.is_file():
            with open(rf) as f:
                model_metrics["lead_scoring"] = json.load(f)
        
        nlp = PROJECT_ROOT / "model_evaluations/nlp_sentiment/metrics.json"
        if nlp.is_file():
            with open(nlp) as f:
                model_metrics["sentiment"] = json.load(f)
        
        return AnalyticsResponse(
            timestamp=datetime.utcnow().isoformat(),
            data_summary=data_summary,
            model_metrics=model_metrics
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/risk/analyze", response_model=RiskAnalysisResponse, tags=["Risk Analysis"])
def analyze_portfolio_risk(request: RiskAnalysisRequest):
    """
    Live portfolio risk analysis using real-time market data.
    
    Calculates:
    - Portfolio risk score (0-100)
    - Value at Risk (VaR 95%)
    - Maximum drawdown estimate
    - Sharpe ratio (risk-adjusted returns)
    - Beta (market correlation)
    - Personalized alerts and recommendations
    
    Example:
    ```json
    {
        "persona": "balanced",
        "holdings": [
            {"fund_id": "F001", "fund_name": "Nippon Small Cap", "category": "equity", "value": 50000, "returns": 15.2},
            {"fund_id": "F004", "fund_name": "HDFC Liquid", "category": "liquid", "value": 30000, "returns": 6.8}
        ]
    }
    ```
    """
    try:
        from lume_platform.risk.live_risk_analyzer import get_live_risk_analysis
        
        holdings = [h.dict() for h in request.holdings]
        result = get_live_risk_analysis(holdings, request.persona)
        
        return RiskAnalysisResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}") from e

@app.get("/risk/market-snapshot", tags=["Risk Analysis"])
def get_market_snapshot():
    """
    Get current market snapshot with live data.
    Returns NIFTY 50, VIX, sector performance, and market sentiment.
    """
    try:
        from lume_platform.risk.live_risk_analyzer import risk_analyzer
        
        market_data = risk_analyzer.fetch_live_market_data()
        
        return {
            "timestamp": market_data.timestamp.isoformat(),
            "market_status": market_data.market_status,
            "nifty_50": market_data.nifty_50,
            "nifty_change_pct": market_data.nifty_change_pct,
            "vix": market_data.vix,
            "market_sentiment": market_data.sentiment,
            "volume_spike": market_data.volume_spike,
            "sector_performance": market_data.sector_performance,
            "top_gainers": market_data.top_gainers,
            "top_losers": market_data.top_losers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/insights", response_model=InsightsResponse, tags=["Insights"])
def insights():
    """
    Model cards and system insights.
    Returns model information, BI tool URLs, and data sources.
    """
    try:
        rf = PROJECT_ROOT / "model_evaluations/random_forest/real_metrics.json"
        nlp = PROJECT_ROOT / "model_evaluations/nlp_sentiment/metrics.json"
        
        models_info = {
            "lead_scoring": None,
            "sentiment": None,
            "investor_clustering": {
                "algorithm": "K-Means",
                "clusters": 4,
                "features": ["ProfManage", "Diversification", "Affordability", 
                           "Liquidity", "Growth", "Trustworthiness", "Technology"]
            }
        }
        
        if rf.is_file():
            with open(rf) as f:
                models_info["lead_scoring"] = json.load(f)
        
        if nlp.is_file():
            with open(nlp) as f:
                models_info["sentiment"] = json.load(f)
        
        bi_tools = {
            "tableau_embed_url": os.environ.get("LUME_TABLEAU_EMBED_URL", TABLEAU_PUBLIC_EMBED_URL),
            "export_csv_path": str(EXPORT_DIR),
            "export_parquet_path": str(PROJECT_ROOT / "artifacts/cleaned_parquet")
        }
        
        data_sources = [
            "structured/leads/lead_scoring/Lead Scoring.csv",
            "structured/leads/mf_investor_behavior/MF_Behavior.xlsx",
            "semi_structured/social_sentiment/data.csv",
            "structured/mutual_funds/nav_history/",
            "structured/stock_prices/nifty500_intraday/"
        ]
        
        return InsightsResponse(
            timestamp=datetime.utcnow().isoformat(),
            models=models_info,
            bi_tools=bi_tools,
            data_sources=data_sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/{model_name}/info", tags=["Models"])
def model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        if model_name == "lead_scoring":
            if not registry.lead_bundle:
                raise HTTPException(status_code=503, detail="Lead model not loaded")
            
            return {
                "model_type": "XGBoost/RandomForest",
                "numeric_features": registry.lead_bundle.numeric_features,
                "categorical_features": registry.lead_bundle.cat_features,
                "decision_threshold": registry.lead_bundle.decision_threshold,
                "description": "Predicts lead conversion probability based on engagement metrics"
            }
        
        elif model_name == "investor_cluster":
            if not registry.investor_bundle:
                raise HTTPException(status_code=503, detail="Investor cluster model not loaded")
            
            return {
                "model_type": "K-Means",
                "n_clusters": 4,
                "behavior_features": registry.investor_bundle.behavior_cols,
                "description": "Clusters investors into persona groups based on behavior attributes"
            }
        
        elif model_name == "sentiment":
            if not registry.sentiment_bundle:
                raise HTTPException(status_code=503, detail="Sentiment model not loaded")
            
            return {
                "model_type": "TF-IDF + LogisticRegression",
                "vectorizer": "TF-IDF (max_features=20000, ngram_range=(1,2))",
                "classes": registry.sentiment_bundle.label_classes.tolist(),
                "description": "Classifies text sentiment using bag-of-words approach"
            }
        
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/exports/{file_name}", tags=["Downloads"])
def download_export(file_name: str):
    """Download exported CSV files for BI tools"""
    try:
        file_path = EXPORT_DIR / file_name
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=file_name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/exports", tags=["Downloads"])
def list_exports():
    """List available exported files"""
    try:
        if not EXPORT_DIR.exists():
            return {"files": []}
        
        files = []
        for f in EXPORT_DIR.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        
        return {"files": files, "export_dir": str(EXPORT_DIR)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ═══════════════════════════════════════════════════════════════════════════════
# Background Tasks (for async processing)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/async_predict", tags=["Async"])
async def async_predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Async prediction endpoint — queues prediction in background"""
    # This is a placeholder for async processing
    # In production, this would queue to Redis/Celery
    return {"status": "queued", "task": request.task}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
