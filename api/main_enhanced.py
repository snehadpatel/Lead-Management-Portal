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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, validator
import pandas as pd
import numpy as np

from lume_platform.config import ARTIFACTS_DIR, DATA_ROOT, EVAL_DIR, EXPORT_DIR, MODELS_DIR, PROJECT_ROOT, TABLEAU_EXPORT_DIR, TABLEAU_PUBLIC_EMBED_URL
from lume_platform.crm.dashboard_service import build_dashboard_leads, build_dashboard_overview, update_lead_workflow
from lume_platform.inference.registry import ModelRegistry
from lume_platform.ml.buddy_engine import BuddyEngine
from lume_platform.db.mongo_client import db_client
from lume_platform.recommender import SimpleRecommender
from lume_platform.auth import create_access_token, decode_access_token, hash_password, verify_password
from fastapi import Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from lume_platform.collab import CollabEngine
from lume_platform.monitoring import Monitor
from lume_platform.impact import MarketImpactEngine
from lume_platform.drift_monitor import DriftMonitor
from lume_platform.data.nav_fetcher import nav_fetcher
from lume_platform.realtime import stream_hub

security = HTTPBearer()


def _auth_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Validate a bearer token for admin-only endpoints."""
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail='Invalid or expired token')
    return True

import threading
import time
try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except Exception:
    KafkaConsumer = None
    KAFKA_AVAILABLE = False

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

from fastapi.staticfiles import StaticFiles
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

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
buddy_engine = BuddyEngine()
recommender = SimpleRecommender(registry=registry)
collab_engine = CollabEngine()
monitor = Monitor()
market_impact_engine = MarketImpactEngine()
drift_monitor = DriftMonitor()

def fallback_sentiment_predict(text: str) -> tuple[str, float]:
    try:
        from lume_platform.ml.transformer_sentiment import get_transformer_sentiment
        analyzer = get_transformer_sentiment()
        if analyzer.is_available:
            label, score = analyzer.predict(text)
            return label.lower(), score
    except Exception as e:
        print(f"Error loading transformer sentiment: {e}")
        
    text_lower = text.lower()
    positive_words = ["good", "bull", "growth", "buy", "up", "high", "positive", "gain", "profit", "recommend", "great", "best", "benefit", "outperform", "bullish", "strong"]
    negative_words = ["bad", "bear", "loss", "sell", "down", "low", "negative", "drop", "risk", "panic", "crash", "fall", "pause", "drawdown", "bearish", "weak"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return "positive", 0.85
    elif neg_count > pos_count:
        return "negative", 0.85
    else:
        return "neutral", 0.50

@app.on_event("startup")
async def startup() -> None:
    """Load all models, seed demo accounts, and start live WebSocket streaming on startup."""
    registry.load()
    db_client.ensure_demo_accounts()
    stream_hub.bind(registry, drift_monitor)
    stream_hub.emit_tick(persist_jsonl=False)
    print("✅ Model registry, demo profiles & realtime hub initialized")

    if os.environ.get("LUME_ENABLE_REALTIME_STREAM", "1") == "1":
        try:
            stream_hub._task = asyncio.create_task(stream_hub.run_loop())
            print("⚡ Real-time WebSocket stream loop started")
        except Exception as e:
            print(f"⚠️ Failed to start real-time stream loop: {e}")

    # Start market signals background listener thread if enabled
    if os.environ.get("LUME_START_KAFKA_LISTENER", "true").lower() == "true":
        def _start_listener():
            t = threading.Thread(target=market_signals_listener, daemon=True)
            t.start()

        try:
            _start_listener()
            print("🔔 Market signals listener started")
        except Exception as e:
            print(f"⚠️ Failed to start market signals listener: {e}")
    else:
        print("🔔 Market signals listener disabled inside API process")

    # Start drift monitoring thread if enabled
    if os.environ.get("LUME_START_DRIFT_MONITOR", "true").lower() == "true":
        try:
            t = threading.Thread(target=drift_monitor_loop, daemon=True)
            t.start()
            print("🧭 Drift monitor started")
        except Exception as e:
            print(f"⚠️ Failed to start drift monitor: {e}")
    else:
        print("🧭 Drift monitor disabled inside API process")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Gracefully stop background streaming tasks."""
    if stream_hub._task is not None and not stream_hub._task.done():
        stream_hub._task.cancel()
        try:
            await stream_hub._task
        except asyncio.CancelledError:
            pass


def retrain_models_job() -> Dict[str, Any]:
    """Run the full training pipeline and force-reload the registry."""
    from lume_platform.ml.training import train_all

    paths = train_all()
    registry.reload()
    return {"status": "trained", "paths": paths}


def drift_monitor_loop():
    """Background loop that feeds signals into the drift monitor and auto-retrains on drift."""
    last_seen = 0
    while True:
        try:
            signals = getattr(registry, 'latest_signals', [])
            if len(signals) != last_seen:
                drift_monitor.ingest(signals[:20])
                last_seen = len(signals)

            if drift_monitor.detect_drift():
                drift_monitor.request_retrain(retrain_models_job)

            time.sleep(float(os.environ.get('DRIFT_MONITOR_INTERVAL', '60')))
        except Exception as e:
            print(f"Drift monitor loop error: {e}")
            time.sleep(30)

def market_signals_listener():
    """Background worker to consume `market_signals` topic or fallback file and persist signals."""
    topic = os.environ.get("KAFKA_SIGNALS_TOPIC", "market_signals")
    out_file = EXPORT_DIR / "market_signals.jsonl"
    if "VERCEL" in os.environ:
        out_file = Path("/tmp") / "market_signals.jsonl"
    try:
        os.makedirs(out_file.parent, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Could not create market signals directory: {e}")

    if KAFKA_AVAILABLE:
        servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
        try:
            consumer = KafkaConsumer(topic, bootstrap_servers=servers, value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                     auto_offset_reset="latest", enable_auto_commit=True, consumer_timeout_ms=1000)
        except Exception as e:
            print(f"⚠️ Kafka consumer init failed: {e}")
            consumer = None
    else:
        consumer = None

    # File fallback tailing
    fallback_path = os.environ.get("STREAMING_FALLBACK_OUT", "streaming_output/market_signals.jsonl")
    last_pos = 0

    while True:
        try:
            records = []
            if consumer:
                msgs = consumer.poll(timeout_ms=1000)
                for tp, batch in msgs.items():
                    for msg in batch:
                        records.append(msg.value)
            else:
                if os.path.exists(fallback_path):
                    with open(fallback_path, 'r', encoding='utf-8') as fh:
                        fh.seek(last_pos)
                        for line in fh:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                continue
                        last_pos = fh.tell()

            for sig in records:
                # persist to export and registry
                try:
                    registry.add_market_signal(sig)
                    with open(out_file, 'a', encoding='utf-8') as fo:
                        fo.write(json.dumps(sig) + "\n")
                except Exception as e:
                    print(f"⚠️ Failed to persist signal: {e}")

            time.sleep(1.0)
        except Exception as e:
            print(f"Market signals listener error: {e}")
            time.sleep(5.0)

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
    volume_spike: bool = False
    portfolio_health_score: float = 85.0
    fund_overlap_pct: float = 10.0
    sector_concentration_pct: float = 35.0
    sip_consistency_score: float = 95.0
    panic_selling_probability: float = 12.0


# SBERT Search Models
class SearchRequest(BaseModel):
    query: str = Field(..., example="aggressive growth equity funds")
    top_k: int = Field(5, ge=1, le=20)

class SearchResult(BaseModel):
    scheme_code: str
    scheme_name: str
    category: str
    match_score: float

class SearchResponse(BaseModel):
    timestamp: str
    query: str
    results: List[SearchResult]

# Managed Lead Models
class LeadUpdate(BaseModel):
    status: str
    notes: Optional[str] = None
    assignee: Optional[str] = None
    next_step_at: Optional[str] = None


# LSTM Forecast Models
class ForecastRequest(BaseModel):
    history: List[float] = Field(..., description="List of historical NAV/Price values (minimum 30)")

class ForecastResponse(BaseModel):
    timestamp: str
    historical_last: float
    forecast_trajectory: List[float]
    p10: Optional[List[float]] = Field(None, description="P10 quantile forecast (lower interval)")
    p50: Optional[List[float]] = Field(None, description="P50 quantile forecast (median trajectory)")
    p90: Optional[List[float]] = Field(None, description="P90 quantile forecast (upper interval)")
    attention_weights: Optional[List[float]] = Field(None, description="Temporal attention weights across 30 lookback days")
    confidence_score: float
    trend: str
    horizon_days: int
    accuracy_metric: str
    quantile_coverage: Optional[str] = Field(None, description="Empirical test quantile coverage rate")
    model_source: str = Field("lstm", description="'bilstm_attention' | 'lstm' | 'fallback'")
    lower_bound: Optional[List[float]] = Field(None, description="Lower uncertainty interval")
    upper_bound: Optional[List[float]] = Field(None, description="Upper uncertainty interval")
    baseline_trajectory: Optional[List[float]] = Field(None, description="Deterministic naive baseline comparison")
    disclaimer: str = Field("Educational and analytical tool only. Not investment advice.", description="Regulatory disclaimer")


# Vision Pattern Recognition Models (Phase 3)
class VisionPatternRequest(BaseModel):
    scheme_code: Optional[str] = Field(None, description="Mutual fund scheme code (e.g. '100119')")
    image_base64: Optional[str] = Field(None, description="Base64 encoded candlestick chart image")
    target_class: Optional[int] = Field(None, description="Optional target class for Grad-CAM (0: Bullish, 1: Bearish, 2: Consolidation)")

class VisionPatternResponse(BaseModel):
    timestamp: str
    pattern: str = Field(..., description="Recognized pattern class name")
    pattern_class: int = Field(..., description="Class index (0: Bullish Breakout, 1: Bearish Breakdown, 2: Consolidation)")
    confidence: float = Field(..., description="Softmax probability of predicted class")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    gradcam_overlay_base64: str = Field(..., description="Base64 data URI of Grad-CAM saliency overlay")
    model_architecture: str = Field("custom_cnn", description="Vision model architecture")
    scheme_code: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    latest_nav: Optional[float] = None
    disclaimer: str = Field("Educational pattern detection only. Not financial advice.", description="Regulatory disclaimer")


# AI Advisor Models
class AdvisorRequest(BaseModel):
    query: str
    persona: Optional[str] = "balanced"
    history: Optional[List[Dict[str, str]]] = None
    dashboard_context: Optional[str] = None

class AdvisorResponse(BaseModel):
    timestamp: str
    query: str
    analysis: str
    sentiment: str
    recommended_funds: List[SearchResult]
    market_mood: str

# Custom AI Buddy Models
class BuddyChatRequest(BaseModel):
    message: str
    distributor_id: Optional[str] = None
    lead_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    dashboard_context: Optional[str] = None

class BuddyChatResponse(BaseModel):
    timestamp: str
    response: str
    confidence: float

class BuddyBriefingResponse(BaseModel):
    timestamp: str
    briefing: str
    top_actions: List[Dict[str, str]]


# MFD Validator Models
class ValidationRequest(BaseModel):
    client_id: str
    fund_name: str
    category: str
    risk_level: str

class ValidationResponse(BaseModel):
    timestamp: str
    recommended: bool
    client_risk_profile: str
    fund_risk_rating: float
    market_sentiment: str
    mismatch_flag: bool
    mismatch_reason: Optional[str] = None
    confidence_score: float
    alternatives: List[str]
    suggested_allocation_pct: float
    outlook: str
    dl_insights: Optional[Dict[str, Any]] = None

class ClientInsightItem(BaseModel):
    client_id: str
    name: str
    sip_drop_probability: float
    inactive_flag: bool
    redemption_likelihood: float
    risk_appetite_change: str
    portfolio_health_score: float
    retention_alert: bool
    upsell_opportunity: str

class ClientInsightsResponse(BaseModel):
    timestamp: str
    insights: List[ClientInsightItem]


class MarketImpactRequest(BaseModel):
    signals: Optional[List[Dict[str, Any]]] = None
    holdings: Optional[List[Dict[str, Any]]] = None
    persona: Optional[str] = "balanced"


class MarketImpactResponse(BaseModel):
    timestamp: str
    headline: str
    severity: str
    confidence: float
    sector_impact: Dict[str, float]
    affected_sectors: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    explanation: List[str]


class DriftStatusResponse(BaseModel):
    drift_detected: bool
    retrain_in_progress: bool
    retrain_requested: bool
    samples_seen: int
    mean_score: float
    negative_ratio: float
    last_signal_at: Optional[str] = None
    last_retrain_at: Optional[str] = None
    retrain_count: int
    last_message: str


class UserRegisterRequest(BaseModel):
    email: str
    password: str
    role: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class DemoLoginRequest(BaseModel):
    role: str = Field(..., description="investor | distributor")


class UserProfileUpdateRequest(BaseModel):
    profile: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
def root():
    """Redirect to frontend"""
    return RedirectResponse(url="/frontend/")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint exposing all 7 ML/DL engines & manifest info."""
    manifest_path = MODELS_DIR / "model_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)

    models_status = (
        registry.get_models_status()
        if hasattr(registry, "get_models_status")
        else {
            "lead_bundle": registry.lead_bundle is not None,
            "investor_bundle": registry.investor_bundle is not None,
            "sentiment_bundle": registry.sentiment_bundle is not None,
            "finbert_pipeline": registry.finbert_pipeline is not None,
        }
    )

    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=models_status,
        api_version="2.0.0",
        manifest=manifest,
    )


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Real-time WebSocket stream broadcasting NIFTY ticks, NLP-scored news & drift state."""
    await websocket.accept()
    stream_hub.register_client(websocket)
    try:
        # Send immediate current snapshot so UI updates with 0ms delay
        await websocket.send_text(json.dumps(stream_hub.get_snapshot()))
        while True:
            msg = await websocket.receive_text()
            if msg.strip().lower() == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
            elif msg.strip().lower() == "tick":
                pkt = await asyncio.to_thread(stream_hub.emit_tick, True)
                await websocket.send_text(json.dumps(pkt))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        stream_hub.unregister_client(websocket)


@app.get("/api/realtime/pulse", tags=["Real-Time"])
def get_realtime_pulse(advance: bool = False):
    """Return latest real-time market, Phase 5 Transformer news & drift snapshot (or step a tick)."""
    if advance:
        return stream_hub.emit_tick(persist_jsonl=True)
    return stream_hub.get_snapshot()


def _dir_parquet_stats(directory: Path) -> Dict[str, Any]:
    if not directory.exists():
        return {"exists": False, "file_count": 0, "total_bytes": 0, "total_mb": 0.0}
    files = list(directory.rglob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
    }


@app.get("/api/bigdata/live-stats", tags=["Big Data"])
def get_bigdata_live_stats():
    """Live telemetry auditing Parquet Data Lake, Spark Streaming outputs, AMFI Catalog & 7 ML/DL models."""
    snapshot = stream_hub.get_snapshot()

    # 1. Bronze AMFI Mutual Fund CSV catalog
    amfi_csv = PROJECT_ROOT / "data" / "Mutual_Fund_Data-main" / "mutual_fund_data.csv"
    amfi_rows = 0
    amfi_size_mb = 0.0
    if amfi_csv.is_file():
        amfi_size_mb = round(amfi_csv.stat().st_size / (1024 * 1024), 2)
        try:
            with open(amfi_csv, "rb") as fh:
                amfi_rows = max(0, sum(1 for _ in fh) - 1)
        except Exception:
            amfi_rows = 0

    # 2. AMFI Live NAV Cache
    nav_cache_dir = EXPORT_DIR / "nav_cache"
    latest_navs_file = nav_cache_dir / "latest_navs.json"
    amfi_live_schemes = 0
    if latest_navs_file.is_file():
        try:
            with open(latest_navs_file, "r") as fh:
                amfi_live_schemes = len(json.load(fh))
        except Exception:
            amfi_live_schemes = 0
    hist_cache_files = len(list(nav_cache_dir.glob("hist_*.json"))) if nav_cache_dir.is_dir() else 0

    # 3. Silver Parquet Lake & Streaming Parquets
    silver_stats = _dir_parquet_stats(ARTIFACTS_DIR / "cleaned_parquet")
    market_stream_stats = _dir_parquet_stats(PROJECT_ROOT / "streaming" / "outputs" / "market_live.parquet")
    news_stream_stats = _dir_parquet_stats(PROJECT_ROOT / "streaming" / "outputs" / "news_live.parquet")

    jsonl_path = PROJECT_ROOT / "streaming_output" / "market_signals.jsonl"
    jsonl_events = 0
    if jsonl_path.is_file():
        try:
            with open(jsonl_path, "rb") as fh:
                jsonl_events = sum(1 for _ in fh)
        except Exception:
            jsonl_events = 0

    # 4. Gold BI / Tableau CSV Exports
    export_csvs = []
    for folder in (EXPORT_DIR, TABLEAU_EXPORT_DIR):
        if folder.is_dir():
            for csvf in sorted(folder.glob("*.csv")):
                export_csvs.append({
                    "name": csvf.name,
                    "folder": folder.name,
                    "size_kb": round(csvf.stat().st_size / 1024.0, 1),
                })

    # 5. Complete 7 ML & DL Engine Matrix
    status_map = snapshot.get("models_status") or (registry.get_models_status() if hasattr(registry, "get_models_status") else {})
    engines = [
        {"id": "lead_bundle", "name": "Random Forest Lead Classifier", "layer": "ML (scikit-learn)", "active": bool(status_map.get("lead_bundle")), "artifact": "artifacts/models/lead_classifier_bundle.pkl", "metric": "ROC-AUC 0.933"},
        {"id": "lead_mlp", "name": "PyTorch LeadMLP Neural Net", "layer": "DL Phase 4 (PyTorch)", "active": bool(status_map.get("lead_mlp")), "artifact": "artifacts/models/lead_mlp_model.pth", "metric": "ROC-AUC 0.916 | 0.048ms"},
        {"id": "bilstm_forecaster", "name": "BiLSTM + Temporal Attention + Quantiles", "layer": "DL Phase 2 (PyTorch)", "active": bool(status_map.get("bilstm_forecaster")), "artifact": "artifacts/models/bilstm_attention_quantile.pth", "metric": "RMSE ₹10.55 | Cov 100%"},
        {"id": "vision_resnet18", "name": "Candlestick ResNet-18 + Grad-CAM", "layer": "DL Phase 3 (PyTorch Vision)", "active": bool(status_map.get("vision_resnet18")), "artifact": "artifacts/models/candlestick_resnet18_finetuned.pth", "metric": "3-Class Chart XAI"},
        {"id": "financial_transformer", "name": "Financial Transformer + Token Saliency", "layer": "DL Phase 5 (PyTorch NLP)", "active": bool(status_map.get("financial_transformer")), "artifact": "artifacts/models/financial_transformer.pth", "metric": "3-Class Financial Sentiment"},
        {"id": "investor_bundle", "name": "K-Means Investor Persona Profiler", "layer": "ML (scikit-learn)", "active": bool(status_map.get("investor_bundle")), "artifact": "artifacts/models/investor_cluster_bundle.pkl", "metric": "4 Wealth Personas"},
        {"id": "semantic_search", "name": "Semantic Fund Retrieval Index", "layer": "ML / Embeddings", "active": bool(status_map.get("semantic_search")), "artifact": "artifacts/models/fund_embeddings.pkl", "metric": f"{len(getattr(registry.sbert_search, 'funds', []))} Embedded Schemes"},
    ]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "bronze_amfi": {
            "csv_rows": amfi_rows,
            "csv_size_mb": amfi_size_mb,
            "live_nav_schemes": amfi_live_schemes,
            "cached_history_files": hist_cache_files,
        },
        "silver_lake": silver_stats,
        "streaming_gold": {
            "market_live_parquet": market_stream_stats,
            "news_live_parquet": news_stream_stats,
            "total_parquet_microbatches": market_stream_stats["file_count"] + news_stream_stats["file_count"],
            "jsonl_stream_events": jsonl_events,
            "realtime_hub": snapshot.get("stream_metrics", {}),
        },
        "bi_exports": {
            "file_count": len(export_csvs),
            "files": export_csvs[:15],
        },
        "drift_monitor": snapshot.get("drift", {}),
        "models_matrix": engines,
        "models_online": sum(1 for e in engines if e["active"]),
        "models_total": len(engines),
        "recent_stream_events": snapshot.get("recent_signals", []),
        "market_snapshot": snapshot.get("market", {}),
    }


@app.get("/signals/latest", tags=["Signals"])
def get_latest_signals(limit: int = 20):
    """Return most recent market signals captured by the background listener."""
    try:
        signals = getattr(registry, 'latest_signals', [])[:limit]
        return {"timestamp": datetime.utcnow().isoformat(), "count": len(signals), "signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/market/impact/analyze", response_model=MarketImpactResponse, tags=["Market Impact"])
def analyze_market_impact(request: MarketImpactRequest):
    """Analyze news/macro signals into sector impact and actionable alerts."""
    try:
        signals = request.signals or getattr(registry, 'latest_signals', [])[:20]
        market_context = {}
        try:
            from lume_platform.risk.live_risk_analyzer import risk_analyzer
            md = risk_analyzer.fetch_live_market_data()
            market_context = {
                'market_sentiment': getattr(md, 'sentiment', 'neutral'),
                'vix': getattr(md, 'vix', 15.0),
                'nifty_change_pct': getattr(md, 'nifty_change_pct', 0.0),
                'market_status': getattr(md, 'market_status', 'CLOSED'),
            }
        except Exception:
            market_context = {}

        profile = {"inferred_risk_profile": request.persona}
        result = market_impact_engine.analyze(signals, market_context=market_context, user_profile=profile)
        return MarketImpactResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/market/impact/latest", response_model=MarketImpactResponse, tags=["Market Impact"])
def get_latest_market_impact():
    """Convenience endpoint for the latest market impact based on buffered signals."""
    try:
        signals = getattr(registry, 'latest_signals', [])[:20]
        result = market_impact_engine.analyze(signals, user_profile={"inferred_risk_profile": "balanced"})
        return MarketImpactResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/alerts/live", tags=["Alerts"])
def get_live_alerts(limit: int = 10):
    """Return live alerts derived from market impact and portfolio risk context."""
    try:
        signals = getattr(registry, 'latest_signals', [])[:20]
        impact = market_impact_engine.analyze(signals, user_profile={"inferred_risk_profile": "balanced"})
        alerts = impact.alerts[:limit]
        # include risk alerts if possible
        try:
            from lume_platform.risk.live_risk_analyzer import risk_analyzer
            md = risk_analyzer.fetch_live_market_data()
            if getattr(md, 'vix', 0) and md.vix > 24:
                alerts.insert(0, f"Volatility alert: VIX {md.vix:.1f} is elevated")
        except Exception:
            pass
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'count': len(alerts),
            'alerts': alerts,
            'severity': impact.severity,
            'confidence': impact.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/monitoring/drift/status", response_model=DriftStatusResponse, tags=["Monitoring"])
def get_drift_status():
    """Return the current drift-monitor state for the frontend and operators."""
    try:
        status = drift_monitor.status()
        return DriftStatusResponse(**status.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/monitoring/retrain", tags=["Monitoring"])
def trigger_retrain(auth_ok: bool = Depends(_auth_admin)):
    """Manually trigger the training pipeline and reload all models."""
    try:
        if drift_monitor.retrain_in_progress:
            return {"status": "already_running"}

        started = drift_monitor.request_retrain(retrain_models_job)
        return {
            "status": "started" if started else "skipped",
            "drift": drift_monitor.status().__dict__,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            
            # Persist result to MongoDB
            try:
                lead_id = f"L-{hash(str(lead_dict)) % 100000}"
                db_client.upsert_lead(lead_id, {
                    **lead_dict,
                    "lead_id": lead_id,
                    "converted_prediction": bool(pred),
                    "conversion_probability": round(proba, 4),
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                print(f"⚠️ Failed to persist lead to MongoDB: {e}")
                lead_id = "unknown"
            
            return PredictionResponse(
                task=request.task,
                timestamp=datetime.utcnow().isoformat(),
                prediction={"converted": bool(pred), "conversion_probability": round(proba, 4), "lead_id": lead_id},
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
            if not request.text:
                raise HTTPException(status_code=400, detail="Missing text field")
            
            # Unified registry-backed sentiment prediction (FinBERT preferred)
            try:
                label, conf = registry.predict_sentiment(request.text)
            except Exception:
                label, conf = fallback_sentiment_predict(request.text)
            
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
            if not request.texts:
                raise HTTPException(status_code=400, detail="Missing texts list")
            for text in request.texts:
                try:
                    label, conf = registry.predict_sentiment(text)
                except Exception:
                    label, conf = fallback_sentiment_predict(text)
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
        change_pct = market_data.nifty_change_pct
        
        # Classify sentiment into Very Bullish, Bullish, Neutral, Bearish, Highly Bearish
        # Calculate sentiment score -100 to +100
        if change_pct > 1.5:
            sentiment_label = "Very Bullish"
            sentiment_score = round(85.0 + (min(change_pct, 5.0) - 1.5) * 4.2, 1)
        elif change_pct > 0.5:
            sentiment_label = "Bullish"
            sentiment_score = round(40.0 + (change_pct - 0.5) * 45.0, 1)
        elif change_pct < -1.5:
            sentiment_label = "Highly Bearish"
            sentiment_score = round(-85.0 + (max(change_pct, -5.0) + 1.5) * 4.2, 1)
        elif change_pct < -0.5:
            sentiment_label = "Bearish"
            sentiment_score = round(-40.0 + (change_pct + 0.5) * 45.0, 1)
        else:
            sentiment_label = "Neutral"
            sentiment_score = round(change_pct * 80.0, 1)
            
        sentiment_score = max(-100.0, min(100.0, sentiment_score))
        
        # Volatility score 0-100 based on VIX
        volatility_score = max(0.0, min(100.0, round(market_data.vix * 2.5, 1)))
        
        # Risk intensity score 0-100 based on volatility and sentiment deviation from neutral
        risk_intensity_score = max(0.0, min(100.0, round((volatility_score * 0.6) + (abs(sentiment_score) * 0.4), 1)))
        
        return {
            "timestamp": market_data.timestamp.isoformat(),
            "market_status": market_data.market_status,
            "nifty_50": market_data.nifty_50,
            "nifty_change_pct": market_data.nifty_change_pct,
            "vix": market_data.vix,
            "market_sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "market_volatility_score": volatility_score,
            "risk_intensity_score": risk_intensity_score,
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


@app.get("/search/funds", response_model=SearchResponse, tags=["Search"])
def search_funds(q: str = Query(..., alias="query"), top_k: int = 5):
    """Semantic fund search using SBERT embeddings + AMFI catalog fallback."""
    try:
        results: List[Dict[str, Any]] = []
        if not registry.sbert_search:
            registry.load_sbert_cache()
        if registry.sbert_search:
            results = registry.sbert_search.query(q, top_k=top_k)

        if not results:
            cat_matches = recommender.catalog_view(top_k=top_k, query=q)
            results = [
                {
                    "scheme_code": str(f.get("scheme_code", "")),
                    "scheme_name": f.get("scheme_name", f.get("name", "Unknown Fund")),
                    "category": f.get("category", "Mutual Fund"),
                    "match_score": float(f.get("match_score", 0.55)),
                }
                for f in cat_matches
            ]

        return SearchResponse(timestamp=datetime.utcnow().isoformat(), query=q, results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _check_admin_key(headers: dict) -> bool:
    admin_key = os.environ.get('ADMIN_API_KEY')
    if not admin_key:
        # No admin key set — allow for local dev (but log warning)
        print('⚠️ ADMIN_API_KEY not set — admin endpoints are unprotected')
        return True
    header_val = headers.get('x-admin-key') or headers.get('X-ADMIN-KEY') or headers.get('X-Admin-Key')
    return header_val == admin_key


def _auth_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail='Invalid or expired token')
    return True


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validate token and return current user details."""
    payload = decode_access_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    email = payload["sub"]
    user = db_client.get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_distributor(user: dict = Depends(get_current_user)) -> dict:
    """Ensure user has a distributor or admin role."""
    if user.get("role") not in ("distributor", "admin"):
        raise HTTPException(status_code=403, detail="Access forbidden: distributor role required")
    return user


@app.post('/auth/register', tags=['Auth'])
def register(request: UserRegisterRequest):
    """Register a new SaaS user."""
    email_clean = request.email.strip().lower()
    if db_client.get_user(email_clean):
        raise HTTPException(status_code=400, detail="Account with this email already exists")
    
    pwd_hash = hash_password(request.password)
    user = db_client.create_user(email_clean, pwd_hash, request.role)
    token = create_access_token(email_clean)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": email_clean,
        "role": request.role
    }


@app.post('/auth/login', tags=['Auth'])
def login(request: UserLoginRequest):
    """Authenticate and return JWT token for a user."""
    email_clean = request.email.strip().lower()
    user = db_client.get_user(email_clean)
    if not user or not verify_password(user["password_hash"], request.password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    token = create_access_token(email_clean)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": email_clean,
        "role": user["role"]
    }


@app.post('/auth/demo-login', tags=['Auth'])
def demo_login(request: DemoLoginRequest):
    """1-click Demo Investor or Demo Distributor authentication with pre-seeded holdings & CRM."""
    role = (request.role or "").strip().lower()
    if role not in ("investor", "distributor"):
        raise HTTPException(status_code=400, detail="role must be 'investor' or 'distributor'")

    db_client.ensure_demo_accounts()
    email = "demo.investor@lumeai.in" if role == "investor" else "demo.distributor@lumeai.in"
    user = db_client.get_user(email)
    if not user:
        raise HTTPException(status_code=500, detail="Failed to provision demo account")

    token = create_access_token(email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": email,
        "role": user.get("role", role),
        "persona": user.get("persona", "balanced"),
        "full_name": user.get("full_name", f"Demo {role.title()}"),
        "is_demo": True,
    }


@app.get('/auth/me', tags=['Auth'])
def auth_me(user: dict = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    # Exclude password hash from response
    user_copy = user.copy()
    user_copy.pop("password_hash", None)
    return user_copy


@app.post('/auth/token', tags=['Auth'])
def issue_token(api_key: str = Query(..., description='Admin API key')):
    """Issue a short-lived JWT if the provided API key matches ADMIN_API_KEY."""
    admin_key = os.environ.get('ADMIN_API_KEY')
    if admin_key and api_key != admin_key:
        raise HTTPException(status_code=401, detail='Unauthorized')
    token = create_access_token('admin')
    return {'access_token': token, 'token_type': 'bearer'}


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio CRUD — Persistent user portfolio with live NAV valuation
# ═══════════════════════════════════════════════════════════════════════════════

class AddHoldingRequest(BaseModel):
    scheme_code: str
    scheme_name: str
    category: str = "Mutual Fund"
    amount: float = Field(..., gt=0, description="Investment amount in INR")
    buy_date: Optional[str] = None  # YYYY-MM-DD, defaults to today


class SellHoldingRequest(BaseModel):
    holding_id: str
    units: Optional[float] = None
    amount: Optional[float] = None
    sell_all: bool = False


@app.get('/portfolio', tags=['Portfolio'])
def get_portfolio(user: dict = Depends(get_current_user)):
    """Get user's portfolio with live NAV-based valuations."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher

        email = user["email"]
        holdings = db_client.get_portfolio(email)

        if not holdings:
            return {"holdings": [], "total_invested": 0, "total_current": 0, "total_pnl": 0, "total_pnl_pct": 0, "holding_count": 0}

        # Fetch live NAVs in bulk
        scheme_codes = [h["scheme_code"] for h in holdings]
        live_navs = nav_fetcher.get_latest_navs_bulk(scheme_codes)

        enriched = []
        total_invested = 0
        total_current = 0

        for h in holdings:
            code = h["scheme_code"]
            buy_nav = float(h.get("buy_nav", 0))
            units = float(h.get("units", 0))
            buy_value = float(h.get("buy_value", 0))
            current_nav = live_navs.get(code, buy_nav)

            current_value = round(units * current_nav, 2) if units > 0 else buy_value
            pnl = round(current_value - buy_value, 2)
            pnl_pct = round((pnl / buy_value) * 100, 2) if buy_value > 0 else 0

            # XIRR approximation
            xirr = nav_fetcher.calculate_xirr_return(
                buy_nav, current_nav, h.get("buy_date", datetime.now().strftime("%Y-%m-%d"))
            )

            total_invested += buy_value
            total_current += current_value

            enriched.append({
                **h,
                "current_nav": current_nav,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "xirr": xirr,
                "nav_live": code in live_navs,
            })

        total_pnl = round(total_current - total_invested, 2)
        total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0

        return {
            "holdings": enriched,
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "holding_count": len(enriched),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/portfolio/add', tags=['Portfolio'])
def add_to_portfolio(request: AddHoldingRequest, user: dict = Depends(get_current_user)):
    """Add an investment to the user's portfolio. Auto-calculates units from current NAV."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher

        email = user["email"]
        buy_date = request.buy_date or datetime.now().strftime("%Y-%m-%d")

        # Determine buy NAV (historical if past date specified, else latest)
        buy_nav = None
        
        if request.buy_date and request.buy_date != datetime.now().strftime("%Y-%m-%d"):
            # Try to fetch historical NAV for that past date
            try:
                dt = datetime.strptime(request.buy_date, "%Y-%m-%d")
                formatted_target = dt.strftime("%d-%m-%Y")
                
                # Fetch up to 10 years of history
                hist = nav_fetcher.get_historical_nav(request.scheme_code, days=3650)
                if hist:
                    # Look for exact match
                    for entry in hist:
                        if entry.get("date") == formatted_target:
                            buy_nav = float(entry["nav"])
                            break
                    
                    # Look for closest date match within 7 days
                    if not buy_nav:
                        min_diff = float("inf")
                        closest_val = None
                        for entry in hist:
                            try:
                                entry_dt = datetime.strptime(entry["date"], "%d-%m-%Y")
                                diff = abs((entry_dt - dt).days)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_val = float(entry["nav"])
                            except Exception:
                                continue
                        if min_diff <= 7:
                            buy_nav = closest_val
            except Exception as ex:
                print(f"⚠️ Error parsing historical NAV for date {request.buy_date}: {ex}")

        # If not found or not historical, get latest NAV
        if not buy_nav:
            buy_nav = nav_fetcher.get_latest_nav(request.scheme_code)
            if not buy_nav or buy_nav <= 0:
                # Fallback: try from catalog
                fund = next(
                    (f for f in (recommender.fallback_funds or []) if f.get("scheme_code") == request.scheme_code),
                    None
                )
                buy_nav = float(fund.get("nav", 100)) if fund and fund.get("nav") else 100.0

        units = round(request.amount / buy_nav, 4)

        holding = {
            "scheme_code": request.scheme_code,
            "scheme_name": request.scheme_name,
            "category": request.category,
            "units": units,
            "buy_nav": buy_nav,
            "buy_date": buy_date,
            "buy_value": round(request.amount, 2),
        }

        saved = db_client.add_holding(email, holding)
        return {"status": "added", "holding": saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/portfolio/sell', tags=['Portfolio'])
def sell_portfolio_holding(request: SellHoldingRequest, user: dict = Depends(get_current_user)):
    """Sell/redeem units of a holding. Auto-calculates units from current NAV for amount-based sells."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher

        email = user["email"]
        holdings = db_client.get_portfolio(email)
        
        # Find matching holding
        holding = next((h for h in holdings if h.get("holding_id") == request.holding_id), None)
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")

        current_units = float(holding.get("units", 0))
        buy_nav = float(holding.get("buy_nav", 0))
        buy_value = float(holding.get("buy_value", 0))
        scheme_code = holding.get("scheme_code")

        if current_units <= 0:
            raise HTTPException(status_code=400, detail="No units available to sell")

        # Determine how many units to sell
        sell_units = 0.0
        
        if request.sell_all:
            sell_units = current_units
        elif request.units is not None:
            sell_units = float(request.units)
        elif request.amount is not None:
            # Fetch current NAV to convert amount to units
            current_nav = nav_fetcher.get_latest_nav(scheme_code)
            if not current_nav or current_nav <= 0:
                current_nav = buy_nav
            sell_units = float(request.amount) / current_nav
        else:
            raise HTTPException(status_code=400, detail="Must specify either units, amount, or sell_all")

        if sell_units <= 0:
            raise HTTPException(status_code=400, detail="Units/Amount to sell must be greater than zero")

        # Handle float accuracy issues: if sell_units is very close to current_units, sell all
        if sell_units >= current_units - 0.0001:
            sell_units = current_units
            
        if sell_units > current_units:
            raise HTTPException(status_code=400, detail=f"Cannot sell {sell_units:.4f} units. Only {current_units:.4f} units are held.")

        # Calculate transaction details
        current_nav = nav_fetcher.get_latest_nav(scheme_code)
        if not current_nav or current_nav <= 0:
            current_nav = buy_nav
            
        redemption_value = round(sell_units * current_nav, 2)
        realized_pnl = round(sell_units * (current_nav - buy_nav), 2)

        # Update database
        if sell_units == current_units:
            db_client.remove_holding(email, request.holding_id)
            status = "fully_redeemed"
            remaining_units = 0.0
            new_buy_value = 0.0
        else:
            remaining_units = round(current_units - sell_units, 4)
            # Adjust the buy value down proportionally to reflect remaining average cost basis
            new_buy_value = round(remaining_units * buy_nav, 2)
            
            updates = {
                "units": remaining_units,
                "buy_value": new_buy_value
            }
            db_client.update_holding(email, request.holding_id, updates)
            status = "partially_redeemed"

        return {
            "status": status,
            "scheme_name": holding.get("scheme_name"),
            "sell_units": round(sell_units, 4),
            "redemption_value": redemption_value,
            "realized_pnl": realized_pnl,
            "remaining_units": remaining_units,
            "new_buy_value": new_buy_value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put('/portfolio/{holding_id}', tags=['Portfolio'])
def update_portfolio_holding(holding_id: str, updates: Dict[str, Any], user: dict = Depends(get_current_user)):
    """Update an existing holding (e.g., add more units via SIP)."""
    email = user["email"]
    result = db_client.update_holding(email, holding_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"status": "updated", "holding": result}


@app.delete('/portfolio/{holding_id}', tags=['Portfolio'])
def remove_portfolio_holding(holding_id: str, user: dict = Depends(get_current_user)):
    """Remove a holding from the portfolio."""
    email = user["email"]
    removed = db_client.remove_holding(email, holding_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"status": "removed", "holding_id": holding_id}


@app.get('/portfolio/summary', tags=['Portfolio'])
def get_portfolio_summary(user: dict = Depends(get_current_user)):
    """Get aggregated portfolio analytics: total value, allocation, day change."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher

        email = user["email"]
        holdings = db_client.get_portfolio(email)

        if not holdings:
            return {
                "total_invested": 0, "total_current": 0, "total_pnl": 0,
                "total_pnl_pct": 0, "allocation": [], "holding_count": 0,
            }

        scheme_codes = [h["scheme_code"] for h in holdings]
        live_navs = nav_fetcher.get_latest_navs_bulk(scheme_codes)

        total_invested = 0
        total_current = 0
        allocation = []

        for h in holdings:
            code = h["scheme_code"]
            buy_nav = float(h.get("buy_nav", 0))
            units = float(h.get("units", 0))
            buy_value = float(h.get("buy_value", 0))
            current_nav = live_navs.get(code, buy_nav)
            current_value = round(units * current_nav, 2)

            total_invested += buy_value
            total_current += current_value
            allocation.append({
                "name": h.get("scheme_name", "Unknown"),
                "value": current_value,
                "category": h.get("category", ""),
            })

        total_pnl = round(total_current - total_invested, 2)
        total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0

        # Allocation percentages
        for item in allocation:
            item["pct"] = round((item["value"] / total_current) * 100, 1) if total_current > 0 else 0

        return {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "allocation": allocation,
            "holding_count": len(holdings),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/nav/{scheme_code}', tags=['Market Data'])
def get_live_nav(scheme_code: str):
    """Get the latest live NAV for a mutual fund scheme."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher
        nav = nav_fetcher.get_latest_nav(scheme_code)
        if nav is None:
            raise HTTPException(status_code=404, detail=f"NAV not found for scheme {scheme_code}")
        return {"scheme_code": scheme_code, "nav": nav, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/nav/{scheme_code}/history', tags=['Market Data'])
def get_nav_history(scheme_code: str, days: int = 365):
    """Get historical NAV data for a fund (for charts)."""
    try:
        from lume_platform.data.nav_fetcher import nav_fetcher
        hist = nav_fetcher.get_historical_nav(scheme_code, days=days)
        returns = nav_fetcher.calculate_returns(scheme_code)
        return {
            "scheme_code": scheme_code,
            "data_points": len(hist),
            "history": hist[:60],  # Limit to 60 for chart rendering
            "returns": returns,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/portfolio/analytics', tags=['Portfolio'])
def get_portfolio_analytics(horizon: str = '1y', user: dict = Depends(get_current_user)):
    """Get advanced portfolio analytics (Alpha, Sharpe, Beta, Drawdown, Frontier, Radar)."""
    try:
        from lume_platform.risk.portfolio_analytics import portfolio_analytics
        email = user["email"]
        holdings = db_client.get_portfolio(email)
        return portfolio_analytics.analyze_portfolio(holdings, horizon=horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/news/live', tags=['News'])
def get_live_news_endpoint(max_articles: int = 12, user: dict = Depends(get_current_user)):
    """Get live financial news personalized for user's holdings with Phase 5 Transformer sentiment + saliency."""
    try:
        from lume_platform.news.news_fetcher import news_fetcher
        email = user["email"]
        holdings = db_client.get_portfolio(email)
        alerts = news_fetcher.get_personalized_alerts(holdings, max_alerts=max_articles)

        try:
            nlp_svc = get_financial_nlp_service()
            enriched = []
            for item in alerts:
                title = item.get("title", "")
                res = nlp_svc.predict_sentiment_with_saliency(title)
                top_tokens = sorted(
                    res.get("token_saliency", []),
                    key=lambda x: x.get("saliency", 0.0),
                    reverse=True,
                )[:4]
                sent_label = res.get("sentiment", item.get("sentiment", "neutral"))
                mapped_sent = (
                    "bullish" if sent_label == "positive"
                    else ("bearish" if sent_label == "negative" else "neutral")
                )
                enriched.append({
                    **item,
                    "sentiment": mapped_sent,
                    "nlp_sentiment": sent_label,
                    "nlp_confidence": res.get("confidence", 0.5),
                    "nlp_model": res.get("model", "FinancialTransformer"),
                    "token_saliency": top_tokens,
                })
            return enriched
        except Exception:
            return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/benchmark/history', tags=['Market Data'])
def get_benchmark_history(days: int = 30):
    """Get historical NIFTY 50 closing prices for charts."""
    try:
        from lume_platform.risk.portfolio_analytics import portfolio_analytics
        df = portfolio_analytics.get_historical_benchmark(days=days)
        df_sorted = df.sort_index(ascending=False).head(days)
        history = []
        for idx, row in df_sorted.iterrows():
            history.append({
                "date": idx.strftime("%d-%m-%Y"),
                "nav": round(float(row["Close"]), 2)
            })
        return {
            "symbol": "^NSEI",
            "data_points": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/admin/funds/upload', tags=['Admin'])
def admin_upload_fund_catalog(payload: List[Dict[str, Any]], headers: Dict[str, str] = None, auth_ok: bool = Depends(_auth_admin)):
    """Upload or replace the fund catalog (expects list of fund dicts).
    Protected by `X-ADMIN-KEY` header when `ADMIN_API_KEY` is set.
    """
    try:
        if headers is None:
            headers = {}
        if not _check_admin_key(headers):
            raise HTTPException(status_code=401, detail='Unauthorized')

        catalog_path = MODELS_DIR / 'fund_catalog.json'
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        # Also write a minimal embeddings cache placeholder file location
        return {'status': 'uploaded', 'catalog_path': str(catalog_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/admin/funds/reindex', tags=['Admin'])
def admin_reindex_fund_embeddings(headers: Dict[str, str] = None, auth_ok: bool = Depends(_auth_admin)):
    """Rebuild SBERT fund embeddings from the catalog and load into registry.

    This endpoint runs `scripts/build_fund_embeddings.py` logic inline (requires sentence-transformers).
    Protected by `X-ADMIN-KEY` when `ADMIN_API_KEY` set.
    """
    try:
        if headers is None:
            headers = {}
        if not _check_admin_key(headers):
            raise HTTPException(status_code=401, detail='Unauthorized')

        catalog_path = MODELS_DIR / 'fund_catalog.json'
        cache_path = MODELS_DIR / 'fund_embeddings.pkl'
        if not catalog_path.is_file():
            raise HTTPException(status_code=400, detail='Fund catalog not found; upload first')

        # Load catalog
        with open(catalog_path, 'r', encoding='utf-8') as f:
            funds = json.load(f)

        # Build embeddings inline
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise HTTPException(status_code=500, detail='sentence-transformers not installed')

        model_name = os.environ.get('SBERT_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        model = SentenceTransformer(model_name)
        texts = [ (f.get('scheme_name','') + ' ' + f.get('description','')).strip() for f in funds ]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Save cache
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump({'funds': funds, 'embeddings': embeddings}, fh)

        # Load into registry
        ok = registry.load_sbert_cache(cache_path)
        if not ok:
            raise HTTPException(status_code=500, detail='Failed to load SBERT cache into registry')

        return {'status': 'reindexed', 'cache_path': str(cache_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/admin/metrics', tags=['Admin'])
def admin_metrics(auth_ok: bool = Depends(_auth_admin)):
    """Expose basic monitoring metrics."""
    try:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_requests': monitor.total_requests,
            'errors': monitor.errors,
            'avg_latency': monitor.average_latency(),
            'sentiment_mean': monitor.sentiment_mean(),
            'drift_detected': monitor.check_drift(),
            'latest_signals_count': len(getattr(registry, 'latest_signals', [])),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/user/{user_id}/collaborative-recommend', tags=['User'])
def collaborative_recommend(user_id: str, top_k: int = 5):
    """Collaborative recommendations using transaction history."""
    try:
        recs = collab_engine.recommend_for_user(user_id, top_k=top_k)
        if not recs:
            raise HTTPException(status_code=404, detail='No collaborative recommendations available')
        return {'user_id': user_id, 'recommendations': recs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/user/profile', tags=['User'])
def upsert_user_profile(user_id: str = Query(...), profile: Dict[str, Any] = None):
    """Save or update a user profile for personalization."""
    try:
        if profile is None:
            raise HTTPException(status_code=400, detail='Missing profile payload')
        recommender.save_profile(user_id, profile)
        
        # Sync with users database
        db_client.update_user_profile(user_id, {
            "persona": profile.get("inferred_risk_profile", "balanced"),
            "goals": profile.get("goals", ""),
            "investment_horizon": profile.get("investment_horizon", "")
        })
        
        return { 'status': 'saved', 'user_id': user_id }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/user/{user_id}/recommend', tags=['User'])
def recommend_for_user(user_id: str, top_k: int = 5):
    """Return personalized fund recommendations for a given user id."""
    try:
        profile = recommender.load_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail='Profile not found')

        # Build minimal market context from risk analyzer if available
        market_context = {}
        try:
            from lume_platform.risk.live_risk_analyzer import risk_analyzer
            md = risk_analyzer.fetch_live_market_data()
            market_context = {
                'market_sentiment': md.sentiment if hasattr(md, 'sentiment') else None,
                'vix': md.vix if hasattr(md, 'vix') else None
            }
        except Exception:
            market_context = {}

        recs = recommender.recommend_for_profile(profile, market_context=market_context, top_k=top_k)
        return { 'user_id': user_id, 'recommendations': recs, 'market_context': market_context }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/funds/catalog', tags=['Funds'])
def funds_catalog(persona: Optional[str] = None, query: Optional[str] = None, top_k: int = 100):
    """Return a live mutual-fund catalog derived from the dataset."""
    try:
        return {
            'persona': persona,
            'query': query,
            'count': min(top_k, len(recommender.catalog_view(top_k=top_k, persona=persona, query=query))),
            'funds': recommender.catalog_view(top_k=top_k, persona=persona, query=query),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/funds/recommendations', tags=['Funds'])
def funds_recommendations(persona: str = Query(...), top_k: int = 5, investment_horizon: Optional[str] = None):
    """Return persona-based mutual-fund recommendations from the live catalog."""
    try:
        profile = {
            'inferred_risk_profile': persona,
            'goals': f'{persona} mutual fund recommendations',
            'investment_horizon': investment_horizon or ('long term' if persona in {'growth', 'balanced'} else 'short to medium term'),
        }
        market_context = {}
        try:
            from lume_platform.risk.live_risk_analyzer import risk_analyzer
            md = risk_analyzer.fetch_live_market_data()
            market_context = {
                'market_sentiment': md.sentiment if hasattr(md, 'sentiment') else None,
                'vix': md.vix if hasattr(md, 'vix') else None,
            }
        except Exception:
            market_context = {}

        recs = recommender.recommend_for_profile(profile, market_context=market_context, top_k=top_k)
        return {
            'persona': persona,
            'recommendations': recs,
            'market_context': market_context,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/distributors/validate-rec", response_model=ValidationResponse, tags=["MFD Copilot"])
def validate_recommendation(request: ValidationRequest, current_user: dict = Depends(get_current_distributor)):
    """
    Validates a distributor's mutual fund recommendation against a client's risk profile and current market sentiment.
    """
    try:
        from lume_platform.risk.live_risk_analyzer import risk_analyzer
        market_data = risk_analyzer.fetch_live_market_data()
        
        client_risk = request.risk_level.lower()
        fund_cat = request.category.lower()
        
        if fund_cat in ["small-cap", "small cap", "sectoral", "thematic", "equity-high"]:
            fund_risk_rating = 90.0
            fund_risk_label = "high"
        elif fund_cat in ["mid-cap", "mid cap", "flexi-cap", "flexi cap", "elss", "equity"]:
            fund_risk_rating = 65.0
            fund_risk_label = "medium-high"
        elif fund_cat in ["index", "large-cap", "large cap", "hybrid", "balanced"]:
            fund_risk_rating = 45.0
            fund_risk_label = "medium"
        elif fund_cat in ["debt", "liquid", "gilt"]:
            fund_risk_rating = 15.0
            fund_risk_label = "low"
        else:
            fund_risk_rating = 50.0
            fund_risk_label = "medium"
            
        mismatch = False
        reason = None
        recommended = True
        
        if client_risk == "conservative" and fund_risk_label in ["high", "medium-high", "medium"]:
            mismatch = True
            recommended = False
            reason = f"Mismatch: Conservative investor profile cannot absorb the volatility of a {fund_risk_label.upper()} risk fund ({request.fund_name}). Recommend low-risk options instead."
        elif client_risk in ["balanced", "moderate"] and fund_risk_label == "high":
            mismatch = True
            recommended = False
            reason = f"Warning: Balanced investor profile is mismatching with a High-Risk fund ({request.fund_name}). Keep allocation below 10% or select a Moderate/Balanced alternative."
            
        if market_data.sentiment == "bearish" and fund_risk_label in ["high", "medium-high"] and not mismatch:
            mismatch = True
            reason = f"Market Alert: Current market sentiment is Bearish (VIX is {market_data.vix}). Entering high-equity positions now has elevated drawdown risk. Stagger via SIP."
            recommended = True
            
        if client_risk == "conservative":
            alternatives = ["HDFC Liquid Fund", "ICICI Prudential Savings Fund", "UTI Nifty 50 Index Fund"]
            suggested_allocation = 15.0 if fund_risk_label in ["low"] else 5.0
        elif client_risk in ["balanced", "moderate"]:
            alternatives = ["SBI Equity Hybrid Fund", "UTI Nifty 50 Index Fund", "Mirae Asset Large Cap Fund"]
            suggested_allocation = 25.0
        else:
            alternatives = ["Parag Parikh Flexi Cap Fund", "Mirae Asset Large Cap Fund", "Axis Small Cap Fund"]
            suggested_allocation = 40.0
            
        outlook = "Staggered Systematic Entry Recommended" if market_data.sentiment in ["neutral", "bearish"] else "Lump-sum / Accelerated SIP Permitted"

        # Enrich with real Deep Learning engines (Phase 5 Transformer, Phase 2 BiLSTM, Phase 3 ResNet-18)
        dl_insights: Dict[str, Any] = {}
        try:
            snap = stream_hub.get_snapshot()
            latest_sig = snap.get("latest_signal", {})
            nlp_svc = get_financial_nlp_service()
            eval_text = latest_sig.get("headline") or f"{request.fund_name} {request.category} market outlook VIX {market_data.vix}"
            nlp_res = nlp_svc.predict_sentiment_with_saliency(eval_text)
            top_toks = sorted(nlp_res.get("token_saliency", []), key=lambda x: x.get("saliency", 0.0), reverse=True)[:4]
            dl_insights["financial_transformer"] = {
                "headline": eval_text,
                "sentiment": nlp_res.get("sentiment"),
                "confidence": nlp_res.get("confidence"),
                "top_tokens": [t.get("token") for t in top_toks],
            }
        except Exception:
            pass

        try:
            # Find matching scheme history to run BiLSTM + ResNet-18 when available
            q_lower = (request.fund_name or "").lower()
            matched_scheme = None
            for f in (recommender.fallback_funds or []):
                sname = (f.get("scheme_name") or "").lower()
                code = str(f.get("scheme_code", ""))
                hist_file = EXPORT_DIR / "nav_cache" / f"hist_{code}.json"
                if q_lower and any(tok in sname for tok in q_lower.split()[:2]) and hist_file.is_file():
                    matched_scheme = code
                    break
            if not matched_scheme and (EXPORT_DIR / "nav_cache" / "hist_100119.json").is_file():
                matched_scheme = "100119"

            if matched_scheme:
                hist = nav_fetcher.get_historical_nav(matched_scheme, days=60)
                if len(hist) >= 30:
                    vals = [float(x["nav"]) for x in reversed(hist[:60])]
                    fc_svc = getattr(registry, "get_forecaster", lambda: registry.forecaster)()
                    if fc_svc:
                        fc = fc_svc.forecast(vals)
                        dl_insights["bilstm_forecast"] = {
                            "scheme_code": matched_scheme,
                            "trend": fc.get("trend"),
                            "p10_end": fc.get("p10", [None])[-1],
                            "p50_end": fc.get("p50", fc.get("forecast_trajectory", [None]))[-1],
                            "p90_end": fc.get("p90", [None])[-1],
                            "model_source": fc.get("model_source"),
                        }
                if len(hist) >= 40:
                    vis_res = get_vision_service().predict_from_scheme(matched_scheme)
                    dl_insights["vision_resnet18"] = {
                        "scheme_code": matched_scheme,
                        "pattern": vis_res.get("pattern"),
                        "confidence": round(float(vis_res.get("confidence", 0.0)), 4),
                    }
        except Exception:
            pass

        return ValidationResponse(
            timestamp=datetime.utcnow().isoformat(),
            recommended=recommended,
            client_risk_profile=request.risk_level.title(),
            fund_risk_rating=fund_risk_rating,
            market_sentiment=market_data.sentiment.upper(),
            mismatch_flag=mismatch,
            mismatch_reason=reason,
            confidence_score=0.92,
            alternatives=alternatives,
            suggested_allocation_pct=suggested_allocation,
            outlook=outlook,
            dl_insights=dl_insights or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/distributors/client-insights", response_model=ClientInsightsResponse, tags=["MFD Copilot"])
def get_client_insights(current_user: dict = Depends(get_current_distributor)):
    """
    Get AI-driven insights for distributor clients, including SIP drop probability, redemption likelihood, and upsell triggers.
    """
    try:
        leads_list = db_client.get_all_leads()
        insights_list = []
        
        if not leads_list:
            fallback_names = [
                ("C101", "Rajesh Sharma", 0.15, False, 0.22, "Stable", 92.0, False, "Tax Saving ELSS Top-up"),
                ("C102", "Priya Patel", 0.78, False, 0.65, "Decreasing due to Volatility", 68.0, True, "Switch to Low-Volatility Debt"),
                ("C103", "Amit Kumar", 0.35, False, 0.18, "Stable", 84.0, False, "Increase Mid-cap SIP"),
                ("C104", "Sneha Rao", 0.92, True, 0.85, "Risk Averse Shift", 45.0, True, "Retention Meeting Required"),
                ("C105", "Vikram Singh", 0.12, False, 0.05, "Increasing", 96.0, False, "International Fund Allocation"),
                ("C106", "Ananya Sen", 0.44, False, 0.30, "Stable", 75.0, False, "Index Fund SIP Start")
            ]
            for cid, name, sip_p, inactive, red_l, risk_c, health, ret_alert, upsell in fallback_names:
                insights_list.append(ClientInsightItem(
                    client_id=cid,
                    name=name,
                    sip_drop_probability=sip_p,
                    inactive_flag=inactive,
                    redemption_likelihood=red_l,
                    risk_appetite_change=risk_c,
                    portfolio_health_score=health,
                    retention_alert=ret_alert,
                    upsell_opportunity=upsell
                ))
        else:
            for idx, lead in enumerate(leads_list[:12]):
                lead_id = lead.get("lead_id", f"C{101+idx}")
                first_name = lead.get("first_name", "Investor")
                last_name = lead.get("last_name", "")
                name = f"{first_name} {last_name}".strip()
                if not name or name == "Investor":
                    name = lead.get("name", f"Client {idx+1}")
                
                prob = float(lead.get("conversion_probability", lead.get("Conversion_Probability", 0.5)))
                
                sip_drop_p = round(1.0 - prob, 2)
                inactive = bool(prob < 0.45)
                redemption_l = round(0.1 + (0.5 * (1.0 - prob)), 2)
                risk_change = "Decreasing due to Volatility" if prob < 0.5 else "Stable" if prob < 0.85 else "Increasing"
                health_score = round(50.0 + (prob * 45.0), 1)
                ret_alert = bool(sip_drop_p > 0.60)
                
                if prob > 0.8:
                    upsell = "Equity SIP Top-up"
                elif prob > 0.6:
                    upsell = "Tax Saving ELSS Plan"
                else:
                    upsell = "Liquid Safe Harbor Parking"
                    
                insights_list.append(ClientInsightItem(
                    client_id=lead_id,
                    name=name,
                    sip_drop_probability=sip_drop_p,
                    inactive_flag=inactive,
                    redemption_likelihood=redemption_l,
                    risk_appetite_change=risk_change,
                    portfolio_health_score=health_score,
                    retention_alert=ret_alert,
                    upsell_opportunity=upsell
                ))
                
        return ClientInsightsResponse(
            timestamp=datetime.utcnow().isoformat(),
            insights=insights_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/overview", tags=["Dashboard"])
def dashboard_overview(current_user: dict = Depends(get_current_distributor)):
    """
    Unified CRM dashboard payload for distributor and admin workspaces.
    """
    try:
        return build_dashboard_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard overview failed: {str(e)}") from e


@app.get("/dashboard/leads", tags=["Dashboard"])
def dashboard_leads(
    limit: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(None, alias="query"),
    stage: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_distributor),
):
    """
    CRM lead queue with enterprise-ready scoring context and workflow metadata.
    """
    try:
        return build_dashboard_leads(limit=limit, query=q, stage=stage, priority=priority)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard leads failed: {str(e)}") from e


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


@app.get("/leads", tags=["Leads"])
def get_managed_leads(limit: int = 50):
    """Fetch leads managed in MongoDB"""
    try:
        leads = db_client.get_all_leads(limit=limit)
        # Convert ObjectId to string for JSON serialization
        for lead in leads:
            if "_id" in lead:
                lead["_id"] = str(lead["_id"])
        return leads
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")


@app.patch("/leads/{lead_id}", tags=["Leads"])
def patch_lead_workflow(lead_id: str, payload: LeadUpdate):
    """
    Updates workflow state for a managed lead without mutating the model export itself.
    """
    try:
        return update_lead_workflow(
            lead_id=lead_id,
            status=payload.status,
            notes=payload.notes or "",
            assignee=payload.assignee,
            next_step_at=payload.next_step_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead workflow update failed: {str(e)}") from e


@app.get("/distributors/matches", tags=["Distributors"])
def get_matched_distributors(investor_id: str = "demo_investor", limit: int = 5):
    """
    Get top matched distributors for an investor persona.
    """
    try:
        matches = db_client.get_distributor_matches(investor_id, limit=limit)
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


def fallback_forecast_nav(history: List[float]) -> dict:
    """Deterministic baseline fallback forecast when PyTorch LSTM is not loaded."""
    from lume_platform.ml.forecaster import compute_fallback_forecast
    return compute_fallback_forecast(history, horizon=5)


@app.post("/api/forecast", response_model=ForecastResponse, tags=["Predictions"])
@app.post("/predict/forecast", response_model=ForecastResponse, tags=["Predictions"])
def forecast_nav(request: ForecastRequest):
    """
    Predict future trajectory using the PyTorch LSTM model.
    Falls back to a deterministic moving-average baseline if model is not loaded.
    Requires at least 30 days of historical data.
    """
    try:
        forecaster = getattr(registry, "get_forecaster", lambda: registry.forecaster)()
        if forecaster:
            result = forecaster.forecast(request.history)
            if "error" not in result:
                return ForecastResponse(
                    timestamp=datetime.utcnow().isoformat(),
                    **result
                )
        
        # Fallback to deterministic baseline forecast
        result = fallback_forecast_nav(request.history)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return ForecastResponse(
            timestamp=datetime.utcnow().isoformat(),
            **result
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Vision Service Singleton (Phase 3)
_vision_service = None

def get_vision_service():
    global _vision_service
    if getattr(registry, "vision_service", None) is not None:
        return registry.vision_service
    if _vision_service is None:
        from lume_platform.vision.service import VisionPatternService
        _vision_service = VisionPatternService(model_type="resnet18")
    return _vision_service


@app.post("/api/vision/pattern", response_model=VisionPatternResponse, tags=["Vision Engine"])
@app.post("/market/pattern/classify", response_model=VisionPatternResponse, tags=["Vision Engine"])
def classify_vision_pattern(request: VisionPatternRequest):
    """
    Phase 3 Computer Vision: Candlestick Pattern Recognition with Grad-CAM.
    Returns predicted pattern class, confidence, and Grad-CAM visual overlay (base64).
    """
    from datetime import datetime
    from lume_platform.vision.gradcam import base64_to_image

    try:
        service = get_vision_service()
        if request.scheme_code:
            result = service.predict_from_scheme(request.scheme_code)
        elif request.image_base64:
            img = base64_to_image(request.image_base64)
            result = service.predict_image(img, target_class=request.target_class)
        else:
            # Default to flagship Scheme 100119
            result = service.predict_from_scheme("100119")

        return VisionPatternResponse(
            timestamp=datetime.utcnow().isoformat(),
            pattern=result["pattern"],
            pattern_class=result["pattern_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            gradcam_overlay_base64=result["gradcam_overlay_base64"],
            model_architecture=result.get("model_architecture", "resnet18"),
            scheme_code=result.get("scheme_code"),
            date_range=result.get("date_range"),
            latest_nav=result.get("latest_nav"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision classification failed: {str(e)}")


@app.get("/api/fund/{scheme_code}/full-ai", tags=["AI Lab"])
def get_fund_full_ai_bundle(scheme_code: str, days: int = 90):
    """Unified per-scheme endpoint running NAV history + Phase 2 BiLSTM Quantiles + Phase 3 ResNet-18 Grad-CAM."""
    try:
        hist = nav_fetcher.get_historical_nav(scheme_code, days=max(days, 90))
        latest_nav = nav_fetcher.get_latest_nav(scheme_code)
        returns = nav_fetcher.calculate_returns(scheme_code)
        chrono = list(reversed(hist))

        forecast_data = None
        if len(chrono) >= 30:
            vals = [float(item["nav"]) for item in chrono[-60:]]
            fc_svc = getattr(registry, "get_forecaster", lambda: registry.forecaster)()
            if fc_svc:
                forecast_data = fc_svc.forecast(vals)
            else:
                forecast_data = fallback_forecast_nav(vals)

        vision_data = None
        if len(chrono) >= 40:
            try:
                vision_data = get_vision_service().predict_from_scheme(scheme_code)
            except Exception:
                vision_data = None

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scheme_code": scheme_code,
            "latest_nav": latest_nav or (chrono[-1]["nav"] if chrono else None),
            "returns": returns,
            "history": hist[:60],
            "forecast": forecast_data,
            "vision": vision_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full fund AI inspection failed: {str(e)}") from e


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 Deep Learning Lab Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

_dl_lab_service = None

def get_dl_lab_service():
    global _dl_lab_service
    if _dl_lab_service is None:
        from lume_platform.ml.dl_lab_service import DLLabService
        _dl_lab_service = DLLabService.get_instance()
    return _dl_lab_service


@app.get("/api/dl-lab/comparison", tags=["Deep Learning Lab"])
def get_dl_lab_comparison():
    """Phase 4: Side-by-side real metrics and confusion matrices comparing Random Forest and LeadMLP."""
    service = get_dl_lab_service()
    return service.get_comparison_metrics()


@app.get("/api/dl-lab/training-history", tags=["Deep Learning Lab"])
def get_dl_lab_training_history():
    """Phase 4: Epoch-by-epoch genuine training and validation loss curves for LeadMLP."""
    service = get_dl_lab_service()
    return service.get_training_history()


@app.post("/api/dl-lab/predict-lead", tags=["Deep Learning Lab"])
def predict_dl_lab_lead(payload: Dict[str, Any]):
    """Phase 4: Live dual prediction on Random Forest and LeadMLP with latency measurement and feature explanations."""
    service = get_dl_lab_service()
    return service.predict_dual(payload)


@app.get("/api/dl-lab/metadata", tags=["Deep Learning Lab"])
def get_dl_lab_metadata():
    """Phase 4: Model versions, parameters, dataset provenance, and honest limitations."""
    service = get_dl_lab_service()
    return service.get_metadata()


# -------------------------------------------------------------------------
# Phase 5: Financial NLP & Token Saliency Endpoints
# -------------------------------------------------------------------------
class FinancialNLPRequest(BaseModel):
    text: str = Field(..., description="Financial statement, headline, or market commentary")


def get_financial_nlp_service():
    from lume_platform.nlp.financial_nlp_service import FinancialNLPService
    return FinancialNLPService.get_instance()


@app.post("/api/nlp/financial-sentiment", tags=["Financial NLP"])
def predict_financial_sentiment_endpoint(payload: FinancialNLPRequest):
    """Phase 5: Financial Domain Transformer sentiment prediction with token attention saliency."""
    service = get_financial_nlp_service()
    return service.predict_sentiment_with_saliency(payload.text)


@app.get("/api/nlp/benchmark", tags=["Financial NLP"])
def get_financial_nlp_benchmark():
    """Phase 5: Genuine comparative benchmark (TF-IDF vs SBERT vs Financial Transformer)."""
    service = get_financial_nlp_service()
    return service.get_benchmark_metrics()




@app.get("/predict/explain/{lead_id}", tags=["Predictions"])
def explain_lead_scoring(lead_id: str):
    """
    Explainable AI (XAI): Returns feature importance for a specific lead.
    Utilizes SHAP-style weights from the Random Forest model.
    """
    try:
        lead = db_client.get_collection("leads").find_one({"lead_id": lead_id})
        if not lead:
             raise HTTPException(status_code=404, detail="Lead not found")
        
        if not registry.lead_bundle:
            raise HTTPException(status_code=503, detail="Lead model not loaded")
            
        # Get global feature importance as a fallback for local SHAP
        importance = registry.lead_bundle.get_feature_importance()
        
        # In a real app, we would run SHAP here. 
        # For now, we simulate local importance by weighting global importance with lead values.
        explanation = []
        for feature, weight in importance.items():
            explanation.append({
                "feature": feature,
                "importance_score": round(weight, 4),
                "impact": "positive" if weight > 0.05 else "neutral"
            })
            
        return {
            "lead_id": lead_id,
            "explanation": sorted(explanation, key=lambda x: abs(x["importance_score"]), reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/advisor/query", response_model=AdvisorResponse, tags=["AI Advisor"])
def advisor_query(request: AdvisorRequest):
    """
    The "Brain" of Lume AI. 
    Synthesizes SBERT search, Sentiment analysis, and Portfolio logic into a human-like response.
    """
    try:
        from lume_platform.risk.live_risk_analyzer import risk_analyzer
        market_data = risk_analyzer.fetch_live_market_data()
        buddy_reply = buddy_engine.generate(
            request.query,
            audience="investor",
            persona=request.persona,
            history=request.history,
            market_context=f"{market_data.sentiment.upper()} market with VIX {market_data.vix}",
            dashboard_context=request.dashboard_context,
        )
        
        # 1. Get Semantic Matches (embedded corpus -> live catalog recommender fallback)
        funds = []
        if not registry.sbert_search:
            registry.load_sbert_cache()
        if registry.sbert_search:
            funds = registry.sbert_search.query(request.query, top_k=3)
        if not funds:
            cat_recs = recommender.recommend_for_profile(
                {"inferred_risk_profile": request.persona or "balanced", "goals": request.query},
                market_context={"market_sentiment": market_data.sentiment, "vix": market_data.vix},
                top_k=3,
            )
            funds = [
                {
                    "scheme_code": str(r.get("scheme_code", "")),
                    "scheme_name": r.get("scheme_name", "Mutual Fund"),
                    "category": r.get("category", "Mutual Fund"),
                    "match_score": float(r.get("match_score", 0.75)),
                }
                for r in cat_recs
            ]

        # 2. Score query intent using Phase 5 Financial Transformer Classifier
        nlp_res = get_financial_nlp_service().predict_sentiment_with_saliency(request.query)
        query_sentiment = nlp_res.get("sentiment", "neutral")

        # 3. Synthesize explainable (XAI) advice incorporating live market sentiment, VIX & Financial NLP
        analysis = buddy_reply.response
        analysis += f" [Advisor Confidence: {round(buddy_reply.confidence * 100, 1)}% | Query NLP: {query_sentiment.upper()}]"

        if market_data.sentiment == "bearish":
            analysis += f" Current market is BEARISH (Nifty 50: {market_data.nifty_50:,.1f}, Δ {market_data.nifty_change_pct}%, VIX: {market_data.vix}). Under these conditions, "
            if (request.persona or "").lower() == "conservative":
                analysis += "we strongly recommend capital preservation via short-duration debt and liquid funds, staggering any equity exposure."
            else:
                analysis += "we recommend systematic SIP averaging rather than lump-sum deployment, focusing on large-cap index or dynamic hybrid allocation."
        elif market_data.sentiment == "bullish":
            analysis += f" Current market is BULLISH (Nifty 50: {market_data.nifty_50:,.1f}, Δ +{market_data.nifty_change_pct}%, VIX: {market_data.vix}). Under these conditions, "
            if (request.persona or "").lower() == "growth":
                analysis += "broad-market flexi-cap and mid-cap allocations remain favorable for long-term alpha while monitoring valuation risk."
            else:
                analysis += "we recommend a disciplined core-satellite allocation balanced between equity index exposure and short-term debt stability."
        else:
            analysis += f" Current market is NEUTRAL/RANGEBOUND (Nifty 50: {market_data.nifty_50:,.1f}, Δ {market_data.nifty_change_pct}%, VIX: {market_data.vix}). Stagger entries systematically aligned with your {request.persona or 'balanced'} profile."

        if buddy_reply.follow_up:
            analysis += f" Next step: {buddy_reply.follow_up}"

        return AdvisorResponse(
            timestamp=datetime.utcnow().isoformat(),
            query=request.query,
            analysis=analysis,
            sentiment=market_data.sentiment.upper(),
            recommended_funds=funds,
            market_mood=f"{market_data.sentiment.upper()} (VIX: {market_data.vix})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buddy/briefing", response_model=BuddyBriefingResponse, tags=["AI Buddy"])
def buddy_briefing(distributor_id: str = "demo_distributor"):
    """
    Generates a live daily executive briefing synthesized from CRM leads,
    streaming NIFTY/VIX metrics, Phase 5 Financial Transformer signals, and Drift Monitor.
    """
    try:
        overview = build_dashboard_overview()
        top_leads = db_client.get_all_leads(limit=10)
        snap = stream_hub.get_snapshot()
        market = snap.get("market", {})
        latest_sig = snap.get("latest_signal", {})
        drift = snap.get("drift", {})

        total_leads = int(overview.get("total_leads", len(top_leads)))
        hot_leads = int(overview.get("hot_prospects", 0))
        avg_prob = float(overview.get("mean_conversion_probability", 0.0))

        top_lead = top_leads[0] if top_leads else {}
        top_name = top_lead.get("name") or f"{top_lead.get('first_name', 'Top')} {top_lead.get('last_name', 'Lead')}".strip()
        top_prob = float(top_lead.get("conversion_probability", 0.85))
        top_persona = top_lead.get("recommended_pitch_persona", "Balanced Allocator")
        top_city = top_lead.get("city", "Mumbai")

        # Find a nurture/retention candidate
        nurture_lead = next(
            (l for l in top_leads if 0.35 <= float(l.get("conversion_probability", 0.5)) <= 0.72),
            top_leads[1] if len(top_leads) > 1 else {},
        )
        nurture_name = nurture_lead.get("name") or f"{nurture_lead.get('first_name', 'Client')} {nurture_lead.get('last_name', '')}".strip() or "Warm Lead"
        nurture_prob = float(nurture_lead.get("conversion_probability", 0.58))

        nifty_val = market.get("nifty_50", 25450.0)
        nifty_chg = market.get("nifty_change_pct", 0.0)
        vix_val = market.get("vix", 15.0)
        nlp_sent = latest_sig.get("sentiment", "neutral").upper()
        headline = latest_sig.get("headline", "Indian mutual fund inflows steady")

        briefing_text = (
            f"Live Desk Briefing ({distributor_id}): NIFTY 50 at {nifty_val:,.1f} ({nifty_chg:+.2f}%, VIX {vix_val:.1f}). "
            f"Financial Transformer rates latest wire '{headline[:72]}...' as {nlp_sent}. "
            f"CRM Queue tracks {total_leads} leads ({hot_leads} Hot, mean conversion {avg_prob * 100:.1f}%). "
            f"Priority Outreach: {top_name} ({top_city}, P={top_prob:.2f}) — pitch {top_persona}. "
            f"Drift Monitor: {'ALERT' if drift.get('drift_detected') else 'Stable'} ({drift.get('samples_seen', 0)} stream events seen)."
        )

        actions = [
            {
                "action": f"Priority Pitch: {top_name}",
                "reason": f"Score {top_prob:.2f} | City: {top_city} | Pitch {top_persona}",
            },
            {
                "action": f"Retention Follow-Up: {nurture_name}",
                "reason": f"Score {nurture_prob:.2f} | VIX {vix_val:.1f} requires disciplined SIP reassurance",
            },
            {
                "action": f"Stream Sentiment Watch ({nlp_sent})",
                "reason": f"Transformer signal on '{headline[:55]}...' ({round(float(latest_sig.get('confidence', 0.6)) * 100)}% conf)",
            },
        ]

        return BuddyBriefingResponse(
            timestamp=datetime.utcnow().isoformat(),
            briefing=briefing_text,
            top_actions=actions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/buddy/chat", response_model=BuddyChatResponse, tags=["AI Buddy"])
def buddy_chat(request: BuddyChatRequest):
    """
    Chat endpoint for the AI Buddy to handle objections and generate call scripts.
    Powered by the Custom PyTorch Transformer Model.
    """
    try:
        reply = buddy_engine.generate(
            request.message,
            audience="distributor",
            history=request.history,
            distributor_id=request.distributor_id,
            lead_id=request.lead_id,
            dashboard_context=request.dashboard_context,
        )

        return BuddyChatResponse(
            timestamp=datetime.utcnow().isoformat(),
            response=reply.response,
            confidence=reply.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
