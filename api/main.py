"""
FastAPI service: /predict, /analytics, /insights.
Models load once at startup via ModelRegistry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from lume_platform.config import DATA_ROOT, EXPORT_DIR, PROJECT_ROOT, TABLEAU_PUBLIC_EMBED_URL
from lume_platform.inference.registry import ModelRegistry
from lume_platform.db.mongo_client import db_client

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lume AI Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()


@app.on_event("startup")
def startup() -> None:
    registry.load()


class LeadFeatures(BaseModel):
    """Aligned with ``lume_platform.ml.feature_config`` (training / bundles)."""

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


class PredictRequest(BaseModel):
    task: str = Field(..., description="lead_scoring | investor_cluster | sentiment")
    lead: LeadFeatures | None = None
    investor_behavior: dict[str, float] | None = None
    text: str | None = None


@app.post("/predict")
def predict(body: PredictRequest):
    try:
        if body.task == "lead_scoring":
            if not registry.lead_bundle or not body.lead:
                raise HTTPException(status_code=503, detail="Lead model not loaded or missing payload")
            d = body.lead.model_dump(by_alias=True)
            pred, proba = registry.lead_bundle.predict_row(d)
            
            # Persist to MongoDB
            lead_id = f"L-{hash(str(d)) % 100000}"
            db_client.upsert_lead(lead_id, {
                **d,
                "converted_prediction": int(pred),
                "conversion_probability": float(proba),
                "timestamp": str(Path(__file__).stat().st_mtime) # Simple timestamp
            })
            
            return {"task": body.task, "label": pred, "conversion_probability": proba, "lead_id": lead_id}
        if body.task == "investor_cluster":
            if not registry.investor_bundle or not body.investor_behavior:
                raise HTTPException(status_code=503, detail="Cluster model not loaded or missing payload")
            cid = registry.investor_bundle.predict_row(body.investor_behavior)
            return {"task": body.task, "cluster_id": cid}
        if body.task == "sentiment":
            if not registry.sentiment_bundle or not body.text:
                raise HTTPException(status_code=503, detail="NLP model not loaded or missing text")
            label, conf = registry.sentiment_bundle.predict_text(body.text)
            return {"task": body.task, "sentiment": label, "confidence": conf}
        raise HTTPException(status_code=400, detail="Unknown task")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/analytics")
def analytics():
    """Aggregates from production export CSVs (post-Spark / post-training)."""
    out: dict = {"data_root": str(DATA_ROOT)}
    leads_path = EXPORT_DIR / "distributor_leads_master.csv"
    if leads_path.is_file():
        import pandas as pd

        df = pd.read_csv(leads_path)
        out["distributor_leads"] = {
            "rows": int(len(df)),
            "mean_conversion_probability": float(df["Conversion_Probability"].mean())
            if "Conversion_Probability" in df.columns
            else None,
            "tier_hot_count": int((df["Conversion_Probability"] > 0.85).sum())
            if "Conversion_Probability" in df.columns
            else None,
        }
    inv_path = EXPORT_DIR / "investor_routing_matches.csv"
    if inv_path.is_file():
        import pandas as pd

        df = pd.read_csv(inv_path)
        out["investor_routing"] = {
            "rows": int(len(df)),
            "persona_counts": df["Persona_Cluster"].value_counts().to_dict()
            if "Persona_Cluster" in df.columns
            else {},
        }
    return out


@app.get("/leads")
def get_leads(limit: int = 50):
    """Fetches managed leads from MongoDB."""
    return db_client.get_all_leads(limit=limit)


@app.get("/insights")
def insights():
    """Model cards + BI embed URL for operators."""
    rf = PROJECT_ROOT / "model_evaluations/random_forest/real_metrics.json"
    nlp = PROJECT_ROOT / "model_evaluations/nlp_sentiment/metrics.json"
    payload: dict = {
        "tableau_embed_url": os.environ.get("LUME_TABLEAU_EMBED_URL", TABLEAU_PUBLIC_EMBED_URL),
        "random_forest": None,
        "nlp_sentiment": None,
    }
    if rf.is_file():
        with open(rf) as f:
            payload["random_forest"] = json.load(f)
    if nlp.is_file():
        with open(nlp) as f:
            payload["nlp_sentiment"] = json.load(f)
    payload["models_loaded"] = {
        "lead_bundle": registry.lead_bundle is not None,
        "investor_bundle": registry.investor_bundle is not None,
        "sentiment_bundle": registry.sentiment_bundle is not None,
    }
    return payload


@app.get("/health")
def health():
    return {"status": "ok"}
