"""
End-to-End Tests for Real-Time Streaming, Demo Investor & Demo Distributor Profiles,
All 7 ML/DL Models, and Big Data Lakehouse Telemetry.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LUME_OFFLINE_MARKET", "1")
os.environ.setdefault("LUME_ENABLE_REALTIME_STREAM", "0")
os.environ.setdefault("LUME_START_KAFKA_LISTENER", "false")
os.environ.setdefault("LUME_START_DRIFT_MONITOR", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from api.main_enhanced import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_health_all_models_loaded(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    models = body["models_loaded"]
    for key in (
        "lead_bundle",
        "investor_bundle",
        "semantic_search",
        "lead_mlp",
        "bilstm_forecaster",
        "vision_resnet18",
        "financial_transformer",
    ):
        assert models.get(key) is True, f"Expected model {key} to be loaded, got: {models}"


def test_demo_investor_login_and_portfolio(client: TestClient):
    res = client.post("/auth/demo-login", json={"role": "investor"})
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "investor"
    assert data["is_demo"] is True
    assert data["email"] == "demo.investor@lumeai.in"
    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    me_json = me.json()
    assert me_json["role"] == "investor"
    assert me_json["is_demo"] is True

    port = client.get("/portfolio", headers=headers)
    assert port.status_code == 200
    port_json = port.json()
    assert len(port_json["holdings"]) >= 4
    assert port_json["total_invested"] > 0
    assert port_json["total_current"] > 0

    pa = client.get("/portfolio/analytics?horizon=1y", headers=headers)
    assert pa.status_code == 200
    pa_json = pa.json()
    assert "kpis" in pa_json

    news = client.get("/news/live?max_articles=4", headers=headers)
    assert news.status_code == 200
    news_items = news.json()
    assert isinstance(news_items, list) and len(news_items) > 0
    assert "token_saliency" in news_items[0]


def test_demo_distributor_login_crm_and_dl_validation(client: TestClient):
    res = client.post("/auth/demo-login", json={"role": "distributor"})
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "distributor"
    assert data["is_demo"] is True
    assert data["email"] == "demo.distributor@lumeai.in"
    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    ov = client.get("/dashboard/overview", headers=headers)
    assert ov.status_code == 200
    ov_json = ov.json()
    assert ov_json["total_leads"] > 0
    assert "conversion_rate" in ov_json

    leads = client.get("/dashboard/leads?limit=10", headers=headers)
    assert leads.status_code == 200
    leads_json = leads.json()
    assert isinstance(leads_json, list) and len(leads_json) > 0

    briefing = client.get("/buddy/briefing?distributor_id=demo_distributor")
    assert briefing.status_code == 200
    b_json = briefing.json()
    assert "Live Desk Briefing" in b_json["briefing"]
    assert len(b_json["top_actions"]) >= 2

    val = client.post(
        "/distributors/validate-rec",
        headers=headers,
        json={
            "client_id": "DEMO-C1",
            "fund_name": "HDFC Balanced Advantage Fund",
            "category": "Balanced",
            "risk_level": "balanced",
        },
    )
    assert val.status_code == 200
    val_json = val.json()
    assert "dl_insights" in val_json and val_json["dl_insights"] is not None
    assert "financial_transformer" in val_json["dl_insights"]


def test_websocket_realtime_and_pulse(client: TestClient):
    pulse = client.get("/api/realtime/pulse?advance=true")
    assert pulse.status_code == 200
    p_json = pulse.json()
    assert p_json["type"] == "tick"
    assert p_json["models_online"] == 7
    assert "market" in p_json and "nifty_50" in p_json["market"]
    assert "latest_signal" in p_json and "token_saliency" in p_json["latest_signal"]

    with client.websocket_connect("/ws/realtime") as ws:
        init_pkt = ws.receive_json()
        assert "market" in init_pkt
        assert init_pkt["models_total"] == 7

        ws.send_text("tick")
        tick_pkt = ws.receive_json()
        assert tick_pkt["type"] == "tick"
        assert tick_pkt["tick_seq"] >= init_pkt["tick_seq"]
        assert "drift" in tick_pkt

        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_bigdata_live_stats_and_all_ai_engines(client: TestClient):
    bd = client.get("/api/bigdata/live-stats")
    assert bd.status_code == 200
    bd_json = bd.json()
    assert bd_json["bronze_amfi"]["csv_rows"] > 10000
    assert bd_json["silver_lake"]["file_count"] > 0
    assert bd_json["models_total"] == 7
    assert bd_json["models_online"] == 7
    assert len(bd_json["models_matrix"]) == 7

    # Search
    sr = client.get("/search/funds?query=balanced&top_k=5")
    assert sr.status_code == 200
    assert len(sr.json()["results"]) >= 1

    # Dual Lead RF + MLP
    dl_lead = client.post(
        "/api/dl-lab/predict-lead",
        json={
            "totalvisits": 9,
            "total_time_spent_on_website": 1400,
            "page_views_per_visit": 4.2,
            "lead_quality": "High in Relevance",
            "lead_source": "Google",
        },
    )
    assert dl_lead.status_code == 200
    dl_json = dl_lead.json()
    assert dl_json["random_forest"]["status"] == "online"
    assert dl_json["lead_mlp"]["status"] == "online"

    # Phase 5 Transformer NLP
    nlp = client.post(
        "/api/nlp/financial-sentiment",
        json={"text": "Strong SIP inflows drive mutual fund AUM growth cleanly"},
    )
    assert nlp.status_code == 200
    assert nlp.json()["sentiment"] in ("positive", "neutral", "negative")

    # Phase 2 BiLSTM Forecast
    fc = client.post(
        "/api/forecast",
        json={"history": [140.0 + i * 0.25 for i in range(35)]},
    )
    assert fc.status_code == 200
    fc_json = fc.json()
    assert len(fc_json["forecast_trajectory"]) == 5

    # Phase 3 ResNet-18 Vision + Grad-CAM
    vis = client.post("/api/vision/pattern", json={"scheme_code": "100119"})
    assert vis.status_code == 200
    vis_json = vis.json()
    assert vis_json["gradcam_overlay_base64"].startswith("data:image/png;base64,")
    assert vis_json["model_architecture"] == "resnet18"

    # Full per-fund AI bundle + Advisor
    bundle = client.get("/api/fund/100119/full-ai?days=90")
    assert bundle.status_code == 200
    b_json = bundle.json()
    assert b_json["forecast"] is not None
    assert b_json["vision"] is not None

    adv = client.post(
        "/advisor/query",
        json={"query": "Should I invest in balanced advantage funds?", "persona": "balanced"},
    )
    assert adv.status_code == 200
    assert "Query NLP:" in adv.json()["analysis"]
