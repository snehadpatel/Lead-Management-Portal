"""
Lume AI — Streamlit dashboard (dataset audit, ML inference, BI embed).
Run from repo root: PYTHONPATH=src streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lume_platform.audit.dataset_audit import quick_audit_report
from lume_platform.config import DATA_ROOT, EXPORT_DIR, TABLEAU_PUBLIC_EMBED_URL
from lume_platform.inference.registry import ModelRegistry

# Legacy search engine (TF-IDF funds)
sys.path.append(str(ROOT / "scripts"))
try:
    from search_algorithm import MutualFundSearchEngine
except ImportError:
    MutualFundSearchEngine = None  # type: ignore

st.set_page_config(
    page_title="Lume AI Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "registry" not in st.session_state:
    reg = ModelRegistry()
    reg.load()
    st.session_state.registry = reg

if "search_engine" not in st.session_state:
    try:
        p = DATA_ROOT / "structured/mutual_funds/amfi_scheme_list.csv"
        st.session_state.search_engine = (
            MutualFundSearchEngine(str(p)) if MutualFundSearchEngine and p.is_file() else None
        )
    except Exception:
        st.session_state.search_engine = None


@st.cache_data
def load_csv(path: str, nrows: int | None = None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()


with st.sidebar:
    st.title("Lume AI")
    st.caption("Spark-scale ingestion · sklearn / XGBoost · NLP core")
    st.divider()
    page = st.radio(
        "Modules",
        [
            "Dataset audit (pipeline)",
            "Command center",
            "Model evidence (reports)",
            "Distributor leads",
            "Investor clustering",
            "Semantic fund search",
            "NLP sentiment",
            "LSTM-style forecast demo",
            "⚡ Live AI Pulse",
            "BI dashboard (embedded)",
        ],
    )
    st.divider()
    st.caption("v3.0 · modular `lume_platform`")


if page == "Dataset audit (pipeline)":
    st.header("Dataset audit — sample → Pandas")
    st.markdown(
        "Full **~31 GB** tree is processed in **Apache Spark** (see `notebooks/colab_spark_pipeline.ipynb`). "
        "Here we run the same audit steps on a bounded CSV sample for speed."
    )
    audit_path = st.text_input(
        "CSV path (under datasets/)",
        value=str(DATA_ROOT / "structured/leads/lead_scoring/Lead Scoring.csv"),
    )
    sample = st.slider("Sample rows", 1000, 50_000, 20_000)
    if st.button("Run audit"):
        plots_dir = ROOT / "dataset_visualisations" / "audit_latest"
        with st.spinner("Auditing..."):
            rep = quick_audit_report(Path(audit_path), sample_rows=sample, plot_out=plots_dir)
        st.subheader("Missing % (top columns)")
        st.json(rep["missing_pct_top"])
        st.subheader("Duplicates")
        st.json(rep["duplicates"])
        st.subheader("Summary (sample)")
        st.dataframe(pd.DataFrame(rep["summary"]).transpose().head(40))
        c1, c2, c3 = st.columns(3)
        for col, key, title in [
            (c1, "histogram", "Histogram"),
            (c2, "boxplot", "Box plots"),
            (c3, "heatmap", "Heatmap"),
        ]:
            p = rep["plots"].get(key)
            if p and Path(p).is_file():
                col.image(Image.open(p), caption=title, width='stretch')


elif page == "Command center":
    st.title("Command center")
    st.metric("Dataset root", str(DATA_ROOT))
    reg: ModelRegistry = st.session_state.registry
    st.json(
        {
            "lead_model": reg.lead_bundle is not None,
            "investor_model": reg.investor_bundle is not None,
            "sentiment_model": reg.sentiment_bundle is not None,
        }
    )
    st.caption("Regenerate artifacts: `PYTHONPATH=src python -m lume_platform.ml.training`")


elif page == "Model evidence (reports)":
    st.title("Model evidence — aligned with training JSON")
    st.caption("Numbers in `model_evaluations/**.json` are the source of truth; Markdown reports are auto-generated.")
    master = ROOT / "model_evaluations/MODELS_EVALUATION_REPORT.md"
    if master.is_file():
        st.markdown(master.read_text(encoding="utf-8"))
    for path, cap in [
        ("model_evaluations/random_forest/rf_confusion_matrix.png", "Lead model — confusion matrix"),
        ("model_evaluations/kmeans/kmeans_cluster_projection.png", "K-Means — PCA projection"),
        ("model_evaluations/kmeans/kmeans_centroids_heatmap.png", "K-Means — centroids"),
        ("model_evaluations/xai_insights/xai_rf_feature_weights.png", "Legacy RF feature weights (if present)"),
        ("model_evaluations/tfidf_search/cosine_similarity_decay.png", "TF-IDF retrieval"),
        ("model_evaluations/lstm_forecaster/lstm_nav_predictions.png", "LSTM validation plot"),
    ]:
        p = ROOT / path
        if p.is_file():
            st.subheader(cap)
            st.image(Image.open(p), width='stretch')


elif page == "Distributor leads":
    st.title("Distributor lead intelligence")
    df = load_csv(str(EXPORT_DIR / "distributor_leads_master.csv"))
    if df.empty:
        st.warning("Run training: `PYTHONPATH=src python -m lume_platform.ml.training`")
    else:
        st.dataframe(df.head(200), width='stretch')
        st.bar_chart(df["Recommended_Pitch_Persona"].value_counts())
    reg: ModelRegistry = st.session_state.registry
    st.subheader("Live inference (bundle)")
    if reg.lead_bundle:
        b = reg.lead_bundle
        row: dict = {}
        num_defaults = {
            "TotalVisits": 5.0,
            "Total Time Spent on Website": 800.0,
            "Page Views Per Visit": 2.5,
            "Asymmetrique Activity Score": 15.0,
            "Asymmetrique Profile Score": 15.0,
        }
        cat_defaults = {
            "Lead Origin": "API",
            "Lead Source": "Organic Search",
            "Specialization": "Select",
            "What is your current occupation": "Unemployed",
            "Last Activity": "Page Visited on Website",
            "Country": "India",
            "Lead Quality": "Low in Relevance",
            "Do Not Email": "No",
            "Do Not Call": "No",
        }
        c1, c2 = st.columns(2)
        for i, name in enumerate(b.numeric_features):
            w = c1 if i % 2 == 0 else c2
            row[name] = w.number_input(
                name, value=float(num_defaults.get(name, 0.0)), format="%.2f"
            )
        for i, name in enumerate(b.cat_features):
            w = c1 if (i + len(b.numeric_features)) % 2 == 0 else c2
            row[name] = w.text_input(name, cat_defaults.get(name, "Unknown"))
        if st.button("Score lead"):
            pred, proba = b.predict_row(row)
            st.success(f"Predicted converted={pred}, P(convert)={proba:.3f}")
    try:
        with open(ROOT / "model_evaluations/random_forest/real_metrics.json") as f:
            st.json(json.load(f))
    except FileNotFoundError:
        pass


elif page == "Investor clustering":
    st.title("Investor personas (K-Means + optional DBSCAN)")
    df = load_csv(str(EXPORT_DIR / "investor_routing_matches.csv"))
    if not df.empty:
        st.dataframe(df.head(200), width='stretch')
    reg: ModelRegistry = st.session_state.registry
    if reg.investor_bundle:
        st.subheader("Assign cluster from behavior vector")
        cols = reg.investor_bundle.behavior_cols
        vals = {}
        c1, c2 = st.columns(2)
        for i, name in enumerate(cols):
            w = c1 if i % 2 == 0 else c2
            vals[name] = w.slider(name, 0.0, 10.0, 5.0)
        if st.button("Predict cluster"):
            cid = reg.investor_bundle.predict_row(vals)
            st.info(f"Cluster id: {cid}")


elif page == "NLP sentiment":
    st.title("Sentiment (TF-IDF + logistic)")
    reg: ModelRegistry = st.session_state.registry
    txt = st.text_area("Text", "Markets rally after policy support for infrastructure.")
    if st.button("Classify") and reg.sentiment_bundle:
        label, conf = reg.sentiment_bundle.predict_text(txt)
        st.success(f"**{label}** (confidence ~ {conf:.3f})")
    try:
        with open(ROOT / "model_evaluations/nlp_sentiment/metrics.json") as f:
            st.json(json.load(f))
    except FileNotFoundError:
        pass


elif page == "Semantic fund search":
    st.title("TF-IDF fund retrieval")
    q = st.text_input("Query", "low risk liquid debt")
    if st.button("Search") and st.session_state.search_engine:
        for r in st.session_state.search_engine.query(q, top_k=5):
            st.write(f"**{r['Match_Score']}%** — {r['Scheme_Name']} ({r['Category']})")


elif page == "LSTM-style forecast demo":
    st.title("NAV / index trajectory demo")
    st.caption("For full LSTM training see `scripts/train_lstm_predictor.py`. This page plots historical tail + stochastic drift.")
    reg: ModelRegistry = st.session_state.registry
    nifty = ROOT / "datasets/structured/stock_prices/nifty50_index/nse_nifty50_historical_merged.csv"
    if Path(nifty).is_file():
        px = load_csv(nifty)
        lookback_vals = px["Close"].tail(30).values.tolist()
        
        if reg.forecaster:
            with st.spinner("AI Generating 5-day trajectory..."):
                f_res = reg.forecaster.forecast(lookback_vals)
                if "error" not in f_res:
                    preds = f_res["forecast_trajectory"]
                    chart = pd.DataFrame({
                        "historical": lookback_vals + [None] * 5,
                        "forecast": [None] * 29 + [lookback_vals[-1]] + preds
                    })
                    st.line_chart(chart)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend", f_res["trend"])
                    c2.metric("Confidence", f"{f_res['confidence_score']}%")
                    c3.metric("Precision", f_res["accuracy_metric"])
                else:
                    st.error(f_res["error"])
        else:
            # Fallback to random if model not loaded
            base = np.array(lookback_vals)
            preds = [float(base[-1]) + float(np.random.normal(0, 15)) for _ in range(5)]
            chart = pd.DataFrame({"hist": list(base) + [None] * 5, "proj": [None] * 4 + [base[-1]] + preds})
            st.line_chart(chart)
            st.warning("Forecasting model not found. Showing stochastic drift placeholder.")


elif page == "BI dashboard (embedded)":
    st.title("Tableau Public (embedded)")
    url = os.environ.get("LUME_TABLEAU_EMBED_URL", TABLEAU_PUBLIC_EMBED_URL)
    st.markdown(
        "Publish a workbook to [Tableau Public](https://public.tableau.com/) and set `LUME_TABLEAU_EMBED_URL` "
        "to the share embed link. Export CSVs from `tableau_exports/` or Parquet from `artifacts/cleaned_parquet/`."
    )
    components.iframe(url, height=900, scrolling=True)


else:
    st.error("Unknown page")

if page == "⚡ Live AI Pulse":
    st.header("⚡ Real-Time AI Market Pulse")
    st.markdown("Distributed Spark stream scoring news & market velocity in real-time.")
    
    col_mkt, col_sent = st.columns([2, 1])
    
    # 1. Market Stream
    with col_mkt:
        st.subheader("🏦 Live Index Monitor")
        mkt_path = ROOT / "streaming/outputs/market_live.parquet"
        if mkt_path.exists():
            try:
                df_live = pd.read_parquet(mkt_path).sort_values("timestamp", ascending=False).head(20)
                if not df_live.empty:
                    latest = df_live.iloc[0]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("NIFTY 50", f"₹{latest.get('last_price', 0):,.2f}", f"{latest.get('pct_change', 0):+.2f}%")
                    m2.metric("Variation", f"{latest.get('variation', 0):+.2f}")
                    m3.metric("Status", str(latest.get('status', 'N/A')))
                    
                    import plotly.express as px
                    fig = px.line(df_live, x="timestamp", y="last_price", title="Real-Time Price Velocity")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Waiting for data packets...")
            except Exception as e:
                st.warning("Connecting to Spark... (Check if streaming_pipeline.py is running)")
        else:
            st.info("📡 Spark Stream Offline. Run `src/lume_platform/spark/streaming_pipeline.py` first.")

    # 2. Sentiment Stream
    with col_sent:
        st.subheader("🧠 Intelligence Feed")
        news_path = ROOT / "streaming/outputs/news_live.parquet"
        if news_path.exists():
            try:
                df_news = pd.read_parquet(news_path).sort_values("ingestion_timestamp", ascending=False).head(10)
                if not df_news.empty:
                    latest_sent = df_news.iloc[0].get('sentiment_score', 0.5)
                    st.metric("Market Sentiment", f"{latest_sent*100:.1f}%", help="AI scored headline context")
                    
                    st.markdown("#### Latest News Impact")
                    for _, row in df_news.head(5).iterrows():
                        impact = row.get('market_impact', 'NEUTRAL')
                        st.write(f"**{row.get('title', 'N/A')}**")
                        st.caption(f"Impact: {impact}")
                        st.divider()
                else:
                    st.info("Awaiting news scoring...")
            except Exception:
                pass
        else:
            st.info("No live news stream detected.")

    if st.button("Refresh Stream"):
        st.rerun()
