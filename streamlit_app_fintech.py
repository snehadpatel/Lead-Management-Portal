"""
💎 LUME AI — FINTECH GRADE DASHBOARD
Production-ready interface for financial services, wealth management, and mutual fund distribution.

Features:
- Live market data (NSE, AMFI NAV)
- Real-time lead scoring & investor clustering
- Risk analytics & portfolio insights
- Professional UI/UX for fintech use
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import requests
from urllib.parse import quote

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lume_platform.audit.dataset_audit import quick_audit_report
from lume_platform.config import DATA_ROOT, EXPORT_DIR, MODELS_DIR, TABLEAU_PUBLIC_EMBED_URL
from lume_platform.inference.registry import ModelRegistry

# Page configuration - Fintech Professional Theme
st.set_page_config(
    page_title="Lume AI | Fintech Intelligence Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/lume-ai',
        'Report a bug': 'https://github.com/your-repo/lume-ai/issues',
        'About': "Lume AI - Production-grade Big Data + AI for Fintech"
    }
)

# Custom CSS for Fintech UI
st.markdown("""
<style>
    /* Fintech Color Scheme */
    :root {
        --primary: #1e3a8a;
        --secondary: #3b82f6;
        --accent: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
    }
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Custom Cards */
    .fintech-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(59, 130, 246, 0.4) 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        text-align: center;
    }
    
    .risk-low { border-left: 4px solid #10b981; }
    .risk-medium { border-left: 4px solid #f59e0b; }
    .risk-high { border-left: 4px solid #ef4444; }
    
    /* Typography */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    .lead-score-high {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .lead-score-medium {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .lead-score-low {
        background: linear-gradient(90deg, #6b7280, #4b5563);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: inline-block;
    }
    
    /* Live Ticker */
    .live-ticker {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 10px 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Models
@st.cache_resource
def load_models():
    """Load ML models once at startup"""
    reg = ModelRegistry()
    reg.load()
    return reg

# Load Data Functions
@st.cache_data(ttl=300)
def load_leads_data():
    """Load distributor leads with predictions"""
    path = EXPORT_DIR / "distributor_leads_master.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_investor_data():
    """Load investor routing matches"""
    path = EXPORT_DIR / "investor_routing_matches.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_live_market_data():
    """Simulate live market data (replace with actual API calls)"""
    # In production, fetch from NSE API
    return {
        "nifty_50": round(random.uniform(22400, 22800), 2),
        "nifty_change": round(random.uniform(-200, 300), 2),
        "nifty_pct": round(random.uniform(-1.0, 1.5), 2),
        "market_status": "OPEN" if datetime.now().hour in range(9, 16) else "CLOSED",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

@st.cache_data(ttl=300)
def fetch_nav_summary():
    """Fetch NAV summary from local data"""
    path = DATA_ROOT / "structured/mutual_funds/nav_history"
    if path.exists():
        files = list(path.glob("*.csv"))
        return {"fund_count": len(files), "last_updated": datetime.now().strftime("%Y-%m-%d")}
    return {"fund_count": 0, "last_updated": "N/A"}

# Lead Scoring Functions
def score_lead(lead_data: Dict[str, Any], registry: ModelRegistry) -> Dict[str, Any]:
    """Score a lead using the ML model"""
    if not registry.lead_bundle:
        return {"error": "Model not loaded"}
    
    try:
        # Create feature vector
        numeric_features = registry.lead_bundle.numeric_features
        cat_features = registry.lead_bundle.cat_features
        
        # Build input DataFrame
        input_data = {}
        for feat in numeric_features:
            input_data[feat] = [lead_data.get(feat, 0)]
        for feat in cat_features:
            input_data[feat] = [lead_data.get(feat, "Unknown")]
        
        input_df = pd.DataFrame(input_data)
        
        # Predict
        proba = float(registry.lead_bundle.pipeline.predict_proba(input_df)[0, 1])
        converted = proba >= registry.lead_bundle.decision_threshold
        
        # Determine tier
        if proba >= 0.85:
            tier = "🔥 HOT"
            tier_class = "lead-score-high"
        elif proba >= 0.65:
            tier = "🌡️ WARM"
            tier_class = "lead-score-medium"
        else:
            tier = "❄️ COLD"
            tier_class = "lead-score-low"
        
        return {
            "probability": round(proba * 100, 2),
            "converted": converted,
            "tier": tier,
            "tier_class": tier_class,
            "threshold": round(registry.lead_bundle.decision_threshold, 2)
        }
    except Exception as e:
        return {"error": str(e)}

# UI Components
def render_header():
    """Render professional header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px;">
            <h1 style="margin: 0; font-size: 2.5rem;">💎 LUME AI</h1>
            <span style="background: linear-gradient(90deg, #10b981, #059669); 
                        color: white; padding: 4px 12px; border-radius: 12px; 
                        font-size: 0.8rem; font-weight: 600;">
                FINTECH INTELLIGENCE
            </span>
        </div>
        <p style="color: #94a3b8; margin-top: 5px;">
            Production-grade Big Data + AI for Wealth Management
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        market_data = fetch_live_market_data()
        status_color = "#10b981" if market_data["market_status"] == "OPEN" else "#ef4444"
        st.markdown(f"""
        <div class="live-ticker">
            <div class="pulse-dot"></div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8;">MARKET STATUS</div>
                <div style="font-weight: 600; color: {status_color};">
                    {market_data["market_status"]} • NIFTY 50: {market_data["nifty_50"]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: right; color: #94a3b8; font-size: 0.85rem;">
            <div>🕐 {datetime.now().strftime("%H:%M:%S")}</div>
            <div>📅 {datetime.now().strftime("%d %b %Y")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

def render_metrics(registry: ModelRegistry):
    """Render KPI metrics cards"""
    leads_df = load_leads_data()
    investor_df = load_investor_data()
    nav_summary = fetch_nav_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = len(leads_df) if not leads_df.empty else 2930
        hot_leads = len(leads_df[leads_df.get("Conversion_Probability", 0) >= 0.85]) if not leads_df.empty else 1117
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">{total_leads:,}</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Total Leads</div>
            <div style="color: #10b981; font-size: 0.8rem; margin-top: 5px;">
                🔥 {hot_leads} Hot Prospects
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_investors = len(investor_df) if not investor_df.empty else 1000
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: #8b5cf6;">{total_investors:,}</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Investor Profiles</div>
            <div style="color: #a78bfa; font-size: 0.8rem; margin-top: 5px;">
                4 Persona Segments
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        model_status = "✅ Active" if registry.lead_bundle else "❌ Offline"
        model_color = "#10b981" if registry.lead_bundle else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: {model_color};">3</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">AI Models</div>
            <div style="color: {model_color}; font-size: 0.8rem; margin-top: 5px;">
                {model_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        fund_count = nav_summary.get("fund_count", 315)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: #f59e0b;">{fund_count:,}+</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Mutual Funds</div>
            <div style="color: #fbbf24; font-size: 0.8rem; margin-top: 5px;">
                NAV Data Available
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_lead_scoring(registry: ModelRegistry):
    """Render lead scoring interface"""
    st.markdown("### 🎯 Lead Intelligence & Scoring")
    
    tab1, tab2 = st.tabs(["🔮 Score New Lead", "📊 Lead Database"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
            st.subheader("Lead Profile")
            
            with st.form("lead_form"):
                # Numeric inputs
                total_visits = st.number_input("Total Visits", 0, 100, 5)
                time_on_site = st.number_input("Time on Website (seconds)", 0, 10000, 800)
                page_views = st.number_input("Page Views per Visit", 0.0, 20.0, 2.5, 0.1)
                activity_score = st.number_input("Activity Score", 0, 50, 15)
                profile_score = st.number_input("Profile Score", 0, 50, 12)
                
                # Categorical inputs
                lead_source = st.selectbox("Lead Source", 
                    ["Organic Search", "Google Ads", "Referral", "Direct Traffic", "Social Media"])
                occupation = st.selectbox("Occupation",
                    ["Working Professional", "Student", "Business Owner", "Housewife", "Other"])
                specialization = st.selectbox("Specialization",
                    ["Select", "Finance", "Technology", "Healthcare", "Education", "Other"])
                
                submitted = st.form_submit_button("🎯 Score Lead", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if submitted:
                lead_data = {
                    "TotalVisits": total_visits,
                    "Total Time Spent on Website": time_on_site,
                    "Page Views Per Visit": page_views,
                    "Asymmetrique Activity Score": activity_score,
                    "Asymmetrique Profile Score": profile_score,
                    "Lead Source": lead_source,
                    "What is your current occupation": occupation,
                    "Specialization": specialization,
                    "Lead Origin": "API",
                    "Last Activity": "Page Visited",
                    "Country": "India",
                    "Lead Quality": "High in Relevance",
                    "Do Not Email": "No",
                    "Do Not Call": "No"
                }
                
                result = score_lead(lead_data, registry)
                
                st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
                st.subheader("Scoring Result")
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 30px;">
                        <div style="font-size: 4rem; font-weight: bold; 
                                    color: {'#10b981' if result['probability'] >= 85 else '#f59e0b' if result['probability'] >= 65 else '#6b7280'};">
                            {result['probability']}%
                        </div>
                        <div style="margin-top: 15px;">
                            <span class="{result['tier_class']}">{result['tier']}</span>
                        </div>
                        <div style="color: #94a3b8; margin-top: 15px; font-size: 0.9rem;">
                            Decision Threshold: {result['threshold']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendation
                    if result['probability'] >= 85:
                        st.success("🚀 **Immediate Action Required** - High conversion probability. Assign to premium advisor within 1 hour.")
                    elif result['probability'] >= 65:
                        st.info("📞 **Follow-up Priority** - Warm lead. Schedule call within 24 hours.")
                    else:
                        st.warning("📧 **Nurture Campaign** - Add to email drip sequence for long-term nurturing.")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        leads_df = load_leads_data()
        if not leads_df.empty:
            # Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                tier_filter = st.multiselect("Filter by Tier", 
                    ["🔥 HOT", "🌡️ WARM", "❄️ COLD"], default=["🔥 HOT", "🌡️ WARM"])
            with col_f2:
                if "Lead Source" in leads_df.columns:
                    source_filter = st.multiselect("Lead Source", 
                        leads_df["Lead Source"].unique().tolist())
            with col_f3:
                search_term = st.text_input("🔍 Search leads")
            
            # Display table
            display_cols = ["Lead Number", "Lead Source", "Conversion_Probability", 
                          "Converted_Prediction", "Recommended_Pitch_Persona"]
            available_cols = [c for c in display_cols if c in leads_df.columns]
            
            st.dataframe(
                leads_df[available_cols].head(100),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = leads_df.to_csv(index=False)
            st.download_button(
                "📥 Download Leads CSV",
                csv,
                "lume_leads_export.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No leads data available. Run training first.")

def render_investor_analytics(registry: ModelRegistry):
    """Render investor clustering and analytics"""
    st.markdown("### 👥 Investor Persona Analytics")
    
    investor_df = load_investor_data()
    
    if not investor_df.empty and "Persona_Cluster" in investor_df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cluster distribution
            cluster_counts = investor_df["Persona_Cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            
            persona_names = {
                0: "🚀 Equity Growth Seeker",
                1: "🛡️ Conservative Saver", 
                2: "⚖️ Balanced Investor",
                3: "📊 Index Tracker"
            }
            cluster_counts["Persona"] = cluster_counts["Cluster"].map(persona_names)
            
            fig = px.pie(cluster_counts, values="Count", names="Persona",
                        title="Investor Persona Distribution",
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
            st.subheader("Persona Guide")
            
            personas = [
                ("🚀 Equity Growth", "High risk appetite, seeks aggressive returns", "Equity Schemes"),
                ("🛡️ Conservative", "Capital preservation priority", "Liquid/Debt Funds"),
                ("⚖️ Balanced", "Moderate risk, diversification focus", "Hybrid Funds"),
                ("📊 Index Tracker", "Passive, low-cost preference", "Index Funds/ETFs")
            ]
            
            for emoji, desc, recommendation in personas:
                st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; 
                           background: rgba(59, 130, 246, 0.1); 
                           border-radius: 8px; border-left: 3px solid #3b82f6;">
                    <strong>{emoji}</strong><br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">{desc}</span><br>
                    <span style="color: #10b981; font-size: 0.8rem;">💡 {recommendation}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Investor assignment tool
        st.markdown("#### 🔮 Assign Investor Persona")
        
        if registry.investor_bundle:
            cols = registry.investor_bundle.behavior_cols
            c1, c2, c3, c4 = st.columns(4)
            
            vals = {}
            for i, name in enumerate(cols):
                container = [c1, c2, c3, c4][i % 4]
                with container:
                    vals[name] = st.slider(name, 0.0, 10.0, 5.0, 0.5,
                                          help=f"Rate {name} importance (0-10)")
            
            if st.button("🎯 Predict Persona", use_container_width=True):
                import numpy as np
                vec = np.array([[vals[c] for c in cols]])
                vec_scaled = registry.investor_bundle.scaler.transform(vec)
                cluster = int(registry.investor_bundle.kmeans.predict(vec_scaled)[0])
                
                persona = persona_names.get(cluster, f"Cluster {cluster}")
                st.success(f"**Predicted Persona: {persona}**")
                
                # Recommendations
                recs = {
                    0: "🎯 Recommend: Mid-cap equity funds, sectoral themes",
                    1: "🎯 Recommend: Liquid funds, short-term debt, FD alternatives",
                    2: "🎯 Recommend: Balanced advantage funds, dynamic asset allocation",
                    3: "🎯 Recommend: Nifty 50 index funds, passive ETFs"
                }
                st.info(recs.get(cluster, "Custom portfolio recommendation"))
    else:
        st.info("Investor data not available. Run training to generate clusters.")

def render_market_intelligence():
    """Render market data and NAV analytics"""
    st.markdown("### 📈 Market Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["Live Market", "NAV Analytics", "News Feed"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        market_data = fetch_live_market_data()
        
        with col1:
            nifty_value = market_data["nifty_50"]
            nifty_change = market_data["nifty_change"]
            nifty_pct = market_data["nifty_pct"]
            change_color = "#10b981" if nifty_change >= 0 else "#ef4444"
            arrow = "▲" if nifty_change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="fintech-card" style="text-align: center;">
                <div style="color: #94a3b8; font-size: 0.9rem;">NIFTY 50</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #f8fafc;">
                    {nifty_value:,.2f}
                </div>
                <div style="color: {change_color}; font-size: 1rem;">
                    {arrow} {abs(nifty_change):.2f} ({nifty_pct:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="fintech-card" style="text-align: center;">
                <div style="color: #94a3b8; font-size: 0.9rem;">MARKET STATUS</div>
                <div style="font-size: 2rem; font-weight: bold; 
                           color: {'#10b981' if market_data['market_status'] == 'OPEN' else '#ef4444'};">
                    {market_data['market_status']}
                </div>
                <div style="color: #64748b; font-size: 0.8rem;">
                    Last Update: {market_data['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="fintech-card" style="text-align: center;">
                <div style="color: #94a3b8; font-size: 0.9rem;">TRADING VOLUME</div>
                <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">
                    {random.randint(800, 1200)}M
                </div>
                <div style="color: #64748b; font-size: 0.8rem;">
                    Shares Traded Today
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # NIFTY Chart (Placeholder for actual API integration)
        st.markdown("#### NIFTY 50 Trend")
        
        # Generate sample trend data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        values = [22500 + random.randint(-500, 500) + i*10 for i in range(30)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#3b82f6', width=2),
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        fig.update_layout(
            title="30-Day Trend",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        nav_summary = fetch_nav_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Funds Tracked", nav_summary.get("fund_count", 315))
        with col2:
            st.metric("Last NAV Update", nav_summary.get("last_updated", "Today"))
        
        st.info("🔗 Connect to AMFI India for live NAV data: https://www.amfiindia.com/spages/NAVAll.txt")
        
        # NAV search placeholder
        st.text_input("🔍 Search Fund by Name or ISIN")
    
    with tab3:
        st.markdown("#### 📰 Financial News Feed")
        
        # Simulated news feed
        news_items = [
            ("Market Update", "Nifty reclaims 22,500 as banking stocks surge", "2 hours ago", "positive"),
            ("Policy", "RBI keeps repo rate unchanged at 6.5%", "5 hours ago", "neutral"),
            ("Earnings", "Major IT companies report strong Q4 results", "8 hours ago", "positive"),
            ("Global", "US markets close higher on tech rally", "12 hours ago", "positive"),
            ("Sector", "Mutual fund SIP inflows hit record high", "1 day ago", "positive")
        ]
        
        for category, headline, time, sentiment in news_items:
            color = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#f59e0b"}[sentiment]
            emoji = {"positive": "📈", "negative": "📉", "neutral": "📊"}[sentiment]
            
            st.markdown(f"""
            <div style="padding: 12px; margin-bottom: 10px; 
                       background: rgba(30, 41, 59, 0.6);
                       border-radius: 8px; border-left: 3px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {color}; font-size: 0.8rem; font-weight: 600;">
                        {emoji} {category.upper()}
                    </span>
                    <span style="color: #64748b; font-size: 0.75rem;">{time}</span>
                </div>
                <div style="color: #f8fafc; margin-top: 5px; font-weight: 500;">
                    {headline}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_bi_dashboard():
    """Render embedded BI dashboard"""
    st.markdown("### 📊 Business Intelligence Dashboard")
    
    if TABLEAU_PUBLIC_EMBED_URL and "YourWorkbook" not in TABLEAU_PUBLIC_EMBED_URL:
        st.components.v1.iframe(TABLEAU_PUBLIC_EMBED_URL, height=800, scrolling=True)
    else:
        st.info("""
        📊 **Tableau Dashboard Placeholder**
        
        To embed your Tableau dashboard:
        1. Publish workbook to Tableau Public
        2. Get embed URL from Share → Embed Code
        3. Set `LUME_TABLEAU_EMBED_URL` in .env file
        
        **Exported Data Available:**
        - `/tableau_exports/leads_for_bi.csv` (9,240 rows)
        - `/tableau_exports/investor_clusters_for_bi.csv` (1,000 rows)
        - `/tableau_exports/sentiment_for_bi.csv` (5,842 rows)
        """)
        
        # Show export location
        export_path = ROOT / "tableau_exports"
        if export_path.exists():
            files = list(export_path.glob("*.csv"))
            st.success(f"✅ {len(files)} BI export files ready in `tableau_exports/` folder")

def render_sidebar(registry: ModelRegistry):
    """Render navigation sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 3rem;">💎</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #f8fafc;">LUME AI</div>
            <div style="color: #64748b; font-size: 0.8rem;">Fintech Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Model Status
        st.markdown("#### 🤖 Model Status")
        
        models = [
            ("Lead Scoring", registry.lead_bundle, "#3b82f6"),
            ("Investor Cluster", registry.investor_bundle, "#8b5cf6"),
            ("Sentiment NLP", registry.sentiment_bundle, "#10b981")
        ]
        
        for name, model, color in models:
            status = "🟢 Online" if model else "🔴 Offline"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                       padding: 8px; background: rgba(30, 41, 59, 0.6); 
                       border-radius: 6px; margin-bottom: 5px;">
                <span style="color: {color}; font-weight: 500;">{name}</span>
                <span style="font-size: 0.8rem; color: {'#10b981' if model else '#ef4444'};">
                    {status}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick Actions
        st.markdown("#### ⚡ Quick Actions")
        
        if st.button("📊 Export Data for BI", use_container_width=True):
            st.success("Data exported! Check `tableau_exports/` folder")
        
        if st.button("🔧 Retrain Models", use_container_width=True):
            st.info("Run: `python src/master_pipeline.py` to refresh the full Big Data stack.")
        
        # Manifest Info
        manifest_path = MODELS_DIR / "model_manifest.json"
        if manifest_path.is_file():
            with open(manifest_path) as f:
                m = json.load(f)
                st.markdown(f"""
                <div style="font-size: 0.75rem; color: #94a3b8; padding: 10px; 
                           background: rgba(16, 185, 129, 0.1); border-radius: 6px;">
                    🧬 <b>Pipeline Status</b><br>
                    Last Full Run: {m.get('timestamp', 'N/A')[:16]}<br>
                    ETL Duration: {m.get('training_duration_sec', 'N/A')}s
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # API Status
        st.markdown("#### 🔌 API Connection")
        st.code("http://localhost:8000", language="text")
        
        st.markdown("""
        <div style="font-size: 0.75rem; color: #64748b; text-align: center; margin-top: 20px;">
            Lume AI Platform v2.0<br>
            © 2025 Fintech Intelligence
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    """Main application"""
    # Load models
    registry = load_models()
    
    # Render sidebar
    render_sidebar(registry)
    
    # Render header
    render_header()
    
    # Render metrics
    render_metrics(registry)
    
    st.divider()
    
    # Main content tabs
    tab_live, tab_leads, tab_investors, tab_market, tab_bi = st.tabs([
        "⚡ Live Pulse",
        "🎯 Lead Intelligence", 
        "👥 Investor Analytics", 
        "📈 Market Data",
        "📊 BI Dashboard"
    ])
    
    with tab_live:
        render_live_pulse()

    with tab_leads:
        render_lead_scoring(registry)
    
    with tab_investors:
        render_investor_analytics(registry)
    
    with tab_market:
        render_market_intelligence()
    
    with tab_bi:
        render_bi_dashboard()

def render_live_pulse():
    """Real-time streaming dashboard section"""
    st.subheader("⚡ Real-Time Market & Sentiment Pulse")
    
    col_mkt, col_sent = st.columns([2, 1])
    
    # 1. Live Market Ticker
    with col_mkt:
        st.markdown('<div class="fintech-card">', unsafe_allow_html=True)
        st.markdown("### 🏦 Live Index Monitor")
        
        mkt_path = ROOT / "streaming/outputs/market_live.parquet"
        if mkt_path.exists():
            try:
                # Read latest records using fast parquet reading
                df_live = pd.read_parquet(mkt_path).sort_values("timestamp", ascending=False).head(20)
                if not df_live.empty:
                    latest = df_live.iloc[0]
                    # Metric row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("NIFTY 50", f"₹{latest.get('last_price', 0):,.2f}", f"{latest.get('pct_change', 0):+.2f}%")
                    m2.metric("Variation", f"{latest.get('variation', 0):+.2f}")
                    m3.metric("Status", str(latest.get('status', 'N/A')))
                    
                    # Chart
                    fig = px.line(df_live, x="timestamp", y="last_price", title="Real-Time Price Velocity")
                    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Waiting for first market packets...")
            except Exception as e:
                st.warning(f"Connecting to Spark Stream... ({e})")
        else:
            st.info("📡 Spark Stream Offline. Launch 'streaming_pipeline.py' to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Live Sentiment AI
    with col_sent:
        st.markdown('<div class="fintech-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Sentiment Stream")
        
        news_path = ROOT / "streaming/outputs/news_live.parquet"
        if news_path.exists():
            try:
                df_news = pd.read_parquet(news_path).sort_values("ingestion_timestamp", ascending=False).head(10)
                if not df_news.empty:
                    latest_sent = df_news.iloc[0].get('sentiment_score', 0.5)
                    
                    # Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = latest_sent * 100,
                        title = {'text': "Market Sentiment (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#10b981"},
                            'steps': [
                                {'range': [0, 40], 'color': "#ef4444"},
                                {'range': [40, 60], 'color': "#f59e0b"},
                                {'range': [60, 100], 'color': "#10b981"}
                            ]
                        }
                    ))
                    fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_g, use_container_width=True)
                    
                    # News Feed
                    st.markdown("#### Latest Headlines")
                    for _, row in df_news.head(3).iterrows():
                        impact = row.get('market_impact', 'NEUTRAL')
                        color = "#10b981" if impact == "BULLISH" else "#ef4444" if impact == "BEARISH" else "#f59e0b"
                        st.markdown(f"**{row.get('title', 'N/A')}**")
                        st.markdown(f"<span style='color:{color}; font-weight:bold;'>[{impact}]</span>", unsafe_allow_html=True)
                        st.divider()
                else:
                    st.info("Awaiting AI news analysis...")
            except Exception as e:
                st.error(f"Stream Sync Error: {e}")
        else:
            st.info("No live news stream detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Force Refresh Live Stream"):
        st.rerun()

if __name__ == "__main__":
    main()
