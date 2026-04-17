"""MintLeads Streamlit Dashboard.

Main entry point for the MintLeads dashboard application.
Provides navigation between different dashboard pages.
"""

import streamlit as st

st.set_page_config(
    page_title="MintLeads - AI-Powered Lead Recommendation",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💰 MintLeads Dashboard")
st.markdown("### AI-Powered Lead Recommendation System for Mutual Fund Distributors")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Hot Leads", "NAV Predictor", "Investor Profiles", "Sentiment Gauge"],
)

if page == "Hot Leads":
    from pages import hot_leads
    hot_leads.render()
elif page == "NAV Predictor":
    from pages import nav_predictor
    nav_predictor.render()
elif page == "Investor Profiles":
    from pages import investor_profiles
    investor_profiles.render()
elif page == "Sentiment Gauge":
    from pages import sentiment_gauge
    sentiment_gauge.render()

st.sidebar.markdown("---")
st.sidebar.info("MintLeads v1.0 - Built for MFDs")
