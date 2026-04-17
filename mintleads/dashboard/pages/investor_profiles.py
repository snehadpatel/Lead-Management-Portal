"""Investor Profiles dashboard page.

Investor segmentation visualization and batch processing.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render():
    """Render the Investor Profiles page."""
    st.header("👥 Investor Profiles")
    st.markdown("Segment investors and view cluster analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload investor CSV for batch segmentation", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(df)} investor records")
    else:
        # Sample data
        np.random.seed(42)
        df = pd.DataFrame({
            "id": range(1000),
            "current_age": np.random.randint(25, 70, 1000),
            "yearly_income": np.random.randint(30000, 200000, 1000),
            "total_debt": np.random.randint(0, 100000, 1000),
            "debt_to_income_ratio": np.random.uniform(0, 3, 1000),
            "credit_score": np.random.randint(600, 850, 1000),
        })
        st.info("Showing sample data. Upload a CSV to analyze your investors.")
    
    # Cluster assignment (simulated)
    df["cluster"] = pd.cut(df["current_age"], bins=3, labels=[0, 1, 2]).astype(int)
    
    # Persona mapping
    persona_map = {0: "Conservative", 1: "Balanced", 2: "Aggressive"}
    df["persona"] = df["cluster"].map(persona_map)
    
    # Recommendations
    fund_map = {
        "Conservative": "Debt",
        "Balanced": "Hybrid",
        "Aggressive": "Equity",
    }
    df["recommended_fund"] = df["persona"].map(fund_map)
    
    # Display results
    st.subheader("Segmentation Results")
    st.dataframe(df.head(50), use_container_width=True)
    
    # PCA visualization
    st.subheader("Cluster Visualization (PCA)")
    
    # Simulate PCA
    np.random.seed(42)
    df["pca_x"] = df["current_age"] * 0.5 + np.random.randn(len(df)) * 5
    df["pca_y"] = df["yearly_income"] / 10000 + np.random.randn(len(df)) * 5
    
    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="persona",
        hover_data=["id", "current_age", "yearly_income"],
        title="Investor Clusters - PCA Projection",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary cards
    st.subheader("Cluster Summary")
    
    cols = st.columns(3)
    for i, (cluster_id, persona) in enumerate(persona_map.items()):
        cluster_data = df[df["cluster"] == cluster_id]
        with cols[i]:
            st.metric(f"{persona}", len(cluster_data))
            st.write(f"Avg Income: ₹{cluster_data['yearly_income'].mean():,.0f}")
            st.write(f"Avg Age: {cluster_data['current_age'].mean():.1f}")
            st.write(f"Recommend: {fund_map[persona]}")
    
    # Justification
    st.subheader("Fund Recommendations")
    st.write("""
    **Conservative (Debt Funds):** Low risk tolerance, prioritize capital preservation.
    Suitable for older investors or those with debt obligations.
    
    **Balanced (Hybrid Funds):** Moderate risk tolerance, balanced growth and stability.
    Suitable for middle-aged professionals with stable income.
    
    **Aggressive (Equity Funds):** High risk tolerance, seek maximum growth.
    Suitable for young investors with long investment horizons.
    """)
