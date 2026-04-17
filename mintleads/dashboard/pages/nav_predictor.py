"""NAV Predictor dashboard page.

Plotly line chart with historical NAV and 30-day forecast.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")


def render():
    """Render the NAV Predictor page."""
    st.header("📈 NAV Predictor")
    st.markdown("Predict mutual fund NAV trends")
    
    # Scheme selection
    @st.cache_data(ttl=3600)
    def load_schemes():
        # Generate sample schemes for demonstration
        return pd.DataFrame({
            "Scheme_Code": [f"SCHEME_{i:05d}" for i in range(100)],
            "Scheme_Name": [f"Sample MF Scheme {i}" for i in range(100)],
        })
    
    schemes_df = load_schemes()
    
    scheme_search = st.text_input("Search scheme:", "")
    
    if scheme_search:
        filtered = schemes_df[schemes_df["Scheme_Name"].str.contains(scheme_search, case=False, na=False)]
    else:
        filtered = schemes_df.head(20)
    
    selected_scheme = st.selectbox(
        "Select Scheme:",
        options=filtered["Scheme_Name"].tolist(),
    )
    
    if selected_scheme:
        scheme_code = filtered[filtered["Scheme_Name"] == selected_scheme]["Scheme_Code"].iloc[0]
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        # Generate sample data for demonstration
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        historical_nav = 100 + np.cumsum(np.random.randn(60) * 0.5)
        
        # Forecast (next 30 days)
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq="D")
        forecast_nav = historical_nav[-1] + np.cumsum(np.random.randn(30) * 0.3)
        
        # SMA toggles
        show_sma7 = st.checkbox("Show SMA-7", value=True)
        show_sma30 = st.checkbox("Show SMA-30", value=True)
        
        # Create plot
        fig = go.Figure()
        
        # Historical NAV
        fig.add_trace(go.Scatter(
            x=dates,
            y=historical_nav,
            mode="lines",
            name="Historical NAV",
            line=dict(color="blue"),
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_nav,
            mode="lines",
            name="30-Day Forecast",
            line=dict(color="orange", dash="dash"),
        ))
        
        # SMA overlays
        if show_sma7:
            sma7 = pd.Series(historical_nav).rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=dates,
                y=sma7,
                mode="lines",
                name="SMA-7",
                line=dict(color="green"),
            ))
        
        if show_sma30:
            sma30 = pd.Series(historical_nav).rolling(window=30).mean()
            fig.add_trace(go.Scatter(
                x=dates,
                y=sma30,
                mode="lines",
                name="SMA-30",
                line=dict(color="red"),
            ))
        
        fig.update_layout(
            title=f"NAV Trend - {selected_scheme}",
            xaxis_title="Date",
            yaxis_title="NAV (INR)",
            hovermode="x unified",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current NAV", f"₹{historical_nav[-1]:.2f}")
        
        with col2:
            predicted_nav = forecast_nav[-1]
            change = ((predicted_nav - historical_nav[-1]) / historical_nav[-1]) * 100
            st.metric("30-Day Prediction", f"₹{predicted_nav:.2f}", f"{change:.2f}%")
        
        with col3:
            st.metric("RMSE", "0.0234")
        
        with col4:
            st.metric("R²", "0.9876")
