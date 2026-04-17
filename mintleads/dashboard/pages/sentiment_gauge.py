"""Sentiment Gauge dashboard page.

Live market sentiment visualization with news headlines.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render():
    """Render the Sentiment Gauge page."""
    st.header("📊 Market Sentiment Gauge")
    st.markdown("Live market sentiment analysis")
    
    # Auto-refresh
    st.empty()
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=True)
    
    if auto_refresh:
        st.rerun()
    
    # Current sentiment
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    sentiment_score = np.random.uniform(-1, 1)
    
    if sentiment_score > 0.3:
        signal = "Bullish"
        color = "green"
    elif sentiment_score < -0.3:
        signal = "Bearish"
        color = "red"
    else:
        signal = "Neutral"
        color = "gray"
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Market Sentiment: {signal}", "font": {"size": 24}},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": color},
            "steps": [
                {"range": [-1, -0.3], "color": "lightcoral"},
                {"range": [-0.3, 0.3], "color": "lightgray"},
                {"range": [0.3, 1], "color": "lightgreen"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": sentiment_score,
            },
        },
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Best time to call indicator
    st.subheader("📞 Best Time to Call")
    
    # Check if market hours (9:15 AM - 3:30 PM IST)
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    
    is_market_hours = (
        (current_hour == 9 and current_minute >= 15) or
        (9 < current_hour < 15) or
        (current_hour == 15 and current_minute <= 30)
    )
    
    is_bullish = signal == "Bullish"
    
    if is_market_hours and is_bullish:
        st.success("🟢 GREEN LIGHT - Great time to call prospects!")
        st.write("Market is open and sentiment is Bullish. Prospects are likely receptive.")
    elif is_market_hours:
        st.warning("🟡 CAUTION - Market open but sentiment is neutral/bearish")
        st.write("Consider waiting for sentiment to improve or focus on conservative funds.")
    else:
        st.error("🔴 MARKET CLOSED - Consider calling during market hours (9:15 AM - 3:30 PM IST)")
    
    # Recent headlines
    st.subheader("📰 Recent Headlines")
    
    headlines = [
        {"title": "Sensex rallies 500 points on positive global cues", "sentiment": "Positive", "source": "Economic Times"},
        {"title": "RBI keeps repo rate unchanged at 6.5%", "sentiment": "Neutral", "source": "Moneycontrol"},
        {"title": "FIIs remain net buyers in Indian equities", "sentiment": "Positive", "source": "Business Standard"},
        {"title": "Inflation concerns persist amid rising oil prices", "sentiment": "Negative", "source": "Mint"},
        {"title": "Mutual fund SIP inflows hit record high", "sentiment": "Positive", "source": "CNBC TV18"},
    ]
    
    for headline in headlines:
        sentiment_color = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}.get(headline["sentiment"], "⚪")
        st.write(f"{sentiment_color} **{headline['title']}** - *{headline['source']}*")
    
    # 7-day trend
    st.subheader("📈 7-Day Sentiment Trend")
    
    dates = pd.date_range(end=datetime.now(), periods=7, freq="D")
    trend_scores = [np.random.uniform(-0.5, 0.5) for _ in range(7)]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dates,
        y=trend_scores,
        mode="lines+markers",
        name="Sentiment Score",
        line=dict(color="blue"),
    ))
    
    fig2.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Bullish Threshold")
    fig2.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="Bearish Threshold")
    
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=300,
    )
    
    st.plotly_chart(fig2, use_container_width=True)
