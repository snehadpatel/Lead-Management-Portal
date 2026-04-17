"""Hot Leads dashboard page.

Table showing top leads sorted by conversion probability.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")


def render():
    """Render the Hot Leads page."""
    st.header("🔥 Hot Leads")
    st.markdown("Top leads ranked by conversion probability")
    
    # Load leads data
    @st.cache_data(ttl=300)
    def load_leads():
        # Try to load from processed data
        data_path = "data/processed/leads_processed.csv"
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        
        # Generate sample data for demonstration
        return pd.DataFrame({
            "Prospect ID": [f"LEAD_{i:05d}" for i in range(50)],
            "Lead Origin": ["API", "Landing Page", "Organic Search"] * 17,
            "Lead Source": ["Google", "Direct", "Referral", "Social"] * 13,
            "Country": ["India"] * 50,
            "Occupation": ["Working Professional", "Unemployed", "Student"] * 17,
            "TotalVisits": range(1, 51),
            "conversion_probability": [0.95 - (i * 0.01) for i in range(50)],
        })
    
    df = load_leads()
    
    # Add conversion probability if not present
    if "conversion_probability" not in df.columns:
        df["conversion_probability"] = [0.9 - (i * 0.015) for i in range(len(df))]
    
    # Determine tier
    def get_tier(prob):
        if prob > 0.85:
            return "🔴 Hot"
        elif prob > 0.65:
            return "🟠 Warm"
        else:
            return "🟢 Cold"
    
    df["Tier"] = df["conversion_probability"].apply(get_tier)
    
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    lead_sources = st.sidebar.multiselect(
        "Lead Source",
        options=df.get("Lead Source", pd.Series(["All"])).unique().tolist(),
        default=[],
    )
    
    lead_origins = st.sidebar.multiselect(
        "Lead Origin",
        options=df.get("Lead Origin", pd.Series(["All"])).unique().tolist(),
        default=[],
    )
    
    countries = st.sidebar.multiselect(
        "Country",
        options=df.get("Country", pd.Series(["India"])).unique().tolist(),
        default=[],
    )
    
    # Apply filters
    filtered_df = df.copy()
    if lead_sources:
        filtered_df = filtered_df[filtered_df["Lead Source"].isin(lead_sources)]
    if lead_origins:
        filtered_df = filtered_df[filtered_df["Lead Origin"].isin(lead_origins)]
    if countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
    
    # Sort by conversion probability
    filtered_df = filtered_df.sort_values("conversion_probability", ascending=False)
    
    # Display top 50
    display_df = filtered_df.head(50).copy()
    
    # Color coding
    def highlight_tiers(row):
        prob = row["conversion_probability"]
        if prob > 0.85:
            return ["background-color: #ffcccc"] * len(row)
        elif prob > 0.65:
            return ["background-color: #ffe6cc"] * len(row)
        else:
            return ["background-color: #ccffcc"] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_tiers, axis=1),
        use_container_width=True,
    )
    
    # Export button
    if st.button("Export to CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="hot_leads.csv",
            mime="text/csv",
        )
    
    # Lead detail expansion
    st.subheader("Lead Details")
    selected_lead = st.selectbox(
        "Select a lead to view details:",
        options=display_df["Prospect ID"].tolist(),
    )
    
    if selected_lead:
        lead_data = display_df[display_df["Prospect ID"] == selected_lead].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conversion Probability", f"{lead_data['conversion_probability']:.2%}")
        
        with col2:
            st.metric("Tier", lead_data["Tier"])
        
        with col3:
            st.metric("Visits", int(lead_data.get("TotalVisits", 0)))
        
        # Call script generation
        if st.button("Generate Call Script"):
            st.info("📞 **Suggested Call Script:**")
            st.write(f"""
            Hello, this is [Your Name] from [Your Company]. 
            I noticed you've shown interest in our mutual fund offerings. 
            Based on your profile as a {lead_data.get('Occupation', 'professional')} with 
            {int(lead_data.get('TotalVisits', 0))} visits to our platform, 
            I'd like to discuss investment options tailored to your needs.
            """)
