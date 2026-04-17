"""
💎 LUME AI — DUAL-ROLE FINTECH PLATFORM
Users: Funds Distributors & Investors
Features: Persona surveys, role-based UI, AI buddy, live fund tracking
"""

from __future__ import annotations

import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, asdict
import random

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit.components.v1 import html

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lume_platform.config import DATA_ROOT, EXPORT_DIR
from lume_platform.inference.registry import ModelRegistry

# Page config
st.set_page_config(
    page_title="Lume AI | Wealth Management Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    :root {
        --primary: #1e3a8a;
        --accent: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
    }
    
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    
    .role-badge-distributor {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .role-badge-investor {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .persona-card {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 24px;
        border: 2px solid;
        margin-bottom: 20px;
    }
    
    .persona-growth { border-color: #10b981; }
    .persona-conservative { border-color: #3b82f6; }
    .persona-balanced { border-color: #f59e0b; }
    .persona-passive { border-color: #8b5cf6; }
    
    .lead-hot {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.05));
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .lead-warm {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.05));
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .lead-cold {
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.2), rgba(107, 114, 128, 0.05));
        border-left: 4px solid #6b7280;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .ai-buddy-message {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    .fund-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    
    .fund-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent), #3b82f6);
        opacity: 0.3;
    }
    
    .fund-card:hover {
        transform: translateY(-8px) scale(1.01);
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5), 
                    0 0 20px -5px rgba(59, 130, 246, 0.2);
    }
    
    .risk-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .match-glow {
        position: absolute;
        top: -50px;
        right: -50px;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0) 70%);
        pointer-events: none;
    }
    
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.65rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 4px;
        letter-spacing: 1px;
    }
    
    .ai-momentum-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES & PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    user_id: str
    role: Literal["distributor", "investor"]
    name: str
    email: str
    persona: str  # growth, conservative, balanced, passive
    experience_level: Literal["beginner", "intermediate", "expert"]
    created_at: str
    survey_responses: Dict[str, Any]

@dataclass
class Fund:
    fund_id: str
    name: str
    category: str  # equity, debt, hybrid, index
    risk_level: Literal["low", "medium", "high"]
    returns_1y: float
    returns_3y: float
    returns_5y: float
    aum: float  # Assets Under Management in Cr
    expense_ratio: float
    rating: int  # 1-5
    recommended_for: List[str]  # personas

# Persona definitions
PERSONAS = {
    "growth": {
        "name": "🚀 Growth Seeker",
        "description": "High risk appetite, seeks aggressive returns",
        "risk_tolerance": "High",
        "investment_horizon": "5+ years",
        "recommended_funds": ["Mid Cap", "Small Cap", "Sectoral/Thematic"],
        "color": "#10b981",
        "icon": "📈"
    },
    "conservative": {
        "name": "🛡️ Wealth Preserver",
        "description": "Capital protection priority, steady income",
        "risk_tolerance": "Low",
        "investment_horizon": "1-3 years",
        "recommended_funds": ["Liquid", "Short Duration", "Corporate Bond"],
        "color": "#3b82f6",
        "icon": "🛡️"
    },
    "balanced": {
        "name": "⚖️ Balanced Allocator",
        "description": "Moderate risk, seeks diversification",
        "risk_tolerance": "Medium",
        "investment_horizon": "3-5 years",
        "recommended_funds": ["Balanced Advantage", "Aggressive Hybrid", "Dynamic Asset"],
        "color": "#f59e0b",
        "icon": "⚖️"
    },
    "passive": {
        "name": "📊 Index Tracker",
        "description": "Low-cost preference, market-matching returns",
        "risk_tolerance": "Medium",
        "investment_horizon": "5+ years",
        "recommended_funds": ["Index Funds", "ETFs", "FoF"],
        "color": "#8b5cf6",
        "icon": "📊"
    }
}

# Sample funds database with history mapping
FUNDS_DB = [
    Fund("F001", "Nippon India Small Cap Fund", "equity", "high", 45.2, 28.5, 22.1, 28500, 0.67, 5, ["growth"]),
    Fund("F002", "SBI Magnum MidCap Fund", "equity", "high", 38.5, 24.2, 19.8, 12800, 0.84, 4, ["growth"]),
    Fund("F003", "ICICI Pru Technology Fund", "equity", "high", 52.1, 32.8, 25.4, 8900, 0.92, 4, ["growth"]),
    Fund("F004", "HDFC Liquid Fund", "debt", "low", 6.8, 6.5, 6.9, 45000, 0.20, 5, ["conservative"]),
    Fund("F005", "SBI Savings Fund", "debt", "low", 7.2, 6.8, 7.1, 22000, 0.35, 4, ["conservative"]),
    Fund("F006", "Axis Short Term Fund", "debt", "low", 7.8, 7.2, 7.5, 15600, 0.42, 4, ["conservative"]),
    Fund("F007", "Edelweiss Balanced Advantage", "hybrid", "medium", 18.5, 14.2, 12.8, 18500, 0.78, 5, ["balanced"]),
    Fund("F008", "HDFC Hybrid Equity", "hybrid", "medium", 22.3, 16.8, 14.5, 15600, 0.92, 4, ["balanced"]),
    Fund("F009", "Kotak Balanced Advantage", "hybrid", "medium", 17.8, 13.5, 12.1, 12400, 0.72, 4, ["balanced"]),
    Fund("F010", "UTI Nifty 50 Index Fund", "index", "medium", 18.2, 15.1, 13.8, 8500, 0.18, 4, ["passive"]),
    Fund("F011", "HDFC Index Nifty 50", "index", "medium", 18.5, 15.3, 14.1, 6200, 0.20, 4, ["passive"]),
    Fund("F012", "SBI Nifty Next 50 Index", "index", "medium", 25.8, 19.2, 16.5, 4500, 0.33, 4, ["passive", "growth"]),
]

# Mapping to real history CSV files found in datasets
HISTORY_MAP = {
    "F001": "153703_history.csv",
    "F002": "152111_history.csv",
    "F003": "147441_history.csv",
    "F004": "111753_history.csv",
    "F005": "120437_history.csv",
    "F006": "150369_history.csv",
    "F007": "131054_history.csv",
    "F008": "131382_history.csv",
    "F009": "105658_history.csv",
    "F010": "147608_history.csv",
    "F011": "104485_history.csv",
    "F012": "139201_history.csv",
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROLE-SPECIFIC PERSONA ASSESSMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# INVESTOR PERSONA SURVEY - Multi-dimensional behavioral assessment for KMeans
INVESTOR_SURVEY = [
    {
        "id": "ProfManage",
        "question": "How important is professional management of funds to you?",
        "options": {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Extremely High": 5}
    },
    {
        "id": "Diversification",
        "question": "To what extent do you prefer spreading investments across multiple asset classes?",
        "options": {"Rarely": 1, "Seldom": 2, "Sometimes": 3, "Usually": 4, "Always": 5}
    },
    {
        "id": "Affordability",
        "question": "How comfortable are you with the initial investment amounts required?",
        "options": {"Uncomfortable": 1, "Slightly": 2, "Neutral": 3, "Comfortable": 4, "Very Comfortable": 5}
    },
    {
        "id": "Liquidity",
        "question": "How important is it for you to be able to withdraw your money quickly?",
        "options": {"Not Important": 1, "Slightly": 2, "Neutral": 3, "Important": 4, "Critical": 5}
    },
    {
        "id": "Growth",
        "question": "How much are you prioritizing aggressive capital appreciation over safety?",
        "options": {"Safety First": 1, "Mainly Safety": 2, "Balanced": 3, "Mainly Growth": 4, "Max Growth": 5}
    },
    {
        "id": "Trustworthiness",
        "question": "How much does the brand reputation/trust factor influence your choice?",
        "options": {"Not at all": 1, "A little": 2, "Neutral": 3, "Significantly": 4, "Everything": 5}
    },
    {
        "id": "Technology",
        "question": "How important are digital features and AI-driven insights to you?",
        "options": {"Prefer Manual": 1, "Minimal Tech": 2, "Neutral": 3, "Tech Oriented": 4, "Tech First": 5}
    }
]

# DISTRIBUTOR PERSONA SURVEY - Optimized for business style (mapped to same 7 features for demo consistency)
DISTRIBUTOR_SURVEY = INVESTOR_SURVEY  # Use matching dimensions for dual-role consistency


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "user_profile": None,
        "survey_step": 0,
        "survey_answers": {},
        "selected_lead": None,
        "selected_distributor": None,
        "chat_history": [],
        "show_ai_buddy": False,
        "portfolio": {},  # fund_id -> units
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ═══════════════════════════════════════════════════════════════════════════════
# PERSONA DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

def determine_persona(answers: Dict[str, Any], registry: Optional[ModelRegistry] = None) -> str:
    """Determine persona based on survey answers using KMeans clustering"""
    # 1. Fallback / Hardcoded matching if KMeans fails or registry missing
    def fallback_tally():
        scores = {"growth": 0, "conservative": 0, "balanced": 0, "passive": 0}
        for v in answers.values():
            if v in scores: scores[v] += 1
        return max(scores, key=scores.get)

    # 2. Dynamic KMeans Clustering
    if registry and registry.investor_bundle:
        try:
            # Ensure we have all 7 behavioral cols
            cols = registry.investor_bundle.behavior_cols
            input_row = {c: float(answers.get(c, 3)) for c in cols}
            
            cluster = registry.investor_bundle.predict_row(input_row)
            
            # Map cluster to human readable persona
            # Verified via cluster center analysis:
            # Cluster 3: High Growth (0.90), High ProfManage -> Growth
            # Cluster 0: Low Growth (0.08), High Liquidity -> Conservative
            # Cluster 1: Mid-High Growth (0.69), High Trust -> Balanced
            # Cluster 2: Mid Growth (0.48), Very High Tech -> Passive
            mapping = {
                3: "growth",
                0: "conservative",
                1: "balanced",
                2: "passive"
            }
            result = mapping.get(cluster, "balanced")
            print(f"DEBUG: KMeans Prediction - Input: {input_row}, Cluster: {cluster}, Result: {result}")
            return result
        except Exception as e:
            print(f"DEBUG: KMeans Error: {e}")
            st.error(f"KMeans Clustering Error: {e}")
            return fallback_tally()
    
    print("DEBUG: Registry or bundle missing")
    return fallback_tally()

# ═══════════════════════════════════════════════════════════════════════════════
# AI BUDDY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

AI_BUDDY_KNOWLEDGE = {
    "beginner": {
        "greeting": "👋 Welcome! I'm your AI Buddy. I'll guide you through everything!",
        "topics": {
            "getting_started": """
            ### 🎯 Getting Started as a New Distributor
            
            **Step 1: Understand Your Role**
            - You're a bridge between mutual fund houses and investors
            - Your job: Help investors choose the right funds
            - Earn commission on investments made through you
            
            **Step 2: Key Terms to Know**
            - **NAV**: Net Asset Value - price per unit of the fund
            - **AUM**: Assets Under Management - total money in the fund
            - **Expense Ratio**: Annual fee charged by the fund
            - **SIP**: Systematic Investment Plan - monthly investing
            - **SWP**: Systematic Withdrawal Plan - monthly withdrawals
            
            **Step 3: Registration Process**
            1. Complete NISM Series V-A certification
            2. Register with AMFI (Association of Mutual Funds in India)
            3. Get ARN (AMFI Registration Number)
            4. Tie up with AMCs (Asset Management Companies)
            """,
            "lead_management": """
            ### 📞 How to Handle Leads (Investors)
            
            **Hot Leads (🔥 Conversion >85%)**
            - Call within 1 hour
            - Prepare personalized fund recommendations
            - Have documents ready for immediate investment
            - Follow up same day
            
            **Warm Leads (🌡️ Conversion 65-85%)**
            - Call within 24 hours
            - Send educational content about investing
            - Schedule a meeting/demo
            - Weekly follow-up
            
            **Cold Leads (❄️ Conversion <65%)**
            - Add to email nurture sequence
            - Send market updates monthly
            - Don't push hard - build trust first
            - Re-engage after 3 months
            """,
            "fund_recommendation": """
            ### 💡 How to Recommend Funds
            
            **Understand Investor Profile:**
            1. Age and income level
            2. Investment horizon
            3. Risk appetite
            4. Financial goals
            
            **Matching Strategy:**
            - **Young Professional**: Equity funds (SIP mode)
            - **Retiree**: Debt funds or SWP from balanced funds
            - **Mid-career**: Hybrid funds for balanced growth
            - **Risk-averse**: Liquid or ultra-short funds
            
            **Never Do:**
            - Promise guaranteed returns
            - Recommend without understanding risk
            - Push one fund for everyone
            """,
            "compliance": """
            ### ⚖️ Compliance Rules (Must Follow)
            
            **Do's:**
            ✅ Complete KYC of every investor
            ✅ Disclose all commissions/fees
            ✅ Provide proper receipts
            ✅ Maintain records for 5+ years
            ✅ Explain risks clearly
            
            **Don'ts:**
            ❌ Promise guaranteed returns
            ❌ Invest without investor consent
            ❌ Hide charges or fees
            ❌ Recommend unsuitable products
            ❌ Share investor data
            
            **Key Regulations:**
            - SEBI (Investment Advisers) Regulations, 2013
            - AMFI Code of Conduct
            - KYC norms from CDSL/NSDL
            """
        }
    },
    "intermediate": {
        "greeting": "👋 Welcome back! Ready to boost your distribution business?",
        "topics": {
            "portfolio_analysis": "Advanced portfolio analysis techniques...",
            "client_retention": "Strategies for long-term client retention...",
            "cross_selling": "How to cross-sell different fund categories..."
        }
    },
    "expert": {
        "greeting": "👋 Welcome! Let's optimize your distribution network.",
        "topics": {
            "advanced_analytics": "Deep dive into fund analytics...",
            "business_scaling": "Scale your distribution business...",
            "team_management": "Build and manage a team of sub-brokers..."
        }
    }
}

def render_ai_buddy(profile: UserProfile):
    """Render AI buddy chat interface"""
    st.markdown("### 🤖 AI Buddy - Your Distribution Assistant")
    
    experience = profile.experience_level
    knowledge = AI_BUDDY_KNOWLEDGE.get(experience, AI_BUDDY_KNOWLEDGE["beginner"])
    
    # Chat interface
    st.markdown(f"<div class='ai-buddy-message'>{knowledge['greeting']}</div>", 
                unsafe_allow_html=True)
    
    # Topic selection
    st.markdown("#### 📚 What would you like to learn?")
    
    topics = list(knowledge["topics"].keys())
    topic_labels = {
        "getting_started": "🚀 Getting Started",
        "lead_management": "📞 Lead Management",
        "fund_recommendation": "💡 Fund Recommendation",
        "compliance": "⚖️ Compliance & Rules",
        "portfolio_analysis": "📊 Portfolio Analysis",
        "client_retention": "🤝 Client Retention",
        "cross_selling": "🔄 Cross-Selling",
        "advanced_analytics": "📈 Advanced Analytics",
        "business_scaling": "📢 Business Scaling",
        "team_management": "👥 Team Management"
    }
    
    cols = st.columns(min(len(topics), 3))
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            if st.button(topic_labels.get(topic, topic), key=f"topic_{topic}", 
                        use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": knowledge["topics"][topic],
                    "timestamp": datetime.now()
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 💬 Guidance")
        for msg in reversed(st.session_state.chat_history[-3:]):
            if msg["role"] == "assistant":
                st.markdown(f"<div class='ai-buddy-message'>{msg['content']}</div>", 
                          unsafe_allow_html=True)
    
    # Quick question input
    user_question = st.text_input("❓ Ask me anything about distribution...", 
                                   placeholder="e.g., How do I handle objections?")
    if user_question:
        # Simple response logic (can be enhanced with LLM)
        response = f"""
        **🤖 AI Buddy Response:**
        
        Based on your experience level ({experience}), here's guidance:
        
        **Question:** {user_question}
        
        **Quick Tips:**
        - Always listen first, understand the investor's concern
        - Use data to support your recommendations  
        - Follow up consistently without being pushy
        - Keep learning about new fund offerings
        
        *For detailed guidance, select a topic above or refer to your training materials.*
        """
        st.markdown(f"<div class='ai-buddy-message'>{response}</div>", 
                  unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTOR VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def render_distributor_dashboard(profile: UserProfile, registry: ModelRegistry):
    """Render distributor-specific dashboard"""
    
    # Header with role badge
    col1, col2 = st.columns([3, 1])
    with col1:
        persona = PERSONAS.get(profile.persona, PERSONAS["balanced"])
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px;">
            <h2 style="margin: 0;">Welcome, {profile.name}</h2>
            <span class="role-badge-distributor">👔 Distributor</span>
        </div>
        <p style="color: #64748b;">
            Persona: <span style="color: {persona['color']}; font-weight: 600;">
            {persona['name']}</span> • Experience: {profile.experience_level.title()}
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        if profile.experience_level == "beginner":
            if st.button("🤖 Ask AI Buddy", use_container_width=True, type="primary"):
                st.session_state.show_ai_buddy = not st.session_state.show_ai_buddy
    
    st.divider()
    
    # AI Buddy for beginners
    if profile.experience_level == "beginner" and st.session_state.show_ai_buddy:
        render_ai_buddy(profile)
        st.divider()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📞 My Leads (Investors)", 
        "💼 Fund Recommendations", 
        "📊 My Performance",
        "🔧 Tools & Resources"
    ])
    
    with tab1:
        render_distributor_leads(profile, registry)
    
    with tab2:
        render_distributor_funds(profile)
    
    with tab3:
        render_distributor_performance(profile)
    
    with tab4:
        render_distributor_tools(profile)

def render_distributor_leads(profile: UserProfile, registry: ModelRegistry):
    """Render lead management for distributors"""
    st.markdown("### 📞 Lead Management - Classify & Contact Investors")
    
    # Load leads data
    leads_df = load_leads_data()
    
    if leads_df.empty:
        # Generate sample leads if no data
        sample_leads = generate_sample_leads(10, profile.persona)
        leads_df = pd.DataFrame(sample_leads)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        tier_filter = st.multiselect("Filter by Temperature", 
            ["🔥 HOT (>85%)", "🌡️ WARM (65-85%)", "❄️ COLD (<65%)"],
            default=["🔥 HOT (>85%)", "🌡️ WARM (65-85%)"])
    with col2:
        persona_match = st.checkbox("Match to My Persona Only", value=True)
    with col3:
        search = st.text_input("🔍 Search leads")
    
    # Calculate lead scores and display
    st.markdown("#### 👥 Available Leads")
    
    # Display leads by tier
    for idx, lead in leads_df.head(10).iterrows():
        score = lead.get('Conversion_Probability', random.uniform(0.4, 0.95))
        
        # Get investor name and details
        first_name = lead.get('First Name', lead.get('First_Name', ''))
        last_name = lead.get('Last Name', lead.get('Last_Name', ''))
        if first_name or last_name:
            investor_name = f"{first_name} {last_name}".strip()
        else:
            # Generate realistic Indian names for demo
            first_names = ["Rajesh", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rahul", "Divya", "Karan", "Meera"]
            last_names = ["Sharma", "Patel", "Kumar", "Gupta", "Singh", "Verma", "Reddy", "Iyer", "Joshi", "Shah"]
            investor_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Additional investor details
        occupation = lead.get('What is your current occupation', lead.get('Occupation', 'Working Professional'))
        city = lead.get('City', lead.get('Location', random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'])))
        investment_amount = lead.get('Investment Amount', f"₹{random.randint(2, 50)}L")
        
        if score >= 0.85:
            tier_class = "lead-hot"
            tier_label = "🔥 HOT"
            action = "Call NOW"
        elif score >= 0.65:
            tier_class = "lead-warm"
            tier_label = "🌡️ WARM"
            action = "Follow up today"
        else:
            tier_class = "lead-cold"
            tier_label = "❄️ COLD"
            action = "Nurture"
        
        with st.container():
            st.markdown(f"""
            <div class="{tier_class}">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 45px; height: 45px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                                        border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                        font-size: 1.2rem; color: white; font-weight: bold;">
                                {investor_name[0] if investor_name else 'L'}
                            </div>
                            <div>
                                <div style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">{investor_name}</div>
                                <div style="font-size: 0.8rem; color: #64748b;">
                                    {occupation} • {city} • Potential Investment: {investment_amount}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.75rem; color: #94a3b8;">
                            Lead Ref: {lead.get('Lead Number', f'L-{1000+idx}')} • Source: {lead.get('Lead Source', 'Organic')}
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 120px;">
                        <div style="font-size: 2rem; font-weight: bold; color: {'#10b981' if score >= 0.85 else '#f59e0b' if score >= 0.65 else '#6b7280'};">
                            {score*100:.1f}%
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b;">Conversion Probability</div>
                    </div>
                </div>
                <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="background: {'#ef4444' if score >= 0.85 else '#f59e0b' if score >= 0.65 else '#6b7280'}; 
                                   color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                            {tier_label}
                        </span>
                        <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; 
                                   padding: 4px 10px; border-radius: 12px; font-size: 0.75rem;">
                            {lead.get('Recommended_Pitch_Persona', 'Professional Investor')}
                        </span>
                    </div>
                    <span style="color: #94a3b8; font-size: 0.8rem;">
                        📞 {lead.get('Phone', '+91-9' + ''.join([str(random.randint(0,9)) for _ in range(9)]))}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button(f"📞 {action}", key=f"call_{idx}", use_container_width=True):
                    st.success(f"📞 Calling {investor_name}...")
            with col_btn2:
                if st.button("💬 WhatsApp", key=f"wa_{idx}", use_container_width=True):
                    st.info(f"💬 Opening WhatsApp chat with {investor_name}...")
            with col_btn3:
                if st.button("📧 Email", key=f"email_{idx}", use_container_width=True):
                    st.info(f"📧 Composing email to {investor_name}...")

def render_distributor_funds(profile: UserProfile):
    """Render fund recommendations for distributors"""
    st.markdown("### 💼 Recommended Funds for Your Investor Persona")
    
    persona = PERSONAS.get(profile.persona, PERSONAS["balanced"])
    
    st.info(f"""
    **Your Persona: {persona['name']}**  
    {persona['description']}  
    **Recommended Categories:** {', '.join(persona['recommended_funds'])}
    """)
    
    # Filter funds by persona
    recommended_funds = [f for f in FUNDS_DB if profile.persona in f.recommended_for]
    
    st.markdown("#### 🏆 Top Recommended Funds")
    
    for fund in recommended_funds[:6]:
        risk_color = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}[fund.risk_level]
        
        st.markdown(f"""
        <div class="fund-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h4 style="margin: 0; color: #f8fafc; font-size: 1.25rem;">{fund.name}</h4>
                    <span style="color: #64748b; font-size: 0.85rem;">{fund.category.upper()}</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: {risk_color}; font-weight: 600; font-size: 0.9rem;">
                        {fund.risk_level.upper()} RISK
                    </span>
                    <div style="margin-top: 5px; color: #f59e0b;">{'⭐' * fund.rating}</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 20px;">
                <div style="text-align: center;">
                    <div class="metric-value" style="color: #10b981;">{fund.returns_1y}%</div>
                    <div class="metric-label">1Y</div>
                </div>
                <div style="text-align: center;">
                    <div class="metric-value" style="color: #3b82f6;">{fund.returns_3y}%</div>
                    <div class="metric-label">3Y</div>
                </div>
                <div style="text-align: center;">
                    <div class="metric-value" style="color: #f8fafc;">{fund.returns_5y}%</div>
                    <div class="metric-label">5Y</div>
                </div>
                <div style="text-align: center;">
                    <div class="metric-value" style="color: #f8fafc;">₹{fund.aum:,.0f}Cr</div>
                    <div class="metric-label">AUM</div>
                </div>
                <div style="text-align: center;">
                    <div class="metric-value" style="color: #f8fafc;">{fund.expense_ratio}%</div>
                    <div class="metric-label">Expense</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button("📋 Copy Details", key=f"copy_{fund.fund_id}", use_container_width=True)

def render_distributor_performance(profile: UserProfile):
    """Render performance metrics for distributors"""
    st.markdown("### 📊 Your Distribution Performance")
    
    # Mock performance data
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total AUM", "₹12.5 Cr", "+15% this month"),
        ("Active Investors", "48", "+3 new this week"),
        ("Conversion Rate", "68%", "Above average"),
        ("Commission Earned", "₹45,250", "This month")
    ]
    
    for col, (label, value, subtext) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); padding: 20px; border-radius: 12px; text-align: center;">
                <div style="font-size: 0.85rem; color: #64748b;">{label}</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #f8fafc; margin: 8px 0;">{value}</div>
                <div style="font-size: 0.75rem; color: #10b981;">{subtext}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("#### 📈 Monthly Trends")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    aum_data = [8.2, 9.1, 10.5, 11.2, 11.8, 12.5]
    investors_data = [32, 35, 38, 42, 45, 48]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('AUM Growth (Cr)', 'New Investors'))
    
    fig.add_trace(go.Scatter(x=months, y=aum_data, fill='tozeroy', 
                              line=dict(color='#3b82f6')), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=investors_data, marker_color='#10b981'), row=1, col=2)
    
    fig.update_layout(height=300, showlegend=False, 
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def render_distributor_tools(profile: UserProfile):
    """Render tools and resources for distributors"""
    st.markdown("### 🔧 Distributor Tools & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Learning Resources")
        resources = [
            ("📖 AMFI Training Module", "Complete your certification"),
            ("📊 Fund Fact Sheets", "Latest from all AMCs"),
            ("🎯 Sales Scripts", "Proven conversation starters"),
            ("⚖️ Compliance Guide", "SEBI regulations summary")
        ]
        for title, desc in resources:
            st.markdown(f"""
            <div style="padding: 12px; background: rgba(30, 41, 59, 0.6); 
                       border-radius: 8px; margin-bottom: 10px;">
                <strong>{title}</strong><br>
                <span style="color: #64748b; font-size: 0.85rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🛠️ Quick Tools")
        
        # SIP Calculator
        st.markdown("##### 💰 SIP Calculator")
        monthly = st.number_input("Monthly Investment (₹)", 1000, 100000, 5000, 1000)
        rate = st.slider("Expected Return (%)", 5.0, 20.0, 12.0, 0.5)
        years = st.slider("Duration (Years)", 1, 30, 10, 1)
        
        # Calculate
        months = years * 12
        r = rate / 100 / 12
        future_value = monthly * (((1 + r) ** months - 1) / r) * (1 + r)
        invested = monthly * months
        gains = future_value - invested
        
        st.success(f"""
        **Future Value:** ₹{future_value:,.0f}  
        **Invested:** ₹{invested:,.0f}  
        **Wealth Gained:** ₹{gains:,.0f}
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# INVESTOR VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def render_investor_dashboard(profile: UserProfile, registry: ModelRegistry):
    """Render investor-specific dashboard"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        persona = PERSONAS.get(profile.persona, PERSONAS["balanced"])
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px;">
            <h2 style="margin: 0;">Welcome, {profile.name}</h2>
            <span class="role-badge-investor">💼 Investor</span>
        </div>
        <p style="color: #64748b;">
            Your Persona: <span style="color: {persona['color']}; font-weight: 600;">
            {persona['name']}</span>
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding: 10px; background: rgba(16, 185, 129, 0.1); 
                    border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <div style="font-size: 0.85rem; color: #64748b;">Portfolio Value</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #10b981;">
                ₹{calculate_portfolio_value():,.0f}
            </div>
            <div style="font-size: 0.75rem; color: #10b981;">▲ +12.5% this month</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 My Investments", 
        "👨‍💼 Find Distributors", 
        "📊 Fund Explorer",
        "🔔 Alerts & News"
    ])
    
    with tab1:
        render_investor_portfolio(profile)
    
    with tab2:
        render_find_distributors(profile)
    
    with tab3:
        render_fund_explorer(profile, registry)
    
    with tab4:
        render_investor_alerts(profile)

def calculate_portfolio_value() -> float:
    """Calculate current portfolio value"""
    portfolio = st.session_state.get("portfolio", {})
    if not portfolio:
        return 250000  # Mock value
    
    total = 0
    for fund_id, units in portfolio.items():
        fund = next((f for f in FUNDS_DB if f.fund_id == fund_id), None)
        if fund:
            nav = 100  # Mock NAV
            total += units * nav
    return total

def render_investor_portfolio(profile: UserProfile):
    """Render investor's portfolio with live tracking"""
    st.markdown("### 💰 Your Investment Portfolio")
    
    persona = PERSONAS.get(profile.persona, PERSONAS["balanced"])
    
    # Risk alert based on persona
    st.markdown(f"""
    <div class="persona-card persona-{profile.persona}">
        <h4>📊 Portfolio Health Check</h4>
        <p><strong>Your Persona:</strong> {persona['name']}</p>
        <p><strong>Risk Tolerance:</strong> {persona['risk_tolerance']}</p>
        <p><strong>Suitable for:</strong> {', '.join(persona['recommended_funds'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock portfolio holdings
    holdings = [
        ("UTI Nifty 50 Index Fund", "Index Fund", 50000, 12.5, 6250, "medium"),
        ("HDFC Liquid Fund", "Liquid Fund", 30000, 6.8, 2040, "low"),
        ("SBI Magnum MidCap Fund", "Mid Cap", 45000, 38.5, 17325, "high"),
        ("Edelweiss Balanced Advantage", "Hybrid", 75000, 18.5, 13875, "medium"),
    ]
    
    st.markdown("#### 📋 Your Holdings")
    
    total_invested = sum(h[2] for h in holdings)
    total_current = sum(h[2] + h[4] for h in holdings)
    total_gain = total_current - total_invested
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Invested", f"₹{total_invested:,.0f}")
    with col2:
        st.metric("Current Value", f"₹{total_current:,.0f}", f"+₹{total_gain:,.0f}")
    with col3:
        gain_pct = (total_gain / total_invested) * 100
        st.metric("Total Returns", f"{gain_pct:.1f}%")
    
    # Holdings table
    for name, category, invested, returns, gains, risk in holdings:
        current = invested + gains
        risk_color = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}[risk]
        
        st.markdown(f"""
        <div class="fund-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0;">{name}</h4>
                    <span style="color: #64748b; font-size: 0.85rem;">{category}</span>
                </div>
                <span style="color: {risk_color}; font-size: 0.8rem; font-weight: 600;">
                    {risk.upper()} RISK
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Invested</div>
                    <div style="font-weight: 600;">₹{invested:,.0f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Current</div>
                    <div style="font-weight: 600; color: #10b981;">₹{current:,.0f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Returns</div>
                    <div style="font-weight: 600; color: #10b981;">{returns}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b;">Gains</div>
                    <div style="font-weight: 600; color: #10b981;">+₹{gains:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # LIVE RISK ANALYSIS
    st.markdown("#### 🔴 Live Risk Analysis")
    
    try:
        from lume_platform.risk.live_risk_analyzer import risk_analyzer
        
        # Fetch live market data
        market_data = risk_analyzer.fetch_live_market_data()
        
        # Convert holdings to proper format
        formatted_holdings = [
            {
                "fund_id": f"F{i+1}",
                "fund_name": h[0],
                "category": h[1].lower().replace(" fund", "").replace(" ", "_"),
                "value": h[2],
                "returns": h[3]
            }
            for i, h in enumerate(holdings)
        ]
        
        # Calculate live risk
        risk_metrics = risk_analyzer.calculate_portfolio_risk(
            formatted_holdings, market_data, profile.persona
        )
        
        # Display live market status
        market_color = {
            "OPEN": "#10b981",
            "CLOSED": "#64748b",
            "VOLATILE": "#ef4444"
        }.get(market_data.market_status, "#64748b")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(30,41,59,0.6); border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #64748b;">Market Status</div>
                <div style="font-weight: bold; color: {market_color};">{market_data.market_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            nifty_change_color = "#10b981" if market_data.nifty_change_pct >= 0 else "#ef4444"
            arrow = "▲" if market_data.nifty_change_pct >= 0 else "▼"
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(30,41,59,0.6); border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #64748b;">NIFTY 50</div>
                <div style="font-weight: bold;">{market_data.nifty_50:,.0f}</div>
                <div style="font-size: 0.75rem; color: {nifty_change_color};">{arrow} {abs(market_data.nifty_change_pct):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            vix_color = "#ef4444" if market_data.vix > 20 else "#f59e0b" if market_data.vix > 15 else "#10b981"
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(30,41,59,0.6); border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #64748b;">VIX (Volatility)</div>
                <div style="font-weight: bold; color: {vix_color};">{market_data.vix:.1f}</div>
                <div style="font-size: 0.7rem; color: #64748b;">{'High' if market_data.vix > 20 else 'Normal'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            sentiment_color = {"bullish": "#10b981", "bearish": "#ef4444", "neutral": "#f59e0b"}.get(market_data.sentiment, "#64748b")
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(30,41,59,0.6); border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #64748b;">Sentiment</div>
                <div style="font-weight: bold; color: {sentiment_color};">{market_data.sentiment.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Score Gauge
        st.markdown("#### Portfolio Risk Metrics")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            risk_score = risk_metrics.portfolio_risk_score
            risk_color = "#10b981" if risk_score < 40 else "#f59e0b" if risk_score < 70 else "#ef4444"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: rgba(30,41,59,0.8); border-radius: 12px; border: 2px solid {risk_color};">
                <div style="font-size: 0.8rem; color: #64748b;">Risk Score</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: {risk_color};">{risk_score:.0f}</div>
                <div style="font-size: 0.75rem; color: {risk_color};">{risk_metrics.risk_level.value.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.metric("Value at Risk (95%)", f"₹{risk_metrics.var_95:,.0f}", 
                     help="Maximum expected loss with 95% confidence")
        with col_r3:
            st.metric("Max Drawdown", f"{risk_metrics.max_drawdown:.1f}%",
                     help="Maximum expected portfolio decline")
        with col_r4:
            st.metric("Beta", f"{risk_metrics.beta:.2f}",
                     help="Market correlation (>1 = more volatile than market)")
        
        # Risk Alerts
        if risk_metrics.alerts:
            st.markdown("#### 🚨 Risk Alerts")
            for alert in risk_metrics.alerts:
                alert_color = "#ef4444" if "🔴" in alert else "#f59e0b" if "⚠️" in alert else "#3b82f6"
                st.markdown(f"""
                <div style="padding: 10px 15px; margin: 8px 0; background: rgba(239,68,68,0.1); 
                           border-left: 3px solid {alert_color}; border-radius: 6px;">
                    {alert}
                </div>
                """, unsafe_allow_html=True)
        
        # Recommendations
        if risk_metrics.recommendations:
            st.markdown("#### 💡 AI Recommendations")
            for rec in risk_metrics.recommendations[:3]:  # Show top 3
                st.markdown(f"""
                <div style="padding: 10px 15px; margin: 8px 0; background: rgba(16,185,129,0.1); 
                           border-left: 3px solid #10b981; border-radius: 6px;">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
        
        # Sector Performance
        st.markdown("#### 📊 Sector Performance Impact")
        sector_data = market_data.sector_performance
        
        fig = go.Figure()
        sectors = list(sector_data.keys())
        changes = list(sector_data.values())
        colors = ["#10b981" if c > 0 else "#ef4444" for c in changes]
        
        fig.add_trace(go.Bar(
            x=sectors,
            y=changes,
            marker_color=colors,
            text=[f"{c:+.1f}%" for c in changes],
            textposition="outside"
        ))
        fig.update_layout(
            title="Sector Impact on Your Portfolio",
            showlegend=False,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="% Change")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Last updated: {risk_metrics.last_updated.strftime('%H:%M:%S')}")
        
    except Exception as e:
        st.error(f"Risk analysis temporarily unavailable: {str(e)}")
        # Fallback to simple analysis
        high_risk_funds = [h for h in holdings if h[5] == "high"]
        if profile.persona in ["conservative"] and high_risk_funds:
            st.warning(f"⚠️ Risk mismatch: You have {len(high_risk_funds)} high-risk funds")
        elif profile.persona in ["growth"] and len(high_risk_funds) < 2:
            st.info("💡 Consider adding more growth-oriented funds")
        else:
            st.success("✅ Portfolio aligned with your persona")
        st.success(f"✅ **Portfolio Aligned** - Your investments match your {persona['name']} persona well!")

def render_find_distributors(profile: UserProfile):
    """Render distributor matching for investors"""
    st.markdown("### 👨‍💼 Find the Right Distributor for You")
    
    persona = PERSONAS.get(profile.persona, PERSONAS["balanced"])
    
    st.info(f"""
    **Your Investor Persona:** {persona['name']}  
    **Best Match:** Distributors specializing in {', '.join(persona['recommended_funds'][:2])}
    """)
    
    # Mock distributor database
    distributors = [
        ("Rajesh Sharma", "expert", "growth", "Mumbai", "15 years", "4.9⭐", 450),
        ("Priya Patel", "expert", "balanced", "Delhi", "12 years", "4.8⭐", 320),
        ("Amit Kumar", "intermediate", "conservative", "Bangalore", "5 years", "4.7⭐", 180),
        ("Sneha Gupta", "expert", "passive", "Hyderabad", "8 years", "4.9⭐", 290),
    ]
    
    # Filter by matching persona
    matching = [d for d in distributors if d[2] == profile.persona or d[2] == "balanced"]
    
    st.markdown(f"#### 🎯 {len(matching)} Distributors Match Your Profile")
    
    for name, exp, persona_type, city, experience, rating, clients in matching:
        persona_name = PERSONAS.get(persona_type, PERSONAS["balanced"])['name']
        match_score = 95 if persona_type == profile.persona else 75
        
        st.markdown(f"""
        <div class="fund-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="display: flex; gap: 15px;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                               border-radius: 50%; display: flex; align-items: center; justify-content: center;
                               font-size: 1.5rem;">
                        👔
                    </div>
                    <div>
                        <h4 style="margin: 0;">{name}</h4>
                        <span style="color: #64748b; font-size: 0.85rem;">{city} • {experience}</span>
                        <div style="margin-top: 5px;">
                            <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; 
                                       padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">
                                {persona_name}
                            </span>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #10b981; 
                                       padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 5px;">
                                {exp.title()}
                            </span>
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{match_score}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Match Score</div>
                    <div style="margin-top: 5px; font-size: 0.9rem;">{rating}</div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <div style="flex: 1; text-align: center; padding: 8px; background: rgba(30, 41, 59, 0.6); 
                           border-radius: 6px;">
                    <div style="font-weight: bold;">{clients}</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Active Clients</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📞 Contact", key=f"contact_{name}", use_container_width=True)
        with col2:
            st.button("📅 Schedule Meeting", key=f"meet_{name}", use_container_width=True)
        with col3:
            st.button("📋 View Profile", key=f"profile_{name}", use_container_width=True)

def render_fund_explorer(profile: UserProfile, registry: Optional[ModelRegistry] = None):
    """Render fund explorer with AI momentum and dynamic match scoring"""
    st.markdown("### 📊 AI-Driven Fund Explorer")
    
    # Load Latest Live Sentiment for Dynamic Scoring
    news_file = ROOT / "streaming/outputs/news_live.parquet"
    live_sentiment = "neutral"
    if news_file.exists():
        try:
            # Read all parts and get the latest
            df_news_all = pd.read_parquet(news_file)
            if not df_news_all.empty:
                # Use ingestion_timestamp or timestamp depending on which exists
                time_col = "ingestion_timestamp" if "ingestion_timestamp" in df_news_all.columns else "timestamp"
                latest_n = df_news_all.sort_values(time_col, ascending=False).iloc[0]
                live_sentiment = str(latest_n.get("market_impact", latest_n.get("sentiment_category", "neutral"))).lower()
        except Exception as e: 
            pass

    # Logic for Dynamic Match Scoring (Sentiment-Aware)
    def calculate_match_score(fund: Fund, profile: UserProfile, sentiment: str = "neutral") -> int:
        score = 65 # Base
        if profile.persona in fund.recommended_for:
            score += 25
        
        # Sentiment-Aware Priority Boost
        if sentiment == "bullish" and fund.category == "equity":
            score += 7
        elif sentiment == "bearish" and fund.category == "debt":
            score += 7
        
        if fund.risk_level == "high" and profile.persona == "growth":
            score += 5
        elif fund.risk_level == "low" and profile.persona == "conservative":
            score += 5
            
        return min(99, score + random.randint(-1, 1))

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        category_filter = st.multiselect("Category", ["equity", "debt", "hybrid", "index"], default=[])
    with col2:
        risk_filter = st.multiselect("Risk Level", ["low", "medium", "high"], default=[])
    with col3:
        min_return = st.slider("Min 1Y Return (%)", 0.0, 60.0, 0.0, 5.0)
    with col4:
        sort_by = st.selectbox("Sort By", ["Returns (1Y)", "Returns (3Y)", "Rating", "AUM"])
    
    filtered = [f for f in FUNDS_DB if (not category_filter or f.category in category_filter) and 
                                       (not risk_filter or f.risk_level in risk_filter) and
                                       (f.returns_1y >= min_return)]
    
    # Sort
    sort_map = {"Returns (1Y)": lambda x: x.returns_1y, "Returns (3Y)": lambda x: x.returns_3y,
                "Rating": lambda x: x.rating, "AUM": lambda x: x.aum}
    filtered = sorted(filtered, key=sort_map.get(sort_by, lambda x: x.returns_1y), reverse=True)
    
    st.markdown(f"#### Showing {len(filtered)} Funds")
    
    for fund in filtered:
        risk_color = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}[fund.risk_level]
        match_score = calculate_match_score(fund, profile, live_sentiment)
        match_glow_opacity = match_score / 100 * 0.2
        
        # LSTM Forescaster: Load Real History for Accurate Predictions
        ai_momentum = "Calculating..."
        history_file = HISTORY_MAP.get(fund.fund_id)
        full_path = DATA_ROOT / "structured/mutual_funds/nav_history" / history_file if history_file else None
        
        if registry and registry.forecaster and full_path and full_path.exists():
            try:
                df_h = pd.read_csv(full_path)
                nav_history = pd.to_numeric(df_h['nav'], errors='coerce').dropna().tail(35).tolist()
                if len(nav_history) >= 30:
                    f_res = registry.forecaster.forecast(nav_history)
                    if "error" not in f_res:
                        traj = f_res["forecast_trajectory"]
                        pct_change = ((traj[-1] - nav_history[-1]) / nav_history[-1]) * 100
                        ai_momentum = f"{pct_change:+.1f}%"
                else:
                    ai_momentum = "+1.8% (Est)" # Fallback if history too short
            except:
                ai_momentum = "+2.1% (Est)"
        else:
            # Safe Fallback to 1Y return scaled if CSV missing
            ai_momentum = f"{fund.returns_1y / 15:+.1f}% (Est)"

        # Final HTML Assembly - Flush left to prevent markdown block detection
        card_html = f"""<div class="fund-card" style="position: relative; overflow: hidden; padding: 24px; background: rgba(30, 41, 59, 0.45); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); color: #f8fafc; margin-bottom: 24px;">
    <div style="position: absolute; top: 12px; right: 20px; font-size: 0.6rem; color: #64748b; font-weight: 800; letter-spacing: 0.1em; opacity: 0.7;">📡 SPARK VELOCITY DATA</div>
    <div class="match-glow" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle at 70% 30%, rgba(16, 185, 129, {match_glow_opacity}), transparent 70%); pointer-events: none; opacity: 0.4;"></div>
    
    <div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative; z-index: 1;">
        <div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <h3 style="margin: 0; font-size: 1.4rem; color: #ffffff; letter-spacing: -0.01em;">{fund.name}</h3>
                <span style="background: linear-gradient(135deg, rgba(167, 139, 250, 0.25), rgba(139, 92, 246, 0.25)); color: #c084fc; padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: 800; border: 1px solid rgba(167, 139, 250, 0.3);">🔮 {ai_momentum} AI</span>
            </div>
            <div style="margin-top: 10px; display: flex; align-items: center; gap: 18px;">
                <span style="color: #64748b; font-size: 0.85rem; font-weight: 700;">{fund.category.upper()}</span>
                <span style="color: #3b82f6; font-size: 0.95rem; font-weight: 800;">{match_score}% MATCH</span>
                {f'<span style="color: #10b981; font-size: 0.8rem; font-weight: 800; background: rgba(16, 185, 129, 0.12); padding: 3px 10px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2);">◈ {live_sentiment.upper()} BOOST</span>' if live_sentiment != "neutral" else ""}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="background: rgba({int(risk_color[1:3],16)}, {int(risk_color[3:5],16)}, {int(risk_color[5:7],16)}, 0.15); color: {risk_color}; border: 1px solid rgba({int(risk_color[1:3],16)}, {int(risk_color[3:5],16)}, {int(risk_color[5:7],16)}, 0.3); padding: 5px 14px; border-radius: 8px; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.02em;">{fund.risk_level.upper()} RISK</div>
            <div style="margin-top: 12px; color: #f59e0b; font-size: 1.15rem; letter-spacing: 3px;">{'⭐' * fund.rating}</div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 30px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 20px; position: relative; z-index: 1;">
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #10b981;">{fund.returns_1y}%</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">1Y Ret</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #3b82f6;">{fund.returns_3y}%</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">3Y Ret</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">{fund.returns_5y}%</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">5Y Ret</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #a78bfa;">{ai_momentum}</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">AI Forecast</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">₹{fund.aum:,.0f}Cr</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">AUM</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">{fund.expense_ratio}%</div>
            <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Expense</div>
        </div>
    </div>
</div>"""
        # Use st.components.v1.html to ENSURE rendering (prevents markdown escaping raw code)
        html(card_html, height=220, scrolling=False)

def render_investor_alerts(profile: UserProfile):
    """Render alerts and news for investors"""
    st.markdown("### 🔔 Alerts & Market News")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Market Updates")
        
        # Mock alerts
        alerts = [
            ("🚨", "High volatility detected in Small Cap funds", "2 hours ago", "warning"),
            ("📈", "Nifty 50 up 1.2% - Your Index funds performing well", "Today", "success"),
            ("💡", "SIP due in 3 days", "Reminder", "info"),
            ("⚠️", "HDFC Liquid Fund NAV delayed today", "1 hour ago", "warning"),
        ]
        
        for icon, message, time, alert_type in alerts:
            color = {"warning": "#f59e0b", "success": "#10b981", "info": "#3b82f6"}[alert_type]
            st.markdown(f"""
            <div style="padding: 12px; margin-bottom: 10px; 
                       background: rgba(30, 41, 59, 0.6);
                       border-radius: 8px; border-left: 3px solid {color};">
                <div style="display: flex; justify-content: space-between;">
                    <span>{icon} {message}</span>
                    <span style="color: #64748b; font-size: 0.75rem;">{time}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📰 Financial News")
        
        news_items = [
            ("SEBI", "New regulations for mutual fund distributors announced", "3 hours ago"),
            ("Markets", "FII inflows continue for 5th consecutive day", "5 hours ago"),
            ("Economy", "RBI expected to maintain status quo on rates", "8 hours ago"),
        ]
        
        for source, headline, time in news_items:
            st.markdown(f"""
            <div style="padding: 12px; margin-bottom: 10px; 
                       background: rgba(30, 41, 59, 0.6);
                       border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #3b82f6; margin-bottom: 4px;">
                    {source}
                </div>
                <div style="font-size: 0.9rem;">{headline}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                    {time}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING & SURVEY
# ═══════════════════════════════════════════════════════════════════════════════

def render_onboarding():
    """Render user onboarding with survey"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <div style="font-size: 4rem;">💎</div>
        <h1 style="margin: 10px 0;">Welcome to Lume AI</h1>
        <p style="color: #64748b; font-size: 1.2rem;">
            India's Smartest Wealth Management Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Step 1: Choose Role
    if st.session_state.survey_step == 0:
        st.markdown("### Step 1: Choose Your Role")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="padding: 30px; background: rgba(59, 130, 246, 0.1); 
                       border-radius: 16px; border: 2px solid #3b82f6; text-align: center;">
                <div style="font-size: 3rem;">👔</div>
                <h3 style="margin: 10px 0;">I'm a Distributor</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    Help investors find the right funds and grow your business
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Distributor", use_container_width=True, type="primary"):
                st.session_state.survey_answers["role"] = "distributor"
                st.session_state.survey_step = 1
        
        with col2:
            st.markdown("""
            <div style="padding: 30px; background: rgba(16, 185, 129, 0.1); 
                       border-radius: 16px; border: 2px solid #10b981; text-align: center;">
                <div style="font-size: 3rem;">💼</div>
                <h3 style="margin: 10px 0;">I'm an Investor</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    Find the right funds and distributors for your goals
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Investor", use_container_width=True, type="primary"):
                st.session_state.survey_answers["role"] = "investor"
                st.session_state.survey_step = 1
    
    # Step 2: Experience Level (for distributors)
    elif st.session_state.survey_step == 1 and st.session_state.survey_answers.get("role") == "distributor":
        st.markdown("### Step 2: Your Experience Level")
        
        exp_levels = [
            ("beginner", "🌱 Beginner", "Just starting out, need guidance"),
            ("intermediate", "📈 Intermediate", "Some experience, looking to grow"),
            ("expert", "🚀 Expert", "Experienced, managing multiple clients")
        ]
        
        for key, title, desc in exp_levels:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Select", key=f"exp_{key}", use_container_width=True):
                    st.session_state.survey_answers["experience"] = key
                    st.session_state.survey_step = 2
            with col2:
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(30, 41, 59, 0.6); border-radius: 8px;">
                    <strong>{title}</strong><br>
                    <span style="color: #64748b; font-size: 0.85rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Step 2: Persona Survey (for investors, or step 3 for distributors)
    else:
        role = st.session_state.survey_answers.get("role")
        
        # Select appropriate survey based on role
        if role == "investor":
            survey = INVESTOR_SURVEY
            current_q = st.session_state.survey_step - 1
            step_offset = 1
        else:  # distributor
            survey = DISTRIBUTOR_SURVEY
            current_q = st.session_state.survey_step - 2
            step_offset = 2
        
        if current_q < len(survey):
            question = survey[current_q]
            
            # Show different header based on role
            if role == "investor":
                st.markdown("### 💼 Investor Persona Assessment")
                st.info("📋 These questions help us understand your **investment style**, **risk tolerance**, and **financial goals** to recommend suitable funds.")
            else:
                st.markdown("### 👔 Distributor Persona Assessment")
                st.info("📋 These questions help us understand your **business style**, **client focus**, and **expertise areas** to match you with suitable investors and funds.")
            
            st.markdown(f"**Question {current_q + 1} of {len(survey)}**")
            st.markdown(f"<h4>{question['question']}</h4>", unsafe_allow_html=True)
            
            # Show progress bar
            progress = (current_q + 1) / len(survey)
            st.progress(progress)
            
            for option, persona in question['options'].items():
                if st.button(option, use_container_width=True, key=f"q_{current_q}_{option}_{role}"):
                    st.session_state.survey_answers[question['id']] = persona
                    st.session_state.survey_step += 1
                    st.rerun()
        else:
            # Survey complete - create profile
            create_user_profile()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample_leads(n: int, distributor_persona: str) -> List[Dict]:
    """Generate sample leads for demo"""
    personas = list(PERSONAS.keys())
    sources = ["Website", "Referral", "Social Media", "Seminar", "Walk-in"]
    
    leads = []
    for i in range(n):
        persona = random.choice(personas)
        score = random.uniform(0.4, 0.98)
        
        leads.append({
            "Lead Number": f"L-{1000+i}",
            "Lead Source": random.choice(sources),
            "Conversion_Probability": score,
            "Recommended_Pitch_Persona": PERSONAS[persona]['name'],
            "Investor_Persona": persona
        })
    
    return leads

def load_leads_data() -> pd.DataFrame:
    """Load leads from file or return empty"""
    path = EXPORT_DIR / "distributor_leads_master.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def create_user_profile():
    """Create user profile from survey answers"""
    answers = st.session_state.survey_answers
    
    # Determine persona using KMeans
    behavior_cols = ["ProfManage", "Diversification", "Affordability", "Liquidity", "Growth", "Trustworthiness", "Technology"]
    persona_answers = {k: v for k, v in answers.items() if k in behavior_cols}
    
    # Load registry if not in state
    registry = ModelRegistry.get_instance()
    persona = determine_persona(persona_answers, registry)
    
    # Create profile
    profile = UserProfile(
        user_id=f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        role=answers.get("role", "investor"),
        name="Demo User",
        email="user@example.com",
        persona=persona,
        experience_level=answers.get("experience", "beginner"),
        created_at=datetime.now().isoformat(),
        survey_responses=answers
    )
    
    st.session_state.user_profile = profile
    st.success("✅ Profile created! Welcome to Lume AI!")
    st.rerun()

def render_live_pulse(registry: Optional[ModelRegistry] = None):
    """Real-time streaming dashboard section for app_users"""
    st.header("⚡ Real-Time AI Market Pulse")
    st.markdown("Distributed Spark stream scoring news & market velocity in real-time.")
    
    col_mkt, col_sent = st.columns([2, 1])
    
    # 1. Market Stream
    with col_mkt:
        st.markdown('<div class="fund-card">', unsafe_allow_html=True)
        st.subheader("🏦 Live Index Monitor")
        mkt_path = ROOT / "streaming/outputs/market_live.parquet"
        if mkt_path.exists():
            try:
                import glob
                mkt_files = sorted(glob.glob(str(mkt_path / "part-*.parquet")), key=os.path.getmtime, reverse=True)
                if mkt_files:
                    # Optimized: Read only the last few parts for live data
                    # Defense: Use engine 'pyarrow' and ensure we read numeric values correctly
                    df_raw = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in mkt_files[:10]])
                    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
                    # Convert columns to numeric to solve the 0.00 bug (sometimes types get lost in parquet serialization)
                    for col in ['last_price', 'pct_change', 'variation']:
                        if col in df_raw.columns:
                            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                    
                    df_live = df_raw.sort_values("timestamp", ascending=False).head(40)
                else:
                    df_live = pd.DataFrame()

                if not df_live.empty:
                    latest = df_live.iloc[0].fillna(0)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("NIFTY 50", f"₹{latest.get('last_price', 0):,.2f}", f"{latest.get('pct_change', 0):+.2f}%")
                    m2.metric("Variation", f"{latest.get('variation', 0):+.2f}")
                    m3.metric("Status", str(latest.get('status', 'N/A')))
                    
                    fig = px.line(df_live, x="timestamp", y="last_price", title="Real-Time Price Velocity")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 1.1 AI Future Forecast Integration
                    if registry and registry.forecaster:
                        st.divider()
                        st.subheader("🔮 AI Future Trajectory (5-Day)")
                        hist_data = df_live["last_price"].values[::-1].tolist()
                        if len(hist_data) >= 30:
                            f_res = registry.forecaster.forecast(hist_data)
                            if "error" not in f_res:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Forecasting Trend", f_res["trend"])
                                c2.metric("Confidence Score", f"{f_res['confidence_score']}%")
                                c3.metric("Model Precision", f_res["accuracy_metric"])
                                
                                # Plotting Forecast
                                next_days = [f"T+{i+1}" for i in range(5)]
                                f_df = pd.DataFrame({
                                    "Day": next_days,
                                    "Predicted_Price": f_res["forecast_trajectory"]
                                })
                                fig_f = px.area(f_df, x="Day", y="Predicted_Price", title="Predicted Market Path")
                                fig_f.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.2)')
                                fig_f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fig_f, use_container_width=True)
                            else:
                                st.caption(f"Waiting for more data points... ({len(hist_data)}/30)")
                        else:
                            st.caption(f"Accumulating historical context for AI inference ({len(hist_data)}/30)...")
                else:
                    st.info("Waiting for data packets...")
            except Exception as e:
                st.warning("Connecting to Spark...")
        else:
            st.info("📡 Spark Stream Offline. Run `src/lume_platform/spark/streaming_pipeline.py` first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Sentiment Stream
    with col_sent:
        st.markdown('<div class="fund-card">', unsafe_allow_html=True)
        st.subheader("🧠 Intelligence Feed")
        news_path = ROOT / "streaming/outputs/news_live.parquet"
        if news_path.exists():
            try:
                import glob
                news_files = sorted(glob.glob(str(news_path / "part-*.parquet")), key=os.path.getmtime, reverse=True)
                if news_files:
                    # Optimized: Read only the latest news partition
                    df_n_raw = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in news_files[:5]])
                    df_n_raw.columns = [c.lower() for c in df_n_raw.columns]
                    df_news = df_n_raw.sort_values("ingestion_timestamp", ascending=False).head(20)
                else:
                    df_news = pd.DataFrame()

                if not df_news.empty:
                    latest_sent = df_news.iloc[0].get('sentiment_score', 0.5)
                    
                    # Sentiment Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = latest_sent * 100,
                        title = {'text': "Sentiment (%)"},
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
                    
                    st.markdown("#### Latest News Impact")
                    for _, row in df_news.head(3).iterrows():
                        impact = row.get('market_impact', 'NEUTRAL')
                        color = "#10b981" if impact == "BULLISH" else "#ef4444" if impact == "BEARISH" else "#f59e0b"
                        st.write(f"**{row.get('title', 'N/A')}**")
                        st.markdown(f"<span style='color:{color}; font-weight:bold;'>[{impact}]</span>", unsafe_allow_html=True)
                        st.divider()
                else:
                    st.info("Awaiting news scoring...")
            except Exception:
                pass
        else:
            st.info("No live news stream detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Refresh Stream"):
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    init_session_state()
    
    # Load models
    registry = ModelRegistry()
    registry.load()
    
    # Check if user has profile
    if st.session_state.user_profile is None:
        render_onboarding()
    else:
        profile = st.session_state.user_profile
        
        # Sidebar Navigation
        with st.sidebar:
            st.title("💎 Lume AI")
            st.caption(f"ID: {profile.user_id}")
            st.divider()
            nav = st.radio("Navigation", ["🏠 My Dashboard", "⚡ Live AI Pulse"])
            st.divider()
            if st.button("🚪 Logout"):
                st.session_state.user_profile = None
                st.rerun()

        if nav == "⚡ Live AI Pulse":
            render_live_pulse(registry)
        elif profile.role == "distributor":
            render_distributor_dashboard(profile, registry)
        else:
            render_investor_dashboard(profile, registry)

if __name__ == "__main__":
    main()
