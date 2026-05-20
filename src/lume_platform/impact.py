"""Market impact engine for translating macro/news signals into sector and portfolio impact.

The engine ingests market/news signals, maps them to sectors, and returns:
- sector impact scores
- severity/alert level
- recommended actions
- explanation text for dashboards and APIs

This is intentionally deterministic and interpretable so it can be used in demos
and production prototypes without requiring an opaque model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ImpactResult:
    timestamp: str
    headline: str
    severity: str
    confidence: float
    sector_impact: Dict[str, float]
    affected_sectors: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    explanation: List[str]


class MarketImpactEngine:
    """Rule-based market impact engine."""

    sector_keywords = {
        "Banking": ["repo", "rate hike", "rate cut", "rbi", "gdp", "liquidity", "banks", "banking"],
        "Debt": ["repo", "rate hike", "inflation", "bond", "yield", "debt", "gilt"],
        "Energy": ["oil", "crude", "energy", "brent", "diesel", "petrol"],
        "FMCG": ["inflation", "demand", "rural", "consumption", "fmcg"],
        "IT": ["usd", "dollar", "it spending", "software", "tech", "recession"],
        "Mid Cap": ["mid cap", "mid-cap", "broadening", "growth", "domestic"],
        "Large Cap": ["gdp", "blue chip", "large cap", "stability", "earnings"],
        "Metals": ["china", "commodities", "metals", "steel", "aluminum"],
    }

    positive_tokens = ["growth", "beat", "upgrade", "cut", "support", "stable", "record", "surge", "strong", "bullish", "ease"]
    negative_tokens = ["hike", "slowdown", "miss", "downgrade", "stress", "inflation", "crash", "selloff", "weak", "bearish", "tighten"]

    def analyze(self, signals: List[Dict[str, Any]], market_context: Optional[Dict[str, Any]] = None, user_profile: Optional[Dict[str, Any]] = None) -> ImpactResult:
        market_context = market_context or {}
        user_profile = user_profile or {}

        text_blob = self._build_text_blob(signals)
        headline = self._headline_from_signals(signals)

        raw_score = self._score_text(text_blob)
        sector_impact = self._sector_impact(text_blob, market_context)
        severity = self._severity_from_score(raw_score, market_context)
        confidence = self._confidence(signals, market_context)
        affected = [s for s, score in sector_impact.items() if abs(score) >= 0.25]

        alerts = self._alerts(severity, sector_impact, market_context)
        actions = self._recommendations(severity, sector_impact, user_profile)
        explanation = self._explain(text_blob, sector_impact, severity)

        return ImpactResult(
            timestamp=datetime.utcnow().isoformat(),
            headline=headline,
            severity=severity,
            confidence=round(confidence, 4),
            sector_impact={k: round(v, 3) for k, v in sector_impact.items()},
            affected_sectors=affected,
            alerts=alerts,
            recommended_actions=actions,
            explanation=explanation,
        )

    def _build_text_blob(self, signals: List[Dict[str, Any]]) -> str:
        parts = []
        for s in signals or []:
            for key in ("title", "headline", "category", "sentiment", "market_impact", "summary", "body", "content"):
                val = s.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip().lower())
        return " ".join(parts)

    def _headline_from_signals(self, signals: List[Dict[str, Any]]) -> str:
        if not signals:
            return "No market signals available"
        first = signals[0]
        return first.get("title") or first.get("headline") or first.get("category") or "Market signal"

    def _score_text(self, text: str) -> float:
        pos = sum(1 for t in self.positive_tokens if t in text)
        neg = sum(1 for t in self.negative_tokens if t in text)
        score = (pos - neg) / max(1, pos + neg)
        return max(-1.0, min(1.0, score))

    def _sector_impact(self, text: str, market_context: Dict[str, Any]) -> Dict[str, float]:
        impact: Dict[str, float] = {sector: 0.0 for sector in self.sector_keywords}
        sentiment = str(market_context.get("market_sentiment", market_context.get("sentiment", "neutral"))).lower()
        vix = float(market_context.get("vix", 15.0) or 15.0)
        market_bias = -0.15 if sentiment == "bearish" else 0.15 if sentiment == "bullish" else 0.0

        for sector, keywords in self.sector_keywords.items():
            score = 0.0
            for kw in keywords:
                if kw in text:
                    score += 0.35
            if sector in ("Banking", "Large Cap") and ("gdp" in text or "earnings" in text):
                score += 0.25
            if sector in ("Debt",) and ("rate hike" in text or "inflation" in text):
                score -= 0.5
            if sector in ("Energy",) and ("oil" in text or "crude" in text):
                score += -0.35 if "increase" in text or "spike" in text else 0.2

            # overall market volatility dampens risk assets
            if sector in ("Mid Cap", "Large Cap", "IT"):
                score += market_bias - (0.1 if vix > 20 else 0.0)
            else:
                score += market_bias * 0.5

            impact[sector] = max(-1.0, min(1.0, score))

        return impact

    def _severity_from_score(self, score: float, market_context: Dict[str, Any]) -> str:
        vix = float(market_context.get("vix", 15.0) or 15.0)
        if score <= -0.6 or vix >= 28:
            return "critical"
        if score <= -0.25 or vix >= 22:
            return "high"
        if score >= 0.45 and vix <= 18:
            return "positive"
        return "moderate"

    def _confidence(self, signals: List[Dict[str, Any]], market_context: Dict[str, Any]) -> float:
        base = 0.55 + min(len(signals), 8) * 0.04
        if market_context.get("vix"):
            base += 0.05
        return max(0.35, min(0.95, base))

    def _alerts(self, severity: str, sector_impact: Dict[str, float], market_context: Dict[str, Any]) -> List[str]:
        alerts = []
        vix = float(market_context.get("vix", 15.0) or 15.0)
        if severity in ("high", "critical"):
            alerts.append("Market impact is negative enough to warrant defensive positioning.")
        if vix >= 25:
            alerts.append(f"Volatility elevated: VIX at {vix:.1f}.")

        top_negative = sorted(sector_impact.items(), key=lambda kv: kv[1])[:2]
        for sector, score in top_negative:
            if score <= -0.25:
                alerts.append(f"{sector} may underperform based on current signals (impact {score:+.2f}).")
        return alerts

    def _recommendations(self, severity: str, sector_impact: Dict[str, float], user_profile: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        risk = str(user_profile.get("inferred_risk_profile", user_profile.get("risk_profile", "balanced"))).lower()
        if severity in ("high", "critical"):
            recs.append("Increase allocation to debt or liquid funds for short-term defense.")
            recs.append("Stagger equity purchases through SIP/STP rather than lump sum.")
        if "conservative" in risk:
            recs.append("Prefer large-cap and debt-oriented products until volatility normalizes.")
        elif "aggressive" in risk or "growth" in risk:
            recs.append("Use the correction to build positions selectively in strong sectors.")

        top_positive = sorted(sector_impact.items(), key=lambda kv: kv[1], reverse=True)[:2]
        if top_positive and top_positive[0][1] > 0.25:
            recs.append(f"Overweight {top_positive[0][0]} exposure while the signal remains supportive.")
        return recs

    def _explain(self, text: str, sector_impact: Dict[str, float], severity: str) -> List[str]:
        messages = [f"Detected sentiment blend from signals: '{text[:140]}'"]
        messages.append(f"Impact severity classified as {severity.upper()}.")
        strongest = sorted(sector_impact.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        for sector, score in strongest:
            messages.append(f"{sector} impact score: {score:+.2f}")
        return messages
