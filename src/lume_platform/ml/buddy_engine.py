"""Hybrid NLP engine for the AI Buddy chat surfaces.

v2: Context-aware dashboard assistant with multi-variant responses,
    deduplication, and structured chart explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
import random
import re
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class BuddyKnowledgeItem:
    title: str
    audience: str
    tags: tuple[str, ...]
    answers: tuple[str, ...]  # Multiple variants
    follow_up: str


@dataclass(frozen=True)
class BuddyReply:
    response: str
    confidence: float
    category: str
    audience: str
    matched_topic: str
    follow_up: str


# ── Chart/graph-related keyword detection ──
_CONTEXT_KEYWORDS = frozenset({
    "graph", "graphs", "chart", "charts", "screen", "explain", "what does",
    "show me", "what is shown", "numbers", "on screen", "display", "visible",
    "what are these", "describe", "interpret", "analysis shown", "dashboard",
    "kpi", "metrics", "frontier", "radar", "benchmark", "allocation",
})


def _is_context_question(text: str) -> bool:
    """Detect if the user is asking about visible dashboard data."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _CONTEXT_KEYWORDS)


class BuddyEngine:
    """Semantic retrieval plus intent routing for distributor and investor chat."""

    def __init__(self) -> None:
        self.knowledge_base = self._build_knowledge_base()
        self._corpus = [self._item_text(item) for item in self.knowledge_base]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self._corpus)
        self.transformer = self._load_transformer()
        self._recent_hashes: list[str] = []  # Track recent responses for dedup

    def _build_knowledge_base(self) -> list[BuddyKnowledgeItem]:
        return [
            BuddyKnowledgeItem(
                title="Distributor onboarding and basics",
                audience="distributor",
                tags=("getting started", "arn", "amfi", "nism", "basic"),
                answers=(
                    "To get started as a mutual fund distributor, complete NISM certification and AMFI registration first. Build trust with clients by centering conversations on their goals, risk appetite, and suitability — recommend products that match their profile rather than pushing a single fund.",
                    "The onboarding path is: NISM exam → AMFI registration → ARN number → KYC setup. Focus on understanding each client's financial goals before making any recommendations. Trust-first, product-second.",
                    "Start with your NISM certification, then register with AMFI for your ARN. The key to success is genuinely understanding investor goals — when you match the right product to the right person, conversions follow naturally.",
                ),
                follow_up="Want me to break this into a 7-day onboarding checklist?",
            ),
            BuddyKnowledgeItem(
                title="Lead handling priorities",
                audience="distributor",
                tags=("lead", "hot lead", "warm lead", "follow up", "crm", "cold"),
                answers=(
                    "Hot leads should be contacted within an hour, warm leads within 24 hours, and cold leads need a structured nurture plan. Personalize every pitch using the investor's horizon, risk profile, and recent market concerns.",
                    "Lead prioritization matters: respond to hot leads within 60 minutes — every hour of delay reduces conversion by ~10%. Warm leads get a same-day callback. For cold leads, set up a 3-touch nurture sequence over 2 weeks.",
                    "Think of lead management in tiers: 🔥 Hot (call now, within 1 hour) → ☀️ Warm (today, personalized follow-up) → ❄️ Cold (weekly nurture with educational content). Always reference the investor's specific needs.",
                ),
                follow_up="I can generate a call script or CRM follow-up plan for a specific lead.",
            ),
            BuddyKnowledgeItem(
                title="Objection handling",
                audience="distributor",
                tags=("objection", "risk", "not now", "market fall", "volatility", "panic", "scared"),
                answers=(
                    "When handling objections, follow the AER framework: Acknowledge the concern, Explain the trade-off in plain language, and Restate the original goal. For volatility worries, SIPs reduce timing risk because purchases are spread across multiple market levels.",
                    "The best objection handlers don't argue — they empathize first. 'I understand the market feels uncertain. That's exactly why SIPs work: you buy more units when prices dip, which lowers your average cost over time.'",
                    "Every objection is a hidden concern. 'Market is too volatile' means 'I'm afraid of losing money.' Address the fear directly: show historical SIP returns through market crashes, and explain how rupee-cost averaging protects them.",
                ),
                follow_up="Share the exact objection and I'll draft a response you can use on the call.",
            ),
            BuddyKnowledgeItem(
                title="Compliance and suitability",
                audience="distributor",
                tags=("compliance", "sebi", "amfi", "kyc", "suitability", "disclosure"),
                answers=(
                    "Always complete KYC, disclose fees and commissions transparently, and document why each recommendation fits the client's risk profile and horizon. Never promise guaranteed returns — that's a SEBI violation.",
                    "Compliance is non-negotiable: KYC first, suitability assessment second, recommendation third. Every interaction should be documented. If SEBI audits your practice, your records should show clear reasoning for every recommendation.",
                    "The compliance checklist: ✅ KYC complete ✅ Risk profile documented ✅ Fee/commission disclosed ✅ Suitability match recorded. Skipping any of these puts your ARN at risk.",
                ),
                follow_up="I can give you a compliance-safe script for this scenario.",
            ),
            BuddyKnowledgeItem(
                title="What SIP means",
                audience="investor",
                tags=("sip", "systematic investment plan", "monthly investing", "regular"),
                answers=(
                    "A SIP (Systematic Investment Plan) lets you invest a fixed amount regularly — usually monthly. It removes the pressure of market timing and uses rupee-cost averaging to smooth out your purchase price over time.",
                    "Think of SIP as a financial habit, like a subscription to wealth-building. You invest a fixed amount every month regardless of market conditions. When markets dip, you automatically buy more units at lower prices.",
                    "SIP is the most disciplined way to invest. Instead of trying to time the perfect entry point (which even experts can't do consistently), you invest systematically and let time + compounding do the heavy lifting.",
                ),
                follow_up="Share your time horizon and risk level, and I'll suggest a SIP strategy that fits.",
            ),
            BuddyKnowledgeItem(
                title="Risk and asset allocation",
                audience="investor",
                tags=("risk", "asset allocation", "diversification", "portfolio", "allocate"),
                answers=(
                    "The right asset mix depends on three factors: your goal, your time horizon, and how much volatility you can tolerate. Longer horizons can absorb more equity risk, while shorter goals should lean toward debt or liquid exposure.",
                    "Asset allocation is the most important investment decision you'll make — it determines ~90% of portfolio returns over time. A simple rule of thumb: equity % = 100 minus your age. But your actual comfort with risk matters more than any formula.",
                    "Diversification isn't just owning multiple funds — it's having exposure across asset classes that don't move together. A well-diversified portfolio might include large-cap equity for stability, mid-cap for growth, and short-term debt for safety.",
                ),
                follow_up="Tell me your goal and timeline, and I'll translate it into a concrete allocation framework.",
            ),
            BuddyKnowledgeItem(
                title="Market timing and lumpsum investing",
                audience="investor",
                tags=("market timing", "lumpsum", "buy now", "should i invest", "small cap", "right now", "when"),
                answers=(
                    "Timing the market consistently is nearly impossible — even professional fund managers rarely get it right. If uncertainty feels high, stagger your entry with SIP or a 3-month lumpsum split. Stay invested in a way you can follow consistently.",
                    "The data is clear: time in the market beats timing the market. Over any 10-year period, the Nifty 50 has delivered positive returns regardless of entry point. If you're investing for the long term, the best time to start is now — but use SIP to reduce entry-point anxiety.",
                    "Rather than asking 'should I buy now?', ask 'what's my time horizon?' If it's 5+ years, market timing matters very little. Start a SIP today and let compounding work. If it's shorter, consider a balanced or debt-oriented approach.",
                ),
                follow_up="I can suggest a conservative, balanced, or growth-oriented entry approach.",
            ),
            BuddyKnowledgeItem(
                title="Portfolio review",
                audience="both",
                tags=("portfolio review", "review my portfolio", "rebalance", "how am i doing"),
                answers=(
                    "Review your holdings against your original goal, horizon, and risk level. Rebalance when one segment grows too large (more than 5-10% deviation) or when your objectives change — not in reaction to every market headline.",
                    "A good portfolio review checks three things: 1) Are you still on track for your goal? 2) Has your risk tolerance changed? 3) Is any single holding dominating your portfolio? If the answer to #3 is yes, it's time to rebalance.",
                    "Portfolio drift happens naturally — winners grow larger, changing your allocation. Review quarterly, rebalance annually. The goal is to maintain the risk level you originally signed up for.",
                ),
                follow_up="I can review either an investor portfolio or a distributor lead portfolio.",
            ),
            BuddyKnowledgeItem(
                title="Fund category guidance",
                audience="both",
                tags=("mutual fund", "best fund", "recommend fund", "fund type", "equity", "debt", "hybrid", "which fund"),
                answers=(
                    "Match the fund category to your use case: equity for long-term growth (5+ years), debt for capital preservation and stability, hybrid for a balanced approach, and liquid funds for short-term cash parking. The best fund is the one that fits your specific goal, horizon, and risk appetite.",
                    "There's no universally 'best' fund — it depends entirely on you. For retirement 20 years away? Large-cap or flexi-cap equity. For a house down payment in 3 years? Short-duration debt. For emergency funds? Liquid fund. Tell me your goal and I'll narrow it down.",
                    "Fund selection in 3 steps: 1) Define the goal and timeline 2) Match to a category (equity/debt/hybrid) 3) Within that category, pick funds with consistent 5-year track records and low expense ratios.",
                ),
                follow_up="Tell me the specific goal — tax saving, retirement, or parking cash — and I'll recommend specific categories.",
            ),
            BuddyKnowledgeItem(
                title="Tax saving basics",
                audience="investor",
                tags=("tax", "elss", "80c", "tax saving"),
                answers=(
                    "ELSS (Equity Linked Savings Scheme) combines tax benefits under Section 80C with long-term equity growth potential. The 3-year lock-in is the shortest among tax-saving instruments. But remember — choose ELSS for the investment thesis first, tax break second.",
                    "Tax-saving through ELSS is smart if your investment horizon is 3+ years anyway. You get up to ₹46,800 in tax savings (on ₹1.5L investment at 30% bracket) while staying invested in diversified equity. Start an ELSS SIP to spread out both investment and tax-saving.",
                    "ELSS vs PPF vs NPS for tax saving? ELSS has the shortest lock-in (3 years) and highest return potential (but with equity risk). PPF is safe but locked for 15 years. NPS has the best tax benefit but locks until retirement. Pick based on your time flexibility.",
                ),
                follow_up="I can compare ELSS with other tax-saving options for your situation.",
            ),
            BuddyKnowledgeItem(
                title="General first response",
                audience="both",
                tags=("hello", "help", "what can you do", "general", "hi"),
                answers=(
                    "I'm your Lume AI financial assistant. I can help with portfolio analysis, fund recommendations, SIP planning, risk assessment, market sentiment, lead management, and compliance guidance. What would you like to explore?",
                    "Welcome! I'm here to help with your financial questions — from portfolio optimization and fund selection to market analysis and compliance checks. Ask me anything specific, and I'll give you a data-driven answer.",
                    "Hi! I can assist with investment strategy, portfolio review, fund comparisons, risk analysis, and market outlook. The more specific your question, the better my answer. What's on your mind?",
                ),
                follow_up="Try asking about SIPs, fund selection, portfolio risk, or market outlook.",
            ),
        ]

    @staticmethod
    def _item_text(item: BuddyKnowledgeItem) -> str:
        return " ".join((item.title, item.audience, " ".join(item.tags), item.answers[0], item.follow_up)).lower()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _count_hits(text: str, terms: Iterable[str]) -> int:
        return sum(1 for term in terms if term in text)

    def _load_transformer(self) -> Any | None:
        if os.getenv("LUME_ENABLE_BUDDY_TRANSFORMER", "0") != "1":
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("LUME_BUDDY_MODEL", "all-MiniLM-L6-v2")
            return SentenceTransformer(model_name)
        except Exception:
            return None

    def _audience_from_text(self, text: str, distributor_id: str | None, lead_id: str | None) -> str:
        if distributor_id or lead_id:
            return "distributor"
        investor_terms = ("sip", "mutual fund", "portfolio", "risk", "retirement", "tax", "invest", "lumpsum")
        distributor_terms = ("lead", "objection", "arn", "amfi", "compliance", "nism", "call script", "pitch")
        inv_hits = self._count_hits(text, investor_terms)
        dist_hits = self._count_hits(text, distributor_terms)
        if dist_hits > inv_hits:
            return "distributor"
        if inv_hits > dist_hits:
            return "investor"
        return "both"

    def _intent_from_text(self, text: str) -> str:
        intent_rules = (
            ("compliance", ("compliance", "sebi", "amfi", "kyc", "disclosure", "suitability")),
            ("objection", ("objection", "risk", "volatility", "safe", "panic")),
            ("lead_management", ("lead", "crm", "follow up", "hot lead", "warm lead")),
            ("sip", ("sip", "systematic investment plan")),
            ("tax", ("tax", "elss", "80c")),
            ("portfolio", ("portfolio", "asset allocation", "diversification", "rebalance")),
            ("fund_selection", ("best fund", "fund type", "recommend fund", "equity", "debt", "hybrid", "liquid")),
            ("market_timing", ("buy now", "should i invest", "lumpsum", "market timing", "now", "small cap", "right now")),
        )
        for intent, terms in intent_rules:
            if self._count_hits(text, terms):
                return intent
        return "general"

    def _candidate_items(self, audience: str) -> list[BuddyKnowledgeItem]:
        if audience == "distributor":
            return [item for item in self.knowledge_base if item.audience in {"distributor", "both"}]
        if audience == "investor":
            return [item for item in self.knowledge_base if item.audience in {"investor", "both"}]
        return list(self.knowledge_base)

    def _score_with_tfidf(self, query: str, items: list[BuddyKnowledgeItem]) -> tuple[BuddyKnowledgeItem | None, float]:
        if not items:
            return None, 0.0
        texts = [self._item_text(item) for item in items]
        query_vec = self.vectorizer.transform([query])
        matrix = self.vectorizer.transform(texts)
        scores = cosine_similarity(query_vec, matrix)[0]
        best_index = int(np.argmax(scores))
        return items[best_index], float(scores[best_index])

    def _score_with_transformer(self, query: str, items: list[BuddyKnowledgeItem]) -> tuple[BuddyKnowledgeItem | None, float]:
        if not self.transformer or not items:
            return None, 0.0
        try:
            query_vec = self.transformer.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            item_vecs = self.transformer.encode([self._item_text(item) for item in items], convert_to_numpy=True, normalize_embeddings=True)
            scores = (query_vec @ item_vecs.T)[0]
            best_index = int(np.argmax(scores))
            return items[best_index], float(scores[best_index])
        except Exception:
            return None, 0.0

    def _pick_variant(self, item: BuddyKnowledgeItem) -> str:
        """Pick a response variant that hasn't been used recently."""
        available = list(item.answers)
        # Try to avoid the last few responses
        for ans in available:
            h = hashlib.md5(ans[:60].encode()).hexdigest()[:8]
            if h not in self._recent_hashes[-6:]:
                self._recent_hashes.append(h)
                if len(self._recent_hashes) > 20:
                    self._recent_hashes = self._recent_hashes[-10:]
                return ans
        # All variants used recently — pick random
        choice = random.choice(available)
        return choice

    def _compose_contextual_response(
        self,
        query: str,
        dashboard_context: str,
        audience: str,
        persona: str | None,
        market_context: str | None,
    ) -> BuddyReply:
        """Generate a response grounded in the actual dashboard data."""
        ctx = dashboard_context.strip()
        query_lower = query.lower()

        # Build a data-grounded explanation
        parts: list[str] = []

        if "graph" in query_lower or "chart" in query_lower or "explain" in query_lower or "screen" in query_lower or "show" in query_lower:
            parts.append("Here's what's currently displayed on your dashboard:\n\n")
            parts.append(ctx)
            parts.append("\n\nIn summary: these visualizations help you understand your portfolio's position relative to market benchmarks, risk-return tradeoffs, and allocation balance.")
        elif "kpi" in query_lower or "metric" in query_lower or "number" in query_lower:
            parts.append("Looking at your current dashboard metrics:\n\n")
            parts.append(ctx)
        elif "portfolio" in query_lower or "holding" in query_lower:
            parts.append("Based on the data visible on your dashboard:\n\n")
            parts.append(ctx)
            if persona:
                parts.append(f"\n\nThis is aligned with your {persona} risk profile.")
        else:
            # General context-aware answer
            parts.append("Based on your current dashboard data:\n\n")
            parts.append(ctx)

        if market_context:
            parts.append(f"\n\nCurrent market: {market_context}.")

        response = "".join(parts)

        return BuddyReply(
            response=response,
            confidence=0.88,
            category="dashboard_context",
            audience=audience,
            matched_topic="Dashboard Explanation",
            follow_up="Want me to explain any specific chart in more detail, or suggest optimizations?",
        )

    def _compose_response(
        self,
        query: str,
        item: BuddyKnowledgeItem | None,
        score: float,
        audience: str,
        intent: str,
        persona: str | None,
        history: list[dict[str, str]] | None,
        market_context: str | None,
        reference_funds: list[dict[str, Any]] | None,
    ) -> BuddyReply:
        # Natural language intro — no robotic prefix
        persona_phrase = f" (based on your {persona} risk profile)" if persona else ""

        if item is not None:
            response = self._pick_variant(item)
            follow_up = item.follow_up
            category = item.title
        else:
            response = (
                "I can help with SIPs, fund selection, portfolio review, compliance, lead handling, and market timing."
                " Try narrowing your question to a specific goal, fund category, or scenario for a more targeted answer."
            )
            follow_up = "Tell me the goal, user type, and timeframe, and I'll tailor the answer."
            category = "general"

        # Add persona context naturally
        if persona and item and item.audience in ("investor", "both"):
            response += persona_phrase + "."

        # Add market context
        if market_context:
            response += f" Current market: {market_context}."

        # Add fund references
        if reference_funds:
            names = []
            for fund in reference_funds[:3]:
                name = fund.get("scheme_name") or fund.get("name") or fund.get("fund_name")
                if name:
                    names.append(str(name))
            if names:
                response += f" Relevant funds: {', '.join(names)}."

        if intent == "compliance":
            response += " (This is guidance, not legal advice — verify the latest SEBI/AMFI circulars.)"

        if item and item.follow_up:
            response += f" {item.follow_up}"

        confidence = min(0.97, 0.42 + score * 2.0)
        if item is None:
            confidence = 0.52 if intent == "general" else 0.60
        if audience != "both" and item and item.audience == audience:
            confidence += 0.05
        confidence = float(round(min(confidence, 0.97), 2))

        return BuddyReply(
            response=response,
            confidence=confidence,
            category=category,
            audience=audience,
            matched_topic=item.title if item else "general",
            follow_up=follow_up,
        )

    def generate(
        self,
        query: str,
        audience: str | None = None,
        persona: str | None = None,
        history: list[dict[str, str]] | None = None,
        distributor_id: str | None = None,
        lead_id: str | None = None,
        market_context: str | None = None,
        reference_funds: list[dict[str, Any]] | None = None,
        dashboard_context: str | None = None,
    ) -> BuddyReply:
        query_text = self._normalize(query)
        inferred_audience = audience or self._audience_from_text(query_text, distributor_id, lead_id)

        # ── Context-aware path: if user asks about charts/screen and context is provided ──
        if dashboard_context and _is_context_question(query_text):
            return self._compose_contextual_response(
                query=query_text,
                dashboard_context=dashboard_context,
                audience=inferred_audience,
                persona=persona,
                market_context=market_context,
            )

        # ── Standard knowledge-base path ──
        intent = self._intent_from_text(query_text)
        candidates = self._candidate_items(inferred_audience)

        transformer_item, transformer_score = self._score_with_transformer(query_text, candidates)
        tfidf_item, tfidf_score = self._score_with_tfidf(query_text, candidates)

        if transformer_item is not None and transformer_score >= tfidf_score:
            best_item, best_score = transformer_item, transformer_score
        else:
            best_item, best_score = tfidf_item, tfidf_score

        if best_item is not None:
            tag_hits = self._count_hits(query_text, best_item.tags)
            if tag_hits:
                best_score = min(1.0, best_score + 0.04 * tag_hits)

        if intent == "general" and best_score < 0.12:
            best_item = next((item for item in candidates if item.title == "General first response"), best_item)

        return self._compose_response(
            query=query_text,
            item=best_item,
            score=best_score,
            audience=inferred_audience,
            intent=intent,
            persona=persona,
            history=history,
            market_context=market_context,
            reference_funds=reference_funds,
        )
