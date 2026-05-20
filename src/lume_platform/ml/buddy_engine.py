"""Hybrid NLP engine for the AI Buddy chat surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    answer: str
    follow_up: str


@dataclass(frozen=True)
class BuddyReply:
    response: str
    confidence: float
    category: str
    audience: str
    matched_topic: str
    follow_up: str


class BuddyEngine:
    """Semantic retrieval plus intent routing for distributor and investor chat."""

    def __init__(self) -> None:
        self.knowledge_base = self._build_knowledge_base()
        self._corpus = [self._item_text(item) for item in self.knowledge_base]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self._corpus)
        self.transformer = self._load_transformer()

    def _build_knowledge_base(self) -> list[BuddyKnowledgeItem]:
        return [
            BuddyKnowledgeItem(
                title="Distributor onboarding and basics",
                audience="distributor",
                tags=("getting started", "arn", "amfi", "nism", "basic"),
                answer=(
                    "Start with NISM certification, complete AMFI registration, and keep the conversation centered on"
                    " investor goals, risk appetite, and suitability. Build trust first, then recommend products that match"
                    " the client profile rather than pushing a single fund."
                ),
                follow_up="If you want, I can break this into a 7-day onboarding checklist.",
            ),
            BuddyKnowledgeItem(
                title="Lead handling priorities",
                audience="distributor",
                tags=("lead", "hot lead", "warm lead", "follow up", "crm"),
                answer=(
                    "Handle hot leads within an hour, warm leads within a day, and cold leads with a nurture plan."
                    " Personalize the pitch using the investor's horizon, risk profile, and recent market concerns."
                ),
                follow_up="I can also turn this into a call script or a CRM follow-up plan.",
            ),
            BuddyKnowledgeItem(
                title="Objection handling",
                audience="distributor",
                tags=("objection", "risk", "not now", "market fall", "volatility"),
                answer=(
                    "Acknowledge the objection, restate the goal, and explain the trade-off in plain language."
                    " For volatility concerns, SIPs reduce timing risk because purchases are spread across multiple market levels."
                ),
                follow_up="Share the exact objection and I will draft a response you can say on the call.",
            ),
            BuddyKnowledgeItem(
                title="Compliance and suitability",
                audience="distributor",
                tags=("compliance", "sebi", "amfi", "kyc", "suitability", "disclosure"),
                answer=(
                    "Always complete KYC, disclose fees and commissions, and document why the recommendation fits the"
                    " client's risk profile and horizon. Never promise guaranteed returns or push products without"
                    " suitability checks."
                ),
                follow_up="I can give you a compliance-safe script for the same question.",
            ),
            BuddyKnowledgeItem(
                title="What SIP means",
                audience="investor",
                tags=("sip", "systematic investment plan", "monthly investing"),
                answer=(
                    "A SIP is a disciplined way to invest a fixed amount regularly, usually monthly. It reduces the"
                    " pressure of market timing and helps average out purchase price over time."
                ),
                follow_up="If you share your time horizon and risk level, I can suggest a suitable SIP style.",
            ),
            BuddyKnowledgeItem(
                title="Risk and asset allocation",
                audience="investor",
                tags=("risk", "asset allocation", "diversification", "portfolio"),
                answer=(
                    "The right mix depends on your goal, time horizon, and how much volatility you can tolerate."
                    " Longer horizons can absorb more equity risk, while shorter goals should lean toward liquid or debt"
                    " exposure."
                ),
                follow_up="Tell me your goal and I will translate it into a simple allocation framework.",
            ),
            BuddyKnowledgeItem(
                title="Market timing and lumpsum investing",
                audience="investor",
                tags=("market timing", "lumpsum", "buy now", "volatility", "should i invest now"),
                answer=(
                    "If uncertainty is high, stagger entries with SIP or a staggered lumpsum plan. The goal is not to"
                    " predict every move but to stay invested in a way you can follow consistently."
                ),
                follow_up="I can suggest a conservative, balanced, or growth-oriented entry approach.",
            ),
            BuddyKnowledgeItem(
                title="Portfolio review",
                audience="both",
                tags=("portfolio review", "review my portfolio", "rebalance", "allocation"),
                answer=(
                    "Review holdings against the original goal, horizon, and risk level. Rebalance when one segment grows"
                    " too large or when the investor's objective changes, rather than reacting to every market headline."
                ),
                follow_up="I can review either an investor portfolio or a distributor lead portfolio.",
            ),
            BuddyKnowledgeItem(
                title="Fund category guidance",
                audience="both",
                tags=("mutual fund", "best fund", "recommend fund", "fund type", "equity", "debt", "hybrid"),
                answer=(
                    "Match the fund category to the use case: equity for long-term growth, debt for stability, hybrid for"
                    " balance, and liquid funds for short-term parking. The best fund is the one that fits the investor's"
                    " goal, horizon, and risk appetite."
                ),
                follow_up="If you want, I can narrow this to a specific goal like tax saving, retirement, or parking cash.",
            ),
            BuddyKnowledgeItem(
                title="Tax saving basics",
                audience="investor",
                tags=("tax", "elss", "80c", "tax saving"),
                answer=(
                    "Tax-saving mutual funds such as ELSS can help long-term investors combine growth potential with"
                    " Section 80C benefits. They still carry equity risk, so they should be chosen for the right horizon"
                    " and not just the tax break."
                ),
                follow_up="I can compare ELSS with other tax-saving choices if needed.",
            ),
            BuddyKnowledgeItem(
                title="General first response",
                audience="both",
                tags=("hello", "help", "what can you do", "general"),
                answer=(
                    "I can help with investor questions, distributor workflows, compliance, SIPs, portfolio reviews,"
                    " fund categories, and market timing. Ask in plain language and I will route the question to the"
                    " right financial context."
                ),
                follow_up="Try asking about SIPs, fund selection, lead handling, or compliance.",
            ),
        ]

    @staticmethod
    def _item_text(item: BuddyKnowledgeItem) -> str:
        return " ".join((item.title, item.audience, " ".join(item.tags), item.answer, item.follow_up)).lower()

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
            ("market_timing", ("buy now", "should i invest", "lumpsum", "market timing", "now")),
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
        audience_label = "distributor" if audience == "distributor" else "investor" if audience == "investor" else "finance"
        persona_text = f" for a {persona} risk profile" if persona else ""
        intro = f"For {audience_label} questions{persona_text}, "

        if item is not None:
            response = intro + item.answer
            follow_up = item.follow_up
            category = item.title
        else:
            response = intro + (
                "I can help with SIPs, fund selection, portfolio review, compliance, lead handling, and market timing."
                " If you want, narrow the question to one goal, one fund category, or one objection and I will answer"
                " directly."
            )
            follow_up = "Tell me the goal, user type, and timeframe, and I will tailor the answer."
            category = "general"

        if history:
            recent_user_messages = [msg.get("content", "") for msg in history[-2:] if msg.get("role") == "user"]
            if recent_user_messages:
                response += f" I am also keeping the latest context in view: {recent_user_messages[-1][:120]}."

        if market_context:
            response += f" Current market context: {market_context}."

        if reference_funds:
            names = []
            for fund in reference_funds[:3]:
                name = fund.get("name") or fund.get("fund_name") or fund.get("distributor_name")
                if name:
                    names.append(str(name))
            if names:
                response += f" Matching options I found: {', '.join(names)}."

        if intent == "compliance":
            response += " This is guidance, not legal advice; always verify the latest SEBI or AMFI circulars."

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
    ) -> BuddyReply:
        query_text = self._normalize(query)
        inferred_audience = audience or self._audience_from_text(query_text, distributor_id, lead_id)
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
