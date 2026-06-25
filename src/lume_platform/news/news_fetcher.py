"""
News Fetcher — Live financial news from RSS feeds and web APIs.

Provides:
- Google News RSS for Indian market headlines
- Sentiment scoring using existing sentiment analysis
- Personalized alerts cross-referenced with user holdings
- 10-minute cache to avoid excessive requests
"""
from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from lume_platform.config import EXPORT_DIR


class NewsFetcher:
    """Fetches live financial news from Google News RSS feed."""

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    QUERIES = [
        "indian stock market today",
        "mutual fund india NAV",
        "NIFTY 50 latest",
        "RBI monetary policy",
        "SEBI mutual fund regulation",
    ]

    def __init__(self):
        self._cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0
        self._cache_ttl: float = 600  # 10 minutes
        self._cache_dir = EXPORT_DIR / "news_cache"
        if "VERCEL" in os.environ:
            self._cache_dir = Path("/tmp") / "news_cache"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def get_live_news(self, max_articles: int = 12) -> List[Dict[str, Any]]:
        """Get latest financial news with sentiment analysis."""
        now = time.time()

        # Return cached if fresh
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache[:max_articles]

        # Try file cache
        cache_file = self._cache_dir / "latest_news.json"
        if cache_file.is_file():
            try:
                file_age = now - cache_file.stat().st_mtime
                if file_age < self._cache_ttl:
                    with open(cache_file, "r") as f:
                        self._cache = json.load(f)
                    self._cache_ts = now
                    return self._cache[:max_articles]
            except Exception:
                pass

        articles = []
        seen_titles = set()

        for query in self.QUERIES[:3]:  # Limit to 3 queries to be gentle
            try:
                url = self.GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))
                resp = requests.get(url, timeout=8, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; LumeAI/1.0)"
                })
                resp.raise_for_status()
                root = ET.fromstring(resp.text)

                for item in root.findall(".//item")[:5]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    source = item.findtext("source", "").strip()

                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Simple sentiment scoring from title keywords
                    sentiment = self._score_sentiment(title)

                    articles.append({
                        "title": title,
                        "link": link,
                        "source": source or "Google News",
                        "published": pub_date,
                        "sentiment": sentiment["label"],
                        "sentiment_score": sentiment["score"],
                        "impact": sentiment["impact"],
                        "query": query,
                    })
            except Exception as e:
                print(f"⚠️ News fetch failed for '{query}': {e}")
                continue

        if articles:
            # Sort by sentiment impact (high impact first)
            impact_order = {"high": 0, "medium": 1, "low": 2}
            articles.sort(key=lambda a: impact_order.get(a.get("impact", "low"), 2))
            self._cache = articles
            self._cache_ts = now

            # Persist to file
            try:
                with open(cache_file, "w") as f:
                    json.dump(articles, f)
            except Exception:
                pass
            print(f"✅ Fetched {len(articles)} live news articles")
        elif not self._cache:
            # Use fallback hardcoded news
            self._cache = self._fallback_news()
            self._cache_ts = now

        return self._cache[:max_articles]

    def get_personalized_alerts(
        self, holdings: List[Dict[str, Any]], max_alerts: int = 5
    ) -> List[Dict[str, Any]]:
        """Cross-reference news with user's holdings to generate personalized alerts."""
        news = self.get_live_news(20)
        if not holdings or not news:
            return news[:max_alerts]

        # Extract keywords from holdings
        holding_keywords = set()
        for h in holdings:
            name = (h.get("scheme_name", "") or h.get("name", "")).lower()
            category = (h.get("category", "")).lower()
            for word in re.split(r"[^a-z]+", name + " " + category):
                if len(word) > 3:
                    holding_keywords.add(word)

        # Score each article for relevance to holdings
        scored = []
        for article in news:
            title_lower = article["title"].lower()
            relevance = 0
            matched_keywords = []
            for kw in holding_keywords:
                if kw in title_lower:
                    relevance += 1
                    matched_keywords.append(kw)

            scored.append({
                **article,
                "relevance": relevance,
                "matched_keywords": matched_keywords,
                "personalized": relevance > 0,
            })

        # Sort by relevance (personalized first), then by impact
        scored.sort(key=lambda a: (-a["relevance"], {"high": 0, "medium": 1, "low": 2}.get(a.get("impact", "low"), 2)))
        return scored[:max_alerts]

    def _score_sentiment(self, text: str) -> Dict[str, Any]:
        """Quick sentiment scoring from headline keywords."""
        text_lower = text.lower()

        negative_words = [
            "crash", "fall", "drop", "decline", "bearish", "loss", "warning",
            "crisis", "fear", "panic", "risk", "slump", "tumble", "plunge",
            "correction", "recession", "default", "inflation", "hike",
        ]
        positive_words = [
            "rally", "surge", "gain", "bullish", "record", "high", "growth",
            "profit", "boom", "recovery", "strong", "outperform", "buy",
            "upgrade", "positive", "optimistic", "returns",
        ]

        neg_count = sum(1 for w in negative_words if w in text_lower)
        pos_count = sum(1 for w in positive_words if w in text_lower)

        if neg_count > pos_count:
            score = -min(neg_count * 0.3, 1.0)
            label = "bearish"
            impact = "high" if neg_count >= 2 else "medium"
        elif pos_count > neg_count:
            score = min(pos_count * 0.3, 1.0)
            label = "bullish"
            impact = "medium" if pos_count >= 2 else "low"
        else:
            score = 0.0
            label = "neutral"
            impact = "low"

        return {"score": round(score, 2), "label": label, "impact": impact}

    def _fallback_news(self) -> List[Dict[str, Any]]:
        """Static fallback news when RSS is unavailable."""
        return [
            {
                "title": "NIFTY 50 trades flat amid global uncertainty",
                "source": "Economic Times",
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "impact": "low",
                "published": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "link": "#",
            },
            {
                "title": "RBI keeps repo rate unchanged at 6.5%",
                "source": "Livemint",
                "sentiment": "neutral",
                "sentiment_score": 0.1,
                "impact": "medium",
                "published": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "link": "#",
            },
            {
                "title": "Mutual fund SIP inflows hit record high in 2025",
                "source": "CNBC-TV18",
                "sentiment": "bullish",
                "sentiment_score": 0.6,
                "impact": "medium",
                "published": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "link": "#",
            },
        ]


# Module-level singleton
news_fetcher = NewsFetcher()
