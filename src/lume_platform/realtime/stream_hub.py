"""
Real-Time Streaming Hub — Live WebSocket broadcaster & streaming AI pipeline.

On each tick:
1. Steps live / anchored NIFTY 50, VIX, and sector metrics (`LiveRiskAnalyzer`).
2. Fetches latest financial headlines (`NewsFetcher`) and scores them on-the-fly
   through the Phase 5 `FinancialTransformerClassifier` (getting class probabilities
   + token attention saliency).
3. Ingests scored events into `ModelRegistry.latest_signals`, `DriftMonitor`, and
   persists JSONL events into `streaming_output/market_signals.jsonl`.
4. Broadcasts typed JSON ticks to all connected WebSocket clients at `/ws/realtime`.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import WebSocket

from lume_platform.config import EXPORT_DIR, PROJECT_ROOT


class RealtimeStreamHub:
    """Manages connected WebSocket clients and live AI stream ticks."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = threading.Lock()
        self._tick_seq: int = 0
        self._news_cursor: int = 0
        self._started_at: str = datetime.utcnow().isoformat()
        self._last_tick_at: Optional[str] = None
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._recent_events: Deque[Dict[str, Any]] = deque(maxlen=30)
        self._jsonl_path: Path = Path(
            os.environ.get(
                "STREAMING_FALLBACK_OUT",
                str(PROJECT_ROOT / "streaming_output" / "market_signals.jsonl"),
            )
        )
        if "VERCEL" in os.environ:
            self._jsonl_path = Path("/tmp") / "market_signals.jsonl"
        self._task: Optional[asyncio.Task] = None
        self._registry = None
        self._drift_monitor = None

    def bind(self, registry: Any, drift_monitor: Any) -> None:
        """Bind shared API singletons so ticks update registry & drift state directly."""
        self._registry = registry
        self._drift_monitor = drift_monitor

    @property
    def connected_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def register_client(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.add(ws)

    def unregister_client(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)

    def _next_headline(self) -> Dict[str, Any]:
        """Pull live headline from NewsFetcher (non-blocking cache / stale fallback)."""
        try:
            from lume_platform.news.news_fetcher import news_fetcher
            articles = news_fetcher.get_live_news(max_articles=12)
        except Exception:
            articles = []

        if not articles:
            articles = [
                {
                    "title": "NIFTY 50 banking and mid-cap flows show steady domestic SIP participation",
                    "source": "Lume Stream Feed",
                    "link": "#",
                },
                {
                    "title": "RBI liquidity commentary stabilizes short-term corporate debt yields",
                    "source": "Lume Stream Feed",
                    "link": "#",
                },
                {
                    "title": "Global tech volatility drives defensive rotation into hybrid funds",
                    "source": "Lume Stream Feed",
                    "link": "#",
                },
            ]

        idx = self._news_cursor % len(articles)
        self._news_cursor += 1
        return articles[idx]

    def _score_headline_nlp(self, text: str) -> Dict[str, Any]:
        """Score headline using Phase 5 FinancialTransformerClassifier + saliency."""
        try:
            from lume_platform.nlp.financial_nlp_service import FinancialNLPService
            svc = FinancialNLPService.get_instance()
            res = svc.predict_sentiment_with_saliency(text)
            saliency = sorted(
                res.get("token_saliency", []),
                key=lambda x: x.get("saliency", 0.0),
                reverse=True,
            )[:5]
            return {
                "sentiment": res.get("sentiment", "neutral"),
                "confidence": float(res.get("confidence", 0.5)),
                "probabilities": res.get("probabilities", {}),
                "token_saliency": saliency,
                "model": res.get("model", "FinancialTransformer"),
                "latency_ms": float(res.get("latency_ms", 0.0)),
            }
        except Exception:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "probabilities": {"negative": 0.25, "neutral": 0.5, "positive": 0.25},
                "token_saliency": [],
                "model": "Fallback",
                "latency_ms": 0.1,
            }

    def emit_tick(self, persist_jsonl: bool = True) -> Dict[str, Any]:
        """Advance market state, score latest news through Transformer, update drift & return packet."""
        from lume_platform.risk.live_risk_analyzer import risk_analyzer

        md = risk_analyzer.step_realtime_tick()
        article = self._next_headline()
        headline = article.get("title", "Indian market equities track global cues")
        nlp = self._score_headline_nlp(headline)

        now_iso = datetime.utcnow().isoformat()
        with self._lock:
            self._tick_seq += 1
            seq = self._tick_seq
            self._last_tick_at = now_iso

        # Sentiment label normalization for market_snapshot UI gauge
        change_pct = float(md.nifty_change_pct)
        if change_pct > 1.5:
            sentiment_label = "Very Bullish"
            sentiment_score = round(85.0 + (min(change_pct, 5.0) - 1.5) * 4.2, 1)
        elif change_pct > 0.5:
            sentiment_label = "Bullish"
            sentiment_score = round(40.0 + (change_pct - 0.5) * 45.0, 1)
        elif change_pct < -1.5:
            sentiment_label = "Highly Bearish"
            sentiment_score = round(-85.0 + (max(change_pct, -5.0) + 1.5) * 4.2, 1)
        elif change_pct < -0.5:
            sentiment_label = "Bearish"
            sentiment_score = round(-40.0 + (change_pct + 0.5) * 45.0, 1)
        else:
            sentiment_label = "Neutral"
            sentiment_score = round(change_pct * 80.0, 1)
        sentiment_score = max(-100.0, min(100.0, sentiment_score))
        vol_score = max(0.0, min(100.0, round(md.vix * 2.5, 1)))
        risk_intensity = max(0.0, min(100.0, round((vol_score * 0.6) + (abs(sentiment_score) * 0.4), 1)))

        signal_event = {
            "event_id": f"RT-{seq:05d}",
            "tick_seq": int(seq),
            "timestamp": now_iso,
            "headline": headline,
            "source": article.get("source", "Market Wire"),
            "link": article.get("link", "#"),
            "sentiment": str(nlp["sentiment"]),
            "confidence": round(float(nlp["confidence"]), 4),
            "probabilities": {str(k): float(v) for k, v in nlp["probabilities"].items()},
            "token_saliency": nlp["token_saliency"],
            "model": str(nlp["model"]),
            "latency_ms": float(nlp["latency_ms"]),
            "market_impact": str(md.sentiment),
            "nifty_50": float(md.nifty_50),
            "nifty_change_pct": float(md.nifty_change_pct),
            "vix": float(md.vix),
        }

        with self._lock:
            self._recent_events.appendleft(signal_event)
            recent_list = list(self._recent_events)[:10]

        # Push into registry & drift monitor
        if self._registry is not None:
            try:
                self._registry.add_market_signal(signal_event)
            except Exception:
                pass
        if self._drift_monitor is not None:
            try:
                self._drift_monitor.ingest([signal_event])
            except Exception:
                pass

        # Persist to streaming JSONL log
        if persist_jsonl:
            try:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(signal_event) + "\n")
            except Exception:
                pass

        models_status = (
            self._registry.get_models_status()
            if self._registry is not None and hasattr(self._registry, "get_models_status")
            else {}
        )
        core_keys = (
            "lead_bundle",
            "investor_bundle",
            "semantic_search",
            "lead_mlp",
            "bilstm_forecaster",
            "vision_resnet18",
            "financial_transformer",
        )
        models_online = sum(1 for k in core_keys if models_status.get(k))

        drift_dict = (
            self._drift_monitor.status().__dict__
            if self._drift_monitor is not None
            else {
                "drift_detected": False,
                "samples_seen": seq,
                "mean_score": 0.0,
                "negative_ratio": 0.0,
                "last_message": "idle",
            }
        )

        packet = {
            "type": "tick",
            "tick_seq": seq,
            "timestamp": now_iso,
            "market": {
                "timestamp": md.timestamp.isoformat(),
                "nifty_50": float(md.nifty_50),
                "nifty_change_pct": float(md.nifty_change_pct),
                "vix": float(md.vix),
                "market_status": str(md.market_status),
                "market_sentiment": str(sentiment_label),
                "sentiment_score": float(sentiment_score),
                "market_volatility_score": float(vol_score),
                "risk_intensity_score": float(risk_intensity),
                "volume_spike": bool(md.volume_spike),
                "sector_performance": {str(k): float(v) for k, v in md.sector_performance.items()},
                "top_gainers": [[str(s), float(p)] for s, p in md.top_gainers],
                "top_losers": [[str(s), float(p)] for s, p in md.top_losers],
                "data_source": str(md.data_source),
            },
            "latest_signal": signal_event,
            "recent_signals": recent_list,
            "drift": drift_dict,
            "models_online": models_online,
            "models_total": len(core_keys),
            "models_status": models_status,
            "stream_metrics": {
                "live_ticks_emitted": seq,
                "connected_clients": self.connected_count,
                "tick_interval_seconds": float(os.environ.get("LUME_REALTIME_TICK_INTERVAL", "4.0")),
                "started_at": self._started_at,
                "last_tick_at": now_iso,
            },
        }
        with self._lock:
            self._last_snapshot = packet
        return packet

    def get_snapshot(self) -> Dict[str, Any]:
        """Return latest real-time snapshot (emits initial tick if none emitted yet)."""
        with self._lock:
            snap = self._last_snapshot
        if snap is None:
            snap = self.emit_tick(persist_jsonl=False)
        else:
            snap = {
                **snap,
                "stream_metrics": {
                    **snap.get("stream_metrics", {}),
                    "connected_clients": self.connected_count,
                },
            }
        return snap

    async def broadcast_packet(self, packet: Dict[str, Any]) -> None:
        """Broadcast JSON packet to all active WebSocket connections."""
        with self._lock:
            targets = list(self._clients)
        if not targets:
            return
        dead: List[WebSocket] = []
        payload_text = json.dumps(packet)
        for ws in targets:
            try:
                await ws.send_text(payload_text)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    async def run_loop(self) -> None:
        """Asyncio background loop emitting ticks & broadcasting over WebSockets."""
        interval = max(1.0, float(os.environ.get("LUME_REALTIME_TICK_INTERVAL", "4.0")))
        while True:
            try:
                await asyncio.sleep(interval)
                packet = await asyncio.to_thread(self.emit_tick, True)
                await self.broadcast_packet(packet)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"⚠️ RealtimeStreamHub tick error: {exc}")
                await asyncio.sleep(2.0)


stream_hub = RealtimeStreamHub()
