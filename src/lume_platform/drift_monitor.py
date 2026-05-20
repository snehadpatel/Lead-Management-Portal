"""Automatic drift monitoring and retraining trigger utilities.

This module tracks recent market-signal sentiment, detects sustained drift,
and can trigger a retraining callback when the signal distribution changes.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, Optional


@dataclass
class DriftStatus:
    drift_detected: bool
    retrain_in_progress: bool
    retrain_requested: bool
    samples_seen: int
    mean_score: float
    negative_ratio: float
    last_signal_at: Optional[str]
    last_retrain_at: Optional[str]
    retrain_count: int
    last_message: str


class DriftMonitor:
    """Tracks a rolling signal window and triggers retraining on sustained drift."""

    def __init__(
        self,
        window_size: int = 50,
        min_samples: int = 12,
        drift_threshold: float = -0.20,
        negative_ratio_threshold: float = 0.60,
        cooldown_seconds: int = 1800,
    ) -> None:
        self.window: Deque[float] = deque(maxlen=window_size)
        self.min_samples = min_samples
        self.drift_threshold = drift_threshold
        self.negative_ratio_threshold = negative_ratio_threshold
        self.cooldown_seconds = cooldown_seconds

        self.retrain_requested = False
        self.retrain_in_progress = False
        self.retrain_count = 0
        self.last_signal_at: Optional[str] = None
        self.last_retrain_at: Optional[str] = None
        self.last_message = "idle"

        self._seen_signatures: Deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()

    def _score_signal(self, signal: Dict[str, Any]) -> float:
        sentiment = str(signal.get("sentiment", signal.get("market_sentiment", signal.get("severity", "neutral")))).lower()
        impact = str(signal.get("market_impact", signal.get("impact", signal.get("severity", "neutral")))).lower()

        if sentiment in {"positive", "bullish", "up", "positive sentiment"}:
            base = 1.0
        elif sentiment in {"negative", "bearish", "down", "critical", "high"}:
            base = -1.0
        else:
            base = 0.0

        if impact in {"positive", "bullish"}:
            base += 0.25
        elif impact in {"negative", "bearish", "critical", "high"}:
            base -= 0.25

        conf = signal.get("confidence", signal.get("score", 0.5))
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        return max(-1.0, min(1.0, base * max(0.3, min(conf, 1.0))))

    def _signature(self, signal: Dict[str, Any]) -> str:
        raw = "|".join(str(signal.get(k, "")) for k in ("title", "headline", "source", "sentiment", "market_impact", "timestamp"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def ingest(self, signals: Iterable[Dict[str, Any]]) -> int:
        """Ingest new signals and update the rolling drift window.

        Returns the number of new unique signals processed.
        """
        new_count = 0
        with self._lock:
            for signal in signals or []:
                if not isinstance(signal, dict):
                    continue
                sig = self._signature(signal)
                if sig in self._seen_signatures:
                    continue
                self._seen_signatures.append(sig)
                self.window.append(self._score_signal(signal))
                self.last_signal_at = datetime.utcnow().isoformat()
                new_count += 1
        return new_count

    def detect_drift(self) -> bool:
        with self._lock:
            return self._detect_drift_unlocked()

    def _detect_drift_unlocked(self) -> bool:
        if len(self.window) < self.min_samples:
            return False
        mean_score = sum(self.window) / len(self.window)
        negative_ratio = sum(1 for s in self.window if s < 0) / len(self.window)
        return mean_score <= self.drift_threshold or negative_ratio >= self.negative_ratio_threshold

    def _can_retrain_now(self) -> bool:
        if self.retrain_in_progress:
            return False
        if not self.last_retrain_at:
            return True
        try:
            last = datetime.fromisoformat(self.last_retrain_at)
            return (datetime.utcnow() - last).total_seconds() >= self.cooldown_seconds
        except Exception:
            return True

    def request_retrain(self, retrain_callback: Callable[[], Dict[str, Any]]) -> bool:
        """Trigger retraining in a background thread if cooldown allows."""
        with self._lock:
            if not self._detect_drift_unlocked():
                self.last_message = "drift not strong enough to retrain"
                return False
            if not self._can_retrain_now():
                self.last_message = "cooldown active"
                return False
            if self.retrain_in_progress:
                self.last_message = "retrain already in progress"
                return False

            self.retrain_requested = True
            self.retrain_in_progress = True
            self.last_message = "retrain requested"

        def _run():
            try:
                retrain_callback()
                with self._lock:
                    self.retrain_count += 1
                    self.last_retrain_at = datetime.utcnow().isoformat()
                    self.last_message = "retrain completed"
            except Exception as exc:
                with self._lock:
                    self.last_message = f"retrain failed: {exc}"
            finally:
                with self._lock:
                    self.retrain_in_progress = False
                    self.retrain_requested = False

        threading.Thread(target=_run, daemon=True).start()
        return True

    def status(self) -> DriftStatus:
        with self._lock:
            mean_score = sum(self.window) / len(self.window) if self.window else 0.0
            negative_ratio = sum(1 for s in self.window if s < 0) / len(self.window) if self.window else 0.0
            return DriftStatus(
                drift_detected=(len(self.window) >= self.min_samples and (mean_score <= self.drift_threshold or negative_ratio >= self.negative_ratio_threshold)),
                retrain_in_progress=self.retrain_in_progress,
                retrain_requested=self.retrain_requested,
                samples_seen=len(self.window),
                mean_score=round(mean_score, 4),
                negative_ratio=round(negative_ratio, 4),
                last_signal_at=self.last_signal_at,
                last_retrain_at=self.last_retrain_at,
                retrain_count=self.retrain_count,
                last_message=self.last_message,
            )
