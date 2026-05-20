"""Minimal monitoring and retraining trigger utilities.

Tracks simple counters and a rolling window of sentiment rates to detect drift.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque


class Monitor:
    def __init__(self, window_size: int = 100):
        self.total_requests = 0
        self.errors = 0
        self.latencies = []
        self.sentiment_window: Deque[float] = deque(maxlen=window_size)

    def observe_request(self, latency: float, success: bool = True):
        self.total_requests += 1
        if not success:
            self.errors += 1
        self.latencies.append(latency)

    def observe_sentiment_score(self, score: float):
        # score normalized -1..1
        self.sentiment_window.append(score)

    def average_latency(self) -> float:
        return sum(self.latencies[-100:]) / max(1, min(len(self.latencies), 100))

    def sentiment_mean(self) -> float:
        if not self.sentiment_window:
            return 0.0
        return sum(self.sentiment_window) / len(self.sentiment_window)

    def check_drift(self, baseline: float = 0.0, threshold: float = 0.3) -> bool:
        # Simple drift if sentiment_mean deviates from baseline by threshold
        mean = self.sentiment_mean()
        return abs(mean - baseline) > threshold
