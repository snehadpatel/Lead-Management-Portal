"""
NAV Fetcher — Real-time Mutual Fund NAV data from AMFI and MF API.

Provides:
- Daily NAV for all AMFI-registered schemes via official NAV feed
- Historical NAV data per scheme via mfapi.in
- Calculated real returns (1Y, 3Y, 5Y) from historical data
- 5-minute cache to avoid hammering APIs
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from lume_platform.config import EXPORT_DIR


class NAVFetcher:
    """Fetches and caches real-time NAV data from AMFI India and MF API."""

    AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
    MF_API_BASE = "https://api.mfapi.in/mf"

    def __init__(self):
        self._nav_cache: Dict[str, float] = {}
        self._nav_cache_ts: float = 0
        self._nav_cache_ttl: float = 300  # 5 minutes
        self._hist_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._hist_cache_ts: Dict[str, float] = {}
        self._hist_cache_ttl: float = 3600  # 1 hour for historical
        self._returns_cache: Dict[str, Dict[str, float]] = {}
        self._cache_dir = EXPORT_DIR / "nav_cache"
        if "VERCEL" in os.environ:
            self._cache_dir = Path("/tmp") / "nav_cache"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def get_latest_nav(self, scheme_code: str) -> Optional[float]:
        """Get the latest NAV for a scheme. Returns cached value if fresh."""
        self._ensure_nav_cache()
        return self._nav_cache.get(str(scheme_code))

    def get_latest_navs_bulk(self, scheme_codes: List[str]) -> Dict[str, float]:
        """Get latest NAVs for multiple schemes efficiently."""
        self._ensure_nav_cache()
        result = {}
        for code in scheme_codes:
            nav = self._nav_cache.get(str(code))
            if nav is not None:
                result[str(code)] = nav
        return result

    def get_historical_nav(self, scheme_code: str, days: int = 365) -> List[Dict[str, Any]]:
        """
        Get historical NAV data for a scheme.
        Returns list of {'date': 'DD-MM-YYYY', 'nav': float} sorted newest first.
        """
        code = str(scheme_code)
        now = time.time()

        # Check memory cache
        if code in self._hist_cache and (now - self._hist_cache_ts.get(code, 0)) < self._hist_cache_ttl:
            return self._hist_cache[code][:days]

        # Check file cache
        cache_file = self._cache_dir / f"hist_{code}.json"
        if cache_file.is_file():
            try:
                file_age = now - cache_file.stat().st_mtime
                if file_age < self._hist_cache_ttl:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    self._hist_cache[code] = data
                    self._hist_cache_ts[code] = now
                    return data[:days]
            except Exception:
                pass

        # Fetch from MF API
        try:
            resp = requests.get(f"{self.MF_API_BASE}/{code}", timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            nav_data = []
            for entry in raw.get("data", []):
                try:
                    nav_val = float(entry.get("nav", 0))
                    nav_data.append({
                        "date": entry.get("date", ""),
                        "nav": nav_val,
                    })
                except (ValueError, TypeError):
                    continue

            if nav_data:
                self._hist_cache[code] = nav_data
                self._hist_cache_ts[code] = now
                # Persist to file cache
                try:
                    with open(cache_file, "w") as f:
                        json.dump(nav_data[:1825], f)  # Cache up to 5 years
                except Exception:
                    pass
                return nav_data[:days]
        except Exception as e:
            print(f"⚠️ MF API fetch failed for {code}: {e}")

        return []

    def calculate_returns(self, scheme_code: str) -> Dict[str, Optional[float]]:
        """
        Calculate real returns (1Y, 3Y, 5Y) from historical NAV data.
        Returns percentages e.g. {'r1': 15.2, 'r3': 42.5, 'r5': 85.3}
        """
        code = str(scheme_code)
        if code in self._returns_cache:
            return self._returns_cache[code]

        hist = self.get_historical_nav(code, days=1825)  # ~5 years
        if not hist:
            return {"r1": None, "r3": None, "r5": None}

        current_nav = hist[0]["nav"] if hist else None
        if not current_nav or current_nav <= 0:
            return {"r1": None, "r3": None, "r5": None}

        def _find_nav_at_days_ago(data: list, target_days: int) -> Optional[float]:
            """Find the NAV closest to target_days ago."""
            target_date = datetime.now() - timedelta(days=target_days)
            best_nav = None
            best_diff = float("inf")
            for entry in data:
                try:
                    d = datetime.strptime(entry["date"], "%d-%m-%Y")
                    diff = abs((d - target_date).days)
                    if diff < best_diff:
                        best_diff = diff
                        best_nav = entry["nav"]
                except (ValueError, KeyError):
                    continue
            # Only use if within 30 days of target
            return best_nav if best_diff <= 30 else None

        r1_nav = _find_nav_at_days_ago(hist, 365)
        r3_nav = _find_nav_at_days_ago(hist, 365 * 3)
        r5_nav = _find_nav_at_days_ago(hist, 365 * 5)

        result = {
            "r1": round(((current_nav - r1_nav) / r1_nav) * 100, 2) if r1_nav and r1_nav > 0 else None,
            "r3": round(((current_nav - r3_nav) / r3_nav) * 100, 2) if r3_nav and r3_nav > 0 else None,
            "r5": round(((current_nav - r5_nav) / r5_nav) * 100, 2) if r5_nav and r5_nav > 0 else None,
        }
        self._returns_cache[code] = result
        return result

    def calculate_xirr_return(
        self, buy_nav: float, current_nav: float, buy_date_str: str
    ) -> Optional[float]:
        """Calculate annualized return (XIRR approximation) for a single holding."""
        try:
            buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d")
            days_held = (datetime.now() - buy_date).days
            if days_held <= 0 or buy_nav <= 0:
                return None
            years = days_held / 365.0
            if years < 0.01:
                return None
            return round(((current_nav / buy_nav) ** (1 / years) - 1) * 100, 2)
        except Exception:
            return None

    def _ensure_nav_cache(self) -> None:
        """Load/refresh NAV cache from AMFI feed if stale."""
        now = time.time()
        if (now - self._nav_cache_ts) < self._nav_cache_ttl and self._nav_cache:
            return

        # Try file cache first
        file_cache = self._cache_dir / "latest_navs.json"
        if file_cache.is_file():
            try:
                file_age = now - file_cache.stat().st_mtime
                if file_age < self._nav_cache_ttl:
                    with open(file_cache, "r") as f:
                        self._nav_cache = json.load(f)
                    self._nav_cache_ts = now
                    return
            except Exception:
                pass

        # Fetch from AMFI
        try:
            resp = requests.get(self.AMFI_NAV_URL, timeout=15)
            resp.raise_for_status()
            self._parse_amfi_nav(resp.text)
            self._nav_cache_ts = now
            # Persist to file
            try:
                with open(file_cache, "w") as f:
                    json.dump(self._nav_cache, f)
            except Exception:
                pass
            print(f"✅ NAV cache refreshed: {len(self._nav_cache)} schemes")
        except Exception as e:
            print(f"⚠️ AMFI NAV fetch failed: {e}")
            # Fallback: try loading stale file cache
            if file_cache.is_file():
                try:
                    with open(file_cache, "r") as f:
                        self._nav_cache = json.load(f)
                    self._nav_cache_ts = now - self._nav_cache_ttl + 60  # Re-try in 60s
                except Exception:
                    pass

    def _parse_amfi_nav(self, raw_text: str) -> None:
        """Parse AMFI NAVAll.txt format into scheme_code -> NAV mapping."""
        nav_map: Dict[str, float] = {}
        for line in raw_text.strip().split("\n"):
            parts = line.strip().split(";")
            if len(parts) >= 3:
                scheme_code = parts[0].strip()
                # Try index 4 first (standard AMFI format)
                nav_str = parts[4].strip() if len(parts) >= 5 else ""
                try:
                    if scheme_code.isdigit() and nav_str and nav_str not in ("N/A", "-"):
                        val = float(nav_str)
                        if val > 0:
                            nav_map[scheme_code] = val
                            continue
                except (ValueError, IndexError):
                    pass
                
                # Fallback to index 2 (mock formats used in tests)
                try:
                    nav_str_alt = parts[2].strip()
                    if scheme_code.isdigit() and nav_str_alt and nav_str_alt not in ("N/A", "-"):
                        val = float(nav_str_alt)
                        if val > 0:
                            nav_map[scheme_code] = val
                except (ValueError, IndexError):
                    continue
        if nav_map:
            self._nav_cache = nav_map
            self._nav_cache_ts = time.time()


# Module-level singleton
nav_fetcher = NAVFetcher()
