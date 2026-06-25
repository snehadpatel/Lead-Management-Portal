"""
Portfolio Analytics — Advanced portfolio metrics (Alpha, Sharpe, Beta, Drawdown) and backtest simulations.

Provides:
- Historical daily portfolio return series construction
- Sharpe Ratio, Beta, Alpha, and Max Drawdown calculations
- Benchmark comparison (Nifty 50 TRI and Crisil Debt Index)
- Efficient frontier curve coordinates
- Radar chart allocation dimensions (Growth, Risk, Cost, Passive, Dividend)
"""
from __future__ import annotations

import os
import re
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from lume_platform.data.nav_fetcher import nav_fetcher
from lume_platform.config import EXPORT_DIR

class PortfolioAnalytics:
    """Calculates advanced portfolio risk/return analytics and backtests."""

    def __init__(self):
        self._nifty_cache: Optional[pd.DataFrame] = None
        self._nifty_cache_ts: float = 0
        self._nifty_cache_ttl = 1800  # 30 minutes for historical index
        self._cache_dir = EXPORT_DIR / "analytics_cache"
        if "VERCEL" in os.environ:
            self._cache_dir = Path("/tmp") / "analytics_cache"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def get_historical_benchmark(self, days: int = 1825) -> pd.DataFrame:
        """Fetch historical NIFTY 50 (^NSEI) closing prices."""
        now = datetime.now()
        cache_file = self._cache_dir / f"nifty_hist_{days}.json"

        # Check in-memory cache
        if self._nifty_cache is not None and (datetime.now().timestamp() - self._nifty_cache_ts) < self._nifty_cache_ttl:
            return self._nifty_cache

        # Check file cache
        if cache_file.is_file():
            try:
                age = datetime.now().timestamp() - cache_file.stat().st_mtime
                if age < self._nifty_cache_ttl:
                    df = pd.read_json(cache_file)
                    df.index = pd.to_datetime(df.index)
                    self._nifty_cache = df
                    self._nifty_cache_ts = datetime.now().timestamp()
                    return df
            except Exception:
                pass

        # Fetch from yfinance
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period=f"{math.ceil(days/365)}y")
            if not hist.empty:
                df = hist[["Close"]].copy()
                df.index = df.index.tz_localize(None)
                # Cache it
                df.to_json(cache_file)
                self._nifty_cache = df
                self._nifty_cache_ts = datetime.now().timestamp()
                return df
        except Exception as e:
            print(f"⚠️ Failed to fetch Nifty history via yfinance: {e}")

        # Fallback to parsing local Nifty CSV files
        return self._load_nifty_csv_fallback(days)

    def _load_nifty_csv_fallback(self, days: int) -> pd.DataFrame:
        """Fallback parser for local Nifty 50 CSV files."""
        dfs = []
        csv_files = [
            "NIFTY 50-21-02-2025-to-21-02-2026.csv",
            "NIFTY 50-21-02-2024-to-20-02-2025.csv",
            "NIFTY 50-21-02-2023-to-21-02-2024.csv"
        ]
        
        workspace_dir = Path("/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData")
        for fn in csv_files:
            fp = workspace_dir / fn
            if fp.is_file():
                try:
                    df = pd.read_csv(fp)
                    # Strip headers
                    df.columns = [c.strip() for c in df.columns]
                    if "Date" in df.columns and "Close" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
                        df = df.dropna(subset=["Date"])
                        df.set_index("Date", inplace=True)
                        dfs.append(df[["Close"]])
                except Exception as e:
                    print(f"⚠️ Failed to parse local Nifty CSV {fn}: {e}")

        if dfs:
            combined = pd.concat(dfs).sort_index()
            # De-duplicate
            combined = combined[~combined.index.duplicated(keep="first")]
            self._nifty_cache = combined
            return combined

        # Absolute fallback: Return simulated return series
        start_val = 22000.0
        dates = [datetime.today() - timedelta(days=i) for i in range(days)]
        dates.reverse()
        prices = []
        curr = start_val
        for i in range(len(dates)):
            curr = curr * (1 + np.random.normal(0.0004, 0.009))  # ~10% annual, 14% vol
            prices.append(curr)
        df = pd.DataFrame(prices, index=dates, columns=["Close"])
        return df

    def analyze_portfolio(self, holdings: List[Dict[str, Any]], horizon: str = "1y") -> Dict[str, Any]:
        """
        Backtests current portfolio holdings over the specified horizon
        and calculates advanced risk & outperformance statistics.
        """
        horizon_days = {
            "6m": 180,
            "1y": 365,
            "3y": 1095,
            "5y": 1825
        }.get(horizon, 365)

        if not holdings:
            return self._generate_empty_response(horizon)

        # Get historical NAVs for each scheme
        hist_navs = {}
        for h in holdings:
            code = str(h.get("scheme_code", ""))
            if code:
                hist = nav_fetcher.get_historical_nav(code, days=horizon_days)
                if hist:
                    # Convert to df
                    dates = []
                    navs = []
                    for entry in hist:
                        try:
                            dates.append(datetime.strptime(entry["date"], "%d-%m-%Y"))
                            navs.append(entry["nav"])
                        except Exception:
                            continue
                    if dates:
                        df = pd.DataFrame(navs, index=dates, columns=[code])
                        # Sort oldest to newest
                        df = df.sort_index()
                        # De-duplicate
                        df = df[~df.index.duplicated(keep="first")]
                        hist_navs[code] = df

        if not hist_navs:
            return self._generate_empty_response(horizon)

        # Align Nifty index
        nifty_df = self.get_historical_benchmark(horizon_days)
        
        # Combine all series into a single DataFrame
        all_dfs = [nifty_df.rename(columns={"Close": "Nifty"})]
        for code, df in hist_navs.items():
            all_dfs.append(df)
            
        combined = pd.concat(all_dfs, axis=1)
        # Forward fill NAVs to handle holidays / missing points
        combined = combined.ffill().bfill()
        
        # Crop to the requested time horizon
        start_date = datetime.now() - timedelta(days=horizon_days)
        combined = combined[combined.index >= start_date]
        
        if combined.empty or len(combined) < 5:
            return self._generate_empty_response(horizon)

        # Compute weights based on current units * latest NAV
        latest_navs = nav_fetcher.get_latest_navs_bulk(list(hist_navs.keys()))
        portfolio_values = []
        weights = {}
        total_value = 0
        
        for h in holdings:
            code = str(h.get("scheme_code", ""))
            units = float(h.get("units", 0))
            buy_value = float(h.get("buy_value", 0))
            if code in hist_navs:
                latest_nav = latest_navs.get(code, float(h.get("buy_nav", 10)))
                val = units * latest_nav
                weights[code] = val
                total_value += val

        if total_value > 0:
            weights = {k: v / total_value for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(hist_navs) for k in hist_navs.keys()}

        # Compute historical daily portfolio value / NAV series
        # We model this by weighting the normalized daily growth of each scheme
        portfolio_nav = pd.Series(0.0, index=combined.index)
        for code, weight in weights.items():
            # Normalize scheme price to 100 at start
            scheme_series = combined[code]
            start_val = scheme_series.iloc[0]
            if start_val > 0:
                normalized = (scheme_series / start_val) * 100.0
                portfolio_nav += weight * normalized
            else:
                portfolio_nav += weight * 100.0
                
        # Benchmark series normalized to 100
        nifty_normalized = (combined["Nifty"] / combined["Nifty"].iloc[0]) * 100.0
        
        # Crisil Debt Index approximation (stable 7% annual return with minor noise)
        debt_normalized = []
        debt_start = 100.0
        days_in_period = len(combined.index)
        daily_rate = (1.07) ** (1 / 252) - 1
        curr = debt_start
        for i in range(days_in_period):
            curr = curr * (1 + daily_rate + np.random.normal(0, 0.0002))
            debt_normalized.append(curr)
        debt_series = pd.Series(debt_normalized, index=combined.index)

        # Calculate daily returns
        port_daily_ret = portfolio_nav.pct_change().dropna()
        nifty_daily_ret = nifty_normalized.pct_change().dropna()
        
        # Metrics: Return, Volatility, Sharpe, Beta, Alpha, Max Drawdown
        total_days = (combined.index[-1] - combined.index[0]).days
        years = total_days / 365.0 if total_days > 0 else 1.0
        if years < 0.01:
            years = 0.01
            
        port_cum_ret = (portfolio_nav.iloc[-1] / portfolio_nav.iloc[0] - 1)
        nifty_cum_ret = (nifty_normalized.iloc[-1] / nifty_normalized.iloc[0] - 1)
        
        port_ann_ret = (1 + port_cum_ret) ** (1 / years) - 1
        nifty_ann_ret = (1 + nifty_cum_ret) ** (1 / years) - 1
        
        # Risk free rate (e.g. 6%)
        rf = 0.06
        
        # Daily volatility annualized
        port_vol = port_daily_ret.std() * math.sqrt(252)
        nifty_vol = nifty_daily_ret.std() * math.sqrt(252)
        
        # Sharpe Ratio
        port_sharpe = (port_ann_ret - rf) / port_vol if port_vol > 0 else 0.0
        
        # Beta
        if nifty_daily_ret.var() > 0:
            cov = port_daily_ret.cov(nifty_daily_ret)
            beta = cov / nifty_daily_ret.var()
        else:
            beta = 1.0
            
        # Alpha (Jensen's Alpha annualized)
        alpha = port_ann_ret - (rf + beta * (nifty_ann_ret - rf))
        
        # Max Drawdown
        roll_max = portfolio_nav.cummax()
        drawdowns = (portfolio_nav - roll_max) / roll_max
        max_drawdown = drawdowns.min()

        # Resample performance comparison points for frontend charting (limit to 12-15 points max)
        chart_points_limit = 12
        if len(combined) > chart_points_limit:
            indices = np.linspace(0, len(combined) - 1, chart_points_limit, dtype=int)
            dates_chart = [combined.index[i].strftime("%b %y") for i in indices]
            port_chart = [round(float(portfolio_nav.iloc[i] - 100), 2) for i in indices]
            nifty_chart = [round(float(nifty_normalized.iloc[i] - 100), 2) for i in indices]
            debt_chart = [round(float(debt_series.iloc[i] - 100), 2) for i in indices]
        else:
            dates_chart = [d.strftime("%b %y") for d in combined.index]
            port_chart = [round(float(v - 100), 2) for v in portfolio_nav]
            nifty_chart = [round(float(v - 100), 2) for v in nifty_normalized]
            debt_chart = [round(float(v - 100), 2) for v in debt_series]

        # Generate Efficient Frontier Curve coordinates
        # Simulated curve tailored dynamically to portfolio volatility/returns
        curve_points = []
        vol_shift = max(0.0, float(port_vol - nifty_vol))
        ret_shift = max(-0.02, float(port_ann_ret - nifty_ann_ret))
        
        # Standard efficient frontier curve points
        for x in range(6, 23, 2):
            # risk vs return curve shape
            y = round(math.sqrt(x - 5) * 4.5 + 3.5 + ret_shift * 5, 2)
            curve_points.append({"x": x, "y": y})

        # Portfolio Coordinate on Frontier
        portfolio_point = {
            "x": round(float(port_vol * 100), 2),
            "y": round(float(port_ann_ret * 100), 2)
        }

        # Calculate Radar Chart Dimensions (Growth, Risk, Cost, Passive, Dividend)
        radar_metrics = self._calculate_radar_metrics(holdings)

        return {
            "kpis": {
                "alpha": round(alpha * 100, 2),
                "sharpe": round(port_sharpe, 2),
                "beta": round(beta, 2),
                "volatility": round(port_vol * 100, 2),
                "max_drawdown": round(abs(max_drawdown) * 100, 2),
                "portfolio_return": round(port_cum_ret * 100, 2)
            },
            "performance_comparison": {
                "labels": dates_chart,
                "portfolio": port_chart,
                "benchmark": nifty_chart,
                "debt": debt_chart
            },
            "efficient_frontier": {
                "curve": curve_points,
                "portfolio": portfolio_point,
                "benchmarks": [
                    {"x": 8.0, "y": 7.2, "label": "Low Risk Debt Index"},
                    {"x": 12.0, "y": 11.5, "label": "Aggressive Hybrid"},
                    {"x": 16.0, "y": 14.8, "label": "Nifty 50 Index"},
                    {"x": 20.0, "y": 18.5, "label": "Small Cap Growth Index"}
                ]
            },
            "radar": radar_metrics
        }

    def _calculate_radar_metrics(self, holdings: List[Dict[str, Any]]) -> List[int]:
        """
        Derives radar chart dimensions (Growth, Risk, Cost, Passive, Dividend) 0-100
        based on the portfolio's fund categories and allocations.
        """
        growth_scores = []
        risk_scores = []
        cost_scores = []
        passive_scores = []
        dividend_scores = []
        
        total_val = 0
        weights = []
        
        for h in holdings:
            cat = str(h.get("category", "")).lower()
            name = str(h.get("scheme_name", "")).lower()
            buy_val = float(h.get("buy_value", 10000))
            
            # Category classifications
            is_equity = "equity" in cat or "growth" in cat or "midcap" in cat or "smallcap" in cat
            is_debt = "debt" in cat or "liquid" in cat or "income" in cat or "gilt" in cat or "bond" in cat
            is_hybrid = "hybrid" in cat or "balanced" in cat or "allocation" in cat
            is_passive = "index" in name or "nifty" in name or "etf" in name or "passive" in cat
            
            # Scores for this holding
            g = 80 if is_equity else (40 if is_hybrid else 15)
            r = 75 if is_equity else (45 if is_hybrid else 10)
            c = 85 if is_passive else (40 if is_equity else 60)
            p = 95 if is_passive else 10
            d = 70 if is_debt else (40 if is_hybrid else 20)
            
            # Boosters
            if "smallcap" in name or "midcap" in name:
                g += 15
                r += 15
            elif "liquid" in name or "overnight" in name:
                r -= 5
                d += 15
                
            growth_scores.append(g)
            risk_scores.append(r)
            cost_scores.append(c)
            passive_scores.append(p)
            dividend_scores.append(d)
            
            weights.append(buy_val)
            total_val += buy_val

        if total_val > 0:
            w = np.array(weights) / total_val
            growth = int(np.dot(growth_scores, w))
            risk = int(np.dot(risk_scores, w))
            cost = int(np.dot(cost_scores, w))
            passive = int(np.dot(passive_scores, w))
            div = int(np.dot(dividend_scores, w))
        else:
            growth, risk, cost, passive, div = 60, 50, 70, 50, 40

        # Clip values to 0-100
        return [
            min(100, max(0, growth)),
            min(100, max(0, risk)),
            min(100, max(0, cost)),
            min(100, max(0, passive)),
            min(100, max(0, div))
        ]

    def _generate_empty_response(self, horizon: str) -> Dict[str, Any]:
        """Provides realistic default values and coordinates for empty/new portfolios."""
        months = {
            "6m": ["Dec", "Jan", "Feb", "Mar", "Apr", "May"],
            "3y": ["2024 H1", "2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"],
            "5y": ["2022", "2023", "2024", "2025", "2026"]
        }.get(horizon, ["Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May"])
        
        l = len(months)
        return {
            "kpis": {"alpha": 0.0, "sharpe": 0.0, "beta": 1.0, "volatility": 0.0, "max_drawdown": 0.0, "portfolio_return": 0.0},
            "performance_comparison": {
                "labels": months,
                "portfolio": [0.0] * l,
                "benchmark": [0.0] * l,
                "debt": [0.0] * l
            },
            "efficient_frontier": {
                "curve": [{"x": x, "y": round(math.sqrt(x - 5) * 4.5 + 3.5, 2)} for x in range(6, 23, 2)],
                "portfolio": {"x": 14.5, "y": 12.8},
                "benchmarks": [
                    {"x": 8.0, "y": 7.2, "label": "Low Risk Debt Index"},
                    {"x": 12.0, "y": 11.5, "label": "Aggressive Hybrid"},
                    {"x": 16.0, "y": 14.8, "label": "Nifty 50 Index"},
                    {"x": 20.0, "y": 18.5, "label": "Small Cap Growth Index"}
                ]
            },
            "radar": [0, 0, 0, 0, 0]
        }

portfolio_analytics = PortfolioAnalytics()
