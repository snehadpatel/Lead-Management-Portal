"""Chronological time-series dataset builder and scaler without data leakage.

Guarantees:
1. Strictly chronological split: Train (oldest) -> Validation -> Test (newest).
2. Scaler fit on TRAIN split only; Val and Test transformed using train parameters.
3. Target sequence y strictly follows input sequence X without temporal overlap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lume_platform.config import PROJECT_ROOT


def load_nifty50_benchmark(project_root: Optional[Path] = None) -> pd.DataFrame:
    """Load and merge the 3 NIFTY 50 CSVs into a single chronological DataFrame."""
    root = project_root or PROJECT_ROOT
    nifty_patterns = [
        "NIFTY 50-21-02-2023-to-21-02-2024.csv",
        "NIFTY 50-21-02-2024-to-20-02-2025.csv",
        "NIFTY 50-21-02-2025-to-21-02-2026.csv",
    ]
    dfs = []
    for fname in nifty_patterns:
        fpath = root / fname
        if fpath.is_file():
            df = pd.read_csv(fpath)
            # Strip column names
            df.columns = [c.strip() for c in df.columns]
            # Standardize date to YYYY-MM-DD
            df["date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
            # Select and rename columns
            cols = {
                "Open": "nifty_open",
                "High": "nifty_high",
                "Low": "nifty_low",
                "Close": "nifty_close",
                "Shares Traded": "nifty_volume",
                "Turnover (₹ Cr)": "nifty_turnover_cr",
            }
            renamed = df[["date"] + [c for c in cols.keys() if c in df.columns]].rename(columns=cols)
            dfs.append(renamed)

    if not dfs:
        raise FileNotFoundError(f"No NIFTY 50 benchmark CSVs found under {root}")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return merged


def load_nav_history(scheme_code: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load historical NAV JSON for a given scheme from nav_cache."""
    root_cache = cache_dir or (PROJECT_ROOT / "output_production_final" / "nav_cache")
    nav_file = root_cache / f"hist_{scheme_code}.json"
    if not nav_file.is_file():
        raise FileNotFoundError(f"NAV cache file not found: {nav_file}")

    with open(nav_file, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    if "date" not in df.columns or "nav" not in df.columns:
        raise ValueError(f"Malformed NAV JSON in {nav_file}")

    # Standardize date to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def compute_technical_features(
    df: pd.DataFrame,
    nav_col: str = "nav",
    nifty_close_col: str = "nifty_close",
    include_sentiment: bool = False,
) -> pd.DataFrame:
    """
    Computes zero-leakage technical indicators and returns strictly using past data.

    Features generated:
    1. nav_return: (NAV_t - NAV_{t-1}) / NAV_{t-1}
    2. nifty_return: (NIFTY_t - NIFTY_{t-1}) / NIFTY_{t-1}
    3. nifty_volatility: 10-day rolling standard deviation of NIFTY daily returns
    4. rsi: 14-day Relative Strength Index on NAV normalized to [0, 1]
    5. sma20_dev: (NAV_t - SMA20) / SMA20
    6. sentiment: Optional financial sentiment score (default 0.0)
    """
    out = df.sort_values("date").reset_index(drop=True).copy()

    # 1. NAV Daily Return
    out["nav_return"] = out[nav_col].pct_change().fillna(0.0)

    # 2. NIFTY50 Return & Volatility
    if nifty_close_col in out.columns:
        out["nifty_return"] = out[nifty_close_col].pct_change().fillna(0.0)
        # 10-day rolling volatility (backward-looking only)
        out["nifty_volatility"] = (
            out["nifty_return"].rolling(window=10, min_periods=1).std().fillna(0.0)
        )
    else:
        out["nifty_return"] = 0.0
        out["nifty_volatility"] = 0.0

    # 3. RSI (14-day Relative Strength Index on NAV)
    delta = out[nav_col].diff().fillna(0.0)
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi_raw = 100.0 - (100.0 / (1.0 + rs))
    out["rsi"] = (rsi_raw / 100.0).clip(0.0, 1.0).fillna(0.5)

    # 4. SMA-20 Deviation on NAV
    sma20 = out[nav_col].rolling(window=20, min_periods=1).mean()
    out["sma20_dev"] = ((out[nav_col] - sma20) / (sma20 + 1e-8)).fillna(0.0)

    # 5. Sentiment feature flag (wired for Phase 4)
    if include_sentiment and "sentiment" in out.columns:
        out["sentiment"] = out["sentiment"].astype(float).fillna(0.0)
    else:
        out["sentiment"] = 0.0

    return out


def load_merged_market_series(
    scheme_code: str = "100119",
    cache_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    include_sentiment: bool = False,
) -> pd.DataFrame:
    """Merge fund NAV with NIFTY 50 benchmark and compute technical indicators."""
    nav_df = load_nav_history(scheme_code, cache_dir=cache_dir)
    nifty_df = load_nifty50_benchmark(project_root=project_root)

    merged = pd.merge(nav_df, nifty_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = compute_technical_features(merged, include_sentiment=include_sentiment)
    return merged


def create_sliding_windows(
    data: np.ndarray,
    lookback: int = 30,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sliding windows.

    X shape: (N - lookback - horizon + 1, lookback, features)
    y shape: (N - lookback - horizon + 1, horizon)
    Target y contains strictly the unscaled or scaled target values immediately following the window.
    """
    if len(data) < lookback + horizon:
        return np.empty((0, lookback, data.shape[1] if data.ndim > 1 else 1)), np.empty((0, horizon))

    xs, ys = [], []
    for i in range(len(data) - lookback - horizon + 1):
        x = data[i : (i + lookback)]
        # Target is the 0-th column (primary target, usually NAV) over the horizon
        if data.ndim == 1:
            y = data[(i + lookback) : (i + lookback + horizon)]
        else:
            y = data[(i + lookback) : (i + lookback + horizon), 0]
        xs.append(x)
        ys.append(y)

    X_arr = np.array(xs)
    if X_arr.ndim == 2:
        X_arr = np.expand_dims(X_arr, -1)
    y_arr = np.array(ys)
    return X_arr, y_arr


class ChronologicalTimeSeriesDataset:
    """Splits time-series chronologically into Train / Val / Test and scales without leakage."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "nav",
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        lookback: int = 30,
        horizon: int = 5,
    ):
        self.df = df.sort_values("date").reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = feature_cols or [target_col]
        self.lookback = lookback
        self.horizon = horizon

        n = len(self.df)
        if n < (lookback + horizon + 20):
            raise ValueError(f"Insufficient series length ({n}) for lookback={lookback}, horizon={horizon}")

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        self.train_df = self.df.iloc[:n_train].copy().reset_index(drop=True)
        self.val_df = self.df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[n_train + n_val :].copy().reset_index(drop=True)

        # Fit Scaler STRICTLY on Train split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_raw = self.train_df[self.feature_cols].values
        self.scaler.fit(train_raw)

        # Target scaler for inverting predictions (uses target_col index in feature_cols)
        self.target_idx = self.feature_cols.index(self.target_col)
        self.target_min_ = float(self.scaler.data_min_[self.target_idx])
        self.target_max_ = float(self.scaler.data_max_[self.target_idx])

        # Transform each split using TRAIN scaler
        self.train_scaled = self.scaler.transform(train_raw)
        self.val_scaled = self.scaler.transform(self.val_df[self.feature_cols].values)
        self.test_scaled = self.scaler.transform(self.test_df[self.feature_cols].values)

        # Generate windows within each split
        self.X_train, self.y_train = create_sliding_windows(self.train_scaled, lookback, horizon)
        self.X_val, self.y_val = create_sliding_windows(self.val_scaled, lookback, horizon)
        self.X_test, self.y_test = create_sliding_windows(self.test_scaled, lookback, horizon)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert scaled target back to original real-world units using train parameters."""
        scale = self.target_max_ - self.target_min_
        if scale == 0:
            scale = 1.0
        return y_scaled * scale + self.target_min_

    def get_summary(self) -> Dict[str, Any]:
        """Metadata for audit, logging, and data leakage verification."""
        return {
            "total_rows": len(self.df),
            "train_rows": len(self.train_df),
            "val_rows": len(self.val_df),
            "test_rows": len(self.test_df),
            "train_date_range": [str(self.train_df["date"].min()), str(self.train_df["date"].max())],
            "val_date_range": [str(self.val_df["date"].min()), str(self.val_df["date"].max())],
            "test_date_range": [str(self.test_df["date"].min()), str(self.test_df["date"].max())],
            "target_train_min": self.target_min_,
            "target_train_max": self.target_max_,
            "X_train_shape": list(self.X_train.shape),
            "X_val_shape": list(self.X_val.shape),
            "X_test_shape": list(self.X_test.shape),
            "features": self.feature_cols,
            "lookback": self.lookback,
            "horizon": self.horizon,
        }
