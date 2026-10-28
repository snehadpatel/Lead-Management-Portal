"""
Unit tests asserting ZERO data leakage in Lume AI time-series data pipelines.

Verifies:
1. Chronological order: Train timestamps < Val timestamps < Test timestamps.
2. Train-only scaling: Scaler parameters fit strictly on train split, not global dataset.
3. Window temporal separation: Target sequences strictly follow input windows with no overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from lume_platform.config import PROJECT_ROOT
from lume_platform.data.timeseries_dataset import (
    ChronologicalTimeSeriesDataset,
    create_sliding_windows,
    load_nav_history,
    load_merged_market_series,
)


def test_chronological_split_strict_ordering():
    """Verify that split boundary timestamps do not overlap and are strictly monotonic."""
    df = load_nav_history("100119")
    dataset = ChronologicalTimeSeriesDataset(df, target_col="nav", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_dates = pd.to_datetime(dataset.train_df["date"])
    val_dates = pd.to_datetime(dataset.val_df["date"])
    test_dates = pd.to_datetime(dataset.test_df["date"])

    # 1. Monotonic ascending within splits
    assert train_dates.is_monotonic_increasing, "Train split is not sorted chronologically"
    assert val_dates.is_monotonic_increasing, "Validation split is not sorted chronologically"
    assert test_dates.is_monotonic_increasing, "Test split is not sorted chronologically"

    # 2. Strict chronological boundaries
    assert train_dates.max() < val_dates.min(), (
        f"Temporal leakage detected: Train max date ({train_dates.max()}) >= Val min date ({val_dates.min()})"
    )
    assert val_dates.max() < test_dates.min(), (
        f"Temporal leakage detected: Val max date ({val_dates.max()}) >= Test min date ({test_dates.min()})"
    )


def test_scaler_fit_on_train_only():
    """Verify that MinMaxScaler parameters are derived strictly from train set, not test or global set."""
    df = load_nav_history("100119")
    dataset = ChronologicalTimeSeriesDataset(df, target_col="nav", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_nav_min = dataset.train_df["nav"].min()
    train_nav_max = dataset.train_df["nav"].max()
    global_nav_min = df["nav"].min()
    global_nav_max = df["nav"].max()

    # The dataset scaler parameters MUST match the train set bounds
    assert np.isclose(dataset.target_min_, train_nav_min, atol=1e-5), (
        f"Scaler min {dataset.target_min_} does not match train min {train_nav_min}"
    )
    assert np.isclose(dataset.target_max_, train_nav_max, atol=1e-5), (
        f"Scaler max {dataset.target_max_} does not match train max {train_nav_max}"
    )

    # In a typical multi-year market with growth/inflation, global max exceeds train max
    # If the scaler was mistakenly fit globally, this assertion would catch it
    scaled_test = dataset.test_scaled
    # Scaled test values are permitted to exceed 1.0 or fall below 0.0 because they reflect real out-of-sample shifts
    assert dataset.scaler.data_min_[dataset.target_idx] == train_nav_min
    assert dataset.scaler.data_max_[dataset.target_idx] == train_nav_max


def test_window_target_no_overlap():
    """Verify that target horizon y(t) starts strictly after lookback window X(t)."""
    # Create synthetic linear sequence: 0, 1, 2, ..., 99
    data = np.arange(100, dtype=np.float32).reshape(-1, 1)
    lookback = 30
    horizon = 5

    X, y = create_sliding_windows(data, lookback=lookback, horizon=horizon)

    assert len(X) == (100 - lookback - horizon + 1)
    for i in range(len(X)):
        # Last element in window X[i]
        last_x = X[i, -1, 0]
        # First element in target y[i]
        first_y = y[i, 0]
        # Target must immediately succeed lookback without overlap or gap
        assert np.isclose(first_y, last_x + 1), (
            f"Target sequence does not cleanly succeed input window at index {i}: last_x={last_x}, first_y={first_y}"
        )
        assert len(y[i]) == horizon


def test_multivariate_merged_series_leakage_free():
    """Verify that multi-asset merged dataset (NAV + NIFTY50 + Volume) obeys chronological constraints."""
    merged = load_merged_market_series("100119")
    assert len(merged) > 100, "Merged series too short"
    assert "nav" in merged.columns
    assert "nifty_close" in merged.columns
    assert "nifty_volume" in merged.columns

    dates = pd.to_datetime(merged["date"])
    assert dates.is_monotonic_increasing, "Merged series is not chronological"

    dataset = ChronologicalTimeSeriesDataset(
        merged,
        target_col="nav",
        feature_cols=["nav", "nifty_close", "nifty_volume"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    assert dataset.X_train.shape[2] == 3, f"Expected 3 features, got {dataset.X_train.shape[2]}"
    assert dataset.X_val.shape[2] == 3
    assert dataset.X_test.shape[2] == 3
