"""
Candlestick Dataset Generation, Ground-Truth Labeling & PyTorch Dataset.

Strict Leakage Controls:
- Label is derived exclusively from FUTURE price behavior (t+1 ... t+H) over a 10-day horizon.
- The visual window (t-W+1 ... t) contains ZERO forward-looking information.
- Train / Val / Test partitions are split strictly BY MUTUAL FUND (group split),
  guaranteeing that fund-specific baselines or recurring chart geometries never leak.
"""

from __future__ import annotations

import glob
import io
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from lume_platform.config import PROJECT_ROOT

VISION_DATA_DIR = PROJECT_ROOT / "artifacts" / "vision_candlesticks"
IMAGES_DIR = VISION_DATA_DIR / "images"
MANIFEST_PATH = VISION_DATA_DIR / "dataset_manifest.csv"

# Label mapping
LABEL_NAMES = {
    0: "Bullish Breakout",
    1: "Bearish Breakdown",
    2: "Consolidation",
}


def render_candlestick_image(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Renders high-contrast financial candlestick chart image without axes or margins.
    Bullish candles: green (#26a69a)
    Bearish candles: red (#ef5350)
    """
    n = len(opens)
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = "#26a69a" if c >= o else "#ef5350"
        # High/Low wick line
        ax.plot([i, i], [l, h], color=color, linewidth=1.2)
        # Body rectangle
        lower = min(o, c)
        height = max(abs(c - o), 0.05)
        rect = Rectangle((i - 0.35, lower), 0.7, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(-1, n)
    ax.set_ylim(lows.min() - 0.5, highs.max() + 0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(out_size)
    return img


def synthesize_ohlc_from_nav(
    nav_series: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts daily mutual fund NAV values into realistic OHLC wicks and bodies
    using local rolling volatility to reflect intraday price discovery.
    """
    rng = np.random.RandomState(seed)
    n = len(nav_series)
    closes = nav_series.astype(float)
    opens = np.empty(n, dtype=float)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]

    # Compute rolling 10-day percentage returns std deviation
    returns = np.zeros(n)
    returns[1:] = np.diff(closes) / closes[:-1]
    vol = pd.Series(returns).rolling(10, min_periods=3).std().fillna(0.005).values

    # Realistic wicks based on local volatility
    wick_scale = vol * closes * 0.8
    highs = np.maximum(opens, closes) + rng.exponential(wick_scale, size=n)
    lows = np.minimum(opens, closes) - rng.exponential(wick_scale, size=n)
    return opens, highs, lows, closes


def generate_vision_dataset(
    nav_cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    lookback_window: int = 40,
    horizon: int = 10,
    stride: int = 15,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Generates labeled candlestick chart dataset from historical NAV caches.
    Splits by fund:
      - Train: 13 funds
      - Val: 3 funds
      - Test: 3 funds
    """
    cache_dir = nav_cache_dir or (PROJECT_ROOT / "output_production_final" / "nav_cache")
    out_dir = output_dir or IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (out_dir.parent / "dataset_manifest.csv")

    if manifest_path.exists() and not force_rebuild:
        print(f"Loading existing dataset manifest from {manifest_path}")
        return pd.read_csv(manifest_path)

    files = sorted(glob.glob(str(cache_dir / "hist_*.json")))
    if not files:
        raise FileNotFoundError(f"No fund history JSON files found in {cache_dir}")

    all_funds = [Path(f).stem.replace("hist_", "") for f in files]
    # Deterministic split by fund
    train_funds = set(all_funds[:13])
    val_funds = set(all_funds[13:16])
    test_funds = set(all_funds[16:])

    records: List[Dict] = []
    print(f"Generating candlestick images for {len(files)} mutual funds (stride={stride})...")

    img_count = 0
    for f in files:
        fund_code = Path(f).stem.replace("hist_", "")
        split_name = "train" if fund_code in train_funds else ("val" if fund_code in val_funds else "test")
        
        with open(f, "r") as fp:
            data = json.load(fp)
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)

        n = len(df)
        nav_vals = df["nav"].values
        opens, highs, lows, closes = synthesize_ohlc_from_nav(nav_vals, seed=int(fund_code) % 10000)

        for i in range(0, n - lookback_window - horizon + 1, stride):
            # Window data
            w_opens = opens[i : i + lookback_window]
            w_highs = highs[i : i + lookback_window]
            w_lows = lows[i : i + lookback_window]
            w_closes = closes[i : i + lookback_window]

            # Future forward trajectory
            p0 = w_closes[-1]
            p_end = nav_vals[i + lookback_window + horizon - 1]
            future_navs = nav_vals[i + lookback_window : i + lookback_window + horizon]

            r_forward = (p_end - p0) / p0
            max_r = (np.max(future_navs) - p0) / p0
            min_r = (np.min(future_navs) - p0) / p0

            # Technical pattern label from forward price trajectory
            # Bullish Breakout: forward return >= +0.6% and peak >= +0.8%
            # Bearish Breakdown: forward return <= -0.2% and trough <= -0.4%
            # Consolidation: range-bound between -0.2% and +0.6%
            if r_forward >= 0.006 and max_r >= 0.008:
                label = 0
            elif r_forward <= -0.002 and min_r <= -0.004:
                label = 1
            else:
                label = 2

            start_dt = str(df["date"].iloc[i].date())
            end_dt = str(df["date"].iloc[i + lookback_window - 1].date())
            img_id = f"mf_{fund_code}_{i}_{end_dt}.png"
            img_path = out_dir / img_id

            # Render and save chart image
            img = render_candlestick_image(w_opens, w_highs, w_lows, w_closes)
            img.save(img_path)

            records.append({
                "image_id": img_id,
                "fund": fund_code,
                "split": split_name,
                "start_date": start_dt,
                "end_date": end_dt,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "r_forward": float(r_forward),
                "max_forward_r": float(max_r),
                "min_forward_r": float(min_r),
                "image_path": str(img_path),
            })
            img_count += 1

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated {img_count} candlestick charts saved in {out_dir}")
    print(f"Manifest saved to {manifest_path}")
    return manifest_df


class CandlestickDataset(Dataset):
    """
    PyTorch Dataset for 2D Candlestick Chart Images.
    Supports ImageNet normalization, training augmentations, and memory preloading.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str = "train",
        transform: Optional[Callable] = None,
        preload: bool = True,
    ):
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.split = split
        self.labels = self.df["label"].values.astype(int)
        self.paths = self.df["image_path"].values.tolist()

        if transform is not None:
            self.transform = transform
        elif split == "train":
            # Augmentations that preserve vertical price directionality
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images: List[Image.Image] = []
        if preload:
            print(f"Preloading {len(self.paths)} images into memory for split '{split}'...")
            for p in self.paths:
                self.images.append(Image.open(p).convert("RGB"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img = self.images[idx] if self.images else Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform(img)
        label = int(self.labels[idx])
        return tensor, label, self.paths[idx]
