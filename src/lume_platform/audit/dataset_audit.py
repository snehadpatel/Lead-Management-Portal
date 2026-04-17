"""Pandas-based dataset audit: missing %, duplicates, stats, plots (Step 3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        for c in out.columns
    )
    return out


def missing_pct(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).sort_values(ascending=False)


def duplicate_stats(df: pd.DataFrame) -> dict[str, Any]:
    dup_mask = df.duplicated()
    return {
        "duplicate_rows": int(dup_mask.sum()),
        "unique_rows": int((~dup_mask).sum()),
        "pct_duplicate": float(dup_mask.mean() * 100),
    }


@dataclass
class AuditPlots:
    histogram_path: Path | None = None
    boxplot_path: Path | None = None
    heatmap_path: Path | None = None


def run_audit_plots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    out_dir: Path,
    sample_max: int = 5000,
) -> AuditPlots:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = AuditPlots()
    work = df[numeric_cols].dropna(axis=1, how="all")
    if work.empty:
        return plots
    if len(work) > sample_max:
        work = work.sample(sample_max, random_state=42)

    # Histogram (first numeric column)
    col0 = work.select_dtypes(include=[np.number]).columns[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(work[col0].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram: {col0}")
    hp = out_dir / "audit_histogram.png"
    fig.tight_layout()
    fig.savefig(hp, dpi=150)
    plt.close(fig)
    plots.histogram_path = hp

    # Box plots (up to 6 numeric cols)
    num_cols = list(work.select_dtypes(include=[np.number]).columns[:6])
    if num_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=work[num_cols], orient="h", ax=ax)
        ax.set_title("Box plots (sample)")
        bp = out_dir / "audit_boxplots.png"
        fig.tight_layout()
        fig.savefig(bp, dpi=150)
        plt.close(fig)
        plots.boxplot_path = bp

    # Correlation heatmap
    corr = work.select_dtypes(include=[np.number]).corr(numeric_only=True)
    if corr.shape[0] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0)
        ax.set_title("Correlation heatmap (numeric)")
        hm = out_dir / "audit_heatmap.png"
        fig.tight_layout()
        fig.savefig(hm, dpi=150)
        plt.close(fig)
        plots.heatmap_path = hm

    return plots


def quick_audit_report(
    csv_path: Path,
    sample_rows: int = 20_000,
    plot_out: Path | None = None,
) -> dict[str, Any]:
    """Load a bounded sample for interactive audit (full data handled in Spark)."""
    df = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
    df_norm = normalize_column_names(df)
    miss = missing_pct(df_norm)
    dups = duplicate_stats(df_norm)
    desc = df_norm.describe(include="all").transpose()
    plots: dict[str, str | None] = {}
    if plot_out:
        num_cols = list(df_norm.select_dtypes(include=[np.number]).columns)
        ap = run_audit_plots(df_norm, num_cols, plot_out)
        plots = {
            "histogram": str(ap.histogram_path) if ap.histogram_path else None,
            "boxplot": str(ap.boxplot_path) if ap.boxplot_path else None,
            "heatmap": str(ap.heatmap_path) if ap.heatmap_path else None,
        }
    return {
        "path": str(csv_path),
        "rows_sampled": len(df_norm),
        "columns": list(df_norm.columns),
        "missing_pct_top": miss.head(15).to_dict(),
        "duplicates": dups,
        "summary": desc.head(30).to_dict(),
        "plots": plots,
    }
