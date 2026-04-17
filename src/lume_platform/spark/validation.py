"""Data validation: null rates, uniques, IQR outliers."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def null_percentage(df: DataFrame) -> dict[str, float]:
    total = df.count()
    if total == 0:
        return {}
    out: dict[str, float] = {}
    for c in df.columns:
        nulls = df.filter(F.col(c).isNull()).count()
        out[c] = round(100.0 * nulls / total, 4)
    return out


def approx_unique_counts(df: DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: df.select(c).distinct().approx_countDistinct() for c in cols if c in df.columns}


def iqr_outlier_flags(df: DataFrame, col: str) -> DataFrame:
    """Add column {col}_outlier_iqr based on Tukey IQR on non-null values."""
    qs = df.approxQuantile(col, [0.25, 0.75], 0.01)
    if not qs or qs[0] is None or qs[1] is None:
        return df.withColumn(f"{col}_outlier_iqr", F.lit(False))
    q1, q3 = qs[0], qs[1]
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df.withColumn(
        f"{col}_outlier_iqr",
        (F.col(col) < low) | (F.col(col) > high),
    )
