"""Spark-side cleaning: normalize names, dedupe, numeric imputation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def normalize_columns(df: DataFrame) -> DataFrame:
    for c in df.columns:
        if c.startswith("_"):
            continue
        nc = (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )
        if nc != c:
            df = df.withColumnRenamed(c, nc)
    return df


def drop_duplicates(df: DataFrame, subset: list[str] | None = None) -> DataFrame:
    if subset:
        return df.dropDuplicates(subset)
    return df.dropDuplicates()


def impute_numeric_mean(df: DataFrame, numeric_cols: list[str]) -> DataFrame:
    """Impute numeric columns with column means (single aggregate pass, scalable)."""
    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        return df
    row = df.select([F.mean(F.col(c)).alias(c) for c in present]).collect()[0]
    fill = {c: row[c] for c in present if row[c] is not None}
    return df.fillna(fill)


def fill_na_median_approx(df: DataFrame, col: str) -> DataFrame:
    """Exact median via approx quantile (cheap on large data)."""
    if col not in df.columns:
        return df
    med = df.approxQuantile(col, [0.5], 0.01)
    if not med:
        return df
    return df.fillna({col: med[0]})
