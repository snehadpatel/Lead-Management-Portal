"""
End-to-end Spark ETL sketch for the full tree under DATA_ROOT.

Designed for Google Colab: set LUME_DATA_ROOT to your Drive-mounted datasets path.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import functions as F

from lume_platform.config import CLEANED_DIR, DATA_ROOT, ensure_dirs
from lume_platform.spark.cleaning import drop_duplicates, impute_numeric_mean, normalize_columns
from lume_platform.spark.ingestion import read_csv_dir, read_json_paths, write_parquet_partitioned
from lume_platform.spark.session import build_spark
from lume_platform.spark.validation import null_percentage


def process_lead_scoring_full_sample(spark, data_root: Path | None = None, sample: float | None = None) -> None:
    """Example full pass on lead scoring CSV (fits in driver-friendly parquet)."""
    ensure_dirs()
    root = data_root or DATA_ROOT
    path = root / "structured/leads/lead_scoring/Lead Scoring.csv"
    if path.is_file():
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("mode", "PERMISSIVE")
            .csv(str(path))
        )
    else:
        df = read_csv_dir(spark, root / "structured/leads/lead_scoring")

    if sample:
        df = df.sample(False, sample, seed=42)

    df = normalize_columns(df)
    df = drop_duplicates(df)
    # Impute known numeric columns after rename
    for c in ["totalvisits", "total_time_spent_on_website", "page_views_per_visit"]:
        if c in df.columns:
            df = df.fillna(0, subset=[c])
    df = df.persist()
    _ = null_percentage(df)
    write_parquet_partitioned(df, CLEANED_DIR / "lead_scoring_clean", partition_cols=["converted"] if "converted" in df.columns else None)
    df.unpersist()


def process_nifty_intraday_tree(spark, data_root: Path | None = None, sample: float | None = None) -> None:
    """Heavy path: recursive CSV read of NIFTY 500 intraday folder (~19GB)."""
    root = data_root or DATA_ROOT
    intra = root / "structured/stock_prices/nifty500_intraday"
    if not intra.is_dir():
        return
    df = read_csv_dir(spark, intra)
    if sample:
        df = df.sample(False, sample, seed=42)
    df = normalize_columns(df)
    df = df.filter(F.col("close").isNotNull() & F.col("open").isNotNull())
    df = impute_numeric_mean(df, [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns])
    df = df.persist()
    write_parquet_partitioned(df, CLEANED_DIR / "nifty500_intraday_clean", partition_cols=None)
    df.unpersist()


def process_social_sentiment(spark, data_root: Path | None = None) -> None:
    """Process semi-structured sentiment data."""
    root = data_root or DATA_ROOT
    path = root / "semi_structured/social_sentiment/data.csv"
    if not path.is_file():
        return
    df = spark.read.option("header", True).option("inferSchema", True).csv(str(path))
    df = normalize_columns(df)
    df = df.dropDuplicates()
    write_parquet_partitioned(df, CLEANED_DIR / "social_sentiment_clean")


def run_default_etl(data_root: Path | None = None, sample: float | None = None) -> None:
    spark = build_spark()
    try:
        process_lead_scoring_full_sample(spark, data_root, sample)
        process_nifty_intraday_tree(spark, data_root, sample)
        process_social_sentiment(spark, data_root)
    finally:
        spark.stop()
