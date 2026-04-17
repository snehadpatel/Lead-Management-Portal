"""Distributed ingestion: CSV / JSON / Parquet with recursive paths for bulk folders."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def read_csv_dir(spark: SparkSession, path: Path, **options) -> DataFrame:
    p = str(path)
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("recursiveFileLookup", True)
        .option("mode", "PERMISSIVE")
        .csv(p, **options)
    )
    return df


def read_json_paths(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.option("multiLine", True).option("mode", "PERMISSIVE").json(str(path))


def read_parquet_dir(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.parquet(str(path))


def add_source_path(df: DataFrame) -> DataFrame:
    """Keep file path for lineage (useful when merging many intraday files)."""
    meta = "input_file_name()"
    return df.withColumn("_source_file", F.expr(meta))


def write_parquet_partitioned(df: DataFrame, out: Path, partition_cols: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    writer = df.write.mode("overwrite").option("compression", "snappy")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(str(out))
