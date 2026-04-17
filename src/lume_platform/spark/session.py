"""Spark session factory tuned for Colab / local / Kaggle."""

from __future__ import annotations

from pyspark.sql import SparkSession

from lume_platform.config import SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS


def build_spark(app_name: str = "lume-bigdata") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE_PARTITIONS))
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
