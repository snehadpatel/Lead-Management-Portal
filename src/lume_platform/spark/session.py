"""Spark session factory tuned for Colab / local / Kaggle."""

from __future__ import annotations

from pyspark.sql import SparkSession

from lume_platform.config import SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS


def build_spark(app_name: str = "lume-bigdata") -> SparkSession:
    import os
    
    fs = os.environ.get("LUME_SPARK_FS", "file:///")
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE_PARTITIONS))
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.hadoop.fs.defaultFS", fs)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )

    # Optional credentials configuration for cloud storages
    if fs.startswith("gs://"):
        builder = builder.config(
            "spark.hadoop.google.cloud.auth.service.account.enable",
            os.environ.get("GCS_SA_ENABLE", "false")
        ).config(
            "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
            os.environ.get("GCS_SA_KEYFILE", "")
        )
    elif fs.startswith("s3a://") or fs.startswith("s3n://") or fs.startswith("s3://"):
        builder = builder.config(
            "spark.hadoop.fs.s3a.access.key",
            os.environ.get("AWS_ACCESS_KEY_ID", "")
        ).config(
            "spark.hadoop.fs.s3a.secret.key",
            os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        ).config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

    return builder.getOrCreate()
