"""
Spark Structured Streaming Pipeline for Lume AI 2.0.
Monitors datasets/velocity/inbox for new leads and market data.
Includes Data Quality Watchdog and real-time aggregations.
"""

import sys
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Setup paths for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

from lume_platform.config import DATA_ROOT
from lume_platform.spark.session import build_spark
from lume_platform.db.mongo_client import db_client

def start_streaming_pipeline():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1. Define Schema for Incoming Leads (Data Quality Watchdog)
    lead_schema = StructType([
        StructField("lead_id", StringType(), True),
        StructField("First Name", StringType(), True),
        StructField("Last Name", StringType(), True),
        StructField("City", StringType(), True),
        StructField("Occupation", StringType(), True),
        StructField("Lead Source", StringType(), True),
        StructField("total_visits", DoubleType(), True),
        StructField("total_time_spent_on_website", DoubleType(), True),
        StructField("page_views_per_visit", DoubleType(), True),
        StructField("Conversion_Probability", DoubleType(), True),
        StructField("timestamp", StringType(), True)
    ])

    # 2. Read Stream from Inbox
    local_inbox = str(DATA_ROOT / "velocity/inbox")
    inbox_path = f"file://{local_inbox}"
    os.makedirs(local_inbox, exist_ok=True)
    
    print(f"🛰️ Monitoring Stream at: {inbox_path}")
    
    raw_stream = (
        spark.readStream
        .schema(lead_schema)
        .json(inbox_path)
    )

    # 3. Data Quality Watchdog (Filter bad data)
    valid_stream = raw_stream.filter(
        F.col("lead_id").isNotNull() & 
        (F.col("total_visits") >= 0) &
        (F.col("Conversion_Probability") >= 0)
    )

    # 4. Real-time Aggregations (Sliding Window)
    # Convert string timestamp to proper timestamp type
    processed_stream = valid_stream.withColumn("ts", F.to_timestamp("timestamp"))
    
    city_metrics = (
        processed_stream
        .groupBy(
            F.window("ts", "1 minute", "30 seconds"),
            "City"
        )
        .agg(
            F.avg("Conversion_Probability").alias("avg_lead_score"),
            F.count("lead_id").alias("lead_count")
        )
    )

    # 5. Sinks
    
    # A. Write Aggregations to Console for Debug
    console_query = (
        city_metrics.writeStream
        .outputMode("complete")
        .format("console")
        .start()
    )

    # B. Write Raw Leads to MongoDB (using foreachBatch)
    def write_to_mongo(df, epoch_id):
        batch_data = df.collect()
        for row in batch_data:
            data_dict = row.asDict()
            db_client.upsert_lead(data_dict["lead_id"], data_dict)
        print(f"✅ Batch {epoch_id} processed: {len(batch_data)} leads synced to MongoDB.")

    mongo_query = (
        valid_stream.writeStream
        .foreachBatch(write_to_mongo)
        .start()
    )

    # C. Write Metrics to JSON (for Dashboard polling)
    metrics_path = f"file://{DATA_ROOT / 'velocity/outputs/city_metrics'}"
    checkpoint_path = f"file://{DATA_ROOT / 'velocity/checkpoints/city_metrics'}"
    
    json_query = (
        city_metrics.writeStream
        .outputMode("complete")
        .format("json")
        .option("path", metrics_path)
        .option("checkpointLocation", checkpoint_path)
        .start()
    )

    print("🚀 Streaming Pipeline ACTIVE. Waiting for data...")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    start_streaming_pipeline()
