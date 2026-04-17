"""
Spark Structured Streaming: Real-Time Ingest & AI Scoring.
Ingests from streaming/inbox and applies Lead/Sentiment models.
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Project path discovery
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Set correct JAVA_HOME for Local Spark compatibility
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"

# Ensure Spark workers use the same Python version as the driver (the venv)
VENV_PYTHON = str(ROOT / "venv/bin/python")
os.environ["PYSPARK_PYTHON"] = VENV_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = VENV_PYTHON

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from lume_platform.spark.session import build_spark
from lume_platform.spark.cleaning import normalize_columns
from lume_platform.inference.spark_sklearn_udf import score_sentiment_udf, score_lead_probability_udf

# Define Schemas
MARKET_SCHEMA = StructType([
    StructField("Timestamp", StringType(), True),
    StructField("Market", StringType(), True),
    StructField("Status", StringType(), True),
    StructField("Index", StringType(), True),
    StructField("Last_Price", DoubleType(), True),
    StructField("Variation", DoubleType(), True),
    StructField("Pct_Change", DoubleType(), True),
    StructField("Trade_Date", StringType(), True),
])

NEWS_SCHEMA = StructType([
    StructField("Ingestion_Timestamp", StringType(), True),
    StructField("Source", StringType(), True),
    StructField("Title", StringType(), True),
    StructField("Link", StringType(), True),
    StructField("Published_Date", StringType(), True),
    StructField("Summary", StringType(), True),
])

def run_streaming_engine():
    spark = build_spark()
    # Set logging to WARN to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    inbox_market = str(ROOT / "streaming/inbox/market")
    inbox_news = str(ROOT / "streaming/inbox/news")
    checkpoint_dir = str(ROOT / "streaming/checkpoints")
    output_dir = str(ROOT / "streaming/outputs")
    
    print("="*60)
    print("🚀 LUME AI SPARK STREAMING ENGINE INITIALIZED")
    print(f"Watching MARKET: {inbox_market}")
    print(f"Watching NEWS:   {inbox_news}")
    print("="*60)

    # 1. Market Pulse Stream
    market_stream = spark.readStream \
        .schema(MARKET_SCHEMA) \
        .option("maxFilesPerTrigger", 1) \
        .csv(inbox_market)
    
    # Process market data (Normalize)
    market_clean = normalize_columns(market_stream)
    
    # 2. News/Sentiment Stream (Intelligence)
    news_stream = spark.readStream \
        .schema(NEWS_SCHEMA) \
        .option("maxFilesPerTrigger", 1) \
        .csv(inbox_news)
    
    # Apply Real-Time Sentiment AI (Senior Feature)
    scored_news = news_stream.withColumn("sentiment_score", score_sentiment_udf(F.col("Title")))
    scored_news = scored_news.withColumn("market_impact", 
                                       F.when(F.col("sentiment_score") > 0.6, "BULLISH")
                                        .when(F.col("sentiment_score") < 0.4, "BEARISH")
                                        .otherwise("NEUTRAL"))

    # 3. Output to Parquet (The "Live Lake")
    market_query = market_clean.writeStream \
        .format("parquet") \
        .option("path", f"{output_dir}/market_live.parquet") \
        .option("checkpointLocation", f"{checkpoint_dir}/market") \
        .outputMode("append") \
        .start()

    news_query = scored_news.writeStream \
        .format("parquet") \
        .option("path", f"{output_dir}/news_live.parquet") \
        .option("checkpointLocation", f"{checkpoint_dir}/news") \
        .outputMode("append") \
        .start()

    # 4. Console Sink for Demo Monitoring
    console_query = scored_news.select("Title", "market_impact") \
        .writeStream \
        .format("console") \
        .start()

    print("Queries started. Waiting for termination...")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run_streaming_engine()
