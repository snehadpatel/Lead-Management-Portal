"""
Distributed batch scoring: apply ``LeadScoringPipelineBundle`` inside Spark ``mapInPandas``.

The bundle declares ``feature_columns``; each input partition must contain those columns.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType, StringType

from lume_platform.ml.bundles import LeadScoringPipelineBundle, SentimentBundle
from lume_platform.inference.registry import ModelRegistry


def broadcast_lead_bundle(spark: SparkSession, bundle: LeadScoringPipelineBundle):
    return spark.sparkContext.broadcast(pickle.dumps(bundle))


def score_partition_pdf(pdf: pd.DataFrame, bundle: LeadScoringPipelineBundle) -> pd.DataFrame:
    """Vectorized row scoring for one Pandas batch (used in ``mapInPandas``)."""
    preds: list[int] = []
    probas: list[float] = []
    cols = bundle.feature_columns
    for _, row in pdf.iterrows():
        payload = {c: row[c] if c in row.index else None for c in cols}
        p, pr = bundle.predict_row(payload)
        preds.append(p)
        probas.append(pr)
    out = pdf.copy()
    out["pred_label"] = preds
    out["pred_proba"] = probas
    return out


def make_map_in_pandas_fn(broadcast):
    """Return a generator for ``df.mapInPandas`` (Spark 3+)."""

    def _fn(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        bundle: LeadScoringPipelineBundle = pickle.loads(broadcast.value)
        for pdf in iterator:
            yield score_partition_pdf(pdf, bundle)

    return _fn


def output_schema_from_input(input_schema: StructType) -> StructType:
    """Append prediction fields to an existing DataFrame schema."""
    fields = list(input_schema.fields) + [
        StructField("pred_label", IntegerType(), True),
        StructField("pred_proba", DoubleType(), True),
    ]
    return StructType(fields)

# --- Real-Time UDFs (Singleton Registry based) ---

@udf(returnType=DoubleType())
def score_sentiment_udf(text: str) -> float:
    """Predict sentiment score (0-1) for a single piece of text."""
    if not text:
        return 0.5
    registry = ModelRegistry.get_instance()
    bundle = registry.sentiment_bundle
    if not bundle:
        return 0.5
    try:
        # LogisticRegression.predict_proba returns [ [prob_neg, prob_pos] ]
        # We want prob_pos
        proba = bundle.model.predict_proba(bundle.vectorizer.transform([text]))
        return float(proba[0][1])
    except:
        return 0.5

@udf(returnType=DoubleType())
def score_lead_probability_udf(total_visits, time_spent, page_views) -> float:
    """Predict lead conversion probability based on live activity."""
    registry = ModelRegistry.get_instance()
    bundle = registry.lead_bundle
    if not bundle:
        return 0.0
    try:
        payload = {
            "totalvisits": float(total_visits or 0),
            "total_time_spent_on_website": float(time_spent or 0),
            "page_views_per_visit": float(page_views or 0)
        }
        _, proba = bundle.predict_row(payload)
        return float(proba)
    except:
        return 0.0
