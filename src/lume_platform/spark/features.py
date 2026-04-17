"""Feature engineering in Spark: encoding, scaling, TF-IDF for text columns."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler, HashingTF, IDF, Tokenizer


def index_and_onehot(_df: DataFrame, cat_cols: list[str]) -> Pipeline:
    stages = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        idx = f"{c}_idx"
        ohe = f"{c}_ohe"
        si = StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
        enc = OneHotEncoder(inputCols=[idx], outputCols=[ohe])
        stages.extend([si, enc])
    return Pipeline(stages=stages)


def scale_numeric_assembler(df: DataFrame, numeric_cols: list[str], out_col: str = "num_features") -> Pipeline:
    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        raise ValueError("No numeric columns found for scaling")
    assembler = VectorAssembler(inputCols=present, outputCol="num_vec", handleInvalid="skip")
    scaler = MinMaxScaler(inputCol="num_vec", outputCol=out_col)
    return Pipeline(stages=[assembler, scaler])


def tf_idf_pipeline(input_col: str, out_col: str = "tfidf") -> Pipeline:
    tok = Tokenizer(inputCol=input_col, outputCol="tokens")
    ht = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=1 << 16)
    idf = IDF(inputCol="raw_features", outputCol=out_col)
    return Pipeline(stages=[tok, ht, idf])
