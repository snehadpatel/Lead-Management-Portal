"""
Train core models (classification, clustering, NLP) and persist .pkl bundles.

Spark integration: train on Spark-written Parquet by swapping `read_csv` for `spark.read.parquet`.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from lume_platform.config import CLEANED_DIR, DATA_ROOT, EVAL_DIR, MODELS_DIR, PROJECT_ROOT, ensure_dirs
from lume_platform.ml.bundles import InvestorClusterBundle, LeadScoringPipelineBundle, SentimentBundle
from lume_platform.ml.eval_reports import (
    write_kmeans_sidecar,
    write_lead_scoring_markdown,
    write_master_report,
    write_nlp_markdown,
)
from lume_platform.ml.feature_config import LEAD_CATEGORICAL_FEATURES, LEAD_NUMERIC_FEATURES

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _load_smart_data(parquet_subpath: str, csv_fallback_path: Path) -> pd.DataFrame:
    """Prefer Spark Parquet; fallback to raw CSV for backward compatibility."""
    pq = CLEANED_DIR / parquet_subpath
    if pq.exists() and any(pq.iterdir()):
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print(f"⚠️ Failed to read Parquet at {pq}: {e}. Falling back to CSV.")
    
    if csv_fallback_path.exists():
        df = pd.read_csv(csv_fallback_path, low_memory=False)
        # Normalize CSV columns to match Spark (lowercase/snake_case)
        df.columns = df.columns.str.lower().str.replace(" ", "_"
                                      ).str.replace("(", ""
                                      ).str.replace(")", ""
                                      ).str.replace("-", "_")
        return df
    
    raise FileNotFoundError(f"Data not found at {pq} or {csv_fallback_path}")


def _save_confusion(y_true, y_pred, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _threshold_search(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose decision threshold maximizing F1 on holdout probabilities."""
    best_t = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    for t in np.arange(0.25, 0.81, 0.02):
        yp = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, yp, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_prec = float(precision_score(y_true, yp, zero_division=0))
            best_rec = float(recall_score(y_true, yp, zero_division=0))
    return best_t, {"f1": best_f1, "precision": best_prec, "recall": best_rec}


def train_lead_classifier() -> Path:
    ensure_dirs()
    csv_path = DATA_ROOT / "structured/leads/lead_scoring/Lead Scoring.csv"
    df = _load_smart_data("lead_scoring_clean", csv_path)
    numeric_features = list(LEAD_NUMERIC_FEATURES)
    cat_features = list(LEAD_CATEGORICAL_FEATURES)
    use_cols = numeric_features + cat_features + ["converted"]
    df_ml = df[[c for c in use_cols if c in df.columns]].copy()
    for c in numeric_features:
        if c not in df_ml.columns:
            df_ml[c] = 0
    for c in cat_features:
        if c not in df_ml.columns:
            df_ml[c] = "Unknown"
    df_ml[cat_features] = df_ml[cat_features].fillna("Unknown").astype(str)
    df_ml[numeric_features] = df_ml[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df_ml.drop(columns=["converted"])
    y = df_ml["converted"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if HAS_XGB:
        xgb_grid = list(
            ParameterGrid(
                {
                    "n_estimators": [180, 240],
                    "max_depth": [5, 6],
                    "learning_rate": [0.04, 0.06],
                    "subsample": [0.85, 0.95],
                    "colsample_bytree": [0.85, 0.95],
                }
            )
        )
        best_cfg = None
        best_cv_f1 = -1.0
        skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for cfg in xgb_grid:
            base = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                **cfg,
            )
            preprocess = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
                ]
            )
            pipe_try: Pipeline = Pipeline([("prep", preprocess), ("clf", base)])
            cv_scores = cross_val_score(pipe_try, X_train, y_train, cv=skf_inner, scoring="f1")
            cv_f1 = float(np.mean(cv_scores))
            if cv_f1 > best_cv_f1:
                best_cv_f1 = cv_f1
                best_cfg = cfg
        clf = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            **(best_cfg or {}),
        )
        model_name = "xgboost"
    else:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
        )
        model_name = "random_forest"
        best_cfg = {"n_estimators": 200, "max_depth": 12, "class_weight": "balanced"}

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ]
    )
    pipe: Pipeline = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    best_threshold, t_stats = _threshold_search(y_test, proba_test)
    preds = (proba_test >= best_threshold).astype(int)
    metrics = {
        "accuracy": f"{accuracy_score(y_test, preds) * 100:.2f}%",
        "precision": f"{precision_score(y_test, preds, zero_division=0):.2f}",
        "recall": f"{recall_score(y_test, preds, zero_division=0):.2f}",
        "f1_score": f"{f1_score(y_test, preds, zero_division=0):.2f}",
        "backend": model_name,
        "feature_numeric": numeric_features,
        "feature_categorical": cat_features,
        "decision_threshold": round(best_threshold, 4),
        "threshold_precision": round(t_stats["precision"], 4),
        "threshold_recall": round(t_stats["recall"], 4),
        "threshold_f1": round(t_stats["f1"], 4),
        "tuned_params": best_cfg,
    }
    try:
        metrics["roc_auc"] = f"{roc_auc_score(y_test, proba_test):.3f}"
    except ValueError:
        metrics["roc_auc"] = "n/a"
    metrics["average_precision"] = f"{average_precision_score(y_test, proba_test):.3f}"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
    metrics["cv_f1_mean"] = f"{float(np.mean(cv)):.3f}"
    metrics["cv_f1_std"] = f"{float(np.std(cv)):.3f}"

    eval_rf = EVAL_DIR / "random_forest"
    eval_rf.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(y_test, preds, output_dict=True)
    with open(eval_rf / "real_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _save_confusion(y_test, preds, eval_rf / "rf_confusion_matrix.png", "Lead conversion confusion matrix")
    with open(eval_rf / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    write_lead_scoring_markdown(eval_rf, metrics, report_dict)

    bundle = LeadScoringPipelineBundle(
        pipeline=pipe,
        numeric_features=numeric_features,
        cat_features=cat_features,
        decision_threshold=best_threshold,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "lead_classifier_bundle.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    with open(MODELS_DIR / "distributor_lead_scorer.pkl", "wb") as f:
        pickle.dump(pipe, f)
    legacy = DATA_ROOT / "models/saved_models"
    legacy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, legacy / "lead_classifier_bundle.pkl")
    shutil.copy2(MODELS_DIR / "distributor_lead_scorer.pkl", legacy / "distributor_lead_scorer.pkl")

    df_out = df.copy()
    df_out["Conversion_Probability"] = pipe.predict_proba(X)[:, 1]
    df_out["Converted_Prediction"] = (df_out["Conversion_Probability"] >= best_threshold).astype(int)

    def assign_pitch_persona(t):
        if t > 1500:
            return "Aggressive Growth Chaser"
        if t > 800:
            return "Brand Loyalist"
        return "Conservative Wealth Preserver"

    df_out["Recommended_Pitch_Persona"] = df_out["total_time_spent_on_website"].fillna(0).apply(assign_pitch_persona)
    export_dir = Path(__file__).resolve().parents[2] / "output_production_final"
    export_dir.mkdir(parents=True, exist_ok=True)
    df_out[df_out["Conversion_Probability"] >= best_threshold].sort_values("Conversion_Probability", ascending=False).to_csv(
        export_dir / "distributor_leads_master.csv", index=False
    )
    with open(eval_rf / "best_model.json", "w") as f:
        json.dump(
            {
                "backend": model_name,
                "decision_threshold": best_threshold,
                "tuned_params": best_cfg,
                "summary_metrics": {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics.get("roc_auc"),
                    "average_precision": metrics.get("average_precision"),
                },
            },
            f,
            indent=2,
        )
    return out


def train_investor_kmeans_and_dbscan() -> tuple[Path, Path | None]:
    ensure_dirs()
    xlsx = DATA_ROOT / "structured/leads/mf_investor_behavior/MF_Behavior.xlsx"
    # Note: Spark pipeline currently doesn't clean the Excel behavior file in the default etl script,
    # so we prioritize the original Excel for now, but keep the door open for parquet.
    pq = CLEANED_DIR / "investor_behavior_clean"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_excel(xlsx)
    behavior_cols = [
        "ProfManage",
        "Diversification",
        "Affordability",
        "Liquidity",
        "Growth",
        "Trustworthiness",
        "Technology",
    ]
    X = df[behavior_cols].dropna()
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df.loc[X.index, "Persona_Cluster"] = labels
    sil = float(silhouette_score(Xs, labels))
    write_kmeans_sidecar(EVAL_DIR / "kmeans", sil, int(len(np.unique(labels))))

    def route_investor(cluster: int) -> tuple[str, str]:
        if cluster == 0:
            return "Equity Schemes (High Risk)", "Lume Elite Advisory"
        if cluster == 1:
            return "Liquid/Debt Funds (Safe)", "MintLeads Premier Capital"
        if cluster == 2:
            return "Hybrid Allocation Funds", "Global Wealth Partners India"
        return "Index Trackers (Passive)", "Index Advisory Group"

    routes = df.loc[X.index, "Persona_Cluster"].astype(int).apply(route_investor)
    df.loc[X.index, "Recommended_Fund_Type"] = routes.apply(lambda t: t[0])
    df.loc[X.index, "Recommended_Distributor_To_Contact"] = routes.apply(lambda t: t[1])

    export_dir = Path(__file__).resolve().parents[2] / "output_production_final"
    export_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(export_dir / "investor_routing_matches.csv", index=False)

    bundle = InvestorClusterBundle(scaler=scaler, kmeans=km, behavior_cols=behavior_cols)
    p1 = MODELS_DIR / "investor_cluster_bundle.pkl"
    with open(p1, "wb") as f:
        pickle.dump(bundle, f)
    with open(MODELS_DIR / "investor_kmeans_model.pkl", "wb") as f:
        pickle.dump(km, f)
    with open(MODELS_DIR / "investor_behavior_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    legacy = DATA_ROOT / "models/saved_models"
    legacy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p1, legacy / "investor_cluster_bundle.pkl")
    shutil.copy2(MODELS_DIR / "investor_kmeans_model.pkl", legacy / "investor_kmeans_model.pkl")
    shutil.copy2(MODELS_DIR / "investor_behavior_scaler.pkl", legacy / "investor_behavior_scaler.pkl")

    db_path = None
    try:
        db = DBSCAN(eps=0.35, min_samples=10).fit(Xs)
        with open(MODELS_DIR / "investor_dbscan.pkl", "wb") as f:
            pickle.dump(db, f)
        db_path = MODELS_DIR / "investor_dbscan.pkl"
    except Exception:
        pass
    return p1, db_path


def train_sentiment_nlp() -> Path:
    ensure_dirs()
    csv_path = DATA_ROOT / "semi_structured/social_sentiment/data.csv"
    df = _load_smart_data("social_sentiment_clean", csv_path)
    text_col = "Sentence" if "Sentence" in df.columns else df.columns[0]
    label_col = "Sentiment" if "Sentiment" in df.columns else df.columns[1]
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df[label_col].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str), y, test_size=0.2, random_state=42, stratify=y
    )
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    eval_nlp = EVAL_DIR / "nlp_sentiment"
    eval_nlp.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }
    with open(eval_nlp / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_nlp_markdown(eval_nlp, metrics)
    bundle = SentimentBundle(vectorizer=vec, model=clf, label_classes=label_enc.classes_)
    out = MODELS_DIR / "sentiment_bundle.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    legacy = DATA_ROOT / "models/saved_models"
    legacy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, legacy / "sentiment_bundle.pkl")
    return out


def train_all() -> dict[str, str]:
    import time
    start = time.time()
    paths = {
        "lead_classifier": str(train_lead_classifier()),
        "investor_cluster": str(train_investor_kmeans_and_dbscan()[0]),
        "sentiment": str(train_sentiment_nlp()),
    }
    
    # Generate model manifest for the 'Startup' Dashboard / API
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "training_duration_sec": round(time.time() - start, 2),
        "models": {}
    }
    
    # Harvest metrics from JSON files created during training
    for key, folder in [("leads", "random_forest/real_metrics.json"), 
                       ("nlp", "nlp_sentiment/metrics.json"),
                       ("clustering", "kmeans/kmeans_metrics.json")]:
        p = EVAL_DIR / folder
        if p.exists():
            with open(p) as f:
                manifest["models"][key] = json.load(f)

    with open(MODELS_DIR / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    write_master_report(PROJECT_ROOT)
    return paths


if __name__ == "__main__":
    print(json.dumps(train_all(), indent=2))
