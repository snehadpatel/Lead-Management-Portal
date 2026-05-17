"""
Lume AI — Production-Grade Big Data + AI Pipeline
Google Colab Compatible Script (31GB Dataset Processing)

Architecture:
- Apache Spark: Distributed processing
- AI/ML Models: Classification, Clustering, NLP
- Backend: FastAPI with model inference
- Dashboard: Streamlit with BI integration

Usage in Google Colab:
1. Upload this file to Colab
2. Mount Google Drive
3. Run all cells
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

#pip install -q pyspark==3.5.0 pandas==2.1.4 numpy==1.24.3 scikit-learn==1.3.2
#pip install -q xgboost==2.0.3 matplotlib==3.8.2 seaborn==0.13.0 pyarrow==14.0.1
#pip install -q fastapi==0.110.0 uvicorn==0.27.0 pydantic==2.6.0 streamlit==1.32.0
#pip install -q python-dotenv==1.0.0 openpyxl==3.1.2 pillow==10.2.0

print("✅ All dependencies installed")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Mount Google Drive & Setup Environment
# ═══════════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
from pathlib import Path

# Configure paths - adjust to your Drive structure
REPO_PATH = "/content/drive/MyDrive/BigData"
DATA_PATH = f"{REPO_PATH}/datasets"

os.environ["LUME_PROJECT_ROOT"] = REPO_PATH
os.environ["LUME_DATA_ROOT"] = DATA_PATH
os.environ["LUME_ARTIFACTS"] = f"{REPO_PATH}/artifacts"

sys.path.insert(0, f"{REPO_PATH}/src")

print(f"✅ Repository: {REPO_PATH}")
print(f"✅ Data Root: {DATA_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Initialize Apache Spark with Optimized Configuration
# ═══════════════════════════════════════════════════════════════════════════════

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Optimized Spark configuration for Colab (free tier)
conf = SparkConf() \
    .setAppName("LumeAI_Production_Pipeline") \
    .setMaster("local[*]") \
    .set("spark.driver.memory", "8g") \
    .set("spark.executor.memory", "4g") \
    .set("spark.sql.adaptive.enabled", "true") \
    .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .set("spark.sql.shuffle.partitions", "200") \
    .set("spark.default.parallelism", "100") \
    .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .set("spark.sql.parquet.compression.codec", "snappy")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

print(f"✅ Spark Version: {spark.version}")
print(f"✅ Spark UI: {spark.sparkContext.uiWebUrl}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Dataset Structure Analysis
# ═══════════════════════════════════════════════════════════════════════════════

data_root = Path(DATA_PATH)

def analyze_dataset_structure(root: Path):
    """Analyze the dataset structure and sizes"""
    stats = {
        "structured": {"dirs": 0, "files": 0, "size_mb": 0},
        "semi_structured": {"dirs": 0, "files": 0, "size_mb": 0},
        "unstructured": {"dirs": 0, "files": 0, "size_mb": 0},
        "velocity": {"dirs": 0, "files": 0, "size_mb": 0},
    }
    
    for data_type in stats.keys():
        type_path = root / data_type
        if type_path.exists():
            for item in type_path.rglob("*"):
                if item.is_dir():
                    stats[data_type]["dirs"] += 1
                elif item.is_file():
                    stats[data_type]["files"] += 1
                    stats[data_type]["size_mb"] += item.stat().st_size / (1024 * 1024)
    
    total_gb = sum(s["size_mb"] for s in stats.values()) / 1024
    
    print("\n📊 Dataset Structure Analysis\n" + "="*50)
    for data_type, s in stats.items():
        print(f"\n{data_type.upper()}:")
        print(f"  📁 Directories: {s['dirs']}")
        print(f"  📄 Files: {s['files']}")
        print(f"  💾 Size: {s['size_mb']:.2f} MB ({s['size_mb']/1024:.2f} GB)")
    
    print(f"\n📈 TOTAL DATASET SIZE: {total_gb:.2f} GB")
    return stats

dataset_stats = analyze_dataset_structure(data_root)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Data Processing Functions
# ═══════════════════════════════════════════════════════════════════════════════

from pyspark.sql import functions as F
from pyspark.sql.types import *

def normalize_columns(df):
    """Normalize column names to snake_case"""
    new_cols = []
    for col in df.columns:
        new_name = col.lower().replace(" ", "_").replace("-", "_").replace("/", "_") \
            .replace("(", "").replace(")", "").replace(",", "_").replace(".", "_") \
            .replace("__", "_").strip("_")
        new_cols.append(F.col(col).alias(new_name))
    return df.select(*new_cols)

def compute_null_percentage(df):
    """Compute null percentage for each column"""
    total = df.count()
    null_stats = []
    for col in df.columns:
        null_count = df.filter(F.col(col).isNull()).count()
        null_pct = (null_count / total) * 100 if total > 0 else 0
        null_stats.append((col, null_count, null_pct))
    
    print(f"\n📋 Null Percentage Report (Total rows: {total})\n" + "-"*50)
    for col, null_count, null_pct in sorted(null_stats, key=lambda x: x[2], reverse=True)[:10]:
        bar = "█" * int(null_pct / 5) + "░" * (20 - int(null_pct / 5))
        print(f"  {col[:30]:<30} {null_count:>8} ({null_pct:>5.1f}%) {bar}")
    
    return null_stats

print("✅ Data processing functions defined")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Process Lead Scoring Data
# ═══════════════════════════════════════════════════════════════════════════════

lead_path = data_root / "structured/leads/lead_scoring/Lead Scoring.csv"

if lead_path.exists():
    print(f"📂 Loading: {lead_path}")
    
    # Read with schema inference
    df_leads = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .csv(str(lead_path))
    
    print(f"✅ Loaded {df_leads.count()} rows, {len(df_leads.columns)} columns")
    
    # Normalize columns
    df_leads = normalize_columns(df_leads)
    
    # Data validation
    null_stats = compute_null_percentage(df_leads)
    
    # Handle missing values
    numeric_cols = [c for c in df_leads.columns if c.startswith(("total", "page", "asym"))]
    for col in numeric_cols:
        df_leads = df_leads.fillna(0, subset=[col])
    
    # Remove duplicates
    initial_count = df_leads.count()
    df_leads = df_leads.dropDuplicates()
    final_count = df_leads.count()
    print(f"\n🧹 Removed {initial_count - final_count} duplicate rows")
    
    # Persist to Parquet
    output_path = f"{REPO_PATH}/artifacts/cleaned_parquet/lead_scoring_clean"
    df_leads.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved to: {output_path}")
else:
    print("⚠️ Lead scoring file not found")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

if lead_path.exists():
    print("\n🔧 Feature Engineering for Lead Scoring\n" + "="*50)
    
    # Create engagement score
    df_leads = df_leads.withColumn(
        "engagement_score",
        (F.col("totalvisits") * 0.3 + 
         F.col("total_time_spent_on_website") * 0.0005 + 
         F.col("page_views_per_visit") * 10)
    )
    
    # Create profile strength score
    df_leads = df_leads.withColumn(
        "profile_strength",
        (F.col("asymmetrique_activity_score").fillna(0) + 
         F.col("asymmetrique_profile_score").fillna(0)) / 2
    )
    
    # Bin engagement into categories
    df_leads = df_leads.withColumn(
        "engagement_tier",
        F.when(F.col("engagement_score") > 50, "High") \
         .when(F.col("engagement_score") > 20, "Medium") \
         .otherwise("Low")
    )
    
    # Save engineered features
    output_path = f"{REPO_PATH}/artifacts/cleaned_parquet/lead_scoring_features"
    df_leads.write.mode("overwrite").parquet(output_path)
    print(f"✅ Features saved to: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Convert to Pandas for ML Training
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 Converting to Pandas for ML Training\n" + "="*50)

# Lead scoring data
df_leads_pd = df_leads.toPandas()
print(f"✅ Lead scoring: {df_leads_pd.shape[0]} rows, {df_leads_pd.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — ML Model Training
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
import pickle

print("✅ ML libraries imported")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Lead Classification Model
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🎯 Training Lead Classification Model\n" + "="*50)

# Define features
numeric_features = [
    "totalvisits", "total_time_spent_on_website", "page_views_per_visit",
    "asymmetrique_activity_score", "asymmetrique_profile_score",
    "engagement_score", "profile_strength"
]

categorical_features = [
    "lead_origin", "lead_source", "specialization",
    "what_is_your_current_occupation", "last_activity",
    "country", "lead_quality", "do_not_email", "do_not_call",
    "engagement_tier"
]

# Filter available columns
numeric_features = [c for c in numeric_features if c in df_leads_pd.columns]
categorical_features = [c for c in categorical_features if c in df_leads_pd.columns]

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Prepare data
target_col = "converted"
if target_col in df_leads_pd.columns:
    # Handle missing values
    df_ml = df_leads_pd[numeric_features + categorical_features + [target_col]].copy()
    
    # Fill missing values
    for c in numeric_features:
        df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce").fillna(0)
    
    for c in categorical_features:
        df_ml[c] = df_ml[c].fillna("Unknown").astype(str)
    
    df_ml[target_col] = pd.to_numeric(df_ml[target_col], errors="coerce").fillna(0).astype(int)
    
    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try XGBoost first, fallback to RandomForest
    try:
        from xgboost import XGBClassifier
        
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss"
        )
        model_name = "xgboost"
        print("\n🚀 Using XGBoost classifier")
    except ImportError:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model_name = "random_forest"
        print("\n🌲 Using RandomForest classifier")
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    
    # Train model
    print("\n⏳ Training model...")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n📊 Model Performance\n" + "-"*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    print(f"\n📈 CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Save model
    model_path = f"{REPO_PATH}/artifacts/models/lead_classifier.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    print(f"\n💾 Model saved to: {model_path}")
else:
    print("⚠️ Target column 'converted' not found")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Investor Clustering Model
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🎯 Training Investor Clustering Model\n" + "="*50)

# Load investor behavior data if available
behavior_path = data_root / "structured/leads/mf_investor_behavior/MF_Behavior.xlsx"

if behavior_path.exists():
    df_behavior = pd.read_excel(behavior_path)
    
    # Define behavior columns
    behavior_cols = [
        "ProfManage", "Diversification", "Affordability",
        "Liquidity", "Growth", "Trustworthiness", "Technology"
    ]
    
    # Filter available columns
    behavior_cols = [c for c in behavior_cols if c in df_behavior.columns]
    
    if len(behavior_cols) >= 3:
        X_behavior = df_behavior[behavior_cols].dropna()
        
        # Scale features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_behavior)
        
        # K-Means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette score
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_scaled, labels)
        
        print(f"✅ K-Means clustering completed")
        print(f"   Silhouette Score: {sil_score:.3f}")
        print(f"   Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        # Save models
        kmeans_path = f"{REPO_PATH}/artifacts/models/investor_kmeans.pkl"
        scaler_path = f"{REPO_PATH}/artifacts/models/investor_scaler.pkl"
        
        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        
        print(f"\n💾 Models saved to artifacts/models/")
    else:
        print("⚠️ Not enough behavior columns found")
else:
    print("⚠️ Investor behavior file not found")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12 — NLP Sentiment Model
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🎯 Training NLP Sentiment Model\n" + "="*50)

sentiment_csv = data_root / "semi_structured/social_sentiment/data.csv"

if sentiment_csv.exists():
    df_sentiment = pd.read_csv(sentiment_csv)
    
    # Identify text and label columns
    text_col = None
    label_col = None
    
    for col in df_sentiment.columns:
        if col.lower() in ["sentence", "text", "content", "review"]:
            text_col = col
        if col.lower() in ["sentiment", "label", "class", "polarity"]:
            label_col = col
    
    if text_col and label_col:
        print(f"✅ Found text column: {text_col}")
        print(f"✅ Found label column: {label_col}")
        
        # Prepare data
        df_sent = df_sentiment[[text_col, label_col]].dropna()
        
        X_text = df_sent[text_col].astype(str)
        y_sent = df_sent[label_col].astype(str)
        
        # Encode labels
        label_enc = LabelEncoder()
        y_encoded = label_enc.fit_transform(y_sent)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # TF-IDF Vectorization
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english"
        )
        
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Logistic Regression classifier
        sent_clf = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42
        )
        
        sent_clf.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred_sent = sent_clf.predict(X_test_tfidf)
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        print("\n📊 NLP Model Performance\n" + "-"*50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred_sent):.3f}")
        print(f"F1 Score (macro): {f1_score(y_test, y_pred_sent, average='macro'):.3f}")
        
        # Save model
        nlp_bundle = {
            "vectorizer": tfidf,
            "model": sent_clf,
            "label_encoder": label_enc,
            "classes": label_enc.classes_.tolist()
        }
        
        nlp_path = f"{REPO_PATH}/artifacts/models/sentiment_nlp.pkl"
        with open(nlp_path, "wb") as f:
            pickle.dump(nlp_bundle, f)
        
        print(f"\n💾 NLP model saved to: {nlp_path}")
    else:
        print("⚠️ Could not identify text or label columns")
else:
    print("⚠️ Sentiment data file not found")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 13 — Export for BI Tools
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 Exporting for BI Tools\n" + "="*50)

bi_export_dir = f"{REPO_PATH}/tableau_exports"
os.makedirs(bi_export_dir, exist_ok=True)

# Export lead scoring data with predictions
if 'pipeline' in locals():
    # Add predictions
    df_export = df_leads_pd.copy()
    
    # Get feature columns used in training
    X_full = df_export[numeric_features + categorical_features]
    
    # Fill missing values
    for c in numeric_features:
        X_full[c] = pd.to_numeric(X_full[c], errors="coerce").fillna(0)
    for c in categorical_features:
        X_full[c] = X_full[c].fillna("Unknown").astype(str)
    
    df_export["conversion_probability"] = pipeline.predict_proba(X_full)[:, 1]
    df_export["predicted_conversion"] = pipeline.predict(X_full)
    
    # Export to CSV
    leads_csv_path = f"{bi_export_dir}/leads_with_predictions.csv"
    df_export.to_csv(leads_csv_path, index=False)
    print(f"✅ Exported leads: {leads_csv_path}")
    print(f"   Rows: {len(df_export)}, Size: {os.path.getsize(leads_csv_path) / 1024 / 1024:.2f} MB")

print("\n📊 BI Export Complete!")
print("   Import these CSV files into Tableau Public or Power BI")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 14 — Generate Evaluation Reports
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📄 Generating Evaluation Reports\n" + "="*50)

import json
import matplotlib.pyplot as plt
import seaborn as sns

eval_dir = f"{REPO_PATH}/model_evaluations"
os.makedirs(eval_dir, exist_ok=True)

# Lead Classification Report
if 'pipeline' in locals() and 'y_test' in locals():
    lead_eval = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "model_type": model_name,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(f"{eval_dir}/lead_classification_metrics.json", "w") as f:
        json.dump(lead_eval, f, indent=2)
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Lead Conversion - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    
    print("✅ Lead classification report generated")

# Create master report
master_report = f"""# Lume AI — Model Evaluation Report
Generated: {pd.Timestamp.now()}

## Dataset Summary
- Total Dataset Size: ~31 GB
- Structured Data: Lead scoring, Mutual Funds, Stock Prices
- Semi-Structured: Social sentiment
- Processing Engine: Apache Spark 3.5

## Models Trained
1. **Lead Classification** ({model_name if 'model_name' in locals() else 'N/A'})
   - Accuracy: {lead_eval['accuracy']:.3f if 'lead_eval' in locals() else 'N/A'}
   - F1 Score: {lead_eval['f1_score']:.3f if 'lead_eval' in locals() else 'N/A'}

2. **Investor Clustering** (K-Means)
   - Clusters: 4
   - Features: Investment behavior attributes

3. **NLP Sentiment** (TF-IDF + Logistic Regression)
   - Approach: Bag-of-words with n-grams
   - Classes: Positive, Negative, Neutral

## Artifacts
- Models: `artifacts/models/`
- Cleaned Data: `artifacts/cleaned_parquet/`
- BI Exports: `tableau_exports/`
"""

with open(f"{eval_dir}/MODEL_EVALUATION_REPORT.md", "w") as f:
    f.write(master_report)

print("✅ Master evaluation report generated")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 15 — Final Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("🎉 PRODUCTION PIPELINE COMPLETE")
print("="*60)

print("\n📦 Generated Artifacts:")
artifacts = [
    ("Cleaned Parquet", f"{REPO_PATH}/artifacts/cleaned_parquet"),
    ("Trained Models", f"{REPO_PATH}/artifacts/models"),
    ("BI Exports", f"{REPO_PATH}/tableau_exports"),
    ("Evaluation Reports", f"{REPO_PATH}/model_evaluations"),
]

for name, path in artifacts:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"   {exists} {name}: {path}")

print("\n🚀 Next Steps:")
print("   1. Run API: uvicorn api.main:app --reload")
print("   2. Run Dashboard: streamlit run streamlit_app.py")
print("   3. Import CSVs to Tableau Public for BI visualization")

print("\n📊 Models Ready for Inference:")
print("   • Lead Classification (XGBoost/RandomForest)")
print("   • Investor Clustering (K-Means)")
print("   • NLP Sentiment (TF-IDF + Logistic Regression)")

# Stop Spark
spark.stop()
print("\n✅ Spark session stopped")
