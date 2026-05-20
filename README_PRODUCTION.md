# 🚀 Lume AI — Production-Grade Big Data + AI System

A scalable, production-ready data platform for lead scoring, investor clustering, and sentiment analysis using **Apache Spark** and **AI/ML models**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange.svg)](https://spark.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
<!-- Streamlit is legacy; the active UI is the static frontend in /frontend. -->

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [AI/ML Models](#aiml-models)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [BI Integration](#bi-integration)
- [Google Colab Setup](#google-colab-setup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LUME AI PLATFORM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Static Front  │  │   FastAPI    │  │   Tableau    │          │
│  │   Frontend   │  │    Server    │  │   Embedded   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                  ┌────────▼────────┐                          │
│                  │  Model Registry │                          │
│                  │  (Pickle Bundles)│                          │
│                  └────────┬────────┘                          │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                   │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐          │
│  │  Lead       │  │  Investor   │  │  Sentiment  │          │
│  │  Scoring    │  │  Clustering │  │  NLP        │          │
│  │  (XGBoost)  │  │  (K-Means)  │  │  (TF-IDF)   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                   │
│         └────────────────┼────────────────┘                   │
│                          │                                   │
│                 ┌────────▼────────┐                          │
│                 │  Apache Spark   │                          │
│                 │  (31GB Dataset) │                          │
│                 └────────┬────────┘                          │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         │                │                │                 │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │ Structured  │  │ Semi-Struct │  │ Unstructured│        │
│  │ (CSV/Excel) │  │ (JSON/XML)  │  │ (Text/News) │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                 │
│  Data Sources:                                                  │
│  • Lead Scoring (~MB)                                          │
│  • NIFTY 500 Intraday (~19GB)                                  │
│  • Mutual Funds NAV (~GB)                                      │
│  • Social Sentiment (~MB)                                        │
│  • Investor Behavior (~MB)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone repository
cd /Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My\ Drive/BigData

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
PYTHONPATH=src python -m lume_platform.ml.training

# Start API server
PYTHONPATH=src uvicorn api.main_enhanced:app --reload --port 8000

# Start the static frontend (in new terminal)
cd frontend && python -m http.server 5500

# Export for BI tools
PYTHONPATH=src python scripts/bi_export.py
```

### Option 2: Google Colab (Free Cloud)

1. Open `notebooks/colab_production_pipeline.py` in Google Colab
2. Mount your Google Drive
3. Run all cells
4. Models and exports will be saved to your Drive

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for full dataset)
- Java 8+ (for Spark)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create `.env` file or set environment variables:

```bash
# Paths
export LUME_PROJECT_ROOT="/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData"
export LUME_DATA_ROOT="$LUME_PROJECT_ROOT/datasets"
export LUME_ARTIFACTS="$LUME_PROJECT_ROOT/artifacts"

# Spark Configuration
export LUME_SPARK_DRIVER_MEMORY="8g"
export LUME_SPARK_SHUFFLE_PARTITIONS="200"

# BI Tools
export LUME_TABLEAU_EMBED_URL="your_tableau_public_embed_url"

# API Configuration
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5000"
export FRONTEND_PORT="5500"
```

---

## 📊 Data Pipeline

### Apache Spark Processing

The system uses Apache Spark for distributed processing of the 31GB dataset:

```python
from pyspark.sql import SparkSession
from lume_platform.spark.pipeline import run_default_etl

# Initialize Spark
spark = SparkSession.builder \
    .appName("LumeAI") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Run full ETL pipeline
run_default_etl()
```

### Data Types Handled

| Type | Format | Processing |
|------|--------|------------|
| **Structured** | CSV, Excel | DataFrames, Parquet |
| **Semi-Structured** | JSON, XML | Schema inference, flattening |
| **Unstructured** | Text, News | TF-IDF, preprocessing |
| **Velocity** | Streaming | Micro-batch processing |

### Output Formats

- **Parquet**: Optimized columnar storage
- **CSV**: BI tool compatibility
- **Pickle**: Serialized ML models

---

## 🤖 AI/ML Models

### 1. Lead Classification Model

**Algorithm**: XGBoost (or RandomForest fallback)  
**Task**: Binary classification (converted/not converted)  
**Features**:
- Numeric: `totalvisits`, `total_time_spent_on_website`, `page_views_per_visit`
- Categorical: `lead_origin`, `lead_source`, `specialization`, `occupation`
- Engineered: `engagement_score`, `profile_strength`, `engagement_tier`

**Performance**:
```
Accuracy: ~85%
F1 Score: ~0.82
ROC-AUC: ~0.88
CV F1: 0.80 ± 0.03
```

### 2. Investor Clustering Model

**Algorithm**: K-Means (k=4) + DBSCAN  
**Task**: Unsupervised clustering of investor personas  
**Features**: Investment behavior attributes (0-10 scale)
- `ProfManage` — Professional management preference
- `Diversification` — Portfolio diversification
- `Affordability` — Cost sensitivity
- `Liquidity` — Cash flow needs
- `Growth` — Growth vs. stability
- `Trustworthiness` — Trust in institutions
- `Technology` — Tech adoption

**Clusters**:
| ID | Persona | Recommended Products |
|----|---------|---------------------|
| 0 | Equity Schemes (High Risk) | Lume Elite Advisory |
| 1 | Liquid/Debt Funds (Safe) | MintLeads Premier Capital |
| 2 | Hybrid Allocation Funds | Global Wealth Partners India |
| 3 | Index Trackers (Passive) | Index Advisory Group |

### 3. NLP Sentiment Model

**Algorithm**: TF-IDF + Logistic Regression  
**Task**: Text sentiment classification  
**Features**:
- TF-IDF vectorizer (max_features=20000, ngram_range=(1,2))
- Stop word removal
- Term frequency normalization

**Classes**: Positive, Negative, Neutral

---

## 🌐 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00",
  "models_loaded": {
    "lead_bundle": true,
    "investor_bundle": true,
    "sentiment_bundle": true
  },
  "api_version": "2.0.0"
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json
```

**Lead Scoring**:
```json
{
  "task": "lead_scoring",
  "lead": {
    "TotalVisits": 5,
    "Total Time Spent on Website": 800,
    "Page Views Per Visit": 2.5,
    "Asymmetrique Activity Score": 15,
    "Asymmetrique Profile Score": 15,
    "Lead Origin": "API",
    "Lead Source": "Organic Search"
  }
}
```

**Response**:
```json
{
  "task": "lead_scoring",
  "timestamp": "2024-01-15T10:30:00",
  "prediction": {
    "converted": true,
    "conversion_probability": 0.8765
  },
  "confidence": 0.8765,
  "model_version": "2.0.0"
}
```

**Investor Clustering**:
```json
{
  "task": "investor_cluster",
  "investor_behavior": {
    "ProfManage": 7,
    "Diversification": 6,
    "Affordability": 5,
    "Liquidity": 4,
    "Growth": 8,
    "Trustworthiness": 7,
    "Technology": 6
  }
}
```

**Response**:
```json
{
  "task": "investor_cluster",
  "timestamp": "2024-01-15T10:30:00",
  "prediction": {
    "cluster_id": 0,
    "persona": "Equity Schemes (High Risk)"
  },
  "model_version": "2.0.0"
}
```

**Sentiment Analysis**:
```json
{
  "task": "sentiment",
  "text": "Markets rally after policy support for infrastructure."
}
```

**Response**:
```json
{
  "task": "sentiment",
  "timestamp": "2024-01-15T10:30:00",
  "prediction": {
    "sentiment": "positive"
  },
  "confidence": 0.9234,
  "model_version": "2.0.0"
}
```

#### Batch Prediction
```bash
POST /batch_predict
Content-Type: application/json
```

```json
{
  "task": "lead_scoring",
  "leads": [
    {"TotalVisits": 5, ...},
    {"TotalVisits": 3, ...}
  ]
}
```

#### Analytics
```bash
GET /analytics
```

Returns aggregated statistics from production data.

#### Insights
```bash
GET /insights
```

Returns model information and BI tool URLs.

#### Model Information
```bash
GET /model/{model_name}/info
```

Available models: `lead_scoring`, `investor_cluster`, `sentiment`

---

## 📈 Dashboard

### Static Frontend Dashboard

The active dashboard is the static frontend in [`frontend/`](frontend/). It provides:
- Dataset audit and exploration
- Real-time AI predictions
- Data visualizations
- Model performance metrics
- BI tool embedding

**Run**:
```bash
cd frontend && python -m http.server 5500
```

**Access**: http://localhost:8501

### Dashboard Pages

1. **Dataset Audit** — Pipeline statistics and data quality
2. **Command Center** — System overview
3. **Model Evidence** — Evaluation reports and visualizations
4. **Distributor Leads** — Lead scoring interface
5. **Investor Clustering** — Persona assignment
6. **Semantic Fund Search** — TF-IDF fund retrieval
7. **NLP Sentiment** — Text classification
8. **LSTM Forecast** — Time series predictions
9. **BI Dashboard** — Embedded Tableau Public

---

## 📊 BI Integration

### Tableau Public

**Export Data**:
```bash
python scripts/bi_export.py
```

**Files Generated**:
- `tableau_exports/leads_for_bi.csv`
- `tableau_exports/investor_clusters_for_bi.csv`
- `tableau_exports/sentiment_for_bi.csv`
- `tableau_exports/summary_metrics.csv`

**Integration Steps**:
1. Go to [Tableau Public](https://public.tableau.com/)
2. Create new workbook
3. Connect to Text File → Select CSV
4. Build visualizations
5. Publish workbook
6. Get embed URL from Share → Embed Code
7. Set environment variable: `LUME_TABLEAU_EMBED_URL`

### Power BI

1. Open Power BI Desktop
2. Get Data → Text/CSV
3. Select exported CSV files
4. Build reports
5. Publish to Power BI Service
6. Embed in application

---

## ☁️ Google Colab Setup

### Step 1: Upload Repository to Google Drive

```bash
# Local: Copy to Drive
rsync -av --exclude='venv' --exclude='__pycache__' \
  /path/to/BigData/ \
  "/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData/"
```

### Step 2: Open Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/colab_production_pipeline.py`
3. Run all cells

### Step 3: Spark Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LumeAI") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```

### Step 4: Access Outputs

All outputs are saved to your Google Drive:
- Models: `artifacts/models/`
- Cleaned data: `artifacts/cleaned_parquet/`
- BI exports: `tableau_exports/`

---

## 📁 Project Structure

```
BigData/
├── api/                          # FastAPI application
│   ├── main.py                   # Basic API
│   └── main_enhanced.py          # Enhanced API with batch prediction
├── artifacts/                    # Generated artifacts
│   ├── cleaned_parquet/          # Spark output
│   └── models/                   # Serialized ML models
├── datasets/                     # Raw data (~31GB)
│   ├── structured/               # CSV, Excel files
│   ├── semi_structured/          # JSON, XML
│   ├── unstructured/             # Text, documents
│   └── velocity/                 # Streaming data
├── model_evaluations/            # Model performance reports
├── notebooks/                    # Jupyter/Colab notebooks
│   └── colab_production_pipeline.py  # Colab script
├── output_production_final/      # Production exports
├── scripts/                      # Utility scripts
│   ├── bi_export.py              # BI tool export
│   └── search_algorithm.py       # TF-IDF search
├── src/                          # Source code
│   └── lume_platform/
│       ├── audit/                # Data audit tools
│       ├── config.py             # Configuration
│       ├── inference/            # Model inference
│       ├── ml/                   # ML training
│       └── spark/                # Spark ETL
├── frontend/                     # Static dashboard UI
├── tableau_exports/              # BI tool exports
├── requirements.txt              # Dependencies
└── README_PRODUCTION.md          # This file
```

---

## 🔧 Configuration

### Spark Tuning

For large datasets (>10GB), adjust Spark settings:

```python
# In notebooks or config
SPARK_CONFIG = {
    "spark.driver.memory": "8g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "100",
    "spark.sql.adaptive.enabled": "true"
}
```

### Model Hyperparameters

Edit in `src/lume_platform/ml/training.py`:

```python
# XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9
}

# K-Means
KMEANS_CLUSTERS = 4
```

---

## 🧪 Testing

### Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### API Testing

```bash
# Using curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task": "sentiment",
    "text": "Great investment opportunity!"
  }'
```

```bash
# Using httpie
http POST localhost:8000/predict task=sentiment text="Positive news"
```

---

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t lume-ai .

# Run container
docker run -p 8000:8000 -p 8501:8501 lume-ai
```

### Docker Compose

```bash
docker-compose up -d
```

### Cloud Deployment

**Google Cloud Run**:
```bash
gcloud run deploy lume-ai --source .
```

**AWS ECS**:
```bash
aws ecs create-service --cluster lume-ai-cluster ...
```

---

## 📈 Performance Benchmarks

### Dataset Processing

| Dataset | Size | Processing Time | Output Format |
|---------|------|-----------------|---------------|
| Lead Scoring | ~10 MB | 2s | Parquet |
| NIFTY 500 Intraday | ~19 GB | 5 min | Parquet |
| Mutual Funds NAV | ~2 GB | 45s | Parquet |
| Social Sentiment | ~50 MB | 5s | Parquet |

### Model Training

| Model | Training Time | Inference (Single) | Inference (Batch 1000) |
|-------|--------------|-------------------|----------------------|
| Lead Classifier | 30s | 5ms | 50ms |
| Investor Clustering | 2s | 2ms | 10ms |
| Sentiment NLP | 15s | 3ms | 30ms |

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce Spark memory
export LUME_SPARK_DRIVER_MEMORY="4g"
```

**2. Model Not Found**
```bash
# Run training first
PYTHONPATH=src python -m lume_platform.ml.training
```

**3. Port Already in Use**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

**4. Spark Session Fails**
```bash
# Install Java
brew install openjdk@8
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

---

## 📝 License

MIT License — See LICENSE file for details.

---

## 🙏 Acknowledgments

- Apache Spark for distributed computing
- scikit-learn for ML algorithms
- FastAPI for API framework
- Static HTML/CSS/JS frontend for dashboard
- Tableau Public for BI visualization

---

## 📞 Support

For questions or issues:
- Open GitHub Issue
- Email: support@lume-ai.com
- Documentation: [docs.lume-ai.com](https://docs.lume-ai.com)

---

**Built with ❤️ by the Lume AI Engineering Team**
