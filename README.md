# Lume AI: Big Data + AI Platform for Mutual Fund Intelligence

Lume AI is an end-to-end system that combines large-scale data processing, ML model training, API inference, and dashboard delivery for mutual fund lead intelligence.

The repository supports local execution and free cloud workflows (Google Colab + Drive), with model metrics stored in reproducible evaluation artifacts.

## What this project does

- Processes large, mixed-format datasets (CSV, JSON, text) using PySpark.
- Trains multiple AI workloads in one pipeline:
  - Lead conversion prediction (supervised classification)
  - Investor persona clustering (unsupervised learning)
  - Financial sentiment analysis and semantic search (NLP)
  - Optional sequence forecasting (LSTM script)
- Serves predictions via FastAPI and visualizes outputs via Streamlit.
- Exports analytics for BI tools (Tableau/Power BI).

## Core features

- **Scalable data pipeline**
  - Recursive Spark ingestion for nested data trees
  - Cleaning + feature processing + Parquet persistence
- **Multi-model AI layer**
  - Random Forest/XGBoost-style lead scoring flow
  - K-Means persona grouping
  - SBERT/TF-IDF style NLP inference paths
- **Production-style serving**
  - FastAPI inference endpoints
  - Batch-friendly model registry loading
  - Streamlit operator dashboard
- **Deployment options**
  - Local Python setup
  - Docker Compose stack (`api`, `dashboard`, `mongodb`, optional extras)
  - Colab notebook path for free runtime
- **Evaluation-first outputs**
  - JSON metrics as source of truth under `model_evaluations/`
  - Reports and plots generated from the same pipeline

## Architecture

| Layer | Key components |
|---|---|
| Data Engineering | `src/lume_platform/spark/`, Spark ETL, Parquet outputs |
| AI/ML | `src/lume_platform/ml/`, model training and evaluation |
| Inference | `src/lume_platform/inference/`, model bundles and registry |
| API | `api/main.py`, `api/main_enhanced.py` |
| UI | `streamlit_app.py` |
| Notebook Runtime | `notebooks/colab_spark_pipeline.ipynb` |

## Repository structure

```text
BigData/
├── api/                         # FastAPI services
├── src/lume_platform/           # Core data, ML, inference modules
├── scripts/                     # ETL, exports, optional training utilities
├── datasets/                    # Raw/source datasets
├── artifacts/                   # Generated models and pipeline outputs
├── model_evaluations/           # Metrics JSON + reports + plots
├── notebooks/                   # Colab/local notebooks
├── streamlit_app.py             # Dashboard app
├── docker-compose.yml           # Multi-service local deployment
└── README.md
```

## Quick start (local)

```bash
cd "/path/to/BigData"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

### 1) Train models and generate evaluation artifacts

```bash
python -m lume_platform.ml.training
```

### 2) Run API

```bash
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

For the enhanced API:

```bash
PYTHONPATH=src uvicorn api.main_enhanced:app --reload --port 8000
```

### 3) Run dashboard

```bash
streamlit run streamlit_app.py
```

## API overview

Common endpoints (available in base/enhanced variants depending on app):

- `POST /predict` - task-based inference (`lead_scoring`, `investor_cluster`, `sentiment`)
- `GET /analytics` - aggregate analytics from exported outputs
- `GET /insights` - model cards/metrics references + embed metadata
- `GET /health` - service health

Enhanced API includes additional operational endpoints (batch and domain-specific routes in `api/main_enhanced.py`).

## Example prediction payloads

Lead scoring:

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
    "Lead Source": "Organic Search",
    "Specialization": "Select",
    "What is your current occupation": "Unemployed",
    "Last Activity": "Page Visited on Website",
    "Country": "India",
    "Lead Quality": "Low in Relevance",
    "Do Not Email": "No",
    "Do Not Call": "No"
  }
}
```

Sentiment:

```json
{
  "task": "sentiment",
  "text": "Markets rally on strong GDP print"
}
```

## Model evaluation and metrics

Primary evaluation index:

- `model_evaluations/MODELS_EVALUATION_REPORT.md`

Source-of-truth metric JSON files:

- `model_evaluations/random_forest/real_metrics.json`
- `model_evaluations/kmeans/kmeans_metrics.json`
- `model_evaluations/nlp_sentiment/metrics.json`

Regenerate metrics:

```bash
export PYTHONPATH=src
python -m lume_platform.ml.training
```

## Data handling notes

- Structured data: large CSV trees (Spark recursive lookup)
- Semi-structured data: JSON and mixed files
- Unstructured data: textual/news content for NLP pipelines
- Intermediate outputs: Parquet under `artifacts/cleaned_parquet/`

## AI Phase (Model Development & Evaluation)

Purpose: develop, validate and produce serveable ML artifacts (models, evaluation metrics, visualizations) that address the project's prediction and intelligence objectives.

What the code provides:
- End-to-end training pipelines for supervised and unsupervised models (lead classifier, K-Means personas, NLP sentiment/search, optional LSTM forecaster).
- Hyperparameter search, cross-validation (5-fold StratifiedKFold), decision-threshold optimization for classification tasks.
- Automated metric export to JSON and visual reports (confusion matrix, ROC/PR, cluster plots) under `model_evaluations/`.

Key files and locations:
- Training orchestrator: `src/lume_platform/ml/training.py` and `src/master_pipeline.py` (`train_all()` entry).
- Model serving and registry: `src/lume_platform/inference/registry.py`, saved bundles in `artifacts/models/`.
- Evaluation outputs: `model_evaluations/<model>/real_metrics.json` and PNG visualizations.

Run (train + eval):
```bash
export PYTHONPATH=src
python -m lume_platform.ml.training
```

Produced artifacts:
- Serialized model bundles (pickle/serialized pipeline) in `artifacts/models/`.
- Machine-readable metrics JSON in `model_evaluations/` and human-friendly plots/MD reports.

How reviewers should evaluate the AI phase:
- Re-run training on a small sample to verify reproducibility of the pipeline and metrics.
- Inspect `model_evaluations/*/real_metrics.json` for Accuracy, Precision, Recall, F1, ROC-AUC and CV stats.
- Validate that decision threshold tuning steps are documented (search results recorded in JSON) and explainability outputs (feature importances) are present.
- Check that model bundles load cleanly via the registry by calling the `/predict` endpoint against a known sample.

Evaluation criteria recommended for the poster & review:
- Metrics: accuracy, precision, recall, F1, ROC-AUC and PR-AUC for classification; silhouette for clustering; MAE/RMSE for forecasting.
- Operational metrics: inference latency (ms), batch throughput, and resource utilization for representative loads.

Reproducibility notes:
- The training code is deterministic when seeds are fixed in `src/lume_platform/ml/training.py`.
- Cross-validation and hyperparameter results are exported to `model_evaluations/` to enable independent verification.


## Results & Validation

This section summarizes the model-specific outputs you can cite in the poster, demo, or viva. The evaluation folders contain the screenshots, charts, and metric tables used for validation.

| Model | Prototype output screenshots | Dashboard / visualizations | Performance metrics | Charts / tables | Resource utilization |
|---|---|---|---|---|---|
| Lead conversion model (Random Forest) | `model_evaluations/random_forest/rf_confusion_matrix.png` | Confusion matrix plus explainability chart in `model_evaluations/xai_insights/xai_rf_feature_weights.png` | Accuracy: 85.00%; class 0 precision/recall/F1: 0.932/0.818/0.871; class 1 precision/recall/F1: 0.751/0.903/0.820; macro F1: 0.846; weighted F1: 0.852; ROC-AUC: 0.933; PR-AUC: 0.890; CV F1: 0.806 ± 0.018; tuned threshold: 0.43 | `real_metrics.json`, `classification_report.json`, `best_model.json`, `rf_evaluation_report.md` | Not separately benchmarked in repo; low-latency tree inference after preprocessing |
| Investor persona clustering (K-Means) | `model_evaluations/kmeans/kmeans_cluster_projection.png` | `model_evaluations/kmeans/kmeans_centroids_heatmap.png` | Clusters: 4; silhouette: 0.395 | `kmeans_metrics.json`, `clustering_evaluation_report.md` | Not separately benchmarked in repo; cheap batch scoring over 7 survey features |
| Sentiment model (SBERT + classifier) | `model_evaluations/nlp_sentiment/nlp_evaluation_report.md` | Sentiment report used as the evaluation summary for the dashboard NLP path | Accuracy: 0.7160; macro F1: 0.6888; method: SBERT | `metrics.json`, `nlp_evaluation_report.md` | Not separately benchmarked in repo; transformer embedding cost is the main compute factor |
| Fund semantic search (SBERT retrieval) | `model_evaluations/tfidf_search/semantic_engine_report.md` | `model_evaluations/tfidf_search/cosine_similarity_decay.png` | Top-K retrieval: 5; practical query latency: under ~120 ms on the recorded test path | Semantic engine report and similarity decay chart | Retrieval is vector-based; memory depends on the cached embedding matrix size |
| LSTM NAV forecaster | `model_evaluations/lstm_forecaster/lstm_nav_predictions.png` | Forecast trajectory plot comparing predicted NAV path against the holdout trend | MSE: 0.0152; RMSE: 0.1233; R²: 0.89 | `neural_regression_report.md` | Not separately benchmarked in repo; heavier sequence inference than classical ML |

Suggested validation language for the poster:

- Lead scoring: "The Random Forest lead classifier achieved 85% accuracy with 0.933 ROC-AUC, 0.890 PR-AUC, class-1 recall of 0.903, and a tuned threshold of 0.43 for better lead capture."
- Investor clustering: "K-Means produced four investor personas with a silhouette score of 0.395, which is acceptable for behavior-driven segmentation."
- Sentiment: "The SBERT sentiment pipeline reached 0.716 accuracy and 0.689 macro F1, providing a reusable NLP baseline for financial text."
- Semantic search: "The retrieval engine returns relevant funds from a 14k-row corpus in roughly 120 ms, making it suitable for interactive fund lookup."
- LSTM forecasting: "The NAV forecaster achieved MSE 0.0152, RMSE 0.1233, and R² 0.89, showing good trend capture on the holdout sequence."

Detailed evaluation breakdown by model:

### Lead conversion model

- Holdout accuracy: 85.00%
- Positive-class precision/recall/F1: 0.751 / 0.903 / 0.820
- Negative-class precision/recall/F1: 0.932 / 0.818 / 0.871
- Macro average precision/recall/F1: 0.842 / 0.860 / 0.846
- Weighted average precision/recall/F1: 0.864 / 0.850 / 0.852
- ROC-AUC: 0.933
- Average precision / PR-AUC: 0.890
- Five-fold CV F1: 0.806 ± 0.018
- Decision threshold: 0.43
- Tuned parameters: `n_estimators=200`, `max_depth=12`, `class_weight=balanced`
- Artifacts: `real_metrics.json`, `classification_report.json`, `best_model.json`, `rf_confusion_matrix.png`, `xai_rf_feature_weights.png`

### Investor persona clustering

- Number of clusters: 4
- Silhouette score: 0.395
- Artifacts: `kmeans_metrics.json`, `kmeans_cluster_projection.png`, `kmeans_centroids_heatmap.png`, `clustering_evaluation_report.md`

### Sentiment model

- Accuracy: 0.7160
- Macro F1: 0.6888
- Method: SBERT-based sentiment classifier
- Artifacts: `metrics.json`, `nlp_evaluation_report.md`

### Fund semantic search

- Retrieval method: TF-IDF / cosine similarity engine backed by SBERT-style semantic lookup
- Top-K setting: 5
- Recorded query latency: under ~120 ms
- Evaluation focus: semantic matching rather than exact keyword lookup
- Artifacts: `semantic_engine_report.md`, `cosine_similarity_decay.png`

### LSTM NAV forecaster

- MSE: 0.0152
- RMSE: 0.1233
- R²: 0.89
- Evaluation focus: holdout NAV trend capture over sequence forecasting
- Artifacts: `neural_regression_report.md`, `lstm_nav_predictions.png`

### Resource utilization summary

- Random Forest lead scoring: low compute, CPU-friendly inference.
- K-Means clustering: very low compute for batch persona assignment.
- SBERT sentiment: moderate compute due to embedding generation.
- Semantic search: moderate memory use because embeddings are cached for retrieval.
- LSTM forecasting: highest model cost among the listed pipelines because it runs sequence inference over historical data.

If you want a stricter report layout, the same material can be reformatted into a poster-ready table with columns for screenshot, metric, insight, and takeaway.


## Big Data Phase (Data Engineering)

Purpose: build a scalable, auditable ingestion and preprocessing layer that converts multi-format raw sources into a reproducible columnar lake for analytics and ML.

What the code provides:
- Recursive ingestion of CSV/JSON/text (PySpark) with schema discovery and lineage columns.
- Data cleaning: normalization, deduplication, null/NA handling, numeric imputation and categorical fills.
- Data validation: null rate checks, approximate cardinality, IQR outlier flags.
- Materialization to Parquet with partitioning for efficient downstream queries.

Key files and locations:
- ETL orchestration: `src/lume_platform/spark/pipeline.py` and `src/master_pipeline.py` (entrypoint `run_default_etl()` / `run_full_system()`).
- Validation utilities: `src/lume_platform/spark/validation.py`.
- Output location: `artifacts/cleaned_parquet/` (partitioned Parquet files).

Run (local / Colab):
```bash
export PYTHONPATH=src
python -c "from lume_platform.spark.pipeline import run_default_etl; run_default_etl(sample=0.2)"
```

Produced artifacts:
- Partitioned Parquet dataset(s) under `artifacts/cleaned_parquet/`.
- Data quality summaries in `dataset_visualisations/audit_latest/`.

How reviewers should evaluate the Big Data phase:
- Verify ingestion reproducibility: re-run `run_default_etl()` with a sample and compare row counts with the logs.
- Inspect Parquet schema and partitioning strategy (read with `pyarrow`/`pyspark`).
- Check data-quality outputs (null rates, outlier flags) in the `dataset_visualisations/` reports.
- Confirm performance notes (approx runtime on dev machine) and scalability path (multi-node Spark / S3/HDFS mapping).

Reproducibility notes:
- The ETL is parameterized by `LUME_DATA_ROOT` and `LUME_ARTIFACTS` in `src/lume_platform/config.py`.
- For very large sources use the Colab notebook `notebooks/colab_spark_pipeline.ipynb` to run chunked ingestion.


## Docker deployment

Run full stack:

```bash
docker compose up --build
```

Main services in `docker-compose.yml`:

- `api` (FastAPI on port `8000`)
- `dashboard` (Streamlit on port `8501`)
- `mongodb` (state store)
- optional services (`redis`, `mlflow`)

## Google Colab workflow (free cloud)

1. Sync project to Google Drive.
2. Open `notebooks/colab_spark_pipeline.ipynb`.
3. Mount Drive and set dataset path.
4. Run ETL cells to generate Parquet artifacts.
5. Run training and export metrics/artifacts.

## Environment variables

| Variable | Purpose |
|---|---|
| `LUME_PROJECT_ROOT` | Repository root |
| `LUME_DATA_ROOT` | Dataset directory |
| `LUME_ARTIFACTS` | Artifacts directory |
| `LUME_TABLEAU_EMBED_URL` | Tableau embed in dashboard |
| `LUME_SPARK_SHUFFLE_PARTITIONS` | Spark tuning |

## BI export integration

- Export files for Tableau/Power BI:

```bash
python scripts/export_tableau_data.py
```

- Optionally use:

```bash
python scripts/bi_export.py
```

## Related docs

- `README_PRODUCTION.md` - deep production deployment notes
- `LUME_AI_FINAL_ARCHITECTURE_REPORT.md` - architecture rationale
- `Dataset_Documentation.md` - dataset provenance and source details
- `A1_Studio_Project_Poster_Content.md` - poster-ready technical summary

## Poster generation (A1)

- I added a printable A1 HTML poster: `poster_a1.html` (landscape A1 layout) and a refined poster content file: `A1_Studio_Project_Poster_Content.md`.
- To preview locally, open `poster_a1.html` in a browser (double-click or `open poster_a1.html`).

Export to PDF (A1 landscape) — examples:

Using Google Chrome (macOS):
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --headless --disable-gpu \
  --print-to-pdf=poster_a1.pdf --window-size=1684,1191 poster_a1.html
```

Using `wkhtmltopdf`:
```bash
wkhtmltopdf --page-size A1 --orientation Landscape poster_a1.html poster_a1.pdf
```

Notes on visuals:
- The poster references generated visual assets under `model_evaluations/` and `dataset_visualisations/` (confusion matrix, PCA, ROC). If images are missing, regenerate model artifacts by running the training pipeline which produces the evaluation plots:

```bash
export PYTHONPATH=src
python -m lume_platform.ml.training
```

- After training, place/verify the output images in `model_evaluations/<model>/` and refresh `poster_a1.html` in the browser.

If you prefer, I can generate the common plots (confusion matrix, ROC, PCA) from the evaluation JSONs and save them into `model_evaluations/` — tell me to proceed and I will add the generation script and run it.

## What I changed in the repo (workflow summary)

- Added `A1_Studio_Project_Poster_Content.md` — concise, poster-ready bullets mapped to repository evidence.
- Added `poster_a1.html` — printable A1 layout with embedded Mermaid architecture diagram and placeholders for screenshots/plots.
- Updated internal documentation in this `README.md` to include poster export instructions.

---


## License and data usage

This repo includes integrations with external datasets. Follow the source-specific terms documented in `Dataset_Documentation.md` before redistribution or commercial use.
