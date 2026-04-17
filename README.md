# Lume AI — Big Data + ML Platform (~31 GB composite dataset)

Production-style layout for an **AI-powered mutual fund lead** stack: **Apache Spark** for distributed ETL on the full tree, **sklearn / XGBoost / NLP** trained on Spark-sampled or tabular exports, **FastAPI** for JSON services, and **Streamlit** for operators. **No paid cloud is required** — primary runtime is **Google Colab** (or Kaggle) with the dataset on **Google Drive**; this repo runs locally the same code paths.

## Architecture

| Layer | Role |
|--------|------|
| `src/lume_platform/spark/` | PySpark ingestion (recursive CSV), cleaning, feature helpers, validation, `persist` + Parquet |
| `src/lume_platform/ml/` | Training: lead classifier (XGBoost if installed else RF), K-Means (+ DBSCAN), TF-IDF + logistic sentiment |
| `src/lume_platform/inference/` | Pickle bundles, `ModelRegistry`, Spark **pandas_udf** bridge for batch scoring |
| `api/main.py` | FastAPI: `/predict`, `/analytics`, `/insights`, `/health` |
| `streamlit_app.py` | Audit plots, live inference, BI iframe |
| `notebooks/colab_spark_pipeline.ipynb` | Colab: Drive mount, chunk reads, full ETL |

## Quick start (local)

```bash
cd "/path/to/BigData"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m lume_platform.ml.training   # models + exports + syncs model_evaluations/*.md to JSON
streamlit run streamlit_app.py
```

**Evaluation truth path:** after training, open `model_evaluations/MODELS_EVALUATION_REPORT.md` (master index) and `model_evaluations/random_forest/real_metrics.json`. Markdown narrative files are **auto-generated** so they cannot drift from metrics.

API:

```bash
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

Example `POST /predict`:

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

```json
{
  "task": "sentiment",
  "text": "Markets rally on strong GDP print"
}
```

## Google Colab (free)

1. Upload/sync the `BigData` folder to Drive (dataset already under `datasets/`).
2. Open `notebooks/colab_spark_pipeline.ipynb`.
3. Mount Drive, set `LUME_DATA_ROOT` to `/content/drive/MyDrive/.../datasets`.
4. Run cells: installs PySpark, runs ETL, writes `artifacts/cleaned_parquet`.

Chunk strategy: use `pandas.read_csv(..., chunksize=...)` for single huge files in audit cells; Spark uses `recursiveFileLookup` for millions of small CSVs (NAV, intraday).

## Full dataset handling

- **Structured**: `spark.read.csv` with `recursiveFileLookup=true` on `nifty500_intraday`, `nav_history`, etc.
- **Semi-structured**: `read.json` for news JSON; CSV clickstream as DataFrame.
- **Unstructured**: text via Spark `text` + regex / UDF preprocessing (extend `spark/ingestion.py`).

Intermediate and cleaned outputs: **Parquet** under `artifacts/cleaned_parquet/` (configurable via `LUME_ARTIFACTS`).

## BI (Tableau Public or Power BI Desktop)

1. Export flat files: `python scripts/export_tableau_data.py` (or use Parquet → CSV export from Spark).
2. **Tableau Public:** build and publish a workbook; set `LUME_TABLEAU_EMBED_URL` to the embed link. Streamlit’s **BI dashboard (embedded)** page loads it in an iframe.
3. **Power BI (free desktop):** import the same CSV/Parquet exports; publish to **Power BI** service if you need a web embed, then point `LUME_TABLEAU_EMBED_URL` at that embed URL (any HTTPS embed works in the iframe).

## Docker

```bash
docker build -t lume-api .
docker run -p 8000:8000 -e LUME_TABLEAU_EMBED_URL="https://public.tableau.com/..." lume-api
```

Include trained `artifacts/models/*.pkl` in the image or mount a volume.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `LUME_PROJECT_ROOT` | Repo root (auto-detected) |
| `LUME_DATA_ROOT` | `datasets/` directory |
| `LUME_ARTIFACTS` | `artifacts/` (Parquet + models) |
| `LUME_TABLEAU_EMBED_URL` | Tableau Public embed |
| `LUME_SPARK_SHUFFLE_PARTITIONS` | Tuning (default 200) |

## LSTM (optional)

Use existing `scripts/train_lstm_predictor.py` for NAV sequence models; Streamlit includes a lightweight demo page.

## License note

Dataset sources are documented in `Dataset_Documentation.md` (Kaggle / AMFI / NSE / etc.). Use according to each source’s terms.
