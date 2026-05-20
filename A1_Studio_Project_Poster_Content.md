# A1 Studio Project Poster — Lume AI (poster-ready content)

Poster size: A1 (594 × 841 mm) — Visual balance: 60% diagrams · 20% text · 20% white space

Use the attached template and paste each box's bullet content below into the corresponding placeholder.

---

## Header (top strip)
- Title: Lume AI — Big Data + ML Platform for Mutual-Fund Lead Intelligence
- Subtitle: Studio project — Artificial Intelligence + Big Data Analytics
- Team / Guide / Institute: (fill)
- One-line objective: Real-time lead scoring, investor personas, and fund discovery at scale
- QR: repo or live demo link

---

## 1. Introduction & Problem (left column)
- Context: Indian mutual-fund distribution — fragmented CRM + market + prospectus data
- Problem: Manual lead routing and fund discovery do not scale across channels
- Impact: Missed conversions, poor targeting, slow turnaround for distributors
- Project goal (measurable): Automate lead scoring + persona routing; improve conversion F1 to ≥0.80

---

## 2. Tri-Course Integration Architecture (center, main visual)
Place a labeled diagram showing: Sources → Spark ETL → Parquet lake → Feature store → Models → API → Dashboard/BI

Key components (legend):
- Big Data: `PySpark` ingestion, validation, Parquet materialization (`artifacts/cleaned_parquet/`)
- AI: Lead classifier (RF/XGBoost), Investor clustering (K-Means), NLP (TF-IDF / SBERT path), LSTM forecasting
- Deployment: `FastAPI` (endpoints), `Streamlit` dashboard, `docker-compose` orchestration, MongoDB persistence

Caption: `~31 GB multi-source data → Spark ETL → model training → model bundle → FastAPI + Streamlit` 

---

## 3. Methodology & Prototype Implementation (right of center)
- Data: structured (CSV, NAV), semi-structured (clickstream, JSON), unstructured (news, PDFs)
- ETL: `run_default_etl()` in `src/master_pipeline.py` → Parquet; validation & outlier flagging
- Features: numeric normalization (MinMax), categorical OHE, column transformer, serialized pipeline bundle
- Training: stratified 80/20 split, 5-fold CV, hyperparameter sweep (XGBoost/RF), threshold optimization (0.43)
- Serving: `api/main_enhanced.py` — `/predict`, `/batch_predict`, `/analytics`, `/health`

---

## 4. Results & Validation (right column — metric cards + visuals)
Use metric cards (one per metric) and 2 visuals (confusion matrix + ROC curve or PR curve).
- Lead classifier (from `model_evaluations/random_forest/real_metrics.json`):
  - Accuracy: 85.00%
  - Precision: 0.75
  - Recall: 0.90
  - F1: 0.82
  - ROC-AUC: 0.933
  - Decision threshold: 0.43 (precision@thr=0.7514, recall@thr=0.9028)
- Persona clustering: K-Means (k=4), Silhouette ≈ 0.40 — show 2D PCA cluster plot
- Display: confusion matrix image (`model_evaluations/random_forest/rf_confusion_matrix.png`) and cluster PCA plot

---

## 5. Prototype Demo (bottom-left)
- Short flow: `uvicorn api.main_enhanced:app` → `streamlit run streamlit_app_fintech.py`
- Key screenshots: Streamlit dashboard (lead routing), API docs (`/docs`), sample JSON request/response
- Docker: `docker-compose.yml` for multi-service local deployment (api, dashboard, mongodb, optional redis)

---

## 6. Conclusion & Future Scope (bottom-right)
- Achieved: End-to-end pipeline — ingestion → training → serving → dashboard; reproducible artifacts
- Business value: Faster distributor routing, higher conversion recall, explainable feature signals
- Next: deploy streaming Kafka ingestion, automated retraining, SHAP-based explainability, BERT/SBERT semantic retrieval

---

## Visual & Print Checklist (footer)
- Diagram types required: architecture diagram, ETL flowchart, confusion matrix, PCA cluster, one UI screenshot
- File sources for visuals: `model_evaluations/`, `dataset_visualisations/`, `frontend/` screenshots
- Layout: Title strip (top), central diagram box (largest), metrics & visuals to the right, method & demo below
- Text rules: bullets only; max 8–10 words per bullet; headings concise
- Print checks: fonts readable at 1.5–2m, export to PDF at 300 dpi, check color contrast

---

If you want, I will now:
1) Fill each poster box with final trimmed bullets sized for A1 (I can paste directly into your `A1_Studio_Project_Poster_Template.docx`), and
2) Generate the central architecture diagram as a PNG (Mermaid → PNG) and prepare the visuals folder for export.

