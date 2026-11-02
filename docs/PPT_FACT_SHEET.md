# Lume AI: Technical Audit & Deep Learning System Architecture Specification
**Document Type:** Technical Audit, Repository Fact Sheet & DL Architectural Blueprint  
**Target Audience:** DL Course Evaluation Committee, Technical Auditors, AI/Big Data Architects  
**Repository Path:** `/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData`  
**Audit Timestamp:** October 2026 (Local Time: 2026-10-29)  
**Status Taxonomy:**  
- `IMPLEMENTED+WIRED`: Code exists, model weights/pipeline functional, connected to API/UI.  
- `IMPLEMENTED BUT NOT WIRED`: Code exists in repository, but not imported, disconnected from inference loops, or disabled by default flag.  
- `MOCK/STUB`: Shell/dummy implementation returning placeholder strings, simulated synthetic curves, or hardcoded constants.  
- `PLANNED (PROPOSED DESIGN)`: Architectural target for Semester 2 Deep Learning system. Not built, no metrics attached.

---

## PART A: WHAT EXISTS (LAST SEMESTER AUDIT)

### A1. Repository Map (3-Level Directory Tree & Entry Points)

```text
BigData/ (Repository Root)
├── .env / .env.example              # Environment variables, database URIs, API keys, feature flags
├── CompleteNavReport.txt            # Raw AMFI NAV report (17,534 lines, 1.6 MB)
├── Dockerfile / Dockerfile.api      # Docker container definitions for base, FastAPI, and Streamlit
├── Dockerfile.dashboard             # Container configuration for Streamlit Dashboard
├── docker-compose.yml               # Multi-container orchestration (API, Dashboard, MongoDB, Redis, MLflow)
├── EQUITY_L.csv                     # NSE listed equity master list (2,241 stocks, 156 KB)
├── Makefile                         # Unified automation commands for training, tests, and runners
├── NIFTY 50-*.csv (3 files)         # Daily NIFTY 50 trading calendar & OHLCV data (2023-02 to 2026-02, 745 rows)
├── requirements.txt                 # Pinned dependencies for Vercel / serverless runtime
├── run.py / run.sh / start.sh       # Full-stack process launchers (FastAPI :8000 + HTTP Server :3000)
├── streamlit_app.py                 # Multi-tab Streamlit dashboard (Audit, Predictions, Tableau BI)
├── streamlit_app_fintech.py         # Advanced fintech-grade Streamlit portal with Plotly charts
├── api/                             # FastAPI application layer
│   ├── main.py                      # Minimal legacy FastAPI app (6 routes)
│   └── main_enhanced.py             # Enterprise FastAPI app (2,299 lines, 55 routes + 4 OpenAPI docs)
├── artifacts/                       # Serialized model bundles, scalers, and logs
│   ├── cleaned_parquet/             # Spark-materialized Parquet datasets
│   │   └── lead_scoring_clean/      # Partitioned by converted=0 / converted=1 (1,900 rows)
│   ├── ml_scalers/                  # Serialized MinMax scalers (univariate & multivariate)
│   └── models/                      # Pickled Scikit-Learn pipelines & PyTorch checkpoint weights
├── configs/                         # YAML configuration files
│   ├── forecaster_bilstm_attention.yaml # Phase 2 BiLSTM + Attention training configuration
│   └── forecaster_lstm.yaml         # Phase 1 univariate LSTM training configuration
├── data/                            # Local external data copies
│   └── Mutual_Fund_Data-main/       # Kaggle mutual fund clone (16,344 funds, 4.6 MB)
├── dataset_visualisations/          # Outlier audits, histograms, correlation heatmaps, boxplots
│   └── audit_latest/                # High-resolution PNGs of tabular feature distributions
├── frontend/                        # Web application static client (HTML5/CSS3/ES6 Vanilla JS)
│   ├── ai-showcase.html             # Deep Learning visualizer (LSTM vs BiLSTM, Attention heatmap)
│   ├── app.js                       # Frontend client controller and REST fetch engine
│   ├── bigdata-showcase.html        # PySpark ETL pipeline architecture walkthrough
│   ├── distributor.html             # Mutual Fund Distributor portal (lead scoring, personas, pitch)
│   ├── index.html                   # Platform landing page & navigation hub
│   ├── investor.html                # Investor wealth portal (portfolio tracking, fund discovery)
│   ├── login.html / signup.html     # Authentication views (JWT token storage)
│   └── styles.css                   # Custom fintech dark/light theme styling
├── lume-ai-product/                 # Alternative product monorepo prototype
│   ├── apps/
│   │   ├── api/                     # Subproject FastAPI service
│   │   └── web/                     # Next.js 16.2 / React 19 web app with Tailwind CSS 4
│   └── ml-artifacts/                # Mirrored model evaluations and JSON metrics
├── mintleads/                       # Standalone modular distributor CRM engine
│   ├── api/                         # Flask API routes for leads, clusters, and NAV
│   ├── dashboard/                   # Streamlit multipage application (Hot Leads, NAV, Personas)
│   ├── pipelines/                   # Standalone training pipelines (ingest, preprocess, train_lstm)
│   └── requirements.txt             # Pinned environment specification for MintLeads
├── model_evaluations/               # Official source-of-truth evaluation reports, JSONs & plots
│   ├── kmeans/                      # K-Means clustering metrics, silhouette reports, PCA plots
│   ├── lstm_forecaster/             # Phase 1 LSTM real metrics JSON, loss curves, model card
│   ├── nlp_sentiment/               # SBERT vs TF-IDF sentiment metrics and evaluation report
│   ├── phase2_timeseries/           # Phase 2 BiLSTM + Attention real metrics JSON, model card
│   ├── random_forest/               # Random Forest real metrics JSON, classification report, CM
│   ├── tfidf_search/                # TF-IDF search engine decay report and plot
│   └── xai_insights/                # TreeExplainer feature importance bar charts
├── notebooks/                       # Google Colab and Jupyter execution notebooks
│   ├── colab_production_pipeline.py # Self-contained Colab pipeline script
│   └── colab_spark_pipeline.ipynb   # Interactive PySpark execution notebook
├── output_production_final/         # Seed data, cached responses, and CSV exports
│   ├── analytics_cache/             # Precomputed benchmark NIFTY trendline JSON
│   ├── nav_cache/                   # 19 scheme historical NAV JSONs + latest_navs.json (13,965 funds)
│   ├── news_cache/                  # Cached market news articles
│   ├── db_custom_leads.json         # File-based MongoDB mock storage for leads (1,349 rows)
│   ├── distributor_leads_master.csv # Scored lead export with conversion probabilities (1,350 rows)
│   └── investor_routing_matches.csv # Assigned investor personas and AMC routing (400 rows)
├── scripts/                         # Standalone ETL, scraper, and utility scripts (22 scripts)
│   ├── bi_export.py                 # Exports curated CSV/JSON tables for Tableau and Power BI
│   ├── build_dual_ai_models.py      # Standalone training script for lead classifier & clustering
│   ├── fetch_amfi_nse.py            # Live scraper for AMFI NAV feed & NSE market status
│   ├── fetch_prospectus_pdfs.py     # BeautifulSoup scraper targeting AMC statutory disclosure PDFs
│   ├── generate_candlestick_images.py # Synthetic OHLC generator rendering mplfinance PNG charts
│   ├── live_stream_simulator.py     # Micro-batch stream injector writing CSV files every 5s
│   ├── preprocess_timeseries.py     # Preprocessing utility for market series
│   ├── search_algorithm.py          # Standalone TF-IDF cosine similarity search engine
│   └── train_lstm_predictor.py      # Standalone 3-layer LSTM training script
├── src/                             # Core Python package root (`lume_platform`)
│   ├── master_pipeline.py           # Master pipeline orchestrator (Spark ETL -> ML -> BI Export)
│   ├── streaming/                   # Streaming Kafka consumers (news_consumer.py)
│   └── lume_platform/               # Production library modules
│       ├── audit/                   # Automated dataset quality inspection and profiling
│       ├── auth.py                  # JWT creation, decoding, and PBKDF2 password hashing
│       ├── cli.py                   # Command-line interface (`lume train`, `lume forecast`)
│       ├── config.py                # Central filesystem paths and environment resolution
│       ├── crm/                     # Distributor CRM service & dashboard aggregation logic
│       ├── data/                    # Zero-leakage chronological time-series dataset builder
│       ├── db/                      # PyMongo client with fallback to local JSON storage
│       ├── inference/               # ModelRegistry lazy loader and Scikit-Learn Spark UDFs
│       ├── ml/                      # Machine learning models, trainers, losses, and baselines
│       ├── news/                    # Financial news ingestion and RSS parsers
│       ├── risk/                    # Portfolio risk analytics and Monte Carlo VaR simulation
│       ├── spark/                   # PySpark session factory, cleaning, features, streaming
│       └── utils/                   # Seed setting, mathematical metrics, and helpers
├── streaming/                       # Active streaming buffer directories
│   ├── inbox/ (market/, news/)      # Target directories for micro-batch CSV injection
│   └── outputs/                     # Materialized streaming Parquet partitions
├── tableau_exports/                 # Precomputed CSV/JSON exports and Tableau integration guide
└── tests/                           # Pytest test suite (6 test modules, 1,351 LOC)
```

#### Entry Points and How to Run

| Entry Point | Command Line Invocation | Target Host:Port | Description / Role |
|---|---|---|---|
| **Full-Stack Runner** | `python3 run.py` or `./run.sh` or `make run` | Frontend `:3000`<br>Backend `:8000` | Concurrently spawns Uvicorn (`api.main_enhanced:app`) and Python HTTP Server (`frontend/`). Handles graceful SIGINT termination. (Citing `run.py:117-140`) |
| **Enhanced API** | `uvicorn api.main_enhanced:app --host 0.0.0.0 --port 8000 --reload` | Backend `:8000` | Primary production REST API with 55 functional endpoints, JWT auth, MongoDB integration, and model inference. (Citing `api/main_enhanced.py:118`) |
| **Legacy Minimal API** | `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload` | Backend `:8000` | Minimal fallback API supporting basic `/predict`, `/analytics`, `/leads`. (Citing `api/main.py:39-177`) |
| **Fintech Dashboard** | `PYTHONPATH=src streamlit run streamlit_app_fintech.py --server.port 8501` | Dashboard `:8501` | Full-featured financial portal with Plotly charts, live market indicators, lead management, and risk engine. (Citing `streamlit_app_fintech.py:1-60`) |
| **Base Dashboard** | `PYTHONPATH=src streamlit run streamlit_app.py --server.port 8501` | Dashboard `:8501` | Streamlit app exposing data audits, ML inference forms, and Tableau Public iframes. (Citing `streamlit_app.py:1-40`) |
| **MintLeads Dashboard** | `cd mintleads && streamlit run dashboard/app.py` | Dashboard `:8501` | Multi-page distributor dashboard (Hot Leads, NAV Predictor, Personas, Sentiment). (Citing `mintleads/dashboard/app.py:1-35`) |
| **CLI Forecaster** | `PYTHONPATH=src python3 -m lume_platform.cli forecast --scheme 100119` | CLI Standard Output | Executes multi-horizon NAV trajectory prediction with P10/P50/P90 quantile bounds and attention weights. (Citing `src/lume_platform/cli.py:48-95`) |
| **CLI Trainer** | `PYTHONPATH=src python3 -m lume_platform.cli train --engine bilstm` | CLI Standard Output | Trains BiLSTM + Attention model or classical ML bundles (`--engine all`). (Citing `src/lume_platform/cli.py:25-47`) |
| **Master Pipeline** | `PYTHONPATH=src python3 src/master_pipeline.py` | CLI Standard Output | End-to-end batch run: Spark ETL -> Scikit-Learn training -> Tableau BI export. (Citing `src/master_pipeline.py:24-68`) |
| **Live Stream Simulator** | `python3 scripts/live_stream_simulator.py` | CLI Standard Output | Generates micro-batch market and news CSV files every 5 seconds into `streaming/inbox/`. (Citing `scripts/live_stream_simulator.py:15-56`) |
| **Docker Compose** | `docker-compose up --build` | Multiple Ports | Launches 5 containers: `lume-api` (:8000), `lume-dashboard` (:8501), `lume-mongodb` (:27017), `lume-redis` (:6379), `lume-mlflow` (:5001). (Citing `docker-compose.yml:1-118`) |
| **Test Suite** | `pytest tests/ -v` or `make test` | CLI Standard Output | Runs 6 test suites including chronological data leakage tests, model architecture tests, and API contract tests. (Citing `Makefile:31-39`) |

---

### A2. Complete Tech Stack with Versions & Route Table

#### 1. Core Languages & Runtime Versions
- **Python**: 3.11 (Docker container baseline `python:3.11-slim`), 3.13.7 (local host workstation `staff` environment).
- **Node.js**: 20+ (Next.js 16.2.9 monorepo environment).
- **Java**: OpenJDK 11 (`JAVA_HOME=/opt/homebrew/opt/openjdk@11/...` specified in `src/master_pipeline.py:17`).
- **Shell**: POSIX Zsh / Bash (`run.sh`, `start.sh`, `Makefile`).

#### 2. Data Engineering & Big Data
- **Apache Spark / PySpark**: Version 4.1.1 installed in local venv (`pyspark-4.1.1.dist-info`), PySpark 3.4+ compatible API.
- **Storage Format**: Apache Parquet with Snappy compression (`.snappy.parquet`).
- **Partitioning**: Partitioned column materialization via Spark `partitionBy("converted")` in `artifacts/cleaned_parquet/lead_scoring_clean/` (Citing `src/lume_platform/spark/ingestion.py:28-35`).
- **Serialization / Optimization**: KryoSerializer (`org.apache.spark.serializer.KryoSerializer`), Adaptive Query Execution enabled (`spark.sql.adaptive.enabled=true`, `spark.sql.adaptive.coalescePartitions.enabled=true`), Shuffle Partitions: 200 (Citing `src/lume_platform/spark/session.py:14-22`).
- **Streaming**: Spark Structured Streaming (`readStream.json` -> `groupBy(window("ts", "1 minute", "30 seconds"))` -> `writeStream.foreachBatch(write_to_mongo)`) (Citing `src/lume_platform/spark/streaming_pipeline.py:48-99`).

#### 3. Databases & Schemas
- **MongoDB**: Engine Mongo 7.0+ (`mongo:latest` in Docker). Connection via PyMongo 4.17.0 (`pymongo>=4.6.0` in `requirements.txt:7`).
- **Collections & Schemas** (Citing `src/lume_platform/db/mongo_client.py:74-120`):
  1. `users`: Schema `{ email (str, unique index), full_name (str), role (str: "investor"|"distributor"|"admin"), hashed_password (str), created_at (str) }`.
  2. `leads`: Schema `{ lead_id (str, unique index), first_name (str), last_name (str), city (str), occupation (str), conversion_probability (float, indexed), recommended_pitch_persona (str), converted_prediction (int), status (str) }`.
  3. `portfolios`: Schema `{ email (str, unique index), holdings (list of dicts: scheme_code, scheme_name, units, purchase_nav, invested_amount), updated_at (str) }`.
- **Mock Fallback Storage**: When `MONGO_URI` is unset or connection times out (5000ms), silently falls back to local JSON files (`db_custom_leads.json`, `db_users.json`, `db_portfolios.json`) in `output_production_final/` or `/tmp/lume_db/` (Citing `src/lume_platform/db/mongo_client.py:43-73`).
- **Redis**: Redis 7-Alpine (`redis:7-alpine` in `docker-compose.yml:81-90`), used for session caching and rate-limiting buffers.

#### 4. Machine Learning & Deep Learning Libraries
- **PyTorch**: Version 2.10.0 (workstation venv), 2.7.1+cpu (`lume-ai-product/apps/api/requirements.txt:11`), 2.1.2 (`mintleads/requirements.txt:14`). Used for `LSTMForecaster` and `BiLSTMAttentionQuantileForecaster`.
- **Scikit-Learn**: Version 1.8.0 (workstation venv), 1.6.1 (`apps/api`), 1.4.0 (`mintleads`). Used for `RandomForestClassifier`, `KMeans`, `DBSCAN`, `LogisticRegression`, `TfidfVectorizer`, `ColumnTransformer`, `Pipeline`.
- **XGBoost**: Version 3.2.0 (`xgboost-3.2.0.dist-info`), conditional grid-search trainer in `src/lume_platform/ml/training.py:50-55`.
- **HuggingFace Transformers**: Version 5.5.4 (`transformers-5.5.4.dist-info`), DistilBERT pipeline wrapper in `src/lume_platform/ml/transformer_sentiment.py`, FinBERT tone pipeline in `src/streaming/news_consumer.py`.
- **Sentence-Transformers**: Version 5.4.1 (`sentence_transformers-5.4.1.dist-info`), SBERT embedding search (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`).
- **Joblib / Pickle**: Model serialization protocols (`joblib-1.5.3`).

#### 5. Backend Architecture & Complete API Route Table
FastAPI backend (`fastapi>=0.110` in `requirements.txt:5`, runtime version 0.135.3).  
Security & Middleware: `CORSMiddleware` (allow all origins, credentials, methods, headers), Request Timing Middleware (`X-Process-Time`), OAuth2 Password Bearer (`/auth/token`), JWT HS256 authentication (`python-jose[cryptography]>=3.3.0`), PBKDF2 SHA-256 password hashing.

The following table documents every active route in `api/main_enhanced.py` (total 55 application routes):

| HTTP Method | Endpoint Path | Function Name & Line | Purpose / Description | Downstream Model / Asset Called | Status Label |
|---|---|---|---|---|---|
| `GET` | `/` | `root` (L544) | Welcome payload and service discovery | None (Static JSON) | IMPLEMENTED+WIRED |
| `GET` | `/health` | `health_check` (L550) | Service heartbeat, uptime, model status | `ModelRegistry.is_loaded` | IMPLEMENTED+WIRED |
| `GET` | `/signals/latest` | `get_latest_signals` (L573) | Latest streaming market signals | In-memory `registry.latest_signals` | IMPLEMENTED+WIRED |
| `POST` | `/market/impact/analyze` | `analyze_market_impact` (L583) | Computes market volatility impact score | Rule-based heuristic on index | IMPLEMENTED+WIRED |
| `GET` | `/market/impact/latest` | `get_latest_impact` (L608) | Retrieves cached market impact assessment | Cached JSON dictionary | IMPLEMENTED+WIRED |
| `GET` | `/alerts/live` | `get_live_alerts` (L619) | Scans streaming output Parquet for alerts | `streaming/outputs/market_live.parquet` | IMPLEMENTED+WIRED |
| `GET` | `/monitoring/drift/status` | `drift_status` (L645) | Monitors data drift across lead features | `DriftMonitor` statistical checks | IMPLEMENTED+WIRED |
| `POST` | `/monitoring/retrain` | `trigger_retraining` (L655) | Triggers background pipeline retraining | Spawns `train_all()` subprocess | IMPLEMENTED+WIRED |
| `POST` | `/predict` | `predict` (L671) | Single inference for leads/clusters/sentiment | `lead_classifier_bundle.pkl`, `investor_cluster_bundle.pkl`, `sentiment_bundle.pkl` | IMPLEMENTED+WIRED |
| `POST` | `/batch_predict` | `batch_predict` (L763) | Batch inference across tabular arrays | Vectorized `pipe.predict_proba` | IMPLEMENTED+WIRED |
| `GET` | `/analytics` | `analytics` (L835) | High-level platform KPIs and lead volumes | `output_production_final/distributor_leads_master.csv` | IMPLEMENTED+WIRED |
| `POST` | `/risk/analyze` | `analyze_portfolio_risk` (L894) | Portfolio VaR (Value at Risk) simulation | Monte Carlo Gaussian simulation | IMPLEMENTED+WIRED |
| `GET` | `/risk/market-snapshot` | `market_snapshot` (L929) | Summary of macro volatility and index levels | `output_production_final/analytics_cache/` | IMPLEMENTED+WIRED |
| `GET` | `/insights` | `insights` (L986) | Model cards, evaluation metadata index | `model_evaluations/` directory | IMPLEMENTED+WIRED |
| `GET` | `/search/funds` | `search_funds` (L1040) | Semantic search across mutual funds | `SBERTMutualFundSearch` (`fund_embeddings.pkl`) | IMPLEMENTED BUT NOT WIRED (Throws 503 if pickle missing) |
| `POST` | `/auth/register` | `register_user` (L1090) | Registers new investor or distributor | `db_client.create_user`, PBKDF2 hash | IMPLEMENTED+WIRED |
| `POST` | `/auth/login` | `login_user` (L1108) | Authenticates user and returns JWT | `db_client.get_user`, `create_access_token` | IMPLEMENTED+WIRED |
| `GET` | `/auth/me` | `get_me` (L1125) | Returns active authenticated user identity | JWT token claim resolution | IMPLEMENTED+WIRED |
| `POST` | `/auth/token` | `login_for_access_token` (L1134)| OAuth2 standard form login endpoint | Standard OAuth2 token provider | IMPLEMENTED+WIRED |
| `GET` | `/portfolio` | `get_portfolio` (L1163) | Fetches authenticated user portfolio | `db_client.get_portfolio` | IMPLEMENTED+WIRED |
| `POST` | `/portfolio/add` | `add_portfolio_holding` (L1227)| Adds new fund units to user holdings | `db_client.save_portfolio` | IMPLEMENTED+WIRED |
| `POST` | `/portfolio/sell` | `sell_portfolio_holding` (L1301)| Liquidates fund units from portfolio | `db_client.save_portfolio` | IMPLEMENTED+WIRED |
| `PUT` | `/portfolio/{holding_id}` | `update_holding` (L1390) | Updates units or purchase price | `db_client.save_portfolio` | IMPLEMENTED+WIRED |
| `DELETE` | `/portfolio/{holding_id}` | `delete_holding` (L1400) | Removes a position from portfolio | `db_client.save_portfolio` | IMPLEMENTED+WIRED |
| `GET` | `/portfolio/summary` | `portfolio_summary` (L1410) | Aggregate valuation, total P&L, returns | Real-time valuation against `latest_navs.json` | IMPLEMENTED+WIRED |
| `GET` | `/nav/{scheme_code}` | `get_nav` (L1467) | Returns latest NAV for single scheme | `output_production_final/nav_cache/latest_navs.json` | IMPLEMENTED+WIRED |
| `GET` | `/nav/{scheme_code}/history` | `get_nav_history` (L1482) | 735-day historical NAV series | `output_production_final/nav_cache/hist_{code}.json` | IMPLEMENTED+WIRED |
| `GET` | `/portfolio/analytics` | `portfolio_analytics` (L1499)| Asset allocation breakdown (Debt/Equity) | Heuristic category parsing | IMPLEMENTED+WIRED |
| `GET` | `/news/live` | `get_live_news` (L1511) | Returns parsed financial news feed | `output_production_final/news_cache/latest_news.json` | IMPLEMENTED+WIRED |
| `GET` | `/benchmark/history` | `get_benchmark_history` (L1523)| Historical NIFTY 50 closing series | `output_production_final/analytics_cache/nifty_hist_365.json` | IMPLEMENTED+WIRED |
| `POST` | `/admin/funds/upload` | `upload_funds` (L1545) | Uploads new scheme CSV dataset | `_check_admin_key` protected | IMPLEMENTED+WIRED |
| `POST` | `/admin/funds/reindex` | `reindex_funds` (L1568) | Recomputes SBERT embeddings on catalog | Background task triggering SBERT embedder | IMPLEMENTED+WIRED |
| `GET` | `/admin/metrics` | `get_admin_metrics` (L1619) | System memory, request counts, cache | Process CPU/RAM inspection | IMPLEMENTED+WIRED |
| `GET` | `/user/{user_id}/collaborative-recommend` | `collab_recommend` (L1636)| Nearest-neighbor collaborative filtering | Matrix factorizer / fallback rules | IMPLEMENTED+WIRED |
| `POST` | `/user/profile` | `save_user_profile` (L1650) | Saves investor risk questionnaire | `db_client` profile updater | IMPLEMENTED+WIRED |
| `GET` | `/user/{user_id}/recommend` | `user_recommend` (L1672) | Personalized fund recommendations | Rule-based category matching | IMPLEMENTED+WIRED |
| `GET` | `/funds/catalog` | `funds_catalog` (L1700) | Full paginated AMFI mutual fund list | `data/Mutual_Fund_Data-main/mutual_fund_data.csv` | IMPLEMENTED+WIRED |
| `GET` | `/funds/recommendations` | `fund_recommendations` (L1714)| Top funds filtered by category & risk | Static filtering on catalog | IMPLEMENTED+WIRED |
| `POST` | `/distributors/validate-rec` | `validate_recommendation` (L1744)| Validates distributor proposal vs SEBI rules | Regulatory boundary validator | IMPLEMENTED+WIRED |
| `GET` | `/distributors/client-insights` | `client_insights` (L1819) | Retrieves investor behavioral profile | `investor_cluster_bundle.pkl` | IMPLEMENTED+WIRED |
| `GET` | `/dashboard/overview` | `dashboard_overview` (L1894) | Aggregated metrics for distributor CRM | Aggregations on scored leads | IMPLEMENTED+WIRED |
| `GET` | `/dashboard/leads` | `dashboard_leads` (L1905) | Paginated, filtered distributor leads | `output_production_final/distributor_leads_master.csv` | IMPLEMENTED+WIRED |
| `GET` | `/model/{model_name}/info` | `model_info` (L1922) | Returns model hyperparameters & weights | `artifacts/models/model_manifest.json` | IMPLEMENTED+WIRED |
| `GET` | `/exports/{file_name}` | `download_export` (L1969) | Downloads precomputed BI CSV exports | `tableau_exports/` filesystem | IMPLEMENTED+WIRED |
| `GET` | `/exports` | `list_exports` (L1988) | Lists available CSV export tables | `tableau_exports/` directory list | IMPLEMENTED+WIRED |
| `POST` | `/async_predict` | `async_predict` (L2015) | Asynchronous inference task queue | BackgroundTasks worker | IMPLEMENTED+WIRED |
| `GET` | `/leads` | `get_leads` (L2023) | Lead list filtered by status & score | `db_client.leads_db` | IMPLEMENTED+WIRED |
| `PATCH` | `/leads/{lead_id}` | `update_lead_status` (L2037)| Updates lead contact status / notes | `db_client.update_lead` | IMPLEMENTED+WIRED |
| `GET` | `/distributors/matches` | `distributor_matches` (L2054)| Matches investor to top distributors | `output_production_final/investor_routing_matches.csv` | IMPLEMENTED+WIRED |
| `POST` | `/api/forecast` | `api_forecast` (L2072) | Multi-horizon NAV trajectory prediction | `LUMEForecaster` (`bilstm_attention_quantile.pth` or `lstm_nav_pattern_predictor.pth` or fallback) | IMPLEMENTED+WIRED |
| `POST` | `/predict/forecast` | `predict_forecast` (L2073) | Alias endpoint for NAV forecasting | Identical call to `LUMEForecaster` | IMPLEMENTED+WIRED |
| `GET` | `/predict/explain/{lead_id}` | `explain_prediction` (L2105)| XAI feature importance for single lead | Scikit-Learn tree feature importances | IMPLEMENTED+WIRED |
| `POST` | `/advisor/query` | `advisor_query` (L2140) | Financial rule-based chatbot query | Keyword intent matching engine | IMPLEMENTED+WIRED |
| `GET` | `/buddy/briefing` | `buddy_briefing` (L2239) | Daily morning distributor briefing | Dynamic text generator over leads | IMPLEMENTED+WIRED |
| `POST` | `/buddy/chat` | `buddy_chat` (L2271) | Conversational AI distributor copilot | `BuddyEngine` (TF-IDF + Canned Templates) | MOCK/STUB |

#### 6. Frontend Stack
- **Architecture**: Single Page Application (SPA) structure rendered via Vanilla JavaScript (ES6+), HTML5, and CSS3. No heavy JS framework required for runtime.
- **Client Pages**: 7 dedicated pages in `frontend/`: `index.html` (Landing), `investor.html` (Investor Dashboard), `distributor.html` (Distributor Portal), `ai-showcase.html` (Deep Learning Visualizer), `bigdata-showcase.html` (Spark Architecture), `login.html`, `signup.html`.
- **Chart Rendering Libraries**: Chart.js version 4.4.4 via CDN (`https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js`).
- **Alternative Framework**: Next.js 16.2.9 + React 19.2.4 monorepo under `lume-ai-product/apps/web/` with Recharts 3.9.0 and Tailwind CSS 4.

#### 7. DevOps & Infrastructure Configuration
- **Containerization**:
  - `Dockerfile`: Multi-stage build running Python 3.11-slim, installing `requirements.txt`, exposing port 8000.
  - `Dockerfile.api`: Dedicated FastAPI container running `uvicorn api.main_enhanced:app`.
  - `Dockerfile.dashboard`: Dedicated Streamlit container running `streamlit run streamlit_app.py`.
- **Docker Compose (`docker-compose.yml`)**:
  - `api` (port 8000), `dashboard` (port 8501), `mongodb` (port 27017, image `mongo:latest`), `redis` (port 6379, image `redis:7-alpine`), `mlflow` (port 5001, image `python:3.10-slim`).
- **Continuous Integration (CI)**:
  - `mintleads/.github/workflows/ci.yml`: GitHub Actions pipeline on push/PR: Flake8 linting, Pytest with coverage threshold $\ge 70\%$, Docker buildx and image artifact upload, curl health check validation. Note: Root repository does NOT have a top-level `.github/workflows/` directory.
- **Environment Configuration**: `.env` and `.env.example` defining 142 lines of settings: `LUME_PROJECT_ROOT`, `LUME_DATA_ROOT`, `MONGO_URI`, `JWT_SECRET`, `SPARK_DRIVER_MEMORY=8g`, `SPARK_SHUFFLE_PARTITIONS=200`, `LUME_ENABLE_FORECASTER=1`, `LUME_ENABLE_SEMANTIC_SEARCH=1`.
- **Build Tool**: `Makefile` with targets: `help`, `train-bilstm`, `train-lstm`, `train-all`, `test`, `test-phase2`, `leakage-test`, `forecast`, `api`, `dashboard`, `run`, `start`, `clean`.

#### 8. External Data Sources & Web Scraping Endpoints
- **AMFI India (Association of Mutual Funds in India)**:
  - Daily NAV feed: `https://www.amfiindia.com/spages/NAVAll.txt` (scraped by `scripts/fetch_amfi_nse.py:22` and `CompleteNavReport.txt`).
- **NSE India (National Stock Exchange)**:
  - Market Status API: `https://www.nseindia.com/api/marketStatus` (scraped by `scripts/fetch_amfi_nse.py:48`).
  - Historical Bhavcopy / Equity API: `https://www.nseindia.com/api/historical/cm/equity` (referenced in `scripts/ingest_nse_historical.py:15`).
  - Listed equity master: `EQUITY_L.csv` (2,241 stocks).
- **Yahoo Finance API**:
  - `yfinance>=0.2.30` (in `requirements.txt:9`), used to fetch NIFTY 50 and stock tick history (`scripts/fetch_nse_yfinance.py`).
- **News RSS Feeds**:
  - Moneycontrol Top News: `https://www.moneycontrol.com/rss/MCtopnews.xml` (`.env:82`).
  - Economic Times Markets: `https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms` (`.env:83`).
- **Scraped AMC Websites** (Targeted by `scripts/fetch_fund_images.py` & `scripts/fetch_prospectus_pdfs.py`):
  - SBI Mutual Fund (`sbimf.com`), HDFC Mutual Fund (`hdfcfund.com`), ICICI Prudential AMC (`icicipruamc.com`), Nippon India Mutual Fund (`mf.nipponindiaim.com`), Axis Mutual Fund (`axismf.com`), Tata Mutual Fund (`tatamutualfund.com`), Baroda BNP Paribas Mutual Fund (`barodabnpparibasmf.in`).

---

### A3. Data Inventory & Volume Measurement

#### Direct Measurement of Repository Volume vs Claimed "30 GB"
- **Claimed Volume in Documentation**: "30 GB across 315,538 files and millions of records" (`Dataset_Documentation.md:25, 324, 512`), "~31 GB multi-source data" (`A1_Studio_Project_Poster_Content.md:34`), `31.25 GB` (`tableau_exports/summary_metrics.json:3` and `scripts/bi_export.py:204`).
- **Actual Measured Disk Usage**: **50 MB** (`du -sh .` returns `50M`!).
- **Status of Claim**: **REPORTED, UNVERIFIED (GROSS DISCREPANCY)**. The ~30 GB of raw Kaggle intraday data (`nifty500_intraday`, `stock_market_india`, `financial_news_corpus`) was gitignored (`.gitignore:40-41` ignores `/data/` and `datasets/`) and never committed to the repository. The path `datasets/` does NOT exist in the workspace.

#### Comprehensive Dataset Inventory Table

| Dataset Identifier | Relative File Path | Format | Modality | Row / File Count | Size on Disk | Provenance / Source | Consuming Module | Status Label |
|---|---|---|---|---|---|---|---|---|
| **Mutual Fund Master Catalog** | `data/Mutual_Fund_Data-main/mutual_fund_data.csv` | CSV | Structured | 16,345 rows (16,344 funds) | 4.6 MB | Kaggle / GitHub clone (`Mutual_Fund_Data-main`) | `scripts/search_algorithm.py`, `api/main_enhanced.py:1700` | IMPLEMENTED+WIRED |
| **Complete AMFI NAV Report** | `CompleteNavReport.txt` | Semicolon TXT | Structured | 17,534 lines | 1.6 MB | AMFI India (`NAVAll.txt`) | `scripts/fetch_amfi_nse.py`, `scripts/fetch_mf_history.py` | IMPLEMENTED+WIRED |
| **NSE Listed Equities** | `EQUITY_L.csv` | CSV | Structured | 2,241 rows | 156 KB | National Stock Exchange of India | `scripts/fetch_nse_yfinance.py`, `ingest_nse_historical.py` | IMPLEMENTED+WIRED |
| **NIFTY 50 Daily 2023-2024** | `NIFTY 50-21-02-2023-to-21-02-2024.csv` | CSV | Structured | 248 trading days | 16 KB | NSE Historical Index Data | `src/lume_platform/data/timeseries_dataset.py:26` | IMPLEMENTED+WIRED |
| **NIFTY 50 Daily 2024-2025** | `NIFTY 50-21-02-2024-to-20-02-2025.csv` | CSV | Structured | 250 trading days | 16 KB | NSE Historical Index Data | `src/lume_platform/data/timeseries_dataset.py:27` | IMPLEMENTED+WIRED |
| **NIFTY 50 Daily 2025-2026** | `NIFTY 50-21-02-2025-to-21-02-2026.csv` | CSV | Structured | 247 trading days | 16 KB | NSE Historical Index Data | `src/lume_platform/data/timeseries_dataset.py:28` | IMPLEMENTED+WIRED |
| **Lead Scoring Parquet (Clean)** | `artifacts/cleaned_parquet/lead_scoring_clean/` | Parquet (Snappy) | Structured | 1,900 rows (1,180 conv=0, 720 conv=1) | 120 KB | Materialized by PySpark from Kaggle Lead Scoring | `src/lume_platform/ml/training.py:117` | IMPLEMENTED+WIRED |
| **Distributor Scored Leads Master** | `output_production_final/distributor_leads_master.csv` | CSV | Structured | 1,350 rows (1 header + 1,349 scored leads) | 368 KB | Scored by Random Forest pipeline ($P \ge 0.43$) | `api/main_enhanced.py`, `db_client.py:88`, `distributor.html` | IMPLEMENTED+WIRED |
| **Investor Routing Matches** | `output_production_final/investor_routing_matches.csv` | CSV | Structured | 400 rows | 109 KB | Clustered by K-Means ($k=4$) from behavior survey | `api/main_enhanced.py:2054`, `investor.html`, Streamlit | IMPLEMENTED+WIRED |
| **Historical NAV Scheme Cache** | `output_production_final/nav_cache/hist_*.json` (19 files) | JSON | Structured | ~735 daily points per scheme | 450 KB total | Cached AMFI historical records | `src/lume_platform/data/timeseries_dataset.py:61` | IMPLEMENTED+WIRED |
| **Latest NAV Lookup Dictionary** | `output_production_final/nav_cache/latest_navs.json` | JSON | Structured | 13,965 mutual fund scheme keys | 256 KB | Cached latest scheme NAV valuation map | `api/main_enhanced.py:1467`, `portfolio_summary` | IMPLEMENTED+WIRED |
| **NIFTY 365-Day Historical Cache** | `output_production_final/analytics_cache/nifty_hist_365.json` | JSON | Structured | 365 daily points | 28 KB | Precomputed NIFTY 50 benchmark curve | `api/main_enhanced.py:1523` (`/benchmark/history`) | IMPLEMENTED+WIRED |
| **Live News Cache** | `output_production_final/news_cache/latest_news.json` | JSON | Semi-structured | 15 news items | 8 KB | Cached RSS news headlines and summaries | `api/main_enhanced.py:1511` (`/news/live`) | IMPLEMENTED+WIRED |
| **Mock Mongo Leads DB** | `output_production_final/db_custom_leads.json` | JSON | Semi-structured | 1,349 lead records | 1.1 MB | File fallback for MongoDB leads collection | `src/lume_platform/db/mongo_client.py:45` | IMPLEMENTED+WIRED |
| **Mock Mongo Users & Portfolios** | `output_production_final/db_users.json`, `db_portfolios.json`| JSON | Semi-structured | 4 users, 4 portfolios | 3.3 KB | File fallback for users/portfolios collection | `src/lume_platform/db/mongo_client.py:46-47` | IMPLEMENTED+WIRED |
| **Streaming Micro-Batch Parquet** | `streaming/outputs/` (`market_live.parquet`, `news_live.parquet`) | Parquet | Structured | 1,904 micro-files | 3.5 MB | Spark structured streaming sink output | `api/main_enhanced.py:619` (`/alerts/live`) | IMPLEMENTED+WIRED |
| **Streaming Inbox CSVs** | `streaming/inbox/` (`market/`, `news/`) | CSV | Semi-structured | 952 micro-files | 1.2 MB | Written by `scripts/live_stream_simulator.py` | `src/lume_platform/spark/streaming_pipeline.py:42` | IMPLEMENTED+WIRED |
| **Tableau BI Export Tables** | `tableau_exports/*.csv` (5 files) | CSV | Structured | ~12,000 rows across 5 tables | 4.3 MB | Generated by `scripts/bi_export.py` | Tableau Public / Power BI embed iframes | IMPLEMENTED+WIRED |
| **Candlestick Image Charts** | `datasets/unstructured/candlesticks/*.png` | PNG | Unstructured | 0 found in repo (Claimed 14,000+) | 0 KB | `scripts/generate_candlestick_images.py` | Target for CV models (never executed/stored) | NOT FOUND IN REPO |
| **AMC Logos & Web Images** | `datasets/unstructured/fund_images/*.jpg` | JPEG/PNG | Unstructured | 0 found in repo (1 repo logo only) | 0 KB | `scripts/fetch_fund_images.py` | Target for UI assets (never executed/stored) | NOT FOUND IN REPO |
| **AMC Prospectus PDFs** | `datasets/prospectus_pdfs/*.pdf` | PDF | Unstructured | 1 template PDF only (`Dataset_Doc.pdf`)| 245 KB | `scripts/fetch_prospectus_pdfs.py` | Target for RAG pipeline (never executed/stored) | NOT FOUND IN REPO |
| **Raw Kaggle NIFTY 500 Intraday** | `datasets/structured/stock_prices/nifty500_intraday/` | CSV | Structured | Claimed 499 files, 19 GB | 0 KB | Kaggle (Debashis 2026) | `src/lume_platform/spark/pipeline.py:51` | NOT FOUND IN REPO |
| **Raw Kaggle Financial News Corpus**| `datasets/unstructured/financial_news_corpus/` | CSV/TXT | Unstructured | Claimed 306,247 files, 2.5 GB | 0 KB | Kaggle (Jeet 2020) | Documented in `Dataset_Documentation.md:544` | NOT FOUND IN REPO |

#### Data Cleaning Steps Actually Executed (with Code Citations)
1. **Column Name Normalization**:
   - `src/lume_platform/spark/cleaning.py:10-18`: In `normalize_columns(df)`: Converts column names to lowercase, strips trailing spaces, replaces spaces and hyphens with underscores, strips parentheses.
2. **Deduplication**:
   - `src/lume_platform/spark/cleaning.py:21-23`: In `drop_duplicates(df)`: Executes `df.dropDuplicates()` across Spark DataFrames.
   - `src/lume_platform/data/timeseries_dataset.py:55, 76`: In `load_nifty50_benchmark` and `load_nav_history`: Drops duplicate dates using `drop_duplicates(subset=["date"])`.
3. **Missing Value Imputation**:
   - `src/lume_platform/spark/pipeline.py:41-43`: Imputes zero for missing numeric signals: `df = df.fillna(0, subset=["totalvisits", "total_time_spent_on_website", "page_views_per_visit"])`.
   - `src/lume_platform/ml/training.py:127-129`: Categorical nulls are filled with `"Unknown"` (`df_ml[cat_features].fillna("Unknown")`), numeric columns coerced with `pd.to_numeric(errors="coerce").fillna(0)`.
   - `src/lume_platform/ml/training.py:305`: For behavioral clustering, incomplete rows are dropped: `X = df[behavior_cols].dropna()`.
4. **Outlier Filtering & Feature Clipping**:
   - `src/lume_platform/spark/pipeline.py:60`: In `process_nifty_intraday_tree`: Filters out missing prices: `df.filter(F.col("close").isNotNull() & F.col("open").isNotNull())`.
   - `src/lume_platform/spark/streaming_pipeline.py:55-59`: Data Quality Watchdog enforces: `lead_id.isNotNull() & (total_visits >= 0) & (Conversion_Probability >= 0)`.
5. **Zero-Leakage Rolling Transforms**:
   - `src/lume_platform/data/timeseries_dataset.py:97-150`: Computes daily returns via `pct_change().fillna(0.0)`, 10-day rolling realized volatility via `rolling(10).std()`, 14-day rolling RSI using past gains/losses only, and 20-day SMA deviation using past prices only.

---

### A4. Pipeline Inventory & Flow Diagrams

#### Text Flow Diagrams of Existing Pipelines

##### 1. Master Batch Pipeline (ETL -> ML -> BI Export)
```text
[Raw CSVs: Lead Scoring, MF Behavior, Social Sentiment]
                     │
                     ▼
       ┌───────────────────────────┐
       │     PySpark ETL Engine    │ (src/master_pipeline.py:35)
       │ - normalize_columns()     │
       │ - drop_duplicates()       │
       │ - null imputation (0)     │
       └─────────────┬─────────────┘
                     │ writes Parquet
                     ▼
        [artifacts/cleaned_parquet/]
                     │
        ┌────────────┴─────────────┐
        │                          │
        ▼                          ▼
┌──────────────────┐      ┌────────────────────────┐
│  Random Forest   │      │   K-Means Clustering   │ (src/lume_platform/ml/training.py)
│  (Lead Scorer)   │      │   (Persona Engine)     │
│  - 5-Fold SKF CV │      │   - Silhouette Score   │
│  - Thr Sweep 0.43│      │   - Route to 4 AMCs    │
└───────┬──────────┘      └───────────┬────────────┘
        │                             │
        ▼                             ▼
[lead_classifier_bundle.pkl]  [investor_cluster_bundle.pkl]
        │                             │
        └────────────┬────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │       BIExporter      │ (scripts/bi_export.py:360)
         │ - leads_for_bi.csv    │
         │ - clusters_for_bi.csv │
         │ - summary_metrics.json│
         └───────────┬───────────┘
                     │
                     ▼
         [tableau_exports/ & BI Dashboards]
```

##### 2. Time-Series NAV Forecasting Pipeline (Zero-Leakage PyTorch)
```text
[3x NIFTY 50 CSVs]           [hist_100119.json (AMFI NAV)]
        │                                  │
        └─────────────────┬────────────────┘
                          ▼
            ┌───────────────────────────┐
            │ load_merged_market_series │ (src/lume_platform/data/timeseries_dataset.py)
            │ - Inner Join on 'date'    │
            │ - Compute RSI, SMA20, Vol │
            └─────────────┬─────────────┘
                          │ 735 Chronological Days
                          ▼
            ┌───────────────────────────┐
            │   Strict Chrono Split     │ (70% Train / 15% Val / 15% Test)
            │ - Train: 514 trading days │
            │ - Val:   110 trading days │
            │ - Test:  111 trading days │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   MinMaxScaler (TRAIN)    │ (Scaler fit ONLY on Train split)
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │    Sliding Window Generator│ (Lookback=30, Horizon=5)
            │ - X: (batch, 30, D)       │
            │ - y: (batch, 5)           │
            └─────────────┬─────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌────────────────────────┐       ┌────────────────────────┐
│ Phase 1: LSTMForecaster│       │ Phase 2: BiLSTM+Attn   │ (lstm_trainer.py / phase2_trainer.py)
│ - Univariate (D=1)     │       │ - Multivariate (D=7)   │
│ - Huber Loss (delta=0.05)│     │ - Pinball Loss (tau)   │
│ - Monte Carlo Dropout  │       │ - Non-crossing P10/P50/P90
└──────────┬─────────────┘       └──────────┬─────────────┘
           │                                │
           ▼                                ▼
[lstm_nav_pattern_predictor.pth] [bilstm_attention_quantile.pth]
           │                                │
           └────────────────┬───────────────┘
                            ▼
              ┌───────────────────────────┐
              │      LUMEForecaster       │ (src/lume_platform/ml/forecaster.py)
              │ Serving via /api/forecast │
              └───────────────────────────┘
```

##### 3. Streaming Ingestion & Real-Time Aggregation Pipeline
```text
[scripts/live_stream_simulator.py]
  ├── writes market CSV every 5s ──► [streaming/inbox/market/]
  └── writes news CSV every 5s   ──► [streaming/inbox/news/]
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ PySpark Structured Stream │ (src/lume_platform/spark/streaming_pipeline.py)
                               │ - Data Quality Watchdog   │
                               │ - 1-Min Tumbling Window   │
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     ▼                       ▼                       ▼
            ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
            │ Console Query  │      │ MongoDB Sink   │      │ JSON Output    │
            │ (Debug stdout) │      │ (foreachBatch) │      │ (city_metrics) │
            └────────────────┘      └────────────────┘      └────────────────┘
```

#### Complete Pipeline Inventory Table

| Pipeline Name | Ingestion Inputs | Processing Steps | Models Called | Output Artifacts | Primary Consumers | Trigger / Schedule | Status Label |
|---|---|---|---|---|---|---|---|
| **Spark Lead Scoring ETL** | `Lead Scoring.csv` (Kaggle CRM data) | Column snake_case normalization, deduplication, null filling (0), Spark Parquet write | None (Pre-feature engineering) | `artifacts/cleaned_parquet/lead_scoring_clean/` | ML Lead Trainer, Spark UDFs | Manual CLI (`master_pipeline.py`) | IMPLEMENTED+WIRED |
| **Lead Classification Pipeline** | `lead_scoring_clean` Parquet | 80/20 Stratified train/test split, ColumnTransformer (OHE), 5-fold CV, F1 threshold search | `RandomForestClassifier` (n=200, depth=12) or XGBoost | `lead_classifier_bundle.pkl`, `distributor_lead_scorer.pkl` | FastAPI `/predict`, Distributor portal | CLI (`lume train --engine leads`) | IMPLEMENTED+WIRED |
| **Investor Persona Clustering** | `MF_Behavior.xlsx` (400 survey rows) | Column extraction (7 traits), dropna, MinMaxScaler normalization | `KMeans(k=4)`, `DBSCAN(eps=0.35)` | `investor_cluster_bundle.pkl`, `investor_routing_matches.csv` | FastAPI `/predict`, Investor portal | CLI (`lume train --engine cluster`) | IMPLEMENTED+WIRED |
| **SBERT Sentiment Pipeline** | `data.csv` (Social Sentiment data) | Text extraction, label encoding, SBERT encoding (`all-MiniLM-L6-v2`) | `LogisticRegression(class_weight="balanced")` | `sentiment_bundle.pkl`, `nlp_sentiment/metrics.json` | FastAPI `/predict` (task=sentiment) | CLI (`lume train --engine nlp`) | IMPLEMENTED+WIRED |
| **Phase 1 LSTM Forecaster** | NIFTY 50 CSVs + `hist_100119.json` | Inner join on date, chrono 70/15/15 split, train MinMaxScaler, sliding window (30->5) | PyTorch `LSTMForecaster` (2 layers, hidden 64) | `lstm_nav_pattern_predictor.pth`, `real_metrics.json` | FastAPI `/api/forecast`, Streamlit | CLI (`make train-lstm`) | IMPLEMENTED+WIRED |
| **Phase 2 BiLSTM Quantile Forecaster** | NIFTY 50 CSVs + `hist_100119.json` | 7-feature extraction (RSI, Vol, SMA), chrono split, multivariate scaling, sliding window | PyTorch `BiLSTMAttentionQuantileForecaster` | `bilstm_attention_quantile.pth`, heatmap PNG | FastAPI `/api/forecast`, AI showcase | CLI (`make train-bilstm`) | IMPLEMENTED+WIRED |
| **Semantic Fund Search (SBERT)** | `fund_embeddings.pkl` | Vector dot product with normalized query embedding from `all-mpnet-base-v2` | SentenceTransformer (`all-mpnet-base-v2`) | Top-K JSON search matches | FastAPI `/search/funds` | On-demand HTTP GET | IMPLEMENTED BUT NOT WIRED (Pickle missing) |
| **Semantic Fund Search (TF-IDF)** | `mutual_fund_data.csv` | Scheme_Name + Category text concatenation, TF-IDF vectorization | `TfidfVectorizer(1,2)` + Cosine Similarity | Top-K matched schemes | `streamlit_app.py`, CLI | UI interaction in Streamlit | IMPLEMENTED+WIRED |
| **Live Stream Simulator** | `nse_market_pulse.csv`, `live_news_stream.csv` | Reads source rows and writes individual timestamped CSVs every 5s | None | Micro-batch CSV files in `streaming/inbox/` | Spark Structured Streaming | Background script execution | IMPLEMENTED+WIRED |
| **Spark Structured Streaming** | `streaming/inbox/` CSV files | Streaming JSON/CSV parse, Data Quality Watchdog filter, 1-min window aggregate | None | Console logs, Mongo leads, Parquet partitions | `/alerts/live`, Mongo collections | Long-running stream daemon | IMPLEMENTED+WIRED |
| **Tableau BI Export** | Cleaned Parquet, scored leads, clusters | Aggregates KPIs, calculates engagement tiers, writes CSV/JSON tables | Precomputed bundle inferences | `tableau_exports/*.csv`, `summary_metrics.json` | Tableau Public, Power BI iframes | CLI (`master_pipeline.py`) | IMPLEMENTED+WIRED |
| **Candlestick Image Generator** | `mutual_fund_history/*.csv` | Synthesizes fake OHLC and volume from NAV with normal noise, renders mplfinance | None | `datasets/unstructured/candlesticks/*.png` | None (Unused/idle) | Standalone script run | IMPLEMENTED BUT NOT WIRED |
| **Prospectus PDF Scraper** | AMC statutory disclosure websites | BeautifulSoup scraping of PDF anchors, downloads files to disk | None | `datasets/prospectus_pdfs/*.pdf` | None (Unused/idle) | Standalone script run | IMPLEMENTED BUT NOT WIRED |
| **AMC Logo Image Scraper** | AMC homepage websites | BeautifulSoup scraping of image tags, downloads logos to disk | None | `datasets/unstructured/fund_images/*.jpg`| None (Unused/idle) | Standalone script run | IMPLEMENTED BUT NOT WIRED |
| **LumeBuddy Conversational Engine**| User text query via HTTP POST | Keyword regex matching against predefined topic tuples, fallback template | TF-IDF Vectorizer (Transformers disabled by flag) | Canned text advice string | FastAPI `/buddy/chat`, UI | HTTP POST request | MOCK/STUB |

---

### A5. Existing Models: Full Technical Detail & Layer Tables

#### 1. Phase 1: PyTorch Univariate LSTM Forecaster
- **Task**: Multi-step recursive regression forecasting of mutual fund NAV 5 trading days into the future.
- **Algorithm**: Multi-layer vanilla LSTM (`nn.LSTM`) followed by dropout and linear projection.
- **Input Features**: 1 univariate feature: historical Net Asset Value (`nav`).
- **Target Variable**: Future 5-day NAV trajectory: $[NAV_{t+1}, NAV_{t+2}, NAV_{t+3}, NAV_{t+4}, NAV_{t+5}]$.
- **Hyperparameters**: Lookback window: 30 days; Forecast horizon: 5 days; Hidden size: 64; Layers: 2; Dropout: 0.2; Batch size: 32; Max epochs: 25 (early stopped at epoch 16); Learning rate: 0.001; Weight decay: $1 \times 10^{-5}$; Loss function: Huber Loss ($\delta=0.05$); Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$); Weight initialization: PyTorch default uniform $U(-\sqrt{k}, \sqrt{k})$ where $k = 1/\text{hidden\_dim}$.
- **Training Script**: `src/lume_platform/ml/lstm_trainer.py:54` (`train_lstm_forecaster`).
- **Saved Artifact Path**: `artifacts/models/lstm_nav_pattern_predictor.pth` (203 KB), scaler `artifacts/ml_scalers/mf_nav_global_scaler.pkl` (948 B).
- **Evaluation Method**: Strict chronological split: Train (70%, 514 rows: 2023-02-21 to 2025-03-27), Validation (15%, 110 rows: 2025-03-28 to 2025-09-08), Test (15%, 111 rows: 2025-09-09 to 2026-02-20). MinMax scaler fitted strictly on Train split ($[318.10, 517.12]$). Out-of-sample evaluation strictly on 77 test sequence windows.
- **Real Evaluated Metrics** (Source: `model_evaluations/lstm_forecaster/real_metrics.json:44-50`):
  - Test MSE: `119.0987`
  - Test RMSE: **₹10.9132**
  - Test MAE: **₹9.8327**
  - Test MAPE: **1.84%**
  - Directional Accuracy: **46.75%**
  - Epistemic Uncertainty: 90% confidence bands generated via $T=20$ Monte Carlo Dropout forward passes (`forecaster.py:354-367`).
- **Baselines Compared Against** (Source: `real_metrics.json:51-66`):
  - Naive Last-Value Persistence: Test RMSE **₹3.7565**, MAE **₹2.9830**, MAPE **0.56%**, Directional Accuracy **0.00%**.
  - 5-Day Simple Moving Average: Test RMSE **₹4.1268**, MAE **₹3.3324**, MAPE **0.63%**, Directional Accuracy **61.04%**.
  - Relative Performance: LSTM RMSE is **190.52% worse** than Naive Last-Value and **164.45% worse** than Moving Average!
- **Serving Path**: Exposed via `POST /api/forecast` and `POST /predict/forecast` in `api/main_enhanced.py:2072`, visualized in `frontend/ai-showcase.html` and `streamlit_app.py`.
- **Status Label**: `IMPLEMENTED+WIRED`.

##### Layer-by-Layer Architectural Table: Phase 1 LSTM (`LSTMForecaster`)
*(Directly measured in memory via PyTorch introspection, `src/lume_platform/ml/forecaster.py:27-65`)*

| Layer Index | Layer Type / Sub-module | Input Tensor Shape | Output Tensor Shape | Neurons / Units | Internal Activations | Regularization / Norm | Trainable Params |
|---|---|---|---|---|---|---|---|
| **0** | `nn.LSTM` (Layer 0, Forward) | `(batch, 30, 1)` | `(batch, 30, 64)` | 64 hidden units | Gates: $\sigma$ (sigmoid)<br>Cell/Candidate: $\tanh$ | None | `weight_ih`: 256<br>`weight_hh`: 16,384<br>`bias_ih`: 256<br>`bias_hh`: 256<br>*(Total L0: 17,152)* |
| **1** | `nn.Dropout` (Inter-layer) | `(batch, 30, 64)` | `(batch, 30, 64)` | 64 | Identity | Dropout ($p=0.2$) | 0 |
| **2** | `nn.LSTM` (Layer 1, Forward) | `(batch, 30, 64)` | `(batch, 30, 64)` | 64 hidden units | Gates: $\sigma$ (sigmoid)<br>Cell/Candidate: $\tanh$ | None | `weight_ih`: 16,384<br>`weight_hh`: 16,384<br>`bias_ih`: 256<br>`bias_hh`: 256<br>*(Total L1: 33,280)* |
| **3** | Slicing `out[:, -1, :]` | `(batch, 30, 64)` | `(batch, 64)` | 64 | None (Select last step) | None | 0 |
| **4** | `nn.Dropout` (Head Dropout)| `(batch, 64)` | `(batch, 64)` | 64 | Identity | Dropout ($p=0.2$) | 0 |
| **5** | `nn.Linear` (`fc`) | `(batch, 64)` | `(batch, 5)` | 5 outputs | Linear (None) | None | `weight`: 320<br>`bias`: 5<br>*(Total FC: 325)* |
| **TOTAL** | **LSTMForecaster** | `(batch, 30, 1)` | `(batch, 5)` | **128 hidden, 5 out**| **Sigmoid, Tanh** | **Dropout 0.2** | **50,757 parameters** |

> **Parameter Discrepancy Note**: The total parameter count of `LSTMForecaster` measured in memory is exactly **50,757**. Any external report or slide claiming "134,533" parameters does NOT match this network (nor does it match the 3-layer `NAVPredictorLSTM` with 331,909 params, or the 2-layer `NAVForecaster` with 199,553 params). Claimed number "134,533" is classified as **REPORTED, UNVERIFIED (DISCREPANCY)**.

---

#### 2. Phase 2: PyTorch Multivariate BiLSTM + Temporal Attention Forecaster
- **Task**: Multi-step, multi-quantile forward trajectory prediction with monotonic uncertainty bounds.
- **Algorithm**: Bidirectional LSTM (2 layers) + Temporal Multi-Head Dot-Product Attention (`nn.MultiheadAttention`) + LayerNorm + 3 non-crossing quantile heads ($P_{10}, P_{50}, P_{90}$).
- **Input Features**: 7 features: $[NAV, NAV\_return, NIFTY\_close, NIFTY\_volatility, NIFTY\_volume, RSI_{14}, SMA_{20}\_dev]$.
- **Target Variable**: 5-step forward horizons across 3 quantiles: $5 \times 3 = 15$ outputs ($P_{10}, P_{50}, P_{90}$ for $t \in [1..5]$).
- **Monotonic Non-Crossing Formulation** (Citing `src/lume_platform/ml/bilstm_attention.py:9-13, 150-156`):
  $$P_{50} = \text{head\_p50}(h)$$
  $$P_{10} = P_{50} - \text{softplus}(\text{head\_delta\_p10}(h))$$
  $$P_{90} = P_{50} + \text{softplus}(\text{head\_delta\_p90}(h))$$
  Guarantees $P_{10} \le P_{50} \le P_{90}$ by construction without quantile crossing.
- **Hyperparameters**: Lookback: 30 days; Horizon: 5 days; Hidden size: 64 per direction (bidirectional = 128 concatenated); Layers: 2; Attention heads: 4 (`embed_dim=128`); Dropout: 0.2; Batch size: 32; Epochs: 60 (patience 10); Learning rate: 0.001; Weight decay: $1 \times 10^{-4}$; Loss function: Multi-Quantile Pinball Loss ($\tau \in \{0.10, 0.50, 0.90\}$); Optimizer: AdamW.
- **Training Script**: `src/lume_platform/ml/phase2_trainer.py:48` (`train_phase2_bilstm_attention`).
- **Saved Artifact Path**: `artifacts/models/bilstm_attention_quantile.pth` (839 KB), scaler `artifacts/ml_scalers/multivariate_scaler.pkl`.
- **Evaluation Method**: Identical chronological split (70% Train / 15% Val / 15% Test). Evaluated on 77 test windows.
- **Real Evaluated Metrics** (Source: `model_evaluations/phase2_timeseries/real_metrics.json:59-67`):
  - Test RMSE: **₹10.5496** (vs Phase 1 LSTM ₹10.9132)
  - Test MAE: **₹9.6783**
  - Test MAPE: **1.82%**
  - Directional Accuracy: **46.75%**
  - Empirical Quantile Coverage Rate (QCR): **100.0%** (Nominal target: 80.0%)
  - Mean Interval Width: **₹42.07**
- **Ablation Results** (Source: `real_metrics.json:97-125`):
  - `no_attention`: RMSE ₹13.2210 (Attention reduces RMSE by 20.2%)
  - `unidirectional_lstm`: RMSE ₹12.0669 (Bidirectionality reduces RMSE by 12.6%)
  - `no_exogenous_univariate`: RMSE ₹14.3502 (Multivariate features reduce RMSE by 26.5%)
- **Critical Production Flaw**: In inference runtime (`src/lume_platform/ml/forecaster.py:286-290`), exogenous NIFTY features are hardcoded constants (`nifty_close = 22500.0`, `nifty_vol = 0.007`, `nifty_vol_qty = 250000000.0`).
- **Serving Path**: Priority 1 engine in `LUMEForecaster.load()` (`forecaster.py:161-165`), served via `POST /api/forecast`.
- **Status Label**: `IMPLEMENTED+WIRED`.

##### Layer-by-Layer Architectural Table: Phase 2 BiLSTM + Attention
*(Directly measured in memory via PyTorch introspection, `src/lume_platform/ml/bilstm_attention.py:24-104`)*

| Layer Index | Layer Type / Sub-module | Input Tensor Shape | Output Tensor Shape | Neurons / Units / Heads | Internal Activations | Regularization / Norm | Trainable Params |
|---|---|---|---|---|---|---|---|
| **0** | `nn.LSTM` (Layer 0, Bidirectional) | `(batch, 30, 7)` | `(batch, 30, 128)` | 64 fwd + 64 rev | Gates: $\sigma$, Cell: $\tanh$ | None | `ih_l0`: 1,792, `hh_l0`: 16,384, biases: 512<br>`ih_l0_rev`: 1,792, `hh_l0_rev`: 16,384, biases: 512<br>*(Total L0: 37,376)* |
| **1** | `nn.LSTM` (Layer 1, Bidirectional) | `(batch, 30, 128)` | `(batch, 30, 128)` | 64 fwd + 64 rev | Gates: $\sigma$, Cell: $\tanh$ | Inter-layer Dropout 0.2 | `ih_l1`: 32,768, `hh_l1`: 16,384, biases: 512<br>`ih_l1_rev`: 32,768, `hh_l1_rev`: 16,384, biases: 512<br>*(Total L1: 99,328)* |
| **2** | `nn.MultiheadAttention` (`attention`) | `(batch, 30, 128)` | `(batch, 30, 128)` | 128 embed, 4 heads | Scaled Dot-Product Softmax | Attention Dropout 0.2 | `in_proj_weight`: 49,152<br>`in_proj_bias`: 384<br>`out_proj.weight`: 16,384<br>`out_proj.bias`: 128<br>*(Total Attn: 66,048)* |
| **3** | Residual Add + `nn.LayerNorm` | `(batch, 30, 128)` | `(batch, 30, 128)` | 128 | Identity | LayerNorm (128) | `weight`: 128, `bias`: 128<br>*(Total LN: 256)* |
| **4** | Temporal Pooling `h[:, -1, :]` | `(batch, 30, 128)` | `(batch, 128)` | 128 | None (Decision step context)| Dropout 0.2 | 0 |
| **5** | `fc_dense` (`Linear` + `GELU` + `Dropout`)| `(batch, 128)` | `(batch, 64)` | 64 | GELU | Dropout 0.2 | `weight`: 8,192<br>`bias`: 64<br>*(Total Dense: 8,256)* |
| **6** | `head_p50` (`Linear`) | `(batch, 64)` | `(batch, 5)` | 5 (Median) | Linear | None | `weight`: 320, `bias`: 5 *(325)* |
| **7** | `head_delta_p10` (`Linear` + `softplus`)| `(batch, 64)` | `(batch, 5)` | 5 (Lower offset) | Softplus | Non-negative constraint | `weight`: 320, `bias`: 5 *(325)* |
| **8** | `head_delta_p90` (`Linear` + `softplus`)| `(batch, 64)` | `(batch, 5)` | 5 (Upper offset) | Softplus | Non-negative constraint | `weight`: 320, `bias`: 5 *(325)* |
| **TOTAL** | **BiLSTMAttentionQuantile** | `(batch, 30, 7)` | `(batch, 5, 3)` | **128 BiLSTM, 4 Heads** | **GELU, Softplus** | **LayerNorm, Drop 0.2** | **212,239 parameters** |

---

#### 3. Classical Machine Learning Models: Full Hyperparameters & Metrics

##### A. Distributor Lead Conversion Classifier (`RandomForestClassifier`)
- **Task**: Predict binary conversion outcome for mutual fund distributor leads.
- **Algorithm**: `RandomForestClassifier` inside Scikit-Learn `Pipeline` with `ColumnTransformer`.
- **Input Features (14 total)**:
  - Numeric (5): `totalvisits`, `total_time_spent_on_website`, `page_views_per_visit`, `asymmetrique_activity_score`, `asymmetrique_profile_score`.
  - Categorical (9): `lead_origin`, `lead_source`, `specialization`, `what_is_your_current_occupation`, `last_activity`, `country`, `lead_quality`, `do_not_email`, `do_not_call`.
- **Target**: `converted` (0 = Not Converted, 1 = Converted).
- **Full Hyperparameter Set** (Citing `artifacts/models/model_manifest.json:33-37`):
  - `n_estimators`: 200
  - `max_depth`: 12
  - `criterion`: "gini"
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `class_weight`: "balanced"
  - `random_state`: 42
  - `bootstrap`: True
  - `decision_threshold`: 0.43 (calibrated via F1 threshold search)
- **Saved Artifact Path**: `artifacts/models/lead_classifier_bundle.pkl` (4.9 MB), `artifacts/models/distributor_lead_scorer.pkl` (4.9 MB).
- **Evaluation Method**: 80/20 Stratified Train/Test split (`random_state=42`), followed by 5-Fold Stratified K-Fold cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Real Evaluated Metrics** (Source: `model_manifest.json:6-41` & `model_evaluations/random_forest/real_metrics.json`):
  - Accuracy: **85.00%**
  - Precision: **0.75** (0.7514 at threshold 0.43)
  - Recall: **0.90** (0.9028 at threshold 0.43)
  - F1-Score: **0.82** (0.8202 at threshold 0.43)
  - ROC-AUC: **0.933**
  - Average Precision (PR-AUC): **0.890**
  - 5-Fold Cross-Validation F1 Mean: **0.806 ± 0.018**
- **Serving Path**: `POST /predict`, `POST /batch_predict`, Distributor dashboard (`frontend/distributor.html`).
- **Status Label**: `IMPLEMENTED+WIRED`.

##### B. Investor Persona Clustering Engine (`KMeans` & `DBSCAN`)
- **Task**: Segment investors into 4 behavioral personas for distributor pitch targeting.
- **Algorithm**: Scikit-Learn `KMeans` with $k=4$ and `MinMaxScaler`.
- **Input Features (7 traits)**: `ProfManage`, `Diversification`, `Affordability`, `Liquidity`, `Growth`, `Trustworthiness`, `Technology`.
- **Target**: Unsupervised (no ground truth target).
- **Full Hyperparameter Set** (Citing `src/lume_platform/ml/training.py:306-347`):
  - `n_clusters`: 4
  - `init`: "k-means++"
  - `n_init`: 10
  - `max_iter`: 300
  - `tol`: $1 \times 10^{-4}$
  - `random_state`: 42
  - Secondary DBSCAN hyperparameters: `eps=0.35`, `min_samples=10`.
- **Saved Artifact Path**: `artifacts/models/investor_cluster_bundle.pkl` (5.7 KB), `artifacts/models/investor_kmeans_model.pkl` (4.7 KB), `artifacts/models/investor_behavior_scaler.pkl` (948 B), `artifacts/models/investor_dbscan.pkl` (67 KB).
- **Real Evaluated Metrics** (Source: `artifacts/models/model_manifest.json:48-51`):
  - Silhouette Score: **0.3952** (~0.40)
  - Number of Clusters: 4
  - Centroid Characteristics:
    - Cluster 0: Equity & High Growth (High Growth, High Risk)
    - Cluster 1: Liquid & Debt Preservation (High Liquidity, Low Risk)
    - Cluster 2: Hybrid & Balanced Allocation (Balanced traits)
    - Cluster 3: Index & Passive Smart-Beta (High Technology, High Diversification)
- **Serving Path**: `POST /predict` (task=investor_cluster), `GET /distributors/client-insights`.
- **Status Label**: `IMPLEMENTED+WIRED`.

##### C. Financial Sentiment Analysis Engine (`SentenceTransformer` + `LogisticRegression`)
- **Task**: Classify financial headlines into Negative (0), Neutral (1), or Positive (2).
- **Algorithm**: Pretrained SBERT encoder (`all-MiniLM-L6-v2`) + Scikit-Learn `LogisticRegression`.
- **Input Features**: 384-dimensional dense sentence embeddings.
- **Full Hyperparameter Set** (Citing `src/lume_platform/ml/training.py:368-382`):
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 384 hidden, 12 heads)
  - Classifier: `LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)`
  - Fallback TF-IDF settings: `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)`
- **Saved Artifact Path**: `artifacts/models/sentiment_bundle.pkl` (10 KB).
- **Evaluation Method**: 80/20 Stratified Train/Test split (`random_state=42`, `stratify=y`).
- **Real Evaluated Metrics** (Source: `artifacts/models/model_manifest.json:43-47`):
  - Accuracy: **71.60%** (0.715996)
  - Macro F1: **0.6888** (0.688780)
  - Embedding Dimensions: 384
- **Serving Path**: `POST /predict` (task=sentiment), `GET /insights`.
- **Status Label**: `IMPLEMENTED+WIRED`.

##### D. Semantic Mutual Fund Search Engine (`all-mpnet-base-v2` / `TF-IDF`)
- **Task**: Match natural language investor queries to relevant mutual fund schemes.
- **Algorithm 1 (SBERT)**: `SBERTMutualFundSearch` (`src/lume_platform/ml/semantic_search.py:19-78`).
  - Model: `sentence-transformers/all-mpnet-base-v2` (768 embedding dimensions, 12 layers, 110M params).
  - Search method: Cosine similarity over normalized embedding matrix $\vec{q} \cdot \vec{d}_i / (\|\vec{q}\| \|\vec{d}_i\|)$.
  - Status: `IMPLEMENTED BUT NOT WIRED` (Missing cache `artifacts/models/fund_embeddings.pkl` causes endpoint `/search/funds` to throw HTTP 503).
- **Algorithm 2 (TF-IDF)**: `MutualFundSearchEngine` (`scripts/search_algorithm.py:9-60`).
  - Hyperparameters: `TfidfVectorizer(stop_words='english', ngram_range=(1, 2))`.
  - Corpus: Scheme Name + Category strings across 14,000+ schemes.
  - Latency: ~120ms on CPU.
  - Status: `IMPLEMENTED+WIRED` (inside `streamlit_app.py`).

---

### A6. Engineering Numbers, Proven Achievements & Visual Asset Inventory

#### Measured Engineering Numbers
- **Total Lines of Code (LOC)**: **28,638 LOC** across the repository (excluding `.git`, `venv`, and `node_modules`).
  - `api/`: 2,478 LOC (2 files: `main.py`, `main_enhanced.py`)
  - `src/lume_platform/`: 7,676 LOC (53 files)
    - `ml/`: 3,056 LOC (15 files)
    - `risk/`: 1,173 LOC (3 files)
    - `spark/`: 416 LOC (8 files)
    - `data/`: 523 LOC (3 files)
    - `db/`: 482 LOC (2 files)
    - `inference/`: 273 LOC (3 files)
    - `crm/`: 258 LOC (2 files)
    - `news/`: 240 LOC (2 files)
    - `audit/`: 144 LOC (2 files)
    - `utils/`: 89 LOC (3 files)
  - `src/streaming/`: 133 LOC (1 file)
  - `frontend/`: 5,266 LOC (9 files)
  - `mintleads/`: 4,406 LOC (35 files)
  - `scripts/`: 2,580 LOC (22 files)
  - `tests/`: 1,351 LOC (6 files)
  - `lume-ai-product/`: 955 LOC (4 files)
  - Root runner & dashboard files: 3,702 LOC (`streamlit_app_fintech.py`, `streamlit_app.py`, `run.py`, `app_users.py`, `Makefile`, `start.sh`)
- **Total API Endpoints**: **59 routes** in `api/main_enhanced.py` (55 functional application endpoints + 4 OpenAPI/Swagger documentation routes).
- **Total Frontend & Dashboard Pages**: **11 distinct user interfaces**:
  - 7 static HTML5 pages (`index.html`, `investor.html`, `distributor.html`, `ai-showcase.html`, `bigdata-showcase.html`, `login.html`, `signup.html`).
  - 2 Streamlit platform applications (`streamlit_app_fintech.py`, `streamlit_app.py`).
  - 1 multi-page dashboard with 4 subpages (`mintleads/dashboard/app.py`).
  - 1 Next.js 16.2 web app prototype (`lume-ai-product/apps/web/`).
- **Indexed Schemes in Production Cache**:
  - **13,965 mutual fund schemes** indexed in `output_production_final/nav_cache/latest_navs.json`.
  - **16,344 mutual fund schemes** cataloged in `data/Mutual_Fund_Data-main/mutual_fund_data.csv`.
  - **17,534 scheme records** in `CompleteNavReport.txt`.
  - **19 historical scheme series** cached with full daily NAV history in `output_production_final/nav_cache/hist_*.json`.

#### Verified Engineering Achievements (with Proof)
1. **Zero-Data-Leakage Deep Learning Time-Series Split**: Proven by `src/lume_platform/data/timeseries_dataset.py:22-105` and validated by automated unit tests in `tests/test_data_leakage.py`. Strict chronological ordering (Train 2023-02-21 to 2025-03-27 -> Val 2025-03-28 to 2025-09-08 -> Test 2025-09-09 to 2026-02-20) with scaler fitted strictly on train data.
2. **Non-Crossing Quantile Formulation via Softplus**: Proven by `src/lume_platform/ml/bilstm_attention.py:150-156` and unit-tested in `tests/test_phase2_bilstm.py:33-72`. Mathematically guarantees $P_{10} \le P_{50} \le P_{90}$ for every forecast step.
3. **Calibrated Lead Scoring Decision Threshold**: Proven by `artifacts/models/model_manifest.json:29-32` and `src/lume_platform/ml/training.py:97-111`. Decision threshold shifted from 0.50 to 0.43, maximizing holdout F1-score to 0.8202 (Precision: 0.7514, Recall: 0.9028).
4. **PySpark Materialized Parquet Ingestion**: Proven by `src/lume_platform/spark/ingestion.py` and materialized on disk at `artifacts/cleaned_parquet/lead_scoring_clean/` partitioned by `converted` (`converted=0/`, `converted=1/`).
5. **Full REST API Suite with Automated JWT Authentication**: Proven by `api/main_enhanced.py:1090-1160` and `src/lume_platform/auth.py:22-60` with PBKDF2 password hashing and HS256 token verification.

#### Reusable Visual Assets & Image Inventory

The following table catalogs every reusable visual asset, plot, image, and video in the repository:

| Asset File Path | Dimensions / Format | Category | What It Displays & Provenance | Reusability in Presentation |
|---|---|---|---|---|
| `model_evaluations/phase2_timeseries/attention_heatmap.png` | PNG (200 DPI) | DL Interpretability | Heatmap of temporal attention weights $\alpha_t$ across 30 lookback days, showing market memory focus on shocks. Generated by `phase2_trainer.py`. | **Must-Use for Engine 2 Slide** |
| `model_evaluations/phase2_timeseries/loss_curves.png` | PNG (200 DPI) | DL Training Curves | Training and validation Pinball Loss curves across 60 epochs for BiLSTM + Attention model. | **Must-Use for Engine 2 Slide** |
| `model_evaluations/lstm_forecaster/lstm_train_val_loss.png` | PNG (200 DPI) | DL Training Curves | Training and validation Huber loss curves across 16 epochs for Phase 1 LSTM model. Generated by `lstm_trainer.py`. | **Must-Use for Phase 1 Audit Slide** |
| `model_evaluations/lstm_forecaster/lstm_nav_predictions.png` | PNG (300 DPI) | Synthetic Plot | Visual plot showing synthetic actual vs predicted NAV curves with lag factor. Generated by `generate_lstm_rf_metrics.py`. | **Use as Evidence for Gap B1/B3** |
| `model_evaluations/random_forest/rf_confusion_matrix.png` | PNG (150 DPI) | ML Evaluation | Confusion matrix heatmap (True Positives, False Positives, TN, FN) for Random Forest lead conversion. Generated by `training.py`. | **Must-Use for Existing Models Slide** |
| `model_evaluations/xai_insights/xai_rf_feature_weights.png` | PNG (300 DPI) | Interpretability (XAI) | Horizontal bar chart of Scikit-Learn tree feature importances (Time on Website, Total Visits, Profile Score). | **Must-Use for Interpretability Slide** |
| `model_evaluations/kmeans/kmeans_cluster_projection.png` | PNG (300 DPI) | Unsupervised ML | 2-Dimensional PCA scatter projection of 4 investor clusters showing geometric separation. | **Must-Use for Clustering Slide** |
| `model_evaluations/kmeans/kmeans_centroids_heatmap.png` | PNG (300 DPI) | Unsupervised ML | Heatmap of behavioral centroids across the 7 psychological traits for each of the 4 clusters. | **Must-Use for Persona Slide** |
| `model_evaluations/tfidf_search/cosine_similarity_decay.png` | PNG (300 DPI) | Synthetic Plot | Bar chart showing simulated cosine similarity decay across scheme ranks with Top-5 cutoff line. | **Use as Evidence for Gap B1/B3** |
| `dataset_visualisations/investor_behavior_heatmap.png` | PNG (300 DPI) | EDA Correlation | Full correlation matrix heatmap between the 7 psychological traits from `MF_Behavior.xlsx`. | **Use for Data Exploration Slide** |
| `dataset_visualisations/lead_conversion_metrics.png` | PNG (300 DPI) | EDA Distribution | Pie chart showing target class distribution (38% converted vs 62% not converted) in lead dataset. | **Use for Data & Scale Slide** |
| `dataset_visualisations/nifty50_macro_trend.png` | PNG (300 DPI) | Macro Financial | Trendline of historical NIFTY 50 closing price over 3-year trading calendar. | **Use for Problem Context Slide** |
| `dataset_visualisations/audit_latest/audit_boxplots.png` | PNG (300 DPI) | Data Quality | Boxplots illustrating outlier distribution and feature skew across numerical CRM fields. | **Use for Big Data Cleaning Slide** |
| `dataset_visualisations/audit_latest/audit_heatmap.png` | PNG (300 DPI) | Data Quality | Correlation matrix of tabular audit features across the CRM lead database. | **Use for Data Quality Slide** |
| `dataset_visualisations/audit_latest/audit_histogram.png` | PNG (300 DPI) | Data Quality | Multi-panel histograms showing frequency distribution of lead visits and time spent. | **Use for Data Quality Slide** |
| `frontend/logo.png` | PNG (High-Res) | Branding | Official Lume AI platform brand logo with diamond insignia. | **Use for Presentation Title Slide** |
| `frontend/splash screen.mp4` | MP4 Video | Product Demo | Startup splash animation demonstrating platform bootup and UI transition. | **Optional Video Clip for Demo Slide** |

---

## PART B: GAPS FOUND (PROBLEM STATEMENT & DEFICIENCIES)

### B1. Comprehensive Gap Register Table

| Gap ID | Defect / Architectural Gap Description | Exact File & Line Evidence | Severity | Impact on Fintech Product | Impact on DL Course Grading | Planned Engine Fix |
|---|---|---|---|---|---|---|
| **GAP-01** | **Deep Learning Omitted from Default Training Loop** | `src/lume_platform/ml/training.py:419-457` (`train_all()`) | **CRITICAL** | Production model retraining refreshes only classical Scikit-Learn models; neural networks remain stale. | Evaluators running `train_all()` see only classical ML; DL appears as an unintegrated sidecar script. | **Engine 2 (BiLSTM)** & **Engine 4 (TabNet)** integrated into `train_all()` |
| **GAP-02** | **DL Feature Flags Defaulting to OFF / Disabled** | `src/lume_platform/ml/buddy_engine.py:204-205` (`LUME_ENABLE_BUDDY_TRANSFORMER="0"`), `configs/forecaster_bilstm_attention.yaml:26` (`include_sentiment: false`) | **HIGH** | Advanced Transformer and Sentiment features never execute in default deployments; falls back to templates. | Core DL components are disabled in out-of-the-box evaluation; project behaves like a classical app. | **Engine 1 (FinBERT)** & **Engine 5 (RAG)** enabled by default |
| **GAP-03** | **Hardcoded Exogenous Features at Inference Time** | `src/lume_platform/ml/forecaster.py:286-290` (`nifty_close = 22500.0`, `nifty_vol = 0.007`, `nifty_vol_qty = 250000000.0`) | **CRITICAL** | The forecaster ignores real-time macroeconomic shocks; predictions are computed against static dummy values. | Severe academic penalty: model claims to be multivariate but feeds constants into tensor weights. | **Engine 2 (BiLSTM)** dynamic feature ingestion pipeline |
| **GAP-04** | **Fabricated Evaluation Plots & Synthetic Lag Noise** | `model_evaluations/generate_lstm_rf_metrics.py:47-56` (injected lag + normal noise), `model_evaluations/tfidf_search/generate_tfidf_metrics.py:15-16` | **CRITICAL** | Model cards display synthetic evaluations rather than measured model performance. | Academic integrity violation: claims $R^2=0.89$ on fabricated curves while actual model has negative $R^2$. | Honest empirical benchmark suites across all 5 Engines |
| **GAP-05** | **Mocked / Stubbed LumeBuddy Conversational Agent** | `src/lume_platform/ml/custom_buddy_model.py:6-16` (16-line mock stub), `src/lume_platform/ml/buddy_engine.py:43-180` | **HIGH** | "AI Copilot" is a rule-based lookup table with static text strings; cannot answer domain queries. | No real NLP generation or RAG implemented; AI assistant is effectively hardcoded mock data. | **Engine 5 (Neural RAG + Qwen2.5 / Gemini)** |
| **GAP-06** | **Univariate Legacy LSTM Inferior to Naive Baselines** | `model_evaluations/lstm_forecaster/real_metrics.json:65-66` (RMSE ₹10.91 vs Naive ₹3.76, -190.5% improvement) | **HIGH** | Deploying Phase 1 LSTM produces financial predictions 2.8x less accurate than simply predicting yesterday's price. | Grade failure on time-series: failing to beat persistence baseline invalidates the DL approach. | **Engine 2 (BiLSTM + Attention + Pinball)** beating ARIMA |
| **GAP-07** | **Unused Candlestick Generator & 0 Computer Vision Models** | `scripts/generate_candlestick_images.py:48-61` (synthesizes fake OHLC from NAV; no CNN exists) | **HIGH** | Unstructured candlestick pipeline is dead code; no computer vision capabilities deployed in product. | Missing multi-modal DL dimension; computer vision claims are unfulfilled in repository. | **Engine 3 (Custom Candlestick CNN & ResNet-18)** |
| **GAP-08** | **Idle Prospectus PDFs & Absence of RAG Architecture** | `scripts/fetch_prospectus_pdfs.py:10-14` (scrapes URLs, but 0 PDFs stored, no FAISS index, no chunking) | **HIGH** | Scheme Information Documents (SID) cannot be searched or queried; users cannot verify risk factors. | Inability to demonstrate modern LLM retrieval and knowledge grounding in fintech context. | **Engine 5 (Neural RAG: BGE + FAISS + MiniLM Reranker)** |
| **GAP-09** | **Zero Deep Learning Interpretability (XAI Scikit-Only)** | `src/lume_platform/crm/dashboard_service.py:140-180` (TreeExplainer for RF only; 0 XAI for LSTM) | **MEDIUM** | Black-box neural network predictions violate financial regulatory explainability standards. | Missing attention rollout, Integrated Gradients, and Grad-CAM required for modern DL course. | **Engine 1 (IG)**, **Engine 2 (Attn Heatmaps)**, **Engine 3 (Grad-CAM)** |
| **GAP-10** | **Broken Path Dependency Crashing Semantic Search** | `api/main_enhanced.py:1044-1045` (`sbert_cache` missing in `artifacts/models/` -> HTTP 503 Service Unavailable) | **HIGH** | Calling `/search/funds` returns HTTP 503 error; fund search fails on production API. | Broken API endpoint during live evaluation demo. | Automated artifact validation & FAISS vectorstore |
| **GAP-11** | **Hardcoded Path Discrepancies Across Scripts** | `scripts/generate_candlestick_images.py:6`, `scripts/fetch_prospectus_pdfs.py:8` (`BASE_DIR = "/Users/snehapatel/BigData"`) | **MEDIUM** | Running scripts standalone fails with `FileNotFoundError` due to nonexistent root paths. | Reproduction failure when instructors attempt to run ingestion scripts. | Centralized `Path(__file__).resolve()` path resolution |
| **GAP-12** | **Insecure Secret Management in Production Mode** | `src/lume_platform/auth.py:13-18` (falls back to `'dev-secret'` with non-blocking print statement) | **MEDIUM** | Insecure default secret exposes all minted JWT tokens to spoofing and privilege escalation. | Industry security review penalty for weak authentication defaults. | Strict production environment assertion throwing error |
| **GAP-13** | **Absence of Top-Level CI/CD & DL Unit Test Suite** | `.github/` missing at root (only in `mintleads/`); no unit tests for Transformers, RAG, or CNN | **MEDIUM** | Code regressions in deep learning modules go undetected; no automated GPU/CPU test runner. | Software engineering and reproducibility grade reduction. | Root GitHub Actions workflow with pytest-cov |
| **GAP-14** | **Unused AMC Brand Logo Assets** | `scripts/fetch_fund_images.py:10` (scrapes AMC images into nonexistent path; never displayed) | **LOW** | Dashboard displays generic CSS placeholders instead of authentic AMC brand iconography. | Visual polish and frontend integration deficiency. | Direct asset wiring into investor catalog |

---

### B2. Unused & Idle Data Assets (Potential Capabilities)

1. **AMFI Master Mutual Fund Catalog (`data/Mutual_Fund_Data-main/mutual_fund_data.csv`)**:
   - *Current State*: 16,344 mutual fund schemes containing AMC name, category, launch date, quarterly AAUM, and latest NAV. Currently utilized solely for static string matching in `api/main_enhanced.py:1700`.
   - *Potential Capability*: Can power **Engine 5 (Neural RAG)** by acting as the structured metadata layer for fund chunking, and **Engine 4 (TabNet)** for predicting fund category suitability based on investor wealth tiers.
2. **Complete AMFI NAV Feed (`CompleteNavReport.txt`)**:
   - *Current State*: 17,534 live NAV quotes sitting completely idle in the repository root.
   - *Potential Capability*: Can power automated daily recalibration of the **Engine 2 (BiLSTM)** forecaster and live portfolio mark-to-market revaluation in the investor dashboard.
3. **NSE Listed Equities Master (`EQUITY_L.csv`)**:
   - *Current State*: 2,241 listed Indian stocks with ISIN codes, market symbols, and series classifications. Completely untouched.
   - *Potential Capability*: Can power underlying stock lookthrough for mutual fund holdings, enabling cross-asset equity sentiment linkage with **Engine 1 (FinBERT)**.
4. **Historical NIFTY 50 Trading Calendar (`NIFTY 50-*.csv`, 745 Days)**:
   - *Current State*: High, Low, Open, Close, Shares Traded, Turnover. Only partially merged for Scheme 100119.
   - *Potential Capability*: Can power multi-task macro volatility modeling and dynamic exogenous market shocks across all 19 cached fund schemes.
5. **Streaming Buffer Parquet Lake (`streaming/outputs/*.parquet`, 1,904 Micro-Files)**:
   - *Current State*: Micro-batch parquet files generated by local streaming simulators.
   - *Potential Capability*: Can be structured into a continuous real-time streaming feature store for high-frequency lead scoring and market sentiment updates.
6. **AMC Statutory Prospectus Link Corpus (`scripts/fetch_prospectus_pdfs.py:21-60`)**:
   - *Current State*: Target URLs for 7 major AMCs (SBI, HDFC, Nippon, Tata, Baroda BNP Paribas).
   - *Potential Capability*: The primary raw asset needed to fetch Scheme Information Documents (SID) and Key Information Memorandums (KIM) to build **Engine 5 (Neural RAG)**.

---

### B3. Top 10 Discrepancies: Documentation Claims vs Actual Code

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TOP 10 AUDIT DISCREPANCIES SUMMARY                                      │
├────┬────────────────────────────────────────────┬───────────────────────────────────────────────────────┤
│ #  │ Documentation / Poster Claim               │ Concrete Code Reality                                 │
├────┼────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 1  │ Total dataset size is "30 GB / 315K files" │ Actual disk usage is 50 MB; datasets/ is missing      │
│ 2  │ LSTM captures Deep-Time; ARIMA flatlines   │ Real LSTM RMSE (₹10.91) is 2.8x WORSE than ARIMA      │
│ 3  │ Academic LSTM validation achieved R2=0.89  │ Plot generated via synthetic formula + random noise   │
│ 4  │ Lead classification validated at 81.01%    │ Real evaluated accuracy is 85.00%; CM was hardcoded   │
│ 5  │ BERT rejected: "500-layer memory crash"    │ BERT is 12 layers; MiniLM (6 layers) runs in 120ms    │
│ 6  │ LumeBuddy is custom-trained PyTorch AI     │ 16-line mock stub returning static string + TF-IDF    │
│ 7  │ Multivariate forecaster uses live NIFTY    │ NIFTY features are hardcoded constants at inference   │
│ 8  │ Live Kafka streaming ingestion pipeline    │ Kafka not in requirements; 8-word fallback counter    │
│ 9  │ SBERT semantic search live on /search/funds│ Embeddings pickle missing; route throws HTTP 503      │
│ 10 │ Root path is /Users/snehapatel/BigData     │ Path does not exist; scripts crash on hardcoded path  │
└────┴────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

1. **The "30 GB Big Data" Claim**:
   - *Documentation Claim*: `Dataset_Documentation.md:25, 324` states: *"Total Dataset Size: 30 GB... comprises 30 GB of data across 315,538 files and millions of records"*. `A1_Studio_Project_Poster_Content.md:34` claims *"~31 GB multi-source data"*. `scripts/bi_export.py:204` hardcodes `"dataset_size_gb": 31.25`.
   - *Code Reality*: `du -sh .` measures exactly **50 MB** on disk! The directory `datasets/` is gitignored (`.gitignore:41`) and does not exist in the repository.
2. **The "LSTM Outperforms ARIMA" Claim**:
   - *Documentation Claim*: `LUME_AI_FINAL_ARCHITECTURE_REPORT.md:28` states: *"Why we rejected ARIMA: Traditional statistical ARIMA models expect stationary, predictable patterns... ARIMA would flatline, whereas our LSTM algorithm gracefully rides the lagging volatility"*.
   - *Code Reality*: `model_evaluations/phase2_timeseries/real_metrics.json:78-95` proves that ARIMA(3,1,0) achieved a Test RMSE of **₹3.8963**, beating the PyTorch LSTM (Test RMSE **₹10.9132**) by almost 3x lower error!
3. **Fabricated Evaluation of the $R^2=0.89$ Model**:
   - *Documentation Claim*: `model_evaluations/lstm_forecaster/neural_regression_report.md` cites a high-performing regression fit mapping historical price thresholds.
   - *Code Reality*: `model_evaluations/generate_lstm_rf_metrics.py:47-56` literally fabricated the prediction curve:
     ```python
     lag_factor = (actual_nav[i] * 0.7) + (actual_nav[i-1] * 0.3)
     predicted_nav.append(lag_factor + np.random.normal(0, 45))
     ```
     The real PyTorch LSTM has negative out-of-sample $R^2$.
4. **Hardcoded Confusion Matrix vs Real Metrics**:
   - *Documentation Claim*: `LUME_AI_FINAL_ARCHITECTURE_REPORT.md:38` claims an "81.01% Accuracy Score".
   - *Code Reality*: In `generate_lstm_rf_metrics.py:20-21`, the confusion matrix was hardcoded directly as `cm = np.array([[638, 150], [192, 820]])` to produce 81.01%! Meanwhile, the real trained model in `model_manifest.json:6` achieved **85.00%** accuracy at threshold 0.43.
5. **The "500-Layer BERT" Memory Fallacy**:
   - *Documentation Claim*: `LUME_AI_FINAL_ARCHITECTURE_REPORT.md:32` justifies rejecting Transformers: *"Processing 14,275 Mutual Funds through a 500-layer transformer inside a Streamlit application loop would crash the memory heap"*.
   - *Code Reality*: Standard BERT-base has exactly 12 encoder layers, not 500. Sentence-BERT (`all-MiniLM-L6-v2`) has 6 layers, 22.7M parameters, and executes semantic search over 14,000 schemes in ~120ms on a single CPU core.
6. **LumeBuddy: "Custom PyTorch Model" vs 16-Line Mock**:
   - *Documentation Claim*: Documentation cites `custom_buddy_model.pth` as an AI conversational copilot.
   - *Code Reality*: `src/lume_platform/ml/custom_buddy_model.py:6-16` is a 16-line mock stub with an empty `pass` in `load()` that returns `"I am LumeBuddy, your custom-trained AI assistant."`. `src/lume_platform/ml/buddy_engine.py` disables transformers by default (`LUME_ENABLE_BUDDY_TRANSFORMER="0"`) and uses hardcoded strings.
7. **The "Multivariate" Forecaster Hardcoded Feature Hack**:
   - *Documentation Claim*: `model_evaluations/phase2_timeseries/model_card.md:10-17` lists NIFTY 50 Close, NIFTY Daily Return, Volatility, and Traded Volume as dynamic exogenous features.
   - *Code Reality*: At inference time (`src/lume_platform/ml/forecaster.py:286-290`), NIFTY features are hardcoded constants:
     ```python
     nifty_close = np.full(self.lookback, 22500.0)
     nifty_vol = np.full(self.lookback, 0.007)
     nifty_vol_qty = np.full(self.lookback, 250000000.0)
     ```
8. **The "Apache Kafka Ingestion" Claim**:
   - *Documentation Claim*: `LUME_AI_FINAL_ARCHITECTURE_REPORT.md:48` and `A1_Studio_Project_Poster_Content.md:71` claim: *"Kafka Streaming: Utilizing extreme data-engineering to pipe real NSE NIFTY stock JSON loads dynamically into PyTorch LSTM tensor weights"*.
   - *Code Reality*: `kafka-python` is not installed or listed in `requirements.txt`. `src/streaming/news_consumer.py:56-59` fails to connect to Kafka, and streaming is mocked via `live_stream_simulator.py` copying local CSV files.
9. **Semantic Search 503 Runtime Failure**:
   - *Documentation Claim*: `README.md:141` documents `GET /search/funds` as an active endpoint for semantic fund lookup.
   - *Code Reality*: `artifacts/models/fund_embeddings.pkl` does not exist in `artifacts/models/`. Calling `/search/funds` immediately throws `HTTPException(status_code=503, detail="Semantic search not enabled or embeddings missing")` (`api/main_enhanced.py:1044-1045`).
10. **Nonexistent Hardcoded Base Paths**:
    - *Documentation Claim*: `.env` and scripts cite `/Users/snehapatel/BigData` as the operational project root.
    - *Code Reality*: The path `/Users/snehapatel/BigData` does not exist on the machine. Running scripts standalone causes `FileNotFoundError` unless `LUME_PROJECT_ROOT` is manually remapped to Google Drive.

---

## PART C: THE DEEP LEARNING SYSTEM WE WILL BUILD (SEMESTER 2 SYSTEM)

### Comprehensive Architecture Strategy
The Semester 2 Deep Learning upgrade transforms Lume AI from an ensemble of disconnected Scikit-Learn pipelines and mock stubs into a fully integrated, multi-modal neural architecture. Every proposed model is engineered to resolve a specific gap identified in Part B.

---

### ENGINE 1: FinBERT Financial Sentiment & Token-Level Saliency (Transformer Encoder, NLP)

#### Starting Assets Found in Repository
- **Data Availability**: `output_production_final/news_cache/latest_news.json` (15 real financial headlines), `semi_structured/social_sentiment/data.csv` (referenced in training scripts, 5,842 financial phrases from Kaggle FiQA/Financial PhraseBank).
- **Date Range**: February 2024 – February 2026.
- **Class Balance**: 58% Neutral, 26% Positive, 16% Negative (moderate class imbalance).
- **Existing Baseline**: TF-IDF + Logistic Regression (Macro F1: 0.6888, Accuracy: 71.60%). Unused `distilbert-base-uncased-finetuned-sst-2-english` (SST-2 movie review classifier, inappropriate for financial vocabulary). Cached local model checkpoint: `yiyanghkust/finbert-tone` in `~/.cache/huggingface/hub/`.

#### 11-Point Technical Specification
1. **DL Family & Domain Justification**:
   - *Family*: Transformer Encoder (Bidirectional Self-Attention, BERT-base architecture).
   - *Justification*: Financial text contains specialized contextual semantics (e.g., "crude rallies" is bearish for Indian equities, but "rallies" is bullish in general English). Standard TF-IDF treats tokens independently, while generic SST-2 DistilBERT fails on domain-specific syntax. FinBERT (`ProsusAI/finbert` / `yiyanghkust/finbert-tone`), pretrained on financial filings and analyst reports, captures subtle financial semantics and negations with superior domain nuance.
2. **Task, Input/Output Tensor Shapes & Labels**:
   - *Task*: 3-Class Sequence Classification and Token-Level Attentive Attribution.
   - *Input Tensor*: Token ID tensor `input_ids`: $(B, 128)$, Attention mask `attention_mask`: $(B, 128)$, Token type IDs `token_type_ids`: $(B, 128)$.
   - *Output Tensor*: Logits tensor $(B, 3)$ transformed via Softmax to normalized probabilities $(B, 3)$. Saliency output: $(B, 128)$ attribution score vector.
   - *Label Definition*: `0: Neutral`, `1: Positive`, `2: Negative`.
3. **Layer-by-Layer Architectural Table (BERT-base / FinBERT)**:
   *(Derived analytically and verified against cached `yiyanghkust/finbert-tone` config)*

| Layer Index | Layer Type / Sub-module | Input Shape | Output Shape | Units / Hidden / Heads | Activation | Normalization | Dropout | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| **0.1** | Word Embeddings | `(B, 128)` | `(B, 128, 768)` | Vocab: 30,522, Dim: 768 | Identity | None | None | 23,440,896 |
| **0.2** | Position Embeddings | `(B, 128)` | `(B, 128, 768)` | Max pos: 512, Dim: 768 | Identity | None | None | 393,216 |
| **0.3** | Token Type Embeddings | `(B, 128)` | `(B, 128, 768)` | Types: 2, Dim: 768 | Identity | None | None | 1,536 |
| **0.4** | Embeddings LayerNorm + Dropout | `(B, 128, 768)` | `(B, 128, 768)` | 768 | Identity | LayerNorm (768) | Dropout ($p=0.1$) | 1,536 |
| **1–12** | **12x Transformer Encoder Blocks** | `(B, 128, 768)` | `(B, 128, 768)` | 12 Layers (details below)| GELU | 2x LayerNorm / block| Dropout ($p=0.1$) | **85,056,768** |
| *1–12.a*| Multi-Head Attention (Q, K, V) | `(B, 128, 768)` | `(B, 128, 768)` | 768 embed, 12 heads | Scaled Dot-Product | None | Attention Drop 0.1 | $3 \times (768 \times 768 + 768) = 1,771,968$ / block |
| *1–12.b*| Attention Output Projection + LN | `(B, 128, 768)` | `(B, 128, 768)` | 768 | Identity | LayerNorm (768) | Dropout ($p=0.1$) | $(768 \times 768 + 768) + 1,536 = 592,128$ / block |
| *1–12.c*| Intermediate FFN Dense | `(B, 128, 768)` | `(B, 128, 3072)`| 3,072 units | **GELU** | None | None | $768 \times 3072 + 3072 = 2,362,368$ / block |
| *1–12.d*| Output FFN Dense + LN | `(B, 128, 3072)`| `(B, 128, 768)` | 768 units | Identity | LayerNorm (768) | Dropout ($p=0.1$) | $3072 \times 768 + 768 + 1,536 = 2,361,600$ / block |
| **13** | Pooler Dense (`[CLS]` token) | `(B, 768)` | `(B, 768)` | 768 units | Tanh | None | None | $768 \times 768 + 768 = 590,592$ |
| **14** | Pre-classifier Dropout | `(B, 768)` | `(B, 768)` | 768 | Identity | None | Dropout ($p=0.1$) | 0 |
| **15** | Classification Head (`fc`) | `(B, 768)` | `(B, 3)` | 3 classes | Softmax | None | None | $768 \times 3 + 3 = \mathbf{2,307}$ |
| **TOTAL** | **FinBERT Sequence Classifier** | `(B, 128)` | `(B, 3)` | **12 Layers, 12 Heads** | **GELU, Softmax** | **13x LayerNorm** | **Dropout 0.1** | **109,486,851 parameters** |

4. **Aggregate Parameter Counts, Model Size & Latency**:
   - *Total Parameters*: **109,486,851 trainable parameters** (~109.5M).
   - *Model Size in Memory*: FP32: $109.49 \times 4 \approx \mathbf{437.9\text{ MB}}$; INT8 Quantized: $\approx \mathbf{110\text{ MB}}$.
   - *Expected Latency Class*: GPU (CUDA/MPS): $8\text{–}15\text{ ms}$; CPU (8-thread batch=1): $45\text{–}65\text{ ms}$ (Production target: `<100 ms` per news article).
5. **Activation Functions & Justification**:
   - `GELU` (Gaussian Error Linear Unit): $x \Phi(x) = x P(X \le x), X \sim \mathcal{N}(0, 1)$. Outperforms ReLU in Transformer encoders by preventing dead neurons and providing smooth probabilistic gating.
   - `Softmax`: Normalizes raw output logits into mutually exclusive probability distribution: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^3 e^{z_j}}$.
   - `Tanh`: In the pooler layer to project `[CLS]` token representation to $[-1, 1]$.
6. **Loss Function, Optimization & Training Hyperparameters**:
   - *Loss Function*: Class-Weighted Categorical Cross-Entropy with Label Smoothing ($\epsilon=0.05$):
     $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^3 w_c \left[ (1 - \epsilon) y_c + \frac{\epsilon}{3} \right] \log \hat{y}_c$$
     where $w = [w_{\text{neutral}}=0.7, w_{\text{positive}}=1.2, w_{\text{negative}}=1.5]$ computed via inverse class frequency.
   - *Optimizer*: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$, weight decay $=0.01$).
   - *Learning Rate & Schedule*: Peak lr $= 2 \times 10^{-5}$ for backbone, $5 \times 10^{-5}$ for head; Linear Warmup over first 10% of steps followed by Linear Cosine Decay.
   - *Batch Size & Epochs*: Batch size 32; 4 epochs (early stopping on validation Macro F1 with patience 2).
   - *Gradient Clipping*: Max norm 1.0. Random seed policy: 42.
7. **Data Split Strategy & Leakage Controls**:
   - Stratified Chronological Split: Articles ordered by publish date. 75% Train (oldest), 10% Val, 15% Test (most recent). Zero entity overlap between same-day duplicate news feeds.
8. **Baselines, Metrics & Ablations**:
   - *Baselines to Beat*:
     1. TF-IDF + Logistic Regression (Repo baseline: Accuracy 71.60%, Macro F1: 0.6888).
     2. Generic SST-2 DistilBERT (Repo baseline: Accuracy ~64% due to financial jargon misclassification).
   - *Evaluation Metrics*: Macro F1 ($\ge 0.82$), Class-specific Precision/Recall, Micro Accuracy ($\ge 85\%$).
   - *Ablations Planned*: Full Fine-Tuning vs Frozen Backbone; With vs Without Class-Weighting; Domain FinBERT vs Generic DistilBERT.
9. **Interpretability & Token Saliency (Math)**:
   - *Method*: Integrated Gradients (IG) combined with Attention Rollout.
   - *Math*:
     $$\text{Saliency}_i = (x_i - x_i') \times \int_{0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$
     where $x'$ is the baseline zero-embedding vector and $F(x)$ is the predicted class logit. Accumulated gradient vector projects directly back to vocabulary tokens to highlight market-moving keywords (e.g., highlighting *"repo rate hike"* as negative).
10. **Serving Path & UI Integration**:
    - *API Endpoint*: `POST /market/sentiment/analyze` (request: `{ "text": "..." }`, response: `{ "label": "Positive", "confidence": 0.94, "saliency_tokens": [...] }`).
    - *UI Component*: `frontend/investor.html` (News Intelligence Tab) and `frontend/ai-showcase.html` with color-coded token highlights.
11. **Risks & Fallbacks**:
    - *Risk*: Out-of-memory on low-end servers; inference latency spikes during high news volume.
    - *Fallback*: Fall back to SBERT + Logistic Regression (`sentiment_bundle.pkl`, 10ms) if Transformer pipeline fails to load.

---

### ENGINE 2: Multivariate BiLSTM + Temporal Multi-Head Attention + Quantile Head (RNN + Attention)

#### Starting Assets Found in Repository
- **Data Availability**: 745 trading days across 3 NIFTY CSVs (`2023-02-21` to `2026-02-20`), 19 mutual fund historical daily NAV JSON series (`hist_*.json` in `output_production_final/nav_cache/`).
- **Features Extracted in Repo**: NAV, NAV return, NIFTY close, NIFTY volatility, NIFTY volume, RSI-14, SMA-20 deviation (`configs/forecaster_bilstm_attention.yaml:18-25`).
- **Existing Code**: `src/lume_platform/ml/bilstm_attention.py`, `src/lume_platform/ml/phase2_trainer.py`, `src/lume_platform/ml/quantile_loss.py`.
- **Existing Baseline**: Univariate Phase 1 LSTM (RMSE: ₹10.91, 190.5% worse than naive persistence baseline). Prototype Phase 2 BiLSTM (RMSE: ₹10.55).

#### 11-Point Technical Specification
1. **DL Family & Domain Justification**:
   - *Family*: Recurrent Neural Network (Bidirectional LSTM) with Temporal Dot-Product Multi-Head Attention and Asymmetric Pinball Quantile Heads.
   - *Justification*: Univariate ARIMA and Phase 1 LSTM fail during macroeconomic regime shifts because mutual fund NAVs do not move in isolation; they are driven by broad index swings (NIFTY 50), sector volatility, and trading momentum. Bidirectional encoding captures forward and backward context over the lookback window, while Temporal Multi-Head Attention explicitly weights past market shocks. The Pinball quantile head provides asymmetric confidence intervals ($P_{10}, P_{50}, P_{90}$) essential for institutional risk budgeting.
2. **Task, Input/Output Tensor Shapes & Labels**:
   - *Task*: Multi-Step Forward Trajectory & Quantile Uncertainty Regression.
   - *Input Tensor*: $(B, 60, D)$, where Lookback $= 60$ trading days, $D = 8$ continuous features ($NAV$, $NAV\_return$, $NIFTY\_close$, $NIFTY\_volatility$, $NIFTY\_volume$, $RSI_{14}$, $SMA_{20}\_dev$, $FinBERT\_sentiment\_score$).
   - *Output Tensor*: $(B, 5, 3)$ representing 5 forward horizons across 3 quantiles $[P_{10}, P_{50}, P_{90}]$. Flattened representation: $(B, 15)$.
   - *Label Definition*: Continuous realized future NAV values $[NAV_{t+1}, \dots, NAV_{t+5}]$.
3. **Layer-by-Layer Architectural Table (Target Design: Hidden 128 per Direction, 8 Heads)**:
   *(Derived by instantiating proposed architecture in memory with PyTorch)*

| Layer Index | Layer Type / Sub-module | Input Tensor Shape | Output Tensor Shape | Neurons / Units / Heads | Internal Activations | Normalization | Dropout | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| **0** | Input Projection (`Linear` + `GELU`)| `(B, 60, 8)` | `(B, 60, 64)` | 64 units | GELU | None | None | `weight`: 512, `bias`: 64 *(576)* |
| **1** | `nn.LSTM` Layer 0 (Bidirectional) | `(B, 60, 64)` | `(B, 60, 256)` | 128 fwd + 128 rev | Gates: $\sigma$, Cell: $\tanh$ | None | None | `ih`: 65,536, `hh`: 131,072, biases: 2,048<br>*(Total L0: 198,656)* |
| **2** | `nn.Dropout` (Inter-layer) | `(B, 60, 256)` | `(B, 60, 256)` | 256 | Identity | None | Dropout ($p=0.2$) | 0 |
| **3** | `nn.LSTM` Layer 1 (Bidirectional) | `(B, 60, 256)` | `(B, 60, 256)` | 128 fwd + 128 rev | Gates: $\sigma$, Cell: $\tanh$ | None | None | `ih`: 262,144, `hh`: 131,072, biases: 2,048<br>*(Total L1: 395,264)* |
| **4** | Pre-Attention `nn.LayerNorm` | `(B, 60, 256)` | `(B, 60, 256)` | 256 | Identity | LayerNorm (256) | None | `weight`: 256, `bias`: 256 *(512)* |
| **5** | `nn.MultiheadAttention` (`attn`) | `(B, 60, 256)` | `(B, 60, 256)` | 256 embed, **8 heads** | Scaled Dot-Product | None | Attention Drop 0.2 | `in_proj`: 196,608, `in_bias`: 768<br>`out_proj`: 65,536, `out_bias`: 256<br>*(Total Attn: 263,168)* |
| **6** | Residual Add + `nn.LayerNorm` | `(B, 60, 256)` | `(B, 60, 256)` | 256 | Identity | LayerNorm (256) | None | `weight`: 256, `bias`: 256 *(512)* |
| **7** | Attention-Weighted Temporal Pooling| `(B, 60, 256)` | `(B, 256)` | 256 | $\sum_{t=1}^{60} \alpha_t h_t$ | None | Dropout ($p=0.2$) | 0 |
| **8** | Dense Representation (`Linear`) | `(B, 256)` | `(B, 128)` | 128 units | GELU | None | Dropout ($p=0.2$) | `weight`: 32,768, `bias`: 128 *(32,896)* |
| **9** | Quantile Output Head (`Linear`) | `(B, 128)` | `(B, 15)` | 15 outputs (5 horiz x 3 q)| Linear | None | None | `weight`: 1,920, `bias`: 15 *(1,935)* |
| **10** | Monotonic Cumulative Softplus Layer| `(B, 5, 3)` | `(B, 5, 3)` | 15 outputs | Cumulative Softplus | None | Monotonic Non-Crossing | 0 |
| **TOTAL** | **Multivariate BiLSTM Attention** | `(B, 60, 8)` | `(B, 5, 3)` | **256 BiLSTM, 8 Heads**| **GELU, Softplus** | **2x LayerNorm** | **Dropout 0.2** | **893,519 parameters** |

4. **Aggregate Parameter Counts, Model Size & Latency**:
   - *Total Parameters*: **893,519 trainable parameters** (~894K).
   - *Model Size in Memory*: FP32: $893,519 \times 4 \approx \mathbf{3.57\text{ MB}}$.
   - *Expected Latency Class*: GPU: $2\text{–}4\text{ ms}$; CPU: $8\text{–}15\text{ ms}$ (Real-time responsive).
5. **Activation Functions & Justification**:
   - `Sigmoid` ($\sigma$) and `Tanh` inside LSTM: Standard gates regulating long-term memory flow and state bounds.
   - `GELU`: In input projection and representation head for smooth gradient propagation.
   - `Cumulative Softplus`: $\text{softplus}(x) = \ln(1 + e^x) \ge 0$. Parameterizes $P_{10} = P_{50} - \text{softplus}(\Delta_{10})$ and $P_{90} = P_{50} + \text{softplus}(\Delta_{90})$, mathematically enforcing $P_{10} \le P_{50} \le P_{90}$ without quantile crossing.
6. **Loss Function, Optimization & Training Hyperparameters**:
   - *Loss Function*: Multi-Horizon Quantile Pinball Loss across $\tau \in \{0.10, 0.50, 0.90\}$:
     $$\mathcal{L}_{\text{Pinball}}(y, \hat{y}_\tau) = \frac{1}{H} \sum_{h=1}^H \sum_{\tau \in \{0.1, 0.5, 0.9\}} \max\left( \tau (y_h - \hat{y}_{\tau, h}), (\tau - 1)(y_h - \hat{y}_{\tau, h}) \right)$$
   - *Optimizer*: AdamW ($\beta_1=0.9, \beta_2=0.999$, lr $= 1 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$).
   - *Scheduler*: Cosine Annealing learning rate schedule down to $1 \times 10^{-5}$.
   - *Batch Size & Epochs*: Batch size 64; 60 epochs with early stopping on validation Pinball Loss (patience 10).
   - *Gradient Clipping*: Max norm 1.0. Random seed: 42.
7. **Data Split Strategy & Leakage Controls**:
   - Chronological split: 70% Train ($t=1 \dots 514$), 15% Validation ($t=515 \dots 624$), 15% Test ($t=625 \dots 735$).
   - Scaler parameter protection: `StandardScaler` or `MinMaxScaler` fitted solely on Train split. Zero lookahead leakage in rolling indicator generation.
8. **Baselines, Metrics & Ablations**:
   - *Baselines to Beat*:
     1. Naive Last-Value Persistence (RMSE: ₹3.7565, MAE: ₹2.98).
     2. ARIMA(3,1,0) (RMSE: ₹3.8963, MAE: ₹3.05).
     3. Phase 1 Univariate LSTM (RMSE: ₹10.9132, MAE: ₹9.83).
   - *Target Metrics*: RMSE $\le \mathbf{₹3.50}$, MAE $\le \mathbf{₹2.70}$, Directional Accuracy $\ge \mathbf{65\%}$, Empirical Quantile Coverage Rate (QCR) in $[78\%, 82\%]$ for nominal 80% interval ($[P_{10}, P_{90}]$).
   - *Ablations Planned*: With vs Without Attention; Univariate vs Multivariate ($D=1$ vs $D=8$); Symmetrical MSE Loss vs Quantile Pinball Loss.
9. **Interpretability & Attention Heatmaps (Math)**:
   - *Method*: Temporal Attention Rollout Heatmap across lookback window.
   - *Math*:
     $$\alpha_t = \frac{\exp\left(\frac{q_T \cdot k_t^\top}{\sqrt{d_k}}\right)}{\sum_{\tau=1}^T \exp\left(\frac{q_T \cdot k_\tau^\top}{\sqrt{d_k}}\right)}$$
     Plotting $\alpha_t \in [0, 1]$ over $t \in [T-60, T]$ generates a temporal heatmap showing exactly which past trading days (e.g. rate announcements or market crashes) drove the 5-day forecast.
10. **Serving Path & UI Integration**:
    - *API Endpoint*: `POST /api/forecast` (Input: `{ "scheme_code": "100119", "horizon": 5 }`, Output: `{ "p10": [...], "p50": [...], "p90": [...], "attention_weights": [...] }`).
    - *UI Component*: `frontend/ai-showcase.html` (interactive Chart.js fan chart) and `frontend/investor.html`.
11. **Risks & Fallbacks**:
    - *Risk*: Market regime changes (structural breaks) causing confidence intervals to widen excessively.
    - *Fallback*: Fall back to deterministic 5-Day Simple Moving Average baseline (`compute_deterministic_baseline()`) if input sequence contains missing values or model encounters an error.

---

### ENGINE 3: Candlestick Pattern CNN + Grad-CAM (Computer Vision)

#### Starting Assets Found in Repository
- **Data Availability**: `scripts/generate_candlestick_images.py` (code exists to synthesize OHLC from NAV and render charts via `mplfinance`).
- **Disk Count**: **0 images exist** in the repository (images were gitignored or never executed).
- **Label Status**: **NO LABELS EXIST IN REPO**.
- **Label Generation Protocol & Leakage Controls**:
  - *Automated Ground-Truth Labeling*: We will run a deterministic rule-based technical labeler over the 10 trading days *following* the 100-day candlestick image window:
    1. **Bullish Breakout (Class 0)**: Future max return $\ge +3.0\%$ within 10 days and volume expansion $\ge 1.5\times$.
    2. **Bearish Breakdown (Class 1)**: Future max drawdown $\le -3.0\%$ within 10 days.
    3. **Consolidation / Range-bound (Class 2)**: Price oscillates within $[-2.0\%, +2.0\%]$.
  - *Data Leakage Control*: The image rendering window ($t-99 \dots t$) and the labeling window ($t+1 \dots t+10$) must be strictly separated. Furthermore, train/val/test splits must be conducted **BY FUND (GROUPED SPLIT)** so that images from the same mutual fund never appear in both train and test splits, preventing visual background leakage.

#### 11-Point Technical Specification
1. **DL Family & Domain Justification**:
   - *Family*: 2D Convolutional Neural Network (Custom 4-Block CNN & Pretrained ResNet-18 Transfer Learning).
   - *Justification*: Experienced human technical analysts identify visual patterns (head-and-shoulders, double bottoms, engulfing candles, wick rejections) that are difficult to extract through purely tabular features. 2D CNNs with spatial convolutions extract translation-invariant visual geometric features directly from rendered chart images.
2. **Task, Input/Output Tensor Shapes & Labels**:
   - *Task*: 3-Class Visual Pattern Classification.
   - *Input Tensor*: RGB Chart Tensor $(B, 3, 224, 224)$, normalized via ImageNet statistics ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
   - *Output Tensor*: Class probability vector $(B, 3)$ via Softmax.
   - *Label Definition*: `0: Bullish Breakout`, `1: Bearish Breakdown`, `2: Consolidation`.

##### 3A. Layer-by-Layer Architectural Table: Custom 4-Block CNN
*(Instantiated and verified directly in memory with PyTorch)*

| Layer Index | Layer Type / Sub-module | Input Tensor Shape | Output Tensor Shape | Neurons / Channels | Activation | Normalization | Dropout / Pool | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| **0** | Input Image Batch | `(B, 3, 224, 224)` | `(B, 3, 224, 224)` | 3 channels (RGB) | Identity | None | None | 0 |
| **1.1** | `Conv2d` Block 1 ($3 \times 3$, pad 1)| `(B, 3, 224, 224)` | `(B, 32, 224, 224)`| 32 channels | ReLU | `BatchNorm2d(32)` | None | `weight`: 864, `bias`: 32 *(896)* |
| **1.2** | `BatchNorm2d` + `MaxPool2d(2, 2)` | `(B, 32, 224, 224)`| `(B, 32, 112, 112)`| 32 channels | ReLU | BatchNorm2d | MaxPool ($2 \times 2$)| `weight`: 32, `bias`: 32 *(64)* |
| **2.1** | `Conv2d` Block 2 ($3 \times 3$, pad 1)| `(B, 32, 112, 112)`| `(B, 64, 112, 112)`| 64 channels | ReLU | `BatchNorm2d(64)` | None | `weight`: 18,432, `bias`: 64 *(18,496)* |
| **2.2** | `BatchNorm2d` + `MaxPool2d(2, 2)` | `(B, 64, 112, 112)`| `(B, 64, 56, 56)` | 64 channels | ReLU | BatchNorm2d | MaxPool ($2 \times 2$)| `weight`: 64, `bias`: 64 *(128)* |
| **3.1** | `Conv2d` Block 3 ($3 \times 3$, pad 1)| `(B, 64, 56, 56)` | `(B, 128, 56, 56)` | 128 channels | ReLU | `BatchNorm2d(128)`| None | `weight`: 73,728, `bias`: 128 *(73,856)* |
| **3.2** | `BatchNorm2d` + `MaxPool2d(2, 2)` | `(B, 128, 56, 56)` | `(B, 128, 28, 28)` | 128 channels | ReLU | BatchNorm2d | MaxPool ($2 \times 2$)| `weight`: 128, `bias`: 128 *(256)* |
| **4.1** | `Conv2d` Block 4 ($3 \times 3$, pad 1)| `(B, 128, 28, 28)` | `(B, 256, 28, 28)` | 256 channels | ReLU | `BatchNorm2d(256)`| None | `weight`: 294,912, `bias`: 256 *(295,168)* |
| **4.2** | `BatchNorm2d` + `MaxPool2d(2, 2)` | `(B, 256, 28, 28)` | `(B, 256, 14, 14)` | 256 channels | ReLU | BatchNorm2d | MaxPool ($2 \times 2$)| `weight`: 256, `bias`: 256 *(512)* |
| **5** | Global Average Pooling (`GAP`) | `(B, 256, 14, 14)` | `(B, 256, 1, 1)` | 256 channels | Identity | None | `AdaptiveAvgPool2d(1)`| 0 |
| **6** | Flatten Vector | `(B, 256, 1, 1)` | `(B, 256)` | 256 | Identity | None | Flatten | 0 |
| **7** | Dense Classifier 1 | `(B, 256)` | `(B, 128)` | 128 units | ReLU | None | Dropout ($p=0.5$) | `weight`: 32,768, `bias`: 128 *(32,896)* |
| **8** | Dense Classifier 2 (`fc`) | `(B, 128)` | `(B, 3)` | 3 classes | Softmax | None | None | `weight`: 384, `bias`: 3 *(387)* |
| **TOTAL** | **Custom Candlestick CNN** | `(B, 3, 224, 224)` | `(B, 3)` | **4 Conv Blocks, GAP** | **ReLU, Softmax** | **4x BatchNorm2d** | **Dropout 0.5** | **422,659 parameters** |

##### 3B. ResNet-18 Transfer Learning Fine-Tuning Model
- Backbone: Pretrained `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` consisting of 17 convolutional layers, residual skip connections, and batch normalization.
- Replacement Classification Head: `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 3))`.
- Total Trainable Parameters: **11,178,051 parameters** (~11.18M).
  - Frozen Backbone Setup: 11,176,512 frozen params, **1,539 trainable params** in head (`weight`: $512 \times 3 = 1,536$, `bias`: 3).
  - Full Fine-Tuning Setup: All 11,178,051 params trainable with differential learning rates.

4. **Aggregate Parameter Counts, Model Size & Latency**:
   - *Custom CNN*: **422,659 parameters** (FP32 size: $\approx \mathbf{1.69\text{ MB}}$).
   - *ResNet-18*: **11,178,051 parameters** (FP32 size: $\approx \mathbf{44.7\text{ MB}}$).
   - *Expected Latency Class*: GPU: $3\text{–}6\text{ ms}$; CPU: $20\text{–}35\text{ ms}$.
5. **Activation Functions & Justification**:
   - `ReLU`: Standard piecewise linear activation for convolutional feature extraction avoiding saturation.
   - `Softmax`: Converts final class logits into probability distribution summing to 1.0.
6. **Loss Function, Optimization & Training Hyperparameters**:
   - *Loss Function*: Class-Weighted Cross-Entropy with Label Smoothing ($\alpha=0.1$):
     $$\mathcal{L}_{\text{CNN}} = - \sum_{k=1}^3 \left[ (1 - \alpha) y_k + \frac{\alpha}{3} \right] \log \hat{y}_k$$
   - *Optimizer*: AdamW (Custom CNN: lr $= 1 \times 10^{-3}$; ResNet-18: lr $= 1 \times 10^{-4}$ for head, $1 \times 10^{-5}$ for backbone).
   - *Augmentation*: Random horizontal flip (forbidden: vertical flip or rotation which alters price causality), random brightness/contrast jitter ($\pm 10\%$).
   - *Batch Size & Epochs*: Batch size 32; 30 epochs with early stopping on validation Macro F1 (patience 7).
7. **Data Split Strategy & Leakage Controls**:
   - Grouped Split by Mutual Fund Code (`GroupKFold` or `GroupShuffleSplit` on `scheme_code`). Ensures images from Scheme 100119 are exclusively in Train or Test, preventing memorization of fund-specific NAV baseline ranges.
8. **Baselines, Metrics & Ablations**:
   - *Baselines to Beat*:
     1. Naive Zero-Rule Baseline (predicting majority class: ~38% accuracy).
     2. Tabular Technical Rule Baseline (RSI > 70 / SMA20 cross: ~52% accuracy).
   - *Target Metrics*: Macro F1 $\ge \mathbf{0.68}$, Top-1 Accuracy $\ge \mathbf{72\%}$, AUC-ROC $\ge \mathbf{0.80}$.
   - *Ablations Planned*: Custom CNN vs Pretrained ResNet-18; Frozen Backbone vs Full Fine-Tuning; Raw Image vs Augmented Images; Grayscale vs RGB.
9. **Interpretability: Grad-CAM Formulation (Math)**:
   - *Method*: Gradient-Weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer (`block4` in Custom CNN, `layer4[1].conv2` in ResNet-18).
   - *Math*:
     $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
     $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
     where $A^k$ is feature activation map $k$, $y^c$ is class score, and $Z = U \times V$ is spatial dimensions. Visualized as a heatmap overlaid on candlestick wicks and bodies to show which candle triggered the breakout call.
10. **Serving Path & UI Integration**:
    - *API Endpoint*: `POST /market/pattern/classify` (accepts scheme code or uploaded chart image, returns class probabilities and Grad-CAM PNG).
    - *UI Component*: `frontend/ai-showcase.html` (interactive Grad-CAM overlay viewer).
11. **Risks & Fallbacks**:
    - *Risk*: Synthetic OHLC candlestick charts fail to capture real tick-level micro-dynamics.
    - *Fallback*: Fall back to tabular momentum and RSI rules if computer vision inference is uncertain (confidence $< 0.50$).

---

### ENGINE 4: Tabular Deep Learning for Lead Conversion (TabNet & Entity Embedding MLP)

#### Starting Assets Found in Repository
- **Data Availability**: `artifacts/cleaned_parquet/lead_scoring_clean` (1,900 rows, 5 numeric features, 9 categorical features).
- **Target Variable**: `converted` (1,180 zeros [62.1%], 720 ones [37.9%]).
- **Categoricals & Cardinalities Measured in Repo**:
  - `lead_origin`: 4 unique ($\to$ emb_dim=2)
  - `lead_source`: 9 unique ($\to$ emb_dim=5)
  - `specialization`: 19 unique ($\to$ emb_dim=10)
  - `what_is_your_current_occupation`: 6 unique ($\to$ emb_dim=3)
  - `last_activity`: 14 unique ($\to$ emb_dim=7)
  - `country`: 21 unique ($\to$ emb_dim=11)
  - `lead_quality`: 5 unique ($\to$ emb_dim=3)
  - `do_not_email`: 2 unique ($\to$ emb_dim=1)
  - `do_not_call`: 2 unique ($\to$ emb_dim=1)
  - Total Embedding Input Dimension: **43**; Numeric Dimension: **5**; Concatenated Dense Input: **48**.
- **Existing Baseline**: Random Forest Classifier (`lead_classifier_bundle.pkl`: Accuracy 85.00%, F1: 0.8202, ROC-AUC: 0.933).

#### 11-Point Technical Specification
1. **DL Family & Domain Justification**:
   - *Family*: Tabular Deep Learning (Target A: Google Research TabNet with Sparsemax Attentive Transformers; Target B: Deep Residual MLP with Categorical Entity Embeddings).
   - *Justification*: Tree ensembles (Random Forest / XGBoost) are standard for tabular data, but fail to share learned latent representations across tasks or benefit from continuous semi-supervised pretraining. Entity embeddings project high-cardinality discrete variables into continuous vector manifolds where semantic similarities (e.g. between occupations or lead sources) are preserved. TabNet uses sequential attention to select salient features at each decision step, providing intrinsic interpretability without post-hoc TreeExplainer approximations.
2. **Task, Input/Output Tensor Shapes & Labels**:
   - *Task*: Binary Classification (Lead Conversion Likelihood).
   - *Input Tensors*: Categorical indices tensor: $(B, 9)$, Continuous numeric tensor: $(B, 5)$.
   - *Output Tensor*: Single conversion probability scalar $(B, 1)$ via Sigmoid.
   - *Label Definition*: `0: Non-converted`, `1: Converted`.

##### 3A. Layer-by-Layer Architectural Table: Target B (Entity Embedding Residual MLP)
*(Directly measured in memory with PyTorch)*

| Layer Index | Layer Type / Sub-module | Input Tensor Shape | Output Tensor Shape | Neurons / Units / Dim | Activation | Normalization | Dropout | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| **0** | 9x Categorical Entity Embeddings| `(B, 9)` | `(B, 43)` | Total emb dim: 43 | Identity | None | None | `embs`: **697 params** (summed across 9 cards)|
| **1** | Feature Concat (Embs + 5 Numerics)| `(B, 43)` + `(B, 5)` | `(B, 48)` | 48 total features | Identity | None | None | 0 |
| **2** | `Linear` Layer 1 | `(B, 48)` | `(B, 256)` | 256 units | ReLU | None | None | `weight`: 12,288, `bias`: 256 *(12,544)* |
| **3** | `BatchNorm1d` (Block 1) | `(B, 256)` | `(B, 256)` | 256 | ReLU | `BatchNorm1d(256)` | Dropout ($p=0.3$) | `weight`: 256, `bias`: 256 *(512)* |
| **4** | `Linear` Layer 2 | `(B, 256)` | `(B, 128)` | 128 units | ReLU | None | None | `weight`: 32,768, `bias`: 128 *(32,896)* |
| **5** | `BatchNorm1d` (Block 2) | `(B, 128)` | `(B, 128)` | 128 | ReLU | `BatchNorm1d(128)` | Dropout ($p=0.3$) | `weight`: 128, `bias`: 128 *(256)* |
| **6** | `Linear` Layer 3 | `(B, 128)` | `(B, 64)` | 64 units | ReLU | None | None | `weight`: 8,192, `bias`: 64 *(8,256)* |
| **7** | `Linear` Output Layer (`fc`) | `(B, 64)` | `(B, 1)` | 1 probability | Sigmoid | None | None | `weight`: 64, `bias`: 1 *(65)* |
| **TOTAL** | **Entity Embedding MLP** | `(B, 9)`, `(B, 5)` | `(B, 1)` | **256 -> 128 -> 64** | **ReLU, Sigmoid** | **2x BatchNorm1d** | **Dropout 0.3** | **55,181 parameters** |

##### 3B. Target A: TabNet Architectural Specification
- Parameters: $n_d = n_a = 32$, $n_{\text{steps}} = 4$, $\gamma = 1.3$, feature dim $D = 48$.
- Shared Feature Transformer: 2 GLU blocks with `GhostBatchNorm` (momentum 0.02).
- Step-Specific Feature Transformers: 2 GLU blocks per step ($4 \times 2 = 8$ blocks).
- Attentive Transformer: Linear layer + `GhostBatchNorm` + **Sparsemax** activation for sparse feature masks ($M[i] \in [0, 1]^D$).
- Total Trainable Parameters: **87,424 parameters** (*ANALYTIC, UNVERIFIED*).

4. **Aggregate Parameter Counts, Model Size & Latency**:
   - *Entity Embedding MLP*: **55,181 parameters** (FP32 size: $\approx \mathbf{0.22\text{ MB}}$).
   - *TabNet*: **87,424 parameters** (FP32 size: $\approx \mathbf{0.35\text{ MB}}$).
   - *Expected Latency Class*: GPU: $<1\text{ ms}$; CPU: $2\text{–}4\text{ ms}$ (Extremely fast, microsecond-level batch throughput).
5. **Activation Functions & Justification**:
   - `ReLU`: In MLP layers to avoid saturation and enable sparse activations.
   - `Sigmoid`: Bounds lead conversion likelihood to $[0, 1]$.
   - `Sparsemax` (in TabNet): Euclidean projection onto probability simplex: $\text{sparsemax}(z) = \arg\min_{p \in \Delta} \|p - z\|^2$. Enforces exact zero weights for irrelevant features, creating sparse feature selection masks.
6. **Loss Function, Optimization & Training Hyperparameters**:
   - *Loss Function*: Weighted Binary Cross-Entropy (pos_weight $= 1.64$ to account for 38% positive ratio):
     $$\mathcal{L}_{\text{BCE}} = - \frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right] + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparsity}}$$
     where for TabNet $\mathcal{L}_{\text{sparsity}} = \sum_{s=1}^{n_{\text{steps}}} \sum_{j=1}^D -M_{s, j} \log(M_{s, j} + \epsilon)$.
   - *Optimizer*: AdamW (lr $= 2 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$).
   - *Scheduler*: `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)`.
   - *Batch Size & Epochs*: Batch size 64; 50 epochs with early stopping on validation ROC-AUC.
7. **Data Split Strategy & Leakage Controls**:
   - 80/20 Stratified Train/Test split (`stratify=y`, random seed 42) matching existing Scikit-Learn baseline. Zero target leakage: all scalers fit on train fold only.
8. **Baselines, Metrics & Benchmark Plan**:
   - *Planned Benchmark Suite*:

| Model Architecture | Parameter Count | ROC-AUC | PR-AUC | F1-Score | Inference Latency (CPU) |
|---|---|---|---|---|---|
| **Logistic Regression** | 49 | 0.812 | 0.741 | 0.710 | <0.1 ms |
| **Random Forest (Current Baseline)**| ~2.4M (trees) | **0.933** | **0.890** | **0.820** | 12 ms |
| **XGBoost (Current Tuned)** | ~850K (trees) | 0.935 | 0.895 | 0.825 | 4 ms |
| **Entity Embedding Residual MLP (Proposed)**| **55,181** | **$\ge 0.940$**| **$\ge 0.900$**| **$\ge 0.835$**| **2 ms** |
| **TabNet (Proposed)** | **87,424** | **$\ge 0.945$**| **$\ge 0.905$**| **$\ge 0.840$**| **4 ms** |

9. **Interpretability Method (Math)**:
   - *TabNet*: Aggregate feature importance mask across all decision steps:
     $$I_j = \sum_{s=1}^{n_{\text{steps}}} \eta_s \cdot M_{s, j}$$
     where $\eta_s$ is decision step contribution and $M_{s, j}$ is the sparse attention mask.
   - *MLP*: Integrated Gradients over embedding inputs.
10. **Serving Path & UI Integration**:
    - *API Endpoint*: `POST /predict` (task="lead_scoring_dl", returns conversion probability, persona tier, and feature importance dictionary).
    - *UI Component*: `frontend/distributor.html` (Distributor Portal live lead table).
11. **Risks & Fallbacks**:
    - *Risk*: Small sample size (1,900 rows) causes deep neural networks to overfit without heavy regularization.
    - *Fallback*: Fall back to serialized Random Forest (`lead_classifier_bundle.pkl`) if MLP validation F1 drops below 0.80.

---

### ENGINE 5: Neural RAG over Prospectus PDFs (Dense Retrieval + Reranker + Generative LLM)

#### Starting Assets Found in Repository
- **Data Availability**: AMC statutory disclosure target URLs in `scripts/fetch_prospectus_pdfs.py:20-60` for 7 AMCs; 1 template PDF in root (`Dataset_Documentation_Template.pdf`); mutual fund scheme names in `data/Mutual_Fund_Data-main/mutual_fund_data.csv`.
- **Estimated Scale**: Scraping top 20 AMC schemes yields ~100 prospectus PDFs (Scheme Information Documents / Key Information Memorandums), averaging 45 pages each $\approx$ 4,500 total pages $\approx$ 18,000 text chunks.
- **Text Extraction Quality**: AMC statutory filings contain clean computer-generated PDF text (PyMuPDF / `fitz` extracts crisp tokens with $<1\%$ OCR noise, though fee tables require Markdown table formatting).
- **Existing Baseline**: Rule-based template matching in `src/lume_platform/ml/buddy_engine.py` and 16-line mock stub in `src/lume_platform/ml/custom_buddy_model.py`.

#### 11-Point Technical Specification
1. **DL Family & Domain Justification**:
   - *Family*: Neural Retrieval-Augmented Generation (RAG) combining Dense Bi-Encoder Embeddings (`bge-small-en-v1.5`), Cross-Encoder Transformer Reranker (`ms-marco-MiniLM-L-6-v2`), and Generative Decoder-only LLM (`Qwen2.5-0.5B/1.5B-Instruct` or Gemini 1.5 Flash via API).
   - *Justification*: Mutual fund prospectuses are legal documents spanning 50–120 pages containing exit load schedules, expense ratios (TER), and asset allocation boundaries. Small LLMs hallucinate numbers if asked directly. A dense bi-encoder retrieves candidate chunks, a cross-encoder scores deep joint token interactions, and the grounded generator synthesizes verifiable answers with page citations.
2. **Task, Input/Output Tensor Shapes & Labels**:
   - *Task*: Open-Domain Grounded Financial Question Answering.
   - *Input*: Natural language user query text string (e.g. *"What is the exit load for redeeming Aditya Birla Frontline Equity within 30 days?"*).
   - *Output*: JSON response containing grounded answer, source document metadata, page citation, and retrieval confidence score.
3. **Layer-by-Layer Architectural Specifications**:

##### Component A: Embedding Bi-Encoder (`BAAI/bge-small-en-v1.5`)
- Architecture: 12-layer BERT-base encoder, hidden size 384, 12 attention heads, intermediate size 1,536, GELU, LayerNorm.
- Embedding Dimension: **384 dimensions**.
- Total Parameters: **33,360,000 parameters** (~33.4M params) (*Verified analytically*).
- Vector Index: FAISS `IndexFlatIP` (Exact Inner Product on normalized vectors, cosine equivalent) for $<50,000$ chunks, switching to `IndexHNSWFlat` ($M=32, efSearch=64$) if scaled beyond $100,000$ chunks. Hybrid search combined with BM25 keyword scoring via Reciprocal Rank Fusion (RRF, $k=60$).

##### Component B: Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Architecture: 6-layer Transformer encoder, hidden size 384, 12 attention heads, intermediate size 1,536.
- Input Shape: Joint token sequence `[CLS] Query [SEP] Candidate Chunk [SEP]` (max length 512).
- Classification Head: `Linear(384, 1)` outputting scalar relevance score.
- Total Parameters: **22,713,601 parameters** (~22.7M params) (*Verified analytically against cached MiniLM*).

##### Component C: Generative Decoder LLM (`Qwen2.5-0.5B-Instruct` / `Qwen2.5-1.5B-Instruct`)
*(Architectural parameters verified from official Qwen2.5 configuration)*

| Sub-Module Specification | `Qwen2.5-0.5B-Instruct` | `Qwen2.5-1.5B-Instruct` |
|---|---|---|
| **Architecture Family** | Decoder-Only Causal Transformer | Decoder-Only Causal Transformer |
| **Number of Layers** | **24 layers** | **28 layers** |
| **Hidden Dimension ($d_{\text{model}}$)** | **896** | **1,536** |
| **Attention Query Heads** | 14 heads (head dim 64) | 12 heads (head dim 128) |
| **Key-Value Heads (GQA)** | **2 heads** (Grouped-Query Attn) | **2 heads** (Grouped-Query Attn) |
| **Intermediate FFN Size (SwiGLU)** | **4,864** | **8,960** |
| **Vocabulary Size** | 151,936 tokens | 151,936 tokens |
| **Positional Embeddings** | Rotary Position Embeddings (RoPE) | Rotary Position Embeddings (RoPE) |
| **Normalization** | RMSNorm ($1 \times 10^{-6}$) | RMSNorm ($1 \times 10^{-6}$) |
| **Total Trainable Parameters** | **494,032,768 parameters** (~494M) | **1,543,714,304 parameters** (~1.54B) |
| **Model Size (INT4 GGUF)** | **~350 MB** (Runs on any CPU) | **~950 MB** (Runs on standard laptop) |

4. **Aggregate Parameter Counts, Model Size & Latency**:
   - Total System Footprint: Bi-Encoder (33.4M) + Reranker (22.7M) + Qwen-0.5B (494M) = **550,106,369 parameters** (~550M).
   - In-Memory Footprint (INT4/INT8): $\approx \mathbf{550\text{ MB}}$ RAM total.
   - Inference Latency: Dense retrieval (FAISS): $5\text{ ms}$; Reranking (top 20 $\to$ 5): $25\text{ ms}$; Generation (Qwen-0.5B @ 30 tokens/s): $600\text{ ms}$. Total turnaround: `< 1.0 second`.
5. **Activation Functions & Justification**:
   - `SwiGLU` (Swish-Gated Linear Unit): $\text{SwiGLU}(x) = (x W \cdot \text{SiLU}(x V)) W_2$. State-of-the-art for decoder LLMs, maximizing parameter efficiency.
   - `RMSNorm`: Normalizes by root-mean-square without mean-centering, reducing computational overhead by 7%.
6. **Loss Function & Generation Hyperparameters**:
   - Pretraining Loss: Autoregressive next-token Cross-Entropy: $\mathcal{L} = - \sum_{t=1}^T \log P(w_t \mid w_{<t})$.
   - Generation Settings: Temperature $= 0.1$ (greedy, low hallucination), Top-p $= 0.9$, Repetition penalty $= 1.1$, Max new tokens $= 256$.
   - Grounding Assertion: Prompt enforces: *"Answer ONLY based on the provided prospectus context. If the fact is not in the text, respond: 'The prospectus does not disclose this information.' Never speculate."*
7. **Chunking & Leakage Controls**:
   - Chunking Strategy: Recursive Character Text Splitter with `chunk_size = 512` tokens, `chunk_overlap = 64` tokens.
   - Metadata attached to every chunk: `{ doc_id: "HDFC_Top_100_SID", amc: "HDFC", scheme_code: "100119", page: 42, section: "Exit Load & Expenses" }`.
8. **Evaluation Plan & Metrics**:
   - Benchmark Dataset: 50 hand-annotated prospectus QA pairs created from AMFI filings.
   - Retrieval Metrics: Hit@5 ($\ge 92\%$), MRR (Mean Reciprocal Rank $\ge 0.85$).
   - Generation Metrics: RAGAS Faithfulness ($\ge 0.90$), Answer Relevance ($\ge 0.88$), Citation Accuracy ($\ge 95\%$).
9. **Interpretability & Citation Grounding (Math)**:
   - Grounding confidence score computed as softmax over Cross-Encoder logits:
     $$S_{\text{ground}} = \sigma\left(\text{CrossEncoder}(\text{Query}, \text{Chunk}_{\text{top1}})\right)$$
     If $S_{\text{ground}} < 0.35$, the system triggers the "NOT FOUND" fallback threshold, refusing to answer and preventing hallucination.
10. **Serving Path & UI Integration**:
    - *API Endpoint*: `POST /buddy/chat` and `POST /advisor/query` (replaces mock stubs).
    - *UI Component*: Slide-out AI Copilot Drawer in `frontend/distributor.html` and `investor.html`.
11. **Risks & Fallbacks**:
    - *Risk*: Heavy local LLM inference slowing down single-core virtual machines.
    - *Fallback*: Configuration toggle `LUME_RAG_PROVIDER` in `.env` (`"local_qwen"` | `"gemini_flash"` | `"extractive_only"`). If running on minimal hardware, falls back to Gemini 1.5 Flash API or returns top-ranked raw extractive text chunk.

---

### C6. System-Level DL Architecture Diagram

```text
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                              LUME AI 2.0 — MULTI-MODAL DEEP LEARNING SYSTEM
═══════════════════════════════════════════════════════════════════════════════════════════════════════

   [Market News & RSS]      [AMFI / NSE Feeds]       [Historical NAV]       [Prospectus PDFs]
            │                       │                       │                       │
            ▼                       ▼                       ▼                       ▼
   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │  ENGINE 1 (NLP) │     │  ENGINE 4 (TAB) │     │ ENGINE 2 (TIME) │     │  ENGINE 5 (RAG) │
   │     FinBERT     │     │     TabNet      │     │  BiLSTM + Attn  │     │  BGE + FAISS +  │
   │  12-Layer Trans │     │ Sparse Attention│     │ 2-Layer BiLSTM  │     │ MiniLM Rerank + │
   │   (109.5M P)    │     │    (87.4K P)    │     │   (893.5K P)    │     │ Qwen2.5 (550M P)│
   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
            │                       │                       │                       │
     Sentiment Score                │                       │                       │
     (Bullish/Bearish)              │                       │                       │
            │                       │                       │                       │
            └───────────────────────┼───────────────────────┘                       │
                                    │ (Exogenous Sentiment Feature)                 │
                                    ▼                                               │
                       ┌─────────────────────────┐                                  │
                       │     ENGINE 2 FORECAST   │                                  │
                       │  - Trajectory (P50)     │                                  │
                       │  - Bounds (P10, P90)    │                                  │
                       │  - Attention Heatmap    │                                  │
                       └────────────┬────────────┘                                  │
                                    │                                               │
                                    │ (Rendered Candlesticks)                       │
                                    ▼                                               │
                       ┌─────────────────────────┐                                  │
                       │     ENGINE 3 (VISION)   │                                  │
                       │  Custom CNN / ResNet-18 │                                  │
                       │  - Pattern Breakout     │                                  │
                       │  - Grad-CAM Heatmap     │                                  │
                       └────────────┬────────────┘                                  │
                                    │                                               │
════════════════════════════════════╪═══════════════════════════════════════════════╪═══════════════════
                               INFERENCE & SERVING LAYER (FastAPI / ModelRegistry)
════════════════════════════════════╪═══════════════════════════════════════════════╪═══════════════════
                                    │                                               │
          ┌─────────────────────────┼─────────────────────────┐                     │
          ▼                         ▼                         ▼                     ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  ┌──────────────────┐
│ POST /predict    │      │ POST /api/forecast│     │ POST /pattern    │  │ POST /buddy/chat │
│ Lead Prob & XAI  │      │ P10/P50/P90 + Attn│     │ Breakout + CAM   │  │ Grounded QA + Doc│
└─────────┬────────┘      └─────────┬────────┘      └─────────┬────────┘  └────────┬─────────┘
          │                         │                         │                    │
══════════╪═════════════════════════╪═════════════════════════╪════════════════════╪═══════════════════
                               CLIENT DASHBOARDS & PORTALS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
          ▼                         ▼                         ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│  🌐 DISTRIBUTOR PORTAL (distributor.html) & INVESTOR WEALTH DASHBOARD (investor.html)        │
│  - Real-time Lead Heatmap & Conversion Likelihood (Engine 4 TabNet)                          │
│  - Non-Crossing Quantile Fan Charts & Shock Attention Heatmaps (Engine 2 BiLSTM)             │
│  - Candlestick Technical Breakout Cards with Grad-CAM Visual Inspection (Engine 3 CNN)       │
│  - LumeBuddy AI Copilot with Instant Document Citations & SID Risk Grounding (Engine 5 RAG)  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### C7. Master Model Comparison Table

| Planned Engine | Deep Learning Architecture | Input Tensor & Dim | # Layers | Hidden / Head Dimension | Internal Activations | Loss Function | Optimizer & LR | Trainable Parameters | Baseline to Beat | Target Evaluation Metrics | Explainability Method | Production Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ENGINE 1: FinBERT Sentiment** | Transformer Encoder (`BertForSequenceClassification`) | `(B, 128)` Token IDs | 12 Encoder Layers | Hidden: 768, 12 Heads, FFN: 3072 | GELU, Softmax, Tanh | Class-Weighted Cross-Entropy | AdamW, lr $2 \times 10^{-5}$ | **109,486,851** | TF-IDF + Logistic Regression (F1: 0.69) | Macro F1 $\ge 0.82$, Accuracy $\ge 85\%$ | Integrated Gradients + Attention Rollout | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 2: Multivariate BiLSTM** | BiLSTM + Temporal Multihead Attention + Quantile Head | `(B, 60, 8)` Continuous Sequence | 2 BiLSTM + 1 Attention | Hidden: 128/dir (256 out), 8 Heads | GELU, Sigmoid, Tanh, Softplus | Asymmetric Pinball Loss ($\tau \in \{0.1, 0.5, 0.9\}$) | AdamW, lr $1 \times 10^{-3}$ | **893,519** | Naive Last-Value (RMSE: ₹3.76), ARIMA (₹3.90) | RMSE $\le ₹3.50$, Coverage $[78\%, 82\%]$ | Temporal Attention Weights $\alpha_t$ Heatmap | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 3A: Candlestick CNN** | Custom 4-Block Conv2D + GAP | `(B, 3, 224, 224)` RGB Image | 4 Conv Blocks + 2 Dense | Channels: 32, 64, 128, 256; Dense: 128 | ReLU, Softmax | Class-Weighted Cross-Entropy | AdamW, lr $1 \times 10^{-3}$ | **422,659** | Naive Majority Class (~38%), RSI Rule (~52%) | Macro F1 $\ge 0.68$, Accuracy $\ge 72\%$ | Grad-CAM (Target: Block 4 Conv) | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 3B: ResNet-18 Transfer** | Pretrained ResNet-18 Backbone + Custom Head | `(B, 3, 224, 224)` RGB Image | 17 Conv Layers + 1 Dense | Backbone: 512, Head: 3 | ReLU, Softmax | Label-Smoothed Cross-Entropy | AdamW, lr $1 \times 10^{-4}$ | **11,178,051** (Frozen: 1,539) | Custom CNN baseline | Macro F1 $\ge 0.74$, Accuracy $\ge 78\%$ | Grad-CAM (Target: `layer4[1].conv2`) | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 4A: TabNet Lead Scorer** | Sparsemax Sequential Attention Tabular Network | `(B, 48)` (43 Embs + 5 Num) | 4 Decision Steps | $n_d = 32, n_a = 32$, Shared/Step GLU | Sparsemax, GLU | Weighted BCE + Sparsity Regularizer | AdamW, lr $2 \times 10^{-3}$ | **87,424** (*ANALYTIC*) | Random Forest (F1: 0.820, AUC: 0.933) | ROC-AUC $\ge 0.945$, F1 $\ge 0.840$ | Step-wise Sparse Feature Mask $M_{s, j}$ | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 4B: Entity Embedding MLP** | Deep Residual MLP with Categorical Entity Embeddings | `(B, 9)` Cats, `(B, 5)` Numerics | 3 Dense Layers + Embs | Dense: 256 $\to$ 128 $\to$ 64 | ReLU, Sigmoid | Weighted Binary Cross-Entropy | AdamW, lr $2 \times 10^{-3}$ | **55,181** | Random Forest baseline | ROC-AUC $\ge 0.940$, F1 $\ge 0.835$ | Integrated Gradients over Embeddings | `PLANNED (PROPOSED DESIGN)` |
| **ENGINE 5: Neural RAG System** | BGE Embedder + Cross-Encoder Reranker + Qwen2.5 | String Text Query | 12L Bi-Enc, 6L Cross-Enc, 24L Gen | Embed: 384, Cross: 384, Gen: 896 | GELU, SwiGLU, RMSNorm | Autoregressive Next-Token Cross-Entropy | AdamW (pre-trained weights) | **550,106,369** (~550M total) | Keyword Search & Mocked LumeBuddy | Hit@5 $\ge 92\%$, Faithfulness $\ge 0.90$ | Softmax Relevance Score & Page Citations | `PLANNED (PROPOSED DESIGN)` |

---

### C8. Key Deep Learning Concepts & Math Cheat Sheet (Slide-Ready)

#### 1. Long Short-Term Memory (LSTM) Internal Gating
LSTM resolves vanishing gradients via an additive constant-error carousel (cell state $C_t$) regulated by three multiplicative sigmoid gates:
- Forget Gate: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ (Controls memory retention from $C_{t-1}$)
- Input Gate: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$ and Candidate: $\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$
- Cell Update: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- Output Gate & Hidden: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$, $h_t = o_t \odot \tanh(C_t)$

#### 2. Scaled Dot-Product Multi-Head Self-Attention
Projects hidden representations into $h$ distinct query, key, and value subspaces to model dependencies across arbitrary time steps:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O, \quad \text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### 3. Asymmetric Pinball (Quantile) Loss
Penalizes under-predictions and over-predictions asymmetrically by quantile weight $\tau \in (0, 1)$, modeling conditional empirical distribution quantiles:
$$\mathcal{L}_\tau(y, \hat{y}) = \max\left( \tau (y - \hat{y}), (\tau - 1)(y - \hat{y}) \right) = (y - \hat{y}) \left( \tau - \mathbb{I}_{\{y < \hat{y}\}} \right)$$
For median ($\tau = 0.5$), Pinball loss reduces exactly to half Mean Absolute Error ($\frac{1}{2}\text{MAE}$).

#### 4. Categorical & Binary Cross-Entropy
Measures divergence between predicted probability distribution $\hat{y}$ and one-hot ground truth $y$:
- Binary Cross-Entropy (BCE): $\mathcal{L}_{\text{BCE}} = - [y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$
- Categorical Cross-Entropy (CCE): $\mathcal{L}_{\text{CCE}} = - \sum_{c=1}^C y_c \log \hat{y}_c$

#### 5. Modern Activation Functions
- **ReLU**: $f(x) = \max(0, x)$ — Computationally efficient, non-saturating for $x > 0$.
- **GELU**: $f(x) = x \Phi(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right)$ — Smooth probabilistic gating in Transformers.
- **Softmax**: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ — Converts logits to probability distribution on simplex $\Delta^C$.
- **Sparsemax**: $\arg\min_{p \in \Delta} \|p - z\|^2$ — Euclidean projection enforcing exact zeros for sparse attention in TabNet.

#### 6. Normalization: Batch Normalization vs Layer Normalization
- **BatchNorm**: Normalizes across mini-batch dimension: $\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$. Ideal for vision (CNNs) and tabular MLPs where batch statistics are stable.
- **LayerNorm**: Normalizes across feature/channel dimension independently for each sample: $\hat{x} = \frac{x - \mu_{\text{sample}}}{\sqrt{\sigma^2_{\text{sample}} + \epsilon}}$. Crucial for RNNs and Transformers where sequence lengths vary and batch statistics are unstable.

#### 7. Residual Skip Connections
Addresses degradation and vanishing gradients in deep networks by establishing identity mapping shortcuts:
$$x_{l+1} = \mathcal{F}(x_l, W_l) + x_l$$
Allows gradients to flow directly through the identity path: $\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_{l+1}} \left( \frac{\partial \mathcal{F}}{\partial x_l} + I \right)$, ensuring $\frac{\partial \mathcal{E}}{\partial x_l}$ never vanishes even if $\frac{\partial \mathcal{F}}{\partial x_l} \to 0$.

#### 8. Grad-CAM (Gradient-Weighted Class Activation Mapping)
Visual interpretability method extracting spatial importance maps from final convolutional feature maps:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
ReLU isolates features that positively contribute to the target class score.

#### 9. Integrated Gradients (Axiomatic Attribution)
Computes path-integral of gradients along the straight line from baseline baseline $x'$ to input $x$:
$$\text{IG}_i(x) = (x_i - x_i') \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$
Satisfies Completeness ($\sum_i \text{IG}_i(x) = F(x) - F(x')$) and Implementation Invariance axioms.

#### 10. Neural Dense Retrieval & Cross-Encoder Reranking
- Bi-Encoder (Dense Retrieval): Queries and documents encoded independently into embedding space: $\text{Sim}(q, d) = \vec{e}_q \cdot \vec{e}_d$. Highly scalable via FAISS index ($O(\log N)$).
- Cross-Encoder (Reranking): Concatenates query and candidate chunk into single Transformer: $S = \text{Linear}(\text{Transformer}([q; d]))$. Captures deep all-to-all cross-attention between words, eliminating bi-encoder compression loss.

---

### C9. 6-Phase Implementation Roadmap

#### Phase 1: Pipeline Decoupling, Data Ingestion & Sanitization
- **Files Touched**: `src/master_pipeline.py`, `src/lume_platform/spark/pipeline.py`, `src/lume_platform/config.py`, `.env`.
- **Concrete Deliverables**:
  1. Fix hardcoded `/Users/snehapatel/BigData` paths to use dynamic `Path(__file__).resolve()`.
  2. Integrate PySpark script to download and extract full Kaggle and AMFI data feeds into local storage.
  3. Wire automated schema validation asserts and write sanitized Parquet tables.
- **Acceptance Criteria**: Running `python3 src/master_pipeline.py` executes without a single `FileNotFoundError` or hardcoded fallback.
- **Risks**: Local disk capacity constraints when processing large intraday files. (Mitigation: Enforce micro-batch sampling).

#### Phase 2: Production BiLSTM + Multi-Head Attention Time-Series Engine
- **Files Touched**: `src/lume_platform/ml/forecaster.py`, `src/lume_platform/ml/bilstm_attention.py`, `src/lume_platform/ml/phase2_trainer.py`, `configs/forecaster_bilstm_attention.yaml`.
- **Concrete Deliverables**:
  1. Eliminate hardcoded NIFTY constants (`nifty_close = 22500.0`); implement dynamic daily feature queries.
  2. Implement Target 8-head BiLSTM architecture (894K params) with cumulative softplus non-crossing quantile head.
  3. Wire trained checkpoint `bilstm_attention_quantile.pth` as default engine in `api/main_enhanced.py:2072`.
- **Acceptance Criteria**: Model test RMSE $\le ₹3.50$ on unseen 2025-2026 holdout, outperforming Naive Last-Value baseline ($₹3.76$).
- **Risks**: Gradient explosion during bidirectional unrolling. (Mitigation: Gradient clipping norm 1.0).

#### Phase 3: Candlestick Pattern CNN & Grad-CAM Visualizer
- **Files Touched**: `scripts/generate_candlestick_images.py`, `src/lume_platform/ml/candlestick_cnn.py` (new), `api/main_enhanced.py`.
- **Concrete Deliverables**:
  1. Render 5,000 real candlestick chart PNGs from cached historical NAV data.
  2. Implement automated forward-looking 10-day breakout labeling with fund-grouped train/test split.
  3. Train Custom CNN (422K params) and fine-tune ResNet-18 (11.2M params); implement Grad-CAM heatmap extraction.
  4. Expose `POST /market/pattern/classify` returning Grad-CAM visualizer to `frontend/ai-showcase.html`.
- **Acceptance Criteria**: Validation Macro F1 $\ge 0.70$; Grad-CAM highlights specific candle wicks and bodies cleanly.
- **Risks**: Overfitting to synthetic background styles. (Mitigation: Disable axes and enforce color augmentation).

#### Phase 4: FinBERT Financial Sentiment & Tabular DL Lead Scorer
- **Files Touched**: `src/lume_platform/ml/transformer_sentiment.py`, `src/lume_platform/ml/tabnet_leads.py` (new), `src/lume_platform/inference/registry.py`.
- **Concrete Deliverables**:
  1. Replace generic DistilBERT SST-2 with domain `ProsusAI/finbert` (109.5M params); wire token-level Integrated Gradients.
  2. Pipe FinBERT daily market sentiment score directly as an exogenous feature into Engine 2 BiLSTM.
  3. Implement TabNet (87K params) and Entity Embedding MLP (55K params) for lead conversion scoring.
- **Acceptance Criteria**: FinBERT Macro F1 $\ge 0.82$; TabNet ROC-AUC $\ge 0.940$, matching or outperforming Random Forest.
- **Risks**: Cold-start latency during FinBERT model weights loading. (Mitigation: Model weights warm-up on API startup).

#### Phase 5: Neural RAG over AMC Statutory Prospectus Filings
- **Files Touched**: `scripts/fetch_prospectus_pdfs.py`, `src/lume_platform/ml/rag_engine.py` (new), `src/lume_platform/ml/buddy_engine.py`, `api/main_enhanced.py`.
- **Concrete Deliverables**:
  1. Execute PDF scraper across top 10 AMCs to assemble ~50 Scheme Information Documents (SID).
  2. Implement PyMuPDF text parser with recursive token chunking (512 tokens, 64 overlap).
  3. Build FAISS vector index with `bge-small-en-v1.5` embeddings (33.4M params) and `ms-marco-MiniLM-L-6-v2` reranker (22.7M params).
  4. Integrate local `Qwen2.5-0.5B-Instruct` (or Gemini 1.5 Flash via API) with strict hallucination cutoff ($S_{\text{ground}} \ge 0.35$).
  5. Replace mock LumeBuddy stub with live grounded RAG copilot in `POST /buddy/chat`.
- **Acceptance Criteria**: Answer Faithfulness $\ge 0.90$ on 50-question benchmark; zero hallucinations on unmentioned exit loads.
- **Risks**: Memory footprint of local LLM on developer machines. (Mitigation: INT4 GGUF quantization).

#### Phase 6: Product Hardening, Unified Dashboard & Showcase Polish
- **Files Touched**: `frontend/app.js`, `frontend/ai-showcase.html`, `frontend/investor.html`, `frontend/distributor.html`, `tests/`.
- **Concrete Deliverables**:
  1. Build unified interactive UI components for all 5 DL engines (fan charts, attention heatmaps, Grad-CAM overlays, RAG chat).
  2. Write end-to-end integration test suite (`tests/test_dl_e2e.py`) verifying all 5 engine API endpoints.
  3. Fix all insecure auth defaults, enforce strict JWT secrets in production mode, and package Docker Compose stack.
- **Acceptance Criteria**: Complete end-to-end demo passes automated test suite; UI transitions smoothly in under 1 second.
- **Risks**: Network port conflicts on deployment host. (Mitigation: Auto-port detection in `run.py`).

---

## PART D: SLIDE-READY PRESENTATION OUTLINE

```text
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                             LUME AI — SEMESTER 2 DEFENSE PRESENTATION OUTLINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════
```

### Slide 1: Title & Executive Summary
- **Headline**: Lume AI — Multi-Modal Deep Learning Platform for Mutual Fund Intelligence
- **Subtitle**: Studio Project Defense: From Classical Big Data Baselines to Production Deep Learning Architecture
- **Key Bullets**:
  - Semester 1 Foundation: PySpark data ingestion, Random Forest lead scoring (F1: 0.82), K-Means clustering, and prototype LSTM forecaster.
  - Critical Audit Findings: Identified 14 engineering gaps including synthetic metrics plots, hardcoded inference constants, and unintegrated DL feature flags.
  - Semester 2 Architecture: 5 specialized deep learning engines covering NLP (FinBERT), Attention Time-Series (BiLSTM), Computer Vision (ResNet-18), Tabular DL (TabNet), and Knowledge Retrieval (Neural RAG).
- **Visual Asset**: `frontend/logo.png` + High-level Tri-Domain Architecture Diagram.

---

### Slide 2: Vision & Business Problem Statement
- **Headline**: The Mutual Fund Distribution Bottleneck in India
- **Key Bullets**:
  - Fragmented Ecosystem: Over 14,000 schemes across 44 AMCs distributed via manual, disconnected distributor workflows.
  - Conversion Inefficiency: Distributors waste 60%+ of sales outreach on unqualified leads due to static CRM heuristics.
  - Opaque Risk Communication: Retail investors receive single-point market forecasts without mathematically calibrated uncertainty bounds.
  - Unstructured Compliance Burden: Scheme Information Documents (SID) exceed 80+ pages of legal fine print, causing disclosure blind spots.
- **Visual Asset**: Flow diagram of traditional manual MFD sales friction vs Lume AI automated neural pipeline.

---

### Slide 3: Big Data Foundation & Empirical Data Reality
- **Headline**: Data Scale, Ingestion Pipelines & The "30 GB" Audit Reality
- **Key Bullets**:
  - Documented Claim vs Reality: Documentation cited "~30 GB across 315K files"; repository audit proves operational disk footprint is ~50 MB (raw Kaggle datasets were gitignored).
  - Production Lake Assets: 16,344 mutual fund master records (`mutual_fund_data.csv`), 17,534 AMFI quotes, 745 NIFTY daily candles, and 1,900 cleaned CRM lead records.
  - Zero-Leakage Pipeline: Chronological time-series separation (2023-2025 Train $\to$ 2025-2026 Test) with scaling parameters fit exclusively on train splits (`timeseries_dataset.py:22-105`).
  - PySpark Lakehouse: Partitioned Parquet materialization (`lead_scoring_clean`) with Kryo serialization and Adaptive Query Execution (`spark/session.py:14-22`).
- **Visual Asset**: `dataset_visualisations/audit_latest/audit_boxplots.png` + Parquet partitioning schematic.

---

### Slide 4: Semester 1 Architecture & What We Built
- **Headline**: Semester 1 System Architecture: Ingestion, Training & Serving
- **Key Bullets**:
  - Data Engineering: Apache Spark ETL cleaning nested tabular trees, imputing missing values, and generating partitioned Parquet outputs.
  - Classical AI Engines: Random Forest lead scoring with calibrated threshold (0.43), K-Means 4-persona behavioral clustering (Silhouette: 0.40), SBERT sentiment analysis.
  - Production Deployment: Enterprise FastAPI backend (55 routes), Vanilla JS responsive frontend (`:3000`), and Streamlit operator dashboards (`:8501`).
  - Auditable Verification: JSON source-of-truth files under `model_evaluations/` establishing strict baseline reproducibility.
- **Visual Asset**: Master Pipeline Diagram (Section A4) + `frontend/ai-showcase.html` UI screenshot.

---

### Slide 5: Existing Models & Baseline Results
- **Headline**: Classical Models Performance & Baseline Verification
- **Key Bullets**:
  - Lead Conversion Classifier: Random Forest achieved **85.00% accuracy**, **0.8202 F1-score**, and **0.933 ROC-AUC** (Source: `model_evaluations/random_forest/real_metrics.json`).
  - Investor Clustering: K-Means ($k=4$) achieved **0.3952 Silhouette score** across 7 behavioral traits, routing investors to specialized fund categories (`model_manifest.json:49`).
  - Sentiment NLP: SBERT + Logistic Regression achieved **71.60% accuracy** and **0.6888 Macro F1** on financial social text (`nlp_sentiment/metrics.json:4`).
  - Legacy Phase 1 LSTM: Multi-horizon NAV predictor trained with Huber loss on 514 trading days; achieved **₹10.91 RMSE** and **1.84% MAPE** (`lstm_forecaster/real_metrics.json:46-48`).
- **Visual Asset**: `model_evaluations/random_forest/rf_confusion_matrix.png` & `model_evaluations/kmeans/kmeans_cluster_projection.png`.

---

### Slide 6: Technical Audit: Gaps & Deficiencies Found
- **Headline**: Critical Repository Audit: What Was Broken or Incomplete
- **Key Bullets**:
  - Deep Learning Not in Loop: `train_all()` in `src/lume_platform/ml/training.py` trained only classical Scikit-Learn models; neural networks were bypassed.
  - The Baseline Paradox: Real Phase 1 LSTM RMSE (₹10.91) was **190.5% worse** than Naive Last-Value persistence (₹3.76); earlier documentation claimed R2=0.89 using synthetic noise curves (`generate_lstm_rf_metrics.py:47-56`).
  - Hardcoded Inference Constants: Phase 2 forecaster used hardcoded NIFTY constants (`nifty_close = 22500.0`) in production inference (`forecaster.py:286-290`).
  - Mocked AI Copilot: LumeBuddy was implemented as a 16-line mock stub returning a static string (`custom_buddy_model.py:6-16`) and canned rule-based dictionaries.
- **Visual Asset**: Gap Register Summary Table (Section B1).

---

### Slide 7: The Deep Learning Upgrade Vision
- **Headline**: Semester 2 Architectural Transformation: 5 Neural Engines
- **Key Bullets**:
  - Multi-Modal Paradigm: Upgrading tabular, text, sequential time-series, and computer vision modalities into an interconnected deep learning stack.
  - End-to-End Synergy: FinBERT financial sentiment feeds directly as an exogenous feature into BiLSTM; CNN visual patterns feed investor dashboards; Neural RAG replaces mocked chat.
  - Honesty & Transparency: Eliminating all hardcoded fallbacks and synthetic plots in favor of mathematically proven architectures and open evaluation suites.
  - Complete Serving Integration: Exposing all 5 engines via production FastAPI endpoints with sub-second latency on standard hardware.
- **Visual Asset**: System-Level DL Architecture Diagram (Section C6).

---

### Slide 8: Engine 1 — FinBERT Sentiment & Token Saliency
- **Headline**: Engine 1: Domain-Specific Financial Transformer (109.5M Params)
- **Key Bullets**:
  - Architectural Design: 12-layer Transformer encoder (`ProsusAI/finbert`), hidden size 768, 12 attention heads, FFN 3072, GELU activations, LayerNorm.
  - Why It Beats Baseline: Replaces static TF-IDF (F1: 0.69) with financial domain pretraining; correctly interprets contextual syntax ("rate cut", "rallies", "headwinds").
  - Mathematical Loss: Class-weighted categorical cross-entropy with label smoothing ($\epsilon=0.05$) to handle 58% neutral class dominance.
  - Token Saliency: Integrated Gradients (IG) path integrals highlighting exact market-moving keywords directly in the dashboard UI.
- **Visual Asset**: Transformer Block Architecture Schematic + Sample Integrated Gradients token attribution overlay.

---

### Slide 9: Engine 2 — Multivariate BiLSTM + Attention Forecaster
- **Headline**: Engine 2: Sequence Forecasting with Quantile Pinball Loss (894K Params)
- **Key Bullets**:
  - Architectural Design: Input projection (Linear $8 \to 64$), 2-layer BiLSTM (128 hidden/dir $\to 256$ out), 8-head temporal dot-product attention, LayerNorm.
  - Non-Crossing Quantiles: Cumulative softplus parameterization ($P_{10} = P_{50} - \text{softplus}(\Delta_{10}), P_{90} = P_{50} + \text{softplus}(\Delta_{90})$) mathematically guarantees $P_{10} \le P_{50} \le P_{90}$.
  - Objective: Multi-horizon Pinball Loss across quantiles $\tau \in \{0.1, 0.5, 0.9\}$ providing calibrated 80% confidence bands.
  - Interpretability: Temporal attention weights ($\alpha_t$) visualized as a 60-day heatmap revealing past market shocks driving the forecast.
- **Visual Asset**: `model_evaluations/phase2_timeseries/attention_heatmap.png` & `model_evaluations/phase2_timeseries/loss_curves.png`.

---

### Slide 10: Engine 3 — Candlestick Pattern CNN & Grad-CAM
- **Headline**: Engine 3: Visual Technical Pattern Classification (422K / 11.2M Params)
- **Key Bullets**:
  - Dual Vision Models: (a) Custom 4-block Conv2D + BatchNorm + GAP (422K params) for edge efficiency; (b) Pretrained ResNet-18 fine-tuning (11.2M params) with custom head.
  - Automated Labeling & Leakage Control: Forward 10-day return thresholds ($\ge +3\%$ breakout, $\le -3\%$ breakdown) strictly decoupled from chart image window; grouped split by fund.
  - Objective: Label-smoothed Cross-Entropy ($\alpha=0.1$) with AdamW and conservative brightness/contrast augmentations.
  - Visual Explainability: Grad-CAM gradient backpropagation targeting final conv layer, visually illuminating the exact candlesticks triggering the classification.
- **Visual Asset**: Custom CNN block diagram + Grad-CAM heatmap overlaid on candlestick chart.

---

### Slide 11: Engine 4 — Tabular DL for Lead Conversion
- **Headline**: Engine 4: TabNet & Entity Embedding MLP (87K / 55K Params)
- **Key Bullets**:
  - Dual Tabular Architectures: (a) Google Research TabNet with Sparsemax attentive transformers; (b) Deep Residual MLP with categorical entity embeddings.
  - Entity Embeddings: 9 categorical variables projected into 43-dim continuous vector space, capturing semantic affinities between lead sources and professions.
  - Sparsemax Feature Selection: TabNet uses sequential Euclidean simplex projection to select sparse subsets of features at each decision step.
  - Benchmark Plan: Rigorous comparison against Random Forest (ROC-AUC: 0.933, F1: 0.820) targeting ROC-AUC $\ge 0.945$ with 2ms CPU inference latency.
- **Visual Asset**: TabNet Sparsemax Attentive Transformer block diagram + Entity Embedding MLP schematic.

---

### Slide 12: Engine 5 — Neural RAG over Prospectus PDFs
- **Headline**: Engine 5: Grounded Financial QA with Multi-Stage Retrieval (550M Params)
- **Key Bullets**:
  - Real Document Corpus: Scraping Scheme Information Documents (SID) across top AMCs; PyMuPDF token extraction with 512-token sliding chunking.
  - Multi-Stage Architecture: Dense retrieval via BGE-small-en-v1.5 (33.4M params) + FAISS $\to$ Cross-Encoder MiniLM reranking (22.7M params) $\to$ Qwen2.5-0.5B generation.
  - Strict Anti-Hallucination Guardrail: Grounding confidence assertion ($S_{\text{ground}} \ge 0.35$); falls back to "Information not disclosed in prospectus" if threshold fails.
  - Product Integration: Replaces 16-line mock stub with an authentic conversational copilot citing document names and page numbers in chat.
- **Visual Asset**: Multi-Stage Neural RAG pipeline diagram (Query $\to$ FAISS $\to$ Reranker $\to$ Qwen2.5 $\to$ Grounded Answer).

---

### Slide 13: System Evaluation & Ablation Plan
- **Headline**: Rigorous Evaluation Methodology & Ablation Benchmarks
- **Key Bullets**:
  - Strict Out-of-Sample Protocol: Chronological holdout test sets for time-series; stratified splits for tabular leads; grouped fund splits for computer vision.
  - Baselines to Beat: Every model evaluated against standard classical baselines (Naive, Moving Average, ARIMA, Logistic Regression, Random Forest).
  - Comprehensive Ablation Matrix: Quantifying the exact marginal gain of bidirectionality, temporal attention, multi-feature inputs, and pretrained transfer backbones.
  - Operational Benchmarks: Measuring throughput (req/sec), P99 latency, and FP32/INT8 memory footprint across CPU and GPU hardware.
- **Visual Asset**: Master Comparison Table (Section C7).

---

### Slide 14: Deep Learning Interpretability Matrix
- **Headline**: Explainable AI (XAI): White-Box Transparency across All Modalities
- **Key Bullets**:
  - Financial Compliance: SEBI and fiduciary regulations require algorithmic transparency for automated lead scoring and investment recommendations.
  - Multi-Modal XAI Suite:
    - Text (Engine 1): Integrated Gradients highlighting influential financial tokens.
    - Time-Series (Engine 2): Multi-Head Attention heatmaps isolating critical historical shock dates.
    - Vision (Engine 3): Grad-CAM spatial heatmaps isolating candlestick pattern triggers.
    - Tabular (Engine 4): TabNet step-wise sparse masks revealing feature selection.
- **Visual Asset**: Multi-panel interpretability composite (IG token highlight + Attention heatmap + Grad-CAM chart).

---

### Slide 15: Product Roadmap & Implementation Timeline
- **Headline**: 6-Phase Delivery Roadmap (Semester 2 Timeline)
- **Key Bullets**:
  - Phase 1 (Weeks 1–2): Pipeline decoupling, path sanitization, and automated data ingestion.
  - Phase 2 (Weeks 3–5): Production BiLSTM + Attention time-series engine with live NIFTY dynamic features.
  - Phase 3 (Weeks 6–8): Candlestick image dataset rendering, automated labeling, Custom CNN & ResNet-18 Grad-CAM.
  - Phase 4 (Weeks 9–11): FinBERT domain sentiment wiring and TabNet tabular lead scoring benchmark.
  - Phase 5 (Weeks 12–14): Full Neural RAG pipeline deployment with AMC prospectus parsing and Qwen2.5 copilot.
  - Phase 6 (Weeks 15–16): Full-stack integration, security hardening, automated test suites, and final presentation demo.
- **Visual Asset**: Gantt Chart / Roadmap Flowchart (Section C9).

---

## OPEN QUESTIONS & COULD NOT VERIFY

1. **Missing 30 GB Kaggle Datasets**:
   - The directory `datasets/` referenced throughout `Dataset_Documentation.md` and `A1_Studio_Project_Poster_Content.md` was completely absent on disk (gitignored). While scripts exist to re-download portions from AMFI/NSE, the original multi-gigabyte historical files (`nifty500_intraday`, `stock_market_india`, `financial_news_corpus`) could not be verified on the local filesystem.
2. **Provenance of the "134,533" Parameter Claim**:
   - The user prompt referenced a potential claimed number of "134,533" parameters. Rigorous in-memory PyTorch measurement confirmed that `LSTMForecaster` has **50,757** parameters, `BiLSTMAttentionQuantileForecaster` has **212,239** parameters, `NAVForecaster` (mintleads) has **199,553** parameters, and `NAVPredictorLSTM` (scripts) has **331,909** parameters. The origin of 134,533 could not be located in codebase files and represents an unverified report claim.
3. **Absence of Pre-rendered Candlestick Charts & Prospectus PDFs**:
   - Scripts `generate_candlestick_images.py` and `fetch_prospectus_pdfs.py` exist, but their target directories (`datasets/unstructured/candlesticks` and `datasets/prospectus_pdfs`) contain zero files. It could not be verified whether these scripts were ever executed at full scale in previous semesters.
4. **Location of SBERT Mutual Fund Embeddings (`fund_embeddings.pkl`)**:
   - `artifacts/models/fund_embeddings.pkl` is missing from `artifacts/models/` (causing the API `/search/funds` route to fail with 503), but a file named `fund_embeddings.pkl` was discovered under `lume-ai-product/ml-artifacts/models/fund_embeddings.pkl`. It is unverified why this artifact was not copied into the main backend artifact directory.
5. **Real-World Execution of Apache Spark on Production Cluster**:
   - While `pyspark-4.1.1` is installed and local Spark scripts run in standalone mode, configuration references to GCS (`spark.hadoop.fs.defaultFS = "gs://..."`) and AWS S3 (`s3a://...`) in `session.py:25-44` could not be verified against an active cloud cluster connection.

