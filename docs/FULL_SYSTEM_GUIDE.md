# Lume AI: Full System Architecture, Execution & Presentation Guide
**From-Scratch Engineering Blueprint, Setup Manual, and Viva Defense Handbook**

---

## 📌 Executive Summary & Project Purpose

**Lume AI** is an enterprise-grade fintech intelligence platform designed to bridge the gap between Big Data engineering, advanced financial analytics, and modern Deep Learning. The platform serves two primary stakeholders in the asset management ecosystem:

1. **Mutual Fund Distributors (MFDs) & Wealth Managers:**
   - **Lead Conversion & Client Retention:** Automated scoring of prospective investors using Machine Learning (Random Forest / TabNet) to prioritize high-value leads.
   - **Persona Clustering:** Unsupervised segmentation (K-Means & DBSCAN) mapping investor psychographics into actionable allocation strategies (Growth Seeker, Wealth Preserver, Balanced Allocator, Index Tracker).
   - **Regulatory Compliance & Portfolio Validation:** Automated checks ensuring recommended schemes match client risk profiles and AMFI guidelines.

2. **Retail Investors & Wealth Creators:**
   - **Forward Predictive Navigation:** 5-day multi-step forward NAV forecasting with non-crossing uncertainty intervals ($P_{10}, P_{50}, P_{90}$) powered by a **Multivariate BiLSTM with Temporal Multi-Head Attention**.
   - **Computer Vision Chart Intelligence:** Recognition of technical chart breakouts and breakdowns using **Custom 4-Block CNNs** and **Fine-Tuned ResNet-18** evaluated on 40-day candlestick charts, explainable via **Native PyTorch Grad-CAM**.
   - **Market Sentiment & RAG Intelligence:** FinBERT-driven sentiment scoring on financial news feeds coupled with a Hybrid Retrieval-Augmented Generation (RAG) assistant for mutual fund statutory prospectuses (SIDs/KIMs).

---

## 🗺️ Repository Structure & Component Map

The repository is organized into distinct layers separating raw ingestion, PySpark Big Data transformations, PyTorch Deep Learning modules, FastAPI web services, and Vanilla JS / CSS3 client interfaces:

```text
BigData/ (Repository Root)
├── run.py / run.sh              # Unified full-stack launcher (Backend :8000 + Frontend :3000)
├── Makefile                     # Build, test, and training automation targets
├── CompleteNavReport.txt         # Raw AMFI NAV report (17,534 records, 14,400+ schemes)
├── EQUITY_L.csv                  # NSE listed equity master list (2,241 stocks)
├── NIFTY 50-*.csv (3 files)      # Benchmark NIFTY 50 daily OHLCV trading calendar (745 trading days)
│
├── api/                          # FastAPI Service Layer
│   ├── main.py                   # Legacy minimal router
│   └── main_enhanced.py          # Enterprise API (62 routes, JWT Auth, Model Registry, Swagger Docs)
│
├── artifacts/                    # Serialized Models, Scalers, and Big Data Assets
│   ├── cleaned_parquet/          # Spark-materialized Parquet datasets (partitioned by target)
│   ├── ml_scalers/               # MinMax and StandardScaler objects for inference
│   ├── models/                   # Serialized PyTorch checkpoints & Scikit-Learn pipelines
│   └── vision_candlesticks/      # 2,208 generated $224 \times 224$ candlestick charts + manifest
│
├── frontend/                     # High-Performance Vanilla JS / CSS3 Client (Zero npm overhead)
│   ├── index.html                # Platform Landing Page & Navigation Hub
│   ├── investor.html             # Investor Wealth Portal (Forecasting, Risk Profiles, Fund Explorer)
│   ├── distributor.html          # MFD CRM Portal (Lead Scoring Pipeline, Client Retention)
│   ├── ai-showcase.html          # Deep Learning Visualizer (BiLSTM Attention Heatmap & Grad-CAM)
│   ├── bigdata-showcase.html     # Interactive PySpark Big Data Architecture Walkthrough
│   ├── app.js                    # Core application logic, API client, survey state engine
│   └── styles.css                # Custom fintech dark/light glassmorphic styling
│
├── model_evaluations/            # Verification & Audit Source-of-Truth
│   ├── phase2_timeseries/        # BiLSTM + Attention model card, metrics JSON, loss curves, heatmap
│   ├── phase3_vision/            # Custom CNN & ResNet-18 model card, metrics JSON, Grad-CAM overlays
│   ├── random_forest/            # Lead scoring metrics, ROC-AUC curve, confusion matrix
│   └── kmeans/                   # Silhouette analysis, PCA cluster projections
│
├── src/                          # Production Python Source Root (`lume_platform`)
│   ├── master_pipeline.py        # Master Spark ETL -> ML training pipeline orchestrator
│   └── lume_platform/
│       ├── cli.py                # Unified CLI interface (`lume train`, `lume forecast`)
│       ├── auth.py               # JWT generation, token verification, PBKDF2 password hashing
│       ├── forecaster/           # Engine 2: BiLSTM + Attention + Pinball Quantile Forecaster
│       ├── vision/               # Engine 3: Candlestick CNN, ResNet-18, and Native Grad-CAM
│       └── crm/                  # Lead scoring, K-Means clustering, and MFD dashboard services
│
└── tests/                        # Comprehensive Pytest Verification Suite
    ├── test_phase2_bilstm.py     # BiLSTM architecture, tensor shapes, and quantile bounds
    ├── test_phase3_vision.py     # Vision CNN, ResNet-18, Grad-CAM hooks, and vision API
    ├── test_api_forecast.py      # End-to-end HTTP integration tests for `/predict/forecast`
    └── test_data_leakage.py      # Strict group-split and chronological verification tests
```

---

## ⚡ Quickstart: How to Run the Entire Project (From Scratch)

You do **not** need complex node package installations or Docker setups to run the primary platform. The system is designed to run with a single command launching both the backend REST API and the frontend web application concurrently.

### Step 1: Environment Preparation
Ensure Python 3.10+ is installed on your machine. In your terminal, navigate to the repository root:

```bash
cd "/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData"
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate       # On macOS / Linux
# .\venv\Scripts\activate      # On Windows
```

Install core dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Set Environment Variables
Set the Python path and Matplotlib temporary configuration cache:
```bash
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/mplconfig
```

### Step 3: Launch Frontend & Backend Concurrently
Execute the unified full-stack runner:
```bash
python3 run.py --force
# Or using the Makefile:
make run
```

#### What `run.py` Does Automatically:
1. **Checks Port Availability:** Inspects ports `8000` (Backend) and `3000` (Frontend). Passing `--force` automatically frees up any stale background processes occupying those ports.
2. **Starts Backend Server:** Launches `api.main_enhanced:app` via Uvicorn on `http://localhost:8000`. Loads the model registry (Random Forest, K-Means, BiLSTM Forecaster, Vision Engine) into memory.
3. **Starts Frontend Server:** Spawns an asynchronous Python HTTP server on `http://localhost:3000` serving the `frontend/` directory.
4. **Graceful Shutdown:** Pressing `Ctrl+C` cleanly terminates both processes simultaneously.

---

## 🌐 Navigating the Web Application & Interactive Demos

Once `run.py` is running, open your web browser and explore the platform modules:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        LUME AI PLATFORM DASHBOARD                          │
├───────────────────────────────┬────────────────────────────────────────────┤
│ URL                           │ Purpose & Features                         │
├───────────────────────────────┼────────────────────────────────────────────┤
│ http://localhost:3000/index.html        │ 🚀 Platform Landing Page & Architecture Hub │
│ http://localhost:3000/investor.html     │ 📈 Investor Portal: Forecasts & Risk Cones │
│ http://localhost:3000/distributor.html  │ 💼 MFD CRM: Lead Scoring & Client Retention│
│ http://localhost:3000/ai-showcase.html  │ 🧠 DL Showcase: Attention Heatmaps & Grad-CAM│
│ http://localhost:3000/bigdata-showcase.html │ ⚡ Big Data: PySpark Ingestion Architecture │
│ http://localhost:8000/docs              │ 📖 Interactive FastAPI Swagger Documentation│
└───────────────────────────────┴────────────────────────────────────────────┘
```

### 1. Landing Page (`/index.html`)
- Displays an overview of Lume AI, real-time API health status pinging `GET /health`, and navigational links to investor, distributor, and deep learning showcase pages.

### 2. Investor Portal (`/investor.html`)
- **Psychographic Risk Profiling:** 7-question interactive survey mapping user responses into 4 investor personas (`Growth Seeker`, `Wealth Preserver`, `Balanced Allocator`, `Index Tracker`).
- **Scheme Explorer & Historical Charting:** Interactive multi-year NAV charting across 19 flagship AMFI mutual funds with volume and benchmark comparisons.
- **Deep Learning Forward Forecaster:** Click **"Generate 5-Day Forecast"** to call `POST /predict/forecast`. The UI visualizes:
  - Historical NAV curve (solid line).
  - 5-Day Median Expected Trajectory ($P_{50}$, dashed line).
  - Shaded Uncertainty Cone bounded strictly between $P_{10}$ (lower band) and $P_{90}$ (upper band).
  - Mandatory AMFI disclaimer: *"Educational and analytical tool only. Not investment advice."*

### 3. Distributor CRM Portal (`/distributor.html`)
- **Lead Qualification Pipeline:** Evaluates incoming client leads based on annual income, SIP tenure, risk appetite, and past equity experience.
- **Machine Learning Conversion Probability:** Instant scoring via Random Forest Classifier providing a conversion probability score (0% to 100%) and a categorical recommendation (`HOT LEAD`, `WARM LEAD`, `COLD LEAD`).
- **Client Retention Alerts:** Monitors churn probability and flags dormant accounts before SIP discontinuation.

### 4. Deep Learning Showcase (`/ai-showcase.html`)
- **Temporal Attention Saliency:** Renders the 30-day multi-head attention weight distribution $\alpha_t$, highlighting macroeconomic turning points and volatility spikes that drove the forecast.
- **Candlestick Pattern Recognition & Grad-CAM Overlay:** Displays a 40-day candlestick chart alongside the model's prediction (`Bullish Breakout`, `Bearish Breakdown`, or `Consolidation`), class confidence, and the visual Grad-CAM heatmap overlay showing exactly which candle wicks influenced the decision.

### 5. Interactive Swagger API Documentation (`http://localhost:8000/docs`)
- Allows interactive testing of all 62 REST endpoints directly from your browser.

---

## 🏗️ Deep Learning Engines: Technical Architecture & Mathematics

```
                               ┌──────────────────────────────────────────────────────────┐
                               │                    LUME AI DEEP LEARNING                 │
                               │                         SYSTEM ZOO                       │
                               └──────────────────────────────────────────────────────────┘
                                                             │
                  ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                  │                                          │                                          │
                  ▼                                          ▼                                          ▼
     ┌─────────────────────────────┐            ┌─────────────────────────────┐            ┌─────────────────────────────┐
     │          ENGINE 1           │            │          ENGINE 2           │            │          ENGINE 3           │
     │      FinBERT Sentiment      │            │   Multivariate BiLSTM with  │            │     Candlestick Pattern     │
     │      & Token Saliency       │            │   Temporal Attention        │            │     CNN & Grad-CAM          │
     ├─────────────────────────────┤            ├─────────────────────────────┤            ├─────────────────────────────┤
     │ • Model: ProsusAI/finbert   │            │ • Input: (B, 30, 8)         │            │ • Custom 4-Block CNN (422K) │
     │ • Params: ~110 Million      │            │ • Encoder: 2-Layer BiLSTM   │            │ • Fine-Tuned ResNet-18 (11M)│
     │ • Saliency: Integrated      │            │ • Attention: 4-Head Self-Att│            │ • Native PyTorch Hooks      │
     │   Gradients + Rollout       │            │ • Loss: Pinball (P10,50,90) │            │ • Zero In-Image Leakage     │
     │ • Endpoint: /api/nlp/...    │            │ • Endpoint: /predict/fore.. │            │ • Endpoint: /api/vision/..  │
     └─────────────────────────────┘            └─────────────────────────────┘            └─────────────────────────────┘
```

---

### Engine 2: Multivariate BiLSTM + Temporal Multi-Head Attention Forecaster

- **File Path:** [`src/lume_platform/forecaster/bilstm_attention.py`](file:///Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My%20Drive/BigData/src/lume_platform/forecaster/bilstm_attention.py)
- **Model Checkpoint:** `artifacts/models/bilstm_attention_quantile.pth` (~240K parameters)
- **Input Tensor Shape:** $(B, 30, 8)$ — Batch size $B$, 30-day historical lookback window, 8 multivariate features:
  1. $R_{\text{NAV}}$: Fund daily NAV return: $\frac{\text{NAV}_t - \text{NAV}_{t-1}}{\text{NAV}_{t-1}}$
  2. $P_{\text{NIFTY}}$: NIFTY 50 Benchmark closing price
  3. $R_{\text{NIFTY}}$: NIFTY 50 daily return
  4. $\sigma_{10}$: NIFTY 50 10-day rolling realized volatility
  5. $V_{\text{NIFTY}}$: NIFTY 50 traded daily volume
  6. $\text{RSI}_{14}$: 14-day Relative Strength Index of the fund NAV
  7. $\Delta_{\text{SMA20}}$: 20-day Simple Moving Average deviation: $\frac{\text{NAV}_t - \text{SMA}_{20}}{\text{SMA}_{20}}$
  8. $S_{\text{macro}}$: Financial sentiment score derived from news headlines

#### Network Architecture:
1. **Input Projection:** `Linear(8 -> 64)` $\to$ `GELU()` $\to$ `LayerNorm(64)`
2. **Bidirectional LSTM Encoder:** 2-layer BiLSTM, hidden dimension $H = 64$ per direction. Output shape: $(B, 30, 128)$.
3. **Temporal Multi-Head Attention:** `nn.MultiheadAttention(embed_dim=128, num_heads=4)` applied across the 30-day time axis. Computes self-attention query-key-value projections, capturing long-range market inflection points.
4. **Residual & Normalization:** Residual add + `LayerNorm(128)` followed by attention-weighted temporal pooling.
5. **Non-Crossing Quantile Head:**
   - Multi-layer perceptron: `Linear(128 -> 64)` $\to$ `GELU()` $\to$ `Dropout(0.2)`.
   - Projects to 5 forecast horizons ($H=5$) across 3 quantiles ($q \in \{0.10, 0.50, 0.90\}$).
   - **Monotonic Non-Crossing Constraint:**
     $$\hat{y}_{50} = \text{Dense}_{\text{median}}(h)$$
     $$\hat{y}_{10} = \hat{y}_{50} - \text{softplus}(\Delta_{10})$$
     $$\hat{y}_{90} = \hat{y}_{50} + \text{softplus}(\Delta_{90})$$
     *Mathematical Guarantee:* Because $\text{softplus}(x) > 0$ for all $x \in \mathbb{R}$, this parameterization ensures $P_{10} \le P_{50} \le P_{90}$ under all market conditions, completely eliminating quantile crossing errors.

#### Multi-Task Pinball Loss:
$$\mathcal{L}_{\text{Pinball}}(y, \hat{y}, q) = \frac{1}{N} \sum_{i=1}^N \max\left(q(y_i - \hat{y}_{i,q}), (1 - q)(\hat{y}_{i,q} - y_i)\right)$$
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Pinball}}(y, \hat{y}_{10}, 0.10) + \mathcal{L}_{\text{Pinball}}(y, \hat{y}_{50}, 0.50) + \mathcal{L}_{\text{Pinball}}(y, \hat{y}_{90}, 0.90)$$

---

### Engine 3: Candlestick Pattern Engine & Native Grad-CAM

- **File Path:** [`src/lume_platform/vision/`](file:///Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My%20Drive/BigData/src/lume_platform/vision/)
- **Model Checkpoints:**
  - `artifacts/models/candlestick_cnn_custom.pth` (422,659 params, 1.7 MB)
  - `artifacts/models/candlestick_resnet18_finetuned.pth` (11,178,051 params, 44.8 MB)

#### Future-Trajectory Labeling Scheme (Zero In-Image Leakage):
Labels are generated strictly from the forward 10-day price movement following the chart cutoff date:
- $R_{10} = \frac{P_{t+10} - P_t}{P_t}, \quad R_{\max} = \frac{\max_{k \in [1, 10]} P_{t+k} - P_t}{P_t}, \quad R_{\min} = \frac{\min_{k \in [1, 10]} P_{t+k} - P_t}{P_t}$
1. **Bullish Breakout (Class 0):** $R_{10} \ge +0.60\%$ AND $R_{\max} \ge +0.80\%$
2. **Bearish Breakdown (Class 1):** $R_{10} \le -0.20\%$ AND $R_{\min} \le -0.40\%$
3. **Consolidation (Class 2):** All other windows (range-bound)

#### Strict Fund-Level Group Split (No Visual Cross-Contamination):
- **Train (13 Funds, 1,547 images):** `100119`, `100375`, `100682`, `100868`, `101993`, `103340`, `104685`, `108272`, `112117`, `115943`, `119551`, `119552`, `119553`
- **Validation (3 Funds, 357 images):** `119771`, `133565`, `144333`
- **Holdout Test (3 Funds, 304 images):** `145034`, `147567`, `149132`

#### Native Grad-CAM from Scratch:
Implemented using forward and backward PyTorch hooks without third-party wrapper dependencies:
1. Feature maps $A^k \in \mathbb{R}^{U \times V}$ captured via forward hook on target conv layer.
2. Backpropagated gradients $\frac{\partial Y^c}{\partial A^k}$ captured via backward hook.
3. Channel importance weights computed via Global Average Pooling:
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. Saliency heatmap generated via positive ReLU combination:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. Scaled via min-max normalization, blended with a Turbo colormap at opacity $\alpha = 0.55$, and exported as a base64 string.

#### Measured Test Set Performance Comparison:
| Model Architecture | Parameters | Test Accuracy | Macro-F1 | Weighted-F1 | Bearish F1 | Breakdown F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule-Based Baseline (RSI+SMA)** | — | 80.59% | **0.3159** | 0.7208 | 0.0541 | 0.0000 |
| **Custom 4-Block CNN** | 422,659 | **81.25%** | **0.4204** | 0.7537 | 0.0000 | 0.3590 |
| **Fine-Tuned ResNet-18** | 11,178,051 | 80.59% | **0.4857** | **0.7806** | **0.1702** | **0.3774** |

*Key Result:* Fine-tuned ResNet-18 delivers a **+53.7% relative gain in Macro-F1** over the rule-based baseline, proving transfer learning extracts meaningful predictive candlestick patterns.

---

## 💻 API Reference: Key Endpoints & JSON Payloads

### 1. Health Check
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-11-02T11:14:36.123456",
    "version": "2.1.0",
    "models_loaded": {
      "leads": true,
      "forecaster": true,
      "vision": true
    }
  }
  ```

### 2. Multi-Step NAV Forecast with Quantile Bounds
- **Endpoint:** `POST /predict/forecast`
- **Request Body:**
  ```json
  {
    "scheme_code": "100119",
    "history": [152.1, 152.4, 153.0, 152.8, 153.5, 154.2],
    "horizon_days": 5
  }
  ```
- **Response Body:**
  ```json
  {
    "scheme_code": "100119",
    "historical_last": 154.2,
    "forecast_trajectory": [154.45, 154.72, 154.91, 155.15, 155.38],
    "lower_bound": [152.80, 152.95, 153.05, 153.12, 153.20],
    "upper_bound": [156.10, 156.48, 156.78, 157.18, 157.56],
    "model_source": "bilstm_attention",
    "accuracy_metric": "Empirical Coverage: 100.0% (P10-P90)",
    "disclaimer": "Educational and analytical tool only. Not investment advice."
  }
  ```

### 3. Candlestick Pattern Recognition & Grad-CAM Saliency
- **Endpoint:** `POST /api/vision/pattern`
- **Request Body:**
  ```json
  {
    "scheme_code": "100119",
    "lookback_days": 40
  }
  ```
- **Response Body:**
  ```json
  {
    "timestamp": "2026-11-02T11:15:00.123456",
    "pattern": "Bearish Breakdown",
    "pattern_class": 1,
    "confidence": 0.7421,
    "probabilities": {
      "Bullish Breakout": 0.1215,
      "Bearish Breakdown": 0.7421,
      "Consolidation": 0.1364
    },
    "gradcam_overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
    "model_architecture": "custom_cnn",
    "scheme_code": "100119",
    "latest_nav": 154.23
  }
  ```

### 4. Lead Scoring & Conversion Probability
- **Endpoint:** `POST /predict/lead`
- **Request Body:**
  ```json
  {
    "annual_income": 1800000,
    "sip_amount": 25000,
    "age": 34,
    "risk_appetite": "High",
    "past_equity_exp_years": 5
  }
  ```
- **Response Body:**
  ```json
  {
    "conversion_probability": 0.884,
    "classification": "HOT LEAD",
    "priority_level": "High Priority",
    "top_contributing_features": [
      {"feature": "sip_amount", "importance": 0.32},
      {"feature": "past_equity_exp_years", "importance": 0.28}
    ]
  }
  ```

---

## 🧪 Automated Testing & Verification Commands

All models and API endpoints are covered by comprehensive automated tests. Run them using `pytest` or the provided `Makefile`:

```bash
# Run all unit and integration tests across the platform
make test
# Or directly:
pytest tests/ -v

# Run Phase 3 Vision Engine + Native Grad-CAM tests
make test-phase3
# Tests: Custom CNN shape/params, ResNet-18 shape, hook execution,
#        CAM normalization [0, 1], base64 encoding, and /api/vision/pattern HTTP 200.

# Run Phase 2 BiLSTM + Attention tests
make test-phase2
# Tests: Tensor projection, BiLSTM encoding, Multi-Head Attention,
#        monotonic non-crossing bounds (P10 <= P50 <= P90), and Pinball loss.

# Run Data Leakage Audits
make leakage-test
# Tests: Group split by fund, chronological forward splits, no future data in features.
```

---

## 👥 3-Student Work Division & Viva Presentation Playbook

To ensure equal, academically rigorous distribution for the presentation and project viva, the project is divided into three distinct roles:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    STUDENT RESPONSIBILITY MATRIX                           │
├─────────────┬──────────────────────────┬───────────────────────────────────┤
│ Student     │ Primary Domain           │ Core Modules & DL Engines         │
├─────────────┼──────────────────────────┼───────────────────────────────────┤
│ Student 1   │ Data Engineering &       │ • PySpark ETL & Snappy Parquet    │
│             │ Time-Series DL           │ • Micro-Batch Streaming Simulator │
│             │                          │ • Engine 2: BiLSTM + Attention    │
│             │                          │ • Pinball Loss & Quantile Bounds  │
├─────────────┼──────────────────────────┼───────────────────────────────────┤
│ Student 2   │ Computer Vision, ML &    │ • Random Forest & K-Means CRM     │
│             │ Full-Stack Architecture  │ • Engine 3: Candlestick CNN &     │
│             │                          │   Fine-Tuned ResNet-18            │
│             │                          │ • Native Grad-CAM from Scratch    │
│             │                          │ • FastAPI Backend (62 Routes)     │
│             │                          │ • Vanilla JS Web App Client       │
├─────────────┼──────────────────────────┼───────────────────────────────────┤
│ Student 3   │ Financial NLP, Tabular   │ • TF-IDF & SBERT Fund Search      │
│             │ DL & RAG Intelligence    │ • Engine 1: FinBERT Sentiment     │
│             │                          │ • Token Saliency via Rollout      │
│             │                          │ • Engine 4/5: TabNet & Prospectus │
│             │                          │   Hybrid RAG (BGE + BM25)         │
└─────────────┴──────────────────────────┴───────────────────────────────────┘
```

### Presentation Flow & Slide Breakdown (18–20 Minutes)

#### Slides 1–2 (Joint): Introduction & Problem Statement
- **Speaker:** Student 1 or Joint
- **Key Points:** The dual problem in mutual fund distribution: retail investors need forward risk bounds rather than misleading single-point forecasts; MFDs need intelligent lead conversion and persona segmentation.

#### Slides 3–6: Big Data Architecture & Time-Series Deep Learning
- **Speaker:** **Student 1**
- **Key Points:**
  - PySpark ETL pipeline ingesting 17,534 raw records, handling schema validation, and partitioning into snappy Parquet.
  - Streaming simulation injecting live micro-batches every 5 seconds.
  - Limitations of last semester's univariate LSTM (134,533 params, single-point drift, no macro features).
  - Architecture of Engine 2: 8 multivariate features, 2-layer BiLSTM, 4-head Temporal Self-Attention.
  - Mathematical derivation of Pinball Loss and the Softplus non-crossing quantile head guaranteeing $P_{10} \le P_{50} \le P_{90}$.
  - Demo: Temporal attention heatmap identifying market volatility inflection points.

#### Slides 7–11: Classical ML, Computer Vision & Full-Stack Platform
- **Speaker:** **Student 2**
- **Key Points:**
  - Random Forest lead classifier (83.33% accuracy, ROC-AUC 0.9167) and K-Means persona clustering.
  - Design of the Candlestick Computer Vision dataset: 2,208 images ($224 \times 224$), labeled strictly on future 10-day forward returns ($H=10$) to eliminate in-image visual leakage.
  - Enforcing group splits by fund (13 train funds vs 3 test funds) to measure true out-of-distribution visual generalization.
  - Custom 4-block CNN (422K params) vs Fine-Tuned ResNet-18 (11.18M params) yielding a **+53.7% gain in Macro-F1** over technical rule-based heuristics.
  - Implementation of Native PyTorch Grad-CAM hooks, channel weights $\alpha_k^c$, and colormap blending.
  - Full-stack architecture: FastAPI (62 routes) + Vanilla JS / CSS3 web frontend on port 3000.

#### Slides 12–15: Financial NLP, Tabular Deep Learning & Prospectus RAG
- **Speaker:** **Student 3**
- **Key Points:**
  - Semantic search using SBERT (`all-MiniLM-L6-v2`) and TF-IDF across 14,400+ scheme descriptions.
  - Engine 1: FinBERT sentiment classifier (~110M params) fine-tuned on financial phrasing with token-level saliency via attention rollout.
  - Engine 4: TabNet architecture using Sparsemax sequential decision steps for non-linear lead scoring.
  - Engine 5: Statutory Prospectus RAG combining BM25 keyword matching with dense BGE-Large embeddings and a Cross-Encoder reranker.

#### Slides 16–18 (Joint): Live System Walkthrough & Conclusion
- **Speaker:** Joint / Student 2
- **Demo:**
  1. Open `http://localhost:3000/investor.html` $\to$ show 5-day BiLSTM forecast with uncertainty bands.
  2. Open `http://localhost:3000/ai-showcase.html` $\to$ show Grad-CAM overlay highlighting candlestick wick rejections.
  3. Open `http://localhost:8000/docs` $\to$ execute live query on `/api/vision/pattern`.

---

## ❓ Anticipated Viva Questions & Technical Answers

### For Student 1:
- **Q: Why is directional accuracy often near ~50% in financial time series while RMSE is low?**
  * **Answer:** Asset returns in efficient markets follow near-martingale properties where the conditional expectation of tomorrow's return is close to zero. Directional sign classification on daily noise is inherently difficult; however, estimating volatility and quantile dispersion ($P_{10}$ to $P_{90}$) is statistically tractable and provides institutional risk bounds.
- **Q: How does Multi-Head Attention help over a simple BiLSTM?**
  * **Answer:** An RNN compresses historical sequence information sequentially into a hidden state vector, which can suffer from bottlenecking over longer windows. Self-attention provides direct shortcut paths between any time step $t_i$ and $t_j$, allowing the model to attend directly to macro shock events regardless of how many trading days have elapsed.

### For Student 2:
- **Q: Why did you split your vision dataset by fund rather than random image sampling?**
  * **Answer:** If we randomly split overlapping 40-day windows from the same mutual fund across train and test sets, consecutive sliding windows would share 39 days of identical visual data. Splitting by fund scheme code ensures the test set contains completely unseen funds, proving the network learns invariant technical patterns rather than memorizing fund-specific trajectories.
- **Q: What is the purpose of the ReLU in the Grad-CAM formula?**
  * **Answer:** We compute $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$. We apply ReLU because we are interested only in features that have a positive influence on the target class score $Y^c$. Negative gradients correspond to pixels that suppress the class, which would degrade the visual explanation.

### For Student 3:
- **Q: Why use Sparsemax instead of Softmax in TabNet's feature selection masks?**
  * **Answer:** Softmax computes exponentiated normalizations, meaning every single feature receives a non-zero probability even if its true predictive importance is negligible. Sparsemax projects activations onto the probability simplex using an Euclidean projection, allowing weights to hit exact zeros and providing truly sparse, explainable feature selection.
- **Q: Why is a Cross-Encoder needed after dense vector retrieval in RAG?**
  * **Answer:** Bi-encoders (like BGE or Sentence-BERT) encode queries and documents into independent vector embeddings for fast similarity search; however, they cannot model token-level cross-attention interactions between the query and the document. A Cross-Encoder feeds the query and candidate passage into the transformer simultaneously, yielding substantially higher reranking accuracy on nuanced statutory disclaimers.

---

## 🛠️ Troubleshooting & Frequently Asked Questions

1. **Port Already in Use Error:**
   - *Issue:* `Address already in use: 8000` or `3000`.
   - *Fix:* Run `python3 run.py --force`, which executes `lsof -ti :8000 :3000 | xargs kill -9` to automatically clear stale servers.

2. **Matplotlib Font Cache Permission Warning:**
   - *Issue:* Sandbox permission denied writing to `~/.matplotlib`.
   - *Fix:* Ensure `export MPLCONFIGDIR=/tmp/mplconfig` is executed before running scripts.

3. **ModuleNotFoundError: No module named 'lume_platform':**
   - *Issue:* Python cannot locate internal packages when running standalone scripts.
   - *Fix:* Set `export PYTHONPATH=src` or run via `python3 -m lume_platform.cli`.

---

*Lume AI — Built for the Advanced Deep Learning & Big Data Systems Evaluation.*

