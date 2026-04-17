# MintLeads

**AI-Powered Lead Recommendation System for Mutual Fund Distributors**

MintLeads is a production-ready AI dashboard that scores leads, segments investors, forecasts NAV trends, and gauges live market sentiment - all designed specifically for Mutual Fund Distributors (MFDs) and Independent Financial Advisors (IFAs).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         MintLeads                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Streamlit  │    │    Flask     │    │    MLflow    │    │
│  │  Dashboard   │◄──►│    REST API  │◄──►│   Tracking   │    │
│  │   (Port      │    │   (Port      │    │   (Port      │    │
│  │    8501)     │    │    5000)     │    │    5001)     │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Data Pipelines                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │   │
│  │  │   ingest    │  │  preprocess │  │  train_*         │  │   │
│  │  │  (AMFI,     │  │  (KNN,      │  │  (RF, KMeans,   │  │   │
│  │  │   NSE, RSS) │  │   Scaling)  │  │   LSTM)         │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Data Layer                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │   raw/   │ │processed/│ │velocity/ │ │  Redis   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **🔥 Hot Leads**: AI-powered lead scoring using Random Forest
- **📈 NAV Predictor**: LSTM-based mutual fund NAV forecasting
- **👥 Investor Profiles**: K-Means clustering for investor segmentation
- **📊 Sentiment Gauge**: Real-time market sentiment analysis

## Prerequisites

- **Python 3.10+
- **Docker 20.10+
- **30 GB of available data storage
- **Redis** (optional, for caching)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
cd mintleads

# Copy environment template
cp .env.example .env

# Start all services
docker-compose up -d

# Access the services
# Dashboard: http://localhost:8501
# API: http://localhost:5000
# MLflow: http://localhost:5001
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Run preprocessing
python -m pipelines.preprocess

# Train models
python -m pipelines.train_lead_scorer
python -m pipelines.train_cluster
python -m pipelines.train_lstm

# Start API
python -m api.app

# Start Dashboard (in another terminal)
streamlit run dashboard/app.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_HOST` | Flask API host | `0.0.0.0` |
| `FLASK_PORT` | Flask API port | `5000` |
| `STREAMLIT_PORT` | Dashboard port | `8501` |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `MLFLOW_TRACKING_URI` | MLflow URL | `http://localhost:5001` |
| `AMFI_NAV_URL` | AMFI NAV feed URL | `https://www.amfiindia.com/spages/NAVAll.txt` |
| `DATASETS_BASE` | Path to datasets | `/path/to/datasets` |

## API Documentation

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models": {
    "lead_scorer": true,
    "investor_cluster": true,
    "nav_forecaster": true
  }
}
```

### Score Leads

```bash
curl -X POST http://localhost:5000/api/leads/score \
  -H "Content-Type: application/json" \
  -d '{
    "leads": [
      {
        "Prospect ID": "LEAD001",
        "TotalVisits": 10,
        "Total Time Spent on Website": 500,
        "Page Views Per Visit": 3
      }
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "lead_id": "LEAD001",
      "conversion_probability": 0.92,
      "tier": "hot"
    }
  ]
}
```

### Segment Investors

```bash
curl -X POST http://localhost:5000/api/investors/segment \
  -H "Content-Type: application/json" \
  -d '{
    "investors": [
      {
        "id": "INV001",
        "current_age": 35,
        "yearly_income": 75000,
        "debt_to_income_ratio": 0.27,
        "credit_score": 720
      }
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "investor_id": "INV001",
      "cluster_id": 1,
      "persona": "Balanced",
      "recommended_fund_type": "Hybrid"
    }
  ]
}
```

### Forecast NAV

```bash
curl "http://localhost:5000/api/nav/forecast?scheme_code=SCHEME001&days=30"
```

**Response:**
```json
{
  "scheme_code": "SCHEME001",
  "scheme_name": "Sample MF Scheme",
  "historical": [...],
  "forecast": [
    {"date": "2024-02-01", "predicted_nav": 105.32},
    ...
  ]
}
```

### Get Sentiment

```bash
curl http://localhost:5000/api/sentiment/current
```

**Response:**
```json
{
  "signal": "Bullish",
  "score": 0.65,
  "top_headlines": [...],
  "last_updated": "2024-01-01T12:00:00"
}
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## MLflow UI

Access the MLflow experiment tracking UI at: **http://localhost:5001**

View all training runs, metrics, and model artifacts.

## Project Structure

```
mintleads/
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned, encoded data
│   └── velocity/             # Live streaming data
├── models/
│   ├── lead_scorer/          # Random Forest artifacts
│   ├── investor_cluster/     # K-Means artifacts
│   └── nav_forecaster/       # LSTM/PyTorch artifacts
├── pipelines/
│   ├── ingest.py             # Data ingestion
│   ├── preprocess.py         # Feature engineering
│   ├── train_lead_scorer.py  # RF training
│   ├── train_cluster.py      # K-Means training
│   ├── train_lstm.py         # LSTM training
│   └── evaluate.py           # Model evaluation
├── api/
│   ├── app.py                # Flask REST API
│   └── routes/
│       ├── leads.py
│       ├── clusters.py
│       └── nav.py
├── dashboard/
│   ├── app.py                # Streamlit entry
│   └── pages/
│       ├── hot_leads.py
│       ├── nav_predictor.py
│       ├── investor_profiles.py
│       └── sentiment_gauge.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
└── README.md
```

## Known Limitations

1. **NSE API**: May require session cookies for authenticated access
2. **Sentiment Analysis**: Currently uses rule-based scoring; NLP model can be added
3. **LSTM Training**: Requires GPU for optimal performance on large datasets

## Roadmap

- [ ] Add NLP-based sentiment analysis
- [ ] Implement real-time WebSocket updates
- [ ] Add user authentication and RBAC
- [ ] Support for more data sources (BSE, international markets)
- [ ] Mobile app companion

## License

MIT License - See LICENSE file for details

## Support

For issues and feature requests, please open a GitHub issue.

---

**Built with ❤️ for Mutual Fund Distributors**
