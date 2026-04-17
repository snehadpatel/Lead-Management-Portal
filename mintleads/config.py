"""MintLeads configuration module.

This module provides centralized configuration management for the MintLeads
AI-Powered Lead Recommendation System. All paths and settings are loaded from
environment variables with sensible defaults.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VELOCITY_DATA_DIR = DATA_DIR / "velocity"
MODELS_DIR = BASE_DIR / "models"

# Model subdirectories
LEAD_SCORER_DIR = MODELS_DIR / "lead_scorer"
INVESTOR_CLUSTER_DIR = MODELS_DIR / "investor_cluster"
NAV_FORECASTER_DIR = MODELS_DIR / "nav_forecaster"

# API Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# Dashboard Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mintleads")

# External API Endpoints
AMFI_NAV_URL = os.getenv("AMFI_NAV_URL", "https://www.amfiindia.com/spages/NAVAll.txt")
NSE_MARKET_STATUS_URL = os.getenv("NSE_MARKET_STATUS_URL", "https://www.nseindia.com/api/marketStatus")
NSE_HISTORICAL_URL = os.getenv("NSE_HISTORICAL_URL", "https://www.nseindia.com/api/historical/cm/equity")

# RSS Feeds
MONEYCONTROL_RSS = os.getenv("MONEYCONTROL_RSS", "https://www.moneycontrol.com/rss/MCtopnews.xml")
ECONOMIC_TIMES_RSS = os.getenv("ECONOMIC_TIMES_RSS", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")

# Data paths for existing datasets
DATASETS_BASE = Path(os.getenv("DATASETS_BASE", "/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/BigData/datasets"))
LEADS_DATA_PATH = Path(os.getenv("LEADS_DATA_PATH", DATASETS_BASE / "structured/leads/lead_scoring/Lead Scoring.csv"))
INVESTOR_BEHAVIOR_PATH = Path(os.getenv("INVESTOR_BEHAVIOR_PATH", DATASETS_BASE / "models/ml_features/investor_profiles_scaled.csv"))
NAV_HISTORY_DIR = Path(os.getenv("NAV_HISTORY_DIR", DATASETS_BASE / "structured/mutual_funds/nav_history"))
AMFI_SCHEME_LIST_PATH = Path(os.getenv("AMFI_SCHEME_LIST_PATH", DATASETS_BASE / "structured/mutual_funds/amfi_scheme_list.csv"))
NEWS_STREAM_PATH = Path(os.getenv("NEWS_STREAM_PATH", DATASETS_BASE / "velocity/live_news_stream.csv"))
NSE_PULSE_PATH = Path(os.getenv("NSE_PULSE_PATH", DATASETS_BASE / "velocity/nse_market_pulse.csv"))

# Model Training Configuration
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

# LSTM Configuration
LSTM_WINDOW_SIZE = int(os.getenv("LSTM_WINDOW_SIZE", "60"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "100"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "32"))
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
LSTM_DROPOUT = float(os.getenv("LSTM_DROPOUT", "0.2"))

# Lead Scoring Thresholds
HOT_LEAD_THRESHOLD = float(os.getenv("HOT_LEAD_THRESHOLD", "0.85"))
WARM_LEAD_THRESHOLD = float(os.getenv("WARM_LEAD_THRESHOLD", "0.65"))

# Cache TTL (seconds)
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Set up logging with the configured level and format.
    
    Args:
        name: Logger name. If None, returns the root logger.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger


def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    dirs_to_create = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VELOCITY_DATA_DIR,
        LEAD_SCORER_DIR,
        INVESTOR_CLUSTER_DIR,
        NAV_FORECASTER_DIR,
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on module load
ensure_directories()
