# Lume AI — Development & CI Makefile

PYTHON ?= python3
PORT ?= 8000
STREAMLIT_PORT ?= 8501

.PHONY: help train train-lstm train-all test leakage-test api dashboard clean

help:
	@echo "Available commands:"
	@echo "  make train-lstm    Train PyTorch LSTM forecaster on real NAV data"
	@echo "  make train-all     Train all models (Leads, Clustering, NLP, LSTM)"
	@echo "  make test          Run pytest suite"
	@echo "  make leakage-test  Run data leakage unit tests"
	@echo "  make api           Start FastAPI backend"
	@echo "  make dashboard     Start Streamlit dashboard"
	@echo "  make clean         Clean cache and temp files"

train-lstm:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli train --engine lstm

train-all:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli train --engine all

test:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

leakage-test:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m pytest tests/test_data_leakage.py -v

forecast:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli forecast --scheme 100119

api:
	uvicorn api.main_enhanced:app --host 0.0.0.0 --port $(PORT) --reload

dashboard:
	PYTHONPATH=src streamlit run streamlit_app.py --server.port $(STREAMLIT_PORT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
