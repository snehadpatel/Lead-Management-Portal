# Lume AI — Development & CI Makefile

PYTHON ?= python3
PORT ?= 8000
STREAMLIT_PORT ?= 8501

.PHONY: help train train-lstm train-all test leakage-test api dashboard run start clean

help:
	@echo "Available commands:"
	@echo "  make run           Launch both frontend (3000) and backend (8000) together"
	@echo "  make train-bilstm  Train PyTorch BiLSTM + Attention quantile forecaster"
	@echo "  make train-lstm    Train legacy PyTorch LSTM forecaster on NAV data"
	@echo "  make train-all     Train all models (Leads, Clustering, NLP, BiLSTM)"
	@echo "  make test          Run pytest suite"
	@echo "  make test-phase2   Run Phase 2 BiLSTM + Attention test suite"
	@echo "  make leakage-test  Run data leakage unit tests"
	@echo "  make api           Start FastAPI backend only"
	@echo "  make dashboard     Start Streamlit dashboard"
	@echo "  make clean         Clean cache and temp files"

train-bilstm:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli train --engine bilstm

train-lstm:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli train --engine lstm

train-all:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli train --engine all

test:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

test-phase2:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m pytest tests/test_phase2_bilstm.py -v

leakage-test:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m pytest tests/test_data_leakage.py -v

forecast:
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src $(PYTHON) -m lume_platform.cli forecast --scheme 100119

api:
	uvicorn api.main_enhanced:app --host 0.0.0.0 --port $(PORT) --reload

run:
	$(PYTHON) run.py

start: run

dashboard:
	PYTHONPATH=src streamlit run streamlit_app.py --server.port $(STREAMLIT_PORT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
