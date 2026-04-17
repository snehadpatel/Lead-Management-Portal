#!/bin/bash
# Lume AI — Production Startup Script
# Starts API server and Streamlit dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              🚀 LUME AI PLATFORM STARTER                  ║"
echo "║         Apache Spark + AI/ML + FastAPI + Streamlit        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠ No virtual environment found. Using system Python${NC}"
fi

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Loading environment variables from .env${NC}"
    set -a
    source .env
    set +a
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p artifacts/models
mkdir -p artifacts/cleaned_parquet
mkdir -p model_evaluations
mkdir -p output_production_final
mkdir -p tableau_exports
echo -e "${GREEN}✓ Directories created${NC}"

# Check if models exist
if [ ! -f "artifacts/models/lead_classifier.pkl" ]; then
    echo -e "${YELLOW}⚠ Models not found. Training required.${NC}"
    echo "Run: PYTHONPATH=src python -m lume_platform.ml.training"
    
    read -p "Train models now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Training models...${NC}"
        python -m lume_platform.ml.training
        echo -e "${GREEN}✓ Models trained${NC}"
    fi
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start API Server
echo -e "\n${BLUE}Starting API Server...${NC}"
API_PORT=${FLASK_PORT:-8000}

if check_port $API_PORT; then
    echo -e "${YELLOW}⚠ Port $API_PORT is already in use${NC}"
    echo "Killing existing process..."
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Use enhanced API if available
if [ -f "api/main_enhanced.py" ]; then
    API_MODULE="api.main_enhanced:app"
else
    API_MODULE="api.main:app"
fi

uvicorn $API_MODULE --host 0.0.0.0 --port $API_PORT --reload &
API_PID=$!
echo -e "${GREEN}✓ API Server started on http://localhost:$API_PORT${NC}"
echo -e "  ${BLUE}Docs: http://localhost:$API_PORT/docs${NC}"
echo -e "  ${BLUE}Health: http://localhost:$API_PORT/health${NC}"

# Wait for API to be ready
sleep 3

# Start Streamlit Dashboard
echo -e "\n${BLUE}Starting Streamlit Dashboard...${NC}"
DASHBOARD_PORT=${STREAMLIT_PORT:-8501}

if check_port $DASHBOARD_PORT; then
    echo -e "${YELLOW}⚠ Port $DASHBOARD_PORT is already in use${NC}"
    echo "Killing existing process..."
    lsof -ti:$DASHBOARD_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

streamlit run streamlit_app.py --server.port $DASHBOARD_PORT &
DASHBOARD_PID=$!
echo -e "${GREEN}✓ Dashboard started on http://localhost:$DASHBOARD_PORT${NC}"

# Print summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   🎉 ALL SERVICES STARTED                   ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  API Server:    http://localhost:$API_PORT${NC}"
echo -e "${GREEN}║  API Docs:      http://localhost:$API_PORT/docs${NC}"
echo -e "${GREEN}║  Dashboard:     http://localhost:$DASHBOARD_PORT${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $DASHBOARD_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Keep script running
wait
