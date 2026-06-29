# Lume AI MVP — FastAPI Backend
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY apps/api/ ./apps/api/
COPY ml-artifacts/ ./ml-artifacts/

# Environment
ENV PYTHONPATH=/app/apps/api
ENV LUME_PROJECT_ROOT=/app
ENV LUME_ML_ARTIFACTS=/app/ml-artifacts
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
