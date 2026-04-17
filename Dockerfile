FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV LUME_PROJECT_ROOT=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY api/ /app/api/
COPY output_production_final/ /app/output_production_final/
COPY artifacts/ /app/artifacts/
COPY model_evaluations/ /app/model_evaluations/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
