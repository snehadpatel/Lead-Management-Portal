# Models evaluation — master index (auto-generated)

Refreshed by `python -m lume_platform.ml.training`. **JSON under `model_evaluations/` is the source of truth** for metrics.

## 1. Lead conversion (supervised)
- Metrics: `model_evaluations/random_forest/real_metrics.json`
- Report: `model_evaluations/random_forest/rf_evaluation_report.md`
- Confusion matrix: `model_evaluations/random_forest/rf_confusion_matrix.png`

## 2. Investor personas (K-Means)
- Metrics: `model_evaluations/kmeans/kmeans_metrics.json`
- Report: `model_evaluations/kmeans/clustering_evaluation_report.md`
- Plots: `model_evaluations/kmeans/`

## 3. Sentiment (TF-IDF + logistic)
- Metrics: `model_evaluations/nlp_sentiment/metrics.json`
- Report: `model_evaluations/nlp_sentiment/nlp_evaluation_report.md`

## 4. Fund semantic search (TF-IDF retrieval)
- Narrative: `model_evaluations/tfidf_search/semantic_engine_report.md`
- Plot: `model_evaluations/tfidf_search/cosine_similarity_decay.png`

## 5. LSTM / sequence (optional script)
- Narrative: `model_evaluations/lstm_forecaster/neural_regression_report.md`
- Plot: `model_evaluations/lstm_forecaster/lstm_nav_predictions.png`

## Regenerate
```bash
export PYTHONPATH=src
python -m lume_platform.ml.training
```
