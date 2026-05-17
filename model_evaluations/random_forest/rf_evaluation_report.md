# Lead conversion model — evaluation (auto-generated)

This file is **generated** when you run `python -m lume_platform.ml.training`. Do not hand-edit metrics here; change features or estimator in training instead.

## Headline metrics (holdout)
- **Accuracy:** `85.00%`
- **Precision (positive class):** `0.75`
- **Recall (positive class):** `0.90`
- **F1 (positive class):** `0.82`
- **Backend:** `random_forest`
- **5-fold CV F1:** `0.806` ± `0.018`

### Per-class (converted = 1)
- **Precision:** `0.751`
- **Recall:** `0.903`
- **F1:** `0.820`
- **Support:** `144`

## Ranking / calibration (holdout)
- **ROC-AUC:** `0.933`
- **Average precision (PR-AUC):** `0.890`

## Artifacts
- `real_metrics.json` — machine-readable
- `classification_report.json` — sklearn report
- `rf_confusion_matrix.png` — confusion matrix
