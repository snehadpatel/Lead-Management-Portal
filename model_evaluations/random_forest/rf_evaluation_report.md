# Lead conversion model — evaluation (auto-generated)

This file is **generated** when you run `python -m lume_platform.ml.training`. Do not hand-edit metrics here; change features or estimator in training instead.

## Headline metrics (holdout)
- **Accuracy:** `87.37%`
- **Precision (positive class):** `0.81`
- **Recall (positive class):** `0.87`
- **F1 (positive class):** `0.84`
- **Backend:** `xgboost`
- **5-fold CV F1:** `0.809` ± `0.020`

### Per-class (converted = 1)
- **Precision:** `0.812`
- **Recall:** `0.868`
- **F1:** `0.839`
- **Support:** `144`

## Ranking / calibration (holdout)
- **ROC-AUC:** `0.939`
- **Average precision (PR-AUC):** `0.910`

## Artifacts
- `real_metrics.json` — machine-readable
- `classification_report.json` — sklearn report
- `rf_confusion_matrix.png` — confusion matrix
