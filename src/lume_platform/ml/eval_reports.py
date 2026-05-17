"""
Generate Markdown evaluation reports from JSON so docs always match trained models.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_lead_scoring_markdown(
    eval_dir: Path,
    metrics: dict,
    class_report: dict | None = None,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Lead conversion model — evaluation (auto-generated)",
        "",
        "This file is **generated** when you run `python -m lume_platform.ml.training`. "
        "Do not hand-edit metrics here; change features or estimator in training instead.",
        "",
        "## Headline metrics (holdout)",
        f"- **Accuracy:** `{metrics.get('accuracy', 'n/a')}`",
        f"- **Precision (positive class):** `{metrics.get('precision', 'n/a')}`",
        f"- **Recall (positive class):** `{metrics.get('recall', 'n/a')}`",
        f"- **F1 (positive class):** `{metrics.get('f1_score', 'n/a')}`",
        f"- **Backend:** `{metrics.get('backend', 'n/a')}`",
        f"- **5-fold CV F1:** `{metrics.get('cv_f1_mean', 'n/a')}` ± `{metrics.get('cv_f1_std', 'n/a')}`",
        "",
    ]
    if class_report and "1" in class_report:
        c1 = class_report["1"]
        lines.extend(
            [
                "### Per-class (converted = 1)",
                f"- **Precision:** `{c1.get('precision', 0):.3f}`",
                f"- **Recall:** `{c1.get('recall', 0):.3f}`",
                f"- **F1:** `{c1.get('f1-score', 0):.3f}`",
                f"- **Support:** `{int(c1.get('support', 0))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranking / calibration (holdout)",
            f"- **ROC-AUC:** `{metrics.get('roc_auc', 'n/a')}`",
            f"- **Average precision (PR-AUC):** `{metrics.get('average_precision', 'n/a')}`",
            "",
            "## Artifacts",
            "- `real_metrics.json` — machine-readable",
            "- `classification_report.json` — sklearn report",
            "- `rf_confusion_matrix.png` — confusion matrix",
            "",
        ]
    )
    (eval_dir / "rf_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_kmeans_sidecar(eval_dir: Path, silhouette: float, n_clusters: int) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"silhouette": float(silhouette), "n_clusters": int(n_clusters)}
    (eval_dir / "kmeans_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            "# K-Means investor personas — evaluation (auto-generated)",
            "",
            f"- **Clusters (k):** `{n_clusters}`",
            f"- **Silhouette:** `{silhouette:.3f}` (higher is better; >0.5 is strong separation)",
            "",
            "## Artifacts",
            "- `kmeans_cluster_projection.png`",
            "- `kmeans_centroids_heatmap.png`",
            "- `kmeans_metrics.json`",
            "",
        ]
    )
    (eval_dir / "clustering_evaluation_report.md").write_text(md, encoding="utf-8")


def write_nlp_markdown(eval_dir: Path, metrics: dict) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# NLP sentiment — evaluation (auto-generated)",
            "",
            f"- **Accuracy:** `{metrics.get('accuracy', 0):.4f}`",
            f"- **F1 (macro):** `{metrics.get('f1_macro', 0):.4f}`",
            f"- **Method:** `{metrics.get('method', 'tfidf')}`",
            "",
            "## Artifact",
            "- `metrics.json`",
            "",
        ]
    )
    (eval_dir / "nlp_evaluation_report.md").write_text(body, encoding="utf-8")


def write_master_report(project_root: Path) -> None:
    """Master index: repo-relative paths so the doc is portable (Colab, teammates, CI)."""
    me = project_root / "model_evaluations"
    parts: list[str] = [
        "# Models evaluation — master index (auto-generated)",
        "",
        "Refreshed by `python -m lume_platform.ml.training`. "
        "**JSON under `model_evaluations/` is the source of truth** for metrics.",
        "",
        "## 1. Lead conversion (supervised)",
        "- Metrics: `model_evaluations/random_forest/real_metrics.json`",
        "- Report: `model_evaluations/random_forest/rf_evaluation_report.md`",
        "- Confusion matrix: `model_evaluations/random_forest/rf_confusion_matrix.png`",
        "",
        "## 2. Investor personas (K-Means)",
        "- Metrics: `model_evaluations/kmeans/kmeans_metrics.json`",
        "- Report: `model_evaluations/kmeans/clustering_evaluation_report.md`",
        "- Plots: `model_evaluations/kmeans/`",
        "",
        "## 3. Sentiment (SBERT + logistic)",
        "- Metrics: `model_evaluations/nlp_sentiment/metrics.json`",
        "- Report: `model_evaluations/nlp_sentiment/nlp_evaluation_report.md`",
        "",
        "## 4. Fund semantic search (SBERT)",
        "- Narrative: `model_evaluations/tfidf_search/semantic_engine_report.md`",
        "- Plot: `model_evaluations/tfidf_search/cosine_similarity_decay.png`",
        "",
        "## 5. LSTM / sequence (optional script)",
        "- Narrative: `model_evaluations/lstm_forecaster/neural_regression_report.md`",
        "- Plot: `model_evaluations/lstm_forecaster/lstm_nav_predictions.png`",
        "",
        "## Regenerate",
        "```bash",
        "export PYTHONPATH=src",
        "python -m lume_platform.ml.training",
        "```",
        "",
    ]
    (me / "MODELS_EVALUATION_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
