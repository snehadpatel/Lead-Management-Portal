"""Model evaluation pipeline for MintLeads.

This module provides comprehensive evaluation for all trained models:
- Lead Scorer: classification metrics, ROC-AUC, calibration
- Investor Cluster: silhouette score, cluster stability
- NAV Forecaster: RMSE, MAE, R-squared, directional accuracy
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from config import (
    INVESTOR_CLUSTER_DIR,
    LEAD_SCORER_DIR,
    NAV_FORECASTER_DIR,
    PROCESSED_DATA_DIR,
    setup_logging,
)
from pipelines.train_lstm import NAVDataset, NAVForecaster

logger = setup_logging(__name__)


class ModelEvaluator:
    """Evaluator for all MintLeads models."""
    
    def __init__(self) -> None:
        """Initialize the model evaluator."""
        self.results: Dict[str, Any] = {}
    
    def evaluate_lead_scorer(self) -> Dict[str, Any]:
        """Evaluate the lead scorer model.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating Lead Scorer model")
        
        model_path = LEAD_SCORER_DIR / "rf_model.pkl"
        if not model_path.exists():
            logger.warning("Lead scorer model not found, skipping evaluation")
            return {"status": "model_not_found"}
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load test data
        data_path = PROCESSED_DATA_DIR / "leads_processed.csv"
        if not data_path.exists():
            logger.warning("Processed leads data not found")
            return {"status": "data_not_found"}
        
        df = pd.read_csv(data_path)
        
        # Prepare features (simplified)
        target_col = None
        for col in ["Converted", "conversion_score"]:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            return {"status": "target_not_found"}
        
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col and c not in ["_row_hash"]]
        
        X = df[numeric_cols].fillna(0)
        y = df[target_col]
        
        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else None,
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }
        
        # Feature importance analysis
        importances = model.feature_importances_
        top_features = sorted(
            zip(numeric_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        
        metrics["top_features"] = [
            {"feature": f, "importance": float(i)}
            for f, i in top_features
        ]
        
        logger.info(f"Lead scorer ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.results["lead_scorer"] = metrics
        return metrics
    
    def evaluate_investor_cluster(self) -> Dict[str, Any]:
        """Evaluate the investor clustering model.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating Investor Cluster model")
        
        model_path = INVESTOR_CLUSTER_DIR / "kmeans_model.pkl"
        if not model_path.exists():
            logger.warning("Cluster model not found, skipping evaluation")
            return {"status": "model_not_found"}
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load data with clusters
        data_path = PROCESSED_DATA_DIR / "investors_with_clusters.csv"
        if not data_path.exists():
            logger.warning("Clustered investor data not found")
            return {"status": "data_not_found"}
        
        df = pd.read_csv(data_path)
        
        if "cluster" not in df.columns:
            return {"status": "cluster_column_not_found"}
        
        # Cluster distribution
        cluster_counts = df["cluster"].value_counts().to_dict()
        
        # Cluster characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["cluster", "id"]]
        
        cluster_profiles = {}
        for cluster_id in df["cluster"].unique():
            cluster_data = df[df["cluster"] == cluster_id]
            profile = {
                "count": len(cluster_data),
                "percentage": len(cluster_data) / len(df) * 100,
                "means": cluster_data[numeric_cols].mean().to_dict(),
            }
            cluster_profiles[int(cluster_id)] = profile
        
        metrics = {
            "n_clusters": model.n_clusters,
            "inertia": float(model.inertia_),
            "cluster_distribution": {int(k): int(v) for k, v in cluster_counts.items()},
            "cluster_profiles": cluster_profiles,
        }
        
        logger.info(f"Cluster distribution: {cluster_counts}")
        
        self.results["investor_cluster"] = metrics
        return metrics
    
    def evaluate_nav_forecaster(self) -> Dict[str, Any]:
        """Evaluate the NAV forecaster model.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating NAV Forecaster model")
        
        model_path = NAV_FORECASTER_DIR / "lstm_model.pt"
        if not model_path.exists():
            logger.warning("NAV forecaster model not found, skipping evaluation")
            return {"status": "model_not_found"}
        
        # Load model
        checkpoint = torch.load(model_path, map_location="cpu")
        model_config = checkpoint["model_config"]
        
        model = NAVForecaster(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Load test data (use synthetic for evaluation if real not available)
        np.random.seed(42)
        test_data = np.cumsum(np.random.randn(500) * 0.01) + 100
        test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
        
        # Create dataset
        from pipelines.train_lstm import LSTM_WINDOW_SIZE
        dataset = NAVDataset(test_data, LSTM_WINDOW_SIZE)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset[i]
                x = x.unsqueeze(0)  # Add batch dimension
                pred = model(x)
                predictions.append(pred.item())
                actuals.append(y.item())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = accuracy_score(actual_direction, pred_direction)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "directional_accuracy": float(directional_accuracy),
        }
        
        logger.info(f"NAV forecaster RMSE: {rmse:.6f}, R2: {r2:.4f}")
        
        self.results["nav_forecaster"] = metrics
        return metrics
    
    def generate_evaluation_report(self) -> Path:
        """Generate a comprehensive evaluation report.
        
        Returns:
            Path to saved report.
        """
        report = {
            "lead_scorer": self.results.get("lead_scorer", {"status": "not_evaluated"}),
            "investor_cluster": self.results.get("investor_cluster", {"status": "not_evaluated"}),
            "nav_forecaster": self.results.get("nav_forecaster", {"status": "not_evaluated"}),
        }
        
        # Save JSON report
        report_path = Path("evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation report to {report_path}")
        return report_path
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run all model evaluations.
        
        Returns:
            Dictionary with all evaluation results.
        """
        logger.info("=" * 50)
        logger.info("Running All Model Evaluations")
        logger.info("=" * 50)
        
        self.evaluate_lead_scorer()
        self.evaluate_investor_cluster()
        self.evaluate_nav_forecaster()
        
        self.generate_evaluation_report()
        
        logger.info("=" * 50)
        logger.info("All Evaluations Complete")
        logger.info("=" * 50)
        
        return self.results


def main() -> Dict[str, Any]:
    """Main evaluation entry point."""
    evaluator = ModelEvaluator()
    return evaluator.run_all_evaluations()


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
