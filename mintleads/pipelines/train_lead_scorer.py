"""Lead Scorer training pipeline for MintLeads.

This module trains a Random Forest classifier for lead scoring:
- Stratified train-test split
- GridSearchCV for hyperparameter tuning
- Stratified K-Fold cross-validation
- MLflow logging
- Model artifacts and evaluation plots
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from config import (
    LEAD_SCORER_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    setup_logging,
)

logger = setup_logging(__name__)


class LeadScorerTrainer:
    """Trainer for Random Forest lead scoring model."""
    
    def __init__(self) -> None:
        """Initialize the lead scorer trainer."""
        self.model: RandomForestClassifier = None
        self.best_params: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        
    def load_data(self) -> pd.DataFrame:
        """Load preprocessed lead data.
        
        Returns:
            DataFrame with processed lead data.
        """
        data_path = PROCESSED_DATA_DIR / "leads_processed.csv"
        
        if not data_path.exists():
            # Fallback to raw data if processed doesn't exist
            from config import LEADS_DATA_PATH
            data_path = LEADS_DATA_PATH
            logger.warning(f"Processed data not found, using raw data from {data_path}")
        
        logger.info(f"Loading lead data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.
        
        Args:
            df: DataFrame with lead data.
            
        Returns:
            Tuple of (features, target).
        """
        df = df.copy()
        
        # Identify target column
        target_col = None
        for col in ["Converted", "conversion_score", "Converted_Lead"]:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("No target column found in data")
        
        logger.info(f"Using target column: {target_col}")
        y = df[target_col]
        
        # Drop non-feature columns
        drop_cols = [
            target_col, "Prospect ID", "Lead Number", "_row_hash",
            "Tags", "Last Notable Activity",
        ]
        
        # Also drop object columns that weren't encoded
        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        
        # Keep only numeric columns for now
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        self.feature_names = X.columns.tolist()
        logger.info(f"Prepared {len(self.feature_names)} features")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train-test split.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train conversion rate: {y_train.mean():.3f}")
        logger.info(f"Test conversion rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_with_grid_search(self) -> RandomForestClassifier:
        """Train Random Forest with GridSearchCV.
        
        Returns:
            Best trained model.
        """
        logger.info("Starting GridSearchCV for hyperparameter tuning")
        
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 15, 20, 25],
            "min_samples_split": [2, 5, 10],
        }
        
        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        
        logger.info(f"Fitting GridSearchCV with {CV_FOLDS}-fold CV")
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Log per-fold F1 scores
        cv_results = grid_search.cv_results_
        for i in range(CV_FOLDS):
            fold_key = f"split{i}_test_score"
            if fold_key in cv_results:
                fold_scores = cv_results[fold_key]
                logger.info(f"Fold {i+1} F1 scores: mean={fold_scores.mean():.4f}")
        
        return self.model
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating model on test set")
        
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(self.y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(self.y_test, y_prob)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        metrics = {
            "f1_weighted": f1,
            "roc_auc": roc_auc,
            "classification_report": report,
        }
        
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        return metrics
    
    def save_plots(self) -> Dict[str, Path]:
        """Generate and save evaluation plots.
        
        Returns:
            Dictionary mapping plot names to file paths.
        """
        logger.info("Generating evaluation plots")
        plot_paths = {}
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        cm_path = LEAD_SCORER_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths["confusion_matrix"] = cm_path
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        # 2. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc_score = roc_auc_score(self.y_test, y_prob)
        
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True)
        
        roc_path = LEAD_SCORER_DIR / "roc_curve.png"
        fig.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths["roc_curve"] = roc_path
        logger.info(f"Saved ROC curve to {roc_path}")
        
        # 3. Feature Importance
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        ax.barh(range(len(indices)), importances[indices], align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top 20 Feature Importances")
        ax.grid(True, axis="x")
        
        fi_path = LEAD_SCORER_DIR / "feature_importance.png"
        fig.savefig(fi_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths["feature_importance"] = fi_path
        logger.info(f"Saved feature importance to {fi_path}")
        
        return plot_paths
    
    def save_model(self) -> Path:
        """Save the trained model.
        
        Returns:
            Path to saved model file.
        """
        model_path = LEAD_SCORER_DIR / "rf_model.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def log_to_mlflow(
        self,
        metrics: Dict[str, Any],
        plot_paths: Dict[str, Path],
    ) -> None:
        """Log training run to MLflow.
        
        Args:
            metrics: Evaluation metrics.
            plot_paths: Paths to evaluation plots.
        """
        logger.info("Logging run to MLflow")
        
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            
            with mlflow.start_run(run_name="lead_scorer_rf"):
                # Log parameters
                mlflow.log_params(self.best_params)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("cv_folds", CV_FOLDS)
                
                # Log metrics
                mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
                mlflow.log_metric("roc_auc", metrics["roc_auc"])
                
                # Log artifacts
                for name, path in plot_paths.items():
                    mlflow.log_artifact(str(path))
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                logger.info("Successfully logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def train(self) -> Dict[str, Any]:
        """Run full training pipeline.
        
        Returns:
            Dictionary with training results.
        """
        logger.info("=" * 50)
        logger.info("Starting Lead Scorer Training")
        logger.info("=" * 50)
        
        # Load and prepare data
        df = self.load_data()
        X, y = self.prepare_features(df)
        self.split_data(X, y)
        
        # Train model
        self.train_with_grid_search()
        
        # Evaluate
        metrics = self.evaluate_model()
        
        # Save artifacts
        model_path = self.save_model()
        plot_paths = self.save_plots()
        
        # Log to MLflow
        self.log_to_mlflow(metrics, plot_paths)
        
        logger.info("=" * 50)
        logger.info("Lead Scorer Training Complete")
        logger.info("=" * 50)
        
        return {
            "model_path": str(model_path),
            "best_params": self.best_params,
            "metrics": metrics,
            "plots": {k: str(v) for k, v in plot_paths.items()},
        }


def main() -> Dict[str, Any]:
    """Main training entry point."""
    trainer = LeadScorerTrainer()
    return trainer.train()


if __name__ == "__main__":
    import json
    
    results = main()
    print(json.dumps(results, indent=2))
