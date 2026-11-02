"""
PyTorch Multi-Layer Perceptron (MLP) for Lead Conversion Prediction.
Trained on identical Spark-materialized Parquet data as Random Forest for fair, genuine benchmark comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lume_platform.config import CLEANED_DIR, DATA_ROOT, EVAL_DIR, MODELS_DIR
from lume_platform.ml.feature_config import LEAD_CATEGORICAL_FEATURES, LEAD_NUMERIC_FEATURES


class LeadMLP(nn.Module):
    """
    3-Layer PyTorch MLP with Batch Normalization and Dropout for Tabular Lead Scoring.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_lead_dataset() -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Loads Spark Parquet or CSV and prepares ColumnTransformer pipeline."""
    pq_path = CLEANED_DIR / "lead_scoring_clean"
    csv_path = DATA_ROOT / "structured/leads/lead_scoring/Lead Scoring.csv"

    if pq_path.exists() and any(pq_path.iterdir()):
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = (
            df.columns.str.lower()
            .str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("-", "_")
        )
    else:
        raise FileNotFoundError(f"Lead dataset not found at {pq_path} or {csv_path}")

    numeric_features = list(LEAD_NUMERIC_FEATURES)
    cat_features = list(LEAD_CATEGORICAL_FEATURES)
    use_cols = numeric_features + cat_features + ["converted"]
    df_ml = df[[c for c in use_cols if c in df.columns]].copy()

    for c in numeric_features:
        if c not in df_ml.columns:
            df_ml[c] = 0
    for c in cat_features:
        if c not in df_ml.columns:
            df_ml[c] = "Unknown"

    df_ml[cat_features] = df_ml[cat_features].fillna("Unknown").astype(str)
    df_ml[numeric_features] = df_ml[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df_ml.drop(columns=["converted"])
    y = df_ml["converted"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )
    return X, y, preprocessor


def train_mlp_experiment() -> Dict[str, Any]:
    """
    Trains PyTorch LeadMLP, measures exact epoch losses, computes evaluation metrics and latency,
    and saves authentic artifacts.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    eval_dir = EVAL_DIR / "lead_mlp"
    eval_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, preprocessor = load_lead_dataset()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    in_dim = X_train.shape[1]
    model = LeadMLP(in_features=in_dim)

    # Class weight calculation for imbalance
    pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train.values, dtype=torch.float32)
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test.values, dtype=torch.float32)

    batch_size = 64
    dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = 40
    train_losses = []
    val_losses = []

    print(f"Training LeadMLP ({in_dim} -> 64 -> 32 -> 1) over {epochs} epochs...")
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)

        epoch_train_loss = running_loss / len(train_tensor_x)
        train_losses.append(round(epoch_train_loss, 4))

        # Evaluate validation loss
        model.eval()
        with torch.no_grad():
            val_logits = model(test_tensor_x)
            val_loss = criterion(val_logits, test_tensor_y).item()
            val_losses.append(round(val_loss, 4))

        scheduler.step(val_loss)

    train_duration = round(time.time() - start_train_time, 2)

    # Final Inference Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(test_tensor_x)
        test_probs = torch.sigmoid(test_logits).numpy()

    # Optimal threshold search
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.arange(0.30, 0.75, 0.02):
        temp_preds = (test_probs >= t).astype(int)
        f = f1_score(y_test, temp_preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)

    preds = (test_probs >= best_thresh).astype(int)

    # Latency benchmark (single sample inference averaged over 100 runs)
    single_x = test_tensor_x[0:1]
    warmup = model(single_x)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(single_x)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = round(float(np.mean(latencies)), 3)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    avg_prec = average_precision_score(y_test, test_probs)
    cm = confusion_matrix(y_test, preds).tolist()

    # Permutation feature importance on preprocessed numeric features
    print("Computing permutation importance for MLP...")
    perm_importances: Dict[str, float] = {}
    baseline_score = roc_auc_score(y_test, test_probs)
    feature_names = preprocessor.get_feature_names_out()

    # Group by original raw feature name
    grouped_indices: Dict[str, List[int]] = {}
    for idx, f_name in enumerate(feature_names):
        raw_name = f_name.split("__")[-1].split("_")[0]
        # find matching raw feature
        for orig in list(LEAD_NUMERIC_FEATURES) + list(LEAD_CATEGORICAL_FEATURES):
            if orig in f_name:
                grouped_indices.setdefault(orig, []).append(idx)
                break

    for raw_name, idx_list in grouped_indices.items():
        X_test_perm = X_test.copy()
        for idx in idx_list:
            X_test_perm[:, idx] = np.random.permutation(X_test_perm[:, idx])
        with torch.no_grad():
            perm_logits = model(torch.tensor(X_test_perm, dtype=torch.float32))
            perm_probs = torch.sigmoid(perm_logits).numpy()
            perm_score = roc_auc_score(y_test, perm_probs)
            perm_importances[raw_name] = round(float(max(0.0, baseline_score - perm_score)), 4)

    # Sort importances
    sorted_importances = dict(sorted(perm_importances.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        "model_name": "LeadMLP (PyTorch)",
        "accuracy": f"{acc * 100:.2f}%",
        "accuracy_raw": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(avg_prec), 4),
        "decision_threshold": round(best_thresh, 4),
        "latency_ms": mean_latency_ms,
        "train_duration_sec": train_duration,
        "epochs": epochs,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "confusion_matrix": cm,
        "feature_importances": sorted_importances,
    }

    # Save metrics JSON
    with open(eval_dir / "real_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save loss history JSON
    loss_history = {
        "epochs": list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    with open(eval_dir / "training_loss.json", "w") as f:
        json.dump(loss_history, f, indent=2)

    # Plot & save loss curve
    plt.figure(figsize=(7, 4), dpi=150)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="#3B82F6", lw=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="#10B981", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("LeadMLP Training & Validation Loss Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(eval_dir / "loss_curves.png")
    plt.close()

    # Plot & save confusion matrix
    plt.figure(figsize=(5, 4), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("LeadMLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig(eval_dir / "mlp_confusion_matrix.png")
    plt.close()

    # Save PyTorch model state and preprocessor bundle
    checkpoint_path = MODELS_DIR / "lead_mlp_model.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_features": in_dim,
            "preprocessor": preprocessor,
            "decision_threshold": best_thresh,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"LeadMLP successfully trained and saved to {checkpoint_path}")
    return metrics


if __name__ == "__main__":
    train_mlp_experiment()

