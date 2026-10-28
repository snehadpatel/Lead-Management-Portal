"""Modular PyTorch LSTM Training Pipeline for Mutual Fund NAV Forecasting.

Zero-Leakage Guarantee:
1. Chronological split: Train (70%) -> Val (15%) -> Test (15%).
2. Scaler fit on Train split only.
3. Out-of-sample evaluation strictly on unseen Test split.
4. Concrete comparison against deterministic Naive and Moving Average baselines.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from lume_platform.config import ARTIFACTS_DIR, EVAL_DIR, MODELS_DIR, PROJECT_ROOT
from lume_platform.data.timeseries_dataset import ChronologicalTimeSeriesDataset, load_merged_market_series
from lume_platform.ml.forecaster import LSTMForecaster
from lume_platform.utils.metrics import compute_regression_metrics
from lume_platform.utils.seed import seed_everything


def _save_loss_curve(train_loss: list[float], val_loss: list[float], out_path: Path, scheme_code: str) -> None:
    """Save training/validation loss curve to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(train_loss, label="Train Loss (Huber)", color="#2563eb", lw=2)
        plt.plot(val_loss, label="Val Loss (Huber)", color="#ea580c", lw=2, linestyle="--")
        plt.title(f"PyTorch LSTM Loss Curves (Scheme {scheme_code})")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate loss curve plot: {e}", flush=True)


def train_lstm_forecaster(
    config_path: Optional[Path] = None,
    scheme_code: str = "100119",
) -> Dict[str, Any]:
    """Train PyTorch LSTM Forecaster on real market time-series data."""
    # 1. Load config
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "forecaster_lstm.yaml")
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed)

    lookback = cfg.get("model", {}).get("lookback", 30)
    horizon = cfg.get("model", {}).get("horizon", 5)
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 64)
    num_layers = cfg.get("model", {}).get("num_layers", 2)
    dropout = cfg.get("model", {}).get("dropout", 0.2)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    learning_rate = cfg.get("training", {}).get("learning_rate", 0.001)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-5)
    epochs = cfg.get("training", {}).get("epochs", 25)
    patience = cfg.get("training", {}).get("early_stopping_patience", 7)

    device_str = cfg.get("training", {}).get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 Training PyTorch LSTM Forecaster on device: {device} (Scheme: {scheme_code})", flush=True)

    # 2. Load dataset with zero data leakage
    df_merged = load_merged_market_series(scheme_code=scheme_code)
    dataset = ChronologicalTimeSeriesDataset(
        df_merged,
        target_col="nav",
        feature_cols=["nav"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        lookback=lookback,
        horizon=horizon,
    )

    X_train_t = torch.tensor(dataset.X_train, dtype=torch.float32)
    y_train_t = torch.tensor(dataset.y_train, dtype=torch.float32)
    X_val_t = torch.tensor(dataset.X_val, dtype=torch.float32)
    y_val_t = torch.tensor(dataset.y_val, dtype=torch.float32)
    X_test_t = torch.tensor(dataset.X_test, dtype=torch.float32)
    y_test_t = torch.tensor(dataset.y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    # 3. Model setup
    model = LSTMForecaster(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    ).to(device)

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 4. Training loop with early stopping
    history_loss = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss = float(np.mean(val_losses))
        history_loss["train"].append(epoch_train_loss)
        history_loss["val"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", flush=True)

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (Patience: {patience})", flush=True)
            break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # 5. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scalers_dir = ARTIFACTS_DIR / "ml_scalers"
    scalers_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "lstm_nav_pattern_predictor.pth"
    scaler_path = scalers_dir / "mf_nav_global_scaler.pkl"

    torch.save(model.state_dict(), str(model_path))
    with open(str(scaler_path), "wb") as f:
        pickle.dump(dataset.scaler, f)
    print(f"✅ Checkpoint saved: {model_path}", flush=True)
    print(f"✅ Scaler saved: {scaler_path}", flush=True)

    # 6. Out-of-sample Evaluation on TEST set (Unseen holdout)
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t.to(device)).cpu().numpy()

    # Invert to real NAV values using train scaler
    test_preds_real = dataset.inverse_transform_target(test_preds_scaled)
    test_actual_real = dataset.inverse_transform_target(dataset.y_test)
    test_last_known = dataset.inverse_transform_target(dataset.X_test[:, -1, 0])

    lstm_metrics = compute_regression_metrics(test_actual_real, test_preds_real, last_known=test_last_known)

    # Baselines on the exact same test set:
    naive_preds = np.repeat(test_last_known[:, np.newaxis], horizon, axis=1)
    naive_metrics = compute_regression_metrics(test_actual_real, naive_preds, last_known=test_last_known)

    test_X_real = dataset.inverse_transform_target(dataset.X_test[:, :, 0])
    ma_preds = np.repeat(np.mean(test_X_real[:, -5:], axis=1, keepdims=True), horizon, axis=1)
    ma_metrics = compute_regression_metrics(test_actual_real, ma_preds, last_known=test_last_known)

    print("\n📊 Out-of-Sample Test Set Evaluation (Real Data, Zero Leakage):", flush=True)
    print(f"  • PyTorch LSTM:    RMSE = ₹{lstm_metrics['rmse']:.2f} | MAE = ₹{lstm_metrics['mae']:.2f} | MAPE = {lstm_metrics['mape_pct']:.2f}% | Dir Acc = {lstm_metrics.get('directional_accuracy_pct', 0.0)}%", flush=True)
    print(f"  • Naive Last-Val:  RMSE = ₹{naive_metrics['rmse']:.2f} | MAE = ₹{naive_metrics['mae']:.2f} | MAPE = {naive_metrics['mape_pct']:.2f}% | Dir Acc = {naive_metrics.get('directional_accuracy_pct', 0.0)}%", flush=True)
    print(f"  • 5-Day Mov Avg:   RMSE = ₹{ma_metrics['rmse']:.2f} | MAE = ₹{ma_metrics['mae']:.2f} | MAPE = {ma_metrics['mape_pct']:.2f}% | Dir Acc = {ma_metrics.get('directional_accuracy_pct', 0.0)}%", flush=True)

    # 7. Persist Evaluation Reports & Plots
    eval_dir = EVAL_DIR / "lstm_forecaster"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve plot
    loss_plot_path = eval_dir / "lstm_train_val_loss.png"
    _save_loss_curve(history_loss["train"], history_loss["val"], loss_plot_path, scheme_code)

    # Save real metrics JSON
    metrics_summary = {
        "scheme_code": scheme_code,
        "seed": seed,
        "dataset_summary": dataset.get_summary(),
        "lstm_test_metrics": lstm_metrics,
        "naive_baseline_metrics": naive_metrics,
        "moving_average_metrics": ma_metrics,
        "rmse_improvement_over_naive_pct": round(float((naive_metrics["rmse"] - lstm_metrics["rmse"]) / naive_metrics["rmse"] * 100.0), 2),
        "rmse_improvement_over_ma_pct": round(float((ma_metrics["rmse"] - lstm_metrics["rmse"]) / ma_metrics["rmse"] * 100.0), 2),
        "epochs_trained": len(history_loss["train"]),
        "best_val_loss": round(float(best_val_loss), 6),
    }

    metrics_file = eval_dir / "real_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Also save to models dir for registry consumption
    with open(MODELS_DIR / "forecaster_metrics.json", "w") as f:
        json.dump({
            "test_rmse": lstm_metrics["rmse"],
            "test_mae": lstm_metrics["mae"],
            "test_mape_pct": lstm_metrics["mape_pct"],
            "accuracy_metric": f"Test RMSE: ₹{lstm_metrics['rmse']:.2f} | MAPE: {lstm_metrics['mape_pct']:.2f}%",
        }, f, indent=2)

    # Write Model Card Markdown
    model_card = f"""# Model Card: PyTorch Multi-Horizon LSTM Forecaster

## 1. Overview
- **Model Architecture**: Multi-Layer LSTM with Monte Carlo Dropout ({num_layers} layers, {hidden_dim} hidden units, dropout={dropout})
- **Task**: 5-Day Multi-Step NAV Forecast (Lookback = {lookback} days, Horizon = {horizon} days)
- **Primary Asset**: Mutual Fund Scheme `{scheme_code}` (Merged with NIFTY 50 trading calendar)

## 2. Zero-Data-Leakage Design
- **Splits**: Strictly chronological
  - Train: {dataset.train_df['date'].min()} to {dataset.train_df['date'].max()} ({len(dataset.train_df)} trading days)
  - Validation: {dataset.val_df['date'].min()} to {dataset.val_df['date'].max()} ({len(dataset.val_df)} trading days)
  - Test: {dataset.test_df['date'].min()} to {dataset.test_df['date'].max()} ({len(dataset.test_df)} trading days)
- **Scaling**: `MinMaxScaler` fitted solely on the Train split. Out-of-sample evaluation strictly on unseen Test split.

## 3. Real Performance Benchmarks (Unseen Test Holdout)
| Model | RMSE (₹) | MAE (₹) | MAPE (%) | Directional Accuracy (%) |
|---|---|---|---|---|
| **PyTorch LSTM** | **₹{lstm_metrics['rmse']:.2f}** | **₹{lstm_metrics['mae']:.2f}** | **{lstm_metrics['mape_pct']:.2f}%** | **{lstm_metrics.get('directional_accuracy_pct', 0.0)}%** |
| Naive Last-Value | ₹{naive_metrics['rmse']:.2f} | ₹{naive_metrics['mae']:.2f} | {naive_metrics['mape_pct']:.2f}% | {naive_metrics.get('directional_accuracy_pct', 0.0)}% |
| 5-Day Moving Average | ₹{ma_metrics['rmse']:.2f} | ₹{ma_metrics['mae']:.2f} | {ma_metrics['mape_pct']:.2f}% | {ma_metrics.get('directional_accuracy_pct', 0.0)}% |

## 4. Uncertainty & Limitations
- **Epistemic Uncertainty**: Quantified via $T=20$ Monte Carlo Dropout evaluations to produce empirical 90% confidence bands.
- **Product Honesty**: Not investment advice. Mutual fund NAV trajectories reflect market-wide drift and systematic index shocks.
"""
    with open(eval_dir / "model_card.md", "w") as f:
        f.write(model_card)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metrics_path": str(metrics_file),
        "test_rmse": lstm_metrics["rmse"],
        "test_mape_pct": lstm_metrics["mape_pct"],
    }
