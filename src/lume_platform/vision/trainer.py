"""
Phase 3 Computer Vision Training, Evaluation & Grad-CAM Pipeline.

Executes:
1. Candlestick dataset generation with strict fund-based partitioning.
2. Training of Custom 4-Block CNN with class-weighted cross-entropy and early stopping.
3. Training of Fine-Tuned ResNet-18 with differential learning rates.
4. Classical technical indicator baseline (RSI-14 + SMA-20).
5. Comprehensive test evaluation: Accuracy, Macro-F1, Confusion Matrix.
6. Grad-CAM generation and saving of successful & failure overlay cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lume_platform.config import EVAL_DIR, MODELS_DIR, PROJECT_ROOT
from lume_platform.utils.seed import seed_everything
from lume_platform.vision.candlestick_dataset import (
    LABEL_NAMES,
    CandlestickDataset,
    generate_vision_dataset,
)
from lume_platform.vision.gradcam import GradCAM, overlay_gradcam_on_image
from lume_platform.vision.models import CandlestickCNN, CandlestickResNet18


def compute_technical_baseline(manifest_df: pd.DataFrame, split: str = "test") -> Dict[str, Any]:
    """
    Evaluates rule-based technical indicator baseline (RSI-14 + SMA-20) on specified split.
    """
    test_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
    y_true = test_df["label"].values
    y_pred = []

    for idx, row in test_df.iterrows():
        # Retrieve original fund history to compute RSI and SMA
        fund = str(row["fund"])
        end_date = pd.to_datetime(row["end_date"])
        cache_path = PROJECT_ROOT / "output_production_final" / "nav_cache" / f"hist_{fund}.json"
        
        with open(cache_path, "r") as fp:
            data = json.load(fp)
        df_f = pd.DataFrame(data)
        df_f["date"] = pd.to_datetime(df_f["date"], format="%d-%m-%Y", errors="coerce")
        df_f = df_f.dropna().sort_values("date").reset_index(drop=True)
        
        # Up to end_date
        sub = df_f[df_f["date"] <= end_date].tail(40).copy()
        navs = sub["nav"].values
        if len(navs) < 20:
            y_pred.append(2)  # Default consolidation
            continue

        # RSI-14
        deltas = np.diff(navs)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.001
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.001
        rs = (avg_gain / avg_loss) if avg_loss > 0 else 1.0
        rsi = 100 - (100 / (1 + rs))

        # SMA-20
        sma20 = np.mean(navs[-20:])
        curr = navs[-1]

        # Rule:
        if rsi < 45 and curr > sma20:
            pred = 0  # Bullish
        elif rsi > 55 and curr < sma20:
            pred = 1  # Bearish
        else:
            pred = 2  # Consolidation
        y_pred.append(pred)

    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "model_name": "Rule-Based Technical Baseline (RSI-14 + SMA-20)",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Trains a model with early stopping on validation Macro-F1."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_running_loss += loss.item() * len(y)
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (best Val Macro-F1: {best_val_f1:.4f})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on a dataset split."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()
    report = classification_report(
        all_targets,
        all_preds,
        target_names=["Bullish Breakout", "Bearish Breakdown", "Consolidation"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def generate_and_save_gradcam_artifacts(
    model: nn.Module,
    test_loader: DataLoader,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, str]:
    """
    Extracts exemplary Grad-CAM overlays:
    - Bullish Breakout (Success)
    - Bearish Breakdown (Success)
    - Consolidation (Success)
    - Failure Cases (Misclassifications)
    """
    target_layer = model.get_target_layer()
    gradcam = GradCAM(model, target_layer)

    samples = {"bullish": None, "bearish": None, "consolidation": None, "failures": []}

    for x_batch, y_batch, paths in test_loader:
        for i in range(len(y_batch)):
            tensor_single = x_batch[i : i + 1].to(device)
            true_label = int(y_batch[i].item())
            img_path = paths[i]

            cam, pred_label, conf = gradcam.generate(tensor_single)
            orig_img = Image.open(img_path).convert("RGB")
            overlay = overlay_gradcam_on_image(orig_img, cam, alpha=0.55)

            if pred_label == true_label:
                if true_label == 0 and samples["bullish"] is None:
                    samples["bullish"] = (orig_img, overlay, img_path, conf)
                elif true_label == 1 and samples["bearish"] is None:
                    samples["bearish"] = (orig_img, overlay, img_path, conf)
                elif true_label == 2 and samples["consolidation"] is None:
                    samples["consolidation"] = (orig_img, overlay, img_path, conf)
            else:
                if len(samples["failures"]) < 2:
                    samples["failures"].append((orig_img, overlay, img_path, true_label, pred_label, conf))

        if (
            samples["bullish"] is not None
            and samples["bearish"] is not None
            and samples["consolidation"] is not None
            and len(samples["failures"]) >= 2
        ):
            break

    gradcam.remove_hooks()

    saved_paths = {}

    def _save_side_by_side(orig, over, title, dest_path):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
        axes[0].imshow(orig)
        axes[0].set_title("Input Chart Image")
        axes[0].axis("off")
        axes[1].imshow(over)
        axes[1].set_title(title)
        axes[1].axis("off")
        plt.tight_layout()
        fig.savefig(dest_path, bbox_inches="tight")
        plt.close(fig)

    if samples["bullish"]:
        p = out_dir / "gradcam_bullish_breakout.png"
        _save_side_by_side(
            samples["bullish"][0],
            samples["bullish"][1],
            f"Grad-CAM: Bullish Breakout (Conf: {samples['bullish'][3]:.2f})",
            p,
        )
        saved_paths["bullish"] = str(p)

    if samples["bearish"]:
        p = out_dir / "gradcam_bearish_breakdown.png"
        _save_side_by_side(
            samples["bearish"][0],
            samples["bearish"][1],
            f"Grad-CAM: Bearish Breakdown (Conf: {samples['bearish'][3]:.2f})",
            p,
        )
        saved_paths["bearish"] = str(p)

    if samples["consolidation"]:
        p = out_dir / "gradcam_consolidation.png"
        _save_side_by_side(
            samples["consolidation"][0],
            samples["consolidation"][1],
            f"Grad-CAM: Consolidation (Conf: {samples['consolidation'][3]:.2f})",
            p,
        )
        saved_paths["consolidation"] = str(p)

    for idx, fail in enumerate(samples["failures"], 1):
        p = out_dir / f"gradcam_failure_case_{idx}.png"
        title = f"FAILURE CASE: True={LABEL_NAMES[fail[3]]} | Pred={LABEL_NAMES[fail[4]]} ({fail[5]:.2f})"
        _save_side_by_side(fail[0], fail[1], title, p)
        saved_paths[f"failure_{idx}"] = str(p)

    return saved_paths


def run_phase3_vision_experiment() -> Dict[str, Any]:
    """Complete Phase 3 pipeline execution."""
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Phase 3 Vision Pipeline running on device: {device}")

    eval_dir = EVAL_DIR / "phase3_vision"
    eval_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation
    manifest_df = generate_vision_dataset(force_rebuild=False)

    # Report class balance
    train_df = manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "val"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    class_counts_train = train_df["label"].value_counts().to_dict()
    total_train = len(train_df)
    class_weights_vals = [total_train / (3.0 * class_counts_train[i]) for i in range(3)]
    class_weights = torch.tensor(class_weights_vals, dtype=torch.float32)

    train_dataset = CandlestickDataset(manifest_df, split="train")
    val_dataset = CandlestickDataset(manifest_df, split="val")
    test_dataset = CandlestickDataset(manifest_df, split="test")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Train Custom 4-Block CNN
    print("\n--- Training Model A: Custom 4-Block CNN (422K params) ---")
    cnn_model = CandlestickCNN(num_classes=3).to(device)
    cnn_model, cnn_train_loss, cnn_val_loss = train_single_model(
        cnn_model,
        train_loader,
        val_loader,
        class_weights,
        device,
        epochs=15,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
    )
    cnn_test_metrics = evaluate_model(cnn_model, test_loader, device)
    print(f"Custom CNN Test Accuracy: {cnn_test_metrics['accuracy']:.4f} | Macro-F1: {cnn_test_metrics['macro_f1']:.4f}")
    torch.save(cnn_model.state_dict(), MODELS_DIR / "candlestick_cnn_custom.pth")

    # 3. Train Fine-Tuned ResNet-18
    print("\n--- Training Model B: Fine-Tuned ResNet-18 (11.18M params) ---")
    resnet_model = CandlestickResNet18(num_classes=3).to(device)
    resnet_model, resnet_train_loss, resnet_val_loss = train_single_model(
        resnet_model,
        train_loader,
        val_loader,
        class_weights,
        device,
        epochs=15,
        lr=2e-4,
        weight_decay=1e-4,
        patience=5,
    )
    resnet_test_metrics = evaluate_model(resnet_model, test_loader, device)
    print(f"ResNet-18 Test Accuracy: {resnet_test_metrics['accuracy']:.4f} | Macro-F1: {resnet_test_metrics['macro_f1']:.4f}")
    torch.save(resnet_model.state_dict(), MODELS_DIR / "candlestick_resnet18_finetuned.pth")

    # 4. Technical Baseline
    print("\n--- Evaluating Technical Indicator Baseline (RSI-14 + SMA-20) ---")
    baseline_metrics = compute_technical_baseline(manifest_df, split="test")
    print(f"Baseline Test Accuracy: {baseline_metrics['accuracy']:.4f} | Macro-F1: {baseline_metrics['macro_f1']:.4f}")

    # 5. Grad-CAM Artifacts
    print("\n--- Generating Grad-CAM Overlays & Failure Cases ---")
    gradcam_paths = generate_and_save_gradcam_artifacts(cnn_model, test_loader, eval_dir, device)

    # 6. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)
    for idx, (title, cm) in enumerate([
        ("Rule-Based Baseline", baseline_metrics["confusion_matrix"]),
        ("Custom 4-Block CNN", cnn_test_metrics["confusion_matrix"]),
        ("Fine-Tuned ResNet-18", resnet_test_metrics["confusion_matrix"]),
    ]):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Bullish", "Bearish", "Consol"], rotation=20)
        ax.set_yticklabels(["Bullish", "Bearish", "Consol"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(cm[r][c]), ha="center", va="center", color="black" if cm[r][c] < np.max(cm)/2 else "white")
    plt.tight_layout()
    cm_path = eval_dir / "confusion_matrices.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # 7. Compile Final Real Metrics JSON
    results = {
        "phase": "Phase 3: Computer Vision Engine",
        "dataset": {
            "total_windows": len(manifest_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_funds": list(train_df["fund"].unique()),
            "val_funds": list(val_df["fund"].unique()),
            "test_funds": list(test_df["fund"].unique()),
            "class_distribution_overall": manifest_df["label_name"].value_counts().to_dict(),
            "class_distribution_train": train_df["label_name"].value_counts().to_dict(),
            "class_distribution_test": test_df["label_name"].value_counts().to_dict(),
            "class_weights": {LABEL_NAMES[i]: float(class_weights_vals[i]) for i in range(3)},
        },
        "models": {
            "rule_based_baseline": baseline_metrics,
            "custom_4block_cnn": {
                "parameters": 422659,
                "metrics": cnn_test_metrics,
            },
            "fine_tuned_resnet18": {
                "parameters": 11178051,
                "metrics": resnet_test_metrics,
            },
        },
        "gradcam_artifacts": gradcam_paths,
        "plots": {
            "confusion_matrices": str(cm_path),
        },
    }

    metrics_path = eval_dir / "real_metrics.json"
    def _json_serial(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(metrics_path, "w") as fp:
        json.dump(results, fp, indent=2, default=_json_serial)
    print(f"\nPhase 3 metrics successfully saved to {metrics_path}")
    return results


if __name__ == "__main__":
    run_phase3_vision_experiment()
