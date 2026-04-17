"""NAV Forecaster training pipeline for MintLeads.

This module trains an LSTM model for NAV prediction:
- PyTorch LSTM with BatchNorm
- Sliding window sequences
- Optuna hyperparameter tuning
- MLflow logging
- Early stopping and LR scheduling
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_WINDOW_SIZE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NAV_FORECASTER_DIR,
    NAV_HISTORY_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    setup_logging,
)

logger = setup_logging(__name__)

# Set random seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class NAVForecaster(nn.Module):
    """LSTM-based NAV forecasting model."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the NAV Forecaster.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super(NAVForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch, 1).
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Batch norm
        normalized = self.batch_norm(last_hidden)
        
        # Linear layer
        output = self.fc(normalized)
        
        return output


class NAVDataset(Dataset):
    """Dataset for NAV time-series."""
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 60,
    ) -> None:
        """Initialize NAV dataset.
        
        Args:
            data: NAV values array.
            window_size: Sliding window size.
        """
        self.data = data
        self.window_size = window_size
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index.
        
        Args:
            idx: Index.
            
        Returns:
            Tuple of (input_sequence, target_value).
        """
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size]
        
        return (
            torch.FloatTensor(x).unsqueeze(-1),  # (window_size, 1)
            torch.FloatTensor([y]),  # (1,)
        )


class NAVForecasterTrainer:
    """Trainer for LSTM NAV forecasting model."""
    
    def __init__(self, scheme_code: str = "sample") -> None:
        """Initialize the NAV forecaster trainer.
        
        Args:
            scheme_code: Mutual fund scheme code.
        """
        self.scheme_code = scheme_code
        self.model: NAVForecaster = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_nav_data(self) -> np.ndarray:
        """Load NAV data for training.
        
        Returns:
            Array of NAV values.
        """
        # Try processed data first
        processed_path = PROCESSED_DATA_DIR / "nav_processed.csv"
        
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            # Filter for specific scheme
            scheme_data = df[df["scheme_code"] == self.scheme_code]
            if len(scheme_data) > 0:
                nav_values = scheme_data["nav_scaled"].values
                logger.info(f"Loaded {len(nav_values)} NAV values for {self.scheme_code}")
                return nav_values
        
        # Fallback to individual NAV file
        nav_file = NAV_HISTORY_DIR / f"{self.scheme_code}.csv"
        if nav_file.exists():
            df = pd.read_csv(nav_file)
            df.columns = [col.lower().strip() for col in df.columns]
            if "nav" in df.columns:
                # MinMax scale
                nav_values = df["nav"].values
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                nav_scaled = scaler.fit_transform(nav_values.reshape(-1, 1)).flatten()
                logger.info(f"Loaded and scaled {len(nav_scaled)} NAV values")
                return nav_scaled
        
        # Generate synthetic data if no real data
        logger.warning("No NAV data found, generating synthetic data")
        np.random.seed(RANDOM_STATE)
        synthetic = np.cumsum(np.random.randn(1000) * 0.01) + 100
        # MinMax scale
        synthetic = (synthetic - synthetic.min()) / (synthetic.max() - synthetic.min())
        return synthetic
    
    def create_dataloaders(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> None:
        """Create train/val/test dataloaders.
        
        Args:
            data: NAV values array.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
        """
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Temporal split (no shuffling)
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Create datasets
        train_dataset = NAVDataset(train_data, LSTM_WINDOW_SIZE)
        val_dataset = NAVDataset(val_data, LSTM_WINDOW_SIZE)
        test_dataset = NAVDataset(test_data, LSTM_WINDOW_SIZE)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=LSTM_BATCH_SIZE,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=LSTM_BATCH_SIZE,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=LSTM_BATCH_SIZE,
            shuffle=False,
        )
    
    def train_epoch(self, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch.
        
        Args:
            optimizer: Optimizer instance.
            criterion: Loss function.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self, criterion: nn.Module) -> float:
        """Validate model.
        
        Args:
            criterion: Loss function.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train_model(
        self,
        hidden_size: int = 128,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        patience: int = 3,
    ) -> Dict[str, Any]:
        """Train the LSTM model.
        
        Args:
            hidden_size: LSTM hidden size.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            patience: Early stopping patience.
            
        Returns:
            Dictionary with training results.
        """
        logger.info(f"Training LSTM: hidden_size={hidden_size}, lr={learning_rate}")
        
        # Initialize model
        self.model = NAVForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=LSTM_DROPOUT,
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(LSTM_EPOCHS):
            train_loss = self.train_epoch(optimizer, criterion)
            val_loss = self.validate(criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{LSTM_EPOCHS}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_path = NAV_FORECASTER_DIR / f"lstm_best_{self.scheme_code}.pt"
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        best_model_path = NAV_FORECASTER_DIR / f"lstm_best_{self.scheme_code}.pt"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    
    def save_loss_plot(self) -> Path:
        """Save training/validation loss curve.
        
        Returns:
            Path to saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.train_losses, label="Training Loss", color="blue")
        ax.plot(self.val_losses, label="Validation Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"LSTM Training Loss - {self.scheme_code}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = NAV_FORECASTER_DIR / f"lstm_loss_curve_{self.scheme_code}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved loss curve to {plot_path}")
        return plot_path
    
    def save_model(self) -> Path:
        """Save the trained model.
        
        Returns:
            Path to saved model.
        """
        model_path = NAV_FORECASTER_DIR / "lstm_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_size": 1,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": LSTM_DROPOUT,
            },
        }, model_path)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def train(self, use_optuna: bool = True) -> Dict[str, Any]:
        """Run full training pipeline.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter tuning.
            
        Returns:
            Dictionary with training results.
        """
        logger.info("=" * 50)
        logger.info(f"Starting NAV Forecaster Training - {self.scheme_code}")
        logger.info("=" * 50)
        
        # Load data
        data = self.load_nav_data()
        self.create_dataloaders(data)
        
        if use_optuna:
            # Optuna hyperparameter tuning
            logger.info("Starting Optuna hyperparameter tuning")
            
            def objective(trial):
                hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
                lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                
                results = self.train_model(
                    hidden_size=hidden_size,
                    learning_rate=lr,
                    patience=3,
                )
                return results["best_val_loss"]
            
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=50)
            
            best_params = study.best_params
            logger.info(f"Best Optuna params: {best_params}")
            
            # Train with best params
            self.train_model(
                hidden_size=best_params["hidden_size"],
                learning_rate=best_params["lr"],
            )
        else:
            # Train with default params
            self.train_model(hidden_size=LSTM_HIDDEN_SIZE)
        
        # Evaluate
        metrics = self.evaluate()
        logger.info(f"Test metrics: {metrics}")
        
        # Save artifacts
        loss_plot = self.save_loss_plot()
        model_path = self.save_model()
        
        logger.info("=" * 50)
        logger.info("NAV Forecaster Training Complete")
        logger.info("=" * 50)
        
        return {
            "model_path": str(model_path),
            "loss_plot": str(loss_plot),
            "metrics": metrics,
        }


def main(scheme_code: str = "sample") -> Dict[str, Any]:
    """Main training entry point.
    
    Args:
        scheme_code: Mutual fund scheme code.
        
    Returns:
        Dictionary with training results.
    """
    trainer = NAVForecasterTrainer(scheme_code)
    return trainer.train()


if __name__ == "__main__":
    import json
    import sys
    
    scheme = sys.argv[1] if len(sys.argv) > 1 else "sample"
    results = main(scheme)
    print(json.dumps(results, indent=2))
