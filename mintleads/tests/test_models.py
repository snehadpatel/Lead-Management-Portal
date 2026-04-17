"""Tests for trained models.

This module contains unit tests for the ML models.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

# Optional imports
try:
    import torch
    from pipelines.train_lstm import NAVForecaster
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    NAVForecaster = None

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


class TestModels:
    """Test cases for ML models."""
    
    def test_rf_predict_proba_range(self):
        """Test RF predict_proba returns values in [0, 1]."""
        # Create simple RF model
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test predictions
        X_test = np.random.randn(10, 5)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (10, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_kmeans_returns_valid_clusters(self):
        """Test K-Means returns cluster in {0, 1, 2}."""
        X = np.random.randn(100, 3)
        
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        assert all(label in [0, 1, 2] for label in labels)
        assert len(set(labels)) <= 3
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_lstm_output_shape(self):
        """Test LSTM forecast output has correct shape."""
        model = NAVForecaster(input_size=1, hidden_size=64, num_layers=2)
        model.eval()
        
        # Create dummy input (batch=1, seq_len=60, features=1)
        x = torch.randn(1, 60, 1)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be (batch, 1)
        assert output.shape == (1, 1)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_model_save_and_load(self, tmp_path):
        """Test saving and loading models."""
        # Create and save model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1], 50)
        model.fit(X, y)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Load model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        
        # Test loaded model
        X_test = np.random.randn(5, 3)
        pred1 = model.predict(X_test)
        pred2 = loaded_model.predict(X_test)
        
        assert np.array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
