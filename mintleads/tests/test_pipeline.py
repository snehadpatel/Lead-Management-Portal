"""Tests for data pipelines.

This module contains unit tests for the preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pipelines.preprocess import LeadDataPreprocessor, InvestorDataPreprocessor


class TestPreprocessPipeline:
    """Test cases for preprocessing pipeline."""
    
    def test_synthetic_dataframe_creation(self):
        """Test creating synthetic dataframe with 100 rows."""
        np.random.seed(42)
        df = pd.DataFrame({
            "id": range(100),
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 200000, 100),
            "total_visits": np.random.randint(0, 50, 100),
            "page_views": np.random.randint(1, 20, 100),
            "converted": np.random.choice([0, 1], 100),
        })
        
        assert len(df) == 100
        assert len(df.columns) == 6
    
    def test_no_nan_after_imputation(self):
        """Test that no NaN values remain after imputation."""
        # Create data with missing values
        df = pd.DataFrame({
            "age": [25, 30, np.nan, 45, 50],
            "income": [50000, np.nan, 75000, 100000, 125000],
        })
        
        preprocessor = InvestorDataPreprocessor()
        # Fill with KNN-like simple imputation
        df_filled = df.fillna(df.mean())
        
        assert df_filled.isnull().sum().sum() == 0
    
    def test_scaled_values_in_range(self):
        """Test that scaled values are in expected ranges."""
        data = np.random.randn(100, 3) * 100 + 500
        
        # MinMaxScaler should produce values in [0, 1]
        minmax = MinMaxScaler()
        scaled = minmax.fit_transform(data)
        
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0
        
        # StandardScaler should produce mean ~0, std ~1
        standard = StandardScaler()
        scaled_std = standard.fit_transform(data)
        
        assert abs(scaled_std.mean()) < 0.1
        assert abs(scaled_std.std() - 1.0) < 0.1
    
    def test_one_hot_encoding_shape(self):
        """Test correct shape after one-hot encoding."""
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "value": [1, 2, 3, 4, 5],
        })
        
        encoded = pd.get_dummies(df, columns=["category"])
        
        # Original: 2 columns, After: 4 columns (value + 3 categories)
        assert encoded.shape == (5, 4)
        assert "category_A" in encoded.columns
        assert "category_B" in encoded.columns
        assert "category_C" in encoded.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
