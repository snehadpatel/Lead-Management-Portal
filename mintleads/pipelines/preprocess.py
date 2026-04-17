"""Data preprocessing pipeline for MintLeads.

This module provides comprehensive preprocessing for all MintLeads datasets:
- Lead scoring data: cleaning, encoding, feature engineering
- Investor behavior data: imputation, scaling, clustering features
- NAV time-series: forward-fill, scaling, sequence generation

All preprocessing steps are idempotent and logged.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from config import (
    INVESTOR_BEHAVIOR_PATH,
    LEAD_SCORER_DIR,
    LEADS_DATA_PATH,
    NAV_FORECASTER_DIR,
    NAV_HISTORY_DIR,
    PROCESSED_DATA_DIR,
    INVESTOR_CLUSTER_DIR,
    setup_logging,
)

logger = setup_logging(__name__)


class LeadDataPreprocessor:
    """Preprocessor for lead scoring data."""
    
    def __init__(self) -> None:
        """Initialize the lead data preprocessor."""
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.target_means: Dict[str, float] = {}
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load raw lead data from CSV.
        
        Args:
            file_path: Path to lead data CSV.
            
        Returns:
            DataFrame with raw lead data.
        """
        logger.info(f"Loading lead data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df.copy()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the lead data.
        
        - Removes duplicates
        - Handles missing values
        - Winsorizes outliers
        
        Args:
            df: Raw lead DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        
        # MD5 hash deduplication
        logger.info("Computing MD5 hashes for deduplication")
        df["_row_hash"] = df.apply(
            lambda row: hashlib.md5(
                str(row.values).encode()
            ).hexdigest(),
            axis=1,
        )
        initial_count = len(df)
        df = df.drop_duplicates(subset=["_row_hash"], keep="first")
        df = df.drop(columns=["_row_hash"])
        logger.info(f"Removed {initial_count - len(df)} duplicate rows")
        
        # Winsorize TotalVisits at 99th percentile
        if "TotalVisits" in df.columns:
            upper_limit = df["TotalVisits"].quantile(0.99)
            df["TotalVisits"] = df["TotalVisits"].clip(upper=upper_limit)
            logger.info(f"Winsorized TotalVisits at {upper_limit}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features for lead scoring.
        
        Args:
            df: Cleaned lead DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()
        
        # EngagementVelocity: Page Views per Visit / Total Time Spent
        if "Page Views Per Visit" in df.columns and "Total Time Spent on Website" in df.columns:
            df["EngagementVelocity"] = df.apply(
                lambda row: (
                    row["Page Views Per Visit"] / row["Total Time Spent on Website"]
                    if row["Total Time Spent on Website"] > 0 else 0
                ),
                axis=1,
            )
            logger.info("Created EngagementVelocity feature")
        
        # ActivityScore: combination of TotalVisits and Page Views
        if "TotalVisits" in df.columns and "Page Views Per Visit" in df.columns:
            df["ActivityScore"] = df["TotalVisits"] * df["Page Views Per Visit"]
            logger.info("Created ActivityScore feature")
        
        # LeadAgeBin: categorize leads by activity
        if "TotalVisits" in df.columns:
            df["LeadAgeBin"] = pd.cut(
                df["TotalVisits"],
                bins=[-1, 0, 5, 15, float("inf")],
                labels=["New", "Active", "Engaged", "Hot"],
            )
            logger.info("Created LeadAgeBin feature")
        
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.
        
        - One-Hot encode: LeadOrigin, SchemeCategory, Country
        - Label encode: RiskLevel
        - Target encode: LeadSource
        
        Args:
            df: DataFrame with features to encode.
            
        Returns:
            DataFrame with encoded features.
        """
        df = df.copy()
        
        # One-Hot encode categorical columns
        one_hot_cols = []
        for col in ["Lead Origin", "Country"]:
            if col in df.columns:
                one_hot_cols.append(col)
        
        if one_hot_cols:
            logger.info(f"One-hot encoding: {one_hot_cols}")
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(df[one_hot_cols])
            encoded_cols = encoder.get_feature_names_out(one_hot_cols)
            
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=one_hot_cols), df_encoded], axis=1)
            
            self.encoders["one_hot"] = encoder
            self.encoders["one_hot_cols"] = one_hot_cols
        
        # Label encode RiskLevel
        if "Lead Quality" in df.columns:
            logger.info("Label encoding Lead Quality")
            le = LabelEncoder()
            df["LeadQualityEncoded"] = le.fit_transform(df["Lead Quality"].astype(str))
            self.encoders["label_lead_quality"] = le
        
        # Target encode LeadSource
        if "Lead Source" in df.columns and "Converted" in df.columns:
            logger.info("Target encoding Lead Source")
            target_means = df.groupby("Lead Source")["Converted"].mean()
            df["LeadSourceTargetEnc"] = df["Lead Source"].map(target_means)
            self.target_means = target_means.to_dict()
        
        return df
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using KNN for Age and Income.
        
        Args:
            df: DataFrame with missing values.
            
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        # For lead scoring dataset, impute numeric columns if they exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols and df[numeric_cols].isnull().sum().sum() > 0:
            logger.info(f"Imputing missing values in {len(numeric_cols)} numeric columns")
            
            # Fill columns with all NaN with 0
            for col in numeric_cols:
                if df[col].isnull().all():
                    df[col] = df[col].fillna(0)
            
            # Use KNN imputation for remaining missing values
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            self.scalers["knn_imputer"] = imputer
            logger.info("Completed KNN imputation")
        
        return df
    
    def save_artifacts(self) -> None:
        """Save all fitted encoders and scalers."""
        artifacts = {
            "encoders": self.encoders,
            "target_means": self.target_means,
        }
        
        artifact_path = LEAD_SCORER_DIR / "preprocessing_artifacts.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Saved preprocessing artifacts to {artifact_path}")
    
    def process(self, file_path: Path) -> pd.DataFrame:
        """Run full preprocessing pipeline on lead data.
        
        Args:
            file_path: Path to raw lead data.
            
        Returns:
            Preprocessed DataFrame.
        """
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.impute_missing(df)
        df = self.engineer_features(df)
        df = self.encode_features(df)
        self.save_artifacts()
        
        # Save processed data
        output_path = PROCESSED_DATA_DIR / "leads_processed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed leads to {output_path}")
        
        return df


class InvestorDataPreprocessor:
    """Preprocessor for investor behavior data."""
    
    def __init__(self) -> None:
        """Initialize the investor data preprocessor."""
        self.scaler: Optional[StandardScaler] = None
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load investor behavior data.
        
        Args:
            file_path: Path to investor data CSV.
            
        Returns:
            DataFrame with investor data.
        """
        logger.info(f"Loading investor data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} investor records")
        return df.copy()
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing Age and Income using KNN.
        
        Args:
            df: DataFrame with investor data.
            
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        # Identify numeric columns for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            missing_count = df[numeric_cols].isnull().sum().sum()
            if missing_count > 0:
                logger.info(f"Imputing {missing_count} missing values with KNN (k=5)")
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                logger.info("Completed KNN imputation")
            else:
                logger.info("No missing values found")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler for clustering.
        
        Args:
            df: DataFrame with numeric features.
            
        Returns:
            DataFrame with scaled features.
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            logger.info(f"Scaling {len(numeric_cols)} features with StandardScaler")
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            
            # Save scaler
            scaler_path = INVESTOR_CLUSTER_DIR / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {scaler_path}")
        
        return df
    
    def process(self, file_path: Path) -> pd.DataFrame:
        """Run full preprocessing pipeline on investor data.
        
        Args:
            file_path: Path to raw investor data.
            
        Returns:
            Preprocessed DataFrame.
        """
        df = self.load_data(file_path)
        df = self.impute_missing(df)
        df = self.scale_features(df)
        
        # Save processed data
        output_path = PROCESSED_DATA_DIR / "investors_processed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed investors to {output_path}")
        
        return df


class NAVDataPreprocessor:
    """Preprocessor for NAV time-series data."""
    
    def __init__(self) -> None:
        """Initialize the NAV data preprocessor."""
        self.scalers: Dict[str, MinMaxScaler] = {}
        
    def load_nav_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single NAV history file.
        
        Args:
            file_path: Path to NAV CSV file.
            
        Returns:
            DataFrame with NAV data.
        """
        df = pd.read_csv(file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Standardize column names
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        return df
    
    def forward_fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing values for market holidays.
        
        Args:
            df: DataFrame with NAV time-series.
            
        Returns:
            DataFrame with filled gaps.
        """
        df = df.copy()
        
        if "date" in df.columns and "nav" in df.columns:
            # Ensure date is sorted
            df = df.sort_values("date")
            
            # Create complete date range
            date_range = pd.date_range(
                start=df["date"].min(),
                end=df["date"].max(),
                freq="D",
            )
            
            # Reindex and forward fill
            df = df.set_index("date").reindex(date_range)
            df["nav"] = df["nav"].fillna(method="ffill")
            df = df.reset_index()
            df = df.rename(columns={"index": "date"})
            
            logger.info(f"Forward-filled {len(date_range) - len(df.dropna())} gaps")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer NAV features: SMA-7 and SMA-30.
        
        Args:
            df: DataFrame with NAV data.
            
        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()
        
        if "nav" in df.columns:
            # Simple Moving Average 7-day
            df["NAV_SMA_7"] = df["nav"].rolling(window=7, min_periods=1).mean()
            
            # Simple Moving Average 30-day
            df["NAV_SMA_30"] = df["nav"].rolling(window=30, min_periods=1).mean()
            
            logger.info("Created NAV_SMA_7 and NAV_SMA_30 features")
        
        return df
    
    def scale_nav(self, df: pd.DataFrame, scheme_code: str) -> pd.DataFrame:
        """Scale NAV using MinMaxScaler [0, 1].
        
        Args:
            df: DataFrame with NAV data.
            scheme_code: Scheme code for scaler naming.
            
        Returns:
            DataFrame with scaled NAV.
        """
        df = df.copy()
        
        if "nav" in df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df["nav_scaled"] = scaler.fit_transform(df[["nav"]])
            self.scalers[scheme_code] = scaler
            
            logger.info(f"Scaled NAV for scheme {scheme_code}")
        
        return df
    
    def process_all_nav(self, nav_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process all NAV files in directory.
        
        Args:
            nav_dir: Directory containing NAV CSV files.
            
        Returns:
            Dictionary mapping scheme codes to processed DataFrames.
        """
        processed = {}
        
        nav_files = list(nav_dir.glob("*.csv"))
        logger.info(f"Found {len(nav_files)} NAV files to process")
        
        for i, nav_file in enumerate(nav_files):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(nav_files)} NAV files")
            
            try:
                scheme_code = nav_file.stem
                df = self.load_nav_file(nav_file)
                df = self.forward_fill_gaps(df)
                df = self.engineer_features(df)
                df = self.scale_nav(df, scheme_code)
                df["scheme_code"] = scheme_code
                
                processed[scheme_code] = df
            except Exception as e:
                logger.error(f"Error processing {nav_file}: {e}")
                continue
        
        # Save all scalers
        scalers_path = NAV_FORECASTER_DIR / "nav_scalers.pkl"
        with open(scalers_path, "wb") as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved {len(self.scalers)} NAV scalers to {scalers_path}")
        
        # Combine and save
        if processed:
            combined = pd.concat(processed.values(), ignore_index=True)
            output_path = PROCESSED_DATA_DIR / "nav_processed.csv"
            combined.to_csv(output_path, index=False)
            logger.info(f"Saved processed NAV data to {output_path}")
        
        return processed


class StockDataPreprocessor:
    """Preprocessor for stock market data."""
    
    def __init__(self) -> None:
        """Initialize the stock data preprocessor."""
        self.volume_scaler: Optional[RobustScaler] = None
    
    def scale_volume(self, df: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
        """Scale volume using RobustScaler.
        
        Args:
            df: DataFrame with stock data.
            volume_col: Name of volume column.
            
        Returns:
            DataFrame with scaled volume.
        """
        df = df.copy()
        
        if volume_col in df.columns:
            self.volume_scaler = RobustScaler()
            df[f"{volume_col}_scaled"] = self.volume_scaler.fit_transform(df[[volume_col]])
            logger.info(f"Scaled volume column {volume_col}")
        
        return df


def run_full_preprocessing() -> Dict[str, Any]:
    """Run complete preprocessing pipeline for all data types.
    
    Returns:
        Dictionary with preprocessing results.
    """
    results = {}
    
    # Preprocess lead data
    logger.info("=" * 50)
    logger.info("Starting Lead Data Preprocessing")
    logger.info("=" * 50)
    
    lead_preprocessor = LeadDataPreprocessor()
    leads_df = lead_preprocessor.process(LEADS_DATA_PATH)
    results["leads"] = {
        "records": len(leads_df),
        "columns": len(leads_df.columns),
        "output": str(PROCESSED_DATA_DIR / "leads_processed.csv"),
    }
    
    # Preprocess investor data
    logger.info("=" * 50)
    logger.info("Starting Investor Data Preprocessing")
    logger.info("=" * 50)
    
    investor_preprocessor = InvestorDataPreprocessor()
    investors_df = investor_preprocessor.process(INVESTOR_BEHAVIOR_PATH)
    results["investors"] = {
        "records": len(investors_df),
        "columns": len(investors_df.columns),
        "output": str(PROCESSED_DATA_DIR / "investors_processed.csv"),
    }
    
    # Preprocess NAV data
    logger.info("=" * 50)
    logger.info("Starting NAV Data Preprocessing")
    logger.info("=" * 50)
    
    nav_preprocessor = NAVDataPreprocessor()
    nav_data = nav_preprocessor.process_all_nav(NAV_HISTORY_DIR)
    results["nav"] = {
        "schemes": len(nav_data),
        "output": str(PROCESSED_DATA_DIR / "nav_processed.csv"),
    }
    
    logger.info("=" * 50)
    logger.info("Preprocessing Complete")
    logger.info("=" * 50)
    
    return results


if __name__ == "__main__":
    import json
    
    results = run_full_preprocessing()
    print(json.dumps(results, indent=2))
