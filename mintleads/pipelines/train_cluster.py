"""Investor clustering pipeline for MintLeads.

This module trains a K-Means clustering model for investor segmentation:
- Elbow method for optimal k determination
- Silhouette analysis for cluster validation
- Bootstrap validation for stability
- 2D PCA visualization
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.utils import resample

from config import (
    INVESTOR_CLUSTER_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    setup_logging,
)

logger = setup_logging(__name__)


class InvestorClusterTrainer:
    """Trainer for K-Means investor clustering model."""
    
    def __init__(self) -> None:
        """Initialize the investor cluster trainer."""
        self.model: KMeans = None
        self.optimal_k: int = 3
        self.feature_names: List[str] = []
        self.X: pd.DataFrame = None
        self.labels: np.ndarray = None
        self.pca_result: np.ndarray = None
        
    def load_data(self) -> pd.DataFrame:
        """Load preprocessed investor data.
        
        Returns:
            DataFrame with processed investor data.
        """
        data_path = PROCESSED_DATA_DIR / "investors_processed.csv"
        
        if not data_path.exists():
            # Fallback to raw data
            from config import INVESTOR_BEHAVIOR_PATH
            data_path = INVESTOR_BEHAVIOR_PATH
            logger.warning(f"Processed data not found, using raw data from {data_path}")
        
        logger.info(f"Loading investor data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} investor records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering.
        
        Args:
            df: DataFrame with investor data.
            
        Returns:
            DataFrame with numeric features.
        """
        df = df.copy()
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Drop ID columns
        id_cols = ["id", "investor_id", "ID"]
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        X = df[numeric_cols].fillna(0)
        self.feature_names = numeric_cols
        
        logger.info(f"Prepared {len(self.feature_names)} features for clustering")
        logger.info(f"Features: {self.feature_names}")
        
        return X
    
    def find_optimal_k(self, k_range: range = range(2, 11)) -> Tuple[int, plt.Figure]:
        """Find optimal k using Elbow Method.
        
        Args:
            k_range: Range of k values to test.
            
        Returns:
            Tuple of (optimal_k, elbow_plot_figure).
        """
        logger.info(f"Running Elbow Method for k in {k_range}")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                random_state=RANDOM_STATE,
                n_init=10,
            )
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
            
            labels = kmeans.labels_
            sil_score = silhouette_score(self.X, labels)
            silhouette_scores.append(sil_score)
            
            logger.info(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.4f}")
        
        # Plot Elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        ax1.plot(k_range, inertias, "bo-")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Within-cluster sum of squares)")
        ax1.set_title("Elbow Method for Optimal k")
        ax1.grid(True)
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, "ro-")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Find optimal k (highest silhouette score)
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        logger.info(f"Optimal k based on silhouette score: {optimal_k}")
        
        return optimal_k, fig
    
    def plot_silhouette_analysis(self, k: int) -> plt.Figure:
        """Generate detailed silhouette analysis plot.
        
        Args:
            k: Number of clusters.
            
        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating silhouette analysis for k={k}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute silhouette scores
        silhouette_vals = silhouette_samples(self.X, self.labels)
        
        y_lower = 10
        for i in range(k):
            cluster_silhouette_vals = silhouette_vals[self.labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster
            
            color = plt.cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
            y_lower = y_upper + 10
        
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Label")
        ax.set_title(f"Silhouette Plot for k={k}")
        ax.axvline(x=silhouette_score(self.X, self.labels), color="red", linestyle="--")
        
        return fig
    
    def train_kmeans(self, k: int) -> KMeans:
        """Train K-Means model.
        
        Args:
            k: Number of clusters.
            
        Returns:
            Trained KMeans model.
        """
        logger.info(f"Training K-Means with k={k}")
        
        self.model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=RANDOM_STATE,
            n_init=10,
        )
        
        self.model.fit(self.X)
        self.labels = self.model.labels_
        
        logger.info(f"K-Means training complete. Inertia: {self.model.inertia_:.2f}")
        
        return self.model
    
    def bootstrap_validation(self, n_iterations: int = 100) -> Dict[str, Any]:
        """Validate cluster stability using bootstrap.
        
        Args:
            n_iterations: Number of bootstrap iterations.
            
        Returns:
            Dictionary with validation results.
        """
        logger.info(f"Running bootstrap validation with {n_iterations} iterations")
        
        sample_size = int(0.8 * len(self.X))
        scores = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            X_sample = resample(self.X, n_samples=sample_size, random_state=i)
            
            # Fit K-Means
            kmeans = KMeans(
                n_clusters=self.optimal_k,
                init="k-means++",
                random_state=RANDOM_STATE,
                n_init=10,
            )
            kmeans.fit(X_sample)
            
            # Calculate silhouette score
            score = silhouette_score(X_sample, kmeans.labels_)
            scores.append(score)
        
        results = {
            "mean_silhouette": np.mean(scores),
            "std_silhouette": np.std(scores),
            "min_silhouette": np.min(scores),
            "max_silhouette": np.max(scores),
        }
        
        logger.info(f"Bootstrap validation complete: mean={results['mean_silhouette']:.4f}, "
                   f"std={results['std_silhouette']:.4f}")
        
        return results
    
    def create_pca_projection(self) -> plt.Figure:
        """Create 2D PCA projection of clusters.
        
        Returns:
            Matplotlib figure.
        """
        logger.info("Creating PCA projection")
        
        # Fit PCA
        pca = PCA(n_components=2)
        self.pca_result = pca.fit_transform(self.X)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ["red", "blue", "green", "orange", "purple"]
        for i in range(self.optimal_k):
            mask = self.labels == i
            ax.scatter(
                self.pca_result[mask, 0],
                self.pca_result[mask, 1],
                c=colors[i % len(colors)],
                label=f"Cluster {i}",
                alpha=0.6,
                s=50,
            )
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_title("Investor Clusters - PCA Projection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        logger.info(f"PCA explains {sum(pca.explained_variance_ratio_):.2%} of variance")
        
        return fig
    
    def get_cluster_personas(self) -> Dict[int, str]:
        """Map cluster labels to investor personas.
        
        Returns:
            Dictionary mapping cluster IDs to persona names.
        """
        personas = {
            0: "Conservative",
            1: "Balanced",
            2: "Aggressive",
        }
        
        logger.info(f"Cluster personas: {personas}")
        return personas
    
    def save_model(self) -> Path:
        """Save the trained K-Means model.
        
        Returns:
            Path to saved model.
        """
        model_path = INVESTOR_CLUSTER_DIR / "kmeans_model.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def save_cluster_mapping(self) -> Path:
        """Save cluster-to-persona mapping.
        
        Returns:
            Path to saved mapping.
        """
        personas = self.get_cluster_personas()
        
        mapping_path = INVESTOR_CLUSTER_DIR / "cluster_mapping.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(personas, f)
        
        logger.info(f"Saved cluster mapping to {mapping_path}")
        return mapping_path
    
    def train(self) -> Dict[str, Any]:
        """Run full clustering pipeline.
        
        Returns:
            Dictionary with training results.
        """
        logger.info("=" * 50)
        logger.info("Starting Investor Cluster Training")
        logger.info("=" * 50)
        
        # Load and prepare data
        df = self.load_data()
        self.X = self.prepare_features(df)
        
        # Find optimal k
        optimal_k, elbow_fig = self.find_optimal_k()
        
        # Save elbow plot
        elbow_path = INVESTOR_CLUSTER_DIR / "elbow_plot.png"
        elbow_fig.savefig(elbow_path, dpi=300, bbox_inches="tight")
        plt.close(elbow_fig)
        logger.info(f"Saved elbow plot to {elbow_path}")
        
        # Use optimal k or default to 3
        self.optimal_k = min(optimal_k, 3)
        
        # Train K-Means
        self.train_kmeans(self.optimal_k)
        
        # Silhouette analysis
        sil_fig = self.plot_silhouette_analysis(self.optimal_k)
        sil_path = INVESTOR_CLUSTER_DIR / "silhouette_plot.png"
        sil_fig.savefig(sil_path, dpi=300, bbox_inches="tight")
        plt.close(sil_fig)
        logger.info(f"Saved silhouette plot to {sil_path}")
        
        # Bootstrap validation
        validation_results = self.bootstrap_validation()
        
        # PCA projection
        pca_fig = self.create_pca_projection()
        pca_path = INVESTOR_CLUSTER_DIR / "pca_projection.png"
        pca_fig.savefig(pca_path, dpi=300, bbox_inches="tight")
        plt.close(pca_fig)
        logger.info(f"Saved PCA projection to {pca_path}")
        
        # Save artifacts
        model_path = self.save_model()
        mapping_path = self.save_cluster_mapping()
        
        # Add cluster labels to original data
        df["cluster"] = self.labels
        output_path = PROCESSED_DATA_DIR / "investors_with_clusters.csv"
        df.to_csv(output_path, index=False)
        
        logger.info("=" * 50)
        logger.info("Investor Cluster Training Complete")
        logger.info("=" * 50)
        
        return {
            "optimal_k": self.optimal_k,
            "model_path": str(model_path),
            "cluster_mapping": str(mapping_path),
            "validation": validation_results,
            "plots": {
                "elbow": str(elbow_path),
                "silhouette": str(sil_path),
                "pca": str(pca_path),
            },
        }


def main() -> Dict[str, Any]:
    """Main training entry point."""
    trainer = InvestorClusterTrainer()
    return trainer.train()


if __name__ == "__main__":
    import json
    
    results = main()
    print(json.dumps(results, indent=2))
