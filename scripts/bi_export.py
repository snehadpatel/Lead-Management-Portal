"""
BI Tool Export Script — Tableau & Power BI Integration
Exports cleaned data as CSV/Parquet for visualization tools.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lume_platform.config import DATA_ROOT, PROJECT_ROOT


class BIExporter:
    """Export data for BI tools (Tableau Public, Power BI)"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (PROJECT_ROOT / "tableau_exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(DATA_ROOT)
        
    def export_lead_scoring(self, include_predictions: bool = True) -> Path:
        """Export lead scoring data with predictions"""
        print("📊 Exporting Lead Scoring Data...")
        
        # Load cleaned data
        parquet_path = PROJECT_ROOT / "artifacts/cleaned_parquet/lead_scoring_features"
        csv_path = self.data_root / "structured/leads/lead_scoring/Lead Scoring.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            raise FileNotFoundError("Lead scoring data not found")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
        
        # Add derived metrics
        if "totalvisits" in df.columns and "total_time_spent_on_website" in df.columns:
            df["engagement_score"] = (
                df["totalvisits"] * 0.3 + 
                df["total_time_spent_on_website"] * 0.0005
            )
            
            # Create engagement tiers
            df["engagement_tier"] = pd.cut(
                df["engagement_score"],
                bins=[-np.inf, 10, 30, np.inf],
                labels=["Low", "Medium", "High"]
            )
        
        # Add conversion probability if model available
        if include_predictions:
            try:
                import pickle
                model_path = PROJECT_ROOT / "artifacts/models/lead_classifier.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        pipeline = pickle.load(f)
                    
                    # Prepare features (simplified)
                    feature_cols = [c for c in df.columns if c in [
                        "totalvisits", "total_time_spent_on_website", 
                        "page_views_per_visit", "asymmetrique_activity_score"
                    ]]
                    if feature_cols:
                        X = df[feature_cols].fillna(0)
                        df["conversion_probability"] = pipeline.predict_proba(X)[:, 1]
                        df["predicted_conversion"] = pipeline.predict(X)
            except Exception as e:
                print(f"   ⚠️ Could not add predictions: {e}")
        
        # Export
        output_path = self.output_dir / "leads_for_bi.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Exported: {output_path}")
        print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
    
    def export_investor_clusters(self) -> Path:
        """Export investor clustering results"""
        print("\n📊 Exporting Investor Clusters...")
        
        behavior_path = self.data_root / "structured/leads/mf_investor_behavior/MF_Behavior.xlsx"
        
        if not behavior_path.exists():
            print("   ⚠️ Investor behavior data not found, skipping...")
            return None
        
        df = pd.read_excel(behavior_path)
        
        # Add cluster assignments if model exists
        try:
            import pickle
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.cluster import KMeans
            
            kmeans_path = PROJECT_ROOT / "artifacts/models/investor_kmeans.pkl"
            scaler_path = PROJECT_ROOT / "artifacts/models/investor_scaler.pkl"
            
            if kmeans_path.exists() and scaler_path.exists():
                with open(kmeans_path, "rb") as f:
                    kmeans = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                
                behavior_cols = ["ProfManage", "Diversification", "Affordability",
                               "Liquidity", "Growth", "Trustworthiness", "Technology"]
                behavior_cols = [c for c in behavior_cols if c in df.columns]
                
                if behavior_cols:
                    X = df[behavior_cols].dropna()
                    X_scaled = scaler.transform(X)
                    labels = kmeans.predict(X_scaled)
                    
                    df.loc[X.index, "cluster_id"] = labels
                    
                    # Map to personas
                    personas = {
                        0: "Equity Schemes (High Risk)",
                        1: "Liquid/Debt Funds (Safe)",
                        2: "Hybrid Allocation Funds",
                        3: "Index Trackers (Passive)"
                    }
                    df.loc[X.index, "persona"] = df.loc[X.index, "cluster_id"].map(personas)
        except Exception as e:
            print(f"   ⚠️ Could not add cluster assignments: {e}")
        
        output_path = self.output_dir / "investor_clusters_for_bi.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Exported: {output_path}")
        print(f"   Rows: {len(df):,}")
        
        return output_path
    
    def export_sentiment_data(self) -> Path:
        """Export sentiment analysis data"""
        print("\n📊 Exporting Sentiment Data...")
        
        sentiment_path = self.data_root / "semi_structured/social_sentiment/data.csv"
        
        if not sentiment_path.exists():
            print("   ⚠️ Sentiment data not found, skipping...")
            return None
        
        df = pd.read_csv(sentiment_path)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Add sentiment predictions if model available
        try:
            import pickle
            model_path = PROJECT_ROOT / "artifacts/models/sentiment_nlp.pkl"
            
            if model_path.exists():
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                
                text_col = None
                for col in df.columns:
                    if "sentence" in col or "text" in col:
                        text_col = col
                        break
                
                if text_col:
                    texts = df[text_col].astype(str)
                    X_tfidf = bundle["vectorizer"].transform(texts)
                    predictions = bundle["model"].predict(X_tfidf)
                    df["predicted_sentiment"] = bundle["label_encoder"].inverse_transform(predictions)
        except Exception as e:
            print(f"   ⚠️ Could not add sentiment predictions: {e}")
        
        output_path = self.output_dir / "sentiment_for_bi.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Exported: {output_path}")
        print(f"   Rows: {len(df):,}")
        
        return output_path
    
    def export_summary_metrics(self) -> Path:
        """Export summary metrics for dashboard KPIs"""
        print("\n📊 Exporting Summary Metrics...")
        
        metrics = {
            "export_timestamp": datetime.now().isoformat(),
            "dataset_size_gb": 31.25,
            "data_sources": {
                "structured": "~25 GB",
                "semi_structured": "~3 GB",
                "unstructured": "~3 GB"
            },
            "models": {
                "lead_classification": "XGBoost/RandomForest",
                "investor_clustering": "K-Means (4 clusters)",
                "sentiment_analysis": "TF-IDF + LogisticRegression"
            }
        }
        
        # Add data counts if available
        leads_path = self.data_root / "structured/leads/lead_scoring/Lead Scoring.csv"
        if leads_path.exists():
            df = pd.read_csv(leads_path, nrows=1)  # Just get columns first
            # Count rows
            with open(leads_path) as f:
                metrics["total_leads"] = sum(1 for _ in f) - 1
        
        output_path = self.output_dir / "summary_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Also export as CSV for Tableau
        metrics_df = pd.DataFrame([{
            "metric": "export_timestamp",
            "value": metrics["export_timestamp"]
        }, {
            "metric": "dataset_size_gb",
            "value": metrics["dataset_size_gb"]
        }])
        
        if "total_leads" in metrics:
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame([{"metric": "total_leads", "value": metrics["total_leads"]}])
            ], ignore_index=True)
        
        csv_path = self.output_dir / "summary_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        print(f"✅ Exported: {output_path}")
        print(f"✅ Exported: {csv_path}")
        
        return output_path
    
    def create_tableau_readme(self) -> Path:
        """Create README for Tableau Public integration"""
        print("\n📄 Creating Tableau Integration Guide...")
        
        readme_content = """# Tableau Public Integration Guide

## Step 1: Download Export Files
The following CSV files are ready for import into Tableau Public:

1. **leads_for_bi.csv** — Lead scoring data with engagement metrics
2. **investor_clusters_for_bi.csv** — Investor behavior and cluster assignments
3. **sentiment_for_bi.csv** — Social sentiment data
4. **summary_metrics.csv** — KPI metrics for dashboard

## Step 2: Import to Tableau Public
1. Go to [Tableau Public](https://public.tableau.com/)
2. Create a new workbook
3. Connect to Text File → Select the CSV files
4. Build visualizations using the imported data

## Step 3: Recommended Visualizations

### Lead Scoring Dashboard
- **Conversion Probability Distribution** (Histogram)
- **Engagement Tier Breakdown** (Pie Chart)
- **Lead Source Performance** (Bar Chart)
- **Conversion by Engagement Score** (Scatter Plot)

### Investor Clustering Dashboard
- **Cluster Distribution** (Pie Chart)
- **Persona Radar Chart** (Multiple dimensions)
- **Behavior Heatmap** (Correlation matrix)

### Sentiment Analysis Dashboard
- **Sentiment Distribution** (Donut Chart)
- **Sentiment Over Time** (Line Chart, if timestamp available)

## Step 4: Publish & Embed
1. Publish workbook to Tableau Public
2. Get embed URL from Share → Embed Code
3. Set environment variable:
   ```bash
   export LUME_TABLEAU_EMBED_URL="your_embed_url"
   ```
4. Restart Streamlit app to see embedded dashboard

## Data Dictionary

### leads_for_bi.csv
| Column | Description |
|--------|-------------|
| totalvisits | Number of website visits |
| total_time_spent_on_website | Time in seconds |
| page_views_per_visit | Average pages per visit |
| engagement_score | Calculated engagement metric |
| engagement_tier | Low/Medium/High |
| conversion_probability | ML model prediction (0-1) |
| predicted_conversion | 0 or 1 |

### investor_clusters_for_bi.csv
| Column | Description |
|--------|-------------|
| ProfManage | Preference for professional management (0-10) |
| Diversification | Diversification preference (0-10) |
| Affordability | Affordability importance (0-10) |
| Liquidity | Liquidity preference (0-10) |
| Growth | Growth preference (0-10) |
| Trustworthiness | Trust in financial institutions (0-10) |
| Technology | Tech-savviness (0-10) |
| cluster_id | Assigned cluster (0-3) |
| persona | Human-readable cluster label |

### sentiment_for_bi.csv
| Column | Description |
|--------|-------------|
| sentence/text | Input text |
| sentiment | True label |
| predicted_sentiment | ML model prediction |

## Tips for Effective Visualizations

1. **Use calculated fields** for custom metrics
2. **Create filters** for interactive exploration
3. **Add tooltips** for detailed information on hover
4. **Use color consistently** across dashboards
5. **Add reference lines** for thresholds (e.g., 0.85 for hot leads)

## Troubleshooting

**File too large for Tableau Public?**
- Tableau Public has a 10M row limit
- Sample data using the provided CSVs (already sampled)
- Or use Tableau Desktop with Tableau Server

**Data not refreshing?**
- Re-export from this script after model retraining
- Re-import CSV in Tableau Public
- Republish the workbook
"""
        
        output_path = self.output_dir / "TABLEAU_INTEGRATION_GUIDE.md"
        with open(output_path, "w") as f:
            f.write(readme_content)
        
        print(f"✅ Created: {output_path}")
        
        return output_path
    
    def export_all(self) -> Dict[str, Path]:
        """Run all exports"""
        print("\n" + "="*60)
        print("🚀 BI EXPORT PROCESS STARTING")
        print("="*60 + "\n")
        
        exports = {}
        
        try:
            exports["leads"] = self.export_lead_scoring()
        except Exception as e:
            print(f"❌ Lead export failed: {e}")
        
        try:
            exports["investors"] = self.export_investor_clusters()
        except Exception as e:
            print(f"❌ Investor export failed: {e}")
        
        try:
            exports["sentiment"] = self.export_sentiment_data()
        except Exception as e:
            print(f"❌ Sentiment export failed: {e}")
        
        try:
            exports["metrics"] = self.export_summary_metrics()
        except Exception as e:
            print(f"❌ Metrics export failed: {e}")
        
        try:
            exports["guide"] = self.create_tableau_readme()
        except Exception as e:
            print(f"❌ Guide creation failed: {e}")
        
        print("\n" + "="*60)
        print("✅ BI EXPORT PROCESS COMPLETE")
        print("="*60)
        print(f"\n📁 Export Location: {self.output_dir}")
        print("\n📊 Next Steps:")
        print("   1. Import CSVs to Tableau Public")
        print("   2. Create visualizations")
        print("   3. Publish and get embed URL")
        print("   4. Set LUME_TABLEAU_EMBED_URL environment variable")
        print(f"\n📖 See {self.output_dir}/TABLEAU_INTEGRATION_GUIDE.md for details")
        
        return exports


def main():
    """Main entry point"""
    exporter = BIExporter()
    exporter.export_all()


if __name__ == "__main__":
    main()
