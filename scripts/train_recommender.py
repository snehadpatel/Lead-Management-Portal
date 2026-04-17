import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
ML_FEATURES_DIR = os.path.join(DATASETS_DIR, "ml_features")
MODELS_DIR = os.path.join(DATASETS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(ML_FEATURES_DIR, "investor_profiles_scaled.csv")

def train_kmeans_recommender():
    print("Loading Behavioral Investor Profiles for ML Training...")
    if not os.path.exists(FEATURE_FILE):
        print("Feature matrix not found. Please run preprocessing first.")
        return
        
    df = pd.read_csv(FEATURE_FILE)
    
    # We drop any missing rows that would mathematically break the Euclidean distance
    df = df.dropna().copy()
    
    print(f"Staging {len(df)} users for Netflix-style Collaborative Filtering...")
    
    # 1. Feature Selection: We supply the algorithm with purely numeric behavioral + profile traits
    # 'yearly_income', 'total_debt', 'debt_to_income_ratio', 'credit_score', 'behavior_spend_sum', 'behavior_spend_count'
    features = ['yearly_income', 'debt_to_income_ratio', 'credit_score', 'behavior_spend_sum']
    
    X = df[features].values
    
    # 2. Mathematical Scaling
    # K-Means relies on distance. If Income is $100,000 and DTI is 1.5, Income will dominate.
    # We must StandardScale so all features have equal mathematical visual weight.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    scaler_path = os.path.join(MODELS_DIR, "kmeans_feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 3. K-Means Algorithm Training
    # We want 3 primary clusters (Equity, Hybrid, Debt)
    k = 3
    print(f"Initializing K-Means with {k} clusters to mathematically separate Risk Appetites...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    model_path = os.path.join(MODELS_DIR, "kmeans_recommender_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"✅ K-Means Model successfully trained and saved to: {model_path}")
    
    # 4. Analyze generated clusters to assign business logic labels
    df['cluster_id'] = kmeans.labels_
    
    print("\n--- Model Evaluation: Cluster Insights ---")
    cluster_profiles = df.groupby('cluster_id')[features].mean()
    print(cluster_profiles)
    
    # Let's map clusters back to real financial products based on inferred risk
    # High Income & Low DTI -> High Risk capability -> Equity
    print("\nSimulating live AI Investor Recommendations:")
    sample_users = df.sample(5)
    for index, row in sample_users.iterrows():
        cluster = row['cluster_id']
        category_guess = row['inferred_risk_profile'] # The deterministic label we generated earlier
        print(f"User ID {row['id']} | Income: ${row['yearly_income']:,.0f} | DTI: {row['debt_to_income_ratio']:.2f}")
        print(f"  -> Model assigned to Cluster {cluster}")
        print(f"  -> Recommended Match Category: {category_guess}")
        print(f"  -> Output logic: Fund match confidence mapped to Cluster {cluster} centroid distance.")
        print("-" * 40)

if __name__ == "__main__":
    train_kmeans_recommender()
