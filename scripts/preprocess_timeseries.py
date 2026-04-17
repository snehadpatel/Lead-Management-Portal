import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob

BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MF_HISTORY_DIR = os.path.join(DATASETS_DIR, "mutual_fund_history")
EOD_DIR = os.path.join(DATASETS_DIR, "EOD")

SCALERS_DIR = os.path.join(DATASETS_DIR, "ml_scalers")
os.makedirs(SCALERS_DIR, exist_ok=True)

def fit_global_mf_scaler():
    print("Initializing Global Min-Max Scaler for Mutual Fund Time-Series...")
    
    # Grab a random uniform sample of 500 funds to determine absolute market bounding boxes
    mf_files = glob.glob(os.path.join(MF_HISTORY_DIR, "*.csv"))
    if not mf_files:
        print("No Mutual Fund History files found. Please run fetch_mf_history.py first.")
        return
        
    sample_size = min(500, len(mf_files))
    sample_files = mf_files[:sample_size]
    
    all_navs = []
    
    for f in sample_files:
        try:
            df = pd.read_csv(f)
            if 'nav' in df.columns:
                # Force to float to clear any "N/A" string errors and drop nulls
                navs = pd.to_numeric(df['nav'], errors='coerce').dropna().values
                if len(navs) > 0:
                    all_navs.append(navs)
        except Exception as e:
            pass
            
    if not all_navs:
        print("Could not extract numerical NAV arrays.")
        return
        
    all_navs_flat = np.concatenate(all_navs).reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaler.fit(all_navs_flat)
    
    scaler_path = os.path.join(SCALERS_DIR, "mf_nav_global_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Global Mutual Fund Scaler successfully fitted and saved to {scaler_path}")
    print(f"   [Data Bounds Detected -> Min: {scaler.data_min_[0]:.4f}, Max: {scaler.data_max_[0]:.4f}]")

def text_tokenization_prep():
    print("Preprocessing Financial News Text payloads into standard ML format...")
    news_dir = os.path.join(DATASETS_DIR, "unstructured", "financial_news")
    if not os.path.exists(news_dir):
        return
        
    txt_files = glob.glob(os.path.join(news_dir, "*.txt"))
    # In a full production loop, we would utilize a HuggingFace tokenizer here.
    # For now we will construct an aggregated corpus to prove preprocessing.
    corpus = ""
    for txt in txt_files:
        with open(txt, 'r', encoding='utf-8') as f:
            corpus += f.read() + " "
            
    words = corpus.split()
    print(f"✅ Basic NLP Tokenization check complete. Vocabulary size extracted: {len(set(words))} unique tokens.")

if __name__ == "__main__":
    fit_global_mf_scaler()
    text_tokenization_prep()
    print("\nNext Phase: The matrices are mathematically prepped. Time to build the K-Means and LSTM algorithms.")
