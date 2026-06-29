"""Create SBERT fund embeddings cache pickle for MVP demo.
Uses a subset of 500 funds across unique categories to keep it lightweight.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "Mutual_Fund_Data-main" / "mutual_fund_data.csv"
OUT_PATH = ROOT / "lume-ai-product" / "ml-artifacts" / "models" / "fund_embeddings.pkl"

def main():
    if not CSV_PATH.is_file():
        print(f"CSV not found at {CSV_PATH}")
        return

    print("Reading fund catalog...")
    df = pd.read_csv(CSV_PATH)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Expected columns: scheme_code, scheme_name, scheme_category (as category)
    # Let's map them
    funds_df = pd.DataFrame()
    funds_df['scheme_code'] = df['scheme_code'].astype(str)
    funds_df['scheme_name'] = df['scheme_name'].fillna('Unknown Fund')
    funds_df['category'] = df['scheme_category'].fillna('Mutual Fund')
    
    # Dedup by scheme_name to keep it clean
    funds_df = funds_df.drop_duplicates(subset=['scheme_name'])
    
    # Grab a sample of 500 funds to keep the file size low
    # Let's group by category to make sure we get a representative sample of all categories
    categories = funds_df['category'].unique()
    sample_dfs = []
    per_cat = max(5, int(500 / len(categories)))
    for cat in categories:
        cat_df = funds_df[funds_df['category'] == cat]
        sample_dfs.append(cat_df.head(per_cat))
        
    sampled_funds_df = pd.concat(sample_dfs).head(500)
    funds = sampled_funds_df.to_dict(orient='records')
    
    print(f"Sourced {len(funds)} unique mutual funds for demo index.")
    
    print("Encoding with SentenceTransformer...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Smaller, faster model (384 dims, ~90MB)
    
    texts = [f"{f['scheme_name']} - Category: {f['category']}" for f in funds]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Save cache
    data = {
        'funds': funds,
        'embeddings': np.asarray(embeddings)
    }
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"✅ Generated embeddings and saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
