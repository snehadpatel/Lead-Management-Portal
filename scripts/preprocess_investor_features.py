import pandas as pd
import numpy as np
import os

BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
ML_FEATURES_DIR = os.path.join(DATASETS_DIR, "ml_features")
os.makedirs(ML_FEATURES_DIR, exist_ok=True)

USER_DATA_PATH = os.path.join(DATASETS_DIR, "kaggle/financial_user_behaviors/users_data.csv")
TXN_DATA_PATH = os.path.join(DATASETS_DIR, "kaggle/financial_user_behaviors/transactions_data.csv")

def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', ''))
    return x

def preprocess_investors():
    print("Loading user profile data...")
    users = pd.read_csv(USER_DATA_PATH)
    
    # Clean currency columns to standard floats
    print("Formatting currency string arrays to numerics...")
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        users[col] = users[col].apply(clean_currency)
        
    # Calculate Debt-to-Income Ratio (Core Risk Proxy)
    users['debt_to_income_ratio'] = users['total_debt'] / users['yearly_income'].replace(0, 1)
    
    print("Users data cleaned. Processing 1.25GB transaction log in chunks to avoid memory overload...")
    
    # Aggregate transaction data in chunks
    chunk_size = 5000000
    client_stats = []
    
    try:
        for chunk in pd.read_csv(TXN_DATA_PATH, chunksize=chunk_size):
            # Clean currency in amount column
            chunk['amount'] = chunk['amount'].apply(clean_currency)
            # Group by client_id to summarize their behavior footprint
            stats = chunk.groupby('client_id').agg({
                'amount': ['sum', 'count']
            })
            stats.columns = ['behavior_spend_sum', 'behavior_spend_count']
            client_stats.append(stats)
    except Exception as e:
        print(f"Error during chunking: {e}")
        
    print("Combining behavioral interaction footprints...")
    if client_stats:
        combined_stats = pd.concat(client_stats)
        final_stats = combined_stats.groupby('client_id').sum()
    else:
        final_stats = pd.DataFrame(columns=['behavior_spend_sum', 'behavior_spend_count'])
    
    print("Merging user profiles with transaction behavior...")
    # Join the behavioral data cleanly to the main Profile dataset
    users_merged = pd.merge(users, final_stats, left_on='id', right_on='client_id', how='left')
    
    # Fill NaN for users with zero transactions
    users_merged['behavior_spend_sum'] = users_merged['behavior_spend_sum'].fillna(0)
    users_merged['behavior_spend_count'] = users_merged['behavior_spend_count'].fillna(0)
    
    # Establish proxy mapping for Mutual Fund Category matching based on financial health
    def map_risk_appetite(row):
        score = row['credit_score']
        dti = row['debt_to_income_ratio']
        
        # High credit, low debt -> Can absorb higher risk -> Equity
        if score > 720 and dti < 1.5:
            return 'Aggressive (Equity)'
        # Medium credit, medium debt -> Balanced -> Hybrid
        elif score > 650 and dti < 2.5:
            return 'Moderate (Hybrid)'
        # Low credit, high debt -> Must preserve capital -> Debt
        else:
            return 'Conservative (Debt)'
            
    users_merged['inferred_risk_profile'] = users_merged.apply(map_risk_appetite, axis=1)
    
    # Select our engineered features out of the raw noise
    ml_features = [
        'id', 'current_age', 'yearly_income', 'total_debt', 'debt_to_income_ratio', 
        'credit_score', 'behavior_spend_sum', 'behavior_spend_count', 'inferred_risk_profile'
    ]
    
    final_df = users_merged[ml_features]
    
    out_path = os.path.join(ML_FEATURES_DIR, "investor_profiles_scaled.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"Successfully engineered and linked {len(final_df)} investor profiles.")
    print(f"Dataset securely staged at: {out_path}")
    print("Pre-processing step complete. Ready for K-Means.")

if __name__ == "__main__":
    preprocess_investors()
