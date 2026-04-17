import os
import pandas as pd
import glob

BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def merge_local_nse_history():
    print("Ingesting local NSE NIFTY 50 Historical data files...")
    
    # Find all NIFTY 50 CSVs in the base directory
    nifty_files = glob.glob(os.path.join(BASE_DIR, "NIFTY *.csv"))
    
    if not nifty_files:
        print("No local NIFTY 50 CSV files found in the root directory.")
        return
        
    all_dfs = []
    
    for file in nifty_files:
        print(f"  Reading {os.path.basename(file)}...")
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean up column names (strip whitespace)
        merged_df.columns = merged_df.columns.str.strip()
        
        # Ensure 'Date' column is converted properly and sort by it
        if 'Date' in merged_df.columns:
            merged_df['Date'] = pd.to_datetime(merged_df['Date'])
            merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)
            
        out_file = os.path.join(DATASETS_DIR, "nse_nifty50_historical_merged.csv")
        merged_df.to_csv(out_file, index=False)
        print(f"Successfully merged {len(merged_df)} historical records to {out_file}")

if __name__ == "__main__":
    merge_local_nse_history()
