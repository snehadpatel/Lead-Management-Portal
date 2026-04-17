import os
from datetime import date
from jugaad_data.nse import index_df
import pandas as pd

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def fetch_historical_reports():
    print("Fetching NSE Historical Reports (NIFTY 50)...")
    try:
        # Fetching Nifty 50 historical data for the last 6 months 
        df = index_df(symbol="NIFTY 50", from_date=date(2023, 1, 1), to_date=date(2023, 6, 30))
        
        # Save historical NSE data
        out_file = os.path.join(DATASETS_DIR, "nse_historical_nifty50_2023.csv")
        df.to_csv(out_file, index=False)
        print(f"Successfully saved {len(df)} historical NSE records to {out_file}")
    except Exception as e:
        print(f"Failed to fetch NSE historical reports: {e}")

if __name__ == "__main__":
    fetch_historical_reports()
