import yfinance as yf
import pandas as pd
import os
import requests

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
EOD_DIR = os.path.join(DATASETS_DIR, "EOD")
os.makedirs(EOD_DIR, exist_ok=True)

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

def download_equity_list():
    file_path = os.path.join(BASE_DIR, "EQUITY_L.csv")
    
    # Only download if we don't have it already
    if not os.path.exists(file_path):
        print("Downloading official NSE Equity list...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        res = requests.get(NSE_EQUITY_LIST_URL, headers=headers)
        if res.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(res.content)
            print("Successfully downloaded EQUITY_L.csv")
        else:
            print(f"Failed to download EQUITY_L.csv: {res.status_code}")
            return None
    return file_path

def fetch_yfinance_eod_data():
    csv_path = download_equity_list()
    
    if not csv_path or not os.path.exists(csv_path):
        print("EQUITY_L.csv not found. Aborting.")
        return
        
    print("Loading NSE equity symbols...")
    try:
        equity_details = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading EQUITY_L.csv: {e}")
        return
        
    if 'SYMBOL' not in equity_details.columns:
        print("SYMBOL column not found in equity list.")
        return
        
    symbols = equity_details['SYMBOL'].tolist()
    
    print(f"Fetching 20 years EOD data for all {len(symbols)} symbols...")
    
    success_count = 0
    
    for name in symbols:
        try:
            print(f"  Fetching {name}...")
            data = yf.download(f"{name}.NS", period="20y", progress=False)
            if not data.empty:
                out_path = os.path.join(EOD_DIR, f"{name}.csv")
                data.to_csv(out_path)
                success_count += 1
            else:
                print(f"    No data found for {name}")
        except Exception as e:
            print(f"    Error fetching {name} ===> {e}")

    print(f"Successfully saved {success_count} EOD data files to {EOD_DIR}")

if __name__ == "__main__":
    fetch_yfinance_eod_data()
