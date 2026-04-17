import requests
import pandas as pd
from io import StringIO
import os

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

def fetch_latest_amfi_nav():
    print("Fetching latest NAV data from AMFI...")
    response = requests.get(AMFI_URL)
    
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return
    
    lines = response.text.split('\n')
    
    # AMFI data is semicolon separated, but has headers and blank lines intermixed
    parsed_data = []
    current_category = "Unknown"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # If line doesn't have semicolons, it's likely a category header
        if ';' not in line:
            current_category = line
            continue
            
        parts = line.split(';')
        
        # Skip the header row itself
        if parts[0] == 'Scheme Code':
            continue
            
        if len(parts) >= 6:
            parsed_data.append({
                'Category': current_category,
                'Scheme_Code': parts[0],
                'ISIN_Div_Payout': parts[1],
                'ISIN_Div_Reinvestment': parts[2],
                'Scheme_Name': parts[3],
                'Net_Asset_Value': parts[4],
                'Date': parts[5]
            })

    if not parsed_data:
        print("No valid data parsed.")
        return

    df = pd.DataFrame(parsed_data)
    
    # Clean numerical strings to floats where appropriate
    df['Net_Asset_Value'] = pd.to_numeric(df['Net_Asset_Value'], errors='coerce')
    
    out_file = os.path.join(DATASETS_DIR, "amfi_mutual_funds_latest.csv")
    df.to_csv(out_file, index=False)
    print(f"Successfully saved {len(df)} mutual fund entries to {out_file}")

def fetch_nse_market_status():
    print("Fetching live market status from NSE...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # NSE requires visiting the homepage first to set cookies
        session.get("https://www.nseindia.com", timeout=10)
        
        # Now fetch the market status API
        api_url = "https://www.nseindia.com/api/marketStatus"
        res = session.get(api_url, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            market_state = data.get('marketState', [])
            
            if market_state:
                df = pd.DataFrame(market_state)
                out_file = os.path.join(DATASETS_DIR, "nse_market_status.csv")
                df.to_csv(out_file, index=False)
                print(f"Successfully saved NSE Market Status to {out_file}")
            else:
                print("Failed to parse market state from NSE JSON.")
        else:
            print(f"Failed to fetch NSE market status. Status code: {res.status_code}")
    except Exception as e:
        print(f"Error fetching NSE data: {e}")

if __name__ == "__main__":
    fetch_latest_amfi_nav()
    fetch_nse_market_status()
