from mftool import Mftool
import pandas as pd
import os

BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MF_HISTORY_DIR = os.path.join(DATASETS_DIR, "mutual_fund_history")
os.makedirs(MF_HISTORY_DIR, exist_ok=True)

def fetch_mf_history():
    print("Initializing MFtool to fetch historical Daily NAV data...")
    mf = Mftool()
    
    # We load the scheme codes from the AMFI list we already downloaded
    amfi_csv = os.path.join(DATASETS_DIR, "amfi_mutual_funds_latest.csv")
    if not os.path.exists(amfi_csv):
        print("AMFI base list not found. Cannot proceed without scheme codes.")
        return
        
    df_amfi = pd.read_csv(amfi_csv)
    
    # Let's extract specific valid scheme codes
    scheme_codes = df_amfi['Scheme_Code'].dropna().astype(str).tolist()
    
    print(f"Attempting to fetch 10-year NAV history for all {len(scheme_codes)} funds...")
    
    success_count = 0
    
    for code in scheme_codes:
        try:
            print(f"  Fetching history for Scheme Code: {code}")
            # Get historical NAV data (usually returns as a dict with 'data' key containing a list of {date, nav})
            history = mf.get_scheme_historical_nav(code, as_json=False)
            
            if history and 'data' in history and len(history['data']) > 0:
                history_df = pd.DataFrame(history['data'])
                
                # Format to standardize dates
                if 'date' in history_df.columns:
                    history_df['date'] = pd.to_datetime(history_df['date'], format="%d-%m-%Y").dt.strftime('%Y-%m-%d')
                
                out_path = os.path.join(MF_HISTORY_DIR, f"{code}_history.csv")
                history_df.to_csv(out_path, index=False)
                success_count += 1
            else:
                print(f"    No historical data found for {code}.")
        except Exception as e:
            print(f"    Error fetching {code}: {e}")

    print(f"Successfully saved {success_count} MF Historical data files to {MF_HISTORY_DIR}")

if __name__ == "__main__":
    fetch_mf_history()
