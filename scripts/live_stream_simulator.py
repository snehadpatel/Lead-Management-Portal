import os
import time
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project paths
ROOT = Path(__file__).resolve().parent.parent
VELOCITY_DIR = ROOT / "datasets" / "velocity"
INBOX_DIR = ROOT / "streaming" / "inbox"
MARKET_INBOX = INBOX_DIR / "market"
NEWS_INBOX = INBOX_DIR / "news"

def simulate_streams():
    print("="*60)
    print(f"📡 LUME AI LIVE STREAM SIMULATOR STARTING — {datetime.now()}")
    print("="*60)
    
    # Ensure inbox exists
    MARKET_INBOX.mkdir(parents=True, exist_ok=True)
    NEWS_INBOX.mkdir(parents=True, exist_ok=True)
    
    # Load source data
    market_df = pd.read_csv(VELOCITY_DIR / "nse_market_pulse.csv")
    news_df = pd.read_csv(VELOCITY_DIR / "live_news_stream.csv")
    
    print(f"Loaded {len(market_df)} market rows and {len(news_df)} news rows.")
    print("Trickling data into 'streaming/inbox' every 5 seconds...")

    idx = 0
    try:
        while idx < max(len(market_df), len(news_df)):
            # 1. Simulate Market Pulse (1 row at a time)
            if idx < len(market_df):
                chunk = market_df.iloc[[idx]]
                out_path = MARKET_INBOX / f"mkt_{int(time.time() * 1000)}.csv"
                chunk.to_csv(out_path, index=False)
                # print(f" [MKT] Injected row {idx}")
            
            # 2. Simulate News Stream (1 row at a time)
            if idx < len(news_df):
                chunk = news_df.iloc[[idx]]
                out_path = NEWS_INBOX / f"news_{int(time.time() * 1000)}.csv"
                chunk.to_csv(out_path, index=False)
                # print(f" [NEWS] Injected row {idx}")
            
            idx += 1
            if idx % 5 == 0:
                print(f"📡 Stream Simulator Progress: {idx} rows processed...")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopping Simulator...")

if __name__ == "__main__":
    simulate_streams()
