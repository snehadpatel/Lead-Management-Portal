"""
stream_velocity_data.py
-----------------------
Demonstrates the VELOCITY dimension of Big Data by streaming real-time
financial data from 3 official Indian sources simultaneously.

Sources:
  1. AMFI Live NAV    — Full dump of 14K+ mutual fund NAVs (daily snapshot)
  2. NSE Market Pulse — NIFTY 50 index polled every 5 seconds
  3. Financial News   — RSS headlines polled every 30 seconds

All outputs are timestamped to prove high-frequency ingestion.
"""

import os
import csv
import json
import time
import threading
from datetime import datetime

import requests
import feedparser

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = "/Users/snehapatel/BigData"
VELOCITY_DIR = os.path.join(BASE_DIR, "datasets", "velocity")
NAV_DIR = os.path.join(VELOCITY_DIR, "amfi_nav_snapshots")
os.makedirs(NAV_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

# ── 1. AMFI Live NAV Snapshot ────────────────────────────────────────────────

def fetch_amfi_nav_snapshot():
    """Pull the complete live NAV file from AMFI and save a timestamped copy."""
    print("\n[AMFI NAV] Fetching live NAV data for 14,000+ mutual funds...")
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nav_snapshot_{ts}.txt"
            filepath = os.path.join(NAV_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(r.text)
            lines = r.text.strip().split("\n")
            print(f"[AMFI NAV] ✅ Saved {len(lines)} lines → {filename}")
        else:
            print(f"[AMFI NAV] ⚠️  HTTP {r.status_code}")
    except Exception as e:
        print(f"[AMFI NAV] ❌ Error: {e}")


# ── 2. NSE Market Pulse ──────────────────────────────────────────────────────

def stream_nse_market_pulse(duration_seconds=120, interval=5):
    """
    Poll the NSE Market Status API every `interval` seconds for `duration_seconds`.
    Logs NIFTY 50 price, variation, and market state into a timestamped CSV.
    """
    csv_path = os.path.join(VELOCITY_DIR, "nse_market_pulse.csv")
    file_exists = os.path.exists(csv_path)
    
    print(f"\n[NSE PULSE] Starting market pulse (every {interval}s for {duration_seconds}s)...")
    url = "https://www.nseindia.com/api/marketStatus"
    
    # NSE requires a session with cookies
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Warm up session (NSE needs cookies from homepage first)
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    
    end_time = time.time() + duration_seconds
    tick_count = 0
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Market", "Status", "Index",
                "Last_Price", "Variation", "Pct_Change", "Trade_Date"
            ])
        
        while time.time() < end_time:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            try:
                r = session.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    for market in data.get("marketState", []):
                        writer.writerow([
                            ts,
                            market.get("market", ""),
                            market.get("marketStatus", ""),
                            market.get("index", ""),
                            market.get("last", ""),
                            market.get("variation", ""),
                            market.get("percentChange", ""),
                            market.get("tradeDate", ""),
                        ])
                    tick_count += 1
                    f.flush()
                    if tick_count % 5 == 0:
                        nifty = data["marketState"][0]
                        print(f"[NSE PULSE] Tick #{tick_count}: NIFTY 50 = ₹{nifty.get('last', 'N/A')} "
                              f"({nifty.get('percentChange', '')}%)")
                else:
                    print(f"[NSE PULSE] ⚠️  HTTP {r.status_code}")
            except Exception as e:
                print(f"[NSE PULSE] ⚠️  {e}")
            
            time.sleep(interval)
    
    print(f"[NSE PULSE] ✅ Logged {tick_count} ticks → nse_market_pulse.csv")


# ── 3. Live News Stream ──────────────────────────────────────────────────────

RSS_FEEDS = [
    ("Moneycontrol_Markets", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("Moneycontrol_Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("ET_Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("ET_MutualFunds", "https://economictimes.indiatimes.com/markets/rssfeeds/17999563.cms"),
    ("Livemint_Market", "https://www.livemint.com/rss/markets"),
]


def stream_news_feed(duration_seconds=120, interval=30):
    """
    Poll multiple financial RSS feeds every `interval` seconds for `duration_seconds`.
    Appends new headlines with exact ingestion timestamps.
    """
    csv_path = os.path.join(VELOCITY_DIR, "live_news_stream.csv")
    file_exists = os.path.exists(csv_path)
    seen_titles = set()
    
    # Load already-seen titles
    if file_exists:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) > 2:
                        seen_titles.add(row[2])
        except Exception:
            pass
    
    print(f"\n[NEWS STREAM] Starting live news stream (every {interval}s for {duration_seconds}s)...")
    
    end_time = time.time() + duration_seconds
    total_new = 0
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Ingestion_Timestamp", "Source", "Title",
                "Link", "Published_Date", "Summary"
            ])
        
        poll_count = 0
        while time.time() < end_time:
            poll_count += 1
            batch_new = 0
            
            for feed_name, feed_url in RSS_FEEDS:
                try:
                    parsed = feedparser.parse(feed_url)
                    for entry in parsed.entries:
                        title = entry.get("title", "").strip()
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            writer.writerow([
                                ts,
                                feed_name,
                                title,
                                entry.get("link", ""),
                                entry.get("published", ""),
                                entry.get("summary", "")[:200],
                            ])
                            batch_new += 1
                            total_new += 1
                except Exception:
                    pass
            
            f.flush()
            print(f"[NEWS STREAM] Poll #{poll_count}: {batch_new} new headlines ingested")
            time.sleep(interval)
    
    print(f"[NEWS STREAM] ✅ Total new headlines: {total_new} → live_news_stream.csv")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  VELOCITY DATA COLLECTOR — Real-Time Financial Streams")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # 1. Quick AMFI snapshot (one-shot, takes ~2s)
    fetch_amfi_nav_snapshot()
    
    # 2. Run NSE pulse and News stream in parallel threads
    nse_thread = threading.Thread(
        target=stream_nse_market_pulse,
        kwargs={"duration_seconds": 120, "interval": 5}
    )
    news_thread = threading.Thread(
        target=stream_news_feed,
        kwargs={"duration_seconds": 120, "interval": 30}
    )
    
    nse_thread.start()
    news_thread.start()
    
    nse_thread.join()
    news_thread.join()
    
    print(f"\n{'=' * 65}")
    print(f"  VELOCITY COLLECTION COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output directory: {VELOCITY_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
