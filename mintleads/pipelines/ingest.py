"""Data ingestion pipeline for MintLeads.

This module provides data ingestion capabilities from multiple sources:
- AMFI NAV feed (daily)
- NSE Market Status API (during trading hours)
- RSS feeds from Moneycontrol and Economic Times

All data is stored in the velocity directory for live streaming analysis.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import feedparser
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    AMFI_NAV_URL,
    ECONOMIC_TIMES_RSS,
    MONEYCONTROL_RSS,
    NSE_MARKET_STATUS_URL,
    VELOCITY_DATA_DIR,
    setup_logging,
)

logger = setup_logging(__name__)


def create_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504),
) -> requests.Session:
    """Create a requests session with retry logic.
    
    Args:
        retries: Number of retries.
        backoff_factor: Backoff factor for retries.
        status_forcelist: HTTP status codes to retry on.
        
    Returns:
        Configured requests Session.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_amfi_nav() -> Optional[pd.DataFrame]:
    """Fetch latest NAV data from AMFI India.
    
    Parses the pipe-delimited format from AMFI and converts to DataFrame.
    
    Returns:
        DataFrame with NAV data or None if fetch failed.
    """
    logger.info("Fetching latest NAV data from AMFI...")
    
    try:
        session = create_retry_session()
        response = session.get(AMFI_NAV_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch AMFI data: {e}")
        return None
    
    lines = response.text.split("\n")
    parsed_data: List[Dict[str, Any]] = []
    current_category = "Unknown"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Lines without semicolons are category headers
        if ";" not in line:
            current_category = line
            continue
        
        parts = line.split(";")
        
        # Skip header row
        if parts[0] == "Scheme Code":
            continue
        
        if len(parts) >= 6:
            parsed_data.append({
                "Category": current_category,
                "Scheme_Code": parts[0].strip(),
                "ISIN_Div_Payout": parts[1].strip() if len(parts) > 1 else "",
                "ISIN_Div_Reinvestment": parts[2].strip() if len(parts) > 2 else "",
                "Scheme_Name": parts[3].strip() if len(parts) > 3 else "",
                "Net_Asset_Value": parts[4].strip() if len(parts) > 4 else None,
                "Date": parts[5].strip() if len(parts) > 5 else "",
            })
    
    if not parsed_data:
        logger.warning("No valid data parsed from AMFI feed")
        return None
    
    df = pd.DataFrame(parsed_data)
    
    # Clean numerical data
    df["Net_Asset_Value"] = pd.to_numeric(df["Net_Asset_Value"], errors="coerce")
    df["Ingestion_Timestamp"] = datetime.now().isoformat()
    
    # Compute MD5 hash for deduplication
    df["md5_hash"] = df.apply(
        lambda row: hashlib.md5(
            f"{row['Scheme_Code']}_{row['Net_Asset_Value']}_{row['Date']}".encode()
        ).hexdigest(),
        axis=1,
    )
    
    logger.info(f"Successfully parsed {len(df)} mutual fund entries from AMFI")
    return df


def save_amfi_nav(df: pd.DataFrame) -> Path:
    """Save AMFI NAV data to velocity directory.
    
    Args:
        df: DataFrame with NAV data.
        
    Returns:
        Path to saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = VELOCITY_DATA_DIR / f"amfi_nav_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved AMFI NAV data to {output_file}")
    return output_file


def fetch_nse_market_status() -> Optional[Dict[str, Any]]:
    """Fetch live market status from NSE India.
    
    Returns:
        Dictionary with market status data or None if fetch failed.
    """
    logger.info("Fetching live market status from NSE...")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        session = create_retry_session()
        session.headers.update(headers)
        
        # NSE requires visiting homepage first for cookies
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.5)  # Brief pause to mimic browser behavior
        
        response = session.get(NSE_MARKET_STATUS_URL, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        market_state = data.get("marketState", [])
        
        if not market_state:
            logger.warning("No market state data in NSE response")
            return None
        
        # Add timestamp
        for item in market_state:
            item["Timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Successfully fetched NSE market status for {len(market_state)} markets")
        return {"marketState": market_state, "ingestion_time": datetime.now().isoformat()}
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NSE market status: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse NSE response: {e}")
        return None


def append_nse_pulse(data: Dict[str, Any]) -> Path:
    """Append NSE market data to pulse file.
    
    Args:
        data: Dictionary with market status data.
        
    Returns:
        Path to pulse file.
    """
    pulse_file = VELOCITY_DATA_DIR / "nse_pulse.jsonl"
    
    with open(pulse_file, "a") as f:
        f.write(json.dumps(data) + "\n")
    
    logger.debug(f"Appended NSE pulse data to {pulse_file}")
    return pulse_file


def fetch_rss_feed(url: str, source_name: str) -> List[Dict[str, Any]]:
    """Fetch and parse an RSS feed.
    
    Args:
        url: RSS feed URL.
        source_name: Name of the source for logging.
        
    Returns:
        List of parsed feed entries.
    """
    logger.info(f"Fetching RSS feed from {source_name}...")
    
    try:
        feed = feedparser.parse(url)
        
        if feed.bozo and hasattr(feed, "bozo_exception"):
            logger.warning(f"RSS parse warning for {source_name}: {feed.bozo_exception}")
        
        entries = []
        for entry in feed.entries:
            entries.append({
                "Ingestion_Timestamp": datetime.now().isoformat(),
                "Source": source_name,
                "Title": entry.get("title", ""),
                "Link": entry.get("link", ""),
                "Published_Date": entry.get("published", ""),
                "Summary": entry.get("summary", ""),
                "md5_hash": hashlib.md5(
                    entry.get("link", "").encode()
                ).hexdigest(),
            })
        
        logger.info(f"Parsed {len(entries)} entries from {source_name}")
        return entries
        
    except Exception as e:
        logger.error(f"Failed to fetch RSS from {source_name}: {e}")
        return []


def append_news_stream(entries: List[Dict[str, Any]]) -> Path:
    """Append news entries to stream file.
    
    Args:
        entries: List of news entries.
        
    Returns:
        Path to stream file.
    """
    stream_file = VELOCITY_DATA_DIR / "news_stream.jsonl"
    
    with open(stream_file, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    logger.debug(f"Appended {len(entries)} news entries to {stream_file}")
    return stream_file


def is_market_hours() -> bool:
    """Check if current time is within Indian market hours (09:15-15:30 IST).
    
    Returns:
        True if market is open, False otherwise.
    """
    # Note: This is a simplified check. In production, consider timezone handling
    now = datetime.now()
    
    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() > 4:
        return False
    
    # Check market hours (09:15 to 15:30)
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end


def run_ingestion_cycle() -> Dict[str, Any]:
    """Run a complete ingestion cycle.
    
    Fetches data from all configured sources.
    
    Returns:
        Dictionary with ingestion results.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "amfi_nav": None,
        "nse_pulse": None,
        "news_stream": None,
    }
    
    # Fetch AMFI NAV (daily)
    nav_df = fetch_amfi_nav()
    if nav_df is not None:
        nav_file = save_amfi_nav(nav_df)
        results["amfi_nav"] = {"records": len(nav_df), "file": str(nav_file)}
    
    # Fetch NSE market status (only during market hours)
    if is_market_hours():
        nse_data = fetch_nse_market_status()
        if nse_data is not None:
            pulse_file = append_nse_pulse(nse_data)
            results["nse_pulse"] = {"markets": len(nse_data.get("marketState", [])), "file": str(pulse_file)}
    else:
        logger.info("Outside market hours, skipping NSE market status fetch")
        results["nse_pulse"] = {"skipped": True, "reason": "Outside market hours"}
    
    # Fetch RSS feeds
    all_news = []
    
    mc_entries = fetch_rss_feed(MONEYCONTROL_RSS, "Moneycontrol")
    all_news.extend(mc_entries)
    
    et_entries = fetch_rss_feed(ECONOMIC_TIMES_RSS, "Economic_Times")
    all_news.extend(et_entries)
    
    if all_news:
        stream_file = append_news_stream(all_news)
        results["news_stream"] = {"records": len(all_news), "file": str(stream_file)}
    
    return results


def continuous_ingestion(
    nse_poll_interval: int = 5,
    rss_poll_interval: int = 30,
    amfi_poll_interval: int = 86400,
) -> None:
    """Run continuous data ingestion with configurable polling intervals.
    
    Args:
        nse_poll_interval: Seconds between NSE market status polls.
        rss_poll_interval: Seconds between RSS feed polls.
        amfi_poll_interval: Seconds between AMFI NAV polls.
    """
    logger.info(
        f"Starting continuous ingestion: "
        f"NSE={nse_poll_interval}s, RSS={rss_poll_interval}s, AMFI={amfi_poll_interval}s"
    )
    
    last_nse_poll = 0
    last_rss_poll = 0
    last_amfi_poll = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Poll AMFI NAV (once per day)
            if current_time - last_amfi_poll >= amfi_poll_interval:
                nav_df = fetch_amfi_nav()
                if nav_df is not None:
                    save_amfi_nav(nav_df)
                last_amfi_poll = current_time
            
            # Poll NSE (during market hours only)
            if current_time - last_nse_poll >= nse_poll_interval:
                if is_market_hours():
                    nse_data = fetch_nse_market_status()
                    if nse_data is not None:
                        append_nse_pulse(nse_data)
                last_nse_poll = current_time
            
            # Poll RSS feeds
            if current_time - last_rss_poll >= rss_poll_interval:
                mc_entries = fetch_rss_feed(MONEYCONTROL_RSS, "Moneycontrol")
                et_entries = fetch_rss_feed(ECONOMIC_TIMES_RSS, "Economic_Times")
                all_news = mc_entries + et_entries
                if all_news:
                    append_news_stream(all_news)
                last_rss_poll = current_time
            
            # Sleep to prevent CPU spinning
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user")
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise


if __name__ == "__main__":
    # Run a single ingestion cycle
    results = run_ingestion_cycle()
    print(json.dumps(results, indent=2))
