"""Utilities for resolving real mutual fund catalog and NAV history data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from mintleads.config import DATASETS_BASE

REPO_ROOT = Path(__file__).resolve().parents[1]
RICH_CATALOG_PATH = REPO_ROOT / "data/Mutual_Fund_Data-main/mutual_fund_data.csv"
FALLBACK_CATALOG_PATH = DATASETS_BASE / "structured/mutual_funds/amfi_scheme_list.csv"
NAV_HISTORY_DIR = DATASETS_BASE / "structured/mutual_funds/nav_history"

CLUSTER_KEYWORDS: dict[int, tuple[str, ...]] = {
    0: ("equity", "growth", "large cap", "mid cap", "flexi cap", "multicap", "small cap"),
    1: ("debt", "liquid", "overnight", "money market", "ultra short", "short duration", "credit risk"),
    2: ("hybrid", "asset allocation", "balanced advantage", "multi asset", "arbitrage", "dynamic"),
    3: ("index", "etf", "passive", "nifty", "sensex", "fof", "smart beta"),
}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _catalog_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["_aum"] = pd.to_numeric(ranked.get("Average_AUM_Cr", 0), errors="coerce").fillna(0)
    ranked["_nav"] = pd.to_numeric(ranked.get("NAV", 0), errors="coerce").fillna(0)
    if "Scheme_Category" in ranked.columns:
        ranked["_category"] = ranked["Scheme_Category"].fillna("").astype(str)
    else:
        ranked["_category"] = pd.Series([""] * len(ranked), index=ranked.index)
    if "Scheme_Name" in ranked.columns:
        ranked["_scheme_name"] = ranked["Scheme_Name"].fillna("").astype(str)
    else:
        ranked["_scheme_name"] = pd.Series([""] * len(ranked), index=ranked.index)
    return ranked


@lru_cache(maxsize=1)
def load_fund_catalog() -> pd.DataFrame:
    """Load the richest available scheme catalog."""
    for path in (RICH_CATALOG_PATH, FALLBACK_CATALOG_PATH):
        if path.is_file():
            df = pd.read_csv(path, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            if "Scheme_Code" in df.columns:
                df["Scheme_Code"] = df["Scheme_Code"].astype(str)
            return df
    return pd.DataFrame()


def pick_scheme_for_cluster(cluster_id: int) -> dict[str, Any]:
    """Pick a real scheme row for an investor cluster."""
    catalog = load_fund_catalog()
    if catalog.empty:
        return {}

    ranked = _catalog_sort_key(catalog)
    keywords = CLUSTER_KEYWORDS.get(int(cluster_id), ())
    haystack = ranked["_category"].str.lower().fillna("") + " " + ranked["_scheme_name"].str.lower().fillna("")
    score = pd.Series(0, index=ranked.index, dtype="int64")
    for keyword in keywords:
        score = score + haystack.str.contains(keyword, na=False).astype(int)
    ranked["_score"] = score

    if ranked["_score"].max() > 0:
        ranked = ranked[ranked["_score"] > 0]

    ranked = ranked.sort_values(by=["_score", "_aum", "_nav"], ascending=[False, False, False])
    if "Scheme_Code" in ranked.columns:
        ranked_with_history = ranked[ranked["Scheme_Code"].astype(str).map(_nav_history_exists)]
        if not ranked_with_history.empty:
            ranked = ranked_with_history
    row = ranked.iloc[0].to_dict()
    row["Scheme_Code"] = _clean_text(row.get("Scheme_Code"))
    row["Scheme_Name"] = _clean_text(row.get("Scheme_Name"))
    row["Scheme_Category"] = _clean_text(row.get("Scheme_Category"))
    row["AMC"] = _clean_text(row.get("AMC"))
    row["Scheme_Type"] = _clean_text(row.get("Scheme_Type"))
    row["Scheme_Min_Amt"] = _clean_text(row.get("Scheme_Min_Amt"))
    row["NAV"] = pd.to_numeric(pd.Series([row.get("NAV")]), errors="coerce").iloc[0]
    row["Average_AUM_Cr"] = pd.to_numeric(pd.Series([row.get("Average_AUM_Cr")]), errors="coerce").iloc[0]
    row["Latest_NAV_Date"] = _clean_text(row.get("Latest_NAV_Date"))
    row["Launch_Date"] = _clean_text(row.get("Launch_Date"))
    return row


def load_nav_history(scheme_code: str) -> pd.DataFrame:
    """Load historical NAV values for a scheme code."""
    history_path = NAV_HISTORY_DIR / f"{str(scheme_code)}_history.csv"
    if not history_path.is_file():
        return pd.DataFrame(columns=["date", "nav"])

    history = pd.read_csv(history_path, low_memory=False)
    if history.empty:
        return history

    history.columns = [str(c).strip().lower() for c in history.columns]
    if "date" not in history.columns or "nav" not in history.columns:
        return pd.DataFrame(columns=["date", "nav"])

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["nav"] = pd.to_numeric(history["nav"], errors="coerce")
    history = history.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return history


def _nav_history_exists(scheme_code: str) -> bool:
    return (NAV_HISTORY_DIR / f"{str(scheme_code)}_history.csv").is_file()


def _scheme_category_reason(category: str) -> str:
    category_lower = category.lower()
    if any(token in category_lower for token in ("equity", "growth", "large cap", "mid cap", "small cap")):
        return "equity funds move with stock-market cycles, so the NAV can rise and fall more sharply"
    if any(token in category_lower for token in ("debt", "liquid", "duration", "money market", "credit")):
        return "debt funds are shaped by interest-rate changes and credit conditions, so returns are steadier but still move"
    if any(token in category_lower for token in ("hybrid", "asset allocation", "balanced", "arbitrage", "multi asset")):
        return "hybrid funds blend equity and debt exposure, so the return reflects both growth and stability drivers"
    if any(token in category_lower for token in ("index", "etf", "passive", "smart beta")):
        return "passive funds mostly track an index, so the return follows the benchmark path with lower tracking error"
    return "the NAV changes because the fund’s underlying holdings are repriced every market day"


def summarize_fund_return(scheme_code: str, investment_date: str | None = None, invested_amount: float = 10000.0) -> dict[str, Any]:
    """Summarize actual return from NAV history for a scheme."""
    catalog = load_fund_catalog()
    if catalog.empty:
        return {"error": "Fund catalog not available"}

    row = catalog[catalog["Scheme_Code"].astype(str) == str(scheme_code)]
    if row.empty:
        return {"error": f"Scheme code {scheme_code} not found"}

    scheme = row.iloc[0].to_dict()
    history = load_nav_history(str(scheme_code))
    if history.empty:
        return {"error": f"NAV history not available for {scheme_code}"}

    latest_row = history.iloc[-1]
    latest_date = pd.Timestamp(latest_row["date"])
    latest_nav = float(latest_row["nav"])

    if investment_date:
        entry_target = pd.to_datetime(investment_date, errors="coerce")
    else:
        entry_target = pd.to_datetime(scheme.get("Launch_Date"), errors="coerce")

    if pd.isna(entry_target):
        entry_target = history.iloc[0]["date"]

    entry_slice = history[history["date"] <= entry_target]
    if entry_slice.empty:
        entry_row = history.iloc[0]
    else:
        entry_row = entry_slice.iloc[-1]

    entry_date = pd.Timestamp(entry_row["date"])
    entry_nav = float(entry_row["nav"])
    units = float(invested_amount) / entry_nav if entry_nav else 0.0
    current_value = units * latest_nav
    gain = current_value - float(invested_amount)
    return_pct = (gain / float(invested_amount) * 100.0) if invested_amount else 0.0
    holding_days = max(0, int((latest_date - entry_date).days))
    annualized_return = None
    if holding_days > 0 and invested_amount > 0:
        annualized_return = ((current_value / float(invested_amount)) ** (365.0 / holding_days) - 1.0) * 100.0

    category = _clean_text(scheme.get("Scheme_Category") or scheme.get("Category"))
    return {
        "scheme_code": _clean_text(scheme.get("Scheme_Code")),
        "scheme_name": _clean_text(scheme.get("Scheme_Name")),
        "scheme_category": category,
        "amc": _clean_text(scheme.get("AMC")),
        "scheme_type": _clean_text(scheme.get("Scheme_Type")),
        "latest_nav": latest_nav,
        "latest_nav_date": latest_date.strftime("%Y-%m-%d"),
        "entry_nav": entry_nav,
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "invested_amount": float(invested_amount),
        "units": units,
        "current_value": current_value,
        "gain": gain,
        "return_pct": return_pct,
        "annualized_return_pct": annualized_return,
        "holding_days": holding_days,
        "return_reason": _scheme_category_reason(category),
        "launch_date": _clean_text(scheme.get("Launch_Date")),
        "scheme_min_amt": _clean_text(scheme.get("Scheme_Min_Amt")),
        "average_aum_cr": pd.to_numeric(pd.Series([scheme.get("Average_AUM_Cr")]), errors="coerce").iloc[0],
    }


def build_cluster_fund_profile(cluster_id: int, investment_date: str | None = None, invested_amount: float = 10000.0) -> dict[str, Any]:
    """Combine the recommended scheme and its actual return summary."""
    scheme = pick_scheme_for_cluster(cluster_id)
    if not scheme:
        return {"error": "Fund catalog not available"}

    summary = summarize_fund_return(scheme["Scheme_Code"], investment_date=investment_date, invested_amount=invested_amount)
    if "error" in summary:
        return {"error": summary["error"]}

    summary.update(
        {
            "cluster_id": int(cluster_id),
            "recommended_fund_type": scheme.get("Scheme_Category") or scheme.get("Scheme_Name"),
            "recommended_scheme": {
                "scheme_code": scheme.get("Scheme_Code"),
                "scheme_name": scheme.get("Scheme_Name"),
                "scheme_category": scheme.get("Scheme_Category"),
                "amc": scheme.get("AMC"),
                "scheme_type": scheme.get("Scheme_Type"),
                "nav": scheme.get("NAV"),
                "latest_nav_date": scheme.get("Latest_NAV_Date"),
                "min_investment": scheme.get("Scheme_Min_Amt"),
                "average_aum_cr": scheme.get("Average_AUM_Cr"),
                "launch_date": scheme.get("Launch_Date"),
            },
        }
    )
    return summary
