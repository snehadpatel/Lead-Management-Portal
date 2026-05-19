"""
Mock MongoDB Client: A local file-based and in-memory mock database
that loads and stores leads and investor/distributor routing matches,
enabling the platform to function without a live MongoDB instance.
"""

from __future__ import annotations

import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from lume_platform.config import EXPORT_DIR

class MockCollection:
    def __init__(self, name: str, db_client: MockMongoClient):
        self.name = name
        self.db_client = db_client

    def find_one(self, query: dict) -> Optional[dict]:
        """Find a single document matching the query."""
        if self.name == "leads":
            lead_id = query.get("lead_id")
            if lead_id:
                return self.db_client.leads_db.get(str(lead_id))
        return None

    def find(self, query: dict = None) -> List[dict]:
        """Find documents matching the query."""
        # Simple implementation returning all documents
        if self.name == "leads":
            return list(self.db_client.leads_db.values())
        return []

class MockMongoClient:
    def __init__(self):
        self.leads_db: Dict[str, dict] = {}
        self.custom_leads_file = EXPORT_DIR / "db_custom_leads.json"
        self._load_initial_data()

    def _load_initial_data(self) -> None:
        """Seed leads and matches from output_production_final CSV exports."""
        # 1. Load distributor leads master CSV
        leads_csv_path = EXPORT_DIR / "distributor_leads_master.csv"
        if leads_csv_path.is_file():
            try:
                df = pd.read_csv(leads_csv_path)
                for _, row in df.iterrows():
                    lead_id = str(row.get("Lead Number", f"L-{1000 + _}"))
                    # Map CSV columns to lead features dict
                    # Also keep original keys to support aliases
                    self.leads_db[lead_id] = {
                        "lead_id": lead_id,
                        "first_name": str(row.get("First Name", "")),
                        "last_name": str(row.get("Last Name", "")),
                        "name": f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                        "city": str(row.get("City", "Unknown")),
                        "occupation": str(row.get("Occupation", "Unknown")),
                        "What is your current occupation": str(row.get("Occupation", "Unknown")),
                        "conversion_probability": float(row.get("Conversion_Probability", 0.5)),
                        "recommended_pitch_persona": str(row.get("Recommended_Pitch_Persona", "Unknown")),
                        "psychological_profile": str(row.get("Psychological_Profile", "Unknown")),
                        "ai_rationale": str(row.get("AI_Rationale", "Unknown")),
                        "potential_investment": str(row.get("Potential_Investment", "Unknown")),
                        "lead_source": str(row.get("Lead Source", "Unknown")),
                        "industry": str(row.get("Industry", "Unknown")),
                        "converted_prediction": int(row.get("Conversion_Probability", 0.5) > 0.65),
                        "status": "New",
                        "notes": "",
                        "assignee": "",
                        "next_step_at": ""
                    }
                print(f"✅ Loaded {len(self.leads_db)} leads from master CSV.")
            except Exception as e:
                print(f"⚠️ Error loading leads master CSV: {e}")
        else:
            print(f"⚠️ Master leads CSV not found at {leads_csv_path}")

        # 2. Load custom persistent leads if any
        if self.custom_leads_file.is_file():
            try:
                with open(self.custom_leads_file, "r") as f:
                    custom_leads = json.load(f)
                    for lead_id, lead_data in custom_leads.items():
                        # Override/add custom leads
                        self.leads_db[lead_id] = lead_data
                print(f"✅ Loaded persistent custom leads from {self.custom_leads_file}")
            except Exception as e:
                print(f"⚠️ Error loading custom leads JSON: {e}")

    def _save_custom_leads(self) -> None:
        """Write custom/updated leads to JSON file for persistence."""
        # Find leads not from original master CSV (or all of them to be safe)
        try:
            with open(self.custom_leads_file, "w") as f:
                json.dump(self.leads_db, f, indent=4)
        except Exception as e:
            print(f"⚠️ Error writing custom leads JSON: {e}")

    def upsert_lead(self, lead_id: str, data: dict) -> None:
        """Upsert lead details."""
        lead_id = str(lead_id)
        if lead_id in self.leads_db:
            self.leads_db[lead_id].update(data)
        else:
            # Create new lead
            self.leads_db[lead_id] = {
                "lead_id": lead_id,
                "status": "New",
                "notes": "",
                "assignee": "",
                "next_step_at": "",
                **data
            }
            # Make sure 'name' is formatted if first/last name exists
            if "first_name" in data or "last_name" in data:
                self.leads_db[lead_id]["name"] = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            elif "First Name" in data or "Last Name" in data:
                self.leads_db[lead_id]["name"] = f"{data.get('First Name', '')} {data.get('Last Name', '')}".strip()
            elif "name" not in self.leads_db[lead_id]:
                self.leads_db[lead_id]["name"] = f"Lead {lead_id}"

        self._save_custom_leads()

    def get_all_leads(self, limit: int = 50) -> List[dict]:
        """Fetch all leads up to a limit."""
        # Sort leads: hot leads first
        sorted_leads = sorted(
            self.leads_db.values(),
            key=lambda x: x.get("conversion_probability", x.get("Conversion_Probability", 0.0)),
            reverse=True
        )
        return sorted_leads[:limit]

    def get_distributor_matches(self, investor_id: str, limit: int = 5) -> List[dict]:
        """Get matched distributors for an investor."""
        matches_csv_path = EXPORT_DIR / "investor_routing_matches.csv"
        
        # Default fallback distributors if file not found or ID not matched
        fallback_distributors = [
            {
                "distributor_name": "NJ IndiaInvest Pvt Ltd (National Distributor)",
                "match_score": 0.95,
                "fund_type": "Equity & High Growth Funds",
                "description": "Premium national distributor specializing in high-growth equity portfolios."
            },
            {
                "distributor_name": "Prudent Corporate Advisory Services Ltd",
                "match_score": 0.88,
                "fund_type": "Balanced & Hybrid Funds",
                "description": "Wealth management company providing diversified asset allocation strategies."
            },
            {
                "distributor_name": "State Bank of India (Wealth)",
                "match_score": 0.82,
                "fund_type": "Debt & Low Volatility Funds",
                "description": "Secure, conservative advisory services for stable capital appreciation."
            },
            {
                "distributor_name": "Zerodha Fund House / Coin",
                "match_score": 0.79,
                "fund_type": "Index & Passive Funds",
                "description": "Modern direct mutual fund platform optimized for passive index trackers."
            }
        ]

        if not matches_csv_path.is_file():
            return fallback_distributors[:limit]

        try:
            df = pd.read_csv(matches_csv_path)
            # Find the row matching the investor ID
            # investor_id could be a string like "demo_investor" or numeric string/int
            matching_row = None
            for _, row in df.iterrows():
                row_id = str(row.get("Investor_ID", ""))
                if row_id == str(investor_id):
                    matching_row = row
                    break
            
            if matching_row is not None:
                rec_dist = str(matching_row.get("Recommended_Distributor_To_Contact", ""))
                rec_fund = str(matching_row.get("Recommended_Fund_Type", "Mutual Funds"))
                persona = str(matching_row.get("Persona_Cluster", "Balanced Allocator"))
                
                # Make the recommended distributor the top match
                results = [
                    {
                        "distributor_name": rec_dist,
                        "match_score": 0.98,
                        "fund_type": rec_fund,
                        "persona": persona,
                        "description": f"Optimally matched distributor specializing in {rec_fund} for {persona} profiles."
                    }
                ]
                
                # Add other fallback distributors to fill limit
                for dist in fallback_distributors:
                    if dist["distributor_name"] != rec_dist:
                        results.append({
                            "distributor_name": dist["distributor_name"],
                            "match_score": round(dist["match_score"] - 0.05, 2),
                            "fund_type": dist["fund_type"],
                            "description": dist["description"]
                        })
                return results[:limit]
            
            return fallback_distributors[:limit]
        except Exception as e:
            print(f"Error matching distributors: {e}")
            return fallback_distributors[:limit]

    def get_collection(self, name: str) -> MockCollection:
        """Get collection interface."""
        return MockCollection(name, self)

# Instantiate singleton client
db_client = MockMongoClient()
