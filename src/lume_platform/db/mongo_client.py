"""
MongoDB Client: A production-ready client that supports real MongoDB connections
with connection pooling, automated index creation, and fallbacks to a local file-based
database when MONGO_URI is not set or the connection fails.
"""

from __future__ import annotations

import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

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
        if self.name == "leads":
            return list(self.db_client.leads_db.values())
        return []

class MockMongoClient:
    def __init__(self):
        self.leads_db: Dict[str, dict] = {}
        self.users_db: Dict[str, dict] = {}
        self.portfolios_db: Dict[str, list] = {}
        self.custom_leads_file = EXPORT_DIR / "db_custom_leads.json"
        self.users_file = EXPORT_DIR / "db_users.json"
        self.portfolios_file = EXPORT_DIR / "db_portfolios.json"
        if "VERCEL" in os.environ:
            self.custom_leads_file = Path("/tmp") / "db_custom_leads.json"
            self.users_file = Path("/tmp") / "db_users.json"
            self.portfolios_file = Path("/tmp") / "db_portfolios.json"

        # Check for real MongoDB URI
        self.mongo_uri = os.environ.get("MONGO_URI")
        self.use_real_mongo = False
        self.client = None
        self.db = None

        if self.mongo_uri:
            try:
                # 5-second timeout for server selection check
                self.client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                # Force a connection test to ping the database
                self.client.admin.command('ping')
                self.db = self.client.get_database("lume_db")
                self.use_real_mongo = True
                print("✅ Connected to real MongoDB.")
                self._setup_indices()
            except Exception as e:
                print(f"⚠️ Failed to connect to MongoDB ({e}). Falling back to local file-based database.")
                self.use_real_mongo = False

        if not self.use_real_mongo:
            self._load_initial_data()
            self._load_users()
            self._load_portfolios()

    def _setup_indices(self) -> None:
        """Create database indices automatically on startup."""
        try:
            self.db.users.create_index("email", unique=True)
            self.db.leads.create_index("lead_id", unique=True)
            self.db.leads.create_index("conversion_probability")
            self.db.portfolios.create_index("email", unique=True)
            print("✅ MongoDB indices verified.")
        except Exception as e:
            print(f"⚠️ Error setting up MongoDB indices: {e}")

    def _load_initial_data(self) -> None:
        """Seed leads and matches from output_production_final CSV exports."""
        # 1. Load distributor leads master CSV
        leads_csv_path = EXPORT_DIR / "distributor_leads_master.csv"
        if leads_csv_path.is_file():
            try:
                df = pd.read_csv(leads_csv_path)
                for _, row in df.iterrows():
                    lead_id = str(row.get("Lead Number", f"L-{1000 + _}"))
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
                        self.leads_db[lead_id] = lead_data
                print(f"✅ Loaded persistent custom leads from {self.custom_leads_file}")
            except Exception as e:
                print(f"⚠️ Error loading custom leads JSON: {e}")

    def _save_custom_leads(self) -> None:
        """Write custom/updated leads to JSON file for persistence (mock mode only)."""
        if self.use_real_mongo:
            return
        try:
            with open(self.custom_leads_file, "w") as f:
                json.dump(self.leads_db, f, indent=4)
        except Exception as e:
            print(f"⚠️ Error writing custom leads JSON: {e}")

    def _load_users(self) -> None:
        """Load persistent user profiles from JSON file (mock mode only)."""
        if self.use_real_mongo:
            return
        if self.users_file.is_file():
            try:
                with open(self.users_file, "r") as f:
                    self.users_db = json.load(f)
                print(f"✅ Loaded {len(self.users_db)} persistent user profiles from {self.users_file}")
            except Exception as e:
                print(f"⚠️ Error loading users JSON: {e}")

    def _save_users(self) -> None:
        """Save persistent user profiles to JSON file (mock mode only)."""
        if self.use_real_mongo:
            return
        try:
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump(self.users_db, f, indent=4)
        except Exception as e:
            print(f"⚠️ Error writing users JSON: {e}")

    def get_user(self, email: str) -> Optional[dict]:
        """Fetch user by email."""
        email_clean = email.strip().lower()
        if self.use_real_mongo:
            user = self.db.users.find_one({"email": email_clean})
            if user and "_id" in user:
                user["_id"] = str(user["_id"])
            return user
        return self.users_db.get(email_clean)

    def create_user(self, email: str, password_hash: str, role: str) -> dict:
        """Create a new user profile."""
        email_clean = email.strip().lower()
        user_data = {
            "email": email_clean,
            "password_hash": password_hash,
            "role": role,
            "persona": "balanced",  # Default persona
            "created_at": str(pd.Timestamp.now())
        }
        if self.use_real_mongo:
            self.db.users.insert_one(user_data.copy())
            inserted = self.db.users.find_one({"email": email_clean})
            if inserted and "_id" in inserted:
                inserted["_id"] = str(inserted["_id"])
            return inserted
        else:
            self.users_db[email_clean] = user_data
            self._save_users()
            return user_data

    def update_user_profile(self, email: str, profile: dict) -> Optional[dict]:
        """Update user profile fields."""
        email_clean = email.strip().lower()
        if self.use_real_mongo:
            self.db.users.update_one({"email": email_clean}, {"$set": profile})
            user = self.db.users.find_one({"email": email_clean})
            if user and "_id" in user:
                user["_id"] = str(user["_id"])
            return user
        else:
            if email_clean in self.users_db:
                self.users_db[email_clean].update(profile)
                self._save_users()
                return self.users_db[email_clean]
            return None

    def upsert_lead(self, lead_id: str, data: dict) -> None:
        """Upsert lead details."""
        lead_id = str(lead_id)
        
        # Clean data keys for MongoDB suitability (if any nested keys have '.' or '$')
        clean_data = {}
        for k, v in data.items():
            k_clean = k.replace(".", "_").replace("$", "_")
            clean_data[k_clean] = v

        if self.use_real_mongo:
            existing = self.db.leads.find_one({"lead_id": lead_id})
            if existing:
                self.db.leads.update_one({"lead_id": lead_id}, {"$set": clean_data})
            else:
                lead_doc = {
                    "lead_id": lead_id,
                    "status": "New",
                    "notes": "",
                    "assignee": "",
                    "next_step_at": "",
                    **clean_data
                }
                if "first_name" in clean_data or "last_name" in clean_data:
                    lead_doc["name"] = f"{clean_data.get('first_name', '')} {clean_data.get('last_name', '')}".strip()
                elif "First Name" in clean_data or "Last Name" in clean_data:
                    lead_doc["name"] = f"{clean_data.get('First Name', '')} {clean_data.get('Last Name', '')}".strip()
                elif "name" not in lead_doc:
                    lead_doc["name"] = f"Lead {lead_id}"
                self.db.leads.insert_one(lead_doc)
        else:
            if lead_id in self.leads_db:
                self.leads_db[lead_id].update(clean_data)
            else:
                self.leads_db[lead_id] = {
                    "lead_id": lead_id,
                    "status": "New",
                    "notes": "",
                    "assignee": "",
                    "next_step_at": "",
                    **clean_data
                }
                if "first_name" in clean_data or "last_name" in clean_data:
                    self.leads_db[lead_id]["name"] = f"{clean_data.get('first_name', '')} {clean_data.get('last_name', '')}".strip()
                elif "First Name" in clean_data or "Last Name" in clean_data:
                    self.leads_db[lead_id]["name"] = f"{clean_data.get('First Name', '')} {clean_data.get('Last Name', '')}".strip()
                elif "name" not in self.leads_db[lead_id]:
                    self.leads_db[lead_id]["name"] = f"Lead {lead_id}"

            self._save_custom_leads()

    def get_all_leads(self, limit: int = 50) -> List[dict]:
        """Fetch all leads up to a limit."""
        if self.use_real_mongo:
            cursor = self.db.leads.find().sort("conversion_probability", pymongo.DESCENDING).limit(limit)
            leads = list(cursor)
            for lead in leads:
                if "_id" in lead:
                    lead["_id"] = str(lead["_id"])
            return leads
        else:
            sorted_leads = sorted(
                self.leads_db.values(),
                key=lambda x: x.get("conversion_probability", x.get("Conversion_Probability", 0.0)),
                reverse=True
            )
            return sorted_leads[:limit]

    def get_distributor_matches(self, investor_id: str, limit: int = 5) -> List[dict]:
        """Get matched distributors for an investor."""
        matches_csv_path = EXPORT_DIR / "investor_routing_matches.csv"
        
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
                
                results = [
                    {
                        "distributor_name": rec_dist,
                        "match_score": 0.98,
                        "fund_type": rec_fund,
                        "persona": persona,
                        "description": f"Optimally matched distributor specializing in {rec_fund} for {persona} profiles."
                    }
                ]
                
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

    def get_collection(self, name: str) -> Any:
        """Get collection interface."""
        if self.use_real_mongo:
            return self.db[name]
        return MockCollection(name, self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Portfolio Management — Persistent per-user investment tracking
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_portfolios(self) -> None:
        """Load persistent portfolio data from JSON file (mock mode only)."""
        if self.use_real_mongo:
            return
        if self.portfolios_file.is_file():
            try:
                with open(self.portfolios_file, "r") as f:
                    self.portfolios_db = json.load(f)
                print(f"✅ Loaded {len(self.portfolios_db)} user portfolios")
            except Exception as e:
                print(f"⚠️ Error loading portfolios JSON: {e}")

    def _save_portfolios(self) -> None:
        """Save portfolio data to JSON file (mock mode only)."""
        if self.use_real_mongo:
            return
        try:
            self.portfolios_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.portfolios_file, "w") as f:
                json.dump(self.portfolios_db, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error writing portfolios JSON: {e}")

    def get_portfolio(self, email: str) -> List[dict]:
        """Get all holdings for a user."""
        email_clean = email.strip().lower()
        if self.use_real_mongo:
            doc = self.db.portfolios.find_one({"email": email_clean})
            return doc.get("holdings", []) if doc else []
        return self.portfolios_db.get(email_clean, [])

    def add_holding(self, email: str, holding: dict) -> dict:
        """Add a new investment to user's portfolio."""
        import uuid
        email_clean = email.strip().lower()
        holding["holding_id"] = str(uuid.uuid4())[:8]
        holding["added_at"] = str(pd.Timestamp.now())

        if self.use_real_mongo:
            self.db.portfolios.update_one(
                {"email": email_clean},
                {"$push": {"holdings": holding}},
                upsert=True
            )
            return holding
        else:
            if email_clean not in self.portfolios_db:
                self.portfolios_db[email_clean] = []
            self.portfolios_db[email_clean].append(holding)
            self._save_portfolios()
            return holding

    def update_holding(self, email: str, holding_id: str, updates: dict) -> Optional[dict]:
        """Update units/details for an existing holding."""
        email_clean = email.strip().lower()
        if self.use_real_mongo:
            set_fields = {}
            for k, v in updates.items():
                set_fields[f"holdings.$.{k}"] = v
            
            result = self.db.portfolios.update_one(
                {"email": email_clean, "holdings.holding_id": holding_id},
                {"$set": set_fields}
            )
            if result.modified_count > 0 or result.matched_count > 0:
                doc = self.db.portfolios.find_one({"email": email_clean})
                if doc:
                    for h in doc.get("holdings", []):
                        if h.get("holding_id") == holding_id:
                            return h
            return None
        else:
            holdings = self.portfolios_db.get(email_clean, [])
            for h in holdings:
                if h.get("holding_id") == holding_id:
                    h.update(updates)
                    self._save_portfolios()
                    return h
            return None

    def remove_holding(self, email: str, holding_id: str) -> bool:
        """Remove a holding from user's portfolio."""
        email_clean = email.strip().lower()
        if self.use_real_mongo:
            result = self.db.portfolios.update_one(
                {"email": email_clean},
                {"$pull": {"holdings": {"holding_id": holding_id}}}
            )
            return result.modified_count > 0
        else:
            holdings = self.portfolios_db.get(email_clean, [])
            original_len = len(holdings)
            self.portfolios_db[email_clean] = [
                h for h in holdings if h.get("holding_id") != holding_id
            ]
            if len(self.portfolios_db[email_clean]) < original_len:
                self._save_portfolios()
                return True
            return False

    def get_portfolio_summary(self, email: str) -> dict:
        """Get portfolio value summary (before NAV enrichment)."""
        holdings = self.get_portfolio(email)
        if not holdings:
            return {
                "total_invested": 0,
                "holding_count": 0,
                "holdings": [],
            }
        total_invested = sum(float(h.get("buy_value", 0)) for h in holdings)
        return {
            "total_invested": round(total_invested, 2),
            "holding_count": len(holdings),
            "holdings": holdings,
        }

# Instantiate singleton client
db_client = MockMongoClient()
