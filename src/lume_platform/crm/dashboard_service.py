"""
CRM Dashboard Service: Manages distributor workflow updates and formats leads
specifically for the CRM dashboard visualizations and filtering query operations.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional
from lume_platform.db.mongo_client import db_client

def parse_investment_amount(val: Any) -> float:
    """Parses potential investment string like '₹45L', '₹5L', or numeric values to float in INR."""
    if not val:
        return 500000.0  # default 5 Lakhs
    
    val_str = str(val).strip().replace(" ", "").replace(",", "")
    
    # Try to extract numbers
    nums = re.findall(r"\d+\.?\d*", val_str)
    if not nums:
        return 500000.0
        
    num = float(nums[0])
    
    # Check suffixes
    if "cr" in val_str.lower():
        return num * 10000000.0  # 1 Crore = 10,000,000
    elif "l" in val_str.lower() or "lakh" in val_str.lower():
        return num * 100000.0    # 1 Lakh = 100,000
    elif "k" in val_str.lower():
        return num * 1000.0
    
    # If the number is small (e.g. 5, 12, 45) and no Lakh/Cr unit is specified, assume it is Lakhs
    if num < 1000:
        return num * 100000.0
        
    return num

def build_dashboard_overview() -> dict:
    """
    Computes aggregated summary stats across all active leads.
    Used for Distributor KPIs.
    """
    if getattr(db_client, "use_real_mongo", False):
        total_leads = db_client.db.leads.count_documents({})
        if total_leads == 0:
            return {
                "total_leads": 0,
                "hot_prospects": 0,
                "mean_conversion_probability": 0.0,
                "conversion_rate": "0%",
                "pipeline_value": 0.0,
                "recent_activities": []
            }
            
        hot_count = db_client.db.leads.count_documents({"conversion_probability": {"$gt": 0.85}})
        
        # Avg conversion probability
        agg_res = list(db_client.db.leads.aggregate([
            {"$group": {"_id": None, "avg_prob": {"$avg": "$conversion_probability"}}}
        ]))
        mean_prob = agg_res[0]["avg_prob"] if agg_res else 0.5
        
        # Pipeline value
        pipeline_cursor = db_client.db.leads.find({}, {"potential_investment": 1, "Potential_Investment": 1, "_id": 0})
        total_pipeline = sum(parse_investment_amount(doc.get("potential_investment") or doc.get("Potential_Investment")) for doc in pipeline_cursor)
        
        # Recent activities
        recent_activities = []
        import pymongo
        sorted_leads = db_client.db.leads.find().sort("conversion_probability", pymongo.DESCENDING).limit(5)
        for lead in sorted_leads:
            status = lead.get("status", "New")
            name = lead.get("name", "Unknown Lead")
            recent_activities.append({
                "timestamp": datetime.utcnow().isoformat(),
                "lead_id": lead.get("lead_id"),
                "name": name,
                "activity": f"Lead state set to {status}" if status != "New" else f"AI scored conversion at {round(float(lead.get('conversion_probability', 0.5))*100)}%"
            })
            
        return {
            "total_leads": total_leads,
            "hot_prospects": hot_count,
            "mean_conversion_probability": round(mean_prob, 4),
            "conversion_rate": f"{round(mean_prob * 100)}%",
            "pipeline_value": round(total_pipeline, 2),
            "recent_activities": recent_activities
        }

    all_leads = db_client.get_all_leads(limit=10000000)
    
    total_leads = len(all_leads)
    if total_leads == 0:
        return {
            "total_leads": 0,
            "hot_prospects": 0,
            "mean_conversion_probability": 0.0,
            "conversion_rate": "0%",
            "pipeline_value": 0.0,
            "recent_activities": []
        }
    
    probabilities = []
    hot_count = 0
    total_pipeline = 0.0
    
    for lead in all_leads:
        prob = float(lead.get("conversion_probability", lead.get("Conversion_Probability", 0.5)))
        probabilities.append(prob)
        if prob > 0.85:
            hot_count += 1
            
        inv_str = lead.get("potential_investment", lead.get("Potential_Investment", ""))
        total_pipeline += parse_investment_amount(inv_str)
        
    mean_prob = sum(probabilities) / len(probabilities)
    
    recent_activities = []
    sorted_leads = sorted(all_leads, key=lambda x: float(x.get("conversion_probability", 0.0)), reverse=True)
    for i, lead in enumerate(sorted_leads[:5]):
        status = lead.get("status", "New")
        name = lead.get("name", "Unknown Lead")
        recent_activities.append({
            "timestamp": datetime.utcnow().isoformat(),
            "lead_id": lead.get("lead_id"),
            "name": name,
            "activity": f"Lead state set to {status}" if status != "New" else f"AI scored conversion at {round(float(lead.get('conversion_probability', 0.5))*100)}%"
        })

    return {
        "total_leads": total_leads,
        "hot_prospects": hot_count,
        "mean_conversion_probability": round(mean_prob, 4),
        "conversion_rate": f"{round(mean_prob * 100)}%",
        "pipeline_value": round(total_pipeline, 2),
        "recent_activities": recent_activities
    }

def build_dashboard_leads(
    limit: int = 20,
    query: Optional[str] = None,
    stage: Optional[str] = None,
    priority: Optional[str] = None
) -> List[dict]:
    """
    Returns filtered and prioritized leads list matching search query, workflow stage,
    or priority category.
    """
    if getattr(db_client, "use_real_mongo", False):
        mongo_query = {}
        if stage:
            mongo_query["status"] = {"$regex": f"^{stage}$", "$options": "i"}
            
        if priority:
            p_upper = priority.upper()
            if p_upper == "HOT":
                mongo_query["conversion_probability"] = {"$gt": 0.85}
            elif p_upper == "WARM":
                mongo_query["conversion_probability"] = {"$gt": 0.65, "$lte": 0.85}
            elif p_upper == "COLD":
                mongo_query["conversion_probability"] = {"$lte": 0.65}
                
        if query:
            rgx = {"$regex": query, "$options": "i"}
            mongo_query["$or"] = [
                {"name": rgx},
                {"city": rgx},
                {"occupation": rgx},
                {"lead_id": rgx},
                {"industry": rgx},
                {"lead_source": rgx}
            ]
            
        import pymongo
        cursor = db_client.db.leads.find(mongo_query).sort("conversion_probability", pymongo.DESCENDING).limit(limit)
        leads = list(cursor)
        for lead in leads:
            if "_id" in lead:
                lead["_id"] = str(lead["_id"])
        return leads

    all_leads = db_client.get_all_leads(limit=10000000)
    filtered = []
    
    for lead in all_leads:
        if query:
            q = query.lower()
            name = lead.get("name", "").lower()
            city = lead.get("city", "").lower()
            occupation = lead.get("occupation", "").lower()
            lead_id = lead.get("lead_id", "").lower()
            industry = lead.get("industry", "").lower()
            source = lead.get("lead_source", "").lower()
            
            if not (q in name or q in city or q in occupation or q in lead_id or q in industry or q in source):
                continue
                
        if stage:
            lead_stage = lead.get("status", "New").lower()
            if lead_stage != stage.lower():
                continue
                
        if priority:
            prob = float(lead.get("conversion_probability", lead.get("Conversion_Probability", 0.5)))
            lead_priority = "COLD"
            if prob > 0.85:
                lead_priority = "HOT"
            elif prob > 0.65:
                lead_priority = "WARM"
                
            if lead_priority != priority.upper():
                continue
                
        filtered.append(lead)
        
    return filtered[:limit]

def update_lead_workflow(
    lead_id: str,
    status: str,
    notes: str = "",
    assignee: Optional[str] = None,
    next_step_at: Optional[str] = None
) -> dict:
    """
    Updates status and details of a managed lead workflow, persisting the state.
    """
    updates = {
        "status": status,
        "notes": notes,
        "updated_at": datetime.utcnow().isoformat()
    }
    if assignee:
        updates["assignee"] = assignee
    if next_step_at:
        updates["next_step_at"] = next_step_at
        
    db_client.upsert_lead(lead_id, updates)
    
    # Retrieve updated lead to return
    updated_lead = db_client.get_collection("leads").find_one({"lead_id": lead_id})
    
    return {
        "status": "success",
        "lead_id": lead_id,
        "workflow": {
            "status": status,
            "notes": notes,
            "assignee": assignee,
            "next_step_at": next_step_at
        },
        "lead_details": updated_lead
    }
