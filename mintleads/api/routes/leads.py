"""Lead scoring routes for MintLeads API.

This module provides additional lead-related endpoints.
"""

from flask import Blueprint, jsonify, request

leads_bp = Blueprint("leads", __name__, url_prefix="/api/leads")


@leads_bp.route("/batch", methods=["POST"])
def batch_score():
    """Batch score multiple leads."""
    return jsonify({"message": "Batch scoring endpoint"})


@leads_bp.route("/<lead_id>/details", methods=["GET"])
def get_lead_details(lead_id: str):
    """Get detailed information about a specific lead."""
    return jsonify({"lead_id": lead_id, "details": {}})
