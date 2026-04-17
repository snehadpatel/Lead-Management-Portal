"""NAV forecasting routes for MintLeads API.

This module provides additional NAV-related endpoints.
"""

from flask import Blueprint, jsonify, request

nav_bp = Blueprint("nav", __name__, url_prefix="/api/nav")


@nav_bp.route("/schemes", methods=["GET"])
def list_schemes():
    """List all available mutual fund schemes."""
    return jsonify({"schemes": []})


@nav_bp.route("/history/<scheme_code>", methods=["GET"])
def get_nav_history(scheme_code: str):
    """Get historical NAV data for a scheme."""
    days = request.args.get("days", 30, type=int)
    return jsonify({
        "scheme_code": scheme_code,
        "days": days,
        "history": [],
    })
