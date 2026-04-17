"""Investor clustering routes for MintLeads API.

This module provides additional investor-related endpoints.
"""

from flask import Blueprint, jsonify

clusters_bp = Blueprint("clusters", __name__, url_prefix="/api/investors")


@clusters_bp.route("/profiles", methods=["GET"])
def get_cluster_profiles():
    """Get cluster profile summaries."""
    return jsonify({
        "profiles": [
            {"id": 0, "name": "Conservative", "description": "Low risk, debt-focused"},
            {"id": 1, "name": "Balanced", "description": "Medium risk, hybrid approach"},
            {"id": 2, "name": "Aggressive", "description": "High risk, equity-focused"},
        ]
    })


@clusters_bp.route("/recommendations/<cluster_id>", methods=["GET"])
def get_recommendations(cluster_id: str):
    """Get fund recommendations for a cluster."""
    return jsonify({"cluster_id": cluster_id, "recommendations": []})
