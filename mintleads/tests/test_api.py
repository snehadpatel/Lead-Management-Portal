"""Tests for Flask API.

This module contains unit tests for the REST API endpoints.
"""

import pytest

# Optional imports for API testing
try:
    from api.app import create_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    create_app = None


@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not installed")
class TestAPI:
    """Test cases for Flask API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_health_returns_200(self, client):
        """Test /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models" in data
    
    def test_leads_score_valid_payload(self, client):
        """Test /api/leads/score with valid payload."""
        payload = {
            "leads": [
                {
                    "Prospect ID": "TEST001",
                    "TotalVisits": 10,
                    "Total Time Spent on Website": 500,
                    "Page Views Per Visit": 3,
                }
            ]
        }
        
        response = client.post(
            "/api/leads/score",
            json=payload,
            content_type="application/json",
        )
        
        # May return 200 or 503 depending on model loading
        assert response.status_code in [200, 503]
    
    def test_leads_score_invalid_payload(self, client):
        """Test /api/leads/score with invalid payload."""
        payload = {"invalid": "data"}
        
        response = client.post(
            "/api/leads/score",
            json=payload,
            content_type="application/json",
        )
        
        assert response.status_code == 400
    
    def test_investors_segment_valid_payload(self, client):
        """Test /api/investors/segment with valid payload."""
        payload = {
            "investors": [
                {
                    "id": "INV001",
                    "current_age": 35,
                    "yearly_income": 75000,
                    "total_debt": 20000,
                    "debt_to_income_ratio": 0.27,
                    "credit_score": 720,
                }
            ]
        }
        
        response = client.post(
            "/api/investors/segment",
            json=payload,
            content_type="application/json",
        )
        
        # May return 200 or 503 depending on model loading
        assert response.status_code in [200, 503]
    
    def test_sentiment_current_returns_expected_keys(self, client):
        """Test /api/sentiment/current returns expected keys."""
        response = client.get("/api/sentiment/current")
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert "signal" in data
        assert "score" in data
        assert "last_updated" in data
        assert data["signal"] in ["Bullish", "Neutral", "Bearish"]
    
    def test_nav_forecast_requires_scheme_code(self, client):
        """Test /api/nav/forecast requires scheme_code."""
        response = client.get("/api/nav/forecast")
        
        assert response.status_code == 400
        
        data = response.get_json()
        assert "error" in data
    
    def test_nav_forecast_valid_request(self, client):
        """Test /api/nav/forecast with valid request."""
        response = client.get("/api/nav/forecast?scheme_code=TEST001&days=30")
        
        # May return 200 or 503 depending on model loading
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
