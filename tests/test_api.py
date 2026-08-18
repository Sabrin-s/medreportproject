"""
Integration tests for FastAPI server REST API endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_api_analyze_report():
    response = client.post(
        "/api/analyze",
        data={"text": "Patient age 55. BP 138/84 mmHg. ECG demonstrates sinus rhythm with rare PACs."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "specialty" in data
    assert "explanation" in data
    assert "execution_pipeline" in data

def test_api_knowledge_list():
    response = client.get("/api/knowledge")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
