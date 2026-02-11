"""Basic tests for QS Parser Service"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test that health endpoint returns OK"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ocr_available" in data


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200


def test_parse_no_file():
    """Test parse endpoint without file returns error"""
    response = client.post("/parse")
    assert response.status_code == 422  # Validation error


def test_parse_wrong_type():
    """Test parse endpoint with unsupported file type returns error"""
    response = client.post(
        "/parse",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
