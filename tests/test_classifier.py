import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_classify_valid_input(client):
    """Test standard valid narrative payload."""
    payload = {"narrative": "Patient experienced severe allergic reaction following administration."}
    response = client.post("/classify", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "probabilities" in data
    assert data["predicted_label"] in ['D', 'I', 'M', 'O']

def test_classify_empty_narrative(client):
    """Test empty narrative (triggers Pydantic 422 validation)."""
    payload = {"narrative": " "}
    response = client.post("/classify", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()

def test_classify_malformed_input(client):
    """Test invalid JSON schema / non-string narrative input."""
    payload = {"narrative": 12345}  # Non-string input
    response = client.post("/classify", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()