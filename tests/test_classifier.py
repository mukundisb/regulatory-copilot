import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from app import app
from config import settings
from app import lifespan

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

def test_missing_model_file_fails_startup(monkeypatch):
    """Verify application startup fails when model path points to a missing file."""
    # Temporarily change the model path to a non-existent file
    monkeypatch.setattr(settings, "model_path", "invalid/path/non_existent_model.joblib")

    test_app = FastAPI(lifespan = lifespan)

    # Assert that instantiating the TestClient context raises FileNotFoundError (or Exception)
    with pytest.raises(Exception) as exc_info:
        with TestClient(test_app):
            pass  # The context manager will attempt to start the app and load the model

    assert exc_info.typename in ['FileNotFoundError', 'NoSuchFileError'] or "No such file or directory" in str(exc_info.value)