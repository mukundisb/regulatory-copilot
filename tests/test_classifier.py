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

def test_retrieve_db_crash(client, monkeypatch):
    """Test /retrieve endpoint behavior when the underlying vector database crashes."""
    def mock_crash(*args, **kwargs):
        raise RuntimeError("ChromaDB database connection lost")

    # Override query_store using pytest's built-in monkeypatch fixture
    monkeypatch.setattr("rag_pipeline.query_store", mock_crash)

    payload = {"narrative": "What are the language requirements for device labels?"}
    with pytest.raises(RuntimeError) as exc_info:
        client.post("/retrieve", json=payload)

    assert "ChromaDB database connection lost" in str(exc_info.value)


def test_retrieve_empty_query_string(client):
    """Test /retrieve with empty/whitespace query string (triggers ClassifyRequest 422 validation)."""
    payload = {"narrative": "   "}
    response = client.post("/retrieve", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_retrieve_unrelated_input_query(client, monkeypatch):
    """Test /retrieve with an out-of-domain query string that yields no vector matches."""
    # Monkeypatch query_store to return an empty list without external mock libraries
    monkeypatch.setattr("rag_pipeline.query_store", lambda *args, **kwargs: [])

    payload = {"narrative": "How do I bake a double chocolate sourdough bread at home?"}
    response = client.post("/retrieve", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

def test_retrieve_valid_query_mocked(client, monkeypatch):
    """Test /retrieve returns structured non-empty results without requiring a live DB."""
    mock_results = [
        {
            "chunk_id": "doc_chunk_283",
            "section": "ANNEX I - GENERAL SAFETY AND PERFORMANCE REQUIREMENTS",
            "text": "[ANNEX I] REQUIREMENTS REGARDING THE INFORMATION SUPPLIED WITH THE DEVICE...",
            "similarity_score": "0.6947"
        },
        {
            "chunk_id": "doc_chunk_74",
            "section": "Article 16 - Cases in which obligations of manufacturers apply...",
            "text": "[Article 16] Distributors or importers carrying out translation...",
            "similarity_score": "0.6117"
        }
    ]

    monkeypatch.setattr("rag_pipeline.query_store", lambda *args, **kwargs: mock_results)

    payload = {
        "narrative": "In what language must device labels and packaging information be provided?"
    }
    response = client.post("/retrieve", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["chunk_id"] == "doc_chunk_283"
    assert "ANNEX I" in data[0]["section"]