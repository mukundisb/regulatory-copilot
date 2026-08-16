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

    assert isinstance(data[0]['similarity_score'], float)
    assert data[0]['similarity_score'] == 0.6947
    assert isinstance(data,list)
    assert len(data) == 2
    assert data[0]['chunk_id'] == "doc_chunk_283"
    assert "ANNEX I" in data[0]["section"]

def test_retrieve_e2e_real_store(client):
    """Real end-to-end test exercising sentence-transformers and ChromaDB."""
    payload = {
        "narrative": "In what language must device labels, packaging information, and instructions for use be provided?"
    }
    response = client.post("/retrieve", json=payload)

    assert response.status_code == 200
    data = response.json()
    
    assert len(data) > 0
    top_result = data[0]
    assert isinstance(top_result["similarity_score"], float)
    assert any(header in top_result["section"] for header in ["ANNEX I", "Article"])

def test_assess_death_branch_orchestration(client, monkeypatch):
    """Verifies that predicted label 'D' branches into a vigilance query and returns structured advice."""
    monkeypatch.setattr("app.predict_single", lambda pipeline, text: {"predicted_label": "D", "probabilities": {"D": 0.94, "I": 0.03, "M": 0.02, "O": 0.01}})
    
    mock_chunks = [{
        "chunk_id": "doc_chunk_180",
        "section": "Article 87 - Reporting of serious incidents and field safety corrective actions",
        "text": "[Article 87] Manufacturers shall report any serious incident...",
        "similarity_score": 0.7250
    }]
    monkeypatch.setattr("app.query_store", lambda q, top_k: mock_chunks)

    payload = {"narrative": "Patient passed away during surgery after stent dislodged."}
    response = client.post("/assess", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "D"
    assert data["confidence"] == 0.94
    assert "serious incident reporting vigilance" in data["retrieval_query_used"]
    assert len(data["retrieved_chunks"]) == 1
    assert "Article 87" in data["recommendation"]
    assert "Mandatory vigilance reporting" in data["recommendation"]


def test_assess_malfunction_branch_orchestration(client, monkeypatch):
    """Verifies that predicted label 'M' branches into a CAPA/trend analysis query."""
    monkeypatch.setattr("app.predict_single", lambda pipeline, text: {"predicted_label": "M", "probabilities": {"D": 0.01, "I": 0.04, "M": 0.88, "O": 0.07}})
    
    mock_chunks = [{
        "chunk_id": "doc_chunk_182",
        "section": "Article 88 - Trend reporting",
        "text": "[Article 88] Manufacturers shall report any statistically significant increase...",
        "similarity_score": 0.6810
    }]
    monkeypatch.setattr("app.query_store", lambda q, top_k: mock_chunks)

    payload = {"narrative": "Display went blank during self-test routine."}
    response = client.post("/assess", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "M"
    assert "device malfunction root cause analysis" in data["retrieval_query_used"]
    assert "Article 88" in data["recommendation"]


def test_assess_empty_payload_validation(client):
    """Ensures Pydantic validation on ClassifyRequest is enforced on /assess."""
    response = client.post("/assess", json={"narrative": "   "})
    assert response.status_code == 422

def test_assess_e2e_real_pipeline(client):
    """End-to-end integration test: exercises real ML classifier + persistent ChromaDB store."""
    payload = {
        "narrative": "Patient died of acute myocardial infarction following catheter fracture and embolization."
    }
    response = client.post("/assess", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Classification assertions
    assert data["predicted_label"] in ["D", "I"], (
        f"Upstream model regression: expected 'D' or 'I' for death narrative, "
        f" but got '{data['predicted_label']}' (confidence: {data.get('confidence')})"
    )
    assert 0.0 <= data["confidence"] <= 1.0
    
    # Branching query assertions (Death narrative should trigger vigilance keywords)
    assert "serious incident reporting vigilance" in data["retrieval_query_used"]
    
    # Retrieval assertions
    assert len(data["retrieved_chunks"]) > 0
    assert isinstance(data["retrieved_chunks"][0]["similarity_score"], float)
    
    # Recommendation assertions
    assert "Primary regulatory basis:" in data["recommendation"]

def test_assess_e2e_malfunction_branch_real_pipeline(client):
    """E2E Test: Malfunction narrative classifies as M and triggers CAPA/investigation query steering."""
    payload = {
        "narrative": "Infusion pump screen froze and stopped delivery of medication, showing error code E-402."
    }
    response = client.post("/assess", json=payload)

    assert response.status_code == 200
    data = response.json()

    # 1. Assert upstream classification
    assert data["predicted_label"] == "M", (
        f"Expected label 'M', got '{data['predicted_label']}'"
    )
    assert 0.0 <= data["confidence"] <= 1.0

    # 2. Assert query reformulation
    assert "device malfunction root cause analysis" in data["retrieval_query_used"]
    assert payload["narrative"] in data["retrieval_query_used"]

    # 3. Assert retrieval results & recommendation
    assert len(data["retrieved_chunks"]) > 0
    assert "Device malfunction identified" in data["recommendation"]

# ============================================================================
# MOCKED DECISION-POINT ISOLATION TESTS
# ============================================================================

@pytest.mark.parametrize("mock_label, expected_query_prefix", [
    ("D", "serious incident reporting vigilance timelines manufacturer obligations"),
    ("I", "serious incident reporting vigilance timelines manufacturer obligations"),
    ("M", "device malfunction root cause analysis trend reporting corrective action"),
    ("O", None),  # 'O' passes the raw narrative without prefix
])
def test_assess_decision_branch_reformulation(client, monkeypatch, mock_label, expected_query_prefix):
    """Unit Test: Proves the decision point dynamically generates the exact intended retrieval query."""
    # 1. Mock classifier to force specific label
    monkeypatch.setattr(
        "app.predict_single",
        lambda *args, **kwargs: {
            "predicted_label": mock_label,
            "probabilities": {mock_label: 0.95, "other": 0.05}
        }
    )

    # 2. Mock query_store to capture the exact query sent to retrieval
    captured_queries = []
    def mock_query_store(query_string, top_k=3):
        captured_queries.append(query_string)
        return [{
            "chunk_id": "mock_chunk_1",
            "section": "Article 87",
            "text": "Mock text",
            "similarity_score": 0.85
        }]

    monkeypatch.setattr("app.query_store", mock_query_store)

    raw_narrative = "Sample device event description for unit test."
    response = client.post("/assess", json={"narrative": raw_narrative})

    assert response.status_code == 200
    data = response.json()

    # 3. Assert the exact query string constructed
    assert len(captured_queries) == 1
    actual_query = captured_queries[0]

    if expected_query_prefix:
        expected_full_query = f"{expected_query_prefix} {raw_narrative}"
        assert actual_query == expected_full_query
        assert data["retrieval_query_used"] == expected_full_query
    else:
        assert actual_query == raw_narrative
        assert data["retrieval_query_used"] == raw_narrative