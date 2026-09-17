from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app import app
from rag_recommender import StructuredRecommendation

client = TestClient(app)


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-mock-key"})
@patch("rag_recommender.genai.Client")
@patch("app.query_store")
@patch("app._classify_narrative")
def test_assess_endpoint_mocked_gemini_structural(
    mock_classify,
    mock_query,
    mock_genai_client_cls,
):
    """
    Assert /assess endpoint response shape, grounded citations, and schema fields
    hermetically without external GCP credentials or local vector store state.
    """
    # 1. Mock upstream classification layer
    mock_classify.return_value = {
        "predicted_label": "D",
        "confidence": 0.9650,
        "probabilities": {"D": 0.9650, "I": 0.02, "M": 0.01, "O": 0.005},
        "backend_used": "mocked_vertex",
    }

    # 2. Mock RAG retrieval layer
    controlled_section = "Article 87 - Reporting of serious incidents and field safety corrective actions"
    mock_query.return_value = [
        {
            "chunk_id": "mock_chunk_204",
            "section": controlled_section,
            "text": "Manufacturers of devices shall report to the competent authorities any serious incident...",
            "similarity_score": 0.7820,
        }
    ]

    # 3. Setup controlled mock response for Gemini
    controlled_rec_text = (
        "The manufacturer must submit a serious incident report to the competent "
        "authority within statutory deadlines (no later than 10 days) per Article 87."
    )
    mock_parsed_payload = StructuredRecommendation(
        recommendation=controlled_rec_text,
        citations=[controlled_section],
    )

    mock_gemini_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = mock_parsed_payload
    mock_gemini_instance.models.generate_content.return_value = mock_response
    mock_genai_client_cls.return_value = mock_gemini_instance

    # 4. Invoke /assess
    payload = {
        "narrative": "Pacemaker lead dislodged resulting in patient cardiac arrest and emergency resuscitation."
    }
    response = client.post("/assess", json=payload)

    # 5. Assert HTTP status & top-level schema contract
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_label"] == "D"
    assert data["confidence"] == 0.9650
    assert "retrieval_query_used" in data
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["chunk_id"] == "mock_chunk_204"

    # 6. Assert LLM recommendation fields and strict citation containment
    assert data["recommendation"] == controlled_rec_text
    assert data["citations"] == [controlled_section]
    assert data["llm_grounding_verified"] is True
    assert data["fallback_triggered"] is False
    assert data["backend_used"] == "mocked_vertex"

    retrieved_sections = {c["section"] for c in data["retrieved_chunks"]}
    for citation in data["citations"]:
        assert citation in retrieved_sections