from unittest.mock import MagicMock
from rag_recommender import generate_grounded_recommendation, StructuredRecommendation


def test_citation_containment_valid():
    mock_chunks = [
        {"chunk_id": "c1", "section": "Article 87 - Vigilance Reporting", "text": "Report within 15 days.", "similarity_score": 0.72},
        {"chunk_id": "c2", "section": "Article 89 - Analysis of Incidents", "text": "Investigate root cause.", "similarity_score": 0.65},
    ]

    mock_client = MagicMock()
    mock_parsed = StructuredRecommendation(
        recommendation="Mandatory reporting required under Article 87.",
        citations=["Article 87 - Vigilance Reporting"],
    )
    mock_response = MagicMock()
    mock_response.parsed = mock_parsed
    mock_client.models.generate_content.return_value = mock_response

    res = generate_grounded_recommendation(
        narrative="Lead fractured causing shock",
        predicted_label="D",
        chunks=mock_chunks,
        retrieval_fallback_triggered=False,
        top_score=0.72,
        client=mock_client,
    )

    assert res["llm_grounding_verified"] is True
    assert res["citations"] == ["Article 87 - Vigilance Reporting"]
    assert set(res["citations"]).issubset({c["section"] for c in mock_chunks})


def test_citation_containment_strips_hallucinations():
    mock_chunks = [
        {"chunk_id": "c1", "section": "Article 87 - Vigilance Reporting", "text": "Report within 15 days.", "similarity_score": 0.72}
    ]

    mock_client = MagicMock()
    mock_parsed = StructuredRecommendation(
        recommendation="Notify authorities per Article 87 and Article 10.",
        citations=["Article 87 - Vigilance Reporting", "Article 10 - General Obligations"],
    )
    mock_response = MagicMock()
    mock_response.parsed = mock_parsed
    mock_client.models.generate_content.return_value = mock_response

    res = generate_grounded_recommendation(
        narrative="Stapler jammed during procedure",
        predicted_label="M",
        chunks=mock_chunks,
        retrieval_fallback_triggered=False,
        top_score=0.72,
        client=mock_client,
    )

    assert res["llm_grounding_verified"] is False
    assert res["citations"] == ["Article 87 - Vigilance Reporting"]
    assert "Article 10 - General Obligations" not in res["citations"]


def test_fallback_sets_llm_grounding_verified_false():
    """Assert fallback when API key is missing or call throws sets llm_grounding_verified=False."""
    mock_chunks = [
        {"chunk_id": "c1", "section": "Article 87 - Vigilance Reporting", "text": "Report within 15 days.", "similarity_score": 0.72}
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("404 Model Not Found")

    res = generate_grounded_recommendation(
        narrative="Battery depleted prematurely",
        predicted_label="M",
        chunks=mock_chunks,
        retrieval_fallback_triggered=False,
        top_score=0.72,
        client=mock_client,
    )

    assert res["llm_grounding_verified"] is False
    assert "warning" in res
    assert "404 Model Not Found" in res["warning"]
    assert res["citations"] == ["Article 87 - Vigilance Reporting"]