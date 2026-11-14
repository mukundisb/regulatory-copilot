import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import app
from rag_recommender import StructuredRecommendation, _extract_usage
from telemetry import estimate_gemini_cost_usd

client = TestClient(app)


def test_cost_estimate_uses_configured_prices():
    # 1,000,000 input tokens and 1,000,000 output tokens at the configured prices
    with patch("telemetry.settings") as mock_settings:
        mock_settings.gemini_input_price_per_mtok_usd = 0.75
        mock_settings.gemini_output_price_per_mtok_usd = 3.75
        assert estimate_gemini_cost_usd(1_000_000, 1_000_000) == 4.5
        assert estimate_gemini_cost_usd(2000, 400) == 0.003


def test_cost_estimate_is_none_without_token_counts():
    assert estimate_gemini_cost_usd(None, 100) is None
    assert estimate_gemini_cost_usd(100, None) is None


def test_extract_usage_adds_thinking_tokens_to_output():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=1200, candidates_token_count=150, thoughts_token_count=50)
    )
    assert _extract_usage(response) == {"input_tokens": 1200, "output_tokens": 200}


def test_extract_usage_tolerates_missing_metadata():
    assert _extract_usage(SimpleNamespace()) == {"input_tokens": None, "output_tokens": None}
    assert _extract_usage(MagicMock()) == {"input_tokens": None, "output_tokens": None}


def _telemetry_lines(captured_stdout: str) -> list[dict]:
    events = []
    for line in captured_stdout.splitlines():
        if line.startswith("{") and '"assess_request"' in line:
            events.append(json.loads(line))
    return events


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-mock-key"})
@patch("rag_recommender.genai.Client")
@patch("app.query_store")
@patch("app._classify_narrative")
def test_assess_emits_one_ok_telemetry_event(mock_classify, mock_query, mock_genai_client_cls, capsys):
    section = "Article 87 - Reporting of serious incidents and field safety corrective actions"
    mock_classify.return_value = {"predicted_label": "D", "confidence": 0.96, "backend_used": "clinicalbert_vertex"}
    mock_query.return_value = [
        {"chunk_id": "c1", "section": section, "text": "Manufacturers shall report...", "similarity_score": 0.78}
    ]
    mock_response = MagicMock()
    mock_response.parsed = StructuredRecommendation(recommendation="Report under Article 87.", citations=[section])
    mock_response.usage_metadata = SimpleNamespace(
        prompt_token_count=2000, candidates_token_count=300, thoughts_token_count=100
    )
    mock_genai_client_cls.return_value.models.generate_content.return_value = mock_response

    response = client.post("/assess", json={"narrative": "Pacemaker lead dislodged causing cardiac arrest."})
    assert response.status_code == 200

    events = _telemetry_lines(capsys.readouterr().out)
    assert len(events) == 1
    event = events[0]
    assert event["outcome"] == "ok"
    assert event["severity"] == "INFO"
    assert event["classifier_backend"] == "clinicalbert_vertex"
    assert event["classifier_fallback"] is False
    assert event["llm_fallback"] is False
    assert event["input_tokens"] == 2000
    assert event["output_tokens"] == 400
    assert event["est_cost_usd"] is not None and event["est_cost_usd"] > 0
    for field in ("latency_ms", "classify_ms", "retrieve_ms", "generate_ms"):
        assert event[field] >= 0
    # The stages are timed inside the request, so they cannot add up to more than the total.
    assert event["classify_ms"] + event["retrieve_ms"] + event["generate_ms"] <= event["latency_ms"] + 1


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-mock-key"})
@patch("rag_recommender.genai.Client")
@patch("app.query_store")
@patch("app._classify_narrative")
def test_assess_marks_fallbacks_as_degraded(mock_classify, mock_query, mock_genai_client_cls, capsys):
    mock_classify.return_value = {
        "predicted_label": "M", "confidence": 0.7, "backend_used": "tfidf_fallback",
        "warning": "Degraded to TF-IDF fallback: Vertex AI cold start timeout exceeded",
    }
    mock_query.return_value = [
        {"chunk_id": "c1", "section": "Article 88 - Trend reporting", "text": "...", "similarity_score": 0.6}
    ]
    mock_genai_client_cls.return_value.models.generate_content.side_effect = RuntimeError("503 unavailable")

    response = client.post("/assess", json={"narrative": "Touchscreen froze during pre-use check."})
    assert response.status_code == 200

    event = _telemetry_lines(capsys.readouterr().out)[0]
    assert event["outcome"] == "degraded"
    assert event["severity"] == "WARNING"
    assert event["classifier_fallback"] is True
    assert event["llm_fallback"] is True
    assert event["est_cost_usd"] is None


@patch("app._classify_narrative")
def test_assess_logs_error_event_when_request_fails(mock_classify, capsys):
    mock_classify.side_effect = RuntimeError("standby model missing")
    failing_client = TestClient(app, raise_server_exceptions=False)

    response = failing_client.post("/assess", json={"narrative": "Any narrative."})
    assert response.status_code == 500

    event = _telemetry_lines(capsys.readouterr().out)[0]
    assert event["outcome"] == "error"
    assert event["severity"] == "ERROR"
    assert event["error_type"] == "RuntimeError"
