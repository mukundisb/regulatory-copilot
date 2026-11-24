"""
The second LLM provider (Azure OpenAI) and the provider switch. No network: the HTTP layer is a stand-in.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import rag_recommender
import telemetry
from app import app
from rag_recommender import StructuredRecommendation, generate_grounded_recommendation

SECTION = "Article 87 - Reporting of serious incidents and field safety corrective actions"
CHUNKS = [{"chunk_id": "c1", "section": SECTION, "text": "Manufacturers shall report...", "similarity_score": 0.78}]


class FakeHTTP:
    """Records the request and returns a chat-completions style answer."""

    def __init__(self, status=200, content=None, usage=None, finish_reason="stop", statuses=None, headers=None):
        self.status, self.content, self.finish_reason = status, content, finish_reason
        self.statuses = list(statuses or [])   # statuses for the first calls, then `status`
        self.headers = headers or {}
        self.usage = usage if usage is not None else {"prompt_tokens": 1800, "completion_tokens": 350}
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        status = self.statuses.pop(0) if self.statuses else self.status
        body = {"choices": [{"message": {"content": self.content}, "finish_reason": self.finish_reason}],
                "usage": self.usage}
        return SimpleNamespace(status_code=status, text="denied" if status >= 400 else "",
                               headers=self.headers if status >= 400 else {}, json=lambda: body)


def answer(recommendation="Report under Article 87.", citations=(SECTION,)):
    return json.dumps({"recommendation": recommendation, "citations": list(citations)})


@pytest.fixture
def azure_settings(monkeypatch):
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_endpoint", "https://copilot-aoai.openai.azure.com/")
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_deployment", "copilot-mini")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")


def generate(http, provider="azure_openai"):
    return generate_grounded_recommendation(
        narrative="Pacemaker lead dislodged; patient died.", predicted_label="D", chunks=CHUNKS,
        retrieval_fallback_triggered=False, top_score=0.78, client=http, provider=provider,
    )


# --------------------------------------------------------------------------- Azure OpenAI call

def test_azure_answer_is_parsed_and_grounded(azure_settings):
    http = FakeHTTP(content=answer())
    result = generate(http)
    assert result["llm_grounding_verified"] is True and result["citations"] == [SECTION]
    assert result["provider"] == "azure_openai" and result["model"] == "copilot-mini"
    assert result["usage"] == {"input_tokens": 1800, "output_tokens": 350}


def test_azure_request_shape(azure_settings):
    http = FakeHTTP(content=answer())
    generate(http)
    call = http.calls[0]
    assert call["url"] == "https://copilot-aoai.openai.azure.com/openai/v1/chat/completions"
    assert call["headers"]["api-key"] == "test-key"
    body = call["json"]
    assert body["model"] == "copilot-mini"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body  # reasoning models accept only the default
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "STRICT GROUNDING RULES" in body["messages"][1]["content"]  # same prompt as Gemini


def test_schema_matches_the_pydantic_model():
    schema = rag_recommender.RECOMMENDATION_JSON_SCHEMA["schema"]
    assert set(schema["properties"]) == set(StructuredRecommendation.model_fields)
    assert set(schema["required"]) == set(StructuredRecommendation.model_fields)


def test_azure_citations_outside_the_context_are_pruned(azure_settings):
    http = FakeHTTP(content=answer(citations=(SECTION, "Article 10 - General obligations")))
    result = generate(http)
    assert result["llm_grounding_verified"] is False
    assert result["citations"] == [SECTION]
    assert "Pruned hallucinated citations" in result["warning"]


def test_reformatted_title_of_a_retrieved_section_is_accepted(azure_settings):
    # Seen with gpt-5.4-mini: the " - " separator dropped from a retrieved section's title.
    http = FakeHTTP(content=answer(citations=(
        "Article 87 Reporting of serious incidents and field safety corrective actions",
        "article 87 - REPORTING of serious incidents and field-safety corrective actions")))
    result = generate(http)
    assert result["llm_grounding_verified"] is True
    assert result["citations"] == [SECTION]  # canonical title, once


def test_a_different_section_with_a_similar_name_is_still_rejected(azure_settings):
    http = FakeHTTP(content=answer(citations=(SECTION, "Article 88 Reporting of serious incidents")))
    result = generate(http)
    assert result["llm_grounding_verified"] is False and result["citations"] == [SECTION]


def test_azure_http_error_falls_back_to_the_template(azure_settings):
    result = generate(FakeHTTP(status=401))
    assert result["llm_grounding_verified"] is False
    assert result["warning"].startswith("Degraded to deterministic template")
    assert "Azure OpenAI HTTP 401" in result["warning"]
    assert result["provider"] == "azure_openai"


def test_azure_empty_answer_names_the_finish_reason(azure_settings):
    result = generate(FakeHTTP(content=None, finish_reason="length"))
    assert "finish_reason=length" in result["warning"]


def test_missing_azure_configuration_is_reported_by_name(monkeypatch):
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_endpoint", None)
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_deployment", "copilot-mini")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    result = generate(None)
    assert "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY unset" in result["warning"]


def test_gemini_results_are_tagged_with_provider_and_model():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=StructuredRecommendation(recommendation="r", citations=[SECTION]), usage_metadata=None)
    result = generate(mock_client, provider="gemini")
    assert result["provider"] == "gemini" and result["model"] == rag_recommender.settings.gemini_model
    assert result["llm_grounding_verified"] is True


def test_no_chunks_is_tagged_too():
    result = generate_grounded_recommendation("n", "M", [], False, 0.0, provider="azure_openai")
    assert result["provider"] == "azure_openai" and result["warning"] == "No regulatory context chunks available"


# --------------------------------------------------------------------------- rate limits

@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(rag_recommender.time, "sleep", waits.append)
    return waits


def test_rate_limit_is_retried_with_azure_retry_after(azure_settings, no_sleep):
    http = FakeHTTP(content=answer(), statuses=[429, 429], headers={"Retry-After": "7"})
    result = generate(http)
    assert result["llm_grounding_verified"] is True and len(http.calls) == 3
    assert no_sleep == [7.0, 7.0]


def test_retry_after_ms_is_preferred_and_waits_are_capped(azure_settings, no_sleep, monkeypatch):
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_max_retry_wait_seconds", 5.0)
    generate(FakeHTTP(content=answer(), statuses=[429], headers={"retry-after-ms": "1500"}))
    generate(FakeHTTP(content=answer(), statuses=[429], headers={"Retry-After": "120"}))
    assert no_sleep == [1.5, 5.0]


def test_backoff_without_a_header(azure_settings, no_sleep):
    generate(FakeHTTP(content=answer(), statuses=[503, 429]))
    assert no_sleep == [2.0, 4.0]


def test_gives_up_after_the_retry_budget(azure_settings, no_sleep, monkeypatch):
    monkeypatch.setattr(rag_recommender.settings, "azure_openai_max_retries", 2)
    http = FakeHTTP(status=429)
    result = generate(http)
    assert len(http.calls) == 3 and len(no_sleep) == 2
    assert "Azure OpenAI HTTP 429" in result["warning"]


def test_other_errors_are_not_retried(azure_settings, no_sleep):
    http = FakeHTTP(status=401)
    generate(http)
    assert len(http.calls) == 1 and no_sleep == []


# --------------------------------------------------------------------------- cost

def test_cost_uses_the_provider_prices(monkeypatch):
    monkeypatch.setattr(telemetry.settings, "azure_openai_input_price_per_mtok_usd", 0.4)
    monkeypatch.setattr(telemetry.settings, "azure_openai_output_price_per_mtok_usd", 1.6)
    assert telemetry.estimate_cost_usd("azure_openai", 1_000_000, 1_000_000) == 2.0
    assert telemetry.estimate_cost_usd("azure_openai", 2000, 500) == 0.0016


def test_cost_is_unknown_when_prices_are_not_configured(monkeypatch):
    monkeypatch.setattr(telemetry.settings, "azure_openai_input_price_per_mtok_usd", None)
    assert telemetry.estimate_cost_usd("azure_openai", 2000, 500) is None


# --------------------------------------------------------------------------- /assess and /health

@patch("app.query_store")
@patch("app._classify_narrative")
def test_assess_on_azure_reports_the_provider(mock_classify, mock_query, azure_settings, monkeypatch, capsys):
    monkeypatch.setattr(rag_recommender.settings, "llm_provider", "azure_openai")
    monkeypatch.setattr(telemetry.settings, "azure_openai_input_price_per_mtok_usd", 0.4)
    monkeypatch.setattr(telemetry.settings, "azure_openai_output_price_per_mtok_usd", 1.6)
    mock_classify.return_value = {"predicted_label": "D", "confidence": 0.9, "backend_used": "clinicalbert_vertex"}
    mock_query.return_value = CHUNKS
    http = FakeHTTP(content=answer())
    with patch("rag_recommender.requests", http):
        response = TestClient(app).post("/assess", json={"narrative": "Pacemaker lead dislodged causing cardiac arrest."})
    assert response.status_code == 200 and response.json()["llm_provider"] == "azure_openai"
    event = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith("{") and '"assess_request"' in line][0]
    assert event["outcome"] == "ok" and event["llm_provider"] == "azure_openai" and event["llm_model"] == "copilot-mini"
    assert event["est_cost_usd"] == telemetry.estimate_cost_usd("azure_openai", 1800, 350)


def test_health_names_the_llm_provider():
    assert TestClient(app).get("/health").json()["llm_provider"] == rag_recommender.settings.llm_provider


# --------------------------------------------------------------------------- RAG evaluation with a provider

def test_rag_evaluation_records_provider_timing_tokens_and_cost(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_rag.py"
    spec = importlib.util.spec_from_file_location("evaluate_rag", path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"recommendation": "r", "citations": [SECTION], "llm_grounding_verified": True,
                "usage": {"input_tokens": 1000, "output_tokens": 200}, "provider": kwargs["provider"]}

    monkeypatch.setattr(rag_recommender, "generate_grounded_recommendation", fake)
    monkeypatch.setattr(telemetry.settings, "azure_openai_input_price_per_mtok_usd", 0.4)
    monkeypatch.setattr(telemetry.settings, "azure_openai_output_price_per_mtok_usd", 1.6)
    row = ev.score_citations({"query": "q", "label": "D"}, CHUNKS, {"Article 87"}, "azure_openai")
    assert seen["provider"] == "azure_openai"
    assert row["citation_hit"] and row["llm_used"] and row["generate_ms"] >= 0
    assert row["input_tokens"] == 1000 and row["est_cost_usd"] == telemetry.estimate_cost_usd("azure_openai", 1000, 200)


def test_faithfulness_judge_retries_once_on_malformed_json(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace as NS
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_faithfulness.py"
    spec = importlib.util.spec_from_file_location("evaluate_faithfulness", path)
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    replies = ['{"claims": [{"id": 1 "label": "supported"}]}',
               '{"claims": [{"id": 1, "label": "supported", "reason": "ok"}]}']
    monkeypatch.setattr(ef, "build_prompt", lambda item, claims: "p")
    monkeypatch.setattr(ef, "call_anthropic", lambda prompt, model: replies.pop(0))
    verdicts = ef.judge_item({}, ["claim"], NS(judge="anthropic", judge_model="m"))
    assert verdicts[0]["label"] == "supported" and replies == []
