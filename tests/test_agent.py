"""
Tests for the agent: the three tools, the graph's guardrails and the human approval step.
No network, no model files, no API keys: tools, planner and recommender are replaced by stand-ins.
"""

import json
from types import SimpleNamespace

import pytest
import requests

pytest.importorskip("langgraph")

from agent import graph as agent_graph  # noqa: E402
from agent.graph import build_graph, pending_packet, resume_run, start_run  # noqa: E402
from agent.planner import Decision, GeminiPlanner, RulePlanner  # noqa: E402
from agent.tools import classify_severity, retrieve_regulation, search_maude  # noqa: E402

NARRATIVE = "ICD failed to deliver therapy during ventricular fibrillation and the patient died."
CHUNKS = [
    {"section": "Article 87 - Reporting of serious incidents", "similarity_score": 0.71, "text": "..."},
    {"section": "Article 89 - Analysis of serious incidents", "similarity_score": 0.62, "text": "..."},
]


# --------------------------------------------------------------------------- tool: MAUDE

class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code, self._payload = status_code, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def fake_openfda(calls):
    def get(url, params, timeout):
        calls.append(params)
        if "count" in params:
            return FakeResponse(200, {"results": [{"term": "Malfunction", "count": 700}, {"term": "Death", "count": 20}]})
        return FakeResponse(200, {"results": [{
            "report_number": "R-1", "date_received": "20261001", "event_type": "Death",
            "device": [{"brand_name": "ICD X", "generic_name": "defibrillator"}],
        }]})
    return get


def test_search_maude_returns_counts_and_samples():
    calls = []
    result = search_maude("implantable defibrillator", http_get=fake_openfda(calls), api_key="k")
    assert result["ok"] and result["found"]
    assert result["total_reports"] == 720
    assert result["by_event_type"] == {"Malfunction": 700, "Death": 20}
    assert result["sample_reports"][0]["report_number"] == "R-1"
    assert calls[0]["search"] == 'device.generic_name:"implantable defibrillator" device.brand_name:"implantable defibrillator"'
    assert calls[0]["api_key"] == "k"


def test_search_maude_strips_characters_that_could_alter_the_query():
    calls = []
    search_maude('pump" OR device.brand_name:"x', http_get=fake_openfda(calls))
    assert '"pump OR device brand name x"' in calls[0]["search"]
    assert calls[0]["search"].count('"') == 4  # only the two pairs we place ourselves


def test_search_maude_no_match_is_not_an_error():
    result = search_maude("nonexistent gadget", http_get=lambda *a, **k: FakeResponse(404))
    assert result["ok"] and result["found"] is False and result["total_reports"] == 0


def test_search_maude_network_failure_is_returned_not_raised():
    def boom(*a, **k):
        raise requests.ConnectionError("no route")
    result = search_maude("pump", http_get=boom)
    assert result["ok"] is False and "ConnectionError" in result["error"]


def test_search_maude_rejects_empty_term():
    assert search_maude("   !!! ")["ok"] is False


# --------------------------------------------------------------------------- tools: classifier, retrieval

def test_classify_severity_wraps_the_classifier():
    result = classify_severity(NARRATIVE, classify_fn=lambda t: {"predicted_label": "D", "backend_used": "tfidf_fallback", "warning": "w"})
    assert result["ok"] and result["label"] == "D" and result["label_name"] == "Death"
    assert result["backend_used"] == "tfidf_fallback"


def test_classify_severity_failure_and_empty_input():
    def boom(_):
        raise RuntimeError("model missing")
    assert classify_severity(NARRATIVE, classify_fn=boom)["ok"] is False
    assert classify_severity("  ", classify_fn=lambda t: {})["ok"] is False


def test_retrieve_regulation_uses_label_steering_and_reports_scores():
    seen = {}

    def build(label, narrative):
        seen["label"] = label
        return f"steered {narrative}"

    def retrieve(primary, fallback, top_k):
        seen["args"] = (primary, fallback, top_k)
        return CHUNKS, primary, False

    result = retrieve_regulation(NARRATIVE, "D", retrieve_fn=retrieve, build_query_fn=build)
    assert result["ok"] and seen["label"] == "D"
    assert seen["args"][0].startswith("steered") and seen["args"][1] == NARRATIVE
    assert result["sections"][0].startswith("Article 87") and result["top_score"] == 0.71


def test_retrieve_regulation_failure_is_returned():
    def boom(*a):
        raise RuntimeError("index missing")
    assert retrieve_regulation(NARRATIVE, "D", retrieve_fn=boom, build_query_fn=lambda l, n: n)["ok"] is False


def test_default_wiring_failures_are_returned_not_raised(monkeypatch):
    # app.py (and the model stack behind it) failing to import must not crash the agent
    import sys
    monkeypatch.setitem(sys.modules, "app", None)  # makes `from app import ...` raise ImportError
    assert classify_severity(NARRATIVE)["ok"] is False
    assert retrieve_regulation(NARRATIVE, "D")["ok"] is False


# --------------------------------------------------------------------------- graph helpers

def make_tools(log, classify_ok=True):
    def classify(state, decision):
        log.append("classify_severity")
        if not classify_ok:
            return {"ok": False, "error": "classifier down"}
        return {"ok": True, "label": "D", "label_name": "Death", "backend_used": "stub"}

    def maude(state, decision):
        log.append(("search_maude", decision.device_term))
        return {"ok": True, "found": True, "device_term": decision.device_term, "total_reports": 5,
                "by_event_type": {"Death": 5}, "sample_reports": []}

    def retrieve(state, decision):
        log.append(("retrieve_regulation", state.get("label")))
        return {"ok": True, "chunks": CHUNKS, "sections": [c["section"] for c in CHUNKS],
                "top_score": 0.71, "retrieval_fallback": False}

    return {"classify_severity": classify, "search_maude": maude, "retrieve_regulation": retrieve}


def stub_recommender(**kwargs):
    return {"recommendation": f"Report under Article 87 (label {kwargs['predicted_label']}).",
            "citations": ["Article 87 - Reporting of serious incidents"], "llm_grounding_verified": True, "warning": None}


def run_to_pause(planner=None, log=None, device=None, classify_ok=True, thread="t1"):
    log = log if log is not None else []
    graph = build_graph(planner=planner, tools=make_tools(log, classify_ok), recommender=stub_recommender)
    output = start_run(graph, NARRATIVE, thread, device_hint=device)
    return graph, output, log


# --------------------------------------------------------------------------- graph: the happy path and approval

def test_run_pauses_for_review_and_runs_tools_in_order():
    graph, output, log = run_to_pause(device="implantable defibrillator")
    packet = pending_packet(output)
    assert packet is not None, "the run must stop at the approval step"
    assert log == ["classify_severity", ("search_maude", "implantable defibrillator"), ("retrieve_regulation", "D")]
    assert packet["severity_label"] == "D" and packet["problems"] == []
    assert packet["maude"]["total_reports"] == 5
    assert packet["citations"] == ["Article 87 - Reporting of serious incidents"]
    assert [s["action"] for s in packet["steps"]] == ["classify_severity", "search_maude", "retrieve_regulation"]


def test_nothing_is_final_while_waiting_for_the_reviewer():
    graph, output, _ = run_to_pause()
    assert "status" not in output and "final" not in output


def test_approve_keeps_the_draft():
    graph, _, _ = run_to_pause()
    result = resume_run(graph, "t1", "approve", note="ok")
    assert result["status"] == "approved"
    assert result["final"]["recommendation"].startswith("Report under Article 87")
    assert result["final"]["reviewer_note"] == "ok"


def test_edit_replaces_the_recommendation_and_marks_it():
    graph, _, _ = run_to_pause()
    result = resume_run(graph, "t1", "edit", recommendation="  Report within 2 days under Article 87.  ")
    assert result["status"] == "edited"
    assert result["final"]["recommendation"] == "Report within 2 days under Article 87."
    assert result["final"]["edited_by_reviewer"] is True


def test_reject_produces_no_recommendation():
    graph, _, _ = run_to_pause()
    result = resume_run(graph, "t1", "reject", note="wrong article")
    assert result["status"] == "rejected" and result["final"] is None


@pytest.mark.parametrize("decision,text", [("edit", "   "), ("maybe", ""), ("", ""), ("APPROVE", "")])
def test_unclear_reviewer_answers_count_as_rejection(decision, text):
    graph, _, _ = run_to_pause()
    result = resume_run(graph, "t1", decision, recommendation=text)
    assert result["status"] == "rejected" and result["final"] is None


def test_rule_planner_skips_maude_without_a_device():
    _, _, log = run_to_pause()
    assert [entry if isinstance(entry, str) else entry[0] for entry in log] == ["classify_severity", "retrieve_regulation"]


# --------------------------------------------------------------------------- graph: guardrails

class Scripted:
    """A planner that returns a fixed list of decisions, then keeps drafting."""
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    def __call__(self, state):
        return self.decisions.pop(0) if self.decisions else Decision(action="draft_recommendation", reason="done")


def events_of(output):
    return [e["type"] for e in output.get("events", [])]


def test_draft_first_is_corrected_to_classify_then_retrieve():
    planner = Scripted(Decision(action="draft_recommendation", reason="skip ahead"))
    graph, output, log = run_to_pause(planner=planner)
    assert log == ["classify_severity", ("retrieve_regulation", "D")]
    assert events_of(output).count("order_enforced") == 2
    assert pending_packet(output) is not None


def test_retrieve_before_classify_is_corrected():
    planner = Scripted(Decision(action="retrieve_regulation", reason="go"))
    _, output, log = run_to_pause(planner=planner)
    assert log[0] == "classify_severity" and log[1] == ("retrieve_regulation", "D")


def test_repeated_tool_is_blocked():
    cls = Decision(action="classify_severity", reason="again")
    planner = Scripted(cls, cls, cls)
    _, output, log = run_to_pause(planner=planner)
    assert log.count("classify_severity") == 1
    assert "repeat_blocked" in events_of(output)


def test_maude_without_a_device_term_is_skipped():
    planner = Scripted(Decision(action="classify_severity", reason="x"),
                       Decision(action="search_maude", reason="x"))
    _, output, log = run_to_pause(planner=planner)
    assert not any(isinstance(e, tuple) and e[0] == "search_maude" for e in log)
    assert "maude_skipped" in events_of(output)


def test_planner_term_is_used_when_the_user_gave_none():
    planner = Scripted(Decision(action="classify_severity", reason="x"),
                       Decision(action="search_maude", device_term="icd", reason="x"))
    _, _, log = run_to_pause(planner=planner)
    assert ("search_maude", "icd") in log


def test_failing_planner_falls_back_to_rules():
    def broken(state):
        raise RuntimeError("no api key")
    graph, output, log = run_to_pause(planner=broken)
    assert "planner_failed" in events_of(output)
    assert log == ["classify_severity", ("retrieve_regulation", "D")]
    assert pending_packet(output) is not None


def test_step_budget_sends_the_run_to_the_draft(monkeypatch):
    monkeypatch.setattr(agent_graph, "MAX_PLANNER_CALLS", 1)
    graph, output, log = run_to_pause()
    assert "step_budget" in events_of(output)
    assert pending_packet(output) is not None  # still reaches the human


def test_failed_tool_does_not_stop_the_run():
    graph, output, log = run_to_pause(classify_ok=False)
    packet = pending_packet(output)
    assert packet is not None
    assert log == ["classify_severity", ("retrieve_regulation", None)]  # retrieve ran without a label
    assert packet["steps"][0]["ok"] is False and "classifier down" in packet["steps"][0]["summary"]
    assert packet["problems"] == ["classify_severity: failed: classifier down"]


def test_two_runs_do_not_share_state():
    log = []
    graph = build_graph(tools=make_tools(log), recommender=stub_recommender)
    start_run(graph, NARRATIVE, "a")
    out_b = start_run(graph, "different narrative", "b")
    assert pending_packet(out_b)["narrative"] == "different narrative"
    assert resume_run(graph, "a", "approve")["narrative"] == NARRATIVE


# --------------------------------------------------------------------------- Gemini planner

def stub_client(parsed=None, text=None):
    response = SimpleNamespace(parsed=parsed, text=text)
    models = SimpleNamespace(generate_content=lambda **kw: response)
    return SimpleNamespace(models=models)


def test_gemini_planner_returns_parsed_decision():
    decision = Decision(action="classify_severity", reason="start")
    state = {"narrative": NARRATIVE, "trace": []}
    assert GeminiPlanner(client=stub_client(parsed=decision))(state) == decision


def test_gemini_planner_reads_json_text_when_not_parsed():
    text = json.dumps({"action": "search_maude", "device_term": "pump", "reason": "named device"})
    result = GeminiPlanner(client=stub_client(text=text))({"narrative": NARRATIVE, "trace": []})
    assert result.action == "search_maude" and result.device_term == "pump"


def test_gemini_planner_rejects_unknown_actions():
    text = json.dumps({"action": "delete_everything", "reason": "x"})
    with pytest.raises(Exception):
        GeminiPlanner(client=stub_client(text=text))({"narrative": NARRATIVE, "trace": []})


# --------------------------------------------------------------------------- decision record (used by the evaluation)

def test_each_planner_call_is_recorded_with_whether_it_was_accepted():
    planner = Scripted(Decision(action="draft_recommendation", reason="skip ahead"),
                       Decision(action="retrieve_regulation", reason="ok"))
    _, output, _ = run_to_pause(planner=planner)
    first, second = output["decisions"][0], output["decisions"][1]
    assert first["proposed"]["action"] == "draft_recommendation"
    assert first["executed"] == "classify_severity" and first["accepted"] is False
    assert second["proposed"]["action"] == "retrieve_regulation" and second["accepted"] is True


def test_failed_planner_call_is_recorded_as_not_accepted():
    def broken(state):
        raise RuntimeError("quota")
    _, output, _ = run_to_pause(planner=broken)
    assert output["decisions"][0]["proposed"] is None and output["decisions"][0]["accepted"] is False


def test_device_named_by_the_user_outranks_the_planner(monkeypatch):
    from agent.graph import default_tools
    seen = []
    monkeypatch.setattr(agent_graph.agent_tools, "search_maude", lambda term: seen.append(term) or {"ok": True})
    default_tools()["search_maude"]({"device_hint": "insulin pump"}, Decision(action="search_maude", device_term="pump", reason=""))
    default_tools()["search_maude"]({}, Decision(action="search_maude", device_term="pump", reason=""))
    assert seen == ["insulin pump", "pump"]


def test_classifier_backend_is_reported_even_when_the_classifier_does_not_name_it(monkeypatch):
    import sys
    from types import SimpleNamespace
    from config import settings
    monkeypatch.setitem(sys.modules, "app", SimpleNamespace(_classify_narrative=lambda text: {"predicted_label": "M"}))
    result = classify_severity(NARRATIVE)
    assert result["ok"] and result["backend_used"] == settings.classifier_backend


# --------------------------------------------------------------------------- retrieval retry

def retry_tools(log, second_finds=True):
    """Retrieval finds nothing the first time; the retry (with a query) finds passages if second_finds."""
    tools = make_tools(log)

    def retrieve(state, decision):
        attempt = sum(1 for e in log if isinstance(e, tuple) and e[0] == "retrieve_regulation") + 1
        log.append(("retrieve_regulation", decision.query))
        found = attempt > 1 and second_finds
        chunks = CHUNKS if found else []
        return {"ok": True, "chunks": chunks, "sections": [c["section"] for c in chunks],
                "top_score": 0.71 if found else 0.0, "retrieval_fallback": True,
                "query_from_planner": decision.query}
    tools["retrieve_regulation"] = retrieve
    return tools


def run_retry(planner, second_finds=True):
    log = []
    graph = build_graph(planner=planner, tools=retry_tools(log, second_finds), recommender=stub_recommender)
    return start_run(graph, NARRATIVE, "r1"), log


CLASSIFY = Decision(action="classify_severity", reason=".")
RETRIEVE = Decision(action="retrieve_regulation", reason=".")


def test_planner_retry_with_its_own_query_after_empty_retrieval():
    output, log = run_retry(Scripted(CLASSIFY, RETRIEVE, Decision(action="retrieve_regulation", query="serious incident reporting", reason="empty")))
    assert log[1:] == [("retrieve_regulation", None), ("retrieve_regulation", "serious incident reporting")]
    assert output["retrieval_log"] == [{"query_from_planner": None, "passages": 0},
                                       {"query_from_planner": "serious incident reporting", "passages": 2}]
    assert pending_packet(output)["retrieved_sections"][0].startswith("Article 87")
    assert events_of(output) == []
    assert "retry with query 'serious incident reporting'" in output["trace"][-1]["summary"]


def test_retry_without_a_query_is_refused():
    output, log = run_retry(Scripted(CLASSIFY, RETRIEVE, RETRIEVE))
    assert sum(1 for e in log if isinstance(e, tuple) and e[0] == "retrieve_regulation") == 1
    assert "retry_needs_query" in events_of(output)
    assert pending_packet(output) is not None  # drafts with no passages; the reviewer sees it


def test_only_one_retry():
    retry = Decision(action="retrieve_regulation", query="q", reason=".")
    output, log = run_retry(Scripted(CLASSIFY, RETRIEVE, retry, retry), second_finds=False)
    assert sum(1 for e in log if isinstance(e, tuple) and e[0] == "retrieve_regulation") == 2
    assert "repeat_blocked" in events_of(output)


def test_no_retry_when_the_first_retrieval_found_passages():
    log = []
    graph = build_graph(planner=Scripted(CLASSIFY, RETRIEVE, Decision(action="retrieve_regulation", query="q", reason=".")),
                        tools=make_tools(log), recommender=stub_recommender)
    output = start_run(graph, NARRATIVE, "r2")
    assert sum(1 for e in log if isinstance(e, tuple) and e[0] == "retrieve_regulation") == 1
    assert "repeat_blocked" in events_of(output)


def test_rule_planner_retries_with_the_fixed_query_for_the_label():
    from agent.planner import RETRY_QUERIES
    output, log = run_retry(RulePlanner())
    assert log[-1] == ("retrieve_regulation", RETRY_QUERIES["D"])
    assert pending_packet(output)["retrieved_sections"]


def test_retrieve_tool_tries_the_planner_query_first():
    seen = {}

    def retrieve(primary, fallback, top_k):
        seen["args"] = (primary, fallback)
        return CHUNKS, primary, False

    result = retrieve_regulation(NARRATIVE, "D", query="serious incident reporting",
                                 retrieve_fn=retrieve, build_query_fn=lambda l, n: f"steered {n}")
    assert seen["args"] == ("serious incident reporting", f"steered {NARRATIVE}")
    assert result["query_from_planner"] == "serious incident reporting"
