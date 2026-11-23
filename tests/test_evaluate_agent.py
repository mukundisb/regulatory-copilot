"""
Tests for scripts/evaluate_agent.py scoring and for the scenario file. Stand-in tools, no network.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from agent.graph import build_graph  # noqa: E402
from agent.planner import Decision  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("evaluate_agent", REPO_ROOT / "scripts" / "evaluate_agent.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

SCENARIO = {
    "id": "X01", "category": "device_in_text", "narrative": "Insulin pump alarm failed; patient hospitalised.",
    "device_hint": None, "expected_label": "I", "expect_maude": True,
    "device_keywords": ["insulin pump"], "expected_sections": ["Article 87", "Article 89"],
}
CHUNKS = [{"section": "Article 87 - Reporting of serious incidents", "similarity_score": 0.7, "text": "..."}]


def stub_tools(label="I"):
    return {
        "classify_severity": lambda s, d: {"ok": True, "label": label, "label_name": "x", "backend_used": "stub"},
        "search_maude": lambda s, d: {"ok": True, "found": True, "device_term": d.device_term or s.get("device_hint"),
                                      "total_reports": 3, "by_event_type": {}, "sample_reports": []},
        "retrieve_regulation": lambda s, d: {"ok": True, "chunks": CHUNKS, "sections": [c["section"] for c in CHUNKS],
                                             "top_score": 0.7, "retrieval_fallback": False},
    }


def recommender(cites=("Article 87 - Reporting of serious incidents",), grounded=True):
    return lambda **kw: {"recommendation": "r", "citations": list(cites), "llm_grounding_verified": grounded, "warning": None}


class Scripted:
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    def __call__(self, state):
        return self.decisions.pop(0) if self.decisions else Decision(action="draft_recommendation", reason="done")


GOOD_PLAN = (Decision(action="classify_severity", reason="."),
             Decision(action="search_maude", device_term="insulin pump", reason="."),
             Decision(action="retrieve_regulation", reason="."))


def evaluate(planner, scenario=SCENARIO, label="I", rec=None):
    graph = build_graph(planner=planner, tools=stub_tools(label), recommender=rec or recommender())
    return ev.score_scenario(scenario, *ev.run_scenario(graph, scenario))


def test_a_clean_run_scores_correct_on_everything():
    row = evaluate(Scripted(*GOOD_PLAN))
    assert row["tool_selection"] and row["task_success"] and row["fully_correct"]
    assert row["planner_calls"] == 4 and row["decisions_accepted"] == 4
    assert row["tools_run"] == ["classify_severity", "search_maude", "retrieve_regulation"]
    assert row["classifier_backend"] == "stub" and row["maude_term_ok"] is True


def test_guardrail_intervention_fails_tool_selection_but_not_the_task():
    row = evaluate(Scripted(Decision(action="draft_recommendation", reason="rush"), *GOOD_PLAN[1:]))
    assert row["guardrail_events"] == ["order_enforced"]
    assert row["tool_selection"] is False and row["task_success"] is True and row["fully_correct"] is False


def test_missing_maude_lookup_fails_tool_selection():
    row = evaluate(Scripted(GOOD_PLAN[0], GOOD_PLAN[2]))
    assert row["maude_called"] is False and row["tool_selection"] is False


def test_unneeded_maude_lookup_fails_tool_selection():
    scenario = {**SCENARIO, "expect_maude": False, "device_keywords": []}
    row = evaluate(Scripted(*GOOD_PLAN), scenario=scenario)
    assert row["maude_called"] and row["tool_selection"] is False


def test_wrong_device_term_fails_tool_selection():
    plan = (GOOD_PLAN[0], Decision(action="search_maude", device_term="all devices", reason="."), GOOD_PLAN[2])
    row = evaluate(Scripted(*plan))
    assert row["maude_term_ok"] is False and row["tool_selection"] is False


@pytest.mark.parametrize("label,rec,failed", [
    ("D", None, "label_correct"),
    ("I", recommender(cites=("Article 2 - Definitions",)), "citation_hit"),
    ("I", recommender(grounded=False), "grounded"),
])
def test_each_outcome_check_can_fail_the_task(label, rec, failed):
    row = evaluate(Scripted(*GOOD_PLAN), label=label, rec=rec)
    assert row[failed] is False and row["task_success"] is False


def test_summary_arithmetic():
    rows = [
        evaluate(Scripted(*GOOD_PLAN)),                                       # all correct
        evaluate(Scripted(GOOD_PLAN[0], GOOD_PLAN[2])),                       # missed MAUDE
        evaluate(Scripted(*GOOD_PLAN), label="D"),                            # wrong label
        evaluate(Scripted(*GOOD_PLAN), scenario={**SCENARIO, "expect_maude": False, "device_keywords": []}),  # extra lookup
    ]
    s = ev.summarise(rows)
    assert s["scenarios"] == 4
    assert s["tool_selection"] == 0.5          # rows 1 and 3
    assert s["task_success"] == 0.75           # all but row 3
    assert s["fully_correct"] == 0.25
    assert s["maude_recall"] == round(2 / 3, 3)    # expected in rows 1-3, called in 1 and 3
    assert s["maude_precision"] == round(2 / 3, 3)  # called in rows 1, 3, 4; expected in 1 and 3
    assert s["device_term_accuracy"] == 1.0
    assert s["decision_accuracy"] == 1.0       # every pick was valid; the misses are about what was not picked


def test_scenario_file_is_well_formed():
    data = json.loads((REPO_ROOT / "eval" / "agent_scenarios.json").read_text(encoding="utf-8"))
    items = data["items"]
    assert len(items) == 30 and len({i["id"] for i in items}) == 30
    for item in items:
        assert item["expected_label"] in {"D", "I", "M", "O"}
        assert item["expected_sections"] and item["narrative"].strip()
        assert bool(item["device_keywords"]) == item["expect_maude"], item["id"]
        if item["device_hint"]:
            assert item["expect_maude"], item["id"]


def test_retry_is_scored():
    def tools_empty_then_found():
        tools = stub_tools()
        calls = []

        def retrieve(s, d):
            calls.append(d.query)
            found = len(calls) > 1
            chunks = CHUNKS if found else []
            return {"ok": True, "chunks": chunks, "sections": [c["section"] for c in chunks],
                    "top_score": 0.7 if found else 0.0, "retrieval_fallback": True, "query_from_planner": d.query}
        tools["retrieve_regulation"] = retrieve
        return tools

    plan = (*GOOD_PLAN, Decision(action="retrieve_regulation", query="serious incident reporting", reason="."))
    graph = build_graph(planner=Scripted(*plan), tools=tools_empty_then_found(), recommender=recommender())
    row = ev.score_scenario(SCENARIO, *ev.run_scenario(graph, SCENARIO))
    assert row["first_retrieval_empty"] and row["retried"] and row["retry_found_passages"]
    assert row["retry_query"] == "serious incident reporting" and row["task_success"]
    s = ev.summarise([row, evaluate(Scripted(*GOOD_PLAN))])
    assert s["first_retrieval_empty"] == 1 and s["retry_rate"] == 1.0 and s["retry_rescued"] == 1.0
