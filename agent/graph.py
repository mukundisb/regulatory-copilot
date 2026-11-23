"""
The agent, as a LangGraph state graph.

    START -> plan -> act -> plan -> ...        (loop while the planner picks a tool)
    plan -> draft -> approval -> finalize -> END   (when the planner picks the draft)

plan      the planner picks the next action. The graph then checks the pick (guardrails below).
act       runs the chosen tool and records what came back.
draft     writes the recommendation from the retrieved EU-MDR passages (the same grounded,
          citation-checked generator POST /assess uses).
approval  pauses the run and hands a review packet to a person. Nothing leaves the graph as a
          recommendation until the person approves or edits it.
finalize  records the outcome. A rejected draft produces no recommendation at all.

Guardrails applied to whatever the planner returns, so a poor plan cannot skip a required step:
  - a tool is never run twice, with one exception: when retrieve_regulation returned no passages,
    the planner may call it once more with a query it writes itself (a retry without a query is refused);
  - retrieve_regulation needs a severity label first, so classify_severity is run before it;
  - draft_recommendation needs both classify_severity and retrieve_regulation to have run;
  - search_maude without a device term is skipped;
  - at most MAX_PLANNER_CALLS planner calls per run, after which the graph goes straight to the draft.
If the planner itself fails (no API key, network, bad JSON) the deterministic RulePlanner takes over.
"""

import logging
from typing import Any, Callable, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent import tools as agent_tools
from agent.planner import Decision, RulePlanner

logger = logging.getLogger("regulatory_copilot.agent")

MAX_PLANNER_CALLS = 8
REQUIRED_BEFORE_DRAFT = ("classify_severity", "retrieve_regulation")


class AgentState(TypedDict, total=False):
    narrative: str
    device_hint: Optional[str]
    trace: list            # one entry per tool run: {action, ok, summary}
    events: list           # guardrail interventions and planner failures
    decisions: list        # each planner call: what it proposed, what ran, and whether the graph changed it
    planner_calls: int
    next_action: str
    next_device_term: Optional[str]
    next_query: Optional[str]
    retrieval_log: list    # one entry per retrieval: {query_from_planner, passages}
    label: Optional[str]
    classifier_backend: Optional[str]
    maude: Optional[dict]
    chunks: list
    retrieval_fallback: bool
    top_score: float
    draft: Optional[dict]
    review: Optional[dict]
    status: str            # approved | edited | rejected
    final: Optional[dict]


# --------------------------------------------------------------------------- default tool wiring

def default_tools() -> dict[str, Callable[[AgentState, Decision], dict]]:
    return {
        "classify_severity": lambda s, d: agent_tools.classify_severity(s["narrative"]),
        # A device named by the user outranks the planner's reading of the narrative.
        "search_maude": lambda s, d: agent_tools.search_maude(s.get("device_hint") or d.device_term or ""),
        "retrieve_regulation": lambda s, d: agent_tools.retrieve_regulation(s["narrative"], s.get("label") or "O", query=d.query),
    }


def _summarise(action: str, result: dict) -> str:
    if not result.get("ok"):
        return f"failed: {result.get('error')}"
    if action == "classify_severity":
        return f"{result['label']} ({result['label_name']}) via {result.get('backend_used') or 'classifier'}"
    if action == "search_maude":
        if not result.get("found"):
            return f"no MAUDE reports match '{result['device_term']}'"
        return f"{result['total_reports']} reports for '{result['device_term']}': {result['by_event_type']}"
    if action == "retrieve_regulation":
        prefix = f"retry with query '{result['query_from_planner']}': " if result.get("query_from_planner") else ""
        return f"{prefix}{len(result['chunks'])} passages {result['sections']}, top score {result['top_score']}"
    return "done"


# --------------------------------------------------------------------------- graph

def build_graph(
    planner: Optional[Callable[[dict], Decision]] = None,
    tools: Optional[dict] = None,
    recommender: Optional[Callable[..., dict]] = None,
    checkpointer: Any = None,
):
    planner = planner or RulePlanner()
    tools = tools or default_tools()
    if recommender is None:
        from rag_recommender import generate_grounded_recommendation as recommender  # lazy: pulls in genai

    fallback_planner = RulePlanner()

    def plan(state: AgentState) -> dict:
        events = list(state.get("events", []))
        calls = state.get("planner_calls", 0) + 1
        done = {s["action"] for s in state.get("trace", [])}

        if calls > MAX_PLANNER_CALLS:
            events.append({"type": "step_budget", "detail": f"more than {MAX_PLANNER_CALLS} planner calls; drafting now"})
            return {"next_action": "draft_recommendation", "planner_calls": calls, "events": events}

        events_before = len(events)
        proposed = None
        try:
            decision = planner(state)
            proposed = {"action": decision.action, "device_term": decision.device_term, "reason": decision.reason}
        except Exception as err:
            events.append({"type": "planner_failed", "detail": f"{type(err).__name__}: {err}"[:200] + "; using the rule planner"})
            decision = fallback_planner(state)

        action, device_term, query = decision.action, decision.device_term, (decision.query or "").strip() or None
        retrievals = sum(1 for s in state.get("trace", []) if s["action"] == "retrieve_regulation")
        retry_allowed = retrievals == 1 and not state.get("chunks")

        if action == "retrieve_regulation" and retry_allowed:
            if not query:
                events.append({"type": "retry_needs_query", "detail": "a second retrieval needs a query written by the planner"})
                action = "draft_recommendation"
        elif action in done and action != "draft_recommendation":
            events.append({"type": "repeat_blocked", "detail": f"planner asked for {action} again"})
            action = "draft_recommendation"  # re-resolved by the checks below

        if action == "search_maude" and not (device_term or state.get("device_hint")):
            events.append({"type": "maude_skipped", "detail": "no device term available"})
            done.add("search_maude")
            action = "draft_recommendation"

        if action in ("retrieve_regulation", "draft_recommendation") and "classify_severity" not in done:
            events.append({"type": "order_enforced", "detail": f"{action} needs a severity label; classifying first"})
            action = "classify_severity"
        elif action == "draft_recommendation" and "retrieve_regulation" not in done:
            events.append({"type": "order_enforced", "detail": "draft needs retrieved passages; retrieving first"})
            action = "retrieve_regulation"

        if proposed is not None:
            proposed["query"] = decision.query
        decisions = list(state.get("decisions", [])) + [{
            "proposed": proposed,                        # None when the planner failed
            "executed": action,
            "accepted": len(events) == events_before,   # False when a guardrail changed or replaced the pick
        }]
        return {"next_action": action, "next_device_term": device_term,
                "next_query": query if action == "retrieve_regulation" else None,
                "planner_calls": calls, "events": events, "decisions": decisions}

    def route_after_plan(state: AgentState) -> str:
        return "draft" if state["next_action"] == "draft_recommendation" else "act"

    def act(state: AgentState) -> dict:
        action = state["next_action"]
        decision = Decision(action=action, device_term=state.get("next_device_term"),
                            query=state.get("next_query"), reason="")
        result = tools[action](state, decision)
        trace = list(state.get("trace", [])) + [{"action": action, "ok": result.get("ok", False), "summary": _summarise(action, result)}]

        update: dict = {"trace": trace}
        if action == "classify_severity" and result.get("ok"):
            update["label"] = result["label"]
            update["classifier_backend"] = result.get("backend_used")
        elif action == "search_maude":
            update["maude"] = result
        elif action == "retrieve_regulation":
            update["retrieval_log"] = list(state.get("retrieval_log", [])) + [{
                "query_from_planner": result.get("query_from_planner"),
                "passages": len(result.get("chunks", [])) if result.get("ok") else None,
            }]
            if result.get("ok"):
                update.update(chunks=result["chunks"], top_score=result["top_score"],
                              retrieval_fallback=result["retrieval_fallback"])
        return update

    def draft(state: AgentState) -> dict:
        chunks = state.get("chunks", [])
        result = recommender(
            narrative=state["narrative"],
            predicted_label=state.get("label") or "O",
            chunks=chunks,
            retrieval_fallback_triggered=state.get("retrieval_fallback", False),
            top_score=state.get("top_score", 0.0),
        )
        return {"draft": result}

    def approval(state: AgentState) -> dict:
        # The node restarts from the top when the run resumes, so nothing before interrupt() may have side effects.
        draft_result = state["draft"] or {}
        packet = {
            "narrative": state["narrative"],
            "severity_label": state.get("label"),
            "maude": state.get("maude"),
            "retrieved_sections": [c.get("section") for c in state.get("chunks", [])],
            "recommendation": draft_result.get("recommendation"),
            "citations": draft_result.get("citations", []),
            "grounding_verified": draft_result.get("llm_grounding_verified", False),
            "warning": draft_result.get("warning"),
            "steps": state.get("trace", []),
            "problems": [f"{t['action']}: {t['summary']}" for t in state.get("trace", []) if not t.get("ok")],
        }
        reply = interrupt(packet)
        return {"review": reply if isinstance(reply, dict) else {}}

    def finalize(state: AgentState) -> dict:
        review = state.get("review") or {}
        decision = review.get("decision")
        draft_result = state.get("draft") or {}
        if decision == "approve":
            return {"status": "approved", "final": {**draft_result, "reviewer_note": review.get("note")}}
        if decision == "edit" and (review.get("recommendation") or "").strip():
            return {"status": "edited", "final": {
                **draft_result,
                "recommendation": review["recommendation"].strip(),
                "edited_by_reviewer": True,
                "reviewer_note": review.get("note"),
            }}
        # Anything else, including a missing or unrecognised reply, is a rejection.
        return {"status": "rejected", "final": None}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("draft", draft)
    graph.add_node("approval", approval)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, {"act": "act", "draft": "draft"})
    graph.add_edge("act", "plan")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


# --------------------------------------------------------------------------- run helpers

def start_run(graph, narrative: str, thread_id: str, device_hint: Optional[str] = None) -> dict:
    """Run until the approval pause (or the end). Returns the graph output; look for '__interrupt__'."""
    initial = {"narrative": narrative, "device_hint": device_hint, "trace": [], "events": [], "decisions": [],
               "retrieval_log": [], "planner_calls": 0}
    return graph.invoke(initial, {"configurable": {"thread_id": thread_id}})


def resume_run(graph, thread_id: str, decision: str, note: str = "", recommendation: str = "") -> dict:
    """Give the reviewer's answer to a paused run and finish it."""
    reply = {"decision": decision, "note": note, "recommendation": recommendation}
    return graph.invoke(Command(resume=reply), {"configurable": {"thread_id": thread_id}})


def pending_packet(output: dict) -> Optional[dict]:
    """The review packet of a paused run, or None when the run has finished."""
    interrupts = output.get("__interrupt__")
    return interrupts[0].value if interrupts else None
