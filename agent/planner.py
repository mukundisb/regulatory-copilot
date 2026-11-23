"""
Planners: the part of the agent that decides which tool to call next.

A planner is any callable  planner(state) -> Decision.  Two are provided:

  GeminiPlanner  asks Gemini to choose the next action from the tool descriptions and what has
                 happened so far (structured JSON output, temperature 0).
  RulePlanner    a fixed order with no LLM: classify, MAUDE (when a device term is known),
                 retrieve, draft. Used offline, in tests, and as the fallback when Gemini fails.

The graph does not trust either one: it enforces ordering and a step budget (see graph.py).
"""

import json
import logging
import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("regulatory_copilot.agent")

Action = Literal["classify_severity", "search_maude", "retrieve_regulation", "draft_recommendation"]

TOOL_DESCRIPTIONS = """\
classify_severity     Predict the event severity (D death, I injury, M malfunction, O other) from the narrative.
                      Needed before retrieve_regulation, which uses the label to steer the search.
search_maude          Count FDA MAUDE reports for a device term (needs device_term: the generic device name,
                      e.g. "implantable defibrillator"). Optional; use it when the narrative names a device type.
retrieve_regulation   Fetch the EU-MDR passages that apply to this event. Required before drafting.
                      If it returned 0 passages, call it ONCE more with `query`: a short search written in the
                      regulation's own wording for this kind of event (e.g. "reporting of serious incidents to the
                      competent authority"), not the narrative. Never more than two retrievals.
draft_recommendation  Write the grounded recommendation. Call it once classify_severity and retrieve_regulation are done."""


class Decision(BaseModel):
    action: Action
    device_term: Optional[str] = Field(default=None, description="Only for search_maude: generic device name")
    query: Optional[str] = Field(default=None, description="Only for a retry of retrieve_regulation after 0 passages")
    reason: str = Field(description="One sentence: why this is the next step")


def state_summary(state: dict) -> str:
    """What the planner is shown: narrative plus a compact record of what has happened."""
    lines = [f"Narrative: {state['narrative']}"]
    if state.get("device_hint"):
        lines.append(f"Device named by the user: {state['device_hint']}")
    if not state.get("trace"):
        lines.append("Nothing has been done yet.")
    for step in state.get("trace", []):
        lines.append(f"- {step['action']}: {step['summary']}")
    return "\n".join(lines)


# The rule planner's retry: one fixed query per severity label.
RETRY_QUERIES = {
    "D": "reporting of serious incidents to the competent authority by the manufacturer",
    "I": "reporting of serious incidents to the competent authority by the manufacturer",
    "M": "trend reporting of incidents that are not serious incidents",
    "O": "manufacturer obligations for labelling and instructions for use",
}


class RulePlanner:
    """Deterministic order. No network, no API key."""

    def __call__(self, state: dict) -> Decision:
        done = {s["action"] for s in state.get("trace", [])}
        retrievals = sum(1 for s in state.get("trace", []) if s["action"] == "retrieve_regulation")
        if retrievals == 1 and not state.get("chunks"):
            return Decision(action="retrieve_regulation", query=RETRY_QUERIES.get(state.get("label") or "O", RETRY_QUERIES["O"]),
                            reason="The first retrieval found nothing; retry with a fixed query for this severity.")
        if "classify_severity" not in done:
            return Decision(action="classify_severity", reason="Severity decides how retrieval is steered.")
        if "search_maude" not in done and state.get("device_hint"):
            return Decision(action="search_maude", device_term=state["device_hint"],
                            reason="A device was named, so check MAUDE for similar reports.")
        if "retrieve_regulation" not in done:
            return Decision(action="retrieve_regulation", reason="Find the EU-MDR passages that apply.")
        return Decision(action="draft_recommendation", reason="Everything needed is gathered.")


class GeminiPlanner:
    def __init__(self, model: str = "gemini-3.8-flash", client: Any = None):
        self.model = model
        self._client = client

    def __call__(self, state: dict) -> Decision:
        from google import genai
        from google.genai import types

        client = self._client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = (
            "You are the planner of a regulatory assistant for medical-device incident reports.\n"
            "Pick the single next action. Tools:\n"
            f"{TOOL_DESCRIPTIONS}\n\n"
            f"{state_summary(state)}\n\n"
            "Do not repeat an action that is already in the list above, except the one retry of "
            "retrieve_regulation described in the tools."
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Decision,
                temperature=0.0,
            ),
        )
        parsed = getattr(response, "parsed", None)
        return parsed if isinstance(parsed, Decision) else Decision(**json.loads(response.text))
