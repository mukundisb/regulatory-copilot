"""
Request telemetry for the /assess endpoint.

Each request writes one JSON object on a single line to stdout. Cloud Run forwards stdout to
Cloud Logging, which parses a JSON line into `jsonPayload`, so every field below can be used
in a log-based metric (latency, cost per request, degraded/error counts).
"""

import json
from typing import Any, Optional

from config import settings


def estimate_gemini_cost_usd(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Cost of one Gemini call from its token counts. None when the counts are unknown."""
    if input_tokens is None or output_tokens is None:
        return None
    cost = (
        input_tokens * settings.gemini_input_price_per_mtok_usd
        + output_tokens * settings.gemini_output_price_per_mtok_usd
    ) / 1_000_000
    return round(cost, 8)


def emit_assess_event(outcome: str, **fields: Any) -> None:
    """
    outcome: "ok"       - classifier and LLM both answered normally
             "degraded" - the request was served, but through a fallback path
             "error"    - the request failed with an unhandled exception
    """
    severity = {"ok": "INFO", "degraded": "WARNING", "error": "ERROR"}.get(outcome, "INFO")
    payload = {
        "severity": severity,
        "message": f"assess_request outcome={outcome}",
        "event": "assess_request",
        "outcome": outcome,
        **fields,
    }
    print(json.dumps(payload, default=str), flush=True)
