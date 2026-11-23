"""
Request telemetry for the /assess endpoint.

Each request writes one JSON object on a single line to stdout. Cloud Run forwards stdout to
Cloud Logging, which parses a JSON line into `jsonPayload`, so every field below can be used
in a log-based metric (latency, cost per request, degraded/error counts).
"""

import json
from typing import Any, Optional

from config import settings


def _prices(provider: str) -> tuple[Optional[float], Optional[float]]:
    if provider == "azure_openai":
        return settings.azure_openai_input_price_per_mtok_usd, settings.azure_openai_output_price_per_mtok_usd
    return settings.gemini_input_price_per_mtok_usd, settings.gemini_output_price_per_mtok_usd


def estimate_cost_usd(provider: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Cost of one LLM call from its token counts. None when the counts or the provider's prices are unknown."""
    input_price, output_price = _prices(provider)
    if None in (input_tokens, output_tokens, input_price, output_price):
        return None
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 8)


def estimate_gemini_cost_usd(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Cost of one Gemini call from its token counts. None when the counts are unknown."""
    return estimate_cost_usd("gemini", input_tokens, output_tokens)


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
