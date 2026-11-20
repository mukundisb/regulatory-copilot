"""
The three tools the agent can call.

  search_maude         FDA MAUDE reports for a device term, via the public openFDA API
  classify_severity    the project's severity classifier (Death / Injury / Malfunction / Other)
  retrieve_regulation  EU-MDR passages from the Chroma index, steered by the severity label

Every tool returns a plain dict and never raises on an expected failure (network error, no
match, empty retrieval). A failed tool call is information the agent can act on, so it comes
back as {"ok": False, "error": ...} rather than crashing the run.

The classifier and retrieval imports are lazy: importing this module costs nothing, and the
heavy stack (fastapi, chromadb, sentence-transformers) loads only when a tool is first used.
"""

import os
import re
from typing import Any, Callable, Optional

import requests

OPENFDA_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov/device/event.json")
OPENFDA_TIMEOUT_SECONDS = 20
MAX_SAMPLE_REPORTS = 3

LABEL_NAMES = {"D": "Death", "I": "Injury", "M": "Malfunction", "O": "Other"}


def _clean_device_term(term: str) -> str:
    """
    Keep letters, digits, spaces and hyphens only. The term is placed inside a quoted
    openFDA search expression, so quotes and operators must not get through.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 \-]", " ", term or "")).strip()[:80]


# --------------------------------------------------------------------------- tool 1: MAUDE

def search_maude(
    device_term: str,
    http_get: Callable[..., Any] = requests.get,
    api_key: Optional[str] = None,
) -> dict:
    """
    Look up MAUDE reports whose generic or brand device name matches device_term.

    Returns the number of matching reports by event type and up to three recent reports.
    MAUDE is the US FDA database; it is a signal about how a device type has behaved in the
    field, not EU data and not a count of all events (reporting is voluntary for users).
    """
    term = _clean_device_term(device_term)
    if not term:
        return {"ok": False, "tool": "search_maude", "error": "empty device term"}

    key = api_key or os.getenv("OPENFDA_API_KEY")
    search = f'device.generic_name:"{term}" device.brand_name:"{term}"'  # a space means OR

    def call(extra: dict) -> Any:
        params = {"search": search, **extra}
        if key:
            params["api_key"] = key.strip().lstrip("﻿").strip("\"'")
        return http_get(OPENFDA_URL, params=params, timeout=OPENFDA_TIMEOUT_SECONDS)

    try:
        counts_response = call({"count": "event_type.exact"})
        if counts_response.status_code == 404:  # openFDA answers 404 when nothing matches
            return {"ok": True, "tool": "search_maude", "device_term": term, "found": False,
                    "total_reports": 0, "by_event_type": {}, "sample_reports": []}
        counts_response.raise_for_status()
        by_event_type = {row["term"]: row["count"] for row in counts_response.json().get("results", [])}

        sample_response = call({"limit": MAX_SAMPLE_REPORTS, "sort": "date_received:desc"})
        sample_response.raise_for_status()
        samples = []
        for report in sample_response.json().get("results", []):
            device = (report.get("device") or [{}])[0]
            samples.append({
                "report_number": report.get("report_number"),
                "date_received": report.get("date_received"),
                "event_type": report.get("event_type"),
                "brand_name": device.get("brand_name"),
                "generic_name": device.get("generic_name"),
            })
    except (requests.RequestException, ValueError, KeyError) as err:
        return {"ok": False, "tool": "search_maude", "device_term": term,
                "error": f"{type(err).__name__}: {err}"[:300]}

    return {
        "ok": True, "tool": "search_maude", "device_term": term, "found": bool(by_event_type),
        "total_reports": sum(by_event_type.values()),
        "by_event_type": by_event_type,
        "sample_reports": samples,
    }


# --------------------------------------------------------------------------- tool 2: classifier

def classify_severity(narrative: str, classify_fn: Optional[Callable[[str], dict]] = None) -> dict:
    """Predict the severity label for an event narrative with the project's classifier."""
    if not (narrative or "").strip():
        return {"ok": False, "tool": "classify_severity", "error": "empty narrative"}
    try:
        if classify_fn is None:
            from app import _classify_narrative  # same function POST /assess uses, with its TF-IDF fallback
            classify_fn = _classify_narrative
        result = classify_fn(narrative)
    except Exception as err:
        return {"ok": False, "tool": "classify_severity", "error": f"{type(err).__name__}: {err}"[:300]}

    label = result.get("predicted_label")
    return {
        "ok": True, "tool": "classify_severity",
        "label": label,
        "label_name": LABEL_NAMES.get(label, str(label)),
        "backend_used": result.get("backend_used"),
        "warning": result.get("warning"),
    }


# --------------------------------------------------------------------------- tool 3: retrieval

def retrieve_regulation(
    narrative: str,
    label: str,
    top_k: int = 3,
    retrieve_fn: Optional[Callable[..., tuple]] = None,
    build_query_fn: Optional[Callable[[str, str], str]] = None,
) -> dict:
    """
    Retrieve EU-MDR passages for the narrative. The query is steered by the severity label
    and falls back to the plain narrative when the steered query scores low, exactly as
    POST /assess does.
    """
    try:
        if retrieve_fn is None or build_query_fn is None:
            import rag_pipeline
            from app import build_retrieval_query, retrieve_with_fallback
            rag_pipeline.init_store()  # loads the embedding model and opens ./chroma_db
            retrieve_fn = retrieve_fn or retrieve_with_fallback
            build_query_fn = build_query_fn or build_retrieval_query
        steered = build_query_fn(label, narrative)
        chunks, query_used, fallback = retrieve_fn(steered, narrative, top_k)
    except Exception as err:
        return {"ok": False, "tool": "retrieve_regulation", "error": f"{type(err).__name__}: {err}"[:300]}

    return {
        "ok": True, "tool": "retrieve_regulation",
        "chunks": chunks,
        "sections": [c.get("section") for c in chunks],
        "top_score": round(chunks[0]["similarity_score"], 4) if chunks else 0.0,
        "retrieval_fallback": fallback,
        "query_used": query_used,
    }
