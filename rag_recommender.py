import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

import requests

from config import settings

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


logger = logging.getLogger("regulatory_copilot.llm")

SYSTEM_INSTRUCTION = (
    "You are an EU-MDR regulatory compliance engine. Return only strictly grounded JSON matching the schema."
)

# JSON schema for Azure OpenAI structured output. Strict mode needs every property required
# and additionalProperties false; it mirrors StructuredRecommendation.
RECOMMENDATION_JSON_SCHEMA = {
    "name": "structured_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "recommendation": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["recommendation", "citations"],
        "additionalProperties": False,
    },
}


class StructuredRecommendation(BaseModel):
    recommendation: str = Field(
        description="Actionable compliance guidance strictly grounded in the provided EU-MDR statutory articles."
    )
    citations: List[str] = Field(
        description="Exact section titles cited from the provided context chunks. Must be a subset of available sections."
    )


def build_recommendation_prompt(
    narrative: str,
    predicted_label: str,
    chunks: List[Dict[str, Any]],
    retrieval_fallback_triggered: bool,
    top_score: float,
    threshold: float = 0.55,
) -> str:
    sections_text = "\n\n".join(
        [
            f"--- Section: {c.get('section', 'Unknown Section')} "
            f"(Chunk ID: {c.get('chunk_id', 'unknown')}, Score: {c.get('similarity_score', 0.0):.4f}) ---\n"
            f"{c.get('text', '')}"
            for c in chunks
        ]
    )

    confidence_advisory = (
        "CONFIDENCE LEVEL: HIGH. The retrieved EU-MDR sections directly match the incident context."
        if top_score >= threshold
        else (
            "CONFIDENCE LEVEL: LOW / AMBIGUOUS. Vector similarity is below the operational threshold (0.55). "
            "You MUST hedge your recommendation explicitly, noting that retrieved statutory provisions "
            "are weak candidates requiring manual verification by a regulatory affairs specialist."
        )
    )

    return f"""You are a certified EU-MDR Regulatory Compliance Officer.
Draft a concise, deterministic recommendation for a medical device event narrative.

CLASSIFICATION CONTEXT:
- Predicted Event Class: {predicted_label}
- Device Narrative: {narrative}

RETRIEVAL STATUS:
- {confidence_advisory}
- Fallback Query Executed: {retrieval_fallback_triggered}

RETRIEVED EU-MDR STATUTORY CONTEXT:
{sections_text}

STRICT GROUNDING RULES:
1. Base all statements ONLY on the provided EU-MDR excerpt text above.
2. In 'citations', you MUST ONLY list exact section titles that match the excerpt headers above.
3. Do NOT invent, cite external articles, or cite general knowledge. If none of the provided excerpts apply, output citations: [].
"""


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    """
    Token counts for one Gemini call, read from response.usage_metadata.
    Thinking tokens are billed as output, so they are added to the output count.
    Returns None values when the SDK response carries no usable counts.
    """
    meta = getattr(response, "usage_metadata", None)

    def count(name: str) -> Optional[int]:
        value = getattr(meta, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    input_tokens = count("prompt_token_count")
    output_tokens = count("candidates_token_count")
    thinking_tokens = count("thoughts_token_count")
    if output_tokens is not None and thinking_tokens:
        output_tokens += thinking_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _model_name(provider: str) -> Optional[str]:
    return settings.azure_openai_deployment if provider == "azure_openai" else settings.gemini_model


def _fallback_deterministic_template(predicted_label: str, chunks: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    """
    Fallback deterministic output.
    `llm_grounding_verified` is strictly False because an LLM did NOT ground or verify these citations.
    """
    top_section = chunks[0]["section"] if chunks and "section" in chunks[0] else "General EU-MDR Provisions"
    return {
        "recommendation": f"Event classified as '{predicted_label}'. Primary regulatory basis: {top_section}.",
        "citations": [top_section] if chunks else [],
        "llm_grounding_verified": False,
        "warning": f"Degraded to deterministic template: {reason}",
    }


def _call_gemini(prompt: str, client: Optional[Any]) -> Tuple[StructuredRecommendation, Dict[str, Optional[int]]]:
    gemini_client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = gemini_client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuredRecommendation,
            temperature=0.0,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )
    if hasattr(response, "parsed") and response.parsed is not None:
        parsed: StructuredRecommendation = response.parsed
    else:
        parsed = StructuredRecommendation(**json.loads(response.text))
    return parsed, _extract_usage(response)


def _call_azure_openai(prompt: str, client: Optional[Any]) -> Tuple[StructuredRecommendation, Dict[str, Optional[int]]]:
    """
    Azure OpenAI through its v1 REST API (https://<resource>.openai.azure.com/openai/v1/chat/completions),
    with structured output. `client` is anything with a requests-style .post(); requests is the default.
    Temperature is not sent: reasoning models accept only the default.
    """
    http = client or requests
    url = settings.azure_openai_endpoint.rstrip("/") + "/openai/v1/chat/completions"
    request = {
        "url": url,
        "headers": {"api-key": os.getenv("AZURE_OPENAI_API_KEY", ""), "Content-Type": "application/json"},
        "json": {
            "model": settings.azure_openai_deployment,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_schema", "json_schema": RECOMMENDATION_JSON_SCHEMA},
            "max_completion_tokens": settings.azure_openai_max_completion_tokens,
        },
        "timeout": settings.azure_openai_timeout_seconds,
    }
    for attempt in range(settings.azure_openai_max_retries + 1):
        response = http.post(**request)
        if response.status_code not in (429, 503) or attempt == settings.azure_openai_max_retries:
            break
        wait = _retry_wait_seconds(response, attempt)
        logger.warning(f"Azure OpenAI HTTP {response.status_code}; retry {attempt + 1} in {wait:.1f} s")
        time.sleep(wait)
    if response.status_code >= 400:
        raise RuntimeError(f"Azure OpenAI HTTP {response.status_code}: {response.text[:300]}")
    data = response.json()
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        # An empty answer usually means the token budget ran out during reasoning.
        raise RuntimeError(f"Azure OpenAI returned no content (finish_reason={choice.get('finish_reason')})")
    usage = data.get("usage") or {}

    def count(name: str) -> Optional[int]:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    # completion_tokens already includes reasoning tokens, which are billed as output.
    return StructuredRecommendation(**json.loads(content)), {
        "input_tokens": count("prompt_tokens"),
        "output_tokens": count("completion_tokens"),
    }


def _retry_wait_seconds(response: Any, attempt: int) -> float:
    """Azure's Retry-After header (milliseconds or seconds) if present, else 2, 4, 8 ... s; capped."""
    headers = {k.lower(): v for k, v in (getattr(response, "headers", None) or {}).items()}
    try:
        if "retry-after-ms" in headers:
            wait = float(headers["retry-after-ms"]) / 1000
        elif "retry-after" in headers:
            wait = float(headers["retry-after"])
        else:
            wait = 2.0 ** (attempt + 1)
    except ValueError:
        wait = 2.0 ** (attempt + 1)
    return max(0.0, min(wait, settings.azure_openai_max_retry_wait_seconds))


def _normalise_section(title: str) -> str:
    """'Article 30 - Electronic system ...' and 'Article 30 Electronic System ...' compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _resolve_citations(cited: List[str], valid_sections: set) -> Tuple[List[str], List[str]]:
    """
    Map each cited title to the retrieved section it names, ignoring case, punctuation and spacing.
    Returns (resolved canonical titles, citations that match no retrieved section). A title that is
    only reformatted still points at a retrieved passage; one that names another section does not.
    """
    canonical = {_normalise_section(s): s for s in valid_sections}
    resolved, unmatched = [], []
    for citation in cited:
        match = canonical.get(_normalise_section(citation))
        if match is None:
            unmatched.append(citation)
        elif match not in resolved:
            resolved.append(match)
    return resolved, unmatched


def _missing_configuration(provider: str) -> Optional[str]:
    if provider == "azure_openai":
        missing = [name for name, value in (
            ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", settings.azure_openai_deployment),
            ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
        ) if not value]
        return f"{', '.join(missing)} unset" if missing else None
    return None if os.getenv("GEMINI_API_KEY") else "GEMINI_API_KEY unset"


def generate_grounded_recommendation(
    narrative: str,
    predicted_label: str,
    chunks: List[Dict[str, Any]],
    retrieval_fallback_triggered: bool,
    top_score: float,
    threshold: float = 0.55,
    client: Optional[Any] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates the grounded recommendation with the configured LLM provider (settings.llm_provider,
    or `provider` to override): 'gemini' or 'azure_openai'. Both use the same prompt and structured
    output, and both go through the same citation containment check: cited sections must be a strict
    subset of the retrieved chunk sections.
    Falls back to deterministic templating if the provider is not configured or the call fails.
    `client` replaces the provider's client (tests).
    """
    provider = provider or settings.llm_provider
    model = _model_name(provider)

    def tagged(result: Dict[str, Any]) -> Dict[str, Any]:
        return {**result, "provider": provider, "model": model}

    if not chunks:
        return tagged({
            "recommendation": f"Classified as '{predicted_label}'. No regulatory text available.",
            "citations": [],
            "llm_grounding_verified": False,
            "warning": "No regulatory context chunks available",
        })

    valid_sections = {c["section"] for c in chunks if "section" in c}
    prompt = build_recommendation_prompt(
        narrative=narrative,
        predicted_label=predicted_label,
        chunks=chunks,
        retrieval_fallback_triggered=retrieval_fallback_triggered,
        top_score=top_score,
        threshold=threshold,
    )

    missing = _missing_configuration(provider)
    if missing and client is None:
        logger.warning(f"{missing}. Degrading to deterministic recommendation.")
        return tagged(_fallback_deterministic_template(predicted_label, chunks, reason=missing))

    try:
        call = _call_azure_openai if provider == "azure_openai" else _call_gemini
        parsed, usage = call(prompt, client)
        # Enforce citation containment: every citation must name a retrieved section.
        cited_sections, hallucinated = _resolve_citations(parsed.citations, valid_sections)
        if hallucinated:
            logger.error(
                f"LLM hallucinated citations outside retrieved context: {hallucinated}. "
                "Pruning invalid citations."
            )
            return tagged({
                "recommendation": parsed.recommendation,
                "citations": cited_sections,
                "llm_grounding_verified": False,
                "warning": f"Pruned hallucinated citations: {hallucinated}",
                "usage": usage,
            })

        return tagged({
            "recommendation": parsed.recommendation,
            "citations": cited_sections,
            "llm_grounding_verified": True,
            "usage": usage,
        })

    except Exception as e:
        logger.error(f"{provider} recommendation generation failed ({e}). Degrading to deterministic template.")
        return tagged(_fallback_deterministic_template(predicted_label, chunks, reason=str(e)))
