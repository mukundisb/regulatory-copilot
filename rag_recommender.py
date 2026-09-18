import json
import logging
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from config import settings

logger = logging.getLogger("regulatory_copilot.llm")


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


def generate_grounded_recommendation(
    narrative: str,
    predicted_label: str,
    chunks: List[Dict[str, Any]],
    retrieval_fallback_triggered: bool,
    top_score: float,
    threshold: float = 0.55,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes grounded recommendation generation using the Gemini API.
    Validates that returned citations are a strict subset of retrieved chunk sections.
    Falls back to deterministic templating if API key is missing or call fails.
    """
    if not chunks:
        return {
            "recommendation": f"Classified as '{predicted_label}'. No regulatory text available.",
            "citations": [],
            "llm_grounding_verified": False,
            "warning": "No regulatory context chunks available",
        }

    valid_sections = {c["section"] for c in chunks if "section" in c}
    prompt = build_recommendation_prompt(
        narrative=narrative,
        predicted_label=predicted_label,
        chunks=chunks,
        retrieval_fallback_triggered=retrieval_fallback_triggered,
        top_score=top_score,
        threshold=threshold,
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and client is None:
        logger.warning("GEMINI_API_KEY is not set. Degrading to deterministic recommendation.")
        return _fallback_deterministic_template(predicted_label, chunks, reason="GEMINI_API_KEY unset")

    try:
        gemini_client = client or genai.Client(api_key=api_key)

        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredRecommendation,
                temperature=0.0,
                system_instruction="You are an EU-MDR regulatory compliance engine. Return only strictly grounded JSON matching the schema.",
            ),
        )

        if hasattr(response, "parsed") and response.parsed is not None:
            parsed: StructuredRecommendation = response.parsed
        else:
            raw_data = json.loads(response.text)
            parsed = StructuredRecommendation(**raw_data)

        cited_sections = parsed.citations

        # Enforce citation containment
        hallucinated = set(cited_sections) - valid_sections
        if hallucinated:
            logger.error(
                f"LLM hallucinated citations outside retrieved context: {hallucinated}. "
                "Pruning invalid citations."
            )
            valid_citations = [s for s in cited_sections if s in valid_sections]
            return {
                "recommendation": parsed.recommendation,
                "citations": valid_citations,
                "llm_grounding_verified": False,
                "warning": f"Pruned hallucinated citations: {list(hallucinated)}",
            }

        return {
            "recommendation": parsed.recommendation,
            "citations": cited_sections,
            "llm_grounding_verified": True,
        }

    except Exception as e:
        logger.error(f"Gemini API recommendation generation failed ({e}). Degrading to deterministic template.")
        return _fallback_deterministic_template(predicted_label, chunks, reason=str(e))