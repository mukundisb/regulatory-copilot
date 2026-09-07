import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import settings
from maude_classifier.classifier import load_model, predict_single as predict_tfidf
from maude_classifier.text_cleaner import clean_text
from maude_classifier.vertex_client import VertexClassifierClient, VertexColdStartException
from rag_pipeline import init_store, query_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("regulatory_copilot")

# Empirically calibrated threshold based on true match distribution (0.6117 - 0.7412)
RETRIEVAL_QUALITY_THRESHOLD = 0.55

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Always load local TF-IDF model as primary or resilient fallback
    ml_models["maude_pipeline"] = load_model(settings.model_path)
    logger.info(f"Loaded local model binary from {settings.model_path}")

    # 2. Pre-warm Vertex AI client at startup if configured as backend
    if settings.classifier_backend == "clinicalbert_vertex":
        if not settings.vertex_endpoint_id:
            logger.warning("CLASSIFIER_BACKEND is 'clinicalbert_vertex' but VERTEX_ENDPOINT_ID is empty!")
            ml_models["vertex_client"] = None
        else:
            try:
                logger.info(f"Pre-warming Vertex AI client for endpoint: {settings.vertex_endpoint_id}...")
                ml_models["vertex_client"] = VertexClassifierClient()
                logger.info("Vertex AI persistent client successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to pre-warm Vertex AI client on startup: {e}")
                ml_models["vertex_client"] = None
    else:
        ml_models["vertex_client"] = None
        logger.info("Running on local TF-IDF backend.")

    # 3. Initialize ChromaDB collection
    init_store()

    yield

    ml_models.clear()
    logger.info("Application shutdown completed.")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://regulatory-copilot.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    narrative: str

    @field_validator("narrative")
    @classmethod
    def narrative_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("narrative must not be empty")
        return value


class RetrieveResult(BaseModel):
    chunk_id: str
    section: str
    text: str
    similarity_score: float


class ClassifyResponse(BaseModel):
    predicted_label: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    backend_used: str | None = None
    warning: str | None = None


class AssessResponse(BaseModel):
    predicted_label: str
    confidence: float
    retrieval_query_used: str
    retrieved_chunks: list[RetrieveResult]
    recommendation: str
    fallback_triggered: bool = False
    backend_used: str | None = None
    warning: str | None = None


def _classify_narrative(raw_narrative: str) -> dict:
    """
    (Availability Over Fidelity):
    Attempts inference against Vertex AI ClinicalBERT first. On ANY failure 
    (cold start timeout, network disruption, authentication failure, or gRPC error),
    it logs an internal error with stack trace and gracefully fails over to the 
    in-memory TF-IDF model so the caller always receives a complete response.
    """
    cleaned = clean_text(raw_narrative)

    if settings.classifier_backend == "clinicalbert_vertex":
        vertex_client = ml_models.get("vertex_client")
        if vertex_client is None and settings.vertex_endpoint_id:
            try:
                vertex_client = VertexClassifierClient()
                ml_models["vertex_client"] = vertex_client
            except Exception as e:
                logger.error(f"Failed to lazily instantiate VertexClassifierClient: {e}", exc_info=True)

        if vertex_client:
            try:
                return vertex_client.predict_with_backoff(cleaned)
            except VertexColdStartException as e:
                logger.warning(f"Vertex cold start timeout: {e} - Degrading to TF-IDF fallback.")
                fallback_res = predict_tfidf(ml_models["maude_pipeline"], cleaned)
                fallback_res["backend_used"] = "tfidf_fallback"
                fallback_res["warning"] = "Vertex AI container warming up; served via local TF-IDF fallback."
                return fallback_res
            except Exception as e:
                logger.error(
                    f"Vertex AI inference error ({type(e).__name__}: {e}). Failing over to TF-IDF.",
                    exc_info=True,
                )
                fallback_res = predict_tfidf(ml_models["maude_pipeline"], cleaned)
                fallback_res["backend_used"] = "tfidf_fallback"
                fallback_res["warning"] = f"Vertex AI unavailable ({type(e).__name__}); served via local TF-IDF fallback."
                return fallback_res

    # Default: Local TF-IDF model
    result = predict_tfidf(ml_models["maude_pipeline"], cleaned)
    result["backend_used"] = "tfidf"
    return result


def build_retrieval_query(predicted_label: str, raw_narrative: str) -> str:
    """Decision Point 1: Steer retrieval based on classifier label."""
    if predicted_label in ["D", "I"]:
        return f"serious incident reporting vigilance timelines manufacturer obligations {raw_narrative}"
    elif predicted_label == "M":
        return f"device malfunction root cause analysis trend reporting corrective action {raw_narrative}"
    return raw_narrative


def retrieve_with_fallback(primary_query: str, fallback_query: str, top_k: int = 3) -> tuple[list[dict], str, bool]:
    """
    Decision Point 2: Evaluate retrieval quality and conditionally fallback.
    Returns (chunks, query_used, fallback_triggered).
    """
    primary_chunks = query_store(primary_query, top_k=top_k)
    primary_top_score = (
        primary_chunks[0]["similarity_score"]
        if primary_chunks and "similarity_score" in primary_chunks[0]
        else 0.0
    )

    if primary_top_score >= RETRIEVAL_QUALITY_THRESHOLD:
        return primary_chunks, primary_query, False

    fallback_chunks = query_store(fallback_query, top_k=top_k)
    fallback_top_score = (
        fallback_chunks[0]["similarity_score"]
        if fallback_chunks and "similarity_score" in fallback_chunks[0]
        else 0.0
    )

    if fallback_top_score > primary_top_score:
        return fallback_chunks, fallback_query, True
    return primary_chunks, primary_query, True


def generate_recommendation(label: str, top_section: str, confidence: float) -> str:
    """Generates a deterministic regulatory guidance summary based on label and top retrieved section."""
    actions = {
        "D": "Mandatory vigilance reporting required. Initiate immediate risk assessment and submit report within strict statutory timelines (within 2 to 10 days depending on public health threat severity).",
        "I": "Serious deterioration in health detected. Notify competent authority within 15 days of becoming aware, record in vigilance register, and initiate root-cause investigation.",
        "M": "Device malfunction identified. Log in post-market surveillance system, verify if incident meets trend-reporting thresholds, and assess need for Field Safety Corrective Action (FSCA).",
        "O": "No immediate serious adverse event or critical malfunction detected. Archive under standard customer complaints register and continue routine post-market surveillance monitoring.",
    }
    action_text = actions.get(label, "Review event under standard quality management system procedures.")
    return (
        f"Event classified as '{label}' (confidence: {confidence:.2%}). "
        f"Primary regulatory basis: {top_section}. "
        f"Recommended Action: {action_text}"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "backend": settings.classifier_backend,
        "vertex_endpoint_ready": ml_models.get("vertex_client") is not None,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(data: ClassifyRequest):
    start_time = time.perf_counter()
    result = _classify_narrative(data.narrative)
    predicted_label = result["predicted_label"]

    # Normalize confidence across Vertex AI and TF-IDF outputs
    confidence = result.get("confidence")
    if confidence is None:
        probs = result.get("probabilities") or {}
        confidence = float(probs.get(predicted_label, 1.0))

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"event = classify_success narrative_len = {len(data.narrative)} "
        f"label = {predicted_label} confidence = {confidence:.4f} backend = {result.get('backend_used')} "
        f"latency_ms = {latency_ms:.2f}"
    )
    return ClassifyResponse(
        predicted_label=predicted_label,
        confidence=round(confidence, 4),
        probabilities=result.get("probabilities"),
        backend_used=result.get("backend_used"),
        warning=result.get("warning"),
    )


@app.post("/retrieve", response_model=list[RetrieveResult])
def retrieve(data: ClassifyRequest):
    start_time = time.perf_counter()
    results = query_store(data.narrative, top_k=3)
    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"event = retrieve_success narrative_len = {len(data.narrative)} "
        f"num_results = {len(results)} latency_ms = {latency_ms:.2f}"
    )
    return results


@app.post("/assess", response_model=AssessResponse)
def assess(data: ClassifyRequest):
    start_time = time.perf_counter()

    # 1. ML Inference via unified router
    clf_result = _classify_narrative(data.narrative)
    predicted_label = clf_result["predicted_label"]

    # Calculate confidence robustly across tfidf and vertex payloads
    confidence = clf_result.get("confidence")
    if confidence is None:
        probs = clf_result.get("probabilities") or {}
        confidence = float(probs.get(predicted_label, 1.0))

    # 2. Label-driven query synthesis
    primary_query = build_retrieval_query(predicted_label, data.narrative)

    # 3. Adaptive retrieval with quality evaluation and fallback
    chunks, query_used, fallback_triggered = retrieve_with_fallback(
        primary_query=primary_query,
        fallback_query=data.narrative,
        top_k=3,
    )

    # 4. Deterministic guidance templating
    top_section = chunks[0]["section"] if chunks else "General EU-MDR Provisions"
    recommendation = generate_recommendation(predicted_label, top_section, confidence)

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"event = assess_success predicted_label = {predicted_label} confidence = {confidence:.4f} "
        f"backend = {clf_result.get('backend_used')} fallback_triggered = {fallback_triggered} "
        f"latency_ms = {latency_ms:.2f}"
    )

    return AssessResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        retrieval_query_used=query_used,
        retrieved_chunks=chunks,
        recommendation=recommendation,
        fallback_triggered=fallback_triggered,
        backend_used=clf_result.get("backend_used"),
        warning=clf_result.get("warning"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=True)