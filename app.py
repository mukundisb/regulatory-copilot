import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import joblib

from config import settings
from maude_classifier.classifier import predict_single as predict_tfidf
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

def get_maude_pipeline():
    """Retrieve the preloaded TF-IDF pipeline or lazy-load if uninitialized."""
    if "maude_pipeline" not in ml_models or ml_models["maude_pipeline"] is None:
        logger.warning(f"Standby pipeline missing from cache. Lazy loading from {settings.model_path}...")
        try:
            ml_models["maude_pipeline"] = joblib.load(settings.model_path)
        except Exception as e:
            logger.critical(f"FATAL: Standby TF-IDF model could not be loaded: {e}")
            raise RuntimeError(f"Standby TF-IDF model unavailable at {settings.model_path}") from e
    return ml_models["maude_pipeline"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Fail-Fast: Standby TF-IDF model MUST exist and load cleanly
    logger.info(f"Loading mandatory standby TF-IDF pipeline from {settings.model_path}...")
    try:
        ml_models["maude_pipeline"] = joblib.load(settings.model_path)
        logger.info("Standby TF-IDF pipeline successfully cached.")
    except Exception as e:
        logger.critical(f"FATAL: Could not load standby TF-IDF model from {settings.model_path}: {e}")
        raise RuntimeError(f"Startup aborted: Standby model missing or corrupt at {settings.model_path}") from e

    # 2. Pre-warm Vertex AI client at startup if configured as backend (non-fatal; retried on demand)
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
                logger.warning(f"Pre-warming Vertex AI client failed on startup ({e}). Will retry on demand.")
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
    it logs an internal error and gracefully fails over to the standby TF-IDF model,
    returning an explicit warning field.
    """
    cleaned = clean_text(raw_narrative)

    if settings.classifier_backend == "clinicalbert_vertex":
        vertex_client = ml_models.get("vertex_client")

        # Retry-on-None: attempt lazy re-instantiation if boot pre-warming failed
        if vertex_client is None and settings.vertex_endpoint_id:
            try:
                logger.info("Vertex client uninitialized. Attempting on-demand instantiation...")
                vertex_client = VertexClassifierClient()
                ml_models["vertex_client"] = vertex_client
                logger.info("On-demand Vertex AI client instantiation succeeded.")
            except Exception as e:
                logger.error(f"On-demand Vertex AI client instantiation failed: {e}")
                vertex_client = None

        if vertex_client is not None:
            try:
                return vertex_client.predict_with_backoff(cleaned)
            except VertexColdStartException as e:
                warning_msg = f"Vertex AI cold start timeout exceeded ({e})"
                logger.warning(f"{warning_msg}. Degrading to standby TF-IDF.")
                return _degrade_to_tfidf(cleaned, warning_msg)
            except Exception as e:
                warning_msg = f"Vertex AI inference error: {e}"
                logger.error(f"{warning_msg}. Degrading to standby TF-IDF.")
                return _degrade_to_tfidf(cleaned, warning_msg)
        else:
            warning_msg = "Vertex AI client unavailable (failed instantiation)"
            logger.warning(f"{warning_msg}. Degrading to standby TF-IDF.")
            return _degrade_to_tfidf(cleaned, warning_msg)
    else:
        pipeline = get_maude_pipeline()
        return predict_tfidf(pipeline, cleaned)


def _degrade_to_tfidf(cleaned: str, reason: str) -> dict:
    """Helper to run fallback inference and attach standard audit/warning fields."""
    pipeline = get_maude_pipeline()
    fallback_res = predict_tfidf(pipeline, cleaned)
    return {
        **fallback_res,
        "backend_used": "tfidf_fallback",
        "warning": f"Degraded to TF-IDF fallback: {reason}",
    }


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