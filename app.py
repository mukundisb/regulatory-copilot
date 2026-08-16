import logging
import sys
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from maude_classifier.classifier import load_model, predict_single
from maude_classifier.text_cleaner import clean_text
from config import settings
from rag_pipeline import init_store, query_store

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("regulatory_copilot")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["maude_pipeline"] = load_model(settings.model_path)
    logger.info(f"Loaded model binary from {settings.model_path}")
    init_store()  # Initialize ChromaDB collection
    yield
    ml_models.clear()


app = FastAPI(title = settings.app_name, lifespan = lifespan)

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
    probabilities: dict[str, float] | None = None

class AssessResponse(BaseModel):
    predicted_label: str
    confidence: float
    retrieval_query_used: str
    retrieved_chunks: list[RetrieveResult]
    recommendation: str

def generate_recommendation(label: str, top_section: str, confidence: float) -> str:
    """Generates a deterministic regulatory guidance summary based on label and top retrieved section."""
    actions = {
        "D": "Mandatory vigilance reporting required. Initiate immediate risk assessment and submit report within strict statutory timelines (within 2 to 10 days depending on public health threat severity).",
        "I": "Serious deterioration in health detected. Notify competent authority within 15 days of becoming aware, record in vigilance register, and initiate root-cause investigation.",
        "M": "Device malfunction identified. Log in post-market surveillance system, verify if incident meets trend-reporting thresholds, and assess need for Field Safety Corrective Action (FSCA).",
        "O": "No immediate serious adverse event or critical malfunction detected. Archive under standard customer complaints register and continue routine post-market surveillance monitoring."
    }
    action_text = actions.get(label, "Review event under standard quality management system procedures.")
    return (
        f"Event classified as '{label}' (confidence: {confidence:.2%}). "
        f"Primary regulatory basis: {top_section}. "
        f"Recommended Action: {action_text}"
    )

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(data: ClassifyRequest):
    start_time = time.perf_counter()
    cleaned = clean_text(data.narrative)
    result = predict_single(ml_models["maude_pipeline"], cleaned)

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"event = classify_success narrative_len = {len(data.narrative)} "
        f"label = {result['predicted_label']} latency_ms = {latency_ms:.2f}"
    )
    return result

@app.post("/retrieve", response_model=list[RetrieveResult])
def retrieve(data: ClassifyRequest):
    from rag_pipeline import query_store

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

    # 1. Step 1: Access ing clean input and run loaded model pipeline
    cleaned = clean_text(data.narrative)
    clf_result = predict_single(ml_models["maude_pipeline"], cleaned)

    predicted_label = clf_result["predicted_label"]
    confidence = float(clf_result['probabilities'][predicted_label])

    # 2. Step 2: Decision Branching - Formulate retrieval query based on predicted category
    if predicted_label in ["D", "I"]:
        # Severe safety event: steer retrieval towards vigilance reporting and incident timelines
        retrieval_query = f"serious incident reporting vigilance timelines manufacturer obligations {data.narrative}"
    elif predicted_label == "M":
        # Malfunction: steer retrieval towards CAPA, trend analysis, and investigation
        retrieval_query = f"device malfunction root cause analysis trend reporting corrective action {data.narrative}"
    else:
        # Default / Other: retrieve against general regulatory requirements
        retrieval_query = data.narrative

    # 3. Step 3: Vector retrieval
    chunks = query_store(retrieval_query, top_k=3)

    # 4. Step 4: Deterministic guidance templating
    top_section = chunks[0]["section"] if chunks else "General EU-MDR Provisions"
    recommendation = generate_recommendation(predicted_label, top_section, confidence)

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"event = assess_success predicted_label = {predicted_label} confidence = {confidence:.4f} "
        f"num_results = {len(chunks)} latency_ms = {latency_ms:.2f}"
    )

    return AssessResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        retrieval_query_used=retrieval_query,
        retrieved_chunks=chunks,
        recommendation=recommendation
    )                      

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=True)
