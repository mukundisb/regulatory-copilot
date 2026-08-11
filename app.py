import logging
import sys
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from maude_classifier.classifier import load_model, predict_single
from maude_classifier.text_cleaner import clean_text
from config import settings

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


class ClassifyResponse(BaseModel):
    predicted_label: str
    probabilities: dict[str, float] | None = None


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

@app.post("/retrieve", response_model=list[dict[str, str]])
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=True)
