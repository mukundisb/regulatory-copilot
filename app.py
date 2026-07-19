from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from maude_classifier.classifier import load_model, predict_single
from maude_classifier.text_cleaner import clean_text

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["maude_pipeline"] = load_model()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


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
    cleaned = clean_text(data.narrative)
    result = predict_single(ml_models["maude_pipeline"], cleaned)
    return ClassifyResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
