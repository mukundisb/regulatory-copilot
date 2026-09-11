"""
maude_classifier/serve.py

Vertex AI custom serving application for MAUDE regulatory classification.
Adheres to AIP_PREDICT_ROUTE, AIP_HEALTH_ROUTE, and AIP_HTTP_PORT environment variables.
"""

import os
import sys
from pathlib import Path
from typing import List, Union, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maude_classifier.model import (
    ClinicalBERTConcatClassifier,
    ID2LABEL,
    NUM_LABELS,
)

MODEL_VERSION = "Bio_ClinicalBERT-cls_mean_concat-v1"
HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")
MODEL_DIR = os.getenv("AIP_STORAGE_URI", "maude_classifier/model")

state: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device
    
    weights_path = Path(MODEL_DIR) / "pytorch_model.bin"
    if not weights_path.exists():
        raise RuntimeError(f"Weights artifact not found at {weights_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ClinicalBERTConcatClassifier(pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT")
    
    saved_weights = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(saved_weights)
    model.to(device)
    model.eval()

    state["model"] = model
    state["tokenizer"] = tokenizer
    yield
    state.clear()


app = FastAPI(title="MAUDE Adverse Event Classifier", lifespan=lifespan)


class PredictInstance(BaseModel):
    narrative: str


class PredictRequest(BaseModel):
    instances: List[Union[str, PredictInstance]] = Field(
        ..., description="List of text narratives or objects with narrative key."
    )


class PredictionResult(BaseModel):
    predicted_label: str
    probabilities: Dict[str, float]
    model_version: str


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]


@app.get(HEALTH_ROUTE)
async def health_check():
    if state.get("model") is None or state.get("tokenizer") is None:
        raise HTTPException(status_code=503, detail="Model weights not loaded")
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "device": str(state["device"]),
    }


@app.post(PREDICT_ROUTE, response_model=PredictResponse)
async def predict(request: PredictRequest):
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    texts = []
    for item in request.instances:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, PredictInstance):
            texts.append(item.narrative)
        elif isinstance(item, dict) and "narrative" in item:
            texts.append(item["narrative"])
        else:
            raise HTTPException(status_code=400, detail=f"Invalid instance format: {item}")

    if not texts:
        return PredictResponse(predictions=[])

    device = state["device"]
    tokenizer = state["tokenizer"]
    model = state["model"]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        probs = F.softmax(outputs["logits"], dim=-1).cpu()

    pred_indices = torch.argmax(probs, dim=-1).tolist()
    probs_list = probs.tolist()

    results = []
    for idx, row_probs in zip(pred_indices, probs_list):
        class_prob_map = {
            ID2LABEL[c]: round(row_probs[c], 4) for c in range(NUM_LABELS)
        }
        results.append(
            PredictionResult(
                predicted_label=ID2LABEL[idx],
                probabilities=class_prob_map,
                model_version=MODEL_VERSION,
            )
        )

    return PredictResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)