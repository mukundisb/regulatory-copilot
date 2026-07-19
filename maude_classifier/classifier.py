# Vendored from mukundisb/maude-nlp-classifier (src/model/classifier.py),
# trimmed to the load + single-narrative inference path only (no training code).

import os
import logging

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "maude_classifier.joblib")


def load_model(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the persisted fine-tuned pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline


def predict_single(pipeline: Pipeline, text: str) -> dict:
    """
    Run inference on a single narrative text string.

    Returns:
        Dict with predicted label and per-class probabilities (if available).
    """
    prediction = pipeline.predict([text])[0]
    result = {"predicted_label": prediction}

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        result["probabilities"] = dict(zip(pipeline.classes_, proba.tolist()))
    elif hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([text])[0]
        result["decision_scores"] = dict(zip(pipeline.classes_, scores.tolist()))

    return result
