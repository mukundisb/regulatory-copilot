# Vendored from mukundisb/maude-nlp-classifier (src/model/classifier.py),
# trimmed to the load + single-narrative inference path only (no training code).

from sklearn.pipeline import Pipeline


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
