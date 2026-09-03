"""
Model training pipeline for MAUDE adverse event narrative classification.

Builds an end-to-end scikit-learn Pipeline with steps:
  - 'tfidf': TfidfVectorizer (sublinear tf scaling, n-gram bounds)
  - 'clf': LogisticRegression (multinomial / multi_class='ovr' or 'multinomial', balanced class weighting)

Maintains exact step naming parity with the production v1 model.
Saves metrics to JSON and serializes the new model to a v2 destination.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configurable paths via environment or defaults
DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/processed/maude_train.csv")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "maude_classifier/model/maude_classifier_v2.joblib")
METRICS_OUTPUT_PATH = os.getenv("METRICS_OUTPUT_PATH", "reports/metrics_v2.json")

TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.20"))
RANDOM_STATE = int(os.getenv("TRAIN_RANDOM_STATE", "42"))


def build_pipeline() -> Pipeline:
    """
    Constructs the canonical two-stage classifier.
    Step names ('tfidf', 'clf') must remain identical to production v1.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def run_training() -> Dict[str, Any]:
    in_data = Path(DATA_PATH)
    if not in_data.exists():
        raise FileNotFoundError(f"Training dataset not found: {in_data}. Run transform pipeline first.")

    logger.info("Loading processed dataset from %s", in_data)
    df = pd.read_csv(in_data)

    if "narrative_text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'narrative_text' and 'label' columns. Found: {list(df.columns)}")

    # Filter out missing records
    df = df.dropna(subset=["narrative_text", "label"])
    X = df["narrative_text"].astype(str)
    y = df["label"].astype(str)

    logger.info("Dataset shape: %d records. Class distribution:\n%s", len(df), y.value_counts().to_dict())

    # Check minimum class threshold for stratified split
    min_class_count = y.value_counts().min()
    stratify_target = y if min_class_count >= 2 else None
    if stratify_target is None:
        logger.warning("Smallest class count is %d; disabling stratification.", min_class_count)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    logger.info("Training set size: %d | Test evaluation set size: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()

    logger.info("Fitting Pipeline (tfidf -> LogisticRegression)...")
    pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    logger.info("Evaluation Complete:\n\n%s", report_text)
    logger.info("Accuracy: %.4f | Balanced Accuracy: %.4f", acc, bal_acc)

    metrics_payload = {
        "model_version": "v2",
        "dataset_path": str(in_data),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "classes": sorted(list(pipeline.named_steps["clf"].classes_)),
        "classification_report": report_dict,
    }

    # Save metrics report (git-tracked)
    metrics_path = Path(METRICS_OUTPUT_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Metrics report exported to %s", metrics_path)

    # Save trained model artifact (isolated v2 path)
    model_path = Path(MODEL_OUTPUT_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Trained model serialized to %s (config.py remains untouched)", model_path)

    return metrics_payload


if __name__ == "__main__":
    run_training()