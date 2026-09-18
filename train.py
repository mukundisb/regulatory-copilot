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
import mlflow
from sklearn.metrics import classification_report, accuracy_score, f1_score

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
        raise FileNotFoundError(
            f"Training dataset not found: {in_data}. Run transform pipeline first."
        )

    logger.info("Loading processed dataset from %s", in_data)
    df = pd.read_csv(in_data)

    if "narrative_text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'narrative_text' and 'label' columns. Found: {list(df.columns)}"
        )

    # Filter out missing records
    df = df.dropna(subset=["narrative_text", "label"])
    X = df["narrative_text"].astype(str)
    y = df["label"].astype(str)

    logger.info(
        "Dataset shape: %d records. Class distribution:\n%s",
        len(df),
        y.value_counts().to_dict(),
    )

    # Check minimum class threshold for stratified split
    min_class_count = y.value_counts().min()
    stratify_target = y if min_class_count >= 2 else None
    if stratify_target is None:
        logger.warning(
            "Smallest class count is %d; disabling stratification.",
            min_class_count,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    logger.info(
        "Training set size: %d | Test evaluation set size: %d",
        len(X_train),
        len(X_test),
    )

    # Set up MLflow tracking store
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("eu-mdr-event-classification")

    with mlflow.start_run(run_name="tfidf_standby_trainer"):
        # Log metadata & hyperparameters
        mlflow.set_tags(
            {
                "model_type": "tfidf_logistic_regression",
                "role": "standby_fallback",
                "deployed_backend": "tfidf_local",
            }
        )

        pipeline = build_pipeline()

        # Log hyperparameters directly from the constructed pipeline
        clf_step = pipeline.named_steps.get("clf")
        tfidf_step = pipeline.named_steps.get("tfidf")

        params = {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        if tfidf_step:
            params.update(
                {
                    "max_features": getattr(tfidf_step, "max_features", None),
                    "ngram_range": str(getattr(tfidf_step, "ngram_range", None)),
                    "sublinear_tf": getattr(tfidf_step, "sublinear_tf", None),
                }
            )
        if clf_step:
            params.update(
                {
                    "classifier": clf_step.__class__.__name__,
                    "C": getattr(clf_step, "C", None),
                    "class_weight": str(getattr(clf_step, "class_weight", None)),
                    "solver": getattr(clf_step, "solver", None),
                    "max_iter": getattr(clf_step, "max_iter", None),
                }
            )

        mlflow.log_params(params)

        logger.info("Fitting Pipeline (tfidf -> LogisticRegression)...")
        pipeline.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_test, y_pred, zero_division=0)

        logger.info("Evaluation Complete:\n\n%s", report_text)
        logger.info(
            "Accuracy: %.4f | Balanced Accuracy: %.4f | Macro F1: %.4f",
            acc,
            bal_acc,
            macro_f1,
        )

        # Log evaluation metrics to MLflow
        mlflow_metrics = {
            "test_accuracy": acc,
            "test_balanced_accuracy": bal_acc,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "f1_class_D": float(report_dict.get("D", {}).get("f1-score", 0.0)),
            "precision_class_D": float(
                report_dict.get("D", {}).get("precision", 0.0)
            ),
            "recall_class_D": float(report_dict.get("D", {}).get("recall", 0.0)),
            "f1_class_I": float(report_dict.get("I", {}).get("f1-score", 0.0)),
            "f1_class_M": float(report_dict.get("M", {}).get("f1-score", 0.0)),
            "f1_class_O": float(report_dict.get("O", {}).get("f1-score", 0.0)),
        }
        mlflow.log_metrics(mlflow_metrics)

        # Build exported metrics payload
        metrics_payload = {
            "model_version": "v2",
            "dataset_path": str(in_data),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classes": sorted(list(pipeline.named_steps["clf"].classes_)),
            "classification_report": report_dict,
        }

        # Save metrics JSON artifact (git-tracked)
        metrics_path = Path(METRICS_OUTPUT_PATH)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info("Metrics report exported to %s", metrics_path)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        # Save trained model artifact
        model_path = Path(MODEL_OUTPUT_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        logger.info(
            "Trained model serialized to %s (config.py remains untouched)",
            model_path,
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")

    return metrics_payload


if __name__ == "__main__":
    run_training()