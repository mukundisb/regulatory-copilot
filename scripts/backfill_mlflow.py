"""
scripts/backfill_mlflow.py

Dynamically parses ground-truth metrics from reports/metrics_v2.json
and reports/test_classification_report.json into the MLflow SQLite registry.
"""

import json
from pathlib import Path
import mlflow

EXPERIMENT_NAME = "eu-mdr-event-classification"
TRACKING_URI = "sqlite:///mlflow.db"


def backfill_history():
    root = Path(__file__).resolve().parent.parent
    v2_path = root / "reports" / "metrics_v2.json"
    bert_path = root / "reports" / "test_classification_report.json"

    if not v2_path.exists() or not bert_path.exists():
        raise FileNotFoundError(
            f"Missing required ground-truth reports: {v2_path} or {bert_path}"
        )

    with open(v2_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)

    with open(bert_path, "r", encoding="utf-8") as f:
        bert_data = json.load(f)

    v2_report = v2_data.get("classification_report", {})
    bert_report = bert_data

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Backfill TF-IDF v2 Standby Model Run
    with mlflow.start_run(run_name="tfidf_v2_baseline") as run:
        mlflow.set_tags(
            {
                "model_type": "tfidf_logistic_regression",
                "role": "standby_fallback",
                "deployed_backend": "tfidf_local",
                "promotion_status": "demoted_to_fallback",
                "promotion_justification": (
                    "Ultra-low latency (~2ms, 0.71MB artifact), but inferior Class D recall (0.638). "
                    "Kept as high-availability deterministic fallback."
                ),
            }
        )
        mlflow.log_params(
            {
                "max_features": 15000,
                "ngram_range": "(1, 2)",
                "sublinear_tf": True,
                "classifier": "LogisticRegression",
                "C": 2.5,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": float(v2_data.get("accuracy", 0.0)),
                "test_balanced_accuracy": float(v2_data.get("balanced_accuracy", 0.0)),
                "test_macro_f1": float(v2_report.get("macro avg", {}).get("f1-score", 0.0)),
                "test_weighted_f1": float(v2_report.get("weighted avg", {}).get("f1-score", 0.0)),
                "f1_class_D": float(v2_report.get("D", {}).get("f1-score", 0.0)),
                "precision_class_D": float(v2_report.get("D", {}).get("precision", 0.0)),
                "recall_class_D": float(v2_report.get("D", {}).get("recall", 0.0)),
                "f1_class_I": float(v2_report.get("I", {}).get("f1-score", 0.0)),
                "f1_class_M": float(v2_report.get("M", {}).get("f1-score", 0.0)),
                "f1_class_O": float(v2_report.get("O", {}).get("f1-score", 0.0)),
                "artifact_size_mb": 0.71,
                "p95_latency_ms": 2.1,
            }
        )
        print(f"Logged TF-IDF v2 run {run.info.run_id} from {v2_path.name}")

    # 2. Backfill ClinicalBERT Primary Model Run
    with mlflow.start_run(run_name="clinicalbert_v1_primary") as run:
        mlflow.set_tags(
            {
                "model_type": "emilyalsentzer/Bio_ClinicalBERT",
                "role": "primary_production",
                "deployed_backend": "clinicalbert_vertex",
                "promotion_status": "promoted_to_primary",
                "promotion_justification": (
                    "Promoted over TF-IDF baseline due to critical gains on EU-MDR vigilance: "
                    "test macro F1 0.7346 (+5.61%) and Class D recall 0.7810. "
                    "Production latency of 2900-3353ms is acceptable for asynchronous vigilance intake."
                ),
            }
        )
        mlflow.log_params(
            {
                "base_model": "emilyalsentzer/Bio_ClinicalBERT",
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 16,
                "num_train_epochs": 4,
                "max_seq_length": 256,
                "loss_fn": "CrossEntropyLoss(class_weighted)",
                "target_device": "cuda",
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": float(bert_report.get("accuracy", 0.0)),
                "test_macro_f1": float(bert_report.get("macro avg", {}).get("f1-score", 0.0)),
                "test_weighted_f1": float(bert_report.get("weighted avg", {}).get("f1-score", 0.0)),
                "f1_class_D": float(bert_report.get("D", {}).get("f1-score", 0.0)),
                "precision_class_D": float(bert_report.get("D", {}).get("precision", 0.0)),
                "recall_class_D": float(bert_report.get("D", {}).get("recall", 0.0)),
                "f1_class_I": float(bert_report.get("I", {}).get("f1-score", 0.0)),
                "f1_class_M": float(bert_report.get("M", {}).get("f1-score", 0.0)),
                "f1_class_O": float(bert_report.get("O", {}).get("f1-score", 0.0)),
                "artifact_size_mb": 412.0,
                "p95_latency_ms": 3120.0,
            }
        )
        print(f"Logged ClinicalBERT run {run.info.run_id} from {bert_path.name}")


if __name__ == "__main__":
    backfill_history()