"""
training/train_bert.py

Fine-tunes ClinicalBERT using the native PyTorch training loop:
- Class-weighted CrossEntropyLoss
- Parquet data ingestion (train.parquet, val.parquet)
- Model contract: ClinicalBERTConcatClassifier(pretrained_model_name=...)
- Forward output: outputs["logits"]
- Checkpoint sync to AIP_MODEL_DIR / GCS
- MLflow SQLite experiment tracking (sqlite:///mlflow.db)
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maude_classifier.model import (
    ClinicalBERTConcatClassifier,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_clinicalbert")


class ParquetMAUDEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["narrative"].astype(str).tolist()
        self.labels = [LABEL2ID[label] for label in df["label"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(val_loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "val_loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def train_clinicalbert(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Resolve Parquet dataset paths
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Parquet splits not found: {train_path} or {val_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # 2. Compute balanced class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=np.array([LABEL2ID[l] for l in train_df["label"]]),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # 3. Tokenizer and DataLoader initialization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = ParquetMAUDEDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = ParquetMAUDEDataset(val_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # 4. Instantiate model using exact pretrained_model_name signature
    model = ClinicalBERTConcatClassifier(
        pretrained_model_name=args.model_name_or_path,
        num_labels=NUM_LABELS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    # 5. MLflow Tracking Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("eu-mdr-event-classification")

    run_name = f"clinicalbert_{args.run_tag}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_type": args.model_name_or_path,
                "role": "primary_candidate",
                "loss_mechanism": "balanced_class_weights",
            }
        )
        mlflow.log_params(
            {
                "base_model": args.model_name_or_path,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "class_weights": class_weights.tolist(),
            }
        )

        best_macro_f1 = 0.0
        best_eval_res = None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / "clinicalbert_best.pt"

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            eval_res = evaluate(model, val_loader, loss_fn, device)
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f",
                epoch,
                args.epochs,
                train_loss / max(len(train_loader), 1),
                eval_res["val_loss"],
                eval_res["macro_f1"],
            )

            if eval_res["macro_f1"] > best_macro_f1:
                best_macro_f1 = eval_res["macro_f1"]
                best_eval_res = eval_res
                torch.save(model.state_dict(), model_save_path)
                tokenizer.save_pretrained(str(output_dir))
                logger.info("Saved new best model checkpoint to %s", model_save_path)

        # 6. Log metrics
        if best_eval_res:
            rep = best_eval_res["report"]
            mlflow.log_metrics(
                {
                    "val_loss": best_eval_res["val_loss"],
                    "val_accuracy": best_eval_res["accuracy"],
                    "val_macro_f1": best_eval_res["macro_f1"],
                    "val_weighted_f1": best_eval_res["weighted_f1"],
                    "f1_class_D": float(rep.get("D", {}).get("f1-score", 0.0)),
                    "precision_class_D": float(rep.get("D", {}).get("precision", 0.0)),
                    "recall_class_D": float(rep.get("D", {}).get("recall", 0.0)),
                    "f1_class_I": float(rep.get("I", {}).get("f1-score", 0.0)),
                    "f1_class_M": float(rep.get("M", {}).get("f1-score", 0.0)),
                    "f1_class_O": float(rep.get("O", {}).get("f1-score", 0.0)),
                }
            )

        # 7. Sync artifacts to MLflow & GCS AIP_MODEL_DIR
        mlflow.log_artifacts(str(output_dir), artifact_path="model_checkpoints")

        aip_model_dir = os.getenv("AIP_MODEL_DIR")
        if aip_model_dir:
            logger.info("Detected Vertex AI AIP_MODEL_DIR: %s. Syncing artifacts...", aip_model_dir)
            if aip_model_dir.startswith("gs://"):
                from google.cloud import storage

                storage_client = storage.Client()
                path_parts = aip_model_dir.replace("gs://", "").split("/", 1)
                bucket_name = path_parts[0]
                prefix = path_parts[1] if len(path_parts) > 1 else ""

                bucket = storage_client.bucket(bucket_name)
                for file_path in output_dir.glob("*"):
                    if file_path.is_file():
                        blob_name = f"{prefix.rstrip('/')}/{file_path.name}" if prefix else file_path.name
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))
                        logger.info("Uploaded %s to gs://%s/%s", file_path.name, bucket_name, blob_name)
            else:
                dest = Path(aip_model_dir)
                dest.mkdir(parents=True, exist_ok=True)
                for item in output_dir.glob("*"):
                    if item.is_file():
                        shutil.copy(item, dest / item.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ClinicalBERT classifier with native PyTorch & MLflow")
    parser.add_argument("--train-data", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val-data", type=str, default="data/processed/val.parquet")
    parser.add_argument("--output-dir", type=str, default="models/clinicalbert")
    parser.add_argument("--model-name-or-path", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--run-tag", type=str, default="v1")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    train_clinicalbert(parse_args())