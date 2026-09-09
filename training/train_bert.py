"""
training/train_bert.py

Fine-tunes Bio_ClinicalBERT with cls_mean_concat head on processed MAUDE splits.
Supports standard PyTorch execution and Google Cloud Vertex AI Custom Training.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

# Ensure local imports resolve when executed on Vertex or locally
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maude_classifier.model import (
    ClinicalBERTConcatClassifier,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_clinicalbert")


class MAUDEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["narrative"].tolist()
        self.labels = [LABEL2ID[label] for label in df["label"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
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


def compute_class_weights(labels_series: pd.Series, device: torch.device) -> torch.Tensor:
    counts = labels_series.map(LABEL2ID).value_counts().sort_index()
    total = len(labels_series)
    weights = [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)]
    logger.info(f"Class Weights [D, I, M, O]: {[round(w, 4) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float, device=device)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, str]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)

    return avg_loss, macro_f1, report


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Resolve output directory: Vertex AI sets AIP_MODEL_DIR in custom containers
    model_output_dir = os.getenv("AIP_MODEL_DIR", args.output_dir)
    # If GCS URI, keep string format for gcsfs/Vertex, otherwise Path
    if not model_output_dir.startswith("gs://"):
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Export target destination: {model_output_dir}")

    # Load Parquet Data
    logger.info(f"Reading train split from {args.train_data}...")
    train_df = pd.read_parquet(args.train_data)
    logger.info(f"Reading val split from {args.val_data}...")
    val_df = pd.read_parquet(args.val_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    train_dataset = MAUDEDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = MAUDEDataset(val_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = ClinicalBERTConcatClassifier(
        pretrained_model_name=args.pretrained_model,
        num_labels=NUM_LABELS,
        dropout_rate=args.dropout
    ).to(device)

    # Balanced Cross-Entropy Loss
    class_weights = compute_class_weights(train_df["label"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with differential weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Commencing fine-tuning: {args.epochs} epochs | Total Steps: {total_steps} | Warmup: {warmup_steps}")

    best_val_macro_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_epoch = time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs["logits"], labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)

            if step % args.log_interval == 0 or step == len(train_loader):
                logger.info(
                    f"Epoch [{epoch}/{args.epochs}] | Step [{step}/{len(train_loader)}] | "
                    f"Batch Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss = epoch_loss / len(train_dataset)
        val_loss, val_macro_f1, val_report = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_epoch

        logger.info(
            f"\nEpoch {epoch} Summary ({elapsed:.1f}s):\n"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}\n"
            f"Validation Detailed Metrics:\n{val_report}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            logger.info(f"New best validation Macro F1: {best_val_macro_f1:.4f}. Preserving checkpoint...")
            
            # Temporary local path for packaging
            save_dir = Path("saved_checkpoint") if model_output_dir.startswith("gs://") else Path(model_output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            model.config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # If writing directly to GCS via Vertex AI environment
            if model_output_dir.startswith("gs://"):
                import subprocess
                subprocess.run(["gsutil", "-m", "cp", "-r", f"{save_dir}/*", model_output_dir], check=True)

    logger.info(f"Training completed. Peak Validation Macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bio_ClinicalBERT on MAUDE records.")
    parser.add_argument("--train-data", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val-data", type=str, default="data/processed/val.parquet")
    parser.add_argument("--output-dir", type=str, default="models/maude_clinicalbert")
    parser.add_argument("--pretrained-model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=100)
    parsed_args = parser.parse_args()

    run_training(parsed_args)