"""
scripts/evaluate_test.py

Statutory test-set evaluation script for MAUDE regulatory classification.
Evaluates the fine-tuned Bio_ClinicalBERT cls_mean_concat model against
the held-out test split (data/processed/test.parquet) and outputs canonical
metrics to disk and stdout.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Ensure root directory is on PYTHONPATH
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_test")


class MAUDETestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["narrative"].astype(str).tolist()
        self.labels = [LABEL2ID[label] for label in df["label"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAUDE model on test set.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="maude_classifier/model",
        help="Directory containing pytorch_model.bin and tokenizer configs.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.parquet",
        help="Path to held-out test parquet partition.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to dump evaluation JSON and metrics summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Sequence truncation/padding length.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        logger.error(f"Missing weights file at {weights_path}")
        sys.exit(1)
    if not test_path.exists():
        logger.error(f"Missing test data split at {test_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # Load dataset
    logger.info(f"Loading test split from {test_path}...")
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(test_df)} test records.")

    # Load model and tokenizer from local checkpoint
    logger.info(f"Loading model checkpoint from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ClinicalBERTConcatClassifier(pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT")
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_dataset = MAUDETestDataset(test_df, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_targets = []
    start_time = time.time()

    logger.info(f"Beginning inference across {len(test_loader)} batches...")
    with torch.no_grad():
        for step, batch in enumerate(test_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

            if step % 20 == 0 or step == len(test_loader):
                elapsed = time.time() - start_time
                processed = min(step * args.batch_size, len(test_dataset))
                logger.info(
                    f"Batch [{step}/{len(test_loader)}] | "
                    f"Processed {processed}/{len(test_dataset)} samples ({processed/elapsed:.1f} samples/s)"
                )

    total_eval_time = time.time() - start_time
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    # Compute statutory metrics
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    report_dict = classification_report(
        all_targets, all_preds, target_names=target_names, digits=4, output_dict=True
    )
    report_str = classification_report(
        all_targets, all_preds, target_names=target_names, digits=4
    )
    conf_mat = confusion_matrix(all_targets, all_preds).tolist()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATUTORY TEST EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"\n{report_str}")
    logger.info(f"Total Test Wall-Clock: {total_eval_time:.2f}s")
    logger.info(f"Empirical Test Macro F1: {macro_f1:.4f}")

    # Serialize results
    results_payload = {
        "model_dir": str(model_dir),
        "test_dataset": str(test_path),
        "num_samples": len(test_df),
        "total_inference_time_seconds": round(total_eval_time, 2),
        "test_macro_f1": round(macro_f1, 4),
        "test_weighted_f1": round(weighted_f1, 4),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "target_names": target_names,
    }

    report_json_path = output_dir / "test_classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"Saved full JSON metrics report to {report_json_path}")


if __name__ == "__main__":
    main()