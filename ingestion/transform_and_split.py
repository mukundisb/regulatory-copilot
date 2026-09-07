"""
ingestion/transform_and_split.py

Transforms raw openFDA CSV records into clean (narrative, label) pairs.
Reuses maude_classifier.text_cleaner.clean_text to ensure training matches
runtime /classify preprocessing exactly.
"""

import sys
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# CANONICAL SOURCES OF TRUTH
from maude_classifier.text_cleaner import clean_text
from ingestion.fetch_maude_events import resolve_severity_label, SEVERITY_RANK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("maude_transformer")

VALID_LABELS = {k for k in SEVERITY_RANK.keys() if k != "UNKNOWN"}

"""
ingestion/transform_and_split.py

Transforms raw openFDA records into clean (narrative, label) pairs.
Reuses maude_classifier.text_cleaner.clean_text and
ingestion.fetch_maude_events.resolve_severity_label as canonical single sources of truth.
Exports stratified parquet splits and writes reports/dataset_split_metrics.json.
"""

import sys
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import argparse
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split

# CANONICAL SOURCES OF TRUTH
from maude_classifier.text_cleaner import clean_text
from ingestion.fetch_maude_events import resolve_severity_label, SEVERITY_RANK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("maude_transformer")

VALID_LABELS = {k for k in SEVERITY_RANK.keys() if k != "UNKNOWN"}


def process_csv_dataset(
    input_csv: Path,
    output_dir: Path,
    report_path: Path = Path("reports/dataset_split_metrics.json"),
    seed: int = 42
):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading raw dataset from {input_csv}...")

    # Detect format: check if lines are JSON Lines
    with open(input_csv, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if first_line.startswith("{"):
        logger.info("Detected JSON Lines formatting. Loading with lines=True...")
        df = pd.read_json(input_csv, lines=True)
    else:
        logger.info("Detected standard CSV formatting. Loading with read_csv...")
        try:
            df = pd.read_csv(input_csv, engine="python", on_bad_lines="skip", encoding="utf-8")
        except Exception:
            import csv
            df = pd.read_csv(input_csv, engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip", encoding="utf-8")

    raw_record_count = len(df)
    logger.info(f"Loaded {raw_record_count:,} records. Columns found: {list(df.columns)}")

    # Detect text column
    text_col = None
    for candidate in ["narrative_text", "narrative", "text", "event_description"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if not text_col:
        raise ValueError(f"Could not locate narrative column. Available: {list(df.columns)}")

    # Detect label column (severity_label preferred, falls back to event_type)
    label_col = None
    for candidate in ["severity_label", "event_type", "label"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if not label_col:
        raise ValueError(f"Could not locate label column. Available: {list(df.columns)}")

    logger.info(f"Using text column: '{text_col}', label column: '{label_col}'")

    # Step 1: Filter nulls and resolve statutory labels using canonical shared resolver
    df = df.dropna(subset=[text_col, label_col]).copy()
    logger.info("Validating/resolving statutory labels via shared resolve_severity_label...")
    df["label"] = df[label_col].apply(resolve_severity_label)
    df = df[df["label"].isin(VALID_LABELS)].copy()
    valid_labels_count = len(df)
    logger.info(f"Retained {valid_labels_count:,} records with valid statutory labels.")

    # Step 2: Clean narratives using production cleaner
    logger.info("Executing text cleaning via maude_classifier.text_cleaner...")
    df["narrative"] = df[text_col].astype(str).apply(clean_text)

    # Filter out empty or trivially short cleaned narratives (<20 chars)
    df = df[df["narrative"].str.len() >= 20].copy()
    sufficient_text_count = len(df)
    logger.info(f"Retained {sufficient_text_count:,} records meeting minimum text length.")

    # Step 3: Deduplicate on cleaned narrative to eliminate template leakage across splits
    initial_unique_candidates = len(df)
    df = df.drop_duplicates(subset=["narrative"]).reset_index(drop=True)
    deduplicated_count = len(df)
    duplicates_removed = initial_unique_candidates - deduplicated_count
    logger.info(
        f"Deduplication complete: removed {duplicates_removed:,} duplicate texts. "
        f"Remaining unique events: {deduplicated_count:,}"
    )

    # Label distribution summaries
    total_counts = df["label"].value_counts().to_dict()
    total_proportions = df["label"].value_counts(normalize=True).round(6).to_dict()

    logger.info("Class Distribution (Normalized):\n" + df["label"].value_counts(normalize=True).to_string())
    logger.info("Class Distribution (Counts):\n" + df["label"].value_counts().to_string())

    # Retain final training columns
    report_id_col = "report_number" if "report_number" in df.columns else None
    cols_to_keep = ["narrative", "label"]
    if report_id_col:
        cols_to_keep.insert(0, report_id_col)
    df = df[cols_to_keep]

    # Step 4: Stratified Split: 80% Train, 10% Validation, 10% Test
    logger.info("Splitting dataset into stratified 80/10/10 partitions...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )

    logger.info(
        f"Partition Sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    # Step 5: Export Parquet Partitions
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved Parquet partitions to: {output_dir}")

    # Step 6: Write Structured Metrics Report
    split_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "input_source": str(input_csv),
        "hygiene_pipeline": {
            "raw_records_loaded": raw_record_count,
            "valid_statutory_labels": valid_labels_count,
            "sufficient_text_retained": sufficient_text_count,
            "duplicate_narratives_removed": duplicates_removed,
            "clean_unique_records_retained": deduplicated_count,
        },
        "overall_distribution": {
            "counts": total_counts,
            "proportions": total_proportions,
        },
        "partitions": {
            "train": {
                "count": len(train_df),
                "ratio": round(len(train_df) / deduplicated_count, 4),
                "counts_by_label": train_df["label"].value_counts().to_dict(),
                "proportions_by_label": train_df["label"].value_counts(normalize=True).round(6).to_dict(),
                "path": str(train_path),
            },
            "validation": {
                "count": len(val_df),
                "ratio": round(len(val_df) / deduplicated_count, 4),
                "counts_by_label": val_df["label"].value_counts().to_dict(),
                "proportions_by_label": val_df["label"].value_counts(normalize=True).round(6).to_dict(),
                "path": str(val_path),
            },
            "test": {
                "count": len(test_df),
                "ratio": round(len(test_df) / deduplicated_count, 4),
                "counts_by_label": test_df["label"].value_counts().to_dict(),
                "proportions_by_label": test_df["label"].value_counts(normalize=True).round(6).to_dict(),
                "path": str(test_path),
            },
        },
    }

    with open(report_path, "w", encoding="utf-8") as f_rep:
        json.dump(split_metrics, f_rep, indent=2)

    logger.info(f"Dataset split metrics report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform raw MAUDE CSV and produce stratified Parquet splits.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/maude_raw.csv"), help="Path to input CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_split_metrics.json"), help="Output metrics path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    process_csv_dataset(args.input, args.output_dir, report_path=args.report, seed=args.seed)