"""
Text preprocessing and transformation pipeline for MAUDE narrative reports.
Cleans and normalizes raw MDR narrative text for NLP ingestion and resolves
event types via statutory worst-case severity hierarchy.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd

# Load environment variables if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

from ingestion.fetch_maude_events import EVENT_TYPE_SEVERITY, SEVERITY_RANK

# Common medical abbreviation expansions
ABBREVIATION_MAP = {
    r"\bpt\b": "patient",
    r"\bpts\b": "patients",
    r"\bmd\b": "physician",
    r"\bdr\b": "doctor",
    r"\bhosp\b": "hospital",
    r"\badm\b": "admitted",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\brx\b": "prescription",
    r"\bs/p\b": "status post",
    r"\bw/\b": "with",
    r"\bh/o\b": "history of",
    r"\bc/o\b": "complaint of",
    r"\bn/v\b": "nausea vomiting",
    r"\bSOB\b": "shortness of breath",
    r"\bUNK\b": "unknown",
}

BOILERPLATE_PATTERNS = [
    r"it was reported that",
    r"the reporter stated",
    r"according to the report",
    r"this is a report",
    r"per the report",
    r"the following information was received",
    r"information has been received",
    r"no further information (is|was) available",
    r"follow.?up (is|will be) requested",
]

PLACEHOLDER_STRINGS = {
    "n/a", "none", "not provided", "unknown", "no text", "null", "none reported"
}

MIN_NARRATIVE_LENGTH = 20


def expand_abbreviations(text: str) -> str:
    """Replace common clinical abbreviations with expanded full forms."""
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove boilerplate narrative framing phrases that carry no diagnostic signal."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text(
    text: str,
    lowercase: bool = True,
    preserve_digits: bool = False,
) -> str:
    """
    Standard text normalization pipeline applied identically during training and inference.
    Preserves device codes, error designations (e.g., E-402), and dosages.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Filter trivial placeholders
    stripped = text.strip()
    if stripped.lower() in PLACEHOLDER_STRINGS:
        return ""

    # 2. Normalize whitespace characters (tabs, carriage returns, newlines)
    text = re.sub(r"[\r\n\t]+", " ", stripped)

    # 3. Strip boilerplate and expand clinical abbreviations
    text = remove_boilerplate(text)
    text = expand_abbreviations(text)

    # 4. Remove special characters (preserve alphanumeric tokens and hyphens for error codes)
    if preserve_digits:
        text = re.sub(r"[^a-zA-Z0-9\-\s]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5. Collapse duplicate whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def resolve_event_label(raw_event_type: Union[List[Any], str, None]) -> Optional[str]:
    """
    Resolves multi-label records using statutory worst-case escalation hierarchy:
    Death (D) > Injury (I) > Malfunction (M) > Other (O).
    """
    if isinstance(raw_event_type, list):
        candidates = raw_event_type
    elif isinstance(raw_event_type, str):
        candidates = [raw_event_type]
    else:
        return None

    mapped_labels = [
        EVENT_TYPE_SEVERITY.get(str(item).strip(), "UNKNOWN")
        for item in candidates
    ]

    highest_label = max(
        mapped_labels,
        key=lambda l: SEVERITY_RANK.get(l, 0),
        default="UNKNOWN",
    )

    return highest_label if highest_label != "UNKNOWN" else None


def extract_narrative(record: Dict[str, Any]) -> str:
    """Extract and aggregate narrative text from raw JSON dictionary."""
    mdr_texts = record.get("mdr_text", [])
    if isinstance(mdr_texts, list):
        chunks = [
            item.get("text", "")
            for item in mdr_texts
            if isinstance(item, dict) and item.get("text")
        ]
        return " ".join(chunks).strip()
    elif isinstance(record.get("narrative_text"), str):
        return record["narrative_text"].strip()
    return ""


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "narrative_text",
    output_col: str = "clean_text",
    label_col: Optional[str] = "event_type",
    output_label_col: str = "label",
    preserve_digits: bool = False,
) -> pd.DataFrame:
    """
    Cleans raw DataFrame text and resolves labels into statutory classes.
    Drops records falling below the minimum length threshold.
    """
    df = df.copy()

    # 1. Clean narrative text
    logger.info(f"Cleaning narrative text in column '{text_col}'...")
    df[output_col] = df[text_col].apply(
        lambda t: clean_text(t, preserve_digits=preserve_digits)
    )

    # 2. Resolve multi-label entries if label column exists
    if label_col and label_col in df.columns:
        logger.info(f"Resolving statutory labels from column '{label_col}'...")
        df[output_label_col] = df[label_col].apply(resolve_event_label)
        df = df[df[output_label_col].notna()].reset_index(drop=True)

    # 3. Filter insufficient text
    before_len = len(df)
    df = df[df[output_col].str.len() >= MIN_NARRATIVE_LENGTH].reset_index(drop=True)
    dropped = before_len - len(df)

    logger.info(f"Filtered {dropped} rows below {MIN_NARRATIVE_LENGTH} chars. Retained {len(df)} records.")
    return df


def transform_raw_json(
    input_path: str,
    output_csv: str,
    sample_csv: str,
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    End-to-end ingestion transform: Loads raw openFDA JSON, cleans narratives,
    resolves statutory labels, and writes clean CSV artifacts.
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_records = data.get("results", data) if isinstance(data, dict) else data
    logger.info(f"Loaded {len(raw_records)} records from {in_path}")

    transformed = []
    for r in raw_records:
        raw_narrative = extract_narrative(r)
        cleaned = clean_text(raw_narrative, preserve_digits=False)
        if len(cleaned) < MIN_NARRATIVE_LENGTH:
            continue

        label = resolve_event_label(r.get("event_type"))
        if not label:
            continue

        transformed.append({
            "report_number": r.get("report_number", ""),
            "narrative_text": cleaned,
            "label": label,
        })

    df = pd.DataFrame(transformed)
    logger.info(f"Successfully processed {len(df)} clean records.")

    if not df.empty:
        # Save full processed dataset (gitignored)
        out_file = Path(output_csv)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        logger.info(f"Full dataset written to {out_file}")

        # Save small sample for git tracking and inspection
        sample_file = Path(sample_csv)
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        df.head(sample_size).to_csv(sample_file, index=False)
        logger.info(f"Verification sample ({sample_size} rows) written to {sample_file}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    raw_csv_path = os.getenv("OPENFDA_RAW_CSV", "data/raw/maude_raw.csv")
    processed_csv_path = os.getenv("OPENFDA_PROCESSED_OUTPUT", "data/processed/maude_train.csv")
    sample_csv_path = os.getenv("OPENFDA_SAMPLE_OUTPUT", "data/samples/transformed_sample.csv")
    sample_size = int(os.getenv("TRANSFORM_SAMPLE_SIZE", "5"))

    in_file = Path(raw_csv_path)
    if not in_file.exists():
        logger.error(f"Input file not found at: {in_file}. Run fetch_maude_events.py first.")
        raise FileNotFoundError(f"Input CSV not found at {in_file}")

    # 1. Read raw CSV
    logger.info(f"Loading raw records from {in_file}...")
    raw_df = pd.read_csv(in_file)
    logger.info(f"Loaded {len(raw_df)} records.")

    # 2. Clean narrative text and resolve statutory labels
    clean_df = clean_dataframe(
        raw_df,
        text_col="narrative_text",
        output_col="narrative_text",
        label_col="event_type",
        output_label_col="label",
        preserve_digits=False,
    )

    # Retain standard training schema
    columns_to_keep = [c for c in ["report_number", "narrative_text", "label"] if c in clean_df.columns]
    final_df = clean_df[columns_to_keep]

    # 3. Write full processed dataset (gitignored)
    out_file = Path(processed_csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_file, index=False)
    logger.info(f"Saved {len(final_df)} processed records to {out_file}")

    # 4. Write small git-tracked sample
    sample_file = Path(sample_csv_path)
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.head(sample_size).to_csv(sample_file, index=False)
    logger.info(f"Saved {min(sample_size, len(final_df))} sample records to {sample_file}")

    print("\n--- Final Processed Label Distribution ---")
    if "label" in final_df.columns:
        print(final_df["label"].value_counts())