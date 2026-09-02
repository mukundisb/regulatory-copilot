"""
openFDA MAUDE API Client
Ingests Medical Device Adverse Event (MAUDE) reports from the openFDA API.

Docs: https://open.fda.gov/apis/device/event/
"""

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd

# Load environment variables from root .env if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov/device/event.json")

# Severity label mapping based on MAUDE event_type field
EVENT_TYPE_SEVERITY = {
    "Death": "D",
    "D": "D",
    "Injury": "I",
    "I": "I",
    "Malfunction": "M",
    "M": "M",
    "Other": "O",
    "O": "O",
    "No Answer Provided": "UNKNOWN",
    "*": "UNKNOWN",
}

# Explicit triage hierarchy: Death > Injury > Malfunction > Other
SEVERITY_RANK = {
    "D": 4,
    "I": 3,
    "M": 2,
    "O": 1,
    "UNKNOWN": 0,
}

# Reverse lookup for event_type naming (keeps column consistent)
LABEL_TO_EVENT_TYPE = {
    "D": "Death",
    "I": "Injury",
    "M": "Malfunction",
    "O": "Other",
    "UNKNOWN": "Other",
}

def _build_params(
    query: str,
    limit: int,
    skip: int,
    api_key: Optional[str] = None,
) -> dict:
    params = {
        "search": query,
        "limit": min(limit, 1000),
        "skip": skip,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def _parse_record(r: dict) -> Optional[dict]:
    """Extract and validate fields from an openFDA MAUDE record."""
    try:
        # 1. Extract narrative from mdr_text
        mdr_texts = r.get("mdr_text", [])
        narrative = " ".join(
            item.get("text", "") for item in mdr_texts if isinstance(item, dict)
        ).strip()
        if not narrative:
            return None

        # 2. Extract device details
        devices = r.get("device", [])
        device_name = ""
        if devices and isinstance(devices, list):
            d = devices[0]
            device_name = d.get("brand_name", "") or d.get("generic_name", "")

        # 3. Hierarchical Worst-Case Severity Resolution
        raw_event_type = r.get("event_type")

        # Normalize into an iterable of strings
        if isinstance(raw_event_type, list):
            event_type_candidates = raw_event_type
        elif isinstance(raw_event_type, str):
            event_type_candidates = [raw_event_type]
        else:
            event_type_candidates = ["Other"]

        # Map all candidate strings to their statutory codes ('D', 'I', 'M', 'O', 'UNKNOWN')
        mapped_labels = [
            EVENT_TYPE_SEVERITY.get(str(t).strip(), "UNKNOWN")
            for t in event_type_candidates
        ]

        # Select the label with highest severity rank (Death > Injury > Malfunction > Other)
        highest_severity_label = max(
            mapped_labels, 
            key=lambda label: SEVERITY_RANK.get(label, 0),
            default="UNKNOWN"
        )

        if highest_severity_label == "UNKNOWN":
            return None

        # Clean string representation for event_type column
        canonical_event_type = LABEL_TO_EVENT_TYPE.get(highest_severity_label, "Other")

        return {
            "report_number": r.get("report_number", ""),
            "date_received": r.get("date_received", ""),
            "event_type": canonical_event_type,
            "severity_label": highest_severity_label,
            "device_name": device_name,
            "narrative_text": narrative,
        }
    except Exception as e:
        logger.debug(f"Skipping malformed record: {e}")
        return None


def _fetch_natural(
    total: int,
    api_key: Optional[str],
    delay: float,
    page_size: int = 500,
) -> list[dict]:
    """Fetch records with pagination and rate limit resilience."""
    query = "_exists_:mdr_text"
    records = []
    skip = 0

    # Guard against openFDA window limits
    max_allowed_skip = 24000 if api_key else 4500

    while len(records) < total:
        if skip >= max_allowed_skip:
            logger.warning(
                f"Reached openFDA pagination ceiling (skip={skip}). Stopping fetch loop."
            )
            break

        remaining = total - len(records)
        fetch_limit = min(page_size, remaining)
        params = _build_params(query, fetch_limit, skip, api_key)

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 404:
                logger.warning("No more results available (404). Stopping.")
                break
            if response.status_code == 429:
                logger.warning("Rate limit exceeded (429). Backing off for 60s...")
                time.sleep(60)
                continue
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise

        results = response.json().get("results", [])
        if not results:
            break

        for r in results:
            record = _parse_record(r)
            if record:
                records.append(record)
                if len(records) >= total:
                    break

        skip += len(results)
        logger.info(f"Progress: {len(records)} / {total} valid records fetched.")
        time.sleep(delay)

    return records


def fetch_maude_records(
    total_records: Optional[int] = None,
    api_key: Optional[str] = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch natural distribution of MAUDE records."""
    if total_records is None:
        total_records = int(os.getenv("OPENFDA_LIMIT", "100"))
    if api_key is None:
        api_key = os.getenv("OPENFDA_API_KEY")

    logger.info(f"Initiating fetch for {total_records} records.")
    all_records = _fetch_natural(total_records, api_key, delay)
    df = pd.DataFrame(all_records)
    logger.info(f"Extraction complete. Retained {len(df)} structured records.")
    return df


def save_raw_data(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """Persist structured records to CSV."""
    if path is None:
        path = os.getenv("OPENFDA_OUTPUT_PATH", "data/raw/maude_raw.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Persisted data to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = fetch_maude_records()
    if not df.empty:
        print("\n--- Distribution Summary ---")
        print(df["severity_label"].value_counts())
        save_raw_data(df)