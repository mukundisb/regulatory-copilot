"""
openFDA MAUDE API Client
Ingests Medical Device Adverse Event (MAUDE) reports from the openFDA API
using search_after cursor pagination to bypass the 26,000 skip limit.

Docs: https://open.fda.gov/apis/paging/
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Union, List, Any

import requests
import pandas as pd

# Load environment variables from root .env if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("maude_fetcher")

BASE_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov/device/event.json")
DEFAULT_TARGET_RECORDS = 160_000
PAGE_LIMIT = 1000  # openFDA maximum batch size

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

# Reverse lookup for event_type naming
LABEL_TO_EVENT_TYPE = {
    "D": "Death",
    "I": "Injury",
    "M": "Malfunction",
    "O": "Other",
    "UNKNOWN": "Other",
}


def resolve_severity_label(raw_event_type: Union[List[Any], str, None]) -> Optional[str]:
    """
    Resolves raw event_type (single string, list of strings, or None) into a
    statutory severity label ('D', 'I', 'M', 'O') using the worst-case escalation
    hierarchy: Death (D) > Injury (I) > Malfunction (M) > Other (O).
    Returns None if no candidate maps to a valid statutory label.
    """
    if raw_event_type is None:
        return None

    if isinstance(raw_event_type, list):
        event_type_candidates = raw_event_type
    elif isinstance(raw_event_type, str):
        event_type_candidates = [raw_event_type]
    else:
        event_type_candidates = [str(raw_event_type)]

    mapped_labels = [
        EVENT_TYPE_SEVERITY.get(str(t).strip(), "UNKNOWN")
        for t in event_type_candidates
    ]

    highest_severity_label = max(
        mapped_labels,
        key=lambda label: SEVERITY_RANK.get(label, 0),
        default="UNKNOWN"
    )

    return highest_severity_label if highest_severity_label != "UNKNOWN" else None


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

        # 3. Canonical Hierarchical Severity Resolution (D > I > M > O)
        highest_severity_label = resolve_severity_label(r.get("event_type"))
        if not highest_severity_label:
            return None

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


def extract_next_url(link_header: Optional[str]) -> Optional[str]:
    """
    Parses openFDA HTTP Link header:
    <https://api.fda.gov/device/event.json?...search_after=...>; rel="next"
    """
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def fetch_maude_dataset(
    target_count: int,
    output_path: Path,
    api_key: Optional[str] = None,
    delay: float = 0.1,
) -> int:
    """
    Fetches records using search_after cursor pagination, streaming results directly
    to a JSONL file to prevent memory exhaustion.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initial query requirements for search_after:
    # 1. Do NOT include 'skip'
    # 2. Must include 'sort' parameter
    params = {
        "search": "_exists_:mdr_text",
        "limit": PAGE_LIMIT,
        "sort": "date_received:asc",
    }
    if api_key:
        params["api_key"] = api_key

    next_url: Optional[str] = BASE_URL
    total_valid = 0
    batch_num = 0

    logger.info(f"Initiating openFDA cursor extraction. Target: {target_count:,} valid records.")
    logger.info(f"Writing parsed records to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        while total_valid < target_count and next_url:
            batch_num += 1
            start_batch = time.perf_counter()

            try:
                if next_url == BASE_URL:
                    response = requests.get(BASE_URL, params=params, timeout=30)
                else:
                    # Subsequent cursor pages use the full extracted next URL
                    response = requests.get(next_url, timeout=30)

                if response.status_code == 429:
                    logger.warning("Rate limit reached (429). Sleeping for 30s...")
                    time.sleep(30)
                    continue

                if response.status_code == 404:
                    logger.info("No further records available (404). Cursor pagination complete.")
                    break

                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    logger.info("Empty results array. Extraction complete.")
                    break

                # Parse and stream directly to disk
                batch_valid = 0
                for raw_item in results:
                    parsed = _parse_record(raw_item)
                    if parsed:
                        f_out.write(json.dumps(parsed) + "\n")
                        batch_valid += 1
                        total_valid += 1
                        if total_valid >= target_count:
                            break

                batch_latency_ms = (time.perf_counter() - start_batch) * 1000

                # Extract cursor for next batch
                link_header = response.headers.get("Link")
                next_url = extract_next_url(link_header)

                # Ensure api_key is retained on the cursor URL if needed
                if next_url and api_key and "api_key=" not in next_url:
                    delimiter = "&" if "?" in next_url else "?"
                    next_url = f"{next_url}{delimiter}api_key={api_key}"

                logger.info(
                    f"Batch {batch_num:03d} | Valid: +{batch_valid:,} | "
                    f"Progress: {total_valid:,}/{target_count:,} ({(total_valid/target_count)*100:.1f}%) | "
                    f"Batch Time: {batch_latency_ms:.0f}ms"
                )

                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error on batch {batch_num}: {e}. Retrying in 5s...")
                time.sleep(5)

    logger.info(f"Extraction complete. Successfully wrote {total_valid:,} records to {output_path}")
    return total_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch openFDA MAUDE records via search_after cursor.")
    parser.add_argument("--target", type=int, default=int(os.getenv("OPENFDA_LIMIT", str(DEFAULT_TARGET_RECORDS))))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("OPENFDA_OUTPUT_PATH", "data/raw/maude_raw.jsonl")))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", None))
    parser.add_argument("--delay", type=float, default=0.1 if os.getenv("OPENFDA_API_KEY") else 0.3)
    args = parser.parse_args()

    fetch_maude_dataset(
        target_count=args.target,
        output_path=args.output,
        api_key=args.api_key,
        delay=args.delay,
    )