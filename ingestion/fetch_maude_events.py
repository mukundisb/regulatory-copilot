"""
openFDA MAUDE API Client
Incremental ingestion script for openFDA MAUDE adverse event reports.
Tracks high-water mark via GCS or local JSON to pull only delta records.

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
from typing import Optional, Union, List, Any, Set, Dict
from datetime import datetime, timezone, timedelta

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
DEFAULT_WATERMARK_FILE = "data/watermark.json"

def sanitize_api_key(raw_key: str | None) -> str | None:
    """Strip whitespace, surrounding quotes, and UTF-8 Byte Order Marks (BOM)."""
    if not raw_key:
        return None
    # Strip UTF-8 BOM characters (\ufeff) and general whitespace/quotes
    cleaned = raw_key.strip().lstrip("\ufeff").strip("\"'")
    return cleaned if cleaned else None

def get_openfda_api_key() -> str | None:
    raw_key = os.environ.get("OPENFDA_API_KEY")
    return sanitize_api_key(raw_key)

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

def get_next_day(date_str: str) -> str:
    """Advances YYYYMMDD string by 1 day to ensure strictly exclusive lower bound."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    return (dt + timedelta(days=1)).strftime("%Y%m%d")


def load_watermark(watermark_path: str) -> Optional[str]:
    """Loads the last ingested date_received (YYYYMMDD)."""
    if watermark_path.startswith("gs://"):
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket_name, blob_name = watermark_path.replace("gs://", "").split("/", 1)
            blob = client.bucket(bucket_name).blob(blob_name)
            if blob.exists():
                data = json.loads(blob.download_as_text())
                return data.get("last_date_received")
        except Exception as e:
            logger.warning("Could not load GCS watermark from %s: %s", watermark_path, e)
            return None
    else:
        local_file = Path(watermark_path)
        if local_file.exists():
            with open(local_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_date_received")
    return None


def save_watermark(watermark_path: str, last_date_received: str, record_count: int):
    """Persists updated high-water mark timestamp."""
    payload = {
        "last_date_received": last_date_received,
        "record_count_total": record_count,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if watermark_path.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_name = watermark_path.replace("gs://", "").split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2))
        logger.info("Persisted GCS watermark to %s", watermark_path)
    else:
        local_file = Path(watermark_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Persisted local watermark to %s", watermark_path)


def fetch_incremental_events(
    watermark_path: str = DEFAULT_WATERMARK_FILE,
    output_dir: str = "data/raw/incremental",
    api_key: Optional[str] = None,
    limit_batches: int = 5,
    existing_seen_ids: Optional[Set[str]] = None,
) -> int:
    last_date = load_watermark(watermark_path)
    current_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    # If watermark exists, strictly advance lower bound by 1 calendar day to prevent re-querying processed date
    if last_date:
        start_date = get_next_day(last_date)
        if start_date > current_date:
            logger.info("Watermark %s is already up to date with %s. No delta pull required.", last_date, current_date)
            return 0
        logger.info("Found watermark %s. Querying delta window: [%s TO %s]", last_date, start_date, current_date)
        search_query = f"date_received:[{start_date} TO {current_date}]"
    else:
        start_date = "20240101"
        logger.info("No prior watermark detected. Fetching baseline from %s...", start_date)
        search_query = f"date_received:[{start_date} TO {current_date}]"

    # Route through sanitize_api_key defensively
    clean_api_key = sanitize_api_key(api_key or os.environ.get("OPENFDA_API_KEY"))

    params = {
        "search": search_query,
        "limit": 100,
        "sort": "date_received:asc",
    }
    if clean_api_key:
        params["api_key"] = clean_api_key

    seen_ids: Set[str] = set(existing_seen_ids) if existing_seen_ids else set()
    records_fetched = 0
    max_seen_date = last_date or start_date
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(limit_batches):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
            
            # openFDA returns 404 when query matches 0 records (genuinely caught up)
            if resp.status_code == 404:
                logger.info("No records returned for search range: %s", search_query)
                break
                
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])
            if not raw_results:
                break

            # Deduplicate by report_number
            unique_batch: List[Dict] = []
            for r in raw_results:
                r_id = r.get("report_number")
                if r_id and r_id in seen_ids:
                    continue
                if r_id:
                    seen_ids.add(r_id)
                unique_batch.append(r)

                d = r.get("date_received")
                if d and d > max_seen_date:
                    max_seen_date = d

            if unique_batch:
                records_fetched += len(unique_batch)
                batch_file = out_path / f"maude_delta_{max_seen_date}_{batch_idx}.json"
                with open(batch_file, "w", encoding="utf-8") as f:
                    json.dump(unique_batch, f)

            if len(raw_results) < 100:
                break

            # Advance window forward if date progressed, or step +1 day past current max_seen_date
            next_start = get_next_day(max_seen_date)
            if next_start > current_date:
                break
            params["search"] = f"date_received:[{next_start} TO {current_date}]"
            time.sleep(0.5)

        # Re-raise real HTTP/network errors so job exits non-zero
        except requests.exceptions.HTTPError as err:
            logger.error("Fatal HTTP error during openFDA batch fetch: %s", err)
            raise RuntimeError(f"openFDA fetch aborted due to HTTP {resp.status_code}") from err
        except requests.exceptions.RequestException as err:
            logger.error("Network/connection error during openFDA batch fetch: %s", err)
            raise RuntimeError("openFDA fetch aborted due to network failure") from err

    if records_fetched > 0:
        save_watermark(watermark_path, max_seen_date, records_fetched)
    
    logger.info("Incremental fetch finished. New records: %d | New High-Water Mark: %s", records_fetched, max_seen_date)
    return records_fetched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="openFDA MAUDE ingestion supporting incremental watermark tracking and full baseline pulls."
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Run incremental delta pull via watermark (default) or execute full historical crawl.",
    )
    # Incremental options
    parser.add_argument(
        "--watermark-path", type=str, default=DEFAULT_WATERMARK_FILE
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/incremental"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Max batches to pull for incremental runs",
    )
    # Full extraction options
    parser.add_argument(
        "--target",
        type=int,
        default=int(os.getenv("OPENFDA_LIMIT", str(DEFAULT_TARGET_RECORDS))),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("OPENFDA_OUTPUT_PATH", "data/raw/maude_raw.jsonl")),
    )
    parser.add_argument(
        "--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", None)
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1 if os.getenv("OPENFDA_API_KEY") else 0.3,
    )
    args = parser.parse_args()

    if args.mode == "full":
        fetch_maude_dataset(
            target_count=args.target,
            output_path=args.output,
            api_key=args.api_key,
            delay=args.delay,
        )
    else:
        fetch_incremental_events(
            watermark_path=args.watermark_path,
            output_dir=args.output_dir,
            limit_batches=args.batches,
            api_key=args.api_key,
        )