"""
tests/test_watermark_ingestion.py

Verifies that:
1. Watermark advances on new records.
2. Subsequent runs with identical watermark do not re-pull existing records.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from ingestion.fetch_maude_events import fetch_incremental_events, get_next_day, load_watermark, save_watermark, sanitize_api_key


def test_get_next_day():
    assert get_next_day("20240101") == "20240102"
    assert get_next_day("20241231") == "20250101"
    assert get_next_day("20240228") == "20240229"  # 2024 leap year


def test_watermark_roundtrip(tmp_path):
    wm_file = str(tmp_path / "watermark.json")
    save_watermark(wm_file, "20240315", 50)
    assert load_watermark(wm_file) == "20240315"


@patch("ingestion.fetch_maude_events.requests.get")
def test_incremental_fetch_advances_date_and_skips_watermark_date(mock_get, tmp_path):
    """Verifies search query lower bound strictly advances +1 day past watermark."""
    wm_file = str(tmp_path / "watermark.json")
    out_dir = str(tmp_path / "output")
    save_watermark(wm_file, "20240101", 100)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"report_number": "R-101", "date_received": "20240102", "event_type": "Injury"}
        ]
    }
    mock_get.return_value = mock_resp

    fetch_incremental_events(watermark_path=wm_file, output_dir=out_dir, limit_batches=1)

    # Check search param passed to requests.get
    called_params = mock_get.call_args[1]["params"]
    assert "date_received:[20240102 TO " in called_params["search"]
    assert load_watermark(wm_file) == "20240102"


@patch("ingestion.fetch_maude_events.requests.get")
def test_id_deduplication_prevents_duplicate_records(mock_get, tmp_path):
    """Verifies that duplicate report_numbers in the response stream are discarded."""
    wm_file = str(tmp_path / "watermark.json")
    out_dir = str(tmp_path / "output")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"report_number": "DUP-1", "date_received": "20240105", "event_type": "Malfunction"},
            {"report_number": "DUP-1", "date_received": "20240105", "event_type": "Malfunction"},  # Duplicate
            {"report_number": "NEW-2", "date_received": "20240106", "event_type": "Injury"},
        ]
    }
    mock_get.return_value = mock_resp

    pulled = fetch_incremental_events(
        watermark_path=wm_file,
        output_dir=out_dir,
        limit_batches=1,
        existing_seen_ids={"DUP-1"},  # DUP-1 was already ingested prior
    )

    # Only NEW-2 should be counted and saved
    assert pulled == 1
    assert load_watermark(wm_file) == "20240106"

    saved_batch = list(Path(out_dir).glob("*.json"))[0]
    with open(saved_batch, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 1
    assert records[0]["report_number"] == "NEW-2"

def test_sanitize_api_key_strips_bom_quotes_and_whitespace():
    """Guards against BOM corruption (e.g. Windows Out-File/Set-Content) and messy env formatting."""
    # Byte-order mark prefix with trailing whitespace and newlines
    dirty_bom = "\ufeffmy-fda-key-12345 \n"
    assert sanitize_api_key(dirty_bom) == "my-fda-key-12345"

    # Accidental wrapping quotes
    quoted = '"quoted-api-key"'
    assert sanitize_api_key(quoted) == "quoted-api-key"

    # Both BOM and quotes together
    bom_and_quoted = "\ufeff'secret-key-abc'\t"
    assert sanitize_api_key(bom_and_quoted) == "secret-key-abc"

    # None and empty strings return cleanly
    assert sanitize_api_key(None) is None
    assert sanitize_api_key("   ") is None


@patch("ingestion.fetch_maude_events.requests.get")
def test_fetch_incremental_events_raises_on_http_error(mock_get, tmp_path):
    """Guards against silent green executions when openFDA returns 401/403/500."""
    wm_file = str(tmp_path / "watermark.json")
    out_dir = str(tmp_path / "output")

    # Set up a prior watermark so the delta window logic activates
    with open(wm_file, "w", encoding="utf-8") as f:
        f.write('{"last_date_received": "20240101", "record_count_total": 50}')

    # Mock an openFDA 403 Forbidden response
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "403 Client Error: Forbidden for url"
    )
    mock_get.return_value = mock_resp

    # Verify that HTTPError is not swallowed and actually raises RuntimeError
    with pytest.raises(RuntimeError, match="openFDA fetch aborted due to HTTP 403"):
        fetch_incremental_events(
            watermark_path=wm_file,
            output_dir=out_dir,
            api_key="test-key",
            limit_batches=1,
        )