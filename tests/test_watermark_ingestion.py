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
from ingestion.fetch_maude_events import fetch_incremental_events, get_next_day, load_watermark, save_watermark


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