"""
scripts/measure_live.py reads both log exports: Google Cloud Logging and Azure Log Analytics.
"""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("measure_live", REPO_ROOT / "scripts" / "measure_live.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

EVENT = {"event": "assess_request", "outcome": "ok", "latency_ms": 5000.0, "llm_provider": "azure_openai"}


def test_google_cloud_logging_export(tmp_path):
    path = tmp_path / "gcp.json"
    path.write_text(json.dumps([
        {"timestamp": "2026-11-15T10:00:01.123456789Z", "jsonPayload": EVENT},
        {"timestamp": "2026-11-15T10:00:02Z", "jsonPayload": {"event": "other"}},
    ]))
    events = ml.load_log_entries(path)
    assert len(events) == 1 and events[0]["latency_ms"] == 5000.0
    assert events[0]["timestamp"].isoformat() == "2026-11-15T10:00:01.123456+00:00"


def test_azure_log_analytics_export(tmp_path):
    path = tmp_path / "azure.json"
    path.write_text(json.dumps([
        {"TimeGenerated": "2026-12-15T10:00:02.1234567Z", "Log_s": json.dumps(EVENT)},
        {"TimeGenerated": "2026-12-15T10:00:01Z", "Log_s": "INFO: 10.0.0.1 - POST /assess 200"},
        {"TimeGenerated": "2026-12-15T10:00:03Z", "Log": "stdout F " + json.dumps({**EVENT, "latency_ms": 7000.0})},
        {"TimeGenerated": "2026-12-15T10:00:04Z", "Log_s": "assess_request outcome=ok {not json"},
    ]))
    events = ml.load_log_entries(path)
    assert [e["latency_ms"] for e in events] == [5000.0, 7000.0]
    assert events[0]["llm_provider"] == "azure_openai"


def test_tagged_report_files_do_not_overwrite_the_cloud_run_ones():
    assert ml.run_file("").name == "live_run.json" and ml.metrics_file("").name == "live_metrics.json"
    assert ml.run_file("azure").name == "live_run_azure.json"
    assert ml.metrics_file("azure").name == "live_metrics_azure.json"
