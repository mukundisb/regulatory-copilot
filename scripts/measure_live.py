"""
Measure latency, cost and error rate of the deployed /assess endpoint.

Two steps, because per-stage timings and cost are written to Cloud Logging by the server
(telemetry.py) and are not part of the API response.

Step 1 - send traffic:
    python scripts/measure_live.py send
  Sends one warm-up request (to wake the Vertex AI endpoint), then every golden-set item,
  one at a time. Writes reports/live_run.json and prints the Logs Explorer query to use next.

Step 2 - summarise the server-side telemetry:
    python scripts/measure_live.py summarise --logs path/to/downloaded-logs.json
  Reads the log entries downloaded from Logs Explorer ("Download" > JSON), keeps the ones from
  the run, and writes reports/live_metrics.json with median (p50) and 95th percentile (p95)
  latency per stage, cost per request and the outcome counts.

Percentiles use linear interpolation between the two nearest ranks (the same method as
numpy.percentile's default).
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET = REPO_ROOT / "eval" / "golden_set.json"


def run_file(tag: str) -> Path:
    return REPO_ROOT / "reports" / (f"live_run_{tag}.json" if tag else "live_run.json")


def metrics_file(tag: str) -> Path:
    return REPO_ROOT / "reports" / (f"live_metrics_{tag}.json" if tag else "live_metrics.json")


DEFAULT_URL = "https://regulatory-copilot-679089218302.asia-south1.run.app"
WARMUP_NARRATIVE = (
    "Patient with an implanted cardiac defibrillator collapsed at home. The device failed to "
    "deliver therapy during ventricular fibrillation and the patient died before emergency services arrived."
)
# The Vertex AI endpoint scales from zero; the API waits up to 210 s for it.
REQUEST_TIMEOUT_SECONDS = 300
STAGES = ("latency_ms", "classify_ms", "retrieve_ms", "generate_ms")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str) -> datetime:
    # Cloud Logging timestamps carry nanoseconds and a trailing Z; keep microseconds only.
    value = value.replace("Z", "+00:00")
    if "." in value:
        head, rest = value.split(".", 1)
        n_digits = len(rest) - len(rest.lstrip("0123456789"))
        value = f"{head}.{rest[:n_digits][:6]}{rest[n_digits:]}"
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# --------------------------------------------------------------------------- step 1: send

def call_assess(url: str, narrative: str) -> dict:
    started = time.perf_counter()
    try:
        response = requests.post(f"{url}/assess", json={"narrative": narrative}, timeout=REQUEST_TIMEOUT_SECONDS)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "http_status": response.status_code,
            "client_latency_ms": elapsed_ms,
            "backend_used": body.get("backend_used"),
            "llm_grounding_verified": body.get("llm_grounding_verified"),
            "citations": body.get("citations"),
            "warning": body.get("warning"),
        }
    except requests.RequestException as err:
        return {
            "http_status": None,
            "client_latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(err).__name__}: {err}"[:300],
        }


def send(args) -> int:
    url = args.url.rstrip("/")
    items = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]

    # Cloud Run scales to zero. The first call may have to wait for a container to start
    # and load its models, so the health check is given the same budget as a request.
    print("Health check (can take a few minutes if the service is starting from zero)...")
    health_started = time.perf_counter()
    try:
        health = requests.get(f"{url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as err:
        print(f"[STOP] Health check failed: {type(err).__name__}. Nothing was sent. "
              "Wait a minute and run the command again.")
        return 2
    print(f"Health check: HTTP {health.status_code} after {time.perf_counter() - health_started:.0f} s")
    if health.status_code != 200:
        print("[STOP] The service is not healthy. Nothing was sent.")
        return 2

    started_at = utc_now()
    print(f"Run started {started_at}. Sending 1 warm-up request, then {len(items)} items to {url}/assess")
    print("The warm-up can take a few minutes if the Vertex AI endpoint is starting from zero.")
    warmup = call_assess(url, WARMUP_NARRATIVE)
    print(f"  warm-up   HTTP {warmup['http_status']}  {warmup['client_latency_ms']:.0f} ms")

    results = []
    for number, item in enumerate(items, 1):
        result = {"id": item["id"], "type": item["type"], **call_assess(url, item["query"])}
        results.append(result)
        print(f"  {number:>2}/{len(items)} {item['id']}  HTTP {result['http_status']}  {result['client_latency_ms']:.0f} ms")
        time.sleep(args.pause)
    finished_at = utc_now()

    out = run_file(args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "url": url, "started_at": started_at, "finished_at": finished_at,
        "warmup": warmup, "requests": results,
    }, indent=2), encoding="utf-8")

    failed = sum(1 for r in results if r["http_status"] != 200)
    print(f"\nDone. {len(results)} requests, {failed} without HTTP 200. Saved {out}")
    end = (parse_time(finished_at) + timedelta(minutes=1)).isoformat(timespec="seconds")
    tag_arg = f" --tag {args.tag}" if args.tag else ""
    if ".run.app" in url:
        print("\nNext: wait about two minutes, open Logs Explorer and run this query:\n")
        print('  jsonPayload.event="assess_request"')
        print(f'  timestamp>="{started_at}"')
        print(f'  timestamp<="{end}"')
        print("\nThen choose Actions > Download logs > JSON, and run:")
    else:
        print("\nNext: wait about five minutes for Log Analytics, then export the entries with the Azure CLI:\n")
        print('  az monitor log-analytics query --workspace <workspace customer id> --output json --analytics-query \\')
        print(f'    "ContainerAppConsoleLogs_CL | where TimeGenerated between (datetime({started_at}) .. datetime({end})) '
              f'| where Log_s has \'assess_request\' | project TimeGenerated, Log_s" > azure-logs.json')
        print("\nThen run:")
    print(f"  python scripts/measure_live.py summarise --logs <downloaded file>{tag_arg}")
    return 0


# --------------------------------------------------------------------------- step 2: summarise

def load_log_entries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("["):
        entries = json.loads(text)
    else:  # one JSON object per line
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    events = []
    for entry in entries:
        if "jsonPayload" in entry:  # Google Cloud Logging export
            payload, timestamp = entry.get("jsonPayload") or {}, entry.get("timestamp")
        else:  # Azure Log Analytics row: the container's stdout line is a string column
            raw = entry.get("Log_s") or entry.get("Log") or ""
            if "assess_request" not in raw or "{" not in raw:
                continue
            try:
                payload = json.loads(raw[raw.index("{"):])
            except ValueError:
                continue
            timestamp = entry.get("TimeGenerated") or entry.get("TimeGenerated [UTC]")
        if payload.get("event") == "assess_request" and timestamp:
            events.append({"timestamp": parse_time(timestamp), **payload})
    return sorted(events, key=lambda e: e["timestamp"])


def stage_stats(events: list[dict]) -> dict:
    stats = {}
    for stage in STAGES:
        values = [e[stage] for e in events if isinstance(e.get(stage), (int, float))]
        stats[stage] = {
            "p50": round(percentile(values, 50), 1) if values else None,
            "p95": round(percentile(values, 95), 1) if values else None,
            "max": round(max(values), 1) if values else None,
        }
    return stats


def summarise(args) -> int:
    events = load_log_entries(Path(args.logs))
    if run_file(args.tag).exists() and not args.all_entries:
        run = json.loads(run_file(args.tag).read_text(encoding="utf-8"))
        start = parse_time(run["started_at"])
        end = parse_time(run["finished_at"]) + timedelta(minutes=1)
        events = [e for e in events if start <= e["timestamp"] <= end]
    if not events:
        print("[ERROR] No assess_request entries found for this run in the log file.")
        return 2

    # The first request of the run is the warm-up; it is reported on its own.
    warmup, measured = events[0], events[1:]
    if not measured:
        print("[ERROR] Only the warm-up request was found.")
        return 2

    outcomes = {name: sum(1 for e in measured if e.get("outcome") == name) for name in ("ok", "degraded", "error")}
    llm_status = {}
    for e in measured:
        key = e.get("llm_status") or "unknown"
        llm_status[key] = llm_status.get(key, 0) + 1
    costs = [e["est_cost_usd"] for e in measured if isinstance(e.get("est_cost_usd"), (int, float))]
    in_tokens = [e["input_tokens"] for e in measured if isinstance(e.get("input_tokens"), int)]
    out_tokens = [e["output_tokens"] for e in measured if isinstance(e.get("output_tokens"), int)]
    # A request that waited on a classifier cold start, even after the warm-up.
    cold = [e for e in measured if isinstance(e.get("classify_ms"), (int, float)) and e["classify_ms"] > 10_000]

    summary = {
        "requests_measured": len(measured),
        "window": {"first": measured[0]["timestamp"].isoformat(), "last": measured[-1]["timestamp"].isoformat()},
        "warmup_request": {k: warmup.get(k) for k in (*STAGES, "outcome", "classifier_backend")},
        "latency_ms": stage_stats(measured),
        "latency_ms_ok_requests_only": stage_stats([e for e in measured if e.get("outcome") == "ok"]),
        "outcomes": outcomes,
        "error_rate": round((outcomes["degraded"] + outcomes["error"]) / len(measured), 3),
        "llm_status": llm_status,
        "classifier_fallbacks": sum(1 for e in measured if e.get("classifier_fallback")),
        "classifier_cold_starts_over_10s": len(cold),
        "cost_usd": {
            "requests_with_llm_call": len(costs),
            "mean": round(statistics.mean(costs), 6) if costs else None,
            "p50": round(percentile(costs, 50), 6) if costs else None,
            "p95": round(percentile(costs, 95), 6) if costs else None,
            "per_1000_requests": round(statistics.mean(costs) * 1000, 2) if costs else None,
        },
        "tokens": {
            "mean_input": round(statistics.mean(in_tokens)) if in_tokens else None,
            "mean_output": round(statistics.mean(out_tokens)) if out_tokens else None,
        },
    }
    metrics_file(args.tag).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nLive /assess metrics over {len(measured)} requests (warm-up excluded)")
    print(f"{'stage':<14}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}")
    for stage, s in summary["latency_ms"].items():
        print(f"{stage:<14}{s['p50']:>10}{s['p95']:>10}{s['max']:>10}")
    ok_total = summary["latency_ms_ok_requests_only"]["latency_ms"]
    print(f"Requests with outcome ok only: total latency p50 {ok_total['p50']} ms, p95 {ok_total['p95']} ms")
    print(f"\nWarm-up request: {warmup.get('latency_ms')} ms total, {warmup.get('classify_ms')} ms in classify")
    print(f"Outcomes: {outcomes}   error rate (degraded + error): {summary['error_rate']}")
    print(f"LLM status: {llm_status}   classifier fallbacks: {summary['classifier_fallbacks']}   "
          f"cold starts over 10 s: {len(cold)}")
    c = summary["cost_usd"]
    print(f"Cost per request with an LLM call: mean ${c['mean']}, p50 ${c['p50']}, p95 ${c['p95']} "
          f"(about ${c['per_1000_requests']} per 1,000 requests)")
    print(f"Tokens: mean input {summary['tokens']['mean_input']}, mean output {summary['tokens']['mean_output']}")
    print(f"\nSaved {metrics_file(args.tag)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_send = sub.add_parser("send", help="Send the warm-up and the golden-set items to the live endpoint")
    p_send.add_argument("--url", default=DEFAULT_URL)
    p_send.add_argument("--limit", type=int, default=0, help="Send only the first N items (for a trial run)")
    p_send.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between requests")
    p_send.add_argument("--tag", default="", help="Suffix for the report files, e.g. azure -> reports/live_run_azure.json")
    p_send.set_defaults(func=send)

    p_sum = sub.add_parser("summarise", help="Summarise downloaded Cloud Logging entries")
    p_sum.add_argument("--logs", required=True, help="JSON export from Cloud Logging or Azure Log Analytics")
    p_sum.add_argument("--all-entries", action="store_true", help="Do not restrict to the last run's time window")
    p_sum.add_argument("--tag", default="", help="Same tag as the send step")
    p_sum.set_defaults(func=summarise)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
