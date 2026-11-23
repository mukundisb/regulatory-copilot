"""
Evaluate the agent's tool selection and task success on eval/agent_scenarios.json.

Each scenario runs through the agent up to the human approval step; the review packet and the
planner's recorded decisions are then scored. The approval step itself is not scored: it is
the person's decision.

Tool selection (did the agent choose its tools well?)
  decision_accuracy     share of planner calls the graph accepted unchanged (no guardrail stepped in)
  maude_precision       of the runs that looked up MAUDE, the share where a lookup was expected
  maude_recall          of the runs where a lookup was expected, the share that made one
  device_term_accuracy  of the expected lookups that were made, the share whose term matched the device
  tool_selection        share of scenarios with no guardrail intervention, the right MAUDE decision
                        and (when looked up) an acceptable device term

Task success (was the result right?)
  label_accuracy        severity label equals expected_label
  retrieval_hit         an expected section is among the retrieved passages
  citation_hit          the draft cites an expected section
  grounded              the draft passed the citation-containment check
  task_success          all four of the above, and the run reached the approval step
  fully_correct         task_success and tool_selection together

Retrieval retry (after an empty first retrieval the planner may retry once with its own query)
  first_retrieval_empty runs whose first retrieval returned no passages
  retry_rate            of those, the share where the planner retried
  retry_rescued         of the retries, the share that then found passages

Usage (from the repository root):
  python scripts/evaluate_agent.py                     # Gemini planner, all 30 scenarios
  python scripts/evaluate_agent.py --planner rules     # fixed-order baseline, no planning calls
  python scripts/evaluate_agent.py --ids A25 A26 A27   # a subset

Writes reports/agent_eval_<planner>.json. Live calls: Gemini (planning and the draft) and openFDA.
Set CLASSIFIER_BACKEND=tfidf to skip the Vertex AI cold start; the report records which classifier ran.
"""

import argparse
import contextlib
import io
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

SCENARIOS = REPO_ROOT / "eval" / "agent_scenarios.json"


def section_key(section: str) -> str:
    """'Article 87 - Reporting of serious incidents...' -> 'Article 87'."""
    return (section or "").split(" - ")[0].strip()


def ratio(numerator: int, denominator: int):
    return round(numerator / denominator, 3) if denominator else None


# --------------------------------------------------------------------------- one scenario

def run_scenario(graph, scenario: dict) -> tuple[dict, dict | None, float]:
    """Run to the approval pause. Returns (graph state, review packet or None, seconds)."""
    from agent.graph import pending_packet, start_run

    thread = f"eval-{scenario['id']}-{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # rag_pipeline prints every query
        output = start_run(graph, scenario["narrative"], thread, device_hint=scenario.get("device_hint"))
    seconds = round(time.perf_counter() - started, 2)
    # The paused run's full state lives in the checkpointer; the invoke output alone can omit keys.
    state = graph.get_state({"configurable": {"thread_id": thread}}).values
    return state, pending_packet(output), seconds


def score_scenario(scenario: dict, state: dict, packet: dict | None, seconds: float) -> dict:
    """Score one finished run. Pure function: no I/O."""
    expected = set(scenario["expected_sections"])
    events = [e["type"] for e in state.get("events", [])]
    decisions = state.get("decisions", [])
    tools_run = [step["action"] for step in state.get("trace", [])]

    maude_called = "search_maude" in tools_run
    maude = state.get("maude") or {}
    term = (maude.get("device_term") or "").lower()
    term_ok = any(k.lower() in term for k in scenario["device_keywords"]) if maude_called else None

    tool_selection = (
        not events
        and maude_called == scenario["expect_maude"]
        and (not maude_called or bool(term_ok))
    )

    reached = packet is not None
    retrieved = [section_key(s) for s in (packet or {}).get("retrieved_sections", [])]
    cited = [section_key(s) for s in (packet or {}).get("citations", [])]
    label = state.get("label")
    retrievals = state.get("retrieval_log", [])
    first_empty = bool(retrievals) and not retrievals[0].get("passages")
    retried = len(retrievals) > 1

    row = {
        "id": scenario["id"],
        "category": scenario["category"],
        "seconds": seconds,
        "tools_run": tools_run,
        "planner_calls": len(decisions),
        "decisions_accepted": sum(1 for d in decisions if d["accepted"]),
        "decisions": decisions,
        "guardrail_events": events,
        "expect_maude": scenario["expect_maude"],
        "maude_called": maude_called,
        "maude_term": maude.get("device_term"),
        "maude_term_ok": term_ok,
        "maude_ok": maude.get("ok") if maude_called else None,
        "maude_total_reports": maude.get("total_reports") if maude_called else None,
        "tool_selection": tool_selection,
        "classifier_backend": state.get("classifier_backend"),
        "label": label,
        "expected_label": scenario["expected_label"],
        "label_correct": label == scenario["expected_label"],
        "retrieval_attempts": len(retrievals),
        "first_retrieval_empty": first_empty,
        "retried": retried,
        "retry_query": retrievals[1].get("query_from_planner") if retried else None,
        "retry_found_passages": bool(retrievals[1].get("passages")) if retried else None,
        "retrieved_sections": retrieved,
        "retrieval_hit": any(s in expected for s in retrieved),
        "cited_sections": cited,
        "citation_hit": any(s in expected for s in cited),
        "grounded": bool((packet or {}).get("grounding_verified")),
        "reached_approval": reached,
        "warning": (packet or {}).get("warning"),
        "recommendation": (packet or {}).get("recommendation"),
    }
    row["task_success"] = all((reached, row["label_correct"], row["retrieval_hit"], row["citation_hit"], row["grounded"]))
    row["fully_correct"] = row["task_success"] and tool_selection
    return row


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"scenarios": 0}
    planner_calls = sum(r["planner_calls"] for r in rows)
    called = [r for r in rows if r["maude_called"]]
    expected = [r for r in rows if r["expect_maude"]]
    expected_and_called = [r for r in expected if r["maude_called"]]

    def share(key):
        return ratio(sum(1 for r in rows if r[key]), n)

    return {
        "scenarios": n,
        "tool_selection": share("tool_selection"),
        "decision_accuracy": ratio(sum(r["decisions_accepted"] for r in rows), planner_calls),
        "maude_precision": ratio(sum(1 for r in called if r["expect_maude"]), len(called)),
        "maude_recall": ratio(len(expected_and_called), len(expected)),
        "device_term_accuracy": ratio(sum(1 for r in expected_and_called if r["maude_term_ok"]), len(expected_and_called)),
        "label_accuracy": share("label_correct"),
        "retrieval_hit": share("retrieval_hit"),
        "citation_hit": share("citation_hit"),
        "grounded": share("grounded"),
        "task_success": share("task_success"),
        "fully_correct": share("fully_correct"),
        "first_retrieval_empty": sum(1 for r in rows if r.get("first_retrieval_empty")),
        "retry_rate": ratio(sum(1 for r in rows if r.get("first_retrieval_empty") and r.get("retried")),
                            sum(1 for r in rows if r.get("first_retrieval_empty"))),
        "retry_rescued": ratio(sum(1 for r in rows if r.get("retry_found_passages")),
                               sum(1 for r in rows if r.get("retried"))),
        "mean_planner_calls": round(planner_calls / n, 2),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
        "guardrail_events": dict(Counter(e for r in rows for e in r["guardrail_events"])),
    }


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=["gemini", "rules"], default="gemini")
    parser.add_argument("--scenarios", default=str(SCENARIOS))
    parser.add_argument("--ids", nargs="*", help="Run only these scenario ids")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between scenarios")
    args = parser.parse_args()

    from agent.graph import build_graph
    from agent.planner import GeminiPlanner, RulePlanner

    data = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    scenarios = [s for s in data["items"] if not args.ids or s["id"] in args.ids]
    planner = GeminiPlanner() if args.planner == "gemini" else RulePlanner()
    graph = build_graph(planner=planner)

    rows, failures = [], []
    for number, scenario in enumerate(scenarios, 1):
        try:
            row = score_scenario(scenario, *run_scenario(graph, scenario))
        except Exception as err:  # one broken run must not lose the others
            failures.append({"id": scenario["id"], "error": f"{type(err).__name__}: {err}"[:300]})
            print(f"  {number:>2}/{len(scenarios)} {scenario['id']}  ERROR {failures[-1]['error'][:100]}")
            continue
        rows.append(row)
        flags = "".join([
            "T" if row["tool_selection"] else "t",
            "S" if row["task_success"] else "s",
        ])
        print(f"  {number:>2}/{len(scenarios)} {row['id']}  {flags}  label {row['label']}/{row['expected_label']}  "
              f"tools {','.join(t.split('_')[0] for t in row['tools_run'])}  "
              f"maude '{row['maude_term'] or '-'}'  retry '{row['retry_query'] or '-'}'  "
              f"events {row['guardrail_events'] or '-'}  {row['seconds']}s")
        time.sleep(args.pause)

    categories = sorted({r["category"] for r in rows})
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planner": args.planner,
        "scenario_file": Path(args.scenarios).name,
        "scenario_status": data.get("status"),
        "classifier_backends": dict(Counter(r["classifier_backend"] for r in rows)),
        "summary": summarise(rows),
        "by_category": {c: summarise([r for r in rows if r["category"] == c]) for c in categories},
        "failures": failures,
        "items": rows,
    }
    out = REPO_ROOT / "reports" / f"agent_eval_{args.planner}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    s = report["summary"]
    print(f"\nAgent evaluation, planner={args.planner}, {s.get('scenarios', 0)} scenarios, {len(failures)} errors "
          f"(T/S = tool selection / task success correct; lower case = wrong)")
    if rows:
        for key in ("tool_selection", "decision_accuracy", "maude_precision", "maude_recall", "device_term_accuracy",
                    "label_accuracy", "retrieval_hit", "citation_hit", "grounded", "task_success", "fully_correct",
                    "first_retrieval_empty", "retry_rate", "retry_rescued", "mean_planner_calls"):
            print(f"  {key:<22}{s[key]}")
        print(f"  {'guardrail_events':<22}{s['guardrail_events'] or 'none'}")
        print(f"  {'classifier':<22}{report['classifier_backends']}")
        print("\nBy category (tool selection / task success):")
        for c, cs in report["by_category"].items():
            print(f"  {c:<16}{cs['scenarios']:>3}   {cs['tool_selection']}   {cs['task_success']}")
    print(f"\nSaved {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
