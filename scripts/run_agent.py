"""
Run the regulatory agent on one adverse-event narrative.

The agent classifies the event, optionally looks up FDA MAUDE reports for the device, retrieves
the EU-MDR passages that apply and drafts a recommendation. It then stops and asks you to
approve, edit or reject the draft. Only an approved or edited draft is saved as a result.

Usage
-----
  python scripts/run_agent.py "Patient with an ICD collapsed ..." --device "implantable defibrillator"
  python scripts/run_agent.py "..." --planner rules          # no Gemini call for planning
  python scripts/run_agent.py --maude-only "infusion pump"   # test the openFDA tool by itself

Each run writes agent/runs/<timestamp>.json (the steps taken, the draft and your decision).
Run from the repository root. For a quick offline run set CLASSIFIER_BACKEND=tfidf so the
classifier does not wait for the Vertex AI endpoint to wake up.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

RUNS_DIR = REPO_ROOT / "agent" / "runs"


def print_packet(packet: dict) -> None:
    print("\n" + "=" * 70)
    print("REVIEW REQUIRED - nothing is final until you answer")
    print("=" * 70)
    print(f"Narrative:  {packet['narrative']}")
    print(f"Severity:   {packet['severity_label']}")
    maude = packet.get("maude")
    if maude and maude.get("ok") and maude.get("found"):
        print(f"MAUDE:      {maude['total_reports']} reports for '{maude['device_term']}' {maude['by_event_type']}")
        for r in maude["sample_reports"]:
            print(f"            {r['date_received']}  {r['event_type']}  {r['brand_name']}  ({r['report_number']})")
    elif maude:
        print(f"MAUDE:      {maude.get('error') or 'no matching reports'}")
    print(f"Sections:   {packet['retrieved_sections']}")
    if packet.get("problems"):
        print(f"\n*** {len(packet['problems'])} TOOL(S) FAILED - the draft below may be incomplete ***")
        for problem in packet["problems"]:
            print(f"    {problem}")
    print(f"\nDraft recommendation:\n{packet['recommendation']}")
    print(f"\nCitations:  {packet['citations']}   grounding verified: {packet['grounding_verified']}")
    if packet.get("warning"):
        print(f"Warning:    {packet['warning']}")
    print("\nSteps taken:")
    for step in packet["steps"]:
        print(f"  - {step['action']}: {step['summary']}")


def ask_reviewer() -> tuple[str, str, str]:
    while True:
        answer = input("\n[a]pprove, [e]dit, [r]eject: ").strip().lower()[:1]
        if answer == "a":
            return "approve", input("Note (optional): ").strip(), ""
        if answer == "r":
            return "reject", input("Reason (optional): ").strip(), ""
        if answer == "e":
            text = input("Your corrected recommendation: ").strip()
            if text:
                return "edit", input("Note (optional): ").strip(), text
            print("An edit needs text.")


def maude_only(term: str) -> int:
    from agent.tools import search_maude
    result = search_maude(term)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("narrative", nargs="?", help="The adverse-event narrative")
    parser.add_argument("--device", help="Generic device name, used for the MAUDE lookup (e.g. 'infusion pump')")
    parser.add_argument("--planner", choices=["gemini", "rules"], default="gemini")
    parser.add_argument("--auto-approve", action="store_true", help="Approve without asking (for scripted trials only)")
    parser.add_argument("--maude-only", metavar="TERM", help="Call only the openFDA tool and print the result")
    args = parser.parse_args()

    if args.maude_only:
        return maude_only(args.maude_only)
    if not args.narrative:
        parser.error("a narrative is required")

    from agent.graph import build_graph, pending_packet, resume_run, start_run
    from agent.planner import GeminiPlanner, RulePlanner

    planner = GeminiPlanner() if args.planner == "gemini" else RulePlanner()
    graph = build_graph(planner=planner)
    thread_id = str(uuid.uuid4())

    output = start_run(graph, args.narrative, thread_id, device_hint=args.device)
    packet = pending_packet(output)
    if packet is None:
        print("[ERROR] The run ended without reaching the approval step.")
        return 2

    print_packet(packet)
    if args.auto_approve:
        # Scripted trials only: never approve a draft built after a tool failure.
        decision, note, edited = ("reject", "auto-reject: a tool failed", "") if packet.get("problems") else ("approve", "auto-approved", "")
    else:
        decision, note, edited = ask_reviewer()
    output = resume_run(graph, thread_id, decision, note=note, recommendation=edited)

    record = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planner": args.planner,
        "narrative": args.narrative,
        "device_hint": args.device,
        "steps": output.get("trace", []),
        "guardrail_events": output.get("events", []),
        "draft": output.get("draft"),
        "reviewer_decision": decision,
        "reviewer_note": note,
        "status": output.get("status"),
        "final": output.get("final"),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nStatus: {record['status'].upper()}")
    if output.get("events"):
        print("Guardrail events:", [e["type"] for e in output["events"]])
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
