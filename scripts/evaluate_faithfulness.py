"""
Faithfulness evaluation for generated recommendations.

Question answered: is every statement in a recommendation supported by the retrieved
EU-MDR text (or by the input narrative itself)?

How it works
------------
1. Reads reports/rag_eval.json, produced by `python scripts/evaluate_rag.py --with-llm`.
   Each item there holds the recommendation and the retrieved chunks.
2. Splits each recommendation into claims, one per sentence (split_claims).
3. A judge model labels every claim "supported" or "not_supported".
   Item faithfulness = supported claims / total claims.
4. If eval/faithfulness_labels.csv holds human labels, the judge is compared with them
   claim by claim and the agreement rate is reported.

The judge can be a different model from the one that wrote the answers:
  --judge gemini             uses google-genai and GEMINI_API_KEY (default)
  --judge anthropic          uses the Claude Messages API and JUDGE_API_KEY
  --judge openai-compatible  uses any /chat/completions endpoint; needs --judge-base-url,
                             --judge-model and JUDGE_API_KEY
Instead of the flags, set JUDGE_PROVIDER and JUDGE_MODEL in .env. A key on its own does not
select a judge: the provider must be named.

Usage (from the repository root)
--------------------------------
  python scripts/evaluate_faithfulness.py --make-labels   # write the hand-labelling files for 20 items
  python scripts/evaluate_faithfulness.py                 # judge all items, compare with human labels
  python scripts/evaluate_faithfulness.py --only-labelled # judge only the hand-labelled items (cheap check)
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

RAG_REPORT = REPO_ROOT / "reports" / "rag_eval.json"
LABELS_CSV = REPO_ROOT / "eval" / "faithfulness_labels.csv"
REVIEW_MD = REPO_ROOT / "eval" / "faithfulness_review.md"
OUTPUT = REPO_ROOT / "reports" / "faithfulness_eval.json"

DEFAULT_GEMINI_JUDGE = "gemini-3.8-flash"
DEFAULT_ANTHROPIC_JUDGE = "claude-sonnet-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
VALID_LABELS = {"supported", "not_supported"}

# Sentence boundaries that are not really boundaries in regulatory prose.
ABBREVIATIONS = ("Art.", "No.", "e.g.", "i.e.", "para.", "cf.", "etc.", "vs.")


# --------------------------------------------------------------------------- claims

def split_claims(text: str) -> list[str]:
    """Split a recommendation into claims, one per sentence."""
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", protected.strip())
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- judge

JUDGE_PROMPT = """You are auditing an AI-generated regulatory recommendation for faithfulness.

INPUT GIVEN TO THE SYSTEM:
{query}

RETRIEVED EU-MDR TEXT (the only regulatory source the system was allowed to use):
{context}

CLAIMS TO CHECK (numbered):
{claims}

For each claim decide:
- "supported": the retrieved text states it or directly entails it, OR the claim only restates
  the input above (for example, describing the event or its classification), OR the claim makes
  no regulatory assertion at all. Sentences that only say retrieval confidence is low, that the
  retrieved provisions are weak candidates, or that a regulatory affairs specialist should verify
  or evaluate something are always "supported": the system adds them by design.
- "not_supported": the claim adds a regulatory fact that the retrieved text does not contain
  (a deadline, an obligation, an actor, a number, an article reference), or contradicts it.

Judge only against the retrieved text and the input. Do not use your own knowledge of EU-MDR:
a claim that is true in the regulation but absent from the retrieved text is "not_supported".

Return JSON only, in this exact shape:
{{"claims": [{{"id": 1, "label": "supported", "evidence": "short quote from the retrieved text or input, or empty"}}]}}
"""


def build_prompt(item: dict, claims: list[str]) -> str:
    context = "\n\n".join(
        f"--- {c.get('section', 'Unknown section')} ---\n{c.get('text', '')}"
        for c in item["retrieved_chunks"]
    )
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    return JUDGE_PROMPT.format(query=item["query"], context=context, claims=numbered)


def call_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return response.text


def call_anthropic(prompt: str, model: str) -> str:
    import requests

    headers = {
        "x-api-key": os.environ["JUDGE_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {"model": model, "max_tokens": 4096, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=120)
    if response.status_code == 400 and "temperature" in response.text:
        # Some models do not accept sampling parameters; retry with the model default.
        body.pop("temperature")
        response = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=120)
    if not response.ok:
        raise RuntimeError(f"Anthropic API {response.status_code}: {response.text[:200]}")
    return "".join(block.get("text", "") for block in response.json()["content"])


def call_openai_compatible(prompt: str, model: str, base_url: str) -> str:
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['JUDGE_API_KEY']}"},
        json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_judge_output(raw: str, n_claims: int) -> list[dict]:
    """Extract the JSON object and check it covers every claim exactly once."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge output")
    verdicts = json.loads(match.group(0))["claims"]
    by_id = {int(v["id"]): v for v in verdicts}
    if sorted(by_id) != list(range(1, n_claims + 1)):
        raise ValueError(f"judge returned ids {sorted(by_id)}, expected 1..{n_claims}")
    for v in by_id.values():
        if v["label"] not in VALID_LABELS:
            raise ValueError(f"unexpected label {v['label']!r}")
    return [by_id[i] for i in range(1, n_claims + 1)]


def judge_item(item: dict, claims: list[str], args, attempts: int = 2) -> list[dict]:
    """Ask the judge; if its reply is not valid JSON covering every claim, ask once more."""
    prompt = build_prompt(item, claims)
    for attempt in range(attempts):
        if args.judge == "gemini":
            raw = call_gemini(prompt, args.judge_model)
        elif args.judge == "anthropic":
            raw = call_anthropic(prompt, args.judge_model)
        else:
            raw = call_openai_compatible(prompt, args.judge_model, args.judge_base_url)
        try:
            return parse_judge_output(raw, len(claims))
        except (ValueError, KeyError, TypeError):  # json.JSONDecodeError is a ValueError
            if attempt == attempts - 1:
                raise


# --------------------------------------------------------------------------- human labels

def pick_items_to_label(items: list[dict], n: int = 20) -> list[dict]:
    """All retrieval misses, then narratives, then questions spread evenly across the set."""
    misses = [i for i in items if not i["hit_at_k"]]
    narratives = [i for i in items if i["hit_at_k"] and i["type"] == "narrative"]
    questions = [i for i in items if i["hit_at_k"] and i["type"] == "question"]
    chosen = (misses + narratives)[:n]
    remaining = n - len(chosen)
    if remaining > 0 and questions:
        step = max(1, len(questions) // remaining)
        chosen += questions[::step][:remaining]
    return chosen


def make_label_files(items: list[dict], force: bool) -> int:
    if LABELS_CSV.exists() and not force:
        print(f"[STOP] {LABELS_CSV} already exists. Use --force to overwrite it and lose any labels in it.")
        return 2
    chosen = pick_items_to_label(items)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "claim_id", "claim", "human_label", "note"])
        for item in chosen:
            for i, claim in enumerate(split_claims(item["recommendation"]), 1):
                writer.writerow([item["id"], i, claim, "", ""])

    lines = ["# Faithfulness review sheet", "",
             "Read each item here, then fill `human_label` in `faithfulness_labels.csv` "
             "with `supported` or `not_supported` for every claim.", ""]
    for item in chosen:
        lines += [f"## {item['id']} ({item['type']}, retrieval {'hit' if item['hit_at_k'] else 'MISS'})", "",
                  f"**Input:** {item['query']}", "",
                  f"**Expected sections:** {', '.join(item['expected_sections'])}", "",
                  "**Recommendation, split into claims:**", ""]
        lines += [f"{i}. {c}" for i, c in enumerate(split_claims(item["recommendation"]), 1)]
        lines += ["", "**Retrieved text:**", ""]
        for chunk in item["retrieved_chunks"]:
            lines += [f"> **{chunk.get('section', '')}** (score {chunk.get('similarity_score', 0):.3f})", ">",
                      "> " + chunk.get("text", "").replace("\n", "\n> "), ""]
    REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {LABELS_CSV} ({len(chosen)} items) and {REVIEW_MD}")
    return 0


def normalise_label(value: str) -> str | None:
    v = value.strip().lower().replace(" ", "_")
    if v in {"supported", "s", "yes", "y"}:
        return "supported"
    if v in {"not_supported", "n", "no", "ns", "unsupported"}:
        return "not_supported"
    return None


def load_human_labels() -> dict[tuple[str, int], dict]:
    """Returns {(item_id, claim_id): {"claim": text, "label": label}} for filled-in rows only."""
    if not LABELS_CSV.exists():
        return {}
    labels = {}
    with open(LABELS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = normalise_label(row.get("human_label", ""))
            if label:
                labels[(row["item_id"], int(row["claim_id"]))] = {"claim": row["claim"].strip(), "label": label}
    return labels


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-labels", action="store_true", help="Write the hand-labelling CSV and review sheet, then stop")
    parser.add_argument("--force", action="store_true", help="With --make-labels: overwrite an existing labels file")
    parser.add_argument("--only-labelled", action="store_true", help="Judge only the items that have human labels")
    parser.add_argument("--judge", choices=["gemini", "anthropic", "openai-compatible"],
                        default=os.getenv("JUDGE_PROVIDER", "gemini"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL"))
    parser.add_argument("--input", default=str(RAG_REPORT),
                        help="Report from evaluate_rag.py --with-llm (e.g. reports/rag_eval_azure_openai.json)")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    rag_report, output = Path(args.input), Path(args.output)
    # The hand labels were made on the answers in the default report; they say nothing about another one.
    use_human_labels = rag_report.resolve() == RAG_REPORT.resolve()

    if not rag_report.exists():
        print(f"[ERROR] {rag_report} not found. Run: python scripts/evaluate_rag.py --with-llm")
        return 2
    rag = json.loads(rag_report.read_text(encoding="utf-8"))
    items = [i for i in rag["items"] if i.get("recommendation") and i.get("retrieved_chunks") and i.get("llm_used")]
    if not items:
        print("[ERROR] rag_eval.json has no recommendations or chunk text. "
              "Rerun: python scripts/evaluate_rag.py --with-llm")
        return 2

    if args.make_labels:
        if not use_human_labels:
            print("[STOP] --make-labels works on the default report only.")
            return 2
        return make_label_files(items, args.force)

    if args.judge == "gemini":
        args.judge_model = args.judge_model or DEFAULT_GEMINI_JUDGE
        if not os.getenv("GEMINI_API_KEY"):
            print("[ERROR] GEMINI_API_KEY is not set.")
            return 2
        if os.getenv("JUDGE_API_KEY"):
            print("[NOTE] JUDGE_API_KEY is set but the judge is Gemini. "
                  "Pass --judge anthropic (or set JUDGE_PROVIDER in .env) to use the other model.")
    elif args.judge == "anthropic":
        args.judge_model = args.judge_model or DEFAULT_ANTHROPIC_JUDGE
        if not os.getenv("JUDGE_API_KEY"):
            print("[ERROR] --judge anthropic needs JUDGE_API_KEY in .env.")
            return 2
    else:
        if not (args.judge_model and args.judge_base_url and os.getenv("JUDGE_API_KEY")):
            print("[ERROR] --judge openai-compatible needs --judge-model, --judge-base-url and JUDGE_API_KEY.")
            return 2

    print(f"Judge: {args.judge}:{args.judge_model}")
    human = load_human_labels() if use_human_labels else {}
    if not use_human_labels:
        print(f"[NOTE] Judging {rag_report.name}: the hand labels belong to {RAG_REPORT.name}, so no agreement is computed.")
    labelled_ids = {item_id for item_id, _ in human}
    if args.only_labelled:
        items = [i for i in items if i["id"] in labelled_ids]

    rows, errors = [], []
    for item in items:
        claims = split_claims(item["recommendation"])
        try:
            verdicts = judge_item(item, claims, args)
        except Exception as err:
            errors.append({"id": item["id"], "error": str(err)[:300]})
            continue
        supported = sum(v["label"] == "supported" for v in verdicts)
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "retrieval_hit": item["hit_at_k"],
            "faithfulness": round(supported / len(claims), 3),
            "claims": [
                {"id": i, "text": c, "judge_label": v["label"], "evidence": v.get("evidence", "")}
                for i, (c, v) in enumerate(zip(claims, verdicts), 1)
            ],
        })

    # Judge versus human, claim by claim. A claim is compared only if its text is unchanged.
    compared = agree = stale = 0
    disagreements = []
    for row in rows:
        for claim in row["claims"]:
            h = human.get((row["id"], claim["id"]))
            if not h:
                continue
            if h["claim"] != claim["text"]:
                stale += 1
                continue
            compared += 1
            if h["label"] == claim["judge_label"]:
                agree += 1
            else:
                disagreements.append({"id": row["id"], "claim_id": claim["id"], "claim": claim["text"],
                                      "human": h["label"], "judge": claim["judge_label"]})

    def mean(values):
        return round(sum(values) / len(values), 3) if values else None

    all_claims = [c for r in rows for c in r["claims"]]
    misses = [r for r in rows if not r["retrieval_hit"]]
    summary = {
        "items_judged": len(rows),
        "judge_errors": len(errors),
        "claims": len(all_claims),
        "unsupported_claims": sum(c["judge_label"] == "not_supported" for c in all_claims),
        "mean_faithfulness": mean([r["faithfulness"] for r in rows]),
        "items_below_1": sum(r["faithfulness"] < 1 for r in rows),
        "mean_faithfulness_retrieval_hit": mean([r["faithfulness"] for r in rows if r["retrieval_hit"]]),
        "mean_faithfulness_retrieval_miss": mean([r["faithfulness"] for r in misses]),
        "fully_faithful_on_retrieval_miss": sum(r["faithfulness"] == 1 for r in misses),
        "human_claims_compared": compared,
        "judge_human_agreement": round(agree / compared, 3) if compared else None,
        "human_claims_skipped_text_changed": stale,
    }
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_report": rag_report.name,
        "source_report_run_at": rag.get("run_at"),
        "llm": rag.get("llm"),
        "judge": {"provider": args.judge, "model": args.judge_model},
        "summary": summary,
        "disagreements": disagreements,
        "errors": errors,
        "items": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nFaithfulness evaluation, judge = {args.judge}:{args.judge_model}")
    for key, value in summary.items():
        print(f"  {key:<36} {value}")
    if stale:
        print(f"\n[WARN] {stale} human-labelled claims no longer match the current recommendation text. "
              "The answers were regenerated after labelling; relabel or restore the old report.")
    for d in disagreements:
        print(f"  [DISAGREE] {d['id']} claim {d['claim_id']}: human={d['human']} judge={d['judge']}")
    for e in errors:
        print(f"  [ERROR] {e['id']}: {e['error']}")
    print(f"\nFull report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
