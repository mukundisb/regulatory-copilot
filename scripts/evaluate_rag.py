"""
Evaluate EU-MDR retrieval and citation quality against a golden set.

What it measures
----------------
Retrieval (always):
  hit@1   - share of items where the top retrieved chunk is from an expected section
  hit@k   - share of items where any of the top-k chunks is from an expected section
  MRR     - mean reciprocal rank of the first expected section (0 when none is retrieved)
  no_result - items where retrieval returned nothing above the similarity cut-off

Citation (only with --with-llm, needs GEMINI_API_KEY):
  citation_accuracy  - share of items where the answer cites at least one expected section
  citation_precision - of all sections cited, the share that were expected
  llm_unavailable    - items where the recommender fell back to its deterministic template

Usage
-----
  python scripts/evaluate_rag.py                       # retrieval only, no API calls
  python scripts/evaluate_rag.py --with-llm            # adds citation metrics
  python scripts/evaluate_rag.py --min-hit-rate 0.8    # exit 1 if hit@k is below 0.8 (CI gate)

Run it from the repository root so that rag_pipeline finds ./chroma_db.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Hide the per-query "Batches" progress bars from sentence-transformers.
os.environ.setdefault("TQDM_DISABLE", "1")

# rag_recommender reads GEMINI_API_KEY with os.getenv, so .env must be loaded into the environment.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import rag_pipeline  # noqa: E402


def section_key(section_title: str) -> str:
    """'Article 87 - Reporting of serious incidents...' -> 'Article 87'."""
    return section_title.split(" - ")[0].strip()


def quiet(fn, *args, **kwargs):
    """Call fn while hiding the debug prints inside rag_pipeline.query_store."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def load_narrative_retrieval():
    """
    Narrative items must go through the same label-steered retrieval as POST /assess.
    Those functions live in app.py, which also imports the classifier stack.
    Returns (build_retrieval_query, retrieve_with_fallback) or None if app.py cannot be imported.
    """
    try:
        # Imported without hiding stdout: app.py attaches its log handler to stdout at import,
        # and warnings from the recommender must stay visible.
        from app import build_retrieval_query, retrieve_with_fallback
        return build_retrieval_query, retrieve_with_fallback
    except Exception as err:  # missing optional dependency, missing model file, etc.
        print(f"[WARN] Could not import app.py ({err}). Narrative items will be skipped.")
        return None


def retrieve(item: dict, top_k: int, narrative_fns) -> list[dict]:
    if item["type"] == "narrative":
        build_query, retrieve_with_fallback = narrative_fns
        steered = build_query(item["label"], item["query"])
        chunks, _query_used, _fallback = quiet(
            retrieve_with_fallback, steered, item["query"], top_k
        )
        return chunks
    return quiet(rag_pipeline.query_store, item["query"], top_k=top_k)


def score_retrieval(chunks: list[dict], expected: set[str]) -> dict:
    retrieved = [section_key(c.get("section", "")) for c in chunks]
    rank = next((i + 1 for i, s in enumerate(retrieved) if s in expected), None)
    return {
        "retrieved_sections": retrieved,
        "top_score": round(chunks[0]["similarity_score"], 4) if chunks else None,
        "first_hit_rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_k": rank is not None,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }


def score_citations(item: dict, chunks: list[dict], expected: set[str], provider: str) -> dict:
    from rag_recommender import generate_grounded_recommendation
    from telemetry import estimate_cost_usd

    started = time.perf_counter()
    result = generate_grounded_recommendation(
        narrative=item["query"],
        predicted_label=item.get("label", "O"),
        chunks=chunks,
        retrieval_fallback_triggered=False,
        top_score=chunks[0]["similarity_score"] if chunks else 0.0,
        provider=provider,
    )
    generate_ms = round((time.perf_counter() - started) * 1000, 1)
    usage = result.get("usage") or {}
    cited = [section_key(s) for s in result.get("citations", [])]
    warning = result.get("warning") or ""
    llm_used = not warning.startswith(("Degraded", "No regulatory context"))
    return {
        "recommendation": result.get("recommendation", ""),
        "cited_sections": cited,
        "llm_used": llm_used,
        "llm_warning": warning or None,
        "llm_grounding_verified": result.get("llm_grounding_verified", False),
        "citation_hit": any(s in expected for s in cited),
        "cited_expected": sum(1 for s in cited if s in expected),
        "cited_total": len(cited),
        "generate_ms": generate_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "est_cost_usd": estimate_cost_usd(provider, usage.get("input_tokens"), usage.get("output_tokens")),
    }


def _model(provider: str):
    from rag_recommender import _model_name
    return _model_name(provider)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-set", default=str(REPO_ROOT / "eval" / "golden_set.json"))
    parser.add_argument("--output", help="Default: reports/rag_eval.json for Gemini, reports/rag_eval_<provider>.json otherwise")
    parser.add_argument("--provider", choices=["gemini", "azure_openai"],
                        help="LLM for --with-llm (default: LLM_PROVIDER from config, normally gemini)")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks retrieved per item (the API uses 3)")
    parser.add_argument("--with-llm", action="store_true", help="Also generate answers and score citations")
    parser.add_argument("--min-hit-rate", type=float, default=0.0, help="Exit 1 if hit@k is below this value")
    args = parser.parse_args()
    from config import settings
    args.provider = args.provider or settings.llm_provider
    if not args.output:
        name = "rag_eval.json" if args.provider == "gemini" else f"rag_eval_{args.provider}.json"
        args.output = str(REPO_ROOT / "reports" / name)

    golden = json.loads(Path(args.golden_set).read_text(encoding="utf-8"))
    items = golden["items"]

    rag_pipeline.init_store()
    chunk_count = rag_pipeline.collection.count()
    if chunk_count == 0:
        print("[ERROR] The Chroma collection is empty. Run `python rag_pipeline.py` to ingest first.")
        return 2
    registry_ok, mismatches = rag_pipeline.verify_feature_registry()
    for m in mismatches:
        print(f"[WARN] Feature registry: {m}")

    narrative_fns = None
    if any(i["type"] == "narrative" for i in items):
        narrative_fns = load_narrative_retrieval()

    rows, skipped = [], []
    for item in items:
        if item["type"] == "narrative" and narrative_fns is None:
            skipped.append(item["id"])
            continue
        expected = set(item["expected_sections"])
        chunks = retrieve(item, args.top_k, narrative_fns)
        row = {
            "id": item["id"],
            "type": item["type"],
            "query": item["query"],
            "expected_sections": item["expected_sections"],
            **score_retrieval(chunks, expected),
        }
        if args.with_llm:
            row.update(score_citations(item, chunks, expected, args.provider))
            # Kept so that scripts/evaluate_faithfulness.py can check each answer against its sources.
            row["retrieved_chunks"] = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "section": c.get("section", ""),
                    "similarity_score": round(c.get("similarity_score", 0.0), 4),
                    "text": c.get("text", ""),
                }
                for c in chunks
            ]
        rows.append(row)

    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"items": 0}
        out = {
            "items": n,
            "hit_at_1": round(sum(r["hit_at_1"] for r in subset) / n, 3),
            f"hit_at_{args.top_k}": round(sum(r["hit_at_k"] for r in subset) / n, 3),
            "mrr": round(sum(r["reciprocal_rank"] for r in subset) / n, 3),
            "no_result": sum(1 for r in subset if not r["retrieved_sections"]),
        }
        if args.with_llm:
            answered = [r for r in subset if r["llm_used"]]
            cited_total = sum(r["cited_total"] for r in answered)
            out.update({
                "llm_unavailable": n - len(answered),
                "citation_accuracy": round(sum(r["citation_hit"] for r in answered) / len(answered), 3) if answered else None,
                "citation_precision": round(sum(r["cited_expected"] for r in answered) / cited_total, 3) if cited_total else None,
                "grounding_verified": round(sum(r["llm_grounding_verified"] for r in answered) / len(answered), 3) if answered else None,
                "median_generate_ms": round(statistics.median(r["generate_ms"] for r in answered), 1) if answered else None,
                "mean_cost_usd": (round(statistics.mean(costs), 6) if (costs := [r["est_cost_usd"] for r in answered
                                  if r["est_cost_usd"] is not None]) else None),
            })
        return out

    summary = {
        "overall": summarise(rows),
        "question": summarise([r for r in rows if r["type"] == "question"]),
        "narrative": summarise([r for r in rows if r["type"] == "narrative"]),
    }
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_set": Path(args.golden_set).name,
        "golden_set_status": golden.get("status"),
        "top_k": args.top_k,
        "llm": {"provider": args.provider, "model": _model(args.provider)} if args.with_llm else None,
        "index": {
            "chunk_count": chunk_count,
            "embedding_model": rag_pipeline.EMBEDDING_MODEL_NAME,
            "chunk_size": rag_pipeline.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": rag_pipeline.DEFAULT_OVERLAP,
            "registry_ok": registry_ok,
        },
        "skipped_items": skipped,
        "summary": summary,
        "items": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    hit_key = f"hit_at_{args.top_k}"
    print(f"\nRAG evaluation: {len(rows)} items scored, {len(skipped)} skipped, top_k={args.top_k}")
    print(f"{'subset':<10} {'items':>5} {'hit@1':>7} {'hit@' + str(args.top_k):>7} {'MRR':>7} {'no_result':>10}")
    for name, s in summary.items():
        if s["items"]:
            print(f"{name:<10} {s['items']:>5} {s['hit_at_1']:>7.3f} {s[hit_key]:>7.3f} {s['mrr']:>7.3f} {s['no_result']:>10}")
    if args.with_llm:
        o = summary["overall"]
        print(f"\nLLM: {args.provider} ({_model(args.provider)})")
        print(f"Citation accuracy: {o['citation_accuracy']}  precision: {o['citation_precision']}  "
              f"grounding verified: {o['grounding_verified']}  LLM unavailable on {o['llm_unavailable']} items")
        print(f"Median generation: {o['median_generate_ms']} ms   mean cost per answer: ${o['mean_cost_usd']}")
        reasons = sorted({r["llm_warning"][:200] for r in rows if not r["llm_used"] and r["llm_warning"]})
        for reason in reasons:
            print(f"  [WARN] LLM not used: {reason}")

    misses = [r for r in rows if not r["hit_at_k"]]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(f"  {r['id']}  expected {r['expected_sections']}  got {r['retrieved_sections'] or 'nothing'}")
    print(f"\nFull report written to {out_path}")

    overall_hit = summary["overall"].get(hit_key, 0.0)
    if overall_hit < args.min_hit_rate:
        print(f"[FAIL] {hit_key} {overall_hit} is below the required {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
