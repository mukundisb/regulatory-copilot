# RAG evaluation

This folder holds the golden set used to measure how well the Copilot retrieves and cites EU-MDR sections.

## Files

- `golden_set.json`: 50 test items. 42 are regulatory questions, sent to retrieval as-is (the same path as `POST /retrieve`). 8 are adverse event narratives with a MAUDE label, sent through the label-steered retrieval used by `POST /assess`. Each item lists the sections accepted as correct.
- `../scripts/evaluate_rag.py`: runs every item and writes `reports/rag_eval.json`.

## How to run

From the repository root, with the virtual environment active:

```bash
python scripts/evaluate_rag.py              # retrieval only, no API calls
python scripts/evaluate_rag.py --with-llm   # adds citation metrics, needs GEMINI_API_KEY
```

## What the numbers mean

| Metric | Meaning |
| :--- | :--- |
| hit@1 | The top retrieved chunk is from an expected section |
| hit@3 | Any of the top 3 chunks is from an expected section |
| MRR | Mean reciprocal rank of the first expected section |
| no_result | Items where nothing passed the similarity cut-off |
| citation accuracy | The generated answer cites at least one expected section |
| citation precision | Of all sections cited, the share that were expected |

## Before publishing results

1. Review `expected_sections` for every item against the regulation text, then change `status` in `golden_set.json` from DRAFT to REVIEWED with the date.
2. Run the script and copy the summary table into the main `README.md` under an "Evaluation" heading, with the run date and index settings (chunk size, overlap, embedding model).
3. List the misses. They are the backlog for improving chunking, query steering or the similarity threshold.

## Faithfulness

Faithfulness asks whether every statement in a generated recommendation is supported by the retrieved EU-MDR text (or simply restates the input). It is measured by `../scripts/evaluate_faithfulness.py`.

1. `python scripts/evaluate_rag.py --with-llm` saves each recommendation and its retrieved chunks in `reports/rag_eval.json`.
2. `python scripts/evaluate_faithfulness.py --make-labels` writes `faithfulness_labels.csv` (20 items, one row per claim) and `faithfulness_review.md` (the same items with their retrieved text, for reading).
3. Fill `human_label` in the CSV with `supported` or `not_supported`.
4. `python scripts/evaluate_faithfulness.py` has a judge model label every claim, compares it with the human labels, and writes `reports/faithfulness_eval.json`.

Item faithfulness = supported claims / total claims. A claim is one sentence of the recommendation.

Do not rerun step 1 between labelling and judging: regenerated answers may be worded differently, and human labels are only compared when the claim text is unchanged.

The judge can be a different model from the generator. The published results use Claude: `--judge anthropic`, with `JUDGE_API_KEY` in `.env` (or set `JUDGE_PROVIDER=anthropic` there). Any OpenAI-style endpoint also works: `--judge openai-compatible --judge-base-url <url> --judge-model <name>`.
