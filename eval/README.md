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
