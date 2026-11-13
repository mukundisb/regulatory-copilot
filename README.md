# Regulatory Co-Pilot: AI Adverse Event Classifier & EU-MDR Retrieval Engine

[![CI Test Suite](https://github.com/mukundisb/regulatory-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundisb/regulatory-copilot/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Netlify-success?logo=netlify)](https://regulatory-copilot.netlify.app)
[![Cloud Run Deployment](https://img.shields.io/badge/GCP-Cloud%20Run%20Deployed-blue?logo=googlecloud)](https://cloud.google.com/run)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)

An automated regulatory triage and decision-support API built for medical device post-market surveillance. The system classifies unstructured FDA MAUDE adverse event narratives into statutory reporting tiers, dynamically steers regulatory vector queries across Regulation (EU) 2017/745 (EU-MDR), and applies quality gates with adaptive fallback retrieval.

---

🔗 **Live Frontend Application:** [https://regulatory-copilot.netlify.app](https://regulatory-copilot.netlify.app)  
*(Backend hosted serverless on Google Cloud Run)*

## Architecture at a Glance

| Component | Runs on | Role |
| :--- | :--- | :--- |
| **API** (`app.py`) | Cloud Run, scale-to-zero | Classification routing, EU-MDR retrieval, grounded recommendation |
| **Classifier** (`maude_classifier/serve.py`) | Vertex AI endpoint, 0–1 replicas | Fine-tuned Bio_ClinicalBERT (`cls_mean_concat`); TF-IDF warm standby inside the API |
| **Training** (`training/train_bert.py`) | Vertex AI Custom Job, T4 GPU | MLflow-tracked; promotion to serving is manual and human-gated |
| **Ingestion** (`ingestion/fetch_maude_events.py`) | Cloud Run Job + Cloud Scheduler, weekly | Watermarked incremental pull from openFDA |
| **Frontend** (`frontend/`) | Netlify | Vite + React client |

Full diagram, request flow, image targets and known limitations: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Architectural Problem & Context

Pharma and MedTech AI engineering roles require systems that bridge statistical ML with deterministic statutory requirements. A pure LLM or generic RAG pipeline risks hallucinating compliance timelines or missing critical reporting triggers. 

This engine implements a **three-stage agentic workflow**:
1. **Upstream Classification:** Multi-class classification of medical device adverse events into MAUDE categories: **Death (`D`)**, **Injury (`I`)**, **Malfunction (`M`)**, or **Other (`O`)**.
2. **Dynamic Query Steering:** Reformulates statutory vector search queries with regulatory criteria (e.g., EU-MDR Article 87 vigilance timelines vs. Article 88 trend reporting/CAPA).
3. **Adaptive Quality Gate:** Evaluates retrieval confidence against an empirical cosine similarity cutoff ($0.55$) and triggers an automatic fallback pass if steered search underperforms.
4. **Grounded Recommendation:** Gemini drafts the recommendation from the retrieved EU-MDR sections only; any citation outside those sections is pruned in code, with a deterministic fallback when the LLM is unavailable.
 

## Documentation Links

* **[Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md):** Services, request flow, training and promotion, weekly ingestion, container images, and known limitations.
* **[API Reference (`docs/API.md`)](docs/API.md):** Complete OpenAPI request/response schemas, sample curl commands, and agentic orchestration design notes.
* **[Regulatory Mapping (`docs/REGULATORY_MAPPING.md`)](docs/REGULATORY_MAPPING.md):** Mapping codebase artifacts to **IEC 62304** (Medical Device Software Lifecycle) and FDA **PCCP** (Predetermined Change Control Plan) frameworks.
* **[Interview Defense Notes (`docs/INTERVIEW_NOTES.md`)](docs/INTERVIEW_NOTES.md):** Engineering rationale, debugging logs, cold-start analyses, and architectural trade-offs.

---

## Core API Endpoints

| Method | Endpoint | Function |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness and readiness probe verifying model artifacts and vector store state. |
| `POST` | `/classify` | Classifies raw adverse event narratives into `D`, `I`, `M`, or `O` with class probabilities (ClinicalBERT on Vertex AI; TF-IDF standby on failure, reported in `backend_used`). |
| `POST` | `/retrieve` | Executes dense semantic search across indexed EU-MDR regulatory text. |
| `POST` | `/assess` | End-to-end orchestration: classification $\rightarrow$ dynamic query steering $\rightarrow$ threshold validation $\rightarrow$ structured regulatory recommendation. |

---
## Evaluation

Retrieval and citation quality are measured against a 50-item golden set ([`eval/golden_set.json`](eval/golden_set.json)): 42 regulatory questions and 8 adverse event narratives, each mapped to the EU-MDR sections accepted as correct. The expected sections were reviewed against the regulation text.

**Baseline run: 13 November 2026.** Index of 412 chunks (350 words, 50-word overlap, `all-MiniLM-L6-v2` embeddings), top 3 chunks per query, Gemini at temperature 0.

| Subset | Items | hit@1 | hit@3 | MRR | Citation accuracy | Citation precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 50 | 0.700 | 0.840 | 0.767 | 0.840 | 0.662 |
| Questions (`/retrieve` path) | 42 | 0.738 | 0.881 | 0.806 | 0.881 | 0.727 |
| Narratives (`/assess` path) | 8 | 0.500 | 0.625 | 0.562 | 0.625 | 0.385 |

* **hit@1 / hit@3:** the top chunk, or any of the top 3 chunks, comes from an expected section.
* **MRR:** mean reciprocal rank of the first expected section.
* **Citation accuracy:** the generated recommendation cites at least one expected section.
* **Citation precision:** of all sections cited, the share that were expected.

### What the baseline shows

* **Retrieval is the bottleneck, not generation.** In all 42 items where retrieval surfaced an expected section, the recommendation cited it. Citations are restricted to retrieved chunks, so citation accuracy cannot exceed hit@3.
* **Citation containment held on every item.** No citation fell outside the retrieved context (50 of 50).
* **Narratives score lower than questions.** Label-steered retrieval for adverse event narratives reaches hit@3 of 0.625, and 7 of the 8 top similarity scores fall below the 0.55 quality threshold.

### Known limitations

* **No abstention on a retrieval miss.** In all 8 misses the model still cited a retrieved section. Containment confirms that a citation comes from the retrieved context, not that it is the right provision.
* **Malfunction narratives miss Article 88.** None of the three malfunction narratives retrieved trend reporting (Article 88), and two retrieved neither Article 88 nor Article 87.
* **Recitals compete with articles.** Preamble chunks ranked first for two questions (authorised representative tasks, device risk classes).
* **Neighbouring sections win on some questions.** For example, Annex IX is retrieved instead of Article 10 for the quality management system, and Article 120 instead of Article 56 for certificate validity.
* **Extra citations lower precision.** 23 of 68 cited sections were not on the expected list; Article 2 (definitions) accounts for 6 of them.
* **Small golden set.** 50 items, with only 8 narratives, so the narrative figures move by 12.5 points per item.

### Reproduce

```bash
python scripts/evaluate_rag.py              # retrieval metrics only, no API calls
python scripts/evaluate_rag.py --with-llm   # adds citation metrics, needs GEMINI_API_KEY in .env
```

Per-item results are written to `reports/rag_eval.json`. Method notes: [`eval/README.md`](eval/README.md).

---
## Local Development & Setup

### 1. Backend Service (GCP Cloud Run)
```bash
git clone https://github.com/mukundisb/regulatory-copilot.git
cd regulatory-copilot

python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
#### 2. Run Test Suite
```bash
pytest tests/ -v
```
#### 3. Launch Local Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Local Container Test (Optional)

The repository includes a production-ready Dockerfile optimized for CPU inference and dynamic $PORT evaluation.

```bash
# Build and run locally (--target is required: the Dockerfile's last stage is the ingestion job)
docker build --target cloudrun -t regulatory-copilot .
docker run -p 8000:8000 -e PORT=8000 regulatory-copilot
```

## Deployment & Operations

### 1. Backend Service (GCP Cloud Run)
```bash
# 1. Build the API image and push it to Artifact Registry
docker build --target cloudrun -t asia-south1-docker.pkg.dev/<PROJECT_ID>/regulatory-copilot/regulatory-copilot:latest .
docker push asia-south1-docker.pkg.dev/<PROJECT_ID>/regulatory-copilot/regulatory-copilot:latest

# 2. Deploy service revision
gcloud run deploy regulatory-copilot \
    --image=asia-south1-docker.pkg.dev/<PROJECT_ID>/regulatory-copilot/regulatory-copilot:latest \
    --region=asia-south1 \
    --memory=2Gi \
    --cpu=2 \
    --allow-unauthenticated
```
#### 2. Frontend Client (Netlify/Vite)
```bash
cd frontend
# Set production backend URL in .env or Netlify Build Environment:
# VITE_API_URL=https://regulatory-copilot-xxxxxxxx-el.a.run.app

npm run build
# dist/ contains static HTML/JS/CSS assets ready for CDN deployment
```