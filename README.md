# Regulatory Co-Pilot: AI Adverse Event Classifier & EU-MDR Retrieval Engine

[![CI Test Suite](https://github.com/mukundisb/regulatory-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundisb/regulatory-copilot/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Netlify-success?logo=netlify)](https://regulatory-copilot.netlify.app)
[![Cloud Run Deployment](https://img.shields.io/badge/GCP-Cloud%20Run%20Deployed-blue?logo=googlecloud)](https://cloud.google.com/run)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)

An automated regulatory triage and decision-support API built for medical device post-market surveillance. The system classifies unstructured FDA MAUDE adverse event narratives into statutory reporting tiers, dynamically steers regulatory vector queries across Regulation (EU) 2017/745 (EU-MDR), and applies quality gates with adaptive fallback retrieval.

---

🔗 **Live Frontend Application:** [https://regulatory-copilot.netlify.app](https://regulatory-copilot.netlify.app)  
*(Backend hosted serverless on Google Cloud Run)*



## Architectural Problem & Context

Pharma and MedTech AI engineering roles require systems that bridge statistical ML with deterministic statutory requirements. A pure LLM or generic RAG pipeline risks hallucinating compliance timelines or missing critical reporting triggers. 

This engine implements a **two-stage agentic workflow**:
1. **Upstream Classification:** Multi-class classification of medical device adverse events into MAUDE categories: **Death (`D`)**, **Injury (`I`)**, **Malfunction (`M`)**, or **Other (`O`)**.
2. **Dynamic Query Steering:** Reformulates statutory vector search queries with regulatory criteria (e.g., EU-MDR Article 87 vigilance timelines vs. Article 88 trend reporting/CAPA).
3. **Adaptive Quality Gate:** Evaluates retrieval confidence against an empirical cosine similarity cutoff ($0.55$) and triggers an automatic fallback pass if steered search underperforms.

                              +-----------------------+
                              | Raw Adverse Narrative |
                              +-----------------------+
                                          |
                                          v
                               [ POST /classify ]
                     (TF-IDF + Calibrated Logistic Regression)
                                          |
                 +------------------------+------------------------+
                 |                        |                        |
             Label: D / I              Label: M                 Label: O
                 |                        |                        |
                 v                        v                        v
         [ Article 87 Prefix ]    [ Article 88 Prefix ]     [ Raw Narrative ]
         "vigilance timelines"    "malfunction root cause"      (No prefix)
                 \                        |                        /
                  \                       |                       /
                   +----------------------+----------------------+
                                          |
                                          v
                                [ Primary ChromaDB Query ]
                                (all-MiniLM-L6-v2 Embeddings)
                                          |
                               { Score >= 0.55 Gate? }
                                 /                 \
                         YES    /                   \   NO
                               v                     v
                     [ Accept Primary ]     [ Fallback Raw Query ]
                               \                     /
                                \                   /
                                 v                 v
                           +-----------------------------+
                           |      [ POST /assess ]       |
                           | Recommendation + Citations  |
                           +-----------------------------+

---

## Documentation Links

* **[API Reference (`docs/API.md`)](docs/API.md):** Complete OpenAPI request/response schemas, sample curl commands, and agentic orchestration design notes.
* **[Regulatory Mapping (`docs/REGULATORY_MAPPING.md`)](docs/REGULATORY_MAPPING.md):** Mapping codebase artifacts to **IEC 62304** (Medical Device Software Lifecycle) and FDA **PCCP** (Predetermined Change Control Plan) frameworks.
* **[Interview Defense Notes (`docs/INTERVIEW_NOTES.md`)](docs/INTERVIEW_NOTES.md):** Engineering rationale, debugging logs, cold-start analyses, and architectural trade-offs.

---

## Core API Endpoints

| Method | Endpoint | Function |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness and readiness probe verifying model artifacts and vector store state. |
| `POST` | `/classify` | Classifies raw adverse event narratives into `D`, `I`, `M`, or `O` with calibrated probabilities. |
| `POST` | `/retrieve` | Executes dense semantic search across indexed EU-MDR regulatory text. |
| `POST` | `/assess` | End-to-end orchestration: classification $\rightarrow$ dynamic query steering $\rightarrow$ threshold validation $\rightarrow$ structured regulatory recommendation. |

---

## Deployment & Operations

### 1. Backend Service (GCP Cloud Run)
```bash
# 1. Build and push container to Artifact Registry
gcloud builds submit --tag asia-south1-docker.pkg.dev/<PROJECT_ID>/regulatory-copilot/regulatory-copilot:latest .

# 2. Deploy service revision
gcloud run deploy regulatory-copilot \
    --image=asia-south1-docker.pkg.dev/<PROJECT_ID>/regulatory-copilot/regulatory-copilot:latest \
    --region=asia-south1 \
    --memory=4Gi \
    --cpu=2 \
    --allow-unauthenticated
```
#### 2. Frontend Clinet (Netlify/Vite)
```bash
cd frontend
# Set production backend URL in .env or Netlify Build Environment:
# VITE_API_URL=[https://regulatory-copilot-xxxxxxxx-el.a.run.app](https://regulatory-copilot-xxxxxxxx-el.a.run.app)

nom run build
# dist/ contains static HTML/JS/CSS assets ready for CDN deployment
```