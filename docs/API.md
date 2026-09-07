# API Documentation

## Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/classify` | Classifies adverse event narratives into MAUDE reporting categories (`D`, `I`, `M`, `O`). |
| `POST` | `/retrieve` | Performs semantic search against indexed EU-MDR (Regulation EU 2017/745) sections. |
| `POST` | `/assess` | Orchestrates classification, dynamic query routing, and vector search into an actionable regulatory assessment. |

---

## 1. `POST /classify`

Classifies medical device event descriptions into MAUDE categories using a trained scikit-learn pipeline.

### Request

* **Endpoint:** `/classify`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Schema (`ClassifyRequest`)

```json
{
  "narrative": "string (required, non-empty)"
}
```
### Response
* **Status:** 200 OK
* **Schema:** (ClassifyResponse)

```json
{
  "predicted_label": "string (D | I | M | O)",
  "probabilities": {
    "D": "float",
    "I": "float",
    "M": "float",
    "O": "float"
  }
}
```

### Example
* **Request:**
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "narrative": "During laparoscopic surgery, the endo-stapler jammed and failed to deploy staples across the tissue resection line."
     }'
```
* **Response:** 200 OK
```json
{
  "predicted_label": "M",
  "probabilities": {
    "D": 0.0102,
    "I": 0.0615,
    "M": 0.9124,
    "O": 0.0159
  }
}
```
## 2. `POST /retrieve`

Performs dense vector retrieval against indexed sections of Regulation (EU) 2017/745 (EU-MDR) using ChromaDB and all-MiniLM-L6-v2 embeddings

### Request
* **Endpoint:** `/retrieve`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Schema (`ClassifyRequest`)

```json
{
  "narrative": "string (required, non-empty)"
}
```
### Response
* **Status:** 200 OK
* **Schema:** list[RetrieveResult]
```json
[
  {
    "chunk_id": "string",
    "section": "string",
    "text": "string",
    "similarity_score": "float"
  }
]
```
### Example
* **Request:**
```bash
curl -X POST "http://127.0.0.1:8000/retrieve" \
     -H "Content-Type: application/json" \
     -d '{
       "narrative": "In what language must device labels, packaging information, and instructions for use be provided?"
     }'
```
* **Response:** 200 OK
```json
[
  {
    "chunk_id": "doc_chunk_283",
    "section": "ANNEX I - GENERAL SAFETY AND PERFORMANCE REQUIREMENTS",
    "text": "[ANNEX I - GENERAL SAFETY AND PERFORMANCE REQUIREMENTS]\nREQUIREMENTS REGARDING THE INFORMATION SUPPLIED WITH THE DEVICE 23. Label and instructions for use 23.1. General requirements regarding the information supplied by the manufacturer Each device shall be accompanied by the information needed to identify the device and its manufacturer, and by any safety and performance information relevant to the user...",
    "similarity_score": 0.6947
  },
  {
    "chunk_id": "doc_chunk_74",
    "section": "Article 16 - Cases in which obligations of manufacturers apply to importers, distributors or other persons",
    "text": "[Article 16 - Cases in which obligations of manufacturers apply to importers, distributors or other persons]\n...distributor or importer carrying out translation shall provide the manufacturer and the competent authority with a sample or mock-up of the relabelled or repackaged device, including any translated label and instructions for use...",
    "similarity_score": 0.6117
  }
]
```
## 3. `POST /assess`

Orchestrates classification, dynamic query routing, quality-checked retrieval, and rule-based recommendation drafting into an actionable regulatory assessment.

#### Two-Stage Agentic Workflow

1. **Decision Point 1 (Dynamic Steering):** Upstream classification predicts event reporting category (`D`, `I`, `M`, `O`) and reformulates the primary vector search query with regulatory criteria.
2. **Decision Point 2 (Adaptive Fallback):** Evaluates the top chunk's similarity score against an empirical threshold (`0.55`). If the primary steered query produces a weak match (`< 0.55`), the orchestrator triggers a fallback retrieval using the raw unmodified narrative and selects whichever candidate set yields the stronger match.

| Category | Primary Steering Strategy | Target Framework | Fallback Trigger Condition |
| :--- | :--- | :--- | :--- |
| **`D` / `I`** | `serious incident reporting vigilance timelines manufacturer obligations <narrative>` | Article 87 (Vigilance reporting timelines) | Primary Top Score < `0.55` |
| **`M`** | `device malfunction root cause analysis trend reporting corrective action <narrative>` | Articles 88 & 89 (Trend reporting & CAPA) | Primary Top Score < `0.55` |
| **`O`** | `<narrative>` (Unmodified) | Annex I (GSPR) & Article 16 (General provisions) | Primary Top Score < `0.55` |

_Note: retrieval is similarity-based (via ChromaDB embeddings), not a hard-coded mapping — the query is steered toward the relevant EU-MDR area, but the exact article/section returned can vary by narrative and isn't guaranteed. This endpoint is a decision-support aid, not a certified legal or regulatory compliance determination; outputs should be reviewed by qualified personnel before use in an actual regulatory submission._

### Request
* **Endpoint:** `/assess`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Schema (`ClassifyRequest`)

```json
{
  "narrative": "string (required, non-empty)"
}
```
### Response
* **Status:** 200 OK
* **Schema:** (AssessResponse)

```json
{
  "predicted_label": "string (D | I | M | O)",
  "confidence": "float",
  "retrieval_query_used": "string",
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "section": "string",
      "text": "string",
      "similarity_score": "float"
    }
  ],
  "recommendation": "string",
  "fallback_triggered": "boolean"
}
```
### Example
* **Request:**
```bash
curl -X POST "http://127.0.0.1:8000/assess" \
     -H "Content-Type: application/json" \
     -d '{
       "narrative": "Patient experienced acute cardiac arrest following catheter balloon rupture during angioplasty procedure."
     }'
```
* **Response:** 200 OK
```json
{
  "predicted_label": "D",
  "confidence": 0.9428,
  "retrieval_query_used": "serious incident reporting vigilance timelines manufacturer obligations Patient experienced acute cardiac arrest following catheter balloon rupture during angioplasty procedure.",
  "retrieved_chunks": [
    {
      "chunk_id": "doc_chunk_180",
      "section": "Article 87 - Reporting of serious incidents and field safety corrective actions",
      "text": "[Article 87 - Reporting of serious incidents and field safety corrective actions]\n1. Manufacturers of devices made available on the Union market... shall report to the relevant competent authorities: (a) any serious incident involving devices made available on the Union market, except expected side-effects which are clearly documented in the product information...",
      "similarity_score": 0.7412
    },
  ],
  "recommendation": "Event classified as 'D' (confidence: 94.28%). Primary regulatory basis: Article 87 - Reporting of serious incidents and field safety corrective actions. Recommended Action: Mandatory vigilance reporting required. Initiate immediate risk assessment and submit report within strict statutory timelines (within 2 to 10 days depending on public health threat severity.",
  "fallback_triggered": false
}
```

## Design Notes: Agentic Orchestration & Adaptive Retrieval

The `/assess` endpoint implements a two-stage agentic workflow rather than a static pipeline. The **first decision point** uses the classification model's predicted label (`D`, `I`, `M`, `O`) to dynamically steer the primary vector query toward statutory frameworks (e.g., injecting vigilance timeline criteria for serious incidents or CAPA root-cause keywords for malfunctions). The **second decision point** evaluates the retrieval quality of the primary pass against an empirical similarity threshold (`0.55`, calibrated against observed true matches ranging between `0.6117` and `0.7412` versus semantic noise at `< 0.45`). If the steered query produces weak similarity scores, the orchestrator triggers a fallback pass using the raw, unmodified narrative and selects the candidate set with the superior top score. Encapsulating this branching into dedicated helper functions (`build_retrieval_query` and `retrieve_with_fallback`) preserves clean unit testability, isolates failure domains, and ensures the system adaptively recovers from over-constrained prompt steering.

## Continuous Integration & Test Suite Design Note

### 1. Architectural Motivation
The CI pipeline (`.github/workflows/ci.yml`) serves as an automated quality and regression gate across pull requests and pushes to `main`. Machine Learning and RAG systems present unique CI challenges—specifically large binary weight footprints (`.joblib` models), heavyweight embedding models (`sentence-transformers`), and persistent local vector indices (`chroma_db/`).

Rather than committing gigabytes of static binaries to version control or triggering expensive external asset downloads on every test run, the test harness relies on **deterministic synthetic fixtures** and **ephemeral test-state bootstrapping**.

---

### 2. Artifact & Dependency Strategy

| Component | Production Runtime | CI Test Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| **Classifier Model** | Pre-trained Scikit-Learn pipeline (`maude_classifier.joblib`) | Session-scoped synthetic `Pipeline` generated on-the-fly via `tests/conftest.py` | Avoids storing multi-megabyte binary artifacts in Git while verifying end-to-end FastAPI lifespan loading, inference schema, and class probability distribution invariants (`D`, `I`, `M`, `O`). |
| **PyTorch Runtime** | Containerized CPU-only wheel | CPU-only wheel via PyTorch index (`--index-url https://download.pytorch.org/whl/cpu`) | Prevents downloading default 2GB+ CUDA dependencies, cutting runner setup time by ~75%. |
| **Vector Index (`ChromaDB`)** | Persistent disk store (`chroma_db/`) seeded from EU-MDR corpus | In-process initialized store seeded with domain-anchored text chunks during test setup | Validates real distance metric calculations and similarity threshold gates (`1 - cosine_distance >= 0.45`) without coupling tests to external disk states. |

---

### 3. Core Decision Invariants Tested

The CI suite validates critical runtime contracts across the four primary endpoints (`/health`, `/classify`, `/retrieve`, `/assess`):

1. **Lifespan Startup Resilience (`test_missing_model_file_fails_startup`):**
   * Verifies that the FastAPI application fails fast with a clean `FileNotFoundError` if the configured model artifact path is invalid.
2. **Payload Schema Validation (Pydantic V2):**
   * Enforces 422 Unprocessable Entity responses on empty strings, non-string payloads, and malformed JSON bodies across all ingress routes.
3. **Decision Point 1: Label-Steered Query Reformulation:**
   * Validates that predicted class branches dynamically inject regulatory keyword anchors:
     * **Death (`D`) / Injury (`I`):** `serious incident reporting vigilance timelines manufacturer obligations`
     * **Malfunction (`M`):** `device malfunction root cause analysis trend reporting corrective action`
     * **Other (`O`):** Bypasses prefix injection and retains raw user narrative.
4. **Decision Point 2: Similarity Score Quality Gate & Fallback Evaluation:**
   * Asserts the minimum similarity cutoff ($0.55$) behavior:
     * If primary query similarity $\ge 0.55$, fallback search is bypassed (`fallback_triggered=False`).
     * If primary query similarity $< 0.55$, fallback raw narrative query is executed and the higher-scoring chunk set is returned.
5. **Real Store & E2E Integration:**
   * Exercises sentence-transformers vector generation, Chroma querying, distance-to-similarity conversion, and structural header parsing (`ANNEX I`, `Article 87`, `Article 88`) on live execution graphs.

---

### 4. Running the Suite Locally vs. CI

* **Run all tests locally:**
  ```bash
  pytest tests/ -v
  ```

* **Run with captured logging / stdout:**
  ```bash
  pytest tests/ -v -s
  ```

* **Run specific branch unit tests:**
  ```bash
  pytest tests/test_classifier.py -k "test_assess_decision_branch_reformulation" -v
  ```

## Empirical Scale-to-Zero Latency Telemetry

Empirical profiling conducted on a live Vertex AI endpoint deployed to `asia-south1` on an `n1-standard-4` node serving `Bio_ClinicalBERT` via the official Hugging Face PyTorch CPU container.

### 1. Measured Performance States

| System State | Observed Latency | Mechanism & Observations |
|---|---|---|
| **True Cold Start (0 Replicas)** | **~65–75 seconds** | VM allocation, Docker container pull, and PyTorch weight initialization. First request was rejected with 429 across 6 backoff cycles ($3\text{s} \rightarrow 6\text{s} \rightarrow 12\text{s} \rightarrow 20\text{s} \rightarrow 20\text{s} \rightarrow 20\text{s} = 64.1\text{s}$) and completed successfully at $t \approx 68\text{s}$. |
| **Quasi-Warm (Container Active, JIT Compiling)** | **~4.2–4.9 seconds** | Per-request client re-instantiation, TLS handshakes, and unpinned connection overhead. |
| **True Warm (Lifespan Persistent gRPC)** | **310 ms** | Pre-warmed `VertexClassifierClient` with pooled gRPC channel inside FastAPI process memory. |
| **Failover Fallback** | **~3 ms** | Local scikit-learn TF-IDF model served instantly if Vertex AI exceeds the 90.0s deadline. |

### 2. Architectural Takeaways
* A 20–25s target was a speculative underestimate; full container provisioning in `asia-south1` requires a **90.0s backoff budget**.
* Capping backoff at 6 attempts prematurely terminated the client right before the container finished spinning up.
* **Option A Fallback Validation:** The automated failover to local TF-IDF is strictly required for consumer-facing availability during the 70s provisioning window.

### 3. Server-Side Execution Telemetry (Cloud Logging)

Extracted from container logs on `asia-south1`:
* **Model Download (HF Hub):** ~14.0 s
* **PyTorch Graph & CPU Init:** ~1.56 s
* **Container Ready Timestamp:** `05:13:38 UTC`
* **Server-Side Forward Pass Duration (`POST /predict`):** **239.04 ms**
* **Total Warm Client Round-Trip:** **310.03 ms** (Delta: ~71 ms transit/serialization)
* **Head Verification:** Confirmed `BertForSequenceClassification` requires fine-tuned weights (`mukundisb/maude-clinicalbert`) to replace the default randomly-initialized binary head (`['classifier.weight', 'classifier.bias']`).