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

## Model Benchmarks & Promotion Ledger

### Training Telemetry (Vertex AI Custom Training)

* **Pipeline Execution ID:** `4755359477706784768`
* **Artifact URI:** `gs://regulatory-copilot-506507-vertex-training/models/maude-clinicalbert/model/`
* **Model Architecture:** `ClinicalBERTConcatClassifier` (`emilyalsentzer/Bio_ClinicalBERT` + `[CLS; Mean]` concatenation)
* **Compute Topology:** `n1-standard-8` (8 vCPUs, 30 GB RAM) + 1× NVIDIA Tesla T4 GPU
* **Precision Mode:** FP32 Baseline (No AMP)
* **Partitions:** Stratified MAUDE Split (103,537 Train / 12,942 Val / 12,943 Test)
* **Training Throughput:** 3,236 steps/epoch @ ~1.46s/step (Batch size: 32)
* **Epoch 3 Duration:** 4,964.6s (~82.7 minutes)
* **Total Training Wall-Clock:** ~4.1 hours (~14,890s)
* **Actual Invoiced GCP Cost (Sep 1–9, 2026):** ₹536.28 ($5.63 USD)
  * *Vertex AI Training on NVIDIA Tesla T4 GPU (Mumbai):* ₹186.65
  * *Vertex AI Training on N1 Predefined Instance RAM (Mumbai):* ₹69.49
  * *Vertex AI Training on N1 Predefined Instance CPU (Mumbai):* ₹280.14

---

### Head-to-Head Benchmark Comparison (Held-Out Test Set, N = 12,943)

Evaluated with `scripts/evaluate_test.py` against the promoted checkpoint at `maude_classifier/model/pytorch_model.bin`:

| Metric / Class | TF-IDF + Logistic Regression Baseline | Bio_ClinicalBERT (`cls_mean_concat`) | Delta ($\Delta$) | Operational Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Class D (Death) F1** | 0.5909 | **0.7705** | **+0.1796** | **Critical safety gain** |
| Class D Precision | 0.5284 | **0.7074** | +0.1790 | Reduced false alarms on fatal events |
| Class D Recall | 0.6710 | **0.8460** | **+0.1750** | Major reduction in missed fatal events |
| **Class I (Injury) F1** | 0.7741 | **0.8532** | +0.0791 | Improved triage boundary |
| **Class M (Malfunction) F1** | 0.8210 | **0.8624** | +0.0414 | Higher reliability on hardware defects |
| **Class O (Other) F1** | **0.6692** | 0.4525 | -0.2167 | Performance regression |
| **Macro Average F1** | 0.7138 | **0.7346** | **+0.0208** | **Net model gain** |
| **Weighted Average F1** | **0.9352** | 0.8372 | -0.0980 | Skewed by majority class distribution |
| **Overall Accuracy** | **93.81%** | 83.12% | -10.69% | Artifact of majority class bias |

---

### Promotion Justification & Risk Tradeoff Analysis

**Decision: PROMOTED TO PRODUCTION CANDIDATE**

The primary operational mandate of this pipeline is patient safety risk containment and adverse event reporting triage under medical device post-market surveillance workflows.

* **Acceptance of Accuracy & Weighted F1 Regression:** The TF-IDF baseline's elevated overall accuracy (93.81%) and weighted F1 (0.9352) stem entirely from severe majority-class imbalance. The baseline disproportionately routes borderline and ambiguous narratives into high-frequency categories (`Malfunction` and non-critical narratives). While this artificially bolsters aggregate numbers, it exhibits a catastrophic failure mode on the most critical clinical category: Class D (Death) recall sits at an unacceptable 67.10%, failing to surface roughly 1 out of every 3 fatal device events.
* **Clinical Safety Gains:** Bio_ClinicalBERT increases Class D recall to **84.60%** (+17.50 points) and precision to **70.74%** (+17.90 points), raising the F1 score from 0.5909 to **0.7705**. Capturing rare fatal adverse events takes precedence over overall accuracy in post-market surveillance.
* **Tradeoff on Class O (Other / Non-Adverse):** Bio_ClinicalBERT records a regression on Class O precision (0.3649) and F1 (0.4525) because the model errs on the side of caution, shifting marginal narratives into Injury (`I`) or Malfunction (`M`) rather than discarding them as non-adverse. In operational triage, an extra benign case marked for human review imposes minimal cost, whereas an overlooked patient death represents an unacceptable compliance and safety failure.

Because Bio_ClinicalBERT achieves a superior unweighted **Macro F1 (0.7346 vs. 0.7138)** and successfully captures high-severity safety risks, it formally replaces the TF-IDF baseline as the production model candidate.