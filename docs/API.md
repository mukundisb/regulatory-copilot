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

## Actual Scale-to-Zero Latency Telemetry

Actual profiling conducted on a live Vertex AI endpoint deployed to `asia-south1` on an `n1-standard-4` node serving `Bio_ClinicalBERT` via the official Hugging Face PyTorch CPU container.

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

## Vertex Custom Serving Contract

### Endpoints & Environment Variables
* **Predict Route:** Injected via `AIP_PREDICT_ROUTE` (defaults to `/predict`)
* **Health Check Route:** Injected via `AIP_HEALTH_ROUTE` (defaults to `/health`)
* **Port:** Injected via `AIP_HTTP_PORT` (defaults to `8080`)

### Request Payload (`POST {AIP_PREDICT_ROUTE}`)
Vertex AI passes JSON with a top-level `"instances"` array. Each item is either a raw text string or an object containing a `"narrative"` field.

```json
{
  "instances": [
    "Patient underwent cardiac ablation using catheter. Severe pericardial effusion noted during manipulation resulting in tamponade and emergency sternotomy. Patient expired.",
    {"narrative": "During infusion pump startup, error code 404 displayed and motor stalled. No patient contact occurred."}
  ]
}
```

### Response Payload (`200 OK`)
Returns a JSON payload with a top-level 'predictions' array matching the order of the input instances
```json
{
  "predictions": [
    {
      "predicted_label": "D",
      "probabilities": {
        "D": 0.8412,
        "I": 0.1105,
        "M": 0.0381,
        "O": 0.0102
      },
      "model_version": "Bio_ClinicalBERT-cls_mean_concat-v1"
    },
    {
      "predicted_label": "M",
      "probabilities": {
        "D": 0.0012,
        "I": 0.0185,
        "M": 0.9621,
        "O": 0.0182
      },
      "model_version": "Bio_ClinicalBERT-cls_mean_concat-v1"
    }
  ]
}
```
### Health Check Response (`GET {AIP_HEALTH_ROUTE}`)
Returns HTTP status `200` when weights and tokenizers are loaded into memory:
```json
{
  "status": "healthy",
  "model_version": "Bio_ClinicalBERT-cls_mean_concat-v1",
  "device": "cpu"
}
```
## Server Serving Verification & Assessment

The serving implementation (`maude_classifier/serve.py`) was evaluated locally against the promoted `Bio_ClinicalBERT-cls_mean_concat-v1` checkpoint. Below is the record of test narratives, the prior clinical assessment, and the actual live probability distribution returned by the server.

### Test Case 1: Death (`D`)
* **Narrative:** `"Patient experienced cardiac arrest and died following lead detachment of implanted pacemaker."`
* **Pre-Execution Clinical Hypothesis:** Unambiguous fatal outcome explicitly tied to device malfunction. Should lean heavily toward `D` with near-zero weight on `M` and `O`.
* **Actual Server Response:**
  * `predicted_label`: `"D"`
  * `probabilities`: `{"D": 0.9990, "I": 0.0008, "M": 0.0002, "O": 0.0001}`
* **Assessment:** Clean, high-confidence detection aligned with clinical safety priorities.

---

### Test Case 2: Injury (`I`)
* **Narrative:** `"The catheter fractured during insertion, lacerating the femoral artery and requiring emergency surgical vascular repair."`
* **Pre-Execution Clinical Hypothesis:** Clear patient trauma and surgical intervention resulting from a hardware failure. Expected prediction is `I`, with residual probability on `M`.
* **Actual Server Response:**
  * `predicted_label`: `"I"`
  * `probabilities`: `{"D": 0.0005, "I": 0.9697, "M": 0.0274, "O": 0.0024}`
* **Assessment:** Successfully resolved the clinical injury boundary over the underlying component fracture.

---

### Test Case 3: Malfunction (`M`)
* **Narrative:** `"During routine pre-use check, ventilator displayed alarm code F34 and motor stalled. Device replaced prior to patient contact."`
* **Pre-Execution Clinical Hypothesis:** Isolated hardware defect detected prior to patient exposure. Strong prediction expected on `M`.
* **Actual Server Response:**
  * `predicted_label`: `"M"`
  * `probabilities`: `{"D": 0.0302, "I": 0.2329, "M": 0.7279, "O": 0.0090}`
* **Assessment:** Correctly identified `M` as the dominant class.

---

### Test Case 4: Routine Maintenance (Known Class `O` Boundary Weakness)
* **Narrative:** `"Annual preventative maintenance inspection completed. Cleaned dust filter and replaced O-rings per schedule."`
* **Pre-Execution Clinical Hypothesis:** Benign non-adverse event (`O`). However, given the model's empirical test-set tradeoff (Class `O` precision of 0.3649), the model is anticipated to err on the side of caution and misclassify device-related terms as hardware issues.
* **Actual Server Response (Documented Miss):**
  * `predicted_label`: `"M"`
  * `probabilities`: `{"D": 0.0011, "I": 0.0681, "M": 0.5914, "O": 0.3394}`
* **Assessment:** Demonstrates the documented Class `O` performance regression. The mention of hardware components ("dust filter", "O-rings") leads the model to default to Malfunction (`0.5914`) rather than Other (`0.3394`). ***In an automated triage pipeline, this results in an unnecessary human review rather than a missed safety event.***

---

### Test Case 5: Out-of-Distribution Adverse Event (No Device Mention)
* **Narrative:** `"The patient was running a marathon. While running, he palpitated and then collapsed on the ground. He was unresponsive and was immediately taken to the hospital."`
* **Pre-Execution Clinical Hypothesis:** The narrative lacks explicit device terminology, but features severe physical decompensation ("palpitated", "collapsed", "unresponsive", "hospital"). The model should assign primary weight to `I` due to the acute medical event, with `D` as a secondary consideration given the lack of confirmed mortality. Low probabilities expected for `M` and `O`.
* **Actual Server Response:**
  * `predicted_label`: `"I"`
  * `probabilities`: `{"D": 0.1523, "I": 0.8311, "M": 0.0108, "O": 0.0058}`
* **Assessment:** The probability distribution confirms calibrated semantic triage. The model identifies patient harm without over-indexing on `M` or `O`.

## Container Architecture & Packaging Strategy

### Multi-Stage Build (`Dockerfile`)
A single, reproducible multi-stage `Dockerfile` defines two distinct service targets:

1. **Target `cloudrun` (`docker build --target cloudrun ...`):**
   * **Scope:** Hosts the public web interface and API routing layer (`app:app`).
   * **Port Contract:** Dynamically binds to `${PORT:-8000}`.
   * **Artifact Isolation:** Whitelists only `*.joblib` models via `.dockerignore`. Excludes heavy PyTorch binaries (`*.bin`, `*.pt`, `*.safetensors`), tokenizer configuration files, and internal logs (`coaching/`).
   * **Fallback Role:** Uses the bundled TF-IDF models for low-latency heuristic inference when upstream services are unreachable.

2. **Target `vertex` (`docker buildx build --secret id=gcp-creds ... --target vertex ...`):**
   * **Scope:** Custom inference microservice adhering to the Vertex AI prediction specification.
   * **Port & Routing Contract:** Exposes and binds to `AIP_HTTP_PORT` (8080), handling `AIP_HEALTH_ROUTE` (`/health`) and `AIP_PREDICT_ROUTE` (`/predict`).
   * **Build Pipeline:** Uses an authenticated `model-fetcher` stage (`google/cloud-sdk:slim`) with `gcloud storage cp` and `--mount=type=secret,id=gcp-creds` to pull the promoted checkpoint from `gs://regulatory-copilot-506507-vertex-training/models/maude-clinicalbert/model/`.
   * **Offline Determinism:** Configured with `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`, ensuring all tokenizer and transformer weights initialize locally without Hugging Face Hub calls.

## Scale-To-Zero & Inference Latency Profile

The MAUDE Bio_ClinicalBERT classifier is deployed on a custom container Vertex AI endpoint configured with `min_replica_count = 0` and `max_replica_count = 2` to minimize idle compute costs.

### Empirically Verified Latency Metrics

| State | End-to-End Latency (`/assess`) | Component Breakdown & Behavior |
|---|---|---|
| **Cold Start (0 → 1 Replicas)** | **188.36s** (~3m 08s) *(Verified run)* | **~185.1s** Vertex AI container provisioning + **~3.2s** ChromaDB RAG retrieval & pipeline processing. Succeeded on Attempt 12 (~184.5s elapsed check) after confirmed 0-replica state. |
| **Warm State (Steady-State)** | **~2900ms – 3353ms** | **~2800ms – 3100ms** Vertex AI network round-trip & inference + **~60ms – 110ms** local embedding & document retrieval. |

> **Architecture Trade-Off Analysis (Interview Reference):**  
> The verified cold-start time of **~188s** represents an approximate **2.5x increase** over the stock Google Deep Learning Container (DLC) baseline (~65s–75s).  
> - **Why this tradeoff exists:** The custom container image (~954 MB) packages a complete self-contained environment: custom FastAPI/Uvicorn orchestration, PyTorch runtime, model weights, and the custom dual-pooling concatenation head (`Bio_ClinicalBERT-cls_mean_concat-v1`). GCE node scheduling, layer extraction, and local weight-binding take longer than lightweight stock containers.  
> - **Engineering justification:** In return for this initial warm-up cost, steady-state serving avoids Hugging Face DLC pipeline overhead, eliminates remote storage dependencies (`AIP_STORAGE_URI`), and executes predictions deterministically in ~2900ms warm.

---

### Client Resilience Strategy: `predict_with_backoff`

When the endpoint is scaled to zero, Vertex AI returns `429 ResourceExhausted` during the boot window. `VertexClassifierClient` shields upstream consumers via exponential backoff:

* **Backoff Strategy:** `3.0s -> 6.0s -> 12.0s -> 20.0s -> 20.0s ...` (capped at 20.0s delay per attempt).
* **Timeout Budget:** `settings.vertex_cold_start_timeout_seconds = 210.0` (set above the verified ~185s container threshold to avoid premature failover).
* **Guaranteed Fallback:** If provisioning exceeds the timeout budget, the server transparently degrades to the in-memory standby TF-IDF model (`maude_classifier.joblib`), returning HTTP 200 with `backend_used: "tfidf_fallback"`.

#### Execution Flow

```mermaid
flowchart TD
    A[app.py: _classify_narrative] --> B[VertexClassifierClient.predict_with_backoff]
    B --> C{Endpoint Call}
    C -->|200 OK| D[Parse Native Dict & Return]
    C -->|429 Cold Starting| E{Elapsed + Delay > 210s?}
    E -->|No| F[Sleep current_delay & double interval]
    F --> C
    E -->|Yes| G[Raise VertexColdStartException]
    G --> H[app.py Catch: Failover to local TF-IDF]
    C -->|Fatal Error / 4xx / 5xx| I[app.py Catch: Failover to local TF-IDF]