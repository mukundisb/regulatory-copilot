# Engineering Defense & Interview Preparation Notes

This document captures candid engineering decisions, operational debugging lessons, and architectural trade-offs encountered while building and deploying the Regulatory Co-Pilot API.

---

### Q1: Why GCP Cloud Run over alternatives (GKE, AWS Lambda, standard VM)?
* **Vs. AWS Lambda (Cold Starts, Concurrency & Compute Control):**
  * **Explicit Compute Provisioning:** Lambda couples compute strictly to memory size (you cannot configure vCPUs independently). Cloud Run allows explicit, decoupled provisioning (e.g., `2 vCPU` with `2Gi` RAM) plus **Startup CPU Boost**, which accelerates heavy PyTorch and `sentence-transformers` weight imports.
  * **Multi-Concurrency Efficiency:** Lambda enforces strict **single concurrency** (1 concurrent request = 1 isolated container instance). Ten concurrent requests force 10 separate cold starts, duplicating the ~1.5 GB in-memory model footprint 10 times. Cloud Run handles multiple concurrent requests within a single warm container instance (`--concurrency`), drastically reducing memory bloat and cold-start frequency.
  * **Standard OCI Runtime:** Cloud Run runs standard Docker containers directly on dynamic `$PORT` HTTP contracts without requiring vendor-specific shims like AWS Lambda Runtime Interface Client (RIC).

* **Vs. Google Kubernetes Engine / GKE (Operational Overhead & Idle Cost):**
  * **Zero Ingress/Plumbing Maintenance:** GKE requires managing cluster control planes, node pool lifecycle upgrades, VPC networking, ingress controllers, and YAML manifest fleets. For four stateless REST endpoints, this introduces unnecessary operational maintenance.
  * **True Scale-to-Zero vs. Fixed Cluster Baseline:** Medical device post-market vigilance triage is inherently **bursty and batch-oriented** (e.g., quarterly MAUDE feed ingestion or periodic batch audit imports). GKE incurs constant minimum compute and cluster management fees 24/7 even during zero-traffic windows. Cloud Run scales down to **absolute 0 instances ($0 compute cost)** when idle and automatically scales up on demand.

* **Vs. Standard Compute Engine VMs:**
  * Avoids OS patching, manual auto-scaling group configuration, static infrastructure provisioning, and dedicated load balancer costs for a bursty microservice workload.

---

### Q2: Why regenerate synthetic models and in-memory Chroma indices in CI instead of checking in real binary files?
* **Git Anti-Patterns:** Committing multi-megabyte binary `.joblib` files and SQLite `.parquet`/Chroma indices bloats git history, slows down clones, and risks binary merge collisions.
* **Speed & Runner Determinism:** Downloading full Hugging Face weights and real EU-MDR datasets on every GitHub Actions runner boot introduces network latency and external dependency risks.
* **Contract-Driven Fixtures:** By creating `tests/conftest.py` with calibrated scikit-learn pipelines and synthetic Chroma embeddings, we test the exact **code paths, lifespan handlers, Pydantic schemas, and threshold branches** in ~7 seconds without external bloat.

---

### Q3: What are the two `/assess` decision points, and why was the first originally called "branches once" instead of "fully agentic"?
* **Decision Point 1 (Query Steering):** Takes the upstream classifier label (`D`, `I`, `M`, `O`) and prepends statutory keywords (e.g., Article 87 vigilance language for deaths vs. Article 88 CAPA language for malfunctions).
* **Decision Point 2 (Confidence Fallback Gate):** Checks if the top chunk similarity is $\ge 0.55$. If not, it executes a fallback query on the raw narrative and selects the candidate set with the higher score.
* **Why "Branches Once" vs. "Agentic":** A single static prompt rewrite (`prefix + query`) is a deterministic rule branch, not an autonomous agent. The architecture becomes *agentic* only when the second decision point evaluates runtime output quality and dynamically decides whether to execute corrective secondary actions (the fallback loop).

---

### Q4: Explain the Cloud Run port mismatch bug and why `CMD ["sh", "-c", "..."]` was required.
* **The Failure:** Local Uvicorn ran on `0.0.0.0:8000`, while Cloud Run routes traffic to port `8080` by injecting `PORT=8080`. Cloud Run's container health probe failed with `Container failed to start on port 8080`.
* **The Root Cause:** In Docker, exec form `CMD ["uvicorn", "app:app", "--port", "${PORT}"]` does **not** invoke a subshell. `${PORT}` was passed as a literal string rather than expanding to `8080`.
* **The Resolution:** Changing to `CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]` allowed the shell to evaluate `${PORT}` dynamically at runtime while retaining the local fallback to `8000`.

---

### Q5: rag_pipeline.py's query_store() filters results below min_similarity=0.45. Why would a real, clinically valid adverse-event narrative sometimes return zero results under that threshold - and why is that correct behavior rather than a retrieval failure? 
* **The Semantic Distribution Mismatch:** 
  * The indexed knowledge base contains **statutory regulatory text** (EU-MDR Regulation 2017/745: legal articles, obligations, timelines, quality management requirements).
  * Clinical adverse event narratives are written in **unstructured clinical/surgical language** (e.g., *"Patient presented with acute hypotension and hemoperitoneum following trocar placement during laparoscopic cholecystectomy"*).
  * A dense embedding model (`all-MiniLM-L6-v2`) computes semantic proximity, not clinical diagnosis. When an adverse event describes pure procedural pathology with zero lexical or conceptual overlap with statutory reporting articles or General Safety and Performance Requirements (GSPR), cosine similarity legitimately falls below `0.45` (which empirical calibration shows is the floor for meaningful regulatory relevance).

* **Why Returning Zero Results is the Correct Engineering Behavior:**
  * **Zero Hallucination / Anti-Garbage Ingestion:** In medical device compliance, **no citation is strictly better than an irrelevant citation**. If the system is forced to return the `top_k` nearest neighbors without a similarity cutoff, it will return distant, irrelevant regulatory articles (e.g., matching a surgical narrative to an unrelated IVD software rule simply because they share generic words like *"device"* or *"evaluation"*).
  * **Preserves Agentic Fallback Semantics:** In the `/assess` pipeline, returning empty or sub-threshold matches is the explicit trigger signal that allows Decision Point 2 to recognize a low-confidence retrieval pass, flag `fallback_triggered=True`, or gracefully alert a human reviewer that the clinical event requires manual regulatory analysis rather than automated rule mapping.
  * **Aligns with SaMD / Decision-Support Safety:** Under IEC 62304 / FDA guidance for Clinical Decision Support software, presenting false-positive regulatory citations creates confirmation bias for vigilance officers, risking misfiling or non-compliance. Explicitly returning `[]` communicates an honest *out-of-distribution / no-match* state.

---

### Q6: What would you build differently if starting from scratch today?
* **Hybrid Search over Pure Dense Retrieval:** Pure vector search on `all-MiniLM-L6-v2` occasionally blurs precise regulatory alphanumeric references (e.g., distinguishing "Article 87(1)(a)" from "Article 87(1)(b)"). I would implement **hybrid search (BM25 keyword search + Dense Cosine Embeddings)** with reciprocal rank fusion (RRF).
* **Asynchronous Lifespan Model Warming:** Instead of on-demand first-inference model evaluation, execute a synthetic forward pass during FastAPI's `@asynccontextmanager` startup lifecycle to pre-warm CPU caches before opening traffic ingress.
* **Asynchronous Task Queue for Assessment:** While `/classify` is sub-50ms, `/assess` executes two sequential retrieval calls and LLM formatting. In high-throughput hospital reporting feeds, this should be backed by Celery/Redis or Google Cloud Tasks with webhook callbacks.

### Q7: Why did updating `frontend/.env` fail to update the live production site until a rebuild occurred, and why didn't pushing a CORS fix in `app.py` automatically fix Cloud Run?

* **Vite's Build-Time Substitution vs. Runtime Environment Variables:**
  * In a Vite single-page application (SPA), there is no Node.js runtime executing in the client's browser.
  * During `vite build`, Vite uses static analysis to find all references to `import.meta.env.VITE_*` and **statically inlines the literal string values into the compiled JavaScript bundle** (`dist/assets/index-*.js`).
  * Changing `.env` on disk or modifying hosting dashboard environment variables post-build does not modify already-compiled static JavaScript files on the CDN. A **full rebuild (`npm run build`) is strictly required** for the bundler to inject the new API endpoint string into the output assets.

* **Decoupled Deployment: CI Gate vs. Continuous Deployment (CD):**
  * Pushing a commit containing the `app.py` CORS update to `main` triggered our GitHub Actions workflow (`.github/workflows/ci.yml`), but that workflow is strictly a **Continuous Integration (CI) test gate**, not a Continuous Deployment (CD) pipeline.
  * The production Cloud Run instance runs an immutable container image hosted in GCP Artifact Registry. 
  * Unless an automated deployment step (e.g., `google-github-actions/deploy-cloudrun`) is explicitly configured in GitHub Actions, Cloud Run continues serving the older container revision until a developer manually executes `gcloud builds submit` and `gcloud run deploy`. Recognizing the boundary between CI verification and CD release automation is essential for diagnosing stale cloud deployments.

 ### Q8: What did you measure for the Cloud Run cold-start latency, and why configure `--min-instances=1` over a purely UI-side mitigation despite the cost trade-off?

* **Measured Telemetry & Latency Profile:**
  * When scaling from absolute zero, cold-start latency measured **~12–15 seconds** for the initial `/assess` invocation (dominated by container image pull, Python runtime initialization, PyTorch weight loading, and on-demand `sentence-transformers` embedding graph construction). Subsequent warm requests execute consistently in **95.55ms/100ms** for /assess, **5.06ms/9ms** for /classify.
* **Architectural Decision (Server-Side Provisioning vs. UI Skeleton/Spinner):**
  * The cold-start bottleneck was confirmed fully fixable via server-side container provisioning (`--min-instances=1`), but keeping an instance permanently warm is an irresponsible baseline posture for a personal portfolio architecture.
  * The standing operational default is **true scale-to-zero (`--min-instances=0`)**. Setting `--min-instances=1` is treated as an intentional, ephemeral toggle executed via a single `gcloud run services update --min-instances=1` command right before active live demo or interview windows, and flipped back to 0 immediately afterward.
  *Operational toggle command:* `gcloud run services update regulatory-copilot --region=asia-south1 --min-instances=1` (and `--min-instances=0` after the session).
* **Cost vs. Availability Trade-off:**
  * Maintaining a continuous 24/7 warm instance with a 2-vCPU / 2Gi footprint costs approximately **₹10,054/month ($120/month)** (GCP Pricing Calculator estimate, asia-south1, priced [31 Aug 2026]) on Cloud Run. For a portfolio system with bursty, on-demand evaluation traffic, paying ₹10,000+ per month is unnecessary overhead.
  * Toggling min-instances only for scheduled demonstration windows yields the exact same zero-latency, sub-100ms evaluator experience during live reviews while reducing monthly idle spend to effectively **₹0**.

# ***CORE ARCHITECTURE DECISIONS***

  ### 1. Cloud Provider & Scale-to-Zero Compute Architecture (Days 15 & 25)

* **Decision Chosen:** GCP Cloud Run for serving APIs and microservices (`--min-instances=0`, scale-to-zero) with Cloud Storage (GCS) for artifact storage and Vertex AI Custom Jobs for burst training compute.
* **Real Alternatives Considered:** 
  * AWS Stack (ECS Fargate behind an Application Load Balancer + S3 + SageMaker Serverless/Endpoints).
  * Persistent Dedicated Compute (GCP GKE cluster or Compute Engine / AWS EC2).
* **Concrete Tradeoff:**
  * Market Share vs. Architectural Fit: AWS holds ~28% market share vs. GCP's ~15%. However, ECS Fargate and GKE mandate minimum cluster baseline billing (an ALB alone is ~$18–$25/month idle before traffic).
  * Scale-to-Zero vs. Idle Hosting Cost: Cloud Run scales to true zero instances ($0.00 idle cost). An always-warm baseline (`--min-instances=1`) costs ~₹10,054/month ($120/month) based on the GCP Pricing Calculator for `2 vCPU / 2GiB` RAM in `asia-south1`. The tradeoff accepted is a ~12–15 second container cold start after idle periods, mitigated operationally by a documented manual toggle command (`gcloud run services update regulatory-copilot --region=asia-south1 --min-instances=1`) prior to planned demo/audit windows.
* **Documented Evidence & Sourced Figures:**
  * Live Cloud Run configuration: Revision `regulatory-copilot-00004-55q` confirmed running on 2 vCPU and 2GiB RAM on port 8080.
  * Pricing: ~₹10,054/month always-warm vs. $0.00 idle burn with `--min-instances=0`.
  * Cold start latency: Measured at ~12–15 seconds.

---

### 2. Elasticsearch/openFDA Deep Pagination: `search_after` Cursor vs. `skip`/`limit` (Day 30)

* **Decision Chosen:** `search_after` cursor-based pagination with deterministic tie-breaking.
* **Real Alternatives Considered:** Standard offset-based pagination (`skip` and `limit`).
* **Concrete Tradeoff:**
  * openFDA's Elasticsearch backend imposes a hard result cap: `skip` is capped at 25,000, and `skip + limit` together cap accessible records at ~26,000 hits per query regardless of daily API request quotas (120,000 requests/day).
  * Offset pagination (`skip`/`limit`) forces coordinator nodes to collect and sort $N + \text{limit}$ documents in memory across shards, leading to $O(N)$ heap bloat and eventual timeout/rejection.
  * `search_after` provides stateless, constant-memory $O(\text{page\_size})$ traversal using Lucene doc-values, completely bypassing the 26,000-hit window ceiling to pull the entire corpus without coordinator heap exhaustion.
* **Documented Evidence & Sourced Figures:**
  * Ingested 160,000 raw MAUDE adverse event records in a single backfill sweep using `search_after`, yielding 129,422 deduplicated records without hitting pagination limits or dropping records.

---

### 3. Custom `cls_mean_concat` Pooling & Custom Vertex AI Serving Container (Days 31–33)

* **Decision Chosen:** Custom `cls_mean_concat` dual-pooling architecture deployed in a custom-built Docker container on Vertex AI (`maude_classifier/serve.py`), rejecting stock Hugging Face DLCs.
* **Real Alternatives Considered:**
  * Standard `[CLS]` token sequence classification head (`AutoModelForSequenceClassification`).
  * Stock Hugging Face Deep Learning Container (DLC) on Vertex AI.
* **Concrete Tradeoff:**
  * Architectural Execution: Standard `AutoModelForSequenceClassification` executes a vanilla forward pass (taking only the `[CLS]` token's final hidden state through a linear layer). Clinical safety narratives distribute critical diagnostic terminology throughout paragraphs. Dual pooling concatenates `[CLS]` with sequence mean pooling ($768 \times 2 \to 1536 \to d_{\text{out}}$), retaining global semantic context and localized clinical signals.
  * Serving Restriction: The stock Hugging Face DLC cannot execute a `cls_mean_concat` forward pass. It expects standard Hugging Face model classes and auto-loads a randomly initialized classification head if custom state dict keys are present. Serving this custom architecture required a custom container with our own FastAPI inference harness (`serve.py`).
  * Latency & Cold-Start Penalty: The custom serving container (~954MB image) pays a verified cold-start latency of **188.36 seconds (~3 min 08s)** on Vertex AI, compared to the stock DLC baseline of ~65–75 seconds. Warm round-trip latency sits at ~2900–3353 ms. This cold start was accommodated by raising `vertex_cold_start_timeout_seconds` in `config.py` and `.env` to `210.0s`.
* **Documented Evidence & Sourced Figures:**
  * Proven external benchmark: Prior 5-fold CV run verified `0.869` weighted F1 with this exact pooling layout.
  * In-repo cold start benchmark: Verified cold start of `188,359.94 ms` (188.36s) at attempt 12 of exponential backoff.

---

### 4. Model Promotion: TF-IDF v2 vs. ClinicalBERT (Day 31)

* **Decision Chosen:** Promoted **ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT` with `cls_mean_concat`) to production; retired TF-IDF v2 to a standby fallback.
* **Real Alternatives Considered:** Retaining TF-IDF v2 (which had a superior aggregate weighted-F1).
* **Concrete Tradeoff:**
  * Aggregate Regression Accepted for Safety Sensitivity:
    * **Weighted-F1:** Dropped from **0.9352** (TF-IDF v2) to **0.8372** (ClinicalBERT).
    * **Macro-F1:** Improved from **0.7138** (TF-IDF v2) to **0.7346** (ClinicalBERT).
    * **Class D (Death) F1:** Surged from **0.5909** (TF-IDF v2) to **0.7705** (ClinicalBERT).
  * Clinical Rationale: In medical device vigilance (FDA 21 CFR 803 / EU-MDR), a false negative on a death report (Class D) is a catastrophic regulatory and safety compliance failure, triggering enforcement action. TF-IDF v2 achieved a high weighted-F1 simply because it matched high-frequency lexical boilerplate on the dominant classes (Malfunction `M` and Injury `I`), but missed nuanced, implicitly worded mortality reports. ClinicalBERT was promoted specifically for safety triage sensitivity.
* **Documented Evidence & Sourced Figures (N=12,943 Held-Out Test Set):**
  * TF-IDF v2: Macro-F1 = `0.7138`, Weighted-F1 = `0.9352`, Class D F1 = `0.5909` (support = 17 in baseline).
  * ClinicalBERT: Macro-F1 = `0.7346`, Weighted-F1 = `0.8372`, Class D F1 = **`0.7705`** (precision `0.7074`, recall `0.8460`, support = 383).

---

### 5. LLM-Grounded `/assess` & Deterministic Hallucination Containment (Days 35–36)

* **Decision Chosen:** Deterministic post-generation set-difference citation pruning in code (`rag_recommender.py`), backed by prompt grounding.
* **Real Alternatives Considered:** Relying strictly on system prompt instructions ("Only cite provided text") without programmatic post-hoc verification.
* **Concrete Tradeoff:**
  * Disambiguation of Mechanisms: `RETRIEVAL_QUALITY_THRESHOLD = 0.55` in `app.py` is **not** an anti-hallucination gate; it is a vector retrieval quality gate that triggers a fallback query with the raw narrative if steered queries return low similarity.
  * Deterministic Anti-Hallucination: Grounding is enforced via a strict set-difference operation in Python:
    ```python
    valid_sections = {chunk["section"] for chunk in retrieved_chunks}
    hallucinated = set(cited_sections) - valid_sections
    # Deterministically strip invalid citations from the response:
    valid_citations = [s for s in cited_sections if s in valid_sections]
    ```
  * If hallucinated citations exist, `llm_grounding_verified` is explicitly set to `False` and a warning string is attached to the payload.
* **Documented Evidence & Sourced Figures:**
  * Tested and guarded by `tests/test_rag_recommender.py::test_fallback_sets_llm_grounding_verified_false` and `tests/test_assess_endpoint.py`. Eliminates fabricated EU-MDR section references before client serialization.

---

### 6. Experiment Tracking: MLflow vs. Vertex AI Experiments (Day 37)

* **Decision Chosen:** MLflow tracking server backed by a SQLite metadata store (`sqlite:///mlflow.db`) and local/GCS artifact storage.
* **Real Alternatives Considered:** Google Cloud Vertex AI Experiments & Metadata.
* **Concrete Tradeoff:**
  * Infrastructure Overhead vs. Industry Portability: Vertex AI Experiments integrates natively into GCP and requires less new infrastructure since training already runs on Vertex AI Custom Training. However, MLflow is the dominant, portable experiment-tracking standard explicitly requested in biopharma/pharma AI job postings.
  * Reproducibility: An external auditor or ML engineer can inspect the MLflow dashboard alone and reconstruct the model promotion:
    * Run parameters: base checkpoint (`emilyalsentzer/Bio_ClinicalBERT`), learning rate (`2e-5`), pooling architecture (`cls_mean_concat`), warmup schedule, batch size (`16`).
    * Run metrics & artifacts: Per-class metrics highlighting the Class D F1 delta (`0.7705` vs `0.5909`), confusion matrices, and saved model weights (`clinicalbert_best.pt`).
* **Documented Evidence & Sourced Figures:**
  * Historical backfill script (`scripts/backfill_mlflow.py`) retroactively logged TF-IDF v2 metrics from `reports/metrics_v2.json` and ClinicalBERT metrics from `reports/test_classification_report.json`.

---

### 7. Ingestion Cadence & Human-Gated Retraining Policy (Day 38)

* **Decision Chosen:** Weekly scheduled incremental ingestion via Cloud Scheduler triggering a Cloud Run Job (`maude-incremental-ingestion`) with persistent GCS watermark tracking (`watermark.json`), under an explicit **human-in-the-loop, no-auto-retrain** policy.
* **Real Alternatives Considered:** Continuous Training (CT) auto-triggered via GitHub Actions cron or GCS upload webhooks.
* **Concrete Tradeoff:**
  * Safety & GxP Compliance vs. Full Automation: In clinical post-market surveillance, auto-retraining on new adverse event data risks catastrophic model drift and silent regressions on safety-critical minority classes (Class D). Promotion demands human evaluation of confusion matrices and per-class recall.
  * Infrastructure Resilience: GitHub Actions runners have execution limits (6 hours), unpredictable egress, and lack native IAM Workload Identity integration. A Cloud Run Job executes natively within the GCP IAM boundary with a dedicated service account (`maude-ingestion-invoker`).
  * Watermark Ingestion: Ingestion tracks `date_received` advancing by `+1 day` (`start_date = get_next_day(last_date)`) and in-batch deduplication by `report_number` to prevent re-ingesting duplicate records across boundary dates.
* **Documented Evidence & Sourced Figures:**
  * Deployment arc: Resolved 6 distinct failure rounds during container build and deploy (handling BuildKit `--mount` stage-skipping, Docker tag collision prevention, and Secret Manager BOM stripping).
  * Live Verification: Executed live run `maude-incremental-ingestion-xhtj7` on 2026-09-22, pulling 500 records and advancing the watermark in GCS from `20240102` to `20240107`.