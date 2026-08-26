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