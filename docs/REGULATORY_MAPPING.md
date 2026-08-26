# Regulatory Mapping: IEC 62304 & FDA PCCP Alignment

## 1. Executive Summary & Regulatory Classification Context

This document establishes concrete traceability between the **Regulatory Co-Pilot API** codebase and formal medical device software lifecycle standards (**IEC 62304:2006/AMD 1:2015**) and the FDA's **Predetermined Change Control Plan (PCCP)** framework for AI/ML-enabled device software functions (SaMD / SiMD).

### Core Regulatory & Safety Features

* **Clinical Decision Support & Vigilance Triage**  
  Functions as an automated triage aid for clinical workflows.

* **EU MDR 2017/745 Qualification**  
  Classified under **Annex VIII Rule 11** for decision-support screening.

* **IEC 62304 Software Safety Class**  
  Assigned **Class B** software safety classification.

* **Human-in-the-Loop Risk Mitigation**  
  Addresses non-serious injury potential from triage delays via mandatory human review guardrails.

---

## 2. IEC 62304 Software Lifecycle Mapping

### 2.1 Software Verification Strategy (IEC 62304 Clause 5.6 & 5.7)
IEC 62304 distinguishes between unit verification (Clause 5.5.5), integration testing (Clause 5.6), and system verification (Clause 5.7). The test suite (`tests/test_classifier.py`) implements this tiered verification boundary:

| Verification Level | Standard Clause | Repo Implementation Artifact | Technical Rationale & Isolation Strategy |
| :--- | :--- | :--- | :--- |
| **Unit Verification (Isolated Logic)** | **5.5.5** | `test_assess_decision_branch_reformulation`<br>`test_assess_triggers_fallback_when_steered_query_scores_low`<br>`test_assess_fallback_bypassed_on_strong_primary_score` | Uses `monkeypatch` to force static classifier predictions and mock `query_store` scores. Proves deterministic query steering and threshold branch execution without external runtime noise. |
| **Software Integration Testing** | **5.6.2** | `test_classify_valid_input`<br>`test_retrieve_valid_query_mocked`<br>`test_missing_model_file_fails_startup` | Validates inter-module boundaries: Pydantic schema validation (`ClassifyRequest`), FastAPI lifespan boot checks, and error bubbling (`RuntimeError` on database crash). |
| **System E2E Verification** | **5.7.1** | `test_retrieve_e2e_real_store`<br>`test_assess_e2e_real_pipeline`<br>`test_assess_e2e_malfunction_branch_real_pipeline`<br>`test_assess_e2e_real_fallback_evaluation` | Executes live forward passes across the calibrated pipeline (`maude_classifier.joblib`), `sentence-transformers` embedding generation, and `ChromaDB` cosine retrieval. |

---

### 2.2 Traceability Matrix (IEC 62304 Clause 5.1.1 & 7.3.3)
IEC 62304 requires bi-directional traceability linking Software Requirements (SR) to Software Architecture/Design (SD), Implementation (SI), and Verification (SV):

```mermaid
flowchart LR
    SR["Requirement (SR)"] --> SD["Architecture (SD)"] --> SI["Implementation (SI)"] --> SV["Verification Test (SV)"]
```


* **SR-01 (Adverse Event Triage):** The system shall classify adverse event narratives into MAUDE statutory categories (`D`, `I`, `M`, `O`).
  * **Design & Spec:** `docs/API.md` Section 1 (`POST /classify`).
  * **Implementation:** `app.py::classify()` invoking `maude_classifier/model/maude_classifier.joblib`.
  * **Structured Telemetry:** `event="classify_success"`, capturing predicted label and probability vector.
  * **Verification:** `tests/test_classifier.py::test_classify_valid_input`.

* **SR-02 (Statutory Vigilance Steering):** Classifications of `D` or `I` shall augment retrieval queries with MDR Article 87 reporting criteria.
  * **Design & Spec:** `docs/API.md` Section 3 (Two-Stage Agentic Workflow, Decision Point 1).
  * **Implementation:** `app.py::build_retrieval_query()` injecting `"serious incident reporting vigilance timelines manufacturer obligations"`.
  * **Verification:** `tests/test_classifier.py::test_assess_decision_branch_reformulation[D-...]`.

* **SR-03 (Adaptive Retrieval Quality Gate):** If primary search yields top similarity $< 0.55$, fallback search on raw narrative shall execute.
  * **Design & Spec:** `docs/API.md` Section 3 (Decision Point 2 table).
  * **Implementation:** `app.py::retrieve_with_fallback()` evaluating cosine threshold `0.55`.
  * **Structured Telemetry:** `event="assess_success"`, emitting `fallback_triggered=True/False`.
  * **Verification:** `tests/test_classifier.py::test_assess_triggers_fallback_when_steered_query_scores_low`.

---

## 3. FDA Predetermined Change Control Plan (PCCP) Specification

Under the FDA's guidance *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI/ML-Enabled Device Software Functions*, manufacturers must pre-specify the **Scope of Modifications (Description of Modifications)** and the **Verification & Validation Protocol (Modification Protocol)**.

If this system's models are retrained or swapped post-deployment, the following PCCP controls apply:

### 3.1 Description of Planned Modifications (What is allowed to change)
* **Model Component 1 (`maude_classifier.joblib`):** Hyperparameter tuning or periodic retraining on newly released FDA MAUDE / CDRH quarterly adverse event databases. Permitted architectures: Linear SVM, Logistic Regression, or DistilBERT fine-tuning.
* **Model Component 2 (Vector Embeddings):** Migration of `all-MiniLM-L6-v2` to domain-specific biomedical embedding models (e.g., `BioLinkBERT` or `Med-CPT`) to improve semantic separation on clinical jargon.

### 3.2 Modification Protocol (How changes are validated)
No formal calibration run or golden validation set exists yet for this
project, so the specific numeric acceptance thresholds below are not
established - this section instead specifies *what categories* of
metrics a real PCCP would need to pre-specify, and how they would be
derived once a validation set exists.

1. **Classifier performance invariants:** a PCCP would require a held-out,
   version-controlled validation set (not yet built) and would pre-specify
   a minimum acceptable macro-F1 across D/I/M/O, plus a minimum recall
   floor specifically for the `D` (Death) class - recall matters more
   than precision here because a missed mortality-adjacent case is a
   worse failure than a false positive. Actual numeric floors would be
   set from the current model's baseline performance on that validation
   set, not assumed in advance.
2. **Retrieval quality invariants:** a re-calibrated minimum for the
   `RETRIEVAL_QUALITY_THRESHOLD` fallback gate (currently 0.55, set
   empirically) would need to be re-validated any time the embedding model changes, since cosine similarity scores are not comparable across different embedding geometries.
3. **Automated regression gate:** full `.github/workflows/ci.yml`
   execution (21 tests as of this writing) passing is a necessary but
   not sufficient condition - it verifies code paths, not statistical
   model quality.

---

## 4. Gap Assessment: Real vs. Simulated Regulatory Posture

To maintain engineering candor under technical audit, the following boundaries distinguish this reference architecture from a certified commercial Medical Device Software submission:

| Regulatory Area | Full Commercial Regulatory Bar (FDA 21 CFR 820 / MDR) | Current Repository Status | Gap & Mitigation |
| :--- | :--- | :--- | :--- |
| **Design History File (DHF)** | Formal design inputs, design outputs, design reviews, and DHF sign-off per 21 CFR 820.30. | Lightweight documentation in `docs/` and git commit history. | Sufficient for technical architectural defense; lacks formal Quality Management System (QMS) audit trail. |
| **Risk Management (ISO 14971)** | Formal Hazard Analysis, FMEA, Risk Management File, and ALARP mitigations. | Fallback search logic (`0.55` threshold) acts as a functional risk mitigation. | No formal FMEA risk matrix linking failure modes to clinical harm severities. |
| **Clinical Validation** | Prospective multi-center clinical evaluation comparing model triage against board-certified regulatory affairs professionals. | Offline evaluation on historical MAUDE narratives and automated pytest assertions. | Outputs are marked as non-certified decision support; mandatory human-in-the-loop review required before regulatory filing. |