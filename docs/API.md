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

Executes an orchestration pipeline where the classification output functions as an agentic decision point to dynamically formulate targeted regulatory queries against the vector store before synthesizing a deterministic compliance recommendation.

| Predicted Label | Event Type | Query Formulation Strategy | Target EU-MDR Framework |
| :--- | :--- | :--- | :--- |
| D / I | Death / Injury | Serious incident reporting vigilance timelines manufacturer obligations <narrative> | Article 87 (Vigilance reporting timelines & statutory obligations) |
| M | Malfunction | Device malfunction root cause analysis trend reporting corrective action <narrative> | "Articles 88 & 89 (Trend reporting, CAPA & FSCA investigations)" |
| O | Other / Inquiry | <narrative> (Unmodified) | Annex I (GSPR) & Article 16 (General translation/labeling) |

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
  "recommendation": "string"
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
    {
      "chunk_id": "doc_chunk_181",
      "section": "Article 87 - Reporting of serious incidents and field safety corrective actions",
      "text": "[Article 87]\n3. Manufacturers shall report any serious incident immediately after they have established the causal relationship between the device and the incident or that a causal relationship is reasonably possible and, in the event of a serious public health threat, not later than 2 days...",
      "similarity_score": 0.6835
    }
  ],
  "recommendation": "Event classified as 'D' (confidence: 94.28%). Primary regulatory basis: Article 87 - Reporting of serious incidents and field safety corrective actions. Recommended Action: Mandatory vigilance reporting required. Initiate immediate risk assessment and submit report within strict statutory timelines (within 2 to 10 days depending on public health threat severity)."
}
```