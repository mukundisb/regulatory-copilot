REGULATORY CO-PILOT

This project is a RAG + Agentic-AI assistant over the FDA MAUDE/ ADVERSE event data project built.

Existing project for the MAUDE-NLP data adverse event portfolio present at https://github.com/mukundisb/maude-nlp-classifier

# Regulatory Copilot API

A FastAPI backend service that wraps a fine-tuned MAUDE (Manufacturer and User Facility Device Experience) machine learning pipeline to classify adverse medical device event narratives.

## Features
- **Pydantic Validation:** Rejects empty or malformed narrative payloads with HTTP 422 errors.
- **Lifespan Context Management:** Efficiently loads joblib ML model binary into memory once at application boot.
- **Environment Configuration:** Managed via `pydantic-settings` (`.env` integration).
- **Structured Logging:** Standardized request tracing and execution latency reporting.
- **Unit Test Suite:** Pytest coverage across validation, classification pipeline, and startup error contracts.

## Setup Instructions

### 1. Prerequisites
- Python 3.12+

### 2. Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
```bash
cp .env.example .env
```
- Ensure MODEL_PATH points to your model binary location (default: maude_classifier/model.joblib).

### 4. Running the API
Start the server using Uvicorn CLI:
```bash
uvicorn app:app --reload
```
Interactive API documentation will be available at http://127.0.0.1:8000/docs.

### 5. Running tests
Execute the pytest suite from project root:
```bash
pytest -v
```