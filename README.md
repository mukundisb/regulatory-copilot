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

### 3. Environment Variables (optional)
The app runs out of the box using the defaults in `config.py` — no `.env` file is required.

To override any setting (e.g. a different model file location, or a different host/port for deployment), create a `.env` file in the project root:

MODEL_PATH=maude_classifier/model/maude_classifier.joblib
HOST=127.0.0.1
PORT=8000
APP_NAME=Regulatory Co-pilot API

Only the variables you want to change need to be set — anything omitted falls back to its default.

### 4. Running the API
Start the server using Uvicorn CLI:
```bash
uvicorn app:app --reload
```
Interactive API documentation will be available at http://127.0.0.1:8000/docs.

For a full written reference — request/response examples and design notes for `/classify`, `/retrieve`, and `/assess` — see [`docs/API.md`](docs/API.md).

### 5. Running tests
Execute the pytest suite from project root:
```bash
pytest -v
```