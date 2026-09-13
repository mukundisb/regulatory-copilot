# ==============================================================================
# Base Stage: Common Python Runtime & Dependencies
# ==============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime OpenMP (libgomp1) for CPU PyTorch execution and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch wheel to minimize container footprint
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Model Fetcher Stage: Pulls Promoted Checkpoint from GCS Source of Truth
# ==============================================================================
FROM google/cloud-sdk:slim AS model-fetcher

ARG GCS_MODEL_URI="gs://regulatory-copilot-506507-vertex-training/models/maude-clinicalbert/model"
WORKDIR /model

# Point authentication environment variables to the mounted secret path
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/creds/adc.json \
    CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/creds/adc.json

RUN --mount=type=secret,id=gcp-creds,target=/tmp/creds/adc.json \
    gcloud storage cp -r "${GCS_MODEL_URI}/*" /model/


# ==============================================================================
# Target 1: Cloud Run (Web API & Fallback Classifier)
# ==============================================================================
FROM base AS cloudrun

# Copy source tree (strictly excludes coaching/, *.bin, and tokenizer configs via .dockerignore)
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]


# ==============================================================================
# Target 2: Vertex AI Custom Prediction Serving
# ==============================================================================
FROM base AS vertex

ENV AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Copy serving source code only
COPY maude_classifier/ /app/maude_classifier/

# Populate model directory from GCS fetcher
RUN mkdir -p /app/maude_classifier/model
COPY --from=model-fetcher /model/ /app/maude_classifier/model/

EXPOSE 8080

ENTRYPOINT ["python", "-m", "uvicorn", "maude_classifier.serve:app", "--host", "0.0.0.0", "--port", "8080"]