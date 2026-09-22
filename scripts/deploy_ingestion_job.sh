#!/usr/bin/env bash
set -euo pipefail

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-south1"
JOB_NAME="maude-incremental-ingestion"
SCHEDULER_NAME="maude-weekly-ingest-trigger"
SA_NAME="maude-ingestion-invoker"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
ARTIFACT_BUCKET="gs://regulatory-copilot-506507-vertex-training"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/regulatory-copilot/regulatory-copilot:ingestion"
WATERMARK_GCS_PATH="${ARTIFACT_BUCKET}/ingestion/watermark.json"

echo "=== 1. Ensuring Required GCP Services are Enabled ==="
gcloud services enable \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com

echo "=== 2. Setting Up Dedicated Least-Privilege Service Account ==="
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --description="Service account for automated openFDA MAUDE ingestion" \
        --display-name="MAUDE Ingestion Job Runner"
fi

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.invoker"

gcloud storage buckets add-iam-policy-binding "${ARTIFACT_BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud secrets add-iam-policy-binding OPENFDA_API_KEY \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

echo "=== 3. Deploying Cloud Run Job ==="
gcloud run jobs deploy "${JOB_NAME}" \
    --image="${IMAGE_TAG}" \
    --region="${REGION}" \
    --service-account="${SA_EMAIL}" \
    --tasks=1 \
    --max-retries=2 \
    --task-timeout=3600s \
    --memory=512Mi \
    --cpu=1 \
    --set-env-vars="WATERMARK_PATH=${WATERMARK_GCS_PATH},OUTPUT_DIR=/tmp/incremental" \
    --set-secrets="OPENFDA_API_KEY=OPENFDA_API_KEY:latest" \
    --command="python" \
    --args="ingestion/fetch_maude_events.py,--mode,incremental,--watermark-path,${WATERMARK_GCS_PATH},--output-dir,/tmp/incremental,--batches,5"

echo "=== 4. Provisioning Weekly Cloud Scheduler Trigger ==="
CRON_SCHEDULE="0 2 * * 0"

if gcloud scheduler jobs describe "${SCHEDULER_NAME}" --location="${REGION}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
        --location="${REGION}" \
        --schedule="${CRON_SCHEDULE}" \
        --time-zone="Etc/UTC" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email="${SA_EMAIL}"
else
    gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
        --location="${REGION}" \
        --schedule="${CRON_SCHEDULE}" \
        --time-zone="Etc/UTC" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email="${SA_EMAIL}"
fi

echo "=== Ingestion Pipeline Deployed Successfully ==="