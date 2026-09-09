"""
training/submit_vertex_job.py

Submits a containerized Vertex AI Custom Training Job using the Google Cloud AI Platform SDK.
"""

import os
import sys
from pathlib import Path
from google.cloud import aiplatform, storage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def upload_to_gcs(bucket_name: str, source_file: str, destination_blob: str):
    storage_client = storage.Client(project=settings.gcp_project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    print(f"Uploading {source_file} -> gs://{bucket_name}/{destination_blob}...")
    blob.upload_from_filename(source_file)


def submit_custom_training_job():
    project_id = settings.gcp_project_id
    region = settings.gcp_region
    bucket_name = f"{project_id}-vertex-training"
    staging_bucket = f"gs://{bucket_name}"

    container_image_uri = f"{region}-docker.pkg.dev/{project_id}/maude-artifacts/maude-training:latest"
    model_output_uri = f"{staging_bucket}/models/maude-clinicalbert"
    data_gcs_dir = f"{staging_bucket}/data"

    logger_name = "vertex_submitter"
    print(f"[{logger_name}] Initializing Vertex AI client for project '{project_id}' in region '{region}'...")
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    print(f"[{logger_name}] Uploading train/val Parquet partitions to {data_gcs_dir}...")
    upload_to_gcs(bucket_name, "data/processed/train.parquet", "data/train.parquet")
    upload_to_gcs(bucket_name, "data/processed/val.parquet", "data/val.parquet")

    job = aiplatform.CustomContainerTrainingJob(
        display_name="clinicalbert-cls-mean-concat-finetune",
        container_uri=container_image_uri,
        model_serving_container_image_uri=None,
    )

    args = [
        f"--train-data={data_gcs_dir}/train.parquet",
        f"--val-data={data_gcs_dir}/val.parquet",
        f"--output-dir={model_output_uri}",
        "--epochs=3",
        "--batch-size=32",
        "--eval-batch-size=64",
        "--learning-rate=2e-5",
        "--weight-decay=0.01",
        "--warmup-ratio=0.1",
        "--max-length=256",
    ]

    print(f"[{logger_name}] Submitting custom training job to Vertex AI...")
    custom_job = job.run(
        args=args,
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir=model_output_uri,
        sync=False,
    )

    print(f"[{logger_name}] Job successfully dispatched.")


if __name__ == "__main__":
    submit_custom_training_job()