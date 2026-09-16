"""Deploys custom containerized Bio_ClinicalBERT model to a Vertex AI Endpoint."""

import argparse
import os
import sys
import time
from google.cloud import aiplatform

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "regulatory-copilot-506507")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "asia-south1")
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/maude-artifacts/maude-serving:v1"
MODEL_DISPLAY_NAME = "maude-clinicalbert-v1"
ENDPOINT_DISPLAY_NAME = "maude-classifier-endpoint"


def deploy_model(sync: bool = True):
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Uploading model {MODEL_DISPLAY_NAME} using custom image: {IMAGE_URI}...")
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        serving_container_image_uri=IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "AIP_HTTP_PORT": "8080",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        },
    )
    print(f"Model successfully registered: {model.resource_name}")

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
        order_by="create_time desc",
    )

    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing Endpoint: {endpoint.resource_name}")
    else:
        print(f"Creating new Endpoint: {ENDPOINT_DISPLAY_NAME}...")
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
        print(f"Endpoint created: {endpoint.resource_name}")

    print("Deploying model to endpoint (n1-standard-4)...")
    start_time = time.time()
    
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type="n1-standard-4",
        min_replica_count=0,
        max_replica_count=1,
        traffic_percentage=100,
        sync=sync,
    )
    elapsed = time.time() - start_time
    print(f"Model deployed successfully in {elapsed:.2f} seconds.")
    print(f"Endpoint Resource Name: {endpoint.resource_name}")
    print(f"Endpoint ID: {endpoint.name}")
    return endpoint


if __name__ == "__main__":
    deploy_model()