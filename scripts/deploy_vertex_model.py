"""
Deployment script to register and serve Hugging Face models on Vertex AI
using the google-cloud-aiplatform SDK and official Hugging Face DLC.
"""

import os
import argparse
import logging
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pre-built Hugging Face PyTorch CPU Inference Container for Vertex AI
HF_CPU_SERVING_CONTAINER = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cpu.2-3.transformers.4-48.ubuntu2204.py311"
)


def deploy_huggingface_model(
    project_id: str,
    region: str,
    model_id: str,
    display_name: str,
    machine_type: str = "n1-standard-4",
    min_replicas: int = 0,
    max_replicas: int = 1,
    hf_task: str = "text-classification",
):
    """
    Registers a Hugging Face model in Vertex AI Model Registry and deploys
    it to an online prediction endpoint with configurable compute resources.
    """
    logger.info(f"Initializing Vertex AI SDK for project '{project_id}' in '{region}'...")
    aiplatform.init(project=project_id, location=region)

    # 1. Register Model with Hugging Face DLC
    logger.info(f"Uploading model '{model_id}' to Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=HF_CPU_SERVING_CONTAINER,
        serving_container_environment_variables={
            "HF_MODEL_ID": model_id,
            "HF_TASK": hf_task,
        },
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    logger.info(f"Model successfully registered: {model.resource_name}")

    # 2. Create Endpoint
    endpoint_display_name = f"{display_name}-endpoint"
    logger.info(f"Creating Vertex AI Endpoint '{endpoint_display_name}'...")
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        project=project_id,
        location=region,
    )
    logger.info(f"Endpoint created: {endpoint.resource_name}")

    # 3. Deploy Model to Endpoint with Custom Compute Parameters
    logger.info(
        f"Deploying to endpoint with machine_type='{machine_type}', "
        f"min_replicas={min_replicas}, max_replicas={max_replicas}..."
    )
    
    # Note: min_replica_count=0 enables the Scale-To-Zero preview
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{display_name}-deployment",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        sync=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"Endpoint ID: {endpoint.name}")
    logger.info(f"Set this in your root .env file:")
    logger.info(f"VERTEX_ENDPOINT_ID={endpoint.name}")
    logger.info("=" * 60 + "\n")

    return endpoint.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Hugging Face models to Vertex AI")
    parser.add_argument(
        "--model-id",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="Hugging Face repo ID (e.g. emilyalsentzer/Bio_ClinicalBERT or mukundisb/maude-clinicalbert)",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default="bio-clinicalbert-classifier",
        help="Display name in Vertex AI console",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.getenv("GCP_PROJECT_ID", "regulatory-copilot-506507"),
        help="GCP Project ID",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("GCP_REGION", "asia-south1"),
        help="GCP Region (e.g. asia-south1 or us-central1)",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="n1-standard-4",
        help="Supported CPU machine type (use 'n1-standard-4' instead of 'c3-highcpu-4')",
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=0,
        help="Set to 0 for Scale-To-Zero, or 1 for warm instance",
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=1,
        help="Maximum instances for autoscaling",
    )

    args = parser.parse_args()

    deploy_huggingface_model(
        project_id=args.project_id,
        region=args.region,
        model_id=args.model_id,
        display_name=args.display_name,
        machine_type=args.machine_type,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
    )