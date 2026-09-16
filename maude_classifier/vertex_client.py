import os
import time
import logging
from typing import Dict, Any, Optional
from google.cloud import aiplatform
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from config import settings

logger = logging.getLogger(__name__)

LABEL_MAPPING = {
    # Fine-tuned 4-class ordinal or ID head
    "LABEL_0": "M",
    "LABEL_1": "I",
    "LABEL_2": "D",
    "LABEL_3": "O",
    # Pass-through statutory strings
    "D": "D", "Death": "D",
    "I": "I", "Injury": "I",
    "M": "M", "Malfunction": "M",
    "O": "O", "Other": "O",
}

def normalize_label(raw_label: str) -> str:
    # Fail safely to a known statutory category rather than leaking raw pipeline tokens
    mapped = LABEL_MAPPING.get(raw_label)
    if not mapped:
        logger.warning(f"Unrecognized label '{raw_label}' from classifier backend. Defaulting to 'M'.")
        return "M"
    return mapped

class VertexColdStartException(Exception):
    """Raised when the Vertex AI endpoint remains provisioning beyond timeout."""
    pass


class VertexClassifierClient:
    def __init__(
        self,
        endpoint_id: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.project_id = project_id or settings.gcp_project_id
        self.location = location or settings.gcp_region
        self.endpoint_id = endpoint_id or settings.vertex_endpoint_id
        self.timeout = timeout or 60.0

        if not self.endpoint_id:
            self.endpoint_path = None
            self._endpoint = None
            logger.warning("VertexClassifierClient initialized without an active endpoint_id.")
            return
        
        aiplatform.init(project=self.project_id, location=self.location)

        self.endpoint_path = (
            f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
            if not self.endpoint_id.startswith("projects/")
            else self.endpoint_id
        )
        self._endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_path)
        logger.info(f"Initialized persistent Vertex AI client for endpoint {self.endpoint_id}")

    def predict_with_backoff(self, narrative_text: str) -> Dict[str, Any]:
        """
        Executes prediction with exponential backoff designed for Scale-To-Zero cold starts.
        Custom container expects instances=[narrative_text].
        """
        start_time = time.time()
        attempt = 0
        current_delay = settings.vertex_retry_backoff_base_seconds

        while True:
            elapsed = time.time() - start_time
            if elapsed > settings.vertex_cold_start_timeout_seconds:
                break

            try:
                attempt += 1
                logger.info(
                    f"Invoking Vertex Endpoint (Attempt {attempt}, elapsed {elapsed:.1f}s)..."
                )

                # Custom container expects flat string instances
                response = self._endpoint.predict(instances=[narrative_text])
                raw_predictions = response.predictions
                if not raw_predictions:
                    raise ValueError("Empty predictions returned from Vertex AI.")

                if not self._endpoint:
                    raise RuntimeError(
                        f"VertexClassifierClient has no active endpoint configured. "
                        f"Check settings.vertex_endpoint_id in config.py or .env."
                    )

                pred_item = raw_predictions[0]

                # Custom container schema: {"predicted_label": "D", "probabilities": {...}, "model_version": "..."}
                if isinstance(pred_item, dict) and "predicted_label" in pred_item:
                    raw_label = pred_item["predicted_label"]
                    probs = pred_item.get("probabilities", {})
                    confidence = probs.get(raw_label, 1.0)
                    model_version = pred_item.get("model_version", "custom-container")
                # Legacy / fallback handling if a list is returned
                elif isinstance(pred_item, list):
                    best = max(pred_item, key=lambda x: x.get("score", 0))
                    raw_label = best["label"]
                    confidence = best["score"]
                    probs = {item["label"]: item["score"] for item in pred_item}
                    model_version = "legacy-hf-dlc"
                else:
                    raw_label = str(pred_item)
                    confidence = 1.0
                    probs = {raw_label: 1.0}
                    model_version = "unknown"

                normalized = normalize_label(raw_label)

                return {
                    "predicted_label": normalized,
                    "confidence": round(confidence, 4),
                    "probabilities": probs,
                    "model_version": model_version,
                    "backend_used": "clinicalbert_vertex",
                }

            except ResourceExhausted as e:
                elapsed = time.time() - start_time
                if elapsed + current_delay > settings.vertex_cold_start_timeout_seconds:
                    logger.warning(
                        f"Next backoff delay ({current_delay}s) would breach timeout "
                        f"({settings.vertex_cold_start_timeout_seconds}s). Aborting backoff loop."
                    )
                    break

                logger.warning(
                    f"Vertex AI returned 429 (Container Cold Starting). Sleeping for {current_delay}s... (Error: {e.message})"
                )
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, 20.0)

            except GoogleAPICallError as e:
                logger.error(f"Vertex AI API call error: {e}")
                raise e

        raise VertexColdStartException(
            f"Vertex AI endpoint {self.endpoint_id} is still warming up after {time.time() - start_time:.1f}s. "
            "Model is provisioning from 0 replicas; please retry shortly."
        )