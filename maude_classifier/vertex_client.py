import time
import logging
from typing import Dict, Any
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
    def __init__(self):
        self.endpoint_id = settings.vertex_endpoint_id
        self.project = settings.gcp_project_id
        self.location = settings.gcp_region
        
        if not self.endpoint_id:
            raise ValueError("VERTEX_ENDPOINT_ID is not configured in settings.")

        self.endpoint_path = (
            f"projects/{self.project}/locations/{self.location}/endpoints/{self.endpoint_id}"
        )
        
        # Eagerly initialize and bind the gRPC channel on instantiation
        aiplatform.init(project=self.project, location=self.location)
        self._endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_path)
        logger.info(f"Initialized persistent Vertex AI client for endpoint {self.endpoint_id}")

    def predict_with_backoff(self, narrative_text: str) -> Dict[str, Any]:
        """
        Executes prediction with exponential backoff designed for Scale-To-Zero cold starts.
        Hugging Face DLC expects instances=[{"inputs": text}].
        """
        start_time = time.time()
        attempt = 0
        current_delay = settings.vertex_retry_backoff_base_seconds

        while attempt < settings.vertex_retry_max_attempts:
            elapsed = time.time() - start_time
            if elapsed > settings.vertex_cold_start_timeout_seconds:
                break

            try:
                attempt += 1
                logger.info(
                    f"Invoking Vertex Endpoint (Attempt {attempt}/{settings.vertex_retry_max_attempts}, elapsed {elapsed:.1f}s)..."
                )

                response = self._endpoint.predict(instances=[{"inputs": narrative_text}])
                raw_predictions = response.predictions
                if not raw_predictions:
                    raise ValueError("Empty predictions returned from Vertex AI.")

                pred_item = raw_predictions[0]
                if isinstance(pred_item, list):
                    best = max(pred_item, key=lambda x: x.get("score", 0))
                    raw_label = best["label"]
                    confidence = best["score"]
                    probs = {item["label"]: item["score"] for item in pred_item}
                elif isinstance(pred_item, dict):
                    raw_label = pred_item.get("label", "M")
                    confidence = pred_item.get("score", 0.0)
                    probs = {raw_label: confidence}
                else:
                    raw_label = str(pred_item)
                    confidence = 1.0
                    probs = {raw_label: 1.0}

                normalized = normalize_label(raw_label)

                return {
                    "predicted_label": normalized,
                    "confidence": round(confidence, 4),
                    "probabilities": probs,
                    "backend_used": "clinicalbert_vertex",
                }

            except ResourceExhausted as e:
                # 429: Scale-to-zero container spinning up
                logger.warning(
                    f"Vertex AI returned 429 (Container Cold Starting). Sleeping for {current_delay}s... (Error: {e.message})"
                )
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, 20.0)

            except GoogleAPICallError as e:
                logger.error(f"Vertex AI API call error: {e}")
                raise e

        raise VertexColdStartException(
            f"Vertex AI endpoint {self.endpoint_id} is still warming up after {elapsed:.1f}s. "
            "Model is provisioning from 0 replicas; please retry shortly."
        )