from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Literal, Optional

class Settings(BaseSettings):
    app_name: str = "Regulatory Co-pilot"
    model_path: str = "maude_classifier/model/maude_classifier.joblib"
    host: str = "127.0.0.1"
    port: int = 8000

# Model & Classifier Backend Strategy
    # Options: 'tfidf' (local fast path) or 'clinicalbert_vertex' (Vertex AI managed endpoint)
    classifier_backend: Literal["tfidf", "clinicalbert_vertex"] = "tfidf"

    # Vertex AI Endpoint Configuration
    gcp_project_id: str = "regulatory-copilot-506507"
    gcp_region: str = "asia-south1"
    vertex_endpoint_id: str = "8682713582174994432"

    # Vertex AI Scale-To-Zero Backoff & Cold Start Tuning
    vertex_retry_max_attempts: int = 6
    vertex_retry_backoff_base_seconds: float = 3.0
    vertex_cold_start_timeout_seconds: float = 90.0

    # OpenFDA Ingestion (Optional fields so they are recognized)
    openfda_base_url: Optional[str] = None
    openfda_api_key: Optional[str] = None
    openfda_limit: Optional[int] = None
    openfda_output_path: Optional[str] = None

    # CRITICAL: Allow extra environment variables in .env without crashing
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()