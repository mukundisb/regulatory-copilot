from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class Settings(BaseSettings):
    app_name: str = "Regulatory Co-pilot"
    model_path: str = "maude_classifier/model/maude_classifier.joblib"
    host: str = "127.0.0.1"
    port: int = 8000

    # Model & Classifier Backend Strategy
    # Options: 'tfidf' (local fast path) or 'clinicalbert_vertex' (Vertex AI managed endpoint)
    classifier_backend: Literal["tfidf", "clinicalbert_vertex"] = "clinicalbert_vertex"

    # Vertex AI Endpoint Configuration
    gcp_project_id: str = "regulatory-copilot-506507"
    gcp_region: str = "asia-south1"
    vertex_endpoint_id: str = "4702516673997963264"

    # Vertex AI Scale-To-Zero Backoff & Cold Start Tuning
    vertex_retry_backoff_base_seconds: float = 3.0
    vertex_cold_start_timeout_seconds: float = 210.0

    # Gemini pricing, USD per 1M tokens, used only to estimate cost per request in telemetry.
    # gemini-3.8-flash paid tier through 31 Dec 2026. From 1 Jan 2027: 1.50 input, 7.50 output.
    # Output price includes thinking tokens.
    gemini_input_price_per_mtok_usd: float = 0.75
    gemini_output_price_per_mtok_usd: float = 3.75

    # LLM provider for the grounded recommendation: 'gemini' (Google) or 'azure_openai' (Azure OpenAI)
    llm_provider: Literal["gemini", "azure_openai"] = "gemini"
    gemini_model: str = "gemini-3.8-flash"

    # Azure OpenAI (used when llm_provider = 'azure_openai'). The API key is read from AZURE_OPENAI_API_KEY.
    # Endpoint is the resource address, e.g. https://<resource-name>.openai.azure.com
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None      # the deployment name you chose, not the model name
    azure_openai_max_completion_tokens: int = 4000     # reasoning models spend part of this on reasoning
    azure_openai_timeout_seconds: float = 60.0
    # Retries on HTTP 429 (rate limit) and 503, waiting as long as Azure's Retry-After says (capped).
    azure_openai_max_retries: int = 3
    azure_openai_max_retry_wait_seconds: float = 30.0
    # USD per 1M tokens for your deployment, from the Azure pricing page or portal. Unset = cost not estimated.
    azure_openai_input_price_per_mtok_usd: Optional[float] = None
    azure_openai_output_price_per_mtok_usd: Optional[float] = None

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