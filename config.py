from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Regulatory Co-pilot API"
    model_path: str = "maude_classifier/model/maude_classifier.joblib"
    host: str = "127.0.0.1"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()