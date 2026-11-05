"""
Smart Farming AI Agent
Configuration management using pydantic-settings
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # IBM watsonx.ai
    watsonx_api_key: str = Field(default="", env="WATSONX_API_KEY")
    watsonx_url: str = Field(default="https://us-south.ml.cloud.ibm.com", env="WATSONX_URL")
    watsonx_project_id: str = Field(default="", env="WATSONX_PROJECT_ID")
    granite_model_id: str = Field(default="ibm/granite-13b-chat-v2", env="GRANITE_MODEL_ID")

    # IBM Watson Discovery (optional)
    watson_discovery_api_key: str = Field(default="", env="WATSON_DISCOVERY_API_KEY")
    watson_discovery_url: str = Field(default="", env="WATSON_DISCOVERY_URL")
    watson_discovery_project_id: str = Field(default="", env="WATSON_DISCOVERY_PROJECT_ID")

    # IBM COS (optional)
    cos_api_key: str = Field(default="", env="COS_API_KEY")
    cos_instance_crn: str = Field(default="", env="COS_INSTANCE_CRN")
    cos_endpoint: str = Field(default="", env="COS_ENDPOINT")
    cos_bucket_name: str = Field(default="farming-agent-kb", env="COS_BUCKET_NAME")

    # Weather
    weather_api_provider: str = Field(default="open_meteo", env="WEATHER_API_PROVIDER")
    open_meteo_url: str = Field(default="https://api.open-meteo.com/v1/forecast", env="OPEN_METEO_URL")

    # Market prices
    agmarknet_api_key: str = Field(default="", env="AGMARKNET_API_KEY")
    agmarknet_api_url: str = Field(
        default="https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
        env="AGMARKNET_API_URL",
    )

    # App
    app_name: str = Field(default="Smart Farming AI Agent", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # RAG
    vector_db_path: str = Field(default="./data/vectorstore", env="VECTOR_DB_PATH")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # LLM generation
    llm_max_new_tokens: int = Field(default=800, env="LLM_MAX_NEW_TOKENS")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.95, env="LLM_TOP_P")
    llm_top_k: int = Field(default=50, env="LLM_TOP_K")
    llm_repetition_penalty: float = Field(default=1.1, env="LLM_REPETITION_PENALTY")

    # Language
    default_language: str = Field(default="en", env="DEFAULT_LANGUAGE")
    supported_languages: str = Field(default="en,hi,pa,te,ta,kn,mr,bn,gu,or", env="SUPPORTED_LANGUAGES")
    auto_detect_language: bool = Field(default=True, env="AUTO_DETECT_LANGUAGE")

    # Security
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000", env="CORS_ORIGINS")
    secret_key: str = Field(default="dev-secret-change-in-prod", env="SECRET_KEY")

    @property
    def supported_language_list(self) -> List[str]:
        return [lang.strip() for lang in self.supported_languages.split(",")]

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
