"""Pydantic Environment Settings Module for MedAgentX."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Settings Schema."""

    APP_NAME: str = "MedAgentX"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM Settings
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = "mock-api-key-for-testing"

    # Vector DB Settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "medagentx"

    # Graph DB Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medagentx_password"

    # Relational DB Settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "medagentx"
    POSTGRES_USER: str = "medagentx"
    POSTGRES_PASSWORD: str = "medagentx_password"

    # Cache Settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Retrieval & ACMR Parameters
    VECTOR_TOP_K: int = 5
    MAX_RETRIEVAL_ITERATIONS: int = 2

    # Confidence Thresholds
    CONFIDENCE_HIGH_THRESHOLD: float = 0.80
    CONFIDENCE_LOW_THRESHOLD: float = 0.50

    # Reproducibility Parameters
    RANDOM_SEED: int = 42
    TEMPERATURE: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
