from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GOOGLE_SERVICE_ACCOUNT_JSON_PATH: str = "key.json"
    # Fallback credentials path (standard GCP env var) if the above is unset.
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    GOOGLE_CLOUD_PROJECT: str = "dtd-general"
    GEMINI_LOCATION: str = "global"
    PROJECT_NAME: str = "hr-feedback-generation"
    # Vertex AI cost-tracking label stamped on every LLM call (services/llm.py).
    PROJECT_ENVIRONMENT: str = "dev"

    GEMINI_MODEL: str = "gemini-3-flash-preview"
    MAX_TOKENS: int = 16000
    THINKING_LEVEL: str = "minimal"
    LLM_TIMEOUT_SECONDS: int = 60
    
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_CONCURRENCY: int = 8
    EMBEDDING_DIMENSIONS: int = 3072
    SIMILARITY_SEARCH_TOP_K: int = 10

    # Retrieval/ranking knobs.
    # Per-section retrieval bound (must be >> the K shown). Chroma has no score-threshold
    # query, so we retrieve top-N then gate in code.
    TOP_N_PER_SECTION: int = 50
    # Minimum blended semantic score (0-100) to be shown. Below it = "bad match" -> dropped;
    # if no candidate clears it, the query cleanly returns zero results. Provisional 60 —
    # VERIFY against the score-spread log on real vs. nonsense queries (embedding cosine has
    # a high baseline for unrelated text, so 60 may be too high or too low). 0 disables.
    RELEVANCE_THRESHOLD: float = 60.0
    # Multiplicative ranking penalty per soft-failed filter (lower = push failures down harder).
    SOFT_PASS_PENALTY: float = 1
    # Log per-section score spread on each search (to help calibrate RELEVANCE_THRESHOLD).
    LOG_SCORE_SPREAD: bool = True

    # Storage / data locations.
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "candidates"
    DATA_DIR: str = "data/raw"

    # Streamlit -> FastAPI base URL (overridden to http://api:8000 in docker-compose).
    API_BASE_URL: str = "http://localhost:8000"

    # Langfuse observability (tracing). Disabled automatically if keys are absent.
    TRACING_ENABLED: bool = True
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings