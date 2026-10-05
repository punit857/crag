from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    # LLM Provider configuration (Free-tier only per Rule 20)
    llm_provider: str = "groq"  # "groq" or "gemini"
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    # NOTE: llama-3.1-8b-instant was shut down by Groq on 2026-08-16.
    # openai/gpt-oss-20b is the official recommended replacement.
    groq_model: str = "openai/gpt-oss-20b"
    gemini_model: str = "gemini-3.1-flash-lite"

    # Web Search Fallback configuration
    tavily_api_key: Optional[str] = None
    # Rule 22: Domain-restricted web search fallback — industrial equipment
    # efficiency/maintenance sources (DOE, NIST, and manufacturer/engineering
    # references).
    allowed_web_domains: List[str] = [
        "energy.gov",
        "nrel.gov",
        "nist.gov",
        "osha.gov",
    ]

    # Vector DB configuration (Self-hosted/Local Qdrant per Rule 4)
    qdrant_use_local: bool = False  # Toggle for local file mode vs Phase 6 Docker server
    qdrant_local_path: str = "./local_qdrant_db"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "industrial_equipment_kb"

    # Execution & Retry limits (Rule 15)
    max_llm_calls_per_eval_run: int = 450
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 2.0

        # Redis cache (Phase 6)
    cache_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 86400        # local-only answers: 24h
    cache_ttl_web_seconds: int = 3600     # answers that used live web search: 1h
    cache_version: str = "v1"             # bump when you re-ingest or change prompts/models
    # Rate limiting (Phase 6) - placeholder values, tune after measuring tokens per query
    rate_limit_per_minute: int = 5        # pipeline runs per IP per minute (cache hits are free)
    rate_limit_daily_global: int = 60     # pipeline runs per UTC day across all users
    
    # Fallback-trigger configuration (Phase 3) — defaults are placeholders;
    # update after reviewing results/grader_comparison.md.
    fallback_trigger_method: str = "threshold"
    reranker_threshold_high: float = -4.2437
    reranker_threshold_low: float = -9.8662

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Global singleton for configuration
config = Settings()