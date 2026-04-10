"""
Centralized application configuration.

All environment variables are loaded and validated here via Pydantic Settings.
No other module should read os.environ directly.

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.MONGODB_URI)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Anthropic (Claude) ───────────────────────────────────────────────
    ANTHROPIC_API_KEY: str

    # ── MongoDB Atlas ────────────────────────────────────────────────────
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "tuvi_ai"
    MONGODB_COLLECTION_NAME: str = "knowledge_base"
    MONGODB_INDEX_NAME: str = "vector_index"

    # ── Telegram Bot ─────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    WEBHOOK_BASE_URL: str = ""

    # ── MongoDB Collection / Index Names ────────────────────────────────
    MONGODB_CHAT_COLLECTION: str = "chat_histories"
    MONGODB_USERS_COLLECTION: str = "users"
    MONGODB_CHAT_INDEX_NAME: str = "chat_vector_index"

    # ── Claude Model ─────────────────────────────────────────────────────
    CLAUDE_MODEL: str = "claude-haiku-4-5"
    CLAUDE_EXTRACTION_MODEL: str = "claude-haiku-4-5"
    CLAUDE_MAX_TOKENS: int = 16000
    CLAUDE_OCR_MAX_TOKENS: int = 2048

    # ── Google Gemini (optional, for future use) ────────────────────────
    GOOGLE_API_KEY: str = ""

    # ── Embedding Model (local sentence-transformers) ────────────────────
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIMENSIONS: int = 768

    # ── RAG Configuration ────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
