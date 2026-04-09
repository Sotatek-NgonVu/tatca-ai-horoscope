"""
FastAPI dependency injection wiring.

All service instantiation and wiring happens here.
Route handlers declare dependencies via `Depends(get_rag_pipeline)`, etc.

This is the composition root — the only place that knows about
concrete implementations.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.domain.pipeline import RAGPipeline
from app.infrastructure.database import DatabaseManager
from app.services.document_loader import create_default_loader_registry
from app.services.embedding import GeminiEmbeddingService
from app.services.llm import ClaudeLLMService
from app.services.ocr import ClaudeOCRService
from app.services.vector_store import MongoVectorStore

logger = logging.getLogger(__name__)

# ── Module-level singletons (created once, reused) ──────────────────────────
# These are cached by @lru_cache so they're only instantiated on first call.

_db_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Return the singleton DatabaseManager (must be initialized at startup)."""
    if _db_manager is None:
        raise RuntimeError(
            "DatabaseManager not initialized. "
            "Call init_database_manager() during application startup."
        )
    return _db_manager


def init_database_manager(settings: Settings) -> DatabaseManager:
    """Create and connect the global DatabaseManager. Called once at startup."""
    global _db_manager
    _db_manager = DatabaseManager(
        uri=settings.MONGODB_URI,
        db_name=settings.MONGODB_DB_NAME,
    )
    _db_manager.connect()
    return _db_manager


def shutdown_database_manager() -> None:
    """Close the global DatabaseManager. Called once at shutdown."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None


@lru_cache(maxsize=1)
def get_embedding_service() -> GeminiEmbeddingService:
    """Return a cached Gemini embedding service."""
    settings = get_settings()
    return GeminiEmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=settings.EMBEDDING_DIMENSIONS,
    )


@lru_cache(maxsize=1)
def get_llm_service() -> ClaudeLLMService:
    """Return a cached Claude LLM service."""
    settings = get_settings()
    return ClaudeLLMService(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.CLAUDE_MODEL,
    )


@lru_cache(maxsize=1)
def get_ocr_service() -> ClaudeOCRService:
    """Return a cached Claude OCR service."""
    settings = get_settings()
    return ClaudeOCRService(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.CLAUDE_MODEL,
        max_tokens=settings.CLAUDE_OCR_MAX_TOKENS,
    )


def get_vector_store() -> MongoVectorStore:
    """Return a MongoVectorStore using the shared DB connection."""
    settings = get_settings()
    db_manager = get_database_manager()
    collection = db_manager.get_collection(settings.MONGODB_COLLECTION_NAME)
    return MongoVectorStore(
        collection=collection,
        index_name=settings.MONGODB_INDEX_NAME,
        embedding_model_name=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


def get_rag_pipeline() -> RAGPipeline:
    """Return a fully-wired RAGPipeline instance."""
    settings = get_settings()
    return RAGPipeline(
        vector_store=get_vector_store(),
        llm=get_llm_service(),
        ocr=get_ocr_service(),
        loader_registry=create_default_loader_registry(),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        rag_top_k=settings.RAG_TOP_K,
        max_tokens=settings.CLAUDE_MAX_TOKENS,
    )
