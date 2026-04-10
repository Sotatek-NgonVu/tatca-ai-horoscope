"""
FastAPI dependency injection wiring --- the Composition Root.

All service instantiation and wiring happens here.
Route handlers declare dependencies via ``Depends(get_rag_pipeline)``, etc.

This is the **only** module that knows about concrete implementations.
Everything else depends only on the abstract ports in ``core/interfaces.py``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.core.interfaces import EmbeddingService, LLMService, TuViEnginePort
from app.domain.pipeline import RAGPipeline
from app.infrastructure.database import DatabaseManager
from app.services.embedding import SentenceTransformerEmbeddingService
from app.services.llm import ClaudeLLMService
from app.services.document_loader import create_default_loader_registry
from app.services.tuvi_engine import MockTuViEngine
from app.services.vector_store import MongoVectorStore

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Database Manager (lifecycle managed by lifespan in main.py)
# ═════════════════════════════════════════════════════════════════════════════

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
    global _db_manager  # noqa: PLW0603
    _db_manager = DatabaseManager(
        uri=settings.MONGODB_URI,
        db_name=settings.MONGODB_DB_NAME,
    )
    _db_manager.connect()
    return _db_manager


def shutdown_database_manager() -> None:
    """Close the global DatabaseManager. Called once at shutdown."""
    global _db_manager  # noqa: PLW0603
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None


# ═════════════════════════════════════════════════════════════════════════════
#  Embedding Service  (local sentence-transformers)
# ═════════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Return a cached SentenceTransformer embedding service.

    Uses ``paraphrase-multilingual-mpnet-base-v2`` (768-dim, Vietnamese
    native support, runs locally --- no API cost).
    """
    settings = get_settings()
    return SentenceTransformerEmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  LLM Service  (Anthropic Claude)
# ═════════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Return a cached Claude LLM service."""
    settings = get_settings()
    return ClaudeLLMService(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.CLAUDE_MODEL,
        extraction_model=settings.CLAUDE_EXTRACTION_MODEL,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Tu Vi Engine  (mock implementation)
# ═════════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def get_tuvi_engine() -> TuViEnginePort:
    """Return a cached mock Tu Vi engine."""
    return MockTuViEngine()


# ═════════════════════════════════════════════════════════════════════════════
#  Vector Store  (MongoDB Atlas with native $vectorSearch)
# ═════════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def get_vector_store() -> MongoVectorStore:
    """
    Return a cached MongoVectorStore wired to the shared DB connection and the
    local embedding service.
    """
    settings = get_settings()
    db_manager = get_database_manager()
    embedding_service = get_embedding_service()

    return MongoVectorStore(
        database=db_manager.get_database(),
        embedding_service=embedding_service,
        knowledge_collection_name=settings.MONGODB_COLLECTION_NAME,
        chat_collection_name=settings.MONGODB_CHAT_COLLECTION,
        users_collection_name=settings.MONGODB_USERS_COLLECTION,
        knowledge_index_name=settings.MONGODB_INDEX_NAME,
        chat_index_name=settings.MONGODB_CHAT_INDEX_NAME,
    )


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """
    Return a fully-wired RAGPipeline singleton.

    This is the primary dependency for Telegram handlers and API routes.
    Cached so the in-memory _pending dict persists across requests.
    """
    settings = get_settings()
    return RAGPipeline(
        vector_store=get_vector_store(),
        llm=get_llm_service(),
        embedding=get_embedding_service(),
        tuvi_engine=get_tuvi_engine(),
        loader_registry=create_default_loader_registry(),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        rag_top_k=settings.RAG_TOP_K,
        short_term_limit=5,
        max_tokens=settings.CLAUDE_MAX_TOKENS,
    )
