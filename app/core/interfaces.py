"""
Abstract interfaces (ports) for all pluggable services.

Domain and orchestration code depends ONLY on these ABCs.
Concrete implementations live in app/services/.

This is the D in SOLID --- Dependency Inversion Principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.domain.models import BirthData, ChatMessage, Chunk


# ═════════════════════════════════════════════════════════════════════════════
#  Embedding
# ═════════════════════════════════════════════════════════════════════════════


class EmbeddingService(ABC):
    """Converts text into dense vector representations."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vectors."""


# ═════════════════════════════════════════════════════════════════════════════
#  LLM
# ═════════════════════════════════════════════════════════════════════════════


class LLMService(ABC):
    """Generates text from a language model."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response given a system prompt and user message."""

    @abstractmethod
    def generate_with_cache(
        self,
        *,
        system_prompt: str,
        chart_json: dict[str, Any],
        conversation_context: str,
        query: str,
        max_tokens: int = 16000,
    ) -> str:
        """
        Generate a response using Anthropic prompt caching.

        Caches the system prompt and chart JSON as ``ephemeral`` content
        blocks so they are reused across turns without re-processing costs.

        Args:
            system_prompt: The full Tu Vi expert system prompt.
            chart_json: The user's pre-serialised Tu Vi chart dict.
            conversation_context: Short-term + long-term memory as a string.
            query: The user's current question.
            max_tokens: Maximum output tokens.

        Returns:
            The model's generated response string.
        """

    @abstractmethod
    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> dict[str, Any] | None:
        """
        Extract structured data from user text as a JSON dict.

        Returns ``None`` if the text does not contain extractable
        information matching the system prompt's schema.

        Implementations should use deterministic settings (temperature=0)
        and no reasoning/thinking blocks for clean JSON output.
        """


# ═════════════════════════════════════════════════════════════════════════════
#  OCR
# ═════════════════════════════════════════════════════════════════════════════


class OCRService(ABC):
    """Extracts text from images."""

    @abstractmethod
    def extract_text(self, image_bytes: bytes, *, media_type: str = "image/jpeg") -> str:
        """Extract all visible text from an image."""


# ═════════════════════════════════════════════════════════════════════════════
#  Tu Vi Engine
# ═════════════════════════════════════════════════════════════════════════════


class TuViEnginePort(ABC):
    """
    Port for computing a Tu Vi (Vietnamese astrology) chart.

    Given birth data, produces a JSON-serializable dict representing
    the full chart (cung, sao, han, etc.).
    """

    @abstractmethod
    def generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """
        Generate a Tu Vi chart from the given birth data.

        Returns:
            A JSON-serializable dictionary containing the full chart.
        """


# ═════════════════════════════════════════════════════════════════════════════
#  Vector Store  (knowledge base + chat history)
# ═════════════════════════════════════════════════════════════════════════════


class VectorStoreRepository(ABC):
    """
    Stores and retrieves vector-embedded document chunks.

    Also manages chat history with vector search capabilities.
    """

    # ── Knowledge base (document chunks) ─────────────────────────────────

    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        Embed and store document chunks in the vector database.

        Returns:
            Number of chunks successfully stored.
        """

    @abstractmethod
    def similarity_search(self, query: str, *, top_k: int = 5) -> list[str]:
        """
        Find the most similar document chunks for a query.

        Returns:
            List of page_content strings from the top-k matching chunks.
        """

    # ── Chat history (short-term + long-term memory) ─────────────────────

    @abstractmethod
    def save_chat_message(self, message: ChatMessage) -> None:
        """
        Persist a chat message (with its embedding) to the chat_histories
        collection.
        """

    @abstractmethod
    def get_recent_messages(
        self, user_id: str, *, limit: int = 5
    ) -> list[ChatMessage]:
        """
        Retrieve the most recent messages for a user, ordered newest-first.

        This provides the **short-term memory** window.
        """

    @abstractmethod
    def vector_search_messages(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        top_k: int = 5,
    ) -> list[ChatMessage]:
        """
        Perform vector similarity search over chat history, filtered
        strictly by ``user_id``.

        This provides the **long-term memory** retrieval.

        CRITICAL: The ``$vectorSearch`` pipeline MUST use a metadata
        pre-filter on ``user_id`` so that one user never sees another
        user's history.
        """

    # ── User persistence ─────────────────────────────────────────────────

    @abstractmethod
    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Retrieve a user document by user_id. Returns None if not found."""

    @abstractmethod
    def upsert_user(self, user_id: str, data: dict[str, Any]) -> None:
        """Create or update a user document."""


# ═════════════════════════════════════════════════════════════════════════════
#  Document Loader (preserved from v2)
# ═════════════════════════════════════════════════════════════════════════════


class DocumentLoader(ABC):
    """Loads and parses a document file into raw text chunks."""

    @abstractmethod
    def supports(self, content_type: str, file_path: str) -> bool:
        """Return True if this loader can handle the given file."""

    @abstractmethod
    def load(self, file_path: str) -> list[Chunk]:
        """Parse the file and return a list of Chunk objects."""
