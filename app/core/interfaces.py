"""
Abstract interfaces (ports) for all pluggable services.

Domain and orchestration code depends ONLY on these ABCs.
Concrete implementations live in app/services/.

This is the D in SOLID — Dependency Inversion Principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models import Chunk


class EmbeddingService(ABC):
    """Converts text into dense vector representations."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vectors."""


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


class OCRService(ABC):
    """Extracts text from images."""

    @abstractmethod
    def extract_text(self, image_bytes: bytes, *, media_type: str = "image/jpeg") -> str:
        """Extract all visible text from an image."""


class VectorStoreRepository(ABC):
    """Stores and retrieves vector-embedded document chunks."""

    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        Embed and store chunks in the vector database.

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


class DocumentLoader(ABC):
    """Loads and parses a document file into raw text chunks."""

    @abstractmethod
    def supports(self, content_type: str, file_path: str) -> bool:
        """Return True if this loader can handle the given file."""

    @abstractmethod
    def load(self, file_path: str) -> list[Chunk]:
        """Parse the file and return a list of Chunk objects."""
