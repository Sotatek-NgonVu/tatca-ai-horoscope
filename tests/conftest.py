"""
Shared test fixtures.

Provides mock implementations of all service interfaces, a test FastAPI
client, and common test data.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.interfaces import (
    DocumentLoader,
    EmbeddingService,
    LLMService,
    OCRService,
    VectorStoreRepository,
)
from app.domain.models import AnalysisResult, Chunk, IngestionResult


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock Services
# ═══════════════════════════════════════════════════════════════════════════════


class MockEmbeddingService(EmbeddingService):
    """Returns fixed-length zero vectors for testing."""

    def __init__(self, dimensions: int = 768) -> None:
        self._dims = dimensions

    def embed_text(self, text: str) -> list[float]:
        return [0.0] * self._dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dims for _ in texts]


class MockLLMService(LLMService):
    """Returns a canned response for testing."""

    def __init__(self, response: str = "Mock LLM response") -> None:
        self._response = response
        self.call_count = 0
        self.last_call: dict[str, Any] = {}

    def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        self.call_count += 1
        self.last_call = {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "max_tokens": max_tokens,
        }
        return self._response


class MockOCRService(OCRService):
    """Returns a canned OCR text for testing."""

    def __init__(self, extracted_text: str = "Mock OCR text") -> None:
        self._text = extracted_text
        self.call_count = 0

    def extract_text(self, image_bytes: bytes, *, media_type: str = "image/jpeg") -> str:
        self.call_count += 1
        return self._text


class MockVectorStore(VectorStoreRepository):
    """In-memory vector store for testing."""

    def __init__(self) -> None:
        self.stored_chunks: list[Chunk] = []
        self._search_results: list[str] = []

    def set_search_results(self, results: list[str]) -> None:
        """Configure what similarity_search will return."""
        self._search_results = results

    def add_documents(self, chunks: list[Chunk]) -> int:
        self.stored_chunks.extend(chunks)
        return len(chunks)

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[str]:
        return self._search_results[:top_k]


class MockDocumentLoader(DocumentLoader):
    """Accepts all files and returns canned chunks."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = chunks or [
            Chunk(content="Test document content page 1", metadata={"page": 0}),
            Chunk(content="Test document content page 2", metadata={"page": 1}),
        ]

    def supports(self, content_type: str, file_path: str) -> bool:
        return True

    def load(self, file_path: str) -> list[Chunk]:
        return self._chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    return MockEmbeddingService()


@pytest.fixture
def mock_llm_service() -> MockLLMService:
    return MockLLMService(response="Day la luan giai Tu Vi cua ban.")


@pytest.fixture
def mock_ocr_service() -> MockOCRService:
    return MockOCRService(extracted_text="Cung Menh: Thien Dong - Thien Luong")


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    store = MockVectorStore()
    store.set_search_results([
        "Tu Vi: Thien Dong la sao chu tri tri thuc...",
        "Thien Luong la sao phu tinh, chu y nghia...",
    ])
    return store


@pytest.fixture
def mock_document_loader() -> MockDocumentLoader:
    return MockDocumentLoader()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(content="Tu Vi la mot bo mon tien doan phuong Dong", metadata={"page": 0}),
        Chunk(content="Cung Menh la cung quan trong nhat", metadata={"page": 1}),
        Chunk(content="Thien Dong la sao chi tri tri thuc", metadata={"page": 2}),
    ]
