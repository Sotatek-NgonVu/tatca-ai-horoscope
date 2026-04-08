"""
Unit tests for the mock vector store (validates the test infrastructure).

Integration tests with real MongoDB are in tests/integration/.
"""

from __future__ import annotations

from app.domain.models import Chunk
from tests.conftest import MockVectorStore


class TestMockVectorStore:
    """Validate MockVectorStore behavior used in other tests."""

    def test_add_and_count(self) -> None:
        store = MockVectorStore()
        chunks = [
            Chunk(content="chunk 1", metadata={}),
            Chunk(content="chunk 2", metadata={}),
        ]
        count = store.add_documents(chunks)
        assert count == 2
        assert len(store.stored_chunks) == 2

    def test_add_empty_list(self) -> None:
        store = MockVectorStore()
        count = store.add_documents([])
        assert count == 0

    def test_similarity_search_returns_configured_results(self) -> None:
        store = MockVectorStore()
        store.set_search_results(["result 1", "result 2", "result 3"])
        results = store.similarity_search("query", top_k=2)
        assert results == ["result 1", "result 2"]

    def test_similarity_search_empty_by_default(self) -> None:
        store = MockVectorStore()
        results = store.similarity_search("query")
        assert results == []
