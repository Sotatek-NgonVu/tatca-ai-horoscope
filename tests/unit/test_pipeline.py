"""
Unit tests for RAGPipeline.

Tests both the ingestion and image analysis workflows with mocked services.
"""

from __future__ import annotations

import pytest

from app.domain.models import Chunk, IngestionResult
from app.domain.pipeline import RAGPipeline
from app.services.document_loader import DocumentLoaderRegistry
from tests.conftest import MockDocumentLoader, MockLLMService, MockOCRService, MockVectorStore


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def pipeline(
    mock_llm_service: MockLLMService,
    mock_ocr_service: MockOCRService,
    mock_vector_store: MockVectorStore,
    mock_document_loader: MockDocumentLoader,
) -> RAGPipeline:
    """Create a RAGPipeline with all mocked services."""
    registry = DocumentLoaderRegistry()
    registry.register(mock_document_loader)

    return RAGPipeline(
        vector_store=mock_vector_store,
        llm=mock_llm_service,
        ocr=mock_ocr_service,
        loader_registry=registry,
        chunk_size=500,
        chunk_overlap=50,
        rag_top_k=3,
        max_tokens=1000,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Ingestion Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIngestion:
    """Tests for RAGPipeline.ingest()."""

    def test_ingest_returns_result(
        self, pipeline: RAGPipeline, mock_vector_store: MockVectorStore
    ) -> None:
        """Ingest should return an IngestionResult with correct counts."""
        result = pipeline.ingest(
            file_path="/tmp/test.pdf",
            original_name="test.pdf",
            content_type="application/pdf",
        )
        assert isinstance(result, IngestionResult)
        assert result.chunks_stored > 0
        assert result.original_name == "test.pdf"
        assert result.upload_date  # Not empty

    def test_ingest_stores_chunks_in_vector_store(
        self, pipeline: RAGPipeline, mock_vector_store: MockVectorStore
    ) -> None:
        """All chunks should be stored in the vector store."""
        pipeline.ingest(
            file_path="/tmp/test.pdf",
            original_name="test.pdf",
            content_type="application/pdf",
        )
        assert len(mock_vector_store.stored_chunks) > 0

    def test_ingest_enriches_metadata(
        self, pipeline: RAGPipeline, mock_vector_store: MockVectorStore
    ) -> None:
        """Each chunk should have original_name and upload_date in metadata."""
        pipeline.ingest(
            file_path="/tmp/test.pdf",
            original_name="my_document.pdf",
            content_type="application/pdf",
        )
        for chunk in mock_vector_store.stored_chunks:
            assert chunk.metadata["original_name"] == "my_document.pdf"
            assert "upload_date" in chunk.metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestImageAnalysis:
    """Tests for RAGPipeline.analyze_image()."""

    def test_analyze_returns_answer(self, pipeline: RAGPipeline) -> None:
        """analyze_image should return a non-empty AnalysisResult."""
        result = pipeline.analyze_image(b"fake-image-bytes")
        assert result.answer
        assert isinstance(result.answer, str)

    def test_analyze_calls_ocr(
        self, pipeline: RAGPipeline, mock_ocr_service: MockOCRService
    ) -> None:
        """analyze_image should call the OCR service."""
        pipeline.analyze_image(b"fake-image-bytes")
        assert mock_ocr_service.call_count == 1

    def test_analyze_calls_llm(
        self, pipeline: RAGPipeline, mock_llm_service: MockLLMService
    ) -> None:
        """analyze_image should call the LLM service."""
        pipeline.analyze_image(b"fake-image-bytes")
        assert mock_llm_service.call_count == 1

    def test_analyze_includes_ocr_text(
        self, pipeline: RAGPipeline, mock_llm_service: MockLLMService
    ) -> None:
        """The LLM prompt should include the OCR-extracted text."""
        pipeline.analyze_image(b"fake-image-bytes")
        assert "Cung Menh: Thien Dong - Thien Luong" in mock_llm_service.last_call["user_message"]

    def test_analyze_handles_error_gracefully(
        self,
        mock_vector_store: MockVectorStore,
        mock_document_loader: MockDocumentLoader,
    ) -> None:
        """If LLM raises, analyze_image should return a user-friendly error."""

        class FailingLLM(MockLLMService):
            def generate(self, **kwargs):
                raise RuntimeError("API down")

        registry = DocumentLoaderRegistry()
        registry.register(mock_document_loader)

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm=FailingLLM(),
            ocr=MockOCRService(),
            loader_registry=registry,
        )

        result = pipeline.analyze_image(b"fake-image-bytes")
        assert result.answer  # Should still have an error message
        assert "loi" in result.answer.lower() or "error" in result.answer.lower()
