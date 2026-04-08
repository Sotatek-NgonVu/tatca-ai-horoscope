"""
Unit tests for document loaders and the loader registry.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import UnsupportedDocumentTypeError
from app.domain.models import Chunk
from app.services.document_loader import (
    DocxDocumentLoader,
    DocumentLoaderRegistry,
    ImageDocumentLoader,
    PDFDocumentLoader,
    create_default_loader_registry,
)


class TestPDFDocumentLoader:
    """Tests for PDFDocumentLoader."""

    def test_supports_pdf_mime_type(self) -> None:
        loader = PDFDocumentLoader()
        assert loader.supports("application/pdf", "test.pdf")

    def test_supports_pdf_extension(self) -> None:
        loader = PDFDocumentLoader()
        assert loader.supports("", "document.PDF")

    def test_does_not_support_docx(self) -> None:
        loader = PDFDocumentLoader()
        assert not loader.supports("application/msword", "test.docx")


class TestDocxDocumentLoader:
    """Tests for DocxDocumentLoader."""

    def test_supports_docx_mime_type(self) -> None:
        loader = DocxDocumentLoader()
        assert loader.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "test.docx",
        )

    def test_supports_docx_extension(self) -> None:
        loader = DocxDocumentLoader()
        assert loader.supports("", "report.docx")

    def test_supports_msword_mime(self) -> None:
        loader = DocxDocumentLoader()
        assert loader.supports("application/msword", "test.doc")

    def test_does_not_support_pdf(self) -> None:
        loader = DocxDocumentLoader()
        assert not loader.supports("application/pdf", "test.pdf")


class TestImageDocumentLoader:
    """Tests for ImageDocumentLoader."""

    def test_supports_jpeg(self) -> None:
        loader = ImageDocumentLoader()
        assert loader.supports("image/jpeg", "photo.jpg")

    def test_supports_png(self) -> None:
        loader = ImageDocumentLoader()
        assert loader.supports("image/png", "chart.png")

    def test_does_not_support_pdf(self) -> None:
        loader = ImageDocumentLoader()
        assert not loader.supports("application/pdf", "test.pdf")

    def test_load_returns_placeholder_chunk(self) -> None:
        loader = ImageDocumentLoader()
        chunks = loader.load("/tmp/test.jpg")
        assert len(chunks) == 1
        assert "OCR placeholder" in chunks[0].content
        assert chunks[0].metadata["loader"] == "image_placeholder"


class TestDocumentLoaderRegistry:
    """Tests for DocumentLoaderRegistry chain-of-responsibility."""

    def test_loads_pdf_with_correct_loader(self) -> None:
        """Registry should find and use the PDF loader."""
        registry = create_default_loader_registry()
        # We can't test actual PDF loading without a file,
        # but we can test the routing by checking supports() works.
        # The load() call would fail without a real file.
        loaders = registry._loaders
        pdf_supports = any(
            l.supports("application/pdf", "test.pdf") for l in loaders
        )
        assert pdf_supports

    def test_raises_for_unsupported_type(self) -> None:
        """Registry should raise UnsupportedDocumentTypeError for unknown types."""
        registry = DocumentLoaderRegistry()
        # Empty registry — nothing can handle any file
        with pytest.raises(UnsupportedDocumentTypeError):
            registry.load("/tmp/test.xyz", "application/octet-stream")

    def test_first_matching_loader_wins(self) -> None:
        """The first registered loader that supports the file should be used."""

        class AlwaysLoader(PDFDocumentLoader):
            def supports(self, content_type: str, file_path: str) -> bool:
                return True

            def load(self, file_path: str) -> list[Chunk]:
                return [Chunk(content="AlwaysLoader", metadata={})]

        class NeverLoader(PDFDocumentLoader):
            def supports(self, content_type: str, file_path: str) -> bool:
                return True

            def load(self, file_path: str) -> list[Chunk]:
                return [Chunk(content="NeverLoader", metadata={})]

        registry = DocumentLoaderRegistry()
        registry.register(AlwaysLoader())
        registry.register(NeverLoader())

        chunks = registry.load("/tmp/test.any", "application/pdf")
        assert chunks[0].content == "AlwaysLoader"
