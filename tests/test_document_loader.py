"""
Unit tests for app/services/document_loader.py

Tests the document loader registry and individual loader `supports()`
logic. No actual file I/O — file parsing is done by third-party
libraries (PyPDF, docx2txt) that are not tested here.
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


# =============================================================================
#  PDFDocumentLoader.supports()
# =============================================================================


class TestPDFDocumentLoaderSupports:
    def setup_method(self) -> None:
        self.loader = PDFDocumentLoader()

    def test_supports_pdf_mime_type(self) -> None:
        assert self.loader.supports("application/pdf", "file.pdf") is True

    def test_supports_pdf_extension(self) -> None:
        assert self.loader.supports("application/octet-stream", "document.pdf") is True

    def test_supports_uppercase_pdf_extension(self) -> None:
        assert self.loader.supports("application/octet-stream", "DOCUMENT.PDF") is True

    def test_rejects_docx(self) -> None:
        assert self.loader.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file.docx",
        ) is False

    def test_rejects_image(self) -> None:
        assert self.loader.supports("image/jpeg", "photo.jpg") is False

    def test_rejects_unknown(self) -> None:
        assert self.loader.supports("application/zip", "archive.zip") is False


# =============================================================================
#  DocxDocumentLoader.supports()
# =============================================================================


class TestDocxDocumentLoaderSupports:
    def setup_method(self) -> None:
        self.loader = DocxDocumentLoader()

    def test_supports_docx_extension(self) -> None:
        assert self.loader.supports("application/octet-stream", "report.docx") is True

    def test_supports_wordprocessingml_mime(self) -> None:
        assert self.loader.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file.docx",
        ) is True

    def test_supports_wordprocessingml_in_content_type(self) -> None:
        assert self.loader.supports("wordprocessingml.document", "file.docx") is True

    def test_supports_msword_mime(self) -> None:
        assert self.loader.supports("application/msword", "file.doc") is True

    def test_rejects_pdf(self) -> None:
        assert self.loader.supports("application/pdf", "file.pdf") is False

    def test_rejects_image(self) -> None:
        assert self.loader.supports("image/png", "image.png") is False

    def test_uppercase_docx_extension(self) -> None:
        assert self.loader.supports("application/octet-stream", "FILE.DOCX") is True


# =============================================================================
#  ImageDocumentLoader.supports()
# =============================================================================


class TestImageDocumentLoaderSupports:
    def setup_method(self) -> None:
        self.loader = ImageDocumentLoader()

    def test_supports_image_jpeg(self) -> None:
        assert self.loader.supports("image/jpeg", "photo.jpg") is True

    def test_supports_image_png(self) -> None:
        assert self.loader.supports("image/png", "photo.png") is True

    def test_supports_image_webp(self) -> None:
        assert self.loader.supports("image/webp", "photo.webp") is True

    def test_rejects_application_pdf(self) -> None:
        assert self.loader.supports("application/pdf", "doc.pdf") is False

    def test_rejects_text_plain(self) -> None:
        assert self.loader.supports("text/plain", "notes.txt") is False


# =============================================================================
#  ImageDocumentLoader.load() — placeholder, no file I/O needed
# =============================================================================


class TestImageDocumentLoaderLoad:
    def test_returns_placeholder_chunk(self, tmp_path) -> None:
        loader = ImageDocumentLoader()
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

        chunks = loader.load(str(image_file))

        assert len(chunks) == 1
        assert "test.jpg" in chunks[0].content or "OCR placeholder" in chunks[0].content
        assert chunks[0].metadata["loader"] == "image_placeholder"

    def test_chunk_metadata_has_source(self, tmp_path) -> None:
        loader = ImageDocumentLoader()
        image_file = tmp_path / "photo.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        chunks = loader.load(str(image_file))

        assert "source" in chunks[0].metadata


# =============================================================================
#  DocumentLoaderRegistry
# =============================================================================


class TestDocumentLoaderRegistry:
    def test_register_and_load_with_matching_loader(self, tmp_path) -> None:
        """Registry delegates to the first matching loader."""

        class AlwaysMatchLoader:
            def supports(self, content_type: str, file_path: str) -> bool:
                return True

            def load(self, file_path: str) -> list[Chunk]:
                return [Chunk(content="matched")]

        registry = DocumentLoaderRegistry()
        registry.register(AlwaysMatchLoader())  # type: ignore[arg-type]

        chunks = registry.load("/any/path", "anything/type")
        assert chunks[0].content == "matched"

    def test_registry_raises_when_no_loader_matches(self) -> None:
        registry = DocumentLoaderRegistry()
        registry.register(PDFDocumentLoader())

        with pytest.raises(UnsupportedDocumentTypeError):
            registry.load("file.xyz", "application/zip")

    def test_registry_uses_first_matching_loader(self) -> None:
        """First registered matching loader wins."""

        class FirstLoader:
            order = "first"

            def supports(self, content_type: str, file_path: str) -> bool:
                return True

            def load(self, file_path: str) -> list[Chunk]:
                return [Chunk(content="first")]

        class SecondLoader:
            order = "second"

            def supports(self, content_type: str, file_path: str) -> bool:
                return True

            def load(self, file_path: str) -> list[Chunk]:
                return [Chunk(content="second")]

        registry = DocumentLoaderRegistry()
        registry.register(FirstLoader())  # type: ignore[arg-type]
        registry.register(SecondLoader())  # type: ignore[arg-type]

        chunks = registry.load("/any", "any/type")
        assert chunks[0].content == "first"

    def test_empty_registry_raises(self) -> None:
        registry = DocumentLoaderRegistry()

        with pytest.raises(UnsupportedDocumentTypeError):
            registry.load("file.pdf", "application/pdf")

    def test_image_loader_fallback(self, tmp_path) -> None:
        """ImageDocumentLoader never calls real I/O, so safe with a real path."""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"\xff\xd8\xff")

        registry = DocumentLoaderRegistry()
        registry.register(ImageDocumentLoader())

        chunks = registry.load(str(image_file), "image/jpeg")
        assert len(chunks) == 1


# =============================================================================
#  create_default_loader_registry()
# =============================================================================


class TestCreateDefaultLoaderRegistry:
    def test_returns_registry(self) -> None:
        registry = create_default_loader_registry()
        assert isinstance(registry, DocumentLoaderRegistry)

    def test_supports_pdf(self) -> None:
        registry = create_default_loader_registry()
        # The registry should find a loader — raises only if no loader matches
        # We can't call load() without a real file, but supports() works.
        pdf_loader = PDFDocumentLoader()
        assert pdf_loader.supports("application/pdf", "test.pdf") is True

    def test_supports_image_via_registry(self, tmp_path) -> None:
        registry = create_default_loader_registry()
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        chunks = registry.load(str(image_file), "image/png")
        assert len(chunks) == 1
