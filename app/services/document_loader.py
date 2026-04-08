"""
Document loaders and registry.

Each loader implements the DocumentLoader interface and handles a specific
file type (PDF, DOCX, image). The DocumentLoaderRegistry provides a
chain-of-responsibility pattern — it tries each registered loader until
one can handle the file.

This follows the Open/Closed Principle: add new file types by creating a
new loader class and registering it, without modifying existing code.
"""

from __future__ import annotations

import logging
import os

from app.core.exceptions import DocumentLoadError, UnsupportedDocumentTypeError
from app.core.interfaces import DocumentLoader
from app.domain.models import Chunk

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Concrete Loaders
# ═══════════════════════════════════════════════════════════════════════════════


class PDFDocumentLoader(DocumentLoader):
    """Loads PDF files using PyPDF."""

    def supports(self, content_type: str, file_path: str) -> bool:
        return content_type == "application/pdf" or file_path.lower().endswith(".pdf")

    def load(self, file_path: str) -> list[Chunk]:
        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(file_path).load()
            return [
                Chunk(content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ]
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to load PDF: {file_path}", cause=exc
            ) from exc


class DocxDocumentLoader(DocumentLoader):
    """Loads DOCX files using docx2txt."""

    _DOCX_MIME_TYPES = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }

    def supports(self, content_type: str, file_path: str) -> bool:
        return (
            file_path.lower().endswith(".docx")
            or "wordprocessingml" in content_type
            or content_type in self._DOCX_MIME_TYPES
        )

    def load(self, file_path: str) -> list[Chunk]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            docs = Docx2txtLoader(file_path).load()
            return [
                Chunk(content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ]
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to load DOCX: {file_path}", cause=exc
            ) from exc


class ImageDocumentLoader(DocumentLoader):
    """
    Placeholder loader for image files ingested via the API endpoint.

    Real OCR for Telegram bot images is handled by ClaudeOCRService.
    This is only reached when someone uploads an image directly to
    POST /api/ingest.
    """

    def supports(self, content_type: str, file_path: str) -> bool:
        return content_type.startswith("image/")

    def load(self, file_path: str) -> list[Chunk]:
        logger.info("Placeholder OCR for image: %s", file_path)
        return [
            Chunk(
                content=(
                    "[OCR placeholder] — image ingested via API endpoint.\n"
                    f"File: {os.path.basename(file_path)}\n"
                    "For real OCR, send the image via the Telegram bot instead."
                ),
                metadata={"source": file_path, "loader": "image_placeholder"},
            )
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry — Chain of Responsibility
# ═══════════════════════════════════════════════════════════════════════════════


class DocumentLoaderRegistry:
    """
    Tries registered loaders in order until one can handle the file.

    Usage:
        registry = DocumentLoaderRegistry()
        registry.register(PDFDocumentLoader())
        registry.register(DocxDocumentLoader())
        registry.register(ImageDocumentLoader())

        chunks = registry.load(file_path, content_type)
    """

    def __init__(self) -> None:
        self._loaders: list[DocumentLoader] = []

    def register(self, loader: DocumentLoader) -> None:
        """Register a document loader."""
        self._loaders.append(loader)

    def load(self, file_path: str, content_type: str) -> list[Chunk]:
        """
        Find the first loader that supports the file and use it.

        Raises:
            UnsupportedDocumentTypeError: If no loader can handle the file.
        """
        for loader in self._loaders:
            if loader.supports(content_type, file_path):
                logger.info(
                    "Loading document: path=%s type=%s loader=%s",
                    file_path,
                    content_type,
                    type(loader).__name__,
                )
                return loader.load(file_path)

        raise UnsupportedDocumentTypeError(
            f"No loader found for content type '{content_type}'. "
            "Supported: PDF, DOCX, image/*."
        )


def create_default_loader_registry() -> DocumentLoaderRegistry:
    """Create a registry with all built-in document loaders."""
    registry = DocumentLoaderRegistry()
    registry.register(PDFDocumentLoader())
    registry.register(DocxDocumentLoader())
    registry.register(ImageDocumentLoader())
    return registry
