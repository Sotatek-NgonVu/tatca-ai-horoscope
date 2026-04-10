"""
Domain-specific exception hierarchy.

Every layer raises from this hierarchy instead of bare Exception.
FastAPI exception handlers in main.py translate these to HTTP responses.
"""

from __future__ import annotations


class AppError(Exception):
    """Base exception for the entire application."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        self.message = message
        self.cause = cause
        super().__init__(message)


# ── Document Loading ─────────────────────────────────────────────────────────

class DocumentLoadError(AppError):
    """Raised when a document cannot be parsed or loaded."""


class UnsupportedDocumentTypeError(DocumentLoadError):
    """Raised when the document MIME type is not supported."""


# ── Embedding ────────────────────────────────────────────────────────────────

class EmbeddingError(AppError):
    """Raised when text embedding fails."""


# ── Vector Store ─────────────────────────────────────────────────────────────

class VectorStoreError(AppError):
    """Raised when vector store operations (insert / search) fail."""


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMError(AppError):
    """Raised when an LLM API call fails."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM API authentication fails."""


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""


# ── OCR ──────────────────────────────────────────────────────────────────────

class OCRError(AppError):
    """Raised when OCR text extraction fails."""


# ── Tu Vi Engine ─────────────────────────────────────────────────────────────

class TuViEngineError(AppError):
    """Raised when Tu Vi chart generation fails."""


# ── Birth Data ───────────────────────────────────────────────────────────────

class BirthDataMissingError(AppError):
    """Raised when a user's birth data is not yet collected."""
