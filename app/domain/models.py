"""
Domain models — pure data structures with no external dependencies.

These are the core objects passed between layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text chunk from a parsed document, with associated metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of a successful document ingestion."""

    chunks_stored: int
    original_name: str
    upload_date: str


@dataclass
class AnalysisResult:
    """Result of the full OCR -> RAG -> Answer pipeline."""

    answer: str
    ocr_text: str
    chunks_used: int
