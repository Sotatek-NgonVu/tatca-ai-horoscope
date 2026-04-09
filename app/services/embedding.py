"""
Google Gemini embedding service.

Wraps Google's Generative AI Embedding API to implement EmbeddingService.
Uses the text-embedding-004 model by default.

Model: models/text-embedding-004
  - 768-dimensional vectors (same as paraphrase-multilingual-mpnet-base-v2)
  - Supports Vietnamese and 100+ languages
  - Free tier: 1,500 requests/minute
  - No local model download — runs via API
"""

from __future__ import annotations

import logging

from app.core.exceptions import EmbeddingError
from app.core.interfaces import EmbeddingService

logger = logging.getLogger(__name__)


class GeminiEmbeddingService(EmbeddingService):
    """Embedding service using Google Gemini text-embedding API."""

    def __init__(
        self,
        model_name: str,
        google_api_key: str | None = None,
        output_dimensionality: int = 768,
    ) -> None:
        self._model_name = model_name
        self._client = None  # Lazy-loaded
        self._google_api_key = google_api_key
        self._output_dimensionality = output_dimensionality

    def _get_client(self):
        """Lazy-load the Google GenAI client on first use."""
        if self._client is None:
            try:
                from google import genai

                kwargs = {}
                if self._google_api_key:
                    kwargs["api_key"] = self._google_api_key
                self._client = genai.Client(**kwargs)
                logger.info("Google Gemini embedding client initialized: %s", self._model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to initialize Gemini embedding client for '{self._model_name}'",
                    cause=exc,
                ) from exc
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a 768-dim vector."""
        try:
            client = self._get_client()
            from google.genai import types
            result = client.models.embed_content(
                model=self._model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._output_dimensionality
                ),
            )
            return list(result.embeddings[0].values)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Failed to embed text", cause=exc
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into 768-dim vectors."""
        if not texts:
            return []
        try:
            client = self._get_client()
            from google.genai import types
            result = client.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._output_dimensionality
                ),
            )
            return [list(e.values) for e in result.embeddings]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} document(s)", cause=exc
            ) from exc
