"""
Local sentence-transformers embedding service.

Uses ``paraphrase-multilingual-mpnet-base-v2`` by default, which
natively supports Vietnamese and produces 768-dimensional vectors.

No external API calls --- runs entirely locally after first model
download.
"""

from __future__ import annotations

import logging

from app.core.exceptions import EmbeddingError
from app.core.interfaces import EmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingService(EmbeddingService):
    """
    Embedding service using a local ``sentence-transformers`` model.

    The model is lazy-loaded on first call so that import-time stays fast.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    ) -> None:
        self._model_name = model_name
        self._model = None  # Lazy-loaded

    def _get_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading sentence-transformers model: %s", self._model_name
                )
                self._model = SentenceTransformer(self._model_name)
                logger.info("Model loaded successfully")
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}'",
                    cause=exc,
                ) from exc
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a 768-dim vector."""
        try:
            model = self._get_model()
            embedding = model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to embed text", cause=exc) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into 768-dim vectors."""
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            return [e.tolist() for e in embeddings]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} document(s)", cause=exc
            ) from exc
