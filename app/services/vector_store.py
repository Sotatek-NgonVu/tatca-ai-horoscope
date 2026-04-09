"""
MongoDB Atlas Vector Store implementation.

Uses LangChain's MongoDBAtlasVectorSearch for storing and retrieving
vector-embedded document chunks, backed by a shared MongoClient.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.collection import Collection

from app.core.exceptions import VectorStoreError
from app.core.interfaces import VectorStoreRepository
from app.domain.models import Chunk

logger = logging.getLogger(__name__)


class MongoVectorStore(VectorStoreRepository):
    """Vector store backed by MongoDB Atlas Vector Search + Google Gemini embeddings."""

    def __init__(
        self,
        collection: Collection,
        index_name: str,
        embedding_model_name: str,
        google_api_key: str | None = None,
    ) -> None:
        self._collection = collection
        self._index_name = index_name
        # LangChain wrapper for Google Gemini embeddings — free API, no local model
        # output_dimensionality=768 uses Matryoshka truncation so vectors match the
        # MongoDB Atlas index (created with numDimensions: 768).
        kwargs: dict[str, Any] = {
            "model": embedding_model_name,
            "output_dimensionality": 768,
        }
        if google_api_key:
            kwargs["google_api_key"] = google_api_key
        self._langchain_embeddings = GoogleGenerativeAIEmbeddings(**kwargs)

    def _get_vector_search(self) -> MongoDBAtlasVectorSearch:
        """Build a MongoDBAtlasVectorSearch instance for querying."""
        return MongoDBAtlasVectorSearch(
            collection=self._collection,
            embedding=self._langchain_embeddings,
            index_name=self._index_name,
        )

    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        Embed chunks via Gemini API and store in MongoDB Atlas.

        Chunks are embedded one at a time with a 1-second delay between each
        call to stay within the Gemini free-tier rate limit (1 req/s).

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        try:
            from langchain_core.documents import Document

            for i, chunk in enumerate(chunks):
                doc = Document(page_content=chunk.content, metadata=chunk.metadata)
                MongoDBAtlasVectorSearch.from_documents(
                    documents=[doc],
                    embedding=self._langchain_embeddings,
                    collection=self._collection,
                    index_name=self._index_name,
                )
                logger.debug("Embedded and stored chunk %d/%d", i + 1, len(chunks))
                # Rate-limit: 1 request per second to Gemini embedding API.
                # Skip the sleep after the last chunk.
                if i < len(chunks) - 1:
                    time.sleep(1)

            logger.info("Stored %d chunk(s) in vector store", len(chunks))
            return len(chunks)

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store {len(chunks)} chunk(s)", cause=exc
            ) from exc

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[str]:
        """
        Perform similarity search against stored documents.

        Returns:
            List of page_content strings from the most relevant chunks.
        """
        if not query:
            logger.warning("Empty query for similarity search, returning []")
            return []

        try:
            vector_search = self._get_vector_search()
            retriever = vector_search.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            docs = retriever.invoke(query)
            results = [doc.page_content for doc in docs]
            logger.info(
                "Similarity search returned %d chunk(s) (top_k=%d)",
                len(results),
                top_k,
            )
            return results

        except Exception as exc:
            raise VectorStoreError(
                f"Similarity search failed: {exc}", cause=exc
            ) from exc
