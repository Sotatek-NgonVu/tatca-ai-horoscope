"""
MongoDB Atlas Vector Store implementation.

Uses LangChain's MongoDBAtlasVectorSearch for storing and retrieving
vector-embedded document chunks, backed by a shared MongoClient.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.collection import Collection

from app.core.exceptions import VectorStoreError
from app.core.interfaces import VectorStoreRepository
from app.domain.models import Chunk

logger = logging.getLogger(__name__)


class MongoVectorStore(VectorStoreRepository):
    """Vector store backed by MongoDB Atlas Vector Search + sentence-transformers."""

    def __init__(
        self,
        collection: Collection,
        index_name: str,
        embedding_model_name: str,
    ) -> None:
        self._collection = collection
        self._index_name = index_name
        # LangChain wrapper for sentence-transformers — reuses same model instance
        self._langchain_embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model_name
        )

    def _get_vector_search(self) -> MongoDBAtlasVectorSearch:
        """Build a MongoDBAtlasVectorSearch instance for querying."""
        return MongoDBAtlasVectorSearch(
            collection=self._collection,
            embedding=self._langchain_embeddings,
            index_name=self._index_name,
        )

    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        Embed chunks locally and store in MongoDB Atlas.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        try:
            from langchain_core.documents import Document

            langchain_docs = [
                Document(page_content=c.content, metadata=c.metadata)
                for c in chunks
            ]

            MongoDBAtlasVectorSearch.from_documents(
                documents=langchain_docs,
                embedding=self._langchain_embeddings,
                collection=self._collection,
                index_name=self._index_name,
            )

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
