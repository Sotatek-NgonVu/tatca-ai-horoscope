"""
MongoDB Atlas Vector Store implementation.

Uses **native pymongo** aggregation pipelines with ``$vectorSearch``
for both the knowledge base (document chunks) and the chat history
(long-term memory).

No LangChain dependency --- all vector operations go through the
Anthropic-compatible ``EmbeddingService`` port and raw pymongo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection
from pymongo.database import Database

from app.core.exceptions import VectorStoreError
from app.core.interfaces import EmbeddingService, VectorStoreRepository
from app.domain.models import ChatMessage, Chunk, MessageRole

logger = logging.getLogger(__name__)


class MongoVectorStore(VectorStoreRepository):
    """
    Vector store backed by MongoDB Atlas Vector Search.

    Manages two collections:
      - ``knowledge_base``: document chunks with embeddings
      - ``chat_histories``: user messages with embeddings (hybrid memory)
      - ``users``: user profile documents
    """

    def __init__(
        self,
        *,
        database: Database,
        embedding_service: EmbeddingService,
        knowledge_collection_name: str = "knowledge_base",
        chat_collection_name: str = "chat_histories",
        users_collection_name: str = "users",
        knowledge_index_name: str = "vector_index",
        chat_index_name: str = "chat_vector_index",
    ) -> None:
        self._db = database
        self._embedding = embedding_service

        self._knowledge_col: Collection = database[knowledge_collection_name]
        self._chat_col: Collection = database[chat_collection_name]
        self._users_col: Collection = database[users_collection_name]

        self._knowledge_index = knowledge_index_name
        self._chat_index = chat_index_name

    # ═════════════════════════════════════════════════════════════════════
    #  Knowledge Base --- Document Chunks
    # ═════════════════════════════════════════════════════════════════════

    def add_documents(self, chunks: list[Chunk]) -> int:
        """
        Embed each chunk and insert into the knowledge_base collection.

        Each document stored has the shape::

            {
                "content":   "...",
                "embedding": [0.12, -0.34, ...],
                "metadata":  { ... }
            }

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        try:
            texts = [c.content for c in chunks]
            embeddings = self._embedding.embed_documents(texts)

            docs: list[dict[str, Any]] = []
            for chunk, emb in zip(chunks, embeddings):
                docs.append(
                    {
                        "content": chunk.content,
                        "embedding": emb,
                        "metadata": chunk.metadata,
                    }
                )

            result = self._knowledge_col.insert_many(docs)
            stored = len(result.inserted_ids)
            logger.info("Stored %d chunk(s) in knowledge_base", stored)
            return stored

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store {len(chunks)} chunk(s)", cause=exc
            ) from exc

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[str]:
        """
        Perform ``$vectorSearch`` on the knowledge_base collection.

        Returns:
            List of ``content`` strings from the top-k matching chunks.
        """
        if not query:
            logger.warning("Empty query for similarity_search, returning []")
            return []

        try:
            query_embedding = self._embedding.embed_text(query)

            pipeline: list[dict[str, Any]] = [
                {
                    "$vectorSearch": {
                        "index": self._knowledge_index,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "_id": 0,
                    }
                },
            ]

            results = list(self._knowledge_col.aggregate(pipeline))
            contents = [doc["content"] for doc in results]

            logger.info(
                "Knowledge similarity_search returned %d chunk(s) (top_k=%d)",
                len(contents),
                top_k,
            )
            return contents

        except Exception as exc:
            raise VectorStoreError(
                f"Knowledge similarity search failed: {exc}", cause=exc
            ) from exc

    # ═════════════════════════════════════════════════════════════════════
    #  Chat History --- Short-Term + Long-Term Memory
    # ═════════════════════════════════════════════════════════════════════

    def save_chat_message(self, message: ChatMessage) -> None:
        """
        Persist a ``ChatMessage`` to the ``chat_histories`` collection.

        The message should already have its ``embedding`` populated by
        the caller (the pipeline embeds before saving).
        """
        try:
            doc: dict[str, Any] = {
                "user_id": message.user_id,
                "role": message.role.value,
                "content": message.content,
                "embedding": message.embedding,
                "metadata": message.metadata,
                "created_at": message.created_at,
            }
            self._chat_col.insert_one(doc)
            logger.debug(
                "Saved %s message for user %s (%d chars)",
                message.role.value,
                message.user_id,
                len(message.content),
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save chat message for user {message.user_id}",
                cause=exc,
            ) from exc

    def get_recent_messages(
        self, user_id: str, *, limit: int = 5
    ) -> list[ChatMessage]:
        """
        Retrieve the ``limit`` most-recent messages for ``user_id``,
        ordered from **oldest to newest** (ascending ``created_at``).

        This is the **short-term memory** window.
        """
        try:
            cursor = (
                self._chat_col.find({"user_id": user_id})
                .sort("created_at", -1)
                .limit(limit)
            )

            messages: list[ChatMessage] = []
            for doc in cursor:
                messages.append(
                    ChatMessage(
                        user_id=doc["user_id"],
                        role=MessageRole(doc["role"]),
                        content=doc["content"],
                        embedding=doc.get("embedding"),
                        metadata=doc.get("metadata", {}),
                        created_at=doc.get("created_at", datetime.now(timezone.utc)),
                    )
                )

            # Reverse so chronological order is oldest -> newest
            messages.reverse()
            return messages

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to fetch recent messages for user {user_id}",
                cause=exc,
            ) from exc

    def vector_search_messages(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        top_k: int = 5,
    ) -> list[ChatMessage]:
        """
        ``$vectorSearch`` over chat_histories with a **pre-filter** on
        ``user_id``.

        CRITICAL: The ``filter`` clause in ``$vectorSearch`` ensures strict
        data isolation --- a user will NEVER see another user's messages.

        Requires a MongoDB Atlas Vector Search index on ``chat_histories``
        with:
          - path: ``embedding``
          - filter field: ``user_id`` (type: ``token``)
        """
        try:
            pipeline: list[dict[str, Any]] = [
                {
                    "$vectorSearch": {
                        "index": self._chat_index,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                        "filter": {
                            "user_id": user_id,
                        },
                    }
                },
                {
                    "$project": {
                        "user_id": 1,
                        "role": 1,
                        "content": 1,
                        "metadata": 1,
                        "created_at": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "_id": 0,
                    }
                },
            ]

            results = list(self._chat_col.aggregate(pipeline))

            messages: list[ChatMessage] = []
            for doc in results:
                messages.append(
                    ChatMessage(
                        user_id=doc["user_id"],
                        role=MessageRole(doc["role"]),
                        content=doc["content"],
                        metadata=doc.get("metadata", {}),
                        created_at=doc.get("created_at", datetime.now(timezone.utc)),
                    )
                )

            logger.info(
                "Chat vector_search returned %d message(s) for user %s (top_k=%d)",
                len(messages),
                user_id,
                top_k,
            )
            return messages

        except Exception as exc:
            raise VectorStoreError(
                f"Chat vector search failed for user {user_id}: {exc}",
                cause=exc,
            ) from exc

    # ═════════════════════════════════════════════════════════════════════
    #  User Persistence
    # ═════════════════════════════════════════════════════════════════════

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Look up a user document by ``user_id``."""
        try:
            doc = self._users_col.find_one({"user_id": user_id})
            return doc if doc else None
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to fetch user {user_id}", cause=exc
            ) from exc

    def upsert_user(self, user_id: str, data: dict[str, Any]) -> None:
        """
        Create or update a user document.

        Uses ``$set`` so only the supplied fields are overwritten.
        ``updated_at`` is always refreshed.
        """
        try:
            data["updated_at"] = datetime.now(timezone.utc)
            self._users_col.update_one(
                {"user_id": user_id},
                {
                    "$set": data,
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.now(timezone.utc),
                    },
                },
                upsert=True,
            )
            logger.debug("Upserted user %s", user_id)

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert user {user_id}", cause=exc
            ) from exc
