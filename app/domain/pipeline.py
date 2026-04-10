"""
RAG Pipeline orchestrator --- the heart of the application.

Coordinates the full Tu Vi chatbot workflow:
  1. Check if user has BirthData; if missing, extract from text.
  2. Generate a Tu Vi chart via TuViEnginePort.
  3. Retrieve short-term memory (last N messages).
  4. Retrieve long-term memory (vector search, filtered by user_id).
  5. Assemble context and call the LLM.
  6. Persist both query and response (with embeddings).

CRITICAL DESIGN CONSTRAINT:
  This module imports ONLY from ``app.core.interfaces`` and
  ``app.domain.models``.  It NEVER imports from ``app.services.*``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import (
    AppError,
    LLMAuthenticationError,
    LLMRateLimitError,
    VectorStoreError,
)
from app.core.interfaces import (
    EmbeddingService,
    LLMService,
    TuViEnginePort,
    VectorStoreRepository,
)
from app.domain.birth_data_collector import BirthDataCollector, _PartialBirthData
from app.domain.models import (
    BirthData,
    ChatMessage,
    Chunk,
    IngestionResult,
    MessageRole,
)
from app.domain.prompts import (
    SYSTEM_PROMPT as _SYSTEM_PROMPT,
    _can_chi,
)

logger = logging.getLogger(__name__)


# =============================================================================
#  RAG Pipeline
# =============================================================================


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline for the Tu Vi chatbot.

    All dependencies are injected through the constructor --- no global state.
    The pipeline depends ONLY on abstract ports, never on concrete adapters.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStoreRepository,
        llm: LLMService,
        embedding: EmbeddingService,
        tuvi_engine: TuViEnginePort,
        loader_registry: Any | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        rag_top_k: int = 5,
        short_term_limit: int = 5,
        max_tokens: int = 4096,
    ) -> None:
        self._store = vector_store
        self._llm = llm
        self._embedding = embedding
        self._tuvi = tuvi_engine
        self._loader_registry = loader_registry
        self._rag_top_k = rag_top_k
        self._short_term_limit = short_term_limit
        self._max_tokens = max_tokens

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Birth data collector owns the multi-turn _pending dict.
        self._collector = BirthDataCollector(
            llm=llm,
            store=vector_store,
            tuvi=tuvi_engine,
            generate_answer_fn=self._generate_answer,
            persist_messages_fn=self._persist_messages,
        )

        # Expose _pending for tests that inspect internal state directly.
        self._pending = self._collector._pending

    # =================================================================
    #  Pipeline 1 --- Document Ingestion (knowledge base)
    # =================================================================

    def ingest(
        self,
        file_path: str,
        original_name: str,
        content_type: str,
    ) -> IngestionResult:
        """Full ingestion pipeline: load -> split -> enrich -> store."""
        logger.info("Starting ingestion for '%s'", original_name)

        if self._loader_registry is None:
            raise RuntimeError("No loader_registry configured for ingestion")

        # 1. Load raw chunks from the document
        raw_chunks: list[Chunk] = self._loader_registry.load(file_path, content_type)
        logger.info("Loaded %d raw chunk(s) from '%s'", len(raw_chunks), original_name)

        # 2. Split into smaller pieces
        all_chunks: list[Chunk] = []
        for chunk in raw_chunks:
            splits = self._splitter.split_text(chunk.content)
            for split_text in splits:
                enriched_metadata = {
                    **chunk.metadata,
                    "source": original_name,
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                }
                all_chunks.append(Chunk(content=split_text, metadata=enriched_metadata))

        logger.info("Split into %d chunk(s)", len(all_chunks))

        # 3. Store in vector database
        stored = self._store.add_documents(all_chunks)

        result = IngestionResult(
            chunks_stored=stored,
            original_name=original_name,
            upload_date=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Ingestion complete: %s", result)
        return result

    # =================================================================
    #  Pipeline 2 --- Chat (the main RAG pipeline)
    # =================================================================

    def chat(self, user_id: str, query: str) -> str:
        """
        Full RAG pipeline for a single user query.

        Flow:
            1. If user has no birth data, collect it (multi-turn).
            2. Generate Tu Vi chart.
            3. Fetch short-term memory.
            4. Fetch long-term memory (vector search).
            5. Assemble context and call the LLM.
            6. Persist both query and response.

        Never raises to the caller --- returns a user-friendly error
        message on failure.
        """
        try:
            # -- Step 1: Check birth data --
            birth_data = self._collector.get_birth_data(user_id)

            if birth_data is None:
                return self._collector.handle(user_id, query)

            # -- Step 2: Generate Tu Vi chart --
            chart_json = self._generate_chart(birth_data)

            # -- Step 3 + 4 (parallel): short-term memory and embed query --
            # Embedding the query and fetching the chat log are independent;
            # run them concurrently to cut the serial wait time roughly in half.
            with ThreadPoolExecutor(max_workers=2) as pool:
                short_term_future = pool.submit(
                    self._fetch_short_term_memory, user_id
                )
                embedding_future = pool.submit(
                    self._embedding.embed_text, query
                )
                short_term = short_term_future.result()
                query_embedding = embedding_future.result()

            # -- Step 4b: long-term memory using the embedding already computed --
            long_term = self._fetch_long_term_memory_with_embedding(
                user_id, query, query_embedding
            )

            # -- Step 5: Assemble context and call LLM --
            answer = self._generate_answer(
                query=query,
                chart_json=chart_json,
                short_term=short_term,
                long_term=long_term,
            )

            if not answer:
                answer = (
                    "Xin loi, toi khong the tra loi luc nay. "
                    "Vui long thu lai sau."
                )

            # -- Step 6: Persist messages (reuse query embedding already computed) --
            self._persist_messages(user_id, query, answer, query_embedding=query_embedding)

            return answer

        except LLMAuthenticationError:
            logger.error("Anthropic API key is invalid")
            return "Loi xac thuc API. Vui long lien he quan tri vien."

        except LLMRateLimitError:
            logger.warning("Anthropic rate limit hit")
            return "He thong dang ban. Vui long thu lai sau vai giay."

        except Exception:
            logger.exception("Unexpected error in RAGPipeline.chat")
            return (
                "Da xay ra loi khi xu ly yeu cau cua ban. "
                "Vui long thu lai hoac lien he quan tri vien."
            )

    # =================================================================
    #  Birth Data (persistence — delegated to BirthDataCollector)
    # =================================================================

    def _get_birth_data(self, user_id: str) -> BirthData | None:
        """Retrieve the user's birth data from the store."""
        return self._collector.get_birth_data(user_id)

    def save_birth_data(self, user_id: str, birth_data: BirthData) -> None:
        """Persist birth data for a user."""
        self._collector.save_birth_data(user_id, birth_data)

    def _reprompt_missing(self, partial: _PartialBirthData) -> str:
        """Build a friendly re-prompt for missing birth data fields."""
        return self._collector._reprompt(partial)

    # =================================================================
    #  Tu Vi Chart
    # =================================================================

    def _generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """Generate the Tu Vi chart. Returns empty dict on failure."""
        try:
            return self._tuvi.generate_chart(birth_data)
        except AppError:
            logger.exception("Tu Vi chart generation failed")
            return {}

    # =================================================================
    #  Short-Term Memory
    # =================================================================

    def _fetch_short_term_memory(self, user_id: str) -> list[ChatMessage]:
        """Last N messages for the user. Best-effort: returns [] on failure."""
        try:
            return self._store.get_recent_messages(
                user_id, limit=self._short_term_limit
            )
        except VectorStoreError:
            logger.exception("Short-term memory retrieval failed for user %s", user_id)
            return []

    # =================================================================
    #  Long-Term Memory (Vector Search)
    # =================================================================

    def _fetch_long_term_memory(
        self, user_id: str, query: str
    ) -> list[ChatMessage]:
        """Embed the query and vector search, filtered by user_id."""
        if not query.strip():
            return []

        try:
            query_embedding = self._embedding.embed_text(query)
            return self._fetch_long_term_memory_with_embedding(
                user_id, query, query_embedding
            )
        except Exception:
            logger.exception("Long-term memory retrieval failed for user %s", user_id)
            return []

    def _fetch_long_term_memory_with_embedding(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float],
    ) -> list[ChatMessage]:
        """Vector search using a pre-computed query embedding."""
        if not query.strip():
            return []

        try:
            return self._store.vector_search_messages(
                query_embedding,
                user_id,
                top_k=self._rag_top_k,
            )
        except Exception:
            logger.exception("Long-term memory retrieval failed for user %s", user_id)
            return []

    # =================================================================
    #  LLM Generation
    # =================================================================

    def _generate_answer(
        self,
        *,
        query: str,
        chart_json: dict[str, Any],
        short_term: list[ChatMessage],
        long_term: list[ChatMessage],
    ) -> str:
        """Assemble the full prompt context and call the LLM with prompt caching."""
        current_year = datetime.now(tz=timezone.utc).year

        system_prompt = _SYSTEM_PROMPT.format(
            current_year=current_year,
            can_chi=_can_chi(current_year),
        )

        # Build the conversation context block (not cached — changes every turn).
        context_sections: list[str] = []

        if short_term:
            convo_lines: list[str] = []
            for msg in short_term:
                role_label = "Nguoi dung" if msg.role == MessageRole.USER else "Tu Vi AI"
                convo_lines.append(f"**{role_label}**: {msg.content}")
            context_sections.append(
                "## Lich su tro chuyen gan day\n\n"
                + "\n\n".join(convo_lines)
            )

        if long_term:
            lt_lines: list[str] = []
            for i, msg in enumerate(long_term, 1):
                role_label = "Nguoi dung" if msg.role == MessageRole.USER else "Tu Vi AI"
                lt_lines.append(f"[{i}] **{role_label}**: {msg.content}")
            context_sections.append(
                "## Ky uc dai han lien quan\n\n"
                + "\n\n".join(lt_lines)
            )

        conversation_context = "\n\n---\n\n".join(context_sections)

        return self._llm.generate_with_cache(
            system_prompt=system_prompt,
            chart_json=chart_json,
            conversation_context=conversation_context,
            query=query,
            max_tokens=self._max_tokens,
        )

    # =================================================================
    #  Persist Messages (with Embeddings)
    # =================================================================

    def _persist_messages(
        self,
        user_id: str,
        query: str,
        answer: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> None:
        """
        Embed and save both the user query and the assistant response.

        If ``query_embedding`` is provided (pre-computed during the vector
        search step), it is reused to avoid a redundant embedding call.

        Best-effort: failures are logged but never propagated.
        """
        try:
            if query_embedding is not None:
                # Reuse precomputed embedding; only embed the answer.
                answer_embeddings = self._embedding.embed_documents([answer])
                answer_embedding: list[float] | None = (
                    answer_embeddings[0] if answer_embeddings else None
                )
            else:
                embeddings = self._embedding.embed_documents([query, answer])
                query_embedding = embeddings[0] if len(embeddings) > 0 else None
                answer_embedding = embeddings[1] if len(embeddings) > 1 else None

            now = datetime.now(timezone.utc)

            user_msg = ChatMessage(
                user_id=user_id,
                role=MessageRole.USER,
                content=query,
                embedding=query_embedding,
                created_at=now,
            )
            self._store.save_chat_message(user_msg)

            assistant_msg = ChatMessage(
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                embedding=answer_embedding,
                created_at=now,
            )
            self._store.save_chat_message(assistant_msg)

            logger.info(
                "Persisted query + response for user %s (%d + %d chars)",
                user_id,
                len(query),
                len(answer),
            )

        except Exception:
            logger.exception(
                "Failed to persist messages for user %s (non-fatal)", user_id
            )
