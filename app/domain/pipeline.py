"""
RAG Pipeline orchestrator.

Central domain service that coordinates the full pipelines:
  1. Document ingestion: load -> split -> enrich -> embed -> store
  2. Image analysis:     OCR  -> retrieve context -> generate answer

Depends only on abstractions (interfaces), never on concrete implementations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    OCRError,
    VectorStoreError,
)
from app.core.interfaces import LLMService, OCRService, VectorStoreRepository
from app.domain.models import AnalysisResult, Chunk, IngestionResult
from app.services.document_loader import DocumentLoaderRegistry

logger = logging.getLogger(__name__)

# ── System prompt for Tử Vi interpretation ───────────────────────────────────
_ASTROLOGER_SYSTEM_PROMPT = """\
Bạn là một chuyên gia Tử Vi Đông Phương uyên thâm với hơn 30 năm kinh nghiệm \
luận giải lá số. Bạn giải thích rõ ràng, chi tiết và dễ hiểu bằng tiếng Việt.

Khi trả lời:
- Dựa trên thông tin Tử Vi được trích xuất từ hình ảnh và kiến thức tham chiếu.
- Phân tích các cung, sao, cục và ý nghĩa của chúng.
- Đưa ra luận giải cụ thể, tránh chung chung.
- Nếu hình ảnh không rõ hoặc thiếu thông tin, hãy nêu rõ phần nào không thể \
  đọc được và vẫn cố gắng giải thích phần đọc được.
- Trả lời hoàn toàn bằng tiếng Việt.\
"""


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline for both document ingestion
    and image analysis workflows.

    All dependencies are injected through the constructor — no global state.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStoreRepository,
        llm: LLMService,
        ocr: OCRService,
        loader_registry: DocumentLoaderRegistry,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        rag_top_k: int = 5,
        max_tokens: int = 4096,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._ocr = ocr
        self._loader_registry = loader_registry
        self._rag_top_k = rag_top_k
        self._max_tokens = max_tokens

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Pipeline 1 — Document Ingestion
    # ═══════════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        file_path: str,
        original_name: str,
        content_type: str,
    ) -> IngestionResult:
        """
        Full ingestion pipeline: load -> split -> enrich -> store.

        Returns:
            IngestionResult with the number of chunks stored.
        """
        logger.info("Starting ingestion for '%s'", original_name)

        # 1. Load document using the appropriate loader
        raw_chunks = self._loader_registry.load(file_path, content_type)
        logger.info("Loaded %d raw section(s)", len(raw_chunks))

        # 2. Split into smaller chunks
        from langchain_core.documents import Document as LCDocument

        lc_docs = [
            LCDocument(page_content=c.content, metadata=c.metadata)
            for c in raw_chunks
        ]
        split_docs = self._splitter.split_documents(lc_docs)

        chunks = [
            Chunk(content=doc.page_content, metadata=doc.metadata)
            for doc in split_docs
        ]
        logger.info("Split into %d chunk(s)", len(chunks))

        # 3. Enrich metadata
        upload_date = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            chunk.metadata["original_name"] = original_name
            chunk.metadata["upload_date"] = upload_date

        # 4. Embed and store
        stored_count = self._vector_store.add_documents(chunks)
        logger.info("Stored %d chunk(s) in vector store", stored_count)

        return IngestionResult(
            chunks_stored=stored_count,
            original_name=original_name,
            upload_date=upload_date,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Pipeline 2 — Image Analysis (OCR -> RAG -> Answer)
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze_image(self, image_bytes: bytes) -> AnalysisResult:
        """
        Full three-stage pipeline: OCR -> RAG Retrieval -> Answer Generation.

        Never raises — returns a user-friendly error message on failure.
        """
        try:
            # Stage 1 — OCR
            raw_text = self._ocr.extract_text(image_bytes)

            # Stage 2 — RAG Retrieval
            context_chunks = self._retrieve_context(raw_text)

            # Stage 3 — Generate answer
            answer = self._generate_answer(raw_text, context_chunks)

            if not answer:
                answer = (
                    "Xin loi, toi khong the tao ra phan tich cho la so nay. "
                    "Vui long thu lai voi hinh anh ro hon."
                )

            return AnalysisResult(
                answer=answer,
                ocr_text=raw_text,
                chunks_used=len(context_chunks),
            )

        except LLMAuthenticationError:
            logger.error("Anthropic API key is invalid")
            return AnalysisResult(
                answer="Loi xac thuc API. Vui long lien he quan tri vien.",
                ocr_text="",
                chunks_used=0,
            )

        except LLMRateLimitError:
            logger.warning("Anthropic rate limit hit")
            return AnalysisResult(
                answer="He thong dang ban. Vui long thu lai sau vai giay.",
                ocr_text="",
                chunks_used=0,
            )

        except Exception:
            logger.exception("Unexpected error in analyze_image")
            return AnalysisResult(
                answer=(
                    "Da xay ra loi khi phan tich hinh anh. "
                    "Vui long thu lai hoac lien he quan tri vien."
                ),
                ocr_text="",
                chunks_used=0,
            )

    def _retrieve_context(self, query_text: str) -> list[str]:
        """Retrieve relevant knowledge chunks from the vector store."""
        if not query_text:
            logger.warning("Empty query text, skipping retrieval")
            return []

        try:
            return self._vector_store.similarity_search(
                query_text, top_k=self._rag_top_k
            )
        except VectorStoreError:
            # Retrieval is best-effort; the bot can still answer from OCR alone.
            logger.exception("RAG retrieval failed, proceeding without context")
            return []

    def _generate_answer(self, raw_text: str, context_chunks: list[str]) -> str:
        """Build the prompt and call the LLM to generate an interpretation."""
        # Build context section
        context_section = ""
        if context_chunks:
            formatted = "\n\n---\n\n".join(
                f"[Tai lieu tham khao {i + 1}]\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            )
            context_section = (
                "## Kien thuc tham khao tu co so du lieu Tu Vi\n\n"
                f"{formatted}\n\n"
            )

        # Build horoscope section
        horoscope_section = (
            "## Thong tin tu la so (trich xuat tu hinh anh)\n\n"
            f"{raw_text if raw_text else '[Khong trich xuat duoc noi dung tu hinh anh]'}"
        )

        user_message = (
            f"{context_section}"
            f"{horoscope_section}\n\n"
            "---\n\n"
            "Dua tren thong tin la so tren va kien thuc tham khao, "
            "hay phan tich va luan giai la so Tu Vi nay mot cach chi tiet."
        )

        return self._llm.generate(
            system_prompt=_ASTROLOGER_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=self._max_tokens,
        )
